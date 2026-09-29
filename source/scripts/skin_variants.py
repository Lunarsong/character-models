"""Skin-tone albedo variants for the customisation material variants (run AFTER bake_skin.py).

Each tone (cust_lib.SKIN_TONES) mixes CC0 MakeHuman skin albedos (same UV layout) in linear space, then goes through
exactly the base albedo recipe of bake_skin.py: x soft baked AO (read back from the R channel of
<kind>_skin_rough.png), warm grade + saturation from the preset, scalp tinted towards the preset hair colour, and the
mouth-bag darkening of face_lib (the shade map bake_skin.py writes to <kind>_mouth_shade.png).
Output: assets/textures/<kind>_skin_<tone>_base.jpg (2048^2). 'fair' is the existing <kind>_skin_base.jpg.

run: Blender -b --python-exit-code 1 -P scripts/skin_variants.py -- male female
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cust_lib import *
import face_lib
TEXO = os.environ.get("RTS_TEX_OUT") or TEX   # bake_skin.py's output folder (scratch trial bakes)

args = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
kinds = [a for a in args if a in ("male", "female")] or ["male", "female"]
RES = 2048


def load_px(path, colorspace="sRGB"):
    im = bpy.data.images.load(path, check_existing=False)
    im.colorspace_settings.name = colorspace
    if tuple(im.size) != (RES, RES):
        im.scale(RES, RES)
    a = np.array(im.pixels[:], dtype=np.float32).reshape(RES, RES, 4)
    bpy.data.images.remove(im)
    return a


def lin(c):
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def srgb(c):
    c = np.clip(c, 0, 1)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * c ** (1 / 2.4) - 0.055)


def scalp_mask(kind):
    """Emission bake of MakeHuman's 'scalp' vertex group (as bake_skin.py does), softened in texture space."""
    clear_scene()
    rig, bm = build_human(kind, parts=False, face=False)
    bake_body_for_export(bm)
    bm.shape_key_clear()
    n = len(bm.data.vertices)
    vals = np.zeros(n)
    gi = bm.vertex_groups["scalp"].index
    for v in bm.data.vertices:
        for g in v.groups:
            if g.group == gi:
                vals[v.index] = g.weight
    vals = smooth_scalp(bm.data, vals)                  # same smoothed hairline as bake_skin.py / hair_gen.py
    at = bm.data.attributes.new("m_scalp", 'FLOAT', 'POINT'); at.data.foreach_set("value", vals)
    img = bpy.data.images.new("scalp", RES, RES, alpha=False, float_buffer=False)
    img.colorspace_settings.name = "Non-Color"
    m = bpy.data.materials.new("BAKE_scalp"); m.use_nodes = True
    N = m.node_tree.nodes; L = m.node_tree.links
    for nd in list(N):
        N.remove(nd)
    out = N.new("ShaderNodeOutputMaterial"); em = N.new("ShaderNodeEmission")
    a = N.new("ShaderNodeAttribute"); a.attribute_name = "m_scalp"; a.attribute_type = 'GEOMETRY'
    L.new(a.outputs["Fac"], em.inputs["Strength"]); L.new(em.outputs[0], out.inputs["Surface"])
    t = N.new("ShaderNodeTexImage"); t.image = img; N.active = t
    bm.data.materials.clear(); bm.data.materials.append(m)
    sc = bpy.context.scene
    sc.render.engine = 'CYCLES'; sc.cycles.samples = 1
    sc.render.bake.margin = 24; sc.render.bake.margin_type = 'EXTEND'; sc.render.bake.use_selected_to_active = False
    for o in bpy.context.view_layer.objects:
        o.select_set(False)
    bm.select_set(True); bpy.context.view_layer.objects.active = bm
    bpy.ops.object.bake(type='EMIT')
    s = np.array(img.pixels[:], dtype=np.float32).reshape(RES, RES, 4)[:, :, 0]
    for _ in range(3):                                   # as bake_skin.py
        s = (s + np.roll(s, 1, 0) + np.roll(s, -1, 0) + np.roll(s, 1, 1) + np.roll(s, -1, 1)) / 5
    return s


for kind in kinds:
    t0 = time.time()
    P = PRESETS[kind]
    scalp = scalp_mask(kind)
    ao = load_px(os.path.join(TEXO, "%s_skin_rough.png" % kind), "Non-Color")[:, :, 0]
    occ = 0.62 + 0.38 * np.clip(ao, 0, 1)
    lmp = os.path.join(TEXO, "%s_lid_mask.png" % kind)   # closing-lid region (bake_skin.py / face_lib.lid_detail)
    lidm = load_px(lmp, "Non-Color")[:, :, 0] if os.path.exists(lmp) else np.zeros((RES, RES), np.float32)
    ldp = os.path.join(TEXO, "%s_lid_domain.npz" % kind)
    lid_dom = face_lib.load_lid_domain(ldp) if os.path.exists(ldp) else None
    occ = 1 - (1 - occ) * (1 - 0.6 * lidm)
    grade = np.array(P.get("skin_grade", (1.0, 0.97, 0.93)), dtype=np.float32)
    hc = np.array(P.get("hair_rgb", (0.25, 0.17, 0.11)), dtype=np.float32) * 0.75
    shp = os.path.join(TEXO, "%s_mouth_shade.png" % kind)
    shade = load_px(shp, "Non-Color")[:, :, 0] if os.path.exists(shp) else np.ones((RES, RES), np.float32)
    log(kind, "mouth-bag shade map", shp if os.path.exists(shp) else "missing (no darkening)", "min %.2f" % shade.min())
    for tone, mix, satm in SKIN_TONES[kind]:
        if mix is None:
            continue
        acc = np.zeros((RES, RES, 3), np.float32)
        for skin, w in mix:
            d = asset_file("skins", skin)
            png = [f for f in os.listdir(d) if f.endswith(".png")][0]
            acc += w * lin(load_px(os.path.join(d, png))[:, :, :3])
        A = srgb(acc)
        rgb = np.clip(A * occ[:, :, None] * grade, 0, 1)
        lum = rgb @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
        rgb = np.clip(lum[:, :, None] + (rgb - lum[:, :, None]) * P.get("skin_sat", 1.12) * satm, 0, 1)
        if lid_dom is not None:                             # as the base albedo (closed-lid texel stretch, item 8)
            rgb = face_lib.lid_detail(rgb, lidm, lid_dom, "albedo")
        tt = np.clip((scalp - 0.15) / 0.45, 0, 1)         # scalp tint as bake_skin.py (fades out at the hairline)
        tint = 0.72 * (tt * tt * (3 - 2 * tt))[:, :, None]
        rgb = rgb * (1 - tint) + hc[None, None, :] * tint
        rgb = rgb * shade[:, :, None]                      # mouth bag darkens with depth (as the base albedo)
        out = bpy.data.images.new("%s_skin_%s_base" % (kind, tone), RES, RES, alpha=False)
        B = np.ones((RES, RES, 4), dtype=np.float32); B[:, :, :3] = rgb
        out.pixels.foreach_set(B.ravel())
        dst = os.path.join(TEXO, "%s_skin_%s_base.jpg" % (kind, tone))
        out.filepath_raw = dst + ".part"; out.file_format = 'JPEG'          # temp file + rename (atomic install)
        bpy.context.scene.render.image_settings.quality = 90
        out.save()
        os.replace(dst + ".part", dst); out.filepath_raw = dst
        log("skin tone", kind, tone, out.filepath_raw, "%.2f MB" % (os.path.getsize(out.filepath_raw) / 1e6))
    log(kind, "skin variants in %.1fs" % (time.time() - t0))
