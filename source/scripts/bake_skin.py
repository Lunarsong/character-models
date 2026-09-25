"""Bakes the skin texture set for a base human (Cycles), written to assets/textures/:
  <kind>_skin_normal.png  tangent-space normal map (OpenGL +Y, as glTF): the subdivided (level 2) body plus a
                          procedural object-space pore / skin-grain bump, baked onto the game mesh, so the game LOD
                          shades like the subdivided cutscene mesh and skin has micro detail up close
  <kind>_skin_base.jpg    the CC0 MakeHuman skin albedo with baked ambient occlusion (creases, ears, nostrils,
                          armpits) multiplied in softly and a mild warm grade
  <kind>_skin_rough.png   roughness (G channel, glTF metallicRoughness layout; B = metallic = 0): lips / nails
                          glossier, T-zone slightly oily, body drier, plus fine variation
run BEFORE base_humans.py:  Blender -b --python-exit-code 1 -P scripts/bake_skin.py -- male [female]
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from chr_lib import *
import face_lib

args = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
kinds = [a for a in args if a in PRESETS] or ["male", "female"]
RES = 2048
TEXO = os.environ.get("RTS_TEX_OUT") or TEX   # output folder (a scratch folder for trial bakes; default assets/textures)
MOUTH_BAG_ROUGH = 0.55       # moist, not mirror-wet (0.28 + pore normals sparkled in the web viewer; 0.42 still showed the
                             # inner cheek as a bright grey wall at grazing angles in an open mouth: no occlusion in there)
os.makedirs(TEXO, exist_ok=True)


def new_image(name, alpha=False, non_color=True):
    im = bpy.data.images.get(name)
    if im:
        bpy.data.images.remove(im)
    im = bpy.data.images.new(name, RES, RES, alpha=alpha, float_buffer=False)
    im.colorspace_settings.name = "Non-Color" if non_color else "sRGB"
    return im


def bake_target_material(obj, img, extra=None):
    """Material whose active node is the image to bake into (plus optional emission graph)."""
    m = bpy.data.materials.new("BAKE_" + img.name)
    m.use_nodes = True
    nt = m.node_tree
    if extra:
        for nd in list(nt.nodes):          # the graph brings its own output node
            nt.nodes.remove(nd)
    t = nt.nodes.new("ShaderNodeTexImage"); t.image = img
    nt.nodes.active = t
    if extra:
        extra(nt)
    obj.data.materials.clear(); obj.data.materials.append(m)
    return m


def mask_attribute(obj, groups, name):
    """Float point attribute = max weight of the given vertex groups."""
    n = len(obj.data.vertices)
    vals = np.zeros(n)
    gi = {obj.vertex_groups[g].index for g in groups if g in obj.vertex_groups}
    for v in obj.data.vertices:
        for g in v.groups:
            if g.group in gi:
                vals[v.index] = max(vals[v.index], g.weight)
    a = obj.data.attributes.new(name, 'FLOAT', 'POINT')
    a.data.foreach_set("value", vals)
    return vals


def save_atomic(img, path, fmt):
    """Save a Blender image to `path` through a temporary file + rename (other builds read these textures while this
    one runs: nobody may read a half-written file)."""
    tmp = path + ".part"
    img.filepath_raw = tmp; img.file_format = fmt
    img.save()
    os.replace(tmp, path)
    img.filepath_raw = path


def closed_lid_coords(kind):
    """(rest, closed) coordinates of the MakeHuman basemesh body vertices: the live face pipeline of base_humans.py
    (face keys, eye-socket repair, face_lib.improve_face_keys with its rolled blink) on a temporary human; its
    eyeBlinkLeft + eyeBlinkRight give the closed-lid surface for the lid texture (face_lib.lid_detail)."""
    clear_scene()
    rig, bm = build_human(kind)
    make_materials(rig, kind)
    # the socket-lining test reads the lid-independent reference albedo (face_lib.lining_ref_image), as base_humans.py
    skin_img = face_lib.lining_ref_image(kind) or \
        [nd for nd in bm.data.materials[0].node_tree.nodes if nd.type == 'TEX_IMAGE'][0].image
    fix_eye_socket_keys(bm, rig, skin_img)
    face_lib.improve_face_keys(bm, rig, kind, skin_img)
    nb = face_lib.NBODY
    co = face_lib.modelled_coords(bm)[:nb]
    D = sum(face_lib.key_delta(bm, k)[:nb] for k in ("eyeBlinkLeft", "eyeBlinkRight"))
    return co, co + D


for kind in kinds:
    t0 = time.time()
    lid_rest, lid_closed = closed_lid_coords(kind)
    log("closed-lid surface for the lid texture: max lid travel %.1f mm (%.1fs)" % (
        float(np.linalg.norm(lid_closed - lid_rest, axis=1).max()) * 1e3, time.time() - t0))
    clear_scene()
    rig, bm = build_human(kind, parts=False, face=False)
    bake_body_for_export(bm)                      # modelled shape, helpers removed (keeps lips / nails groups)
    bm.shape_key_clear()                          # the rig stays: AO is baked in a spread pose (see below)
    pose_reset(rig)
    low = bm; low.name = "LOW"
    # region masks (MakeHuman extra vertex groups) for roughness
    lips = mask_attribute(low, ["lips"], "m_lips")
    mask_attribute(low, ["fingernails", "toenails"], "m_nails")
    sc_raw = mask_attribute(low, ["scalp"], "m_scalp")
    low.data.attributes["m_scalp"].data.foreach_set("value", smooth_scalp(low.data, sc_raw))   # smooth hairline
    mask_attribute(low, ["nipple", "nippleTip", "genitals"], "m_soft")
    # closing-lid region (face_lib.lid_vertex_mask, user item 8): its maps are smoothed along the stretch direction
    lidv = face_lib.lid_vertex_mask(low, rig)
    lidm = np.clip(face_lib.rasterize_uv(low, lidv, RES, lambda tv: lidv[tv].max(1) > 0, fill=0.0)[::-1], 0, 1)
    for _ in range(2):
        lidm = (lidm + np.roll(lidm, 1, 0) + np.roll(lidm, -1, 0) + np.roll(lidm, 1, 1) + np.roll(lidm, -1, 1)) / 5
    lidm = lidm.astype(np.float32)
    # noise domain of the lid detail: the lid surface part-way closed (face_lib.lid_noise_domain / lid_detail)
    lid_dom = face_lib.lid_noise_domain(low, lidv, lid_rest, lid_closed, RES)
    face_lib.save_lid_domain(os.path.join(TEXO, "%s_lid_domain.npz" % kind) + ".part.npz", lid_dom)
    os.replace(os.path.join(TEXO, "%s_lid_domain.npz" % kind) + ".part.npz", os.path.join(TEXO, "%s_lid_domain.npz" % kind))
    # mouth bag (the skin inside the lips, face_lib.mouth_cavity_depth): no pore normals there (with the wet roughness
    # they sparkled as white specks inside an open mouth in the web viewer), a smooth moist surface instead
    depth = face_lib.mouth_cavity_depth(low, rig)
    inbag = depth >= 0
    bagmask = lambda tv: inbag[tv].any(1)
    Mw = face_lib.rasterize_uv(low, inbag.astype(float), RES, bagmask, fill=0.0)[::-1]

    sc = bpy.context.scene
    sc.render.engine = 'CYCLES'
    try:
        bpy.context.preferences.addons['cycles'].preferences.compute_device_type = 'METAL'
        bpy.context.preferences.addons['cycles'].preferences.get_devices()
        sc.cycles.device = 'GPU'
    except Exception:
        pass
    sc.render.bake.margin = 24
    sc.render.bake.margin_type = 'EXTEND'

    # ---------------- normal: subdivided + pores -> game mesh ----------------
    hi = low.copy(); hi.data = low.data.copy(); hi.name = "HI"
    sc.collection.objects.link(hi)
    sd = hi.modifiers.new("sub", 'SUBSURF'); sd.levels = 2; sd.render_levels = 2; sd.quality = 3
    mh = bpy.data.materials.new("HI_pores"); mh.use_nodes = True
    nt = mh.node_tree; N = nt.nodes; L = nt.links
    bsdf = N["Principled BSDF"]
    tc = N.new("ShaderNodeTexCoord")
    vor = N.new("ShaderNodeTexVoronoi"); vor.feature = 'F1'; vor.distance = 'EUCLIDEAN'
    vor.inputs["Scale"].default_value = 1100.0; vor.inputs["Randomness"].default_value = 1.0
    L.new(tc.outputs["Object"], vor.inputs["Vector"])
    pore = N.new("ShaderNodeMapRange")            # pores: small pits where the F1 distance is small
    pore.inputs["From Min"].default_value = 0.0; pore.inputs["From Max"].default_value = 0.35
    pore.inputs["To Min"].default_value = -1.0; pore.inputs["To Max"].default_value = 0.0
    pore.clamp = True
    L.new(vor.outputs["Distance"], pore.inputs["Value"])
    grain = N.new("ShaderNodeTexNoise"); grain.inputs["Scale"].default_value = 2600.0
    grain.inputs["Detail"].default_value = 2.0
    L.new(tc.outputs["Object"], grain.inputs["Vector"])
    soft = N.new("ShaderNodeTexNoise"); soft.inputs["Scale"].default_value = 90.0; soft.inputs["Detail"].default_value = 3.0
    L.new(tc.outputs["Object"], soft.inputs["Vector"])
    a1 = N.new("ShaderNodeMath"); a1.operation = 'MULTIPLY_ADD'
    a1.inputs[1].default_value = 0.35
    L.new(grain.outputs["Fac"], a1.inputs[0]); L.new(pore.outputs["Result"], a1.inputs[2])
    a2 = N.new("ShaderNodeMath"); a2.operation = 'MULTIPLY_ADD'; a2.inputs[1].default_value = 0.6
    L.new(soft.outputs["Fac"], a2.inputs[0]); L.new(a1.outputs[0], a2.inputs[2])
    bump = N.new("ShaderNodeBump"); bump.inputs["Strength"].default_value = 0.35
    bump.inputs["Distance"].default_value = 0.00025
    L.new(a2.outputs[0], bump.inputs["Height"]); L.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
    hi.data.materials.clear(); hi.data.materials.append(mh)
    nimg = new_image("%s_skin_normal" % kind)
    bake_target_material(low, nimg)
    for o in bpy.context.view_layer.objects:
        o.select_set(False)
    hi.select_set(True); low.select_set(True); bpy.context.view_layer.objects.active = low
    sc.cycles.samples = 4
    sc.render.bake.use_selected_to_active = True
    sc.render.bake.cage_extrusion = 0.004
    sc.render.bake.max_ray_distance = 0.02
    bpy.ops.object.bake(type='NORMAL', normal_space='TANGENT', normal_r='POS_X', normal_g='POS_Y', normal_b='POS_Z')
    # rays that missed in tight crevices (between toes) leave strongly bent normals: fade those to flat
    Nm = np.array(nimg.pixels[:], dtype=np.float32).reshape(RES, RES, 4)
    nz = Nm[:, :, 2] * 2 - 1
    f = np.clip((nz - 0.25) / 0.35, 0, 1)[:, :, None]
    Nm[:, :, :3] = Nm[:, :, :3] * f + np.array([0.5, 0.5, 1.0], dtype=np.float32) * (1 - f)
    Nm[:, :, :3] = face_lib.lid_detail(Nm[:, :, :3], lidm, lid_dom, "normal")   # closed lid: no stretched pores (item 8)
    fb = np.clip(Mw * 1.2, 0, 1)[:, :, None] * 0.9                                 # mouth bag: smooth (see above)
    Nm[:, :, :3] = Nm[:, :, :3] * (1 - fb) + face_lib._blur2(Nm[:, :, :3], 3.0) * fb
    nimg.pixels.foreach_set(Nm.ravel())
    save_atomic(nimg, os.path.join(TEXO, "%s_skin_normal.png" % kind), 'PNG')
    log("baked", nimg.filepath_raw, "%.1fs" % (time.time() - t0))
    bpy.data.objects.remove(hi, do_unlink=True)

    # ---------------- ambient occlusion on the game mesh ----------------
    # baked with arms raised to T and legs apart, so fingers / arms do not print shadows onto the hips and torso
    for sd_ in ("_l", "_r"):
        sgn = 1 if sd_ == "_l" else -1
        for b in ("upperarm", "lowerarm", "hand"):
            aim(rig, b + sd_, (sgn, 0, -0.05))
        rot(rig, "thigh" + sd_, (0, 1, 0), 9 * sgn)
    sc.render.bake.use_selected_to_active = False
    aimg = new_image("%s_ao" % kind)
    bake_target_material(low, aimg)
    sc.cycles.samples = 96
    sc.world = sc.world or bpy.data.worlds.new("W")
    sc.world.light_settings.distance = 0.15
    bpy.ops.object.bake(type='AO')
    pose_reset(rig)
    ao = np.array(aimg.pixels[:], dtype=np.float32).reshape(RES, RES, 4)[:, :, 0]
    log("baked AO %.1fs" % (time.time() - t0))

    # ---------------- roughness (emission bake of a mask graph) ----------------
    rimg = new_image("%s_skin_rough" % kind)

    def rough_graph(nt):
        N = nt.nodes; L = nt.links
        out = N.new("ShaderNodeOutputMaterial"); em = N.new("ShaderNodeEmission")
        L.new(em.outputs[0], out.inputs["Surface"])
        val = N.new("ShaderNodeValue"); val.outputs[0].default_value = 0.56      # body skin
        cur = val.outputs[0]
        for attr, r in (("m_scalp", 0.62), ("m_soft", 0.45), ("m_lips", 0.36), ("m_nails", 0.28)):
            at = N.new("ShaderNodeAttribute"); at.attribute_name = attr; at.attribute_type = 'GEOMETRY'
            mx = N.new("ShaderNodeMix"); mx.data_type = 'FLOAT'
            L.new(at.outputs["Fac"], mx.inputs["Factor"]); L.new(cur, mx.inputs[2]); mx.inputs[3].default_value = r
            cur = mx.outputs[0]
        # T-zone (forehead / nose): slightly oilier; object-space mask around the face centre line
        tc = N.new("ShaderNodeTexCoord"); sep = N.new("ShaderNodeSeparateXYZ"); L.new(tc.outputs["Object"], sep.inputs[0])
        ab = N.new("ShaderNodeMath"); ab.operation = 'ABSOLUTE'; L.new(sep.outputs["X"], ab.inputs[0])
        mr = N.new("ShaderNodeMapRange"); mr.inputs["From Min"].default_value = 0.035; mr.inputs["From Max"].default_value = 0.0
        L.new(ab.outputs[0], mr.inputs["Value"])
        fr = N.new("ShaderNodeMapRange"); fr.inputs["From Min"].default_value = -0.07; fr.inputs["From Max"].default_value = -0.11
        L.new(sep.outputs["Y"], fr.inputs["Value"])                               # only the front of the head
        hz = N.new("ShaderNodeMapRange"); hz.inputs["From Min"].default_value = HEAD_Z - 0.16; hz.inputs["From Max"].default_value = HEAD_Z - 0.06
        L.new(sep.outputs["Z"], hz.inputs["Value"])
        m1 = N.new("ShaderNodeMath"); m1.operation = 'MULTIPLY'; L.new(mr.outputs[0], m1.inputs[0]); L.new(fr.outputs[0], m1.inputs[1])
        m2 = N.new("ShaderNodeMath"); m2.operation = 'MULTIPLY'; L.new(m1.outputs[0], m2.inputs[0]); L.new(hz.outputs[0], m2.inputs[1])
        mt = N.new("ShaderNodeMix"); mt.data_type = 'FLOAT'; L.new(m2.outputs[0], mt.inputs["Factor"])
        L.new(cur, mt.inputs[2]); mt.inputs[3].default_value = 0.44; cur = mt.outputs[0]
        nz = N.new("ShaderNodeTexNoise"); nz.inputs["Scale"].default_value = 60.0; L.new(tc.outputs["Object"], nz.inputs["Vector"])
        ad = N.new("ShaderNodeMath"); ad.operation = 'MULTIPLY_ADD'; ad.inputs[1].default_value = 0.08
        L.new(nz.outputs["Fac"], ad.inputs[0]); L.new(cur, ad.inputs[2])
        sub = N.new("ShaderNodeMath"); sub.operation = 'SUBTRACT'; sub.inputs[1].default_value = 0.04
        L.new(ad.outputs[0], sub.inputs[0])
        cmb = N.new("ShaderNodeCombineColor")
        L.new(sub.outputs[0], cmb.inputs["Green"]); L.new(sub.outputs[0], cmb.inputs["Red"])
        L.new(cmb.outputs[0], em.inputs["Color"])

    co = np.array([v.co[:] for v in low.data.vertices])
    HEAD_Z = float(co[:, 2].max())
    bake_target_material(low, rimg, rough_graph)
    sc.cycles.samples = 1
    bpy.ops.object.bake(type='EMIT')
    R = np.array(rimg.pixels[:], dtype=np.float32).reshape(RES, RES, 4)
    R[:, :, 2] = 0.0                              # B = metallic 0
    R[:, :, 0] = ao                               # R = AO (kept for engines that read ORM; glTF ignores it here)
    rimg.pixels.foreach_set(R.ravel())
    log("baked roughness %.1fs" % (time.time() - t0))

    # ---------------- scalp mask (emission bake of the MakeHuman scalp group) ----------------
    simg = new_image("%s_scalp_mask" % kind)

    def scalp_graph(nt):
        N = nt.nodes; L = nt.links
        out = N.new("ShaderNodeOutputMaterial"); em = N.new("ShaderNodeEmission")
        at = N.new("ShaderNodeAttribute"); at.attribute_name = "m_scalp"; at.attribute_type = 'GEOMETRY'
        L.new(at.outputs["Fac"], em.inputs["Strength"]); L.new(em.outputs[0], out.inputs["Surface"])
    bake_target_material(low, simg, scalp_graph)
    bpy.ops.object.bake(type='EMIT')
    scalp = np.array(simg.pixels[:], dtype=np.float32).reshape(RES, RES, 4)[:, :, 0]
    for _ in range(3):                                # soften the hairline in texture space
        scalp = (scalp + np.roll(scalp, 1, 0) + np.roll(scalp, -1, 0) + np.roll(scalp, 1, 1) + np.roll(scalp, -1, 1)) / 5

    # ---------------- albedo: CC0 skin * soft AO, mild warm grade ----------------
    skin_dir = os.path.dirname(asset_file("skins", PRESETS[kind]["skin"]))
    src = bpy.data.images.load(os.path.join(skin_dir, [f for f in os.listdir(skin_dir) if f.endswith(".png")][0]))
    if tuple(src.size) != (RES, RES):
        src.scale(RES, RES)
    A = np.array(src.pixels[:], dtype=np.float32).reshape(RES, RES, 4)
    occ = 0.62 + 0.38 * np.clip(ao, 0, 1)
    occ = 1 - (1 - occ) * (1 - 0.6 * lidm)          # the lid crease's AO would become a dark band on the closed lid
    rgb = A[:, :, :3] * occ[:, :, None]
    grade = np.array(PRESETS[kind].get("skin_grade", (1.0, 0.97, 0.93)), dtype=np.float32)
    rgb = np.clip(rgb * grade, 0, 1)
    lum = rgb @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    rgb = np.clip(lum[:, :, None] + (rgb - lum[:, :, None]) * PRESETS[kind].get("skin_sat", 1.12), 0, 1)
    # lining reference (face_lib.lining_ref_image): the same albedo WITHOUT the closed-lid treatment below. The eye-socket
    # lining test of the face pipeline (chr_lib.fix_eye_socket_keys, red texels) reads it, so the face keys never depend
    # on the lid texture (round 5: lid_detail's de-reddened margin rows moved 3-8 lining vertices and broke the male
    # right blink's stretch gate; and every bake fed the previous bake's lid texture back into the geometry)
    ref = rgb.copy()
    rgb = face_lib.lid_detail(rgb, lidm, lid_dom, "albedo")   # closed-lid texel stretch (item 8)
    lm = new_image("%s_lid_mask" % kind)                # for skin_variants.py (same treatment on the tone albedos)
    Lm = np.ones((RES, RES, 4), dtype=np.float32); Lm[:, :, 0] = Lm[:, :, 1] = Lm[:, :, 2] = lidm
    lm.pixels.foreach_set(Lm.ravel())
    save_atomic(lm, os.path.join(TEXO, "%s_lid_mask.png" % kind), 'PNG')
    # hair roots under the groom: tint the scalp towards the hair colour so card gaps read as dense hair
    hc = np.array(PRESETS[kind].get("hair_rgb", (0.25, 0.17, 0.11)), dtype=np.float32) * 0.75
    tt = np.clip((scalp - 0.15) / 0.45, 0, 1)
    tint = 0.72 * (tt * tt * (3 - 2 * tt))[:, :, None]      # strong under the groom, fading out at the hairline
    rgb = rgb * (1 - tint) + hc[None, None, :] * tint
    ref = ref * (1 - tint) + hc[None, None, :] * tint
    # mouth cavity: the mouth bag darkens with depth behind the lips (engines have no occlusion inside the mouth,
    # an open mouth would otherwise look lit and hollow) and is wet (low roughness)
    shade = np.where(inbag, 1 - 0.84 * face_lib.smoothstep(0.0015, 0.020, depth), 1.0)
    S = face_lib.rasterize_uv(low, shade, RES, bagmask)[::-1]               # Blender pixel rows (bottom first)
    rgb = rgb * S[:, :, None]
    ref = ref * S[:, :, None]
    # the same shade map for the skin-tone variants (skin_variants.py multiplies its albedos by it)
    shimg = new_image("%s_mouth_shade" % kind)
    Sh = np.ones((RES, RES, 4), dtype=np.float32); Sh[:, :, 0] = Sh[:, :, 1] = Sh[:, :, 2] = S
    shimg.pixels.foreach_set(Sh.ravel())
    save_atomic(shimg, os.path.join(TEXO, "%s_mouth_shade.png" % kind), 'PNG')
    R = np.array(rimg.pixels[:], dtype=np.float32).reshape(RES, RES, 4)
    R[:, :, 1] = R[:, :, 1] * (1 - Mw) + MOUTH_BAG_ROUGH * Mw
    rimg.pixels.foreach_set(R.ravel())
    save_atomic(rimg, os.path.join(TEXO, "%s_skin_rough.png" % kind), 'PNG')
    log("mouth cavity darkening: %d bag vertices, shade %.2f..1" % (int(inbag.sum()), float(shade.min())))
    out = new_image("%s_skin_base" % kind, non_color=False)
    B = np.ones((RES, RES, 4), dtype=np.float32); B[:, :, :3] = rgb
    out.pixels.foreach_set(B.ravel())
    sc.render.image_settings.quality = 92
    save_atomic(out, os.path.join(TEXO, "%s_skin_base.jpg" % kind), 'JPEG')
    oref = new_image("%s_skin_lining_ref" % kind, non_color=False)
    B[:, :, :3] = ref
    oref.pixels.foreach_set(B.ravel())
    save_atomic(oref, os.path.join(TEXO, face_lib.LINING_REF % kind), 'JPEG')
    log(kind, "skin textures done in %.1fs" % (time.time() - t0))
