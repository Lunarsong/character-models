"""Shared helpers for the character pipeline (run inside Blender with BLENDER_USER_RESOURCES=characters/blender_profile).

Every script adds characters/scripts to sys.path and imports this module. It owns: enabling MPFB, the male / female
presets, building an MPFB human with the rts_human rig + CC0 body parts + ARKit/viseme shape keys, glTF-friendly PBR
materials, the export bake (modelling targets baked, helpers deleted, face keys kept, <=4 weights) and render setup.
"""
import bpy, os, sys, json, math, addon_utils
import numpy as np
from mathutils import Vector, Matrix, Euler

CH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(CH, "out"); REN = os.path.join(CH, "renders"); ASSETS = os.path.join(CH, "assets")
OUT = os.environ.get("RTS_OUT") or OUT          # scratch output dir for trial builds (face / hair agent, iteration 2)
TEX = os.path.join(ASSETS, "textures")
addon_utils.enable("bl_ext.user_default.mpfb", default_set=False)
bpy.context.preferences.filepaths.save_version = 0      # no .blend1 backups in out/
from bl_ext.user_default.mpfb.services import (HumanService, TargetService, RigService, ObjectService, AssetService,
                                               LocationService)
from bl_ext.user_default.mpfb.services.faceservice import FaceService, ARKIT_FACEUNITS, META_VISEMES

USER_DATA = LocationService.get_user_data()
# 15 Oculus/OVR visemes. Shape keys keep MPFB's (Ready Player Me style) names; OVR id -> key name:
OVR_VISEMES = [("sil", "viseme_sil"), ("PP", "viseme_PP"), ("FF", "viseme_FF"), ("TH", "viseme_TH"),
               ("DD", "viseme_DD"), ("kk", "viseme_kk"), ("CH", "viseme_CH"), ("SS", "viseme_SS"),
               ("nn", "viseme_nn"), ("RR", "viseme_RR"), ("aa", "viseme_aa"), ("E", "viseme_E"),
               ("ih", "viseme_I"), ("oh", "viseme_O"), ("ou", "viseme_U")]
# extra (non-ARKit) expression shapes built by face_lib.improve_face_keys, appended after ARKit + visemes
EXTRA_FACE_KEYS = ["smileOpenLeft", "smileOpenRight", "snarl", "grimace"]
FACE_KEYS = list(ARKIT_FACEUNITS) + [k for _, k in OVR_VISEMES] + EXTRA_FACE_KEYS


def log(*a):
    print("CHR", *a, flush=True)


def clear_scene():
    for o in list(bpy.data.objects):
        bpy.data.objects.remove(o, do_unlink=True)
    for coll in (bpy.data.meshes, bpy.data.armatures, bpy.data.materials, bpy.data.images, bpy.data.cameras,
                 bpy.data.lights, bpy.data.actions):
        for d in list(coll):
            if d.users == 0:
                coll.remove(d)


def children_meshes(rig):
    return [o for o in rig.children if o.type == 'MESH']


def find_part(rig, suffix):
    for o in rig.children:
        if o.name.endswith(suffix):
            return o
    return None


def render_setup(engine="BLENDER_EEVEE", res=(1024, 1024), samples=64, world=0.35):
    sc = bpy.context.scene
    sc.render.engine = engine
    sc.render.resolution_x, sc.render.resolution_y = res
    sc.render.resolution_percentage = 100
    sc.render.film_transparent = False
    sc.view_settings.view_transform = 'AgX'
    sc.view_settings.look = 'AgX - Medium High Contrast'
    if engine == 'CYCLES':
        sc.cycles.samples = samples
        sc.cycles.use_denoising = True
        try:
            bpy.context.preferences.addons['cycles'].preferences.compute_device_type = 'METAL'
            sc.cycles.device = 'GPU'
        except Exception:
            pass
    else:
        sc.eevee.taa_render_samples = samples
    w = bpy.data.worlds.get("W") or bpy.data.worlds.new("W")
    sc.world = w
    w.use_nodes = True
    nt = w.node_tree
    bg = nt.nodes.get("Background")
    bg.inputs[0].default_value = (0.18, 0.19, 0.22, 1)
    bg.inputs[1].default_value = world
    return sc


def add_light(name, kind, loc, rot, energy, size=1.0, color=(1, 1, 1)):
    ld = bpy.data.lights.new(name, kind)
    ld.energy = energy
    ld.color = color
    if kind == 'AREA':
        ld.size = size
    elif kind == 'SUN':
        ld.angle = math.radians(size)
    o = bpy.data.objects.new(name, ld)
    bpy.context.scene.collection.objects.link(o)
    o.location = loc
    o.rotation_euler = [math.radians(a) for a in rot]
    return o


def studio_lights(scale=1.0):
    """Three-point studio rig around a character standing at the origin facing -Y."""
    for o in [o for o in bpy.data.objects if o.type == 'LIGHT']:
        bpy.data.objects.remove(o, do_unlink=True)
    add_light("Key", 'AREA', (-2.2, -3.0, 2.8), (55, 0, -35), 900 * scale, 2.0, (1.0, 0.96, 0.9))
    add_light("Fill", 'AREA', (3.0, -2.2, 1.6), (70, 0, 55), 300 * scale, 3.0, (0.85, 0.9, 1.0))
    add_light("Rim", 'AREA', (0.8, 3.2, 2.9), (-55, 0, 170), 800 * scale, 1.5, (0.9, 0.95, 1.0))


def camera(loc, target, lens=50, name="Cam", ortho=None):
    cd = bpy.data.cameras.get(name) or bpy.data.cameras.new(name)
    cd.lens = lens
    cd.clip_start = 0.01
    if ortho:
        cd.type = 'ORTHO'; cd.ortho_scale = ortho
    else:
        cd.type = 'PERSP'
    o = bpy.data.objects.get(name) or bpy.data.objects.new(name, cd)
    if o.name not in bpy.context.scene.collection.objects:
        bpy.context.scene.collection.objects.link(o)
    o.location = loc
    d = Vector(target) - Vector(loc)
    o.rotation_euler = d.to_track_quat('-Z', 'Y').to_euler()
    bpy.context.scene.camera = o
    return o


def render(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    bpy.context.scene.render.filepath = path
    bpy.ops.render.render(write_still=True)
    log("render", path)


# ------------------------------------------------------------------------------------------------------------------
# Presets. Macro values are MakeHuman's 0..1 sliders; 'height' is solved by fit_height() to hit 'stature' (metres,
# floor to top of skull, hair excluded). Detail targets are MakeHuman CC0 targets (data/targets/**).
PRESETS = {
    "male": dict(
        stature=1.85,
        phenotype=dict(gender=1.0, age=0.5, muscle=0.85, weight=0.58, proportions=1.0, height=0.5, cupsize=0.5,
                       firmness=0.5, race=dict(caucasian=1.0, african=0.0, asian=0.0)),
        targets={"torso-vshape-incr": 0.45, "measure-shoulder-dist-incr": 0.35, "torso-muscle-dorsi-incr": 0.35,
                 "torso-muscle-pectoral-incr": 0.25, "measure-waist-circ-decr": 0.15,
                 "measure-neck-circ-incr": 0.45, "neck-scale-horiz-incr": 0.15,
                 "l-upperarm-shoulder-muscle-incr": 0.35, "r-upperarm-shoulder-muscle-incr": 0.35,
                 "l-upperarm-muscle-incr": 0.2, "r-upperarm-muscle-incr": 0.2,
                 "l-lowerarm-muscle-incr": 0.2, "r-lowerarm-muscle-incr": 0.2,
                 "chin-width-incr": 0.35, "chin-prominent-incr": 0.25, "chin-bones-incr": 0.3,
                 "l-cheek-bones-incr": 0.3, "r-cheek-bones-incr": 0.3, "forehead-temple-decr": 0.2,
                 "nose-hump-incr": 0.15, "nose-scale-vert-incr": 0.1, "head-square": 0.3,
                 "eyebrows-trans-down": 0.15, "eyebrows-angle-down": 0.2,
                 # chiselled, slightly older hero face (chosen with scripts/explore_targets.py)
                 "head-fat-decr": 0.55, "l-cheek-volume-decr": 0.35, "r-cheek-volume-decr": 0.35,
                 "eyebrows-trans-forward": 0.5, "l-eye-push1-in": 0.3, "r-eye-push1-in": 0.3,
                 "chin-prognathism-incr": 0.2, "chin-height-incr": 0.15, "nose-greek-incr": 0.3, "head-age-incr": 0.2},
        parts=dict(eyes="high-poly/high-poly.mhclo", eyebrows="eyebrow008/eyebrow008.mhclo",
                   eyelashes="eyelashes01/eyelashes01.mhclo", teeth="teeth_base/teeth_base.mhclo",
                   tongue="tongue01/tongue01.mhclo", hair="rts_crop/rts_crop.mhclo"),
        skin="young_caucasian_male/young_caucasian_male.mhmat", eye_color="brownlight",
        hair_rgb=(0.17, 0.11, 0.07), hair_tint=(0.42, 0.29, 0.19), brow_color=(0.065, 0.042, 0.028), lash_color=(0.05, 0.035, 0.03),
        skin_grade=(1.0, 0.96, 0.9), skin_sat=1.06),
    "female": dict(
        stature=1.72,
        phenotype=dict(gender=0.0, age=0.5, muscle=0.78, weight=0.52, proportions=1.0, height=0.5, cupsize=0.45,
                       firmness=0.7, race=dict(caucasian=1.0, african=0.0, asian=0.0)),
        targets={"measure-shoulder-dist-incr": 0.3, "torso-vshape-incr": 0.2, "measure-waist-circ-decr": 0.15,
                 "head-fat-decr": 0.3, "l-eye-scale-incr": 0.15, "r-eye-scale-incr": 0.15,
                 "eyebrows-trans-forward": 0.15, "l-upperarm-muscle-incr": 0.15, "r-upperarm-muscle-incr": 0.15,
                 "measure-neck-circ-incr": 0.1, "l-upperarm-shoulder-muscle-incr": 0.2,
                 "r-upperarm-shoulder-muscle-incr": 0.2, "l-cheek-bones-incr": 0.25, "r-cheek-bones-incr": 0.25,
                 "chin-prominent-incr": 0.1, "nose-point-width-decr": 0.2, "mouth-lowerlip-volume-incr": 0.15,
                 "eyebrows-angle-up": 0.1},
        parts=dict(eyes="high-poly/high-poly.mhclo", eyebrows="eyebrow010/eyebrow010.mhclo",
                   eyelashes="eyelashes02/eyelashes02.mhclo", teeth="teeth_base/teeth_base.mhclo",
                   tongue="tongue01/tongue01.mhclo", hair="rts_braid/rts_braid.mhclo"),
        skin="young_caucasian_female/young_caucasian_female.mhmat", eye_color="blue",
        hair_rgb=(0.20, 0.11, 0.07), hair_tint=(0.36, 0.2, 0.12), brow_color=(0.09, 0.05, 0.035), lash_color=(0.03, 0.02, 0.02),
        skin_grade=(1.0, 0.965, 0.92), skin_sat=0.98),
}
RIG_NAME = "custom.rts_human"


def body_verts_z(bm):
    """(min, max) z of the evaluated body (non-helper) vertices including all shape keys."""
    dg = bpy.context.evaluated_depsgraph_get()
    kb = bm.data.shape_keys.key_blocks if bm.data.shape_keys else None
    if kb:
        tmp = bm.shape_key_add(name="__mix", from_mix=True)
        co = np.empty(len(bm.data.vertices) * 3); tmp.data.foreach_get("co", co)
        bm.shape_key_remove(tmp)
    else:
        co = np.empty(len(bm.data.vertices) * 3); bm.data.vertices.foreach_get("co", co)
    z = co.reshape(-1, 3)[:13380, 2] + bm.matrix_world.translation.z
    return float(z.min()), float(z.max())


def _human_info(kind, height):
    p = PRESETS[kind]
    hi = HumanService._create_default_human_info_dict()
    ph = json.loads(json.dumps(p["phenotype"])); ph["height"] = height
    hi["phenotype"] = ph
    hi["targets"] = [{"target": k, "value": v} for k, v in p["targets"].items()]
    hi["alternative_materials"] = {}
    hi["name"] = kind
    return hi


def fit_height(kind, tol=0.002):
    """Bisection on the MakeHuman 'height' macro so the body stands PRESETS[kind]['stature'] m tall."""
    want = PRESETS[kind]["stature"]
    s = HumanService.get_default_deserialization_settings(); s["subdiv_levels"] = 0
    lo, hi_, best = 0.0, 1.0, None
    for _ in range(14):
        mid = (lo + hi_) / 2
        clear_scene()
        bm = HumanService.deserialize_from_dict(_human_info(kind, mid), s)
        z0, z1 = body_verts_z(bm)
        h = z1 - z0
        best = (mid, h)
        if abs(h - want) < tol:
            break
        if h < want:
            lo = mid
        else:
            hi_ = mid
    clear_scene()
    log("fit_height", kind, "height macro %.4f -> %.3f m (want %.2f)" % (best[0], best[1], want))
    return best[0]


def build_human(kind, height=None, parts=True, face=True):
    """MPFB human (live: macros as shape keys, helpers kept) + rts_human rig + CC0 body parts + face keys."""
    p = PRESETS[kind]
    if height is None:
        height = fit_height(kind)
    AssetService.invalidate_custom_rig_cache()
    hi = _human_info(kind, height)
    hi["rig"] = RIG_NAME
    if parts:
        hi.update(p["parts"])
    s = HumanService.get_default_deserialization_settings(); s["subdiv_levels"] = 0
    bm = HumanService.deserialize_from_dict(hi, s)
    rig = bm.parent
    assert rig and rig.type == 'ARMATURE' and len(rig.data.bones) == 97, "rts_human rig missing"
    canonical_rest(rig)                            # M15: identical rest rotations on every body (see canonical_rest)
    if face:
        FaceService.load_targets(bm, load_microsoft_visemes=False, load_meta_visemes=True, load_arkit_faceunits=True)
        for k in bm.data.shape_keys.key_blocks:
            if k.name in FACE_KEYS:
                k.value = 0.0
        FaceService.interpolate_targets(bm)
    rig["rts_kind"] = kind
    rig["rts_height_macro"] = height
    return rig, bm


# ------------------------------------------------------------------------------------------------------------------
# Materials: plain Principled BSDF + image textures only, so the glTF exporter maps them 1:1 to metallic-roughness PBR
# (baseColor / normal / metallicRoughness / alpha). Blender-only extras (subsurface) do not affect the glTF.
def _img(path, colorspace="sRGB"):
    im = bpy.data.images.load(path, check_existing=True)
    im.colorspace_settings.name = colorspace
    return im


def pbr_material(name, base=None, base_color=(0.8, 0.8, 0.8, 1), rough=0.5, rough_img=None, normal=None,
                 normal_strength=1.0, alpha_mode=None, alpha_img_from_base=False, alpha=1.0, sss=0.0,
                 sss_radius=(1.0, 0.35, 0.2), spec=0.5, tint=None, coat=0.0, metallic=0.0, ior=1.45, flat_color=None):
    m = bpy.data.materials.get(name)
    if m:
        bpy.data.materials.remove(m)
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree; N = nt.nodes; L = nt.links
    for n in list(N):
        N.remove(n)
    out = N.new("ShaderNodeOutputMaterial"); out.location = (500, 0)
    b = N.new("ShaderNodeBsdfPrincipled"); b.location = (150, 0)
    L.new(b.outputs[0], out.inputs[0])
    b.inputs["Base Color"].default_value = base_color
    b.inputs["Roughness"].default_value = rough
    b.inputs["Metallic"].default_value = metallic
    b.inputs["IOR"].default_value = ior
    b.inputs["Specular IOR Level"].default_value = spec
    if sss:
        b.inputs["Subsurface Weight"].default_value = sss
        b.inputs["Subsurface Radius"].default_value = sss_radius
        b.inputs["Subsurface Scale"].default_value = 0.012
    if coat:
        b.inputs["Coat Weight"].default_value = coat
    if base:
        t = N.new("ShaderNodeTexImage"); t.location = (-500, 150); t.image = _img(base)
        src = t.outputs["Color"]
        if flat_color is not None:            # texture used only for its alpha (brows / lashes cards)
            rgb = N.new("ShaderNodeRGB"); rgb.outputs[0].default_value = (*flat_color, 1.0); src = rgb.outputs[0]
        if tint:
            mix = N.new("ShaderNodeMix"); mix.data_type = 'RGBA'; mix.blend_type = 'MULTIPLY'; mix.location = (-200, 200)
            mix.inputs["Factor"].default_value = 1.0
            L.new(src, mix.inputs[6]); mix.inputs[7].default_value = (*tint, 1.0)
            src = mix.outputs[2]
        L.new(src, b.inputs["Base Color"])
        if alpha_img_from_base:
            L.new(t.outputs["Alpha"], b.inputs["Alpha"])
    if rough_img:
        t = N.new("ShaderNodeTexImage"); t.location = (-500, -150); t.image = _img(rough_img, "Non-Color")
        sep = N.new("ShaderNodeSeparateColor"); sep.location = (-200, -150)
        L.new(t.outputs["Color"], sep.inputs[0]); L.new(sep.outputs[1], b.inputs["Roughness"])  # glTF: G = roughness
    if normal:
        t = N.new("ShaderNodeTexImage"); t.location = (-500, -450); t.image = _img(normal, "Non-Color")
        nm = N.new("ShaderNodeNormalMap"); nm.location = (-200, -450); nm.inputs["Strength"].default_value = normal_strength
        L.new(t.outputs["Color"], nm.inputs["Color"]); L.new(nm.outputs[0], b.inputs["Normal"])
    if alpha < 1.0:
        b.inputs["Alpha"].default_value = alpha
    if alpha_mode == 'MASK':
        m.surface_render_method = 'DITHERED'
        m.alpha_threshold = 0.5 if hasattr(m, "alpha_threshold") else None
        # glTF exporter reads a MASK cutoff from a 'Math: Round' / greater-than on the alpha input
        if alpha_img_from_base:
            lk = [l for l in L if l.to_socket == b.inputs["Alpha"]][0]
            gt = N.new("ShaderNodeMath"); gt.operation = 'GREATER_THAN'; gt.inputs[1].default_value = 0.35
            gt.location = (-150, -250)
            L.new(lk.from_socket, gt.inputs[0]); L.remove(lk); L.new(gt.outputs[0], b.inputs["Alpha"])
    elif alpha_mode == 'BLEND':
        m.surface_render_method = 'BLENDED'
        m.use_backface_culling = False
    return m


def set_material(obj, mat):
    obj.data.materials.clear()
    obj.data.materials.append(mat)
    for p in obj.data.polygons:
        p.material_index = 0


def asset_file(sub, *parts):
    return os.path.join(USER_DATA, sub, *parts)


def split_eye_cornea(eyes, eye_png):
    """The CC0 high-poly eye is one mesh; its cornea shell is UV-mapped to the texture's transparent disc. Give those
    faces a second material slot (index 1) so the eyeball stays opaque and the cornea is a clear glossy layer."""
    im = _img(eye_png)
    w, h = im.size
    px = np.array(im.pixels[:], dtype=np.float32).reshape(h, w, 4)
    uv = eyes.data.uv_layers.active.data
    n = 0
    for p in eyes.data.polygons:
        u = sum(uv[i].uv[0] for i in p.loop_indices) / p.loop_total
        v = sum(uv[i].uv[1] for i in p.loop_indices) / p.loop_total
        a = px[min(h - 1, max(0, int(v * h))), min(w - 1, max(0, int(u * w))), 3]
        p.material_index = 1 if a < 0.5 else 0
        n += p.material_index
    return n


def make_materials(rig, kind):
    """glTF-friendly PBR materials for the body and CC0 body parts."""
    p = PRESETS[kind]
    tk = p.get("tex_kind", kind)          # proportion variants (cust_lib.VARIANT_PRESETS) reuse the base textures
    skin_dir = os.path.dirname(asset_file("skins", p["skin"]))
    skin_png = [f for f in os.listdir(skin_dir) if f.endswith(".png")][0]
    tex = lambda f: os.path.join(TEX, f.replace(kind, tk, 1)) if os.path.exists(os.path.join(TEX, f.replace(kind, tk, 1))) else None
    body = find_part(rig, ".body")
    skin = pbr_material("M_%s_skin" % kind, base=tex("%s_skin_base.jpg" % kind) or os.path.join(skin_dir, skin_png),
                        rough=0.52, rough_img=tex("%s_skin_rough.png" % kind), normal=tex("%s_skin_normal.png" % kind),
                        sss=0.15, spec=0.45)
    if not tex("%s_skin_normal.png" % kind):
        log("WARNING: skin textures not baked yet, run scripts/bake_skin.py first")
    set_material(body, skin)
    eyes = find_part(rig, ".high-poly")
    eye_png = asset_file("eyes", "materials", p["eye_color"] + "_eye.png")
    eyes.data.materials.clear()
    eyes.data.materials.append(pbr_material("M_%s_eye" % kind, base=eye_png, rough=0.25, spec=0.5))
    eyes.data.materials.append(pbr_material("M_%s_cornea" % kind, base_color=(0.0, 0.0, 0.0, 1), rough=0.03, alpha=0.12,
                                            alpha_mode='BLEND', spec=1.0, ior=1.376))
    ncor = split_eye_cornea(eyes, eye_png)   # after the slots exist: clearing slots resets material_index
    log("eyes cornea faces", ncor)
    for suffix, sub in (("eyebrow", "eyebrows"), ("eyelashes", "eyelashes")):
        o = [c for c in rig.children if sub[:-1] in c.name or suffix in c.name]
        for ob in o:
            fn = p["parts"][sub].split("/")[0]
            png = asset_file(sub, fn, fn + ".png")
            col = p["brow_color"] if sub == "eyebrows" else p["lash_color"]
            set_material(ob, pbr_material("M_%s_%s" % (kind, sub), base=png, rough=0.6, alpha_mode='MASK',
                                          alpha_img_from_base=True, flat_color=col, spec=0.3))
    teeth = find_part(rig, ".teeth_base")
    set_material(teeth, pbr_material("M_%s_teeth" % kind, base=asset_file("teeth", "teeth_base", "teeth.png"), rough=0.22, spec=0.6))
    tongue = find_part(rig, ".tongue01")
    set_material(tongue, pbr_material("M_%s_tongue" % kind, base=asset_file("tongue", "tongue01", "tongue01_diffuse.png"), rough=0.35, sss=0.1))
    hname = p["parts"]["hair"].split("/")[0]
    hair = find_part(rig, "." + hname)
    hdir = asset_file("hair", hname)
    if hname.startswith("rts_"):          # procedural groom (scripts/hair_gen.py + hair_tex.py)
        base, hn = os.path.join(TEX, "hair_strands_%s_base.png" % tk), os.path.join(TEX, "hair_strands_normal.png")
    else:                                 # MakeHuman CC0 hair
        base, hn = os.path.join(hdir, hname + "_diffuse.png"), os.path.join(hdir, hname + "_normal.png")
    set_material(hair, pbr_material("M_%s_hair" % kind, base=base, rough=0.42, alpha_mode='MASK', alpha_img_from_base=True,
                                    normal=hn if os.path.exists(hn) else None, normal_strength=1.4, spec=0.5))
    hair.data.materials[0].use_backface_culling = False


# ------------------------------------------------------------------------------------------------------------------
# Export bake: turns the live MPFB human into an engine mesh set (in place; save the live .blend first).
def bake_body_for_export(bm):
    """Bake the modelling shape keys into the mesh, keep FACE_KEYS as deltas on top of it, delete helper geometry
    (joint cubes, eye/teeth/tongue/tights/skirt/hair helpers) and remove every non-bone vertex group."""
    import bmesh
    me = bm.data
    kb = me.shape_keys.key_blocks
    n = len(me.vertices)
    get = lambda k: (lambda a: (k.data.foreach_get("co", a), a)[1])(np.empty(n * 3, dtype=np.float64)).reshape(-1, 3)
    basis = get(kb["Basis"])
    model = basis.copy()
    face = {}
    for k in kb:
        if k.name == "Basis":
            continue
        d = get(k) - basis
        if k.name in FACE_KEYS or k.name.startswith("cust_"):     # cust_*: customisation morphs (cust_lib.py)
            face[k.name] = d
        elif k.value != 0.0:
            assert k.relative_key.name == "Basis", k.name
            model += d * k.value
    bm.shape_key_clear()
    me.vertices.foreach_set("co", model.ravel())
    # delete helpers (everything outside the 'body' group) while the mesh has no shape keys: bmesh round trips
    # write the vertex positions into the *active* shape key, which would corrupt a keyed mesh
    gi = bm.vertex_groups["body"].index
    keep = np.zeros(n, dtype=bool)
    for v in me.vertices:
        for g in v.groups:
            if g.group == gi and g.weight > 0.5:
                keep[v.index] = True
    b = bmesh.new(); b.from_mesh(me)
    b.verts.ensure_lookup_table()
    bmesh.ops.delete(b, geom=[b.verts[i] for i in np.nonzero(~keep)[0]], context='VERTS')
    b.to_mesh(me); b.free()
    assert len(me.vertices) == int(keep.sum())
    kept = model[keep]
    assert np.abs(np.array([v.co[:] for v in me.vertices]) - kept).max() < 1e-6   # bmesh kept vertex order
    bm.shape_key_add(name="Basis", from_mix=False)
    for name in FACE_KEYS + [n for n in face if n.startswith("cust_")]:
        if name in face:
            sk = bm.shape_key_add(name=name, from_mix=False)
            sk.data.foreach_set("co", (kept + face[name][keep]).ravel())
            sk.value = 0.0
    bm.active_shape_key_index = 0
    for m in [m for m in bm.modifiers if m.type != 'ARMATURE']:
        bm.modifiers.remove(m)
    return len(me.vertices)


def clean_weights(obj, rig, limit=4):
    """Keep only bone groups, cut to <= limit influences, normalise, drop zero weights."""
    bones = {b.name for b in rig.data.bones}
    for g in list(obj.vertex_groups):
        if g.name not in bones:
            obj.vertex_groups.remove(g)
    bpy.context.view_layer.objects.active = obj
    for o in bpy.context.selected_objects:
        o.select_set(False)
    obj.select_set(True)
    bpy.ops.object.vertex_group_clean(group_select_mode='ALL', limit=0.0005)
    bpy.ops.object.vertex_group_limit_total(group_select_mode='ALL', limit=limit)
    bpy.ops.object.vertex_group_normalize_all(group_select_mode='ALL', lock_active=False)
    unweighted = sum(1 for v in obj.data.vertices if not v.groups)
    return unweighted


def prune_shape_keys(obj, eps=1e-5):
    """Remove face keys that do not move this part. The body keeps every FACE_KEYS name (viseme_sil is the
    neutral pose, tongueOut only moves the tongue mesh) so engines find all 52 ARKit + 15 visemes on one mesh."""
    if not obj.data.shape_keys:
        return 0
    kb = obj.data.shape_keys.key_blocks
    n = len(obj.data.vertices)
    base = np.empty(n * 3); kb["Basis"].data.foreach_get("co", base)
    removed = 0
    for k in list(kb)[1:]:
        a = np.empty(n * 3); k.data.foreach_get("co", a)
        if np.abs(a - base).max() < eps and not obj.name.endswith("_body"):   # the body keeps all 67 names
            obj.shape_key_remove(k); removed += 1
    if len(obj.data.shape_keys.key_blocks) == 1:
        obj.shape_key_clear()
    return removed


PART_NAMES = {".body": "body", ".high-poly": "eyes", ".teeth_base": "teeth", ".tongue01": "tongue"}


def finalize_names(rig, kind):
    """Engine-facing names: rig 'rts_<kind>', meshes '<kind>_body', '<kind>_eyes', '<kind>_hair', ..."""
    rig.name = "rts_" + kind
    rig.data.name = "rts_" + kind + "_skeleton"
    for o in children_meshes(rig):
        if o.get("rts_alt"):              # alternate parts (cust_lib.name_parts names them)
            continue
        role = None
        for suf, r in PART_NAMES.items():
            if o.name.endswith(suf):
                role = r
        if role is None:
            if "eyebrow" in o.name: role = "eyebrows"
            elif "eyelash" in o.name: role = "eyelashes"
            else: role = "hair"
        o.name = "%s_%s" % (kind, role)
        o.data.name = o.name


# ------------------------------------------------------------------------------------------------------------------
# Canonical rest orientations (judge item M15, iteration 2). MPFB places every bone end with its strategy (joint cube /
# vertex mean) on the current body and keeps the rig JSON's roll value, so the rest ROTATIONS used to depend on the
# body: 80 of 97 joints differed male vs female (eye_l/r 122 deg: bones pointing along -Y, where Blender's roll
# convention is unstable; spine_05 31, neck_01 30, ...), and clips could not be shared. Convention now:
#   every rts_human bone's armature-space rest rotation = MatrixFromAxisRoll(tail - head, roll) of the rig JSON's DEFAULT
#   positions (assets/rig/rts_human.json, the neutral MakeHuman basemesh), for every body, proportion variant and
#   dressed knight. Only the joint positions (heads) and bone lengths are per body. In glTF this means identical node
#   rotations for every joint on every body (only translations differ): a clip authored on one body plays on all of
#   them (with translations taken from the target skeleton, as usual for retarget-free sharing).
# The bone's +Y axis therefore no longer runs exactly through the child joint on a given body (a few degrees off);
# deformation, skinning and the correctives do not depend on it (they use the joint positions and the rest matrices).
RIG_JSON = os.path.join(ASSETS, "rig", "rts_human.json")


def canonical_rest_frames(path=RIG_JSON):
    """{bone: 3x3 armature-space rest rotation} from the rig JSON's default head / tail positions and roll."""
    J = json.load(open(path))["bones"]
    out = {}
    for n, b in J.items():
        h, t = Vector(b["head"]["default_position"]), Vector(b["tail"]["default_position"])
        out[n] = bpy.types.Bone.MatrixFromAxisRoll((t - h).normalized(), float(b["roll"]))
    return out


def rest_rotation_error(rig, frames=None):
    """Largest angle (deg) between a bone's rest rotation and its canonical one, and the bone."""
    F = frames or canonical_rest_frames()
    worst = (0.0, "")
    for b in rig.data.bones:
        if b.name in F:
            R = F[b.name].transposed() @ b.matrix_local.to_3x3()
            a = math.degrees(math.acos(max(-1.0, min(1.0, (R[0][0] + R[1][1] + R[2][2] - 1) / 2))))
            if a > worst[0]:
                worst = (a, b.name)
    return worst


def canonical_rest(rig, frames=None):
    """Give every bone that has a canonical frame that exact rest rotation (head and length kept). Safe on a skinned
    rig: at rest every deform matrix stays identity, so no mesh moves (checked by rest_deviation at export)."""
    F = frames or canonical_rest_frames()
    for o in bpy.context.selected_objects:
        o.select_set(False)
    bpy.context.view_layer.objects.active = rig; rig.select_set(True)
    bpy.ops.object.mode_set(mode='EDIT')
    n = 0
    for eb in rig.data.edit_bones:
        M = F.get(eb.name)
        if M is None:
            continue
        L = eb.length
        eb.matrix = Matrix.Translation(eb.head) @ M.to_4x4()
        eb.length = L
        n += 1
    bpy.ops.object.mode_set(mode='OBJECT')
    err = rest_rotation_error(rig, F)
    rig["rts_rest"] = ("canonical v1: every rts_human joint's rest rotation = the rig JSON default (neutral basemesh) "
                       "frame, identical on every body; only joint translations differ (chr_lib.canonical_rest)")
    # Blender stores bones in float32 and rebuilds matrix_local down the hierarchy: ~0.05-0.2 deg of noise at the end
    # of long chains is the floor (the cross-body gate, scripts/skeleton_check.py, allows 1 deg)
    log("canonical rest rotations: %d bones, max error %.4f deg (%s)" % (n, err[0], err[1]))
    assert err[0] < 0.5, err
    return err[0]


def rest_deviation(rig):
    """Max distance between each mesh's armature-evaluated rest shape and its own vertices (must be ~0)."""
    pose_reset(rig)
    dg = bpy.context.evaluated_depsgraph_get()
    worst = {}
    for o in children_meshes(rig):
        ev = o.evaluated_get(dg); me = ev.to_mesh()
        a = np.empty(len(me.vertices) * 3); me.vertices.foreach_get("co", a)
        b = np.empty(len(o.data.vertices) * 3); o.data.vertices.foreach_get("co", b)
        worst[o.name] = float(np.abs(a - b).max()) if len(a) == len(b) else 1e9
        ev.to_mesh_clear()
    return worst


def export_glb(rig, path, meshes=None):
    bpy.ops.object.mode_set(mode='OBJECT') if bpy.context.object and bpy.context.object.mode != 'OBJECT' else None
    for o in bpy.context.view_layer.objects:
        o.select_set(False)
    meshes = children_meshes(rig) if meshes is None else meshes
    assert rig.matrix_world == Matrix.Identity(4)
    for o in meshes:
        # skinned meshes are exported as root nodes (glTF: parent transforms never affect a skinned mesh, and the
        # validator warns NODE_SKINNED_MESH_NON_ROOT otherwise); the Armature modifier still binds the skin
        assert o.matrix_world == Matrix.Identity(4), o.name
        o.parent = None
        o.select_set(True)
    rig.select_set(True)
    bpy.context.view_layer.objects.active = rig
    bpy.ops.export_scene.gltf(
        filepath=path, export_format='GLB', use_selection=True, export_yup=True, export_apply=False,
        export_skins=True, export_all_influences=False, export_morph=True, export_morph_normal=True,
        export_morph_tangent=False, export_try_sparse_sk=True, export_animations=False, export_tangents=True,
        export_image_format='AUTO', export_materials='EXPORT', export_extras=True, export_def_bones=False,
        export_rest_position_armature=True, export_leaf_bone=False)
    for o in meshes:
        o.parent = rig
        o.matrix_parent_inverse = Matrix.Identity(4)
    log("glb", path, "%.2f MB" % (os.path.getsize(path) / 1e6))


# ------------------------------------------------------------------------------------------------------------------
# Posing helpers (armature space, character faces -Y, +X is the character's left, +Z up).
def pose_reset(rig):
    for pb in rig.pose.bones:
        pb.matrix_basis = Matrix.Identity(4)
    bpy.context.view_layer.update()


def _apply(rig, pb, R):
    M = pb.matrix.copy()
    h = M.translation.copy()
    pb.matrix = Matrix.Translation(h) @ R @ Matrix.Translation(-h) @ M
    bpy.context.view_layer.update()


def rot(rig, bone, axis, deg):
    """Rotate a bone (and its children) about an armature-space axis through its head."""
    _apply(rig, rig.pose.bones[bone], Matrix.Rotation(math.radians(deg), 4, Vector(axis).normalized()))


def aim(rig, bone, direction, frac=1.0):
    """Swing a bone (minimal rotation) so its head->tail points along `direction` (armature space)."""
    pb = rig.pose.bones[bone]
    cur = (pb.tail - pb.head).normalized()
    q = cur.rotation_difference(Vector(direction).normalized())
    if frac != 1.0:
        q = Quaternion().slerp(q, frac)
    _apply(rig, pb, q.to_matrix().to_4x4())


def twist(rig, bone, deg):
    pb = rig.pose.bones[bone]
    rot(rig, bone, (pb.tail - pb.head), deg)


def bend(rig, bone, deg, hinge_ref=(1, 0, 0)):
    """Rotate about the axis perpendicular to the bone and `hinge_ref` (e.g. elbow / knee hinge)."""
    pb = rig.pose.bones[bone]
    ax = (pb.tail - pb.head).normalized().cross(Vector(hinge_ref)).normalized()
    rot(rig, bone, ax, deg)


from mathutils import Quaternion


# Shared pose recipe (iteration 1's corr_arm_up key pose; knight_qa.py and the look-dev viewer's 'Arms up' use it):
# arms raised with an 18 deg clavicle lift, as game animation does.
def _side_sign(s):
    return 1 if s == "_l" else -1


def _pose_arm_up(rig):
    for s in ("_l", "_r"):
        rot(rig, "clavicle" + s, (0, 1, 0), -18 * _side_sign(s))
        aim(rig, "upperarm" + s, (0.22 * _side_sign(s), 0.05, 1)); aim(rig, "lowerarm" + s, (0.12 * _side_sign(s), 0.05, 1))


# ------------------------------------------------------------------------------------------------------------------
# Face-key repair.
def _vertex_uv_colors(obj, img):
    me = obj.data
    w, h = img.size
    px = np.array(img.pixels[:], dtype=np.float32).reshape(h, w, 4)
    uv = np.empty(len(me.loops) * 2); me.uv_layers.active.data.foreach_get("uv", uv); uv = uv.reshape(-1, 2)
    vi = np.empty(len(me.loops), dtype=np.int64); me.loops.foreach_get("vertex_index", vi)
    acc = np.zeros((len(me.vertices), 2)); cnt = np.zeros(len(me.vertices))
    np.add.at(acc, vi, uv); np.add.at(cnt, vi, 1)
    vuv = acc / np.maximum(cnt, 1)[:, None]
    return px[np.clip((vuv[:, 1] * h).astype(int), 0, h - 1), np.clip((vuv[:, 0] * w).astype(int), 0, w - 1), :3]


def _adjacency(me):
    e = np.empty(len(me.edges) * 2, dtype=np.int64); me.edges.foreach_get("vertices", e); e = e.reshape(-1, 2)
    return e


def fix_eye_socket_keys(bm, rig, skin_img, names=None):
    """names: the shape keys to repair (default FACE_KEYS; base_humans.py runs it again on the cust_* keys).
    The CC0 faceunits move the lower-lid / cheek skin but leave MakeHuman's red eye-socket lining (the inside walls
    behind the lids) almost still, so eyeSquint / cheekSquint / eyeLookIn push the lining through the skin below the
    eye. The lining's inward-facing wall vertices and the whole lower-lid lining get their deltas replaced, in every
    face key, by a harmonic (Laplace) interpolation of the surrounding skin deltas, so the lining travels with the
    lids and cheek. The upper-lid margin keeps its authored motion (blinks still close)."""
    me = bm.data
    n = len(me.vertices)
    col = _vertex_uv_colors(bm, skin_img)
    # modelled (shape-key mixed) positions and normals; the Mask modifier is muted so indices stay 1:1
    masks = [m for m in bm.modifiers if m.type == 'MASK']
    for m in masks:
        m.show_viewport = False
    for k in me.shape_keys.key_blocks:
        if k.name in FACE_KEYS:
            k.value = 0.0
    dg = bpy.context.evaluated_depsgraph_get()
    ev = bm.evaluated_get(dg); em = ev.to_mesh()
    assert len(em.vertices) == n
    co = np.empty(n * 3); em.vertices.foreach_get("co", co); co = co.reshape(-1, 3)
    nrm = np.empty(n * 3); em.vertices.foreach_get("normal", nrm); nrm = nrm.reshape(-1, 3)
    ev.to_mesh_clear()
    for m in masks:
        m.show_viewport = True
    from mathutils.kdtree import KDTree
    unknown = np.zeros(n, bool)
    follow = []                                        # (lower-lining vertex, skin vertices, weights)
    red = (col[:, 0] > col[:, 1] * 1.5) & (col[:, 0] > 0.35)
    body = np.zeros(n, bool); body[:13380] = True
    for eb in ("eye_l", "eye_r"):
        c = np.array(rig.data.bones[eb].head_local)
        r = co - c; d = np.linalg.norm(r, axis=1)
        radial = (nrm * r).sum(1) / np.maximum(d, 1e-9)
        lower = (r[:, 2] < -0.004) & (d < 0.03) & red & body   # lower-lid lining: sits under the rising cheek skin
        unknown |= (d < 0.03) & red & body & ((radial < -0.2) | lower)
        skin = np.nonzero(body & ~red & (r[:, 2] < -0.006) & (d < 0.045))[0]
        kd = KDTree(len(skin))
        for i, v in enumerate(skin):
            kd.insert(co[v], i)
        kd.balance()
        for v in np.nonzero(lower)[0]:
            near = kd.find_n(co[v], 6)
            idx = [skin[i] for _, i, _ in near]; w = np.array([1.0 / (dist + 0.001) for _, _, dist in near])
            follow.append((v, idx, w / w.sum()))
    E = _adjacency(me)
    kb = me.shape_keys.key_blocks
    basis = np.empty(n * 3); kb["Basis"].data.foreach_get("co", basis); basis = basis.reshape(-1, 3)
    deg = np.zeros(n); np.add.at(deg, E[:, 0], 1); np.add.at(deg, E[:, 1], 1)
    fixed = 0
    for k in kb:
        if k.name not in (FACE_KEYS if names is None else names):
            continue
        a = np.empty(n * 3); k.data.foreach_get("co", a); D = a.reshape(-1, 3) - basis
        if np.abs(D[unknown]).max() < 1e-7 and np.abs(D).max() < 1e-7:
            continue
        X = D.copy()
        for it in range(300):
            s = np.zeros_like(X); np.add.at(s, E[:, 0], X[E[:, 1]]); np.add.at(s, E[:, 1], X[E[:, 0]])
            X[unknown] = s[unknown] / deg[unknown][:, None]
            for v, idx, w in follow:                   # lower lining rides with the nearest cheek skin
                X[v] = (D[idx] * w[:, None]).sum(0)
        k.data.foreach_set("co", (basis + X).ravel())
        fixed += 1
    log("eye socket lining: %d wall vertices re-interpolated (%d follow the cheek skin) in %d face keys"
        % (int(unknown.sum()), len(follow), fixed))
    return int(unknown.sum())


def _islands(n, edges):
    """Connected-component label per vertex (union-find)."""
    par = np.arange(n)

    def find(x):
        while par[x] != x:
            par[x] = par[par[x]]; x = par[x]
        return x
    for a, b in edges:
        ra, rb = find(a), find(b)
        if ra != rb:
            par[ra] = rb
    return np.array([find(i) for i in range(n)])


def decimate_with_keys(obj, ratio):
    """Collapse-decimate a keyed, skinned part (the CC0 teeth are 7.1k tris) and carry its shape keys and skin
    weights over: every new vertex takes the barycentric blend of the original triangle it lies on."""
    from mathutils.bvhtree import BVHTree
    from mathutils.geometry import barycentric_transform
    import bmesh
    me = obj.data
    n = len(me.vertices)
    kb = me.shape_keys.key_blocks if me.shape_keys else []
    keys = {k.name: np.array([d.co[:] for d in k.data]) for k in kb}
    base = np.array([v.co[:] for v in me.vertices])
    groups = [g.name for g in obj.vertex_groups]
    Wt = np.zeros((n, len(groups)))
    for v in me.vertices:
        for g in v.groups:
            Wt[v.index, g.group] = g.weight
    bm0 = bmesh.new(); bm0.from_mesh(me); bmesh.ops.triangulate(bm0, faces=bm0.faces[:])
    tris = [[v.index for v in f.verts] for f in bm0.faces]; bm0.free()
    lab0 = _islands(n, [(t[0], t[1]) for t in tris] + [(t[1], t[2]) for t in tris])
    bvhs = {}                                  # one BVH per island: upper / lower teeth never mix
    for isl in set(lab0.tolist()):
        tl = [t for t in tris if lab0[t[0]] == isl]
        bvhs[isl] = (BVHTree.FromPolygons([Vector(c) for c in base], tl), tl)
    tris_before = sum(len(p.vertices) - 2 for p in me.polygons)
    names = [k.name for k in kb]
    obj.shape_key_clear()
    mod = obj.modifiers.new("dec", 'DECIMATE'); mod.decimate_type = 'COLLAPSE'; mod.ratio = ratio
    mod.use_symmetry = True; mod.symmetry_axis = 'X'
    arm = [m for m in obj.modifiers if m.type == 'ARMATURE']
    for m in arm:
        m.show_viewport = False
    ctx = bpy.context
    for o in ctx.selected_objects:
        o.select_set(False)
    ctx.view_layer.objects.active = obj; obj.select_set(True)
    bpy.ops.object.modifier_apply(modifier="dec")
    for m in arm:
        m.show_viewport = True
    me = obj.data
    new = np.array([v.co[:] for v in me.vertices])
    e1 = np.empty(len(me.edges) * 2, dtype=np.int64); me.edges.foreach_get("vertices", e1)
    lab1 = _islands(len(new), e1.reshape(-1, 2).tolist())
    from mathutils.kdtree import KDTree
    kd = KDTree(n)
    for i, c in enumerate(base):
        kd.insert(Vector(c), i)
    kd.balance()
    isl_map = {}
    for isl in set(lab1.tolist()):
        votes = np.bincount([lab0[kd.find(Vector(new[i]))[1]] for i in np.nonzero(lab1 == isl)[0]])
        isl_map[isl] = int(votes.argmax())
    NW = np.zeros((len(new), len(groups)))
    ND = {k: np.zeros_like(new) for k in names}
    for i, p in enumerate(new):
        bvh, tris_i = bvhs[isl_map[lab1[i]]]
        loc, nrm, fi, dist = bvh.find_nearest(Vector(p))
        a, b, c = tris_i[fi]
        # barycentric weights of loc in triangle (a, b, c)
        wv = barycentric_transform(loc, Vector(base[a]), Vector(base[b]), Vector(base[c]),
                                   Vector((1, 0, 0)), Vector((0, 1, 0)), Vector((0, 0, 1)))
        w = np.clip(np.array(wv[:]), 0, None); w /= max(w.sum(), 1e-9)
        NW[i] = w[0] * Wt[a] + w[1] * Wt[b] + w[2] * Wt[c]
        for k in names:
            D = keys[k] - base
            ND[k][i] = w[0] * D[a] + w[1] * D[b] + w[2] * D[c]
    for gi, gname in enumerate(groups):
        vg = obj.vertex_groups[gname]
        vg.remove(list(range(len(new))))
        for i in np.nonzero(NW[:, gi] > 1e-5)[0]:
            vg.add([int(i)], float(NW[i, gi]), 'REPLACE')
    if names:
        obj.shape_key_add(name="Basis", from_mix=False)
        for k in names[1:]:
            sk = obj.shape_key_add(name=k, from_mix=False)
            sk.data.foreach_set("co", (new + ND[k]).ravel())
            sk.value = 0.0
    tris_after = sum(len(p.vertices) - 2 for p in me.polygons)
    log("decimated %s: %d -> %d tris, %d keys carried" % (obj.name, tris_before, tris_after, max(0, len(names) - 1)))


def set_hair_normals(hair, body, reach=0.03):
    """Hair cards shade like the head they lie on: every vertex within `reach` of the body gets a custom normal
    blended from its own normal towards the nearest skin normal (1 at the skin, 0 at `reach`). Braids / tails
    further out keep their own normals. Works with shape keys (custom normals are plain mesh data)."""
    from mathutils.bvhtree import BVHTree
    dg = bpy.context.evaluated_depsgraph_get()
    masks = [m for m in body.modifiers if m.type == 'MASK']
    for m in masks:
        m.show_viewport = False
    em = body.evaluated_get(dg).to_mesh()
    bvh = BVHTree.FromPolygons([v.co.copy() for v in em.vertices], [tuple(p.vertices) for p in em.polygons])
    pn = [p.normal.copy() for p in em.polygons]
    body.evaluated_get(dg).to_mesh_clear()
    for m in masks:
        m.show_viewport = True
    me = hair.data
    out = []
    for v in me.vertices:
        loc, nrm, fi, d = bvh.find_nearest(v.co)
        w = max(0.0, 1.0 - d / reach) ** 0.5
        n = (v.normal * (1 - w) + pn[fi] * w).normalized()
        out.append(n)
    me.normals_split_custom_set_from_vertices(out)


# ------------------------------------------------------------------------------------------------------------------
# Pose-space corrective shape keys (engine driver). The spec lives on the body mesh (custom property
# 'rts_correctives', exported to the glTF mesh extras) and is evaluated the same way in the viewer and any engine:
#   d      = the bone's local `axis` (bone direction, +Y) expressed in the local frame of the `ref` bone (current pose)
#   angle  = angle in degrees between d and `target` (a unit vector in the ref bone's local frame)
#   weight = smoothstep(clamp((angle_from - angle) / (angle_from - angle_to), 0, 1))
# Weights are applied as ordinary morph target weights (additive, before skinning).
def corrective_spec(body):
    s = body.data.get("rts_correctives")
    if s is None:
        return None
    return json.loads(s) if isinstance(s, str) else s.to_dict()


def corrective_weight(rig, c):
    pb, pr = rig.pose.bones[c["bone"]], rig.pose.bones[c["ref"]]
    d = pr.matrix.to_3x3().inverted() @ (pb.matrix.to_3x3() @ Vector(c["axis"]))
    ang = math.degrees(d.angle(Vector(c["target"]), 0.0))
    t = min(1.0, max(0.0, (c["angle_from"] - ang) / (c["angle_from"] - c["angle_to"])))
    return t * t * (3 - 2 * t)


def drive_correctives(rig, body, enabled=True):
    """Set every corrective shape key of `body` from the current pose (or 0 when disabled). Returns {name: weight}."""
    spec = corrective_spec(body)
    out = {}
    if not spec:
        return out
    kb = body.data.shape_keys.key_blocks
    for c in spec["correctives"]:
        w = corrective_weight(rig, c) if enabled else 0.0
        kb[c["name"]].value = w
        out[c["name"]] = w
    bpy.context.view_layer.update()
    return out


def smooth_scalp(me, vals, iters=12, lo=0.3, hi=0.7):
    """MakeHuman's 'scalp' group follows the quad edges (a stair-stepped hairline). Laplacian-smooth the per-vertex
    values over the mesh, then re-sharpen with a smoothstep so the hairline contour is smooth but still defined."""
    e = np.empty(len(me.edges) * 2, dtype=np.int64); me.edges.foreach_get("vertices", e); e = e.reshape(-1, 2)
    n = len(vals)
    deg = np.zeros(n); np.add.at(deg, e[:, 0], 1); np.add.at(deg, e[:, 1], 1)
    v = vals.astype(np.float64).copy()
    for _ in range(iters):
        s = np.zeros(n); np.add.at(s, e[:, 0], v[e[:, 1]]); np.add.at(s, e[:, 1], v[e[:, 0]])
        v = 0.5 * v + 0.5 * s / np.maximum(deg, 1)
    t = np.clip((v - lo) / (hi - lo), 0, 1)
    return t * t * (3 - 2 * t)
