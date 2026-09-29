"""Before / after close-ups for the face + hair items (user round 2 items 7-11, judge M17 / M18 / M24) on the ENGINE mesh
(out/base_<kind>_export.blend), Eevee, real materials, studio light, 0.25 - 1 m. Hidden alternates stay hidden unless
--part picks one (e.g. --part hair=male_hair__long01 or eyebrows=male_eyebrows__eyebrow003).

groups:
  brows   eyebrows front / grazing from below / 3-4 in Neutral, Angry, browDown, Surprised, brow + forehead sliders
  eyes    left eye front / 3-4 / profile (lashes against the lid) open, half blink, blink, Wink L, blink + squint;
          'uv' = the same with a UV checker on the skin (texture stretch on the closed lid)
  mouth   lips front / 3-4 / profile at rest, viseme_PP, mouthPress, mouthClose + jawOpen, jawOpen 0.35, Smile,
          Big grin, Laugh (teeth, gums, lip seal)
  teeth   the dentition + tongue alone, front / 3-4, closed and jawOpen 0.6
  hair    head front / 3-4 / side / back / top / RTS-like high angle at 0.6 - 1 m, + 0.35 m hairline close-ups
writes renders/facehair/<tag>_<kind>_<group>_<shot>.png
run: Blender -b out/base_<kind>_export.blend --python-exit-code 1 -P scripts/face_hair_shots.py -- <kind> <tag> [groups] [--part role=obj]
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from chr_lib import *

args = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
kind, tag = args[0], args[1]
groups = [a for a in args[2:] if not a.startswith("--") and "=" not in a] or ["brows", "eyes", "mouth", "hair"]
parts = dict(a.split("=", 1) for a in args[2:] if "=" in a and not a.startswith("--"))
D = os.path.join(REN, "facehair")
os.makedirs(D, exist_ok=True)
rig = bpy.data.objects["rts_" + kind]
pose_reset(rig)
MESH = [o for o in bpy.data.objects if o.type == 'MESH' and o.parent == rig]
for role, name in parts.items():                      # swap a part: hide the default of that role, show the alternate
    for o in MESH:
        if o.get("rts_part") == role:
            o.hide_render = o.name != name
            o.hide_set(o.name != name)
meshes = [o for o in MESH if o.data.shape_keys]
B = lambda n: rig.matrix_world @ rig.pose.bones[n].head
eyeL, eyeR = B("eye_l"), B("eye_r")
eye = (eyeL + eyeR) / 2
mouth = (B("lip_upper_c") + B("lip_lower_c")) / 2
PRE = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "expressions.json")))["presets"]
render_setup("BLENDER_EEVEE", (640, 480), 48, world=0.5)
studio_lights(0.25)


def set_mix(mix):
    for o in meshes:
        for k in o.data.shape_keys.key_blocks[1:]:
            k.value = float(mix.get(k.name, 0.0))


def shot(group, name, loc, tgt, mix, res=(640, 480), lens=100):
    set_mix({k: v for k, v in mix.items() if not k.startswith("_")})
    bpy.context.scene.render.resolution_x, bpy.context.scene.render.resolution_y = res
    camera(Vector(loc), Vector(tgt), lens=lens)
    render(os.path.join(D, "%s_%s_%s_%s.png" % (tag, kind, group, name)))


def checker():
    """UV checker on the skin (per-object material swap, restored afterwards)."""
    body = bpy.data.objects[kind + "_body"]
    old = body.data.materials[0]
    m = bpy.data.materials.new("UVCHECK"); m.use_nodes = True
    nt = m.node_tree; b = nt.nodes["Principled BSDF"]
    ch = nt.nodes.new("ShaderNodeTexChecker"); ch.inputs["Scale"].default_value = 400.0
    ch.inputs["Color1"].default_value = (0.85, 0.6, 0.5, 1); ch.inputs["Color2"].default_value = (0.25, 0.12, 0.1, 1)
    uvn = nt.nodes.new("ShaderNodeUVMap"); nt.links.new(uvn.outputs[0], ch.inputs["Vector"])
    nt.links.new(ch.outputs["Color"], b.inputs["Base Color"])
    body.data.materials[0] = m
    return lambda: body.data.materials.__setitem__(0, old)


if "brows" in groups:
    for o in MESH:
        if o.get("rts_part") == "hair":
            o.hide_render = True
    bz = eye + Vector((0, 0, 0.022))
    for st, mix in (("neutral", {}), ("angry", PRE["Angry"]), ("browDown", {"browDownLeft": 1, "browDownRight": 1}),
                    ("surprised", PRE["Surprised"]), ("ridge_heavy", {"cust_brows_ridge_pos": 1}),
                    ("brows_low", {"cust_brows_height_neg": 1, "browDownLeft": 1, "browDownRight": 1}),
                    ("old_gaunt", {"cust_face_age_pos": 1, "cust_face_weight_neg": 1, "browDownLeft": 0.7, "browDownRight": 0.7})):
        shot("brows", "front_" + st, bz + Vector((0, -0.34, 0.0)), bz, mix, (760, 380))
        shot("brows", "graze_" + st, eyeL + Vector((0.09, -0.19, -0.06)), eyeL + Vector((0.0, -0.01, 0.018)), mix, (640, 420))
    for o in MESH:
        if o.get("rts_part") == "hair" and (o.name == parts.get("hair") or (not parts.get("hair") and not o.get("rts_alt"))):
            o.hide_render = False

if "eyes" in groups or "uv" in groups:
    for o in MESH:
        if o.get("rts_part") == "hair":
            o.hide_render = True
    variants = []
    if "eyes" in groups:
        variants.append(("", None))
    if "uv" in groups:
        variants.append(("uv_", checker))
    for pre, fn in variants:
        restore = fn() if fn else None
        for st, mix in (("open", {}), ("blink50", {"eyeBlinkLeft": 0.5}), ("blink", {"eyeBlinkLeft": 1}),
                        ("wink", PRE["Wink L"]), ("blink_squint", {"eyeBlinkLeft": 1, "eyeSquintLeft": 1}),
                        ("squint", {"eyeSquintLeft": 1})):
            shot("eyes", pre + "front_" + st, eyeL + Vector((0.0, -0.25, 0.004)), eyeL + Vector((0, 0, 0.002)), mix, (640, 400))
            shot("eyes", pre + "34_" + st, eyeL + Vector((0.16, -0.19, 0.01)), eyeL + Vector((0, 0, 0.002)), mix, (640, 400))
            if not pre:
                shot("eyes", "profile_" + st, eyeL + Vector((0.25, -0.03, 0.005)), eyeL + Vector((0, -0.012, 0.002)), mix, (640, 400))
        if restore:
            restore()
    for o in MESH:
        if o.get("rts_part") == "hair" and (o.name == parts.get("hair") or (not parts.get("hair") and not o.get("rts_alt"))):
            o.hide_render = False

if "mouth" in groups:
    m = mouth + Vector((0, 0, 0.002))
    for st, mix in (("rest", {}), ("PP", {"viseme_PP": 1}), ("press", {"mouthPressLeft": 1, "mouthPressRight": 1}),
                    ("close_jaw", {"mouthClose": 0.6, "jawOpen": 0.6}), ("open", {"jawOpen": 0.35}),
                    ("aa", {"viseme_aa": 1}), ("smile", PRE["Smile"]), ("grin", PRE["Big grin"]), ("laugh", PRE["Laugh"])):
        shot("mouth", "front_" + st, m + Vector((0.0, -0.3, 0.012)), m, mix, (600, 420))
        shot("mouth", "34_" + st, m + Vector((-0.17, -0.25, 0.02)), m, mix, (600, 420))
        if st in ("rest", "PP", "press", "open"):
            shot("mouth", "profile_" + st, m + Vector((0.3, -0.03, 0.01)), m + Vector((0, -0.01, 0)), mix, (600, 420))

if "teeth" in groups:                                   # the dentition alone (shape, contacts, papillae, shading)
    shown = {o.name: o.hide_render for o in MESH}
    for o in MESH:
        o.hide_render = not (o.name.endswith("_teeth") or o.name.endswith("_tongue"))
    m = mouth + Vector((0, 0, -0.002))
    for st, mix in (("closed", {}), ("open", {"jawOpen": 0.6})):
        shot("teeth", "front_" + st, m + Vector((0.0, -0.22, 0.006)), m, mix, (720, 480))
        shot("teeth", "34_" + st, m + Vector((-0.13, -0.17, 0.012)), m + Vector((-0.008, 0.01, 0)), mix, (720, 480))
    for o in MESH:
        o.hide_render = shown[o.name]

if "hair" in groups:
    hb = B("head")
    top = max((rig.matrix_world @ v.co).z for o in MESH if o.name == kind + "_body" for v in o.data.vertices)
    c = Vector((0, hb.y + 0.01, top - 0.1))
    for nm, off, lens in (("front", (0, -0.9, 0.05), 85), ("34", (-0.62, -0.62, 0.12), 85), ("side", (0.9, 0.0, 0.05), 85),
                          ("back", (0, 0.9, 0.08), 85), ("back34", (0.6, 0.62, 0.2), 85), ("top", (0.0, -0.05, 0.9), 85),
                          ("rtsangle", (-0.35, -0.55, 0.75), 85)):
        shot("hair", nm, c + Vector(off), c + Vector((0, 0, 0.02)), {}, (620, 620), lens)
    for nm, off, tg in (("hairline_front", (0, -0.35, 0.06), (0, -0.07, 0.035)), ("hairline_34", (-0.25, -0.26, 0.06), (-0.03, -0.05, 0.03)),
                        ("crown", (0.1, 0.25, 0.33), (0, 0.03, 0.06)), ("temple", (0.33, -0.12, 0.03), (0.06, -0.02, 0.0))):
        shot("hair", nm, c + Vector(off), c + Vector(tg), {}, (620, 480), 85)
set_mix({})
