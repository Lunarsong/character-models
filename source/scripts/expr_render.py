"""Renders expression presets (scripts/expressions.json, the same data the HTML viewer plays) on the ENGINE mesh
(out/base_<kind>_export.blend), Eevee, real materials, studio light, hair shown. One PNG per preset and view:
renders/<prefix>_<view>_<nn>_<preset>.png, then sheets with scripts/sheet.py.

views: face (front, 0.62 m), q34 (three-quarter, 0.55 m), mouth (mouth close-up, 0.3 m), side (profile of the mouth),
       eyes (both eyes, 0.32 m)
run: Blender -b out/base_<kind>_export.blend --python-exit-code 1 -P scripts/expr_render.py -- <kind>
         [--presets scripts/expressions.json] [--prefix expr_<kind>] [--views face,q34,mouth] [--only Smile,Wink L]
         [--keys "jawOpen=0.4;jawOpen=0.4+tongueOut=0.5"]    (ad-hoc key mixes instead of presets)
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from chr_lib import *

args = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
kind = args[0] if args else "male"


def opt(name, default=None):
    return args[args.index(name) + 1] if name in args else default


presets_path = opt("--presets", os.path.join(os.path.dirname(os.path.abspath(__file__)), "expressions.json"))
prefix = opt("--prefix", "expr_" + kind)
views = opt("--views", "face,q34,mouth").split(",")
only = opt("--only")
if opt("--keys"):
    presets = {}
    for mix in opt("--keys").split(";"):
        presets[mix.replace("=", "").replace("+", "_")] = {k: float(v) for k, v in (p.split("=") for p in mix.split("+"))}
else:
    presets = json.load(open(presets_path))["presets"]
if only:
    presets = {k: v for k, v in presets.items() if k in only.split(",")}
rig = bpy.data.objects["rts_" + kind]
pose_reset(rig)
meshes = [o for o in children_meshes(rig) if o.data.shape_keys]
B = lambda n: rig.matrix_world @ rig.pose.bones[n].head
eye = (B("eye_l") + B("eye_r")) / 2
mouth = (B("lip_upper_c") + B("lip_lower_c")) / 2
tgt = Vector((0, eye.y, eye.z - 0.045))
render_setup("BLENDER_EEVEE", (520, 600), 32, world=0.5)
studio_lights(0.25)
CAMS = {"face": ((tgt + Vector((0.0, -0.62, 0.02)), tgt), (520, 600)),
        "q34": ((tgt + Vector((-0.34, -0.44, 0.03)), tgt + Vector((0, 0, -0.005))), (520, 600)),
        "mouth": ((mouth + Vector((0.035, -0.30, 0.025)), mouth + Vector((0, 0, 0.004))), (560, 420)),
        "side": ((mouth + Vector((0.30, -0.14, 0.02)), mouth + Vector((0, 0.01, 0.004))), (560, 420)),
        "eyes": ((eye + Vector((0.0, -0.32, 0.0)), eye), (640, 300))}


def set_mix(mix):
    for o in meshes:
        for k in o.data.shape_keys.key_blocks[1:]:
            k.value = float(mix.get(k.name, 0.0))


for view in views:
    (loc, look), res = CAMS[view]
    camera(loc, look, lens=85 if view != "mouth" else 100)
    bpy.context.scene.render.resolution_x, bpy.context.scene.render.resolution_y = res
    for i, (name, mix) in enumerate(presets.items()):
        if name == "Talk":
            continue
        set_mix(mix)
        render(os.path.join(REN, "%s_%s_%02d_%s.png" % (prefix, view, i, name.replace(" ", "_"))))
set_mix({})
