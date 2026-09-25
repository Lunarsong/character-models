"""Close-up look-dev of the nose / lips / mouth at 0.3 - 0.5 m (Eevee, real materials, engine mesh).
Views: nose_front, nose_34, nose_below, lips_front, lips_34, lips_profile, plus open-mouth views (jawOpen 0.35, 'Big grin'
preset from scripts/expressions.json when present) -> renders/<prefix>_<view>.png
run: Blender -b out/base_<kind>_export.blend --python-exit-code 1 -P scripts/closeup_render.py -- <kind> [prefix]
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from chr_lib import *

args = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
kind = args[0] if args else "male"
prefix = args[1] if len(args) > 1 else "close_" + kind
rig = bpy.data.objects["rts_" + kind]
pose_reset(rig)
meshes = [o for o in children_meshes(rig) if o.data.shape_keys]
B = lambda n: rig.matrix_world @ rig.pose.bones[n].head
eye = (B("eye_l") + B("eye_r")) / 2
mouth = (B("lip_upper_c") + B("lip_lower_c")) / 2
nose = (B("nose_l") + B("nose_r")) / 2
render_setup("BLENDER_EEVEE", (700, 560), 48, world=0.5)
studio_lights(0.25)
pp = os.path.join(os.path.dirname(os.path.abspath(__file__)), "expressions.json")
PRE = json.load(open(pp))["presets"] if os.path.exists(pp) else {}


def set_mix(mix):
    for o in meshes:
        for k in o.data.shape_keys.key_blocks[1:]:
            k.value = float(mix.get(k.name, 0.0))


n = nose + Vector((0, 0, -0.008))
m = mouth + Vector((0, 0, 0.003))
SHOTS = [("nose_front", n + Vector((0.0, -0.32, 0.03)), n, {}), ("nose_34", n + Vector((-0.2, -0.24, 0.02)), n, {}),
         ("nose_below", n + Vector((0.0, -0.22, -0.2)), n, {}),
         ("lips_front", m + Vector((0.0, -0.3, 0.01)), m, {}), ("lips_34", m + Vector((-0.19, -0.23, 0.02)), m, {}),
         ("lips_profile", m + Vector((0.3, -0.02, 0.01)), m + Vector((0, -0.01, 0)), {}),
         ("open_front", m + Vector((0.0, -0.3, 0.02)), m, {"jawOpen": 0.35}),
         ("open_34", m + Vector((-0.17, -0.25, 0.03)), m, {"jawOpen": 0.35})]
for nm in ("Smile", "Big grin", "Laugh"):
    if nm in PRE:
        SHOTS.append((nm.replace(" ", "_").lower() + "_front", m + Vector((0.0, -0.3, 0.02)), m, PRE[nm]))
        SHOTS.append((nm.replace(" ", "_").lower() + "_34", m + Vector((-0.17, -0.25, 0.03)), m, PRE[nm]))
for name, loc, tgt, mix in SHOTS:
    set_mix(mix)
    camera(loc, tgt, lens=100)
    render(os.path.join(REN, "%s_%s.png" % (prefix, name)))
set_mix({})
