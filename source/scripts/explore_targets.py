"""Art-direction helper: renders a base human's face with candidate MakeHuman detail targets one at a time (front +
profile tiles) so presets can be chosen by eye. Output renders/explore_<kind>_<target>.png, then sheet.py.
run: Blender -b --python-exit-code 1 -P scripts/explore_targets.py -- male 0.7 target1 target2 ...
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from chr_lib import *

args = sys.argv[sys.argv.index("--") + 1:]
kind, val, cands = args[0], float(args[1]), args[2:]
rig, bm = build_human(kind, parts=False, face=False)
make_materials_body_only = True
skin = pbr_material("M_skin", base=os.path.join(TEX, "%s_skin_base.jpg" % kind), rough_img=os.path.join(TEX, "%s_skin_rough.png" % kind),
                    normal=os.path.join(TEX, "%s_skin_normal.png" % kind), sss=0.15, spec=0.45)
set_material(bm, skin)
keys = {}
for c in cands:
    names = c.split("+")
    ks = []
    for nm in names:
        p = TargetService.target_full_path(nm)
        assert p, nm
        ks.append(TargetService.load_target(bm, p, weight=0.0, name="X_" + nm))
    keys[c] = ks
render_setup("BLENDER_EEVEE", (360, 440), 32, world=0.35)
studio_lights(0.5)
e = (rig.pose.bones["eye_l"].head + rig.pose.bones["eye_r"].head) / 2
f = Vector((0, e.y, e.z - 0.035))
for c in ["base"] + cands:
    for kk in keys.values():
        for k in kk:
            k.value = 0.0
    for k in keys.get(c, []):
        k.value = val
    for view, loc in (("a", f + Vector((0, -0.8, 0.02))), ("b", f + Vector((0.8, -0.02, 0.02)))):
        camera(loc, f, lens=85)
        render(os.path.join(REN, "explore_%s_%s_%s.png" % (kind, c.replace("+", "_"), view)))
