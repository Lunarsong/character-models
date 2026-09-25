"""Look-dev renders of an engine export (out/<name>_export.blend or any blend with an rts_<kind> rig):
full body front / side / back / 3-4 and face front / 3-4 / profile, studio lighting, to renders/<prefix>_*.png.

run: Blender -b out/base_male_export.blend --python-exit-code 1 -P scripts/render_look.py -- male [prefix] [cycles]
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from chr_lib import *

args = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
kind = args[0] if args else "male"
prefix = args[1] if len(args) > 1 and args[1] != "cycles" else "look_" + kind
engine = 'CYCLES' if "cycles" in args else 'BLENDER_EEVEE'
rig = bpy.data.objects["rts_" + kind]
pose_reset(rig)
H = max((rig.matrix_world @ v.co).z for o in children_meshes(rig) if o.name.endswith("_body") for v in o.data.vertices)
eye = (rig.pose.bones["eye_l"].head + rig.pose.bones["eye_r"].head) / 2

sc = render_setup(engine, (900, 1400), 128 if engine == 'CYCLES' else 64, world=0.25)
sc.view_settings.look = 'AgX - Base Contrast'
sc.view_settings.exposure = -0.3
# backdrop: large curved floor+wall so the character has contact shadows and a readable silhouette
bpy.ops.mesh.primitive_plane_add(size=60, location=(0, 0, 0))
floor = bpy.context.object; floor.name = "Floor"
fm = pbr_material("M_floor", base_color=(0.16, 0.16, 0.17, 1), rough=0.8)
set_material(floor, fm)
studio_lights(0.9 * (H / 1.85) ** 2)


def shot(name, loc, tgt, lens, res):
    sc.render.resolution_x, sc.render.resolution_y = res
    camera(loc, tgt, lens=lens)
    render(os.path.join(REN, "%s_%s.png" % (prefix, name)))


c = H * 0.52
k = H / 1.85
shot("body_front", (0, -3.5 * k, c + 0.1), (0, 0, c), 55, (800, 1300))
shot("body_34", (-2.2 * k, -2.75 * k, c + 0.25), (0, 0, c), 55, (800, 1300))
shot("body_side", (3.5 * k, 0, c + 0.1), (0, 0, c), 55, (800, 1300))
rig.rotation_euler.z = math.pi                 # back view: turn the character, keep the lighting
bpy.context.view_layer.update()
shot("body_back", (0, -3.5 * k, c + 0.1), (0, 0, c), 55, (800, 1300))
rig.rotation_euler.z = 0.0
bpy.context.view_layer.update()
f = Vector((0, eye.y, eye.z - 0.03))
shot("face_front", f + Vector((0, -0.75, 0.02)), f, 85, (900, 1000))
shot("face_34", f + Vector((-0.45, -0.6, 0.05)), f + Vector((0, 0, -0.01)), 85, (900, 1000))
shot("face_profile", f + Vector((0.75, -0.05, 0.02)), f + Vector((0, 0.02, 0)), 85, (900, 1000))
