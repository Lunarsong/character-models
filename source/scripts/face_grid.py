"""Visual check of every face shape key (52 ARKit + 15 OVR visemes) on the ENGINE mesh.
Renders renders/face_<kind>_<view>_<nn>_<key>.png (Eevee, real materials) with the key at 1.0 on every mesh that has
it, then build sheets with: python3 scripts/sheet.py face_<kind>_front 10 256

run: Blender -b out/base_<kind>_export.blend --python-exit-code 1 -P scripts/face_grid.py -- <kind> [front] [side]
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from chr_lib import *

args = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
kind = args[0] if args else "male"
views = [a for a in args[1:] if a in ("front", "side", "eyes", "mouth")] or ["front", "side", "eyes", "mouth"]
rig = bpy.data.objects["rts_" + kind]
meshes = [o for o in children_meshes(rig) if o.data.shape_keys]
head = rig.matrix_world @ rig.pose.bones["head"].head
eye = (rig.matrix_world @ rig.pose.bones["eye_l"].head + rig.matrix_world @ rig.pose.bones["eye_r"].head) / 2
tgt = Vector((0, eye.y, eye.z - 0.045))
render_setup("BLENDER_EEVEE", (360, 400), 24, world=0.5)
studio_lights(0.25)
mouth = (rig.matrix_world @ rig.pose.bones["lip_upper_c"].head + rig.matrix_world @ rig.pose.bones["lip_lower_c"].head) / 2
CAMS = {"front": (tgt + Vector((0.0, -0.62, 0.02)), tgt), "side": (tgt + Vector((0.42, -0.42, 0.0)), tgt),
        "eyes": (eye + Vector((0.0, -0.3, 0.0)), eye), "mouth": (mouth + Vector((0.07, -0.26, 0.02)), mouth)}
REGION = {"eyes": ("eye", "brow", "cheekSquint", "noseSneer"),
          "mouth": ("mouth", "jaw", "tongue", "viseme", "cheekPuff", "smileOpen", "snarl", "grimace")}
for o in children_meshes(rig):
    if o.name.endswith("_hair"):
        o.hide_render = True          # the hair cards cover the brows at this distance


def set_key(name):
    on = name.split("+")
    for o in meshes:
        for k in o.data.shape_keys.key_blocks[1:]:
            k.value = 1.0 if k.name in on else 0.0


have = {k.name for o in meshes for k in o.data.shape_keys.key_blocks}
keys = ["neutral"] + [k for k in FACE_KEYS if k in have] + ["jawOpen+tongueOut", "jawOpen+mouthClose"]
for view in views:
    camera(*CAMS[view], lens=85)
    if view in REGION:
        bpy.context.scene.render.resolution_x, bpy.context.scene.render.resolution_y = (440, 220) if view == "eyes" else (320, 260)
    else:
        bpy.context.scene.render.resolution_x, bpy.context.scene.render.resolution_y = 360, 400
    for i, name in enumerate(keys):
        if view in REGION and name != "neutral" and not any(name.startswith(r) for r in REGION[view]):
            continue
        if view not in REGION and "+" in name:
            continue
        set_key(name)
        render(os.path.join(REN, "face_%s_%s_%02d_%s.png" % (kind, view, i, name)))
set_key("neutral")
