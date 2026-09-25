"""Sagittal cut-away of the mouth (QA): hides every vertex left of a plane x = cut (character's left side) on all
meshes and renders the cut from the side, so teeth, gums, tongue and the body's mouth bag / lips can be inspected.
run: Blender -b out/base_<kind>_export.blend --python-exit-code 1 -P scripts/mouth_section.py -- <kind> [prefix]
       [--cut 0.0] [--keys "jawOpen=0.3"] [--zoom 2.5]  -> renders/<prefix>_x<cut mm>_<keys>.png
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from chr_lib import *

args = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
kind = args[0] if args else "male"
prefix = args[1] if len(args) > 1 and not args[1].startswith("--") else "section_" + kind
cut = float(args[args.index("--cut") + 1]) if "--cut" in args else 0.0
keys = args[args.index("--keys") + 1] if "--keys" in args else ""
mix = {k: float(v) for k, v in (p.split("=") for p in keys.split("+") if p)}
rig = bpy.data.objects["rts_" + kind]
pose_reset(rig)
B = lambda n: rig.matrix_world @ rig.pose.bones[n].head
mouth = (B("lip_upper_c") + B("lip_lower_c")) / 2
render_setup("BLENDER_EEVEE", (900, 760), 32, world=0.8)
studio_lights(0.25)
for o in children_meshes(rig):
    if o.data.shape_keys:
        for k in o.data.shape_keys.key_blocks[1:]:
            k.value = mix.get(k.name, 0.0)
    vg = o.vertex_groups.new(name="__keep")
    vg.add([v.index for v in o.data.vertices if v.co.x < cut + 0.0002], 1.0, 'REPLACE')
    m = o.modifiers.new("__cut", 'MASK'); m.vertex_group = "__keep"
    for mt in o.data.materials:
        mt.use_backface_culling = False
    if o.name.endswith("_hair"):
        o.hide_render = True
zoom = float(args[args.index("--zoom") + 1]) if "--zoom" in args else 1.0
m = mouth + Vector((cut, 0.028 / zoom ** 0.5 - 0.006, -0.004))
camera(m + Vector((0.3 / zoom, -0.02 / zoom, 0.02 / zoom)), m, lens=85)
tag = (keys.replace("=", "").replace("+", "_") or "rest") + ("_z%g" % zoom if zoom != 1.0 else "")
render(os.path.join(REN, "%s_x%02d_%s.png" % (prefix, int(round(cut * 1000)), tag)))
