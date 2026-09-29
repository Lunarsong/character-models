"""Outfit pipeline smoke test: proves that MPFB clothes (.mhclo) load onto the LIVE base humans, fit the body, get
skinned to the rts_human skeleton, hide the body under them (MPFB delete group -> MASK modifier) and follow the
customisation morphs, and export as one validated dressed GLB (covered body deleted). Armour / clothes agents author
their pieces the same way (MakeClothes -> .mhclo) and load / export them with scripts/outfit_lib.py.

run: BLENDER_USER_RESOURCES=characters/blender_profile Blender -b characters/out/base_<kind>.blend --python-exit-code 1 \
       -P characters/scripts/outfit_smoke.py -- <kind> [clothes_dir ...]
default clothes: male -> male_worksuit01 shoes01, female -> female_casualsuit01 shoes02 (MakeHuman CC0 system assets)
writes out/test/outfit_smoke_<kind>.glb (+ _export.blend), renders/outfit_smoke_<kind>_<pose>.png (the exported dressed
mesh posed) and prints OUTFIT lines (exit 1 on a failed check); then: python3 scripts/check_glb.py out/test/outfit_smoke_*.glb
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from outfit_lib import *

args = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
kind = args[0] if args and args[0] in PRESETS else "male"
items = args[1:] or (["male_worksuit01", "shoes01"] if kind == "male" else ["female_casualsuit01", "shoes02"])
rig = [o for o in bpy.data.objects if o.type == 'ARMATURE'][0]
bm = bpy.data.objects[kind + "_body"]
bones = {b.name for b in rig.data.bones}
fail = []


def out(*a):
    print("OUTFIT", *a, flush=True)


added = []
for it in items:
    path = asset_file("clothes", it, it + ".mhclo")
    cl = add_piece(rig, bm, path, slot=it)
    added.append(cl)
    arm = [md for md in cl.modifiers if md.type == 'ARMATURE']
    vg = {g.name for g in cl.vertex_groups}
    unknown = sorted(vg - bones)
    # influences per vertex and unweighted vertices (bone groups only)
    bi = {g.index for g in cl.vertex_groups if g.name in bones}
    infl = [sum(1 for g in v.groups if g.group in bi and g.weight > 1e-4) for v in cl.data.vertices]
    unw = sum(1 for c in infl if c == 0)
    out(kind, it, "verts", len(cl.data.vertices), "parent", cl.parent and cl.parent.name,
        "armature", arm and arm[0].object and arm[0].object.name, "bone groups", len(vg & bones),
        "non-bone groups", unknown[:8], "max infl", max(infl), "unweighted", unw)
    if cl.parent != rig or not arm or arm[0].object != rig:
        fail.append(it + ": not parented / skinned to the rig")
    if unw:
        fail.append(it + ": %d unweighted verts" % unw)
    ks = cl.data.shape_keys.key_blocks.keys()[1:] if cl.data.shape_keys else []
    out(kind, it, "morphs", len(ks), "cust", sum(k.startswith("cust_") for k in ks), "face", sum(k in FACE_KEYS for k in ks))
mask = [md for md in bm.modifiers if md.type == 'MASK' and md.vertex_group != "HelperGeometry" and md.name != "Hide helpers"]
hide, ndel = covered_verts(bm)
out(kind, "body delete group verts", ndel, "mask modifiers", [(m.name, m.vertex_group, m.invert_vertex_group) for m in mask])
if not any(ndel.values()) or not mask:
    fail.append("no delete group / mask on the body")

# ---- dressed engine export (covered body deleted), then the visual check on the exported mesh ----
os.makedirs(os.path.join(OUT, "test"), exist_ok=True)
res = export_dressed(rig, bm, kind, os.path.join(OUT, "test", "outfit_smoke_%s.glb" % kind))
out(kind, "export", res)
bpy.ops.wm.save_as_mainfile(filepath=os.path.join(OUT, "test", "outfit_smoke_%s_export.blend" % kind))
added = [bpy.data.objects[kind + "_" + it] for it in items]
bm = bpy.data.objects[kind + "_body"]
render_setup("BLENDER_WORKBENCH", res=(900, 1200))
sc = bpy.context.scene
sc.display.shading.light = 'STUDIO'; sc.display.shading.color_type = 'RANDOM'; sc.display.shading.show_cavity = True
for o in bpy.data.objects:
    if o.type == 'MESH' and o.get("rts_variant_group") == "eyebrows" and not o.get("rts_default"):
        o.hide_render = True
camera((1.6, -3.2, 1.25), (0, 0, 0.95), lens=55)


def set_keys(val):
    for o in [bm] + added:
        kb = o.data.shape_keys.key_blocks if o.data.shape_keys else {}
        for nm in ("cust_body_chest_pos", "cust_body_belly_pos", "cust_body_shoulders_pos", "cust_body_arms_pos",
                   "cust_body_legs_pos", "cust_body_muscle_pos"):          # cust_lib slider morphs (iteration 2)
            if nm in kb:
                kb[nm].value = val


for pose in ("rest", "arms_up", "squat", "cust_heavy"):
    pose_reset(rig); set_keys(0.0)
    if pose == "arms_up":
        for s in ("_l", "_r"):
            sg = 1 if s == "_l" else -1
            rot(rig, "clavicle" + s, (0, 1, 0), -18 * sg)
            aim(rig, "upperarm" + s, (0.25 * sg, 0.05, 1.0))
    elif pose == "squat":                      # same as deform_test.P_squat
        z = rig.pose.bones["foot_l"].head.z
        rot(rig, "spine_01", (1, 0, 0), 22); rot(rig, "spine_03", (1, 0, 0), 10)
        for s in ("_l", "_r"):
            sg = 1 if s == "_l" else -1
            aim(rig, "thigh" + s, (0.32 * sg, -0.93, 0.12)); aim(rig, "calf" + s, (0.12 * sg, 0.38, -0.92))
            aim(rig, "foot" + s, (0.12 * sg, -1, -0.25))
            aim(rig, "upperarm" + s, (0.15 * sg, -1, -0.1)); aim(rig, "lowerarm" + s, (0.0, -1, 0.1))
        pb = rig.pose.bones["pelvis"]
        low = min(rig.pose.bones["foot_l"].head.z, rig.pose.bones["foot_r"].head.z)
        M = pb.matrix.copy(); M.translation.z -= (low - z); pb.matrix = M
        bpy.context.view_layer.update()
    elif pose == "cust_heavy":
        set_keys(1.0)
    drive_correctives(rig, bm)
    render(os.path.join(REN, "outfit_smoke_%s_%s.png" % (kind, pose)))

for f in fail:
    out("FAIL", f)
if fail:
    sys.exit(1)
out(kind, "OK")
