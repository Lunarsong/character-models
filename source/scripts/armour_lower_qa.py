"""Look-dev and deformation QA renders of the dressed lower armour (out/knight_lower.blend or the _female one).

run: Blender -b out/knight_lower.blend --python-exit-code 1 -P scripts/armour_lower_qa.py -- <kind> [look] [poses] [prefix]
writes renders/<prefix>_*.png (default prefix lower_<kind>): EEVEE studio turnaround (front / 3-4 / side / back),
close-ups (hips, knee, foot, cape, props), an RTS-distance view, and posed renders (walk stride, lunge, squat, high
knee, kneel) with the tabard / cape drivers live; posed shots also in Workbench clay (poke-through check).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from chr_lib import *

args = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
kind = args[0] if args and args[0] in ("male", "female") else "male"
MODES = ("look", "poses", "combo", "cust")
do_cust = "cust" in args
do_combo = "combo" in args
do_look = "look" in args or do_combo or not any(a in args for a in MODES)
do_poses = "poses" in args or (not do_combo and not any(a in args for a in MODES))
if do_cust:
    do_look = do_poses = False
prefix = [a for a in args[1:] if a not in MODES]
prefix = prefix[0] if prefix else ("combo_" if do_combo else "lower_") + kind
rig = bpy.data.objects["rts_" + kind]
pose_reset(rig)
for o in rig.children:
    if o.type == 'MESH' and o.get("rts_variant_group") == "eyebrows" and not o.get("rts_default"):
        o.hide_render = True
H = 1.85 if kind == "male" else 1.72
k = H / 1.85

if do_combo:
    # fit check with the upper-armour agent's published pieces (render only; the blend is not saved)
    import armour_upper as AU
    from outfit_lib import add_piece
    bm = bpy.data.objects[kind + "_body"]
    man = json.load(open(os.path.join(ASSETS, "mpfb_assets", "clothes", "knight_upper_manifest.json")))
    for slot in man["pieces"]:
        path = asset_file("clothes", "knight_" + slot, "knight_" + slot + ".mhclo")
        pc = add_piece(rig, bm, path, slot=slot)
        set_material(pc, AU.make_material(AU.piece_kind(slot) if AU.piece_kind(slot) in ("mail", "plume") else slot))
        AU.ensure_ao_attr(pc)
    for o in rig.children:
        if o.name.endswith("_hair"):
            o.hide_render = True


def studio():
    sc = render_setup("BLENDER_EEVEE", (900, 1400), 64, world=0.55)
    sc.view_settings.look = 'AgX - Base Contrast'
    sc.view_settings.exposure = -0.2
    if not bpy.data.objects.get("Floor"):
        bpy.ops.mesh.primitive_plane_add(size=60, location=(0, 0, 0))
        fl = bpy.context.object; fl.name = "Floor"
        set_material(fl, pbr_material("M_floor", base_color=(0.15, 0.15, 0.16, 1), rough=0.8))
    studio_lights(0.95 * k * k)
    try:
        sc.eevee.use_shadows = True
    except Exception:
        pass
    return sc


def shot(name, loc, tgt, lens, res=(800, 1300)):
    sc = bpy.context.scene
    sc.render.resolution_x, sc.render.resolution_y = res
    camera(loc, tgt, lens=lens)
    render(os.path.join(REN, "%s_%s.png" % (prefix, name)))


def turn(deg):
    rig.rotation_euler.z = math.radians(deg)
    bpy.context.view_layer.update()


if do_look:
    studio()
    c = H * 0.52
    shot("front", (0, -3.6 * k, c + 0.1), (0, 0, c), 55)
    shot("34", (-2.3 * k, -2.8 * k, c + 0.3), (0, 0, c), 55)
    shot("side", (3.6 * k, 0, c + 0.1), (0, 0, c), 55)
    turn(180); shot("back", (0, -3.6 * k, c + 0.1), (0, 0, c), 55)
    shot("back34", (2.2 * k, -2.8 * k, c + 0.3), (0, 0, c), 55); turn(0)
    shot("hips", (-0.55 * k, -1.35 * k, 0.98 * k), (0.02, 0, 0.88 * k), 50, (1000, 1000))
    shot("knee", (-0.25 * k, -0.95 * k, 0.55 * k), (0.12 * k, 0, 0.45 * k), 50, (900, 1000))
    shot("foot", (0.55 * k, -0.75 * k, 0.28 * k), (0.2 * k, -0.06, 0.08), 50, (1000, 900))
    turn(180); shot("cape", (-0.3, -1.9 * k, 1.1 * k), (0, 0, 0.95 * k), 55, (900, 1200)); turn(0)
    shot("hand_r", (-0.95 * k, -0.95 * k, 1.05 * k), (-0.55 * k, -0.25, 0.95 * k), 45, (1000, 1000))
    shot("shield", (1.35 * k, -1.35 * k, 1.15 * k), (0.55 * k, -0.2, 0.85 * k), 45, (1000, 1100))
    # RTS camera: ~12 m away, 55 degrees down, long lens; the character ~ 110 px tall in a 480 px frame
    shot("rts", (-5.5, -8.0, 10.5), (0, 0, 0.9), 60, (480, 480))
    turn(0)

POSES = {}


def P_walk(r):
    aim(r, "thigh_l", (0.08, -0.62, -1.0)); aim(r, "calf_l", (0.06, -0.12, -1.0))
    aim(r, "thigh_r", (-0.08, 0.42, -1.0)); aim(r, "calf_r", (-0.05, 0.75, -1.0)); aim(r, "foot_r", (-0.05, -0.2, -1.0))
    rot(r, "pelvis", (0, 0, 1), 6)


def P_lunge(r):
    z = r.pose.bones["foot_l"].head.z
    aim(r, "thigh_l", (0.12, -1.0, -0.55)); aim(r, "calf_l", (0.06, 0.05, -1.0)); aim(r, "foot_l", (0.05, -1.0, -0.3))
    aim(r, "thigh_r", (-0.08, 0.55, -1.0)); aim(r, "calf_r", (-0.04, 0.95, -0.6)); aim(r, "foot_r", (0, -0.1, -1.0))
    pb = r.pose.bones["pelvis"]; M = pb.matrix.copy(); M.translation.z -= 0.16; pb.matrix = M
    bpy.context.view_layer.update()


def P_squat(r):
    z = r.pose.bones["foot_l"].head.z
    rot(r, "spine_01", (1, 0, 0), 22); rot(r, "spine_03", (1, 0, 0), 10)
    for s in ("_l", "_r"):
        sg = 1 if s == "_l" else -1
        aim(r, "thigh" + s, (0.32 * sg, -0.93, 0.12)); aim(r, "calf" + s, (0.12 * sg, 0.38, -0.92))
        aim(r, "foot" + s, (0.12 * sg, -1, -0.25))
    pb = r.pose.bones["pelvis"]
    low = min(r.pose.bones["foot_l"].head.z, r.pose.bones["foot_r"].head.z)
    M = pb.matrix.copy(); M.translation.z -= (low - z); pb.matrix = M
    bpy.context.view_layer.update()


def P_highknee(r):
    aim(r, "thigh_l", (0.1, -1.0, -0.05)); aim(r, "calf_l", (0.05, -0.1, -1.0))


def P_kneel(r):
    z = r.pose.bones["foot_l"].head.z
    aim(r, "thigh_l", (0.1, -1.0, -0.25)); aim(r, "calf_l", (0.05, 0.1, -1.0))
    aim(r, "thigh_r", (-0.08, -0.05, -1.0)); aim(r, "calf_r", (-0.03, 1.0, -0.12)); aim(r, "foot_r", (0, 0.2, -1.0))
    pb = r.pose.bones["pelvis"]; M = pb.matrix.copy(); M.translation.z -= 0.36; pb.matrix = M
    bpy.context.view_layer.update()


if do_poses:
    studio()
    for o in rig.children:                       # props would hide the legs in the stress poses
        if o.get("rts_prop"):
            o.hide_render = True
    for name, fn in (("walk", P_walk), ("lunge", P_lunge), ("squat", P_squat), ("highknee", P_highknee),
                     ("kneel", P_kneel)):
        pose_reset(rig)
        fn(rig)
        bpy.context.view_layer.update()
        c = H * 0.45
        shot("pose_%s_side" % name, (-3.4 * k, -0.6, c + 0.1), (0, 0, c), 55, (800, 1100))
        shot("pose_%s_front" % name, (0.3, -3.4 * k, c + 0.2), (0, 0, c), 55, (800, 1100))
        shot("pose_%s_34" % name, (-2.2 * k, -2.6 * k, c + 0.3), (0, 0, c), 55, (800, 1100))
        turn(180); shot("pose_%s_back34" % name, (2.0 * k, -2.6 * k, c + 0.3), (0, 0, c), 55, (800, 1100)); turn(0)
    pose_reset(rig)


if do_cust:
    # customisation morphs carried by every piece (propagated by name through the mhclo): heavy / muscular extremes
    studio()
    for o in rig.children:
        if o.get("rts_prop"):
            o.hide_render = True
    # cust_lib slider morphs (cust_<slider>_neg / _pos; iteration-2 reconcile renamed the iteration-1 one-way morphs)
    SETS = {"heavy": {"cust_body_belly_pos": 1.0, "cust_body_waist_pos": 1.0, "cust_body_hips_pos": 1.0, "cust_body_weight_pos": 0.6},
            "muscle": {"cust_body_muscle_pos": 1.0, "cust_body_legs_pos": 0.6, "cust_body_arms_pos": 0.6, "cust_body_chest_pos": 0.5},
            "thin": {"cust_body_waist_neg": 1.0, "cust_body_weight_neg": 0.8}}
    c = H * 0.45
    for nm, keys in SETS.items():
        for o in rig.children:
            kb = o.data.shape_keys.key_blocks if o.type == 'MESH' and o.data.shape_keys else {}
            for sk in kb:
                if sk.name.startswith("cust_"):
                    sk.value = keys.get(sk.name, 0.0)
        pose_reset(rig)
        shot("cust_%s_front" % nm, (0.4, -3.2 * k, c + 0.2), (0, 0, c), 55, (800, 1100))
        shot("cust_%s_side" % nm, (-3.2 * k, -0.4, c + 0.2), (0, 0, c), 55, (800, 1100))
        P_walk(rig)
        shot("cust_%s_walk" % nm, (-2.2 * k, -2.6 * k, c + 0.3), (0, 0, c), 55, (800, 1100))
