"""Deformation stress test: poses the engine mesh (out/base_<kind>_export.blend) on the rts_human skeleton, drives the
pose-space corrective shape keys the way an engine does (chr_lib.drive_correctives, spec in body['rts_correctives']),
renders clay views and writes deformation metrics (scripts/deform_metrics.py) per pose and body region.

Twist bones are driven the way the engine should drive them: upperarm_twist_01 / thigh_twist_01 counter-roll 50% of
their parent's roll, lowerarm_twist_01 takes 50% of the hand's roll about the forearm axis.

Leg poses (round 6, POSES builder): squat / viewer_squat / deep_squat / lunge / deep_lunge / kneel / sitting / high_kick /
walk_* come from the viewers' pose table through scripts/pose_qa.py (the same foot-ground contact solver as the browser:
feet planted by leg IK + a skin fit, heel-up rear foot with the toes flat, knees over the toes). Their metric baselines
reset on 25 Sep (round 6): the poses themselves changed. fists / hand_<preset> use scripts/hand_poses.py (HANDS builder)
when its presets are solved, else the legacy finger curl; the hand_<preset> family runs with the option `hands` or by name.

run: Blender -b out/base_male_export.blend --python-exit-code 1 -P scripts/deform_test.py -- male [options] [pose ...]
  options: out=<dir under renders/> (default: renders/)   nocor | nocorr (correctives off: plain LBS, files and metrics
           get a 'nocor_' prefix so they never overwrite the default run)   norender   nometrics   hands (+ hand_<preset>)
  writes <dir>/deform_<kind>_<pose>_<view>.png, <dir>/deform_<kind>_<pose>_z<detail>.png, <dir>/deform_<kind>_metrics.json
then: python3 scripts/sheet.py deform_male   (contact sheet)
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from chr_lib import *
import deform_metrics as DM

args = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
kind = args[0] if args else "male"
args = ["nocor" if a == "nocorr" else a for a in args]          # (iteration-1 spelling)
opts = {a.split("=")[0]: (a.split("=", 1)[1] if "=" in a else True) for a in args[1:] if "=" in a or a in ("nocor", "norender", "nometrics", "hands")}
only = [a for a in args[1:] if a not in opts and "=" not in a]
outdir = os.path.join(REN, opts["out"]) if "out" in opts else REN
rig = bpy.data.objects["rts_" + kind]
body = bpy.data.objects[kind + "_body"]
if kind not in PRESETS:
    import cust_lib                            # registers the proportion variants (male_stocky, ...) in PRESETS
S = PRESETS[kind]["stature"] / PRESETS["male"]["stature"]
USE_COR = not opts.get("nocor") and corrective_spec(body) is not None


def legacy_correctives(enabled):
    """Iteration-1 builds (characters_pre_reconcile/): corr_* keys driven by the angle drivers stored as a JSON list in
    the armature's 'rts_correctives' (bone->child vs ref->ref_child, linear ramp, optional forward gate), so the old
    export .blend files can be measured with the same poses / cameras / metrics for before / after comparisons."""
    if "rts_correctives" not in rig.keys() or not body.data.shape_keys:
        return {}
    kb = body.data.shape_keys.key_blocks
    out = {}
    for d in json.loads(rig["rts_correctives"]):
        pb = rig.pose.bones
        a = (pb[d["child"]].head - pb[d["bone"]].head).normalized()
        b = (pb[d["ref_child"]].head - pb[d["ref"]].head).normalized()
        b = -b if d["ref_neg"] else b
        v = min(1.0, max(0.0, (math.degrees(a.angle(b)) - d["deg"][0]) / (d["deg"][1] - d["deg"][0]))) if enabled else 0.0
        if d.get("gate_fwd"):
            g = min(1.0, max(0.0, a.dot(Vector((0, -1, 0))) / d["gate_fwd"])); v *= g * g * (3 - 2 * g)
        if d["morph"] in kb:
            kb[d["morph"]].value = v
        out[d["morph"]] = v
    bpy.context.view_layer.update()
    return out


LEGACY = corrective_spec(body) is None and "rts_correctives" in rig.keys()
if LEGACY:
    log("iteration-1 build: legacy corr_* drivers from the armature extras")
TAG = "nocor_" if opts.get("nocor") else ""                     # plain-LBS runs: separate files / metrics


def side(s):
    return 1 if s == "_l" else -1


def ankle_z(s):
    return rig.pose.bones["foot" + s].head.z


def plant_feet(ref_z):
    """Lower the pelvis so the lowest ankle returns to its rest height (feet stay on the floor)."""
    low = min(ankle_z("_l"), ankle_z("_r"))
    pb = rig.pose.bones["pelvis"]
    M = pb.matrix.copy(); M.translation.z -= (low - ref_z)
    pb.matrix = M
    bpy.context.view_layer.update()


def keep_foot(s, d=(0.0, -1.0, -0.35)):
    aim(rig, "foot" + s, (0.12 * side(s) * abs(d[1]), d[1], d[2]))


# ---- the viewers' pose table + foot-ground contact solver (scripts/pose_qa.py = POSEKIT in the viewer templates)
try:
    import pose_qa as PQA
except Exception as _e:                      # guarded: without it the leg poses fall back to their pre-round-6 versions
    PQA = None
    log("pose_qa unavailable (%s): legacy leg poses" % _e)
_KIT = []


def viewer_pose(name, w=1.0, walk_t=None):
    """pose `name` of the viewer pose table at weight w (0 = rest, 1 = the pose; in-betweens re-solve the planted legs),
    or the Walk clip at time walk_t, exactly as the browser does (G-P6 parity in pose_qa.py)."""
    if not _KIT:
        pose_reset(rig)
        _KIT.append(PQA.Kit(rig, body, kind))
    k = _KIT[0]
    k.apply(k.walk_sol(walk_t) if walk_t is not None else name, w)
    k.restore_visibility()


try:                                         # HANDS builder's applicator (guarded; only once its presets are solved)
    import hand_poses as HP
    _HD = HP.load_data()
    HAND_PRESETS = [p for p, v in (_HD or {}).get("presets", {}).items() if v.get("solved")]
except Exception:
    HP, HAND_PRESETS = None, []


def curl_fingers(s, deg, thumb=0.6):
    for f in ("index", "middle", "ring", "pinky"):
        for i, d in ((1, deg), (2, deg * 1.1), (3, deg * 0.8)):
            pb = rig.pose.bones["%s_%02d%s" % (f, i, s)]
            pb.rotation_mode = 'XYZ'
            pb.rotation_euler.x += math.radians(d)
    for i in (2, 3):
        pb = rig.pose.bones["thumb_%02d%s" % (i, s)]
        pb.rotation_mode = 'XYZ'
        pb.rotation_euler.x += math.radians(deg * thumb)
    bpy.context.view_layer.update()


def P_rest():
    pass


def P_t_pose():
    for s in ("_l", "_r"):
        aim(rig, "upperarm" + s, (side(s), 0, 0)); aim(rig, "lowerarm" + s, (side(s), 0, 0)); aim(rig, "hand" + s, (side(s), 0, 0))


def P_arms_up():
    for s in ("_l", "_r"):
        rot(rig, "clavicle" + s, (0, 1, 0), -18 * side(s))
        aim(rig, "upperarm" + s, (0.22 * side(s), 0.05, 1)); aim(rig, "lowerarm" + s, (0.12 * side(s), 0.05, 1))


def P_viewer_arms_up():
    """Exactly the viewer's 'Arms up' (what the user looked at): no clavicle / scapula motion at all."""
    for s in ("_l", "_r"):
        aim(rig, "upperarm" + s, (0.22 * side(s), 0.05, 1)); aim(rig, "lowerarm" + s, (0.12 * side(s), 0.05, 1))


def P_arms_down():
    """Idle: arms hanging at the sides (the most common game pose; the rest pose is an A-pose)."""
    for s in ("_l", "_r"):
        aim(rig, "upperarm" + s, (0.12 * side(s), 0.02, -1)); aim(rig, "lowerarm" + s, (0.05 * side(s), -0.12, -1))


def P_arms_45():
    """Half-way: arms 45 deg above horizontal (corrective in-between)."""
    for s in ("_l", "_r"):
        aim(rig, "upperarm" + s, (0.7 * side(s), 0.02, 0.7)); aim(rig, "lowerarm" + s, (0.7 * side(s), 0.02, 0.7))


def P_arms_up_180():
    """Arms straight overhead, no clavicle lift (worst case for the armpit / lat / neck)."""
    for s in ("_l", "_r"):
        aim(rig, "upperarm" + s, (0.04 * side(s), 0.03, 1)); aim(rig, "lowerarm" + s, (0.0, 0.03, 1))


def P_arm_forward_up():
    """Flexion ~135 deg (reaching forward and up), clavicle slightly raised."""
    for s in ("_l", "_r"):
        rot(rig, "clavicle" + s, (0, 1, 0), -8 * side(s))
        aim(rig, "upperarm" + s, (0.14 * side(s), -0.72, 0.68)); aim(rig, "lowerarm" + s, (0.08 * side(s), -0.6, 0.8))


def P_arm_behind_back():
    """Shoulder extension + internal rotation: forearms folded behind the lower back."""
    for s in ("_l", "_r"):
        aim(rig, "upperarm" + s, (0.3 * side(s), 0.42, -0.86)); aim(rig, "lowerarm" + s, (-0.85 * side(s), 0.45, 0.28))


def P_cross_body_reach():
    """Horizontal adduction: the left arm reaches across the chest to the right (clavicle protracted), right arm down."""
    rot(rig, "clavicle_l", (0, 0, 1), 12)
    aim(rig, "upperarm_l", (-0.5, -0.85, 0.12)); aim(rig, "lowerarm_l", (-0.75, -0.6, 0.2))
    aim(rig, "upperarm_r", (-0.12, 0.02, -1)); aim(rig, "lowerarm_r", (-0.05, -0.12, -1))


def P_arms_forward():
    for s in ("_l", "_r"):
        rot(rig, "clavicle" + s, (0, 0, 1), 10 * side(s))
        aim(rig, "upperarm" + s, (0.12 * side(s), -1, 0.05)); aim(rig, "lowerarm" + s, (0.05 * side(s), -1, 0.05))


def P_elbow_wrist():
    for s in ("_l", "_r"):
        aim(rig, "upperarm" + s, (0.25 * side(s), -0.25, -1)); aim(rig, "lowerarm" + s, (0.1 * side(s), -1, 0.0))
        bend(rig, "lowerarm" + s, 0)
        rot(rig, "lowerarm" + s, (1, 0, 0), -65)            # elbow ~150 deg total flexion
    # left: forearm pronation carried by the hand + 50% on the twist bone; right: supination
    for s, d in (("_l", 80), ("_r", -80)):
        twist(rig, "hand" + s, d); twist(rig, "lowerarm_twist_01" + s, d * 0.5)


def P_squat():
    """The viewers' parallel squat (thighs horizontal, feet flat and 22 deg out, knees over the toes, trunk parallel
    to the shin): pose table 'Squat'."""
    if PQA:
        return viewer_pose("Squat")
    z = ankle_z("_l")
    rot(rig, "spine_01", (1, 0, 0), 22); rot(rig, "spine_03", (1, 0, 0), 10)
    for s in ("_l", "_r"):
        aim(rig, "thigh" + s, (0.32 * side(s), -0.93, 0.12)); aim(rig, "calf" + s, (0.12 * side(s), 0.38, -0.92))
        aim(rig, "foot" + s, (0.12 * side(s), -1, -0.25))
        aim(rig, "upperarm" + s, (0.15 * side(s), -1, -0.1)); aim(rig, "lowerarm" + s, (0.0, -1, 0.1))
    plant_feet(z)


def P_viewer_squat():
    """Exactly the viewer's 'Squat' (= P_squat since round 6; both names kept for the metric history)."""
    if PQA:
        return viewer_pose("Squat")
    z = ankle_z("_l")
    for s in ("_l", "_r"):
        aim(rig, "thigh" + s, (0.32 * side(s), -0.93, 0.12)); aim(rig, "calf" + s, (0.12 * side(s), 0.38, -0.92))
        aim(rig, "upperarm" + s, (0.15 * side(s), -1, -0.1)); aim(rig, "lowerarm" + s, (0.0, -1, 0.1))
    pb = rig.pose.bones["spine_01"]
    rot(rig, "spine_01", (pb.matrix.col[0][0], pb.matrix.col[0][1], pb.matrix.col[0][2]), 20)
    plant_feet(z)


def P_deep_squat():
    """Heels-down deep squat (viewers' 'Deep squat'): shin 35 deg, knee 145, feet 28 deg out, trunk balanced."""
    if PQA:
        return viewer_pose("Deep squat")
    z = ankle_z("_l")
    rot(rig, "spine_01", (1, 0, 0), 26); rot(rig, "spine_03", (1, 0, 0), 12)
    for s in ("_l", "_r"):
        aim(rig, "thigh" + s, (0.4 * side(s), -0.86, 0.32)); aim(rig, "calf" + s, (0.14 * side(s), 0.5, -0.86))
        keep_foot(s, (0, -1, -0.3))
        aim(rig, "upperarm" + s, (0.2 * side(s), -1, 0.0)); aim(rig, "lowerarm" + s, (0.0, -1, 0.15))
    plant_feet(z)


def P_lunge():
    """Half-way down the viewers' 'Lunge' (solver weight 0.55: the rear heel already up, toes flat)."""
    if PQA:
        return viewer_pose("Lunge", 0.55)
    z = ankle_z("_l")
    aim(rig, "thigh_l", (0.12, -0.85, -0.5)); aim(rig, "calf_l", (0.05, 0.08, -1)); aim(rig, "foot_l", (0.1, -1, -0.3))
    aim(rig, "thigh_r", (-0.08, 0.55, -0.83)); aim(rig, "calf_r", (-0.05, 0.93, -0.35)); aim(rig, "foot_r", (0, 0.2, -1))
    rot(rig, "spine_02", (0, 0, 1), -15)
    plant_feet(z)


def P_deep_lunge():
    """The viewers' 'Lunge' at the bottom: front shin 10 deg, rear knee 8 cm off the floor, rear heel up (MTP 50)."""
    if PQA:
        return viewer_pose("Lunge")
    z = ankle_z("_l")
    aim(rig, "thigh_l", (0.1, -1, -0.04)); aim(rig, "calf_l", (0.04, -0.1, -1)); keep_foot("_l", (0, -1, -0.3))
    aim(rig, "thigh_r", (-0.06, 0.5, -0.86)); aim(rig, "calf_r", (-0.03, 1, -0.1)); aim(rig, "foot_r", (0, 0.3, -1))
    plant_feet(z)


def P_kneel():
    """The viewers' 'Kneel': rear knee on the floor, rear toes tucked (MTP 58), front foot flat."""
    viewer_pose("Kneel")


def P_walk_heel_strike():
    """Walk clip, left heel strike (heel on the floor, toes 12 deg up; right foot heel-up before toe-off)."""
    viewer_pose(None, walk_t=0.0)


def P_walk_toe_off():
    """Walk clip, left toe-off (left heel up, MTP extension 40; right foot flat)."""
    viewer_pose(None, walk_t=0.6667)


def P_sitting():
    """Seated (viewers' 'Sit', on its stool): knees 90, shins vertical, feet flat."""
    if PQA:
        return viewer_pose("Sit")
    z = ankle_z("_l")
    for s in ("_l", "_r"):
        aim(rig, "thigh" + s, (0.08 * side(s), -1, 0.0)); aim(rig, "calf" + s, (0.03 * side(s), 0.02, -1))
        keep_foot(s, (0, -1, -0.35))
        aim(rig, "upperarm" + s, (0.12 * side(s), -0.35, -0.93)); aim(rig, "lowerarm" + s, (0.02 * side(s), -1, -0.25))
    plant_feet(z)


def P_high_kick():
    """Front kick (viewers' 'High kick'): standing knee 10, CoM over the standing foot, pelvis tilted back 8."""
    if PQA:
        return viewer_pose("High kick")
    rot(rig, "spine_01", (1, 0, 0), -8)
    aim(rig, "thigh_l", (0.1, -0.75, 0.65)); aim(rig, "calf_l", (0.08, -0.7, 0.71)); aim(rig, "foot_l", (0.05, -0.55, 0.83))
    for s in ("_l", "_r"):
        aim(rig, "upperarm" + s, (0.45 * side(s), -0.3, -0.84))


def P_torso_twist():
    for b in ("spine_01", "spine_02", "spine_03", "spine_04", "spine_05"):
        rot(rig, b, (0, 0, 1), 11)
    for b in ("neck_01", "neck_02"):
        rot(rig, b, (0, 0, 1), 12)
    rot(rig, "spine_04", (0, 1, 0), 8)


def P_fists():
    for s in ("_l", "_r"):
        aim(rig, "upperarm" + s, (0.3 * side(s), -0.6, -0.75)); rot(rig, "lowerarm" + s, (1, 0, 0), -60)
        if "fist" not in HAND_PRESETS:
            curl_fingers(s, 75)
    if "fist" in HAND_PRESETS:
        HP.apply(rig, "fist", side="both", kind=kind)


def P_hand(preset):
    """hand_<preset>: the fists arm pose with a HANDS grip preset (scripts/hand_poses.py)."""
    for s in ("_l", "_r"):
        aim(rig, "upperarm" + s, (0.3 * side(s), -0.6, -0.75)); rot(rig, "lowerarm" + s, (1, 0, 0), -60)
    HP.apply(rig, preset, side="both", kind=kind)


def P_head_turn():
    for b, a in (("neck_01", 25), ("neck_02", 25), ("head", 25)):
        rot(rig, b, (0, 0, 1), a)
    rot(rig, "neck_01", (1, 0, 0), -12); rot(rig, "head", (1, 0, 0), -18)


def P_head_down_tilt():
    rot(rig, "neck_01", (1, 0, 0), 25); rot(rig, "neck_02", (1, 0, 0), 15); rot(rig, "head", (1, 0, 0), 12)
    rot(rig, "head", (0, 1, 0), 18)


def P_shrug():
    for s in ("_l", "_r"):
        rot(rig, "clavicle" + s, (0, 1, 0), -22 * side(s))
        rot(rig, "clavicle" + s, (0, 0, 1), -10 * side(s))


def P_arm_roll():
    """Upper-arm roll (internal / external rotation) with the twist bone counter-rolling 50%."""
    for s, d in (("_l", 70), ("_r", -70)):
        aim(rig, "upperarm" + s, (side(s), -0.2, -0.2)); rot(rig, "lowerarm" + s, (1, 0, 0), -80)
        twist(rig, "upperarm" + s, d); twist(rig, "upperarm_twist_01" + s, -d * 0.5)


POSES = [("rest", P_rest), ("t_pose", P_t_pose), ("arms_down", P_arms_down), ("arms_45", P_arms_45), ("arms_up", P_arms_up), ("viewer_arms_up", P_viewer_arms_up),
         ("arms_up_180", P_arms_up_180), ("arm_forward_up", P_arm_forward_up), ("arm_behind_back", P_arm_behind_back),
         ("cross_body_reach", P_cross_body_reach), ("arms_forward", P_arms_forward),
         ("elbow_wrist", P_elbow_wrist), ("arm_roll", P_arm_roll), ("squat", P_squat), ("viewer_squat", P_viewer_squat),
         ("deep_squat", P_deep_squat), ("lunge", P_lunge), ("deep_lunge", P_deep_lunge), ("sitting", P_sitting),
         ("high_kick", P_high_kick), ("torso_twist", P_torso_twist), ("fists", P_fists), ("head_turn", P_head_turn),
         ("head_down_tilt", P_head_down_tilt), ("shrug", P_shrug)]
if PQA:
    POSES += [("kneel", P_kneel), ("walk_heel_strike", P_walk_heel_strike), ("walk_toe_off", P_walk_toe_off)]
if opts.get("hands") or any(n.startswith("hand_") for n in only):
    POSES += [("hand_" + p, (lambda pr: (lambda: P_hand(pr)))(p)) for p in HAND_PRESETS]

# clay look (Workbench): shapes and creases read clearly
sc = bpy.context.scene
sc.render.engine = 'BLENDER_WORKBENCH'
sc.display.shading.light = 'STUDIO'
sc.display.shading.color_type = 'SINGLE'
sc.display.shading.single_color = (0.72, 0.66, 0.6)
sc.display.shading.show_cavity = True
sc.display.shading.cavity_type = 'WORLD'
sc.display.shading.show_object_outline = False
sc.render.resolution_x, sc.render.resolution_y = 560, 760
sc.view_settings.view_transform = 'Standard'
for o in rig.children:
    if o.type == 'MESH' and o.name.endswith(("_hair",)):
        o.hide_render = True

VIEWS = {"front34": ((-3.4 * S, -5.4 * S, 1.35 * S), (0, 0, 0.95 * S)), "back34": ((3.6 * S, 5.2 * S, 1.5 * S), (0, 0, 0.95 * S))}
# detail cameras: (label, bone whose posed head is the target, camera offset from it in metres[, target offset])
ARMPIT = (-0.035, 0.0, -0.1)                   # armpit hollow relative to the shoulder joint
CROTCH = (0.0, -0.01, -0.1)                    # perineum relative to the pelvis joint
DETAIL = {
    "rest": [("shoulder", "upperarm_l", (0.25, -0.75, 0.15)), ("face", "head", (0.1, -0.6, 0.12))],
    "t_pose": [("shoulder_front", "upperarm_l", (0.1, -0.8, 0.1)), ("shoulder_back", "upperarm_l", (0.2, 0.8, 0.15))],
    "arms_up": [("shoulder_front", "upperarm_l", (0.35, -0.75, 0.0)), ("armpit_back", "upperarm_l", (0.45, 0.7, -0.05)),
                ("armpit_side", "upperarm_l", (0.75, 0.05, -0.12))],
    "arms_down": [("armpit_front", "upperarm_l", (0.35, -0.75, -0.05), ARMPIT), ("armpit_back", "upperarm_l", (0.35, 0.75, -0.05), ARMPIT)],
    "arms_45": [("shoulder_front", "upperarm_l", (0.3, -0.8, 0.05), ARMPIT), ("armpit_back34", "upperarm_l", (0.5, 0.58, -0.08), ARMPIT),
                ("neck_shoulders", "neck_01", (0.0, -0.8, 0.12), (0.0, 0.0, -0.08))],
    "viewer_arms_up": [("armpit_back34", "upperarm_l", (0.5, 0.58, -0.08), ARMPIT),
                       ("armpit_front34", "upperarm_l", (0.5, -0.58, -0.08), ARMPIT),
                       ("neck_shoulders", "neck_01", (0.0, -0.8, 0.12), (0.0, 0.0, -0.08)),
                       ("armpit_side", "upperarm_l", (0.75, 0.05, -0.12), ARMPIT)],
    "arms_up_180": [("armpit_back34", "upperarm_l", (0.5, 0.58, -0.08), ARMPIT),
                    ("armpit_front34", "upperarm_l", (0.5, -0.58, -0.08), ARMPIT),
                    ("neck_shoulders", "neck_01", (0.0, -0.8, 0.12), (0.0, 0.0, -0.08))],
    "arm_forward_up": [("shoulder_side", "upperarm_l", (0.8, -0.05, 0.0), ARMPIT),
                       ("shoulder_back", "upperarm_l", (0.35, 0.75, 0.1))],
    "arm_behind_back": [("shoulder_front", "upperarm_l", (0.45, -0.7, -0.05), ARMPIT),
                        ("shoulder_back", "upperarm_l", (0.3, 0.8, 0.05))],
    "cross_body_reach": [("shoulder_back", "upperarm_l", (0.45, 0.75, 0.1)),
                         ("shoulder_front_side", "upperarm_l", (0.7, -0.45, 0.05), ARMPIT),
                         ("chest_above", "spine_05", (0.35, -0.65, 0.45), (0.1, 0.0, 0.05))],
    "arms_forward": [("shoulder_side", "upperarm_l", (0.8, -0.2, 0.15)), ("shoulder_back", "upperarm_l", (0.4, 0.75, 0.25))],
    "elbow_wrist": [("elbow", "lowerarm_l", (0.55, 0.45, 0.05)), ("wrist_twist", "hand_l", (0.45, -0.5, 0.05))],
    "arm_roll": [("upperarm_roll", "upperarm_l", (0.15, -0.85, 0.15)), ("upperarm_roll_back", "upperarm_l", (0.15, 0.85, 0.2))],
    "squat": [("hip_knee_side", "thigh_l", (0.95, -0.35, 0.1)), ("glutes_back", "pelvis", (0.35, 0.95, 0.05)),
              ("crotch_front_low", "pelvis", (0.0, -0.75, -0.45)), ("feet_side", "foot_l", (0.5, -0.15, 0.03))],
    "viewer_squat": [("groin_below_front", "pelvis", (0.0, -0.62, -0.42), CROTCH),
                     ("groin_front", "pelvis", (0.18, -0.85, -0.05), CROTCH),
                     ("glutes_below_back", "pelvis", (0.1, 0.7, -0.4), CROTCH)],
    "deep_squat": [("groin_below_front", "pelvis", (0.0, -0.62, -0.42), CROTCH),
                   ("hip_knee_side", "thigh_l", (0.95, -0.3, 0.05)), ("glutes_back", "pelvis", (0.3, 0.9, -0.1)),
                   ("feet_side", "foot_l", (0.5, -0.15, 0.03))],
    "lunge": [("hip_front", "pelvis", (0.35, -0.95, -0.1)), ("knee_back", "calf_r", (0.6, 0.7, 0.05)),
              ("rear_toes", "ball_r", (-0.4, 0.12, 0.05))],
    "deep_lunge": [("groin_front", "pelvis", (0.3, -0.8, -0.2), CROTCH), ("hip_side", "pelvis", (0.9, -0.1, -0.05), CROTCH),
                   ("rear_toes", "ball_r", (-0.4, 0.12, 0.05)), ("rear_toes_medial", "ball_r", (0.35, 0.05, 0.04))],
    "kneel": [("knee_floor", "calf_r", (-0.5, -0.3, 0.05)), ("rear_toes", "ball_r", (-0.4, 0.12, 0.05)),
              ("groin_front", "pelvis", (0.3, -0.8, -0.2), CROTCH)],
    "walk_heel_strike": [("heel_strike", "foot_l", (0.45, -0.1, 0.02)), ("trail_toes", "ball_r", (-0.4, 0.1, 0.05))],
    "walk_toe_off": [("toe_off", "ball_l", (0.4, 0.1, 0.05)), ("stance_foot", "foot_r", (-0.45, -0.1, 0.02))],
    "sitting": [("groin_front", "pelvis", (0.25, -0.85, 0.05), CROTCH), ("hip_side", "thigh_l", (0.85, 0.2, 0.05))],
    "high_kick": [("groin_front", "pelvis", (0.35, -0.8, -0.25), CROTCH), ("glutes_back_side", "pelvis", (0.55, 0.75, -0.15), CROTCH)],
    "torso_twist": [("waist", "spine_02", (0.1, -0.95, 0.1)), ("back", "spine_03", (0.1, 0.95, 0.1))],
    "fists": [("hand_l", "hand_l", (0.25, -0.22, 0.05)), ("hand_l_back", "hand_l", (-0.05, 0.3, 0.15))],
    "head_turn": [("neck_front", "neck_01", (0.1, -0.7, 0.1)), ("neck_side", "neck_01", (-0.65, -0.3, 0.1))],
    "head_down_tilt": [("neck_back", "neck_01", (0.15, 0.7, 0.2)), ("neck_front", "neck_01", (0.3, -0.65, 0.05))],
    "shrug": [("shoulders_front", "neck_01", (0.1, -0.9, 0.0)), ("shoulders_back", "neck_01", (0.1, 0.9, 0.05))],
}
for _p in HAND_PRESETS:
    DETAIL["hand_" + _p] = DETAIL["fists"]


# ---- metric regions (rest pose): (centre bone for the volume, sphere centre, radius, half-space) ----
def rest_head(b):
    return np.array(rig.data.bones[b].head_local)


pel = rest_head("pelvis"); th_l = rest_head("thigh_l")
CROTCH_REST = np.array([0.0, pel[1] - 0.01 * S, th_l[2] - 0.085 * S])
REGIONS = {}
for s in ("_l", "_r"):
    REGIONS["shoulder" + s] = ("upperarm" + s, rest_head("upperarm" + s), 0.17 * S, None)
    REGIONS["groin" + s] = ("thigh" + s, CROTCH_REST, 0.13 * S, side(s))
    REGIONS["hip" + s] = ("thigh" + s, rest_head("thigh" + s), 0.17 * S, None)
    REGIONS["elbow" + s] = ("lowerarm" + s, rest_head("lowerarm" + s), 0.08 * S, None)
    REGIONS["knee" + s] = ("calf" + s, rest_head("calf" + s), 0.10 * S, None)
REGIONS["neck"] = ("neck_01", rest_head("neck_01"), 0.11 * S, None)

topo = DM.Topo(body) if not opts.get("nometrics") else None
masks = {}
if topo is not None:
    Wb, bnames = DM.weight_matrix(body, rig)
    Wb /= np.maximum(Wb.sum(1, keepdims=True), 1e-12)
    for rn, (cb, c, r, half) in REGIONS.items():
        m = DM.region_mask(topo, c, r)
        if half is not None:
            m &= (topo.rest[:, 0] * half) >= -1e-6
        masks[rn] = m

os.makedirs(outdir, exist_ok=True)
metrics = {}
for name, fn in POSES:
    if only and name not in only:
        continue
    pose_reset(rig)
    fn()
    cw = legacy_correctives(not opts.get("nocor")) if LEGACY else drive_correctives(rig, body, enabled=USE_COR)
    if topo is not None:
        X = DM.evaluated_co(body)
        M = DM.skin_matrices(rig)
        A, _ = DM.blended_linear(Wb, M)
        metrics[name] = {"correctives": {k: round(v, 3) for k, v in cw.items() if v > 0.001}}
        for rn, (cb, c, r, half) in REGIONS.items():
            cpose = np.array(rig.pose.bones[cb].head)
            metrics[name][rn] = DM.measure(topo, X, masks[rn], rest_head(cb), cpose, A=A)
        worst = max(((rn, m["fold_max"]) for rn, m in metrics[name].items() if rn != "correctives"), key=lambda t: t[1])
        log("metrics", name, "worst fold %s %.1f" % worst,
            "isect", sum(m["isect"] for rn, m in metrics[name].items() if rn != "correctives"))
    if opts.get("norender"):
        continue
    for vn, (loc, tgt) in VIEWS.items():
        camera(loc, tgt, lens=50)
        render(os.path.join(outdir, "deform_%s_%s%s_%s.png" % (kind, TAG, name, vn)))
    sc.render.resolution_x, sc.render.resolution_y = 640, 640
    for det in DETAIL[name]:
        label, bone, off = det[:3]
        toff = det[3] if len(det) > 3 else (0, 0, 0)
        t = rig.matrix_world @ rig.pose.bones[bone].head + Vector(toff) * S
        camera(tuple(t[i] + off[i] * S for i in range(3)), tuple(t), lens=50)
        render(os.path.join(outdir, "deform_%s_%s%s_z%s.png" % (kind, TAG, name, label)))
    sc.render.resolution_x, sc.render.resolution_y = 560, 760
pose_reset(rig)
legacy_correctives(False) if LEGACY else drive_correctives(rig, body, enabled=False)
if metrics:
    mp = os.path.join(outdir, "deform_%s_%smetrics.json" % (kind, TAG))
    old = json.load(open(mp)) if os.path.exists(mp) and only else {}
    old.update(metrics)
    json.dump(old, open(mp, "w"), indent=1)
    log("metrics ->", mp)
