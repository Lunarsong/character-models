"""Stress poses for the civilian-outfit QA (Blender, armature space: faces -Y, +X = its left, +Z up).
The 25 pose recipes are the ones of scripts/deform_test.py (same names and numbers, copied here as functions of the
rig so the outfit QA does not execute that script), plus a few outfit poses: walk / run strides, kneel, bow draw.
Twist bones are driven like the engine: upper arm / thigh twist counter-roll 50 %, forearm twist 50 % of the hand."""
import math
from chr_lib import pose_reset, rot, aim, twist, bend
import bpy


def side(s):
    return 1 if s == "_l" else -1


def _ankle_z(rig, s):
    return rig.pose.bones["foot" + s].head.z


def plant_feet(rig, ref_z):
    low = min(_ankle_z(rig, "_l"), _ankle_z(rig, "_r"))
    pb = rig.pose.bones["pelvis"]
    M = pb.matrix.copy(); M.translation.z -= (low - ref_z)
    pb.matrix = M
    bpy.context.view_layer.update()


def keep_foot(rig, s, d=(0.0, -1.0, -0.35)):
    aim(rig, "foot" + s, (0.12 * side(s) * abs(d[1]), d[1], d[2]))


def curl_fingers(rig, s, deg, thumb=0.6):
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


def P_rest(r):
    pass


def P_t_pose(r):
    for s in ("_l", "_r"):
        aim(r, "upperarm" + s, (side(s), 0, 0)); aim(r, "lowerarm" + s, (side(s), 0, 0)); aim(r, "hand" + s, (side(s), 0, 0))


def P_arms_up(r):
    for s in ("_l", "_r"):
        rot(r, "clavicle" + s, (0, 1, 0), -18 * side(s))
        aim(r, "upperarm" + s, (0.22 * side(s), 0.05, 1)); aim(r, "lowerarm" + s, (0.12 * side(s), 0.05, 1))


def P_viewer_arms_up(r):
    for s in ("_l", "_r"):
        aim(r, "upperarm" + s, (0.22 * side(s), 0.05, 1)); aim(r, "lowerarm" + s, (0.12 * side(s), 0.05, 1))


def P_arms_down(r):
    for s in ("_l", "_r"):
        aim(r, "upperarm" + s, (0.12 * side(s), 0.02, -1)); aim(r, "lowerarm" + s, (0.05 * side(s), -0.12, -1))


def P_arms_45(r):
    for s in ("_l", "_r"):
        aim(r, "upperarm" + s, (0.7 * side(s), 0.02, 0.7)); aim(r, "lowerarm" + s, (0.7 * side(s), 0.02, 0.7))


def P_arms_up_180(r):
    for s in ("_l", "_r"):
        aim(r, "upperarm" + s, (0.04 * side(s), 0.03, 1)); aim(r, "lowerarm" + s, (0.0, 0.03, 1))


def P_arm_forward_up(r):
    for s in ("_l", "_r"):
        rot(r, "clavicle" + s, (0, 1, 0), -8 * side(s))
        aim(r, "upperarm" + s, (0.14 * side(s), -0.72, 0.68)); aim(r, "lowerarm" + s, (0.08 * side(s), -0.6, 0.8))


def P_arm_behind_back(r):
    for s in ("_l", "_r"):
        aim(r, "upperarm" + s, (0.3 * side(s), 0.42, -0.86)); aim(r, "lowerarm" + s, (-0.85 * side(s), 0.45, 0.28))


def P_cross_body_reach(r):
    rot(r, "clavicle_l", (0, 0, 1), 12)
    aim(r, "upperarm_l", (-0.5, -0.85, 0.12)); aim(r, "lowerarm_l", (-0.75, -0.6, 0.2))
    aim(r, "upperarm_r", (-0.12, 0.02, -1)); aim(r, "lowerarm_r", (-0.05, -0.12, -1))


def P_arms_forward(r):
    for s in ("_l", "_r"):
        rot(r, "clavicle" + s, (0, 0, 1), 10 * side(s))
        aim(r, "upperarm" + s, (0.12 * side(s), -1, 0.05)); aim(r, "lowerarm" + s, (0.05 * side(s), -1, 0.05))


def P_elbow_wrist(r):
    for s in ("_l", "_r"):
        aim(r, "upperarm" + s, (0.25 * side(s), -0.25, -1)); aim(r, "lowerarm" + s, (0.1 * side(s), -1, 0.0))
        bend(r, "lowerarm" + s, 0)
        rot(r, "lowerarm" + s, (1, 0, 0), -65)
    for s, d in (("_l", 80), ("_r", -80)):
        twist(r, "hand" + s, d); twist(r, "lowerarm_twist_01" + s, d * 0.5)


def P_squat(r):
    z = _ankle_z(r, "_l")
    rot(r, "spine_01", (1, 0, 0), 22); rot(r, "spine_03", (1, 0, 0), 10)
    for s in ("_l", "_r"):
        aim(r, "thigh" + s, (0.32 * side(s), -0.93, 0.12)); aim(r, "calf" + s, (0.12 * side(s), 0.38, -0.92))
        aim(r, "foot" + s, (0.12 * side(s), -1, -0.25))
        aim(r, "upperarm" + s, (0.15 * side(s), -1, -0.1)); aim(r, "lowerarm" + s, (0.0, -1, 0.1))
    plant_feet(r, z)


def P_viewer_squat(r):
    z = _ankle_z(r, "_l")
    for s in ("_l", "_r"):
        aim(r, "thigh" + s, (0.32 * side(s), -0.93, 0.12)); aim(r, "calf" + s, (0.12 * side(s), 0.38, -0.92))
        aim(r, "upperarm" + s, (0.15 * side(s), -1, -0.1)); aim(r, "lowerarm" + s, (0.0, -1, 0.1))
    pb = r.pose.bones["spine_01"]
    rot(r, "spine_01", (pb.matrix.col[0][0], pb.matrix.col[0][1], pb.matrix.col[0][2]), 20)
    plant_feet(r, z)


def P_deep_squat(r):
    z = _ankle_z(r, "_l")
    rot(r, "spine_01", (1, 0, 0), 26); rot(r, "spine_03", (1, 0, 0), 12)
    for s in ("_l", "_r"):
        aim(r, "thigh" + s, (0.4 * side(s), -0.86, 0.32)); aim(r, "calf" + s, (0.14 * side(s), 0.5, -0.86))
        keep_foot(r, s, (0, -1, -0.3))
        aim(r, "upperarm" + s, (0.2 * side(s), -1, 0.0)); aim(r, "lowerarm" + s, (0.0, -1, 0.15))
    plant_feet(r, z)


def P_lunge(r):
    z = _ankle_z(r, "_l")
    aim(r, "thigh_l", (0.12, -0.85, -0.5)); aim(r, "calf_l", (0.05, 0.08, -1)); aim(r, "foot_l", (0.1, -1, -0.3))
    aim(r, "thigh_r", (-0.08, 0.55, -0.83)); aim(r, "calf_r", (-0.05, 0.93, -0.35)); aim(r, "foot_r", (0, 0.2, -1))
    rot(r, "spine_02", (0, 0, 1), -15)
    plant_feet(r, z)


def P_deep_lunge(r):
    z = _ankle_z(r, "_l")
    aim(r, "thigh_l", (0.1, -1, -0.04)); aim(r, "calf_l", (0.04, -0.1, -1)); keep_foot(r, "_l", (0, -1, -0.3))
    aim(r, "thigh_r", (-0.06, 0.5, -0.86)); aim(r, "calf_r", (-0.03, 1, -0.1)); aim(r, "foot_r", (0, 0.3, -1))
    plant_feet(r, z)


def P_sitting(r):
    z = _ankle_z(r, "_l")
    for s in ("_l", "_r"):
        aim(r, "thigh" + s, (0.08 * side(s), -1, 0.0)); aim(r, "calf" + s, (0.03 * side(s), 0.02, -1))
        keep_foot(r, s, (0, -1, -0.35))
        aim(r, "upperarm" + s, (0.12 * side(s), -0.35, -0.93)); aim(r, "lowerarm" + s, (0.02 * side(s), -1, -0.25))
    plant_feet(r, z)


def P_high_kick(r):
    rot(r, "spine_01", (1, 0, 0), -8)
    aim(r, "thigh_l", (0.1, -0.75, 0.65)); aim(r, "calf_l", (0.08, -0.7, 0.71)); aim(r, "foot_l", (0.05, -0.55, 0.83))
    for s in ("_l", "_r"):
        aim(r, "upperarm" + s, (0.45 * side(s), -0.3, -0.84))


def P_torso_twist(r):
    for b in ("spine_01", "spine_02", "spine_03", "spine_04", "spine_05"):
        rot(r, b, (0, 0, 1), 11)
    for b in ("neck_01", "neck_02"):
        rot(r, b, (0, 0, 1), 12)
    rot(r, "spine_04", (0, 1, 0), 8)


def P_fists(r):
    for s in ("_l", "_r"):
        aim(r, "upperarm" + s, (0.3 * side(s), -0.6, -0.75)); rot(r, "lowerarm" + s, (1, 0, 0), -60)
        curl_fingers(r, s, 75)


def P_head_turn(r):
    for b, a in (("neck_01", 25), ("neck_02", 25), ("head", 25)):
        rot(r, b, (0, 0, 1), a)
    rot(r, "neck_01", (1, 0, 0), -12); rot(r, "head", (1, 0, 0), -18)


def P_head_down_tilt(r):
    rot(r, "neck_01", (1, 0, 0), 25); rot(r, "neck_02", (1, 0, 0), 15); rot(r, "head", (1, 0, 0), 12)
    rot(r, "head", (0, 1, 0), 18)


def P_shrug(r):
    for s in ("_l", "_r"):
        rot(r, "clavicle" + s, (0, 1, 0), -22 * side(s))
        rot(r, "clavicle" + s, (0, 0, 1), -10 * side(s))


def P_arm_roll(r):
    for s, d in (("_l", 70), ("_r", -70)):
        aim(r, "upperarm" + s, (side(s), -0.2, -0.2)); rot(r, "lowerarm" + s, (1, 0, 0), -80)
        twist(r, "upperarm" + s, d); twist(r, "upperarm_twist_01" + s, -d * 0.5)


# ------------------------------------------------------------------------------------------------ outfit poses
def P_walk(r):
    """mid stride: left thigh forward 28 deg, right back 16 deg, arms swinging opposite"""
    z = _ankle_z(r, "_r")
    aim(r, "thigh_l", (0.05, -0.47, -0.88)); aim(r, "calf_l", (0.03, -0.1, -1)); aim(r, "foot_l", (0.1, -1, -0.2))
    aim(r, "thigh_r", (-0.05, 0.28, -0.96)); aim(r, "calf_r", (-0.03, 0.45, -0.89)); aim(r, "foot_r", (-0.1, -0.6, -0.8))
    aim(r, "upperarm_l", (0.15, 0.25, -0.96)); aim(r, "lowerarm_l", (0.08, 0.05, -1))
    aim(r, "upperarm_r", (-0.15, -0.3, -0.94)); aim(r, "lowerarm_r", (-0.06, -0.45, -0.89))
    plant_feet(r, z)


def P_run(r):
    """run: high knee forward, trailing leg extended back, torso leaning, arms pumping"""
    z = _ankle_z(r, "_r")
    rot(r, "spine_01", (1, 0, 0), 12)
    aim(r, "thigh_l", (0.06, -0.85, -0.52)); aim(r, "calf_l", (0.03, 0.2, -0.98)); aim(r, "foot_l", (0.1, -1, -0.3))
    aim(r, "thigh_r", (-0.05, 0.42, -0.9)); aim(r, "calf_r", (-0.03, 0.85, -0.5)); aim(r, "foot_r", (0, 0.2, -1))
    aim(r, "upperarm_l", (0.15, 0.55, -0.82)); aim(r, "lowerarm_l", (0.1, -0.5, -0.86))
    aim(r, "upperarm_r", (-0.15, -0.5, -0.85)); aim(r, "lowerarm_r", (-0.1, -0.9, 0.4))
    plant_feet(r, z)


def P_kneel(r):
    """right knee on the ground, left foot planted forward"""
    z = _ankle_z(r, "_l")
    aim(r, "thigh_l", (0.12, -1, -0.1)); aim(r, "calf_l", (0.05, 0.05, -1)); keep_foot(r, "_l", (0, -1, -0.3))
    aim(r, "thigh_r", (-0.06, -0.1, -1)); aim(r, "calf_r", (-0.02, 1, -0.05)); aim(r, "foot_r", (0, 0.5, -0.86))
    plant_feet(r, z - 0.0)
    pb = r.pose.bones["pelvis"]
    kz = r.pose.bones["calf_r"].head.z
    M = pb.matrix.copy(); M.translation.z -= (kz - 0.06)
    pb.matrix = M
    bpy.context.view_layer.update()


def P_bow_draw(r):
    """longbow at full draw: left arm extended forward-left (bow hand at shoulder height), right hand anchored at the
    jaw, torso turned side-on"""
    for b in ("spine_03", "spine_04", "spine_05"):
        rot(r, b, (0, 0, 1), 12)
    rot(r, "neck_01", (0, 0, 1), -18); rot(r, "head", (0, 0, 1), -16)
    aim(r, "upperarm_l", (0.55, -0.83, 0.07)); aim(r, "lowerarm_l", (0.5, -0.86, 0.08))
    aim(r, "upperarm_r", (-0.95, -0.05, 0.28)); aim(r, "lowerarm_r", (0.85, -0.45, 0.25))
    curl_fingers(r, "_l", 70); curl_fingers(r, "_r", 40, thumb=0.2)


POSES = {"rest": P_rest, "t_pose": P_t_pose, "arms_down": P_arms_down, "arms_45": P_arms_45, "arms_up": P_arms_up,
         "viewer_arms_up": P_viewer_arms_up, "arms_up_180": P_arms_up_180, "arm_forward_up": P_arm_forward_up,
         "arm_behind_back": P_arm_behind_back, "cross_body_reach": P_cross_body_reach, "arms_forward": P_arms_forward,
         "elbow_wrist": P_elbow_wrist, "arm_roll": P_arm_roll, "squat": P_squat, "viewer_squat": P_viewer_squat,
         "deep_squat": P_deep_squat, "lunge": P_lunge, "deep_lunge": P_deep_lunge, "sitting": P_sitting,
         "high_kick": P_high_kick, "torso_twist": P_torso_twist, "fists": P_fists, "head_turn": P_head_turn,
         "head_down_tilt": P_head_down_tilt, "shrug": P_shrug,
         "walk": P_walk, "run": P_run, "kneel": P_kneel, "bow_draw": P_bow_draw}
STRESS = ["rest", "t_pose", "arms_down", "arms_45", "arms_up", "viewer_arms_up", "arms_up_180", "arm_forward_up",
          "arm_behind_back", "cross_body_reach", "arms_forward", "elbow_wrist", "arm_roll", "squat", "viewer_squat",
          "deep_squat", "lunge", "deep_lunge", "sitting", "high_kick", "torso_twist", "fists", "head_turn",
          "head_down_tilt", "shrug"]
EXTRA = ["walk", "run", "kneel", "bow_draw"]


def apply(rig, name):
    pose_reset(rig)
    POSES[name](rig)
    bpy.context.view_layer.update()
