"""Clips for the civilian outfits (30 fps, in place), built with the knight's procedural pose engine (knight_anim.Engine:
two-bone IK, foot placement, twist bones, the 'grip' hand solve on a socket frame) with civilian arm work:
  idle   6 s loop: breathing, weight shift, look-around; peasant arms hang relaxed, the archer holds the longbow in the
         left hand (upper limb forward, lower tip clear of the floor)
  walk   1.1 s loop (1.15 m/s): the knight's gait (heel strike / roll / toe-off, pelvis bob / sway) + arm swing
  run    0.7 s loop (3.6 m/s): the knight's run gait + pumping arms (the archer's bow arm swings less)
  work   2.4 s loop (peasant): two-handed hoeing swing: hands up over the right shoulder, chop down in front, pull back
  shoot  3.6 s (archer): side-on stance, raise the bow at the target (bow arm straight, bow canted 8 deg), string hand
         to the string (solved on the posed bow), draw to the anchor under the right jaw along the arrow line, hold,
         loose (event 'release'), follow-through past the ear, recover; the string bone follows the drawing fingers,
         the limbs bend with the pull (outfit_civ_bow.bow_pose), the nocked arrow is shown from the nock to the loose
All hand targets are authored relative to the shoulders (scaled by the arm length), so they fit every body.
Every clip keys every core bone; bow bones are keyed in every archer clip.
"""
import bpy, math, json
import numpy as np
from mathutils import Vector, Matrix, Quaternion
import knight_anim as KA
from knight_anim import V, nrm, rx, ry, rz, R, smooth, smoother, lerp, FPS

SIDES = ("l", "r")


def arm_target(E, Cf, s, off):
    """wrist target: the shoulder + an offset authored for the male (rest chest frame), scaled by the arm length and
    carried by the posed chest"""
    sh = E.arm[s]["sh"]
    return Cf @ (sh + V(off) * E.arm_scale)


def hang(E, sp, Cf, s, sway=0.0, fwd=0.0, out=0.0, up=0.0, pole=None):
    sg = 1 if s == "l" else -1
    off = V(sg * (0.085 + out), -0.075 - fwd, -0.505 + up)
    sp["arms"][s] = dict(wrist=arm_target(E, Cf, s, off + V(0, -sway, 0)),
                         pole=pole or (Cf.to_3x3() @ nrm(V(0.35 * sg, 0.55, -0.3))),
                         hand=rx(-8) @ rz(-sg * 6))
    E.curl(sp, s, 0.28, 0.3)


def has_bow(E):
    return "bow_root" in E.B and "socket_hand_l" in E.sock


def bow_hold(E, sp, Cf, swing=0.0, lift=0.0):
    """left fist on the bow grip at the side, forearm bent, bow tilted 24 deg forward (lower tip ~0.25 m off the
    floor), the bow's back facing forward-out"""
    Cr = Cf.to_3x3()
    off = V(0.105, -0.13 - 0.05 * swing, -0.40 + lift + 0.02 * abs(swing))
    sp["arms"]["l"] = dict(grip=Cf @ (E.arm["l"]["sh"] + off * E.arm_scale),
                           blade=Cr @ nrm(V(0.06, -0.45 - 0.15 * swing, 1.0)),
                           knuckle=Cr @ nrm(V(0.45, -1.0, 0.0)), kmix=0.0, pole=Cr @ nrm(V(0.5, 0.6, -0.3)))
    E.curl(sp, "l", 0.92, 0.85)


def has_hoe(E):
    return "prop_hoe" in E.B and "socket_weapon_r" in E.sock


def hoe_hold(E, sp, Cf, swing=0.0, lift=0.0):
    """right fist on the hoe at the side, the haft upright (blade up, leaning a little forward): a staff carry"""
    Cr = Cf.to_3x3()
    off = V(-0.10, -0.11 - 0.05 * swing, -0.41 + lift + 0.02 * abs(swing))
    sp["arms"]["r"] = dict(grip=Cf @ (E.arm["r"]["sh"] + off * E.arm_scale),
                           blade=Cr @ nrm(V(-0.04, -0.22 - 0.12 * swing, 1.0)),
                           knuckle=Cr @ nrm(V(-0.45, -1.0, 0.0)), kmix=0.0, pole=Cr @ nrm(V(-0.5, 0.6, -0.3)))
    E.curl(sp, "r", 0.92, 0.85)


# ------------------------------------------------------------------------------------------------ idle / walk / run
def idle_spec(E, t, T=6.0):
    sp = KA.base_spec(E)
    w = 2 * math.pi * t / T
    br = math.sin(2 * math.pi * t / 3.0)
    shift = math.sin(w)
    sp["pelvis_loc"] = V(0.014 * shift, 0.003 * math.sin(2 * w), -0.024 + 0.002 * math.cos(2 * w))
    sp["rel"]["pelvis"] = rz(2.0 * math.sin(w + 0.6)) @ ry(-1.5 * shift) @ rx(1.5)
    sp["rel"]["spine_01"] = ry(0.7 * shift)
    sp["rel"]["spine_03"] = ry(0.6 * shift) @ rx(-0.6 * br)
    sp["rel"]["spine_04"] = rx(-1.0 * br) @ rz(-1.2 * math.sin(w + 0.6))
    sp["rel"]["spine_05"] = rx(-0.8 * br)
    for s in SIDES:
        sg = 1 if s == "l" else -1
        sp["rel"]["clavicle_" + s] = R((0, 1, 0), -sg * 1.3 * (br + 1) * 0.5)
    sp["rel"]["neck_02"] = rx(0.8 * br)
    look = math.sin(w + 1.3) * 6.0 + math.sin(w * 3 + 0.4) * 1.2
    sp["rel"]["head"] = rz(look) @ rx(1.0 + 1.2 * math.sin(w * 2 + 2.0))
    sp["legs"]["l"] = KA.foot_pose(E, "l", (0.0, -0.05), yaw=7)
    sp["legs"]["r"] = KA.foot_pose(E, "r", (0.01, 0.04), yaw=11)
    p0 = KA.partial_pose(E, sp); Cf = KA.pose_spine5(E, p0)
    for s in SIDES:
        hang(E, sp, Cf, s, sway=0.006 * math.sin(w + (0 if s == "l" else 2.0)), up=0.004 * br)
    if has_bow(E):
        bow_hold(E, sp, Cf, lift=0.004 * br)
    if has_hoe(E):
        hoe_hold(E, sp, Cf, lift=0.004 * br)
    return sp


def clip_idle(E, outfit):
    c = KA.Clip("idle", 180, True)
    return c, [idle_spec(E, f / FPS) for f in range(c.frames + 1)]


def clip_walk(E, outfit):
    n, speed = 33, 1.15
    c = KA.Clip("walk", n, True, speed=speed * E.leg_scale)
    out = []
    for sp, ph in KA.gait_specs(E, "walk", n, speed, stance=0.62, lift=0.085, pelvis_drop=0.045, bob=0.016, sway=0.018,
                                yaw=5.0, lean=3.0, arm_swing=1.0):
        p0 = KA.partial_pose(E, sp); Cf = KA.pose_spine5(E, p0)
        sw = math.cos(2 * math.pi * ph)          # + at the left heel strike: right arm forward, left arm back
        for s in SIDES:
            k = -sw if s == "l" else sw
            hang(E, sp, Cf, s, sway=0.0, fwd=0.11 * k, up=0.035 * abs(k) + 0.01 * k, out=0.012)
        if has_bow(E):
            bow_hold(E, sp, Cf, swing=-0.4 * sw)
        if has_hoe(E):
            hoe_hold(E, sp, Cf, swing=0.4 * sw)
        out.append(sp)
    return c, out


def clip_run(E, outfit):
    n, speed = 21, 3.6
    c = KA.Clip("run", n, True, speed=speed * E.leg_scale)
    out = []
    for sp, ph in KA.gait_specs(E, "run", n, speed, stance=0.34, lift=0.20, pelvis_drop=0.075, bob=0.028, sway=0.012,
                                yaw=8.0, lean=13.0, arm_swing=1.0, run=True):
        p0 = KA.partial_pose(E, sp); Cf = KA.pose_spine5(E, p0)
        sw = math.cos(2 * math.pi * ph)
        for s in SIDES:
            k = -sw if s == "l" else sw
            sg = 1 if s == "l" else -1
            off = V(sg * 0.10, -0.12 - 0.20 * k, -0.30 + 0.10 * k)            # elbows bent ~90 deg, pumping
            sp["arms"][s] = dict(wrist=arm_target(E, Cf, s, off), pole=Cf.to_3x3() @ nrm(V(0.3 * sg, 0.8, -0.2)),
                                 hand=rx(-5))
            E.curl(sp, s, 0.7, 0.6)
        if has_bow(E):
            bow_hold(E, sp, Cf, swing=-0.7 * sw, lift=0.06)
        if has_hoe(E):
            hoe_hold(E, sp, Cf, swing=0.7 * sw, lift=0.06)
        out.append(sp)
    return c, out


def clip_work(E, outfit):
    """two-handed hoe: hands together (right above left) swing from over the right shoulder down to the front; targets
    relative to the right shoulder (male offsets, scaled by the arm length)"""
    n = 72
    c = KA.Clip("work", n, True)
    out = []
    # haft grip points relative to the chest centre (between the shoulders), male metres (scaled by the arm length)
    hi, mid, lo = V(-0.10, -0.27, 0.03), V(-0.06, -0.35, -0.13), V(-0.02, -0.40, -0.30)
    for f in range(n + 1):
        t = f / n
        if t < 0.35:
            u = smooth(t / 0.35); a = lerp(0.0, 1.0, u)
        elif t < 0.5:
            u = smoother((t - 0.35) / 0.15); a = lerp(1.0, -0.6, u)
        else:
            u = smooth((t - 0.5) / 0.5); a = lerp(-0.6, 0.0, u)
        sp = KA.base_spec(E)
        bend = max(0.0, -a) * 22 + 6
        sp["pelvis_loc"] = V(0, 0.0, -0.035 - 0.03 * max(0.0, -a))
        sp["rel"]["spine_01"] = rx(bend * 0.4)
        sp["rel"]["spine_03"] = rx(bend * 0.35) @ rz(-8 * a)
        sp["rel"]["spine_04"] = rx(bend * 0.25) @ rz(-6 * a)
        sp["rel"]["head"] = rx(-bend * 0.3)
        sp["legs"]["l"] = KA.foot_pose(E, "l", (0.02, -0.16), yaw=10)
        sp["legs"]["r"] = KA.foot_pose(E, "r", (-0.01, 0.10), yaw=20)
        p0 = KA.partial_pose(E, sp); Cf = KA.pose_spine5(E, p0); Cr = Cf.to_3x3()
        g = hi.lerp(mid, 1 - a) if a >= 0 else mid.lerp(lo, -a / 0.6)
        axis = nrm(V(0.12, -0.45, -1.0).lerp(V(0.10, -0.75, -1.0), max(0.0, -a) / 0.6) if a < 0 else
                   V(0.12, -0.45, -1.0).lerp(V(0.16, -0.10, -1.0), a))
        cc = (E.arm["r"]["sh"] + E.arm["l"]["sh"]) * 0.5
        if "prop_hoe" in E.B:
            # the hoe: right fist near the butt (grip mode on socket_weapon_r, +Y along the haft to the blade), left
            # fist further down the haft
            import outfit_civ_props as PR
            gr = Cf @ (cc + g * E.arm_scale)
            hd = Cr @ axis
            sp["arms"]["r"] = dict(grip=gr, blade=hd, knuckle=Cr @ nrm(V(-0.3, -0.5, 0.4)), kmix=0.0,
                                   pole=Cr @ nrm(V(-0.4, 0.7, -0.4)))
            # overhand grips: the upper (right) thumb points down the haft, the lower (left) thumb up toward it
            sp["arms"]["l"] = dict(grip=gr + hd * (PR.LEFT_GRIP - 0.10 * max(0.0, -a) / 0.6) * E.arm_scale, blade=-hd,
                                   knuckle=Cr @ nrm(V(0.3, -0.6, 0.3)), kmix=0.0, pole=Cr @ nrm(V(0.4, 0.7, -0.4)))
            E.curl(sp, "r", 0.9, 0.85); E.curl(sp, "l", 0.9, 0.85)
        else:
            for s, dz in (("r", 0.0), ("l", -0.11)):
                p = cc + (g + axis * (-dz)) * E.arm_scale
                sg = 1 if s == "l" else -1
                sp["arms"][s] = dict(wrist=Cf @ p, pole=Cr @ nrm(V(0.4 * sg, 0.7, -0.4)), hand=rx(-10))
                E.curl(sp, s, 0.85, 0.8)
        out.append(sp)
    return c, out


# ------------------------------------------------------------------------------------------------ archery
def _head_point(E, pose, off):
    """a point fixed to the head: rest head position + off (male metres, scaled by the arm length) carried by the pose"""
    rest = E.rest["head"]
    p_rest = rest.translation + V(off) * E.arm_scale
    return pose["head"] @ (rest.inverted() @ p_rest)


def shoot_spec(E, t, turn=None):
    """one frame of the shot (before the string-hand target is resolved on the posed bow); turn: how far the torso
    has turned side-on (the nock happens with the chest still half open, the draw completes the turn)"""
    raise_ = smooth(min(1.0, t / 0.55))
    rec = smooth(min(1.0, max(0.0, (t - 2.85) / 0.7)))
    k = raise_ * (1 - rec)
    kt = k if turn is None else k * turn
    sp = KA.base_spec(E)
    # side-on stance: torso turned right (left shoulder to the target at -Y), head looking at the target
    sp["pelvis_loc"] = V(0.0, 0.0, -0.032 * k)
    # shoulders in line with the arrow at full draw: ~88 deg of turn from the pelvis up
    sp["rel"]["pelvis"] = rz(-46 * k)
    sp["rel"]["spine_03"] = rz(-15 * kt) @ ry(-2 * k)
    sp["rel"]["spine_04"] = rz(-15 * kt)
    sp["rel"]["spine_05"] = rz(-12 * kt) @ rx(-2 * k)
    sp["abs"]["head"] = rz(-8 * (1 - k)) @ rx(-3 * k) @ ry(5 * k)
    sp["legs"]["l"] = KA.foot_pose(E, "l", (0.04 * k, -0.12 * k - 0.05 * (1 - k)), yaw=7 - 52 * k)
    sp["legs"]["r"] = KA.foot_pose(E, "r", (0.01 - 0.02 * k, 0.04 + 0.14 * k), yaw=11 + 62 * k)
    return sp, k, rec


def clip_shoot(E, outfit):
    """raise, nock, draw, hold, loose, recover; per frame the string hand is placed on the posed bow's string (before
    the draw), on the arrow line to the anchor (the draw), then flies back past the ear (the loose)"""
    n = 108
    c = KA.Clip("shoot", n, False, events={"release": 70}, blend_out=16)
    import outfit_civ_bow as BOW
    specs = []
    c.string = []
    aim = nrm(V(0.0, -1.0, 0.035))                     # to the target: straight ahead, a touch upward
    L = (E.arm["l"]["L1"] + E.arm["l"]["L2"]) * 0.965
    t_nock, t_rel = 0.80, 70 / FPS
    anchor_off = V(-0.045, -0.035, -0.075)            # under the right jaw corner (male, rest)
    for f in range(n + 1):
        t = f / FPS
        if t < t_nock + 0.05:
            turn = 0.35
        elif t < t_rel:
            turn = lerp(0.35, 1.0, smoother(min(1.0, (t - t_nock - 0.05) / 0.85)))
        else:
            turn = 1.0
        sp, k, rec = shoot_spec(E, t, turn)
        p0 = KA.partial_pose(E, sp); Cf = KA.pose_spine5(E, p0); Cr = Cf.to_3x3()
        # the hold (idle) grip and the aim grip for the bow hand, blended by k
        hold = {"arms": {}, "local": {}}
        bow_hold(E, hold, Cf)
        pose_head = E.solve(sp)                         # head / chest posed (arms at rest) for the anchor
        A = _head_point(E, pose_head, anchor_off)
        S = pose_head["upperarm_l"].translation
        u = aim
        q = A - S
        b_ = u.dot(q)
        d = -b_ + math.sqrt(max(b_ * b_ - q.dot(q) + L * L, 0.0))
        # the bow is nocked close (arm bent, the string within reach of the right hand), pushed out while drawing
        if t < t_nock + 0.05:
            ext = 0.0
        elif t < t_rel:
            ext = smoother(min(1.0, (t - t_nock - 0.05) / 0.85))
        else:
            ext = 1.0
        G = A + u * lerp(min(0.40, d), d, ext)          # bow grip on the arrow line, bow arm straight at full draw
        blade_aim = nrm(V(0.14, 0.0, 1.0))               # canted 8 deg
        g0 = hold["arms"]["l"]
        sp["arms"]["l"] = dict(grip=g0["grip"].lerp(G, k), blade=nrm(g0["blade"].lerp(blade_aim, k)),
                               knuckle=nrm(g0["knuckle"].lerp(u, k)), kmix=0.0,
                               pole=Cr @ nrm(V(0.5, 0.6, -0.3)).lerp(V(0.3, 0.2, -1.0), k))
        E.curl(sp, "l", 0.92, 0.85)
        # string hand, provisional (resolved on the posed bow in pass 2)
        sp["arms"]["r"] = dict(wrist=arm_target(E, Cf, "r", V(-0.085, -0.075, -0.505)),
                               pole=Cr @ nrm(V(-0.35, 0.55, -0.3)), hand=rx(-8))
        pose1 = E.solve(sp)
        root = pose1["bow_root"]
        S0 = root @ Vector((0, 0, BOW.STRING_Z))       # the braced string's nocking point
        after = A - u * 0.14 + V(0.0, 0.0, 0.03) + Cr @ V(-0.05, 0.0, 0.0)
        idle_r = arm_target(E, Cf, "r", V(-0.085, -0.075, -0.505))
        held = t_nock <= t < t_rel
        if t < t_nock:
            a = smooth(max(0.0, (t - 0.30) / (t_nock - 0.30)))
            tgt = None if a <= 0 else S0
        elif t < t_rel:
            dr = smoother(min(1.0, (t - t_nock - 0.05) / 0.85)) if t > t_nock + 0.05 else 0.0
            tgt = S0.lerp(A, dr)
        else:
            a = smooth(min(1.0, (t - t_rel) / 0.22))
            tgt = A.lerp(after, a)
        if tgt is not None:
            blade_r = nrm(root.to_3x3() @ Vector((0, 1, 0)))
            knuck = nrm(root.to_3x3() @ Vector((0, 0, 1)))
            arm = dict(grip=tgt, blade=blade_r, knuckle=knuck, kmix=0.0, pole=Cr @ nrm(V(-0.9, 0.35, 0.1)))
            if t < t_nock:
                # reach: blend the wrist from hanging to the string (as a grip target lerp in world space)
                a = smooth(max(0.0, (t - 0.30) / (t_nock - 0.30)))
                arm["grip"] = (idle_r + Cr @ V(0.0, -0.04, 0.10)).lerp(S0, a)
            if rec > 0:
                arm["grip"] = arm["grip"].lerp(idle_r + Cr @ V(0.0, -0.04, 0.1), rec)
            sp["arms"]["r"] = arm
        E.curl(sp, "r", 0.55 if held else 0.35, 0.3)
        # string state: held (follows the fingers), free with a decaying vibration after the loose
        vib = 0.0
        if t >= t_rel:
            tt = t - t_rel
            vib = 0.018 * math.exp(-tt / 0.09) * math.sin(tt * 2 * math.pi * 11.0)
        c.string.append(dict(held=held, vib=vib, arrow=(t_nock - 0.08 <= t < t_rel)))
        specs.append(sp)
    return c, specs


CLIPS = {"idle": clip_idle, "walk": clip_walk, "run": clip_run, "work": clip_work, "shoot": clip_shoot}


def bow_frames(E, c, poses):
    """bow bones for every solved pose: string on the drawing fingers while held, braced (+ vibration) otherwise;
    returns the per-frame arrow visibility"""
    import outfit_civ_bow as BOW
    vis = []
    pulls = []
    for f, pose in enumerate(poses):
        st = c.string[f] if hasattr(c, "string") else dict(held=False, vib=0.0, arrow=False)
        sw = None
        if st["held"]:
            M = pose["socket_weapon_r"]
            # the string sits in the finger crease, a little toward the palm from the grip centre
            sw = M.translation + M.to_3x3() @ Vector((0.0, 0.0, -0.012))
        bp, pull = BOW.bow_pose(E, pose, string_world=sw, vib=st["vib"])
        pose.update(bp)
        if "bow_arrow" in E.rel_rest:
            s_ = pose["bow_string"]
            Ma = s_ @ E.rel_rest["bow_arrow"]
            # aim the arrow at the arrow rest
            tgt = pose["bow_socket_arrow"].translation
            y0 = nrm(Ma.to_3x3() @ Vector((0, 1, 0)))
            y1 = nrm(tgt - Ma.translation)
            Rq = y0.rotation_difference(y1).to_matrix()
            pose["bow_arrow"] = Matrix.Translation(Ma.translation) @ (Rq @ Ma.to_3x3()).to_4x4()
        vis.append(1.0 if st.get("arrow") else 0.001)
        pulls.append(pull)
    return vis, pulls


def key_arrow_scale(rig, act, vis):
    fr = np.arange(len(vis), dtype=float)
    pb = rig.pose.bones["bow_arrow"]
    for k in range(3):
        KA._fc(act, rig, 'pose.bones["bow_arrow"].scale', k, "bow_arrow", fr, np.array(vis))


def make_clips(rig, kind, outfit, names):
    sc = bpy.context.scene
    sc.render.fps = FPS; sc.render.fps_base = 1.0
    E = KA.Engine(rig)
    table = {}
    bow = has_bow(E)
    for name in names:
        E.last_miss = 0.0; E.prev_swivel = {}
        c, specs = CLIPS[name](E, outfit)
        E.prev_swivel = {}
        poses = [E.solve(s) for s in specs]
        if c.loop:
            poses[-1] = poses[0]
        vis = pulls = None
        if bow:
            vis, pulls = bow_frames(E, c, poses)
        if c.blend_out:
            E.prev_swivel = {}
            idle0 = E.solve(idle_spec(E, 0.0))
            if bow:
                bow_frames(E, KA.Clip("idle", 0, True), [idle0])
            nb = c.blend_out
            for i in range(nb):
                f = len(poses) - nb + i
                poses[f] = KA.blend_pose(E, poses[f], idle0, smoother((i + 1) / nb))
        act = KA.key_poses(E, rig, name, poses)
        if bow and "bow_arrow" in rig.pose.bones:
            key_arrow_scale(rig, act, vis)
        table[name] = dict(frames=len(poses) - 1, fps=FPS, loop=c.loop, speed_mps=round(c.speed, 3), events=c.events,
                           ik_miss_mm=round(E.last_miss * 1000, 1))
        if pulls:
            table[name]["max_pull_m"] = round(max(pulls), 3)
        print("CIV clip %-6s %3d frames loop=%s speed %.2f m/s IK miss %.1f mm%s" % (
            name, len(poses) - 1, c.loop, c.speed, E.last_miss * 1000,
            (" max pull %.2f m" % max(pulls)) if pulls else ""), flush=True)
    rig["rts_clips"] = json.dumps(table)
    rig.animation_data.action = bpy.data.actions.get("idle")
    return list(table)
