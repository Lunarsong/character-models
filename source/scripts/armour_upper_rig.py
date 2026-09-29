"""Extra joints of the knight's UPPER armour (iteration 2) and the hand poses that go with the gauntlets.

Helper joints (deform, weighted 100 % by the plates, rest rotation = a canonical rts_human frame so they are identical
on every body, M15):
  upperarm_helper_01..03_<s>   parent clavicle_<s>, head = the shoulder joint, rest frame = upperarm_<s>'s.
                               SWING-ONLY slerp toward the upper arm by SHOULDER_STEPS (0.30 / 0.55 / 0.80): the
                               pauldron cop rides helper 01, the lames helper 02 / 03 / upperarm (stepped lames, C1).
  lowerarm_helper_<s>          parent upperarm_<s>, head = the elbow, rest frame = lowerarm_<s>'s; half the elbow swing
                               (the couter cop, M1 / user item 17).
  plume_01..03                 parent head: the horsehair plume's spine (spring chain; armour_upper builds them).
Rule for engines (also in the armature extras 'rts_helpers'):
  q_helper_local = slerp(identity, swing(target), w)
  swing(target) = the minimal rotation taking the target bone's REST +Y (in the helper's parent space) to its POSED +Y
  (twist about the target's own axis is ignored), w = the step weight. Evaluate after the target is posed.
In Blender every helper carries a Damped Track constraint (target = the bone's tail, influence = w), which is exactly
that rule; the glTF exporter samples it into every clip (export_force_sampling).

Hand poses: solve_grip() closes a hand around a cylinder (the sword grip at socket_weapon_r) phalanx by phalanx
until each segment touches the gauntleted grip surface; apply_hand_pose() writes the result into every clip.
"""
import bpy, json, math, os
import numpy as np
from mathutils import Vector, Matrix, Quaternion

SIDES = ("l", "r")
SHOULDER_STEPS = (0.30, 0.55, 0.80)
ELBOW_STEP = 0.5
FINGERS = ("index", "middle", "ring", "pinky")


def helper_specs():
    out = {}
    for s in SIDES:
        for k, w in enumerate(SHOULDER_STEPS, 1):
            out["upperarm_helper_%02d_%s" % (k, s)] = dict(parent="clavicle_" + s, frame="upperarm_" + s,
                                                           target="upperarm_" + s, w=w, length=0.09)
        out["lowerarm_helper_" + s] = dict(parent="upperarm_" + s, frame="lowerarm_" + s, target="lowerarm_" + s,
                                           w=ELBOW_STEP, length=0.07)
    return out


def _edit(rig, fn):
    for o in bpy.context.view_layer.objects:
        o.select_set(False)
    bpy.context.view_layer.objects.active = rig
    rig.select_set(True)
    bpy.ops.object.mode_set(mode='EDIT')
    try:
        return fn(rig.data.edit_bones)
    finally:
        bpy.ops.object.mode_set(mode='OBJECT')


def ensure_helpers(rig):
    """Create / refit the helper joints and their Damped Track constraints; idempotent. Returns the helper names."""
    specs = helper_specs()

    def f(eb):
        for n, sp in specs.items():
            b = eb.get(n) or eb.new(n)
            src = eb[sp["frame"]]
            # head = the joint, rest rotation = the canonical frame of `frame` (a new edit bone has zero length, and
            # the matrix setter keeps the current length: give it the direction first, then the exact matrix)
            b.head = src.head.copy()
            b.tail = src.head + (src.tail - src.head).normalized() * sp["length"]
            b.roll = src.roll
            b.matrix = src.matrix.copy()
            b.length = sp["length"]
            b.parent = eb[sp["parent"]]
            b.use_connect = False
            b.use_deform = True
    _edit(rig, f)
    for n, sp in specs.items():
        pb = rig.pose.bones[n]
        pb.rotation_mode = 'QUATERNION'
        c = pb.constraints.get("rts_helper") or pb.constraints.new('DAMPED_TRACK')
        c.name = "rts_helper"
        c.target = rig
        c.subtarget = sp["target"]
        c.head_tail = 1.0
        c.track_axis = 'TRACK_Y'
        c.influence = sp["w"]
    rules = {n: dict(parent=sp["parent"], target=sp["target"], weight=sp["w"], type="swing_slerp") for n, sp in specs.items()}
    old = json.loads(rig.get("rts_helpers", "{}")).get("rules", {})
    old.update(rules)
    rig["rts_helpers"] = json.dumps({
        "doc": "helper joints: local rotation = slerp(identity, swing(target), weight); swing = minimal rotation of "
               "the target bone's rest +Y (in the helper's parent space) to its posed +Y (twist ignored). Evaluate "
               "after the target. plume_*: spring chain (see 'springs').",
        "rules": old, "springs": json.loads(rig.get("rts_helpers", "{}")).get("springs", {})})
    bpy.context.view_layer.update()
    return list(specs)


def add_springs(rig, chain, stiffness=0.35, damping=0.25, gravity=0.4, radius=0.03):
    d = json.loads(rig.get("rts_helpers", "{}") or "{}")
    d.setdefault("rules", {})
    d.setdefault("springs", {})
    d["springs"]["plume"] = dict(bones=list(chain), stiffness=stiffness, damping=damping, gravity=gravity,
                                 collider_radius=radius, doc="spring-bone chain (engine: VRM-style spring bones or a "
                                 "damped pendulum per bone); baked into the knight clips by armour_upper.post_clips")
    d.setdefault("doc", "")
    rig["rts_helpers"] = json.dumps(d)


def helper_check(rig):
    """At rest every helper must be identity (Damped Track already aligned): max angle (deg)."""
    from chr_lib import pose_reset
    pose_reset(rig)
    worst = 0.0
    for n in helper_specs():
        pb = rig.pose.bones.get(n)
        if pb is None:
            continue
        R = rig.data.bones[n].matrix_local.to_3x3().inverted() @ pb.matrix.to_3x3()
        a = math.degrees(R.to_quaternion().angle)
        worst = max(worst, a)
    return worst


# ============================================================================================== hand poses
def _seg_dist(p0, p1, a, d):
    """distance between segment p0-p1 and the infinite line a + t d (numpy)"""
    best = 1e9
    for t in np.linspace(0, 1, 9):
        p = p0 + (p1 - p0) * t
        v = p - a
        best = min(best, float(np.linalg.norm(v - np.dot(v, d) * d)))
    return best


def solve_grip(rig, side, centre, axis, radius, finger_r=0.0125, amounts=(1.35, 1.35, 1.2), thumb=True, gap=0.0005):
    """Local finger rotations (bone -> Quaternion, relative to rest) that close hand `side` around the cylinder
    (centre, axis, radius) in the CURRENT pose of the hand (armature space): every joint curls toward the palm until its
    segment (radius finger_r = finger + glove + scale) touches the cylinder, proximal joints first (a wrapping fist)."""
    B = rig.data.bones
    rig_M = {n: B[n].matrix_local.copy() for n in B.keys()}
    ph = rig.pose.bones
    H = lambda n: np.array((rig.matrix_world.inverted() @ rig.matrix_world @ ph[n].head)[:])
    c = np.array(centre); d = np.array(axis) / np.linalg.norm(axis)
    out = {}
    hand = ph["hand_" + side]
    kn = np.mean([np.array(ph[f + "_01_" + side].head[:]) for f in FINGERS], 0)
    wr = np.array(hand.head[:])
    for f in FINGERS + (("thumb",) if thumb else ()):
        chain = ["%s_%02d_%s" % (f, i, side) for i in (1, 2, 3)]
        if not all(n in ph for n in chain):
            continue
        for k, n in enumerate(chain):
            pb = ph[n]
            pb.rotation_mode = 'QUATERNION'
            # curl axis: perpendicular to the bone and the direction to the grip axis (rotate the tip toward it)
            h = np.array(pb.head[:]); t = np.array(pb.tail[:])
            v = h - c; foot = c + np.dot(v, d) * d
            to_axis = foot - h
            bd = (t - h) / max(np.linalg.norm(t - h), 1e-9)
            ax = np.cross(bd, to_axis)
            if np.linalg.norm(ax) < 1e-6:
                continue
            ax /= np.linalg.norm(ax)
            M = pb.matrix.to_3x3()
            ax_l = Vector((M.transposed() @ Vector(ax)))      # armature axis -> bone-local axis (M orthonormal)
            if f != "thumb":
                # fingers flex about their own local X (a hinge): pick the sense that brings the tip toward the grip
                base0 = pb.rotation_quaternion.copy()
                dists = []
                for sg_ in (1.0, -1.0):
                    pb.rotation_quaternion = base0 @ Quaternion(Vector((sg_, 0, 0)), math.radians(15))
                    bpy.context.view_layer.update()
                    tp = np.array(pb.tail[:]); vv = tp - c
                    dists.append(float(np.linalg.norm(vv - np.dot(vv, d) * d)))
                pb.rotation_quaternion = base0
                bpy.context.view_layer.update()
                ax_l = Vector((1.0 if dists[0] <= dists[1] else -1.0, 0, 0))
            # anatomical flexion limits: MCP 95, PIP 105, DIP 80 deg; thumb 55 / 60 / 70
            hi_f = (95.0, 105.0, 80.0) if f != "thumb" else (55.0, 60.0, 70.0)
            lo, hi = 0.0, hi_f[k]
            base = pb.rotation_quaternion.copy()
            def touches(deg):
                # the joint closes until its phalanx's distal end lies on the gauntleted grip surface (a wrap: MCP,
                # then PIP, then DIP), or the distal part of the finger meets the grip earlier
                pb.rotation_quaternion = base @ Quaternion(ax_l, math.radians(deg))
                bpy.context.view_layer.update()
                tp = np.array(pb.tail[:]); vv = tp - c
                if float(np.linalg.norm(vv - np.dot(vv, d) * d)) < radius + finger_r - gap:
                    return True
                for sn in chain[k + 1:]:
                    q = ph[sn]
                    if _seg_dist(np.array(q.head[:]), np.array(q.tail[:]), c, d) < radius + finger_r - gap - 0.002:
                        return True
                return False
            if touches(0.0):
                pb.rotation_quaternion = base
                bpy.context.view_layer.update()
                out[n] = base.copy()
                continue
            if not touches(hi):                   # never reaches the grip within the joint's range: full flexion
                lo = hi
            for _ in range(18 if lo < hi else 0):
                mid = 0.5 * (lo + hi)
                if touches(mid):
                    hi = mid
                else:
                    lo = mid
            pb.rotation_quaternion = base @ Quaternion(ax_l, math.radians(lo))
            bpy.context.view_layer.update()
            out[n] = pb.rotation_quaternion.copy()
    return out


def _mesh_weights(ob, bones):
    """per vertex: [(bone, w)] (normalised, only `bones` kept; others count as the hand) and the dominant bone"""
    names = {g.index: g.name for g in ob.vertex_groups}
    out, dom = [], []
    for v in ob.data.vertices:
        ws = [(names[g.group], g.weight) for g in v.groups if g.weight > 1e-4]
        tot = sum(w for _, w in ws) or 1.0
        ws = [(b, w / tot) for b, w in ws]
        out.append(ws)
        dom.append(max(ws, key=lambda x: x[1])[0] if ws else "")
    return out, np.array(dom, dtype=object)


def solve_grip_mesh(rig, side, centre, axis, radius, ob, clear=0.0006, thumb_gap=0.0015, span=(-0.075, 0.040),
                    report=None, obstacle=None):
    """Iteration 2b (user item 29, integrity G5): MESH-based fist. The gauntlet mesh `ob` (glove + finger scales, rest
    shape, its own skin weights) is posed by forward kinematics of the finger chains (numpy, no depsgraph); every finger
    joint flexes about its hinge (bone local X), MCP then PIP then DIP, until any point of that phalanx or the ones
    beyond it (vertices and face centres / edge midpoints, so no facet cuts the grip between vertices) reaches the grip
    cylinder (+ `clear`; only along the grip's length `span`, socket-frame metres): a wrap that touches the grip and
    never passes into it. The thumb is then searched (base swing + two flexions) to lie over the index finger's middle
    phalanx, clear of the grip and of the fingers by thumb_gap. The arm is at rest (pose_reset); returns
    {bone: local Quaternion} like solve_grip()."""
    B = rig.data.bones
    Mw = np.array(ob.matrix_world)
    V = np.array([v.co[:] for v in ob.data.vertices]) @ Mw[:3, :3].T + Mw[:3, 3]
    wts, dom = _mesh_weights(ob, None)
    bones = sorted({b for ws in wts for b, _ in ws if b in B})
    bi = {b: k for k, b in enumerate(bones)}
    nv = len(V)
    Wi = np.zeros((nv, 4), int); Ww = np.zeros((nv, 4))
    for v, ws in enumerate(wts):
        ws = sorted([x for x in ws if x[0] in bi], key=lambda x: -x[1])[:4]
        tot = sum(w for _, w in ws) or 1.0
        for k, (b, w) in enumerate(ws):
            Wi[v, k] = bi[b]; Ww[v, k] = w / tot
    polys = [list(p.vertices) for p in ob.data.polygons]
    c = np.asarray(centre, float); d = np.asarray(axis, float); d = d / np.linalg.norm(d)
    rest = {n: np.array(B[n].matrix_local) for n in B.keys()}
    Rinv = {n: np.linalg.inv(M) for n, M in rest.items()}
    posed = {}

    def pm(n):
        return posed.get(n, rest[n])

    def fk(chain, qs):
        out = {}
        par = B[chain[0]].parent.name
        Mp = pm(par)
        for n, q in zip(chain, qs):
            Q = np.eye(4); Q[:3, :3] = np.array(q.to_matrix())
            M = Mp @ (Rinv[par] @ rest[n]) @ Q
            out[n] = M; Mp = M; par = n
        return out

    def deforms(M):
        D = np.zeros((len(bones), 4, 4))
        for b, k in bi.items():
            D[k] = M.get(b, pm(b)) @ Rinv[b]
        return D

    def dist_axis(P):
        """distance to the grip surface (negative inside); +inf outside the grip's length. With `obstacle` (the prop's
        BVH at rest: grip, ferrules, guard, pommel) the signed distance to that surface instead"""
        v = P - c; al = v @ d
        r = np.linalg.norm(v - np.outer(al, d), axis=1) - radius
        out = np.where((al > span[0]) & (al < span[1]), r, np.inf)
        if obstacle is not None:
            # the prop's other parts (ferrules, guard, pommel): nearest-surface distance, signed by its normal (shallow
            # contacts only: the grip itself is the exact cylinder above)
            for k, p in enumerate(P):
                loc, nor, fi, dist = obstacle.find_nearest(Vector(p), 0.02)
                if loc is not None:
                    out[k] = min(out[k], float(np.dot(p - np.array(loc[:]), np.array(nor[:]))))
        return out

    part_cache = {}

    def part_of(vset):
        key = tuple(sorted(vset))
        if key not in part_cache:
            vs = np.array(key, int)
            loc = {v: k for k, v in enumerate(key)}
            fp = [[loc[v] for v in f] for f in polys if all(v in loc for v in f)]
            part_cache[key] = (vs, fp)
        return part_cache[key]

    def sample_pts(vset, M):
        vs, fp = part_of(vset)
        D = deforms(M)
        X = np.concatenate([V[vs], np.ones((len(vs), 1))], 1)
        P = np.zeros((len(vs), 3))
        for k in range(4):
            P += Ww[vs, k][:, None] * np.einsum("nij,nj->ni", D[Wi[vs, k]], X)[:, :3]
        if fp:
            cen = np.array([P[f].mean(0) for f in fp])
            mids = np.array([0.5 * (P[f[a]] + P[f[(a + 1) % len(f)]]) for f in fp for a in range(len(f))])
            P = np.concatenate([P, cen, mids])
        return P

    out = {}
    finger_pts = []
    for f in FINGERS:
        chain = ["%s_%02d_%s" % (f, i, side) for i in (1, 2, 3)]
        if not all(n in B for n in chain):
            continue
        part = [set(np.nonzero([str(x) in chain[k:] for x in dom])[0].tolist()) for k in range(3)]
        if not part[0]:
            continue
        # the flexion sense of every joint (bone local X hinge): the one that brings the fingertip toward the grip axis
        sgn = []
        for k in range(3):
            def tip_d(sg, k=k):
                q2 = [Quaternion() for _ in chain]; q2[k] = Quaternion(Vector((sg, 0, 0)), math.radians(15))
                M = fk(chain, q2)
                t = (M[chain[-1]] @ np.array([0, B[chain[-1]].length, 0, 1.0]))[:3]
                v = t - c
                return float(np.linalg.norm(v - np.dot(v, d) * d))
            sgn.append(1.0 if tip_d(1.0) <= tip_d(-1.0) else -1.0)
        LIM = (95.0, 105.0, 80.0)

        def qs_of(angles):
            return [Quaternion(Vector((sgn[k], 0, 0)), math.radians(angles[k])) for k in range(3)]

        def gap(angles, k0=0):
            return float(dist_axis(sample_pts(part[k0], fk(chain, qs_of(angles)))).min()) - clear

        def first_contact(fn, lo, hi, n=17):
            """largest t in [lo, hi] reached from lo without contact (march, then bisection; a finger that curls THROUGH
            the grip is clear again at full flexion, so the end is never tested first)"""
            prev = lo
            for x in np.linspace(lo, hi, n)[1:]:
                if fn(x) <= 0:
                    a_, b_ = prev, x
                    for _ in range(16):
                        m_ = 0.5 * (a_ + b_)
                        if fn(m_) > 0:
                            a_ = m_
                        else:
                            b_ = m_
                    return a_
                prev = x
            return hi
        ang = [0.0, 0.0, 0.0]
        if gap(ang) <= 0:
            # the finger already crosses the grip line at rest (the pinky of a diagonal grip): lift it by extending the
            # knuckle until it is clear, then wrap from there
            e = 0.0
            for e in np.linspace(0, -45, 19)[1:]:
                if gap([e, 0.0, 0.0]) > 0:
                    break
            ang[0] = e
        # 1. a coupled curl (MCP / PIP / DIP together, like a closing fist) until any part of the finger meets the grip
        base0 = list(ang)
        if os.environ.get("RTS_GRIP_DEBUG"):
            print("GRIPDBG %s sgn %s part %s rest-gap %.4f curl-gaps %s" % (
                f, sgn, [len(x) for x in part], gap(ang), [round(gap([base0[0] + x * (LIM[0] - base0[0]), x * LIM[1], x * LIM[2]]), 4)
                                                     for x in (0.2, 0.4, 0.6, 0.8, 1.0)]))
        cc = first_contact(lambda x: gap([base0[0] + x * (LIM[0] - base0[0]), x * LIM[1], x * LIM[2]]), 0.0, 1.0)
        ang = [base0[0] + cc * (LIM[0] - base0[0]), cc * LIM[1], cc * LIM[2]]
        # 2. the joints beyond the contact keep closing onto the grip one by one (PIP, then DIP), each until its part
        # of the finger touches: the middle and distal phalanges wrap the grip
        for k in (1, 2):
            if not part[k]:
                continue
            a0 = ang[k]
            ang[k] = first_contact(lambda x, k=k: gap([ang[j] if j != k else x for j in range(3)], k), a0, LIM[k], n=9)
        qs = qs_of(ang)
        M = fk(chain, qs)
        posed.update(M)
        for n, q in zip(chain, qs):
            out[n] = q
        finger_pts.append(sample_pts(part[0], M))
    # ---- thumb: over the index finger's middle phalanx (target = its back, 1 cm out from the grip axis side)
    chain = ["thumb_%02d_%s" % (i, side) for i in (1, 2, 3)]
    if all(n in B for n in chain) and ("index_02_" + side) in posed:
        from mathutils.kdtree import KDTree
        FP = np.concatenate(finger_pts)
        kd = KDTree(len(FP))
        for i_, p in enumerate(FP):
            kd.insert(Vector(p), i_)
        kd.balance()
        Mi = posed["index_02_" + side]
        mid = (Mi @ np.array([0, 0.5 * B["index_02_" + side].length, 0, 1.0]))[:3]
        # the thumb closes the fist's OPEN sector: the widest run of grip azimuths that no finger / palm point lies
        # within 12 mm of (at the index's height on the grip), its pad on the grip there (the G5 'enclosure')
        e1 = np.cross(d, [1.0, 0, 0]); e1 = e1 / max(np.linalg.norm(e1), 1e-9); e2 = np.cross(d, e1)
        HP = np.concatenate([FP, sample_pts(set(np.nonzero([str(x).startswith("hand") for x in dom])[0].tolist()), {})])
        v = HP - c; alv = v @ d; radv = v - np.outer(alv, d)
        gm = (np.linalg.norm(radv, axis=1) - radius < 0.012) & (alv > span[0]) & (alv < span[1])
        cov = np.zeros(36, bool)
        cov[np.floor((np.arctan2(radv[gm] @ e2, radv[gm] @ e1) + math.pi) / (2 * math.pi) * 36).astype(int) % 36] = True
        best_run = (0, 0)
        for st in range(36):
            n = 0
            while n < 36 and not cov[(st + n) % 36]:
                n += 1
            if n > best_run[0]:
                best_run = (n, st)
        phi = (best_run[1] + 0.5 * best_run[0]) / 36 * 2 * math.pi - math.pi
        alt = float(np.dot(mid - c, d))
        target = c + d * alt + (math.cos(phi) * e1 + math.sin(phi) * e2) * (radius + 0.006)
        tpart = set(np.nonzero([str(x) in chain for x in dom])[0].tolist())
        tip_part = set(np.nonzero([str(x) == chain[-1] for x in dom])[0].tolist())

        def qs_t(prm):
            bx, by, bz, c2, c3 = prm
            return [Quaternion(Vector((0, 0, 1)), math.radians(bz)) @ Quaternion(Vector((0, 1, 0)), math.radians(by)) @
                    Quaternion(Vector((1, 0, 0)), math.radians(bx)),
                    Quaternion(Vector((1, 0, 0)), math.radians(c2)), Quaternion(Vector((1, 0, 0)), math.radians(c3))]

        def cost(prm):
            qs = qs_t(prm)
            M = fk(chain, qs)
            P = sample_pts(tpart, M)
            pen = max(0.0, clear - float(dist_axis(P).min()))
            fd = min(kd.find(Vector(p))[2] for p in P[::3])
            pen += max(0.0, thumb_gap - fd)
            Pt = sample_pts(tip_part, M)
            tip = Pt.mean(0)
            # the thumb pad should touch (the grip or the index finger over it): a closed fist, not a thumb in the air
            tg = float(dist_axis(Pt).min())
            touch = min(tg, min(kd.find(Vector(p))[2] for p in Pt[::2]))
            # wrap: the pad on the grip (or on the index finger lying on it), then near the index's middle phalanx
            return 40.0 * pen + 0.3 * float(np.linalg.norm(tip - target)) + 3.0 * max(0.0, touch - 0.0015) + \
                0.5 * max(0.0, tg - 0.006), qs
        LO = np.array([-80.0, -40.0, -80.0, -80.0, -80.0]); HI = np.array([80.0, 40.0, 80.0, 80.0, 80.0])
        rng = np.random.default_rng(7)
        best = None
        for prm in [np.zeros(5)] + [LO + (HI - LO) * rng.random(5) for _ in range(1200)]:
            cv, qs = cost(prm)
            if best is None or cv < best[0]:
                best = (cv, tuple(prm), qs)
        prm = np.array(best[1]); stp = 10.0
        while stp > 0.6:
            imp = False
            for k in range(5):
                for sg in (1, -1):
                    q = prm.copy(); q[k] = float(np.clip(q[k] + sg * stp, LO[k], HI[k]))
                    cv, qs = cost(q)
                    if cv < best[0]:
                        best = (cv, tuple(q), qs); prm = q; imp = True
            if not imp:
                stp *= 0.5
        for n, q in zip(chain, best[2]):
            out[n] = q
        posed.update(fk(chain, best[2]))
    if report is not None:
        # every point of the posed gauntlet vs the grip (vertices + facet samples): the closest approach, and the
        # angular coverage of the grip by the hand within 12 mm of its surface (the integrity G5 'enclosure')
        allv = set(range(nv))
        P = sample_pts(allv, {})
        g = dist_axis(P)
        report["min_gap"] = float(g.min())
        v = P - c; al = v @ d; rad = v - np.outer(al, d)
        m = (al > span[0]) & (al < span[1]) & (g < 0.012)
        e1 = np.cross(d, [1.0, 0, 0]); e1 = e1 / max(np.linalg.norm(e1), 1e-9); e2 = np.cross(d, e1)
        ang = np.arctan2(rad[m] @ e2, rad[m] @ e1)
        report["enclosure"] = float(len(np.unique(np.floor((ang + math.pi) / (2 * math.pi) * 36).astype(int))) / 36.0) \
            if m.any() else 0.0
        worst = int(np.argmin(g))
        report["worst_at"] = P[worst].round(4).tolist()
    return out


def G_nrm(v):
    v = np.asarray(v, float)
    return v / max(np.linalg.norm(v), 1e-12)


def apply_hand_pose(rig, pose, actions):
    """Overwrite the finger rotation keys of the given actions with the constant local rotations `pose`."""
    n = 0
    if rig.animation_data is None:
        rig.animation_data_create()
    cur = rig.animation_data.action
    for act in actions:
        rig.animation_data.action = act            # Blender 5 layered actions: assign before fcurve_ensure_for_datablock
        f0, f1 = act.frame_range
        nf = int(round(f1 - f0)) + 1
        frames = np.arange(nf, dtype=float) + f0
        for bn, q in pose.items():
            rig.pose.bones[bn].rotation_mode = 'QUATERNION'
            for k in range(4):
                fc = act.fcurve_ensure_for_datablock(rig, 'pose.bones["%s"].rotation_quaternion' % bn, index=k, group_name=bn)
                fc.keyframe_points.clear()
                fc.keyframe_points.add(nf)
                co = np.empty(nf * 2); co[0::2] = frames; co[1::2] = q[k]
                fc.keyframe_points.foreach_set("co", co)
                fc.keyframe_points.foreach_set("interpolation", [1] * nf)
                fc.update()
            n += 1
    rig.animation_data.action = cur
    return n
