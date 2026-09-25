"""HANDS (user round 6, items 37-38; POSES_HANDS_PLAN.md sections 6-8): anatomical hand poses and grip presets for the
shared 97-bone rts_human skeleton.

* Every finger joint flexes about its ANATOMICAL axis (perpendicular to the moving segment and the palm normal, carried
  by the parent bone), not the bone's local X (rolled 13-16 deg on the index: the old fist curled the index into the
  middle finger). Abduction turns about the palm normal. The thumb has its own frame: CMC flex (across the palm, about
  the palm normal), CMC palmar abduction, CMC opposition (pronation about the metacarpal), MCP / IP flexion toward the
  thumb pad. Angles are anatomical degrees INCLUDING the rest curl (flexion + palmar, 0 = straight in line with the
  parent segment); the solved result is stored as quaternions, so the rig's bone rolls no longer matter.
* Contact solve (numpy LBS of the hand, the same maths as Blender's armature modifier and three.js skinning; checked
  against Blender's evaluated mesh by `check`): fist fingers are clamped at the palm skin, grip fingers wrap the test prop
  link by link (MCP -> PIP -> DIP, each phalanx tangent to the prop), the thumb is placed by a bounded pattern search.
* Outputs: assets/hand_poses.json (per body, per side quaternions + prop frames + checks), viewer/hand_poses.js (the
  viewer module with the JSON inlined), out/hand_grips_<kind>.glb (skeleton-only clips), renders/poses_hands/hands/.

run (Blender 5.2 headless):
  Blender -b out/base_<kind>_export.blend --python-exit-code 1 -P scripts/hand_poses.py -- <kind> <cmd ...> [presets=a,b]
    solve   angles -> quaternions about the anatomical axes -> contact / ROM clamps -> merges this body into the JSON
    check   gates G-H1..H5, H7 -> renders/poses_hands/hands/hands_report_<kind>.json (exit 1 on a failure)
    render  close-ups of every preset with its prop (Workbench) + before / after fist shots
    bake    out/hand_grips_<kind>.glb
    js      viewer/hand_poses.js (also: python3 scripts/hand_poses.py js, no Blender needed)
importable (deform_test.py, other Blender scripts):
  import hand_poses; hand_poses.apply(rig, "fist", side="both", weight=1.0)
"""
import sys, os, json, math, time

import numpy as np

try:
    import bpy
    from mathutils import Vector, Matrix, Quaternion
    from mathutils.bvhtree import BVHTree
except ImportError:          # plain python3: only the `js` command works
    bpy = None

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
CH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JSON_PATH = os.path.join(CH, "assets", "hand_poses.json")
JS_PATH = os.path.join(CH, "viewer", "hand_poses.js")
RDIR = os.path.join(CH, "renders", "poses_hands", "hands")

FINGERS = ("index", "middle", "ring", "pinky")
FJ = ("MCP", "PIP", "DIP")
SIDES = ("_l", "_r")
BONES = (["thumb_01", "thumb_02", "thumb_03"] + ["%s_0%d" % (f, i) for f in FINGERS for i in (1, 2, 3)]
         + ["ring_metacarpal", "pinky_metacarpal"])
PALM_BONES = ("hand", "index_metacarpal", "middle_metacarpal", "ring_metacarpal", "pinky_metacarpal", "thumb_01")
# range of motion (AAOS / ASSH, plan section 2 / gate G-H3): anatomical degrees
ROM = {"MCP": (-5.0, 95.0), "PIP": (-5.0, 110.0), "DIP": (-5.0, 90.0), "abd": 20.0,
       "tMCP": (-5.0, 55.0), "tIP": (-5.0, 80.0), "ring_mc": (0.0, 8.0), "pinky_mc": (0.0, 20.0),
       "CMC_flex": (-25.0, 60.0), "CMC_abd": (-35.0, 60.0), "CMC_opp": (-20.0, 75.0)}


def log(*a):
    print("[hands]", *a, flush=True)


# ---------------------------------------------------------------------------------------------------- quaternion maths
# numpy quaternions are [w, x, y, z] (Blender order); the JSON stores glTF order [x, y, z, w]
def q_axis(axis, deg):
    a = np.asarray(axis, float); a = a / np.linalg.norm(a)
    h = math.radians(deg) / 2
    return np.array([math.cos(h), *(a * math.sin(h))])


def q_mul(a, b):
    w1, x1, y1, z1 = a; w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def q_mat(q):
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def mat_q(m):
    m = np.asarray(m, float)[:3, :3]
    t = np.trace(m)
    if t > 0:
        s = math.sqrt(t + 1.0) * 2; q = [0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s]
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2; q = [(m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s]
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2; q = [(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s]
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2; q = [(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s]
    q = np.array(q); q /= np.linalg.norm(q)
    return q if q[0] >= 0 else -q


def q_slerp(a, b, t):
    a = np.asarray(a, float); b = np.asarray(b, float)
    d = float(a @ b)
    if d < 0:
        b = -b; d = -d
    if d > 0.9995:
        r = a + t * (b - a); return r / np.linalg.norm(r)
    th = math.acos(min(1.0, d))
    return (math.sin((1 - t) * th) * a + math.sin(t * th) * b) / math.sin(th)


def to_gltf(q):
    return [round(float(q[1]), 7), round(float(q[2]), 7), round(float(q[3]), 7), round(float(q[0]), 7)]


def from_gltf(q):
    return np.array([q[3], q[0], q[1], q[2]], float)


def mirror_delta(q):
    """left delta -> right delta for the X-mirror-symmetric rig (bone local frames of _r = X-mirror of _l with the local
    X flipped): [w, x, y, z] -> [w, x, -y, -z] (glTF [x, y, z, w] -> [x, -y, -z, w])."""
    return np.array([q[0], q[1], -q[2], -q[3]])


def unit(v):
    v = np.asarray(v, float); return v / np.linalg.norm(v)


def signed_angle(a, b, axis):
    """angle from a to b about axis, after projecting both onto the plane perpendicular to the axis (degrees)."""
    ax = unit(axis)
    ap = a - (a @ ax) * ax; bp = b - (b @ ax) * ax
    return math.degrees(math.atan2(np.cross(ap, bp) @ ax, ap @ bp))


# ---------------------------------------------------------------------------------------------------- the hand model
class Hand:
    """One hand of one body at rest, in Blender armature space (Z up, metres), with numpy FK + LBS for its vertices."""

    def __init__(self, rig, body, side):
        self.rig, self.body, self.side = rig, body, side
        s = side
        B = rig.data.bones
        self.names = [b.name for b in B if b.name.endswith(s) and b.name[:-2] in ["hand"] + BONES + list(PALM_BONES)]
        self.names.sort(key=lambda n: len(B[n].parent_recursive))            # parents before children
        self.bi = {n: i for i, n in enumerate(self.names)}
        self.parent = {n: (B[n].parent.name if B[n].parent.name in self.bi else None) for n in self.names}
        self.rest = {n: np.array(B[n].matrix_local) for n in self.names}
        self.rest_inv = {n: np.linalg.inv(self.rest[n]) for n in self.names}
        self.head = {n: np.array(B[n].head_local) for n in self.names}
        self.tail = {n: np.array(B[n].tail_local) for n in self.names}
        # parent-relative rest (the glTF node rotation of every hand bone below 'hand')
        self.rest_local = {}
        for n in self.names:
            p = B[n].parent
            M = np.linalg.inv(np.array(p.matrix_local)) @ self.rest[n]
            self.rest_local[n] = mat_q(M)
        # ---- mesh subset: every vertex any hand bone moves
        me = body.data
        NV = len(me.vertices)
        co = np.empty(NV * 3); me.vertices.foreach_get("co", co); co = co.reshape(-1, 3)
        gname = {g.index: g.name for g in body.vertex_groups}
        allb = [b.name for b in B]; ai = {n: i for i, n in enumerate(allb)}
        W = np.zeros((NV, len(allb)))
        for v in me.vertices:
            for g in v.groups:
                nm = gname.get(g.group)
                if nm in ai:
                    W[v.index, ai[nm]] += g.weight
        hb = [ai[n] for n in self.names]
        sub = W[:, hb].sum(1) > 1e-6
        self.vidx = np.nonzero(sub)[0]
        self.co0 = co[self.vidx]
        self.W = W[self.vidx][:, hb]                                         # weights on the hand bones
        self.w_other = np.clip(1.0 - self.W.sum(1), 0, 1)                    # lowerarm / twist: stay at rest
        dom_all = np.array([allb[i] for i in W.argmax(1)])
        self.dom = dom_all[self.vidx]
        loc = {int(v): k for k, v in enumerate(self.vidx)}
        self.faces = [[loc[v] for v in p.vertices] for p in me.polygons if all(v in loc for v in p.vertices)]
        self.face_dom = [None] * len(self.faces)
        # ---- region sets (dominant bone, as the audit: renders/poses_hands/audit/tools/audit_bl.py)
        base = np.array([d[:-2] if d.endswith(s) else "" for d in self.dom])
        self.sets = {"palm": np.isin(base, list(PALM_BONES)), "palm_core": np.isin(base, [b for b in PALM_BONES if b != "thumb_01"])}
        for f in FINGERS:
            self.sets[f] = np.isin(base, ["%s_0%d" % (f, i) for i in (1, 2, 3)])
            for i in (1, 2, 3):
                self.sets["%s_0%d" % (f, i)] = base == "%s_0%d" % (f, i)
        self.sets["thumb"] = np.isin(base, ["thumb_02", "thumb_03"])
        for i in (1, 2, 3):
            self.sets["thumb_0%d" % i] = base == "thumb_0%d" % i
        self.set_faces = {}
        # ---- palm frame (rest): w wrist, d toward the middle MCP, r radial (toward the thumb), n palmar
        H = lambda b: self.head[b + s]
        self.w = H("hand")
        d = unit(H("middle_01") - self.w)
        k = H("pinky_01") - H("index_01")
        r = -k - (-k @ d) * d; r = unit(r)
        n = unit(np.cross(d, r))
        # palmar = the side the finger flesh sits on (the joints are dorsal: plan H2)
        m = self.sets["middle_01"] | self.sets["middle_02"]
        seg = unit(H("middle_02") - H("middle_01"))
        off = (self.co0[m] - H("middle_01")); off = off - np.outer(off @ seg, seg)
        if off.mean(0) @ n < 0:
            n = -n
        self.d, self.r, self.n = d, r, n
        self.mirror = (np.cross(d, r) @ n) < 0        # left-handed frame on the right hand
        self._axes()
        self.pivots = {}                               # {bone: p (bone-local offset of the rotation centre from the joint)}
        self.joint_depth = self._joint_depth()

    def _joint_depth(self):
        """per finger joint: depth under the dorsal skin and the palmar skin (ray casts on the rest mesh, as
        renders/poses_hands/audit/tools/pivots.py), mm."""
        me = self.body.data
        co = np.empty(len(me.vertices) * 3); me.vertices.foreach_get("co", co); co = co.reshape(-1, 3)
        tree = BVHTree.FromPolygons([Vector(v) for v in co], [list(p.vertices) for p in me.polygons], all_triangles=False)
        out = {}
        for f in FINGERS:
            for j, jn in enumerate(FJ, start=1):
                b = "%s_0%d%s" % (f, j, self.side)
                c = self.head[b]; ray = unit(self.tail[b] - c)
                nn = unit(self.n - (self.n @ ray) * ray)
                hd = tree.ray_cast(Vector(c - nn * 0.05), Vector(nn), 0.1)[3]
                hp = tree.ray_cast(Vector(c + nn * 0.05), Vector(-nn), 0.1)[3]
                out[b] = (0.05 - hd, 0.05 - hp, nn)
        return out

    def set_pivots(self, frac=0.45, cap_mcp=0.007, cap=0.005):
        """virtual pivots (plan H-4): move each finger joint's rotation centre palmar to `frac` of the local thickness
        (MCP capped: the palmar thickness there includes the distal palm pad)."""
        self.pivots = {}
        for b, (dd, dp, nn) in self.joint_depth.items():
            delta = frac * (dd + dp) - dd
            delta = max(0.0, min(cap_mcp if b.split("_")[1] == "01" else cap, delta))
            self.pivots[b] = self.rest[b][:3, :3].T @ (nn * delta)
        return {b: round(float(np.linalg.norm(v)) * 1000, 2) for b, v in self.pivots.items()}

    # ---- anatomical axes (armature rest space) and rest angles
    def seg_of(self, f, i):
        s = self.side
        if f == "thumb":
            pts = [self.head["thumb_01" + s], self.head["thumb_02" + s], self.head["thumb_03" + s], self.tail["thumb_03" + s]]
            return pts[i] - pts[i - 1]
        pts = [self.head["%s_metacarpal%s" % (f, s)]] + [self.head["%s_0%d%s" % (f, j, s)] for j in (1, 2, 3)] + [self.tail["%s_03%s" % (f, s)]]
        return pts[i + 1] - pts[i]            # i = 0 metacarpal, 1 P1, 2 P2, 3 P3

    def _axes(self):
        s, n, r = self.side, self.n, self.r
        A = {}; rest_ang = {}
        for f in FINGERS:
            for j, jn in enumerate(FJ, start=1):
                seg = self.seg_of(f, j)
                ax = unit(np.cross(seg, n))                       # + moves the segment palmar
                if np.cross(ax, seg) @ n < 0:
                    ax = -ax
                ab = n.copy()                                     # + away from the middle finger
                away = r if f in ("index", "middle") else -r
                if np.cross(ab, seg) @ away < 0:
                    ab = -ab
                A["%s_0%d" % (f, j)] = {"flex": ax, "abd": ab}
                rest_ang["%s.%s" % (f, jn)] = signed_angle(self.seg_of(f, j - 1), seg, ax)
            if f in ("ring", "pinky"):
                seg = self.seg_of(f, 0)
                ax = unit(np.cross(seg, n))
                if np.cross(ax, seg) @ n < 0:
                    ax = -ax
                A["%s_metacarpal" % f] = {"flex": ax}
        # thumb
        t1, t2, t3 = self.seg_of("thumb", 1), self.seg_of("thumb", 2), self.seg_of("thumb", 3)
        m = self.sets["thumb_02"]
        off = self.co0[m] - self.head["thumb_02" + s]; u2 = unit(t2); off = off - np.outer(off @ u2, u2)
        pad = unit(off.mean(0))                                   # the thumb pad side (flesh; the joints are dorsal)
        self.thumb_pad = pad
        fl = n.copy()
        if np.cross(fl, t1) @ (-r) < 0:                           # CMC flex: across the palm (ulnar)
            fl = -fl
        ab = unit(np.cross(t1, n))                                # CMC palmar abduction: away from the palm plane
        if np.cross(ab, t1) @ n < 0:
            ab = -ab
        op = unit(t1)                                             # CMC opposition: pronation (pad turns to face the fingers)
        pad1 = pad - (pad @ op) * op
        if np.cross(op, pad1) @ (-n) < 0:
            op = -op
        A["thumb_01"] = {"CMC_flex": fl, "CMC_abd": ab, "CMC_opp": op}
        for b, seg in (("thumb_02", t2), ("thumb_03", t3)):
            p = pad - (pad @ unit(seg)) * unit(seg)
            ax = unit(np.cross(seg, p))
            if np.cross(ax, seg) @ p < 0:
                ax = -ax
            A[b] = {"flex": ax}
        rest_ang["thumb.MCP"] = signed_angle(t1, t2, A["thumb_02"]["flex"])
        rest_ang["thumb.IP"] = signed_angle(t2, t3, A["thumb_03"]["flex"])
        self.axes = A
        self.rest_ang = rest_ang

    # ---- anatomical angles -> per-bone delta quaternions (bone-local, = Blender rotation_quaternion)
    def deltas(self, ang):
        """ang: {'index': {'MCP','PIP','DIP','abd'}, ..., 'ring_mc', 'pinky_mc', 'thumb': {'CMC_flex','CMC_abd',
        'CMC_opp','MCP','IP'}} (anatomical degrees; missing = rest). Returns {bone name (with side): q [w,x,y,z]}."""
        s = self.side
        out = {}

        def local(b, q_arm):          # armature-space rotation at rest -> bone-local basis rotation
            R = self.rest[b][:3, :3]
            return mat_q(R.T @ q_mat(q_arm) @ R)

        for f in FINGERS:
            a = ang.get(f, {})
            for j, jn in enumerate(FJ, start=1):
                b = "%s_0%d" % (f, j)
                fl = a.get(jn)
                q = np.array([1.0, 0, 0, 0])
                if j == 1 and a.get("abd"):
                    q = q_axis(self.axes[b]["abd"], a["abd"])
                if fl is not None:
                    q = q_mul(q_axis(self.axes[b]["flex"], fl - self.rest_ang["%s.%s" % (f, jn)]), q)
                out[b + s] = local(b + s, q)
        for f in ("ring", "pinky"):
            v = ang.get(f + "_mc", 0.0) or 0.0
            out["%s_metacarpal%s" % (f, s)] = local("%s_metacarpal%s" % (f, s), q_axis(self.axes["%s_metacarpal" % f]["flex"], v))
        t = ang.get("thumb", {})
        A = self.axes["thumb_01"]
        q = q_axis(A["CMC_opp"], t.get("CMC_opp", 0.0))
        q = q_mul(q_axis(A["CMC_abd"], t.get("CMC_abd", 0.0)), q)
        q = q_mul(q_axis(A["CMC_flex"], t.get("CMC_flex", 0.0)), q)
        out["thumb_01" + s] = local("thumb_01" + s, q)
        for b, key in (("thumb_02", "MCP"), ("thumb_03", "IP")):
            v = t.get(key)
            q = q_axis(self.axes[b]["flex"], v - self.rest_ang["thumb." + key]) if v is not None else np.array([1.0, 0, 0, 0])
            out[b + s] = local(b + s, q)
        return out

    # ---- FK and skinning
    def fk(self, deltas):
        P = {}
        for n in self.names:
            p = self.parent[n]
            B4 = np.eye(4)
            if n in deltas:
                R = q_mat(deltas[n]); B4[:3, :3] = R
                if n in self.pivots:
                    pv = self.pivots[n]; B4[:3, 3] = pv - R @ pv
            P[n] = (self.rest[n] if p is None else P[p] @ self.rest_inv[p] @ self.rest[n]) @ B4
        return P

    def locations(self, deltas):
        """pose-bone locations (bone-local) that make each joint rotate about its virtual pivot."""
        return {n: (self.pivots[n] - q_mat(q) @ self.pivots[n]) for n, q in deltas.items() if n in self.pivots}

    def skin(self, deltas, mask=None):
        P = self.fk(deltas)
        S = np.stack([P[n] @ self.rest_inv[n] for n in self.names])           # (nb, 4, 4)
        co = self.co0 if mask is None else self.co0[mask]
        W = self.W if mask is None else self.W[mask]
        wo = self.w_other if mask is None else self.w_other[mask]
        X = np.einsum("bij,vj->vbi", S[:, :3, :3], co) + S[None, :, :3, 3]    # (nv, nb, 3)
        return np.einsum("vb,vbi->vi", W, X) + wo[:, None] * co, P

    def faces_of(self, key):
        if key not in self.set_faces:
            m = self.sets[key]
            self.set_faces[key] = [f for f in self.faces if all(m[v] for v in f)]
        return self.set_faces[key]


# ---------------------------------------------------------------------------------------------------- distance queries
class RegionTree:
    """BVH of one body region (an OPEN surface) + which of its faces lie on the region's border band (faces touching a
    border edge). A nearest point on the border band says nothing about inside / outside."""

    def __init__(self, co, faces, border):
        self.tree = BVHTree.FromPolygons([Vector(v) for v in co], faces, all_triangles=False)
        self.border = border

    def find_nearest(self, p, maxd):
        return self.tree.find_nearest(p, maxd)


def border_faces(faces):
    from collections import Counter
    ec = Counter()
    for f in faces:
        for i in range(len(f)):
            ec[tuple(sorted((f[i], f[(i + 1) % len(f)])))] += 1
    bv = set(v for e, n in ec.items() if n == 1 for v in e)
    return np.array([any(v in bv for v in f) for f in faces], bool)


def bvh(co, faces, border=None):
    if not faces:
        return None
    if border is None:
        return BVHTree.FromPolygons([Vector(v) for v in co], faces, all_triangles=False)
    return RegionTree(co, faces, border)


def signed_dist(tree, pts, maxd=0.03, strict=0.8):
    """distance to the nearest face, negative when the point is inside (behind the face). A region of the body mesh is
    an OPEN surface, so a point beside its border (nearest point on a border edge) has a meaningless face-normal sign:
    'inside' is only trusted when the offset runs along the inward normal (|dot| >= strict * distance; a genuine
    penetration of a smooth surface has its nearest point inside a face, offset exactly along the normal). maxd when
    nothing is near."""
    out = np.full(len(pts), maxd)
    if tree is None:
        return out
    border = getattr(tree, "border", None)
    for k, p in enumerate(pts):
        loc, nrm, fi, dd = tree.find_nearest(Vector(p), maxd)
        if loc is not None:
            dot = (Vector(p) - loc).dot(nrm)
            inside = dot < 0 and -dot >= strict * dd and (border is None or not border[fi])
            out[k] = -dd if inside else dd
    return out


def tri_cross(co, fa, fb):
    """number of triangle pairs of face sets fa / fb that intersect and share no vertex."""
    if not fa or not fb:
        return 0
    V = [Vector(v) for v in co]
    ta = BVHTree.FromPolygons(V, fa, all_triangles=False); tb = BVHTree.FromPolygons(V, fb, all_triangles=False)
    return sum(1 for i, j in ta.overlap(tb) if not (set(fa[i]) & set(fb[j])))




# ---------------------------------------------------------------------------------------------------- presets
# Plan section 7 (left hand authored; right = mirror). Angles are anatomical start values (degrees, rest curl included);
# 'fingers' picks each finger's solve: 'palm' = fist clamp at the palm skin, 'wrap' = close MCP -> PIP -> DIP until each
# phalanx touches the prop, 'hold' = keep the start values (only a penetration clamp), 'rest_on' = wrap with a larger gap
# (fingers resting near the prop, not squeezing). 'thumb' picks the thumb target.
def F(mcp, pip, dip, abd=0.0):
    return {"MCP": mcp, "PIP": pip, "DIP": dip, "abd": abd}


def TH(f, a, o, mcp, ip):
    return {"CMC_flex": f, "CMC_abd": a, "CMC_opp": o, "MCP": mcp, "IP": ip}


FIST_ANGLES = {"index": F(85, 100, 70), "middle": F(88, 105, 72, 0), "ring": F(90, 105, 72, -3), "pinky": F(92, 100, 70, -5),
               "ring_mc": 4, "pinky_mc": 10, "thumb": TH(35, 15, 25, 50, 30)}
PRESETS = {
    "relaxed": {"label": "Relaxed", "group": "basic",
                "angles": {"index": F(20, 25, 10), "middle": F(25, 30, 12), "ring": F(30, 35, 15), "pinky": F(35, 40, 18),
                           "ring_mc": 0, "pinky_mc": 3, "thumb": TH(8, -12, 8, 15, 12)},
                "fingers": "hold", "thumb": "hold", "prop": None,
                "source": "resting cascade (flexion increasing toward the pinky); no contacts"},
    "fist": {"label": "Fist", "group": "basic", "angles": FIST_ANGLES, "fingers": "palm", "thumb": "fist", "prop": None,
             "source": "ASSH MCP 85 / PIP 110 / DIP 65 (TAM 260); pads on the palm skin; thumb over the index + middle P2"},
    "hammer": {"label": "Hammer", "group": "power",
               "angles": {"index": F(53, 78, 42), "middle": F(62, 78, 50), "ring": F(58, 81, 43, -2), "pinky": F(50, 64, 49, -4),
                          "ring_mc": 3, "pinky_mc": 8, "thumb": TH(40, 35, 30, 30, 25)},
               "fingers": "wrap", "thumb": "wrap", "prop": "hammer",
               "source": "power grip on a 34 mm cylinder (Shimawaki 2019 CT angles, interpolated 10 -> 60 mm); thumb closes over the index P2"},
    "sword": {"label": "Sword", "group": "weapon",
              "angles": {"index": F(35, 70, 40, 3), "middle": F(55, 80, 48), "ring": F(62, 82, 48, -2), "pinky": F(68, 78, 45, -4),
                         "ring_mc": 4, "pinky_mc": 10, "thumb": TH(40, 30, 30, 35, 30)},
              "fingers": {"index": "hook", "middle": "wrap", "ring": "wrap", "pinky": "wrap"}, "thumb": "wrap", "prop": "sword",
              "source": "one-hand handshake grip (est.): 29.2 mm round grip (= the knight's socket_weapon_r radius 14.6 mm) diagonal across the palm, index against the guard"},
    "knife": {"label": "Knife", "group": "weapon",
              "angles": {"index": F(60, 88, 50), "middle": F(68, 90, 52), "ring": F(70, 92, 52, -2), "pinky": F(68, 85, 50, -4),
                         "ring_mc": 4, "pinky_mc": 10, "thumb": TH(40, 30, 30, 35, 25)},
              "fingers": "wrap", "thumb": "wrap", "prop": "knife",
              "source": "forward knife grip on a 24 mm handle (Shimawaki trend for small diameters); blade leaves the thumb side, edge down"},
    "icepick": {"label": "Ice-pick", "group": "weapon",
                "angles": {"index": F(60, 88, 50), "middle": F(68, 90, 52), "ring": F(70, 92, 52, -2), "pinky": F(68, 85, 50, -4),
                           "ring_mc": 4, "pinky_mc": 10, "thumb": TH(20, 45, 10, 10, 5)},
                "fingers": "wrap", "thumb": "cap", "prop": "icepick",
                "source": "reverse (ice-pick) knife grip (est.): blade leaves the little-finger side, thumb pad caps the pommel end"},
    "dagger_index": {"label": "Dagger", "group": "weapon",
                     "angles": {"index": F(15, 10, 5, -2), "middle": F(60, 85, 50), "ring": F(65, 88, 50, -2), "pinky": F(68, 85, 48, -4),
                                "ring_mc": 4, "pinky_mc": 10, "thumb": TH(30, 35, 25, 20, 10)},
                     "fingers": {"index": "spine", "middle": "wrap", "ring": "wrap", "pinky": "wrap"}, "thumb": "blade", "prop": "dagger",
                     "source": "dagger with the index along the spine (est.): 22 mm handle, index pad on the 4 mm spine, thumb pad on the flat opposite the index P2"},
    "bow_hand": {"label": "Bow hand", "group": "bow",
                 "angles": {"index": F(35, 55, 25), "middle": F(40, 60, 30), "ring": F(45, 60, 30, -2), "pinky": F(50, 60, 30, -4),
                            "ring_mc": 0, "pinky_mc": 5, "thumb": TH(30, 40, 25, 15, 10)},
                 "fingers": "rest_on", "thumb": "rest", "prop": "bow",
                 "source": "holding the bow riser (est.): pressure on the thenar pad at the grip pivot, fingers relaxed 2-10 mm in front of the riser, not squeezing"},
    "bow_draw": {"label": "Bow draw", "group": "bow",
                 "angles": {"index": F(5, 35, 60, 4), "middle": F(5, 40, 60, -2), "ring": F(10, 45, 60), "pinky": F(45, 75, 45, 0),
                            "ring_mc": 0, "pinky_mc": 5, "thumb": TH(15, 20, 10, 20, 20)},
                 "fingers": {"index": "hold", "middle": "hold", "ring": "hold", "pinky": "hold"}, "thumb": "hold", "prop": "string",
                 "source": "Mediterranean draw (est.): string in the distal (DIP) creases of the index, middle and ring, arrow nock between the index and middle, back of the hand flat"},
    "shield_grip": {"label": "Shield grip", "group": "shield",
                    "angles": {"index": F(55, 78, 45), "middle": F(62, 80, 48), "ring": F(60, 80, 45, -2), "pinky": F(55, 70, 45, -4),
                               "ring_mc": 3, "pinky_mc": 8, "thumb": TH(40, 35, 30, 35, 30)},
                    "fingers": "wrap", "thumb": "wrap", "prop": "shield_bar",
                    "source": "centre-grip shield: 28 mm bar behind the boss (Shimawaki 2019 interpolated)"},
    "shield_strap": {"label": "Shield strap", "group": "shield",
                     "angles": {"index": F(70, 95, 65), "middle": F(72, 98, 66), "ring": F(74, 98, 66, -2), "pinky": F(74, 92, 62, -4),
                                "ring_mc": 4, "pinky_mc": 10, "thumb": TH(35, 20, 25, 45, 30)},
                     "fingers": "wrap", "thumb": "wrap", "prop": "strap",
                     "source": "enarme hand strap (est.): a 35 x 5 mm leather strap across the palm (the knight's socket_hand_l strap), fingers hooked round it, thumb over the index P2"},
    "torch": {"label": "Torch", "group": "power",
              "angles": {"index": F(48, 70, 40), "middle": F(55, 72, 45), "ring": F(52, 74, 38, -2), "pinky": F(45, 58, 43, -4),
                         "ring_mc": 3, "pinky_mc": 8, "thumb": TH(40, 35, 30, 25, 20)},
              "fingers": "wrap", "thumb": "wrap", "prop": "torch",
              "source": "power grip on a 40 mm haft (Shimawaki 2019 interpolated)"},
    "reins": {"label": "Reins", "group": "other",
              "angles": {"index": F(60, 85, 45), "middle": F(62, 88, 48), "ring": F(64, 88, 48, -2), "pinky": F(55, 80, 45, 8),
                         "ring_mc": 3, "pinky_mc": 8, "thumb": TH(35, 20, 25, 40, 20)},
              "fingers": "wrap", "thumb": "wrap", "prop": "rein",
              "source": "English-style reins (est.): a 20 x 4 mm rein through a loose fist (the pinky abducted where it enters between the ring and little fingers); the thumb closes on the rein / over the index P2. Not modelled: the rein's exit over the index under the thumb (the test rein is straight)"},
    "pinch": {"label": "Pinch", "group": "precision",
              "angles": {"index": F(45, 55, 35), "middle": F(45, 60, 30), "ring": F(50, 65, 30, -2), "pinky": F(55, 70, 30, -4),
                         "ring_mc": 0, "pinky_mc": 3, "thumb": TH(45, 45, 40, 25, 35)},
              "fingers": {"index": "pad", "middle": "hold", "ring": "hold", "pinky": "hold"}, "thumb": "pinch", "prop": "bead",
              "source": "tip-to-tip pinch of an 8 mm bead: thumb and index pads on the bead"},
    "point": {"label": "Point", "group": "basic",
              "angles": dict(FIST_ANGLES, index=F(0, 5, 5)),
              "fingers": {"index": "hold", "middle": "palm", "ring": "palm", "pinky": "palm"}, "thumb": "fist_mid", "prop": None,
              "source": "index straight; the other fingers and the thumb follow the fist rule (thumb over the middle P2)"},
    "open": {"label": "Open", "group": "basic",
             "angles": {"index": F(0, 0, 0, 10), "middle": F(0, 0, 0, 0), "ring": F(0, 0, 0, 6), "pinky": F(0, 0, 0, 14),
                        "ring_mc": 0, "pinky_mc": 0, "thumb": TH(-15, -30, -8, 0, 0)},
             "fingers": "hold", "thumb": "hold", "prop": None,
             "source": "flat, spread hand; radial thumb abduction; no finger contacts, hyperextension <= 5"},
    "fist_legacy": {"label": "Old fist (before)", "group": "legacy", "legacy": True, "angles": None, "fingers": "legacy",
                    "thumb": "legacy", "prop": None,
                    "source": "the pre-round-6 viewer fist (localRot X 75 / 82 / 60, thumb 45 / 45), kept for before / after renders"},
}
ORDER = ["relaxed", "fist", "hammer", "sword", "knife", "icepick", "dagger_index", "bow_hand", "bow_draw", "shield_grip",
         "shield_strap", "torch", "reins", "pinch", "point", "open", "fist_legacy"]
AUTO = {"Fists": "fist", "P_fists": "fist"}


# ---------------------------------------------------------------------------------------------------- props
# A prop is a list of parts in the prop frame (+Y = the grip axis, pointing to the thumb side of the hand; +Z = the
# blade edge / the 'down' side; origin = the grip centre). 'c' parts take part in the contact solve; 'd' parts are
# decoration (drawn, and checked for penetration). Shapes: cyl (r, h = half length, along Y), box (x, y, z half sizes,
# rr = edge rounding), sph (r). 'col' is an sRGB hex for the viewers.
WOOD, STEEL, LEATHER, BRASS, CLOTH, STRING = "#8a6a45", "#aeb4bb", "#5b3b22", "#b08a3c", "#3d2c20", "#ddd6c6"
PROPS = {
    "hammer": {"place": {"axis": 75, "alpha": 0.90, "gap": 0.8, "shift": 0.0, "search": {"alpha": [0.84, 0.88, 0.92, 0.96, 1.0]}},
               "parts": [{"k": "c", "s": "cyl", "r": 0.017, "h": 0.15, "t": [0, 0.07, 0], "col": WOOD},
                         {"k": "d", "s": "box", "b": [0.022, 0.024, 0.062], "rr": 0.004, "t": [0, 0.232, 0.018], "col": STEEL, "name": "head"}]},
    "sword": {"place": {"axis": 65, "alpha": 0.90, "gap": 0.8, "shift": 0.0,
                        "search": {"alpha": [0.84, 0.88, 0.92], "shift": [-0.03, -0.02, -0.01, 0.0], "theta": [62, 70, 78]}},
              "parts": [{"k": "c", "s": "cyl", "r": 0.0146, "h": 0.0525, "t": [0, 0, 0], "col": LEATHER},
                        {"k": "d", "s": "box", "b": [0.011, 0.008, 0.085], "rr": 0.003, "t": [0, 0.0605, 0], "col": STEEL, "name": "guard"},
                        {"k": "c", "s": "sph", "r": 0.016, "t": [0, -0.0675, 0], "col": STEEL, "name": "pommel"},
                        {"k": "d", "s": "box", "b": [0.0035, 0.40, 0.022], "rr": 0.001, "t": [0, 0.4685, 0], "col": STEEL, "name": "blade"}]},
    "knife": {"place": {"axis": 70, "alpha": 0.90, "gap": 0.8, "shift": 0.0,
                        "search": {"alpha": [0.86, 0.90, 0.94, 0.98], "shift": [-0.03, -0.02, -0.01, 0.0, 0.01]}},
              "parts": [{"k": "c", "s": "cyl", "r": 0.012, "h": 0.055, "t": [0, 0, 0], "col": WOOD},
                        {"k": "d", "s": "box", "b": [0.0035, 0.006, 0.019], "rr": 0.002, "t": [0, 0.061, 0.002], "col": BRASS, "name": "guard"},
                        {"k": "d", "s": "box", "b": [0.0015, 0.075, 0.014], "rr": 0.0008, "t": [0, 0.142, 0.004], "col": STEEL, "name": "blade"}]},
    "icepick": {"place": {"axis": 70, "alpha": 0.90, "gap": 0.8, "shift": 0.0,
                          "search": {"alpha": [0.86, 0.90, 0.94, 0.98], "shift": [-0.01, 0.0, 0.01, 0.02, 0.03]}},
                "parts": [{"k": "c", "s": "cyl", "r": 0.012, "h": 0.055, "t": [0, 0, 0], "col": WOOD},
                          {"k": "d", "s": "box", "b": [0.0035, 0.006, 0.019], "rr": 0.002, "t": [0, -0.061, 0.002], "col": BRASS, "name": "guard"},
                          {"k": "d", "s": "box", "b": [0.0015, 0.075, 0.014], "rr": 0.0008, "t": [0, -0.142, 0.004], "col": STEEL, "name": "blade"}]},
    "dagger": {"place": {"axis": 30, "alpha": 0.90, "gap": 0.8, "shift": 0.0,
                         "search": {"alpha": [0.84, 0.88, 0.92, 0.96], "shift": [-0.03, -0.02, -0.01, 0.0], "theta": [20, 30, 40]}},
               "parts": [{"k": "c", "s": "cyl", "r": 0.011, "h": 0.05, "t": [0, 0, 0], "col": LEATHER},
                         {"k": "d", "s": "box", "b": [0.005, 0.005, 0.022], "rr": 0.002, "t": [0, 0.055, 0], "col": STEEL, "name": "guard"},
                         {"k": "c", "s": "box", "b": [0.002, 0.10, 0.012], "rr": 0.0015, "t": [0, 0.16, 0.0], "col": STEEL, "name": "blade"}]},
    "bow": {"place": {"axis": 72, "alpha": 0.62, "gap": 0.6, "shift": 0.0, "zdir": "n",
                      "search": {"alpha": [0.55, 0.60, 0.65, 0.70]}},
            "parts": [{"k": "c", "s": "box", "b": [0.016, 0.065, 0.024], "rr": 0.013, "t": [0, 0, 0], "col": WOOD},
                      {"k": "d", "s": "box", "b": [0.014, 0.26, 0.009], "rr": 0.004, "t": [0, 0.33, 0.0], "col": WOOD, "name": "limb"},
                      {"k": "d", "s": "box", "b": [0.014, 0.26, 0.009], "rr": 0.004, "t": [0, -0.33, 0.0], "col": WOOD, "name": "limb"}]},
    "string": {"place": {"mode": "string"},
               "parts": [{"k": "c", "s": "cyl", "r": 0.0015, "h": 0.22, "t": [0, 0, 0], "col": STRING},
                         {"k": "d", "s": "cyl", "r": 0.0035, "h": 0.30, "t": [0, 0, 0.30], "q_axis": "z", "col": WOOD, "name": "arrow"}]},
    "shield_bar": {"place": {"axis": 80, "alpha": 0.90, "gap": 0.8, "shift": 0.0, "search": {"alpha": [0.84, 0.88, 0.92, 0.96, 1.0]}},
                   "parts": [{"k": "c", "s": "cyl", "r": 0.014, "h": 0.09, "t": [0, 0, 0], "col": WOOD},
                             {"k": "d", "s": "box", "b": [0.012, 0.008, 0.03], "rr": 0.003, "t": [0, 0.098, 0.012], "col": STEEL, "name": "mount"},
                             {"k": "d", "s": "box", "b": [0.012, 0.008, 0.03], "rr": 0.003, "t": [0, -0.098, 0.012], "col": STEEL, "name": "mount"}]},
    "strap": {"place": {"axis": 80, "alpha": 0.92, "gap": 0.8, "shift": 0.0, "zdir": "n", "search": {"alpha": [0.84, 0.88, 0.92, 0.96]}},
              "parts": [{"k": "c", "s": "box", "b": [0.0175, 0.07, 0.0025], "rr": 0.0024, "t": [0, 0, 0], "col": LEATHER}]},
    "torch": {"place": {"axis": 75, "alpha": 0.90, "gap": 0.8, "shift": 0.0, "search": {"alpha": [0.84, 0.88, 0.92, 0.96, 1.0]}},
              "parts": [{"k": "c", "s": "cyl", "r": 0.020, "h": 0.25, "t": [0, 0.17, 0], "col": WOOD},
                        {"k": "d", "s": "cyl", "r": 0.032, "h": 0.06, "t": [0, 0.42, 0], "col": CLOTH, "name": "head"}]},
    "rein": {"place": {"axis": 65, "alpha": 0.92, "gap": 0.8, "shift": 0.0, "zdir": "n",
                       "search": {"alpha": [0.86, 0.92, 0.98], "theta": [55, 62, 70]}},
             "parts": [{"k": "c", "s": "box", "b": [0.010, 0.13, 0.002], "rr": 0.0018, "t": [0, 0, 0], "col": LEATHER, "name": "rein"}]},
    "bead": {"place": {"mode": "pinch"},
             "parts": [{"k": "c", "s": "sph", "r": 0.004, "t": [0, 0, 0], "col": BRASS}]},
}


# per-body prop sizes: the sword grip grows with the hand along its length (the grip radius stays the knight's contract)
BODY_SCALE_Y = {"sword": {"male": 0.060 / 0.0525}}


def apply_body_sizes(kind):
    for name, per in BODY_SCALE_Y.items():
        k = per.get(kind, 1.0)
        base = PROPS[name].setdefault("_base_parts", json.loads(json.dumps(PROPS[name]["parts"])))
        parts = json.loads(json.dumps(base))
        for p in parts:
            if p["k"] == "c" and p["s"] == "cyl":
                p["h"] = round(p["h"] * k, 6)
            p["t"][1] = round(p["t"][1] * k, 6) if abs(p["t"][1]) > 0.02 else p["t"][1]
        # keep the guard / pommel against the grip ends and the blade beyond the guard
        grip = next(p for p in parts if p["k"] == "c" and p["s"] == "cyl")
        bgrip = next(p for p in base if p["k"] == "c" and p["s"] == "cyl")
        for p, b in zip(parts, base):
            if p is grip:
                continue
            d = grip["h"] - bgrip["h"]
            p["t"][1] = round(b["t"][1] + (d if b["t"][1] > 0 else -d), 6)
        PROPS[name]["parts"] = parts


def part_quat(p):
    """part rotation in the prop frame ([w, x, y, z]); 'q' = an explicit rotation (glTF order); 'q_axis': 'z' turns a Y
    cylinder to lie along Z."""
    if p.get("q"):
        return from_gltf(p["q"])
    if p.get("q_axis") == "z":
        return q_axis((1, 0, 0), 90)
    return np.array([1.0, 0, 0, 0])


def sdf_parts(parts, pts, kinds=("c",)):
    """signed distance of pts (N, 3; prop frame) to the union of the parts of the given kinds."""
    out = np.full(len(pts), 1.0)
    for p in parts:
        if p["k"] not in kinds:
            continue
        R = q_mat(part_quat(p))
        q = (pts - np.array(p["t"])) @ R          # into the part frame (R orthonormal: inverse = transpose)
        if p["s"] == "cyl":
            dr = np.hypot(q[:, 0], q[:, 2]) - p["r"]; dy = np.abs(q[:, 1]) - p["h"]
            d = np.minimum(np.maximum(dr, dy), 0) + np.hypot(np.maximum(dr, 0), np.maximum(dy, 0))
        elif p["s"] == "box":
            b = np.array(p["b"]) - p["rr"]
            qq = np.abs(q) - b
            d = np.linalg.norm(np.maximum(qq, 0), axis=1) + np.minimum(qq.max(1), 0) - p["rr"]
        else:
            d = np.linalg.norm(q, axis=1) - p["r"]
        out = np.minimum(out, d)
    return out


def mirror_parts(parts):
    """parts for the mirrored (right-hand) prop frame: the frame is mirrored with its local X flipped, so a part at
    (x, y, z) with rotation R becomes (-x, y, z) with S R S (S = diag(-1, 1, 1)); x-symmetric parts are unchanged."""
    out = []
    for p in parts:
        q = dict(p)
        q["t"] = [-p["t"][0], p["t"][1], p["t"][2]]
        if p.get("q"):
            R = q_mat(from_gltf(p["q"]))
            q["q"] = to_gltf(mat_q(S_MIR @ R @ S_MIR))
        out.append(q)
    return out


class Placed:
    """a prop placed in armature space: M (4x4) = prop frame -> armature."""

    def __init__(self, name, M, parts=None):
        self.name, self.M = name, np.asarray(M, float)
        self.parts = parts if parts is not None else PROPS[name]["parts"]
        self.Minv = np.linalg.inv(self.M)

    def local(self, pts):
        return pts @ self.Minv[:3, :3].T + self.Minv[:3, 3]

    def sdf(self, pts, kinds=("c",)):
        return sdf_parts(self.parts, self.local(pts), kinds)


# ---------------------------------------------------------------------------------------------------- the solver
GAP = {"prop": 0.0010, "palm": 0.0008, "finger": 0.0004}
REST_FILTER = 0.003


class Solver:
    def __init__(self, hand):
        self.h = hand
        h = hand
        self.rest_sd = {}
        self.borders = {}
        self.pad_idx = self._pad_vertices()

    # rest-filtered vertices of src (only those >= 3 mm outside dst at rest), as in tools/audit_bl.py
    def filt(self, src, dst):
        key = (src, dst)
        if key not in self.rest_sd:
            h = self.h
            idx = np.nonzero(h.sets[src])[0]
            tree = self.tree(h.co0, dst)
            sd = signed_dist(tree, h.co0[idx])
            self.rest_sd[key] = idx[sd > REST_FILTER]
        return self.rest_sd[key]

    def _pad_vertices(self):
        """palmar pad vertices of every distal phalanx and of the thumb (rest): flesh side, middle 20-85 % of the
        segment."""
        h, s = self.h, self.h.side
        out = {}
        for f in FINGERS + ("thumb",):
            b = ("%s_03" % f) + s
            idx = np.nonzero(h.sets["%s_03" % f])[0]
            hd, tl = h.head[b], h.tail[b]
            L = np.linalg.norm(tl - hd); u = (tl - hd) / L
            rel = h.co0[idx] - hd
            t = rel @ u / L
            perp = rel - np.outer(rel @ u, u)
            pdir = h.thumb_pad if f == "thumb" else h.n
            pdir = pdir - (pdir @ u) * u; pdir /= np.linalg.norm(pdir)
            side = perp @ pdir / np.maximum(np.linalg.norm(perp, axis=1), 1e-9)
            out[f] = idx[(t > 0.2) & (t < 0.85) & (side > 0.55)]
        return out

    # ---- evaluation helpers
    def posed(self, ang):
        D = self.h.deltas(ang)
        co, P = self.h.skin(D)
        return co, D, P

    def tree(self, co, key):
        if key not in self.borders:
            self.borders[key] = border_faces(self.h.faces_of(key))
        return bvh(co, self.h.faces_of(key), self.borders[key])

    # ---- finger solves
    def finger_links(self, f, j):
        """vertex indices of the links moved by joint j (1 = MCP) of finger f."""
        m = np.zeros(len(self.h.co0), bool)
        for i in range(j, 4):
            m |= self.h.sets["%s_0%d" % (f, i)]
        return np.nonzero(m)[0]

    def clearance(self, ang, f, j, obst, links=None):
        """min over obstacles of (signed distance - obstacle gap) for the links of finger f moved by joint j (or the
        given link numbers)."""
        idx = self.finger_links(f, j) if links is None else np.nonzero(np.any([self.h.sets["%s_0%d" % (f, i)] for i in links], 0))[0]
        D = self.h.deltas(ang)
        co_l, _ = self.h.skin(D, idx)
        pos = {int(v): k for k, v in enumerate(idx)}
        c = 1.0
        if "prop" in obst:
            c = min(c, float(obst["prop"].sdf(co_l, kinds=obst.get("prop_kinds", ("c", "d"))).min()) - obst.get("prop_gap", GAP["prop"]))
        if "palm" in obst:
            fi = np.intersect1d(idx, np.concatenate([self.filt("%s_0%d" % (f, i), "palm") for i in range(1, 4)]))
            if len(fi):
                c = min(c, float(signed_dist(obst["palm"], co_l[[pos[int(v)] for v in fi]]).min()) - GAP["palm"])
        for nb, tr in obst.get("fingers", {}).items():
            fi = np.intersect1d(idx, self.filt(f, nb))
            if len(fi):
                c = min(c, float(signed_dist(tr, co_l[[pos[int(v)] for v in fi]]).min()) - GAP["finger"])
        return c

    def close_joint(self, ang, f, jn, lo, hi, obst, step=3.0, couple=None, links=None):
        """raise ang[f][jn] from lo toward hi (with couple = (joint, k): that joint follows as rest + k * (x - rest));
        stop at the first contact (clearance 0). Returns (angle, contact?, the finger's angles)."""
        j = FJ.index(jn) + 1
        a = dict(ang); a[f] = dict(ang[f])
        rj = self.h.rest_ang["%s.%s" % (f, jn)]

        def setx(x):
            a[f][jn] = x
            if couple:
                cj, k = couple
                a[f][cj] = min(ROM[cj][1], max(a[f][cj], k * x))
        setx(lo)
        if self.clearance(a, f, j, obst, links) < 0:
            return lo, True, dict(a[f])
        x = lo
        while x < hi:
            x2 = min(hi, x + step)
            setx(x2)
            if self.clearance(a, f, j, obst, links) < 0:
                l, u = x, x2
                for _ in range(10):
                    mid = 0.5 * (l + u); setx(mid)
                    if self.clearance(a, f, j, obst, links) < 0:
                        u = mid
                    else:
                        l = mid
                setx(l)
                return l, True, dict(a[f])
            x = x2
        setx(hi)
        return hi, False, dict(a[f])

    def open_until_clear(self, ang, f, obst, joints=("MCP",)):
        """reduce the given joints together (proportionally toward the rest curl) until the finger clears."""
        a = dict(ang); a[f] = dict(ang[f])
        if self.clearance(a, f, 1, obst) >= 0:
            return a[f], 1.0
        base = {jn: ang[f][jn] for jn in joints}
        rest = {jn: self.h.rest_ang["%s.%s" % (f, jn)] for jn in joints}
        l, u = 0.0, 1.0
        for _ in range(14):
            mid = 0.5 * (l + u)
            for jn in joints:
                a[f][jn] = rest[jn] + mid * (base[jn] - rest[jn])
            if self.clearance(a, f, 1, obst) >= 0:
                l = mid
            else:
                u = mid
        for jn in joints:
            a[f][jn] = rest[jn] + l * (base[jn] - rest[jn])
        return a[f], l

    # ---- obstacles for a posed hand (trees rebuilt from the posed mesh: palm skin near the MCPs moves with the fingers)
    def obstacles(self, co, prop=None, fingers=(), palm=True, prop_gap=None):
        o = {}
        if palm:
            o["palm"] = self.tree(co, "palm")
        if prop is not None:
            o["prop"] = prop
            if prop_gap is not None:
                o["prop_gap"] = prop_gap
        o["fingers"] = {nb: self.tree(co, nb) for nb in fingers}
        return o

    def solve_palm_finger(self, ang, f, done):
        """fist rule on this rig (dorsal joint pivots, plan H2): search MCP over rest..ROM and, for each MCP, the largest
        PIP / DIP curl (along rest -> target, extended toward the ROM limit) that keeps the finger >= 0.8 mm off the palm
        skin and the already-placed fingers; keep the pair closest to the anatomical targets (weights PIP > MCP > DIP)
        among those whose distal pad touches the palm (<= 1.5 mm)."""
        h = self.h
        tgt = dict(ang[f])
        rest = {jn: h.rest_ang["%s.%s" % (f, jn)] for jn in FJ}
        smax = min((ROM[jn][1] - rest[jn]) / max(1e-6, tgt[jn] - rest[jn]) for jn in ("PIP", "DIP"))
        a = dict(ang); a[f] = dict(tgt)
        co, _, _ = self.posed(a)
        pad = np.intersect1d(self.pad_idx[f], self.filt("%s_03" % f, "palm"))

        def setp(mcp, sc):
            a[f] = dict(tgt, MCP=mcp, PIP=rest["PIP"] + sc * (tgt["PIP"] - rest["PIP"]),
                        DIP=rest["DIP"] + sc * (tgt["DIP"] - rest["DIP"]))

        def clear(mcp, sc):
            setp(mcp, sc)
            c, _, _ = self.posed(a)
            o = self.obstacles(c, fingers=done)
            return self.clearance(a, f, 1, o), c, o
        best = None
        mcp = rest["MCP"]
        while mcp <= ROM["MCP"][1] + 1e-6:
            cl, c, o = clear(mcp, 0.0)
            if cl >= 0:
                cl1, c1, o1 = clear(mcp, smax)
                if cl1 >= 0:
                    sc = smax
                else:
                    l, u = 0.0, smax
                    for _ in range(11):
                        m = 0.5 * (l + u)
                        if clear(mcp, m)[0] >= 0:
                            l = m
                        else:
                            u = m
                    sc = l
                setp(mcp, sc)
                c, _, _ = self.posed(a)
                padd = float(signed_dist(self.tree(c, "palm"), c[pad], 0.2).min()) if len(pad) else 1.0
                err = (((a[f]["MCP"] - tgt["MCP"]) / 1.0) ** 2 + ((a[f]["PIP"] - tgt["PIP"]) / 0.7) ** 2
                       + ((a[f]["DIP"] - tgt["DIP"]) / 1.2) ** 2)
                cand = (padd <= 0.0015, -err, mcp, sc, padd)
                if best is None or cand[:2] > best[:2]:
                    best = cand
            mcp += 2.5
        _, _, mcp, sc, padd = best
        setp(mcp, sc)
        for _ in range(25):                       # the surfaces may still graze (edges through faces): open a little
            c, _, _ = self.posed(a)
            if sum(tri_cross(c, h.faces_of("%s_0%d" % (f, i)), h.faces_of("palm_core")) for i in (2, 3)) == 0:
                break
            sc -= 0.01
            setp(mcp, sc)
        c, _, _ = self.posed(a)
        padd = float(signed_dist(self.tree(c, "palm"), c[pad], 0.2).min()) if len(pad) else 1.0
        return dict(a[f]), bool(padd <= 0.0015)

    def wrap_finger(self, ang, f, prop, done, gap=None, start=0.0, k_dip=0.65):
        """wrap a finger round the prop: (1) MCP closes with the finger straight until any phalanx touches; (2) PIP
        closes with the DIP coupled (DIP = 0.65 PIP above the rest curl: the ORL coupling) until P2 or P3 touches; (3)
        the DIP closes until P3 touches. Every stop is at the prop gap; the palm and the already-placed fingers are
        obstacles too (ROM-limited)."""
        a = dict(ang); a[f] = dict(ang[f])
        for jn in FJ:
            a[f][jn] = start
        hits = {}
        co, _, _ = self.posed(a)
        o = self.obstacles(co, prop=prop, fingers=done, prop_gap=gap)
        x, hits["MCP"], a[f] = self.close_joint(a, f, "MCP", a[f]["MCP"], ROM["MCP"][1], o, step=2.0)
        x, hits["PIP"], a[f] = self.close_joint(a, f, "PIP", a[f]["PIP"], ROM["PIP"][1], o, step=2.0, couple=("DIP", k_dip))
        x, hits["DIP"], a[f] = self.close_joint(a, f, "DIP", a[f]["DIP"], ROM["DIP"][1], o, step=2.0)
        return a[f], hits

    # ---- thumb
    def thumb_search(self, ang, mode, prop=None, x0=None, gap=None):
        """bounded pattern search over the thumb's CMC flex / abd / opp, MCP, IP for the preset's target; penetration
        (thumb into the palm / fingers / prop, fingers into the thumb) is a steep penalty."""
        h, s = self.h, self.h.side
        keys = ("CMC_flex", "CMC_abd", "CMC_opp", "MCP", "IP")
        lim = [ROM["CMC_flex"], ROM["CMC_abd"], ROM["CMC_opp"], ROM["tMCP"], ROM["tIP"]]
        t0 = ang["thumb"]
        x = np.array([t0[k] for k in keys], float) if x0 is None else np.array(x0, float)
        x0v = x.copy()
        co, _, _ = self.posed(ang)
        fingers_t = {f: self.tree(co, f) for f in FINGERS}
        p2 = {f: self.tree(co, "%s_02" % f) for f in ("index", "middle", "ring")}
        palm_core = self.tree(co, "palm_core")
        th_idx = np.nonzero(h.sets["thumb"])[0]
        pad = self.pad_idx["thumb"]
        mask = np.zeros(len(h.co0), bool); mask[th_idx] = True; mask[pad] = True
        sub = np.nonzero(mask)[0]; pos = {int(v): k for k, v in enumerate(sub)}
        mp = lambda idx: np.array([pos[int(v)] for v in idx], int)
        fil_m = {f: mp(self.filt("thumb", f)) for f in FINGERS}; fil_pm = mp(self.filt("thumb", "palm_core"))
        pad_m = mp(pad); th_m = mp(th_idx); t3_m = mp(np.nonzero(h.sets["thumb_03"])[0])
        # finger vertices that could enter the thumb (rest-filtered, near the thumb's reach)
        reach = np.linalg.norm(co - h.head["thumb_02" + s], axis=1) < 0.09
        fin_in = {f: np.intersect1d(self.filt(f, "thumb"), np.nonzero(reach)[0]) for f in FINGERS}
        thumb_faces = h.faces_of("thumb")
        sub_faces = [[pos[v] for v in fc] for fc in thumb_faces if all(v in pos for v in fc)]
        sub_border = border_faces(sub_faces)
        g = gap or 0.0012
        margin = [0.0004]

        def evalx(xx):
            a = dict(ang); a["thumb"] = dict(zip(keys, xx))
            D = h.deltas(a)
            c, _ = h.skin(D, mask)
            pen = 0.0
            for f in FINGERS:
                if len(fil_m[f]):
                    sd = signed_dist(fingers_t[f], c[fil_m[f]], 0.01)
                    pen += float((np.maximum(0, margin[0] - sd) ** 2).sum())
            if len(fil_pm):
                sd = signed_dist(palm_core, c[fil_pm], 0.01)
                pen += float((np.maximum(0, 0.0006 - sd) ** 2).sum())
            tt = bvh(c, sub_faces, sub_border)
            for f in FINGERS:
                if len(fin_in[f]):
                    sd = signed_dist(tt, co[fin_in[f]], 0.01)
                    pen += float((np.maximum(0, margin[0] - sd) ** 2).sum())
            if prop is not None:
                sdp = prop.sdf(c[th_m], kinds=("c", "d"))
                pen += float((np.maximum(0, 0.0005 - sdp) ** 2).sum())
            cpad = c[pad_m]
            obj = 0.0
            if mode in ("fist", "fist_mid"):
                f1, f2 = ("index", "middle") if mode == "fist" else ("middle", "ring")
                d1 = signed_dist(p2[f1], cpad, 0.05).min()
                d2 = signed_dist(p2[f2], c[th_m], 0.05).min()
                obj = (d1 - g) ** 2 + 0.5 * max(0.0, d2 - g) ** 2
            elif mode == "wrap":
                dp = min(prop.sdf(cpad).min(), prop.sdf(c[t3_m]).min())
                d1 = min(signed_dist(p2["index"], c[t3_m], 0.05).min(), signed_dist(p2["middle"], c[t3_m], 0.05).min())
                obj = (min(dp, d1) - g) ** 2 + 0.5 * max(0.0, d1 - g) ** 2 + 0.2 * max(0.0, dp - g) ** 2
            elif mode == "cap":
                loc = prop.local(cpad)
                top = max(p["t"][1] + p["h"] for p in prop.parts if p["k"] == "c" and p["s"] == "cyl")
                dp = prop.sdf(cpad).min()
                obj = (dp - g) ** 2 + (float(loc[:, 1].mean()) - top - g) ** 2
            elif mode in ("blade", "pinch"):
                dp = prop.sdf(cpad).min()
                obj = (dp - g) ** 2
            elif mode == "rein":                 # the distal thumb presses the rein where it leaves the fist
                dp = prop.sdf(c[t3_m]).min()
                obj = (dp - g) ** 2
            elif mode == "rest":
                dp = prop.sdf(c[th_m], kinds=("c", "d")).min()
                obj = (dp - 0.003) ** 2
            reg = float((((xx - x0v) / 60.0) ** 2).sum()) * 1e-6
            return obj + 50.0 * pen + reg

        n_eval = 0
        for attempt in range(4):
            fx = evalx(x)
            step = 8.0 if attempt == 0 else 3.0
            while step > 0.2:
                improved = False
                for i in range(5):
                    for sg in (1, -1):
                        y = x.copy(); y[i] = min(lim[i][1], max(lim[i][0], y[i] + sg * step))
                        if y[i] == x[i]:
                            continue
                        fy = evalx(y); n_eval += 1
                        if fy < fx - 1e-14:
                            x, fx = y, fy; improved = True
                if not improved:
                    step *= 0.5
            a = dict(ang); a["thumb"] = dict(zip(keys, x))
            cfull, _, _ = self.posed(a)
            cr = sum(tri_cross(cfull, h.faces_of("thumb"), h.faces_of(f)) for f in FINGERS)
            if cr == 0:
                break
            margin[0] += 0.0006                       # surfaces still cross: demand more clearance and search again
        return dict(zip(keys, [round(float(v), 2) for v in x])), fx, n_eval

    def thumb_contact_ok(self, ang, tm, prop):
        """the thumb part of gate G-H2 (same numbers as measure()) + no thumb penetration beyond 0.5 mm."""
        h = self.h
        co, _, _ = self.posed(ang)
        pad = self.pad_idx["thumb"]; t3 = np.nonzero(h.sets["thumb_03"])[0]
        if self.thumb_pen(ang, prop, co) > 0.0005:
            return False
        d = lambda idx, key: float(signed_dist(self.tree(co, key), co[idx], 0.2).min()) * 1000
        if tm == "fist":
            return 0.0 <= d(pad, "index_02") <= 3.0
        if tm == "fist_mid":
            return 0.0 <= d(pad, "middle_02") <= 3.0
        if prop is None:
            return True
        pp = float(prop.sdf(co[pad]).min()) * 1000; p3 = float(prop.sdf(co[t3]).min()) * 1000
        if tm == "wrap":
            return (-0.5 <= min(pp, p3) <= 3.0) or (0.0 <= min(d(t3, "index_02"), d(t3, "middle_02")) <= 3.0)
        if tm == "pinch":
            return -0.5 <= pp <= 1.5
        if tm == "rein":
            return -0.5 <= min(pp, p3) <= 3.0
        return -0.5 <= pp <= 3.0

    def thumb_hold(self, ang, prop=None):
        """keep the start angles; if the thumb penetrates, back it off toward rest (MCP / IP first, then the CMC)."""
        h = self.h
        co, _, _ = self.posed(ang)
        if self.thumb_pen(ang, prop, co) <= 0.0005:
            return ang["thumb"], 1.0
        t0 = dict(ang["thumb"])
        rest = {"CMC_flex": 0, "CMC_abd": 0, "CMC_opp": 0, "MCP": h.rest_ang["thumb.MCP"], "IP": h.rest_ang["thumb.IP"]}
        l, u = 0.0, 1.0
        a = dict(ang)
        for _ in range(12):
            m = 0.5 * (l + u)
            a["thumb"] = {k: rest[k] + m * (t0[k] - rest[k]) for k in t0}
            if self.thumb_pen(a, prop) <= 0.0005:
                l = m
            else:
                u = m
        return {k: round(rest[k] + l * (t0[k] - rest[k]), 2) for k in t0}, l

    def thumb_pen(self, ang, prop=None, co=None):
        if co is None:
            co, _, _ = self.posed(ang)
        pen = 0.0
        for f in FINGERS:
            fi = self.filt("thumb", f)
            if len(fi):
                pen = max(pen, -float(signed_dist(self.tree(co, f), co[fi]).min()))
        fi = self.filt("thumb", "palm_core")
        if len(fi):
            pen = max(pen, -float(signed_dist(self.tree(co, "palm_core"), co[fi]).min()))
        if prop is not None:
            pen = max(pen, -float(prop.sdf(co[self.h.sets["thumb"]], kinds=("c", "d")).min()))
        return pen

    # ---- prop placement (left hand, armature rest space)
    def place_prop(self, name, ang, alpha=None, shift=None, theta=None):
        """the prop's grip axis crosses the palm at `theta` degrees from the middle metacarpal toward the thumb, through
        the palm point alpha * (wrist -> middle MCP) (+ shift along the axis); the prop is then pushed out along the palm
        normal until it clears the palm skin by the placement gap (it rests on the palm / thenar / hypothenar pads).
        Prop +Z = distal (blade edges, 'zdir': 'd') or palmar ('zdir': 'n', flat straps and the bow riser)."""
        h = self.h
        spec = PROPS[name]["place"]
        alpha = spec["alpha"] if alpha is None else alpha
        shift = spec.get("shift", 0.0) if shift is None else shift
        theta = spec["axis"] if theta is None else theta
        L = np.linalg.norm(h.head["middle_01" + h.side] - h.w)
        Wd = np.linalg.norm(h.head["index_01" + h.side] - h.head["pinky_01" + h.side])
        th = math.radians(theta)
        a = unit(math.cos(th) * h.d + math.sin(th) * h.r)
        zref = h.n if spec.get("zdir") == "n" else h.d
        z = unit(zref - (zref @ a) * a)
        x = np.cross(a, z)
        A0 = h.w + alpha * L * h.d + spec.get("beta", 0.0) * Wd * h.r + shift * a

        def M_at(hgt):
            M = np.eye(4); M[:3, 0] = x; M[:3, 1] = a; M[:3, 2] = z; M[:3, 3] = A0 + hgt * h.n
            return M
        co, _, _ = self.posed(ang)
        pm = h.sets["palm"]
        gap = spec.get("gap", 0.8) / 1000.0
        lo, hi = -0.03, 0.10
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            if Placed(name, M_at(mid)).sdf(co[pm], kinds=("c", "d")).min() < gap:
                lo = mid
            else:
                hi = mid
        pl = Placed(name, M_at(hi))
        pl.params = {"alpha": round(alpha, 4), "shift": round(shift, 4), "theta": round(theta, 2), "lift_mm": round(hi * 1000, 2)}
        return pl

    def place_string(self, ang):
        """bow draw: the string (a thin cylinder) passes through the crook (palmar side of the DIP) of the middle finger,
        running from the ring crook toward the index crook; then the index and ring (MCP, PIP; DIP from the preset)
        are solved on a grid so their crooks also sit on the string (P2 and P3 both 0.2-1.5 mm off it, no penetration).
        The arrow (decoration) lies perpendicular to the string between the index and middle, along their P2."""
        h, s = self.h, self.h.side
        r = PROPS["string"]["parts"][0]["r"]

        def crook(a, f):
            co, D, P = self.posed(a)
            j = P["%s_03%s" % (f, s)][:3, 3]
            u2 = unit(j - P["%s_02%s" % (f, s)][:3, 3])
            Rb = P["%s_03%s" % (f, s)][:3, :3] @ h.rest["%s_03%s" % (f, s)][:3, :3].T
            u3 = unit(Rb @ (h.tail["%s_03%s" % (f, s)] - h.head["%s_03%s" % (f, s)]))
            bis = unit(u3 - u2)
            tr = bvh(co, h.faces_of("%s_02" % f) + h.faces_of("%s_03" % f))
            tt = 0.03                                   # from outside the hook, move in until the string touches
            while tt > 0.0 and signed_dist(tr, [j + bis * tt], 0.05)[0] > r + 0.0007:
                tt -= 0.0002
            return j + bis * tt, u2
        a = json.loads(json.dumps(ang))
        pm, u_m = crook(a, "middle")
        pi, u_i = crook(a, "index"); pr, u_r = crook(a, "ring")
        ax = unit(pi - pr)
        z = unit(u_m - (u_m @ ax) * ax); x = np.cross(ax, z)
        M = np.eye(4); M[:3, 0] = x; M[:3, 1] = ax; M[:3, 2] = z; M[:3, 3] = pm
        pl = Placed("string", M)
        info = {}
        for f in ("index", "ring"):
            best = None
            for mcp in np.arange(-5.0, 45.1, 2.5):
                for pip in np.arange(15.0, 80.1, 2.5):
                    a[f] = dict(a[f], MCP=float(mcp), PIP=float(pip))
                    D = h.deltas(a)
                    idx = np.nonzero(h.sets["%s_02" % f] | h.sets["%s_03" % f])[0]
                    c, _ = h.skin(D, idx)
                    sd = pl.sdf(c)
                    n2 = int(h.sets["%s_02" % f][idx].sum())
                    g2, g3 = float(sd[:n2].min()) if False else float(sd[h.sets["%s_02" % f][idx]].min()), float(sd[h.sets["%s_03" % f][idx]].min())
                    cost = (g2 - 0.0007) ** 2 + (g3 - 0.0007) ** 2 + 50 * (min(0.0, min(g2, g3) - 0.0002)) ** 2
                    cost += 1e-8 * ((mcp - ang[f]["MCP"]) ** 2 + (pip - ang[f]["PIP"]) ** 2)
                    if best is None or cost < best[0]:
                        best = (cost, float(mcp), float(pip), g2, g3)
            a[f] = dict(a[f], MCP=best[1], PIP=best[2])
            info[f] = {"gap_P2_mm": round(best[3] * 1000, 2), "gap_P3_mm": round(best[4] * 1000, 2)}
        mid_pt = 0.5 * (crook(a, "index")[0] + pm)
        yoff = float((mid_pt - M[:3, 3]) @ ax)
        PROPS["string"]["parts"][1]["t"] = [0.0, round(yoff, 5), 0.30]
        pl = Placed("string", M)
        pl.params = {"mode": "string", "fingers": info}
        return pl, a

    def place_bead(self, ang):
        h, s = self.h, self.h.side
        co, D, P = self.posed(ang)
        pad = self.pad_idx["index"]
        R = P["index_03" + s][:3, :3] @ h.rest["index_03" + s][:3, :3].T
        nrm = unit(R @ h.n)
        c = co[pad].mean(0)
        rr = PROPS["bead"]["parts"][0]["r"]
        idx = np.nonzero(h.sets["index_03"])[0]
        M = np.eye(4); M[:3, :3] = h.rest["hand" + s][:3, :3]
        tt = 0.0
        for _ in range(200):
            M[:3, 3] = c + nrm * (rr + tt)
            if Placed("bead", M).sdf(co[idx]).min() >= 0.0006:
                break
            tt += 0.0002
        return Placed("bead", M)

    # ---- one preset
    def solve_fingers(self, ang, modes, prop, pname):
        info = {}
        done = []
        for f in FINGERS:
            md = modes[f]
            if md == "palm":
                ang[f], hit = self.solve_palm_finger(ang, f, done)
                info[f] = {"mode": md, "contact": hit}
            elif md in ("wrap", "rest_on"):
                gap = 0.004 if md == "rest_on" else None
                ang[f], hits = self.wrap_finger(ang, f, prop, done, gap=gap)
                info[f] = {"mode": md, "contact": hits}
            elif md == "spine":
                co, _, _ = self.posed(ang)
                o = self.obstacles(co, prop=prop, fingers=done)
                x, hit, _ = self.close_joint(ang, f, "MCP", -5.0, ROM["MCP"][1], o, step=1.0)
                ang[f] = dict(ang[f], MCP=x)
                info[f] = {"mode": md, "contact": hit}
            else:          # hold / pad
                co, _, _ = self.posed(ang)
                o = self.obstacles(co, prop=prop if pname not in ("string",) else None, fingers=done)
                ang[f], frac = self.open_until_clear(ang, f, o, joints=("MCP", "PIP", "DIP"))
                info[f] = {"mode": md, "kept": round(frac, 3)}
            done.append(f)
        self.fix_palm_crossings(ang, modes, info)
        return ang, info

    def fix_palm_crossings(self, ang, modes, info):
        """a later finger (or the thumb's CMC, which moves the thenar skin) can move the palm skin under an earlier
        fingertip: open the PIP / DIP of any fist finger whose P2 / P3 surface still crosses the palm."""
        palm_f = [f for f in FINGERS if modes[f] == "palm"]
        for _ in range(40):
            co, _, _ = self.posed(ang)
            bad = [f for f in palm_f if sum(tri_cross(co, self.h.faces_of("%s_0%d" % (f, i)), self.h.faces_of("palm_core")) for i in (2, 3))]
            if not bad:
                break
            for f in bad:
                ang[f] = dict(ang[f], PIP=ang[f]["PIP"] - 1.0, DIP=ang[f]["DIP"] - 0.7)
                info.setdefault(f, {})["opened_for_crossings"] = info.get(f, {}).get("opened_for_crossings", 0) + 1

    # ---- data-driven grip solve: literature angles -> fit the prop into the cavity -> refine each finger
    def fit_prop(self, co, placed, targets, rot=True, iters=None, pen_idx=None):
        """move the prop (6 DOF: translation + small rotation about its grip centre) so that each target vertex set's
        minimum distance to the prop's contact parts approaches its gap, with no hand vertex inside any prop part.
        targets: [(vertex indices, gap m, weight)]."""
        M0 = placed.M.copy()
        allv = co if pen_idx is None else co[pen_idx]

        def M_of(x):
            R = q_mat(q_axis((1, 0, 0), 0.0))
            w = np.array(x[3:6])
            if np.linalg.norm(w) > 1e-9:
                R = q_mat(q_axis(w, math.degrees(np.linalg.norm(w))))
            M = M0.copy()
            M[:3, :3] = R @ M0[:3, :3]
            M[:3, 3] = M0[:3, 3] + np.array(x[:3])
            return M

        def f(x):
            pl = Placed(placed.name, M_of(x), placed.parts)
            e = 0.0
            for idx, gap, w in targets:
                if len(idx):
                    e += w * (float(pl.sdf(co[idx]).min()) - gap) ** 2
            sd = pl.sdf(allv, kinds=("c", "d"))
            e += 400.0 * float((np.maximum(0.0, 0.0006 - sd) ** 2).sum())
            return e
        x = np.zeros(6)
        fx = f(x)
        steps = [0.002, 0.001, 0.0005, 0.00025, 0.0001]
        rsteps = [0.03, 0.015, 0.008, 0.004, 0.002]
        for st, rs in zip(steps, rsteps):
            improved = True
            while improved:
                improved = False
                for i in range(6 if rot else 3):
                    for sg in (1, -1):
                        y = x.copy(); y[i] += sg * (st if i < 3 else rs)
                        fy = f(y)
                        if fy < fx - 1e-16:
                            x, fx = y, fy; improved = True
        out = Placed(placed.name, M_of(x), placed.parts)
        out.params = dict(getattr(placed, "params", {}), fit_move_mm=round(float(np.linalg.norm(x[:3])) * 1000, 2),
                          fit_rot_deg=round(math.degrees(float(np.linalg.norm(x[3:]))), 2), fit_cost=fx)
        return out

    def refine_finger(self, ang, f, placed, done, gaps=(0.0010, 0.0010, 0.0010), weights=(0.3, 1.0, 1.0), table=None,
                      lam=0.15, free=(0, 1, 2), kinds=("c",)):
        """pattern search over the finger's MCP / PIP / DIP (ROM-bounded) so each phalanx sits at its gap from the prop,
        without entering the prop, the palm or the already-placed fingers; a small pull toward the start angles."""
        h = self.h
        idx = self.finger_links(f, 1)
        pos = {int(v): k for k, v in enumerate(idx)}
        seg = [np.array([pos[int(v)] for v in np.nonzero(h.sets["%s_0%d" % (f, i)])[0]]) for i in (1, 2, 3)]
        co, _, _ = self.posed(ang)
        palm = self.tree(co, "palm")
        fil_palm = np.array([pos[int(v)] for v in np.intersect1d(idx, np.concatenate([self.filt("%s_0%d" % (f, i), "palm") for i in (1, 2, 3)]))], int)
        nb = {d: (self.tree(co, d), np.array([pos[int(v)] for v in self.filt(f, d)], int)) for d in done}
        table = table or ang[f]
        a = dict(ang); a[f] = dict(ang[f])

        def cost(x):
            a[f] = dict(a[f], MCP=x[0], PIP=x[1], DIP=x[2])
            c, _ = h.skin(h.deltas(a), idx)
            sd = placed.sdf(c, kinds=kinds)
            sdall = placed.sdf(c, kinds=("c", "d"))
            e = 0.0
            for k in range(3):
                if weights[k] > 0 and gaps[k] is not None:
                    e += weights[k] * (float(sd[seg[k]].min()) - gaps[k]) ** 2
            e += 400.0 * float((np.maximum(0.0, 0.0005 - sdall) ** 2).sum())
            if len(fil_palm):
                sp = signed_dist(palm, c[fil_palm], 0.01)
                e += 400.0 * float((np.maximum(0.0, 0.0006 - sp) ** 2).sum())
            for d, (tr, fi) in nb.items():
                if len(fi):
                    sq = signed_dist(tr, c[fi], 0.01)
                    e += 400.0 * float((np.maximum(0.0, 0.0003 - sq) ** 2).sum())
            e += lam * 1e-6 * sum(((x[i] - table[jn]) / 15.0) ** 2 for i, jn in enumerate(FJ))
            return e
        x = np.array([a[f][jn] for jn in FJ], float)
        fx = cost(x)
        step = 8.0
        while step > 0.1:
            improved = False
            for i, jn in enumerate(FJ):
                if i not in free:
                    continue
                for sg in (1, -1):
                    y = x.copy(); y[i] = min(ROM[jn][1], max(ROM[jn][0], y[i] + sg * step))
                    if y[i] == x[i]:
                        continue
                    fy = cost(y)
                    if fy < fx - 1e-16:
                        x, fx = y, fy; improved = True
            if not improved:
                step *= 0.5
        a[f] = dict(a[f], MCP=round(float(x[0]), 2), PIP=round(float(x[1]), 2), DIP=round(float(x[2]), 2))
        return a[f], fx

    def grip_targets(self, modes, pname, co):
        h = self.h
        T = []
        for f in FINGERS:
            md = modes[f]
            if md == "wrap":
                for i, w in ((1, 0.3), (2, 1.0), (3, 1.0)):
                    T.append((np.nonzero(h.sets["%s_0%d" % (f, i)])[0], 0.0010, w))
            elif md == "rest_on":
                for i, w in ((2, 0.5), (3, 1.0)):
                    T.append((np.nonzero(h.sets["%s_0%d" % (f, i)])[0], 0.0040, w))
            elif md == "hook":
                for i in (2, 3):
                    T.append((np.nonzero(h.sets["%s_0%d" % (f, i)])[0], 0.0040, 0.5))
            elif md in ("spine", "string"):
                for i, w in ((2, 1.0), (3, 1.0)):
                    T.append((np.nonzero(h.sets["%s_0%d" % (f, i)])[0], 0.0007, w))
        if pname == "bow":
            T.append((np.nonzero(h.sets["thumb_01"])[0], 0.0006, 2.0))
            T.append((np.nonzero(h.sets["palm_core"])[0], 0.0010, 0.5))
        elif pname not in ("string", "bead"):
            T.append((np.nonzero(h.sets["palm_core"])[0], 0.0010, 1.0))
        return T

    FMODE_GAPS = {"wrap": ((0.0010, 0.0010, 0.0010), (0.3, 1.0, 1.0)), "rest_on": ((None, 0.0040, 0.0040), (0.0, 0.5, 1.0)),
                  "hook": ((None, 0.0040, 0.0040), (0.0, 1.0, 1.0)),
                  "spine": ((None, 0.0007, 0.0007), (0.0, 1.0, 1.0)), "string": ((None, 0.0007, 0.0007), (0.0, 1.0, 1.0))}

    def close_on_prop(self, ang, modes, prop, table):
        info = {}; total = 0.0
        done = []
        for f in FINGERS:
            md = modes[f]
            if md in self.FMODE_GAPS:
                g, w = self.FMODE_GAPS[md]
                kw = {"rest_on": {"lam": 2.0}, "string": {"free": (0, 1)}, "hook": {"kinds": ("c", "d")}}.get(md, {})
                if md == "string":
                    ang[f] = dict(ang[f], DIP=table[f]["DIP"])
                ang[f], c = self.refine_finger(ang, f, prop, done, gaps=g, weights=w, table=table[f], **kw)
                info[f] = {"mode": md, "cost": c}; total += c
            else:
                co, _, _ = self.posed(ang)
                o = self.obstacles(co, prop=prop if md != "hold_free" else None, fingers=done)
                ang[f], frac = self.open_until_clear(ang, f, o, joints=("MCP", "PIP", "DIP"))
                info[f] = {"mode": md, "kept": round(frac, 3)}
            done.append(f)
        return ang, info, total

    def post_place(self, ang, prop, modes, pname, table, spec):
        """after the placement search: small prop fit, re-close, the ice-pick cap alignment, then separate anything left
        inside the prop and re-close."""
        co, _, _ = self.posed(ang)
        prop = self.fit_prop(co, prop, self.grip_targets(modes, pname, co))
        ang, inf, _ = self.close_on_prop(ang, modes, prop, table)
        if spec["thumb"] == "cap":            # the butt end flush with the index's radial side (the thumb caps it)
            co, _, _ = self.posed(ang)
            top = max(p["t"][1] + p["h"] for p in prop.parts if p["k"] == "c" and p["s"] == "cyl")
            ymax = float(prop.local(co[self.h.sets["index"] | self.h.sets["thumb_01"]])[:, 1].max())
            M = prop.M.copy(); M[:3, 3] -= M[:3, 1] * (top - (ymax + 0.003))
            prm = prop.params; prop = Placed(prop.name, M); prop.params = dict(prm, cap_shift_mm=round((top - ymax - 0.003) * 1000, 1))
            ang, inf, _ = self.close_on_prop(ang, modes, prop, table)
        for _ in range(2):                    # nothing may stay inside the prop: separate, then re-close
            co, _, _ = self.posed(ang)
            if -float(prop.sdf(co, kinds=("c", "d")).min()) <= 0.0003:
                break
            T = [(i, g, w * 0.02) for i, g, w in self.grip_targets(modes, pname, co)]
            prm = prop.params; prop = self.fit_prop(co, prop, T); prop.params = dict(prm, separated=True)
            ang, inf, _ = self.close_on_prop(ang, modes, prop, table)
        return ang, prop, inf

    def gate_score(self, ang, prop, modes, spec):
        """lower = better: contact misses of the wrapped phalanges (mm beyond 3), prop penetration (mm beyond 0.5)."""
        co, _, _ = self.posed(ang)
        sd = prop.sdf(co); sda = prop.sdf(co, kinds=("c", "d"))
        e = 0.0
        for f in FINGERS:
            if modes[f] in ("wrap",):
                for i in (2, 3):
                    g = float(sd[self.h.sets["%s_0%d" % (f, i)]].min()) * 1000
                    e += max(0.0, g - 3.0) + max(0.0, -g - 0.5)
            elif modes[f] == "rest_on":
                g = float(sd[self.h.sets["%s_03" % f]].min()) * 1000
                e += max(0.0, g - 10.0) + max(0.0, 2.0 - g)
            elif modes[f] == "hook":
                g = min(float(sda[self.h.sets["%s_0%d" % (f, i)]].min()) for i in (2, 3)) * 1000
                e += max(0.0, g - 6.0)
        e += 3.0 * max(0.0, -float(sda.min()) * 1000 - 0.5)
        return e

    def solve(self, name):
        spec = PRESETS[name]
        table = spec["angles"]
        ang = json.loads(json.dumps(table))
        modes = spec["fingers"] if isinstance(spec["fingers"], dict) else {f: spec["fingers"] for f in FINGERS}
        info = {"fingers": {}, "thumb": {}, "search": []}
        prop = None
        pname = spec.get("prop")
        if pname == "string":
            modes = dict(modes, index="string", middle="string", ring="string")
        if pname and pname != "bead":
            if pname == "string":
                prop, _a = self.place_string(ang)
                ang, info["fingers"], _ = self.close_on_prop(ang, modes, prop, table)
            else:
                open_ang = json.loads(json.dumps(ang))
                for f in FINGERS:
                    if modes[f] in self.FMODE_GAPS:
                        open_ang[f].update({jn: 0.25 * table[f][jn] for jn in FJ})
                sp = PROPS[pname]["place"]; grid = sp.get("search", {})
                cands = []
                for al in grid.get("alpha", [sp["alpha"]]):
                    for sh in grid.get("shift", [sp.get("shift", 0.0)]):
                        for th in grid.get("theta", [sp["axis"]]):
                            pr = self.place_prop(pname, open_ang, alpha=al, shift=sh, theta=th)
                            a2, inf, tot = self.close_on_prop(json.loads(json.dumps(open_ang)), modes, pr, table)
                            dev = sum(((a2[f][jn] - table[f][jn]) / 15.0) ** 2 for f in FINGERS if modes[f] in self.FMODE_GAPS for jn in FJ)
                            sc = tot * 1e6 + 0.3 * dev
                            info["search"].append([round(al, 3), round(sh, 3), round(th, 1), round(sc, 3)])
                            cands.append((sc, pr, a2, inf))
                cands.sort(key=lambda c: c[0])
                finals = []
                for sc0, pr, a2, inf in cands[:3]:
                    a3, pr3, inf3 = self.post_place(json.loads(json.dumps(a2)), pr, modes, pname, table, spec)
                    finals.append((self.gate_score(a3, pr3, modes, spec), pr3, a3, inf3))
                finals.sort(key=lambda c: c[0])
                info["post"] = [round(f[0], 3) for f in finals]
                _, prop, ang, info["fingers"] = finals[0]
        else:
            ang, info["fingers"] = self.solve_fingers(ang, modes, None, pname)
        if pname == "bead":
            prop = self.place_bead(ang)
        tm = spec["thumb"]
        t = time.time()
        if tm == "hold":
            ang["thumb"], frac = self.thumb_hold(ang, prop)
            info["thumb"] = {"mode": tm, "kept": round(frac, 3)}
        else:
            best = None
            starts = [None, [35, 15, 25, 50, 30], [40, 35, 30, 30, 25], [30, 40, 45, 20, 20], [55, 30, 65, 30, 50], [45, 45, 50, 40, 40]]
            if tm == "rein":
                starts = [[ang["thumb"][k] for k in ("CMC_flex", "CMC_abd", "CMC_opp", "MCP", "IP")]] + starts
            for x0 in starts:
                th, fx, ne = self.thumb_search(ang, tm, prop, x0=x0)
                key = (not self.thumb_contact_ok(dict(ang, thumb=th), tm, prop), fx)
                if best is None or key < best[3]:
                    best = (th, fx, ne, key)
            ang["thumb"] = best[0]
            info["thumb"] = {"mode": tm, "objective": best[1], "evals": best[2]}
        info["thumb"]["t"] = round(time.time() - t, 2)
        if any(m == "palm" for m in modes.values()):
            self.fix_palm_crossings(ang, modes, info["fingers"])
        if prop is not None and hasattr(prop, "params"):
            info["prop"] = prop.params
        return ang, prop, info


# ---------------------------------------------------------------------------------------------------- measurements
def _pen(tree, pts):
    """(inside count < -0.5 mm, depth mm) of pts against a surface tree."""
    if tree is None or not len(pts):
        return 0, 0.0
    sd = signed_dist(tree, pts)
    return int((sd < -0.0005).sum()), round(max(0.0, -float(sd.min())) * 1000, 2)


def measure(sv, ang, prop=None, co=None, P=None):
    """G-H1..H4 / H7 numbers for one solved hand (one side). Distances in mm."""
    h, s = sv.h, sv.h.side
    if co is None:
        co, D, P = sv.posed(ang)
    T = lambda key: sv.tree(co, key)
    out = {"pen": {}, "cross": {}, "contact": {}, "angles": {}, "dev": {}}
    palm = T("palm")
    worst = 0.0
    for f in FINGERS:
        for i in (1, 2, 3):
            idx = sv.filt("%s_0%d" % (f, i), "palm")
            n_, d_ = _pen(palm, co[idx])
            out["pen"]["%s_P%d_palm" % (f, i)] = [n_, d_]; worst = max(worst, d_)
    pairs = [("index", "middle"), ("middle", "index"), ("middle", "ring"), ("ring", "middle"), ("ring", "pinky"), ("pinky", "ring")]
    for a, b in pairs:
        n_, d_ = _pen(T(b), co[sv.filt(a, b)])
        out["pen"]["%s_in_%s" % (a, b)] = [n_, d_]; worst = max(worst, d_)
    for b in FINGERS:
        n_, d_ = _pen(T(b), co[sv.filt("thumb", b)])
        out["pen"]["thumb_in_%s" % b] = [n_, d_]; worst = max(worst, d_)
        n_, d_ = _pen(T("thumb"), co[sv.filt(b, "thumb")])
        out["pen"]["%s_in_thumb" % b] = [n_, d_]; worst = max(worst, d_)
    n_, d_ = _pen(T("palm_core"), co[sv.filt("thumb", "palm_core")])
    out["pen"]["thumb_in_palm"] = [n_, d_]; worst = max(worst, d_)
    out["pen_worst_mm"] = round(worst, 2)
    # triangle crossings between non-adjacent segments (P2 / P3 vs palm; neighbour fingers except the P1-P1 web;
    # thumb P1 / P2 vs the fingers)
    cr = 0
    for f in FINGERS:
        for i in (2, 3):
            c = tri_cross(co, h.faces_of("%s_0%d" % (f, i)), h.faces_of("palm_core"))
            out["cross"]["%s_P%d_palm" % (f, i)] = c; cr += c
    for a, b in (("index", "middle"), ("middle", "ring"), ("ring", "pinky")):
        c = 0
        for i in (1, 2, 3):
            for j in (1, 2, 3):
                if i == 1 and j == 1:
                    continue
                c += tri_cross(co, h.faces_of("%s_0%d" % (a, i)), h.faces_of("%s_0%d" % (b, j)))
        out["cross"]["%s_%s" % (a, b)] = c; cr += c
    for b in FINGERS:
        c = tri_cross(co, h.faces_of("thumb"), h.faces_of(b)); out["cross"]["thumb_%s" % b] = c; cr += c
    out["cross_total"] = cr
    # prop
    if prop is not None:
        allv = np.arange(len(co))
        sd_c = prop.sdf(co); sd_all = prop.sdf(co, kinds=("c", "d"))
        out["prop_pen_mm"] = round(max(0.0, -float(sd_all.min())) * 1000, 2)
        out["prop_pen_verts"] = int((sd_all < -0.0005).sum())
        for f in FINGERS + ("thumb",):
            for i in ((1, 2, 3) if f != "thumb" else (2, 3)):
                key = "%s_0%d" % (f, i)
                out["contact"]["%s_P%d_prop" % (f, i)] = round(float(sd_c[h.sets[key]].min()) * 1000, 2)
                out["contact"]["%s_P%d_prop_all" % (f, i)] = round(float(sd_all[h.sets[key]].min()) * 1000, 2)
            out["contact"]["%s_pad_prop" % f] = round(float(sd_c[sv.pad_idx[f]].min()) * 1000, 2)
        out["contact"]["palm_prop"] = round(float(sd_c[h.sets["palm_core"]].min()) * 1000, 2)
        out["contact"]["thenar_prop"] = round(float(sd_c[h.sets["thumb_01"]].min()) * 1000, 2)
    # fist-type contacts
    for f in FINGERS:
        pad = sv.pad_idx[f]
        fi = np.intersect1d(pad, sv.filt("%s_03" % f, "palm"))
        out["contact"]["%s_pad_palm" % f] = round(float(signed_dist(palm, co[fi], 0.2).min()) * 1000, 2) if len(fi) else None
    pad = sv.pad_idx["thumb"]
    t3 = np.nonzero(h.sets["thumb_03"])[0]
    for f in ("index", "middle", "ring"):
        tr = T("%s_02" % f)
        out["contact"]["thumb_pad_%s_P2" % f] = round(float(signed_dist(tr, co[pad], 0.2).min()) * 1000, 2)
        out["contact"]["thumb_P3_%s_P2" % f] = round(float(signed_dist(tr, co[t3], 0.2).min()) * 1000, 2)
    # angles (the solved parameters) + geometric check of the flexion plane (the audit's 'dev')
    for f in FINGERS:
        out["angles"][f] = {k: round(float(v), 1) for k, v in ang[f].items()}
    out["angles"]["thumb"] = {k: round(float(v), 1) for k, v in ang["thumb"].items()}
    out["angles"]["ring_mc"] = ang.get("ring_mc", 0); out["angles"]["pinky_mc"] = ang.get("pinky_mc", 0)
    for f in FINGERS:
        pts = [P["%s_metacarpal%s" % (f, s)][:3, 3]] + [P["%s_0%d%s" % (f, i, s)][:3, 3] for i in (1, 2, 3)]
        Rb = P["%s_03%s" % (f, s)][:3, :3] @ h.rest["%s_03%s" % (f, s)][:3, :3].T
        pts.append(pts[-1] + Rb @ (h.tail["%s_03%s" % (f, s)] - h.head["%s_03%s" % (f, s)]))
        segs = [pts[i + 1] - pts[i] for i in range(4)]
        d = {}
        for j, jn in enumerate(FJ):
            par = ("%s_metacarpal" if j == 0 else "%s_0" + str(j)) % f + s
            Rp = P[par][:3, :3] @ h.rest[par][:3, :3].T
            ax = Rp @ h.axes["%s_0%d" % (f, j + 1)]["flex"]
            b_ = segs[j + 1]
            d[jn] = {"flex": round(signed_angle(segs[j], b_, ax), 1),
                     "dev": round(math.degrees(math.asin(np.clip(b_ @ ax / np.linalg.norm(b_), -1, 1))), 1)}
        out["dev"][f] = d
    return out


def gates(name, m, spec):
    """pass / fail per gate for one side's measurement."""
    g = {}
    g["H1_pen"] = m["pen_worst_mm"] <= 1.0 and m["cross_total"] == 0 and m.get("prop_pen_mm", 0.0) <= 0.5
    rom_ok = True
    for f in FINGERS:
        a = m["angles"][f]
        for jn in FJ:
            lo, hi = ROM[jn]
            rom_ok &= lo - 0.05 <= a[jn] <= hi + 0.05
        rom_ok &= abs(a.get("abd", 0)) <= ROM["abd"]
    t = m["angles"]["thumb"]
    rom_ok &= ROM["tMCP"][0] - 0.05 <= t["MCP"] <= ROM["tMCP"][1] + 0.05 and ROM["tIP"][0] - 0.05 <= t["IP"] <= ROM["tIP"][1] + 0.05
    g["H3_rom"] = bool(rom_ok)
    dev_ok = True
    for f in FINGERS:
        abd = abs(m["angles"][f].get("abd", 0) or 0)
        for jn in FJ:
            lim = 5.0 + (abd if jn == "MCP" else 0.0)
            dev_ok &= abs(m["dev"][f][jn]["dev"]) <= lim + 0.05
    g["H4_plane"] = bool(dev_ok)
    c = m["contact"]
    tm = spec["thumb"]; fm = spec["fingers"]
    ok = True
    if spec.get("prop") in ("hammer", "sword", "knife", "icepick", "shield_bar", "torch", "strap", "rein"):
        for f in FINGERS:
            if (fm if isinstance(fm, str) else fm[f]) == "wrap":
                ok &= -0.5 <= c["%s_P3_prop" % f] <= 3.0 and -0.5 <= c["%s_P2_prop" % f] <= 3.0
            if (fm if isinstance(fm, str) else fm[f]) == "hook":
                ok &= -0.5 <= min(c["%s_P2_prop_all" % f], c["%s_P3_prop_all" % f]) <= 6.0
        ok &= -0.5 <= c["palm_prop"] <= 3.0 or -0.5 <= c["thenar_prop"] <= 3.0
    if tm in ("cap", "blade", "pinch"):
        ok &= -0.5 <= c["thumb_pad_prop"] <= 3.0
    if tm == "rein":
        ok &= -0.5 <= min(c["thumb_pad_prop"], c["thumb_P3_prop"]) <= 3.0
    if tm == "wrap":           # the distal thumb on the prop, or closed over the index / middle P2 (plan section 7)
        ok &= (-0.5 <= min(c["thumb_pad_prop"], c["thumb_P3_prop"]) <= 3.0) or (0.0 <= min(c["thumb_P3_index_P2"], c["thumb_P3_middle_P2"]) <= 3.0)
    if tm == "fist":
        ok &= 0.0 <= c["thumb_pad_index_P2"] <= 3.0
    if tm == "fist_mid":
        ok &= 0.0 <= c["thumb_pad_middle_P2"] <= 3.0
    if fm == "palm" or (isinstance(fm, dict) and "palm" in fm.values()):
        for f in FINGERS:
            if (fm if isinstance(fm, str) else fm[f]) == "palm":
                ok &= c["%s_pad_palm" % f] is not None and -0.5 <= c["%s_pad_palm" % f] <= 3.0
    if spec.get("prop") == "bead":
        ok &= -0.5 <= c["index_pad_prop"] <= 1.5 and -0.5 <= c["thumb_pad_prop"] <= 1.5
    if spec.get("prop") == "string":
        for f in ("index", "middle", "ring"):
            ok &= -0.5 <= min(c["%s_P2_prop" % f], c["%s_P3_prop" % f]) <= 1.5
    g["H2_contact"] = bool(ok)
    return g


# ---------------------------------------------------------------------------------------------------- Blender posing
def pose_reset(rig):
    for pb in rig.pose.bones:
        pb.matrix_basis = Matrix.Identity(4)
    bpy.context.view_layer.update()


def pose_hand(rig, deltas, locs=None):
    """write delta quaternions ([w,x,y,z], bone-local) and pivot locations into the pose bones."""
    for n, q in deltas.items():
        pb = rig.pose.bones.get(n)
        if pb is None:
            continue
        pb.rotation_mode = "QUATERNION"
        pb.rotation_quaternion = Quaternion([float(v) for v in q])
        pb.location = Vector([float(v) for v in (locs or {}).get(n, (0.0, 0.0, 0.0))])
    bpy.context.view_layer.update()


_DATA = {}


def load_data(path=JSON_PATH):
    if path not in _DATA:
        _DATA[path] = json.load(open(path)) if os.path.exists(path) else None
    return _DATA[path]


def rig_kind(rig):
    k = rig.get("rts_kind") if hasattr(rig, "get") else None
    if k:
        return str(k)
    nm = rig.name[4:] if rig.name.startswith("rts_") else rig.name
    return nm


def apply(rig, preset, side="both", weight=1.0, kind=None, data=None):
    """Blender applicator: pose the finger bones of `rig` to a solved preset (plan section 8). rotation_quaternion =
    slerp(identity, delta, weight); location = the virtual-pivot translation for that rotation. Returns {bone: Quaternion}.
    `kind` defaults to the rig's body ('rts_<kind>'); a body that is not in the JSON (e.g. a proportion variant) uses the
    nearer solved body by hand size."""
    data = data or load_data()
    if data is None or preset not in data["presets"]:
        raise KeyError("hand preset %r not solved (run hand_poses.py solve)" % preset)
    kind = kind or rig_kind(rig)
    P = data["presets"][preset]
    if kind not in P["solved"]:
        B = rig.data.bones
        hs = float((Vector(B["middle_01_l"].head_local) - Vector(B["hand_l"].head_local)).length)
        kind = min(P["solved"], key=lambda k: abs(data["bodies"][k]["hand_len_m"] - hs))
    piv = data.get("pivots", {}).get(kind, {}) if P.get("pivots", True) else {}
    out = {}
    sides = {"both": ("_l", "_r"), "l": ("_l",), "r": ("_r",), "_l": ("_l",), "_r": ("_r",)}[side]
    for s in sides:
        for b, rec in P["solved"][kind][s].items():
            pb = rig.pose.bones.get(b)
            if pb is None:
                continue
            q = q_slerp(np.array([1.0, 0, 0, 0]), from_gltf(rec["delta"]), weight)
            pb.rotation_mode = "QUATERNION"
            pb.rotation_quaternion = Quaternion([float(v) for v in q])
            p = piv.get(b[:-2])
            if p is not None:
                p = np.array(p, float) * (np.array([-1.0, 1, 1]) if s == "_r" else 1.0)
                pb.location = Vector([float(v) for v in (p - q_mat(q) @ p)])
            out[b] = pb.rotation_quaternion.copy()
    bpy.context.view_layer.update()
    return out


# ---------------------------------------------------------------------------------------------------- renders
def _hand_view_setup(body, hand):
    """Workbench close-up setup: only the hand + wrist of `hand`'s side is drawn (Mask modifier after the armature)."""
    scn = bpy.context.scene
    scn.render.engine = "BLENDER_WORKBENCH"
    sh = scn.display.shading
    sh.light = "STUDIO"; sh.color_type = "VERTEX"; sh.show_cavity = True; sh.cavity_type = "WORLD"
    sh.cavity_ridge_factor = 0.6; sh.cavity_valley_factor = 0.8
    sh.show_shadows = False; sh.show_specular_highlight = True
    scn.display.shading.background_type = "VIEWPORT"
    try:
        scn.view_settings.view_transform = "Standard"
    except Exception:
        pass
    world = scn.world or bpy.data.worlds.new("w"); scn.world = world
    world.color = (0.16, 0.17, 0.19)
    for o in bpy.data.objects:
        if o.type == "MESH" and o is not body:
            o.hide_render = True
    me = body.data
    vg = body.vertex_groups.get("hands_view") or body.vertex_groups.new(name="hands_view")
    co = np.empty(len(me.vertices) * 3); me.vertices.foreach_get("co", co); co = co.reshape(-1, 3)
    w = hand.head["hand" + hand.side]
    keep = set(int(v) for v in hand.vidx) | set(np.nonzero(np.linalg.norm(co - w, axis=1) < 0.06)[0].tolist())
    vg.add(sorted(keep), 1.0, "REPLACE")
    mm = body.modifiers.get("hands_mask") or body.modifiers.new("hands_mask", "MASK")
    mm.vertex_group = "hands_view"
    ca = me.color_attributes.get("hand_col") or me.color_attributes.new("hand_col", "BYTE_COLOR", "POINT")
    me.color_attributes.active_color = ca
    me.color_attributes.default_color_name = ca.name
    return ca


SKIN = (0.80, 0.62, 0.52)


def _paint(body, hand, red=None, blue=None):
    me = body.data
    ca = me.color_attributes["hand_col"]
    cols = np.tile(np.array(SKIN + (1.0,)), (len(me.vertices), 1))
    if blue is not None:
        cols[hand.vidx[blue]] = (0.42, 0.55, 0.85, 1.0)
    if red is not None:
        cols[hand.vidx[red]] = (0.95, 0.15, 0.1, 1.0)
    ca.data.foreach_set("color", cols.astype(np.float32).ravel())


def _hex(c):
    c = c.lstrip("#"); return tuple(int(c[i:i + 2], 16) / 255.0 for i in (0, 2, 4))


def prop_objects(placed, name="prop"):
    """mesh objects for a placed prop (armature space = world here: the rig's matrix_world is identity)."""
    import bmesh
    objs = []
    for k, p in enumerate(placed.parts):
        bm = bmesh.new()
        if p["s"] == "cyl":
            bmesh.ops.create_cone(bm, cap_ends=True, segments=40, radius1=p["r"], radius2=p["r"], depth=2 * p["h"])
            bmesh.ops.rotate(bm, verts=bm.verts, cent=(0, 0, 0), matrix=Matrix.Rotation(math.radians(-90), 3, "X"))
        elif p["s"] == "box":
            bmesh.ops.create_cube(bm, size=1.0)
            bmesh.ops.scale(bm, vec=[2 * v for v in p["b"]], verts=bm.verts)
            if p.get("rr", 0) > 0.0005:
                bmesh.ops.bevel(bm, geom=list(bm.edges) + list(bm.verts), offset=min(p["rr"], min(p["b"]) * 0.95),
                                segments=3, affect="EDGES")
        else:
            bmesh.ops.create_uvsphere(bm, u_segments=24, v_segments=16, radius=p["r"])
        me = bpy.data.meshes.new("%s_%d" % (name, k)); bm.to_mesh(me); bm.free()
        ob = bpy.data.objects.new("%s_%d" % (name, k), me)
        bpy.context.scene.collection.objects.link(ob)
        Mp = np.eye(4); Mp[:3, :3] = q_mat(part_quat(p)); Mp[:3, 3] = p["t"]
        ob.matrix_world = Matrix((placed.M @ Mp).tolist())
        ca = me.color_attributes.new("hand_col", "BYTE_COLOR", "POINT")
        col = _hex(p["col"]) + (1.0,)
        ca.data.foreach_set("color", np.tile(np.array(col, np.float32), len(me.vertices)).ravel())
        me.color_attributes.active_color = ca
        me.color_attributes.default_color_name = ca.name
        for poly in me.polygons:
            poly.use_smooth = p["s"] != "box"
        objs.append(ob)
    return objs


def hand_camera(hand, co, view, res=(640, 640), extra_pts=None, lens=85):
    """a camera looking at the posed hand from a palm-frame direction."""
    d, r, n = hand.d, hand.r, hand.n
    dirs = {"palm": (unit(n + 0.45 * r + 0.25 * d), r), "thumb": (unit(r + 0.55 * n + 0.15 * d), unit(-d + 0.2 * n)),
            "back": (unit(-n + 0.45 * d + 0.35 * r), r), "front": (unit(d + 0.5 * n + 0.2 * r), r),
            "ulnar": (unit(-r + 0.4 * n + 0.2 * d), unit(-d)), "side": (unit(r + 0.1 * n), unit(-d))}
    v, up = dirs[view]
    pts = co if extra_pts is None else np.vstack([co, extra_pts])
    c = 0.5 * (pts.min(0) + pts.max(0))
    rad = float(np.linalg.norm(pts - c, axis=1).max())
    scn = bpy.context.scene
    cam = bpy.data.objects.get("HandCam")
    if cam is None:
        cam = bpy.data.objects.new("HandCam", bpy.data.cameras.new("HandCam")); scn.collection.objects.link(cam)
    cam.data.lens = lens
    dist = rad / math.tan(math.atan(18.0 / lens)) * 1.08
    loc = c + v * dist
    z = unit(loc - c); x = unit(np.cross(up, z)); y = np.cross(z, x)
    M = np.eye(4); M[:3, 0] = x; M[:3, 1] = y; M[:3, 2] = z; M[:3, 3] = loc
    cam.matrix_world = Matrix(M.tolist())
    cam.data.clip_start = 0.01; cam.data.clip_end = 10
    scn.camera = cam
    scn.render.resolution_x, scn.render.resolution_y = res
    scn.render.resolution_percentage = 100
    return cam


def render_to(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    bpy.context.scene.render.filepath = path
    bpy.ops.render.render(write_still=True)


# ---------------------------------------------------------------------------------------------------- legacy fist
def legacy_deltas(side):
    """the pre-round-6 viewer fist: localRot(X) 75 / 82 / 60 on every finger, 45 / 45 on thumb_02 / 03."""
    D = {}
    for f in FINGERS:
        for i, a in ((1, 75), (2, 82), (3, 60)):
            D["%s_0%d%s" % (f, i, side)] = q_axis((1, 0, 0), a)
    for i, a in ((2, 45), (3, 45)):
        D["thumb_0%d%s" % (i, side)] = q_axis((1, 0, 0), a)
    return D


# ---------------------------------------------------------------------------------------------------- CLI helpers
S_MIR = np.diag([-1.0, 1.0, 1.0])


def mirror_frame(M):
    """left armature-space frame -> right: world X mirror, then flip the frame's own X axis (props are symmetric in
    their local x), so the result stays a proper rotation."""
    R = np.eye(4); R[:3, :3] = S_MIR @ M[:3, :3] @ S_MIR; R[:3, 3] = S_MIR @ M[:3, 3]
    return R


def hands_for(rig, body):
    hl = Hand(rig, body, "_l"); hl.set_pivots()
    hr = Hand(rig, body, "_r")
    err = max(np.abs(hr.rest[n[:-2] + "_r"][:3, :3] - S_MIR @ hl.rest[n][:3, :3] @ S_MIR).max() for n in hl.names)
    hr.pivots = {b[:-2] + "_r": S_MIR @ p for b, p in hl.pivots.items()}
    return hl, hr, float(err)


def glb_json(path):
    import struct
    with open(path, "rb") as f:
        f.read(12); ln, ty = struct.unpack("<II", f.read(8)); return json.loads(f.read(ln))


def new_data():
    return {"version": 1, "skeleton": "rts_human", "generated_by": "scripts/hand_poses.py", "date": "",
            "convention": {
                "quat": "[x, y, z, w] (glTF order) everywhere",
                "delta": "rotation relative to the bone's rest pose = Blender PoseBone.rotation_quaternion (reordered); three.js: bone.quaternion = restQuaternion * delta",
                "local": "the full glTF node rotation (rest local * delta), for engines that set local rotations directly",
                "pivot": "virtual joint pivots (plan H-4 / RIG-2 workaround): pivots[kind][bone] = the rotation centre as an offset from the joint, in the bone's local rest frame (metres, left side; right = [-x, y, z]). For a rotation R (the slerped delta) the bone's translation becomes rest + restLocalRotation * (p - R p) (Blender: PoseBone.location = p - R p). Presets with 'pivots': false (fist_legacy) rotate about the rig joints.",
                "side": "authored on the left; right delta = [x, -y, -z, w] of the left (X-mirror of the symmetric rig; gate G-H5)",
                "angles": "anatomical degrees including the rest curl: flexion + palmar (0 = in line with the parent segment); abd + away from the middle finger; thumb CMC flex (across the palm) / abd (palmar) / opp (pronation), MCP, IP",
                "prop_frame": "prop frame in the hand bone's local space (Blender bone local = glTF joint local): +Y = grip axis (toward the thumb side), +Z = blade edge / distal (or palmar for straps and the bow riser); parts in that frame: cyl (r, h = half length along Y), box (b = half sizes, rr = edge rounding), sph (r); 'q_axis': 'z' turns a cylinder to lie along Z. 'k': 'c' = contact part, 'd' = decoration."},
            "bones": BONES, "order": ORDER, "auto": AUTO, "bodies": {}, "pivots": {}, "rest": {}, "axes": {}, "presets": {}}


def angles_clean(ang):
    out = {}
    for f in FINGERS:
        out[f] = {k: round(float(v), 2) for k, v in ang[f].items()}
    out["ring_mc"] = round(float(ang.get("ring_mc", 0) or 0), 2); out["pinky_mc"] = round(float(ang.get("pinky_mc", 0) or 0), 2)
    out["thumb"] = {k: round(float(v), 2) for k, v in ang["thumb"].items()}
    return out


def cmd_solve(kind, rig, body, names):
    t0 = time.time()
    apply_body_sizes(kind)
    hl, hr, merr = hands_for(rig, body)
    log("mirror check: right rest frames = mirror of left within %.2e" % merr)
    sv = Solver(hl); svr = Solver(hr)
    data = load_data() or new_data()
    data = json.loads(json.dumps(data))
    data["date"] = time.strftime("%Y-%m-%d %H:%M")
    data["order"] = ORDER; data["auto"] = AUTO; data["bones"] = BONES
    B = rig.data.bones
    data["bodies"][kind] = {"hand_len_m": round(float((Vector(B["middle_01_l"].head_local) - Vector(B["hand_l"].head_local)).length), 5),
                            "pivot_mm": {b[:-2]: round(float(np.linalg.norm(p)) * 1000, 2) for b, p in hl.pivots.items()},
                            "rest_curl_deg": {k: round(v, 2) for k, v in hl.rest_ang.items()},
                            "joint_depth_mm": {b[:-2]: [round(v[0] * 1000, 1), round(v[1] * 1000, 1)] for b, v in hl.joint_depth.items()}}
    data["pivots"][kind] = {b[:-2]: [round(float(x), 7) for x in p] for b, p in hl.pivots.items()}
    data["rest"][kind] = {n: to_gltf(hl.rest_local[n]) for n in hl.names if n[:-2] in BONES}
    data["rest"][kind].update({n: to_gltf(hr.rest_local[n]) for n in hr.names if n[:-2] in BONES})
    data["axes"][kind] = {b: {k: [round(float(x), 5) for x in (hl.rest[b + "_l"][:3, :3].T @ v)] for k, v in ax.items()}
                          for b, ax in hl.axes.items()}
    for name in names:
        t = time.time()
        spec = PRESETS[name]
        rec = data["presets"].get(name, {})
        old_body_parts = dict((rec.get("prop") or {}).get("body_parts", {}))
        rec.update({"label": spec["label"], "group": spec["group"], "source": spec["source"],
                    "start_angles": spec["angles"], "fingers": spec["fingers"], "thumb": spec["thumb"],
                    "pivots": not spec.get("legacy", False)})
        if spec.get("legacy"):
            Dl = legacy_deltas("_l"); prop = None; ang = None; info = {"legacy": True}
            hl_piv, hr_piv = hl.pivots, hr.pivots
            hl.pivots, hr.pivots = {}, {}
        else:
            ang, prop, info = sv.solve(name)
            Dl = hl.deltas(ang)
        Dr = {b[:-2] + "_r": mirror_delta(q) for b, q in Dl.items()}
        solved = {"_l": {}, "_r": {}}
        for s, D, h in (("_l", Dl, hl), ("_r", Dr, hr)):
            for b in BONES:
                n = b + s
                q = D.get(n, np.array([1.0, 0, 0, 0]))
                solved[s][n] = {"delta": to_gltf(q), "local": to_gltf(q_mul(h.rest_local[n], q))}
        rec.setdefault("solved", {})[kind] = solved
        if ang is not None:
            rec.setdefault("angles", {})[kind] = angles_clean(ang)
        # prop
        chk = {}
        if prop is not None:
            Mr = mirror_frame(prop.M)
            frames = {}
            for s, M in (("_l", prop.M), ("_r", Mr)):
                L = np.linalg.inv(hl.rest["hand_l"] if s == "_l" else hr.rest["hand_r"]) @ M
                frames[s] = {"parent": "hand" + s, "t": [round(float(x), 6) for x in L[:3, 3]], "q": to_gltf(mat_q(L))}
            parts_l = json.loads(json.dumps(prop.parts)); parts_r = mirror_parts(parts_l)
            rec["prop"] = {"name": prop.name, "parts": parts_l,
                           "frame": dict(rec.get("prop", {}).get("frame", {}) if rec.get("prop") else {}, **{kind: frames}),
                           "place": getattr(prop, "params", {})}
            if parts_r != parts_l:
                rec["prop"]["parts_r"] = parts_r
            bp = old_body_parts
            bp[kind] = {"_l": parts_l, "_r": parts_r}
            rec["prop"]["body_parts"] = bp
            props = {"_l": prop, "_r": Placed(prop.name, Mr, parts_r)}
        else:
            rec["prop"] = None
            props = {"_l": None, "_r": None}
        for s, D, svx in (("_l", Dl, sv), ("_r", Dr, svr)):
            co, P = svx.h.skin(D)
            ang_s = ang if ang is not None else None
            if ang_s is None:
                m = {"legacy": True}
                m.update(measure_legacy(svx, co))
            else:
                m = measure(svx, ang_s, props[s], co, P)
                m["gates"] = gates(name, m, spec)
            chk[s] = m
        # G-H5 symmetry: mirrored right joints vs left (armature space)
        Pl = hl.fk(Dl); Pr = hr.fk(Dr)
        sym = max(float(np.linalg.norm(S_MIR @ Pr[b + "_r"][:3, 3] - Pl[b + "_l"][:3, 3])) for b in BONES) * 1000
        chk["H5_sym_mm"] = round(sym, 4)
        chk["info"] = info
        rec.setdefault("check", {})[kind] = chk
        data["presets"][name] = rec
        gl = chk["_l"].get("gates", {}); gr = chk["_r"].get("gates", {})
        log("%-13s %5.1fs  L %s  R %s  sym %.3f mm  pen %s/%s  cross %s/%s" % (
            name, time.time() - t, "".join("+" if v else "-" for v in gl.values()), "".join("+" if v else "-" for v in gr.values()),
            sym, chk["_l"].get("pen_worst_mm"), chk["_r"].get("pen_worst_mm"), chk["_l"].get("cross_total"), chk["_r"].get("cross_total")))
        if spec.get("legacy"):
            hl.pivots, hr.pivots = hl_piv, hr_piv
    data["presets"] = {k: data["presets"][k] for k in ORDER if k in data["presets"]}
    os.makedirs(os.path.dirname(JSON_PATH), exist_ok=True)
    with open(JSON_PATH, "w") as f:
        json.dump(data, f, indent=1)
    _DATA.pop(JSON_PATH, None)
    log("wrote %s (%.0f s)" % (JSON_PATH, time.time() - t0))
    return data


def measure_legacy(sv, co):
    """the old fist: penetration numbers only (the audit's fist metrics, rest-filtered)."""
    h = sv.h
    palm = sv.tree(co, "palm")
    out = {"pen": {}}
    worst = 0.0
    for f in FINGERS:
        for i in (1, 2, 3):
            n_, d_ = _pen(palm, co[sv.filt("%s_0%d" % (f, i), "palm")])
            out["pen"]["%s_P%d_palm" % (f, i)] = [n_, d_]; worst = max(worst, d_)
    for a, b in (("index", "middle"), ("middle", "ring"), ("ring", "pinky")):
        n_, d_ = _pen(sv.tree(co, b), co[sv.filt(a, b)])
        out["pen"]["%s_in_%s" % (a, b)] = [n_, d_]; worst = max(worst, d_)
    out["pen_worst_mm"] = round(worst, 2)
    out["cross_total"] = sum(tri_cross(co, h.faces_of("%s_0%d" % (f, i)), h.faces_of("palm_core")) for f in FINGERS for i in (2, 3))
    out["cross_fingers"] = {"%s_%s" % (a, b): sum(tri_cross(co, h.faces_of("%s_0%d" % (a, i)), h.faces_of("%s_0%d" % (b, j)))
                                                  for i in (1, 2, 3) for j in (1, 2, 3) if not (i == 1 and j == 1))
                            for a, b in (("index", "middle"), ("middle", "ring"), ("ring", "pinky"))}
    return out


def deltas_from_json(rec, kind, side):
    return {b: from_gltf(v["delta"]) for b, v in rec["solved"][kind][side].items()}


def placed_from_json(rec, kind, side, hand):
    if not rec.get("prop"):
        return None
    fr = rec["prop"]["frame"][kind][side]
    L = np.eye(4); L[:3, :3] = q_mat(from_gltf(fr["q"])); L[:3, 3] = fr["t"]
    if kind in rec["prop"].get("body_parts", {}):
        parts = rec["prop"]["body_parts"][kind][side]
    else:
        parts = rec["prop"]["parts_r"] if side == "_r" and rec["prop"].get("parts_r") else rec["prop"]["parts"]
    return Placed(rec["prop"]["name"], hand.rest[fr["parent"]] @ L, parts)


def cmd_check(kind, rig, body, names):
    """gates from the JSON (not from solver memory) on both hands, plus the Blender applicator vs the numpy skinning."""
    data = load_data()
    hl, hr, _ = hands_for(rig, body)
    svs = {"_l": Solver(hl), "_r": Solver(hr)}
    rep = {"kind": kind, "date": time.strftime("%Y-%m-%d %H:%M"), "presets": {}, "failures": []}
    me = body.data
    for name in names:
        rec = data["presets"][name]
        spec = PRESETS[name]
        r = {}
        for s in SIDES:
            h = svs[s].h
            piv = h.pivots
            if not rec.get("pivots", True):
                h.pivots = {}
            D = deltas_from_json(rec, kind, s)
            co, P = h.skin(D)
            if rec.get("angles"):
                m = measure(svs[s], rec["angles"][kind], placed_from_json(rec, kind, s, h), co, P)
                m["gates"] = gates(name, m, spec)
                for g, ok in m["gates"].items():
                    if not ok:
                        rep["failures"].append("%s %s %s" % (name, s, g))
            else:
                m = measure_legacy(svs[s], co)
            r[s] = m
            h.pivots = piv
        # the Blender applicator (apply()) must reproduce the numpy skinning (vertex positions, both hands)
        pose_reset(rig)
        apply(rig, name, "both", 1.0, kind=kind, data=data)
        dg = bpy.context.evaluated_depsgraph_get(); ev = body.evaluated_get(dg); m2 = ev.to_mesh()
        ec = np.empty(len(m2.vertices) * 3); m2.vertices.foreach_get("co", ec); ec = ec.reshape(-1, 3)
        ev.to_mesh_clear()
        err = 0.0
        for s in SIDES:
            h = svs[s].h
            piv = h.pivots
            if not rec.get("pivots", True):
                h.pivots = {}
            co, _ = h.skin(deltas_from_json(rec, kind, s))
            err = max(err, float(np.abs(ec[h.vidx] - co).max()))
            h.pivots = piv
        pose_reset(rig)
        r["blender_vs_numpy_mm"] = round(err * 1000, 5)
        if err > 0.0001:
            rep["failures"].append("%s applicator mismatch %.4f mm" % (name, err * 1000))
        r["H5_sym_mm"] = rec["check"][kind]["H5_sym_mm"]
        if name != "fist_legacy" and r["H5_sym_mm"] > 0.5:
            rep["failures"].append("%s H5 symmetry %.3f mm" % (name, r["H5_sym_mm"]))
        if rec.get("prop") and rec["prop"]["name"] == "sword":
            gc = json.load(open(os.path.join(CH, "out", "grip_contract.json")))
            rr = gc["bodies"].get(kind, gc["bodies"].get("male"))["socket_weapon_r"]["radius"]
            grip = [p for p in rec["prop"]["parts"] if p["k"] == "c"][0]["r"]
            r["H7_sword_radius_mm"] = [round(grip * 1000, 2), round(rr * 1000, 2)]
            if abs(grip - rr) > 1e-6:
                rep["failures"].append("sword grip radius %.2f != grip_contract %.2f mm" % (grip * 1000, rr * 1000))
        rep["presets"][name] = r
        g = {s: "".join("+" if v else "-" for v in r[s].get("gates", {}).values()) for s in SIDES}
        log("check %-13s L %s R %s  pen %s/%s mm  cross %s/%s  prop %s/%s  bl-np %.5f mm" % (
            name, g["_l"], g["_r"], r["_l"]["pen_worst_mm"], r["_r"]["pen_worst_mm"], r["_l"]["cross_total"], r["_r"]["cross_total"],
            r["_l"].get("prop_pen_mm"), r["_r"].get("prop_pen_mm"), r["blender_vs_numpy_mm"]))
    rep["pass"] = not rep["failures"]
    os.makedirs(RDIR, exist_ok=True)
    with open(os.path.join(RDIR, "hands_report_%s.json" % kind), "w") as f:
        json.dump(rep, f, indent=1)
    log("check %s: %s (%d failures) -> %s" % (kind, "PASS" if rep["pass"] else "FAIL", len(rep["failures"]),
                                              os.path.join(RDIR, "hands_report_%s.json" % kind)))
    for fl in rep["failures"]:
        log("  FAIL", fl)
    return rep


# ---------------------------------------------------------------------------------------------------- render command
def cmd_render(kind, rig, body, names, views=("palm", "thumb", "back"), res=(560, 560)):
    """close-ups of every preset (left hand, the arm at rest, only the hand drawn) with its prop, + before / after of
    the fist with the finger vertices inside the palm painted red."""
    data = load_data()
    hl, hr, _ = hands_for(rig, body)
    sv = Solver(hl)
    _hand_view_setup(body, hl)
    out = {}
    for name in names:
        rec = data["presets"][name]
        pose_reset(rig)
        piv = hl.pivots
        if not rec.get("pivots", True):
            hl.pivots = {}
        D = deltas_from_json(rec, kind, "_l")
        co, P = hl.skin(D)
        pose_hand(rig, D, hl.locations(D))
        red = None
        if name in ("fist", "fist_legacy", "point"):
            palm = sv.tree(co, "palm")
            bad = []
            for f in FINGERS:
                for i in (1, 2, 3):
                    idx = sv.filt("%s_0%d" % (f, i), "palm")
                    sd = signed_dist(palm, co[idx])
                    bad.extend(idx[sd < -0.0005].tolist())
            for a, b in (("index", "middle"), ("middle", "ring"), ("ring", "pinky")):
                idx = sv.filt(a, b)
                sd = signed_dist(sv.tree(co, b), co[idx])
                bad.extend(idx[sd < -0.0005].tolist())
            red = np.array(sorted(set(bad)), int)
        _paint(body, hl, red=red)
        prop = placed_from_json(rec, kind, "_l", hl)
        objs = prop_objects(prop) if prop is not None else []
        extra = None
        if prop is not None:            # frame the hand and the prop parts near it
            pts = []
            for ob in objs:
                for v in ob.data.vertices:
                    w = np.array(ob.matrix_world @ v.co)
                    if np.linalg.norm(w - hl.head["middle_01_l"]) < 0.13:
                        pts.append(w)
            extra = np.array(pts) if pts else None
        out[name] = []
        vs = list(views) + (["xray", "xray_side"] if name in ("fist", "fist_legacy", "point") else [])
        sh = bpy.context.scene.display.shading
        for v in vs:
            sh.show_xray = v.startswith("xray"); sh.xray_alpha = 0.45
            hand_camera(hl, co[hl.sets["palm"] | hl.sets["index"] | hl.sets["middle"] | hl.sets["ring"] | hl.sets["pinky"] | hl.sets["thumb"]],
                        {"xray": "palm", "xray_side": "front"}.get(v, v), res=res, extra_pts=extra)
            fn = os.path.join(RDIR, "grips", "%s_%s_%s.png" % (kind, name, v))
            render_to(fn); out[name].append(fn)
        sh.show_xray = False
        for ob in objs:
            bpy.data.objects.remove(ob)
        hl.pivots = piv
    pose_reset(rig)
    body.modifiers.remove(body.modifiers["hands_mask"])
    return out


# ---------------------------------------------------------------------------------------------------- grips GLB
def cmd_bake(kind, rig, body, fps=30):
    """out/hand_grips_<kind>.glb: the rts_human skeleton (the base GLB's joint nodes and inverse binds, copied), a small
    skinned preview mesh (the body's two hands), and one 2-key clip per preset and side (hand_<preset>_<l|r>) with
    rotation channels on the finger bones and translation channels where the virtual pivots move a joint. The scene
    extras carry the hand_poses JSON."""
    import struct
    data = load_data()
    base = os.path.join(CH, "out", "base_%s.glb" % kind)
    with open(base, "rb") as f:
        blob = f.read()
    off = 12; J = None; BIN = None
    while off < len(blob):
        ln, ty = struct.unpack_from("<II", blob, off); off += 8
        if ty == 0x4E4F534A: J = json.loads(blob[off:off + ln])
        elif ty == 0x004E4942: BIN = blob[off:off + ln]
        off += ln
    skin = J["skins"][0]
    joints = skin["joints"]
    names = [J["nodes"][j]["name"] for j in joints]
    arm = next(i for i, n in enumerate(J["nodes"]) if "children" in n and joints[0] in n["children"] and "mesh" not in n and i not in joints)
    keep = [arm] + joints
    remap = {old: new for new, old in enumerate(keep)}
    nodes = []
    for old in keep:
        n = {k: v for k, v in J["nodes"][old].items() if k in ("name", "translation", "rotation", "scale", "children")}
        if "children" in n:
            n["children"] = [remap[c] for c in n["children"] if c in remap]
            if not n["children"]:
                del n["children"]
        nodes.append(n)
    buf = bytearray()
    views, accs = [], []

    def add(arr, comp, typ, target=None, minmax=False):
        a = np.ascontiguousarray(arr)
        while len(buf) % 4:
            buf.append(0)
        bv = {"buffer": 0, "byteOffset": len(buf), "byteLength": a.nbytes}
        if target:
            bv["target"] = target
        buf.extend(a.tobytes()); views.append(bv)
        acc = {"bufferView": len(views) - 1, "componentType": comp, "count": int(a.shape[0]), "type": typ}
        if minmax:
            acc["min"] = [float(x) for x in a.reshape(a.shape[0], -1).min(0)]
            acc["max"] = [float(x) for x in a.reshape(a.shape[0], -1).max(0)]
        accs.append(acc)
        return len(accs) - 1
    # inverse binds (copied)
    ia = J["accessors"][skin["inverseBindMatrices"]]; ibv = J["bufferViews"][ia["bufferView"]]
    st = ibv.get("byteOffset", 0) + ia.get("byteOffset", 0)
    ibm = np.frombuffer(BIN, np.float32, 16 * ia["count"], st).reshape(-1, 16)
    ibm_acc = add(ibm.astype(np.float32), 5126, "MAT4")
    # preview mesh: both hands of the body (rest), Blender -> glTF (x, z, -y)
    hl, hr, _ = hands_for(rig, body)
    me = body.data
    sel = np.unique(np.concatenate([hl.vidx, hr.vidx]))
    loc = {int(v): k for k, v in enumerate(sel)}
    co = np.array([me.vertices[int(v)].co for v in sel]); nr = np.array([me.vertices[int(v)].normal for v in sel])
    g2 = lambda a: np.stack([a[:, 0], a[:, 2], -a[:, 1]], 1).astype(np.float32)
    tris = []
    for p in me.polygons:
        vs = list(p.vertices)
        if all(v in loc for v in vs):
            for k in range(1, len(vs) - 1):
                tris.append([loc[vs[0]], loc[vs[k]], loc[vs[k + 1]]])
    gname = {g.index: g.name for g in body.vertex_groups}
    jidx = {n: i for i, n in enumerate(names)}
    JW = np.zeros((len(sel), 4), np.uint16); WW = np.zeros((len(sel), 4), np.float32)
    for k, v in enumerate(sel):
        gs = sorted([(g.weight, jidx[gname[g.group]]) for g in me.vertices[int(v)].groups if gname.get(g.group) in jidx], reverse=True)[:4]
        tot = sum(w for w, _ in gs) or 1.0
        for m, (w, j) in enumerate(gs):
            JW[k, m] = j; WW[k, m] = w / tot
    WW[:, 0] += 1.0 - WW.sum(1)
    prim = {"attributes": {"POSITION": add(g2(co), 5126, "VEC3", 34962, True), "NORMAL": add(g2(nr), 5126, "VEC3", 34962),
                           "JOINTS_0": add(JW, 5123, "VEC4", 34962), "WEIGHTS_0": add(WW, 5126, "VEC4", 34962)},
            "indices": add(np.array(tris, np.uint32).ravel(), 5125, "SCALAR", 34963), "material": 0}
    nodes.append({"name": "hands_preview", "mesh": 0, "skin": 0})
    # animations
    anims = []
    tkey = add(np.array([0.0, 1.0 / fps], np.float32), 5126, "SCALAR", None, True)
    node_of = {J["nodes"][j]["name"]: remap[j] for j in joints}
    for name in data["order"]:
        rec = data["presets"].get(name)
        if not rec or kind not in rec.get("solved", {}):
            continue
        for s in SIDES:
            ch, sm = [], []
            for b, v in rec["solved"][kind][s].items():
                nd = node_of.get(b)
                if nd is None:
                    continue
                q = np.array([v["local"], v["local"]], np.float32)
                sm.append({"input": tkey, "output": add(q, 5126, "VEC4"), "interpolation": "LINEAR"})
                ch.append({"sampler": len(sm) - 1, "target": {"node": nd, "path": "rotation"}})
                p = data["pivots"][kind].get(b[:-2]) if rec.get("pivots", True) else None
                if p is not None:
                    p = np.array(p) * (np.array([-1.0, 1, 1]) if s == "_r" else 1.0)
                    dq = from_gltf(v["delta"])
                    locv = p - q_mat(dq) @ p
                    rl = from_gltf(J["nodes"][[j for j in joints if J["nodes"][j]["name"] == b][0]]["rotation"])
                    t0 = np.array(J["nodes"][[j for j in joints if J["nodes"][j]["name"] == b][0]]["translation"])
                    tt = t0 + q_mat(rl) @ locv
                    sm.append({"input": tkey, "output": add(np.array([tt, tt], np.float32), 5126, "VEC3"), "interpolation": "LINEAR"})
                    ch.append({"sampler": len(sm) - 1, "target": {"node": nd, "path": "translation"}})
            anims.append({"name": "hand_%s_%s" % (name, s[1]), "channels": ch, "samplers": sm})
    while len(buf) % 4:
        buf.append(0)
    extras = json.loads(json.dumps(data))
    for pr in extras["presets"].values():            # this file's body only
        for key in ("solved", "angles", "check"):
            if key in pr:
                pr[key] = {k: v for k, v in pr[key].items() if k == kind}
        if pr.get("prop"):
            pr["prop"]["frame"] = {k: v for k, v in pr["prop"]["frame"].items() if k == kind}
    gl = {"asset": {"version": "2.0", "generator": "RTS_Models characters/scripts/hand_poses.py"},
          "scene": 0, "scenes": [{"name": "hand_grips_%s" % kind, "nodes": [0, len(nodes) - 1],
                                   "extras": {"rts_hand_poses": extras, "note": "skeleton = out/base_%s.glb (97 rts_human joints, same rest + inverse binds); clips hand_<preset>_<l|r>: finger rotations + virtual-pivot translations (keep joint translations when retargeting, or the fingers rotate about the rig joints and the fists / grips can enter the palm)" % kind}}],
          "nodes": nodes, "skins": [{"name": "rts_%s" % kind, "joints": [remap[j] for j in joints], "inverseBindMatrices": ibm_acc, "skeleton": remap[joints[0]]}],
          "meshes": [{"name": "hands_preview", "primitives": [prim]}],
          "materials": [{"name": "skin_preview", "pbrMetallicRoughness": {"baseColorFactor": [0.8, 0.62, 0.52, 1.0], "metallicFactor": 0.0, "roughnessFactor": 0.6}}],
          "animations": anims, "buffers": [{"byteLength": len(buf)}], "bufferViews": views, "accessors": accs}
    js = json.dumps(gl, separators=(",", ":")).encode()
    js += b" " * ((4 - len(js) % 4) % 4)
    outp = os.path.join(CH, "out", "hand_grips_%s.glb" % kind)
    with open(outp, "wb") as f:
        f.write(struct.pack("<III", 0x46546C67, 2, 12 + 8 + len(js) + 8 + len(buf)))
        f.write(struct.pack("<II", len(js), 0x4E4F534A)); f.write(js)
        f.write(struct.pack("<II", len(buf), 0x004E4942)); f.write(bytes(buf))
    log("wrote %s (%.2f MB, %d clips)" % (outp, os.path.getsize(outp) / 1e6, len(anims)))
    return outp


# ---------------------------------------------------------------------------------------------------- viewer module
JS_TEMPLATE = r"""/* RTS hand poses (user round 6, items 37-38): the anatomical fist and the grip presets for the rts_human skeleton.
   GENERATED by scripts/hand_poses.py js from assets/hand_poses.json (__DATE__) - do not edit by hand.

   window.RTS_HANDS = {
     data,                                  the preset JSON (per body: finger quaternions, virtual pivots, prop frames)
     apply(bones, side, preset, weight),    pure setter used by the pages' pose code (e.g. the Fists pose):
                                            bone.quaternion = rest * slerp(I, delta, weight) on the finger bones
     set(side, preset, weight),             manual override (the Hand pose picker); preset null = back to the pose
     pivotize(),                            apply the virtual-pivot translations now (for CPU measurements); returns undo()
     kindOf(bones), presets(), info()
   }
   Rendering: renderer.render is wrapped once the page has loaded. Just before each render the manual preset (if any)
   is written into the finger bones, and every finger joint whose rotation lies on the path of a preset applied through
   this module turns about its virtual pivot (plan H-4: the rig's finger joints sit 3-5 mm under the dorsal skin); right
   after the render the page's own values are restored, so pose code, clips, refit() and captures never see a change.
   The whole file is one function: it defines nothing but window.RTS_HANDS (the pages have globals named tick, bones, V,
   S, X ...). A missing file is harmless: the pages fall back to their old finger curl. */
(function () {
'use strict';
if (window.RTS_HANDS) return;
const DATA = __DATA__;
const PRESETS = DATA.presets, ORDER = DATA.order.filter(n => PRESETS[n]);
const GROUPS = [['basic', 'Basic'], ['power', 'Power grips'], ['weapon', 'Weapons'], ['bow', 'Bow'], ['shield', 'Shield'],
                ['precision', 'Precision'], ['other', 'Other'], ['legacy', 'Before (old fist)']];
const H = {manual: null, side: 'both', props: true, weight: 1, last: {}, kind: null, kindBones: null, saved: [],
           hooked: false, ui: null, propObjs: {}, err: null, frames: 0};
const T = () => window.THREE;
const sidesOf = s => (s === 'l' || s === '_l') ? ['_l'] : (s === 'r' || s === '_r') ? ['_r'] : ['_l', '_r'];
const pageBones = () => { try { return (typeof bones !== 'undefined' && bones && bones.hand_l) ? bones : null; } catch (e) { return null; } };

function kindOf(B) {
  if (H.kindBones === B && H.kind) return H.kind;
  let k = null;
  let root = B.hand_l; while (root.parent) root = root.parent;
  root.traverse(o => { if (!k && o.isSkinnedMesh) { const m = /(female|male)_body$/.exec(o.name || ''); if (m && DATA.bodies[m[1]]) k = m[1]; } });
  if (!k) {
    const THREE = T(), a = B.hand_l.getWorldPosition(new THREE.Vector3()), b = B.middle_01_l.getWorldPosition(new THREE.Vector3());
    const d = a.distanceTo(b); let best = 1e9;
    for (const kk in DATA.bodies) { const e = Math.abs(DATA.bodies[kk].hand_len_m - d); if (e < best) { best = e; k = kk; } }
  }
  H.kind = k; H.kindBones = B;
  return k;
}
const qOf = a => new (T().Quaternion)(a[0], a[1], a[2], a[3]);
function restQ(k, name, bone) {
  const r = (DATA.rest[k] || {})[name];
  return r ? qOf(r) : bone.quaternion.clone();
}
function solvedFor(P, k) { return P.solved[k] || P.solved[Object.keys(P.solved)[0]]; }

// pure setter (no positions): the pages' pose functions call it; clips captured from those poses carry the fingers
function apply(B, side, name, w) {
  const P = PRESETS[name]; B = B || pageBones();
  if (!P || !B || !B.hand_l) return false;
  const k = kindOf(B), sol = solvedFor(P, k), I = new (T().Quaternion)();
  w = w === undefined ? 1 : w;
  for (const s of sidesOf(side)) for (const b in sol[s]) {
    const bone = B[b]; if (!bone) continue;
    const d = qOf(sol[s][b].delta);
    bone.quaternion.copy(restQ(k, b, bone)).multiply(I.clone().slerp(d, w));
    H.last[b] = {d, piv: P.pivots !== false, name};
  }
  return true;
}
// is R = slerp(I, D, t) for some t in [0, 1.02]?  'id' when R is the identity
function onPath(R, D) {
  let rw = R.w, rx = R.x, ry = R.y, rz = R.z; if (rw < 0) { rw = -rw; rx = -rx; ry = -ry; rz = -rz; }
  const rn = Math.hypot(rx, ry, rz); if (rn < 1e-6) return 'id';
  let dw = D.w, dx = D.x, dy = D.y, dz = D.z; if (dw < 0) { dw = -dw; dx = -dx; dy = -dy; dz = -dz; }
  const dn = Math.hypot(dx, dy, dz); if (dn < 1e-9) return false;
  if ((rx * dx + ry * dy + rz * dz) / (rn * dn) < 0.9995) return false;
  return Math.atan2(rn, rw) <= Math.atan2(dn, dw) * 1.02 + 1e-4;
}
function save(bone) { if (!bone.__rtsSaved) { bone.__rtsSaved = true; H.saved.push([bone, bone.quaternion.clone(), bone.position.clone()]); } }
function pre() {
  const B = pageBones(); if (!B) return;
  const THREE = T(), k = kindOf(B);
  H.frames++;
  if (!H.propsBuilt) buildProps(B, k);
  const marks = {};
  if (H.manual && PRESETS[H.manual]) {
    const P = PRESETS[H.manual], sol = solvedFor(P, k), I = new THREE.Quaternion();
    for (const s of sidesOf(H.side)) for (const b in sol[s]) {
      const bone = B[b]; if (!bone) continue;
      save(bone);
      const d = qOf(sol[s][b].delta);
      bone.quaternion.copy(restQ(k, b, bone)).multiply(I.clone().slerp(d, H.weight));
      marks[b] = {d, piv: P.pivots !== false, name: H.manual};
    }
  }
  const piv = DATA.pivots[k] || {}, active = {_l: null, _r: null};
  for (const base in piv) for (const s of ['_l', '_r']) {
    const b = base + s, bone = B[b]; if (!bone) continue;
    const m = marks[b] || H.last[b]; if (!m) continue;
    const rq = restQ(k, b, bone), R = rq.clone().invert().multiply(bone.quaternion);
    const st = onPath(R, m.d); if (!st) continue;
    if (st !== 'id') active[s] = active[s] || m.name;
    if (st === 'id' || !m.piv) continue;
    save(bone);
    const p = piv[base], pv = new THREE.Vector3(s === '_r' ? -p[0] : p[0], p[1], p[2]);
    const loc = pv.clone().sub(pv.clone().applyQuaternion(R));
    bone.position.add(loc.applyQuaternion(rq));
  }
  for (const s of ['_l', '_r']) for (const n in H.propObjs) {
    const o = H.propObjs[n][s]; if (o) o.visible = H.props && active[s] === n;
  }
}
function post() {
  for (let i = H.saved.length - 1; i >= 0; i--) {
    const [bone, q, p] = H.saved[i]; bone.quaternion.copy(q); bone.position.copy(p); bone.__rtsSaved = false; bone.updateMatrixWorld(true);
  }
  H.saved.length = 0;
}
function pivotize() { pre(); return post; }

// ---- test props (procedural; frames in the hand bone's local space, parts in the prop frame)
function buildProps(B, k) {
  H.propsBuilt = true;
  const THREE = T();
  for (const n of ORDER) {
    const P = PRESETS[n]; if (!P.prop) continue;
    const fr = P.prop.frame[k] || P.prop.frame[Object.keys(P.prop.frame)[0]];
    H.propObjs[n] = {};
    for (const s of ['_l', '_r']) {
      const f = fr[s], hand = B[f.parent]; if (!hand) continue;
      const g = new THREE.Group(); g.name = 'rts_hand_prop_' + n + s;
      g.position.set(f.t[0], f.t[1], f.t[2]); g.quaternion.copy(qOf(f.q));
      const bp = (P.prop.body_parts || {})[k];
      for (const p of bp ? bp[s] : (s === '_r' && P.prop.parts_r) ? P.prop.parts_r : P.prop.parts) {
        let geo;
        if (p.s === 'cyl') geo = new THREE.CylinderGeometry(p.r, p.r, 2 * p.h, 32);
        else if (p.s === 'box') geo = new THREE.BoxGeometry(2 * p.b[0], 2 * p.b[1], 2 * p.b[2]);
        else geo = new THREE.SphereGeometry(p.r, 24, 16);
        const col = new THREE.Color(p.col); if (col.convertSRGBToLinear) col.convertSRGBToLinear();
        const steel = /^#a|^#b/.test(p.col) && p.s !== 'cyl';
        const m = new THREE.Mesh(geo, new THREE.MeshStandardMaterial({color: col, roughness: steel ? 0.35 : 0.75, metalness: steel ? 0.7 : 0.0}));
        m.position.set(p.t[0], p.t[1], p.t[2]);
        if (p.q) m.quaternion.copy(qOf(p.q)); else if (p.q_axis === 'z') m.rotation.x = Math.PI / 2;
        m.castShadow = true; m.receiveShadow = true; m.frustumCulled = false;
        g.add(m);
      }
      g.visible = false; hand.add(g); H.propObjs[n][s] = g;
    }
  }
}

// ---- UI: a 'Hand pose' picker under the pose controls (char viewer) or the animation controls (look-dev)
function buildUI() {
  if (H.ui || !document.body) return;
  const lookdev = !!document.getElementById('animSec');
  const anchor = document.getElementById('poses') || document.getElementById('animSec');
  if (!anchor) return;
  const box = document.createElement('div'); box.id = 'rtsHands';
  const lab = document.createElement(lookdev ? 'h2' : 'div'); lab.textContent = 'Hand pose';
  if (!lookdev) lab.className = 'lbl';
  const sel = document.createElement('select'); sel.id = 'rtsHandPreset'; sel.setAttribute('aria-label', 'Hand pose');
  sel.style.width = '100%';
  const o0 = document.createElement('option'); o0.value = ''; o0.textContent = 'Pose (auto)'; sel.appendChild(o0);
  for (const [g, gl] of GROUPS) {
    const ns = ORDER.filter(n => PRESETS[n].group === g); if (!ns.length) continue;
    const og = document.createElement('optgroup'); og.label = gl;
    for (const n of ns) { const o = document.createElement('option'); o.value = n; o.textContent = PRESETS[n].label + (PRESETS[n].prop ? ' *' : ''); og.appendChild(o); }
    sel.appendChild(og);
  }
  sel.onchange = () => { H.manual = sel.value || null; sync(); };
  const row = document.createElement('div'); row.className = 'row'; row.style.marginTop = '5px';
  const sideB = {};
  for (const [s, t] of [['l', 'Left'], ['r', 'Right'], ['both', 'Both']]) {
    const b = document.createElement('button'); b.type = 'button'; b.textContent = t; b.dataset.hside = s;
    b.onclick = () => { H.side = s; sync(); }; row.appendChild(b); sideB[s] = b;
  }
  const pb = document.createElement('button'); pb.type = 'button'; pb.textContent = 'Props';
  pb.onclick = () => { H.props = !H.props; sync(); }; row.appendChild(pb);
  const note = document.createElement('div'); note.style.cssText = 'font-size:11.5px;opacity:.75;margin:2px 0 6px';
  note.textContent = '* with its test prop. Pose (auto): the pose sets the fingers (Fists = Fist).';
  box.append(lab, sel, row, note);
  if (lookdev) anchor.after(box); else anchor.after(box);
  H.ui = {sel, sideB, pb};
  function sync() {
    sel.value = H.manual || '';
    for (const s in sideB) { const on = H.side === s; sideB[s].classList.toggle('on', on); sideB[s].setAttribute('aria-pressed', on); }
    pb.classList.toggle('on', H.props); pb.setAttribute('aria-pressed', H.props);
  }
  H.sync = sync; sync();
}
function set(side, name, w) {
  H.manual = name && PRESETS[name] ? name : null;
  if (side) H.side = side;
  if (w !== undefined) H.weight = w;
  if (H.sync) H.sync();
  return H.manual;
}
function hook() {
  if (H.hooked) return true;
  let R = null;
  try { R = renderer; } catch (e) { R = null; }
  if (!R || typeof R.render !== 'function') return false;
  const orig = R.render.bind(R);
  R.render = function (scene, camera) {
    let ok = false;
    try { pre(); ok = true; } catch (e) { if (!H.err) { H.err = e; console.warn('RTS_HANDS:', e); } }
    try { return orig(scene, camera); } finally { if (ok) post(); }
  };
  H.hooked = true;
  return true;
}
function start() {
  if (window.__TEMPLATE__) return;
  buildUI();
  if (window.__HAND__) set(window.__HANDSIDE__ || 'both', window.__HAND__);
  if (window.__HANDPROPS__ !== undefined) { H.props = !!window.__HANDPROPS__; if (H.sync) H.sync(); }
  if (!hook()) { let n = 0; const t = setInterval(() => { if (hook() || ++n > 400) clearInterval(t); }, 50); }
}
if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', start); else setTimeout(start, 0);
window.RTS_HANDS = {data: DATA, apply, set, pivotize, kindOf, tick: pre,
  presets: () => ORDER.slice(), info: () => ({manual: H.manual, side: H.side, props: H.props, kind: H.kind, hooked: H.hooked, frames: H.frames, error: H.err ? String(H.err) : null})};
})();
"""


def cmd_js(path=JSON_PATH, out=JS_PATH):
    data = json.load(open(path))
    slim = {k: data[k] for k in ("version", "skeleton", "date", "convention", "bones", "order", "auto", "bodies", "pivots", "rest")}
    slim["bodies"] = {k: {"hand_len_m": v["hand_len_m"]} for k, v in data["bodies"].items()}
    slim["presets"] = {}
    for n, p in data["presets"].items():
        q = {"label": p["label"], "group": p["group"], "pivots": p.get("pivots", True), "solved": {}}
        for kind, sides in p["solved"].items():
            q["solved"][kind] = {s: {b: {"delta": v["delta"]} for b, v in bs.items()} for s, bs in sides.items()}
        q["prop"] = None if not p.get("prop") else {k: p["prop"][k] for k in ("name", "parts", "parts_r", "body_parts", "frame") if k in p["prop"]}
        q["gates"] = {kind: {s: c[s].get("gates") for s in SIDES} for kind, c in p.get("check", {}).items()}
        slim["presets"][n] = q
    js = JS_TEMPLATE.replace("__DATA__", json.dumps(slim, separators=(",", ":"))).replace("__DATE__", data.get("date", ""))
    with open(out, "w") as f:
        f.write(js)
    log("wrote %s (%.0f KB)" % (out, os.path.getsize(out) / 1024))
    return out


# ---------------------------------------------------------------------------------------------------- parity reference
def cmd_parity(kind, rig, body):
    """Blender reference for gate G-H6: every preset applied to both hands with apply() (the Blender applicator), the
    world position of every finger joint and fingertip in glTF coordinates (x, z, -y) ->
    renders/poses_hands/hands/parity_blender_<kind>.json (compared with the page by tools/compare_parity.py)."""
    data = load_data()
    g = lambda v: [float(v[0]), float(v[2]), -float(v[1])]
    out = {"kind": kind, "tip_len": {}, "presets": {}}
    for s in SIDES:
        for b in ("index_03", "middle_03", "ring_03", "pinky_03", "thumb_03"):
            out["tip_len"][b + s] = float(rig.data.bones[b + s].length)
    for name in data["order"]:
        if name not in data["presets"]:
            continue
        pose_reset(rig)
        apply(rig, name, "both", 1.0, kind=kind, data=data)
        r = {}
        for s in SIDES:
            for b in BONES:
                pb = rig.pose.bones[b + s]
                r[b + s] = g(pb.head)
                if b.endswith("_03"):
                    r[b + s + "_tip"] = g(pb.tail)
        out["presets"][name] = r
    pose_reset(rig)
    os.makedirs(RDIR, exist_ok=True)
    fn = os.path.join(RDIR, "parity_blender_%s.json" % kind)
    json.dump(out, open(fn, "w"))
    log("wrote", fn)


# ---------------------------------------------------------------------------------------------------- main
def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    if bpy is None:
        if "js" in argv:
            cmd_js(); return 0
        print(__doc__); return 2
    kind = argv[0]
    cmds = [a for a in argv[1:] if "=" not in a]
    opts = dict(a.split("=", 1) for a in argv[1:] if "=" in a)
    names = opts["presets"].split(",") if "presets" in opts else ORDER
    rig = bpy.data.objects["rts_" + kind]; body = bpy.data.objects[kind + "_body"]
    ok = True
    for c in cmds:
        if c == "solve":
            cmd_solve(kind, rig, body, names)
        elif c == "check":
            ok &= cmd_check(kind, rig, body, [n for n in names if n in load_data()["presets"]])["pass"]
        elif c == "render":
            cmd_render(kind, rig, body, [n for n in names if n in load_data()["presets"]])
        elif c == "bake":
            cmd_bake(kind, rig, body)
        elif c == "js":
            cmd_js()
        elif c == "parity":
            cmd_parity(kind, rig, body)
        else:
            raise SystemExit("unknown command %r" % c)
    return 0 if ok else 1


if __name__ == "__main__":
    rc = main()
    if rc:
        sys.exit(rc)
