"""Pose QA + the Python twin of the viewers' foot-ground contact solver (POSES builder, user round 6 items 35 / 36;
plan: POSES_HANDS_PLAN.md sections 4, 5, 10).

The pose table (every viewer pose: aims, world rotations, leg solver specs, the walk) lives ONCE, as JSON, inside
viewer/char_viewer_template.html between /*POSE_TABLE_BEGIN*/ and /*POSE_TABLE_END*/; viewer/lookdev_template.html carries
the identical POSEKIT block (the solver in JS + the table), and this file reads the table from the template and runs
the same algorithm on a virtual copy of the rig (glTF frame: +Y up, the character faces +Z, +X is its left), then writes
the pose into Blender (pose bones' matrix_basis) so the real skinned mesh can be measured. Parity with the browser is
gated (G-P6, tools in renders/poses_hands/poses/tools/).

Solver (per pose, section 4 of the plan):
  * foot modes: flat (sole planted), heel_up (heel raised about a virtual MTP crease pivot, toes flat: the `ball` bone gets
    a pivot translation so the toes bend at the crease, not at the rig's mid-foot ball joint), heel (heel strike: toes up,
    rotating about the heel), free (swing / kick).
  * two-bone leg IK whose knee lies in the foot's vertical plane (knees track over the 2nd toe), keeping the rest knee
    hinge (thigh / calf roll follow the knee plane, the knee never bends sideways in the skin).
  * pelvis solved per pose type from anatomical targets (squat: shin tilt + knee flexion / thigh horizontal; split stance:
    rear knee height / hip extension + front shin tilt; single support: CoM over the standing foot), the trunk lean solved
    for balance (segment-mass CoM, Winter 2009) or set, then a skin fit: the posed plantar samples (lowest vertices per
    foot bin) put each planted foot at +0.5 mm, re-solving the leg IK.

run (Blender 5.2 headless, on the engine export .blend = the GLB content):
  Blender -b out/base_<kind>_export.blend --python-exit-code 1 -P scripts/pose_qa.py -- <kind> [options] [pose ...]
  options: out=<dir> (default renders/poses_hands/poses)  render (Workbench feet / side views, red = below the floor)
           list=char|lookdev|all (default all)   nowalk   parity=<web json from tools/measure_poses.js>
           dump (write the solved bone transforms for the browser cross-check)
  writes <out>/pose_qa_<kind>.json and exits 1 when a gate fails.
import: deform_test.py uses `Kit(rig, body, kind).apply(name, w)` (poses the Blender rig) and `load_table()`.
"""
import sys, os, json, math
import numpy as np

try:
    import bpy
    from mathutils import Matrix, Vector
except ImportError:            # plain python: only load_table / the math are usable
    bpy = None

CH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.path.join(CH, "scripts") not in sys.path:
    sys.path.insert(0, os.path.join(CH, "scripts"))
TEMPLATES = {"char": os.path.join(CH, "viewer", "char_viewer_template.html"),
             "lookdev": os.path.join(CH, "viewer", "lookdev_template.html")}
KIT_BEGIN, KIT_END = "/*POSEKIT_BEGIN*/", "/*POSEKIT_END*/"
TAB_BEGIN, TAB_END = "/*POSE_TABLE_BEGIN*/", "/*POSE_TABLE_END*/"
S2 = ("_l", "_r")
UP = np.array([0.0, 1.0, 0.0])
XA = np.array([1.0, 0.0, 0.0])
I4 = np.array([0.0, 0.0, 0.0, 1.0])


def sg(s):
    return 1.0 if s == "_l" else -1.0


def other(s):
    return "_r" if s == "_l" else "_l"


def load_table(path=None):
    src = open(path or TEMPLATES["char"]).read()
    a = src.index(TAB_BEGIN) + len(TAB_BEGIN)
    return json.loads(src[a:src.index(TAB_END)])


def kit_block(path):
    src = open(path).read()
    return src[src.index(KIT_BEGIN):src.index(KIT_END) + len(KIT_END)]


# ---------------------------------------------------------------- math: three.js conventions, quaternion = [x, y, z, w]
def v3(a):
    return np.array(a, dtype=float)


def nrm(v):
    v = np.asarray(v, float)
    n = math.sqrt(float(v @ v))
    return v / n if n > 0 else v.copy()


def qmul(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array([ax * bw + aw * bx + ay * bz - az * by, ay * bw + aw * by + az * bx - ax * bz,
                     az * bw + aw * bz + ax * by - ay * bx, aw * bw - ax * bx - ay * by - az * bz])


def qinv(q):
    return np.array([-q[0], -q[1], -q[2], q[3]])


def qrot(q, v):
    x, y, z, w = q
    vx, vy, vz = v
    ix = w * vx + y * vz - z * vy
    iy = w * vy + z * vx - x * vz
    iz = w * vz + x * vy - y * vx
    iw = -x * vx - y * vy - z * vz
    return np.array([ix * w + iw * -x + iy * -z - iz * -y, iy * w + iw * -y + iz * -x - ix * -z,
                     iz * w + iw * -z + ix * -y - iy * -x])


def qaxis(axis, deg):
    a = nrm(axis)
    h = math.radians(deg) / 2
    s = math.sin(h)
    return np.array([a[0] * s, a[1] * s, a[2] * s, math.cos(h)])


def qunit(a, b):
    """three.js Quaternion.setFromUnitVectors (minimal arc)."""
    r = float(a @ b) + 1
    if r < 2.220446049250313e-16:
        r = 0.0
        q = np.array([-a[1], a[0], 0.0, r]) if abs(a[0]) > abs(a[2]) else np.array([0.0, -a[2], a[1], r])
    else:
        q = np.array([a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0], r])
    return q / math.sqrt(float(q @ q))


def qslerp(a, b, t):
    """three.js Quaternion.slerp (a -> b)."""
    if t == 0:
        return np.array(a, float)
    if t == 1:
        return np.array(b, float)
    x, y, z, w = a
    cos_half = w * b[3] + x * b[0] + y * b[1] + z * b[2]
    if cos_half < 0:
        bw, bx, by, bz = -b[3], -b[0], -b[1], -b[2]
        cos_half = -cos_half
    else:
        bx, by, bz, bw = b
    if cos_half >= 1.0:
        return np.array([x, y, z, w], float)
    sqr_sin = 1.0 - cos_half * cos_half
    if sqr_sin <= 2.220446049250313e-16:
        s = 1 - t
        q = np.array([s * x + t * bx, s * y + t * by, s * z + t * bz, s * w + t * bw])
        return q / math.sqrt(float(q @ q))
    sin_half = math.sqrt(sqr_sin)
    half = math.atan2(sin_half, cos_half)
    ra = math.sin((1 - t) * half) / sin_half
    rb = math.sin(t * half) / sin_half
    return np.array([x * ra + bx * rb, y * ra + by * rb, z * ra + bz * rb, w * ra + bw * rb])


def qmat(m):
    """three.js Quaternion.setFromRotationMatrix; m = 3x3 (rows)."""
    m11, m12, m13 = m[0]
    m21, m22, m23 = m[1]
    m31, m32, m33 = m[2]
    tr = m11 + m22 + m33
    if tr > 0:
        s = 0.5 / math.sqrt(tr + 1.0)
        return np.array([(m32 - m23) * s, (m13 - m31) * s, (m21 - m12) * s, 0.25 / s])
    if m11 > m22 and m11 > m33:
        s = 2.0 * math.sqrt(1.0 + m11 - m22 - m33)
        return np.array([0.25 * s, (m12 + m21) / s, (m13 + m31) / s, (m32 - m23) / s])
    if m22 > m33:
        s = 2.0 * math.sqrt(1.0 + m22 - m11 - m33)
        return np.array([(m12 + m21) / s, 0.25 * s, (m23 + m32) / s, (m13 - m31) / s])
    s = 2.0 * math.sqrt(1.0 + m33 - m11 - m22)
    return np.array([(m13 + m31) / s, (m23 + m32) / s, 0.25 * s, (m21 - m12) / s])


def qframe(u, n):
    """rotation whose columns are (n, u, n x u): maps the frame (X = hinge n, Y = bone direction u) to world."""
    z = np.cross(n, u)
    return qmat(np.array([[n[0], u[0], z[0]], [n[1], u[1], z[1]], [n[2], u[2], z[2]]]))


def qmat3(q):
    x, y, z, w = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def angle(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    c = float(a @ b) / max(1e-12, math.sqrt(float(a @ a) * float(b @ b)))
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


def lerp(a, b, t):
    return np.asarray(a, float) + (np.asarray(b, float) - np.asarray(a, float)) * t


def smooth(e0, e1, x):
    t = min(1.0, max(0.0, (x - e0) / (e1 - e0)))
    return t * t * (3 - 2 * t)


def mirror_dir(d, s):
    return v3([d[0] * sg(s), d[1], d[2]])


def mirror_axis(a, s):
    return v3(a) if s == "_l" else v3([a[0], -a[1], -a[2]])


def b2g(v):            # Blender armature space -> glTF
    return np.array([v[0], v[2], -v[1]], dtype=float)


def g2b(v):
    return np.array([v[0], -v[2], v[1]], dtype=float)


CM = np.array([[1.0, 0, 0], [0, 0, 1.0], [0, -1.0, 0]])      # b2g as a matrix


# ---------------------------------------------------------------- virtual skeleton (glTF frame), three.js Object3D semantics
class Skel:
    """Local position / quaternion per bone, world = parent world * T(p) * R(q) (scale 1): the three.js scene graph of
    the viewers. Built from the Blender rest (armature space converted with CM, so local matrices equal Blender's)."""

    def __init__(self, rig):
        self.rig = rig
        bones = rig.data.bones
        self.names = [b.name for b in bones]
        self.parent = {b.name: (b.parent.name if b.parent else None) for b in bones}
        Wq, Wt = {}, {}
        for b in bones:
            M = b.matrix_local
            R = CM @ np.array([[M[i][j] for j in range(3)] for i in range(3)])
            Wq[b.name] = qmat(R)
            Wt[b.name] = b2g(M.translation)
        self.rq, self.rp = {}, {}
        for n in self.names:
            p = self.parent[n]
            if p is None:
                self.rq[n], self.rp[n] = Wq[n], Wt[n]
            else:
                iq = qinv(Wq[p])
                self.rq[n] = qmul(iq, Wq[n])
                self.rp[n] = qrot(iq, Wt[n] - Wt[p])
        self.reset()

    def reset(self):
        self.q = {n: self.rq[n].copy() for n in self.names}
        self.p = {n: self.rp[n].copy() for n in self.names}
        self._w = {}

    def world(self, n):
        w = self._w.get(n)
        if w is None:
            par = self.parent[n]
            if par is None:
                w = (self.q[n], self.p[n])
            else:
                pq, pt = self.world(par)
                w = (qmul(pq, self.q[n]), pt + qrot(pq, self.p[n]))
            self._w[n] = w
        return w

    def wpos(self, n):
        return self.world(n)[1]

    def wquat(self, n):
        return self.world(n)[0]

    def has(self, n):
        return n in self.q

    def set_world_q(self, n, q):
        par = self.parent[n]
        self.q[n] = q.copy() if par is None else qmul(qinv(self.wquat(par)), q)
        self._w = {}

    def set_world(self, n, pos, q):
        par = self.parent[n]
        if par is None:
            self.q[n], self.p[n] = q.copy(), v3(pos)
        else:
            pq, pt = self.world(par)
            iq = qinv(pq)
            self.q[n], self.p[n] = qmul(iq, q), qrot(iq, v3(pos) - pt)
        self._w = {}

    def local_rot(self, n, q):
        self.q[n] = qmul(self.q[n], q)
        self._w = {}

    def to_blender(self):
        """matrix_basis = rest local^-1 @ posed local (the locals equal Blender's: both worlds are converted by CM)."""
        rig = self.rig
        for pb in rig.pose.bones:
            n = pb.name
            if np.allclose(self.q[n], self.rq[n], atol=1e-12) and np.allclose(self.p[n], self.rp[n], atol=1e-12):
                pb.matrix_basis = Matrix.Identity(4)
                continue
            Lr = np.eye(4); Lr[:3, :3] = qmat3(self.rq[n]); Lr[:3, 3] = self.rp[n]
            Lp = np.eye(4); Lp[:3, :3] = qmat3(self.q[n]); Lp[:3, 3] = self.p[n]
            if self.parent[n] is None:          # root: undo the CM of its world
                C4 = np.eye(4); C4[:3, :3] = CM
                Lr, Lp = np.linalg.inv(C4) @ Lr, np.linalg.inv(C4) @ Lp
            pb.matrix_basis = Matrix((np.linalg.inv(Lr) @ Lp).tolist())
        bpy.context.view_layer.update()


# ---------------------------------------------------------------- the kit (mirror of POSEKIT in the templates)
class Kit:
    """The solver + interpreter. env: the Blender body (skinned samples) and rig; `apply(name, w)` poses the rig."""

    def __init__(self, rig, body, kind, table=None, hands=True):
        self.rig, self.body, self.kind = rig, body, kind
        self.T = table or load_table()
        self.C = self.T["consts"]
        self.sk = Skel(rig)
        self.cache = {}
        self.hand_mod = None
        if hands:                         # HANDS builder's applicator, only once its presets are solved
            try:
                import hand_poses
                data = hand_poses.load_data()
                if data and "fist" in data.get("presets", {}) and data["presets"]["fist"].get("solved"):
                    self.hand_mod = hand_poses
            except Exception:
                self.hand_mod = None
        self.pending_hands = []
        self._others_hidden = False
        self.rest_data()

    # ---------------- evaluation of the real mesh (Blender) = the viewers' CPU skinning of the same vertices
    def hide_others(self):
        if self._others_hidden:
            return
        for o in self.rig.children:
            if o.type == 'MESH' and o != self.body:
                o.hide_viewport = True
        self._others_hidden = True

    def eval_co(self):
        """posed body vertices, glTF frame (writes the virtual pose into Blender, drives the correctives)."""
        import chr_lib as CL
        self.hide_others()
        self.sk.to_blender()
        self.apply_hands_blender()
        CL.drive_correctives(self.rig, self.body, enabled=True)
        dg = bpy.context.evaluated_depsgraph_get()
        ev = self.body.evaluated_get(dg)
        me = ev.to_mesh()
        co = np.empty(len(me.vertices) * 3)
        me.vertices.foreach_get("co", co)
        ev.to_mesh_clear()
        co = co.reshape(-1, 3)
        return np.stack([co[:, 0], co[:, 2], -co[:, 1]], 1)

    def apply_hands_blender(self):
        for s, preset, w in self.pending_hands:
            self.hand_mod.apply(self.rig, preset, side=s, weight=w, kind=self.kind)
        if self.pending_hands:
            bpy.context.view_layer.update()

    # ---------------- rest data (from the rest skeleton + the rest mesh)
    def rest_data(self):
        sk, C = self.sk, self.C
        sk.reset()
        me = self.body.data
        co = np.empty(len(me.vertices) * 3)
        me.vertices.foreach_get("co", co)
        co = co.reshape(-1, 3)
        R = np.stack([co[:, 0], co[:, 2], -co[:, 1]], 1)
        self.REST = R
        self.P0, self.PQ0 = sk.wpos("pelvis").copy(), sk.wquat("pelvis").copy()
        self.stature = float(R[:, 1].max())
        self.SHW = float(np.linalg.norm(sk.wpos("upperarm_l") - sk.wpos("upperarm_r")))
        self.HIPW = float(np.linalg.norm(sk.wpos("thigh_l") - sk.wpos("thigh_r")))
        self.head0 = sk.wpos("head").copy()
        self.head_top0 = np.array([self.head0[0], self.stature, self.head0[2]])
        # dominant bone per vertex (buttock samples)
        gname = {g.index: g.name for g in self.body.vertex_groups}
        dom = []
        for v in me.vertices:
            best, bw = None, -1
            for g in v.groups:
                if g.weight > bw:
                    best, bw = gname.get(g.group), g.weight
            dom.append(best)
        self.DOM = np.array([d or "" for d in dom])
        F = {}
        for s in S2:
            d = {}
            for b in ("thigh", "calf", "foot", "ball"):
                d[b + "_p"] = sk.wpos(b + s).copy()
                d[b + "_q"] = sk.wquat(b + s).copy()
            d["L1"] = float(np.linalg.norm(d["calf_p"] - d["thigh_p"]))
            d["L2"] = float(np.linalg.norm(d["foot_p"] - d["calf_p"]))
            idx = np.nonzero((R[:, 1] < C["foot_y"]) & (R[:, 0] * sg(s) > 0.01))[0]
            P = R[idx]
            hb = P[np.argmin(P[:, 2])]
            # foot line = heel back -> 2nd toe tip (the ball bone's +Y ray, as far as the toe vertices reach)
            bdir = qrot(d["ball_q"], UP)
            tip = d["ball_p"] + bdir * float(((P - d["ball_p"]) @ bdir).max())
            f0 = nrm([tip[0] - hb[0], 0.0, tip[2] - hb[2]])
            proj = (P - hb) @ f0
            FL = float(proj.max())
            d.update(idx=idx, hb=hb, hbf=np.array([hb[0], 0.0, hb[2]]), f0=f0, FL=FL,
                     side0=nrm(np.cross(UP, f0)), turnout0=math.degrees(math.atan2(f0[0] * sg(s), f0[2])))
            d["frac"] = proj / FL
            d["pmtp"] = d["hbf"] + f0 * (C["mtp_frac"] * FL) + UP * (C["mtp_h"] * self.stature)
            d["pheel"] = d["hbf"] + f0 * (C["heel_frac"] * FL)
            d["toe_tip"] = P[np.argmax(proj)]
            d["toe2"] = tip
            d["samples"] = self.pick_samples(idx, P, proj / FL)
            # knee samples: in front of the rest knee joint
            kp = d["calf_p"]
            dk = np.linalg.norm(R - kp, axis=1)
            d["knee_idx"] = np.nonzero((dk < C["knee_r"]) & (R[:, 2] > kp[2]))[0]
            F[s] = d
        self.F = F
        hipy = 0.5 * (F["_l"]["thigh_p"][1] + F["_r"]["thigh_p"][1])
        but = np.isin(self.DOM, ["pelvis", "thigh_l", "thigh_r", "thigh_twist_01_l", "thigh_twist_01_r"])
        self.butt_idx = np.nonzero(but & (R[:, 1] > hipy - 0.25) & (R[:, 1] < hipy + 0.05))[0]
        self.rest_q = {n: sk.wquat(n).copy() for n in sk.names}

    def pick_samples(self, idx, P, frac):
        """per foot: the lowest `samp_n` vertices of each of `samp_bins` bins along the heel -> toe axis, plus the
        `samp_n` rearmost (heel back); duplicates (same position) count once. Identical rule in POSEKIT."""
        C = self.C
        nb, n = C["samp_bins"], C["samp_n"]
        keys = {}
        for k, i in enumerate(idx):
            key = tuple(np.round(P[k] * 1e5).astype(np.int64))
            if key not in keys:
                keys[key] = k
        ks = sorted(keys.values())
        chosen = set()
        bins = {}
        for k in ks:
            b = min(nb - 1, max(0, int(math.floor(frac[k] * nb))))
            bins.setdefault(b, []).append(k)
        for b, lst in bins.items():
            lst.sort(key=lambda k: (P[k][1], k))
            chosen.update(lst[:n])
        rear = sorted(ks, key=lambda k: (frac[k], k))[:n]
        chosen.update(rear)
        return np.array(sorted(idx[k] for k in chosen))

    # ---------------- ops
    def rotW(self, n, axis, deg):
        if not self.sk.has(n):
            return
        self.sk.set_world_q(n, qmul(qaxis(axis, deg), self.sk.wquat(n)))

    def aim(self, n, child, d, w=1.0):
        sk = self.sk
        if not (sk.has(n) and sk.has(child)):
            return
        cur = nrm(sk.wpos(child) - sk.wpos(n))
        q = qunit(cur, nrm(d))
        if w != 1.0:
            q = qslerp(I4, q, w)
        sk.set_world_q(n, qmul(q, sk.wquat(n)))

    def run_ops(self, ops, w):
        for op in ops:
            k = op[0]
            if k == "aim":
                self.aim(op[1], op[2], op[3], w)
            elif k == "aimS":
                for s in S2:
                    self.aim(op[1] + s, op[2] + s, mirror_dir(op[3], s), w)
            elif k == "rotW":
                self.rotW(op[1], op[2], op[3] * w)
            elif k == "rotWS":
                for s in S2:
                    self.rotW(op[1] + s, mirror_axis(op[2], s), op[3] * w)
            elif k == "localRot":
                if self.sk.has(op[1]):
                    self.sk.local_rot(op[1], qaxis(op[2], op[3] * w))
            elif k == "hands":
                for s in S2:
                    if self.hand_mod is not None:
                        self.pending_hands.append((s, op[1], w))
                    else:
                        self.legacy_hand(s, op[1], w)
            else:
                raise ValueError("unknown op %r" % (k,))

    def legacy_hand(self, s, preset, w):
        """the viewers' old Fists curl (local X), kept as the fallback when hand_poses is not available."""
        if preset not in ("fist", "fist_legacy"):
            return
        for f in ("index", "middle", "ring", "pinky"):
            for i, a in ((1, 75), (2, 82), (3, 60)):
                n = "%s_0%d%s" % (f, i, s)
                if self.sk.has(n):
                    self.sk.local_rot(n, qaxis(XA, a * w))
        for i, a in ((2, 45), (3, 45)):
            n = "thumb_0%d%s" % (i, s)
            if self.sk.has(n):
                self.sk.local_rot(n, qaxis(XA, a * w))

    # ---------------- foot placement: G = T(pos) * Yaw(yaw) * T(-heel_back_floor); mode rotation in the rest frame
    def place_xf(self, s, pl):
        """(qG, fn) where fn(X_rest_point) -> world point, and the mode rotation (q_mode, pivot) for the foot."""
        F = self.F[s]
        qG = qaxis(UP, pl["yaw"])
        if pl["mode"] == "swing":        # the ankle is placed directly (pivot = the ankle)
            pos = v3(pl["ankle"]) - qrot(qG, F["foot_p"] - F["hbf"])
        else:
            pos = v3(pl["pos"])

        def G(X):
            return pos + qrot(qG, v3(X) - F["hbf"])
        mode, ang = pl["mode"], pl.get("ang", 0.0)
        if mode == "heel_up":
            qm, piv = qaxis(F["side0"], ang), F["pmtp"]
        elif mode == "heel":
            qm, piv = qaxis(F["side0"], -ang), F["pheel"]
        elif mode == "swing":            # pitch about the ankle (+ = plantarflexion); pos places the ankle directly
            qm, piv = qaxis(F["side0"], ang), F["foot_p"]
        else:
            qm, piv = I4.copy(), F["foot_p"]

        def M(X):                        # rest point -> mode-rotated rest point
            return piv + qrot(qm, v3(X) - piv)
        return qG, G, qm, M

    def foot_targets(self, s, pl):
        F = self.F[s]
        qG, G, qm, M = self.place_xf(s, pl)
        At = G(M(F["foot_p"]))
        return At, qmul(qG, qmul(qm, F["foot_q"])), qG, G, qm, M

    # ---------------- two-bone IK, the knee in the foot's vertical plane
    def leg_ik(self, s, At, Q, m, fwd):
        """At: ankle target; the knee lies in the vertical plane through Q with horizontal normal m, on the `fwd` side.
        Sets thigh + calf world rotations (rest knee hinge kept). Returns the reach shortfall (m, 0 if reached)."""
        sk, F = self.sk, self.F[s]
        H = sk.wpos("thigh" + s)
        L1, L2 = F["L1"], F["L2"]
        D = At - H
        d = math.sqrt(float(D @ D))
        short = max(0.0, d - (L1 + L2) * 0.99999)
        d = min(max(d, abs(L1 - L2) + 1e-6), (L1 + L2) * 0.99999)
        u = D / max(1e-12, math.sqrt(float(D @ D)))
        a = (L1 * L1 - L2 * L2 + d * d) / (2 * d)
        r = math.sqrt(max(0.0, L1 * L1 - a * a))
        Cc = H + u * a
        e1 = nrm(np.cross(u, XA) if abs(u[0]) < 0.9 else np.cross(u, UP))
        e2 = np.cross(u, e1)
        g1, g2 = float(e1 @ m), float(e2 @ m)
        R0 = r * math.sqrt(g1 * g1 + g2 * g2)
        rhs = -float((Cc - Q) @ m)
        ph0 = math.atan2(g2, g1)
        cands = []
        if R0 > 1e-9 and abs(rhs) <= R0:
            dph = math.acos(rhs / R0)
            cands = [ph0 + dph, ph0 - dph]
        else:
            cands = [ph0 if rhs > 0 else ph0 + math.pi]
        best, bestv = None, -1e9
        for ph in cands:
            K = Cc + r * (math.cos(ph) * e1 + math.sin(ph) * e2)
            val = float((K - H) @ fwd)
            if val > bestv:
                best, bestv = K, val
        K = best
        At2 = H + u * d
        n = np.cross(At2 - H, K - H)
        if float(n @ n) < 1e-14:
            n = -np.cross(UP, fwd)
        n = nrm(n)
        ut, uc = nrm(K - H), nrm(At2 - K)
        u0t, u0c = nrm(F["calf_p"] - F["thigh_p"]), nrm(F["foot_p"] - F["calf_p"])
        n0 = -XA
        n0t, n0c = nrm(n0 - (n0 @ u0t) * u0t), nrm(n0 - (n0 @ u0c) * u0c)
        nt, nc = nrm(n - (n @ ut) * ut), nrm(n - (n @ uc) * uc)
        Qt = qmul(qframe(ut, nt), qinv(qframe(u0t, n0t)))
        Qc = qmul(qframe(uc, nc), qinv(qframe(u0c, n0c)))
        sk.set_world_q("thigh" + s, qmul(Qt, F["thigh_q"]))
        sk.set_world_q("calf" + s, qmul(Qc, F["calf_q"]))
        return short

    def pose_leg(self, s, pl):
        """IK + foot orientation + toe pivot for one planted / placed leg. Returns the reach shortfall."""
        sk, F = self.sk, self.F[s]
        At, qf, qG, G, qm, M = self.foot_targets(s, pl)
        if pl["mode"] == "swing":
            At = v3(pl["ankle"])
        Q = G(F["hbf"]) if pl["mode"] != "swing" else At
        fwd = qrot(qG, F["f0"])
        m = qrot(qG, F["side0"])
        short = self.leg_ik(s, At, Q, m, fwd)
        sk.set_world_q("foot" + s, qf)
        e = pl.get("mtp", pl["ang"] if pl["mode"] == "heel_up" else 0.0)
        if e and sk.has("ball" + s):
            # toe world = foot world * foot_rest^-1 * Rot_about(P_mtp, side0, -e) * toe_rest
            qfw, tfw = sk.world("foot" + s)
            qe = qaxis(F["side0"], -e)
            p_rest = F["pmtp"] + qrot(qe, F["ball_p"] - F["pmtp"])        # rest-frame toe head after the MTP bend
            rel_q = qmul(qfw, qinv(F["foot_q"]))
            pos = tfw + qrot(rel_q, p_rest - F["foot_p"])
            sk.set_world("ball" + s, pos, qmul(rel_q, qmul(qe, F["ball_q"])))
        return short

    def set_pelvis(self, P, dq):
        self.sk.set_world("pelvis", P, qmul(dq, self.PQ0))

    # ---------------- realize a solution at weight w (0 = rest, 1 = the solved pose)
    def realize(self, sol, w=1.0, fit=True, ops=None):
        sk, C = self.sk, self.C
        sk.reset()
        self.pending_hands = []
        self.reach_used = False
        P = lerp(self.P0, sol["P"], w)
        dq = qslerp(I4, sol["dqP"], w)
        self.set_pelvis(P, dq)
        for b, ax, deg in sol.get("spine", []):
            self.rotW(b, ax, deg * w)
        self.run_ops(ops if ops is not None else sol.get("ops", []), w)
        pls = {}
        for s, pl in sol.get("feet", {}).items():
            if pl["mode"] == "free":
                continue
            q = dict(pl)
            if pl["mode"] != "swing":
                q["pos"] = lerp(self.F[s]["hbf"], pl["pos"], w)
                q["yaw"] = pl["yaw"] * w
                q["ang"] = pl.get("ang", 0.0) * w
                if "mtp" in pl:
                    q["mtp"] = pl["mtp"] * w
            pls[s] = q
        self.pls = pls
        self.legs_all(P, dq)
        if fit and pls:
            for _ in range(C["fit_iters"]):
                co = self.eval_co()
                for s, q in pls.items():
                    if q["mode"] == "swing":
                        continue
                    mn = float(co[self.F[s]["samples"], 1].min())
                    q["pos"] = v3(q["pos"]) - np.array([0.0, mn - C["clear"], 0.0])
                P = self.legs_all(P, dq)
        self.P_final = P
        return P

    def legs_all(self, P, dq):
        """IK of every placed leg; lowers the pelvis by the reach shortfall (then re-solves) when a leg cannot reach."""
        for _ in range(4):
            short = 0.0
            for s, q in self.pls.items():
                short = max(short, self.pose_leg(s, q))
            if short <= 1e-6:
                break
            self.reach_used = True             # safety net only: the table's poses are tuned so that it never triggers
            P = P - np.array([0.0, short + 0.002, 0.0])
            self.set_pelvis(P, dq)
        return P

    # ---------------- measurements used by the solve (joints only, cheap)
    def leg_state(self, s):
        sk = self.sk
        H, K, A = sk.wpos("thigh" + s), sk.wpos("calf" + s), sk.wpos("foot" + s)
        pl = self.pls.get(s)
        fwd = qrot(qaxis(UP, pl["yaw"]), self.F[s]["f0"]) if pl else nrm([0, 0, 1])
        sh = K - A
        beta = math.degrees(math.atan2(float(sh @ fwd), float(sh @ UP)))
        kappa = angle(K - H, A - K)
        return dict(H=H, K=K, A=A, beta=beta, kappa=kappa, thigh_el=float(K[1] - H[1]))

    def trunk_lean(self):
        t = self.sk.wpos("neck_01") - self.sk.wpos("pelvis")
        return math.degrees(math.atan2(float(t[2]), float(t[1])))

    def foot_point(self, s, X):
        """a rest-space point carried rigidly by the posed foot bone."""
        qf, tf = self.sk.world("foot" + s)
        return tf + qrot(qmul(qf, qinv(self.F[s]["foot_q"])), v3(X) - self.F[s]["foot_p"])

    def com_seg(self):
        """segment-mass CoM (Winter 2009 Table 4.1) from the joints; the head CoM is 45 % of the way from the head
        joint to the rest top of the head, carried by the head bone."""
        sk = self.sk
        W = sk.wpos
        hipc = 0.5 * (W("thigh_l") + W("thigh_r"))
        shc = 0.5 * (W("upperarm_l") + W("upperarm_r"))
        tot = 0.497 * (0.5 * (hipc + shc))
        qh, th = sk.world("head")
        tot = tot + 0.081 * (th + qrot(qmul(qh, qinv(self.rest_q["head"])), 0.45 * (self.head_top0 - self.head0)))
        for s in S2:
            F = self.F[s]
            tot = tot + 0.028 * lerp(W("upperarm" + s), W("lowerarm" + s), 0.436)
            tot = tot + 0.016 * lerp(W("lowerarm" + s), W("hand" + s), 0.430)
            tot = tot + 0.006 * lerp(W("hand" + s), W("middle_01" + s), 0.6)
            tot = tot + 0.100 * lerp(W("thigh" + s), W("calf" + s), 0.433)
            tot = tot + 0.0465 * lerp(W("calf" + s), W("foot" + s), 0.433)
            tot = tot + 0.0145 * 0.5 * (self.foot_point(s, F["hb"]) + self.foot_point(s, F["toe2"]))
        return tot

    def feet_frac(self, c):
        """CoM position along the feet: 0 = the heel line, 1 = the 2nd-toe-tip line (mean of both feet)."""
        hb = 0.5 * (self.foot_point("_l", self.F["_l"]["hb"]) + self.foot_point("_r", self.F["_r"]["hb"]))
        tp = 0.5 * (self.foot_point("_l", self.F["_l"]["toe2"]) + self.foot_point("_r", self.F["_r"]["toe2"]))
        u = np.array([tp[0] - hb[0], 0.0, tp[2] - hb[2]])
        L = math.sqrt(float(u @ u))
        return float((c - hb) @ (u / L)) / L

    def head_pitch(self):
        u = qrot(qmul(self.sk.wquat("head"), qinv(self.rest_q["head"])), UP)
        return math.degrees(math.atan2(float(u[2]), float(u[1])))

    # ---------------- solve helpers
    def newton(self, f, x, h=1e-4, tol=0.01, it=25, maxstep=0.08):
        x = v3(x)
        for _ in range(it):
            r = v3(f(x))
            if float(np.abs(r).max()) < tol:
                break
            J = np.zeros((len(r), len(x)))
            for i in range(len(x)):
                e = np.zeros(len(x)); e[i] = h
                J[:, i] = (v3(f(x + e)) - r) / h
            try:
                dx = -np.linalg.solve(J, r) if J.shape[0] == J.shape[1] else -np.linalg.lstsq(J, r, rcond=None)[0]
            except np.linalg.LinAlgError:
                dx = -np.linalg.lstsq(J, r, rcond=None)[0]
            n = float(np.abs(dx).max())
            if n > maxstep:
                dx *= maxstep / n
            x = x + dx
        return x

    def spine_list(self, delta, look=None):
        """trunk lean delta (deg, + forward) over spine_01..03, then the neck / head counter-rotation for `look`."""
        out = [[b, [1.0, 0.0, 0.0], delta * wgt] for b, wgt in self.C["spine_w"]]
        return out

    def trunk_solve(self, sol, target, look):
        """find the spine delta for a trunk lean (deg) or for the CoM fraction along the feet ('balance', target=frac)."""
        base = sol.get("spine0", [])

        def lean_of(delta):
            sol["spine"] = base + self.spine_list(delta)
            self.realize(sol, 1.0, fit=False)
            return self.trunk_lean()

        def frac_of(delta):
            sol["spine"] = base + self.spine_list(delta)
            self.realize(sol, 1.0, fit=False)
            return self.feet_frac(self.com_seg())
        if target[0] == "lean":
            d = target[1] - lean_of(0.0)
            for _ in range(5):
                err = target[1] - lean_of(d)
                if abs(err) < 0.02:
                    break
                d += err
        else:
            lo, hi = -25.0, 85.0
            flo, fhi = frac_of(lo) - target[1], frac_of(hi) - target[1]
            d = 0.0
            if flo * fhi > 0:
                d = lo if abs(flo) < abs(fhi) else hi
            else:
                for _ in range(40):
                    d = 0.5 * (lo + hi)
                    fm = frac_of(d) - target[1]
                    if abs(fm) < 1e-4:
                        break
                    if (fm < 0) == (flo < 0):
                        lo, flo = d, fm
                    else:
                        hi = d
        sol["spine"] = base + self.spine_list(d)
        if look is not None:
            self.realize(sol, 1.0, fit=False)
            c = look - self.head_pitch()
            sol["spine"] += [[b, [1.0, 0.0, 0.0], c * wgt] for b, wgt in self.C["look_w"]]
        return d

    # ---------------- per pose type
    def thigh_vs_foot(self, s):
        """angle (deg) of the thigh's floor projection vs the heel -> 2nd-toe axis; + = the thigh points more inward."""
        F, W = self.F[s], self.sk.wpos
        hb, t2 = self.foot_point(s, F["hb"]), self.foot_point(s, F["toe2"])
        fa = nrm([t2[0] - hb[0], 0.0, t2[2] - hb[2]])
        th = W("calf" + s) - W("thigh" + s)
        thh = nrm([th[0], 0.0, th[2]])
        return math.degrees(math.atan2(float(np.cross(fa, thh) @ UP), float(fa @ thh))) * -sg(s)

    def solve_squat(self, L, ops):
        """flat feet, symmetric: pelvis (height, fore / aft) from the shin tilt + knee flexion (or thigh horizontal),
        optional heel width for thigh / foot alignment, trunk lean set / parallel to the shin / for balance."""
        feet = {}

        def set_feet(k):
            for s in S2:
                F = self.F[s]
                yaw = sg(s) * (L["yaw"] - F["turnout0"]) if "yaw" in L else 0.0
                x = F["hbf"][0] if k == "rest" else sg(s) * k * self.SHW / 2
                feet[s] = dict(mode="flat", pos=v3([x, 0.0, F["hbf"][2]]), yaw=yaw, ang=0.0)
        stance = L.get("stance", "rest")
        set_feet(1.0 if stance == "align" else stance)
        sol = dict(P=self.P0.copy(), dqP=qaxis(XA, L.get("pelvis_tilt", 0.0)), spine=[], feet=feet, ops=ops)
        second = ("thigh", 0.0) if L.get("thigh") == "horizontal" else ("knee", L.get("knee", 90.0))
        x0 = [self.P0[1] - 0.45, self.P0[2] - 0.12]

        def legs_for(beta_t):
            sol["spine"] = []

            def f(x):
                sol["P"] = v3([self.P0[0], x[0], x[1]])
                self.realize(sol, 1.0, fit=False)
                st = [self.leg_state(s) for s in S2]
                b = sum(t["beta"] for t in st) / 2
                if second[0] == "thigh":
                    e = sum(math.degrees(math.asin(max(-1.0, min(1.0, t["thigh_el"] / self.F[s]["L1"])))) for t, s in zip(st, S2)) / 2
                    return [b - beta_t, e - second[1]]
                return [b - beta_t, sum(t["kappa"] for t in st) / 2 - second[1]]
            x = self.newton(f, x0)
            sol["P"] = v3([self.P0[0], x[0], x[1]])

        def stance_for(beta_t):       # heel width so the thighs run along the feet (<= align_deg wider)
            lo, hi = L.get("stance_range", [0.85, 1.3])

            def ga(k):
                set_feet(k)
                legs_for(beta_t)
                return sum(self.thigh_vs_foot(s) for s in S2) / 2 - L.get("align_deg", 0.0)
            glo, ghi = ga(lo), ga(hi)
            if (glo < 0) == (ghi < 0):
                k = lo if abs(glo) < abs(ghi) else hi
            else:
                for _ in range(30):
                    k = 0.5 * (lo + hi)
                    gm = ga(k)
                    if abs(gm) < 0.01:
                        break
                    if (gm < 0) == (glo < 0):
                        lo, glo = k, gm
                    else:
                        hi = k
            set_feet(k)
            sol["stance"] = k

        def solve_at(beta_t, look):
            if stance == "align":
                stance_for(beta_t)
            legs_for(beta_t)
            tr = L.get("trunk", 0.0)
            if tr == "shin":                  # trunk parallel to the shin (Myer 2014)
                self.realize(sol, 1.0, fit=False)
                tr = sum(self.leg_state(s)["beta"] for s in S2) / 2
            self.trunk_solve(sol, ("frac", L["com"]) if tr == "balance" else ("lean", tr), look)
        look = L.get("look")
        if L.get("beta") == "balance":        # shin tilt solved for the CoM target (with the trunk rule above)
            def g(beta):
                solve_at(beta, None)
                self.realize(sol, 1.0, fit=False)
                return self.feet_frac(self.com_seg()) - L["com"]
            b0, b1 = 30.0, 38.0
            g0, g1 = g(b0), g(b1)
            for _ in range(12):
                if abs(g1) < 2e-4 or g1 == g0:
                    break
                b0, b1, g0 = b1, min(55.0, max(15.0, b1 - g1 * (b1 - b0) / (g1 - g0))), g1
                g1 = g(b1)
            sol["beta"] = b1
            solve_at(b1, look)
        else:
            solve_at(L["beta"], look)
        if L.get("stool"):
            self.realize(sol, 1.0, fit=True)
            sol["stool"] = self.stool_from_pose()
        return sol

    def stool_from_pose(self):
        co = self.eval_co()
        B = co[self.butt_idx]
        kz = 0.5 * (self.sk.wpos("calf_l")[2] + self.sk.wpos("calf_r")[2])
        m = (B[:, 2] < kz - self.C["stool_front"]) & (np.abs(B[:, 0]) < 0.2)
        top = float(B[m, 1].min())
        z0 = float(B[m, 2].min())
        return dict(top=round(top - self.C["clear"], 5), x=[-0.21, 0.21], z=[round(z0 - 0.04, 4), round(kz - self.C["stool_front"], 4)])

    def solve_split(self, L, ops):
        f = L["front"]
        r = other(f)
        Ff, Fr = self.F[f], self.F[r]
        dq = qaxis(XA, L.get("pelvis_tilt", 0.0))
        hr_off = qrot(dq, Fr["thigh_p"] - self.P0)
        hf_off = qrot(dq, Ff["thigh_p"] - self.P0)
        e = math.radians(L["rear_hip_ext"])
        th = L["rear_mtp"]
        contact = L["rear_knee_h"] == "contact"
        kh = self.C["kneel_h0"] if contact else L["rear_knee_h"] * self.stature
        yaw_r = sg(r) * (L.get("rear_yaw", 0.0) - Fr["turnout0"])
        yaw_f = sg(f) * (L.get("front_yaw", 0.0) - Ff["turnout0"])
        qGr, qGf = qaxis(UP, yaw_r), qaxis(UP, yaw_f)
        w = L.get("width", 1.0)
        sol = dict(P=self.P0.copy(), dqP=dq, spine=[], ops=ops, feet={})
        for it in range(self.C["kneel_iters"] if contact else 1):
            P = v3([self.P0[0], kh + Fr["L1"] * math.cos(e) - hr_off[1], self.P0[2]])
            Hr = P + hr_off
            Kr = Hr + Fr["L1"] * v3([0.0, -math.cos(e), -math.sin(e)])
            A_up = Fr["pmtp"] + qrot(qaxis(Fr["side0"], th), Fr["foot_p"] - Fr["pmtp"])      # rest-frame heel-up ankle
            ax = Fr["foot_p"][0] * w
            dy, dxx = A_up[1] - Kr[1], ax - Kr[0]
            dz = math.sqrt(max(0.0, Fr["L2"] ** 2 - dy * dy - dxx * dxx))
            Ar = v3([ax, A_up[1], Kr[2] - dz])
            pos_r = Ar - qrot(qGr, A_up - Fr["hbf"])
            pos_r[1] = 0.0
            Hf = P + hf_off
            ff = qrot(qGf, Ff["f0"])
            bf = math.radians(L["front_beta"])
            Afx, Afy = Ff["foot_p"][0] * w, Ff["foot_p"][1]
            Kx = Afx + Ff["L2"] * math.sin(bf) * ff[0]
            Ky = Afy + Ff["L2"] * math.cos(bf)
            rad = Ff["L1"] ** 2 - (Kx - Hf[0]) ** 2 - (Ky - Hf[1]) ** 2
            Kz = Hf[2] + math.sqrt(max(0.0, rad))
            Af = v3([Afx, Afy, Kz - Ff["L2"] * math.sin(bf) * ff[2]])
            pos_f = Af - qrot(qGf, Ff["foot_p"] - Ff["hbf"])
            pos_f[1] = 0.0
            sol["P"] = P
            sol["feet"] = {r: dict(mode="heel_up", pos=pos_r, yaw=yaw_r, ang=th),
                           f: dict(mode="flat", pos=pos_f, yaw=yaw_f, ang=0.0)}
            if contact:
                self.realize(sol, 1.0, fit=True, ops=[])
                co = self.eval_co()
                mn = float(co[Fr["knee_idx"], 1].min())
                kh -= mn - self.C["clear"]
        tr = L.get("trunk", 0.0)
        if tr == "balance":                 # CoM over the middle of the stance (front foot centre <-> rear toe pads)
            qf_, Gf, _, _ = self.place_xf(f, sol["feet"][f])
            qr_, Gr, _, _ = self.place_xf(r, sol["feet"][r])
            zt = 0.5 * (Gf(Ff["hbf"] + Ff["f0"] * (0.5 * Ff["FL"]))[2] + Gr(Fr["hbf"] + Fr["f0"] * (0.85 * Fr["FL"]))[2])
            lo, hi = L.get("trunk_range", [0.0, 12.0])

            def cz(lam):
                self.trunk_solve(sol, ("lean", lam), None)
                self.realize(sol, 1.0, fit=False)
                return float(self.com_seg()[2]) - zt
            flo, fhi = cz(lo), cz(hi)
            lam = lo if flo >= 0 else (hi if fhi <= 0 else None)
            if lam is None:
                for _ in range(30):
                    lam = 0.5 * (lo + hi)
                    fm = cz(lam)
                    if abs(fm) < 1e-4:
                        break
                    if fm < 0:
                        lo = lam
                    else:
                        hi = lam
            self.trunk_solve(sol, ("lean", lam), L.get("look"))
            sol["trunk"] = lam
        else:
            self.trunk_solve(sol, ("lean", tr), L.get("look"))
        sol["rear"] = r
        return sol

    def solve_single(self, L, ops):
        st = L["stand"]
        F = self.F[st]
        pos = F["hbf"].copy()
        if "stand_width" in L:                # standing heel under the hip (x = stand_width * half the hip width)
            pos[0] = sg(st) * L["stand_width"] * self.HIPW / 2
        sol = dict(P=self.P0.copy(), dqP=qaxis(XA, L.get("pelvis_tilt", 0.0)), spine=[], ops=ops,
                   feet={st: dict(mode="flat", pos=pos, yaw=0.0, ang=0.0), other(st): dict(mode="free")})
        # support centre: com fraction along the standing foot, on its centre line
        c_rest = pos + F["f0"] * (L["com"] * F["FL"])

        # knee flexion is singular near a straight leg: solve on the hip -> ankle distance for that flexion instead
        d_t = math.sqrt(F["L1"] ** 2 + F["L2"] ** 2 + 2 * F["L1"] * F["L2"] * math.cos(math.radians(L["knee"])))

        def f(x):
            sol["P"] = v3(x)
            self.realize(sol, 1.0, fit=False)
            c = self.com_seg()
            st_ = self.leg_state(st)
            return [(float(np.linalg.norm(st_["A"] - st_["H"])) - d_t) * 1000, (c[0] - c_rest[0]) * 1000, (c[2] - c_rest[2]) * 1000]
        x = self.newton(f, self.P0 + v3([sg(st) * 0.05, -0.03, 0.0]), tol=0.02)
        sol["P"] = v3(x)
        return sol

    def solve(self, name):
        if name in self.cache:
            return self.cache[name]
        spec = self.T["poses"][name]
        ops = spec.get("ops", [])
        L = spec.get("legs")
        if not L:
            sol = dict(P=self.P0.copy(), dqP=I4.copy(), spine=[], feet={}, ops=ops)
        elif L["type"] == "squat":
            sol = self.solve_squat(L, ops)
        elif L["type"] == "split":
            sol = self.solve_split(L, ops)
        elif L["type"] == "single":
            sol = self.solve_single(L, ops)
        else:
            raise ValueError(L["type"])
        sol["name"] = name
        self.cache[name] = sol
        return sol

    # ---------------- the walk (look-dev clip 'Walk (in place)'): treadmill contact phases
    def walk_sol(self, t):
        W, st = self.T["walk"], self.stature
        T, duty = W["period"], W["duty"]
        stride = 2 * W["step"] * st
        D = duty * stride
        pl_ = (t / T + W["phase0"]) % 1.0
        feet, ops = {}, []
        for s in S2:
            F = self.F[s]
            p = pl_ if s == "_l" else (pl_ + 0.5) % 1.0
            yaw = sg(s) * (W["turnout"] - F["turnout0"])
            z_hs = F["thigh_p"][2] - (F["foot_p"][2] - F["hbf"][2]) + D / 2 + W["ahead"] * st
            x = F["hbf"][0]
            if p < duty:
                pos = v3([x, 0.0, z_hs - D * p / duty])
                if p < W["flat_at"]:
                    feet[s] = dict(mode="heel", pos=pos, yaw=yaw, ang=W["heel_pitch"] * (1 - smooth(0.0, W["flat_at"], p)))
                elif p < W["heel_off"]:
                    feet[s] = dict(mode="flat", pos=pos, yaw=yaw, ang=0.0)
                else:
                    feet[s] = dict(mode="heel_up", pos=pos, yaw=yaw, ang=W["mtp_toe_off"] * smooth(W["heel_off"], duty, p))
            else:
                u = (p - duty) / (1 - duty)
                A_to = self.foot_targets(s, dict(mode="heel_up", pos=v3([x, 0.0, z_hs - D]), yaw=yaw, ang=W["mtp_toe_off"]))[0]
                A_hs = self.foot_targets(s, dict(mode="heel", pos=v3([x, 0.0, z_hs]), yaw=yaw, ang=W["heel_pitch"]))[0]
                e = smooth(0.0, 1.0, u)
                A = lerp(A_to, A_hs, e) + UP * (W["lift"] * st * math.sin(math.pi * u ** W["lift_skew"]))
                pitch = W["mtp_toe_off"] + (-W["heel_pitch"] - W["mtp_toe_off"]) * smooth(0.0, W["pitch_end"], u)
                feet[s] = dict(mode="swing", ankle=A, yaw=yaw, ang=pitch, mtp=W["mtp_toe_off"] * (1 - smooth(0.0, W["mtp_relax"], u)))
            c = -W["arm"] * math.cos(2 * math.pi * p)                                   # arm swing, + = forward
            ops.append(["aim", "upperarm" + s, "lowerarm" + s, [0.3 * sg(s), -math.cos(math.radians(c)), math.sin(math.radians(c))]])
            ee = c + W["elbow"] + W["elbow_fwd"] * max(0.0, -math.cos(2 * math.pi * p))
            ops.append(["aim", "lowerarm" + s, "hand" + s, [0.16 * sg(s), -math.cos(math.radians(ee)), math.sin(math.radians(ee))]])
        ph = 2 * math.pi * (pl_ - duty / 2)
        P = self.P0 + v3([W["sway"] * st * math.cos(ph), -W["drop"] * st + W["bob"] * st * math.cos(2 * ph), 0.0])
        dq = qaxis(UP, -W["yaw"] * math.cos(2 * math.pi * pl_))
        spine = [["spine_01", [1.0, 0.0, 0.0], W["lean"]], ["spine_03", [0.0, 1.0, 0.0], W["twist"] * math.cos(2 * math.pi * pl_)]]
        return dict(P=P, dqP=dq, spine=spine, feet=feet, ops=ops, name="Walk t=%.3f" % t)

    def walk_times(self):
        W = self.T["walk"]
        n = int(round(W["period"] * W["fps"])) + 1
        return [round(W["period"] * i / (n - 1), 4) for i in range(n)]      # 4 decimals: the browser probe's frame names

    # ---------------- public
    def apply(self, name, w=1.0, fit=True):
        """pose the Blender rig: resets, solves (cached), realizes at weight w, writes the pose, hands, correctives."""
        import chr_lib as CL
        sol = self.solve(name) if isinstance(name, str) else name
        self.realize(sol, w, fit=fit)
        self.sk.to_blender()
        self.apply_hands_blender()
        CL.drive_correctives(self.rig, self.body, enabled=True)
        return sol

    def restore_visibility(self):
        if self._others_hidden:
            for o in self.rig.children:
                if o.type == 'MESH' and o != self.body:
                    o.hide_viewport = False
            self._others_hidden = False


# ================================================================ QA (Blender)
REG = (("heel", 0.0, 0.30), ("sole", 0.30, 0.60), ("ball", 0.60, 0.80), ("toes", 0.80, 9.0))
BAND = (-0.0005, 0.004)


class QA:
    def __init__(self, kit, targets):
        self.K, self.TG = kit, targets
        me = kit.body.data
        self.polys = [list(p.vertices) for p in me.polygons]
        self.TRI = np.array([(p[0], p[k], p[k + 1]) for p in self.polys for k in range(1, len(p) - 1)])
        self.topo = None
        rig = kit.rig
        self.ball_len = {s: float(rig.data.bones["ball" + s].length) for s in S2}

    # ---- floor / contact per foot region
    def floor(self, co):
        K = self.K
        out = {"below": int((co[:, 1] < BAND[0]).sum()), "min_mm": round(float(co[:, 1].min()) * 1000, 2),
               "below_3mm": int((co[:, 1] < -0.003).sum())}
        for s in S2:
            F = K.F[s]
            y = co[F["idx"], 1]
            r = {}
            for nm, a, b in REG:
                m = (F["frac"] >= a) & (F["frac"] < b)
                yy = y[m]
                r[nm] = {"below": int((yy < BAND[0]).sum()), "contact": int(((yy >= BAND[0]) & (yy < BAND[1])).sum()),
                         "min_mm": round(float(yy.min()) * 1000, 2)}
            tip = int(np.argmax(F["frac"]))
            r["toe_tip_mm"] = round(float(y[tip]) * 1000, 2)
            out[s] = r
        return out

    # ---- leg angles (glTF frame, the audit's definitions; foot line = heel back -> 2nd toe tip as in the solver)
    def legs(self, co):
        K, sk = self.K, self.K.sk
        W = sk.wpos
        out = {}
        t = W("neck_01") - W("pelvis")
        out["trunk_lean"] = round(math.degrees(math.atan2(float(t[2]), float(t[1]))), 1)
        out["pelvis_y"] = round(float(W("pelvis")[1]), 3)
        qp = sk.wquat("pelvis")
        for s in S2:
            F = K.F[s]
            H, Kn, A, B = W("thigh" + s), W("calf" + s), W("foot" + s), W("ball" + s)
            # foot / toe vectors carried by their bones (the toe joint may carry a pivot translation)
            dqf = qmul(sk.wquat("foot" + s), qinv(F["foot_q"]))
            dqb = qmul(sk.wquat("ball" + s), qinv(F["ball_q"]))
            fv0, tv0 = F["ball_p"] - F["foot_p"], qrot(F["ball_q"], UP)
            fv, tv = qrot(dqf, fv0), qrot(dqb, tv0)
            d = {"knee_flex": round(angle(Kn - H, A - Kn), 1)}
            d["ankle_df"] = round(angle(F["calf_p"] - F["foot_p"], fv0) - angle(Kn - A, fv), 1)
            lat = qrot(dqf, F["side0"])

            def sag(v0, v1, ax):
                return math.degrees(math.atan2(float(np.cross(v0, v1) @ ax), float(v0 @ v1)))
            d["mtp_ext"] = round(-(sag(fv, tv, lat) - sag(fv0, tv0, F["side0"])), 1)
            tl = qrot(qinv(qp), Kn - H)
            t0 = qrot(qinv(K.PQ0), F["calf_p"] - F["thigh_p"])
            d["hip_flex"] = round(math.degrees(math.atan2(tl[2], -tl[1])) - math.degrees(math.atan2(t0[2], -t0[1])), 1)
            d["hip_abd"] = round(math.degrees(math.asin(max(-1, min(1, tl[0] * sg(s) / np.linalg.norm(tl))))) -
                                 math.degrees(math.asin(max(-1, min(1, t0[0] * sg(s) / np.linalg.norm(t0))))), 1)
            sh = Kn - A
            hb = K.foot_point(s, F["hb"])
            t2 = K.foot_point(s, F["toe2"])
            fa = nrm([t2[0] - hb[0], 0.0, t2[2] - hb[2]])
            d["shin_fwd"] = round(math.degrees(math.atan2(float(sh @ fa), float(sh @ UP))), 1)
            d["turnout"] = round(math.degrees(math.atan2(fa[0] * sg(s), fa[2])), 1)
            lat2 = nrm(np.cross(UP, fa)) * sg(s)                     # horizontal, toward the outside of the body
            d["knee_lateral_mm"] = round(float((Kn - hb) @ lat2) * 1000, 1)
            th = Kn - H
            d["thigh_vs_foot"] = round(K.thigh_vs_foot(s), 1) if math.hypot(th[0], th[2]) > 0.05 else None
            d["thigh_el"] = round(math.degrees(math.asin(max(-1, min(1, (Kn[1] - H[1]) / np.linalg.norm(th))))), 1)
            d["knee_y_mm"] = round(float(Kn[1]) * 1000, 1)
            y = co[F["idx"], 1]
            d["heel_mm"] = round(float(y[F["frac"] < 0.3].min()) * 1000, 1)
            out[s] = d
        hl, hr = K.foot_point("_l", K.F["_l"]["hb"]), K.foot_point("_r", K.F["_r"]["hb"])
        out["heel_width"] = round(float(np.linalg.norm((hl - hr)[[0, 2]])), 3)
        out["heel_width_over_shoulder"] = round(out["heel_width"] / K.SHW, 2)
        out["com_seg_frac"] = round(K.feet_frac(K.com_seg()), 3)
        return out

    # ---- mesh-volume CoM, support hull (feet within 4 mm of the floor; + knee / seat when used)
    def com_mesh(self, co):
        o = co.mean(0)
        a, b, c = co[self.TRI[:, 0]] - o, co[self.TRI[:, 1]] - o, co[self.TRI[:, 2]] - o
        v = (a * np.cross(b, c)).sum(1) / 6.0
        return o + ((a + b + c) / 4.0 * v[:, None]).sum(0) / v.sum()

    @staticmethod
    def hull(pts):
        P = sorted(map(tuple, pts))
        if len(P) < 3:
            return P
        def cross(o, a, b):
            return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
        lo, up = [], []
        for p in P:
            while len(lo) >= 2 and cross(lo[-2], lo[-1], p) <= 0:
                lo.pop()
            lo.append(p)
        for p in reversed(P):
            while len(up) >= 2 and cross(up[-2], up[-1], p) <= 0:
                up.pop()
            up.append(p)
        return lo[:-1] + up[:-1]

    def hull_margin(self, pts2, p):
        H = np.array(self.hull(pts2))
        if len(H) < 3:
            return -1e9
        inside, dmin = True, 1e9
        for i in range(len(H)):
            a, b = H[i], H[(i + 1) % len(H)]
            e = b - a
            n = np.array([e[1], -e[0]]) / np.linalg.norm(e)
            if (p - a) @ n > 0:
                inside = False
            t = np.clip((p - a) @ e / (e @ e), 0, 1)
            dmin = min(dmin, float(np.linalg.norm(p - (a + t * e))))
        return dmin if inside else -dmin

    def balance(self, co, extra_support=None, stand=None):
        K = self.K
        c = self.com_mesh(co)
        idx = np.concatenate([K.F[s]["idx"] for s in (S2 if stand is None else (stand,))])
        low = co[idx]
        sup = low[low[:, 1] < BAND[1]][:, [0, 2]]
        if extra_support is not None and len(extra_support):
            sup = np.concatenate([sup, extra_support])
        r = {"com": [round(float(x), 3) for x in c], "com_frac": round(K.feet_frac(c), 3)}
        r["margin_mm"] = round(self.hull_margin(sup, c[[0, 2]]) * 1000, 1) if len(sup) >= 3 else None
        if stand is not None:
            F = K.F[stand]
            y = co[F["idx"]]
            fc = y[y[:, 1] < BAND[1]]
            lat = float(abs(c[0] - fc[:, 0].mean()))
            r["lateral_from_foot_centre_mm"] = round(lat * 1000, 1)
        return r

    # ---- toe crease deformation (deform_metrics on a sphere at the virtual MTP pivot)
    def crease(self, co):
        import deform_metrics as DM
        K = self.K
        if self.topo is None:
            self.topo = DM.Topo(K.body)
            W, _ = DM.weight_matrix(K.body, K.rig)
            self.Wb = W / np.maximum(W.sum(1, keepdims=True), 1e-12)
        out = {}
        Xb = np.stack([co[:, 0], -co[:, 2], co[:, 1]], 1)              # back to Blender space
        A, _ = DM.blended_linear(self.Wb, DM.skin_matrices(K.rig))       # the pose is in Blender (eval_co wrote it)
        for s in S2:
            F = K.F[s]
            # the anatomical crease (72 % of the foot, 12 mm up), 40 mm sphere: dorsal toe base, toe webs, plantar crease
            cg = F["hbf"] + F["f0"] * (0.72 * F["FL"]) + UP * 0.012
            cb = g2b(cg)
            m = DM.region_mask(self.topo, cb, 0.04)
            posed_c = g2b(K.foot_point(s, cg))
            r = DM.measure(self.topo, Xb, m, cb, posed_c, A=A, isect=False)
            out[s] = {k: r[k] for k in ("fold_max", "folds", "flips", "comp_min") if k in r}
            if "fold_at" in r:
                v = K.REST[r["fold_at"]]
                out[s]["fold_at_frac"] = round(float((v - F["hb"]) @ F["f0"]) / F["FL"], 2)
                out[s]["fold_at_y_mm"] = round(float(v[1]) * 1000, 1)
        return out


def gate_pose(name, rep, tg, kind):
    """returns a list of failure strings for one pose (tg: assets/pose_targets.json entry)."""
    fails = []
    fl = rep["floor"]
    if fl["below"] > 0:
        fails.append("G-P1 %d verts below -0.5 mm (min %.1f mm)" % (fl["below"], fl["min_mm"]))
    feet = tg.get("feet", {})
    for s, mode in feet.items():
        r = fl[s]
        if mode == "flat":
            for reg in ("heel", "ball", "toes"):
                if r[reg]["contact"] < 5:
                    fails.append("G-P2 %s %s contact %d < 5" % (s, reg, r[reg]["contact"]))
        elif mode == "heel_up":
            for reg in ("ball", "toes"):
                if r[reg]["contact"] < 5:
                    fails.append("G-P2 %s %s contact %d < 5" % (s, reg, r[reg]["contact"]))
            if r["heel"]["min_mm"] < 30:
                fails.append("G-P2 %s heel only %.1f mm up" % (s, r["heel"]["min_mm"]))
            if r["toe_tip_mm"] - rep["rest_toe_tip_mm"][s] > 6:
                fails.append("G-P2 %s toe tip %.1f mm above its rest height" % (s, r["toe_tip_mm"] - rep["rest_toe_tip_mm"][s]))
            m = rep["legs"][s]["mtp_ext"]
            if not (35 <= m <= 65):
                fails.append("G-P3 %s MTP extension %.1f outside 35-65" % (s, m))
            c = rep.get("crease", {}).get(s)
            if c and (c["fold_max"] >= 60 or c["flips"] > 0):
                wv = (tg.get("_waivers") or {}).get("G-P3 crease")
                msg = "G-P3 %s toe crease fold %.1f flips %d (at %s of the foot, %s mm up)" % (s, c["fold_max"], c["flips"], c.get("fold_at_frac"), c.get("fold_at_y_mm"))
                if wv and c["flips"] <= wv["max_flips"]:
                    rep.setdefault("waived", []).append(msg + " [waived: RIG-5 toe-web weights]")
                else:
                    fails.append(msg)
        elif mode == "free":
            pass
    T = (tg.get("targets") or {}).get(kind, {})
    tol = tg.get("tol", {})
    L = rep["legs"]
    def chk(label, val, key, tkey=None):
        if key in T and val is not None:
            if abs(val - T[key]) > tol.get(tkey or key, 8):
                fails.append("G-P4 %s %.1f vs target %.1f +- %s" % (label, val, T[key], tol.get(tkey or key, 8)))
    for s in S2:
        sT = T.get(s, T)
        for key, lab in (("knee", "knee_flex"), ("shin", "shin_fwd"), ("hip_ext", "hip_flex")):
            if key in sT:
                v = L[s][lab] if key != "hip_ext" else -L[s]["hip_flex"]
                if abs(v - sT[key]) > tol.get(key, 8):
                    fails.append("G-P4 %s %s %.1f vs %.1f +- %s" % (s, key, v, sT[key], tol.get(key, 8)))
        if "turnout" in sT and feet.get(s) in ("flat", "heel_up"):
            if abs(L[s]["turnout"] - sT["turnout"]) > tol.get("turnout", 7):
                fails.append("G-P4 %s turnout %.1f vs %.1f" % (s, L[s]["turnout"], sT["turnout"]))
        lim = tg.get("limits", {})
        if L[s]["hip_flex"] > lim.get("hip_flex", 125):
            fails.append("G-P4 %s hip flexion %.1f > %s" % (s, L[s]["hip_flex"], lim.get("hip_flex", 125)))
        if -L[s]["hip_flex"] > lim.get("hip_ext", 30):
            fails.append("G-P4 %s hip extension %.1f > %s" % (s, -L[s]["hip_flex"], lim.get("hip_ext", 30)))
        if L[s]["ankle_df"] > lim.get("df", 45):
            fails.append("G-P4 %s ankle DF %.1f > %s" % (s, L[s]["ankle_df"], lim.get("df", 45)))
    if "trunk" in T:
        chk("trunk lean", L["trunk_lean"], "trunk")
    al = tg.get("align")
    if al:
        for s in S2:
            if abs(L[s]["knee_lateral_mm"]) > al.get("knee_mm", 15):
                fails.append("G-P5 %s knee %.1f mm off the heel -> 2nd-toe plane" % (s, L[s]["knee_lateral_mm"]))
            if L[s]["thigh_vs_foot"] is not None and abs(L[s]["thigh_vs_foot"]) > al.get("thigh_deg", 10):
                fails.append("G-P5 %s thigh vs foot %.1f deg" % (s, L[s]["thigh_vs_foot"]))
    bal = tg.get("balance")
    B = rep.get("balance", {})
    if bal:
        if "frac" in bal and not (bal["frac"][0] <= B["com_frac"] <= bal["frac"][1]):
            fails.append("G-P5 CoM at %.2f of the foot (want %s)" % (B["com_frac"], bal["frac"]))
        if "margin_mm" in bal and (B.get("margin_mm") is None or B["margin_mm"] < bal["margin_mm"]):
            fails.append("G-P5 CoM hull margin %s mm < %s" % (B.get("margin_mm"), bal["margin_mm"]))
        if "lateral_mm" in bal and B.get("lateral_from_foot_centre_mm", 1e9) > bal["lateral_mm"]:
            fails.append("G-P5 CoM %.1f mm lateral of the standing foot centre" % B["lateral_from_foot_centre_mm"])
    if tg.get("seat"):
        st = rep.get("seat", {})
        if st.get("contact", 0) < 5 or st.get("below", 1) > 0:
            fails.append("G-P5 seat contact %s below %s" % (st.get("contact"), st.get("below")))
    if tg.get("knee_contact"):
        kc = rep.get("knee_contact", {})
        if kc.get("contact", 0) < 3:
            fails.append("G-P2 kneeling knee contact %s" % kc.get("contact"))
    return fails


def main():
    import chr_lib as CL
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    kind = argv[0] if argv else "female"
    opts = {a.split("=")[0]: (a.split("=", 1)[1] if "=" in a else True) for a in argv[1:] if "=" in a or a in ("render", "nowalk", "dump", "nocheck")}
    only = [a for a in argv[1:] if "=" not in a and a not in opts]
    out = opts.get("out", os.path.join(CH, "renders", "poses_hands", "poses"))
    out = out if os.path.isabs(out) else os.path.join(CH, out)
    os.makedirs(out, exist_ok=True)
    table = load_table(opts.get("table"))
    targets = json.load(open(os.path.join(CH, "assets", "pose_targets.json")))
    rig = bpy.data.objects["rts_" + kind]
    body = bpy.data.objects[kind + "_body"]
    kit = Kit(rig, body, kind, table=table)
    qa = QA(kit, targets)
    rep = {"kind": kind, "hands_module": kit.hand_mod is not None, "rest": {}, "poses": {}, "walk": {}}
    # template parity of the kit block (the same code in both templates)
    try:
        rep["kit_blocks_identical"] = kit_block(TEMPLATES["char"]) == kit_block(TEMPLATES["lookdev"])
    except ValueError:
        rep["kit_blocks_identical"] = None
    for s in S2:
        F = kit.F[s]
        rep["rest"][s] = {"FL": round(F["FL"], 4), "turnout0": round(F["turnout0"], 1), "samples": int(len(F["samples"])),
                          "mtp_pivot_frac": kit.C["mtp_frac"], "mtp_pivot_h_mm": round(kit.C["mtp_h"] * kit.stature * 1000, 1),
                          "ball_joint_frac": round(float((F["ball_p"] - F["hb"]) @ F["f0"]) / F["FL"], 3)}
    rep["rest"].update(SHW=round(kit.SHW, 3), HIPW=round(kit.HIPW, 3), stature=round(kit.stature, 3))
    names = list(table["poses"].keys())
    lst = opts.get("list", "all")
    if lst == "char":
        names = table["lists"]["char"]
    elif lst == "lookdev":
        names = [k for _, k in table["lists"]["lookdev"]]
    if only:
        names = [n for n in names if n in only]
    fails_all = {}
    dump = {}
    kit.sk.reset()
    co0 = kit.eval_co()
    rest_tip = {s: round(float(co0[kit.F[s]["idx"][int(np.argmax(kit.F[s]["frac"]))], 1]) * 1000, 2) for s in S2}
    for name in names:
        sol = kit.apply(name)
        co = kit.eval_co()
        r = {"floor": qa.floor(co), "legs": qa.legs(co)}
        tg = dict(targets["poses"].get(name, {"feet": {}}), _waivers=targets.get("waivers", {}))
        stand = (tg.get("legs_type") == "single") and table["poses"][name]["legs"]["stand"] or None
        extra = None
        if sol.get("stool"):
            stl = sol["stool"]
            B = co[kit.butt_idx]
            m = (B[:, 0] > stl["x"][0]) & (B[:, 0] < stl["x"][1]) & (B[:, 2] > stl["z"][0]) & (B[:, 2] < stl["z"][1])
            dy = B[m, 1] - stl["top"]
            r["seat"] = {"top_m": stl["top"], "top_over_stature": round(stl["top"] / kit.stature, 3), "below": int((dy < BAND[0]).sum()),
                         "contact": int(((dy >= BAND[0]) & (dy < BAND[1])).sum())}
            extra = B[m][(dy < BAND[1])][:, [0, 2]]
        if sol.get("rear") and table["poses"][name]["legs"].get("rear_knee_h") == "contact":
            kk = co[kit.F[sol["rear"]]["knee_idx"], 1]
            r["knee_contact"] = {"min_mm": round(float(kk.min()) * 1000, 2), "contact": int(((kk >= BAND[0]) & (kk < BAND[1])).sum())}
            kn = co[kit.F[sol["rear"]]["knee_idx"]]
            extra = kn[kn[:, 1] < BAND[1]][:, [0, 2]]
        if table["poses"][name].get("legs"):
            r["balance"] = qa.balance(co, extra, stand)
            r["crease"] = qa.crease(co)
            r["solve"] = {"pelvis": [round(float(x), 4) for x in kit.sk.wpos("pelvis")], "beta": round(sol.get("beta", 0.0), 2),
                          "stance": round(sol["stance"], 3) if "stance" in sol else None, "trunk": sol.get("trunk")}
        r["rest_toe_tip_mm"] = rest_tip
        f = gate_pose(name, r, tg, kind) if not opts.get("nocheck") else []
        r["reach_fallback"] = bool(kit.reach_used)
        if kit.reach_used:
            f.append("reach fallback used (a planted leg could not reach its ankle target)")
        r["fails"] = f
        if f:
            fails_all[name] = f
        rep["poses"][name] = r
        dump[name] = {b: [round(float(x), 5) for x in kit.sk.wpos(b)] for b in DUMP_BONES if kit.sk.has(b)}
        dump[name]["__min_y_mm"] = r["floor"]["min_mm"]
        print("POSE %-14s below %4d min %6.1f mm  %s" % (name, r["floor"]["below"], r["floor"]["min_mm"], "; ".join(f) if f else "ok"))
    if not opts.get("nowalk") and not only:
        wt = targets.get("walk", {})
        for i, t in enumerate(kit.walk_times()):
            sol = kit.walk_sol(t)
            kit.apply(sol)
            co = kit.eval_co()
            fl = qa.floor(co)
            r = {"t": round(t, 4), "floor": fl, "modes": {s: sol["feet"][s]["mode"] for s in S2}, "legs": qa.legs(co),
                 "reach_fallback": bool(kit.reach_used)}
            f = []
            if kit.reach_used:
                f.append("reach fallback used (a planted leg could not reach its ankle target)")
            if fl["below"] > 0:
                f.append("G-P1 %d below (min %.1f mm)" % (fl["below"], fl["min_mm"]))
            for s in S2:
                pl = sol["feet"][s]
                y = co[kit.F[s]["idx"], 1]
                if pl["mode"] in ("flat", "heel", "heel_up") and y.min() > 0.004:
                    f.append("G-P2 %s stance foot %.1f mm up" % (s, y.min() * 1000))
                if pl["mode"] == "swing":
                    p = ((t / table["walk"]["period"] + table["walk"]["phase0"] + (0 if s == "_l" else 0.5)) % 1.0)
                    u = (p - table["walk"]["duty"]) / (1 - table["walk"]["duty"])
                    r.setdefault("swing_clear_mm", {})[s] = round(float(y.min()) * 1000, 1)
                    if wt.get("swing_window", [0.2, 0.8])[0] <= u <= wt.get("swing_window", [0.2, 0.8])[1] and y.min() < wt.get("clear", 0.015):
                        f.append("G-P2 %s swing clearance %.1f mm" % (s, y.min() * 1000))
            r["fails"] = f
            if f:
                fails_all["Walk t=%.3f" % t] = f
            rep["walk"]["%.4f" % t] = r
            dump["Walk t=%.4f" % t] = {b: [round(float(x), 5) for x in kit.sk.wpos(b)] for b in DUMP_BONES if kit.sk.has(b)}
            dump["Walk t=%.4f" % t]["__min_y_mm"] = fl["min_mm"]
        print("WALK %d frames, %d failing" % (len(rep["walk"]), sum(1 for k in fails_all if k.startswith("Walk"))))
    if opts.get("parity"):              # parity=<char page json>[,<look-dev page json>]
        pr = [parity(dump, json.load(open(f))) for f in opts["parity"].split(",")]
        rep["parity"] = {"pages": [dict(file=os.path.basename(f), **r["summary"]) for f, r in zip(opts["parity"].split(","), pr)],
                         "poses": {("%s|%s" % (os.path.basename(f), k)): v for f, r in zip(opts["parity"].split(","), pr) for k, v in r["poses"].items()},
                         "summary": {"n": sum(r["summary"]["n"] for r in pr), "joint_max_mm": max(r["summary"]["joint_max_mm"] for r in pr),
                                     "floor_max_mm": max(r["summary"]["floor_max_mm"] for r in pr)}}
        print("PARITY", json.dumps(rep["parity"]["summary"]))
        for k, v in rep["parity"]["summary"].items():
            pass
        if rep["parity"]["summary"]["joint_max_mm"] > 0.5 or rep["parity"]["summary"]["floor_max_mm"] > 0.2:
            fails_all["G-P6 parity"] = ["joint %.3f mm, floor %.3f mm" % (rep["parity"]["summary"]["joint_max_mm"], rep["parity"]["summary"]["floor_max_mm"])]
    if rep["kit_blocks_identical"] is False:
        fails_all["G-P6 kit"] = ["the POSEKIT blocks of the two templates differ"]
    rep["fails"] = fails_all
    if opts.get("dump"):
        json.dump(dump, open(os.path.join(out, "pose_dump_%s.json" % kind), "w"))
    jp = os.path.join(out, "pose_qa_%s.json" % kind)
    json.dump(rep, open(jp, "w"), indent=1, default=float)
    print("WROTE", jp, "failing:", len(fails_all))
    if opts.get("render"):
        render_all(kit, names, out, kind, opts)
    kit.restore_visibility()
    CL.pose_reset(rig)
    if fails_all and not opts.get("nocheck"):
        sys.exit(1)


DUMP_BONES = ("pelvis", "thigh_l", "thigh_r", "calf_l", "calf_r", "foot_l", "foot_r", "ball_l", "ball_r", "spine_03", "neck_01",
              "head", "hand_l", "hand_r", "lowerarm_l", "lowerarm_r")


def parity(dump, web):
    """browser (tools/measure_poses.js) vs Blender: joint positions (glTF frame) and the lowest body vertex per pose."""
    res, jm, fm = {}, 0.0, 0.0
    for name, d in dump.items():
        w = web.get("poses", {}).get(name)
        if not w:
            continue
        dj = max(float(np.linalg.norm(np.array(d[b]) - np.array(w["joints"][b]))) * 1000 for b in d if not b.startswith("__") and b in w["joints"])
        df = abs(d["__min_y_mm"] - w["min_y_mm"]) if "min_y_mm" in w else None
        res[name] = {"joint_mm": round(dj, 3), "floor_mm": None if df is None else round(df, 3)}
        jm = max(jm, dj)
        if df is not None:
            fm = max(fm, df)
    return {"poses": res, "summary": {"n": len(res), "joint_max_mm": round(jm, 3), "floor_max_mm": round(fm, 3)}}


def render_all(kit, names, out, kind, opts):
    """Workbench views (red = below the floor, green = 0-4 mm contact): side + front 3/4 of the pose, both feet side and
    from below, the audit's camera rules, into <out>/after/."""
    import chr_lib as CL
    rd = os.path.join(out, "after")
    os.makedirs(rd, exist_ok=True)
    sc = bpy.context.scene
    sc.render.engine = 'BLENDER_WORKBENCH'
    sh = sc.display.shading
    sh.light = 'STUDIO'; sh.color_type = 'VERTEX'; sh.show_cavity = True; sh.cavity_type = 'WORLD'
    sh.show_backface_culling = True
    sc.view_settings.view_transform = 'Standard'
    rig, body = kit.rig, kit.body
    for o in rig.children:
        if o.type == 'MESH' and o != body:
            o.hide_render = True
    me = body.data
    ca = me.color_attributes.get("pqa") or me.color_attributes.new("pqa", 'FLOAT_COLOR', 'POINT')
    me.color_attributes.active_color = ca
    me.color_attributes.render_color_index = list(me.color_attributes).index(ca)
    bpy.ops.mesh.primitive_plane_add(size=4, location=(0, 0, 0))
    floor = bpy.context.active_object
    fc = floor.data.color_attributes.new("pqa", 'FLOAT_COLOR', 'POINT')
    for d in fc.data:
        d.color = (0.42, 0.44, 0.47, 1)
    stool = None
    CLAY = np.array([0.72, 0.66, 0.6, 1.0])
    NV = len(me.vertices)
    for name in names:
        sol = kit.apply(name)
        co = kit.eval_co()
        cols = np.tile(CLAY, (NV, 1))
        cols[co[:, 1] < BAND[1]] = (0.35, 0.75, 0.35, 1)
        cols[co[:, 1] < BAND[0]] = (0.9, 0.12, 0.1, 1)
        ca.data.foreach_set("color", cols.ravel())
        me.update()
        if stool:
            bpy.data.objects.remove(stool, do_unlink=True); stool = None
        if sol.get("stool"):
            st = sol["stool"]
            bpy.ops.mesh.primitive_cube_add(size=1)
            stool = bpy.context.active_object
            x0, x1 = st["x"]; z0, z1 = st["z"]
            stool.scale = (x1 - x0, z1 - z0, st["top"])
            stool.location = ((x0 + x1) / 2, -(z0 + z1) / 2, st["top"] / 2)
            scol = stool.data.color_attributes.new("pqa", 'FLOAT_COLOR', 'POINT')
            for d in scol.data:
                d.color = (0.45, 0.32, 0.22, 1)
        tag = name.lower().replace(" ", "_")
        W = kit.sk.wpos
        mid = g2b(0.5 * (W("foot_l") + W("foot_r")))
        pel = g2b(W("pelvis"))
        ty = (mid[1] + pel[1]) / 2
        for vn, loc, tgt in (("side", (mid[0] + 3.0, ty, 0.7), (mid[0], ty, 0.55)),
                             ("front34", (mid[0] - 1.3, ty - 2.6, 0.85), (mid[0], ty, 0.55))):
            sc.render.resolution_x, sc.render.resolution_y = 900, 760
            CL.camera(loc, tgt, lens=45)
            CL.render(os.path.join(rd, "pqa_%s_%s_%s.png" % (kind, tag, vn)))
        if not table_has_legs(kit, name):
            continue
        for s in S2:
            a, b = g2b(W("foot" + s)), g2b(W("ball" + s))
            fc_ = (a + b) / 2
            sc.render.resolution_x, sc.render.resolution_y = 760, 520
            CL.camera((fc_[0] + 0.55 * sg(s), fc_[1] - 0.05, 0.07), (fc_[0], fc_[1], 0.04), lens=50)
            CL.render(os.path.join(rd, "pqa_%s_%s_feet%s_side.png" % (kind, tag, s)))
            CL.camera((fc_[0] + 0.25 * sg(s), fc_[1] - 0.35, -0.32), (fc_[0], fc_[1], 0.02), lens=50)
            CL.render(os.path.join(rd, "pqa_%s_%s_feet%s_below.png" % (kind, tag, s)))
    if stool:
        bpy.data.objects.remove(stool, do_unlink=True)
    bpy.data.objects.remove(floor, do_unlink=True)


def table_has_legs(kit, name):
    return bool(kit.T["poses"].get(name, {}).get("legs"))


if __name__ == "__main__":
    main()
