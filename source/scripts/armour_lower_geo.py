"""Geometry helpers for scripts/armour_lower.py (knight lower armour, cloth and props). Blender Python, numpy only.

Coordinates: Blender world = rig space, metres, +Z up, the character faces -Y, +X is the character's LEFT.

Building blocks
  BodySurf     evaluated rest shape of the live MPFB body (all modelling keys mixed, helpers excluded): BVH for rays and
               closest points.
  AxisField    a smooth radius field r(t, theta) around a limb axis sampled from the body (rays from the axis),
               upper-envelope smoothed so plates bridge concavities instead of dipping into them.
  Mesh         quad-mesh builder with per-corner UVs, per-face material slots, per-vertex tags (used for weights).
  plate()      a hard-surface plate from a surface function S(u, v): outer skin with optional raised trim bands along
               any edge, rolled / bevelled rim, inner skin, all quads, UVs laid out on the plate trim sheet.
"""
import bpy, math
import numpy as np
from mathutils import Vector, Matrix
from mathutils.bvhtree import BVHTree

# ------------------------------------------------------------------------------------------------ UV conventions
# Faces carry a material SLOT name (Mesh.M indexes SLOTS). UVs are METRIC: tileable slots get (u, v) in metres on the
# surface (the dress step multiplies them by the texture set's uv_per_m from assets/textures/textures_char.json);
# strip slots ('trim', 'strap') get u = metres along the strip, v = 0..1 across it; atlas slots ('tabard', 'cape',
# 'shield', ...) get v / u normalised to the panel (0..1), mapped into the atlas region at dress time.
SLOTS = ["steel", "gold", "trim", "leather", "mail", "cloth", "lining", "strap", "tabard", "cape", "cape_lining",
         "wood", "blade", "grip", "enamel", "shield", "rim", "armour"]
SLOT = {n: i for i, n in enumerate(SLOTS)}
STRIP_SLOTS = ("trim", "strap")

# ------------------------------------------------------------------------------------------------ small math
def smoothstep(e0, e1, x):
    t = np.clip((np.asarray(x, dtype=float) - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def gauss1d(a, sigma, axis, wrap=False):
    """Separable Gaussian along one axis of a 2-D array (sigma in samples)."""
    if sigma <= 0:
        return a
    r = int(math.ceil(3 * sigma))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2); k /= k.sum()
    a = np.moveaxis(a, axis, 0)
    if wrap:
        p = np.concatenate([a[-r:], a, a[:r]], 0)
    else:
        p = np.concatenate([np.repeat(a[:1], r, 0), a, np.repeat(a[-1:], r, 0)], 0)
    out = np.zeros_like(a)
    for i, w in enumerate(k):
        out += w * p[i:i + len(a)]
    return np.moveaxis(out, 0, axis)


def v3(a):
    return Vector((float(a[0]), float(a[1]), float(a[2])))


def nrm(a):
    a = np.asarray(a, dtype=float)
    return a / max(np.linalg.norm(a), 1e-12)


# ------------------------------------------------------------------------------------------------ body surface
class BodySurf:
    """Rest-pose shape of the live MPFB body: shape keys mixed (the fitted human), helper geometry excluded."""

    NBODY = 13380                                       # hm08: vertices 0..13379 are the skin, the rest are helpers

    ARM_BONES = ("upperarm", "lowerarm", "hand", "thumb", "index", "middle", "ring", "pinky")

    def __init__(self, bm, exclude_arms=False):
        me = bm.data
        tmp = bm.shape_key_add(name="__kl_mix", from_mix=True)
        co = np.empty(len(me.vertices) * 3); tmp.data.foreach_get("co", co)
        bm.shape_key_remove(tmp)
        self.co = co.reshape(-1, 3) + np.array(bm.matrix_world.translation)
        # dominant rts_human bone per vertex (skin weights)
        gname = {g.index: g.name for g in bm.vertex_groups}
        self.dom = [""] * len(me.vertices)
        for v in me.vertices:
            best = (0.0, "")
            for g in v.groups:
                nm = gname[g.group]
                if g.weight > best[0] and ("_" in nm or nm in ("pelvis", "head", "jaw")) and \
                        not nm.startswith(("joint", "helper", "Delete", "kl_del", "rts_")):
                    best = (g.weight, nm)
            self.dom[v.index] = best[1]
        arm = np.array([d.startswith(self.ARM_BONES) for d in self.dom])
        self.arm = arm
        polys = [tuple(p.vertices) for p in me.polygons if max(p.vertices) < self.NBODY
                 and not (exclude_arms and arm[list(p.vertices)].any())]
        self.polys = polys
        self.bvh = BVHTree.FromPolygons([v3(c) for c in self.co], polys)
        # vertex normals of the skin (area weighted)
        n = np.zeros_like(self.co)
        for p in polys:
            a, b, c = self.co[p[0]], self.co[p[1]], self.co[p[2]]
            fn = np.cross(b - a, c - a)
            for i in p:
                n[i] += fn
        self.no = n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)

    def ray(self, o, d, maxd=0.6):
        hit = self.bvh.ray_cast(v3(o), v3(nrm(d)), maxd)
        return None if hit[0] is None else hit[3]

    def closest(self, p):
        loc, n, i, dist = self.bvh.find_nearest(v3(p))
        return np.array(loc), np.array(n), dist


# ------------------------------------------------------------------------------------------------ limb axis field
class AxisField:
    """Radius field around an axis: P(t, th, off) = O + A t + (r(t, th) + off) (cos th F + sin th L).
    th = 0 at the front (F), +90 deg towards L (pass the limb's OUTER side for L so th > 0 is lateral on both legs).
    The field is sampled with rays from the axis, then upper-envelope smoothed (plates bridge hollows)."""

    def __init__(self, body, O, A, F, L, t0, t1, nt=48, nth=72, sig_t=0.02, sig_th=14.0, envelope=True, rmax=0.35,
                 fill=None):
        self.O, self.A = np.array(O, float), nrm(A)
        F = np.array(F, float); F = nrm(F - self.A * F.dot(self.A))
        L = np.array(L, float); L = nrm(L - self.A * L.dot(self.A) - F * L.dot(F))
        self.F, self.L = F, L
        self.t0, self.t1, self.nt, self.nth = t0, t1, nt, nth
        ts = np.linspace(t0, t1, nt); ths = np.linspace(0, 2 * math.pi, nth, endpoint=False)
        r = np.full((nt, nth), np.nan)
        for i, t in enumerate(ts):
            c = self.O + self.A * t
            for j, th in enumerate(ths):
                d = math.cos(th) * F + math.sin(th) * L
                h = body.ray(c, d, rmax)
                if h is not None:
                    r[i, j] = h
        # fill misses (outside the body) from the neighbours in theta, then in t
        for i in range(nt):
            row = r[i]
            if np.isnan(row).all():
                continue
            ok = ~np.isnan(row)
            if not ok.all():
                idx = np.arange(nth)
                row[~ok] = np.interp(idx[~ok], idx[ok], row[ok], period=nth)
        for j in range(nth):
            col = r[:, j]; ok = ~np.isnan(col)
            if not ok.all() and ok.any():
                idx = np.arange(nt); col[~ok] = np.interp(idx[~ok], idx[ok], col[ok])
        if fill is not None:
            r = fill(ts, ths, r)
        dt = (t1 - t0) / (nt - 1)
        st, sth = sig_t / dt, sig_th / (360.0 / nth)
        if envelope:
            r1 = gauss1d(gauss1d(r, st, 0), sth, 1, wrap=True)
            r = np.maximum(r, r1)
        self.raw = r
        self.r = gauss1d(gauss1d(r, st, 0), sth, 1, wrap=True)
        self.ts, self.ths = ts, ths

    def radius(self, t, th):
        """Bilinear lookup, t clamped, th periodic (radians)."""
        ft = np.clip((np.asarray(t, float) - self.t0) / (self.t1 - self.t0) * (self.nt - 1), 0, self.nt - 1.0001)
        fj = (np.asarray(th, float) % (2 * math.pi)) / (2 * math.pi) * self.nth
        i0 = np.floor(ft).astype(int); a = ft - i0
        j0 = np.floor(fj).astype(int) % self.nth; b = fj - np.floor(fj); j1 = (j0 + 1) % self.nth
        r = self.r
        return (r[i0, j0] * (1 - a) * (1 - b) + r[i0 + 1, j0] * a * (1 - b) + r[i0, j1] * (1 - a) * b
                + r[i0 + 1, j1] * a * b)

    def point(self, t, th, off=0.0):
        rr = self.radius(t, th) + off
        d = np.cos(th)[..., None] * self.F + np.sin(th)[..., None] * self.L
        return self.O + self.A * np.asarray(t, float)[..., None] + rr[..., None] * d


# ------------------------------------------------------------------------------------------------ mesh builder
class Mesh:
    def __init__(self):
        self.V = []; self.F = []; self.UV = []; self.M = []; self.tag = []; self.tagw = []

    def add_v(self, p, tag=0, w=None):
        self.V.append(tuple(float(x) for x in p)); self.tag.append(tag); self.tagw.append(w)
        return len(self.V) - 1

    def add_f(self, vs, uvs, mat=0):
        self.F.append(tuple(vs)); self.UV.append([tuple(map(float, u)) for u in uvs]); self.M.append(mat)

    def grid(self, P, UVg, mat=0, tag=0, flip=False, closed_u=False):
        """P: (nv, nu, 3) points, UVg: (nv, nu, 2). Quads between rows; returns the vertex index grid."""
        nv, nu = P.shape[:2]
        tags = tag if isinstance(tag, np.ndarray) else np.full((nv, nu), tag)
        idx = np.array([[self.add_v(P[j, i], int(tags[j, i])) for i in range(nu)] for j in range(nv)])
        self.quads(idx, UVg, mat, flip, closed_u)
        return idx

    def quads(self, idx, UVg, mat=0, flip=False, closed_u=False):
        """Quads over a vertex index grid (nv, nu). closed_u: wrap around in u (UVg then has nu + 1 columns)."""
        nv, nu = idx.shape
        for j in range(nv - 1):
            for i in range(nu - 1 + (1 if closed_u else 0)):
                i1 = (i + 1) % nu
                vs = [idx[j, i], idx[j, i1], idx[j + 1, i1], idx[j + 1, i]]
                uv = [UVg[j, i], UVg[j, i + 1], UVg[j + 1, i + 1], UVg[j + 1, i]]
                if flip:
                    vs = vs[::-1]; uv = uv[::-1]
                self.add_f(vs, uv, mat)

    def merge(self, other, mat_map=None):
        o = len(self.V)
        self.V += other.V; self.tag += other.tag; self.tagw += other.tagw
        for f, uv, m in zip(other.F, other.UV, other.M):
            self.F.append(tuple(i + o for i in f)); self.UV.append(uv)
            self.M.append(mat_map[m] if mat_map else m)
        return o

    def triangulate(self):
        """split quads into triangles (keeps slots / UVs), for pieces that mix quads and triangle caps"""
        F, UV, M = [], [], []
        for f, uv, mt in zip(self.F, self.UV, self.M):
            if len(f) == 3:
                F.append(f); UV.append(uv); M.append(mt)
            else:
                for a, b, c in ((0, 1, 2), (0, 2, 3)):
                    F.append((f[a], f[b], f[c])); UV.append([uv[a], uv[b], uv[c]]); M.append(mt)
        self.F, self.UV, self.M = F, UV, M

    def mixed(self):
        return len(set(len(f) for f in self.F)) > 1

    def mirror_x(self):
        """Mirror copy (x -> -x) with reversed winding; tags keep their value (caller remaps sides)."""
        m = Mesh()
        m.V = [(-x, y, z) for x, y, z in self.V]; m.tag = list(self.tag); m.tagw = list(self.tagw)
        m.F = [tuple(f[::-1]) for f in self.F]; m.UV = [list(uv[::-1]) for uv in self.UV]; m.M = list(self.M)
        return m

    def to_object(self, name, coll=None, smooth=True, sharp_angle=None):
        me = bpy.data.meshes.new(name)
        me.from_pydata(self.V, [], self.F)
        uvl = me.uv_layers.new(name="UVMap")
        flat = [uv for f in self.UV for uv in f]
        uvl.data.foreach_set("uv", np.array(flat, dtype=np.float32).ravel())
        me.polygons.foreach_set("material_index", np.array(self.M, dtype=np.int32))
        me.polygons.foreach_set("use_smooth", np.full(len(self.F), smooth))
        me.validate(clean_customdata=False)
        me.update()
        if sharp_angle is not None:
            me.set_sharp_from_angle(angle=math.radians(sharp_angle))
        ob = bpy.data.objects.new(name, me)
        (coll or bpy.context.scene.collection).objects.link(ob)
        ob["kl_tags"] = list(self.tag)
        return ob


# ------------------------------------------------------------------------------------------------ plate builder
def _arclen(P):
    """cumulative arc length along axis 0 of (n, 3) points"""
    d = np.linalg.norm(np.diff(P, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(d)])


def _params(total, edge_rows, n_mid, lo_trim, hi_trim):
    """Parameter distances along one direction: dense rows at the trimmed edges (band, step), uniform in between."""
    lo = [0.0] + (edge_rows if lo_trim else [])
    hi = [total] + ([total - e for e in edge_rows] if hi_trim else [])
    a = max(lo); b = min(hi)
    mid = list(np.linspace(a, b, n_mid + 1))[1:-1] if n_mid > 1 else []
    return sorted(set([round(x, 6) for x in lo + mid + hi]))


def _ls_uv(P):
    """(iteration 2b, integrity G6: the plates' steel / trim texels were sheared 1.5-8x where the grid's rows and
    columns are not orthogonal, e.g. at a cuisse's curved lower edge) a near-isometric UV embedding of the grid P
    (NV, NU, 3): every triangle's edges, expressed in a frame aligned with its quad's rows (u) and N x u (v), are
    matched in the least-squares sense; u and v decouple. Returns (NV, NU, 2) metres with the minimum at 0."""
    NV, NU = P.shape[:2]
    nvar = NV * NU
    rows_u, rows_v = [], []
    for j in range(NV - 1):
        for i in range(NU - 1):
            q = [(j, i), (j, i + 1), (j + 1, i + 1), (j + 1, i)]
            t1 = (P[j, i + 1] - P[j, i]) + (P[j + 1, i + 1] - P[j + 1, i])
            t2 = (P[j + 1, i] - P[j, i]) + (P[j + 1, i + 1] - P[j, i + 1])
            n = np.cross(t1, t2)
            if np.linalg.norm(n) < 1e-14 or np.linalg.norm(t1) < 1e-12:
                continue
            t1 = t1 / np.linalg.norm(t1); n = n / np.linalg.norm(n); t2 = np.cross(n, t1)
            for a, b in ((0, 1), (1, 2), (2, 0), (0, 2), (2, 3), (3, 0)):
                ia = q[a][0] * NU + q[a][1]; ib = q[b][0] * NU + q[b][1]
                e = P[q[b]] - P[q[a]]
                rows_u.append((ia, ib, float(e.dot(t1)))); rows_v.append((ia, ib, float(e.dot(t2))))
    if not rows_u:
        return None
    M = np.zeros((len(rows_u) + 1, nvar))
    for r, (ia, ib, _) in enumerate(rows_u):
        M[r, ib] += 1.0; M[r, ia] -= 1.0
    M[-1, 0] = 1.0                                              # gauge
    bu = np.array([x for _, _, x in rows_u] + [0.0]); bv = np.array([x for _, _, x in rows_v] + [0.0])
    sol = np.linalg.lstsq(M, np.stack([bu, bv], 1), rcond=None)[0]
    W = sol.reshape(NV, NU, 2)
    return W - W.reshape(-1, 2).min(0)


def ls_uv_quads(pos, quads):
    """(iteration 2b, G6) near-isometric UVs for a set of quads with shared vertex KEYS: pos = {key: xyz}, quads =
    [(k0, k1, k2, k3)] with k0 -> k3 the 'u' direction and k0 -> k1 the 'v' direction of each quad (a grid-like
    sheet; cut seams are expressed by giving the two sides different keys). Least squares on every triangle's edges
    in its quad's (u, N x u) frame, u and v solved separately. Returns {key: (u, v)} in metres."""
    keys = sorted(pos, key=lambda k: str(k))
    ix = {k: i for i, k in enumerate(keys)}
    P = {k: np.asarray(pos[k], float) for k in keys}
    rows, bu, bv = [], [], []
    for q in quads:
        p0, p1, p2, p3 = (P[k] for k in q)
        t1 = (p3 - p0) + (p2 - p1)
        t2 = (p1 - p0) + (p2 - p3)
        n = np.cross(t1, t2)
        if np.linalg.norm(n) < 1e-14 or np.linalg.norm(t1) < 1e-12:
            continue
        t1 = t1 / np.linalg.norm(t1); n = n / np.linalg.norm(n); t2 = np.cross(n, t1)
        for a, b in ((0, 1), (1, 2), (2, 0), (0, 2), (2, 3), (3, 0)):
            e = P[q[b]] - P[q[a]]
            rows.append((ix[q[a]], ix[q[b]])); bu.append(float(e @ t1)); bv.append(float(e @ t2))
    n_ = len(keys)
    M = np.zeros((len(rows) + 1, n_))
    for r, (ia, ib) in enumerate(rows):
        M[r, ib] += 1.0; M[r, ia] -= 1.0
    M[-1, 0] = 1.0
    sol = np.linalg.lstsq(M, np.stack([np.array(bu + [0.0]), np.array(bv + [0.0])], 1), rcond=None)[0]
    sol = sol - sol.min(0)
    return {k: (float(sol[ix[k], 0]), float(sol[ix[k], 1])) for k in keys}


def ls_uv_frames(pos, faces, t1s, t2s=None):
    """(2b, G6) near-isometric UVs for polygons with shared vertex KEYS and a given 'u' direction per face (t1s) and,
    optionally, a 'v' direction hint (t2s: the frame's v side is taken from it, so faces may be wound either way):
    every edge of each polygon (and its fan diagonals) matched in the face's (u, v) frame. Returns {key: (u, v)} m."""
    keys = sorted(pos, key=lambda k: str(k))
    ix = {k: i for i, k in enumerate(keys)}
    P = {k: np.asarray(pos[k], float) for k in keys}
    rows, bu, bv = [], [], []
    for fi, (f, t1) in enumerate(zip(faces, t1s)):
        pts = [P[k] for k in f]
        n = np.zeros(3)
        for a_ in range(1, len(pts) - 1):
            n += np.cross(pts[a_] - pts[0], pts[a_ + 1] - pts[0])
        if np.linalg.norm(n) < 1e-14:
            continue
        n = n / np.linalg.norm(n)
        t1 = np.asarray(t1, float); t1 = t1 - n * (t1 @ n)
        if np.linalg.norm(t1) < 1e-9:
            continue
        t1 = t1 / np.linalg.norm(t1); t2 = np.cross(n, t1)
        if t2s is not None and float(t2 @ np.asarray(t2s[fi], float)) < 0:
            t2 = -t2
        L = len(f)
        pairs = [(a_, (a_ + 1) % L) for a_ in range(L)] + [(0, a_) for a_ in range(2, L - 1)]
        for a_, b_ in pairs:
            e = P[f[b_]] - P[f[a_]]
            rows.append((ix[f[a_]], ix[f[b_]])); bu.append(float(e @ t1)); bv.append(float(e @ t2))
    n_ = len(keys)
    M = np.zeros((len(rows) + 1, n_))
    for r, (ia, ib) in enumerate(rows):
        M[r, ib] += 1.0; M[r, ia] -= 1.0
    M[-1, 0] = 1.0
    sol = np.linalg.lstsq(M, np.stack([np.array(bu + [0.0]), np.array(bv + [0.0])], 1), rcond=None)[0]
    sol = sol - sol.min(0)
    return {k: (float(sol[ix[k], 0]), float(sol[ix[k], 1])) for k in keys}


TRIM_REF = 0.012          # physical band width the trim-sheet / kl_trim strips are scaled for (armour_lower.to_sheet)


def plate(S, nu, nv, *, trim=(), band=0.016, roll=0.003, body_slot="steel", band_slot="trim", rim_gold="gold",
          tag=0, flip=False, outward=None, mesh=None, rim_gold_all=False, rim_angles=(0, -90), uv0=(0.0, 0.0),
          inner=False, inner_slot="lining"):
    """Hard-surface plate over the param square u, v in [0, 1] (S(u, v) -> (..., 3) points).
    trim: edges carrying a gold trim band: 0 = v0 (first row), 1 = u1 (last column), 2 = v1 (last row), 3 = u0.
    nu, nv: interior segments. The rim rolls away from the outer normal (radius roll, sheet 2 roll thick): gold on
    trimmed edges. The outer skin is the only skin (double-sided material) unless inner=True.
    outward(p) -> unit vector: the side the outer skin faces (else Pu x Pv, reversed with flip=True).
    Returns (Mesh, info). All quads; metric UVs (see SLOTS)."""
    m = mesh or Mesh()
    K = 64
    uu = np.linspace(0, 1, K)
    Lu = _arclen(S(uu, np.full(K, 0.5)))[-1]
    alv = _arclen(S(np.full(K, 0.5), uu)); Lv = alv[-1]
    du = _params(Lu, [band], nu, 3 in trim, 1 in trim)
    dv = _params(Lv, [band], nv, 0 in trim, 2 in trim)
    NU, NV = len(du), len(dv)
    Ug = np.zeros((NV, NU)); Vg = np.zeros((NV, NU))
    vv0 = np.interp(np.array(dv), alv, uu)
    for j in range(NV):
        al = _arclen(S(uu, np.full(K, vv0[j])))
        Ug[j] = np.interp(np.array(du) / Lu * al[-1], al, uu)
    for i in range(NU):
        al = _arclen(S(np.full(K, Ug[NV // 2, i]), uu))
        Vg[:, i] = np.interp(np.array(dv) / Lv * al[-1], al, uu)
    P = S(Ug.ravel(), Vg.ravel()).reshape(NV, NU, 3)
    e = 1e-3
    f = lambda a, b: S(np.clip(a, 0, 1).ravel(), np.clip(b, 0, 1).ravel()).reshape(NV, NU, 3)
    Pu = f(Ug + e, Vg) - f(Ug - e, Vg)
    Pv = f(Ug, Vg + e) - f(Ug, Vg - e)
    N = np.cross(Pu, Pv); N /= np.maximum(np.linalg.norm(N, axis=2, keepdims=True), 1e-12)
    rev = bool(flip)
    if outward is not None:
        c = P[NV // 2, NU // 2]
        rev = N[NV // 2, NU // 2].dot(outward(c)) < 0
    if rev:
        N = -N
    # real arc lengths on the fitted grid for the UVs
    Du = np.zeros((NV, NU)); Dv = np.zeros((NV, NU))
    for j in range(NV):
        Du[j] = _arclen(P[j])
    for i in range(NU):
        Dv[:, i] = _arclen(P[:, i])
    Lu_r = Du[:, -1:].repeat(NU, 1); Lv_r = Dv[-1:, :].repeat(NV, 0)
    dist = {0: Dv, 2: Lv_r - Dv, 3: Du, 1: Lu_r - Du}
    bs = SLOT[body_slot]
    # (2b, G6) texture coordinates from the least-squares embedding (the distances to the edges above still decide
    # which faces are trim bands); bands: u along the band scaled so a band of width `band` shows the strip isotropic
    Wls = _ls_uv(P)
    Eu, Ev = (Wls[..., 0], Wls[..., 1]) if Wls is not None else (Du, Dv)
    ks = (TRIM_REF / band) if band_slot == "trim" and band > 0 else 1.0

    def face(vs, uv, flipit, slot):
        if flipit:
            vs = vs[::-1]; uv = uv[::-1]
        m.add_f(vs, uv, slot)

    idx = np.array([[m.add_v(P[j, i], tag) for i in range(NU)] for j in range(NV)])
    for j in range(NV - 1):
        for i in range(NU - 1):
            vs = [idx[j, i], idx[j, i + 1], idx[j + 1, i + 1], idx[j + 1, i]]
            cs = [(j, i), (j, i + 1), (j + 1, i + 1), (j + 1, i)]
            kind = None
            for s_ in trim:
                if max(dist[s_][c] for c in cs) <= band * 1.05 + 1e-6:
                    kind = s_; break
            if kind is None:
                uv = [(uv0[0] + Eu[c], uv0[1] + Ev[c]) for c in cs]; slot = bs
            else:
                along = Eu if kind in (0, 2) else Ev
                uv = [(along[c] * ks, 1.0 - min(dist[kind][c] / band, 1.0)) for c in cs]; slot = SLOT[band_slot]
            face(vs, uv, rev, slot)
    info = dict(idx=idx, P=P, N=N, Lu=Lu, Lv=Lv, Pu=Pu, Pv=Pv)
    loop = ([(0, i) for i in range(NU)] + [(j, NU - 1) for j in range(1, NV)] +
            [(NV - 1, i) for i in range(NU - 2, -1, -1)] + [(j, 0) for j in range(NV - 2, 0, -1)])
    nL = len(loop)
    E = []
    for (j, i) in loop:
        d = np.zeros(3)
        if j == 0: d = d - nrm(Pv[j, i])
        if j == NV - 1: d = d + nrm(Pv[j, i])
        if i == 0: d = d - nrm(Pu[j, i])
        if i == NU - 1: d = d + nrm(Pu[j, i])
        n = N[j, i]; d = d - n * d.dot(n)
        E.append(nrm(d))
    E = np.array(E)
    LP = np.array([P[j, i] for (j, i) in loop])
    # (2b, G6 / G2: on a skewed grid the column / row direction leans along the edge, so the rolled rim sheared
    # sideways) the rim rolls perpendicular to the edge itself (tangent from the loop neighbours), on the old side
    Tl = np.roll(LP, -1, 0) - np.roll(LP, 1, 0)
    for k, (j, i) in enumerate(loop):
        e_ = np.cross(Tl[k], N[j, i])
        if np.linalg.norm(e_) > 1e-12:
            e_ = nrm(e_)
            E[k] = e_ if e_.dot(E[k]) >= 0 else -e_
    al = _arclen(np.vstack([LP, LP[:1]]))
    R = roll
    rim_idx = [[idx[j, i] for (j, i) in loop]]
    for ang in rim_angles:
        ar = math.radians(ang)
        ring = []
        for k, (j, i) in enumerate(loop):
            c = P[j, i] - N[j, i] * R
            ring.append(m.add_v(c + E[k] * R * math.cos(ar) + N[j, i] * R * math.sin(ar), tag))
        rim_idx.append(ring)

    def edge_of(k):
        (j, i) = loop[k]; (j2, i2) = loop[(k + 1) % nL]
        if j == 0 and j2 == 0: return 0
        if i == NU - 1 and i2 == NU - 1: return 1
        if j == NV - 1 and j2 == NV - 1: return 2
        if i == 0 and i2 == 0: return 3
        return -1
    nr = len(rim_angles)
    arc = math.pi * R / nr
    for r in range(nr):
        for k in range(nL):
            k2 = (k + 1) % nL
            gold = rim_gold_all or edge_of(k) in trim
            slot = SLOT["rim" if rim_gold == "gold" else rim_gold] if gold else bs
            v0_, v1_ = r * arc, (r + 1) * arc
            vs = [rim_idx[r][k], rim_idx[r][k2], rim_idx[r + 1][k2], rim_idx[r + 1][k]]
            uv = [(al[k], v0_), (al[k + 1], v0_), (al[k + 1], v1_), (al[k], v1_)]
            face(vs, uv, not rev, slot)
    info["loop"] = loop
    info["rim_last"] = rim_idx[-1]
    if inner:
        Pin = P - N * (2 * R)
        lk = {loop[k]: rim_idx[-1][k] for k in range(nL)}
        iidx = np.array([[lk[(j, i)] if (j, i) in lk else m.add_v(Pin[j, i], tag) for i in range(NU)]
                         for j in range(NV)])
        for j in range(NV - 1):
            for i in range(NU - 1):
                vs = [iidx[j, i], iidx[j + 1, i], iidx[j + 1, i + 1], iidx[j, i + 1]]
                cs = [(j, i), (j + 1, i), (j + 1, i + 1), (j, i + 1)]
                face(vs, [(uv0[0] + Eu[c], uv0[1] + Ev[c]) for c in cs], rev, SLOT[inner_slot])
    return m, info


def lathe(profile, axis_o, axis_d, ref, nseg, uv_u=(0.0, 1.0), uv_v=None, mat=0, tag=0, mesh=None, cap_ends=False,
          scale=(1.0, 1.0), mats=None, solid=True, metric=False):
    """Surface of revolution: profile = [(r, h), ...] (radius, height along axis_d); closed around.
    UV: u around (uv_u range), v from uv_v list per profile point (or arc length).
    Iteration 2b (user item 28, integrity G1): the faces are wound so the normals point OUT of the solid the profile
    bounds (the profile's (r, h) polygon closed along the axis: its orientation decides; before, a profile running
    outward -> up -> inward came out inside-out, the clasps vanished from outside); a profile point with r == 0 is
    the pole (one vertex, a triangle fan), so a profile from axis to axis is a closed solid. scale = (sx, sy) squashes
    the section (elliptic lathe); mats = per profile segment material slots (len(profile) - 1); solid=False for an
    open band (orientation from the profile direction: normals to the profile's right in (r, h))."""
    m = mesh or Mesh()
    A = nrm(axis_d); X = nrm(np.array(ref) - A * np.dot(ref, A)); Y = np.cross(A, X)
    O = np.array(axis_o, float)
    n = len(profile)
    pr = np.array([[r, h] for r, h in profile], float)
    if uv_v is None:
        al = _arclen(np.c_[pr, np.zeros(n)])
        uv_v = al if metric else al / max(al[-1], 1e-9)
    # orientation: signed area of the (r, h) polygon closed along the axis; the quad order below faces outward for a
    # counter-clockwise (outward -> up -> inward, area > 0) profile, a clockwise one is reversed
    q = np.vstack([[0.0, pr[0, 1]], pr, [0.0, pr[-1, 1]]]) if solid else pr
    area = 0.5 * float(np.sum(q[:-1, 0] * q[1:, 1] - q[1:, 0] * q[:-1, 1]) +
                       (q[-1, 0] * q[0, 1] - q[0, 0] * q[-1, 1]))
    flip = area < 0
    pole = [abs(r) < 1e-9 for r, h in profile]
    per_pt = len(scale) == n and hasattr(scale[0], "__len__")
    idx = np.zeros((n, nseg), int)
    for k, (r, h) in enumerate(profile):
        if pole[k]:
            idx[k, :] = m.add_v(O + A * h, tag)
            continue
        sx, sy = scale[k] if per_pt else scale
        for s in range(nseg):
            a = 2 * math.pi * s / nseg
            idx[k, s] = m.add_v(O + A * h + r * (math.cos(a) * sx * X + math.sin(a) * sy * Y), tag)
    # (2b, G6) metric u = the real arc length round each ring (elliptic sections included: the mean-radius
    # circumference streaked flattened pommels / guards up to 1.8x), poles at the u of the ring next to them (the
    # pole took the 0..1 fraction while the ring took metres: 1e5-fold shear at every pole fan)
    Ur = np.zeros((n, nseg + 1))
    if metric:
        Vall = np.array(m.V)
        for k in range(n):
            if pole[k]:
                continue
            ring = Vall[idx[k]]
            d = np.linalg.norm(np.roll(ring, -1, 0) - ring, axis=1)
            Ur[k, 1:] = np.cumsum(d)
    for k in range(n - 1):
        mk = mats[k] if mats is not None else mat
        for s in range(nseg):
            s2 = (s + 1) % nseg
            u0 = uv_u[0] + (uv_u[1] - uv_u[0]) * s / nseg; u1 = uv_u[0] + (uv_u[1] - uv_u[0]) * (s + 1) / nseg
            vs = [idx[k, s], idx[k, s2], idx[k + 1, s2], idx[k + 1, s]]
            if metric:
                uv = [(Ur[k, s], uv_v[k]), (Ur[k, s + 1], uv_v[k]), (Ur[k + 1, s + 1], uv_v[k + 1]),
                      (Ur[k + 1, s], uv_v[k + 1])]
                pu0 = 0.5 * (Ur[k + 1, s] + Ur[k + 1, s + 1]); pu1 = 0.5 * (Ur[k, s] + Ur[k, s + 1])
            else:
                uv = [(u0, uv_v[k]), (u1, uv_v[k]), (u1, uv_v[k + 1]), (u0, uv_v[k + 1])]
                pu0 = pu1 = (u0 + u1) / 2
            if pole[k] and pole[k + 1]:
                continue
            if pole[k]:                               # triangle at the pole
                vs, uv = [vs[0], vs[2], vs[3]], [(pu0, uv_v[k]), uv[2], uv[3]]
            elif pole[k + 1]:
                vs, uv = [vs[0], vs[1], vs[2]], [uv[0], uv[1], (pu1, uv_v[k + 1])]
            if flip:
                vs = vs[::-1]; uv = uv[::-1]
            m.add_f(vs, uv, mk)
    return m, idx


# ------------------------------------------------------------------------------------------------ hull / drape sampling
def hull_radius(body, z, phis, centre=(0.0, 0.0), R0=0.7):
    """Distance from the vertical axis through `centre` to the outer body hull at height z along directions phi
    (phi = 0 is the front (-Y), +90 deg the character's left (+X)). Rays cast inward from outside; NaN = miss."""
    out = np.full(len(phis), np.nan)
    cx, cy = centre
    for k, ph in enumerate(phis):
        d = np.array([math.sin(ph), -math.cos(ph), 0.0])
        o = np.array([cx, cy, z]) + d * R0
        h = body.ray(o, -d, R0)
        if h is not None:
            out[k] = R0 - h
    return out


def hull_field(body, zs, nphi=96, centre=(0.0, 0.0), sig_z=0.02, sig_phi=10.0, envelope=True):
    """Hull radius on a (z, phi) grid, misses filled around phi (bridges the gap between the legs), smoothed."""
    phis = np.linspace(0, 2 * math.pi, nphi, endpoint=False)
    R = np.array([hull_radius(body, z, phis, centre) for z in zs])
    idx = np.arange(nphi)
    for i in range(len(zs)):
        ok = ~np.isnan(R[i])
        if ok.any() and not ok.all():
            R[i, ~ok] = np.interp(idx[~ok], idx[ok], R[i, ok], period=nphi)
    dz = abs(zs[1] - zs[0]) if len(zs) > 1 else 1.0
    st, sp = sig_z / dz, sig_phi / (360.0 / nphi)
    if envelope:
        R = np.maximum(R, gauss1d(gauss1d(R, st, 0), sp, 1, wrap=True))
    return phis, gauss1d(gauss1d(R, st, 0), sp, 1, wrap=True)


def depth_map(body, xs, zs, direction, y0):
    """First-hit y of rays cast along +-Y (direction = +1: from the front (y = y0 < 0) towards +Y; -1: from behind)
    over the (z, x) grid. NaN = miss."""
    D = np.full((len(zs), len(xs)), np.nan)
    d = np.array([0.0, float(direction), 0.0])
    for i, z in enumerate(zs):
        for j, x in enumerate(xs):
            h = body.ray((x, y0, z), d, 2.0)
            if h is not None:
                D[i, j] = y0 + direction * h
    return D


def fill_nan_rows(D):
    D = D.copy()
    for i in range(D.shape[0]):
        ok = ~np.isnan(D[i])
        if ok.any() and not ok.all():
            idx = np.arange(D.shape[1]); D[i, ~ok] = np.interp(idx[~ok], idx[ok], D[i, ok])
    for j in range(D.shape[1]):
        ok = ~np.isnan(D[:, j])
        if ok.any() and not ok.all():
            idx = np.arange(D.shape[0]); D[~ok, j] = np.interp(idx[~ok], idx[ok], D[ok, j])
    return D


class Grid2:
    """bilinear lookup of a (z, x) or (z, phi) sampled field"""

    def __init__(self, a0, a1, b0, b1, F, periodic_b=False):
        self.a0, self.a1, self.b0, self.b1, self.F, self.per = a0, a1, b0, b1, F, periodic_b

    def __call__(self, a, b):
        F = self.F; na, nb = F.shape
        fa = np.clip((np.asarray(a, float) - self.a0) / (self.a1 - self.a0) * (na - 1), 0, na - 1.0001)
        if self.per:
            fb = ((np.asarray(b, float) - self.b0) / (self.b1 - self.b0) * nb) % nb
            i0 = np.floor(fa).astype(int); ta = fa - i0
            j0 = np.floor(fb).astype(int) % nb; tb = fb - np.floor(fb); j1 = (j0 + 1) % nb
        else:
            fb = np.clip((np.asarray(b, float) - self.b0) / (self.b1 - self.b0) * (nb - 1), 0, nb - 1.0001)
            i0 = np.floor(fa).astype(int); ta = fa - i0
            j0 = np.floor(fb).astype(int); tb = fb - j0; j1 = j0 + 1
        return (F[i0, j0] * (1 - ta) * (1 - tb) + F[i0 + 1, j0] * ta * (1 - tb) + F[i0, j1] * (1 - ta) * tb
                + F[i0 + 1, j1] * ta * tb)


def sheet(Pf, nu, nv, thick, *, slot_out, slot_in, slot_rim, uv_out, uv_in, tag=0, mesh=None, hem_rows=0):
    """Cloth panel with thickness: outer grid Pf(u, v) (u across, v along; returns points and outward normals),
    inner grid offset by -thick along the normal, a one-quad rim strip closing the four edges.
    uv_out / uv_in(u, v) -> (U, V) arrays. Returns (Mesh, outer index grid, inner index grid)."""
    m = mesh or Mesh()
    U, V = np.meshgrid(np.linspace(0, 1, nu + 1), np.linspace(0, 1, nv + 1))
    P, N = Pf(U, V)
    th = thick(U, V) if callable(thick) else np.full(U.shape, float(thick))
    Pin = P - N * th[..., None]
    io = np.array([[m.add_v(P[j, i], tag) for i in range(nu + 1)] for j in range(nv + 1)])
    ii = np.array([[m.add_v(Pin[j, i], tag) for i in range(nu + 1)] for j in range(nv + 1)])
    uo = np.stack(uv_out(U, V), -1); ui = np.stack(uv_in(U, V), -1)
    # iteration 2b (G1: the tabard / cape shells came out inside-out): wind the outer skin along the outward normal N
    # the caller computed, the inner skin and the rim the other way, so the panel is a closed, outward-wound solid
    g = np.cross(P[1:, :-1] - P[:-1, :-1], P[:-1, 1:] - P[:-1, :-1])        # (v x u) = minus the quads() winding normal
    rev = float((g * N[:-1, :-1]).sum()) > 0                                  # default winding faces against N: flip
    m.quads(io, uo, SLOT[slot_out], flip=rev)
    m.quads(ii, ui, SLOT[slot_in], flip=not rev)
    # rim: bottom (v=0), right (u=1), top (v=1), left (u=0)
    loops = [[(0, i) for i in range(nu + 1)], [(j, nu) for j in range(nv + 1)],
             [(nv, i) for i in range(nu, -1, -1)], [(j, 0) for j in range(nv, -1, -1)]]
    for L in loops:
        for k in range(len(L) - 1):
            a, b = L[k], L[k + 1]
            vs = [io[a], io[b], ii[b], ii[a]]
            uv = [tuple(uo[a]), tuple(uo[b]), tuple(uo[b]), tuple(uo[a])]
            if not rev:
                vs, uv = vs[::-1], uv[::-1]
            m.add_f(vs, uv, SLOT[slot_rim])
    return m, io, ii


def rounded_box(c, X, Y, Z, size, e=0.25, nu=16, nv=10, slot="leather", tag=0, mesh=None, uv_scale=1.0):
    """Superquadric box (rounded edges) centred at c with unit axes X, Y, Z and half sizes (a, b, h); triangle fans at
    the two poles (the piece is triangulated when written)."""
    m = mesh or Mesh()
    f0 = len(m.F)
    a, b, h = size
    sgn = lambda x: math.copysign(abs(x) ** e, x)
    c = np.array(c, float); X, Y, Z = (np.array(v, float) for v in (X, Y, Z))
    per = 4 * (a + b)
    rings = []
    for j in range(1, nv):
        ph = -math.pi / 2 + math.pi * j / nv
        ring = []
        for i in range(nu):
            th = 2 * math.pi * i / nu
            x = a * sgn(math.cos(ph)) * sgn(math.cos(th)); y = b * sgn(math.cos(ph)) * sgn(math.sin(th))
            z = h * sgn(math.sin(ph))
            ring.append(m.add_v(c + x * X + y * Y + z * Z, tag))
        rings.append(ring)
    bot = m.add_v(c - h * Z, tag); top = m.add_v(c + h * Z, tag)
    uvf = lambda i, j: (per * i / nu * uv_scale, 2 * h * j / nv * uv_scale)
    for r in range(len(rings) - 1):
        for i in range(nu):
            i2 = (i + 1) % nu
            vs = [rings[r][i], rings[r][i2], rings[r + 1][i2], rings[r + 1][i]]
            uv = [uvf(i, r + 1), uvf(i + 1, r + 1), uvf(i + 1, r + 2), uvf(i, r + 2)]
            m.add_f(vs, uv, SLOT[slot])
    for i in range(nu):
        i2 = (i + 1) % nu
        m.add_f([bot, rings[0][i2], rings[0][i]], [uvf(i + 0.5, 0), uvf(i + 1, 1), uvf(i, 1)], SLOT[slot])
        m.add_f([top, rings[-1][i], rings[-1][i2]], [uvf(i + 0.5, nv), uvf(i, nv - 1), uvf(i + 1, nv - 1)], SLOT[slot])
    orient_solid(m, f0)                        # left-handed (X, Y, Z) frames came out inside-out (G1)
    return m, rings


def orient_solid(m, f0=0, f1=None):
    """make the closed solid formed by faces f0..f1 of m outward-wound (positive signed volume): reverses them if
    needed; returns True when it flipped"""
    f1 = len(m.F) if f1 is None else f1
    vol = 0.0
    for f in m.F[f0:f1]:
        p0 = np.array(m.V[f[0]])
        for k in range(1, len(f) - 1):
            vol += float(np.dot(p0, np.cross(np.array(m.V[f[k]]), np.array(m.V[f[k + 1]]))))
    if vol < 0:
        for i in range(f0, f1):
            m.F[i] = tuple(m.F[i][::-1]); m.UV[i] = list(m.UV[i][::-1])
        return True
    return False


class Composite:
    """ray-cast target made of a BodySurf (its current polys) plus extra Mesh layers (e.g. the mail skirt), so cloth
    and belts can be draped over the layers under them"""

    def __init__(self, body, meshes=()):
        V = [v3(c) for c in body.co]; P = list(body.polys)
        for m in meshes:
            o = len(V)
            V += [v3(c) for c in m.V]
            for f in m.F:
                if len(f) == 4:
                    P.append((f[0] + o, f[1] + o, f[2] + o, f[3] + o))
                else:
                    P.append(tuple(i + o for i in f))
        self.bvh = BVHTree.FromPolygons(V, P)
        self.co = body.co

    def ray(self, o, d, maxd=0.6):
        hit = self.bvh.ray_cast(v3(o), v3(nrm(d)), maxd)
        return None if hit[0] is None else hit[3]
