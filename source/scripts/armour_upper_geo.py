"""Geometry toolkit for the procedural armour (used by armour_upper.py; plain Blender Python + numpy).

Everything is built as clean quads with explicit UVs into the armour TRIM SHEET (armour_upper_tex.py LAYOUT):
  shell()   a plate from a surface grid: outer surface + a swept RIM PROFILE around every boundary loop (gold band, rolled
            bead, lip turned inward = visible thickness) + optional inner surface. Holes in the grid (eye slit, vents) get
            their own rim loops.
  sweep()   a profile swept along a curve lying on a surface (applied trim bands, crest comb, straps).
  lathe()   a profile revolved around an axis (rivets, rosettes, plume socket), centre closed with a quad grid fill.
  grid_fill() quad patch closing a ring of 4k vertices (helmet top).
MB collects vertices / quads / per-corner UVs / per-vertex bone weights and tags and turns them into a Blender object.
"""
import bpy, math, os
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree
from mathutils.kdtree import KDTree
import armour_upper_tex as TX

EPS = 1e-9


def nrm(v, axis=-1):
    v = np.asarray(v, dtype=np.float64)
    return v / np.maximum(np.linalg.norm(v, axis=axis, keepdims=True), EPS)


def smoothstep(e0, e1, x):
    t = np.clip((np.asarray(x, dtype=np.float64) - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3 - 2 * t)


# ------------------------------------------------------------------------------------------------------------ body
class Body:
    """Rest-pose body surface of the live MPFB human (modelling keys mixed, helpers excluded) for fitting queries."""

    def __init__(self, bm, nbody=13380):
        me = bm.data
        tmp = bm.shape_key_add(name="__mix_geo", from_mix=True)
        co = np.empty(len(me.vertices) * 3); tmp.data.foreach_get("co", co)
        bm.shape_key_remove(tmp)
        self.co_all = co.reshape(-1, 3)
        self.co = self.co_all[:nbody]
        self.n = nbody
        faces = [tuple(p.vertices) for p in me.polygons if max(p.vertices) < nbody]
        self.faces = faces
        tris = []
        for f in faces:
            for k in range(1, len(f) - 1):
                tris.append((f[0], f[k], f[k + 1]))
        self.tris = np.array(tris)
        self.bvh = BVHTree.FromPolygons([Vector(c) for c in self.co], [tuple(t) for t in self.tris])
        # vertex normals (area weighted)
        a, b, c = self.co[self.tris[:, 0]], self.co[self.tris[:, 1]], self.co[self.tris[:, 2]]
        fn = np.cross(b - a, c - a)
        vn = np.zeros_like(self.co)
        for k in range(3):
            np.add.at(vn, self.tris[:, k], fn)
        self.vn = nrm(vn)
        self.kd = KDTree(nbody)
        for i, p in enumerate(self.co):
            self.kd.insert(p, i)
        self.kd.balance()
        self.groups = {}
        names = {g.index: g.name for g in bm.vertex_groups}
        self.bw = {}
        for v in me.vertices:
            if v.index >= nbody:
                continue
            for g in v.groups:
                nm = names[g.group]
                if g.weight > 0.5:
                    self.groups.setdefault(nm, []).append(v.index)
                if not nm.startswith(("Delete", "joint", "helper")) and g.weight > 1e-4:
                    self.bw.setdefault(nm, np.zeros(nbody))[v.index] = g.weight
        self.groups = {k: np.array(v) for k, v in self.groups.items()}

    def weight_sum(self, prefixes):
        w = np.zeros(self.n)
        for nm, a in self.bw.items():
            if nm.startswith(tuple(prefixes)):
                w += a
        return w

    def subset(self, vmask):
        """A Body-like object whose ray queries only see faces with all vertices in vmask."""
        sub = Body.__new__(Body)
        sub.__dict__.update(self.__dict__)
        keep = np.all(vmask[self.tris], axis=1)
        sub.tris = self.tris[keep]
        sub.bvh = BVHTree.FromPolygons([Vector(c) for c in self.co], [tuple(t) for t in sub.tris])
        return sub

    def ray(self, origin, direction, maxd=1.0):
        """Distance from origin to the first body hit along direction (None if nothing within maxd)."""
        loc, nor, fi, d = self.bvh.ray_cast(Vector(origin), Vector(direction), maxd)
        return d

    def radius_field(self, axis_pts, frames, thetas, default=0.05, maxd=0.5):
        """For axis points A[t] with frames (e1[t], e2[t]) cast rays A + r(cos th e1 + sin th e2): (nt, nth) radii."""
        R = np.full((len(axis_pts), len(thetas)), np.nan)
        for i, (A, (e1, e2)) in enumerate(zip(axis_pts, frames)):
            for j, th in enumerate(thetas):
                d = math.cos(th) * e1 + math.sin(th) * e2
                r = self.ray(A, d, maxd)
                if r is not None:
                    R[i, j] = r
        # fill misses from neighbours
        for _ in range(50):
            bad = np.isnan(R)
            if not bad.any():
                break
            Rp = np.pad(R, 1, mode="edge")
            nb = np.stack([Rp[:-2, 1:-1], Rp[2:, 1:-1], Rp[1:-1, :-2], Rp[1:-1, 2:]])
            fill = np.nanmean(np.where(np.isnan(nb), np.nan, nb), axis=0)
            R[bad] = fill[bad]
        R[np.isnan(R)] = default
        return R

    def nearest(self, p):
        loc, nor, fi, d = self.bvh.find_nearest(Vector(p))
        return np.array(loc[:]), np.array(nor[:]), d


def blur_field(R, sig_t, sig_th, wrap_th=True, iters=None):
    """Separable gaussian blur of a (nt, nth) field; wraps in theta."""
    from math import exp
    def kern(s):
        if s <= 0:
            return np.array([1.0])
        r = int(max(1, round(3 * s)))
        k = np.array([exp(-0.5 * (i / s) ** 2) for i in range(-r, r + 1)])
        return k / k.sum()
    out = R.copy()
    k = kern(sig_th)
    r = len(k) // 2
    if len(k) > 1:
        P = np.pad(out, ((0, 0), (r, r)), mode="wrap" if wrap_th else "edge")
        out = sum(k[i] * P[:, i:i + out.shape[1]] for i in range(len(k)))
    k = kern(sig_t)
    r = len(k) // 2
    if len(k) > 1:
        P = np.pad(out, ((r, r), (0, 0)), mode="edge")
        out = sum(k[i] * P[i:i + out.shape[0], :] for i in range(len(k)))
    return out


def envelope(R, size_t=2, size_th=2, sig_t=1.5, sig_th=1.5, wrap_th=True):
    """Armour envelope of a body radius field: grey closing (max filter then blur) = convex-ish, no dips."""
    M = R.copy()
    for d in range(1, size_th + 1):
        M = np.maximum(M, np.roll(R, d, 1) if wrap_th else np.pad(R, ((0, 0), (d, 0)), mode="edge")[:, :R.shape[1]])
        M = np.maximum(M, np.roll(R, -d, 1) if wrap_th else np.pad(R, ((0, 0), (0, d)), mode="edge")[:, d:])
    M2 = M.copy()
    for d in range(1, size_t + 1):
        M2 = np.maximum(M2, np.pad(M, ((d, 0), (0, 0)), mode="edge")[:M.shape[0]])
        M2 = np.maximum(M2, np.pad(M, ((0, d), (0, 0)), mode="edge")[d:])
    return blur_field(M2, sig_t, sig_th, wrap_th)


class RadField:
    """Body radius around an axis: R(t, theta) sampled with rays A(t) + r * (cos th * front(t) + sin th * left(t)),
    enveloped (no dips) and blurred; evaluated with bilinear interpolation (periodic in theta)."""

    def __init__(self, body, axis_fn, frame_fn, t0, t1, nt=48, nth=72, env=(2, 2, 1.5, 1.5), maxd=0.5, default=0.05):
        self.ts = np.linspace(t0, t1, nt)
        self.ths = np.linspace(-np.pi, np.pi, nth, endpoint=False)
        self.axis_fn, self.frame_fn = axis_fn, frame_fn
        A = [np.asarray(axis_fn(t)) for t in self.ts]
        fr = [frame_fn(t) for t in self.ts]
        R = body.radius_field(A, fr, self.ths, default=default, maxd=maxd)
        self.raw = R
        self.R = envelope(R, *env) if env else R

    def __call__(self, t, th):
        t = np.asarray(t, float); th = np.asarray(th, float)
        t, th = np.broadcast_arrays(t, th)
        ft = np.clip((t - self.ts[0]) / (self.ts[-1] - self.ts[0]) * (len(self.ts) - 1), 0, len(self.ts) - 1.0001)
        i0 = np.floor(ft).astype(int); a = ft - i0
        n = len(self.ths)
        fth = ((th + np.pi) % (2 * np.pi)) / (2 * np.pi) * n
        j0 = np.floor(fth).astype(int) % n; b = fth - np.floor(fth); j1 = (j0 + 1) % n
        R = self.R
        return ((1 - a) * ((1 - b) * R[i0, j0] + b * R[i0, j1]) + a * ((1 - b) * R[i0 + 1, j0] + b * R[i0 + 1, j1]))

    def point(self, t, th, r):
        t = np.asarray(t, float); th = np.asarray(th, float); r = np.asarray(r, float)
        t, th, r = np.broadcast_arrays(t, th, r)
        out = np.zeros(t.shape + (3,))
        for idx in np.ndindex(t.shape):
            A = np.asarray(self.axis_fn(t[idx])); f, l = self.frame_fn(t[idx])
            out[idx] = A + r[idx] * (math.cos(th[idx]) * f + math.sin(th[idx]) * l)
        return out


# ------------------------------------------------------------------------------------------------------ mesh builder
class MB:
    """Accumulates quads with per-corner UVs; per-vertex weights {bone: w} and tags."""

    def __init__(self, name):
        self.name = name
        self.V = []
        self.F = []
        self.UV = []
        self.W = []
        self.tag = []
        self.loopinfo = []           # every swept rim loop (positions, normals, outward binormal, amp) for decals

    def add(self, P, w=None, tag=""):
        P = np.asarray(P, dtype=np.float64)
        shp = P.shape[:-1]
        flat = P.reshape(-1, 3)
        i0 = len(self.V)
        self.V.extend(list(flat))
        if callable(w):
            ws = w(flat)
        elif isinstance(w, (list, tuple)):
            ws = list(w)
        else:
            ws = [dict(w or {})] * len(flat)
        assert len(ws) == len(flat), (len(ws), len(flat))
        self.W.extend([dict(x) for x in ws])
        self.tag.extend([tag] * len(flat))
        return (np.arange(len(flat)) + i0).reshape(shp)

    def quad(self, a, b, c, d, uv):
        self.F.append((int(a), int(b), int(c), int(d)))
        self.UV.append([tuple(map(float, x)) for x in uv])

    @property
    def nv(self):
        return len(self.V)

    def pos(self, i):
        return self.V[i]

    def to_object(self, name=None, collection=None):
        name = name or self.name
        me = bpy.data.meshes.new(name)
        V = np.array(self.V)
        me.from_pydata([tuple(v) for v in V], [], self.F)
        me.update()
        uvl = me.uv_layers.new(name="UVMap")
        uv = np.array([c for f in self.UV for c in f], dtype=np.float64)
        # from_pydata keeps face order and corner order
        uvl.data.foreach_set("uv", uv.ravel())
        for p in me.polygons:
            p.use_smooth = True
        ob = bpy.data.objects.new(name, me)
        (collection or bpy.context.scene.collection).objects.link(ob)
        groups = {}
        for i, w in enumerate(self.W):
            for bn, x in w.items():
                if x > 1e-4:
                    groups.setdefault(bn, []).append((i, x))
        for bn, lst in groups.items():
            g = ob.vertex_groups.new(name=bn)
            for i, x in lst:
                g.add([i], x, 'REPLACE')
        return ob


# ------------------------------------------------------------------------------------------------------ grids
def grid_normals(P, closed=False):
    """Vertex normals of a (nu, nv, 3) grid (central differences), un-oriented (cross(dU, dV))."""
    if closed:
        dU = np.roll(P, -1, axis=0) - np.roll(P, 1, axis=0)
    else:
        dU = np.zeros_like(P)
        dU[1:-1] = P[2:] - P[:-2]; dU[0] = P[1] - P[0]; dU[-1] = P[-1] - P[-2]
    dV = np.zeros_like(P)
    dV[:, 1:-1] = P[:, 2:] - P[:, :-2]; dV[:, 0] = P[:, 1] - P[:, 0]; dV[:, -1] = P[:, -1] - P[:, -2]
    return nrm(np.cross(dU, dV))


def grid_arclen(P, closed=False):
    """(U, V) arc-length parameters of a grid: U along axis 0 (averaged over rows), V along axis 1 per column.
    For closed grids U has nu+1 entries (seam)."""
    Q = np.concatenate([P, P[:1]], axis=0) if closed else P
    du = np.linalg.norm(np.diff(Q, axis=0), axis=-1)            # (nu-1|nu, nv)
    U = np.concatenate([np.zeros((1, Q.shape[1])), np.cumsum(du, axis=0)], axis=0)
    dv = np.linalg.norm(np.diff(Q, axis=1), axis=-1)
    Vv = np.concatenate([np.zeros((Q.shape[0], 1)), np.cumsum(dv, axis=1)], axis=1)
    return U, Vv



def laplace_grid(P, iters=3, lam=0.5, closed=False, fix_border=True):
    P = P.copy()
    for _ in range(iters):
        if closed:
            nb = np.roll(P, 1, 0) + np.roll(P, -1, 0)
        else:
            nb = np.zeros_like(P); nb[1:-1] = P[:-2] + P[2:]; nb[0] = 2 * P[0]; nb[-1] = 2 * P[-1]
        nb2 = np.zeros_like(P); nb2[:, 1:-1] = P[:, :-2] + P[:, 2:]; nb2[:, 0] = 2 * P[:, 0]; nb2[:, -1] = 2 * P[:, -1]
        Q = P + lam * ((nb + nb2) / 4 - P)
        if fix_border:
            Q[:, 0] = P[:, 0]; Q[:, -1] = P[:, -1]
            if not closed:
                Q[0] = P[0]; Q[-1] = P[-1]
        P = Q
    return P


# ------------------------------------------------------------------------------------------------------ rim profiles
def prof(*steps):
    """Rim profile: steps (db, dn, band, v0, v1) -> ring k+1 at in-surface offset db and normal offset dn (metres, relative
    to the border vertex); the quad row from ring k to k+1 maps to band `band` from v0 to v1 (fractions)."""
    return list(steps)


LEAN = True          # game LOD0 profiles (fewer rings); False = denser cutscene profiles
_OLD_RIMS = os.environ.get("RTS_OLD_RIMS", "").split(",")   # dev comparison only: b0, lb = the pre-session-4 rims


def rim_gold(w=0.012, bead=0.0035, t=0.003, lip=0.008, band="fil_narrow", bead_band="gold_bead", step=0.0012):
    """Raised engraved gold band of width w, then a rolled bead, then the edge turned under into a lip."""
    r = bead
    if LEAN:
        # session 4 (integrity G2 'spikes'): the first bevel row was 1.2 mm wide: on plate borders with 20-40 mm
        # segments its quads split into needle triangles folded 45-90 deg; it now takes ~a third of the band (the
        # rim's total width is unchanged)
        b0 = min(0.004, 0.0012 + 0.35 * w) if "b0" not in _OLD_RIMS else 0.0012
        steps = [
            (b0, step, "gold_plain", 0.1, 0.4),
            (0.0012 + w, step, band, 0.0, 1.0),
            (0.0012 + w + 0.9 * r, step + 1.15 * r, bead_band, 0.0, 0.5),
            (0.0012 + w + 1.7 * r, step - 0.2 * r, bead_band, 0.5, 1.0),
        ]
        if lip > 0:
            steps.append((0.0012 + w + 0.6 * r - lip, -t - 0.001, "steel_plain", 0.0, 1.0))
        else:
            steps.append((0.0012 + w + 1.2 * r, -t, "steel_plain", 0.0, 0.3))
        return prof(*steps)
    return prof(
        (0.0006, step * 0.7, "gold_plain", 0.0, 0.25),
        (0.0015, step, "gold_plain", 0.25, 0.5),
        (0.0015 + w, step, band, 0.0, 1.0),
        (0.0015 + w + 0.35 * r, step + 0.9 * r, bead_band, 0.0, 0.3),
        (0.0015 + w + 1.1 * r, step + 1.25 * r, bead_band, 0.3, 0.55),
        (0.0015 + w + 1.8 * r, step + 0.35 * r, bead_band, 0.55, 0.8),
        (0.0015 + w + 1.5 * r, -t, bead_band, 0.8, 1.0),
        (0.0015 + w + 0.6 * r - lip, -t - 0.001, "steel_plain", 0.0, 1.0),
    )


def rim_bead(bead=0.003, t=0.003, lip=0.006, band="gold_bead"):
    """Plain rolled edge (no flat band)."""
    r = bead
    if LEAN:
        return prof((0.7 * r, 1.1 * r, band, 0.0, 0.5), (1.6 * r, -0.2 * r, band, 0.5, 1.0),
                    (0.8 * r - lip, -t - 0.0005, "steel_plain", 0.0, 1.0))
    return prof(
        (0.35 * r, 0.9 * r, band, 0.0, 0.3),
        (1.1 * r, 1.2 * r, band, 0.3, 0.55),
        (1.8 * r, 0.3 * r, band, 0.55, 0.8),
        (1.5 * r, -t, band, 0.8, 1.0),
        (0.6 * r - lip, -t - 0.001, "steel_plain", 0.0, 1.0),
    )


def rim_chamfer(t=0.003, depth=0.004, band="steel_plain", c=0.0012):
    """Small chamfer turning into the hole / under the plate (slits, vents, finger plates)."""
    return prof(
        (c, -c * 0.6, band, 0.0, 0.3),
        (c * 1.2, -t, band, 0.3, 0.7),
        (c * 1.2 - depth, -t - 0.0005, "dark", 0.0, 1.0),
    )


# ------------------------------------------------------------------------------------------------------ shell
def boundary_loops(faces):
    """Directed boundary loops (face on the left) of a quad list."""
    he = {}
    for f in faces:
        for k in range(4):
            a, b = f[k], f[(k + 1) % 4]
            he[(a, b)] = True
    nxt = {}
    for (a, b) in he:
        if (b, a) not in he:
            nxt[a] = b
    loops, seen = [], set()
    for s in list(nxt):
        if s in seen:
            continue
        loop = [s]; seen.add(s)
        c = nxt[s]
        while c != s:
            loop.append(c); seen.add(c)
            c = nxt[c]
            if len(loop) > 100000:
                raise RuntimeError("bad boundary")
        loops.append(loop)
    return loops


def sweep_loop(mb, loop, P, N, Q, profile, u0=0.0, lip_inner=None, closed=True, weight_of=None, tag="", amp=None,
               dbscale=None, sphere=None):
    """Sweep `profile` around the ordered boundary `loop` (vertex indices into mb) with positions P (len(loop),3),
    normals N and interior-neighbour points Q (for curvature). amp (len(loop),) scales the raised (dn > 0) part of the
    profile per vertex: 0 on edges hidden under another plate (a flat turned edge, no bead to catch the plate above),
    1 on visible edges. Returns the last ring's vertex indices."""
    L = len(loop)
    idx = np.arange(L)
    prv = np.roll(idx, 1) if closed else np.maximum(idx - 1, 0)
    nxt = np.roll(idx, -1) if closed else np.minimum(idx + 1, L - 1)
    e1 = nrm(P - P[prv]); e2 = nrm(P[nxt] - P)
    if not closed:
        e1[0] = e2[0]; e2[-1] = e1[-1]
    b1 = nrm(np.cross(e1, N)); b2 = nrm(np.cross(e2, N))
    b = nrm(b1 + b2)
    mit = 1.0 / np.maximum(0.45, np.sum(b * b1, -1))
    # signed normal curvature across the border (quadratic continuation of the surface)
    d = Q - P
    hb = np.maximum(-np.sum(d * b, -1), 1e-4)
    kap = np.clip(2 * np.sum(d * N, -1) / hb ** 2, -60, 60)
    seg = np.linalg.norm(np.diff(np.concatenate([P, P[:1]]) if closed else P, axis=0), axis=-1)
    arc = np.concatenate([[0.0], np.cumsum(seg)])
    prev = np.array(loop)
    pdb, pdn = 0.0, 0.0
    A = np.ones(L) if amp is None else np.asarray(amp, float)
    # iteration 2b: hidden edges (amp 0) get a narrow flange (35 %) instead of the full band width, and the curvature
    # continuation of wide rims is capped (a dip of at most 0.13 x the band width): on strongly curved plate edges
    # (torso sides, a collar's top) the full quadratic continuation dived up to 12 mm into the plate underneath
    Sd = (0.35 + 0.65 * np.clip(A, 0, 1)) if dbscale is None else np.asarray(dbscale, float)
    prev_p = np.asarray(P, float)
    for (db, dn, band, v0, v1) in profile:
        width = max(math.hypot(db - pdb, dn - pdn), 1e-4)
        pdb, pdn = db, dn
        dbm = db * mit * Sd
        dnv = np.where(dn > 0, dn * A, dn) if np.ndim(dn) == 0 else dn
        if sphere is not None:
            # iteration 2b: a rim on a plate that is a sphere round a joint (couter, pauldron shells, flares over
            # them) FOLLOWS that sphere: move along the surface, then back onto the radius |P - C| + dn, so plates
            # sliding over it keep their clearance (a tangent continuation bulged 12 mm off the couter)
            C = np.asarray(sphere, float)
            rad = np.linalg.norm(P - C, axis=1)
            q = P + dbm[:, None] * b
            ring_p = C + nrm(q - C) * (rad + dnv)[:, None]
        else:
            curv = np.maximum(0.5 * kap * np.clip(dbm, 0, None) ** 2, -0.13 * np.abs(dbm))
            curv = np.minimum(curv, 0.13 * np.abs(dbm))
            ring_p = P + dbm[:, None] * b + (dnv + curv)[:, None] * N
        # iteration 2b (G6 anisotropy): U density per vertex = this row's V density over its ACTUAL width (miter,
        # hidden-edge narrowing and sphere projection change the nominal width), along the row's mid line
        wv = np.maximum(np.linalg.norm(ring_p - prev_p, axis=1), 1e-4)
        if band in TX.BANDS:
            dv = np.clip(TX.band_px(band) * abs(v1 - v0) / wv, 0.25 * TX.TRIM_DENS, 6 * TX.TRIM_DENS)
        else:
            dv = np.full(L, TX.TRIM_DENS)
        midl = 0.5 * (ring_p + prev_p)
        sl = np.linalg.norm(np.diff(np.concatenate([midl, midl[:1]]) if closed else midl, axis=0), axis=-1)
        dseg = 0.5 * (dv + np.roll(dv, -1))[:len(sl)]
        ucum = np.concatenate([[0.0], np.cumsum(sl * dseg)]) / TX.W + u0
        # shear: each ring vertex's offset from the mid line along the loop tangent (mitered corners, sphere rims)
        Tt = nrm(e1 + e2)
        sh_prev = np.sum((prev_p - midl) * Tt, -1) * dv / TX.W
        sh_ring = np.sum((ring_p - midl) * Tt, -1) * dv / TX.W
        prev_p = ring_p
        ws = [mb.W[i] for i in loop]
        ring = mb.add(ring_p, w=lambda pts, ws=ws: ws, tag=tag)
        Vh, Vl = TX.band_v(band, v0), TX.band_v(band, v1)
        nseg = L if closed else L - 1
        for i in range(nseg):
            j = (i + 1) % L
            ua, ub = ucum[i], ucum[i + 1]
            upa, upb = ua + sh_prev[i], ub + sh_prev[j]
            ura, urb = ua + sh_ring[i], ub + sh_ring[j]
            # the surface face walks the border edge i->j, so the rim quad walks it j->i (consistent winding)
            mb.quad(prev[j], prev[i], ring[i], ring[j], [(upb, Vh), (upa, Vh), (ura, Vl), (urb, Vl)])
        prev = ring
    mb.loopinfo.append(dict(P=P.copy(), N=N.copy(), b=b.copy(), kap=kap.copy(), amp=A.copy(), tag=tag,
                            W=[dict(mb.W[i]) for i in loop]))
    return prev


def shell(mb, P, *, closed=False, mask=None, inside=None, uvfn=None, rim=None, inner=False, thick=0.003, w=None,
          tag="", smooth_normals=0, inner_uv=None, etch=None, sphere=None):
    """Plate from a (nu, nv, 3) outer-surface grid. mask[(i, j)] selects faces (holes = False). inside: a point inside
    the body (orients normals outward). uvfn(U, V, P) -> (nu[+1], nv, 2) UVs from arc lengths (defaults to the STEEL
    region). rim(loop_positions, loop_normals) -> profile for that boundary loop (None = rim_gold()). w: weights (dict
    or callable on points). inner=True: a CLOSED thick plate (inner surface `thick` under the outer one, every rim turned
    under to it with rim_closed). etch: [dict(axis, rng=(lo, hi), border, band, width)] strips of grid faces between a
    border line and a line `width` inside it (insert_line) mapped to an etch band where the rim is visible (amp >= 0.5).
    Returns dict(idx, normals, loops)."""
    nu, nv = P.shape[:2]
    ncol = nu if closed else nu - 1
    if mask is None:
        mask = np.ones((ncol, nv - 1), bool)
    N = grid_normals(P, closed)
    if inside is not None:
        # inside: a point inside the body, or a callable giving per-vertex inside points (e.g. the foot on a limb /
        # neck axis): one fixed point misorients flared rings whose axis runs past it (the old gorget collar)
        ins = inside(P.reshape(-1, 3)).reshape(P.shape) if callable(inside) else np.asarray(inside)
        if np.mean(np.sum(N * (P - ins), -1)) < 0:
            N = -N
    flip = False
    # face winding consistent with N: cross(dU, dV) orientation of the raw grid
    raw = grid_normals(P, closed)
    if np.mean(np.sum(raw * N, -1)) < 0:
        flip = True
    used = np.zeros((nu, nv), bool)
    for i in range(ncol):
        for j in range(nv - 1):
            if mask[i, j]:
                i2 = (i + 1) % nu
                used[i, j] = used[i2, j] = used[i, j + 1] = used[i2, j + 1] = True
    idx = -np.ones((nu, nv), dtype=np.int64)
    pts = P[used]
    ids = mb.add(pts, w=w, tag=tag)
    idx[used] = ids
    U, Vv = grid_arclen(P, closed)
    if uvfn is None:
        uv = TX.steel_uv(U, Vv)
    else:
        uv = uvfn(U, Vv, P)
    faces = []
    fidx = {}
    for i in range(ncol):
        for j in range(nv - 1):
            if not mask[i, j]:
                continue
            i2 = (i + 1) % nu
            a, b_, c, d = idx[i, j], idx[i2, j], idx[i2, j + 1], idx[i, j + 1]
            ua = uv[i, j]; ub = uv[i + 1 if closed else i2, j]; uc = uv[i + 1 if closed else i2, j + 1]; ud = uv[i, j + 1]
            if flip:
                f = (a, d, c, b_); fu = [ua, ud, uc, ub]
            else:
                f = (a, b_, c, d); fu = [ua, ub, uc, ud]
            fidx[(i, j)] = len(mb.F)
            mb.quad(*f, fu)
            faces.append(f)
    # per mb-vertex normal and interior neighbour
    vN = {}
    for i in range(nu):
        for j in range(nv):
            if idx[i, j] >= 0:
                vN[idx[i, j]] = N[i, j]
    nbr = {}
    loops = boundary_loops(faces)
    bset = set(v for lp in loops for v in lp)
    for f in faces:
        for k in range(4):
            a = f[k]
            for bb in (f[(k + 1) % 4], f[(k + 3) % 4], f[(k + 2) % 4]):
                if a in bset and bb not in bset:
                    nbr.setdefault(a, []).append(mb.V[bb])
    out_loops = []
    vamp = {}
    for lp in loops:
        Pl = np.array([mb.V[i] for i in lp]); Nl = np.array([vN[i] for i in lp])
        Ql = np.array([np.mean(nbr[i], axis=0) if i in nbr else Pl[k] - 0.01 * Nl[k] for k, i in enumerate(lp)])
        profile = rim(Pl, Nl) if rim else rim_gold()
        amp = None; dbs = None; sph = sphere
        if isinstance(profile, tuple):
            if len(profile) == 4:
                profile, amp, dbs, sph = profile
            elif len(profile) == 3:
                profile, amp, dbs = profile
            else:
                profile, amp = profile
        if profile is None:
            out_loops.append((lp, None))           # open border (e.g. closed later by a cap): no weld
            continue
        if inner and not (abs(profile[-1][0]) < 1e-9 and abs(profile[-1][1] + thick) < 1e-9):
            profile = rim_closed(profile, thick)
        for k, i in enumerate(lp):
            vamp[i] = 1.0 if amp is None else float(amp[k])
        last = sweep_loop(mb, lp, Pl, Nl, Ql, profile, tag=tag, amp=amp, dbscale=dbs, sphere=sph)
        out_loops.append((lp, last))
    if etch:
        _etch_uvs(mb, P, idx, fidx, flip, closed, etch, vamp, ncol, nv)
    if inner:
        # inner surface (reversed) connected to the last rim rings: offset the grid inward and weld the border
        # vertices to the rim's last ring (rim profiles for inner shells must end at dn = -thick, db = 0)
        pin = P - thick * N
        idx2 = -np.ones((nu, nv), dtype=np.int64)
        border_map = {}
        for lp, last in out_loops:
            if last is None:
                continue
            for a, bl in zip(lp, last):
                border_map[a] = bl
        for i in range(nu):
            for j in range(nv):
                if idx[i, j] >= 0:
                    a = idx[i, j]
                    if a in border_map:
                        idx2[i, j] = border_map[a]
                    else:
                        idx2[i, j] = mb.add(pin[i:i + 1, j], w=[mb.W[a]], tag=tag)[0]
        uvi = uv if inner_uv is None else inner_uv(U, Vv, pin)
        for i in range(ncol):
            for j in range(nv - 1):
                if not mask[i, j]:
                    continue
                i2 = (i + 1) % nu
                a, b_, c, d = idx2[i, j], idx2[i2, j], idx2[i2, j + 1], idx2[i, j + 1]
                ua = uvi[i, j]; ub = uvi[i + 1 if closed else i2, j]; uc = uvi[i + 1 if closed else i2, j + 1]; ud = uvi[i, j + 1]
                if flip:
                    mb.quad(a, b_, c, d, [ua, ub, uc, ud])
                else:
                    mb.quad(a, d, c, b_, [ua, ud, uc, ub])
    return dict(idx=idx, N=N, loops=out_loops, idx2=idx2 if inner else None)


def _etch_uvs(mb, P, idx, fidx, flip, closed, etch, vamp, ncol, nv):
    """re-map the grid faces of each etch strip to its trim band (V across the strip from the border, U along the
    border at the band's isotropic density), where the border's rim is visible"""
    nu = P.shape[0]
    for e in etch:
        ax, (lo, hi), bd, band, width = e["axis"], e["rng"], e["border"], e.get("band", "etch_band"), e["width"]
        dens = TX.band_px(band) / max(width, 1e-4)
        if ax == 1:
            # along the border row j = bd, per column i
            inner_line = lo if bd == hi else hi
            arc = _strip_arc(P, bd, inner_line, band, width, closed)
            for i in range(ncol):
                i2 = (i + 1) % nu
                bv = [idx[i, bd], idx[i2, bd]]
                if min(bv) < 0 or np.mean([vamp.get(int(v), 0.0) for v in bv]) < 0.5:
                    continue
                for j in range(lo, hi):
                    if (i, j) not in fidx:
                        continue
                    def vv(ii, jj, i_arc):
                        col = P[ii]
                        s = abs(np.sum(np.linalg.norm(np.diff(col[min(jj, bd):max(jj, bd) + 1], axis=0), axis=-1)))
                        stot = np.sum(np.linalg.norm(np.diff(col[min(inner_line, bd):max(inner_line, bd) + 1], axis=0), axis=-1))
                        return (arc[i_arc, jj], TX.band_v(band, 1.0 - min(1.0, s / max(stot, 1e-9))))
                    ia = i + 1 if closed else i2
                    ua, ub, uc, ud = vv(i, j, i), vv(i2, j, ia), vv(i2, j + 1, ia), vv(i, j + 1, i)
                    mb.UV[fidx[(i, j)]] = [ua, ud, uc, ub] if flip else [ua, ub, uc, ud]
        else:
            inner_line = lo if bd == hi else hi
            arc = _strip_arc(P.transpose(1, 0, 2), bd, inner_line, band, width, False)
            for j in range(nv - 1):
                bv = [idx[bd, j], idx[bd, j + 1]]
                if min(bv) < 0 or np.mean([vamp.get(int(v), 0.0) for v in bv]) < 0.5:
                    continue
                for i in range(lo, hi):
                    if (i, j) not in fidx:
                        continue
                    def vv(ii, jj):
                        row = P[:, jj]
                        s = np.sum(np.linalg.norm(np.diff(row[min(ii, bd):max(ii, bd) + 1], axis=0), axis=-1))
                        stot = np.sum(np.linalg.norm(np.diff(row[min(inner_line, bd):max(inner_line, bd) + 1], axis=0), axis=-1))
                        return (arc[jj, ii], TX.band_v(band, 1.0 - min(1.0, s / max(stot, 1e-9))))
                    ua, ub, uc, ud = vv(i, j), vv(i + 1, j), vv(i + 1, j + 1), vv(i, j + 1)
                    mb.UV[fidx[(i, j)]] = [ua, ud, uc, ub] if flip else [ua, ub, uc, ud]


def _strip_arc(P, bd, il, band, width, closed):
    """(nu, nv) U (texture units) of the strip between grid rows bd and il: along the strip's MID line at the band's
    isotropic density over its PERPENDICULAR local width, plus each vertex's offset along the border tangent (so
    columns that run obliquely to the border do not shear the pattern). Iteration 2b, G6: the border arc at the
    nominal width stretched / sheared the etch 1.5-2.5x on curved, clamped or oblique strips."""
    nu = P.shape[0]
    B = P[:, bd]; Mid = 0.5 * (P[:, bd] + P[:, il])
    if closed:
        T = nrm(np.roll(B, -1, 0) - np.roll(B, 1, 0))
    else:
        T = np.zeros_like(B); T[1:-1] = B[2:] - B[:-2]; T[0] = B[1] - B[0]; T[-1] = B[-1] - B[-2]; T = nrm(T)
    c = P[:, il] - P[:, bd]
    wp = np.maximum(np.linalg.norm(c - np.sum(c * T, -1, keepdims=True) * T, axis=-1), 1e-4)
    dens = np.clip(TX.band_px(band) / wp, 0.25 * TX.TRIM_DENS, 6 * TX.TRIM_DENS)
    Mq = np.concatenate([Mid, Mid[:1]]) if closed else Mid
    dq = np.concatenate([dens, dens[:1]]) if closed else dens
    sl = np.linalg.norm(np.diff(Mq, axis=0), axis=-1)
    arc = np.concatenate([[0.0], np.cumsum(sl * 0.5 * (dq[:-1] + dq[1:]))]) / TX.W
    U = np.zeros(P.shape[:2] if not closed else (nu + 1, P.shape[1]))
    for i in range(U.shape[0]):
        ii = i % nu
        U[i] = arc[i] + ((P[ii] - Mid[ii]) @ T[ii]) * dens[ii] / TX.W
    return U


def etch_grid(P, mask, sides, width, band="etch_band", closed=False, snap=False):
    """insert the etch lines for `sides` (subset of 'j0', 'j1', 'i0', 'i1') into grid P (+ mask); returns
    (P, mask, etch spec list for shell())"""
    spec = []
    for sd in sides:
        ax = 1 if sd[0] == "j" else 0
        side = int(sd[1])
        P, mask, k, rng = insert_line(P, mask, ax, side, width, closed=closed, snap=snap)
        # earlier strips' ranges shift when a line is inserted before them on the same axis (none when an existing
        # line was moved instead: k is None)
        for e in (spec if k is not None else []):
            if e["axis"] == ax and k <= e["rng"][0]:
                e["rng"] = (e["rng"][0] + 1, e["rng"][1] + 1); e["border"] += 1
            elif e["axis"] == ax and e["rng"][0] < k < e["rng"][1]:
                e["rng"] = (e["rng"][0], e["rng"][1] + 1)
                if e["border"] >= k:
                    e["border"] += 1
        border = rng[0] if side == 0 else rng[1]
        spec.append(dict(axis=ax, rng=rng, border=border, band=band, width=width))
    return P, mask, spec


def rim_inner(profile, thick):
    """Make a rim profile end exactly on the inner surface border (db = 0, dn = -thick)."""
    p = [s for s in profile if s[1] > -thick * 0.9]
    last = p[-1]
    p.append((0.0, -thick, "steel_plain", 0.0, 1.0))
    return p


def rim_closed(profile, thick, lip=None):
    """Iteration 2b (user item 28): a rim that CLOSES the plate. The raised part of the authored profile (step, band,
    bead) is kept, then the edge is turned under to the inner-surface level (dn = -thick) `lip` metres back from the
    bead, and runs back to the inner border (db = 0, dn = -thick): a thick rolled edge that welds to the inner surface
    (shell(..., inner=True, thick=thick)). Authored lips deeper than the inner surface are dropped."""
    p = [s for s in profile if s[1] > -0.9 * thick]
    db_e = max(s[0] for s in p)
    # session 4 (G2 spikes): back 40 % of the rim width (was min(3 mm, 70 %): the flat return row of a 2.6-3.4 mm bead
    # rim was 1.2-2.4 mm wide, needle triangles folded 90 deg)
    if lip is not None:
        lb = min(lip, 0.7 * db_e)
    else:
        lb = (0.4 * db_e if db_e > 0.004 else min(0.003, 0.7 * db_e)) if "lb" not in _OLD_RIMS else min(0.003, 0.7 * db_e)
    p.append((db_e - lb, -thick, "steel_plain", 0.0, 0.5))
    p.append((0.0, -thick, "steel_plain", 0.5, 1.0))
    return p


def insert_line(P, mask, axis, side, width, closed=False, snap=False):
    """Insert a grid line at arc length `width` from the border line (axis 1: rows j, side 0 = j=0 / 1 = j=nv-1;
    axis 0: columns i of an open grid), measured along each grid line, so an etch strip of that width sits on the plate
    surface itself (iteration 2b: the floating decal ribbons were one-sided, z-fought and got half buried). Returns
    (P', mask', index of the new line, strip range (lo, hi) of line indices between the border and the new line)."""
    P = np.asarray(P, float)
    if axis == 0:
        Pt, Mt, k, rng = insert_line(P.transpose(1, 0, 2), None if mask is None else mask.T, 1, side, width, snap=snap)
        return Pt.transpose(1, 0, 2), (None if Mt is None else Mt.T), k, rng
    nu, nv = P.shape[:2]
    Q = P if side == 0 else P[:, ::-1]
    seg = np.linalg.norm(np.diff(Q, axis=1), axis=-1)                 # (nu, nv-1)
    cum = np.concatenate([np.zeros((nu, 1)), np.cumsum(seg, 1)], 1)   # (nu, nv)
    w = np.minimum(width, cum[:, -1] * 0.45)
    # the strip index where the width falls (the same for every column: the max over columns, clamped)
    ks = np.array([int(np.searchsorted(cum[i], w[i])) for i in range(nu)])
    k = int(np.clip(np.median(ks), 1, nv - 1))
    new = np.zeros((nu, 3))
    tt = np.zeros(nu)
    for i in range(nu):
        # position at arc length w[i] along column i (clamped into strip k-1..k)
        a, b = cum[i, k - 1], cum[i, k]
        t = float(np.clip((w[i] - a) / max(b - a, 1e-9), 0.05, 0.95))
        tt[i] = t
        new[i] = Q[i, k - 1] + t * (Q[i, k] - Q[i, k - 1])
    # session 4 (integrity G2 spikes, couter / vambrace folds): a new line within 25 % of an existing interior line
    # makes a sliver row (0.2-0.5 mm) that folds; snap=True MOVES that existing line to the etch width instead. OFF by
    # default: the decision depends on the body, so the female vambrace / gauntlet got a different topology from the
    # authoring (male) body and regen_on_body kept their MakeClothes fit (couter|vambrace cut on the female)
    tm = float(np.median(tt))
    allow = snap
    snap = None
    if allow and tm < 0.25 and k - 1 >= 1:
        snap = k - 1
    elif allow and tm > 0.75 and k <= nv - 2:
        snap = k
    if snap is not None:
        Q2 = Q.copy(); Q2[:, snap] = new
        if side == 1:
            m = nv - 1 - snap
            return Q2[:, ::-1], mask, None, (m, nv - 1)
        return Q2, mask, None, (0, snap)
    Q2 = np.concatenate([Q[:, :k], new[:, None], Q[:, k:]], 1)
    M2 = None
    if mask is not None:
        Mq = mask if side == 0 else mask[:, ::-1]
        M2 = np.concatenate([Mq[:, :k], Mq[:, k - 1:k], Mq[:, k:]], 1)
        if side == 1:
            M2 = M2[:, ::-1]
    if side == 1:
        Q2 = Q2[:, ::-1]
        kk = nv - k
        return Q2, M2, kk, (kk, nv)
    return Q2, M2, k, (0, k)


# ------------------------------------------------------------------------------------------------------ sweep
def sweep(mb, path, normals, profile, *, closed=False, band="gold_plain", taper=0.0, w=None, tag="", u0=0.0,
          across=None, ends=True, solid=False, seat=None):
    """Sweep an open cross-section `profile` [(x_lateral, y_up), ...] along `path` (M, 3) with surface `normals`.
    The profile is mapped across `band` (v from 0 at the first profile point to 1 at the last). taper: fraction of the
    length over which the profile scales to 0 at open ends. across: optional (M, 3) lateral directions.
    solid=True (iteration 2b): a CLOSED rib: the profile's end points are joined along the bottom (lying on the
    surface) and open paths get end caps, so it has an inside (no one-sided strip) and sits on the plate."""
    if solid:
        return _sweep_solid(mb, path, normals, profile, closed=closed, band=band, taper=taper, w=w, tag=tag, u0=u0,
                            across=across, seat=seat)
    path = np.asarray(path, float); normals = nrm(normals)
    M = len(path)
    if closed:
        T = nrm(np.roll(path, -1, 0) - np.roll(path, 1, 0))
    else:
        T = np.zeros_like(path); T[1:-1] = path[2:] - path[:-2]; T[0] = path[1] - path[0]; T[-1] = path[-1] - path[-2]
        T = nrm(T)
    Bl = nrm(np.cross(normals, T)) if across is None else nrm(across)
    seg = np.linalg.norm(np.diff(np.concatenate([path, path[:1]]) if closed else path, axis=0), axis=-1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    Ltot = s[-1] if closed else s[-1]
    scale = np.ones(M)
    if taper > 0 and not closed:
        t = s[:M] / max(Ltot, 1e-6)
        scale = smoothstep(0, taper, t) * smoothstep(0, taper, 1 - t)
        scale = 0.15 + 0.85 * scale
    prof_ = np.asarray(profile, float)
    K = len(prof_)
    lat = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(prof_, axis=0), axis=-1))])
    vfr = lat / max(lat[-1], 1e-9)
    rings = []
    for k in range(K):
        x, y = prof_[k]
        pts = path + (x * scale)[:, None] * Bl + (y * scale)[:, None] * normals
        rings.append(mb.add(pts, w=w, tag=tag))
    ptot = max(lat[-1], 1e-4)
    dens = TX.band_px(band) / ptot if band in TX.BANDS else TX.TRIM_DENS
    ucum = s * dens / TX.W + u0
    nseg = M if closed else M - 1
    # winding: the quads of the profile's middle segment must face along `normals` (outward); with an explicit
    # `across` direction the natural winding depends on the path direction (the helmet comb / prow rib faced inward)
    km = max(0, (K - 1) // 2 - (1 if K % 2 == 0 else 0))
    flips = 0.0
    for i in range(nseg):
        j = (i + 1) % M
        a_, b_, d_ = (np.asarray(mb.V[rings[km][i]]), np.asarray(mb.V[rings[km][j]]), np.asarray(mb.V[rings[km + 1][i]]))
        flips += float(np.dot(np.cross(b_ - a_, d_ - a_), normals[i]))
    flip = flips < 0
    for k in range(K - 1):
        Vh, Vl = TX.band_v(band, vfr[k]), TX.band_v(band, vfr[k + 1])
        for i in range(nseg):
            j = (i + 1) % M
            q = [rings[k][i], rings[k][j], rings[k + 1][j], rings[k + 1][i]]
            uv = [(ucum[i], Vh), (ucum[i + 1], Vh), (ucum[i + 1], Vl), (ucum[i], Vl)]
            if flip:
                q = q[::-1]; uv = uv[::-1]
            mb.quad(*q, uv)
    return rings


def _sweep_solid(mb, path, normals, profile, *, closed=False, band="gold_plain", taper=0.0, w=None, tag="", u0=0.0,
                 across=None, seat=None):
    path = np.asarray(path, float); normals = nrm(normals)
    M = len(path)
    if closed:
        T = nrm(np.roll(path, -1, 0) - np.roll(path, 1, 0))
    else:
        T = np.zeros_like(path); T[1:-1] = path[2:] - path[:-2]; T[0] = path[1] - path[0]; T[-1] = path[-1] - path[-2]
        T = nrm(T)
    Bl = nrm(np.cross(normals, T)) if across is None else nrm(across)
    Bl = nrm(Bl - np.sum(Bl * normals, -1, keepdims=True) * normals)
    seg = np.linalg.norm(np.diff(np.concatenate([path, path[:1]]) if closed else path, axis=0), axis=-1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    scale = np.ones(M)
    if taper > 0 and not closed:
        t = s[:M] / max(s[-1], 1e-6)
        scale = 0.15 + 0.85 * smoothstep(0, taper, t) * smoothstep(0, taper, 1 - t)
    prof_ = np.asarray(profile, float)
    K = len(prof_)
    lat = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(prof_, axis=0), axis=-1))])
    ptot = max(lat[-1], 1e-4)
    vfr = lat / ptot
    rings = []
    ring_pts = [path + (prof_[k, 0] * scale)[:, None] * Bl + (prof_[k, 1] * scale)[:, None] * normals for k in range(K)]
    if seat is not None:
        # the bottom edges lie ON the plate (seat = its BVH), lifted by the plate's bulge between them so the flat
        # bottom face never dips into a convex plate
        e0 = seat_points(seat, ring_pts[0], normals, lift=0.0)
        e1 = seat_points(seat, ring_pts[K - 1], normals, lift=0.0)
        mid = seat_points(seat, 0.5 * (e0 + e1), normals, lift=0.0)
        bulge = np.maximum(np.sum((mid - 0.5 * (e0 + e1)) * normals, -1), 0.0) + 0.0004
        ring_pts[0] = e0 + normals * bulge[:, None]
        ring_pts[K - 1] = e1 + normals * bulge[:, None]
        for k in range(1, K - 1):
            ring_pts[k] = ring_pts[k] + normals * bulge[:, None]
    for k in range(K):
        rings.append(mb.add(ring_pts[k], w=w, tag=tag))
    dens = TX.band_px(band) / ptot if band in TX.BANDS else TX.TRIM_DENS
    # U follows the local (tapered) profile width: isotropic texels to the ends
    sm = np.concatenate([[0.0], np.cumsum(seg[:M - 1 + (1 if closed else 0)] * 2.0 /
                                           np.maximum(scale[:len(seg)] + np.roll(scale, -1)[:len(seg)], 0.2))])
    ucum = sm * dens / TX.W + u0
    nseg = M if closed else M - 1
    # orientation: the top faces must face along the surface normal
    km = (K - 1) // 2
    a_, b_, d_ = (np.asarray(mb.V[rings[km][0]]), np.asarray(mb.V[rings[km][1]]), np.asarray(mb.V[rings[km + 1][0]]))
    flip = float(np.dot(np.cross(b_ - a_, d_ - a_), normals[0])) < 0
    VB = TX.band_v("steel_plain", 0.5)
    for k in range(K):
        k2 = (k + 1) % K
        bottom = k == K - 1
        Vh, Vl = (VB, VB) if bottom else (TX.band_v(band, vfr[k]), TX.band_v(band, vfr[k2]))
        for i in range(nseg):
            j = (i + 1) % M
            q = [rings[k][i], rings[k][j], rings[k2][j], rings[k2][i]]
            uv = [(ucum[i], Vh), (ucum[i + 1], Vh), (ucum[i + 1], Vl), (ucum[i], Vl)]
            if flip:
                q = q[::-1]; uv = uv[::-1]
            mb.quad(*q, uv)
    if not closed and K >= 4 and K % 2 == 0:
        # end caps: pair profile points k and K-1-k
        for end, i in ((0, 0), (1, M - 1)):
            pts = [np.asarray(mb.V[rings[k][i]]) for k in range(K)]
            for k in range(K // 2 - 1):
                q = [rings[k][i], rings[k + 1][i], rings[K - 2 - k][i], rings[K - 1 - k][i]]
                a_, b_, d_ = np.asarray(mb.V[q[0]]), np.asarray(mb.V[q[1]]), np.asarray(mb.V[q[3]])
                n_ = np.cross(b_ - a_, d_ - a_)
                out = -T[i] if end == 0 else T[i]
                uvq = [(ucum[i], VB)] * 4
                if float(np.dot(n_, out)) < 0:
                    q = q[::-1]
                mb.quad(*q, uvq)
    return rings


def unwrap_faces(mb, fsel, labels, around, metres_per_uv, iterations=60, u_offset=0.0, seam_edges=None,
                 islands_out=None, method='MINIMUM_STRETCH'):
    """Iteration 2b (user item 31, mail stretch): near-isometric UVs for the faces `fsel` of mb (in place in mb.UV).
    Faces are cut into islands along every edge between different `labels` (developable panels: torso front / back,
    collar, sleeve halves), each island is unwrapped with Blender's Minimum Stretch (SLIM: angle AND area), then rotated so
    `around(face_centre, face_normal)` (the direction the ring rows run: round the body / round the arm) maps to +U and
    scaled to true size (1 UV = metres_per_uv, area-weighted mean scale 1). Returns per-face (s1, s2) principal stretches
    (surface metres per nominal metre) for the log."""
    import bmesh
    fsel = list(fsel)
    if not fsel:
        return {}
    vid = sorted({v for fi in fsel for v in mb.F[fi]})
    remap = {v: k for k, v in enumerate(vid)}
    V = np.array([mb.V[v] for v in vid], float)
    F = [tuple(remap[v] for v in mb.F[fi]) for fi in fsel]
    me = bpy.data.meshes.new("_uvtmp")
    me.from_pydata([tuple(p) for p in V], [], F)
    me.update()
    me.uv_layers.new(name="UVMap")
    b = bmesh.new(); b.from_mesh(me); b.edges.ensure_lookup_table(); b.faces.ensure_lookup_table()
    lab = [labels[k] for k in range(len(fsel))]
    for e in b.edges:
        fs = e.link_faces
        e.seam = len(fs) == 2 and lab[fs[0].index] != lab[fs[1].index]
        if seam_edges is not None and not e.seam:
            key = tuple(sorted((vid[e.verts[0].index], vid[e.verts[1].index])))
            e.seam = key in seam_edges
    b.to_mesh(me); b.free()
    ob = bpy.data.objects.new("_uvtmp", me)
    bpy.context.scene.collection.objects.link(ob)
    for o in bpy.context.view_layer.objects:
        o.select_set(False)
    bpy.context.view_layer.objects.active = ob; ob.select_set(True)
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    try:
        if method == 'MINIMUM_STRETCH':
            bpy.ops.uv.unwrap(method='MINIMUM_STRETCH', fill_holes=True, correct_aspect=False, margin=0.0,
                              iterations=iterations)
        else:
            bpy.ops.uv.unwrap(method=method, fill_holes=True, correct_aspect=False, margin=0.0)
    except TypeError:
        bpy.ops.uv.unwrap(method='ANGLE_BASED', fill_holes=True, correct_aspect=False, margin=0.0)
    bpy.ops.object.mode_set(mode='OBJECT')
    uv = np.empty(len(me.loops) * 2); me.uv_layers[0].data.foreach_get("uv", uv); uv = uv.reshape(-1, 2)
    loops = [list(p.loop_indices) for p in me.polygons]
    # islands: faces joined by non-seam edges (the labels)
    par = list(range(len(F)))

    def find(x):
        while par[x] != x:
            par[x] = par[par[x]]; x = par[x]
        return x
    ekey = {}
    for fi, f in enumerate(F):
        for k in range(4):
            e = tuple(sorted((f[k], f[(k + 1) % 4])))
            ge = tuple(sorted((vid[e[0]], vid[e[1]])))
            if e in ekey and lab[ekey[e]] == lab[fi] and (seam_edges is None or ge not in seam_edges):
                par[find(fi)] = find(ekey[e])
            ekey.setdefault(e, fi)
    isl = {}
    for fi in range(len(F)):
        isl.setdefault(find(fi), []).append(fi)
    stats = {}
    out_uv = [None] * len(F)
    for root, fl in isl.items():
        # per face: 3D <-> UV Jacobian from its first triangle
        A3 = 0.0; Auv = 0.0; ang_c = 0.0; ang_s = 0.0
        jac = []
        for fi in fl:
            f = F[fi]; L = loops[fi]
            p0, p1, p2, p3 = V[f[0]], V[f[1]], V[f[2]], V[f[3]]
            q0, q1, q2, q3 = uv[L[0]], uv[L[1]], uv[L[2]], uv[L[3]]
            a3 = 0.5 * (np.linalg.norm(np.cross(p1 - p0, p2 - p0)) + np.linalg.norm(np.cross(p2 - p0, p3 - p0)))
            auv = 0.5 * abs((q1 - q0)[0] * (q2 - q0)[1] - (q1 - q0)[1] * (q2 - q0)[0]) + \
                0.5 * abs((q2 - q0)[0] * (q3 - q0)[1] - (q2 - q0)[1] * (q3 - q0)[0])
            A3 += a3; Auv += auv
            # UV-space image of the 'around' direction: solve [e1 e2] in the triangle frame
            e1, e2 = p1 - p0, p2 - p0; d1, d2 = q1 - q0, q2 - q0
            n = nrm(np.cross(e1, e2))
            t = around(0.25 * (p0 + p1 + p2 + p3), n) if callable(around) else np.asarray(around[fsel[fi]], float)
            t = t - np.dot(t, n) * n
            if np.linalg.norm(t) < 1e-9:
                continue
            Mx = np.stack([e1, e2], 1)                       # 3x2
            coef, *_ = np.linalg.lstsq(Mx, t, rcond=None)     # t = a e1 + b e2
            duv = coef[0] * d1 + coef[1] * d2
            if np.linalg.norm(duv) < 1e-12:
                continue
            th = math.atan2(duv[1], duv[0])
            ang_c += a3 * math.cos(2 * th); ang_s += a3 * math.sin(2 * th)   # axial mean (rows have no sign)
        th0 = 0.5 * math.atan2(ang_s, ang_c)
        c, s = math.cos(-th0), math.sin(-th0)
        Rm = np.array([[c, -s], [s, c]])
        k = math.sqrt(A3 / max(Auv, 1e-18)) / metres_per_uv
        allL = [l for fi in fl for l in loops[fi]]
        cen = uv[allL].mean(0)
        for fi in fl:
            out_uv[fi] = [tuple((Rm @ (uv[l] - cen)) * k + np.array([u_offset, 0.0])) for l in loops[fi]]
        if islands_out is not None:
            islands_out.append([fsel[fi] for fi in fl])
    bpy.data.objects.remove(ob, do_unlink=True)
    bpy.data.meshes.remove(me)
    # write back into mb (per face corner, same corner order as mb.F: from_pydata keeps it)
    s_all = []
    for k, fi in enumerate(fsel):
        mb.UV[fi] = out_uv[k]
        f = F[k]; q = np.array(out_uv[k]) * metres_per_uv
        for (i0, i1, i2) in ((0, 1, 2), (0, 2, 3)):
            e1, e2 = V[f[i1]] - V[f[i0]], V[f[i2]] - V[f[i0]]
            d1, d2 = q[i1] - q[i0], q[i2] - q[i0]
            det = d1[0] * d2[1] - d1[1] * d2[0]
            if abs(det) < 1e-14:
                continue
            Pu = (e1 * d2[1] - e2 * d1[1]) / det; Pv = (e2 * d1[0] - e1 * d2[0]) / det
            a_ = Pu @ Pu; c_ = Pv @ Pv; b_ = Pu @ Pv
            tr = a_ + c_; dt = math.sqrt(max((a_ - c_) ** 2 + 4 * b_ * b_, 0))
            s_all.append((math.sqrt(max((tr + dt) / 2, 0)), math.sqrt(max((tr - dt) / 2, 0)),
                          0.5 * np.linalg.norm(np.cross(e1, e2)), lab[k]))
    return s_all


def seat_points(bvh, pts, dirs, lift=0.00005, reach=0.02):
    """project points onto a surface (BVH) along -dirs (then +dirs), `lift` above it: attachments sit ON the plate"""
    out = np.array(pts, float).copy()
    for k, (p, d) in enumerate(zip(out, dirs)):
        d = nrm(d)
        best = None
        for sg in (1.0, -1.0):
            o = p + d * reach * sg
            r = Vector(-d * sg)
            tot = 0.0
            for _ in range(6):                       # only OUTWARD-facing surfaces (a closed plate's inner side is skipped)
                loc, nor, fi, dist = bvh.ray_cast(Vector(o), r, 2 * reach - tot)
                if loc is None:
                    break
                tot += dist
                if float(np.dot(np.array(nor[:]), d)) > 0.0:
                    if best is None or abs(tot - reach) < abs(best[1] - reach):
                        best = (np.array(loc[:]), tot)
                    break
                o = np.array(loc[:]) + np.array(r[:]) * 1e-6
        if best is not None:
            out[k] = best[0] + d * lift
    return out


def mb_bvh(mb, pred):
    """BVH of the faces of mb whose vertices all satisfy pred(tag)"""
    T = []
    for f in mb.F:
        if all(pred(mb.tag[i]) for i in f):
            T += [(f[0], f[1], f[2]), (f[0], f[2], f[3])]
    return BVHTree.FromPolygons([Vector(v) for v in mb.V], T) if T else None


# ------------------------------------------------------------------------------------------------------ grid fill
def grid_fill(mb, ring_idx, project, uvfn):
    """Close a ring of 4k mb-vertices with a k x k quad patch (Coons interpolation, then `project(p) -> p'`)."""
    L = len(ring_idx)
    assert L % 4 == 0, L
    k = L // 4
    R = np.array([mb.V[i] for i in ring_idx])
    G = -np.ones((k + 1, k + 1), dtype=np.int64)
    # boundary: bottom (0..k), right (k..2k), top (2k..3k reversed), left (3k..4k reversed)
    for a in range(k + 1):
        G[a, 0] = ring_idx[a]
        G[k, a] = ring_idx[(k + a) % L]
        G[k - a, k] = ring_idx[(2 * k + a) % L]
        G[0, k - a] = ring_idx[(3 * k + a) % L]
    B = lambda a, b: np.array(mb.V[G[a, b]])
    w = mb.W[ring_idx[0]]
    for a in range(1, k):
        for b in range(1, k):
            u, v = a / k, b / k
            p = ((1 - v) * B(a, 0) + v * B(a, k) + (1 - u) * B(0, b) + u * B(k, b)
                 - ((1 - u) * (1 - v) * B(0, 0) + u * (1 - v) * B(k, 0) + (1 - u) * v * B(0, k) + u * v * B(k, k)))
            G[a, b] = mb.add(project(p)[None], w=[w])[0]
    # winding: the patch must walk the ring edge ring[0]->ring[1] opposite to the face that already uses it
    he = set()
    for f in mb.F:
        for t in range(4):
            he.add((f[t], f[(t + 1) % 4]))
    flip = (int(ring_idx[0]), int(ring_idx[1])) in he
    for a in range(k):
        for b in range(k):
            q = [G[a, b], G[a + 1, b], G[a + 1, b + 1], G[a, b + 1]]
            uv = [uvfn(np.array(mb.V[i])) for i in q]
            if flip:
                q = q[::-1]; uv = uv[::-1]
            mb.quad(*q, uv)
    return G


# ------------------------------------------------------------------------------------------------------ lathe
def lathe(mb, center, axis, profile, *, seg=12, band="gold_plain", ref=None, w=None, tag="", cap_uv=None, planar=None,
          seat=None, base=None):
    """Revolve profile [(r, h), ...] (from the outer radius inward; the last point is the innermost ring) around
    `axis` at `center`. The innermost ring is closed with a quad grid fill. seg must be a multiple of 4.
    seat=bvh (iteration 2b): the first (outer) ring is projected onto that surface (+0.05 mm) so the boss / rivet
    sits ON the plate: no floating base, no buried base ring."""
    axis = nrm(axis)
    ref = nrm(np.cross(axis, (0, 0, 1)) if ref is None else ref)
    if np.linalg.norm(np.cross(axis, ref)) < 1e-3:
        ref = nrm(np.cross(axis, (1, 0, 0)))
    e1 = nrm(ref - np.dot(ref, axis) * axis); e2 = np.cross(axis, e1)
    ang = np.linspace(0, 2 * np.pi, seg, endpoint=False)
    dirs = np.cos(ang)[:, None] * e1 + np.sin(ang)[:, None] * e2
    prof_ = np.asarray(profile, float)
    rings = []
    for k, (r, h) in enumerate(prof_):
        pts = np.asarray(center) + r * dirs + h * axis
        if seat is not None and k == 0:
            pts = seat_points(seat, pts, np.tile(axis, (len(pts), 1)))
        rings.append(mb.add(pts, w=w, tag=tag))
    lat = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(prof_, axis=0), axis=-1))])
    vfr = lat / max(lat[-1], 1e-9)
    circ = 2 * np.pi * prof_[:, 0].max()
    # iteration 2b (G6): a decal square on a dome is mapped CONFORMALLY (rho' = rho / r along the profile, from the
    # innermost ring outward), not as a planar projection (the steep sides of a rivet / boss were squashed 2-6x)
    rho = np.zeros(len(prof_)); rho[-1] = max(prof_[-1, 0], 1e-5)
    for k in range(len(prof_) - 2, -1, -1):
        ds = float(np.linalg.norm(prof_[k] - prof_[k + 1])); rm = max(0.5 * (prof_[k, 0] + prof_[k + 1, 0]), 1e-5)
        rho[k] = rho[k + 1] * math.exp(ds / rm)
    rho *= prof_[:, 0].max() / max(rho.max(), 1e-9)
    for k in range(len(rings) - 1):
        Vh, Vl = TX.band_v(band, vfr[k]), TX.band_v(band, vfr[k + 1])
        for i in range(seg):
            j = (i + 1) % seg
            q = [rings[k][i], rings[k][j], rings[k + 1][j], rings[k + 1][i]]
            if planar:
                sz = 2.2 * prof_[:, 0].max()
                rk = [rho[k], rho[k], rho[k + 1], rho[k + 1]]; ak = [ang[i], ang[j] if j else 2 * np.pi, ang[j] if j else 2 * np.pi, ang[i]]
                uv = [TX.square_uv(planar, float(r_ * math.cos(a_)), float(r_ * math.sin(a_)), sz) for r_, a_ in zip(rk, ak)]
            else:
                # iteration 2b (G6): U density = the band's V density over the profile length, round each row's
                # own circumference (was the outer ring's for every row: inner rings 2-9x anisotropic)
                dvn = TX.band_px(band) / max(lat[-1], 1e-4) if band in TX.BANDS else TX.TRIM_DENS
                rm = 0.5 * (prof_[k, 0] + prof_[k + 1, 0])
                ua = i / seg * 2 * np.pi * rm * dvn / TX.W; ub = (i + 1) / seg * 2 * np.pi * rm * dvn / TX.W
                uv = [(ua, Vh), (ub, Vh), (ub, Vl), (ua, Vl)]
            mb.quad(*q, uv)
    top = prof_[-1]
    cpt = np.asarray(center) + top[1] * axis
    size = 2.2 * prof_[:, 0].max() if planar else 2.6 * prof_[:, 0].max()
    kc = rho[-1] / max(prof_[-1, 0], 1e-9) if planar else 1.0         # the cap continues the conformal map
    uvf = cap_uv or (lambda p: TX.square_uv(planar or "boss", kc * float((p - cpt) @ e1), kc * float((p - cpt) @ e2), size))
    grid_fill(mb, list(rings[-1]), lambda p: p, uvf)
    if base is None:
        base = seat is not None
    if base:
        # a flat bottom on the plate: the boss / rivet is a closed solid (G1: no one-sided dome); faces away from axis
        proj = (lambda p: seat_points(seat, [p], [axis], lift=0.00005)[0]) if seat is not None else (lambda p: p)
        grid_fill(mb, list(rings[0]), proj, uvf)
    return rings


# ------------------------------------------------------------------------------------------------------ misc
def frame_along(d, up=(0, 0, 1)):
    d = nrm(d)
    e1 = np.cross(d, up)
    if np.linalg.norm(e1) < 1e-4:
        e1 = np.cross(d, (1, 0, 0))
    e1 = nrm(e1); e2 = np.cross(d, e1)
    return e1, e2


def disc_grid(n):
    """(n, n) points of a square grid mapped onto the unit disc (elliptical grid mapping, quads stay quads)."""
    a = np.linspace(-1, 1, n)
    A, B = np.meshgrid(a, a, indexing="ij")
    X = A * np.sqrt(1 - B ** 2 / 2)
    Y = B * np.sqrt(1 - A ** 2 / 2)
    return X, Y


def dome_dirs(D, ref, X, Y, rho_fn):
    """Directions of a spherical patch around pole D: disc coords (X, Y) -> polar (rho, psi), rho = |XY| * rho_fn(psi).
    psi = 0 along `ref` (projected), psi = 90 deg along D x ref."""
    D = nrm(D)
    e1 = nrm(np.asarray(ref, float) - np.dot(ref, D) * D)
    e2 = np.cross(D, e1)
    r = np.sqrt(X ** 2 + Y ** 2)
    psi = np.arctan2(Y, X)
    rho = r * rho_fn(psi)
    dirs = (np.cos(rho)[..., None] * D + np.sin(rho)[..., None] * (np.cos(psi)[..., None] * e1 + np.sin(psi)[..., None] * e2))
    return dirs, rho, psi, (e1, e2)


def body_dist(body, O, dirs, maxd=0.4, default=0.0):
    out = np.zeros(dirs.shape[:-1])
    for idx in np.ndindex(out.shape):
        d = body.ray(O, dirs[idx], maxd)
        out[idx] = d if d is not None else default
    return out



def body_weights_at(body, pts, bones=None):
    """Skin weights for points from the nearest body triangle (barycentric blend of its 3 vertices' weights)."""
    names = list(body.bw.keys()) if bones is None else bones
    W = np.stack([body.bw[n] for n in names], 1)          # (nbody, nb)
    out = []
    for p in pts:
        loc, nor, fi, d = body.bvh.find_nearest(Vector(p))
        tri = body.tris[fi]
        a, b, c = body.co[tri]
        v0, v1, v2 = b - a, c - a, np.array(loc[:]) - a
        d00, d01, d11 = v0 @ v0, v0 @ v1, v1 @ v1
        d20, d21 = v2 @ v0, v2 @ v1
        den = max(d00 * d11 - d01 * d01, 1e-12)
        v = (d11 * d20 - d01 * d21) / den; w = (d00 * d21 - d01 * d20) / den; u = 1 - v - w
        bc = np.clip(np.array([u, v, w]), 0, 1); bc /= bc.sum()
        ww = bc @ W[tri]
        out.append({names[k]: float(ww[k]) for k in np.nonzero(ww > 1e-3)[0]})
    return out


def region_mesh(verts, faces, subdiv=0):
    """(V, F) quad region -> optionally Catmull-Clark subdivided once or twice (Blender Subsurf), returns (V, F)."""
    me = bpy.data.meshes.new("__reg")
    me.from_pydata([tuple(v) for v in verts], [], [tuple(f) for f in faces])
    ob = bpy.data.objects.new("__reg", me)
    bpy.context.scene.collection.objects.link(ob)
    if subdiv:
        md = ob.modifiers.new("s", 'SUBSURF'); md.levels = subdiv; md.render_levels = subdiv
        md.boundary_smooth = 'PRESERVE_CORNERS'
    dg = bpy.context.evaluated_depsgraph_get()
    em = ob.evaluated_get(dg).to_mesh()
    V = np.array([v.co[:] for v in em.vertices]); F = [tuple(p.vertices) for p in em.polygons]
    ob.evaluated_get(dg).to_mesh_clear()
    bpy.data.objects.remove(ob, do_unlink=True); bpy.data.meshes.remove(me)
    return V, F


def mesh_adjacency(nv, faces):
    nb = [set() for _ in range(nv)]
    for f in faces:
        for k in range(len(f)):
            a, b = f[k], f[(k + 1) % len(f)]
            nb[a].add(b); nb[b].add(a)
    return [list(x) for x in nb]


def shrinkwrap_offset(body, V, F, d, iters=3, lam=0.5, fix_boundary=False):
    """Project points onto the body surface offset by d (smoothed normals), with Laplacian relaxation in between."""
    V = np.array(V, float)
    nb = mesh_adjacency(len(V), F)
    for it in range(iters + 1):
        out = np.empty_like(V)
        for i, p in enumerate(V):
            loc, nor, fi, dist = body.bvh.find_nearest(Vector(p))
            tri = body.tris[fi]
            n = body.vn[tri].mean(0); n /= np.linalg.norm(n)
            out[i] = np.array(loc[:]) + d * n
        V = out
        if it < iters:
            S = np.array([V[x].mean(0) if x else V[i] for i, x in enumerate(nb)])
            V = V + lam * (S - V)
    return V


# ------------------------------------------------------------------------------------------------------ decals
def decal_ribbon(mb, info, width, band, inset=0.0015, lift=0.0005, min_amp=0.5, w=None, tag="decal", seg_min=6):
    """Etched-band decal: a flat ribbon lying `lift` above the plate just inside a swept rim loop (`info` from
    mb.loopinfo), `width` wide, mapped across trim band `band` (V 0 at the border, 1 inside). Only where the loop's amp
    >= min_amp (the visible edges). One ribbon per contiguous run."""
    P, N, b, kap, A = info["P"], info["N"], info["b"], info["kap"], info["amp"]
    L = len(P)
    on = A >= min_amp
    if not on.any():
        return 0
    if on.all():
        runs = [list(range(L)) + [0]]
    else:
        start = int(np.argmin(on))           # an 'off' vertex: runs start after it
        runs, cur = [], []
        for k in range(1, L + 1):
            i = (start + k) % L
            if on[i]:
                cur.append(i)
            elif cur:
                runs.append(cur); cur = []
        if cur:
            runs.append(cur)
    n = 0
    for r in runs:
        if len(r) < seg_min:
            continue
        idx = np.array(r)
        Pr, Nr, br, kr = P[idx], N[idx], b[idx], kap[idx]
        rings = []
        for k, dd in enumerate((inset, inset + width)):
            q = Pr - br * dd + (lift + 0.5 * kr * dd * dd)[:, None] * Nr
            ws = [info["W"][i] for i in idx] if w is None else None
            rings.append(mb.add(q, w=(lambda pts, ws=ws: ws) if w is None else w, tag=tag))
        seg = np.linalg.norm(np.diff(Pr, axis=0), axis=-1)
        arc = np.concatenate([[0.0], np.cumsum(seg)])
        dens = TX.band_px(band) / max(width, 1e-4)
        ucum = arc * dens / TX.W
        Vh, Vl = TX.band_v(band, 0.0), TX.band_v(band, 1.0)
        # face the ribbon along the plate normal (the loop direction alone does not fix it on every border)
        fs = 0.0
        for i in range(len(idx) - 1):
            a_, b_, d_ = (np.asarray(mb.V[rings[0][i]]), np.asarray(mb.V[rings[0][i + 1]]), np.asarray(mb.V[rings[1][i]]))
            fs += float(np.dot(np.cross(b_ - a_, d_ - a_), Nr[i]))
        for i in range(len(idx) - 1):
            q = [rings[0][i], rings[0][i + 1], rings[1][i + 1], rings[1][i]]
            uv = [(ucum[i], Vl), (ucum[i + 1], Vl), (ucum[i + 1], Vh), (ucum[i], Vh)]
            if fs < 0:
                q = q[::-1]; uv = uv[::-1]
            mb.quad(*q, uv)
            n += 1
    return n


def rivet(mb, c, n, r=0.0042, h=0.0022, w=None, tag="rivet", seg=8, seat=None):
    """Domed rivet head on the surface point c with normal n (gold boss square). seat=bvh of the plate: the base ring
    lies on it (iteration 2b: base rings 1 mm under the plate read as half-buried slivers, others floated 1 mm)."""
    if seat is not None:
        c = seat_points(seat, [c], [n], lift=0.0)[0]
        return lathe(mb, np.asarray(c), n, [(r, 0.00005), (0.55 * r, h)], seg=seg, band="gold_plain", w=w, tag=tag,
                     planar="rivet_head", seat=seat)
    return lathe(mb, np.asarray(c) - 0.0006 * nrm(n), n, [(r, -0.0004), (0.55 * r, h)], seg=seg,
                 band="gold_plain", w=w, tag=tag, planar="rivet_head")      # 2 rings + cap: 24 tris (budget)
