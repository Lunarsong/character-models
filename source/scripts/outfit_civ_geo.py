"""Garment construction for the civilian outfits (outfit_peasant.py, outfit_archer.py): tights-derived body garments,
cut + snapped openings, hanging skirts with slits, belts, pouches, lathed hats, tubes. Blender Python + numpy.
Conventions: see outfit_civ_lib.py (character faces -Y, +X = its left, +Z up, metres; metric UVs; slot names).
"""
import math, os
import numpy as np
from outfit_civ_lib import *


# ================================================================================================= body measures
class Measures:
    """joint-relative heights of the authoring body (every garment dimension is written against these)"""

    def __init__(self, cb):
        H = cb.H
        self.H = H
        self.z_pelvis = H["pelvis"][2]
        self.z_hip = H["thigh_l"][2]
        self.z_knee = H["calf_l"][2]
        self.z_ankle = H["foot_l"][2]
        self.z_neck = H["neck_01"][2]
        self.z_sh = H["upperarm_l"][2]
        self.z_belt = self.z_pelvis + 0.035
        self.z_crotch = self.z_hip - 0.105
        co = cb.co
        self.z_top = co[:, 2].max()
        self.leg_len = self.z_hip - self.z_ankle

    def arm_t(self, p, s):
        """0 at the shoulder joint, 1 at the wrist, along the straight shoulder -> wrist line"""
        sh, wr = self.H["upperarm_" + s], self.H["hand_" + s]
        A = wr - sh
        return np.dot(np.asarray(p) - sh, A) / np.dot(A, A)

    def elbow_t(self, s):
        return self.arm_t(self.H["lowerarm_" + s], s)

    def leg_t(self, p, s):
        hp, an = self.H["thigh_" + s], self.H["foot_" + s]
        A = an - hp
        return np.dot(np.asarray(p) - hp, A) / np.dot(A, A)


def _limb_ramp(w, upper, lower, r):
    """re-split the upper / lower limb share of a weights dict by the ramp r (0 = all upper, 1 = all lower), keeping
    each side's twist-bone ratio"""
    up = {b: x for b, x in w.items() if b.startswith(upper)}
    lo = {b: x for b, x in w.items() if b.startswith(lower)}
    tot = sum(up.values()) + sum(lo.values())
    if tot < 1e-6:
        return w
    out = {b: x for b, x in w.items() if b not in up and b not in lo}
    su, sl = sum(up.values()), sum(lo.values())
    ups = {b: x / su for b, x in up.items()} if su > 1e-6 else {upper[0]: 1.0}
    los = {b: x / sl for b, x in lo.items()} if sl > 1e-6 else {lower[0]: 1.0}
    for b, x in ups.items():
        out[b] = out.get(b, 0.0) + tot * (1 - r) * x
    for b, x in los.items():
        out[b] = out.get(b, 0.0) + tot * r * x
    return out


def garment_weights(w, p, M):
    """weights of a garment vertex from its tights vertex, applied identically to every tights garment (same line ->
    same weights, so the layers still share their skinning):
      * the thigh share fades out above the hip joint (the waist / belt line follows the pelvis and spine)
      * elbow and knee: the upper / lower limb split is a wide smooth ramp across the joint (+-11 / +-13 cm) instead of
        the skin's tight band: thick layered sleeves and trouser legs bend with a larger radius, so the offset
        surfaces do not fold through each other in the joint crease"""
    w = dict(w)
    arm = sum(x for b, x in w.items() if b.startswith(("upperarm", "lowerarm")))
    if arm > 0.3:
        sd = "l" if p[0] > 0 else "r"
        L = np.linalg.norm(M.H["hand_" + sd] - M.H["upperarm_" + sd])
        t, te = M.arm_t(p, sd), M.elbow_t(sd)
        r = float(smoothstep(te - 0.11 / L, te + 0.11 / L, t))
        if 0.0 < r < 1.0 or abs(t - te) < 0.16 / L:
            w = _limb_ramp(w, ("upperarm_" + sd, "upperarm_twist_01_" + sd), ("lowerarm_" + sd, "lowerarm_twist_01_" + sd), r)
    leg = sum(x for b, x in w.items() if b.startswith(("thigh", "calf")))
    if leg > 0.3:
        sd = "l" if p[0] > 0 else "r"
        L = np.linalg.norm(M.H["foot_" + sd] - M.H["thigh_" + sd])
        t, tk = M.leg_t(p, sd), M.leg_t(M.H["calf_" + sd], sd)
        r = float(smoothstep(tk - 0.13 / L, tk + 0.13 / L, t))
        if abs(t - tk) < 0.18 / L:
            w = _limb_ramp(w, ("thigh_" + sd, "thigh_twist_01_" + sd), ("calf_" + sd, "calf_twist_01_" + sd), r)
    w = norm_w(w)
    s = smoothstep(M.z_hip - 0.035, M.z_hip + 0.035, p[2])
    if s <= 0:
        return w
    out = {}
    moved = 0.0
    for b, x in w.items():
        if b.startswith("thigh"):
            out[b] = x * (1 - s); moved += x * s
        else:
            out[b] = out.get(b, 0.0) + x
    if moved:
        out["pelvis"] = out.get("pelvis", 0.0) + moved
    return norm_w(out)


def side_of(p):
    return "l" if p[0] >= 0 else "r"


# ================================================================================================= tights garments
class Garment:
    """a body-following garment grown from the tights helper: basemesh index per vertex (origin), points, faces,
    weights (tights weights, face bones -> head), body region per vertex"""

    def __init__(self, A, keep):
        cb = A.cb
        self.A = A
        self.idx, self.P, self.F = cb.tights_region(keep)          # idx = line keys (see CivBody.tights_region)
        self.W = [garment_weights(w, p, A.M) for w, p in zip(tights_weights(cb, self.idx), self.P)] \
            if getattr(A, "M", None) else tights_weights(cb, self.idx)
        self.region = [region_of(w) for w in self.W]
        N = vertex_normals(self.P, self.F)
        # the helper's faces point outward (checked: normals vs the offset from the nearest skin point)
        dots = []
        for p, n in zip(self.P[::7], N[::7]):
            loc, nn_, d = cb.nearest(p)
            dots.append(np.dot(p - loc, n))
        if np.mean(dots) < 0:
            self.F = [tuple(f[::-1]) for f in self.F]

    def loops(self):
        return boundary_loops(self.F)

    def snap_loop_plane(self, loop, O, Nrm, iters=2, rings=2):
        """move a boundary loop onto the plane (O, Nrm) along the plane normal; the next rings are kept on the garment
        side of the plane (the side most of the garment lies on) by at least 45 % / 90 % of a ring spacing, so the
        faces next to the opening never fold back over it"""
        P = self.P
        nb = adjacency(len(P), self.F)
        ls = set(loop)
        side = np.sign(np.median((P - O) @ Nrm)) or -1.0
        for v in loop:
            P[v] = P[v] - Nrm * np.dot(P[v] - O, Nrm)
        sp = np.mean([np.linalg.norm(P[loop[k]] - P[loop[k - 1]]) for k in range(len(loop))])
        done = set(ls)
        front = set(loop)
        for r in range(1, rings + 1):
            nxt = set(j for v in front for j in nb[v]) - done
            for v in nxt:
                d = np.dot(P[v] - O, Nrm) * side
                want = 0.45 * r * sp
                if d < want:
                    P[v] = P[v] + Nrm * side * (want - d)
            done |= nxt
            front = nxt

    def snap_loop_zcurve(self, loop, zfun, rings=2):
        """neck-style opening: loop vertices move vertically onto z = zfun(p); the next rings are kept below the curve
        (at least 45 % / 90 % of a ring spacing), so no face folds over the opening"""
        P = self.P
        nb = adjacency(len(P), self.F)
        ls = set(loop)
        for v in loop:
            P[v][2] = zfun(P[v])
        sp = np.mean([np.linalg.norm(P[loop[k]] - P[loop[k - 1]]) for k in range(len(loop))])
        done = set(ls)
        front = set(loop)
        for r in range(1, rings + 1):
            nxt = set(j for v in front for j in nb[v]) - done
            for v in nxt:
                lim = zfun(P[v]) - 0.45 * r * sp
                if P[v][2] > lim:
                    P[v][2] = lim
            done |= nxt
            front = nxt

    def to_mesh(self, slot, uvfn=None, axes=None, g="body"):
        m = CMesh()
        for p, w in zip(self.P, self.W):
            m.add_v(p, dict(w), g)
        for f in self.F:
            regs = [self.region[i] for i in f]
            r = max(set(regs), key=regs.count)
            if r.startswith("hand"):
                r = "arm_" + r[-1]
            uv = uvfn(self.P, f, r) if uvfn else axes.uv(self.P, f, r)
            m.add_f(f, uv, slot)
        return m


def classify_loops(G, M):
    """name the boundary loops of a tights garment by geometry: 'neck', 'waist' (around the trunk), 'cuff_l/r'
    (arm openings), 'ankle_l/r' / 'thigh_l/r' (leg openings)"""
    out = {}
    trunk = []
    for lp in G.loops():
        Q = G.P[lp]
        c = Q.mean(0)
        regs = [G.region[i] for i in lp]
        r = max(set(regs), key=regs.count)
        if r.startswith(("arm", "hand")) or np.abs(Q[:, 0]).max() > 0.32:
            out["cuff_" + ("l" if c[0] > 0 else "r")] = lp
        elif c[2] > M.z_sh - 0.06 and np.abs(Q[:, 0]).max() < 0.2:
            out["neck"] = lp
        elif Q[:, 0].min() < -0.04 and Q[:, 0].max() > 0.04:
            trunk.append((c[2], lp))
        else:
            s = "l" if c[0] > 0 else "r"
            out[("ankle_" if c[2] < M.z_knee else "thigh_") + s] = lp
    trunk.sort(key=lambda t: -t[0])
    for k, (z, lp) in enumerate(trunk):
        out["waist" if k == 0 else "waist%d" % (k + 1)] = lp
    return out


def arm_plane(M, s, t):
    sh, wr = M.H["upperarm_" + s], M.H["hand_" + s]
    return sh + (wr - sh) * t, nrm(wr - sh)


def leg_plane(M, s, t):
    hp, an = M.H["thigh_" + s], M.H["foot_" + s]
    return hp + (an - hp) * t, nrm(an - hp)


# ================================================================================================= folds
def fold_noise(P, seed=1, scale=6.0):
    """smooth pseudo-random scalar field (sum of oriented sines) for irregular fold phases"""
    rng = np.random.default_rng(seed)
    out = np.zeros(len(P))
    for k in range(6):
        d = nrm(rng.normal(size=3)); f = scale * (0.6 + rng.uniform() * 0.8); ph = rng.uniform(0, 2 * math.pi)
        out += np.sin(P @ d * f * 2 * math.pi + ph) / 6
    return out


# ================================================================================================= hanging skirts
class HangField:
    """non-shrinking outer hull (cloth hangs from its widest point) of a composite ray target around the vertical axis"""

    def __init__(self, comp, z0, z1, nz=56, nphi=96, centre=(0.0, 0.012), sig_z=0.012, sig_phi=8.0, shrink=0.998):
        self.z0, self.z1, self.centre = z0, z1, centre
        zs = np.linspace(z0, z1, nz); phis = np.linspace(0, 2 * math.pi, nphi, endpoint=False)
        R0 = 0.75
        R = np.full((nz, nphi), np.nan)
        cx, cy = centre
        for i, z in enumerate(zs):
            for k, ph in enumerate(phis):
                d = np.array([math.sin(ph), -math.cos(ph), 0.0])
                h = comp.ray(np.array([cx, cy, z]) + d * R0, -d, R0)
                if h is not None:
                    R[i, k] = R0 - h
        idx = np.arange(nphi)
        for i in range(nz):
            ok = ~np.isnan(R[i])
            if ok.any() and not ok.all():
                R[i, ~ok] = np.interp(idx[~ok], idx[ok], R[i, ok], period=nphi)
            elif not ok.any():
                R[i] = 0.12
        dz = (z1 - z0) / (nz - 1)
        st, sp = sig_z / dz, sig_phi / (360.0 / nphi)
        R = np.maximum(R, gauss1d(gauss1d(R, st, 0), sp, 1, wrap=True))
        S = R.copy()
        for i in range(nz - 2, -1, -1):                      # zs ascending: walk down from the top
            S[i] = np.maximum(S[i], S[i + 1] * shrink)
        self.R = gauss1d(gauss1d(R, st, 0), sp, 1, wrap=True)
        self.S = gauss1d(gauss1d(S, 1.2, 0), 1.2, 1, wrap=True)
        self.nz, self.nphi = nz, nphi

    def _look(self, F, z, ph):
        fa = np.clip((np.asarray(z, float) - self.z0) / (self.z1 - self.z0) * (self.nz - 1), 0, self.nz - 1.0001)
        fb = (np.asarray(ph, float) % (2 * math.pi)) / (2 * math.pi) * self.nphi
        i0 = np.floor(fa).astype(int); ta = fa - i0
        j0 = np.floor(fb).astype(int) % self.nphi; tb = fb - np.floor(fb); j1 = (j0 + 1) % self.nphi
        return (F[i0, j0] * (1 - ta) * (1 - tb) + F[i0 + 1, j0] * ta * (1 - tb) + F[i0, j1] * (1 - ta) * tb
                + F[i0 + 1, j1] * ta * tb)

    def hang(self, z, ph):
        return self._look(self.S, z, ph)

    def hull(self, z, ph):
        return self._look(self.R, z, ph)

    def point(self, z, ph, r):
        cx, cy = self.centre
        return np.array([cx + r * math.sin(ph), cy - r * math.cos(ph), z])


def phi_of(p, centre=(0.0, 0.012)):
    return math.atan2(p[0] - centre[0], -(p[1] - centre[1])) % (2 * math.pi)


def hang_skirt(m, loop, hf, z_hem, *, clear=0.02, flare=0.03, n_folds=12, fold_amp=0.010, fold_seed=3,
               row_step=0.035, slit_top=None, back_drop=0.02, slot="cloth", r_uv=0.16, law=None, blend_rows=3,
               min_clear_fn=None, gap_deg=1.5):
    """Hang a skirt from boundary `loop` (vertex indices of m, the garment's lower opening) down to z_hem.
    Columns = loop vertices (their angle phi around the body axis); the columns nearest phi = 0 (front) and pi (back)
    are split below slit_top (front / back slits, each half follows its thigh). law(z, side) -> weights dict.
    Returns dict(rows=index grid, hem=bottom loop (ordered for add_hem), halves)."""
    P0 = m.arr()
    cen = hf.centre
    # keep the loop's own vertex order (its edges are the faces' edges: re-sorting by angle broke the connection where
    # the loop zig-zags and left 1-2 vertex holes at the waist); orient it toward increasing phi, start at the front
    lp = list(loop)
    ph = np.unwrap(np.array([phi_of(P0[v], cen) for v in lp]))
    if ph[-1] - ph[0] < 0:
        lp = lp[::-1]
        ph = np.unwrap(np.array([phi_of(P0[v], cen) for v in lp]))
    n = len(lp)
    phw = ph % (2 * math.pi)
    i_front = int(np.argmin(np.minimum(phw, 2 * math.pi - phw)))
    lp = lp[i_front:] + lp[:i_front]
    ph = np.unwrap(np.array([phi_of(P0[v], cen) for v in lp]))
    ph = ph - 2 * math.pi * round(ph[0] / (2 * math.pi))                   # starts near 0, increases to ~2 pi
    ph = np.maximum.accumulate(ph)                                          # monotonic (tiny zig-zags flattened)
    i_back = int(np.argmin(np.abs(((ph - math.pi) + math.pi) % (2 * math.pi) - math.pi)))
    z_top = float(np.mean(P0[lp][:, 2]))
    nrow = max(3, int(round((z_top - z_hem) / row_step)))
    zs = np.linspace(z_top, z_hem, nrow + 1)
    r_top = np.array([np.hypot(P0[v][0] - cen[0], P0[v][1] - cen[1]) for v in lp])
    rng = np.random.default_rng(fold_seed)
    fph = rng.uniform(0, 2 * math.pi)
    fold_ph2 = rng.uniform(0, 2 * math.pi, 3)
    grid = [list(lp)]
    split = {}                                   # row -> duplicated vertex of the front / back columns
    prev_r = r_top.copy()
    s_top = slit_top if slit_top is not None else z_hem - 1
    for j in range(1, nrow + 1):
        t = j / nrow
        row = []
        rr = np.zeros(n)
        for i in range(n):
            phi = ph[i]
            zz = zs[j] - back_drop * (1 - math.cos(phi)) / 2 * t
            base = hf.hang(zz, phi) + clear
            a = t ** 1.3
            fl = flare * a * (1.0 + 0.25 * math.cos(2 * phi))
            fold = fold_amp * a * (math.sin(n_folds * phi + fph) + 0.35 * math.sin((n_folds * 1.7) * phi + fold_ph2[0])) \
                + fold_amp * 0.4 * a * math.sin(n_folds * 0.5 * phi + fold_ph2[1])
            r = max(base + fl + fold, prev_r[i] * 0.985)
            blend = smoothstep(0, blend_rows, j)
            r = lerp(max(r_top[i], hf.hull(zz, phi) + clear * 0.6), r, blend) if j <= blend_rows else r
            if min_clear_fn is not None:
                r = max(r, hf.hull(zz, phi) + min_clear_fn(zz))
            rr[i] = r
            p = hf.point(zz, phi, r)
            if j <= blend_rows:                       # ease out of the loop's own shape (keeps the join smooth)
                q = P0[lp[i]].copy(); q[2] = zz
                d0 = np.array([q[0] - cen[0], q[1] - cen[1]])
                q[:2] = np.array(cen) + d0 / max(np.linalg.norm(d0), 1e-9) * r
                p = lerp(q, p, blend)
            side = "l" if i < i_back else "r"          # column 0 starts the left half, i_back the right one
            w = law(p, zz, side) if law else None
            row.append(m.add_v(p, w, "body"))
        prev_r = rr
        grid.append(row)
        if zs[j] < s_top:
            # duplicate the front and back columns (slits): the copy belongs to the other half
            for i in (0, i_back):
                v = row[i]
                pp = np.array(m.V[v])
                gap = math.radians(gap_deg) * smoothstep(s_top, z_hem, zs[j])
                phi = ph[i]
                side_a = "r" if i == 0 else "l"          # the half that ends at this column coming from phi -> 2pi
                r_ = np.hypot(pp[0] - cen[0], pp[1] - cen[1])
                pa = hf.point(pp[2], phi - gap, r_); pb = hf.point(pp[2], phi + gap, r_)
                m.V[v] = tuple(pb)                        # original = start of the next half (increasing phi)
                w = law(pa, pp[2], side_a) if law else None
                split[(j, i)] = m.add_v(pa, w, "body")
    # blend the first rows' weights from the loop weights to the law
    if law:
        for j in range(1, min(blend_rows + 1, nrow + 1)):
            s = smoothstep(0, blend_rows, j)
            for i in range(n):
                v = grid[j][i]
                lw = m.W[lp[i]]
                m.W[v] = mix_w(lw, m.W[v], s)
                if (j, i) in split:
                    m.W[split[(j, i)]] = mix_w(lw, m.W[split[(j, i)]], s)
    # faces (wound consistently with the garment above: its faces hold the directed loop edges lp[i] -> lp[i+1] or
    # the reverse; the skirt faces must hold the opposite direction)
    upper_dir = set()
    for f in m.F:
        k = len(f)
        for q in range(k):
            upper_dir.add((f[q], f[(q + 1) % k]))
    fwd = sum(1 for i in range(n) if (lp[i], lp[(i + 1) % n]) in upper_dir)
    flip_skirt = fwd < n / 2
    f0 = len(m.F)

    def uv(i_, j_):
        return (ph[i_] * r_uv if i_ < n else (ph[0] + 2 * math.pi) * r_uv, zs[j_])
    for j in range(nrow):
        for i in range(n):
            i2 = (i + 1) % n
            a, b = grid[j][i], grid[j][i2]
            c, d = grid[j + 1][i2], grid[j + 1][i]
            # the face column [i, i2] ends a half at i2 == 0 or i2 == i_back: use the split copy there
            if (j + 1, i2) in split and i2 in (0, i_back):
                c = split[(j + 1, i2)]
            if j >= 1 and (j, i2) in split and i2 in (0, i_back):
                b = split[(j, i2)]
            ua, ub = uv(i, j), (uv(i2, j) if i2 != 0 else uv(n, j))
            uc, ud = ((uv(i2, j + 1) if i2 != 0 else uv(n, j + 1))), uv(i, j + 1)
            m.add_f([a, d, c, b], [ua, ud, uc, ub], slot)
    if flip_skirt:
        for fi in range(f0, len(m.F)):
            m.F[fi] = tuple(m.F[fi][::-1]); m.UV[fi] = list(m.UV[fi][::-1])
    return dict(grid=grid, split=split, i_back=i_back, n=n, zs=zs, ph=ph)


def polar_hull(R):
    """radii (uniform angles phi_k = 2 pi k / n, direction (sin, -cos)) -> radii of their 2D convex hull"""
    n = len(R)
    ph = np.arange(n) * 2 * math.pi / n
    pts = np.stack([R * np.sin(ph), -R * np.cos(ph)], 1)
    order = np.lexsort((pts[:, 1], pts[:, 0]))
    P = pts[order]

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lower, upper = [], []
    for p in P:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in P[::-1]:
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    hull = np.array(lower[:-1] + upper[:-1])
    out = R.copy()
    A = hull; B = np.roll(hull, -1, 0)
    for k in range(n):
        d = np.array([math.sin(ph[k]), -math.cos(ph[k])])
        best = R[k]
        # ray t*d against segment A + s (B - A): solve t d - s (B - A) = A
        e = B - A
        den = d[0] * (-e[:, 1]) - d[1] * (-e[:, 0])
        ok = np.abs(den) > 1e-12
        t = (A[:, 0] * (-e[:, 1]) - A[:, 1] * (-e[:, 0])) / np.where(ok, den, 1)
        sp = (d[0] * A[:, 1] - d[1] * A[:, 0]) / np.where(ok, den, 1)
        hit = ok & (t > 0) & (sp >= -1e-9) & (sp <= 1 + 1e-9)
        if hit.any():
            best = max(best, float(t[hit].max()))
        out[k] = best
    return out


def hang_torso(cb, idx, T, h, F, torso, z_top, z_join, gather=0.11, drop=0.08, side_drop=0.25, hull_above=0.12,
               centre=(0.0, 0.012), nphi=96, dz=0.008, max_add=0.04, fold_amp=0.0, fold_n=14, seed=5):
    """Cloth hangs from the chest / shoulder blades instead of following every hollow: on the torso lines (mask
    `torso`), below z_top the garment radius (around the body's vertical axis) may not shrink going down (running max
    of the radius over the heights above, per direction), then it is gathered back to its own radius over `gather`
    above the belt line z_join (a belted tunic blouses). Only the line heights h change (h += dr / N.radial), so the
    shared-lines layering still holds. Where the cloth is gathered, vertical folds proportional to the gathered
    width are added (fold_amp per metre of extra radius). The push is diffused a few rings into the sleeves so the
    armhole seam stays smooth. Returns the new h."""
    n = len(idx)
    N = cb.line_N(idx)
    P = T + N * h[:, None]
    cx, cy = centre
    ph = np.array([math.atan2(p[0] - cx, -(p[1] - cy)) % (2 * math.pi) for p in P])
    r = np.hypot(P[:, 0] - cx, P[:, 1] - cy)
    z = P[:, 2]
    z0, z1 = z_join - 0.03, z_top + max(0.02, hull_above)
    nz = int(math.ceil((z1 - z0) / dz)) + 1
    zs = np.linspace(z0, z1, nz)
    R = np.full((nz, nphi), np.nan)
    # the trunk only: shoulder / clavicle-weighted lines (deltoid caps classified 'torso') would widen the hull.
    # The outer radius of the garment's trunk faces is ray-sampled on a (z, phi) grid (vertex bins were too sparse).
    sh_w = np.array([sum(x for b, x in w.items() if b.startswith(("clavicle", "upperarm"))) for w in cb.line_W(idx)])
    okv = torso & (sh_w < 0.2)
    tf = [tuple(f) for f in F if all(okv[i] for i in f)]
    bvh = BVHTree.FromPolygons([v3(p) for p in P], tf)
    R0 = 0.75
    for a in range(nz):
        for b in range(nphi):
            pp = 2 * math.pi * b / nphi
            d = np.array([math.sin(pp), -math.cos(pp), 0.0])
            hit = bvh.ray_cast(v3(np.array([cx, cy, zs[a]]) + d * R0), v3(-d), R0)
            if hit[0] is not None:
                R[a, b] = R0 - hit[3]
    # fill holes along phi (periodic) then along z
    col = np.arange(nphi)
    for a in range(nz):
        ok = ~np.isnan(R[a])
        if ok.sum() >= 3:
            R[a, ~ok] = np.interp(col[~ok], col[ok], R[a, ok], period=nphi)
    row = np.arange(nz)
    for b in range(nphi):
        ok = ~np.isnan(R[:, b])
        if ok.sum() >= 2:
            R[~ok, b] = np.interp(row[~ok], row[ok], R[ok, b])
        else:
            R[:, b] = np.nanmean(R) if np.isfinite(np.nanmean(R)) else 0.15
    R = gauss1d(R, 1.5, 1, wrap=True)
    # cloth under tension spans the hollows of each horizontal cross-section: convex hull radius per direction
    Hh = np.array([polar_hull(R[a]) for a in range(nz)])
    S = Hh.copy()
    phis_ = np.arange(nphi) * 2 * math.pi / nphi
    dr_ = (drop + (side_drop - drop) * np.sin(phis_) ** 2) * dz      # metres inward per row: the side seams taper more
    for a in range(nz - 2, -1, -1):                    # zs ascending: walk down from z_top
        if zs[a] < z_top:
            S[a] = np.maximum(S[a], S[a + 1] - dr_)
    S = gauss1d(gauss1d(S, 1.0, 0), 2.0, 1, wrap=True)
    if os.environ.get("CIV_DEBUG"):
        for b in (0, nphi // 4, nphi // 2):
            clog("hang col phi %.2f: " % (b * 2 * math.pi / nphi) + " ".join(
                "z%.2f R%.3f H%.3f S%.3f" % (zs[a], R[a, b], Hh[a, b], S[a, b]) for a in range(0, nz, 4)))

    def look(F_, zz, pp):
        fa = np.clip((zz - z0) / dz, 0, nz - 1.001); fb = (pp % (2 * math.pi)) / (2 * math.pi) * nphi
        a0 = int(fa); ta = fa - a0; b0 = int(fb) % nphi; tb = fb - int(fb); b1 = (b0 + 1) % nphi
        return (F_[a0, b0] * (1 - ta) * (1 - tb) + F_[a0 + 1, b0] * ta * (1 - tb) + F_[a0, b1] * (1 - ta) * tb
                + F_[a0 + 1, b1] * ta * tb)
    dh = np.zeros(n)
    extra = np.zeros(n)
    for k in np.nonzero(torso)[0]:
        if not (z0 < z[k] < z1 - 0.012):
            continue
        g = smoothstep(z_join, z_join + gather, z[k])
        tgt = look(R, z[k], ph[k]) + (look(S, z[k], ph[k]) - look(R, z[k], ph[k])) * g
        dr = tgt - r[k]
        if dr > 0:
            rad = np.array([P[k][0] - cx, P[k][1] - cy, 0.0]); rad /= max(np.linalg.norm(rad), 1e-9)
            dh[k] = min(dr / max(0.35, float(np.dot(N[k], rad))), max_add)
        extra[k] = max(0.0, look(S, z[k], ph[k]) - look(R, z[k], ph[k])) * (1 - g) * smoothstep(z_join - 0.01, z_join + 0.03, z[k])
    nb = adjacency(n, F)
    for _ in range(6):                                   # diffuse into the sleeves / neck with decay
        new = dh.copy()
        for k in np.nonzero(~torso)[0]:
            if nb[k]:
                new[k] = max(dh[k], 0.55 * float(np.mean(dh[nb[k]])))
        dh = new
    for _ in range(3):
        dh = np.array([0.5 * dh[k] + 0.5 * float(np.mean(dh[nb[k]])) if nb[k] else dh[k] for k in range(n)])
    h = h + dh
    if fold_amp:
        rng = np.random.default_rng(seed)
        p0, p1 = rng.uniform(0, 2 * math.pi, 2)
        fold = np.array([0.5 + 0.5 * math.sin(fold_n * a + p0 + 0.8 * math.sin(3 * a + p1)) for a in ph])
        h = h + fold_amp * extra * fold
    return h


def add_band(m, loop, width, slot, lift=0.0009, rings=3, surface=None, flip_check=True, smooth=4, overhang=0.003):
    """a sewn-on band (tablet-woven trim, binding) lying on the garment along a boundary loop: `rings` rows of
    clean quads from the edge inward, each row projected onto the garment surface and lifted `lift` above it.
    The band follows the loop FAIRED along its length (`smooth` Laplacian passes: where a cut crosses the helper grid
    diagonally, e.g. a neckline over the shoulders, the garment edge is a staircase) and overhangs the edge by
    `overhang` like a binding, so the staircase stays hidden under it.
    u = arc length (metres; the trim slot scales it), v 0 = edge -> 1 = inner edge. Weights: nearest garment vertex.
    Returns the band's edge ring (vertex indices)."""
    P = m.arr()
    base_faces = [f for f, h in zip(m.F, m.HF) if not h]
    N = vertex_normals(P, base_faces)
    inw = loop_inward(m, loop)
    nl_ = len(loop)
    E = P[loop].copy(); I = inw.copy(); NN = N[loop].copy()
    for _ in range(smooth):
        E = 0.5 * E + 0.25 * (np.roll(E, 1, 0) + np.roll(E, -1, 0))
        I = 0.5 * I + 0.25 * (np.roll(I, 1, 0) + np.roll(I, -1, 0))
        NN = 0.5 * NN + 0.25 * (np.roll(NN, 1, 0) + np.roll(NN, -1, 0))
    I = nrm(I - NN * np.sum(I * NN, 1, keepdims=True)); NN = nrm(NN)
    bvh = BVHTree.FromPolygons([v3(p) for p in P], [tuple(f) for f in base_faces])
    kd = KDTree(len(P))
    for i, p in enumerate(P):
        kd.insert(p, i)
    kd.balance()
    nl = len(loop)
    al = [0.0]
    for k in range(1, nl + 1):
        al.append(al[-1] + np.linalg.norm(E[k % nl] - E[k - 1]))
    grid = []
    for q in range(rings):
        t = q / (rings - 1)
        row = []
        for k, v in enumerate(loop):
            if q:
                p = E[k] + I[k] * (width * t - overhang)
                loc, nn_, fi, d = bvh.find_nearest(v3(p))
                if loc is not None:
                    p = np.array(loc[:])
                    nv = np.array(nn_[:])
                else:
                    nv = NN[k]
            else:
                # the outer edge: beyond the (faired) garment edge, lifted a touch more (a binding over the edge)
                loc, nn_, fi, d = bvh.find_nearest(v3(E[k]))
                base = np.array(loc[:]) if loc is not None else E[k]
                p = base - I[k] * overhang
                nv = NN[k] * 1.4
            j = kd.find(p)[1]
            w = dict(m.W[j])
            row.append(m.add_v(p + nv * lift, w, m.G[v]))
        grid.append(row)
    new_faces = []
    for q in range(rings - 1):
        for k in range(nl):
            k2 = (k + 1) % nl
            vs = [grid[q][k], grid[q][k2], grid[q + 1][k2], grid[q + 1][k]]
            uvs = [(al[k], q / (rings - 1)), (al[k + 1], q / (rings - 1)), (al[k + 1], (q + 1) / (rings - 1)),
                   (al[k], (q + 1) / (rings - 1))]
            new_faces.append((vs, uvs, k))
    # orientation: the band's normal must agree with the garment normal under it (majority vote)
    Q = m.arr()
    s = 0.0
    for vs, _, k in new_faces:
        fn = np.cross(Q[vs[1]] - Q[vs[0]], Q[vs[3]] - Q[vs[0]])
        s += np.sign(np.dot(fn, N[loop[k]]))
    for vs, uvs, k in new_faces:
        if s < 0:
            vs, uvs = vs[::-1], uvs[::-1]
        m.add_f(vs, uvs, slot, hem=True)
    return grid[0]


def orient_outward(m, centre_fn):
    """flip faces whose normal points toward centre_fn(face centroid) (per connected piece: all or nothing)"""
    P = m.arr()
    s = 0.0
    for f in m.F:
        c = P[list(f)].mean(0)
        fn = np.cross(P[f[1]] - P[f[0]], P[f[2]] - P[f[0]])
        s += np.sign(np.dot(fn, c - centre_fn(c)))
    if s < 0:
        m.F = [tuple(f[::-1]) for f in m.F]; m.UV = [list(u[::-1]) for u in m.UV]
    return s


# ================================================================================================= belts / straps
def ring_on(comp, z, centre, nseg, off, tilt=0.0, half_h=0.0):
    """closed ring at height z (+ tilt * sin(phi)) around the composite's outer hull (largest radius over the band
    height z +- half_h), `off` outside it"""
    pts, rads = [], []
    for k in range(nseg):
        ph = 2 * math.pi * k / nseg
        zz = z + tilt * math.sin(ph)
        r = hull_radius_span(comp, zz - half_h, zz + half_h, ph, centre, nz=5 if half_h else 1)
        rads.append(r if r > 0 else np.nan)
    rads = np.array(rads)
    ok = ~np.isnan(rads)
    idx = np.arange(nseg)
    rads[~ok] = np.interp(idx[~ok], idx[ok], rads[ok], period=nseg)
    # convex-ish: a belt bridges hollows (max of raw and a smoothed copy), then smoothed
    sm = gauss1d(rads[None], 1.5, 1, wrap=True)[0]
    rads = gauss1d(np.maximum(rads, sm)[None], 1.0, 1, wrap=True)[0]
    for k in range(nseg):
        ph = 2 * math.pi * k / nseg
        r = rads[k] + off
        pts.append(np.array([centre[0] + r * math.sin(ph), centre[1] - r * math.cos(ph), z + tilt * math.sin(ph)]))
    return np.array(pts)


def band_along(m, path, up, height, thick, slot, w_fn, closed=True, uv_u0=0.0, prof=None):
    """a strap / belt band swept along `path` (N, 3) with `up` vectors (N, 3): rectangular profile with rounded
    edges, `thick` outward (away from the body = cross(tangent, up) oriented outward). Returns the ring index grid."""
    n = len(path)
    prof = prof or [(0.0, -0.5), (0.55, -0.5), (1.0, -0.4), (1.0, 0.4), (0.55, 0.5), (0.0, 0.5)]
    pv = np.linspace(0, 1, len(prof))
    T = np.array([nrm(path[(k + 1) % n] - path[k - 1]) if closed else nrm(path[min(k + 1, n - 1)] - path[max(k - 1, 0)])
                  for k in range(n)])
    rings = []
    for k in range(n):
        u = nrm(up[k] - T[k] * np.dot(up[k], T[k]))
        o = np.cross(T[k], u)
        rings.append([m.add_v(path[k] + o * a * thick + u * b * height, w_fn(path[k]), "body") for a, b in prof])
    al = [uv_u0]
    for k in range(1, n + (1 if closed else 0)):
        al.append(al[-1] + np.linalg.norm(path[k % n] - path[k - 1]))
    rng = range(n) if closed else range(n - 1)
    for k in rng:
        k2 = (k + 1) % n
        for q in range(len(prof) - 1):
            vs = [rings[k][q], rings[k2][q], rings[k2][q + 1], rings[k][q + 1]]
            uvs = [(al[k], pv[q]), (al[k + 1], pv[q]), (al[k + 1], pv[q + 1]), (al[k], pv[q + 1])]
            m.add_f(vs, uvs, slot)
    if not closed:                                  # end caps (quad strip across the profile)
        for k, flip in ((0, True), (n - 1, False)):
            r = rings[k]
            h = len(prof) // 2
            for q in range(h - 1):
                a, b, c, d = r[q], r[q + 1], r[len(prof) - 2 - q], r[len(prof) - 1 - q]
                vs = [a, b, c, d] if not flip else [d, c, b, a]
                m.add_f(vs, [(0, 0), (0.01, 0), (0.01, 0.01), (0, 0.01)], slot)
    return rings


def rounded_box(m, c, X, Y, Z, size, e=0.3, nu=16, nv=8, slot="leather", w=None, uv_scale=1.0, g="body"):
    """superquadric box, poles closed with quad-compatible caps (a 4x4 grid would be nicer; poles are small rings
    closed by a fan -> the piece gets triangulated when written)"""
    a, b, h = size
    sgn = lambda x: math.copysign(abs(x) ** e, x)
    c = np.array(c, float); X, Y, Z = (np.array(v, float) for v in (X, Y, Z))
    per = 4 * (a + b)
    rings = []
    for j in range(1, nv):
        phv = -math.pi / 2 + math.pi * j / nv
        ring = []
        for i in range(nu):
            th = 2 * math.pi * i / nu
            x = a * sgn(math.cos(phv)) * sgn(math.cos(th)); y = b * sgn(math.cos(phv)) * sgn(math.sin(th))
            z = h * sgn(math.sin(phv))
            ring.append(m.add_v(c + x * X + y * Y + z * Z, dict(w) if w else None, g))
        rings.append(ring)
    bot = m.add_v(c - h * Z, dict(w) if w else None, g); top = m.add_v(c + h * Z, dict(w) if w else None, g)
    uvf = lambda i, j: (per * i / nu * uv_scale, 2 * h * j / nv * uv_scale)
    for r in range(len(rings) - 1):
        for i in range(nu):
            i2 = (i + 1) % nu
            m.add_f([rings[r][i], rings[r][i2], rings[r + 1][i2], rings[r + 1][i]],
                    [uvf(i, r + 1), uvf(i + 1, r + 1), uvf(i + 1, r + 2), uvf(i, r + 2)], slot)
    for i in range(nu):
        i2 = (i + 1) % nu
        m.add_f([bot, rings[0][i2], rings[0][i]], [uvf(i + 0.5, 0), uvf(i + 1, 1), uvf(i, 1)], slot)
        m.add_f([top, rings[-1][i], rings[-1][i2]], [uvf(i + 0.5, nv), uvf(i, nv - 1), uvf(i + 1, nv - 1)], slot)
    return rings


def lathe(m, O, A, ref, profile, nseg, slot, w=None, g="body", uv_v=None, r_uv=None, close_top=False,
          close_bottom=False):
    """surface of revolution; profile = [(r, h)] along axis A; UV u = angle * r_uv (or arc 0..1), v = profile arc"""
    A = nrm(A); X = nrm(np.array(ref, float) - A * np.dot(ref, A)); Y = np.cross(A, X)
    O = np.array(O, float)
    pts = np.array(profile, float)
    if uv_v is None:
        al = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])
        uv_v = al
    rows = []
    for (r, h) in profile:
        rows.append([m.add_v(O + A * h + r * (math.cos(2 * math.pi * s / nseg) * X + math.sin(2 * math.pi * s / nseg) * Y),
                             dict(w) if w else None, g) for s in range(nseg)])
    ru = r_uv if r_uv is not None else max(p[0] for p in profile)
    for k in range(len(profile) - 1):
        for s in range(nseg):
            s2 = (s + 1) % nseg
            u0, u1 = 2 * math.pi * s / nseg * ru, 2 * math.pi * (s + 1) / nseg * ru
            m.add_f([rows[k][s], rows[k + 1][s], rows[k + 1][s2], rows[k][s2]],
                    [(u0, uv_v[k]), (u0, uv_v[k + 1]), (u1, uv_v[k + 1]), (u1, uv_v[k])], slot)
    for close, k, hsign in ((close_top, len(profile) - 1, 1), (close_bottom, 0, -1)):
        if close:
            c = m.add_v(O + A * profile[k][1], dict(w) if w else None, g)
            for s in range(nseg):
                s2 = (s + 1) % nseg
                vs = [c, rows[k][s], rows[k][s2]] if hsign > 0 else [c, rows[k][s2], rows[k][s]]
                m.add_f(vs, [(0, 0), (0.01, 0), (0.01, 0.01)], slot)
    return rows


def tube_path(m, path, radius, nsides, slot, w_fn, closed=False, cap=True, r_uv=None, twist_ref=(0, 0, 1)):
    """round tube along a polyline (arrows, laces, cords, bow string)"""
    n = len(path)
    T = [nrm(path[min(k + 1, n - 1)] - path[max(k - 1, 0)]) for k in range(n)]
    ref = np.array(twist_ref, float)
    rings = []
    rad = radius if hasattr(radius, "__len__") else [radius] * n
    for k in range(n):
        X = nrm(ref - T[k] * np.dot(ref, T[k])) if abs(np.dot(ref, T[k])) < 0.95 else nrm(np.cross(T[k], (1, 0, 0)))
        Y = np.cross(T[k], X)
        rings.append([m.add_v(path[k] + rad[k] * (math.cos(2 * math.pi * s / nsides) * X + math.sin(2 * math.pi * s / nsides) * Y),
                              w_fn(path[k]), "body") for s in range(nsides)])
    al = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(np.array(path), axis=0), axis=1))])
    ru = r_uv or max(rad)
    for k in range(n - 1):
        for s in range(nsides):
            s2 = (s + 1) % nsides
            u0, u1 = 2 * math.pi * s / nsides * ru, 2 * math.pi * (s + 1) / nsides * ru
            m.add_f([rings[k][s], rings[k][s2], rings[k + 1][s2], rings[k + 1][s]],
                    [(u0, al[k]), (u1, al[k]), (u1, al[k + 1]), (u0, al[k + 1])], slot)
    if cap:
        for k, flip in ((0, True), (n - 1, False)):
            c = m.add_v(path[k], w_fn(path[k]), "body")
            for s in range(nsides):
                s2 = (s + 1) % nsides
                vs = [c, rings[k][s], rings[k][s2]] if not flip else [c, rings[k][s2], rings[k][s]]
                m.add_f(vs, [(0, 0), (0.005, 0), (0.005, 0.005)], slot)
    return rings


# ================================================================================================= preview
def preview_objects(pieces, rig, coll_name="civ_preview"):
    """CMesh pieces -> skinned objects (vertex groups from the weights, Armature modifier) for clay previews / posing"""
    coll = bpy.data.collections.get(coll_name) or bpy.data.collections.new(coll_name)
    if coll.name not in bpy.context.scene.collection.children:
        bpy.context.scene.collection.children.link(coll)
    rng = np.random.default_rng(4)
    out = {}
    for name, m in pieces.items():
        ob = m.to_object("prev_" + name)
        for c in ob.users_collection:
            c.objects.unlink(ob)
        coll.objects.link(ob)
        groups = {}
        for i, w in enumerate(m.W):
            for b, x in (w or {}).items():
                if b not in groups:
                    groups[b] = ob.vertex_groups.new(name=b)
                groups[b].add([i], x, 'REPLACE')
        md = ob.modifiers.new("arm", 'ARMATURE'); md.object = rig
        ob.parent = rig
        mat = bpy.data.materials.new("prev_" + name)
        mat.diffuse_color = (*rng.uniform(0.35, 0.9, 3), 1)
        ob.data.materials.append(mat)
        out[name] = ob
    return out
