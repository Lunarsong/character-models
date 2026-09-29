"""Leg armour of the knight (scripts/armour_lower.py): cuisses, poleyns, greaves, sabatons, and the stepped boot weights,
layered so that nothing interpenetrates in motion (user feedback 17, judge M1 / M2 / M11 / m3).

Principle (hinge layering): two rigid plates on joints that turn relative to each other about a hinge (knee, ankle,
ball of the foot) can only collide where they overlap. In that overlap band the OUTER plate is shaped (pushed out) to
a radius about the hinge axis that is larger than every point of the INNER plate in the band widened by the relative
swing the joint can make. A rotation about the hinge axis moves every point along a circle at its own radius and keeps
its position along the axis, so the inner plate can then never reach the outer one: it slides under it.
  knee   knee_helper_<s> (rig_helpers: half the knee swing) carries the poleyn cop and its side fan; the thigh lame
         lies OVER the cop's top edge and the cuisse over the thigh lame; the shin lame lies over the cop's bottom edge
         and the greave's top slides under the shin lame. With flexion the cop turns away from both lames.
  ankle  the greave's flared foot lies over the rear sabaton lame (foot); both directions of foot pitch are covered.
  ball   the front sabaton lame + toe cap (ball) slide under the last instep lame (foot) when the toes bend.
Every plate component is weighted 100 % to one joint (rigid; the pieces carry rts_weights = "authored").
Coordinates: Blender rig space, metres, +Z up, the character faces -Y, +X is the character's LEFT (left leg built,
right leg mirrored).
"""
import math
import numpy as np
from armour_lower_geo import (smoothstep, nrm, gauss1d, Mesh, plate, AxisField, SLOT)

MAIL = 0.0095          # legs_mail thickness over the skin (tights helper ~6 mm + 2.5 mm offset + margin)
GAP = 0.0025           # radial clearance between layered plates
ROLL_COP, ROLL_LAME, ROLL_PLATE = 0.0030, 0.0018, 0.0026
ROLL_SAB = 0.0016             # sabaton lame / toe-cap rim (plate 3.2 mm thick; 2b, G2: at 1.1 mm the rolled rims'
                              # 20 mm x 1.6 mm quads along the wide rear lame were needle triangles folded 90 deg)
# iteration 2b (user item 28, integrity G1): every plate is a CLOSED shell: outer skin, rolled rim, inner skin 2 x roll
# under the outer one (the hinge rules below already clear an outer plate's full 2 x roll depth)
SHELL = dict(inner=True, inner_slot="steel")


def _lerp(a, b, t):
    return a + (b - a) * t


# ============================================================================================ hinge frames
class Hinge:
    """Cylindrical frame about a joint's hinge axis: point(x, r, psi) = C + X x + r (cos psi F + sin psi U).
    psi = 0 forward (F), +90 deg up (U); x along the axis X (lateral). skin(x, psi) = distance from the axis to the
    outer body surface in the plane at x (rays from outside), NaN-free (0 where the plane misses the body)."""

    def __init__(self, body, C, X, F, xr=(-0.10, 0.10), nx=41, npsi=144, r0=0.30):
        self.C = np.asarray(C, float); self.X = nrm(X)
        F = np.asarray(F, float); self.F = nrm(F - self.X * F.dot(self.X)); self.U = np.cross(self.F, self.X)
        if self.U[2] < 0:                           # keep psi > 0 = up
            self.U = -self.U
        self.xr, self.nx, self.npsi = xr, nx, npsi
        xs = np.linspace(xr[0], xr[1], nx); ps = np.linspace(-math.pi, math.pi, npsi, endpoint=False)
        R = np.zeros((nx, npsi))
        for i, x in enumerate(xs):
            for j, p in enumerate(ps):
                d = math.cos(p) * self.F + math.sin(p) * self.U
                o = self.C + self.X * x + d * r0
                h = body.ray(o, -d, r0)
                R[i, j] = (r0 - h) if h is not None else 0.0
        self.xs, self.ps, self.R = xs, ps, R

    def point(self, x, r, psi):
        x = np.asarray(x, float); r = np.asarray(r, float); psi = np.asarray(psi, float)
        d = np.cos(psi)[..., None] * self.F + np.sin(psi)[..., None] * self.U
        return self.C + self.X * x[..., None] + r[..., None] * d

    def coords(self, P):
        Q = np.asarray(P, float) - self.C
        x = Q @ self.X
        a = Q @ self.F; b = Q @ self.U
        return x, np.hypot(a, b), np.arctan2(b, a)

    def skin(self, x, psi):
        fx = np.clip((np.asarray(x, float) - self.xr[0]) / (self.xr[1] - self.xr[0]) * (self.nx - 1), 0, self.nx - 1.0001)
        fp = ((np.asarray(psi, float) + math.pi) / (2 * math.pi) * self.npsi) % self.npsi
        i0 = np.floor(fx).astype(int); a = fx - i0
        j0 = np.floor(fp).astype(int) % self.npsi; b = fp - np.floor(fp); j1 = (j0 + 1) % self.npsi
        R = self.R
        return (R[i0, j0] * (1 - a) * (1 - b) + R[i0 + 1, j0] * a * (1 - b) + R[i0, j1] * (1 - a) * b
                + R[i0 + 1, j1] * a * b)

    def skin_max(self, x, psi0, psi1, n=24):
        """max skin radius over a psi range (per x)"""
        x = np.asarray(x, float)
        out = np.zeros_like(x)
        for t in np.linspace(0, 1, n):
            out = np.maximum(out, self.skin(x, psi0 + (psi1 - psi0) * t))
        return out


class Profile:
    """radius-per-x profile of a point set inside a psi band (max for 'what is under', min for 'what is over'),
    dilated over neighbouring x bins; lookup with __call__(x)"""

    def __init__(self, H, P, psi0, psi1, mode="max", xr=(-0.12, 0.12), nb=49, dilate=2, fill=None):
        x, r, p = H.coords(P)
        sel = (p >= psi0) & (p <= psi1)
        edges = np.linspace(xr[0], xr[1], nb + 1)
        vals = np.full(nb, np.nan)
        for k in range(nb):
            m = sel & (x >= edges[k]) & (x < edges[k + 1])
            if m.any():
                vals[k] = r[m].max() if mode == "max" else r[m].min()
        ok = ~np.isnan(vals)
        if not ok.any():
            vals[:] = fill if fill is not None else (0.0 if mode == "max" else 9.0)
        else:
            vals[~ok] = fill if fill is not None else (0.0 if mode == "max" else 9.0)
        V = vals.copy()
        for k in range(nb):
            lo, hi = max(0, k - dilate), min(nb, k + dilate + 1)
            V[k] = vals[lo:hi].max() if mode == "max" else vals[lo:hi].min()
        # (iteration 2b, G2: the step where the plate over / under ends folded the pushed plate) a smooth, still safe
        # profile: dilate once more and blur within it (never below / above the dilated values' own envelope)
        V2 = V.copy()
        for k in range(nb):
            lo, hi = max(0, k - dilate), min(nb, k + dilate + 1)
            V2[k] = V[lo:hi].max() if mode == "max" else V[lo:hi].min()
        V = gauss1d(V2[None], 1.2, 1)[0]
        V = np.maximum(V, vals) if mode == "max" else np.minimum(V, vals)
        self.c = 0.5 * (edges[:-1] + edges[1:]); self.v = V

    def __call__(self, x):
        return np.interp(np.asarray(x, float), self.c, self.v)


def push_radial(H, P, prof, psi0, psi1, gap, mode="out", feather=math.radians(6)):
    """move points P (..., 3) radially about the hinge so r >= prof(x) + gap ('out') or r <= prof(x) - gap ('in')
    inside the psi band (feathered over `feather` at the band ends)"""
    shp = P.shape
    Q = P.reshape(-1, 3)
    x, r, p = H.coords(Q)
    w = smoothstep(psi0 - feather, psi0, p) * (1 - smoothstep(psi1, psi1 + feather, p))
    tgt = prof(x) + (gap if mode == "out" else -gap)
    if mode == "out":
        r2 = np.where(r < tgt, r + (tgt - r) * w, r)
    else:
        r2 = np.where(r > tgt, r + (tgt - r) * w, r)
    return H.point(x, r2, p).reshape(shp)


# ============================================================================================ the knee
def knee_hinge(B, side="l"):
    K = B.h("calf_l"); A = nrm(B.t("calf_l") - K)
    X = np.array([1.0, 0.0, 0.0]); X = nrm(X - A * X.dot(A))
    return Hinge(B.s, K, X, np.array([0.0, -1.0, 0.0]), xr=(-0.10, 0.10))


class KneeSpec:
    """cop / lame layout in knee-hinge coordinates (psi in radians)"""
    XC, HW = 0.006, 0.056                 # cop centre (lateral offset from the joint) and half width
    TOP, BOT = math.radians(46), math.radians(-54)      # cop top / bottom at the centre line
    RIDGE = 0.005                         # keel down the middle of the cop

    def psi_top(self, s):
        return self.TOP - math.radians(9) * np.asarray(s) ** 2

    def psi_bot(self, s):
        return self.BOT + math.radians(11) * np.asarray(s) ** 2


def cummax_psi(H, xs, psis, extra):
    """(nx, npsi) running maximum from the lowest psi upwards of skin(x, psi) + extra: the smallest radius profile that
    clears the skin and never decreases towards the top (the cop rule, see poleyn())"""
    X, P = np.meshgrid(xs, psis, indexing="ij")
    R = H.skin(X, P) + extra
    return np.maximum.accumulate(R, axis=1)


def cuisse(B, KH, kinfo=None):
    """Thigh plate over the front and outer thigh, from under the mail skirt (its top 11 cm up under the skirt's
    thigh-weighted hem zone) to 1-3 cm above the knee joint; its lower edge goes under the poleyn cop."""
    F = B.thigh
    t0, t1 = 0.285, B.Lth - 0.030
    th0, th1 = math.radians(-72), math.radians(112)

    def S(u, v):
        u = np.asarray(u, float); v = np.asarray(v, float)
        th = _lerp(th0, th1, u)
        c = np.cos(th * 1.1)
        tt = _lerp(t0, t1 - 0.010 * (1 - np.clip(c, 0, 1)), v)
        off = 0.0135 + 0.003 * smoothstep(0.55, 1.0, v)          # 4 mm over the mail chausses
        P = F.point(tt, th, off)
        if kinfo is not None:            # (2b) clear of the knee fan in the surface itself (a push of the finished
            P = clear_fan(KH, P.reshape(-1, 3), kinfo).reshape(P.shape)      # shell folded its rim: G2 spikes)
        return P
    m, info = plate(S, 12, 10, trim=(2,), band=0.014, roll=ROLL_PLATE, inner=True, inner_slot="steel",
                    outward=lambda p: nrm(p - F.O - F.A * np.dot(p - F.O, F.A)))
    for i in range(len(m.V)):
        m.tagw[i] = {"thigh_l": 1.0}
    return m


def poleyn(B, KH, cuisse_m):
    """Knee cop (knee_helper), side fan (knee_helper), shin lame (calf). Returns (Mesh, info).
    Motion rule: with flexion the cop (half the knee swing) turns DOWN against the thigh and UP against the calf. The
    cop lies OVER the cuisse's lower edge and UNDER the shin lame; its radius about the knee hinge never decreases from
    its bottom edge to its top edge, so whatever part of the cop turns over a fixed spot is always lower than what was
    there: it moves away from the cuisse under it and from the shin lame over it (and the fan is a disc about the
    hinge axis, which turns in its own plane)."""
    S_ = KneeSpec()
    m = Mesh()
    xs = np.linspace(-0.10, 0.10, 81)
    BT, BB = math.radians(13), math.radians(12)            # flat top / bottom bands (psi) of the cop
    # top band: over the cuisse's lower edge; bottom band: under the shin lame. Constant radius (per x) inside each
    # band, so the cop turning over a fixed spot never rises (see the docstring); a free dome between them.
    cu = np.array(cuisse_m.V)
    xc, rc, pc = KH.coords(cu)
    cu = cu[(rc < 0.115) & (np.abs(xc - S_.XC) < S_.HW * 1.35)]
    prof_cu = Profile(KH, cu, S_.TOP - BT - math.radians(24), S_.TOP + math.radians(6), xr=(-0.10, 0.10), nb=41)
    # the cop is an ellipsoidal dome: r(x, psi) = Rc(psi) * shape(x), shape a smooth convex arch across the knee, and
    # Rc constant inside the top / bottom bands (overlaps), free (domed) between them
    A = S_.HW * 1.75
    shape = lambda x: np.sqrt(np.clip(1 - ((np.asarray(x, float) - S_.XC) / A) ** 2, 0.05, 1))
    xf = np.linspace(S_.XC - S_.HW, S_.XC + S_.HW, 25)
    psis = np.linspace(S_.BOT - math.radians(2), S_.TOP + math.radians(2), 77)
    X_, P_ = np.meshgrid(xf, psis, indexing="ij")
    need = (KH.skin(X_, P_) + MAIL + 0.006) / shape(X_)
    rc_need = need.max(0)                                                   # per psi
    # smooth upper envelope of the need (a max over +-9 deg, then blurred): a clean dome instead of a lumpy one that
    # follows the kneecap / tendons under it
    k9 = 9
    rc_need = np.array([rc_need[max(0, i - k9):i + k9 + 1].max() for i in range(len(rc_need))])
    rc_need = np.maximum(rc_need, gauss1d(rc_need[None], 4.0, 1)[0])
    rc_cu = ((prof_cu(xf) + 2 * ROLL_COP + GAP + 0.001) / shape(xf)).max()
    tb0, bb1 = S_.TOP - BT, S_.BOT + BB
    rc_top = max(rc_cu, rc_need[psis >= tb0 - math.radians(9)].max())
    rc_bot = rc_need[psis <= bb1 + math.radians(11)].max()
    DOME = 0.009

    def Rc(psi):
        psi = np.asarray(psi, float)
        w = smoothstep(bb1, tb0, psi)
        mid = np.clip((psi - bb1) / (tb0 - bb1), 0, 1)
        dome = DOME * np.sin(mid * math.pi) ** 1.1
        base = rc_bot + (rc_top - rc_bot) * w
        nd = np.interp(psi, psis, rc_need)
        return np.where(psi >= tb0, rc_top, np.where(psi <= bb1, rc_bot, np.maximum(base + dome, nd + 0.002)))

    def Rv(x, psi, s):
        return Rc(psi) * shape(x)
    out_knee = lambda p: nrm(np.asarray(p) - KH.C - KH.X * np.dot(np.asarray(p) - KH.C, KH.X))

    def cop(u, v):
        u = np.asarray(u, float); v = np.asarray(v, float)
        s = (u - 0.5) * 2
        vv = np.clip(v, 0, 1)
        wf = np.where(vv < 0.4, 0.72 + 0.28 * np.sin(vv / 0.4 * math.pi / 2), 0.58 + 0.42 * np.cos((vv - 0.4) / 0.6 * math.pi / 2))
        x = S_.XC + s * S_.HW * wf
        psi = _lerp(S_.psi_bot(s), S_.psi_top(s), v)
        ridge = S_.RIDGE * np.exp(-((x - S_.XC) / 0.016) ** 2)            # depends on x only: rotation keeps it
        return KH.point(x, Rv(x, psi, s) + ridge, psi)
    n0 = len(m.V)
    plate(cop, 12, 9, trim=(0, 1, 2, 3), band=0.008, roll=ROLL_COP, outward=out_knee, mesh=m, **SHELL)
    for i in range(n0, len(m.V)):
        m.tagw[i] = {"knee_helper_l": 1.0}
    cop_pts = np.array(m.V[n0:])
    # --- side fan: a disc about the hinge axis on the outer side, over the cop's lateral edge (dome by radius only)
    fx = _knee_lateral(B, KH) + MAIL + 0.004
    FAN_R = 0.048

    def fan(u, v):
        u = np.asarray(u, float); v = np.asarray(v, float)
        # (2b, user item 32: the 130-deg annulus from 16 mm read as a loose curved hook / shard beside the knee) a
        # rounded wing: a 170-deg sector from 8 mm round the hinge axis (still a disc about it: it turns in its plane)
        psi = _lerp(math.radians(55), math.radians(225), u)
        rho_max = 0.032 + (FAN_R - 0.032) * np.sin(np.clip(u, 0, 1) * math.pi) ** 0.6
        rho = _lerp(0.008, rho_max, v)                                      # (not from 0: a degenerate row spikes)
        dome = 0.005 * (1 - (rho / FAN_R) ** 2)                            # by radius only
        return KH.point(fx + dome, rho, psi)
    n0 = len(m.V)
    plate(fan, 10, 3, trim=(2,), band=0.008, roll=0.0022, outward=lambda p: KH.X, mesh=m, **SHELL)
    for i in range(n0, len(m.V)):
        m.tagw[i] = {"knee_helper_l": 1.0}
    fan_pts = np.array(m.V[n0:])
    # --- shin lame over the cop's bottom edge (calf): clears the cop + its rim, and leaves room for the greave under it
    band = (-math.radians(13), math.radians(11))
    psi_a, psi_b = S_.BOT + band[0], S_.BOT + math.radians(16) + band[1]
    prof_cop = Profile(KH, cop_pts, psi_a - math.radians(4), psi_b + math.radians(4), xr=(-0.10, 0.10), nb=41)
    need_skin = KH.skin_max(xs, psi_a, psi_b) + MAIL + 0.0175
    rl = np.maximum(need_skin, prof_cop(xs) + 2 * ROLL_LAME + GAP)
    rl = np.maximum(rl, gauss1d(rl[None], 1.5, 1)[0])
    Rl = lambda x: np.interp(np.asarray(x, float), xs, rl)

    def lame(u, v):
        u = np.asarray(u, float); v = np.asarray(v, float)
        s = (u - 0.5) * 2 * 0.97
        x = S_.XC + s * S_.HW * 0.97
        psi = S_.psi_bot(s) + _lerp(band[0], band[1], v)
        bow = 0.0015 * np.sin(np.clip(u, 0, 1) * math.pi)
        return KH.point(x, Rl(x) + bow, psi)
    n0 = len(m.V)
    plate(lame, 10, 2, trim=(0,), band=0.008, roll=ROLL_LAME, outward=out_knee, mesh=m, **SHELL)
    for i in range(n0, len(m.V)):
        m.tagw[i] = {"calf_l": 1.0}
    lame_pts = np.array(m.V[n0:])
    info = dict(cop=cop_pts, fan=fan_pts, lame_dn=lame_pts, psi_dn=(psi_a, psi_b), fan_x=fx, fan_r=FAN_R, spec=S_)
    return m, info


def clear_fan(KH, P, kinfo):
    """keep points of a THIGH plate that come near the knee fan's disc outside it (the fan turns in its plane)"""
    x, r, p = KH.coords(P)
    need = kinfo["fan_x"] + 0.006 + 2 * ROLL_PLATE + GAP
    w = smoothstep(kinfo["fan_r"] + 0.020, kinfo["fan_r"] + 0.006, r) * smoothstep(0.015, 0.035, x)
    x2 = np.where(x < need, x + (need - x) * w, x)
    return KH.point(x2, r, p)


def _knee_lateral(B, KH):
    """largest lateral offset (along the hinge axis) of the knee skin within the fan's reach of the hinge axis"""
    P = np.array(B.s.co[:B.s.NBODY])
    x, r, p = KH.coords(P)
    sel = (r < 0.046) & (x > 0.0) & (x < 0.12)
    return float(x[sel].max()) if sel.any() else 0.06


# ============================================================================================ shin + foot
def ankle_hinge(B):
    C = B.h("foot_l")
    return Hinge(B.s, C, np.array([1.0, 0.0, 0.0]), np.array([0.0, -1.0, 0.0]), xr=(-0.07, 0.07), nx=29)


def ball_hinge(B):
    C = B.h("ball_l")
    return Hinge(B.s, C, np.array([1.0, 0.0, 0.0]), np.array([0.0, -1.0, 0.0]), xr=(-0.07, 0.07), nx=29)


def ankle_blocked(AH, P, rear, gap):
    """hinge layering at the ankle (iteration 2b, integrity G4 greaves|sabatons: the flared foot of the greave cut the
    rear sabaton lame in every pose): True where a greave point lies inside the band the rear lame (foot) sweeps when
    the foot pitches (its rest psi band widened by the clips' plantar / dorsal flexion) at a radius about the ankle
    hinge the lame reaches (+ the greave's shell thickness + gap): a rotation about the hinge keeps every lame point on
    its circle, so a greave that stays outside that radius (or outside the band) is never met by the lame"""
    if rear is None:
        return np.zeros(np.asarray(P).shape[:-1], bool)
    prof, p0, p1 = rear
    x, r, p = AH.coords(np.asarray(P).reshape(-1, 3))
    bad = (p >= p0) & (p <= p1) & (r < prof(x) + gap + 2 * ROLL_PLATE)
    return bad.reshape(np.asarray(P).shape[:-1])


def greaves(B, KH, kinfo, AH, rear=None):
    """Front shell (calf swell, medial ridge, flared ankle) and back shell. The front shell's top slides under the
    poleyn's shin lame (clamped inside it about the knee hinge); its flared foot lies over the rear sabaton lame
    (handled by sabatons(), which keeps that lame under the flare about the ankle hinge)."""
    C = B.calf
    m = Mesh()
    out = lambda p: nrm(p - C.O - C.A * np.dot(p - C.O, C.A))
    ta, tb = 0.046, B.Lcf - 0.026
    lame = kinfo["lame_dn"]
    pa, pb = kinfo["psi_dn"]
    # the shin lame's inner side (outer surface minus its rolled rim) above the greave's top
    prof_over = Profile(KH, lame, pa - math.radians(6), pb + math.radians(6), mode="min")

    cut = {"u": None, "dt": None}

    def front(u, v):
        u = np.asarray(u, float); v = np.asarray(v, float)
        th = _lerp(math.radians(-100), math.radians(100), u)
        cu = np.cos(np.clip(u, 0, 1) * math.pi - math.pi / 2)
        top = ta + 0.024 * (1 - cu) ** 1.3
        bot = tb - 0.020 * (1 - cu) ** 1.2                        # (2b) the lower edge rises over the ankle bones
        if cut["u"] is not None:                                  # (2b) and over the rear sabaton lame's swing
            bot = bot - np.interp(u, cut["u"], cut["dt"])
        tt = _lerp(top, bot, v)
        ridge = 0.0045 * np.exp(-((u - 0.47) / 0.07) ** 2) * (1 - smoothstep(0.75, 0.95, v))
        swell = 0.0035 * np.exp(-((v - 0.30) / 0.18) ** 2) * np.sin(np.clip(u, 0, 1) * math.pi)
        vf = np.clip((tt - ta) / (tb - ta), 0.0, 1.0)
        flare = 0.017 * smoothstep(0.74, 1.0, vf) ** 1.4
        P = C.point(tt, th, 0.0125 + ridge + swell + flare)
        # under the shin lame (inside its inner side) but outside the mail
        P = push_radial(KH, P, prof_over, pa - math.radians(6), pb + math.radians(4), GAP + 0.001, mode="in")
        x, r, p = KH.coords(P)
        rmin = KH.skin(x, p) + MAIL + 0.002 + 2 * ROLL_PLATE            # (2b) the shell's inner face clears the mail
        # (2b, G2 spikes: the hard clamp folded the top rows) a soft max, feathered at the band's ends
        wpsi = smoothstep(pa - math.radians(18), pa - math.radians(12), p) * (1 - smoothstep(pb + math.radians(8), pb + math.radians(14), p))
        rs = 0.5 * (r + rmin + np.sqrt((r - rmin) ** 2 + 0.002 ** 2))
        return KH.point(x, r + (rs - r) * wpsi, p)
    if rear is not None:
        # per column: how far the lower edge must rise so no point of the shell (outer face; its inner skin is inside
        # the thickness counted by ankle_blocked) lies where the rear lame can reach; a smooth running max across
        us = np.linspace(0, 1, 61); vs = np.linspace(0.55, 1.0, 46)
        need = np.zeros(len(us))
        for i, u_ in enumerate(us):
            for dt in np.arange(0.0, 0.060, 0.002):
                cut["u"], cut["dt"] = np.array([0.0, 1.0]), np.array([dt, dt])
                P = front(np.full_like(vs, u_), vs)
                if not ankle_blocked(AH, P, rear, GAP).any():
                    need[i] = dt; break
            else:
                need[i] = 0.060
        need = np.maximum.reduce([np.roll(need, k) for k in (-3, -2, -1, 0, 1, 2, 3)])
        need = np.maximum(need, gauss1d(need[None], 2.5, 1)[0])
        need = np.minimum(need, 0.030)          # (a deeper cut folds the shell's side rows: G2 spikes)
        cut["u"], cut["dt"] = us, need
        __import__("chr_lib").log("greaves: lower edge raised over the rear sabaton lame by up to %.1f mm (centre %.1f)"
                                  % (need.max() * 1000, need[len(need) // 2] * 1000))
    n0 = len(m.V)
    plate(front, 11, 12, trim=(0, 2), band=0.013, roll=ROLL_PLATE, outward=out, mesh=m, **SHELL)
    for i in range(n0, len(m.V)):
        m.tagw[i] = {"calf_l": 1.0}
    front_pts = np.array(m.V[n0:])

    def back(u, v):
        th = _lerp(math.radians(86), math.radians(274), u)
        tt = _lerp(ta + 0.085, tb + 0.004, v)
        swell = 0.004 * np.exp(-((v - 0.25) / 0.2) ** 2) * np.sin(np.clip(u, 0, 1) * math.pi)
        # (iteration 2b, G4: the back shell's flare followed its own v, not the front's height: they crossed at the
        # ankle) the front shell's flare law by height along the calf, then the 8.5 mm step over it
        vf = np.clip((tt - ta) / (tb - ta), 0.0, 1.0)
        flare = 0.017 * smoothstep(0.74, 1.0, vf) ** 1.4
        return C.point(tt, th, 0.0125 + 0.0085 + swell + flare)
    n0 = len(m.V)
    plate(back, 9, 10, trim=(0, 2), band=0.012, roll=ROLL_PLATE, outward=out, mesh=m, **SHELL)
    for i in range(n0, len(m.V)):
        m.tagw[i] = {"calf_l": 1.0}
    return m, {"front": front_pts, "all": np.array(m.V)}


def sabatons(B, AH, BH, ginfo):
    """Laminated sabatons (iteration 2b rewrite, user items 28 / 32, G2 shards / G4 lame-through-lame): three lames
    over the instep and toes and a pointed toe cap, CONCENTRIC bands round the foot field (same angular extent and
    parametrisation, constant offset per lame), each rear lame over the one in front of it by one shell thickness +
    0.9 mm: the old per-point pushes (under the greave, under the neighbour) folded the lame ends into shards.
    Bones: lames 0 / 1 on the foot, lame 2 + cap on the ball. Across the ball hinge lame 2 (and the cap with it) is
    lowered as a whole until it clears lame 1 about the hinge (toes bend); the greave's lower edge rises over the
    ankle bones instead of forcing the rear lame under it. Returns (Mesh, foot field)."""
    s = B.s
    m = Mesh()
    ank, ball, toe = B.ankle, B.ball, B.toe
    x0 = ank[0]
    O = np.array([x0, ank[1] + 0.07, 0.042]); A = np.array([0.0, -1.0, 0.0])
    # (2b) the foot field is the BOOT's surface (rays from the foot axis meet the boot upper / sole slab first), so every
    # lame keeps its offset over the leather itself (the skin field + a constant let the boot through: G4 boots|sabatons)
    bt = getattr(B, "boot_mesh", None)
    fsurf = MeshSurf(bt, keep=lambda c: c[0] > 0.0) if bt is not None else s
    Ff = AxisField(fsurf, O, A, (0, 0, 1), (1, 0, 0), 0.0, 0.34, nt=60, nth=72, sig_t=0.020, sig_th=22, rmax=0.10)
    out = lambda p: nrm(p - Ff.O - Ff.A * np.dot(p - Ff.O, Ff.A))
    t_ank = O[1] - (ank[1] - 0.042)            # (2b) 4.2 cm in front of the ankle: clear of the boot's ankle crease
    t_ball = O[1] - ball[1]
    t_toe = O[1] - toe[1]
    t_cap = t_toe - 0.058
    BOOT = 0.0022 if bt is not None else 0.0095        # over the boot: the lame's own shell (2 x 1.1 mm)
    STEP = 2 * ROLL_SAB + 0.0009            # (2b s3: +2.0 mm tried: G1 barely moved, the greave's flare then cut the lames)
    HALF = math.radians(64)             # (2b: at 76 deg the lames' ends ran up under the greave's sides)
    # (a, b, bone): 3 lames, 10 mm overlaps
    spans = [(t_ank, t_ank + (t_ball - t_ank) * 0.55 + 0.005, "foot_l"),
             (t_ank + (t_ball - t_ank) * 0.55 - 0.005, t_ball + 0.006, "foot_l"),
             (t_ball - 0.004, t_cap + 0.006, "ball_l")]
    n = len(spans)
    base = BOOT + 0.0025
    # (2b) lame 1 is the outermost: lame 0 (the throat lame) lies UNDER it at the front and under the greave's foot at
    # the rear (ankle hinge), lame 2 under lame 1 (ball hinge), the cap under lame 2
    offs = [base + STEP, base + 2 * STEP, base + STEP]
    lames = {}

    def slope_k(t, th):
        """(2b, G1 / G4 sabatons#0|#1: on the steep instep a RADIAL step of one shell + 0.9 mm left only ~0.2 mm
        between the lames along the surface normal, so the rear lame's inner skin cut the lame under it) 1 / cos of
        the angle between the field's radial direction and the surface normal (smoothed field slopes, capped): radial
        offsets x this = offsets along the normal"""
        t = np.asarray(t, float); th = np.asarray(th, float)
        r = Ff.radius(t, th)
        drt = (Ff.radius(t + 0.006, th) - Ff.radius(t - 0.006, th)) / 0.012
        drh = (Ff.radius(t, th + 0.10) - Ff.radius(t, th - 0.10)) / 0.20 / np.maximum(r, 0.01)
        return np.minimum(np.sqrt(1 + drt ** 2 + drh ** 2), 1.8)

    def build_lame(k, off):
        a, b, bone = spans[k]

        def lame(u, v, a=a, b=b, off=off, k=k):
            u = np.asarray(u, float); v = np.asarray(v, float)
            th = _lerp(-HALF, HALF, u)
            t = _lerp(a, b, v)
            lift = 0.0
            if k == 0:
                # (2b) the throat lame's rear rows ride over the boot's ankle crease behind them (its rolled edge cut
                # the leather where the shin rises, G4 boots|sabatons)
                r0 = Ff.radius(t, th)
                rb = np.maximum.reduce([Ff.radius(t - d_, th) for d_ in (0.008, 0.016, 0.024)])
                lift = np.maximum(rb - r0, 0.0) * (1 - smoothstep(0.0, 0.7, v))
            return Ff.point(t, th, (off + lift + 0.0028 * np.sin(np.clip(u, 0, 1) * math.pi)) * slope_k(t, th))
        mm = Mesh()
        # (12 x 4: no needle rim strips along the short end edges, G2 spikes)
        plate(lame, 10, 6, trim=(2,), band=0.008, roll=ROLL_SAB, outward=out, mesh=mm, **SHELL)
        for i in range(len(mm.V)):
            mm.tagw[i] = {bone: 1.0}
        return mm
    for k in range(n):
        lames[k] = build_lame(k, offs[k])
    # across the ball hinge: lame 2 (ball) under lame 1 (foot) for any toe bend: lower lame 2 (and the cap) as a whole
    prof1 = Profile(BH, np.array(lames[1].V), math.radians(0), math.radians(175), mode="min")
    need = 0.0
    x2, r2, p2 = BH.coords(np.array(lames[2].V))
    band = (p2 > math.radians(10)) & (p2 < math.radians(170))
    if band.any():
        need = max(0.0, float((r2[band] - (prof1(x2[band]) - GAP * 0.4)).max()))
    drop = min(need, STEP * 0.8)
    if drop > 0:
        offs[2] -= drop
        lames[2] = build_lame(2, offs[2])
    cap_off = offs[2] - STEP

    def cap(u, v):
        u = np.asarray(u, float); v = np.asarray(v, float)
        th = _lerp(-HALF - math.radians(14), HALF + math.radians(14), u)
        tt = _lerp(t_cap - 0.006, t_toe + 0.013, v)
        tq = np.minimum(tt, t_toe - 0.014)
        r0 = Ff.radius(tq, th) + (cap_off + 0.0028 * np.sin(np.clip(u, 0, 1) * math.pi)) * slope_k(tq, th)
        k_ = np.clip((tt - (t_toe - 0.036)) / 0.049, 0, 1)
        r = r0 * np.sqrt(np.clip(1 - k_ ** 2, 0.0, 1.0)) ** 0.85
        r = np.maximum(r, 0.006)
        side = np.sin(th)
        d = np.cos(th)[..., None] * Ff.F + (side * (1 - 0.16 * k_))[..., None] * Ff.L
        return Ff.O + Ff.A * np.minimum(tt, t_toe + 0.006)[..., None] + r[..., None] * d
    cm = Mesh()
    plate(cap, 10, 5, trim=(0,), band=0.009, roll=ROLL_SAB, outward=out, mesh=cm, **SHELL)
    for i in range(len(cm.V)):
        cm.tagw[i] = {"ball_l": 1.0}
    for k in range(n):
        m.merge(lames[k])
    m.merge(cm)
    log_ = __import__("chr_lib").log
    log_("sabatons: lame offsets %s mm over the foot field, toe cap %.1f mm, ball-hinge drop %.1f mm" % (
        " ".join("%.1f" % (o * 1000) for o in offs), cap_off * 1000, drop * 1000))
    # the rear lame (foot) as seen from the ankle hinge: its largest radius per x and the psi band it sweeps when the
    # foot pitches (plantar 28 deg / dorsal 32 deg: the clips' range plus margin)
    R0 = np.array(lames[0].V)
    x0_, r0_, p0_ = AH.coords(R0)
    rear = (Profile(AH, R0, float(p0_.min()) - 0.1, float(p0_.max()) + 0.1, mode="max", xr=(-0.08, 0.08), nb=33),
            float(p0_.min()) - math.radians(28), float(p0_.max()) + math.radians(32))
    return m, Ff, rear


class MeshSurf:
    """ray target made of one Mesh (e.g. the boot), with the BodySurf ray() interface"""

    def __init__(self, mesh, keep=None):
        from mathutils.bvhtree import BVHTree
        from mathutils import Vector
        V = np.asarray(mesh.V, float)
        F = [f for f in mesh.F if keep is None or keep(V[list(f)].mean(0))]
        self.bvh = BVHTree.FromPolygons([Vector(v) for v in V], F)
        self._V = Vector

    def ray(self, o, d, maxd=0.6):
        h = self.bvh.ray_cast(self._V(o), self._V(nrm(d)), maxd)
        return None if h[0] is None else h[3]


def ff_coords(Ff, P):
    """(t, th, r) of points in an AxisField's frame"""
    Q = np.asarray(P, float) - Ff.O
    t = Q @ Ff.A
    R = Q - np.outer(t, Ff.A)
    return t, np.arctan2(R @ Ff.L, R @ Ff.F), np.linalg.norm(R, axis=1)


def stack_under(Ff, inner, outer, gap, nb=48, feather=0.006):
    """static lame layering (two lames on one bone): move the INNER lame's points radially (about the field axis) to
    at least `gap` under the OUTER lame's innermost point per angle bin (its inner skin), inside the outer lame's
    span along the axis (feathered in over `feather` before it)"""
    to, tho, ro = ff_coords(Ff, outer)
    ti, thi, ri = ff_coords(Ff, inner)
    edges = np.linspace(-math.pi, math.pi, nb + 1)
    prof = np.full(nb, 9.0)
    bi = np.clip(np.digitize(tho, edges) - 1, 0, nb - 1)
    for k in range(nb):
        sel = bi == k
        if sel.any():
            prof[k] = ro[sel].min()
    prof = np.array([prof[[(k - 1) % nb, k, (k + 1) % nb]].min() for k in range(nb)])
    c = 0.5 * (edges[:-1] + edges[1:])
    tgt = np.interp(thi, c, prof, period=2 * math.pi) - gap
    w = smoothstep(to.min() - feather, to.min() - 0.001, ti) * (1 - smoothstep(to.max() + 0.001, to.max() + feather, ti))
    r2 = np.where(ri > tgt, ri + (tgt - ri) * w, ri)
    d = np.cos(thi)[:, None] * Ff.F + np.sin(thi)[:, None] * Ff.L
    return Ff.O + Ff.A * ti[:, None] + r2[:, None] * d


_TOE = {}


def toe_share(B, P, k=6):
    """ball / (foot + ball) of the SKIN nearest each point (inverse-distance mean of k skin vertices of the feet).
    (iteration 2b, G4 boots|underlayer 116: the boot bent about the ball bone's head 8 cm behind the skin's toe crease,
    so at toe-off the sole's front lifted while the foot-weighted skin under it stayed and came out through the sole;
    the leather now bends where the skin does, the rigid sabaton lames keep their hinge at the joint)"""
    if _TOE.get("bm") != B.bm.name:
        from mathutils import kdtree
        bm = B.bm; me = bm.data; co = B.s.co
        gi = {g.name: g.index for g in bm.vertex_groups}
        want = {gi.get(n): n for n in ("foot_l", "ball_l", "foot_r", "ball_r") if n in gi}
        pts, sh = [], []
        for v in me.vertices:
            if v.index >= B.s.NBODY or co[v.index][2] > 0.12:
                continue
            f = b = 0.0
            for g in v.groups:
                n = want.get(g.group)
                if n and n.startswith("foot"):
                    f += g.weight
                elif n:
                    b += g.weight
            if f + b > 0.05:
                pts.append(co[v.index]); sh.append(b / (f + b))
        kd = kdtree.KDTree(len(pts))
        for i, p in enumerate(pts):
            kd.insert(p, i)
        kd.balance()
        _TOE.update(bm=B.bm.name, kd=kd, sh=np.array(sh))
    kd, sh = _TOE["kd"], _TOE["sh"]
    out = []
    for p in np.asarray(P, float):
        hits = kd.find_n(p, k)
        w = np.array([1.0 / max(d, 0.002) ** 2 for _, _, d in hits])
        out.append(float((w * sh[[i for _, i, _ in hits]]).sum() / w.sum()))
    return np.array(out)


def boot_weights(B, P):
    """stepped weights for the leather boot (judge m3: the MakeHuman-interpolated weights stretched the ankle 137-170 %):
    the shaft above the ankle rides the calf with a short blend at the ankle crease; below it the foot / ball split is
    the skin's own (toe_share), so the upper and the sole slab bend at the skin's toe crease."""
    ank, ball = B.ankle, B.ball
    fwv = toe_share(B, P)
    out = []
    for p, fw in zip(P, fwv):
        side = "_l" if p[0] > 0 else "_r"
        up = smoothstep(ank[2] + 0.005, ank[2] + 0.030, p[2])
        w = {"calf" + side: up}
        rest = 1 - up
        if rest > 1e-4:
            w["foot" + side] = rest * (1 - fw)
            w["ball" + side] = rest * fw
        out.append({k: v for k, v in w.items() if v > 1e-4})
    return out


def build_legs(B):
    """-> [(name, Mesh(left + mirrored right), kind)], info (hinges)"""
    from armour_lower import mirror_mesh
    KH = knee_hinge(B); AH = ankle_hinge(B); BH = ball_hinge(B)
    cui = cuisse(B, KH)
    pol, kinfo = poleyn(B, KH, cui)
    cui = cuisse(B, KH, kinfo)                    # the same topology, cleared of the fan in its surface function
    # (iteration 2b) the sabatons first: the greave's flared foot is shaped over the rear lame about the ankle hinge
    sab, Ff, rear = sabatons(B, AH, BH, None)
    grv, ginfo = greaves(B, KH, kinfo, AH, rear=rear)
    out = []
    for name, m in (("cuisses", cui), ("poleyns", pol), ("greaves", grv), ("sabatons", sab)):
        m.merge(mirror_mesh(m))
        out.append((name, m, "plate"))
    return out, dict(knee=KH, ankle=AH, ball=BH, kinfo=kinfo)
