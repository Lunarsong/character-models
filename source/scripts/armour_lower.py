"""Knight LOWER armour, cloth and weapons (reference: refs/knight_sheet.png), as modular MPFB clothes on the rts_human rig.

Pieces (each its own mesh, one MPFB clothes asset rts_knight_<slot>: .mhclo + .obj + .mhmat + .mhw rigid weights +
.rts.json (material slots per face, extra bones)); authored procedurally on the male base human, fitted to any body
by MakeClothes (mhclo correspondence) and loaded with outfit_lib.add_piece:
  legs_mail   mail chausses + breeches (from MakeHuman's tights helper)       mail
  mail_skirt  hauberk skirt, split front / back, brass hem                     mail + plate(gold)
  cuisses     thigh plates, gold hem                                           plate
  poleyns     knee cops with side fans + articulation lames                    plate
  greaves     front + back shells, flared, gold bands                          plate
  sabatons    leather boots + laminated steel sabatons, toe caps               plate (boot leather is on the sheet)
  tassets     3-lame hip tassets hung from the belt                            plate
  belts       waist belt + hip belt, buckles, pouches, frogs                   leather
  tabard      front + back panels hung from under the gorget / pauldrons, gold border, lion, fleur-de-lis   tabard
  cape        mantle over the gorget + pauldron backs, long hanging cape, lion, gold border, lining          cape
  clasps      two gold bosses pinning the cape's mantle to the pauldrons (rigid on the pauldron joint)       plate
Props (own GLBs, attached to socket bones): sword (longsword), shield (kite), scabbard.
Extra bones joined into rts_<kind>: tabard_f_01..03, tabard_b_01..03 (pelvis), cape_{l,c,r}_01..04 (spine_05),
socket_weapon_r (hand_r), socket_shield_l (lowerarm_l), socket_scabbard_l (pelvis), socket_back (spine_05).

Iteration 2: every piece is REGENERATED on the dressed body over the upper armour dressed on it (load_lower ->
regen_on_body; body landmarks Z() / XS()), and the layers that rest on others are CO-SKINNED to them (coskin, again in
the post_clips hook after rig_helpers.apply_plate_rules). Per-frame fit / penetration QA: scripts/armour_lower_fit.py.
run (see scripts/armour_lower.sh):
  Blender -b out/base_male.blend -P scripts/armour_lower.py -- author            write the clothes assets
  Blender -b out/base_<kind>.blend -P scripts/armour_lower.py -- dress <kind>    load them, bones, props -> blend + glb
"""
import sys, os, json, math, time, shutil, uuid
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from chr_lib import *
from armour_lower_geo import *

KL_TEX = os.path.join(TEX, "knight_lower")
# proportions vs the reference sheet, relative to the knee joint height (authoring body)
SKIRT_HEM = 0.115          # mail skirt hem above the knee joint (2b: over the knee cops, was 0.085 and cut them)
TABARD_HEM = 0.012         # front flap hem above the knee joint (reaches the knee cops); back flap +0.02
CAPE_HEM = 0.335           # cape hem (centre point) below the knee joint: ~ankle height
KL_ASSETS = os.path.join(ASSETS, "mpfb_assets", "clothes")
SIDES = ("_l", "_r")


def swap_side(name):
    if name.endswith("_l"):
        return name[:-2] + "_r"
    if name.endswith("_r"):
        return name[:-2] + "_l"
    return name


def mirror_mesh(m):
    mm = m.mirror_x()
    mm.tagw = [({swap_side(k): v for k, v in w.items()} if w else w) for w in mm.tagw]
    return mm


def setw(m, n0, w):
    """rigid weights (bone -> weight) for the vertices added since index n0"""
    for i in range(n0, len(m.V)):
        m.tagw[i] = dict(w)


# ------------------------------------------------------------------------------------------------ body landmarks
# The generators are written in the male authoring body's heights / widths. Z() maps such a height to the current body
# piecewise-linearly between joint landmarks, XS(z) is the trunk width ratio at that (authoring) height; both are the
# identity on the authoring male, so the same generators rebuild every piece on any body (female, proportion builds):
# load_lower() regenerates the rest shapes on the dressed body over its actual upper armour (female sizing).
MALE_LM = (("floor", 0.0), ("foot_l", 0.0768), ("calf_l", 0.5403), ("thigh_l", 1.0124), ("spine_02", 1.1079),
           ("spine_04", 1.2374), ("spine_05", 1.3833), ("upperarm_l", 1.4923), ("clavicle_l", 1.5208),
           ("neck_01", 1.599), ("head", 1.70))
MALE_W = ((0.95, 0.1858), (1.10, 0.1563), (1.30, 0.1756))        # trunk half widths (authoring male) at those heights
_ZMAP = {"src": np.array([v for k, v in MALE_LM]), "dst": np.array([v for k, v in MALE_LM]), "w": np.ones(3)}


def Z(z):
    """authoring-male height -> current body height (see MALE_LM)"""
    a, b = _ZMAP["src"], _ZMAP["dst"]
    z = np.asarray(z, float)
    out = np.interp(z, a, b)
    out = np.where(z > a[-1], b[-1] + (z - a[-1]) * (b[-1] - b[-2]) / (a[-1] - a[-2]), out)
    return float(out) if out.ndim == 0 else out


def XS(z=1.30):
    """trunk width ratio (current body / authoring male) at authoring height z"""
    return float(np.interp(z, [w for w, _ in MALE_W], _ZMAP["w"]))


def set_landmarks(rig, bm):
    Bn = rig.data.bones
    dst = [0.0 if k == "floor" else float(Bn[k].head_local[2]) for k, v in MALE_LM]
    for i in range(1, len(dst)):                       # keep it monotonic
        dst[i] = max(dst[i], dst[i - 1] + 1e-3)
    _ZMAP["dst"] = np.array(dst)
    trunk = BodySurf(bm, exclude_arms=True)
    ws = []
    for zm, wm in MALE_W:
        z = Z(zm); best = 0.0
        for y in np.linspace(-0.08, 0.10, 10):
            h = trunk.ray((0.6, y, z), (-1, 0, 0), 0.6)
            if h is not None:
                best = max(best, 0.6 - h)
        ws.append(best / wm if best > 0 else 1.0)
    _ZMAP["w"] = np.array(ws)
    log("landmarks: z %s  width ratios %s" % (" ".join("%.3f" % v for v in dst[1:]), " ".join("%.3f" % v for v in ws)))


# ------------------------------------------------------------------------------------------------ the authoring body
class Body:
    def __init__(self, rig, bm):
        self.rig, self.bm = rig, bm
        set_landmarks(rig, bm)
        self.s = BodySurf(bm)
        B = rig.data.bones
        self.h = lambda n: np.array(B[n].head_local)
        self.t = lambda n: np.array(B[n].tail_local)
        self.hip = self.h("thigh_l"); self.knee = self.h("calf_l"); self.ankle = self.h("foot_l")
        self.ball = self.h("ball_l"); self.toe = self.t("ball_l")
        # leg fields (left leg; the right is mirrored): front = -Y, outer = +X
        self.thigh = AxisField(self.s, self.hip, self.knee - self.hip, (0, -1, 0), (1, 0, 0), 0.05, 0.52, nt=48, nth=72,
                               sig_t=0.015, sig_th=12)
        self.calf = AxisField(self.s, self.knee, self.ankle - self.knee, (0, -1, 0), (1, 0, 0), -0.12, 0.47, nt=60,
                              nth=72, sig_t=0.012, sig_th=12)
        self.Lth = np.linalg.norm(self.knee - self.hip); self.Lcf = np.linalg.norm(self.ankle - self.knee)
        log("body: thigh %.3f calf %.3f; thigh r(front) %.3f knee r %.3f calf r %.3f ankle r %.3f" % (
            self.Lth, self.Lcf, self.thigh.radius(0.25, 0.0), self.calf.radius(0.0, 0.0), self.calf.radius(0.14, math.pi),
            self.calf.radius(0.42, 0.0)))


# ------------------------------------------------------------------------------------------------ leg plates
def _lerp(a, b, t):
    return a + (b - a) * t


def cuisse(B):
    """Thigh plate: wraps the front and outer thigh from under the mail skirt to just above the knee cop."""
    F = B.thigh
    t0, t1 = 0.20, B.Lth - 0.045
    th0, th1 = math.radians(-72), math.radians(118)

    def S(u, v):
        th = _lerp(th0, th1, u)
        # lower edge pointed at the front (follows the knee cop), top edge straight (hidden)
        c = np.cos(th * 1.1)
        tt = _lerp(t0, t1 - 0.025 * (1 - np.clip(c, 0, 1)), v)
        off = 0.017 + 0.004 * smoothstep(0.6, 1.0, v)          # slight flare at the hem
        return F.point(tt, th, off)
    m, info = plate(S, 12, 6, trim=(2,), band=0.014, roll=0.003, outward=lambda p: nrm(p - F.O - F.A * np.dot(p - F.O, F.A)))
    setw(m, 0, {"thigh_l": 1.0})
    return m


def poleyn(B):
    """Knee cop (domed, pointed top and bottom), outer side fan, one lame above and one below."""
    C = B.calf
    m = Mesh()
    out = lambda p: nrm(p - C.O - C.A * np.dot(p - C.O, C.A))
    # --- the cop
    th0, th1 = math.radians(-78), math.radians(84)
    tc0, tc1 = -0.078, 0.070

    def cop(u, v):
        th = _lerp(th0, th1, u)
        w = np.clip(1 - np.abs(u - 0.47) * 2.1, 0, 1)            # 1 at the front centre
        tt = _lerp(tc0 - 0.016 * w ** 1.5, tc1 + 0.018 * w ** 1.5, v)
        dome = 0.013 * np.sin(np.clip(u, 0, 1) * math.pi) ** 1.2 * np.sin(np.clip(v, 0, 1) * math.pi) ** 0.9
        return C.point(tt, th, 0.0265 + dome)
    n0 = len(m.V)
    plate(cop, 9, 7, trim=(0, 1, 2, 3), band=0.012, roll=0.003, outward=out, mesh=m)
    setw(m, n0, {"thigh_l": 0.5, "calf_l": 0.5})
    # --- side fan (outer side): a scallop that widens towards the back of the knee, with a centre ridge
    f0, f1 = math.radians(70), math.radians(132)

    def fan(u, v):
        u = np.asarray(u, float); v = np.asarray(v, float)
        th = _lerp(f0, f1, u)
        half = 0.022 + 0.036 * np.sin(np.clip(u, 0, 1) * math.pi * 0.62) ** 0.8   # narrow at the cop, wide behind
        tt = _lerp(-half, half, v) - 0.003
        ridge = 0.005 * np.exp(-((v - 0.5) / 0.16) ** 2) * (0.4 + 0.6 * u)
        dome = 0.005 * np.sin(np.clip(v, 0, 1) * math.pi)
        return C.point(tt, th, 0.030 + dome + ridge + 0.003 * u)
    n0 = len(m.V)
    plate(fan, 5, 5, trim=(1, 0, 2), band=0.009, roll=0.0026, outward=out, mesh=m)
    setw(m, n0, {"thigh_l": 0.5, "calf_l": 0.5})
    # --- articulation lames (under the cop edges)
    for (ta, tb, wt, off) in ((-0.118, -0.074, {"thigh_l": 0.8, "calf_l": 0.2}, 0.018),
                              (0.066, 0.104, {"thigh_l": 0.2, "calf_l": 0.8}, 0.017)):
        def lame(u, v, ta=ta, tb=tb, off=off):
            th = _lerp(math.radians(-70), math.radians(96), u)
            return C.point(_lerp(ta, tb, v), th, off + 0.002 * np.sin(np.clip(u, 0, 1) * math.pi))
        n0 = len(m.V)
        plate(lame, 8, 1, trim=((0,) if ta < 0 else (2,)), band=0.009, roll=0.0025, outward=out, mesh=m)
        setw(m, n0, wt)
    return m


def greaves(B):
    """Front shell (with a centre ridge) and back shell, flared over the boot, gold bands top and bottom."""
    C = B.calf
    m = Mesh()
    out = lambda p: nrm(p - C.O - C.A * np.dot(p - C.O, C.A))
    ta, tb = 0.098, B.Lcf - 0.028

    def front(u, v):
        th = _lerp(math.radians(-98), math.radians(98), u)
        cu = np.cos(np.clip(u, 0, 1) * math.pi - math.pi / 2)    # 1 at the front centre
        top = ta + 0.022 * (1 - cu)                               # top edge dips under the knee lame at the front
        tt = _lerp(top, tb, v)
        ridge = 0.0045 * np.exp(-((u - 0.5) / 0.07) ** 2)         # soft centre ridge (shin)
        flare = 0.016 * smoothstep(0.72, 1.0, v)
        return C.point(tt, th, 0.0135 + ridge + flare)
    n0 = len(m.V)
    plate(front, 11, 8, trim=(0, 2), band=0.015, roll=0.003, outward=out, mesh=m)
    setw(m, n0, {"calf_l": 1.0})

    def back(u, v):
        th = _lerp(math.radians(84), math.radians(276), u)
        tt = _lerp(ta + 0.045, tb + 0.004, v)
        flare = 0.017 * smoothstep(0.72, 1.0, v)
        return C.point(tt, th, 0.0175 + flare)
    n0 = len(m.V)
    plate(back, 9, 6, trim=(0, 2), band=0.013, roll=0.003, outward=out, mesh=m)
    setw(m, n0, {"calf_l": 1.0})
    return m


def sabatons(B):
    """Leather boot (upper from the foot surface + sole and heel) with laminated steel lames and a toe cap."""
    s = B.s
    m = Mesh()
    ank, ball, toe = B.ankle, B.ball, B.toe
    x0 = ank[0]
    # foot field: axis along the foot (heel -> toe) at mid-foot height
    O = np.array([x0, ank[1] + 0.07, 0.042]); A = np.array([0.0, -1.0, 0.0])
    Ff = AxisField(s, O, A, (0, 0, 1), (1, 0, 0), 0.0, 0.34, nt=60, nth=72, sig_t=0.010, sig_th=10, rmax=0.10)
    out = lambda p: nrm(p - Ff.O - Ff.A * np.dot(p - Ff.O, Ff.A))
    y_heel = float(np.min([c[1] for c in s.co[s.co[:, 0] > 0.1] if c[2] < 0.03] or [0.0]))
    # --- steel: 4 instep lames (rear ones outside, overlapping forward) + a toe cap
    t_ank = O[1] - (ank[1] - 0.035)                          # just in front of the ankle
    t_toe = O[1] - toe[1]
    lames = []
    L = (t_toe - 0.035 - t_ank)
    n = 4
    for k in range(n):
        a = t_ank + L * k / n - 0.004; b = t_ank + L * (k + 1) / n + 0.010
        lames.append((a, b, 0.0115 + 0.0028 * (n - k)))
    for k, (a, b, off) in enumerate(lames):
        def lame(u, v, a=a, b=b, off=off):
            th = _lerp(math.radians(-84), math.radians(84), u)
            return Ff.point(_lerp(a, b, v), th, off + 0.003 * np.sin(np.clip(u, 0, 1) * math.pi))
        n0 = len(m.V)
        plate(lame, 7, 1, trim=(2,), band=0.009, roll=0.0025, outward=out, mesh=m)
        f = k / (n - 1)
        setw(m, n0, {"foot_l": 1.0 - 0.6 * f, "ball_l": 0.6 * f} if f > 0 else {"foot_l": 1.0})

    def cap(u, v):
        u = np.asarray(u, float); v = np.asarray(v, float)
        th = _lerp(math.radians(-94), math.radians(94), u)
        tc = t_toe - 0.050
        tt = _lerp(tc, t_toe + 0.012, v)
        r0 = Ff.radius(np.minimum(tt, t_toe - 0.012), th) + 0.0105
        k = np.clip((tt - (t_toe - 0.030)) / (0.042), 0, 1)       # rounded nose over the last 4 cm
        r = r0 * np.sqrt(np.clip(1 - k ** 2, 0.0, 1.0)) ** 0.9
        r = np.maximum(r, 0.004)
        d = np.cos(th)[..., None] * Ff.F + np.sin(th)[..., None] * Ff.L
        return Ff.O + Ff.A * np.minimum(tt, t_toe + 0.004)[..., None] + r[..., None] * d
    n0 = len(m.V)
    plate(cap, 7, 4, trim=(0,), band=0.010, roll=0.0025, outward=out, mesh=m)
    setw(m, n0, {"ball_l": 1.0})
    return m, Ff


# ------------------------------------------------------------------------------------------------ helper-based layers
_TIGHTS = {}       # (2b) piece -> the authored helper polygon indices (asset sidecar 'tights_faces'), set by load_lower


def tights_region(B, keep, name=None):
    """Faces of MakeHuman's tights helper (a skin-tight suit ~6 mm off the skin, part of the basemesh so it follows
    every body shape) whose centroid passes keep(c). Returns (points, faces as index tuples into points, normals).
    Iteration 2b: with `name` and an authored face list (_TIGHTS, from the asset) the SAME basemesh faces are taken
    on every body, so the piece keeps the asset's topology and regen_on_body rebuilds it on this body (the female
    boots / chausses kept the male-fitted MakeClothes shapes: the sabatons, greaves and skirt were shaped over a boot
    and chausses that were not the ones she wore). The polygon indices are kept in B.tights_src[name]."""
    bm = B.bm; me = bm.data
    gi = bm.vertex_groups["helper-tights"].index
    ins = np.zeros(len(me.vertices), bool)
    for v in me.vertices:
        for g in v.groups:
            if g.group == gi and g.weight > 0.5:
                ins[v.index] = True
    co = B.s.co
    fixed = _TIGHTS.get(name) if name else None
    if fixed:
        fs = set(int(i) for i in fixed)
        pol = [p for p in me.polygons if p.index in fs]
    else:
        pol = [p for p in me.polygons if ins[list(p.vertices)].all() and keep(co[list(p.vertices)].mean(0))]
    if name:
        if not hasattr(B, "tights_src"):
            B.tights_src = {}
        B.tights_src[name] = [int(p.index) for p in pol]
    faces = [tuple(p.vertices) for p in pol]
    used = sorted(set(i for f in faces for i in f)); remap = {v: k for k, v in enumerate(used)}
    P = co[used].copy(); F = [tuple(remap[i] for i in f) for f in faces]
    N = np.zeros_like(P)
    for f in F:
        fn = np.cross(P[f[1]] - P[f[0]], P[f[2]] - P[f[0]])
        for i in f:
            N[i] += fn
    N /= np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-12)
    return P, F, N


def cyl_uv(P, f, centre, r_ref):
    """metric cylindrical UVs for one face around a vertical axis (unwrapped per face, no seam smear)"""
    ang = [math.atan2(P[i][0] - centre[0], -(P[i][1] - centre[1])) for i in f]
    a0 = ang[0]
    ang = [a0 + ((a - a0 + math.pi) % (2 * math.pi) - math.pi) for a in ang]
    return [(a * r_ref, P[i][2]) for a, i in zip(ang, f)]


def legs_mail(B):
    """Mail chausses + breeches under the plates: MakeHuman tights helper from the waist to mid-calf."""
    # (2b) the chausses run down to the boot shaft (Z 0.155, the boot tops end at 0.17) so the leg inside the greave
    # is closed (G3: rays between greave and sabaton / poleyn reached the hollow leg); push_under keeps them under the
    # greave, whose top now clears them by its own thickness
    zb, zt = Z(0.155), Z(1.00)
    P, F, N = tights_region(B, lambda c: zb < c[2] < zt and abs(c[0]) < 0.32, name="legs_mail")
    # 8.5 mm off the skin on the lower legs, ~4 mm under the skirt / tassets (iteration 2b: the thighs' swing pushed the
    # chausses through the skirt and the tasset lames, G4), ~3 mm at the top (it tucks under the hauberk / mail shirt)
    z_ = P[:, 2]
    # (2b session 3) only under the skirt's side panels / tassets (36-144 deg from the front); at the front and back
    # (under the flaps, no skirt any more) 10 mm off the skin: the skin's hip-flex correctives came 1-5 mm out of the
    # 4 mm chausses on the upper thighs in the run / attack / block (the assembly's gap scan then cut filler mail there)
    aph = np.abs(np.degrees(np.arctan2(P[:, 0], -(P[:, 1] - 0.015))))
    side = smoothstep(26.0, 40.0, aph) * (1 - smoothstep(140.0, 154.0, aph))
    P = P + N * (0.0025 + 0.0015 * (1 - side) * smoothstep(B.knee[2] + 0.10, B.knee[2] + 0.20, z_)
                 - 0.0045 * side * smoothstep(B.knee[2] + 0.20, B.knee[2] + 0.26, z_)
                 - 0.001 * smoothstep(Z(0.90), Z(0.97), z_))[:, None]
    m = Mesh()
    for p in P:
        m.add_v(p)
    crotch = B.hip[2] - 0.09
    # (iteration 2b, user item 31 / G6: one fixed 8 cm reference radius put the rings 1.7x too wide on the calves and
    # squashed on the thighs) u = angle x the chausses' own mean ring radius at that height, per leg / for the hips;
    # the angle's cut is at the inner-back of each leg (least seen), the hips' at the back
    zs_ = [B.ankle[2], B.knee[2], B.hip[2]]
    legc = lambda z, sx: (np.interp(z, zs_, [B.ankle[0], B.knee[0], B.hip[0]]) * sx, np.interp(z, zs_, [B.ankle[1], B.knee[1], B.hip[1]]))
    Pz = P[:, 2]
    leg = Pz < crotch + 0.02
    rad = np.zeros(len(P))
    for i, p in enumerate(P):
        if leg[i]:
            cx, cy = legc(p[2], 1.0 if p[0] > 0 else -1.0)
        else:
            cx, cy = 0.0, 0.02
        rad[i] = math.hypot(p[0] - cx, p[1] - cy)
    zb_ = np.linspace(Pz.min(), Pz.max(), 40)

    def mean_r(sel):
        out = []
        for k in range(len(zb_)):
            mk = sel & (np.abs(Pz - zb_[k]) < 0.02)
            out.append(float(rad[mk].mean()) if mk.any() else np.nan)
        out = np.array(out); ok = ~np.isnan(out)
        return np.interp(zb_, zb_[ok], out[ok]) if ok.any() else np.full(len(zb_), 0.08)
    R_leg, R_hip = mean_r(leg), mean_r(~leg)

    def uv_face(f, c):
        if c[2] < crotch:
            sx = 1.0 if c[0] > 0 else -1.0
            cx, cy = legc(c[2], sx); off = -math.radians(45) * sx; Rt = R_leg
        else:
            cx, cy = 0.0, 0.02; off = 0.0; Rt = R_hip
        ang = [math.atan2(P[i][0] - cx, -(P[i][1] - cy)) + off for i in f]
        ang = [(a + math.pi) % (2 * math.pi) - math.pi for a in ang]
        a0 = ang[0]
        ang = [a0 + ((a - a0 + math.pi) % (2 * math.pi) - math.pi) for a in ang]
        return [(a * float(np.interp(P[i][2], zb_, Rt)), P[i][2]) for a, i in zip(ang, f)]
    # (2b session 3, user item 31 / G6: the per-face cylindrical mapping left 77-80 % of the chausses' visible area
    # outside +-15 % of the ring scale, aniso p50 1.4) one near-isometric sheet by Blender's minimum-stretch (SLIM)
    # unwrap, cut like a pair of hose: up the inner-back line of each leg to the crotch and up the back to the waist;
    # rings level (the world's up = +v), metres. Falls back to the per-face mapping.
    uvs = None
    try:
        def cut_ang(p, c):
            if c[2] < crotch:
                sx = 1.0 if c[0] > 0 else -1.0
                cx, cy = legc(c[2], sx); off = -math.radians(45) * sx
            else:
                cx, cy = 0.0, 0.02; off = 0.0
            a_ = math.atan2(p[0] - cx, -(p[1] - cy)) + off
            return (a_ + math.pi) % (2 * math.pi) - math.pi
        # islands: each leg below the crotch (a tube cut up its inner-back line: near-developable) and the hips (cut
        # up the back and through the crotch from front to back)
        def region(c):
            return 0 if c[2] >= crotch + 0.02 else (1 if c[0] > 0 else 2)
        reg = [region(P[list(f)].mean(0)) for f in F]
        seams = set()
        eface = {}
        for fi, f in enumerate(F):
            c = P[list(f)].mean(0)
            angs = [cut_ang(P[i], c) for i in f]
            straddle = max(angs) - min(angs) > math.pi          # the face lies across the cut line
            for k in range(len(f)):
                i, j = f[k], f[(k + 1) % len(f)]
                e = (min(i, j), max(i, j))
                eface.setdefault(e, []).append(fi)
                if straddle and angs[k] > 0 and angs[(k + 1) % len(f)] > 0:
                    seams.add(e)                                 # its edge on the cut's + side: a line along the cut
                if reg[fi] == 0 and c[2] < crotch + 0.06 and (P[i][0] > 0) != (P[j][0] > 0) and abs(P[i][0] - P[j][0]) < 0.05:
                    seams.add(e)                          # the crotch cut (x = 0) at the bottom of the hips
        for e, fs in eface.items():
            if len({reg[fi] for fi in fs}) > 1:
                seams.add(e)
        # the legs' cuts as connected paths (the straddle rule above can break where the cut runs through a vertex):
        # the cheapest edge path from the ankle rim to the crotch cut, weighted by the angular distance from the cut
        import heapq
        for r_ in (1, 2):
            vset = sorted({i for fi, f in enumerate(F) if reg[fi] == r_ for i in f})
            if not vset:
                continue
            cz = np.array([P[i][2] for i in vset])
            cref = np.array([np.mean([P[i] for i in vset], 0)])[0]
            dev = {i: math.pi - abs(cut_ang(P[i], np.array([cref[0], cref[1], P[i][2] if P[i][2] < crotch else crotch - 0.01]))) for i in vset}
            adj = {}
            for (a_, b_), fs in eface.items():
                if any(reg[fi] == r_ for fi in fs) and a_ in dev and b_ in dev:
                    w_ = float(np.linalg.norm(np.asarray(P[a_]) - np.asarray(P[b_]))) * (1.0 + 6.0 * (dev[a_] + dev[b_]))
                    adj.setdefault(a_, []).append((b_, w_)); adj.setdefault(b_, []).append((a_, w_))
            bot = [i for i in vset if P[i][2] < cz.min() + 0.012]
            top_ = {i for (a_, b_), fs in eface.items() if len({reg[fi] for fi in fs}) > 1 and r_ in {reg[fi] for fi in fs}
                    for i in (a_, b_)}
            if not bot or not top_:
                continue
            src = min(bot, key=lambda i: dev[i])
            dist_ = {src: 0.0}; prev = {}; pq = [(0.0, src)]; hit = None
            while pq:
                d_, u_ = heapq.heappop(pq)
                if d_ > dist_.get(u_, 9e9):
                    continue
                if u_ in top_ and dev[u_] < 0.6:
                    hit = u_; break
                for v_, w_ in adj.get(u_, []):
                    nd = d_ + w_
                    if nd < dist_.get(v_, 9e9):
                        dist_[v_] = nd; prev[v_] = u_; heapq.heappush(pq, (nd, v_))
            if hit is not None:
                # replace this leg's straddle seams by the path
                legv = set(vset)
                seams = {e for e in seams if not (e[0] in legv and e[1] in legv and e not in
                                                  {e2 for e2, fs in eface.items() if len({reg[fi] for fi in fs}) > 1})}
                u_ = hit
                while u_ in prev:
                    seams.add((min(u_, prev[u_]), max(u_, prev[u_]))); u_ = prev[u_]
        uvs = slim_uv(P, F, seams, regions=reg)
    except Exception as e:
        log("WARNING legs_mail SLIM UVs failed (%r): per-face cylindrical UVs" % e)
        uvs = None
    for k, f in enumerate(F):
        c = P[list(f)].mean(0)
        m.add_f(f, uvs[k] if uvs is not None else uv_face(f, c), SLOT["mail"])
    return m


def slim_uv(P, F, seams, iterations=30, regions=None):
    """(2b session 3) near-isometric UVs (metres, per face corner) of the polygons F over the points P by Blender's
    minimum-stretch unwrap with the given seams ((i, j) vertex pairs), rotated so the world's +Z runs along +v"""
    vl = bpy.context.view_layer
    act0 = vl.objects.active
    sel0 = [o for o in vl.objects if o.select_get()]
    mode0 = act0.mode if act0 is not None else 'OBJECT'
    if act0 is not None and mode0 != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')
    me = bpy.data.meshes.new("_slim_uv")
    me.from_pydata([tuple(map(float, p)) for p in P], [], [tuple(int(i) for i in f) for f in F])
    me.update()
    for e in me.edges:
        a_, b_ = e.vertices
        e.use_seam = (min(a_, b_), max(a_, b_)) in seams
    me.uv_layers.new(name="UV")
    ob = bpy.data.objects.new("_slim_uv", me)
    bpy.context.scene.collection.objects.link(ob)
    try:
        for o in vl.objects:
            o.select_set(False)
        vl.objects.active = ob; ob.select_set(True)
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_all(action='SELECT')
        bpy.ops.uv.unwrap(method=os.environ.get("LW_UVM", "MINIMUM_STRETCH"), fill_holes=True, correct_aspect=True,
                          margin=0.0, iterations=iterations)
        bpy.ops.object.mode_set(mode='OBJECT')
        uvl = me.uv_layers.active.data
        out = [[tuple(uvl[li].uv) for li in poly.loop_indices] for poly in me.polygons]
    finally:
        bpy.data.objects.remove(ob, do_unlink=True)
        bpy.data.meshes.remove(me)
        for o in vl.objects:
            o.select_set(o in sel0)
        vl.objects.active = act0
        if act0 is not None and mode0 != 'OBJECT':
            bpy.ops.object.mode_set(mode=mode0)
    # metres: total 3D area / total UV area; orientation: the UV direction of the world's +Z (least squares over faces)
    A3 = Auv = 0.0
    G = np.zeros(2)
    for f, uv in zip(F, out):
        Pf = np.asarray([P[i] for i in f], float); U = np.asarray(uv, float)
        for k in range(1, len(f) - 1):
            e1, e2 = Pf[k] - Pf[0], Pf[k + 1] - Pf[0]
            d1, d2 = U[k] - U[0], U[k + 1] - U[0]
            A3 += 0.5 * np.linalg.norm(np.cross(e1, e2)); Auv += 0.5 * abs(d1[0] * d2[1] - d1[1] * d2[0])
            # gradient of z over the UV triangle
            Mt = np.array([d1, d2]); rhs = np.array([e1[2], e2[2]])
            if abs(np.linalg.det(Mt)) > 1e-14:
                G += np.linalg.solve(Mt, rhs) * 0.5 * np.linalg.norm(np.cross(e1, e2))
    sc_ = math.sqrt(A3 / max(Auv, 1e-18))
    regions = regions if regions is not None else [0] * len(F)
    res = [None] * len(F)
    for r_ in sorted(set(regions)):                        # each island rotated on its own: world +Z along +v
        idx = [k for k in range(len(F)) if regions[k] == r_]
        G = np.zeros(2)
        for k in idx:
            f, uv = F[k], out[k]
            Pf = np.asarray([P[i] for i in f], float); U = np.asarray(uv, float)
            for q in range(1, len(f) - 1):
                e1, e2 = Pf[q] - Pf[0], Pf[q + 1] - Pf[0]
                d1, d2 = U[q] - U[0], U[q + 1] - U[0]
                Mt = np.array([d1, d2]); rhs = np.array([e1[2], e2[2]])
                if abs(np.linalg.det(Mt)) > 1e-14:
                    G += np.linalg.solve(Mt, rhs) * 0.5 * np.linalg.norm(np.cross(e1, e2))
        ang = math.atan2(G[0], G[1])                       # rotate G onto +v
        ca, sa = math.cos(ang), math.sin(ang)
        for k in idx:
            res[k] = [(sc_ * (ca * u - sa * v), sc_ * (sa * u + ca * v)) for u, v in out[k]]
    return res


def boot_cover(B, P, F, N, clr=0.0035, reach=0.014, iters=4, zmin=None, label="boots: upper"):
    """(iteration 2b, integrity G4 boots|underlayer: MakeHuman's tights-helper foot does not cover the skin everywhere
    (the right big toe's medial side stood 5-8 mm OUTSIDE the boot), so the assembly's gap scan saw skin there and cut
    an under-layer patch that runs through the boot) push the boot upper out along its normals until every skin vertex
    of the feet lies at least `clr` inside it (falloff over `reach` round the nearest boot point), smoothed"""
    from mathutils import Vector
    from mathutils.bvhtree import BVHTree
    s = B.s
    co = s.co[:s.NBODY]
    sk = co[(co[:, 2] < float(P[:, 2].max()) - 0.01) & ~s.arm[:s.NBODY] & (co[:, 2] > (zmin if zmin is not None else -1.0))]
    P = P.copy()
    adj = [set() for _ in range(len(P))]
    for f in F:
        for k in range(len(f)):
            adj[f[k]].add(f[(k + 1) % len(f)]); adj[f[k]].add(f[k - 1])
    moved = 0.0
    for it in range(iters):
        bvh = BVHTree.FromPolygons([Vector(p) for p in P], F)
        D = np.zeros(len(P))
        for q in sk:
            loc, nn, fi, dist = bvh.find_nearest(Vector(q), 0.05)
            if loc is None:
                continue
            loc = np.array(loc); nn = np.array(nn)
            vs = list(F[fi])
            nv = N[vs].mean(0); nv = nv / max(np.linalg.norm(nv), 1e-9)
            c = float((loc - q) @ nv)                  # > 0: the boot is outside the skin by c
            deficit = clr - c
            if deficit <= 0:
                continue
            d_ = np.linalg.norm(P - loc, axis=1)
            near = np.flatnonzero(d_ < reach)
            fall = 1 - (d_[near] / reach) ** 2
            D[near] = np.maximum(D[near], deficit * fall)
        if not (D > 1e-5).any():
            break
        # one smoothing ring (no dents at the edge of the pushed patch), never below the need
        D2 = D.copy()
        for i in np.flatnonzero(D > 0):
            for j in adj[i]:
                D2[j] = max(D2[j], D[i] * 0.5)
        P = P + N * D2[:, None]
        moved = max(moved, float(D2.max()))
    log("%s pushed out over the skin by up to %.1f mm" % (label, moved * 1000))
    return P


def boots(B):
    """Leather boot uppers (tights helper feet, 1 cm off the skin) + a sole slab under each foot."""
    P, F, N = tights_region(B, lambda c: c[2] < Z(0.17) and abs(c[0]) < 0.35, name="boots")
    P = P + N * 0.0045
    P = boot_cover(B, P, F, N)
    # the sole slab carries the foot below this: the upper stands 0.8 mm into the slab's flat top (14.5 mm), so the
    # welt is a seated contact, not the upper cut 2.5 mm deep through the sole (iteration 2b, G4 boots self)
    P[:, 2] = np.maximum(P[:, 2], 0.0137)
    m = Mesh()
    m.upper_part = (len(P), F, N)              # (2b) boot_cover again after push_under (build_all_pieces)
    for p in P:
        m.add_v(p)
    for f in F:
        c = P[list(f)].mean(0)
        uv = cyl_uv(P, f, (math.copysign(B.ankle[0], c[0]), -0.06), 0.07)
        m.add_f(f, uv, SLOT["leather"])
    import armour_lower_legs as LG
    m.tagw = LG.boot_weights(B, np.array(m.V))
    # sole: convex footprint of the skin below 2.5 cm, offset 7 mm, 1.4 cm thick with a rounded edge.
    # (iteration 2b, G2 spikes: the upper's clamped underside folded where the tights helper stood outside the
    # skin's mirrored footprint, and showed past the slab) each foot's slab follows ITS OWN footprint of the skin AND
    # the boot upper's low vertices, so the flattened underside is always inside the slab
    s = B.s
    pts = s.co[:s.NBODY]
    Pl = P[P[:, 2] < 0.030]
    for side in (1, -1):
        foot = pts[(pts[:, 2] < 0.025) & (pts[:, 0] * side > 0.05)][:, :2]
        up_ = Pl[Pl[:, 0] * side > 0.05][:, :2]
        hull = _convex_hull(np.vstack([foot, up_]) if len(up_) else foot)
        rr = _resample_closed(hull, 28)                    # counter-clockwise on both feet (own hulls, no mirror)
        cc = rr.mean(0)
        out = np.array([nrm(np.r_[p - cc, 0])[:2] for p in rr])
        prof = [(0.0070, 0.0145), (0.0085, 0.0120), (0.0090, 0.0040), (0.0075, 0.0005)]
        rings = []
        for off, z in prof:
            rings.append([m.add_v((p[0] + o[0] * off, p[1] + o[1] * off, z)) for p, o in zip(rr, out)])
        n = len(rr)
        al = _arclen_closed(rr)
        for r in range(len(rings) - 1):
            for k in range(n):
                k2 = (k + 1) % n
                vs = [rings[r][k], rings[r + 1][k], rings[r + 1][k2], rings[r][k2]]
                uv = [(al[k], prof[r][1]), (al[k], prof[r + 1][1]), (al[k + 1], prof[r + 1][1]), (al[k + 1], prof[r][1])]
                m.add_f(vs, uv, SLOT["leather"])
        # caps: the tread is concentric rings to a centre fan (iteration 2b: the sole bends at the skin's toe crease with
        # the skin's own foot / ball share, so the tread needs vertices across the crease; one fan from the heel to the
        # toe tip lifted its whole front at toe-off and the skin came out under it)
        for rg, z, flip in ((rings[0], prof[0][1], False), (rings[-1], prof[-1][1], True)):
            outer = rg
            base = [np.array(m.V[i][:2]) for i in rg]
            for s_ in ((0.72, 0.46, 0.22) if flip else ()):           # the top cap is inside the upper
                inner = [m.add_v((cc[0] + (b[0] - cc[0]) * s_, cc[1] + (b[1] - cc[1]) * s_, z)) for b in base]
                for k in range(n):
                    k2 = (k + 1) % n
                    vs = [inner[k], outer[k], outer[k2], inner[k2]]
                    vs = vs if not flip else vs[::-1]
                    m.add_f(vs, [tuple(np.array(m.V[i][:2])) for i in vs], SLOT["leather"])
                outer = inner
            c = m.add_v((cc[0], cc[1], z))
            for k in range(n):
                k2 = (k + 1) % n
                vs = [c, outer[k], outer[k2]] if not flip else [c, outer[k2], outer[k]]
                pp = [np.array(m.V[i][:2]) for i in vs]
                m.add_f(vs, [tuple(q) for q in pp], SLOT["leather"])
    import armour_lower_legs as LG
    m.tagw = LG.boot_weights(B, np.array(m.V))
    return m


def _convex_hull(pts):
    pts = sorted(set(map(tuple, np.round(pts, 5))))
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lo, up = [], []
    for p in pts:
        while len(lo) >= 2 and cross(lo[-2], lo[-1], p) <= 0:
            lo.pop()
        lo.append(p)
    for p in reversed(pts):
        while len(up) >= 2 and cross(up[-2], up[-1], p) <= 0:
            up.pop()
        up.append(p)
    return np.array(lo[:-1] + up[:-1])


def _arclen_closed(P):
    Q = np.vstack([P, P[:1]])
    return np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(Q, axis=0), axis=1))])


def _resample_closed(P, n):
    al = _arclen_closed(P); Q = np.vstack([P, P[:1]])
    t = np.linspace(0, al[-1], n, endpoint=False)
    return np.stack([np.interp(t, al, Q[:, k]) for k in range(P.shape[1])], 1)


# ------------------------------------------------------------------------------------------------ upper-armour layers
def upper_layer(slot):
    """The upper-armour agent's authored piece (assets/mpfb_assets/clothes/knight_<slot>/knight_<slot>.obj, written
    on the same authoring body) as a Mesh, so this kit drapes over it; None when it is not there yet."""
    path = os.path.join(KL_ASSETS, "knight_" + slot, "knight_" + slot + ".obj")
    if not os.path.exists(path):
        return None
    before = set(bpy.data.objects)
    bpy.ops.wm.obj_import(filepath=path, use_split_objects=False, use_split_groups=False)
    new = [o for o in bpy.data.objects if o not in before]
    if not new:
        return None
    ob = new[0]
    for o in bpy.context.view_layer.objects:
        o.select_set(o == ob)
    bpy.context.view_layer.objects.active = ob
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=False)
    m = Mesh()
    m.V = [tuple(ob.matrix_world @ v.co) for v in ob.data.vertices]
    m.F = [tuple(p.vertices) for p in ob.data.polygons]
    m.UV = [[(0, 0)] * len(f) for f in m.F]; m.M = [0] * len(m.F); m.tag = [0] * len(m.V); m.tagw = [None] * len(m.V)
    for o in new:
        bpy.data.objects.remove(o, do_unlink=True)
    log("upper layer %s: %d verts (%s)" % (slot, len(m.V), os.path.basename(path)))
    return m


# ------------------------------------------------------------------------------------------------ hips: skirt, tassets
class Hips:
    """Outer hull of the trunk and legs (arms excluded) around the vertical body axis, used by the mail skirt, the
    tassets and the belts."""

    def __init__(self, B):
        self.B = B
        self.trunk = BodySurf(B.bm, exclude_arms=True)
        self.z0, self.z1 = Z(0.50), Z(1.14)
        zs = np.linspace(self.z0, self.z1, 65)
        self.centre = (0.0, 0.015)
        phis, R = hull_field(self.trunk, zs, nphi=96, centre=self.centre, sig_z=0.012, sig_phi=9)
        self.R = Grid2(self.z0, self.z1, 0.0, 2 * math.pi, R, periodic_b=True)
        # hanging skirt radius: from the top down it may not shrink (cloth / mail hangs from the widest point)
        S = R.copy() + 0.010
        for i in range(len(zs) - 2, -1, -1):             # zs ascending: walk down from the top
            S[i] = np.maximum(S[i], S[i + 1] * 0.998 + 0.0005)
        S = gauss1d(gauss1d(S, 1.5, 0), 1.5, 1, wrap=True)
        self.S = Grid2(self.z0, self.z1, 0.0, 2 * math.pi, S, periodic_b=True)

    def resample(self, meshes):
        """hull R over the trunk + extra layers (belts sit over the skirt / tabard)"""
        comp = Composite(self.trunk, meshes)
        self.comp, self.layers = comp, list(meshes)
        zs = np.linspace(self.z0, self.z1, 65)
        phis, R = hull_field(comp, zs, nphi=96, centre=self.centre, sig_z=0.012, sig_phi=9)
        self.R = Grid2(self.z0, self.z1, 0.0, 2 * math.pi, R, periodic_b=True)

    def point(self, z, phi, off=0.0, field="S"):
        F = self.S if field == "S" else self.R
        r = F(z, phi) + off
        cx, cy = self.centre
        return np.stack([cx + r * np.sin(phi), cy - r * np.cos(phi), np.asarray(z, float) * np.ones_like(r)], -1)


SKIRT_NPHI = 40                 # columns round the ring (divisible by 4)
SKIRT_PANEL = (4, 16)           # (2b session 3) side panels span ring columns 4..16 (36..144 deg) on each side


def rim_at(W, ph):
    """(2b s3) the cuirass rim's lowest z in the sector at angle ph (rad from the front, + = left); rim_min without data"""
    rim = W.get("rim") if W else None
    if not rim:
        return np.full_like(np.asarray(ph, float), float((W or {}).get("rim_min", Z(1.030))))
    phis, zm = rim
    P = np.r_[np.asarray(phis) - 2 * math.pi, phis, np.asarray(phis) + 2 * math.pi]; Zm = np.r_[zm, zm, zm]
    return np.interp(np.asarray(ph, float) % (2 * math.pi), P, Zm)


def skirt_top(W):
    """(2b session 3) the skirt's top edge: 1 cm up inside the cuirass's lower rim (it hangs from under the plate)"""
    return float(W.get("rim_min", Z(1.035))) + 0.010


def skirt_off(z, ph, W, zh):
    """the skirt's stand-off over the hull (the tassets lie over it by the same law)"""
    return 0.011 + 0.008 * smoothstep(skirt_top(W), zh, z) + skirt_pad(z, ph, W)


def mail_skirt(H, full=False):
    """Hauberk skirt as two mail panels, one on each hip (the SIDES, 36-144 deg from the front: under the tassets and
    the flaps' edges), from 1 cm up under the cuirass's lower rim down to mid-thigh, gold (brass) hem band.
    Iteration 2b session 3 (integrity G4 legs_mail / cuisses | mail_skirt, mail_skirt | tabard / belts, user items 26,
    27): the front and back of the old ring skirt lay between the chausses / cuisses and the tabard flaps, where the
    thighs' swing (up to 8-11 cm against the pelvis in the lunges) pushed them through the skirt and the skirt through
    the flaps in every clip; under the flaps the chausses show (mail on mail) when a flap swings.
    full=True: a closed ring down to the slit + both halves all round (only a drape layer for the tabard / cape, so
    the flaps keep their shape over where the skirt used to be)."""
    B = H.B
    W = getattr(B, "waist", None) or waist_band(B)
    rim = float(W.get("rim_min", Z(1.030)))
    zt, zh, zs = skirt_top(W), B.knee[2] + SKIRT_HEM, B.hip[2] - 0.13     # top, hem (front), slit top
    band = 0.022
    nphi = SKIRT_NPHI
    gap = math.radians(2.5)
    m = Mesh()
    z_ring = zs if full else rim - 0.004                  # panels: only the rows up under the rim go all round
    ru = [zt, rim - 0.004, rim - 0.020, rim - 0.045]
    ru = [z for z in ru if z_ring + 0.006 < z <= zt]
    rows_up = np.array(sorted(set([round(float(z), 5) for z in ru] + [z_ring]), reverse=True))
    rows_lo = (list(np.linspace(z_ring, zh + band, 8)) if full else
               [z_ring, rim - 0.020, rim - 0.040] + list(np.linspace(rim - 0.065, zh + band, 6))) + [zh]
    def off(z, ph=0.0):
        return skirt_off(z, ph, W, zh)
    def zhem(phi):                                          # the back hangs 2 cm lower
        return zh - 0.02 * (1 - np.cos(phi)) / 2
    phis = np.linspace(0, 2 * math.pi, nphi, endpoint=False)
    c0_, c1_ = SKIRT_PANEL
    in_panel = lambda i: full or (c0_ <= i < c1_) or (nphi - c1_ <= i < nphi - c0_)
    need = lambda i: in_panel(i) or in_panel((i - 1) % nphi)
    up = np.full((len(rows_up), nphi), -1, int)
    for j, z0_ in enumerate(rows_up):
        for i, ph in enumerate(phis):
            if not need(i):
                continue
            z = z0_
            if not full and j < 2:
                # (2b s3, female: the rim is 3 cm higher at the sides than at the front, the skin showed between the rim
                # and a level skirt top) the top rows follow the rim round the body
                z = float(rim_at(W, ph)) + (0.010 if j == 0 else -0.004)
            up[j, i] = m.add_v(H.point(z, ph, off(z, ph)))
            m.tagw[up[j, i]] = skirt_weights(z, ph, zt, W=W)
    Rrow = lambda idx: float(np.mean([math.hypot(m.V[k][0] - H.centre[0], m.V[k][1] - H.centre[1]) for k in idx if k >= 0]))
    Ru = [Rrow(up[j]) for j in range(len(rows_up))]
    mailq = []                                      # (face index, corner keys): the least-squares UVs below (2b, G6)
    for j in range(len(rows_up) - 1):
        for i in range(nphi):
            if not in_panel(i):                          # (2b s3) no skirt under the flaps
                continue
            i2 = (i + 1) % nphi
            vs = [up[j, i], up[j + 1, i], up[j + 1, i2], up[j, i2]]
            a0 = phis[i] - math.pi; a1 = phis[i] + 2 * math.pi / nphi - math.pi
            uv = [(a0 * Ru[j], rows_up[j]), (a0 * Ru[j + 1], rows_up[j + 1]), (a1 * Ru[j + 1], rows_up[j + 1]),
                  (a1 * Ru[j], rows_up[j])]
            mailq.append((len(m.F), [vs[0], vs[1], ("R", vs[2]) if i2 == 0 else vs[2], ("R", vs[3]) if i2 == 0 else vs[3]]))
            m.add_f(vs, uv, SLOT["mail"])
    half = nphi // 2
    if full:
        spans = ((0, half, "thigh_l", True), (half, nphi, "thigh_r", True))
    else:
        c0, c1 = SKIRT_PANEL
        spans = ((c0, c1, "thigh_l", False), (nphi - c1, nphi - c0, "thigh_r", False))
    for side, (ca, cb, bone, slit) in enumerate(spans):
        ncol = cb - ca
        cols = np.array([phis[c % nphi] if c < nphi else 2 * math.pi for c in range(ca, cb + 1)], float)
        grid = np.zeros((len(rows_lo), ncol + 1), int)
        for j, z0 in enumerate(rows_lo):
            for i, ph in enumerate(cols):
                if j == 0:
                    grid[j, i] = up[-1, (ca + i) % nphi]
                    continue
                phx = ph
                if slit:                                    # open the slit: pull the edge columns away from the centre
                    t = (z0 - zs) / (zh - zs)
                    phx = ph + (gap * t if i == 0 else -gap * t if i == ncol else 0.0)
                z = z0 if j < len(rows_lo) - 1 else zhem(phx)
                if j == len(rows_lo) - 2:
                    z = zhem(phx) + band
                grid[j, i] = m.add_v(H.point(z, phx, off(z, phx)))
                m.tagw[grid[j, i]] = skirt_weights(z, phx, zt, side=bone, W=W)
        r = 0.17
        cm = 0.5 * (cols[0] + cols[-1])
        Rl = [Rrow(grid[j]) for j in range(len(rows_lo))]
        for j in range(len(rows_lo) - 1):
            hem = j == len(rows_lo) - 2
            for i in range(ncol):
                vs = [grid[j, i], grid[j + 1, i], grid[j + 1, i + 1], grid[j, i + 1]]
                if hem:
                    al = [cols[i] * r, cols[i + 1] * r]
                    uv = [(al[0], 1.0), (al[0], 0.0), (al[1], 0.0), (al[1], 1.0)]
                    m.add_f(vs, uv, SLOT["trim"])
                else:
                    a0, a1 = cols[i] - cm, cols[i + 1] - cm
                    uv = [(a0 * Rl[j], rows_lo[j]), (a0 * Rl[j + 1], rows_lo[j + 1]), (a1 * Rl[j + 1], rows_lo[j + 1]),
                          (a1 * Rl[j], rows_lo[j])]
                    ks = list(vs)
                    if (ca + i + 1) % nphi == 0 and j == 0:
                        ks[3] = ("R", vs[3])                 # the ring's cut (phi = 2 pi side)
                    mailq.append((len(m.F), ks))
                    m.add_f(vs, uv, SLOT["mail"])
        # rolled hem: one ring curling inwards
        last = grid[-1]
        ring = []
        for i, vi in enumerate(last):
            p = np.array(m.V[vi]); c = np.array([H.centre[0], H.centre[1], p[2]])
            d = nrm(p - c)
            ring.append(m.add_v(p - d * 0.004 + np.array([0, 0, 0.003])))
            m.tagw[ring[-1]] = dict(m.tagw[vi])
        for i in range(ncol):
            vs = [last[i], ring[i], ring[i + 1], last[i + 1]]
            m.add_f(vs, [(cols[i] * r, 0.0), (cols[i] * r, 0.1), (cols[i + 1] * r, 0.1), (cols[i + 1] * r, 0.0)], SLOT["gold"])
        # (iteration 2b, user item 28 / G1) the hem band is a closed solid: an inner band 4 mm inside the brass band,
        # from the rolled hem up to the band's top row, closed there against the outer band
        top = grid[-2]
        inner_top = []
        for i, vi in enumerate(top):
            p = np.array(m.V[vi]); c = np.array([H.centre[0], H.centre[1], p[2]])
            inner_top.append(m.add_v(p - nrm(p - c) * 0.004))
            m.tagw[inner_top[-1]] = dict(m.tagw[vi])
        for i in range(ncol):
            al0, al1 = cols[i] * r, cols[i + 1] * r
            m.add_f([ring[i], inner_top[i], inner_top[i + 1], ring[i + 1]],             # faces the body axis
                    [(al0, 0.0), (al0, 1.0), (al1, 1.0), (al1, 0.0)], SLOT["trim"])
            m.add_f([inner_top[i], top[i], top[i + 1], inner_top[i + 1]],               # faces up (inside)
                    [(al0, 0.9), (al0, 1.0), (al1, 1.0), (al1, 0.9)], SLOT["gold"])
        # the band's two ends (front / back edge of the panel): a quad each
        for i in (0, ncol):
            e = [last[i], top[i], inner_top[i], ring[i]]                              # faces -phi at i = 0
            if i == ncol:
                e = e[::-1]
            m.add_f(e, [(0.0, 0.0), (0.0, 1.0), (0.004, 1.0), (0.004, 0.0)], SLOT["gold"])
    # (iteration 2b, user item 31 / integrity G6) one near-isometric least-squares UV sheet over the band and panels
    try:
        pos = {}
        for fi, ks in mailq:
            for k in ks:
                pos[k] = m.V[k[1] if isinstance(k, tuple) else k]
        W_ = ls_uv_quads(pos, [tuple(ks) for fi, ks in mailq])
        for fi, ks in mailq:
            m.UV[fi] = [W_[k] for k in ks]
    except Exception as e:
        log("WARNING mail skirt LS UVs failed: %r" % e)
    return m


# thigh share of the mail skirt by height: it follows the tasset lames over it (pelvis / hip_helper_01 (1/3) /
# hip_helper_02 (2/3)) and is fully on the thigh from the cuisses' top down, so plates over it move with it
def skirt_pad(z, ph, W):
    """(iteration 2b) extra stand-off of the skirt at the hips' sides just under the cuirass rim (session 3: was under
    the old belt line), where the chausses lie only ~1 cm under it and the thigh's swing pushes them out (G4)"""
    W = W or _WAIST or {"rim_min": Z(1.030)}
    rim = float(W.get("rim_min", Z(1.030)))
    return (0.006 * np.sin(ph) ** 2 * smoothstep(rim - 0.09, rim - 0.04, z)
            * (1 - smoothstep(rim - 0.012, rim, z)))


SKIRT_Z = (0.975, 0.925, 0.872, 0.820, 0.765)
SKIRT_W = (0.0, 0.0, 0.60, 0.95, 1.0)                # (2b: with the tasset lames on 0 / 2/3 / 1 of the hip swing)


def skirt_weights(z, ph, zt, side=None, W=None):
    w = float(np.interp(-z, [-Z(q) for q in SKIRT_Z], SKIRT_W))
    W = W or _WAIST or {"rim_min": Z(1.030)}
    # (2b session 3) up under the cuirass rim: rides the rim (authored placeholder; outfit time: coskin 'beneath')
    rim = float(W.get("rim_min", Z(1.030)))
    top = float(smoothstep(rim - STACK_RAMP_SKIRT[1], rim - STACK_RAMP_SKIRT[0], z))
    out = {}
    if w < 1.0:
        out["pelvis"] = (1.0 - w) * (1 - top)
        if top > 0:
            out[RIM_BONE] = (1.0 - w) * top
    if w > 0:
        if side:
            out[side] = w
        else:                                           # closed ring: left / right thigh, 50/50 at the centre lines
            a = (ph % (2 * math.pi))
            left = float(np.clip(0.5 + 0.5 * math.sin(a) / 0.26, 0, 1))
            if left > 0:
                out["thigh_l"] = w * left
            if left < 1:
                out["thigh_r"] = w * (1 - left)
    tot = sum(out.values())
    return {k: v / tot for k, v in out.items() if v / tot > 1e-4}


def tassets(H):
    """Three overlapping lames on each hip, hung from the hip belt, gold rolled edges."""
    B = H.B
    m = Mesh()
    # (iteration 2b) the top lame hangs right under the waist belt and rides the cuirass rim like the belt and the
    # skirt under it (RIM_BONE: static against both in every pose); its top edge leans in onto the skirt (inner face
    # 1.2 mm over it) so no ray runs up between skirt and tasset (G3 mail skirt-tassets)
    W = getattr(B, "waist", None) or waist_band(B)
    # (2b session 3) hung from under the cuirass rim (the belt is over the plate now), 3 lames down to the old bottom
    z1 = float(W.get("rim_min", Z(1.030))) + 0.006
    zb3 = Z(0.785)
    L = (z1 - zb3)
    lames = [(z1, z1 - 0.40 * L, 0.036), (z1 - 0.33 * L, z1 - 0.70 * L, 0.028), (z1 - 0.63 * L, zb3, 0.020)]
    pa, pb = math.radians(TASSET_SECTOR[0]), math.radians(TASSET_SECTOR[1])
    out = lambda p: nrm(np.r_[p[0] - H.centre[0], p[1] - H.centre[1], 0.0])
    # rigid lames on the rig's hip helpers (rig_helpers: 0 / 1/3 / 2/3 of the hip swing); each LOWER lame lies over
    # the bottom edge of the one above it (it swings further forward, so it rides up over it instead of into it)
    # (2b: the idle stance flexes the leading hip ~16 deg; lames on 1/3 and 2/3 of the swing let the chausses and the
    # skirt under them punch through) lame 2 on 2/3 of the hip swing, lame 3 rides the thigh
    # (2b session 3) top lame on the rim (tucked under it, so no ray runs up between rim and tasset: G3 faulds-
    # tassets), lame 2 on 2/3 of the hip swing, lame 3 on the thigh. (Tried and dropped this session: a per-frame
    # solved hinge chain per side - swung sideways the lames flared out like wings, swung forward about the hip's axis
    # they could not clear the thigh's lateral bulge: mail_skirt | tassets rose from 106 to 219 visible points.)
    # Instead the lames' FRONT ends (where the thigh's front rises under them) flare out and the lower half of the top
    # lame stands further off the skirt, which follows the thigh under it (coskin 'beneath').
    wts = [{RIM_BONE: 1.0}, {"hip_helper_02_l": 1.0}, {"thigh_l": 1.0}]
    W["tasset1_bot"] = float(lames[0][1])
    _WAIST["tasset1_bot"] = float(lames[0][1])
    for k, (za, zb, off) in enumerate(lames):
        def S(u, v, za=za, zb=zb, off=off, k=k):
            ph = _lerp(pa, pb, u)
            curve = 0.012 * np.sin(np.clip(u, 0, 1) * math.pi)       # lower edge rounded
            if k == 0:                                                # (2b s3) the top edge follows the rim, tucked 6 mm
                za = rim_at(W, ph) + 0.006
            z = _lerp(za, zb - curve, v)
            bow = 0.006 * np.sin(np.clip(u, 0, 1) * math.pi)          # plates bow outwards across
            # over the mail skirt surface (same offset law as mail_skirt), upper lames outside the lower ones
            sk = skirt_off(z, ph, W, B.knee[2] + SKIRT_HEM)
            base = 0.011
            if k == 0:
                base = (2 * 0.0026 + 0.0012) + (0.011 - (2 * 0.0026 + 0.0012)) * smoothstep(0.0, 0.35, v)
                bow = bow * smoothstep(0.0, 0.35, v)
            # (by height, the same for overlapping lames: a per-lame law made the lame below poke out through the one
            # above, G1 / G4 tassets self)
            flare = 0.012 * (1 - smoothstep(0.0, 0.40, np.asarray(u, float))) * smoothstep(z1 - 0.030, z1 - 0.070, z)
            lift = 0.008 * smoothstep(z1 - 0.030, z1 - 0.070, z)      # (every lame, by height: order kept)
            return H.point(z, ph, sk + base + 0.0095 * k + bow + flare + lift)
        n0 = len(m.V)
        plate(S, 10, 2, trim=(2,), band=0.011, roll=0.0026, outward=out, mesh=m, rim_gold_all=True, inner=True,
              inner_slot="steel")
        setw(m, n0, wts[k])
    return m


# ------------------------------------------------------------------------------------------------ belts, buckles, pouches
def tube_loop(m, pts, radius, nsides, slot, closed=True, w=None):
    """Tube (round cross-section) along a polyline; returns the ring index lists."""
    pts = np.asarray(pts, float); n = len(pts)
    rings = []
    al = _arclen_closed(pts) if closed else np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])
    for k in range(n):
        a = pts[(k - 1) % n] if (closed or k > 0) else pts[k]
        b = pts[(k + 1) % n] if (closed or k < n - 1) else pts[k]
        T = nrm(b - a)
        ref = np.array([0.0, 0.0, 1.0]) if abs(T[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
        X = nrm(np.cross(T, ref)); Y = np.cross(T, X)
        rings.append([m.add_v(pts[k] + radius * (math.cos(2 * math.pi * s / nsides) * X + math.sin(2 * math.pi * s / nsides) * Y))
                      for s in range(nsides)])
        if w:
            for i in rings[-1]:
                m.tagw[i] = dict(w)
    for k in range(n if closed else n - 1):
        k2 = (k + 1) % n
        for s in range(nsides):
            s2 = (s + 1) % nsides
            # iteration 2b (G1): wound outward (the old order faced into the tube: the buckle frame vanished)
            vs = [rings[k][s2], rings[k2][s2], rings[k2][s], rings[k][s]]
            uv = [(al[k], (s + 1) / nsides * 2 * math.pi * radius), (al[k + 1], (s + 1) / nsides * 2 * math.pi * radius),
                  (al[k + 1], s / nsides * 2 * math.pi * radius), (al[k], s / nsides * 2 * math.pi * radius)]
            m.add_f(vs, uv, SLOT[slot])
    if not closed:                                      # open tube: domed end caps (closed solid)
        for k, sgn in ((0, -1.0), (n - 1, 1.0)):
            a = pts[max(k - 1, 0)] if k else pts[0]; b = pts[min(k + 1, n - 1)] if k < n - 1 else pts[-1]
            T = nrm(pts[1] - pts[0]) if k == 0 else nrm(pts[-1] - pts[-2])
            c = m.add_v(pts[k] + T * sgn * radius * 0.6)
            if w:
                m.tagw[c] = dict(w)
            for s in range(nsides):
                s2 = (s + 1) % nsides
                vs = [c, rings[k][s], rings[k][s2]] if sgn > 0 else [c, rings[k][s2], rings[k][s]]
                m.add_f(vs, [(0.0, 0.0), (0.0, radius), (radius, radius)], SLOT[slot])
    return rings


def buckle(m, c, X, Y, Z, w, h, r=0.0035, w_=None):
    """Gold rectangular buckle frame centred at c (X across the belt, Y up, Z out), plus a steel prong."""
    pts = []
    rr = 0.009                                                # corner radius
    corners = [(w / 2 - rr, h / 2 - rr, 0), (-w / 2 + rr, h / 2 - rr, 90), (-w / 2 + rr, -h / 2 + rr, 180), (w / 2 - rr, -h / 2 + rr, 270)]
    for cx, cy, a0 in corners:
        for k in range(4):
            a = math.radians(a0 + 90 * k / 3)
            pts.append(c + X * (cx + rr * math.cos(a)) + Y * (cy + rr * math.sin(a)) + Z * 0.0)
    # (iteration 2b, G2 spikes: 6-sided tubes fold 60 deg between their long thin faces) 8-sided tubes; the straight
    # runs between the corners are split into <= 5 mm segments (a 28-46 mm quad on a 3.5 mm tube is a needle pair)
    def dense(P_, closed, step=0.005):
        out = []
        n_ = len(P_)
        for k in range(n_ if closed else n_ - 1):
            a_, b_ = np.asarray(P_[k]), np.asarray(P_[(k + 1) % n_])
            m_ = max(1, int(math.ceil(np.linalg.norm(b_ - a_) / step - 1e-3)))     # (float noise must not change the count)
            out += [a_ + (b_ - a_) * (q / m_) for q in range(m_)]
        if not closed:
            out.append(np.asarray(P_[-1]))
        return out
    tube_loop(m, dense(pts, True), r, 8, "gold", w=w_)
    # centre bar and prong
    bar = [c + X * (-w / 2 + 0.004) + Y * (-h / 2 + 0.006) + Z * 0.001, c + X * (-w / 2 + 0.004) + Y * (h / 2 - 0.006) + Z * 0.001]
    tube_loop(m, dense(bar, False), r * 0.8, 8, "gold", closed=False, w=w_)
    prong = [c + X * (-w / 2 + 0.004) + Z * 0.004, c + X * (w / 2 - 0.002) + Z * 0.006]
    tube_loop(m, dense(prong, False), 0.0022, 8, "steel", closed=False, w=w_)


def pouch(m, c, X, Y, Z, size, w_):
    """Leather pouch: rounded box (X across, Y up, Z out from the body) + a flap over the top and front + gold stud."""
    a, b, h = size
    c = np.asarray(c, float)
    n0 = len(m.V)
    # (iteration 2b) the box's axes match the flap's: a across (X), b out from the body (Z), h up (Y) - the old call
    # put b up and h out, so the box stood 7.6 cm off the belt under a flap shaped for 3.4 cm ('brick' pouches)
    f0 = len(m.F)
    rounded_box(c, X, Z, Y, (a, b, h), e=0.28, nu=12, nv=6, slot="leather", mesh=m)
    # (iteration 2b, user item 28 / G1: a solid 4.8 x 2.6 x 5.4 cm box has no inner face within reach of the
    # thickness test from its sides / top, and its superquadric u streaked the leather, G6) a HOLLOW pouch: 3 mm
    # walls (an inner box wound into the cavity), planar metric UVs per face (box mapping by the face normal)
    f1 = len(m.F)
    wall = 0.003
    rounded_box(c, X, Z, Y, (a - wall, b - wall, h - wall), e=0.28, nu=12, nv=6, slot="leather", mesh=m)
    for i in range(f1, len(m.F)):
        m.F[i] = tuple(m.F[i][::-1]); m.UV[i] = list(m.UV[i][::-1])
    Ax = (np.asarray(X, float), np.asarray(Y, float), np.asarray(Z, float))
    for i in range(f0, len(m.F)):
        P_ = np.array([m.V[v] for v in m.F[i]]) - c
        nn = np.cross(P_[1] - P_[0], P_[-1] - P_[0])
        k = int(np.argmax([abs(nn.dot(ax)) for ax in Ax]))
        e1, e2 = [Ax[q] for q in range(3) if q != k]
        m.UV[i] = [(float(q.dot(e1)) + 0.25, float(q.dot(e2)) + 0.25) for q in P_]
    # flap profile in (out, up) from the back-top edge over the top and down the front
    # the flap's closed shell (2 x 1.8 mm) rests on the box's top / front with 0.2 mm to spare (a flap sunk into the
    # box left its outer skin without an inner face in front of the box: G1 one-sided)
    t = 2 * 0.0018 + 0.0010                     # (2b s3: +0.2 mm let the box's rounded edges into the flap, G1)
    prof = np.array([(-b * 0.85, h + t), (0.0, h + t + 0.0008), (b * 0.78, h + t), (b + t, h * 0.72),
                     (b + t, h * 0.25), (b + t, -h * 0.10)])
    al = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(prof, axis=0), axis=1))]); al /= al[-1]

    def S(u, v):
        u = np.asarray(u, float); v = np.asarray(v, float)
        o = np.interp(v, al, prof[:, 0]); up = np.interp(v, al, prof[:, 1])
        x = (u - 0.5) * 2 * a * 1.06 * (1 - 0.12 * v ** 3)      # tapers slightly towards the tip
        return c + x[..., None] * X + up[..., None] * Y + o[..., None] * Z
    plate(S, 6, 5, trim=(), roll=0.0018, body_slot="leather", rim_gold="leather", mesh=m, inner=True, inner_slot="leather",
          outward=lambda p: nrm(np.asarray(p) - c))
    rounded_box(c + Z * (b + t + 0.0036) + Y * (h * 0.08), X, Y, Z, (0.0065, 0.0065, 0.0035), e=0.6, nu=8, nv=4,
                slot="gold", mesh=m)                             # stud on the flap's front (0.1 mm off it)
    setw(m, n0, w_)


def convex_hull_radius(phis, R):
    """radius (per phi) of the convex hull of the closed polar curve R(phi) (centre inside)"""
    P = np.stack([R * np.sin(phis), -R * np.cos(phis)], 1)
    H = _convex_hull(P)
    out = np.empty_like(R)
    Hc = np.vstack([H, H[:1]])
    for k, ph in enumerate(phis):
        d = np.array([math.sin(ph), -math.cos(ph)])
        best = R[k]
        for a, b in zip(Hc[:-1], Hc[1:]):
            # ray from the centre along d against segment a-b
            M = np.array([d, a - b]).T
            if abs(np.linalg.det(M)) < 1e-12:
                continue
            t, u = np.linalg.solve(M, a)
            if t > 0 and -1e-9 <= u <= 1 + 1e-9:
                best = max(best, t)
                break
        out[k] = best
    return out


class BeltPath:
    """The ring a belt lies on (user 21: belts never float): per phi the LARGEST radius of the layers under it across the
    belt's height (rays against the composite: trunk + mail skirt + tassets + tabard + cuirass), so the leather rests on
    the tops and bridges the hollows; closed-ring smoothing never goes below what is underneath."""

    def __init__(self, H, zc, h, tilt=0.0, nphi=240, margin=0.004):
        comp = H.comp
        self.zc, self.h, self.tilt, self.centre = zc, h, tilt, H.centre
        self.phis = np.linspace(0, 2 * math.pi, nphi, endpoint=False)
        R = np.full(nphi, np.nan)
        for k, ph in enumerate(self.phis):
            z0 = zc + tilt * math.sin(ph)
            best = np.nan
            for dz in np.linspace(-h / 2 - margin, h / 2 + margin, 19):
                for dph in (-0.012, 0.0, 0.012):
                    r = hull_radius(comp, z0 + dz, [ph + dph], centre=H.centre)[0]
                    if not np.isnan(r):
                        best = r if np.isnan(best) else max(best, r)
            R[k] = best
        ok = ~np.isnan(R); idx = np.arange(nphi)
        R[~ok] = np.interp(idx[~ok], idx[ok], R[ok], period=nphi)
        raw = R.copy()
        # a belt under tension takes the convex hull of the cross-section; leather over cloth still settles into the
        # hollows, so it stays within 4 mm of what is under it (and never inside it)
        hull = convex_hull_radius(self.phis, raw)
        R = np.minimum(hull, raw + 0.004)
        R = np.maximum(raw, gauss1d(R[None], 1.5, 1, wrap=True)[0])
        # (iteration 2b) circular max filter over +-1 sample (3 deg): the belt ring's straight chords and the plates /
        # straps laid on the path never cut a step under it (the tabard's edge under the belt)
        # (2b s3: +-4 samples of 1.5 deg, more than half a belt segment: the new cuirass's side-plate trims, narrow ridges
        # between the old 3-deg samples, came through the belt's flat faces: G4 belts | cuirass at bind)
        R = np.maximum.reduce([np.roll(R, k) for k in range(-4, 5)])
        self.raw, self.R = raw, R

    def radius(self, ph):
        P = np.r_[self.phis, 2 * math.pi]; Rr = np.r_[self.R, self.R[0]]
        return np.interp(np.asarray(ph, float) % (2 * math.pi), P, Rr)

    def z(self, ph):
        return self.zc + self.tilt * np.sin(ph)

    def point(self, ph, off=0.0, dz=0.0):
        ph = np.asarray(ph, float)
        r = self.radius(ph) + off
        cx, cy = self.centre
        return np.stack([cx + r * np.sin(ph), cy - r * np.cos(ph), self.z(ph) + dz], -1)


def belt_ring(path, clr, nseg=72, thick=0.006, m=None, weights=None):
    """Closed leather belt on a BeltPath (inner face clr off the layers under it). Returns (Mesh, info)."""
    m = m or Mesh()
    h = path.h
    phis = np.linspace(0, 2 * math.pi, nseg, endpoint=False)
    prof = [(0.0, 0.5), (0.6, 0.5), (1.0, 0.36), (1.0, -0.36), (0.6, -0.5), (0.0, -0.5)]   # (out * thick, up * h)
    pv = [0.0, 0.07, 0.14, 0.86, 0.93, 1.0]
    rings = []
    cen = []
    # (iteration 2b) the ring's vertices take the largest path radius within +-0.6 segment of them, so its straight
    # chords never cut a step under it (the tabard's edge under the belt)
    for ph in phis:
        d = nrm(np.r_[math.sin(ph), -math.cos(ph), 0.0])
        p0 = path.point(ph, clr)
        cen.append(p0)
        rings.append([m.add_v(p0 + d * (o * thick) + np.array([0, 0, u * h])) for o, u in prof])
    cen = np.array(cen)
    al = _arclen_closed(cen[:, :2])
    for k in range(nseg):
        k2 = (k + 1) % nseg
        for r in range(len(prof)):
            r2 = (r + 1) % len(prof)
            vs = [rings[k][r], rings[k2][r], rings[k2][r2], rings[k][r2]]
            v0, v1 = pv[r], pv[r2] if r2 else 0.0
            uv = [(al[k], v0), (al[k + 1], v0), (al[k + 1], v1), (al[k], v1)]
            m.add_f(vs[::-1], uv[::-1], SLOT["strap"])
    if weights:
        for rg, ph in zip(rings, phis):
            for i in rg:
                m.tagw[i] = weights(ph)
    return m, dict(phis=phis, centre=cen, rings=rings)


BELT_CLR, BELT_THICK = 0.0010, 0.0045

# ------------------------------------------------------------------------------------------------ the waist band
# Iteration 2b (user item 27: "belt floats and goes through the tabard"; integrity G2 / G3 / G4 belts): the old waist
# belt ran ACROSS the cuirass's lower rim, so it took the rim's radius over its whole height and stood 15-35 mm off the
# tabard / skirt below the rim (it floated), and the crossing hip belt copied the weights of whatever lay under it
# (tabard chain, thighs) and stretched up to 10x. Measured on the knight (every 3rd clip frame): the rim (100 %
# spine_03) moves up to 28 mm against the pelvis in the attack (26 mm down, 21 mm in at the front), so a belt that
# rides the pelvis next to the rim is struck by it, whatever its shape. Now:
#   * one waist belt, its top WAIST_UNDER_RIM below the rim's lowest point (on every body, from the upper armour
#     actually dressed), over a CINCHED tabard band (the cloth pulled onto the mail skirt under the belt, rising over
#     the rim above it) - nothing under the belt is bridged;
#   * the belt, the tabard band and the mail skirt's top rows move EXACTLY like the rim (authored as RIM_BONE, the
#     live rim weights substituted at outfit time by coskin 'waist'), so belt / tabard / skirt / cuirass keep their
#     rest clearances in every pose; the cloth and mail below the belt take up the motion against the pelvis;
#   * the tabard flaps pivot just under the belt, the pouches are threaded on the belt behind the hips; no hip belt
#     (the sword hanger of the sheet would have to cross the swinging flap / tassets; no scabbard is worn).
WAIST_H = 0.050                   # belt height
WAIST_UNDER_RIM = 0.015           # (old layout) belt top below the cuirass's lowest rim point
BELT_OVER_RIM = 0.010             # (2b session 3) belt's lower edge above the cuirass rim's highest point
RIM_BONE = "spine_03"             # authored placeholder for the rim's joint(s) (coskin 'waist' substitutes the live ones)
TAB_BAND_CLR, TAB_BAND_THICK = 0.0045, 0.0035      # tabard outer face over the skirt / cloth thickness in the band
_WAIST = {}


def cuirass_rim(B, nb=24):
    """(phis, zmin): the lowest z of the dressed cuirass per sector round the body axis (nb bins), or None"""
    c = getattr(B, "upper", {}).get("cuirass")
    if c is None:
        return None
    V = np.array(c.V)
    V = V[V[:, 2] < Z(1.20)]
    ph = np.arctan2(V[:, 0], -(V[:, 1] - 0.015)) % (2 * math.pi)
    phis = (np.arange(nb) + 0.5) * 2 * math.pi / nb
    zm = np.full(nb, np.nan)
    for k in range(nb):
        s = np.abs(((ph - phis[k] + math.pi) % (2 * math.pi)) - math.pi) < math.pi / nb
        if s.any():
            zm[k] = V[s, 2].min()
    ok = ~np.isnan(zm)
    if not ok.any():
        return None
    zm[~ok] = np.interp(np.flatnonzero(~ok), np.flatnonzero(ok), zm[ok], period=nb)
    return phis, zm


def waist_band(B):
    """the waist belt's band on this body: dict(zc, h, bot, top, rim_min, rim_hi); also kept in _WAIST for tabard_pivot().
    Iteration 2b session 3 (user item 27, integrity G4 belts | legs_mail, G3 belts-tabard): the belt is buckled OVER THE
    CUIRASS (over the tabard), BELT_OVER_RIM above the plate's highest lowest-edge point, i.e. at the natural waist as
    on the reference sheet. Under the rim (the old belt line, at the hip joints' height) the thighs' swing carried the
    chausses 22-45 mm out through anything that rode the rim; over the plate the belt, the tabard under it and the
    cuirass are one rigid stack in every pose."""
    rim = cuirass_rim(B)
    rim_min = float(rim[1].min()) if rim is not None else Z(1.030)
    rim_hi = float(rim[1].max()) if rim is not None else Z(1.036)
    zc = float(np.clip(rim_hi + BELT_OVER_RIM + WAIST_H / 2, Z(1.00), Z(1.14)))
    W = dict(zc=zc, h=WAIST_H, bot=zc - WAIST_H / 2, top=zc + WAIST_H / 2, rim_min=rim_min, rim_hi=rim_hi, rim=rim)
    _WAIST.clear(); _WAIST.update(W)
    log("waist band: belt %.3f..%.3f over the cuirass (rim lowest %.3f, highest %.3f)" % (W["bot"], W["top"], rim_min, rim_hi))
    return W


def tabard_pivot():
    """height where the tabard flaps start to swing (their chain's first joint): just under the waist belt"""
    bot = _WAIST.get("bot", Z(1.015) - WAIST_H / 2)
    # (2b session 3) under the cuirass rim's lowest point: the cloth over the whole rim rides the rim (rigid with the
    # belt / cuirass), the flaps swing below it
    return min(bot - 0.015, _WAIST.get("rim_min", bot) - 0.005)


def belts(H):
    """Waist belt (5.5 cm, buckle + tongue at the front) over the tabard / breastplate rim, and a hip belt (4 cm) slanting
    down to the left hip over the tabard flap, the mail skirt and the tasset tops, with two pouches hanging from it.
    Both rest on BeltPath rings; outfit time: coskin() gives them the weights of the layers they rest on."""
    B = H.B
    m = Mesh()
    W = getattr(B, "waist", None) or waist_band(B)
    rigid = {RIM_BONE: 1.0}                          # the whole belt rides the cuirass rim (see the waist band notes)
    waist = BeltPath(H, W["zc"], W["h"], margin=0.002)
    belt_ring(waist, BELT_CLR, thick=BELT_THICK, m=m, weights=lambda ph: dict(rigid))
    B.belt_paths = {"waist": waist}
    # buckle at the front of the waist belt: the frame (tube r 3.5 mm) seated 0.5 mm into the belt's outer face
    front = waist.point(0.0, BELT_CLR + BELT_THICK)
    Xa, Ya, Za = np.array([1.0, 0, 0]), np.array([0, 0, 1.0]), np.array([0, -1.0, 0])
    n0 = len(m.V)
    buckle(m, front + Za * 0.0045, Xa, Ya, Za, 0.046, 0.058)          # (2b) frame 1 mm over the belt, not in it (s3: 0.2 mm
                                                                       # cut the belt's rounded face, G1 one-sided)
    # belt tongue (strap end through the buckle, lying on the belt to the character's right): outer skin 3.9 mm over
    # the belt, so its closed shell (2 x roll 2.2 mm) sits 0.5 mm into the belt, not 3 mm through it

    def tongue(u, v):                                   # from just right of the buckle frame (not through it)
        ph = _lerp(-0.14, -0.42, np.asarray(u, float))
        return waist.point(ph, BELT_CLR + BELT_THICK + 0.0046, dz=_lerp(-0.019, 0.019, np.asarray(v, float)))
    plate(tongue, 6, 2, roll=0.0022, body_slot="leather", rim_gold="leather", mesh=m, inner=True, inner_slot="leather",
          outward=lambda p: nrm(np.r_[p[0] - H.centre[0], p[1] - H.centre[1], 0]))
    setw(m, n0, rigid)
    # two pouches threaded on the belt behind the hips (their back on the belt's outer face over its whole height, so
    # nothing hangs free): clear of the tassets (+-52..108 deg), of the hands (front) and deep inside the cape's hang
    for ph, sz in ((math.radians(142), (0.024, 0.013, 0.027)), (math.radians(-143), (0.025, 0.013, 0.028))):
        a, b, h = sz
        rr = float(waist.radius(ph)) + BELT_CLR + BELT_THICK
        # the belt ring bends round the body: seat the pouch's back on the belt's outer face across its width
        dphi = a / max(rr, 0.08)
        r_seat = max(float(waist.radius(ph + d)) for d in np.linspace(-dphi, dphi, 7)) + BELT_CLR + BELT_THICK
        rc = r_seat + b * 0.96
        cx, cy = H.centre
        p = np.array([cx + rc * math.sin(ph), cy - rc * math.cos(ph), W["zc"] - 0.002])
        Zd = nrm(np.r_[math.sin(ph), -math.cos(ph), 0.0]); Xd = nrm(np.cross(np.array([0, 0, 1.0]), Zd)) * -1
        pouch(m, p, Xd, np.array([0, 0, 1.0]), Zd, sz, dict(rigid))
    return m


# ------------------------------------------------------------------------------------------------ tabard and cape
def _winfilter(A, r, fn):
    """(2r+1)^2 window min / max filter (edge-clamped)"""
    P = np.pad(A, r, mode="edge")
    stack = [P[i:i + A.shape[0], j:j + A.shape[1]] for i in range(2 * r + 1) for j in range(2 * r + 1)]
    return fn(np.stack(stack, 0), axis=0)


class Drape:
    """Front / back depth maps of the trunk (arms excluded) for panels that lie on the body and hang below it."""

    def __init__(self, B, H, layers=()):
        self.B, self.H = B, H
        t = Composite(H.trunk, layers) if layers else H.trunk
        self.xs = np.linspace(-0.34, 0.34, 69); self.zs = np.linspace(Z(0.20), Z(1.66), 293)
        Df = fill_nan_rows(depth_map(t, self.xs, self.zs, +1, -0.7))
        Db = fill_nan_rows(depth_map(t, self.xs, self.zs, -1, +0.7))
        # upper envelope before smoothing: cloth bridges over plate rims / rivets instead of cutting through them
        Df = gauss1d(gauss1d(_winfilter(Df, 2, np.min), 1.6, 0), 1.0, 1)
        Db = gauss1d(gauss1d(_winfilter(Db, 2, np.max), 1.6, 0), 1.0, 1)
        self.Df = Grid2(self.zs[0], self.zs[-1], self.xs[0], self.xs[-1], Df)
        self.Db = Grid2(self.zs[0], self.zs[-1], self.xs[0], self.xs[-1], Db)
        self.rawf, self.rawb = Df, Db

    def hang(self, x, z, z_top, clr, side):
        """side = -1 front (panel y = min over [z, z_top] of front depth - clr), +1 back (max of back depth + clr)"""
        zz = np.linspace(0, 1, 24)
        best = None
        for t in zz:
            zq = z + (z_top - z) * t
            if side < 0:
                y = self.Df(zq, x) - clr(zq)
                best = y if best is None else np.minimum(best, y)
            else:
                y = self.Db(zq, x) + clr(zq)
                best = y if best is None else np.maximum(best, y)
        return best


def canvas_xy(P, Wc):
    """canvas metres (X from the canvas' left edge, Y down from the top) of a panel grid P (rows hem..top). Iteration
    2b (integrity G6: arc lengths across each row and down the CENTRE column gave every row one V although the rows
    follow the cuirass rim / neckline per column, so the cloth streaked 4-10x at the sides of the belt band and the
    corners): the near-isometric least-squares embedding of the grid, anchored at the top row's centre"""
    from armour_lower_geo import _ls_uv
    nv1, nu1 = P.shape[:2]
    ic = nu1 // 2
    W = _ls_uv(np.asarray(P, float))
    X = Wc / 2 + (W[..., 0] - W[-1, ic, 0])
    Y = W[-1, ic, 1] - W[..., 1]
    return X, Y


def lining_uv(X, Y, patch, Wc, Hc, u0=0.0, uw=1.0):
    """(2b, G6: the lining squeezed a 0.35 x 0.9 m panel into a square patch, 2.5x anisotropic) the lining at ONE
    scale in both directions into the plain patch (x0, y0, size) of the canvas (metres)"""
    x0, y0, sz = patch
    ex = max(float(X.max() - X.min()), float(Y.max() - Y.min()), 1e-6)
    k = sz / ex
    return u0 + uw * (x0 + k * (X - X.min())) / Wc, 1.0 - (y0 + k * (Y - Y.min())) / Hc


def canvas_uv(P, Wc, Hc, u0=0.0, uw=1.0):
    """Physical UVs of a panel grid P (rows j = hem..top, cols i = left..right as seen on the grid): arc length across
    each row from the centre column and down the centre column from the top, on a canvas Wc x Hc metres mapped to
    u in [u0, u0 + uw], v in [0, 1] (v = 1 at the top). Returns (U, V arrays, outline polygon in canvas metres with
    x from the canvas' left edge and y down from its top edge, as textures_char.Panel expects)."""
    nv1, nu1 = P.shape[:2]
    X, Y = canvas_xy(P, Wc)
    U = u0 + uw * X / Wc; V = 1.0 - Y / Hc
    poly = ([(X[j, 0], Y[j, 0]) for j in range(nv1 - 1, -1, -1)] + [(X[0, i], Y[0, i]) for i in range(1, nu1)] +
            [(X[j, -1], Y[j, -1]) for j in range(1, nv1)] + [(X[-1, i], Y[-1, i]) for i in range(nu1 - 2, 0, -1)])
    return U, V, [[round(float(a), 5), round(float(b), 5)] for a, b in poly]


TABARD_CANVAS = (0.46, 1.05)          # metres per atlas half (front = u 0..0.5, back = u 0.5..1)
CAPE_CANVAS = (1.02, 1.50)


def mesh_depth(mesh, xs, zs, direction, y0, inner=False, stop=0.0):
    """first-hit y of rays along +-Y (direction +1: from the front at y0 < 0) against one Mesh over the (z, x) grid
    (inner=True: the plate's INNER face, i.e. the next hit within 1.5 cm behind the first one, or first + 2 mm);
    NaN = miss"""
    from mathutils.bvhtree import BVHTree
    bvh = BVHTree.FromPolygons([v3(v) for v in mesh.V], [tuple(f) for f in mesh.F])
    Dm = np.full((len(zs), len(xs)), np.nan)
    d = v3((0.0, float(direction), 0.0))
    for i, z in enumerate(zs):
        for j, x in enumerate(xs):
            h = bvh.ray_cast(v3((x, y0, z)), d, abs(y0) + stop)       # this half only (front / back)
            if h[0] is None:
                continue
            y = h[0][1]
            if inner:
                h2 = bvh.ray_cast(h[0] + d * 0.0003, d, 0.015)
                y = h2[0][1] if h2[0] is not None else y + direction * 0.002
            Dm[i, j] = y
    return Dm


def tabard(B, H, D, clr_chest=0.036, over_cuirass=False, D0=None):
    """Tabard: front + back panels from the shoulders to above the knee, open at the sides, narrower below the belt.
    UV (tabard atlas): u 0..1 edge to edge across the panel at every height, v 0..1 hem to top; front panel in the
    left half of the atlas, back in the right half (the dress step maps (u, v) into the atlas regions)."""
    m = Mesh()
    zb = Z(1.015)                                           # waist belt centre
    ztf, ztb = Z(1.50), Z(1.52)
    g = getattr(B, "upper", {}).get("gorget")
    if g is not None:                                       # tuck the neckline 1.2 cm under the gorget's lowest lame
        G = np.array(g.V)
        fr = G[(np.abs(G[:, 0]) < 0.03) & (G[:, 1] < 0)]; bk = G[(np.abs(G[:, 0]) < 0.03) & (G[:, 1] > 0)]
        if len(fr):
            ztf = float(fr[:, 2].min()) + 0.012
        if len(bk):
            ztb = float(bk[:, 2].min()) + 0.012
    # judge M9: the flap was narrow (46 % of the hip width vs the sheet's ~54 %): widened
    specs = {"front": dict(side=-1, z_top=ztf, z_hem=B.knee[2] + TABARD_HEM, hw_top=0.170 * XS(1.30), hw_flap=0.136 * XS(0.95)),
             "back": dict(side=+1, z_top=ztb, z_hem=B.knee[2] + TABARD_HEM + 0.02, hw_top=0.165 * XS(1.30),
                          hw_flap=0.148 * XS(0.95))}
    # necklines (user 15: the tabard hangs from the shoulders, under the gorget and the pauldrons): per column the top
    # edge is tucked 14 mm up under the gorget's lowest lame (22 mm under a pauldron's edge at the front corners) where
    # there is room between that plate and the breast / back plate, and never above the breast / back plate's top edge
    # (it lies on the plate, not over the rim into the arm opening). Found with depth maps of the plates' inner faces.
    ztop_fn = {}
    over = {}                                                # inner-face depth maps of the plates the neckline tucks under
    c = getattr(B, "upper", {}).get("cuirass")
    if g is not None and c is not None:
        xs = np.linspace(0.0, 0.24, 25); zs = np.linspace(Z(1.30), Z(1.64), 86)
        pl = getattr(B, "upper", {}).get("pauldron_l")
        for name, side in (("front", -1), ("back", +1)):
            y0 = -0.6 if side < 0 else 0.6
            Dg = mesh_depth(g, xs, zs, -side, y0, inner=True)
            Dc = mesh_depth(c, xs, zs, -side, y0)
            Dp = mesh_depth(pl, xs, zs, -side, y0, inner=True) if (pl is not None and side < 0) else np.full_like(Dg, np.nan)
            zt = []
            for j, x in enumerate(xs):
                col_c = Dc[:, j]
                okc = ~np.isnan(col_c)
                ctop = zs[okc].max() - 0.008 if okc.any() else Z(1.45)
                best = ctop
                for D_, tuck in ((Dg[:, j], 0.011), (Dp[:, j], -0.012)):
                    ok = ~np.isnan(D_) & okc & (zs < ctop + 0.02)
                    if not ok.any():
                        continue
                    ze = zs[ok].min()                          # the plate's lower edge over the breast / back plate
                    k0 = np.searchsorted(zs, ze); k1 = min(np.searchsorted(zs, ze + max(tuck, 0.0)), len(zs) - 1)
                    gaps = [-side * (col_c[k] - D_[k]) for k in range(k0, k1 + 1)
                            if not (np.isnan(col_c[k]) or np.isnan(D_[k]))]
                    gap = min(gaps) if gaps else 0.0
                    # room for the cloth (outer face 8 mm off the plate, 1.5 mm under the plate over it) + margin
                    best = min(best, ze + tuck if (gap >= 0.0115 or tuck < 0) else ze - 0.004)
                zt.append(best)
            zt = np.array(zt)
            zt = np.convolve(np.pad(zt, 1, mode="edge"), np.ones(3) / 3, mode="valid")
            ztop_fn[name] = lambda x, xs=xs, zt=zt: np.interp(np.abs(x), xs, zt)
            over[name] = [Grid2(zs[0], zs[-1], xs[0], xs[-1], np.nan_to_num(D_, nan=side * 9.0)) for D_ in (Dg, Dp)]
            specs[name]["z_top"] = float(zt.max())
            log("tabard %s neckline z(x): %s" % (name, " ".join("%.3f" % v for v in zt[::3])))
        specs["front"]["hw_top"] = 0.178 * XS(1.30)
        specs["front"]["widen"] = 0.004 * XS(1.30)                    # the top corners spread out under the pauldrons
    # iteration 2b (G2 spikes: the 4 mm rim strips along 5-6 cm side rows were needle triangles folded 90 deg off the
    # panel): 5.5 mm cloth + lining. User item 27 / 26 (see the waist band notes at belts()): under the waist belt the
    # cloth is CINCHED onto the mail skirt (outer face TAB_BAND_CLR over it, 3.5 mm thick), above the belt it rises
    # over the cuirass's lower rim (reaching the over-plate depth 2 mm below the rim at every column), below it it
    # widens into the hanging flap over 4 cm; rows are placed on those heights (knots), not spread evenly.
    W = getattr(B, "waist", None) or waist_band(B)
    wb, wt = W["bot"], W["top"]
    zb = wb                                                  # the flaps hang from the belt's lower edge
    thick0 = 0.0055
    info = {}
    for name, sp in specs.items():
        side = sp["side"]
        # (2b session 3) the belt and the cinched band lie OVER the cuirass (waist_band): above the belt the cloth lies on
        # the plate (depth maps with the plate), under the belt it is cinched onto the plate, below the belt it falls
        # over the plate's lower rim (D.hang: the rim flare is in the depth maps) into the flaps, which swing from under
        # the rim (tabard_pivot)
        def clr(z, U=None):
            if over_cuirass:                               # the depth maps already contain the breast / back plate
                # 11 mm under the gorget's lowest lame at the neckline, 7.5 mm over the lower breast / back plate
                c0 = 0.0075 + 0.0025 * smoothstep(wt + 0.04, wt + 0.10, z)
                c_ = c0 + (0.011 - c0) * smoothstep(Z(1.26), Z(1.36), z) - 0.003 * smoothstep(Z(1.36), Z(1.43), z)
                c_ = np.where(z > zb - 0.05, c_, 0.010)
            else:
                chest = clr_chest if side < 0 else clr_chest + 0.004
                c_ = np.where(z > zb + 0.04, chest, np.where(z > zb - 0.05, chest - 0.010, 0.010))
            if U is not None:
                # G3 tabard-breastplate: the side edges lie on the plate (a 1.5 mm stand-off), so no ray runs up the
                # gap between cloth and plate to the collar / helm interior
                ed = smoothstep(0.72, 0.97, np.abs(2 * np.asarray(U, float) - 1)) * smoothstep(wt + 0.035, wt + 0.06, z)
                c_ = c_ * (1 - ed) + (thick0 + 0.0015) * ed
            return c_

        def hw(z, sp=sp):
            # the width changes under the belt's upper half (hidden; its lower edge then lies on a straight panel
            # edge: no slot under it at the flap's edges, G3 belts-tabard)
            t = smoothstep(wb + 0.026, wt - 0.004, z)
            if sp.get("widen"):
                neck = sp["hw_top"] + sp["widen"] * smoothstep(Z(1.34), Z(1.45), z)
            else:
                neck = sp["hw_top"] - 0.018 * smoothstep(Z(1.36), sp["z_top"], z)    # slightly narrower over the shoulders
            return sp["hw_flap"] + (neck - sp["hw_flap"]) * t

        # row knots (hem .. top): flap rows up to the pivot (under the rim), one over the rim, the band (bottom / middle /
        # top), 6 rows over the belt; the chest rows above spread to the neckline
        zh = sp["z_hem"]
        zp = tabard_pivot()
        nv_ = 32
        n_flap = 10
        kz0 = [float(k) for k in np.linspace(zh, zp, n_flap + 1)] + [0.5 * (zp + wb - 0.006), wb - 0.006, wb,
                                                                     wb + W["h"] / 2, wt]
        KUP = (0.006, 0.013, 0.021, 0.031, 0.044, 0.060)
        nk = len(kz0) + len(KUP)
        z_chest0 = wt + KUP[-1]

        def zmap(U, V, zt_top):
            V = np.asarray(V, float); U = np.asarray(U, float)
            K = [np.full_like(V, k) for k in kz0] + [np.full_like(V, wt + d_) for d_ in KUP]
            K = np.stack(K, 0)                                   # (nk, ...)
            r = V * nv_
            k0 = np.clip(np.floor(r).astype(int), 0, nk - 2)
            f = np.clip(r - k0, 0.0, 1.0)
            za = np.take_along_axis(K, k0[None], 0)[0]; zb_ = np.take_along_axis(K, (k0 + 1)[None], 0)[0]
            lo = za + (zb_ - za) * f
            vk = (nk - 1) / nv_
            t = np.clip((V - vk) / max(1.0 - vk, 1e-9), 0.0, 1.0)
            return np.where(V <= vk, lo, z_chest0 + (zt_top - z_chest0) * t)

        def band_w(z):
            """1 in the cinched band under the belt, 0 away from it"""
            return smoothstep(wb - 0.012, wb - 0.002, z) * (1 - smoothstep(wt + 0.001, wt + 0.006, z))

        def Pf(U, V, sp=sp, side=side, clr=clr, hw=hw, name=name):
            if name in ztop_fn:                              # shaped neckline: top row height depends on x
                xt = (U - 0.5) * 2 * hw(np.full_like(U, sp["z_top"]))
                zt_top = ztop_fn[name](xt)
            else:
                zt_top = np.full_like(np.asarray(U, float), sp["z_top"])
            z = zmap(U, V, zt_top)
            x = (U - 0.5) * 2 * hw(z)
            # lying on the body (over the breastplate) above the belt, hanging below it
            cz = clr(z, U)
            y_on = (D.Df(z, x) - cz) if side < 0 else (D.Db(z, x) + cz)
            for Gd in over.get(name, []):                   # stay 3.5 mm inside the gorget / pauldron it tucks under
                yg = Gd(z, np.abs(x))                       # (2b, G4 gorget|tabard at bind: 1.5 mm let the lame's rolled
                if side < 0:                                # rim, below its inner face, cut the neckline)
                    y_on = np.where(yg > -1.0, np.maximum(y_on, yg + 0.0035), y_on)
                else:
                    y_on = np.where(yg < 1.0, np.minimum(y_on, yg - 0.0035), y_on)
            # the cinched band: on the cuirass under the belt (2b session 3), rising to the over-plate drape over 3 cm
            y_band = (D.Df(z, x) - TAB_BAND_CLR) if side < 0 else (D.Db(z, x) + TAB_BAND_CLR)
            rise = smoothstep(wt, wt + 0.030, z)
            y_up = y_band + (y_on - y_band) * rise
            # below the belt: from the band into the hanging flap (2b session 3: over the cuirass's rolled lower rim
            # with 12 mm, the smoothed depth maps round its flare off; the fall takes 1 cm, not 4)
            clr_h = lambda zq, clr=clr: np.maximum(clr(zq), 0.012 * smoothstep(zp - 0.030, zp - 0.005, zq))
            y_hang = D.hang(x, np.minimum(z, zb), zb, clr_h, side)
            fall = smoothstep(zb, zb - 0.010, z)
            y_dn = y_band + (y_hang - y_band) * fall
            y = np.where(z >= wt, y_up, np.where(z >= zb, y_band, y_dn))
            below = smoothstep(zb, sp["z_hem"], z)
            y = y + side * (0.012 * below)                        # slight flare away from the body
            # judge M9: 2-3 soft vertical folds down the flap, fine gathers just under the belt
            y = y + side * 0.012 * below ** 0.8 * (0.5 + 0.5 * np.sin(U * math.pi * 5.0 + 0.6)) ** 0.9   # one-sided
            gath = smoothstep(zb - 0.012, zb - 0.035, z) * (1 - smoothstep(zb - 0.06, zb - 0.17, z))
            y = y + side * 0.0040 * gath * (0.5 + 0.5 * np.sin(U * math.pi * 11.0 + 0.3))
            return np.stack([x, y, z], -1)
        e = 1e-3

        def PfN(U, V, Pf=Pf, side=side):
            P = Pf(U, V)
            Pu = Pf(np.clip(U + e, 0, 1), V) - Pf(np.clip(U - e, 0, 1), V)
            Pv = Pf(U, np.clip(V + e, 0, 1)) - Pf(U, np.clip(V - e, 0, 1))
            N = np.cross(Pu, Pv); N /= np.maximum(np.linalg.norm(N, axis=-1, keepdims=True), 1e-12)
            N = N * np.sign((N[..., 1] * side).mean() or 1.0)
            return P, N

        def thick(U, V, Pf=Pf):
            z = Pf(U, V)[..., 2]
            return thick0 + (TAB_BAND_THICK - thick0) * band_w(z)
        ox = 0.0 if side < 0 else 0.5
        n0 = len(m.V)
        nu_ = 12                                            # (2b) short top-row edges at the neckline corners (G2)
        Ug, Vg = np.meshgrid(np.linspace(0, 1, nu_ + 1), np.linspace(0, 1, nv_ + 1))
        Pg = Pf(Ug, Vg)
        if side > 0:                                        # seen from behind: mirror so the canvas reads upright
            Pg_c = Pg[:, ::-1]
        else:
            Pg_c = Pg
        Uc, Vc, poly = canvas_uv(Pg_c, *TABARD_CANVAS, u0=ox, uw=0.5)
        Xl, Yl = canvas_xy(Pg_c, TABARD_CANVAS[0])
        Ul, Vl = lining_uv(Xl, Yl, (0.16, 0.58, 0.14), *TABARD_CANVAS, u0=ox, uw=0.5)
        if side > 0:
            Uc = Uc[:, ::-1]; Ul = Ul[:, ::-1]; Vl = Vl[:, ::-1]
        sp["outline_m"] = poly
        # lining: the inside faces sample a plain (emblem-free) patch of the same atlas: canvas x 0.16..0.30 m,
        # y 0.58..0.72 m below the top (between the belt zone and the hem fleur) -> one material per piece
        def lin(U, V, Ul=Ul, Vl=Vl):
            return (Ul, Vl)
        sheet(PfN, nu_, nv_, thick, slot_out="tabard", slot_in="tabard", slot_rim="tabard",
              uv_out=lambda U, V, Uc=Uc, Vc=Vc: (Uc, Vc), uv_in=lin, mesh=m)
        # weights: spine over the chest, the rim's joint over the band (and up to the rim), pelvis under the belt,
        # the flap chain below the pivot
        pre = "tabard_f_" if side < 0 else "tabard_b_"
        for i in range(n0, len(m.V)):
            x, y, z = m.V[i]
            m.tagw[i] = tabard_weights(z, x, pre, W, sp["z_hem"])
        sp["corners"] = {"l": Pg[-1, -1] if Pg[-1, -1][0] > 0 else Pg[-1, 0], "r": Pg[-1, 0] if Pg[-1, 0][0] < 0 else Pg[-1, -1]}
        info[name] = sp
    return m, info


def tabard_weights(z, x, pre, W, zhem):
    """authored tabard weights: chest on the spine (outfit time: coskin copies the cuirass above the belt), the band
    under the waist belt and the rise to the rim on RIM_BONE (coskin 'waist' substitutes the live rim joints), the
    cloth between the belt and the flap pivot blends to the pelvis, the flap chain below the pivot"""
    w = {}
    z30, z17 = Z(1.30), Z(1.17)
    wb, wt = W["bot"], W["top"]
    if z > z30:
        w = {"spine_05": 1.0}
        if abs(x) > 0.09 * XS(1.30) and z > Z(1.42):
            w = {"spine_05": 0.7, ("clavicle_l" if x > 0 else "clavicle_r"): 0.3}
    elif z > z17:
        t = (z - z17) / (z30 - z17); w = {"spine_05": t, "spine_04": 1 - t}
    elif z > wt + 0.03:
        t = (z - (wt + 0.03)) / (z17 - wt - 0.03); w = {"spine_04": t * 0.6, RIM_BONE: 1.0 - 0.6 * t}
    else:
        # (2b session 3) the band, the cloth over the cuirass rim and down to the flap pivot (under the rim) ride the
        # rim with the belt; the flap chain (rooted on the rim's joint) below the pivot
        zp = tabard_pivot()
        if z >= zp:
            w = {RIM_BONE: 1.0}
        else:
            # chain below the pivot: bone k spans [z_k, z_{k+1}], linear blend between segment centres
            zs = [zp, zp - (zp - zhem) * 0.34, zp - (zp - zhem) * 0.68, zhem - 0.02]
            s = np.interp(-z, [-zz for zz in zs], [0, 1, 2, 3])
            w = {RIM_BONE: max(0.0, 1 - s / 0.5) * 0.5}
            for k in range(3):
                wk = max(0.0, 1 - abs(s - (k + 0.5)))
                if s > 2.5 and k == 2:
                    wk = 1.0
                if wk > 0:
                    w[pre + "%02d" % (k + 1)] = wk
    tot = sum(w.values()) or 1.0
    return {k: v / tot for k, v in w.items() if v > 1e-4}


CAPE_TOP = 1.49
CAPE_HW = (0.222, 0.165, 1.7)     # half-width under the pauldrons, extra at the hem, how fast it widens below them
# mantle (user 15 / judge M8): the cape's top is a collar lying on the gorget's lowest back lame; towards the shoulders
# it rises over the trapezius and wraps over the BACK of each pauldron, where a gold clasp (piece 'clasps') pins it.
# Frame: the X axis through CAPE_OC; alpha = 0 points straight back (+Y), 90 deg straight up.
CAPE_OC = (0.0, 0.0, 1.385)
CAPE_WRAP_X = (0.115, 0.200)      # |x| band over which the top edge climbs from the gorget collar onto the pauldron
CAPE_TOP_ALPHA = 80.0             # deg: how far the corners wrap over the pauldron top
CAPE_W_TOP = 0.240                # half width of the top edge (over the pauldron backs)
CAPE_ALPHA_BOT = -24.0            # deg: where the mantle meets the hanging part (below the shoulder blades)
CAPE_THICK = 0.0070                # (iteration 2b: 7 mm, the sheet below uses the same above the hem)
CAPE_CLR = 0.0050                 # lining over the armour in the mantle
CLASP_X, CLASP_DALPHA, CLASP_R = 0.200, 12.0, 0.0265     # clasp centre |x|, below the top edge (deg), radius


def cape_hw(z, z_top, z_hem):
    """Cape half-width at height z: stays inside the arms down to the elbows, then flares to ~0.77 m at the hem
    (assembly pass vs the reference: the old 0.40 -> 0.55 m cape read as a narrow banner from the back). Scaled by the
    shoulder / back width only (judge M8: not by the waist or hips, the female cape pinched at the waist)."""
    t = np.clip((z_top - np.asarray(z, float)) / (z_top - z_hem), 0.0, 1.0)
    k = XS(1.30)
    return (CAPE_HW[0] + CAPE_HW[1] * (1.0 - (1.0 - t) ** CAPE_HW[2]) * smoothstep(0.0, 0.30, t)) * k


def cape_top(B):
    """Top of the cape at the spine (the collar's edge on the gorget's lowest back lame, 1.6 cm above the lame's lower
    edge; the tabard's back panel ends under that lame)."""
    g = getattr(B, "upper", {}).get("gorget")
    if g is None:
        return Z(CAPE_TOP)
    G = np.array(g.V)
    bk = G[(np.abs(G[:, 0]) < 0.03) & (G[:, 1] > 0)]
    return float(bk[:, 2].min()) + 0.016 if len(bk) else Z(CAPE_TOP)


class Mantle:
    """Envelope R(x, alpha) of the upper back, shoulders, gorget and pauldrons around the X axis through CAPE_OC
    (rays from outside towards the axis, in planes x = const), upper-envelope smoothed so the cloth bridges rims,
    rivets and the gap between the gorget and the pauldrons instead of dipping into them."""

    def __init__(self, B, layers, O=CAPE_OC, als=(-40.0, 112.0, 103), xs=(-0.36, 0.36, 145), win=3):
        trunk = BodySurf(B.bm, exclude_arms=True)
        co = trunk.co
        trunk.polys = [p for p in trunk.polys if not (co[list(p)][:, 2].mean() > Z(1.64))]   # no head (the gorget covers the neck)
        comp = Composite(trunk, layers)
        self.O = np.array(O, float)
        self.xs = np.linspace(*xs)
        self.als = np.radians(np.linspace(*als))
        R = np.full((len(self.xs), len(self.als)), np.nan)
        R0 = 0.75
        for i, x in enumerate(self.xs):
            for j, a in enumerate(self.als):
                d = np.array([0.0, math.cos(a), math.sin(a)])
                h = comp.ray(self.O + np.array([x, 0, 0]) + d * R0, -d, R0)
                if h is not None:
                    R[i, j] = R0 - h
        for i in range(len(self.xs)):                   # misses (beside the body): nearest valid alpha
            ok = ~np.isnan(R[i])
            if ok.any() and not ok.all():
                idx = np.arange(len(self.als)); R[i, ~ok] = np.interp(idx[~ok], idx[ok], R[i, ok])
        for j in range(len(self.als)):
            ok = ~np.isnan(R[:, j])
            if ok.any() and not ok.all():
                idx = np.arange(len(self.xs)); R[~ok, j] = np.interp(idx[~ok], idx[ok], R[ok, j])
        R = np.nan_to_num(R, nan=0.15)
        R = gauss1d(gauss1d(_winfilter(R, win, np.max), 1.0, 0), 1.0, 1)
        self.raw = R
        self.R = Grid2(self.xs[0], self.xs[-1], self.als[0], self.als[-1], R)

    def point(self, x, a, off=0.0):
        x = np.asarray(x, float); a = np.asarray(a, float)
        r = self.R(x, a) + off
        return np.stack([x, self.O[1] + r * np.cos(a), self.O[2] + r * np.sin(a)], -1)

    def alpha_at_z(self, x, z, off=0.0, rng=(-30.0, 100.0)):
        """alpha of the envelope point at height z (per x), scanning alpha from rng[0] towards rng[1]"""
        a = np.radians(np.linspace(rng[0], rng[1], 261))
        out = []
        for xx, zz in zip(np.atleast_1d(x), np.atleast_1d(z)):
            pz = self.point(np.full_like(a, xx), a, off)[:, 2]
            k = np.nonzero(pz >= zz)[0]
            out.append(a[k[0]] if len(k) else a[-1])
        return np.array(out)


def gorget_back_edge(B):
    """lower edge height of the gorget's lowest lame at the back, as a function of |x| (None without a gorget)"""
    g = getattr(B, "upper", {}).get("gorget")
    if g is None:
        return None
    G = np.array(g.V)
    xs = np.linspace(0.0, 0.20, 21); zs = []
    for x in xs:
        s = G[(np.abs(np.abs(G[:, 0]) - x) < 0.012) & (G[:, 1] > 0.02)]
        zs.append(s[:, 2].min() if len(s) else np.nan)
    zs = np.array(zs); ok = ~np.isnan(zs)
    zs = np.interp(xs, xs[ok], zs[ok])
    # a straight collar edge: over the gorget's scalloped lame edge (running max over +-2 cm, then smoothed)
    zs = np.maximum.reduce([np.r_[zs[k:], [zs[-1]] * k] if k >= 0 else np.r_[[zs[0]] * -k, zs[:k]] for k in (-2, -1, 0, 1, 2)])
    zs = np.convolve(np.pad(zs, 2, mode="edge"), np.ones(5) / 5, mode="valid")
    return lambda x: np.interp(np.abs(np.asarray(x, float)), xs, zs)


def cape(B, H, D, over_cuirass=False):
    """Long cape: a MANTLE (collar on the gorget's lowest back lame, rising over the trapezius and wrapping over the
    backs of the pauldrons, pinned there by the clasps) and a hanging part from below the shoulder blades to the
    ankles, flaring below the elbows, chevron hem, deep folds growing towards the hem, standing off the calves, the
    hem corners wrapping forward; lining. Returns (Mesh, info)."""
    m = Mesh()
    z_top, z_hem = cape_top(B), B.knee[2] - CAPE_HEM
    thick = CAPE_THICK
    layers = [v for k, v in getattr(B, "upper", {}).items() if k in ("cuirass", "mail", "gorget", "pauldron_l", "pauldron_r")]
    layers += [x for x in getattr(B, "cape_under", [])]
    MT = Mantle(B, layers, O=(0.0, CAPE_OC[1], Z(CAPE_OC[2])))
    gz = gorget_back_edge(B)
    a_bot = math.radians(CAPE_ALPHA_BOT)
    z_join = float(MT.point(0.0, a_bot, CAPE_CLR + thick)[2])

    def clr(z):
        if over_cuirass:
            return 0.015 + 0.008 * smoothstep(Z(1.45), Z(1.1), z)
        return 0.040 + 0.012 * smoothstep(Z(1.45), Z(1.1), z)

    def hw(z):
        return cape_hw(z, z_top, z_hem)

    kx = XS(1.30)

    def alpha_top(xt):
        xt = np.asarray(xt, float)
        if gz is not None:
            a_col = MT.alpha_at_z(xt, gz(xt) + 0.016, CAPE_CLR + thick)
        else:
            a_col = MT.alpha_at_z(xt, np.full_like(xt, z_top), CAPE_CLR + thick)
        w = smoothstep(CAPE_WRAP_X[0] * kx, CAPE_WRAP_X[1] * kx, np.abs(xt))
        return a_col + (math.radians(CAPE_TOP_ALPHA) - a_col) * w

    def P_hang(U, z):
        """the hanging part (below the mantle): lies over the back plate, hangs straight below the buttocks"""
        uu = (U - 0.5) * 2
        z = z + 0.085 * np.abs(uu) ** 1.15 * smoothstep(z_join - 0.25, z_hem, z)   # chevron hem (centre lowest)
        x = uu * hw(z)
        xc = np.clip(x, -0.30, 0.30)
        zt = z_join + 0.05
        y = D.hang(xc, np.minimum(z, zt), zt, clr, +1)
        yb = np.maximum.reduce([D.hang(np.clip(xc + dx, -0.30, 0.30), np.minimum(z, zt), zt, clr, +1)
                                for dx in (-0.08, -0.04, 0.04, 0.08)])
        tb = smoothstep(Z(1.20), z_join, z)
        y = y + np.maximum(0.0, yb - y) * tb
        below = smoothstep(Z(1.36), z_hem, z)
        y = y + 0.040 * below ** 1.2                          # hangs away from the buttocks / calves
        y = y - 0.085 * below ** 1.4 * np.abs(uu) ** 2.2      # lower sides wrap forward around the legs
        # deep vertical folds (7 across, phase drifting down the length), a few broad + finer ones; they start at the
        # shoulder blades (gathered by the mantle) and deepen towards the weighted hem
        fold = 0.012 * smoothstep(z_join + 0.02, z_join - 0.15, z) + 0.040 * smoothstep(Z(1.30), z_hem + 0.25, z) ** 0.9
        ph = uu * math.pi * 3.5 + 0.4 + 0.7 * below
        # one-sided: the valleys touch the hang surface, the ridges stand out (a valley must never dip into the tabard
        # flap / legs under the cape); rounded ridges, tighter valleys
        wave = 0.78 * (0.5 + 0.5 * np.sin(ph)) ** 0.8 + 0.22 * (0.5 + 0.5 * np.sin(uu * math.pi * 8.3 + 1.3))
        y = y + fold * 1.25 * wave
        x = x + 0.012 * below * np.cos(ph)
        return np.stack([x, y, z], -1)

    NV_H, NV_M = 20, 9                                         # rows: hanging part, mantle
    nu_ = 18

    def Pf(U, V):
        U = np.asarray(U, float); V = np.asarray(V, float)
        uu = (U - 0.5) * 2
        vh = NV_H / (NV_H + NV_M)
        out = np.zeros(U.shape + (3,))
        lo = V <= vh
        # hanging part: V 0..vh = hem .. join
        tl = np.clip(V / vh, 0, 1)
        zc = z_hem + (z_join - z_hem) * tl
        Ph = P_hang(U, zc)
        # mantle: t 0..1 = join .. top edge
        t = np.clip((V - vh) / (1 - vh), 0, 1)
        xt = uu * CAPE_W_TOP * kx
        at = alpha_top(xt.ravel()).reshape(xt.shape)
        a = a_bot + (at - a_bot) * t
        wj = hw(z_join)
        x = uu * (wj + (CAPE_W_TOP * kx - wj) * smoothstep(0.0, 0.8, t))
        Pm = MT.point(x, a, CAPE_CLR + thick)
        # blend the first mantle rows from the hanging part's top row (continuity at the join)
        Pj = P_hang(U, np.full_like(U, z_join))
        bj = smoothstep(0.0, 0.45, t)[..., None]
        Pm = Pj + (Pm - Pj) * bj
        out = np.where(lo[..., None], Ph, Pm)
        return out
    e = 1e-3

    def PfN(U, V):
        P = Pf(U, V)
        Pu = Pf(np.clip(U + e, 0, 1), V) - Pf(np.clip(U - e, 0, 1), V)
        Pv = Pf(U, np.clip(V + e, 0, 1)) - Pf(U, np.clip(V - e, 0, 1))
        N = np.cross(Pu, Pv); N /= np.maximum(np.linalg.norm(N, axis=-1, keepdims=True), 1e-12)
        # outward = away from the body: +Y on the hanging part, away from the mantle axis above
        ref = P - MT.O * np.array([0, 1, 1])
        ref[..., 0] = 0.0
        ref = np.where((P[..., 2] < z_join)[..., None], np.array([0.0, 1.0, 0.0]), ref)
        s = np.sign((N * ref).sum(-1)); s[s == 0] = 1
        return P, N * s[..., None]
    n0 = len(m.V)
    nv_ = NV_H + NV_M
    Ug, Vg = np.meshgrid(np.linspace(0, 1, nu_ + 1), np.linspace(0, 1, nv_ + 1))
    Pg = Pf(Ug, Vg)
    Uc, Vc, poly = canvas_uv(Pg[:, ::-1], *CAPE_CANVAS)     # seen from behind: canvas left = the wearer's left
    Uc = Uc[:, ::-1]
    # lining: plain patch of the cape atlas between the emblem and the hem fleur (canvas x 0.36..0.66, y 0.74..1.04 m),
    # one scale in both directions (2b, G6)
    Xl, Yl = canvas_xy(Pg[:, ::-1], CAPE_CANVAS[0])
    Ul, Vl = lining_uv(Xl, Yl, (0.36, 0.74, 0.30), *CAPE_CANVAS)
    Ul = Ul[:, ::-1]; Vl = Vl[:, ::-1]
    lin = lambda U, V: (Ul, Vl)
    # (iteration 2b, G2 spikes: a weighted, thicker hem; 7 mm cloth + lining above it)
    thick_v = lambda U, V: 0.0070 + 0.0016 * (1 - smoothstep(0.0, 0.25, V))
    _, io, ii = sheet(PfN, nu_, nv_, thick_v, slot_out="cape", slot_in="cape", slot_rim="cape",
                      uv_out=lambda U, V: (Uc, Vc), uv_in=lin, mesh=m)
    Vrow = np.linspace(0, 1, nv_ + 1)
    tm = np.clip((Vrow - NV_H / nv_) / (1 - NV_H / nv_), 0, 1)
    info = dict(z_top=z_top, z_hem=z_hem, z_join=z_join, hw=hw, outline_m=poly, grid_out=io, grid_in=ii, nu=nu_, mesh=m,
                nv=nv_, nv_hang=NV_H, mantle=MT, alpha_top=alpha_top, t_row=tm)
    for j in range(nv_ + 1):
        for i in range(nu_ + 1):
            for vi in (io[j, i], ii[j, i]):
                x, y, z = m.V[vi]
                m.tagw[vi] = cape_weights(x, z, z_top, z_hem, hw, t_mantle=float(tm[j]), uu=(i / nu_ - 0.5) * 2)
    return m, info


CAPE_ROWS = 4


def cape_joint_z(z_top, z_hem):
    """heights of the cape chain joints (bone heads) and the chain end"""
    a = z_top - 0.10
    return [a + (z_hem - 0.02 - a) * k / CAPE_ROWS for k in range(CAPE_ROWS + 1)]


def pauldron_joint(side):
    """joint that carries the pauldron cop (armour_upper_rig / rig_helpers.rule_pauldron); the cape corners and the
    clasps ride it (outfit time: armour_lower.coskin copies the actual pauldron's joint)"""
    return "upperarm_helper_01_" + side


def cape_weights(x, z, z_top, z_hem, hw, t_mantle=0.0, uu=None):
    js = cape_joint_z(z_top, z_hem)
    xs = x / max(float(hw(z)), 0.05) if uu is None else uu
    cols = {"l": max(0.0, 1 - abs(xs - 0.62) / 0.62), "c": max(0.0, 1 - abs(xs) / 0.62), "r": max(0.0, 1 - abs(xs + 0.62) / 0.62)}
    if xs > 0.62: cols["l"] = 1.0
    if xs < -0.62: cols["r"] = 1.0
    tot = sum(cols.values()); cols = {k: v / tot for k, v in cols.items() if v > 1e-3}
    # attachment: the mantle rides spine_05 (collar on the gorget, over the back plate) and, where it lies on a
    # pauldron's back, that pauldron's joint; below the mantle the chains take over
    att = float(max(smoothstep(js[0] - 0.02, js[0] + 0.07, z), smoothstep(0.30, 0.70, t_mantle)))
    s = float(np.interp(-z, [-q for q in js], list(range(CAPE_ROWS + 1))))
    w = {}
    for c, cw in cols.items():
        for k in range(CAPE_ROWS):
            wk = max(0.0, 1 - abs(s - (k + 0.5))) if s > 0.5 or k > 0 else 1.0
            if k == CAPE_ROWS - 1 and s > CAPE_ROWS - 0.5:
                wk = 1.0
            if wk > 1e-3:
                w["cape_%s_%02d" % (c, k + 1)] = w.get("cape_%s_%02d" % (c, k + 1), 0) + wk * cw * (1 - att)
    if att > 0:
        kx = XS(1.30)
        pj = float(smoothstep((CAPE_WRAP_X[0] + 0.02) * kx, (CAPE_WRAP_X[1] - 0.005) * kx, abs(x))) * float(smoothstep(0.35, 0.75, t_mantle))
        w["spine_05"] = att * (1 - pj)
        if pj > 1e-3:
            w[pauldron_joint("l" if x > 0 else "r")] = att * pj
    tot = sum(w.values()) or 1.0
    return {k: v / tot for k, v in w.items() if v / tot > 1e-3}


def clasps(B, cinfo):
    """Two gold clasps (round bosses with the armour sheet's rosette, beaded rims) pinning the cape's mantle to the
    pauldron backs; rigid on the pauldron joint. Returns a Mesh (slot 'armour', UVs on the armour trim sheet)."""
    MT = cinfo["mantle"]
    TX = sheet_layout()
    m = Mesh()
    for side, sx in (("l", 1.0), ("r", -1.0)):
        xt = sx * CLASP_X * XS(1.30)
        at = float(cinfo["alpha_top"](np.array([xt]))[0]) - math.radians(CLASP_DALPHA)
        # a narrow band through the centre: the cape surface normal from the mantle field
        e = 0.004
        P0 = MT.point(xt, at, CAPE_CLR + CAPE_THICK)
        Px = MT.point(xt + e, at, CAPE_CLR + CAPE_THICK) - MT.point(xt - e, at, CAPE_CLR + CAPE_THICK)
        Pa = MT.point(xt, at + 0.02, CAPE_CLR + CAPE_THICK) - MT.point(xt, at - 0.02, CAPE_CLR + CAPE_THICK)
        N = nrm(np.cross(Px, Pa))
        if N.dot(np.array([0.0, math.cos(at), math.sin(at)])) < 0:
            N = -N
        # the cape mesh is coarser than the field: lift the seat over every cape vertex under the clasp
        cm_ = cinfo.get("mesh")
        if cm_ is not None:
            Cv = np.array(cm_.V)[np.asarray(cinfo["grid_out"]).ravel()]
            rel = Cv - P0
            lat = np.linalg.norm(rel - np.outer(rel @ N, N), axis=1)
            near = lat < CLASP_R * 1.6
            if near.any():
                P0 = P0 + N * max(0.0, float((rel[near] @ N).max()) + 0.0008)
        # boss profile (r, h) along N from the cloth surface: seat, beaded rim, domed centre
        R_ = CLASP_R
        # (a closed solid from pole to pole: lathe() winds it outward; user item 28 / G1: the clasps were inside-out)
        prof = [(0.0, 0.0005), (R_ * 0.55, 0.0005), (R_ * 0.92, 0.0008), (R_ * 1.0, 0.0028), (R_ * 0.96, 0.0048),
                (R_ * 0.86, 0.0052), (R_ * 0.78, 0.0060), (R_ * 0.55, 0.0085), (R_ * 0.28, 0.0100), (0.0, 0.0104)]
        ref = nrm(np.cross(N, np.array([1.0, 0.0, 0.0])) if abs(N[0]) < 0.9 else np.cross(N, np.array([0, 0, 1.0])))
        n0 = len(m.V)
        mm, idx = lathe(prof, P0, N, ref, 32, mat=SLOT["armour"], mesh=m)
        # UVs: face (h > rim) planar into the rosette square, the seat under it planar into the dark square, the rim
        # into the gold bead band by profile arc length (iteration 2b, G6: the rim was mapped by height only, so the
        # flat seat collapsed to a line and the bead streaked 3-25x round the boss): u = angle x radius, both at the
        # band's own scale (isotropic texels)
        X_ = ref; Y_ = np.cross(N, X_)
        pr_ = np.array(prof, float)
        s_ = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(pr_, axis=0), axis=1))])
        k_rim = [k for k in range(len(prof)) if 0.54 * R_ <= pr_[k, 0] and pr_[k, 1] <= 0.0052]
        s0_, s1_ = s_[min(k_rim)], s_[max(k_rim)]
        kv = (TX.band_v("gold_bead", 0.0) - TX.band_v("gold_bead", 1.0)) / max(s1_ - s0_, 1e-6) if TX is not None else 1.0

        def s_of(p):
            q = p - P0; h = float(q.dot(N)); r = float(np.linalg.norm(q - N * h))
            d = np.hypot(pr_[:, 0] - r, pr_[:, 1] - h)
            k = int(np.argmin(d))
            return s_[k], r
        nf0 = len(m.F) - 32 * (len(prof) - 1)
        for fi in range(nf0, len(m.F)):
            vs = m.F[fi]
            pts = [np.array(m.V[v]) for v in vs]
            hs = [float((p - P0).dot(N)) for p in pts]
            if TX is not None and min(hs) >= 0.0050:
                uv = [TX.square_uv("boss", float((p - P0).dot(X_)), float((p - P0).dot(Y_)), size=2 * R_ * 0.9) for p in pts]
            elif TX is not None and max(hs) <= 0.00051:
                uv = [TX.square_uv("dark_sq", float((p - P0).dot(X_)), float((p - P0).dot(Y_)), size=2 * R_ * 1.1) for p in pts]
            elif TX is not None:
                cm = np.mean(pts, 0) - P0
                a_m = math.atan2(cm.dot(Y_), cm.dot(X_))
                uv = []
                for p in pts:
                    q = p - P0
                    a = math.atan2(q.dot(Y_), q.dot(X_)); a = a_m + ((a - a_m + math.pi) % (2 * math.pi) - math.pi)
                    sv, r = s_of(p)
                    uv.append((a * r * kv, TX.band_v("gold_bead", float(np.clip((sv - s0_) / max(s1_ - s0_, 1e-6), 0, 1)))))
            else:
                uv = [(float((p - P0).dot(X_)), float((p - P0).dot(Y_))) for p in pts]
            m.UV[fi] = uv
            if TX is None:
                m.M[fi] = SLOT["gold"]
        for i in range(n0, len(m.V)):
            m.tagw[i] = {pauldron_joint(side): 1.0}
    return m


# ------------------------------------------------------------------------------------------------ assets (MakeClothes)
MANIFEST = os.path.join(TEX, "textures_char.json")
# texture sets per slot: shared tileables from scripts/textures_char.py (materials agent), own strips / atlases from
# scripts/armour_lower_tex.py (assets/textures/knight_lower/). uv: 'metric' (x uv_per_m), 'strip' (u metric, v 0..1),
# 'atlas' (0..1, region rect)
SLOT_TEX = {
    "steel": ("steel_worn", "metric"), "gold": ("gold_worn", "metric"), "leather": ("leather_brown", "metric"),
    "mail": ("mail_riveted", "metric"), "cloth": ("cloth_blue", "metric"), "lining": ("steel_worn", "metric"),
    "cape_lining": ("cloth_blue", "metric"), "trim": ("kl_trim", "strip"), "strap": ("kl_strap", "strip"),
    "tabard": ("kl_tabard", "atlas"), "cape": ("kl_cape", "atlas"), "wood": ("leather_brown", "metric"),
    "blade": ("steel_worn", "metric"), "grip": ("cloth_blue", "metric"), "enamel": ("flat", "atlas"),
    "shield": ("shield_lion", "atlas"), "rim": ("gold_worn", "metric"), "armour": ("knight_armour", "atlas"),
}
STRIP_M = {"kl_trim": 0.20, "kl_strap": 0.40}            # metres per 1.0 of u along the strip textures


def uv_per_m(set_name):
    try:
        return json.load(open(MANIFEST))["sets"][set_name]["uv_per_m"] or 1.0
    except Exception:
        return {"steel_worn": 2.0, "gold_worn": 2.0, "leather_brown": 2.0, "mail_riveted": 5.95, "cloth_blue": 4.0}.get(set_name, 1.0)


# Plate pieces share the upper armour's trim sheet (scripts/armour_upper_tex.py layout: steel field, filigree / bead /
# plain-gold strips, leather square): one material per piece (M_knight_armour), identical steel and gold on the whole
# knight. Falls back to the separate slot materials when the upper kit is not present.
SHEET_PIECES = ("cuisses", "poleyns", "greaves", "sabatons", "tassets", "mail_skirt", "belts")
SHEET_SLOTS = ("steel", "gold", "trim", "rim", "leather")
TRIM_W = {"mail_skirt": 0.022}                           # physical trim band width (m), default 0.012


def sheet_layout():
    try:
        import armour_upper_tex as TX
        return TX if os.path.exists(TX.FILES["armour_base"]) else None
    except Exception:
        return None


def gold_face_uv(TX, V, f, uv):
    """(iteration 2b, integrity G6: 'gold' faces were mapped u x 2 per metre but v / 0.02 m into the 42-px plain-gold
    band, 3-10x anisotropic) one face into the gold_plain band with ISOTROPIC texels: the face projected on its own
    frame (u along the face's authored u direction), native 2 UV / m, reduced for faces taller than the band"""
    P = np.array([V[i] for i in f], float)
    U = np.array(uv, float)
    c = P.mean(0)
    n = np.cross(P[1] - P[0], P[-1] - P[0])
    n = n / max(np.linalg.norm(n), 1e-12)
    # the face's u direction in 3D: least squares of P ~ c + dP/du * (u - u_mean) (+ dP/dv (v - v_mean))
    du = U - U.mean(0)
    try:
        J = np.linalg.lstsq(du, P - c, rcond=None)[0]           # (2, 3)
        e1 = J[0] - n * J[0].dot(n)
    except Exception:
        e1 = P[1] - P[0]
    if np.linalg.norm(e1) < 1e-9:
        e1 = P[1] - P[0]
    e1 = e1 / max(np.linalg.norm(e1), 1e-12); e2 = np.cross(n, e1)
    s_ = (P - c) @ e1; t_ = (P - c) @ e2
    top, bot = TX.band_v("gold_plain", 0.0), TX.band_v("gold_plain", 1.0)
    H = (top - bot) * 0.9
    k = min(2.0, H / max(float(t_.max() - t_.min()), 1e-6))
    u0 = float(U[:, 0].mean()) * k                              # keeps neighbouring faces roughly continuous along u
    vm = 0.5 * (top + bot)
    return [(u0 + float(a) * k, vm + float(b) * k) for a, b in zip(s_, t_)]


def to_sheet(name, m):
    """Copy of m with steel / gold / trim / rim (/ pouch leather on the belts) faces moved onto the armour trim sheet."""
    TX = sheet_layout()
    if TX is None or name not in SHEET_PIECES:
        return m
    out = Mesh(); out.V, out.F, out.tag, out.tagw = m.V, m.F, m.tag, m.tagw
    out.UV, out.M = [], []
    nat = lambda b: TX.band_px(b) / TX.TRIM_DENS                  # native strip width (m)
    kt = 2.0 * nat("fil_narrow") / TRIM_W.get(name, 0.012)        # u per metre, keeps the filigree aspect
    kr = 2.0 * nat("gold_bead") / (math.pi * 0.003)
    for uv, sl in zip(m.UV, m.M):
        slot = SLOTS[sl]
        if slot not in SHEET_SLOTS or (slot == "leather" and name != "belts"):
            out.UV.append(uv); out.M.append(sl); continue
        U = np.array([a for a, b in uv]); V = np.array([b for a, b in uv])
        if slot == "steel":
            q = TX.steel_uv(U, V); new = [tuple(x) for x in q]
        elif slot == "trim":
            new = [(u * kt, TX.band_v("fil_narrow", 0.04 + 0.92 * float(np.clip(v, 0, 1)))) for u, v in uv]
        elif slot == "rim":
            new = [(u * kr, TX.band_v("gold_bead", 0.05 + 0.9 * float(np.clip(v / (math.pi * 0.003), 0, 1)))) for u, v in uv]
        elif slot == "gold":
            new = gold_face_uv(TX, out.V, out.F[len(out.UV)], uv)
        else:                                                      # pouch / tongue leather -> leather square
            new = [TX.square_uv("leather", u - 0.25, v - 0.25, size=0.5) for u, v in uv]
        out.UV.append(new); out.M.append(SLOT["armour"])
    return out


def final_uvs(m):
    """metric UVs -> texture UVs per slot (tileables x uv_per_m, strips u / strip length)"""
    out = []
    for uv, slot in zip(m.UV, m.M):
        name, kind = SLOT_TEX[SLOTS[slot]]
        if kind == "metric":
            k = uv_per_m(name); out.append([(u * k, v * k) for u, v in uv])
        elif kind == "strip":
            L = STRIP_M[name]; out.append([(u / L, v) for u, v in uv])
        else:
            out.append(list(uv))
    return out


LOWER_BONES = ("pelvis", "spine_01", "thigh_l", "thigh_r", "thigh_twist_01_l", "thigh_twist_01_r", "calf_l", "calf_r",
               "calf_twist_01_l", "calf_twist_01_r", "foot_l", "foot_r", "ball_l", "ball_r")


def covered_body_verts(B, name):
    """basemesh vertices a piece hides (MakeHuman delete_verts): the skin it fully covers"""
    s = B.s; co = s.co[:s.NBODY]; dom = s.dom[:s.NBODY]
    kz = B.knee[2]
    low = np.array([d in LOWER_BONES for d in dom]) & ~s.arm[:s.NBODY]
    z = co[:, 2]
    if name == "legs_mail":
        sel = low & (z > kz - 0.11) & (z < 0.975)
    elif name == "boots":
        sel = low & (z < 0.160)
    elif name == "greaves":
        sel = low & (z >= 0.150) & (z <= kz - 0.10)
    else:
        return []
    return list(np.nonzero(sel)[0])


HANG_ON_SKIRT = {"cape": 1.02, "tabard": 0.97}          # z below which a piece is matched to helper-skirt
Z_DEPTH = {"legs_mail": 32, "boots": 36, "greaves": 40, "sabatons": 42, "cuisses": 40, "poleyns": 44, "mail_skirt": 50,
           "tassets": 54, "tabard": 60, "belts": 66, "cape": 70, "clasps": 74}


def _jsonable(o):
    if hasattr(o, "tolist"):
        return o.tolist()
    if hasattr(o, "__float__"):
        return float(o)
    raise TypeError(type(o))


def match_clothes(B, clothes, props, delete_group=None):
    """MakeClothes matching (ClothesService.create_mhclo_from_clothes_matching, allow_exact=False: every authored offset
    kept) with the basemesh cross-reference built once per session instead of once per piece (~10x faster)."""
    from bl_ext.user_default.mpfb.services import ClothesService, MeshService
    from bl_ext.user_default.mpfb.entities.clothes.mhclo import Mhclo
    from bl_ext.user_default.mpfb.entities.clothes.vertexmatch import VertexMatch
    from bl_ext.user_default.mpfb.entities.meshcrossref import MeshCrossRef
    from bl_ext.user_default.mpfb.entities.objectproperties import GeneralObjectProperties
    bm = B.bm
    if getattr(B, "_xref", None) is None:
        B._xref = MeshCrossRef(bm, after_modifiers=True, build_faces_by_group_reference=True)
        B._refscale = ClothesService.get_reference_scale(bm)
    mh = Mhclo(); mh.verts = dict(); mh.clothes = clothes
    for k, v in props.items():
        if hasattr(mh, k):
            setattr(mh, k, v)
    cxr = MeshCrossRef(clothes, after_modifiers=True, build_faces_by_group_reference=True)
    sf = GeneralObjectProperties.get_value("scale_factor", entity_reference=bm)
    mh.max_pole = max(len(e) for e in cxr.edges_by_vertex)
    for vi in range(len(cxr.vertex_coordinates)):
        vm = VertexMatch(clothes, vi, cxr, bm, B._xref, scale_factor=sf, reference_scale=B._refscale, allow_exact=False)
        mh.verts[vi] = vm.mhclo_line
    if delete_group and delete_group in bm.vertex_groups:
        mh.delete_group = delete_group
        mh.delverts = sorted(set(int(v[0]) for v in MeshService.find_vertices_in_vertex_group(bm, delete_group)))
        mh.delete = True
    return mh


def write_piece(B, name, m, extra=None):
    """Mesh -> MPFB clothes asset rts_knight_<name> (MakeClothes matching against the live authoring body)."""
    import uuid as _uuid
    from bl_ext.user_default.mpfb.services import ClothesService
    rig, bm = B.rig, B.bm
    if m.mixed():
        m.triangulate()
    m = to_sheet(name, m)
    asset = "rts_knight_" + name
    d = os.path.join(KL_ASSETS, asset); os.makedirs(d, exist_ok=True)
    uvs = final_uvs(m)
    mm = Mesh(); mm.V, mm.F, mm.UV, mm.M, mm.tag, mm.tagw = m.V, m.F, uvs, m.M, m.tag, m.tagw
    ob = mm.to_object(asset)
    ob.data.polygons.foreach_set("material_index", np.zeros(len(ob.data.polygons), dtype=np.int32))
    # hanging cloth (cape, tabard flaps below the belt) is matched to MakeHuman's smooth skirt helper instead of the
    # skin, so on other bodies it follows the hips smoothly instead of printing the glutes / calves through the cloth
    hang = [i for i, v in enumerate(mm.V) if name in HANG_ON_SKIRT and v[2] < HANG_ON_SKIRT[name]]
    rest = sorted(set(range(len(mm.V))) - set(hang))
    vg = ob.vertex_groups.new(name="body"); vg.add(rest, 1.0, 'REPLACE')
    if hang:
        vh = ob.vertex_groups.new(name="helper-skirt"); vh.add(hang, 1.0, 'REPLACE')
    chk = ClothesService.mesh_is_valid_as_clothes(ob, bm)
    assert chk["all_checks_ok"], (name, chk)
    dele = covered_body_verts(B, name)
    dg = None
    if dele:
        dg = "kl_del_" + name
        g = bm.vertex_groups.get(dg) or bm.vertex_groups.new(name=dg)
        g.add([int(i) for i in dele], 1.0, 'REPLACE')
    props = {"name": asset, "description": "RTS knight (lower armour): " + name,
             "author": "RTS project (procedural, scripts/armour_lower.py)", "license": "CC0", "homepage": "",
             "uuid": str(_uuid.uuid5(_uuid.NAMESPACE_URL, "rts-knight-" + name))}
    t0 = time.time()
    mh = match_clothes(B, ob, props, delete_group=dg)
    mh.material = asset + ".mhmat"
    path = os.path.join(d, asset + ".mhclo")
    mh.write_mhclo(path, reference_scale=ClothesService.get_reference_scale(bm), also_export_mhmat=False)
    txt = open(path).read().replace("verts 0", "z_depth %d\n\nverts 0" % Z_DEPTH.get(name, 50), 1)
    if "material " not in txt:
        txt = txt.replace("# Scale references:", "material %s.mhmat\n\n# Scale references:" % asset, 1)
    open(path, "w").write(txt)
    open(os.path.join(d, asset + ".mhmat"), "w").write(
        "# MakeHuman material (preview only; the RTS build assigns PBR slots from %s.rts.json)\nname %s\n"
        "diffuseColor 0.6 0.6 0.62\nshininess 0.5\nbackfaceCull False\n" % (asset, asset))
    # rigid / chain weights: every bone listed so MPFB replaces the interpolated groups
    if all(w for w in m.tagw):
        bones = [b.name for b in rig.data.bones] + sorted({k for w in m.tagw for k in w} - {b.name for b in rig.data.bones})
        W = {b: [] for b in bones}
        for i, w in enumerate(m.tagw):
            for k, v in w.items():
                W[k].append([i, round(float(v), 5)])
        json.dump({"name": asset + " weights", "license": "CC0", "version": 110, "weights": W},
                  open(os.path.join(d, asset + ".mhw"), "w"))
    elif os.path.exists(os.path.join(d, asset + ".mhw")):
        os.remove(os.path.join(d, asset + ".mhw"))
    meta = {"asset": asset, "slot": name, "slots": [SLOTS[i] for i in sorted(set(m.M))],
            "face_slot": [SLOTS[i] for i in m.M], "face_keys": [min(f) * 100000 + max(f) for f in m.F],
            "verts": len(m.V), "faces": len(m.F),
            "tris": sum(len(f) - 2 for f in m.F), "delete_verts": len(dele),
            "rigid_weights": bool(all(w for w in m.tagw)), "tex": {SLOTS[i]: SLOT_TEX[SLOTS[i]] for i in set(m.M)}}
    if extra:
        meta.update(extra)
    json.dump(meta, open(os.path.join(d, asset + ".rts.json"), "w"), default=_jsonable)
    dst = os.path.join(USER_DATA, "clothes", asset)
    if os.path.exists(dst):
        shutil.rmtree(dst)
    shutil.copytree(d, dst)
    bpy.data.objects.remove(ob, do_unlink=True)
    log("asset %-22s verts %5d tris %5d delete %5d rigid %s  (%.1fs)" % (asset, len(m.V), meta["tris"], len(dele),
                                                                    meta["rigid_weights"], time.time() - t0))
    return path


def chain_specs(m, bones, pts_fn, k=4):
    """extra bones located by vertex means (MPFB 'mean of vertices' strategy): for every bone, the k piece vertices
    nearest to its authored head and tail -> {bone: {parent, head: [idx], tail: [idx]}}"""
    V = np.array(m.V)
    out = {}
    for name, spec in bones.items():
        parent, head, tail = spec[:3]
        zhint = spec[3] if len(spec) > 3 else (0, 1, 0)          # (2b s3) the tasset joints' Z points outward
        hi = list(np.argsort(np.linalg.norm(V - head, axis=1))[:k])
        ti = list(np.argsort(np.linalg.norm(V - tail, axis=1))[:k])
        out[name] = {"parent": parent, "head": [int(i) for i in hi], "tail": [int(i) for i in ti],
                     "head_off": list(map(float, head - V[hi].mean(0))), "tail_off": list(map(float, tail - V[ti].mean(0))),
                     # canonical rest frame (M15): the AUTHORED direction, identical on every body the asset is fitted to
                     "frame": frame_yz(np.asarray(tail) - np.asarray(head), zhint),
                     "head_authored": list(map(float, head)), "tail_authored": list(map(float, tail))}
    return out


def author(rig, bm, only=None):
    pose_reset(rig)
    B = Body(rig, bm)
    pieces, H, D = build_all_pieces(B)
    bank = {n: m for n, m, k in pieces}
    extra = piece_extras(B, bank, H)
    for n, fl in getattr(B, "tights_src", {}).items():          # (2b) the helper faces, for every body's regen
        extra.setdefault(n, {})["tights_faces"] = fl
    paths = {}
    for n, m, k in pieces:
        if only and n not in only:
            continue
        paths[n] = write_piece(B, n, m, extra.get(n))
    json.dump({"pieces": [n for n, m, k in pieces], "order": [n for n, m, k in pieces]},
              open(os.path.join(KL_ASSETS, "rts_knight_lower.json"), "w"), indent=1)
    return paths


def piece_extras(B, bank, H):
    """per-piece sidecar data (.rts.json): extra bones (tabard / cape chains as vertex-mean anchors + authored frames),
    panel canvases / outlines for the texture painter, belt anchors"""
    extra = {}
    # tabard chains (front / back flap) at the panel centre line
    tb = bank["tabard"]; V = np.array(tb.V)
    tinfo, cinfo = B.tinfo, B.cinfo
    zb, specs = Z(1.015), {}
    for pre, side in (("tabard_f_", -1), ("tabard_b_", 1)):
        zhem = min(V[(V[:, 1] * side > 0) & (np.abs(V[:, 0]) < 0.02)][:, 2])
        zp = tabard_pivot()
        zs = [zp, zp - (zp - zhem) * 0.34, zp - (zp - zhem) * 0.68, zhem - 0.02]
        for k in range(3):
            def at(z):
                sel = (V[:, 1] * side > 0) & (np.abs(V[:, 0]) < 0.03)
                c = V[sel][np.argmin(np.abs(V[sel][:, 2] - z))]
                return np.array([0.0, c[1], z])
            # (2b session 3) the chain hangs from the cuirass rim's joint (the belted cloth over the rim rides it)
            specs[pre + "%02d" % (k + 1)] = (RIM_BONE if k == 0 else pre + "%02d" % k, at(zs[k]), at(zs[k + 1]))
    extra["tabard"] = {"bones": chain_specs(tb, specs, None), "canvas_m": TABARD_CANVAS,
                       "outline_m": {"front": tinfo["front"]["outline_m"], "back": tinfo["back"]["outline_m"]},
                       "panel": {k: {kk: vv for kk, vv in v.items() if kk != "outline_m"} for k, v in tinfo.items()}}
    # cape chains: three columns x CAPE_ROWS
    cp = bank["cape"]; V = np.array(cp.V)
    zt = max(V[:, 2]); zh = min(V[:, 2])
    ctop, chem = cinfo["z_top"], B.knee[2] - CAPE_HEM
    js = cape_joint_z(ctop, chem)
    hwf = lambda z: float(cape_hw(z, ctop, chem))
    specs = {}
    for col, xs in (("l", 0.62), ("c", 0.0), ("r", -0.62)):
        for k in range(CAPE_ROWS):
            def at(z, xs=xs):
                x = xs * hwf(z)
                c = V[np.argmin(np.linalg.norm(V[:, [0, 2]] - np.array([x, z]), axis=1))]
                return np.array([x, c[1], z])
            specs["cape_%s_%02d" % (col, k + 1)] = ("spine_05" if k == 0 else "cape_%s_%02d" % (col, k), at(js[k]), at(js[k + 1]))
    extra["cape"] = {"bones": chain_specs(cp, specs, None), "canvas_m": CAPE_CANVAS, "outline_m": cinfo["outline_m"]}
    # scabbard frog on the hip belt (left hip): vertex anchors for the socket
    bl = bank["belts"]; V = np.array(bl.V)
    ph = math.radians(100)
    frog = B.belt_paths["waist"].point(ph, BELT_CLR + BELT_THICK) if getattr(B, "belt_paths", None) else H.point(Z(1.0), ph, 0.004, field="R")
    extra["belts"] = {"anchors": {"scabbard_frog": [int(i) for i in np.argsort(np.linalg.norm(V - frog, axis=1))[:6]]}}
    return extra


# ------------------------------------------------------------------------------------------------ dress (load onto a live human)
STEEL_SET = "steel_worn"
# slot -> (texture set, material options); kl_* sets are this kit's own (assets/textures/knight_lower)
SLOT_MAT = {
    "steel": (STEEL_SET, {}), "blade": (STEEL_SET, {}), "gold": ("gold_worn", {}), "trim": ("kl_trim", {}),
    "leather": ("leather_brown", {}), "wood": ("leather_brown", {}), "strap": ("kl_strap", {}),
    "mail": ("kl_mail_opaque", {}), "cloth": ("cloth_blue", {}), "grip": ("cloth_blue", {}),
    "cape_lining": ("cloth_blue", {"tint": (0.72, 0.74, 0.82)}), "lining": (STEEL_SET, {"tint": (0.5, 0.5, 0.5)}),
    "tabard": ("kl_tabard", {}), "cape": ("kl_cape", {}), "shield": ("shield_lion", {}), "enamel": (None, {}),
    "rim": ("gold_worn", {}), "armour": ("knight_armour", {}),
}


def _cm():
    import char_materials as cm
    man = cm.manifest()
    klm = os.path.join(KL_TEX, "kl_textures.json")
    if os.path.exists(klm):
        for k, v in json.load(open(klm)).items():
            files = {kk: (vv if vv.startswith("..") else "knight_lower/" + vv) for kk, vv in v["files"].items()}
            files = {kk: (vv.replace("../", "") if vv.startswith("..") else vv) for kk, vv in files.items()}
            man[k] = dict(v, files=files, alpha=False)
    return cm


def slot_material(slot):
    if slot == "armour":                                   # the upper armour's trim-sheet material (shared)
        import armour_upper as AU
        m = bpy.data.materials.get("M_knight_armour") or AU.make_material("cuirass")
        m["rts_slot"] = "armour"
        return m
    name = "M_kl_" + slot
    m = bpy.data.materials.get(name)
    if m:
        return m
    set_name, opts = SLOT_MAT[slot]
    cm = _cm()
    if set_name is None or set_name not in cm.manifest():
        if set_name is not None:
            log("WARNING texture set %s missing, flat placeholder for slot %s" % (set_name, slot))
        col = {"enamel": (0.035, 0.09, 0.42, 1)}.get(slot, (0.5, 0.5, 0.5, 1))
        m = pbr_material(name, base_color=col, rough=0.16 if slot == "enamel" else 0.5, spec=0.6,
                         coat=0.4 if slot == "enamel" else 0.0)
    else:
        m = cm.material(set_name, name=name, double_sided=True, **opts)
    m.use_backface_culling = False
    m["rts_slot"] = slot
    return m


def apply_slots(pc, meta):
    """material slots per face from the asset's .rts.json (faces matched by their vertex keys: OBJ face order safe)"""
    names = meta["slots"]
    pc.data.materials.clear()
    for nm in names:
        pc.data.materials.append(slot_material(nm))
    fs = meta["face_slot"]
    keys = meta.get("face_keys")
    polys = pc.data.polygons
    if keys and len(keys) == len(polys):
        lut = {}
        for k, sl in zip(keys, fs):
            lut.setdefault(k, sl)
        miss = 0
        for p in polys:
            k = min(p.vertices) * 100000 + max(p.vertices)
            sl = lut.get(k)
            if sl is None:
                miss += 1; sl = names[0]
            p.material_index = names.index(sl)
        assert miss == 0, (pc.name, miss)
    else:
        assert len(fs) == len(polys), (pc.name, len(fs), len(polys))
        for p, sl in zip(polys, fs):
            p.material_index = names.index(sl)


def edit_bones(rig, fn):
    for o in bpy.context.view_layer.objects:
        o.select_set(False)
    bpy.context.view_layer.objects.active = rig; rig.select_set(True)
    bpy.ops.object.mode_set(mode='EDIT')
    try:
        fn(rig.data.edit_bones)
    finally:
        bpy.ops.object.mode_set(mode='OBJECT')


def ensure_bones(rig, specs):
    """specs: {name: dict(parent, head, tail, deform, z, frame)}; creates or moves bones (z = the local Z direction).
    Bones with a 'frame' (3x3 armature-space rest rotation, body independent) then get exactly that rest rotation
    through chr_lib.canonical_rest (judge M15: the extra joints must have the same rest rotation on every body, only
    their positions are fitted per body)."""
    def f(eb):
        for n, sp in specs.items():
            b = eb.get(n) or eb.new(n)
            b.head = Vector(sp["head"]); b.tail = Vector(sp["tail"])
            if sp.get("z") is not None:
                b.align_roll(Vector(sp["z"]))
            b.parent = eb[sp["parent"]]
            b.use_connect = False
            b.use_deform = sp.get("deform", True)
    edit_bones(rig, f)
    frames = {n: Matrix([list(r) for r in sp["frame"]]) for n, sp in specs.items() if sp.get("frame") is not None}
    if frames:
        canonical_rest(rig, frames)


def frame_yz(y, z):
    """3x3 rest rotation (columns X, Y, Z) with Y along y and Z as close to z as possible (Blender align_roll)"""
    Y = nrm(np.asarray(y, float)); Z = np.asarray(z, float)
    Z = nrm(Z - Y * Z.dot(Y)); X = np.cross(Y, Z)
    return [list(r) for r in np.stack([X, Y, Z], 1)]


# ------------------------------------------------------------------------------------------------ helper joints (M1)
# The plate helper joints are the rig's (scripts/rig_helpers.py, judge C1 / M1): knee_helper_<s> (parent thigh, half the
# knee swing) carries the poleyn cops, hip_helper_01/02_<s> (parent pelvis, 1/3 and 2/3 of the hip swing) the middle and
# lower tasset lames. This kit authors its rigid weights on them directly (so rig_helpers.apply_plate_rules leaves the
# pieces alone) and shapes the overlapping plates as shells round the joint centres, so a lame on one joint slides
# under the edge of the plate on the next one instead of cutting through it.
def ensure_helpers(rig):
    import rig_helpers as RH
    out = RH.ensure_helper_bones(rig, log=log)
    if "upperarm_helper_01_l" not in rig.data.bones:        # the pauldron joints the cape corners / clasps ride
        try:
            import armour_upper_rig as AUR
            AUR.ensure_helpers(rig)
        except Exception as e:
            log("arm helper joints unavailable (armour_upper_rig): %s" % e)
    return out


# ------------------------------------------------------------------------------------------------ grip sockets (user 13)
FINGER_CURL = (("01", 72.0), ("02", 88.0), ("03", 55.0))            # knight_anim.Engine.curl (deg at amount 1)
FINGER_K = {"index": 0.9, "middle": 1.0, "ring": 1.05, "pinky": 1.1}


def curl_amount(side):
    """the fist curl the knight's clips use (knight_anim SWORD_CURL / SHIELD_CURL), read-only"""
    try:
        import knight_anim as KA
        return (KA.SWORD_CURL if side == "r" else KA.SHIELD_CURL)[0]
    except Exception:
        return 0.82 if side == "r" else 0.55


def fist_axis(hd, tl, side, amount):
    """Closed-fist grip axis: curl each finger chain like knight_anim.Engine.curl (FK in armature space from the rest
    heads / tails given by hd(n) / tl(n)), fit a circle to its knuckle / middle / end joints and tip, and fit a line
    through the four circle centres. Returns (centre, axis pinky -> index, mean finger-line radius, per-finger radii)."""
    kn = np.mean([hd(f + "_01_" + side) for f in FINGER_K], 0)
    wr = hd("hand_" + side)
    ax_h = nrm(kn - wr)
    ac = hd("index_01_" + side) - hd("pinky_01_" + side); ac = nrm(ac - ax_h * ac.dot(ax_h))
    c = wr * 0.35 + kn * 0.65
    palm = hd("thumb_02_" + side) - c
    palm = nrm(palm - ax_h * palm.dot(ax_h) - ac * palm.dot(ac))
    cents, radii = [], []
    for f, k in FINGER_K.items():
        names = ["%s_%s_%s" % (f, i, side) for i, _ in FINGER_CURL]
        Racc = Matrix.Identity(3)
        p = Vector(hd(names[0]))
        pts = [np.array(p)]
        for j, (n, (i, deg)) in enumerate(zip(names, FINGER_CURL)):
            d = nrm(tl(n) - hd(n))
            ax = Vector(nrm(np.cross(d, palm)))
            ax = Racc @ ax
            Racc = Matrix.Rotation(math.radians(deg * amount * k), 3, ax) @ Racc
            nxt = hd(names[j + 1]) if j + 1 < len(names) else tl(n)
            p = p + Racc @ Vector(nxt - hd(n))
            pts.append(np.array(p))
        P = np.array(pts)
        # circle fit in the curl plane (least squares, algebraic)
        m = P.mean(0); U, S, Vt = np.linalg.svd(P - m); e1, e2 = Vt[0], Vt[1]
        q = np.stack([(P - m) @ e1, (P - m) @ e2], 1)
        A = np.c_[2 * q, np.ones(len(q))]; b = (q ** 2).sum(1)
        sol = np.linalg.lstsq(A, b, rcond=None)[0]
        cc = m + sol[0] * e1 + sol[1] * e2
        cents.append(cc); radii.append(float(np.mean(np.linalg.norm(P[1:] - cc, axis=1))))
    C = np.array(cents); cen = C.mean(0)
    U, S, Vt = np.linalg.svd(C - cen); axis = nrm(Vt[0])
    if axis.dot(ac) < 0:
        axis = -axis
    return cen, axis, float(np.mean(radii)), radii, ax_h


def piece_basis(pc):
    kb = pc.data.shape_keys.key_blocks["Basis"] if pc.data.shape_keys else None
    n = len(pc.data.vertices)
    co = np.empty(n * 3)
    (kb.data if kb else pc.data.vertices).foreach_get("co", co)
    return co.reshape(-1, 3)


def chain_bone_specs(meta_bones, co=None, fallback_centre=None):
    """bone specs from the vertex-mean strategy (fitted piece coords co) or a placeholder under the parent"""
    out = {}
    for n, sp in meta_bones.items():
        if co is not None:
            h = co[sp["head"]].mean(0) + np.array(sp["head_off"]); t = co[sp["tail"]].mean(0) + np.array(sp["tail_off"])
        else:
            h = np.array(fallback_centre) + np.array([0, 0, -0.01]); t = h + np.array([0, 0, -0.1])
        out[n] = dict(parent=sp["parent"], head=h, tail=t, deform=True, z=(0, 1, 0), frame=sp.get("frame"))
    return out


from armour_lower_props import SHIELD_STRAP_AT, SHIELD_STRAP_OFF, SHIELD_STRAP_DIAG    # the enarmes contract
GRIP_CONTRACT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "out", "grip_contract.json")


def grip_contract(kind):
    """armour_upper's grip contract (out/grip_contract.json): per body, per hand socket the head (hand-local) and the
    grip radius the fist is solved round; {} when absent"""
    try:
        return json.load(open(GRIP_CONTRACT))["bodies"].get(kind, {})
    except Exception:
        return {}


def shield_hand_rot():
    """the clips' shield-hand wrist rotation about the wrist relative to the forearm, in rest space (knight_anim holds
    the shield hand at shield_arm's fixed wrist angle in every clip: measured 0.0 mm spread, iteration 2)"""
    try:
        import knight_anim as KA
        return np.array(KA.shield_arm(Matrix.Identity(3), Vector((0, 0, -1)), Vector((0, -1, 0)),
                                      Vector((1, 0, 0)))["hand"])
    except Exception:
        return np.array(Matrix.Rotation(math.radians(-6.0), 3, 'X'))


def rig_json_positions():
    """(hd, tl) lookups of the rig JSON's DEFAULT joint positions (the neutral basemesh): body-independent data for the
    canonical frames of the sockets (the same data chr_lib.canonical_rest_frames uses for the core joints)"""
    J = json.load(open(RIG_JSON))["bones"]
    H = {n: np.array(b["head"]["default_position"], float) for n, b in J.items()}
    T = {n: np.array(b["tail"]["default_position"], float) for n, b in J.items()}
    return (lambda n: H[n]), (lambda n: T[n])


SCABBARD_TILT = 32.0          # deg from vertical, tip back (fixed: one canonical socket frame for every body)


def socket_specs(rig, belts=None, belts_meta=None):
    """weapon / shield / scabbard / back sockets: POSITIONS from the fitted skeleton (and the hip belt anchor), rest
    ROTATIONS ('frame') from body-independent data (rig JSON default positions / constants), so every body has the
    same socket frames (M15) and one prop GLB attaches the same way to all of them."""
    B = rig.data.bones
    hd = lambda n: np.array(B[n].head_local); tl = lambda n: np.array(B[n].tail_local)
    dh, dt = rig_json_positions()
    out = {}
    # grips (user feedback 13): the socket is the axis of the CLOSED fist the clips use (fingers curled by
    # knight_anim.SWORD_CURL / SHIELD_CURL), so the grip sits where the fingers close round it; +Y = pinky -> index
    # (blade side), +Z = towards the knuckles. The old socket (a point 3.4 cm in front of the OPEN palm) sat below the
    # fingers' loop: the fist closed through the grip and the pommel end dug into the cuff.
    kind = rig.get("rts_kind", rig.name.replace("rts_", ""))
    con = grip_contract(kind)
    for side, name in (("r", "socket_weapon_r"), ("l", "socket_hand_l")):
        amt = curl_amount(side)
        cen, axis, rad, radii, axh = fist_axis(hd, tl, side, amt)
        _, axis_c, _, _, axh_c = fist_axis(dh, dt, side, amt)
        fr = frame_yz(axis_c, axh_c)
        Y = np.array(fr)[:, 1]
        c = con.get(name)
        if c and "head_in_hand_local" in c:
            # iteration 2b (user item 29, grip contract): the fist-axis centre lay 4.8-7.3 mm from the gauntlet's
            # palm, so the grip passed through the palm; the contract moves the head palmar-away (hand-local head)
            Mh = np.array(B["hand_" + side].matrix_local)
            head = Mh[:3, :3] @ np.array(c["head_in_hand_local"], float) + Mh[:3, 3]
            log("grip socket %s: contract head %.1f mm from the fist-axis centre (radius %.1f mm)" % (
                name, np.linalg.norm(head - cen) * 1000, c.get("radius", 0) * 1000))
            cen = head
        out[name] = dict(parent="hand_" + side, head=cen, tail=cen + Y * 0.1, deform=False, frame=fr,
                         fist_radius=rad)
        log("grip socket %s: fist radius %.1f mm (fingers %s), curl %.2f" % (
            name, rad * 1000, " ".join("%.0f" % (r * 1000) for r in radii), amt))
    # left forearm shield: the enarmes at the wrist end of the forearm (95 %), outside (dorsal / lateral) of it, 9.5 cm
    # off the forearm axis so the strap pad rests on the vambrace; with the enarmes 28 cm below the shield's top edge
    # (armour_lower_props) the top stays below the elbow / upper-arm plates and the pauldron when the arm hangs. (Assembly pass, user feedback 13: at 55 % / 4.5 cm along the forearm
    # the shield's back cut through the vambrace and its top edge reached 28 cm past the elbow into the couter,
    # rerebrace and pauldron.)
    # The enarmes run DIAGONALLY across the back of a kite shield: the socket's +Y (the shield's long axis, towards the
    # point) is the forearm direction turned SHIELD_STRAP_DIAG degrees in the shield plane, so the shield hangs upright
    # when the forearm is carried forward-down / across the body, instead of lying along the forearm.
    def shield_frame(hd_, tl_):
        e_, w_ = hd_("lowerarm_l"), tl_("lowerarm_l")
        fa_ = nrm(w_ - e_); od = np.array([1.0, 0.25, 0.0]); od = nrm(od - fa_ * od.dot(fa_))
        th_ = math.radians(SHIELD_STRAP_DIAG)
        d_ = nrm(math.cos(th_) * fa_ - math.sin(th_) * np.cross(fa_, od))
        return frame_yz(d_, od)
    fr = shield_frame(dh, dt)
    Fm = np.array(fr)
    e, w = hd("lowerarm_l"), tl("lowerarm_l")
    c = e + (w - e) * SHIELD_STRAP_AT + Fm[:, 2] * SHIELD_STRAP_OFF
    # iteration 2b (user item 24, G5): the left fist HOLDS the shield: the prop's handle lies on the grip contract's
    # socket_hand_l axis in the clips' wrist pose (knight_anim's fixed shield-hand angle), and the forearm strap goes
    # round this body's forearm at 40 %: both fitted per body in the socket frame (rig extras rts_shield_fit, used by
    # add_props / export_props; the board, rim and face are the same on every body)
    hs = out["socket_hand_l"]
    wr = hd("hand_l"); Rw = shield_hand_rot()
    fist = wr + Rw @ (np.asarray(hs["head"]) - wr)
    fax = Rw @ np.array(hs["frame"])[:, 1]
    loc = lambda p: Fm.T @ (np.asarray(p) - c)
    fa_l = Fm.T @ nrm(w - e)
    import armour_lower_props as PR
    fit = dict(handle=[loc(fist).tolist(), (Fm.T @ fax).tolist()],
               loop=[loc(e + (w - e) * PR.STRAP_LOOP_FRAC).tolist(), fa_l.tolist(), PR.STRAP_LOOP_R])
    rig["rts_shield_fit"] = json.dumps(fit)
    log("shield fit: handle %s dir %s; forearm strap at %s along %s" % (
        np.round(fit["handle"][0], 4).tolist(), np.round(fit["handle"][1], 3).tolist(),
        np.round(fit["loop"][0], 4).tolist(), np.round(fit["loop"][1], 3).tolist()))
    out["socket_shield_l"] = dict(parent="lowerarm_l", head=c, tail=c + Fm[:, 1] * 0.1, deform=False, frame=fr)
    # back: centre of the upper back, outside the cape (shield / sword carried on the back)
    sp5 = hd("spine_05"); c = np.array([0.0, sp5[1] + 0.20, sp5[2] + 0.02])
    out["socket_back"] = dict(parent="spine_05", head=c, tail=c + np.array([0, 0, -0.1]), deform=False,
                              frame=frame_yz((0, 0, -1), (0, 1, 0)))
    # scabbard on the left hip (belt anchor), hanging tip down and back at a fixed tilt
    if belts is not None and belts_meta and "anchors" in belts_meta:
        co = piece_basis(belts)
        f = co[belts_meta["anchors"]["scabbard_frog"]].mean(0)
    else:
        pv = hd("thigh_l"); f = np.array([pv[0] + 0.08, pv[1] + 0.02, pv[2] - 0.06])
    f = f + np.array([0.022, 0.0, -0.012])
    a = math.radians(SCABBARD_TILT)
    d = nrm(np.array([0.10 * math.cos(a), math.sin(a), -math.cos(a)]))
    fr = frame_yz(d, (1, 0, 0))
    out["socket_scabbard_l"] = dict(parent="pelvis", head=f, tail=f + d * 0.1, deform=False, frame=fr)
    s0 = f - d * 0.066
    out["socket_sword_sheathed"] = dict(parent="pelvis", head=s0, tail=s0 + d * 0.1, deform=False, frame=fr)
    return out


def add_driver_max(rig, bone, idx, expr, thighs=("thigh_l", "thigh_r")):
    pb = rig.pose.bones[bone]
    pb.rotation_mode = 'XYZ'
    fc = pb.driver_add("rotation_euler", idx)
    d = fc.driver; d.type = 'SCRIPTED'
    for k, tb in enumerate(thighs):
        v = d.variables.new(); v.name = "ab"[k]; v.type = 'TRANSFORMS'
        t = v.targets[0]; t.id = rig; t.bone_target = tb; t.transform_type = 'ROT_X'
        t.rotation_mode = 'SWING_TWIST_Y'; t.transform_space = 'LOCAL_SPACE'
    d.expression = expr
    return fc


SECONDARY = {
    # bone: (euler index, expression in a = thigh_l local swing X, b = thigh_r local swing X (radians; negative = the
    #        thigh swings forward, positive = back)).  Flaps are pushed by the leading thigh, the cape by the trailing one.
    # (the flaps pivot under the hip belt, lower than the thigh joint: they need more than the thigh's swing)
    "tabard_f_01": (0, "-1.08*max(0,-a,-b)"), "tabard_f_02": (0, "-0.16*max(0,-a,-b)"),
    # the cape must swing at least as far as the back flap under it (displacement at the flap hem: cape
    # 0.70 * 0.45 m + 0.30 * 0.15 m = 0.36 m/rad > flap 0.85 * 0.365 m = 0.31 m/rad, plus the 3 cm rest gap)
    "tabard_b_01": (0, "0.85*max(0,a,b)"), "tabard_b_02": (0, "0.10*max(0,a,b)"),
    "cape_c_02": (0, "0.70*max(0,a,b)"), "cape_c_03": (0, "0.30*max(0,a,b)"),
    "cape_l_02": (0, "0.70*max(0,a,b)"), "cape_l_03": (0, "0.30*max(0,a,b)"),
    "cape_r_02": (0, "0.70*max(0,a,b)"), "cape_r_03": (0, "0.30*max(0,a,b)"),
}


def add_secondary(rig):
    for b, (i, ex) in SECONDARY.items():
        if b in rig.pose.bones:
            add_driver_max(rig, b, i, ex)
    rig["rts_secondary"] = json.dumps({
        "doc": "tabard / cape chain bones follow the thighs so the flaps clear the legs; engine: evaluate per frame "
               "(or bake into clips / replace with spring bones). a / b = local swing about X (swing-twist, Y = bone "
               "axis) of thigh_l / thigh_r in radians; result = local Euler X rotation of the bone.",
        "rules": {b: {"axis": "X", "expr": ex} for b, (i, ex) in SECONDARY.items() if b in rig.pose.bones}})


def bake_vertex_ao(objs, rig, bm, rays=24, reach=0.035, strength=0.45, attr="Color"):
    """Per-vertex ambient occlusion (rest pose) against the whole dressed stack (body + every piece), written to the
    colour attribute the shared armour material multiplies into the base colour (armour_upper AO_ATTR = 'Color').
    Plates get contact darkening where lames overlap and where they sit over mail / cloth."""
    from mathutils.bvhtree import BVHTree
    dg = bpy.context.evaluated_depsgraph_get()
    V, P = [], []
    for o in [bm] + [c for c in children_meshes(rig) if c != bm]:
        ev = o.evaluated_get(dg); me = ev.to_mesh()
        off = len(V)
        V += [o.matrix_world @ v.co for v in me.vertices]
        P += [tuple(i + off for i in p.vertices) for p in me.polygons]
        ev.to_mesh_clear()
    bvh = BVHTree.FromPolygons(V, P)
    rng = np.random.default_rng(7)
    dirs = rng.normal(size=(rays, 3)); dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    for o in objs:
        me = o.data
        n = len(me.vertices)
        ao = np.ones(n)
        for i, v in enumerate(me.vertices):
            nrm_ = v.normal
            p0 = o.matrix_world @ (v.co + nrm_ * 0.0012)
            hits = 0; tot = 0
            for d in dirs:
                dv = Vector(d)
                c = dv.dot(nrm_)
                if c < 0:
                    dv = -dv; c = -c
                w = c
                tot += w
                if bvh.ray_cast(p0, dv, reach)[0] is not None:
                    hits += w
            ao[i] = 1.0 - strength * hits / max(tot, 1e-6)
        ao = np.clip(ao, 0.5, 1.0)
        if attr in me.color_attributes:
            me.color_attributes.remove(me.color_attributes[attr])
        ca = me.color_attributes.new(attr, 'BYTE_COLOR', 'POINT')
        col = np.repeat(ao[:, None], 4, 1); col[:, 3] = 1.0
        ca.data.foreach_set("color", col.ravel())
        log("vertex AO %-16s mean %.2f min %.2f" % (o.name, ao.mean(), ao.min()))


def floor_clamp(o, z0=0.0005):
    """lift fitted vertices that ended below the floor (boot soles on smaller feet) in the Basis and every shape key"""
    me = o.data
    kb = me.shape_keys.key_blocks if me.shape_keys else None
    n = len(me.vertices)
    base = np.empty(n * 3)
    (kb["Basis"].data if kb else me.vertices).foreach_get("co", base); base = base.reshape(-1, 3)
    lift = np.maximum(z0 - base[:, 2], 0.0)
    if lift.max() <= 0:
        return 0
    if kb:
        for k in kb:
            a = np.empty(n * 3); k.data.foreach_get("co", a); a = a.reshape(-1, 3); a[:, 2] += lift
            k.data.foreach_set("co", a.ravel())
    b2 = base.copy(); b2[:, 2] += lift
    me.vertices.foreach_set("co", b2.ravel()); me.update()
    log("floor clamp %s: %d verts lifted (max %.1f mm)" % (o.name, int((lift > 0).sum()), lift.max() * 1000))
    return int((lift > 0).sum())


def set_rest(o, new_world):
    """replace the rest shape of a (fitted) piece, keeping every shape key's delta; returns the max move (m)"""
    me = o.data
    n = len(me.vertices)
    Mi = np.array(o.matrix_world.inverted())
    new = np.asarray(new_world, float) @ Mi[:3, :3].T + Mi[:3, 3]
    old = piece_basis(o)
    d = new - old
    kb = me.shape_keys.key_blocks if me.shape_keys else None
    if kb:
        for k in kb:
            a = np.empty(n * 3); k.data.foreach_get("co", a)
            k.data.foreach_set("co", (a.reshape(-1, 3) + d).ravel())
    me.vertices.foreach_set("co", new.ravel()); me.update()
    return float(np.linalg.norm(d, axis=1).max()) if n else 0.0


def set_uvs_from(o, name, m):
    """(iteration 2b, integrity G6: the regenerated pieces kept the authoring body's UVs, so on the female the
    cuisses' steel streaked on 75 % of their area) the UVs of the regenerated mesh (the same mapping write_piece wrote),
    matched to the object's polygons by their vertex sets"""
    mm = to_sheet(name, m)
    uvs = final_uvs(mm)
    lut = {frozenset(f): (f, uv) for f, uv in zip(mm.F, uvs)}
    me = o.data
    uvl = me.uv_layers.active.data if me.uv_layers.active else None
    if uvl is None:
        return 0
    n = 0
    for p in me.polygons:
        e = lut.get(frozenset(p.vertices))
        if e is None:
            continue
        d = dict(zip(e[0], e[1]))
        for li in p.loop_indices:
            uvl[li].uv = d[me.loops[li].vertex_index]
        n += 1
    return n


def cover_fitted_boot(B, o):
    """(2b, G4 boots|underlayer) a MakeClothes-fitted boot (topology from another body) pushed out over THIS body's
    skin like boot_cover(); the sole slab (below 16 mm) is kept. Returns the largest move (m)"""
    me = o.data
    Pw = piece_basis(o); M = np.array(o.matrix_world); Pw = Pw @ M[:3, :3].T + M[:3, 3]
    F = [tuple(p.vertices) for p in me.polygons]
    N = np.zeros_like(Pw)
    for f in F:
        fn = np.cross(Pw[f[1]] - Pw[f[0]], Pw[f[2]] - Pw[f[0]])
        for i in f:
            N[i] += fn
    N /= np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-12)
    up = Pw[:, 2] > 0.016
    Fu = [f for f in F if up[list(f)].all()]
    idx = np.flatnonzero(up); rm = {int(v): k for k, v in enumerate(idx)}
    Pu = boot_cover(B, Pw[idx], [tuple(rm[i] for i in f) for f in Fu], N[idx])
    Pu[:, 2] = np.maximum(Pu[:, 2], 0.0137)
    new = Pw.copy(); new[idx] = Pu
    set_rest(o, new)
    return float(np.linalg.norm(new - Pw, axis=1).max())


def regen_on_body(rig, bm, objs, upper=None):
    """Rebuild every lower piece's rest shape on THIS body over the upper armour actually dressed on it (female and
    proportion builds, and the upper kit's own per-body regeneration): the generators run on the body's landmarks
    (Z / XS) and layers, and where the topology matches the loaded MakeClothes-fitted piece its rest shape is
    replaced (shape-key deltas kept). Pieces whose topology depends on the body (tights-helper mail / boots) keep the
    MakeClothes fit. Returns {piece: max move mm or reason}."""
    upper = upper if upper is not None else upper_layers_live(rig)
    if not upper:
        return {"skipped": "no upper armour dressed"}
    pose_reset(rig)
    B = Body(rig, bm)
    pieces, H, D = build_all_pieces(B, upper)
    rig["rts_waist_band"] = json.dumps({k: float(B.waist[k]) for k in ("bot", "top", "zc", "rim_min", "rim_hi", "tasset1_bot")
                                        if k in B.waist})
    rep = {}
    for n, m, k in pieces:
        o = objs.get(n)
        if o is None:
            continue
        if m.mixed():
            m.triangulate()
        if len(m.V) != len(o.data.vertices):
            rep[n] = "kept MakeClothes fit (%d vs %d verts)" % (len(m.V), len(o.data.vertices))
            if n == "boots":
                rep[n] += ", covered %.1f mm" % (cover_fitted_boot(B, o) * 1000)
            continue
        rep[n] = round(set_rest(o, np.array(m.V)) * 1000, 1)
        try:
            set_uvs_from(o, n, m)
        except Exception as e:
            log("WARNING regen UVs %s: %r" % (n, e))
    log("regen on body: %s" % rep)
    rig["rts_lower_regen"] = json.dumps(rep)
    return rep


# ------------------------------------------------------------------------------------------------ co-skinning
# A layer that LIES ON another layer must move exactly like it, or the two shear apart (floating belts, user 21) or
# cut into each other (tabard through the breastplate in a bend, M2). coskin() copies the final skin weights of the
# layer underneath onto the parts of these pieces that rest on it (after the plate weights are final: load_lower and
# again in post_clips, after rig_helpers.apply_plate_rules re-weighted the upper plates).
COSKIN = [   # target, sources (layers it rests on), reach (m), mode
    # iteration 2b: the waist band (belt, tabard band, skirt top rows, top tasset lame) takes the cuirass rim's live
    # joints in place of the authored RIM_BONE ('waist', first), then the tabard above the belt copies the cuirass
    ("belts", ("cuirass",), 0.0, "waist"), ("tabard", ("cuirass",), 0.0, "waist"),
    ("mail_skirt", ("cuirass",), 0.0, "waist"), ("tassets", ("cuirass",), 0.0, "waist"),
    ("tabard", ("cuirass", "mail", "gorget"), 0.030, "above_belt"),
    # (2b session 3, integrity G4 legs_mail | mail_skirt / belts / tabard / tassets: measured on the knight, the
    # chausses' body weights carry the hip front 22-35 mm OUT of the belt band and 40-45 mm just under it in the
    # attack / block (the thigh's share runs up to the waist), through every layer over them) the chausses' top ramps
    # to the rim's joints over STACK_RAMP under the belt (hidden under the closed skirt ring); the skirt then copies the
    # layer BENEATH it (cuisse plate: rigid thigh, else the chausses), rim above the same ramp, so the stack keeps its
    # rest clearances; the front / back under the flaps still ride the flaps (next line)
    ("legs_mail", ("cuirass",), 0.0, "ramp_rim"),
    # (2b s3, integrity G4 boots | sabatons: the leather under the rigid lames bent with the skin's stepped foot / ball
    # share and rose through the lames at the toe crease in the walk / run) the boot upper under a sabaton lame rides
    # that lame's joint (blended out over 6-15 mm from it; the sole slab keeps its own)
    ("boots", ("sabatons",), 0.015, "under_plate"),
    ("mail_skirt", ("cuisses", "legs_mail", "mail"), 0.10, "beneath"),
    # (2b, tried and dropped: the skirt copying the tasset lames and the chausses copying the skirt made the stack lag
    # the thigh; the assembly's gap scan then saw the skin through it and cut an 1800 cm2 under-layer patch over the
    # hips that ran through the skirt, belts and tabard. The modes 'lames' / 'under' stay available below.)
    ("cape", ("gorget", "cuirass", "pauldron_l", "pauldron_r", "mail"), 0.030, "attached"),
    # (2b, integrity G3 clasp-cape / cape-pauldron: the clasp took the dominant joint of the nearest PAULDRON face - a
    # lame in the attack - while the cape corner under it copied other layers, so in the attack the clasp stood cm off
    # the cape) the clasp rides the joint of the CAPE under it, and the cape round the clasp rides the clasp's joint
    ("clasps", ("cape",), 0.080, "rigid"),
    ("cape", ("clasps",), 0.080, "under_clasp"),
]


STACK_RAMP_LEGS = (0.000, 0.015)     # (2b s3) below the cuirass rim's lowest point: the chausses ride the rim above
STACK_RAMP_SKIRT = (0.004, 0.035)    # the first offset, their own (the skirt: the layer beneath's) weights below the second


TASSET_SECTOR = (56.0, 110.0)        # deg from the front: the tasset lames (tassets(); 2b s3: were 52-108)


def stack_t(p, rim, ramp, t1b=None):
    """1 = rides the rim (at / above rim - ramp[0]), 0 = own / beneath weights (below rim - ramp[1]); under the tasset
    lames (TASSET_SECTOR +- 10 deg) the rim share reaches down to the top lame's lower edge t1b (it rides the rim)"""
    z = p[2]
    hi, lo = rim - ramp[0], rim - ramp[1]
    if t1b is not None:
        a = abs(math.degrees(math.atan2(p[0], -(p[1] - 0.015))))
        sw = float(smoothstep(TASSET_SECTOR[0] - 12, TASSET_SECTOR[0] - 2, a) * (1 - smoothstep(TASSET_SECTOR[1] + 2, TASSET_SECTOR[1] + 12, a)))
        d = max(0.0, (rim - ramp[0]) - (t1b + 0.006))
        hi, lo = hi - sw * d, lo - sw * d
    return float(smoothstep(lo, hi, z))


def rim_weights(srcs, bones):
    """the weights of the cuirass's lowest 2 cm (averaged, normalised): the joints the waist band rides"""
    co = np.concatenate([_world_basis(so) for so in srcs])
    W = sum((bone_weights(so, bones) for so in srcs), [])
    zmin = float(co[:, 2].min())
    acc = {}
    for p, w in zip(co, W):
        if p[2] < zmin + 0.02:
            for b, x in w.items():
                acc[b] = acc.get(b, 0.0) + x
    tot = sum(acc.values()) or 1.0
    out = {b: v / tot for b, v in acc.items() if v / tot > 0.02}
    t2 = sum(out.values()) or 1.0
    return {b: v / t2 for b, v in out.items()}


def _world_basis(o):
    co = piece_basis(o); M = np.array(o.matrix_world)
    return co @ M[:3, :3].T + M[:3, 3]


def bone_weights(o, bones):
    names = {g.index: g.name for g in o.vertex_groups}
    W = [dict() for _ in range(len(o.data.vertices))]
    for v in o.data.vertices:
        for g in v.groups:
            nm = names.get(g.group)
            if nm in bones and g.weight > 1e-5:
                W[v.index][nm] = W[v.index].get(nm, 0.0) + g.weight
    return W


def set_bone_weights(o, W, bones, maxinf=4):
    for g in list(o.vertex_groups):
        if g.name in bones:
            o.vertex_groups.remove(g)
    groups = {}
    for i, w in enumerate(W):
        top = sorted(w.items(), key=lambda kv: -kv[1])[:maxinf]
        tot = sum(v for _, v in top) or 1.0
        for b, v in top:
            if v / tot > 1e-4:
                groups.setdefault(b, []).append((i, v / tot))
    for b, lst in groups.items():
        g = o.vertex_groups.new(name=b)
        for i, v in lst:
            g.add([i], v, 'REPLACE')


def _mix(a, b, t):
    out = {k: v * (1 - t) for k, v in a.items()}
    for k, v in b.items():
        out[k] = out.get(k, 0.0) + v * t
    return {k: v for k, v in out.items() if v > 1e-5}


def _smooth_weights(o, W, sel, iters=2):
    adj = [[] for _ in range(len(o.data.vertices))]
    for e in o.data.edges:
        a, b = e.vertices
        adj[a].append(b); adj[b].append(a)
    for _ in range(iters):
        W2 = list(W)
        for i in sel:
            acc = dict(W[i])
            for j in adj[i]:
                for k, v in W[j].items():
                    acc[k] = acc.get(k, 0.0) + v
            tot = sum(acc.values()) or 1.0
            W2[i] = {k: v / tot for k, v in acc.items()}
        W = W2
    return W


def islands_of(o):
    n = len(o.data.vertices)
    par = list(range(n))

    def f(i):
        while par[i] != i:
            par[i] = par[par[i]]; i = par[i]
        return i
    for e in o.data.edges:
        a, b = f(e.vertices[0]), f(e.vertices[1])
        if a != b:
            par[a] = b
    g = {}
    for i in range(n):
        g.setdefault(f(i), []).append(i)
    return sorted(g.values(), key=len, reverse=True)


def ring_islands(co, isl, centre=(0.0, 0.015), cover=300.0):
    """the islands that go all the way round the body axis (belt rings), by angular coverage (deg)"""
    out = []
    for g in isl:
        P = np.asarray(co)[g]
        a = np.degrees(np.arctan2(P[:, 0] - centre[0], -(P[:, 1] - centre[1]))) % 360
        h = np.zeros(36, bool); h[(a // 10).astype(int) % 36] = True
        if h.sum() * 10 >= cover:
            out.append(g)
    return out


def coskin(rig, pieces=None, log_=None):
    """see COSKIN; pieces = {slot: object} (default: every rts_part mesh of the rig). Returns a report."""
    from mathutils.bvhtree import BVHTree
    log_ = log_ or log
    if pieces is None:
        pieces = {o["rts_part"]: o for o in children_meshes(rig) if o.get("rts_part")}
    bones = {b.name for b in rig.data.bones}
    pose_reset(rig)
    kind = rig.get("rts_kind", rig.name.replace("rts_", ""))
    if bpy.data.objects.get(kind + "_body") is not None:
        set_landmarks(rig, bpy.data.objects[kind + "_body"])
    rep = {}
    try:
        wtop = float(json.loads(rig["rts_waist_band"])["top"])
        wrim = float(json.loads(rig["rts_waist_band"]).get("rim_min", wtop - 0.07))
        wt1b = json.loads(rig["rts_waist_band"]).get("tasset1_bot")
    except Exception:
        wtop = Z(1.015) + 0.02
        wrim = Z(1.030)
        wt1b = None
    rimw = None
    for tgt, srcs, reach, mode in COSKIN:
        o = pieces.get(tgt)
        S = [ob for sl, ob in pieces.items()
             if sl in srcs or (ob.get("rts_group") != "knight_lower" and not ob.get("rts_prop") and upper_role(sl) in srcs)]
        if o is None or not S:
            continue
        if mode == "ramp_rim":
            if rimw is None:
                rimw = rim_weights(S, bones)
            if o.get("rts_rim_ramp"):                    # once per piece (coskin runs again in post_clips)
                rep[tgt + ":ramp"] = "kept"
                continue
            o["rts_rim_ramp"] = "%.3f" % wrim
            co = _world_basis(o)
            W = bone_weights(o, bones)
            new, changed = [], 0
            for i, w in enumerate(W):
                t = stack_t(co[i], wrim, STACK_RAMP_LEGS)
                if t <= 0:
                    new.append(w); continue
                new.append(_mix(w, rimw, t)); changed += 1
            set_bone_weights(o, new, bones)
            rep[tgt + ":ramp"] = changed
            continue
        if mode == "waist":
            # the authored RIM_BONE share of every vertex -> the live rim joints (identity on the authoring male)
            if rimw is None:
                rimw = rim_weights(S, bones)
                log_("coskin: waist band rides the cuirass rim joints %s" % {k: round(v, 2) for k, v in rimw.items()})
            W = bone_weights(o, bones)
            new, changed = [], 0
            for w in W:
                r = w.get(RIM_BONE, 0.0)
                if r <= 1e-5 or rimw == {RIM_BONE: 1.0}:
                    new.append(w); continue
                q = {b: v for b, v in w.items() if b != RIM_BONE}
                for b, v in rimw.items():
                    q[b] = q.get(b, 0.0) + v * r
                new.append(q); changed += 1
            if changed:
                set_bone_weights(o, new, bones)
            rep[tgt + ":waist"] = changed
            continue
        V, P, SW, owner = [], [], [], []
        for so in S:
            off = len(V); co = _world_basis(so)
            V += [Vector(c) for c in co]; P += [tuple(i + off for i in p.vertices) for p in so.data.polygons]
            SW += bone_weights(so, bones); owner += [so.get("rts_part")] * len(co)
        bvh = BVHTree.FromPolygons(V, P)
        Vn = np.array(V)
        co = _world_basis(o)
        W = bone_weights(o, bones)
        new = list(W)

        def copied(p):
            loc, nn, fi, dist = bvh.find_nearest(Vector(p), reach)
            if loc is None:
                return None
            vs = P[fi]; q = np.array(loc)
            d = np.linalg.norm(Vn[list(vs)] - q, axis=1)
            k = 1.0 / np.maximum(d, 1e-4); k /= k.sum()
            acc = {}
            for vi, kk in zip(vs, k):
                for b, x in SW[vi].items():
                    acc[b] = acc.get(b, 0.0) + x * kk
            return acc
        changed = []
        if mode == "rigid":
            for isl in islands_of(o):
                c = co[isl].mean(0)
                loc, nn, fi, dist = bvh.find_nearest(Vector(c), reach)
                if loc is None:
                    continue
                acc = {}
                for vi in P[fi]:
                    for b, x in SW[vi].items():
                        acc[b] = acc.get(b, 0.0) + x
                b0 = max(acc, key=acc.get)
                for i in isl:
                    new[i] = {b0: 1.0}
                changed += isl
        elif mode == "belt":
            isl = islands_of(o)
            rg = ring_islands(co, isl)
            rings = [i for g in rg for i in g]
            isl = rg + [g for g in isl if not any(g is r_ for r_ in rg)]
            isl = rg + [g for g in islands_of(o) if g[0] not in set(rings)]
            for i in rings:
                w = copied(co[i])
                if w:
                    new[i] = w; changed.append(i)
            new = _smooth_weights(o, new, changed, iters=1)
            rc = co[rings] if rings else co
            for g in (isl[len(rg):] if rings else []):                  # buckle, tongue, pouches, studs: rigid on the ring point they hang from
                c = co[g].mean(0)
                j = rings[int(np.argmin(np.linalg.norm(rc - c, axis=1)))]
                w = dict(new[j])
                if c[2] < co[j][2] - 0.025:                  # a pouch hanging below the belt rides what it rests on
                    back = co[g][np.argmin(np.linalg.norm(co[g][:, :2] - np.array([0.0, 0.015]), axis=1))]
                    w = copied(back) or w
                for i in g:
                    new[i] = dict(w)
                changed += g
        elif mode == "flap":
            # (2b) the skirt's front under the tabard flap copies the flap's weights (chain), so the flap solve never
            # has to clear the skirt and the skirt never pokes through the flap; blended out at the flap's edges.
            # Likewise the skirt's back under the back flap (G4 mail_skirt|tabard: the back flap follows the trailing
            # thigh by its rule, the skirt's back its own thigh share)
            for fb, pre in ((-1.0, "tabard_f_01"), (1.0, "tabard_b_01")):
                b_ = rig.data.bones.get(pre)
                piv = float((rig.matrix_world @ b_.head_local).z) if b_ is not None else tabard_pivot_live(rig)
                fr = [i for i, p in enumerate(Vn) if fb * (p[1] - 0.015) > 0.035 and p[2] < piv - 0.005]
                if not fr:
                    continue
                hwf = float(np.abs(Vn[fr][:, 0]).max())
                for i, p in enumerate(co):
                    if fb * (p[1] - 0.015) < 0.015 or p[2] > piv - 0.01:
                        continue
                    t = float((1 - smoothstep(hwf - 0.01, hwf + 0.03, abs(p[0]))) * smoothstep(piv - 0.01, piv - 0.05, p[2]))
                    if t <= 0:
                        continue
                    w = copied(p)
                    if w:
                        new[i] = _mix(W[i], w, t); changed.append(i)
            new = _smooth_weights(o, new, changed, iters=1)
        elif mode == "beneath":
            # the layer straight under each vertex (a ray towards the body axis, level; the nearest surface when it
            # misses), its weights blended to the rim's joints over the stack ramp (rimw from 'ramp_rim' / 'waist')
            rw = rimw or {RIM_BONE: 1.0}
            for i, p in enumerate(co):
                t = stack_t(p, wrim, STACK_RAMP_SKIRT)
                d = Vector((-(p[0] - 0.0), -(p[1] - 0.015), 0.0))
                w = None
                if d.length > 1e-6:
                    d.normalize()
                    h = bvh.ray_cast(Vector(p) - d * 0.002, d, reach)
                    if h[0] is not None:
                        vs = P[h[2]]; q = np.array(h[0])
                        dd = np.linalg.norm(Vn[list(vs)] - q, axis=1); k = 1.0 / np.maximum(dd, 1e-4); k /= k.sum()
                        w = {}
                        for vi, kk in zip(vs, k):
                            for b, x in SW[vi].items():
                                w[b] = w.get(b, 0.0) + x * kk
                if w is None:
                    w = copied(p)
                if not w:
                    continue
                new[i] = _mix(w, rw, t) if t > 0 else w
                changed.append(i)
            new = _smooth_weights(o, new, changed, iters=1)
        elif mode == "under_plate":
            for i, p in enumerate(co):
                if p[2] < 0.017:                                  # the sole slab
                    continue
                loc, nn, fi, dist = bvh.find_nearest(Vector(p), reach)
                if loc is None:
                    continue
                t = float(1 - smoothstep(0.006, reach, dist))
                if t <= 0:
                    continue
                acc = {}
                for vi in P[fi]:
                    for b, x in SW[vi].items():
                        acc[b] = acc.get(b, 0.0) + x
                tot = sum(acc.values()) or 1.0
                new[i] = _mix(W[i], {b: v / tot for b, v in acc.items()}, t); changed.append(i)
            new = _smooth_weights(o, new, changed, iters=1)
        elif mode == "lames":
            # the skirt under a tasset lame: 5 rays outward from the body axis (the vertex and +-1.5 cm / +-6 deg round
            # it) meet the lames' inner skins; the share of rays that hit = how much of the lame's joint it takes
            for i, p in enumerate(co):
                acc = {}; hits = 0
                ph0 = math.atan2(p[0], -(p[1] - 0.015))
                for dz, dph in ((0.0, 0.0), (0.015, 0.0), (-0.015, 0.0), (0.0, 0.105), (0.0, -0.105)):
                    ph = ph0 + dph
                    d = Vector((math.sin(ph), -math.cos(ph), 0.0))
                    o_ = Vector((p[0], p[1], p[2] + dz))
                    h = bvh.ray_cast(o_, d, reach)
                    if h[0] is None:
                        continue
                    hits += 1
                    vs = P[h[2]]; q = np.array(h[0])
                    dd = np.linalg.norm(Vn[list(vs)] - q, axis=1); k = 1.0 / np.maximum(dd, 1e-4); k /= k.sum()
                    for vi, kk in zip(vs, k):
                        for b, x in SW[vi].items():
                            acc[b] = acc.get(b, 0.0) + x * kk
                if hits:
                    tot = sum(acc.values()) or 1.0
                    new[i] = _mix(W[i], {b: v / tot for b, v in acc.items()}, hits / 5.0); changed.append(i)
            new = _smooth_weights(o, new, changed, iters=1)
        elif mode == "under_clasp":
            cen = []
            for so in S:
                cso = _world_basis(so); Wso = bone_weights(so, bones)
                for isl in islands_of(so):
                    acc = {}
                    for i in isl:
                        for b, x in Wso[i].items():
                            acc[b] = acc.get(b, 0.0) + x
                    cen.append((cso[isl].mean(0), max(acc, key=acc.get)))
            R0 = CLASP_R * 1.25
            for i, p in enumerate(co):
                best = None
                for c_, b_ in cen:
                    d_ = float(np.linalg.norm(p - c_))
                    if best is None or d_ < best[0]:
                        best = (d_, b_)
                if best is None:
                    continue
                t = float(1 - smoothstep(R0, R0 + 0.035, best[0]))
                if t <= 0:
                    continue
                chain = {b: v for b, v in W[i].items() if b.startswith("cape_")}
                if sum(chain.values()) > 0.5:                # the hanging part keeps its chain
                    continue
                new[i] = _mix(W[i], {best[1]: 1.0}, t); changed.append(i)
            new = _smooth_weights(o, new, changed, iters=1)
        elif mode == "under":
            # the chausses under the skirt (above the cuisses' top; blended out towards the visible inner thighs by
            # the distance to the skirt)
            for i, p in enumerate(co):
                tz = float(smoothstep(Z(0.70), Z(0.78), p[2]))
                if tz <= 0:
                    continue
                loc, nn, fi, dist = bvh.find_nearest(Vector(p), reach)
                if loc is None:
                    continue
                t = tz * float(1 - smoothstep(0.035, 0.065, dist))
                if t <= 0:
                    continue
                w = copied(p)
                if w:
                    new[i] = _mix(W[i], w, t); changed.append(i)
            new = _smooth_weights(o, new, changed, iters=1)
        elif mode == "above_belt":
            for i, p in enumerate(co):
                t = float(smoothstep(wtop + 0.002, wtop + 0.020, p[2]))
                if t <= 0:
                    continue
                w = copied(p)
                if w:
                    new[i] = _mix(W[i], w, t); changed.append(i)
            new = _smooth_weights(o, new, changed, iters=2)
        elif mode == "attached":
            for i, p in enumerate(co):
                att = sum(v for b, v in W[i].items() if not b.startswith("cape_"))
                if att <= 1e-4:
                    continue
                w = copied(p)
                if not w:
                    continue
                chain = {b: v for b, v in W[i].items() if b.startswith("cape_")}
                tot = sum(w.values()) or 1.0
                new[i] = dict(chain, **{b: v / tot * att for b, v in w.items()})
                changed.append(i)
            new = _smooth_weights(o, new, changed, iters=1)
        set_bone_weights(o, new, bones)
        rep[tgt] = len(set(changed))
    log_("coskin: %s" % rep)
    return rep


# ------------------------------------------------------------------------------------------------ front flap clearance
# Iteration 2b (user item 26, integrity G4 poleyns / cuisses / legs_mail | tabard): the front flap's driver rule
# (tabard_f_01 = 1.08 x the leading thigh's swing) is a rule of thumb; in the lunge of the attack, the run and the walk
# the knee cop, the cuisse and the chausses still came through the flap. Per clip frame the flap is now swung forward
# about its chain joints (top-down) by the least extra angle that keeps every leg-armour vertex under it FLAP_MARGIN
# behind its inner face (the real evaluated meshes, compared in each chain bone's rest frame, so any swing angle is
# handled), max-filtered and smoothed over time. The rule + the extra are keyed on the chain bones (their drivers are
# replaced by the keys; the rule stays in rts_secondary for engines), so knight_anim.bake_secondary bakes them as they
# are and the rest pose is untouched. The mail skirt's front under the flap rides the flap (coskin 'flap'), so the flap
# never has to clear it and it never pokes through the flap.
FLAP_BONES = ("tabard_f_01", "tabard_f_02", "tabard_f_03")
FLAP_HITS = ("poleyns", "cuisses", "legs_mail")
FLAP_MARGIN = 0.012              # flap inner face -> leg surface (the skirt's front lies in between)
FLAP_MAX = np.radians([30.0, 25.0, 20.0])      # extra swing limits (on top of the rule)
FLAP_LEVER_MIN = 0.10            # points this close to a joint are not cleared by its swing (2b s3: 0.035 tried: the
                                 # near-pivot chausses drove the top joint to its limit, the flap flew up like a board)


def tabard_pivot_live(rig):
    """the front flap chain's first joint height on this rig (rest)"""
    b = rig.data.bones.get("tabard_f_01")
    return float((rig.matrix_world @ b.head_local).z) if b is not None else 0.93


FLAP_SETS = ((("tabard_f_01", "tabard_f_02", "tabard_f_03"), -1.0), (("tabard_b_01", "tabard_b_02", "tabard_b_03"), 1.0))
FLAP_HITS_ALL = ("poleyns", "cuisses", "legs_mail", "greaves", "mail_skirt")   # (2b s3: the skirt's side panels' edges)
DRAPE_MAX = np.radians([80.0, 60.0, 45.0])     # outward swing limits over the hang (2b s3: 100 / 75 / 65 tried:
                                               # the flaps stood out as horizontal boards in the run and the attack)


def flap_clearance(rig, kind=None, step=2, iters=10):
    """Iteration 2b (user item 26 / integrity G4 legs | tabard; the flaps stood out as boards, horizontal in the run):
    the front AND back flaps DRAPE. Per clip frame, top-down, each chain joint first HANGS (the rotation about its
    own X axis that gives it its rest direction in the world, whatever the pelvis and the joint above do), then swings
    OUT (away from the legs) by the least angle that keeps every leg-armour vertex under it FLAP_MARGIN behind its
    inner face (the real evaluated meshes, compared in the joint's rest frame; the flap's top 10 cm cannot be cleared
    by a swing and is left to the rest layout). The outward swings are max-filtered and smoothed over time and keyed
    on the chain bones with the hang (their drivers are replaced by the keys; the thigh rules stay in rts_secondary
    for engines), so knight_anim.bake_secondary bakes them as they are and the rest pose is untouched. The mail
    skirt under each flap rides it (coskin 'flap')."""
    from mathutils.kdtree import KDTree
    kind = kind or rig.get("rts_kind", rig.name.replace("rts_", ""))
    tab = bpy.data.objects.get(kind + "_tabard")
    if tab is None or rig.animation_data is None:
        return {}
    hits = [o for o in (bpy.data.objects.get(kind + "_" + n) for n in FLAP_HITS_ALL) if o is not None]
    hit_keep = {}
    for o in hits:
        gi_ = {g.index: g.name for g in o.vertex_groups}
        hit_keep[o.name] = np.array([sum(g.weight for g in v.groups if gi_.get(g.group, "").startswith("tabard_")) < 0.05
                                     for v in o.data.vertices], bool)
    gi = {g.index: g.name for g in tab.vertex_groups}
    Mw = np.array(tab.matrix_world)
    rest = piece_basis(tab) @ Mw[:3, :3].T + Mw[:3, 3]
    sc = bpy.context.scene
    act0 = rig.animation_data.action
    table = json.loads(rig.get("rts_clips", "{}"))
    clips = [c for c in table if c != "talk_emote" and bpy.data.actions.get(c)]

    def hit_points():
        dg = bpy.context.evaluated_depsgraph_get()
        out = []
        for o in hits:
            ev = o.evaluated_get(dg); me = ev.to_mesh()
            co = np.empty(len(me.vertices) * 3); me.vertices.foreach_get("co", co); ev.to_mesh_clear()
            M = np.array(o.matrix_world)
            q_ = co.reshape(-1, 3) @ M[:3, :3].T + M[:3, 3]
            k_ = hit_keep.get(o.name)
            out.append(q_[k_] if k_ is not None and len(k_) == len(q_) else q_)
        return np.concatenate(out) if out else np.zeros((0, 3))

    flaps = []
    for names, side in FLAP_SETS:
        bones = [b for b in names if b in rig.pose.bones]
        if not bones:
            continue
        fv, fb = [], []
        for v in tab.data.vertices:
            best = max(((g.weight, gi.get(g.group, "")) for g in v.groups), default=(0, ""))
            if best[1] in bones:
                fv.append(v.index); fb.append(bones.index(best[1]))
        if not fv:
            continue
        fv = np.array(fv); fb = np.array(fb)
        piv = float((rig.matrix_world @ rig.data.bones[bones[0]].head_local).z)
        keep = rest[fv, 2] < piv - 0.03
        fv, fb = fv[keep], fb[keep]
        R = rest[fv]
        kds = []
        for k in range(len(bones)):
            sel = np.flatnonzero(fb == k)
            kd = KDTree(max(len(sel), 1))
            for i in sel:
                kd.insert((R[i, 0], 0.0, R[i, 2]), int(i))
            kd.balance()
            kds.append((kd, len(sel)))
        sgn, rdir = {}, {}
        for b in bones:
            M3 = rig.matrix_world.to_3x3() @ rig.data.bones[b].matrix_local.to_3x3()
            zax = M3 @ Vector((0.0, 0.0, 1.0))
            sgn[b] = 1.0 if zax.y * side > 0 else -1.0             # + local X rotation swings the flap OUT
            rdir[b] = np.array((M3 @ Vector((0.0, 1.0, 0.0))).normalized())  # the joint's rest direction
        flaps.append(dict(bones=bones, side=side, R=R, fb=fb, kds=kds, sgn=sgn, rdir=rdir,
                          x0=R[:, 0].min() - 0.01, x1=R[:, 0].max() + 0.01,
                          zlo=[float(R[fb == k, 2].min()) if (fb == k).any() else 9.0 for k in range(len(bones))],
                          zhi=[float(R[fb == k, 2].max()) if (fb == k).any() else -9.0 for k in range(len(bones))],
                          MR=[np.array(rig.data.bones[b].matrix_local) for b in bones],
                          root_parent=rig.data.bones[bones[0]].parent.name,
                          rel=[np.linalg.inv(np.array(rig.data.bones[b].parent.matrix_local)) @ np.array(rig.data.bones[b].matrix_local)
                               for b in bones],
                          heads=[np.array(rig.data.bones[b].head_local) for b in bones]))
    if not flaps:
        return {}
    allb = [b for F in flaps for b in F["bones"]]
    for fc in list(rig.animation_data.drivers):
        if any(fc.data_path == 'pose.bones["%s"].rotation_euler' % b for b in allb):
            rig.animation_data.drivers.remove(fc)
    for b in allb:
        rig.pose.bones[b].rotation_mode = 'XYZ'
        rig.pose.bones[b].rotation_euler = (0.0, 0.0, 0.0)

    rep = {}
    for c in clips:
        act = bpy.data.actions[c]
        rig.animation_data.action = act
        nf = int(round(act.frame_range[1])) + 1
        frames = list(range(0, nf, step)) + ([nf - 1] if (nf - 1) % step else [])
        TH = {b: np.zeros(len(frames)) for b in allb}           # hang angles
        EX = {b: np.zeros(len(frames)) for b in allb}           # outward swings
        worst = 0.0
        for fi, f in enumerate(frames):
            for b in allb:
                rig.pose.bones[b].rotation_euler = (0.0, 0.0, 0.0)
            sc.frame_set(f)
            for b in allb:
                rig.pose.bones[b].rotation_euler = (0.0, 0.0, 0.0)
            bpy.context.view_layer.update()
            MWR = np.array(rig.matrix_world)
            H = hit_points()
            for F in flaps:
                bones = F["bones"]
                Mpar0 = MWR @ np.array(rig.pose.bones[F["root_parent"]].matrix)    # the chain's posed parent
                ext = np.zeros(len(bones))
                th0 = np.zeros(len(bones))
                for it in range(iters):
                    # top-down (numpy chain): each joint hangs, then swings out by the swing found so far
                    Mp = Mpar0
                    Mk = []
                    for k, b in enumerate(bones):
                        M0 = Mp @ F["rel"][k]
                        xw = M0[:3, 0] / np.linalg.norm(M0[:3, 0]); yw = M0[:3, 1] / np.linalg.norm(M0[:3, 1])
                        g = F["rdir"][b]; gp = g - xw * (g @ xw)
                        if np.linalg.norm(gp) > 1e-6:
                            gp = gp / np.linalg.norm(gp)
                            th0[k] = math.atan2(float(np.cross(yw, gp) @ xw), float(yw @ gp))
                        th = th0[k] + F["sgn"][b] * min(ext[k], DRAPE_MAX[k])
                        c_, s_ = math.cos(th), math.sin(th)
                        Rx = np.array([[1, 0, 0, 0], [0, c_, -s_, 0], [0, s_, c_, 0], [0, 0, 0, 1.0]])
                        M = M0 @ Rx
                        Mk.append(M); Mp = M
                    Hs = H[(H[:, 0] > F["x0"] - 0.05) & (H[:, 0] < F["x1"] + 0.05)] if len(H) else H
                    need = np.zeros(len(bones))
                    for k, b in enumerate(bones):
                        if F["kds"][k][1] == 0 or not len(Hs):
                            continue
                        Dk = Mk[k] @ np.linalg.inv(F["MR"][k])
                        Q = (Hs - Dk[:3, 3]) @ np.linalg.inv(Dk[:3, :3]).T
                        m = (Q[:, 2] > F["zlo"][k] - 0.01) & (Q[:, 2] < F["zhi"][k] + 0.01) & (Q[:, 0] > F["x0"]) & (Q[:, 0] < F["x1"])
                        hk = F["heads"][k]
                        for q in Q[m]:
                            near = F["kds"][k][0].find_range((q[0], 0.0, q[2]), 0.018)
                            if not near:
                                continue
                            if F["side"] < 0:
                                y_in = max(F["R"][n[1], 1] for n in near)
                                pen = (y_in + FLAP_MARGIN) - q[1]
                            else:
                                y_in = min(F["R"][n[1], 1] for n in near)
                                pen = q[1] - (y_in - FLAP_MARGIN)
                            if pen <= 0:
                                continue
                            lever = float(np.hypot(q[1] - hk[1], q[2] - hk[2]))
                            if lever < FLAP_LEVER_MIN:
                                continue
                            if it == 0:
                                worst = max(worst, pen)
                            need[k] = max(need[k], pen / lever)
                    if not need.any():
                        break
                    k0 = int(np.flatnonzero(need > 0)[0])
                    ext[k0] = ext[k0] + need[k0] * 1.1
                for k, b in enumerate(bones):
                    TH[b][fi] = th0[k]; EX[b][fi] = min(ext[k], DRAPE_MAX[k])
        loop = bool(table.get(c, {}).get("loop"))
        for F in flaps:
            for k, b in enumerate(F["bones"]):
                A = EX[b]
                if loop:
                    Af = np.maximum.reduce([np.roll(A, q) for q in (-1, 0, 1)])
                else:
                    Ap = np.concatenate([A[:1], A, A[-1:]])
                    Af = np.maximum.reduce([Ap[:-2], Ap[1:-1], Ap[2:]])
                Af = gauss1d(Af[None], 1.0, 1, wrap=loop)[0]
                tot = TH[b] + F["sgn"][b] * Af
                full = np.interp(np.arange(nf), frames, tot)
                if loop:
                    full[-1] = full[0]
                fc = act.fcurve_ensure_for_datablock(rig, 'pose.bones["%s"].rotation_euler' % b, index=0, group_name=b)
                fc.keyframe_points.clear(); fc.keyframe_points.add(nf)
                co = np.empty(nf * 2); co[0::2] = np.arange(nf); co[1::2] = full
                fc.keyframe_points.foreach_set("co", co)
                fc.keyframe_points.foreach_set("interpolation", [1] * nf)
                fc.update()
            rep.setdefault(c, {})["front" if F["side"] < 0 else "back"] = dict(
                swing_max_deg=[round(float(np.degrees(EX[b].max())), 1) for b in F["bones"]],
                hang_range_deg=[round(float(np.degrees(np.ptp(TH[b]))), 1) for b in F["bones"]])
        rep[c]["worst_pen_mm"] = round(worst * 1000, 1)
    rig.animation_data.action = act0
    for b in allb:
        rig.pose.bones[b].rotation_euler = (0.0, 0.0, 0.0)
    pose_reset(rig)
    log("flap drape: %s" % rep)
    rig["rts_flap_clearance"] = json.dumps(rep)
    return rep


def post_clips(rig, kind=None):
    """build_knight hook (after the clips, after rig_helpers.apply_plate_rules): re-copy the final plate weights onto
    the cloth / belts / clasps that rest on them (coskin), then the front flap's per-frame clearance (flap_clearance)"""
    rep = coskin(rig)
    try:
        rep["flap"] = flap_clearance(rig, kind)
    except Exception as e:                                   # never break the knight build
        import traceback; traceback.print_exc()
        log("WARNING flap clearance failed: %r" % e)
    return rep


def dress_upper(rig, bm, kind):
    """the upper armour under this kit (dress mode's full stack, as build_knight assembles it): armour_upper.dress,
    or the plain MPFB fit of its assets when that module is unavailable; {slot: object}"""
    try:
        import armour_upper as AU
        return AU.dress(kind, check_fit=False)
    except Exception as e:
        log("armour_upper.dress unavailable (%s): plain MPFB fit of the upper assets" % e)
    from outfit_lib import add_piece
    ensure_helpers(rig)
    out = {}
    man = os.path.join(KL_ASSETS, "knight_upper_manifest.json")
    for slot in (json.load(open(man))["slots"] if os.path.exists(man) else []):
        path = os.path.join(KL_ASSETS, "knight_" + slot, "knight_" + slot + ".mhclo")
        if os.path.exists(path):
            out[slot] = add_piece(rig, bm, path, slot=slot)
    return out


def load_lower(rig, bm, only=None, props=True):
    """Dress the live MPFB human with the lower-armour assets, extra bones, secondary drivers and props."""
    from outfit_lib import add_piece
    kind = rig.get("rts_kind", rig.name.replace("rts_", ""))
    pose_reset(rig)
    lst = json.load(open(os.path.join(KL_ASSETS, "rts_knight_lower.json")))["order"]
    order = [n for n in lst if not only or n in only]
    meta = {n: json.load(open(os.path.join(KL_ASSETS, "rts_knight_" + n, "rts_knight_" + n + ".rts.json"))) for n in order}
    # chain bones must exist before MPFB loads the .mhw weights (placeholders now, fitted after loading)
    pel = np.array(rig.data.bones["pelvis"].head_local)
    ensure_helpers(rig)                                      # knee / hip helpers carry poleyn cops + tasset lames
    pre = {}
    for n in order:
        if "bones" in meta[n]:
            pre.update(chain_bone_specs(meta[n]["bones"], None, pel))
    if pre:
        ensure_bones(rig, pre)
    objs = {}
    for n in order:
        t0 = time.time()
        path = asset_file("clothes", "rts_knight_" + n, "rts_knight_" + n + ".mhclo")
        pc = add_piece(rig, bm, path, slot=n)                # MPFB purges unused materials while loading
        tmp = [m for m in pc.data.materials if m]
        apply_slots(pc, meta[n])
        for m in tmp:
            if m.users == 0:
                bpy.data.materials.remove(m)
        bpy.context.view_layer.objects.active = pc
        for o in bpy.context.view_layer.objects:
            o.select_set(o == pc)
        try:
            bpy.ops.mesh.customdata_custom_splitnormals_clear()
        except Exception:
            pass
        pc["rts_group"] = "knight_lower"
        if meta[n].get("rigid_weights"):
            pc["rts_weights"] = "authored"               # rig_helpers.apply_plate_rules leaves authored plates alone
        objs[n] = pc
        groups = {g.name for g in pc.vertex_groups}
        log("dress %-11s %5d verts  %d groups  %.1fs" % (n, len(pc.data.vertices), len(groups), time.time() - t0))
    # rebuild the rest shapes on this body over the upper armour dressed on it (female / proportion sizing)
    _TIGHTS.clear()
    _TIGHTS.update({n: meta[n]["tights_faces"] for n in order if meta[n].get("tights_faces")})
    regen_on_body(rig, bm, objs)
    # fit the chain bones to the fitted cloth, add sockets
    specs = {}
    for n in order:
        if "bones" in meta[n]:
            specs.update(chain_bone_specs(meta[n]["bones"], piece_basis(objs[n])))
    specs.update(socket_specs(rig, objs.get("belts"), meta.get("belts")))
    ensure_bones(rig, specs)
    refit_shield_handle(rig)
    for n in ("boots", "sabatons"):
        if n in objs:
            floor_clamp(objs[n])
    sheet = [o for o in objs.values() if any(m and m.get("rts_slot") == "armour" for m in o.data.materials)]
    if sheet:
        bake_vertex_ao(sheet, rig, bm)
    add_secondary(rig)
    rig["rts_sockets"] = json.dumps({k: v["parent"] for k, v in specs.items() if k.startswith("socket_")})
    coskin(rig)
    if props:
        add_props(rig, which=("sword", "shield"))       # the empty scabbard is not worn (see STATUS); its GLB is exported
    return objs


PROP_SOCKET = {"sword": "socket_weapon_r", "shield": "socket_shield_l", "scabbard": "socket_scabbard_l"}


def refit_shield_handle(rig):
    """(iteration 2b, integrity G4 gauntlet_l|shield) the handle from the socket bones AS POSED: the clips hold the
    shield hand at knight_anim's wrist delta (rest space, 6 deg about -X, measured in every clip) on the forearm, so
    the forearm is left at rest, hand_l gets that delta, and socket_hand_l's head / +Y are read in socket_shield_l's
    frame (the analytic fit was 7.7 deg off the fist axis in the clips: the handle's ends stood 5-17 mm off it).
    Updates rig extras rts_shield_fit (used by add_props / export_props)."""
    try:
        f = json.loads(rig["rts_shield_fit"])
    except Exception:
        return None
    if not all(n in rig.pose.bones for n in ("socket_hand_l", "socket_shield_l", "hand_l")):
        return None
    pose_reset(rig)
    pb = rig.pose.bones["hand_l"]
    Rw = Matrix(shield_hand_rot().tolist()).to_4x4()
    M = pb.matrix.copy()
    T = Matrix.Translation(M.translation)
    pb.matrix = T @ Rw @ T.inverted() @ M
    bpy.context.view_layer.update()
    S = rig.pose.bones["socket_shield_l"].matrix; H = rig.pose.bones["socket_hand_l"].matrix
    L = S.inverted() @ H
    loc = np.array(L.translation); axl = np.array(L.to_3x3().col[1]); axl = axl / np.linalg.norm(axl)
    pose_reset(rig)
    old = np.array(f["handle"][1], float)
    f["handle"] = [loc.tolist(), axl.tolist()]
    rig["rts_shield_fit"] = json.dumps(f)
    log("shield handle refit (posed): axis %.1f deg from the analytic fit, head %s" % (
        math.degrees(math.acos(float(np.clip(old @ axl, -1, 1)))), np.round(loc, 4).tolist()))
    return f


def shield_fit(rig):
    """the per-body shield handle / forearm-strap fit (socket_specs, rig extras rts_shield_fit) or None"""
    try:
        f = json.loads(rig["rts_shield_fit"]) if rig is not None and rig.get("rts_shield_fit") else None
    except Exception:
        f = None
    if not f:
        return None
    return dict(handle=(np.array(f["handle"][0]), np.array(f["handle"][1])),
                loop=(np.array(f["loop"][0]), np.array(f["loop"][1]), float(f["loop"][2])))


def add_props(rig, which=("sword", "shield", "scabbard")):
    """Props as meshes skinned 100 % to their socket bone (placed in the socket frame), children of the rig so the
    dressed GLB shows them in hand; the stand-alone prop GLBs are written by export_props()."""
    import armour_lower_props as PR
    kind = rig.get("rts_kind", "male")
    built = PR.build_all(shield_fit(rig))
    out = {}
    for n in which:
        m = built[n]
        if m.mixed():
            m.triangulate()
        sock = PROP_SOCKET[n]
        M = rig.matrix_world @ rig.data.bones[sock].matrix_local           # socket frame (bone head, Y along bone)
        mm = Mesh(); mm.V = [tuple(M @ Vector(v)) for v in m.V]; mm.F, mm.UV, mm.M = m.F, final_uvs(m), m.M
        mm.tag, mm.tagw = m.tag, m.tagw
        ob = mm.to_object("%s_%s" % (kind, n))
        used = sorted(set(m.M))
        for u in used:
            ob.data.materials.append(slot_material(SLOTS[u]))
        ob.data.polygons.foreach_set("material_index", np.array([used.index(i) for i in m.M], dtype=np.int32))
        ob.parent = rig
        vg = ob.vertex_groups.new(name=sock); vg.add(list(range(len(ob.data.vertices))), 1.0, 'REPLACE')
        md = ob.modifiers.new("Armature", 'ARMATURE'); md.object = rig
        ob["rts_part"] = n; ob["rts_prop"] = True; ob["rts_socket"] = sock; ob["rts_group"] = "props"
        out[n] = ob
        log("prop %-8s tris %5d on %s" % (n, sum(len(f) - 2 for f in m.F), sock))
    return out


def export_props(path_dir, rig=None, only=None):
    """Stand-alone prop GLBs in the socket frame: the prop root node is the attach point, identity offset to the
    socket joint of the character GLB. The Blender glTF exporter keeps joint frames in Blender bone axes (+Y along
    the bone; only the root gets the Y-up conversion), so the prop vertices are pre-rotated by Rx(+90 deg) to cancel
    the exporter's Y-up conversion: the GLB then stores them in the socket's own axes (checked numerically:
    identity-attached props land on the dressed GLB's props to 0.0 mm)."""
    R = Matrix.Rotation(math.radians(90.0), 3, 'X')
    import armour_lower_props as PR
    os.makedirs(path_dir, exist_ok=True)
    kind = rig.get("rts_kind", "male") if rig is not None else "male"
    for n, m in PR.build_all(shield_fit(rig)).items():
        if only and n not in only:
            continue
        for o in list(bpy.data.objects):
            o.select_set(False)
        if m.mixed():
            m.triangulate()
        mm = Mesh(); mm.F, mm.UV, mm.M, mm.tag, mm.tagw = m.F, final_uvs(m), m.M, m.tag, m.tagw
        mm.V = [tuple(R @ Vector(v)) for v in m.V]
        ob = mm.to_object("knight_" + n)
        used = sorted(set(m.M))
        for u in used:
            ob.data.materials.append(slot_material(SLOTS[u]))
        ob.data.polygons.foreach_set("material_index", np.array([used.index(i) for i in m.M], dtype=np.int32))
        ob["rts_prop"] = n; ob["rts_socket"] = PROP_SOCKET[n]
        ob.select_set(True); bpy.context.view_layer.objects.active = ob
        # the shield's handle / forearm strap are fitted per body: knight_shield.glb (male), knight_shield_<kind>.glb
        p = os.path.join(path_dir, "knight_%s.glb" % n if (n != "shield" or kind == "male") else "knight_%s_%s.glb" % (n, kind))
        bpy.ops.export_scene.gltf(filepath=p, export_format='GLB', use_selection=True, export_yup=True,
                                  export_apply=True, export_tangents=True, export_extras=True, export_materials='EXPORT',
                                  export_image_format='AUTO')
        bpy.data.objects.remove(ob, do_unlink=True)
        log("prop glb", p, "%.2f MB" % (os.path.getsize(p) / 1e6))


# ------------------------------------------------------------------------------------------------ preview (dev)
def preview_objects(B, pieces):
    cols = {"plate": (0.62, 0.64, 0.68, 1), "mail": (0.25, 0.26, 0.28, 1), "leather": (0.35, 0.2, 0.1, 1),
            "cloth": (0.1, 0.18, 0.6, 1)}
    obs = []
    for name, m, kind in pieces:
        ob = m.to_object(name)
        mat = bpy.data.materials.get("pv_" + kind) or bpy.data.materials.new("pv_" + kind)
        mat.diffuse_color = cols.get(kind, (0.5, 0.5, 0.5, 1)); ob.data.materials.append(mat)
        obs.append(ob)
    return obs


def build_leg_pieces(B):
    """cuisses / poleyns / greaves / sabatons: hinge-layered rigid plates (scripts/armour_lower_legs.py)"""
    import armour_lower_legs as LG
    out, B.legs = LG.build_legs(B)
    return out


UPPER_LAYER_SLOTS = ("cuirass", "mail", "gorget", "pauldron_l", "pauldron_r")
# inner layers pushed under the rigid plates over them at rest (gap m): the leather boot under the sabaton lames and the
# greave's flare, the mail chausses under the cuisses / poleyns / greaves, the mail skirt under the tassets
UNDER = [("boots", ("sabatons", "greaves"), 0.0035), ("legs_mail", ("cuisses", "poleyns", "greaves"), 0.0030),
         ("mail_skirt", ("tassets",), 0.0030)]


def push_over(m, inners, body, gap=0.0015, reach=0.03, rigid_small=200, centre=(0.0, 0.015)):
    """Keep the Mesh m OUTSIDE the layers under it (list of Meshes): points closer than `gap` over (or inside) them move
    out along the layer normal (oriented away from the body); islands with fewer than `rigid_small` vertices (buckles,
    pouches) move rigidly by their largest push, big ones (belt rings) per vertex, smoothed. In place."""
    from mathutils.bvhtree import BVHTree
    V, Fo = [], []
    for mm in inners:
        o = len(V); V += [v3(v) for v in mm.V]; Fo += [tuple(i + o for i in f) for f in mm.F]
    if not Fo:
        return m
    bvh = BVHTree.FromPolygons(V, Fo)
    P = np.array(m.V, float)
    D = np.zeros_like(P)
    # radial test from OUTSIDE (the outermost surface of the layers along the horizontal ray through the point: a
    # nearest-point test picks a cloth's lining face and lets a point sit inside the cloth)
    for i, p in enumerate(P):
        d = np.array([p[0] - centre[0], p[1] - centre[1], 0.0]); ln = np.linalg.norm(d)
        if ln < 1e-6:
            continue
        d /= ln
        h = bvh.ray_cast(v3(p + d * reach), v3(-d), reach + 0.08)
        if h[0] is None:
            continue
        s_ = (p - np.array(h[0])).dot(d)
        if s_ < gap:
            D[i] = d * (gap - s_)
    # islands
    par = list(range(len(P)))

    def f(i):
        while par[i] != i:
            par[i] = par[par[i]]; i = par[i]
        return i
    for fc in m.F:
        for a in fc[1:]:
            ra, rb = f(fc[0]), f(a)
            if ra != rb:
                par[ra] = rb
    groups = {}
    for i in range(len(P)):
        groups.setdefault(f(i), []).append(i)
    for g in groups.values():
        if len(g) < rigid_small:
            k = int(np.argmax(np.linalg.norm(D[g], axis=1)))
            D[g] = D[g][k]
    m.V = [tuple(v) for v in P + D]
    return m


def push_under(P, F, outers, body, gap=0.003, reach=0.025, smooth=2):
    """Move the points P (n, 3) of an inner layer (faces F for smoothing) to at least `gap` under the outer layers
    (list of Meshes) wherever they are within `reach` of them: signed distance along the outer surface normal,
    oriented away from the body. The displacement is smoothed over the inner mesh (no creases). Returns new P."""
    from mathutils.bvhtree import BVHTree
    V, Fo = [], []
    for m in outers:
        o = len(V); V += [v3(v) for v in m.V]; Fo += [tuple(i + o for i in f) for f in m.F]
    if not Fo:
        return P
    bvh = BVHTree.FromPolygons(V, Fo)
    P = np.asarray(P, float)
    D = np.zeros_like(P); hit = np.zeros(len(P), bool)
    for i, p in enumerate(P):
        q, n, fi, dist = bvh.find_nearest(v3(p), reach)
        if q is None:
            continue
        q = np.array(q); n = np.array(n)
        bq = body.closest(q)[0]
        if n.dot(q - bq) < 0:
            n = -n
        sd = (p - q).dot(n)
        if sd > -gap:
            D[i] = -n * (sd + gap); hit[i] = True
    if hit.any() and smooth:
        adj = [[] for _ in range(len(P))]
        for f in F:
            for a in range(len(f)):
                adj[f[a]].append(f[(a + 1) % len(f)]); adj[f[(a + 1) % len(f)]].append(f[a])
        for _ in range(smooth):
            D2 = D.copy()
            for i in range(len(P)):
                if adj[i]:
                    nb = D[adj[i]]
                    # keep the full push where it is needed (never less than the vertex's own push)
                    avg = (D[i] + nb.sum(0)) / (1 + len(nb))
                    D2[i] = avg if np.linalg.norm(avg) > np.linalg.norm(D[i]) else D[i]
            D = D2
    return P + D


# upper-armour slots per layer role (the upper kit may split its cuirass into breast / back plates, plackart, faulds)
UPPER_ROLES = {"cuirass": ("cuirass", "breast", "backplate", "plackart", "fauld"), "gorget": ("gorget",),
               "mail": ("mail",), "pauldron_l": ("pauldron_l",), "pauldron_r": ("pauldron_r",)}


def upper_role(slot):
    if not slot or slot.startswith(("gorget_top",)):
        return None
    for role, keys in UPPER_ROLES.items():
        if slot == role or any(k in slot for k in keys):
            if role in ("pauldron_l", "pauldron_r") or not slot.endswith(("_l", "_r")) or role == "cuirass":
                return role
    return None


def upper_layers_live(rig):
    """the upper armour dressed on this rig (rest shapes, world space) as Meshes per layer role, {role: Mesh}: the
    layers this kit drapes over when it regenerates on a body (the upper armour regenerates its own pieces per body)"""
    out = {}
    for o in children_meshes(rig):
        sl = o.get("rts_part")
        role = upper_role(sl)
        if role is None or o.get("rts_group") == "knight_lower" or o.get("rts_prop"):
            continue
        co = piece_basis(o); M = np.array(o.matrix_world)
        W = co @ M[:3, :3].T + M[:3, 3]
        m = Mesh(); m.V = [tuple(map(float, v)) for v in W]; m.F = [tuple(p.vertices) for p in o.data.polygons]
        m.UV = [[(0, 0)] * len(f) for f in m.F]; m.M = [0] * len(m.F); m.tag = [0] * len(m.V); m.tagw = [None] * len(m.V)
        if role in out:
            out[role].merge(m)
        else:
            out[role] = m
    return out


def build_all_pieces(B, upper=None):
    """every lower piece on body B, over the upper armour layers `upper` ({slot: Mesh}; default: the upper kit's
    authored OBJs, i.e. the authoring male)"""
    t0 = time.time()
    up = upper if upper is not None else {k: upper_layer(k) for k in UPPER_LAYER_SLOTS}
    B.upper = {k: v for k, v in up.items() if v is not None}
    cuir = [B.upper[k] for k in ("cuirass", "mail") if k in B.upper]
    bt = boots(B)
    B.boot_mesh = bt                          # (2b) the sabatons are shaped over the boot itself, not skin + a constant
    out = build_leg_pieces(B)
    out.append(("legs_mail", legs_mail(B), "mail"))
    out.append(("boots", bt, "leather"))
    H = Hips(B)
    B.waist = waist_band(B)                 # the waist belt's band under the cuirass rim (iteration 2b, user item 27)
    sk = mail_skirt(H); out.append(("mail_skirt", sk, "mail"))
    # (2b session 3, G4 mail | mail_skirt at bind) the side panels lie OVER the mail shirt's hem, which hangs below
    # the cuirass rim over the hips
    if "mail" in B.upper:
        push_over(sk, [B.upper["mail"]], B.s, gap=0.003, rigid_small=0)
    # (2b session 3) the drapes keep the old closed skirt as a layer: the flaps hang where they did over the skirt's
    # front / back (now removed under them), i.e. with the same clearance over the chausses and cuisses
    skf = mail_skirt(H, full=True)
    tm = tassets(H)
    push_over(tm, [sk], B.s, gap=0.0015, rigid_small=10 ** 6)     # (2b s3) the left set as ONE rigid move over the skirt
    tm.merge(mirror_mesh(tm)); out.append(("tassets", tm, "plate"))    # as it is (mail hem push); lames keep their order
    # (2b, integrity G4 poleyns|tabard at bind / cape|poleyns: the drapes only knew the trunk and the hip layers, so the
    # front flap's hem lay through the knee cops) the cuisses and poleyns are layers of the drapes too
    legs = [m for n, m, k in out if n in ("cuisses", "poleyns")]
    D = Drape(B, H, layers=[skf, tm] + cuir + legs)
    tb, tinfo = tabard(B, H, D, over_cuirass="cuirass" in B.upper); out.append(("tabard", tb, "cloth"))
    B.tinfo = tinfo

    H.resample([sk, tm, tb] + cuir)
    bl = belts(H)
    push_over(bl, [sk, tm, tb] + cuir, B.s, gap=0.0015)       # user 21: seated, never inside what it is buckled over
    out.append(("belts", bl, "leather"))
    D2 = Drape(B, H, layers=[skf, tm, tb, bl] + cuir + legs)
    B.cape_under = [tb, bl]
    bank = {n: m for n, m, k in out}
    for inner, outs, gap in UNDER:
        if inner in bank:
            mi = bank[inner]
            V_ = np.array(mi.V)
            if inner == "legs_mail":
                # (2b s3, female: the knees' and hips' skin stood 1-5 mm OUTSIDE the chausses at rest, the assembly's gap
                # scan cut filler mail there that then ran through the chausses) first cover the skin (2 mm), then
                # the plates keep them under (where both cannot hold, the plate over it hides the skin)
                Nl = np.zeros_like(V_)
                for f in mi.F:
                    fn = np.cross(V_[f[1]] - V_[f[0]], V_[f[2]] - V_[f[0]])
                    for i in f:
                        Nl[i] += fn
                Nl /= np.maximum(np.linalg.norm(Nl, axis=1, keepdims=True), 1e-12)
                V_ = boot_cover(B, V_, mi.F, Nl, clr=0.002, reach=0.012, zmin=float(V_[:, 2].min()) + 0.04,
                                label="legs_mail")
            # (2b) 3 passes: the plates are closed shells now, a point caught between a shell's skins is first pushed
            # under the outer skin and then, nearest to the inner skin, under that
            for _ in range(3):
                V_ = push_under(V_, mi.F, [bank[o] for o in outs if o in bank], B.s, gap=gap)
            # (2b, G4 boots|underlayer) push_under must never press the leather under the skin (it pushed the boot
            # up to 29 mm in round the sabatons' toe cap, and the smoothing carried it to the uncovered big toe)
            if inner == "boots" and getattr(mi, "upper_part", None):
                nu, Fu, Nu = mi.upper_part
                Pu = boot_cover(B, V_[:nu], Fu, Nu)
                Pu[:, 2] = np.maximum(Pu[:, 2], 0.0137)
                V_[:nu] = Pu
            mi.V = [tuple(v) for v in V_]
    cp, cinfo = cape(B, H, D2, over_cuirass="cuirass" in B.upper); out.append(("cape", cp, "cloth")); B.cinfo = cinfo
    out.append(("clasps", clasps(B, cinfo), "plate"))
    log("built %d pieces in %.1fs (upper layers: %s)" % (len(out), time.time() - t0, sorted(B.upper)))
    return out, H, D2


if __name__ == "__main__":
    args = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    mode = args[0] if args else "preview"
    if mode == "author":
        rig = [o for o in bpy.data.objects if o.type == 'ARMATURE'][0]
        kind = rig.get("rts_kind", "male")
        t0 = time.time()
        author(rig, bpy.data.objects[kind + "_body"], only=args[1:] or None)
        log("author done in %.1fs" % (time.time() - t0))
    if mode == "dress":
        kind = args[1] if len(args) > 1 else "male"
        only = [a for a in args[2:] if not a.startswith("--")] or None
        rig = bpy.data.objects["rts_" + kind]; bm = bpy.data.objects[kind + "_body"]
        t0 = time.time()
        up = {} if "--no-upper" in args else dress_upper(rig, bm, kind)
        objs = load_lower(rig, bm, only=only)
        if up:                                             # the knight build's plate rules, then the co-skin hook
            import rig_helpers as RH
            RH.apply_plate_rules(rig, {o["rts_part"]: o for o in children_meshes(rig)
                                       if o.get("rts_part") and not o.get("rts_prop")}, kind, log)
            post_clips(rig, kind)
        tris = {n: sum(len(p.vertices) - 2 for p in o.data.polygons) for n, o in objs.items()}
        log("dressed %s: %d pieces, %d tris %s (%.1fs)" % (kind, len(objs), sum(tris.values()), tris, time.time() - t0))
        blend = os.path.join(OUT, "knight_lower.blend" if kind == "male" else "knight_lower_%s.blend" % kind)
        bpy.ops.file.make_paths_relative()
        bpy.ops.wm.save_as_mainfile(filepath=blend, relative_remap=True)
        log("saved", blend)
        if "--export" in args:
            from outfit_lib import export_dressed
            res = export_dressed(rig, bm, kind, os.path.join(OUT, "knight_lower_%s.glb" % kind), drop=("hair",) if up else ())
            log("export", res)
            bpy.ops.wm.save_as_mainfile(filepath=os.path.join(OUT, "knight_lower_%s_export.blend" % kind))
            if kind == "male":
                export_props(os.path.join(OUT, "props"))
    if mode == "props":
        # on a base human (out/base_<kind>.blend): the sockets (per-body shield fit) first; without one: male defaults
        rigs = [o for o in bpy.data.objects if o.type == 'ARMATURE']
        rig = rigs[0] if rigs else None
        if rig is not None:
            pose_reset(rig)
            ensure_bones(rig, socket_specs(rig))
            refit_shield_handle(rig)
        kind = rig.get("rts_kind", "male") if rig is not None else "male"
        export_props(os.path.join(OUT, "props"), rig, only=None if kind == "male" else ("shield",))
    if mode == "preview":
        rig = [o for o in bpy.data.objects if o.type == 'ARMATURE'][0]
        kind = rig.get("rts_kind", "male")
        bm = bpy.data.objects[kind + "_body"]
        pose_reset(rig)
        B = Body(rig, bm)
        pieces, H, D = build_all_pieces(B)
        for n, m, k in pieces:
            log("piece", n, "verts", len(m.V), "faces", len(m.F), "tris", sum(len(f) - 2 for f in m.F))
        preview_objects(B, pieces)
        out = args[1] if len(args) > 1 else os.path.join(REN, "lower_preview")
        render_setup("BLENDER_WORKBENCH", res=(900, 1300))
        sc = bpy.context.scene; sc.display.shading.light = 'STUDIO'; sc.display.shading.color_type = 'MATERIAL'
        sc.display.shading.show_cavity = True
        for o in rig.children:
            if o.type == 'MESH' and o.get("rts_variant_group") == "eyebrows" and not o.get("rts_default"):
                o.hide_render = True
        for nm, loc, tgt, lens in (("front", (0, -2.6, 0.55), (0, 0, 0.5), 50), ("side", (2.6, -0.3, 0.55), (0, 0, 0.5), 50),
                                   ("34", (1.3, -1.9, 0.75), (0.12, 0, 0.45), 50), ("knee", (0.45, -0.9, 0.62), (0.16, 0, 0.5), 50),
                                   ("back", (0.3, 2.6, 0.55), (0, 0, 0.5), 50), ("foot", (0.6, -0.8, 0.25), (0.2, -0.08, 0.08), 50),
                                   ("full", (0.0, -4.2, 1.0), (0, 0, 0.95), 50), ("fullback", (0.0, 4.2, 1.0), (0, 0, 0.95), 50),
                                   ("fullside", (4.2, 0.0, 1.0), (0, 0, 0.95), 50), ("hips", (0.9, -1.4, 1.05), (0.05, 0, 0.9), 50)):
            camera(loc, tgt, lens=lens)
            render("%s_%s.png" % (out, nm))
