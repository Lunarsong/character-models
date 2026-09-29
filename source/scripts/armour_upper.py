"""Knight UPPER armour (reference: refs/knight_sheet.png), procedural, authored on the live male base human and written
as MPFB clothes assets (.mhclo + .obj + .mhmat + rigid .mhw weights + body delete groups), so every piece fits any
body / customisation and is skinned to the shared rts_human skeleton (+ the helper joints of armour_upper_rig.py).

Iteration 2 (judge C1-C3, M1, M4-M7, M14, user items 12, 14, 16, 17, 19, 20): every plate component is 100 % on ONE
bone, articulated plates are built as families of surfaces around their common pivot so they slide instead of
cutting each other, and the knight's presets are REGENERATED on their own body at dress time (same topology as the
asset, so the MPFB correspondence still carries the cust_* morphs): the female gets her own helm, pauldrons, cuffs
and gorget instead of the male-sized ones.

Pieces (slot = asset knight_<slot>):
  helmet        slim close helm: skull + pointed prow visor (V keel, gold centre ridge), eye slot in shadow, 3+3
                breaths, gold comb, flared lower rim, pivot rosettes, short plume socket; dark inner lining (inward
                faces: seen through the slot / vents and from below)                                     -> head
  plume         horsehair crest: 3 layers of tapered V-section clumps along a 0.45 m arched spine    -> plume_01..03
  gorget        lame A over the breastplate's neck roll (spine_05) + lame B round the neck (neck_02), B telescoped
                inside A with the overlap at the neck joints; B's top edge is SOLVED below the chin / helmet rim
                over a head pose grid (pitch -20..+25, turn +-45, jawOpen 0 / 1)
  cuirass       globose breastplate (gently curved neck opening, turned roll, stop-rib, arm gussets) + backplate
                (spine_05); plackart + lower back plate (spine_03) overlapping it
  pauldron_l/r  cop (upperarm_helper_01) + 3 stepped lames (helper_02, helper_03, upperarm): concentric shells
                around the shoulder joint (each lame 6.5 mm inside the one above), stop-rib, lion, rivets
  rerebrace_l/r upper-arm plate (upperarm), top tucked under the last lame, bottom under the couter
  couter_l/r    elbow cop with a fan wing + rosette (lowerarm_helper = half the elbow swing), a sphere around the elbow
  vambrace_l/r  two cannons: upper (lowerarm) over lower (lowerarm_twist_01), round where they overlap (twist slides)
  gauntlet_l/r  leather glove (body-weighted) + hourglass cuff (lowerarm_twist_01) + 4 metacarpal lames and a knuckle
                ridge (hand) + 3 scales per finger and 3 thumb plates (100 % per phalanx)
  mail          hauberk to the hips, sleeves to the WRIST (cut under the cuff, arm / twist weights only), collar
Layering (user items 14 / 19): settle_layers() runs after generation on every body: a skin floor for the plates, then
ordered clearance rules (outer plate pushed out / inner layer pulled in, mail down to 2 mm over the skin) with a
reverse pass for rims and lips; per-rule report in the log. Grip poses (C2): post_clips() solves grip_r / grip_l.

run:  BLENDER_USER_RESOURCES=$PWD/blender_profile $BL -b out/base_male.blend --python-exit-code 1 \
        -P scripts/armour_upper.py -- <mode> [pieces...]
modes: preview  build on the male, clay previews (renders/armour_upper_prev_*.png)
       author   build + write the MPFB clothes assets (assets/mpfb_assets/clothes/knight_*/, installed into MPFB)
       dress    (on out/base_<kind>.blend) load the pieces (regenerated on that body), save out/knight_upper[_female]
                .blend, export out/test/knight_upper_<kind>.glb
       render / poses  look-dev and pose renders on a dressed file
"""
import sys, os, json, math, time, uuid, shutil
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from chr_lib import *
import armour_upper_geo as G
import armour_upper_tex as TX
import armour_upper_rig as RG

ARGS = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
MODE = ARGS[0] if ARGS else "preview"
ONLY = set(ARGS[1:])
# RTS_UPPER_ASSETS=<dir>: author into / dress from a private asset dir (trial builds: the installed assets that other
# agents read stay untouched). Without it, `author` writes staging dirs and swaps each piece in with a rename.
ASSET_OVR = os.environ.get("RTS_UPPER_ASSETS")
ASSET_DIR = ASSET_OVR or os.path.join(ASSETS, "mpfb_assets", "clothes")
PREFIX = "knight_"


def piece_file(name, ext=".mhclo"):
    """installed file of piece `name` (the MPFB user-data copy, or the RTS_UPPER_ASSETS dir)"""
    if ASSET_OVR:
        return os.path.join(ASSET_OVR, PREFIX + name, PREFIX + name + ext)
    return asset_file("clothes", PREFIX + name, PREFIX + name + ext)


def swap_dir(src, dst):
    """install directory src as dst with renames (a reader sees the old or the new piece, never a half-written one)"""
    old = dst + ".old_%d" % os.getpid()
    if os.path.exists(dst):
        os.rename(dst, old)
    os.rename(src, dst)
    if os.path.exists(old):
        shutil.rmtree(old)
D2R = math.pi / 180
GAP = 0.0065            # radial step between articulated plates (plate 2-3 mm + lip + 3 mm clearance)
PAUL_GAP = 0.0052       # pauldron lame step (iteration 2b, closed shells): plate 2.5 mm + 2.7 mm clearance; the lames'
                        # hidden top edges are flat (no bead under the plate above). Was 8.5 mm: item 25 'bulky'
PAUL_T = 0.0025         # pauldron plate thickness
MAIL_OFF = 0.0085       # mail shirt offset over the skin
RERE_OFF = MAIL_OFF + 0.0062   # rerebrace outer surface over the skin: mail + 4 mm + plate (item 25: was mail + 9.5 mm)
M_Z3, M_ZN = 1.1714, 1.5990        # male spine_03 / neck_01 head heights: torso design heights are mapped from these


def log_(*a):
    print("ARM", *a, flush=True)


# ------------------------------------------------------------------------------------------------------ context
class Ctx:
    """Body landmarks and fitting fields of the live human `kind` (every generator works on any body)."""

    def __init__(self, kind="male"):
        self.kind = kind
        self.rig = bpy.data.objects["rts_" + kind]
        self.bm = bpy.data.objects[kind + "_body"]
        self.body = G.Body(self.bm)
        self.bone = {b.name: (np.array(b.head_local[:]), np.array(b.tail_local[:])) for b in self.rig.data.bones}
        co = self.body.co
        headw = self.body.weight_sum(["head", "jaw", "eye", "brow", "cheek", "nose", "lip", "mouth", "tongue"])
        ears = np.zeros(self.body.n, bool); ears[self.body.groups.get("ears", [])] = True
        hv = (headw > 0.5)
        self.headmask = hv
        self.head_top = float(co[hv, 2].max())
        self.eye = 0.5 * (self.head("eye_l") + self.head("eye_r"))
        ez = self.eye[2]
        band = co[hv & ~ears & (np.abs(co[:, 2] - (ez + 0.02)) < 0.02) & (np.abs(co[:, 0]) < 0.05)]
        self.head_front, self.head_back = float(band[:, 1].min()), float(band[:, 1].max())
        tb = co[hv & ~ears & (np.abs(co[:, 2] - (ez + 0.035)) < 0.015)]
        self.head_hw = float(np.abs(tb[:, 0]).max())
        chin = co[hv & (np.abs(co[:, 0]) < 0.015) & (co[:, 1] < self.eye[1])]
        self.chin_z = float(chin[:, 2].min())
        self.nose_y = float(co[hv & (np.abs(co[:, 0]) < 0.01)][:, 1].min())
        self.z3 = self.head("spine_03")[2]; self.zn = self.head("neck_01")[2]
        armw = self.body.weight_sum(["upperarm", "lowerarm", "hand", "thumb", "index", "middle", "ring", "pinky"])
        self.torso = self.body.subset(armw < 0.35)          # trunk only (arm-hole rays pass through the arms)
        self.shared = {}
        log_("ctx %s: head top %.3f eye z %.3f front %.3f back %.3f hw %.3f chin %.3f nose %.3f" % (
            kind, self.head_top, ez, self.head_front, self.head_back, self.head_hw, self.chin_z, self.nose_y))

    def head(self, b):
        return self.bone[b][0]

    def tail(self, b):
        return self.bone[b][1]

    def tz(self, zm):
        """a torso height designed on the male (metres) -> this body (linear between spine_03 and neck_01)"""
        return self.z3 + (np.asarray(zm, float) - M_Z3) * (self.zn - self.z3) / (M_ZN - M_Z3)

    def tscale(self):
        return (self.zn - self.z3) / (M_ZN - M_Z3)


def side_sign(side):
    return 1.0 if side == "l" else -1.0


class Sph:
    """Spherical coordinates around pivot O: gamma = angle from axis u, alpha = azimuth (0 along `lat`, +90 along fw)."""

    def __init__(self, O, u, lat_hint, fw_sign=1.0):
        self.O = np.asarray(O, float)
        self.u = G.nrm(u)
        l = np.asarray(lat_hint, float); l = l - np.dot(l, self.u) * self.u
        self.lat = G.nrm(l)
        self.fw = G.nrm(np.cross(self.u, self.lat)) * fw_sign

    def d(self, g, a):
        g = np.asarray(g, float); a = np.asarray(a, float)
        g, a = np.broadcast_arrays(g, a)
        return (np.cos(g)[..., None] * self.u + np.sin(g)[..., None] *
                (np.cos(a)[..., None] * self.lat + np.sin(a)[..., None] * self.fw))

    def pt(self, g, a, r):
        return self.O + np.asarray(r, float)[..., None] * self.d(g, a)

    def ga(self, p):
        v = np.asarray(p, float) - self.O
        r = np.linalg.norm(v, axis=-1)
        g = np.arccos(np.clip(np.sum(v * self.u, -1) / np.maximum(r, 1e-9), -1, 1))
        a = np.arctan2(np.sum(v * self.fw, -1), np.sum(v * self.lat, -1))
        return g, a, r


def skin_radius(body, sph, G_, A_, maxd=0.45):
    """outermost body surface distance from sph.O along directions (gamma grid x alpha grid), misses filled"""
    R = np.full((len(G_), len(A_)), np.nan)
    for i, g in enumerate(G_):
        for j, a in enumerate(A_):
            d = sph.d(g, a)
            r = body.ray(sph.O + d * maxd, -d, maxd)
            if r is not None:
                R[i, j] = maxd - r
    for _ in range(60):
        bad = np.isnan(R)
        if not bad.any():
            break
        Rp = np.pad(R, 1, mode="edge")
        nb = np.stack([Rp[:-2, 1:-1], Rp[2:, 1:-1], Rp[1:-1, :-2], Rp[1:-1, 2:]])
        fill = np.nanmean(np.where(np.isnan(nb), np.nan, nb), axis=0)
        R[bad] = fill[bad]
    R[np.isnan(R)] = 0.05
    return R


def interp2(Gs, As, F, g, a):
    """bilinear lookup of a (gamma, alpha) field"""
    g = np.clip(np.asarray(g, float), Gs[0], Gs[-1]); a = np.clip(np.asarray(a, float), As[0], As[-1])
    fi = (g - Gs[0]) / (Gs[-1] - Gs[0]) * (len(Gs) - 1); fj = (a - As[0]) / (As[-1] - As[0]) * (len(As) - 1)
    i0 = np.clip(np.floor(fi).astype(int), 0, len(Gs) - 2); j0 = np.clip(np.floor(fj).astype(int), 0, len(As) - 2)
    x = fi - i0; y = fj - j0
    return ((1 - x) * ((1 - y) * F[i0, j0] + y * F[i0, j0 + 1]) + x * ((1 - y) * F[i0 + 1, j0] + y * F[i0 + 1, j0 + 1]))


def hidden_amp(flag_fn):
    """rim callback helper: amp per loop vertex = 0 where flag_fn(Pl) says the edge hides under another plate"""
    def f(Pl):
        return np.where(flag_fn(Pl), 0.0, 1.0)
    return f


def smooth_amp(a, it=2):
    a = np.asarray(a, float).copy()
    for _ in range(it):
        a = 0.5 * a + 0.25 * (np.roll(a, 1) + np.roll(a, -1))
    return a


# ======================================================================================================= NECK
def neck_frame(cx):
    """the neck axis (from 15 cm below neck_01 to the head joint) and the neck radius fields (torso-only skin):
    shared by the gorget lames and the helmet's lower rim"""
    if "neck" in cx.shared:
        return cx.shared["neck"]
    n01, hd = cx.head("neck_01"), cx.head("head")
    A0 = np.array([0.0, n01[1] + 0.012, n01[2] - 0.15 * cx.tscale()]); A1 = hd.copy()
    ax = G.nrm(A1 - A0); L = float(np.linalg.norm(A1 - A0))
    front = G.nrm(np.array([0, -1.0, 0]) - np.dot([0, -1.0, 0], ax) * ax); left = np.cross(front, ax) * -1
    if left[0] < 0:
        left = -left
    axis = lambda t: A0 + t * (A1 - A0)
    frame = lambda t: (front, left)
    RFn = G.RadField(cx.torso, axis, frame, 0.0, 1.0, nt=48, nth=72, env=(0, 2, 0.6, 1.5))
    RB = Field2(RFn, G.blur_field(RFn.R, 0.8, 2.0))
    nf = dict(A0=A0, A1=A1, ax=ax, L=L, front=front, left=left, axis=axis, frame=frame, RFn=RFn, RB=RB,
              tpar=lambda p: float(np.dot(np.asarray(p) - A0, ax) / L),
              RBf=lambda T, TH: RB(T, TH) + MAIL_OFF + 0.0045)
    cx.shared["neck"] = nf
    return nf


# ======================================================================================================= HELMET
def column_angles(required, max_step, back_fill=(120, 180)):
    """Sorted azimuths (deg, -180..180) containing `required`, gaps <= max_step, count a multiple of 4."""
    a = sorted(set(round(x, 3) for x in required))
    out = []
    for x, y in zip(a, a[1:] + [a[0] + 360]):
        out.append(x)
        n = int(math.ceil((y - x) / max_step - 1e-6))
        for k in range(1, n):
            out.append(x + (y - x) * k / n)
    out = sorted(((x + 180) % 360) - 180 for x in out)
    while len(out) % 4:
        gaps = [(((out[(i + 1) % len(out)] - out[i]) % 360), i) for i in range(len(out))
                if abs(out[i]) >= back_fill[0]]
        g, i = max(gaps)
        out.append(((out[i] + g / 2 + 180) % 360) - 180)
        out = sorted(out)
    return out


OPEN_TH = 84.0          # skull face opening half-angle (covered by the visor)
V_DIP = 0.010           # the brow edge and the visor top dip this much at the centre (V-shaped eye slot)


class Helm:
    """Slim close helm fitted to the head of `cx`. Skull: s in [0, 1] = rim .. brow (z_eq), s in (1, 2] = dome
    (phi = (s-1) * 90 deg). The visor is a separate plate in front (pt_visor)."""

    def __init__(self, cx):
        self.C = np.array([0.0, 0.5 * (cx.head_front + cx.head_back) - 0.003])
        ez = cx.eye[2]
        # brow line: the skull's brow rim (1.4 cm wide, extends down from here) ends ~1.1 cm above the eye centre and the
        # visor's top rim ~1.1 cm below it: a 2.2 cm eye slot in shadow at eye level (judge M4)
        self.z_eq = ez + 0.021
        self.z_top = cx.head_top + 0.017
        self.Ht = self.z_top - self.z_eq
        self.phimax = 90 * D2R
        front = self.C[1] - cx.head_front; back = cx.head_back - self.C[1]
        self.ax, self.af, self.ab = cx.head_hw + 0.015, front + 0.016, back + 0.012
        self.rx, self.rf, self.rb = self.ax - 0.012, self.af - 0.006, self.ab - 0.020
        # lower rim: 1.2 cm below the chin in front, 1.4 cm above it behind: the helmeted head pitches -10..+12 deg and
        # turns +-45 deg without touching the gorget (armour_upper_qa 'head')
        self.zr_front, self.zr_back = cx.chin_z - 0.012, cx.chin_z + 0.014
        self.scale = (cx.head_top - cx.chin_z) / 0.236          # 1.0 on the male
        # the lower rim must ride OVER the gorget's neck lame with room for the head's own pitch / turn: the lame's
        # surface (neck field + mail + lame) seen from the plan centre C at the rim height, + 2.3 cm
        nf = neck_frame(cx)
        zr = 0.5 * (self.zr_front + self.zr_back)
        tr = nf["tpar"](np.array([0, 0, zr]))
        thn = np.linspace(-np.pi, np.pi, 72, endpoint=False)
        ring = nf["RFn"].point(np.full_like(thn, tr), thn, nf["RBf"](np.full_like(thn, tr), thn) + 0.0025)
        rel = ring[:, :2] - self.C
        hth = np.arctan2(rel[:, 0], -rel[:, 1]); hr = np.linalg.norm(rel, axis=1)
        self.neck_th = np.linspace(-np.pi, np.pi, 37)
        self.neck_need = np.array([hr[angdist(hth, t) < 12 * D2R].max() + 0.017 if (angdist(hth, t) < 12 * D2R).any()
                                   else 0.0 for t in self.neck_th])

    @staticmethod
    def plan(th, ax, af, ab):
        c, s = np.cos(th), np.sin(th)
        a = np.where(c > 0, af, ab)
        return 1.0 / np.sqrt((s / ax) ** 2 + (c / a) ** 2)

    def zrim(self, th):
        return self.zr_back + (self.zr_front - self.zr_back) * (0.5 + 0.5 * np.cos(th))

    def dip(self, th):
        """V-dip of the brow line toward the front centre"""
        return V_DIP * np.clip(1 - np.abs(np.arctan2(np.sin(th), np.cos(th))) / (OPEN_TH * D2R), 0, 1) ** 1.2

    def R_low(self, th, z):
        Req = self.plan(th, self.ax, self.af, self.ab)
        Rr = self.plan(th, self.rx, self.rf, self.rb)
        thw = np.arctan2(np.sin(th), np.cos(th))
        nneed = np.interp(thw, self.neck_th, self.neck_need)
        Rr = np.maximum(Rr, nneed)
        sl = np.clip((z - self.zrim(th)) / (self.z_eq - self.zrim(th)), 0, 1)
        f = 1 - (1 - sl) ** 2.2
        back = G.smoothstep(0.2, -0.75, np.cos(th))           # rear neck guard flares out
        R = Rr + (Req - Rr) * f + (0.003 + 0.010 * back) * np.clip(1 - sl / (0.12 + 0.12 * back), 0, 1) ** 2
        # the neck guard keeps its room over the gorget's neck lame up to 6 cm above the rim (not only at the rim)
        keep = 1 - G.smoothstep(self.zrim(th) + 0.035, self.zrim(th) + 0.075, z)
        return np.maximum(R, nneed * keep + (1 - keep) * 0.0)

    def pt(self, th, s):
        th = np.asarray(th, float); s = np.asarray(s, float)
        th, s = np.broadcast_arrays(th, s)
        Req = self.plan(th, self.ax, self.af, self.ab)
        low = s <= 1.0
        sl = np.clip(s, 0, 1)
        zl = self.zrim(th) + (self.z_eq - self.zrim(th)) * sl
        Rl = self.R_low(th, zl)
        phi = np.clip(s - 1, 0, 1) * self.phimax
        zd = self.z_eq + self.Ht * np.sin(phi)
        Rd = Req * np.cos(phi) ** 0.92                          # slightly fuller crown, not a bowl
        z = np.where(low, zl, zd)
        R = np.where(low, Rl, Rd)
        # V-dip of the brow at the front: the rows around the brow line bend down, fading out over the dome
        wdip = np.where(low, G.smoothstep(0.55, 1.0, s), 1 - G.smoothstep(1.0, 1.35, s))
        z = z - self.dip(th) * wdip
        return np.stack([self.C[0] + R * np.sin(th), self.C[1] - R * np.cos(th), z], -1)

    def theta_open(self, sl):
        return (38.0 + 46.0 * np.clip(sl, 0, 1) ** 0.75) * D2R

    def skull_theta(self, a, s):
        o = self.theta_open(np.clip(s, 0, 1))
        o = np.where(s > 1.0, OPEN_TH * D2R, o)
        A = np.abs(a); sg = np.sign(a)
        f = np.where(A <= OPEN_TH * D2R, A / (OPEN_TH * D2R) * o,
                     o + (A - OPEN_TH * D2R) / (np.pi - OPEN_TH * D2R) * (np.pi - o))
        wgt = 1 - G.smoothstep(1.0, 1.25, s)
        return sg * (A * (1 - wgt) + f * wgt)

    # ---- visor: a proud plate over the face opening: pointed prow (V keel in plan), beak profile, chin
    V_TOP = -0.037                   # visor top border relative to z_eq (its rim reaches ~1.3 cm higher)
    V_OVER = 7.0                     # visor overlaps the skull opening by this many degrees

    V_TOP_SIDE = -0.013              # iteration 2b (user item 23): at the sides the visor's top rises over the brow band
    SLIT_U = (0.50, 0.80)            # the eye slot spans |u| < 0.5 (about +-45 deg), closed from 0.8

    def visor_z(self, u, sv):
        c = np.clip(np.cos(u * np.pi / 2), 0, 1)
        # iteration 2b (user item 23): the lower edge follows the skull's lower rim all round (the old edge rose 3 cm
        # toward the pivots and left the skull's face opening uncovered at the lower rear corners: a hole into the
        # helmet); the top edge rises at the sides so the eye slot is a slot at the front only (it ran round to the
        # pivots and one could look through the helmet from the side)
        th0 = u * (self.theta_open(0.0) + self.V_OVER * D2R)
        zb = np.minimum(self.zr_front - 0.004 - 0.016 * c ** 2 + 0.03 * (1 - c), self.zrim(th0) - 0.004)
        rise = (self.V_TOP_SIDE - self.V_TOP) * G.smoothstep(self.SLIT_U[0], self.SLIT_U[1], np.abs(u))
        zt = self.z_eq + self.V_TOP - V_DIP * np.clip(1 - np.abs(u), 0, 1) ** 1.2 + rise
        return zb + (zt - zb) * sv

    def visor_theta(self, u, z):
        sl = (z - self.zr_front) / (self.z_eq - self.zr_front)
        return u * (self.theta_open(sl) + self.V_OVER * D2R)

    def pt_visor(self, u, sv):
        u = np.asarray(u, float); sv = np.asarray(sv, float)
        u, sv = np.broadcast_arrays(u, sv)
        z = self.visor_z(u, sv)
        th = self.visor_theta(u, z)
        c = np.clip(np.cos(th), 0, 1)
        # the lateral edges and the raised top at the sides hug the skull (plate 2.5 mm + 1.5 mm): no crevice between
        # visor and skull at the pivots or over the brow band
        hug = np.maximum(G.smoothstep(0.78, 1.0, np.abs(u)),
                         G.smoothstep(0.72, 0.98, sv) * G.smoothstep(self.SLIT_U[0] - 0.05, self.SLIT_U[1], np.abs(u)))
        off = 0.0065 - (0.0065 - (LINE_T + 0.0015)) * hug
        R = self.R_low(th, np.minimum(z, self.z_eq)) + off
        kl = np.clip(1 - np.abs(u) / 0.9, 0, 1) ** 1.15                    # V keel: flat cheeks, sharp prow
        prof = 0.40 + 0.60 * np.sin(np.pi * np.clip(sv * 0.95 + 0.02, 0, 1)) ** 0.7
        R = R + 0.032 * self.scale * kl * prof
        R = R + 0.007 * c ** 2 * np.sin(np.pi * np.clip(sv * 0.85 + 0.1, 0, 1)) ** 1.3                 # beak
        R = R + 0.003 * (1 - G.smoothstep(0.0, 0.12, sv)) * c                                         # chin lip
        return np.stack([self.C[0] + R * np.sin(th), self.C[1] - R * np.cos(th), z], -1)

    def normal_of(self, fn, th, s, e=1e-3):
        dth = fn(th + e, s) - fn(th - e, s)
        ds = fn(th, s + e) - fn(th, s - e)
        n = G.nrm(np.cross(dth, ds))
        p = fn(th, s)
        cen = np.array([self.C[0], self.C[1], self.z_eq - 0.03])
        sgn = np.sign(np.sum(n * (p - cen), -1))
        return n * sgn[..., None]

    def normal(self, th, s):
        return self.normal_of(self.pt, th, s)

    def cap_project(self, p):
        x, y = p[0] - self.C[0], -(p[1] - self.C[1])
        th = math.atan2(x, y)
        r = math.hypot(x, y)
        Req = float(self.plan(th, self.ax, self.af, self.ab))
        u = min(r / Req, 0.9999)
        # invert Rd = Req cos(phi)^0.92
        cphi = u ** (1 / 0.92)
        z = self.z_eq + self.Ht * math.sqrt(max(0.0, 1 - cphi * cphi))
        return np.array([p[0], p[1], z])


S_ROWS = [0.0, 0.05, 0.14, 0.28, 0.46, 0.66, 0.84, 1.0]
DOME = [1.1, 1.24, 1.4, 1.58, 1.76]
VENTS_U = [0.30, 0.42, 0.54]         # visor breathing slots each side (fraction of the visor half width)
VENT_HW = 0.014
SV_ROWS = [0.0, 0.08, 0.2, 0.34, 0.5, 0.66, 0.8, 0.91, 1.0]
SV_VENT = (0.34, 0.66)
LINE_T = 0.0025          # plate thickness where an inner (lining) surface is built


def dark_uv(U, V, P_):
    """lining: the dark square (padded arming cap)"""
    u0, v0, u1, v1 = TX.SQUARES["dark_sq"]
    return np.stack([np.full(U.shape, 0.5 * (u0 + u1)) + 0.02 * np.sin(U * 7), np.full(U.shape, 0.5 * (v0 + v1)) + 0.02 * np.sin(V * 5)], -1)


def helmet(cx):
    H = Helm(cx)
    cx.shared["helm"] = H
    mb = G.MB("helmet")
    inside = np.array([H.C[0], H.C[1], H.z_eq - 0.03])
    W_ = {"head": 1.0}
    # ---------------- skull (outer + inward lining)
    cols = column_angles([0, OPEN_TH, -OPEN_TH, 120, -120, 180, 45, -45, 150, -150], 11.5)
    acol = np.array(cols) * D2R
    s_ = np.array(S_ROWS + DOME)
    TH = H.skull_theta(acol[:, None], s_[None, :])
    P = H.pt(TH, np.broadcast_to(s_[None, :], TH.shape))
    ncol = len(acol)
    mask = np.ones((ncol, len(s_) - 1), bool)
    jb = S_ROWS.index(1.0)
    for i in range(ncol):
        mid = cols[i] + (((cols[(i + 1) % ncol] - cols[i]) % 360) / 2)
        mid = ((mid + 180) % 360) - 180
        if abs(mid) < OPEN_TH:
            mask[i, :jb] = False

    def rim_skull(Pl, Nl):
        if Pl[:, 2].mean() > H.z_eq + 0.07:
            return None                                            # top ring: closed by the cap
        # the face-opening sides hide under the visor: flat edge there, gold roll on the brow and the lower rim
        rel = Pl[:, :2] - H.C
        th = np.arctan2(rel[:, 0], -rel[:, 1])
        hidden = (np.abs(th) < (OPEN_TH + 2) * D2R) & (Pl[:, 2] < H.z_eq - 0.02)
        # the brow band is covered by the visor's raised top at the sides (iteration 2b): flat there
        th_cov = (H.SLIT_U[0] + 0.12) * (H.theta_open(1.0) + H.V_OVER * D2R)
        hidden = hidden | ((np.abs(th) > th_cov) & (np.abs(th) < (OPEN_TH + 2) * D2R) & (Pl[:, 2] < H.z_eq + 0.004))
        prof = rim_dark_under(G.rim_gold(w=0.008, bead=0.0032, t=0.002, lip=0.006, band="fil_narrow"), LINE_T)
        return prof, smooth_amp(np.where(hidden, 0.0, 1.0))

    def uv_skull(U, V, P_):
        return TX.steel_uv(U, V)

    # etched band INSIDE the plate surface along the lower rim (iteration 2b: the decal ribbons floated / z-fought)
    P, mask, etch = G.etch_grid(P, mask, ["j0"], 0.018 * H.scale, closed=True)
    jb2 = jb + 1
    brow_w = float(np.median(np.linalg.norm(P[:, jb2 + 1] - P[:, jb2], axis=-1)))
    etch.append(dict(axis=1, rng=(jb2, jb2 + 1), border=jb2, band="etch_band", width=brow_w))
    res = G.shell(mb, P, closed=True, mask=mask, inside=inside, uvfn=uv_skull, rim=rim_skull, w=W_, tag="skull",
                  inner=True, thick=LINE_T, inner_uv=dark_uv, etch=etch)
    top_out = list(res["idx"][:, -1]); top_in = list(res["idx2"][:, -1])
    G.grid_fill(mb, top_out, H.cap_project, lambda p: tuple(TX.steel_uv(np.array(0.2 + p[0]), np.array(0.3 + p[1]))))
    G.grid_fill(mb, top_in, lambda p: p, lambda p: tuple(dark_uv(np.array(p[0]), np.array(p[1]), None)))
    # ---------------- visor (outer + lining), breaths as holes
    req = [0, 1, -1, 0.5, -0.5, 0.8, -0.8, 0.15, -0.15]
    for v in VENTS_U:
        req += [v - VENT_HW, v + VENT_HW, -v - VENT_HW, -v + VENT_HW]
    uu = np.array(sorted(set(np.round(req, 4))))
    fine = []
    for x, y in zip(uu[:-1], uu[1:]):
        n = int(math.ceil((y - x) / 0.12 - 1e-6))
        fine += [x + (y - x) * k / n for k in range(n)]
    uu = np.array(fine + [uu[-1]])
    sv = np.array(SV_ROWS)
    PV = H.pt_visor(uu[:, None], sv[None, :])
    vmask = np.ones((len(uu) - 1, len(sv) - 1), bool)
    j0, j1 = SV_ROWS.index(SV_VENT[0]), SV_ROWS.index(SV_VENT[1])
    for i in range(len(uu) - 1):
        mid = (uu[i] + uu[i + 1]) / 2
        for v in VENTS_U:
            if abs(abs(mid) - v) < VENT_HW * 0.9:
                vmask[i, j0:j1] = False

    def rim_visor(Pl, Nl):
        if np.ptp(Pl[:, 2]) < 0.06 and np.ptp(Pl[:, 0]) < 0.03:
            return G.rim_inner(G.rim_chamfer(t=LINE_T, depth=0.004, c=0.0010), LINE_T)   # breath slot
        return rim_dark_under(G.rim_gold(w=0.007, bead=0.003, t=0.002, lip=0.006, band="fil_narrow"), LINE_T)

    G.shell(mb, PV, closed=False, mask=vmask, inside=inside, rim=rim_visor, w=W_, tag="visor", inner=True,
            thick=LINE_T, inner_uv=dark_uv, uvfn=lambda U, V, P_: TX.steel_uv(U + 1.3, V))
    # (session 3: the felt pads behind the breaths are gone: the dark liner behind the visor closes the breaths from any
    # angle, and the pads floated 1 mm off the lining where the liner showed them: integrity G2)
    vis_bvh = G.mb_bvh(mb, lambda t: t == "visor")
    skull_bvh = G.mb_bvh(mb, lambda t: t == "skull")
    # prow ridge: a narrow gold rib down the visor's keel (closed, its base on the visor)
    svr = np.linspace(0.06, 0.97, 10)
    pr = H.pt_visor(np.zeros_like(svr), svr); nr = H.normal_of(H.pt_visor, np.zeros_like(svr), svr)
    pr = G.seat_points(vis_bvh, pr, nr, lift=0.0)
    G.sweep(mb, pr, nr, RIB6(0.0045, 0.0034), band="gold_plain", w=W_, tag="trim", taper=0.06,
            across=np.tile([1.0, 0, 0], (len(pr), 1)), solid=True, seat=vis_bvh)
    # ---------------- comb: gold crest from the brow's V over the crown to the back
    comb = [(-0.0085, -0.0015), (-0.0070, 0.0040), (-0.0038, 0.0092), (0.0, 0.0108),
            (0.0038, 0.0092), (0.0070, 0.0040), (0.0085, -0.0015)]
    # the comb runs from the brow's V over the crown and stops at the plume socket's rosette; a short tail continues
    # below it (the socket sits ON the comb line, not through it)
    for s0, s1, n0, tp in ((1.02, 2.0, 13, 0.05), (1.62, 1.25, 5, 0.18)):
        if s0 < 1.99 and s0 > s1:
            ss = np.linspace(s0, s1, n0); th_ = np.full_like(ss, math.pi)
        else:
            sf = np.linspace(s0, s1, n0); sb = np.linspace(2.0, 1.93, 3)[1:]
            ss = np.concatenate([sf, sb]); th_ = np.concatenate([np.zeros_like(sf), np.full_like(sb, math.pi)])
        pts = H.pt(th_, ss); nn = H.normal(th_, ss)
        nn[np.abs(ss - 2.0) < 1e-6] = (0, 0, 1)
        across = np.tile([1.0, 0, 0], (len(pts), 1))
        pts = G.seat_points(skull_bvh, pts, nn, lift=0.0)
        G.sweep(mb, pts, nn, [(x * H.scale, y * H.scale) for x, y in COMB6], band="crest", w=W_, tag="crest",
                taper=tp, across=across, solid=True, seat=skull_bvh)
    # ---------------- pivot rosettes
    for sg in (1, -1):
        c = H.pt_visor(sg * 0.92, 0.9); nrm_ = H.normal_of(H.pt_visor, sg * 0.92, 0.9)
        c = G.seat_points(vis_bvh, [c], [nrm_], lift=0.0)[0]
        G.lathe(mb, c, nrm_, [(0.0120, 0.0), (0.0114, 0.0026), (0.0090, 0.0040), (0.0055, 0.0049),
                              (0.0020, 0.0056)], seg=12, band="gold_plain", w=W_, tag="trim", planar="boss", seat=vis_bvh)
    # ---------------- plume socket: short gold tube on a rosette, at the top-back of the crown, leaning back
    sc = 1.78
    c = H.pt(math.pi, sc); nrm_ = G.nrm(np.array([0.0, 0.42, 1.0]))
    c = G.seat_points(skull_bvh, [c], [nrm_], lift=0.0)[0]
    G.lathe(mb, c, nrm_, [(0.0190, 0.0), (0.0182, 0.005), (0.0120, 0.009), (0.0112, 0.027),
                          (0.0128, 0.030), (0.0128, 0.035), (0.0098, 0.037), (0.0060, 0.033)],
            seg=12, band="gold_plain", w=W_, tag="socket", seat=skull_bvh)
    cx.shared["plume_root"] = (c + 0.036 * nrm_, nrm_)
    # ---------------- the dark padded liner behind the slot / breaths (user item 30; replaces the face-shaped headform)
    helm_liner(cx, mb, H)
    mb.meta = dict(helm=H)
    return mb


LINER_GAP = 0.0035       # the liner's outer surface this far under the helmet's inner lining
LINER_T = 0.0012         # liner thickness (a closed felt shell)
LINER_AZ = 94.0          # its half extent in azimuth (deg): 10 deg past the skull's face opening, inside the skull
LINER_E_TOP = 32.0       # its top edge (deg elevation from the head centre): behind the brow band, inside the skull


def helm_liner(cx, mb, H):
    """Iteration 2b (user item 30, integrity G7 / G3 skull-visor; session 3): a dark padded LINER that closes the front
    of the helm from inside. The face-shaped headform (skin + 1.5 mm) left a 2-4 cm cavity behind the visor's keel:
    rays in through the eye slot or a breath crossed it and left through the slot / breaths on the other side or under
    the chin (G7: 560 / 700 px of background from the sides and from above). The liner follows the helmet's own inner
    lining LINER_GAP under it (rays from the head centre, the slot / breath holes filled from the rows above and below),
    from the visor's lower edge (where it closes on the lining, 1.2 mm) to behind the brow band, 10 deg past the face
    opening on each side: every sight line through the slot, the breaths or under the visor's chin ends on dark felt
    a few mm behind the opening, from any angle. Its inner side stays >= 2.5 mm over the mail coif. Fixed grid (same
    topology on every body); rigid to the head."""
    body = cx.body
    headw = body.weight_sum(["head", "jaw", "eye", "brow", "cheek", "nose", "lip", "mouth", "tongue"])
    hs = body.subset(headw > 0.3)
    cen = np.array([H.C[0], H.C[1] + 0.005, cx.eye[2] - 0.015])
    hb = G.mb_bvh(mb, lambda t: t in ("skull", "visor"))
    na, ne = 35, 17
    A_ = np.linspace(-LINER_AZ, LINER_AZ, na) * D2R

    def dirv(a, e):
        return np.array([math.sin(a) * math.cos(e), -math.cos(a) * math.cos(e), math.sin(e)])

    def hit(d):
        l_ = hb.ray_cast(Vector(cen), Vector(d), 0.4)
        return l_[3] if l_[0] is not None else np.nan

    e_top = LINER_E_TOP * D2R
    emin = np.zeros(na)
    for i, a in enumerate(A_):
        es = np.arange(-85.0, 10.0, 0.5) * D2R
        hh = np.array([hit(dirv(a, e)) for e in es])
        ok = np.where(np.isfinite(hh))[0]
        emin[i] = es[ok[0]] if len(ok) else -40 * D2R
    emin = np.maximum(emin, np.convolve(np.pad(emin, 1, mode="edge"), np.ones(3) / 3, mode="valid"))
    # the bottom row stays >= 9 mm above the helmet's own lower edge at that azimuth (the first build hung its edge
    # below the visor's lower rim in a nod: G3 skull-visor looked up into it)
    HV = np.array([p for p, t in zip(mb.V, mb.tag) if t in ("skull", "visor")])
    hrel = HV - cen; haz = np.arctan2(hrel[:, 0], -hrel[:, 1]); hhor = np.linalg.norm(hrel[:, :2], axis=1)
    for i, a in enumerate(A_):
        m_ = np.abs(np.arctan2(np.sin(haz - a), np.cos(haz - a))) < 7 * D2R
        if m_.any():
            k_ = np.argmin(HV[m_, 2])
            e_rim = math.atan2(float(HV[m_, 2][k_]) + 0.009 - cen[2], max(float(hhor[m_][k_]), 0.03))
            emin[i] = max(emin[i] + 2.0 * D2R, e_rim)
    tt = np.linspace(0.0, 1.0, ne)
    L = np.full((na, ne), np.nan); Rs = np.full((na, ne), np.nan); D = np.zeros((na, ne, 3))
    for i, a in enumerate(A_):
        e0 = emin[i] + 1.0 * D2R
        for j, t in enumerate(tt):
            e = e0 + (e_top - e0) * t
            d = dirv(a, e); D[i, j] = d
            L[i, j] = hit(d)
            loc, nor, fi, dist = hs.bvh.ray_cast(Vector(cen + d * 0.3), Vector(-d), 0.3)
            if loc is not None:
                Rs[i, j] = 0.3 - dist
    # slot / breaths / any miss: interpolated along the column from the rows that meet the helmet, then ERODED (3 x 5
    # minimum): across the slot the liner stays at the depth of the brow band (the nearer side), and a ray that slipped
    # through the crevice between the skull's face-opening edge and the visor cannot pull it outward
    L0 = L.copy()
    for i in range(na):
        ok = np.isfinite(L[i])
        if ok.sum() >= 2:
            L[i] = np.interp(tt, tt[ok], L[i][ok])
        elif ok.any():
            L[i] = L[i][ok][0]
    for j in range(ne):
        ok = np.isfinite(L[:, j])
        if not ok.all() and ok.any():
            L[:, j] = np.interp(np.arange(na), np.arange(na)[ok], L[:, j][ok])
    L = np.where(np.isfinite(L), L, 0.11)
    Lp = np.pad(np.where(np.isfinite(L0), L0, L), ((2, 2), (1, 1)), mode="edge")
    Le = np.min(np.stack([Lp[2 + di:2 + di + na, 1 + dj:1 + dj + ne] for di in (-2, -1, 0, 1, 2) for dj in (-1, 0, 1)]), 0)
    L = np.minimum(L, Le)
    # the bottom row closes on the visor's lining; the top edge (behind the brow band) hugs the skull's lining, so the
    # coif (>= 3 mm inside the lining) passes under it
    gap = LINER_GAP + (0.0012 - LINER_GAP) * np.maximum(1 - G.smoothstep(0.0, 0.18, tt), G.smoothstep(0.80, 1.0, tt))[None, :]
    R = L - gap
    R = G.blur_field(R, 0.7, 0.7, wrap_th=False)
    R = np.minimum(R, L - 0.0012)
    rmin = np.where(np.isfinite(Rs), Rs + AV_OFF + 0.0025 + LINER_T, 0.0)
    squeezed = int((R < rmin).sum())
    R = np.where(R < rmin, np.minimum(rmin, L - 0.0012), R)
    P = cen + R[..., None] * D
    log_("helm liner: %d x %d, gap %.1f mm (bottom 1.2), %d points squeezed against the coif, e_min %.0f..%.0f deg" % (
        na, ne, LINER_GAP * 1000, squeezed, emin.min() / D2R, emin.max() / D2R))
    flat = G.prof((0.0, -LINER_T, "dark", 0.0, 1.0))          # session 4: a square edge (no sub-mm rim rows)
    G.shell(mb, P, inside=cen, w={"head": 1.0}, tag="liner", uvfn=dark_uv, inner=True, thick=LINER_T,
            inner_uv=dark_uv, rim=lambda Pl, Nl: flat)


RIB6 = lambda hw, h: [(-hw, 0.00005), (-0.62 * hw, 0.75 * h), (-0.22 * hw, h), (0.22 * hw, h), (0.62 * hw, 0.75 * h),
                      (hw, 0.00005)]                 # closed rib section (base on the plate), iteration 2b
COMB6 = [(-0.0085, 0.00005), (-0.0070, 0.0052), (-0.0030, 0.0104), (0.0030, 0.0104), (0.0070, 0.0052), (0.0085, 0.00005)]


def rim_dark_under(profile, thick):
    """closed rim (G.rim_closed) whose turned-under part is DARK: the slot / breath borders show shadow, not lit steel"""
    p = G.rim_closed(profile, thick)
    return p[:-2] + [(a, b, "dark", c, d) for (a, b, _, c, d) in p[-2:]]


def head_form(cx, mb, H):
    """User item 30: the head inside the closed helm. A dark 2 mm shell shaped like this body's face (the padded coif and
    the face in shadow): rays from the head centre to the head skin (+3 mm), blurred to a soft form, kept >= 2 mm inside
    the helmet's lining, spanning ear to ear across the front and from under the chin to the forehead. Every sight
    line in through the eye slot and the breaths ends on it (never the background or the lit far wall); from below
    one sees its dark underside. Rigid to the head like the helmet (the helm look has no face animation)."""
    body = cx.body
    headw = body.weight_sum(["head", "jaw", "eye", "brow", "cheek", "nose", "lip", "mouth", "tongue"])
    hs = body.subset(headw > 0.3)
    cen = np.array([H.C[0], H.C[1] + 0.005, cx.eye[2] - 0.015])        # = the coif's face-opening centre
    hb = G.mb_bvh(mb, lambda t: t in ("skull", "visor", "trim", "crest"))
    # iteration 2b: the mail coif (piece 'aventail') covers the rest of the head; the face shows in its opening only
    A_ = np.linspace(-(FACE_OPEN[0] + 12), FACE_OPEN[0] + 12, 11) * D2R          # azimuth, 0 = front, + = his left
    E_ = np.linspace(FACE_OPEN[1] - FACE_OPEN[2] - 9, FACE_OPEN[1] + FACE_OPEN[2] + 9, 8) * D2R   # elevation
    R = np.full((len(A_), len(E_)), np.nan); Lim = np.full(R.shape, 1.0)
    for i, a in enumerate(A_):
        for j, e in enumerate(E_):
            d = np.array([math.sin(a) * math.cos(e), -math.cos(a) * math.cos(e), math.sin(e)])
            loc, nor, fi, dist = hs.bvh.ray_cast(Vector(cen + d * 0.3), Vector(-d), 0.3)
            if loc is not None:
                R[i, j] = 0.3 - dist + 0.0015
            l2 = hb.ray_cast(Vector(cen), Vector(d), 0.4)
            if l2[0] is not None:
                Lim[i, j] = l2[3] - 0.0025
    for _ in range(40):
        bad = np.isnan(R)
        if not bad.any():
            break
        Rp = np.pad(R, 1, mode="edge")
        nbv = np.stack([Rp[:-2, 1:-1], Rp[2:, 1:-1], Rp[1:-1, :-2], Rp[1:-1, 2:]])
        R[bad] = np.nanmean(nbv, axis=0)[bad]
    R = np.where(np.isnan(R), 0.08, R)
    R0 = R.copy()
    R = G.blur_field(R, 0.8, 0.8, wrap_th=False)
    R = np.minimum(np.minimum(R, R0 + 0.0006), Lim)     # never outward of skin + 2 mm (the coif lies at + 3.5 mm)
    AA, EE = np.meshgrid(A_, E_, indexing="ij")
    D = np.stack([np.sin(AA) * np.cos(EE), -np.cos(AA) * np.cos(EE), np.sin(EE)], -1)
    P = cen + R[..., None] * D

    def uv_d(U, V_, P_):
        return np.stack([np.full(U.shape, 0.875) + 0.02 * np.sin(U * 9), np.full(U.shape, 0.5) + 0.02 * np.sin(V_ * 7)], -1)
    flat = G.prof((0.0015, 0.0, "dark", 0.0, 0.5))
    G.shell(mb, P, inside=cen, w={"head": 1.0}, tag="headform", uvfn=uv_d, inner=True, thick=0.002, inner_uv=uv_d,
            rim=lambda Pl, Nl: flat)


# ======================================================================================================= PLUME
PLUME_SEG = 7
PLUME_BONES = ("plume_01", "plume_02", "plume_03")


def plume_spine(cx):
    """(M, 3) arched crest spine from the socket mouth: rises, arches back over the crown and falls behind the helm
    to the shoulder blades (~0.46 m); scaled by the head size, the SAME shape on every body (canonical plume frames)."""
    root, ax = cx.shared["plume_root"]
    H = cx.shared["helm"]
    k = H.scale
    ctrl = np.array([(0, 0, 0), (0, 0.010, 0.050), (0, 0.055, 0.092), (0, 0.130, 0.095), (0, 0.195, 0.045),
                     (0, 0.235, -0.050), (0, 0.255, -0.170), (0, 0.262, -0.290)]) * k
    # smooth Catmull-Rom through the control points, resampled by arc length
    P = []
    C_ = np.concatenate([ctrl[:1], ctrl, ctrl[-1:]])
    for i in range(1, len(C_) - 2):
        p0, p1, p2, p3 = C_[i - 1], C_[i], C_[i + 1], C_[i + 2]
        for t in np.linspace(0, 1, 12, endpoint=False):
            t2, t3 = t * t, t * t * t
            P.append(0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2 + (-p0 + 3 * p1 - 3 * p2 + p3) * t3))
    P.append(ctrl[-1])
    P = np.array(P)
    seg = np.linalg.norm(np.diff(P, axis=0), axis=1); s = np.concatenate([[0], np.cumsum(seg)])
    t = np.linspace(0, s[-1], 60)
    Q = np.stack([np.interp(t, s, P[:, j]) for j in range(3)], 1)
    return root + Q, s[-1]


def plume_weights(t):
    """plume_01..03 weights along the spine parameter t (0 socket .. 1 end, > 1 beyond the end): thirds, blended"""
    x = float(np.clip(t * 3.0, 0, 2.9999))
    k = int(x); f = x - k
    d = {PLUME_BONES[k]: 1.0}
    if k < 2 and f > 0.75:
        b = (f - 0.75) / 0.5; d = {PLUME_BONES[k]: 1 - b, PLUME_BONES[k + 1]: b}
    elif k > 0 and f < 0.25:
        b = (0.25 - f) / 0.5; d = {PLUME_BONES[k - 1]: b, PLUME_BONES[k]: 1 - b}
    return d


def plume(cx):
    """Horsehair crest (judge M5): a solid core volume along the arched spine (an elliptical tube in the dense-hair
    atlas column: never see-through, never edge-on) + 3 layers of V-section tapered locks rooted in the socket that
    wrap the core and run past its end (tip fray), side locks drooping; per-vertex shade darkens roots, underside and
    tips. Weighted along the spine to plume_01..03 (spring chain)."""
    mb = G.MB("plume")
    H = cx.shared["helm"]
    k = H.scale
    spine, L = plume_spine(cx)
    n_ = len(spine)
    sp_s = np.linspace(0, 1, n_)
    Tg = G.nrm(np.gradient(spine, axis=0))
    lat = np.array([1.0, 0, 0])
    Nup = G.nrm(np.cross(lat, Tg))
    Nup = np.where((np.sum(Nup * np.array([0, 0.3, 1.0]), -1) < 0)[:, None], -Nup, Nup)

    def S(t):
        t = np.atleast_1d(np.asarray(t, float))
        tc = np.clip(t, 0, 1)
        P = np.array([np.interp(tc, sp_s, spine[:, j]) for j in range(3)]).T
        T_ = G.nrm(np.array([np.interp(tc, sp_s, Tg[:, j]) for j in range(3)]).T)
        N_ = G.nrm(np.array([np.interp(tc, sp_s, Nup[:, j]) for j in range(3)]).T)
        over = np.clip(t - 1, 0, None)[:, None] * L                # beyond the end: continue down the tangent
        P = P + over * G.nrm(T_[-1:] + np.array([0, 0.25, -0.6])) if over.any() else P
        return P, T_, N_

    # core half extents: lateral a(t), along the arch normal b(t)
    a_fn = lambda t: k * np.interp(t, [0, 0.08, 0.3, 0.6, 0.85, 1.0], [0.008, 0.032, 0.058, 0.070, 0.064, 0.034])
    b_fn = lambda t: k * np.interp(t, [0, 0.08, 0.3, 0.6, 0.85, 1.0], [0.008, 0.026, 0.040, 0.032, 0.020, 0.008])

    def off_helm(pts, clear=0.012):
        """push points out of the skull (radially in plan, around the helm's axis)"""
        out = pts.copy()
        for kk in range(len(out)):
            q = out[kk]
            if q[2] < H.z_eq - 0.14:
                continue
            x, y = q[0] - H.C[0], -(q[1] - H.C[1])
            th = math.atan2(x, y)
            s_eq = 1.0 + math.asin(float(np.clip((q[2] - H.z_eq) / H.Ht, -1, 1))) / H.phimax if q[2] > H.z_eq else 0.8
            spt = H.pt(th, max(s_eq, 0.2))
            rs = math.hypot(spt[0] - H.C[0], spt[1] - H.C[1]) + clear
            r = math.hypot(x, y)
            if r < rs and q[2] < spt[2] + clear:
                f = rs / max(r, 1e-4)
                out[kk, 0] = H.C[0] + x * f; out[kk, 1] = H.C[1] - y * f
        return out

    ncol = 8
    colu = lambda c: (c / ncol + 0.004, (c + 1) / ncol - 0.004)
    shade = []
    # ---------------- core tube (dense column), 10 around x 18 along, closed at the root inside the socket
    nt, na = 18, 10
    tt = np.linspace(0.0, 0.97, nt) ** 1.1
    ph = np.linspace(0, 2 * np.pi, na, endpoint=False)
    P0, T0, N0 = S(tt)
    B0 = G.nrm(np.cross(T0, N0))
    ring = (a_fn(tt)[:, None, None] * np.cos(ph)[None, :, None] * B0[:, None, :] +
            b_fn(tt)[:, None, None] * np.sin(ph)[None, :, None] * N0[:, None, :])
    PC = P0[:, None, :] + ring
    PC = off_helm(PC.reshape(-1, 3), 0.016).reshape(PC.shape)
    wts = [plume_weights(t) for t in tt for _ in ph]
    ids = mb.add(PC, w=wts, tag="core")
    u0, u1 = colu(1)
    for i in range(nt - 1):
        v0, v1 = 1 - tt[i] * 0.98, 1 - tt[i + 1] * 0.98
        for j in range(na):
            j2 = (j + 1) % na
            ua = u0 + (u1 - u0) * abs(math.sin(ph[j] / 2)); ub = u0 + (u1 - u0) * abs(math.sin((ph[j] + 2 * np.pi / na) / 2))
            mb.quad(ids[i, j], ids[i + 1, j], ids[i + 1, j2], ids[i, j2], [(ua, v0), (ua, v1), (ub, v1), (ub, v0)])
    for t in tt:
        for p_ in ph:
            shade.append((0.62 + 0.26 * G.smoothstep(0.0, 0.3, t)) * (0.80 + 0.20 * max(0.0, math.sin(p_))))
    # ---------------- locks: (count, psi range deg around the arch normal, length range, width, lift, droop, columns)
    rng = np.random.default_rng(11)
    layers = [(30, (-70, 70), (0.80, 1.16), 0.032, 1.05, 0.15, (3, 4, 5)),        # top: the long flowing locks
              (28, (60, 125), (0.60, 1.08), 0.030, 1.02, 0.55, (3, 4, 5, 6)),      # sides (both), drooping
              (14, (125, 180), (0.45, 0.85), 0.030, 0.95, 0.35, (1, 2, 5))]        # underneath: short, dense
    nseg = 7
    for li, (cnt, (p0, p1), (l0, l1), wid, lift, droop, cols) in enumerate(layers):
        for c in range(cnt):
            if li == 0:
                psi = p0 + (p1 - p0) * (c + rng.uniform(0.15, 0.85)) / cnt
            else:
                sgn = 1 if c % 2 == 0 else -1
                psi = sgn * (p0 + (p1 - p0) * ((c // 2) + rng.uniform(0.1, 0.9)) / max(1, (cnt + 1) // 2))
            psi = psi * D2R
            t0 = rng.uniform(0.0, 0.05)
            t1 = rng.uniform(l0, l1)
            t = t0 + (t1 - t0) * np.linspace(0, 1, nseg + 1) ** 0.9
            base, T_, N_ = S(t)
            Bn = G.nrm(np.cross(T_, N_))
            tc = np.clip(t, 0, 1)
            rad = (np.cos(psi) * N_ * b_fn(tc)[:, None] + np.sin(psi) * Bn * a_fn(tc)[:, None]) * lift
            dirn = G.nrm(np.cos(psi) * N_ + np.sin(psi) * Bn)
            fall = droop * 0.06 * k * np.clip(t, 0, None) ** 2 * abs(math.sin(psi))
            jit = rng.normal(0, 0.005 * k, 3) * np.clip(t, 0, 1)[:, None]
            pts = base + rad + dirn * 0.003 - fall[:, None] * np.array([0, 0, 1.0]) + jit
            pts = off_helm(pts, 0.022)
            T2 = G.nrm(np.gradient(pts, axis=0))
            f = (t - t0) / max(t1 - t0, 1e-3)
            w = wid * k * np.interp(f, [0, 0.1, 0.5, 0.85, 1.0], [0.35, 0.85, 1.0, 0.75, 0.25]) * rng.uniform(0.8, 1.2)
            across = G.nrm(np.cross(T2, dirn))
            upv = G.nrm(np.cross(across, T2))
            upv = np.where((np.sum(upv * dirn, -1) < 0)[:, None], -upv, upv)
            ridge = 0.28 * w
            left = pts - across * (w / 2)[:, None]; mid = pts + upv * ridge[:, None]; right = pts + across * (w / 2)[:, None]
            cu0, cu1 = colu(int(rng.choice(cols)))
            if cols[0] in (3, 4) and rng.uniform() < 0.5:           # lock columns hold two locks: use one half
                h_ = rng.integers(0, 2); cu0, cu1 = cu0 + (cu1 - cu0) * 0.5 * h_, cu0 + (cu1 - cu0) * 0.5 * (h_ + 1)
            um = 0.5 * (cu0 + cu1)
            wts = [plume_weights(x) for x in t]
            li_ = mb.add(left, w=wts, tag="card"); mi_ = mb.add(mid, w=wts, tag="card"); ri_ = mb.add(right, w=wts, tag="card")
            for kk in range(nseg):
                v0, v1 = 1 - kk / nseg * 0.98, 1 - (kk + 1) / nseg * 0.98
                mb.quad(li_[kk], mi_[kk], mi_[kk + 1], li_[kk + 1], [(cu0, v0), (um, v0), (um, v1), (cu0, v1)])
                mb.quad(mi_[kk], ri_[kk], ri_[kk + 1], mi_[kk + 1], [(um, v0), (cu1, v0), (cu1, v1), (um, v1)])
            under = 0.5 + 0.5 * math.cos(psi)                     # 1 on top, 0 underneath
            a_ = (0.64 + 0.36 * G.smoothstep(0.0, 0.3, f)) * (1 - 0.20 * G.smoothstep(0.8, 1.0, f)) * (0.78 + 0.22 * under)
            shade += list(a_) * 3
    mb.meta = dict(shade=np.array(shade))
    cx.shared["plume_spine"] = spine
    return mb


def plume_bone_specs(spine):
    """plume_01..03 rest frames along the spine thirds (direction = the chord, canonical by construction: the spine
    shape is the same on every body up to scale)."""
    n = len(spine)
    idx = [0, n // 3, 2 * n // 3, n - 1]
    out = {}
    for k, b in enumerate(PLUME_BONES):
        h, t = spine[idx[k]], spine[idx[k + 1]]
        out[b] = dict(parent="head" if k == 0 else PLUME_BONES[k - 1], head=h, tail=t)
    return out


def ensure_plume_bones(rig, specs=None):
    if specs is None:
        h = np.array(rig.data.bones["head"].tail_local[:])
        specs = {b: dict(parent="head" if k == 0 else PLUME_BONES[k - 1], head=h + np.array([0, 0.02 + 0.1 * k, 0.02]),
                         tail=h + np.array([0, 0.12 + 0.1 * k, 0.02])) for k, b in enumerate(PLUME_BONES)}

    def f(eb):
        for n, sp in specs.items():
            b = eb.get(n) or eb.new(n)
            b.head = Vector(sp["head"]); b.tail = Vector(sp["tail"])
            b.align_roll(Vector((1.0, 0, 0)))
        for n, sp in specs.items():
            eb[n].parent = eb[sp["parent"]]
            eb[n].use_connect = False
            eb[n].use_deform = True
    RG._edit(rig, f)
    for n in specs:
        rig.pose.bones[n].rotation_mode = 'QUATERNION'
    RG.add_springs(rig, PLUME_BONES)


# ======================================================================================================= CUIRASS
class Field2:
    """A RadField with its radius table replaced (e.g. blurred harder): same axis / frame / lookup."""

    def __init__(self, rf, R):
        self.__dict__.update(rf.__dict__)
        self.R = R

    __call__ = G.RadField.__call__
    point = G.RadField.point


def torso_field(cx):
    co = cx.torso.co
    z0, z1 = float(cx.tz(0.98)), float(cx.tz(1.64))
    zs = np.linspace(z0, z1, 32)
    yc = []
    for z in zs:
        m = (np.abs(co[:, 2] - z) < 0.01) & (np.abs(co[:, 0]) < 0.03)
        yc.append((co[m, 1].min() + co[m, 1].max()) / 2 if m.any() else -0.02)
    yc = G.blur_field(np.array(yc)[:, None], 2.0, 0)[:, 0]
    axis = lambda z: np.array([0.0, float(np.interp(z, zs, yc)), z])
    frame = lambda z: (np.array([0, -1.0, 0]), np.array([1.0, 0, 0]))
    rf = G.RadField(cx.torso, axis, frame, z0, z1, nt=66, nth=90, env=(3, 3, 2.5, 2.0))
    # iteration 2b (G4 cuirass|mail at the arm-pits): the trunk-only skin (arm weight < 0.35) has a HOLE at the arm-pit
    # fold, so its rays found nothing there and the filled-in radius dipped under the skin: the plates' sides cut the
    # mail and the flank. The fold (arm weight < 0.65) is added, and the globose blur may not sink below it.
    armw = cx.body.weight_sum(["upperarm", "lowerarm", "hand", "thumb", "index", "middle", "ring", "pinky"])
    rf2 = G.RadField(cx.body.subset(armw < 0.65), axis, frame, z0, z1, nt=66, nth=90, env=(3, 3, 2.5, 2.0))
    Rm = np.maximum(rf.R, rf2.R)
    Rb = G.blur_field(Rm, 3.5, 4.0)                                  # globose: anatomy blurred away
    lift = G.blur_field(np.maximum(0.0, Rm - Rb), 1.5, 2.0) * 1.3
    return rf, Field2(rf, np.maximum(Rb + lift, Rm - 0.0015))


def cuirass(cx):
    mb = G.MB("cuirass")
    RF, RS = torso_field(cx)
    tz, k = cx.tz, cx.tscale()
    inside = np.array([0, -0.02, float(tz(1.3))])
    W5, W3 = {"spine_05": 1.0}, {"spine_03": 1.0}

    def neck_z(th):
        a = np.abs(np.arctan2(np.sin(th), np.cos(th))) / D2R
        neck = tz(1.486) + 0.040 * k * G.smoothstep(0, 44, a) ** 1.4          # gently curved neck opening
        arm = np.interp(a, [0, 42, 52, 64, 78, 90, 100], [0, 0, -0.018, -0.066, -0.108, -0.126, -0.132]) * k
        return neck + arm

    def back_z(th):
        d = 180 - np.abs(np.arctan2(np.sin(th), np.cos(th))) / D2R
        neck = tz(1.522) + 0.022 * k * G.smoothstep(0, 35, d)
        arm = np.interp(d, [0, 45, 60, 75, 90, 100], [0, 0, -0.03, -0.085, -0.125, -0.135]) * k
        return neck + arm

    def R_breast(Z, TH):
        a = np.arctan2(np.sin(TH), np.cos(TH))
        c = 0.020 + 0.006 * (1 - G.smoothstep(tz(1.12), tz(1.30), Z))
        bulge = 0.010 * np.exp(-((Z - tz(1.34)) / (0.10 * k)) ** 2) * np.clip(np.cos(a), 0, 1) ** 2
        ridge = 0.004 * np.clip(1 - np.abs(a) / (26 * D2R), 0, 1) * G.smoothstep(tz(1.10), tz(1.25), Z) * (1 - G.smoothstep(tz(1.40), tz(1.46), Z))
        return RS(Z, TH) + c + bulge + ridge

    SG = CUIR_T + 0.0045                                      # stacked-plate step: plate + clearance (closed shells)

    def R_back(Z, TH):
        a = np.abs(np.arctan2(np.sin(TH), np.cos(TH)))
        r = RS(Z, TH) + 0.018
        side = a < 112 * D2R                                  # under the breastplate (and its side roll) at the sides
        return np.where(side, np.minimum(r, R_breast(Z, TH) - SG), r)

    def R_plack(Z, TH):
        return np.maximum(R_breast(Z, TH) + SG, RS(Z, TH) + 0.022 + SG)

    def R_culet(Z, TH):
        a = np.abs(np.arctan2(np.sin(TH), np.cos(TH)))
        r = np.maximum(R_back(Z, TH) + SG, RS(Z, TH) + 0.020 + SG)
        return np.where(a < 100 * D2R, np.minimum(r, R_plack(Z, TH) - SG), r)

    Pb = lambda Z, TH: RS.point(Z, TH, R_breast(Z, TH))
    cx.shared["cuirass_fields"] = dict(RF=RF, RS=RS, R_breast=R_breast, R_back=R_back, neck_z=neck_z, back_z=back_z)

    arms = [arm_body(cx, "l"), arm_body(cx, "r")]

    def arm_clear(P):
        """how far each point is from clearing the arm skin + mail + plate (> 0 = too close / inside the arm); the arm-pit
        fold counts too (iteration 2b: the arm-hole's low point sat in the fold, above the mail's arm-pit bridge)"""
        out = np.zeros(len(P))
        for k, p in enumerate(P):
            for ab in arms + [cx.body]:
                loc, nor, fi, dist = ab.bvh.find_nearest(Vector(p))
                if loc is None:
                    continue
                sd = float(np.dot(p - np.array(loc), np.array(nor)))
                out[k] = max(out[k], (ARMHOLE_CLEAR if ab is not cx.body else ARMHOLE_CLEAR - 0.004) - sd)
        return out

    def armhole_top(th, ZB, ZT, rfn):
        """iteration 2b: lower the arm-hole edge (per column) until the plate's top rows clear the arm skin by the mail
        + clearance + the plate (the arm-hole cut into the arm: the mail could not pass under it)"""
        ZT = ZT.copy()
        for i in range(len(th)):
            for _ in range(18):
                # the rolled rim reaches ~2 cm past the grid's top edge: test there too (iteration 2b)
                zs_ = ZT[i] + np.array([0.019, 0.010, 0.0, -0.012, -0.024])
                pts = RS.point(zs_, np.full(5, th[i]), rfn(zs_, np.full(5, th[i])))
                if arm_clear(pts).max() <= 0 or ZT[i] - 0.02 < ZB[i]:
                    break
                ZT[i] -= 0.004
        ZT = np.minimum(ZT, G.blur_field(ZT[None, :], 0, 0.8, wrap_th=False)[0])
        return ZT

    def grid(th_deg, zb_fn, zt_fn, rfn, rows, pw=1.0, armhole=False):
        th = np.array(th_deg) * D2R
        v = np.linspace(0, 1, rows)
        ZB, ZT = zb_fn(th), zt_fn(th)
        if armhole:
            ZT = armhole_top(th, ZB, ZT, rfn)
        Z = ZB[:, None] + (ZT - ZB)[:, None] * (v[None, :] ** pw)
        TH = np.broadcast_to(th[:, None], Z.shape)
        return RS.point(Z, TH, rfn(Z, TH)), Z, TH

    def rim_by(hide_fn, prof):
        def f(Pl, Nl):
            return prof, smooth_amp(np.where(hide_fn(Pl), 0.0, 1.0))
        return f

    roll = G.rim_gold(w=0.010, bead=0.0055, t=0.0025, lip=0.008, band="fil_narrow")     # turned roll
    TK = CUIR_T
    ew = 0.024 * k                                                    # etched band width (in the plate surface)
    # ---- breastplate (spine_05): -100..100 deg, bottom hidden under the plackart; a closed thick plate (item 28)
    thb = sorted(set([0, 6, -6] + list(np.arange(-100, 101, 10.0))))
    zb_b = lambda th: tz(1.200) + 0.010 * G.smoothstep(0, 60, np.abs(th) / D2R)
    P, Z, TH = grid(thb, zb_b, neck_z, R_breast, 11, 0.95, armhole=True)
    zlow = float(tz(1.215))
    P, _, eb = G.etch_grid(P, None, ["j1"], ew)
    def rim_breast(Pl, Nl):
        a = np.abs(np.arctan2(Pl[:, 0], -(Pl[:, 1] - float(RS.axis_fn(float(tz(1.3)))[1])))) / D2R
        side = (a > 93) & (Pl[:, 2] < neck_z(np.array(100 * D2R)) - 0.01)
        amp = smooth_amp(np.where(Pl[:, 2] < zlow, 0.0, 1.0))
        return roll, amp, smooth_amp(np.where(side, 0.45, 0.35 + 0.65 * amp))
    G.shell(mb, P, inside=inside, w=W5, tag="breast", uvfn=lambda U, V, P_: TX.steel_uv(U, V),
            rim=rim_breast, inner=True, thick=TK, etch=eb)
    brb = G.mb_bvh(mb, lambda t: t == "breast")
    # stop-rib: a roped rib following the neck opening 4 cm below it, dipping to a gentle V at the centre (closed,
    # seated on the breastplate)
    ths = np.linspace(-36, 36, 29) * D2R
    zr = neck_z(ths) - 0.040 * k - 0.010 * k * np.clip(1 - np.abs(ths) / (36 * D2R), 0, 1)
    pr = Pb(zr, ths)
    e = 1e-3
    nr = G.nrm(np.cross(Pb(zr, ths + e) - Pb(zr, ths - e), Pb(zr + e, ths) - Pb(zr - e, ths)))
    nr = np.where((np.sum(nr * (pr - inside), -1) < 0)[:, None], -nr, nr)
    pr = G.seat_points(brb, pr, nr, lift=0.0)
    G.sweep(mb, pr, nr, RIB6(0.0042, 0.0044), band="gold_bead", w=W5, tag="stoprib", taper=0.12, solid=True, seat=brb)
    # (iteration 2b: the separate arm gussets are gone: they sat on the breastplate's raised arm-hole roll and cut it)
    # ---- plackart (spine_03): over the breastplate's lower part, ogival top
    thp = sorted(set([0, 6, -6] + list(np.arange(-96, 97, 12.0))))
    zt_p = lambda th: tz(1.222) + 0.030 * k * (1 - G.smoothstep(0, 38, np.abs(th) / D2R)) ** 1.6
    zb_p = lambda th: tz(1.045) + 0.012 * G.smoothstep(0, 45, np.abs(th) / D2R)
    P, Z, TH = grid(thp, zb_p, zt_p, R_plack, 7)
    P, _, ep = G.etch_grid(P, None, ["j1"], ew)
    G.shell(mb, P, inside=inside, w=W3, tag="plackart", uvfn=lambda U, V, P_: TX.steel_uv(U + 0.7, V + 0.1),
            rim=lambda Pl, Nl: roll, inner=True, thick=TK, etch=ep)
    # ---- backplate (spine_05): 80..280 deg, under the breastplate at the sides, bottom under the culet
    thk = sorted(set([96.0] + list(np.arange(100, 261, 10.0)) + [180, 264.0]))   # clear of the arm-pit (2b)
    zb_k = lambda th: tz(1.215) + 0.0 * th
    P, Z, TH = grid(thk, zb_k, back_z, R_back, 10, armhole=True)
    P, _, ek = G.etch_grid(P, None, ["j1"], ew)

    def hide_back(Pl):
        # bottom under the culet; the side edges under the breastplate
        a = np.abs(np.arctan2(Pl[:, 0], -(Pl[:, 1] + 0.02))) / D2R
        return (Pl[:, 2] < zlow + 0.004) | (a < 101)
    G.shell(mb, P, inside=inside, w=W5, tag="back", uvfn=lambda U, V, P_: TX.steel_uv(U + 1.0, V),
            rim=rim_by(hide_back, roll), inner=True, thick=TK, etch=ek)
    # ---- lower back plate / culet (spine_03)
    thc = sorted(set(list(np.arange(120, 241, 12.0)) + [180]))
    P, Z, TH = grid(thc, lambda th: tz(1.045) + 0.0 * th, lambda th: tz(1.238) + 0.0 * th, R_culet, 6)
    G.shell(mb, P, inside=inside, w=W3, tag="culet", uvfn=lambda U, V, P_: TX.steel_uv(U + 1.6, V + 0.2),
            rim=lambda Pl, Nl: G.rim_gold(w=0.008, bead=0.0040, t=0.0025, lip=0.007, band="fil_narrow"),
            inner=True, thick=TK)
    return mb


CUIR_T = 0.0025          # cuirass plate thickness (closed shells, iteration 2b)
ARMHOLE_CLEAR = 0.0170   # an arm-hole edge's distance from the arm skin: mail 8.5 + 3 mm + plate 2.5 + rim 3


# ======================================================================================================= GORGET
HEAD_POSES = [(p, t, j) for p in (-20, -10, 0, 12, 25) for t in (-45, 0, 45) for j in (0.0, 1.0)]
NECK_SPLIT = (0.3, 0.3, 0.4)          # pitch / turn shared by neck_01, neck_02, head (engine-style neck rig)
HELM_PITCH_MIN = -10.0                # a closed helm rests on the gorget: helmeted look-up limit (the bare head: -20)
HELM_PITCH_MAX = 12.0                 # helmeted chin-down limit (the bare head: +25)


def pose_head(rig, pitch, turn):
    """head pitch (+ = chin down) and turn (+ = to the character's left) spread over the neck chain"""
    pose_reset(rig)
    for b, f in zip(("neck_01", "neck_02", "head"), NECK_SPLIT):
        if turn:
            rot(rig, b, (0, 0, 1), turn * f)
        if pitch:
            rot(rig, b, (1, 0, 0), pitch * f)


def head_pose_points(cx, frame_bone, extra=None, poses=HEAD_POSES):
    """For each head pose: the lower-face skin (and `extra` rigid-to-head points, e.g. the helmet) expressed in the
    REST frame of `frame_bone` (the piece's bone): list of (n, 3) arrays."""
    rig, bm = cx.rig, cx.bm
    masks = [m for m in bm.modifiers if m.type == 'MASK']
    vis = [m.show_viewport for m in masks]
    for m in masks:
        m.show_viewport = False
    kb = bm.data.shape_keys.key_blocks
    jaw = kb.get("jawOpen")
    co = cx.body.co
    sel = np.nonzero(cx.headmask & (co[:, 2] < cx.eye[2] - 0.01))[0]
    out = []
    Rf = Matrix(rig.data.bones[frame_bone].matrix_local)
    Rh = Matrix(rig.data.bones["head"].matrix_local)
    for pitch, turn, jv in poses:
        pose_head(rig, pitch, turn)
        if jaw is not None:
            jaw.value = jv
        dg = bpy.context.evaluated_depsgraph_get()
        ev = bm.evaluated_get(dg); me = ev.to_mesh()
        P = np.empty(len(me.vertices) * 3); me.vertices.foreach_get("co", P); P = P.reshape(-1, 3)
        ev.to_mesh_clear()
        pts = P[sel]
        M = np.array(Rf @ rig.pose.bones[frame_bone].matrix.inverted())
        q = pts @ M[:3, :3].T + M[:3, 3]
        if extra is not None and len(extra) and HELM_PITCH_MIN <= pitch <= HELM_PITCH_MAX:
            Mh = np.array(Rf @ rig.pose.bones[frame_bone].matrix.inverted() @ rig.pose.bones["head"].matrix @ Rh.inverted())
            q = np.concatenate([q, np.asarray(extra) @ Mh[:3, :3].T + Mh[:3, 3]])
        out.append(q)
    if jaw is not None:
        jaw.value = 0.0
    pose_reset(rig)
    for m, v in zip(masks, vis):
        m.show_viewport = v
    return out


def allowed_top(pose_pts, axis0, axis1, Rfn, thetas, reach=0.016, margin=0.014, t_lo=-1.0):
    """Highest allowed axial parameter t (0..1 along axis0->axis1) of a ring's top edge per azimuth so no head / helmet
    point that comes within `reach` of the ring surface (radially) is closer than `margin` above it, over all poses."""
    ax = axis1 - axis0; L = np.linalg.norm(ax); ax = ax / L
    front = G.nrm(np.array([0, -1.0, 0]) - np.dot([0, -1.0, 0], ax) * ax); left = np.cross(front, ax) * -1
    if left[0] < 0:
        left = -left
    tmax = np.full(len(thetas), 2.0)
    dth = thetas[1] - thetas[0]
    for q in pose_pts:
        v = q - axis0
        t = v @ ax / L
        pr = v - np.outer(v @ ax, ax)
        r = np.linalg.norm(pr, axis=1)
        th = np.arctan2(pr @ left, pr @ front)
        for j, T in enumerate(thetas):
            dd = np.abs(np.arctan2(np.sin(th - T), np.cos(th - T)))
            m = dd < 1.5 * dth
            if not m.any():
                continue
            Rq = Rfn(np.clip(t, 0, 1), np.full_like(t, T))
            ok = m & (r > Rq - 0.008) & (r < Rq + reach) & (t > t_lo)   # near the ring surface, above its bottom
            if ok.any():
                tmax[j] = min(tmax[j], float(t[ok].min()) - margin / L)
    return tmax


def radial_push(RF, T, TH, R, targets, iters=6, smooth=True, look_lo=0.0, look_hi=0.0, max_step=0.004):
    """radius table R of a ring field (points RF.point(T, TH, R)) pushed OUT radially until every point is at least
    `need` (signed, along the surface normal) above each target BVH [(bvh, need)]: a normal clearance (a radial offset
    on a sloped surface, e.g. the trapezius under the gorget, is not one). Smoothed, never below the requirement."""
    R = np.array(R, float)
    nr = R.shape[-1]
    for it in range(iters):
        P = RF.point(T, TH, R)
        add = np.zeros(R.shape)
        for idx in np.ndindex(R.shape):
            pts = [P[idx]]
            # the first / last row also test past the edge (the rolled rim reaches beyond the grid)
            if look_lo and idx[-1] == 0:
                pts.append(RF.point(np.array(T[idx] - look_lo), np.array(TH[idx]), np.array(R[idx])))
            if look_hi and idx[-1] == nr - 1:
                pts.append(RF.point(np.array(T[idx] + look_hi), np.array(TH[idx]), np.array(R[idx])))
            for p in pts:
                for bvh, need in targets:
                    loc, nor, fi, dist = bvh.find_nearest(Vector(p), 0.08)
                    if loc is None:
                        continue
                    sd = float(np.dot(p - np.array(loc[:]), np.array(nor[:])))
                    if sd < need:
                        add[idx] = max(add[idx], need - sd)
        if add.max() < 2e-5:
            break
        R = R + np.minimum(add * 1.05, max_step)
        if smooth and it < iters - 1:
            R = np.maximum(R, G.blur_field(R.T, 0.5, 1.0).T)
    return R


def normal_push(P, targets, iters=6, closed=True, look=None):
    """grid points P (nu, nv, 3) moved along each target's surface NORMAL until they are `need` above it (signed), the
    displacement field smoothed over the grid (never below the raw requirement): an offset that hugs a sloped layer
    (a radial push on the trapezius slope needed 3x the distance and flared the collar out over the shoulders).
    look = [(row, offset_fn)] extra probe points per row (past a rim)"""
    P = np.array(P, float)
    nu, nv = P.shape[:2]
    for it in range(iters):
        D = np.zeros_like(P)
        for i in range(nu):
            for j in range(nv):
                probes = [P[i, j]]
                if look and j == 0:
                    probes.append(P[i, 0] + (P[i, 0] - P[i, 1]) * look)       # past the bottom rim
                for q in probes:
                    for tgt in targets:
                        bvh, need = tgt[0], tgt[1]
                        rad = tgt[2] if len(tgt) > 2 else 0.06
                        loc, nor, fi, dist = bvh.find_nearest(Vector(q), rad)
                        if loc is None:
                            continue
                        n = np.array(nor[:]); sd = float(np.dot(q - np.array(loc[:]), n))
                        if sd < need and np.linalg.norm(D[i, j]) < need - sd:
                            D[i, j] = n * (need - sd)
        m = np.linalg.norm(D, axis=-1)
        if m.max() < 2e-5:
            break
        Ds = D.copy()
        for _ in range(2):
            nb = (np.roll(Ds, 1, 0) + np.roll(Ds, -1, 0)) if closed else np.concatenate([Ds[:1], Ds[:-1]]) + np.concatenate([Ds[1:], Ds[-1:]])
            nb2 = np.concatenate([Ds[:, :1], Ds[:, :-1]], 1) + np.concatenate([Ds[:, 1:], Ds[:, -1:]], 1)
            Ds = 0.5 * Ds + 0.125 * (nb + nb2)
        ms = np.linalg.norm(Ds, axis=-1)
        P = P + np.where((ms < m)[..., None], D, Ds) * 1.02
    return P


def fourier_smooth(y, th, n=3, sym=True, lower=True):
    """low-order (cos) Fourier fit of a periodic profile; lower=True shifts it so it never exceeds y (a safe limit)"""
    y = np.asarray(y, float)
    if sym:
        y = 0.5 * (y + np.interp(-th, th, y, period=2 * np.pi))
    A = np.stack([np.cos(k * th) for k in range(n + 1)], 1)
    c, *_ = np.linalg.lstsq(A, y, rcond=None)
    f = A @ c
    if lower:
        f = f - max(0.0, float((f - y).max()))
    return f, c


def gorget(cx):
    """Lame A (spine_05): a collar on the breastplate's neck roll and the mail over the trapezius, top solved below
    the chin; lame B (neck_02): the neck lame at the back and sides (open at the front, where the chin needs the
    room), telescoped inside A; its top solved below the chin / occiput / helmet rim over the head pose grid."""
    mb = G.MB("gorget")
    nf = neck_frame(cx)
    A0, A1, ax, L, axis, RFn, RB, tpar, RBf = (nf[k] for k in ("A0", "A1", "ax", "L", "axis", "RFn", "RB", "tpar", "RBf"))
    front, left = nf["front"], nf["left"]
    t01, t02 = tpar(cx.head("neck_01")), tpar(cx.head("neck_02"))
    k_ = cx.tscale()
    nth = int(os.environ.get("GOR_NTH", "32"))
    th = np.linspace(-np.pi, np.pi, nth, endpoint=False)
    RAf = lambda T, TH: RBf(T, TH) + GAP
    helm_pts = np.array(cx.shared["helmet_mb"].V)[::3] if "helmet_mb" in cx.shared else None
    tB = allowed_top(head_pose_points(cx, "neck_02", helm_pts), A0, A1, lambda T, TH: RBf(T, TH) + 0.0025, th,
                     t_lo=t01 - 0.012 / L)
    tA = allowed_top(head_pose_points(cx, "spine_05", helm_pts), A0, A1, lambda T, TH: RAf(T, TH) + 0.0025, th,
                     t_lo=t01 - 0.110 / L)
    cx.shared["gorget_solve"] = dict(tB=np.round(tB, 4).tolist(), tA=np.round(tA, 4).tolist(), th=np.round(th, 4).tolist())
    # ---- stop surface (body + cuirass) around the neck axis
    from mathutils.bvhtree import BVHTree
    cm = cx.shared.get("cuirass_mb")
    V = [Vector(p) for p in cx.torso.co]; F = [tuple(t) for t in cx.torso.tris]
    if cm is not None:
        off = len(V)
        V += [Vector(p) for p in cm.V]
        F += [tuple(i + off for i in f) for f in cm.F]
    stop = BVHTree.FromPolygons(V, F)
    skin_bvh = BVHTree.FromPolygons([Vector(p) for p in cx.torso.co], [tuple(t) for t in cx.torso.tris])
    cuir_bvh = BVHTree.FromPolygons([Vector(p) for p in cm.V], [(f[0], f[1], f[2]) for f in cm.F] +
                                    [(f[0], f[2], f[3]) for f in cm.F]) if cm is not None else None

    def r_stop(t, T):
        """radius of what lame A rests on at (t, T): the skin + mail (14.5 mm: mail 8.5 + clearance + plate) or the
        cuirass (7.5 mm: its roll + plate + clearance), whichever needs more (iteration 2b: 8.5 mm over bare skin put
        the collar exactly on the mail)"""
        d = math.cos(T) * front + math.sin(T) * left
        o = Vector(axis(t) + d * 0.35)
        best = 0.0
        loc, nor, fi, dist = skin_bvh.ray_cast(o, Vector(-d), 0.35)
        if loc is not None:
            best = max(best, 0.35 - dist + 0.0060)
        if cuir_bvh is not None:
            loc, nor, fi, dist = cuir_bvh.ray_cast(o, Vector(-d), 0.35)
            if loc is not None:
                best = max(best, 0.35 - dist - 0.0010)
        return best
    # ---- lame A profiles (smooth, symmetric): bottom on the neck roll / trapezius, top below the chin
    cf = cx.shared.get("cuirass_fields")
    zf = float(cf["neck_z"](np.array(0.0))) - 0.012 * k_ if cf else float(cx.tz(1.456))
    zb = float(cf["back_z"](np.array(math.pi))) - 0.012 * k_ if cf else float(cx.tz(1.51))
    a = np.abs(th) / D2R
    zs = zf + 0.062 * k_                                          # sides: a collar round the neck (item 20)
    zA0 = zf + (zs - zf) * G.smoothstep(0, 95, a) + (zb - zs) * G.smoothstep(95, 170, a)
    tA0 = np.array([tpar(np.array([0, 0, z])) for z in zA0])
    # the collar never reaches over the shoulder (the pauldrons live there): per column, the bottom rises until the
    # stop surface is within Rcap of the neck axis (a saddle-shaped lower edge)
    # iteration 2b (item 32: the collar flared into a ruff over the shoulders and its saddle edge looked torn): front
    # and back the lower edge laps 1.5 cm over the breast- / backplate's neck border; over the shoulder tops (no plate
    # there) it rises until what it rests on is within Rcap of the neck axis; then one smooth low-order curve
    Rcap = (0.090 + 0.060 * np.cos(th) ** 2) * k_          # iteration 2b: the collar stops nearer the neck at the sides
    if cf:
        thr = np.arctan2(np.sin(th), np.cos(th))
        zfr = np.array([float(cf["neck_z"](np.array(t_))) for t_ in thr]) - 0.015 * k_
        zbk = np.array([float(cf["back_z"](np.array(t_))) for t_ in thr]) - 0.015 * k_
        t_front = np.array([tpar(np.array([0, 0, z])) for z in zfr])
        t_back = np.array([tpar(np.array([0, 0, z])) for z in zbk])
    else:
        t_front = t_back = tA0.copy()
    t_side = tA0.copy()
    for i in range(nth):
        for _ in range(40):
            if r_stop(t_side[i], th[i]) + 0.008 <= Rcap[i]:
                break
            t_side[i] += 0.003 / L
    wf = 1 - G.smoothstep(40, 62, a); wb = G.smoothstep(118, 140, a)
    tA0 = wf * t_front + wb * t_back + (1 - wf - wb) * np.maximum(t_side, np.minimum(t_front, t_back))
    tA0 = 0.5 * (tA0 + np.interp(-th, th, tA0, period=2 * np.pi))
    tA0, _ = fourier_smooth(tA0, th, n=4, sym=True, lower=False)
    band = 0.040 + 0.034 * G.smoothstep(95, 175, a)               # collar height (taller behind the neck)
    tA1 = np.minimum(tA0 + band * k_ / L, tA - 0.002 / L)
    tA1 = np.maximum(tA1, tA0 + 0.030 * k_ / L)
    tA1 = 0.5 * (tA1 + np.interp(-th, th, tA1, period=2 * np.pi))
    tA1 = np.minimum(np.minimum(tA1, np.roll(tA1, 1)), np.roll(tA1, -1))
    tA1 = G.blur_field(tA1[None, :], 0, 1.3)[0]
    tA1 = np.minimum(tA1, t01 + 0.006 / L)
    H_ = cx.shared.get("helm")
    if H_ is not None:                                            # below the helmet's lower rim (the neck lame B
        zr = H_.zrim(th) - 0.030                                  # rises under it, lame A stays on the cuirass)
        tA1 = np.minimum(tA1, np.array([tpar(np.array([0, 0, z])) for z in zr]))
    tA1 = np.maximum(tA1, tA0 + 0.028 * k_ / L)
    rows = 5
    v = np.linspace(0, 1, rows)
    hv = np.array(cx.shared["helmet_mb"].V) if "helmet_mb" in cx.shared else None
    for it in range(3):
        T = tA0[:, None] + (tA1 - tA0)[:, None] * v[None, :]
        THg = np.broadcast_to(th[:, None], T.shape)
        Rtop = RAf(T, THg)
        Rs = np.array([[r_stop(T[i, j], th[i]) for j in range(rows)] for i in range(nth)])
        # smooth upper envelope of what the collar rests on (no ray noise in the plate)
        Rs = G.envelope(Rs, 1, 2, 0.6, 1.5, wrap_th=True) if Rs.shape[0] > 4 else Rs
        Rs = G.envelope(Rs.T, 2, 1, 1.5, 0.6, wrap_th=True).T
        R = np.maximum(Rtop, Rs + 0.0085)
        R = G.blur_field(R.T, 0.6, 1.2).T
        R = np.maximum(R, Rs + 0.0075)
        R[:, 0] += 0.003                                              # the bottom edge flares a little
        P = RFn.point(T, THg, R)
        if hv is None:
            break
        # the helmet (rest pose) must stay >= 2.2 cm above lame A's top border (its rim + A's rim + clearance)
        hr = hv - A0
        ht = hr @ ax / L
        hp = hr - np.outer(hr @ ax, ax)
        hth = np.arctan2(hp @ left, hp @ front); hrad = np.linalg.norm(hp, axis=1)
        low = False
        for i in range(nth):
            m = (angdist(hth, th[i]) < 9 * D2R) & (hrad > R[i, -1] - 0.012)
            if not m.any():
                continue
            tmin = float(ht[m].min())
            lim = tmin - 0.022 / L
            if tA1[i] > lim:
                tA1[i] = max(lim, tA0[i] + 0.022 / L); low = True
        if not low:
            break
        tA1 = np.minimum(tA1, G.blur_field(tA1[None, :], 0, 0.8)[0])
    # ---- lame B's table first (iteration 2b): both lames keep a NORMAL clearance over what is under them (the radial
    # offsets of the neck field put lame B's lower part INTO the trapezius at the neck base, and the mail through it),
    # then lame A is pushed out over lame B where they overlap
    rowsB = 4
    tB1 = np.minimum(tB, t02 + 0.075 / L)
    tB1 = 0.5 * (tB1 + np.interp(-th, th, tB1, period=2 * np.pi))       # symmetric
    tB1 = np.minimum(np.minimum(tB1, np.roll(tB1, 1)), np.roll(tB1, -1))
    tB1 = G.blur_field(tB1[None, :], 0, 1.0)[0]
    # B's bottom laps 1.4 cm inside A's top; only at the front (where the chin keeps B's top low) it hides deeper in A
    wfr = 1 - G.smoothstep(50, 80, np.abs(th) / D2R)
    B0 = tA1 - 0.014 / L - wfr * np.maximum(0.0, (tA1 - 0.014 / L) - (tB1 - 0.020 / L))
    B0 = np.minimum(B0, G.blur_field(B0[None, :], 0, 1.0)[0])
    tB1 = np.maximum(tB1, B0 + 0.020 / L)
    vB = np.linspace(0, 1, rowsB)
    TB = B0[:, None] + (tB1 - B0)[:, None] * vB[None, :]
    THb = np.broadcast_to(th[:, None], TB.shape)
    RBv = RBf(TB, THb)
    need_mail = MAIL_OFF + 0.0025 + GOR_T
    mm_ = cx.shared.get("mail_mb")
    mail_bvh = G.mb_bvh(mm_, lambda t: True) if mm_ is not None else None
    tgB = [(skin_bvh, need_mail)] + ([(mail_bvh, 0.0030 + GOR_T)] if mail_bvh is not None else [])
    RBv = radial_push(RFn, TB, THb, RBv, tgB, iters=3, max_step=0.002)
    PB_ = normal_push(RFn.point(TB, THb, RBv), tgB, closed=True)
    RBv = np.linalg.norm(PB_ - (A0 + np.einsum("ijk,k->ij", PB_ - A0, ax)[..., None] * ax), axis=-1)
    # lame A outside B (+ plate + GAP) where they overlap, and over the skin / cuirass (normal clearance)
    for i in range(nth):
        for j in range(rows):
            if T[i, j] <= tB1[i] and T[i, j] >= B0[i] - 0.004 / L:
                rb = float(np.interp(T[i, j], TB[i], RBv[i]))
                R[i, j] = max(R[i, j], rb + GOR_T + GAP)
    mm_ = cx.shared.get("mail_mb")
    mail_bvh = G.mb_bvh(mm_, lambda t: True) if mm_ is not None else None
    tg = [(skin_bvh, need_mail)] + ([(cuir_bvh, 0.0025 + GOR_T)] if cuir_bvh is not None else []) + \
        ([(mail_bvh, 0.0030 + GOR_T)] if mail_bvh is not None else [])
    R = radial_push(RFn, T, THg, R, tg, look_lo=0.009 / L, iters=3, max_step=0.002)
    P = RFn.point(T, THg, R)
    # lame B's outer surface under lame A (only where it lies within 2 cm: A's inside clears it by GAP)
    FB = [(i * rowsB + j, (i + 1) % nth * rowsB + j, (i + 1) % nth * rowsB + j + 1, i * rowsB + j + 1)
          for i in range(nth) for j in range(rowsB - 1)]
    PBf = PB_.reshape(-1, 3)
    bB = BVHTree.FromPolygons([Vector(v) for v in PBf], FB)
    P = normal_push(P, tg + [(bB, GAP + GOR_T, 0.02)], closed=True, look=0.6)
    # session 3 (G1 lame A one-sided 1.9 %, G4 gorget own components): at the sides the squeezed collar's rows
    # zig-zagged (consecutive row segments reversed: a Z-fold in the plate): folded columns are replaced by the straight
    # line from their bottom to their top row, pushed out radially by the largest amount any original row stood
    # outside that line (the clearances the pushes found are kept), then blended with the neighbours
    nfold = 0
    Pn = P.copy()
    for i in range(nth):
        d = np.diff(P[i], axis=0); L_ = np.linalg.norm(d, axis=1)
        cosr = [float(np.dot(d[j], d[j + 1]) / max(L_[j] * L_[j + 1], 1e-12)) for j in range(len(d) - 1)]
        if min(cosr) >= 0.2:
            continue
        nfold += 1
        vv = np.linspace(0, 1, rows)
        Q = P[i, 0][None, :] + (P[i, -1] - P[i, 0])[None, :] * vv[:, None]
        foot = A0 + np.outer((Q - A0) @ ax, ax)
        rd = G.nrm(Q - foot)
        def rad(X):
            f_ = A0 + np.outer((X - A0) @ ax, ax)
            return np.linalg.norm(X - f_, axis=1)
        deficit = float(max(0.0, (rad(P[i]) - rad(Q))[1:-1].max())) if rows > 2 else 0.0
        bump = np.sin(np.pi * vv)[:, None]
        Pn[i] = Q + rd * deficit * bump
    if nfold:
        P = Pn
        log_("gorget lame A: %d folded columns straightened" % nfold)
    if os.environ.get("RTS_DEBUG_GORGET"):
        for i in range(nth):
            if abs(abs(th[i]) - math.pi / 2) < 0.35:
                d = np.diff(P[i], axis=0); L_ = np.linalg.norm(d, axis=1)
                cosr = [float(np.dot(d[j], d[j + 1]) / max(L_[j] * L_[j + 1], 1e-12)) for j in range(len(d) - 1)]
                log_("gorgetdbg th %5.0f rows %s seglen mm %s cos %s" % (th[i] / D2R, np.round(P[i][:, 2], 3).tolist(),
                     np.round(L_ * 1000, 1).tolist(), np.round(cosr, 2).tolist()))
    # closed thick collar (iteration 2b, user items 28 / 32): outer + inner surface, rolled rims turned under to it; the
    # etched band lies in the plate surface under the top rim
    foot = lambda X: A0 + np.outer((X - A0) @ ax, ax)
    P, _, etchA = G.etch_grid(P, None, ["j1"], 0.013 * k_, closed=True)
    def rimA(Pl, Nl):
        tt = float(np.mean((Pl - A0) @ ax / L))
        if tt < float(np.mean(0.5 * (tA0 + tA1))):          # the bottom edge: a narrow rolled bead on the cuirass
            return G.rim_bead(bead=0.0030, t=0.002, lip=0.003, band="gold_bead")
        # session 3 (G4 gorget lame A|B, G1 lame A one-sided at the sides): the top edge's 7 mm gold band + bead ran on
        # 14 mm past the border, i.e. toward the neck where the collar lies flat on the trapezius, into lame B: a
        # rolled gold bead only (the etched band inside the plate keeps the gold line)
        return G.rim_bead(bead=0.0036, t=0.002, lip=0.003, band="gold_bead")
    G.shell(mb, P, closed=True, inside=foot, w={"spine_05": 1.0}, tag="lameA",
            rim=rimA, uvfn=lambda U, V_, P_: TX.steel_uv(U, V_), inner=True, thick=GOR_T, etch=etchA)
    # ---- lame B (neck_02): a FULL neck ring telescoped inside A (iteration 2b: the back / sides-only lame ended in
    # jagged column stubs at the front, user item 32). Where the chin needs the room (front) its top stays low and
    # the whole lame hides inside A; at the back / sides it rises under the helmet's rim.
    PB = PB_
    cx.shared["gorget_B"] = dict(TB=TB, RB=RBv, th=th, tA1=tA1)

    def rimB(Pl, Nl):
        prof = G.rim_gold(w=0.007, bead=0.0034, t=0.0025, lip=0.006, band="fil_narrow")
        tt = (Pl - A0) @ ax / L
        Tq = np.arctan2((Pl - A0) @ left, (Pl - A0) @ front)
        b0 = np.interp(Tq, th, B0, period=2 * np.pi)
        ta = np.interp(Tq, th, tA1, period=2 * np.pi)
        # the bottom edge and any part of the top edge still inside A are hidden: flat turned edge there
        return prof, smooth_amp(np.where((tt < b0 + 0.006 / L) | (tt < ta + 0.004 / L), 0.0, 1.0))
    G.shell(mb, PB, closed=True, inside=foot, w={"neck_02": 1.0}, tag="lameB",
            rim=rimB, uvfn=lambda U, V_, P_: TX.steel_uv(U + 0.5, V_ + 0.2), inner=True, thick=GOR_T)
    log_("gorget: A bottom %.3f..%.3f top %.3f..%.3f; B full ring, bottom %.3f..%.3f top %.3f..%.3f" % (
        zA0.min(), zA0.max(), float(axis(tA1.min())[2]), float(axis(tA1.max())[2]),
        float(axis(B0.min())[2]), float(axis(B0.max())[2]), float(axis(tB1.min())[2]), float(axis(tB1.max())[2])))
    return mb


GOR_T = 0.002            # gorget plate thickness (closed shells)


# ======================================================================================================= ARMS
def arm_body(cx, side):
    if not hasattr(cx, "_arm"):
        cx._arm = {}
    if side not in cx._arm:
        w = cx.body.weight_sum(["upperarm_" + side, "upperarm_twist_01_" + side, "lowerarm_" + side,
                                "lowerarm_twist_01_" + side, "hand_" + side])
        cx._arm[side] = cx.body.subset(w > 0.25)
    return cx._arm[side]


def limb_frame(cx, h, t, side, up=(0.75, 0, 0.66)):
    ax = G.nrm(t - h)
    upv = np.array(up) * np.array([side_sign(side), 1, 1])
    o = G.nrm(upv - np.dot(upv, ax) * ax)
    f = G.nrm(np.cross(ax, o)) * side_sign(side)
    if f[1] > 0:
        f = -f
    return ax, o, f


def limb_field(cx, h, t, side, t0=-0.15, t1=1.15, up=(0.75, 0, 0.66), nth=48):
    ax, o, f = limb_frame(cx, h, t, side, up)
    axis = lambda s: h + s * (t - h)
    frame = lambda s: (o, f)
    return G.RadField(arm_body(cx, side), axis, frame, t0, t1, nt=40, nth=nth, env=(2, 2, 1.5, 1.5), maxd=0.25,
                      default=0.04)


def arm_joints(cx, side):
    S = cx.head("upperarm_" + side); E = cx.head("lowerarm_" + side); W = cx.head("hand_" + side)
    return S, E, W


def elbow_dirs(cx, side):
    S, E, W = arm_joints(cx, side)
    a = G.nrm(E - S); b = G.nrm(W - E)
    de = G.nrm(a - b)                                   # posterior elbow point direction
    lat = G.nrm(np.cross(a, b)); lat = lat if lat[0] * side_sign(side) > 0 else -lat
    return a, b, de, lat


ARM_T = 0.0022           # arm plate thickness (closed shells, iteration 2b)
ARM_GAP = 0.0080         # rerebrace / vambrace flare over the couter sphere: plate + the couter's rim bead + clearance


def axis_foot(A, B):
    """per-vertex 'inside' points for shell(): the foot of each vertex on the line A-B (limb tubes, flared ends)"""
    A = np.asarray(A, float); d = G.nrm(np.asarray(B, float) - A)
    return lambda X: A + np.outer((X - A) @ d, d)


def blur_tth(R, sig_t, sig_th, wrap=True):
    """blur a (n_theta, n_t) grid: theta along axis 0 (wrapping for closed tubes), t along axis 1"""
    R = np.asarray(R, float)
    if R.ndim < 2:
        return R
    return G.blur_field(R.T, sig_t, sig_th, wrap_th=wrap).T


def theta_of(o, f, d):
    return math.atan2(float(np.dot(d, f)), float(np.dot(d, o)))


def angdist(th, th0):
    return np.abs(np.arctan2(np.sin(th - th0), np.cos(th - th0)))


def min_sphere(P, C, Rmin, axis_p, axis_d, wmask=None):
    """push grid points radially away from the limb axis so that |p - C| >= Rmin (where wmask > 0)"""
    Q = P.copy()
    for idx in np.ndindex(P.shape[:-1]):
        p = P[idx]
        wv = 1.0 if wmask is None else float(wmask[idx])
        if wv <= 0 or np.linalg.norm(p - C) >= Rmin:
            continue
        foot = axis_p + np.dot(p - axis_p, axis_d) * axis_d
        rv = p - foot; r = np.linalg.norm(rv)
        u = rv / max(r, 1e-9)
        # |foot + s u - C| = Rmin, s >= r
        fc = foot - C
        b = np.dot(fc, u); c = np.dot(fc, fc) - Rmin * Rmin
        sd = -b + math.sqrt(max(b * b - c, 0.0))
        Q[idx] = p + (foot + u * max(sd, r) - p) * wv
    return Q


def limit_sphere(P, C, Rmax, axis_p, axis_d, wmask=None):
    """pull grid points toward the limb axis so that |p - C| <= Rmax (where wmask > 0)"""
    Q = P.copy()
    for idx in np.ndindex(P.shape[:-1]):
        p = P[idx]
        if wmask is not None and wmask[idx] <= 0:
            continue
        if np.linalg.norm(p - C) <= Rmax:
            continue
        foot = axis_p + np.dot(p - axis_p, axis_d) * axis_d
        rv = p - foot; r = np.linalg.norm(rv)
        # solve |foot + s rv/r - C| = Rmax for s in [0, r]
        lo, hi = 0.0, r
        for _ in range(30):
            m = 0.5 * (lo + hi)
            if np.linalg.norm(foot + rv / max(r, 1e-9) * m - C) > Rmax:
                hi = m
            else:
                lo = m
        q = foot + rv / max(r, 1e-9) * lo
        wv = 1.0 if wmask is None else float(wmask[idx])
        Q[idx] = p + (q - p) * wv
    return Q


PAUL_A = 108.0           # cop half extent in alpha (deg); session 4 tried 100 / 90 (user item 25 'bulky': the ends
PAUL_AL = 98.0           # lame half extent             rise to 19 cm from the joint over the plates): the cape clasps
                         #                              (armour_lower, placed on the pauldron's back end) then moved to
                         #                              the nape and the helm / plume cut them: needs a joint change
LAME_STEP, LAME_HIDE = 8.0, 11.0
N_LAMES = 2
LAME_VIS = 6.5            # deg of each lame visible below the plate above
COP_RIM_DEG = 8.4         # angular extent of the cop's bottom rim (rim_gold w 10 mm, bead 4.4 mm at ~0.13 m)
LAME_RIM_DEG = 2.6        # a lame's bottom bead


def pauldron_layout(cx, side):
    """sphere family of the pauldron around the shoulder joint (shared with the rerebrace / QA)"""
    key = "paul_" + side
    if key in cx.shared:
        return cx.shared[key]
    sg = side_sign(side)
    S, E, W = arm_joints(cx, side)
    sp = Sph(S, E - S, np.array([sg, 0, 0.0]), fw_sign=sg)
    Gs = np.linspace(20, 165, 59) * D2R; As = np.linspace(-135, 135, 55) * D2R
    Rs = skin_radius(cx.body, sp, Gs, As)
    k = cx.tscale()
    g_d0 = lambda a: (63.0 + 6.0 * (np.abs(a) / (PAUL_A * D2R)) ** 2) * D2R
    g_d1 = lambda a: (98.0 + 52.0 * np.clip(np.cos(np.abs(a) * 90 / PAUL_A), 0, 1) ** 1.3) * D2R
    # iteration 2b (item 25): TWO lames, each showing a real band of steel below the rim of the plate above it, and
    # tucked LAME_HIDE under it (the lames lag 8-11 deg behind each other in the idle hang: they stay nested)
    bands = []
    top, rim_above = (lambda a: g_d0(a) + LAME_HIDE * D2R), COP_RIM_DEG
    b_prev = g_d0
    for j in range(N_LAMES):
        bot = (lambda a, b_prev=b_prev, ra=rim_above: b_prev(a) - (ra + LAME_VIS) * D2R)
        top = (lambda a, b_prev=b_prev: b_prev(a) + LAME_HIDE * D2R)
        bands.append((top, bot))
        b_prev, rim_above = bot, LAME_RIM_DEG
    CL, CLD = 0.0135, 0.016           # clearance of the lames / cop over the skin (mail 8.5 mm + 5 mm; was 19 / 22)
    CL_LAST = RERE_OFF + 0.0025 + PAUL_T   # the rerebrace's outer surface + 2.5 mm + the lame plate (item 25: was 20.5 mm,
                                            # with the rerebrace at 18 mm the last lame's inside sat ON it)
    need = np.zeros(len(As))
    for jj, a in enumerate(As):
        g = np.linspace(g_d0(a), 100 * D2R, 12)
        need[jj] = max(need[jj], float((interp2(Gs, As, Rs, g, a) + CLD).max()))
        for j, (top, bot) in enumerate(bands):
            g = np.linspace(bot(a) - (4 * D2R if j == len(bands) - 1 else 0.0), top(a), 7)
            cl = CL_LAST if j == len(bands) - 1 else CL          # the last lame covers the rerebrace top + the mail
            need[jj] = max(need[jj], float((interp2(Gs, As, Rs, g, a) + cl).max()) + (j + 1) * PAUL_GAP)
    lat_need = float(np.interp(0.0, As, need))
    need = np.minimum(need, lat_need + 0.016)              # the ends toward the armpit do not balloon
    # near-spherical family: the lames slide about the shoulder joint without cutting each other when the arm swings
    # forward / back (a swing mixes azimuths; a radius varying with azimuth would make the shells collide)
    need = np.maximum(need, lat_need - 0.010)          # item 25: front / back follow the shoulder closer (was -6 mm)
    # iteration 2b (G4 cuirass|pauldron, gorget|pauldron): the cop's spherical zone lies OVER the back- / breastplate
    # and the gorget's collar wherever it reaches them (rays from the shoulder joint: their outermost surface + 2.5 mm
    # + the plate)
    under = [cx.shared.get(k) for k in ("cuirass_mb", "gorget_mb")]
    ub = [G.mb_bvh(m, lambda t: True) for m in under if m is not None]
    if ub:
        for jj, a in enumerate(As):
            for g in np.linspace(g_d0(a), 100 * D2R, 7):
                dv = sp.d(g, a)
                for bv in ub:
                    loc, nor, fi, dist = bv.ray_cast(Vector(S + dv * 0.3), Vector(-dv), 0.3)
                    if loc is not None:
                        need[jj] = max(need[jj], 0.3 - dist + 0.0025 + PAUL_T)
    Rb = G.envelope(need[None, :], 0, 3, 0, 2.0, wrap_th=False)[0]
    Rb = np.maximum(Rb, need)
    Rbase = lambda a: np.interp(np.asarray(a, float), As, Rb)
    lay_under = ub
    lay = dict(sp=sp, Gs=Gs, As=As, Rs=Rs, g_d0=g_d0, g_d1=g_d1, bands=bands, Rbase=Rbase, under=lay_under)
    cx.shared[key] = lay
    log_("pauldron %s: R_base %.3f..%.3f (lateral %.3f); by azimuth %s" % (side, Rb.min(), Rb.max(), float(Rbase(0.0)),
         " ".join("%d:%.0f" % (round(a / D2R), 1000 * float(Rbase(a))) for a in np.linspace(-PAUL_A, PAUL_A, 13) * D2R)))
    return lay


def pauldron(cx, side):
    sg = side_sign(side)
    mb = G.MB("pauldron_" + side)
    L = pauldron_layout(cx, side)
    sp, g_d0, g_d1, bands, Rbase, Gs, As, Rs = (L[k] for k in ("sp", "g_d0", "g_d1", "bands", "Rbase", "Gs", "As", "Rs"))
    S = sp.O
    helpers = ["upperarm_helper_01_" + side, "upperarm_helper_02_" + side, "upperarm_helper_03_" + side,
               "upperarm_" + side]
    # ---- cop (dome): sphere zone over the lames, pulls in toward the shoulder top, upturned neck edge
    na, ng = 19, 10
    a_ = np.linspace(-PAUL_A, PAUL_A, na) * D2R
    v = np.linspace(0, 1, ng)
    GA = g_d0(a_)[:, None] + (g_d1(a_) - g_d0(a_))[:, None] * v[None, :]
    AA = np.broadcast_to(a_[:, None], GA.shape)
    Rt = interp2(Gs, As, Rs, GA, AA) + 0.026
    R0 = Rbase(AA)
    s = G.smoothstep(100 * D2R, 140 * D2R, GA)
    R = R0 + (np.minimum(Rt, R0) - R0) * s
    R = R + 0.008 * G.smoothstep(0.82, 1.0, v)[None, :] * (np.abs(AA) < 100 * D2R)
    R = G.blur_field(R, 0.0, 0.8, wrap_th=False)
    # the part that follows the shoulder toward the neck lies over the gorget's collar / the plates (+ 2.5 mm + plate)
    for idx in np.ndindex(R.shape):
        # the neck-side rows also look ahead over the rolled rim (it reaches ~2 cm past the grid's edge)
        exts = [(0.0, 0.0)] + ([(4 * D2R, 0.0), (9 * D2R, 0.0)] if v[idx[1]] > 0.7 else [])
        if idx[0] in (0, R.shape[0] - 1):                   # the front / back ends: their rim reaches on in azimuth
            sa = -1.0 if idx[0] == 0 else 1.0
            exts += [(0.0, sa * 4 * D2R), (0.0, sa * 8 * D2R)]
        for ge, ae in exts:
            dv = sp.d(GA[idx] + ge, AA[idx] + ae)
            for bv in L.get("under", []):
                loc, nor, fi, dist = bv.ray_cast(Vector(S + dv * 0.3), Vector(-dv), 0.3)
                if loc is not None:
                    R[idx] = max(R[idx], 0.3 - dist + 0.0025 + PAUL_T + (0.004 if (ge > 0 or ae != 0) else 0.0015 * G.smoothstep(0.8, 1.0, v[idx[1]])))
    R = np.maximum(R, G.blur_field(R, 0.0, 0.8, wrap_th=False))
    P = sp.pt(GA, AA, R)
    # session 3 (G4 pauldron#cop|#plaque, 44-47 points in every pose incl. bind): the plaque's lower third hung over the
    # cop's lower rim bead and the first lame (centred at 80 deg with half its 15 cm height = 26 deg below): its bottom
    # now sits 2 deg above the cop's rim band
    k_pl = 0.86 * cx.tscale()
    half_g = 0.5 * 0.150 * k_pl / float(Rbase(0.0))
    g_lion = float(g_d0(0.0)) + (COP_RIM_DEG + 2.0) * D2R + half_g
    lion_c = sp.pt(g_lion, 0.0, float(Rbase(0.0)))
    lion_n = G.nrm(lion_c - S)
    ey = G.nrm(sp.d(90 * D2R, 0.0) - np.dot(sp.d(90 * D2R, 0.0), lion_n) * lion_n)
    ey = G.nrm(-sp.u - np.dot(-sp.u, lion_n) * lion_n)          # 'up' on the cop = up the arm
    ex = np.cross(ey, lion_n) * sg

    prof = G.rim_gold(w=0.010, bead=0.0044, t=0.002, lip=0.0, band="fil_wide")
    P, _, ec = G.etch_grid(P, None, ["j0", "j1", "i0", "i1"], 0.016)
    G.shell(mb, P, inside=S, w={helpers[0]: 1.0}, tag="cop", uvfn=lambda U, V_, P_: TX.steel_uv(U + 2.6, V_ + 0.2),
            inner=True, thick=PAUL_T, rim=lambda Pl, Nl: prof, inner_uv=lambda U, V_, P_: TX.steel_uv(U + 2.2, V_ + 0.4),
            etch=ec, sphere=S)
    cop_bvh = G.mb_bvh(mb, lambda t: t == "cop")
    lion_plaque(mb, sp, lambda g, a: cop_radius(L, g, a), lion_c, ex, ey, helpers[0], k=k_pl, seat=cop_bvh)
    # stop-rib across the cop near its neck edge (closed, seated)
    ar = np.linspace(-1, 1, 29) * 82.0 * (PAUL_A / 108.0) * D2R      # session 4: spans the (narrower) cop as before
    gr = g_d1(ar) - 11 * D2R
    rr = Rbase(ar) + (np.minimum(interp2(Gs, As, Rs, gr, ar) + 0.026, Rbase(ar)) - Rbase(ar)) * G.smoothstep(100 * D2R, 140 * D2R, gr)
    pr = sp.pt(gr, ar, rr); nr = G.nrm(pr - S)
    pr = G.seat_points(cop_bvh, pr, nr, lift=0.0)
    G.sweep(mb, pr, nr, RIB6(0.0045, 0.0050), band="gold_bead", w={helpers[0]: 1.0}, tag="stoprib", taper=0.10,
            solid=True, seat=cop_bvh)
    # rivets along the cop's lower edge
    for a in (-70, 0, 70):
        a = a * D2R; g = g_d0(a) + 5.5 * D2R
        c = sp.pt(g, a, float(Rbase(a))); G.rivet(mb, c, G.nrm(c - S), w={helpers[0]: 1.0}, seat=cop_bvh)
    # ---- lames: concentric bands PAUL_GAP apart, top edge hidden under the plate above (closed thick plates)
    nal = 17
    al = np.linspace(-PAUL_AL, PAUL_AL, nal) * D2R
    e = np.abs(np.linspace(-1, 1, nal))
    pinch = 0.45 * (1 - np.sqrt(np.clip(1 - np.clip((e - 0.72) / 0.28, 0, 1) ** 2, 0, 1)))
    for j, (top, bot) in enumerate(bands):
        vv = np.linspace(0, 1, 4)
        gt, gb = top(al), bot(al)
        gm = 0.5 * (gt + gb)
        gt2 = gm + (gt - gm) * (1 - pinch); gb2 = gm + (gb - gm) * (1 - pinch)
        GA = gb2[:, None] + (gt2 - gb2)[:, None] * vv[None, :]
        AA = np.broadcast_to(al[:, None], GA.shape)
        R = Rbase(AA) - (j + 1) * PAUL_GAP + 0.0006 * (1 - vv[None, :]) ** 2     # lower edge flares 0.6 mm
        P = sp.pt(GA, AA, R)
        # (session 3 tried the last lame on the upper arm itself: pauldron|rerebrace unchanged, cop|lame / lame|lame
        # pairs x2-3 in the clips; reverted)
        bn = helpers[j + 1]

        def rim_l(Pl, Nl, top=top):
            g, a, _ = sp.ga(Pl)
            hid = g > top(a) - 1.2 * D2R
            return G.rim_bead(bead=0.0034, t=0.002, lip=0.0045, band="gold_bead"), smooth_amp(np.where(hid, 0.0, 1.0))
        tag = "lame%d" % j
        G.shell(mb, P, inside=S, w={bn: 1.0}, tag=tag, rim=rim_l, inner=True, thick=PAUL_T, sphere=S,
                uvfn=lambda U, V_, P_, j=j: TX.steel_uv(U + 0.35 * j, V_ + 0.1 * j))
        lb = G.mb_bvh(mb, lambda t, tag=tag: t == tag)
        for a in (-(PAUL_AL - 9), PAUL_AL - 9):
            a = a * D2R; g = float(bot(a)) + (LAME_RIM_DEG + 0.45 * LAME_VIS) * D2R
            c = sp.pt(g, a, float(Rbase(a)) - (j + 1) * PAUL_GAP)
            G.rivet(mb, c, G.nrm(c - S), r=0.0036, w={bn: 1.0}, seat=lb)
    return mb


def cop_radius(L, g, a):
    """radius of the pauldron cop's outer surface at sphere coords (g, a) (the cop grid's formula, no neck flare)"""
    Rt = interp2(L["Gs"], L["As"], L["Rs"], g, a) + 0.026
    R0 = L["Rbase"](a)
    return R0 + (np.minimum(Rt, R0) - R0) * G.smoothstep(100 * D2R, 140 * D2R, g)


def lion_plaque(mb, sp, Rfn, c, ex, ey, bone, k=1.0, W=0.128, Hh=0.150, lift=0.0027, seat=None):
    """Embossed lion plaque (judge M6, reference 'Shoulder Detail'): a heater-shaped plate riveted on the cop, 2.6 mm
    proud with a rolled gold edge; UVs into the trim sheet's LION square (lion rampant in its scroll frame, relief in
    the normal map)."""
    W, Hh = W * k, Hh * k
    us = np.linspace(-1, 1, 9); vs = np.linspace(0, 1, 11)
    P = np.zeros((len(us), len(vs), 3)); XY = np.zeros((len(us), len(vs), 2))
    for j, v in enumerate(vs):
        half = 0.5 * W * (1.0 if v < 0.42 else max(0.10, math.sqrt(max(0.0, 1 - ((v - 0.42) / 0.58) ** 2))))
        y = 0.5 * Hh - v * Hh
        for i, u in enumerate(us):
            x = u * half
            q = c + ex * x + ey * y
            g, a, _ = sp.ga(q)
            P[i, j] = sp.pt(g, a, float(Rfn(g, a)) + lift)
            XY[i, j] = (x, y)
    if seat is not None:                      # the plaque's underside sits ON the cop (0.1 mm): project, then lift
        Pf = P.reshape(-1, 3)
        dn = G.nrm(Pf - sp.O)
        P = (G.seat_points(seat, Pf, dn, lift=0.0) + dn * lift).reshape(P.shape)
    size = 1.07 * W / (0.285 / 0.30)                   # the frame decal fills 95 % of the square
    uvq = lambda x, y: TX.square_uv("lion", x, y + 0.02 * Hh, size)

    def uvfn(U, V_, P_):
        return np.array([[uvq(*XY[i, j]) for j in range(len(vs))] for i in range(len(us))])
    r = 0.0020
    prof = G.prof((0.6 * r, 0.9 * r, "gold_bead", 0.0, 0.5), (1.5 * r, -0.2 * r, "gold_bead", 0.5, 1.0))
    G.shell(mb, P, inside=sp.O, w={bone: 1.0}, tag="plaque", uvfn=uvfn, rim=lambda Pl, Nl: prof, inner=True,
            thick=lift - 0.0013)           # underside 1.3 mm over the cop: its chords never cut the cop's facets
                                           # (0.7 mm left 100 crossings on the cop's convex facet creases; session 3)
    pb = G.mb_bvh(mb, lambda t: t == "plaque")
    for (x, y) in ((0.0, 0.40 * Hh), (0.0, -0.40 * Hh)):
        q = c + ex * x + ey * y
        g, a, _ = sp.ga(q)
        p = sp.pt(g, a, float(Rfn(g, a)) + lift)
        G.rivet(mb, p, G.nrm(p - sp.O), r=0.0030, w={bone: 1.0}, seat=pb)


def rerebrace(cx, side):
    """Upper-arm plate (upperarm): top tucked under the pauldron's last lame (limited per point to that lame's sphere
    radius - 4.5 mm), bottom under the couter; cut back at the inner elbow and at the front / back top, where no lame
    covers it (user item 17: shoulder overlap)."""
    mb = G.MB("rerebrace_" + side)
    S, E, W = arm_joints(cx, side)
    RF = limb_field(cx, S, E, side)
    ax, o, f = limb_frame(cx, S, E, side)
    a_, b_, de, lat = elbow_dirs(cx, side)
    thp = theta_of(o, f, de)                                  # posterior (elbow point) azimuth
    L = pauldron_layout(cx, side)
    sp = L["sp"]
    Rc = couter_radius(cx, side)
    nth, nt = 25, 7
    # session 3 (G4 cuirass|rerebrace, 18-56 visible points in idle / walk): the plate's medial edges ran into the
    # breast- / backplate's arm-hole rims at the arm-pit: the medial opening is 30 deg wider and the medial top starts
    # lower (the mail shows there, as on the reference)
    th = np.linspace(-135, 135, nth) * D2R
    TH = np.broadcast_to(th[:, None], (nth, nt))
    far = angdist(th, thp) / np.pi                            # 0 posterior, 1 anterior (inner elbow)
    tb = 0.845 - 0.10 * G.smoothstep(0.45, 1.0, far)          # cut back on the inner side (elbow bend)
    # top: the lateral half hides under the lames (from t 0.30), front / back / medial start lower
    lat_th = theta_of(o, f, G.nrm(np.array([side_sign(side), 0, 0.3])))
    side_off = angdist(th, lat_th) / np.pi                    # 0 lateral, 1 medial
    t0 = 0.30 + 0.20 * G.smoothstep(0.30, 0.75, side_off)
    T = t0[:, None] + (tb - t0)[:, None] * np.linspace(0, 1, nt)[None, :]
    R = blur_tth(RF(T, TH), 1.0, 2.0, wrap=False) + RERE_OFF
    P = RF.point(T, TH, R)
    # under the last lame: never outside (lame radius at this point's sphere direction) - 4.5 mm, where the lames
    # cover it (sphere angle g above the last lame's lower edge)
    Rl = lambda g, a: L["Rbase"](a) - len(L["bands"]) * PAUL_GAP - PAUL_T - 0.0025
    bot2 = L["bands"][-1][1]
    for idx in np.ndindex(P.shape[:-1]):
        g, a, r = sp.ga(P[idx])
        lim = float(Rl(g, a))
        gb = float(bot2(a))
        wv = float(G.smoothstep(gb - 6 * D2R, gb - 2 * D2R, g))   # also under the lame's bottom bead / rim
        wv *= float(1 - G.smoothstep((PAUL_AL + 4) * D2R, (PAUL_AL + 16) * D2R, abs(a)))  # where a lame (+ its rim) covers it
        if r > lim and wv > 0:
            foot = S + np.dot(P[idx] - S, ax) * ax
            rv = P[idx] - foot
            # radial pull toward the arm axis until inside the limit sphere (bisection), blended by wv
            lo, hi = 0.0, 1.0
            for _ in range(24):
                m = 0.5 * (lo + hi)
                if np.linalg.norm(foot + rv * m - S) > lim:
                    hi = m
                else:
                    lo = m
            P[idx] = P[idx] + (foot + rv * lo - P[idx]) * wv
    # the couter slides UNDER the rerebrace's lower edge (concentric around the elbow joint: no contact in any bend)
    P = min_sphere(P, E, Rc + ARM_GAP, S, ax, wmask=G.smoothstep(0.50, 0.62, T))
    P = G.laplace_grid(P, iters=2, lam=0.35, fix_border=False)
    hide_t = 0.36

    def rim_r(Pl, Nl):
        t = (Pl - S) @ ax / np.linalg.norm(E - S)
        amp = smooth_amp(np.where(t < hide_t, 0.0, 1.0))
        # the part of the border that rides the couter (near the elbow) follows the couter's sphere family
        near = np.linalg.norm(Pl - E, axis=1) < Rc + ARM_GAP + 0.012
        if near.mean() > 0.5:
            return (G.rim_gold(w=0.006, bead=0.003, t=0.002, lip=0.006, band="fil_narrow"), amp, None, E)
        return (G.rim_gold(w=0.006, bead=0.003, t=0.002, lip=0.006, band="fil_narrow"), amp)
    P, _, er = G.etch_grid(P, None, ["j1"], 0.014)
    G.shell(mb, P, inside=axis_foot(S, E), w={"upperarm_" + side: 1.0}, tag="rere", rim=rim_r,
            uvfn=lambda U, V_, P_: TX.steel_uv(U, V_), inner=True, thick=ARM_T, etch=er)
    return mb


COUTER_RHO = lambda p: (44 + 24 * np.clip(np.cos(p), 0, 1) ** 1.5 + 2 * np.cos(2 * p)) * D2R   # cup + lateral fan


def couter_radius(cx, side):
    key = "couter_R_" + side
    if key not in cx.shared:
        S, E, W = arm_joints(cx, side)
        a, b, de, lat = elbow_dirs(cx, side)
        X, Y = G.disc_grid(9)
        dirs, rho, psi, _ = G.dome_dirs(G.nrm(de + 0.45 * lat), lat, X, Y, COUTER_RHO)
        body = arm_body(cx, side)
        rb = []
        for d in dirs.reshape(-1, 3):
            r = body.ray(E + d * 0.2, -d, 0.2)
            rb.append(0.2 - r if r is not None else 0.0)
        cx.shared[key] = max(float(np.max(rb)) + MAIL_OFF + 0.011, 0.055)
    return cx.shared[key]


def couter(cx, side):
    mb = G.MB("couter_" + side)
    S, E, W = arm_joints(cx, side)
    a, b, de, lat = elbow_dirs(cx, side)
    Rc = couter_radius(cx, side)
    bn = "lowerarm_helper_" + side
    X, Y = G.disc_grid(9)
    D = G.nrm(de + 0.45 * lat)
    dirs, rho, psi, (e1, e2) = G.dome_dirs(D, lat, X, Y, COUTER_RHO)
    # a little fluting on the fan (rays from the centre), otherwise a clean sphere around the elbow joint
    flute = 0.0022 * np.cos(6 * psi) * G.smoothstep(0.35, 0.8, np.sqrt(X ** 2 + Y ** 2)) * np.clip(np.cos(psi), 0, 1)
    P = E + (Rc + flute)[..., None] * dirs
    prof = G.rim_gold(w=0.007, bead=0.0028, t=0.002, lip=0.0, band="fil_narrow")
    P, _, ec = G.etch_grid(P, None, ["j0", "j1", "i0", "i1"], 0.010)
    G.shell(mb, P, inside=E, w={bn: 1.0}, tag="cup", inner=True, thick=ARM_T, rim=lambda Pl, Nl: prof, sphere=E,
            uvfn=lambda U, V_, P_: TX.steel_uv(U + 0.2, V_ + 0.5), inner_uv=lambda U, V_, P_: TX.steel_uv(U + 2.0, V_),
            etch=ec)
    # rosette boss on the fan (seated on the cup)
    cb = G.mb_bvh(mb, lambda t: t == "cup")
    Df = G.nrm(D + 0.75 * lat)
    c = G.seat_points(cb, [E + Rc * Df], [Df], lift=0.0)[0]
    G.lathe(mb, c, Df, [(0.0115, 0.0), (0.0108, 0.0027), (0.0078, 0.0041), (0.0040, 0.0051), (0.0012, 0.0055)],
            seg=16, band="gold_plain", w={bn: 1.0}, tag="boss", planar="boss", seat=cb)
    return mb


def vambrace(cx, side):
    mb = G.MB("vambrace_" + side)
    S, E, W = arm_joints(cx, side)
    RF = limb_field(cx, E, W, side, up=(0.3, 0.0, 0.95))
    ax, o, f = limb_frame(cx, E, W, side, up=(0.3, 0.0, 0.95))
    a_, b_, de, lat = elbow_dirs(cx, side)
    thp = theta_of(o, f, de)
    Rc = couter_radius(cx, side)
    nth = 24
    th = np.linspace(-np.pi, np.pi, nth, endpoint=False)
    far = angdist(th, thp) / np.pi
    ov0, ov1 = 0.50, 0.57
    base = lambda T, TH: blur_tth(RF(T, TH), 1.0, 2.0) + MAIL_OFF + 0.0075
    # round overlap radius (upper cannon) >= the lower cannon's needs + GAP
    tt = np.linspace(ov0, ov1, 4)
    need = float(base(np.broadcast_to(tt[None, :], (nth, 4)), np.broadcast_to(th[:, None], (nth, 4))).max())
    r_up = need + GAP
    # ---- upper cannon (lowerarm)
    nt = 7
    tt0 = 0.13 + 0.10 * G.smoothstep(0.5, 1.0, far)            # inner elbow cut back
    T = tt0[:, None] + (ov1 - tt0)[:, None] * np.linspace(0, 1, nt)[None, :]
    TH = np.broadcast_to(th[:, None], T.shape)
    Rn = base(T, TH)
    k = G.smoothstep(ov0 - 0.07, ov0, T)
    R = Rn + (np.maximum(r_up, Rn) - Rn) * k
    P = RF.point(T, TH, R)
    P = min_sphere(P, E, Rc + ARM_GAP, E, ax, wmask=1 - G.smoothstep(0.42, 0.56, T))     # over the couter's lower edge
    P, _, ev = G.etch_grid(P, None, ["j1"], 0.014, closed=True)        # etched band at the wrist-side border
    def rim_up(Pl, Nl):
        pr = G.rim_gold(w=0.007, bead=0.0032, t=0.002, lip=0.006, band="fil_narrow")
        if float(np.mean(np.linalg.norm(Pl - E, axis=1))) < Rc + ARM_GAP + 0.015:
            return pr, None, None, E                       # the top edge rides the couter: rim on its sphere
        return pr
    G.shell(mb, P, closed=True, inside=axis_foot(E, W), w={"lowerarm_" + side: 1.0}, tag="upper",
            rim=rim_up, uvfn=lambda U, V_, P_: TX.steel_uv(U, V_), inner=True, thick=ARM_T, etch=ev)
    # ---- lower cannon (lowerarm_twist_01): round inside the upper at the overlap, then oval to the wrist
    T = ov0 - 0.015 + (0.92 - ov0 + 0.015) * np.linspace(0, 1, nt)[None, :] * np.ones((nth, 1))
    TH = np.broadcast_to(th[:, None], T.shape)
    Rn = base(T, TH) - 0.0015
    k = 1 - G.smoothstep(ov1, ov1 + 0.07, T)
    R = Rn + (np.full_like(Rn, r_up - GAP) - Rn) * k
    P = RF.point(T, TH, R)

    def rim_low(Pl, Nl):
        # both edges hide: the top under the upper cannon, the bottom under the gauntlet cuff. Session 4 (G2 spikes):
        # a plain square 2.2 mm edge (the hidden bead's 35 % flange was a stack of sub-mm rows, needles folded 90 deg)
        return G.prof((0.0, -ARM_T, "steel_plain", 0.0, 1.0))
    G.shell(mb, P, closed=True, inside=axis_foot(E, W), w={"lowerarm_twist_01_" + side: 1.0}, tag="lower",
            rim=rim_low, uvfn=lambda U, V_, P_: TX.steel_uv(U + 0.6, V_), inner=True, thick=ARM_T)
    cx.shared["vamb_low_R_" + side] = (RF, base)
    return mb


# ======================================================================================================= GAUNTLETS
FINGERS = ["index", "middle", "ring", "pinky"]
GLOVE_OFF = 0.0022


def proxy_mesh(cx):
    """CC0 MakeHuman low-poly proxy (male1591), fitted to the current body: (V, F) (removed from the scene again)."""
    if getattr(cx, "_proxy", None) is None:
        g0 = {g.name for g in cx.bm.vertex_groups}; m0 = {m.name for m in cx.bm.modifiers}
        px = HumanService.add_mhclo_asset(asset_file("proxymeshes", "male1591", "male1591.proxy"), cx.bm,
                                          asset_type="Proxymeshes", subdiv_levels=0, set_up_rigging=False)
        dg = bpy.context.evaluated_depsgraph_get()
        me = px.evaluated_get(dg).to_mesh()
        V = np.array([v.co[:] for v in me.vertices]); F = [tuple(p.vertices) for p in me.polygons]
        px.evaluated_get(dg).to_mesh_clear()
        bpy.data.objects.remove(px, do_unlink=True)
        for g in [g for g in cx.bm.vertex_groups if g.name not in g0]:
            cx.bm.vertex_groups.remove(g)
        for m in [m for m in cx.bm.modifiers if m.name not in m0]:
            cx.bm.modifiers.remove(m)
        cx._proxy = (V, F)
    return cx._proxy


def proxy_region(cx, keep_vert, key, subdiv=0):
    """quad region of the fitted proxy. The vertex selection is made on the authoring body and stored with the asset
    (cx.topo[key]) so a regeneration on another body keeps the SAME topology (the mhclo correspondence stays valid)."""
    V, F = proxy_mesh(cx)
    topo = getattr(cx, "topo", {})
    if key in topo:
        k = np.zeros(len(V), bool); k[np.array(topo[key], dtype=int)] = True
    else:
        k = np.array([keep_vert(p) for p in V])
    if not hasattr(cx, "topo_out"):
        cx.topo_out = {}
    cx.topo_out[key] = [int(i) for i in np.nonzero(k)[0]]
    Fk = [f for f in F if all(k[list(f)])]
    used = sorted(set(i for f in Fk for i in f))
    remap = {o: n for n, o in enumerate(used)}
    V2 = V[used]; F2 = [tuple(remap[i] for i in f) for f in Fk]
    return G.region_mesh(V2, F2, subdiv)


def add_region(mb, V, F, uvfn, w, tag):
    ids = mb.add(V, w=w, tag=tag)
    for f in F:
        mb.quad(*[ids[i] for i in f], [uvfn(V[i]) for i in f])
    return ids


def hand_frame(cx, side):
    sg = side_sign(side)
    H = cx.head("hand_" + side)
    kn = np.array([cx.head(f + "_01_" + side) for f in FINGERS])
    hdir = G.nrm(kn.mean(0) - H)
    lat = G.nrm(kn[3] - kn[0])                      # index -> pinky
    nails = cx.body.groups["fingernails"]; nails = nails[cx.body.co[nails, 0] * sg > 0]
    tips = np.array([cx.tail(f + "_03_" + side) for f in FINGERS])
    dors = G.nrm(np.cross(hdir, lat))
    if np.dot(dors, cx.body.co[nails].mean(0) - tips.mean(0)) < 0:
        dors = -dors
    return H, hdir, lat, dors


def _seg_d(p, a, b):
    ab = b - a
    t = float(np.clip(np.dot(p - a, ab) / max(np.dot(ab, ab), 1e-12), 0, 1))
    return float(np.linalg.norm(p - (a + t * ab)))


def glove_weights(cx, side, V):
    """Segment weights for the leather glove (C2): the palm / back of the hand rides `hand` (not the MakeHuman palm
    weights, which drag the palm into a cone when the fingers close round a grip), each finger / thumb region its
    phalanx, blended only across the joint to the neighbouring phalanx (inverse-distance^4 over the two nearest
    segments of adjacent bones); the wrist part rides the forearm twist bone."""
    S, E, W = arm_joints(cx, side)
    kn = np.array([cx.head(f + "_01_" + side) for f in FINGERS]).mean(0)
    hand, twist = "hand_" + side, "lowerarm_twist_01_" + side
    segs = [(hand, W, kn), (twist, E + 0.8 * (W - E), W)]
    chain_of = {}
    for f in FINGERS + ["thumb"]:
        for k in (1, 2, 3):
            bn = "%s_%02d_%s" % (f, k, side)
            if bn in cx.bone:
                segs.append((bn, cx.head(bn), cx.tail(bn)))
                chain_of[bn] = (f, k)

    def adj(a, b):
        if a in chain_of and b in chain_of:
            return chain_of[a][0] == chain_of[b][0] and abs(chain_of[a][1] - chain_of[b][1]) == 1
        pair = {a, b}
        if pair == {hand, twist}:
            return True
        if hand in pair:
            o = (pair - {hand}).pop()
            return o in chain_of and chain_of[o][1] == 1
        return False
    out = []
    for p in V:
        d = sorted(((_seg_d(p, a, b), n) for n, a, b in segs))
        (d0, n0), (d1, n1) = d[0], d[1]
        if adj(n0, n1) and d1 < 1.6 * d0 + 0.002:
            w0, w1 = 1 / max(d0, 1e-4) ** 4, 1 / max(d1, 1e-4) ** 4
            out.append({n0: w0 / (w0 + w1), n1: w1 / (w0 + w1)})
        else:
            out.append({n0: 1.0})
    return out


def rim_roll(bead=0.0014, band="steel_bead"):
    """rolled edge for small plates (finger scales): a low steel roll; shell(inner=True) turns it under to the inside"""
    return G.prof((0.9 * bead, 0.5 * bead, band, 0.0, 0.6), (1.4 * bead, -0.2 * bead, band, 0.6, 1.0))


def drop_degenerate(V, F, min_faces=5):
    """Session 3 (G2 / G5: a 3-face glove island of the CC0 proxy at the pinky tip collapsed to ZERO size under the
    shrink-wrap and was counted as a floating finger plate): drop proxy-region islands of fewer than `min_faces` faces
    and compact V. Topological criterion only, so every body gets the same topology (regen_on_body)."""
    V = np.asarray(V, float)
    F = [tuple(f) for f in F]
    par = list(range(len(V)))

    def find(x):
        while par[x] != x:
            par[x] = par[par[x]]; x = par[x]
        return x
    for f in F:
        for k in range(1, len(f)):
            par[find(f[k])] = find(f[0])
    isl = {}
    for f in F:
        isl.setdefault(find(f[0]), []).append(f)
    F2 = [f for fs in isl.values() if len(fs) >= min_faces for f in fs]
    used = sorted({v for f in F2 for v in f})
    rm = {v: i for i, v in enumerate(used)}
    if len(used) != len(V) or len(F2) != len(F):
        log_("drop_degenerate: %d -> %d faces, %d -> %d vertices" % (len(F), len(F2), len(V), len(used)))
    return V[used], [tuple(rm[v] for v in f) for f in F2]


def closed_sheet(mb, V, F, uvf, wts, tag, thick, rim_band_uv=None, axis_of=None):
    """a CLOSED leather / plate sheet from a quad region (V, F): outer surface + an inner copy `thick` under it along
    the area-weighted vertex normals + a turned edge along every open border (iteration 2b: no open sheets)"""
    from mathutils.bvhtree import BVHTree
    V = np.asarray(V, float)
    vn = np.zeros_like(V)
    for f in F:
        n_ = np.cross(V[f[1]] - V[f[0]], V[f[2]] - V[f[0]]) + np.cross(V[f[3]] - V[f[2]], V[f[0]] - V[f[2]])
        for i in f:
            vn[i] += n_
    vn = G.nrm(vn)
    # session 3 (integrity G1 'vanish' on the gauntlets: the glove's inner copy crossed its outer surface in the proxy
    # hand's tight creases, thumb web and palm, so from outside the FIRST surface was a back face): the offset uses
    # neighbour-smoothed normals and a thickness limited by the local concave curvature (<= 0.45 x the radius, >= 0.25
    # mm), then every inner vertex is pulled >= 0.4 x thick under the nearest point of the outer surface
    nb = [set() for _ in range(len(V))]
    for f in F:
        for k in range(len(f)):
            nb[f[k]].update((f[k - 1], f[(k + 1) % len(f)]))
    vs = vn.copy()
    for _ in range(2):
        vs = G.nrm(np.array([vs[i] + sum(vs[j] for j in nb[i]) / max(len(nb[i]), 1) for i in range(len(V))]))
    th = np.full(len(V), float(thick))
    for i in range(len(V)):
        for j in nb[i]:
            e = V[j] - V[i]; l2 = float(e @ e)
            h_ = float(e @ vn[i])
            if h_ > 1e-6 and l2 > 1e-12:                   # neighbour above the tangent plane: concave that way
                th[i] = min(th[i], max(0.45 * l2 / (2 * h_), 0.00025))
    VI = V - th[:, None] * vs
    tris = []
    for f in F:
        tris += [(f[0], f[1], f[2]), (f[0], f[2], f[3])] if len(f) == 4 else [tuple(f)]
    ob_ = BVHTree.FromPolygons([Vector(p) for p in V], tris, epsilon=0.0)
    for _ in range(3):
        for i in range(len(V)):
            loc, n_, fi, d_ = ob_.find_nearest(Vector(VI[i]), 0.01)
            if loc is None:
                continue
            n_ = np.array(n_[:])
            if float(n_ @ vn[i]) < 0:
                n_ = -n_
            s_ = float((VI[i] - np.array(loc[:])) @ n_)
            if s_ > -0.4 * th[i]:
                VI[i] = np.array(loc[:]) - n_ * 0.4 * th[i]
    # exposed inner faces (the glove's thumb against the index / palm: the outer surface folds into itself there, so
    # the inner copy of one side lies outside the other side's outer surface): a ray from the inner face's centre
    # outward must meet the OUTER surface first; the vertices of faces that do not are pulled in toward the
    # axis of their bone (callers pass `axis_of(i) -> point`), up to 8 x 0.6 mm
    tri_in = [(f, t) for f in F for t in (((f[0], f[1], f[2]), (f[0], f[2], f[3])) if len(f) == 4 else (tuple(f),))]
    nfix = 0
    for it in range(8):
        allb = BVHTree.FromPolygons([Vector(p) for p in np.concatenate([V, VI])],
                                    tris + [tuple(len(V) + x for x in t) for t in tris], epsilon=0.0)
        bad = set()
        for f, t in tri_in:
            p = VI[list(t)]; c = p.mean(0)
            n_in = np.cross(p[1] - p[0], p[2] - p[0]); ln = np.linalg.norm(n_in)
            if ln < 1e-12:
                continue
            nout = n_in / ln
            if float(nout @ vn[t[0]]) < 0:
                nout = -nout
            h = allb.ray_cast(Vector(c + nout * 2e-5), Vector(nout), 0.05)
            if h[0] is None or h[2] >= len(tris):
                bad.update(t)
        if not bad:
            break
        for i in bad:
            if axis_of is not None:
                q = np.asarray(axis_of(i), float); dv = q - VI[i]; l_ = np.linalg.norm(dv)
                VI[i] = VI[i] + dv / max(l_, 1e-9) * min(0.0006, 0.5 * l_)
            else:
                VI[i] = VI[i] - vs[i] * 0.0006
        nfix += len(bad)
    if nfix:
        log_("closed_sheet %s: %d inner-vertex pulls (exposed inner faces), %d left" % (tag, nfix, len(bad)))
    ids = mb.add(V, w=wts, tag=tag)
    idi = mb.add(VI, w=wts, tag=tag)
    for f in F:
        mb.quad(*[ids[i] for i in f], [uvf(V[i]) for i in f])
        r = (f[0], f[3], f[2], f[1]) if len(f) == 4 else tuple(reversed(f))   # same diagonal as the outer quad (session 3)
        mb.quad(*[idi[i] for i in r], [uvf(V[i]) for i in r])
    for lp in G.boundary_loops([tuple(f) for f in F]):
        L_ = len(lp)
        for k in range(L_):
            a, b = lp[k], lp[(k + 1) % L_]
            q = (ids[b], ids[a], idi[a], idi[b])
            mb.quad(*q, [uvf(V[x]) for x in (b, a, a, b)])
    return ids, idi, vn


GLOVE_T = 0.0010         # glove leather thickness (closed sheet)
CUFF_SLIP = 0.0012       # the cuff's inside over the vambrace's lower cannon (both ride lowerarm_twist_01)
CUFF_WRIST = 0.0030      # the cuff's lower edge over the glove at the wrist (inside clearance; the wrist bends in it;
                         # session 4: was 4.5 mm, a tangential slot the G3 wrist views looked through)
SCALE_T = 0.0013         # finger scale thickness
SCALE_LIFT = -0.0001     # session 4 (G2: ring / index / thumb tip scales 0.57-0.88 mm off the glove's facets, 'floating';
                         # was +0.3 mm over the outermost glove vertex) finger scale underside over the glove (G5: the plate RESTS on the finger, <= 2 mm; the
                         # low-poly glove's facets sit up to ~0.8 mm inside its vertices)
HAND_T = 0.0015          # back-of-hand plate thickness


def gauntlet(cx, side):
    """Iteration 2b (user item 29): the leather glove is a closed 1 mm sheet (segment weights, it bends); ONE back-of-hand
    plate (hand bone) carries three stepped lame ridges and the knuckle ridge as one closed shell (the four separate
    metacarpal lames cut each other); every finger scale is a closed 1.3 mm half-shell resting 0.4 mm on the glove, one
    per phalanx, ending short of the joints (the glove shows at the knuckles when the fist closes; no scale overlaps a
    neighbour, so none can cut another in any curl); the thumb has two scales. Cuff = hourglass bell (closed)."""
    sg = side_sign(side)
    mb = G.MB("gauntlet_" + side)
    body = cx.body
    H, hdir, lat, dors = hand_frame(cx, side)
    S, E, W = arm_joints(cx, side)
    fa = G.nrm(W - E); LF = float(np.linalg.norm(W - E))
    handw = body.weight_sum(["hand_" + side, "thumb_0", "index_0", "middle_0", "ring_0", "pinky_0"])
    handw = handw * (body.co[:, 0] * sg > 0) * (np.abs(body.co[:, 0]) > 0.3 * cx.tscale())
    # ---- leather glove: proxy hand region (+ wrist), shrink-wrapped over the skin, segment weights, CLOSED
    def keep(p):
        i = body.kd.find(p)[1]
        s_ = np.dot(p - E, fa) / LF
        return (handw[i] > 0.3 or s_ > 0.9) and s_ > 0.86 and p[0] * sg > 0.25 * cx.tscale()
    V, F = proxy_region(cx, keep, "glove_" + side)
    V = G.shrinkwrap_offset(body.subset((handw > 0.2) | (body.co[:, 0] * sg > 0.45 * cx.tscale())), V, F, GLOVE_OFF, iters=2)
    V, F = drop_degenerate(V, F)
    wts = glove_weights(cx, side, V)
    lcent = V.mean(0)
    ext = 2.08 * max(np.abs((V - lcent) @ lat).max(), np.abs((V - lcent) @ hdir).max())

    def uv_glove(p):
        d = p - lcent
        return TX.square_uv("leather", float(d @ lat), float(d @ hdir), size=ext)
    gdom_ = [max(w_, key=w_.get) for w_ in wts]

    def glove_axis(i):
        b = gdom_[i]
        a, t = np.array(cx.head(b)), np.array(cx.tail(b))
        d = t - a; u = float(np.clip((V[i] - a) @ d / max(d @ d, 1e-12), 0, 1))
        return a + d * u
    g_ids, g_idi, _gvn = closed_sheet(mb, V, F, uv_glove, wts, "glove", GLOVE_T,
                                      axis_of=glove_axis if os.environ.get("GLOVE_AXIS") == "1" else None)
    glove_bvh = G.mb_bvh(mb, lambda t: t == "glove")
    cx.shared["glove_" + side] = (V.copy(), F)
    # ---- cuff: hourglass bell over the wrist (lowerarm_twist_01), outside the lower vambrace cannon
    RF = limb_field(cx, E, W, side, up=(0.3, 0.0, 0.95))
    base = lambda T, TH: blur_tth(RF(T, TH), 1.0, 2.0) + MAIL_OFF + 0.0075
    # iteration 2b (user item 32 'floating rings', G3 wrist / cuff-vambrace): a slimmer hourglass: the mouth flares
    # 11 mm over the vambrace's lower cannon (was 21 mm: a loose ring round the forearm), the waist narrows past the
    # cannon's end and the lower edge closes to CUFF_WRIST over the glove at the wrist (was ~2 cm of open annulus
    # looking into the arm)
    # session 3 (user item 32 'floating rings', G3 wrist seam): the lower edge ends 1.5 % of the forearm BEFORE the wrist
    # joint with a small rolled bead (it ran 2 % past the joint with a 7 mm gold band + bead, which stood off the bent
    # hand as a loose hoop and opened a gap into the cuff in the block / attack)
    t0c, t1c = 0.785, 0.985
    nth = 24
    th = np.linspace(-np.pi, np.pi, nth, endpoint=False)
    T = t0c + (t1c - t0c) * np.array([0.0, 0.18, 0.40, 0.62, 0.82, 1.0])[None, :] * np.ones((nth, 1))
    TH = np.broadcast_to(th[:, None], T.shape)
    v = (T - t0c) / (t1c - t0c)
    over = base(T, TH) - 0.0015 + GAP + 0.0015                       # over the lower cannon (it ends at t 0.92)
    # session 4 (user item 32 'loose rings round the forearm', G3 cuff-vambrace: the mouth stood 12-15 mm off the
    # cannon, an annular window looking past the forearm to the shield / cape): the cuff and the vambrace's lower cannon
    # ride the same bone, so the mouth hugs the cannon's real surface (ray-cast) by CUFF_SLIP, flaring only 3 mm
    vm_ = cx.shared.get("vambrace_%s_mb" % side)
    cann = G.mb_bvh(vm_, lambda t: t == "lower") if vm_ is not None else None
    flare = 0.011
    if cann is not None:
        flare = 0.003
        for i in range(T.shape[0]):
            for j in range(T.shape[1]):
                A_ = E + T[i, j] * (W - E)
                d_ = G.nrm(RF.point(np.array(T[i, j]), np.array(TH[i, j]), 1.0) - A_)
                h_ = cann.ray_cast(Vector(A_ + d_ * 0.15), Vector(-d_), 0.15)
                if h_[0] is not None:
                    over[i, j] = 0.15 - h_[3] + CUFF_SLIP + ARM_T
    tight = blur_tth(RF(T, TH), 1.0, 2.0) + GLOVE_OFF + GLOVE_T + CUFF_WRIST + ARM_T
    wv = G.smoothstep(0.905, 0.985, T)
    R = np.maximum(over * (1 - wv) + tight * wv, tight) + flare * (1 - v) ** 1.8
    P = RF.point(T, TH, R)
    cuff_w = {"lowerarm_twist_01_" + side: 1.0}
    P, _, ecf = G.etch_grid(P, None, ["j0"], 0.012, closed=True)
    def rim_cuff(Pl, Nl):
        tl = float(np.mean((Pl - E) @ fa) / LF)
        if tl > 0.5 * (t0c + t1c):                          # the wrist edge: a small bead, tight on the glove
            return G.rim_bead(bead=0.0026, t=0.002, lip=0.003, band="gold_bead")
        return G.rim_gold(w=0.007, bead=0.0032, t=0.002, lip=0.006, band="fil_narrow")
    G.shell(mb, P, closed=True, inside=axis_foot(E, W), w=cuff_w, tag="cuff",
            rim=rim_cuff,
            uvfn=lambda U, V_, P_: TX.steel_uv(U + 0.3, V_ + 0.2), inner=True, thick=ARM_T, etch=ecf)
    cfb = G.mb_bvh(mb, lambda t: t == "cuff")
    cuff_seals(cx, mb, side, RF, E, W, cfb, glove_bvh, cuff_w, t0c, t1c, V, wts)
    for a in (0.0, math.pi):
        tt = t0c + 0.30 * (t1c - t0c)
        c = RF.point(np.array(tt), np.array(a), float(base(np.array([[tt]]), np.array([[a]]))[0, 0]) + GAP + 0.011 * 0.7 ** 1.8)
        # session 4: seated on the (slimmer) cuff along the ray from the axis
        A_ = E + tt * (W - E); d_ = G.nrm(c - A_)
        h_ = cfb.ray_cast(Vector(A_ + d_ * 0.15), Vector(-d_), 0.15)
        if h_[0] is not None:
            c = np.array(h_[0][:])
        G.rivet(mb, c, G.nrm(c - (E + tt * (W - E))), r=0.0038, w=cuff_w, seat=cfb)
    # ---- back of the hand: ONE plate on `hand`: three stepped lame ridges + the knuckle ridge (closed shell)
    kn = np.array([cx.head(f + "_01_" + side) for f in FINGERS])
    kcen = kn.mean(0)
    hw = {"hand_" + side: 1.0}
    Lh = float(np.linalg.norm(kcen - H))

    def glove_pt(p, default=None):
        """the glove's outer surface under p (straight down the back of the hand)"""
        loc, nor, fi, d = glove_bvh.ray_cast(Vector(p + dors * 0.05), Vector(-dors), 0.10)
        return np.array(loc[:]) if loc is not None else default
    # rows along the hand (v: 0 wrist .. 1 knuckles) with the height over the glove (m): a terrace of 4 lames, each
    # stepping down 1.2 mm under a rolled bead, then the knuckle ridge (+2.8 mm) with a knuckle for each finger
    V_ROWS = []
    lvl = 0.0062
    for k, (a0, a1) in enumerate(((0.02, 0.26), (0.26, 0.48), (0.48, 0.70), (0.70, 0.82))):
        V_ROWS += [(a0 + 0.004, lvl), (a1 - 0.026, lvl), (a1 - 0.010, lvl + 0.0009)]
        lvl -= 0.0011
    # session 4 (G4 glove|hand plate at the knuckles in the fist, 46 points, 6 mm): the knuckle ridge stands 1.5 mm higher
    V_ROWS += [(0.835, lvl + 0.0010), (0.875, lvl + 0.0045), (0.915, lvl + 0.0045), (0.945, lvl + 0.0020)]
    us = np.linspace(-1.0, 1.0, 7)
    P = np.zeros((len(us), len(V_ROWS), 3))
    for j, (vv, off) in enumerate(V_ROWS):
        hwid = 0.029 + 0.017 * min(vv, 1.0)
        for i, uu in enumerate(us):
            b_ = H + (kcen - H) * vv + lat * uu * hwid
            if vv > 0.86:                                   # knuckle bumps over each MCP joint
                off2 = off + 0.0022 * max(0.0, math.cos((uu * 0.5 + 0.5) * 3 * math.pi)) ** 2 * (1 - abs(uu) * 0.3)
            else:
                off2 = off
            q = glove_pt(b_)
            if q is None:
                q = b_ + dors * 0.012
            P[i, j] = q + dors * (off2 + HAND_T)
    P = G.laplace_grid(P, iters=1, lam=0.25, fix_border=True)

    def rim_hand(Pl, Nl):
        s_ = (Pl - H) @ hdir / Lh
        hid = s_ < 0.10                                     # the wrist end hides under the cuff
        return G.rim_bead(bead=0.0022, t=0.0015, lip=0.003, band="steel_bead"), smooth_amp(np.where(hid, 0.0, 1.0))
    G.shell(mb, P, inside=H - dors * 0.03, w=hw, tag="handplate", rim=rim_hand, inner=True, thick=HAND_T,
            uvfn=lambda U, V_, P_: TX.steel_uv(U + 0.2, V_))
    # ---- finger scales: one closed half-shell per phalanx, resting on the glove, gaps at the joints
    fskin = {}

    def finger_skin(f):
        if f not in fskin:
            w = body.weight_sum([f + "_0"]) * (body.co[:, 0] * sg > 0)
            fskin[f] = body.subset(w > 0.3)
        return fskin[f]

    def section(f, h, a, d, l2, s):
        c = h + a * s
        dirs8 = [math.cos(x) * d + math.sin(x) * l2 for x in np.linspace(0, 2 * np.pi, 8, endpoint=False)]
        rr = []
        fsub = finger_skin(f)
        for dv in dirs8:
            loc, nor, fi, dd = fsub.bvh.ray_cast(Vector(c + dv * 0.03), Vector(-dv), 0.03)
            rr.append(0.03 - dd if loc is not None else np.nan)
        rr = np.array(rr)
        if np.isnan(rr).all():
            rr[:] = 0.008
        rr = np.where(np.isnan(rr), np.nanmean(rr), rr)
        off = sum(r_ * dv for r_, dv in zip(rr, dirs8)) / 8 * 2.0
        cen = c + off
        rad = float(np.mean([np.linalg.norm(c + r_ * dv - cen) for r_, dv in zip(rr, dirs8)]))
        return cen, rad

    segs = [(f, k) for f in FINGERS for k in (1, 2, 3)] + [("thumb", 2), ("thumb", 3)]
    GV = np.array(V)                                            # glove OUTER vertices (its outermost points)
    gdom = np.array([max(w_, key=w_.get) for w_ in wts])
    from mathutils.kdtree import KDTree
    built = []                                                   # (finger, points) of the scales made so far
    for f, k in segs:
        bn = "%s_%02d_%s" % (f, k, side)
        h, t = cx.head(bn), cx.tail(bn)
        a = G.nrm(t - h); ln = np.linalg.norm(t - h)
        # the phalanx's own dorsal side: normal to its flexion hinge (bone local X) and its axis, toward the back of the
        # hand (the global dorsal direction leans the scales of the curled rest fingers into their neighbours)
        hx = np.array((cx.rig.data.bones[bn].matrix_local.to_3x3() @ Vector((1, 0, 0)))[:])
        d = G.nrm(np.cross(a, hx))
        if np.dot(d, dors) < 0:
            d = -d
        if f == "thumb":
            d = G.nrm(d + 0.3 * G.nrm(dors - np.dot(dors, a) * a))
        d = G.nrm(d - np.dot(d, a) * a)
        l2 = np.cross(a, d)
        # other fingers' glove vertices + scales already built: the arc narrows on a side that comes within 1.2 mm
        oth = [GV[i] for i in range(len(GV)) if not gdom[i].startswith(f + "_") and not gdom[i].startswith("hand")
               and not gdom[i].startswith("lowerarm")]
        for ff, pts in built:
            if ff != f:
                oth += list(pts)
        kd = KDTree(max(1, len(oth)))
        for q_i, q in enumerate(oth):
            kd.insert(Vector(q), q_i)
        kd.balance()
        # iteration 2b (G5 plate seating in a fist): the scales stay off the joints, where the glove bends away
        s0, s1 = {1: (0.24, 0.80), 2: (0.18, 0.80), 3: (0.14, 0.92)}[k]
        if f == "thumb":
            s0, s1 = {2: (0.16, 0.80), 3: (0.18, 0.84)}[k]     # session 3 (G5): no tip rounding: it overhung the glove
        nu, nv = 5, 3
        ss = np.linspace(s0, s1, nv)
        A_lo, A_hi = -50.0, 50.0        # session 4 (G4 scale|scale of neighbouring fingers in the fist): was +-58 deg
        P = np.zeros((nu, nv, 3))
        # the glove vertices around this phalanx, in its section frame (axial s, arc, radius from the finger axis)
        cen0, _ = section(f, h, a, d, l2, 0.5 * ln)
        rel = GV - h
        s_v = rel @ a / ln
        radial = rel - np.outer(rel @ a, a)
        # radial from the (offset) section centre line: centre offset as at mid phalanx
        c_off = (cen0 - h) - np.dot(cen0 - h, a) * a
        rad_v = radial - c_off
        rr_v = np.linalg.norm(rad_v, axis=1)
        arc_v = np.arctan2(rad_v @ l2, rad_v @ d)
        near = (s_v > s0 - 0.25) & (s_v < s1 + 0.25) & (rr_v < 0.020)
        for attempt in range(5):
          arcs = np.linspace(A_lo, A_hi, nu) * D2R
          ds = 0.6 * (s1 - s0) / (nv - 1) + 0.04; da = 0.75 * (arcs[1] - arcs[0])
          for j, sv in enumerate(ss):
            cen, rad = section(f, h, a, d, l2, ln * min(max(sv, 0.03), 0.97))
            cen = cen + a * ln * (sv - min(max(sv, 0.03), 0.97))
            for i, ag in enumerate(arcs):
                dv = math.cos(ag) * d + math.sin(ag) * l2
                # rest on the glove: over the OUTERMOST glove vertex in this sector (the low-poly glove's facets lie
                # inside its vertices, so projecting onto the facets left glove vertices poking through the scale)
                m = near & (np.abs(s_v - sv) < ds) & (np.abs(np.arctan2(np.sin(arc_v - ag), np.cos(arc_v - ag))) < da)
                if m.any():
                    gr = float((rad_v[m] @ dv).max()) + float(np.dot(cen0 - cen, dv))
                else:
                    loc, nor, fi, dist = glove_bvh.ray_cast(Vector(cen + dv * (rad + 0.01)), Vector(-dv), 0.02)
                    gr = float(np.dot(np.array(loc[:]) - cen, dv)) if loc is not None else rad + GLOVE_OFF
                r_ = max(gr, rad + GLOVE_OFF) + SCALE_LIFT + SCALE_T - (0.0002 if f == "thumb" else 0.0)  # session 4: the
                # female thumb_02 scale stood 0.63 mm off the glove's facets (its sector's outermost vertex is sparse)
                if k == 3 and j == nv - 1 and f != "thumb":
                    r_ = max(r_ * 0.96, max(gr, rad + GLOVE_OFF) + SCALE_T)   # rounds over the fingertip, on the glove
                P[i, j] = cen + dv * r_
          # side clearance to the neighbouring fingers (their glove and scales): narrow the arc where it is too close
          lo_bad = min(kd.find(Vector(P[0, j]))[2] for j in range(nv)) < 0.0028 if oth else False
          hi_bad = min(kd.find(Vector(P[-1, j]))[2] for j in range(nv)) < 0.0028 if oth else False
          if not (lo_bad or hi_bad):
              break
          A_lo += 9.0 if lo_bad else 0.0
          A_hi -= 9.0 if hi_bad else 0.0
        built.append((f, P.reshape(-1, 3).copy()))
        w = {bn: 1.0}
        G.shell(mb, P, inside=h + 0.5 * (t - h), w=w, tag="finger", inner=True, thick=SCALE_T,
                rim=lambda Pl, Nl: rim_roll(), uvfn=lambda U, V_, P_: TX.steel_uv(U + 0.8, V_))
    glove_coskin(mb, V, g_ids, g_idi, _gvn, wts, side)
    return mb


SEAL_T = 0.0010           # cuff floor thickness
SEAL_CLEAR = 0.0010       # the floor's edge off the cuff's inner wall (its hidden rim flange reaches ~0.5 mm further)
SEAL_GLOVE = 0.0012       # the floor's inner edge over the glove (both ride lowerarm_twist_01 there)
SEAL_T_AT = 0.945         # its place along the forearm: past the lower cannon's end (0.92 + rim), before the wrist rim


def cuff_seals(cx, mb, side, RF, E, W, cfb, glove_bvh, cuff_w, t0c, t1c, GV, gwts):
    """Session 4 (integrity G3 wrist seam: a wrist bend opens a crescent between the glove and the cuff's wrist rim;
    the forearm skin inside the cuff is deleted by the knight, so the sight line ran 3-5 cm up the empty cuff to the
    inside of the vambrace's lower cannon): a thin closed conical FLOOR inside the cuff, a centimetre inside the wrist
    rim (proximal of the glove, distal of the mail sleeve's and the lower cannon's ends), from the cuff's inner wall to
    1.5 mm over the forearm skin (the knight's skin-level gap filler lies there: a disc to the axis cut it). The cuff,
    the lower cannon and this floor all ride lowerarm_twist_01; the inner edge sits 3 mm nearer the wrist (the face
    seen from the wrist looks out and down)."""
    LF = float(np.linalg.norm(W - E)); fa = G.nrm(W - E)
    nth = 36
    th = np.linspace(-np.pi, np.pi, nth, endpoint=False)
    tq = SEAL_T_AT
    if cfb is None:
        return
    A = E + tq * (W - E)
    r_out = np.full(nth, np.nan); dirs = []
    for i, a in enumerate(th):
        d = G.nrm(RF.point(np.array(tq), np.array(a), 1.0) - A)
        dirs.append(d)
        h = cfb.ray_cast(Vector(A), Vector(d), 0.2)                # the cuff's inner wall (first cuff surface from the axis)
        if h[0] is not None:
            r_out[i] = h[3] - SEAL_CLEAR
    good = np.isfinite(r_out)
    if good.sum() < nth * 0.8:
        log_("gauntlet_%s: cuff floor skipped (%d / %d wall hits)" % (side, int(good.sum()), nth))
        return
    if glove_bvh is not None:
        hg = glove_bvh.find_nearest(Vector(A), 0.2)
        if hg[0] is not None:
            gt = float((np.array(hg[0][:]) - E) @ fa) / LF
            if gt < tq + 0.004:
                log_("gauntlet_%s: WARNING the glove reaches t %.3f, inside the cuff floor (t %.3f)" % (side, gt, tq))
    idx = np.arange(nth)
    r_out = np.interp(idx, idx[good], r_out[good], period=nth)
    dirs = np.array(dirs)
    # inner edge 1.5 mm over the forearm skin (the knight's skin-level gap filler lies there; a disc to the axis cut it)
    r_in = np.array([float(RF(np.array([[tq]]), np.array([[a]]))[0, 0]) for a in th]) + 0.0015
    r_in = np.minimum(r_in, r_out - 0.004)
    fr = np.array([0.0, 0.3, 0.65, 1.0])
    RR = r_in[:, None] + (r_out - r_in)[:, None] * fr[None, :]
    fr = np.broadcast_to(fr[None, :], RR.shape)
    P = A[None, None, :] + dirs[:, None, :] * RR[..., None]
    P = P + fa[None, None, :] * (0.003 * (1 - fr))[..., None]              # centre 3 mm toward the wrist
    G.shell(mb, P, closed=True, inside=lambda X: X - 0.05 * fa, w=cuff_w, tag="cuffseal",
            rim=lambda Pl, Nl: G.prof((0.0, -SEAL_T, "steel_plain", 0.0, 1.0)),   # a square 1 mm edge (no sub-mm rows)
            uvfn=lambda U, V_, P_: TX.steel_uv(U + 0.45, V_ + 0.3), inner=True, thick=SEAL_T)
    log_("gauntlet_%s: cuff floor at t %.3f, radius %.1f..%.1f mm" % (side, tq, 1000 * r_out.min(), 1000 * r_out.max()))


def glove_coskin(mb, V, g_ids, g_idi, gvn, wts, side, reach=0.012, tilt=25.0):
    """Session 3 (G4 gauntlet own components: glove vs hand plate / finger scales, 5-40 visible points in every clip):
    the leather UNDER a rigid plate rides that plate's bone. The glove's segment weights blend across each joint, so in
    the fist its knuckles and the ends of each phalanx bulged up through the knuckle ridge and the scales. Per glove
    vertex: 5 rays outward (its normal + 4 tilted 25 deg); the fraction that meet the SAME plate (hand plate or a scale)
    within 12 mm, smoothstep 0.5..1, blends it into 100 % that plate's bone (outer and inner layer alike): the glove
    bends only where no plate covers it (the knuckle gaps, which the scales leave open by design)."""
    from mathutils.bvhtree import BVHTree
    tags = mb.tag
    comp_of = {}; Vp = []; Fp = []; lab = []; bones = []
    for fi, f in enumerate(mb.F):
        if tags[f[0]] not in ("finger", "handplate"):
            continue
        w0 = mb.W[f[0]]; bn = max(w0, key=w0.get)
        if bn not in comp_of:
            comp_of[bn] = len(bones); bones.append(bn)
        k = comp_of[bn]
        o = len(Vp); Vp += [mb.V[v] for v in f]
        Fp += [(o, o + 1, o + 2), (o, o + 2, o + 3)]; lab += [k, k]
    if not bones:
        return 0
    bvh = BVHTree.FromPolygons([Vector(p) for p in Vp], Fp, epsilon=0.0)
    ta = math.radians(tilt); n_ = 0
    for k in range(len(V)):
        n0 = np.asarray(gvn[k], float)
        t1 = G.nrm(np.cross(n0, (0.0, 0.0, 1.0)) if abs(n0[2]) < 0.9 else np.cross(n0, (1.0, 0.0, 0.0)))
        t2 = np.cross(n0, t1)
        dirs = [n0] + [G.nrm(n0 * math.cos(ta) + (t1 * math.cos(a_) + t2 * math.sin(a_)) * math.sin(ta))
                       for a_ in np.linspace(0, 2 * np.pi, 4, endpoint=False)]
        hit = []
        for d in dirs:
            h = bvh.ray_cast(Vector(V[k] + d * 2e-4), Vector(d), reach)
            hit.append(lab[h[2]] if h[0] is not None else -1)
        if hit[0] < 0:
            continue
        sw = float(G.smoothstep(0.5, 1.0, sum(1 for x in hit if x == hit[0]) / len(hit)))
        if sw <= 0:
            continue
        bn = bones[hit[0]]
        w = {b: x * (1 - sw) for b, x in wts[k].items()}
        w[bn] = w.get(bn, 0.0) + sw
        w = dict(sorted(((b, x) for b, x in w.items() if x > 1e-4), key=lambda kv: -kv[1])[:4])
        tot = sum(w.values()); w = {b: x / tot for b, x in w.items()}
        mb.W[g_ids[k]] = dict(w); mb.W[g_idi[k]] = dict(w); n_ += 1
    log_("gauntlet %s: %d glove vertices carried by the plate / scale above them" % (side, n_))
    return n_


# ======================================================================================================= MAIL
SLEEVE_T = 0.93            # the mail sleeve ends here along the forearm (under the gauntlet cuff, 0.785 .. 1.045)


def mail(cx):
    mb = G.MB("mail")
    body = cx.body
    armw = {s_: body.weight_sum(["upperarm_" + s_, "upperarm_twist_01_" + s_, "lowerarm_" + s_, "lowerarm_twist_01_" + s_,
                                 "hand_" + s_]) for s_ in "lr"}
    handw = body.weight_sum(["hand_", "thumb_", "index_", "middle_", "ring_", "pinky_"])
    headw = body.weight_sum(["head", "jaw", "lip_", "mouth_", "tongue", "cheek", "nose", "brow", "eye"])
    fl = {s_: arm_joints(cx, s_)[1:] for s_ in "lr"}
    zlo, zhi = float(cx.tz(0.955)), cx.chin_z + 0.045

    def keep(p):
        i = body.kd.find(p)[1]
        arm = max(armw["l"][i], armw["r"][i]) > 0.3
        # mail collar up the neck, under the helmet rim; the hem at the hips (iteration 2b: the hem height no longer
        # cuts the SLEEVES: in the A pose the forearm hangs below it)
        if headw[i] > 0.3 or p[2] > zhi or (p[2] < zlo and not arm):
            return False
        for s_ in "lr":
            E, W = fl[s_]
            t = np.dot(p - E, W - E) / np.dot(W - E, W - E)
            if armw[s_][i] > 0.3:
                # iteration 2b: the low-poly proxy has no ring between just below the elbow and the wrist, so cutting
                # at t 0.955 ended the sleeve at the ELBOW (the vambrace / cuff gaps looked into an empty forearm):
                # the wrist ring is kept and pulled back to SLEEVE_T below (under the gauntlet cuff)
                return t < 1.25
        if handw[i] > 0.25:
            return False
        return True
    V, F = proxy_region(cx, keep, "mail", subdiv=1)
    for s_ in "lr":
        E, W = fl[s_]
        ax = W - E; L2 = float(np.dot(ax, ax))
        for k, p in enumerate(V):
            if p[0] * side_sign(s_) < 0.2 * cx.tscale():
                continue
            t = np.dot(p - E, ax) / L2
            if t > SLEEVE_T - 0.10:
                # compress t in [SLEEVE_T - 0.10, 1.25] into [SLEEVE_T - 0.10, SLEEVE_T] (keeps the ring order)
                tn = SLEEVE_T - 0.10 + 0.10 * min(1.0, (t - (SLEEVE_T - 0.10)) / (1.25 - (SLEEVE_T - 0.10))) ** 0.8
                V[k] = p + ax * (tn - t)
    V = G.shrinkwrap_offset(body, V, F, MAIL_OFF, iters=3)
    wts = G.body_weights_at(body, V)
    # the sleeve end follows the forearm / twist bones only (no hand weights: the hand bends inside the cuff)
    for n_, p in enumerate(V):
        w = wts[n_]
        if any(k.startswith(("hand_", "thumb_", "index_", "middle_", "ring_", "pinky_")) for k in w):
            w2 = {k: x for k, x in w.items() if not k.startswith(("hand_", "thumb_", "index_", "middle_", "ring_", "pinky_"))}
            if not w2:
                s_ = "l" if p[0] > 0 else "r"
                w2 = {"lowerarm_twist_01_" + s_: 1.0}
            tot = sum(w2.values()); wts[n_] = {k: x / tot for k, x in w2.items()}

    def region(p):
        i = body.kd.find(p)[1]
        for s_ in "lr":
            if armw[s_][i] > 0.5:
                return s_
        return "t"
    reg = [region(p) for p in V]
    ids = mb.add(V, w=wts, tag="mail")
    f0 = len(mb.F)
    for f in F:
        mb.quad(*[ids[i] for i in f], [(0.0, 0.0)] * 4)
    mail_uvs(cx, mb, list(range(f0, len(mb.F))), [max(set(rs), key=rs.count) for rs in ([reg[i] for i in f] for f in F)])
    return mb


def mail_uvs(cx, mb, fsel, regions, tag="mail"):
    """Iteration 2b (user item 31): the mail's UVs are near-isometric panels (1 UV = the 0.168 m mail tile, ring rows
    round the body / round the arm): torso front / back, collar front / back, each sleeve in a front and a back half, cut
    along the panel borders (hidden under the plates or along the sleeve's top / underside) and unwrapped with Minimum
    Stretch (G.unwrap_faces). The old cylindrical map (u = azimuth x 0.16 m) smeared the rings 3-30x at the neck, the
    shoulders and the arm-pits."""
    rs_ = cx.shared.get("cuirass_fields", {}).get("RS")
    axis_y = (lambda z: float(rs_.axis_fn(float(z))[1])) if rs_ is not None else (lambda z: -0.02)
    zneck = float(cx.head("neck_01")[2]) - 0.012
    joints = {s_: arm_joints(cx, s_) for s_ in "lr"}
    labels = []
    for fi, r in zip(fsel, regions):
        c = np.mean([mb.V[i] for i in mb.F[fi]], axis=0)
        if r == "t":
            fb = "f" if c[1] < axis_y(c[2]) else "b"
            labels.append(("collar_" if c[2] > zneck else "torso_") + fb)
        else:
            S_, E_, W_ = joints[r]
            a, b = (S_, E_) if _seg_d(c, S_, E_) <= _seg_d(c, E_, W_) else (E_, W_)
            ax = G.nrm(b - a); v = c - a; rad = v - np.dot(v, ax) * ax
            fw = np.array([0, -1.0, 0]); fw = G.nrm(fw - np.dot(fw, ax) * ax)
            labels.append("sleeve_%s_%s" % (r, "f" if np.dot(rad, fw) > 0 else "b"))

    def around(p, n):
        # rows run horizontally round the trunk / round the arm (the nearest arm segment's axis)
        best = None
        for s_ in "lr":
            S_, E_, W_ = joints[s_]
            for a, b in ((S_, E_), (E_, W_)):
                d = _seg_d(p, a, b)
                if best is None or d < best[0]:
                    best = (d, G.nrm(b - a))
        armish = best[0] < 0.075 and abs(p[0]) > 0.16 * cx.tscale()
        ax = best[1] if armish else np.array([0, 0, 1.0])
        return np.cross(ax, n)
    st = G.unwrap_faces(mb, fsel, labels, around, TX.MAIL_TILE_M)
    if st:
        s1 = np.array([x[0] for x in st]); s2 = np.array([x[1] for x in st]); A = np.array([x[2] for x in st])
        sc = np.sqrt(s1 * s2); med = float(np.median(sc))
        bad = (s1 / med > 1.15) | (s2 / med < 0.85)
        log_("%s uv: %d islands, scale median %.3f, out of +-15 %% on %.1f %% of the area (p99 stretch %.2f)" % (
            tag, len(set(labels)), med, 100 * float(A[bad].sum() / A.sum()), float(np.percentile(s1 / med, 99))))
        cx.shared[tag + "_uv_stats"] = dict(med=med, out=float(A[bad].sum() / A.sum()))
    return labels


# ======================================================================================================= AVENTAIL
AV_OFF = 0.0035          # coif over the head skin (a padded arming cap's worth under the helm's lining)
FACE_OPEN = (40.0, 4.0, 19.0)   # coif face opening: half width (deg azimuth), centre and half height (deg elevation)


def aventail(cx):
    """Iteration 2b (user items 23 / 30 / 31; integrity G3 skull-visor, visor-bevor, helmet-gorget, G7, the build's
    gap-filler aventail): a riveted-mail COIF with its aventail under the helm (helm look only: build_knight tags the
    'aventail' slot helm-only, so the bare head keeps its hair). It covers the head and runs down the neck UNDER the mail
    shirt's collar (AV_OFF over the skin, the collar is MAIL_OFF: 5 mm between them, the same skin weights on the
    neck), its hem 2.2 cm below the collar's top edge, with an oval face opening round the eyes (the dark face of the
    helmet's 'headform' shows there). Every sight line in through the eye slot, the breaths, under the visor's chin or
    between the helm and the gorget ends on mail or the face in shadow, never on the empty helmet or the background.
    Built from the low-poly CC0 proxy's head / neck region (subdivided once): above the chin the vertices are placed
    radially from the head centre on the OUTER skin (+AV_OFF; a nearest-point wrap fell into the mouth cavity), below it
    shrink-wrapped; >= 3 mm inside the helm's lining; body weights with the face joints merged into `head`."""
    mb = G.MB("aventail")
    body = cx.body
    H = cx.shared["helm"]; hmb = cx.shared["helmet_mb"]
    headw = body.weight_sum(["head", "jaw", "lip_", "mouth_", "tongue", "cheek", "nose", "brow", "eye"])
    cen = np.array([H.C[0], H.C[1] + 0.005, cx.eye[2] - 0.015])
    nf = neck_frame(cx)
    # the mail shirt's collar: its top edge per azimuth round the neck axis
    mm = cx.shared.get("mail_mb")
    cth, cz = np.linspace(-np.pi, np.pi, 37), np.full(37, cx.chin_z + 0.045)
    if mm is not None:
        bl = [lp for lp in G.boundary_loops([tuple(f) for f in mm.F])]
        top = max(bl, key=lambda lp: float(np.mean([mm.V[i][2] for i in lp])))
        TP = np.array([mm.V[i] for i in top])
        v = TP - nf["A0"]; T = np.arctan2(v @ nf["left"], v @ nf["front"])
        o = np.argsort(T); T, Z = T[o], TP[o, 2]
        cz = np.interp(cth, T, Z, period=2 * np.pi)
    def zhem(p):
        v = p - nf["A0"]; T = math.atan2(float(v @ nf["left"]), float(v @ nf["front"]))
        return float(np.interp(T, cth, cz, period=2 * np.pi)) - 0.022

    def ang(p):
        v = p - cen
        return math.degrees(math.atan2(v[0], -v[1])), math.degrees(math.atan2(v[2], math.hypot(v[0], v[1])))

    hs = body.subset(headw > 0.2)

    def keep(p):
        i = body.kd.find(p)[1]
        if p[2] < zhem(p) - 0.012:
            return False
        if p[2] > cx.chin_z - 0.01:
            # only the OUTER surface (the proxy's mouth / nostril cavity vertices lie behind the lips)
            d = G.nrm(p - cen)
            loc, nor, fi, dist = hs.bvh.ray_cast(Vector(cen + d * 0.3), Vector(-d), 0.3)
            if loc is not None and np.linalg.norm(p - cen) < 0.3 - dist - 0.004:
                return False
        a, e = ang(p)
        if (a / FACE_OPEN[0]) ** 2 + ((e - FACE_OPEN[1]) / FACE_OPEN[2]) ** 2 < 1.0:
            return False                                   # the face opening
        return headw[i] > 0.05 or p[2] > cx.chin_z - 0.08
    V, F = proxy_region(cx, keep, "aventail", subdiv=1)
    V = G.shrinkwrap_offset(body, V, F, AV_OFF, iters=3)
    hb = G.mb_bvh(hmb, lambda t: t in ("skull", "visor", "liner"))      # session 3: and inside the dark liner
    moved_in = 0
    for k, p in enumerate(V):
        d = G.nrm(p - cen)
        # above the chin: on the OUTER head skin along the ray from the head centre
        wr = float(G.smoothstep(cx.chin_z - 0.015, cx.chin_z + 0.005, p[2]))
        if wr > 0:
            loc, nor, fi, dist = hs.bvh.ray_cast(Vector(cen + d * 0.3), Vector(-d), 0.3)
            if loc is not None:
                q = np.array(loc[:]) + d * AV_OFF
                V[k] = p + (q - p) * wr
        d = G.nrm(V[k] - cen)
        l3 = hb.ray_cast(Vector(cen), Vector(d), 0.4)
        if l3[0] is not None:
            lim = l3[3] - 0.0030
            r = float(np.linalg.norm(V[k] - cen))
            if r > lim:
                V[k] = cen + d * lim; moved_in += 1
    V = G.laplace_mesh(V, F, iters=1, lam=0.3) if hasattr(G, "laplace_mesh") else V
    wts = G.body_weights_at(body, V)
    FACEB = ("jaw", "eye", "eyelid", "brow", "cheek", "nose", "mouth", "lip", "tongue")
    for k, w in enumerate(wts):
        hw = sum(x for b, x in w.items() if b == "head" or b.startswith(FACEB))
        w2 = {b: x for b, x in w.items() if not (b == "head" or b.startswith(FACEB))}
        if hw > 0:
            w2["head"] = hw
        tot = sum(w2.values()) or 1.0
        wts[k] = {b: x / tot for b, x in w2.items()}
    ids = mb.add(V, w=wts, tag="aventail")
    f0 = len(mb.F)
    for f in F:
        mb.quad(*[ids[i] for i in f], [(0.0, 0.0)] * 4)
    labels = []
    for f in F:
        c = V[list(f)].mean(0)
        a, e = ang(c)
        # session 3 (G6 aventail: 44 % of the visible mail out of +-15 %): the front chart ringed the face opening (an
        # annulus does not unwrap without stretch): it is split left / right, each a C round half the opening
        lab_ = "top" if e > 50 else (("fl" if a > 0 else "fr") if abs(a) < 45 else ("b" if abs(a) > 135 else ("l" if a > 0 else "r")))
        labels.append(lab_ + ("_n" if e < -28 else ""))          # the neck tube is charted apart from the head
    st = G.unwrap_faces(mb, list(range(f0, len(mb.F))), labels, lambda p, n: np.cross((0, 0, 1.0), n), TX.MAIL_TILE_M)
    if st:
        s1 = np.array([x[0] for x in st]); s2 = np.array([x[1] for x in st]); A = np.array([x[2] for x in st])
        med = float(np.median(np.sqrt(s1 * s2)))
        bad = (s1 / med > 1.15) | (s2 / med < 0.85)
        log_("aventail uv: out of +-15 %% on %.1f %% of the area; %d verts pulled inside the helm; hem %.3f..%.3f" % (
            100 * float(A[bad].sum() / A.sum()), moved_in, cz.min() - 0.022, cz.max() - 0.022))
    return mb


# ======================================================================================================= LAYERS
FOLLOW_TAGS = ("decal", "rivet")          # small parts that ride on their plate: they copy the plate's push


def _tris_of(mb, pred):
    T = []
    for f in mb.F:
        if all(pred(mb.tag[i]) for i in f):
            T += [(f[0], f[1], f[2]), (f[0], f[2], f[3])]
    return T


def _bvh_of(mbs, parts):
    from mathutils.bvhtree import BVHTree
    V, F = [], []
    for name, pred in parts:
        mb = mbs.get(name)
        if mb is None:
            continue
        T = _tris_of(mb, pred)
        if not T:
            continue
        off = len(V)
        V += [Vector(v) for v in mb.V]
        F += [(a + off, b + off, c + off) for a, b, c in T]
    return BVHTree.FromPolygons(V, F) if F else None


def _hits(bvh, o, d, maxd):
    out, t = [], 0.0
    while t < maxd and len(out) < 16:
        loc, nor, idx, dist = bvh.ray_cast(Vector(o + d * t), Vector(d), maxd - t)
        if loc is None:
            break
        t += dist
        out.append(t)
        t += 2e-5
    return out


def skin_dir(body, p):
    """(outward direction, SIGNED distance) of point p w.r.t. the skin: the skin-to-point direction outside, the skin
    normal when the point is on / behind the skin (a rim lip dipping into the body)"""
    loc, nor, fi, dist = body.bvh.find_nearest(Vector(p))
    v = p - np.array(loc[:]); nr = np.array(nor[:])
    if dist < 0.002:
        return nr, float(np.dot(v, nr))
    c = float(np.dot(v, nr)) / dist
    if c < -0.3:
        return nr, -dist
    return v / dist, dist


def _adjacency(mb):
    nb = [set() for _ in range(len(mb.V))]
    for f in mb.F:
        for k in range(4):
            a, b = f[k], f[(k + 1) % 4]
            nb[a].add(b); nb[b].add(a)
    return nb


def settle(cx, mbs, moving, other, clear, mode, min_skin=0.0, reach=0.015, report=None, label=""):
    """Layer clearance (user items 14 / 19, judge M2): the vertices of `moving` ([(piece, tag predicate)]) are pushed
    away from the surfaces of `other` along the skin-to-vertex direction until every `other` surface is at least
    `clear` below them (mode 'outer': the moving part is the outer layer) or above them (mode 'inner', clamped so the
    moving part stays `min_skin` over the skin). Penetrations up to `reach` deep are resolved. The push field is dilated
    and relaxed over each piece's mesh (no kinks); decals / rivets copy the push of their nearest plate vertex."""
    from mathutils.kdtree import KDTree
    bvh = _bvh_of(mbs, other)
    if bvh is None:
        return 0
    total, worst = 0, 0.0
    for name, pred in moving:
        mb = mbs.get(name)
        if mb is None:
            continue
        sel = [i for i in range(len(mb.V)) if pred(mb.tag[i]) and mb.tag[i] not in FOLLOW_TAGS]
        fol = [i for i in range(len(mb.V)) if pred(mb.tag[i]) and mb.tag[i] in FOLLOW_TAGS]
        if not sel:
            continue
        push = np.zeros(len(mb.V)); dirs = np.zeros((len(mb.V), 3)); room = np.full(len(mb.V), 1.0)
        for i in sel:
            p = np.asarray(mb.V[i], float)
            d = piece_dir(cx, name, p)
            dist = skin_signed(cx, p)[0] if mode == "inner" else 1.0
            dirs[i] = d
            if mode == "outer":
                hs = _hits(bvh, p - d * reach, d, reach + clear)
                ys = [h - reach for h in hs if -clear < h - reach < reach]
                if ys:
                    push[i] = max(ys) + clear
            else:
                hs = _hits(bvh, p + d * reach, -d, reach + clear)
                ys = [reach - h for h in hs if -reach < reach - h < clear]
                if ys:
                    push[i] = clear - min(ys)
                room[i] = max(0.0, dist - min_skin)
        # pass B: the other layer's vertices against the moving surface (thin rims / lips poking through the moving
        # piece's faces between its vertices): the deficit goes to the vertices of the face they hit
        from mathutils.bvhtree import BVHTree
        Tm = [t for t in _tris_of(mb, pred) if all(mb.tag[i] not in FOLLOW_TAGS for i in t)]
        if Tm:
            bm_ = BVHTree.FromPolygons([Vector(v) for v in mb.V], Tm)
            for oname, opred in other:
                omb = mbs.get(oname)
                if omb is None:
                    continue
                for j in range(len(omb.V)):
                    if not opred(omb.tag[j]):
                        continue
                    q = np.asarray(omb.V[j], float)
                    d = piece_dir(cx, name, q)
                    if mode == "outer":          # moving surface must be >= clear ABOVE q
                        o_ = q - d * reach
                        loc2, nor2, ti, h = bm_.ray_cast(Vector(o_), Vector(d), reach + clear)
                        if loc2 is None:
                            continue
                        y = h - reach
                        if -reach < y < clear:
                            for i in Tm[ti]:
                                if push[i] < clear - y:
                                    push[i] = clear - y
                                    if not dirs[i].any():
                                        dirs[i] = d
                    else:                        # moving surface must be >= clear BELOW q
                        o_ = q + d * reach
                        loc2, nor2, ti, h = bm_.ray_cast(Vector(o_), Vector(-d), reach + clear)
                        if loc2 is None:
                            continue
                        y = reach - h            # height of the moving surface relative to q
                        if -clear < y < reach:
                            for i in Tm[ti]:
                                if push[i] < clear + y:
                                    push[i] = clear + y
                                    if not dirs[i].any():
                                        dirs[i] = d
        need = push.copy()
        if need.max() <= 0:
            continue
        nb = _adjacency(mb)
        on = np.zeros(len(mb.V), bool); on[sel] = True
        for _ in range(3):                                   # dilate with falloff, then relax (never below need)
            nxt = push.copy()
            for i in sel:
                m = max((push[j] for j in nb[i] if on[j]), default=0.0)
                nxt[i] = max(push[i], 0.6 * m)
            push = nxt
        for _ in range(3):
            nxt = push.copy()
            for i in sel:
                js = [j for j in nb[i] if on[j]]
                if js:
                    nxt[i] = max(need[i], 0.5 * push[i] + 0.5 * float(np.mean(push[js])))
            push = nxt
        if mode == "inner":
            push = np.minimum(push, room)
        sgn = 1.0 if mode == "outer" else -1.0
        for i in sel:
            if push[i] > 0:
                mb.V[i] = np.asarray(mb.V[i], float) + sgn * push[i] * dirs[i]
        if fol:
            kd = KDTree(len(sel))
            for k, i in enumerate(sel):
                kd.insert(Vector(mb.V[i]), k)
            kd.balance()
            for i in fol:
                k = kd.find(Vector(mb.V[i]))[1]
                j = sel[k]
                if push[j] > 0:
                    mb.V[i] = np.asarray(mb.V[i], float) + sgn * push[j] * dirs[j]
        n = int((push > 1e-5).sum())
        total += n; worst = max(worst, float(push.max()))
        if report is not None:
            un = np.maximum(need - push, 0)
            iu = int(np.argmax(un))
            report.append(dict(rule=label, piece=name, moved=n, max_mm=round(float(push.max()) * 1000, 2),
                               unresolved_mm=round(float(un.max()) * 1000, 2),
                               unresolved_n=int((un > 0.0005).sum()),
                               unresolved_at=[round(float(x), 3) for x in mb.V[iu]] if un.max() > 0 else None))
    return total


def _armw(cx):
    if not hasattr(cx, "_armw_all"):
        cx._armw_all = cx.body.weight_sum(["upperarm", "lowerarm", "hand", "thumb", "index", "middle", "ring", "pinky"])
    return cx._armw_all


def skin_signed(cx, p):
    """signed distance of p to the full body skin (+ outside), the nearest skin vertex's arm weight"""
    body = cx.body
    loc, nor, fi, dist = body.bvh.find_nearest(Vector(p))
    v = p - np.array(loc[:]); nr = np.array(nor[:])
    c = float(np.dot(v, nr)) / max(dist, 1e-9)
    sd = dist if (dist < 1e-6 or c > -0.3) else -dist
    tri = body.tris[fi]
    aw = float(_armw(cx)[tri].mean())
    return sd, aw, nr


def piece_dir(cx, name, p):
    """outward direction of a piece at p: radial from its own reference (torso axis, neck axis, head centre, shoulder
    joint, limb axis); the skin normal for the mail, the glove and the hand plates"""
    p = np.asarray(p, float)
    if name == "cuirass":
        return G.nrm(np.array([p[0], p[1] + 0.02, 0.0]))
    if name == "gorget":
        loc, nor, fi, dist = cx.body.bvh.find_nearest(Vector(p))
        return np.array(nor[:])
    if name == "helmet" and "helm" in cx.shared:
        H = cx.shared["helm"]
        return G.nrm(p - np.array([H.C[0], H.C[1], H.z_eq - 0.03]))
    if name.startswith(("pauldron", "rerebrace", "couter", "vambrace")):
        S, E, W = arm_joints(cx, name[-1])
        if name.startswith("pauldron"):
            return G.nrm(p - S)
        a, b = (S, E) if name.startswith("rerebrace") else ((E, W) if name.startswith("vambrace") else (None, None))
        if a is None:
            return G.nrm(p - E)
        ax = G.nrm(b - a); v = p - a
        return G.nrm(v - np.dot(v, ax) * ax)
    loc, nor, fi, dist = cx.body.bvh.find_nearest(Vector(p))
    return G.nrm(p - np.array(loc[:])) if dist > 0.002 else np.array(nor[:])


def own_skin(name, aw):
    """is a skin point with arm weight aw part of the body region a piece covers? (plates never react to the OTHER body
    part: a breastplate edge at the arm-pit is not pushed by the arm)"""
    if name in ("cuirass", "gorget", "helmet"):
        return aw < 0.5
    if name.startswith(("rerebrace", "couter", "vambrace", "gauntlet")):
        return aw > 0.3
    return True


def floor_on_skin(cx, mbs, names, min_h, report=None, cap=0.012):
    """Every plate vertex at least min_h over its own skin region (room for the mail + its clearance under every rim
    lip); pushes along the piece's outward direction, dilated over the mesh"""
    for name in names:
        mb = mbs.get(name)
        if mb is None:
            continue
        push = np.zeros(len(mb.V)); dirs = np.zeros((len(mb.V), 3))
        sel = [i for i in range(len(mb.V)) if mb.tag[i] not in FOLLOW_TAGS and mb.tag[i] != "glove"]
        for i in sel:
            p = np.asarray(mb.V[i], float)
            sd, aw, nr = skin_signed(cx, p)
            trunk = name in ("cuirass", "gorget")
            dirs[i] = nr if trunk else piece_dir(cx, name, p)
            if sd < min_h and (trunk or own_skin(name, aw)):
                c = cap * (5 if trunk else 1)
                push[i] = min((min_h - sd) / max(0.35, float(np.dot(dirs[i], nr))), c)
        if push.max() <= 0:
            continue
        nb = _adjacency(mb)
        on = np.zeros(len(mb.V), bool); on[sel] = True
        for _ in range(3):
            push = np.array([max(push[i], 0.6 * max((push[j] for j in nb[i] if on[j]), default=0.0)) if on[i] else 0.0
                             for i in range(len(mb.V))])
        for i in sel:
            if push[i] > 0:
                mb.V[i] = np.asarray(mb.V[i], float) + push[i] * dirs[i]
        fol = [i for i in range(len(mb.V)) if mb.tag[i] in FOLLOW_TAGS]
        if fol:
            from mathutils.kdtree import KDTree
            kd = KDTree(len(sel))
            for k, i in enumerate(sel):
                kd.insert(Vector(mb.V[i]), k)
            kd.balance()
            for i in fol:
                j = sel[kd.find(Vector(mb.V[i]))[1]]
                if push[j] > 0:
                    mb.V[i] = np.asarray(mb.V[i], float) + push[j] * dirs[j]
        if report is not None:
            report.append(dict(rule="floor %.1f mm" % (min_h * 1000), piece=name, moved=int((push > 0).sum()),
                               max_mm=round(float(push.max()) * 1000, 2), unresolved_mm=0.0, unresolved_n=0,
                               unresolved_at=None))


ALL = lambda t: True
TAG = lambda *ts: (lambda t: t in ts)
NOT = lambda *ts: (lambda t: t not in ts)
PLATES = ["helmet", "gorget", "cuirass", "pauldron_l", "pauldron_r", "rerebrace_l", "rerebrace_r", "couter_l",
          "couter_r", "vambrace_l", "vambrace_r", "gauntlet_l", "gauntlet_r"]


def layer_rules():
    """(label, moving, other, clear, mode, min_skin) in order: plates over plates first, then the soft layers under them"""
    R = [("back<breast", [("cuirass", TAG("back"))], [("cuirass", TAG("breast", "gusset"))], 0.003, "inner", 0.010),
         ("plackart>breast", [("cuirass", TAG("plackart"))], [("cuirass", TAG("breast", "gusset", "stoprib"))], 0.003, "outer", 0),
         ("culet>back", [("cuirass", TAG("culet"))], [("cuirass", TAG("back", "breast"))], 0.003, "outer", 0),
         ("gusset>breast", [("cuirass", TAG("gusset"))], [("cuirass", TAG("breast", "back"))], 0.0025, "outer", 0),
         ("gorget>cuirass", [("gorget", ALL)], [("cuirass", ALL)], 0.003, "outer", 0),
         ("lameB<lameA", [("gorget", TAG("lameB"))], [("gorget", TAG("lameA"))], 0.003, "inner", 0.010)]
    for s_ in "lr":
        R += [("pauldron>cuirass,gorget", [("pauldron_" + s_, ALL)], [("cuirass", ALL), ("gorget", ALL)], 0.0035, "outer", 0),
              ("rerebrace<pauldron", [("rerebrace_" + s_, ALL)], [("pauldron_" + s_, ALL)], 0.0035, "inner", 0.0075),
              ("rerebrace>couter", [("rerebrace_" + s_, ALL)], [("couter_" + s_, ALL)], 0.003, "outer", 0),
              ("vambrace>couter", [("vambrace_" + s_, ALL)], [("couter_" + s_, ALL)], 0.003, "outer", 0),
              ("vamb.lower<upper", [("vambrace_" + s_, TAG("lower"))], [("vambrace_" + s_, TAG("upper"))], 0.003, "inner", MAIL_OFF + 0.002),
              ("cuff>vambrace", [("gauntlet_" + s_, TAG("cuff"))], [("vambrace_" + s_, ALL)], 0.003, "outer", 0)]
    R += [("mail<plates", [("mail", ALL)], [(n, NOT("glove")) for n in PLATES], 0.004, "inner", 0.002)]
    for s_ in "lr":
        R += [("glove<plates", [("gauntlet_" + s_, TAG("glove"))], [("gauntlet_" + s_, NOT("glove"))], 0.0015, "inner", 0.0006),
              ("plates>glove", [("gauntlet_" + s_, NOT("glove"))], [("gauntlet_" + s_, TAG("glove"))], 0.0012, "outer", 0)]
    return R


LIFT_GAP = 0.0040         # the mail rises to this far under the plate above it (integrity G3: no deep slot under a plate)
LIFT_MAX = 0.012          # at most this much
EDGE_REACH = float(os.environ.get("RTS_EDGE_REACH", "0.030"))       # session 4: mail within this of a torso plate's edge (outside its footprint) rises too
LIFT_PLATES = ("cuirass", "gorget", "rerebrace_l", "rerebrace_r", "couter_l", "couter_r", "vambrace_l", "vambrace_r")


def mail_lift(cx, mbs, gap=LIFT_GAP, reach=0.035):
    """Iteration 2b (integrity G3 seams): the mail under the torso and arm plates rises to `gap` below the plate's inside
    (it lay 8.5 mm over the skin, up to 2 cm under the plates): a sight line between two plates meets the mail (an
    under-layer) right under their edges instead of running down a deep slot to a plate's inside or out the other side
    of the arm. Lift along the mail's normal, smoothed over the mesh, never above what the ray found."""
    mm = mbs["mail"]
    from mathutils.bvhtree import BVHTree
    V, F = [], []
    for n in LIFT_PLATES:
        mb = mbs.get(n)
        if mb is None:
            continue
        off = len(V)
        V += [Vector(v) for v in mb.V]
        F += [(f[0] + off, f[1] + off, f[2] + off) for f in mb.F] + [(f[0] + off, f[2] + off, f[3] + off) for f in mb.F]
    if not F:
        return 0
    bvh = BVHTree.FromPolygons(V, F)
    # session 4 (integrity G3 pauldron- / cape- / mail- / helmet-gorget: from above, the sight line slipped past the
    # gorget's edge into the open slot between the breast- / backplate's top edge and the mail, 8-15 mm deep there,
    # and met the plate's inside): the mail just OUTSIDE a torso plate's edge (within EDGE_REACH) also rises toward the
    # edge's height, so the slot closes at the edge, not 2-3 cm inside it
    TV, TF = [], []
    for n in ("cuirass", "gorget"):
        mb = mbs.get(n)
        if mb is None:
            continue
        off = len(TV)
        TV += [Vector(v) for v in mb.V]
        TF += [(f[0] + off, f[1] + off, f[2] + off) for f in mb.F] + [(f[0] + off, f[2] + off, f[3] + off) for f in mb.F]
    tbvh = BVHTree.FromPolygons(TV, TF) if TF else None
    P = np.array(mm.V, float)
    nv = len(P)
    vn = np.zeros_like(P)
    for f in mm.F:
        n_ = np.cross(P[f[1]] - P[f[0]], P[f[2]] - P[f[0]]) + np.cross(P[f[3]] - P[f[2]], P[f[0]] - P[f[2]])
        for i in f:
            vn[i] += n_
    vn = G.nrm(vn)
    # outward: away from the skin
    for i in range(nv):
        loc, nor, fi, dist = cx.body.bvh.find_nearest(Vector(P[i]))
        if loc is not None and np.dot(P[i] - np.array(loc[:]), vn[i]) < 0:
            vn[i] = -vn[i]
    lift = np.zeros(nv)
    nedge = 0
    for i in range(nv):
        loc, nor, fi, dist = bvh.ray_cast(Vector(P[i] + vn[i] * 0.0005), Vector(vn[i]), reach)
        if loc is not None and dist > gap:
            lift[i] = min(dist - gap, LIFT_MAX)
        elif loc is None and tbvh is not None:
            h = tbvh.find_nearest(Vector(P[i]), EDGE_REACH)
            if h[0] is not None:
                q = np.array(h[0][:]) - P[i]
                up = float(q @ vn[i]); lat = float(np.linalg.norm(q - up * vn[i]))
                if up > gap:
                    lift[i] = min(up - gap, LIFT_MAX) * (1 - float(G.smoothstep(0.3 * EDGE_REACH, EDGE_REACH, lat)))
                    nedge += lift[i] > 1e-4
    nb = _adjacency(mm)
    raw = lift.copy()
    for _ in range(3):                      # smooth; where a neighbour has no plate above, fade (no kink at the edge)
        lift = np.array([0.5 * lift[i] + 0.5 * float(np.mean([lift[j] for j in nb[i]])) if nb[i] else lift[i]
                         for i in range(nv)])
        lift = np.minimum(lift, raw + 0.0)
    for i in range(nv):
        if lift[i] > 0:
            mm.V[i] = P[i] + vn[i] * lift[i]
    log_("mail lift: %d verts raised toward the plates (max %.1f mm, mean %.1f mm; %d just outside a torso plate's edge)" % (
        int((lift > 1e-4).sum()), lift.max() * 1000, lift[lift > 1e-4].mean() * 1000 if (lift > 1e-4).any() else 0, nedge))
    return int((lift > 1e-4).sum())


def settle_layers(cx, mbs):
    """Iteration 2b: the plates are CLOSED thick shells shaped with their clearances by construction (stacked radii,
    sphere families, seated attachments); a per-vertex push would tear a shell apart (outer and inner surface moved by
    different amounts) and was what made the shards (user item 32). Only the soft under-layer (the mail sheet) is still
    pulled under the plates here. RTS_SETTLE=all restores the old plate rules (for comparison only)."""
    rep = []
    t0 = time.time()
    old = os.environ.get("RTS_SETTLE") == "all"
    if old:
        floor_on_skin(cx, mbs, [n for n in PLATES if not n.startswith(("helmet", "gauntlet"))], 0.0075, report=rep)
    if os.environ.get("RTS_MAIL_LIFT", "1") != "0" and "mail" in mbs:
        mail_lift(cx, mbs)
    for label, mov, oth, clear, mode, ms in layer_rules():
        if not old and not label.startswith("mail<"):
            continue
        if not any(n in mbs for n, _ in mov) or not any(n in mbs for n, _ in oth):
            continue
        settle(cx, mbs, mov, oth, clear, mode, min_skin=ms, report=rep, label=label,
               reach=0.03 if label.startswith(("gorget", "pauldron", "helmet")) else 0.015)
    for r in rep:
        if r["moved"]:
            log_("layers %-24s %-12s moved %5d  max %5.2f mm  unresolved %.2f mm (%d verts, at %s)" % (
                r["rule"], r["piece"], r["moved"], r["max_mm"], r["unresolved_mm"], r["unresolved_n"], r["unresolved_at"]))
    for n in PLATES + ["mail"]:
        mb = mbs.get(n)
        if mb is None:
            continue
        sdv = np.array([skin_signed(cx, np.asarray(v, float))[0] for v, t in zip(mb.V, mb.tag) if t != "glove"])
        k = int(np.argmin(sdv))
        log_("layers skin clearance %-12s min %6.1f mm  p01 %6.1f mm  (%d verts < 0)" % (
            n, 1000 * sdv.min(), 1000 * np.percentile(sdv, 1), int((sdv < 0).sum())))
        if os.environ.get("RTS_DEBUG_SKIN") == n:
            VV = [(v, t) for v, t in zip(mb.V, mb.tag) if t != "glove"]
            for kk in np.argsort(sdv)[:12]:
                log_("   inside skin %6.1f mm at %s (%s)" % (1000 * sdv[kk], np.round(VV[kk][0], 3), VV[kk][1]))
    if os.environ.get("RTS_MAIL_COSKIN", "1") != "0" and "mail" in mbs:
        mail_coskin(cx, mbs)
    log_("layers settled in %.1fs" % (time.time() - t0))
    cx.shared["layer_report"] = rep
    return rep


# session 3: the gorget only (the cuirass part carried half the shirt with spine_05 / spine_03 and the body-skinned gap
# filler under it then pushed through the mail: mail|underlayer 93 -> 147 visible points, cuirass|mail unchanged)
COSKIN_PLATES = {"gorget": ("lameA", "lameB")}


def mail_coskin(cx, mbs, reach=0.035, tilt=35.0):
    """Session 3 (integrity G4 gorget|mail, cuirass|mail; user items 19 / 25): the mail shirt DEEP under a rigid plate
    rides that plate's bone. With body weights the collar under the gorget follows the clavicles / neck skin while lame A
    rides spine_05 and lame B neck_02, and pushed through them in the attack / idle (110 visible points); likewise the
    mail under the breast- / backplate in a torso twist. Per mail vertex: 7 rays outward (its normal + 6 tilted 35 deg);
    the fraction that meet the SAME plate component within 3.5 cm is how deep under it the vertex lies. That fraction
    (smoothstep 0.55..1) blends its body weights into 100 % the plate's bone, so the mail that shows at a plate's edge
    (arm-holes, below the gorget) keeps its body weights and only the hidden part is carried."""
    from mathutils.bvhtree import BVHTree
    mm = mbs["mail"]
    Vp, Fp, lab = [], [], []
    bones = []
    for pn, tags in COSKIN_PLATES.items():
        pm = mbs.get(pn)
        if pm is None:
            continue
        for tg in tags:
            fs = [f for f in pm.F if pm.tag[f[0]] == tg]
            if not fs:
                continue
            wb = {}
            for f in fs:
                for v in f:
                    for b, x in pm.W[v].items():
                        wb[b] = wb.get(b, 0.0) + x
            bone = max(wb, key=wb.get)
            k = len(bones); bones.append((pn, tg, bone))
            off = len(Vp)
            vs = sorted({v for f in fs for v in f}); rm = {v: i for i, v in enumerate(vs)}
            Vp += [pm.V[v] for v in vs]
            for f in fs:
                Fp.append((off + rm[f[0]], off + rm[f[1]], off + rm[f[2]])); lab.append(k)
                Fp.append((off + rm[f[0]], off + rm[f[2]], off + rm[f[3]])); lab.append(k)
    if not bones:
        return 0
    bvh = BVHTree.FromPolygons([Vector(p) for p in Vp], Fp, epsilon=0.0)
    V = np.asarray(mm.V, float)
    vn = np.zeros_like(V)
    for f in mm.F:
        n_ = np.cross(V[f[2]] - V[f[0]], V[f[3]] - V[f[1]])
        for i in f:
            vn[i] += n_
    vn = G.nrm(vn)
    for i in range(len(V)):                          # outward: away from the skin
        sd, _aw, nn = skin_signed(cx, V[i])
        if float(np.dot(vn[i], nn)) < 0:
            vn[i] = -vn[i]
    ta = math.radians(tilt)
    moved = 0; per = {}
    for i in range(len(V)):
        n0 = vn[i]
        t1 = G.nrm(np.cross(n0, (0.0, 0.0, 1.0)) if abs(n0[2]) < 0.9 else np.cross(n0, (1.0, 0.0, 0.0)))
        t2 = np.cross(n0, t1)
        dirs = [n0] + [G.nrm(n0 * math.cos(ta) + (t1 * math.cos(a_) + t2 * math.sin(a_)) * math.sin(ta))
                       for a_ in np.linspace(0, 2 * np.pi, 6, endpoint=False)]
        hit = []
        for d in dirs:
            h = bvh.ray_cast(Vector(V[i] + d * 2e-4), Vector(d), reach)
            hit.append(lab[h[2]] if h[0] is not None else -1)
        if hit[0] < 0:
            continue
        frac = sum(1 for x in hit if x == hit[0]) / len(hit)
        s = float(G.smoothstep(0.55, 1.0, frac))
        if s <= 0:
            continue
        bone = bones[hit[0]][2]
        w = {b: x * (1 - s) for b, x in mm.W[i].items()}
        w[bone] = w.get(bone, 0.0) + s
        w = dict(sorted(((b, x) for b, x in w.items() if x > 1e-4), key=lambda kv: -kv[1])[:4])   # <= 4 influences
        tot = sum(w.values())
        mm.W[i] = {b: x / tot for b, x in w.items()}
        moved += 1
        key = "%s:%s" % bones[hit[0]][:2]
        per[key] = per.get(key, 0) + 1
    log_("mail coskin: %d vertices carried by the plate above them %s" % (moved, per))
    av = mbs.get("aventail"); hm = mbs.get("helmet")
    if av is not None and hm is not None:
        # session 3 (G6 aventail: the throat mail below the visor stretched 60-75 % out of +-15 % in the attack / block
        # nods, blended head / neck; G4 gorget|aventail): below the helmet's lower rim (5 mm margin) the coif copies the
        # weights of the nearest mail-shirt vertex (the collar, carried by the gorget lames), so the head / neck blend
        # happens inside the helm, out of sight, and the visible throat moves with the collar and lame B
        from mathutils.kdtree import KDTree
        kd = KDTree(len(V))
        for i, p in enumerate(V):
            kd.insert(Vector(p), i)
        kd.balance()
        HV = np.array([p for p, t in zip(hm.V, hm.tag) if t in ("skull", "visor")])
        c0 = HV[:, :2].mean(0)
        haz = np.arctan2(HV[:, 0] - c0[0], -(HV[:, 1] - c0[1]))
        bins = np.linspace(-np.pi, np.pi, 37)
        zr = np.array([HV[(haz >= a) & (haz < b), 2].min() if ((haz >= a) & (haz < b)).any() else np.nan
                       for a, b in zip(bins[:-1], bins[1:])])
        zr = np.where(np.isfinite(zr), zr, np.nanmin(zr))
        n_av = 0
        for i, p in enumerate(av.V):
            a = math.atan2(p[0] - c0[0], -(p[1] - c0[1]))
            z_rim = float(np.interp(a, 0.5 * (bins[:-1] + bins[1:]), zr, period=2 * np.pi))
            s = float(G.smoothstep(z_rim + 0.012, z_rim - 0.008, p[2]))
            if s <= 0:
                continue
            co, j, d = kd.find(Vector(p))
            w = {b: x * (1 - s) for b, x in av.W[i].items()}
            for b, x in mm.W[j].items():
                w[b] = w.get(b, 0.0) + s * x
            w = dict(sorted(((b, x) for b, x in w.items() if x > 1e-4), key=lambda kv: -kv[1])[:4])
            tot = sum(w.values()); av.W[i] = {b: x / tot for b, x in w.items()}
            n_av += 1
        log_("aventail coskin: %d coif vertices below the helm rim follow the mail collar" % n_av)
    return moved


def _uv_region(uv):
    """texture region of a face from its corner UVs: 'steel', 'sq:<name>', 'band:<name>' or 'mixed'"""
    u = float(np.mean(uv[:, 0]) % 1.0); v = float(np.mean(uv[:, 1]))
    if uv[:, 1].min() >= TX.STEEL[1] - 1e-4:
        return "steel"
    if uv[:, 1].max() <= 0.625 + 1e-4 and uv[:, 1].min() >= 0.375 - 1e-4:
        for n, (u0, v0, u1, v1) in TX.SQUARES.items():
            if u0 - 1e-4 <= u <= u1 + 1e-4 and v0 - 1e-4 <= v <= v1 + 1e-4:
                return "sq:" + n
        return "mixed"
    if uv[:, 1].max() <= 0.375 + 1e-4:
        for n, (top, bot) in TX.BANDS.items():
            if bot - 1e-4 <= v <= top + 1e-4:
                return "band:" + n
    return "mixed"


ISO_SQUARES = ("dark_sq", "leather")


def glove_chart_labels(cx, mb, side):
    """the leather glove's charts: one per bone segment (palm, each phalanx) and per half (dorsal / palmar for the
    palm, the two sides of each finger tube), so every chart is a disk (a whole hand does not unwrap without stretch)"""
    H, hdir, lat, dors = hand_frame(cx, side)
    V = np.asarray(mb.V, float)
    cache = {}

    def lab(fi):
        if fi in cache:
            return cache[fi]
        f = mb.F[fi]
        if mb.tag[f[0]] != "glove":
            cache[fi] = ""; return ""
        p = V[list(f)]
        n = G.nrm(np.cross(p[2] - p[0], p[3] - p[1]))
        w = {}
        for i in f:
            for b, x in mb.W[i].items():
                w[b] = w.get(b, 0) + x
        b = max(w, key=w.get)
        if b.startswith("hand") or b.startswith("lowerarm"):
            r = "palm_" + ("d" if np.dot(n, dors) > 0 else "p")
        else:
            h, t = cx.head(b), cx.tail(b)
            ax = G.nrm(t - h); c = p.mean(0) - h; c = c - np.dot(c, ax) * ax
            r = b + ("_a" if np.dot(np.cross(ax, c), dors) > 0 else "_b")
        cache[fi] = r
        return r
    return lab


def iso_uv_pass(mb, label_fn=None):
    """Iteration 2b (integrity G6, trim-sheet texel anisotropy): the plate-steel, dark-lining and leather faces of a
    piece get ISOTROPIC charts. Their arc-length grid UVs sheared by up to 28x where the rows of a dome / flare / cone
    differ in length (U starts at the same column on every row), and the lining / glove squares were constant UVs.
    Islands = connected faces of one region and tag, cut wherever the old UVs were discontinuous (tube seams); each is
    unwrapped with Minimum Stretch, turned so its U follows the old U direction (the steel's brushing), scaled to the
    region's texel density (steel 1024 px/m; squares: fitted inside the square) and placed back in its region (steel:
    the old U position, V kept inside the steel strip). Degenerate faces (hidden rib bottoms) join the steel. Trim
    bands, decals (lion, rosette, rivet heads) and etch strips keep their mapping."""
    nF = len(mb.F)
    if nF == 0:
        return 0
    UV = [np.asarray(q, float) for q in mb.UV]
    V = np.asarray(mb.V, float)
    reg = []; around = {}
    for fi, f in enumerate(mb.F):
        q = UV[fi]
        p = V[list(f)]
        e1, e2 = p[1] - p[0], p[2] - p[0]
        d1, d2 = q[1] - q[0], q[2] - q[0]
        det = d1[0] * d2[1] - d1[1] * d2[0]
        r = _uv_region(q)
        if abs(det) < 1e-12 or np.linalg.norm(np.cross(e1, e2)) < 1e-12:
            r = "steel" if r in ("steel", "mixed") or r.startswith("band") else r
            ed = [np.linalg.norm(p[(k + 1) % 4] - p[k]) for k in range(4)]
            k = int(np.argmax(ed)); around[fi] = p[(k + 1) % 4] - p[k]
        else:
            around[fi] = (e1 * d2[1] - e2 * d1[1]) / det            # 3D direction of +U (Pu)
        reg.append(r)
    # session 3 (G6 helmet streaks): the DARK turned-under rim rows (slot / breath borders, band 'dark', a flat colour)
    # are charted like the dark lining square
    reg = ["sq:dark_sq" if r == "band:dark" else r for r in reg]
    sel = [fi for fi in range(nF) if reg[fi] == "steel" or reg[fi] in ["sq:" + s for s in ISO_SQUARES]]
    fnorm = {}
    for fi in sel:
        p = V[list(mb.F[fi])]
        fnorm[fi] = G.nrm(np.cross(p[2] - p[0], p[3] - p[1]))
    if not sel:
        return 0
    # seams: old UV discontinuities between faces sharing an edge (tube wraps, region / tag borders)
    corner = {}
    for fi in sel:
        for k, v in enumerate(mb.F[fi]):
            corner[(fi, v)] = UV[fi][k]
    efaces = {}
    for fi in sel:
        f = mb.F[fi]
        for k in range(4):
            e = tuple(sorted((f[k], f[(k + 1) % 4])))
            efaces.setdefault(e, []).append(fi)
    seams = set()
    for e, fs in efaces.items():
        if len(fs) != 2:
            continue
        a, b = fs
        if reg[a] != reg[b] or mb.tag[mb.F[a][0]] != mb.tag[mb.F[b][0]] or \
                (label_fn is not None and label_fn(a) != label_fn(b)):
            seams.add(e); continue
        if any(np.abs(corner[(a, v)] - corner[(b, v)]).max() > 1e-5 for v in e):
            seams.add(e); continue
        # sharp creases (a closed sheet's turned edge, a shell's rim fold): charts are cut there
        na, nb_ = fnorm[a], fnorm[b]
        if float(np.dot(na, nb_)) < 0.3:
            seams.add(e)
    labels = ["%s|%s|%s" % (reg[fi], mb.tag[mb.F[fi][0]], label_fn(fi) if label_fn else "") for fi in sel]
    old = {fi: UV[fi].copy() for fi in sel}
    isl = []
    G.unwrap_faces(mb, sel, labels, around, TX.W / TX.STEEL_DENS, seam_edges=seams, islands_out=isl, iterations=40,
                   method=os.environ.get("RTS_ISO_METHOD", "MINIMUM_STRETCH"))
    # session 3 (G6): Minimum Stretch (SLIM) folds some long thin rings (the gorget collar: 45 % of lame A's steel at
    # anisotropy > 1.5); each island keeps the better of SLIM and a conformal (LSCM) unwrap
    if os.environ.get("RTS_ISO_DUAL", "1") == "1":
        uv_a = {fi: list(mb.UV[fi]) for fi in sel}
        isl_b = []
        G.unwrap_faces(mb, sel, labels, around, TX.W / TX.STEEL_DENS, seam_edges=seams, islands_out=isl_b,
                       iterations=40, method="CONFORMAL")
        uv_b = {fi: list(mb.UV[fi]) for fi in sel}

        def bad_area(fl, uvs):
            tot = 0.0; bad = 0.0
            for fi in fl:
                f = mb.F[fi]; q = np.asarray(uvs[fi], float)
                for (i0, i1, i2) in ((0, 1, 2), (0, 2, 3)):
                    e1, e2 = V[f[i1]] - V[f[i0]], V[f[i2]] - V[f[i0]]
                    d1, d2 = q[i1] - q[i0], q[i2] - q[i0]
                    ar = 0.5 * np.linalg.norm(np.cross(e1, e2)); tot += ar
                    det = d1[0] * d2[1] - d1[1] * d2[0]
                    if abs(det) < 1e-16:
                        bad += ar; continue
                    Pu = (e1 * d2[1] - e2 * d1[1]) / det; Pv = (e2 * d1[0] - e1 * d2[0]) / det
                    a_ = Pu @ Pu; c_ = Pv @ Pv; b_ = Pu @ Pv
                    tr = a_ + c_; dt = math.sqrt(max((a_ - c_) ** 2 + 4 * b_ * b_, 0))
                    s1 = math.sqrt(max((tr + dt) / 2, 0)); s2 = math.sqrt(max((tr - dt) / 2, 0))
                    if s1 > 1.5 * max(s2, 1e-12):
                        bad += ar
            return bad / max(tot, 1e-12)
        nb_ = 0
        for fl in isl:
            if bad_area(fl, uv_b) < bad_area(fl, uv_a) - 1e-6:
                nb_ += 1
            else:
                for fi in fl:
                    mb.UV[fi] = uv_a[fi]
        if nb_:
            log_("iso UVs %s: %d of %d islands conformal" % (mb.name if hasattr(mb, "name") else "", nb_, len(isl)))
    lo, hi = TX.STEEL[1] + TX.INSET, TX.STEEL[0] - TX.INSET
    for fl in isl:
        r = reg[fl[0]]
        Q = np.concatenate([np.asarray(mb.UV[fi], float) for fi in fl])
        mn, mx = Q.min(0), Q.max(0)
        if r == "steel":
            O = np.concatenate([old[fi] for fi in fl]); u0, v0 = O.mean(0)
            sc = min(1.0, (hi - lo) / max(mx[1] - mn[1], 1e-9))
            vc = float(np.clip(v0 - sc * 0.5 * (mn[1] + mx[1]), lo - sc * mn[1], hi - sc * mx[1]))
            off = np.array([u0, vc])
        else:
            u0_, v0_, u1_, v1_ = TX.SQUARES[r[3:]]
            m_ = 6.0 / TX.W
            sc = min(1.0, (u1_ - u0_ - 2 * m_) / max(mx[0] - mn[0], 1e-9), (v1_ - v0_ - 2 * m_) / max(mx[1] - mn[1], 1e-9))
            off = np.array([0.5 * (u0_ + u1_), 0.5 * (v0_ + v1_)]) - sc * 0.5 * (mn + mx)
        for fi in fl:
            mb.UV[fi] = [tuple(np.asarray(c_, float) * sc + off) for c_ in mb.UV[fi]]
    return len(sel)


def per_side(fn):
    return [(fn.__name__ + "_l", lambda cx: fn(cx, "l")), (fn.__name__ + "_r", lambda cx: fn(cx, "r"))]


# builder order matters: helmet -> plume (socket), cuirass -> gorget (neck roll, helmet rim), pauldron -> rerebrace
BUILDERS = ([("helmet", helmet), ("plume", plume), ("cuirass", cuirass), ("mail", mail), ("gorget", gorget)] +
            per_side(pauldron) + per_side(couter) + per_side(rerebrace) + per_side(vambrace) + per_side(gauntlet) +
            [("aventail", aventail)])
SLOT_ORDER = ["helmet", "plume", "gorget", "cuirass", "pauldron_l", "pauldron_r", "rerebrace_l", "rerebrace_r",
              "couter_l", "couter_r", "vambrace_l", "vambrace_r", "gauntlet_l", "gauntlet_r", "mail", "aventail"]
DEPS = {"plume": ["helmet"], "gorget": ["helmet", "cuirass", "mail"], "pauldron_l": ["cuirass", "gorget"],
        "pauldron_r": ["cuirass", "gorget"], "rerebrace_l": ["pauldron_l"], "rerebrace_r": ["pauldron_r"], "aventail": ["helmet", "gorget", "mail"],
        "mail": ["cuirass"], "gauntlet_l": ["vambrace_l"], "gauntlet_r": ["vambrace_r"]}   # session 4: the cuff seals


def generate(cx, names):
    """Build the MBs of `names` (+ what they depend on) on the body of cx: {name: MB}"""
    want = set(names)
    for n in list(want):
        want |= set(DEPS.get(n, []))
    out = {}
    for name, fn in BUILDERS:
        if name not in want:
            continue
        t0 = time.time()
        mb = fn(cx)
        if name not in ("plume", "mail", "aventail") and os.environ.get("RTS_ISO_UV", "1") != "0":
            if name.startswith("gauntlet"):
                lfn = glove_chart_labels(cx, mb, name[-1])
            elif name in ("gorget", "cuirass", "helmet"):
                # session 3 (G6: the gorget's lame A collar stayed 45 % anisotropic): rings round the neck / trunk
                # are charted as left / right halves (an annulus does not unwrap without shear)
                lfn = (lambda m_: (lambda fi: "L" if np.mean([m_.V[v][0] for v in m_.F[fi]]) > 0 else "R"))(mb)
            else:
                lfn = None
            iso_uv_pass(mb, label_fn=lfn)
        cx.shared[name + "_mb"] = mb
        out[name] = mb
        log_("piece %-13s verts %6d quads %6d tris %6d (%.1fs)" % (name, len(mb.V), len(mb.F), 2 * len(mb.F), time.time() - t0))
    if os.environ.get("RTS_NO_SETTLE") != "1":
        settle_layers(cx, out)
    return out


# ======================================================================================================= materials
def _img(path, cs):
    im = bpy.data.images.load(path, check_existing=True)
    im.colorspace_settings.name = cs
    if cs == "Non-Color":
        im.alpha_mode = 'NONE'
    return im


def _gltf_group():
    try:
        import char_materials
        return char_materials.gltf_output_group()
    except Exception:
        name = "glTF Material Output"
        ng = bpy.data.node_groups.get(name)
        if ng:
            return ng
        ng = bpy.data.node_groups.new(name, 'ShaderNodeTree')
        ng.interface.new_socket("Occlusion", socket_type="NodeSocketFloat")
        ng.nodes.new('NodeGroupInput'); ng.nodes.new('NodeGroupOutput')
        return ng


def orm_material(name, base, orm, normal, normal_strength=1.0, alpha_clip=None, double_sided=False, rough=None,
                 vcol=None):
    """Principled BSDF wired 1:1 for glTF: base (sRGB) -> Base Color [+ Alpha], ORM (R occlusion via the glTF output
    group, G roughness, B metallic), OpenGL normal map."""
    m = bpy.data.materials.get(name)
    if m is not None and m.get("rts_built"):
        return m                        # shared by every piece: never rebuild it under pieces that use it
    if m is None:
        m = bpy.data.materials.new(name)
    m["rts_built"] = True
    m.use_nodes = True
    nt = m.node_tree; N = nt.nodes; L = nt.links
    for n in list(N):
        N.remove(n)
    out = N.new("ShaderNodeOutputMaterial"); out.location = (600, 0)
    b = N.new("ShaderNodeBsdfPrincipled"); b.location = (250, 0)
    L.new(b.outputs[0], out.inputs[0])
    tb = N.new("ShaderNodeTexImage"); tb.image = _img(base, "sRGB"); tb.location = (-500, 250)
    if vcol:
        ca = N.new("ShaderNodeVertexColor"); ca.layer_name = vcol; ca.location = (-500, 450)
        mx = N.new("ShaderNodeMix"); mx.data_type = 'RGBA'; mx.blend_type = 'MULTIPLY'; mx.location = (-150, 300)
        mx.inputs["Factor"].default_value = 1.0
        L.new(tb.outputs["Color"], mx.inputs[6]); L.new(ca.outputs["Color"], mx.inputs[7])
        L.new(mx.outputs[2], b.inputs["Base Color"])
    else:
        L.new(tb.outputs["Color"], b.inputs["Base Color"])
    if orm:
        to = N.new("ShaderNodeTexImage"); to.image = _img(orm, "Non-Color"); to.location = (-500, -50)
        sp = N.new("ShaderNodeSeparateColor"); sp.location = (-200, -50)
        L.new(to.outputs["Color"], sp.inputs[0])
        L.new(sp.outputs[1], b.inputs["Roughness"]); L.new(sp.outputs[2], b.inputs["Metallic"])
        g = N.new("ShaderNodeGroup"); g.node_tree = _gltf_group(); g.location = (250, -450)
        if "Occlusion" in g.inputs:
            L.new(sp.outputs[0], g.inputs["Occlusion"])
    else:
        b.inputs["Roughness"].default_value = rough if rough is not None else 0.45
        b.inputs["Metallic"].default_value = 0.0
    if normal:
        tn = N.new("ShaderNodeTexImage"); tn.image = _img(normal, "Non-Color"); tn.location = (-500, -350)
        nm = N.new("ShaderNodeNormalMap"); nm.location = (-200, -350); nm.inputs["Strength"].default_value = normal_strength
        L.new(tn.outputs["Color"], nm.inputs["Color"]); L.new(nm.outputs[0], b.inputs["Normal"])
    if alpha_clip is not None:
        gt = N.new("ShaderNodeMath"); gt.operation = 'GREATER_THAN'; gt.inputs[1].default_value = alpha_clip
        L.new(tb.outputs["Alpha"], gt.inputs[0]); L.new(gt.outputs[0], b.inputs["Alpha"])
        m.surface_render_method = 'DITHERED'
    m.use_backface_culling = not double_sided
    return m


def make_material(slot):
    """The material for an armour slot (trim-sheet armour / mail / plume). Used by `dress` and by the assembler:
    outfit_lib.add_piece(rig, body, path, slot, material=armour_upper.make_material(slot))."""
    mf = TX.material_files()
    if slot in ("mail", "aventail"):
        b, o, n = mf["mail"]
        return orm_material("M_knight_mail", b, o, n, normal_strength=1.0, vcol=AO_ATTR, double_sided=True)
    if slot == "plume":
        b, o, n = mf["plume"]
        return orm_material("M_knight_plume", b, o, n, alpha_clip=0.35, double_sided=True, vcol=AO_ATTR)
    return orm_material("M_knight_armour", TX.FILES["armour_base"], TX.FILES["armour_orm"], TX.FILES["armour_normal"],
                        vcol=AO_ATTR)


AO_ATTR = "Color"        # colour attribute Blender's OBJ importer creates from the per-vertex AO in the asset OBJ


# ======================================================================================================= authoring
MATCH = {"helmet": "rts_rigid_head", "plume": "rts_rigid_head", "gorget": "rts_match_neck"}
ZDEPTH = {"mail": 45, "plume": 80, "aventail": 46}
DMAX = {"helmet": 0.075, "plume": 0.0, "aventail": 0.0, "gorget": 0.06, "cuirass": 0.07, "mail": 0.03, "pauldron": 0.10,
        "rerebrace": 0.05, "couter": 0.06, "vambrace": 0.045, "gauntlet": 0.03}


def piece_kind(name):
    return name.rsplit("_", 1)[0] if name.endswith(("_l", "_r")) else name


def match_groups(cx):
    """Body vertex groups used for MakeClothes matching of rigid head / neck pieces: head / neck skin that no facial
    expression moves, so the helmet follows the head shape (cust morphs) but never the face animation."""
    bm = cx.bm
    kb = bm.data.shape_keys.key_blocks
    n = len(bm.data.vertices)
    base = np.empty(n * 3); kb["Basis"].data.foreach_get("co", base); base = base.reshape(-1, 3)
    mv = np.zeros(n)
    for k in FACE_KEYS:
        if k in kb:
            a = np.empty(n * 3); kb[k].data.foreach_get("co", a)
            mv = np.maximum(mv, np.linalg.norm(a.reshape(-1, 3) - base, axis=1))
    body = cx.body
    static = mv[:body.n] < 2e-4
    ears = np.zeros(body.n, bool); ears[body.groups.get("ears", [])] = True
    headw = body.weight_sum(["head", "neck_02"]); neckw = body.weight_sum(["neck_01", "neck_02", "spine_05"])
    groups = {"rts_match_head": np.nonzero(static & ~ears & (headw > 0.5) & (body.co[:, 2] > 1.62))[0],
              "rts_match_neck": np.nonzero(static & (neckw > 0.5) & (body.co[:, 2] > 1.45))[0]}
    # RIGID_GROUP (exactly 3 vertices): MakeClothes matches every vertex of a rigid piece to this one triangle, so the
    # helmet / plume transform as a unit with the head (affine in the triangle plane + scaled normal offset) instead of
    # following individual scalp faces (which crumples a visor 10 cm away on another head shape)
    st = np.nonzero(static & ~ears)[0]
    tri = []
    for tgt in ((0.070, -0.055, 1.795), (-0.070, -0.055, 1.795), (0.0, 0.035, 1.80)):
        tri.append(int(st[np.argmin(np.linalg.norm(body.co[st] - np.array(tgt), axis=1))]))
    groups["rts_rigid_head"] = np.array(tri)
    log_("rigid head triangle", tri, body.co[tri].round(3).tolist())
    for nm, idx in groups.items():
        g = bm.vertex_groups.get(nm) or bm.vertex_groups.new(name=nm)
        g.add([int(i) for i in idx], 1.0, 'REPLACE')
        log_("match group", nm, len(idx))
    return groups


def match_clothes(cx, clothes, props, delete_group=None):
    """ClothesService.create_mhclo_from_clothes_matching with the basemesh cross-reference built once per session
    (MPFB rebuilds it for every piece, ~45 s each). allow_exact=False: keep every authored offset."""
    from bl_ext.user_default.mpfb.services import ClothesService, MeshService
    from bl_ext.user_default.mpfb.entities.clothes.mhclo import Mhclo
    from bl_ext.user_default.mpfb.entities.clothes.vertexmatch import VertexMatch
    from bl_ext.user_default.mpfb.entities.meshcrossref import MeshCrossRef
    from bl_ext.user_default.mpfb.entities.objectproperties import GeneralObjectProperties
    if getattr(cx, "_xref", None) is None:
        cx._xref = MeshCrossRef(cx.bm, after_modifiers=True, build_faces_by_group_reference=True)
        cx._refscale = ClothesService.get_reference_scale(cx.bm)
    mh = Mhclo(); mh.verts = dict(); mh.clothes = clothes
    for k, v in props.items():
        if hasattr(mh, k):
            setattr(mh, k, v)
    cxr = MeshCrossRef(clothes, after_modifiers=True, build_faces_by_group_reference=True)
    sf = GeneralObjectProperties.get_value("scale_factor", entity_reference=cx.bm)
    mh.max_pole = max(len(e) for e in cxr.edges_by_vertex)
    for vi in range(len(cxr.vertex_coordinates)):
        vm = VertexMatch(clothes, vi, cxr, cx.bm, cx._xref, scale_factor=sf, reference_scale=cx._refscale, allow_exact=False)
        mh.verts[vi] = vm.mhclo_line
    if delete_group and delete_group in cx.bm.vertex_groups:
        mh.delete_group = delete_group
        vv = sorted(set(int(v[0]) for v in MeshService.find_vertices_in_vertex_group(cx.bm, delete_group)))
        mh.delverts = vv
        mh.delete = True
    return mh


def covered_verts(cx, ob, name, tags=None):
    """Body vertices hidden by the piece: the normal ray and 4 tilted rays all hit the piece within DMAX, then eroded
    by one ring (so nothing shows at openings, and nothing pokes through in motion under the plates)."""
    dmax = DMAX.get(piece_kind(name), 0.05)
    if dmax <= 0:
        return []
    from mathutils.bvhtree import BVHTree
    me = ob.data
    V = [v.co.copy() for v in me.vertices]
    bvh = BVHTree.FromPolygons(V, [tuple(p.vertices) for p in me.polygons])
    body = cx.body
    lo = np.array([min(v[k] for v in V) for k in range(3)]) - dmax
    hi = np.array([max(v[k] for v in V) for k in range(3)]) + dmax
    cand = np.nonzero(np.all((body.co > lo) & (body.co < hi), axis=1))[0]
    hit = np.zeros(body.n, bool)
    if piece_kind(name) in ("mail", "gauntlet"):
        # skin-tight pieces: skin close to the piece is hidden too (the coarse mesh dips under the skin between
        # vertices), except within 2.5 cm of the piece's open edges (hem, sleeves, collar, wrist)
        from mathutils.kdtree import KDTree
        skin_faces = [tuple(p.vertices) for p in me.polygons
                      if tags is None or all(tags[v] in ("glove", "mail") for v in p.vertices)]
        loops = G.boundary_loops(skin_faces)
        bv = sorted(set(v for lp in loops for v in lp))
        kd = KDTree(max(1, len(bv)))
        for k, v in enumerate(bv):
            kd.insert(V[v], k)
        kd.balance()
        for i in cand:
            loc, nor, fi, dist = bvh.find_nearest(Vector(body.co[i]))
            if loc is not None and dist < (0.016 if piece_kind(name) == "mail" else 0.010):
                if not bv or kd.find(body.co[i])[2] > 0.025:
                    hit[i] = True
        cand = [i for i in cand if not hit[i]]
    for i in cand:
        p = body.co[i]; n = body.vn[i]
        t1 = G.nrm(np.cross(n, (0, 0, 1)) if abs(n[2]) < 0.9 else np.cross(n, (1, 0, 0))); t2 = np.cross(n, t1)
        ok = True
        for d in (n, G.nrm(n + 0.45 * t1), G.nrm(n - 0.45 * t1), G.nrm(n + 0.45 * t2), G.nrm(n - 0.45 * t2)):
            loc, nor, fi, dist = bvh.ray_cast(Vector(p + n * 0.0005), Vector(d), dmax)
            if loc is None:
                ok = False
                break
        hit[i] = ok
    nb = G.mesh_adjacency(body.n, body.faces)
    er = hit.copy()
    for i in np.nonzero(hit)[0]:
        if not all(hit[j] for j in nb[i]):
            er[i] = False
    return sorted(int(i) for i in np.nonzero(er)[0])


def vertex_ao(cx, ob, rays=20, dist=0.07, strength=0.75, floor=0.35):
    """Per-vertex ambient occlusion from the piece itself + the body (not other pieces: stays valid when pieces are
    mixed). Written into the OBJ as vertex colours -> a colour attribute -> glTF COLOR_0 (multiplies base colour)."""
    from mathutils.bvhtree import BVHTree
    me = ob.data
    n = len(me.vertices)
    V = np.array([v.co[:] for v in me.vertices]); N = np.array([v.normal[:] for v in me.vertices])
    polys = [tuple(p.vertices) for p in me.polygons]
    bt = [tuple(int(i) + n for i in t) for t in cx.body.tris]
    allv = [Vector(p) for p in V] + [Vector(p) for p in cx.body.co]
    bvh = BVHTree.FromPolygons(allv, polys + bt)
    rng = np.random.default_rng(3)
    u1, u2 = rng.uniform(size=rays), rng.uniform(size=rays)
    r = np.sqrt(u1); ph = 2 * np.pi * u2
    loc_dirs = np.stack([r * np.cos(ph), r * np.sin(ph), np.sqrt(1 - u1)], -1)       # cosine-weighted hemisphere
    ao = np.ones(n)
    for i in range(n):
        nn = N[i]
        t1 = G.nrm(np.cross(nn, (0, 0, 1)) if abs(nn[2]) < 0.9 else np.cross(nn, (1, 0, 0))); t2 = np.cross(nn, t1)
        o = Vector(V[i] + nn * 0.0006)
        hit = 0
        for d in loc_dirs:
            dd = d[0] * t1 + d[1] * t2 + d[2] * nn
            if bvh.ray_cast(o, Vector(dd), dist)[0] is not None:
                hit += 1
        ao[i] = 1 - strength * hit / rays
    nb = G.mesh_adjacency(n, polys)
    ao = 0.5 * ao + 0.5 * np.array([ao[x].mean() if x else ao[k] for k, x in enumerate(nb)])
    return np.clip(ao, floor, 1.0)


def obj_add_colors(path, ao):
    """Append per-vertex colours to the OBJ's 'v' lines (Blender's OBJ importer reads 'v x y z r g b')."""
    out, k = [], 0
    for line in open(path):
        if line.startswith("v "):
            a = float(ao[k]); k += 1
            line = line.rstrip("\n") + " %.4f %.4f %.4f\n" % (a, a, a)
        out.append(line)
    assert k == len(ao), (k, len(ao))
    open(path, "w").write("".join(out))


def apply_weighted_normals(ob):
    for o in bpy.context.view_layer.objects:
        o.select_set(False)
    bpy.context.view_layer.objects.active = ob
    ob.select_set(True)
    md = ob.modifiers.new("wn", 'WEIGHTED_NORMAL')
    md.mode = 'FACE_AREA'; md.weight = 50; md.keep_sharp = False; md.thresh = 0.01
    bpy.ops.object.modifier_apply(modifier=md.name)


def write_weights(path, mb, rig, name):
    bones = [b.name for b in rig.data.bones]
    W = {b: [] for b in bones}
    for i, w in enumerate(mb.W):
        tot = sum(x for bn, x in w.items() if bn in W)
        if tot <= 0:
            raise RuntimeError("%s: vertex %d has no weights" % (name, i))
        for bn, x in w.items():
            if bn in W and x / tot > 1e-4:
                W[bn].append([i, round(x / tot, 5)])
    with open(path, "w") as f:
        json.dump({"name": "%s weights" % name, "license": "CC0", "version": 110, "weights": W}, f)



def sidecar_path(name, root=None):
    return os.path.join(root or ASSET_DIR, PREFIX + name, PREFIX + name + ".rts.json")


def load_topo(names):
    topo = {}
    for n in names:
        p = sidecar_path(n)
        if not os.path.exists(p):
            p = piece_file(n, ".rts.json")
        if os.path.exists(p):
            topo.update(json.load(open(p)).get("topo", {}))
    return topo


def author(cx, objs):
    from bl_ext.user_default.mpfb.services import ClothesService
    groups = match_groups(cx)
    os.makedirs(ASSET_DIR, exist_ok=True)
    report = {}
    RG.ensure_helpers(cx.rig)
    if "plume_spine" in cx.shared:
        ensure_plume_bones(cx.rig, plume_bone_specs(cx.shared["plume_spine"]))
    stage = os.path.join(ASSET_DIR, ".stage_upper_%d" % os.getpid())
    os.makedirs(stage, exist_ok=True)
    for name, (ob, mb) in objs.items():
        t0 = time.time()
        asset = PREFIX + name
        d = os.path.join(stage, asset)
        if os.path.exists(d):
            shutil.rmtree(d)
        os.makedirs(d)
        if name != "plume":
            apply_weighted_normals(ob)
        mo = bpy.data.objects.new(asset, ob.data.copy())
        bpy.context.scene.collection.objects.link(mo)
        mo.vertex_groups.clear()
        g = mo.vertex_groups.new(name=MATCH.get(name, "body"))
        g.add(list(range(len(mo.data.vertices))), 1.0, 'REPLACE')
        assert len(set(len(p.vertices) for p in mo.data.polygons)) == 1, name
        assert MATCH.get(name, "body") in cx.bm.vertex_groups and tuple(mo.scale) == tuple(cx.bm.scale), name
        used = set(i for p in mo.data.polygons for i in p.vertices)
        assert len(used) == len(mo.data.vertices), (name, "loose vertices")
        cov = covered_verts(cx, ob, name, tags=mb.tag)
        dname = "Delete." + asset
        dg = cx.bm.vertex_groups.get(dname) or cx.bm.vertex_groups.new(name=dname)
        if cov:
            dg.add(cov, 1.0, 'REPLACE')
        props = {"name": asset, "description": "RTS knight upper armour: " + name,
                 "author": "RTS project (procedural, scripts/armour_upper.py)", "license": "CC0", "homepage": "",
                 "uuid": str(uuid.uuid5(uuid.NAMESPACE_URL, "rts-knight-" + name))}
        mh = match_clothes(cx, mo, props, delete_group=dname if cov else None)
        mh.material = asset + ".mhmat"
        mh.zdepth = ZDEPTH.get(name, 70)
        path = os.path.join(d, asset + ".mhclo")
        mh.write_mhclo(path, reference_scale=cx._refscale, also_export_mhmat=False)
        if name == "plume":
            obj_add_colors(path.replace(".mhclo", ".obj"), mb.meta["shade"])
        else:
            obj_add_colors(path.replace(".mhclo", ".obj"), vertex_ao(cx, ob))
        write_weights(os.path.join(d, asset + ".mhw"), mb, cx.rig, asset)
        with open(os.path.join(d, asset + ".mhmat"), "w") as f:
            f.write("# MakeHuman material, RTS knight armour (CC0); the pipeline applies armour_upper.make_material()\n"
                    "name %s\ndiffuseColor %s\nshininess 0.5\n%s" % (
                        asset, "0.1 0.2 0.6" if name == "plume" else "0.55 0.57 0.62",
                        "transparent True\nbackfaceCull False\n" if name == "plume" else ""))
        topo = {k: v for k, v in getattr(cx, "topo_out", {}).items()
                if (name.startswith("gauntlet") and k == "glove_" + name[-1]) or (name in ("mail", "aventail") and k == name)
                or (name == "gorget" and k.startswith("gorget_")) or (name == "helmet" and k.startswith("helm_"))}
        side = dict(verts=len(ob.data.vertices), topo=topo, bones=sorted({b for w in mb.W for b in w}),
                    plume_spine=np.round(cx.shared["plume_spine"], 5).tolist() if name == "plume" else None,
                    note="generator topology data (armour_upper.py): a regeneration on another body reuses it")
        json.dump(side, open(sidecar_path(name, stage), "w"))
        if not ASSET_OVR:                   # MPFB user-data copy (what dress loads), staged next to it, then swapped
            ust = os.path.join(USER_DATA, "clothes", ".stage_upper_%d_%s" % (os.getpid(), asset))
            if os.path.exists(ust):
                shutil.rmtree(ust)
            shutil.copytree(d, ust)
            swap_dir(ust, os.path.join(USER_DATA, "clothes", asset))
        swap_dir(d, os.path.join(ASSET_DIR, asset))
        bpy.data.objects.remove(mo, do_unlink=True)
        report[name] = dict(verts=len(ob.data.vertices), tris=sum(len(p.vertices) - 2 for p in ob.data.polygons),
                            covered=len(cov))
        log_("authored %-13s %5d verts %5d tris covers %5d body verts (%.1fs)" % (
            name, report[name]["verts"], report[name]["tris"], len(cov), time.time() - t0))
    shutil.rmtree(stage, ignore_errors=True)
    mf = os.path.join(ASSET_DIR, "knight_upper_manifest.json")
    old = json.load(open(mf))["pieces"] if os.path.exists(mf) else {}
    old.update(report)
    pieces = {n: old[n] for n in SLOT_ORDER if n in old}
    with open(mf + ".tmp", "w") as f:
        json.dump(dict(pieces=pieces, total_tris=sum(v["tris"] for v in pieces.values()), prefix=PREFIX,
                       slots=list(pieces), material="armour_upper.make_material(slot)",
                       helpers=RG.helper_specs().__len__(), extra_bones=sorted(list(RG.helper_specs()) + list(PLUME_BONES)),
                       load="armour_upper.dress(kind) (regenerates on the target body) or outfit_lib.add_piece(..., "
                            "material=make_material(slot)) after armour_upper_rig.ensure_helpers + ensure_plume_bones"),
                  f, indent=1)
    os.replace(mf + ".tmp", mf)
    return report


# ======================================================================================================= dressing
def ensure_ao_attr(ob):
    """The armour materials multiply base colour by the per-vertex AO attribute; pieces without it get a white one."""
    me = ob.data
    if AO_ATTR not in me.color_attributes:
        ca = me.color_attributes.new(AO_ATTR, 'BYTE_COLOR', 'POINT')
        ca.data.foreach_set("color", np.ones(len(me.vertices) * 4))
    return me.color_attributes[AO_ATTR]


def installed_pieces():
    return [n for n in SLOT_ORDER if os.path.exists(piece_file(n))]


def area_normals(me, co):
    """area-weighted vertex normals of positions co (n, 3) with me's polygons"""
    vn = np.zeros_like(co)
    for p in me.polygons:
        vs = list(p.vertices)
        for k in range(1, len(vs) - 1):
            a, b, c = co[vs[0]], co[vs[k]], co[vs[k + 1]]
            n = np.cross(b - a, c - a)
            vn[vs[0]] += n; vn[vs[k]] += n; vn[vs[k + 1]] += n
    return G.nrm(vn)


def islands(me):
    n = len(me.vertices)
    par = list(range(n))

    def find(x):
        while par[x] != x:
            par[x] = par[par[x]]; x = par[x]
        return x
    for e in me.edges:
        a, b = find(e.vertices[0]), find(e.vertices[1])
        if a != b:
            par[a] = b
    comp = np.array([find(i) for i in range(n)])
    return [np.nonzero(comp == c)[0] for c in np.unique(comp)]


def similarity(A, B):
    """s, R, t minimising |s R A + t - B| (Umeyama)"""
    ma, mb_ = A.mean(0), B.mean(0)
    A0, B0 = A - ma, B - mb_
    U, Sg, Vt = np.linalg.svd(B0.T @ A0)
    D = np.eye(3); D[2, 2] = np.sign(np.linalg.det(U @ Vt))
    R = U @ D @ Vt
    s = float((Sg * np.diag(D)).sum() / max((A0 ** 2).sum(), 1e-12))
    return s, R, mb_ - s * R @ ma


def rigid_island(ob, idx):
    """True if every vertex of the island is 100 % on the same single bone"""
    names = {g.index: g.name for g in ob.vertex_groups}
    first = None
    for i in idx[:: max(1, len(idx) // 40)]:
        gs = [(names[g.group], g.weight) for g in ob.data.vertices[int(i)].groups if g.weight > 1e-3]
        if len(gs) != 1:
            return False
        if first is None:
            first = gs[0][0]
        elif gs[0][0] != first:
            return False
    return True


def transfer_uvs(me, mb):
    """write the generator's UVs (on THIS body) into the fitted piece when its faces are the generator's faces in the
    same order (iteration 2b: the charts of the mail / plates are body-specific; the asset carries the male's)"""
    if len(me.polygons) != len(mb.F) or not me.uv_layers:
        return 0
    uvl = me.uv_layers.active or me.uv_layers[0]
    out = np.empty(len(me.loops) * 2)
    for p, f, q in zip(me.polygons, mb.F, mb.UV):
        vs = list(p.vertices)
        if len(vs) != len(f):
            return 0
        try:
            k = vs.index(f[0])
        except ValueError:
            return 0
        if [vs[(k + i) % len(vs)] for i in range(len(vs))] != list(f):
            return 0
        for i, li in enumerate(p.loop_indices):
            out[2 * li:2 * li + 2] = q[(i - k) % len(vs)]
    uvl.data.foreach_set("uv", out)
    return len(mb.F)


def regen_on_body(kind, objs):
    """Replace the MPFB-fitted rest shape of each piece by the generator's output on THIS body (same topology), keep
    every morph as a delta (rigid islands: the morph's similarity transform, so plates never bend)."""
    cx = Ctx(kind)
    cx.topo = load_topo(list(objs))
    mbs = generate(cx, [n for n in objs])
    report = {}
    for name, ob in objs.items():
        mb = mbs.get(name)
        me = ob.data
        if mb is None or len(mb.V) != len(me.vertices):
            report[name] = "kept MPFB fit (topology %s vs %d)" % (None if mb is None else len(mb.V), len(me.vertices))
            log_("regen %-12s SKIPPED: %s" % (name, report[name]))
            continue
        new = np.array(mb.V)
        n = len(new)
        if me.shape_keys:
            kb = me.shape_keys.key_blocks
            old = np.empty(n * 3); kb[0].data.foreach_get("co", old); old = old.reshape(-1, 3)
            isl = islands(me)
            rig_isl = [rigid_island(ob, ix) for ix in isl]
            for key in list(kb)[1:]:
                co = np.empty(n * 3); key.data.foreach_get("co", co); co = co.reshape(-1, 3)
                out = new + (co - old)
                for ix, rg in zip(isl, rig_isl):
                    if rg and len(ix) >= 3 and np.abs(co[ix] - old[ix]).max() > 1e-6:
                        s, R, t = similarity(old[ix], co[ix])
                        out[ix] = s * new[ix] @ R.T + t
                # sub-micron deltas (similarity round-off) would become float-noise sparse morph entries whose declared
                # min / max the glTF validator rejects: snap them to exactly zero
                dl = out - new
                dl[np.linalg.norm(dl, axis=1) < 2e-5] = 0.0
                out = new + dl
                key.data.foreach_set("co", out.ravel())
            kb[0].data.foreach_set("co", new.ravel())
        me.vertices.foreach_set("co", new.ravel())
        me.update()
        nuv = transfer_uvs(me, mb)
        if nuv:
            log_("regen %-12s UVs regenerated on this body (%d faces)" % (name, nuv))
        try:
            me.normals_split_custom_set_from_vertices([tuple(v) for v in area_normals(me, new)])
        except Exception as ex:
            log_("normals", name, ex)
        dev = float(np.linalg.norm(new - (old if me.shape_keys else new), axis=1).max()) if me.shape_keys else 0.0
        report[name] = "regenerated (max move %.1f mm from the MPFB fit)" % (dev * 1000)
        log_("regen %-12s %s" % (name, report[name]))
    return cx, report


def fit_plume_bones(rig, objs, cx=None):
    spine = None
    if cx is not None and "plume_spine" in cx.shared:
        spine = cx.shared["plume_spine"]
    elif "plume" in objs:
        side = json.load(open(sidecar_path("plume"))) if os.path.exists(sidecar_path("plume")) else {}
        if side.get("plume_spine"):
            # authored spine carried by the plume's fitted similarity (MPFB rigid-head fit)
            path = piece_file("plume", ".obj")
            A = []
            for line in open(path):
                if line.startswith("v "):
                    x, y, z = map(float, line.split()[1:4]); A.append((x, -z, y))
            A = np.array(A); B = np.array([v.co[:] for v in objs["plume"].data.vertices])
            s, R, t = similarity(A, B)
            spine = s * np.array(side["plume_spine"]) @ R.T + t
    if spine is not None:
        ensure_plume_bones(rig, plume_bone_specs(np.asarray(spine)))
    return spine is not None


GRIP_CACHE = {}           # (rig, grip, head) -> solved fist (the contract's joint solve, reused by post_clips)
GRIP_CONTRACT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "out", "grip_contract.json")
GRIP_SPAN = (-0.126, 0.036)   # the sword's grip wrap on the socket axis (+ margins): what the fist closes over
GRIP_TOP = 0.018          # the sword's upper ferrule starts here on the socket axis (armour_lower_props sword)
INDEX_BELOW = 0.014       # the index knuckle this far below it (the index finger's width ~2 cm)
GRIP_CLEAR = 0.0016       # the palm / MCP pads (vertices and facet samples) over the grip surface at the contract head


def grip_contract_for(rig, kind, gauntlets):
    """Grip contract (user item 29, for armour_lower.socket_specs): per hand socket the head (hand-local) of the grip
    cylinder so the gauntlet's palm, metacarpal and MCP pads (vertices + face centres / edge midpoints, rest pose) lie
    GRIP_CLEAR outside it; direction = armour_lower's canonical fist axis (their frame), the head starts at the fist-axis
    centre and moves palmar-away. The fingers are then solved round that cylinder (post_clips / solve_grip_mesh)."""
    import armour_lower as AL
    B = rig.data.bones
    hd = lambda n: np.array(B[n].head_local[:]); tl = lambda n: np.array(B[n].tail_local[:])
    dh, dt = AL.rig_json_positions()
    out = {}
    for side, sock, which in (("r", "socket_weapon_r", "grip_r"), ("l", "socket_hand_l", "grip_l")):
        ob = gauntlets.get("gauntlet_" + side)
        if ob is None:
            continue
        rad = GRIPS[which][2]
        amt = AL.curl_amount(side)
        cen, _, _, _, _ = AL.fist_axis(hd, tl, side, amt)
        _, axis_c, _, _, axh_c = AL.fist_axis(dh, dt, side, amt)
        d = np.array(AL.frame_yz(axis_c, axh_c))[:, 1]
        V = np.array([v.co[:] for v in ob.data.vertices])
        names = {g.index: g.name for g in ob.vertex_groups}
        dom = [max([(x.weight, names[x.group]) for x in v.groups])[1] if v.groups else "" for v in ob.data.vertices]
        mcp = {f: hd("%s_01_%s" % (f, side)) for f in FINGERS}
        obst = np.zeros(len(V), bool)
        for i, b in enumerate(dom):
            if b == "hand_" + side or "metacarpal" in b:
                obst[i] = True
            else:
                f = b.split("_")[0]
                # the whole proximal phalanx at rest (iteration 2b: the diagonal grip ran under the pinky's; no knuckle
                # angle frees a phalanx the grip already passes through)
                if f in mcp and b == "%s_01_%s" % (f, side) and np.linalg.norm(V[i] - mcp[f]) < 0.030:
                    obst[i] = True
        pts = [V[i] for i in np.nonzero(obst)[0]]
        for p in ob.data.polygons:
            vs = list(p.vertices)
            if all(obst[vs]):
                Q = V[vs]
                pts.append(Q.mean(0))
                pts += [0.5 * (Q[a] + Q[(a + 1) % len(vs)]) for a in range(len(vs))]
        X = np.array(pts)
        palm_c = V[[i for i, b in enumerate(dom) if b == "hand_" + side]].mean(0)
        pal = cen - palm_c; pal = pal - np.dot(pal, d) * d; pal = pal / np.linalg.norm(pal)

        def minr(c):
            v = X - c; al = v @ d
            r = np.linalg.norm(v - np.outer(al, d), axis=1)
            m = (al > GRIP_SPAN[0]) & (al < GRIP_SPAN[1])
            return float(r[m].min()) if m.any() else 1.0
        need = rad + GRIP_CLEAR
        # along the grip first: the index finger lies UNDER the sword's upper ferrule / guard (grip wrap up to GRIP_TOP
        # on the socket axis): the socket moves toward the guard until the index knuckle is INDEX_BELOW below it; then
        # palmar-away until the pads clear (the axis runs diagonally across the palm: the order matters)
        c0 = cen.copy()
        if side == "r":
            ai = float(np.dot(hd("index_01_" + side) - cen, d))
            sh = max(0.0, ai - (GRIP_TOP - INDEX_BELOW))
            c0 = cen + d * sh
            log_("grip contract %s: index knuckle %.1f mm along the grip -> socket %.1f mm toward the guard" % (
                kind, ai * 1000, sh * 1000))
        lo, hi = 0.0, 0.06
        if minr(c0) >= need:
            hi = 0.0
        for _ in range(40):
            if hi == 0.0:
                break
            m_ = 0.5 * (lo + hi)
            if minr(c0 + pal * m_) >= need:
                hi = m_
            else:
                lo = m_
        head = c0 + pal * hi
        # joint solve (iteration 2b, G5 'hand triangles through the grip'): close the fist round this head; if any part
        # of the posed gauntlet (a proximal phalanx that already crossed the grip line at rest, a facet) ends inside
        # the grip, move the head further palmar-away and solve again
        rep = {}
        head0 = head.copy()
        for it in range(4):
            rep = {}
            gp = RG.solve_grip_mesh(rig, side, head, d, rad, ob, report=rep, span=GRIP_SPAN if side == "r" else (-0.06, 0.06))
            if rep["min_gap"] >= 0.0002 or np.linalg.norm(head - head0) > 0.006:
                break
            head = head + pal * min(0.003, 0.0005 - rep["min_gap"])
        GRIP_CACHE[(rig.name, which, tuple(np.round(head, 4)), False)] = gp
        Hi = np.linalg.inv(np.array(B["hand_" + side].matrix_local))
        out[sock] = dict(parent="hand_" + side, radius=rad, head_rest=np.round(head, 5).tolist(),
                         axis_rest=np.round(d, 5).tolist(), fist_centre_rest=np.round(cen, 5).tolist(),
                         head_in_hand_local=np.round(Hi[:3, :3] @ head + Hi[:3, 3], 5).tolist(),
                         axis_in_hand_local=np.round(Hi[:3, :3] @ d, 5).tolist(),
                         palmar_shift_m=round(float(np.dot(head - cen, pal)), 5), palmar_dir_rest=np.round(pal, 4).tolist(),
                         min_pad_radius_mm=round(minr(head) * 1000, 2), span_m=list(GRIP_SPAN),
                         fist_min_gap_mm=round(rep.get("min_gap", 0) * 1000, 2), fist_enclosure=round(rep.get("enclosure", 0), 3))
        log_("grip contract %s %s: head %.1f mm palmar of the fist centre (pads >= %.1f mm, grip r %.1f; solved fist: "
             "closest %.2f mm, enclosure %.2f)" % (kind, sock, np.dot(head - cen, pal) * 1000, minr(head) * 1000, rad * 1000,
                                                  rep.get("min_gap", 0) * 1000, rep.get("enclosure", 0)))
    return out


def write_grip_contract(kind, socks, path=None):
    """update this body's entry in out/grip_contract.json (read-modify-write under a lock: male / female assemblies run
    in parallel; atomic replace, so the lower armour never reads half a file)"""
    import fcntl
    path = path or GRIP_CONTRACT
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + ".lock", "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        try:
            d = json.load(open(path)) if os.path.exists(path) else {}
        except Exception:
            d = {}
        old = d.get("bodies", {}).get(kind, {})
        same = all(s in old and np.allclose(old[s].get("head_in_hand_local", [9, 9, 9]), v["head_in_hand_local"], atol=2e-5)
                   and abs(old[s].get("radius", 0) - v["radius"]) < 1e-6 for s, v in socks.items())
        if same and d.get("version") == 2:
            return False
        d.update(version=2, owner="armour_upper.py (grip_contract_for, written by dress(): UPPER fixer iteration 2b)",
                 consumer="armour_lower.socket_specs (sword on socket_weapon_r, shield hand strap on socket_hand_l)",
                 status="current: computed from the gauntlet on every dress (body-specific); the fingers are solved "
                        "round this cylinder by armour_upper.post_clips (solve_grip_mesh, gauntlet mesh vs cylinder)",
                 updated=time.strftime("%Y-%m-%d %H:%M"),
                 rule="socket_weapon_r / socket_hand_l: rest ROTATION = armour_lower's canonical fist-axis frame (+Y "
                      "along the grip, pinky -> index); HEAD = head_in_hand_local (hand bone local; = head_rest in "
                      "armature rest space): the fist-axis centre moved palmar-away until the gauntlet's palm, "
                      "metacarpal and MCP pads (vertices + facet samples) clear the grip cylinder of 'radius' by "
                      "%.1f mm over span_m (socket-frame y). Sword: a round grip of 'radius' under the fist; shield: a "
                      "hand strap / handle of 'radius' whose centre line is socket_hand_l's +Y axis." % (GRIP_CLEAR * 1000),
                 units="metres, Blender armature rest space (Z up, the knight faces -Y)")
        d.setdefault("bodies", {})[kind] = socks
        tmp = path + ".tmp%d" % os.getpid()
        json.dump(d, open(tmp, "w"), indent=1)
        os.replace(tmp, path)
    log_("grip contract written for %s: %s" % (kind, path))
    return True


def dress(kind, pieces=None, check_fit=True, regen=True):
    """Load the upper armour onto the live human in the open file (helper joints + plume bones first, then every
    piece through MPFB, then the rest shapes regenerated on this body); returns {slot: object}."""
    import outfit_lib
    rig = bpy.data.objects["rts_" + kind]; bm = bpy.data.objects[kind + "_body"]
    pose_reset(rig)
    RG.ensure_helpers(rig)
    ensure_plume_bones(rig)
    out = {}
    for name in pieces or installed_pieces():
        path = piece_file(name)
        ob = outfit_lib.add_piece(rig, bm, path, slot=name, material=make_material(name))
        ob["rts_set"] = "knight_upper"
        # every plate component is authored 100 % on one joint / helper joint (C1, M1): rig_helpers.apply_plate_rules
        # keeps authored rigid components as they are (e.g. the vambrace's upper cannon on lowerarm, lower on the twist)
        ob["rts_weights"] = "authored"
        out[name] = ob
    cx = None
    if regen:
        cx, rep = regen_on_body(kind, out)
        rig["rts_upper_regen"] = json.dumps(rep)
    fit_plume_bones(rig, out, cx)
    if "gorget" in out:
        out["gorget"]["rts_bare_safe"] = True        # its top edge clears the chin: no helm-only split needed
    for name, ob in out.items():
        ensure_ao_attr(ob)
    if os.environ.get("RTS_GRIP_CONTRACT", "1") != "0" and any(n.startswith("gauntlet") for n in out):
        try:
            write_grip_contract(kind, grip_contract_for(rig, kind, out))
        except Exception as ex:                          # the armour_lower module mid-edit must not stop the dress
            log_("WARNING grip contract not updated: %r" % ex)
    if check_fit and kind == "male":
        for name, ob in out.items():
            path = piece_file(name)
            bpy.ops.wm.obj_import(filepath=path.replace(".mhclo", ".obj"), use_split_objects=False, use_split_groups=False)
            ref = bpy.context.selected_objects[0]
            bpy.ops.object.transform_apply(location=False, rotation=True, scale=False)
            a = np.array([v.co[:] for v in ref.data.vertices]); b_ = np.array([v.co[:] for v in ob.data.vertices])
            dev = np.linalg.norm(a - b_, axis=1)
            log_("fit %-13s max %.2f mm  mean %.2f mm  (on the authoring body)" % (name, dev.max() * 1000, dev.mean() * 1000))
            bpy.data.objects.remove(ref, do_unlink=True)
    log_("helpers at rest: max %.4f deg" % RG.helper_check(rig))
    return out


# ======================================================================================================= clips hook
def body_actions(rig):
    table = json.loads(rig.get("rts_clips", "{}"))
    names = [n for n in table if n != "talk_emote"] or [a.name for a in bpy.data.actions]
    return [bpy.data.actions[n] for n in names if bpy.data.actions.get(n)]


GRIPS = {"grip_r": ("r", "socket_weapon_r", 0.0146),      # sword grip: a round cylinder under the fist (grip contract)
         "grip_l": ("l", "socket_hand_l", 0.0105)}         # the shield's rolled leather hand strap


def grip_pose(rig, which="grip_r"):
    """closed fist around the cylinder on the hand socket's +Y axis (C2): the sword grip (right) / the shield's hand
    strap (left), solved phalanx by phalanx against the gauntleted finger radius"""
    side, sock, rad = GRIPS[which]
    if sock not in rig.data.bones:
        return {}
    if rig.animation_data:                      # an assigned clip would re-pose the fingers under the solver
        rig.animation_data.action = None
    pose_reset(rig)
    M = rig.data.bones[sock].matrix_local
    c = np.array(M.translation[:]); d = np.array((M.to_3x3() @ Vector((0, 1, 0)))[:])
    kind = rig.name.replace("rts_", "")
    ob = bpy.data.objects.get("%s_gauntlet_%s" % (kind, side))
    if ob is not None and ob.parent == rig:
        # iteration 2b: the fist is solved against the gauntlet MESH (glove + scales), not a finger capsule; the
        # contract's joint solve (dress) is reused when the socket sits where the contract put it
        prop = bpy.data.objects.get("%s_sword" % kind) if side == "r" else None
        key = (rig.name, which, tuple(np.round(c, 4)), prop is not None)
        if key in GRIP_CACHE:
            return GRIP_CACHE[key]
        obst = None; span = GRIP_SPAN if side == "r" else (-0.06, 0.06)
        if prop is not None:
            # the fingers close on the REAL sword at rest: its grip wrap as an exact cylinder (its largest radius, over
            # its length) and the ferrules / guard / pommel as a surface
            from mathutils.bvhtree import BVHTree
            dg = bpy.context.evaluated_depsgraph_get()
            pose_reset(rig)
            me = prop.evaluated_get(dg).to_mesh()
            Mw = prop.matrix_world
            VV = [Mw @ v.co for v in me.vertices]
            gm = {k for k, m in enumerate(prop.data.materials) if m and "grip" in m.name.lower()}
            gv = np.array([VV[i][:] for p in me.polygons if p.material_index in gm for i in p.vertices])
            if len(gv):
                x = gv - c; al = x @ d
                rad = float(np.linalg.norm(x - np.outer(al, d), axis=1).max())
                span = (float(al.min()) - 0.003, float(al.max()) + 0.003)
            obst = BVHTree.FromPolygons(VV, [tuple(p.vertices) for p in me.polygons if p.material_index not in gm])
            prop.evaluated_get(dg).to_mesh_clear()
        gp = RG.solve_grip_mesh(rig, side, c, d, rad, ob, obstacle=obst, span=span)
        GRIP_CACHE[key] = gp
        return gp
        GRIP_CACHE[(rig.name, which, tuple(np.round(c, 4)))] = gp
        return gp
    return RG.solve_grip(rig, side, c, d, rad, finger_r=0.0118)


def plume_spring(rig, actions, k=60.0, c=9.0, g=2.5):
    """Damped-spring lag of the plume chain behind the head motion, baked into the clips (quaternion keys)."""
    chain = [b for b in PLUME_BONES if b in rig.pose.bones]
    if not chain:
        return 0
    meshes = [o for o in rig.children if o.type == 'MESH']
    for o in meshes:
        o.hide_viewport = True
    sc = bpy.context.scene
    fps = sc.render.fps or 30
    dt = 1.0 / fps
    rest = {b: rig.data.bones[b].matrix_local.copy() for b in chain}
    n = 0
    for act in actions:
        rig.animation_data.action = act
        nf = int(round(act.frame_range[1])) + 1
        loop = json.loads(rig.get("rts_clips", "{}")).get(act.name, {}).get("loop", False)
        heads = []
        for f in range(nf):
            sc.frame_set(f)
            heads.append(rig.pose.bones["head"].matrix.copy())
        Rh = rig.data.bones["head"].matrix_local
        # simulate the tail of each bone as a point mass hanging from its (simulated) head
        L = [rig.data.bones[b].length for b in chain]
        cycles = 2 if loop else 1
        pos = None; vel = None
        rec = []
        for cyc in range(cycles):
            for f in range(nf):
                Hm = heads[f] @ Rh.inverted()                   # rest -> posed (head rigid)
                tgt = [np.array((Hm @ rest[b] @ Vector((0, L[i], 0, 1))).to_3d()[:]) for i, b in enumerate(chain)]
                root0 = np.array((Hm @ rest[chain[0]].translation.to_4d()).to_3d()[:])
                if pos is None:
                    pos = [t.copy() for t in tgt]; vel = [np.zeros(3) for _ in tgt]
                for i in range(len(chain)):
                    acc = k * (tgt[i] - pos[i]) - c * vel[i] + np.array([0, 0, -g]) * 0.02
                    vel[i] = vel[i] + acc * dt; pos[i] = pos[i] + vel[i] * dt
                # keep bone lengths (chain from the rigid root)
                prev = root0
                for i in range(len(chain)):
                    v = pos[i] - prev; ln = np.linalg.norm(v)
                    pos[i] = prev + v / max(ln, 1e-9) * L[i]
                    prev = pos[i]
                if cyc == cycles - 1:
                    rec.append(([p.copy() for p in pos], Hm.copy(), root0))
        # local rotations: parent posed frame -> bone rest direction to the simulated direction
        frames = np.arange(nf, dtype=float)
        qs = {b: [] for b in chain}
        for pos_f, Hm, root0 in rec:
            parent = Hm @ rig.data.bones["head"].matrix_local        # posed head (armature)
            prevM = parent; prevR = rig.data.bones["head"].matrix_local; head_pt = root0
            for i, b in enumerate(chain):
                restM = rest[b]
                inh = prevM @ (prevR.inverted() @ restM)           # inherited (unrotated) posed frame
                y0 = (inh.to_3x3() @ Vector((0, 1, 0))).normalized()
                y1 = Vector(pos_f[i] - np.array(inh.translation[:])).normalized()
                qw = y0.rotation_difference(y1)
                ql = inh.to_3x3().to_quaternion().inverted() @ qw @ inh.to_3x3().to_quaternion()
                qs[b].append(ql)
                prevM = Matrix.Translation(inh.translation) @ (qw.to_matrix() @ inh.to_3x3()).to_4x4()
                prevR = restM
        for b in chain:
            arr = np.array([q[:] for q in qs[b]])
            for i in range(1, len(arr)):
                if np.dot(arr[i], arr[i - 1]) < 0:
                    arr[i] = -arr[i]
            if loop:
                arr[-1] = arr[0]
            for kk in range(4):
                fc = act.fcurve_ensure_for_datablock(rig, 'pose.bones["%s"].rotation_quaternion' % b, index=kk, group_name=b)
                fc.keyframe_points.clear(); fc.keyframe_points.add(nf)
                co = np.empty(nf * 2); co[0::2] = frames; co[1::2] = arr[:, kk]
                fc.keyframe_points.foreach_set("co", co)
                fc.keyframe_points.foreach_set("interpolation", [1] * nf)
                fc.update()
        n += 1
    for o in meshes:
        o.hide_viewport = False
    return n


def post_clips(rig, kind, actions=None):
    """Hook for build_knight after the clips exist: closed sword grip (solved against the grip) in every body clip,
    plume spring motion. Idempotent."""
    acts = actions or body_actions(rig)
    cur = rig.animation_data.action if rig.animation_data else None
    poses = {}
    for which in GRIPS:
        gp = grip_pose(rig, which)
        if not gp:
            continue
        RG.apply_hand_pose(rig, gp, acts)
        poses[which] = {b: [round(x, 6) for x in q] for b, q in gp.items()}
        log_("%s pose (solved around %s) written into %d clips: %s" % (
            which, GRIPS[which][1], len(acts), {b: round(math.degrees(q.angle), 1) for b, q in gp.items()}))
    if poses:
        rig["rts_hand_poses"] = json.dumps(poses)
    pose_reset(rig)
    ns = plume_spring(rig, acts)
    log_("plume spring baked into %d clips" % ns)
    if rig.animation_data:
        rig.animation_data.action = cur
    pose_reset(rig)


# ======================================================================================================= objects
def make_objects(builders, cx):
    coll = bpy.data.collections.get("armour") or bpy.data.collections.new("armour")
    if coll.name not in bpy.context.scene.collection.children:
        bpy.context.scene.collection.children.link(coll)
    mbs = generate(cx, [n for n, _ in builders])
    objs = {}
    for name, mb in mbs.items():
        ob = mb.to_object(PREFIX + name, coll)
        ob["rts_part"] = name
        objs[name] = (ob, mb)
    return objs


def bind_to_rig(objs, rig):
    """parent the (unfitted, authoring-body) objects to the rig with an Armature modifier (fast QA without MPFB)"""
    for name, (ob, mb) in objs.items():
        ob.parent = rig
        md = ob.modifiers.new("Armature", 'ARMATURE'); md.object = rig
        ob["rts_part"] = name

# ======================================================================================================= look-dev
def studio_world(strength=1.0, bg=(0.035, 0.036, 0.04)):
    """Procedural studio environment for metals: dark floor, grey horizon, bright top + two softbox strips (lighting and
    reflections only); the camera sees a flat charcoal background like the reference sheet."""
    sc = bpy.context.scene
    w = bpy.data.worlds.get("W_studio") or bpy.data.worlds.new("W_studio")
    sc.world = w
    w.use_nodes = True
    nt = w.node_tree; N = nt.nodes; L = nt.links
    for n in list(N):
        N.remove(n)
    out = N.new("ShaderNodeOutputWorld")
    tc = N.new("ShaderNodeTexCoord")
    sep = N.new("ShaderNodeSeparateXYZ"); L.new(tc.outputs["Generated"], sep.inputs[0])
    ramp = N.new("ShaderNodeValToRGB")
    L.new(sep.outputs["Z"], ramp.inputs[0])
    cr = ramp.color_ramp
    cr.elements[0].position = 0.40; cr.elements[0].color = (0.015, 0.015, 0.018, 1)
    cr.elements[1].position = 0.75; cr.elements[1].color = (0.55, 0.57, 0.62, 1)
    e = cr.elements.new(0.52); e.color = (0.12, 0.12, 0.13, 1)
    col = ramp.outputs[0]
    # softboxes: bright where the view direction is close to two directions
    def box(dirv, sharp, amount, tint):
        v = N.new("ShaderNodeVectorMath"); v.operation = 'DOT_PRODUCT'
        L.new(tc.outputs["Generated"], v.inputs[0]); v.inputs[1].default_value = dirv
        mr = N.new("ShaderNodeMapRange"); mr.inputs[1].default_value = sharp; mr.inputs[2].default_value = 1.0
        mr.inputs[3].default_value = 0.0; mr.inputs[4].default_value = amount
        L.new(v.outputs["Value"], mr.inputs[0])
        mx = N.new("ShaderNodeMix"); mx.data_type = 'RGBA'; mx.blend_type = 'ADD'
        mx.inputs["Factor"].default_value = 1.0
        L.new(col_holder[0], mx.inputs[6])
        rgb = N.new("ShaderNodeRGB"); rgb.outputs[0].default_value = (*tint, 1)
        mul = N.new("ShaderNodeMix"); mul.data_type = 'RGBA'; mul.blend_type = 'MULTIPLY'; mul.inputs["Factor"].default_value = 1.0
        L.new(rgb.outputs[0], mul.inputs[6]); L.new(mr.outputs[0], mul.inputs[7])
        L.new(mul.outputs[2], mx.inputs[7])
        col_holder[0] = mx.outputs[2]
    col_holder = [col]
    box((-0.55, -0.62, 0.55), 0.93, 6.0, (1.0, 0.95, 0.88))
    box((0.75, 0.35, 0.55), 0.95, 3.0, (0.8, 0.88, 1.0))
    box((0.0, 0.2, 1.0), 0.85, 1.5, (1, 1, 1))
    bgl = N.new("ShaderNodeBackground"); L.new(col_holder[0], bgl.inputs[0]); bgl.inputs[1].default_value = strength
    bgc = N.new("ShaderNodeBackground"); bgc.inputs[0].default_value = (*bg, 1); bgc.inputs[1].default_value = 1.0
    lp = N.new("ShaderNodeLightPath")
    mx = N.new("ShaderNodeMixShader")
    L.new(lp.outputs["Is Camera Ray"], mx.inputs[0]); L.new(bgl.outputs[0], mx.inputs[1]); L.new(bgc.outputs[0], mx.inputs[2])
    L.new(mx.outputs[0], out.inputs[0])
    return w


def lookdev_setup(engine="BLENDER_EEVEE", samples=64):
    sc = render_setup(engine, (900, 1300), samples, world=1.0)
    studio_world(1.0)
    sc.view_settings.look = 'AgX - Medium High Contrast'
    sc.view_settings.exposure = 0.0
    if engine == "BLENDER_EEVEE":
        try:
            sc.eevee.use_raytracing = True
            sc.eevee.ray_tracing_options.resolution_scale = '1'
            sc.eevee.use_shadows = True
        except Exception:
            pass
    for o in [o for o in bpy.data.objects if o.type == 'LIGHT']:
        bpy.data.objects.remove(o, do_unlink=True)
    add_light("Key", 'AREA', (-2.4, -2.6, 3.2), (50, 0, -42), 700, 2.5, (1.0, 0.95, 0.88))
    add_light("Rim", 'AREA', (1.6, 2.8, 2.9), (-58, 0, 150), 900, 1.5, (0.75, 0.85, 1.0))
    add_light("Rim2", 'AREA', (-2.2, 2.2, 2.2), (-60, 0, -140), 400, 1.5, (1.0, 0.9, 0.8))
    add_light("Fill", 'AREA', (2.8, -2.6, 1.4), (75, 0, 48), 180, 3.0, (0.85, 0.9, 1.0))
    if not bpy.data.objects.get("Floor"):
        bpy.ops.mesh.primitive_plane_add(size=40, location=(0, 0, 0))
        fl = bpy.context.object; fl.name = "Floor"
        set_material(fl, pbr_material("M_floor", base_color=(0.03, 0.03, 0.033, 1), rough=0.85))
    return sc


def relaxed_pose(rig):
    pose_reset(rig)
    for sd in ("l", "r"):
        sg = 1 if sd == "l" else -1
        aim(rig, "upperarm_" + sd, (0.20 * sg, 0.06, -1.0))
        aim(rig, "lowerarm_" + sd, (0.10 * sg, -0.30, -1.0))
        aim(rig, "hand_" + sd, (0.05 * sg, -0.25, -1.0))


def lookdev(kind, tag="look", engine="BLENDER_EEVEE", shots=None, pose="relaxed"):
    rig = bpy.data.objects["rts_" + kind]
    for o in bpy.data.objects:
        if o.type == 'MESH' and o.get("rts_variant_group") == "eyebrows" and not o.get("rts_default"):
            o.hide_render = True
        if o.type == 'MESH' and o.name.endswith("_hair") and any(x.get("rts_part") == "helmet" for x in rig.children):
            o.hide_render = True
    sc = lookdev_setup(engine, 96 if engine == "CYCLES" else 64)
    if pose == "relaxed":
        relaxed_pose(rig)
    else:
        pose_reset(rig)
    H = 1.9
    c = 1.05
    views = {
        "front": ((0, -4.4, c + 0.15), (0, 0, c), 50, (800, 1300)),
        "right": ((-4.4, 0, c + 0.15), (0, 0, c), 50, (800, 1300)),
        "back": ((0, 4.4, c + 0.15), (0, 0, c), 50, (800, 1300)),
        "left": ((4.4, 0, c + 0.15), (0, 0, c), 50, (800, 1300)),
        "34": ((-2.9, -3.3, c + 0.45), (0, 0, c), 50, (800, 1300)),
        "helm": ((-0.55, -0.72, 1.83), (0, -0.07, 1.72), 70, (900, 1000)),
        "helm_front": ((0.0, -0.95, 1.76), (0, -0.07, 1.72), 70, (900, 1000)),
        "helm_side": ((-0.95, -0.05, 1.8), (0, -0.02, 1.72), 70, (900, 1000)),
        "helm_back": ((0.0, 0.95, 1.78), (0, 0.0, 1.73), 70, (900, 1000)),
        "helm_top": ((0.0, -0.12, 2.75), (0, -0.02, 1.75), 70, (900, 1000)),
        "plume_side": ((-1.45, 0.02, 1.78), (0, 0.10, 1.72), 60, (900, 1000)),
        "plume_back": ((0.55, 1.35, 2.0), (0, 0.08, 1.70), 60, (900, 1000)),
        "neck": ((-0.45, -0.85, 1.68), (0, -0.03, 1.58), 60, (900, 1000)),
        "arm": ((-1.05, -0.55, 1.35), (-0.33, -0.05, 1.22), 50, (900, 1000)),
        "hand_top": ((-0.55, -0.6, 1.25), (-0.45, -0.2, 0.95), 60, (900, 1000)),
        "shoulder": ((-0.95, -0.6, 1.62), (-0.26, -0.03, 1.43), 70, (900, 1000)),
        "torso": ((0.0, -1.35, 1.45), (0, -0.05, 1.3), 55, (900, 1000)),
        "hand": ((-0.7, -0.75, 1.1), (-0.42, -0.2, 0.95), 70, (900, 1000)),
        "rts": ((-7.5, -9.5, 10.5), (0, 0, 0.9), 135, (500, 500)),
    }
    for nm, (loc, tgt, lens, res) in views.items():
        if shots and nm not in shots:
            continue
        sc.render.resolution_x, sc.render.resolution_y = res
        turn = nm in ("back", "left", "helm_back")
        rig.rotation_euler.z = math.pi if nm == "back" else (-math.pi / 2 if nm == "left" else 0.0)
        bpy.context.view_layer.update()
        if nm in ("left", "back"):
            loc = (0, -4.4, c + 0.15)
        camera(loc, tgt, lens=lens)
        render(os.path.join(REN, "armour_upper_%s_%s_%s.png" % (tag, kind, nm)))
        rig.rotation_euler.z = 0.0
        bpy.context.view_layer.update()


# ======================================================================================================= pose test
def _sd(s_):
    return 1 if s_ == "_l" else -1


POSES = {
    "rest": lambda rig: None,
    "arms_up": lambda rig: [(rot(rig, "clavicle" + s_, (0, 1, 0), -18 * _sd(s_)),
                             aim(rig, "upperarm" + s_, (0.22 * _sd(s_), 0.05, 1)),
                             aim(rig, "lowerarm" + s_, (0.12 * _sd(s_), 0.05, 1))) for s_ in ("_l", "_r")],
    "t_pose": lambda rig: [(aim(rig, "upperarm" + s_, (_sd(s_), 0, 0)), aim(rig, "lowerarm" + s_, (_sd(s_), 0, 0)),
                            aim(rig, "hand" + s_, (_sd(s_), 0, 0))) for s_ in ("_l", "_r")],
    "arms_forward": lambda rig: [(rot(rig, "clavicle" + s_, (0, 0, 1), 10 * _sd(s_)),
                                  aim(rig, "upperarm" + s_, (0.12 * _sd(s_), -1, 0.05)),
                                  aim(rig, "lowerarm" + s_, (0.05 * _sd(s_), -1, 0.05))) for s_ in ("_l", "_r")],
    "elbow": lambda rig: [(aim(rig, "upperarm" + s_, (0.25 * _sd(s_), -0.25, -1)),
                           aim(rig, "lowerarm" + s_, (0.1 * _sd(s_), -1, 0.0)),
                           rot(rig, "lowerarm" + s_, (1, 0, 0), -65)) for s_ in ("_l", "_r")],
    "torso_twist": lambda rig: ([rot(rig, b, (0, 0, 1), 11) for b in ("spine_01", "spine_02", "spine_03", "spine_04", "spine_05")]
                                + [rot(rig, b, (0, 0, 1), 12) for b in ("neck_01", "neck_02")]),
    "bend_fwd": lambda rig: [rot(rig, b, (1, 0, 0), 12) for b in ("spine_02", "spine_03", "spine_04", "spine_05")],
    "head_turn": lambda rig: ([rot(rig, b, (0, 0, 1), 25) for b in ("neck_01", "neck_02", "head")]
                              + [rot(rig, "neck_01", (1, 0, 0), -12), rot(rig, "head", (1, 0, 0), -18)]),
    "fists": lambda rig: [curl(rig, s_, 75) for s_ in ("_l", "_r")],
}


def curl(rig, s_, deg):
    aim(rig, "upperarm" + s_, (0.3 * _sd(s_), -0.6, -0.75)); rot(rig, "lowerarm" + s_, (1, 0, 0), -60)
    for f in ("index", "middle", "ring", "pinky"):
        for i, d in ((1, deg), (2, deg * 1.1), (3, deg * 0.8)):
            pb = rig.pose.bones["%s_%02d%s" % (f, i, s_)]
            pb.rotation_mode = 'XYZ'; pb.rotation_euler.x += math.radians(d)
    for i in (2, 3):
        pb = rig.pose.bones["thumb_%02d%s" % (i, s_)]
        pb.rotation_mode = 'XYZ'; pb.rotation_euler.x += math.radians(deg * 0.6)
    bpy.context.view_layer.update()


def pose_test(kind, tag="pose", engine="BLENDER_EEVEE", poses=None):
    rig = bpy.data.objects["rts_" + kind]
    body = bpy.data.objects[kind + "_body"]
    for o in bpy.data.objects:
        if o.type == 'MESH' and o.get("rts_variant_group") == "eyebrows" and not o.get("rts_default"):
            o.hide_render = True
    sc = lookdev_setup(engine, 32)
    for name, fn in POSES.items():
        if poses and name not in poses:
            continue
        pose_reset(rig)
        fn(rig)
        try:
            drive_correctives(rig, body)
        except Exception:
            pass
        for vn, (loc, tgt, lens, res) in {"front34": ((-2.3, -3.6, 1.6), (0, 0, 1.15), 45, (560, 760)),
                                          "back34": ((2.5, 3.4, 1.8), (0, 0, 1.15), 45, (560, 760)),
                                          "shoulder": ((-0.9, -0.9, 1.75), (-0.25, 0, 1.45), 50, (560, 560))}.items():
            sc.render.resolution_x, sc.render.resolution_y = res
            camera(loc, tgt, lens=lens)
            render(os.path.join(REN, "armour_upper_%s_%s_%s_%s.png" % (tag, kind, name, vn)))
    pose_reset(rig)


def check_winding(ob):
    he = {}
    bad = 0
    for p in ob.data.polygons:
        vs = list(p.vertices)
        for k in range(len(vs)):
            e = (vs[k], vs[(k + 1) % len(vs)])
            if e in he:
                bad += 1
            he[e] = True
    return bad


def preview(objs, cx, tag="prev"):
    for o in bpy.data.objects:
        if o.type == 'MESH' and o.parent == cx.rig and not o.name.endswith("_body"):
            o.hide_render = True
    sc = render_setup("BLENDER_WORKBENCH", res=(900, 1100))
    sc.display.shading.light = 'STUDIO'
    sc.display.shading.color_type = 'OBJECT'
    sc.display.shading.show_cavity = True
    sc.display.shading.cavity_type = 'BOTH'
    sc.display.shading.show_specular_highlight = True
    for name, (ob, mb) in objs.items():
        ob.color = (0.78, 0.78, 0.8, 1)
    cx.bm.color = (0.75, 0.55, 0.45, 1)
    H = objs["helmet"][1].meta["helm"] if "helmet" in objs else None
    hc = Vector((0, -0.06, 1.74))
    only = getattr(cx, "shots", None)
    shots = {
        "head_front": ((0, -0.9, 1.76), hc, 60), "head_34": ((-0.55, -0.7, 1.82), hc, 60),
        "head_side": ((0.9, -0.05, 1.76), hc, 60), "head_back": ((0.25, 0.85, 1.85), hc, 60),
        "head_top": ((0.0, -0.25, 2.55), hc, 60),
        "body_front": ((0, -3.2, 1.35), Vector((0, 0, 1.2)), 55), "body_34": ((-2.0, -2.5, 1.6), Vector((0, 0, 1.2)), 55),
        "body_side": ((3.2, 0, 1.35), Vector((0, 0, 1.2)), 55), "body_back": ((0.6, 3.2, 1.5), Vector((0, 0, 1.2)), 55),
        "torso_34": ((-1.0, -1.3, 1.55), Vector((0, -0.02, 1.35)), 60),
        "arm_l_front": ((0.45, -1.2, 1.45), Vector((0.36, -0.06, 1.3)), 60), "arm_l_side": ((1.4, -0.2, 1.5), Vector((0.36, -0.06, 1.3)), 60),
        "arm_l_back": ((0.7, 1.1, 1.5), Vector((0.36, -0.04, 1.3)), 60),
        "hand_l_out": ((1.0, -0.45, 1.25), Vector((0.59, -0.3, 1.09)), 70), "hand_l_top": ((0.8, -0.9, 1.6), Vector((0.59, -0.3, 1.09)), 70), "arm_r_34": ((-1.1, -0.9, 1.55), Vector((-0.36, -0.06, 1.3)), 60),
    }
    for nm, (loc, tgt, lens) in shots.items():
        if only and not any(nm.startswith(o) for o in only):
            continue
        camera(loc, tgt, lens=lens)
        render(os.path.join(REN, "armour_upper_%s_%s.png" % (tag, nm)))




if __name__ == "__main__":
    t0 = time.time()
    if MODE in ("preview", "author"):
        cx = Ctx("male")
        todo = [(n, f) for n, f in BUILDERS if not ONLY or n in ONLY or any(n in DEPS.get(o, []) for o in ONLY)]
        objs = make_objects(todo, cx)
        if ONLY:
            objs = {n: v for n, v in objs.items() if n in ONLY}
        for n, (ob, mb) in objs.items():
            wc = check_winding(ob)
            assert wc == 0, (n, wc)
        if MODE == "preview":
            preview(objs, cx)
        else:
            author(cx, objs)
    elif MODE == "dress":
        kind = [a for a in ARGS[1:] if a in PRESETS][0] if any(a in PRESETS for a in ARGS[1:]) else "male"
        pcs = [a for a in ARGS[1:] if a not in PRESETS] or None
        objs = dress(kind, pcs)
        rig = bpy.data.objects["rts_" + kind]; bm = bpy.data.objects[kind + "_body"]
        fn = os.path.join(OUT, "knight_upper.blend" if kind == "male" else "knight_upper_female.blend")
        bpy.ops.wm.save_as_mainfile(filepath=fn)
        log_("saved", fn)
        import outfit_lib
        os.makedirs(os.path.join(OUT, "test"), exist_ok=True)
        glb = os.path.join(OUT, "test", "knight_upper_%s.glb" % kind)
        res = outfit_lib.export_dressed(rig, bm, kind, glb, drop=("hair", "eyebrows_alt"))
        log_("export", res)
        bpy.ops.wm.save_as_mainfile(filepath=os.path.join(OUT, "test", "knight_upper_%s_export.blend" % kind))
    elif MODE == "render":
        kind = ARGS[1] if len(ARGS) > 1 else "male"
        tag = ARGS[2] if len(ARGS) > 2 else "look"
        eng = "CYCLES" if "cycles" in ARGS else "BLENDER_EEVEE"
        if "m3" in ARGS:                       # the steel the assembled knight ships (steel_pbr M3 maps)
            m3 = os.path.join(TX.TEX_DIR, "_m3")
            for im in list(bpy.data.images):
                q = os.path.join(m3, os.path.basename(im.filepath or ""))
                if im.filepath and os.path.exists(q):
                    im.filepath = q; im.reload()
        shots = [a for a in ARGS[3:] if a not in ("cycles", "m3", "rest")] or None
        lookdev(kind, tag, eng, shots, pose="rest" if "rest" in ARGS else "relaxed")
    elif MODE == "poses":
        kind = ARGS[1] if len(ARGS) > 1 else "male"
        pose_test(kind, "pose", "BLENDER_EEVEE", [a for a in ARGS[2:]] or None)
    elif MODE == "look":
        # fast textured look-dev of the GENERATED pieces on the open base file (no MPFB fit): EEVEE, trim-sheet
        # materials, per-vertex AO optional ('ao'); shots = lookdev view names (default: all)
        kind = "female" if "rts_female" in bpy.data.objects else "male"
        cx = Ctx(kind)
        objs = make_objects(BUILDERS, cx)
        RG.ensure_helpers(cx.rig)
        ensure_plume_bones(cx.rig, plume_bone_specs(cx.shared["plume_spine"]))
        bind_to_rig(objs, cx.rig)
        for n, (ob, mb) in objs.items():
            ob.data.materials.append(make_material(n))
            ca = ensure_ao_attr(ob)
            if n == "plume" and "shade" in mb.meta:
                s = np.repeat(np.asarray(mb.meta["shade"], float), 4).reshape(-1, 4); s[:, 3] = 1
                ca.data.foreach_set("color", s.ravel())
            elif "ao" in ARGS:
                a = vertex_ao(cx, ob)
                s = np.repeat(a, 4).reshape(-1, 4); s[:, 3] = 1
                ca.data.foreach_set("color", s.ravel())
            if n != "plume":
                apply_weighted_normals(ob)
        m3 = os.path.join(TX.TEX_DIR, "_m3")                   # the assembled knight uses steel_pbr's M3 steel
        for im in list(bpy.data.images):
            q = os.path.join(m3, os.path.basename(im.filepath or ""))
            if im.filepath and os.path.exists(q) and "m3" not in ARGS[1:2]:
                im.filepath = q; im.reload()
        shots = [a for a in ARGS[1:] if a not in ("ao", "cycles", "rest")] or None
        lookdev(kind, "dev", "CYCLES" if "cycles" in ARGS else "BLENDER_EEVEE", shots,
                pose="rest" if "rest" in ARGS else "relaxed")
    log_("done %.1fs" % (time.time() - t0))
