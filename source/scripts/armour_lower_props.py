"""Knight props for scripts/armour_lower.py: longsword, kite shield, scabbard (Blender Python, numpy only).

Every prop is built in its SOCKET frame (the socket bone's local space: +Y along the bone, metres):
  sword     origin = centre of the grip where the fist closes, +Y = towards the point, +X = the edges' plane,
            +Z = the flat of the blade. Crossguard at y = +0.052, pommel at y = -0.155. Attach: socket_weapon_r;
            sheathed: socket_sword_sheathed.
  shield    origin = the forearm strap (enarmes) on the back, +Y = down the shield towards the point, +Z = out of the
            face (away from the forearm), +X = across. Attach: socket_shield_l (forearm) or socket_back.
  scabbard  origin = centre of the mouth, +Y = towards the chape (tip). Attach: socket_scabbard_l.
Face UVs use the slot conventions of armour_lower_geo (metric for tileables, the shield face planar on the shared
shield_lion atlas: U = x / 0.56 m from the left edge seen from the front, V = 1 at the top edge, 0 at the point).
"""
import math, json, os
import numpy as np
from armour_lower_geo import Mesh, SLOT, nrm, _arclen, rounded_box, smoothstep

SHIELD_W, SHIELD_H = 0.56, 0.96
# enarmes (forearm straps) contract with the socket (armour_lower.socket_specs reads these): socket_shield_l sits at
# SHIELD_STRAP_AT of the forearm (elbow -> wrist), SHIELD_STRAP_OFF off the forearm axis (along the socket's +Z, out of
# the shield face), and the shield's long axis (+Y) is the forearm turned SHIELD_STRAP_DIAG degrees in the shield
# plane; so in the socket frame the forearm axis runs through (0, 0, -OFF) along (sin DIAG, cos DIAG, 0).
SHIELD_STRAP_AT, SHIELD_STRAP_OFF, SHIELD_STRAP_DIAG = 0.95, 0.110, 45.0
FOREARM_R = 0.067            # strap loop radius round the forearm axis (vambrace + its flared cuff + padding)


def _catmull_closed(P, n=8):
    P = np.asarray(P, float); m = len(P); out = []
    for i in range(m):
        p0, p1, p2, p3 = P[(i - 1) % m], P[i], P[(i + 1) % m], P[(i + 2) % m]
        for t in np.linspace(0, 1, n, endpoint=False):
            t2, t3 = t * t, t * t * t
            out.append(0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2 + (-p0 + 3 * p1 - 3 * p2 + p3) * t3))
    return np.array(out)


def _resample_closed(P, n):
    Q = np.vstack([P, P[:1]]); al = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(Q, axis=0), axis=1))])
    t = np.linspace(0, al[-1], n, endpoint=False)
    return np.stack([np.interp(t, al, Q[:, k]) for k in range(P.shape[1])], 1)


def loft(m, sections, slot, closed=True, uv_scale=1.0, cap=None):
    """Quads between consecutive rings (each ring = (k, 3) points). UV: u = arc around (metres), v = distance along."""
    R = [np.asarray(s, float) for s in sections]
    idx = [[m.add_v(p) for p in r] for r in R]
    along = [0.0]
    for a, b in zip(R[:-1], R[1:]):
        along.append(along[-1] + float(np.linalg.norm(a.mean(0) - b.mean(0))))
    k = len(R[0])
    for j in range(len(R) - 1):
        al = _arclen(np.vstack([R[j], R[j][:1]])) if closed else _arclen(R[j])
        for i in range(k if closed else k - 1):
            i2 = (i + 1) % k
            vs = [idx[j][i], idx[j][i2], idx[j + 1][i2], idx[j + 1][i]]
            uv = [(al[i], along[j]), (al[i + 1], along[j]), (al[i + 1], along[j + 1]), (al[i], along[j + 1])]
            m.add_f(vs, [(u * uv_scale, v * uv_scale) for u, v in uv], SLOT[slot])
    return idx


def fan_cap(m, ring_idx, centre, slot, flip=False):
    c = m.add_v(centre)
    k = len(ring_idx)
    for i in range(k):
        i2 = (i + 1) % k
        vs = [c, ring_idx[i], ring_idx[i2]] if not flip else [c, ring_idx[i2], ring_idx[i]]
        P = [np.array(m.V[v]) for v in vs]
        m.add_f(vs, [(p[0], p[1] + p[2]) for p in P], SLOT[slot])


# ------------------------------------------------------------------------------------------------ longsword
# iteration 2b (user items 28 / 29, integrity G1 / G2 / G4 / G5): every part is ONE closed, outward-wound solid (the
# old grip / guard / blade were open or inside-out lofts: the single-sided grip wrap vanished from outside) and parts
# meet with <= 1 mm of embedding (no plate-through-plate unions): blade (base 1 mm into the guard), crossguard with the
# centre lozenge in the same surface, two enamel cabochons seated 0.6 mm, and the whole hilt (ferrules, grip wrap,
# wire rings, neck, pommel, button) one lathe whose top end sits 1 mm into the guard. Under the fist the grip is a
# ROUND cylinder of the grip contract's radius (out/grip_contract.json: the fingers are solved round that cylinder
# on socket_weapon_r's +Y axis).
GRIP_R = 0.0146               # default = armour_upper's grip contract radius
FIST_SPAN = (-0.062, 0.034)   # socket-frame y range the fist (and the second hand's index) closes over: exactly round


def grip_radius():
    try:
        p = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "out", "grip_contract.json")
        c = json.load(open(p))
        rs = [b["socket_weapon_r"]["radius"] for b in c["bodies"].values() if "socket_weapon_r" in b]
        return float(np.mean(rs)) if rs else GRIP_R
    except Exception:
        return GRIP_R


def loft_solid(m, sections, slot, cap0=None, cap1=None):
    """closed solid through closed rings (each (k, 3)), optional pole points closing the ends; wound outward"""
    f0 = len(m.F)
    idx = loft(m, sections, slot)
    k = len(sections[0])
    for ring, cp, first in ((idx[0], cap0, True), (idx[-1], cap1, False)):
        if cp is None:
            continue
        c = m.add_v(cp)
        for i in range(k):
            i2 = (i + 1) % k
            vs = [c, ring[i2], ring[i]] if first else [c, ring[i], ring[i2]]
            m.add_f(vs, [(0.0, 0.0), (0.01, 0.0), (0.0, 0.01)], SLOT[slot])
    from armour_lower_geo import orient_solid
    orient_solid(m, f0)
    return idx


def sword():
    from armour_lower_geo import lathe
    m = Mesh()
    y_guard, L = 0.052, 0.86
    y_base = 0.064                                        # blade base 1 mm inside the guard's top (guard ry 0.013)

    def half_width(t):                                   # t 0..1 along the blade
        return 0.0265 - 0.0105 * t - 0.0115 * smoothstep(0.80, 1.0, t)

    def thick(t):
        return 0.0036 - 0.0016 * t
    # --- blade: 12-point diamond section with a fuller (no sliver faces: every face >= 1/11 of the section spacing)
    secs = []
    ts = list(np.linspace(0, 0.62, 9)) + list(np.linspace(0.66, 0.99, 13))
    for t in ts:
        y = y_base + L * t
        w = float(half_width(t)); th = float(thick(t))
        fl = 0.0072 * (1 - smoothstep(0.50, 0.64, t)); fd = 0.0013 * (1 - smoothstep(0.50, 0.64, t))
        bev = w * 0.62
        ring = [(w, 0), (bev, th * 0.85), (fl + 0.0012, th), (0, th - fd), (-fl - 0.0012, th), (-bev, th * 0.85),
                (-w, 0), (-bev, -th * 0.85), (-fl - 0.0012, -th), (0, -th + fd), (fl + 0.0012, -th), (bev, -th * 0.85)]
        secs.append([(x, y, z) for x, z in ring])
    loft_solid(m, secs, "blade", cap0=(0.0, y_base, 0.0), cap1=(0.0, y_base + L, 0.0))
    # --- crossguard: one elliptic lathe along X (sections (ry, rz)), flared cross terminals, the centre lozenge swells
    # in the same surface, poles at the tips
    xs = [0.114, 0.109, 0.100, 0.086, 0.064, 0.040, 0.024, 0.014, 0.0]
    ry = [0.0, 0.0085, 0.0118, 0.0095, 0.0074, 0.0072, 0.0085, 0.0118, 0.0132]
    rz = [0.0, 0.0072, 0.0098, 0.0080, 0.0066, 0.0064, 0.0082, 0.0112, 0.0122]
    prof = [(1.0 if r > 0 else 0.0, x) for x, r in zip(xs, ry)]
    prof = prof + [(p, -x) for p, x in prof[-2::-1]]
    scl = [(a, b) for a, b in zip(ry, rz)]; scl = scl + scl[-2::-1]
    gi0 = len(m.V)
    _lathe_scaled(m, prof, (0, y_guard, 0), (-1, 0, 0), (0, 1, 0), 18, scl, "gold")
    # enamel cabochons on both faces of the lozenge (0.6 mm seated)
    for sz in (1.0, -1.0):                              # (the lozenge curves fast in y: small stones, edge 0.3 mm in)
        z0 = sz * 0.0110
        lathe([(0.0, 0.0), (0.0050, 0.0), (0.0046, 0.0012), (0.0030, 0.0024), (0.0, 0.0029)], (0, y_guard, z0),
              (0, 0, sz), (1, 0, 0), 14, mat=SLOT["enamel"], mesh=m, metric=True)
    # --- hilt: ferrule, round grip (contract radius under the fist), wire rings, ferrule, neck, pommel (flattened
    # disc), button: one lathe along +Y (the socket axis) from the pommel button (pole) to 1 mm inside the guard
    r = grip_radius()
    yp = -0.160
    H = [  # (r, y, slot)
        (0.0, yp - 0.036, "gold"), (0.0060, yp - 0.0345, "gold"), (0.0068, yp - 0.030, "gold"), (0.0100, yp - 0.0275, "gold"),
        (0.0215, yp - 0.020, "gold"), (0.0265, yp - 0.008, "gold"), (0.0270, yp + 0.002, "gold"), (0.0240, yp + 0.012, "gold"),
        (0.0160, yp + 0.020, "gold"), (0.0098, yp + 0.025, "gold"), (0.0098, yp + 0.029, "gold"),
        (0.0158, -0.1305, "gold"), (0.0158, -0.1245, "gold"),                       # lower ferrule
        (r + 0.0006, -0.1205, "grip"), (r + 0.0006, -0.100, "grip"),
        (r + 0.0014, -0.0955, "gold"), (r + 0.0014, -0.0915, "gold"),                # wire ring (second hand)
        (r + 0.0003, -0.087, "grip"), (r, FIST_SPAN[0], "grip"), (r, 0.0, "grip"), (r, FIST_SPAN[1], "grip"),
        (0.0158, FIST_SPAN[1] + 0.0008, "gold"), (0.0158, 0.0398, "gold"),          # upper ferrule (1 mm in the guard)
        (0.0, 0.0398, "gold")]
    prof = [(a, b) for a, b, c in H]
    mats = []
    for k in range(len(H) - 1):
        a, b = H[k][2], H[k + 1][2]
        mats.append(SLOT["grip"] if (a == "grip" and b == "grip") else SLOT["gold"])
    lathe(prof, (0, 0, 0), (0, 1, 0), (1, 0, 0), 18, mesh=m, mats=mats, metric=True,
          scale=[(1.0, 0.56) if y < -0.1308 else (1.0, 1.0) for rr, y in prof])
    # enamel medallions on the pommel's faces (centre 0.9 mm, edge 0.2 mm into the curved face)
    for sz in (1.0, -1.0):
        z0 = sz * 0.0141
        lathe([(0.0, 0.0), (0.0080, 0.0), (0.0075, 0.0011), (0.0046, 0.0022), (0.0, 0.0026)], (0, yp - 0.003, z0),
              (0, 0, sz), (1, 0, 0), 14, mat=SLOT["enamel"], mesh=m, metric=True)
    return m


def _lathe_scaled(m, profile, O, A, ref, nseg, scales, slot):
    """lathe with a per-profile-point elliptic section (sx along ref, sy along A x ref); poles where r == 0"""
    A = nrm(np.asarray(A, float)); X = nrm(np.asarray(ref, float) - A * np.dot(ref, A)); Y = np.cross(A, X)
    O = np.asarray(O, float)
    f0 = len(m.F)
    rows = []
    pp = np.array(profile, float)
    alp = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(pp, axis=0), axis=1))])
    for (r, h), (sx, sy) in zip(profile, scales):
        if r == 0:
            rows.append([m.add_v(O + A * h)] * nseg)
            continue
        rows.append([m.add_v(O + A * h + r * (math.cos(2 * math.pi * s / nseg) * sx * X +
                                             math.sin(2 * math.pi * s / nseg) * sy * Y)) for s in range(nseg)])
    def ring_u(k):
        R_ = np.array([m.V[i] for i in rows[k]])
        d_ = np.linalg.norm(np.roll(R_, -1, 0) - R_, axis=1)
        return np.concatenate([[0.0], np.cumsum(d_)])
    for k in range(len(rows) - 1):
        for s in range(nseg):
            s2 = (s + 1) % nseg
            a, b, c, d = rows[k][s], rows[k][s2], rows[k + 1][s2], rows[k + 1][s]
            if a == b:
                vs = [a, c, d]
            elif c == d:
                vs = [a, b, c]
            else:
                vs = [a, b, c, d]
            # (2b, G6) metric UVs: the real arc length round each (elliptic) ring, metres along the profile; the pole
            # corner at the u of the ring next to it
            Ua, Ub = ring_u(k), ring_u(k + 1)
            va, vb = alp[k], alp[k + 1]
            if len(vs) == 4:
                uv = [(Ua[s], va), (Ua[s + 1], va), (Ub[s + 1], vb), (Ub[s], vb)]
            elif a == b:
                uv = [(0.5 * (Ub[s] + Ub[s + 1]), va), (Ub[s + 1], vb), (Ub[s], vb)]
            else:
                uv = [(Ua[s], va), (Ua[s + 1], va), (0.5 * (Ua[s] + Ua[s + 1]), vb)]
            m.add_f(vs, uv, SLOT[slot])
    from armour_lower_geo import orient_solid
    orient_solid(m, f0)


# ------------------------------------------------------------------------------------------------ kite shield
def kite_outline_uv():
    """kite outline (U, V) from the shared shield_lion manifest (materials pipeline); fallback: its control points"""
    try:
        p = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets", "textures", "textures_char.json")
        pts = json.load(open(p))["sets"]["shield_lion"]["kite_outline_uv"]
        return np.array(pts, float)
    except Exception:
        top = [(0.0, 0.9635), (0.25, 0.9875), (0.5, 1.0), (0.75, 0.9875), (1.0, 0.9635)]
        right = [(0.995, 0.77), (0.93, 0.53), (0.78, 0.29), (0.60, 0.094), (0.5, 0.0)]
        left = [(1 - u, v) for u, v in reversed(right[:-1])]
        return np.array(top + right + left)


# iteration 2b (user item 24: the shield went through the arm / elbow / vambrace; integrity G1 / G2 / G4 / G5): a
# closed board (face + rolled gold rim + leather back, all welded, no fan), rivets seated 0.5 mm, a forearm strap
# round the vambrace (never through it), and a leather-wrapped HANDLE on the grip contract's socket_hand_l axis
# (radius 10.5 mm) that the left fist closes round, hung from the board by two straps. The board layout (strap
# depth OFF, strap below the top edge Y_C, bulge) was chosen against every clip frame of both bodies' arm plates
# (vambrace / couter / rerebrace / gauntlet cuff clear by >= 8 mm; renders/iter2b_lower/shield_fit.json).
SHIELD_LAYOUT = dict(Y_C=0.14, RX=0.90, RY=4.5, YD0=0.35, T=0.012)
BOARD_NL = 64                  # outline samples (rim / face / back loops)


def shield_face_z(x, yd, L=None):
    """face height (socket frame z) at (x, y down from the top edge); the back at the strap origin is z = 0"""
    L = L or SHIELD_LAYOUT
    b = lambda x_, y_: -(x_ * x_) / (2 * L["RX"]) - ((y_ - L["YD0"]) ** 2) / (2 * L["RY"])
    return L["T"] + b(x, yd) - b(0.0, L["Y_C"])


def _inset(P, d):
    """closed 2-D polygon inset by d along the inward vertex normals (angle bisectors)"""
    n = len(P); out = np.zeros_like(P)
    area = 0.5 * np.sum(P[:, 0] * np.roll(P[:, 1], -1) - np.roll(P[:, 0], -1) * P[:, 1])
    s = 1.0 if area > 0 else -1.0
    for i in range(n):
        a, b, c = P[i - 1], P[i], P[(i + 1) % n]
        e1 = nrm(b - a); e2 = nrm(c - b)
        n1 = s * np.array([-e1[1], e1[0]]); n2 = s * np.array([-e2[1], e2[0]])      # inward normals
        bis = nrm(n1 + n2)
        cs = max(float(bis @ n1), 0.35)
        out[i] = b + bis * d / cs
    return out


def shield(layout=None, handle=None, loop=None):
    """Kite shield in the socket frame (see the module doc). handle = (centre, axis) of the left fist's grip in the
    socket frame (armour_lower.socket_specs: the grip contract's socket_hand_l in the clips' wrist pose), loop =
    (point on the forearm axis, forearm direction, strap radius) for the forearm strap; defaults = the male's."""
    L = layout or SHIELD_LAYOUT
    m = Mesh()
    W, H = SHIELD_W, SHIELD_H
    uv = kite_outline_uv()
    O = _resample_closed(_catmull_closed(uv, 6), BOARD_NL)
    Ot = np.array([((u - 0.5) * W, (1 - v) * H) for u, v in O])          # (x, y down from the top)
    rim = 0.0108                                                         # the rim's width on the face side
    Pin = _inset(Ot, rim)
    cen = Pin.mean(0)
    yc = L["Y_C"]
    to3 = lambda x, yd, z: (x, yd - yc, z)
    zf = lambda x, yd: float(shield_face_z(x, yd, L))
    T = L["T"]
    # --- face: concentric rings from the rim's inner edge to a pole (no slivers at the kite point)
    sc = [1.0, 0.86, 0.72, 0.58, 0.45, 0.33, 0.22, 0.12]
    face = []
    for s_ in sc:
        R2 = cen + (Pin - cen) * s_
        face.append([m.add_v(to3(x, y, zf(x, y))) for x, y in R2])
    pole = m.add_v(to3(cen[0], cen[1], zf(cen[0], cen[1])))
    fuv = lambda x, y: (x / W + 0.5, 1 - y / H)
    for k in range(len(sc) - 1):
        for i in range(BOARD_NL):
            i2 = (i + 1) % BOARD_NL
            vs = [face[k][i], face[k][i2], face[k + 1][i2], face[k + 1][i]]
            m.add_f(vs, [fuv(m.V[v][0], m.V[v][1] + yc) for v in vs], SLOT["shield"])
    for i in range(BOARD_NL):
        i2 = (i + 1) % BOARD_NL
        vs = [face[-1][i], face[-1][i2], pole]
        m.add_f(vs, [fuv(m.V[v][0], m.V[v][1] + yc) for v in vs], SLOT["shield"])
    # --- back: the same loop T below the face (leather), fewer rings
    bsc = [1.0, 0.70, 0.40, 0.15]
    back = []
    for s_ in bsc:
        R2 = cen + (Pin - cen) * s_
        back.append([m.add_v(to3(x, y, zf(x, y) - T)) for x, y in R2])
    bpole = m.add_v(to3(cen[0], cen[1], zf(cen[0], cen[1]) - T))
    for k in range(len(bsc) - 1):
        for i in range(BOARD_NL):
            i2 = (i + 1) % BOARD_NL
            vs = [back[k][i2], back[k][i], back[k + 1][i], back[k + 1][i2]]
            m.add_f(vs, [(m.V[v][0], m.V[v][1]) for v in vs], SLOT["leather"])
    for i in range(BOARD_NL):
        i2 = (i + 1) % BOARD_NL
        vs = [back[-1][i2], back[-1][i], bpole]
        m.add_f(vs, [(m.V[v][0], m.V[v][1]) for v in vs], SLOT["leather"])
    # --- rim: face loop -> lip crest -> outer edge (top, bottom) -> back lip -> back loop (all welded)
    al = _arclen(np.vstack([Ot, Ot[:1]]))
    rings = [face[0]]
    prof = []
    for i in range(BOARD_NL):
        pin = Pin[i]; pout = Ot[i]
        dd = nrm(pout - pin)
        zi = zf(pin[0], pin[1]); zo = zf(pout[0], pout[1])
        prof.append([(pin + dd * 0.0055, zi + 0.0050), (pout + dd * 0.0015, zo + 0.0022),
                     (pout + dd * 0.0015, zo - T - 0.0006), (pout - dd * 0.0045, zo - T - 0.0016)])
    for r in range(4):
        rings.append([m.add_v(to3(prof[i][r][0][0], prof[i][r][0][1], prof[i][r][1])) for i in range(BOARD_NL)])
    rings.append(back[0])
    vpos = [0.0, 0.0078, 0.0148, 0.0290, 0.0352, 0.041]
    for r in range(len(rings) - 1):
        for i in range(BOARD_NL):
            i2 = (i + 1) % BOARD_NL
            vs = [rings[r][i], rings[r + 1][i], rings[r + 1][i2], rings[r][i2]]
            m.add_f(vs, [(al[i], vpos[r]), (al[i], vpos[r + 1]), (al[i + 1], vpos[r + 1]), (al[i + 1], vpos[r])],
                    SLOT["gold"])
    from armour_lower_geo import orient_solid, lathe
    orient_solid(m, 0)
    # --- rim nails on the face 7 mm inside the rim every ~12 cm (steel domes on the face normal, 0.4 mm seated)
    nrv = 18
    Pn = _inset(Ot, rim + 0.007)
    aln = _arclen(np.vstack([Pn, Pn[:1]]))
    for q in range(nrv):
        s_ = aln[-1] * (q + 0.5) / nrv
        i = int(np.searchsorted(aln, s_)) % BOARD_NL
        x, y = Pn[i]
        nf = nrm(np.array([x / L["RX"], (y - L["YD0"]) / L["RY"], 1.0]))
        c = np.array(to3(x, y, zf(x, y)))
        lathe([(0.0, -0.0004), (0.0036, -0.0004), (0.0028, 0.0014), (0.0, 0.0024)], c, nf, (1, 0, 0), 8,
              mat=SLOT["steel"], mesh=m, metric=True)
    # --- forearm strap: a band (28 x 4 mm) round the vambrace, both legs end 0.8 mm inside the board's back
    ctr, fa, rl = loop if loop is not None else (STRAP_LOOP_AT * nrm(np.array([1.0, 1.0, 0.0])) + np.array([0, 0, -STRAP_OFF]),
                                                 nrm(np.array([1.0, 1.0, 0.0])), STRAP_LOOP_R)
    ctr = np.asarray(ctr, float); fa = nrm(np.asarray(fa, float))
    zo = np.array([0.0, 0.0, 1.0]); ac = nrm(np.cross(fa, zo))
    zb_at = lambda p: zf(p[0], p[1] + yc) - T                    # the board back's z over a socket-frame point
    path = []
    top_l = ctr + ac * rl; top_r = ctr - ac * rl
    nl = max(3, int(math.ceil(abs(zb_at(top_l) - ctr[2]) / 0.028)))
    for q in np.linspace(0, 1, nl + 1)[::-1]:                     # leg up to the board on the +ac side
        p = top_l.copy(); p[2] = ctr[2] + (zb_at(top_l) - ctr[2]) * q
        path.append(p)
    for ph in np.linspace(0.0, math.pi, 13)[1:-1]:                # round the far side of the forearm
        path.append(ctr + rl * (math.cos(ph) * ac - math.sin(ph) * zo))
    for q in np.linspace(0, 1, nl + 1):
        p = top_r.copy(); p[2] = ctr[2] + (zb_at(top_r) - ctr[2]) * q
        path.append(p)
    _band(m, path, fa, 0.014, 0.002, "leather")                # (2b: the ends' caps sit 0.5 mm in the board)
    # --- hand grip: a leather-wrapped handle on the fist axis (grip contract radius), two hanger straps to the board
    hc, hd = handle if handle is not None else (HANDLE_OFF, HANDLE_DIR)
    hc = np.asarray(hc, float); hd = nrm(np.asarray(hd, float))
    hr = HANDLE_R
    (s0, s1), (q0, q1) = HANDLE_SPAN, STRAP_AT
    e0 = hc + hd * q0; e1 = hc + hd * q1               # the straps leave beyond the pinky / index edges of the fist
    lathe([(0.0, s0), (hr * 0.75, s0 + 0.0005), (hr, s0 + 0.006), (hr, s1 - 0.006), (hr * 0.75, s1 - 0.0005), (0.0, s1)],
          hc, hd, np.cross(hd, zo) if abs(hd @ zo) < 0.95 else (1, 0, 0), 14, mat=SLOT["leather"], mesh=m, metric=True)
    p_ = nrm(zo - hd * float(zo @ hd))                 # off the handle square to its axis, towards the board
    # (2b) only the end nearer the board is hung: the fist axis runs ~42 deg off the board, so a strap from the far
    # end up to the board swept back along the handle through the closed fingers (G4 gauntlet_l|shield)
    near_end = [e for e in (e0, e1) if float(zb_at(e) - e[2]) <= float(min(zb_at(e0) - e0[2], zb_at(e1) - e1[2])) + 1e-9]
    for e in near_end[:1]:                              # from the handle's surface (0.8 mm in) up to the board
        e_ = e + p_ * (hr - 0.0003)
        k_ = e + p_ * (hr + 0.016)
        top = k_.copy(); top[2] = zb_at(k_)
        n_ = max(3, int(math.ceil(np.linalg.norm(top - k_) / 0.028)))
        pth = [e_] + [k_ + (top - k_) * q for q in np.linspace(0.0, 1.0, n_ + 1)]
        _band(m, pth, hd, STRAP_HW, 0.002, "leather")
    return m


def _band(m, path, across, hw, ht, slot):
    """closed leather band (rectangular section 2 hw x 2 ht) along a polyline, `across` = the band's width axis;
    capped ends"""
    path = [np.asarray(p, float) for p in path]
    secs = []
    for k, p in enumerate(path):
        a_ = path[max(k - 1, 0)]; b_ = path[min(k + 1, len(path) - 1)]
        t = nrm(b_ - a_); w = nrm(across - t * (across @ t)); n_ = nrm(np.cross(t, w))
        secs.append([p + w * hw + n_ * ht, p - w * hw + n_ * ht, p - w * hw - n_ * ht, p + w * hw - n_ * ht])
    loft_solid(m, secs, slot, cap0=path[0] - nrm(path[1] - path[0]) * 0.0005,
               cap1=path[-1] + nrm(path[-1] - path[-2]) * 0.0005)


# the enarmes contract (armour_lower.socket_specs builds socket_shield_l from it): the forearm axis runs STRAP_OFF under
# the board's back at the strap origin, along (sin DIAG, cos DIAG, 0) in the socket frame; the left fist's handle
# (grip contract socket_hand_l, in the clips' wrist pose) is at HANDLE_OFF along HANDLE_DIR (set from the male by
# socket_specs; the female's forearm lands within a few mm of the male's, well inside the strap's clearance)
STRAP_OFF = SHIELD_STRAP_OFF
STRAP_LOOP_FRAC, STRAP_LOOP_R = 0.40, 0.080      # forearm strap: at 40 % of the forearm (elbow -> wrist), radius
STRAP_LOOP_AT = -(0.95 - STRAP_LOOP_FRAC) * 0.276   # (male default, along the forearm from the origin)
# iteration 2b (user item 24, integrity G4 gauntlet_l|shield 60 visible points at bind and in every clip): the left
# fist's index / middle / ring tips close to 6.5 mm (male) / 7.9 mm (female) from the socket_hand_l axis, inside the
# grip contract's 10.5 mm (armour_upper's finger solve; measured on the knight's grip pose), and the fist spans
# -61 .. +64 mm (male; female -65 .. +52) along the axis, over the old hanger straps at +-58 mm: the handle is a 6 mm
# leather rod (back to the contract radius once the fingers respect it) and the straps (12 mm wide) leave beyond both
# edges of the fist
HANDLE_R = 0.0060
HANDLE_SPAN = (-0.086, 0.088)                             # handle ends along its axis (socket frame, from the fist centre)
STRAP_AT, STRAP_HW = (-0.074, 0.076), 0.006                # hanger straps' centre lines along the axis, half width
HANDLE_OFF = np.array([0.0737, 0.0772, -0.1585])          # male defaults (socket_specs fits every body)
HANDLE_DIR = nrm(np.array([0.496, -0.430, -0.755]))


# ------------------------------------------------------------------------------------------------ scabbard
def scabbard():
    m = Mesh()
    L = 0.90
    ys = list(np.linspace(0, L, 22))
    secs = []
    for y in ys:
        t = y / L
        w = 0.030 - 0.017 * t ** 1.3; th = 0.0125 - 0.005 * t
        w = w * (1 - 0.5 * smoothstep(0.93, 1.0, t)); th = th * (1 - 0.4 * smoothstep(0.93, 1.0, t))
        secs.append([(w * math.cos(a), y, th * math.sin(a)) for a in np.linspace(0, 2 * math.pi, 12, endpoint=False)])
    idx = loft(m, secs, "leather")
    fan_cap(m, idx[-1], (0, L + 0.004, 0), "gold")
    fan_cap(m, idx[0], (0, 0.004, 0), "leather", flip=True)
    def band(y0, y1, extra=0.0012, n=12):
        rr = []
        for y in (y0, y0 + 0.002, y1 - 0.002, y1):
            t = min(max(y / L, 0.0), 1.0)
            w = (0.030 - 0.017 * t ** 1.3) * (1 - 0.5 * smoothstep(0.93, 1.0, t)) + (extra if y0 < y < y1 or True else 0)
            th = (0.0125 - 0.005 * t) * (1 - 0.4 * smoothstep(0.93, 1.0, t)) + extra
            e2 = 0.0 if y in (y0, y1) else extra * 0.6
            rr.append([((w + e2) * math.cos(a), y, (th + e2) * math.sin(a)) for a in np.linspace(0, 2 * math.pi, n, endpoint=False)])
        loft(m, rr, "gold")
    band(-0.002, 0.065)                                   # locket
    band(0.30, 0.325)                                     # middle band
    band(0.80, L + 0.002, extra=0.0015)                   # chape
    return m


def build_all(fit=None):
    fit = fit or {}
    return {"sword": sword(), "shield": shield(handle=fit.get("handle"), loop=fit.get("loop")), "scabbard": scabbard()}
