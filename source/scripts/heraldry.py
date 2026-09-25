"""Vector heraldry for the knight kit: lion rampant, fleur-de-lis, compass star and a scroll cartouche, drawn as paths.

Shapes are lists of (op, polygon) with op 'add' (gold) or 'sub' (cut back to the field), in design units, y down.
Paths are written in a subset of SVG path syntax (M L H V C S Q Z, absolute and relative) or generated as tapered
strokes. `raster()` draws them with supersampling (crisp at any size), `to_svg()` writes a vector master.
The motifs follow the user's reference sheet (refs/knight_sheet.png: tabard / cape / shield emblem): a lion rampant
facing the viewer's left inside a heart-shaped scroll frame ending in a fleur finial, a compass star above.
"""
import math, re
import numpy as np
from PIL import Image, ImageDraw

# ------------------------------------------------------------------------------------------------ path basics
_TOK = re.compile(r"[MLHVCSQZmlhvcsqz]|[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")


def _cubic(p0, p1, p2, p3, n):
    t = np.linspace(0, 1, n)[:, None]
    return (1 - t) ** 3 * p0 + 3 * (1 - t) ** 2 * t * p1 + 3 * (1 - t) * t * t * p2 + t ** 3 * p3


def parse_path(d, n=20):
    """SVG path subset -> list of closed / open point arrays (subpaths)."""
    toks = _TOK.findall(d)
    i = 0; cmd = None; cur = np.zeros(2); start = np.zeros(2); last_c = None
    subs = []; pts = []

    def num():
        nonlocal i
        v = float(toks[i]); i += 1; return v

    while i < len(toks):
        t = toks[i]
        if t.isalpha():
            cmd = t; i += 1
            if cmd in "Zz":
                if pts:
                    subs.append(np.array(pts)); pts = []
                cur = start.copy(); last_c = None
                continue
        rel = cmd.islower(); c = cmd.upper()
        base = cur if rel else np.zeros(2)
        if c == "M":
            if pts:
                subs.append(np.array(pts))
            cur = base + np.array([num(), num()]); start = cur.copy(); pts = [cur.copy()]
            cmd = "l" if rel else "L"; last_c = None
        elif c == "L":
            cur = base + np.array([num(), num()]); pts.append(cur.copy()); last_c = None
        elif c == "H":
            cur = np.array([num() + (cur[0] if rel else 0), cur[1]]); pts.append(cur.copy()); last_c = None
        elif c == "V":
            cur = np.array([cur[0], num() + (cur[1] if rel else 0)]); pts.append(cur.copy()); last_c = None
        elif c == "C":
            p1 = base + np.array([num(), num()]); p2 = base + np.array([num(), num()]); p3 = base + np.array([num(), num()])
            pts.extend(_cubic(cur, p1, p2, p3, n)[1:]); cur = p3; last_c = p2
        elif c == "S":
            p1 = 2 * cur - last_c if last_c is not None else cur.copy()
            p2 = base + np.array([num(), num()]); p3 = base + np.array([num(), num()])
            pts.extend(_cubic(cur, p1, p2, p3, n)[1:]); cur = p3; last_c = p2
        elif c == "Q":
            q = base + np.array([num(), num()]); p3 = base + np.array([num(), num()])
            p1 = cur + 2 / 3 * (q - cur); p2 = p3 + 2 / 3 * (q - p3)
            pts.extend(_cubic(cur, p1, p2, p3, n)[1:]); cur = p3; last_c = None
        else:
            raise ValueError(f"unsupported path command {cmd}")
    if pts:
        subs.append(np.array(pts))
    return subs


def catmull(points, n=12, closed=False, alpha=0.5):
    """Centripetal Catmull-Rom through points -> dense polyline."""
    P = np.asarray(points, float)
    if closed:
        P = np.vstack([P[-1], P, P[0], P[1]])
    else:
        P = np.vstack([2 * P[0] - P[1], P, 2 * P[-1] - P[-2]])
    out = []
    for k in range(1, len(P) - 2):
        p0, p1, p2, p3 = P[k - 1], P[k], P[k + 1], P[k + 2]
        t0 = 0.0
        t1 = t0 + max(np.linalg.norm(p1 - p0), 1e-6) ** alpha
        t2 = t1 + max(np.linalg.norm(p2 - p1), 1e-6) ** alpha
        t3 = t2 + max(np.linalg.norm(p3 - p2), 1e-6) ** alpha
        ts = np.linspace(t1, t2, n, endpoint=False)[:, None]
        A1 = (t1 - ts) / (t1 - t0) * p0 + (ts - t0) / (t1 - t0) * p1
        A2 = (t2 - ts) / (t2 - t1) * p1 + (ts - t1) / (t2 - t1) * p2
        A3 = (t3 - ts) / (t3 - t2) * p2 + (ts - t2) / (t3 - t2) * p3
        B1 = (t2 - ts) / (t2 - t0) * A1 + (ts - t0) / (t2 - t0) * A2
        B2 = (t3 - ts) / (t3 - t1) * A2 + (ts - t1) / (t3 - t1) * A3
        out.append((t2 - ts) / (t2 - t1) * B1 + (ts - t1) / (t2 - t1) * B2)
    if not closed:
        out.append(P[-2][None])
    return np.vstack(out)


def resample(pl, step):
    seg = np.linalg.norm(np.diff(pl, axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(seg)])
    m = max(2, int(s[-1] / step) + 1)
    ss_ = np.linspace(0, s[-1], m)
    return np.stack([np.interp(ss_, s, pl[:, 0]), np.interp(ss_, s, pl[:, 1])], -1)


def taper(center, w0, w1=0.0, prof=1.0, cap0="round", cap1="point", wfun=None):
    """Variable-width stroke polygon along a polyline. Width goes w0 -> w1 (power `prof`) or wfun(t)."""
    c = resample(np.asarray(center, float), 2.0)
    d = np.gradient(c, axis=0)
    d /= np.linalg.norm(d, axis=1, keepdims=True) + 1e-9
    nrm = np.stack([-d[:, 1], d[:, 0]], -1)
    t = np.linspace(0, 1, len(c))
    w = wfun(t) if wfun else w0 + (w1 - w0) * t ** prof
    L = c + nrm * (w[:, None] / 2); R = c - nrm * (w[:, None] / 2)
    poly = [L]
    if cap1 == "round" and w[-1] > 0.5:
        a0 = math.atan2(nrm[-1, 1], nrm[-1, 0])
        ang = np.linspace(a0, a0 - math.pi, 12)
        poly.append(c[-1] + np.stack([np.cos(ang), np.sin(ang)], -1) * w[-1] / 2)
    poly.append(R[::-1])
    if cap0 == "round" and w[0] > 0.5:
        a0 = math.atan2(-nrm[0, 1], -nrm[0, 0])
        ang = np.linspace(a0, a0 - math.pi, 12)
        poly.append(c[0] + np.stack([np.cos(ang), np.sin(ang)], -1) * w[0] / 2)
    return np.vstack(poly)


def lock(p0, p1, bend=0.25, w=30, prof=0.9, curl=0.0, wfun=None):
    """Flame / lock: tapered, curved from p0 (base, width w) to p1 (point). bend = sideways bow (fraction of length),
    curl = extra hook at the tip (fraction)."""
    p0 = np.asarray(p0, float); p1 = np.asarray(p1, float)
    d = p1 - p0; L = np.linalg.norm(d); n = np.array([-d[1], d[0]]) / (L + 1e-9)
    c1 = p0 + d * 0.33 + n * bend * L
    c2 = p0 + d * 0.72 + n * (bend + curl) * L * 0.9
    tip = p1 + n * curl * L * 0.35
    cl = _cubic(p0, c1, c2, tip, 40)
    return taper(cl, w, 0.0, prof, cap0="round", wfun=wfun)


def spiral(center, r0, r1, a0, turns, w0, w1, cw=True, n=120):
    """Archimedean-ish spiral from radius r0 (angle a0) to r1 after `turns` turns (volute of a scroll)."""
    t = np.linspace(0, 1, n)
    a = a0 + (1 if cw else -1) * t * turns * 2 * math.pi
    r = r0 + (r1 - r0) * t
    pts = np.stack([center[0] + r * np.cos(a), center[1] + r * np.sin(a)], -1)
    return taper(pts, w0, w1, 1.0, cap0="round", cap1="round")


def ellipse(cx, cy, rx, ry, rot=0.0, n=64):
    a = np.linspace(0, 2 * math.pi, n, endpoint=False)
    x = rx * np.cos(a); y = ry * np.sin(a)
    c, s = math.cos(rot), math.sin(rot)
    return np.stack([cx + c * x - s * y, cy + s * x + c * y], -1)


def xf(shapes, sx=1.0, sy=None, tx=0.0, ty=0.0, rot=0.0, mirror_x=None):
    """Transform shapes: optional mirror about x = mirror_x, then scale, rotate (about origin), translate."""
    sy = sx if sy is None else sy
    c, s = math.cos(rot), math.sin(rot)
    out = []
    for op, p in shapes:
        q = np.array(p, float)
        if mirror_x is not None:
            q[:, 0] = 2 * mirror_x - q[:, 0]
        q = q * [sx, sy]
        q = np.stack([c * q[:, 0] - s * q[:, 1], s * q[:, 0] + c * q[:, 1]], -1) + [tx, ty]
        out.append((op, q))
    return out


def bbox(shapes):
    allp = np.vstack([p for op, p in shapes if op == "add"])
    return allp.min(0), allp.max(0)


def fit(shapes, cx, cy, w=None, h=None):
    """Scale uniformly to fit width w and/or height h, centred at (cx, cy)."""
    lo, hi = bbox(shapes)
    sz = hi - lo
    k = min([v for v in ((w / sz[0]) if w else None, (h / sz[1]) if h else None) if v is not None])
    mid = (lo + hi) / 2
    return xf(xf(shapes, tx=-mid[0], ty=-mid[1]), sx=k, tx=cx, ty=cy)


def P(d, op="add", n=20):
    return [(op, p) for p in parse_path(d, n)]


# ------------------------------------------------------------------------------------------------ raster / svg
def raster(shapes, W, H, ss=4, to_px=None):
    """Draw shapes into a (H, W) coverage map (anti-aliased by ss x ss supersampling).
    to_px(points) -> pixel coords (default identity). Returns (coverage float32 HxW, hi-res bool mask)."""
    im = Image.new("L", (W * ss, H * ss), 0)
    d = ImageDraw.Draw(im)
    for op, p in shapes:
        q = to_px(p) if to_px else p
        q = np.asarray(q) * ss
        if len(q) < 3:
            continue
        d.polygon([tuple(v) for v in q], fill=255 if op == "add" else 0)
    hi = np.asarray(im) > 127
    cov = np.asarray(im.resize((W, H), Image.BOX), np.float32) / 255.0
    return cov, hi


def to_svg(shapes, path, W, H, fg="#d9a53f", bg="#1c3478", view=None):
    lo, hi = view if view else bbox(shapes)
    vb = f"{lo[0]:.1f} {lo[1]:.1f} {hi[0] - lo[0]:.1f} {hi[1] - lo[1]:.1f}"
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="{vb}">',
           f'<rect x="{lo[0]}" y="{lo[1]}" width="{hi[0] - lo[0]}" height="{hi[1] - lo[1]}" fill="{bg}"/>']
    for op, p in shapes:
        dd = "M" + " L".join(f"{x:.2f},{y:.2f}" for x, y in p) + " Z"
        out.append(f'<path d="{dd}" fill="{fg if op == "add" else bg}"/>')
    out.append("</svg>")
    open(path, "w").write("\n".join(out))


# ================================================================================================ motifs
def star8(r_long=100, r_short=50, r_in=24, concave=0.2):
    """Compass star: 4 long points (N E S W) and 4 short diagonal points, slightly concave flanks."""
    pts = []
    for k in range(16):
        a = -math.pi / 2 + k * math.pi / 8
        if k % 4 == 0:
            r = r_long
        elif k % 2 == 0:
            r = r_short
        else:
            r = r_in
        pts.append((r * math.cos(a), r * math.sin(a)))
    # concave flanks: replace straight edges with slight inward curves
    poly = []
    for i in range(16):
        a = np.array(pts[i]); b = np.array(pts[(i + 1) % 16])
        mid = (a + b) / 2
        poly.extend(_cubic(a, a + (mid - a) * 0.9 * (1 - concave), b + (mid - b) * 0.9 * (1 - concave), b, 10)[:-1])
    shapes = [("add", np.array(poly))]
    shapes.append(("add", ellipse(0, 0, r_in * 1.25, r_in * 1.25)))
    return shapes


def fleur_de_lis():
    """Heraldic fleur-de-lis, ~ 290 wide x 300 tall, centred on x = 0, band at y ~ 20..46 (y down)."""
    s = []
    # central petal: pointed top, broad belly, pinched into the band
    s.append(("add", outline([(0, -160), (22, -126), (48, -84), (56, -40), (46, -4), (22, 24), (-22, 24), (-46, -4),
                              (-56, -40), (-48, -84), (-22, -126), (0, -160)])))
    # side petal (right): rises out of the band, sweeps up and out, rolls over into a drooping bulb
    petal = outline(
        [(16, 26), (32, -8), (62, -44), (104, -62), (142, -54), (162, -24), (160, 12), (144, 40)],
        [(144, 40), (130, 52), (112, 50), (104, 36), (110, 22), (124, 18)],
        [(124, 18), (128, 0), (120, -20), (100, -26), (76, -16), (56, 6), (44, 30)],
        [(44, 30), (30, 34), (16, 26)])
    s.append(("add", petal)); s += xf([("add", petal)], mirror_x=0.0)
    # band
    s.append(("add", outline([(-70, 20), (0, 12), (70, 20)], [(70, 20), (74, 34), (72, 48)], [(72, 48), (0, 40), (-72, 48)],
                             [(-72, 48), (-74, 34), (-70, 20)])))
    # lower part: short pointed stem and two small curled feet
    s.append(("add", outline([(-16, 46), (16, 46)], [(16, 46), (14, 80), (0, 122)], [(0, 122), (-14, 80), (-16, 46)])))
    foot = outline([(14, 46), (34, 64), (62, 72), (90, 66), (104, 50)], [(104, 50), (106, 72), (92, 92), (70, 98)],
                   [(70, 98), (80, 88), (82, 78)], [(82, 78), (58, 82), (34, 74), (14, 60)])
    s.append(("add", foot)); s += xf([("add", foot)], mirror_x=0.0)
    # incised midrib + petal lines
    s.append(("sub", taper(catmull([(0, -118), (0, -60), (0, -4)], n=8), 1.0, 7.0, 0.7, cap0="point", cap1="round")))
    pl = taper(catmull([(40, 8), (70, -26), (110, -40), (138, -30)], n=8), 6, 1, 1.0, cap1="point")
    s.append(("sub", pl)); s += xf([("sub", pl)], mirror_x=0.0)
    return s


def sprig():
    """Small fleurette on a short stem with two leaves (the ornaments flanking the hem fleur-de-lis)."""
    s = xf(fleur_de_lis(), sx=0.55, tx=0, ty=-70)
    s.append(("add", taper(catmull([(0, -40), (2, 10), (-4, 60), (2, 110)], n=10), 14, 5, 1.0, cap1="point")))
    s.append(("add", lock((0, 52), (-46, 0), bend=0.3, w=22, curl=0.25)))
    s.append(("add", lock((0, 74), (44, 24), bend=-0.3, w=20, curl=-0.25)))
    return s


def limb(points, widths, n=14, cap0="round", cap1="round"):
    """Smooth limb through points with widths interpolated (by arc length) at the points."""
    pts = np.asarray(points, float)
    c = catmull(pts, n=n)
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1); sk = np.concatenate([[0], np.cumsum(seg)]) / max(seg.sum(), 1e-9)
    wf = lambda t: np.interp(t, sk, widths)
    return taper(c, 0, 0, wfun=wf, cap0=cap0, cap1=cap1)


def ring(poly, gap):
    """A 'sub' band of width 2*gap along a closed outline (painter trick: sub ring, then add the shape = a gap
    separating the shape from what lies under it)."""
    q = np.vstack([poly, poly[:1]])
    return ("sub", taper(q, 2 * gap, 2 * gap, cap0="round", cap1="round"))


def with_gap(poly, gap):
    return [ring(poly, gap), ("add", poly)] if gap else [("add", poly)]


def paw(c, ang_deg, r, claws=3, spread=34, claw_len=1.0, hook=0.35, gap=0):
    """Heraldic paw: pad + toes + hooked claws, pointing along ang (deg, 0 = +x, 90 = down)."""
    a = math.radians(ang_deg); d = np.array([math.cos(a), math.sin(a)]); c = np.asarray(c, float)
    out = []
    pad = ellipse(c[0] - d[0] * r * 0.15, c[1] - d[1] * r * 0.15, r * 1.25, r * 0.85, a)
    out += with_gap(pad, gap)
    for k in range(claws):
        ak = a + math.radians((k - (claws - 1) / 2) * spread)
        dk = np.array([math.cos(ak), math.sin(ak)])
        toe = c + dk * r * 0.9
        out.append(("add", ellipse(toe[0], toe[1], r * 0.46, r * 0.38, ak)))
        base = toe + dk * r * 0.15
        tip = base + dk * r * 1.25 * claw_len
        out.append(("add", lock(base, tip, bend=-0.24, w=r * 0.48, prof=1.15, curl=hook)))
    for k in range(claws - 1):                              # toe separations
        ak = a + math.radians((k + 0.5 - (claws - 1) / 2) * spread)
        dk = np.array([math.cos(ak), math.sin(ak)])
        out.append(("sub", taper(np.array([c + dk * r * 0.55, c + dk * r * 1.35]), 1.5, r * 0.16, 1.0,
                                 cap0="point", cap1="round")))
    return out


def flame_mass(base_pts, tip_len, n_locks, lean=0.75, inner=60, hook=0.30, bulge=0.35, cut_len=0.55, cut_w=7,
               center=None, sweep=0.25):
    """A mass of flame locks (mane / tuft): the notches lie on the base curve, each lock bulges outward, leans along
    the curve (lean = tip offset in lock widths) and hooks back. Returns [('add', outline), ('sub', incisions)...].
    base_pts: points of the base curve (ordered in the flow direction); tip_len: length(s) of the locks; the locks
    grow away from `center` (default: the centroid of the base curve)."""
    B = resample(catmull(base_pts, n=16), 1.0)
    k = len(B)
    tan = np.gradient(B, axis=0); tan /= np.linalg.norm(tan, axis=1, keepdims=True)
    out = np.stack([tan[:, 1], -tan[:, 0]], -1)
    ctr = np.asarray(center, float) if center is not None else B.mean(0)
    if np.sum(np.einsum("ij,ij->i", out, B - ctr)) < 0:     # make the normal point away from the centre
        out = -out
    L = np.broadcast_to(np.asarray(tip_len, float), (n_locks,)) if np.ndim(tip_len) == 0 else np.asarray(tip_len, float)
    idx = lambda u: min(k - 1, max(0, int(round(u * (k - 1)))))
    outline = []
    cuts = []
    for i in range(n_locks):
        u0 = i / n_locks; u1 = (i + 1) / n_locks; ut = min(1.0, (i + lean) / n_locks + 0.5 / n_locks)
        a = B[idx(u0)]; b = B[idx(u1)]; tb = B[idx(ut)]
        on = out[idx(ut)]; Li = L[i]
        tip = tb + on * Li * (1 - sweep * 0.45) + tan[idx(ut)] * Li * sweep
        # convex leading edge a -> tip, concave trailing edge tip -> b (hooked)
        c1 = a + out[idx(u0)] * Li * bulge * 1.4
        c2 = tip - tan[idx(ut)] * Li * 0.35 - on * Li * 0.05
        c3 = tip + (b - tip) * 0.25 - on * Li * hook * 0.2 + tan[idx(ut)] * Li * 0.05
        c4 = b + out[idx(u1)] * Li * 0.30
        seg1 = _cubic(a, c1, c2, tip, 24); seg2 = _cubic(tip, c3, c4, b, 24)
        outline.append(seg1[:-1]); outline.append(seg2[:-1])
        if i > 0 and cut_len > 0:
            # incision from the notch into the mass, curving against the flow
            p0 = a + out[idx(u0)] * 2
            ui = max(0.0, u0 - 0.6 / n_locks)
            p2 = B[idx(ui)] - out[idx(ui)] * inner * cut_len
            p1 = (p0 + p2) / 2 + out[idx(u0)] * 6
            cuts.append(("sub", taper(_cubic(p0, p1, p1, p2, 16), cut_w, 0.5, 1.2, cap0="round", cap1="point")))
        # a short inner incision along the lock (flame detail)
        q0 = tip - on * Li * 0.18 - tan[idx(ut)] * Li * 0.05
        q2 = (a + b) / 2 + out[idx((u0 + u1) / 2)] * Li * 0.22
        cuts.append(("sub", taper(_cubic(q0, q0 + (q2 - q0) * 0.4 + on * 3, q2, q2, 12), 0.5, cut_w * 0.7, 1.0,
                                  cap0="point", cap1="point")))
    outline.append(B[-1:])
    # close through the inside (offset inward so the mass overlaps what it grows from)
    back = (B - out * inner)[::-1]
    poly = np.vstack(outline + [back])
    return [("add", poly)] + cuts


def outline(*segs, n=10):
    """Closed outline from smooth segments meeting at corners: each segment is a list of points (Catmull-Rom
    through them); consecutive segments share their end / start point, which becomes a sharp corner."""
    out = []
    for sg in segs:
        c = catmull(np.asarray(sg, float), n=n)
        out.append(c[:-1])
    return np.vstack(out)


def lion_rampant(gap=7, parts=False):
    """Lion rampant facing the viewer's left (design box ~ 90..960 x 80..1180, y down). Flat heraldic drawing:
    tail, far foreleg, body silhouette (head + mane + back + standing leg), near hind leg, near foreleg (each
    separated by a thin gap), then incised details."""
    G = {}
    # ---- tail (behind), S-curve up the right side, tufted end curling back
    tail = catmull([(690, 800), (770, 838), (842, 812), (882, 730), (874, 630), (828, 540), (820, 462), (846, 400),
                    (884, 364)], n=16)
    G["tail"] = [("add", taper(tail, 46, 30, 1.0, cap0="round", cap1="round"))]
    tuft = outline(
        [(866, 392), (842, 356), (838, 312), (852, 270)],            # left lock, leading edge
        [(852, 270), (860, 300), (872, 322)],                        # its trailing edge
        [(872, 322), (872, 282), (888, 242), (916, 212)],            # middle lock
        [(916, 212), (912, 250), (912, 286)],
        [(912, 286), (930, 262), (960, 246), (994, 244)],            # right lock
        [(994, 244), (966, 268), (948, 300), (944, 332)],
        [(944, 332), (962, 330), (984, 342), (1000, 364)],           # lower right lock
        [(1000, 364), (966, 364), (938, 372), (912, 392), (888, 404)],
        [(888, 404), (866, 392)])
    G["tail"] += [("add", tuft)]
    G["tail"] += [("sub", taper(catmull([(874, 350), (880, 304), (900, 262)], n=8), 1, 6, 1.0, cap0="point")),
                  ("sub", taper(catmull([(900, 350), (924, 316), (956, 290)], n=8), 1, 6, 1.0, cap0="point"))]
    # ---- far (lower) foreleg, behind the body
    far_fore = outline(
        [(446, 540), (390, 548), (340, 566), (296, 552), (244, 526), (196, 500), (168, 488)],
        [(168, 488), (152, 506), (158, 528)],
        [(158, 528), (196, 546), (250, 580), (300, 610)],
        [(300, 610), (310, 636), (330, 662)],                         # elbow tuft
        [(330, 662), (338, 640), (352, 628)],
        [(352, 628), (400, 628), (446, 618), (470, 580), (446, 540)])
    G["far_fore"] = [("add", far_fore)] + paw((140, 498), -172, 34, hook=0.35)
    # ---- body: head + mane + back + standing hind leg + belly + chest, one silhouette
    body = outline(
        [(136, 236), (140, 212), (170, 196), (214, 186), (240, 166), (274, 146), (316, 136), (348, 140)],
        [(348, 140), (360, 112), (378, 84)],                          # ear
        [(378, 84), (390, 120), (398, 152)],
        # mane locks: (notch -> tip) convex leading edge, (tip -> notch) concave trailing edge
        [(398, 152), (430, 118), (480, 96), (536, 90)], [(536, 90), (502, 116), (474, 144), (462, 172)],
        [(462, 172), (512, 148), (574, 148), (628, 166)], [(628, 166), (584, 178), (546, 198), (522, 228)],
        [(522, 228), (574, 218), (636, 236), (680, 274)], [(680, 274), (632, 272), (584, 284), (556, 306)],
        [(556, 306), (608, 310), (660, 344), (690, 394)], [(690, 394), (644, 380), (600, 380), (568, 396)],
        [(568, 396), (618, 414), (658, 458), (672, 512)], [(672, 512), (634, 488), (598, 478), (574, 488)],
        [(574, 488), (614, 518), (640, 564), (644, 616)], [(644, 616), (614, 584), (588, 570), (566, 574)],
        # back, rump, back of the standing leg, hock tuft, foot
        [(566, 574), (604, 626), (650, 694), (692, 762), (714, 816), (716, 864), (704, 920), (684, 966), (692, 1006)],
        [(692, 1006), (722, 1016), (752, 1036)],                      # hock tuft
        [(752, 1036), (722, 1040), (694, 1046), (676, 1090), (656, 1128), (642, 1150)],
        # paw on the ground, claws pointing left
        [(642, 1150), (612, 1166), (584, 1172)],
        [(584, 1172), (560, 1178), (530, 1180)], [(530, 1180), (552, 1166), (572, 1160)],
        [(572, 1160), (546, 1158), (514, 1150)], [(514, 1150), (540, 1140), (566, 1142)],
        [(566, 1142), (544, 1130), (522, 1114)], [(522, 1114), (552, 1114), (586, 1120)],
        [(586, 1120), (606, 1098), (624, 1060), (636, 1022), (628, 988), (608, 950), (600, 910), (598, 872)],
        [(598, 872), (584, 840), (564, 812)],
        # belly, chest, throat
        [(564, 812), (540, 770), (516, 722), (490, 682), (452, 650), (404, 624), (368, 604), (338, 572), (318, 520),
         (310, 460), (316, 400), (334, 352)],
        # jaw, open mouth, upper jaw, nose
        [(334, 352), (302, 362), (276, 366)],
        [(276, 366), (264, 384), (252, 402)], [(252, 402), (248, 382), (240, 368)],          # beard tufts
        [(240, 368), (224, 382), (208, 394)], [(208, 394), (204, 372), (194, 354)],
        [(194, 354), (178, 338), (166, 318)],                                                  # lower jaw tip
        [(166, 318), (196, 308), (222, 298), (248, 290)],                                      # mouth corner
        [(248, 290), (212, 284), (176, 280), (148, 276)],                                      # upper lip tip
        [(148, 276), (134, 258), (136, 236)])
    G["body"] = with_gap(body, 0)
    G["body"] += [("add", lock((240, 292), (102, 314), bend=-0.22, w=22, prof=0.8, curl=0.32)),     # tongue
                  ("add", lock((174, 280), (178, 302), bend=0.1, w=13, prof=1.0)),                  # fangs
                  ("add", lock((190, 314), (192, 294), bend=-0.1, w=11, prof=1.0))]
    # ---- near (raised) hind leg
    near_hind = outline(
        [(612, 766), (560, 768), (500, 788), (452, 812), (420, 842), (410, 882), (412, 922), (406, 952),
         (380, 968), (346, 980), (322, 986)],
        [(322, 986), (322, 1000), (326, 1014)],
        [(326, 1014), (360, 1012), (394, 1002), (428, 990)],
        [(428, 990), (452, 996), (478, 1008)],                        # hock tuft
        [(478, 1008), (462, 986), (452, 970), (460, 930), (472, 890), (494, 866), (540, 852), (600, 848), (652, 832),
         (656, 792), (612, 766)])
    G["near_hind"] = with_gap(near_hind, gap) + paw((306, 1000), 178, 34, hook=0.3)
    # ---- near (upper) foreleg, raised in front of the face
    near_fore = outline(
        [(470, 400), (404, 398), (346, 404), (296, 386), (244, 356), (196, 330), (166, 314)],
        [(166, 314), (142, 332), (140, 366)],
        [(140, 366), (178, 392), (222, 426), (268, 456), (302, 474)],
        [(302, 474), (316, 502), (338, 532)],                         # elbow tuft
        [(338, 532), (346, 506), (362, 494), (402, 506), (452, 510), (486, 470), (470, 400)])
    # the raised foreleg grows out of the chest: a gap only where the forearm crosses in front of the body, and an
    # incised line along the underside of the upper arm (no box outline on the shoulder)
    G["near_fore"] = [("add", near_fore)] + paw((120, 344), -156, 36, hook=0.35)
    G["near_fore"] += [("sub", taper(catmull([(468, 512), (420, 512), (372, 500), (330, 476), (290, 452)], n=10), 3, 8, 1.0,
                                     wfun=lambda t: 2 + 6 * np.sin(np.pi * t))),
                       ("sub", taper(catmull([(326, 408), (300, 424), (286, 452)], n=8), 2, 6, 1.0,
                                     wfun=lambda t: 1 + 5 * np.sin(np.pi * t)))]
    # ---- incised details
    cut = []
    cut.append(("sub", lock((292, 214), (252, 222), bend=0.3, w=15, prof=0.7)))                      # eye
    cut.append(("sub", taper(catmull([(240, 196), (272, 182), (312, 190)], n=8), 3, 7, 1.0, cap1="round")))  # brow
    cut.append(("sub", ellipse(152, 222, 7, 5, 0.3)))                                                    # nostril
    cut.append(("sub", taper(catmull([(156, 256), (198, 262), (238, 272)], n=8), 3, 1, 1.0, cap1="point")))  # lip
    cut.append(("sub", lock((362, 150), (374, 108), bend=-0.2, w=12, prof=0.9)))                         # inner ear
    # cheek ruff: the boundary between the face and the mane
    cut.append(("sub", taper(catmull([(398, 156), (424, 212), (424, 276), (398, 326), (352, 352)], n=10), 3, 8, 1.0,
                             wfun=lambda t: 3 + 5 * np.sin(np.pi * t))))
    # mane incisions from the notches, curving in
    for a, b, c in (((462, 172), (440, 204), (430, 240)), ((522, 228), (492, 262), (474, 296)),
                    ((556, 306), (524, 340), (500, 372)), ((568, 396), (530, 430), (504, 460)),
                    ((574, 488), (540, 516), (512, 536))):
        cut.append(("sub", taper(catmull([a, b, c], n=8), 7, 1, 1.0, cap1="point")))
    # flame lines inside the locks
    for a, b, c in (((500, 110), (470, 140), (452, 150)), ((580, 160), (536, 176), (510, 196)),
                    ((630, 250), (586, 256), (556, 270)), ((648, 360), (606, 350), (578, 356)),
                    ((640, 466), (608, 450), (584, 452))):
        cut.append(("sub", taper(catmull([a, b, c], n=8), 1, 6, 1.0, cap0="point", cap1="point",
                                 wfun=lambda t: 1 + 5 * np.sin(np.pi * t))))
    # chest locks
    for a, b, c in (((340, 380), (352, 430), (346, 480)), ((372, 420), (388, 470), (384, 520))):
        cut.append(("sub", taper(catmull([a, b, c], n=8), 1, 6, 1.0, wfun=lambda t: 1 + 5 * np.sin(np.pi * t))))
    cut.append(("sub", taper(catmull([(600, 640), (640, 700), (664, 770)], n=8), 1, 7, 1.0,
                             wfun=lambda t: 1 + 6 * np.sin(np.pi * t))))                                 # flank
    cut.append(("sub", taper(catmull([(660, 880), (650, 950), (670, 1000)], n=8), 1, 6, 1.0,
                             wfun=lambda t: 1 + 5 * np.sin(np.pi * t))))                                 # standing thigh
    cut.append(("sub", taper(catmull([(560, 800), (510, 830), (470, 870)], n=8), 1, 6, 1.0,
                             wfun=lambda t: 1 + 5 * np.sin(np.pi * t))))                                 # raised thigh
    G["details"] = cut
    order = ["tail", "far_fore", "body", "near_hind", "near_fore", "details"]
    if parts:
        return {k: G[k] for k in order}
    return [sh for k in order for sh in G[k]]


def swell(center, w_mid, w0, w1, n=16, bias=0.45):
    """Calligraphic stroke through points: thin ends, swelling to w_mid (peak at `bias` along the stroke)."""
    c = catmull(np.asarray(center, float), n=n)
    def wf(t):
        up = np.clip(t / bias, 0, 1); dn = np.clip((1 - t) / (1 - bias), 0, 1)
        return np.where(t < bias, w0 + (w_mid - w0) * np.sin(up * np.pi / 2) ** 0.8,
                        w1 + (w_mid - w1) * np.sin(dn * np.pi / 2) ** 0.8)
    return taper(c, 0, 0, wfun=wf, cap0="round", cap1="round")


def leaf(p, dv, L, w, lobes=3, flip=False):
    """Acanthus leaf growing from p along dv: a curved blade with pointed lobes on its upper side sweeping to the
    tip, and a rolled-over tip."""
    p = np.asarray(p, float); dv = np.asarray(dv, float) / np.linalg.norm(dv)
    nv = np.array([dv[1], -dv[0]]) * (-1 if flip else 1)                 # 'upper' side
    base = [p, p + dv * L * 0.35 + nv * L * 0.10, p + dv * L * 0.72 + nv * L * 0.08, p + dv * L * 0.98 - nv * L * 0.06]
    lens = [w * 0.85, w * 0.75, w * 0.55, w * 0.4][:lobes]
    sh = flame_mass(base, lens, lobes, inner=w * 0.42, lean=0.9, hook=0.45, bulge=0.3, sweep=0.7,
                    center=p - nv * L, cut_len=0.35, cut_w=4)
    tip = base[-1]
    sh.append(("add", lock(tip - dv * w * 0.2, tip + dv * w * 0.5 - nv * w * 0.5, bend=-0.4, w=w * 0.34, curl=-0.5)))
    return sh


def cartouche(stem=54):
    """Heart / lyre-shaped scroll frame after the reference emblem: each half is a swelling scroll rising from the
    bottom point up the side (bulging outward, acanthus leaves on the outside) and rolling OUTWARD into a volute at
    the top corner. Centre x = 0, top y ~ 0, bottom ~ 860."""
    half = []
    # main scroll, from the bottom point up to the top corner, then the volute rolls outward and down
    main = [(26, 770), (100, 690), (196, 590), (282, 460), (330, 320), (336, 190), (318, 100), (330, 40), (380, 14),
            (424, 34)]
    half.append(("add", swell(main, stem, 16, 26, bias=0.55)))
    half.append(("add", spiral((402, 78), 44, 8, math.radians(-80), 1.1, 26, 10, cw=True)))
    # acanthus leaves on the outside, curling out and down
    mm = resample(catmull(np.asarray(main, float), n=16), 1.0); km = len(mm)
    tan = np.gradient(mm, axis=0); tan /= np.linalg.norm(tan, axis=1, keepdims=True)
    for u, ang, L, w in ((0.24, -20, 92, 40), (0.42, -40, 100, 42), (0.60, -58, 86, 38)):
        i = int(u * (km - 1)); p = mm[i]
        outn = np.array([tan[i, 1], -tan[i, 0]]) if tan[i, 1] * 1 + 0 > -2 else None
        outn = outn if outn[0] > 0 else -outn                       # outward = away from the centre line
        p = p + outn * stem * 0.3
        a = math.radians(ang); dv = np.array([math.cos(a), math.sin(a)])
        half += leaf(p, dv, L * 1.35, w * 1.25)
    # inner curls (toward the lion) near the top and the bottom
    half.append(("add", spiral((270, 150), 32, 6, math.radians(-10), 1.0, 18, 8, cw=False)))
    half.append(("add", spiral((120, 600), 30, 6, math.radians(-60), 1.0, 16, 8, cw=False)))
    # incised centre line along the stem (double-line scroll)
    half.append(("sub", taper(catmull(np.asarray(main[1:8], float), n=12), 2, 2, 1.0,
                              wfun=lambda t: 1.0 + 7.0 * np.sin(np.pi * t) ** 1.5, cap0="point", cap1="point")))
    s = half + xf(half, mirror_x=0.0)
    # bottom finial: the stems meet in a knot, a small fleur hangs below
    s.append(("add", ellipse(0, 772, 34, 24)))
    s += xf(fleur_de_lis(), sx=0.40, rot=math.pi, tx=0, ty=842)
    return s


def emblem(with_star=True):
    """Cartouche + lion (+ star above), design units; returns shapes (the lion is scaled into the frame)."""
    fr = cartouche()
    lion = fit(lion_rampant(), 6, 418, w=None, h=600)
    s = fr + lion
    if with_star:
        s += xf(star8(), sx=0.9, tx=0, ty=-100)
    return s
