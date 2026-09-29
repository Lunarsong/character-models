"""Procedural texture sets for the knight's upper armour (system python3: numpy, scipy, Pillow). All original / CC0.

Writes assets/textures/knight_*.png (see armour_upper_tex.py for the trim-sheet layout):
  knight_armour_{base,orm,normal}.png  2048^2 trim sheet: steel plate, embossed gold lion rampant, rosette, rivet,
                                       leather, dark; engraved gold scroll bands, rivet band, roped beads, crest,
                                       etched border band (gold rinceau inlaid in steel), inner steel
  knight_mail_{base,orm,normal}.png    1024^2 tiling riveted 4-in-1 mail (1 UV = 0.10 m)
  knight_plume_{base,normal}.png       2048^2 blue horsehair card atlas (8 columns, alpha in base)
  renders/armour_upper_tex_*.png       previews
run: python3 scripts/armour_upper_texgen.py
"""
import os, sys, math
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from scipy import ndimage as ndi

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import armour_upper_tex as TX

W = H = TX.W
REN = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "renders")
rng = np.random.default_rng(1234)

# ------------------------------------------------------------------------------------------------ colours (linear)
def srgb2lin(c):
    c = np.asarray(c, float)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def lin2srgb(c):
    c = np.clip(c, 0, 1)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * c ** (1 / 2.4) - 0.055)


STEEL = srgb2lin([0.58, 0.60, 0.64])
STEEL_DARK = srgb2lin([0.40, 0.41, 0.44])
GOLD = srgb2lin([0.93, 0.72, 0.38])
GOLD_DEEP = srgb2lin([0.62, 0.40, 0.14])
LEATHER = srgb2lin([0.30, 0.17, 0.09])
LEATHER_DARK = srgb2lin([0.12, 0.07, 0.04])


# ------------------------------------------------------------------------------------------------ noise helpers
def fnoise(h, w, sigma, seed, aniso=1.0):
    """Periodic gaussian-filtered noise, zero mean, unit std (wraps in both axes)."""
    r = np.random.default_rng(seed)
    x = r.standard_normal((h, w))
    y = ndi.gaussian_filter(x, (sigma, sigma * aniso), mode="wrap")
    y -= y.mean(); y /= (y.std() + 1e-9)
    return y


def fbm(h, w, sigmas, seed, weights=None, aniso=1.0):
    weights = weights or [1.0 / (i + 1) for i in range(len(sigmas))]
    out = sum(wt * fnoise(h, w, s, seed + i, aniso) for i, (s, wt) in enumerate(zip(sigmas, weights)))
    return out / (out.std() + 1e-9)


def scratches(h, w, n, seed, lmin=20, lmax=200, width=1, angle_spread=0.4):
    r = np.random.default_rng(seed)
    img = Image.new("L", (w, h), 0)
    d = ImageDraw.Draw(img)
    for _ in range(n):
        x, y = r.uniform(0, w), r.uniform(0, h)
        a = r.normal(0, angle_spread) + (r.uniform() < 0.3) * r.uniform(0, math.pi)
        L = r.uniform(lmin, lmax)
        pts = []
        for k in range(8):
            t = k / 7
            pts.append((x + math.cos(a) * L * t + r.normal(0, 0.6), y + math.sin(a) * L * t + r.normal(0, 0.6)))
        for dx in (-w, 0, w):
            d.line([(p[0] + dx, p[1]) for p in pts], fill=int(r.uniform(90, 255)), width=width)
    return np.asarray(img, np.float32) / 255.0


def normal_from_height(hgt, strength):
    """OpenGL tangent-space normal (x right = +U, y up = +V; image rows go down) from a height field (px units)."""
    dx = (np.roll(hgt, -1, 1) - np.roll(hgt, 1, 1)) * 0.5
    dy = (np.roll(hgt, -1, 0) - np.roll(hgt, 1, 0)) * 0.5
    nx = -dx * strength
    ny = dy * strength
    nz = np.ones_like(hgt)
    n = np.stack([nx, ny, nz], -1)
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    return n


def cavity(hgt, sigma=4.0):
    return np.clip(hgt - ndi.gaussian_filter(hgt, sigma, mode="wrap"), -1, 1)


def supersample_draw(w, h, fn, ss=4):
    img = Image.new("L", (w * ss, h * ss), 0)
    d = ImageDraw.Draw(img)
    fn(d, ss)
    return np.asarray(img.resize((w, h), Image.LANCZOS), np.float32) / 255.0


# ------------------------------------------------------------------------------------------------ canvases
class Sheet:
    def __init__(self, w, h):
        self.base = np.zeros((h, w, 3), np.float32)
        self.rough = np.full((h, w), 0.5, np.float32)
        self.metal = np.zeros((h, w), np.float32)
        self.ao = np.ones((h, w), np.float32)
        self.hgt = np.zeros((h, w), np.float32)      # height in px-equivalent units (normal strength applied later)
        self.nstr = np.ones((h, w), np.float32)
        self.w, self.h = w, h

    def rows(self, v_top, v_bot):
        r0 = int(round((1 - v_top) * self.h)); r1 = int(round((1 - v_bot) * self.h))
        return slice(r0, r1)

    def cols(self, u0, u1):
        return slice(int(round(u0 * self.w)), int(round(u1 * self.w)))


def steel_surface(h, w, seed, tone=1.0, dark=False):
    """Plate steel: brushed + faint hammering + scratches + smudges. Returns base, rough, height."""
    brushed = fnoise(h, w, 0.7, seed, aniso=40.0)
    low = fbm(h, w, [60, 25], seed + 10)
    ham = fbm(h, w, [9, 5], seed + 20)
    sc = scratches(h, w, int(h * w / 9000), seed + 30, width=1)
    smudge = np.clip(fbm(h, w, [40, 15], seed + 40) - 0.6, 0, 3) / 2.0
    col = (STEEL_DARK if dark else STEEL) * tone
    base = col[None, None, :] * (1 + 0.045 * low[..., None] + 0.03 * brushed[..., None] - 0.10 * smudge[..., None]
                                   + 0.12 * sc[..., None])
    rough = 0.30 + 0.035 * brushed + 0.05 * low + 0.14 * smudge - 0.10 * sc + (0.08 if dark else 0)
    hgt = 0.35 * ham + 0.12 * brushed - 0.9 * sc
    return base, np.clip(rough, 0.12, 0.9), hgt


def gold_surface(h, w, seed):
    brushed = fnoise(h, w, 0.8, seed, aniso=25.0)
    low = fbm(h, w, [50, 18], seed + 5)
    wear = np.clip(fbm(h, w, [12, 5], seed + 6) - 1.0, 0, 3)
    base = GOLD[None, None, :] * (1 + 0.05 * low[..., None] + 0.03 * brushed[..., None]) + 0.08 * wear[..., None] * np.array([0.5, 0.45, 0.35])
    rough = 0.26 + 0.03 * brushed + 0.05 * low
    return base, np.clip(rough, 0.12, 0.8), 0.1 * brushed


# ------------------------------------------------------------------------------------------------ motifs
def scroll_mask(w, h, period=256, amp=0.26, stroke=0.075, curl=True, seed=0):
    """Running vine scroll (stem + alternating spiral curls + leaves), periodic in U."""
    def draw(d, ss):
        W_, H_ = w * ss, h * ss
        lw = max(1, int(stroke * h * ss))
        stem = []
        for x in np.arange(-period, w + period, 2):
            y = 0.5 + amp * math.sin(2 * math.pi * x / period)
            stem.append((x * ss, (1 - y) * H_))
        d.line(stem, fill=255, width=lw, joint="curve")
        if curl:
            for k in range(-1, w // period + 2):
                for half in (0, 1):
                    x0 = k * period + period * (0.25 + 0.5 * half)
                    sgn = 1 if half == 0 else -1
                    # a curl springing from the stem crest and winding inward
                    cx, cy = x0 + period * 0.22, 0.5 + sgn * amp * 0.15
                    pts = []
                    r0 = 0.30 * h
                    for t in np.linspace(0, 2.3 * math.pi, 60):
                        r = r0 * (1 - t / (2.6 * math.pi))
                        ang = math.pi + sgn * t
                        pts.append(((cx + r * math.cos(ang) * 1.25) * ss, (1 - (cy + sgn * 0.0) ) * H_ - sgn * r * math.sin(ang) * ss))
                    d.line(pts, fill=255, width=max(1, int(lw * 0.8)), joint="curve")
                    # small pointed leaf springing from the stem on the other side
                    lx = x0 - period * 0.02
                    ly = 0.5 + amp * math.sin(2 * math.pi * lx / period)
                    tip = (lx - period * 0.13, ly - sgn * 0.30)
                    leaf = []
                    for t in np.linspace(0, 1, 12):
                        bx = lx + (tip[0] - lx) * t; by = ly + (tip[1] - ly) * t
                        leaf.append((bx * ss, (1 - by) * H_ - sgn * math.sin(math.pi * t) * 0.07 * H_))
                    for t in np.linspace(1, 0, 12):
                        bx = lx + (tip[0] - lx) * t; by = ly + (tip[1] - ly) * t
                        leaf.append((bx * ss, (1 - by) * H_ + sgn * math.sin(math.pi * t) * 0.02 * H_))
                    d.polygon(leaf, fill=220)
    m = supersample_draw(w, h, draw)
    return m


def etch_mask(w, h, period=352):
    """'etch_band' motif (judge M6: engraved gold filigree on every plate): a running rinceau (stem + curls + leaves)
    inlaid in gold along the plate border, a thin gold fillet on the border side (band frac ~0.88: decal_ribbon maps
    the border to frac 1 and the plate interior to frac 0) and an engraved groove round every gold element.
    Returns (gold mask, groove mask) in 0..1, periodic in U."""
    sc = scroll_mask(w, h, period=period, amp=0.18, stroke=0.055, curl=True)
    fil = border_lines(w, h, [0.88], 1.3)
    y = (np.arange(h) + 0.5)[:, None] / h
    m = np.clip(np.maximum(sc * (y > 0.08) * (y < 0.80), fil), 0, 1)
    m = ndi.gaussian_filter(m, 0.45, mode="wrap")
    md = ndi.gaussian_filter(m, 1.3, mode="wrap")
    groove = np.clip((md - m) * 3.5, 0, 1)
    return m, groove


def border_lines(w, h, positions, width_px=2.0):
    y = np.arange(h)[:, None] + 0.5
    m = np.zeros((h, w), np.float32)
    for p in positions:
        m = np.maximum(m, np.clip(1 - np.abs(y - p * h) / width_px, 0, 1) * np.ones((1, w)))
    return m


def beads_row(w, h, yc, spacing, radius):
    x = np.arange(w)[None, :] + 0.5; y = np.arange(h)[:, None] + 0.5
    xx = (x % spacing) - spacing / 2
    d = np.sqrt(xx ** 2 + (y - yc) ** 2)
    return np.clip(1 - (d / radius) ** 2, 0, 1) ** 0.5


def rope(w, h, pitch=24, slant=1.0):
    x = np.arange(w)[None, :]; y = np.arange(h)[:, None]
    ph = (x + slant * y * pitch / h * 1.2) / pitch
    return 0.5 + 0.5 * np.cos(2 * math.pi * ph)


# ------------------------------------------------------------------------------------------------ lion rampant (SDF)
def seg_sdf(px, py, a, b, ra, rb):
    ax, ay = a; bx, by = b
    vx, vy = bx - ax, by - ay
    L2 = vx * vx + vy * vy + 1e-12
    t = np.clip(((px - ax) * vx + (py - ay) * vy) / L2, 0, 1)
    dx, dy = px - (ax + t * vx), py - (ay + t * vy)
    return np.sqrt(dx * dx + dy * dy) - (ra + (rb - ra) * t)


def ell_sdf(px, py, c, r, rot=0.0):
    cs, sn = math.cos(rot), math.sin(rot)
    x, y = px - c[0], py - c[1]
    xr, yr = cs * x + sn * y, -sn * x + cs * y
    k = np.sqrt((xr / r[0]) ** 2 + (yr / r[1]) ** 2)
    return (k - 1) * min(r)


def smin(a, b, k=0.02):
    h = np.clip(0.5 + 0.5 * (b - a) / k, 0, 1)
    return b * (1 - h) + a * h - k * h * (1 - h)


def chain(px, py, pts, r0, r1, k=0.012):
    d = None
    n = len(pts) - 1
    for i in range(n):
        ra = r0 + (r1 - r0) * i / n; rb = r0 + (r1 - r0) * (i + 1) / n
        s = seg_sdf(px, py, pts[i], pts[i + 1], ra, rb)
        d = s if d is None else smin(d, s, k)
    return d


def bez(p0, p1, p2, p3, n=16):
    t = np.linspace(0, 1, n)[:, None]
    P = [np.array(p) for p in (p0, p1, p2, p3)]
    return [tuple(x) for x in ((1 - t) ** 3 * P[0] + 3 * (1 - t) ** 2 * t * P[1] + 3 * (1 - t) * t ** 2 * P[2] + t ** 3 * P[3])]


def lion_sdf(px, py):
    """Heraldic lion rampant facing dexter (viewer's left), in a unit box (x right, y up)."""
    # mane mass + flame locks
    d = ell_sdf(px, py, (0.405, 0.690), (0.118, 0.128), rot=0.2)
    locks = [(95, 0.10), (70, 0.12), (45, 0.13), (20, 0.13), (-5, 0.12), (-30, 0.12), (-55, 0.11), (-80, 0.10),
             (-105, 0.09), (-130, 0.08), (120, 0.07)]
    for a, L in locks:
        an = math.radians(a)
        c0 = (0.415 + 0.085 * math.cos(an), 0.690 + 0.095 * math.sin(an))
        c1 = (c0[0] + math.cos(an - 0.35) * L * 0.55, c0[1] + math.sin(an - 0.35) * L * 0.55)
        c2 = (c0[0] + math.cos(an - 0.75) * L, c0[1] + math.sin(an - 0.75) * L)
        d = smin(d, chain(px, py, [c0, c1, c2], 0.034, 0.002), 0.02)
    # head: cranium, muzzle, nose, lower jaw with an open mouth, ear
    d = smin(d, ell_sdf(px, py, (0.315, 0.745), (0.068, 0.060), rot=-0.1), 0.02)
    d = smin(d, ell_sdf(px, py, (0.245, 0.728), (0.052, 0.036), rot=-0.12), 0.02)
    d = smin(d, ell_sdf(px, py, (0.262, 0.672), (0.040, 0.018), rot=0.25), 0.012)
    d = np.maximum(d, -ell_sdf(px, py, (0.225, 0.695), (0.040, 0.010), rot=0.2))            # open mouth
    d = smin(d, ell_sdf(px, py, (0.345, 0.815), (0.022, 0.032), rot=-0.5), 0.01)            # ear
    # torso: chest, waist, haunch
    d = smin(d, ell_sdf(px, py, (0.455, 0.555), (0.098, 0.105), rot=-0.35), 0.04)
    d = smin(d, seg_sdf(px, py, (0.48, 0.50), (0.575, 0.355), 0.066, 0.058), 0.04)
    d = smin(d, ell_sdf(px, py, (0.605, 0.315), (0.092, 0.080), rot=0.5), 0.04)
    # forelegs raised, big paws with claws
    d = smin(d, chain(px, py, [(0.42, 0.585), (0.32, 0.605), (0.225, 0.655)], 0.050, 0.031), 0.02)
    d = smin(d, chain(px, py, [(0.45, 0.500), (0.35, 0.465), (0.265, 0.505)], 0.049, 0.030), 0.02)
    for (pp, ang) in (((0.205, 0.665), 150), ((0.245, 0.512), 170)):
        d = smin(d, ell_sdf(px, py, pp, (0.037, 0.031)), 0.01)
        for k in (-40, -10, 20):
            an = math.radians(ang + k)
            q = (pp[0] + math.cos(an) * 0.022, pp[1] + math.sin(an) * 0.022)
            d = smin(d, chain(px, py, [q, (q[0] + math.cos(an) * 0.025, q[1] + math.sin(an) * 0.025),
                                       (q[0] + math.cos(an + 0.9) * 0.034, q[1] + math.sin(an + 0.9) * 0.034)], 0.008, 0.0015), 0.004)
    # hind legs: sinister planted, dexter raised forward
    d = smin(d, chain(px, py, [(0.64, 0.30), (0.56, 0.165), (0.63, 0.075), (0.555, 0.045)], 0.050, 0.022), 0.02)
    d = smin(d, chain(px, py, [(0.56, 0.31), (0.44, 0.27), (0.43, 0.155), (0.345, 0.125)], 0.046, 0.021), 0.02)
    for (pp, ang) in (((0.545, 0.045), 185), ((0.335, 0.125), 180)):
        for k in (-30, 0, 30):
            an = math.radians(ang + k)
            d = smin(d, seg_sdf(px, py, pp, (pp[0] + math.cos(an) * 0.035, pp[1] + math.sin(an) * 0.035), 0.010, 0.002), 0.005)
    # tail: out from the rump, S-curve up, flame tuft
    tail = bez((0.68, 0.29), (0.86, 0.30), (0.74, 0.58), (0.86, 0.70), 22)
    d = smin(d, chain(px, py, tail, 0.027, 0.017), 0.015)
    tp = tail[-1]
    for a, L in ((70, 0.12), (105, 0.13), (35, 0.10), (140, 0.09), (5, 0.08)):
        an = math.radians(a)
        d = smin(d, chain(px, py, [tp, (tp[0] + math.cos(an) * L * 0.5 + 0.012, tp[1] + math.sin(an) * L * 0.5),
                                   (tp[0] + math.cos(an - 0.5) * L, tp[1] + math.sin(an - 0.5) * L)], 0.030, 0.002), 0.012)
    return d


def lion_details(lx, ly):
    """Engraved detail lines inside the lion: mane lock grooves, eye, brow, rib / thigh contours."""
    g = np.zeros_like(lx)
    for a in (80, 50, 20, -10, -40, -70, -100):
        an = math.radians(a)
        c0 = (0.415 + 0.05 * math.cos(an), 0.690 + 0.055 * math.sin(an))
        c2 = (c0[0] + math.cos(an - 0.6) * 0.11, c0[1] + math.sin(an - 0.6) * 0.11)
        g = np.maximum(g, np.clip(1 - np.abs(chain(lx, ly, [c0, c2], 0.0, 0.0)) / 0.006, 0, 1))
    g = np.maximum(g, np.clip(1 - np.hypot(lx - 0.290, ly - 0.765) / 0.011, 0, 1))                    # eye
    g = np.maximum(g, np.clip(1 - np.abs(seg_sdf(lx, ly, (0.26, 0.785), (0.33, 0.775), 0, 0)) / 0.005, 0, 1) * 0.8)
    g = np.maximum(g, np.clip(1 - np.abs(ell_sdf(lx, ly, (0.605, 0.315), (0.070, 0.058), rot=0.5)) / 0.004, 0, 1) * 0.3)
    for k in range(3):
        g = np.maximum(g, np.clip(1 - np.abs(seg_sdf(lx, ly, (0.47 + 0.03 * k, 0.53 - 0.02 * k), (0.53 + 0.03 * k, 0.47 - 0.03 * k), 0, 0)) / 0.004, 0, 1) * 0.6)
    return g


def lion_height(n, scale=0.62, cx=0.5, cy=0.5):
    y, x = np.mgrid[0:n, 0:n]
    px = (x + 0.5) / n; py = 1 - (y + 0.5) / n
    lx = (px - cx) / scale + 0.5; ly = (py - cy) / scale + 0.5
    d = lion_sdf(lx, ly) * scale                      # distance in square units
    inside = np.clip(-d / 0.014, 0, 1)
    h = np.sqrt(inside)                                # rounded repousse dome
    det = lion_details(lx, ly)
    h = h - 0.35 * det * inside
    edge = np.clip(1 - np.abs(d) / 0.004, 0, 1)        # engraved outline
    return h, inside, np.maximum(edge, 0.7 * det * inside)


# ------------------------------------------------------------------------------------------------ build the trim sheet
def build_armour():
    S = Sheet(W, H)
    # --- steel plates
    r = S.rows(*TX.STEEL)
    h = r.stop - r.start
    b, ro, hg = steel_surface(h, W, 1)
    S.base[r] = b; S.rough[r] = ro; S.metal[r] = 1.0; S.hgt[r] = hg; S.nstr[r] = 1.0
    # --- squares
    def sq(name):
        u0, v0, u1, v1 = TX.SQUARES[name]
        return S.rows(v1, v0), S.cols(u0, u1)
    # lion: steel ground + embossed gold lion + engraved ring
    rr, cc = sq("lion")
    n = rr.stop - rr.start
    b, ro, hg = steel_surface(n, n, 7)
    lh, inside, edge = lion_height(n, scale=0.56, cx=0.5, cy=0.5)
    gb, gr, _ = gold_surface(n, n, 8)
    cav = np.clip(-cavity(lh, 3.0) * 4, 0, 1)
    m = np.clip(inside * 1.6, 0, 1)[..., None]
    col = b * (1 - m) + gb * m
    col = col * (1 - 0.55 * cav[..., None]) * (1 - 0.6 * edge[..., None])
    S.base[rr, cc] = col
    S.rough[rr, cc] = ro * (1 - m[..., 0]) + gr * m[..., 0] + 0.25 * cav
    S.metal[rr, cc] = 1.0
    S.hgt[rr, cc] = hg + 9.0 * lh - 1.5 * edge
    S.nstr[rr, cc] = 1.0
    # rosette boss: gold petals
    rr, cc = sq("boss")
    n = rr.stop - rr.start
    y, x = np.mgrid[0:n, 0:n]; px = (x + 0.5) / n - 0.5; py = (y + 0.5) / n - 0.5
    rad = np.hypot(px, py); ang = np.arctan2(py, px)
    petal = np.clip(1 - rad / (0.33 + 0.1 * np.cos(8 * ang)), 0, 1) ** 0.6
    hub = np.clip(1 - rad / 0.1, 0, 1) ** 0.5
    gb, gr, _ = gold_surface(n, n, 9)
    hgt = 5 * petal + 4 * hub
    S.base[rr, cc] = gb * (1 - 0.45 * np.clip(-cavity(hgt, 3) , 0, 1)[..., None])
    S.rough[rr, cc] = gr; S.metal[rr, cc] = 1.0; S.hgt[rr, cc] = hgt
    # rivet head
    rr, cc = sq("rivet_head")
    n = rr.stop - rr.start
    y, x = np.mgrid[0:n, 0:n]; rad = np.hypot((x + 0.5) / n - 0.5, (y + 0.5) / n - 0.5)
    gb, gr, _ = gold_surface(n, n, 10)
    S.base[rr, cc] = gb; S.rough[rr, cc] = gr; S.metal[rr, cc] = 1; S.hgt[rr, cc] = 6 * np.clip(1 - (rad / 0.45) ** 2, 0, 1) ** 0.5
    # knot / etch squares: gold plain
    for nm in ("knot", "etch"):
        rr, cc = sq(nm)
        n = rr.stop - rr.start
        gb, gr, gh = gold_surface(n, cc.stop - cc.start, 11)
        S.base[rr, cc] = gb; S.rough[rr, cc] = gr; S.metal[rr, cc] = 1; S.hgt[rr, cc] = gh
    # leather
    rr, cc = sq("leather")
    n = rr.stop - rr.start
    grain = fbm(n, n, [1.2, 2.5, 6], 21)
    cells = fnoise(n, n, 3.5, 22)
    wear = np.clip(fbm(n, n, [30, 10], 23) * 0.5 + 0.5, 0, 1)
    col = LEATHER[None, None, :] * (0.8 + 0.25 * wear[..., None] + 0.06 * grain[..., None])
    S.base[rr, cc] = col; S.rough[rr, cc] = np.clip(0.62 + 0.08 * grain - 0.12 * wear, 0.35, 0.9); S.metal[rr, cc] = 0
    S.hgt[rr, cc] = 0.6 * grain + 0.8 * np.abs(cells)
    # dark
    rr, cc = sq("dark_sq")
    # iteration 2b (user item 30 / G7): the padded lining in the helm's shadow is near-black (was 0.012 linear = sRGB
    # 0.11, which read as lit grey steel through the eye slot)
    S.base[rr, cc] = 0.0040; S.rough[rr, cc] = 0.75; S.metal[rr, cc] = 0.0
    # --- trim bands (U tiles)
    for name, (vt, vb) in TX.BANDS.items():
        r = S.rows(vt, vb)
        h = r.stop - r.start
        yv = (np.arange(h) + 0.5)[:, None] / h * np.ones((1, W))        # 0 at the band top (v frac 0)
        if name in ("fil_wide", "fil_narrow", "gold_plain", "rivet", "gold_bead", "crest", "scroll"):
            gb, gr, gh = gold_surface(h, W, 100 + h)
        if name == "fil_wide":
            m = scroll_mask(W, h, period=320, amp=0.16, stroke=0.035) * border_mask(h, W, 0.26)
            lines = border_lines(W, h, [0.13, 0.87], 1.4)
            ridge = np.clip(1 - np.abs(yv - 0.06) / 0.05, 0, 1)
            riv = beads_row(W, h, 0.5 * h, 160, 0.17 * h)
            ringr = np.clip(1 - np.abs(np.sqrt(((np.arange(W)[None, :] % 160) - 80) ** 2 + (np.arange(h)[:, None] - 0.5 * h) ** 2) - 0.22 * h) / 1.5, 0, 1)
            m = m * (1 - np.clip(beads_row(W, h, 0.5 * h, 160, 0.30 * h) * 3, 0, 1))
            hgt = 2.2 * ridge - 1.4 * lines - 1.2 * m + 5.0 * riv - 1.2 * ringr + gh
            cav = np.clip(-cavity(hgt, 2.5) * 1.2, 0, 1)
            S.base[r] = gb * (1 - 0.45 * cav[..., None] - 0.35 * m[..., None])
            S.rough[r] = gr + 0.12 * cav + 0.1 * m
            S.metal[r] = 1; S.hgt[r] = hgt
        elif name == "fil_narrow":
            m = scroll_mask(W, h, period=224, amp=0.17, stroke=0.06, curl=True) * border_mask(h, W, 0.2)
            lines = border_lines(W, h, [0.14, 0.86], 1.2)
            hgt = -1.3 * m - 1.2 * lines + gh
            cav = np.clip(-cavity(hgt, 2.0) * 1.5, 0, 1)
            S.base[r] = gb * (1 - 0.40 * m[..., None] - 0.3 * cav[..., None])
            S.rough[r] = gr + 0.12 * m + 0.1 * cav
            S.metal[r] = 1; S.hgt[r] = hgt
        elif name == "rivet":
            dome = beads_row(W, h, 0.5 * h, 128, 0.30 * h)
            lines = border_lines(W, h, [0.12, 0.88], 1.3)
            hgt = 6 * dome + 1.2 * lines + gh
            S.base[r] = gb * (1 - 0.35 * np.clip(-cavity(hgt, 3) , 0, 1)[..., None])
            S.rough[r] = gr; S.metal[r] = 1; S.hgt[r] = hgt
        elif name in ("gold_bead", "steel_bead"):
            rp = rope(W, h, pitch=22)
            if name == "gold_bead":
                col, ro = gb, gr
            else:
                col, ro, _ = steel_surface(h, W, 55)
            hgt = 1.6 * rp
            S.base[r] = col * (1 - 0.3 * (1 - rp)[..., None]); S.rough[r] = ro + 0.08 * (1 - rp)
            S.metal[r] = 1; S.hgt[r] = hgt
        elif name == "gold_plain":
            S.base[r] = gb; S.rough[r] = gr; S.metal[r] = 1; S.hgt[r] = gh
        elif name == "crest":
            cb, cr, ch = steel_surface(h, W, 60)
            g = (np.abs(yv - 0.5) > 0.30).astype(np.float32)
            g = ndi.gaussian_filter(g, (1.0, 0.1))
            lines = border_lines(W, h, [0.2, 0.8], 1.2)
            S.base[r] = cb * (1 - g[..., None]) + gb * g[..., None]
            S.rough[r] = cr * (1 - g) + gr * g; S.metal[r] = 1; S.hgt[r] = ch + 1.2 * lines
        elif name == "strap":
            grain = fbm(h, W, [1.2, 3], 70)
            st = stitches(W, h, [0.16, 0.84], spacing=18)
            edge = np.clip(1 - np.minimum(yv, 1 - yv) / 0.06, 0, 1)
            col = LEATHER[None, None, :] * (0.9 + 0.08 * grain[..., None]) * (1 - 0.45 * edge[..., None])
            col = col * (1 - 0.3 * st[..., None]) + st[..., None] * srgb2lin([0.55, 0.45, 0.30]) * 0.35
            S.base[r] = col; S.rough[r] = 0.6 + 0.08 * grain; S.metal[r] = 0
            S.hgt[r] = 0.5 * grain - 1.5 * edge + 0.9 * st
        elif name == "steel_plain":
            b, ro, hg = steel_surface(h, W, 80, dark=True)
            S.base[r] = b; S.rough[r] = ro; S.metal[r] = 1; S.hgt[r] = hg
        elif name == "dark":
            S.base[r] = 0.0045; S.rough[r] = 0.78; S.metal[r] = 0.0
        elif name == "scroll":
            sb, sr, sh = steel_surface(h, W, 90)
            m = scroll_mask(W, h, period=320, amp=0.25, stroke=0.05)
            S.base[r] = sb * (1 - m[..., None]) + gb * m[..., None]
            S.rough[r] = sr * (1 - m) + gr * m; S.metal[r] = 1; S.hgt[r] = sh - 1.0 * m
        elif name == "steel_ridge":
            b, ro, hg = steel_surface(h, W, 95)
            S.base[r] = b; S.rough[r] = ro; S.metal[r] = 1; S.hgt[r] = hg
        elif name == "etch_band":
            sb, sr, sh = steel_surface(h, W, 97)
            gb, gr, _ = gold_surface(h, W, 98)
            m, groove = etch_mask(W, h)
            S.base[r] = sb * (1 - m[..., None]) + gb * m[..., None]
            S.rough[r] = sr * (1 - m) + gr * m + 0.15 * groove
            S.metal[r] = 1; S.hgt[r] = sh - 1.6 * groove
    # ambient occlusion from the height field (cavities)
    S.ao = np.clip(1 - 0.5 * np.clip(-cavity(S.hgt, 6.0) / 3.0, 0, 1), 0.3, 1)
    return S


def border_mask(h, w, frac):
    y = (np.arange(h) + 0.5)[:, None] / h
    return ((y > frac) & (y < 1 - frac)).astype(np.float32) * np.ones((1, w), np.float32)


def stitches(w, h, rows, spacing=16, length=10, width=1.6):
    x = np.arange(w)[None, :] + 0.5; y = np.arange(h)[:, None] + 0.5
    m = np.zeros((h, w), np.float32)
    for rv in rows:
        dash = ((x % spacing) < length).astype(np.float32)
        m = np.maximum(m, dash * np.clip(1 - np.abs(y - rv * h) / width, 0, 1))
    return m


def _save_atomic(im, path):
    """write next to the target, then rename (a parallel build never reads half a PNG)"""
    tmp = path + ".tmp%d.png" % os.getpid()
    im.save(tmp, optimize=True)
    os.replace(tmp, path)


def save_set(S, prefix, nstrength=0.12):
    base = lin2srgb(np.clip(S.base, 0, 1))
    _save_atomic(Image.fromarray((base * 255 + 0.5).astype(np.uint8), "RGB"), TX.FILES[prefix + "_base"])
    orm = np.stack([S.ao, np.clip(S.rough, 0, 1), np.clip(S.metal, 0, 1)], -1)
    _save_atomic(Image.fromarray((orm * 255 + 0.5).astype(np.uint8), "RGB"), TX.FILES[prefix + "_orm"])
    n = normal_from_height(S.hgt, nstrength * S.nstr)
    if getattr(S, "nrm_over", None) is not None:
        m = S.nrm_mask[..., None]
        n = n * (1 - m) + S.nrm_over * m
        n /= np.linalg.norm(n, axis=-1, keepdims=True)
    _save_atomic(Image.fromarray(((n * 0.5 + 0.5) * 255 + 0.5).astype(np.uint8), "RGB"), TX.FILES[prefix + "_normal"])
    print("TEX", prefix, [TX.FILES[prefix + k] for k in ("_base", "_orm", "_normal")])


# ------------------------------------------------------------------------------------------------ mail
def build_mail(n=1024, cols=12):
    """Riveted 4-in-1 mail: rows of rings, alternate rows offset and tilted opposite ways. 1 tile = 0.10 m."""
    S = Sheet(n, n)
    rows = cols * 2
    px = n / cols; py = n / rows
    y, x = np.mgrid[0:n, 0:n].astype(np.float32) + 0.5
    hgt = np.full((n, n), -1.0, np.float32)
    tint = np.zeros((n, n), np.float32)
    R_out, R_in = px * 0.62, px * 0.36
    for rrow in range(-1, rows + 1):
        off = (rrow % 2) * px / 2
        tilt = 1 if rrow % 2 == 0 else -1
        cy = (rrow + 0.5) * py
        for c in range(-1, cols + 1):
            cx = c * px + off + px / 2
            # local window
            y0, y1 = int(max(0, cy - R_out - 2)), int(min(n, cy + R_out + 2))
            if y1 <= y0:
                continue
            for dxw in (-n, 0, n):
                x0, x1 = int(max(0, cx + dxw - R_out - 2)), int(min(n, cx + dxw + R_out + 2))
                if x1 <= x0:
                    continue
                yy = y[y0:y1, x0:x1] - cy; xx = x[y0:y1, x0:x1] - (cx + dxw)
                yy = yy * 1.35                                   # rings seen slightly edge-on (ellipses)
                rad = np.sqrt(xx * xx + yy * yy)
                mid = (R_out + R_in) / 2; half = (R_out - R_in) / 2
                prof = np.clip(1 - ((rad - mid) / half) ** 2, 0, 1)
                ring = np.sqrt(prof)
                # tilt: one side of the ring higher (overlap order)
                h = ring * (1.0 + 0.45 * tilt * xx / R_out) + 0.2 * rrow % 2 * 0
                rivet = np.clip(1 - np.hypot(xx - tilt * mid * 0.0, yy + mid) / (half * 0.8), 0, 1) * (prof > 0)
                h = h + 0.35 * rivet
                sub = hgt[y0:y1, x0:x1]
                upd = h > sub
                sub[upd] = h[upd]
                tint[y0:y1, x0:x1] = np.maximum(tint[y0:y1, x0:x1], prof)
    gap = (hgt < 0).astype(np.float32)
    hgt = np.where(gap > 0, -0.8, hgt)
    hgt = ndi.gaussian_filter(hgt, 0.7, mode="wrap")
    noise = fnoise(n, n, 20, 5)
    col = srgb2lin([0.52, 0.53, 0.55]) * (0.85 + 0.08 * noise[..., None])
    ao = np.clip(0.25 + 0.75 * np.clip(hgt + 0.5, 0, 1.5) / 1.5, 0.15, 1)
    S.base = col[None, None, :] * np.ones((n, n, 1)) if col.ndim == 1 else col
    S.base = S.base * (ao[..., None] * 0.8 + 0.2) * (1 - 0.9 * gap[..., None])
    S.rough = np.clip(0.36 + 0.1 * noise + 0.4 * gap, 0.2, 0.95)
    S.metal = 1 - gap
    S.ao = ao
    S.hgt = hgt * 5
    return S


# ------------------------------------------------------------------------------------------------ plume
def build_plume(n=2048, cols=8):
    """Horsehair card atlas: long coarse glossy strands, royal blue. Strands run along V (root at the top)."""
    cw = n // cols
    lum = np.zeros((n, n), np.float32); cov = np.zeros((n, n), np.float32); nx = np.zeros((n, n), np.float32)
    r = np.random.default_rng(77)
    y = np.arange(n, dtype=np.float32)[:, None]
    for c in range(cols):
        dense = [700, 520, 520, 380, 380, 260, 180, 120][c]
        for k in range(dense):
            x0 = c * cw + r.uniform(0.06, 0.94) * cw
            wv = r.uniform(0.8, 2.2)
            amp = r.uniform(1, 6); freq = r.uniform(0.002, 0.008); ph = r.uniform(0, 6.28)
            start = r.uniform(0, 0.06) * n; end = n * r.uniform(0.70, 1.0)
            xs = x0 + amp * np.sin(freq * y[:, 0] + ph) + (y[:, 0] / n) * r.normal(0, 8)
            xs = np.clip(xs, c * cw + 2, (c + 1) * cw - 3)
            b = r.uniform(0.55, 1.0)
            for dxp in range(-3, 4):
                xi = np.floor(xs).astype(int) + dxp
                fr = np.clip(1 - np.abs(xi + 0.5 - xs) / wv, 0, 1)
                t = y[:, 0] / n
                fade = np.clip((y[:, 0] - start) / 40, 0, 1) * np.clip((end - y[:, 0]) / (0.25 * n), 0, 1) ** 0.7
                a = fr * fade
                rows_ = np.arange(n)
                ok = a > 0.01
                lum[rows_[ok], xi[ok]] = np.maximum(lum[rows_[ok], xi[ok]], b * a[ok])
                cov[rows_[ok], xi[ok]] = np.maximum(cov[rows_[ok], xi[ok]], a[ok])
                nx[rows_[ok], xi[ok]] = np.clip((xi[ok] + 0.5 - xs[ok]) / wv, -1, 1)
    blue = srgb2lin([0.10, 0.18, 0.62]); hi = srgb2lin([0.30, 0.42, 0.90]); deep = srgb2lin([0.03, 0.05, 0.22])
    t = lum[..., None]
    col = deep * (1 - t) + blue * t
    col = col + (hi - blue) * np.clip(lum - 0.85, 0, 1)[..., None] * 3
    alpha = np.clip(cov * 1.25, 0, 1)
    base = np.concatenate([lin2srgb(col), alpha[..., None]], -1)
    Image.fromarray((base * 255 + 0.5).astype(np.uint8), "RGBA").save(TX.FILES["plume_base"], optimize=True)
    nn = np.stack([nx * 0.7, np.zeros_like(nx), np.sqrt(np.clip(1 - (nx * 0.7) ** 2, 0, 1))], -1)
    Image.fromarray(((nn * 0.5 + 0.5) * 255 + 0.5).astype(np.uint8), "RGB").save(TX.FILES["plume_normal"], optimize=True)
    print("TEX plume", TX.FILES["plume_base"])


def previews():
    os.makedirs(REN, exist_ok=True)
    b = Image.open(TX.FILES["armour_base"]).convert("RGB")
    nm = Image.open(TX.FILES["armour_normal"]).convert("RGB")
    both = Image.new("RGB", (2048, 1024))
    both.paste(b.resize((1024, 1024)), (0, 0)); both.paste(nm.resize((1024, 1024)), (1024, 0))
    both.save(os.path.join(REN, "armour_upper_tex_sheet.png"))
    u0, v0, u1, v1 = TX.SQUARES["lion"]
    b.crop((0, int((1 - v1) * 2048), 512, int((1 - v0) * 2048))).save(os.path.join(REN, "armour_upper_tex_lion.png"))
    b.crop((0, int(0.625 * 2048), 1024, 2048)).save(os.path.join(REN, "armour_upper_tex_trims.png"))
    Image.open(TX.FILES["mail_base"]).convert("RGB").resize((512, 512)).save(os.path.join(REN, "armour_upper_tex_mail.png"))
    p = Image.open(TX.FILES["plume_base"])
    bg = Image.new("RGBA", p.size, (40, 40, 40, 255)); bg.alpha_composite(p)
    bg.convert("RGB").resize((1024, 1024)).save(os.path.join(REN, "armour_upper_tex_plume.png"))


# ------------------------------------------------------------------------------------------------ compositing
class Src:
    """A texture set of the materials agent (textures_char.json): linear base, orm, normal (-1..1) as float arrays."""

    def __init__(self, name):
        import json
        sets = json.load(open(TX.MANIFEST))["sets"]
        self.info = sets[name]
        f = self.info["files"]
        d = TX.TEX_DIR
        self.base = srgb2lin(np.asarray(Image.open(os.path.join(d, f["base"])).convert("RGB"), np.float32) / 255.0)
        self.orm = np.asarray(Image.open(os.path.join(d, f["orm"])).convert("RGB"), np.float32) / 255.0
        self.nrm = np.asarray(Image.open(os.path.join(d, f["normal"])).convert("RGB"), np.float32) / 255.0 * 2 - 1

    def crop(self, r0, r1, c0=0, c1=None):
        c1 = c1 or self.base.shape[1]
        return self.base[r0:r1, c0:c1], self.orm[r0:r1, c0:c1], self.nrm[r0:r1, c0:c1]


def rs(a, w, h):
    """Resample a float image (h, w[, c]) with Lanczos (per channel)."""
    if a.ndim == 2:
        return np.asarray(Image.fromarray(a.astype(np.float32), "F").resize((w, h), Image.LANCZOS))
    return np.stack([rs(a[..., k], w, h) for k in range(a.shape[-1])], -1)


def tile_to(a, w, h):
    reps = (int(math.ceil(h / a.shape[0])), int(math.ceil(w / a.shape[1]))) + (1,) * (a.ndim - 2)
    return np.tile(a, reps)[:h, :w]


def put(S, rows, cols, base, orm, nrm):
    S.base[rows, cols] = base
    S.ao[rows, cols] = orm[..., 0]; S.rough[rows, cols] = orm[..., 1]; S.metal[rows, cols] = orm[..., 2]
    S.nrm_over[rows, cols] = nrm
    S.nrm_mask[rows, cols] = 1.0


def compose_from_materials(S):
    """Overwrite the procedural trim sheet with the materials agent's texture sets, region by region (same layout)."""
    if not os.path.exists(TX.MANIFEST):
        print("TEX no textures_char.json: procedural trim sheet only")
        return []
    import json
    sets = json.load(open(TX.MANIFEST))["sets"]
    used = []
    S.nrm_over = np.zeros((S.h, S.w, 3), np.float32); S.nrm_mask = np.zeros((S.h, S.w), np.float32)
    # ---- plate steel: steel_worn with a share of steel_blued (reference: dark gunmetal), 4096 -> 1024 px/m
    steel = None
    if "steel_worn" in sets:
        sw = Src("steel_worn"); used.append("steel_worn")
        k = TX.STEEL_DENS / sw.info["px_per_m"]
        n = int(round(sw.base.shape[0] * k))
        b, o, nn = rs(sw.base, n, n), rs(sw.orm, n, n), rs(sw.nrm, n, n)
        if "steel_blued" in sets:
            sb = Src("steel_blued"); used.append("steel_blued")
            b = 0.88 * b + 0.12 * rs(sb.base, n, n)
        # aged gunmetal (assembly pass vs the reference): a clean bluish glossy steel read as chrome, a uniformly
        # rough light one as aluminium. The sheet's plates are dark warm steel, mottled (darker, rougher grime clouds
        # and lighter polished areas) with fine pitting, and fairly glossy so they mirror a dark surround with crisp
        # highlights. Albedo ~0.16 linear (sRGB ~0.44) with a slight bronze cast; roughness 0.22..0.55.
        b = b * np.array([0.56, 0.505, 0.435], np.float32)
        o = o.copy(); o[..., 1] = np.clip(o[..., 1] * 0.85 + 0.12, 0.22, 0.75)
        steel = (b, o, nn)
        # the same steel at the source density (4096 px/m): ground of the etched border ribbons (~4-8k px/m)
        bF = sw.base.copy()
        if "steel_blued" in sets and sb.base.shape == bF.shape:
            bF = 0.88 * bF + 0.12 * sb.base
        oF = sw.orm.copy(); oF[..., 1] = np.clip(oF[..., 1] * 0.85 + 0.12, 0.22, 0.75)
        steelF = (bF * np.array([0.56, 0.505, 0.435], np.float32), oF, sw.nrm)
        r = S.rows(*TX.STEEL); h = r.stop - r.start
        bt, ot, nt = tile_to(b, S.w, h), tile_to(o, S.w, h).copy(), tile_to(nn, S.w, h).copy()
        # mottling over the whole 2 m x 0.75 m steel area (not the 0.5 m source tile): no visible repeat on a plate
        mot = fbm(h, S.w, [90, 38, 14], 501)                     # large clouds
        grime = np.clip(fbm(h, S.w, [55, 20, 7], 502) - 0.35, 0, 2.5) / 2.5
        polish = np.clip(fbm(h, S.w, [70, 26], 503) - 0.9, 0, 2.0) / 2.0
        pit_n = fnoise(h, S.w, 0.8, 504)
        pits = np.clip(pit_n - 2.7, 0, 1.2) / 1.2                 # sparse 1-2 px pits
        k = (1.0 + 0.16 * mot - 0.40 * grime + 0.30 * polish - 0.35 * pits)[..., None]
        warm = np.array([1.0, 0.97, 0.9], np.float32)[None, None, :]
        bt = bt * k * (1.0 + (warm - 1.0) * grime[..., None])     # grime is a warmer brown-grey
        ot[..., 1] = np.clip(ot[..., 1] + 0.18 * grime - 0.10 * polish + 0.15 * pits + 0.03 * mot, 0.2, 0.8)
        ot[..., 0] = np.clip(ot[..., 0] * (1 - 0.35 * pits), 0, 1)
        # pits as small dents in the normal map
        dz = ndi.gaussian_filter(pits, 0.8, mode="wrap")
        gx = (np.roll(dz, -1, 1) - np.roll(dz, 1, 1)) * 0.5; gy = (np.roll(dz, -1, 0) - np.roll(dz, 1, 0)) * 0.5
        nt[..., 0] += gx * 0.7; nt[..., 1] -= gy * 0.7
        nt /= np.maximum(np.linalg.norm(nt, axis=-1, keepdims=True), 1e-6)
        put(S, r, slice(0, S.w), bt, ot, nt)
        # inner / lip steel: darker, rougher
        for band, dark in (("steel_plain", 0.62), ("steel_ridge", 1.0), ("steel_bead", 1.0)):
            r = S.rows(*TX.BANDS[band]); h = r.stop - r.start
            o2 = tile_to(o, S.w, h).copy(); o2[..., 1] = np.clip(o2[..., 1] + (0.12 if dark < 1 else 0), 0, 1)
            put(S, r, slice(0, S.w), tile_to(b, S.w, h) * dark, o2, tile_to(nn, S.w, h))
    # ---- gold trims: trim_gold strips at half scale (U period 1024 px), aspect kept
    if "trim_gold" in sets:
        tg = Src("trim_gold"); used.append("trim_gold")
        strips = tg.info["strips"]
        H0 = tg.base.shape[0]
        def strip(name, band):
            r0, r1 = strips[name]["rows_px"]
            b, o, nn = tg.crop(r0, r1)
            h = S.rows(*TX.BANDS[band]); hh = h.stop - h.start
            w2 = int(round(b.shape[1] * hh / b.shape[0]))
            b, o, nn = rs(b, w2, hh), rs(o, w2, hh), rs(nn, w2, hh)
            nn = nn / np.maximum(np.linalg.norm(nn, axis=-1, keepdims=True), 1e-6)
            put(S, h, slice(0, S.w), tile_to(b, S.w, hh), tile_to(o, S.w, hh), tile_to(nn, S.w, hh))
        for name, band in (("filigree_wide", "fil_wide"), ("filigree_narrow", "fil_narrow"), ("rivets", "rivet"),
                           ("rope", "gold_bead"), ("lines", "gold_plain"), ("pearls", "pearls")):
            if name in strips:
                strip(name, band)
        # steel rope bead: the gold rope's relief on steel
        if steel is not None and "rope" in strips:
            r0, r1 = strips["rope"]["rows_px"]
            _, _, nn = tg.crop(r0, r1)
            h = S.rows(*TX.BANDS["steel_bead"]); hh = h.stop - h.start
            w2 = int(round(nn.shape[1] * hh / nn.shape[0]))
            nn = rs(nn, w2, hh); nn /= np.maximum(np.linalg.norm(nn, axis=-1, keepdims=True), 1e-6)
            S.nrm_over[h] = tile_to(nn, S.w, hh)
        # crest: gold lines | steel | gold lines
        if steel is not None and "lines" in strips:
            h = S.rows(*TX.BANDS["crest"]); hh = h.stop - h.start
            r0, r1 = strips["lines"]["rows_px"]
            gb, go, gn = tg.crop(r0, r1)
            gb, go, gn = rs(gb, S.w // 2, hh), rs(go, S.w // 2, hh), rs(gn, S.w // 2, hh)
            gb, go, gn = tile_to(gb, S.w, hh), tile_to(go, S.w, hh), tile_to(gn, S.w, hh)
            sb_, so_, sn_ = tile_to(steel[0], S.w, hh), tile_to(steel[1], S.w, hh), tile_to(steel[2], S.w, hh)
            y = ((np.arange(hh) + 0.5) / hh)[:, None, None]
            g = (np.abs(y - 0.5) > 0.28).astype(np.float32)
            put(S, h, slice(0, S.w), gb * g + sb_ * (1 - g), go * g + so_ * (1 - g), gn * g + sn_ * (1 - g))
        # lion on steel for the pauldron cops: the lion_inlay_steel decal, enlarged, feathered into our steel square
        dec = tg.info.get("decals", {})
        if "lion_inlay_steel" in dec and steel is not None:
            u0, v0, u1, v1 = TX.SQUARES["lion"]
            rr, cc = S.rows(v1, v0), S.cols(u0, u1)
            n = rr.stop - rr.start
            sb_, so_, sn_ = tile_to(steel[0], n, n), tile_to(steel[1], n, n), tile_to(steel[2], n, n)
            ob = sb_.copy(); oo = so_.copy(); on = sn_.copy()

            emb = np.zeros((n, n), np.float32)          # embossed relief of the stamped gold (judge M6)

            def stamp(dname, size_m, cx_, cy_):
                """Paste the gold of a steel-inlay decal (mask = gold saturation) at (cx_, cy_) (fractions)."""
                du, dv = dec[dname]["u"], dec[dname]["v"]
                c0, c1 = int(du[0] * tg.base.shape[1]), int(du[1] * tg.base.shape[1])
                r0, r1 = int((1 - dv[1]) * H0), int((1 - dv[0]) * H0)
                b, o, nn = tg.crop(r0, r1, c0, c1)
                m = int(round(n * size_m / 0.30))
                b, o, nn = rs(b, m, m), rs(o, m, m), rs(nn, m, m)
                sat = b.max(-1) - b.min(-1)
                gm = np.clip((sat - 0.13) / 0.10, 0, 1)
                gm = ndi.gaussian_filter(gm, 0.5)
                y0 = int(cy_ * n - m / 2); x0 = int(cx_ * n - m / 2)
                ys, xs = max(0, y0), max(0, x0)
                ye, xe = min(n, y0 + m), min(n, x0 + m)
                sl = (slice(ys, ye), slice(xs, xe)); dl = (slice(ys - y0, ye - y0), slice(xs - x0, xe - x0))
                g3 = gm[dl][..., None]
                ob[sl] = ob[sl] * (1 - g3) + b[dl] * g3
                oo[sl] = oo[sl] * (1 - g3) + o[dl] * g3
                on[sl] = on[sl] * (1 - g3) + nn[dl] * g3
                emb[sl] = np.maximum(emb[sl], gm[dl])
            if "frame_inlay_steel" in dec:
                stamp("frame_inlay_steel", 0.285, 0.5, 0.52)
            stamp("lion_inlay_steel", 0.175, 0.5, 0.47)
            # raised 2-3 mm: a rounded height from the gold mask, its normal added to the decal's, AO in the recesses
            hgt = ndi.gaussian_filter(emb, 2.2) * 5.0 + ndi.gaussian_filter(emb, 0.8) * 1.5
            ne = normal_from_height(hgt, 0.55)
            on = on + (ne - np.array([0, 0, 1.0], np.float32))
            cav = np.clip(ndi.gaussian_filter(emb, 3.0) - emb, 0, 1)
            oo[..., 0] = oo[..., 0] * (1 - 0.6 * cav)
            ob = ob * (1 - 0.35 * cav[..., None])
            on /= np.maximum(np.linalg.norm(on, axis=-1, keepdims=True), 1e-6)
            put(S, rr, cc, ob, oo, on)
        if "rosette_boss" in dec:
            u0, v0, u1, v1 = TX.SQUARES["boss"]
            rr, cc = S.rows(v1, v0), S.cols(u0, u1)
            n = rr.stop - rr.start
            du, dv = dec["rosette_boss"]["u"], dec["rosette_boss"]["v"]
            c0, c1 = int(du[0] * tg.base.shape[1]), int(du[1] * tg.base.shape[1])
            r0, r1 = int((1 - dv[1]) * H0), int((1 - dv[0]) * H0)
            b, o, nn = tg.crop(r0, r1, c0, c1)
            put(S, rr, cc, rs(b, n, n), rs(o, n, n), rs(nn, n, n))
    # ---- leather: leather_brown, 4096 -> 1024 px/m (one 0.5 m tile = the 512 px square)
    if "leather_brown" in sets:
        lb = Src("leather_brown"); used.append("leather_brown")
        u0, v0, u1, v1 = TX.SQUARES["leather"]
        rr, cc = S.rows(v1, v0), S.cols(u0, u1)
        n = rr.stop - rr.start
        put(S, rr, cc, rs(lb.base, n, n) * 0.8, rs(lb.orm, n, n), rs(lb.nrm, n, n))
    # ---- etched border band (judge M6): gold rinceau inlaid in the plate steel, engraved outline grooves
    if steel is not None:
        h = S.rows(*TX.BANDS["etch_band"]); hh = h.stop - h.start
        sb_, so_, sn_ = (tile_to(a, S.w, hh) for a in steelF)
        if "gold_worn" in sets:
            gw = Src("gold_worn"); used.append("gold_worn")
            gb_, go_, gn_ = (tile_to(a, S.w, hh) for a in (gw.base, gw.orm, gw.nrm))
        else:
            gb_, gr_, _ = gold_surface(hh, S.w, 98)
            go_ = np.stack([np.ones_like(gr_), gr_, np.ones_like(gr_)], -1); gn_ = np.dstack([0 * gr_, 0 * gr_, 1 + 0 * gr_])
        m, groove = etch_mask(S.w, hh)
        m3 = m[..., None]
        base = sb_ * (1 - m3) + gb_ * m3
        base = base * (1 - 0.55 * groove[..., None])
        orm = so_ * (1 - m3) + go_ * m3
        orm[..., 0] = orm[..., 0] * (1 - 0.5 * groove)
        orm[..., 1] = np.clip(orm[..., 1] + 0.18 * groove, 0, 1)
        orm[..., 2] = np.where(groove > 0.5, 0.6, orm[..., 2])
        ne = normal_from_height(-1.8 * groove + 0.6 * m, 0.45)
        nn = sn_ * (1 - m3) + gn_ * m3 + (ne - np.array([0, 0, 1.0], np.float32))
        nn /= np.maximum(np.linalg.norm(nn, axis=-1, keepdims=True), 1e-6)
        put(S, h, slice(0, S.w), base, orm, nn)
    print("TEX composited from", used)
    return used


def main():
    os.makedirs(TX.TEX_DIR, exist_ok=True)
    S = build_armour()
    S.nrm_over = None
    compose_from_materials(S)
    save_set(S, "armour", 0.12)
    save_set(build_mail(), "mail", 0.25)
    build_plume()
    previews()


if __name__ == "__main__":
    main()
