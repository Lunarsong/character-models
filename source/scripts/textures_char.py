#!/usr/bin/env python3
"""Procedural PBR texture sets for the RTS characters (knight kit and reusable outfit materials).

System python3 + numpy + scipy + Pillow. Deterministic (fixed seeds). All art is generated here or drawn here as vector
paths (the heraldry in heraldry.py), so every output is our own (CC0-compatible).

usage:  python3 scripts/textures_char.py [set ...] [--res 2048] [--out DIR] [--no-sheet]
        (no set = all sets; sets: see SETS at the bottom)

Outputs per set, in assets/textures/ (all 2048 x 2048 at the default --res):
  <set>_base.png    base colour, sRGB (RGBA when the set has an alpha: alpha = coverage / cut-out)
  <set>_normal.png  tangent-space normal, OpenGL / glTF convention (+Y = up in the image = +V), linear
  <set>_orm.png     R = ambient occlusion, G = roughness, B = metallic (glTF occlusion + metallicRoughness), linear
  <set>_height.png  16-bit height (0 = lowest, 65535 = highest; range in mm in the manifest), for bakes / parallax
and assets/textures/textures_char.json (manifest: physical tile sizes, texel densities, trim strips, atlas regions,
suggested material parameters). UV conventions are documented in STATUS.md ("Agent: materials / textures").
"""
import os, sys, json, math, time
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
CH = os.path.dirname(HERE)
TEX = os.path.join(CH, "assets", "textures")
RENDERS = os.path.join(CH, "renders")
RES = 2048
MANIFEST = {}


def P(*a):
    print("[tex]", *a, flush=True)


# ---------------------------------------------------------------------------------------------------------- basics
def ss(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def lerp(a, b, t):
    return a + (b - a) * t


def col(*c):
    return np.array(c, np.float32)


def mix(a, b, t):
    """a, b: (...,3) or broadcastable colours; t: (...) weight of b."""
    t = np.asarray(t, np.float32)
    return a * (1.0 - t[..., None]) + b * t[..., None]


def noise(h, w, beta=2.0, seed=0, ax=1.0, ay=1.0, lo=0.0, hi=None, rot=0.0, dims=(1.0, 1.0)):
    """Periodic (tileable) 1/f^beta noise, zero mean, unit std.
    Frequencies are in cycles per metre when dims = (width_m, height_m) of the tile, else cycles per tile.
    ax / ay > 1 penalise frequencies along x / y -> features stretched along that axis (ax = 40: streaks along U).
    rot rotates the anisotropy axes (radians). lo / hi: soft band limits (same units)."""
    g = np.random.default_rng(seed)
    F = np.fft.fft2(g.standard_normal((h, w)).astype(np.float32))
    fy = (np.fft.fftfreq(h) * h / dims[1])[:, None].astype(np.float32)
    fx = (np.fft.fftfreq(w) * w / dims[0])[None, :].astype(np.float32)
    if rot:
        c, s = math.cos(rot), math.sin(rot)
        fx, fy = c * fx + s * fy, -s * fx + c * fy
    f = np.sqrt((fx * ax) ** 2 + (fy * ay) ** 2)
    f[0, 0] = 1.0
    amp = f ** (-beta / 2.0)
    if lo:
        amp = amp * (1.0 - np.exp(-(f / lo) ** 4))
    if hi:
        amp = amp * np.exp(-(f / hi) ** 2)
    amp[0, 0] = 0.0
    o = np.real(np.fft.ifft2(F * amp)).astype(np.float32)
    return o / (o.std() + 1e-12)


def blur(a, sigma, wrap=True):
    return ndimage.gaussian_filter(a, sigma, mode="wrap" if wrap else "nearest")


def height_to_normal(h_m, px_x, px_y=None, wrap=True, strength=1.0):
    """h_m: height in metres; px_x / px_y: texel size in metres. Returns float normal (H,W,3) in [-1,1], OpenGL +Y."""
    px_y = px_x if px_y is None else px_y
    if wrap:
        dx = (np.roll(h_m, -1, 1) - np.roll(h_m, 1, 1)) / (2 * px_x)
        drow = (np.roll(h_m, -1, 0) - np.roll(h_m, 1, 0)) / (2 * px_y)
    else:
        drow, dx = np.gradient(h_m, px_y, px_x)
    dx = dx * strength; drow = drow * strength
    n = np.stack([-dx, drow, np.ones_like(dx)], -1)      # row index grows downward (= -V): +Y up in the image
    return n / np.linalg.norm(n, axis=-1, keepdims=True)


def to8(a, dither=False, seed=0):
    a = np.asarray(a, np.float32) * 255.0
    if dither:
        a = a + np.random.default_rng(seed).uniform(-0.5, 0.5, a.shape).astype(np.float32)
    return np.clip(a + 0.5, 0, 255).astype(np.uint8)


def cavity(h, sigma):
    """> 0 where the surface is below its local mean (crevices), in height units."""
    return np.maximum(blur(h, sigma) - h, 0.0)


def fill_rgb_from_alpha(rgb, alpha, iters=24):
    """Bleed colour into transparent texels so mip-mapping / filtering does not pull in black halos."""
    rgb = rgb.copy(); m = alpha > 0.02
    acc = rgb * m[..., None]; wgt = m.astype(np.float32)
    for s in (1, 2, 4, 8, 16, 32):
        a2 = ndimage.uniform_filter(acc, (2 * s + 1, 2 * s + 1, 1), mode="wrap")
        w2 = ndimage.uniform_filter(wgt, 2 * s + 1, mode="wrap")
        fillm = (~m) & (w2 > 1e-4)
        rgb[fillm] = a2[fillm] / w2[fillm][:, None]
        m = m | fillm; acc = rgb * m[..., None]; wgt = m.astype(np.float32)
        if m.all():
            break
    return rgb


def save_set(name, base, normal, ao, rough, metal, height=None, height_mm=None, alpha=None, meta=None, bleed=True):
    os.makedirs(TEX, exist_ok=True)
    base = np.clip(base, 0, 1)
    if alpha is not None and bleed:
        base = fill_rgb_from_alpha(base, alpha)
        img = np.dstack([to8(base), to8(alpha)])
    else:
        img = to8(base)
    Image.fromarray(img).save(os.path.join(TEX, f"{name}_base.png"), optimize=True)
    Image.fromarray(to8(normal * 0.5 + 0.5, dither=True, seed=7)).save(os.path.join(TEX, f"{name}_normal.png"), optimize=True)
    Image.fromarray(np.dstack([to8(ao), to8(rough), to8(metal)])).save(os.path.join(TEX, f"{name}_orm.png"), optimize=True)
    files = {"base": f"{name}_base.png", "normal": f"{name}_normal.png", "orm": f"{name}_orm.png"}
    if height is not None:
        lo, hi = float(height.min()), float(height.max())
        h16 = np.clip((height - lo) / max(hi - lo, 1e-12) * 65535 + 0.5, 0, 65535).astype(np.uint16)
        Image.fromarray(h16).save(os.path.join(TEX, f"{name}_height.png"))
        files["height"] = f"{name}_height.png"
        height_mm = [lo * 1000, hi * 1000]
    m = dict(meta or {})
    m["files"] = files
    m["res"] = [int(base.shape[1]), int(base.shape[0])]
    if height_mm is not None:
        m["height_range_mm"] = [round(height_mm[0], 4), round(height_mm[1], 4)]
    m["alpha"] = alpha is not None
    MANIFEST[name] = m
    P(f"saved {name}: base{' RGBA' if alpha is not None else ''} / normal / orm" + (" / height" if height is not None else ""),
      f"rough {float(rough.mean()):.2f} metal {float(metal.mean()):.2f} ao {float(ao.mean()):.2f}")


# ----------------------------------------------------------------------------------------- strokes (scratches etc)
def polyline_dist(pts, rad, shape, wrap=True):
    """Distance field of a polyline (pixel coords, x right / y down) inside its bbox.
    Returns (rows, cols, d, s): index vectors (wrapped when wrap) and, on the bbox grid, the distance (px) and the
    normalised arc-length position 0..1 of the closest point. None if the bbox is empty."""
    h, w = shape
    x0 = int(math.floor(pts[:, 0].min() - rad - 1)); x1 = int(math.ceil(pts[:, 0].max() + rad + 1))
    y0 = int(math.floor(pts[:, 1].min() - rad - 1)); y1 = int(math.ceil(pts[:, 1].max() + rad + 1))
    if not wrap:
        x0, y0 = max(x0, 0), max(y0, 0); x1, y1 = min(x1, w - 1), min(y1, h - 1)
        if x1 < x0 or y1 < y0:
            return None
    X, Y = np.meshgrid(np.arange(x0, x1 + 1, dtype=np.float32) + 0.5, np.arange(y0, y1 + 1, dtype=np.float32) + 0.5)
    seg = np.diff(pts, axis=0)
    L = np.hypot(seg[:, 0], seg[:, 1]); cum = np.concatenate([[0], np.cumsum(L)]); tot = max(cum[-1], 1e-6)
    d = np.full(X.shape, 1e9, np.float32); s = np.zeros(X.shape, np.float32)
    for i in range(len(seg)):
        ax, ay = pts[i]; bx, by = seg[i]
        l2 = max(bx * bx + by * by, 1e-9)
        t = np.clip(((X - ax) * bx + (Y - ay) * by) / l2, 0, 1)
        di = np.hypot(X - ax - t * bx, Y - ay - t * by)
        m = di < d
        d = np.where(m, di, d); s = np.where(m, (cum[i] + t * L[i]) / tot, s)
    rows = np.arange(y0, y1 + 1); cols = np.arange(x0, x1 + 1)
    if wrap:
        rows = rows % h; cols = cols % w
    return rows, cols, d, s


def wobbly_line(g, p0, ang, length, curve=0.1, n=10):
    """Slightly curved polyline starting at p0 (px) with heading ang, total length (px)."""
    t = np.linspace(0, 1, n)
    bend = g.normal(0, curve)
    a = ang + bend * (t - 0.5) * 2 + g.normal(0, curve * 0.15, n).cumsum() * 0.3
    step = length / (n - 1)
    x = p0[0] + np.concatenate([[0], np.cumsum(np.cos(a[:-1]) * step)])
    y = p0[1] + np.concatenate([[0], np.cumsum(np.sin(a[:-1]) * step)])
    return np.stack([x, y], -1).astype(np.float32)


def scratch_field(shape, g, count, len_px, width_px, depth_m, dir_ang=0.0, dir_spread=0.4, random_frac=0.3,
                  burr=0.25, curve=0.12, wrap=True):
    """Returns (height_m, mask_fresh, mask_any) from `count` scratches (grooves with a small raised burr)."""
    h, w = shape
    H = np.zeros(shape, np.float32); M = np.zeros(shape, np.float32)
    for _ in range(count):
        L = g.uniform(*len_px); wd = g.uniform(*width_px); dp = g.uniform(*depth_m)
        ang = g.uniform(0, 2 * math.pi) if g.random() < random_frac else dir_ang + g.normal(0, dir_spread) + (math.pi if g.random() < .5 else 0)
        p0 = np.array([g.uniform(0, w), g.uniform(0, h)], np.float32)
        pts = wobbly_line(g, p0, ang, L, curve)
        r = polyline_dist(pts, wd * 1.6 + 1, shape, wrap)
        if r is None:
            continue
        rows, cols, d, s = r
        taper = np.sin(np.clip(s, 0, 1) * math.pi) ** 0.6              # fades in / out along the scratch
        u = d / (wd * 0.5 + 0.35)
        groove = np.where(u < 1, (1 - u * u), 0) * taper
        lip = np.where((u >= 0.7) & (u < 1.6), np.sin((u - 0.7) / 0.9 * math.pi), 0) * burr * taper
        sub = H[np.ix_(rows, cols)]
        H[np.ix_(rows, cols)] = np.minimum(sub, -dp * groove) + dp * lip * (sub > -1e-9)
        mm = M[np.ix_(rows, cols)]
        M[np.ix_(rows, cols)] = np.maximum(mm, np.clip(groove * 1.5, 0, 1))
    return H, M


def dimples(shape, g, count, rad_px, depth_m, wrap=True):
    """Shallow round dents (hammer blows / impacts)."""
    h, w = shape
    H = np.zeros(shape, np.float32)
    for _ in range(count):
        r = g.uniform(*rad_px); dp = g.uniform(*depth_m)
        cx, cy = g.uniform(0, w), g.uniform(0, h)
        R = int(r * 2.2) + 2
        rows = np.arange(int(cy) - R, int(cy) + R + 1); cols = np.arange(int(cx) - R, int(cx) + R + 1)
        Y, X = np.meshgrid(rows + 0.5 - cy, cols + 0.5 - cx, indexing="ij")
        e = g.uniform(0.6, 1.0); a = g.uniform(0, math.pi)
        xr = X * math.cos(a) + Y * math.sin(a); yr = (-X * math.sin(a) + Y * math.cos(a)) / e
        q = (xr * xr + yr * yr) / (r * r)
        prof = -dp * np.exp(-q * 2.2) + dp * 0.12 * np.exp(-(np.sqrt(q) - 1.1) ** 2 * 8)
        if wrap:
            rows, cols = rows % h, cols % w
        H[np.ix_(rows, cols)] += prof.astype(np.float32)
    return H


SETS = {}


def register(name, fn):
    SETS[name] = fn


# ============================================================================================ metals
def steel_maps(tile_m=0.5, seed=101, base_rgb=(0.64, 0.65, 0.68), blued=False, name="steel"):
    """Worn, brushed plate steel maps (n x n tile): returns base, normal, ao, rough, metal, height."""
    n = RES; px = tile_m / n; g = np.random.default_rng(seed); dims = (tile_m, tile_m)
    P(name, "height")
    # forms: faint hammered undulation (reads in reflections only) + impacts
    h = noise(n, n, 3.6, seed + 1, dims=dims, lo=4) * 60e-6
    h += dimples((n, n), g, 26, (12, 46), (20e-6, 90e-6))
    # brushing: long streaks along U, two scales
    brush = noise(n, n, 1.0, seed + 2, ax=45, dims=dims) * 0.6 + noise(n, n, 1.4, seed + 3, ax=12, dims=dims) * 0.4
    brush_fine = noise(n, n, 0.6, seed + 4, ax=80, dims=dims)
    h += brush * 2.5e-6 + brush_fine * 1.2e-6
    # scratches: fine ones along the brushing, fewer random ones, a few deep gouges with burrs
    s1, m1 = scratch_field((n, n), g, 520, (25, 240), (0.6, 1.3), (1.5e-6, 5e-6), dir_spread=0.35, random_frac=0.25)
    s2, m2 = scratch_field((n, n), g, 90, (40, 420), (1.2, 2.6), (6e-6, 16e-6), dir_spread=0.6, random_frac=0.45)
    s3, m3 = scratch_field((n, n), g, 10, (60, 260), (2.5, 4.5), (25e-6, 50e-6), dir_spread=1.0, random_frac=0.6, burr=0.4)
    h += s1 + s2 + s3
    # pits (old corrosion, cleaned)
    pits = np.zeros((n, n), np.float32)
    pm = ss(1.4, 2.6, noise(n, n, 2.2, seed + 5, dims=dims)) * (g.random((n, n)) < 0.004)
    pits = blur(pm.astype(np.float32), 0.9) * -40e-6
    h += pits
    P(name, "colour")
    fresh = np.clip(m1 * 0.5 + m2 * 0.8 + m3, 0, 1)
    # grime is mostly a geometry thing (crevices: baked AO / vertex AO on the piece); the tile only carries a very
    # faint broad breakup so large plates do not look CG-perfect, plus dirt caught in its own dents and gouges
    grime = ss(1.2, 2.6, noise(n, n, 2.6, seed + 6, dims=dims, lo=2)) * (0.5 + 0.5 * ss(-0.2, 1.4, noise(n, n, 1.6, seed + 7, dims=dims)))
    cav = cavity(h, 6)
    crev = np.clip(cav / 18e-6, 0, 1)
    dirt = np.clip(grime * 0.15 + crev * 0.55, 0, 1)
    smudge = noise(n, n, 2.4, seed + 10, dims=dims, lo=3)                                   # finger / oil smudges
    mottle = noise(n, n, 3.0, seed + 8, dims=dims, lo=2) * 0.010 + brush * 0.014
    base = col(*base_rgb)[None, None, :] * (1.0 + mottle)[..., None]
    if blued:
        # blued / oxide-lacquered plate: blue metal, rubbed to bright steel in scratches and on a few worn patches
        rub = np.clip(ss(1.8, 2.9, noise(n, n, 1.8, seed + 9, dims=dims)) * 0.3 + fresh * 0.45, 0, 1)
        base = mix(base, col(0.62, 0.66, 0.74)[None, None, :], rub)
    else:
        base = mix(base, col(0.86, 0.87, 0.88)[None, None, :], fresh * 0.45)
    base = mix(base, col(0.30, 0.27, 0.23)[None, None, :], dirt * 0.5)
    base = mix(base, col(0.36, 0.20, 0.10)[None, None, :], np.clip(-pits / 40e-6, 0, 1) * 0.6)
    rough = 0.27 + brush * 0.035 + brush_fine * 0.025 + smudge * 0.012
    rough = rough - fresh * 0.08 + dirt * 0.22
    metal = 1.0 - dirt * 0.2 - np.clip(-pits / 40e-6, 0, 1) * 0.5
    ao = 1.0 - np.clip(cav / 30e-6, 0, 1) * 0.35
    nrm = height_to_normal(h, px)
    return base, nrm, ao, np.clip(rough, 0.05, 1), np.clip(metal, 0, 1), h


def steel_set(name="steel_worn", tile_m=0.5, seed=101, base_rgb=(0.64, 0.65, 0.68), blued=False):
    """Worn, brushed plate steel. Brushing runs along U (image x): align U with a plate's long axis."""
    n = RES
    base, nrm, ao, rough, metal, h = steel_maps(tile_m, seed, base_rgb, blued, name)
    save_set(name, base, nrm, ao, rough, metal, height=h,
             meta=dict(kind="tileable", tile_m=[tile_m, tile_m], px_per_m=n / tile_m, uv_per_m=1 / tile_m,
                       grain="brushing along U (image x)",
                       material=dict(metallic=1, roughness_from="orm.G", alpha="OPAQUE")))


def gold_set(name="gold_worn", tile_m=0.5, seed=202, base_rgb=(0.86, 0.68, 0.38)):
    """Polished, slightly worn gold / brass: polishing swirls, nicks, tarnish in low areas."""
    n = RES; px = tile_m / n; g = np.random.default_rng(seed); dims = (tile_m, tile_m)
    P(name, "height")
    h = noise(n, n, 3.8, seed + 1, dims=dims, lo=4) * 45e-6
    h += dimples((n, n), g, 18, (10, 36), (10e-6, 50e-6))
    s1, m1 = scratch_field((n, n), g, 1400, (10, 70), (0.5, 0.9), (0.6e-6, 2.0e-6), random_frac=1.0, curve=0.9)  # swirls
    s2, m2 = scratch_field((n, n), g, 70, (30, 220), (1.0, 2.2), (4e-6, 12e-6), random_frac=0.8, curve=0.3)
    s3, m3 = scratch_field((n, n), g, 6, (40, 160), (2.0, 3.5), (15e-6, 35e-6), random_frac=1.0, burr=0.5)
    h += s1 + s2 + s3
    P(name, "colour")
    cav = cavity(h, 8)
    tarn = np.clip(ss(1.0, 2.4, noise(n, n, 2.4, seed + 5, dims=dims, lo=2)) * 0.30 + np.clip(cav / 15e-6, 0, 1) * 0.6, 0, 1)
    tarn *= 0.6 + 0.4 * ss(-0.5, 1.2, noise(n, n, 1.4, seed + 6, dims=dims))
    mottle = noise(n, n, 2.0, seed + 7, dims=dims)
    base = col(*base_rgb)[None, None, :] * (1 + mottle * 0.02)[..., None]
    base = mix(base, col(0.97, 0.84, 0.55)[None, None, :], np.clip(m2 * 0.6 + m3, 0, 1) * 0.6)    # fresh scratches
    base = mix(base, col(0.45, 0.33, 0.16)[None, None, :], tarn * 0.6)
    rough = 0.25 + noise(n, n, 2.4, seed + 8, dims=dims, lo=3) * 0.03 + m1 * 0.05 + tarn * 0.25 - m3 * 0.05
    metal = 1.0 - tarn * 0.12
    ao = 1.0 - np.clip(cav / 25e-6, 0, 1) * 0.3
    save_set(name, base, height_to_normal(h, px), ao, np.clip(rough, 0.05, 1), np.clip(metal, 0, 1), height=h,
             meta=dict(kind="tileable", tile_m=[tile_m, tile_m], px_per_m=n / tile_m, uv_per_m=1 / tile_m,
                       material=dict(metallic=1, alpha="OPAQUE")))


# ============================================================================================ cloth
def twill_height(h, w, threads_x, threads_y, seed, slub=0.12, felt=0.35):
    """2/2 twill weave height (0..1) + warp / weft ids. threads_* = thread count across the image (integers -> tiles)."""
    g = np.random.default_rng(seed)
    y = (np.arange(h, dtype=np.float32) + 0.5) / h * threads_y
    x = (np.arange(w, dtype=np.float32) + 0.5) / w * threads_x
    X, Y = np.meshgrid(x, y)
    j = np.floor(X).astype(np.int64); i = np.floor(Y).astype(np.int64)
    fx = X - j; fy = Y - i
    # thread thickness irregularity (slubs) per thread, along its length
    wj = g.normal(0, 1, threads_x).astype(np.float32); wi = g.normal(0, 1, threads_y).astype(np.float32)
    sl_x = noise(h, w, 2.0, seed + 1, ax=0.05, ay=1.0) * slub                   # varies along y (warp slubs)
    sl_y = noise(h, w, 2.0, seed + 2, ax=1.0, ay=0.05) * slub                   # varies along x (weft slubs)
    across_warp = np.sqrt(np.clip(1 - ((fx - 0.5) / (0.5 * (1 + sl_x * 0.5))) ** 2, 0, 1))
    across_weft = np.sqrt(np.clip(1 - ((fy - 0.5) / (0.5 * (1 + sl_y * 0.5))) ** 2, 0, 1))
    # over / under: warp on top where (i + j) mod 4 < 2 (diagonal twill); smooth along the float
    def over(ii, jj):
        return (((ii + jj) % 4) < 2).astype(np.float32) * 2 - 1
    # warp float state sampled at cell centre and neighbours along y -> smooth interpolation
    o0 = over(i, j); o1 = over(i + 1, j); om = over(i - 1, j)
    t = fy
    s_warp = np.where(t >= 0.5, lerp(o0, o1, ss(0.5, 1.5, t + 0.0)), lerp(om, o0, ss(-0.5, 0.5, t)))
    q0 = over(i, j); q1 = over(i, j + 1); qm = over(i, j - 1)
    s_weft = -np.where(fx >= 0.5, lerp(q0, q1, ss(0.5, 1.5, fx)), lerp(qm, q0, ss(-0.5, 0.5, fx)))
    warp_h = across_warp * (0.55 + 0.45 * s_warp) * (1 + wj[j] * 0.04)
    weft_h = across_weft * (0.55 + 0.45 * s_weft) * (1 + wi[i] * 0.04)
    top_is_warp = warp_h >= weft_h
    hgt = np.maximum(warp_h, weft_h)
    # felting / fulling (heavy wool): soften the weave and add fibre fuzz
    hgt = lerp(hgt, blur(hgt, 1.2), felt)
    return hgt.astype(np.float32), top_is_warp, j, i


def cloth_colour_field(h, w, seed, rgb, top_is_warp, j, i, dims, wear=0.5):
    g = np.random.default_rng(seed)
    base = col(*rgb)
    tx = int(j.max() + 1); ty = int(i.max() + 1)
    warp_v = g.normal(0, 1, tx).astype(np.float32); weft_v = g.normal(0, 1, ty).astype(np.float32)
    thread = np.where(top_is_warp, warp_v[j] * 0.035 + 0.02, weft_v[i] * 0.035 - 0.02)   # heathered yarn, 2 tones
    heather = noise(h, w, 1.2, seed + 3, dims=dims) * 0.025
    fade = ss(0.2, 1.8, noise(h, w, 2.5, seed + 4, dims=dims, lo=3)) * wear
    c = base[None, None, :] * (1 + thread + heather)[..., None]
    c = mix(c, col(0.28, 0.33, 0.50)[None, None, :], fade * 0.22)              # sun / wear fading, desaturated
    return c, fade


def cloth_set(name="cloth_blue", tile_m=0.25, seed=303, rgb=(0.064, 0.112, 0.325), threads=240, wear=0.6):
    """Heavy fulled wool twill. Tile 0.25 m, ~1 mm threads. U / V follow the weft / warp (V = grain, down the garment)."""
    n = RES; px = tile_m / n; dims = (tile_m, tile_m); g = np.random.default_rng(seed)
    P(name, "weave")
    wv, top_warp, j, i = twill_height(n, n, threads, threads, seed)
    fuzz = noise(n, n, 0.8, seed + 5, dims=dims) * 0.10
    lumps = noise(n, n, 2.8, seed + 6, dims=dims, lo=6) * 0.35                      # cloth thickness variation
    pill = (ss(2.3, 3.2, noise(n, n, 1.0, seed + 7, dims=dims)) * 0.5)
    h01 = wv + fuzz + pill
    h = h01 * 0.35e-3 + lumps * 0.08e-3                                              # threads ~0.35 mm relief
    P(name, "colour")
    c, fade = cloth_colour_field(n, n, seed + 8, rgb, top_warp, j, i, dims, wear)
    cav = np.clip((blur(wv, 2.0) - wv) * 2.5, 0, 1)
    c = c * (1 - cav * 0.35)[..., None]
    lint = ss(2.8, 3.6, noise(n, n, 0.6, seed + 9, dims=dims))
    c = mix(c, col(0.55, 0.58, 0.66)[None, None, :], lint * 0.35 + pill * 0.2)
    rough = 0.86 + noise(n, n, 1.6, seed + 10, dims=dims) * 0.03 + cav * 0.08 - wv * 0.04 + fade * 0.03
    ao = 1 - cav * 0.45
    save_set(name, c, height_to_normal(h, px), ao, np.clip(rough, 0, 1), np.zeros((n, n), np.float32), height=h,
             meta=dict(kind="tileable", tile_m=[tile_m, tile_m], px_per_m=n / tile_m, uv_per_m=1 / tile_m,
                       grain="warp along V (down the garment), weft along U",
                       material=dict(metallic=0, sheen=dict(color=[0.25, 0.35, 0.75], roughness=0.5), alpha="OPAQUE")))
    return c


# ============================================================================================ leather
def voronoi_f1f2(h, w, cell_px, seed, jitter=0.9):
    """Periodic Worley F1 / F2 (px) on a jittered grid (cell_px spacing)."""
    g = np.random.default_rng(seed)
    gx = max(1, int(round(w / cell_px))); gy = max(1, int(round(h / cell_px)))
    sx, sy = w / gx, h / gy
    cx, cy = np.meshgrid((np.arange(gx) + 0.5) * sx, (np.arange(gy) + 0.5) * sy)
    pts = np.stack([cx.ravel() + g.uniform(-.5, .5, gx * gy) * sx * jitter,
                    cy.ravel() + g.uniform(-.5, .5, gx * gy) * sy * jitter], -1)
    tiles = [pts + np.array([dx * w, dy * h]) for dx in (-1, 0, 1) for dy in (-1, 0, 1)]
    tree = cKDTree(np.concatenate(tiles))
    X, Y = np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5)
    d, idx = tree.query(np.stack([X.ravel(), Y.ravel()], -1), k=2, workers=-1)
    return d[:, 0].reshape(h, w).astype(np.float32), d[:, 1].reshape(h, w).astype(np.float32), (idx[:, 0] % (gx * gy)).reshape(h, w)


def leather_field(h, w, px, seed, dims, rgb=(0.205, 0.125, 0.080), grain_mm=1.3, wear=1.0):
    """Returns (height_m, base, rough, ao) for a pebble-grain dark brown leather (tileable)."""
    g = np.random.default_rng(seed)
    f1, f2, cid = voronoi_f1f2(h, w, grain_mm * 1e-3 / px, seed)
    edge = f2 - f1
    cellh = ss(0.0, 2.2, edge)                                                   # pebble domes, valleys between
    f1b, f2b, _ = voronoi_f1f2(h, w, grain_mm * 3.2e-3 / px, seed + 1)
    big = ss(0.0, 5.0, f2b - f1b)
    wrinkle = np.abs(noise(h, w, 2.2, seed + 2, ax=5, rot=0.5, dims=dims))            # creases (ridged)
    wrinkle2 = np.abs(noise(h, w, 2.4, seed + 3, ax=3, rot=-0.9, dims=dims))
    crease = np.clip(1 - np.minimum(wrinkle, wrinkle2 * 1.2) * 2.5, 0, 1) ** 3
    pores = (g.random((h, w)) < 0.02).astype(np.float32); pores = blur(pores, 0.7) * 3
    hm = (cellh * 0.55 + big * 0.35) * 40e-6 - crease * 60e-6 - pores * 8e-6
    hm += noise(h, w, 3.2, seed + 4, dims=dims, lo=3) * 50e-6
    scuff = ss(0.9, 2.2, noise(h, w, 2.2, seed + 5, dims=dims, lo=3)) * wear
    scr, sm = scratch_field((h, w), g, 60, (15, 140), (0.8, 2.0), (5e-6, 18e-6), random_frac=0.7, curve=0.5)
    hm += scr
    tone = noise(h, w, 2.4, seed + 6, dims=dims) * 0.06
    base = col(*rgb)[None, None, :] * (1 + tone + (cellh - 0.5) * 0.10 - crease * 0.25 - pores * 0.2)[..., None]
    base = mix(base, col(0.29, 0.22, 0.17)[None, None, :], np.clip(scuff * 0.3 + sm * 0.5, 0, 1))
    base = mix(base, col(0.16, 0.09, 0.055)[None, None, :], ss(0.6, 1.6, noise(h, w, 2.0, seed + 7, dims=dims)) * 0.35)
    rough = 0.58 - (cellh - 0.5) * 0.08 + crease * 0.10 + scuff * 0.18 + sm * 0.1 + noise(h, w, 2, seed + 8, dims=dims) * 0.03
    ao = 1 - crease * 0.35 - (1 - cellh) * 0.10 - pores * 0.2
    return hm, base, rough, ao


def leather_set(name="leather_brown", tile_m=0.5, seed=404):
    n = RES; px = tile_m / n; dims = (tile_m, tile_m)
    P(name)
    hm, base, rough, ao = leather_field(n, n, px, seed, dims)
    save_set(name, base, height_to_normal(hm, px), np.clip(ao, 0, 1), np.clip(rough, 0, 1), np.zeros((n, n), np.float32),
             height=hm, meta=dict(kind="tileable", tile_m=[tile_m, tile_m], px_per_m=n / tile_m, uv_per_m=1 / tile_m,
                                  material=dict(metallic=0, alpha="OPAQUE")))


# ============================================================================================ mail
def mail_set(name="mail_riveted", ring_pitch_m=0.0105, cols=16, seed=505):
    """European 4-in-1 riveted mail. Rows run along U; alternate rows lean opposite ways. Alpha = ring coverage.
    Tile = cols x ring pitch square (16 x 10.5 mm = 0.168 m)."""
    n = RES; g = np.random.default_rng(seed)
    tile_m = cols * ring_pitch_m; px = tile_m / n
    p = n / cols                      # px per ring column
    rows = cols * 2                   # rows at half pitch
    q = n / rows
    Rc = 0.41 * p                     # centre-line radius (px)
    rw = 0.088 * p                    # wire half-width (px; flattened riveted wire)
    tilt = math.radians(28)
    P(name, f"rings {cols}x{rows}, pitch {ring_pitch_m*1000:.1f} mm, tile {tile_m:.3f} m")
    Hbuf = np.full((n, n), -1e9, np.float32)       # height (px units)
    ring_id = np.full((n, n), -1, np.int32)
    ringlit = np.zeros((n, n), np.float32)          # ring-local angle for rivet placement
    Rpad = int(Rc + rw + 3)
    oy, ox = np.meshgrid(np.arange(-Rpad, Rpad + 1), np.arange(-Rpad, Rpad + 1), indexing="ij")
    rid = 0
    for r in range(rows):
        lean = 1 if r % 2 == 0 else -1
        for c in range(cols):
            cx = (c + 0.5 + (0.5 if r % 2 else 0.0)) * p + g.normal(0, 0.012 * p)
            cy = (r + 0.5) * q + g.normal(0, 0.012 * p)
            ang = g.normal(0, 0.06)                       # per-ring slight rotation
            X = ox + (int(cx) - cx) + 0.5; Y = oy + (int(cy) - cy) + 0.5
            ca, sa = math.cos(ang), math.sin(ang)
            xr = X * ca + Y * sa; yr = -X * sa + Y * ca
            # ring plane tilted about the vertical axis: projected ellipse, semi-axes Rc*cos(tilt) (x), Rc (y)
            a = Rc * math.cos(tilt) * (1 + g.normal(0, 0.02))
            b = Rc * (1 + g.normal(0, 0.02))
            th = np.arctan2(yr / b, xr / a)
            ex = a * np.cos(th); ey = b * np.sin(th)
            d = np.hypot(xr - ex, yr - ey)                # ~distance to the centre-line ellipse
            inside = d < rw
            prof = np.sqrt(np.clip(1 - (d / rw) ** 2, 0, 1)) * rw * 0.62
            # height along the ring: tilted plane (one side raised) -> over / under interlock with neighbours
            zc = lean * xr * math.tan(tilt) * 0.55   # ring plane tilted about the vertical: opposite leans interlock
            z = np.where(inside, zc + prof, -1e9)
            rr = (np.arange(int(cy) - Rpad, int(cy) + Rpad + 1)) % n
            cc = (np.arange(int(cx) - Rpad, int(cx) + Rpad + 1)) % n
            sub = Hbuf[np.ix_(rr, cc)]
            upd = z > sub
            sub = np.where(upd, z, sub); Hbuf[np.ix_(rr, cc)] = sub
            ids = ring_id[np.ix_(rr, cc)]; ring_id[np.ix_(rr, cc)] = np.where(upd, rid, ids)
            lt = ringlit[np.ix_(rr, cc)]
            # rivet: the flattened overlap at the upper-outer side of each ring (angle ~ +60 deg or 120 deg by lean)
            rv_th = math.radians(62 if lean > 0 else 118)
            dth = np.angle(np.exp(1j * (th - (-rv_th))))
            riv = np.exp(-(dth / 0.13) ** 2) * inside * np.sqrt(np.clip(1 - (d / rw) ** 2, 0, 1))
            ringlit[np.ix_(rr, cc)] = np.where(upd, riv, lt)
            rid += 1
    cov = (Hbuf > -1e8).astype(np.float32)
    zmin = Hbuf[cov > 0].min()
    Hn = np.where(cov > 0, Hbuf - zmin, 0.0)
    Hn = Hn + ringlit * rw * 0.55                                       # rivet heads on the flattened overlap
    # soft ground under the mail (padding) for the normal map, far below
    ground = -rw * 0.5
    Hpx = np.where(cov > 0, Hn, ground)
    Hs = ndimage.gaussian_filter(Hpx, 0.7, mode="wrap")
    h_m = Hs * px
    nrm = height_to_normal(h_m, px)
    # AO: occlusion from neighbours (height below the local max) + dark gaps
    local_max = ndimage.maximum_filter(Hpx, size=int(p * 0.35), mode="wrap")
    occl = np.clip((local_max - Hpx) / (Rc * 0.6), 0, 1)
    ao = np.clip(1 - occl * 0.7, 0.1, 1) * (0.35 + 0.65 * ndimage.gaussian_filter(cov, 1.0, mode="wrap"))
    # colour: steel rings, darker where occluded, a little rust in the gaps, per-ring tone variation
    rv = g.normal(0, 1, rid + 1).astype(np.float32)
    tone = np.where(ring_id >= 0, rv[np.clip(ring_id, 0, None)], 0) * 0.07
    dims = (tile_m, tile_m)
    grime = ss(0.4, 1.8, noise(n, n, 2.4, seed + 1, dims=dims)) * 0.6 + occl * 0.6
    rust = ss(1.5, 2.6, noise(n, n, 1.8, seed + 2, dims=dims)) * occl
    base = col(0.56, 0.57, 0.59)[None, None, :] * (1 + tone)[..., None]
    base = base * (1 - ringlit * 0.25)[..., None]
    base = mix(base, col(0.26, 0.24, 0.22)[None, None, :], np.clip(grime, 0, 1) * 0.6)
    base = mix(base, col(0.38, 0.20, 0.09)[None, None, :], np.clip(rust, 0, 1) * 0.7)
    base = mix(base, col(0.05, 0.05, 0.06)[None, None, :], 1 - cov)
    rough = 0.36 + grime * 0.25 + rust * 0.3 + ringlit * -0.05
    rough = np.where(cov > 0, rough, 0.9)
    metal = np.where(cov > 0, 1 - rust * 0.6 - np.clip(grime, 0, 1) * 0.15, 0.0)
    alpha = ndimage.gaussian_filter(cov, 0.5, mode="wrap")
    save_set(name, base, nrm, ao, np.clip(rough, 0, 1), np.clip(metal, 0, 1), height=h_m, alpha=alpha, bleed=False,
             meta=dict(kind="tileable", tile_m=[tile_m, tile_m], px_per_m=n / tile_m, uv_per_m=1 / tile_m,
                       ring_pitch_mm=ring_pitch_m * 1000, rows_along="U", rgb_gaps="dark padding colour (usable OPAQUE)",
                       material=dict(metallic=1, alpha="OPAQUE (dark padding in the gaps) or MASK 0.5 for cut-out edges",
                                     double_sided="only for MASK cards")))


# ============================================================================================ horsehair
def horsehair_set(name="horsehair_blue", rgb=(0.085, 0.15, 0.50), seed=606, cols=8):
    """Plume hair-card atlas, same layout convention as hair_tex.py: 8 columns of 256 px, strands along V, ROOT AT THE
    TOP of the image (V = 1 in Blender), tip at the bottom. Column kinds below."""
    n = RES; W = H = n; CW = W // cols; g = np.random.default_rng(seed)
    kinds = ["solid", "dense", "dense", "lock", "lock", "medium", "wispy", "strands"]
    lum = np.zeros((H, W), np.float32); cov = np.zeros((H, W), np.float32); nx = np.zeros((H, W), np.float32)
    tipl = np.zeros((H, W), np.float32)
    P(name, "strands")
    for c, kind in enumerate(kinds):
        x0 = c * CW
        count = dict(solid=1100, dense=620, lock=420, medium=300, wispy=120, strands=40)[kind]
        nclump = dict(solid=12, dense=6, lock=3, medium=4, wispy=5, strands=6)[kind]
        centres = x0 + CW * (0.5 + np.linspace(-0.34, 0.34, nclump) + g.normal(0, 0.025, nclump))
        wave_a = g.uniform(1.5, 5.0, nclump); wave_f = g.uniform(0.8, 2.2, nclump) * 2 * np.pi / H; wave_p = g.uniform(0, 6.3, nclump)
        for s in range(count):
            k = g.integers(nclump); cc = centres[k]
            spread = dict(solid=0.10, dense=0.075, lock=0.07, medium=0.06, wispy=0.11, strands=0.14)[kind] * CW
            base_x = np.clip(g.normal(cc, spread), x0 + 6, x0 + CW - 7)
            start = g.uniform(0, 0.015) * H
            length = (g.uniform(0.80, 1.0) if kind in ("solid", "dense", "lock") else g.uniform(0.45, 1.0)) * H
            end = min(H - 1, start + length)
            ys = np.arange(int(start), int(end))
            t = np.clip((ys - start) / max(1.0, end - start), 0, 1)
            conv = dict(solid=0.2, dense=0.45, lock=0.7, medium=0.5, wispy=0.3, strands=0.1)[kind]
            xs = base_x + (cc - base_x) * conv * t ** 1.3 + wave_a[k] * np.sin(wave_f[k] * ys + wave_p[k]) * (0.3 + t) \
                + g.uniform(0.3, 1.2) * np.sin(g.uniform(3, 9) * 2 * np.pi * ys / H + g.uniform(0, 6.3))
            width = g.uniform(0.9, 1.7) if kind not in ("wispy", "strands") else g.uniform(0.7, 1.2)
            bright = g.uniform(0.55, 1.0)
            wv = width * (1.0 - 0.6 * t ** 4)
            a = np.clip(1.0 - np.maximum(t - 0.88, 0) / 0.12, 0, 1)
            for dx in range(-3, 4):
                pxs = np.floor(xs).astype(int) + dx
                d = np.abs(pxs + 0.5 - xs) / np.maximum(wv, 0.3)
                inside = d < 1.0
                if not inside.any():
                    continue
                prof = np.sqrt(np.clip(1 - d ** 2, 0, 1))
                py = ys[inside]; pxx = np.clip(pxs[inside], x0, x0 + CW - 1)
                cvi = (prof * a)[inside]
                upd = cvi > cov[py, pxx]
                py, pxx, cvi = py[upd], pxx[upd], cvi[upd]
                cov[py, pxx] = cvi
                tt = t[inside][upd]
                lum[py, pxx] = bright * (0.78 + 0.22 * tt)
                tipl[py, pxx] = tt
                sgn = np.sign(pxs[inside] + 0.5 - xs[inside])[upd]
                nx[py, pxx] = sgn * np.sqrt(np.clip(1 - prof[inside][upd] ** 2, 0, 1)) * 0.85
        u = (np.arange(CW) + 0.5) / CW; v = (np.arange(H) + 0.5) / H
        if kind == "solid":
            blk = cov[:, x0:x0 + CW]
            under = ss(0.0, 0.10, 1.0 - v)[:, None] * np.ones((1, CW))
            lum[:, x0:x0 + CW] = np.where(blk > 0.05, lum[:, x0:x0 + CW], 0.40)
            cov[:, x0:x0 + CW] = np.maximum(blk, under * 0.98)
        else:
            side = ss(0.0, 0.2, u) * ss(0.0, 0.2, 1 - u)
            rootlen = dict(dense=0.05, lock=0.05, medium=0.08, wispy=0.15, strands=0.2)[kind]
            cov[:, x0:x0 + CW] *= side[None, :] * ss(0.0, rootlen, v)[:, None]
    alpha = np.clip(cov * 1.6, 0, 1)
    # sheen streaks along the strands (per strand brightness bands, not a hard fake highlight)
    streak = noise(H, W, 1.2, seed + 3, ax=0.02, ay=1.0) * 0.06
    lumf = lum * (1 + streak)
    rgbc = col(*rgb)
    base = np.clip(lumf[..., None] * rgbc[None, None, :] * 1.45, 0, 1)
    base = mix(base, rgbc[None, None, :] * 0.35, (1 - tipl) * 0.35 * (cov > 0))      # darker toward the root
    nz = np.sqrt(np.clip(1 - nx ** 2, 0, 1))
    nrm = np.stack([nx, np.zeros_like(nx), nz], -1)
    rough = np.where(cov > 0.02, 0.42 + (1 - lum) * 0.12, 0.6).astype(np.float32)
    ao = np.clip(0.55 + 0.45 * np.clip(cov * 1.5, 0, 1), 0, 1) * (0.75 + 0.25 * np.clip(tipl * 3, 0, 1))
    ao = np.where(cov > 0.02, ao, 1.0)
    save_set(name, base, nrm, ao, rough, np.zeros((H, W), np.float32), alpha=alpha,
             meta=dict(kind="hair_cards", columns=[dict(col=i, kind=k, u=[i / cols, (i + 1) / cols]) for i, k in enumerate(kinds)],
                       root="top of the image (Blender V = 1), tips at V = 0", strands_along="V",
                       material=dict(metallic=0, alpha="MASK 0.35 (or BLEND for cutscenes)", double_sided=True,
                                     anisotropy=dict(strength=0.6, rotation_deg=90, note="strands along V"))))


# ============================================================================================ skin detail
def skin_detail_set(name="skin_detail", tile_m=0.06, seed=707):
    """Tileable micro-detail normal for skin (pores + fine cross-hatched lines), 6 cm tile. Normal only (+ a height)."""
    n = RES; px = tile_m / n; dims = (tile_m, tile_m); g = np.random.default_rng(seed)
    P(name)
    f1, f2, _ = voronoi_f1f2(n, n, 0.00055 / px, seed)
    pores = -np.exp(-(f1 / (0.00009 / px)) ** 2) * 18e-6
    cells = ss(0.0, 5.0, f2 - f1) * 5e-6
    lines1 = np.abs(noise(n, n, 1.8, seed + 1, ax=8, rot=0.7, dims=dims))
    lines2 = np.abs(noise(n, n, 1.8, seed + 2, ax=8, rot=-0.8, dims=dims))
    creases = -(np.clip(1 - lines1 * 3, 0, 1) ** 2 + np.clip(1 - lines2 * 3, 0, 1) ** 2) * 6e-6
    h = pores + cells + creases + noise(n, n, 2.5, seed + 3, dims=dims) * 3e-6
    nrm = height_to_normal(h, px)
    os.makedirs(TEX, exist_ok=True)
    Image.fromarray(to8(nrm * 0.5 + 0.5, dither=True)).save(os.path.join(TEX, f"{name}_normal.png"), optimize=True)
    MANIFEST[name] = dict(kind="detail_normal", files=dict(normal=f"{name}_normal.png"), tile_m=[tile_m, tile_m],
                          px_per_m=n / tile_m, res=[n, n],
                          note="engine detail-normal slot (tile 6 cm, i.e. ~16 repeats over a face); not a glTF slot")
    P("saved", name)


# ============================================================================================ heraldry
import heraldry as HR

GOLD_THREAD = (0.84, 0.62, 0.29)
FIELD_BLUE = (0.064, 0.112, 0.325)


def edge_sdf(poly_m, X, Y):
    """Signed distance (m, > 0 inside) to a convex-ish polygon given in metres; also the index of the nearest edge
    and the coordinate along that edge (m)."""
    P = np.asarray(poly_m, float); n = len(P)
    best = np.full(X.shape, 1e9, np.float32); along = np.zeros(X.shape, np.float32); eid = np.zeros(X.shape, np.int32)
    for k in range(n):
        a = P[k]; b = P[(k + 1) % n]; d = b - a; L = math.hypot(*d); t = d / L
        rx = X - a[0]; ry = Y - a[1]
        s = np.clip(rx * t[0] + ry * t[1], 0, L)
        dist = np.hypot(rx - s * t[0], ry - s * t[1])
        m = dist < best
        best = np.where(m, dist, best); along = np.where(m, s + (0 if k == 0 else 0), along); eid = np.where(m, k, eid)
    # inside test (even-odd)
    inside = np.zeros(X.shape, bool)
    for k in range(n):
        a = P[k]; b = P[(k + 1) % n]
        cond = ((a[1] > Y) != (b[1] > Y))
        xint = (b[0] - a[0]) * (Y - a[1]) / (b[1] - a[1] + 1e-12) + a[0]
        inside ^= cond & (X < xint)
    return np.where(inside, best, -best).astype(np.float32), eid, along


class Panel:
    """A cloth panel of physical size w_m x h_m rendered into W x H pixels (x right, y DOWN from the top edge)."""

    def __init__(self, W, H, w_m, h_m, seed, rgb=FIELD_BLUE, weave_px=4.0, wear=0.5, dirt_bottom=0.0):
        self.W, self.H, self.w, self.h, self.seed = W, H, w_m, h_m, seed
        self.px_x, self.px_y = w_m / W, h_m / H
        xs = (np.arange(W, dtype=np.float32) + 0.5) * self.px_x
        ys = (np.arange(H, dtype=np.float32) + 0.5) * self.px_y
        self.X, self.Y = np.meshgrid(xs, ys)
        dims = (w_m, h_m)
        self.dims = dims
        # cloth: fulled wool twill at a pitch the texel density can carry (~weave_px texels per thread)
        pitch = max(self.px_x, self.px_y) * weave_px
        tx = max(8, int(round(w_m / pitch))); ty = max(8, int(round(h_m / pitch)))
        wv, top_warp, j, i = twill_height(H, W, tx, ty, seed, felt=0.55)
        fuzz = noise(H, W, 0.9, seed + 5, dims=dims) * 0.12
        lumps = noise(H, W, 2.8, seed + 6, dims=dims, lo=4) * 0.35
        self.height = (wv * 0.7 + fuzz) * 0.30e-3 + lumps * 0.08e-3
        c, fade = cloth_colour_field(H, W, seed + 8, rgb, top_warp, j, i, dims, wear)
        cav = np.clip((blur(wv, 1.0) - wv) * 2.5, 0, 1)
        self.base = c * (1 - cav * 0.3)[..., None]
        self.rough = (0.86 + noise(H, W, 1.6, seed + 10, dims=dims) * 0.03 + cav * 0.06 + fade * 0.03).astype(np.float32)
        self.metal = np.zeros((H, W), np.float32)
        self.ao = (1 - cav * 0.35).astype(np.float32)
        self.alpha = np.ones((H, W), np.float32)
        if dirt_bottom > 0:
            t = ss(self.h - dirt_bottom, self.h, self.Y) * (0.6 + 0.4 * ss(-0.5, 1.2, noise(H, W, 2.2, seed + 11, dims=dims)))
            self.base = mix(self.base, col(0.13, 0.12, 0.12)[None, None, :], t * 0.5)
            self.rough = self.rough + t * 0.05
        self.gold = np.zeros((H, W), np.float32)

    # ---------------------------------------------------------------- motifs (embroidered gold)
    def to_px(self, pts):
        return np.asarray(pts) / [self.px_x, self.px_y]

    def embroider(self, shapes_m, ss_=4, raise_m=0.55e-3, bevel_m=1.4e-3, cord_m=2.4e-3, pitch_m=1.8e-3,
                  rgb=GOLD_THREAD, row_angle=0.0):
        """shapes in metres (panel coords). Couched gold threads in rows + a raised twisted outline cord."""
        cov, hi = HR.raster(shapes_m, self.W, self.H, ss=ss_, to_px=self.to_px)
        # distance to the motif edge (m), from a 2x supersampled mask (anisotropic texels)
        h2 = hi.reshape(self.H, ss_, self.W, ss_)[:, ::ss_ // 2, :, ::ss_ // 2].reshape(self.H * 2, self.W * 2) if ss_ >= 2 else hi
        d2 = ndimage.distance_transform_edt(h2, sampling=(self.px_y / 2, self.px_x / 2)).astype(np.float32)
        d = d2.reshape(self.H, 2, self.W, 2).mean((1, 3))
        g = self.seed + 77
        # laid rows (along row_angle) with a rounded thread profile, couching stitches in a brick pattern
        ca, sa = math.cos(row_angle), math.sin(row_angle)
        u = self.X * ca + self.Y * sa; v = -self.X * sa + self.Y * ca
        rowp = (v / pitch_m) % 1.0
        row_h = np.sin(rowp * math.pi) ** 0.6
        rid = np.floor(v / pitch_m).astype(np.int64)
        couch = ((u / (pitch_m * 3.5) + (rid // 2 % 2) * 0.5) % 1.0)
        couch_d = np.exp(-((couch - 0.5) / 0.07) ** 2)
        # outline cord: diagonal twist across the band
        in_cord = (d < cord_m) & (cov > 0)
        tw = ((self.X + self.Y) / (cord_m * 1.1)) % 1.0
        cord_h = np.sin(np.clip(d / cord_m, 0, 1) * math.pi) * (0.75 + 0.25 * np.sin(tw * 2 * math.pi))
        shoulder = ss(0.0, bevel_m, d)
        h = raise_m * shoulder * (0.78 + 0.22 * row_h - 0.06 * couch_d)
        h = np.where(in_cord, raise_m * (0.9 + 0.55 * cord_h), h)
        h = h * cov
        self.height = np.maximum(self.height * (1 - cov * 0.6), h + self.height * 0.25)
        # colour: gold thread, darker between rows / at couching, brighter on the cord crowns; slight tarnish
        tone = noise(self.H, self.W, 2.0, g, dims=self.dims) * 0.04
        gc = col(*rgb)[None, None, :] * (1 + tone)[..., None]
        shade = np.where(in_cord, 0.80 + 0.30 * cord_h, 0.86 + 0.16 * row_h - 0.07 * couch_d)
        gc = gc * shade[..., None]
        tarn = ss(0.8, 2.2, noise(self.H, self.W, 2.2, g + 1, dims=self.dims)) * 0.35
        gc = mix(gc, col(0.40, 0.30, 0.16)[None, None, :], tarn)
        self.base = mix(self.base, gc, cov)
        rough_g = 0.36 + tarn * 0.25 + (1 - row_h) * 0.08 + couch_d * 0.04
        self.rough = lerp(self.rough, rough_g, cov)
        self.metal = lerp(self.metal, 0.9 - tarn * 0.3, cov)
        self.ao = lerp(self.ao, 0.82 + 0.18 * np.where(in_cord, cord_h, row_h), cov)
        # thin dark gap around the motif edge where the cloth tucks under the cord
        halo = np.clip(1 - np.abs(d) / 1.0, 0, 1) * 0  # (edge handled by the cord profile)
        self.gold = np.maximum(self.gold, cov)
        return cov

    # ---------------------------------------------------------------- border (woven gold galloon with studs)
    def border(self, poly_m, width_m=0.034, inset_m=0.004, stud_every_m=0.075, stud_d_m=0.0075, edge_rgb=(0.035, 0.05, 0.11),
               pattern_m=0.022):
        sdf, eid, along = edge_sdf(poly_m, self.X, self.Y)
        self.alpha = np.clip(sdf / max(self.px_x, self.px_y) + 0.5, 0, 1).astype(np.float32)
        edge = (sdf >= 0) & (sdf < inset_m)
        across = (sdf - inset_m) / width_m                                  # 0 at the outer edge of the band, 1 inner
        band = (across >= 0) & (across <= 1)
        aa = np.clip(np.minimum(across, 1 - across) * width_m / max(self.px_x, self.px_y) + 0.5, 0, 1) * band
        # galloon: two corded selvedges + a woven lozenge ground between them
        sel = (across < 0.16) | (across > 0.84)
        q = np.where(across < 0.16, across / 0.16, (1 - across) / 0.16)
        sel_h = np.sin(np.clip(q, 0, 1) * math.pi) * (0.8 + 0.2 * np.sin((along / 0.0028) * 2 * math.pi + across * 40))
        mid = (across - 0.16) / 0.68
        per = (along / pattern_m) % 1.0
        loz = np.abs(per - 0.5) * 2 + np.abs(mid - 0.5) * 2                   # < 1 inside the lozenge
        loz_in = loz < 0.92
        weave = np.sin(((along + np.where(loz_in, 0, 0.0007)) / 0.0016) * math.pi) ** 2      # fine weft ribs
        mid_h = np.where(loz_in, 0.75 + 0.25 * weave, 0.45 + 0.2 * weave) * (1 - ss(0.86, 1.0, loz) * 0.3)
        hb = np.where(sel, 0.55 + 0.45 * sel_h, mid_h) * 0.9e-3
        # studs on the centre line
        sp = (along % stud_every_m) - stud_every_m / 2
        rd = np.hypot(sp, (across - 0.5) * width_m)
        stud = (rd < stud_d_m / 2) & band
        stud_h = np.sqrt(np.clip(1 - (rd / (stud_d_m / 2)) ** 2, 0, 1)) * 1.8e-3
        hb = np.where(stud, 0.9e-3 + stud_h, hb)
        self.height = np.where(band, self.height * 0.2 + hb, self.height)
        self.height = np.where(edge, self.height * 0.7, self.height)
        g = self.seed + 91
        tone = noise(self.H, self.W, 2.0, g, dims=self.dims) * 0.04
        tarn = ss(0.9, 2.3, noise(self.H, self.W, 2.2, g + 1, dims=self.dims)) * 0.35
        gc = col(*GOLD_THREAD)[None, None, :] * (1 + tone)[..., None]
        gshade = np.where(sel, 0.78 + 0.3 * sel_h, np.where(loz_in, 0.95 + 0.12 * weave, 0.72 + 0.1 * weave))
        gc = gc * gshade[..., None]
        gc = mix(gc, col(0.40, 0.30, 0.16)[None, None, :], tarn)
        brass = col(0.86, 0.68, 0.36)[None, None, :] * (0.85 + 0.25 * stud_h[..., None] / 1.8e-3)
        gc = np.where(stud[..., None], brass, gc)
        self.base = mix(self.base, gc, aa)
        self.base = np.where(edge[..., None], mix(self.base, col(*edge_rgb)[None, None, :], np.full(edge.shape, 0.8)), self.base)
        self.rough = np.where(band, lerp(self.rough, np.where(stud, 0.24, 0.38 + tarn * 0.2 + (1 - weave) * 0.06), aa), self.rough)
        self.metal = np.where(band, lerp(self.metal, np.where(stud, 1.0, 0.88 - tarn * 0.3), aa), self.metal)
        self.ao = np.where(band, lerp(self.ao, np.where(sel, 0.8 + 0.2 * sel_h, 0.78 + 0.2 * weave), aa), self.ao)
        self.gold = np.maximum(self.gold, aa)
        return band

    def fray(self, amount=1.0):
        """Slight fading and fuzz near the outer edge."""
        pass

    def finish(self):
        cav = cavity(self.height, 1.5, ) if False else np.maximum(blur(self.height, 2.0, wrap=False) - self.height, 0)
        self.ao = np.clip(self.ao * (1 - np.clip(cav / 0.25e-3, 0, 1) * 0.35), 0, 1)
        nrm = height_to_normal(self.height, self.px_x, self.px_y, wrap=False)
        return nrm


def m_shapes(shapes, cx, cy, height_m=None, width_m=None):
    """Fit design shapes to a physical size (m) centred at (cx, cy) in panel coords."""
    return HR.fit(shapes, cx, cy, w=width_m, h=height_m)


def tabard_panel(W, H, w_m, h_m, seed, back=False):
    pnl = Panel(W, H, w_m, h_m, seed, wear=0.45)
    poly = [(0, 0), (w_m, 0), (w_m, h_m), (0, h_m)]
    # border on the long sides and the hem; the collar edge (top) gets a plain binding (it sits under the gorget)
    pnl.border([(0, -0.2), (w_m, -0.2), (w_m, h_m), (0, h_m)], width_m=0.030, inset_m=0.004, stud_every_m=0.07)
    em = HR.emblem(with_star=False)
    pnl.embroider(m_shapes(em, w_m / 2, 0.270, width_m=0.255))
    pnl.embroider(m_shapes(HR.star8(), w_m / 2, 0.070, height_m=0.060))
    fl = HR.fleur_de_lis()
    pnl.embroider(m_shapes(fl, w_m / 2, h_m - 0.150, height_m=0.100))
    for sx in (-1, 1):
        pnl.embroider(m_shapes(fl, w_m / 2 + sx * 0.110, h_m - 0.128, height_m=0.046))
    return pnl


def tabard_set(name="tabard_lion", w_m=0.40, h_m=1.05, seed=808):
    """Tabard atlas: FRONT panel in the left half (Blender U 0..0.5), BACK panel in the right half (U 0.5..1), each
    the full height: V = 1 at the collar (top of the image), V = 0 at the hem."""
    n = RES; W = n // 2
    P(name, "front")
    fr = tabard_panel(W, n, w_m, h_m, seed)
    P(name, "back")
    bk = tabard_panel(W, n, w_m, h_m, seed + 1, back=True)
    parts = [fr, bk]
    base = np.concatenate([p_.base for p_ in parts], 1); rough = np.concatenate([p_.rough for p_ in parts], 1)
    metal = np.concatenate([p_.metal for p_ in parts], 1); ao = np.concatenate([p_.ao for p_ in parts], 1)
    nrm = np.concatenate([p_.finish() for p_ in parts], 1)
    ao = np.concatenate([p_.ao for p_ in parts], 1)
    h = np.concatenate([p_.height for p_ in parts], 1)
    save_set(name, base, nrm, ao, np.clip(rough, 0, 1), np.clip(metal, 0, 1), height=h,
             meta=dict(kind="atlas", panel_m=[w_m, h_m],
                       px_per_m=[W / w_m, n / h_m],
                       regions={"front": dict(u=[0.0, 0.5], v=[0.0, 1.0], note="wearer's front; image left = wearer's right"),
                                "back": dict(u=[0.5, 1.0], v=[0.0, 1.0])},
                       layout="V = 1 collar (top), V = 0 hem; gold border on both long sides + hem (30 mm, studs every "
                              "70 mm), plain binding at the collar; star 75 mm below the collar, lion emblem centred "
                              "285 mm below the collar (330 mm tall), belt zone ~0.45-0.52 m left plain, hem fleur "
                              "150 mm above the hem, two sprigs beside it.",
                       material=dict(metallic_from="orm.B (gold thread ~0.9)", sheen=0.3, alpha="OPAQUE",
                                     lining="use cloth_blue (2nd material) or map the inside faces onto the plain "
                                            "cloth between the emblem and the hem fleur")))


def cape_set(name="cape_lion", w_m=0.90, h_m=1.40, side_m=1.22, seed=909):
    """Cape atlas, whole image = the cape's outer face: V = 1 at the shoulders (top), the hem is V-shaped: the sides
    end at side_m, the centre point at h_m (alpha = 0 below the hem line)."""
    n = RES
    P(name)
    pnl = Panel(n, n, w_m, h_m, seed, wear=0.55, dirt_bottom=0.22)
    poly = [(0, -0.3), (w_m, -0.3), (w_m, side_m), (w_m / 2, h_m - 0.004), (0, side_m)]
    pnl.border(poly, width_m=0.040, inset_m=0.005, stud_every_m=0.085, stud_d_m=0.009)
    pnl.embroider(m_shapes(HR.star8(), w_m / 2, 0.105, height_m=0.090))
    pnl.embroider(m_shapes(HR.emblem(with_star=False), w_m / 2, 0.440, width_m=0.500))
    pnl.embroider(m_shapes(HR.fleur_de_lis(), w_m / 2, side_m - 0.085, height_m=0.130))
    for sx in (-1, 1):
        pnl.embroider(m_shapes(HR.fleur_de_lis(), w_m / 2 + sx * 0.290, side_m - 0.150, height_m=0.065))
    nrm = pnl.finish()
    save_set(name, pnl.base, nrm, pnl.ao, np.clip(pnl.rough, 0, 1), np.clip(pnl.metal, 0, 1), height=pnl.height,
             alpha=pnl.alpha,
             meta=dict(kind="atlas", panel_m=[w_m, h_m], px_per_m=[n / w_m, n / h_m],
                       regions={"outer": dict(u=[0.0, 1.0], v=[0.0, 1.0])},
                       hem=dict(side_v=1 - side_m / h_m, point_v=0.0,
                                note="hem line: V = side_v at U = 0 and U = 1, V = 0 at U = 0.5 (straight lines); cut "
                                     "the mesh along it (preferred, keeps the cloth thickness) or use alpha MASK"),
                       layout="border 40 mm on both sides + the V hem, studs every 85 mm; star 105 mm below the "
                              "shoulders, lion emblem centred 450 mm below (500 mm tall), hem fleur and two sprigs; "
                              "bottom 220 mm slightly dirtier",
                       material=dict(sheen=0.3, alpha="OPAQUE after cutting the hem (alpha is only the hem mask)",
                                     lining="cloth_blue as a 2nd material on the inner faces")))


def shield_set(name="shield_lion", w_m=0.56, h_m=0.96, seed=1010):
    """Kite shield face (painted): planar UVs over the face, U across (0 = left edge as seen from the front), V up
    (V = 1 top edge). The gold rim is geometry; this map paints a thin gold inner line 22 mm inside the kite outline."""
    n = RES
    P(name)
    g = np.random.default_rng(seed)
    pnl = Panel(n, n, w_m, h_m, seed, rgb=(0.085, 0.15, 0.40), weave_px=6, wear=0.3)
    # painted wood: replace the cloth relief with brushed paint over planks
    X, Y = pnl.X, pnl.Y; dims = pnl.dims
    brush = noise(n, n, 1.4, seed + 1, ax=18, dims=dims) * 0.5 + noise(n, n, 2.2, seed + 2, dims=dims, lo=6) * 0.5
    grain = noise(n, n, 1.2, seed + 3, ax=0.04, ay=1.0, dims=dims)
    pnl.height = brush * 12e-6 + grain * 18e-6 + noise(n, n, 3.4, seed + 4, dims=dims, lo=3) * 120e-6
    kite = kite_outline(w_m, h_m)
    sdf, _, _ = edge_sdf(kite, X, Y)
    paint = col(0.085, 0.15, 0.40)[None, None, :] * (1 + brush * 0.035 + noise(n, n, 2.4, seed + 5, dims=dims) * 0.03)[..., None]
    pnl.base = paint
    pnl.rough = (0.46 + brush * 0.04).astype(np.float32)
    pnl.metal = np.zeros((n, n), np.float32); pnl.ao = np.ones((n, n), np.float32)
    pnl.alpha = np.ones((n, n), np.float32)
    # painted gold: inner line, star, lion with the scroll frame
    line = (sdf > 0.020) & (sdf < 0.027)
    gold_shapes = []
    star = m_shapes(HR.star8(), w_m / 2, 0.150, height_m=0.105)
    lion = m_shapes(HR.lion_rampant(), w_m / 2 + 0.004, 0.445, height_m=0.40)
    frame = m_shapes(HR.cartouche(), w_m / 2, 0.47, height_m=0.52)
    stem = [("add", HR.taper(np.array([[w_m / 2, 0.735], [w_m / 2, 0.80], [w_m / 2, 0.87]]), 0.010, 0.002, 1.0, cap1="point"))]
    for shp in (star, lion, frame, stem):
        gold_shapes += shp
    cov, hi = HR.raster(gold_shapes, n, n, ss=4, to_px=pnl.to_px)
    cov = np.maximum(cov, line.astype(np.float32))
    # wear: chipped paint + scratches (to the primed wood / steel under it)
    chips = ss(1.9, 2.6, noise(n, n, 1.6, seed + 6, dims=dims)) * ss(0.0, 0.08, 0.10 - np.clip(sdf, 0, 0.10) + 0.03)
    chips = np.clip(chips + ss(2.3, 2.9, noise(n, n, 1.4, seed + 7, dims=dims)) * 0.8, 0, 1)
    sc, sm = scratch_field((n, n), g, 70, (20, 260), (0.8, 2.2), (8e-6, 25e-6), random_frac=0.9, curve=0.3, wrap=False)
    gold_c = col(0.86, 0.64, 0.30)[None, None, :] * (0.9 + 0.1 * brush[..., None])
    pnl.base = mix(pnl.base, gold_c, cov)
    pnl.metal = lerp(pnl.metal, 0.85, cov); pnl.rough = lerp(pnl.rough, 0.34 + brush * 0.04, cov)
    pnl.height = pnl.height + cov * 40e-6
    under = col(0.42, 0.34, 0.25)[None, None, :]           # gesso / wood under the paint
    wear = np.clip(chips + sm, 0, 1)
    pnl.base = mix(pnl.base, under, wear * 0.9)
    pnl.metal = lerp(pnl.metal, 0.0, wear); pnl.rough = lerp(pnl.rough, 0.75, wear)
    pnl.height = pnl.height - chips * 30e-6 + sc
    # grime toward the bottom point and the edges
    grime = np.clip(ss(0.03, 0.0, sdf) + ss(h_m * 0.6, h_m, Y) * 0.4, 0, 1) * (0.5 + 0.5 * ss(-0.4, 1.4, noise(n, n, 2.0, seed + 8, dims=dims)))
    pnl.base = mix(pnl.base, col(0.10, 0.09, 0.08)[None, None, :], grime * 0.35)
    pnl.rough = pnl.rough + grime * 0.12
    nrm = height_to_normal(pnl.height, pnl.px_x, pnl.px_y, wrap=False)
    ao = np.clip(1 - np.clip((blur(pnl.height, 2.0, wrap=False) - pnl.height) / 40e-6, 0, 1) * 0.3, 0, 1)
    save_set(name, pnl.base, nrm, ao, np.clip(pnl.rough, 0, 1), np.clip(pnl.metal, 0, 1), height=pnl.height,
             meta=dict(kind="atlas", panel_m=[w_m, h_m], px_per_m=[n / w_m, n / h_m], kite_outline_uv=[
                 [round(x / w_m, 4), round(1 - y / h_m, 4)] for x, y in kite[::4]],
                 layout="planar front projection of the face: U = x / 0.56 m, V = 1 - y / 0.96 m (top edge V = 1, "
                        "point V = 0); painted gold inner line 20-27 mm inside the outline, star, lion in the scroll "
                        "frame, a stem down to the point",
                 material=dict(alpha="OPAQUE", note="paint: roughness ~0.46, gold paint metallic 0.85")))


def kite_outline(w_m, h_m, n=160):
    """Kite shield outline (m, y down): a gently arched top edge, sides swelling out then curving to the point."""
    top = [(0.0, 0.035), (w_m * 0.25, 0.012), (w_m * 0.5, 0.0), (w_m * 0.75, 0.012), (w_m, 0.035)]
    right = [(w_m, 0.035), (w_m * 0.995, 0.22), (w_m * 0.93, 0.45), (w_m * 0.78, 0.68), (w_m * 0.60, 0.87), (w_m * 0.5, h_m)]
    left = [(x, y) for x, y in reversed([(w_m - x, y) for x, y in right])]
    pts = HR.catmull(np.array(top), n=12)[:-1].tolist() + HR.catmull(np.array(right), n=24)[:-1].tolist() + \
        HR.catmull(np.array(left), n=24)[:-1].tolist()
    return np.array(pts)


register("tabard_lion", lambda: tabard_set("tabard_lion"))
register("cape_lion", lambda: cape_set("cape_lion"))
register("shield_lion", lambda: shield_set("shield_lion"))


# ============================================================================================ trim sheets
def rinceau(P, Hb, amp=0.20, stem=0.10, spiral_r=0.30, leaf=True):
    """One period (length P along x, band height Hb across y, y down) of a running acanthus scroll (mm units):
    a wavy vine; from each crest / trough a branch rolls into a volute in the opposite half, wrapped by a curled
    acanthus blade; small buds fill the rest. Everything stays inside 0.04..0.96 of the band."""
    xs = np.linspace(-0.05 * P, 1.05 * P, 80)
    vine = np.stack([xs, Hb * (0.5 - amp * np.sin(2 * math.pi * xs / P))], -1)
    sh = [("add", HR.taper(vine, stem * Hb, stem * Hb, cap0="round", cap1="round"))]
    for x0, sgn in ((0.25, 1), (0.75, -1)):
        # volute centre in the free half-space after the crest (sgn = +1: crest is up, volute below)
        cx = (x0 + 0.24) * P; cy = Hb * (0.5 + sgn * 0.14)
        r = spiral_r * Hb
        start = np.array([x0 * P + 0.02 * P, Hb * (0.5 - sgn * amp * 0.95)])
        entry = np.array([cx + r * 0.3, cy - sgn * r * 0.95])
        br = HR.catmull(np.array([start, start + [0.10 * P, sgn * 0.02 * Hb], entry]), n=14)
        sh.append(("add", HR.taper(br, stem * Hb * 0.9, stem * Hb * 0.8, cap0="round", cap1="round")))
        a0 = math.atan2(entry[1] - cy, entry[0] - cx)
        sh.append(("add", HR.spiral((cx, cy), r, r * 0.12, a0, 1.2, stem * Hb * 0.8, stem * Hb * 0.45, cw=(sgn > 0))))
        sh.append(("add", HR.ellipse(cx, cy, stem * Hb * 0.6, stem * Hb * 0.6)))
        if leaf:
            # acanthus blade: springs from the vine before the crest, arcs over the volute's outside and curls in
            lp = np.array([(x0 + 0.02) * P, Hb * (0.5 - sgn * amp * 0.5)])
            tip = np.array([cx + r * 1.25, cy + sgn * r * 0.55])
            sh.append(("add", HR.lock(lp, tip, bend=-0.32 * sgn, w=0.20 * Hb, curl=-0.45 * sgn, prof=0.8)))
            sh.append(("sub", HR.lock(lp + [0.02 * P, -sgn * 0.02 * Hb], lp + (tip - lp) * 0.75, bend=-0.3 * sgn, w=0.03 * Hb,
                                      curl=-0.3 * sgn, prof=1.2)))
            # small bud on the other side of the vine
            bp = np.array([(x0 - 0.12) * P, Hb * (0.5 - sgn * amp * 0.45)])
            sh.append(("add", HR.lock(bp, bp + np.array([-0.07 * P, -sgn * 0.20 * Hb]), bend=-0.3 * sgn, w=0.10 * Hb,
                                      curl=-0.4 * sgn)))
    return sh


def wave_scroll(P, Hb, stem=0.13):
    """Vitruvian running-wave scroll, one period."""
    sh = []
    cx, cy, r = 0.55 * P, Hb * 0.42, Hb * 0.30
    base = HR.catmull(np.array([(-0.1 * P, Hb * 0.86), (0.2 * P, Hb * 0.84), (0.42 * P, Hb * 0.72), (cx + r * 0.95, cy + r * 0.25)]), n=12)
    sh.append(("add", HR.taper(base, stem * Hb, stem * Hb, cap0="round", cap1="round")))
    a0 = math.atan2(r * 0.25, r * 0.95)
    sh.append(("add", HR.spiral((cx, cy), r, r * 0.1, a0, 1.1, stem * Hb, stem * Hb * 0.4, cw=False)))
    tail = HR.catmull(np.array([(cx + r * 0.95, cy + r * 0.25), (0.86 * P, Hb * 0.84), (1.1 * P, Hb * 0.86)]), n=10)
    sh.append(("add", HR.taper(tail, stem * Hb, stem * Hb, cap0="round", cap1="round")))
    return sh


class Sheet:
    """A 2048 trim sheet. Strips tile along U (period = tile_m); rows are allocated top-down (image y)."""

    def __init__(self, n, tile_m, seed):
        self.n = n; self.tile = tile_m; self.px = tile_m / n; self.seed = seed
        self.h = np.zeros((n, n), np.float32)
        self.base = np.zeros((n, n, 3), np.float32)
        self.rough = np.full((n, n), 0.3, np.float32)
        self.metal = np.ones((n, n), np.float32)
        self.ao = np.ones((n, n), np.float32)
        self.kind = np.zeros((n, n), np.int8)        # 0 gold, 1 steel, 2 leather etc. (for colouring)
        self.strips = {}
        self.decals = {}

    def rows(self, y0, hpx):
        u = (np.arange(self.n, dtype=np.float32) + 0.5) * self.px
        v = (np.arange(hpx, dtype=np.float32) + 0.5) / hpx
        return np.meshgrid(u, v)

    def add_strip(self, name, y0, hpx, h, note=""):
        self.h[y0:y0 + hpx] = h
        n = self.n
        self.strips[name] = dict(rows_px=[y0, y0 + hpx], v=[round(1 - (y0 + hpx) / n, 6), round(1 - y0 / n, 6)],
                                 width_mm=round(hpx * self.px * 1000, 2), tiles_along_u_m=self.tile, note=note)

    def motif_strip(self, shapes_fn, period_m, y0, hpx):
        """Rasterise a periodic motif (drawn in metres for one period, y down 0..band) across the strip with wrap."""
        band_m = hpx * self.px
        reps = int(round(self.tile / period_m))
        allsh = []
        one = HR.xf(shapes_fn(period_m * 1000, band_m * 1000), sx=1e-3)     # motifs are drawn in mm
        for k in range(-1, reps + 1):
            allsh += HR.xf(one, tx=k * period_m)
        cov, hi = HR.raster(allsh, self.n, hpx, ss=4, to_px=lambda p: np.asarray(p) / self.px)
        d = ndimage.distance_transform_edt(hi, sampling=self.px / 4).astype(np.float32)
        d = d.reshape(hpx, 4, self.n, 4).mean((1, 3))
        return cov, d


def emboss_profile(cov, d, height_m, round_m):
    """Rounded embossing: rises over round_m from the motif edge to a gently domed crown."""
    return cov * height_m * np.sqrt(np.clip(d / round_m, 0, 1)) * (0.85 + 0.15 * np.clip(d / (round_m * 2.5), 0, 1))


def matting(h, w, seed, px, density=0.35):
    """Ring-matting / punched ground texture (small round punch marks)."""
    g = np.random.default_rng(seed)
    pts = (g.random((h, w)) < density * 0.02).astype(np.float32)
    return -blur(pts, 0.8 / 1.0, wrap=True) * 3.0


def fillet(v, v0, v1, height=1.0):
    """Raised half-round fillet between v0 and v1 (fractions across the strip)."""
    t = (v - v0) / (v1 - v0)
    return np.where((t >= 0) & (t <= 1), np.sqrt(np.clip(1 - (2 * t - 1) ** 2, 0, 1)) * height, 0.0)


def trim_gold_set(name="trim_gold", tile_m=0.5, seed=1111):
    """Gold trim sheet (strips tile along U, period 0.5 m = 1 UV unit) + embossed / inlaid decals. Everything metal."""
    n = RES; S = Sheet(n, tile_m, seed); px = S.px
    k = n / 2048.0
    R = lambda v: int(round(v * k))
    P(name, "strips")
    layout = [("filigree_wide", 0, R(256)), ("filigree_narrow", R(256), R(192)), ("rope", R(448), R(128)),
              ("rivets", R(576), R(128)), ("bead", R(704), R(128)), ("lines", R(832), R(96)), ("pearls", R(928), R(96))]
    steel_mask = np.zeros((n, n), np.float32)
    for sname, y0, hp in layout:
        U, V = S.rows(y0, hp)
        band_m = hp * px
        edge_round = ss(0.0, 0.06, V) * ss(0.0, 0.06, 1 - V)                       # rolled-over outer edges
        if sname in ("filigree_wide", "filigree_narrow"):
            per = 0.0833333 if sname == "filigree_wide" else 0.0625
            fn = (lambda P_, B_: rinceau(P_, B_)) if sname == "filigree_wide" else (lambda P_, B_: wave_scroll(P_, B_))
            # the motif lives between the fillets: draw it in a sub-band
            inner0, inner1 = (0.16, 0.84)
            sub_y0 = y0 + int(hp * inner0); sub_h = int(hp * (inner1 - inner0))
            cov, d = S.motif_strip(fn, per, sub_y0, sub_h)
            covf = np.zeros((hp, n), np.float32); df = np.zeros((hp, n), np.float32)
            covf[sub_y0 - y0:sub_y0 - y0 + sub_h] = cov; df[sub_y0 - y0:sub_y0 - y0 + sub_h] = d
            relief = emboss_profile(covf, df, 0.45e-3, 1.1e-3)
            ground = -0.15e-3 + matting(hp, n, seed + y0, px) * 0.02e-3 * (1 - covf)
            fil = (fillet(V, 0.06, 0.14) + fillet(V, 0.86, 0.94)) * 0.55e-3
            inside = (V > 0.14) & (V < 0.86)
            h = np.where(inside, np.maximum(ground, relief - 0.15e-3 * (1 - covf)), 0.0) + fil
            h = h * edge_round + (edge_round - 1) * 0.3e-3
            S.add_strip(sname, y0, hp, h, note=f"embossed {'acanthus rinceau' if sname == 'filigree_wide' else 'running-wave scroll'} "
                                              f"(period {per * 1000:.1f} mm) on a matted ground between two fillets")
        elif sname == "rope":
            pitch = 0.5 / 56                                                  # ~8.9 mm per strand
            phase = (U / pitch + V * 1.3) % 1.0
            strand = np.sqrt(np.clip(1 - (2 * phase - 1) ** 2, 0, 1))
            body = np.sqrt(np.clip(1 - (2 * V - 1) ** 2, 0, 1))
            h = (strand * 0.35 + 0.65) * body * 1.1e-3 - 0.3e-3
            S.add_strip(sname, y0, hp, h, note="twisted rope / cable edge (roped edge)")
        elif sname == "rivets":
            h = (fillet(V, 0.0, 0.1) + fillet(V, 0.9, 1.0)) * 0.35e-3
            groove = np.exp(-((V - 0.2) / 0.018) ** 2) + np.exp(-((V - 0.8) / 0.018) ** 2)
            h = h - groove * 0.12e-3
            sp = 0.0625
            du = ((U + sp / 2) % sp) - sp / 2
            rr = np.hypot(du, (V - 0.5) * band_m)
            rad = 0.0055
            dome = np.sqrt(np.clip(1 - (rr / rad) ** 2, 0, 1)) * 1.6e-3
            ring = np.exp(-((rr - rad * 1.05) / 0.0006) ** 2) * -0.1e-3
            h = h + dome + ring
            S.add_strip(sname, y0, hp, h, note="flat band, incised lines, domed rivets every 62.5 mm (8 per tile)")
        elif sname == "bead":
            h = np.sqrt(np.clip(1 - (2 * V - 1) ** 2, 0, 1)) * 1.4e-3
            S.add_strip(sname, y0, hp, h, note="half-round rolled bead (plain polished)")
        elif sname == "lines":
            groove = sum(np.exp(-((V - c) / 0.03) ** 2) for c in (0.22, 0.32, 0.68, 0.78))
            h = -groove * 0.12e-3 + (edge_round - 1) * 0.2e-3
            S.add_strip(sname, y0, hp, h, note="flat band with two pairs of incised lines")
        elif sname == "pearls":
            sp = 0.5 / 64
            du = ((U + sp / 2) % sp) - sp / 2
            rr = np.hypot(du, (V - 0.5) * band_m)
            rad = min(sp, band_m) * 0.42
            h = np.sqrt(np.clip(1 - (rr / rad) ** 2, 0, 1)) * rad * 0.8 - 0.1e-3
            S.add_strip(sname, y0, hp, h, note="beaded (pearled) row, 7.8 mm pitch")
    # ---------------------------------------------------------------- decals: 2 rows of 4 squares (125 mm each)
    P(name, "decals")
    D = n // 4
    decals = [("lion_plaque", 0, 0), ("star_boss", 1, 0), ("fleur_plaque", 2, 0), ("rosette_boss", 3, 0),
              ("lion_inlay_steel", 0, 1), ("star_inlay_steel", 1, 1), ("fleur_inlay_steel", 2, 1), ("frame_inlay_steel", 3, 1)]
    for dname, cx_i, row in decals:
        x0 = cx_i * D; y0 = n // 2 + row * D
        dm = D * px
        on_steel = "steel" in dname
        if dname.startswith("lion"):
            sh = HR.fit(HR.lion_rampant(), dm / 2, dm / 2, w=dm * 0.80, h=dm * 0.84)
        elif dname.startswith("star"):
            sh = HR.fit(HR.star8(), dm / 2, dm / 2, w=dm * 0.78, h=dm * 0.78)
        elif dname.startswith("fleur"):
            sh = HR.fit(HR.fleur_de_lis(), dm / 2, dm / 2, w=dm * 0.72, h=dm * 0.80)
        elif dname == "rosette_boss":
            sh = []; mm_ = dm * 1000; c = np.array([mm_ / 2, mm_ / 2])
            for kk in range(8):
                a = kk * math.pi / 4 + math.pi / 8; dv = np.array([math.cos(a), math.sin(a)])
                pc = c + dv * mm_ * 0.24
                sh.append(("add", HR.ellipse(pc[0], pc[1], mm_ * 0.15, mm_ * 0.085, a)))
                sh.append(("sub", HR.taper(np.array([c + dv * mm_ * 0.14, c + dv * mm_ * 0.34]), 1.0, 2.2, 1.0,
                                           cap0="point", cap1="point")))
            sh.append(("sub", HR.ellipse(c[0], c[1], mm_ * 0.125, mm_ * 0.125)))
            sh.append(("add", HR.ellipse(c[0], c[1], mm_ * 0.10, mm_ * 0.10)))
            sh = HR.xf(sh, sx=1e-3)
        else:
            # scroll frame (cartouche) for inlay on steel: pauldron / breastplate panels
            sh = HR.fit(HR.cartouche(), dm / 2, dm / 2, w=dm * 0.86, h=dm * 0.86)
        cov, hi = HR.raster(sh, D, D, ss=4, to_px=lambda p: np.asarray(p) / px)
        d = ndimage.distance_transform_edt(hi, sampling=px / 4).astype(np.float32).reshape(D, 4, D, 4).mean((1, 3))
        yy, xx = np.meshgrid((np.arange(D) + 0.5) / D, (np.arange(D) + 0.5) / D, indexing="ij")
        if on_steel:
            # gold inlay, flush with a tiny proud edge, engraved outline groove around it
            dout = ndimage.distance_transform_edt(~hi, sampling=px / 4).astype(np.float32).reshape(D, 4, D, 4).mean((1, 3))
            groove = np.exp(-(dout / 0.00035) ** 2) * (1 - cov)
            h = cov * 0.08e-3 * np.sqrt(np.clip(d / 0.5e-3, 0, 1)) - groove * 0.10e-3
            steel_mask[y0:y0 + D, x0:x0 + D] = 1 - cov
        else:
            # repousse plaque: domed plaque, rolled rim, the motif raised on a matted ground
            r = np.hypot(xx - 0.5, yy - 0.5) * 2
            plaque = np.sqrt(np.clip(1 - r ** 2, 0, 1)) * 0.6e-3 if "boss" in dname else np.zeros_like(r)
            rim = np.where((np.maximum(np.abs(xx - 0.5), np.abs(yy - 0.5)) > 0.46), 1.0, 0.0) if "plaque" in dname else 0
            relief = emboss_profile(cov, d, 0.9e-3 if "boss" not in dname else 0.7e-3, 1.8e-3)
            ground = matting(D, D, seed + x0 + y0, px) * 0.02e-3 * (1 - cov)
            h = plaque + relief + ground + rim * 0.4e-3
        S.h[y0:y0 + D, x0:x0 + D] = h
        S.decals[dname] = dict(u=[round(x0 / n, 6), round((x0 + D) / n, 6)], v=[round(1 - (y0 + D) / n, 6), round(1 - y0 / n, 6)],
                               size_mm=round(dm * 1000, 1), on="steel" if on_steel else "gold")
    # ---------------------------------------------------------------- colour / roughness from the relief
    P(name, "shading")
    g = np.random.default_rng(seed + 5)
    h = S.h
    loc = blur(h, 3.0) ; cav = np.clip((loc - h) / 0.08e-3, 0, 1)            # recesses
    crown = np.clip((h - blur(h, 6.0)) / 0.10e-3, 0, 1)                       # raised crowns (polished)
    s1, m1 = scratch_field((n, n), g, 900, (8, 50), (0.5, 0.9), (0.5e-6, 1.5e-6), random_frac=1.0, curve=0.8)
    s2, m2 = scratch_field((n, n), g, 60, (20, 160), (0.9, 1.8), (3e-6, 8e-6), random_frac=0.7, curve=0.3)
    h = h + s1 + s2
    dims = (tile_m, tile_m)
    tone = noise(n, n, 2.0, seed + 6, dims=dims) * 0.02
    tarn = np.clip(cav * 0.85 + ss(1.2, 2.4, noise(n, n, 2.4, seed + 7, dims=dims, lo=2)) * 0.2, 0, 1)
    gold = col(0.86, 0.68, 0.38)[None, None, :] * (1 + tone)[..., None]
    gold = mix(gold, col(0.96, 0.83, 0.56)[None, None, :], crown * 0.35 + m2 * 0.3)
    gold = mix(gold, col(0.36, 0.25, 0.12)[None, None, :], tarn * 0.75)
    # steel under the inlays = the steel_worn tile itself (same UV scale: 0.5 m per 2048 px), so the decal squares
    # match plates made with steel_worn
    st_base, st_nrm, st_ao, st_rough, st_metal, st_h = steel_maps(0.5, 101, (0.64, 0.65, 0.68), False)
    steel = mix(st_base, col(0.25, 0.23, 0.21)[None, None, :], cav * 0.6)
    base = mix(gold, steel, steel_mask)
    rough = 0.25 + tarn * 0.33 - crown * 0.07 + m1 * 0.04 + noise(n, n, 2.4, seed + 8, dims=dims, lo=3) * 0.025
    rough = lerp(rough, st_rough + cav * 0.2, steel_mask)
    h = h + st_h * steel_mask
    metal = 1.0 - tarn * 0.15
    ao = 1.0 - np.clip((blur(S.h, 4.0) - S.h) / 0.15e-3, 0, 1) * 0.55
    save_set(name, base, height_to_normal(h, px), np.clip(ao, 0, 1), np.clip(rough, 0.05, 1), np.clip(metal, 0, 1),
             height=h, meta=dict(kind="trim_sheet", tile_m=[tile_m, tile_m], px_per_m=n / tile_m, uv_per_m_along=1 / tile_m,
                                 strips=S.strips, decals=S.decals,
                                 usage="unwrap each trim as a straight strip, U along the trim; fit its V to a strip "
                                       "(char_materials.map_strip); U scale keeps the aspect (1 UV unit = 0.5 m at the "
                                       "strip's native width, proportionally less when the trim is narrower). "
                                       "Decals: map a quad / island into the square.",
                                 material=dict(metallic=1, alpha="OPAQUE")))


register("trim_gold", lambda: trim_gold_set("trim_gold"))


def leather_straps_set(name="leather_straps", tile_m=0.5, seed=1212):
    """Leather strap trim sheet: belt / strap strips tiling along U (1 UV unit = 0.5 m at the native width), a blue
    grip wrap, a braided cord, and a plain leather field (tiles along U) for pouches, scabbards, boots."""
    n = RES; px = tile_m / n; dims = (tile_m, tile_m); k = n / 2048.0
    R = lambda v: int(round(v * k))
    P(name, "field")
    hm, base, rough, ao = leather_field(n, n, px, seed, dims)
    metal = np.zeros((n, n), np.float32)
    strips = {}
    layout = [("belt_wide", 0, R(192)), ("belt_medium", R(192), R(128)), ("strap_holes", R(320), R(112)),
              ("strap_narrow", R(432), R(80)), ("strap_studs", R(512), R(128)), ("grip_wrap_blue", R(640), R(128)),
              ("braid", R(768), R(96)), ("field", R(864), n - R(864))]
    thread = col(0.56, 0.45, 0.30)
    for sname, y0, hp in layout:
        u = (np.arange(n, dtype=np.float32) + 0.5) * px
        v = (np.arange(hp, dtype=np.float32) + 0.5) / hp
        U, V = np.meshgrid(u, v)
        band = hp * px
        sl = slice(y0, y0 + hp)
        hh = hm[sl].copy(); bb = base[sl].copy(); rr = rough[sl].copy(); aa = ao[sl].copy(); mm = metal[sl].copy()
        dist_edge = np.minimum(V, 1 - V) * band                               # m from the nearest strap edge
        if sname != "field":
            # bevelled, burnished (darker, smoother) edges
            bev = ss(0.0, 0.0018, dist_edge)
            hh = hh * bev + (bev - 1) * 0.5e-3
            burn = 1 - ss(0.0, 0.0025, dist_edge)
            bb = mix(bb, col(0.10, 0.06, 0.04)[None, None, :], burn * 0.7)
            rr = rr - burn * 0.15
        if sname in ("belt_wide", "belt_medium", "strap_holes", "strap_studs"):
            cpos = 0.0035 if sname != "strap_holes" else 0.003
            crease = np.exp(-((dist_edge - cpos) / 0.00035) ** 2)
            hh = hh - crease * 0.15e-3
            bb = bb * (1 - crease * 0.35)[..., None]
        if sname in ("belt_wide", "belt_medium"):
            pitch = 0.004; slen = 0.0028
            for side in (0, 1):
                vc = (0.0035 / band) if side == 0 else 1 - 0.0035 / band
                du = (U % pitch) - pitch / 2
                dvm = (V - vc) * band
                a = math.radians(30)
                x_ = du * math.cos(a) + dvm * math.sin(a); y_ = -du * math.sin(a) + dvm * math.cos(a)
                st = np.clip(1 - (x_ / (slen / 2)) ** 2, 0, 1) * np.clip(1 - (y_ / 0.0005) ** 2, 0, 1)
                st = np.sqrt(st)
                hole = np.exp(-((np.abs(x_) - slen / 2) ** 2 + y_ ** 2) / (0.00035 ** 2))
                hh = np.where(st > 0, np.maximum(hh, -0.1e-3 + st * 0.25e-3), hh) - hole * 0.1e-3
                bb = mix(bb, thread[None, None, :] * (0.8 + 0.2 * st[..., None]), np.clip(st * 1.6, 0, 1))
                rr = lerp(rr, 0.62, np.clip(st * 1.6, 0, 1))
        if sname == "strap_holes":
            sp = 0.025
            du = ((U + sp / 2) % sp) - sp / 2
            r = np.hypot(du, (V - 0.5) * band)
            hole = r < 0.0022
            rim = np.exp(-((r - 0.0024) / 0.0004) ** 2)
            hh = np.where(hole, -1.2e-3, hh - rim * 0.1e-3)
            bb = np.where(hole[..., None], col(0.02, 0.015, 0.012)[None, None, :], bb * (1 - rim * 0.3)[..., None])
            aa = np.where(hole, 0.2, aa)
        if sname == "strap_studs":
            sp = 0.03125
            du = ((U + sp / 2) % sp) - sp / 2
            r = np.hypot(du, (V - 0.5) * band)
            rad = 0.0055
            dome = np.sqrt(np.clip(1 - (r / rad) ** 2, 0, 1))
            stud = r < rad
            hh = np.where(stud, 0.3e-3 + dome * 1.8e-3, hh - np.exp(-((r - rad) / 0.0006) ** 2) * 0.15e-3)
            brass = col(0.84, 0.66, 0.34)[None, None, :] * (0.8 + 0.25 * dome[..., None])
            bb = np.where(stud[..., None], brass, bb)
            rr = np.where(stud, 0.28, rr); mm = np.where(stud, 1.0, mm)
            aa = aa * (1 - np.exp(-((r - rad) / 0.0008) ** 2) * 0.5)
        if sname == "grip_wrap_blue":
            pitch = 0.012
            ph = (U / pitch + V * band / pitch * 0.9) % 1.0
            wrap = np.sqrt(np.clip(1 - (2 * np.clip(ph / 0.82, 0, 1) - 1) ** 2, 0, 1))
            wire = (ph > 0.84) & (ph < 0.98)
            wph = (ph - 0.84) / 0.14
            wire_h = np.sqrt(np.clip(1 - (2 * wph - 1) ** 2, 0, 1)) * wire
            twist = 0.75 + 0.25 * np.sin(U / 0.0012 * 2 * math.pi)
            hh = np.where(wire, 0.5e-3 * wire_h * twist, wrap * 0.9e-3 + hm[sl] * 0.3)
            blue = col(0.07, 0.12, 0.33)[None, None, :] * (0.7 + 0.3 * wrap[..., None]) * (1 + noise(hp, n, 2, seed + 3) * 0.05)[..., None]
            goldw = col(0.88, 0.68, 0.34)[None, None, :] * (0.7 + 0.35 * wire_h[..., None] * twist[..., None])
            bb = np.where(wire[..., None], goldw, blue)
            rr = np.where(wire, 0.3, 0.5 - wrap * 0.08); mm = np.where(wire, 1.0, 0.0)
            aa = np.where(wire, 0.85 + 0.15 * wire_h, 0.65 + 0.35 * wrap)
        if sname == "braid":
            # 4-strand round braid seen from the side: chevrons of plump strands
            pitch = 0.008
            ph1 = (U / pitch + np.abs(V - 0.5) * band / pitch * 1.6) % 1.0
            strand = np.sqrt(np.clip(1 - (2 * ph1 - 1) ** 2, 0, 1))
            body = np.sqrt(np.clip(1 - (2 * V - 1) ** 2, 0, 1))
            hh = (0.6 * strand + 0.4) * body * 1.6e-3 - 0.4e-3 + hm[sl] * 0.2
            bb = base[sl] * (0.6 + 0.5 * strand[..., None]) * (0.4 + 0.6 * body[..., None])
            rr = 0.55 - strand * 0.1; aa = 0.5 + 0.5 * strand * body
        hm[sl] = hh; base[sl] = bb; rough[sl] = rr; ao[sl] = aa; metal[sl] = mm
        strips[sname] = dict(rows_px=[y0, y0 + hp], v=[round(1 - (y0 + hp) / n, 6), round(1 - y0 / n, 6)],
                             width_mm=round(hp * px * 1000, 2))
    notes = {"belt_wide": "47 mm belt, creases + saddle stitching both edges, burnished edges",
             "belt_medium": "31 mm strap, creases + stitching", "strap_holes": "27 mm strap, buckle holes every 25 mm",
             "strap_narrow": "20 mm plain strap", "strap_studs": "31 mm strap, brass studs every 31 mm",
             "grip_wrap_blue": "blue leather spiral wrap with twisted gold wire (sword grip), wrap pitch 12 mm",
             "braid": "braided leather cord (side view)", "field": "plain leather; tiles along U only (V fills the island)"}
    for k_, v_ in notes.items():
        strips[k_]["note"] = v_
    save_set(name, base, height_to_normal(hm, px), np.clip(ao, 0, 1), np.clip(rough, 0.05, 1), np.clip(metal, 0, 1),
             height=hm, meta=dict(kind="trim_sheet", tile_m=[tile_m, tile_m], px_per_m=n / tile_m, uv_per_m_along=1 / tile_m,
                                  strips=strips, material=dict(metallic_from="orm.B (studs, wire)", alpha="OPAQUE")))


register("leather_straps", lambda: leather_straps_set("leather_straps"))


def heraldry_svg(name="heraldry"):
    """Vector masters of the heraldry (gold on royal blue) for UI, banners, decals: assets/textures/heraldry/*.svg."""
    out = os.path.join(TEX, "heraldry"); os.makedirs(out, exist_ok=True)
    items = {"lion_rampant": HR.lion_rampant(), "emblem": HR.emblem(), "cartouche": HR.cartouche(),
             "fleur_de_lis": HR.fleur_de_lis(), "star8": HR.star8(), "sprig": HR.sprig()}
    files = {}
    for k_, sh in items.items():
        lo, hi = HR.bbox(sh); pad = (hi - lo).max() * 0.06
        view = (lo - pad, hi + pad); wh = view[1] - view[0]
        W = 1024; H = int(W * wh[1] / wh[0])
        HR.to_svg(sh, os.path.join(out, k_ + ".svg"), W, H, view=view)
        files[k_] = f"heraldry/{k_}.svg"
    MANIFEST[name] = dict(kind="vector", files=files, note="gold #d9a53f on royal blue #1c3478; 'sub' shapes are drawn "
                                                          "in the field colour (painter order)")
    P("saved heraldry svgs", list(files))


register("heraldry", heraldry_svg)


# ============================================================================================ main
register("steel_worn", lambda: steel_set("steel_worn"))
register("steel_blued", lambda: steel_set("steel_blued", seed=111, base_rgb=(0.30, 0.40, 0.62), blued=True))
register("gold_worn", lambda: gold_set("gold_worn"))
register("cloth_blue", lambda: cloth_set("cloth_blue"))
register("leather_brown", lambda: leather_set("leather_brown"))
register("mail_riveted", lambda: mail_set("mail_riveted"))
register("horsehair_blue", lambda: horsehair_set("horsehair_blue"))
register("skin_detail", lambda: skin_detail_set("skin_detail"))


def write_manifest():
    path = os.path.join(TEX, "textures_char.json")
    old = {}
    if os.path.exists(path):
        try:
            old = json.load(open(path)).get("sets", {})
        except Exception:
            old = {}
    old.update(MANIFEST)
    doc = dict(generator="scripts/textures_char.py", res=RES,
               conventions=dict(normal="OpenGL / glTF tangent space, +Y = up in the image (= +V)",
                                orm="R occlusion, G roughness, B metallic (linear)", base="sRGB",
                                height="16-bit PNG, 0 = lowest; range in height_range_mm",
                                uv="Blender UV: (0,0) = bottom-left of the image, V up. uv_per_m = UV units per metre "
                                   "for tileables (scale world-size UVs by it)."),
               sets=dict(sorted(old.items())))
    json.dump(doc, open(path, "w"), indent=1)
    P("manifest", path)


def main(argv):
    global RES, TEX
    args = [a for a in argv if not a.startswith("--")]
    if "--res" in argv:
        RES = int(argv[argv.index("--res") + 1]); args = [a for a in args if a != str(RES)]
    if "--out" in argv:                                    # e.g. a 4k cutscene set in its own folder
        TEX = os.path.abspath(argv[argv.index("--out") + 1]); args = [a for a in args if os.path.abspath(a) != TEX]
    names = args or list(SETS)
    bad = [a for a in names if a not in SETS]
    if bad:
        sys.exit(f"unknown sets {bad}; known: {list(SETS)}")
    for nm in names:
        t0 = time.time(); SETS[nm](); P(f"{nm} done in {time.time() - t0:.1f}s")
    write_manifest()
    if "--no-sheet" not in argv:
        contact_sheet(names)


def contact_sheet(names, thumb=384):
    from PIL import ImageFont
    rows = []
    for nm in names:
        m = MANIFEST.get(nm)
        if not m:
            continue
        tiles = []
        for key in ("base", "normal", "orm"):
            f = m["files"].get(key)
            if not f:
                continue
            im = Image.open(os.path.join(TEX, f))
            if im.mode == "RGBA":
                bg = Image.new("RGBA", im.size, (40, 40, 44, 255)); bg.alpha_composite(im); im = bg
            tiles.append(im.convert("RGB").resize((thumb, thumb), Image.LANCZOS))
        if tiles:
            row = Image.new("RGB", (thumb * 3 + 220, thumb), (24, 24, 28))
            for i, t in enumerate(tiles):
                row.paste(t, (220 + i * thumb, 0))
            d = ImageDraw.Draw(row)
            try:
                font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 22)
            except Exception:
                font = None
            d.text((12, 12), nm, fill=(235, 235, 235), font=font)
            info = m.get("kind", "")
            if "tile_m" in m:
                info += f"\ntile {m['tile_m'][0]:.3g} x {m['tile_m'][1]:.3g} m"
            d.multiline_text((12, 48), info, fill=(170, 170, 170), font=font)
            rows.append(row)
    if not rows:
        return
    sheet = Image.new("RGB", (rows[0].width, sum(r.height for r in rows)), (0, 0, 0))
    y = 0
    for r in rows:
        sheet.paste(r, (0, y)); y += r.height
    os.makedirs(RENDERS, exist_ok=True)
    out = os.path.join(RENDERS, "textures_sheet_" + ("all" if len(names) == len(SETS) else "_".join(names)[:60]) + ".png")
    sheet.save(out); P("sheet", out)


if __name__ == "__main__":
    main(sys.argv[1:])
