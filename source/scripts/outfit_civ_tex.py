#!/usr/bin/env python3
"""Texture sets for the CIVILIAN outfits (peasant, archer), in the kit's painterly-realistic style: procedural PBR
(textures_char.py building blocks: periodic noise, weaves, leather grain, height -> normal) plus a painted layer
(broad value / temperature mottling, soft brush streaks, darkened crevices, worn highlights on raised threads), so
the cloth reads well both close up and at RTS distance.

usage:  python3 scripts/outfit_civ_tex.py [set ...] [--res 1024]
writes assets/textures/civ/<set>_{base,normal,orm}.png (+ RGBA base for alpha sets), the manifest
assets/textures/civ/civ_textures.json and renders/civ/civ_textures_sheet.png. System python3 + numpy + scipy + Pillow.
Conventions = textures_char.py: base sRGB, normal OpenGL (+Y up = +V), ORM = R occlusion / G roughness / B metallic,
Blender UV (0,0) bottom-left. uv_per_m = UV units per metre for tileables (the outfit scripts write metric UVs x it).

sets (all own procedural art, CC0-compatible):
  civ_linen          undyed linen tabby, slubby threads                           tile 0.12 m
  civ_wool_russet    fulled wool twill, madder russet (peasant tunic)             tile 0.20 m
  civ_wool_brown     fulled wool, walnut brown (trousers, hat band)               tile 0.20 m
  civ_wool_green     fulled wool, weld + woad green (archer hood)                 tile 0.20 m
  civ_wool_undyed    fulled wool, natural grey-brown (peasant hood variant)       tile 0.20 m
  civ_quilt          gambeson: linen canvas, stitched quilting channels along V   tile 0.20 m (5 channels)
  civ_leather_tan    soft tanned leather (shoes, gloves, pouch)                   tile 0.30 m
  civ_leather_dark   dark oiled leather (belts, bracers, soles, quiver)           tile 0.30 m
  civ_straw          plaited straw rows along U (hat)                             tile 0.12 m
  civ_legwrap        spiral leg wraps: 1 UV = one turn around the leg (u), 0.18 m (v), 4 bands, seamless spiral
  civ_trim           trim sheet: tablet-woven band, twisted cord, stitched hem, crossed lacing, fletching wrap
  civ_iron           forged iron (buckles, knife pommel, arrowheads)              tile 0.20 m
  civ_wood           polished wood, grain along V (bow, arrows, knife grip)       tile 0.40 m
  civ_fletch         feather cards (RGBA, alpha = vane): 3 feathers across U, quill along V
"""
import os, sys, json, math, time
import numpy as np
from PIL import Image
from scipy import ndimage

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import textures_char as TC

CH = os.path.dirname(HERE)
OUTD = os.path.join(CH, "assets", "textures", "civ")
REND = os.path.join(CH, "renders", "civ")
RES = 1024
MAN = {}


def P(*a):
    print("[civtex]", *a, flush=True)


noise, blur, ss, mix, col, lerp = TC.noise, TC.blur, TC.ss, TC.mix, TC.col, TC.lerp


def save(name, base, normal, ao, rough, metal, meta, alpha=None):
    """save through textures_char.save_set into the civ folder (its manifest entry is copied into ours)"""
    TC.TEX = OUTD
    TC.MANIFEST.clear()
    TC.save_set(name, base, normal, ao, rough, metal, alpha=alpha, meta=meta)
    MAN[name] = TC.MANIFEST[name]


def painterly(c, seed, dims, value=0.10, warm=0.04, streak=0.05, streak_ang=0.0):
    """the painted layer: broad value mottling, warm / cool temperature drift and soft directional brush streaks"""
    h, w = c.shape[:2]
    v = noise(h, w, 2.2, seed, dims=dims) * value + noise(h, w, 3.2, seed + 1, dims=dims, lo=1.5) * value * 0.6
    t = noise(h, w, 2.6, seed + 2, dims=dims) * warm
    st = noise(h, w, 1.8, seed + 3, ax=8.0, rot=streak_ang, dims=dims) * streak
    out = c * (1 + v + st)[..., None]
    out = out + np.stack([t, t * 0.3, -t * 0.8], -1) * np.maximum(out.mean(-1, keepdims=True), 0.05)
    return np.clip(out, 0, 1)


def weave_tabby(n, threads, seed, slub=0.2):
    """plain (tabby) weave height 0..1 with slubby threads"""
    g = np.random.default_rng(seed)
    x = (np.arange(n) + 0.5) / n * threads
    X, Y = np.meshgrid(x, x)
    j = np.floor(X).astype(int); i = np.floor(Y).astype(int)
    fx, fy = X - j, Y - i
    sx = noise(n, n, 2.0, seed + 1, ax=0.05, ay=1.0) * slub
    sy = noise(n, n, 2.0, seed + 2, ax=1.0, ay=0.05) * slub
    warp = np.sqrt(np.clip(1 - ((fx - 0.5) / (0.5 * (1 + sx))) ** 2, 0, 1))
    weft = np.sqrt(np.clip(1 - ((fy - 0.5) / (0.5 * (1 + sy))) ** 2, 0, 1))
    over = ((i + j) % 2 == 0)
    sw = np.where(over, 1.0, 0.55); se = np.where(over, 0.55, 1.0)
    # the thread rises and dips along its length (smooth over/under)
    sw = sw * (0.75 + 0.25 * np.cos(math.pi * (fy - 0.5)))
    se = se * (0.75 + 0.25 * np.cos(math.pi * (fx - 0.5)))
    wh, eh = warp * sw, weft * se
    top_warp = wh >= eh
    return np.maximum(wh, eh).astype(np.float32), top_warp, j, i


# ============================================================================================ cloth
def linen():
    n = RES; tile = 0.12; dims = (tile, tile); px = tile / n
    h, top, j, i = weave_tabby(n, 150, 11, slub=0.35)
    fuzz = noise(n, n, 0.9, 12, dims=dims) * 0.06
    lumps = noise(n, n, 2.6, 13, dims=dims, lo=6) * 0.2
    hm = (h + fuzz) * 0.22e-3 + lumps * 0.05e-3
    g = np.random.default_rng(14)
    tv = np.where(top, g.normal(0, 1, 97)[j % 97] * 0.05, g.normal(0, 1, 97)[i % 97] * 0.05)
    base = col(0.74, 0.66, 0.52)[None, None, :] * (1 + tv + (h - 0.6) * 0.12)[..., None]
    base = painterly(base, 15, dims, value=0.06, warm=0.03, streak=0.03)
    cav = np.clip((blur(h, 1.5) - h) * 2.2, 0, 1)
    base = base * (1 - cav * 0.28)[..., None]
    rough = 0.82 + cav * 0.08 - h * 0.05 + noise(n, n, 1.5, 16, dims=dims) * 0.03
    save("civ_linen", base, TC.height_to_normal(hm, px), 1 - cav * 0.4, np.clip(rough, 0, 1), np.zeros((n, n)),
         dict(kind="tileable", tile_m=[tile, tile], px_per_m=n / tile, uv_per_m=1 / tile, material=dict(sheen=0.15)))


WOOLS = {"civ_wool_russet": (0.33, 0.115, 0.06), "civ_wool_brown": (0.21, 0.145, 0.095),
         "civ_wool_green": (0.10, 0.17, 0.075), "civ_wool_undyed": (0.30, 0.26, 0.20)}


def wool(name):
    n = RES; tile = 0.20; dims = (tile, tile); px = tile / n
    rgb = WOOLS[name]
    seed = 300 + list(WOOLS).index(name) * 13
    wv, top, j, i = TC.twill_height(n, n, 140, 140, 301, slub=0.16, felt=0.55)       # same weave for every dye
    fuzz = noise(n, n, 0.8, 305, dims=dims) * 0.12
    lumps = noise(n, n, 2.8, 306, dims=dims, lo=6) * 0.35
    pill = ss(2.3, 3.2, noise(n, n, 1.0, 307, dims=dims)) * 0.5
    hm = (wv + fuzz + pill) * 0.32e-3 + lumps * 0.08e-3
    # heathered two-tone yarn + sun / wear fading toward a paler, greyer version of the same dye
    g = np.random.default_rng(seed)
    tx = int(j.max() + 1); ty = int(i.max() + 1)
    thread = np.where(top, g.normal(0, 1, tx)[j] * 0.04 + 0.02, g.normal(0, 1, ty)[i] * 0.04 - 0.02)
    heather = noise(n, n, 1.2, seed + 3, dims=dims) * 0.03
    fade = ss(0.2, 1.8, noise(n, n, 2.5, seed + 4, dims=dims, lo=3)) * 0.5
    c = col(*rgb)[None, None, :] * (1 + thread + heather)[..., None]
    pale = col(*rgb) * 0.6 + np.mean(rgb) * 0.4 + 0.04
    c = mix(c, pale[None, None, :], fade * 0.25)
    cav = np.clip((blur(wv, 2.0) - wv) * 2.5, 0, 1)
    c = c * (1 - cav * 0.32)[..., None]
    lint = ss(2.8, 3.6, noise(n, n, 0.6, seed + 9, dims=dims))
    c = mix(c, (col(*rgb) * 1.6 + 0.08)[None, None, :], lint * 0.25 + pill * 0.15)
    c = painterly(c, seed + 20, dims, value=0.09, warm=0.03, streak=0.05)
    rough = 0.88 + noise(n, n, 1.6, seed + 10, dims=dims) * 0.03 + cav * 0.06 - wv * 0.04
    save(name, c, TC.height_to_normal(hm, px), 1 - cav * 0.42, np.clip(rough, 0, 1), np.zeros((n, n)),
         dict(kind="tileable", tile_m=[tile, tile], px_per_m=n / tile, uv_per_m=1 / tile,
              grain="warp along V (down the garment)", material=dict(sheen=0.25)))


def quilt():
    """gambeson canvas: tabby linen canvas, quilting stitch lines along V every 4 cm (5 per tile), the padding puffed
    between them, running stitches in the grooves"""
    n = RES; tile = 0.20; dims = (tile, tile); px = tile / n
    h, top, j, i = weave_tabby(n, 110, 41, slub=0.3)
    x = (np.arange(n) + 0.5) / n
    X, Y = np.meshgrid(x, x)
    # channels 4 cm apart, gently wandering (hand quilting): a smooth low-frequency wobble, not a jagged crack
    wob = blur(noise(n, n, 3.5, 42, ax=1.0, ay=0.25, dims=dims), 6) * 0.010
    ch = ((X + wob) * 5) % 1.0
    puff = np.sin(np.clip(ch, 0, 1) * math.pi) ** 0.45      # padded tube
    groove = np.exp(-((np.minimum(ch, 1 - ch)) / 0.05) ** 2)
    # running stitch: dashes along the groove every ~6 mm
    dash = (np.sin(Y * tile / 0.006 * 2 * math.pi) > 0.1).astype(np.float32)
    stitch = groove * dash * ss(0.5, 0.9, np.exp(-((np.minimum(ch, 1 - ch)) / 0.012) ** 2))
    lumps = noise(n, n, 2.4, 43, dims=dims, lo=4) * 0.25
    hm = puff * 2.6e-3 + h * 0.18e-3 + lumps * 0.2e-3 - stitch * 0.25e-3
    base = col(0.56, 0.45, 0.28)[None, None, :] * (1 + (h - 0.6) * 0.10)[..., None]
    base = base * (1 - groove * 0.22 + (puff - 0.6) * 0.10)[..., None]
    base = mix(base, col(0.36, 0.29, 0.19)[None, None, :], stitch * 0.7)
    base = painterly(base, 44, dims, value=0.10, warm=0.04, streak=0.06, streak_ang=math.pi / 2)
    grime = ss(0.4, 1.6, noise(n, n, 2.2, 45, dims=dims)) * 0.18
    base = base * (1 - grime * (1 - puff * 0.5))[..., None]
    rough = 0.84 + groove * 0.08 - puff * 0.04 + noise(n, n, 1.5, 46, dims=dims) * 0.03
    ao = 1 - groove * 0.45 - (1 - puff) * 0.1
    save("civ_quilt", base, TC.height_to_normal(hm, px), np.clip(ao, 0, 1), np.clip(rough, 0, 1), np.zeros((n, n)),
         dict(kind="tileable", tile_m=[tile, tile], px_per_m=n / tile, uv_per_m=1 / tile,
              channels="5 quilting channels per tile along V (4 cm)", material=dict(sheen=0.15)))


# ============================================================================================ leather, straw, wraps
def leather(name, rgb, seed, grain):
    n = RES; tile = 0.30; dims = (tile, tile); px = tile / n
    hm, base, rough, ao = TC.leather_field(n, n, px, seed, dims, rgb=rgb, grain_mm=grain, wear=0.8)
    base = painterly(base, seed + 50, dims, value=0.08, warm=0.03, streak=0.03)
    save(name, base, TC.height_to_normal(hm, px), np.clip(ao, 0, 1), np.clip(rough, 0, 1), np.zeros((n, n)),
         dict(kind="tileable", tile_m=[tile, tile], px_per_m=n / tile, uv_per_m=1 / tile))


def straw():
    """plaited straw: rows along U (the hat's spiral plait), each row a 7-end herringbone plait, rows sewn edge to edge"""
    n = RES; tile = 0.12; dims = (tile, tile); px = tile / n
    rows = 10                                              # 12 mm plait rows
    x = (np.arange(n) + 0.5) / n; X, Y = np.meshgrid(x, x)
    r = (Y * rows) % 1.0; ri = np.floor(Y * rows).astype(int)
    k = 14.0                                               # straws cross the row diagonally (herringbone)
    u = X * rows * 3.0
    a = ((u + r * 1.0) * k / 3.0) % 1.0
    b = ((u - r * 1.0) * k / 3.0) % 1.0
    left = r < 0.5
    s = np.where(left, a, b)
    straw_h = np.sin(np.clip(s, 0, 1) * math.pi) ** 0.5
    edge = np.exp(-((np.minimum(r, 1 - r)) / 0.07) ** 2)
    mid = np.exp(-((r - 0.5) / 0.05) ** 2)
    hm = straw_h * 0.45e-3 * (1 - edge * 0.7) - mid * 0.1e-3 + noise(n, n, 2.0, 61, dims=dims) * 0.05e-3
    g = np.random.default_rng(62)
    tone = g.normal(0, 1, rows + 1)[ri] * 0.06
    base = col(0.78, 0.64, 0.36)[None, None, :] * (1 + tone + (straw_h - 0.7) * 0.18 - edge * 0.25)[..., None]
    streaks = noise(n, n, 1.4, 63, ax=1.0, ay=0.05, dims=dims) * 0.06
    base = base * (1 + streaks)[..., None]
    base = painterly(base, 64, dims, value=0.08, warm=0.05, streak=0.04)
    base = mix(base, col(0.42, 0.33, 0.18)[None, None, :], ss(1.0, 2.2, noise(n, n, 2.4, 65, dims=dims)) * 0.3)
    rough = 0.62 - straw_h * 0.12 + edge * 0.1
    save("civ_straw", base, TC.height_to_normal(hm, px), 1 - edge * 0.35, np.clip(rough, 0, 1), np.zeros((n, n)),
         dict(kind="tileable", tile_m=[tile, tile], px_per_m=n / tile, uv_per_m=1 / tile, rows="plait rows along U"))


def legwrap():
    """spiral leg wraps: u = one turn around the leg, v = 0.18 m (4 bands of 45 mm); band centre lines v = u/4 + k/4
    so the pattern is seamless in u (shifts one band per turn) and in v; each band overlaps the one below it"""
    n = RES; px_v = 0.18 / n
    x = (np.arange(n) + 0.5) / n; X, Y = np.meshgrid(x, x)
    t = (Y * 4 - X) % 1.0                                 # 0 at a band's lower edge -> 1 at its upper edge
    hgt = np.clip(t * 1.2, 0, 1) ** 0.6                     # each band rises over the lower band's edge
    ov = np.exp(-(t / 0.06) ** 2)                           # shadowed overlap line
    wv, top, j, i = weave_tabby(n, 90, 71, slub=0.3)
    hm = hgt * 1.2e-3 + wv * 0.15e-3
    base = col(0.64, 0.58, 0.47)[None, None, :] * (1 + (wv - 0.6) * 0.1 - ov * 0.45 + (t - 0.5) * 0.08)[..., None]
    base = painterly(base, 72, (0.3, 0.18), value=0.08, warm=0.03, streak=0.04)
    dirt = ss(0.9, 0.2, Y) * 0 + ss(0.5, 2.0, noise(n, n, 2.0, 73)) * 0.15
    base = base * (1 - dirt)[..., None]
    rough = 0.86 + ov * 0.06
    save("civ_legwrap", base, TC.height_to_normal(hm, px_v * 1.6, px_v), 1 - ov * 0.5, np.clip(rough, 0, 1),
         np.zeros((n, n)), dict(kind="atlas", uv_per_m=1.0, uv="u = turns around the leg, v = height / 0.18 m"))


# ============================================================================================ trim sheet
TRIM_STRIPS = {  # name: (v0, v1, metres of strip per 1.0 of u)
    "tablet": (0.75, 1.0, 0.12),
    "cord": (0.625, 0.75, 0.06),
    "stitch": (0.5, 0.625, 0.10),
    "lacing": (0.25, 0.5, 0.12),
    "fletch_wrap": (0.125, 0.25, 0.05),
    "leather_edge": (0.0, 0.125, 0.10),
}


def trim():
    n = RES
    base = np.zeros((n, n, 3), np.float32); hm = np.zeros((n, n), np.float32)
    rough = np.full((n, n), 0.8, np.float32); ao = np.ones((n, n), np.float32)
    x = (np.arange(n) + 0.5) / n
    for name, (v0, v1, mu) in TRIM_STRIPS.items():
        r0, r1 = int(round((1 - v1) * n)), int(round((1 - v0) * n))           # image rows (top = v 1)
        hh = r1 - r0
        yy = (np.arange(hh) + 0.5) / hh                                        # 0 top .. 1 bottom of the strip
        X, Yl = np.meshgrid(x, 1 - yy)                                         # Yl: 0 = v0 edge .. 1 = v1 edge
        if name == "tablet":
            # tablet-woven band: chevrons / diamonds in madder red, weld yellow, woad blue on undyed ground
            u = X * 8.0                                                        # 8 motifs per strip repeat
            d = np.abs(((u % 1.0) - 0.5)) * 2
            yv = np.abs(Yl - 0.5) * 2
            diamond = (d + yv) < 0.75
            inner = (d + yv) < 0.4
            border = (yv > 0.78)
            wv, _, _, _ = weave_tabby(hh if hh > 8 else 8, 60, 81)
            c = np.where(diamond[..., None], col(0.52, 0.12, 0.06), col(0.70, 0.62, 0.46))
            c = np.where(inner[..., None], col(0.70, 0.55, 0.14), c)
            c = np.where(border[..., None], col(0.10, 0.16, 0.30), c)
            ribs = 0.5 + 0.5 * np.cos(u * 2 * math.pi * 6)
            h = 0.4e-3 * (0.6 + 0.4 * ribs) + (border * 0.2e-3)
            c = c * (0.9 + 0.1 * ribs)[..., None]
            rgh = np.full(X.shape, 0.82)
        elif name == "cord":
            u = X * 20.0
            tw = np.sin((u + Yl * 1.2) * 2 * math.pi) * 0.5 + 0.5
            prof = np.sqrt(np.clip(1 - ((Yl - 0.5) / 0.45) ** 2, 0, 1))
            h = prof * (0.6 + 0.4 * tw) * 1.0e-3
            c = col(0.62, 0.52, 0.34)[None, None, :] * (0.7 + 0.3 * tw)[..., None] * (0.6 + 0.4 * prof)[..., None]
            rgh = np.full(X.shape, 0.75)
        elif name == "stitch":
            u = X * 30.0
            dash = ((u % 1.0) < 0.6) & (np.abs(Yl - 0.5) < 0.12)
            fold = np.exp(-((Yl - 0.15) / 0.06) ** 2)
            h = dash * 0.25e-3 - fold * 0.2e-3
            c = np.where(dash[..., None], col(0.72, 0.66, 0.52), col(0.45, 0.36, 0.26)) * (1 - fold * 0.3)[..., None]
            rgh = np.full(X.shape, 0.8)
        elif name == "lacing":
            u = X * 6.0
            f = u % 1.0
            e1 = np.abs(Yl - f) < 0.07
            e2 = np.abs(Yl - (1 - f)) < 0.07
            eyelet = ((np.abs(f - 0.0) < 0.06) | (np.abs(f - 1.0) < 0.06)) & ((np.abs(Yl - 0.08) < 0.06) | (np.abs(Yl - 0.92) < 0.06))
            lace = e1 | e2
            h = lace * 0.9e-3 - eyelet * 0.4e-3
            c = col(0.26, 0.16, 0.09)[None, None, :] * np.ones_like(X)[..., None]
            c = np.where(lace[..., None], col(0.55, 0.40, 0.24), c)
            c = np.where(eyelet[..., None], col(0.06, 0.05, 0.04), c)
            rgh = np.where(lace, 0.6, 0.7)
        elif name == "fletch_wrap":
            u = X * 25.0
            tw = 0.5 + 0.5 * np.sin((u + Yl * 0.3) * 2 * math.pi)
            h = tw * 0.2e-3
            c = col(0.55, 0.12, 0.08)[None, None, :] * (0.75 + 0.25 * tw)[..., None]
            rgh = np.full(X.shape, 0.55)
        else:  # leather_edge: a rolled, burnished leather binding
            prof = np.sqrt(np.clip(1 - ((Yl - 0.5) / 0.5) ** 2, 0, 1))
            h = prof * 0.8e-3
            c = col(0.22, 0.13, 0.075)[None, None, :] * (0.7 + 0.35 * prof)[..., None]
            rgh = 0.55 - prof * 0.15
        c = painterly(np.clip(c, 0, 1).astype(np.float32), 90 + r0, (0.2, 0.05), value=0.06, warm=0.02, streak=0.02)
        base[r0:r1] = c; hm[r0:r1] = h; rough[r0:r1] = rgh
        ao[r0:r1] = 1 - np.clip((blur(h, 2, wrap=False) - h) * 2500, 0, 0.5)
    save("civ_trim", base, TC.height_to_normal(hm, 0.12 / n), ao, np.clip(rough, 0, 1), np.zeros((n, n)),
         dict(kind="trim", uv_per_m=1.0,
              strips={k: {"v": [v0, v1], "m_per_u": mu} for k, (v0, v1, mu) in TRIM_STRIPS.items()}))


# ============================================================================================ iron, wood, feathers
def iron():
    n = RES; tile = 0.20; dims = (tile, tile); px = tile / n
    g = np.random.default_rng(101)
    hammer = TC.dimples((n, n), g, 500, (6, 22), (10e-6, 30e-6))
    pits = TC.dimples((n, n), g, 1500, (1, 3), (5e-6, 14e-6))
    hm = hammer + pits + noise(n, n, 2.6, 102, dims=dims) * 15e-6
    rust = ss(1.0, 2.2, noise(n, n, 2.2, 103, dims=dims)) * 0.8
    base = col(0.30, 0.29, 0.28)[None, None, :] * (1 + noise(n, n, 2.0, 104, dims=dims)[..., None] * 0.08)
    base = mix(base, col(0.30, 0.16, 0.08)[None, None, :], rust * 0.6)
    base = painterly(base, 105, dims, value=0.08, warm=0.02, streak=0.03)
    metal = np.clip(1 - rust * 0.9, 0, 1)
    rough = 0.42 + rust * 0.4 + noise(n, n, 1.8, 106, dims=dims) * 0.05
    save("civ_iron", base, TC.height_to_normal(hm, px), np.ones((n, n)) - rust * 0.15, np.clip(rough, 0, 1), metal,
         dict(kind="tileable", tile_m=[tile, tile], px_per_m=n / tile, uv_per_m=1 / tile))


def wood():
    """polished wood, grain along V: yew-like honey heartwood with darker streaks and pin knots"""
    n = RES; tile = 0.40; dims = (tile, tile); px = tile / n
    x = (np.arange(n) + 0.5) / n; X, Y = np.meshgrid(x, x)
    warp = noise(n, n, 2.8, 111, ax=1.0, ay=0.04, dims=dims) * 0.08
    rings = np.sin((X + warp) * 2 * math.pi * 26) * 0.5 + 0.5
    fine = noise(n, n, 1.0, 112, ax=1.0, ay=0.02, dims=dims)
    knots = np.zeros((n, n), np.float32)
    g = np.random.default_rng(113)
    for _ in range(6):
        cx, cy = g.uniform(0, n, 2)
        d = np.hypot(((X * n - cx + n / 2) % n) - n / 2, ((Y * n - cy + n / 2) % n) - n / 2)
        knots += np.exp(-(d / g.uniform(3, 7)) ** 2)
    hm = (rings * 8e-6 + fine * 4e-6) - knots * 10e-6
    base = mix(col(0.55, 0.32, 0.16)[None, None, :] * np.ones((n, n, 1)), col(0.36, 0.18, 0.08)[None, None, :],
               rings * 0.45 + fine * 0.05)
    base = mix(base, col(0.16, 0.08, 0.04)[None, None, :], np.clip(knots, 0, 1) * 0.8)
    base = painterly(base, 114, dims, value=0.07, warm=0.03, streak=0.05, streak_ang=math.pi / 2)
    rough = 0.38 + rings * 0.08 + knots * 0.1
    save("civ_wood", base, TC.height_to_normal(hm, px), np.ones((n, n)) - knots * 0.2, np.clip(rough, 0, 1),
         np.zeros((n, n)), dict(kind="tileable", tile_m=[tile, tile], px_per_m=n / tile, uv_per_m=1 / tile,
                                grain="along V"))


def fletch():
    """feather cards: 3 vanes across U (grey goose, grey goose, madder-dyed cock feather), quill along V
    (V 0 = the nock end, 1 = the front); alpha = vane coverage (MASK)"""
    n = RES
    x = (np.arange(n) + 0.5) / n; X, Y = np.meshgrid(x, x)
    Yv = 1 - Y                                             # image row 0 = V 1
    base = np.zeros((n, n, 3), np.float32); alpha = np.zeros((n, n), np.float32); hm = np.zeros((n, n), np.float32)
    cols = [col(0.55, 0.53, 0.50), col(0.50, 0.49, 0.47), col(0.60, 0.16, 0.10)]
    for k in range(3):
        u = (X - k / 3) * 3                                # 0..1 inside the card
        inside = (u >= 0) & (u < 1)
        # vane outline: a parabolic-cut vane, quill at u = 0.08
        vv = np.clip(Yv, 0, 1)
        width = 0.85 * np.sqrt(np.clip(np.sin(vv * math.pi * 0.95 + 0.1), 0, 1)) * ss(0.0, 0.12, vv) * (1 - 0.3 * vv)
        vane = inside & (u > 0.08) & (u < 0.08 + width)
        quill = inside & (np.abs(u - 0.08) < 0.018)
        barbs = 0.5 + 0.5 * np.sin((u * 1.0 - vv * 3.0) * 2 * math.pi * 18)
        c = cols[k][None, None, :] * (0.75 + 0.25 * barbs)[..., None]
        c = c * (0.8 + 0.2 * (u - 0.08) / 0.85)[..., None]
        tip = ss(0.7, 1.0, (u - 0.08) / np.maximum(width, 1e-3))
        c = c * (1 - 0.25 * tip)[..., None]
        base = np.where(vane[..., None], c, base)
        base = np.where(quill[..., None], col(0.85, 0.82, 0.74), base)
        ragged = noise(n, n, 1.2, 120 + k) * 0.05
        a = (vane & ((u - 0.08) < width * (0.93 + ragged))) | quill
        alpha = np.maximum(alpha, a.astype(np.float32))
        hm += (barbs * 20e-6) * vane + quill * 60e-6
    base = painterly(np.clip(base, 0, 1), 125, (0.1, 0.1), value=0.05, warm=0.02, streak=0.02)
    save("civ_fletch", base, TC.height_to_normal(hm, 0.1 / n), np.ones((n, n)), np.full((n, n), 0.6),
         np.zeros((n, n)), dict(kind="atlas", uv_per_m=1.0, alpha_cut=0.4,
                                cards="3 feathers across U (u 0-1/3, 1/3-2/3, 2/3-1); quill at u = card + 0.027"),
         alpha=alpha)


SETS = {"civ_linen": linen, "civ_quilt": quilt, "civ_straw": straw, "civ_legwrap": legwrap, "civ_trim": trim,
        "civ_iron": iron, "civ_wood": wood, "civ_fletch": fletch,
        "civ_leather_tan": lambda: leather("civ_leather_tan", (0.36, 0.22, 0.12), 131, 1.1),
        "civ_leather_dark": lambda: leather("civ_leather_dark", (0.14, 0.085, 0.05), 137, 1.4)}
for _w in WOOLS:
    SETS[_w] = (lambda nm: (lambda: wool(nm)))(_w)


def main(argv):
    global RES
    if "--res" in argv:
        RES = int(argv[argv.index("--res") + 1]); argv = [a for a in argv if a not in ("--res", str(RES))]
    TC.RES = RES
    names = [a for a in argv if not a.startswith("--")] or list(SETS)
    os.makedirs(OUTD, exist_ok=True)
    mp = os.path.join(OUTD, "civ_textures.json")
    old = json.load(open(mp))["sets"] if os.path.exists(mp) else {}
    for nm in names:
        t0 = time.time(); SETS[nm](); P("%s %.1fs" % (nm, time.time() - t0))
    old.update(MAN)
    json.dump(dict(generator="scripts/outfit_civ_tex.py", res=RES, conventions="as textures_char.json", sets=dict(sorted(old.items()))),
              open(mp, "w"), indent=1)
    # contact sheet
    os.makedirs(REND, exist_ok=True)
    from civ_sheet import sheet
    tiles = []
    for nm in sorted(old):
        f = os.path.join(OUTD, old[nm]["files"]["base"])
        tiles.append(f)
    sheet(os.path.join(REND, "civ_textures_sheet.png"), tiles, cols=5, h=300)


if __name__ == "__main__":
    main(sys.argv[1:])
