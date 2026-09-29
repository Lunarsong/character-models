#!/usr/bin/env python3
"""M3 steel PBR fix (judge iteration 1): the plate steel was base colour 0.171 linear (sRGB 0.45) with roughness 0.36,
i.e. a dark, glossy metal: 'black chrome' that mirrors the dark studio / sky (greaves near-black, pauldrons blown out,
the unit a dark figure at RTS zoom). Real polished-to-worn steel has F0 ~0.50-0.56 linear.

What this does (numpy + Pillow, deterministic), per steel pixel (metallic > 0.5 and low saturation, so the gold trims,
the leather and the dark interior stay as they are):
  - clean steel: base colour 0.50-0.56 linear (slightly warm, (1.02, 1.0, 0.955) x value), keeping the texgen's
    relative value variation (brushing, hammer marks, polish) compressed into that band;
  - roughness remapped into 0.40-0.60 (the texgen's pattern kept, just lifted), streaks along U stay;
  - a NON-METAL grime layer instead of dark metal: where the texgen had painted darkening (grime clouds / pits: pixels
    darker than the median) the pixel gets a dielectric film blended in: soft patina clouds (up to 55 %) and small dark
    pits (up to 90 %) of dark brown-grey (0.045 linear), rough (0.78), metallic 1 - 0.95 x grime. Crevice AO stays in the ORM red channel and the per-vertex AO (COLOR_0).
Outputs go to assets/textures/_m3/ with the SAME file names (build_knight swaps the images at assembly, so the
texgen owners' files are never touched); `python3 scripts/steel_pbr.py [--check]` rebuilds them and prints the stats.
Sets: the upper trim sheet (knight_armour_*, used by every plate piece, upper and lower) and the tileable steel set
steel_worn (sword blade, shield boss / rim); steel_blued (the sword's enamel lozenge) keeps its dark blued look.
"""
import os, sys, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TEX = os.path.join(os.path.dirname(HERE), "assets", "textures")
OUT = os.path.join(TEX, "_m3")
SETS = [  # (base, orm, whole_image_is_steel)
    ("knight_armour_base.png", "knight_armour_orm.png", False),
    ("steel_worn_base.png", "steel_worn_orm.png", True),
]                                          # steel_blued (enamel) keeps its dark blued look
STEEL_LO, STEEL_HI = 0.50, 0.56          # linear base colour band for clean steel
TINT = np.array([1.02, 1.0, 0.955])      # slightly warm (reference: aged steel, not blue chrome)
ROUGH_LO, ROUGH_HI = 0.40, 0.60
GRIME_RGB = np.array([0.050, 0.043, 0.036])
GRIME_ROUGH = 0.78


def s2l(c):
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def l2s(c):
    c = np.clip(c, 0, 1)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * c ** (1 / 2.4) - 0.055)


def luma(c):
    return 0.2126 * c[..., 0] + 0.7152 * c[..., 1] + 0.0722 * c[..., 2]


def box_blur(a, r):
    """separable box blur (wraps: the steel tiles), radius r px"""
    k = 2 * r + 1
    for ax in (0, 1):
        c = np.cumsum(np.concatenate([a.take(range(-r - 1, 0), axis=ax), a, a.take(range(0, r), axis=ax)], axis=ax), axis=ax)
        a = (c.take(range(k, c.shape[ax]), axis=ax) - c.take(range(0, c.shape[ax] - k), axis=ax)) / k
    return a


def fix(base_path, orm_path, whole):
    from PIL import Image
    b = np.asarray(Image.open(base_path).convert("RGBA")).astype(np.float64) / 255.0
    alpha = b[..., 3:4]
    o = np.asarray(Image.open(orm_path).convert("RGB")).astype(np.float64) / 255.0
    lin = s2l(b[..., :3])
    lum = luma(lin)
    mx, mn = lin.max(-1), lin.min(-1)
    sat = (mx - mn) / (mx + 1e-6)
    steel = (o[..., 2] > 0.5) & (sat < 0.30) if not whole else np.ones(lum.shape, bool)
    st = lum[steel]
    p5, p50, p95 = np.percentile(st, [5, 50, 95])
    # relative value: the texgen's brushing / hammering / polish variation
    rel = np.clip((lum - p50) / max(p95 - p5, 1e-4), -1.5, 1.5)
    # grime from the texgen's painted darkening, measured on steel pixels only (masked blur: no halos where the steel
    # meets a gold trim or the lion): soft patina clouds (a thin dielectric film, <= 0.55) + small dark pits (<= 0.9)
    m = steel.astype(np.float64)
    blur = lambda a, r: box_blur(a * m, r) / np.maximum(box_blur(m, r), 1e-3)
    low = blur(lum, 10)
    p3, p60 = np.percentile(low[steel], [3, 60])
    cloud = np.clip((p60 - low) / max(p60 - p3, 1e-4), 0, 1)
    cloud = 0.55 * (cloud * cloud * (3 - 2 * cloud)) ** 1.2
    pit = np.clip((blur(lum, 3) - lum) / max(p50 - p5, 1e-4) - 0.35, 0, 1)
    pit = 0.9 * np.clip(pit * 2.0, 0, 1)
    grime = np.maximum(cloud, pit) * m
    val = (STEEL_LO + STEEL_HI) / 2 + rel * (STEEL_HI - STEEL_LO) / 2
    val = np.clip(val, STEEL_LO - 0.02, STEEL_HI + 0.02)
    # keep the hue of tinted steel (blued) but normalise its value; plain steel gets the warm tint
    hue = lin / np.maximum(lum[..., None], 1e-4)
    hue = np.where((sat > 0.12)[..., None], hue, TINT)
    clean = hue * val[..., None]
    col = clean * (1 - grime[..., None]) + GRIME_RGB * grime[..., None]
    r0 = o[..., 1]
    rp5, rp95 = np.percentile(r0[steel], [5, 95])
    rough = ROUGH_LO + np.clip((r0 - rp5) / max(rp95 - rp5, 1e-4), 0, 1) * (ROUGH_HI - ROUGH_LO)
    rough = rough * (1 - grime) + GRIME_ROUGH * grime
    metal = 1.0 - grime * 0.95
    nb = lin.copy(); nb[steel] = col[steel]
    no = o.copy(); no[..., 1][steel] = rough[steel]; no[..., 2][steel] = metal[steel]
    ob = np.concatenate([l2s(nb), alpha], -1)
    stats = dict(steel_px=float(steel.mean()), before=dict(base_p50=round(float(p50), 3),
                 rough_p50=round(float(np.percentile(r0[steel], 50)), 3)),
                 after=dict(base_clean_p50=round(float(np.percentile(val[steel & (grime < 0.1)], 50)), 3),
                            base_p10_p90=[round(float(x), 3) for x in np.percentile(luma(nb)[steel], [10, 90])],
                            rough_p10_p50_p90=[round(float(x), 3) for x in np.percentile(no[..., 1][steel], [10, 50, 90])],
                            grime_frac=round(float((grime[steel] > 0.3).mean()), 3),
                            metal_p10=round(float(np.percentile(no[..., 2][steel], 10)), 3)))
    return ob, no, stats


def main(check=False):
    from PIL import Image
    os.makedirs(OUT, exist_ok=True)
    rep = {}
    for bn, on, whole in SETS:
        bp, op = os.path.join(TEX, bn), os.path.join(TEX, on)
        if not (os.path.exists(bp) and os.path.exists(op)):
            continue
        ob, no, st = fix(bp, op, whole)
        mode = "RGBA" if (ob[..., 3] < 0.999).any() else "RGB"
        img = (np.clip(ob if mode == "RGBA" else ob[..., :3], 0, 1) * 255 + 0.5).astype(np.uint8)
        Image.fromarray(img).save(os.path.join(OUT, bn), optimize=True)
        Image.fromarray((np.clip(no, 0, 1) * 255 + 0.5).astype(np.uint8)).save(os.path.join(OUT, on), optimize=True)
        rep[bn] = st
        print("STEEL %-24s steel px %4.0f%%  base p50 %.3f -> clean %.3f (p10-p90 %s)  rough %.2f -> %s  grime %.0f%%"
              % (bn, st["steel_px"] * 100, st["before"]["base_p50"], st["after"]["base_clean_p50"],
                 st["after"]["base_p10_p90"], st["before"]["rough_p50"], st["after"]["rough_p10_p50_p90"],
                 st["after"]["grime_frac"] * 100), flush=True)
    json.dump(rep, open(os.path.join(OUT, "steel_pbr.json"), "w"), indent=1)
    if check:
        for st in rep.values():
            a = st["after"]
            assert STEEL_LO - 0.005 <= a["base_clean_p50"] <= STEEL_HI + 0.005, a
            assert ROUGH_LO - 0.01 <= a["rough_p10_p50_p90"][1] <= ROUGH_HI + 0.05, a
    return rep


def swap_images(bpy, log=print):
    """inside Blender: point every image whose file has an _m3 version at it (idempotent)"""
    n = 0
    for im in bpy.data.images:
        f = os.path.basename(bpy.path.abspath(im.filepath))
        p = os.path.join(OUT, f)
        cur = os.path.normpath(bpy.path.abspath(im.filepath))
        if f and os.path.exists(p) and cur != p and "_m3" not in cur:
            im.filepath = p; im.reload(); n += 1
    log("images swapped to the M3 steel versions: %d" % n)
    return n


if __name__ == "__main__":
    main("--check" in sys.argv)
