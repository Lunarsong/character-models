#!/usr/bin/env python3
"""Texture sets owned by the knight LOWER armour (scripts/armour_lower.py), written to assets/textures/knight_lower/.

System python3 + numpy + scipy + Pillow. Reuses the shared painter and heraldry of the materials pipeline
(scripts/textures_char.py `Panel`, scripts/heraldry.py lion / fleur-de-lis / star), so the emblems match the rest of the
kit, but paints them onto the EXACT outlines of this kit's panels (read from the MPFB asset metadata written by
`armour_lower.py -- author`), so borders follow the tapered tabard and the chevron cape hem with no UV distortion.

Sets (each <name>_base.png sRGB, <name>_normal.png OpenGL +Y, <name>_orm.png R=AO G=roughness B=metallic):
  kl_trim      2048 x 128  gold filigree trim strip: u along (0.20 m per repeat), v across the band (0..1)
  kl_strap     2048 x 256  stitched leather belt strip: u along (0.40 m per repeat), v across the belt profile
  kl_tabard    2048 x 2048 tabard atlas: front panel u 0..0.5, back u 0.5..1, canvas 0.42 x 1.05 m each, v = 1 top
  kl_cape      2048 x 2048 cape outer face, canvas from the cape asset (1.02 x 1.50 m), v = 1 at the shoulders
  kl_mail_opaque 2048 x 2048  mail_riveted with the ring gaps baked dark (opaque use under / over deleted skin)
usage: python3 scripts/armour_lower_tex.py [trim strap tabard cape mail props]
"""
import os, sys, json, math
import numpy as np
from PIL import Image
from scipy import ndimage

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import textures_char as TC          # shared painter (Panel, noise, height_to_normal, leather_field, ...)
import heraldry as HR

CH = os.path.dirname(HERE)
TEX = os.path.join(CH, "assets", "textures")
OUT = os.path.join(TEX, "knight_lower")
CLOTHES = os.path.join(CH, "assets", "mpfb_assets", "clothes")
MAN = {}
GOLD = (0.86, 0.66, 0.33)


def log(*a):
    print("[kl-tex]", *a, flush=True)


def save(name, base, normal, ao, rough, metal, meta=None):
    os.makedirs(OUT, exist_ok=True)
    Image.fromarray(TC.to8(np.clip(base, 0, 1))).save(os.path.join(OUT, name + "_base.png"), optimize=True)
    Image.fromarray(TC.to8(normal * 0.5 + 0.5, dither=True, seed=3)).save(os.path.join(OUT, name + "_normal.png"), optimize=True)
    Image.fromarray(np.dstack([TC.to8(np.clip(ao, 0, 1)), TC.to8(np.clip(rough, 0, 1)), TC.to8(np.clip(metal, 0, 1))])).save(
        os.path.join(OUT, name + "_orm.png"), optimize=True)
    m = dict(meta or {}); m["files"] = {k: name + "_%s.png" % k for k in ("base", "normal", "orm")}
    m["res"] = [int(base.shape[1]), int(base.shape[0])]
    MAN[name] = m
    log("saved", name, m["res"])


def ss(e0, e1, x):
    return TC.ss(e0, e1, x)


# --------------------------------------------------------------------------------------------- gold filigree strip
def trim_set(W=2048, H=128, L=0.20, band_mm=12.0, seed=11):
    """Raised gold filigree: rolled beads along both edges, a running vine scroll with volutes and leaves between
    them on a finely hammered, slightly recessed ground. Periodic along u."""
    px = L / W                      # metres per pixel along (~0.1 mm); across ~ band_mm / H
    py = band_mm * 1e-3 / H
    y = (np.arange(H) + 0.5) / H
    Y = np.repeat(y[:, None], W, 1)
    # beads (half-round) at both edges
    bead = np.zeros((H, W), np.float32)
    for c, w in ((0.075, 0.075), (0.925, 0.075)):
        bead = np.maximum(bead, np.sqrt(np.clip(1 - ((Y - c) / w) ** 2, 0, 1)))
    # vine scroll drawn in pixel space (4 periods over the strip -> periodic), rasterised supersampled
    per = W / 4
    shapes = []
    for k in range(-1, 5):
        x0 = k * per
        xs = np.linspace(x0, x0 + per, 60)
        ys = H * (0.5 + 0.20 * np.sin((xs - x0) / per * 2 * math.pi))
        shapes.append(("add", HR.taper(np.stack([xs, ys], -1), 9.0, 9.0, cap0="round", cap1="round")))
        for sgn, xc in ((1, x0 + per * 0.25), (-1, x0 + per * 0.75)):
            # volute curling from the vine towards the band centre line + a leaf on the other side
            yc = H * (0.5 + 0.20 * sgn)
            cx, cy = xc + per * 0.11, yc - sgn * H * 0.13
            shapes.append(("add", HR.spiral((cx, cy), 24, 4, math.pi / 2 * sgn + math.pi, 1.3, 8.0, 4.0, cw=sgn < 0)))
            lx = xc - per * 0.12
            leaf = np.array([[lx - 44, yc], [lx - 18, yc - sgn * 19], [lx + 22, yc - sgn * 17], [lx + 48, yc - sgn * 3],
                             [lx + 20, yc + sgn * 4], [lx - 14, yc + sgn * 5]])
            shapes.append(("add", HR.catmull(leaf, n=6, closed=True)))
            shapes.append(("add", HR.ellipse(xc + per * 0.25, H * 0.5 - sgn * H * 0.03, 8, 8)))
    cov, hi = HR.raster(shapes, W, H, ss=4)
    d = ndimage.distance_transform_edt(hi).astype(np.float32)
    d = d.reshape(H, 4, W, 4).mean((1, 3)) / 4.0            # px
    vine = np.clip(d / 2.6, 0, 1) ** 0.6 * cov
    g = np.random.default_rng(seed)
    ham = TC.noise(H, W, 2.2, seed, dims=(L, band_mm * 1e-3), lo=900) * 0.5
    inner = (Y > 0.15) & (Y < 0.85)
    h = np.where(inner, 0.10e-3 + ham * 0.012e-3, 0) + bead * 0.55e-3
    h = np.maximum(h, vine * 0.40e-3 + 0.08e-3 * cov)
    h = h + TC.noise(H, W, 3.2, seed + 1, dims=(L, band_mm * 1e-3), lo=300) * 4e-6
    nrm = TC.height_to_normal(h, px, py, wrap=True)
    cav = np.clip((TC.blur(h, 2.0) - h) / 0.08e-3, 0, 1)
    crown = np.clip((h - TC.blur(h, 3.0)) / 0.06e-3, 0, 1)
    tarn = np.clip(cav * 0.8 + ss(1.0, 2.2, TC.noise(H, W, 2.4, seed + 2, dims=(L, band_mm * 1e-3))) * 0.3, 0, 1)
    base = TC.col(*GOLD)[None, None, :] * (1 + TC.noise(H, W, 2.0, seed + 3, dims=(L, 0.012)) * 0.025)[..., None]
    base = TC.mix(base, TC.col(0.98, 0.86, 0.56)[None, None, :], crown * 0.45)
    base = TC.mix(base, TC.col(0.36, 0.25, 0.11)[None, None, :], tarn * 0.7)
    ground = inner & (vine < 0.05)
    base = TC.mix(base, TC.col(0.30, 0.20, 0.09)[None, None, :], ground * 0.55)      # antiqued recess
    tarn = np.maximum(tarn, ground * 0.6)
    rough = 0.24 + tarn * 0.28 - crown * 0.06
    metal = 1.0 - tarn * 0.15
    ao = 1 - cav * 0.45
    save("kl_trim", base, nrm, ao, rough, metal,
         dict(kind="strip", along_m=L, note="u along the trim (0.20 m per repeat), v across the band 0..1"))


# --------------------------------------------------------------------------------------------- stitched strap
def strap_set(W=2048, H=256, L=0.40, prof_m=0.067, seed=21):
    """Belt leather across the belt profile used by armour_lower.belt_ring: v 0..0.07 inner top, 0.07..0.14 rounded
    top edge, 0.14..0.86 outer face, 0.86..0.93 bottom edge, 0.93..1 inner bottom. Periodic along u."""
    px = L / W; py = prof_m / H
    hm, base, rough, ao = TC.leather_field(H, W, px, seed, (L, prof_m), rgb=(0.23, 0.135, 0.075), grain_mm=1.1, wear=1.2)
    v = 1 - (np.arange(H) + 0.5) / H           # image row 0 = v 1
    V = np.repeat(v[:, None], W, 1)
    X = np.repeat(((np.arange(W) + 0.5) * px)[None, :], H, 0)
    # burnished, darker edges; a creased groove parallel to each edge
    edge = np.clip(1 - np.minimum(np.abs(V - 0.105), np.abs(V - 0.895)) / 0.05, 0, 1)
    groove = np.exp(-((np.minimum(np.abs(V - 0.175), np.abs(V - 0.825))) / 0.008) ** 2)
    hm = hm - groove * 60e-6
    # saddle stitches: slanted dashes 3.2 mm long every 4 mm, rows just inside the grooves
    stitch = np.zeros((H, W), np.float32)
    for vr in (0.215, 0.785):
        t = (X / 0.004) % 1.0
        on = (t > 0.1) & (t < 0.9)
        dv = (V - vr) * prof_m - (t - 0.5) * 0.0006      # slight slant
        prof = np.clip(1 - (dv / 0.0006) ** 2, 0, 1)
        stitch = np.maximum(stitch, np.sqrt(prof) * on)
        hole = np.exp(-(((t - 0.05) * 0.004) ** 2 + (dv) ** 2) / (0.0004 ** 2))
        hm = hm - hole * 40e-6
    hm = hm + stitch * 0.25e-3
    base = TC.mix(base, TC.col(0.10, 0.055, 0.03)[None, None, :], edge * 0.55)
    base = TC.mix(base, TC.col(0.12, 0.07, 0.04)[None, None, :], groove * 0.4)
    base = TC.mix(base, TC.col(0.56, 0.44, 0.28)[None, None, :] * (0.8 + 0.3 * stitch[..., None]), (stitch > 0.05) * 1.0)
    rough = np.where(stitch > 0.05, 0.75, rough - edge * 0.12)
    nrm = TC.height_to_normal(hm, px, py, wrap=True)
    ao = ao * (1 - groove * 0.3)
    save("kl_strap", base, nrm, ao, rough, np.zeros_like(rough),
         dict(kind="strip", along_m=L, note="u along the belt (0.40 m per repeat), v across the belt profile"))


# --------------------------------------------------------------------------------------------- tabard / cape
def load_meta(asset):
    p = os.path.join(CLOTHES, asset, asset + ".rts.json")
    return json.load(open(p)) if os.path.exists(p) else None


def poly_top_open(poly, lift=0.25):
    """Move the top edge (y ~ 0) off the canvas so the border runs only down the sides and along the hem."""
    return [(x, (y - lift if y < 1e-4 else y)) for x, y in poly]


def hem_y(poly, xc):
    P = np.array(poly)
    near = P[np.abs(P[:, 0] - xc) < 0.03]
    return float(near[:, 1].max()) if len(near) else float(P[:, 1].max())


def fit(shapes, cx, cy, w=None, h=None):
    return HR.fit(shapes, cx, cy, w=w, h=h)


def tabard_set(seed=808):
    meta = load_meta("rts_knight_tabard")
    Wc, Hc = meta["canvas_m"] if meta else (0.42, 1.05)
    outl = meta["outline_m"] if meta else None
    n = 2048; W = n // 2
    halves = []
    for k, part in enumerate(("front", "back")):
        log("tabard", part)
        pnl = TC.Panel(W, n, Wc, Hc, seed + k, wear=0.45, dirt_bottom=0.10)
        poly = outl[part] if outl else [(0.02, 0), (Wc - 0.02, 0), (Wc - 0.02, 0.95), (0.02, 0.95)]
        pnl.border(poly_top_open(poly), width_m=0.026, inset_m=0.004, stud_every_m=0.065, stud_d_m=0.0065,
                   pattern_m=0.02)
        cx = Wc / 2
        hy = hem_y(poly, cx)
        if part == "front":
            pnl.embroider(fit(HR.star8(), cx, 0.075, h=0.058))
            pnl.embroider(fit(HR.emblem(with_star=False), cx, 0.285, w=0.255))
            pnl.embroider(fit(HR.fleur_de_lis(), cx, hy - 0.125, h=0.085))
            for sx in (-1, 1):
                pnl.embroider(fit(HR.sprig(), cx + sx * 0.058, hy - 0.112, h=0.050))
        else:
            pnl.embroider(fit(HR.star8(), cx, 0.110, h=0.060))
            pnl.embroider(fit(HR.fleur_de_lis(), cx, hy - 0.125, h=0.085))
        halves.append(pnl)
    base = np.concatenate([p.base for p in halves], 1); rough = np.concatenate([p.rough for p in halves], 1)
    metal = np.concatenate([p.metal for p in halves], 1)
    nrm = np.concatenate([p.finish() for p in halves], 1)
    ao = np.concatenate([p.ao for p in halves], 1)
    save("kl_tabard", base, nrm, ao, rough, metal,
         dict(kind="atlas", canvas_m=[Wc, Hc], regions={"front": [0.0, 0.5], "back": [0.5, 1.0]},
              note="painted on the exact panel outlines from rts_knight_tabard.rts.json (canvas metres, y down)"))


def cape_set(seed=909):
    meta = load_meta("rts_knight_cape")
    Wc, Hc = meta["canvas_m"] if meta else (0.80, 1.35)
    poly = meta["outline_m"] if meta else [(0.2, 0), (0.6, 0), (0.72, 1.2), (0.4, 1.28), (0.08, 1.2)]
    n = 2048
    log("cape")
    pnl = TC.Panel(n, n, Wc, Hc, seed, wear=0.55, dirt_bottom=0.22)
    pnl.border(poly, width_m=0.036, inset_m=0.004, stud_every_m=0.080, stud_d_m=0.008, pattern_m=0.024)
    cx = Wc / 2
    hy = hem_y(poly, cx)
    P = np.array(poly)
    # sizes scale with the cape width (assembly pass: the cape was widened from 0.80 to 1.02 m of canvas)
    ks = Wc / 0.80
    pnl.embroider(fit(HR.star8(), cx, 0.125, h=0.085 * ks ** 0.5))
    pnl.embroider(fit(HR.emblem(with_star=False), cx, 0.45, w=0.33 * ks ** 0.8))
    pnl.embroider(fit(HR.fleur_de_lis(), cx, hy - 0.12, h=0.115 * ks ** 0.5))
    for sx in (-1, 1):
        xs = cx + sx * 0.165 * ks
        near = P[(np.abs(P[:, 0] - xs) < 0.06) & (P[:, 1] > Hc * 0.5)]       # hem points only (not the top edge)
        hs = float(near[:, 1].max()) if len(near) else hy - 0.05
        pnl.embroider(fit(HR.fleur_de_lis(), xs, hs - 0.085, h=0.055 * ks ** 0.5))
    nrm = pnl.finish()
    save("kl_cape", pnl.base, nrm, pnl.ao, pnl.rough, pnl.metal,
         dict(kind="atlas", canvas_m=[Wc, Hc], note="outer face on the exact cape outline (rts_knight_cape.rts.json)"))


def mail_set():
    """mail_riveted with the ring gaps baked dark (its alpha): used opaque so nothing shows through to deleted skin."""
    src = os.path.join(TEX, "mail_riveted_base.png")
    im = np.asarray(Image.open(src).convert("RGBA"), np.float32) / 255.0
    a = im[..., 3]
    base = im[..., :3] * (0.12 + 0.88 * a[..., None])
    Image.fromarray(TC.to8(base)).save(os.path.join(OUT, "kl_mail_opaque_base.png"), optimize=True)
    MAN["kl_mail_opaque"] = dict(kind="tileable", from_set="mail_riveted", files={
        "base": "kl_mail_opaque_base.png", "normal": "../mail_riveted_normal.png", "orm": "../mail_riveted_orm.png"})
    log("saved kl_mail_opaque")


SETS = {"trim": trim_set, "strap": strap_set, "tabard": tabard_set, "cape": cape_set, "mail": mail_set}

if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    mp = os.path.join(OUT, "kl_textures.json")
    if os.path.exists(mp):
        MAN.update(json.load(open(mp)))
    names = [a for a in sys.argv[1:] if a in SETS] or list(SETS)
    for nm in names:
        if nm == "props":
            continue
        SETS[nm]()
    if "props" in sys.argv[1:] or not sys.argv[1:]:
        try:
            import armour_lower_props_tex as PT
            PT.props_set(save, MAN)
        except ImportError:
            pass
    json.dump(MAN, open(mp, "w"), indent=1)
