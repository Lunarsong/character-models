"""Procedural hair-card texture atlas (numpy + Pillow, system python3). No external art: fully CC0 / our own.

Atlas: 2048 x 2048, 8 vertical card columns (256 px wide, strands run along V = image rows, root at the top).
  col 0     solid   : opaque hair 'fabric' (dense strands over a dark base) for braid tubes and base cap cards;
                      only the last 8 % (tip) fades
  cols 1-2  dense   : full clumps, soft side falloff so neighbouring cards blend
  cols 3-5  medium  : clumps with gaps
  cols 6-7  wispy   : thin sparse strands for hairline fringes and flyaways
Outputs (assets/textures/):
  hair_strands_<name>_base.png    sRGB albedo (hair colour baked in) + alpha
  hair_strands_normal.png         tangent-space normal (strands as cylinders across U), shared by all colours
usage: python3 scripts/hair_tex.py <name> <r> <g> <b> [<name> <r> <g> <b> ...]   (colour 0..1, sRGB)
"""
import sys, os
import numpy as np
from PIL import Image

CH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEX = os.path.join(CH, "assets", "textures")
W = H = 2048
COLS = 8
CW = W // COLS
rng = np.random.default_rng(7)
KIND = ["solid", "dense", "dense", "medium", "medium", "medium", "wispy", "wispy"]


def smooth(x):
    x = np.clip(x, 0, 1)
    return x * x * (3 - 2 * x)


def build():
    lum = np.zeros((H, W), np.float32)
    cov = np.zeros((H, W), np.float32)
    nx = np.zeros((H, W), np.float32)
    for c in range(COLS):
        kind = KIND[c]
        n = {"solid": 700, "dense": 520, "medium": 260, "wispy": 55}[kind]
        x0 = c * CW
        nclump = {"solid": 9, "dense": 5, "medium": 3, "wispy": 4}[kind]
        half = {"solid": 0.36, "dense": 0.25, "medium": 0.3, "wispy": 0.36}[kind]    # clumps stay inside: sparse edges
        centres = x0 + CW * (0.5 + np.linspace(-half, half, nclump) + rng.normal(0, 0.03, nclump))
        for s in range(n):
            cc = rng.choice(centres)
            spread = {"solid": 0.09, "dense": 0.085, "medium": 0.06, "wispy": 0.12}[kind] * CW
            base = rng.normal(cc, spread)
            if not (x0 + 5 <= base <= x0 + CW - 6):          # strands that would leave the column are dropped
                continue
            start = rng.uniform(0, 0.02 if kind in ("solid", "dense") else 0.08) * H
            length = (rng.uniform(0.85, 1.0) if kind != "wispy" else rng.uniform(0.5, 1.0)) * H
            end = min(H - 1, start + length)
            amp = rng.uniform(0.4, 2.5); freq = rng.uniform(0.6, 2.0) * 2 * np.pi / H; ph = rng.uniform(0, 6.28)
            width = rng.uniform(1.0, 2.2) if kind != "wispy" else rng.uniform(0.7, 1.3)
            bright = rng.uniform(0.6, 1.0)
            ys = np.arange(int(start), int(end))
            t = np.clip((ys - start) / max(1.0, end - start), 0, 1)
            # clumps converge towards their centre at the tip (pointed locks)
            xs = base + (cc - base) * 0.45 * t ** 1.5 + amp * np.sin(freq * ys + ph)
            wv = width * (1.0 - 0.7 * t ** 3)
            a = np.clip(1.0 - np.maximum(t - 0.85, 0) / 0.15, 0, 1)
            for dx in range(-3, 4):
                px = np.floor(xs).astype(int) + dx
                d = np.abs(px + 0.5 - xs) / np.maximum(wv, 0.3)
                inside = d < 1.0
                if not inside.any():
                    continue
                prof = np.sqrt(np.clip(1 - d ** 2, 0, 1))
                py = ys[inside]; pxx = np.clip(px[inside], x0, x0 + CW - 1)
                cvi = (prof * a)[inside]
                upd = cvi > cov[py, pxx]
                py, pxx, cvi = py[upd], pxx[upd], cvi[upd]
                cov[py, pxx] = cvi
                lum[py, pxx] = bright * (0.82 + 0.18 * t[inside][upd])
                sgn = np.sign(px[inside] + 0.5 - xs[inside])[upd]
                nx[py, pxx] = sgn * np.sqrt(np.clip(1 - prof[inside][upd] ** 2, 0, 1)) * 0.8
        u = (np.arange(CW) + 0.5) / CW
        v = (np.arange(H) + 0.5) / H
        # the card edge breaks up into single strands (the clumps thin out towards the sides: strand density, not a
        # blurred alpha ramp, which read as soft-edged ribbons); only the last few texels fade so nothing is cut at the
        # column border (user item 11: 'nicer strands')
        side = smooth(u / 0.06) * smooth((1 - u) / 0.06) if kind != "solid" else np.ones(CW)
        if kind == "solid":
            under = smooth((1.0 - v) / 0.08)[:, None] * np.ones((1, CW))        # opaque dark base, tip fades
            blk = cov[:, x0:x0 + CW]
            lum[:, x0:x0 + CW] = np.where(blk > 0.05, lum[:, x0:x0 + CW], 0.45)
            cov[:, x0:x0 + CW] = np.maximum(blk, under)
        else:
            rootlen = {"dense": 0.08, "medium": 0.12, "wispy": 0.3}[kind]     # strands emerge gradually at the root
            cov[:, x0:x0 + CW] *= side[None, :] * smooth(v / rootlen)[:, None]
    return lum, cov, nx


if __name__ == "__main__":
    args = sys.argv[1:]
    lum, cov, nx = build()
    alpha = np.clip(cov * 1.7, 0, 1)
    fill = lum.copy(); mask = cov > 0.02
    for _ in range(8):                        # push colour into empty texels (no dark halos when filtered)
        cand = np.maximum(np.roll(fill, -1, 1), np.roll(fill, 1, 1))
        m2 = np.roll(mask, -1, 1) | np.roll(mask, 1, 1)
        fill = np.where(mask, fill, np.where(m2, cand, fill)); mask = mask | m2
    fill = np.where(mask, fill, 0.7)
    nz = np.sqrt(np.clip(1 - nx ** 2, 0, 1))
    N = np.stack([nx * 0.5 + 0.5, np.full_like(nx, 0.5), nz * 0.5 + 0.5], -1)
    os.makedirs(TEX, exist_ok=True)
    Image.fromarray((N * 255).astype(np.uint8)).save(os.path.join(TEX, "hair_strands_normal.png"))
    for i in range(0, len(args), 4):
        name = args[i]; col = np.array([float(v) for v in args[i + 1:i + 4]], np.float32)
        rgb = np.clip(fill[..., None] * col[None, None, :] * 1.5, 0, 1)
        a = alpha.copy()
        # leather swatch for hair ties: top 24 rows of the last column (above every card's root, v > 0.988)
        x0 = (COLS - 1) * CW
        grain = 0.85 + 0.15 * rng.random((24, CW))
        rgb[:24, x0:x0 + CW] = np.array([0.24, 0.15, 0.09])[None, None, :] * grain[..., None]
        a[:24, x0:x0 + CW] = 1.0
        img = np.concatenate([rgb, a[..., None]], -1)
        Image.fromarray((img * 255).astype(np.uint8)).save(os.path.join(TEX, "hair_strands_%s_base.png" % name))
        print("HAIRTEX", name, "saved")
