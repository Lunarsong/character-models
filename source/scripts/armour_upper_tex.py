"""Knight armour TRIM SHEET: layout (shared with the Blender side, no heavy imports at module level) + generator.

One 2k texture set covers every steel / gold / leather armour piece (one material, one draw call per piece):
  knight_armour_base.png    sRGB base colour
  knight_armour_orm.png     R = ambient occlusion, G = roughness, B = metallic (glTF ORM)
  knight_armour_normal.png  tangent-space normal, OpenGL (+Y), from procedural height fields
Layout (UV space, V up, 2048 x 2048):
  STEEL    V 0.625-1.0   plate steel (brushed + hammered + fine scratches), 1024 px/m, tiles in U
  squares  V 0.375-0.625 LION (U 0-0.25, embossed lion rampant for the pauldron cops, 2048 px/m), BOSS rosette,
                         RIVET head, LEATHER (U 0.5-0.75, 1024 px/m), DARK interior (U 0.75-1)
  TRIMS    V 0-0.375     full-width strips that tile in U at 4096 px/m (bands below); 'etch_band' (iteration 2, was the
                         unused 'strap' strip) = engraved gold rinceau inlaid in steel: the decal ribbons inside every
                         plate border (armour_upper_geo.decal_ribbon, user item M6)
The mail shirt uses its own tiling set (knight_mail_*) and the plume its own hair-card atlas (knight_plume_*).

run (system python3 with numpy + Pillow): python3 scripts/armour_upper_tex.py   -> assets/textures/knight_*.png
"""
import os, math

W = H = 2048
TRIM_DENS = 4096.0          # px per metre along trim strips (U)
STEEL_DENS = 1024.0         # px per metre on plates
STEEL = (1.0, 0.625)        # (V top, V bottom)
BANDS_PX = [                # name, height in px (top to bottom of the trim region, starting at V = 0.375)
    ("fil_wide", 128), ("fil_narrow", 96), ("rivet", 64), ("gold_bead", 64), ("steel_bead", 40),
    ("gold_plain", 48), ("crest", 64), ("etch_band", 96), ("steel_plain", 64), ("dark", 32), ("pearls", 48),
    ("steel_ridge", 24),
]
BANDS = {}
_y = 0.375 * H
for _n, _h in BANDS_PX:
    BANDS[_n] = (_y / H, (_y - _h) / H)      # (V top, V bottom)
    _y -= _h
assert abs(_y) < 1e-6, _y
SQUARES = {                 # name: (U0, V0, U1, V1)
    "lion": (0.0, 0.375, 0.25, 0.625),
    "boss": (0.25, 0.5, 0.375, 0.625),
    "rivet_head": (0.375, 0.5, 0.5, 0.625),
    "knot": (0.25, 0.375, 0.375, 0.5),
    "etch": (0.375, 0.375, 0.5, 0.5),
    "leather": (0.5, 0.375, 0.75, 0.625),
    "dark_sq": (0.75, 0.375, 1.0, 0.625),
}
INSET = 3.0 / H             # keep strip UVs off the band borders (mip bleeding)


def band_px(name):
    return dict(BANDS_PX)[name]


def band_v(name, f):
    top, bot = BANDS[name]
    top -= INSET; bot += INSET
    return top - f * (top - bot)


def band_uv(name, f, u=0.0):
    return (u, band_v(name, f))


def steel_uv(U, V):
    """Arc-length metres (U, V arrays) -> UVs in the STEEL region (V clamped inside it)."""
    import numpy as np
    s = STEEL_DENS / W
    u = U * s
    v = 0.64 + V * s
    v = np.clip(v, STEEL[1] + INSET, STEEL[0] - INSET)
    return np.stack([u, v], axis=-1)


def square_uv(name, x, y, size=0.25):
    """Local metres (x, y) centred on the square -> UVs (size = metres covered by the square)."""
    u0, v0, u1, v1 = SQUARES[name]
    fx = min(max(0.5 + x / size, 0.004), 0.996)
    fy = min(max(0.5 + y / size, 0.004), 0.996)
    return (u0 + fx * (u1 - u0), v0 + fy * (v1 - v0))


TEX_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets", "textures")
FILES = {k: os.path.join(TEX_DIR, "knight_%s.png" % k) for k in
         ("armour_base", "armour_orm", "armour_normal", "mail_base", "mail_orm", "mail_normal",
          "plume_base", "plume_normal")}
# The mail and plume use the materials agent's sets directly when present (textures_char.json), else ours:
MANIFEST = os.path.join(TEX_DIR, "textures_char.json")
MAIL_TILE_M = 0.168          # metres per mail UV unit (mail_riveted tile); ours is generated at the same scale


def material_files():
    """{'mail': (base, orm, normal), 'plume': (base, orm|None, normal)} preferring the materials agent's sets."""
    import json
    out = {"mail": (FILES["mail_base"], FILES["mail_orm"], FILES["mail_normal"]),
           "plume": (FILES["plume_base"], None, FILES["plume_normal"])}
    if os.path.exists(MANIFEST):
        sets = json.load(open(MANIFEST))["sets"]
        def f(sn, k):
            p = os.path.join(TEX_DIR, sets[sn]["files"][k]) if sn in sets and k in sets[sn].get("files", {}) else None
            return p if p and os.path.exists(p) else None
        if f("mail_riveted", "base") and f("mail_riveted", "normal"):
            out["mail"] = (f("mail_riveted", "base"), f("mail_riveted", "orm"), f("mail_riveted", "normal"))
        if f("horsehair_blue", "base"):
            out["plume"] = (f("horsehair_blue", "base"), f("horsehair_blue", "orm"), f("horsehair_blue", "normal"))
    return out


if __name__ == "__main__":
    import armour_upper_texgen
    armour_upper_texgen.main()
