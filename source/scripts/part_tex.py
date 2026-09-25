"""Textures for the swappable parts (system python3 + Pillow), written to assets/textures/parts/:
  hair_<style>.png   CC0 MakeHuman hair diffuse turned into a NEUTRAL strand texture (grey + alpha, 1024^2): luminance
                     normalised so the 70th percentile of the opaque strands sits at 0.72 (~ the procedural atlas),
                     so one baseColorFactor (= hair colour) tints every style the same way
  eye_<colour>.jpg   the 9 CC0 MakeHuman eye colours as JPEG (the eyeball material is opaque; ~7x smaller than PNG)
The procedural grooms use hair_strands_neutral_base.png from `python3 scripts/hair_tex.py neutral 0.6667 0.6667 0.6667`.
usage: python3 scripts/part_tex.py
"""
import os, sys
import numpy as np
from PIL import Image

CH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UD = os.path.join(CH, "blender_profile", "extensions", ".user", "user_default", "mpfb", "data")
OUT = os.path.join(CH, "assets", "textures", "parts")
HAIR = ["ponytail01", "long01", "short02", "bob02", "afro01", "short04", "braid01"]
EYES = ["brownlight", "brown", "green", "bluegreen", "blue", "lightblue", "deepblue", "grey", "ice"]
os.makedirs(OUT, exist_ok=True)

for st in HAIR:
    d = os.path.join(UD, "hair", st)
    src = [f for f in os.listdir(d) if f.endswith("_diffuse.png")][0]
    im = np.asarray(Image.open(os.path.join(d, src)).convert("RGBA")).astype(np.float32) / 255
    g = im[..., 0] * 0.2126 + im[..., 1] * 0.7152 + im[..., 2] * 0.0722
    a = im[..., 3]
    op = a > 0.5
    scale = 0.72 / max(np.percentile(g[op], 70), 1e-3)
    g = np.clip(g * scale, 0, 1)
    L = Image.fromarray((g * 255 + 0.5).astype(np.uint8)).resize((1024, 1024), Image.LANCZOS)
    A = Image.fromarray((a * 255 + 0.5).astype(np.uint8)).resize((1024, 1024), Image.LANCZOS)
    out = os.path.join(OUT, "hair_%s.png" % st)
    Image.merge("LA", (L, A)).save(out, optimize=True)
    print("PARTTEX", out, "%.2f MB" % (os.path.getsize(out) / 1e6), "scale %.2f" % scale)

for c in EYES:
    im = Image.open(os.path.join(UD, "eyes", "materials", c + "_eye.png")).convert("RGB")
    out = os.path.join(OUT, "eye_%s.jpg" % c)
    im.save(out, quality=90, optimize=True)
    print("PARTTEX", out, im.size, "%.2f MB" % (os.path.getsize(out) / 1e6))
