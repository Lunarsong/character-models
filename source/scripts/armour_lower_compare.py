#!/usr/bin/env python3
"""Side-by-side comparison of the reference sheet (refs/knight_sheet.png) with the knight renders.

usage: python3 scripts/armour_lower_compare.py [prefix]     (default prefix: combo_male = upper + lower armour renders
       from armour_lower_qa.py -- male combo; use lower_male for the lower kit alone)
writes renders/<prefix>_vs_ref.png: rows = front / side / back / 3-4, left = reference crop, right = render.
"""
import os, sys
from PIL import Image, ImageDraw

CH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REF = os.path.join(CH, "refs", "knight_sheet.png")
REN = os.path.join(CH, "renders")
prefix = sys.argv[1] if len(sys.argv) > 1 else "combo_male"
# reference view boxes on the 1536 x 1024 sheet (x0, y0, x1, y1) and the matching render names
PAIRS = [((30, 10, 270, 470), "front"), ((300, 10, 520, 470), "side"), ((520, 10, 760, 470), "back"),
         ((1080, 10, 1380, 470), "34")]
H = 760
ref = Image.open(REF).convert("RGB")
tiles = []
for box, name in PAIRS:
    a = ref.crop(box); a = a.resize((int(a.width * H / a.height), H), Image.LANCZOS)
    p = os.path.join(REN, "%s_%s.png" % (prefix, name))
    if not os.path.exists(p):
        continue
    b = Image.open(p).convert("RGB")
    # crop the render to the character (non-background columns) for a fair side by side
    b = b.resize((int(b.width * H / b.height), H), Image.LANCZOS)
    t = Image.new("RGB", (a.width + b.width + 12, H + 34), (24, 24, 26))
    t.paste(a, (0, 34)); t.paste(b, (a.width + 12, 34))
    d = ImageDraw.Draw(t)
    d.text((8, 8), "reference: " + name, fill=(220, 220, 220)); d.text((a.width + 20, 8), "render: " + name, fill=(220, 220, 220))
    tiles.append(t)
W = sum(t.width for t in tiles) + 10 * (len(tiles) - 1)
sheet = Image.new("RGB", (W, H + 34), (16, 16, 18))
x = 0
for t in tiles:
    sheet.paste(t, (x, 0)); x += t.width + 10
out = os.path.join(REN, "%s_vs_ref.png" % prefix)
sheet.save(out)
print("compare", out, sheet.size)
