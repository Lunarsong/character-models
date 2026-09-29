"""Upper armour vs the reference sheet at matching angles (system python3 + Pillow).

  python3 scripts/armour_upper_compare.py [male|female] [tag]
Pairs each crop of refs/knight_sheet.png (helmet close-up, shoulder detail, helmet front / right / back / top views, the
upper body of the front / side / back views) with the matching look-dev render renders/armour_upper_<tag>_<kind>_<view>.png
(armour_upper.py render <kind> <tag> [m3]) -> renders/armour_upper_vs_ref_<kind>.png
"""
import os, sys
from PIL import Image, ImageDraw, ImageFont

CH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REN = os.path.join(CH, "renders")
kind = sys.argv[1] if len(sys.argv) > 1 else "male"
tag = sys.argv[2] if len(sys.argv) > 2 else "final"
REF = Image.open(os.path.join(CH, "refs", "knight_sheet.png")).convert("RGB")
# (label, reference crop box in the 1536 x 1024 sheet, our view, crop of our render as fractions (l, t, r, b))
PAIRS = [
    ("helmet close-up (3/4)", (24, 515, 243, 770), "helm", (0.0, 0.0, 1.0, 1.0)),
    ("helmet front", (1254, 515, 1386, 640), "helm_front", (0.0, 0.0, 1.0, 1.0)),
    ("helmet right", (1388, 515, 1520, 640), "helm_side", (0.0, 0.0, 1.0, 1.0)),
    ("helmet back", (1254, 645, 1386, 770), "helm_back", (0.0, 0.0, 1.0, 1.0)),
    ("helmet top", (1388, 645, 1520, 770), "helm_top", (0.0, 0.0, 1.0, 1.0)),
    ("shoulder detail", (251, 515, 421, 770), "shoulder", (0.0, 0.0, 1.0, 1.0)),
    ("front: upper body", (40, 10, 250, 250), "front", (0.22, 0.02, 0.78, 0.40)),
    ("right side: upper body", (300, 10, 500, 250), "right", (0.25, 0.02, 0.75, 0.40)),
    ("back: upper body", (540, 10, 745, 250), "back", (0.22, 0.02, 0.78, 0.40)),
]
H = 420
try:
    font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 17)
except Exception:
    font = ImageFont.load_default()
tiles = []
for label, box, view, fr in PAIRS:
    p = os.path.join(REN, "armour_upper_%s_%s_%s.png" % (tag, kind, view))
    if not os.path.exists(p):
        print("missing", p)
        continue
    r = REF.crop(box)
    o = Image.open(p).convert("RGB")
    W0, H0 = o.size
    o = o.crop((int(fr[0] * W0), int(fr[1] * H0), int(fr[2] * W0), int(fr[3] * H0)))
    r = r.resize((int(r.width * H / r.height), H), Image.LANCZOS)
    o = o.resize((int(o.width * H / o.height), H), Image.LANCZOS)
    t = Image.new("RGB", (r.width + o.width + 6, H + 26), (22, 22, 26))
    t.paste(r, (0, 26)); t.paste(o, (r.width + 6, 26))
    ImageDraw.Draw(t).text((6, 4), "%s   reference | ours (%s)" % (label, kind), fill=(235, 235, 235), font=font)
    tiles.append(t)
cols = 3
rows = (len(tiles) + cols - 1) // cols
tw = max(t.width for t in tiles)
S = Image.new("RGB", (cols * tw, rows * (H + 30)), (22, 22, 26))
for i, t in enumerate(tiles):
    S.paste(t, ((i % cols) * tw, (i // cols) * (H + 30)))
out = os.path.join(REN, "armour_upper_vs_ref_%s.png" % kind)
S.save(out)
print("COMPARE", out, S.size)
