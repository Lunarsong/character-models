"""Contact sheet for scripts/cust_qa.py grid renders: rows = random characters (seeded), columns = expressions.
usage: python3 scripts/cust_grid_sheet.py <kind>  -> renders/custgrid_<kind>_sheet.png
"""
import sys, os, json
from PIL import Image, ImageDraw, ImageFont

REN = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "renders")
kind = sys.argv[1]
EX = ["neutral", "smile", "angry", "surprise", "battlecry"]
rec = json.load(open(os.path.join(REN, "custgrid_%s_recipes.json" % kind)))
seeds = sorted(rec, key=int)
t = Image.open(os.path.join(REN, "custgrid_%s_%s_%s.png" % (kind, seeds[0], EX[0])))
W, H = t.size
LW, TH = 250, 26
S = Image.new("RGB", (LW + W * len(EX), TH + H * len(seeds)), (24, 24, 28))
d = ImageDraw.Draw(S)
font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 15)
small = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 12)
for c, e in enumerate(EX):
    d.text((LW + c * W + 8, 5), e, fill=(235, 235, 235), font=font)
for r, sd in enumerate(seeds):
    R = rec[sd]
    for c, e in enumerate(EX):
        S.paste(Image.open(os.path.join(REN, "custgrid_%s_%s_%s.png" % (kind, sd, e))).convert("RGB"), (LW + c * W, TH + r * H))
    top = sorted(R["sliders"].items(), key=lambda kv: -abs(kv[1]))[:9]
    lines = ["character %s" % sd,
             "skin %s, eyes %s" % (R["variants"]["skin"], R["variants"]["eyes"]),
             "hair %s (%s)" % (R["parts"]["hair"], R["variants"]["hair"]),
             "brows %s, beard %s" % (R["parts"]["eyebrows"], R["parts"]["beard"]), "strongest sliders:"] + \
            ["  %s %+.2f" % kv for kv in top]
    for i, ln in enumerate(lines):
        d.text((8, TH + r * H + 8 + i * 16), ln, fill=(230, 230, 230) if i == 0 else (185, 190, 200), font=font if i == 0 else small)
out = os.path.join(REN, "custgrid_%s_sheet.png" % kind)
S.save(out)
print("SHEET", out, S.size)
