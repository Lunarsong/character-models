"""Before / after contact sheet: rows of labelled image pairs (or triples), each image fitted into a fixed cell.

usage: python3 scripts/ba_sheet.py <out.png> "<title>" "<col 1>,<col 2>[,...]" "<row label>|<img 1>|<img 2>[|...]" ...
  image paths are relative to the characters folder; an image spec may carry a crop box in fractions:
  path@x0,y0,x1,y1 (e.g. renders/x.png@0.25,0,0.75,1 keeps the middle half).
"""
import os, sys
from PIL import Image, ImageDraw, ImageFont

CH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
out, title, cols = sys.argv[1], sys.argv[2], sys.argv[3].split(",")
rows = [r.split("|") for r in sys.argv[4:]]
CW, CHH = int(os.environ.get("CELL_W", 460)), int(os.environ.get("CELL_H", 460))
LW = int(os.environ.get("LABEL_W", 170))
try:
    font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 18)
    big = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 22)
except Exception:
    font = big = ImageFont.load_default()


def load(spec):
    path, crop = (spec.split("@") + [None])[:2]
    im = Image.open(os.path.join(CH, path)).convert("RGB")
    if crop:
        x0, y0, x1, y1 = [float(v) for v in crop.split(",")]
        im = im.crop((int(x0 * im.width), int(y0 * im.height), int(x1 * im.width), int(y1 * im.height)))
    s = min(CW / im.width, CHH / im.height)
    im = im.resize((max(1, int(im.width * s)), max(1, int(im.height * s))), Image.LANCZOS)
    cell = Image.new("RGB", (CW, CHH), (30, 30, 34))
    cell.paste(im, ((CW - im.width) // 2, (CHH - im.height) // 2))
    return cell


S = Image.new("RGB", (LW + CW * len(cols), 70 + len(rows) * (CHH + 8)), (22, 22, 26))
d = ImageDraw.Draw(S)
d.text((10, 8), title, fill=(240, 240, 240), font=big)
for c, name in enumerate(cols):
    d.text((LW + c * CW + 8, 42), name, fill=(200, 200, 120), font=font)
for r, row in enumerate(rows):
    y = 70 + r * (CHH + 8)
    lines, cur = [], ""
    for w in row[0].split():
        if len(cur) + len(w) > 16:
            lines.append(cur); cur = w
        else:
            cur = (cur + " " + w).strip()
    lines.append(cur)
    for i, ln in enumerate(lines):
        d.text((8, y + 8 + i * 22), ln, fill=(230, 230, 230), font=font)
    for c, spec in enumerate(row[1:]):
        S.paste(load(spec), (LW + c * CW, y))
os.makedirs(os.path.dirname(os.path.join(CH, out)), exist_ok=True)
S.save(os.path.join(CH, out))
print("SHEET", os.path.join(CH, out))
