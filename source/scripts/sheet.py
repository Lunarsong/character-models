"""Contact sheets from renders (system python3 + Pillow).
usage: [FILTER=_z] [EXCLUDE=nocorr] [OUT=name.png] python3 scripts/sheet.py <prefix> [cols] [thumb_height] [order]
       -> renders/<prefix>_sheet.png
Collects renders/<prefix>_*.png (sorted, excluding sheets) and labels each tile with its file stem.
"""
import sys, os, glob
from PIL import Image, ImageDraw, ImageFont

REN = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "renders")
prefix = sys.argv[1]
cols = int(sys.argv[2]) if len(sys.argv) > 2 else 6
th = int(sys.argv[3]) if len(sys.argv) > 3 else 380
order = sys.argv[4].split(",") if len(sys.argv) > 4 else None
FILTER = os.environ.get("FILTER", "")          # keep only files whose name contains this (e.g. _z for close-ups)
EXCLUDE = os.environ.get("EXCLUDE", "")        # drop files whose name contains this
OUTNAME = os.environ.get("OUT", prefix + "_sheet.png")
files = [f for f in sorted(glob.glob(os.path.join(REN, prefix + "_*.png"))) if not f.endswith("_sheet.png")
         and FILTER in os.path.basename(f) and not (EXCLUDE and EXCLUDE in os.path.basename(f))]
if order:
    key = lambda f: next((i for i, o in enumerate(order) if o in os.path.basename(f)), 999)
    files.sort(key=key)
ims = []
for f in files:
    im = Image.open(f).convert("RGB")
    w = int(im.width * th / im.height)
    ims.append((os.path.basename(f)[len(os.path.basename(prefix)) + 1:-4], im.resize((w, th), Image.LANCZOS)))
tw = max(i.width for _, i in ims)
rows = (len(ims) + cols - 1) // cols
S = Image.new("RGB", (cols * tw, rows * (th + 22)), (24, 24, 28))
d = ImageDraw.Draw(S)
try:
    font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 16)
except Exception:
    font = ImageFont.load_default()
for k, (name, im) in enumerate(ims):
    x, y = (k % cols) * tw, (k // cols) * (th + 22)
    S.paste(im, (x + (tw - im.width) // 2, y + 22))
    d.text((x + 6, y + 3), name, fill=(230, 230, 230), font=font)
out = os.path.join(REN, OUTNAME)
S.save(out)
print("SHEET", out, S.size, len(ims))
