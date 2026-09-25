#!/usr/bin/env python3
"""Contact sheet of PNG renders (labels = file name minus the prefix).
usage: python3 scripts/civ_sheet.py OUT.png [--cols N] [--h H] IMG.png ... | python3 scripts/civ_sheet.py OUT.png --glob 'renders/civ/prev_*_rest_*.png'"""
import sys, glob, os
from PIL import Image, ImageDraw, ImageFont


def sheet(out, files, cols=4, h=520, label_prefix=None):
    ims = []
    for f in files:
        im = Image.open(f).convert("RGB")
        w = int(im.width * h / im.height)
        ims.append((os.path.splitext(os.path.basename(f))[0], im.resize((w, h), Image.LANCZOS)))
    if not ims:
        return
    cols = min(cols, len(ims))
    cw = max(i.width for _, i in ims)
    rows = (len(ims) + cols - 1) // cols
    S = Image.new("RGB", (cw * cols, (h + 22) * rows), (22, 22, 26))
    d = ImageDraw.Draw(S)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 15)
    except Exception:
        font = None
    for k, (nm, im) in enumerate(ims):
        x, y = (k % cols) * cw, (k // cols) * (h + 22)
        S.paste(im, (x + (cw - im.width) // 2, y + 22))
        if label_prefix and nm.startswith(label_prefix):
            nm = nm[len(label_prefix):]
        d.text((x + 6, y + 3), nm, fill=(225, 225, 225), font=font)
    S.save(out)
    print("sheet", out, len(ims))


if __name__ == "__main__":
    a = sys.argv[1:]
    out = a.pop(0)
    cols, h, files, pre = 4, 520, [], None
    while a:
        x = a.pop(0)
        if x == "--cols":
            cols = int(a.pop(0))
        elif x == "--h":
            h = int(a.pop(0))
        elif x == "--glob":
            files += sorted(glob.glob(a.pop(0)))
        elif x == "--strip":
            pre = a.pop(0)
        else:
            files.append(x)
    sheet(out, files, cols, h, pre)
