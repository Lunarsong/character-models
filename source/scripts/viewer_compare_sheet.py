"""Side-by-side sheet of the viewer deformation shots (scripts/viewer_deform_shots.sh):
python3 scripts/viewer_compare_sheet.py <kind> [tags]  -> renders/viewer/viewer_<kind>_compare.png
tags default: before,weightsonly,after (baseline GLB | new weights, correctives off | new weights + correctives)."""
import sys, os
from PIL import Image, ImageDraw, ImageFont
D = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "renders", "viewer")
k = sys.argv[1]
tags = (sys.argv[2] if len(sys.argv) > 2 else "before,weightsonly,after").split(",")
views = ["armsup_back", "armsup_front", "armsup_neck", "squat_below", "squat_back"]
try:
    font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 18)
except Exception:
    font = ImageFont.load_default()
T = 360
S = Image.new("RGB", (len(tags) * T, len(views) * (T + 24) + 30), (22, 22, 26)); d = ImageDraw.Draw(S)
d.text((6, 6), "%s viewer (three.js): %s" % (k, " | ".join(tags)), fill=(240, 240, 240), font=font)
for r, v in enumerate(views):
    for c, t in enumerate(tags):
        S.paste(Image.open(os.path.join(D, "viewer_%s_%s_%s.png" % (k, v, t))).convert("RGB").resize((T, T)), (c * T, 30 + r * (T + 24) + 22))
    d.text((6, 30 + r * (T + 24) + 2), v, fill=(200, 200, 120), font=font)
S.save(os.path.join(D, "viewer_%s_compare.png" % k))
print("SHEET", os.path.join(D, "viewer_%s_compare.png" % k))
