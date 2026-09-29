"""Before / after comparison of two deform_test.py runs (system python3 + Pillow).

usage: python3 scripts/compare_sheet.py <before_dir> <after_dir> <kind> [pose ...]
       python3 scripts/compare_sheet.py <before_dir> <after_dir> <kind> --focus  (one overview of the fixed areas)
  dirs are under renders/ (e.g. before after). Writes renders/compare/compare_<kind>_<pose>.png (one row per view:
  before | after) and renders/compare/compare_<kind>_metrics.md (per pose and region: before -> after), and prints
  the summary table.
"""
import sys, os, glob, json
from PIL import Image, ImageDraw, ImageFont

REN = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "renders")
bdir, adir, kind = sys.argv[1:4]
only = [a for a in sys.argv[4:] if not a.startswith("--")]
# the areas the 24 Sep feedback was about (user's views first), then the rest of the new stress set
FOCUS = ["viewer_arms_up_zarmpit_back34", "viewer_arms_up_zarmpit_front34", "viewer_arms_up_zneck_shoulders",
         "arms_up_180_zarmpit_back34", "arms_up_180_zneck_shoulders", "arms_45_zneck_shoulders",
         "arm_forward_up_zshoulder_back", "cross_body_reach_zshoulder_back", "arm_behind_back_zshoulder_front",
         "arms_down_zarmpit_front", "viewer_squat_zgroin_below_front", "viewer_squat_zgroin_front",
         "viewer_squat_zglutes_below_back", "deep_squat_zgroin_below_front", "deep_squat_zglutes_back",
         "deep_lunge_zhip_side", "lunge_zhip_front", "sitting_zgroin_front", "high_kick_zgroin_front",
         "high_kick_zglutes_back_side"]
out = os.path.join(REN, "compare"); os.makedirs(out, exist_ok=True)
try:
    font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 18)
    small = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 14)
except Exception:
    font = small = ImageFont.load_default()
TH = 420
mb = json.load(open(os.path.join(REN, bdir, "deform_%s_metrics.json" % kind)))
ma = json.load(open(os.path.join(REN, adir, "deform_%s_metrics.json" % kind)))
poses = [p for p in ma if (not only or p in only)]
for pose in poses:
    files = sorted(glob.glob(os.path.join(REN, adir, "deform_%s_%s_*.png" % (kind, pose))))
    views = [os.path.basename(f)[len("deform_%s_%s_" % (kind, pose)):-4] for f in files]
    views = [v for v in views if (v in ("front34", "back34") or v.startswith("z"))
             and os.path.exists(os.path.join(REN, bdir, "deform_%s_%s_%s.png" % (kind, pose, v)))]
    if not views:
        continue
    views.sort(key=lambda v: (not v.startswith("z"), v))
    rows = []
    for v in views:
        pair = []
        for d in (bdir, adir):
            im = Image.open(os.path.join(REN, d, "deform_%s_%s_%s.png" % (kind, pose, v))).convert("RGB")
            pair.append(im.resize((int(im.width * TH / im.height), TH), Image.LANCZOS))
        rows.append((v, pair))
    W = max(p[0].width + p[1].width for _, p in rows) + 12
    S = Image.new("RGB", (W, len(rows) * (TH + 28) + 30), (22, 22, 26))
    d = ImageDraw.Draw(S)
    d.text((8, 5), "%s  %s   left: %s   right: %s" % (kind, pose, bdir, adir), fill=(240, 240, 240), font=font)
    y = 30
    for v, (b, a) in rows:
        d.text((8, y + 4), v, fill=(200, 200, 120), font=small)
        S.paste(b, (0, y + 24)); S.paste(a, (b.width + 12, y + 24))
        y += TH + 28
    S.save(os.path.join(out, "compare_%s_%s.png" % (kind, pose)))

if "--focus" in sys.argv:
    T = 300
    cells = [f for f in FOCUS if all(os.path.exists(os.path.join(REN, d, "deform_%s_%s.png" % (kind, f))) for d in (bdir, adir))]
    cols = 4                                                    # 2 pairs per row
    rows = (len(cells) + 1) // 2
    S = Image.new("RGB", (cols * T + 30, rows * (T + 24) + 34), (22, 22, 26))
    d = ImageDraw.Draw(S)
    d.text((8, 6), "%s: before (left) | after (right) of each pair" % kind, fill=(240, 240, 240), font=font)
    for i, f in enumerate(cells):
        x0 = (i % 2) * (2 * T + 30); y0 = 34 + (i // 2) * (T + 24)
        d.text((x0 + 6, y0 + 3), f.replace("_z", " / "), fill=(200, 200, 120), font=small)
        for j, dd in enumerate((bdir, adir)):
            im = Image.open(os.path.join(REN, dd, "deform_%s_%s.png" % (kind, f))).convert("RGB")
            S.paste(im.resize((T, T), Image.LANCZOS), (x0 + j * T, y0 + 22))
    S.save(os.path.join(out, "compare_%s_focus.png" % kind))
    print("FOCUS", os.path.join(out, "compare_%s_focus.png" % kind), len(cells))
    sys.exit(0)

KEYS = ("fold_sum", "folds", "fold_max", "flips", "isect", "comp", "vol")
lines = ["| pose | region | " + " | ".join(KEYS) + " |", "|" + "---|" * (len(KEYS) + 2)]
for pose in poses:
    for r in ma[pose]:
        if r == "correctives" or r not in mb.get(pose, {}):
            continue
        b, a = mb[pose][r], ma[pose][r]
        if all(b.get(k) == a.get(k) for k in KEYS) and b["fold_max"] < 30:
            continue
        lines.append("| %s | %s | " % (pose, r) + " | ".join("%s -> %s" % (b.get(k), a.get(k)) for k in KEYS) + " |")
open(os.path.join(out, "compare_%s_metrics.md" % kind), "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
