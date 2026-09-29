#!/usr/bin/env python3
"""Knight viewer screenshots + reference comparison (build_knight.py stages 'shots' and 'compare').

shots:   viewer/knight_<kind>_lookdev.html through viewer/shot.sh (headless Chrome, one page load, BATCH specs):
         turnaround (studio, idle pose), face close-ups in the bare-head look (expression presets + frames of the
         talk_emote clip), RTS camera in the village, every clip at key frames -> renders/knight_<kind>_view_*.png
         and sheets renders/knight_<kind>_view_{turn,face,clips,rts}_sheet.png
compare: refs/knight_sheet.png crops next to the Blender look-dev renders (dark studio like the sheet):
         renders/knight_<kind>_vs_ref.png (turnaround) and renders/knight_<kind>_vs_ref_details.png (helm, shoulder,
         tabard, shield, sword)
usage: python3 scripts/knight_shots.py [shots|compare] [male female]
"""
import json, os, subprocess, sys
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
CH = os.path.dirname(HERE)
REN = os.path.join(CH, "renders")
VIEWER = os.path.join(CH, "viewer")
SCRATCH = os.environ.get("TMPDIR", "/tmp")


def log(*a):
    print("KNT", *a, flush=True)


def specs(kind):
    P = lambda n: os.path.join(REN, "knight_%s_view_%s.png" % (kind, n))
    S = []
    base = dict(ui=False, blink=False, scene="studio", look="helm", morphs="", yaw=0)
    # turnaround in the idle pose
    for nm, yaw in (("turn_front", 0), ("turn_34", 35), ("turn_right", 90), ("turn_back", 180), ("turn_left", -90)):
        S.append(dict(base, out=P(nm), view="Full body", clip="idle", time=0.0, yaw=yaw))
    # face (bare head): expressions + talk clip frames
    for ex in ("Neutral", "Smile", "Toothy smile", "Angry", "Surprise", "Sad", "Wink", "Battle cry"):
        S.append(dict(base, out=P("face_" + ex.lower().replace(" ", "_")), look="bare", view="Face", clip=None,
                      morphs=("" if ex == "Neutral" else ex)))
    S.append(dict(base, out=P("face_34_smile"), look="bare", view="Face 3/4", clip=None, morphs="Smile"))
    for t in (0.5, 0.97, 1.97, 2.5, 3.9, 5.37, 6.05, 7.1):
        S.append(dict(base, out=P("face_talk_%04.2f" % t), look="bare", view="Face", clip="talk_emote", time=t))
    # RTS camera in the village
    for nm, view, clip, t in (("rts", "RTS", "idle", 0.0), ("rts_zoom", "RTS zoom", "idle", 0.0),
                              ("rts_zoom_walk", "RTS zoom", "walk", 0.3), ("rts_zoom_attack", "RTS zoom", "attack_sword", 0.63)):
        S.append(dict(base, out=P(nm), view=view, clip=clip, time=t, scene="town"))
    # clips at key frames (3/4)
    for clip, times in (("idle", (3.0,)), ("walk", (0.0, 0.27, 0.55)), ("run", (0.0, 0.17, 0.35)),
                        ("attack_sword", (0.37, 0.57, 0.63, 0.8)), ("block_shield", (0.3, 0.53)), ("talk_emote", (3.9,))):
        for t in times:
            S.append(dict(base, out=P("clip_%s_%04.2f" % (clip, t)), view="Full body", clip=clip, time=t, yaw=35))
    return S


def sheet(files, out, cols, th, labels=None):
    ims = [Image.open(f).convert("RGB") for f in files if os.path.exists(f)]
    if not ims:
        return
    ims = [im.resize((int(im.width * th / im.height), th), Image.LANCZOS) for im in ims]
    tw = max(i.width for i in ims)
    rows = (len(ims) + cols - 1) // cols
    S = Image.new("RGB", (cols * tw, rows * (th + 20)), (22, 22, 25))
    d = ImageDraw.Draw(S)
    for k, (im, f) in enumerate(zip(ims, files)):
        x, y = (k % cols) * tw, (k // cols) * (th + 20)
        S.paste(im, (x, y + 20))
        d.text((x + 6, y + 4), (labels[k] if labels else os.path.basename(f)[:-4].split("_view_")[-1]), fill=(220, 220, 220))
    S.save(out)
    log("sheet", out, S.size)


def main(kind):
    page = os.path.join(VIEWER, "knight_%s_lookdev.html" % kind)
    sp = specs(kind)
    spec_file = os.path.join(SCRATCH, "knight_%s_shots.json" % kind)
    json.dump(sp, open(spec_file, "w"))
    env = dict(os.environ, BATCH=spec_file)
    r = subprocess.run([os.path.join(VIEWER, "shot.sh"), page, "-", "Full body", "", "", "", "900", "1100"], env=env,
                       capture_output=True, text=True, cwd=CH)
    print(r.stdout[-3000:], r.stderr[-3000:])
    if r.returncode != 0:
        raise SystemExit("shot.sh failed for %s" % kind)
    V = lambda n: os.path.join(REN, "knight_%s_view_%s.png" % (kind, n))
    sheet([V(n) for n in ("turn_front", "turn_34", "turn_right", "turn_back", "turn_left")],
          os.path.join(REN, "knight_%s_view_turn_sheet.png" % kind), 5, 560)
    sheet([s["out"] for s in sp if "_view_face" in s["out"]], os.path.join(REN, "knight_%s_view_face_sheet.png" % kind), 6, 330)
    sheet([s["out"] for s in sp if "_view_clip" in s["out"]], os.path.join(REN, "knight_%s_view_clips_sheet.png" % kind), 7, 420)
    sheet([s["out"] for s in sp if "_view_rts" in s["out"]], os.path.join(REN, "knight_%s_view_rts_sheet.png" % kind), 4, 420)


REF = os.path.join(CH, "refs", "knight_sheet.png")
PAIRS = [((30, 10, 270, 470), "front", "front"), ((300, 10, 520, 470), "side (right)", "right"),
         ((520, 10, 760, 470), "back", "back"), ((1080, 10, 1380, 470), "3/4", "34")]
DETAILS = [((25, 515, 243, 770), "helmet", "helm"), ((250, 515, 422, 770), "shoulder", "shoulder"),
           ((428, 515, 598, 770), "tabard", "torso"), ((604, 515, 750, 770), "shield", "shield"),
           ((756, 515, 913, 770), "sword", "sword_hand")]


def _crop_char(im, bg_tol=18):
    """crop a render to the character: columns / rows that differ from the corner background"""
    import numpy as np
    a = np.asarray(im.convert("RGB")).astype(int)
    bg = a[:8, :8].reshape(-1, 3).mean(0)
    diff = np.abs(a - bg).sum(2) > bg_tol * 3
    cols = np.nonzero(diff[: int(a.shape[0] * 0.8)].any(0))[0]
    rows = np.nonzero(diff.any(1))[0]
    if len(cols) < 2 or len(rows) < 2:
        return im
    pad = 12
    return im.crop((max(0, cols[0] - pad), max(0, rows[0] - pad), min(a.shape[1], cols[-1] + pad), a.shape[0]))


def compare(kinds):
    ref = Image.open(REF).convert("RGB")
    for kind in kinds:
        for pairs, name, H in ((PAIRS, "vs_ref", 760), (DETAILS, "vs_ref_details", 420)):
            tiles = []
            for box, label, ren in pairs:
                p = os.path.join(REN, "knight_%s_look_%s.png" % (kind, ren))
                if not os.path.exists(p):
                    continue
                a = ref.crop(box); a = a.resize((int(a.width * H / a.height), H), Image.LANCZOS)
                b = Image.open(p).convert("RGB")
                b = b.resize((int(b.width * H / b.height), H), Image.LANCZOS)
                t = Image.new("RGB", (a.width + b.width + 10, H + 26), (20, 20, 22))
                t.paste(a, (0, 26)); t.paste(b, (a.width + 10, 26))
                d = ImageDraw.Draw(t)
                d.text((6, 6), "reference: " + label, fill=(230, 230, 230)); d.text((a.width + 16, 6), "knight_%s" % kind, fill=(230, 230, 230))
                tiles.append(t)
            if not tiles:
                continue
            W = sum(t.width for t in tiles) + 8 * (len(tiles) - 1)
            S = Image.new("RGB", (W, H + 26), (12, 12, 14))
            x = 0
            for t in tiles:
                S.paste(t, (x, 0)); x += t.width + 8
            out = os.path.join(REN, "knight_%s_%s.png" % (kind, name))
            S.save(out)
            log("compare", out, S.size)


if __name__ == "__main__":
    a = sys.argv[1:]
    kinds = [k for k in a if k in ("male", "female")] or ["male", "female"]
    if not a or "shots" in a:
        for k in kinds:
            main(k)
    if not a or "compare" in a:
        compare(kinds)
