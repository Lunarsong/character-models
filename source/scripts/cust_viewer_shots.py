"""Viewer (three.js) screenshots of the customisation layer, through the real GLBs (base + parts, variants, refit):
  renders/viewer_cust_<kind>_panel.png        Customise tab open, a seeded random character, body view
  renders/viewer_cust_<kind>_body_<seed>.png  6 random characters, full body (panels hidden)
  renders/viewer_cust_<kind>_<seed>_<expr>.png  the same 6 characters' faces x Neutral / Smile / Battle cry / Surprise
  renders/viewer_cust_<kind>_grid.png         contact sheet of the faces (rows = characters, columns = expressions)
  renders/viewer_cust_<kind>_poses.png        heavy + muscular random body in Arms up / Squat with the joint refit
  renders/viewer_cust_<kind>_beards.png       (kinds with beards) none / stubble / short / full beard, front + 3/4 +
                                              Battle cry, black and auburn hair
usage: python3 scripts/cust_viewer_shots.py male [female ...]   (after mkviewer_char.py)
"""
import json, os, subprocess, sys, tempfile
from PIL import Image, ImageDraw, ImageFont

CH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REN = os.path.join(CH, "renders")
SEEDS = [11, 12, 13, 14, 15, 16]
EXPRS = ["Neutral", "Smile", "Battle cry", "Surprised"]          # preset names of scripts/expressions.json
BEARDS = ["none", "rts_stubble", "rts_beard_short", "rts_beard_full"]
HIDE = "document.querySelectorAll('.panel').forEach(p => p.style.display = 'none'); status.style.display = 'none';"
SHOW = "document.querySelectorAll('.panel').forEach(p => p.style.display = ''); status.style.display = '';"
FACE = ("goView('Face'); { const t = VIEWS.Face.t; camera.position.set(0.16, t[1] + 0.035, 0.72); "
        "controls.target.set(0, t[1] + 0.005, 0); controls.update(); }")


def expr(e):
    # the viewer's own preset player (keeps the cust_* sliders, applies the weights at once, '_loop' hints)
    return "pickExpr(%s, true);" % json.dumps(e)


def run(kind):
    shots = [{"out": "renders/viewer_cust_%s_panel.png" % kind,
              "js": "showTab('cust'); randomise(3); goView('Body');", "wait": 1.5}]
    tiles = []
    for sd in SEEDS:
        shots.append({"out": "renders/viewer_cust_%s_body_%d.png" % (kind, sd),
                      "js": HIDE + "randomise(%d); clearMorphs(); goView('Body');" % sd, "wait": 1.2})
        for e in EXPRS:
            out = "renders/viewer_cust_%s_%d_%s.png" % (kind, sd, e.replace(" ", "").lower())
            shots.append({"out": out, "js": FACE + expr(e), "wait": 0.8, "clip": [410, 70, 580, 700]})
            tiles.append((sd, e, out))
    shots.append({"out": "renders/viewer_cust_%s_poses_a.png" % kind,
                  "js": "applyRecipe({sliders: {body_muscle: 1, body_weight: 0.8, body_shoulders: 0.6}, parts: {}, variants: {}});"
                        "currentPose = 'Arms up'; resetPose(); POSES['Arms up'](); clearMorphs(); goView('Body');", "wait": 1.2})
    shots.append({"out": "renders/viewer_cust_%s_poses_b.png" % kind,
                  "js": "currentPose = 'Squat'; resetPose(); POSES['Squat']();", "wait": 1.2})
    shots.append({"out": "renders/viewer_cust_%s_poses_c.png" % kind,
                  "js": "applyRecipe({sliders: {body_muscle: -1, body_weight: -1}, parts: {}, variants: {}});"
                        "currentPose = 'Arms up'; resetPose(); POSES['Arms up']();", "wait": 1.2})
    beard_tiles = []
    if kind.startswith("male"):
        for col in ("black", "auburn"):
            for b in BEARDS:
                for view, cam in (("front", FACE), ("34", FACE + " camera.position.set(0.5, VIEWS.Face.t[1] + 0.03, 0.45);"
                                                            " controls.update();"), ("cry", FACE)):
                    out = "renders/viewer_cust_%s_beard_%s_%s_%s.png" % (kind, b, col.replace(" ", ""), view)
                    js = HIDE + "applyRecipe({sliders: {}, parts: {beard: %s}, variants: {hair: %s}}); clearMorphs(); %s" % (
                        json.dumps(b), json.dumps(col), cam)
                    if view == "cry":
                        js += expr("Battle cry")
                    shots.append({"out": out, "js": js, "wait": 0.9, "clip": [410, 70, 580, 700]})
                    beard_tiles.append((col, b, view, out))
    spec = {"w": 1400, "h": 900, "shots": shots}
    fd, path = tempfile.mkstemp(suffix=".json"); os.write(fd, json.dumps(spec).encode()); os.close(fd)
    subprocess.run([sys.executable, os.path.join(CH, "scripts", "cdp_shot.py"),
                    os.path.join(CH, "viewer", "base_%s.html" % kind), path], check=True)
    os.remove(path)
    font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 18)
    tw, th = 290, 350
    S = Image.new("RGB", (tw * (len(EXPRS) + 1), 28 + th * len(SEEDS)), (24, 24, 28))
    d = ImageDraw.Draw(S)
    d.text((8, 5), "seed / body", fill=(230, 230, 230), font=font)
    for c, e in enumerate(EXPRS):
        d.text(((c + 1) * tw + 8, 5), e, fill=(230, 230, 230), font=font)
    for r, sd in enumerate(SEEDS):
        b = Image.open(os.path.join(REN, "viewer_cust_%s_body_%d.png" % (kind, sd))).convert("RGB").crop((420, 0, 980, 900))
        b = b.resize((int(b.width * th / b.height), th))
        S.paste(b, ((tw - b.width) // 2, 28 + r * th))
        d.text((6, 28 + r * th + 4), "seed %d" % sd, fill=(230, 230, 230), font=font)
    for sd, e, out in tiles:
        r, c = SEEDS.index(sd), EXPRS.index(e) + 1
        im = Image.open(os.path.join(CH, out)).convert("RGB").resize((tw, th))
        S.paste(im, (c * tw, 28 + r * th))
    S.save(os.path.join(REN, "viewer_cust_%s_grid.png" % kind))
    P = [Image.open(os.path.join(REN, "viewer_cust_%s_poses_%s.png" % (kind, x))).convert("RGB").crop((350, 0, 1050, 900)) for x in "abc"]
    Q = Image.new("RGB", (700 * 3, 900)); [Q.paste(p, (i * 700, 0)) for i, p in enumerate(P)]
    Q.save(os.path.join(REN, "viewer_cust_%s_poses.png" % kind))
    if beard_tiles:
        bw, bh = 232, 280
        cols = [(b, v) for b in BEARDS for v in ("front", "34", "cry")]
        B = Image.new("RGB", (bw * len(cols), 28 + bh * 2), (24, 24, 28))
        d = ImageDraw.Draw(B)
        for c, (b, v) in enumerate(cols):
            d.text((c * bw + 6, 5), "%s %s" % (b.replace("rts_", ""), {"front": "", "34": "3/4", "cry": "battle cry"}[v]),
                   fill=(230, 230, 230), font=font)
        for col, b, v, out in beard_tiles:
            r = 0 if col == "black" else 1
            B.paste(Image.open(os.path.join(CH, out)).convert("RGB").resize((bw, bh)), (cols.index((b, v)) * bw, 28 + r * bh))
            os.remove(os.path.join(CH, out))
        B.save(os.path.join(REN, "viewer_cust_%s_beards.png" % kind))
    print("SHOTS", kind, "grid + poses + panel written")


for k in sys.argv[1:] or ["male", "female"]:
    run(k)
