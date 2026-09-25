"""Web-viewer (three.js look-dev page) close-up specs for the face / hair items (user round-2 items 7-11), the same cameras
before / after. Prints a viewer/shot.sh BATCH spec (JSON) to stdout.
usage: python3 scripts/face_hair_viewer_shots.py <kind> <tag> <outdir> [hair,eyes,brows,mouth] > spec.json
       SCENE=studio BATCH=spec.json viewer/shot.sh viewer/base_<kind>_lookdev.html -
writes <outdir>/<tag>_<kind>_v_<group>_<shot>.png. Landmarks (glTF metres: eye_l, stomion, head top) of the base presets."""
import sys, json, os
kind, tag, outdir = sys.argv[1:4]
groups = sys.argv[4].split(",") if len(sys.argv) > 4 else ["hair", "eyes", "mouth", "brows"]
LMK = {"male": dict(ex=0.0306, ey=1.7401, ez=0.1357, my=1.6649, mz=0.1642, top=1.8501),
       "female": dict(ex=0.0301, ey=1.6074, ez=0.1169, my=1.5431, mz=0.1467, top=1.7192)}[kind]
ex, ey, ez, top = LMK["ex"], LMK["ey"], LMK["ez"], LMK["top"]
def V(p, t, fov=30):
    return {"p": [round(v, 4) for v in p], "t": [round(v, 4) for v in t], "fov": fov}
L = []
def add(g, name, view, morphs="", **kw):
    d = dict(out=os.path.join(outdir, "%s_%s_v_%s_%s.png" % (tag, kind, g, name)), view=view, morphs=morphs, clip=None, ui=False)
    d.update(kw); L.append(d)
if "hair" in groups:
    add("hair", "34", V([-0.55, ey + 0.12, 0.62], [0, top - 0.08, 0.0], 30))
    add("hair", "back", V([0.0, ey + 0.15, -0.8], [0, top - 0.1, 0.0], 30))
    add("hair", "hairline34", V([-0.22, top + 0.02, 0.30], [0.0, top - 0.05, 0.07], 30))
    add("hair", "crown", V([0.12, top + 0.28, -0.22], [0.0, top - 0.03, 0.0], 30))
    add("hair", "temple", V([0.33, ey + 0.03, 0.14], [0.065, ey + 0.02, 0.03], 30))
    add("hair", "rts", V([-0.9, ey + 1.2, 0.9], [0, top - 0.1, 0.0], 25))
if "eyes" in groups:
    ce = [ex, ey, ez]
    for st, m in (("open", ""), ("blink", "eyeBlinkLeft=1"), ("blink50", "eyeBlinkLeft=0.5"), ("wink", "Wink"),
                  ("blinksquint", "eyeBlinkLeft=1,eyeSquintLeft=1")):
        add("eyes", "front_" + st, V([ex, ey + 0.004, ez + 0.2], ce, 22), m)
        add("eyes", "34_" + st, V([ex + 0.13, ey + 0.01, ez + 0.15], ce, 22), m)
        add("eyes", "profile_" + st, V([ex + 0.2, ey + 0.005, ez - 0.02], [ex, ey, ez + 0.012], 22), m)
if "brows" in groups:
    cb = [0, ey + 0.022, ez + 0.01]
    for st, m in (("neutral", ""), ("angry", "Angry"), ("surprise", "Surprise"), ("browdown", "browDownLeft=1,browDownRight=1")):
        add("brows", "front_" + st, V([0, ey + 0.03, ez + 0.3], cb, 22), m)
        add("brows", "graze_" + st, V([ex + 0.09, ey - 0.06, ez + 0.19], [ex, ey + 0.018, ez + 0.01], 22), m)
if "mouth" in groups:
    my, mz = LMK["my"], LMK["mz"]
    cm = [0, my, mz]
    for st, m in (("rest", ""), ("PP", "viseme_PP=1"), ("open", "jawOpen=0.35"), ("aa", "viseme_aa=1"), ("smile", "Smile"),
                  ("toothy", "Toothy smile"), ("battle", "Battle cry"), ("close", "mouthClose=0.6,jawOpen=0.6")):
        add("mouth", "front_" + st, V([0, my + 0.01, mz + 0.28], cm, 22), m)
        add("mouth", "34_" + st, V([-0.16, my + 0.02, mz + 0.22], cm, 22), m)
json.dump(L, sys.stdout)
