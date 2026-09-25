"""Web-viewer face battery (user items 7-10, judge M17 / M18 / M22): every face item from 5 directions (front, 3/4, profile,
below, above) at 0.2-0.3 m in the look-dev viewer (the user orbits freely). Prints a viewer/shot.sh BATCH spec.
usage: python3 scripts/face_viewer_battery.py <kind> <tag> <outdir> [eyes,mouth,brows] > spec.json && SCENE=studio BATCH=spec.json viewer/shot.sh <page> -"""
import sys, json, os, math
kind, tag, outdir = sys.argv[1:4]
groups = sys.argv[4].split(",") if len(sys.argv) > 4 else ["eyes", "mouth", "brows"]
LMK = {"male": dict(ex=0.0306, ey=1.7401, ez=0.1357, my=1.6649, mz=0.1642, top=1.8501),
       "female": dict(ex=0.0301, ey=1.6074, ez=0.1169, my=1.5431, mz=0.1467, top=1.7192)}[kind]
ex, ey, ez = LMK["ex"], LMK["ey"], LMK["ez"]
my, mz = LMK["my"], LMK["mz"]
L = []
def cam(t, az, el, d, fov=22):
    a, e = math.radians(az), math.radians(el)
    p = [t[0] + d * math.cos(e) * math.sin(a), t[1] + d * math.sin(e), t[2] + d * math.cos(e) * math.cos(a)]
    return {"p": [round(v, 4) for v in p], "t": [round(v, 4) for v in t], "fov": fov}
def add(g, name, view, morphs=""):
    L.append(dict(out=os.path.join(outdir, "%s_%s_v_%s_%s.png" % (tag, kind, g, name)), view=view, morphs=morphs, clip=None, ui=False))
DIRS = {"front": (0, 0), "34": (40, 5), "prof": (85, 0), "below": (15, -40), "above": (10, 45), "back34": (120, 10)}
if "eyes" in groups:
    ce = [ex, ey, ez]
    for st, m in (("open", ""), ("blink", "eyeBlinkLeft=1"), ("blink50", "eyeBlinkLeft=0.5"), ("wink", "Wink"),
                  ("blinksquint", "eyeBlinkLeft=1,eyeSquintLeft=1"), ("smile", "Smile"), ("laughc", "eyeSquintLeft=0.5,cheekSquintLeft=0.8,mouthSmileLeft=0.7"),
                  ("lookdown", "eyeLookDownLeft=1"), ("wide", "eyeWideLeft=1")):
        for dn in ("front", "34", "prof", "below", "above"):
            az, el = DIRS[dn]
            add("eyes", "%s_%s" % (dn, st), cam(ce, az, el, 0.2 if dn != "prof" else 0.22), m)
if "brows" in groups:
    cb = [ex, ey + 0.02, ez]
    for st, m in (("neutral", ""), ("angry", "Angry"), ("surprise", "Surprise"), ("browdown", "browDownLeft=1,browDownRight=1"),
                  ("blink", "eyeBlinkLeft=1"), ("squint", "eyeSquintLeft=1,cheekSquintLeft=1")):
        for dn in ("front", "34", "prof", "below", "above"):
            az, el = DIRS[dn]
            add("brows", "%s_%s" % (dn, st), cam(cb, az, el, 0.3), m)
if "mouth" in groups:
    cm = [0, my, mz]
    for st, m in (("rest", ""), ("PP", "viseme_PP=1"), ("FF", "viseme_FF=1"), ("press", "mouthPressLeft=1,mouthPressRight=1"),
                  ("close", "mouthClose=0.6,jawOpen=0.6"), ("open", "jawOpen=0.35"), ("aa", "viseme_aa=1"), ("smile", "Smile"),
                  ("toothy", "Toothy smile"), ("battle", "Battle cry"), ("angry", "Angry"), ("frown", "Sad"),
                  ("tongue", "jawOpen=0.4,tongueOut=1"), ("oh", "viseme_O=1"), ("ss", "viseme_SS=1")):
        for dn in ("front", "34", "prof", "below", "above"):
            az, el = DIRS[dn]
            add("mouth", "%s_%s" % (dn, st), cam(cm, az, el, 0.28), m)
json.dump(L, sys.stdout)
