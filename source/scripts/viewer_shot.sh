#!/bin/bash
# Headless Chrome screenshot of a character viewer page (Chrome found by work/browser.sh, read-only; driven over the
# DevTools protocol by scripts/cdp_eval.mjs, Node >= 22, real time: it waits for window.__READY__, i.e. for the base
# GLB, the parts GLB and the variant textures).
# usage: viewer_shot.sh viewer/base_male.html out.png [view] [pose] [expr] [w h]
#   RANDOM_FACE=<seed> (iteration-1 name) = SHOT_CUST='{"seed":<seed>}': a seeded Randomise of the Customise tab
#   view: Body|Face|Back|RTS (append ! to hide the UI panels)  pose: any viewer pose ('Arms up', 'Squat', 'Deep squat', ...)
#   expr: any preset name of scripts/expressions.json (Smile, 'Big grin', 'Wink L', ...)
#   env CLICK='Wink L' clicks those buttons (comma separated) once the model is ready, like a user would (the
#       template's __CLICK__ hook, then 1.5 s of frames are played)
#   env BLINK=1 keeps the idle blink running (default: blink off for a stable screenshot)
#   env CAM="px,py,pz,tx,ty,tz" custom camera (glTF metres, +Y up, character faces +Z), or
#       CAM="bone,cx,cy,cz,tx,ty,tz": target = posed bone head + t, camera = target + c     NOCOR=1 correctives off
#   env SHOT_CUST='{"seed":3}' or a recipe JSON (Customise tab 'Copy recipe'), SHOT_PANEL=cust (open the tab)
#   (scripts/cdp_shot.py runs several shots with arbitrary viewer JS on one page load)
IN=$1; OUT=$2; VIEW=${3:-Body}; POSE=${4:-}; EXPR=${5:-}; W=${6:-1280}; H=${7:-860}
CH="$(cd "$(dirname "$0")/.." && pwd)"; WORK="$CH/../work"
HOME_DIR="$WORK"; . "$WORK/browser.sh"
PAGE=$(mktemp "$CH/viewer/.shot_XXXXXX"); mv "$PAGE" "$PAGE.html"; PAGE="$PAGE.html"
[ -n "${RANDOM_FACE:-}" ] && [ -z "${SHOT_CUST:-}" ] && SHOT_CUST="{\"seed\":${RANDOM_FACE}}"
CLICK="${CLICK:-}" BLINK="${BLINK:-0}" CAM="${CAM:-}" NOCOR="${NOCOR:-}" SHOT_CUST="${SHOT_CUST:-}" SHOT_PANEL="${SHOT_PANEL:-}" \
python3 - "$IN" "$PAGE" "$WORK" "$VIEW" "$POSE" "$EXPR" <<'PY'
import sys, json, os
src, page, work, view, pose, expr = sys.argv[1:7]
click = [c for c in os.environ.get("CLICK", "").split(",") if c]
cam = [v if i == 0 and not v.replace('.', '').replace('-', '').isdigit() else float(v)
       for i, v in enumerate(os.environ["CAM"].split(","))] if os.environ.get("CAM") else None
h = open(src).read().replace('https://cdn.jsdelivr.net/npm/three@0.147.0', 'file://%s/vendor/three' % work)
inj = ('<script>window.__VIEW__=%s;window.__POSE__=%s;window.__EXPR__=%s;window.__NOBLINK__=%s;window.__CLICK__=%s;'
       'window.__CAM__=%s;window.__NOCOR__=%s;window.__CUST__=%s;window.__PANEL__=%s;</script>') % (
    json.dumps(view.rstrip('!')), json.dumps(pose or None), json.dumps(expr or None),
    'false' if os.environ.get("BLINK") == "1" else 'true', json.dumps(click), json.dumps(cam),
    'true' if os.environ.get("NOCOR") else 'false', os.environ.get('SHOT_CUST') or 'null',
    json.dumps(os.environ.get('SHOT_PANEL') or None))
h = h.replace('<script type="application/octet-stream" id="glb">', inj + '<script type="application/octet-stream" id="glb">', 1)
h = h.replace('</style>', '.panel{display:none!important}</style>') if view.endswith('!') else h
open(page, 'w').write(h)
PY
# DevTools protocol in real time (scripts/cdp_eval.mjs): wait until the model is loaded (window.__READY__), let the
# expression ease in for 1.5 s, then capture (a --screenshot virtual-time budget can expire before a 30 MB page loads)
rm -f "$OUT"
node "$CH/scripts/cdp_eval.mjs" "$CHROME" "file://$PAGE" "window.__READY__ || null" 240 "$OUT" 1500 "${W}x${H}" >/dev/null
rm -f "$PAGE"
ls -la "$OUT" 2>/dev/null | awk '{print $5}'
