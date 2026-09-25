#!/bin/bash
# Headless screenshots of a look-dev viewer page (viewer/make_viewer.py output). Modelled on work/shot.sh and uses
# work/browser.sh (read-only) to find Chrome and the GPU backend; the page is driven through its window.LD API over
# the DevTools pipe (viewer/shot_cdp.js: no TCP port, waits for the page to finish loading, one load for many shots).
#
# usage: viewer/shot.sh <viewer.html> <out.png> [view] [clip] [time] [morphs] [w h]
#   view    camera preset: 'Full body' | Face | 'Face 3/4' | Back | RTS | 'RTS zoom' (case-insensitive prefix is fine),
#           or '{p:[x,y,z],t:[x,y,z],fov:30}' in metres (glTF space: +Y up, the character faces +Z)
#   clip    animation clip (a GLB clip, or a viewer test clip: Idle, Walk, 'Range of motion', 'T pose', 'Arms up',
#           'Arms forward', Squat, Lunge, Fists, 'Torso twist', 'Head turn', 'Look down', Shrug); '' or none = bind pose
#   time    clip time in seconds (the clip is paused there)
#   morphs  'jawOpen=0.6,eyeBlinkLeft=1' and/or an expression preset: 'Smile' | 'Toothy smile' | 'Battle cry' | Angry |
#           Surprise | Sad | Wink | 'Eyes closed' | 'Look left', e.g. 'Smile,jawOpen=0.2'
#   w h     viewport (default 1280 860)
# env: CLEAN=1 hides the panels · SCENE=studio|town · SHADING=lit|clay|normals|albedo · SKEL=1 · WIRE=1 · RULER=1
#      CORRN=1 applies the GLB normal deltas of corr_* shapes (off by default, see the viewer) ·
#      TALK=<s> talk-demo frame at s seconds (CAPTION=1 keeps its caption with CLEAN=1) · YAW=<deg> turns the character
#      HIDE='male_hair,Eyebrows' / ONLY=... / SHOW=... (mesh node names, labels or slot groups) · VARIANT='eyebrows=eyebrow002'
#      MATVAR=<KHR_materials_variants name> · EXPOSURE= ENV= LIGHT=<deg> FOV=<deg> · SCALE=2 (device pixel ratio)
#      BLINK=1 keeps the idle blink (off by default so shots are repeatable) · TIMEOUT=240 (s)
#      INFO=file.json also writes the page's LD.info(true) (meshes, shapes, clips, visemes, live morph influences)
#      BATCH=specs.json: several shots from ONE page load; a JSON list of {"out": .., "view": .., "clip": .., "time": ..,
#      "morphs": .., "scene": .., "shading": .., "skeleton": true, "wire": true, "talk": 1.2, "hide": [..], "ui": false,
#      "yaw": 30, ...} (window.LD.set keys); <out.png> may then be '-'.
# Several shot.sh calls may run in parallel (each has its own browser profile).
IN=$1; OUT=$2; VIEW=${3:-Full body}; CLIP=${4-__keep__}; TIME=${5:-}; MORPHS=${6-__keep__}; W=${7:-1280}; H=${8:-860}
[ -f "$IN" ] || { echo "usage: shot.sh <viewer.html> <out.png> [view] [clip] [time] [morphs] [w h]" >&2; exit 2; }
VDIR="$(cd "$(dirname "$0")" && pwd)"; WORK="$(cd "$VDIR/../../work" && pwd)"
HOME_DIR="$WORK"; . "$WORK/browser.sh"
command -v node >/dev/null || { echo "node not found" >&2; exit 1; }
SPEC=$(mktemp "${TMPDIR:-/tmp}/ldspec.XXXXXX")
python3 - "$SPEC" "$OUT" "$VIEW" "$CLIP" "$TIME" "$MORPHS" <<'PY'
import sys, os, json, re
spec, out, view, clip, t, morphs = sys.argv[1:7]
E = os.environ.get
def js_obj(s):          # '{p:[1,2,3],t:[0,1,0]}' -> dict
    return json.loads(re.sub(r'([{,]\s*)([A-Za-z_]\w*)\s*:', r'\1"\2":', s))
base = {'ui': E('CLEAN', '0') != '1', 'blink': E('BLINK', '0') == '1', 'turntable': False}
if E('CAPTION') == '1': base['captionAlways'] = True
for k, env, conv in [('scene', 'SCENE', str), ('shading', 'SHADING', str), ('skeleton', 'SKEL', lambda v: v == '1'),
                     ('wire', 'WIRE', lambda v: v == '1'), ('corrNormals', 'CORRN', lambda v: v == '1'), ('ruler', 'RULER', lambda v: v == '1'), ('talk', 'TALK', float),
                     ('yaw', 'YAW', float), ('exposure', 'EXPOSURE', float), ('env', 'ENV', float), ('light', 'LIGHT', float),
                     ('fov', 'FOV', float), ('matVariant', 'MATVAR', str), ('hide', 'HIDE', str), ('only', 'ONLY', str), ('show', 'SHOW', str)]:
    if E(env): base[k] = conv(E(env))
if E('VARIANT'): base['variant'] = dict(kv.split('=', 1) for kv in E('VARIANT').split(',') if '=' in kv)
def one(o):
    d = dict(base); d.update(o)
    v = d.get('view')
    if isinstance(v, str) and v.strip().startswith('{'): d['view'] = js_obj(v)
    elif isinstance(v, str):
        d['view'] = v
    return d
shots = []
if E('BATCH'):
    shots = [one(o) for o in json.load(open(E('BATCH')))]
else:
    o = {'out': out, 'view': view}
    if clip != '__keep__': o['clip'] = clip or None
    if t: o['time'] = float(t)
    o['morphs'] = '' if morphs == '__keep__' else morphs
    shots = [one(o)]
# a preset name may be given in any case / as a prefix: resolved in the page (LD.set matches exact names only)
for s in shots:
    if isinstance(s.get('view'), str):
        s['view'] = {'full': 'Full body', 'body': 'Full body', 'face': 'Face', 'face34': 'Face 3/4', 'back': 'Back',
                     'rts': 'RTS', 'rtszoom': 'RTS zoom'}.get(re.sub(r'[^a-z0-9]', '', s['view'].lower()), s['view'])
if E('INFO'): shots.append({'info': E('INFO')})
for s in shots:
    if s.get('out'): s['out'] = os.path.abspath(s['out'])
    if s.get('info'): s['info'] = os.path.abspath(s['info'])
json.dump(shots, open(spec, 'w'))
PY
[ $? -eq 0 ] || { rm -f "$SPEC"; exit 1; }
node "$VDIR/shot_cdp.js" "$CHROME" "$IN" "$SPEC" "$W" "$H" "${SCALE:-1}" "${TIMEOUT:-240}" $GPU_FLAGS
RC=$?
rm -f "$SPEC"
exit $RC
