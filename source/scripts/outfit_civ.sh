#!/bin/bash
# CIVILIAN outfits (peasant, archer, mix-and-match 'peasant_hood'): textures -> author (male + female) -> dress + clips
# + export (dressed GLB, one GLB per piece, kit pieces, manifest) -> validate -> QA (penetration gate) -> renders.
#   scripts/outfit_civ.sh                      # everything (~15 min)
#   scripts/outfit_civ.sh author dress check   # stages: tex author dress check qa look lineup viewer
#   KINDS="male" OUTFITS="archer" scripts/outfit_civ.sh dress qa
set -e
cd "$(dirname "$0")/.."
export BLENDER_USER_RESOURCES=$PWD/blender_profile
BL=${BL:-/Applications/Blender.app/Contents/MacOS/Blender}
KINDS=${KINDS:-"male female"}
OUTFITS=${OUTFITS:-"peasant archer peasant_hood"}
STAGES=${*:-"tex author dress check qa look lineup viewer"}
LOG=${LOG:-/tmp/rts_civ_logs}; mkdir -p "$LOG"
has() { [[ " $STAGES " == *" $1 "* ]]; }
run() { local name=$1; shift; echo "== $name"; "$@" > "$LOG/$name.log" 2>&1 || { tail -40 "$LOG/$name.log"; echo "FAILED: $name (log $LOG/$name.log)"; exit 1; }; grep -E "^CIV (asset|glb|clip|penetration|   POKE|dressed body|WARNING|done)|^CIV .*(built|kit)" "$LOG/$name.log" | head -60 || true; }

has tex && run tex python3 scripts/outfit_civ_tex.py
if has author; then
  for k in $KINDS; do
    run author_peasant_$k $BL -b out/base_$k.blend --python-exit-code 1 -P scripts/outfit_peasant.py -- author
    run author_archer_$k $BL -b out/base_$k.blend --python-exit-code 1 -P scripts/outfit_archer.py -- author
  done
fi
if has dress; then
  for k in $KINDS; do for o in $OUTFITS; do
    run dress_${o}_$k $BL -b out/base_$k.blend --python-exit-code 1 -P scripts/outfit_civ.py -- dress $k $o
  done; done
fi
if has check; then
  echo "== validate"
  files=""
  for k in $KINDS; do for o in $OUTFITS; do files="$files out/civ/${o}_$k.glb"; done; files="$files $(ls out/civ/pieces/$k/*.glb)"; done
  python3 scripts/check_glb.py $files out/civ/props/longbow.glb > "$LOG/check_glb.log" 2>&1 || { tail -30 "$LOG/check_glb.log"; echo "FAILED: check_glb"; exit 1; }
  tail -5 "$LOG/check_glb.log"
  python3 scripts/skeleton_check.py out/base_male.glb out/base_female.glb $(for k in $KINDS; do echo out/civ/archer_$k.glb; done) --warn-extra > "$LOG/skeleton.log" 2>&1 || { tail -20 "$LOG/skeleton.log"; echo "FAILED: skeleton_check"; exit 1; }
  tail -4 "$LOG/skeleton.log"
  [[ " $KINDS " == *" male "* && " $KINDS " == *" female "* ]] && { python3 scripts/outfit_civ_props_check.py male female || exit 1; }
fi
if has qa; then
  for k in $KINDS; do for o in $OUTFITS; do
    run qa_${o}_$k $BL -b out/civ/${o}_${k}_export.blend --python-exit-code 1 -P scripts/outfit_civ.py -- qa $k $o
  done; done
  python3 scripts/outfit_civ_gate.py $KINDS
fi
if has look; then
  for k in $KINDS; do for o in $OUTFITS; do
    run look_${o}_$k $BL -b out/civ/${o}_${k}_export.blend --python-exit-code 1 -P scripts/outfit_civ_look.py -- look $k $o
  done; done
fi
if has lineup; then
  run lineup_male $BL -b --python-exit-code 1 -P scripts/outfit_civ_look.py -- lineup male
  run lineup_female $BL -b --python-exit-code 1 -P scripts/outfit_civ_look.py -- lineup female
  python3 - <<'PY'
from PIL import Image
import os
for k in ("male", "female"):
    for nm, fr in (("rts", 0.30), ("rts_zoom", 0.60)):
        f = "renders/civ/lineup_%s_%s.png" % (k, nm)
        if os.path.exists(f):
            im = Image.open(f); W, H = im.size; cw, ch = int(W * fr), int(H * fr * 0.72)
            box = ((W - cw) // 2, (H - ch) // 2 - int(H * 0.03), (W + cw) // 2, (H + ch) // 2 - int(H * 0.03))
            s = 3 if nm == "rts" else 2
            im.crop(box).resize((cw * s, ch * s), Image.NEAREST).save(f.replace(".png", "_crop.png"))
PY
fi
if has viewer; then
  # look-dev pages (make_viewer.py, the village lane for the RTS cameras) + headless shots from one page load each
  for k in $KINDS; do for o in peasant archer; do
    page=viewer/civ_${o}_${k}_lookdev.html
    run viewer_${o}_$k python3 viewer/make_viewer.py out/civ/${o}_$k.glb --town -o $page --title "$o ($k)" \
        --sub "civilian outfit (scripts/outfit_civ.sh)"
    clip2=$([ $o = archer ] && echo shoot || echo work); t2=$([ $o = archer ] && echo 2.1 || echo 1.0)
    python3 - "$o" "$k" "$clip2" "$t2" > /tmp/rts_civ_batch_${o}_${k}.json <<'PY'
import json, sys
o, k, c2, t2 = sys.argv[1:5]
p = "renders/civ/viewer_%s_%s_" % (o, k)
print(json.dumps([
    {"out": p + "full.png", "view": "Full body", "clip": "idle", "time": 0.5, "ui": False},
    {"out": p + "face34.png", "view": "Face 3/4", "clip": "idle", "time": 0.5, "ui": False},
    {"out": p + "back.png", "view": "Back", "clip": "idle", "time": 0.5, "ui": False},
    {"out": p + "walk.png", "view": "Full body", "clip": "walk", "time": 0.3, "ui": False},
    {"out": p + "action.png", "view": "Full body", "clip": c2, "time": float(t2), "ui": False, "yaw": 60},
    {"out": p + "rts.png", "view": "RTS", "clip": "idle", "time": 0.5, "scene": "town", "ui": False},
    {"out": p + "rtszoom.png", "view": "RTS zoom", "clip": c2, "time": float(t2), "scene": "town", "ui": False},
    {"out": p + "ui.png", "view": "Full body", "clip": "walk", "time": 0.3, "ui": True}]))
PY
    BATCH=/tmp/rts_civ_batch_${o}_${k}.json viewer/shot.sh $page - > "$LOG/shots_${o}_$k.log" 2>&1 || { tail -5 "$LOG/shots_${o}_$k.log"; echo "shots failed: $o $k"; }
  done; done
fi
echo "civ outfits done (logs in $LOG)"
