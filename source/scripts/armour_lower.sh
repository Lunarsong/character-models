#!/bin/bash
# Knight LOWER armour + cloth + weapons, one command: author the MPFB clothes assets on the male, paint this kit's
# textures, dress the male and the female (upper armour + this kit, the rest shapes regenerated on each body over the
# upper armour actually dressed on it, co-skinned), export + validate the GLBs (+ socket contract, shared-skeleton
# gate incl. the extra joints), measure penetration per animation frame (dev harness), web copies + viewer pages, QA
# renders. Needs out/base_{male,female}.blend (scripts/build_all.sh). Re-run after the upper armour is re-authored:
# the tabard, belts, cape, clasps and mail skirt drape over its pieces (assets/mpfb_assets/clothes/knight_*).
# usage: scripts/armour_lower.sh [author] [tex] [dress] [fit] [web] [qa]    (default: author tex dress fit qa)
#        KINDS="male female"
set -euo pipefail
CH="$(cd "$(dirname "$0")/.." && pwd)"
cd "$CH"
export BLENDER_USER_RESOURCES="$CH/blender_profile"
BL=/Applications/Blender.app/Contents/MacOS/Blender
STEPS=${*:-author tex dress fit qa}
KINDS=${KINDS:-male female}
run() { "$BL" -b "$@" 2>&1 | grep -E '^CHR|^FIT|Error|Traceback|AssertionError|WARNING' || true; }
has() { [[ " $STEPS " == *" $1 "* ]]; }
blend() { [ "$1" = male ] && echo out/knight_lower.blend || echo out/knight_lower_$1.blend; }

has author && run out/base_male.blend --python-exit-code 1 -P scripts/armour_lower.py -- author   # ~5 min (MakeClothes)
has tex && python3 scripts/armour_lower_tex.py                                                       # kl_* sets (~1 min)
if has dress; then
  for k in $KINDS; do
    run out/base_$k.blend --python-exit-code 1 -P scripts/armour_lower.py -- dress $k --export     # blend + GLB (+ props)
  done
  python3 scripts/check_glb.py $(for k in $KINDS; do echo out/knight_lower_$k.glb; done) out/props/knight_*.glb \
      --json out/knight_lower_glb_report.json
  python3 scripts/armour_lower_sockets_check.py | tail -1
  python3 scripts/skeleton_check.py $(for k in $KINDS; do echo out/knight_lower_$k.glb; done)   # strict: extra joints too
fi
if has fit; then          # penetration per animation frame -> renders/lower_fit_<kind>_dev.json (+ gates in the log)
  for k in $KINDS; do
    run out/base_$k.blend --python-exit-code 1 -P scripts/armour_lower_fit.py -- dev $k step=3
  done
fi
if has web; then
  for k in $KINDS; do
    python3 scripts/glb_web.py out/knight_lower_$k.glb out/test/knight_lower_${k}_web.glb --max 1024
    python3 viewer/make_viewer.py out/test/knight_lower_${k}_web.glb -o viewer/knight_lower_${k}_lookdev.html \
        --title "Knight - lower armour, cloth, weapons ($k)" --sub "scripts/armour_lower.py" --town | tail -1
  done
fi
if has qa; then
  for k in $KINDS; do
    run $(blend $k) --python-exit-code 1 -P scripts/armour_lower_qa.py -- $k look      # look-dev (the blend has the upper too)
  done
fi
