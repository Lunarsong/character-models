#!/bin/bash
# Full, reproducible build of the base humans: rig -> skin textures (+ skin-tone variants) -> humans (.blend + .glb:
# face-quality layer scripts/face_lib.py, customisation layer scripts/cust_lib.py, pose-space correctives + their morph
# normals scripts/correctives.py) -> validation (Khronos + GLB morph test) -> QA renders (deformation stress set +
# metrics, face-shape grids, look-dev, expression / mouth renders, customisation sheets) -> HTML viewers -> viewer test.
# Before/after QA: see STATUS.md.
# usage: characters/scripts/build_all.sh [male] [female] [male_stocky male_slim female_slim]   (default: male female)
#        QA=0 skips the QA renders; CUST=0 builds without the customisation layer (cust_lib.py)
set -euo pipefail
CH="$(cd "$(dirname "$0")/.." && pwd)"
cd "$CH"
export BLENDER_USER_RESOURCES="$CH/blender_profile"
BL=/Applications/Blender.app/Contents/MacOS/Blender
KINDS=${*:-male female}
run() { "$BL" -b "$@" 2>&1 | grep -E '^CHR|^RIG|Error|Traceback|AssertionError' || true; }

[ -d "$BLENDER_USER_RESOURCES/extensions/user_default/mpfb" ] || ./scripts/setup.sh
python3 scripts/make_rig.py | head -1                                  # rts_human rig + weights -> MPFB user data
# hair colours must match PRESETS[kind]["hair_rgb"] in chr_lib.py (also used to tint the scalp in bake_skin.py)
# 'neutral' = grey strand atlas that the hair-colour variants tint (customisation)
python3 scripts/hair_tex.py male 0.17 0.11 0.07 female 0.20 0.11 0.07 neutral 0.6667 0.6667 0.6667  # procedural strand atlas
python3 scripts/part_tex.py                                            # neutral CC0 hair textures + eye colour JPEGs
run --python-exit-code 1 -P scripts/hair_gen.py -- crop braid beard_short beard_full stubble cap_crop cap_braid cap_generic   # grooms + stubble shell + scalp caps -> MPFB hair assets
for k in $KINDS; do
  case $k in male|female)                                              # proportion variants reuse these textures
    # clean checkout: the first bake writes the lid-independent lining reference (<k>_skin_lining_ref.jpg) that the
    # face keys read (face_lib.lining_ref_image); the second bakes the lid texture on those keys (reproducible)
    [ -f assets/textures/${k}_skin_lining_ref.jpg ] || run --python-exit-code 1 -P scripts/bake_skin.py -- $k
    run --python-exit-code 1 -P scripts/bake_skin.py -- $k             # assets/textures/<k>_skin_*.{png,jpg} (+ mouth shade map)
    run --python-exit-code 1 -P scripts/skin_variants.py -- $k ;;      # assets/textures/<k>_skin_<tone>_base.jpg
  esac
  run --python-exit-code 1 -P scripts/base_humans.py -- $k             # out/base_<k>.{blend,glb}, out/parts_<k>.glb
done
python3 scripts/check_glb.py $(for k in $KINDS; do echo out/base_$k.glb; [ -f out/parts_$k.glb ] && echo out/parts_$k.glb; done) --json out/glb_report.json
# shared skeleton (M15): every base GLB on disk must have identical joint rest rotations (only translations differ)
SKEL=$(ls out/base_*.glb 2>/dev/null | grep -v _web || true)
[ $(echo $SKEL | wc -w) -lt 2 ] || python3 scripts/skeleton_check.py $(ls out/base_male.glb 2>/dev/null) $(echo $SKEL | tr ' ' '\n' | grep -v '^out/base_male.glb$') --json out/skeleton_report.json
# face keys as three.js sees them: ARKit sides, blink closes, winks, rigid lower teeth, static upper teeth, extras,
# customisation morphs move the tooth sets affinely
python3 scripts/morph_check.py $(for k in $KINDS; do echo out/base_$k.glb; done) --json out/morph_report.json
# face / hair gate on the engine meshes (user round-2 items 7-11): brows and lashes outside the skin in every key /
# mix, closed-lid texture stretch, lids clear of the eye, lip seal, scalp coverage (scripts/face_qa.py)
for k in $KINDS; do
  if "$BL" -b out/base_${k}_export.blend --python-exit-code 1 -P scripts/face_qa.py -- $k --gate --json out/face_qa_$k.json > out/face_qa_$k.log 2>&1; then
    grep -E '^FACEQA [a-z_]+: ' out/face_qa_$k.log || true
  else
    grep -E 'FAIL|Error|Traceback' out/face_qa_$k.log || true; echo "face_qa failed for $k (out/face_qa_$k.log)"; exit 1
  fi
done
if [ "${QA:-1}" = 1 ]; then
  for k in $KINDS; do
    rm -f renders/deform_${k}_* renders/face_${k}_* renders/look_${k}_* renders/expr_${k}_* renders/close_${k}_* renders/section_${k}_*
    run out/base_${k}_export.blend --python-exit-code 1 -P scripts/deform_test.py -- $k
    run out/base_${k}_export.blend --python-exit-code 1 -P scripts/face_grid.py -- $k
    run out/base_${k}_export.blend --python-exit-code 1 -P scripts/render_look.py -- $k
    EXCLUDE=_z python3 scripts/sheet.py deform_$k 6 380 >/dev/null
    FILTER=_z OUT=deform_${k}_details_sheet.png python3 scripts/sheet.py deform_$k 6 330 >/dev/null
    run out/base_${k}_export.blend --python-exit-code 1 -P scripts/deform_test.py -- $k nocor arms_up viewer_arms_up arms_up_180 squat deep_squat
    FILTER=nocor_ OUT=deform_${k}_nocor_sheet.png python3 scripts/sheet.py deform_$k 6 330 >/dev/null   # plain LBS (correctives off)
    for v in front side eyes mouth; do python3 scripts/sheet.py face_${k}_$v 10 260 >/dev/null; done
    python3 scripts/sheet.py look_$k 4 600 body_front,body_34,body_side,body_back,face_front,face_34,face_profile >/dev/null
    # mouth / expression QA: presets (scripts/expressions.json), 0.3 m close-ups, sagittal mouth cut-aways
    run out/base_${k}_export.blend --python-exit-code 1 -P scripts/expr_render.py -- $k --views face,q34,mouth,eyes
    for v in face q34 mouth eyes; do python3 scripts/sheet.py expr_${k}_$v 6 300 >/dev/null; done
    run out/base_${k}_export.blend --python-exit-code 1 -P scripts/closeup_render.py -- $k
    python3 scripts/sheet.py close_$k 4 330 >/dev/null
    for keys in "" "jawOpen=0.35" "smileOpenLeft=1+smileOpenRight=1+mouthSmileLeft=0.7+mouthSmileRight=0.7"; do
      run out/base_${k}_export.blend --python-exit-code 1 -P scripts/mouth_section.py -- $k section_$k --cut 0.001 --zoom 1.6 ${keys:+--keys $keys}
    done
    python3 scripts/sheet.py section_$k 3 400 >/dev/null
    if [ "${CUST:-1}" = 1 ]; then                                      # customisation: every slider +-1, 6 random x 5 expr
      rm -f renders/cust_${k}_* renders/custgrid_${k}_*
      run out/base_${k}_export.blend --python-exit-code 1 -P scripts/cust_qa.py -- $k sheet grid
      python3 scripts/sheet.py cust_$k 8 300 >/dev/null
      python3 scripts/cust_grid_sheet.py $k >/dev/null
    fi
  done
fi
for k in $KINDS; do
  python3 scripts/mkviewer_char.py out/base_$k.glb "Base $k"
done
# the viewers as a user drives them: every expression button, idle blink on (the winking eye must stay closed), on the
# body, the default parts and every alternate part of the parts GLB
python3 scripts/viewer_check.py $(for k in $KINDS; do echo viewer/base_$k.html; done) --json out/viewer_report.json
# look-dev viewer pages (viewer/make_viewer.py + lookdev_template.html; base + parts GLB, village) and their QA shots
if [ "${LOOKDEV:-1}" = 1 ]; then QA=${QA:-1} TESTS=${TESTS:-0} viewer/build_lookdev.sh $KINDS; fi
