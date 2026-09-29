#!/bin/bash
# Before / after contact sheets per face / hair item (user round-2 items 7-11, judge M18), from face_hair_shots.py renders
# (renders/facehair/<tag>_<kind>_*) and face_hair_viewer_shots.py viewer shots.
# usage: [BEFORE=<tag>] [OUT=<prefix>] scripts/face_hair_ba.sh <kind> <after tag> [viewer dir]   -> renders/facehair/<OUT:-ba>_<kind>_<item>.png
K=$1; A=$2; V=${3:-renders/facehair/viewer}; B=${BEFORE:-before}; O=${OUT:-ba}
cd "$(dirname "$0")/.."
R=renders/facehair
sheet() { CELL_W=${CW:-420} CELL_H=${CHH:-280} python3 scripts/ba_sheet.py "$@" | tail -1; }
sheet $R/${O}_${K}_7_brows.png "Item 7 eyebrows on the skin ($K): before / after (Blender 0.3 m, viewer)" "before,after" \
  "graze Angry|$R/${B}_${K}_brows_graze_angry.png|$R/${A}_${K}_brows_graze_angry.png" \
  "graze browDown|$R/${B}_${K}_brows_graze_browDown.png|$R/${A}_${K}_brows_graze_browDown.png" \
  "heavy ridge|$R/${B}_${K}_brows_graze_ridge_heavy.png|$R/${A}_${K}_brows_graze_ridge_heavy.png" \
  "low brows + down|$R/${B}_${K}_brows_graze_brows_low.png|$R/${A}_${K}_brows_graze_brows_low.png" \
  "old gaunt|$R/${B}_${K}_brows_graze_old_gaunt.png|$R/${A}_${K}_brows_graze_old_gaunt.png" \
  "viewer graze Angry|$V/${B}_${K}_v_brows_graze_angry.png|$V/${A}_${K}_v_brows_graze_angry.png"
sheet $R/${O}_${K}_8_lids.png "Item 8 closed lid texture ($K): before / after" "before,after" \
  "UV checker blink 3/4|$R/${B}_${K}_eyes_uv_34_blink.png|$R/${A}_${K}_eyes_uv_34_blink.png" \
  "UV checker blink front|$R/${B}_${K}_eyes_uv_front_blink.png|$R/${A}_${K}_eyes_uv_front_blink.png" \
  "UV checker wink|$R/${B}_${K}_eyes_uv_34_wink.png|$R/${A}_${K}_eyes_uv_34_wink.png" \
  "blink 3/4|$R/${B}_${K}_eyes_34_blink.png|$R/${A}_${K}_eyes_34_blink.png" \
  "viewer blink 3/4|$V/${B}_${K}_v_eyes_34_blink.png@0.25,0.2,0.75,0.75|$V/${A}_${K}_v_eyes_34_blink.png@0.25,0.2,0.75,0.75" \
  "viewer blink 0.5|$V/${B}_${K}_v_eyes_34_blink50.png@0.25,0.2,0.75,0.75|$V/${A}_${K}_v_eyes_34_blink50.png@0.25,0.2,0.75,0.75"
sheet $R/${O}_${K}_9_lashes.png "Item 9 lashes and the lid margin ($K): before / after" "before,after" \
  "profile blink|$R/${B}_${K}_eyes_profile_blink.png|$R/${A}_${K}_eyes_profile_blink.png" \
  "profile blink + squint|$R/${B}_${K}_eyes_profile_blink_squint.png|$R/${A}_${K}_eyes_profile_blink_squint.png" \
  "profile half blink|$R/${B}_${K}_eyes_profile_blink50.png|$R/${A}_${K}_eyes_profile_blink50.png" \
  "3/4 wink|$R/${B}_${K}_eyes_34_wink.png|$R/${A}_${K}_eyes_34_wink.png" \
  "viewer profile blink|$V/${B}_${K}_v_eyes_profile_blink.png@0.25,0.2,0.75,0.75|$V/${A}_${K}_v_eyes_profile_blink.png@0.25,0.2,0.75,0.75"
sheet $R/${O}_${K}_10_teeth.png "Item 10 teeth ($K): before / after" "before,after" \
  "laugh|$R/${B}_${K}_mouth_front_laugh.png|$R/${A}_${K}_mouth_front_laugh.png" \
  "big grin 3/4|$R/${B}_${K}_mouth_34_grin.png|$R/${A}_${K}_mouth_34_grin.png" \
  "viseme aa|$R/${B}_${K}_mouth_front_aa.png|$R/${A}_${K}_mouth_front_aa.png" \
  "jawOpen 0.35|$R/${B}_${K}_mouth_front_open.png|$R/${A}_${K}_mouth_front_open.png" \
  "viewer battle cry|$V/${B}_${K}_v_mouth_front_battle.png@0.2,0.2,0.8,0.8|$V/${A}_${K}_v_mouth_front_battle.png@0.2,0.2,0.8,0.8" \
  "viewer aa 3/4|$V/${B}_${K}_v_mouth_34_aa.png@0.2,0.2,0.8,0.8|$V/${A}_${K}_v_mouth_34_aa.png@0.2,0.2,0.8,0.8"
sheet $R/${O}_${K}_M18_lips.png "Judge M18 lip seal ($K): before / after" "before,after" \
  "rest|$R/${B}_${K}_mouth_front_rest.png|$R/${A}_${K}_mouth_front_rest.png" \
  "viseme PP|$R/${B}_${K}_mouth_front_PP.png|$R/${A}_${K}_mouth_front_PP.png" \
  "mouthPress|$R/${B}_${K}_mouth_front_press.png|$R/${A}_${K}_mouth_front_press.png" \
  "mouthClose + jawOpen 0.6|$R/${B}_${K}_mouth_front_close_jaw.png|$R/${A}_${K}_mouth_front_close_jaw.png" \
  "profile PP|$R/${B}_${K}_mouth_profile_PP.png|$R/${A}_${K}_mouth_profile_PP.png"
sheet $R/${O}_${K}_11_hair.png "Item 11 hair / scalp ($K): before / after (Blender and viewer)" "before,after" \
  "hairline front|$R/${B}_${K}_hair_hairline_front.png|$R/${A}_${K}_hair_hairline_front.png" \
  "hairline 3/4|$R/${B}_${K}_hair_hairline_34.png|$R/${A}_${K}_hair_hairline_34.png" \
  "temple|$R/${B}_${K}_hair_temple.png|$R/${A}_${K}_hair_temple.png" \
  "viewer hairline 3/4|$V/${B}_${K}_v_hair_hairline34.png|$V/${A}_${K}_v_hair_hairline34.png" \
  "viewer temple|$V/${B}_${K}_v_hair_temple.png|$V/${A}_${K}_v_hair_temple.png" \
  "viewer crown|$V/${B}_${K}_v_hair_crown.png|$V/${A}_${K}_v_hair_crown.png" \
  "viewer 3/4|$V/${B}_${K}_v_hair_34.png@0.2,0.05,0.8,0.75|$V/${A}_${K}_v_hair_34.png@0.2,0.05,0.8,0.75"
# round 5: the web-viewer face battery (scripts/face_viewer_battery.py; VB=<its output dir>), 5 directions per state
if [ -n "${VB:-}" ]; then
  row() { echo "$1|$VB/${B}_${K}_v_$2.png@0.2,0.2,0.8,0.8|$VB/${A}_${K}_v_$2.png@0.2,0.2,0.8,0.8"; }
  sheet $R/${O}_${K}_vb_eyes.png "Items 8 / 9 closed lids and lashes, web viewer, 0.2 m, 5 directions ($K): before / after" "before,after" \
    "$(row 'blink front' eyes_front_blink)" "$(row 'blink 3/4' eyes_34_blink)" "$(row 'blink below' eyes_below_blink)" \
    "$(row 'blink above' eyes_above_blink)" "$(row 'wink profile' eyes_prof_wink)" "$(row 'blink + squint 3/4' eyes_34_blinksquint)" \
    "$(row 'laugh eye below' eyes_below_laughc)"
  sheet $R/${O}_${K}_vb_mouth.png "Item 10 / M18 / M22 mouth, web viewer, 0.28 m ($K): before / after" "before,after" \
    "$(row 'rest front' mouth_front_rest)" "$(row 'PP (P/B/M) 3/4' mouth_34_PP)" "$(row 'toothy smile' mouth_front_toothy)" \
    "$(row 'battle cry 3/4' mouth_34_battle)" "$(row 'battle cry front' mouth_front_battle)" "$(row 'aa below' mouth_below_aa)" \
    "$(row 'tongue out 3/4' mouth_34_tongue)"
  sheet $R/${O}_${K}_vb_brows.png "Item 7 brows, web viewer, 0.3 m ($K): before / after" "before,after" \
    "$(row 'angry below' brows_below_angry)" "$(row 'angry profile' brows_prof_angry)" "$(row 'brow down above' brows_above_browdown)" \
    "$(row 'surprise below' brows_below_surprise)" "$(row 'blink 3/4' brows_34_blink)"
fi
