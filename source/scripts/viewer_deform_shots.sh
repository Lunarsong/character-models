#!/bin/bash
# Viewer (three.js) screenshots of the deformation fixes at the user's views: arms up (armpit back / front, neck and
# shoulders) and the squat (groin from below-front, glutes from behind), per body, for a viewer html. Retries blank
# shots (headless Chrome occasionally returns an empty frame for the 30 MB pages).
# usage: scripts/viewer_deform_shots.sh <viewer.html> <kind> <tag> [NOCOR=1]   -> renders/viewer/viewer_<kind>_<view>_<tag>.png
CH="$(cd "$(dirname "$0")/.." && pwd)"; cd "$CH"; mkdir -p renders/viewer
H=$1; K=$2; T=$3; NC=${4:-}
while read -r name pose cam; do
  out=renders/viewer/viewer_${K}_${name}_${T}.png
  for try in 1 2 3; do
    NOCOR=${NC#NOCOR=} CAM="$cam" scripts/viewer_shot.sh "$H" "$out" 'Body!' "${pose//_/ }" '' 800 800 > /dev/null
    [ "$(stat -f%z "$out" 2>/dev/null || echo 0)" -gt 20000 ] && break
  done
  echo "$out $(stat -f%z "$out")"
done <<'LIST'
armsup_back Arms_up upperarm_r,-0.45,-0.05,-0.6,0.035,-0.1,0.0
armsup_front Arms_up upperarm_l,0.45,-0.05,0.6,-0.035,-0.1,0.0
armsup_neck Arms_up neck_01,0.0,-0.05,0.85,0.0,-0.1,0.0
squat_below Squat pelvis,0.0,-0.3,0.5,0.0,-0.06,0.03
squat_back Squat pelvis,0.15,-0.3,-0.6,0.0,-0.1,0.0
LIST
