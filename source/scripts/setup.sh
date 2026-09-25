#!/bin/bash
# One-time (idempotent) setup of the character pipeline:
#   - downloads MPFB 2.0.17 (extensions.blender.org) and the CC0 MakeHuman asset packs into characters/assets/
#   - installs MPFB into the PROJECT Blender profile (characters/blender_profile), never the user's own profile
#   - extracts the asset packs into MPFB's user data dir (what MPFB's "Load pack from zip" operator does)
#   - runs scripts/mpfb_probe.py to verify MPFB loads headless and lists rigs / skins / eyes / hair / face units
# usage: characters/scripts/setup.sh
set -euo pipefail
CH="$(cd "$(dirname "$0")/.." && pwd)"
export BLENDER_USER_RESOURCES="$CH/blender_profile"
BL=/Applications/Blender.app/Contents/MacOS/Blender
A="$CH/assets"
mkdir -p "$A" "$CH/blender_profile" "$CH/out" "$CH/renders" "$CH/viewer"

MPFB_ZIP="$A/add-on-mpfb-v2.0.17.zip"
MPFB_SHA=4f0a879d64a39bf646fbf5f53601ac678855da329d650617dca5737548239a87
if [ ! -f "$MPFB_ZIP" ]; then
  # archive_url for id 'mpfb' in https://extensions.blender.org/api/v1/extensions/?format=json
  curl -fsSL -o "$MPFB_ZIP" "https://extensions.blender.org/download/sha256:$MPFB_SHA/add-on-mpfb-v2.0.17.zip"
fi
echo "$MPFB_SHA  $MPFB_ZIP" | shasum -a 256 -c -

# CC0 packs. Every asset in them is tagged license CC0 in its packs/*.json (checked by mpfb_probe.py).
get() { [ -f "$A/$1" ] || curl -fsSL -o "$A/$1" "$2"; }
get makehuman_system_assets_cc0.zip http://files.makehumancommunity.org/asset_packs/makehuman_system_assets/makehuman_system_assets_cc0.zip
get faceunits01.zip https://files.makehumancommunity.org/functional/faceunits01.zip   # 52 ARKit face units (Mika Suominen, CC0)
get visemes02.zip   https://files.makehumancommunity.org/functional/visemes02.zip     # 15 Meta/Oculus visemes (CC0)
get visemes01.zip   https://files.makehumancommunity.org/functional/visemes01.zip     # 22 Microsoft visemes (CC0, not used in the GLB)

if [ ! -d "$CH/blender_profile/extensions/user_default/mpfb" ]; then
  "$BL" --command extension install-file -r user_default -e "$MPFB_ZIP"
fi
# MPFB user data dir = bpy.utils.extension_path_user('bl_ext.user_default.mpfb') + /data
UD="$CH/blender_profile/extensions/.user/user_default/mpfb/data"
mkdir -p "$UD"
for z in makehuman_system_assets_cc0 faceunits01 visemes02 visemes01; do
  if [ ! -f "$UD/packs/$z.json" ] && [ ! -f "$UD/packs/${z%_cc0}.json" ]; then
    echo "installing pack $z"; unzip -q -o "$A/$z.zip" -d "$UD"
  fi
done
"$BL" -b --python-exit-code 1 -P "$CH/scripts/mpfb_probe.py" 2>&1 | grep -E '^(PROBE|Error|Traceback)' || true
