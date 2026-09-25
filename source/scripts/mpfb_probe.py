# Verifies MPFB loads headless in the project profile and lists what it can build with.
# run: BLENDER_USER_RESOURCES=characters/blender_profile Blender -b --python-exit-code 1 -P characters/scripts/mpfb_probe.py
import bpy, os, json, glob, collections, addon_utils
addon_utils.enable("bl_ext.user_default.mpfb", default_set=False)
import bl_ext.user_default.mpfb as mpfb
from bl_ext.user_default.mpfb.services import LocationService, AssetService, TargetService
from bl_ext.user_default.mpfb.services.faceservice import FaceService, ARKIT_FACEUNITS, META_VISEMES

mf = open(os.path.join(os.path.dirname(mpfb.__file__), "blender_manifest.toml")).read()
ver = [l for l in mf.splitlines() if l.startswith("version")][0]
print("PROBE MPFB", ver, "blender", bpy.app.version_string)
ud = LocationService.get_user_data(); md = LocationService.get_mpfb_data()
print("PROBE user data", ud)
print("PROBE rigs(builtin)", sorted(os.path.basename(p)[4:-5] for p in glob.glob(md + "/rigs/standard/rig.*.json")),
      "rigify", sorted(os.path.basename(p)[4:-5] for p in glob.glob(md + "/rigs/rigify/rig.*.json")))
print("PROBE rigs(custom)", [r["name"] for r in AssetService.get_custom_rigs(use_cache=False)])
for sub, pat in [("skins", "*.mhmat"), ("eyes", "*.mhclo"), ("eyebrows", "*.mhclo"), ("eyelashes", "*.mhclo"),
                 ("teeth", "*.mhclo"), ("tongue", "*.mhclo"), ("hair", "*.mhclo"), ("proxymeshes", "*.proxy"), ("clothes", "*.mhclo")]:
    roots = AssetService.get_asset_roots(sub)
    names = sorted({p.stem for p in AssetService.find_asset_files_matching_pattern(roots, pat)}) if roots else []
    print("PROBE", sub, len(names), names)
lic = collections.Counter()
for pj in glob.glob(ud + "/packs/*.json"):
    for k, v in json.load(open(pj)).items():
        lic[(os.path.basename(pj), v.get("license"))] += 1
print("PROBE pack licenses", dict(lic))
fu = [n for n in ARKIT_FACEUNITS if TargetService.target_full_path(n)]
vi = [n for n in META_VISEMES if TargetService.target_full_path(n)]
print("PROBE faceunits01 installed", FaceService.is_faceunits01_installed(force_recheck=True), len(fu), "/", len(ARKIT_FACEUNITS))
print("PROBE visemes02", len(vi), "/", len(META_VISEMES), vi)
