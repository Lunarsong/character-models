# Character pipeline source

Headless Blender (5.2) scripts that build the characters from MPFB and the CC0 MakeHuman assets, export glTF and build the
web viewers. Everything is reproducible from these scripts; no hand-edited assets.

## Setup

1. Install Blender 5.2 (the scripts expect `/Applications/Blender.app/Contents/MacOS/Blender`; change `BL=` in the shell
   scripts for other platforms).
2. `scripts/setup.sh` (run once) downloads MPFB 2.0.17 and the CC0 MakeHuman asset packs, installs MPFB into a
   **project-local** Blender profile (`blender_profile/`, via `BLENDER_USER_RESOURCES`) so your own Blender setup is never
   touched, and checks that MPFB loads headless.
3. Node.js plus [gltf-validator](https://github.com/KhronosGroup/glTF-Validator) for GLB validation; Google Chrome for the
   headless viewer tests and screenshots.

The scripts expect this folder layout (as in the working repo): `scripts/`, `viewer/`, `assets/`, `out/`, `renders/`,
`refs/`, `blender_profile/` side by side.

## Build

- `scripts/build_all.sh [male] [female]`: the base humans end to end: rig (`make_rig.py`, the shared rts_human skeleton with
  face, twist and helper bones) -> hair and skin textures -> humans (`base_humans.py`; face layer `face_lib.py`,
  customisation `cust_lib.py`, pose-space correctives `correctives.py`) -> validation -> QA renders -> viewers.
- `scripts/mkviewer_char.py` builds the character page from `viewer/char_viewer_template.html`;
  `viewer/make_viewer.py` builds the look-dev page from `viewer/lookdev_template.html`.
- `scripts/viewer_check.py`, `morph_check.py`, `face_qa.py`, `pose_qa.py`, `deform_test.py`: QA (expressions reach every mesh,
  morph targets, face quality, foot-floor contact and joint angles per pose, deformation stress poses).
- `scripts/hand_poses.py` + `viewer/hand_poses.js`: the hand-grip preset library (the same data drives Blender and the viewer).
- Outfits and armour (work in progress): `outfit_*.py`, `armour_upper*.py`, `armour_lower*.py`, `build_knight.py`,
  `knight_*.py`, `kit_*.py`, and the integrity gates in `armour_integrity.py`.

## Rig and face

- One 97-bone skeleton shared by every body and outfit (UE5-style names, face bones, twist and helper bones); male and female
  share identical rest rotations so animations transfer.
- Face: ARKit 52 blendshapes + 15 OVR visemes, customisation morphs (`cust_*`), pose-space correctives driven by the skeleton.
