"""Build the MALE and FEMALE base humans (MPFB 2 / MakeHuman CC0) on the rts_human skeleton, with the
face-quality layer (face_lib.py), the pose-space correctives (correctives.py) and the customisation layer
(cust_lib.py): cust_* face / body morphs, alternate parts, material variants.

Outputs per kind (male, female, or a proportion variant from cust_lib.VARIANT_PRESETS: male_stocky, male_slim, ...):
  out/base_<kind>.blend   live MPFB human: macros + detail targets as shape keys, helper geometry kept, rig (canonical
                          rest rotations, chr_lib.canonical_rest), the DEFAULT CC0 body parts only, ARKit 52 + 15 OVR
                          viseme + 4 extra expression + cust_* shape keys. Load this to fit / author MPFB clothes (mhclo)
                          and to dress the knight (the alternate parts are not saved here: they live in parts_<kind>.glb
                          and in base_<kind>_export.blend, hidden).
  out/base_<kind>.glb     engine export: modelling baked, helpers removed, 71 face morph targets (52 ARKit + 15 visemes
                          + 4 extra expression shapes) + cust_* morphs (body and every face-following part) + pose-space
                          corrective morph targets 'cor_*' on the body with their driver spec in the body mesh extras
                          ('rts_correctives'), <= 4 weights / vertex, +Y up, metres. The CC0 teeth proxy is replaced by
                          the self-built dentition of face_lib.export_face (upper set on `head`, lower set on `jaw`,
                          rigid jaw keys). KHR_materials_variants (skin tones, eye colours, hair colours) and scene
                          extras 'rts_customise' (glb_post.py).
  out/parts_<kind>.glb    the alternate parts (brows, lashes, hair styles, beards) skinned to the same skeleton, same
                          morph names, hair-colour variants.

run: BLENDER_USER_RESOURCES=characters/blender_profile Blender -b --python-exit-code 1 -P characters/scripts/base_humans.py -- [male] [female] [male_stocky ...]
     (CUST=0 in the environment skips the customisation layer)
"""
import sys, os, time, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from chr_lib import *
from cust_lib import *
import correctives
import face_lib
import glb_post
import hair_lib

CUSTOMISE = os.environ.get("CUST", "1") != "0"
args = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
kinds = [a for a in args if a in PRESETS] or ["male", "female"]
os.makedirs(OUT, exist_ok=True)

for kind in kinds:
    t0 = time.time()
    clear_scene()
    rig, bm = build_human(kind)
    make_materials(rig, kind)                                     # default parts only (before the alternates exist)
    alts = add_alt_parts(rig, bm, kind) if CUSTOMISE else []
    alt_materials(rig, kind, alts)
    # scalp cap under every hair part (hair_lib / hair_gen.py cap_*, user item 11): loaded now so it receives the same
    # face + cust keys as the parts, joined into its hair mesh below
    caps = hair_lib.load_caps(rig, bm, kind)
    # the socket-lining test reads bake_skin's albedo WITHOUT the closed-lid treatment (the keys never depend on the lid texture)
    skin_img = face_lib.lining_ref_image(kind) or \
        [nd for nd in bm.data.materials[0].node_tree.nodes if nd.type == 'TEX_IMAGE'][0].image
    fix_eye_socket_keys(bm, rig, skin_img)
    # lip seal, better smile / frown, extra shapes, lid fix; re-propagated to every part incl. the alternates
    face_lib.improve_face_keys(bm, rig, kind, skin_img)
    hair = find_part(rig, "." + PRESETS[kind]["parts"]["hair"].split("/")[0])
    if hair.name.split(".")[-1].startswith("rts_"):
        set_hair_normals(hair, bm)
    cstats, offsets = {}, {}
    if CUSTOMISE:
        cstats = add_cust_keys(bm, kind)
        # the live pipeline's eye-socket repair (iteration 1) also on the cust_* keys: eye / brow / cheek sliders move
        # the lids but not MakeHuman's red socket lining, which then shows through the skin under the eye
        fix_eye_socket_keys(bm, rig, skin_img, names=list(cstats))
        interpolate_keys_to_parts(rig, bm, list(cstats))
        hair_lib.fill_hair_cust_keys(rig, bm, kind, list(cstats))   # hair on the hair helper (long01, ...) follows the skin
        offsets = bone_offsets(rig, bm, list(cstats))
        merge_beard_shadows(rig, kind, alts)                     # card beards + their stubble-shell shadow
    hair_lib.join_caps(rig, kind, caps)                          # every hair mesh + its scalp cap (2nd primitive)
    finalize_names(rig, kind)
    name_parts(rig, kind, alts)
    z0, z1 = body_verts_z(bm)
    log(kind, "stature %.3f m" % (z1 - z0), "shape keys", len(bm.data.shape_keys.key_blocks))
    bpy.context.view_layer.objects.active = rig
    live = os.path.join(OUT, "base_%s.blend" % kind)
    bpy.ops.file.make_paths_relative()
    # the live file carries the default character only (armour authoring, outfit_lib, build_knight load it): the
    # alternates are unlinked for the save and linked back for the parts export below
    # and without the part-catalogue tags of the default parts (rts_part / rts_default / rts_style, cust_lib.name_parts):
    # in the live file rts_part marks OUTFIT pieces (outfit_lib.add_piece, build_knight.piece_objects); the GLBs keep them
    alt_objs = [o for o in children_meshes(rig) if o.get("rts_alt")]
    alt_colls = {o.name: list(o.users_collection) for o in alt_objs}
    for o in alt_objs:
        for c in alt_colls[o.name]:
            c.objects.unlink(o)
    TAGS = ("rts_part", "rts_default", "rts_style")
    tags = {o.name: {k: o[k] for k in TAGS if k in o} for o in children_meshes(rig)}
    for o in children_meshes(rig):
        for k in tags[o.name]:
            del o[k]
    bpy.ops.wm.save_as_mainfile(filepath=live, relative_remap=True)
    for o in children_meshes(rig):
        for k, v in tags[o.name].items():
            o[k] = v
    for o in alt_objs:
        for c in alt_colls[o.name]:
            c.objects.link(o)
        o.hide_render = True                       # alternates: hidden in renders (QA scripts on the export file)
    log("saved", live, "(%d alternate parts kept out of it)" % len(alt_objs))

    # ---- engine export (in place, the live file is already saved) ----
    nv = bake_body_for_export(bm)
    log("body verts after helper removal", nv)
    face_lib.export_face(rig, kind, bm, seat_cards=False)   # self-built teeth + gums (rigid jaw keys), collision
    # correctives, tongue, lip seal (the brow / lash cards are seated below, after the card-key sparsification)
    ground_export(rig, kind)                                        # proportion variants: soles to y = 0
    for o in children_meshes(rig):
        un = clean_weights(o, rig)
        if CUSTOMISE and o.get("rts_part") in CARD_PARTS:
            sparsify_card_keys(o)
        pr = prune_shape_keys(o)
        nk = len(o.data.shape_keys.key_blocks) - 1 if o.data.shape_keys else 0
        nc = sum(1 for k in o.data.shape_keys.key_blocks if k.name.startswith(CUST)) if o.data.shape_keys else 0
        log("  %-30s verts %6d unweighted %d morphs %3d (cust %2d, pruned %d)" % (o.name, len(o.data.vertices), un, nk, nc, pr))
        assert un == 0, o.name
    # brows / lashes on the skin in every face + cust key and mix (face_lib.seat_face_cards, user items 7 / 9)
    face_lib.seat_face_cards(rig, bpy.data.objects[kind + "_body"])
    # pose-space corrective shape keys + their engine driver spec (scripts/correctives.py), on the default body shape
    body = bpy.data.objects[kind + "_body"]
    correctives.build_correctives(rig, body, kind, log=log)
    # no corrective may be active in the bind pose (armour / clothes are fitted to the bind mesh)
    rest_w = drive_correctives(rig, body)
    assert max(rest_w.values() or [0]) < 1e-6, {k: v for k, v in rest_w.items() if v > 0}
    dev = rest_deviation(rig)
    log("rest-pose deviation (m)", {k: "%.1e" % v for k, v in dev.items()})
    assert max(dev.values()) < 1e-4, dev
    err = rest_rotation_error(rig)
    assert err[0] < 0.5, "rest rotations not canonical: %s" % (err,)
    base_glb = os.path.join(OUT, "base_%s.glb" % kind)
    export_glb(rig, base_glb, [o for o in children_meshes(rig) if not o.get("rts_alt")])
    # solved morph normals of the correctives (glb_patch.py), before glb_post compacts / extends the file
    log("corrective morph normals written:", correctives.export_corrective_normals(base_glb, body))
    if CUSTOMISE:
        spec = {"variants": variant_spec(rig, kind, False), "extras": extras_spec(rig, kind, cstats, offsets),
                "strip_morph_normals": CARD_MESH_RX}
        log("post", base_glb, glb_post.post(base_glb, spec))
        parts = [o for o in children_meshes(rig) if o.get("rts_alt")]
        if parts:
            parts_glb = os.path.join(OUT, "parts_%s.glb" % kind)
            export_glb(rig, parts_glb, parts)
            pspec = {"variants": variant_spec(rig, kind, True), "strip_morph_normals": CARD_MESH_RX,
                     "extras": {"version": 1, "kind": kind, "parts": parts_catalogue(rig, kind),
                                "notes": ["alternate parts for base_%s.glb: skinned to the same rts_human joints (bind by "
                                          "joint name), same face + cust_* morph names; hair colour = variants "
                                          "'hair:<colour>' or any baseColorFactor on the neutral strand texture" % kind]}}
            log("post", parts_glb, glb_post.post(parts_glb, pspec))
        json.dump({"morph_stats": cstats}, open(os.path.join(OUT, "cust_%s_stats.json" % kind), "w"), indent=1)
    for o in children_meshes(rig):
        if o.get("rts_alt"):
            o.hide_set(True)
    bpy.ops.wm.save_as_mainfile(filepath=os.path.join(OUT, "base_%s_export.blend" % kind), relative_remap=True)
    log(kind, "done in %.1fs" % (time.time() - t0))
