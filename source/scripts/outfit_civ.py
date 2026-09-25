"""CIVILIAN outfits: dress, animate, export (dressed GLB + one GLB per piece + manifest) and QA, on the live base humans.

Outfits are lists of MPFB clothes assets (authored by outfit_peasant.py / outfit_archer.py on the male), so the same
pieces mix and match: the archer re-uses the peasant's shirt and trousers; 'peasant_hood' swaps the straw hat for the
archer's hood.

run (see scripts/outfit_civ.sh):
  Blender -b out/base_<kind>.blend --python-exit-code 1 -P scripts/outfit_civ.py -- dress <kind> <outfit>
      -> out/civ/<outfit>_<kind>.blend (live MPFB human, dressed, sockets, clips), out/civ/<outfit>_<kind>.glb (engine:
         body with the covered skin deleted + pieces + clips), out/civ/<outfit>_<kind>_export.blend (= the GLB),
         out/civ/pieces/<kind>/<asset>.glb (each piece alone on the shared skeleton, not culled), manifest entries
  Blender -b out/civ/<outfit>_<kind>_export.blend --python-exit-code 1 -P scripts/outfit_civ.py -- qa <kind> <outfit>
      -> penetration per stress pose and clip frame (renders/civ/qa_<outfit>_<kind>.json) + highlight renders
"""
import sys, os, json, math, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from outfit_civ_lib import *

OUTFITS = {
    "peasant": dict(pieces=["rts_peasant_shirt", "rts_peasant_trousers", "rts_peasant_shoes", "rts_peasant_tunic",
                            "rts_peasant_belt", "rts_peasant_hat"], props=["hoe"], clips=["idle", "walk", "run", "work"]),
    "archer": dict(pieces=["rts_peasant_shirt", "rts_peasant_trousers", "rts_archer_boots", "rts_archer_gambeson",
                           "rts_archer_belt", "rts_archer_hood", "rts_archer_bracers", "rts_archer_gloves",
                           "rts_archer_quiver"], props=["longbow"], clips=["idle", "walk", "run", "shoot"]),
    # mix-and-match proof: the peasant with the archer's hood instead of the straw hat
    "peasant_hood": dict(pieces=["rts_peasant_shirt", "rts_peasant_trousers", "rts_peasant_shoes", "rts_peasant_tunic",
                                 "rts_peasant_belt", "rts_archer_hood"], props=[], clips=["idle", "walk"]),
}
MANIFEST = os.path.join(OUTC, "civ_outfits.json")
KIT = os.path.join(OUT, "kit")
# kit contract for the civ pieces (same fields as kit_build.KNIGHT_SLOTS): slot, layer (0 = on the skin), slots it also
# occupies (conflicts with the knight's), pieces it requires, body parts it hides
CIV_KIT = {
    "peasant_shirt": ("shirt", 0, ("torso_inner",), (), ()),
    "peasant_trousers": ("trousers", 0, ("legs_inner",), (), ()),
    "peasant_shoes": ("shoes", 1, ("feet", "feet_inner"), (), ()),
    "peasant_tunic": ("tunic", 2, ("torso_outer", "hips"), (), ()),
    "peasant_belt": ("belt", 3, (), (), ()),
    "peasant_hat": ("hat", 3, ("head",), (), ()),
    "archer_gambeson": ("gambeson", 1, ("torso", "hips"), (), ()),
    "archer_belt": ("belt", 3, (), (), ()),
    "archer_hood": ("hood", 3, ("head", "neck"), (), ("hair",)),
    "archer_bracers": ("forearms", 2, ("forearm_l", "forearm_r"), (), ()),
    "archer_gloves": ("hands", 0, ("hand_l", "hand_r"), (), ()),
    "archer_boots": ("boots", 1, ("feet", "feet_inner", "shins"), (), ()),
    "archer_quiver": ("back", 4, (), (), ()),
    "archer_longbow": ("weapon_l", 5, ("shield_l",), (), ()),
    "archer_arrow": ("arrow_nocked", 5, (), ("archer_longbow",), ()),
    "peasant_hoe": ("weapon_r", 5, (), (), ()),
}
PROP_ASSET = {"longbow": "rts_archer_longbow", "arrow": "rts_archer_arrow", "hoe": "rts_peasant_hoe"}


def kit_entries(rig, bm, kind, objs, props):
    """kit piece records (kit_build.pieces schema) for the civ pieces on the LIVE dressed body: region coverage from
    each piece's MPFB delete group (bitsets over kit_build.region_verts, the same lists as the knight's pieces)"""
    import kit_build as KB
    core = set(json.load(open(RIG_JSON))["bones"])
    regions = KB.region_verts(rig, bm)
    bone_parent = {b.name: (b.parent.name if b.parent else None) for b in rig.data.bones}
    out = {}
    for base, o in list(objs.items()) + [(PROP_ASSET[k], v) for k, v in props.items()]:
        pid = base.replace("rts_", "", 1)
        slot, layer, occ, req, hparts = CIV_KIT.get(pid, (pid, 3, (), (), ()))
        asset = o.get("rts_asset", "")
        cov = KB.group_verts(bm, "Delete.rts_" + pid) if not o.get("rts_prop") else np.zeros(0, dtype=np.int64)
        frac, bits = KB.coverage(regions, cov)
        groups = {g.name for g in o.vertex_groups}
        joints = sorted(g for g in groups if g in bone_parent)
        extra = sorted({j for j in joints if j not in core})
        for j in list(extra):
            pp = bone_parent.get(j)
            while pp and pp not in core:
                extra.append(pp); pp = bone_parent.get(pp)
        keys = [k.name for k in o.data.shape_keys.key_blocks[1:] if k.name.startswith("cust_")] if o.data.shape_keys else []
        out[pid] = dict(slot=slot, occupies=list(occ), layer=layer, set="civilian_" + pid.split("_")[0], requires=list(req),
                        asset=asset, tris=sum(len(p.vertices) - 2 for p in o.data.polygons), verts=len(o.data.vertices),
                        materials=[m.name for m in o.data.materials if m], cust_morphs=len(keys), joints=joints,
                        extra_joints=sorted(set(extra)), socket=o.get("rts_socket"), prop=bool(o.get("rts_prop")),
                        covers=frac, cover_bits=bits, hides=sorted(r for r, f in frac.items() if f >= KB.HIDE_T),
                        hides_parts=list(hparts), file="pieces_civ/%s.glb" % pid)
    return out, {r: int(len(v)) for r, v in regions.items()}


def installed(asset):
    return os.path.exists(asset_file("clothes", asset, asset + ".mhclo"))


def dress(kind, outfit):
    """load the outfit's pieces onto the live human in the open file (out/base_<kind>.blend)"""
    rig = bpy.data.objects["rts_" + kind]; bm = bpy.data.objects[kind + "_body"]
    pose_reset(rig)
    objs, metas = {}, {}
    for base in OUTFITS[outfit]["pieces"]:
        a = asset_for(base, kind)                       # the body's own authored variant, else the male one
        if not installed(a):
            clog("MISSING asset", a); continue
        ob, meta = dress_piece(rig, bm, a)
        meta["piece_id"] = base
        objs[base], metas[base] = ob, meta
        clog("dressed %-24s verts %5d tris %5d cust morphs %d" % (
            a, len(ob.data.vertices), sum(len(p.vertices) - 2 for p in ob.data.polygons),
            len(ob.data.shape_keys.key_blocks) - 1 if ob.data.shape_keys else 0))
    hood = any(m["hides_hair"] for m in metas.values())
    return rig, bm, objs, metas, hood


def add_props(rig, kind, outfit):
    """civ sockets (canonical frames) + the outfit's props on their bones (the longbow and its nocked arrow)"""
    extra = {}
    if "longbow" in OUTFITS[outfit]["props"]:
        import outfit_civ_bow as BOW
        extra.update(BOW.bow_specs(rig)[0])
    if "hoe" in OUTFITS[outfit]["props"]:
        import outfit_civ_props as PR
        extra.update(PR.hoe_specs(rig))
    ensure_sockets(rig, extra)
    props = {}
    if "longbow" in OUTFITS[outfit]["props"]:
        import outfit_civ_bow as BOW
        props["longbow"], props["arrow"] = BOW.add_bow(rig, kind)
    if "hoe" in OUTFITS[outfit]["props"]:
        import outfit_civ_props as PR
        props["hoe"] = PR.add_hoe(rig, kind)
    return props


def export_dressed_civ(rig, bm, kind, path, drop=(), animations=True, cull=True):
    """outfit_lib.export_dressed's sequence (covered skin deleted, modelling baked, dentition, <= 4 weights, cor_*
    correctives on the kept skin) + the clips (glTF animations) and the civ piece extras; cull: inner-layer faces
    that outer layers cover in every test pose / clip frame are deleted (LOD0 of the baked outfit)"""
    import outfit_lib as OL
    hide, counts = OL.covered_verts(bm)
    bm.vertex_groups["body"].remove(hide)
    for o in list(children_meshes(rig)):
        role = o.name[len(kind) + 1:]
        alt = o.get("rts_alt") or (o.get("rts_variant_group") and not o.get("rts_default"))
        if role in drop or (alt and role.split("_")[0] + "_alt" in drop):
            bpy.data.objects.remove(o, do_unlink=True)
    nv = bake_body_for_export(bm)
    clog("dressed body: %d verts kept, %d covered removed" % (nv, len(hide)))
    try:
        OL.export_mouth(rig, kind, bm)
        mouth = "dentition"
    except Exception as e:                       # face_lib is edited by the face agent: keep the proxy teeth, say so
        import traceback
        traceback.print_exc()
        clog("WARNING export_mouth failed (%r): the CC0 proxy teeth are kept in this export" % e)
        mouth = "proxy (export_mouth failed)"
    for o in children_meshes(rig):
        un = clean_weights(o, rig)
        prune_shape_keys(o)
        assert un == 0, "%s: %d unweighted verts" % (o.name, un)
    # no body cor_* correctives: the shoulder / hip regions they reshape are under the garments (deleted); solved on
    # the few kept skin vertices near the collar they produced 10 cm pushes (the reconcile's knight has the same
    # caveat). Garment-level fixes live in the garments' own weights / layer lines.
    culled = {}
    if cull:
        import outfit_civ_qa as QA
        import outfit_civ_poses as PZ
        pcs = {o.name[len(kind) + 1:]: o for o in children_meshes(rig) if o.get("rts_outfit") and not o.get("rts_prop")}
        lay = {n: int(o.get("rts_layer", 50)) for n, o in pcs.items()}
        clips = list(json.loads(rig.get("rts_clips", "{}")))
        culled = QA.cull_hidden_layers(rig, pcs, lay, poses=PZ.STRESS + PZ.EXTRA, clips=clips)
    # the stress poses put finger bones in Euler mode (curl_fingers): back to quaternions, or the clips' finger keys
    # (rotation_quaternion) would be ignored by the exporter's sampling and the saved export file (open hands)
    for pb in rig.pose.bones:
        pb.rotation_mode = 'QUATERNION'
    pose_reset(rig)
    dev = rest_deviation(rig)
    assert max(dev.values()) < 1e-4, dev
    gltf_export(rig, children_meshes(rig), path, animations=animations)
    return dict(body_verts=nv, covered=len(hide), mouth=mouth, culled=culled)


def fix_tangents(path):
    """GLB hygiene (own copy of the kit's kit_glb.fix_tangents): zero / non-unit TANGENT vectors (a few degenerate UV
    triangles on the belt's pouch / tube caps; validator ACCESSOR_VECTOR3_NON_UNIT) are replaced in place by a unit
    vector perpendicular to the vertex normal. Returns the number repaired."""
    import struct
    b = bytearray(open(path, "rb").read())
    jl = struct.unpack_from("<I", b, 12)[0]
    js = json.loads(bytes(b[20:20 + jl]))
    bin0 = 20 + jl + 8
    fixed = 0
    CTN = {5126: (np.float32, 4)}
    for m in js.get("meshes", []):
        for pr in m["primitives"]:
            a = pr.get("attributes", {})
            if "TANGENT" not in a:
                continue
            acc = js["accessors"][a["TANGENT"]]; bv = js["bufferViews"][acc["bufferView"]]
            st = bv.get("byteStride", 16); base = bin0 + bv.get("byteOffset", 0) + acc.get("byteOffset", 0)
            nacc = js["accessors"][a["NORMAL"]]; nbv = js["bufferViews"][nacc["bufferView"]]
            nst = nbv.get("byteStride", 12); nbase = bin0 + nbv.get("byteOffset", 0) + nacc.get("byteOffset", 0)
            for i in range(acc["count"]):
                t = np.array(struct.unpack_from("<4f", b, base + i * st))
                L = np.linalg.norm(t[:3])
                if np.isfinite(L) and abs(L - 1.0) < 5e-4:
                    continue
                n = np.array(struct.unpack_from("<3f", b, nbase + i * nst)); n /= max(np.linalg.norm(n), 1e-9)
                tt = t[:3] - n * t[:3].dot(n) if np.isfinite(t).all() and L > 1e-6 else np.zeros(3)
                if np.linalg.norm(tt) < 1e-6:
                    ref = np.array([1.0, 0, 0]) if abs(n[0]) < 0.9 else np.array([0, 0, 1.0])
                    tt = ref - n * ref.dot(n)
                tt /= np.linalg.norm(tt)
                struct.pack_into("<4f", b, base + i * st, *tt, 1.0 if (t[3] >= 0 or not np.isfinite(t[3])) else -1.0)
                fixed += 1
    if fixed:
        open(path, "wb").write(bytes(b))
    return fixed


def gltf_export(rig, meshes, path, animations=True):
    for o in bpy.context.view_layer.objects:
        o.select_set(False)
    for o in meshes:
        assert o.matrix_world == Matrix.Identity(4), o.name
        o.parent = None
        o.select_set(True)
    rig.select_set(True)
    bpy.context.view_layer.objects.active = rig
    act = rig.animation_data.action if rig.animation_data else None
    if rig.animation_data:
        rig.animation_data.action = None
    pose_reset(rig)
    bpy.ops.export_scene.gltf(
        filepath=path, export_format='GLB', use_selection=True, export_yup=True, export_apply=False,
        export_skins=True, export_all_influences=False, export_morph=True, export_morph_normal=True,
        export_morph_tangent=False, export_try_sparse_sk=True, export_tangents=True, export_image_format='AUTO',
        export_materials='EXPORT', export_extras=True, export_def_bones=False, export_rest_position_armature=True,
        export_leaf_bone=False, export_animations=animations, export_animation_mode='ACTIONS',
        export_force_sampling=True, export_frame_step=1, export_anim_single_armature=True,
        export_reset_pose_bones=True, export_optimize_animation_size=True, export_morph_animation=False,
        export_bake_animation=False, export_merge_animation='ACTION', export_anim_slide_to_zero=True)
    for o in meshes:
        o.parent = rig
        o.matrix_parent_inverse = Matrix.Identity(4)
    if rig.animation_data:
        rig.animation_data.action = act
    nt = fix_tangents(path)
    clog("glb", path, "%.2f MB" % (os.path.getsize(path) / 1e6) + (" (%d degenerate tangents repaired)" % nt if nt else ""))


def export_pieces(rig, kind, objs, metas, outdir):
    """each piece alone on the shared skeleton (97 joints + sockets), not culled against anything: the runtime
    wardrobe. Node extras carry the piece contract (outfit, layer, hidden body regions, hides hair)."""
    os.makedirs(outdir, exist_ok=True)
    out = {}
    for a, o in objs.items():
        if o.name not in bpy.data.objects:
            continue
        path = os.path.join(outdir, a + ".glb")
        gltf_export(rig, [o], path, animations=False)
        out[a] = os.path.relpath(path, CH)
    return out


def write_manifest(kind, outfit, glb, pieces, metas, props, sockets, clips, tris, kit=None, info=None):
    man = json.load(open(MANIFEST)) if os.path.exists(MANIFEST) else {"pieces": {}, "outfits": {}}
    man["skeleton"] = ("rts_human 97 core joints (canonical rest rotations, M15) + civ sockets (non-deform, canonical "
                       "frames) + prop bones (bow_*, prop_hoe) under socket_hand_l / socket_weapon_r")
    man["sockets"] = sockets
    for a, m in metas.items():
        e = man["pieces"].setdefault(a, {})
        e.update({k: m[k] for k in ("outfit", "slot", "desc", "notes", "layer", "hides_hair", "hide_regions", "sets", "tris")})
        e["hides_basemesh_verts"] = m["delete_basemesh_verts"]
        e.setdefault("glb", {})[kind] = pieces.get(a)
        e.setdefault("asset", {})[kind] = m.get("asset")
        if kit and a.replace("rts_", "", 1) in kit:
            k_ = kit[a.replace("rts_", "", 1)]
            e.setdefault("kit", {})[kind] = {x: k_[x] for x in ("slot", "occupies", "layer", "requires", "covers",
                                                                 "hides", "hides_parts", "joints", "extra_joints")}
    for pk, pv in PROP_ASSET.items():
        if pk in props:
            e = man["pieces"].setdefault(pv, {"outfit": pv.split("_")[1], "slot": pk, "prop": True})
            e.setdefault("glb", {})[kind] = pieces.get(pv)
            if kit and pv.replace("rts_", "", 1) in kit:
                k_ = kit[pv.replace("rts_", "", 1)]
                e.setdefault("kit", {})[kind] = {x: k_[x] for x in ("slot", "layer", "requires", "joints", "extra_joints",
                                                                     "socket")}
    man["outfits"].setdefault(outfit, {})
    man["outfits"][outfit].update({"pieces": OUTFITS[outfit]["pieces"], "props": list(props),
                                   "clips": clips, "layer_order": sorted(metas, key=lambda a: metas[a]["layer"])})
    man["outfits"][outfit].setdefault("glb", {})[kind] = os.path.relpath(glb, CH)
    man["outfits"][outfit].setdefault("tris", {})[kind] = tris
    if info:
        man["outfits"][outfit].setdefault("baked_lod0", {})[kind] = {"culled_faces": info.get("culled", {}),
                                                                   "skin_verts_removed": info.get("covered"),
                                                                   "mouth": info.get("mouth")}
    man["notes"] = ("Body garments grow from MakeHuman's helper-tights on shared normal lines (layers keep their order "
                    "under skinning); hides_basemesh_verts are MakeHuman basemesh vertex indices (the same on every "
                    "body); layer = z-depth (inner < outer); pieces carry cust_* body morphs only.")
    json.dump(man, open(MANIFEST, "w"), indent=1)


def main(args):
    mode = args[0]
    kind = args[1] if len(args) > 1 else "male"
    outfit = args[2] if len(args) > 2 else "peasant"
    os.makedirs(OUTC, exist_ok=True)
    t0 = time.time()
    if mode == "dress":
        rig, bm, objs, metas, hood = dress(kind, outfit)
        props = add_props(rig, kind, outfit)
        clips = []
        if "--noclips" not in args:
            import outfit_civ_anim as AN
            clips = AN.make_clips(rig, kind, outfit, OUTFITS[outfit]["clips"])
        live = os.path.join(OUTC, "%s_%s.blend" % (outfit, kind))
        bpy.ops.wm.save_as_mainfile(filepath=live)
        clog("saved", live)
        # the runtime wardrobe first: every piece whole (not culled), <= 4 weights, on the shared skeleton
        for o in list(objs.values()) + [p for p in props.values()]:
            clean_weights(o, rig); prune_shape_keys(o)
        kit, regions = kit_entries(rig, bm, kind, objs, props)
        pieces = export_pieces(rig, kind, {**objs, **{PROP_ASSET[k]: v for k, v in props.items()}}, metas,
                               os.path.join(OUTC, "pieces", kind))
        glb = os.path.join(OUTC, "%s_%s.glb" % (outfit, kind))
        drop = ("hair", "eyebrows_alt") if hood else ()
        info = export_dressed_civ(rig, bm, kind, glb, drop=drop)
        exp = os.path.join(OUTC, "%s_%s_export.blend" % (outfit, kind))
        bpy.ops.wm.save_as_mainfile(filepath=exp)
        tris = sum(sum(len(p.vertices) - 2 for p in o.data.polygons) for o in children_meshes(rig))
        sockets = {b.name: b.parent.name for b in rig.data.bones if b.name.startswith("socket_")}
        write_manifest(kind, outfit, glb, pieces, metas, props, sockets, clips, tris, kit=kit, info=info)
        clog("%s %s: %d meshes, %d tris, clips %s (%.0fs)" % (outfit, kind, len(children_meshes(rig)), tris, clips,
                                                            time.time() - t0))
    elif mode == "qa":
        import outfit_civ_qa as QA
        QA.qa_export(kind, outfit, OUTFITS[outfit], renders="--norender" not in args)
    clog("done %s %s %s in %.1fs" % (mode, kind, outfit, time.time() - t0))


if __name__ == "__main__":
    main(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else [])
