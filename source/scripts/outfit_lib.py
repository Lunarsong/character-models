"""Outfit (clothes / armour) helpers on top of chr_lib for the modular wardrobe.

Every outfit piece is an MPFB clothes asset (.mhclo + .obj [+ .mhmat, optional .mhw weights]) authored against the base
body with MakeClothes (headless: ClothesService.create_mhclo_from_clothes_matching, see hair_gen.py). Loaded onto a LIVE
base human (out/base_<kind>.blend) with add_piece(), a piece
  - fits the body (mhclo vertex correspondence: 3 body verts + weights + offset per piece vertex),
  - is skinned to the rts_human skeleton (MPFB interpolates the body weights, or loads the piece's .mhw),
  - hides what it covers: MPFB puts the mhclo 'delete_verts' in a body group 'Delete.<piece>' + a MASK modifier,
  - follows the cust_* customisation morphs and the face morphs that move it (same correspondence as the face parts).
export_dressed() runs the base_humans.py engine export on the dressed human and removes the covered body vertices for
real (the body under the armour is deleted, not just masked), then writes one validated GLB.

Python (inside Blender):  from outfit_lib import *; rig, bm = live_human("male"); add_piece(rig, bm, path, "torso")
Smoke test: scripts/outfit_smoke.py.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from chr_lib import *
from bl_ext.user_default.mpfb.services import ClothesService
from bl_ext.user_default.mpfb.entities.clothes.mhclo import Mhclo


def live_human(kind):
    """(rig, body) of the live MPFB human in the open file (out/base_<kind>.blend)."""
    rig = bpy.data.objects["rts_" + kind]
    return rig, bpy.data.objects[kind + "_body"]


def piece_keys(bm, piece, names, eps=1e-5):
    """Body shape keys `names` -> `piece` through its mhclo correspondence (keys that do not move it are skipped)."""
    m = Mhclo(); m.load(ClothesService.find_clothes_absolute_path(piece))
    kb = bm.data.shape_keys.key_blocks
    n = len(bm.data.vertices)
    base = np.empty(n * 3); kb["Basis"].data.foreach_get("co", base); base = base.reshape(-1, 3)
    cn = len(piece.data.vertices)
    idx = np.array([m.verts[i]["verts"] for i in range(cn)], dtype=np.int64)
    w = np.array([m.verts[i]["weights"] for i in range(cn)])
    if not piece.data.shape_keys:
        piece.shape_key_add(name="Basis", from_mix=False)
    pk = piece.data.shape_keys.key_blocks
    cb = np.empty(cn * 3); pk["Basis"].data.foreach_get("co", cb); cb = cb.reshape(-1, 3)
    added = []
    for nm in names:
        if nm not in kb or nm in pk:
            continue
        a = np.empty(n * 3); kb[nm].data.foreach_get("co", a)
        off = ((a.reshape(-1, 3) - base)[idx] * w[..., None]).sum(1)
        if np.abs(off).max() > eps:
            sk = piece.shape_key_add(name=nm, from_mix=False)
            sk.data.foreach_set("co", (cb + off).ravel()); sk.value = 0.0
            added.append(nm)
    return added


def mhmat_material(name, mhclo_path, rough=0.7, metallic=0.0):
    """glTF-friendly PBR material from a piece's .mhmat textures (diffuseTexture / normalmapTexture)."""
    d = os.path.dirname(mhclo_path)
    mats = [f for f in os.listdir(d) if f.endswith(".mhmat")]
    tex = {}
    if mats:
        for line in open(os.path.join(d, mats[0])):
            p = line.split()
            if len(p) == 2 and p[0] in ("diffuseTexture", "normalmapTexture"):
                f = os.path.join(d, os.path.basename(p[1]))
                if os.path.exists(f):
                    tex[p[0]] = f
    return pbr_material(name, base=tex.get("diffuseTexture"), normal=tex.get("normalmapTexture"),
                        rough=rough, metallic=metallic, spec=0.4)


def add_piece(rig, bm, mhclo_path, slot=None, keys=None, material=None):
    """Load one outfit piece onto the live human. Returns the piece object, named '<kind>_<slot>' with node extras
    rts_part = slot (engine: one mesh per slot, bind by joint name to the shared skeleton)."""
    kind = rig.get("rts_kind") or rig.name.replace("rts_", "")
    slot = slot or os.path.splitext(os.path.basename(mhclo_path))[0]
    pc = HumanService.add_mhclo_asset(mhclo_path, bm, asset_type="Clothes", subdiv_levels=0, material_type="MAKESKIN")
    assert pc.parent == rig and any(md.type == 'ARMATURE' and md.object == rig for md in pc.modifiers), pc.name
    pc.name = pc.data.name = "%s_%s" % (kind, slot)
    pc["rts_part"] = slot
    pc["rts_asset"] = os.path.splitext(os.path.basename(mhclo_path))[0]
    if keys is None:                     # every face + customisation key of the body (cust_lib names, iteration 2)
        keys = [k.name for k in bm.data.shape_keys.key_blocks if k.name in FACE_KEYS or k.name.startswith("cust_")]
    piece_keys(bm, pc, keys)
    set_material(pc, material or mhmat_material("M_%s_%s" % (kind, slot), mhclo_path))
    return pc


def covered_verts(bm, thresh=0.5):
    """Body vertex indices hidden by any piece (MPFB 'Delete.<piece>' groups), plus the per-group counts."""
    gi = {g.index: g.name for g in bm.vertex_groups if g.name.startswith("Delete")}
    hide = set(); counts = {nm: 0 for nm in gi.values()}
    for v in bm.data.vertices:
        for g in v.groups:
            if g.group in gi and g.weight > thresh:
                hide.add(v.index); counts[gi[g.group]] += 1
    return sorted(hide), counts


def export_mouth(rig, kind, bm):
    """Export-stage mouth of the base pipeline (iteration 2 reconcile): the CC0 teeth proxy of the live file is the
    guide for face_lib's self-built dentition (28 crowns + gums, upper on head / lower on jaw, rigid jaw keys, cust
    keys), plus the lip collision correctives and the refitted tongue. Call after bake_body_for_export."""
    import face_lib
    import chr_lib
    for nm in ("smooth_scalp",):                     # face_lib mid-edit (06:30) used chr_lib names it did not import
        if not hasattr(face_lib, nm) and hasattr(chr_lib, nm):
            setattr(face_lib, nm, getattr(chr_lib, nm))
    teeth = find_part(rig, "_teeth")
    if teeth is None:
        return None
    lips = bm.vertex_groups.get("lips")
    nl = sum(1 for v in bm.data.vertices if lips and any(g.group == lips.index and g.weight > 0.3 for g in v.groups))
    if nl < 50:                          # face skin deleted (closed helm): keep a light proxy, nobody sees the mouth
        if len(teeth.data.vertices) > 3000:
            decimate_with_keys(teeth, 0.3)
        return None
    return face_lib.export_face(rig, kind, bm)


def export_dressed(rig, bm, kind, path, drop=(), correctives=True):
    """Engine export of the dressed live human, in place (save the live .blend first if you need it):
    covered body verts deleted, modelling baked, <= 4 weights, face / cust / cor_* morphs, validated separately with
    check_glb.py. drop: roles to leave out, e.g. ("hair", "eyebrows_alt") under a closed helm. Alternate parts
    (cust_lib 'rts_alt', iteration-1 'rts_variant_group' non-defaults) are dropped when '<role>_alt' is in drop."""
    hide, counts = covered_verts(bm)
    bm.vertex_groups["body"].remove(hide)            # bake_body_for_export keeps only 'body' verts: covered skin goes
    for o in list(children_meshes(rig)):
        role = o.name[len(kind) + 1:]
        alt = o.get("rts_alt") or (o.get("rts_variant_group") and not o.get("rts_default"))
        if role in drop or (alt and role.split("_")[0] + "_alt" in drop):
            bpy.data.objects.remove(o, do_unlink=True)
    nv = bake_body_for_export(bm)
    log("dressed body: %d verts kept, %d covered removed %s" % (nv, len(hide), counts))
    export_mouth(rig, kind, bm)
    for o in children_meshes(rig):
        un = clean_weights(o, rig)
        prune_shape_keys(o)
        assert un == 0, "%s: %d unweighted verts" % (o.name, un)
    if correctives:                      # the base pipeline's cor_* system (scripts/correctives.py), on the kept skin
        import correctives as COR
        COR.build_correctives(rig, bm, kind, log=log)
    dev = rest_deviation(rig)
    assert max(dev.values()) < 1e-4, dev
    export_glb(rig, path)
    if correctives:
        import correctives as COR
        log("corrective morph normals written:", COR.export_corrective_normals(path, bm))
    return dict(body_verts=nv, covered=len(hide), meshes=[o.name for o in children_meshes(rig)])
