"""Hair helpers for base_humans.py (face / hair agent, iteration 2): the scalp CAP under every hair style.

User round-2 item 11: the card grooms showed open skin between the card clumps (refs/feedback/user_hair_gaps.png). Every
hair part (the default groom, the procedural alternates and the CC0 MakeHuman styles in parts_<kind>.glb) gets a scalp
cap: a skin-hugging shell over the scalp (scripts/hair_gen.py: rts_cap_crop / rts_cap_braid / rts_cap_generic, an MPFB
hair asset with a painted strand texture that follows the groom's comb flow, opaque roots, strand-broken hairline).
  load_caps()  loads one cap per hair object right after the parts exist, so the cap receives the same face + cust_*
               morphs and skin weights as every part (FaceService / cust interpolation through its mhclo);
  join_caps()  joins each cap into its hair object as a second primitive (material '<hair material>_cap': neutral grey
               strands x the hair-colour tint, alpha-blended), after all keys are on the parts (a joined mesh no longer
               matches its mhclo), the same way cust_lib.merge_beard_shadows joins the beard shadow shell.
Hair-colour variants (cust_lib.variant_spec) recolour the cap through baseColorFactor only (its texture is not the
strand atlas).
"""
import os
from chr_lib import *

CAPS = {"rts_crop": "rts_cap_crop", "rts_braid": "rts_cap_braid"}      # CC0 styles: rts_cap_generic
CAP_GENERIC = "rts_cap_generic"


def cap_for(style):
    return CAPS.get(style, CAP_GENERIC)


def _hair_objects(rig, kind):
    """(object, style) of every hair part: the default hair (<kind>_hair / MPFB name) and the alternates (rts_alt)."""
    out = []
    dstyle = PRESETS[kind]["parts"]["hair"].split("/")[0]
    for o in children_meshes(rig):
        if o.get("rts_alt"):
            if o.get("rts_part") == "hair":
                out.append((o, o.get("rts_style")))
        elif o.name.endswith("." + dstyle) or o.name == kind + "_hair":
            out.append((o, dstyle))
    return out


def load_caps(rig, bm, kind):
    """One cap object per hair part (MPFB mhclo asset, skinned by MPFB, face keys interpolated). Returns
    [(cap, hair_object, cap_asset)]. Call after cust_lib.add_alt_parts (before the face / cust keys are edited)."""
    from bl_ext.user_default.mpfb.services.faceservice import FaceService
    out = []
    for o, style in _hair_objects(rig, kind):
        asset = cap_for(style)
        path = asset_file("hair", asset, asset + ".mhclo")
        if not os.path.exists(path):
            log("WARNING: scalp cap %s missing (run scripts/hair_gen.py cap_crop cap_braid cap_generic)" % asset)
            continue
        c = HumanService.add_mhclo_asset(path, bm, asset_type="hair", subdiv_levels=0, material_type="MAKESKIN")
        c["rts_cap_for"] = o.name
        out.append((c, o, asset))
    FaceService.interpolate_targets(bm)                  # ARKit + visemes onto the caps (existing parts are skipped)
    log("scalp caps: %d loaded (%s)" % (len(out), ", ".join("%s<-%s" % (o.name, a) for _, o, a in out)))
    return out


CARD_ROUGH, CARD_SPEC = 0.52, 0.38        # hair cards: less of a plastic sheen than chr_lib / cust_lib's defaults


def tune_card_material(o):
    """Hair card material (first slot): roughness CARD_ROUGH, specular CARD_SPEC (the default 0.42-0.45 /
    0.5 gave a broad white plastic highlight over the whole groom in the web viewer, user item 11)."""
    m = o.data.materials[0] if o.data.materials else None
    if not m or not m.node_tree:
        return
    for b in [n for n in m.node_tree.nodes if n.type == 'BSDF_PRINCIPLED']:
        if not b.inputs["Roughness"].is_linked:
            b.inputs["Roughness"].default_value = CARD_ROUGH
        b.inputs["Specular IOR Level"].default_value = CARD_SPEC


def join_caps(rig, kind, caps):
    """Join each cap into its hair object (second material slot). Run after every key is on the parts and before the
    parts are renamed (cust_lib.name_parts), i.e. where base_humans.py calls merge_beard_shadows."""
    from cust_lib import hair_factor
    hf = hair_factor(PRESETS[kind]["hair_rgb"])
    n = 0
    for o, style in _hair_objects(rig, kind):
        tune_card_material(o)
    for c, o, asset in caps:
        if o.name not in bpy.data.objects or c.name not in bpy.data.objects:
            continue
        tex = os.path.join(TEX, "hair_cap_%s_base.png" % asset.replace("rts_cap_", ""))
        base_mat = o.data.materials[0].name if o.data.materials else "M_%s_hair" % kind
        m = pbr_material(base_mat + "_cap", base=tex, rough=0.62, alpha_mode='BLEND', alpha_img_from_base=True,
                         spec=0.35, tint=tuple(hf))
        m.use_backface_culling = True
        set_material(c, m)
        if o.data.uv_layers.active and c.data.uv_layers.active:
            c.data.uv_layers.active.name = o.data.uv_layers.active.name
        keys_o = {k.name for k in o.data.shape_keys.key_blocks} if o.data.shape_keys else set()
        keys_c = {k.name for k in c.data.shape_keys.key_blocks} if c.data.shape_keys else set()
        if keys_c and not keys_o:
            o.shape_key_add(name="Basis", from_mix=False)
        if keys_o and not keys_c:
            c.shape_key_add(name="Basis", from_mix=False)
        nv, nc = len(o.data.vertices), len(c.data.vertices)
        hid = o.hide_get()
        for ob in bpy.context.view_layer.objects:
            ob.select_set(False)
        o.hide_set(False); c.hide_set(False)
        o.select_set(True); c.select_set(True)
        bpy.context.view_layer.objects.active = o
        bpy.ops.object.join()
        o.hide_set(hid)
        assert len(o.data.vertices) == nv + nc, o.name
        got = {k.name for k in o.data.shape_keys.key_blocks} if o.data.shape_keys else set()
        assert got >= (keys_o | keys_c) - {"Basis"}, (o.name, sorted((keys_o | keys_c) - got))
        n += 1
        log("  scalp cap -> %s (%d + %d verts, %d keys, %s)" % (o.name, nv, nc, len(got) - 1, os.path.basename(tex)))
    return n


def fill_hair_cust_keys(rig, bm, kind, names, min_shift=1e-4):
    """Base v2 verification P2 ('long01 hair lacks head morphs'): MakeHuman styles such as long01 are fitted to the
    basemesh's hair HELPER, which most customisation targets leave untouched, so a wider head / bigger ears pushed the
    scalp through the hair. Every hair part (default + alternates, scalp caps excluded: they are fitted to the skin)
    gets each cust_* key it is missing (or carries with less than half the skin's motion under it) from a surface
    binding: every hair vertex follows the closest point of the body skin (triangle + barycentric of the modelled
    body), delta = the barycentric mix of the skin's key deltas. Run after cust_lib.interpolate_keys_to_parts.
    Returns {part: keys added / replaced}."""
    from mathutils.bvhtree import BVHTree
    from face_lib import modelled_coords, verts_np as _v
    NB = 13380
    co = modelled_coords(bm)
    kb = bm.data.shape_keys.key_blocks
    base = _v(bm, kb["Basis"])
    me = bm.data
    me.calc_loop_triangles()
    T = np.empty(len(me.loop_triangles) * 3, dtype=np.int64); me.loop_triangles.foreach_get("vertices", T)
    T = T.reshape(-1, 3)
    T = T[(T < NB).all(1)]
    bvh = BVHTree.FromPolygons(co[:NB].tolist(), T.tolist(), all_triangles=True)
    deltas = {k: _v(bm, kb[k]) - base for k in names if k in kb}
    rep = {}
    for o, style in _hair_objects(rig, kind):
        if o.get("rts_cap_for"):
            continue
        P = _v(o)
        tri = np.zeros(len(P), dtype=np.int64); bar = np.zeros((len(P), 3))
        for i, p in enumerate(P):
            loc, nrm, fi, d = bvh.find_nearest(Vector(p))
            tri[i] = fi
            a, b, c = co[T[fi]]
            v0, v1, v2 = b - a, c - a, np.array(loc[:]) - a
            d00, d01, d11, d20, d21 = v0 @ v0, v0 @ v1, v1 @ v1, v2 @ v0, v2 @ v1
            den = d00 * d11 - d01 * d01
            vv = (d11 * d20 - d01 * d21) / den if abs(den) > 1e-20 else 0.0
            ww = (d00 * d21 - d01 * d20) / den if abs(den) > 1e-20 else 0.0
            bar[i] = np.clip([1 - vv - ww, vv, ww], 0, 1)
        bar /= bar.sum(1)[:, None]
        if not o.data.shape_keys:
            o.shape_key_add(name="Basis", from_mix=False)
        okb = o.data.shape_keys.key_blocks
        ob = _v(o, okb["Basis"])
        done = 0
        for k, D in deltas.items():
            d = (D[T[tri]] * bar[:, :, None]).sum(1)
            d[np.linalg.norm(d, axis=1) < min_shift] = 0.0
            if not d.any():
                continue
            if k in okb:
                have = _v(o, okb[k]) - ob
                # keep the part's own (mhclo) key unless it moves the part clearly less than the skin under it
                if np.linalg.norm(have, axis=1).max() >= 0.5 * np.linalg.norm(d, axis=1).max():
                    continue
                sk = okb[k]
            else:
                sk = o.shape_key_add(name=k, from_mix=False)
            sk.data.foreach_set("co", (ob + d).ravel()); sk.value = 0.0
            done += 1
        rep[o.name] = done
    log("hair parts: cust keys from the skin under them:", rep)
    return rep
