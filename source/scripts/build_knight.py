#!/usr/bin/env python3
"""KNIGHT assembly: one command rebuilds the dressed, animated knight (male + female) from the base humans, the upper
and lower armour scripts and the texture sets.

    cd RTS_Models/characters
    python3 scripts/build_knight.py                       # everything (~25 min): textures -> upper + lower armour
                                                          # assets -> assemble + animate + export (male, female) ->
                                                          # validate + report -> web copies + viewers -> QA renders
    python3 scripts/build_knight.py assemble check web    # just some stages (default kinds: male female)
    python3 scripts/build_knight.py assemble --kinds male
Stages (in order): tex upper lower assemble check kit lod web qa shots compare integrity
  tex       textures_char.py (shared PBR sets) + armour_upper_texgen.py (upper trim sheet composited from them)
  upper     armour_upper.py author  (MPFB clothes assets knight_<slot>, ~30 s)
  lower     armour_lower.py author + armour_lower_tex.py (rts_knight_<slot> assets drape over the upper ones, ~9 min)
  assemble  Blender on out/base_<kind>.blend: upper + lower pieces, extra bones, sockets, props, texture sets on every
            piece, fit report, animations (knight_anim.py) -> out/knight_<kind>.blend (live MPFB human),
            out/knight_<kind>.glb (engine GLB) + out/knight_<kind>_export.blend
  check     Khronos validator (0 errors required) + skin / morph limits (check_glb.py) + zero corr_* normal deltas
            + out/knight_report.json (tris per piece, bones, morphs, texture memory)
  kit       MODULAR KIT (judge C4): every piece alone on the shared skeleton (kit_build.py pieces, not culled, + the
            civilian outfits), the clips (kit_build.py anims), the region-split body + shared textures + manifest +
            validation (kit_manifest.py) -> out/kit/; dress-up pages viewer/kit_<kind>_dressup.html + screenshots
  lod       GAME LODs (judge M16): LOD0 / LOD1 / LOD2 with merged atlases, face joints stripped (knight_lod.py game)
            -> out/game/; cutscene face data separate (knight_lod.py face) -> out/cutscene/; validated
  web       1k-texture web copies (glb_web.py) + viewer/knight_<kind>_lookdev.html (viewer/make_viewer.py --town)
  qa        Blender renders: turnaround, close-ups, clip mid-poses, poke-through / exposed-skin test, cape clearance
  shots     viewer screenshots (viewer/shot.sh): views, face expressions (helm off), RTS camera, clips
  compare   side-by-side sheets with refs/knight_sheet.png
  integrity ARMOUR INTEGRITY gates G1-G8 (scripts/armour_integrity.py, user round-5 items 23-32): open / one-sided
            shells, floating parts, see-through seams, visible penetration, grip, texture stretch, head in helmet, the
            user's views + an orbit battery in the viewer -> renders/integrity/ (<gate>_<kind>.json, sheets,
            summary.json, defects.json with owners)

Inside Blender (run by the stages above):
    Blender -b out/base_<kind>.blend --python-exit-code 1 -P scripts/build_knight.py -- assemble <kind>
    Blender -b out/knight_<kind>_export.blend --python-exit-code 1 -P scripts/build_knight.py -- qa <kind> [what ..]
"""
import sys, os, json, time, math, subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
CH = os.path.dirname(HERE)
OUTD = os.path.join(CH, "out")
# the Blender side of 'assemble' writes into KNIGHT_OUTD when set: the orchestrator points it at out_stage/ (a sibling of
# out/, so relative texture paths saved in the .blends stay valid) and then installs each file with os.replace, so no
# parallel reader ever sees a half-written out/knight_* file
if _outd := os.environ.get("KNIGHT_OUTD"):
    OUTD = _outd
STAGE_DIR = os.path.join(CH, "out_stage")
BL = "/Applications/Blender.app/Contents/MacOS/Blender"
STAGES = ["tex", "upper", "lower", "assemble", "check", "kit", "lod", "web", "qa", "shots", "compare", "integrity"]
CULL = os.environ.get("KNIGHT_CULL", "1") != "0"
# tileable sets that are re-embedded at 1k in the knight GLB (texel density stays >= 2000 px/m, the GPU cost drops
# 4x); the unique atlases (trim sheet, tabard, cape, shield face, plume, skin) stay 2k
TEX_1K = ("gold_worn", "steel_worn", "steel_blued", "leather_straps", "leather_brown", "mail_riveted", "kl_mail_opaque")
TEX_1K_DIR = os.path.join(CH, "assets", "textures", "_1k")     # KNIGHT_CULL=0: keep every face (whole pieces, for mix-and-match tests)
# props dressed on the knight: sword in hand + shield on the forearm. The empty scabbard (a 0.92 m stick hanging tip-back
# from the hip) poked through the full-width cape and is not on the reference sheet: it stays a stand-alone prop
# (out/props/knight_scabbard.glb) for socket_scabbard_l / socket_sword_sheathed, which the knight keeps.
KNIGHT_PROPS = tuple(os.environ.get("KNIGHT_PROPS", "sword shield").split())

try:
    import bpy  # noqa: F401
    IN_BLENDER = True
except ImportError:
    IN_BLENDER = False

# The helmet-on look (game) hides the head; the bare-head look (cutscenes, facial animation) swaps the helmet + plume
# for the head skin under it, hair, brows, lashes, eyes, teeth and tongue. Node extras: rts_look = "helm" | "bare".
HELM_SLOTS = ("helmet", "plume", "aventail", "gorget_top")
HEAD_BONES = ("head", "jaw", "eye_l", "eye_r")
HEAD_PREFIX = ("eyelid_", "brow_", "cheek_", "nose_", "mouth_corner_", "lip_", "tongue_")
BARE_ROLES = ("head", "hair", "eyebrows", "eyelashes", "eyes", "teeth", "tongue")
# viewer / engine grouping of the pieces (node extras 'slot')
SLOT_GROUP = {
    "helmet": "Head", "plume": "Head", "gorget": "Torso", "cuirass": "Torso", "mail": "Torso", "tabard": "Cloth",
    "cape": "Cloth", "belts": "Belts", "pauldron_l": "Arms", "pauldron_r": "Arms", "rerebrace_l": "Arms",
    "rerebrace_r": "Arms", "couter_l": "Arms", "couter_r": "Arms", "vambrace_l": "Arms", "vambrace_r": "Arms",
    "gauntlet_l": "Hands", "gauntlet_r": "Hands", "mail_skirt": "Legs", "tassets": "Legs", "legs_mail": "Legs",
    "cuisses": "Legs", "poleyns": "Legs", "greaves": "Legs", "sabatons": "Feet", "boots": "Feet",
    "sword": "Props", "shield": "Props", "scabbard": "Props", "underlayer": "Torso", "aventail": "Head", "gorget_top": "Head",
}


def log(*a):
    print("KNT", *a, flush=True)


# =================================================================================================== Blender side
if IN_BLENDER:
    sys.path.insert(0, HERE)
    import numpy as np
    from mathutils import Vector, Matrix
    from mathutils.bvhtree import BVHTree
    import chr_lib as C
    import outfit_lib as OL

    def piece_objects(rig):
        return [o for o in C.children_meshes(rig) if o.get("rts_part")]

    def tris_of(o):
        return sum(len(p.vertices) - 2 for p in o.data.polygons)

    # ------------------------------------------------------------------------------------------- materials
    def fix_materials(rig, kind):
        """Every piece on a texture set from assets/textures (no flat placeholders): the sword's enamel lozenge gets
        steel_blued, its grip the blue wrap with gold wire strip of leather_straps (UVs fitted into the strip)."""
        import char_materials as cm
        import armour_lower as AL
        changed = []
        sw = bpy.data.objects.get(kind + "_sword")
        if sw is not None:
            for i, m in enumerate(sw.data.materials):
                if m and m.name == "M_kl_enamel":
                    en = bpy.data.materials.get("M_kl_enamel_blued") or cm.material(
                        "steel_blued", name="M_kl_enamel_blued", tint=(0.55, 0.62, 1.0), coat=0.5)
                    en["rts_slot"] = "enamel"
                    sw.data.materials[i] = en
                    changed.append("sword enamel -> steel_blued")
                if m and m.name == "M_kl_grip":
                    polys = [p for p in sw.data.polygons if p.material_index == i]
                    gm = bpy.data.materials.get("M_kl_grip_wrap") or cm.material("leather_straps", name="M_kl_grip_wrap")
                    gm["rts_slot"] = "grip"
                    k = cm.map_strip(sw, "leather_straps", "grip_wrap_blue", polys=polys, along="V")
                    sw.data.materials[i] = gm
                    changed.append("sword grip -> leather_straps/grip_wrap_blue (k %.2f)" % k)
        for o in C.children_meshes(rig):
            for m in o.data.materials:
                if m is None or not m.use_nodes:
                    continue
                has_img = any(n.type == 'TEX_IMAGE' and n.image for n in m.node_tree.nodes)
                if not has_img:
                    changed.append("WARNING untextured material %s on %s" % (m.name, o.name))
        for m in list(bpy.data.materials):
            if m.users == 0:
                bpy.data.materials.remove(m)
        return changed

    def swap_1k_images():
        """point the TEX_1K images at their 1k copies (made by the orchestrator: build_knight.make_1k_textures); the
        steel sets first go to their M3 versions (steel_pbr.py: 0.50-0.56 linear steel, non-metal grime)"""
        import steel_pbr
        steel_pbr.swap_images(bpy, log)
        n = 0
        for im in bpy.data.images:
            f = os.path.basename(bpy.path.abspath(im.filepath))
            p = os.path.join(TEX_1K_DIR, f)
            if f.startswith(TEX_1K) and os.path.exists(p) and os.path.normpath(bpy.path.abspath(im.filepath)) != p:
                im.filepath = p; im.reload(); n += 1
        log("images swapped to 1k copies: %d" % n)

    def tag_pieces(rig, kind):
        for o in C.children_meshes(rig):
            role = o.name[len(kind) + 1:]
            part = o.get("rts_part")
            if part:
                o["slot"] = SLOT_GROUP.get(part, "Outfit")
                o["rts_look"] = "helm" if part in HELM_SLOTS else "any"
            elif o.get("rts_probe"):
                continue
            else:
                o["rts_look"] = "bare" if role.split("_")[0] in BARE_ROLES else "any"

    # ------------------------------------------------------------------------------------------- fit report
    def _eval_mesh(o, dg):
        ev = o.evaluated_get(dg); me = ev.to_mesh()
        co = np.empty(len(me.vertices) * 3); me.vertices.foreach_get("co", co)
        no = np.empty(len(me.vertices) * 3); me.vertices.foreach_get("normal", no)
        polys = [tuple(p.vertices) for p in me.polygons]
        ev.to_mesh_clear()
        M = np.array(o.matrix_world)
        co = co.reshape(-1, 3) @ M[:3, :3].T + M[:3, 3]
        return co, no.reshape(-1, 3), polys

    LAYERS = [  # (inner, outer): the outer piece must stay outside the inner one where they overlap
        ("mail", "cuirass"), ("mail", "gorget"), ("mail", "pauldron_l"), ("mail", "pauldron_r"),
        ("mail", "rerebrace_l"), ("mail", "rerebrace_r"), ("mail", "couter_l"), ("mail", "couter_r"),
        ("mail", "vambrace_l"), ("mail", "vambrace_r"), ("gauntlet_l", "vambrace_l"), ("gauntlet_r", "vambrace_r"),
        ("cuirass", "tabard"), ("cuirass", "belts"), ("cuirass", "cape"), ("mail", "mail_skirt"),
        ("mail_skirt", "tabard"), ("mail_skirt", "tassets"), ("tabard", "belts"), ("legs_mail", "cuisses"),
        ("legs_mail", "poleyns"), ("legs_mail", "greaves"), ("boots", "sabatons"), ("boots", "greaves"),
        ("gorget", "helmet"), ("cuirass", "pauldron_l"), ("cuirass", "pauldron_r"),
    ]

    def fit_report(rig, bm, kind, tol=0.0015):
        """Rest-pose fit: (a) piece vertices inside the skin that stays visible (skin not hidden by any piece's delete
        group), (b) outer layers dipping under inner layers where the two surfaces run parallel (normals within 60
        deg, within 2 cm). Returns {piece: {...}} with counts of vertices deeper than tol and max depth (mm)."""
        C.pose_reset(rig)
        masks = [m for m in bm.modifiers if m.type == 'MASK']
        for m in masks:
            m.show_viewport = False
        dg = bpy.context.evaluated_depsgraph_get()
        bco, _, bp = _eval_mesh(bm, dg)
        hide = set(OL.covered_verts(bm)[0])
        keep_faces = [p for p in bp if not any(i in hide for i in p)]
        bvh_body = BVHTree.FromPolygons([Vector(v) for v in bco], keep_faces) if keep_faces else None
        pieces = {o["rts_part"]: o for o in piece_objects(rig) if not o.get("rts_prop")}
        geo = {k: _eval_mesh(o, dg) for k, o in pieces.items()}
        bvh = {k: BVHTree.FromPolygons([Vector(v) for v in g[0]], g[2]) for k, g in geo.items()}
        rep = {}
        for k, (co, no, _) in geo.items():
            d_in = []
            for p in (co if bvh_body else []):
                loc, nrm, _, dist = bvh_body.find_nearest(Vector(p), 0.05)
                if loc is not None:
                    s = (Vector(p) - loc).dot(nrm)
                    if s < -tol:
                        d_in.append(-s)
            rep[k] = dict(verts=len(co), inside_skin=len(d_in), inside_skin_max_mm=round(max(d_in, default=0) * 1000, 1))
        for inner, outer in LAYERS:
            if inner not in geo or outer not in geo:
                continue
            co, no = geo[outer][0], geo[outer][1]
            under = []
            for p, pn in zip(co, no):
                loc, nrm, _, dist = bvh[inner].find_nearest(Vector(p), 0.02)
                if loc is not None and nrm.dot(Vector(pn)) > 0.5:
                    s = (Vector(p) - loc).dot(nrm)
                    if s < -tol:
                        under.append(-s)
            if under:
                rep[outer].setdefault("under", {})[inner] = [len(under), round(max(under) * 1000, 1)]
        for m in masks:
            m.show_viewport = True
        return rep

    # ------------------------------------------------------------------------------------------- export
    def prune_keys(o, keep_all=()):
        """Drop shape keys that do not move this mesh (keep every name in keep_all)."""
        if not o.data.shape_keys:
            return
        kb = o.data.shape_keys.key_blocks
        n = len(o.data.vertices)
        base = np.empty(n * 3); kb["Basis"].data.foreach_get("co", base)
        for k in list(kb)[1:]:
            if k.name in keep_all:
                continue
            a = np.empty(n * 3); k.data.foreach_get("co", a)
            if np.abs(a - base).max() < 1e-5:
                o.shape_key_remove(k)
        if len(o.data.shape_keys.key_blocks) == 1:
            o.shape_key_clear()

    def strip_face_keys(o):
        """Armour / cloth pieces keep only the body-shape (cust_*) morphs. MPFB carries every body key through the
        mhclo reference triangles, so pieces whose references touch the jaw / neck got face keys too: two pauldron
        vertices moved 16 mm with jawOpen (spikes on the shoulders whenever the bare-head knight talked), the gorget /
        mail collar 0.3-2 mm. The face clip then only drives the head, brows, lashes, eyes, teeth, tongue and hair."""
        if not o.data.shape_keys:
            return 0
        kb = o.data.shape_keys.key_blocks
        n = 0
        for k in list(kb)[1:]:
            if not k.name.startswith("cust_"):
                o.shape_key_remove(k); n += 1
        if n:
            log("face keys stripped from %s: %d" % (o.name, n))
        return n

    def make_probe(rig, bm, kind):
        """Full-skin probe for QA (not exported): a copy of the live body with every piece's delete mask off, only the
        'body' vertex group shown, skinned to the rig. knight_qa uses it to find gaps in the armour in every pose."""
        pr = bm.copy(); pr.data = bm.data.copy()
        pr.name = pr.data.name = kind + "_skin_probe"
        bpy.context.scene.collection.objects.link(pr)
        pr.parent = None; pr.matrix_world = Matrix.Identity(4)
        for m in [m for m in pr.modifiers if m.type == 'MASK']:
            pr.modifiers.remove(m)
        mk = pr.modifiers.new("body_only", 'MASK'); mk.vertex_group = "body"; mk.threshold = 0.5
        pr["rts_probe"] = True
        pr.hide_render = True
        for k in [k for k in pr.keys() if k.startswith("rts_") and k != "rts_probe"]:
            del pr[k]
        return pr

    def find_gaps(rig, bm, kind, probe, rings=2):
        """Skin that shows between the pieces in any test pose (clip frames + extreme poses, knight_qa.gap_scan),
        ignoring the face inside the helm and the hand inside the glove; dilated by `rings` edge rings and stored as
        the body vertex group 'rts_gapfill' (the under-layer filler is cut from it at export)."""
        import knight_qa as QA
        pieces = [o for o in piece_objects(rig) if not o.get("rts_prop") and o.get("rts_look") in ("any", "helm")]
        scan = QA.gap_scan(rig, probe, pieces)
        dom = QA.dominant_bones(rig, probe)
        gap = set()
        for pname, (ex, nv) in scan.items():
            real = [i for i in ex if not QA.IGNORE_BONE(dom[i])]
            gap |= set(real)
            if real:
                log("gap %-18s %3d skin verts visible" % (pname, len(real)))
        C.pose_reset(rig)
        bpy.context.scene.frame_set(0)
        me = bm.data
        adj = [[] for _ in me.vertices]
        for e in me.edges:
            a, b = e.vertices; adj[a].append(b); adj[b].append(a)
        bi = bm.vertex_groups["body"].index
        inbody = np.zeros(len(me.vertices), bool)
        for v in me.vertices:
            inbody[v.index] = any(g.group == bi and g.weight > 0.5 for g in v.groups)
        for _ in range(rings):
            gap |= {j for i in list(gap) for j in adj[i] if inbody[j]}
        g = bm.vertex_groups.get("rts_gapfill") or bm.vertex_groups.new(name="rts_gapfill")
        g.add(sorted(gap), 1.0, 'REPLACE')
        log("gaps: %d skin verts (dilated %d rings) -> under-layer filler" % (len(gap), rings))
        return len(gap)

    def merge_groups(o, pred, to_name):
        """move the weights of every vertex group whose name satisfies pred onto group to_name (then drop them)"""
        names = {g.index: g.name for g in o.vertex_groups}
        tgt = o.vertex_groups.get(to_name) or o.vertex_groups.new(name=to_name)
        src = {i for i, n in names.items() if pred(n) and n != to_name}
        moved = 0
        for v in o.data.vertices:
            w = sum(g.weight for g in v.groups if g.group in src)
            if w > 0:
                cur = next((g.weight for g in v.groups if g.group == tgt.index), 0.0)
                tgt.add([v.index], cur + w, 'REPLACE')
                moved += 1
        for g in [g for g in o.vertex_groups if g.index in src]:
            o.vertex_groups.remove(g)
        return moved

    CAPSULES = [  # (collider name, bone, from head / to tail of the bone; radius = body under it + armour margin)
        ("pelvis", "pelvis"), ("spine_lower", "spine_02"), ("chest", "spine_04"), ("head", "head"),
        ("thigh_l", "thigh_l"), ("thigh_r", "thigh_r"), ("calf_l", "calf_l"), ("calf_r", "calf_r"),
        ("upperarm_l", "upperarm_l"), ("upperarm_r", "upperarm_r"), ("lowerarm_l", "lowerarm_l"),
        ("lowerarm_r", "lowerarm_r")]
    CHAIN_SPRING = {  # chain prefix: (stiffness, damping, gravity, drag, colliders)
        "cape": (0.30, 0.22, 1.0, 0.45, ["chest", "spine_lower", "pelvis", "thigh_l", "thigh_r", "calf_l", "calf_r",
                                         "upperarm_l", "upperarm_r"]),
        "tabard": (0.50, 0.30, 1.0, 0.30, ["pelvis", "thigh_l", "thigh_r", "calf_l", "calf_r"]),
    }

    def spring_metadata(rig, bm, margin=0.03):
        """judge m5: engine data for secondary motion (the clips have it baked, blends and new clips do not):
        collider capsules on the body (radius = 85th percentile of the skin distance to the bone segment, measured
        on the full rest body, + an armour margin) and a spring entry per cape / tabard chain. MERGED into the shared
        rts_helpers['springs'] table (the upper armour's plume chain stays)."""
        C.pose_reset(rig)
        masks = [m for m in bm.modifiers if m.type == 'MASK']
        for m in masks:
            m.show_viewport = False
        dg = bpy.context.evaluated_depsgraph_get()
        co, _, _ = _eval_mesh(bm, dg)
        for m in masks:
            m.show_viewport = True
        names = {g.index: g.name for g in bm.vertex_groups}
        dom = []
        for v in bm.data.vertices:
            best = max(((names[g.group], g.weight) for g in v.groups if names[g.group] in rig.data.bones),
                       key=lambda x: x[1], default=(None, 0))
            dom.append(best[0])
        dom = np.array(dom, dtype=object)
        cols = []
        for cn, bn in CAPSULES:
            if bn not in rig.data.bones:
                continue
            b = rig.data.bones[bn]
            A, Bt = np.array(b.head_local), np.array(b.tail_local)
            P = co[dom == bn]
            if not len(P):
                continue
            d = Bt - A
            t = np.clip((P - A) @ d / max(d @ d, 1e-9), 0, 1)
            r = float(np.percentile(np.linalg.norm(P - (A + t[:, None] * d), axis=1), 85)) + margin
            cols.append(dict(name=cn, bone=bn, head=[0.0, 0.0, 0.0], tail=[0.0, round(b.length, 4), 0.0],
                             radius=round(r, 3)))
        chains = {}
        for b in rig.data.bones:
            for pre in CHAIN_SPRING:
                if b.name.startswith(pre + "_"):
                    key = b.name.rsplit("_", 1)[0]               # cape_l, cape_c, tabard_f ...
                    chains.setdefault(key, []).append(b.name)
        try:
            d = json.loads(rig.get("rts_helpers", "{}") or "{}")
        except Exception:
            d = {}
        sp = d.setdefault("springs", {})
        for key, bones in sorted(chains.items()):
            st, dmp, grav, drag, cl = CHAIN_SPRING[key.split("_")[0]]
            sp[key] = dict(bones=sorted(bones), stiffness=st, damping=dmp, gravity=grav, drag=drag,
                           colliders=[c for c in cl if any(x["name"] == c for x in cols)], collider_radius=0.025,
                           doc="cloth chain (spring bones / damped pendulum per bone); the knight clips have it baked "
                               "(armature extras rts_secondary + knight_anim.bake_secondary)")
        d["colliders"] = cols
        d["colliders_doc"] = ("capsules in the bone's local frame (glTF joint space: from the joint along +Y to "
                              "'tail'), radius in metres, body + armour margin; for the spring chains and cloth")
        rig["rts_helpers"] = json.dumps(d)
        log("springs: %d chains (%s), %d collider capsules" % (len(chains), ", ".join(sorted(chains)), len(cols)))
        return d

    def make_fillers(rig, bm, kind):
        """Under-layer filler: the skin patches found by find_gaps, cut from the body, baked like the body (cust
        morphs, body weights) and dressed as riveted mail (leather at the feet), so the gaps between plates show mail
        instead of a hole: '<kind>_underlayer' (both looks) and '<kind>_aventail' (the mail under the helmet's rim,
        helm look only; the bare head has its own neck skin there)."""
        import armour_upper as AU
        import armour_lower as AL
        import char_materials as cm
        if "rts_gapfill" not in bm.vertex_groups:
            return []
        gi = bm.vertex_groups["rts_gapfill"].index
        sel = [v.index for v in bm.data.vertices if any(g.group == gi and g.weight > 0.5 for g in v.groups)]
        if not sel:
            return []
        f = bm.copy(); f.data = bm.data.copy()
        bpy.context.scene.collection.objects.link(f)
        f.parent = rig
        f.name = f.data.name = kind + "_underlayer"
        for k in [k for k in f.keys() if k.startswith("rts_")]:
            del f[k]
        bg = f.vertex_groups["body"]
        keep = set(sel)
        bg.remove([v.index for v in f.data.vertices if v.index not in keep])
        for m in [m for m in f.modifiers if m.type == 'MASK']:
            f.modifiers.remove(m)
        C.bake_body_for_export(f)
        for k in list(f.data.shape_keys.key_blocks)[1:]:
            if not k.name.startswith("cust_"):
                f.shape_key_remove(k)
        head_idx = {g.index for g in f.vertex_groups if g.name in HEAD_BONES or g.name.startswith(HEAD_PREFIX)
                    or g.name == "neck_02"}
        foot_idx = {g.index for g in f.vertex_groups if g.name.startswith(("foot_", "ball_", "calf_twist"))}
        me = f.data
        vhead = np.array([sum(g.weight for g in v.groups if g.group in head_idx) > 0.5 for v in me.vertices])
        vfoot = np.array([sum(g.weight for g in v.groups if g.group in foot_idx) > 0.5 for v in me.vertices])
        mail = AU.make_material("mail"); leather = AL.slot_material("leather")
        me.materials.clear(); me.materials.append(mail); me.materials.append(leather)
        foot_polys = []
        for p in me.polygons:
            ft = sum(vfoot[i] for i in p.vertices) * 2 > len(p.vertices)
            p.material_index = 1 if ft else 0
            if ft:
                foot_polys.append(p)
        mail_polys = [p for p in me.polygons if p.material_index == 0]
        if mail_polys:
            cm.set_texel_density(f, "mail_riveted", polys=mail_polys)
        if foot_polys:
            cm.set_texel_density(f, "leather_brown", polys=foot_polys)
        AU.ensure_ao_attr(f)
        out = [f]
        hp = np.array([all(vhead[i] for i in p.vertices) for p in me.polygons])
        if hp.any() and not hp.all():
            me.polygons.foreach_set("select", hp)
            me.vertices.foreach_set("select", np.zeros(len(me.vertices), bool))
            for p in me.polygons:
                if p.select:
                    for i in p.vertices:
                        me.vertices[i].select = True
            for o in bpy.context.view_layer.objects:
                o.select_set(False)
            bpy.context.view_layer.objects.active = f; f.select_set(True)
            f.active_shape_key_index = 0
            bpy.ops.object.mode_set(mode='EDIT'); bpy.ops.mesh.separate(type='SELECTED'); bpy.ops.object.mode_set(mode='OBJECT')
            av = [o for o in bpy.context.selected_objects if o != f][0]
            av.name = av.data.name = kind + "_aventail"
            # judge m6: the aventail (cut from neck / chin skin) was 71 % weighted to the jaw, so the mail under the
            # helm opened with jawOpen / visemes: face-joint weights go to the head, neck weights stay
            n = merge_groups(av, lambda g: g in ("jaw", "eye_l", "eye_r") or g.startswith(HEAD_PREFIX), "head")
            log("aventail: face-joint weights of %d verts moved to head" % n)
            out.append(av)
        for o in out:
            nm = o.name[len(kind) + 1:]
            o["rts_part"] = nm
            o["rts_asset"] = "gap_filler"
            o["slot"] = "Head" if nm == "aventail" else "Torso"
            o["rts_look"] = "helm" if nm == "aventail" else "any"
            log("filler %-10s %5d verts %5d tris" % (o.name, len(o.data.vertices), sum(len(p.vertices) - 2 for p in o.data.polygons)))
        return out

    def export_knight(rig, bm, kind, path):
        """Engine export of the dressed live human, in place (the live .blend is saved before). The full armour
        covers every bit of skin below the neck, so the knight keeps only the HEAD skin (head / face / upper-neck
        weighted vertices) as '<kind>_head' for the bare-head look (cutscenes, facial animation); everything else
        is deleted (no skin can poke through plates in motion, and no triangles are spent on hidden skin).
        Modelling baked, <= 4 weights, face / cust morphs, clips, sockets."""
        make_fillers(rig, bm, kind)
        head_idx = {g.index for g in bm.vertex_groups if g.name in HEAD_BONES or g.name.startswith(HEAD_PREFIX)}
        neck2 = bm.vertex_groups["neck_02"].index if "neck_02" in bm.vertex_groups else -1
        drop, keep = [], []
        for v in bm.data.vertices:
            hw = sum(g.weight for g in v.groups if g.group in head_idx)
            nw = sum(g.weight for g in v.groups if g.group == neck2)
            (keep if hw + nw > 0.5 else drop).append(v.index)
        bm.vertex_groups["body"].remove(drop)
        for m in [m for m in bm.modifiers if m.type == 'MASK']:
            bm.modifiers.remove(m)
        nv = C.bake_body_for_export(bm)
        log("skin after bake: %d head / neck verts kept for the bare-head look (%d body verts under the armour removed)"
            % (nv, len(drop)))
        bm.name = bm.data.name = kind + "_head"
        OL.export_mouth(rig, kind, bm)                # base pipeline dentition (face_lib, iteration-2 reconcile)
        for o in C.children_meshes(rig):
            un = C.clean_weights(o, rig)
            assert un == 0, "%s: %d unweighted verts" % (o.name, un)
        for o in C.children_meshes(rig):
            if o.get("rts_part"):
                strip_face_keys(o)
            prune_keys(o, keep_all=C.FACE_KEYS if o == bm else ())
        rig["rts_correctives"] = json.dumps([])       # no skin below the neck: nothing for pose correctives to fix
        bpy.context.scene["rts_correctives"] = rig["rts_correctives"]
        tag_pieces(rig, kind)
        dev = C.rest_deviation(rig)
        assert max(dev.values()) < 1e-4, dev
        return bm

    def split_faces(o, mask, name):
        """move the faces in `mask` (bool per polygon) into a new object `name` (same rig, weights, keys)"""
        me = o.data
        me.polygons.foreach_set("select", np.array(mask, bool))
        vs = np.zeros(len(me.vertices), bool)
        for p, m in zip(me.polygons, mask):
            if m:
                vs[list(p.vertices)] = True
        me.vertices.foreach_set("select", vs)
        me.edges.foreach_set("select", np.array([vs[e.vertices[0]] and vs[e.vertices[1]] for e in me.edges]))
        for x in bpy.context.view_layer.objects:
            x.select_set(False)
        bpy.context.view_layer.objects.active = o; o.select_set(True)
        if me.shape_keys:
            o.active_shape_key_index = 0
        bpy.ops.object.mode_set(mode='EDIT'); bpy.ops.mesh.separate(type='SELECTED'); bpy.ops.object.mode_set(mode='OBJECT')
        new = [x for x in bpy.context.selected_objects if x != o][0]
        new.name = new.data.name = name
        return new

    def split_gorget_top(rig, kind):
        """The gorget's top lame reaches the jaw line: right under a helmet, but on a bare head it collars the chin (the
        chin passes through it when the jaw opens or the head nods). It becomes its own mesh '<kind>_gorget_top'
        in the helm look; the bare head keeps the two lower lames over the mail collar."""
        import bmesh
        g = bpy.data.objects.get(kind + "_gorget")
        if g is None or g.get("rts_bare_safe"):          # the armour author made the gorget clear the chin
            return None
        b = bmesh.new(); b.from_mesh(g.data); b.verts.ensure_lookup_table(); b.faces.ensure_lookup_table()
        comp = [-1] * len(b.verts); isl = []
        for v in b.verts:
            if comp[v.index] >= 0:
                continue
            st = [v]; comp[v.index] = len(isl); ids = []
            while st:
                x = st.pop(); ids.append(x.index)
                for e in x.link_edges:
                    y = e.other_vert(x)
                    if comp[y.index] < 0:
                        comp[y.index] = len(isl); st.append(y)
            isl.append(ids)
        zmean = [np.mean([b.verts[i].co.z for i in ids]) for ids in isl]
        top = int(np.argmax(zmean))
        mask = [comp[f.verts[0].index] == top for f in b.faces]
        b.free()
        if all(mask) or not any(mask):
            return None
        t = split_faces(g, mask, kind + "_gorget_top")
        t["rts_part"] = "gorget_top"; t["rts_asset"] = g.get("rts_asset", "knight_gorget")
        log("gorget top lame -> %s (%d verts, helm look only)" % (t.name, len(t.data.vertices)))
        return t

    CULL_SKIP = ("plume", "hair", "eyebrows", "eyelashes", "eyes", "head", "teeth", "tongue")

    def cull_hidden(rig, kind, ndir=128):
        """Remove the faces no camera can ever see (game LOD0 of the ASSEMBLED knight; the piece assets stay whole):
        in every test pose (bind, clip frames, extreme poses) and in both looks, rays from each face centre in `ndir`
        directions (front hemisphere, both for two-sided materials) are cast against the other pieces; a face is kept
        if any ray escapes in any pose / look, plus one ring of neighbours as a margin. Props (can be swapped or
        dropped) and alpha cards (plume, hair) never count as occluders and are never culled."""
        import knight_qa as QA
        from mathutils import Vector
        dirs = []
        for i in range(ndir):
            z = 1 - 2 * (i + 0.5) / ndir; r = math.sqrt(1 - z * z); a = i * 2.399963
            dirs.append(Vector((r * math.cos(a), r * math.sin(a), z)))
        QA.face_keys_static(rig)
        occ = {look: [o for o in QA.meshes(rig) if QA.in_look(o, look) and not o.get("rts_prop")
                      and not o.name[len(kind) + 1:].startswith(CULL_SKIP)] for look in ("helm", "bare")}
        cand = {o.name: o for L in occ.values() for o in L}
        vis = {n: np.zeros(len(o.data.polygons), bool) for n, o in cand.items()}
        two = {n: any(m and not m.use_backface_culling for m in o.data.materials) for n, o in cand.items()}
        t0 = time.time()
        for look, objs in occ.items():
            for pname, setter in QA.test_poses(rig):
                setter()
                dg = bpy.context.evaluated_depsgraph_get()
                V, P, geo = [], [], {}
                for o in objs:
                    ev = o.evaluated_get(dg); me = ev.to_mesh()
                    off = len(V)
                    V += [v.co.copy() for v in me.vertices]
                    P += [tuple(i + off for i in p.vertices) for p in me.polygons]
                    geo[o.name] = ([p.center.copy() for p in me.polygons], [p.normal.copy() for p in me.polygons])
                    ev.to_mesh_clear()
                bvh = BVHTree.FromPolygons(V, P)
                for o in objs:
                    cen, nor = geo[o.name]; v = vis[o.name]; ts = two[o.name]
                    for i, (c, n) in enumerate(zip(cen, nor)):
                        if v[i]:
                            continue
                        for d in dirs:
                            dn = d.dot(n)
                            if dn < 0.05 and not (ts and dn < -0.05):
                                continue
                            if bvh.ray_cast(c + (n if dn > 0 else -n) * 0.0008, d, 3.0)[0] is None:
                                v[i] = True
                                break
        QA.play(rig, None, 0)
        rig.animation_data.action = bpy.data.actions.get("idle")
        total = 0; per = {}
        for n, o in cand.items():
            me = o.data
            v = vis[n].copy()
            # one ring of neighbours around every visible face stays (margin for directions between the samples)
            vert_vis = np.zeros(len(me.vertices), bool)
            for p in me.polygons:
                if v[p.index]:
                    vert_vis[list(p.vertices)] = True
            keep = np.array([v[p.index] or vert_vis[list(p.vertices)].any() for p in me.polygons])
            if keep.all():
                continue
            tris = sum(len(p.vertices) - 2 for p in me.polygons if not keep[p.index])
            me.polygons.foreach_set("select", ~keep)
            me.edges.foreach_set("select", np.zeros(len(me.edges), bool)); me.vertices.foreach_set("select", np.zeros(len(me.vertices), bool))
            for x in bpy.context.view_layer.objects:
                x.select_set(False)
            bpy.context.view_layer.objects.active = o; o.select_set(True)
            if me.shape_keys:
                o.active_shape_key_index = 0
            bpy.ops.object.mode_set(mode='EDIT')
            bpy.ops.mesh.select_mode(type='FACE')
            bpy.ops.mesh.delete(type='ONLY_FACE')
            bpy.ops.mesh.select_all(action='SELECT'); bpy.ops.mesh.delete_loose(use_verts=True, use_edges=True, use_faces=False)
            bpy.ops.object.mode_set(mode='OBJECT')
            o["rts_culled_tris"] = tris
            per[o.name] = tris; total += tris
        log("cull: %d never-visible tris removed in %.0fs %s" % (total, time.time() - t0,
                                                                  {k: v for k, v in sorted(per.items(), key=lambda kv: -kv[1])}))
        return per

    def export_glb(rig, path, animations=True):
        for o in bpy.context.view_layer.objects:
            o.select_set(False)
        meshes = C.children_meshes(rig)
        for o in meshes:
            assert o.matrix_world == Matrix.Identity(4), o.name
            o.parent = None
            o.select_set(True)
        rig.select_set(True)
        bpy.context.view_layer.objects.active = rig
        C.pose_reset(rig)
        bpy.ops.export_scene.gltf(
            filepath=path, export_format='GLB', use_selection=True, export_yup=True, export_apply=False,
            export_skins=True, export_all_influences=False, export_morph=True, export_morph_normal=True,
            export_morph_tangent=False, export_try_sparse_sk=True, export_tangents=True, export_image_format='AUTO',
            export_materials='EXPORT', export_extras=True, export_def_bones=False, export_rest_position_armature=True,
            export_leaf_bone=False, export_animations=animations, export_animation_mode='ACTIONS',
            export_force_sampling=True, export_frame_step=1, export_anim_single_armature=True,
            export_reset_pose_bones=True, export_optimize_animation_size=True, export_morph_animation=True,
            export_bake_animation=False, export_merge_animation='ACTION', export_anim_slide_to_zero=True)
        for o in meshes:
            o.parent = rig
            o.matrix_parent_inverse = Matrix.Identity(4)
        import kit_glb
        nt = kit_glb.fix_tangents(path)              # degenerate-UV vertices get zero tangents (validator error)
        log("glb", path, "%.2f MB" % (os.path.getsize(path) / 1e6) + (" (%d zero tangents repaired)" % nt if nt else ""))

    def post_clip_hooks(rig, kind):
        """Armour modules can post-process the clips after knight_anim made them (e.g. armour_upper.post_clips: plume
        spring chain, gauntlet grip poses): any `post_clips(rig, kind)` / `post_clips(rig)` found in armour_upper /
        armour_lower is called here, in that order."""
        import inspect, importlib, traceback
        done = []
        table = json.loads(rig.get("rts_clips", "{}"))
        acts = [bpy.data.actions[n] for n in table if n != "talk_emote" and bpy.data.actions.get(n)]
        for name in ("armour_upper", "armour_lower"):
            mod = sys.modules.get(name) or importlib.import_module(name)
            fn = getattr(mod, "post_clips", None)
            if fn is None:
                continue
            params = inspect.signature(fn).parameters
            log("post_clips hook: %s.post_clips" % name)
            cur = rig.animation_data.action if rig.animation_data else None
            try:
                if "actions" in params:
                    # one action at a time, assigned to the rig first: Blender 5's fcurve_ensure_for_datablock refuses
                    # an action that is not assigned to the datablock (the female build failed on 'walk' that way)
                    for act in acts:
                        rig.animation_data.action = act
                        fn(rig, kind, actions=[act])
                elif len(params) >= 2:
                    fn(rig, kind)
                else:
                    fn(rig)
                done.append(name)
            except Exception as e:                    # an armour module mid-edit must not stop the knight build
                log("WARNING post_clips hook %s failed (clips kept without it): %r" % (name, e))
                traceback.print_exc()
                done.append("%s FAILED: %r" % (name, e))
            if rig.animation_data:
                rig.animation_data.action = cur
            C.pose_reset(rig)
        return done

    # ------------------------------------------------------------------------------------------- assemble
    def assemble(kind):
        import armour_upper as AU
        import armour_lower as AL
        t0 = time.time()
        rig, bm = OL.live_human(kind)
        C.pose_reset(rig)
        import rig_helpers as RH
        RH.ensure_helper_bones(rig, log)                # plate helper joints (C1 / M1), before any .mhw is loaded
        up = AU.dress(kind, check_fit=False)
        log("upper pieces %d (%.0fs)" % (len(up), time.time() - t0))
        low = AL.load_lower(rig, bm, props=False)
        AL.add_props(rig, which=KNIGHT_PROPS)
        RH.refit_helpers(rig, log)                       # every helper identity at rest (shared helper table)
        log("lower pieces %d + props %s (%.0fs)" % (len(low), KNIGHT_PROPS, time.time() - t0))
        for m in fix_materials(rig, kind):
            log("material:", m)
        swap_1k_images()
        tag_pieces(rig, kind)
        info = dict(kind=kind, pieces={})
        # rigid plates: 100 % per component on one joint / helper joint (rig_helpers.PLATE_RULES; pieces whose
        # authors weighted them to helper joints themselves are left alone)
        info["plate_weights"] = RH.apply_plate_rules(
            rig, {o["rts_part"]: o for o in piece_objects(rig) if not o.get("rts_prop")}, kind, log)
        for o in piece_objects(rig):
            info["pieces"][o["rts_part"]] = dict(tris=tris_of(o), verts=len(o.data.vertices),
                                                 mats=[m.name for m in o.data.materials if m],
                                                 keys=len(o.data.shape_keys.key_blocks) - 1 if o.data.shape_keys else 0)
        info["fit"] = fit_report(rig, bm, kind)
        for k, r in sorted(info["fit"].items()):
            log("fit %-12s inside visible skin %4d (max %5.1f mm)  under inner layer %s"
                % (k, r["inside_skin"], r["inside_skin_max_mm"], r.get("under", {})))
        live = os.path.join(OUTD, "knight_%s.blend" % kind)
        bpy.ops.wm.save_as_mainfile(filepath=live)             # dressed, before the clips (animation iteration base)
        import knight_anim as KA
        clips = KA.make_clips(rig, kind)
        info["clips"] = clips
        rig["rts_clips"] = json.dumps(clips)
        info["post_clips"] = post_clip_hooks(rig, kind)     # armour modules' own clip passes (plume springs, grips)
        probe = make_probe(rig, bm, kind)
        info["gap_verts"] = find_gaps(rig, bm, kind, probe)
        info["springs"] = sorted(spring_metadata(rig, bm).get("springs", {}))
        C.pose_reset(rig)
        bpy.ops.wm.save_as_mainfile(filepath=live)
        log("saved", live)
        head = export_knight(rig, bm, kind, None)
        split_gorget_top(rig, kind)
        tag_pieces(rig, kind)
        KA.bake_secondary(rig)                       # driver-driven cape / tabard bones keyed into every clip
        info["culled"] = cull_hidden(rig, kind) if CULL else {}
        KA.make_face_clip(rig, kind)                 # talk / emote clip on the final face meshes
        glb = os.path.join(OUTD, "knight_%s.glb" % kind)
        export_glb(rig, glb)
        bpy.ops.wm.save_as_mainfile(filepath=os.path.join(OUTD, "knight_%s_export.blend" % kind))
        info["meshes"] = {o.name: dict(tris=tris_of(o), look=o.get("rts_look"), slot=o.get("slot"),
                                       keys=len(o.data.shape_keys.key_blocks) - 1 if o.data.shape_keys else 0)
                          for o in C.children_meshes(rig)}
        info["bones"] = len(rig.data.bones)
        json.dump(info, open(os.path.join(OUTD, "knight_%s_assembly.json" % kind), "w"), indent=1)
        log("assembled %s in %.0fs" % (kind, time.time() - t0))

    def blender_main(args):
        mode = args[0]
        if mode == "assemble":
            assemble(args[1] if len(args) > 1 else "male")
        elif mode == "qa":
            import knight_qa
            knight_qa.main(args[1:])
        else:
            raise SystemExit("unknown mode " + mode)


# =================================================================================================== orchestrator
def run(cmd, env=None, filt=True):
    t0 = time.time()
    log("$ " + " ".join(cmd))
    e = dict(os.environ)
    e.update(env or {})
    e["BLENDER_USER_RESOURCES"] = os.path.join(CH, "blender_profile")
    p = subprocess.Popen(cmd, cwd=CH, env=e, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    tail = []
    for line in p.stdout:
        tail = (tail + [line])[-60:]
        if not filt or line.startswith(("KNT", "CHR", "ARM", "ANI", "QA ", "GLB", "   ", "VIEWER", "Traceback", "Error",
                                        "AssertionError", "glb_web", "compare", "WARNING")):
            print(line.rstrip(), flush=True)
    rc = p.wait()
    if rc != 0:
        print("".join(tail))
        raise SystemExit("FAILED (%d): %s" % (rc, " ".join(cmd)))
    log("   done in %.0fs" % (time.time() - t0))


def make_1k_textures():
    """1k copies (Lanczos) of the TEX_1K sets in assets/textures/_1k (regenerated when the 2k master is newer)"""
    from PIL import Image
    import steel_pbr
    src_dir = os.path.join(CH, "assets", "textures")
    os.makedirs(TEX_1K_DIR, exist_ok=True)
    masters = [os.path.join(src_dir, b) for b, _, _ in steel_pbr.SETS] + [os.path.join(src_dir, o) for _, o, _ in steel_pbr.SETS]
    m3 = [os.path.join(steel_pbr.OUT, os.path.basename(p)) for p in masters if os.path.exists(p)]
    if any(not os.path.exists(q) or os.path.getmtime(q) < os.path.getmtime(os.path.join(src_dir, os.path.basename(q)))
           for q in m3):
        steel_pbr.main(check=True)                   # M3 steel maps are stale: rebuild (~10 s)
    n = 0
    for root, _, files in os.walk(src_dir):
        if root.startswith(TEX_1K_DIR) or root.startswith(steel_pbr.OUT) or "heraldry" in root:
            continue
        for f in files:
            if not f.endswith(".png") or not f.startswith(TEX_1K) or f.endswith("_height.png"):
                continue
            src = os.path.join(root, f); dst = os.path.join(TEX_1K_DIR, f)
            if os.path.exists(os.path.join(steel_pbr.OUT, f)):
                src = os.path.join(steel_pbr.OUT, f)       # the 1k copy of a steel set comes from its M3 version
            if os.path.exists(dst) and os.path.getmtime(dst) >= os.path.getmtime(src):
                continue
            im = Image.open(src)
            if max(im.size) <= 1024:
                im.save(dst); continue
            s = 1024 / max(im.size)
            im.resize((max(1, round(im.width * s)), max(1, round(im.height * s))), Image.LANCZOS).save(dst, optimize=True)
            n += 1
    log("1k texture copies: %d written in %s" % (n, TEX_1K_DIR))


def blender(blend, *args, env=None):
    cmd = [BL, "-b"] + ([blend] if blend else []) + ["--python-exit-code", "1", "-P", os.path.join(HERE, "build_knight.py"),
                                                     "--"] + list(args)
    run(cmd, env=env)


def install(src, dst):
    """atomic install of a finished output (same file system: os.replace), so no parallel reader sees a half file"""
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    os.replace(src, dst)
    log("installed", os.path.relpath(dst, CH))


def assemble_staged(kind):
    """'assemble' into out_stage/, then install the four outputs into out/ (atomic per file)"""
    os.makedirs(STAGE_DIR, exist_ok=True)
    names = ["knight_%s.blend" % kind, "knight_%s_export.blend" % kind, "knight_%s.glb" % kind,
             "knight_%s_assembly.json" % kind]
    for n in names:
        for x in (n, n + "1"):
            if os.path.exists(os.path.join(STAGE_DIR, x)):
                os.remove(os.path.join(STAGE_DIR, x))
    blender(os.path.join("out", "base_%s.blend" % kind), "assemble", kind, env={"KNIGHT_OUTD": STAGE_DIR})
    missing = [n for n in names if not os.path.exists(os.path.join(STAGE_DIR, n))]
    if missing:
        raise SystemExit("assemble %s: staged outputs missing %s" % (kind, missing))
    for n in names:
        install(os.path.join(STAGE_DIR, n), os.path.join(OUTD, n))
    for n in names:
        b1 = os.path.join(STAGE_DIR, n + "1")
        if os.path.exists(b1):
            os.remove(b1)


def validate(files):
    """Khronos validator (scripts/kit_validate.js resolves external texture uris) + skin limits: {file: {...}}"""
    import kit_manifest as KM
    byname, txt = KM.run_validator(files)            # keyed by file basename (the validator prints basenames)
    res = {}
    for f in files:
        r = res[f] = dict(byname.get(os.path.basename(f), dict(errors=-1, warnings=-1, tris=0)))
        r["skin"] = KM.skin_limits(f)
        r["MB"] = round(os.path.getsize(f) / 1e6, 2)
    return res


def block_sheet(kinds):
    """renders/knight_<kind>_block_vs_i33.png: the manuscript figure (user reference, I.33 half-shield) next to the
    block_shield hold-frame renders of knight_qa.block (side from the left, 3/4, front)"""
    from PIL import Image, ImageDraw
    ref0 = Image.open(os.path.join(CH, "refs", "feedback", "user_block_guard_reference.png")).convert("RGB")
    for k in kinds:
        fs = [os.path.join(CH, "renders", "knight_%s_block_%s.png" % (k, v)) for v in ("side", "34", "front")]
        if not all(os.path.exists(f) for f in fs):
            continue
        h = 600
        ref = ref0.crop((0, 960, 800, 1580))
        ims = [ref.resize((round(ref.width * h / ref.height), h))]
        ims += [Image.open(f).convert("RGB") for f in fs]
        ims = [ims[0]] + [i.resize((round(i.width * h / i.height), h)) for i in ims[1:]]
        labs = ["I.33 half-shield (user reference)", "block hold: left side", "3/4", "front"]
        sh = Image.new("RGB", (sum(i.width for i in ims), h + 28), (24, 24, 28))
        x = 0
        for im, lab in zip(ims, labs):
            sh.paste(im, (x, 28)); ImageDraw.Draw(sh).text((x + 8, 8), lab, fill=(255, 220, 120)); x += im.width
        sh.save(os.path.join(CH, "renders", "knight_%s_block_vs_i33.png" % k))
        log("block sheet renders/knight_%s_block_vs_i33.png" % k)


def kit_shots(kinds):
    """dress-up page screenshots (viewer/shot.sh BATCH, one page load per body) + a contact sheet per body"""
    from PIL import Image, ImageDraw
    import kit_manifest as KM
    man = json.load(open(os.path.join(KM.KIT, "kit_manifest.json")))
    for k in kinds:
        page = os.path.join(CH, "viewer", "kit_%s_dressup.html" % k)
        if not os.path.exists(page):
            continue
        outfits = [n for n, o in man["outfits"].items() if k in o.get("compatible_bodies", [k])]
        specs, tiles = [], []
        for n in outfits:
            f = "renders/kit_%s_%s_front.png" % (k, n)
            specs.append({"out": f, "view": "Full body", "kit": n, "clip": "idle (anims_knight)", "time": 1.0})
            tiles.append((f, n))
        extra = [("knight_nohelm", {"view": "Full body", "kit": {"outfit": "knight", "unequip": ["helmet"]},
                                    "clip": "walk (anims_knight)", "time": 0.3}),
                 ("knight_block", {"view": "Full body", "kit": "knight", "clip": "block_shield (anims_knight)",
                                   "time": 0.9, "yaw": 40}),
                 ("knight_back", {"view": "Back", "kit": "knight", "clip": "walk (anims_knight)", "time": 0.5})]
        for n, sp in extra:
            f = "renders/kit_%s_%s.png" % (k, n)
            sp = dict(sp); sp["out"] = f
            specs.append(sp); tiles.append((f, n.replace("_", " ")))
        ui = "renders/kit_%s_ui.png" % k
        specs.append({"out": ui, "view": "Full body", "kit": "militia" if "militia" in outfits else "knight",
                      "ui": True, "tab": "kit", "clip": "idle (anims_knight)", "time": 2.0})
        specs.append({"info": "renders/kit_%s_info.json" % k})
        bj = os.path.join(OUTD, "kit", "shots_%s.json" % k)
        json.dump(specs, open(bj, "w"), indent=0)
        e = dict(os.environ); e["BATCH"] = bj; e["CLEAN"] = "1"
        t0 = time.time()
        r = subprocess.run(["viewer/shot.sh", page, "-", "Full body", "", "", "", "640", "860"], cwd=CH, env=e,
                           capture_output=True, text=True)
        if r.returncode != 0:
            print(r.stdout[-3000:], r.stderr[-3000:])
            raise SystemExit("kit shots failed for %s" % k)
        ims = [(Image.open(os.path.join(CH, f)).convert("RGB"), n) for f, n in tiles if os.path.exists(os.path.join(CH, f))]
        if ims:
            cols = min(4, len(ims)); rows = (len(ims) + cols - 1) // cols
            w, h = ims[0][0].size
            sh = Image.new("RGB", (cols * w, rows * h), (20, 22, 28))
            for i, (im, n) in enumerate(ims):
                x, y = (i % cols) * w, (i // cols) * h
                sh.paste(im.resize((w, h)), (x, y))
                ImageDraw.Draw(sh).text((x + 12, y + 10), n, fill=(255, 220, 120))
            sh = sh.resize((sh.width // 2, sh.height // 2), Image.LANCZOS)
            sh.save(os.path.join(CH, "renders", "kit_%s_sheet.png" % k))
        log("kit shots %s: %d in %.0fs -> renders/kit_%s_sheet.png" % (k, len(specs) - 1, time.time() - t0, k))


def main():
    argv = sys.argv[1:]
    kinds = ["male", "female"]
    if "--kinds" in argv:
        i = argv.index("--kinds"); kinds = []
        for a in argv[i + 1:]:
            if a.startswith("-") or a in STAGES:
                break
            kinds.append(a)
        argv = argv[:i] + argv[i + 1 + len(kinds):]
    stages = [a for a in argv if a in STAGES] or STAGES
    t0 = time.time()
    if "tex" in stages:
        run(["python3", "scripts/textures_char.py"])
        run(["python3", "scripts/armour_upper_texgen.py"])
    if "upper" in stages:
        run([BL, "-b", "out/base_male.blend", "--python-exit-code", "1", "-P", "scripts/armour_upper.py", "--", "author"])
    if "lower" in stages:
        run([BL, "-b", "out/base_male.blend", "--python-exit-code", "1", "-P", "scripts/armour_lower.py", "--", "author"])
        run(["python3", "scripts/armour_lower_tex.py"])
    if "assemble" in stages:
        make_1k_textures()
        for k in kinds:
            assemble_staged(k)
    if "check" in stages:
        import knight_report
        knight_report.main(kinds)
        if set(kinds) == {"male", "female"}:
            # prop GLB <-> socket contract (the lower armour owns the props and their stand-alone GLBs: a mismatch
            # while it re-authors them is reported, not fatal for the knight build)
            try:
                run(["python3", "scripts/armour_lower_sockets_check.py", "knight"], filt=False)
            except SystemExit as e:
                log("WARNING prop / socket contract check failed (%s): re-export out/props (armour_lower.py props)" % e)
    if "kit" in stages:
        for k in kinds:
            run([BL, "-b", "out/knight_%s.blend" % k, "--python-exit-code", "1", "-P", "scripts/kit_build.py", "--",
                 "pieces", k])
            run([BL, "-b", "out/knight_%s_export.blend" % k, "--python-exit-code", "1", "-P", "scripts/kit_build.py",
                 "--", "anims", k])
        run(["python3", "scripts/kit_manifest.py"] + kinds, filt=False)     # every body with pieces on disk
        for k in kinds:
            run(["python3", "viewer/make_viewer.py", "--kit", "out/kit/kit_manifest.json", "--kind", k, "--sub",
                 "scripts/build_knight.py kit"])
        kit_shots(kinds)
    if "lod" in stages:
        files = []
        for k in kinds:
            run([BL, "-b", "out/knight_%s_export.blend" % k, "--python-exit-code", "1", "-P", "scripts/knight_lod.py",
                 "--", "game", k])
            run([BL, "-b", "out/knight_%s_export.blend" % k, "--python-exit-code", "1", "-P", "scripts/knight_lod.py",
                 "--", "face", k])
            files += [os.path.join(OUTD, "game", "knight_%s_lod%d.glb" % (k, L)) for L in (0, 1, 2)]
            files += [os.path.join(OUTD, "cutscene", "knight_%s_face.glb" % k)]
        res = validate([f for f in files if os.path.exists(f)])
        bad = {os.path.relpath(f, CH): r for f, r in res.items() if r["errors"] != 0 or r["skin"]}
        for f, r in sorted(res.items()):
            log("lod glb %-40s %6.2f MB  tris %6d  validator %d errors %d warnings  skin %s" % (
                os.path.relpath(f, CH), r["MB"], r["tris"], r["errors"], r["warnings"], r["skin"] or "OK"))
        json.dump({os.path.relpath(f, CH): r for f, r in res.items()},
                  open(os.path.join(OUTD, "game", "lod_validation.json"), "w"), indent=1)
        if bad:
            raise SystemExit("LOD / cutscene GLB validation failed: %s" % bad)
    if "web" in stages:
        for k in kinds:                              # written to temp names, then installed atomically
            run(["python3", "scripts/glb_web.py", "out/knight_%s.glb" % k, "out/test/.knight_%s_web.tmp.glb" % k, "--max", "1024"])
            install(os.path.join(OUTD, "test", ".knight_%s_web.tmp.glb" % k), os.path.join(OUTD, "test", "knight_%s_web.glb" % k))
            run(["python3", "viewer/make_viewer.py", "out/test/knight_%s_web.glb" % k, "-o", "viewer/.knight_%s_lookdev.tmp.html" % k,
                 "--title", "Knight (%s)" % k, "--sub", "scripts/build_knight.py", "--town"])
            install(os.path.join(CH, "viewer", ".knight_%s_lookdev.tmp.html" % k), os.path.join(CH, "viewer", "knight_%s_lookdev.html" % k))
    if "qa" in stages:
        for k in kinds:
            blender(os.path.join("out", "knight_%s_export.blend" % k), "qa", k)
        block_sheet(kinds)
    if "shots" in stages:
        import knight_shots
        for k in kinds:
            knight_shots.main(k)
    if "compare" in stages:
        import knight_shots
        knight_shots.compare(kinds)
    if "integrity" in stages:
        import armour_integrity
        armour_integrity.run_stage(kinds)
    log("build_knight done in %.1f min (%s; %s)" % ((time.time() - t0) / 60, " ".join(stages), " ".join(kinds)))


if __name__ == "__main__":
    if IN_BLENDER:
        a = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
        blender_main(a)
    else:
        sys.path.insert(0, HERE)
        main()
