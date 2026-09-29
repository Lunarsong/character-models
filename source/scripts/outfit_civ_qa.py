"""QA for the civilian outfits: numeric penetration per pose / clip frame (triangle-triangle intersections between
every pair of pieces and between each piece and the visible skin), used as a gate, plus render helpers.

Used by outfit_peasant.py / outfit_archer.py (mode `check`, on unwritten pieces) and outfit_civ.py (mode `qa`, on the
exported dressed file, i.e. exactly what is in the GLB).
"""
import sys, os, json, math, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from outfit_civ_lib import *
import outfit_civ_poses as PZ
from mathutils.bvhtree import BVHTree
from mathutils import Vector
from mathutils.kdtree import KDTree


def evaluated(o, dg):
    ev = o.evaluated_get(dg)
    me = ev.to_mesh()
    co = [ (o.matrix_world @ v.co).copy() for v in me.vertices]
    polys = [tuple(p.vertices) for p in me.polygons]
    ev.to_mesh_clear()
    return co, polys


def masked_body(bm, hide, name="qa_body"):
    """a copy of the body (Armature modifier kept) with the hidden skin (and all helper geometry) removed"""
    import bmesh
    ob = bm.copy(); ob.data = bm.data.copy(); ob.name = name
    bpy.context.scene.collection.objects.link(ob)
    for md in list(ob.modifiers):
        if md.type != 'ARMATURE':
            ob.modifiers.remove(md)
    me = ob.data
    if me.shape_keys:
        # bake the modelling mix into the mesh (the rest shape the garments were built on)
        tmp = ob.shape_key_add(name="__mix", from_mix=True)
        co = np.empty(len(me.vertices) * 3); tmp.data.foreach_get("co", co)
        ob.shape_key_clear()
        me.vertices.foreach_set("co", co)
    drop = np.zeros(len(me.vertices), bool)
    drop[NB:] = True
    drop[list(hide)] = True
    b = bmesh.new(); b.from_mesh(me)
    b.verts.ensure_lookup_table()
    bmesh.ops.delete(b, geom=[b.verts[i] for i in np.nonzero(drop)[0]], context='VERTS')
    b.to_mesh(me); b.free()
    return ob


def _eval_arrays(o, dg):
    ev = o.evaluated_get(dg)
    me = ev.to_mesh()
    n = len(me.vertices)
    co = np.empty(n * 3); me.vertices.foreach_get("co", co)
    no = np.empty(n * 3); me.vertices.foreach_get("normal", no)
    polys = [tuple(p.vertices) for p in me.polygons]
    ev.to_mesh_clear()
    M = np.array(o.matrix_world)
    co = co.reshape(-1, 3) @ M[:3, :3].T + M[:3, 3]
    no = no.reshape(-1, 3) @ M[:3, :3].T
    no /= np.maximum(np.linalg.norm(no, axis=1, keepdims=True), 1e-12)
    return co, no, polys


def _set_frame(rig, pose, clip, f):
    if pose is not None:
        if rig.animation_data:
            rig.animation_data.action = None
        PZ.apply(rig, pose)
    elif clip is None:
        if rig.animation_data:
            rig.animation_data.action = None
        pose_reset(rig)
    else:
        pose_reset(rig)
        rig.animation_data.action = bpy.data.actions[clip]
        bpy.context.scene.frame_set(f)
    bpy.context.view_layer.update()


POKE_TOL = 0.001          # an inner-layer vertex > 1 mm outside the outer layer that covers it = visible poke-through
COVER_REACH = 0.035          # garment layer gaps are < 3.5 cm: farther 'cover' is an accidental neighbour
POKE_REACH = 0.03
RIM_MARGIN = 0.015        # inner vertices within 15 mm of the outer piece's open edge are not 'covered'


def first_facing(bvh, o, d, maxd, n_ref, tries=5):
    """distance of the first ray hit on a face whose normal agrees with n_ref (the outer skin of a garment seen from
    the vertex side), skipping turned-in hems / linings that face the other way; None if none within maxd"""
    travelled = 0.0
    o = Vector(o); d = Vector(d)
    for _ in range(tries):
        hit = bvh.ray_cast(o, d, maxd - travelled)
        if hit[0] is None:
            return None
        travelled += hit[3]
        if hit[1].dot(n_ref) > 0.0:
            return travelled
        o = hit[0] + d * 1e-5
        travelled += 1e-5
    return None


def penetration(rig, pieces, body=None, poses=None, clips=None, step=2, allow=None, verbose=True, tag="",
                layers=None, props=()):
    """pieces: {name: object}; body: the visible-skin object (or None); layers: {name: layer} (inner < outer, the
    body is 0). For every pose (outfit_civ_poses names) and every `step`-th frame of each clip (actions on the rig):
      * tri: intersecting triangle pairs per object pair (strict: includes contacts deep inside folds)
      * pokes: VISIBLE poke-through: vertices of an inner layer (or the skin) that the outer layer covers at rest
        (rest normal ray hits it within 6 cm) and that end up more than 1 mm OUTSIDE the posed outer surface
    props (names) are left out of the stress poses (a bow held at bind in a stress pose means nothing), kept in clips.
    Returns {"pairs", "pokes", "per_pose", "pokes_per_pose", "tested", "gate"}."""
    names = list(pieces)
    objs = [pieces[n] for n in names]
    if body is not None:
        names.append("skin"); objs.append(body)
    lay = dict(layers or {})
    lay.setdefault("skin", 0)
    frames = [("pose:" + p, p, None, None) for p in (poses or [])]
    for c in clips or []:
        act = bpy.data.actions.get(c)
        if not act:
            continue
        nf = int(round(act.frame_range[1]))
        frames += [("%s_f%03d" % (c, f), None, c, f) for f in range(0, nf + 1, step)]
    # rest: which inner vertices each outer layer covers
    _set_frame(rig, None, None, 0)
    dg = bpy.context.evaluated_depsgraph_get()
    rest = {nm: _eval_arrays(o, dg) for nm, o in zip(names, objs)}
    cover = {}
    for a in names:
        for b in names:
            if a == b or a in props or b in props or lay.get(a, 50) >= lay.get(b, 50):
                continue
            co, no, _ = rest[a]
            cb_, nb_, pb_ = rest[b]
            bvh = BVHTree.FromPolygons([Vector(v) for v in cb_], pb_)
            lo, hi = cb_.min(0) - COVER_REACH, cb_.max(0) + COVER_REACH
            cand = np.nonzero(np.all((co > lo) & (co < hi), axis=1))[0]
            # the outer piece's open edges (rims, hems, openings): an inner layer right at a rim legitimately shows
            # when the rim moves, it is not poking through
            ec = {}
            for pp in pb_:
                for q in range(len(pp)):
                    e = (min(pp[q], pp[q - 1]), max(pp[q], pp[q - 1]))
                    ec[e] = ec.get(e, 0) + 1
            bv = sorted(set(v for e, c_ in ec.items() if c_ == 1 for v in e))
            kd = KDTree(max(1, len(bv)))
            for q, v in enumerate(bv):
                kd.insert(Vector(cb_[v]), q)
            kd.balance()
            cov = []
            for i in cand:
                if bv and kd.find(Vector(co[i]))[2] < RIM_MARGIN:
                    continue
                n_ = Vector(no[i])
                if first_facing(bvh, co[i] + no[i] * 1e-4, n_, COVER_REACH, n_) is not None:
                    cov.append(int(i))
            if cov:
                cover[(a, b)] = np.array(cov)
    # dominant joint per vertex (where do the pokes happen)
    dom = {}
    for nm, o in zip(names, objs):
        gn = {g.index: g.name for g in o.vertex_groups}
        d = []
        for v in o.data.vertices:
            best = max(v.groups, key=lambda g: g.weight, default=None)
            d.append(gn.get(best.group, "?") if best is not None else "?")
        dom[nm] = d
    res, per, pk, pkper, visper = {}, {}, {}, {}, {}
    for label, pose, clip, f in frames:
        _set_frame(rig, pose, clip, f)
        dg = bpy.context.evaluated_depsgraph_get()
        act_names = [nm for nm in names if not (pose is not None and nm in props)]
        ev = {nm: _eval_arrays(o, dg) for nm, o in zip(names, objs) if nm in act_names}
        bvh = {nm: BVHTree.FromPolygons([Vector(v) for v in ev[nm][0]], ev[nm][2]) for nm in act_names}
        tot = 0
        for i in range(len(act_names)):
            for j in range(i + 1, len(act_names)):
                a, b = act_names[i], act_names[j]
                n = len(bvh[a].overlap(bvh[b]))
                if n:
                    key = "%s|%s" % (a, b)
                    r = res.setdefault(key, {"max": 0, "at": None, "frames": 0})
                    r["frames"] += 1
                    if n > r["max"]:
                        r["max"], r["at"] = n, label
                    tot += n
        per[label] = tot
        ptot = 0
        vtot = 0
        # everything in the frame (pieces + skin): a poking vertex whose outward ray is blocked within 25 cm sits in a
        # crease / under an arm and cannot be seen
        V_, P_ = [], []
        for nm in act_names:
            o_ = len(V_)
            V_ += [Vector(v) for v in ev[nm][0]]
            P_ += [tuple(i + o_ for i in pp) for pp in ev[nm][2]]
        allbvh = BVHTree.FromPolygons(V_, P_)
        for (a, b), idx in cover.items():
            if a not in ev or b not in ev:
                continue
            co, no = ev[a][0], ev[a][1]
            nmax, sdmax, nvis = 0, 0.0, 0
            bones = {}
            for i in idx:
                n_ = Vector(no[i])
                back = first_facing(bvh[b], co[i], -n_, POKE_REACH, n_)
                if back is None or back < POKE_TOL:
                    continue
                fwd = first_facing(bvh[b], co[i], n_, POKE_REACH, n_)
                if fwd is not None and fwd < back:
                    continue
                nmax += 1; sdmax = max(sdmax, back)
                bones[dom[a][i]] = bones.get(dom[a][i], 0) + 1
                if allbvh.ray_cast(Vector(co[i]) + n_ * 0.003, n_, 0.25)[0] is None:
                    nvis += 1
            vtot += nvis
            if nmax:
                key = "%s>%s" % (a, b)
                r = pk.setdefault(key, {"max": 0, "at": None, "frames": 0, "max_mm": 0.0, "joints": {},
                                        "visible_max": 0, "visible_frames": 0})
                r["visible_max"] = max(r["visible_max"], nvis); r["visible_frames"] += int(nvis > 0)
                for bn, c_ in bones.items():
                    r["joints"][bn] = r["joints"].get(bn, 0) + c_
                r["frames"] += 1
                r["max_mm"] = round(max(r["max_mm"], sdmax * 1000), 1)
                if nmax > r["max"]:
                    r["max"], r["at"] = nmax, label
                ptot += nmax
        pkper[label] = ptot
        visper[label] = vtot
    _set_frame(rig, None, None, 0)
    for pb in rig.pose.bones:
        pb.rotation_mode = 'QUATERNION'
    global LAST_COVER
    LAST_COVER = (names, objs, cover)
    rest_label = "pose:rest" if "pose:rest" in per else None
    out = {"pairs": dict(sorted(res.items(), key=lambda kv: -kv[1]["max"])),
           "pokes": dict(sorted(pk.items(), key=lambda kv: -kv[1]["max"])),
           "per_pose": per, "pokes_per_pose": pkper, "visible_pokes_per_pose": visper, "tested": len(frames),
           "cover_pairs": {"%s>%s" % k: int(len(v)) for k, v in cover.items()}}
    out["gate"] = dict(poke_verts_total=int(sum(pkper.values())), poke_frames=int(sum(1 for v in pkper.values() if v)),
                       visible_poke_verts_total=int(sum(visper.values())),
                       visible_poke_frames=int(sum(1 for v in visper.values() if v)),
                       tri_pairs_at_rest=int(per.get(rest_label, 0)) if rest_label else None,
                       tri_frames=int(sum(1 for v in per.values() if v)))
    if verbose:
        clog("penetration %s: %d poses/frames; poke-through verts %d in %d frames (VISIBLE %d in %d frames); tri-tri "
             "pairs in %d frames" % (tag, len(frames), out["gate"]["poke_verts_total"], out["gate"]["poke_frames"],
                                     out["gate"]["visible_poke_verts_total"], out["gate"]["visible_poke_frames"],
                                     out["gate"]["tri_frames"]))
        for k, r in list(out["pokes"].items())[:20]:
            top = sorted(r["joints"].items(), key=lambda kv: -kv[1])[:4]
            clog("   POKE %-30s max %5d verts (%.1f mm) at %-22s (%d frames; visible max %d in %d frames) %s" % (
                k, r["max"], r["max_mm"], r["at"], r["frames"], r["visible_max"], r["visible_frames"], top))
        for k, r in list(out["pairs"].items())[:20]:
            clog("   tri  %-30s max %5d pairs at %-22s (%d frames)" % (k, r["max"], r["at"], r["frames"]))
    return out


LAST_COVER = None


def poke_vertices(names, objs, cover):
    """{inner name: set(vertex ids)} poking through their covering layers in the current pose"""
    dg = bpy.context.evaluated_depsgraph_get()
    ev = {nm: _eval_arrays(o, dg) for nm, o in zip(names, objs)}
    bvh = {nm: BVHTree.FromPolygons([Vector(v) for v in ev[nm][0]], ev[nm][2]) for nm in names}
    out = {nm: set() for nm in names}
    for (a, b), idx in cover.items():
        co, no = ev[a][0], ev[a][1]
        for i in idx:
            n_ = Vector(no[i])
            back = first_facing(bvh[b], co[i], -n_, POKE_REACH, n_)
            if back is None or back < POKE_TOL:
                continue
            fwd = first_facing(bvh[b], co[i], n_, POKE_REACH, n_)
            if fwd is not None and fwd < back:
                continue
            out[a].add(int(i))
    return out, ev


def highlight_pokes(name="qa_pk"):
    """static copies of the posed meshes; faces around poking vertices red (inner layer showing through)"""
    names, objs, cover = LAST_COVER
    pk, ev = poke_vertices(names, objs, cover)
    red = bpy.data.materials.get("qa_red") or bpy.data.materials.new("qa_red")
    red.diffuse_color = (1.0, 0.05, 0.02, 1)
    out = []
    for nm, o in zip(names, objs):
        co, no, polys = ev[nm]
        me = bpy.data.meshes.new(name + "_" + nm)
        me.from_pydata([tuple(c) for c in co], [], polys)
        ob = bpy.data.objects.new(name + "_" + nm, me)
        bpy.context.scene.collection.objects.link(ob)
        base = o.active_material or (o.data.materials[0] if o.data.materials else None)
        m0 = bpy.data.materials.new("qa_" + nm)
        pal = [(0.45, 0.6, 0.85, 1), (0.55, 0.8, 0.5, 1), (0.85, 0.75, 0.45, 1), (0.7, 0.55, 0.8, 1), (0.5, 0.8, 0.8, 1),
               (0.8, 0.6, 0.5, 1), (0.6, 0.6, 0.65, 1), (0.9, 0.85, 0.7, 1), (0.45, 0.5, 0.4, 1)]
        m0.diffuse_color = (0.75, 0.62, 0.55, 1) if nm == "skin" else pal[sum(map(ord, nm)) % len(pal)]
        me.materials.append(m0); me.materials.append(red)
        bad = pk[nm]
        mi = np.array([1 if any(v in bad for v in p) else 0 for p in polys], np.int32)
        me.polygons.foreach_set("material_index", mi)
        me.polygons.foreach_set("use_smooth", np.ones(len(polys), bool))
        out.append(ob)
    return out, {k: len(v) for k, v in pk.items()}


def highlight(rig, pieces, body=None, name="qa_hi"):
    """bake the posed meshes into static copies; faces that intersect another piece (or the skin) turn red.
    Returns the copies (the originals are hidden from renders)."""
    names = list(pieces); objs = [pieces[n] for n in names]
    if body is not None:
        names.append("skin"); objs.append(body)
    dg = bpy.context.evaluated_depsgraph_get()
    data = {}
    for nm, o in zip(names, objs):
        co, polys = evaluated(o, dg)
        data[nm] = (co, polys, BVHTree.FromPolygons(co, polys))
    bad = {nm: set() for nm in names}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            for fa, fb in data[names[i]][2].overlap(data[names[j]][2]):
                bad[names[i]].add(fa); bad[names[j]].add(fb)
    red = bpy.data.materials.get("qa_red") or bpy.data.materials.new("qa_red")
    red.diffuse_color = (1.0, 0.05, 0.02, 1)
    out = []
    for nm, o in zip(names, objs):
        co, polys, _ = data[nm]
        me = bpy.data.meshes.new(name + "_" + nm)
        me.from_pydata([c[:] for c in co], [], polys)
        ob = bpy.data.objects.new(name + "_" + nm, me)
        bpy.context.scene.collection.objects.link(ob)
        base = o.active_material or (o.data.materials[0] if o.data.materials else None)
        m0 = bpy.data.materials.new("qa_" + nm); m0.diffuse_color = base.diffuse_color if base else (0.7, 0.7, 0.7, 1)
        me.materials.append(m0); me.materials.append(red)
        mi = np.zeros(len(polys), np.int32)
        mi[list(bad[nm])] = 1
        me.polygons.foreach_set("material_index", mi)
        me.polygons.foreach_set("use_smooth", np.ones(len(polys), bool))
        o.hide_render = True
        out.append(ob)
    return out, {k: len(v) for k, v in bad.items()}


QA_RENDER_POSES = ["rest", "arms_up", "squat", "deep_squat", "walk", "sitting", "high_kick", "elbow_wrist"]


def qa_export(kind, outfit, spec, step=3, renders=True):
    """penetration gate on the exported dressed file (exactly the GLB content): every stress / outfit pose and every
    `step`-th frame of every clip; pairs = piece x piece and piece x visible skin (hands, face, neck)"""
    rig = bpy.data.objects["rts_" + kind]
    pieces = {o.name[len(kind) + 1:]: o for o in children_meshes(rig) if o.get("rts_outfit") or o.get("rts_prop")}
    layers = {n: int(o.get("rts_layer", 95 if o.get("rts_prop") else 50)) for n, o in pieces.items()}
    props = tuple(n for n, o in pieces.items() if o.get("rts_prop"))
    body = bpy.data.objects.get(kind + "_body")
    poses = PZ.STRESS + PZ.EXTRA
    clips = [c for c in json.loads(rig.get("rts_clips", "{}"))]
    res = penetration(rig, pieces, body, poses=poses, clips=clips, step=step, tag="%s %s" % (outfit, kind),
                      layers=layers, props=props)
    res["outfit"], res["kind"] = outfit, kind
    os.makedirs(RENC, exist_ok=True)
    out = os.path.join(RENC, "qa_%s_%s.json" % (outfit, kind))
    json.dump(res, open(out, "w"), indent=1)
    clog("qa json", out)
    if renders:
        render_highlights(rig, pieces, body, "qa_%s_%s" % (outfit, kind), QA_RENDER_POSES)
    return res


def render_highlights(rig, pieces, body, tag, poses, views=("34", "back34")):
    """clay renders with the intersecting faces in red, for the poses in `poses`"""
    render_setup("BLENDER_WORKBENCH", res=(640, 900))
    sc = bpy.context.scene
    sc.display.shading.light = 'STUDIO'; sc.display.shading.color_type = 'MATERIAL'; sc.display.shading.show_cavity = True
    sc.view_settings.view_transform = 'Standard'
    hidden = []
    for o in bpy.data.objects:
        if o.type == 'MESH' and not o.hide_render:
            hidden.append(o)
    cams = {"34": ((2.2, -3.2, 1.35), (0, 0, 0.9)), "back34": ((-2.2, 3.2, 1.35), (0, 0, 0.9)),
            "front": ((0, -3.9, 1.0), (0, 0, 0.9))}
    files = []
    for pose in poses:
        if rig.animation_data:
            rig.animation_data.action = None
        PZ.apply(rig, pose)
        for o in hidden:
            o.hide_render = True
        hi, cnt = highlight_pokes() if LAST_COVER else highlight(rig, pieces, body)
        for v in views:
            camera(*cams[v], lens=50)
            f = os.path.join(RENC, "%s_%s_%s.png" % (tag, pose, v))
            render(f); files.append(f)
        for o in hi:
            bpy.data.objects.remove(o, do_unlink=True)
    for o in hidden:
        o.hide_render = False
    pose_reset(rig)
    return files


# ------------------------------------------------------------------------------------------------ baked-outfit culling
def cull_hidden_layers(rig, pieces, layers, poses=None, clips=None, step=4, reach=0.035, erode=1, verbose=True):
    """LOD0 of a baked outfit (the dressed GLB; the per-piece kit GLBs stay whole): faces of an inner layer that an
    outer layer covers in EVERY sampled frame (rest, the stress / outfit poses, every `step`-th clip frame) are
    deleted (eroded by `erode` rings): e.g. the shirt torso and sleeves under a tunic / gambeson. Nothing that can
    ever show goes, and what is gone can no longer poke through. Covered = the vertex normal ray hits an outer
    layer's outward-facing surface within `reach`. Returns {piece: faces removed}."""
    names = sorted(pieces, key=lambda n: layers.get(n, 50))
    frames = [(None, None, 0)] + [(p, None, 0) for p in (poses or [])]
    for c in clips or []:
        act = bpy.data.actions.get(c)
        if act:
            frames += [(None, c, f) for f in range(0, int(round(act.frame_range[1])) + 1, step)]
    cov = {n: None for n in names}
    for pose, clip, f in frames:
        _set_frame(rig, pose, clip, f)
        dg = bpy.context.evaluated_depsgraph_get()
        ev = {n: _eval_arrays(pieces[n], dg) for n in names}
        for k, n in enumerate(names):
            outer = [m for m in names[k + 1:] if layers.get(m, 50) > layers.get(n, 50)]
            if not outer:
                cov[n] = np.zeros(len(ev[n][0]), bool)
                continue
            V, P_ = [], []
            for m in outer:
                o = len(V)
                V += [Vector(v) for v in ev[m][0]]
                P_ += [tuple(i + o for i in p) for p in ev[m][2]]
            bvh = BVHTree.FromPolygons(V, P_)
            co, no, _ = ev[n]
            cur = cov[n] if cov[n] is not None else np.ones(len(co), bool)
            for i in np.nonzero(cur)[0]:
                n_ = Vector(no[i])
                if first_facing(bvh, co[i] + no[i] * 2e-4, n_, reach, n_) is None:
                    cur[i] = False
            cov[n] = cur
    _set_frame(rig, None, None, 0)
    removed = {}
    for n in names:
        o = pieces[n]
        c = cov[n]
        if c is None or not c.any():
            continue
        me = o.data
        nb = [set() for _ in range(len(me.vertices))]
        for e in me.edges:
            a, b = e.vertices
            nb[a].add(b); nb[b].add(a)
        for _ in range(erode):
            c = np.array([c[i] and all(c[j] for j in nb[i]) for i in range(len(c))])
        sel = np.array([all(c[v] for v in p.vertices) for p in me.polygons])
        if not sel.any():
            continue
        for ob in bpy.context.selected_objects:
            ob.select_set(False)
        bpy.context.view_layer.objects.active = o; o.select_set(True)
        me.polygons.foreach_set("select", sel)
        me.vertices.foreach_set("select", np.zeros(len(me.vertices), bool))
        me.edges.foreach_set("select", np.zeros(len(me.edges), bool))
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_mode(type='FACE')
        bpy.ops.object.mode_set(mode='OBJECT')
        me.polygons.foreach_set("select", sel)
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.delete(type='FACE')
        bpy.ops.mesh.select_all(action='SELECT')
        bpy.ops.mesh.delete_loose()
        bpy.ops.object.mode_set(mode='OBJECT')
        removed[n] = int(sel.sum())
        if verbose:
            clog("cull %-22s %5d / %5d faces never visible (%d frames sampled)" % (n, sel.sum(), len(sel), len(frames)))
    return removed
