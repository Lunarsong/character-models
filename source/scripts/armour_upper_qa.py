"""QA for the knight's UPPER armour (iteration 2): rigidity, layering / penetration, head-vs-armour, sizing.

Two ways to run (Blender, headless):
  dev     Blender -b out/base_<kind>.blend --python-exit-code 1 -P scripts/armour_upper_qa.py -- dev <kind> [checks]
          fast in-memory assembly: the upper pieces GENERATED on this body and bound straight to the rig (+ helper
          joints, plume bones), the lower-armour pieces + sword / shield (armour_lower.load_lower / add_props), the
          knight clips (knight_anim.make_clips + armour_upper.post_clips). Nothing is written but renders / json.
  knight  Blender -b out/knight_<kind>_export.blend --python-exit-code 1 -P scripts/armour_upper_qa.py -- knight <kind> [checks]
          the same checks on the exported knight (what is in the GLB).
checks (default: all): rigid self intersect head size render
  rigid      every plate component (island weighted 100 % to one bone) in every test pose: Kabsch residual (mm) and
             the smallest singular value of its best affine map (strain); gate residual < 3 mm, minSV > 0.97
  self       articulated parts inside one piece (parts on different bones): overlaps per pose + bind clearance
  intersect  triangle-triangle intersections per piece pair: bind pose (budget 0 for plate/plate, plate/mail and
             plate/cloth pairs of the upper set, except the listed intended contacts) and every 3rd clip frame +
             extreme poses (per-pair budgets in MOTION_BUDGET)
  head       bare-head look: head skin vs gorget / mail and helmet vs gorget over jawOpen 0 / 0.5 / 1 x pitch -20..+25
             x turn -45 / 0 / 45 (budget 0 against the gorget)
  size       helm / pauldron / cuff sizes against the head / shoulders / wrist of the body (female sizing, M14)
  render     clay + textured close-ups of the rest pose and of the worst poses
Writes renders/armour_upper_qa_<kind>.json (+ renders/armour_upper_qa_<kind>_*.png).
"""
import bpy, os, sys, json, math, time
import numpy as np
from mathutils import Vector, Matrix
from mathutils.bvhtree import BVHTree

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
ARGV = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
sys.argv = [sys.argv[0], "--", "none"]            # armour_upper parses argv at import
import chr_lib as C
import armour_upper as AU
import armour_upper_rig as RG

UPPER = ["helmet", "plume", "gorget", "cuirass", "pauldron_l", "pauldron_r", "rerebrace_l", "rerebrace_r", "couter_l",
         "couter_r", "vambrace_l", "vambrace_r", "gauntlet_l", "gauntlet_r", "mail"]
CLOTH = ("tabard", "cape", "belts")
INTENDED = {("helmet", "plume"), ("gauntlet_r", "sword"), ("gauntlet_l", "shield")}   # contacts by design
MOTION_BUDGET = {"default": 40, "pauldron": 60, "couter": 40, "gorget": 20, "helmet": 20}
REST_TOL_MM = 3.0


def log(*a):
    print("UQA", *a, flush=True)


def pname(o, kind):
    return o.name[len(kind) + 1:] if o.name.startswith(kind + "_") else o.name.replace("L_", "")


# =========================================================================================== dev assembly
def dev_assembly(kind, lower=True):
    rig = bpy.data.objects["rts_" + kind]; bm = bpy.data.objects[kind + "_body"]
    C.pose_reset(rig)
    RG.ensure_helpers(rig)
    AU.ensure_plume_bones(rig)
    cx = AU.Ctx(kind)
    cx.topo = AU.load_topo(UPPER)
    mbs = AU.generate(cx, UPPER)
    AU.ensure_plume_bones(rig, AU.plume_bone_specs(cx.shared["plume_spine"]))
    objs = {}
    for n, mb in mbs.items():
        ob = mb.to_object("%s_%s" % (kind, n))
        ob.parent = rig
        md = ob.modifiers.new("Armature", 'ARMATURE'); md.object = rig
        ob["rts_part"] = n
        mat = AU.make_material(n)
        ob.data.materials.append(mat)
        AU.ensure_ao_attr(ob)
        objs[n] = ob
    # hide the skin each piece covers (the asset's MPFB delete groups do this in the real pipeline): no skin poking
    # through the glove / mail in the dev renders and tests
    hide = set()
    for n, ob in objs.items():
        hide |= set(AU.covered_verts(cx, ob, n, tags=mbs[n].tag))
    if hide:
        g = bm.vertex_groups.get("rts_qa_hide") or bm.vertex_groups.new(name="rts_qa_hide")
        g.add(sorted(hide), 1.0, 'REPLACE')
        md = bm.modifiers.new("rts_qa_hide", 'MASK'); md.vertex_group = g.name; md.invert_vertex_group = True
        log("dev: %d covered skin verts hidden" % len(hide))
    if lower:
        import armour_lower as AL
        AL.load_lower(rig, bm, props=False)
        AL.add_props(rig, which=("sword", "shield"))
    import knight_anim as KA
    clips = KA.make_clips(rig, kind)
    rig["rts_clips"] = json.dumps(clips)
    AU.post_clips(rig, kind)
    C.pose_reset(rig)
    return rig, bm, objs, cx


# =========================================================================================== poses
def _sg(s):
    return 1 if s == "_l" else -1


def extreme_poses(rig):
    import knight_qa as KQ
    return dict(KQ.EXTREME)


def test_poses(rig, step=3, extremes=True):
    """(name, setter) over bind + every `step`-th frame of every body clip + extreme poses"""
    out = [("bind", lambda: (setattr(rig.animation_data, "action", None) if rig.animation_data else None,
                             C.pose_reset(rig)))]
    table = json.loads(rig.get("rts_clips", "{}"))
    for clip in [c for c in table if c != "talk_emote" and bpy.data.actions.get(c)]:
        nf = int(round(bpy.data.actions[clip].frame_range[1]))
        for f in range(0, nf + 1, step):
            def st(clip=clip, f=f):
                rig.animation_data.action = bpy.data.actions[clip]
                bpy.context.scene.frame_set(f)
                bpy.context.view_layer.update()
            out.append(("%s_f%02d" % (clip, f), st))
    if extremes:
        for n, fn in extreme_poses(rig).items():
            def st(fn=fn):
                if rig.animation_data:
                    rig.animation_data.action = None
                C.pose_reset(rig); fn(rig)
            out.append((n, st))
    return out


def eval_co(o, dg):
    ev = o.evaluated_get(dg); me = ev.to_mesh()
    co = np.empty(len(me.vertices) * 3); me.vertices.foreach_get("co", co)
    polys = [tuple(p.vertices) for p in me.polygons]
    ev.to_mesh_clear()
    M = np.array(o.matrix_world)
    return co.reshape(-1, 3) @ M[:3, :3].T + M[:3, 3], polys


# =========================================================================================== rigid
def islands_single_bone(o):
    names = {g.index: g.name for g in o.vertex_groups}
    isl = AU.islands(o.data)
    out = []
    for ix in isl:
        bones = set()
        ok = True
        for i in ix:
            gs = [(names[g.group], g.weight) for g in o.data.vertices[int(i)].groups if g.weight > 1e-3]
            if len(gs) != 1:
                ok = False
                break
            bones.add(gs[0][0])
        if ok and len(bones) == 1 and len(ix) >= 4:
            out.append((ix, bones.pop()))
    return out


def kabsch_residual(A, B):
    """(max rigid-fit residual m, ratio, smallest singular value of the best linear map in the component's own
    principal axes) - the same measure as knight_qa strain (rig_helpers.kabsch): thin / degenerate components (a
    culled lame, a rivet ring) have no thickness axis to measure"""
    try:
        import rig_helpers as RH
        res, smin = RH.kabsch(A, B)
        return float(res), float(smin), float(smin)
    except Exception:
        pass
    a0, b0 = A - A.mean(0), B - B.mean(0)
    U, S, Vt = np.linalg.svd(b0.T @ a0)
    D = np.eye(3); D[2, 2] = np.sign(np.linalg.det(U @ Vt))
    R = U @ D @ Vt
    res = np.linalg.norm(a0 @ R.T - b0, axis=1).max()
    # strain: best affine map singular values
    X, *_ = np.linalg.lstsq(np.c_[a0, np.ones(len(a0))], b0, rcond=None)
    sv = np.linalg.svd(X[:3])[1]
    return float(res), float(sv.min() / max(sv.max(), 1e-9)), float(sv.min())


def check_rigid(rig, objs, poses):
    rest = {}
    C.pose_reset(rig)
    dg = bpy.context.evaluated_depsgraph_get()
    comps = {}
    for n, o in objs.items():
        comps[n] = islands_single_bone(o)
        rest[n] = eval_co(o, dg)[0]
    worst = {n: dict(res_mm=0.0, minSV=1.0, pose=None, comps=len(comps[n])) for n in objs}
    for pn, st in poses:
        st()
        dg = bpy.context.evaluated_depsgraph_get()
        for n, o in objs.items():
            if not comps[n]:
                continue
            co = eval_co(o, dg)[0]
            for ix, bone in comps[n]:
                r, ratio, smin = kabsch_residual(rest[n][ix], co[ix])
                w = worst[n]
                if r * 1000 > w["res_mm"]:
                    w["res_mm"] = round(r * 1000, 3); w["pose"] = pn; w["bone"] = bone
                w["minSV"] = round(min(w["minSV"], smin), 4)
    ok = all(w["res_mm"] < REST_TOL_MM and w["minSV"] > 0.97 for w in worst.values() if w["comps"])
    return dict(pieces=worst, gate="residual < %.0f mm and minSV > 0.97 for every rigid component" % REST_TOL_MM, ok=ok)


# =========================================================================================== intersect
def pair_kind(a, b):
    ka = a.rsplit("_", 1)[0] if a.endswith(("_l", "_r")) else a
    kb = b.rsplit("_", 1)[0] if b.endswith(("_l", "_r")) else b
    return ka, kb


def check_intersect(rig, meshes, poses, kind, upper_names):
    names = list(meshes)
    res = {}
    per_pose = {}
    for pn, st in poses:
        st()
        dg = bpy.context.evaluated_depsgraph_get()
        bvh = {}
        for n in names:
            co, polys = eval_co(meshes[n], dg)
            bvh[n] = BVHTree.FromPolygons([Vector(p) for p in co], polys)
        tot = 0
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a, b = names[i], names[j]
                if a not in upper_names and b not in upper_names:
                    continue
                k = len(bvh[a].overlap(bvh[b]))
                if k:
                    key = "%s|%s" % (a, b)
                    r = res.setdefault(key, {"bind": 0, "max": 0, "at": None, "poses": 0})
                    r["poses"] += 1
                    if pn == "bind":
                        r["bind"] = k
                    if k > r["max"]:
                        r["max"], r["at"] = k, pn
                    tot += k
        per_pose[pn] = tot
    # budgets
    fails = []
    cross = []
    for key, r in res.items():
        a, b = key.split("|")
        if (a, b) in INTENDED or (b, a) in INTENDED:
            r["budget"] = "intended contact"
            continue
        if not (a in upper_names and b in upper_names):
            # one side is a lower-armour piece / prop (authored by the lower-armour module over the upper assets): reported,
            # gated by that module's QA and knight_qa
            r["budget"] = "cross-set (report)"
            cross.append("%s bind %d max %d" % (key, r["bind"], r["max"]))
            continue
        r["bind_budget"] = 0
        ka, kb = pair_kind(a, b)
        bud = max(MOTION_BUDGET.get(ka, MOTION_BUDGET["default"]), MOTION_BUDGET.get(kb, MOTION_BUDGET["default"]))
        r["motion_budget"] = bud
        if r["bind"] > 0:
            fails.append("%s bind %d" % (key, r["bind"]))
        elif r["max"] > bud:
            fails.append("%s max %d at %s (budget %d)" % (key, r["max"], r["at"], bud))
    up_bind = sum(r["bind"] for k, r in res.items() if r.get("budget") not in ("intended contact", "cross-set (report)"))
    out = dict(pairs=dict(sorted(res.items(), key=lambda kv: -kv[1]["max"])), total_per_pose=per_pose,
               poses=len(per_pose), fails=fails, cross_set=cross, upper_bind_total=up_bind, ok=not fails)
    return out


# =========================================================================================== self (articulated plates)
def bone_parts(o):
    """{bone: [triangle (a, b, c)]} for the single-bone plate parts of piece `o` (glove / mail excluded)"""
    names = {g.index: g.name for g in o.vertex_groups}
    main = []
    for v in o.data.vertices:
        gs = [(g.weight, names[g.group]) for g in v.groups if g.weight > 1e-3]
        main.append(max(gs)[1] if len(gs) == 1 else None)
    out = {}
    for p in o.data.polygons:
        bs = {main[i] for i in p.vertices}
        if len(bs) != 1 or None in bs:
            continue
        b = bs.pop()
        vs = list(p.vertices)
        for k in range(1, len(vs) - 1):
            out.setdefault(b, []).append((vs[0], vs[k], vs[k + 1]))
    return out


def check_self(rig, objs, poses):
    """Articulated plates inside one piece (pauldron cop / lames, couter-less cannons, gauntlet lames and finger
    scales, gorget lames, breastplate / plackart): tri-tri overlaps between parts on DIFFERENT bones per pose, and the
    minimum clearance at bind (target >= 3 mm for the pauldron / vambrace / gorget / cuirass steps)"""
    parts = {n: bone_parts(o) for n, o in objs.items() if n not in ("mail", "plume")}
    res = {}
    for pn, st in poses:
        st()
        dg = bpy.context.evaluated_depsgraph_get()
        for n, o in objs.items():
            if n not in parts or len(parts[n]) < 2:
                continue
            co, _ = eval_co(o, dg)
            V = [Vector(p) for p in co]
            trees = {b: BVHTree.FromPolygons(V, T) for b, T in parts[n].items()}
            bl = sorted(trees)
            for i in range(len(bl)):
                for j in range(i + 1, len(bl)):
                    k = len(trees[bl[i]].overlap(trees[bl[j]]))
                    key = "%s:%s|%s" % (n, bl[i], bl[j])
                    r = res.setdefault(key, {"bind": 0, "max": 0, "at": None})
                    if pn == "bind":
                        r["bind"] = k
                        vb = sorted({x for t in parts[n][bl[j]] for x in t})
                        d = min(trees[bl[i]].find_nearest(V[x])[3] for x in vb)
                        r["clear_mm"] = round(d * 1000, 2)
                    if k > r["max"]:
                        r["max"], r["at"] = k, pn
    res = {k: v for k, v in res.items() if v["max"] or v.get("clear_mm", 99) < 3.0}
    summ = {}
    for k, v in res.items():
        n = k.split(":")[0]
        s_ = summ.setdefault(n, {"pairs": 0, "bind": 0, "max_sum": 0})
        s_["pairs"] += 1; s_["bind"] += v["bind"]; s_["max_sum"] += v["max"]
    plate_ok = all(v["bind"] == 0 for k, v in res.items() if not k.startswith("gauntlet"))
    return dict(pairs=dict(sorted(res.items(), key=lambda kv: -kv[1]["max"])), summary=summ, ok=plate_ok,
                gate="0 bind overlaps between articulated parts of every plate piece (gauntlet finger scales reported)")


# =========================================================================================== head
def check_head(rig, bm, meshes, kind):
    """bare look: head skin (head / jaw weighted faces) vs gorget + mail; helm look: helmet vs gorget"""
    kb = bm.data.shape_keys.key_blocks if bm.data.shape_keys else {}
    jaw = kb.get("jawOpen") if kb else None
    headw = {g.index for g in bm.vertex_groups if g.name in ("head", "jaw") or g.name.startswith(("lip_", "tongue", "cheek", "nose", "mouth"))}
    bi = bm.vertex_groups["body"].index if "body" in bm.vertex_groups else None
    hv = np.array([sum(g.weight for g in v.groups if g.group in headw) > 0.5 and (bi is None or any(g.group == bi for g in v.groups))
                   for v in bm.data.vertices])
    masks = [m for m in bm.modifiers if m.type == 'MASK']
    vis = [m.show_viewport for m in masks]
    for m in masks:
        m.show_viewport = False
    res = {}
    worst = {}
    targets = [n for n in ("gorget", "mail") if n in meshes]
    for pitch in (-20, -10, 0, 12, 25):
        for turn in (-45, 0, 45):
            for jv in (0.0, 0.5, 1.0):
                if rig.animation_data:
                    rig.animation_data.action = None
                AU.pose_head(rig, pitch, turn)
                if jaw is not None:
                    jaw.value = jv
                dg = bpy.context.evaluated_depsgraph_get()
                co, polys = eval_co(bm, dg)
                hp = [p for p in polys if all(hv[i] for i in p)]
                hb = BVHTree.FromPolygons([Vector(p) for p in co], hp)
                key = "p%+d_t%+d_j%.1f" % (pitch, turn, jv)
                for n in targets:
                    c2, p2 = eval_co(meshes[n], dg)
                    k = len(hb.overlap(BVHTree.FromPolygons([Vector(p) for p in c2], p2)))
                    if k:
                        res.setdefault(n, {})[key] = k
                    worst[n] = max(worst.get(n, 0), k)
                if "helmet" in meshes and "gorget" in meshes and AU.HELM_PITCH_MIN <= pitch <= AU.HELM_PITCH_MAX:
                    c1, p1 = eval_co(meshes["helmet"], dg); c2, p2 = eval_co(meshes["gorget"], dg)
                    k = len(BVHTree.FromPolygons([Vector(p) for p in c1], p1).overlap(BVHTree.FromPolygons([Vector(p) for p in c2], p2)))
                    if k:
                        res.setdefault("helmet|gorget", {})[key] = k
                    worst["helmet|gorget"] = max(worst.get("helmet|gorget", 0), k)
    if jaw is not None:
        jaw.value = 0.0
    C.pose_reset(rig)
    for m, v in zip(masks, vis):
        m.show_viewport = v
    return dict(worst=worst, cases=res, grid="jawOpen 0/0.5/1 x pitch -20/-10/0/12/25 x turn -45/0/45 "
                "(helmet|gorget only for the helmeted range pitch %d..%d)" % (AU.HELM_PITCH_MIN, AU.HELM_PITCH_MAX),
                ok=worst.get("gorget", 0) == 0 and worst.get("helmet|gorget", 0) == 0)


# =========================================================================================== size
def check_size(rig, bm, meshes, kind):
    cx = AU.Ctx(kind)
    out = {}
    co = cx.body.co
    head_w = 2 * cx.head_hw; head_d = cx.head_back - cx.head_front; head_h = cx.head_top - cx.chin_z
    dg = bpy.context.evaluated_depsgraph_get()
    C.pose_reset(rig)
    if "helmet" in meshes:
        h = eval_co(meshes["helmet"], dg)[0]
        out["helmet"] = dict(width=round(float(np.ptp(h[:, 0])), 4), depth=round(float(np.ptp(h[:, 1])), 4),
                             height=round(float(np.ptp(h[:, 2])), 4), head_w=round(head_w, 4), head_d=round(head_d, 4),
                             head_h=round(head_h, 4), width_ratio=round(float(np.ptp(h[:, 0])) / head_w, 3))
    for s in "lr":
        n = "pauldron_" + s
        if n in meshes:
            p = eval_co(meshes[n], dg)[0]
            S = cx.head("upperarm_" + s)
            out[n] = dict(max_r=round(float(np.linalg.norm(p - S, axis=1).max()), 4),
                          shoulder_breadth=round(float(abs(cx.head("upperarm_l")[0] - cx.head("upperarm_r")[0])), 4))
        n = "gauntlet_" + s
        if n in meshes:
            g = eval_co(meshes[n], dg)[0]
            E, W = cx.head("lowerarm_" + s), cx.head("hand_" + s)
            ax = (W - E) / np.linalg.norm(W - E)
            t = (g - E) @ ax / np.linalg.norm(W - E)
            sel = (t > 0.78) & (t < 1.05)
            r = np.linalg.norm((g - E) - np.outer((g - E) @ ax, ax), axis=1)
            wr = cx.body.co[(np.abs((cx.body.co - W) @ ax) < 0.01) & (np.linalg.norm(cx.body.co - W, axis=1) < 0.06)]
            wr_r = float(np.linalg.norm((wr - W) - np.outer((wr - W) @ ax, ax), axis=1).max()) if len(wr) else 0.0
            out[n] = dict(cuff_max_r=round(float(r[sel].max()), 4) if sel.any() else None, wrist_r=round(wr_r, 4))
    return out


# =========================================================================================== render
def render_set(rig, kind, poses_named, tag):
    import knight_qa as KQ
    sc = C.render_setup("BLENDER_WORKBENCH", res=(700, 900))
    sc.display.shading.light = 'STUDIO'; sc.display.shading.color_type = 'MATERIAL'
    sc.display.shading.show_cavity = True; sc.display.shading.cavity_type = 'BOTH'
    views = {"front": ((0, -3.3, 1.35), (0, 0, 1.2), 55), "34": ((-2.2, -2.6, 1.7), (0, 0, 1.25), 55),
             "back": ((0.5, 3.3, 1.5), (0, 0, 1.25), 55), "shoulder": ((-1.0, -0.75, 1.75), (-0.25, -0.02, 1.45), 60),
             "neck": ((-0.45, -0.8, 1.7), (0, -0.04, 1.58), 60), "hand": ((-0.75, -0.8, 1.2), (-0.45, -0.25, 1.0), 60)}
    out = []
    for pn, st in poses_named:
        st()
        vv = dict(views)
        for sd in ("l", "r"):                      # close-ups that follow the hands (grip / strap poses)
            pb = rig.pose.bones.get("hand_" + sd)
            if pb is not None:
                h = rig.matrix_world @ ((pb.head + pb.tail) * 0.5)
                sg = 1 if sd == "l" else -1
                vv["fist_" + sd] = ((h.x + 0.36 * sg, h.y - 0.46, h.z + 0.22), (h.x, h.y, h.z), 70)
        for vn, (loc, tgt, lens) in vv.items():
            C.camera(loc, tgt, lens=lens)
            f = os.path.join(C.REN, "armour_upper_qa_%s_%s_%s_%s.png" % (tag, kind, pn, vn))
            C.render(f); out.append(f)
    return out


# =========================================================================================== main
def main(args):
    mode = args[0] if args else "dev"
    kind = args[1] if len(args) > 1 else "male"
    checks = args[2:] or ["rigid", "self", "intersect", "head", "size", "render"]
    t0 = time.time()
    if mode == "dev":
        rig, bm, objs, cx = dev_assembly(kind, lower="nolower" not in checks)
    else:
        rig = bpy.data.objects["rts_" + kind]; bm = bpy.data.objects.get(kind + "_head") or bpy.data.objects.get(kind + "_body")
        objs = {pname(o, kind): o for o in C.children_meshes(rig) if o.get("rts_part") and pname(o, kind) in UPPER + ["gorget_top"]}
    meshes = {pname(o, kind): o for o in C.children_meshes(rig)
              if (o.get("rts_part") or o.get("rts_prop")) and o.get("rts_look", "any") in ("any", "helm")}
    log("pieces: upper %d, all %d (%.0fs)" % (len(objs), len(meshes), time.time() - t0))
    rep = dict(kind=kind, mode=mode)
    poses = test_poses(rig)
    if "rigid" in checks:
        rep["rigid"] = check_rigid(rig, objs, poses)
        for n, w in rep["rigid"]["pieces"].items():
            log("rigid %-12s comps %3d  residual %6.3f mm  minSV %.4f  (%s)" % (n, w["comps"], w["res_mm"], w["minSV"], w["pose"]))
        log("rigid gate:", rep["rigid"]["ok"])
    if "intersect" in checks:
        rep["intersect"] = check_intersect(rig, meshes, poses, kind, set(objs))
        for k, r in list(rep["intersect"]["pairs"].items())[:30]:
            log("isect %-26s bind %4d  max %5d at %-18s (%d poses) %s" % (k, r["bind"], r["max"], r["at"], r["poses"],
                                                                          r.get("budget", "")))
        log("intersect upper-set bind total:", rep["intersect"]["upper_bind_total"], "fails:", rep["intersect"]["fails"][:20])
        log("intersect cross-set (lower pieces / props):", rep["intersect"]["cross_set"][:20])
    if "self" in checks:
        rep["self"] = check_self(rig, objs, poses)
        log("self (articulated parts):", json.dumps(rep["self"]["summary"]), "ok", rep["self"]["ok"])
        for k, r in list(rep["self"]["pairs"].items())[:14]:
            log("self %-52s bind %4d max %4d at %-18s clear %s mm" % (k, r["bind"], r["max"], r["at"], r.get("clear_mm")))
    if "head" in checks:
        rep["head"] = check_head(rig, bm, meshes, kind)
        log("head:", rep["head"]["worst"], "ok", rep["head"]["ok"])
    if "size" in checks:
        rep["size"] = check_size(rig, bm, meshes, kind)
        log("size:", json.dumps(rep["size"]))
    if "render" in checks:
        named = [p for p in poses if p[0] in ("bind", "attack_sword_f12", "block_shield_f15", "arms_up", "walk_f09", "elbows")]
        rep["renders"] = render_set(rig, kind, named, mode)
    out = os.path.join(C.REN, "armour_upper_qa_%s%s.json" % (kind, "" if mode == "knight" else "_dev"))
    json.dump(rep, open(out, "w"), indent=1)
    log("wrote", out, "(%.0fs)" % (time.time() - t0))


if __name__ == "__main__":
    main(ARGV)
