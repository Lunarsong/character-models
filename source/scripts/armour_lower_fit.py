"""Knight LOWER armour, cloth and props: fit / layering / penetration QA measured per animation frame (user feedback
13, 14, 15, 17, 19, 21; judge M1, M2, M8-M11, M13).

  dev     Blender -b out/base_<kind>.blend --python-exit-code 1 -P scripts/armour_lower_fit.py -- dev <kind> [opts]
          The knight's stack without MakeClothes (seconds instead of minutes): the upper armour loaded as build_knight
          loads it (MPFB fit of its assets + rig_helpers' plate rules), this kit generated directly on the body over it
          (= load_lower's regen_on_body), authored rigid / chain weights, extra bones (knee / hip / arm helpers, cape /
          tabard chains, sockets), props, co-skinning, the knight's clips (knight_anim) + the post_clips hook; then it
          measures. Works on any body (female, proportion builds).
  knight  Blender -b out/knight_<kind>_export.blend --python-exit-code 1 -P scripts/armour_lower_fit.py -- knight <kind> [opts]
          The same measurements on the assembled engine knight (its baked clips + knight_qa's extreme test poses).
  opts:   clips=idle,walk,...   step=N (clip frame step, default 3)   render[=frame:view,...]   only=piece,..  hide=..
          out=<json path>   noextreme   gate (exit 1 when a gate fails)   KA_DIR=<dir> (env: stable copies of
          knight_anim / knight_qa / rig_helpers while their owners edit them)

Metrics per test frame (every `step`-th frame of every clip + the extreme poses), written to
renders/lower_fit_<kind>[_dev].json with a summary per metric:
  pairs      triangle-triangle intersections (BVHTree.overlap) between each lower piece / prop and every other piece:
             max, the frame, the local crossing depth, hot spots (3 cm bins), max per clip ('by_clip', 'bind',
             'extreme') and the count + depth of EVERY measured frame ('per_frame')
  depth      layered pairs (inner, outer): outer-piece vertices that sink under the inner piece's outward skin
             (perpendicular projection, > 1 mm): count and max depth (mm)
  rigid      every island of the plate pieces: Kabsch residual to its rest shape (mm): 0 = moves as one rigid plate
  belt_seat  waist / hip belt rings: per 15 deg sector around the ring the smallest gap from the belt to the layer it
             is buckled over (tabard / cuirass / skirt / mail): > 6 mm = floating
  shield     tilt of the shield's long axis from vertical (deg) in the carry clips (idle / walk / run)
  grip       closed-fist check: distance of the curled finger bones from the sword grip axis (mm)
"""
import bpy, os, sys, json, math, time
import numpy as np
from mathutils import Vector, Matrix
from mathutils.bvhtree import BVHTree

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
if os.environ.get("KA_DIR"):                 # dev only: a stable copy of knight_anim / knight_qa while their owner edits
    sys.path.insert(0, os.environ["KA_DIR"])
import chr_lib as C

LOWER = ("legs_mail", "boots", "greaves", "sabatons", "poleyns", "cuisses", "mail_skirt", "tassets", "tabard", "belts",
         "cape", "clasps")
PROPS = ("sword", "shield", "scabbard")
RIGID = ("cuisses", "poleyns", "greaves", "sabatons", "tassets", "clasps", "sword", "shield")
# (inner, outer): the outer piece must stay outside the inner one's outward skin where they overlap
LAYERS = [("legs_mail", "cuisses"), ("legs_mail", "poleyns"), ("legs_mail", "greaves"), ("cuisses", "poleyns"),
          ("greaves", "poleyns"), ("boots", "sabatons"), ("boots", "greaves"), ("greaves", "sabatons"),
          ("legs_mail", "mail_skirt"), ("mail", "mail_skirt"), ("cuisses", "mail_skirt"), ("mail_skirt", "tassets"),
          ("cuisses", "tassets"), ("cuirass", "tabard"), ("mail", "tabard"), ("mail_skirt", "tabard"),
          ("tabard", "belts"), ("cuirass", "belts"), ("mail_skirt", "belts"), ("tassets", "belts"),
          ("cuirass", "cape"), ("tabard", "cape"), ("mail", "cape"), ("pauldron_l", "cape"), ("pauldron_r", "cape"),
          ("gorget", "tabard"), ("cape", "clasps"), ("pauldron_l", "clasps"), ("pauldron_r", "clasps")]
# layer the belts are buckled over (belt_seat)
UNDER_BELT = ("tabard", "cuirass", "mail_skirt", "mail", "tassets", "legs_mail", "underlayer")
# body axis segments (bone head -> tail) that define 'outward' for a face
AXIS_BONES = ("pelvis", "spine_01", "spine_02", "spine_03", "spine_04", "spine_05", "neck_01", "neck_02", "head",
              "thigh_l", "thigh_r", "calf_l", "calf_r", "foot_l", "foot_r", "ball_l", "ball_r", "clavicle_l",
              "clavicle_r", "upperarm_l", "upperarm_r", "lowerarm_l", "lowerarm_r", "hand_l", "hand_r")
GATES = {"rigid_mm": 0.5, "belt_gap_mm": 6.0, "shield_tilt_deg": 12.0}


def log(*a):
    print("FIT", *a, flush=True)


# ============================================================================================ scene access
def meshes(rig, look="helm"):
    out = {}
    for o in bpy.data.objects:
        if o.type != 'MESH' or o.get("rts_probe") or not o.get("rts_part"):
            continue
        if not any(m.type == 'ARMATURE' and m.object == rig for m in o.modifiers):
            continue
        lk = o.get("rts_look", "any")
        if lk not in ("any", look):
            continue
        out[o["rts_part"]] = o
    return out


def evaluated(o, dg):
    ev = o.evaluated_get(dg); me = ev.to_mesh()
    co = np.empty(len(me.vertices) * 3); me.vertices.foreach_get("co", co)
    ev.to_mesh_clear()
    M = np.array(o.matrix_world)
    return co.reshape(-1, 3) @ M[:3, :3].T + M[:3, 3]


def polys_of(o):
    return [tuple(p.vertices) for p in o.data.polygons]


def islands(o):
    n = len(o.data.vertices)
    par = list(range(n))

    def f(i):
        while par[i] != i:
            par[i] = par[par[i]]; i = par[i]
        return i
    for e in o.data.edges:
        a, b = f(e.vertices[0]), f(e.vertices[1])
        if a != b:
            par[a] = b
    groups = {}
    for i in range(n):
        groups.setdefault(f(i), []).append(i)
    return [np.array(g) for g in groups.values()]


def kabsch_residual(A, B):
    """max distance after the best rigid fit of rest points A onto posed points B"""
    ca, cb = A.mean(0), B.mean(0)
    H = (A - ca).T @ (B - cb)
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1, 1, d])
    R = Vt.T @ D @ U.T
    return float(np.linalg.norm((A - ca) @ R.T + cb - B, axis=1).max())


def bake_helper_keys(rig, clips):
    """key the plate helper joints (rig_helpers rule) into clips that were made without them (dev harness with a
    knight_anim that predates rig_helpers); the real pipeline's knight_anim keys them itself"""
    if not rig.get("rts_helpers"):
        return
    import rig_helpers as RH
    spec = [h for h in json.loads(rig["rts_helpers"]).get("helpers", []) if h["bone"] in rig.pose.bones]
    sc = bpy.context.scene
    for c in clips:
        act = bpy.data.actions.get(c)
        if act is None:
            continue
        rig.animation_data.action = act
        nf = int(round(act.frame_range[1])) + 1
        vals = {h["bone"]: [] for h in spec}
        for f in range(nf):
            sc.frame_set(f)
            for h in spec:
                q = rig.pose.bones[h["source"]].matrix_basis.to_quaternion()
                vals[h["bone"]].append(RH.helper_basis(q, h["weight"], h.get("swing_only", True)))
        for b, qs in vals.items():
            path = 'pose.bones["%s"].rotation_quaternion' % b
            for fc in [fc for fc in _fcurves(act) if fc.data_path == path]:
                fc.keyframe_points.clear()
                for f, q in enumerate(qs):
                    fc.keyframe_points.insert(f, q[fc.array_index], options={'FAST'})
                fc.update()
    log("helper joints keyed into %s" % clips)


def _fcurves(act):
    try:
        return list(act.fcurves)
    except Exception:
        out = []
        for layer in act.layers:
            for strip in layer.strips:
                for bag in strip.channelbags:
                    out += list(bag.fcurves)
        return out


def near_bones(rig, p, k=2):
    """the k posed bones nearest to point p (labels for where a crossing is)"""
    d = []
    for pb in rig.pose.bones:
        if pb.name.startswith(("socket_", "cape_", "tabard_")) or not rig.data.bones[pb.name].use_deform:
            continue
        a = np.array(rig.matrix_world @ pb.head); b = np.array(rig.matrix_world @ pb.tail)
        t = np.clip(np.dot(p - a, b - a) / max(np.dot(b - a, b - a), 1e-9), 0, 1)
        d.append((float(np.linalg.norm(p - (a + t * (b - a)))), pb.name))
    return [n for _, n in sorted(d)[:k]]


def drive_helpers(rig):
    """pose the plate helper joints from their sources (rig_helpers rule) for static poses and clips made without
    them (the knight's clips have them keyed: same values)"""
    if rig.get("rts_helpers"):
        import rig_helpers as RH
        RH.drive_helpers(rig)


def crossing_depth(ca, pa, cb, pb, ov):
    """local penetration depth of a set of crossing polygon pairs: per pair, the deepest vertex of each polygon on the
    minority side of the other's plane (the tip that went through), the smaller of the two; max over the pairs"""
    best = 0.0
    for i, j in ov:
        A = ca[list(pa[i])]; Bq = cb[list(pb[j])]
        dd = []
        for P, Q in ((A, Bq), (Bq, A)):
            n = np.cross(Q[1] - Q[0], Q[2] - Q[0]); ln = np.linalg.norm(n)
            if ln < 1e-12:
                dd.append(0.0); continue
            s = (P - Q[0]) @ (n / ln)
            pos, neg = s[s > 0], -s[s < 0]
            side = pos if len(pos) <= len(neg) else neg
            dd.append(float(side.max()) if len(side) else 0.0)
        best = max(best, min(dd))
    return best


def axis_segments(rig):
    segs = []
    for n in AXIS_BONES:
        pb = rig.pose.bones.get(n)
        if pb is None:
            continue
        M = rig.matrix_world
        segs.append((np.array(M @ pb.head), np.array(M @ pb.tail)))
    return np.array([s[0] for s in segs]), np.array([s[1] for s in segs])


def outward_faces(co, polys, segA, segB):
    """faces whose normal points away from the nearest body-axis segment"""
    tri = [p[:3] for p in polys]
    P0 = co[[t[0] for t in tri]]; P1 = co[[t[1] for t in tri]]; P2 = co[[t[2] for t in tri]]
    Nf = np.cross(P1 - P0, P2 - P0)
    cen = np.array([co[list(p)].mean(0) for p in polys])
    D = segB - segA
    L2 = np.maximum((D * D).sum(1), 1e-9)
    t = np.clip(((cen[:, None, :] - segA[None]) * D[None]).sum(2) / L2[None], 0, 1)
    Q = segA[None] + t[..., None] * D[None]
    d2 = ((cen[:, None, :] - Q) ** 2).sum(2)
    k = np.argmin(d2, 1)
    q = Q[np.arange(len(cen)), k]
    return ((Nf * (cen - q)).sum(1) > 0)


# ============================================================================================ frames
def test_frames(rig, clips=None, step=3, extreme=True):
    import knight_qa as QA
    out = [("bind", lambda: QA.play(rig, None, 0), None, 0)]
    table = json.loads(rig.get("rts_clips", "{}"))
    for c in table:
        if c == "talk_emote" or not bpy.data.actions.get(c) or (clips and c not in clips):
            continue
        nf = int(round(bpy.data.actions[c].frame_range[1]))
        for f in range(0, nf + 1, step):
            out.append(("%s_f%02d" % (c, f), (lambda c=c, f=f: QA.play(rig, c, f)), c, f))
    if extreme:
        for n, fn in QA.EXTREME.items():
            def st(fn=fn):
                QA.play(rig, None, 0); fn(rig); bpy.context.view_layer.update()
            out.append((n, st, None, 0))
    return out


# ============================================================================================ measurement
def measure(rig, kind, clips=None, step=3, extreme=True, look="helm"):
    objs = meshes(rig, look)
    names = sorted(objs)
    mine = [n for n in names if n in LOWER or n in PROPS]
    polys = {n: polys_of(objs[n]) for n in names}
    isl = {n: [g for g in islands(objs[n]) if len(g) >= 4] for n in names if n in RIGID}
    layers = [(a, b) for a, b in LAYERS if a in objs and b in objs]
    rest = {}
    res = {"pairs": {}, "depth": {}, "rigid": {}, "belt_seat": {}, "shield": {}, "grip": {}, "per_frame": {}}
    belt_rings = []
    if "belts" in objs:
        import armour_lower as AL
        bo = objs["belts"]
        bco = np.array([bo.matrix_world @ v.co for v in bo.data.vertices])
        belt_rings = sorted(AL.ring_islands(bco, islands(bo)), key=lambda g: -bco[g][:, 2].mean())   # ring0 = waist
    t0 = time.time()
    frames = test_frames(rig, clips, step, extreme)
    for fname, setter, clip, f in frames:
        setter()
        drive_helpers(rig)
        dg = bpy.context.evaluated_depsgraph_get()
        co = {n: evaluated(objs[n], dg) for n in names}
        if fname == "bind":
            rest = {n: co[n].copy() for n in names}
        bb = {n: (co[n].min(0) - 0.002, co[n].max(0) + 0.002) for n in names}
        bvh = {}

        def B(n):
            if n not in bvh:
                bvh[n] = BVHTree.FromPolygons([Vector(v) for v in co[n]], polys[n])
            return bvh[n]
        tot = 0
        # --- triangle-triangle intersections
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                if a not in mine and b not in mine:
                    continue
                if (bb[a][1] < bb[b][0]).any() or (bb[b][1] < bb[a][0]).any():
                    continue
                ov = B(a).overlap(B(b))
                n = len(ov)
                if n:
                    key = "%s|%s" % (a, b)
                    dmax = crossing_depth(co[a], polys[a], co[b], polys[b], ov)
                    r = res["pairs"].setdefault(key, {"max": 0, "at": None, "frames": 0, "depth_mm": 0.0, "depth_at": None,
                                                      "by_clip": {}, "per_frame": {}})
                    r["frames"] += 1; tot += n
                    ck = clip or ("bind" if fname == "bind" else "extreme")
                    r["by_clip"][ck] = max(r["by_clip"].get(ck, 0), n)
                    r["per_frame"][fname] = [n, round(dmax * 1000, 1)]
                    if n > r["max"]:
                        r["max"], r["at"] = n, fname
                        cen = np.array([co[a][list(polys[a][i])].mean(0) for i, j in ov])
                        r["where"] = [round(float(v), 3) for v in cen.mean(0)]
                        r["bones_a"] = near_bones(rig, cen.mean(0))
                        # hot spots: crossing centroids binned on a 3 cm grid, the 6 busiest bins
                        cells = {}
                        for c_ in cen:
                            k_ = tuple(np.round(c_ / 0.03).astype(int))
                            cells.setdefault(k_, []).append(c_)
                        spots = sorted(cells.values(), key=len, reverse=True)[:6]
                        r["spots"] = [[len(v_)] + [round(float(q), 3) for q in np.mean(v_, 0)] for v_ in spots]
                    if dmax * 1000 > r["depth_mm"]:
                        r["depth_mm"], r["depth_at"] = round(dmax * 1000, 1), fname
        # --- layering depth
        segA, segB = axis_segments(rig)
        for inner, outer in layers:
            if (bb[inner][1] < bb[outer][0]).any() or (bb[outer][1] < bb[inner][0]).any():
                continue
            ow = outward_faces(co[inner], polys[inner], segA, segB)
            fl = [p for p, k in zip(polys[inner], ow) if k]
            if not fl:
                continue
            bo = BVHTree.FromPolygons([Vector(v) for v in co[inner]], fl)
            depths = []
            for p in co[outer]:
                if (p < bb[inner][0]).any() or (p > bb[inner][1]).any():
                    continue
                loc, nn, _, dist = bo.find_nearest(Vector(p), 0.012)
                if loc is None:
                    continue
                s = (Vector(p) - loc).dot(nn)
                if s < -0.001 and -s >= 0.9 * dist:
                    depths.append(-s)
            if depths:
                key = "%s<%s" % (inner, outer)
                r = res["depth"].setdefault(key, {"max_mm": 0.0, "at": None, "frames": 0, "max_verts": 0})
                r["frames"] += 1
                r["max_verts"] = max(r["max_verts"], len(depths))
                if max(depths) * 1000 > r["max_mm"]:
                    r["max_mm"], r["at"] = round(max(depths) * 1000, 1), fname
        # --- rigidity
        for n, gs in isl.items():
            worst = 0.0
            for g in gs:
                worst = max(worst, kabsch_residual(rest[n][g], co[n][g]))
            r = res["rigid"].setdefault(n, {"max_mm": 0.0, "at": None})
            if worst * 1000 > r["max_mm"]:
                r["max_mm"], r["at"] = round(worst * 1000, 2), fname
        # --- belt seat
        if belt_rings:
            under = [n for n in UNDER_BELT if n in objs]
            V, P = [], []
            for n in under:
                off = len(V); V += [Vector(v) for v in co[n]]; P += [tuple(i + off for i in p) for p in polys[n]]
            bu = BVHTree.FromPolygons(V, P)
            gset = [set(int(i) for i in g) for g in belt_rings]
            for k, g in enumerate(belt_rings):
                # the other ring counts as an under layer too (the hip belt crosses over the waist belt)
                others = [q for j, q in enumerate(belt_rings) if j != k]
                if others:
                    ob = objs["belts"]; oset = set(int(i) for q in others for i in q)
                    Vb = [Vector(v) for v in co["belts"]]
                    Pb = [tuple(p.vertices) for p in ob.data.polygons if p.vertices[0] in oset]
                    bu_k = BVHTree.FromPolygons(V + Vb, P + [tuple(i + len(V) for i in f) for f in Pb])
                else:
                    bu_k = bu
                pts = co["belts"][g]
                c = pts.mean(0)
                U, S, Vt = np.linalg.svd(pts - c)
                e1, e2 = Vt[0], Vt[1]
                ang = np.degrees(np.arctan2((pts - c) @ e2, (pts - c) @ e1)) % 360
                d = np.array([bu_k.find_nearest(Vector(p), 0.1)[3] or 0.1 for p in pts])
                sec = {}
                for a_, dd in zip(ang, d):
                    s_ = int(a_ // 15)
                    sec[s_] = min(sec.get(s_, 1.0), dd)
                gap = max(sec.values()) if sec else 0.0
                key = "ring%d" % k
                r = res["belt_seat"].setdefault(key, {"max_gap_mm": 0.0, "at": None})
                if fname == "bind":
                    r["bind_sectors_mm"] = {int(a_ * 15): round(v_ * 1000, 1) for a_, v_ in sorted(sec.items())}
                if gap * 1000 > r["max_gap_mm"]:
                    r["max_gap_mm"], r["at"] = round(gap * 1000, 1), fname
                    r["worst_sector_deg"] = int(max(sec, key=sec.get) * 15)
                    wi = [i_ for i_, a_ in enumerate(ang) if int(a_ // 15) == max(sec, key=sec.get)]
                    r["worst_where"] = [round(float(q), 3) for q in pts[wi].mean(0)]
        # --- shield carry tilt
        pb = rig.pose.bones.get("socket_shield_l")
        if pb is not None and clip in ("idle", "walk", "run"):
            Y = (rig.matrix_world.to_3x3() @ pb.matrix.to_3x3()) @ Vector((0, 1, 0))
            tilt = math.degrees(Y.angle(Vector((0, 0, -1))))
            r = res["shield"].setdefault(clip, {"max_deg": 0.0, "mean_deg": 0.0, "n": 0})
            r["max_deg"] = round(max(r["max_deg"], tilt), 1)
            r["mean_deg"] = round((r["mean_deg"] * r["n"] + tilt) / (r["n"] + 1), 1); r["n"] += 1
        # --- closed grip
        pw = rig.pose.bones.get("socket_weapon_r")
        if pw is not None and clip in (None, "idle", "walk", "run"):
            if True:
                M = rig.matrix_world @ pw.matrix
                o = np.array(M.translation); ax = np.array(M.to_3x3() @ Vector((0, 1, 0)))
                ds = []
                for fi in ("index", "middle", "ring", "pinky"):
                    for j in ("02", "03"):
                        b = rig.pose.bones.get("%s_%s_r" % (fi, j))
                        if b is None:
                            continue
                        p = np.array(rig.matrix_world @ (b.head + (b.tail - b.head) * 0.5)) - o
                        ds.append(np.linalg.norm(p - ax * p.dot(ax)))
                if ds:
                    key = clip or fname
                    r = res["grip"].setdefault(key, {"min_mm": 1e9, "mean_mm": 0.0, "n": 0})
                    r["min_mm"] = round(min(r["min_mm"], min(ds) * 1000), 1)
                    r["mean_mm"] = round((r["mean_mm"] * r["n"] + np.mean(ds) * 1000) / (r["n"] + 1), 1); r["n"] += 1
        res["per_frame"][fname] = tot
    res["pairs"] = dict(sorted(res["pairs"].items(), key=lambda kv: -kv[1]["max"]))
    res["depth"] = dict(sorted(res["depth"].items(), key=lambda kv: -kv[1]["max_mm"]))
    res["frames"] = len(frames)
    log("measured %d frames x %d pieces in %.0fs" % (len(frames), len(names), time.time() - t0))
    return res


def summary(res, top=30):
    log("---- pairs (tri-tri intersections, max over frames; involving lower pieces / props; local crossing depth)")
    for k, r in list(res["pairs"].items())[:top]:
        log("   %-28s max %5d at %-18s (%3d frames)  depth %5.1f mm at %-16s near %s  by clip %s" % (
            k, r["max"], r["at"], r["frames"], r["depth_mm"], r["depth_at"], ",".join(r.get("bones_a", [])),
            " ".join("%s:%d" % kv for kv in sorted(r.get("by_clip", {}).items()))))
    log("---- layering depth (outer under inner)")
    for k, r in list(res["depth"].items())[:top]:
        log("   %-28s max %5.1f mm at %-18s (%d frames, up to %d verts)" % (k, r["max_mm"], r["at"], r["frames"], r["max_verts"]))
    log("---- rigid plates (Kabsch residual)")
    for k, r in res["rigid"].items():
        log("   %-12s %6.2f mm at %s" % (k, r["max_mm"], r["at"]))
    log("---- belt seat (worst sector gap)", {k: (r["max_gap_mm"], r["at"]) for k, r in res["belt_seat"].items()})
    log("---- shield tilt from vertical", {k: (r["mean_deg"], r["max_deg"]) for k, r in res["shield"].items()})
    log("---- grip: finger bones to grip axis (mm)", {k: (r["min_mm"], r["mean_mm"]) for k, r in res["grip"].items()})


def gates(res):
    bad = []
    for k, r in res["rigid"].items():
        if r["max_mm"] > GATES["rigid_mm"]:
            bad.append("rigid %s %.2f mm" % (k, r["max_mm"]))
    for k, r in res["belt_seat"].items():
        if r["max_gap_mm"] > GATES["belt_gap_mm"]:
            bad.append("belt %s gap %.1f mm" % (k, r["max_gap_mm"]))
    for k, r in res["shield"].items():
        if r["mean_deg"] > GATES["shield_tilt_deg"]:
            bad.append("shield %s tilt %.1f deg" % (k, r["mean_deg"]))
    return bad


# ============================================================================================ dev build
COLORS = {"plate": (0.66, 0.67, 0.70, 1), "mail": (0.30, 0.31, 0.33, 1), "leather": (0.42, 0.25, 0.12, 1),
          "cloth": (0.12, 0.22, 0.62, 1), "gold": (0.80, 0.62, 0.25, 1), "upper": (0.55, 0.56, 0.60, 1)}


def clay(name, col):
    m = bpy.data.materials.get("fit_" + name) or bpy.data.materials.new("fit_" + name)
    m.diffuse_color = col
    return m


def skinned_object(rig, name, V, F, weights, mat):
    me = bpy.data.meshes.new(name); me.from_pydata([tuple(v) for v in V], [], [tuple(f) for f in F]); me.update()
    me.polygons.foreach_set("use_smooth", np.ones(len(F), bool))
    ob = bpy.data.objects.new(name, me); bpy.context.scene.collection.objects.link(ob)
    ob.data.materials.append(mat)
    groups = {}
    for i, w in enumerate(weights):
        for b, x in (w or {}).items():
            groups.setdefault(b, []).append((i, x))
    for b, lst in groups.items():
        vg = ob.vertex_groups.get(b) or ob.vertex_groups.new(name=b)
        for i, x in lst:
            vg.add([i], float(x), 'REPLACE')
    ob.parent = rig
    md = ob.modifiers.new("Armature", 'ARMATURE'); md.object = rig
    return ob


def body_weights(rig, bm):
    """(KDTree over the rest skin, per-vertex {bone: w}) of the live body (deform bones only)"""
    from mathutils.kdtree import KDTree
    import armour_lower_geo as G
    s = G.BodySurf(bm)
    bones = {b.name for b in rig.data.bones}
    names = {g.index: g.name for g in bm.vertex_groups}
    n = G.BodySurf.NBODY
    kd = KDTree(n)
    for i in range(n):
        kd.insert(Vector(s.co[i]), i)
    kd.balance()
    W = []
    for v in bm.data.vertices[:n]:
        w = {names[g.group]: g.weight for g in v.groups if names[g.group] in bones and g.weight > 1e-3}
        t = sum(w.values()) or 1.0
        W.append({k: x / t for k, x in w.items()})
    return kd, W


def load_upper(rig, kind, skip=("plume",)):
    import armour_lower as AL
    out = {}
    man = json.load(open(os.path.join(AL.KL_ASSETS, "knight_upper_manifest.json")))
    for slot in man["slots"]:
        if slot in skip:
            continue
        d = os.path.join(AL.KL_ASSETS, "knight_" + slot)
        m = AL.upper_layer(slot)
        if m is None:
            continue
        W = json.load(open(os.path.join(d, "knight_%s.mhw" % slot)))["weights"]
        wts = [dict() for _ in m.V]
        for b, lst in W.items():
            for i, x in lst:
                if i < len(wts):
                    wts[i][b] = x
        ob = skinned_object(rig, "%s_%s" % (kind, slot), m.V, m.F, wts, clay("upper", COLORS["upper"]))
        ob["rts_part"] = slot
        ob["rts_look"] = "helm" if slot in ("helmet", "plume") else "any"
        out[slot] = ob
    return out


def load_upper_mpfb(rig, bm, kind, skip=("plume",)):
    """the upper armour as the knight build loads it (MPFB fit of the knight_<slot> assets + their .mhw weights, then
    rig_helpers' rigid plate rules, which also create the arm helper joints); clay material"""
    import armour_lower as AL
    import outfit_lib as OL
    import rig_helpers as RH
    try:
        import armour_upper_rig as AUR
        AUR.ensure_helpers(rig)
    except Exception as e:
        log("armour_upper_rig helpers unavailable: %s" % e)
    man = json.load(open(os.path.join(AL.KL_ASSETS, "knight_upper_manifest.json")))
    out = {}
    for slot in man["slots"]:
        if slot in skip:
            continue
        path = os.path.join(AL.KL_ASSETS, "knight_" + slot, "knight_" + slot + ".mhclo")
        if not os.path.exists(path):
            continue
        ob = OL.add_piece(rig, bm, path, slot=slot, material=clay("upper", COLORS["upper"]))
        ob["rts_look"] = "helm" if slot in ("helmet", "plume") else "any"
        out[slot] = ob
    RH.apply_plate_rules(rig, out, kind, log)
    return out


def dev_build(kind, clips=None, upper=True):
    """dev harness build: the upper armour as the knight loads it, then this kit's pieces generated directly on the
    body over it (exactly what load_lower's regen_on_body does after the MakeClothes load), skinned with their
    authored weights, chains, sockets, props, co-skinning, the knight's clips"""
    import armour_lower as AL
    import knight_anim as KA
    import rig_helpers as RH
    rig = bpy.data.objects["rts_" + kind]; bm = bpy.data.objects[kind + "_body"]
    C.pose_reset(rig)
    t0 = time.time()
    RH.ensure_helper_bones(rig, log)
    up = load_upper_mpfb(rig, bm, kind) if upper else {}
    log("upper armour: %d pieces (%.0fs)" % (len(up), time.time() - t0))
    B = AL.Body(rig, bm)
    pieces, H, D = AL.build_all_pieces(B, AL.upper_layers_live(rig) if up else None)
    bank = {n: m for n, m, k in pieces}
    extra = AL.piece_extras(B, bank, H)
    AL.ensure_helpers(rig)
    specs = {}
    for n, e in extra.items():
        if "bones" in e:
            specs.update(AL.chain_bone_specs(e["bones"], np.array(bank[n].V)))
    AL.ensure_bones(rig, specs)
    objs = {}
    body_w = None
    for n, m, k in pieces:
        if m.mixed():
            m.triangulate()
        col = COLORS.get(k, (0.6, 0.6, 0.6, 1))
        wts = m.tagw
        if not all(wts):                     # MakeHuman-interpolated pieces (mail): nearest skin vertex's weights
            if body_w is None:
                body_w = body_weights(rig, bm)
            kd = body_w[0]
            wts = [body_w[1][kd.find(v)[1]] for v in m.V]
        ob = skinned_object(rig, "%s_%s" % (kind, n), m.V, m.F, wts, clay(k, col))
        ob["rts_weights"] = "authored"
        ob["rts_part"] = n
        ob["rts_group"] = "knight_lower"
        objs[n] = ob
    sock = AL.socket_specs(rig, objs.get("belts"), extra.get("belts"))
    AL.ensure_bones(rig, sock)
    rig["rts_sockets"] = json.dumps({k: v["parent"] for k, v in sock.items()})
    props = AL.add_props(rig, which=("sword", "shield"))
    for n, o in props.items():
        o.data.materials.clear(); o.data.materials.append(clay("plate", COLORS["plate"]))
    AL.add_secondary(rig)
    AL.coskin(rig)
    bm.hide_render = True; bm.hide_viewport = True
    for o in rig.children:
        if o.type == 'MESH' and not o.get("rts_part"):
            o.hide_render = True; o.hide_viewport = True
    log("dev build: %d pieces + %d upper + props, bones %d (%.0fs)" % (len(pieces), len(up), len(rig.data.bones), time.time() - t0))
    table = KA.make_clips(rig, kind, only=clips)
    rig["rts_clips"] = json.dumps(table)
    KA.bake_secondary(rig, [c for c in table if c != "talk_emote"])
    bake_helper_keys(rig, [c for c in table if c != "talk_emote"])
    AL.post_clips(rig, kind)
    log("clips %s (%.0fs)" % (sorted(table), time.time() - t0))
    return rig, objs


# ============================================================================================ renders
VIEWS = {  # name: (camera offset from the pelvis in the character frame, target offset, lens)
    "front": ((0, -3.2, 0.2), (0, 0, 0.0), 50), "back": ((0, 3.2, 0.3), (0, 0, 0.05), 50),
    "side": ((3.2, 0, 0.2), (0, 0, 0.0), 50), "b34": ((-2.0, 2.4, 0.5), (0, 0, 0.05), 50),
    "f34": ((1.9, -2.5, 0.4), (0, 0, 0.0), 50),
    "knee": ((0.8, -1.1, -0.4), (0.1, 0, -0.45), 60), "hips": ((0.9, -1.3, 0.2), (0.0, 0, 0.05), 55),
    "shoulder_b": ((-0.9, 1.4, 0.8), (0, 0.05, 0.5), 55), "shoulder_f": ((0.9, -1.3, 0.8), (0.05, 0, 0.5), 55),
    "legs_side": ((1.6, -0.2, -0.4), (0.05, 0, -0.45), 55),
    "rknee": ((-0.55, -0.75, -0.40), (-0.13, -0.02, -0.47), 55), "rknee_side": ((-0.95, -0.05, -0.42), (-0.14, -0.02, -0.47), 55),
    "rknee_in": ((0.35, -0.75, -0.42), (-0.13, -0.02, -0.47), 55),
    "rfoot": ((-0.55, -0.55, -0.80), (-0.19, -0.07, -0.95), 55), "rfoot_side": ((-0.75, -0.05, -0.85), (-0.19, -0.07, -0.95), 55),
    "rlegs_side": ((-1.7, -0.2, -0.4), (-0.1, 0, -0.45), 50), "rhip": ((-0.8, -1.1, 0.0), (-0.1, 0, -0.1), 55),
    "belt_f": ((0.35, -1.0, 0.05), (0.0, 0, 0.0), 50), "belt_b": ((-0.3, 1.0, 0.1), (0.0, 0, 0.0), 50),
    "lsh_top": ((0.30, 0.10, 1.10), (0.17, 0.0, 0.48), 50), "lsh_front": ((0.45, -0.75, 0.62), (0.15, -0.05, 0.45), 50),
    "lsh_back": ((0.45, 0.75, 0.62), (0.15, 0.05, 0.45), 50), "lsh_side": ((0.85, -0.05, 0.55), (0.18, 0.0, 0.45), 50),
    "lshield_in": ((-0.45, -0.85, 0.35), (0.25, -0.05, 0.20), 50), "lshield_top": ((0.30, -0.25, 1.10), (0.30, -0.08, 0.20), 50),
    "lshield_out": ((1.0, -0.6, 0.35), (0.3, -0.05, 0.20), 50), "rhand": ((-0.55, -0.75, 0.05), (-0.36, -0.10, -0.10), 45),
}


def render_views(rig, kind, shots, prefix, hide=(), only=()):
    import knight_qa as QA
    for o in rig.children:
        if o.type == 'MESH' and (o.get("rts_part") in hide or (only and o.get("rts_part") not in only)):
            o.hide_render = True
    sc = C.render_setup("BLENDER_WORKBENCH", (700, 1000))
    sc.display.shading.light = 'STUDIO'; sc.display.shading.color_type = 'MATERIAL'
    sc.display.shading.show_cavity = True
    frames = {n: (s, c, f) for n, s, c, f in test_frames(rig, None, 1, True)}
    for fname, view in shots:
        if fname not in frames:
            log("no frame", fname); continue
        frames[fname][0]()
        drive_helpers(rig)
        pel = rig.matrix_world @ rig.pose.bones["pelvis"].head
        off, tgt, lens = VIEWS[view]
        C.camera(tuple(pel + Vector(off)), tuple(pel + Vector(tgt)), lens=lens)
        C.render("%s_%s_%s.png" % (prefix, fname, view))


def main(argv):
    mode = argv[0]
    kind = argv[1] if len(argv) > 1 else "male"
    opt = {a.split("=")[0]: (a.split("=", 1)[1] if "=" in a else True) for a in argv[2:]}
    clips = opt["clips"].split(",") if isinstance(opt.get("clips"), str) else None
    step = int(opt.get("step", 3))
    if mode == "dev":
        rig, objs = dev_build(kind, clips)
        tag = "_dev"
    else:
        rig = bpy.data.objects["rts_" + kind]
        tag = ""
    res = measure(rig, kind, clips, step, extreme=not opt.get("noextreme"))
    summary(res)
    path = opt.get("out") if isinstance(opt.get("out"), str) else os.path.join(C.REN, "lower_fit_%s%s.json" % (kind, tag))
    json.dump(res, open(path, "w"), indent=1, default=float)
    log("report", path)
    if opt.get("render"):
        pre = opt.get("prefix") if isinstance(opt.get("prefix"), str) else os.path.join(C.REN, "lower_fit_%s%s" % (kind, tag))
        shots = [s.split(":") for s in opt["render"].split(",")] if isinstance(opt["render"], str) else \
            [("bind", "front"), ("bind", "back"), ("bind", "side"), ("bind", "b34"), ("bind", "knee"),
             ("bind", "hips"), ("bind", "shoulder_b"), ("walk_f06", "side"), ("run_f06", "legs_side"),
             ("squat", "knee"), ("kneel", "side"), ("idle_f00", "f34")]
        render_views(rig, kind, shots, pre, hide=opt["hide"].split(",") if isinstance(opt.get("hide"), str) else (),
                     only=opt["only"].split(",") if isinstance(opt.get("only"), str) else ())
    if opt.get("save"):
        bpy.ops.wm.save_as_mainfile(filepath=opt["save"])
    bad = gates(res)
    log("GATES", "OK" if not bad else "FAIL: " + "; ".join(bad))
    if opt.get("gate") and bad:
        raise SystemExit(1)


if __name__ == "__main__":
    main(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else ["dev", "male"])
