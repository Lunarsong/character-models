"""Plate-armour HELPER JOINTS and RIGID PLATE WEIGHTING (judge items C1 / M1, iteration 2).

Linear-blend skinning bends, shears and shrinks a plate whose vertices are split between two bones (the couters were
exactly 0.5 upperarm / 0.5 lowerarm, the poleyns 0.5 thigh / 0.5 calf, the pauldron cops 0.55 clavicle / 0.45 upperarm:
they flattened and collapsed into ribbons in motion). A real plate is rigid: every plate is weighted 100 % to ONE joint,
and plates that sit over a joint ride on a HELPER joint that turns a fraction of the joint's rotation.

Helper joints (deforming leaves; per side _l / _r). ONE shared table in the armature extras `rts_helpers`
({"doc", "rules": {bone: {parent, target, weight, type: "swing_slerp"}}, "springs": {...}}), merged by every module
that adds helpers (never overwritten):
  knee_helper          parent thigh      follows calf       0.50  (poleyn cop + fan)          <- this module
  hip_helper_01        parent pelvis     follows thigh      0.33  (middle tasset lame)        <- this module
  hip_helper_02        parent pelvis     follows thigh      0.66  (lower tasset lame)         <- this module
  upperarm_helper_01..03  parent clavicle, follow upperarm 0.30 / 0.55 / 0.80 (pauldron)     <- armour_upper_rig.py
  lowerarm_helper      parent upperarm, follows lowerarm 0.50 (couter)                        <- armour_upper_rig.py
A helper has the SAME parent, head (joint position) and rest rotation as the bone it follows (canonical frame, so it is
identical on every body). Rule (engine, per frame, glTF local rotations; `rest` = the helper's rest rotation = the
target's rest rotation, both relative to the shared parent):
    d      = inverse(rest) * target.rotation            # the target joint's rotation away from its rest
    swing  = d * inverse(twist_Y(d))                   # drop the roll about the bone axis (Y): swing-twist split
    helper.rotation = rest * slerp(identity, swing, weight)
(= armour_upper_rig's 'minimal rotation of the target's rest +Y to its posed +Y, scaled by weight'). Every exported
clip has the helper tracks baked (knight_anim computes the helpers without a Blender constraint, the Damped Track ones
are sampled by the glTF exporter); engines only need the rule for their own clips / IK.

Rigid plate weighting: apply_plate_rules(rig, pieces) gives each connected component (lame, cop, rivet) of a plate piece
100 % to one joint chosen by a per-slot rule (PLATE_RULES). Small components (rivets, rosettes) inherit the joint of the
nearest large component, so nothing floats off its plate. Coordination with the armour authors: a piece whose weights
already use a *_helper_* joint, or whose object / asset carries rts_weights = "authored", is left untouched.
plate_strain() measures how rigid every component stays (Kabsch residual + smallest singular value) over poses.
"""
import math
import numpy as np

HELPERS = [  # (name stem, parent stem, source stem, weight): the helpers this module creates
    ("knee_helper", "thigh", "calf", 0.50),
    ("hip_helper_01", "pelvis", "thigh", 0.33),
    ("hip_helper_02", "pelvis", "thigh", 0.66),
]
SIDES = ("l", "r")
HELPER_LEN = 0.06


def helper_specs():
    """[{bone, parent, source, weight, swing_only}] for both sides (this module's helpers)"""
    out = []
    for stem, par, src, w in HELPERS:
        for s in SIDES:
            out.append(dict(bone="%s_%s" % (stem, s), parent=par if par == "pelvis" else "%s_%s" % (par, s),
                            source="%s_%s" % (src, s), weight=w, swing_only=True))
    return out


def read_rules(rig):
    """every helper rule on the rig, as [{bone, parent, source, weight, swing_only}] (both JSON flavours)"""
    import json
    try:
        d = json.loads(rig.get("rts_helpers", "{}") or "{}")
    except Exception:
        return []
    out = []
    for n, r in (d.get("rules") or {}).items():
        if r.get("type", "swing_slerp") in ("swing_slerp", "slerp"):
            out.append(dict(bone=n, parent=r["parent"], source=r["target"], weight=r["weight"],
                            swing_only=r.get("type", "swing_slerp") == "swing_slerp"))
    for h in d.get("helpers") or []:               # first iteration-2 flavour of this module
        if h["bone"] not in {o["bone"] for o in out}:
            out.append(h)
    return out


def helper_names():
    return [h["bone"] for h in helper_specs()]


def is_helper(name):
    return "_helper_" in name or name.startswith(tuple(h[0] + "_" for h in HELPERS))


RULE_TEXT = ("helper.rotation = rest * slerp(identity, swing_Y(inverse(rest) * source.rotation), weight); glTF local "
             "rotations; helper and source share parent, head and rest rotation; swing_Y removes the roll about the "
             "bone's +Y axis (swing-twist decomposition). Baked into every exported clip.")


def merge_json(rig, specs):
    """add `specs` to the rig's shared rts_helpers table (keeps every other module's rules and springs)"""
    import json
    try:
        d = json.loads(rig.get("rts_helpers", "{}") or "{}")
    except Exception:
        d = {}
    d.setdefault("rules", {})
    for h in d.pop("helpers", []) or []:
        d["rules"].setdefault(h["bone"], dict(parent=h["parent"], target=h["source"], weight=h["weight"],
                                              type="swing_slerp"))
    for h in specs:
        d["rules"][h["bone"]] = dict(parent=h["parent"], target=h["source"], weight=h["weight"], type="swing_slerp")
    d.setdefault("springs", {})
    d.setdefault("doc", "helper joints: local rotation = slerp(identity, swing(target), weight); swing = minimal rotation "
                 "of the target bone's rest +Y (in the helper's parent space) to its posed +Y (twist ignored). Evaluate "
                 "after the target.")
    d["rule"] = RULE_TEXT
    rig["rts_helpers"] = json.dumps(d)
    return d


# ------------------------------------------------------------------------------------------------------ Blender side
def ensure_helper_bones(rig, log=print):
    """Create / refit the helper joints on `rig` (idempotent). Heads at the source joint, rest rotation = the source's
    rest rotation (canonical when chr_lib.canonical_rest was applied). Stores rig['rts_helpers']."""
    import bpy, json
    from mathutils import Matrix
    specs = [h for h in helper_specs() if h["source"] in rig.data.bones and h["parent"] in rig.data.bones]
    for o in bpy.context.view_layer.objects:
        o.select_set(False)
    bpy.context.view_layer.objects.active = rig
    rig.select_set(True)
    mode = rig.mode
    bpy.ops.object.mode_set(mode='EDIT')
    eb = rig.data.edit_bones
    for h in specs:
        src = eb[h["source"]]
        b = eb.get(h["bone"]) or eb.new(h["bone"])
        M = src.matrix.copy()
        b.head = src.head.copy()
        b.tail = src.head + (src.tail - src.head).normalized() * HELPER_LEN
        b.matrix = Matrix.Translation(src.head) @ M.to_3x3().to_4x4()
        b.length = HELPER_LEN
        b.parent = eb[h["parent"]]
        b.use_connect = False
        b.use_deform = True
        b.inherit_scale = src.inherit_scale
    bpy.ops.object.mode_set(mode='OBJECT' if mode != 'EDIT' else 'EDIT')
    for h in specs:
        pb = rig.pose.bones[h["bone"]]
        pb.rotation_mode = 'QUATERNION'
    merge_json(rig, specs)
    log("helper joints: %d (%s)" % (len(specs), ", ".join(sorted({h['bone'][:-2] for h in specs}))))
    return [h["bone"] for h in specs]


def refit_helpers(rig, log=print, tol=0.05):
    """Every helper of the shared table must have its target's head and rest rotation (else a constraint- or rule-
    driven helper is not identity at rest and its plates jump in the bind pose). Refits the ones that are off (e.g. a
    helper created as a zero-length edit bone, whose matrix assignment silently keeps +Z). Returns the refitted names."""
    import bpy, math
    from mathutils import Matrix
    bad = []
    for h in read_rules(rig):
        B = rig.data.bones
        if h["bone"] not in B or h["source"] not in B:
            continue
        a, s = B[h["bone"]], B[h["source"]]
        R = s.matrix_local.to_3x3().inverted() @ a.matrix_local.to_3x3()
        ang = math.degrees(R.to_quaternion().angle)
        if ang > tol or (a.head_local - s.head_local).length > 1e-5:
            bad.append((h["bone"], h["source"], ang))
    if not bad:
        return []
    for o in bpy.context.view_layer.objects:
        o.select_set(False)
    bpy.context.view_layer.objects.active = rig
    rig.select_set(True)
    bpy.ops.object.mode_set(mode='EDIT')
    eb = rig.data.edit_bones
    for n, src, ang in bad:
        b, sb = eb[n], eb[src]
        L = max(b.length, 1e-3) if b.length > 1e-4 else HELPER_LEN
        b.head = sb.head.copy()
        b.tail = sb.head + (sb.tail - sb.head).normalized() * L
        b.roll = sb.roll
        b.matrix = Matrix.Translation(sb.head) @ sb.matrix.to_3x3().to_4x4()
        b.length = L
    bpy.ops.object.mode_set(mode='OBJECT')
    log("helper joints refitted to their target's rest frame: %s" % ", ".join("%s (%.1f deg)" % (n, a) for n, _, a in bad))
    return [n for n, _, _ in bad]


def swing_y(q):
    """swing part of quaternion q (mathutils) about the local +Y axis"""
    from mathutils import Quaternion
    tw = Quaternion((q.w, 0.0, q.y, 0.0))
    if tw.magnitude < 1e-9:
        return q.copy()
    tw.normalize()
    return q @ tw.inverted()


def helper_basis(src_basis_q, weight, swing_only=True):
    """local (basis) rotation of a helper from its source's local rotation"""
    from mathutils import Quaternion
    q = swing_y(src_basis_q) if swing_only else src_basis_q
    return Quaternion().slerp(q, weight)


def drive_helpers(rig):
    """Pose the helper bones of a Blender rig from their sources (for poses set by hand / QA extreme poses; the clips
    have them keyed). Sources are read from matrix_basis."""
    import bpy
    for h in read_rules(rig):
        if h["bone"] not in rig.pose.bones or h["source"] not in rig.pose.bones:
            continue
        if len(rig.pose.bones[h["bone"]].constraints):
            continue                                   # a Damped Track helper (armour_upper_rig) drives itself
        s = rig.pose.bones[h["source"]]
        q = s.matrix_basis.to_quaternion()
        pb = rig.pose.bones[h["bone"]]
        pb.rotation_mode = 'QUATERNION'
        pb.location = (0, 0, 0)
        pb.rotation_quaternion = helper_basis(q, h["weight"], h.get("swing_only", True))
    bpy.context.view_layer.update()


def helpers_in_pose(E, pose):
    """knight_anim.Engine support: armature-space matrices of the helper bones for a solved pose dict (in place)."""
    from mathutils import Matrix
    for h in getattr(E, "helpers", []):
        n, par, src = h["bone"], h["parent"], h["source"]
        if n not in E.rest or par not in pose or src not in pose:
            continue
        inh_s = pose[par] @ E.rel_rest[src]
        basis_s = inh_s.inverted() @ pose[src]
        q = helper_basis(basis_s.to_quaternion(), h["weight"], h.get("swing_only", True))
        pose[n] = pose[par] @ E.rel_rest[n] @ q.to_matrix().to_4x4()
    return pose


# ------------------------------------------------------------------------------------------------ rigid plate weights
def components(me):
    """connected components (vertex index arrays) of a Blender mesh"""
    n = len(me.vertices)
    par = np.arange(n)

    def find(i):
        r = i
        while par[r] != r:
            r = par[r]
        while par[i] != r:
            par[i], i = r, par[i]
        return r
    ev = np.empty(len(me.edges) * 2, dtype=np.int64)
    me.edges.foreach_get("vertices", ev)
    for a, b in ev.reshape(-1, 2):
        ra, rb = find(a), find(b)
        if ra != rb:
            par[ra] = rb
    roots = np.array([find(i) for i in range(n)])
    out = {}
    for i, r in enumerate(roots):
        out.setdefault(int(r), []).append(i)
    return [np.array(v) for v in sorted(out.values(), key=len, reverse=True)]


def _coords(o):
    kb = o.data.shape_keys.key_blocks["Basis"] if o.data.shape_keys else None
    co = np.empty(len(o.data.vertices) * 3)
    (kb.data if kb else o.data.vertices).foreach_get("co", co)
    co = co.reshape(-1, 3)
    M = np.array(o.matrix_world)
    return co @ M[:3, :3].T + M[:3, 3]


def uses_helpers(o):
    """True if any vertex carries weight on a helper joint (an authored .mhw lists every bone, most of them empty)"""
    hg = {g.index for g in o.vertex_groups if is_helper(g.name)}
    if not hg:
        return False
    return any(g.group in hg and g.weight > 1e-4 for v in o.data.vertices for g in v.groups)


def _side(c):
    return "l" if c[0] > 0 else "r"          # character's left = +X (Blender, faces -Y)


def _head(rig, n):
    return np.array(rig.data.bones[n].head_local)


def _seg_t(rig, a, b, p):
    """parameter of p along the segment head(a) -> head(b)"""
    A, B = _head(rig, a), _head(rig, b)
    d = B - A
    return float(np.dot(p - A, d) / max(np.dot(d, d), 1e-9))


def _stepped(order_vals, bones):
    """map components sorted by order_vals (ascending) onto the bone list (first -> bones[0], last -> bones[-1])"""
    k = len(order_vals)
    idx = np.argsort(order_vals)
    out = [None] * k
    for r, i in enumerate(idx):
        j = 0 if k == 1 else int(round(r * (len(bones) - 1) / (k - 1)))
        out[i] = bones[min(j, len(bones) - 1)]
    return out


def _per_side(rig, big, cen, fn):
    out = [None] * len(big)
    for s in SIDES:
        ids = [i for i in range(len(big)) if _side(cen[i]) == s]
        if ids:
            res = fn(s, ids)
            for i, b in zip(ids, res):
                out[i] = b
    return out


def rule_couter(rig, s, cen, ids):
    return ["lowerarm_helper_" + s] * len(ids)


def rule_vambrace(rig, s, cen, ids):
    return ["lowerarm_twist_01_" + s] * len(ids)


def rule_pauldron(rig, s, cen, ids):
    # fallback when the pauldron is not authored on helpers: cop nearest the shoulder -> armour_upper_rig's 0.30 helper,
    # the lames down the arm -> 0.55 / 0.80 helpers -> upper-arm twist (half of the arm's roll)
    t = [_seg_t(rig, "upperarm_" + s, "lowerarm_" + s, cen[i]) for i in ids]
    return _stepped(t, ["upperarm_helper_01_" + s, "upperarm_helper_02_" + s, "upperarm_helper_03_" + s,
                        "upperarm_twist_01_" + s])


def rule_poleyn(rig, s, cen, ids):
    k = _head(rig, "calf_" + s)[2]
    out = []
    for i in ids:
        dz = cen[i][2] - k
        out.append("knee_helper_" + s if abs(dz) < 0.055 else ("thigh_" + s if dz > 0 else "calf_" + s))
    return out


def rule_tasset(rig, s, cen, ids):
    return _stepped([-cen[i][2] for i in ids], ["pelvis", "hip_helper_01_" + s, "hip_helper_02_" + s])


def rule_sabaton(rig, s, cen, ids):
    ball = _head(rig, "ball_" + s)
    return ["ball_" + s if cen[i][1] < ball[1] - 0.004 else "foot_" + s for i in ids]


def rule_gorget(rig, s, cen, ids):
    return _stepped([cen[i][2] for i in ids], ["spine_05", "neck_01", "neck_02"])


def rule_cuirass(rig, s, cen, ids):
    # breast + back plate rigid on the mid chest; a split plackart / fauld (lower components) steps down the spine
    zs = [cen[i][2] for i in ids]
    out = []
    for z in zs:
        out.append("spine_04" if z > _head(rig, "spine_03")[2] + 0.05 else
                   ("spine_03" if z > _head(rig, "spine_02")[2] else "spine_02"))
    return out


# slot -> (rule, per_side: components split by the character's side first)
PLATE_RULES = {
    "couter_l": (rule_couter, True), "couter_r": (rule_couter, True),
    "vambrace_l": (rule_vambrace, True), "vambrace_r": (rule_vambrace, True),
    "pauldron_l": (rule_pauldron, True), "pauldron_r": (rule_pauldron, True),
    "poleyns": (rule_poleyn, True), "tassets": (rule_tasset, True), "sabatons": (rule_sabaton, True),
    "gorget": (rule_gorget, False), "cuirass": (rule_cuirass, False),
}
# every slot that should stay rigid per component (strain gate); cloth / mail / leather / cards are not listed
PLATE_SLOTS = ("helmet", "gorget", "gorget_top", "cuirass", "pauldron_l", "pauldron_r", "rerebrace_l", "rerebrace_r",
               "couter_l", "couter_r", "vambrace_l", "vambrace_r", "cuisses", "poleyns", "greaves", "sabatons",
               "tassets", "shield", "sword")
SMALL = 40          # components with fewer vertices follow the nearest large component


def plate_bones(rig, o, slot):
    """per-component bone list for piece `o` under PLATE_RULES[slot] (None if no rule)"""
    if slot not in PLATE_RULES:
        return None
    fn, per_side = PLATE_RULES[slot]
    comps = components(o.data)
    co = _coords(o)
    cen = [co[c].mean(0) for c in comps]
    big = [i for i, c in enumerate(comps) if len(c) >= SMALL] or list(range(len(comps)))
    if per_side:
        s_of = slot[-1] if slot[-2:] in ("_l", "_r") else None
        res = {}
        for s in SIDES:
            ids = [i for i in big if (s_of == s) or (s_of is None and _side(cen[i]) == s)]
            if ids:
                res.update(dict(zip(ids, fn(rig, s, cen, ids))))
    else:
        res = dict(zip(big, fn(rig, None, cen, big)))
    bones = [None] * len(comps)
    for i, b in res.items():
        bones[i] = b
    for i in range(len(comps)):
        if bones[i] is None:            # rivet / rosette: the nearest large component's joint
            d = [np.min(np.linalg.norm(co[comps[j]] - cen[i], axis=1)) for j in big]
            bones[i] = bones[big[int(np.argmin(d))]]
    return comps, bones


def set_rigid(o, comps, bones):
    """replace every bone group of `o` by 100 % weights per component"""
    import bpy
    rig_bones = {g.name for g in o.vertex_groups}
    for g in list(o.vertex_groups):
        if g.name.startswith("Delete") or g.name in ("body",):
            continue
        o.vertex_groups.remove(g)
    for c, b in zip(comps, bones):
        g = o.vertex_groups.get(b) or o.vertex_groups.new(name=b)
        g.add([int(i) for i in c], 1.0, 'REPLACE')
    return sorted(set(bones))


def _ensure_joints(rig, bones, log=print):
    """missing joints of a rule; the arm helpers are armour_upper_rig's (created on demand)"""
    miss = {b for b in bones if b not in rig.data.bones}
    if any(b.startswith(("upperarm_helper_", "lowerarm_helper_")) for b in miss):
        try:
            import armour_upper_rig
            armour_upper_rig.ensure_helpers(rig)
        except Exception as e:                       # the upper armour owns them; report instead of guessing
            log("arm helper joints unavailable: %s" % e)
    return {b for b in bones if b not in rig.data.bones}


def component_bones(o, comps, tol=0.999):
    """per component: the one joint that carries it (every vertex >= tol on that joint), else None (blended)"""
    names = {g.index: g.name for g in o.vertex_groups}
    skip = {i for i, n in names.items() if n.startswith("Delete") or n == "body"}
    out = []
    for c in comps:
        b0 = None
        ok = True
        for i in c:
            gs = [(names[g.group], g.weight) for g in o.data.vertices[int(i)].groups
                  if g.group not in skip and g.weight > 1e-4]
            if len(gs) != 1 or gs[0][1] < tol or (b0 is not None and gs[0][0] != b0):
                ok = False
                break
            b0 = gs[0][0]
        out.append(b0 if ok else None)
    return out


def dominant_bone(o, comp):
    names = {g.index: g.name for g in o.vertex_groups}
    acc = {}
    for i in comp:
        for g in o.data.vertices[int(i)].groups:
            n = names[g.group]
            if not n.startswith("Delete") and n != "body":
                acc[n] = acc.get(n, 0.0) + g.weight
    return max(acc, key=acc.get) if acc else None


def apply_plate_rules(rig, pieces, kind="", log=print):
    """Rigid per-component weights for every plate piece (PLATE_SLOTS). pieces: {slot: object}.
    Per connected component: a component that is already carried 100 % by one joint keeps it when the piece is
    AUTHORED (object / asset rts_weights = "authored", or weights on a *_helper_* joint: the armour author chose that
    joint); every blended component (and every component of a non-authored piece with a rule) gets its joint from the
    slot's PLATE_RULES entry, or its dominant joint when the slot has no rule. So a plate is rigid whatever its asset
    carries (an 'authored' flag on old blended weights no longer skips it: iteration-2 finding, the lower pieces were
    flagged authored with 0.5 / 0.5 thigh / calf poleyns). Returns {slot: {...} | "kept: ..."}."""
    rep = {}
    for slot, o in pieces.items():
        if slot not in PLATE_SLOTS:
            continue
        authored = o.get("rts_weights") == "authored" or uses_helpers(o)
        comps = components(o.data)
        now = component_bones(o, comps)
        if (authored or slot not in PLATE_RULES) and all(b is not None for b in now):
            rep[slot] = "kept: rigid per component (%s)" % ",".join(sorted(set(now)))
            continue
        if slot in PLATE_RULES:
            _, bones = plate_bones(rig, o, slot)
        else:
            bones = [dominant_bone(o, c) for c in comps]
        final = [(n if (authored and n is not None) else b) for n, b in zip(now, bones)]
        fixed = sum(1 for n, f in zip(now, final) if n != f)
        miss = _ensure_joints(rig, set(final), log)
        if miss:
            rep[slot] = "skipped: missing joints %s" % sorted(miss)
            continue
        used = set_rigid(o, comps, final)
        rep[slot] = {"components": len(comps), "rigidified": fixed, "bones": used,
                     "authored_kept": sum(1 for n in now if n is not None) if authored else 0}
        o["rts_weights"] = ("authored + rigidified (rig_helpers)" if authored else "rigid (rig_helpers.PLATE_RULES)")
    log("rigid plate weights: " + "; ".join("%s -> %s" % (k, v if isinstance(v, str) else "%s (%d/%d comps set)" % (
        ",".join(v["bones"]), v["rigidified"], v["components"])) for k, v in rep.items()))
    return rep


# --------------------------------------------------------------------------------------------------- strain gate
def kabsch(P0, P):
    """best rigid fit of rest points P0 onto posed points P: (max residual m, smallest singular value of the best-fit
    linear map)"""
    a = P0 - P0.mean(0)
    b = P - P.mean(0)
    H = a.T @ b
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1, 1, d])
    Rm = Vt.T @ D @ U.T
    res = float(np.linalg.norm(a @ Rm.T - b, axis=1).max())
    # affine fit in the component's own principal axes (a thin plate has no thickness axis to measure): singular
    # values of the map = stretch / squash of the component along its extent
    _, Sa, Va = np.linalg.svd(a, full_matrices=False)
    keep = Sa / math.sqrt(len(a)) > 0.004
    if keep.sum() == 0:
        return res, 1.0
    Vk = Va[keep].T
    X = a @ Vk
    B, *_ = np.linalg.lstsq(X, b, rcond=None)          # X @ B ~ b; B rows = images of the principal axes
    sv = np.linalg.svd(B, compute_uv=False)
    return res, float(sv.min())
