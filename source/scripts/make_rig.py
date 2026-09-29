"""Builds the project skeleton 'rts_human' as an MPFB custom rig (rig JSON + weights JSON) from MPFB's CC0 'default'
MakeHuman rig and weights.

Why: MPFB's 'default' rig has the only complete CC0 face rig (jaw, eyes, lids, brows, lips, cheeks, tongue) and full
fingers, but uses MakeHuman names, a hip-level root, split limb chains and 28 toe bones. 'game_engine' has engine
names but no face, twist or toes. rts_human keeps the default rig's authored weights and joint placement and reshapes
them into a UE5-mannequin-style hierarchy (root at the ground, pelvis, spine_01..05, neck_01..02, head, clavicle,
upperarm, lowerarm, hand, metacarpals + 3 finger phalanges, thigh, calf, foot, ball) with one twist leaf bone per
limb segment and a flat face rig parented to head / jaw. Because it is an MPFB custom rig (every bone end is a joint
cube, a vertex or a weighted vertex mean of the basemesh) it refits automatically to any body shape
(male / female / morphs), and MPFB clothes (mhclo) interpolate their skin weights from it.

Weights: each new bone's weight is the sum of the MakeHuman groups mapped to it, then every vertex is cut to its 4
strongest influences and renormalised (game limit), so the basemesh and every mhclo fitted to it inherit <= 4 weights.

Run with any python3 (no Blender needed):  python3 characters/scripts/make_rig.py
Writes characters/assets/rig/{rts_human.json, weights.rts_human.json} and installs both into MPFB's user data rigs/.
"""
import json, os, sys, copy, shutil

CH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MPFB = os.path.join(CH, "blender_profile/extensions/user_default/mpfb/data")
USER = os.path.join(CH, "blender_profile/extensions/.user/user_default/mpfb/data")
SRC_RIG = json.load(open(os.path.join(MPFB, "rigs/standard/rig.default.json")))
SRC_W = json.load(open(os.path.join(MPFB, "rigs/standard/weights.default.json")))["weights"]
GE_RIG = json.load(open(os.path.join(MPFB, "rigs/standard/rig.game_engine.json")))
VG = json.load(open(os.path.join(MPFB, "mesh_metadata/basemesh_vertex_groups.json")))
MAX_INFL = 4


def vg_indices(name):
    out = []
    for r in VG[name]:
        out += list(range(r[0], r[1] + 1)) if isinstance(r, list) else [r]
    return out


def verts_of(end):
    """Vertex index list that an MPFB bone-end strategy averages."""
    s = end["strategy"]
    if s == "CUBE":
        return vg_indices(end["cube_name"])
    if s == "VERTEX":
        return [end["vertex_index"]]
    if s == "MEAN":
        return list(end["vertex_indices"])
    raise ValueError(s)


def end(bone, which, rig=SRC_RIG):
    return copy.deepcopy(rig[bone][which])


def lerp_end(a, b, t):
    """A MEAN strategy that lands at a + (b - a) * t (t in quarters) and follows any body shape: MEAN of a's
    vertices repeated and b's vertices repeated, weighted so each end contributes by its share, not its vertex count."""
    va, vb = verts_of(a), verts_of(b)
    qa, qb = round((1 - t) * 4), round(t * 4)
    idx = va * (len(vb) * qa) + vb * (len(va) * qb)
    pa, pb = a["default_position"], b["default_position"]
    return {"strategy": "MEAN", "vertex_indices": idx,
            "default_position": [pa[i] + (pb[i] - pa[i]) * t for i in range(3)]}


def cube_end(name, offset=None):
    idx = vg_indices(name)
    e = {"strategy": "CUBE", "cube_name": name, "default_position": [0.0, 0.0, 0.0]}
    for b in SRC_RIG.values():   # take the default position from any bone that uses the cube
        for w in ("head", "tail"):
            if b[w].get("cube_name") == name:
                e["default_position"] = list(b[w]["default_position"])
    if offset:
        e["offset"] = list(offset)
        e["default_position"] = [e["default_position"][i] + offset[i] for i in range(3)]
    return e


BONES = {}      # name -> dict(parent, head, tail, roll, sources, coll)


def add(name, parent, head, tail, roll, sources, coll):
    BONES[name] = dict(parent=parent, head=head, tail=tail, roll=roll, sources=sources, coll=coll)


def mh(b, side):
    return b.replace(".S", side)


# ---- trunk --------------------------------------------------------------------------------------------------
add("root", "", cube_end("joint-ground"), cube_end("joint-ground", (0.0, 0.15, 0.0)), 0.0, [], "Body")
add("pelvis", "root", cube_end("joint-pelvis"), cube_end("joint-pelvis", (0.0, 0.0, 0.1)), 0.0,
    ["root", "pelvis.L", "pelvis.R"], "Body")
for new, old, par in [("spine_01", "spine05", "pelvis"), ("spine_02", "spine04", "spine_01"),
                      ("spine_03", "spine03", "spine_02"), ("spine_04", "spine02", "spine_03"),
                      ("spine_05", "spine01", "spine_04"), ("neck_01", "neck01", "spine_05")]:
    add(new, par, end(old, "head"), end(old, "tail"), SRC_RIG[old]["roll"], [old], "Body")
BONES["spine_04"]["sources"] += ["breast.L", "breast.R"]
add("neck_02", "neck_01", end("neck02", "head"), end("neck03", "tail"), 0.0, ["neck02", "neck03"], "Body")
add("head", "neck_02", end("head", "head"), end("head", "tail"), 0.0,
    ["head", "special01", "special03", "special06.L", "special06.R", "temporalis01.L", "temporalis01.R",
     "temporalis02.L", "temporalis02.R", "oculi02.L", "oculi02.R", "risorius02.L", "risorius02.R",
     "levator02.L", "levator02.R", "levator03.L", "levator03.R", "levator04.L", "levator04.R",
     "oris04.L", "oris04.R", "oris06"], "Body")

for side, s in ((".L", "_l"), (".R", "_r")):
    # ---- arm ----
    sh = end(mh("upperarm01.S", side), "head"); el = end(mh("upperarm02.S", side), "tail")
    wr = end(mh("lowerarm02.S", side), "tail")
    add("clavicle" + s, "spine_05", end(mh("clavicle.S", side), "head"), copy.deepcopy(sh),
        SRC_RIG[mh("clavicle.S", side)]["roll"], [mh("clavicle.S", side), mh("shoulder01.S", side)], "Body")
    r_ua = SRC_RIG[mh("upperarm02.S", side)]["roll"]
    add("upperarm" + s, "clavicle" + s, copy.deepcopy(sh), copy.deepcopy(el), r_ua, [mh("upperarm02.S", side)], "Body")
    add("upperarm_twist_01" + s, "upperarm" + s, copy.deepcopy(sh), lerp_end(sh, el, 0.5), r_ua,
        [mh("upperarm01.S", side)], "Twist")
    r_la = SRC_RIG[mh("lowerarm01.S", side)]["roll"]
    add("lowerarm" + s, "upperarm" + s, copy.deepcopy(el), copy.deepcopy(wr), r_la, [mh("lowerarm01.S", side)], "Body")
    add("lowerarm_twist_01" + s, "lowerarm" + s, lerp_end(el, wr, 0.5), copy.deepcopy(wr), r_la,
        [mh("lowerarm02.S", side)], "Twist")
    add("hand" + s, "lowerarm" + s, end(mh("wrist.S", side), "head"), end(mh("wrist.S", side), "tail"),
        SRC_RIG[mh("wrist.S", side)]["roll"], [mh("wrist.S", side)], "Body")
    for i, fn in enumerate(["thumb", "index", "middle", "ring", "pinky"]):
        par = "hand" + s
        if i > 0:
            mc = mh("metacarpal%d.S" % i, side)
            add(fn + "_metacarpal" + s, par, end(mc, "head"), end(mc, "tail"), SRC_RIG[mc]["roll"], [mc], "Fingers")
            par = fn + "_metacarpal" + s
        for j in (1, 2, 3):
            src = mh("finger%d-%d.S" % (i + 1, j), side)
            add("%s_%02d%s" % (fn, j, s), par, end(src, "head"), end(src, "tail"), SRC_RIG[src]["roll"], [src], "Fingers")
            par = "%s_%02d%s" % (fn, j, s)
    # ---- leg ----
    hp = end(mh("upperleg01.S", side), "head"); kn = end(mh("upperleg02.S", side), "tail")
    an = end(mh("lowerleg02.S", side), "tail")
    r_th = SRC_RIG[mh("upperleg02.S", side)]["roll"]
    add("thigh" + s, "pelvis", copy.deepcopy(hp), copy.deepcopy(kn), r_th, [mh("upperleg02.S", side)], "Body")
    add("thigh_twist_01" + s, "thigh" + s, copy.deepcopy(hp), lerp_end(hp, kn, 0.5), r_th, [mh("upperleg01.S", side)], "Twist")
    r_ca = SRC_RIG[mh("lowerleg01.S", side)]["roll"]
    add("calf" + s, "thigh" + s, copy.deepcopy(kn), copy.deepcopy(an), r_ca, [mh("lowerleg01.S", side)], "Body")
    add("calf_twist_01" + s, "calf" + s, lerp_end(kn, an, 0.5), copy.deepcopy(an), r_ca, [mh("lowerleg02.S", side)], "Twist")
    ft = mh("foot.S", side)
    add("foot" + s, "calf" + s, end(ft, "head"), end(ft, "tail"), SRC_RIG[ft]["roll"], [ft], "Body")
    toes = [k for k in SRC_RIG if k.startswith("toe") and k.endswith(side)]
    add("ball" + s, "foot" + s, end(ft, "tail"), end("ball" + s, "tail", GE_RIG), GE_RIG["ball" + s]["roll"], toes, "Body")
    # ---- face (flat: every face bone is a direct child of head or jaw) ----
    for new, old, par in [("eye", "eye.S", "head"), ("eyelid_upper", "orbicularis03.S", "head"),
                          ("eyelid_lower", "orbicularis04.S", "head"), ("brow", "oculi01.S", "head"),
                          ("cheek", "risorius03.S", "head"), ("nose", "levator06.S", "head"),
                          ("mouth_corner", "levator05.S", "head"), ("lip_upper", "oris03.S", "head"),
                          ("lip_lower", "oris07.S", "jaw")]:
        o = mh(old, side)
        add(new + s, par, end(o, "head"), end(o, "tail"), SRC_RIG[o]["roll"], [o], "Face")
    BONES["eye" + s]["sources"].append(mh("special05.S", side))

add("jaw", "head", end("jaw", "head"), end("jaw", "tail"), SRC_RIG["jaw"]["roll"], ["jaw", "special04", "oris02", "oris06.L", "oris06.R"], "Face")
add("lip_upper_c", "head", end("oris05", "head"), end("oris05", "tail"), SRC_RIG["oris05"]["roll"], ["oris05"], "Face")
add("lip_lower_c", "jaw", end("oris01", "head"), end("oris01", "tail"), SRC_RIG["oris01"]["roll"], ["oris01"], "Face")
add("tongue_01", "jaw", end("tongue01", "head"), end("tongue01", "tail"), SRC_RIG["tongue01"]["roll"], ["tongue00", "tongue01"], "Face")
for i, (o, sides) in enumerate([("tongue02", "tongue05"), ("tongue03", "tongue06"), ("tongue04", "tongue07")]):
    add("tongue_%02d" % (i + 2), "tongue_%02d" % (i + 1), end(o, "head"), end(o, "tail"), SRC_RIG[o]["roll"],
        [o, sides + ".L", sides + ".R"], "Face")

# ---- checks ----------------------------------------------------------------------------------------------------
used = [g for b in BONES.values() for g in b["sources"]]
dupe = {g for g in used if used.count(g) > 1}
assert not dupe, dupe
unmapped = [g for g in SRC_RIG if g not in used]
assert not unmapped, unmapped
missing_w = [g for g, w in SRC_W.items() if w and g not in used]
assert not missing_w, missing_w
for n, b in BONES.items():
    assert b["parent"] == "" or b["parent"] in BONES, (n, b["parent"])

# ---- rig JSON (MPFB custom rig format, version 110) ---------------------------------------------------------------
order = []
def visit(n):
    order.append(n)
    for c in [k for k, b in BONES.items() if b["parent"] == n]:
        visit(c)
visit("root")
assert len(order) == len(BONES)
rig_bones = {}
for n in order:
    b = BONES[n]
    rig_bones[n] = {"head": b["head"], "tail": b["tail"], "parent": b["parent"], "roll": b["roll"],
                    "inherit_scale": "FULL", "rigify": {}, "use_connect": False, "use_inherit_rotation": True,
                    "use_local_location": True, "use_deform": True, "collections": [b["coll"]]}
rig = {"bones": rig_bones, "version": 110, "is_subrig": False, "collections": ["Body", "Twist", "Fingers", "Face"],
       "identifying_bones": ["upperarm_twist_01_l", "eyelid_upper_l", "mouth_corner_r"],
       "name": "rts_human", "description": "RTS project game skeleton: UE5-style body + twist + face, derived from "
       "MPFB 'default' (MakeHuman, CC0)", "license": "CC0"}

# ---- weights ------------------------------------------------------------------------------------------------------
import numpy as np
BN = list(BONES)
BI = {n: i for i, n in enumerate(BN)}
NV = 19158
W = np.zeros((NV, len(BN)))
for n, b in BONES.items():
    for g in b["sources"]:
        for v, w in SRC_W.get(g, []):
            W[v, BI[n]] += w

# Joint smoothing. MakeHuman's weights switch bones over a narrow band at elbows, knees, hips, shoulders and neck,
# which folds the skin into a hard crease under linear-blend skinning. Within a radius of each joint the weights of
# that joint's bones are Laplacian-smoothed over the mesh (the joint's share of each vertex is preserved, so other
# bones are untouched), which widens the blend band. Rest positions / adjacency come from MPFB's CC0 base.obj.
OBJ = os.path.join(MPFB, "3dobjs/base.obj")
P = []; F = []
for l in open(OBJ):
    if l.startswith("v "):
        P.append([float(x) for x in l.split()[1:4]])
    elif l.startswith("f "):
        F.append([int(t.split("/")[0]) - 1 for t in l.split()[1:]])
P = np.array(P) / 10.0          # decimetres -> metres (OBJ is Y-up; only distances are used)
nbr = [set() for _ in range(NV)]
for f in F:
    for i in range(len(f)):
        a, b = f[i], f[(i + 1) % len(f)]
        nbr[a].add(b); nbr[b].add(a)
rows = np.repeat(np.arange(NV), [len(n) for n in nbr]); cols = np.array([c for n in nbr for c in n])
deg = np.array([max(1, len(n)) for n in nbr], dtype=float)


def neighbour_mean(X):
    out = np.zeros_like(X)
    np.add.at(out, rows, X[cols])
    return out / deg[:, None]


def cube(name):
    return P[vg_indices(name)].mean(0)


BODY = np.zeros(NV, bool); BODY[:13380] = True
W_MH = W.copy()                  # summed MakeHuman weights (before the deformation passes below)


def smoothstep(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3 - 2 * t)


def move_weight(src, dst, frac):
    """Move `frac` (per vertex, 0..1) of bone `src`'s weight onto bone `dst`."""
    m = W[:, BI[src]] * frac
    W[:, BI[src]] -= m
    W[:, BI[dst]] += m


# Exact L/R mirror map of the (symmetric) basemesh, and the bone permutation that swaps _l / _r.
_mkey = {tuple(np.round(P[v], 5)): v for v in range(NV)}
MIRROR = np.array([_mkey.get((round(-P[v, 0], 5) + 0.0, round(P[v, 1], 5), round(P[v, 2], 5)), -1) for v in range(NV)])
MIRROR[MIRROR < 0] = np.nonzero(MIRROR < 0)[0]
SWAP = [BI[n[:-2] + ("_r" if n.endswith("_l") else "_l")] if n[-2:] in ("_l", "_r") else BI[n] for n in BN]

# ---- deformation pass 2 (24 Sep user feedback: squat groin spike, arms-up armpit fold) --------------------------
# H1  The pelvis owns the hips. MakeHuman puts the buttocks, groin and even the perineum mid-line on spine05
#     (= spine_01, 70-84 % at the crotch), so any lumbar bend (spine_01, as in the viewer's squat) swung the whole
#     pelvis flesh around the hip-joint line while the thighs went the other way: the crotch sheared into a V.
#     UE5-style: everything below the pelvis joint is pelvis; spine_01 / spine_02 fade in over the 15 cm above it
#     (a wide band, so a lumbar bend creases the lower belly softly instead of along one edge loop).
_yp = cube("joint-pelvis")[1]
_f = smoothstep((_yp + 0.12 - P[:, 1]) / 0.15)
for b in ("spine_01", "spine_02"):
    move_weight(b, "pelvis", _f)
# (label, joint cube(s) or a centre point, radius m, iterations, bones whose weights are blended)
JOINTS = []
for s, S_ in (("_l", "l"), ("_r", "r")):
    JOINTS += [
        ("elbow" + s, ["joint-%s-elbow" % S_], 0.075, 30, ["upperarm" + s, "upperarm_twist_01" + s, "lowerarm" + s, "lowerarm_twist_01" + s]),
        ("knee" + s, ["joint-%s-knee" % S_], 0.09, 30, ["thigh" + s, "thigh_twist_01" + s, "calf" + s, "calf_twist_01" + s]),
        ("hip" + s, ["joint-%s-upper-leg" % S_], 0.10, 20, ["pelvis", "thigh" + s, "thigh_twist_01" + s, "spine_01"]),
        ("shoulder" + s, ["joint-%s-shoulder" % S_], 0.09, 15, ["clavicle" + s, "upperarm" + s, "upperarm_twist_01" + s, "spine_05"]),
        ("wrist" + s, ["joint-%s-hand" % S_], 0.035, 10, ["lowerarm" + s, "lowerarm_twist_01" + s, "hand" + s]),
        ("ankle" + s, ["joint-%s-ankle" % S_], 0.045, 10, ["calf" + s, "calf_twist_01" + s, "foot" + s]),
    ]
JOINTS += [("neck", ["joint-neck"], 0.07, 12, ["spine_05", "neck_01", "neck_02"]),
           ("head", ["joint-head"], 0.05, 8, ["neck_02", "head"])]
# A1  Armpit: MakeHuman switches from the upper arm to the rib cage within ~2 cm at the armpit, so raising the arm
#     folds the skin into a hard crease and breaks the lat / pec line. A wide blend band centred on the armpit
#     hollow (basemesh vertex 8295 and its mirror) lets the lat, the armpit and the pec edge stretch instead.
ARMPIT_V = 8295
for s, v in (("_l", ARMPIT_V), ("_r", int(MIRROR[ARMPIT_V]))):
    JOINTS.append(("armpit" + s, P[v], 0.085, 40, ["clavicle" + s, "upperarm" + s, "upperarm_twist_01" + s,
                                                  "spine_03", "spine_04", "spine_05"]))
SMOOTH_ALPHA = 0.5
for label, cubes, rad, it, bones in JOINTS:
    c = np.mean([cube(cn) for cn in cubes], 0) if isinstance(cubes, list) else np.asarray(cubes)
    d = np.linalg.norm(P - c, axis=1)
    m = np.clip(1.0 - d / rad, 0, 1); m = m * m * (3 - 2 * m); m[~BODY] = 0.0
    idx = [BI[b] for b in bones]
    share = W[:, idx].sum(1)
    for _ in range(it):
        Wj = W[:, idx]
        Wj = Wj + (SMOOTH_ALPHA * m)[:, None] * (neighbour_mean(Wj) - Wj)
        tot = Wj.sum(1)
        ok = tot > 1e-9
        Wj[ok] *= (share[ok] / tot[ok])[:, None]
        W[:, idx] = Wj

# Crotch mid-line. Two fixes were made in parallel for the same user item (24 Sep, squat groin V spike): 'b2' (H2
# below: thigh share fades in with the geodesic distance from the seam) and 'live' (iteration 1: position-based rewrite,
# each side's leg share only on its own side, fading in over ~6 cm, then smoothed). The iteration-2 reconcile rebuilt
# both bodies with each (same H1 / A1 / S / T passes, same corrective solve) and measured 25 stress poses
# (renders/reconcile/groin_variants_*.png, STATUS.md '## Iteration 2: base reconcile'): 'live' removes the flipped
# faces and most self-intersections of the male squat groin (flips 27 -> 15, isect 62 -> 53 over all poses; squat
# 2 / 8 -> 0 / 2) and lowers the female groin / hip crease sums (598 -> 550, 9461 -> 9035), with a rounder perineum in
# the close-ups, so it is the default. RTS_GROIN=b2 / both rebuilds the alternatives for comparison.
GROIN = os.environ.get("RTS_GROIN", "live")
if GROIN in ("b2", "both"):
    # H2  Crotch mid-line. The perineum / pubis / gluteal-cleft mid-line switched from ~80 % left thigh to ~80 % right
    #     thigh across 3 cm, so spreading the thighs (squat, lunge, sitting) tore the mid-line into a spike. The thigh
    #     share now fades in with the geodesic distance from the crotch mid-line (20 % on the seam, full at 10 cm);
    #     what it gives up goes to the pelvis, which is where that skin is anchored (ischium / pubis).
    import heapq
    EL = {}
    for a in range(NV):
        for b in nbr[a]:
            EL[(a, b)] = float(np.linalg.norm(P[a] - P[b]))


    def geodesic(seeds, maxd):
        d = np.full(NV, np.inf)
        h = [(0.0, int(v)) for v in seeds]
        for _, v in h:
            d[v] = 0.0
        heapq.heapify(h)
        while h:
            dv, v = heapq.heappop(h)
            if dv > d[v] or dv > maxd:
                continue
            for u in nbr[v]:
                if not BODY[u]:
                    continue
                nd = dv + EL[(v, u)]
                if nd < d[u]:
                    d[u] = nd
                    heapq.heappush(h, (nd, u))
        return d


    _hip_y = cube("joint-l-upper-leg")[1]
    SEAM = np.nonzero(BODY & (np.abs(P[:, 0]) < 1e-4) & (P[:, 1] < _hip_y - 0.01) & (P[:, 1] > _hip_y - 0.25))[0]
    _g = 0.2 + 0.8 * smoothstep(geodesic(SEAM, 0.2) / 0.10)
    for s in ("_l", "_r"):
        for b in ("thigh" + s, "thigh_twist_01" + s):
            move_weight(b, "pelvis", 1.0 - _g)

if GROIN in ("live", "both"):
    # Groin rewrite. MakeHuman gives the crotch midline 10-40 % thigh weight (asymmetric, one side or both), so in a
    # squat both thighs drag the midline forward and down into a sharp V. Here the crotch is re-weighted by position:
    # the leg share of each vertex (all thigh + thigh-twist weight) goes only to its own side, fading from 0 at the
    # midline to the original amount ~6 cm out; the rest goes to the pelvis/lower-spine bones it already had (or pelvis).
    # A falloff mask blends the rewrite into the untouched weights, then the region is Laplacian-smoothed.
    hipL, hipR = cube("joint-l-upper-leg"), cube("joint-r-upper-leg")
    hipc = (hipL + hipR) / 2                                  # base.obj axes: x left(+)/right(-), y up, z forward
    leg = {"_l": [BI["thigh_l"], BI["thigh_twist_01_l"]], "_r": [BI["thigh_r"], BI["thigh_twist_01_r"]]}
    legall = leg["_l"] + leg["_r"]
    trunk = [BI["pelvis"], BI["spine_01"]]
    rx = np.abs(P[:, 0] - hipc[0]); ry = P[:, 1] - hipc[1]; rz = P[:, 2] - hipc[2]
    box = (np.clip(1 - rx / 0.085, 0, 1) * np.clip(1 - np.maximum(0, -ry - 0.13) / 0.05, 0, 1)
           * np.clip(1 - np.maximum(0, ry + 0.01) / 0.05, 0, 1) * np.clip(1 - np.maximum(0, np.abs(rz) - 0.09) / 0.04, 0, 1))
    box = box * box * (3 - 2 * box); box[~BODY] = 0.0
    side_t = np.clip((rx - 0.008) / 0.06, 0, 1); side_t = side_t * side_t * (3 - 2 * side_t)
    Wn = W.copy()
    for i in np.nonzero(box > 0)[0]:
        L = W[i, legall].sum()
        own = leg["_l"] if P[i, 0] > hipc[0] else leg["_r"]
        ownw = W[i, own]; ratio = ownw / ownw.sum() if ownw.sum() > 1e-6 else np.array([0.5, 0.5])
        newL = L * side_t[i]
        Wn[i, legall] = 0.0
        Wn[i, own] = ratio * newL
        freed = L - newL
        tw = W[i, trunk]
        Wn[i, trunk] += (tw / tw.sum()) * freed if tw.sum() > 1e-6 else np.array([freed, 0.0])
    W = W * (1 - box)[:, None] + Wn * box[:, None]
    gi_idx = legall + trunk
    share = W[:, gi_idx].sum(1)
    for _ in range(20):
        Wj = W[:, gi_idx]
        Wj = Wj + (0.5 * box)[:, None] * (neighbour_mean(Wj) - Wj)
        tot = Wj.sum(1); ok = tot > 1e-9
        Wj[ok] *= (share[ok] / tot[ok])[:, None]
        W[:, gi_idx] = Wj

# S   Exact left / right symmetry for every body vertex that has no face-bone weight (the face rig is left alone).
FACE_B = [BI[n] for n, b in BONES.items() if b["coll"] == "Face"]
_sym = BODY & (W[:, FACE_B].sum(1) < 1e-9) & (W[MIRROR][:, FACE_B].sum(1) < 1e-9)
W[_sym] = 0.5 * (W[_sym] + W[MIRROR[_sym]][:, SWAP])

# T   The skin-tight helper (helper-tights, what MPFB fits tight clothes / armour padding to) follows the body where
#     the passes above changed it, so fitted clothes keep deforming like the skin under them.
_chg = np.abs(W - W_MH).sum(1); _chg[~BODY] = 0.0
_z = np.clip(_chg / 0.05, 0, 1)[:, None]
for _ in range(4):
    _z = np.maximum(_z, neighbour_mean(_z))
_z = _z[:, 0]; _z[~BODY] = 0.0
TIGHTS = np.array(vg_indices("helper-tights"))
_bi = np.nonzero(BODY)[0]
for ch in np.array_split(TIGHTS, 16):
    dd = np.linalg.norm(P[ch][:, None, :] - P[_bi][None, :, :], axis=2)
    k = _bi[np.argmin(dd, axis=1)]
    a = _z[k][:, None]
    W[ch] = (1 - a) * W[ch] + a * W[k]

# Cut to MAX_INFL influences. Where the cut drops a noticeable share, the pruned field is smoothed a few times with
# each vertex restricted to its own kept bones, so neighbouring vertices do not jump against each other.
raw = W.copy()
keep = np.zeros_like(W, dtype=bool)
top = np.argsort(-W, axis=1)[:, :MAX_INFL]
np.put_along_axis(keep, top, True, axis=1)
_right = _sym & (P[:, 0] < -1e-6)                   # right side mirrors the left side's choice (ties stay symmetric)
keep[_right] = keep[MIRROR[_right]][:, SWAP]
_mid = np.nonzero(_sym & (MIRROR == np.arange(NV)))[0]    # mid-line: keep an _l/_r pair together or drop both
keep[_mid] &= keep[_mid][:, SWAP]
keep &= W > 1e-6
tot = W.sum(1)
drop = np.where(tot > 0, (W * ~keep).sum(1) / np.maximum(tot, 1e-12), 0.0)
W4 = np.where(keep, W, 0.0)
W4 /= np.maximum(W4.sum(1, keepdims=True), 1e-12)
region = drop > 0.02
for _ in range(2):                          # grow the region by two rings
    region = region | (neighbour_mean(region[:, None].astype(float))[:, 0] > 0)
for _ in range(6):
    nm = neighbour_mean(W4)
    cand = np.where(keep, 0.5 * W4 + 0.5 * nm, 0.0)
    cand /= np.maximum(cand.sum(1, keepdims=True), 1e-12)
    W4[region] = cand[region]
dropped = sorted(drop[drop > 0].tolist())
out = {n: [] for n in BONES}
for v in range(NV):
    for j in np.nonzero(W4[v] > 1e-4)[0]:
        out[BN[j]].append([v, round(float(W4[v, j]), 6)])
weights = {"name": "rts_human weights", "description": "Summed MakeHuman default weights, <=4 influences, normalised",
           "license": "CC0", "copyright": "derived from MakeHuman default weights (CC0)", "version": 110,
           "weights": {n: sorted(l) for n, l in out.items() if l}}

dst = os.path.join(CH, "assets/rig"); os.makedirs(dst, exist_ok=True)
json.dump(rig, open(os.path.join(dst, "rts_human.json"), "w"), indent=1, sort_keys=True)
json.dump(weights, open(os.path.join(dst, "weights.rts_human.json"), "w"), indent=0, sort_keys=True)
os.makedirs(os.path.join(USER, "rigs"), exist_ok=True)
for f in ("rts_human.json", "weights.rts_human.json"):
    shutil.copy(os.path.join(dst, f), os.path.join(USER, "rigs", f))
dropped.sort()
print("RIG rts_human bones", len(BONES), "weighted", len(weights["weights"]), "vertices", int((W4.sum(1) > 0).sum()),
      "verts>4infl", len(dropped), "dropped share max %.3f p95 %.3f" % (dropped[-1] if dropped else 0,
      dropped[int(len(dropped) * 0.95)] if dropped else 0))
print("RIG order", " ".join(order))
