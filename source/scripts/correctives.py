"""Pose-space corrective shape keys for the rts_human body (run inside Blender).

Linear-blend skinning (what every engine does with <= 4 weights) cannot keep volume and a smooth surface where a
joint swings far (arm raised 130+ deg, hip flexed 100+ deg): the blended matrices shrink the skin towards the joint and
fold it where the blend band is narrow. The fix used in AAA rigs is a corrective blendshape per extreme pose, faded in
by the joint angle. Here the corrective shapes are generated from the rig, not sculpted:

1. key pose  : one bone swung to a target direction (both sides at once), everything else at rest (GROUPS below);
2. target    : the linear-blend shape of that pose, with
               - a virtual shoulder girdle: the skin weighted to the clavicle is shaped as if the clavicle had elevated /
                 protracted with the arm (scapulo-humeral rhythm), so the acromion and trapezius make room for the
                 humerus instead of folding, without moving the bone (engines need not animate the clavicle);
               - spherical-blend skinning (Kavan & Zara 2005: quaternion-blended rotation about the centre of rotation
                 the vertex's bones agree on) on the compression side of the bend, which keeps volume instead of
                 collapsing (the stretch side stays linear, like stretching skin; the hip uses it everywhere);
               - a delta-mush relaxation inside the joint region (rest-pose detail kept, folds and spikes smoothed);
3. solve     : per vertex, least squares over all key poses of a group, so that with the drivers' own weights
               LBS(rest + sum_k w_k delta_k) reproduces every key's target (pose-space deformation with cone kernels;
               in-between keys and constraint-only keys keep partial weights honest); the same solve gives the
               morph-normal deltas whose *skinned* normal matches the target surface (engines skin normals, they do
               not recompute them); the result is confined to the joint region and split left / right with a
               partition of unity across the mid-line;
4. driver    : a cone reader (chr_lib.corrective_weight) on the bone's direction in its parent's frame.

The spec (list of {name, bone, ref, axis, target, angle_from, angle_to}) is stored on the body mesh as the custom
property 'rts_correctives' and exported to the glTF mesh extras; the viewer and engines evaluate it. Normal deltas are
written into the GLB after export (export_corrective_normals -> glb_patch.py).

CLI (re-generate the correctives of an existing export .blend and re-export its GLB, ~20 s):
  Blender -b out/base_male_export.blend --python-exit-code 1 -P scripts/correctives.py -- male
"""
import bpy, json, math, os, sys
import numpy as np
from mathutils import Matrix, Vector

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import deform_metrics as DM

# ---------------------------------------------------------------------------------------------------- skinning
def quat_of(M):
    return np.array(Matrix(M[:3, :3].tolist()).to_quaternion())     # w, x, y, z


def qmat(q):
    w, x, y, z = q.T
    R = np.empty((len(q), 3, 3))
    R[:, 0, 0] = 1 - 2 * (y * y + z * z); R[:, 0, 1] = 2 * (x * y - z * w); R[:, 0, 2] = 2 * (x * z + y * w)
    R[:, 1, 0] = 2 * (x * y + z * w); R[:, 1, 1] = 1 - 2 * (x * x + z * z); R[:, 1, 2] = 2 * (y * z - x * w)
    R[:, 2, 0] = 2 * (x * z - y * w); R[:, 2, 1] = 2 * (y * z + x * w); R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def lbs(W, M, X):
    """Linear-blend skinning of rest positions X (n,3) with weights W (n,B) and skin matrices M (B,4,4)."""
    A = np.einsum("vb,bij->vij", W, M[:, :3, :3])
    t = np.einsum("vb,bi->vi", W, M[:, :3, 3])
    return np.einsum("vij,vj->vi", A, X) + t, A, t


def sbs(W, M, X, idx, heads, lam=1e-3):
    """Spherical-blend skinning (Kavan & Zara 2005) for vertices idx (others: LBS): each vertex rotates by the
    quaternion blend of its bones about the centre of rotation c that its bones agree on best in this pose,
    min_c sum_{j<k} w_j w_k |M_j c - M_k c|^2 (the shared joint for two neighbouring bones), so a joint neither
    collapses (LBS) nor needs a precomputed centre. heads: (B,3) rest bone heads (regulariser anchor)."""
    Y, _, _ = lbs(W, M, X)
    Q = np.array([quat_of(m) for m in M])
    w = W[idx]
    top = np.argsort(-w, axis=1)[:, :4]
    tw = np.take_along_axis(w, top, axis=1)
    R, t = M[:, :3, :3], M[:, :3, 3]
    A = np.zeros((len(idx), 3, 3)) + lam * np.eye(3)
    c0 = np.einsum("vk,vki->vi", tw, heads[top]) / np.maximum(tw.sum(1, keepdims=True), 1e-12)
    b = lam * c0
    for x in range(4):
        for y in range(x + 1, 4):
            j, k = top[:, x], top[:, y]
            ww = (tw[:, x] * tw[:, y])[:, None, None]
            D = R[j] - R[k]; e = t[j] - t[k]
            A += ww * np.einsum("vji,vjk->vik", D, D)
            b -= ww[:, :, 0] * np.einsum("vji,vj->vi", D, e)
    c = np.linalg.solve(A, b[:, :, None])[:, :, 0]
    ref = Q[top[:, 0]]
    qs = Q[top]                                            # (v,4,4)
    sgn = np.sign(np.einsum("vki,vi->vk", qs, ref)); sgn[sgn == 0] = 1
    q = np.einsum("vk,vk,vki->vi", tw, sgn, qs)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    Rb = qmat(q)
    Lc = np.einsum("vk,vkij,vj->vi", tw, R[top], c) + np.einsum("vk,vki->vi", tw, t[top])
    Y[idx] = np.einsum("vij,vj->vi", Rb, X[idx] - c) + Lc
    return Y


def delta_mush(topo, X, rest, mask, iters=20, alpha=0.5):
    """Delta mush on the posed positions X inside `mask` (0..1 per vertex): smooth both the rest and the posed
    surface, store the rest detail in the smoothed rest surface's local frames and re-apply it on the smoothed
    posed surface. Removes folds / spikes, keeps the rest-pose surface detail."""
    E = topo.E

    def smooth(Y):
        Y = Y.copy()
        for _ in range(iters):
            s = np.zeros_like(Y); np.add.at(s, E[:, 0], Y[E[:, 1]]); np.add.at(s, E[:, 1], Y[E[:, 0]])
            Y += (alpha * mask)[:, None] * (s / topo.deg[:, None] - Y)
        return Y

    def frames(Y):
        n = np.zeros_like(Y)
        tn = np.cross(Y[topo.T[:, 1]] - Y[topo.T[:, 0]], Y[topo.T[:, 2]] - Y[topo.T[:, 0]])
        for k in range(3):
            np.add.at(n, topo.T[:, k], tn)
        n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
        e = Y[topo.nb1] - Y
        tx = e - (e * n).sum(1, keepdims=True) * n
        tx /= np.maximum(np.linalg.norm(tx, axis=1, keepdims=True), 1e-12)
        ty = np.cross(n, tx)
        return np.stack([tx, ty, n], axis=2)            # columns = local frame
    if not hasattr(topo, "nb1"):
        nb1 = np.zeros(topo.n, dtype=np.int64)
        nb1[topo.E[:, 0]] = topo.E[:, 1]; nb1[topo.E[:, 1]] = topo.E[:, 0]
        topo.nb1 = nb1
    Rs = smooth(rest); Ps = smooth(X)
    Fr, Fp = frames(Rs), frames(Ps)
    d_local = np.einsum("vji,vj->vi", Fr, rest - Rs)
    out = Ps + np.einsum("vij,vj->vi", Fp, d_local)
    m = mask > 1e-6
    Y = X.copy(); Y[m] = out[m]
    return Y


def vertex_normals(topo, X):
    """Area-weighted vertex normals of positions X on the topology's triangles."""
    tn = np.cross(X[topo.T[:, 1]] - X[topo.T[:, 0]], X[topo.T[:, 2]] - X[topo.T[:, 0]])
    n = np.zeros_like(X)
    for k in range(3):
        np.add.at(n, topo.T[:, k], tn)
    return n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)


def body_weights(body, rig):
    W, names = DM.weight_matrix(body, rig)
    W /= np.maximum(W.sum(1, keepdims=True), 1e-12)
    return W, names


def rest_co(body):
    kb = body.data.shape_keys.key_blocks["Basis"] if body.data.shape_keys else None
    n = len(body.data.vertices)
    a = np.empty(n * 3)
    (kb.data if kb else body.data.vertices).foreach_get("co", a)
    return a.reshape(-1, 3)


def set_normal_delta(body, name, dn):
    """Keep a corrective's morph-normal deltas on the mesh (point attribute 'nrm_<name>', not exported by glTF) until
    export_corrective_normals() writes them into the GLB."""
    me = body.data
    an = "nrm_" + name
    if an in me.attributes:
        me.attributes.remove(me.attributes[an])
    a = me.attributes.new(an, 'FLOAT_VECTOR', 'POINT')
    a.data.foreach_set("vector", np.asarray(dn, dtype=np.float64).ravel())


def export_corrective_normals(glb_path, body):
    """Replace the exporter's morph normals of every 'cor_*' target in the GLB with the solved ones (glb_patch.py)."""
    import glb_patch
    from mathutils.kdtree import KDTree
    me = body.data
    spec = json.loads(me["rts_correctives"]) if isinstance(me["rts_correctives"], str) else me["rts_correctives"].to_dict()
    js, bin_ = glb_patch.read_glb(glb_path)
    mesh = [m for m in js["meshes"] if m["name"] == body.data.name or m["name"] == body.name][0]
    prim = mesh["primitives"][0]
    P = glb_patch.accessor_array(js, bin_, prim["attributes"]["POSITION"]).astype(np.float64)
    rest = rest_co(body)
    kd = KDTree(len(rest))
    for i, c in enumerate(rest):
        kd.insert((c[0], c[2], -c[1]), i)                  # Blender (x, y, z) -> glTF (x, z, -y)
    kd.balance()
    vmap = np.array([kd.find(tuple(p))[1] for p in P])
    dist = np.array([kd.find(tuple(p))[2] for p in P])
    assert dist.max() < 1e-4, dist.max()
    deltas = {}
    for c in spec["correctives"]:
        a = me.attributes["nrm_" + c["name"]]
        d = np.empty(len(rest) * 3); a.data.foreach_get("vector", d); d = d.reshape(-1, 3)[vmap]
        deltas[c["name"]] = np.stack([d[:, 0], d[:, 2], -d[:, 1]], axis=1)
    glb_patch.set_morph_normals(glb_path, mesh["name"], deltas)
    return len(deltas)


def set_key(body, name, delta):
    me = body.data
    if not me.shape_keys:
        body.shape_key_add(name="Basis", from_mix=False)
    kb = me.shape_keys.key_blocks
    k = kb.get(name) or body.shape_key_add(name=name, from_mix=False)
    base = np.empty(len(me.vertices) * 3); kb["Basis"].data.foreach_get("co", base)
    k.data.foreach_set("co", (base + delta.ravel()))
    return k


# ---------------------------------------------------------------------------------------------------- builder
def _ss(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3 - 2 * t)


# Key poses, left side (right = mirrored), in groups that share a joint region. dir: the bone's head->tail direction
# in armature space (Blender: +X = character's left, -Y = forward, +Z = up) with its parent at rest. a: driver cone
# (angle_from, angle_to) in degrees. region: (bone whose head is the centre, centre offset (x mirrored), full radius,
# falloff radius), metres for the 1.85 m male (scaled by stature). Target shape at the key pose:
#   girdle [(armature axis, deg), ...]: the skin weighted to the clavicle is shaped as if the clavicle had rotated
#          too (the scapulo-humeral rhythm: a raised arm lifts the acromion / trapezius, a forward reach protracts
#          it), without moving the bone, so engines need not animate it;
#   shape  False: an in-between pose that only adds a least-squares constraint (no corrective of its own)
#   sbs    how much of the spherical-blend (volume keeping) shape to take (sbs_side: only on the compression side
#          of the bend, the stretch side stays linear like stretching skin); mush: delta-mush iterations (crotch_mush:
#          extra relaxation around the perineum per body, where the pelvis-owned mid-line meets the spreading thighs;
#          it helps the male bulge, the female measured best without it).
ELEV, PROT = (0, -1, 0), (0, 0, -1)       # left clavicle: elevation / protraction axes (armature space)
SH_REG = ("upperarm", (-0.03, 0.0, -0.05), 0.13, 0.22)
HIP_REG = ("thigh", (-0.03, 0.0, -0.04), 0.14, 0.23)
GROUPS = [
    dict(group="shoulder", bone="upperarm", ref="clavicle", region=SH_REG, mush=12, sbs=1.0, sbs_side=True, keys=[
        dict(name="arm_side", dir=(1.0, 0.0, 0.0), a=(60, 5), girdle=[(ELEV, 8)]),
        dict(name="arm_up", dir=(0.22, 0.05, 1.0), a=(70, 5), girdle=[(ELEV, 30)]),
        dict(name="arm_up_180", dir=(0.03, 0.03, 1.0), a=(12, 2), girdle=[(ELEV, 45)]),
        dict(name="arm_up_mid", dir=(0.7, 0.02, 0.7), a=(35, 3), girdle=[(ELEV, 18)]),
        dict(name="arm_forward", dir=(0.12, -1.0, 0.05), a=(70, 5), girdle=[(PROT, 12)]),
        dict(name="arm_forward_mid", dir=(0.52, -0.66, -0.47), a=(30, 3), girdle=[(PROT, 6)]),
        dict(name="arm_forward_up", dir=(0.12, -0.7, 0.7), a=(35, 3), girdle=[(ELEV, 15), (PROT, 10)]),
        dict(name="arm_across", dir=(-0.5, -0.85, 0.12), a=(40, 5), girdle=[(PROT, 15)]),
    ]),
    dict(group="hip", bone="thigh", ref="pelvis", region=HIP_REG, mush=15, crotch_mush={"male": 10}, sbs=1.0, sbs_side=False, keys=[
        dict(name="thigh_forward", dir=(0.08, -1.0, 0.0), a=(80, 5)),
        dict(name="thigh_forward_mid", dir=(0.1, -0.7, -0.7), a=(30, 3)),
        dict(name="thigh_forward_65", dir=(0.1, -0.9, -0.42), shape=False),
        dict(name="thigh_squat", dir=(0.35, -0.88, 0.3), a=(40, 5)),
        dict(name="thigh_kick", dir=(0.1, -0.72, 0.68), a=(35, 5)),
        dict(name="thigh_out", dir=(0.87, 0.0, -0.5), a=(50, 5)),
        dict(name="thigh_back", dir=(0.05, 0.55, -0.83), a=(25, 3)),
    ]),
]


def key_dirs(rig, G, K):
    """Armature-space key direction of the group's bone, per side ('_l', '_r'; the right side is mirrored)."""
    out = {}
    for s, sg in (("_l", 1), ("_r", -1)):
        d = np.array([K["dir"][0] * sg, K["dir"][1], K["dir"][2]], dtype=float)
        out[s] = d / np.linalg.norm(d)
    return out


def build_correctives(rig, body, kind, groups=None, lam=1e-3, log=print):
    """Generate the corrective shape keys 'cor_<name>_l/_r' on `body` and store the driver spec on body.data.
    Per group the correctives are solved jointly (per vertex, least squares) so that with the drivers' own weights
    LBS(rest + sum_k w_k delta_k) reproduces the target shape at every key pose of the group."""
    import chr_lib as CL
    groups = groups or GROUPS
    if kind not in CL.PRESETS:
        import cust_lib                        # registers the proportion variants (male_stocky, ...) in PRESETS
    # region sizes scale with the stature; proportion variants use their own stature and their base kind's tuning
    S = CL.PRESETS[kind]["stature"] / CL.PRESETS["male"]["stature"]
    base_kind = CL.PRESETS[kind].get("tex_kind", kind)
    me = body.data
    if me.shape_keys:
        for k in [k for k in me.shape_keys.key_blocks if k.name.startswith("cor_")]:
            body.shape_key_remove(k)
    if "rts_correctives" in me:
        del me["rts_correctives"]
    for a in [a.name for a in me.attributes if a.name.startswith("nrm_cor_")]:
        me.attributes.remove(me.attributes[a])
    CL.pose_reset(rig)
    topo = DM.Topo(body)
    W, names = body_weights(body, rig)
    rest = rest_co(body)
    n = len(rest)
    heads = np.array([b.head_local[:] for b in rig.data.bones])
    bidx = {b.name: i for i, b in enumerate(rig.data.bones)}
    x = rest[:, 0]
    side_w = {"_l": _ss((x + 0.02 * S) / (0.04 * S)), "_r": 1.0 - _ss((x + 0.02 * S) / (0.04 * S))}
    spec = []
    for G in groups:
        bone, ref = G["bone"], G["ref"]
        rb, roff, r0, r1 = G["region"]
        region = np.zeros(n)
        for s, sg in (("_l", 1), ("_r", -1)):
            c = np.array(rig.data.bones[rb + s].head_local) + np.array([roff[0] * sg, roff[1], roff[2]]) * S
            region = np.maximum(region, _ss((r1 * S - np.linalg.norm(rest - c, axis=1)) / ((r1 - r0) * S)) * side_w[s])
        idx = np.nonzero(region > 1e-6)[0]
        # driver spec (left and right) for every key of the group
        entries = {}
        KD = {K["name"]: key_dirs(rig, G, K) for K in G["keys"]}
        for K in G["keys"]:
            if not K.get("shape", True):
                continue
            for s, sg in (("_l", 1), ("_r", -1)):
                refb = ref + s if (ref + s) in rig.data.bones else ref
                dd = KD[K["name"]][s]
                tgt = np.array(rig.data.bones[refb].matrix_local.to_3x3()).T @ dd
                entries[K["name"] + s] = {"name": "cor_%s%s" % (K["name"], s), "bone": bone + s, "ref": refb,
                                          "axis": [0.0, 1.0, 0.0],
                                          "target": [round(float(v), 5) for v in tgt / np.linalg.norm(tgt)],
                                          "angle_from": float(K["a"][0]), "angle_to": float(K["a"][1])}
        # no corrective may be on in the bind pose (armour and clothes are fitted to the bind mesh): where the rest
        # direction is inside a cone (arm_side: the A-pose arm is ~50 deg below the horizontal key, inside its 60 deg
        # cone, which left cor_arm_side at 0.11-0.13 at rest), the cone starts 1 deg past the rest direction instead
        CL.pose_reset(rig)
        for e in entries.values():
            pb, pr = rig.pose.bones[e["bone"]], rig.pose.bones[e["ref"]]
            d = pr.matrix.to_3x3().inverted() @ (pb.matrix.to_3x3() @ Vector(e["axis"]))
            a_rest = math.degrees(d.angle(Vector(e["target"]), 0.0))
            if a_rest < e["angle_from"]:
                af = round(max(e["angle_to"] + 1.0, a_rest - 1.0), 2)
                log("corrective %s: rest direction %.1f deg inside its cone, angle_from %.1f -> %.1f" % (
                    e["name"], a_rest, e["angle_from"], af))
                e["angle_from"] = af
        kn = [K["name"] for K in G["keys"] if K.get("shape", True)]     # keys with their own corrective
        nk = len(kn)
        Mn = np.zeros((len(idx), 3 * nk, 3 * nk)); rhs = np.zeros((len(idx), 3 * nk)); rhs_n = np.zeros((len(idx), 3 * nk))
        n0 = vertex_normals(topo, rest)
        wtab = []
        for K in G["keys"]:
            CL.pose_reset(rig)
            for s in ("_l", "_r"):
                CL.aim(rig, bone + s, tuple(KD[K["name"]][s]))
            bpy.context.view_layer.update()
            M = DM.skin_matrices(rig)
            Mt = M.copy()                          # transforms the target is shaped with (virtual girdle motion)
            if K.get("girdle"):
                gb = "clavicle"
                saved = {p.name: p.matrix_basis.copy() for p in rig.pose.bones}
                for s, sg in (("_l", 1), ("_r", -1)):
                    for gax, gdeg in K["girdle"]:        # mirror: axis (-x, y, z), angle negated
                        CL.rot(rig, gb + s, (gax[0] * sg, gax[1], gax[2]), gdeg * sg)
                Mg = DM.skin_matrices(rig)
                for s in ("_l", "_r"):
                    Mt[bidx[gb + s]] = Mg[bidx[gb + s]]
                for p in rig.pose.bones:
                    p.matrix_basis = saved[p.name]
                bpy.context.view_layer.update()
            Y_lbs, A, t = lbs(W, M, rest)
            Y = lbs(W, Mt, rest)[0]
            sbs_amt = K.get("sbs", G.get("sbs", 0.0))
            if sbs_amt > 0:
                alpha = np.zeros(n)
                for s, sg in (("_l", 1), ("_r", -1)):
                    jc = np.array(rig.data.bones[bone + s].head_local)
                    d0 = np.array(rig.data.bones[bone + s].tail_local) - jc; d0 /= np.linalg.norm(d0)
                    d1 = KD[K["name"]][s]
                    u = d1 - d0; u /= max(np.linalg.norm(u), 1e-9)
                    o = rest - jc
                    cosang = (o @ u) / np.maximum(np.linalg.norm(o, axis=1), 1e-9)
                    side_mode = K.get("sbs_side", G.get("sbs_side", True))
                    mix = (_ss(0.5 + cosang / 1.2) if side_mode is True else
                           _ss(0.5 - cosang / 1.2) if side_mode == "stretch" else 1.0)
                    alpha = np.maximum(alpha, sbs_amt * mix * ((x * sg) > -0.03 * S))
                Y = Y + (alpha * (region > 0))[:, None] * (sbs(W, Mt, rest, idx, heads) - Y)
            mush = K.get("mush", G.get("mush", 0))
            if mush:
                Y = delta_mush(topo, Y, rest, region, iters=mush)
            cmi = G.get("crotch_mush", {}).get(base_kind, 0)
            if cmi:                                    # extra relaxation of the perineum / inner-thigh junction
                cz = 0.5 * (np.array(rig.data.bones["thigh_l"].head_local) + np.array(rig.data.bones["thigh_r"].head_local))
                cz[2] -= 0.09 * S
                cm = _ss((0.12 * S - np.linalg.norm(rest - cz, axis=1)) / (0.06 * S)) * (region > 0)
                Y = delta_mush(topo, Y, rest, cm, iters=cmi)
            r = (region[:, None] * (Y - Y_lbs))[idx]
            # normals: engines skin the morphed rest normal with the same blended matrix, so the rest-space normal
            # that lands on the target surface's normal is A^-1 N_target (normalised)
            Nt = vertex_normals(topo, Y_lbs + region[:, None] * (Y - Y_lbs))
            nr = np.linalg.solve(A, Nt[:, :, None])[:, :, 0]
            nr /= np.maximum(np.linalg.norm(nr, axis=1, keepdims=True), 1e-12)
            rn = np.einsum("vij,vj->vi", A, region[:, None] * (nr - n0))[idx]
            w = np.array([CL.corrective_weight(rig, entries[k + "_l"]) for k in kn])
            wtab.append(w)
            B = np.einsum("k,vij->vikj", w, A[idx]).reshape(len(idx), 3, 3 * nk)   # [w_1 A, ..., w_K A]
            Mn += np.einsum("vai,vaj->vij", B, B)
            rhs += np.einsum("vai,va->vi", B, r)
            rhs_n += np.einsum("vai,va->vi", B, rn)
        Mn += lam * np.eye(3 * nk)
        sol = np.linalg.solve(Mn, np.stack([rhs, rhs_n], axis=2))
        D = sol[:, :, 0].reshape(len(idx), nk, 3)
        DN = sol[:, :, 1].reshape(len(idx), nk, 3)
        wt = np.array(wtab)
        log("group %s: driver weights at the key poses (rows = key pose, cols = corrective)\n%s" % (
            G["group"], "\n".join("   %-17s %s" % (K["name"], " ".join("%.2f" % v for v in wt[i]))
                                    for i, K in enumerate(G["keys"]))))
        for j, k in enumerate(kn):
            full = np.zeros((n, 3)); full[idx] = D[:, j]
            fulln = np.zeros((n, 3)); fulln[idx] = DN[:, j]
            for s in ("_l", "_r"):
                e = entries[k + s]
                dl = full * side_w[s][:, None]
                dl[np.linalg.norm(dl, axis=1) < 1e-5] = 0.0
                set_key(body, e["name"], dl).value = 0.0
                set_normal_delta(body, e["name"], fulln * side_w[s][:, None])
                spec.append(e)
                log("corrective %-20s verts %5d  max %5.1f mm  cone %s->%s deg" % (
                    e["name"], int((np.linalg.norm(dl, axis=1) > 1e-5).sum()), 1000 * np.linalg.norm(dl, axis=1).max(),
                    e["angle_from"], e["angle_to"]))
    CL.pose_reset(rig)
    me["rts_correctives"] = ({
        "version": 1,
        "method": "cone",
        "doc": "weight = smoothstep(clamp((angle_from - a) / (angle_from - angle_to), 0, 1)), a = angle (deg) between "
               "the bone's local `axis` (its +Y, the bone direction) expressed in the local frame of bone `ref` and "
               "`target` (unit vector in ref's local frame), both taken from the current global (model-space) "
               "rotations. Set the morph target `name` of this mesh to that weight (additive, before skinning).",
        "correctives": spec})
    return spec


if __name__ == "__main__":
    # re-generate the correctives of an export .blend (e.g. after tuning GROUPS) and re-export its GLB
    import chr_lib as CL
    args = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    kind = args[0] if args else "male"
    rig = bpy.data.objects["rts_" + kind]
    body = bpy.data.objects[kind + "_body"]
    build_correctives(rig, body, kind, log=CL.log)
    dev = CL.rest_deviation(rig)
    assert max(dev.values()) < 1e-4, dev
    CL.export_glb(rig, os.path.join(CL.OUT, "base_%s.glb" % kind))
    CL.log("corrective morph normals written:", export_corrective_normals(os.path.join(CL.OUT, "base_%s.glb" % kind), body))
    bpy.ops.wm.save_as_mainfile(filepath=os.path.join(CL.OUT, "base_%s_export.blend" % kind), relative_remap=True)
