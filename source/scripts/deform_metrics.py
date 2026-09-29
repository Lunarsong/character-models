"""Deformation metrics for a posed, skinned body (run inside Blender; used by deform_test.py and correctives.py).

For a region (vertices within a radius of a joint in the rest pose) of the evaluated (posed) body it measures:
  fold_max    largest dihedral angle (deg) between two adjacent faces, minus that edge's rest angle: a hard crease /
              fold / spike shows as a big number (smooth skin bending stays < ~35)
  folds       number of edges whose dihedral grew by more than 60 deg
  flips       faces whose posed normal points against the normal the skinning transform predicts (inverted faces)
  isect       pairs of non-adjacent faces that intersect (skin passing through skin)
  comp / str  1st / 99th percentile of edge length ratio posed / rest (0.6 = edges squashed to 60 %)
  vol         region 'cone' volume posed / rest around the joint centre (rigid motion about the joint keeps it at 1;
              LBS collapse drops it)
  spike       99th percentile of the umbrella-Laplacian length ratio posed / rest (local sharpening)
"""
import bpy, math
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree


class Topo:
    """Rest topology + rest geometry of a mesh (vertex positions from the mesh, i.e. rest / bind pose)."""

    def __init__(self, obj):
        me = obj.data
        self.obj = obj
        self.n = len(me.vertices)
        co = np.empty(self.n * 3); me.vertices.foreach_get("co", co); self.rest = co.reshape(-1, 3)
        e = np.empty(len(me.edges) * 2, dtype=np.int64); me.edges.foreach_get("vertices", e); self.E = e.reshape(-1, 2)
        ls = np.empty(len(me.polygons), dtype=np.int64); me.polygons.foreach_get("loop_start", ls)
        lt = np.empty(len(me.polygons), dtype=np.int64); me.polygons.foreach_get("loop_total", lt)
        lv = np.empty(len(me.loops), dtype=np.int64); me.loops.foreach_get("vertex_index", lv)
        le = np.empty(len(me.loops), dtype=np.int64); me.loops.foreach_get("edge_index", le)
        self.polys = [lv[s:s + t].tolist() for s, t in zip(ls, lt)]
        self.pl_start, self.pl_tot, self.lv = ls, lt, lv
        # triangles (fan) for normals / volume
        tris, tri_face = [], []
        for f, p in enumerate(self.polys):
            for k in range(1, len(p) - 1):
                tris.append((p[0], p[k], p[k + 1])); tri_face.append(f)
        self.T = np.array(tris, dtype=np.int64); self.T_face = np.array(tri_face)
        # edge -> two faces
        ef = [[] for _ in range(len(self.E))]
        for f in range(len(ls)):
            for li in range(ls[f], ls[f] + lt[f]):
                ef[le[li]].append(f)
        pairs = [(e, a[0], a[1]) for e, a in enumerate(ef) if len(a) == 2]
        self.EF = np.array(pairs, dtype=np.int64)
        deg = np.zeros(self.n); np.add.at(deg, self.E[:, 0], 1); np.add.at(deg, self.E[:, 1], 1)
        self.deg = np.maximum(deg, 1)
        self.rest_fn = self.face_normals(self.rest)
        self.rest_dih = self.dihedral(self.rest_fn)
        self.rest_len = np.linalg.norm(self.rest[self.E[:, 0]] - self.rest[self.E[:, 1]], axis=1)
        self.rest_lap = np.linalg.norm(self.lap(self.rest), axis=1)

    def face_normals(self, X):
        a, b, c = X[self.T[:, 0]], X[self.T[:, 1]], X[self.T[:, 2]]
        tn = np.cross(b - a, c - a)
        fn = np.zeros((len(self.polys), 3)); np.add.at(fn, self.T_face, tn)
        return fn / np.maximum(np.linalg.norm(fn, axis=1, keepdims=True), 1e-12)

    def dihedral(self, fn):
        d = (fn[self.EF[:, 1]] * fn[self.EF[:, 2]]).sum(1)
        return np.degrees(np.arccos(np.clip(d, -1, 1)))

    def lap(self, X):
        s = np.zeros_like(X); np.add.at(s, self.E[:, 0], X[self.E[:, 1]]); np.add.at(s, self.E[:, 1], X[self.E[:, 0]])
        return s / self.deg[:, None] - X


def evaluated_co(obj):
    dg = bpy.context.evaluated_depsgraph_get()
    ev = obj.evaluated_get(dg); me = ev.to_mesh()
    co = np.empty(len(me.vertices) * 3); me.vertices.foreach_get("co", co)
    ev.to_mesh_clear()
    return co.reshape(-1, 3)


def weight_matrix(obj, rig):
    """(n_verts, n_bones) dense weights of `obj` for the rig's deform bones (bone order = rig.data.bones)."""
    names = [b.name for b in rig.data.bones]
    bi = {n: i for i, n in enumerate(names)}
    gmap = {g.index: bi[g.name] for g in obj.vertex_groups if g.name in bi}
    W = np.zeros((len(obj.data.vertices), len(names)))
    for v in obj.data.vertices:
        for g in v.groups:
            j = gmap.get(g.group)
            if j is not None:
                W[v.index, j] += g.weight
    return W, names


def skin_matrices(rig):
    """(n_bones, 4, 4) armature-space skinning matrices pose @ rest^-1, bone order = rig.data.bones."""
    out = []
    for b in rig.data.bones:
        pb = rig.pose.bones[b.name]
        out.append(np.array(pb.matrix @ b.matrix_local.inverted()))
    return np.array(out)


def blended_linear(W, M):
    """Per-vertex LBS 3x3 linear part A_v = sum_i w_i M_i[:3,:3] and translation t_v."""
    A = np.einsum("vb,bij->vij", W, M[:, :3, :3])
    t = np.einsum("vb,bi->vi", W, M[:, :3, 3])
    return A, t


def region_mask(topo, centre, radius):
    return np.linalg.norm(topo.rest - np.asarray(centre), axis=1) < radius


def measure(topo, X, mask, centre_rest, centre_pose, A=None, isect=True):
    """Metrics of posed positions X (n,3) on the region `mask` (bool per vertex)."""
    fmask = np.array([all(mask[v] for v in p) for p in topo.polys])
    emask = mask[topo.E[:, 0]] & mask[topo.E[:, 1]]
    efm = fmask[topo.EF[:, 1]] & fmask[topo.EF[:, 2]]
    fn = topo.face_normals(X)
    dih = topo.dihedral(fn)
    grow = (dih - topo.rest_dih)[efm]
    r = {}
    r["fold_max"] = round(float(grow.max()), 1) if len(grow) else 0.0
    r["folds"] = int((grow > 60).sum())
    r["fold_sum"] = round(float(np.clip(grow - 30, 0, None).sum()), 0)      # total crease beyond 30 deg per edge
    if len(grow):
        e = topo.EF[np.nonzero(efm)[0][int(np.argmax(grow))], 0]
        r["fold_at"] = int(topo.E[e, 0])                                    # vertex index at the worst crease
    if A is not None:
        # expected normal: cofactor of the face's mean skinning matrix applied to the rest normal
        fidx = np.nonzero(fmask)[0]
        Am = np.array([A[topo.polys[f]].mean(0) for f in fidx])
        cof = np.linalg.inv(Am).transpose(0, 2, 1) * np.linalg.det(Am)[:, None, None]
        ne = np.einsum("fij,fj->fi", cof, topo.rest_fn[fidx])
        r["flips"] = int(((ne * fn[fidx]).sum(1) < 0).sum())
    L = np.linalg.norm(X[topo.E[:, 0]] - X[topo.E[:, 1]], axis=1)[emask] / np.maximum(topo.rest_len[emask], 1e-9)
    r["comp"] = round(float(np.percentile(L, 1)), 3)
    r["str"] = round(float(np.percentile(L, 99)), 3)
    r["comp_min"] = round(float(L.min()), 3)
    # cone volume of the region's triangles around the joint centre
    tm = fmask[topo.T_face]
    T = topo.T[tm]

    def cone(P, o):
        a, b, c = P[T[:, 0]] - o, P[T[:, 1]] - o, P[T[:, 2]] - o
        return float((a * np.cross(b, c)).sum() / 6.0)
    v0 = cone(topo.rest, np.asarray(centre_rest)); v1 = cone(X, np.asarray(centre_pose))
    r["vol"] = round(v1 / v0, 3) if abs(v0) > 1e-9 else 0.0
    lp = np.linalg.norm(topo.lap(X), axis=1)[mask] / np.maximum(topo.rest_lap[mask], 1e-5)
    r["spike"] = round(float(np.percentile(lp, 99)), 2)
    if isect:
        fidx = np.nonzero(fmask)[0]
        polys = [topo.polys[f] for f in fidx]
        tree = BVHTree.FromPolygons([Vector(p) for p in X], polys, all_triangles=False, epsilon=0.0)
        n = 0
        for i, j in tree.overlap(tree):
            if i < j and not (set(polys[i]) & set(polys[j])):
                n += 1
        r["isect"] = n
    return r
