"""Shared code for the CIVILIAN outfits (peasant, archer): garment geometry built from the MakeHuman basemesh, MPFB
clothes assets (MakeClothes .mhclo), materials from the civ texture sets, sockets, dressing, per-piece and dressed
export. Blender Python + numpy; imported by outfit_peasant.py / outfit_archer.py / outfit_civ_qa.py.

Garment geometry (character faces -Y, +X = its left, +Z up, metres):
  * body-following garments (shirt, tunic, gambeson, trousers, shoes) start from MakeHuman's `helper-tights` region
    (a full-body suit ~6 mm off the skin that is part of the basemesh, so it fits every body). Every garment vertex
    keeps the rig weights of the tights vertex it grew from, so ALL layers built this way share their skinning per
    column (shirt / tunic / gambeson vertices above the same tights vertex transform with the same matrix: layer order
    survives any pose under linear blend skinning).
  * drape(): an offset shell relaxed with Laplacian smoothing and pushed out of the skin and the under-layers to a
    per-layer clearance (cloth bridges hollows, never dips into them).
  * hanging parts (tunic / gambeson skirts) are rows hung from the waist boundary loop over a non-shrinking hull of the
    legs + under-layers, split front / back (each half follows its thigh), with flute folds and flare.
  * hems: every open edge gets a turned-in rim (visible cloth thickness).
  * UVs are metric (metres) per region (cylindrical around the torso / limb axes) and scaled by the texture set's
    uv_per_m when the asset is written; trim-strip faces map into civ_trim strips.
Assets: assets/mpfb_assets/clothes/rts_<outfit>_<piece>/ (.mhclo .obj .mhmat .mhw .rts.json), installed into MPFB.
"""
import sys, os, json, math, time, shutil, uuid
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from chr_lib import *
from mathutils import Vector, Matrix
from mathutils.bvhtree import BVHTree
from mathutils.kdtree import KDTree

ASSET_DIR = os.path.join(ASSETS, "mpfb_assets", "clothes")
CIV_TEX = os.path.join(TEX, "civ")
OUTC = os.path.join(OUT, "civ")
RENC = os.path.join(REN, "civ")
NB = 13380                      # hm08 basemesh: vertices 0..13379 are the skin, the rest are helpers
SIDES = ("l", "r")


def clog(*a):
    print("CIV", *a, flush=True)


# ================================================================================================= small math
def nrm(v, axis=-1):
    v = np.asarray(v, dtype=np.float64)
    return v / np.maximum(np.linalg.norm(v, axis=axis, keepdims=True), 1e-12)


def smoothstep(e0, e1, x):
    t = np.clip((np.asarray(x, dtype=np.float64) - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def lerp(a, b, t):
    return a + (b - a) * t


def v3(a):
    return Vector((float(a[0]), float(a[1]), float(a[2])))


def gauss1d(a, sigma, axis, wrap=False):
    if sigma <= 0:
        return a
    r = int(math.ceil(3 * sigma))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2); k /= k.sum()
    a = np.moveaxis(a, axis, 0)
    p = np.concatenate([a[-r:], a, a[:r]], 0) if wrap else np.concatenate([np.repeat(a[:1], r, 0), a, np.repeat(a[-1:], r, 0)], 0)
    out = np.zeros_like(a)
    for i, w in enumerate(k):
        out += w * p[i:i + len(a)]
    return np.moveaxis(out, 0, axis)


def adjacency(n, F):
    nb = [set() for _ in range(n)]
    for f in F:
        k = len(f)
        for i in range(k):
            a, b = f[i], f[(i + 1) % k]
            nb[a].add(b); nb[b].add(a)
    return [sorted(s) for s in nb]


def vertex_normals(P, F):
    N = np.zeros_like(P)
    for f in F:
        f = list(f)
        for k in range(1, len(f) - 1):
            fn = np.cross(P[f[k]] - P[f[0]], P[f[k + 1]] - P[f[0]])
            for i in (f[0], f[k], f[k + 1]):
                N[i] += fn
    return nrm(N)


def boundary_loops(F):
    """ordered boundary loops (vertex index lists) of a polygon soup, oriented along the face winding"""
    cnt = {}
    for f in F:
        k = len(f)
        for i in range(k):
            a, b = f[i], f[(i + 1) % k]
            e = (min(a, b), max(a, b))
            cnt[e] = cnt.get(e, 0) + 1
    nxt = {}
    for f in F:
        k = len(f)
        for i in range(k):
            a, b = f[i], f[(i + 1) % k]
            if cnt[(min(a, b), max(a, b))] == 1:
                nxt[a] = b
    loops, seen = [], set()
    for s in list(nxt):
        if s in seen:
            continue
        lp = [s]; seen.add(s); c = nxt[s]
        while c != s and c not in seen and c in nxt:
            lp.append(c); seen.add(c); c = nxt[c]
        loops.append(lp)
    return loops


# ================================================================================================= the body
class CivBody:
    """Rest shape of the live MPFB human (modelling keys mixed, i.e. the fitted body) + rig weights per basemesh
    vertex (skin AND helpers), skin BVH / normals / KD-tree, bone joints."""

    def __init__(self, rig, bm):
        self.rig, self.bm = rig, bm
        me = bm.data
        tmp = bm.shape_key_add(name="__civ_mix", from_mix=True)
        co = np.empty(len(me.vertices) * 3); tmp.data.foreach_get("co", co)
        bm.shape_key_remove(tmp)
        self.co_all = co.reshape(-1, 3)
        self.co = self.co_all[:NB]
        self.faces = [tuple(p.vertices) for p in me.polygons if max(p.vertices) < NB]
        tris = []
        for f in self.faces:
            for k in range(1, len(f) - 1):
                tris.append((f[0], f[k], f[k + 1]))
        self.tris = np.array(tris)
        self.bvh = BVHTree.FromPolygons([v3(c) for c in self.co], self.faces)
        self.vn = vertex_normals(self.co, self.faces)
        self.kd = KDTree(NB)
        for i, p in enumerate(self.co):
            self.kd.insert(p, i)
        self.kd.balance()
        bones = {b.name for b in rig.data.bones}
        self.bones = bones
        gname = {g.index: g.name for g in bm.vertex_groups}
        self.W = [dict() for _ in range(len(me.vertices))]
        self.groups = {}
        for v in me.vertices:
            for g in v.groups:
                nm = gname[g.group]
                if nm in bones:
                    if g.weight > 1e-4:
                        self.W[v.index][nm] = g.weight
                elif g.weight > 0.5:
                    self.groups.setdefault(nm, []).append(v.index)
        self.groups = {k: np.array(v) for k, v in self.groups.items()}
        B = rig.data.bones
        self.H = {b.name: np.array(b.head_local[:]) for b in B}
        self.T = {b.name: np.array(b.tail_local[:]) for b in B}
        # dominant bone per skin vertex
        self.dom = [max(w.items(), key=lambda kv: kv[1])[0] if w else "" for w in self.W]
        # the tights helper as the common base of every body garment: per basemesh vertex its outward normal (on the
        # whole helper, so all garments share it) and its distance from the skin
        ti = self.groups["helper-tights"]
        ins = np.zeros(len(me.vertices), bool); ins[ti] = True
        tf = [tuple(p.vertices) for p in me.polygons if ins[list(p.vertices)].all()]
        tN = vertex_normals(self.co_all, tf)
        dots = [np.dot(self.co_all[i] - self.nearest(self.co_all[i])[0], tN[i]) for i in ti[::9]]
        if np.mean(dots) < 0:
            tN = -tN
        self.tN = tN
        self.td = np.zeros(len(me.vertices))
        for i in ti:
            loc, n, d = self.nearest(self.co_all[i])
            self.td[i] = d if np.dot(self.co_all[i] - loc, n) >= 0 else -d
        self._subdivide_tights(tf)

    # ------------------------------------------------------------------------------------------- garment lines
    FINE_BONES = ("upperarm", "lowerarm", "hand", "thumb", "index", "middle", "ring", "pinky", "thigh", "calf")

    def _subdivide_tights(self, tf):
        """One Catmull-Clark level over the WHOLE tights helper, computed once per body. Every garment line is keyed:
        an int (basemesh vertex, CC-smoothed position), ('e', a, b) edge points, ('f', a, b, c, d) face points. Limb
        faces (all vertices dominated by arm / leg bones) are used subdivided (the helper is only 14 vertices around a
        sleeve: faceted at hero scale), trunk faces stay coarse; a coarse face bordering a fine one is fanned around its
        own face point (no T-junction cracks). All garments of all outfits read the same keys, positions, normals,
        skin distances and weights, so the shared-lines layering holds on the subdivided lines too."""
        co = self.co_all
        faces = [tuple(int(i) for i in f) for f in tf]
        ekey = lambda a, b: ("e", min(a, b), max(a, b))
        fkey = lambda f: ("f",) + tuple(sorted(f))
        edge_faces = {}
        for fi, f in enumerate(faces):
            for k in range(len(f)):
                edge_faces.setdefault(ekey(f[k], f[(k + 1) % len(f)]), []).append(fi)
        fp = {fkey(f): co[list(f)].mean(0) for f in faces}
        P, W = {}, {}
        for e, fl in edge_faces.items():
            a, b = e[1], e[2]
            if len(fl) == 2:
                P[e] = (co[a] + co[b] + fp[fkey(faces[fl[0]])] + fp[fkey(faces[fl[1]])]) / 4
            else:
                P[e] = (co[a] + co[b]) / 2
            W[e] = mix_raw([self.W[a], self.W[b]])
        for f in faces:
            P[fkey(f)] = fp[fkey(f)]
            W[fkey(f)] = mix_raw([self.W[i] for i in f])
        vf, ve = {}, {}
        for fi, f in enumerate(faces):
            for i in f:
                vf.setdefault(i, []).append(fi)
        for e in edge_faces:
            for i in (e[1], e[2]):
                ve.setdefault(i, []).append(e)
        for i, es in ve.items():
            bnd = [e for e in es if len(edge_faces[e]) == 1]
            if bnd:
                nbv = [e[1] if e[2] == i else e[2] for e in bnd]
                P[i] = (co[nbv[0]] + co[nbv[-1]] + 6 * co[i]) / 8 if len(nbv) >= 2 else co[i].copy()
            else:
                n = len(es)
                Q = np.mean([fp[fkey(faces[fi])] for fi in vf[i]], 0)
                R = np.mean([(co[e[1]] + co[e[2]]) / 2 for e in es], 0)
                P[i] = (Q + 2 * R + (n - 3) * co[i]) / n
            W[i] = dict(self.W[i])
        # which coarse faces are used subdivided (identical decision for every garment)
        self.fine_face = {}
        for f in faces:
            self.fine_face[fkey(f)] = all(self.dom[i].startswith(self.FINE_BONES) for i in f)
        # the subdivided full mesh (for normals): fine faces split in 4, coarse faces fanned / kept
        self._faces, self._edge_faces = faces, edge_faces
        sub = []
        for f in faces:
            sub += self._sub_faces(f)
        keys = sorted(P, key=str)
        kid = {k: n for n, k in enumerate(keys)}
        Pa = np.array([P[k] for k in keys])
        N = vertex_normals(Pa, [tuple(kid[k] for k in f) for f in sub])
        # orient like tN (outward)
        s = np.mean([np.dot(N[kid[i]], self.tN[i]) for i in list(ve)[::11]])
        if s < 0:
            N = -N
        self.lP = {k: Pa[kid[k]] for k in keys}
        self.lN = {k: N[kid[k]] for k in keys}
        self.lW = W
        self.ld = {}
        for k in keys:
            loc, n, d = self.nearest(self.lP[k])
            self.ld[k] = d if np.dot(self.lP[k] - loc, n) >= 0 else -d

    def _sub_faces(self, f):
        """faces (tuples of line keys) replacing coarse tights face f (basemesh indices, original winding)"""
        ekey = lambda a, b: ("e", min(a, b), max(a, b))
        fc = ("f",) + tuple(sorted(f))
        n = len(f)
        if self.fine_face[fc]:
            return [(f[k], ekey(f[k], f[(k + 1) % n]), fc, ekey(f[(k - 1) % n], f[k])) for k in range(n)]
        ring = []
        split = False
        for k in range(n):
            a, b = f[k], f[(k + 1) % n]
            ring.append(a)
            e = ekey(a, b)
            if any(self.fine_face[("f",) + tuple(sorted(self._faces[fi]))] for fi in self._edge_faces[e]):
                ring.append(e); split = True
        if not split:
            return [tuple(f)]
        return [(fc, ring[k], ring[(k + 1) % len(ring)]) for k in range(len(ring))]

    def line_N(self, keys):
        return np.array([self.lN[k] for k in keys])

    def line_d(self, keys):
        return np.array([self.ld[k] for k in keys])

    def line_P(self, keys):
        return np.array([self.lP[k] for k in keys])

    def line_W(self, keys):
        return [self.lW[k] for k in keys]

    def tights_region(self, keep_face=None):
        """the tights faces whose (coarse) centroid passes keep_face(c, basemesh indices), subdivided as above:
        returns (line keys, points, faces in local indices)"""
        co = self.co_all
        faces = self._faces
        if keep_face is not None:
            faces = [f for f in faces if keep_face(co[list(f)].mean(0), f)]
        sub = []
        for f in faces:
            sub += self._sub_faces(f)
        used = sorted(set(k for f in sub for k in f), key=str)
        remap = {k: n for n, k in enumerate(used)}
        return used, self.line_P(used), [tuple(remap[k] for k in f) for f in sub]

    def wsum(self, prefixes, idx=None):
        idx = range(NB) if idx is None else idx
        return np.array([sum(x for b, x in self.W[i].items() if b.startswith(tuple(prefixes))) for i in idx])

    def region(self, group, keep_face=None):
        """faces of a basemesh vertex group (e.g. 'helper-tights') whose centroid passes keep_face(c, widx):
        returns (basemesh indices, points, faces in local indices)"""
        me = self.bm.data
        ins = np.zeros(len(me.vertices), bool); ins[self.groups[group]] = True
        faces = [tuple(p.vertices) for p in me.polygons if ins[list(p.vertices)].all()]
        if keep_face is not None:
            faces = [f for f in faces if keep_face(self.co_all[list(f)].mean(0), f)]
        used = sorted(set(i for f in faces for i in f)); remap = {v: k for k, v in enumerate(used)}
        return np.array(used), self.co_all[used].copy(), [tuple(remap[i] for i in f) for f in faces]

    def nearest(self, p):
        loc, n, i, d = self.bvh.find_nearest(v3(p))
        return np.array(loc[:]), np.array(n[:]), d

    def weights_at(self, p):
        """barycentric blend of the skin weights at the nearest skin point"""
        loc, n, fi, d = self.bvh.find_nearest(v3(p))
        f = self.faces[fi]
        P = self.co[list(f)]
        dd = np.linalg.norm(P - np.array(loc[:]), axis=1)
        w = 1.0 / np.maximum(dd, 1e-5) ** 2; w /= w.sum()
        out = {}
        for k, vi in enumerate(f):
            for b, x in self.W[vi].items():
                out[b] = out.get(b, 0.0) + x * w[k]
        return out


FACE_BONE_PREFIX = ("jaw", "eye", "eyelid", "brow", "cheek", "nose", "mouth", "lip", "tongue")


def head_only(w):
    """face bones merged into head (garments must never follow the jaw, lips, cheeks ...)"""
    out = {}
    for b, x in w.items():
        k = "head" if b.startswith(FACE_BONE_PREFIX) else b
        out[k] = out.get(k, 0.0) + x
    return out


def norm_w(w, limit=4):
    it = sorted(((x, b) for b, x in w.items() if x > 1e-4), reverse=True)[:limit]
    s = sum(x for x, _ in it)
    return {b: x / s for x, b in it} if s > 0 else {}


def mix_raw(ws):
    """plain average of weight dicts (no normalisation / limit)"""
    out = {}
    for w in ws:
        for k, x in w.items():
            out[k] = out.get(k, 0.0) + x / len(ws)
    return out


def mix_w(a, b, t):
    out = {}
    for k, x in a.items():
        out[k] = out.get(k, 0.0) + x * (1 - t)
    for k, x in b.items():
        out[k] = out.get(k, 0.0) + x * t
    return norm_w(out)


# ================================================================================================= mesh builder
class CMesh:
    """quad-mesh builder: per-vertex weights (bone -> w) and MakeClothes match group, per-corner UVs (metric, see the
    module doc), per-face slot names"""

    def __init__(self):
        self.V, self.F, self.UV, self.S, self.W, self.G = [], [], [], [], [], []
        self.HF = []                    # per face: True for turned-in hem faces (not a layer surface)

    def add_v(self, p, w=None, g="body"):
        self.V.append(tuple(float(x) for x in p)); self.W.append(w); self.G.append(g)
        return len(self.V) - 1

    def add_f(self, vs, uvs, slot, hem=False):
        self.F.append(tuple(int(i) for i in vs)); self.UV.append([tuple(float(x) for x in u) for u in uvs])
        self.S.append(slot); self.HF.append(hem)

    def outer_faces(self):
        """faces that form the garment's outer surface (hem strips excluded): the surface later layers clear"""
        return [f for f, h in zip(self.F, self.HF) if not h]

    def merge(self, o):
        off = len(self.V)
        self.V += o.V; self.W += o.W; self.G += o.G
        for f, uv, s, h in zip(o.F, o.UV, o.S, o.HF):
            self.F.append(tuple(i + off for i in f)); self.UV.append(uv); self.S.append(s); self.HF.append(h)
        return off

    def arr(self):
        return np.array(self.V)

    def set_pos(self, P):
        self.V = [tuple(map(float, p)) for p in P]

    def tris(self):
        return sum(len(f) - 2 for f in self.F)

    def triangulate(self):
        F, UV, S, HF = [], [], [], []
        for f, uv, s, h in zip(self.F, self.UV, self.S, self.HF):
            if len(f) == 3:
                F.append(f); UV.append(uv); S.append(s); HF.append(h)
            else:
                for a, b, c in ((0, 1, 2), (0, 2, 3)):
                    F.append((f[a], f[b], f[c])); UV.append([uv[a], uv[b], uv[c]]); S.append(s); HF.append(h)
        self.F, self.UV, self.S, self.HF = F, UV, S, HF

    def mixed(self):
        return len(set(len(f) for f in self.F)) > 1

    def mirror_x(self, swap=True):
        m = CMesh()
        m.V = [(-x, y, z) for x, y, z in self.V]
        m.W = [({swap_side(k): v for k, v in w.items()} if (w and swap) else w) for w in self.W]
        m.G = list(self.G)
        m.F = [tuple(f[::-1]) for f in self.F]; m.UV = [list(uv[::-1]) for uv in self.UV]; m.S = list(self.S)
        m.HF = list(self.HF)
        return m

    def weld(self, tol=1e-6):
        """merge coincident vertices (keeps the first); removes degenerate faces"""
        P = self.arr()
        kd = KDTree(len(P))
        for i, p in enumerate(P):
            kd.insert(p, i)
        kd.balance()
        rep = np.arange(len(P))
        for i, p in enumerate(P):
            if rep[i] != i:
                continue
            for co, j, d in kd.find_range(p, tol):
                if j > i and rep[j] == j:
                    rep[j] = i
        keep = sorted(set(rep.tolist()))
        remap = {o: k for k, o in enumerate(keep)}
        self.V = [self.V[i] for i in keep]; self.W = [self.W[i] for i in keep]; self.G = [self.G[i] for i in keep]
        F, UV, S, HF = [], [], [], []
        for f, uv, s, h in zip(self.F, self.UV, self.S, self.HF):
            nf = [remap[rep[i]] for i in f]
            if len(set(nf)) == len(nf):
                F.append(tuple(nf)); UV.append(uv); S.append(s); HF.append(h)
            elif len(set(nf)) == 3:                     # a quad with one collapsed edge -> triangle
                keep_k = [k for k in range(len(nf)) if nf[k] != nf[k - 1]]
                F.append(tuple(nf[k] for k in keep_k)); UV.append([uv[k] for k in keep_k]); S.append(s); HF.append(h)
        self.F, self.UV, self.S, self.HF = F, UV, S, HF

    def to_object(self, name, slots=None, smooth=True):
        me = bpy.data.meshes.new(name)
        me.from_pydata(self.V, [], self.F)
        uvl = me.uv_layers.new(name="UVMap")
        uvl.data.foreach_set("uv", np.array([u for f in self.UV for u in f], dtype=np.float32).ravel())
        slots = slots or sorted(set(self.S))
        me.polygons.foreach_set("material_index", np.array([slots.index(s) for s in self.S], dtype=np.int32))
        me.polygons.foreach_set("use_smooth", np.full(len(self.F), smooth))
        me.validate(clean_customdata=False)
        me.update()
        ob = bpy.data.objects.new(name, me)
        bpy.context.scene.collection.objects.link(ob)
        return ob


def swap_side(name):
    if name.endswith("_l"):
        return name[:-2] + "_r"
    if name.endswith("_r"):
        return name[:-2] + "_l"
    return name


# ================================================================================================= ray targets
class Target:
    """a surface garments are pushed out of: the skin or an under-layer mesh (outward winding), with a clearance"""

    def __init__(self, P, F, clear, name="", reach=None):
        self.P = np.asarray(P, float); self.F = [tuple(f) for f in F]
        self.bvh = BVHTree.FromPolygons([v3(p) for p in self.P], self.F)
        self.clear = clear; self.name = name
        # open surfaces (under-layers): the sign of the nearest-point test is only meaningful close to the surface
        self.reach = reach if reach is not None else (None if name == "skin" else 0.035)

    @staticmethod
    def of_mesh(m, clear, name=""):
        """a finished piece as an under-layer: its outer surface only (turned-in hems excluded)"""
        return Target(m.arr(), m.outer_faces(), clear, name)

    def signed(self, p):
        if self.reach is not None:
            loc, n, i, d = self.bvh.find_nearest(v3(p), self.reach)
        else:
            loc, n, i, d = self.bvh.find_nearest(v3(p))
        if loc is None:
            return None
        loc = np.array(loc[:]); n = np.array(n[:])
        s = 1.0 if np.dot(p - loc, n) >= 0 else -1.0
        return loc, n, s * d

    def ray(self, o, d, maxd=1.0):
        hit = self.bvh.ray_cast(v3(o), v3(nrm(d)), maxd)
        return None if hit[0] is None else hit[3]


def skin_target(cb, clear, exclude=None):
    """the skin (optionally without faces whose vertices are all dominated by bones starting with `exclude`)"""
    F = cb.faces
    if exclude:
        F = [f for f in F if not all(cb.dom[i].startswith(exclude) for i in f)]
    return Target(cb.co, F, clear, "skin")


class Composite:
    """ray target made of several Targets (hull sampling over the body + the layers under a garment)"""

    def __init__(self, targets):
        V, P = [], []
        for t in targets:
            o = len(V)
            V += [v3(p) for p in t.P]
            P += [tuple(i + o for i in f) for f in t.F]
        self.bvh = BVHTree.FromPolygons(V, P)

    def ray(self, o, d, maxd=1.0):
        hit = self.bvh.ray_cast(v3(o), v3(nrm(d)), maxd)
        return None if hit[0] is None else hit[3]


# ================================================================================================= drape / offset
def drape(P, F, targets, iters=10, lam=0.45, fixed=None, clear_scale=None, boundary_lam=0.25, verbose=""):
    """Relax (umbrella Laplacian) + push out of every target to its clearance (per-vertex array or scalar), repeated.
    fixed: bool mask of vertices that never move. Returns the new points."""
    P = np.array(P, float)
    n = len(P)
    nb = adjacency(n, F)
    bset = set(i for lp in boundary_loops(F) for i in lp)
    bnb = {}
    for i in bset:                                     # boundary vertices relax along the boundary only
        bnb[i] = [j for j in nb[i] if j in bset]
    fixed = np.zeros(n, bool) if fixed is None else fixed
    for it in range(iters):
        Q = P.copy()
        for i in range(n):
            if fixed[i]:
                continue
            if i in bset:
                nn = bnb[i]
                if len(nn) >= 2:
                    Q[i] = P[i] + boundary_lam * (P[nn].mean(0) - P[i])
            elif nb[i]:
                Q[i] = P[i] + lam * (P[nb[i]].mean(0) - P[i])
        P = Q
        for t in targets:
            for i in range(n):
                if fixed[i]:
                    continue
                c = t.clear[i] if np.ndim(t.clear) else t.clear
                if clear_scale is not None:
                    c = c * clear_scale
                r = t.signed(P[i])
                if r is None:
                    continue
                loc, nn_, sd = r
                if sd < c:
                    P[i] = loc + nn_ * c
    if verbose:
        clog("drape %s: %d verts, %d iters" % (verbose, n, iters))
    return P


def drape_lines(cb, idx, F, T, clear, unders=(), iters=30, lam=0.5, fixed=None, verbose="", bridge=0):
    """Body garment on NORMAL LINES: vertex i = T_i + N_i * h_i with N_i the tights normal of its basemesh origin
    idx[i]. Only the heights are solved (umbrella-smoothed, clamped to h >= clearance - tights distance and to the
    under-layers' heights + gap on the same line). All tights garments of all outfits share these lines, so
    under linear blend skinning (identical weights per line) their order along each line survives every pose.
    unders: [(dict basemesh index -> height of the under-layer, gap)]. Returns (P, h)."""
    n = len(idx)
    N = cb.line_N(idx)
    d0 = cb.line_d(idx)
    hmin = np.asarray(clear, float) * np.ones(n) - d0
    for hu, gap in unders:
        for k, i in enumerate(idx):
            if i in hu:
                hmin[k] = max(hmin[k], hu[i] + gap)
    h = hmin.copy()
    nb = adjacency(n, F)
    fixed = np.zeros(n, bool) if fixed is None else fixed
    if bridge:
        # cloth tension: smooth the offset surface freely (fills hollows, flattens bumps), keep what lies above each
        # line's minimum height -> the garment bridges concave spots (chest / spine groove / small of the back)
        Ps = T + N * hmin[:, None]
        bset = set(i for lp in boundary_loops(F) for i in lp)
        for _ in range(bridge):
            Q = Ps.copy()
            for k in range(n):
                if nb[k] and k not in bset:
                    Q[k] = Ps[k] + 0.5 * (Ps[nb[k]].mean(0) - Ps[k])
            Ps = Q
        h = np.maximum(hmin, np.einsum("ij,ij->i", Ps - T, N))
    for it in range(iters):
        hn = h.copy()
        for k in range(n):
            if nb[k] and not fixed[k]:
                hn[k] = h[k] + lam * (h[nb[k]].mean() - h[k])
        h = np.maximum(hn, hmin)
    P = T + N * h[:, None]
    # the skin can still be closer than wanted where the lines converge (concave spots): raise h there (capped,
    # smoothed; a far negative distance means the nearest skin is another limb, not this line's skin)
    raise_ = np.zeros(n)
    for k in range(n):
        c = clear[k] if np.ndim(clear) else clear
        loc, nn_, d = cb.nearest(P[k])
        sd = d if np.dot(P[k] - loc, nn_) >= 0 else -d
        if -0.01 < sd < 0.8 * c:
            raise_[k] = min(0.8 * c - sd, 0.01)
    for _ in range(2):
        raise_ = np.maximum(raise_, np.array([raise_[nb[k]].mean() if nb[k] else 0.0 for k in range(n)]) * 0.7)
    h = h + raise_
    P = T + N * h[:, None]
    if verbose:
        clog("drape_lines %s: %d verts, h %.1f..%.1f mm" % (verbose, n, h.min() * 1000, h.max() * 1000))
    return P, h


def resolve_layers(m, targets, iters=4, lam=0.35, verbose=""):
    """final layering pass on a finished piece (hems included): every vertex is pushed out of each target (a Target
    of the finished under-layer / the skin, clearance = the gap) and the moved region is relaxed, repeated"""
    P = m.arr()
    n = len(P)
    nb = adjacency(n, m.F)
    moved_total = np.zeros(n, bool)
    for it in range(iters):
        moved = np.zeros(n, bool)
        for t in targets:
            for i in range(n):
                c = t.clear[i] if np.ndim(t.clear) else t.clear
                r = t.signed(P[i])
                if r is None:
                    continue
                loc, nn_, sd = r
                if sd < c:
                    P[i] = loc + nn_ * c
                    moved[i] = True
        if not moved.any():
            break
        moved_total |= moved
        # relax the neighbourhood of the moved vertices (keeps the fix smooth), then push again next iteration
        ring = moved.copy()
        for i in np.nonzero(moved)[0]:
            ring[nb[i]] = True
        Q = P.copy()
        for i in np.nonzero(ring)[0]:
            if nb[i]:
                Q[i] = P[i] + lam * (P[nb[i]].mean(0) - P[i])
        P = Q
    # final push without relaxing
    for t in targets:
        for i in range(n):
            c = t.clear[i] if np.ndim(t.clear) else t.clear
            r = t.signed(P[i])
            if r is not None and r[2] < c:
                P[i] = r[0] + r[1] * c
                moved_total[i] = True
    m.set_pos(P)
    if verbose:
        clog("resolve %s: %d verts moved" % (verbose, int(moved_total.sum())))
    return int(moved_total.sum())


def hull_radius_span(comp, z0, z1, ph, centre=(0.0, 0.012), nz=7, R0=0.75):
    """largest outer-hull radius of a composite along direction ph over the heights z0..z1"""
    best = 0.0
    d = np.array([math.sin(ph), -math.cos(ph), 0.0])
    for z in np.linspace(z0, z1, nz):
        h = comp.ray(np.array([centre[0], centre[1], z]) + d * R0, -d, R0)
        if h is not None:
            best = max(best, R0 - h)
    return best


def min_clearance(P, target):
    return min(target.signed(p)[2] for p in P)


# ================================================================================================= hems
def loop_inward(m, loop):
    """per boundary vertex: tangent direction pointing INTO the surface (perpendicular to the loop)"""
    P = m.arr()
    vf = {}
    for fi, f in enumerate(m.F):
        for v in f:
            vf.setdefault(v, []).append(fi)
    out = []
    n = len(loop)
    for k, v in enumerate(loop):
        a, b = loop[k - 1], loop[(k + 1) % n]
        t = nrm(P[b] - P[a])
        cen = np.mean([np.mean(P[list(m.F[fi])], 0) for fi in vf[v]], 0)
        d = cen - P[v]
        d = nrm(d - t * np.dot(d, t))
        out.append(d)
    return np.array(out)


def add_hem(m, loop, thick=0.003, width=0.012, slot=None, normals=None, uv_scale=1.0, closed=True, v_range=None):
    """turned-in hem along a boundary loop: the edge gets its thickness (a strip going inward along -normal) and a
    turned-under strip lying inside the garment. Weights / match groups copied from the edge vertices. Returns the
    new rings (edge-in, hem)."""
    P = m.arr()
    N = normals if normals is not None else vertex_normals(P, m.F)
    inw = loop_inward(m, loop)
    r1, r2 = [], []
    for k, v in enumerate(loop):
        p = P[v]; nn_ = N[v]
        a = p - nn_ * thick
        b = a + inw[k] * width
        r1.append(m.add_v(a, dict(m.W[v]) if m.W[v] else None, m.G[v]))
        r2.append(m.add_v(b, dict(m.W[v]) if m.W[v] else None, m.G[v]))
    n = len(loop)
    al = [0.0]
    for k in range(1, n + 1):
        al.append(al[-1] + np.linalg.norm(P[loop[k % n]] - P[loop[k - 1]]))
    rng = range(n) if closed else range(n - 1)
    sl = slot
    for k in rng:
        k2 = (k + 1) % n
        u0, u1 = al[k] * uv_scale, al[k + 1] * uv_scale
        s_ = sl if sl else m.S[_face_of_edge(m, loop[k], loop[k2])]
        # the loop runs along the face winding, so (v_k2, v_k, r1_k, r1_k2) keeps the outward orientation
        if v_range is None:
            va, vb, vc = 0.0, thick, thick + width
        else:
            va, vc = v_range; vb = lerp(va, vc, thick / (thick + width))
        m.add_f([loop[k2], loop[k], r1[k], r1[k2]], [(u1, va), (u0, va), (u0, vb), (u1, vb)], s_, hem=True)
        m.add_f([r1[k2], r1[k], r2[k], r2[k2]], [(u1, vb), (u0, vb), (u0, vc), (u1, vc)], s_, hem=True)
    return r1, r2


def _face_of_edge(m, a, b):
    if not hasattr(m, "_edge_face") or m._edge_face_n != len(m.F):
        ef = {}
        for fi, f in enumerate(m.F):
            k = len(f)
            for i in range(k):
                ef[(f[i], f[(i + 1) % k])] = fi
        m._edge_face = ef; m._edge_face_n = len(m.F)
    return m._edge_face.get((a, b), m._edge_face.get((b, a), 0))


# ================================================================================================= UVs
def cyl_uv(P, f, O, A, ref, r_ref):
    """metric cylindrical UVs of one face around axis (O, A): u = angle * r_ref (unwrapped per face), v = along A"""
    A = nrm(A); X = nrm(ref - A * np.dot(ref, A)); Y = np.cross(A, X)
    d = P[list(f)] - O
    ang = np.arctan2(d @ Y, d @ X)
    a0 = ang[0]
    ang = a0 + (ang - a0 + math.pi) % (2 * math.pi) - math.pi
    return [(a * r_ref, t) for a, t in zip(ang, d @ A)]


def region_of(w):
    """body region of a weights dict: 'arm_l'/'arm_r', 'leg_l'/'leg_r', 'foot_l'/'foot_r', 'hand_l'/'hand_r', 'torso', 'head'"""
    acc = {}
    for b, x in w.items():
        s = b[-1] if b.endswith(("_l", "_r")) else ""
        if b.startswith(("upperarm", "lowerarm", "clavicle_")) and not b.startswith("clavicle_"):
            k = "arm_" + s
        elif b.startswith(("hand", "thumb", "index", "middle", "ring", "pinky")):
            k = "hand_" + s
        elif b.startswith(("thigh", "calf")):
            k = "leg_" + s
        elif b.startswith(("foot", "ball")):
            k = "foot_" + s
        elif b in ("head", "neck_02") or b.startswith(FACE_BONE_PREFIX):
            k = "head"
        else:
            k = "torso"
        acc[k] = acc.get(k, 0.0) + x
    return max(acc.items(), key=lambda kv: kv[1])[0] if acc else "torso"


class Axes:
    """uv projection axes of the body regions (from the fitted skeleton)"""

    def __init__(self, cb):
        H, T = cb.H, cb.T
        self.ax = {"torso": (np.array([0.0, 0.02, 0.0]), np.array([0.0, 0.0, 1.0]), np.array([0.0, -1.0, 0.0]), 0.16),
                   "head": (H["head"], np.array([0.0, 0.0, 1.0]), np.array([0.0, -1.0, 0.0]), 0.10)}
        for s in SIDES:
            sh, wr = H["upperarm_" + s], H["hand_" + s]
            self.ax["arm_" + s] = (sh, wr - sh, np.array([0.0, -1.0, 0.0]), 0.055)
            self.ax["hand_" + s] = (wr, H["middle_01_" + s] - wr, np.array([0.0, -1.0, 0.0]), 0.04)
            hp, an = H["thigh_" + s], H["foot_" + s]
            self.ax["leg_" + s] = (hp, an - hp, np.array([0.0, -1.0, 0.0]), 0.08)
            self.ax["foot_" + s] = (an, T["ball_" + s] - an, np.array([0.0, 0.0, 1.0]), 0.05)

    def uv(self, P, f, region):
        O, A, ref, r = self.ax[region]
        return cyl_uv(P, f, O, A, ref, r)


# ================================================================================================= weights
def tights_weights(cb, keys):
    return [norm_w(head_only(w)) for w in cb.line_W(keys)]


class SurfaceWeights:
    """weights of the closest point on a finished piece's outer surface (barycentric-ish blend of the face's vertex
    weights): what a piece lying on another one (cowl, belt, baldric, skirt over trousers) should follow. Nearest
    VERTEX weights gave patchy fields that let the under-layer poke through in motion."""

    def __init__(self, m, faces=None):
        self.P = m.arr(); self.W = m.W
        self.F = [tuple(f) for f in (faces if faces is not None else m.outer_faces())]
        self.bvh = BVHTree.FromPolygons([v3(p) for p in self.P], self.F)

    def __call__(self, p, head=True):
        loc, n, fi, d = self.bvh.find_nearest(v3(p))
        if loc is None:
            return {"pelvis": 1.0}
        f = self.F[fi]
        Q = self.P[list(f)]
        dd = np.linalg.norm(Q - np.array(loc[:]), axis=1)
        w = 1.0 / np.maximum(dd, 1e-5) ** 2; w /= w.sum()
        out = {}
        for k, vi in enumerate(f):
            for b, x in (self.W[vi] or {}).items():
                out[b] = out.get(b, 0.0) + x * w[k]
        return norm_w(head_only(out) if head else out)


def smooth_weights(m, iters=2, lam=0.5, mask=None):
    nb = adjacency(len(m.V), m.F)
    for _ in range(iters):
        new = []
        for i, w in enumerate(m.W):
            if (mask is not None and not mask[i]) or not nb[i]:
                new.append(w); continue
            acc = {}
            for j in nb[i]:
                for b, x in m.W[j].items():
                    acc[b] = acc.get(b, 0.0) + x / len(nb[i])
            new.append(mix_w(w, acc, lam))
        m.W = new


# ================================================================================================= hiding the skin
def covered_verts(cb, P, F, dmax, erode=1, tilt=0.45, zmin=-1.0, zmax=9.0, tight=None, open_dist=0.025):
    """skin vertices a garment hides: the normal ray and 4 tilted rays all hit the garment within dmax; tight=d: skin
    closer than d to the garment is hidden too (except within open_dist of its open edges). Eroded by `erode` rings."""
    bvh = BVHTree.FromPolygons([v3(p) for p in P], [tuple(f) for f in F])
    lo = np.min(P, 0) - dmax; hi = np.max(P, 0) + dmax
    cand = np.nonzero(np.all((cb.co > lo) & (cb.co < hi), axis=1) & (cb.co[:, 2] > zmin) & (cb.co[:, 2] < zmax))[0]
    hit = np.zeros(NB, bool)
    if tight:
        loops = boundary_loops(F)
        bv = sorted(set(v for lp in loops for v in lp))
        kd = KDTree(max(1, len(bv)))
        for k, v in enumerate(bv):
            kd.insert(P[v], k)
        kd.balance()
        for i in cand:
            loc, nor, fi, dist = bvh.find_nearest(v3(cb.co[i]))
            if loc is not None and dist < tight and (not bv or kd.find(cb.co[i])[2] > open_dist):
                hit[i] = True
    for i in cand:
        if hit[i]:
            continue
        p = cb.co[i]; n = cb.vn[i]
        t1 = nrm(np.cross(n, (0, 0, 1)) if abs(n[2]) < 0.9 else np.cross(n, (1, 0, 0))); t2 = np.cross(n, t1)
        ok = True
        for d in (n, nrm(n + tilt * t1), nrm(n - tilt * t1), nrm(n + tilt * t2), nrm(n - tilt * t2)):
            loc, nor, fi, dist = bvh.ray_cast(v3(p + n * 0.0005), v3(d), dmax)
            if loc is None:
                ok = False
                break
        hit[i] = ok
    nb = adjacency(NB, cb.faces)
    for _ in range(erode):
        er = hit.copy()
        for i in np.nonzero(hit)[0]:
            if not all(hit[j] for j in nb[i]):
                er[i] = False
        hit = er
    return sorted(int(i) for i in np.nonzero(hit)[0])


# ================================================================================================= vertex AO
def vertex_ao(cb, P, F, rays=20, dist=0.07, strength=0.7, floor=0.4, extra=()):
    """per-vertex ambient occlusion from the piece itself + the body (+ extra Targets), cosine-weighted hemisphere.
    Written into the asset OBJ as vertex colours -> colour attribute 'Color' -> glTF COLOR_0 (x base colour)."""
    n = len(P)
    N = vertex_normals(P, F)
    polys = [tuple(f) for f in F]
    allv = [v3(p) for p in P] + [v3(p) for p in cb.co]
    allp = polys + [tuple(int(i) + n for i in t) for t in cb.tris]
    o = len(allv)
    for t in extra:
        allv += [v3(p) for p in t.P]; allp += [tuple(i + o for i in f) for f in t.F]; o = len(allv)
    bvh = BVHTree.FromPolygons(allv, allp)
    rng = np.random.default_rng(3)
    u1, u2 = rng.uniform(size=rays), rng.uniform(size=rays)
    r = np.sqrt(u1); ph = 2 * np.pi * u2
    dirs = np.stack([r * np.cos(ph), r * np.sin(ph), np.sqrt(1 - u1)], -1)
    ao = np.ones(n)
    for i in range(n):
        nn_ = N[i]
        t1 = nrm(np.cross(nn_, (0, 0, 1)) if abs(nn_[2]) < 0.9 else np.cross(nn_, (1, 0, 0))); t2 = np.cross(nn_, t1)
        org = v3(P[i] + nn_ * 0.0008)
        hit = 0
        for d in dirs:
            if bvh.ray_cast(org, v3(d[0] * t1 + d[1] * t2 + d[2] * nn_), dist)[0] is not None:
                hit += 1
        ao[i] = 1 - strength * hit / rays
    nb = adjacency(n, F)
    ao = 0.5 * ao + 0.5 * np.array([ao[x].mean() if x else ao[k] for k, x in enumerate(nb)])
    return np.clip(ao, floor, 1.0)


# ================================================================================================= texture sets
_CIVMAN = None


def civ_manifest():
    global _CIVMAN
    if _CIVMAN is None:
        p = os.path.join(CIV_TEX, "civ_textures.json")
        _CIVMAN = json.load(open(p))["sets"] if os.path.exists(p) else {}
    return _CIVMAN


def set_info(name):
    """texture set info: civ sets (assets/textures/civ) or the shared kit sets (textures_char.json, read only)"""
    m = civ_manifest()
    if name in m:
        d = dict(m[name]); d["dir"] = CIV_TEX
        return d
    shared = json.load(open(os.path.join(TEX, "textures_char.json")))["sets"]
    d = dict(shared[name]); d["dir"] = TEX
    return d


def uv_scale(slot_set):
    """UV units per metre of a tileable set (or 1 for atlas / strip sets, handled by their slot rules)"""
    i = set_info(slot_set)
    return float(i.get("uv_per_m", 1.0))


def _img(path, cs):
    im = bpy.data.images.load(path, check_existing=True)
    im.colorspace_settings.name = cs
    if cs == "Non-Color":
        im.alpha_mode = 'NONE'
    return im


def _gltf_group():
    name = "glTF Material Output"
    ng = bpy.data.node_groups.get(name)
    if ng:
        return ng
    try:
        from io_scene_gltf2.blender.com.material_helpers import create_settings_group
        return create_settings_group(name)
    except Exception:
        ng = bpy.data.node_groups.new(name, 'ShaderNodeTree')
        ng.interface.new_socket("Occlusion", socket_type="NodeSocketFloat")
        ng.nodes.new('NodeGroupInput'); ng.nodes.new('NodeGroupOutput')
        return ng


def civ_material(set_name, name=None, vcol="Color", double_sided=True, alpha_clip=None, sheen=0.0,
                 sheen_tint=(1, 1, 1)):
    """Principled BSDF wired 1:1 for glTF from a texture set: base (sRGB, x vertex AO colour -> COLOR_0), ORM
    (R occlusion via the glTF output group, G roughness, B metallic), OpenGL normal map."""
    name = name or "M_civ_" + set_name
    m = bpy.data.materials.get(name)
    if m is not None and m.get("rts_built"):
        return m
    if m is None:
        m = bpy.data.materials.new(name)
    m["rts_built"] = True
    m["rts_texture_set"] = set_name
    info = set_info(set_name)
    f = info["files"]; d = info["dir"]
    m.use_nodes = True
    nt = m.node_tree; N = nt.nodes; L = nt.links
    for n in list(N):
        N.remove(n)
    out = N.new("ShaderNodeOutputMaterial"); out.location = (600, 0)
    b = N.new("ShaderNodeBsdfPrincipled"); b.location = (250, 0)
    L.new(b.outputs[0], out.inputs[0])
    tb = N.new("ShaderNodeTexImage"); tb.image = _img(os.path.join(d, f["base"]), "sRGB"); tb.location = (-500, 250)
    if vcol:
        ca = N.new("ShaderNodeVertexColor"); ca.layer_name = vcol; ca.location = (-500, 450)
        mx = N.new("ShaderNodeMix"); mx.data_type = 'RGBA'; mx.blend_type = 'MULTIPLY'; mx.location = (-150, 300)
        mx.inputs["Factor"].default_value = 1.0
        L.new(tb.outputs["Color"], mx.inputs[6]); L.new(ca.outputs["Color"], mx.inputs[7])
        L.new(mx.outputs[2], b.inputs["Base Color"])
    else:
        L.new(tb.outputs["Color"], b.inputs["Base Color"])
    to = N.new("ShaderNodeTexImage"); to.image = _img(os.path.join(d, f["orm"]), "Non-Color"); to.location = (-500, -50)
    sp = N.new("ShaderNodeSeparateColor"); sp.location = (-200, -50)
    L.new(to.outputs["Color"], sp.inputs[0])
    L.new(sp.outputs[1], b.inputs["Roughness"]); L.new(sp.outputs[2], b.inputs["Metallic"])
    g = N.new("ShaderNodeGroup"); g.node_tree = _gltf_group(); g.location = (250, -450)
    if "Occlusion" in g.inputs:
        L.new(sp.outputs[0], g.inputs["Occlusion"])
    tn = N.new("ShaderNodeTexImage"); tn.image = _img(os.path.join(d, f["normal"]), "Non-Color"); tn.location = (-500, -350)
    nm = N.new("ShaderNodeNormalMap"); nm.location = (-200, -350)
    L.new(tn.outputs["Color"], nm.inputs["Color"]); L.new(nm.outputs[0], b.inputs["Normal"])
    if alpha_clip is not None:
        gt = N.new("ShaderNodeMath"); gt.operation = 'GREATER_THAN'; gt.inputs[1].default_value = alpha_clip
        L.new(tb.outputs["Alpha"], gt.inputs[0]); L.new(gt.outputs[0], b.inputs["Alpha"])
        m.surface_render_method = 'DITHERED'
    if sheen:
        b.inputs["Sheen Weight"].default_value = sheen
        b.inputs["Sheen Tint"].default_value = (*sheen_tint, 1)
        b.inputs["Sheen Roughness"].default_value = 0.5
    m.use_backface_culling = not double_sided
    return m


# ================================================================================================= MakeClothes
class Author:
    """authoring context on the live male (out/base_male.blend): body, match groups, cached basemesh cross-reference"""

    def __init__(self, kind="male"):
        self.kind = kind
        self.suffix = asset_suffix(kind)
        self.rig = bpy.data.objects["rts_" + kind]
        self.bm = bpy.data.objects[kind + "_body"]
        pose_reset(self.rig)
        self.cb = CivBody(self.rig, self.bm)
        self.axes = Axes(self.cb)
        self._xref = None
        self.match_groups()

    def match_groups(self):
        """body vertex groups for MakeClothes matching: static head skin (no face key moves it) for hoods / hats,
        a RIGID triangle on the head (hats move as one unit with the head) and one on the upper back (quiver)"""
        bm, cb = self.bm, self.cb
        kb = bm.data.shape_keys.key_blocks
        n = len(bm.data.vertices)
        base = np.empty(n * 3); kb["Basis"].data.foreach_get("co", base); base = base.reshape(-1, 3)
        mv = np.zeros(n)
        for k in FACE_KEYS:
            if k in kb:
                a = np.empty(n * 3); kb[k].data.foreach_get("co", a)
                mv = np.maximum(mv, np.linalg.norm(a.reshape(-1, 3) - base, axis=1))
        static = mv[:NB] < 2e-4
        ears = np.zeros(NB, bool); ears[cb.groups.get("ears", np.array([], int))] = True
        headw = cb.wsum(["head", "neck_02"]); neckw = cb.wsum(["neck_01", "neck_02", "spine_05"])
        z = cb.co[:, 2]; zc = cb.H["head"][2]
        groups = {"civ_match_head": np.nonzero(static & ~ears & (headw > 0.5) & (z > zc - 0.02))[0],
                  "civ_match_neck": np.nonzero(static & (neckw > 0.5) & (z > zc - 0.25))[0]}
        st = np.nonzero(static & ~ears)[0]
        top = cb.co[:, 2].max()

        def tri(targets, pool):
            return np.array([int(pool[np.argmin(np.linalg.norm(cb.co[pool] - np.array(t), axis=1))]) for t in targets])
        hy = cb.co[(z > top - 0.08)][:, 1].mean()
        groups["civ_rigid_head"] = tri([(0.070, hy - 0.06, top - 0.055), (-0.070, hy - 0.06, top - 0.055),
                                        (0.0, hy + 0.05, top - 0.05)], st)
        sp5 = cb.H["spine_05"]
        back = np.nonzero((cb.co[:, 1] > sp5[1]) & (np.abs(cb.co[:, 0]) < 0.2))[0]
        groups["civ_rigid_back"] = tri([(0.09, sp5[1] + 0.12, sp5[2] + 0.05), (-0.09, sp5[1] + 0.12, sp5[2] + 0.05),
                                        (0.0, sp5[1] + 0.12, sp5[2] - 0.12)], back)
        for nm, idx in groups.items():
            g = bm.vertex_groups.get(nm) or bm.vertex_groups.new(name=nm)
            g.add([int(i) for i in idx], 1.0, 'REPLACE')
        self.mgroups = groups
        clog("match groups", {k: len(v) for k, v in groups.items()})

    def match(self, clothes, props, delete_group=None):
        """ClothesService.create_mhclo_from_clothes_matching with the basemesh cross-reference built once"""
        from bl_ext.user_default.mpfb.services import ClothesService, MeshService
        from bl_ext.user_default.mpfb.entities.clothes.mhclo import Mhclo
        from bl_ext.user_default.mpfb.entities.clothes.vertexmatch import VertexMatch
        from bl_ext.user_default.mpfb.entities.meshcrossref import MeshCrossRef
        from bl_ext.user_default.mpfb.entities.objectproperties import GeneralObjectProperties
        if self._xref is None:
            self._xref = MeshCrossRef(self.bm, after_modifiers=True, build_faces_by_group_reference=True)
            self._refscale = ClothesService.get_reference_scale(self.bm)
        mh = Mhclo(); mh.verts = dict(); mh.clothes = clothes
        for k, v in props.items():
            if hasattr(mh, k):
                setattr(mh, k, v)
        cxr = MeshCrossRef(clothes, after_modifiers=True, build_faces_by_group_reference=True)
        sf = GeneralObjectProperties.get_value("scale_factor", entity_reference=self.bm)
        mh.max_pole = max(len(e) for e in cxr.edges_by_vertex)
        for vi in range(len(cxr.vertex_coordinates)):
            vm = VertexMatch(clothes, vi, cxr, self.bm, self._xref, scale_factor=sf, reference_scale=self._refscale,
                             allow_exact=False)
            mh.verts[vi] = vm.mhclo_line
        if delete_group and delete_group in self.bm.vertex_groups:
            mh.delete_group = delete_group
            mh.delverts = sorted(set(int(v[0]) for v in MeshService.find_vertices_in_vertex_group(self.bm, delete_group)))
            mh.delete = True
        return mh


def obj_add_colors(path, col):
    """append per-vertex colours to the OBJ's 'v' lines (Blender's OBJ importer reads 'v x y z r g b')"""
    out, k = [], 0
    for line in open(path):
        if line.startswith("v "):
            c = col[k]; k += 1
            line = line.rstrip("\n") + " %.4f %.4f %.4f\n" % (c[0], c[1], c[2])
        out.append(line)
    assert k == len(col), (k, len(col))
    open(path, "w").write("".join(out))


def face_key(f):
    return int(min(f)) * 100000 + int(max(f))


class Piece:
    """one garment / accessory: geometry + slots -> texture sets + hiding + metadata"""

    def __init__(self, outfit, slot, mesh, sets, layer=50, hide=None, hides_hair=False, desc="", notes="",
                 grime=None, extra=None):
        self.outfit, self.slot, self.m, self.sets = outfit, slot, mesh, sets
        self.layer, self.hide, self.hides_hair, self.desc, self.notes = layer, hide or [], hides_hair, desc, notes
        self.grime = grime            # optional fn(P) -> (n, 3) colour multiplier (painterly dirt at hems / feet)
        self.extra = extra or {}

    @property
    def asset(self):
        return "rts_%s_%s" % (self.outfit, self.slot)


def asset_suffix(kind):
    """cloth is authored on each base body (the drape is solved on that body): rts_<outfit>_<slot> on the male,
    rts_<outfit>_<slot>_<kind> on the others. Every asset is still an MPFB clothes asset (fits customised bodies)."""
    return "" if kind == "male" else "_" + kind


def asset_for(base, kind):
    """the asset to dress on `kind`: its own authored variant if installed, else the male one (MakeClothes auto-fit)"""
    a = base + asset_suffix(kind)
    return a if os.path.exists(asset_file("clothes", a, a + ".mhclo")) else base


STRIPS = {}          # civ_trim strips: name -> (v0, v1, metres per u unit), filled from the manifest


def final_uvs(piece):
    """metric UVs -> texture UVs: tileable slots x uv_per_m; strip slots 'trim:<strip>' mapped into the civ_trim strip"""
    m = piece.m
    out = []
    man = civ_manifest()
    trim = man.get("civ_trim", {})
    strips = trim.get("strips", {})
    for uv, s in zip(m.UV, m.S):
        base = s.split(":")[0]
        if s.startswith("trim:"):
            v0, v1 = strips[s.split(":")[1]]["v"]
            mu = strips[s.split(":")[1]].get("m_per_u", 0.25)
            out.append([(u / mu, v0 + (v1 - v0) * min(max(v, 0.0), 1.0)) for u, v in uv])
        elif s.startswith("atlas:"):
            out.append(uv)                                  # already normalised 0..1 into the set's atlas
        else:
            k = uv_scale(piece.sets[base])
            out.append([(u * k, v * k) for u, v in uv])
    return out


def write_piece(A, piece, delete=None, zdepth=None, ao=True, extra_targets=()):
    """CMesh -> MPFB clothes asset rts_<outfit>_<slot>: MakeClothes matching (per-vertex match groups), delete group,
    custom weights (.mhw, every bone listed), per-face slots + metadata (.rts.json), per-vertex AO (+ grime) colours"""
    m = piece.m
    if m.mixed():
        m.triangulate()
    asset = piece.asset + A.suffix
    d = os.path.join(ASSET_DIR, asset)
    if os.path.exists(d):
        shutil.rmtree(d)
    os.makedirs(d)
    P = m.arr()
    mm = CMesh(); mm.V, mm.F, mm.UV, mm.S, mm.W, mm.G, mm.HF = m.V, m.F, final_uvs(piece), m.S, m.W, m.G, m.HF
    ob = mm.to_object(asset, slots=["cloth"] * 0 or sorted(set(m.S)))
    ob.data.polygons.foreach_set("material_index", np.zeros(len(ob.data.polygons), dtype=np.int32))
    for g in sorted(set(m.G)):
        vg = ob.vertex_groups.new(name=g)
        vg.add([i for i, x in enumerate(m.G) if x == g], 1.0, 'REPLACE')
        assert g in A.bm.vertex_groups, g
    assert len(set(len(f) for f in m.F)) == 1, (asset, "mixed face sizes")
    used = set(i for f in m.F for i in f)
    assert len(used) == len(m.V), (asset, "loose vertices", len(m.V) - len(used))
    dg = None
    delete = delete or []
    if delete:
        dg = "Delete." + piece.asset           # the same group name for every body's variant (kit coverage by name)
        g = A.bm.vertex_groups.get(dg) or A.bm.vertex_groups.new(name=dg)
        g.add([int(i) for i in delete], 1.0, 'REPLACE')
    props = {"name": asset, "description": "RTS %s outfit: %s" % (piece.outfit, piece.desc or piece.slot),
             "author": "RTS project (procedural, scripts/outfit_%s.py)" % piece.outfit, "license": "CC0", "homepage": "",
             "uuid": str(uuid.uuid5(uuid.NAMESPACE_URL, "rts-civ-" + asset))}
    t0 = time.time()
    mh = A.match(ob, props, delete_group=dg)
    mh.material = asset + ".mhmat"
    mh.zdepth = zdepth or piece.layer
    path = os.path.join(d, asset + ".mhclo")
    mh.write_mhclo(path, reference_scale=A._refscale, also_export_mhmat=False)
    col = np.ones((len(P), 3))
    if ao:
        a = vertex_ao(A.cb, P, m.F, extra=extra_targets)
        col *= a[:, None]
    if piece.grime is not None:
        col *= piece.grime(P)
    obj_add_colors(path.replace(".mhclo", ".obj"), col)
    # weights: every rts_human bone listed so MPFB replaces its interpolated groups
    bones = [b.name for b in A.rig.data.bones]
    W = {b: [] for b in bones}
    for i, w in enumerate(m.W):
        w = norm_w(w)
        assert w, (asset, "vertex without weights", i)
        for b, x in w.items():
            assert b in W, (asset, b)
            W[b].append([i, round(float(x), 5)])
    json.dump({"name": asset + " weights", "license": "CC0", "version": 110, "weights": W},
              open(os.path.join(d, asset + ".mhw"), "w"))
    open(os.path.join(d, asset + ".mhmat"), "w").write(
        "# MakeHuman material (preview only; the RTS build assigns PBR slots from %s.rts.json)\nname %s\n"
        "diffuseColor 0.55 0.5 0.42\nshininess 0.2\nbackfaceCull False\n" % (asset, asset))
    slots = sorted(set(s.split(":")[0] for s in m.S))
    meta = {"asset": asset, "outfit": piece.outfit, "slot": piece.slot, "desc": piece.desc, "notes": piece.notes,
            "layer": piece.layer, "hides_hair": piece.hides_hair, "hide_regions": piece.hide,
            "slots": slots, "sets": {s: piece.sets[s] for s in slots},
            "face_slot": [s.split(":")[0] for s in m.S], "face_keys": [face_key(f) for f in m.F],
            "verts": len(m.V), "faces": len(m.F), "tris": m.tris(), "delete_verts": len(delete),
            "delete_basemesh_verts": [int(i) for i in delete], "authored_on": A.kind, "piece_id": piece.asset}
    meta.update(piece.extra)
    json.dump(meta, open(os.path.join(d, asset + ".rts.json"), "w"))
    dst = os.path.join(USER_DATA, "clothes", asset)
    if os.path.exists(dst):
        shutil.rmtree(dst)
    shutil.copytree(d, dst)
    bpy.data.objects.remove(ob, do_unlink=True)
    clog("asset %-26s verts %5d tris %5d hides %5d  (%.1fs)" % (asset, len(m.V), m.tris(), len(delete), time.time() - t0))
    return path


# ================================================================================================= sockets
def _canon_hand(side, pos, frames):
    """hand socket: grip centre in front of the palm, +Y = pinky -> index (along the grip), Z = wrist -> knuckles"""
    kn = np.mean([pos[f + "_01_" + side] for f in ("index", "middle", "ring", "pinky")], 0)
    wr = pos["hand_" + side]
    c = wr * 0.35 + kn * 0.65
    across = nrm(pos["index_01_" + side] - pos["pinky_01_" + side])
    axis = nrm(kn - wr)
    th = pos["thumb_02_" + side]
    palm = th - c; palm = palm - axis * palm.dot(axis) - across * palm.dot(across); palm = nrm(palm)
    return c, palm, across, axis


def civ_socket_specs(rig):
    """Sockets with BODY-INDEPENDENT rest rotations (skeleton convention M15): every frame is computed from the rig
    JSON default joint positions (the neutral basemesh, as chr_lib.canonical_rest does for the core bones); only
    the socket positions come from the fitted body. Names and axis meanings match the knight's (armour_lower
    socket_specs): socket_weapon_r / socket_hand_l = grip centre, +Y along the grip (pinky -> index), Z = wrist ->
    knuckles; socket_back = centre of the upper back, +Y down... (see STATUS). Returns ({name: spec}, {name: 3x3})."""
    J = json.load(open(RIG_JSON))["bones"]
    dpos = {n: np.array(b["head"]["default_position"]) for n, b in J.items()}
    B = rig.data.bones
    fpos = {b.name: np.array(b.head_local[:]) for b in B}
    specs, frames = {}, {}
    for side, name in (("r", "socket_weapon_r"), ("l", "socket_hand_l")):
        c0, palm0, across0, axis0 = _canon_hand(side, dpos, None)
        c, palm, across, axis = _canon_hand(side, fpos, None)
        head = c + palm * 0.030
        y = across0; z = nrm(axis0 - y * axis0.dot(y))
        specs[name] = dict(parent="hand_" + side, head=head, y=y, z=z)
    sp5 = fpos["spine_05"]
    head = np.array([0.0, sp5[1] + 0.20, sp5[2] + 0.02])
    specs["socket_back"] = dict(parent="spine_05", head=head, y=np.array([0.0, 0.0, -1.0]), z=np.array([0.0, 1.0, 0.0]))
    # right hip (tools, knife): outside the belt line
    th = fpos["thigh_r"]
    specs["socket_hip_r"] = dict(parent="pelvis", head=np.array([th[0] - 0.07, th[1] + 0.0, th[2] + 0.06]),
                                 y=np.array([0.0, 0.0, -1.0]), z=np.array([-1.0, 0.0, 0.0]))
    for n, s in specs.items():
        y = nrm(s["y"]); z = nrm(s["z"] - y * np.dot(s["z"], y)); x = np.cross(y, z)
        frames[n] = Matrix([list(x), list(y), list(z)]).transposed()
    return specs, frames


def ensure_sockets(rig, extra=None):
    """create / move the civ sockets (non-deform) with canonical frames; extra: more {name: spec} (parent, head, y, z)"""
    specs, frames = civ_socket_specs(rig)
    if extra:
        for n, s in extra.items():
            specs[n] = s
            y = nrm(s["y"]); z = nrm(s["z"] - y * np.dot(s["z"], y)); x = np.cross(y, z)
            frames[n] = Matrix([list(x), list(y), list(z)]).transposed()
    for o in bpy.context.view_layer.objects:
        o.select_set(False)
    bpy.context.view_layer.objects.active = rig; rig.select_set(True)
    bpy.ops.object.mode_set(mode='EDIT')
    eb = rig.data.edit_bones
    for n, s in specs.items():
        b = eb.get(n) or eb.new(n)
        b.head = v3(s["head"]); b.tail = v3(np.array(s["head"]) + nrm(s["y"]) * s.get("length", 0.1))
        b.parent = eb[s["parent"]]; b.use_connect = False; b.use_deform = s.get("deform", False)
    bpy.ops.object.mode_set(mode='OBJECT')
    canonical_rest(rig, frames)                  # exact socket / prop-bone rest rotations (only these bones)
    rig["rts_sockets"] = json.dumps({n: dict(parent=s["parent"]) for n, s in specs.items()
                                     if n.startswith("socket_") or n == "bow_socket_arrow"})
    return specs


# ================================================================================================= dressing
def piece_meta(asset):
    return json.load(open(asset_file("clothes", asset, asset + ".rts.json")))


def apply_slots(ob, meta, mats):
    """per-face material slots from the .rts.json (keyed by face vertices: safe against OBJ face order)"""
    names = meta["slots"]
    me = ob.data
    me.materials.clear()
    for s in names:
        me.materials.append(mats[s])
    key2slot = {k: s for k, s in zip(meta["face_keys"], meta["face_slot"])}
    idx = []
    for p in me.polygons:
        s = key2slot.get(face_key(p.vertices))
        idx.append(names.index(s) if s in names else 0)
    me.polygons.foreach_set("material_index", np.array(idx, dtype=np.int32))


SHEEN = {"civ_wool": (0.25, (0.9, 0.85, 0.8)), "civ_linen": (0.15, (1, 1, 1)), "civ_quilt": (0.15, (1, 1, 1))}


def material_for(set_name):
    alpha = None
    info = set_info(set_name)
    if info.get("alpha"):
        alpha = info.get("alpha_cut", 0.4)
    sh = next((v for k, v in SHEEN.items() if set_name.startswith(k)), (0.0, (1, 1, 1)))
    return civ_material(set_name, alpha_clip=alpha, sheen=sh[0], sheen_tint=sh[1])


def dress_piece(rig, bm, asset):
    """load one civ asset onto the live human: fitted, skinned (custom weights), hides skin, cust_* morphs only.
    A base asset name (rts_<outfit>_<slot>) resolves to this body's own authored variant when it is installed
    (rts_<outfit>_<slot>_female on the female), so callers (kit_build.add_civ) can pass the outfit table's names."""
    import outfit_lib
    kind = rig.get("rts_kind") or rig.name.replace("rts_", "")
    if not asset.endswith("_" + kind):
        asset = asset_for(asset, kind)
    meta = piece_meta(asset)
    path = asset_file("clothes", asset, asset + ".mhclo")
    keys = [k.name for k in bm.data.shape_keys.key_blocks if k.name.startswith("cust_")]
    ob = outfit_lib.add_piece(rig, bm, path, slot=meta["outfit"] + "_" + meta["slot"], keys=keys,
                              material=material_for(meta["sets"][meta["slots"][0]]))
    mats = {s: material_for(meta["sets"][s]) for s in meta["slots"]}
    apply_slots(ob, meta, mats)
    if "Color" not in ob.data.color_attributes:
        ca = ob.data.color_attributes.new("Color", 'BYTE_COLOR', 'POINT')
        ca.data.foreach_set("color", np.ones(len(ob.data.vertices) * 4))
    ob["rts_set"] = meta["outfit"]
    ob["rts_outfit"] = meta["outfit"]
    ob["rts_layer"] = meta["layer"]
    ob["rts_hides"] = ",".join(meta["hide_regions"])
    ob["rts_hides_hair"] = bool(meta["hides_hair"])
    ob["slot"] = meta.get("group", "Outfit")
    return ob, meta
