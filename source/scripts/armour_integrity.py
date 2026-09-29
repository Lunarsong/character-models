#!/usr/bin/env python3
"""ARMOUR INTEGRITY gates (user feedback round 5, items 23-32): what the user sees when orbiting the knight in the web
viewer at 0.3-1 m, from any angle, in every animation frame, on both bodies. Runs on the ENGINE file
out/knight_<kind>_export.blend (exactly the meshes / weights / clips / culled faces that are in the GLB) and, for G8,
on the web viewer page itself (three.js = what the user looks at).

    python3 scripts/armour_integrity.py [--kinds male female] [g1 g2 g3 g4 g5 g6 g7 g8 viewer sheets summary defects] [step=3]
    python3 scripts/build_knight.py integrity                   # the same, as a build stage
    Blender -b out/knight_<kind>_export.blend --python-exit-code 1 -P scripts/armour_integrity.py -- <kind> [g1 ..] [step=3] [quick]

Gates (renders/integrity/<gate>_<kind>.json, sheets renders/integrity/*.png, defects renders/integrity/defects.json):
  G1 one-sided / open shells: welded boundary edges and inner-face thickness per component; every piece alone from 26
     directions with back-face culling on vs off (a surface the viewer does not draw = 'vanish' pixel) = error
  G2 floating parts: trims / rings / rivets / spikes further than 0.5 mm from the surface they sit on (bind and every
     pose), partly buried pieces (only a sliver shows), sliver components, shard-like triangles
  G3 joins / gaps: named seams; windows along each seam ray-traced from outside in every pose: a ray that passes between
     the two pieces and reaches the background or the inside of the armour = see-through gap (mail under it = ok)
  G4 visible penetration: intersecting triangles per pose and pair (pieces and components of one piece) and whether the
     intersection line can be seen from any direction (a ray escapes the knight): visible = error, hidden = warning
  G5 grip: sword grip vs the closed right fist (finger contact, nothing through the grip, finger plates on the fingers),
     shield on its enarmes / hand strap, every clip frame
  G6 texture stretch: per-face texel stretch (principal stretches of the UV -> surface map vs the set's nominal texel
     density / the piece median) at bind and the max over poses; mail within +-15 %
  G7 head in helmet: rays in through the eye slit / breaths must meet the face / eyes or a dark lining, never background
  G8 user battery: the user_r5_*.png views reproduced in the viewer (camera list in the JSON) + an orbit battery at
     0.4-1 m round the helm, shoulders, elbows, hands, waist and knees from 26 directions (incl. straight down and from
     behind) at key frames of every clip: viewer contact sheets + the same cameras ray-traced for holes (surfaces the
     viewer does not draw)
Visibility rule everywhere = the viewer's: a triangle of a single-sided material (glTF doubleSided false, Blender
use_backface_culling) seen from behind is not drawn; rays pass through it.
"""
import sys, os, json, math, time

HERE = os.path.dirname(os.path.abspath(__file__))
CH = os.path.dirname(HERE)
RDIR = os.path.join(CH, "renders", "integrity")
BL = "/Applications/Blender.app/Contents/MacOS/Blender"

try:
    import bpy  # noqa: F401
    IN_BLENDER = True
except ImportError:
    IN_BLENDER = False

# ------------------------------------------------------------------------------------------------ ownership
UPPER = ("helmet", "plume", "gorget", "gorget_top", "cuirass", "mail", "pauldron_l", "pauldron_r", "rerebrace_l",
         "rerebrace_r", "couter_l", "couter_r", "vambrace_l", "vambrace_r", "gauntlet_l", "gauntlet_r", "aventail")
LOWER = ("tabard", "cape", "clasps", "belts", "tassets", "mail_skirt", "cuisses", "poleyns", "greaves", "sabatons",
         "boots", "legs_mail", "sword", "shield", "scabbard")
ASSEMBLY = ("underlayer",)       # build_knight.make_fillers (rig-kit agent, same owner as knight_anim)


def owner_of(part):
    if part in UPPER:
        return "UPPER"
    if part in LOWER:
        return "LOWER"
    return "ANIM"


# under-layers: seeing them through a gap between plates is correct layering (mail under plate), not a see-through gap
UNDER = ("mail", "underlayer", "aventail", "legs_mail", "mail_skirt", "boots")

# named seams (G3): name -> (A, B); 'piece' or 'piece:selector' (see comp_select); _s = both sides
SEAMS = [
    ("helmet skull-visor", "helmet:skull", "helmet:visor"),
    ("visor-bevor (visor chin vs gorget)", "helmet:visor", "gorget"),
    ("helmet-gorget", "helmet:skull", "gorget"),
    ("gorget-breastplate", "gorget", "cuirass"),
    ("pauldron-breastplate_s", "pauldron_s", "cuirass"),
    ("pauldron-gorget_s", "pauldron_s", "gorget"),
    ("pauldron-rerebrace_s", "pauldron_s", "rerebrace_s"),
    ("rerebrace-couter_s", "rerebrace_s", "couter_s"),
    ("couter-vambrace_s", "couter_s", "vambrace_s"),
    ("cuff-vambrace_s", "gauntlet_s:cuff", "vambrace_s"),
    ("gauntlet-cuff (wrist)_s", "gauntlet_s:hand", "gauntlet_s:cuff"),
    ("mail edge: sleeve-vambrace_s", "mail", "vambrace_s"),
    ("mail edge: mail-gorget", "mail", "gorget"),
    ("mail edge: mail-breastplate", "mail", "cuirass"),
    ("faulds-tassets (cuirass-tassets)", "cuirass", "tassets"),
    ("mail skirt-tassets", "mail_skirt", "tassets"),
    ("tasset-cuisse", "tassets", "cuisses"),
    ("cuisse-poleyn", "cuisses", "poleyns"),
    ("poleyn-greave", "poleyns", "greaves"),
    ("greave-sabaton", "greaves", "sabatons"),
    ("mail edge: chausses-boots", "legs_mail", "boots"),
    ("tabard-gorget", "tabard", "gorget"),
    ("tabard-breastplate", "tabard", "cuirass"),
    ("belts-tabard", "belts", "tabard"),
    ("cape-gorget", "cape", "gorget"),
    ("cape-pauldron_s", "cape", "pauldron_s"),
    ("clasp-cape", "clasps", "cape"),
]


def expand_seams():
    out = []
    for nm, a, b in SEAMS:
        if "_s" in nm:
            for s in ("l", "r"):
                out.append((nm.replace("_s", "_" + s), a.replace("_s", "_" + s), b.replace("_s", "_" + s)))
        else:
            out.append((nm, a, b))
    return out


def dirs26():
    out = []
    for x in (-1, 0, 1):
        for y in (-1, 0, 1):
            for z in (-1, 0, 1):
                if x or y or z:
                    n = math.sqrt(x * x + y * y + z * z)
                    out.append((x / n, y / n, z / n))
    return out


def dir_name(d):
    """Blender frame: the knight faces -Y. 'front' = camera in front of the knight"""
    x, y, z = [(1 if v > 0.3 else -1 if v < -0.3 else 0) for v in d]
    p = []
    p.append({1: "top", -1: "below"}.get(z, ""))
    p.append({-1: "front", 1: "back"}.get(y, ""))
    p.append({1: "left", -1: "right"}.get(x, ""))       # +X = the knight's left
    return "-".join(q for q in p if q) or "?"


# the user's round-5 screenshots (refs/feedback/user_r5_<name>.png) reproduced in the web viewer (town scene, idle clip at
# 3.0 s): target = bone head (glTF, idle 3.0 s) + offset; camera at azimuth (0 = in front of the knight, 90 = on his
# left), elevation and distance from the target, fov 30 (the viewer's). Matched by eye on candidate sheets (the rebuilt
# knight differs in detail from the 07:02 build the user looked at).
USER_VIEWS = {
    "sword_hand_top": ("hand_r", (0.096, 0.202, -0.001), 320, 65, 0.95),
    "shield_elbow": ("upperarm_l", (0.119, -0.218, -0.037), 90, 15, 1.05),
    "elbow_rings": ("hand_r", (0.006, 0.122, 0.009), 210, 62, 0.42),
    "pauldron_gorget": ("upperarm_l", (0.029, 0.062, -0.037), 120, 45, 0.60),
    "pauldron_helm": ("upperarm_r", (0.061, 0.112, 0.003), 325, 18, 0.75),
    "mail_top": ("upperarm_l", (-0.111, 0.082, -0.037), 240, 70, 0.40),
}
USER_CLIP = ("idle", 90)


def fib_dirs(n):
    out = []
    for i in range(n):
        z = 1 - 2 * (i + 0.5) / n; r = math.sqrt(max(0.0, 1 - z * z)); a = i * 2.399963229728653
        out.append((r * math.cos(a), r * math.sin(a), z))
    return out


# =================================================================================================== Blender side
if IN_BLENDER:
    sys.path.insert(0, HERE)
    import numpy as np
    from mathutils import Vector, Matrix
    from mathutils.bvhtree import BVHTree
    import knight_qa as QA

    def log(*a):
        print("INT", *a, flush=True)

    def jdump(obj, name):
        os.makedirs(RDIR, exist_ok=True)
        p = os.path.join(RDIR, name)
        tmp = p + ".tmp"
        json.dump(obj, open(tmp, "w"), indent=1, default=lambda x: x.tolist() if hasattr(x, "tolist") else str(x))
        os.replace(tmp, p)
        log("wrote", os.path.relpath(p, CH))

    # ------------------------------------------------------------------------------------------- static topology
    class Piece:
        """one visible mesh: triangles, materials (single / double sided), welded components, dominant bones, UVs"""

        def __init__(self, o, kind, dg):
            self.o = o
            self.name = o.name
            self.part = o.get("rts_part") or o.name[len(kind) + 1:]
            self.owner = owner_of(self.part)
            ev = o.evaluated_get(dg); me = ev.to_mesh()
            me.calc_loop_triangles()
            nv = len(me.vertices); nt = len(me.loop_triangles)
            self.nv, self.nt = nv, nt
            tv = np.empty(nt * 3, np.int32); me.loop_triangles.foreach_get("vertices", tv)
            tl = np.empty(nt * 3, np.int32); me.loop_triangles.foreach_get("loops", tl)
            tp = np.empty(nt, np.int32); me.loop_triangles.foreach_get("polygon_index", tp)
            pm = np.empty(len(me.polygons), np.int32); me.polygons.foreach_get("material_index", pm)
            self.tris = tv.reshape(-1, 3)
            self.tri_poly = tp
            self.tri_mat = pm[tp] if len(pm) else np.zeros(nt, np.int32)
            mats = list(o.data.materials)
            self.mat_names = [m.name if m else "" for m in mats] or [""]
            self.mat_single = [bool(m.use_backface_culling) if m else False for m in mats] or [False]
            self.tri_single = np.array([self.mat_single[min(i, len(self.mat_single) - 1)] for i in self.tri_mat], bool)
            uvl = me.uv_layers.active
            if uvl is not None and len(me.loops):
                uv = np.empty(len(me.loops) * 2, np.float32); uvl.data.foreach_get("uv", uv)
                self.tri_uv = uv.reshape(-1, 2)[tl.reshape(-1, 3)]
            else:
                self.tri_uv = None
            co = np.empty(nv * 3, np.float64); me.vertices.foreach_get("co", co)
            M = np.array(o.matrix_world)
            self.rest = co.reshape(-1, 3) @ M[:3, :3].T + M[:3, 3]
            ev.to_mesh_clear()
            # dominant bone per vertex (static)
            names = {g.index: g.name for g in o.vertex_groups}
            dom = []
            for v in o.data.vertices:
                gs = [(g.weight, names.get(g.group, "?")) for g in v.groups]
                dom.append(max(gs)[1] if gs else "?")
            self.dom = np.array(dom, dtype=object)
            self._weld_and_components()

        def _weld_and_components(self):
            P = self.rest
            nv = len(P)
            par = np.arange(nv)

            def find(i):
                r = i
                while par[r] != r:
                    r = par[r]
                while par[i] != r:
                    par[i], i = r, par[i]
                return r

            def union(a, b):
                ra, rb = find(a), find(b)
                if ra != rb:
                    par[max(ra, rb)] = min(ra, rb)
            # weld coincident vertices (2 shifted grids of 0.03 mm cells: rims / shells written as separate strips)
            cell = 3e-5
            wpar = np.arange(nv)
            for off in (0.0, 0.5):
                key = np.floor(P / cell + off).astype(np.int64)
                _, inv = np.unique(key, axis=0, return_inverse=True)
                inv = inv.reshape(-1)
                first = {}
                for i, k in enumerate(inv):
                    j = first.setdefault(int(k), i)
                    if j != i:
                        union(i, j)
            weld = np.array([find(i) for i in range(nv)])
            self.weld = weld
            for t in self.tris:
                union(t[0], t[1]); union(t[1], t[2])
            comp_root = np.array([find(i) for i in range(nv)])
            roots, cid = np.unique(comp_root, return_inverse=True)
            self.vcomp = cid.reshape(-1)
            self.tri_comp = self.vcomp[self.tris[:, 0]]
            self.ncomp = len(roots)
            # welded edges -> boundary / non-manifold counts per component
            wt = weld[self.tris]
            e = np.concatenate([wt[:, [0, 1]], wt[:, [1, 2]], wt[:, [2, 0]]])
            e.sort(axis=1)
            ecomp = np.concatenate([self.tri_comp] * 3)
            keep = e[:, 0] != e[:, 1]                         # degenerate welded edges
            e, ecomp = e[keep], ecomp[keep]
            uniq, inv, cnt = np.unique(e, axis=0, return_inverse=True, return_counts=True)
            inv = inv.reshape(-1)
            ec = np.zeros(len(uniq), np.int64); ec[inv] = ecomp
            self.bnd_edges = uniq[cnt == 1]                  # welded vertex ids
            self.bnd_comp = ec[cnt == 1]
            self.nm_edges = uniq[cnt > 2]
            self.nm_comp = ec[cnt > 2]
            A = self.tri_areas(self.rest)
            self.tri_area_rest = A
            self.comp_area = np.bincount(self.tri_comp, A, minlength=self.ncomp)
            self.comp_tris = np.bincount(self.tri_comp, minlength=self.ncomp)
            self.comp_bnd = np.bincount(self.bnd_comp, minlength=self.ncomp)
            self.comp_nm = np.bincount(self.nm_comp, minlength=self.ncomp)
            self.comp_verts = [np.where(self.vcomp == c)[0] for c in range(self.ncomp)]
            self.comp_lo = np.array([P[v].min(0) for v in self.comp_verts])
            self.comp_hi = np.array([P[v].max(0) for v in self.comp_verts])
            self.comp_cen = np.array([P[v].mean(0) for v in self.comp_verts])
            self.comp_diag = np.linalg.norm(self.comp_hi - self.comp_lo, axis=1)
            self.comp_dom = []
            for v in self.comp_verts:
                vals, cn = np.unique(self.dom[v].astype(str), return_counts=True)
                self.comp_dom.append(str(vals[np.argmax(cn)]))
            self.comp_single = np.array([self.tri_single[self.tri_comp == c].mean() > 0.5 for c in range(self.ncomp)])
            self.comp_mat = []
            for c in range(self.ncomp):
                mi = np.bincount(self.tri_mat[self.tri_comp == c], minlength=len(self.mat_names))
                self.comp_mat.append(self.mat_names[min(int(np.argmax(mi)), len(self.mat_names) - 1)])

        def tri_areas(self, P):
            a, b, c = P[self.tris[:, 0]], P[self.tris[:, 1]], P[self.tris[:, 2]]
            return 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)

        def tri_normals(self, P):
            a, b, c = P[self.tris[:, 0]], P[self.tris[:, 1]], P[self.tris[:, 2]]
            n = np.cross(b - a, c - a)
            return n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-20)

        def comp_label(self, c):
            return "%s#%d" % (self.part, c)

    # ------------------------------------------------------------------------------------------- scene per pose
    class Scene:
        """all visible meshes of one look; per pose: world vertex positions, one BVH of everything (ray casting with
        the viewer's back-face rule) and one BVH per piece (overlaps)"""

        def __init__(self, rig, kind, look="helm"):
            self.rig, self.kind, self.look = rig, kind, look
            QA.face_keys_static(rig)
            QA.set_look(rig, look)
            QA.play(rig, None, 0)
            dg = bpy.context.evaluated_depsgraph_get()
            objs = [o for o in QA.meshes(rig) if QA.in_look(o, look) and not o.name.endswith("_skin_probe")]
            self.pieces = [Piece(o, kind, dg) for o in sorted(objs, key=lambda o: o.name)]
            self.by_part = {p.part: p for p in self.pieces}
            off = 0; toff = 0; coff = 0
            self.voff, self.toff, self.coff = [], [], []
            for p in self.pieces:
                self.voff.append(off); self.toff.append(toff); self.coff.append(coff)
                off += p.nv; toff += p.nt; coff += p.ncomp
            self.NV, self.NT, self.NC = off, toff, coff
            self.T = np.concatenate([p.tris + o_ for p, o_ in zip(self.pieces, self.voff)])
            self.tri_piece = np.concatenate([np.full(p.nt, i) for i, p in enumerate(self.pieces)])
            self.tri_single = np.concatenate([p.tri_single for p in self.pieces])
            self.tri_gcomp = np.concatenate([p.tri_comp + c for p, c in zip(self.pieces, self.coff)])
            self.single_l = self.tri_single.tolist()
            self.piece_l = self.tri_piece.tolist()
            self.Tl = self.T.tolist()
            self.pose = None
            self._fib14 = [Vector(d) for d in fib_dirs(14)]
            self.set_pose(("bind", None, 0))

        # ---- posing
        def pose_list(self, step=3, extreme=True, clips=True):
            out = [("bind", None, 0)]
            if extreme:
                out += [(n, "extreme", n) for n in QA.EXTREME]
            if clips:
                table = json.loads(self.rig.get("rts_clips", "{}"))
                for c in [c for c in table if c != "talk_emote" and bpy.data.actions.get(c)]:
                    nf = int(round(bpy.data.actions[c].frame_range[1]))
                    out += [("%s_f%03d" % (c, f), c, f) for f in range(0, nf + 1, step)]
            return out

        def set_pose(self, pose):
            name, clip, f = pose
            rig = self.rig
            if clip == "extreme":
                QA.play(rig, None, 0)
                QA.EXTREME[f](rig)
                import rig_helpers
                rig_helpers.drive_helpers(rig)
            else:
                QA.play(rig, clip, f)
            dg = bpy.context.evaluated_depsgraph_get()
            Ps = []
            for p in self.pieces:
                ev = p.o.evaluated_get(dg); me = ev.to_mesh()
                co = np.empty(p.nv * 3, np.float64); me.vertices.foreach_get("co", co)
                ev.to_mesh_clear()
                M = np.array(p.o.matrix_world)
                Ps.append(co.reshape(-1, 3) @ M[:3, :3].T + M[:3, 3])
            self.P = Ps
            self.V = np.concatenate(Ps)
            self.bvh = BVHTree.FromPolygons(self.V.tolist(), self.Tl, epsilon=0.0)
            self._pbvh = {}
            self.pose = name
            self.pose_group = "bind" if clip is None else ("extreme" if clip == "extreme" else clip)
            self.bones = {b.name: (rig.matrix_world @ b.head, rig.matrix_world @ b.tail, rig.matrix_world @ b.matrix)
                          for b in rig.pose.bones}

        def piece_bvh(self, i):
            if i not in self._pbvh:
                p = self.pieces[i]
                self._pbvh[i] = BVHTree.FromPolygons(self.P[i].tolist(), p.tris.tolist(), epsilon=0.0)
            return self._pbvh[i]

        # ---- rays (viewer rule: a single-sided triangle hit from behind is not drawn -> pass through)
        def cast_vis(self, o, r, maxd=5.0, bvh=None, single=None, count=False):
            """first DRAWN surface along the ray: (dist, tri) or (None, -1); count=True also returns how many
            undrawn back faces it passed (> 0: the ray went through the inside of a shell)"""
            bvh = bvh or self.bvh
            single = single if single is not None else self.single_l
            tot = 0.0; ns = 0
            for _ in range(16):
                h = bvh.ray_cast(o, r, maxd - tot)
                if h[0] is None:
                    return (None, -1, ns) if count else (None, -1)
                loc, n, i, d = h
                tot += d
                if single[i] and n.dot(r) > 0:
                    o = loc + r * 2e-5; ns += 1
                    continue
                return (tot, i, ns) if count else (tot, i)
            return (None, -1, ns) if count else (None, -1)

        def radial(self, x):
            """outward direction at x: away from the nearest body bone segment"""
            if not hasattr(self, "_segs") or self._segs_pose != self.pose:
                H, T = [], []
                for n, (h, t, _) in self.bones.items():
                    if n.startswith(("socket_", "cape_", "tabard_", "plume_", "eye", "brow", "cheek", "nose", "mouth",
                                     "lip", "tongue", "jaw")) or "helper" in n or "twist" in n or n == "root":
                        continue
                    H.append(h[:]); T.append(t[:])
                self._segs = (np.array(H), np.array(T)); self._segs_pose = self.pose
            H, T = self._segs
            x = np.asarray(x, float)
            d = T - H
            u = np.clip(((x - H) * d).sum(1) / np.maximum((d * d).sum(1), 1e-12), 0, 1)
            Q = H + d * u[:, None]
            k = int(np.argmin(np.linalg.norm(x - Q, axis=1)))
            v = x - Q[k]
            return v / max(np.linalg.norm(v), 1e-9)

        def inside(self, o, nd=14, lim=12):
            """camera position enclosed by the knight (hits in >= lim of nd directions within 1.2 m)"""
            n = 0
            for d in self._fib14:
                if self.bvh.ray_cast(o, d, 1.2)[0] is not None:
                    n += 1
            return n >= lim

        def cast_first(self, o, r, maxd=5.0, bvh=None, single=None):
            """first surface of any kind: (dist, tri, drawn?)"""
            bvh = bvh or self.bvh
            single = single if single is not None else self.single_l
            h = bvh.ray_cast(o, r, maxd)
            if h[0] is None:
                return None, -1, False
            return h[3], h[2], not (single[h[2]] and h[1].dot(r) > 0)

        def escapes(self, x, d, maxd=5.0, eps=4e-4):
            """does a camera far along +d see point x? (ray x -> +d; blocked by faces facing the camera or two-sided)"""
            o = x + d * eps
            tot = 0.0
            for _ in range(16):
                h = self.bvh.ray_cast(o, d, maxd - tot)
                if h[0] is None:
                    return True
                loc, n, i, dd = h
                tot += dd
                if self.single_l[i] and n.dot(d) < 0:      # seen from the camera it is a back face: not drawn
                    o = loc + d * 2e-5
                    continue
                return False
            return False

        def gtri_label(self, gi):
            pi = self.piece_l[gi]
            p = self.pieces[pi]
            return p.part, int(self.tri_gcomp[gi] - self.coff[pi])

        def bone_of_point(self, x):
            best, bn = 1e9, "?"
            for n, (h, t, _) in self.bones.items():
                if n.startswith(("socket_", "cape_", "tabard_", "plume_")) or "helper" in n or "twist" in n:
                    continue
                d = (t - h)
                L = d.length_squared
                u = 0.0 if L < 1e-12 else max(0.0, min(1.0, (Vector(x) - h).dot(d) / L))
                dist = (Vector(x) - (h + d * u)).length
                if dist < best:
                    best, bn = dist, n
            return bn

    # =========================================================================================== G1
    def g1(S, res=96):
        """one-sided / open shells (bind): per piece welded boundary edges, non-manifold edges, inner-face thickness
        (ray along -normal inside the piece: first surface must be the piece's own back side), and the piece alone
        seen from 26 directions with back-face culling on vs off"""
        S.set_pose(("bind", None, 0))
        dirs = [Vector(d) for d in dirs26()]
        out = {"pieces": {}, "res": res}
        masks = {}
        for i, p in enumerate(S.pieces):
            P = S.P[i]
            bvh = S.piece_bvh(i)
            single = p.tri_single.tolist()
            N = p.tri_normals(P)
            A = p.tri_areas(P)
            C = P[p.tris].mean(1)
            # thickness per triangle
            th = np.full(p.nt, np.inf)
            for t in range(p.nt):
                n = Vector(N[t]); c = Vector(C[t])
                h = bvh.ray_cast(c - n * 2e-5, -n, 0.03)
                if h[0] is None:
                    th[t] = -1.0                              # nothing behind: an open sheet
                elif h[1].dot(-n) > 0:
                    th[t] = h[3]                              # own back side at this distance
                else:
                    th[t] = -2.0                              # another surface first: no inner face here
            onesided = (th < 0)
            thin = (th >= 0) & (th < 0.0015)
            atot = float(A.sum())
            # winding: a one-sided single-sided face whose normal points at the body (toward the nearest bone
            # segment) is drawn only from INSIDE the armour: seen from outside it vanishes (inverted normals)
            osd = onesided & p.tri_single
            inward = np.zeros(p.nt, bool)
            if osd.any():
                for t in np.where(osd)[0]:
                    inward[t] = float(N[t] @ S.radial(C[t])) < -0.2
            a_osd = float(A[osd].sum())
            inv_comps = []
            for c in range(p.ncomp):
                m_ = osd & (p.tri_comp == c)
                ac = float(A[m_].sum())
                if ac > 2e-5 and float(A[m_ & inward].sum()) > 0.6 * ac:
                    inv_comps.append(dict(comp=int(c), area_cm2=round(ac * 1e4, 2), bone=p.comp_dom[c],
                                          inward_frac=round(float(A[m_ & inward].sum()) / ac, 2)))
            rec = dict(owner=p.owner, tris=p.nt, comps=p.ncomp,
                       single_sided_materials=[m for m, s in zip(p.mat_names, p.mat_single) if s],
                       double_sided_materials=[m for m, s in zip(p.mat_names, p.mat_single) if not s],
                       boundary_edges=int(len(p.bnd_edges)), nonmanifold_edges=int(len(p.nm_edges)),
                       open_comps=int((p.comp_bnd > 0).sum()), closed_comps=int((p.comp_bnd == 0).sum()),
                       area_m2=round(atot, 4),
                       one_sided_area_frac=round(float(A[onesided].sum() / max(atot, 1e-12)), 3),
                       one_sided_single_sided_area_frac=round(float(A[onesided & p.tri_single].sum() / max(atot, 1e-12)), 3),
                       thin_lt_1p5mm_area_frac=round(float(A[thin].sum() / max(atot, 1e-12)), 3),
                       median_thickness_mm=round(float(np.median(th[th >= 0]) * 1000), 2) if (th >= 0).any() else None,
                       one_sided_inward_frac=round(float(A[osd & inward].sum()) / max(a_osd, 1e-12), 3) if a_osd else 0.0,
                       inverted_comps=sorted(inv_comps, key=lambda r: -r["area_cm2"])[:12])
            # worst open components (by open edges), where
            worst = np.argsort(-p.comp_bnd)[:6]
            rec["worst_open_comps"] = [dict(comp=int(c), boundary_edges=int(p.comp_bnd[c]), tris=int(p.comp_tris[c]),
                                            area_cm2=round(float(p.comp_area[c] * 1e4), 2), bone=p.comp_dom[c],
                                            material=p.comp_mat[c])
                                       for c in worst if p.comp_bnd[c] > 0]
            # 26-direction visibility, the piece alone (orthographic)
            lo, hi = P.min(0), P.max(0)
            cen = Vector(((lo + hi) / 2).tolist()); R = float(np.linalg.norm(hi - lo)) / 2 + 0.05
            vd = []
            mk = np.zeros((26, res, res), np.uint8)
            for k, u in enumerate(dirs):
                r = -u
                e1 = u.orthogonal().normalized(); e2 = u.cross(e1).normalized()
                pr1 = (P - np.array(cen)) @ np.array(e1); pr2 = (P - np.array(cen)) @ np.array(e2)
                ext = max(pr1.max() - pr1.min(), pr2.max() - pr2.min()) * 0.5 + 0.004
                px = 2 * ext / res
                c1, c2 = (pr1.max() + pr1.min()) / 2, (pr2.max() + pr2.min()) / 2
                cov = van = vbg = 0
                for iy in range(res):
                    y = c2 + (iy + 0.5 - res / 2) * px
                    for ix in range(res):
                        x = c1 + (ix + 0.5 - res / 2) * px
                        o = cen + e1 * x + e2 * y + u * R
                        h = bvh.ray_cast(o, r, 2 * R + 0.1)
                        if h[0] is None:
                            continue
                        cov += 1
                        if not (single[h[2]] and h[1].dot(r) > 0):
                            mk[k, iy, ix] = 1                   # drawn
                            continue
                        van += 1
                        d, gi = S.cast_vis(h[0] + r * 2e-5, r, 2 * R, bvh=bvh, single=single)
                        if d is None:
                            vbg += 1; mk[k, iy, ix] = 3           # the viewer shows the background: a hole
                        else:
                            mk[k, iy, ix] = 2                     # the viewer shows a farther surface (inside / far side)
                vd.append(dict(dir=dir_name(u), covered_px=cov, vanish_px=van, hole_px=vbg,
                               vanish_frac=round(van / max(cov, 1), 4), px_mm=round(px * 1000, 2)))
            masks[p.part] = mk
            rec["views"] = vd
            rec["vanish_frac_max"] = max(v["vanish_frac"] for v in vd)
            rec["vanish_worst_dir"] = max(vd, key=lambda v: v["vanish_frac"])["dir"]
            rec["hole_px_max"] = max(v["hole_px"] for v in vd)
            rec["error"] = bool(rec["vanish_frac_max"] > 0.002 or rec["one_sided_single_sided_area_frac"] > 0.01)
            out["pieces"][p.part] = rec
            log("G1 %-12s tris %5d comps %4d open %4d bnd %5d | one-sided %5.1f%% (single-sided %5.1f%%) thin %4.1f%% | "
                "vanish max %5.1f%% (%s) holes %4d px | inward %3.0f%% inverted comps %d" % (
                    p.part, p.nt, p.ncomp, rec["open_comps"], rec["boundary_edges"], rec["one_sided_area_frac"] * 100,
                    rec["one_sided_single_sided_area_frac"] * 100, rec["thin_lt_1p5mm_area_frac"] * 100,
                    rec["vanish_frac_max"] * 100, rec["vanish_worst_dir"], rec["hole_px_max"],
                    rec["one_sided_inward_frac"] * 100, len(rec["inverted_comps"])))
        np.savez_compressed(os.path.join(RDIR, "g1_%s_masks.npz" % S.kind), **masks)
        try:
            out["cull_audit"] = cull_audit(S)
            for k, v in out["cull_audit"].items():
                if isinstance(v, dict) and v.get("culled_tris"):
                    log("G1 cull %-12s culled %4d tris, %4d of them visible at bind %s" % (k, v["culled_tris"],
                                                                                        v["culled_visible_tris"], v["where"][:5]))
        except Exception as e:                       # the audit needs the live file; the gate itself does not
            import traceback; traceback.print_exc()
            out["cull_audit"] = {"error": repr(e)}
        return out

    def cull_audit(S):
        """faces the assembly's cull (build_knight.cull_hidden) removed although a camera outside the knight can see
        their place at bind (the viewer rule: single-sided back faces are transparent). The whole pieces come from the
        live dressed file out/knight_<kind>.blend (saved before the cull), matched to the export by triangle centroid."""
        from mathutils.kdtree import KDTree
        live = os.path.join(CH, "out", "knight_%s.blend" % S.kind)
        if not os.path.exists(live):
            return {"missing": live}
        S.set_pose(("bind", None, 0))
        names = [p.name for p in S.pieces]
        before = set(bpy.data.objects)
        with bpy.data.libraries.load(live, link=False) as (src, dst):
            dst.objects = [n for n in src.objects if n in names]
        new = [o for o in bpy.data.objects if o not in before]
        by = {}
        for o in new:
            if o.type == 'MESH':
                by[o.name.rsplit(".", 1)[0]] = o
        for o in new:
            bpy.context.scene.collection.objects.link(o)
            if o.type == 'MESH':
                for m in o.modifiers:
                    m.show_viewport = False
        bpy.context.view_layer.update()
        dg = bpy.context.evaluated_depsgraph_get()
        dirs = [Vector(d) for d in dirs26()]
        out = {}
        for pi, p in enumerate(S.pieces):
            o = by.get(p.name)
            if o is None:
                continue
            ev = o.evaluated_get(dg); me = ev.to_mesh(); me.calc_loop_triangles()
            nv = len(me.vertices); nt = len(me.loop_triangles)
            co = np.empty(nv * 3); me.vertices.foreach_get("co", co); co = co.reshape(-1, 3)
            M = np.array(o.matrix_world); co = co @ M[:3, :3].T + M[:3, 3]
            tv = np.empty(nt * 3, np.int32); me.loop_triangles.foreach_get("vertices", tv); tv = tv.reshape(-1, 3)
            pm = np.empty(len(me.polygons), np.int32); me.polygons.foreach_get("material_index", pm)
            tp = np.empty(nt, np.int32); me.loop_triangles.foreach_get("polygon_index", tp)
            mats = list(o.data.materials)
            ev.to_mesh_clear()
            Cl = co[tv].mean(1)
            Ce = S.P[pi][p.tris].mean(1)
            kd = KDTree(len(Ce))
            for i, c in enumerate(Ce):
                kd.insert(c, i)
            kd.balance()
            culled = np.array([kd.find(c)[2] > 3e-4 for c in Cl])
            matched = 1.0 - culled.mean() if nt else 1.0
            N = np.cross(co[tv[:, 1]] - co[tv[:, 0]], co[tv[:, 2]] - co[tv[:, 0]])
            N /= np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-20)
            vis = []
            for t in np.where(culled)[0]:
                x = Vector(Cl[t]); n = Vector(N[t])
                mi = pm[tp[t]] if len(pm) else 0
                two = not (mats[mi].use_backface_culling if mi < len(mats) and mats[mi] else False)
                for d in dirs:
                    if (d.dot(n) > 0.1 or (two and d.dot(n) < -0.1)) and S.escapes(x, d, eps=3e-4):
                        vis.append(int(t)); break
            out[p.part] = dict(owner=p.owner, live_tris=int(nt), export_tris=int(p.nt), culled_tris=int(culled.sum()),
                               culled_visible_tris=len(vis), match=round(float(matched), 4),
                               where=sorted({S.bone_of_point(Cl[t]) for t in vis[:60]}))
        for o in new:
            if o.name in bpy.data.objects:
                bpy.data.objects.remove(o, do_unlink=True)
        return out

    # =========================================================================================== inspect
    def inspect(S):
        S.set_pose(("bind", None, 0))
        out = {}
        for p in S.pieces:
            big = np.argsort(-p.comp_area)[:12]
            out[p.part] = dict(tris=p.nt, comps=p.ncomp, mats=p.mat_names, single=p.mat_single,
                               top=[dict(c=int(c), area_cm2=round(float(p.comp_area[c]) * 1e4, 1), tris=int(p.comp_tris[c]),
                                         bnd=int(p.comp_bnd[c]), cen=[round(float(x), 3) for x in p.comp_cen[c]],
                                         diag=round(float(p.comp_diag[c]), 3), bone=p.comp_dom[c], mat=p.comp_mat[c])
                                    for c in big])
            log("%-12s tris %5d comps %4d mats %s" % (p.part, p.nt, p.ncomp, list(zip(p.mat_names, p.mat_single))))
            for r in out[p.part]["top"]:
                log("     #%-4d %7.1f cm2 %5d tris bnd %4d cen %s diag %.3f %s %s" % (
                    r["c"], r["area_cm2"], r["tris"], r["bnd"], r["cen"], r["diag"], r["bone"], r["mat"]))
        jdump(out, "inspect_%s.json" % S.kind)

    # ------------------------------------------------------------------------------------------- geometry helpers
    def closest_on_tris(X, A, B, Cc):
        """closest points on triangles (A, B, C: (n, 3)) to points X (n, 3) (Ericson), vectorised"""
        ab = B - A; ac = Cc - A; ap = X - A
        d1 = (ab * ap).sum(1); d2 = (ac * ap).sum(1)
        bp = X - B; d3 = (ab * bp).sum(1); d4 = (ac * bp).sum(1)
        cp = X - Cc; d5 = (ab * cp).sum(1); d6 = (ac * cp).sum(1)
        va = d3 * d6 - d5 * d4; vb = d5 * d2 - d1 * d6; vc = d1 * d4 - d3 * d2
        den = np.where(np.abs(va + vb + vc) < 1e-30, 1e-30, va + vb + vc)
        v = vb / den; w = vc / den
        R = A + ab * v[:, None] + ac * w[:, None]
        m = (d1 <= 0) & (d2 <= 0); R[m] = A[m]
        m = (d3 >= 0) & (d4 <= d3); R[m] = B[m]
        m = (d6 >= 0) & (d5 <= d6); R[m] = Cc[m]
        m = (vc <= 0) & (d1 >= 0) & (d3 <= 0)
        t = d1 / np.where(np.abs(d1 - d3) < 1e-30, 1e-30, d1 - d3); R[m] = (A + ab * t[:, None])[m]
        m = (vb <= 0) & (d2 >= 0) & (d6 <= 0)
        t = d2 / np.where(np.abs(d2 - d6) < 1e-30, 1e-30, d2 - d6); R[m] = (A + ac * t[:, None])[m]
        m = (va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0)
        t = (d4 - d3) / np.where(np.abs((d4 - d3) + (d5 - d6)) < 1e-30, 1e-30, (d4 - d3) + (d5 - d6))
        R[m] = (B + (Cc - B) * t[:, None])[m]
        return R

    def tri_tri_points(P1, P2):
        """points on the intersection segment of triangle pairs (P1, P2: (n, 3, 3)): edge-crosses-plane points of each
        triangle that fall inside the other; returns (points (m, 3), pair index (m,))"""
        out, idx = [], []
        for Pa, Pb in ((P1, P2), (P2, P1)):
            nb = np.cross(Pb[:, 1] - Pb[:, 0], Pb[:, 2] - Pb[:, 0])
            nbl = np.linalg.norm(nb, axis=1, keepdims=True); nb = nb / np.maximum(nbl, 1e-20)
            db = (nb * Pb[:, 0]).sum(1)
            for i, j in ((0, 1), (1, 2), (2, 0)):
                a, b = Pa[:, i], Pa[:, j]
                sa = (nb * a).sum(1) - db; sb = (nb * b).sum(1) - db
                cr = (sa * sb) < 0
                t = sa / np.where(np.abs(sa - sb) < 1e-30, 1e-30, sa - sb)
                x = a + (b - a) * t[:, None]
                # inside Pb (barycentric sign test)
                inside = np.ones(len(x), bool)
                for k in range(3):
                    e0, e1 = Pb[:, k], Pb[:, (k + 1) % 3]
                    inside &= ((np.cross(e1 - e0, x - e0) * nb).sum(1) >= -1e-12)
                m = cr & inside
                if m.any():
                    out.append(x[m]); idx.append(np.where(m)[0])
        if not out:
            return np.zeros((0, 3)), np.zeros(0, np.int64)
        return np.concatenate(out), np.concatenate(idx)

    def voxel_cluster(X, cell):
        if not len(X):
            return np.zeros(0, np.int64)
        key = np.floor(X / cell).astype(np.int64)
        _, first = np.unique(key, axis=0, return_index=True)
        return np.sort(first)

    def vert_normals(p, P):
        N = np.cross(P[p.tris[:, 1]] - P[p.tris[:, 0]], P[p.tris[:, 2]] - P[p.tris[:, 0]])
        vn = np.zeros_like(P)
        for k in range(3):
            np.add.at(vn, p.tris[:, k], N)
        # welded neighbours share normals (split rims)
        return vn / np.maximum(np.linalg.norm(vn, axis=1, keepdims=True), 1e-20)

    def dilate(m, r):
        out = m.copy()
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                if dx * dx + dy * dy > r * r:
                    continue
                sh = np.zeros_like(m)
                ys = slice(max(dy, 0), m.shape[0] + min(dy, 0)); yd = slice(max(-dy, 0), m.shape[0] + min(-dy, 0))
                xs = slice(max(dx, 0), m.shape[1] + min(dx, 0)); xd = slice(max(-dx, 0), m.shape[1] + min(-dx, 0))
                sh[yd, xd] = m[ys, xs]
                out |= sh
        return out

    def erode(m, r):
        return ~dilate(~m, r)

    # =========================================================================================== G2 static
    def comp_select(S, spec):
        """'piece' -> (piece index, None = all components); 'piece:sel' -> (piece index, set of component ids)"""
        part, _, sel = spec.partition(":")
        if part not in S.by_part:
            return None
        pi = [p.part for p in S.pieces].index(part)
        p = S.pieces[pi]
        if not sel:
            return pi, None
        big = [c for c in np.argsort(-p.comp_area) if p.comp_area[c] > 0.02]
        if part == "helmet":
            if not big:
                return None
            skull = int(big[0])
            others = [c for c in big[1:] if p.comp_area[c] > 0.3 * p.comp_area[skull]]
            visor = int(min(others, key=lambda c: p.comp_cen[c][1])) if others else None
            if sel == "skull":
                return pi, {skull}
            if sel == "visor":
                return (pi, {visor}) if visor is not None else None
        if part.startswith("gauntlet"):
            s = part[-1]
            if sel == "cuff":
                return pi, {c for c in range(p.ncomp) if p.comp_dom[c].startswith("lowerarm")}
            if sel == "hand":
                return pi, {c for c in range(p.ncomp) if not p.comp_dom[c].startswith("lowerarm")}
        return pi, None

    FLOAT_MM = 0.55                    # 0.5 mm + 0.05 numerical margin (decals are lifted exactly 0.5 mm)

    def comp_kind(p, c):
        a, dg = float(p.comp_area[c]), float(p.comp_diag[c])
        if a < 3e-3 or dg < 0.06:
            return "small"                               # rivets, bosses, rosettes, rings, clasps, fans
        if p.comp_bnd[c] > 0 and a / max(dg * dg, 1e-9) < 0.15:
            return "strip"                               # trims, ribs, decal ribbons, bands
        return "plate"

    def g2_setup(S):
        """every component except each piece's main plate: its gap to the other surfaces at bind (per vertex, the
        nearest triangle of any other component within 6 mm) and anchors for the per-pose check"""
        S.set_pose(("bind", None, 0))
        cand = []
        for pi, p in enumerate(S.pieces):
            if p.part in ("plume",) or p.ncomp < 1:
                continue
            main = int(np.argmax(p.comp_area))
            for c in range(p.ncomp):
                if c != main or p.ncomp == 1 and p.part in ("clasps",):
                    cand.append((pi, c))
        anchors = []
        rec = {}
        for pi, c in cand:
            p = S.pieces[pi]
            gc = S.coff[pi] + c
            vids = p.comp_verts[c]
            vs = vids if len(vids) <= 400 else vids[np.linspace(0, len(vids) - 1, 400).astype(int)]
            best = np.full(len(vs), np.inf); sgn = np.zeros(len(vs)); ssingle = np.zeros(len(vs), bool)
            anc = []
            for k, v in enumerate(vs):
                x = Vector(S.P[pi][v])
                hits = S.bvh.find_nearest_range(x, 0.006)
                per = {}
                for loc, n, ti, d in hits:
                    g = int(S.tri_gcomp[ti])
                    if g == gc:
                        continue
                    if g not in per or d < per[g][1]:
                        per[g] = (ti, d, n, loc)
                if per:
                    ti, d, n, loc = min(per.values(), key=lambda t: t[1])
                    best[k] = d
                    sgn[k] = (x - loc).dot(n)
                    ssingle[k] = S.single_l[ti]
                    for g, (ti2, d2, _, _) in sorted(per.items(), key=lambda kv: kv[1][1])[:3]:
                        anc.append((S.voff[pi] + int(v), int(ti2)))
            gap = float(best.min()) if len(best) else np.inf
            if not np.isfinite(gap):
                hits = S.bvh.find_nearest_range(Vector(p.comp_cen[c]), 0.08)
                ds = [d for loc, n, ti, d in hits if int(S.tri_gcomp[ti]) != gc]
                gap = min(ds) if ds else 0.08
            fin = np.isfinite(best)
            touching = float((best < 0.0010).mean()) if len(best) else 0.0
            # buried: under the OUTER surface of a single-sided plate (its normals are reliable)
            bm = fin & ssingle
            buried = float((sgn[bm] < -0.0003).mean()) if bm.sum() >= 4 else 0.0
            rec[(pi, c)] = dict(piece=p.part, owner=p.owner, comp=int(c), kind=comp_kind(p, c), tris=int(p.comp_tris[c]),
                                area_mm2=round(float(p.comp_area[c]) * 1e6, 1), diag_mm=round(float(p.comp_diag[c]) * 1000, 1),
                                bone=p.comp_dom[c], bind_gap_mm=round(gap * 1000, 2), touching_frac=round(touching, 2),
                                buried_frac=round(buried, 2), centre=[round(float(x), 3) for x in p.comp_cen[c]],
                                open_edges=int(p.comp_bnd[c]))
            anchors.append(((pi, c), anc))
        return cand, rec, anchors

    def comp_visible(S, pi, c, ndir=26, nsamp=24):
        """None when no camera outside sees the component, else [point, direction] of one view that does"""
        p = S.pieces[pi]
        vids = p.comp_verts[c]
        vs = vids if len(vids) <= nsamp else vids[np.linspace(0, len(vids) - 1, nsamp).astype(int)]
        dirs = [Vector(d) for d in dirs26()]
        for v in vs:
            x = Vector(S.P[pi][v])
            for d in dirs:
                if S.escapes(x, d, eps=6e-4):
                    return [[round(float(q), 4) for q in x], [round(float(q), 4) for q in d]]
        return None

    def g2_static(S, cand, rec):
        """slivers, shards and the visibility of each candidate (bind)"""
        S.set_pose(("bind", None, 0))
        slivers, shards = [], {}
        for pi, p in enumerate(S.pieces):
            if p.part == "plume":
                continue
            P = S.P[pi]
            for c in range(p.ncomp):
                if p.comp_area[c] < 3e-6 or p.comp_diag[c] < 0.0025:
                    vw = comp_visible(S, pi, c)
                    slivers.append(dict(piece=p.part, owner=p.owner, comp=int(c), area_mm2=round(float(p.comp_area[c]) * 1e6, 2),
                                        diag_mm=round(float(p.comp_diag[c]) * 1000, 2), bone=p.comp_dom[c],
                                        visible=vw is not None, view=vw))
            a, b, cc = P[p.tris[:, 0]], P[p.tris[:, 1]], P[p.tris[:, 2]]
            L = np.stack([np.linalg.norm(b - a, axis=1), np.linalg.norm(cc - b, axis=1), np.linalg.norm(a - cc, axis=1)], 1)
            A = p.tri_areas(P)
            lmax = L.max(1)
            Ls = np.sort(L, 1)
            sinmin = 2 * A / np.maximum(Ls[:, 1] * Ls[:, 2], 1e-20)
            N = p.tri_normals(P)
            # needle triangles that stick out of the surface (max dihedral to a welded neighbour > 60 deg)
            wt = p.weld[p.tris]
            E = np.concatenate([np.sort(wt[:, [0, 1]], 1), np.sort(wt[:, [1, 2]], 1), np.sort(wt[:, [2, 0]], 1)])
            T = np.concatenate([np.arange(p.nt)] * 3)
            o_ = np.lexsort((E[:, 1], E[:, 0])); E, T = E[o_], T[o_]
            same = (E[1:] == E[:-1]).all(1)
            dih = np.zeros(p.nt)
            ia, ib = T[:-1][same], T[1:][same]
            cosd = (N[ia] * N[ib]).sum(1)
            ang = np.degrees(np.arccos(np.clip(cosd, -1, 1)))
            np.maximum.at(dih, ia, ang); np.maximum.at(dih, ib, ang)
            needle = (sinmin < math.sin(math.radians(5.0))) & (lmax > 0.005) & (A > 0)
            bad = np.where(needle & (dih > 60))[0]
            vis = []; vdir = []
            for t in bad[:400]:
                x = Vector(P[p.tris[t]].mean(0)); n = Vector(N[t])
                dv = next((d for d in (n, -n, (n + n.orthogonal()).normalized()) if S.escapes(x, d, eps=5e-4)), None)
                if dv is not None:
                    vis.append(int(t)); vdir.append([round(float(q), 4) for q in dv])
            if len(bad):
                pts = P[p.tris[vis]].mean(1) if vis else np.zeros((0, 3))
                shards[p.part] = dict(owner=p.owner, spike_tris=int(len(bad)), visible=len(vis),
                                      needle_tris=int(needle.sum()), where=sorted({S.bone_of_point(x) for x in pts[:40]}),
                                      points=[[round(float(q), 4) for q in x] for x in pts[:8]], dirs=vdir[:8])
        for (pi, c), r in rec.items():
            vw = comp_visible(S, pi, c)
            r["visible"] = vw is not None
            r["view"] = vw
        return slivers, shards

    def g2_pose(S, anchors, rec, acc):
        """per pose: each candidate's gap to its bind anchors"""
        Vw = S.V
        Tw = S.T
        for key, anc in anchors:
            if not anc:
                continue
            a = np.array(anc)
            X = Vw[a[:, 0]]
            tri = Tw[a[:, 1]]
            R = closest_on_tris(X, Vw[tri[:, 0]], Vw[tri[:, 1]], Vw[tri[:, 2]])
            d = np.linalg.norm(X - R, axis=1)
            # per vertex min over its anchors
            vmin = {}
            for (v, _), dd in zip(anc, d):
                if v not in vmin or dd < vmin[v]:
                    vmin[v] = dd
            dv = np.array(list(vmin.values()))
            gap = float(dv.min())
            A = acc.setdefault(key, dict(max_gap=0.0, at=None, min_touch=1.0))
            if gap > A["max_gap"]:
                A["max_gap"], A["at"] = gap, S.pose
            A["min_touch"] = min(A["min_touch"], float((dv < 0.001).mean()))

    def g2_report(S, rec, acc, slivers, shards):
        comps = []
        for key, r in rec.items():
            A = acc.get(key, {})
            r = dict(r)
            r["max_gap_mm"] = round(A.get("max_gap", 0.0) * 1000, 2)
            r["max_gap_at"] = A.get("at")
            r["min_touching_frac"] = round(A.get("min_touch", r["touching_frac"]), 2)
            seated = r["touching_frac"] >= 0.3
            attach = r["kind"] in ("small", "strip") or seated
            fl_bind = attach and r["bind_gap_mm"] > FLOAT_MM and not seated
            fl_motion = seated and r["max_gap_mm"] > FLOAT_MM
            r["seated_at_bind"] = seated
            r["floating"] = "bind" if fl_bind else ("motion" if fl_motion else "")
            r["partly_buried"] = bool(attach and 0.3 <= r["buried_frac"] <= 0.95)
            r["error"] = bool(r["visible"] and (r["floating"] or r["partly_buried"]))
            comps.append(r)
        comps.sort(key=lambda r: (-r["error"], -max(r["bind_gap_mm"] if r["floating"] == "bind" else 0, r["max_gap_mm"])))
        per = {}
        for r in comps:
            q = per.setdefault(r["piece"], dict(owner=r["owner"], checked=0, floating_bind=0, floating_motion=0,
                                                partly_buried=0, visible_errors=0))
            q["checked"] += 1
            q["floating_bind"] += r["floating"] == "bind"
            q["floating_motion"] += r["floating"] == "motion"
            q["partly_buried"] += r["partly_buried"]
            q["visible_errors"] += r["error"]
        for pc, v in shards.items():
            per.setdefault(pc, dict(owner=v["owner"], checked=0, floating_bind=0, floating_motion=0, partly_buried=0,
                                    visible_errors=0))["spikes_visible"] = v["visible"]
        for sv in slivers:
            q = per.setdefault(sv["piece"], dict(owner=sv["owner"], checked=0, floating_bind=0, floating_motion=0,
                                                 partly_buried=0, visible_errors=0))
            q["slivers"] = q.get("slivers", 0) + 1
        return dict(threshold_mm=FLOAT_MM, rule="attachments (small comps < 30 cm2 / 6 cm, thin strips, or anything "
                    "seated on another surface at bind): floating = > 0.55 mm from every other surface at bind (not "
                    "seated) or a seated one lifting > 0.55 mm off in a pose; partly buried = 30-95 % of its vertices "
                    "under a plate's outer surface; spikes = needle triangles (< 5 deg, > 5 mm) folded > 60 deg off "
                    "their neighbours; only visible ones are errors", pieces=per, components=comps, slivers=slivers,
                    shards=shards)

    # =========================================================================================== G3 seams
    def seam_setup(S, max_windows=4, near=0.02):
        """seam windows at bind: points where A's boundary (or surface) comes within `near` of B, clustered"""
        S.set_pose(("bind", None, 0))
        seams = []
        for nm, sa, sb in expand_seams():
            A = comp_select(S, sa); B = comp_select(S, sb)
            if A is None or B is None:
                seams.append(dict(name=nm, a=sa, b=sb, missing=True))
                continue
            seams.append(dict(name=nm, a=sa, b=sb, A=A, B=B, missing=False))
        return seams

    def _sub_bvh(S, pi, comps):
        p = S.pieces[pi]
        m = np.ones(p.nt, bool) if comps is None else np.isin(p.tri_comp, list(comps))
        tris = p.tris[m]
        return BVHTree.FromPolygons(S.P[pi].tolist(), tris.tolist(), epsilon=0.0), m

    def _gset(S, sel):
        pi, comps = sel
        p = S.pieces[pi]
        base = S.coff[pi]
        return {base + c for c in (range(p.ncomp) if comps is None else comps)}

    def seam_windows(S, sd, max_windows=4, near=0.02, cell=0.07):
        """windows (centre, outward normal) along the seam in the current pose"""
        pa, ca = sd["A"]; pb, cb = sd["B"]
        bvhB, _ = _sub_bvh(S, pb, cb); bvhA, _ = _sub_bvh(S, pa, ca)
        pts = []
        for (pi, comps, other) in ((pa, ca, bvhB), (pb, cb, bvhA)):
            p = S.pieces[pi]
            P = S.P[pi]
            be = p.bnd_edges[np.isin(p.bnd_comp, list(comps))] if comps is not None else p.bnd_edges
            vids = np.unique(be)
            if len(vids) < 8:              # closed solids: any vertex near the other side
                vids = np.where(np.isin(p.vcomp, list(comps)))[0] if comps is not None else np.arange(p.nv)
            vids = vids[:: max(1, len(vids) // 1500)]
            for v in vids:
                h = other.find_nearest(Vector(P[v]), near)
                if h[0] is not None:
                    pts.append((P[v] + np.array(h[0])) / 2)
        if not pts:
            return []
        X = np.array(pts)
        key = np.floor(X / cell).astype(np.int64)
        u, inv, cnt = np.unique(key, axis=0, return_inverse=True, return_counts=True)
        inv = inv.reshape(-1)
        order = np.argsort(-cnt)[:max_windows]
        wins = []
        for k in order:
            m = inv == k
            c = X[m][np.argmin(np.linalg.norm(X[m] - X[m].mean(0), axis=1))]      # a real seam point near the mean
            wins.append((c, S.radial(c), int(cnt[k])))
        return wins

    # pixel classes: 0 exterior (nothing / past the silhouette), 1 A, 2 B, 3 other drawn surface near the seam,
    # 4 under-layer, 5 see-through to the background (the ray passed the inside of a shell), 6 see-through into the
    # armour's interior (a drawn inward-facing surface or one reached through the inside of a shell, far behind)
    def seam_raster(S, sd, c, d, res=28, px=0.0035, R=0.5, deep=0.025):
        u = Vector(d); r = -u
        e1 = u.orthogonal().normalized(); e2 = u.cross(e1).normalized()
        ga, gb = sd["gA"], sd["gB"]
        cen = Vector(c)
        lab = np.zeros((res, res), np.int8)
        dep = np.full((res, res), np.inf)
        hitc = np.full((res, res), -1, np.int64)
        skip = np.zeros((res, res), np.int16)
        hx = np.zeros((res, res, 3))
        tg = S.tri_gcomp
        for iy in range(res):
            y = (iy + 0.5 - res / 2) * px
            for ix in range(res):
                x = (ix + 0.5 - res / 2) * px
                o = cen + e1 * x + e2 * y + u * R
                t, gi, ns = S.cast_vis(o, r, 2 * R, count=True)
                skip[iy, ix] = ns
                if t is None:
                    if ns:
                        lab[iy, ix] = 5
                    continue
                dep[iy, ix] = t - R
                g = int(tg[gi])
                hitc[iy, ix] = gi
                hx[iy, ix] = (o + r * t)[:]
                if g in ga:
                    lab[iy, ix] = 1
                elif g in gb:
                    lab[iy, ix] = 2
                elif S.pieces[S.piece_l[gi]].part in UNDER:
                    lab[iy, ix] = 4
                else:
                    lab[iy, ix] = 3
        # far drawn hits of other pieces: interior if reached through a shell or facing inward
        far = (lab == 3) & (dep > deep)
        for iy, ix in zip(*np.where(far)):
            gi = int(hitc[iy, ix])
            pi = S.piece_l[gi]; lt = gi - S.toff[pi]; p = S.pieces[pi]
            Pq = S.P[pi][p.tris[lt]]
            n = np.cross(Pq[1] - Pq[0], Pq[2] - Pq[0])
            inward = float(n @ S.radial(hx[iy, ix])) < 0
            if skip[iy, ix] or inward:
                lab[iy, ix] = 6
            else:
                lab[iy, ix] = 0
        return lab, dep, hitc

    def _reach(m, dy, dx, R):
        """pixels that have an m pixel within R steps along +(dy, dx)"""
        out = np.zeros_like(m)
        H, W = m.shape
        for k in range(1, R + 1):
            sy, sx = dy * k, dx * k
            sh = np.zeros_like(m)
            ys = slice(max(sy, 0), H + min(sy, 0)); yd = slice(max(-sy, 0), H + min(-sy, 0))
            xs = slice(max(sx, 0), W + min(sx, 0)); xd = slice(max(-sx, 0), W + min(-sx, 0))
            sh[yd, xd] = m[ys, xs]
            out |= sh
        return out

    def seam_gaps(lab, rad=4):
        """see-through pixels BETWEEN the two pieces: background (5) or armour interior (6) with A on one side and B on
        the other along one of 4 lines within `rad` px (14 mm). A pixel just outside the pair's silhouette (only one
        side covered) is not a gap, whatever lies behind it"""
        A = lab == 1; B = lab == 2
        if not A.any() or not B.any():
            return np.zeros_like(A), 0
        see = (lab == 5) | (lab == 6)
        between = np.zeros_like(A)
        for dy, dx in ((0, 1), (1, 0), (1, 1), (1, -1)):
            ap, am = _reach(A, dy, dx, rad), _reach(A, -dy, -dx, rad)
            bp, bm = _reach(B, dy, dx, rad), _reach(B, -dy, -dx, rad)
            between |= (ap & bm) | (bp & am)
        gap = see & between
        return gap, int(gap.sum())

    def g3_pose(S, seams, acc, dirs, store):
        for sd in seams:
            if sd["missing"]:
                continue
            sd["gA"] = _gset(S, sd["A"]); sd["gB"] = _gset(S, sd["B"])
            wins = seam_windows(S, sd)
            A = acc.setdefault(sd["name"], dict(max_gap_px=0, at=None, dir=None, frames_with_gap=0, windows=0,
                                                see=dict(background=0, interior=0), worst_centre=None, views=0))
            A["windows"] = max(A["windows"], len(wins))
            fmax = 0
            for c, n, npts in wins:
                for d in dirs:
                    if np.dot(d, n) < 0.25:
                        continue
                    if S.inside(Vector(c) + Vector(d) * 0.5):
                        continue
                    A["views"] += 1
                    lab, dep, hitc = seam_raster(S, sd, c, d)
                    gap, ng = seam_gaps(lab)
                    fmax = max(fmax, ng)
                    if S.pose_group != "extreme" and ng > A.get("clip_worst", {}).get("px", 0):
                        A["clip_worst"] = dict(px=ng, at=S.pose, dir=dir_name(d), mm2=round(ng * 3.5 * 3.5, 1),
                                               centre=[round(float(x), 4) for x in c],
                                               dir_vec=[round(float(x), 4) for x in d])
                    if ng > A["max_gap_px"]:
                        A["max_gap_px"], A["at"], A["dir"] = ng, S.pose, dir_name(d)
                        A["worst_centre"] = [round(float(x), 4) for x in c]
                        A["worst_dir_vec"] = [round(float(x), 4) for x in d]          # a camera along +d sees the gap
                        A["worst_where"] = S.bone_of_point(c)
                        bgp = int((gap & (lab == 5)).sum())
                        A["see"] = dict(background=bgp, interior=ng - bgp)
                        A["see_pieces"] = sorted({S.gtri_label(int(g))[0] for g in hitc[gap & (lab == 6)]})
                        store[sd["name"]] = (lab.copy(), gap.copy(), S.pose, dir_name(d))
            if fmax >= 3:
                A["frames_with_gap"] += 1

    # =========================================================================================== G4 penetration
    INTENDED = {("helmet", "plume"): "plume socket", ("plume",): "plume locks interleave (alpha hair cards)"}
    DEEP = 0.0015          # deeper than this = a plate / prop through another surface; shallower = a seated contact

    def g4_pose(S, acc, vis_dirs, cluster=0.003, cap=400):
        nP = len(S.pieces)
        bv = [S.piece_bvh(i) for i in range(nP)]
        pairs = []
        for i in range(nP):
            for j in range(i + 1, nP):
                ov = bv[i].overlap(bv[j])
                if ov:
                    pairs.append((i, j, np.array(ov)))
            ov = bv[i].overlap(bv[i])
            if ov:
                ov = np.array(ov)
                p = S.pieces[i]
                ov = ov[ov[:, 0] < ov[:, 1]]
                ov = ov[p.tri_comp[ov[:, 0]] != p.tri_comp[ov[:, 1]]]
                if len(ov):
                    # drop pairs sharing a welded vertex (touching neighbours of split rims)
                    wa = p.weld[p.tris[ov[:, 0]]]; wb = p.weld[p.tris[ov[:, 1]]]
                    share = (wa[:, :, None] == wb[:, None, :]).any((1, 2))
                    ov = ov[~share]
                if len(ov):
                    pairs.append((i, i, ov))
        for i, j, ov in pairs:
            pa, pb = S.pieces[i], S.pieces[j]
            if i == j:
                keys = {}
                ca = pa.tri_comp[ov[:, 0]]; cb = pa.tri_comp[ov[:, 1]]
                for k, (x, y) in enumerate(zip(ca, cb)):
                    keys.setdefault((min(x, y), max(x, y)), []).append(k)
                groups = [("%s#%d|%s#%d" % (pa.part, a, pa.part, b), ov[idx], pa.comp_dom[a] + "/" + pa.comp_dom[b])
                          for (a, b), idx in keys.items()]
            else:
                groups = [("%s|%s" % (pa.part, pb.part), ov, "")]
            for key, o2, bones in groups:
                T1 = S.P[i][pa.tris[o2[:, 0]]]; T2 = S.P[j][pb.tris[o2[:, 1]]]
                # local penetration depth per triangle pair: how far each triangle reaches behind the other's plane,
                # the smaller of the two (a trim seated 1 mm into a plate -> 1 mm; plates crossing -> several mm)
                def behind(Ta, Tb):
                    n = np.cross(Ta[:, 1] - Ta[:, 0], Ta[:, 2] - Ta[:, 0])
                    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-20)
                    sd = ((Tb - Ta[:, :1]) * n[:, None]).sum(2)
                    return np.maximum(0, -sd.min(1))
                dep = np.minimum(behind(T1, T2), behind(T2, T1))
                X, pidx = tri_tri_points(T1, T2)
                sel = voxel_cluster(X, cluster)
                pts = X[sel]; pd = dep[pidx[sel]] if len(sel) else np.zeros(0)
                if len(pts) > cap:
                    k_ = np.linspace(0, len(pts) - 1, cap).astype(int); pts = pts[k_]; pd = pd[k_]
                nvis = 0; ndeep = 0; vpts = []; dmax = 0.0
                for x, dd in zip(pts, pd):
                    xv = Vector(x)
                    dv = next((d for d in vis_dirs if S.escapes(xv, d)), None)
                    if dv is not None:
                        nvis += 1
                        if dd > DEEP:
                            ndeep += 1
                            dmax = max(dmax, float(dd))
                            if len(vpts) < 12:
                                vpts.append((x, dv))
                A = acc.setdefault(key, dict(owner=sorted({pa.owner, pb.owner}), max_tri_pairs=0, at=None, frames=0,
                                             max_visible=0, visible_at=None, frames_visible=0, max_visible_deep=0,
                                             deep_at=None, frames_visible_deep=0, max_depth_mm=0.0, where=[],
                                             self=(i == j), bones=bones))
                A["frames"] += 1
                G = A.setdefault("per_group", {}).setdefault(S.pose_group, dict(frames=0, frames_visible_deep=0,
                                                                                max_visible_deep=0, at=None))
                G["frames"] += 1
                if ndeep:
                    G["frames_visible_deep"] += 1
                if ndeep > G["max_visible_deep"]:
                    G["max_visible_deep"], G["at"] = ndeep, S.pose
                    if vpts:                                   # one visible spot + a direction a camera sees it from
                        G["point"] = [round(float(c), 4) for c in vpts[0][0]]
                        G["dir"] = [round(float(c), 4) for c in vpts[0][1]]
                A["max_depth_mm"] = max(A["max_depth_mm"], round(float(dep.max()) * 1000, 1) if len(dep) else 0.0)
                if len(o2) > A["max_tri_pairs"]:
                    A["max_tri_pairs"], A["at"] = int(len(o2)), S.pose
                if nvis:
                    A["frames_visible"] += 1
                if ndeep:
                    A["frames_visible_deep"] += 1
                if nvis > A["max_visible"]:
                    A["max_visible"], A["visible_at"] = nvis, S.pose
                if ndeep > A["max_visible_deep"]:
                    A["max_visible_deep"], A["deep_at"] = ndeep, S.pose
                    A["visible_deep_depth_mm"] = round(dmax * 1000, 1)
                    A["where"] = sorted({S.bone_of_point(x) for x, _ in vpts})
                    A["visible_points"] = [[round(float(c), 4) for c in x] for x, _ in vpts[:6]]
                    A["visible_dirs"] = [[round(float(c), 4) for c in d] for _, d in vpts[:6]]   # a camera along +d sees it

    def g4_report(acc, nposes):
        out = {}
        for k, A in sorted(acc.items(), key=lambda kv: (-kv[1]["max_visible_deep"], -kv[1]["max_visible"], -kv[1]["max_tri_pairs"])):
            A = dict(A)
            parts = tuple(sorted({s.split("#")[0] for s in k.split("|")}))
            A["intended"] = INTENDED.get(parts, "")
            pg = A.get("per_group", {})
            # what the user sees in the viewer = bind + the clips; the extreme test poses are the robustness margin
            A["max_visible_deep_clips"] = max([v["max_visible_deep"] for g_, v in pg.items() if g_ != "extreme"] or [0])
            A["max_visible_deep_extreme"] = pg.get("extreme", {}).get("max_visible_deep", 0)
            if A["intended"]:
                A["verdict"] = "intended"
            elif A["max_visible_deep"] >= 2:
                A["verdict"] = "error"
            else:
                A["verdict"] = "warning" if A["max_tri_pairs"] else "ok"
            A["kind"] = ("visible penetration" if A["max_visible_deep"] >= 2 else
                         "visible seated contact (<= 1.5 mm)" if A["max_visible"] else "hidden")
            A["seen_in"] = ("clips" if A["max_visible_deep_clips"] >= 2 else
                            "extreme poses only" if A["max_visible_deep_extreme"] >= 2 else "-")
            out[k] = A
        return dict(poses=nposes, depth_note="max_depth_mm / visible_deep_depth_mm = how far one triangle of the pair "
                    "reaches behind the other's plane (the smaller of the two): an upper bound of the local penetration",
                    pairs=out)

    # =========================================================================================== G5 grip
    FINGERS = ("index", "middle", "ring", "pinky", "thumb")

    def g5_setup(S):
        info = {}
        sw = S.by_part.get("sword"); ga = S.by_part.get("gauntlet_r")
        if sw is not None:
            gm = [k for k, n in enumerate(sw.mat_names) if "grip" in n.lower()]
            info["grip_tris"] = np.isin(sw.tri_mat, gm)
        for side in ("r", "l"):
            g = S.by_part.get("gauntlet_" + side)
            if g is None:
                continue
            glove = int(np.argmax([a if g.comp_dom[c].startswith("hand") else 0 for c, a in enumerate(g.comp_area)]))
            fplates = [c for c in range(g.ncomp) if g.comp_dom[c].split("_")[0] in FINGERS and c != glove]
            info[side] = dict(glove=glove, fplates=fplates)
        sh = S.by_part.get("shield")
        if sh is not None:
            info["straps"] = [c for c in range(sh.ncomp) if "leather" in sh.comp_mat[c].lower() and sh.comp_area[c] < 0.06]
        return info

    def g5_pose(S, info, acc):
        rig = S.rig
        pn = S.pose
        clip = pn.rsplit("_f", 1)[0] if "_f" in pn else pn
        R = acc.setdefault(clip, {})
        # --- sword grip in the right fist
        sw = S.by_part.get("sword"); gr = S.by_part.get("gauntlet_r")
        if sw is not None and gr is not None and "socket_weapon_r" in S.bones and "r" in info:
            h, t, M = S.bones["socket_weapon_r"]
            ax = (M.to_3x3() @ Vector((0, 1, 0))).normalized(); o = M.translation
            si = [p.part for p in S.pieces].index("sword"); gi = [p.part for p in S.pieces].index("gauntlet_r")
            Pg = S.P[si][np.unique(sw.tris[info["grip_tris"]])]
            rel = Pg - np.array(o)
            along = rel @ np.array(ax)
            rad = np.linalg.norm(rel - np.outer(along, np.array(ax)), axis=1)
            r_grip = float(np.median(rad)); g_lo, g_hi = float(along.min()), float(along.max())
            Pgl = S.P[gi]
            per = {}
            for f in FINGERS:
                vm = np.array([str(b).startswith(f + "_0") and str(b)[-4:-2] in ("02", "03") for b in gr.dom])
                if f == "thumb":
                    vm = np.array([str(b).startswith("thumb_0") for b in gr.dom])
                if not vm.any():
                    continue
                X = Pgl[vm] - np.array(o)
                al = X @ np.array(ax)
                rr = np.linalg.norm(X - np.outer(al, np.array(ax)), axis=1) - r_grip
                inband = (al > g_lo - 0.01) & (al < g_hi + 0.01)
                if not inband.any():
                    per[f] = dict(contact_mm=None, inside_mm=None)
                    continue
                per[f] = dict(contact_mm=round(float(rr[inband].min()) * 1000, 1),
                              inside_mm=round(float(max(0.0, -rr[inband].min())) * 1000, 1))
            # angular enclosure of the grip by the hand (glove + plates within 12 mm of the grip surface)
            X = Pgl - np.array(o); al = X @ np.array(ax)
            radial = X - np.outer(al, np.array(ax)); rr = np.linalg.norm(radial, axis=1)
            m = (al > g_lo) & (al < g_hi) & (rr < r_grip + 0.012)
            cov = 0.0
            if m.any():
                e1 = np.array(ax.orthogonal().normalized()); e2 = np.cross(np.array(ax), e1)
                ang = np.arctan2(radial[m] @ e2, radial[m] @ e1)
                cov = len(np.unique(np.floor((ang + math.pi) / (2 * math.pi) * 36).astype(int))) / 36.0
            # hand triangles through the grip (the grip's own triangles only)
            bvg = BVHTree.FromPolygons(S.P[si].tolist(), sw.tris[info["grip_tris"]].tolist(), epsilon=0.0)
            thr = len(bvg.overlap(S.piece_bvh(gi)))
            # finger plates seated on the glove
            glove = info["r"]["glove"]
            bgl, _ = _sub_bvh(S, gi, {glove})
            fl = []; seat = []
            for c in info["r"]["fplates"]:
                vs = gr.comp_verts[c]
                ds = []
                for v in vs[:: max(1, len(vs) // 40)]:
                    hn = bgl.find_nearest(Vector(Pgl[v]), 0.03)
                    ds.append(hn[3] if hn[0] is not None else 0.03)
                ds = np.array(ds)
                fl.append(float(ds.min()))
                # the plate's inner side (its closest 20 % of vertices) over the glove: a plate 'rests' on the finger
                # when that is <= 2 mm; a larger standoff reads as a floating cage at 0.3-1 m
                seat.append((float(np.percentile(ds, 20)), float(np.median(ds)), gr.comp_dom[c]))
            floating = [s for s in seat if s[0] > 0.002]
            rec = dict(grip_radius_mm=round(r_grip * 1000, 1), enclosure=round(cov, 2), hand_tris_through_grip=int(thr),
                       fingers=per, plate_seat_gap_max_mm=round(max(fl) * 1000, 1) if fl else None,
                       plates_off_glove_gt2mm=int(sum(1 for x in fl if x > 0.002)),
                       plate_standoff_p20_mm=[round(s[0] * 1000, 1) for s in seat],
                       plate_standoff_median_mm=[round(s[1] * 1000, 1) for s in seat],
                       plates_floating=len(floating), plates=len(seat),
                       floating_on=sorted({s[2] for s in floating}))
            G = R.setdefault("sword_grip", dict(worst=None, frames=0, fail_frames=0))
            G["frames"] += 1
            bad = (cov < 0.6) or thr > 0 or any((v.get("inside_mm") or 0) > 1.0 for v in per.values()) \
                or any(v.get("contact_mm") is not None and v["contact_mm"] > 4.0 for k, v in per.items() if k != "thumb") \
                or rec["plates_floating"] > 0
            if bad:
                G["fail_frames"] += 1
            score = thr + (1 - cov) * 100 + sum((v.get("inside_mm") or 0) for v in per.values())
            if G["worst"] is None or score > G["worst"][0]:
                G["worst"] = (score, pn, rec)
        # --- shield: straps round the forearm, left fist on the hand strap
        sh = S.by_part.get("shield"); vl = S.by_part.get("vambrace_l")
        if sh is not None and "straps" in info and "lowerarm_l" in S.bones:
            si = [p.part for p in S.pieces].index("shield")
            h, t, M = S.bones["lowerarm_l"]
            A0, A1 = np.array(h), np.array(t)
            loops = []
            for c in info["straps"]:
                X = S.P[si][sh.comp_verts[c]]
                cen = X.mean(0)
                U, s_, Vt = np.linalg.svd(X - cen)
                nrm = Vt[2]; rad = float(np.median(np.linalg.norm(X - cen, axis=1)))
                d = A1 - A0
                den = float(nrm @ d)
                if abs(den) < 1e-9:
                    loops.append(dict(comp=int(c), encloses=False)); continue
                tt = float(nrm @ (cen - A0)) / den
                xp = A0 + d * tt
                off = float(np.linalg.norm(xp - cen))
                loops.append(dict(comp=int(c), axis_through_loop=bool(-0.2 <= tt <= 1.2 and off < 0.8 * rad),
                                  axis_offset_mm=round(off * 1000, 1), loop_radius_mm=round(rad * 1000, 1)))
            vi = [p.part for p in S.pieces].index("vambrace_l") if vl is not None else None
            thr = 0
            if vi is not None:
                bs, _ = _sub_bvh(S, si, set(info["straps"]))
                thr = len(bs.overlap(S.piece_bvh(vi)))
            hand = None
            if "socket_hand_l" in S.bones:
                x = S.bones["socket_hand_l"][2].translation
                bsh = S.piece_bvh(si)
                hn = bsh.find_nearest(x, 0.2)
                hand = round(hn[3] * 1000, 1) if hn[0] is not None else 200.0
            rec = dict(loops=loops, loops_enclosing=sum(1 for l in loops if l.get("axis_through_loop")),
                       strap_tris_through_vambrace=int(thr), hand_socket_to_shield_mm=hand)
            G = R.setdefault("shield", dict(worst=None, frames=0, fail_frames=0))
            G["frames"] += 1
            bad = thr > 0 or (hand is not None and hand > 15.0)
            if bad:
                G["fail_frames"] += 1
            score = thr + (hand or 0)
            if G["worst"] is None or score > G["worst"][0]:
                G["worst"] = (score, pn, rec)

    # =========================================================================================== G6 stretch
    def set_nominal(p):
        """per material: ('tile', uv_per_m) for tileables (true texel scale), ('median', None) for atlases / trims"""
        out = []
        for m in p.o.data.materials:
            nm = (m.name if m else "").lower()
            img = ""
            if m and m.use_nodes:
                for n in m.node_tree.nodes:
                    if n.type == 'TEX_IMAGE' and n.image and "base" in n.image.name.lower():
                        img = n.image.name.lower(); break
                if not img:
                    for n in m.node_tree.nodes:
                        if n.type == 'TEX_IMAGE' and n.image:
                            img = n.image.name.lower(); break
            kind = "other"
            if "mail" in nm or "mail" in img:
                out.append(("mail", 1.0 / 0.168)); continue
            if "tabard" in nm or "cape" in nm or "shield" in nm:
                out.append(("atlas", None)); continue
            if "armour" in nm or "gold" in nm or "steel" in nm or "blade" in nm or "enamel" in nm:
                out.append(("trim", None)); continue
            if "leather" in nm or "strap" in nm or "grip" in nm:
                out.append(("leather", None)); continue
            if "cloth" in nm:
                out.append(("cloth", None)); continue
            out.append((kind, None))
        return out or [("other", None)]

    def uv_frames(p, P):
        """per triangle: dP/du, dP/dv (metres per UV unit, (nt, 3) each)"""
        a, b, c = P[p.tris[:, 0]], P[p.tris[:, 1]], P[p.tris[:, 2]]
        uv = p.tri_uv.astype(np.float64)
        du1 = uv[:, 1] - uv[:, 0]; du2 = uv[:, 2] - uv[:, 0]
        e1 = b - a; e2 = c - a
        det = du1[:, 0] * du2[:, 1] - du1[:, 1] * du2[:, 0]
        ok = np.abs(det) > 1e-14
        det = np.where(ok, det, 1.0)
        Pu = (e1 * du2[:, 1:2] - e2 * du1[:, 1:2]) / det[:, None]
        Pv = (e2 * du1[:, 0:1] - e1 * du2[:, 0:1]) / det[:, None]
        return Pu, Pv, ok

    def principal(Pu, Pv, ku, kv):
        """singular values of [Pu*ku, Pv*kv] (3x2): surface metres per nominal texel metre along the principal axes"""
        a = (Pu * Pu).sum(1) * ku * ku; c = (Pv * Pv).sum(1) * kv * kv; b = (Pu * Pv).sum(1) * ku * kv
        tr = a + c; dt = np.sqrt(np.maximum((a - c) ** 2 + 4 * b * b, 0))
        return np.sqrt(np.maximum((tr + dt) / 2, 0)), np.sqrt(np.maximum((tr - dt) / 2, 0))

    def rest_frames(P, tris):
        a, b, c = P[tris[:, 0]], P[tris[:, 1]], P[tris[:, 2]]
        e1 = b - a; e2 = c - a
        l1 = np.maximum(np.linalg.norm(e1, axis=1), 1e-12)
        t1 = e1 / l1[:, None]
        cx = (e2 * t1).sum(1)
        t2v = e2 - t1 * cx[:, None]
        cy = np.maximum(np.linalg.norm(t2v, axis=1), 1e-12)
        return l1, cx, cy

    def deform_stretch(P, tris, fr):
        """principal stretches of each triangle vs its bind shape (deformation gradient)"""
        l1, cx, cy = fr
        a, b, c = P[tris[:, 0]], P[tris[:, 1]], P[tris[:, 2]]
        f1 = b - a; f2 = c - a
        Fx = f1 / l1[:, None]
        Fy = (f2 - f1 * (cx / l1)[:, None]) / cy[:, None]
        return principal(Fx, Fy, 1.0, 1.0)

    def g6_setup(S):
        """UV texel distortion at bind (mail: true scale vs the tile and vs the piece median; other sets: anisotropy
        of the texel footprint) + the bind frames for the per-pose deformation stretch"""
        S.set_pose(("bind", None, 0))
        st = {}
        dirs = [Vector(d) for d in fib_dirs(12)]
        for pi, p in enumerate(S.pieces):
            if p.tri_uv is None or p.part in ("plume",):
                continue
            noms = set_nominal(p)
            P = S.P[pi]
            Pu, Pv, ok = uv_frames(p, P)
            kind = np.array([noms[min(m, len(noms) - 1)][0] for m in p.tri_mat])
            A = p.tri_areas(P)
            ku = np.full(p.nt, 1.0); kv = np.full(p.nt, 1.0)
            for k in set(kind.tolist()):
                m = (kind == k) & ok
                if not m.any():
                    continue
                if k == "mail":
                    ku[m] = kv[m] = 0.168
                elif k in ("atlas", "leather", "cloth", "other"):
                    lu = np.linalg.norm(Pu[m], axis=1); lv = np.linalg.norm(Pv[m], axis=1)
                    wu = np.argsort(lu); cu = np.cumsum(A[m][wu]); mu = lu[wu][min(np.searchsorted(cu, cu[-1] / 2), len(wu) - 1)]
                    wv = np.argsort(lv); cv = np.cumsum(A[m][wv]); mv = lv[wv][min(np.searchsorted(cv, cv[-1] / 2), len(wv) - 1)]
                    ku[m] = mu; kv[m] = mv
                # trim sheet: isotropic texels by design (keeps ku = kv = 1: only the anisotropy is read)
            s1, s2 = principal(Pu, Pv, 1.0 / ku, 1.0 / kv)
            N = p.tri_normals(P); C = P[p.tris].mean(1)
            vis = np.zeros(p.nt, bool)
            stp = max(1, p.nt // 3000)
            idx = np.arange(0, p.nt, stp)
            for t in idx:
                n = Vector(N[t]); x = Vector(C[t])
                two = not S.single_l[S.toff[pi] + t]
                for d in dirs:
                    dn = d.dot(n)
                    # front hemisphere always; the back hemisphere too when the material is double-sided (drawn)
                    if (dn > 0.2 or (two and dn < -0.2)) and S.escapes(x, d, eps=5e-4):
                        vis[t] = True; break
            if stp > 1:
                vis = vis[idx][np.minimum(np.arange(p.nt) // stp, len(idx) - 1)]
            # skinned (multi-bone) components of a trim-sheet piece (the gauntlet's leather glove) deform like leather;
            # rigid plates (one bone) must not stretch at all
            multi = np.array([len(set(p.dom[v].astype(str))) > 1 for v in p.comp_verts])
            dtol = np.array([DEFTOL.get(k_, 0.3) for k_ in kind])
            dtol[(kind == "trim") & multi[p.tri_comp]] = 0.30
            st[pi] = dict(kind=kind, ok=ok, vis=vis, A=A, s1=s1, s2=s2, fr=rest_frames(P, p.tris), dtol=dtol,
                          dmax=np.ones(p.nt), dmin=np.ones(p.nt), dat=np.array([""] * p.nt, dtype=object))
        return st

    def g6_pose(S, st):
        for pi, d in st.items():
            p = S.pieces[pi]
            f1, f2 = deform_stretch(S.P[pi], p.tris, d["fr"])
            up = f1 > d["dmax"]; dn = f2 < d["dmin"]
            d["dmax"][up] = f1[up]; d["dmin"][dn] = f2[dn]
            d["dat"][up | dn] = S.pose

    UVTOL = {"mail": 0.15}
    DEFTOL = {"mail": 0.15, "cloth": 0.30, "leather": 0.30, "atlas": 0.30, "trim": 0.05, "other": 0.30}

    def _hot(S, pi, tri_idx):
        p = S.pieces[pi]
        if not len(tri_idx):
            return {}
        C = S.P[pi][p.tris[tri_idx]].mean(1)
        wh = {}
        for x in C[:: max(1, len(C) // 60)]:
            b = S.bone_of_point(x); wh[b] = wh.get(b, 0) + 1
        return dict(sorted(wh.items(), key=lambda kv: -kv[1])[:6])

    def g6_report(S, st):
        S.set_pose(("bind", None, 0))
        out = {}
        for pi, d in st.items():
            p = S.pieces[pi]
            rec = {}
            for k in sorted(set(d["kind"].tolist())):
                m = (d["kind"] == k) & d["ok"]
                if not m.any():
                    continue
                A = d["A"]; vis = d["vis"]; s1, s2 = d["s1"], d["s2"]
                am = A[m].sum(); mv = m & vis; av = max(A[mv].sum(), 1e-12)
                aniso = s1 / np.maximum(s2, 1e-9)
                r = dict(area_cm2=round(float(am) * 1e4, 1), visible_area_cm2=round(float(A[mv].sum()) * 1e4, 1))
                if k == "mail":
                    scale = np.sqrt(s1 * s2)
                    med = float(np.median(scale[mv])) if mv.any() else float(np.median(scale[m]))
                    r1, r2 = s1 / med, s2 / med
                    bad = m & ((r1 > 1 + UVTOL["mail"]) | (r2 < 1 - UVTOL["mail"]))
                    r.update(nominal="mail_riveted: 1 UV = 0.168 m (10.5 mm rings)", ring_scale_vs_nominal=round(med, 3),
                             uv_out_frac=round(float(A[bad].sum() / am), 3),
                             uv_out_visible_frac=round(float(A[bad & vis].sum() / av), 3),
                             stretch_p99=round(float(np.percentile(r1[mv if mv.any() else m], 99)), 2),
                             squash_p1=round(float(np.percentile(r2[mv if mv.any() else m], 1)), 2),
                             uv_hot=_hot(S, pi, np.where(bad & vis)[0]))
                    uv_bad = bad
                else:
                    bad = m & (aniso > 1.5)
                    r.update(uv_aniso_p99=round(float(np.percentile(aniso[mv if mv.any() else m], 99)), 2),
                             uv_out_frac=round(float(A[bad].sum() / am), 3),
                             uv_out_visible_frac=round(float(A[bad & vis].sum() / av), 3),
                             uv_hot=_hot(S, pi, np.where(bad & vis)[0]))
                    uv_bad = bad
                streak = m & (aniso > 4.0) & vis
                r["uv_streak_visible_cm2"] = round(float(A[streak].sum()) * 1e4, 2)
                tol = d["dtol"]
                db = m & ((d["dmax"] > 1 + tol) | (d["dmin"] < 1 - tol))
                r.update(deform_tolerance=sorted({round(float(x), 2) for x in tol[m]}), deform_stretch_max=round(float(d["dmax"][m].max()), 3),
                         deform_squash_min=round(float(d["dmin"][m].min()), 3),
                         deform_out_visible_frac=round(float(A[db & vis].sum() / av), 3),
                         deform_hot=_hot(S, pi, np.where(db & vis)[0]))
                if (db & vis).any():
                    ats = d["dat"][db & vis]
                    vals, cn = np.unique(ats.astype(str), return_counts=True)
                    r["deform_worst_poses"] = [str(v) for v in vals[np.argsort(-cn)][:4]]
                lim = 0.02 if k == "mail" else 0.05
                r["error"] = bool(r["uv_out_visible_frac"] > lim or r["deform_out_visible_frac"] > lim
                                  or r["uv_streak_visible_cm2"] > 2.0)
                rec[k] = r
            out[p.part] = dict(owner=p.owner, sets=rec)
        return out

    # =========================================================================================== G7 head in helmet
    def albedo_sampler(p):
        """base colour texture x vertex colour 'Color' (the AO the material multiplies in) at a hit point"""
        img = None
        m = p.o.data.materials[0] if p.o.data.materials else None
        if m and m.use_nodes:
            bs = next((n for n in m.node_tree.nodes if n.type == 'BSDF_PRINCIPLED'), None)
            for n in m.node_tree.nodes:
                if n.type == 'TEX_IMAGE' and n.image and ("base" in n.image.name.lower() or "albedo" in n.image.name.lower()):
                    img = n.image; break
            if img is None:
                img = next((n.image for n in m.node_tree.nodes if n.type == 'TEX_IMAGE' and n.image), None)
        arr = None
        if img is not None:
            w, h = img.size
            arr = np.array(img.pixels[:], np.float32).reshape(h, w, img.channels)[..., :3]
        col = None
        if "Color" in p.o.data.color_attributes:
            ca = p.o.data.color_attributes["Color"]
            if ca.domain == 'POINT':
                col = np.empty(len(ca.data) * 4, np.float32); ca.data.foreach_get("color", col)
                col = col.reshape(-1, 4)[:, :3]
        return arr, col

    def g7(S, res=110, size=0.26):
        S.set_pose(("bind", None, 0))
        hi = [p.part for p in S.pieces].index("helmet") if "helmet" in S.by_part else None
        if hi is None:
            return {"missing": "no helmet in the look"}
        hp = S.pieces[hi]
        img, vcol = albedo_sampler(hp)
        B = S.bones
        eye = (B["eye_l"][0] + B["eye_r"][0]) / 2 if "eye_l" in B else B["head"][0] + Vector((0, -0.08, 0.1))
        hc = (B["head"][0] + B["head"][1]) / 2
        fwd = Vector((0, -1, 0))
        dirs = [Vector(d) for d in dirs26() if Vector(d).dot(fwd) > -0.3]
        px = size / res
        res_out = {"views": [], "eye_mid": [round(x, 3) for x in eye]}
        tot = dict(through=0, background=0, bright_interior=0, dark_lining=0, face=0, other=0)
        masks = {}
        Ph = S.P[hi]
        for u in dirs:
            r = -u
            e1 = u.orthogonal().normalized(); e2 = u.cross(e1).normalized()
            cls = np.zeros((res, res), np.int8)      # 0 none, 1 shell, 2 through-bg, 3 through-bright, 4 dark, 5 face, 6 other
            shell = np.zeros((res, res), bool)
            for iy in range(res):
                y = (iy + 0.5 - res / 2) * px
                for ix in range(res):
                    x = (ix + 0.5 - res / 2) * px
                    o = eye + e1 * x + e2 * y + u * 0.6
                    t, gi = S.cast_vis(o, r, 1.2)
                    if t is None:
                        continue
                    part, comp = S.gtri_label(gi)
                    hit = o + r * t
                    if part == "helmet":
                        lt = gi - S.toff[hi]
                        tri = hp.tris[lt]
                        n = np.cross(Ph[tri[1]] - Ph[tri[0]], Ph[tri[2]] - Ph[tri[0]])
                        outward = float(np.dot(n, np.array(hit - hc))) > 0
                        if outward and float(np.dot(n, np.array(r))) < 0:
                            cls[iy, ix] = 1; shell[iy, ix] = True
                            continue
                        # inner face / lining: albedo x AO
                        lum = 0.5
                        a, b, c = Ph[tri[0]], Ph[tri[1]], Ph[tri[2]]
                        v0, v1, v2 = b - a, c - a, np.array(hit) - a
                        d00, d01, d11 = v0 @ v0, v0 @ v1, v1 @ v1; d20, d21 = v2 @ v0, v2 @ v1
                        den = d00 * d11 - d01 * d01
                        if abs(den) > 1e-20:
                            bv = (d11 * d20 - d01 * d21) / den; bw = (d00 * d21 - d01 * d20) / den; bu = 1 - bv - bw
                            alb = np.array([0.5, 0.5, 0.5])
                            if img is not None and hp.tri_uv is not None:
                                uvp = hp.tri_uv[lt, 0] * bu + hp.tri_uv[lt, 1] * bv + hp.tri_uv[lt, 2] * bw
                                H, W = img.shape[:2]
                                alb = img[int(np.clip(uvp[1] % 1.0, 0, 0.9999) * H), int(np.clip(uvp[0] % 1.0, 0, 0.9999) * W)]
                            if vcol is not None:
                                alb = alb * (vcol[tri[0]] * bu + vcol[tri[1]] * bv + vcol[tri[2]] * bw)
                            lum = float(0.2126 * alb[0] + 0.7152 * alb[1] + 0.0722 * alb[2])
                        cls[iy, ix] = 4 if lum < 0.06 else 3
                    elif part in ("head", "eyes") or part.startswith(("eyes", "head")):
                        cls[iy, ix] = 5
                    else:
                        cls[iy, ix] = 6
            env = erode(dilate(shell, 6), 6)
            through = env & ~shell
            bgm = through & (cls == 0)
            cls[bgm] = 2
            k = dict(dir=dir_name(u), through_px=int(through.sum()), background_px=int(bgm.sum()),
                     bright_interior_px=int((through & (cls == 3)).sum()), dark_lining_px=int((through & (cls == 4)).sum()),
                     face_px=int((through & (cls == 5)).sum()), other_px=int((through & (cls == 6)).sum()))
            res_out["views"].append(k)
            for a_, b_ in (("through", "through_px"), ("background", "background_px"), ("bright_interior", "bright_interior_px"),
                           ("dark_lining", "dark_lining_px"), ("face", "face_px"), ("other", "other_px")):
                tot[a_] += k[b_]
            cls[~env] = 0
            cls[shell] = 1
            masks[dir_name(u)] = cls
        res_out["total_px"] = tot
        res_out["px_mm"] = round(px * 1000, 2)
        res_out["error"] = bool(tot["background"] > 0)
        res_out["head_present_in_look"] = any(p.part in ("head",) for p in S.pieces)
        np.savez_compressed(os.path.join(RDIR, "g7_%s_masks.npz" % S.kind), **masks)
        return res_out

    # =========================================================================================== G8 battery
    TARGETS = [("helm", "head", 0.5, 0.60), ("shoulder_l", "upperarm_l", 0.0, 0.60), ("shoulder_r", "upperarm_r", 0.0, 0.60),
               ("elbow_l", "lowerarm_l", 0.0, 0.50), ("elbow_r", "lowerarm_r", 0.0, 0.50), ("hand_l", "hand_l", 0.5, 0.45),
               ("hand_r", "hand_r", 0.5, 0.45), ("waist", "spine_01", 0.0, 0.80), ("knee_l", "calf_l", 0.0, 0.60),
               ("knee_r", "calf_r", 0.0, 0.60)]
    KEYS = [("bind", None, 0), ("idle_f090", "idle", 90), ("walk_f008", "walk", 8), ("run_f005", "run", 5),
            ("attack_sword_f015", "attack_sword", 15), ("block_shield_f028", "block_shield", 28)]

    def to_gltf(v):
        return [round(float(v[0]), 4), round(float(v[2]), 4), round(float(-v[1]), 4)]

    def g8_cams(S):
        """battery cameras at the current pose: {target: [(dir name, cam pos, target pos)]}"""
        out = {}
        for nm, bone, t, dist in TARGETS:
            if bone not in S.bones:
                continue
            h, tl, _ = S.bones[bone]
            tgt = h + (tl - h) * t
            cams = []
            for d in dirs26():
                pos = tgt + Vector(d) * dist
                if pos.z < 0.03 or S.inside(pos) or S.bvh.find_nearest(pos, 0.03)[0] is not None:
                    continue                                    # under the floor / enclosed by the knight / in a plate
                h = S.bvh.ray_cast(pos, (tgt - pos).normalized(), dist)
                if h[0] is not None and h[3] < 0.4 * dist:
                    continue                                    # pressed against another part (helm, plume, shield)
                cams.append((dir_name(d), pos, tgt))
            out[nm] = cams
        return out

    def g8_rays(S, cams, res=64, fov=30.0):
        """per camera: pixels whose nearest surface is a single-sided back face (the viewer shows what is behind:
        'vanish') and the subset where the viewer then shows the background ('hole')"""
        out = {}
        masks = {}
        tanh = math.tan(math.radians(fov) / 2)
        for nm, lst in cams.items():
            rows = []
            for dn, pos, tgt in lst:
                f = (tgt - pos).normalized()
                up = Vector((0, 0, 1)) if abs(f.z) < 0.95 else Vector((0, 1, 0))
                rt = f.cross(up).normalized(); up2 = rt.cross(f).normalized()
                van = hole = cov = 0
                mk = np.zeros((res, res), np.uint8)
                pieces = {}
                for iy in range(res):
                    sy = (1 - 2 * (iy + 0.5) / res) * tanh
                    for ix in range(res):
                        sx = (2 * (ix + 0.5) / res - 1) * tanh
                        r = (f + rt * sx + up2 * sy).normalized()
                        d, gi, drawn = S.cast_first(pos, r, 3.0)
                        if d is None:
                            continue
                        cov += 1
                        if drawn:
                            mk[iy, ix] = 1; continue
                        van += 1
                        pn_ = S.pieces[S.piece_l[gi]].part
                        pieces[pn_] = pieces.get(pn_, 0) + 1
                        t2, g2_ = S.cast_vis(pos + r * (d + 2e-5), r, 3.0)
                        if t2 is None:
                            hole += 1; mk[iy, ix] = 3
                        else:
                            mk[iy, ix] = 2
                rows.append(dict(dir=dn, vanish_px=van, hole_px=hole, covered_px=cov,
                                 vanish_frac=round(van / max(cov, 1), 4),
                                 vanish_pieces=dict(sorted(pieces.items(), key=lambda kv: -kv[1])[:4])))
                masks["%s|%s" % (nm, dn)] = mk
            out[nm] = rows
        return out, masks

    # =========================================================================================== runner
    def run_all(S, what, step, quick):
        kind = S.kind
        res = {}
        if "g1" in what:
            t = time.time(); r = g1(S); jdump(r, "g1_%s.json" % kind); log("G1 %.0fs" % (time.time() - t))
            res["g1"] = r
        if "g7" in what:
            t = time.time(); r = g7(S); jdump(r, "g7_%s.json" % kind); log("G7 %.0fs %s" % (time.time() - t, r.get("total_px")))
        per_pose = [g for g in ("g2", "g3", "g4", "g5", "g6") if g in what]
        if per_pose:
            poses = S.pose_list(step=step)
            if quick:
                poses = poses[:1] + [p for p in poses if p[0] in ("arms_up", "walk_f009", "attack_sword_f015", "block_shield_f027",
                                                                  "idle_f090")]
            log("%d poses: %s .. %s" % (len(poses), poses[0][0], poses[-1][0]))
            if "g2" in what:
                cand, rec2, anchors = g2_setup(S)
                sl, shd = g2_static(S, cand, rec2)
                acc2 = {}
                log("G2 %d attachment components" % len(cand))
            if "g3" in what:
                seams = seam_setup(S); acc3 = {}; store3 = {}
                dirs = [np.array(d) for d in dirs26()]
            if "g5" in what:
                info5 = g5_setup(S); acc5 = {}
            if "g6" in what:
                st6 = g6_setup(S)
            acc4 = {}
            vis_dirs = [Vector(d) for d in fib_dirs(42)]
            t0 = time.time()
            for k, pose in enumerate(poses):
                S.set_pose(pose)
                if "g2" in what:
                    g2_pose(S, anchors, rec2, acc2)
                if "g3" in what:
                    g3_pose(S, seams, acc3, dirs, store3)
                if "g4" in what:
                    g4_pose(S, acc4, vis_dirs)
                if "g5" in what and pose[1] not in (None, "extreme"):
                    g5_pose(S, info5, acc5)
                if "g6" in what:
                    g6_pose(S, st6)
                if k % 10 == 0:
                    log("  pose %3d/%d %-20s %.0fs" % (k + 1, len(poses), pose[0], time.time() - t0))
            if "g2" in what:
                r = g2_report(S, rec2, acc2, sl, shd); r["poses"] = len(poses); jdump(r, "g2_%s.json" % kind)
            if "g3" in what:
                out3 = dict(poses=len(poses), window_px_mm=3.5, rule="gap px = pixels between the two pieces (within 14 mm "
                            "of both) whose first drawn surface is the background or lies > 25 mm behind them (not an "
                            "under-layer); error >= 3 px (37 mm2) in any pose / direction", seams={})
                for sd in seams:
                    if sd["missing"]:
                        out3["seams"][sd["name"]] = dict(missing=True, a=sd["a"], b=sd["b"])
                        continue
                    A = acc3.get(sd["name"], {})
                    A = dict(A); A["a"], A["b"] = sd["a"], sd["b"]
                    A["gap_mm2"] = round(A.get("max_gap_px", 0) * 3.5 * 3.5, 1)
                    A["verdict"] = "error" if A.get("max_gap_px", 0) >= 3 else ("warning" if A.get("max_gap_px", 0) else "ok")
                    pa = sd["a"].split(":")[0]; pb = sd["b"].split(":")[0]
                    A["owner"] = sorted({owner_of(pa), owner_of(pb)})
                    out3["seams"][sd["name"]] = A
                jdump(out3, "g3_%s.json" % kind)
                np.savez_compressed(os.path.join(RDIR, "g3_%s_worst.npz" % kind),
                                    **{k.replace("/", "_").replace(" ", "_"): np.stack([v[0].astype(np.int8), v[1].astype(np.int8)])
                                       for k, v in store3.items()})
                json.dump({k: [v[2], v[3]] for k, v in store3.items()}, open(os.path.join(RDIR, "g3_%s_worst.json" % kind), "w"))
            if "g4" in what:
                jdump(g4_report(acc4, len(poses)), "g4_%s.json" % kind)
            if "g5" in what:
                out5 = {}
                for clip, d in acc5.items():
                    out5[clip] = {k: dict(frames=v["frames"], fail_frames=v["fail_frames"], worst_pose=v["worst"][1],
                                          worst=v["worst"][2]) for k, v in d.items()}
                jdump(dict(rule="sword: enclosure >= 0.6 of the grip circumference, 0 hand triangles through the grip "
                           "wrap, no finger > 1 mm inside the grip, fingers 2-4 within 4 mm of it, every finger plate's "
                           "inner side (20th percentile of its vertices) within 2 mm of the glove; shield: 0 strap "
                           "triangles through the vambrace, left fist socket within "
                           "15 mm of a shield surface (hand strap / handle)", clips=out5), "g5_%s.json" % kind)
            if "g6" in what:
                jdump(dict(poses=len(poses), rule="uv: mail principal texel stretch vs the piece's visible median "
                           "(+-15 %; its scale vs the 0.168 m tile reported), other sets texel anisotropy > 1.5 (trim "
                           "sheet isotropic, atlases / leather normalised by their median U / V density); streak = "
                           "anisotropy > 4; deform: principal stretch of each triangle vs bind, max over poses (mail "
                           "+-15 %, cloth / leather +-30 %, plates +-5 %); error when > 2 % (mail) / 5 % of the VISIBLE "
                           "area is out or > 2 cm2 of visible streaks", pieces=g6_report(S, st6)), "g6_%s.json" % kind)
        if "g8" in what:
            t = time.time()
            battery = {"targets": [t_[0] for t_ in TARGETS], "keys": [k[0] for k in KEYS], "cams": {}, "rays": {}}
            allm = {}
            for key in KEYS:
                if key[1] and not bpy.data.actions.get(key[1]):
                    continue
                S.set_pose(key)
                cams = g8_cams(S)
                battery["cams"][key[0]] = {nm: [dict(dir=dn, p=to_gltf(p), t=to_gltf(tg)) for dn, p, tg in lst]
                                           for nm, lst in cams.items()}
                rr, mk = g8_rays(S, cams, res=int(os.environ.get("INT_G8RES", "64")))
                battery["rays"][key[0]] = rr
                for k2, v in mk.items():
                    allm["%s|%s" % (key[0], k2)] = v
                log("G8 %-20s %s" % (key[0], {nm: max(r["vanish_frac"] for r in rows) for nm, rows in rr.items()}))
                if key[1] == USER_CLIP[0] and key[2] == USER_CLIP[1]:
                    battery["user_bones"] = {b: to_gltf(S.bones[b][0]) for b in {v[0] for v in USER_VIEWS.values()}
                                             if b in S.bones}
            np.savez_compressed(os.path.join(RDIR, "g8_%s_masks.npz" % kind), **allm)
            jdump(battery, "g8_%s.json" % kind)
            log("G8 %.0fs" % (time.time() - t))

    def blender_main(args):
        kind = args[0] if args else "male"
        opts = dict(a.split("=", 1) for a in args[1:] if "=" in a)
        what = [a for a in args[1:] if "=" not in a]
        os.makedirs(RDIR, exist_ok=True)
        rig = bpy.data.objects["rts_" + kind]
        t0 = time.time()
        S = Scene(rig, kind, opts.get("look", "helm"))
        log("scene %s: %d pieces, %d tris, %d components (%.1fs)" % (kind, len(S.pieces), S.NT, S.NC, time.time() - t0))
        if "inspect" in what:
            inspect(S)
        gates = [g for g in what if g in ("g1", "g2", "g3", "g4", "g5", "g6", "g7", "g8")]
        if "all" in what:
            gates = ["g1", "g2", "g3", "g4", "g5", "g6", "g7", "g8"]
        run_all(S, gates, int(opts.get("step", 3)), "quick" in what)
        log("integrity %s done in %.0fs" % (kind, time.time() - t0))


# =================================================================================================== orchestrator
def log_(*a):
    print("INT", *a, flush=True)


def run_blender(kinds, gates, step=3, parallel=True):
    """the Blender gates, one process per body (in parallel), on out/knight_<kind>_export.blend"""
    import subprocess
    env = dict(os.environ); env["BLENDER_USER_RESOURCES"] = os.path.join(CH, "blender_profile")
    procs = []
    for k in kinds:
        cmd = [BL, "-b", os.path.join(CH, "out", "knight_%s_export.blend" % k), "--python-exit-code", "1", "-P",
               os.path.abspath(__file__), "--", k] + list(gates) + ["step=%d" % step]
        logf = open(os.path.join(RDIR, "blender_%s.log" % k), "w")
        log_("$ " + " ".join(cmd[2:]) + "  (log renders/integrity/blender_%s.log)" % k)
        p = subprocess.Popen(cmd, cwd=CH, env=env, stdout=logf, stderr=subprocess.STDOUT)
        procs.append((k, p, logf))
        if not parallel:
            p.wait()
    bad = []
    for k, p, f in procs:
        rc = p.wait(); f.close()
        tail = open(os.path.join(RDIR, "blender_%s.log" % k)).read().splitlines()
        for line in tail:
            if line.startswith("INT") and ("done" in line or "G1 " in line[:8]):
                pass
        if rc != 0:
            bad.append(k)
            print("\n".join(tail[-40:]))
    if bad:
        raise SystemExit("armour_integrity Blender gates failed for %s" % bad)


def user_view_specs(kind, battery, size_dir):
    bones = battery.get("user_bones", {})
    specs = []
    for nm, (bone, off, az, el, dist) in USER_VIEWS.items():
        if bone not in bones:
            continue
        t = [bones[bone][i] + off[i] for i in range(3)]
        a, e = math.radians(az), math.radians(el)
        p = [t[0] + dist * math.cos(e) * math.sin(a), t[1] + dist * math.sin(e), t[2] + dist * math.cos(e) * math.cos(a)]
        specs.append({"out": os.path.join(size_dir, "user_%s.png" % nm), "view": {"p": [round(x, 4) for x in p],
                      "t": [round(x, 4) for x in t], "fov": 30}, "clip": USER_CLIP[0], "time": USER_CLIP[1] / 30.0,
                      "scene": "town"})
    return specs


def shots(page, specs, w, h, tag):
    import subprocess
    sp = os.path.join(RDIR, "_specs_%s.json" % tag)
    json.dump(specs, open(sp, "w"))
    env = dict(os.environ); env["BATCH"] = sp; env["CLEAN"] = "1"
    env["TIMEOUT"] = str(int(180 + 2 * len(specs)))        # shot_cdp.js times the WHOLE batch (default 240 s)
    t0 = time.time()
    r = subprocess.run([os.path.join(CH, "viewer", "shot.sh"), page, "-", "Full body", "", "", "", str(w), str(h)],
                       cwd=CH, env=env, capture_output=True, text=True)
    os.remove(sp)
    if r.returncode != 0:
        print(r.stdout[-2000:], r.stderr[-3000:])
        raise SystemExit("viewer shots failed (%s)" % tag)
    log_("viewer %s: %d shots in %.0fs" % (tag, len(specs), time.time() - t0))


def viewer_battery(kinds, tile=240):
    """G8 in the web viewer: the user's views + the orbit battery (cameras from g8_<kind>.json) -> contact sheets"""
    import shutil
    from PIL import Image, ImageDraw
    out = {}
    for k in kinds:
        page = os.path.join(CH, "viewer", "knight_%s_lookdev.html" % k)
        bat = json.load(open(os.path.join(RDIR, "g8_%s.json" % k)))
        tiles = os.path.join(RDIR, "_tiles_%s" % k)
        shutil.rmtree(tiles, ignore_errors=True); os.makedirs(tiles)
        uv = user_view_specs(k, bat, os.path.join(RDIR))
        for sp in uv:
            sp["out"] = os.path.join(RDIR, "g8_%s_%s" % (k, os.path.basename(sp["out"])))
        shots(page, uv, 900, 900, "user_" + k)
        specs = []
        index = []
        clips = {"bind": (None, 0)}
        for key in bat["cams"]:
            if key != "bind":
                c, f = key.rsplit("_f", 1); clips[key] = (c, int(f))
        for key, targets in bat["cams"].items():
            clip, f = clips[key]
            for tg, cams in targets.items():
                for cm in cams:
                    fn = os.path.join(tiles, "%s__%s__%s.png" % (key, tg, cm["dir"]))
                    sp = {"out": fn, "view": {"p": cm["p"], "t": cm["t"], "fov": 30}, "scene": "town"}
                    if clip:
                        sp["clip"] = clip; sp["time"] = f / 30.0
                    else:
                        sp["clip"] = None
                    specs.append(sp); index.append((key, tg, cm["dir"], fn))
        shots(page, specs, tile, tile, "battery_" + k)
        sheets = []
        dirs = [dir_name(d) for d in dirs26()]
        T = 150
        for key in bat["cams"]:
            tgs = list(bat["cams"][key])
            sh = Image.new("RGB", (120 + len(dirs) * T, 22 + len(tgs) * T), (22, 22, 26))
            dr = ImageDraw.Draw(sh)
            for j, dn in enumerate(dirs):
                dr.text((120 + j * T + 3, 5), dn, fill=(255, 220, 120))
            for i, tg in enumerate(tgs):
                dr.text((5, 22 + i * T + T // 2), tg, fill=(255, 220, 120))
                for j, dn in enumerate(dirs):
                    fn = os.path.join(tiles, "%s__%s__%s.png" % (key, tg, dn))
                    if os.path.exists(fn):
                        sh.paste(Image.open(fn).convert("RGB").resize((T, T), Image.LANCZOS), (120 + j * T, 22 + i * T))
            f = os.path.join(RDIR, "g8_%s_battery_%s.jpg" % (k, key))
            sh.save(f, quality=86)
            sheets.append(os.path.relpath(f, CH))
        shutil.rmtree(tiles, ignore_errors=True)
        out[k] = dict(user_views=[os.path.relpath(sp["out"], CH) for sp in uv], battery_sheets=sheets,
                      battery_shots=len(specs))
    # the user's screenshots next to the reproductions
    rows = []
    for nm in USER_VIEWS:
        ref = os.path.join(CH, "refs", "feedback", "user_r5_%s.png" % nm)
        ims = [Image.open(ref).convert("RGB")] + [Image.open(os.path.join(RDIR, "g8_%s_user_%s.png" % (k, nm))).convert("RGB")
                                                  for k in kinds if os.path.exists(os.path.join(RDIR, "g8_%s_user_%s.png" % (k, nm)))]
        H = 420
        ims = [im.resize((round(im.width * H / im.height), H)) for im in ims]
        rows.append((nm, ims))
    W = max(sum(im.width for im in ims) for _, ims in rows) + 10
    sh = Image.new("RGB", (W, len(rows) * 450), (22, 22, 26))
    dr = ImageDraw.Draw(sh)
    for i, (nm, ims) in enumerate(rows):
        x = 0
        labs = ["user_r5_%s (user)" % nm] + ["reproduced: %s (rebuilt knight)" % k for k in kinds]
        for im, lab in zip(ims, labs):
            sh.paste(im, (x, i * 450 + 26)); dr.text((x + 6, i * 450 + 6), lab, fill=(255, 220, 120)); x += im.width + 4
    sh.save(os.path.join(RDIR, "g8_user_views.png"))
    cams = {k: user_view_specs(k, json.load(open(os.path.join(RDIR, "g8_%s.json" % k))), "") for k in kinds}
    json.dump(dict(doc="user_r5 views reproduced in viewer/knight_<kind>_lookdev.html (LD.set specs: view {p, t, fov} "
                       "in glTF metres, clip, time, scene)", views={k: [{kk: vv for kk, vv in sp.items() if kk != "out"}
                                                                          for sp in v] for k, v in cams.items()},
                   definition={n: dict(bone=b, offset=o, azimuth=a, elevation=e, distance=d)
                               for n, (b, o, a, e, d) in USER_VIEWS.items()}, outputs=out),
              open(os.path.join(RDIR, "g8_viewer.json"), "w"), indent=1)
    log_("wrote renders/integrity/g8_viewer.json, g8_user_views.png and %d battery sheets"
         % sum(len(v["battery_sheets"]) for v in out.values()))
    return out


# ------------------------------------------------------------------------------------------------ sheets (PIL)
PAL = {0: (40, 40, 46), 1: (170, 170, 170), 2: (255, 150, 30), 3: (255, 0, 200)}


def _mask_img(m, pal, scale=1, flip=True):
    from PIL import Image
    import numpy as np
    rgb = np.zeros(m.shape + (3,), np.uint8)
    for k, c in pal.items():
        rgb[m == k] = c
    im = Image.fromarray(rgb[::-1] if flip else rgb)      # orthographic rasters are filled bottom-up (+e2)
    return im.resize((m.shape[1] * scale, m.shape[0] * scale), Image.NEAREST) if scale > 1 else im


def sheets(kinds):
    from PIL import Image, ImageDraw
    import numpy as np
    made = []
    for k in kinds:
        # G1: every piece alone from 26 directions: grey drawn, orange = the viewer shows a farther surface through it,
        # magenta = the viewer shows the background (the surface vanished)
        f = os.path.join(RDIR, "g1_%s_masks.npz" % k)
        if os.path.exists(f):
            z = np.load(f)
            g1 = json.load(open(os.path.join(RDIR, "g1_%s.json" % k)))
            parts = [p for p in z.files if g1["pieces"].get(p, {}).get("vanish_frac_max", 0) > 0.002]
            T = 64
            dirs = [dir_name(d) for d in dirs26()]
            sh = Image.new("RGB", (130 + 26 * T, 20 + len(parts) * T), (22, 22, 26))
            dr = ImageDraw.Draw(sh)
            for j, dn in enumerate(dirs):
                dr.text((130 + j * T + 2, 4), dn[:10], fill=(255, 220, 120))
            for i, pn in enumerate(parts):
                m = z[pn]
                dr.text((4, 20 + i * T + 20), "%s %.0f%%" % (pn, 100 * g1["pieces"][pn]["vanish_frac_max"]), fill=(255, 220, 120))
                for j in range(m.shape[0]):
                    sh.paste(_mask_img(m[j], PAL).resize((T, T), Image.NEAREST), (130 + j * T, 20 + i * T))
            fn = os.path.join(RDIR, "g1_%s_vanish_sheet.png" % k); sh.save(fn); made.append(fn)
        # G3: worst window per seam (grey A, light B, blue under-layer, dark other; red = see-through gap)
        f = os.path.join(RDIR, "g3_%s_worst.npz" % k)
        if os.path.exists(f):
            z = np.load(f); meta = json.load(open(os.path.join(RDIR, "g3_%s_worst.json" % k)))
            pal = {0: (30, 30, 34), 1: (150, 150, 150), 2: (215, 215, 200), 3: (90, 90, 110), 4: (60, 90, 170),
                   5: (255, 0, 200), 6: (255, 140, 0)}
            names = list(meta)
            T = 168; cols = 6
            rows = (len(names) + cols - 1) // cols
            sh = Image.new("RGB", (cols * T, rows * (T + 30)), (22, 22, 26)); dr = ImageDraw.Draw(sh)
            for i, nm in enumerate(names):
                key = nm.replace("/", "_").replace(" ", "_")
                if key not in z.files:
                    continue
                lab, gap = z[key][0], z[key][1].astype(bool)
                im = _mask_img(lab, pal)
                arr = np.array(im)
                arr[gap[::-1]] = (255, 0, 0)
                im = Image.fromarray(arr).resize((T, T), Image.NEAREST)
                x, y = (i % cols) * T, (i // cols) * (T + 30)
                sh.paste(im, (x, y + 30))
                dr.text((x + 3, y + 2), nm[:30], fill=(255, 220, 120))
                dr.text((x + 3, y + 15), "%s %s" % tuple(meta[nm]), fill=(200, 200, 200))
            fn = os.path.join(RDIR, "g3_%s_worst_sheet.png" % k); sh.save(fn); made.append(fn)
        # G7: what the rays through the eye slit / breaths meet (grey shell, magenta background, orange bright
        # interior steel, blue dark lining, green face, dark = other pieces)
        f = os.path.join(RDIR, "g7_%s_masks.npz" % k)
        if os.path.exists(f):
            z = np.load(f)
            pal = {0: (22, 22, 26), 1: (150, 150, 150), 2: (255, 0, 200), 3: (255, 150, 30), 4: (40, 70, 200),
                   5: (40, 220, 60), 6: (90, 60, 40)}
            T = 220; cols = 6; names = z.files
            rows = (len(names) + cols - 1) // cols
            sh = Image.new("RGB", (cols * T, rows * (T + 18)), (22, 22, 26)); dr = ImageDraw.Draw(sh)
            for i, nm in enumerate(names):
                x, y = (i % cols) * T, (i // cols) * (T + 18)
                sh.paste(_mask_img(z[nm], pal).resize((T, T), Image.NEAREST), (x, y + 18))
                dr.text((x + 3, y + 3), nm, fill=(255, 220, 120))
            fn = os.path.join(RDIR, "g7_%s_sheet.png" % k); sh.save(fn); made.append(fn)
        # G8 rays: holes per battery camera
        f = os.path.join(RDIR, "g8_%s_masks.npz" % k)
        if os.path.exists(f):
            z = np.load(f)
            keys = sorted({n.split("|")[0] for n in z.files})
            dirs = [dir_name(d) for d in dirs26()]
            for key in keys:
                tgs = []
                for n in z.files:
                    if n.startswith(key + "|"):
                        t = n.split("|")[1]
                        if t not in tgs:
                            tgs.append(t)
                T = 72
                sh = Image.new("RGB", (100 + 26 * T, 18 + len(tgs) * T), (22, 22, 26)); dr = ImageDraw.Draw(sh)
                for i, tg in enumerate(tgs):
                    dr.text((3, 18 + i * T + 28), tg, fill=(255, 220, 120))
                    for j, dn in enumerate(dirs):
                        n = "%s|%s|%s" % (key, tg, dn)
                        if n in z.files:
                            sh.paste(_mask_img(z[n], PAL, flip=False).resize((T, T), Image.NEAREST), (100 + j * T, 18 + i * T))
                for j, dn in enumerate(dirs):
                    dr.text((100 + j * T + 2, 3), dn[:11], fill=(255, 220, 120))
                fn = os.path.join(RDIR, "g8_%s_rays_%s.png" % (k, key)); sh.save(fn); made.append(fn)
    log_("sheets: %d" % len(made))
    return made


# ------------------------------------------------------------------------------------------------ summary / defects
def _load(g, k):
    f = os.path.join(RDIR, "%s_%s.json" % (g, k))
    return json.load(open(f)) if os.path.exists(f) else None


# user feedback round 5 (STATUS.md items 23-32); 0 = the lead sentence ("intersections, clipping, etc.")
USER_ITEMS = {0: "intersections / clipping (general)", 23: "helmet / visor joins, helmet-gorget gaps",
              24: "shield through the arm / elbow / vambrace", 25: "pauldrons bulky, other plates through them",
              26: "tabard through other armour", 27: "belt floats / goes through the tabard",
              28: "one-sided geometry: armour invisible from some angles",
              29: "hand does not hold the sword (fist, finger-plate cage)", 30: "head not inside the helmet",
              31: "chainmail stretching", 32: "floating rings / slivers / spikes / shards, tasset / cuff gaps"}
ARM = ("sword", "shield", "gauntlet_r", "gauntlet_l", "vambrace_r", "vambrace_l", "couter_r", "couter_l",
       "rerebrace_r", "rerebrace_l")
# inner -> outer layer order (for 'which piece emerges' in a pair)
LAYER = {"mail": 0, "underlayer": 0, "aventail": 0, "legs_mail": 0, "boots": 0, "mail_skirt": 1}
ACTION_CLIPS = ("walk", "run", "attack_sword", "block_shield")
BODY_ARMOUR = ("cuirass", "tabard", "belts", "mail_skirt", "tassets", "cape", "legs_mail", "cuisses", "poleyns", "greaves",
               "mail", "gorget", "underlayer", "clasps", "helmet", "plume")
SEAM_NOTES = {"helmet skull-visor": " (includes the authored eye slit between skull and visor: from above / the side it "
                                    "looks into an empty helmet, see G7; plus the visor's side edges)"}


def _owner(p):
    return "ANIM" if p == "underlayer" else owner_of(p)


def _items_for_pieces(ps, gate):
    s = set(ps)
    it = []
    if gate == "G3" and ("helmet" in s):
        it.append(23)
    if gate == "G4" and s == {"helmet", "gorget"}:
        it.append(23)
    if "shield" in s and s & {"vambrace_l", "couter_l", "rerebrace_l", "gauntlet_l", "pauldron_l"}:
        it.append(24)
    if s & {"pauldron_l", "pauldron_r"} and len(s) > 1:
        it.append(25)
    if "tabard" in s and len(s) > 1:
        it.append(26)
    if "belts" in s:
        it.append(27)
    if s == {"gauntlet_r", "sword"}:
        it.append(29)
    if gate == "G3" and s & {"gauntlet_l", "gauntlet_r", "tassets", "mail_skirt"}:
        it.append(32)
    return it or [0]


def _pose_to_viewer(pose):
    """viewer clip / time for a pose name, or None (the extreme test poses exist only in Blender)"""
    if pose == "bind":
        return (None, 0.0)
    if "_f" in pose:
        c, f = pose.rsplit("_f", 1)
        if c in ("idle", "walk", "run", "attack_sword", "block_shield") and f.isdigit():
            return (c, int(f) / 30.0)
    return None


def summary(kinds):
    """baseline numbers per gate + ONE defect list, merged over the bodies: severity, user item, owners (primary
    first), per-body metric, evidence paths and a camera for a viewer close-up (renders/integrity/defects/)"""
    base, found = {}, {}

    def put(key, gate, kind, pieces, owners, sev, items, what, metric, evidence, view=None, score=0.0):
        d = found.setdefault(key, dict(gate=gate, pieces=pieces, owner=list(dict.fromkeys(owners)), severity=sev,
                                       user_items=sorted(set(items)), what=what, bodies=[], metric={}, evidence=[],
                                       view=None, score=0.0))
        d["bodies"].append(kind)
        d["metric"][kind] = metric
        d["evidence"] += [e for e in evidence if e not in d["evidence"]]
        if sev == "major":
            d["severity"] = "major"
        if score > d["score"]:
            d["score"] = score
        if view and d["view"] is None:
            d["view"] = dict(view, body=kind)

    for k in kinds:
        B = base[k] = {}
        R = "renders/integrity/"
        # ---------------- G1
        g1 = _load("g1", k)
        if g1:
            P = g1["pieces"]
            B["G1"] = dict(pieces=len(P), pieces_failing=sum(1 for v in P.values() if v["error"]),
                           open_boundary_edges=sum(v["boundary_edges"] for v in P.values()),
                           one_sided_single_sided_area_m2=round(sum(v["area_m2"] * v["one_sided_single_sided_area_frac"]
                                                                    for v in P.values()), 3),
                           worst_vanish={p: v["vanish_frac_max"] for p, v in sorted(P.items(), key=lambda kv: -kv[1]["vanish_frac_max"])[:10]},
                           culled_but_visible_tris={p: v["culled_visible_tris"] for p, v in g1.get("cull_audit", {}).items()
                                                    if isinstance(v, dict) and v.get("culled_visible_tris")})
            for p, v in P.items():
                if not v["error"]:
                    continue
                items = [28] + ([32] if p.startswith(("vambrace", "gauntlet", "pauldron", "gorget")) else []) + \
                        ([29] if p == "gauntlet_r" else [])
                inv = v.get("inverted_comps") or []
                sev = "major" if v["vanish_frac_max"] >= 0.25 or v["hole_px_max"] >= 500 else "minor"
                put("G1 " + p, "G1", k, [p], [_owner(p)], sev, items,
                    "open single-sided shell: no inner faces / thickness, so from inside or behind (looking up a sleeve, "
                    "down a collar, under a rim) the surface is not drawn and the piece vanishes"
                    + ("; %d component(s) wound inward (normals face the body)" % len(inv) if inv else ""),
                    "one-sided %.0f%% of its area, vanishes on up to %.0f%% of its pixels (%s), %d background-hole px, "
                    "%d open edges in %d open components%s" % (
                        100 * v["one_sided_single_sided_area_frac"], 100 * v["vanish_frac_max"], v["vanish_worst_dir"],
                        v["hole_px_max"], v["boundary_edges"], v["open_comps"],
                        (", inward-wound: " + ", ".join("#%d %s %.1f cm2" % (c["comp"], c["bone"], c["area_cm2"]) for c in inv[:3])) if inv else ""),
                    [R + "g1_%s.json" % k, R + "g1_%s_vanish_sheet.png" % k], score=v["vanish_frac_max"] * 100)
            for p, v in g1.get("cull_audit", {}).items():
                if isinstance(v, dict) and v.get("culled_visible_tris"):
                    sev = "major" if v["culled_visible_tris"] >= 20 else "minor"
                    put("G1 cull " + p, "G1", k, [p], ["ANIM", _owner(p)], sev, [28],
                        "the assembly cull (build_knight.cull_hidden) deleted faces a camera outside can see: its rays "
                        "treat single-sided back faces (not drawn by the viewer) as occluders",
                        "%d of %d culled tris visible at bind (%s)" % (v["culled_visible_tris"], v["culled_tris"],
                                                                       ", ".join(v["where"][:4])),
                        [R + "g1_%s.json (cull_audit)" % k], score=v["culled_visible_tris"] / 5)
        # ---------------- G2
        g2 = _load("g2", k)
        if g2:
            C = g2["components"]
            B["G2"] = dict(components_checked=len(C), visible_errors=sum(1 for c in C if c["error"]),
                           floating_bind=sum(1 for c in C if c["floating"] == "bind" and c["visible"]),
                           floating_motion=sum(1 for c in C if c["floating"] == "motion" and c["visible"]),
                           partly_buried=sum(1 for c in C if c["partly_buried"] and c["visible"]),
                           visible_spike_tris=sum(v["visible"] for v in g2["shards"].values()),
                           slivers=len(g2["slivers"]), visible_slivers=sum(1 for s in g2["slivers"] if s["visible"]))
            per = {}
            for c in C:
                if c["error"]:
                    per.setdefault(c["piece"], []).append(c)
            for p, cs in per.items():
                fb = [c for c in cs if c["floating"] == "bind"]; fm = [c for c in cs if c["floating"] == "motion"]
                pb = [c for c in cs if c["partly_buried"]]
                parts = []
                if fb:
                    parts.append("%d detached at bind (up to %.1f mm off; %s)" % (
                        len(fb), max(c["bind_gap_mm"] for c in fb), ", ".join(sorted({c["bone"] for c in fb})[:4])))
                if fm:
                    w = max(fm, key=lambda c: c["max_gap_mm"])
                    parts.append("%d lift off in motion (up to %.1f mm at %s)" % (len(fm), w["max_gap_mm"], w["max_gap_at"]))
                if pb:
                    parts.append("%d partly buried in the plate under them (a sliver / spindle shows)" % len(pb))
                gap = max([c["bind_gap_mm"] for c in fb] + [c["max_gap_mm"] for c in fm] + [0])
                sev = "major" if gap > 2.0 or len(pb) >= 4 else "minor"
                vw = next((c["view"] for c in sorted(cs, key=lambda c: -max(c["bind_gap_mm"], c["max_gap_mm"])) if c.get("view")), None)
                put("G2 " + p, "G2", k, [p], [_owner(p)] + (["ANIM"] if fm and not fb else []), sev,
                    [27] if p == "belts" else [32],
                    "floating / detached / partly buried parts (trims, rings, rivets, ribs, plates on their support)",
                    "; ".join(parts), [R + "g2_%s.json" % k],
                    view=dict(pose="bind", x=vw[0], d=vw[1]) if vw else None, score=gap + len(pb))
            for p, v in g2["shards"].items():
                if v["visible"]:
                    sev = "major" if v["visible"] >= 20 else "minor"
                    vw = dict(pose="bind", x=v["points"][0], d=v["dirs"][0]) if v.get("points") and v.get("dirs") else None
                    put("G2 spikes " + p, "G2", k, [p], [_owner(p)], sev, [32],
                        "needle triangles folded off the surface (shard / spike look at rims and corners)",
                        "%d visible spike tris (%s)" % (v["visible"], ", ".join(v["where"][:4])),
                        [R + "g2_%s.json (shards)" % k], view=vw, score=v["visible"] / 10)
        # ---------------- G3
        g3 = _load("g3", k)
        if g3:
            S3 = g3["seams"]
            B["G3"] = dict(seams=len(S3), errors=sum(1 for v in S3.values() if v.get("verdict") == "error"),
                           warnings=sum(1 for v in S3.values() if v.get("verdict") == "warning"),
                           ok=sum(1 for v in S3.values() if v.get("verdict") == "ok"),
                           missing=[n for n, v in S3.items() if v.get("missing")],
                           worst_mm2={n: v.get("gap_mm2") for n, v in sorted(S3.items(), key=lambda kv: -kv[1].get("max_gap_px", 0))[:10]})
            for n, v in S3.items():
                if v.get("verdict") != "error":
                    continue
                pa, pb = v["a"].split(":")[0], v["b"].split(":")[0]
                sev = "major" if v["gap_mm2"] >= 300 or v["see"]["background"] >= 10 else "minor"
                cw = v.get("clip_worst")
                vw = (dict(pose=cw["at"], x=cw["centre"], d=cw["dir_vec"]) if cw else
                      dict(pose=v["at"], x=v["worst_centre"], d=v["worst_dir_vec"]) if v.get("worst_dir_vec") else None)
                put("G3 " + n, "G3", k, sorted({pa, pb}), [_owner(pa), _owner(pb)], sev,
                    _items_for_pieces([pa, pb], "G3"), "see-through gap at the seam %s%s" % (n, SEAM_NOTES.get(n, "")),
                    "%.0f mm2 see-through between the two (%d px background, %d px armour interior%s) at %s from %s; "
                    "%d of %d poses have a gap" % (
                        v["gap_mm2"], v["see"]["background"], v["see"]["interior"],
                        (": " + ", ".join(v.get("see_pieces", [])[:3])) if v.get("see_pieces") else "",
                        v["at"], v["dir"], v["frames_with_gap"], g3["poses"]),
                    [R + "g3_%s.json" % k, R + "g3_%s_worst_sheet.png" % k], view=vw, score=v["gap_mm2"] / 50)
        # ---------------- G4
        g4 = _load("g4", k)
        if g4:
            PR = g4["pairs"]
            B["G4"] = dict(poses=g4["poses"], pairs_intersecting=len(PR),
                           visible_penetration_pairs=sum(1 for v in PR.values() if v["verdict"] == "error"),
                           visible_in_clips=sum(1 for v in PR.values() if v["verdict"] == "error" and v.get("seen_in") == "clips"),
                           extreme_poses_only=sum(1 for v in PR.values() if v["verdict"] == "error" and v.get("seen_in") == "extreme poses only"),
                           hidden_or_seated_pairs=sum(1 for v in PR.values() if v["verdict"] == "warning"),
                           intended_pairs=sum(1 for v in PR.values() if v["verdict"] == "intended"),
                           at_bind=sum(1 for v in PR.values() if v["verdict"] == "error" and
                                       v.get("per_group", {}).get("bind", {}).get("max_visible_deep", 0) >= 2))
            grp = {}
            for key, v in PR.items():
                if v["verdict"] != "error":
                    continue
                ps = sorted({s_.split("#")[0] for s_ in key.split("|")})
                gk = "|".join(ps) + (" (self)" if len(ps) == 1 else "")
                g = grp.setdefault(gk, dict(pieces=ps, vis=0, vis_clips=0, depth=0.0, at=None, where=set(), per={},
                                            view=None, self=len(ps) == 1, keys=0))
                g["keys"] += 1
                if v["max_visible_deep"] > g["vis"]:
                    g["vis"], g["at"] = v["max_visible_deep"], v["deep_at"]
                    g["depth"] = v.get("visible_deep_depth_mm") or 0
                    if v.get("visible_points") and v.get("visible_dirs"):
                        g["view"] = dict(pose=v["deep_at"], x=v["visible_points"][0], d=v["visible_dirs"][0])
                g["vis_clips"] = max(g["vis_clips"], v.get("max_visible_deep_clips", 0))
                for gg, q in v.get("per_group", {}).items():   # a camera on the worst bind / clip frame (viewer-renderable)
                    if gg != "extreme" and q.get("point") and q["max_visible_deep"] > g.get("clip_vis", 0):
                        g["clip_vis"] = q["max_visible_deep"]
                        g["clip_view"] = dict(pose=q["at"], x=q["point"], d=q["dir"])
                g["where"] |= set(v["where"])
                for gg, q in v.get("per_group", {}).items():
                    o = g["per"].setdefault(gg, [0, 0, 0])
                    o[0] = max(o[0], q["max_visible_deep"]); o[1] = max(o[1], q["frames_visible_deep"]); o[2] = max(o[2], q["frames"])
            for gk, g in grp.items():
                ps = g["pieces"]
                own = [_owner(p) for p in ps]
                per = g["per"]
                clip_groups = [c for c, q in per.items() if c != "extreme" and q[0] >= 2]
                only_action = clip_groups and all(c in ACTION_CLIPS for c in clip_groups)
                if only_action and any(p in ARM for p in ps) and any(p in BODY_ARMOUR for p in ps):
                    own = ["ANIM"] + own                        # the clip path puts an arm / prop through the body armour
                elif only_action and any(p in ARM for p in ps):
                    own = own + ["ANIM"]                        # arm pieces among themselves / pauldrons: fit first
                sev = "major" if g["vis_clips"] >= 20 else "minor"
                seen = ", ".join("%s %d pts %d/%d frames" % (c, q[0], q[1], q[2]) for c, q in
                                 sorted(per.items(), key=lambda kv: (kv[0] == "extreme", kv[0])) if q[0] >= 2)
                inner = min(ps, key=lambda p: LAYER.get(p, 2)) if not g["self"] and len({LAYER.get(p, 2) for p in ps}) > 1 else None
                put("G4 " + gk, "G4", k, ps, own, sev, _items_for_pieces(ps, "G4"),
                    "visible penetration %s%s" % (gk, (" (%s emerges through the outer layer)" % inner) if inner else ""),
                    "worst %d visible intersection points (3 mm clusters) reaching up to %.1f mm behind the other surface "
                    "at %s (%s); seen in: %s" % (g["vis"], g["depth"], g["at"], ", ".join(sorted(g["where"])[:4]),
                                                 seen or "-") + ("" if clip_groups else " [extreme test poses only]"),
                    [R + "g4_%s.json" % k], view=g.get("clip_view") or g["view"],
                    score=g["vis_clips"] * max(1.0, min(g["depth"], 30) / 5))
        # ---------------- G5
        g5 = _load("g5", k)
        if g5:
            B["G5"] = {c: {g: "%d/%d frames fail" % (v["fail_frames"], v["frames"]) for g, v in d.items()}
                       for c, d in g5["clips"].items()}
            for chk in ("sword_grip", "shield"):
                rows = [(c, d[chk]) for c, d in g5["clips"].items() if chk in d and d[chk]["fail_frames"]]
                if not rows:
                    continue
                c_w, v_w = max(rows, key=lambda r: r[1]["fail_frames"] / max(r[1]["frames"], 1))
                w = v_w["worst"]
                clips = ", ".join("%s %d/%d" % (c, v["fail_frames"], v["frames"]) for c, v in rows)
                if chk == "sword_grip":
                    fing = ", ".join("%s %s" % (f, x["inside_mm"]) for f, x in w["fingers"].items() if x.get("inside_mm"))
                    put("G5 sword grip", "G5", k, ["gauntlet_r", "sword"], ["UPPER", "LOWER"], "major", [29],
                        "the sword is not held: the fist closes through the grip and the finger plates stand off the "
                        "glove (the 'wire cage'); grip pose = armour_upper.post_clips, grip / socket = the sword",
                        "fails in %s; worst %s: %d hand tris through the grip wrap, fingers inside the grip (mm) %s, "
                        "%d of %d finger plates' inner side > 2 mm off the glove (p20 %s mm), grip radius %.1f mm, "
                        "enclosure %.2f" % (clips, v_w["worst_pose"],
                                            w["hand_tris_through_grip"], fing or "-", w.get("plates_floating", w["plates_off_glove_gt2mm"]),
                                            w.get("plates", 0), w.get("plate_standoff_p20_mm", "?"), w["grip_radius_mm"], w["enclosure"]),
                        [R + "g5_%s.json" % k, R + "g8_%s_user_sword_hand_top.png" % k], score=50)
                else:
                    put("G5 shield carry", "G5", k, ["shield", "vambrace_l", "gauntlet_l"], ["LOWER", "UPPER"], "major", [24],
                        "the shield is not carried by its straps / hand grip: the enarmes loops cut the vambrace and the "
                        "left fist closes on nothing (no hand strap / handle at socket_hand_l)",
                        "fails in %s; worst %s: %d strap tris through the vambrace, left fist %s mm from the nearest shield "
                        "surface, %d loops round the forearm" % (
                            clips, v_w["worst_pose"], w["strap_tris_through_vambrace"], w["hand_socket_to_shield_mm"],
                            w["loops_enclosing"]),
                        [R + "g5_%s.json" % k, R + "g8_%s_user_shield_elbow.png" % k], score=40)
        # ---------------- G6
        g6 = _load("g6", k)
        if g6:
            B["G6"] = {}
            for p, v in g6["pieces"].items():
                for s_, r in v["sets"].items():
                    if r["error"]:
                        B["G6"].setdefault(p, {})[s_] = dict(uv_out_vis=r["uv_out_visible_frac"],
                                                             deform_out_vis=r["deform_out_visible_frac"],
                                                             streak_cm2=r["uv_streak_visible_cm2"])
                    if not r["error"]:
                        continue
                    mail = s_ == "mail"
                    sev = "major" if mail or r["uv_streak_visible_cm2"] >= 50 else "minor"
                    hot = r.get("uv_hot") or r.get("deform_hot") or {}
                    put("G6 %s %s" % (p, s_), "G6", k, [p], [_owner(p)], sev, [31] if mail else [0],
                        "texture stretch on %s (%s)" % (p, "chain mail rings" if mail else s_),
                        "UV out of tolerance on %.0f%% of the visible area%s; skinning stretch out on %.0f%% (max %.2f / "
                        "min %.2f); %.1f cm2 of visible streaks (anisotropy > 4); at %s" % (
                            100 * r["uv_out_visible_frac"],
                            (" (ring scale %.2f x the 10.5 mm tile)" % r["ring_scale_vs_nominal"]) if "ring_scale_vs_nominal" in r else "",
                            100 * r["deform_out_visible_frac"], r["deform_stretch_max"], r["deform_squash_min"],
                            r["uv_streak_visible_cm2"], ", ".join(list(hot)[:4])),
                        [R + "g6_%s.json" % k] + ([R + "g8_%s_user_mail_top.png" % k] if p == "mail" else []),
                        score=(30 if mail else 0) + r["uv_streak_visible_cm2"] / 10)
        # ---------------- G7
        g7 = _load("g7", k)
        if g7 and "total_px" in g7:
            t = g7["total_px"]
            B["G7"] = dict(t, px_mm=g7.get("px_mm"), head_in_helm_look=g7.get("head_present_in_look"))
            if t["background"] or t["bright_interior"]:
                put("G7 head in helmet", "G7", k, ["helmet"], ["UPPER", "ANIM"], "major", [30],
                    "no head inside the helmet: the helm look ships no head / face, and the lining does not close the "
                    "eye slit and breaths, so from the side they show the background or bright inner steel",
                    "rays in through the slit / breaths: %d px background, %d px bright interior, %d px dark lining, "
                    "%d px face (%.1f mm px, %d directions)" % (t["background"], t["bright_interior"], t["dark_lining"],
                                                                t["face"], g7["px_mm"], len(g7["views"])),
                    [R + "g7_%s.json" % k, R + "g7_%s_sheet.png" % k, R + "g8_%s_user_pauldron_helm.png" % k], score=45)
        # ---------------- G8 (in context: the battery cameras ray-traced with the viewer's culling rule)
        g8 = _load("g8", k)
        if g8:
            w = {}
            for key, tg in g8["rays"].items():
                for t_, rows in tg.items():
                    for r in rows:
                        if r["vanish_frac"] > w.get(t_, {}).get("frac", 0):
                            w[t_] = dict(frac=round(r["vanish_frac"], 3), at=key, dir=r["dir"], hole_px=r["hole_px"],
                                         pieces=r.get("vanish_pieces", {}))
            B["G8"] = dict(cameras=sum(len(r) for tg in g8["rays"].values() for r in tg.values()),
                           worst_vanish_per_target=w)
            for t_, v in w.items():
                if v["frac"] < 0.05:
                    continue
                ps = list(v["pieces"])[:3] or ["?"]
                put("G8 " + t_, "G8", k, ps, [_owner(p) for p in ps if p != "?"], "major" if v["frac"] >= 0.10 else "minor",
                    [28], "in context (orbit battery round the %s): surfaces the viewer does not draw" % t_,
                    "%.0f%% of the covered pixels are back faces of single-sided plates (%d show the background) at %s "
                    "from %s; vanishing: %s" % (100 * v["frac"], v["hole_px"], v["at"], v["dir"],
                                                ", ".join("%s %d px" % kv for kv in v["pieces"].items())),
                    [R + "g8_%s.json" % k, R + "g8_%s_rays_%s.png" % (k, v["at"]), R + "g8_%s_battery_%s.jpg" % (k, v["at"])],
                    score=v["frac"] * 100)
    # ---- order, ids, owner rollup
    gates = ("G1", "G2", "G3", "G4", "G5", "G6", "G7", "G8")
    defects = sorted(found.values(), key=lambda d: (gates.index(d["gate"]), d["severity"] != "major", -len(d["bodies"]),
                                                    -d["score"]))
    n_ = {}
    for d in defects:
        n_[d["gate"]] = n_.get(d["gate"], 0) + 1
        d["id"] = "%s-%02d" % (d["gate"], n_[d["gate"]])
        d["owner"] = [o for o in dict.fromkeys(d["owner"]) if o]
        d["primary_owner"] = d["owner"][0] if d["owner"] else "?"
        d["bodies"] = sorted(set(d["bodies"]))
        d["score"] = round(d["score"], 1)
    rep = dict(generated=time.strftime("%Y-%m-%d %H:%M"), bodies=list(base), baseline=base,
               user_items=USER_ITEMS,
               counts={g: sum(1 for d in defects if d["gate"] == g) for g in ("G1", "G2", "G3", "G4", "G5", "G6", "G7", "G8")},
               major=sum(1 for d in defects if d["severity"] == "major"),
               by_primary_owner={o: sum(1 for d in defects if d["primary_owner"] == o) for o in ("UPPER", "LOWER", "ANIM")},
               by_owner_any={o: sum(1 for d in defects if o in d["owner"]) for o in ("UPPER", "LOWER", "ANIM")},
               by_user_item={i: [d["id"] for d in defects if i in d["user_items"]] for i in USER_ITEMS},
               defects=defects)
    json.dump(rep, open(os.path.join(RDIR, "summary.json"), "w"), indent=1)
    json.dump(defects, open(os.path.join(RDIR, "defects.json"), "w"), indent=1)
    log_("summary: %d defects (%d major) %s, primary owner %s -> renders/integrity/summary.json, defects.json"
         % (len(defects), rep["major"], rep["counts"], rep["by_primary_owner"]))
    return rep


DEFECT_SHOT_QUOTA = {"G2": 16, "G3": 36, "G4": 36}


def defect_shots(kinds, quota=None, size=520, dist=0.42):
    """a viewer close-up of each defect that has a camera (G2 / G3 / G4: a direction from which the spot is visible,
    pose = the worst bind / clip frame; extreme test poses exist only in Blender) -> renders/integrity/defects/ +
    labelled contact sheets renders/integrity/defects_sheet_<gate>.png"""
    from PIL import Image, ImageDraw
    rep = json.load(open(os.path.join(RDIR, "summary.json")))
    ddir = os.path.join(RDIR, "defects")
    os.makedirs(ddir, exist_ok=True)
    for f in os.listdir(ddir):
        if f.endswith(".png"):
            os.remove(os.path.join(ddir, f))
    quota = quota or DEFECT_SHOT_QUOTA
    per_kind = {k: [] for k in kinds}
    chosen = []
    taken = {}
    for d in rep["defects"]:
        v = d.get("view")
        if not v or v.get("body") not in per_kind or taken.get(d["gate"], 0) >= quota.get(d["gate"], 0):
            continue
        pv = _pose_to_viewer(v["pose"])
        if pv is None:
            continue
        x, dv = v["x"], list(v["d"])
        if abs(dv[2]) > 0.97:                                   # straight up / down: tilt so the viewer's up is defined
            dv = [dv[0] + 0.12, dv[1] - 0.12, dv[2]]
        n = math.sqrt(sum(c * c for c in dv)); dv = [c / n for c in dv]
        cam = [x[i] + dv[i] * dist for i in range(3)]
        g = lambda q: [round(q[0], 4), round(q[2], 4), round(-q[1], 4)]      # Blender -> glTF
        sp = {"out": os.path.join(ddir, "%s_%s.png" % (d["id"], v["body"])), "view": {"p": g(cam), "t": g(x), "fov": 30},
              "scene": "town", "clip": pv[0]}
        if pv[0]:
            sp["time"] = pv[1]
        per_kind[v["body"]].append(sp)
        d["shot"] = os.path.relpath(sp["out"], CH)
        chosen.append(d)
        taken[d["gate"]] = taken.get(d["gate"], 0) + 1
    for k, specs in per_kind.items():
        if specs:
            shots(os.path.join(CH, "viewer", "knight_%s_lookdev.html" % k), specs, size, size, "defects_" + k)
    # one sheet per gate (the defect spot is at the centre of each close-up)
    for gate in sorted({d["gate"] for d in chosen}):
        _defect_sheet([d for d in chosen if d["gate"] == gate], os.path.join(RDIR, "defects_sheet_%s.png" % gate))
    json.dump(rep, open(os.path.join(RDIR, "summary.json"), "w"), indent=1)
    json.dump(rep["defects"], open(os.path.join(RDIR, "defects.json"), "w"), indent=1)
    log_("defect shots: %d %s -> renders/integrity/defects/, defects_sheet_<gate>.png" % (len(chosen), taken))
    return chosen


def _defect_sheet(chosen, fn):
    from PIL import Image, ImageDraw
    T = 300; cols = 6
    rows = (len(chosen) + cols - 1) // cols
    sh = Image.new("RGB", (cols * T, max(1, rows) * (T + 44)), (22, 22, 26)); dr = ImageDraw.Draw(sh)
    for i, d in enumerate(chosen):
        f = os.path.join(CH, d["shot"])
        if not os.path.exists(f):
            continue
        im = Image.open(f).convert("RGB").resize((T, T), Image.LANCZOS)
        x0, y0 = (i % cols) * T, (i // cols) * (T + 44)
        sh.paste(im, (x0, y0 + 44))
        # the defect spot is at the image centre
        dr.ellipse((x0 + T // 2 - 14, y0 + 44 + T // 2 - 14, x0 + T // 2 + 14, y0 + 44 + T // 2 + 14), outline=(255, 40, 40), width=2)
        dr.text((x0 + 4, y0 + 3), "%s %s %s [%s]" % (d["id"], d["gate"], d["severity"], "/".join(d["owner"])), fill=(255, 220, 120))
        dr.text((x0 + 4, y0 + 16), d["what"][:52], fill=(220, 220, 220))
        dr.text((x0 + 4, y0 + 29), "%s %s" % (d["view"]["body"], d["view"]["pose"]), fill=(170, 170, 170))
    sh.save(fn)


ALL = ["g1", "g2", "g3", "g4", "g5", "g6", "g7", "g8"]


def run_stage(kinds, what=None, step=3):
    """build_knight.py 'integrity' stage: the Blender gates (both bodies in parallel), the viewer battery, sheets,
    summary"""
    what = what or (ALL + ["viewer", "sheets", "summary", "defects"])
    os.makedirs(RDIR, exist_ok=True)
    t0 = time.time()
    gates = [g for g in what if g in ALL]
    if gates:
        run_blender(kinds, gates, step=step)
    if "viewer" in what:
        viewer_battery(kinds)
    if "sheets" in what:
        sheets(kinds)
    rep = summary(kinds) if "summary" in what else None
    if "defects" in what:
        defect_shots(kinds)
    log_("integrity stage done in %.1f min" % ((time.time() - t0) / 60))
    return rep


def main():
    argv = sys.argv[1:]
    kinds = ["male", "female"]
    if "--kinds" in argv:
        i = argv.index("--kinds"); kinds = []
        for a in argv[i + 1:]:
            if a.startswith("-") or a in ALL + ["viewer", "sheets", "summary", "defects"] or "=" in a:
                break
            kinds.append(a)
        argv = argv[:i] + argv[i + 1 + len(kinds):]
    step = int(next((a.split("=")[1] for a in argv if a.startswith("step=")), 3))
    what = [a for a in argv if "=" not in a] or None
    run_stage(kinds, what, step)


if __name__ == "__main__":
    if IN_BLENDER:
        a = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
        blender_main(a)
    else:
        main()
