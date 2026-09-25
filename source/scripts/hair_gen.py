"""Procedural hair-card grooms, written as MPFB / MakeHuman clothes assets (.mhclo + .obj + .mhmat) so they fit
any body shape, get skin weights interpolated from the rts_human rig and follow the face shape keys (hairline).

Styles
  rts_crop  (male)   short textured crop combed back: dense base layer + mid layer with lift + wispy top layer +
                     soft hairline fringe cards. ~600 cards, 4 segments each.
  rts_braid (female) hair pulled back over the scalp to a tie at the occiput, a three-strand braid down the back
                     (woven tubes), a leather tie and a short loose tail. Cards follow the scalp to the tie point.
  rts_beard_short    (male, customisation part) trimmed full beard + moustache: three card layers (opaque base,
                     dense, wispy) grown over a beard mask (front of the ears, below a cheek line from the sideburns
                     to the mouth corners and up to the nose base, down to the upper throat, lips excluded)
  rts_beard_full     (male, 'dwarf') the short beard, longer, plus free-hanging long cards from the chin / front jaw
                     (up to ~27 cm) kept clear of the neck and chest, and a long drooping moustache
  rts_cap_crop / rts_cap_braid / rts_cap_generic   scalp CAP shells (skin-hugging, 0.7 mm off the scalp, LSCM
                     unwrap) with a painted strand texture along each groom's comb flow (opaque interior, strand-broken
                     hairline): joined under every hair part by hair_lib.join_caps (user round-2 item 11)
  rts_stubble        (male) skin-hugging shell over the beard mask (body faces, 0.8 mm off the skin, LSCM unwrap)
                     with an alpha-blended stubble texture painted in 3D (beard_stubble_base.png); the same shell with
                     the dense beard_shadow_base.png is joined under the card beards by cust_lib.merge_beard_shadows

Every card is a quad strip lying along the scalp (width across the flow), UV-mapped to one column of the
procedural strand atlas (scripts/hair_tex.py): root at v = 1, tip at v = 0. Braid tubes map into the atlas'
opaque 'solid' column.

The groom is authored on the target body (build_human) and converted with MPFB MakeClothes
(ClothesService.create_mhclo_from_clothes_matching + Mhclo.write_mhclo), then installed into MPFB's user data
hair/ folder (and kept in assets/mpfb_assets/hair/). base_humans.py loads it like any MakeHuman hair.

run: Blender -b --python-exit-code 1 -P scripts/hair_gen.py -- crop braid beard_short beard_full stubble cap_crop cap_braid
     cap_generic   (after make_rig.py;
     before base_humans.py; no style = all)
"""
import sys, os, shutil, uuid, time, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from chr_lib import *
import bmesh
from mathutils.bvhtree import BVHTree
from mathutils.kdtree import KDTree
from bl_ext.user_default.mpfb.services import ClothesService
from bl_ext.user_default.mpfb.services.objectservice import ObjectService as OS_

args = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
STYLE_KIND = {"crop": "male", "braid": "female", "beard_short": "male", "beard_full": "male", "stubble": "male",
              "cap_crop": "male", "cap_braid": "female", "cap_generic": "male"}
styles = [a for a in args if a in STYLE_KIND] or list(STYLE_KIND)
COLS = 8
ASSET_DIR = os.path.join(ASSETS, "mpfb_assets", "hair")


# ------------------------------------------------------------------------------------------------ head surface
class Head:
    def __init__(self, rig, bm):
        me = bm.data
        masks = [m for m in bm.modifiers if m.type == 'MASK']
        for m in masks:
            m.show_viewport = False
        dg = bpy.context.evaluated_depsgraph_get()
        em = bm.evaluated_get(dg).to_mesh()
        self.co = np.array([v.co[:] for v in em.vertices])
        self.no = np.array([v.normal[:] for v in em.vertices])
        tris = []
        self.polys = []                                  # body polygons (vertex lists), used by the stubble shell
        for p in em.polygons:
            vs = list(p.vertices)
            if max(vs) >= 13380:
                continue
            self.polys.append(vs)
            for k in range(1, len(vs) - 1):
                tris.append((vs[0], vs[k], vs[k + 1]))
        bm.evaluated_get(dg).to_mesh_clear()
        for m in masks:
            m.show_viewport = True
        self.tris = np.array(tris)
        self.bvh = BVHTree.FromPolygons([Vector(c) for c in self.co], [tuple(t) for t in self.tris])
        gi = bm.vertex_groups["scalp"].index
        self.scalp = np.zeros(len(self.co))
        for v in me.vertices:
            for g in v.groups:
                if g.group == gi:
                    self.scalp[v.index] = g.weight
        self.scalp = smooth_scalp(me, self.scalp)          # same smoothed hairline as the scalp tint (bake_skin)
        self.head = np.array(rig.pose.bones["head"].head[:])
        self.top = float(self.co[:13380, 2].max())
        sc = self.co[self.scalp > 0.5]
        self.scalp_center = sc.mean(0)
        self.eye = np.array((rig.pose.bones["eye_l"].head + rig.pose.bones["eye_r"].head)[:]) / 2

    def closest(self, p):
        loc, nrm, fi, d = self.bvh.find_nearest(Vector(p))
        t = self.tris[fi]
        w = barycentric(np.array(loc[:]), self.co[t])
        n = (self.no[t] * w[:, None]).sum(0); n /= np.linalg.norm(n)
        s = float((self.scalp[t] * w).sum())
        return np.array(loc[:]), n, s

    def sample(self, count, min_dist, accept, seed):
        """Area-weighted random points on scalp triangles, Poisson-thinned; accept(p, n, scalp_w) filters."""
        rng = np.random.default_rng(seed)
        t = self.tris[(self.scalp[self.tris] > 0.05).any(1)]
        a = self.co[t]
        area = 0.5 * np.linalg.norm(np.cross(a[:, 1] - a[:, 0], a[:, 2] - a[:, 0]), axis=1)
        pick = rng.choice(len(t), size=count * 12, p=area / area.sum())
        pts = []
        kd = KDTree(count * 12); n_in = 0
        for i in pick:
            r1, r2 = rng.random(), rng.random()
            if r1 + r2 > 1:
                r1, r2 = 1 - r1, 1 - r2
            w = np.array([1 - r1 - r2, r1, r2])
            p = (a[i] * w[:, None]).sum(0)
            s = float((self.scalp[t[i]] * w).sum())
            n = (self.no[t[i]] * w[:, None]).sum(0); n /= np.linalg.norm(n)
            if not accept(p, n, s):
                continue
            if n_in and kd.find(Vector(p))[2] < min_dist:
                continue
            pts.append((p, n, s)); kd.insert(Vector(p), n_in); n_in += 1; kd.balance()
            if len(pts) >= count:
                break
        return pts


def barycentric(p, tri):
    a, b, c = tri
    v0, v1, v2 = b - a, c - a, p - a
    d00, d01, d11 = v0 @ v0, v0 @ v1, v1 @ v1
    d20, d21 = v2 @ v0, v2 @ v1
    den = d00 * d11 - d01 * d01
    if abs(den) < 1e-14:
        return np.array([1.0, 0, 0])
    v = (d11 * d20 - d01 * d21) / den; w = (d00 * d21 - d01 * d20) / den
    return np.clip(np.array([1 - v - w, v, w]), 0, 1)


def tangent(d, n):
    t = d - n * (d @ n)
    ln = np.linalg.norm(t)
    return t / ln if ln > 1e-9 else np.cross(n, [1, 0, 0])


def rot_about(v, axis, ang):
    axis = axis / np.linalg.norm(axis)
    return v * np.cos(ang) + np.cross(axis, v) * np.sin(ang) + axis * (axis @ v) * (1 - np.cos(ang))


# ------------------------------------------------------------------------------------------------ geometry builder
class Groom:
    def __init__(self):
        self.V = []; self.F = []; self.UV = []

    def card(self, pts, sides, widths, col, v_range=(0.985, 0.02)):
        """Quad strip: pts (K,3) centre line, sides (K,3) unit side vectors, widths (K,)."""
        base = len(self.V)
        K = len(pts)
        u0 = col / COLS + 0.004; u1 = (col + 1) / COLS - 0.004
        for k in range(K):
            t = k / (K - 1)
            v = v_range[0] + (v_range[1] - v_range[0]) * t
            self.V.append(pts[k] - sides[k] * widths[k] / 2); self.UV.append((u0, v))
            self.V.append(pts[k] + sides[k] * widths[k] / 2); self.UV.append((u1, v))
        for k in range(K - 1):
            a = base + 2 * k
            self.F.append((a, a + 2, a + 3, a + 1))     # counter-clockwise seen from outside: normal = +scalp normal

    def tube(self, rings, col, v_range, u_span=1.0):
        """Closed tube from a list of rings (each (S,3)); UV: U around within the column, V along."""
        base = len(self.V)
        R = len(rings); S = len(rings[0])
        u0 = col / COLS + 0.01; u1 = u0 + (1 / COLS - 0.02) * u_span
        for r in range(R):
            v = v_range[0] + (v_range[1] - v_range[0]) * r / (R - 1)
            for s in range(S + 1):                       # duplicate the seam column for clean UVs
                self.V.append(rings[r][s % S]); self.UV.append((u0 + (u1 - u0) * s / S, v))
        for r in range(R - 1):
            for s in range(S):
                a = base + r * (S + 1) + s; b = a + S + 1
                self.F.append((a, b, b + 1, a + 1))

    def to_object(self, name):
        me = bpy.data.meshes.new(name)
        me.from_pydata([tuple(v) for v in self.V], [], self.F)
        uvl = me.uv_layers.new(name="UVMap")
        for poly in me.polygons:
            for li in poly.loop_indices:
                uvl.data[li].uv = self.UV[me.loops[li].vertex_index]
        me.validate(); me.update()
        ob = bpy.data.objects.new(name, me)
        bpy.context.scene.collection.objects.link(ob)
        for p in me.polygons:
            p.use_smooth = True
        return ob


def walk(head, p0, n0, dirfn, length, K, offset, lift_to=None):
    """Centre line of a card: K points starting at p0 following dirfn over the head surface at offset(t)."""
    pts = []; nrms = []
    p = p0.copy(); n = n0.copy()
    d = tangent(dirfn(p, n), n)
    step = length / (K - 1)
    for k in range(K):
        t = k / (K - 1)
        q, n, _ = head.closest(p)
        pts.append(q + n * offset(t)); nrms.append(n)
        d = tangent(0.6 * dirfn(q, n) + 0.4 * d, n)
        p = q + d * step
    return np.array(pts), np.array(nrms)


def scalp_len(head, pts, smin, overhang=0.004):
    """Arc length along a card's centre line until the scalp weight under it drops below smin (+ overhang), or None
    if it never does: cards grown near the hairline / sideburns are cut there instead of hanging past the hairline as
    square blocks (user item 11)."""
    L = 0.0
    for k in range(1, len(pts)):
        L += float(np.linalg.norm(pts[k] - pts[k - 1]))
        if head.closest(pts[k])[2] < smin:
            return L + overhang
    return None


def sides_for(pts, nrms, twist):
    S = []
    for k in range(len(pts)):
        d = pts[min(k + 1, len(pts) - 1)] - pts[max(k - 1, 0)]
        d /= max(np.linalg.norm(d), 1e-9)
        s = np.cross(nrms[k], d); s /= max(np.linalg.norm(s), 1e-9)
        S.append(rot_about(s, d, twist))
    return np.array(S)


# ------------------------------------------------------------------------------------------------ styles
def crop_flow(head):
    """Comb direction of the crop (and of the generic scalp cap under the CC0 styles): back over the top, down at the
    sides and back of the head."""
    top = head.top; hc = head.scalp_center

    def flow(p, n):
        r = p - hc
        down = np.clip((top - 0.035 - p[2]) / 0.09, 0, 1)            # 0 on top of the head, 1 low on the sides/back
        back = np.clip((p[1] - hc[1]) / 0.08, 0, 1)                  # 0 front half, 1 back of the head
        d = np.array([0.25 * np.sign(r[0]) * min(1, abs(r[0]) / 0.06), 1.0 - 0.6 * back, -0.35 - 1.1 * down - 0.8 * back])
        return d / np.linalg.norm(d)
    return flow


def braid_tie(head):
    hc = head.scalp_center
    tq, tn, _ = head.closest(hc + np.array([0.0, 0.2, -0.045]))    # back of the skull, below the scalp centre
    return tq + tn * 0.012


def braid_flow(head):
    tie = braid_tie(head)

    def flow(p, n):
        d = tie - p
        return d / max(np.linalg.norm(d), 1e-9)
    return flow


def style_crop(head, seed=3):
    rng = np.random.default_rng(seed)
    g = Groom()
    top = head.top; hc = head.scalp_center
    flow = crop_flow(head)

    def length_at(p, lo, hi):
        down = np.clip((top - 0.03 - p[2]) / 0.1, 0, 1)
        return hi - (hi - lo) * down                                   # longer on top, shorter on the sides

    # the opaque 'solid' cap cards of v1 are gone: the scalp cap shell (cap_groom) lies under the cards, so the
    # base layer is dense strand cards (no opaque rectangles at the hairline / in the silhouette)
    layers = [  # count, min spacing, offset root->tip mm, length range cm (sides..top), width cm, columns, twist deg
        dict(n=260, sp=0.0080, off=(1.3, 2.6), L=(2.4, 4.6), w=1.9, cols=(1, 2), tw=6, acc=0.62),
        dict(n=330, sp=0.0068, off=(2.2, 5.0), L=(2.2, 5.0), w=1.6, cols=(1, 2), tw=12, acc=0.5),
        dict(n=260, sp=0.0078, off=(4.5, 9.0), L=(2.0, 5.8), w=1.3, cols=(2, 3, 4), tw=20, acc=0.72),
        dict(n=130, sp=0.011, off=(7.0, 13.0), L=(2.5, 6.2), w=1.1, cols=(3, 4, 5, 6), tw=32, acc=0.78),
    ]
    for li, L in enumerate(layers):
        roots = head.sample(L["n"], L["sp"], lambda p, n, s, a=L["acc"]: s > a, seed + li)
        for p, n, s in roots:
            ln = length_at(p, L["L"][0], L["L"][1]) * 0.01 * rng.uniform(0.85, 1.15)
            o0, o1 = L["off"]
            arch = rng.uniform(0.6, 1.0)
            off = lambda t, o0=o0, o1=o1, arch=arch: 0.001 * (o0 + (o1 - o0) * np.sin(np.pi * min(1, t * 1.3)) * arch)
            ang = np.radians(rng.normal(0, 8))
            dirfn = lambda q, nn, ang=ang: rot_about(flow(q, nn), nn, ang)
            pts, nrms = walk(head, p, n, dirfn, ln, 5, off)
            cut = scalp_len(head, pts, 0.22)
            if cut is not None and cut < ln:                  # ends past the hairline: shorter card, strandier column
                if cut < 0.008:
                    continue
                pts, nrms = walk(head, p, n, dirfn, cut, 5, off)
            sd = sides_for(pts, nrms, np.radians(rng.uniform(-L["tw"], L["tw"])))
            w = L["w"] * 0.01 * rng.uniform(0.8, 1.2) * np.array([0.5, 0.95, 1.0, 0.85, 0.6])
            cols = L["cols"] if cut is None else tuple(c + 2 for c in L["cols"] if c + 2 < COLS) or (5,)
            g.card(pts, sd, w, int(rng.choice(cols)))
    # hairline fringe: short wispy cards just inside the scalp border, low to the skin; denser and finer than v1
    # (the root rows of a real hairline are many fine short hairs), plus a row of dense-column root cards behind them
    fr = head.sample(560, 0.0029, lambda p, n, s: 0.2 < s < 0.9, seed + 9)
    for p, n, s in fr:
        ln = 0.008 + 0.018 * rng.random() * min(1.0, (s - 0.18) / 0.5)   # shortest right at the border
        ang = np.radians(rng.normal(0, 12))
        pts, nrms = walk(head, p, n, lambda q, nn, ang=ang: rot_about(flow(q, nn), nn, ang), ln, 4,
                         lambda t: 0.001 * (0.9 + 1.2 * t))
        sd = sides_for(pts, nrms, np.radians(rng.uniform(-10, 10)))
        g.card(pts, sd, 0.0065 * np.array([0.45, 1.0, 0.85, 0.5]), int(rng.choice((6, 7))))
    roots = head.sample(260, 0.0042, lambda p, n, s: 0.45 < s < 0.95, seed + 10)
    for p, n, s in roots:
        ang = np.radians(rng.normal(0, 8))
        dfn = lambda q, nn, ang=ang: rot_about(flow(q, nn), nn, ang)
        ln = 0.018 + 0.012 * rng.random()
        pts, nrms = walk(head, p, n, dfn, ln, 4, lambda t: 0.001 * (1.1 + 1.6 * t))
        cut = scalp_len(head, pts, 0.22)
        if cut is not None and cut < ln:
            pts, nrms = walk(head, p, n, dfn, max(cut, 0.008), 4, lambda t: 0.001 * (1.1 + 1.6 * t))
        sd = sides_for(pts, nrms, np.radians(rng.uniform(-8, 8)))
        # the whole strand (tapered tips at v ~ 0): v 0.97 -> 0.35 ended mid-strand, a square-cut block at the sideburns
        g.card(pts, sd, 0.011 * np.array([0.5, 1.0, 0.9, 0.6]), int(rng.choice((1, 2, 3))), (0.97, 0.03))
    return g, flow


def style_braid(head, bm, seed=5):
    rng = np.random.default_rng(seed)
    g = Groom()
    hc = head.scalp_center
    # tie point: back of the skull, a little below the scalp centre, lifted off the skin
    tie = braid_tie(head)
    flow = braid_flow(head)

    # base layer: dense strand cards over the scalp cap shell (v1 used opaque 'solid' cards: blocks at the hairline)
    layers = [dict(n=210, sp=0.0085, off=(1.3, 4.5), w=2.1, cols=(1, 2), acc=0.62),
              dict(n=260, sp=0.0072, off=(2.2, 7.0), w=1.7, cols=(1, 2), acc=0.5),
              dict(n=140, sp=0.0095, off=(4.0, 9.0), w=1.3, cols=(3, 4, 5), acc=0.72)]
    for li, L in enumerate(layers):
        roots = head.sample(L["n"], L["sp"], lambda p, n, s, a=L["acc"]: s > a, seed + li)
        for p, n, s in roots:
            dist = np.linalg.norm(tie - p)
            if dist < 0.02:
                continue
            ln = dist * 1.05
            K = max(4, min(7, int(ln / 0.025) + 2))
            o0, o1 = L["off"]
            off = lambda t, o0=o0, o1=o1: 0.001 * (o0 + (o1 - o0) * t)       # rises toward the gathered tie
            ang = np.radians(rng.normal(0, 4))
            pts, nrms = walk(head, p, n, lambda q, nn, ang=ang: rot_about(flow(q, nn), nn, ang), ln, K, off)
            sd = sides_for(pts, nrms, np.radians(rng.uniform(-8, 8)))
            taper = np.linspace(1.0, 0.35, K)
            g.card(pts, sd, L["w"] * 0.01 * rng.uniform(0.85, 1.15) * taper, int(rng.choice(L["cols"])), (0.97, 0.3))
    # hairline wisps (denser, finer) and a row of short dense root cards behind them
    fr = head.sample(480, 0.0030, lambda p, n, s: 0.2 < s < 0.9, seed + 9)
    for p, n, s in fr:
        pts, nrms = walk(head, p, n, flow, 0.014 + 0.012 * rng.random(), 4, lambda t: 0.001 * (0.9 + 1.5 * t))
        g.card(pts, sides_for(pts, nrms, 0.0), 0.0065 * np.array([0.45, 1, 0.85, 0.5]), int(rng.choice((6, 7))))
    roots = head.sample(240, 0.0044, lambda p, n, s: 0.45 < s < 0.95, seed + 10)
    for p, n, s in roots:
        pts, nrms = walk(head, p, n, flow, 0.02 + 0.014 * rng.random(), 4, lambda t: 0.001 * (1.1 + 1.8 * t))
        g.card(pts, sides_for(pts, nrms, np.radians(rng.uniform(-6, 6))), 0.011 * np.array([0.5, 1.0, 0.9, 0.6]),
               int(rng.choice((1, 2, 3))), (0.97, 0.03))

    # braid axis: from the tie, back-and-down, then following the upper back at ~3.5 cm from the skin
    axis = [tie.copy()]
    p = tie.copy()
    L_total = 0.42; seg = 0.012
    down = np.array([0.0, 0.25, -1.0]); down /= np.linalg.norm(down)
    while sum(np.linalg.norm(axis[i + 1] - axis[i]) for i in range(len(axis) - 1)) < L_total:
        p = p + down * seg
        q, n, _ = head.closest(p)
        if np.linalg.norm(p - q) < 0.035 or (p - q) @ n < 0:          # keep clear of the neck / back
            p = q + n * 0.035
        axis.append(p.copy())
    axis = np.array(axis)
    # smooth the axis
    for _ in range(8):
        axis[1:-1] = 0.25 * axis[:-2] + 0.5 * axis[1:-1] + 0.25 * axis[2:]
    arc = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(axis, axis=0), axis=1))])
    Ltot = arc[-1]

    def frame(s):
        i = min(max(int(np.searchsorted(arc, s)), 1), len(arc) - 1); i0 = i - 1
        f = 0 if arc[i] == arc[i0] else (s - arc[i0]) / (arc[i] - arc[i0])
        c = axis[i0] * (1 - f) + axis[i] * f
        t = axis[i] - axis[i0]; t /= max(np.linalg.norm(t), 1e-9)
        side = np.cross(t, [0, 1.0, 0]); side /= np.linalg.norm(side)        # left-right across the back
        depth = np.cross(side, t)
        return c, t, side, depth

    period = 0.055                                     # one full weave cycle along the braid
    start = 0.03; end = Ltot - 0.06
    Nring = 64; S = 6
    for k in range(3):
        rings = []
        for r in range(Nring):
            s = start + (end - start) * r / (Nring - 1)
            c, t, side, depth = frame(s)
            ph = 2 * np.pi * (s / period) + 2 * np.pi * k / 3
            taper = 1.0 - 0.45 * ((s - start) / (end - start)) ** 1.5
            A = 0.011 * taper; B = 0.006 * taper
            ctr = c + side * A * np.sin(ph) + depth * B * np.sin(2 * ph)
            # strand direction ~ derivative; cross-section ellipse (wide across the braid)
            ww, hh = 0.0105 * taper, 0.0068 * taper
            ring = []
            for j in range(S):
                a = 2 * np.pi * j / S
                ring.append(ctr + side * np.cos(a) * ww + depth * np.sin(a) * hh)
            rings.append(np.array(ring))
        g.tube(rings, 0, (0.9, 0.12), 0.9)
    # gathering cone from the tie into the braid (covers where the scalp cards meet the braid)
    rings = []
    for r in range(7):
        s = start * r / 6
        c, t, side, depth = frame(s)
        rad = 0.022 - 0.006 * r / 6
        rings.append(np.array([c + side * np.cos(2 * np.pi * j / 10) * rad + depth * np.sin(2 * np.pi * j / 10) * rad * 0.8
                               for j in range(10)]))
    g.tube(rings, 0, (0.98, 0.8))
    # leather ties: at the head and at the braid end
    for s0, rad in ((start + 0.004, 0.017), (end - 0.004, 0.0125)):
        rings = []
        for r in range(3):
            c, t, side, depth = frame(s0 + (r - 1) * 0.005)
            rings.append(np.array([c + side * np.cos(2 * np.pi * j / 10) * rad + depth * np.sin(2 * np.pi * j / 10) * rad * 0.8
                                   for j in range(10)]))
        g.tube(rings, 7, (0.996, 0.991), 0.8)          # leather swatch at the top of the atlas' last column
    # loose tail after the end tie: fanned cards
    c, t, side, depth = frame(end)
    for j in range(9):
        a = 2 * np.pi * j / 9
        rt = c + (side * np.cos(a) + depth * np.sin(a) * 0.7) * 0.006
        tip_dir = t + (side * np.cos(a) + depth * np.sin(a)) * 0.25
        pts = np.array([rt + tip_dir * 0.075 * q / 4 for q in range(5)])
        sd = np.array([np.cross(t, depth * np.cos(a) - side * np.sin(a))] * 5)
        sd /= np.linalg.norm(sd, axis=1)[:, None]
        g.card(pts, sd, 0.016 * np.array([1, 1, 0.9, 0.7, 0.4]), int(rng.choice((1, 2))))
    return g, flow



# ------------------------------------------------------------------------------------------------ beards
def smoothstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0, 1)
    return t * t * (3 - 2 * t)


class BeardRegion(Head):
    """Head surface whose growth mask ('scalp' channel, used by sample/closest) is the beard area: front of the ears,
    below a cheek line that runs from the sideburns to the mouth corners and up to the nose base over the upper lip
    (moustache), down to the upper throat; lips and the mouth interior excluded. Landmarks come from the rts_human
    face bones and MakeHuman's 'ears' / 'lips' vertex groups, so the mask fits any body."""

    def __init__(self, rig, bm):
        super().__init__(rig, bm)
        inv = bm.matrix_world.inverted() @ rig.matrix_world
        B = lambda b: np.array((inv @ rig.pose.bones[b].head)[:])
        n = len(self.co)
        def grp(name):
            w = np.zeros(n); gi = bm.vertex_groups[name].index
            for v in bm.data.vertices:
                for g in v.groups:
                    if g.group == gi:
                        w[v.index] = g.weight
            return w
        lips, ears = grp("lips"), grp("ears")
        body = np.zeros(n, bool); body[:13380] = True
        co, no = self.co, self.no
        M = (B("lip_upper_c") + B("lip_lower_c")) / 2
        mc = abs(B("mouth_corner_l")[0])
        nb = B("nose_l")[2] - 0.004                       # nose base (nostril wings)
        e = co[(ears > 0.5) & body & (co[:, 0] > 0)]
        self.ear_front, self.ear_cz, self.ear_bot = e[:, 1].min(), e[:, 2].mean(), e[:, 2].min()
        self.lip_front = co[(lips > 0.5) & body][:, 1].min()
        self.M, self.mc, self.nb = M, mc, nb
        self.lips = lips
        # lip outline for the stubble texture: MakeHuman's 'lips' group is binary (weight 1), so its per-texel
        # interpolation cuts a rectangle; the outline is modelled from the group's extents instead (front-facing)
        s = co[(lips > 0.5) & body & (no[:, 1] < -0.2)]
        mid = s[np.abs(s[:, 0]) < 0.004]
        self.lip_xr = float(np.abs(s[:, 0]).max())
        self.lip_top, self.lip_bot = float(mid[:, 2].max() - M[2]), float(M[2] - mid[:, 2].min())
        self.lip_zc = float(s[np.abs(s[:, 0]) > self.lip_xr - 0.003][:, 2].mean() - M[2])
        m = self.mask_at(co, no, lips)
        m[~body] = 0.0
        self.scalp = m
        self.neck_y = float(B("neck_02")[1])

    def lip_factor(self, co):
        """0 on the red lips, 1 from ~4 mm outside the modelled lip outline (lens shape through the lips group's
        extents: corners at lip_xr / lip_zc, upper and lower lip heights at the centre)."""
        M = self.M
        x = np.abs(co[:, 0]); z = co[:, 2] - M[2]
        u = np.clip(x / (self.lip_xr + 0.001), 0, 1)
        zc = self.lip_zc * u ** 2
        env = (1 - u ** 2) ** 0.55
        zt = zc + (self.lip_top + 0.0008) * env
        zb = zc - (self.lip_bot + 0.0008) * env
        d = np.maximum(np.maximum(z - zt, zb - z), x - self.lip_xr - 0.001)
        return smoothstep(0.0006, 0.0045, d)

    def mask_at(self, co, no, lips, lip_shape=False, pad=0.0):
        """Beard growth mask for points co (N,3) with normals no and MakeHuman 'lips' weight (per vertex, or per
        texel for the stubble shell texture: the same function gives smooth, mesh-independent borders; lip_shape uses
        the modelled lip outline instead of the binary lips group; pad (m) widens the cheek-line / sideburn falloff
        both ways and the neckline twice as much: a soft, natural stubble edge)."""
        M, mc, nb = self.M, self.mc, self.nb
        x = np.abs(co[:, 0]); y = co[:, 1]; z = co[:, 2]
        ztop = np.interp(x, [0.0, mc + 0.004, mc + 0.016, 0.068], [nb - 0.004, nb - 0.007, M[2] + 0.02, self.ear_cz + 0.012])
        zbot = np.interp(x, [0.0, 0.04, 0.075], [M[2] - 0.085, M[2] - 0.075, self.ear_bot - 0.015])
        m = smoothstep(-pad, 0.006 + pad, ztop - z) * smoothstep(-2 * pad, 0.012 + 2 * pad, z - zbot)
        m *= smoothstep(-pad, 0.008 + pad, (self.ear_front + 0.006) - y)   # in front of the ears
        m *= self.lip_factor(co) if lip_shape else 1.0 - smoothstep(0.02, 0.3, lips)   # not on the red lips
        if not lip_shape:     # mouth interior (a box: it also cuts the recessed skin around the lips; the modelled
            inside = (x < mc + 0.003) & (np.abs(z - M[2]) < 0.012) & (y > self.lip_front + 0.008)   # lip outline
            m[inside] = 0.0   # already excludes the mouth opening)
        m[no[:, 1] > 0.55] = 0.0                                               # surfaces facing backwards
        return m


def style_beard(head, full, seed=11):
    rng = np.random.default_rng(seed + (7 if full else 0))
    g = Groom()
    M, mc = head.M, head.mc

    def moustache(p):
        return p[2] > M[2] + 0.004 and abs(p[0]) < mc + 0.01

    def flow(p, n):
        if moustache(p):
            d = np.array([np.sign(p[0]) * 0.9, -0.1, -0.75])
        else:
            d = np.array([np.sign(p[0]) * 0.08, -0.3, -1.0])
        return d / np.linalg.norm(d)

    def chin_gain(p):                                   # longer on the chin, shorter up the cheeks / sideburns
        return 1.0 + 0.8 * smoothstep(0.035, 0.0, abs(p[0])) * smoothstep(M[2] - 0.01, M[2] - 0.04, p[2])

    k = 1.3 if full else 1.0
    f = 0.75 if full else 1.0                            # the long beard covers the chin: fewer short cards
    layers = [  # count, spacing, offset root->tip mm, length cm, width cm, atlas columns, min mask
        dict(n=int(430 * f), sp=0.0028, off=(1.0, 1.6), L=(0.8, 1.4), w=0.9, cols=(0,), acc=0.45),
        dict(n=int(600 * f), sp=0.0023, off=(1.5, 3.0), L=(1.0, 2.0), w=0.62, cols=(1, 2), acc=0.3),
        dict(n=int(330 * f), sp=0.0031, off=(1.5, 4.0), L=(1.2, 2.4), w=0.45, cols=(3, 4, 5), acc=0.45),
    ]
    for li, L in enumerate(layers):
        roots = head.sample(L["n"], L["sp"], lambda p, n, s, a=L["acc"]: s > a, seed + li)
        for p, n, s in roots:
            mous = moustache(p)
            ln = (rng.uniform(*L["L"]) * 0.01 * k * (0.85 if mous else chin_gain(p))) * (0.55 + 0.45 * s)
            o0, o1 = L["off"]
            off = lambda t, o0=o0, o1=o1: 0.001 * (o0 + (o1 - o0) * t)
            ang = np.radians(rng.normal(0, 9))
            pts, nrms = walk(head, p, n, lambda q, nn, ang=ang: rot_about(flow(q, nn), nn, ang), ln, 3, off)
            sd = sides_for(pts, nrms, np.radians(rng.uniform(-12, 12)))
            w = L["w"] * 0.01 * rng.uniform(0.8, 1.2) * np.array([0.85, 1.0, 0.6])
            # short cards use the tip half of the strand atlas, so strands are not squashed along the card
            g.card(pts, sd, w, int(rng.choice(L["cols"])), (0.55, 0.02))
    if not full:
        return g
    # long hanging beard: free-hanging cards from the chin / front jaw, kept clear of the neck and chest
    def hang(p0, n0, length, K, spread, clear0, clear1, fwd, wave):
        pts = [p0 + n0 * clear0]
        step = length / (K - 1)
        for i in range(1, K):
            t = i / (K - 1)
            # locks gather towards the centre line as they fall (tapered, dwarf-style) and wave slightly
            conv = -np.clip(pts[-1][0], -0.05, 0.05) * 0.9 * t
            d = np.array([conv + spread * np.sign(p0[0]) * 0.15 + wave * np.sin(7 * t + p0[0] * 90),
                          -fwd * (1 - t) ** 2, -1.0]); d /= np.linalg.norm(d)
            p = pts[-1] + d * step
            q, nq, _ = head.closest(p)
            c = clear0 + (clear1 - clear0) * t
            if (p - q) @ nq < c:
                p = q + nq * c
            pts.append(p)
        pts = np.array(pts)
        nr = []
        for p in pts:                                     # cards face forward / away from the neck axis
            r = p - np.array([0.0, head.neck_y, p[2]]); r[2] = 0
            r = r / max(np.linalg.norm(r), 1e-9)
            v = 0.55 * r + np.array([0.0, -0.45, 0.0]); nr.append(v / np.linalg.norm(v))
        return pts, np.array(nr)
    long_layers = [dict(n=70, sp=0.0055, cols=(0,), w=2.2, c=(0.004, 0.02), L=0.22),
                   dict(n=190, sp=0.0038, cols=(1, 2), w=1.8, c=(0.006, 0.03), L=0.24),
                   dict(n=100, sp=0.0048, cols=(3, 4, 5), w=1.4, c=(0.01, 0.045), L=0.21)]
    for li, L in enumerate(long_layers):
        roots = head.sample(L["n"], L["sp"], lambda p, n, s: s > 0.5 and p[2] < M[2] - 0.016 and abs(p[0]) < 0.056,
                            seed + 20 + li)
        for p, n, s in roots:
            side = min(1.0, abs(p[0]) / 0.056)
            ln = L["L"] * (1 - 0.6 * side ** 1.3) * rng.uniform(0.85, 1.1)
            pts, nrms = hang(p, n, ln, 7, rng.uniform(0.0, 0.3), L["c"][0], L["c"][1] + 0.012 * rng.random(), 0.4,
                             rng.uniform(-0.06, 0.06))
            sd = sides_for(pts, nrms, np.radians(rng.uniform(-25, 25)))
            taper = np.linspace(1.0, 0.3, 7) ** 0.8
            g.card(pts, sd, L["w"] * 0.01 * rng.uniform(0.85, 1.15) * taper, int(rng.choice(L["cols"])), (0.975, 0.02))
    # long drooping moustache merging into the beard
    roots = head.sample(50, 0.0045, lambda p, n, s: s > 0.5 and moustache(p), seed + 30)
    for p, n, s in roots:
        pts = [p + n * 0.002]
        d = np.array([np.sign(p[0]) * 0.85, -0.15, -0.5])
        for i in range(1, 6):
            t = i / 5
            dd = d * (1 - t) + np.array([np.sign(p[0]) * 0.1, -0.2, -1.0]) * t; dd /= np.linalg.norm(dd)
            q = pts[-1] + dd * 0.013
            c, nq, _ = head.closest(q)
            if (q - c) @ nq < 0.003 + 0.01 * t:
                q = c + nq * (0.003 + 0.01 * t)
            pts.append(q)
        pts = np.array(pts)
        nrms = np.array([head.closest(q)[1] for q in pts])
        sd = sides_for(pts, nrms, np.radians(rng.uniform(-10, 10)))
        g.card(pts, sd, 0.012 * rng.uniform(0.8, 1.2) * np.linspace(1, 0.35, 6), int(rng.choice((1, 2, 3))))
    return g


# ------------------------------------------------------------------------------------------------ stubble / beard shadow
# A skin-hugging SHELL over the beard area (the body faces under the beard mask, 0.8 mm off the skin) with an
# alpha-BLENDED texture painted in 3D: short stubble hairs (area-sampled on the shell, oriented along the beard flow,
# splatted into the shell's LSCM unwrap) over a density mask. Two textures share the one unwrap:
#   beard_stubble_base.png  stubble alone (the 'rts_stubble' beard part)
#   beard_shadow_base.png   dense beard shadow, joined under the card beards by cust_lib.merge_beard_shadows so the
#                           skin never shows between the cards (the cards alone read as blocky patches)
# RGB is a flat neutral grey: the hair-colour baseColorFactor tints it exactly like the neutral strand atlas.
SHELL_OFFSET = 0.0008
SHELL_TEX = 1024
SHELL_GREY = 0.56


SOFT_PAD = 0.003                                         # stubble edge softening (BeardRegion.mask_at pad)


def shell_groom(head):
    body = np.zeros(len(head.co), bool); body[:13380] = True
    m = head.mask_at(head.co, head.no, head.lips, lip_shape=True, pad=SOFT_PAD)
    m[~body] = 0.0
    head.shell_mask = m                                  # superset of the card beards' mask (lip outline, soft pad)
    polys = [p for p in head.polys if m[p].max() > 0.02]
    vids = sorted({v for p in polys for v in p})
    remap = {v: i for i, v in enumerate(vids)}
    g = Groom()
    g.V = [head.co[v] + head.no[v] * SHELL_OFFSET for v in vids]
    g.UV = [(0.0, 0.0)] * len(vids)
    g.F = [tuple(remap[v] for v in p) for p in polys]
    return g, np.array(vids)


def unwrap(ob):
    for o in bpy.context.view_layer.objects:
        o.select_set(False)
    bpy.context.view_layer.objects.active = ob
    ob.select_set(True)
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.uv.unwrap(method='CONFORMAL', margin=0.01)
    bpy.ops.object.mode_set(mode='OBJECT')


def _save_rgba(path, rgba):
    S = rgba.shape[0]
    im = bpy.data.images.new(os.path.basename(path), S, S, alpha=True)
    im.pixels.foreach_set(np.clip(rgba, 0, 1).astype(np.float32).ravel())
    im.filepath_raw = path
    im.file_format = 'PNG'
    im.save()
    bpy.data.images.remove(im)


def paint_shell(ob, vids, head, seed=5):
    """Paints the stubble + shadow textures for the shell `ob` (LSCM unwrap, no seams; vids = the body vertex of
    each shell vertex). The beard mask is evaluated per texel (BeardRegion.mask_at on the interpolated skin position,
    normal and lips weight), so its borders are smooth whatever the mesh density. Returns stats."""
    mask_v = head.shell_mask[vids]
    me = ob.data
    n = len(me.vertices)
    co = np.array([v.co[:] for v in me.vertices])
    uvd = me.uv_layers.active.data
    uv = np.full((n, 2), np.nan)
    for p in me.polygons:
        for li in p.loop_indices:
            vi = me.loops[li].vertex_index
            u = np.array(uvd[li].uv[:])
            assert np.isnan(uv[vi, 0]) or np.abs(uv[vi] - u).max() < 1e-5, "shell unwrap has seams"
            uv[vi] = u
    tris = np.array([(p.vertices[0], p.vertices[k], p.vertices[k + 1]) for p in me.polygons
                     for k in range(1, len(p.vertices) - 1)])
    S = SHELL_TEX
    P = uv * S
    # ---- density mask: rasterise the per-vertex beard mask in UV space (row 0 = v 0, Blender's pixel order)
    M = np.zeros((S, S)); W = np.zeros((S, S), bool)
    CO = np.zeros((S, S, 3)); NO = np.zeros((S, S, 3)); LI = np.zeros((S, S))
    bco, bno, bli = head.co[vids], head.no[vids], head.lips[vids]
    flipped = 0
    for t in tris:
        a, b, c = P[t]
        v0, v1 = b - a, c - a
        den = v0[0] * v1[1] - v1[0] * v0[1]
        if abs(den) < 1e-12:
            continue
        flipped += den < 0
        x0, y0 = np.maximum(np.floor(np.minimum(np.minimum(a, b), c)).astype(int) - 1, 0)
        x1, y1 = np.minimum(np.ceil(np.maximum(np.maximum(a, b), c)).astype(int) + 1, S - 1)
        xs, ys = np.meshgrid(np.arange(x0, x1 + 1) + 0.5, np.arange(y0, y1 + 1) + 0.5)
        px, py = xs - a[0], ys - a[1]
        l1 = (px * v1[1] - v1[0] * py) / den
        l2 = (v0[0] * py - px * v0[1]) / den
        l0 = 1 - l1 - l2
        inside = (l0 >= -1e-3) & (l1 >= -1e-3) & (l2 >= -1e-3)
        val = l0 * mask_v[t[0]] + l1 * mask_v[t[1]] + l2 * mask_v[t[2]]
        sub = (slice(y0, y1 + 1), slice(x0, x1 + 1))
        M[sub] = np.where(inside, val, M[sub]); W[sub] |= inside
        lw = np.stack([l0, l1, l2], -1)
        CO[sub] = np.where(inside[..., None], lw @ bco[t], CO[sub])
        NO[sub] = np.where(inside[..., None], lw @ bno[t], NO[sub])
        LI[sub] = np.where(inside, lw @ bli[t], LI[sub])
    cover = W.mean()
    nn = NO[W]; nn /= np.maximum(np.linalg.norm(nn, axis=1), 1e-9)[:, None]
    # analytic masks per texel (hard: the card beards' shadow; soft: stubble), faded to 0 over the last faces where
    # the shell geometry ends (interpolated vertex mask)
    fade = smoothstep(0.0, 0.25, M[W])
    MS = np.zeros_like(M)
    MS[W] = head.mask_at(CO[W], nn, LI[W], lip_shape=True, pad=SOFT_PAD) * fade
    M[W] = head.mask_at(CO[W], nn, LI[W], lip_shape=True) * fade
    del CO, NO, LI
    for _ in range(6):                                   # pad the island (mip / bilinear bleed)
        f = W.astype(float); acc = np.zeros_like(M); cnt = np.zeros_like(M)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                acc += np.roll(np.roll(M * f, dy, 0), dx, 1); cnt += np.roll(np.roll(f, dy, 0), dx, 1)
        new = ~W & (cnt > 0)
        M[new] = acc[new] / cnt[new]; W |= new
    # ---- stubble hairs, area-sampled in 3D, splatted along their UV direction (triangle Jacobian)
    A = co[tris]
    area = 0.5 * np.linalg.norm(np.cross(A[:, 1] - A[:, 0], A[:, 2] - A[:, 0]), axis=1)
    texel_mm = np.sqrt(area.sum() / (cover * S * S)) * 1000
    rng = np.random.default_rng(seed)
    M_c = head.M

    def hairs(density_cm2, L_mm, width_mm, pad):
        N = int(area.sum() * 1e4 * density_cm2)
        ti = rng.choice(len(tris), N, p=area / area.sum())
        r1, r2 = rng.random(N), rng.random(N)
        fl = r1 + r2 > 1; r1[fl], r2[fl] = 1 - r1[fl], 1 - r2[fl]
        w = np.stack([1 - r1 - r2, r1, r2], 1)
        T = tris[ti]
        nh = (bno[T] * w[:, :, None]).sum(1); nh /= np.linalg.norm(nh, axis=1)[:, None]
        mk = head.mask_at((co[T] * w[:, :, None]).sum(1), nh, (bli[T] * w).sum(1), lip_shape=True, pad=pad)
        keep = rng.random(N) < smoothstep(0.02, 0.5, mk)
        T, w = T[keep], w[keep]; N = len(T)
        p3 = (co[T] * w[:, :, None]).sum(1); puv = (P[T] * w[:, :, None]).sum(1)
        e1 = co[T[:, 1]] - co[T[:, 0]]; e2 = co[T[:, 2]] - co[T[:, 0]]
        f1 = P[T[:, 1]] - P[T[:, 0]]; f2 = P[T[:, 2]] - P[T[:, 0]]
        nrm = np.cross(e1, e2); nrm /= np.linalg.norm(nrm, axis=1)[:, None]
        mous = (p3[:, 2] > M_c[2] + 0.004) & (np.abs(p3[:, 0]) < head.mc + 0.01)
        d = np.where(mous[:, None], np.stack([np.sign(p3[:, 0]) * 0.9, np.full(N, -0.1), np.full(N, -0.75)], 1),
                     np.stack([np.sign(p3[:, 0]) * 0.1, np.full(N, -0.3), np.full(N, -1.0)], 1))
        d += rng.normal(0, 0.25, (N, 3))
        d -= nrm * (d * nrm).sum(1)[:, None]                                 # into the tangent plane
        under = np.linalg.norm(d, axis=1) < 0.45                              # under the chin: towards the throat
        back = np.array([0.0, 1.0, -0.3]); db = back - nrm * (nrm @ back)[:, None]
        d[under] = d[under] + db[under]
        d /= np.maximum(np.linalg.norm(d, axis=1), 1e-9)[:, None]
        g11 = (e1 * e1).sum(1); g12 = (e1 * e2).sum(1); g22 = (e2 * e2).sum(1)
        q1 = (d * e1).sum(1); q2 = (d * e2).sum(1); det = g11 * g22 - g12 * g12
        a_ = (g22 * q1 - g12 * q2) / det; b_ = (g11 * q2 - g12 * q1) / det
        duv = a_[:, None] * f1 + b_[:, None] * f2                             # pixels per metre along the hair
        L = rng.uniform(*L_mm, N) * 0.001
        K = int(np.ceil(L_mm[1] / texel_mm)) + 2
        H = np.zeros((S, S))
        wpx = min(1.0, width_mm / texel_mm)
        for k in range(K):
            t = k / (K - 1)
            q = puv + duv * (L * t)[:, None] - 0.5
            x0 = np.floor(q[:, 0]).astype(int); y0 = np.floor(q[:, 1]).astype(int)
            fx, fy = q[:, 0] - x0, q[:, 1] - y0
            amp = wpx * (1.0 - 0.55 * t)                                      # tapering tip
            for dx, dy, ww in ((0, 0, (1 - fx) * (1 - fy)), (1, 0, fx * (1 - fy)), (0, 1, (1 - fx) * fy), (1, 1, fx * fy)):
                xi, yi = np.clip(x0 + dx, 0, S - 1), np.clip(y0 + dy, 0, S - 1)
                np.add.at(H, (yi, xi), amp * ww)
        return 1.0 - np.exp(-1.6 * H), N

    Hs, ns = hairs(240, (0.4, 1.3), 0.07, SOFT_PAD)                            # stubble (a few days): fine grain
    Hd, nd = hairs(160, (1.5, 3.5), 0.11, 0.0)                                 # under the card beards
    a_st = np.clip(smoothstep(0.3, 0.95, MS) * 0.40 + smoothstep(0.03, 0.5, MS) * 0.5 * Hs, 0, 1)
    a_sh = np.clip(np.maximum(smoothstep(0.12, 0.6, M) * 0.86, smoothstep(0.03, 0.5, M) * 0.95 * Hd), 0, 1)
    for name, alpha in (("stubble", a_st), ("shadow", a_sh)):
        rgba = np.empty((S, S, 4)); rgba[..., :3] = SHELL_GREY; rgba[..., 3] = alpha
        _save_rgba(os.path.join(TEX, "beard_%s_base.png" % name), rgba)
    st = dict(verts=n, tris=len(tris), flipped_uv_tris=int(flipped), uv_cover=round(float(cover), 3),
              texel_mm=round(float(texel_mm), 3), stubble_hairs=ns, shadow_hairs=nd, area_cm2=round(float(area.sum() * 1e4), 1))
    log("stubble shell", st)
    return st

# ------------------------------------------------------------------------------------------------ scalp cap
# User round-2 item 11 (scalp gaps between the card clumps): a skin-hugging SHELL over the scalp (the body faces under
# the smoothed MakeHuman scalp mask, CAP_OFFSET off the skin, LSCM unwrap) with a painted strand texture that follows
# the groom's own comb flow: dense roots in the interior (opaque: skin never shows between the cards), a hairline that
# breaks up into single strands (alpha = strand coverage in the feather zone), neutral grey RGB so the hair-colour
# baseColorFactor tints it like the strand atlas. Written as its own MPFB hair asset (rts_cap_<style>) and joined into
# the hair mesh as a second primitive by hair_lib.join_caps (as cust_lib.merge_beard_shadows does for the beards).
CAP_OFFSET = 0.0007
CAP_TEX = 1024
CAP_FLOW = {"crop": crop_flow, "braid": braid_flow, "generic": crop_flow}


def cap_groom(head):
    body = np.zeros(len(head.co), bool); body[:13380] = True
    m = np.where(body, head.scalp, 0.0)
    polys = [p for p in head.polys if m[p].max() > 0.03]
    vids = sorted({v for p in polys for v in p})
    remap = {v: i for i, v in enumerate(vids)}
    g = Groom()
    g.V = [head.co[v] + head.no[v] * CAP_OFFSET for v in vids]
    g.UV = [(0.0, 0.0)] * len(vids)
    g.F = [tuple(remap[v] for v in p) for p in polys]
    return g, np.array(vids)


def paint_cap(ob, vids, head, flow, name, seed=17):
    """Paints hair_cap_<name>_base.png for the cap shell `ob` (vids = body vertex of each shell vertex): strands
    splatted along the comb flow in the shell's unwrap (triangle Jacobians, as paint_shell), alpha from the smoothed
    scalp mask per texel (opaque inside, strand-broken hairline). Returns stats."""
    me = ob.data
    n = len(me.vertices)
    co = np.array([v.co[:] for v in me.vertices])
    uvd = me.uv_layers.active.data
    uv = np.full((n, 2), np.nan)
    for p in me.polygons:
        for li in p.loop_indices:
            vi = me.loops[li].vertex_index
            u = np.array(uvd[li].uv[:])
            assert np.isnan(uv[vi, 0]) or np.abs(uv[vi] - u).max() < 1e-5, "cap unwrap has seams"
            uv[vi] = u
    tris = np.array([(p.vertices[0], p.vertices[k], p.vertices[k + 1]) for p in me.polygons
                     for k in range(1, len(p.vertices) - 1)])
    S = CAP_TEX
    P = uv * S
    sv = head.scalp[vids]
    M = np.zeros((S, S)); W = np.zeros((S, S), bool)
    for t in tris:
        a, b, c = P[t]
        v0, v1 = b - a, c - a
        den = v0[0] * v1[1] - v1[0] * v0[1]
        if abs(den) < 1e-12:
            continue
        x0, y0 = np.maximum(np.floor(np.minimum(np.minimum(a, b), c)).astype(int) - 1, 0)
        x1, y1 = np.minimum(np.ceil(np.maximum(np.maximum(a, b), c)).astype(int) + 1, S - 1)
        xs, ys = np.meshgrid(np.arange(x0, x1 + 1) + 0.5, np.arange(y0, y1 + 1) + 0.5)
        px, py = xs - a[0], ys - a[1]
        l1 = (px * v1[1] - v1[0] * py) / den
        l2 = (v0[0] * py - px * v0[1]) / den
        l0 = 1 - l1 - l2
        inside = (l0 >= -1e-3) & (l1 >= -1e-3) & (l2 >= -1e-3)
        val = l0 * sv[t[0]] + l1 * sv[t[1]] + l2 * sv[t[2]]
        sub = (slice(y0, y1 + 1), slice(x0, x1 + 1))
        M[sub] = np.where(inside, val, M[sub]); W[sub] |= inside
    cover = W.mean()
    for _ in range(8):                                   # pad the island (mip / bilinear bleed)
        f = W.astype(float); acc = np.zeros_like(M); cnt = np.zeros_like(M)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                acc += np.roll(np.roll(M * f, dy, 0), dx, 1); cnt += np.roll(np.roll(f, dy, 0), dx, 1)
        new = ~W & (cnt > 0)
        M[new] = acc[new] / cnt[new]; W |= new
    A3 = co[tris]
    area = 0.5 * np.linalg.norm(np.cross(A3[:, 1] - A3[:, 0], A3[:, 2] - A3[:, 0]), axis=1)
    texel_mm = np.sqrt(area.sum() / (cover * S * S)) * 1000
    rng = np.random.default_rng(seed)
    bno = head.no[vids]
    wt = area * np.clip(sv[tris].mean(1) * 3, 0.05, 1)
    N = int(area.sum() * 1e4 * 70)                      # ~70 painted strands per cm2
    ti = rng.choice(len(tris), N, p=wt / wt.sum())
    r1, r2 = rng.random(N), rng.random(N)
    fl = r1 + r2 > 1; r1[fl], r2[fl] = 1 - r1[fl], 1 - r2[fl]
    w = np.stack([1 - r1 - r2, r1, r2], 1)
    T = tris[ti]
    p3 = (co[T] * w[:, :, None]).sum(1); puv = (P[T] * w[:, :, None]).sum(1)
    nh = (bno[T] * w[:, :, None]).sum(1); nh /= np.linalg.norm(nh, axis=1)[:, None]
    d = np.array([flow(p3[i], nh[i]) for i in range(N)]) + rng.normal(0, 0.12, (N, 3))
    e1 = co[T[:, 1]] - co[T[:, 0]]; e2 = co[T[:, 2]] - co[T[:, 0]]
    f1 = P[T[:, 1]] - P[T[:, 0]]; f2 = P[T[:, 2]] - P[T[:, 0]]
    nrm = np.cross(e1, e2); nrm /= np.linalg.norm(nrm, axis=1)[:, None]
    d -= nrm * (d * nrm).sum(1)[:, None]
    d /= np.maximum(np.linalg.norm(d, axis=1), 1e-9)[:, None]
    g11 = (e1 * e1).sum(1); g12 = (e1 * e2).sum(1); g22 = (e2 * e2).sum(1)
    q1 = (d * e1).sum(1); q2 = (d * e2).sum(1); det = g11 * g22 - g12 * g12
    a_ = (g22 * q1 - g12 * q2) / det; b_ = (g11 * q2 - g12 * q1) / det
    duv = a_[:, None] * f1 + b_[:, None] * f2                                 # pixels per metre along the strand
    L = rng.uniform(6, 18, N) * 0.001
    lum = rng.uniform(0.55, 1.0, N)
    K = int(np.ceil(18 / texel_mm)) + 2
    H = np.zeros((S, S)); HL = np.zeros((S, S))
    wpx = min(1.0, 0.09 / texel_mm)
    for k in range(K):
        t = k / (K - 1)
        q = puv + duv * (L * t)[:, None] - 0.5
        x0 = np.floor(q[:, 0]).astype(int); y0 = np.floor(q[:, 1]).astype(int)
        fx, fy = q[:, 0] - x0, q[:, 1] - y0
        amp = wpx * (1.0 - 0.6 * t)                                           # tapering tip
        for dx, dy, ww in ((0, 0, (1 - fx) * (1 - fy)), (1, 0, fx * (1 - fy)), (0, 1, (1 - fx) * fy), (1, 1, fx * fy)):
            xi, yi = np.clip(x0 + dx, 0, S - 1), np.clip(y0 + dy, 0, S - 1)
            np.add.at(H, (yi, xi), amp * ww); np.add.at(HL, (yi, xi), amp * ww * lum)
    cov = 1.0 - np.exp(-1.4 * H)
    L_ = np.where(H > 1e-6, HL / np.maximum(H, 1e-6), 0.75)
    grey = np.clip(0.34 + 0.46 * cov * L_, 0, 1)                             # dark roots between the strands
    alpha = np.clip(np.maximum(smoothstep(0.25, 0.6, M), smoothstep(0.04, 0.25, M) * cov * 1.15), 0, 1)
    rgba = np.empty((S, S, 4)); rgba[..., 0] = rgba[..., 1] = rgba[..., 2] = grey; rgba[..., 3] = alpha
    _save_rgba(os.path.join(TEX, "hair_cap_%s_base.png" % name), rgba)
    st = dict(verts=n, tris=len(tris), uv_cover=round(float(cover), 3), texel_mm=round(float(texel_mm), 3), strands=N,
              area_cm2=round(float(area.sum() * 1e4), 1))
    log("scalp cap", name, st)
    return st


# ------------------------------------------------------------------------------------------------ MakeClothes
def write_mhclo_asset(name, clothes, bm, description, tex=None):
    d = os.path.join(ASSET_DIR, name)
    os.makedirs(d, exist_ok=True)
    vg = clothes.vertex_groups.new(name="body")
    vg.add(list(range(len(clothes.data.vertices))), 1.0, 'REPLACE')
    chk = ClothesService.mesh_is_valid_as_clothes(clothes, bm)
    assert chk["all_checks_ok"], chk
    props = {"name": name, "description": description, "author": "RTS project (procedural, scripts/hair_gen.py)",
             "license": "CC0", "homepage": "", "uuid": str(uuid.uuid5(uuid.NAMESPACE_URL, "rts-hair-" + name))}
    t0 = time.time()
    # allow_exact=False: an 'exact' match snaps a card vertex onto a skin vertex (zero offset), which would press the
    # cap cards into the scalp; offset matches keep the authored lift
    mhclo = ClothesService.create_mhclo_from_clothes_matching(bm, clothes, properties_dict=props, allow_exact=False)
    log("mhclo matching %s: %d verts in %.1fs" % (name, len(clothes.data.vertices), time.time() - t0))
    mhclo.material = name + ".mhmat"
    path = os.path.join(d, name + ".mhclo")
    mhclo.write_mhclo(path, reference_scale=ClothesService.get_reference_scale(bm), also_export_mhmat=False)
    # z_depth: draw order hint for MakeHuman (hair ~ 60)
    txt = open(path).read().replace("verts 0", "z_depth 60\n\nverts 0", 1)
    open(path, "w").write(txt)
    if tex:                                              # stubble shell: its own texture, no normal map
        shutil.copy(os.path.join(TEX, tex), os.path.join(d, tex))
        open(os.path.join(d, name + ".mhmat"), "w").write(
            "# MakeHuman material, RTS procedural stubble (CC0)\nname %s\ndiffuseColor 1.0 1.0 1.0\nshininess 0.3\n"
            "transparent True\nbackfaceCull True\ndiffuseTexture %s\nshaderConfig transparency True\n" % (name, tex))
    else:
        tex = "hair_strands_%s_base.png" % STYLE_KIND[name[4:]]
        shutil.copy(os.path.join(TEX, tex), os.path.join(d, tex))
        shutil.copy(os.path.join(TEX, "hair_strands_normal.png"), os.path.join(d, "hair_strands_normal.png"))
        open(os.path.join(d, name + ".mhmat"), "w").write(
            "# MakeHuman material, RTS procedural hair (CC0)\nname %s\ndiffuseColor 1.0 1.0 1.0\nshininess 0.4\n"
            "transparent True\nalphaToCoverage True\nbackfaceCull False\ndiffuseTexture %s\nnormalmapTexture hair_strands_normal.png\n"
            "shaderConfig transparency True\nshaderConfig normal True\n" % (name, tex))
    dst = os.path.join(USER_DATA, "hair", name)
    if os.path.exists(dst):
        shutil.rmtree(dst)
    shutil.copytree(d, dst)
    log("hair asset", path, "->", dst)
    return path


for style in styles:
    kind = STYLE_KIND[style]
    clear_scene()
    rig, bm = build_human(kind, parts=False, face=False)
    head = Head(rig, bm)
    t0 = time.time()
    if style.startswith("cap_"):
        cname = style[4:]
        g, vids = cap_groom(head)
        ob = g.to_object("rts_" + style)
        unwrap(ob)
        paint_cap(ob, vids, head, CAP_FLOW[cname](head), cname)
        log("groom %s: shell faces %d verts %d (%.1fs)" % (style, len(g.F), len(g.V), time.time() - t0))
        write_mhclo_asset("rts_" + style, ob, bm, "Procedural scalp cap under the hair cards: " + cname,
                          tex="hair_cap_%s_base.png" % cname)
        continue
    if style == "stubble":
        head = BeardRegion(rig, bm)
        g, vids = shell_groom(head)
        ob = g.to_object("rts_stubble")
        unwrap(ob)
        paint_shell(ob, vids, head)
        log("groom stubble: shell faces %d verts %d (%.1fs)" % (len(g.F), len(g.V), time.time() - t0))
        write_mhclo_asset("rts_stubble", ob, bm, "Procedural stubble / beard-shadow shell", tex="beard_stubble_base.png")
        continue
    if style.startswith("beard"):
        head = BeardRegion(rig, bm)
        g = style_beard(head, full=style == "beard_full")
    else:
        g, _ = style_crop(head) if style == "crop" else style_braid(head, bm)
    V = np.array(g.V)
    assert np.isfinite(V).all(), "non-finite groom vertices"
    ob = g.to_object("rts_" + style)
    log("groom %s: %d cards/tubes quads %d verts %d (%.1fs)" % (style, 0, len(g.F), len(g.V), time.time() - t0))
    write_mhclo_asset("rts_" + style, ob, bm, "Procedural hair cards: " + style)
