"""Face quality module (mouth, teeth, tongue, expressions). Imported by base_humans.py / bake_skin.py after chr_lib.

What it owns:
  build_dentition()    lays out 28 self-built crowns (incisors, canines, premolars, molars per quadrant; Wheeler
  dentition_mesh()     crown sizes, labial inclination, incisal edges / cusps) in contact along a regular arch: the CC0
  make_teeth_object()  teeth proxy (fitted to any MPFB body) gives the arch and the occlusal curve, the arch is narrowed
                       smoothly where the body's mouth bag is narrower, the lower arch is derived from the upper one
                       (1.3 mm overbite, posterior buccal cusps in the upper fossae, relaxed rest gap) and cleared by one
                       rigid move. Scalloped gum band with papillae, palate vault, floor of mouth. Upper set skinned
                       100 % to `head`, lower set 100 % to `jaw`; the proxy's jaw-key deltas become RIGID motions (Kabsch
                       fit) of the lower set, so the teeth never bend and the upper teeth never move with any face
                       key. Customisation morphs (cust_*, cust_lib.py) move each set by one affine map that follows the
                       customised mouth bag + lips (cust_dentition_follow).
  mouth_textures()     procedural albedo / roughness atlas for teeth + gums (incisal translucency, cervical warmth,
                       proximal / lingual darkening, depth-in-mouth darkening of the posterior teeth).
  fit_tongue(),        CC0 tongue moved to the new lower crowns (all keys kept) and a graded wet texture (darker root /
  tongue_look()        underside) with an ORM (occlusion, matt back); tongue_jaw_flatten(): the dorsum drops as the jaw
                       opens; subdivide_keyed(): one Catmull-Clark level with every key.
  mouth_occlusion()    the skin's glTF occlusion = the mouth-bag shade (engines have no occlusion inside the mouth).
  lining_ref_image()   the lid-independent albedo the eye-socket lining test reads (face keys never depend on the lid
                       texture); lid_blink_relax(): lower lid / canthus triangles never open > LID_RELAX_MAX in a blink.
  mouth_cavity_depth() per-vertex depth of the body's mouth bag behind the lips (bake_skin.py darkens the skin albedo
                       and roughness with it: an open mouth does not look lit or hollow).
  teeth_collision_correctives()  per face key, lips / cheek lining that the key drives into a crown are pushed out.
  improve_face_keys()  live-stage: lip seal at rest, better smiles / frowns, extra (non-ARKit) expression shapes, lid /
                       eyeball correction for the squints, see its docstring.
"""
import bpy, os, math, json, time
import numpy as np
from mathutils import Vector, Matrix
from chr_lib import (log, PRESETS, TEX, find_part, children_meshes, pbr_material, set_material, FACE_KEYS,
                     EXTRA_FACE_KEYS, _vertex_uv_colors)

MM = 0.001

# ------------------------------------------------------------------------------------------------------------------
# small geometry helpers


def verts_np(obj, key=None):
    me = obj.data
    a = np.empty(len(me.vertices) * 3)
    (key.data if key is not None else me.vertices).foreach_get("co", a)
    return a.reshape(-1, 3)


def mesh_islands(obj):
    me = obj.data
    n = len(me.vertices)
    e = np.empty(len(me.edges) * 2, dtype=np.int64); me.edges.foreach_get("vertices", e); e = e.reshape(-1, 2)
    par = np.arange(n)

    def find(x):
        while par[x] != x:
            par[x] = par[par[x]]; x = par[x]
        return x
    for a, b in e:
        ra, rb = find(a), find(b)
        if ra != rb:
            par[ra] = rb
    return np.array([find(i) for i in range(n)])


def kabsch(A, B):
    """Rigid (R, t) minimising |R a + t - b| over the rows of A -> B."""
    ca, cb = A.mean(0), B.mean(0)
    H = (A - ca).T @ (B - cb)
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1, 1, d])
    R = Vt.T @ D @ U.T
    return R, cb - R @ ca


def group_weights(obj, name):
    n = len(obj.data.vertices)
    w = np.zeros(n)
    if name not in obj.vertex_groups:
        return w
    gi = obj.vertex_groups[name].index
    for v in obj.data.vertices:
        for g in v.groups:
            if g.group == gi:
                w[v.index] = g.weight
    return w


def smoothstep(e0, e1, x):
    t = np.clip((np.asarray(x, dtype=float) - e0) / (e1 - e0), 0, 1)
    return t * t * (3 - 2 * t)


# ------------------------------------------------------------------------------------------------------------------
# lip landmarks on the body (Blender space: +Z up, the character faces -Y, +X = character's left)


def lip_landmarks(body, co=None):
    """Stomion height and the lips' inner / outer surfaces on the midline, measured on the (baked) body mesh."""
    me = body.data
    n = len(me.vertices)
    if co is None:
        co = verts_np(body)
    nr = np.empty(n * 3); me.vertices.foreach_get("normal", nr); nr = nr.reshape(-1, 3)
    mid = np.abs(co[:, 0]) < 0.0015
    face_front = co[mid & (nr[:, 1] < -0.2), 1].min()                # nose tip / lips: most forward midline point
    # outer lip surface: front-facing midline verts within 2.5 cm of the most forward lip point
    lipz = co[mid & (nr[:, 1] < -0.2) & (co[:, 1] < face_front + 0.03)]
    down = mid & (nr[:, 1] < -0.2) & (nr[:, 2] < -0.3)                # undersides (upper lip, nose)
    up = mid & (nr[:, 1] < -0.2) & (nr[:, 2] > 0.3)                   # upper faces (lower lip, chin)
    # the stomion is the lowest upper-lip underside vertex with a lower-lip top vertex just below it
    best = None
    for i in np.nonzero(down)[0]:
        below = np.nonzero(up & (co[:, 2] < co[i, 2]) & (co[:, 2] > co[i, 2] - 0.0025) &
                           (np.abs(co[:, 1] - co[i, 1]) < 0.004))[0]
        if len(below) and (best is None or co[i, 2] < best[0]):
            j = below[np.argmax(co[below, 2])]
            best = (co[i, 2], (co[i] + co[j]) / 2)
    st = best[1]
    inner = mid & (nr[:, 1] > 0.7) & (co[:, 1] < st[1] + 0.012) & (co[:, 1] > st[1] - 0.004)
    iu = inner & (co[:, 2] > st[2]) & (co[:, 2] < st[2] + 0.012)
    il = inner & (co[:, 2] < st[2]) & (co[:, 2] > st[2] - 0.010)
    ou = mid & (nr[:, 1] < -0.5) & (co[:, 2] > st[2]) & (co[:, 2] < st[2] + 0.012)
    return dict(stomion=st, z_st=float(st[2]), y_inner_upper=float(co[iu, 1].min()), y_inner_lower=float(co[il, 1].min()),
                y_outer_upper=float(co[ou, 1].min()))


# ------------------------------------------------------------------------------------------------------------------
# dentition specification (mm): name, mesiodistal width W, labiolingual depth D, crown height H, labial inclination
# of the long axis (deg, crown tip forward), crown type. Standard adult crown dimensions (Wheeler), no third molars.
DENT = {
    "upper": [("CI", 8.6, 7.0, 10.5, 17, "inc"), ("LI", 6.6, 6.0, 9.0, 15, "inc"), ("C", 7.6, 8.0, 10.0, 9, "can"),
              ("PM1", 7.0, 9.0, 8.5, 4, "pm"), ("PM2", 6.6, 9.0, 7.8, 2, "pm"), ("M1", 10.4, 11.0, 7.5, 0, "mol"),
              ("M2", 9.8, 11.0, 7.0, 0, "mol")],
    "lower": [("CI", 5.3, 5.7, 9.0, 18, "inc"), ("LI", 5.9, 6.1, 9.5, 16, "inc"), ("C", 6.9, 7.5, 11.0, 9, "can"),
              ("PM1", 7.0, 7.5, 8.5, 5, "pm"), ("PM2", 7.1, 8.0, 8.0, 3, "pm"), ("M1", 11.2, 10.5, 7.5, 0, "mol"),
              ("M2", 10.5, 10.0, 7.0, 0, "mol")],
}
# ring profiles along the crown, f = distance from the incisal edge / cusp tips in crown heights (1 = cemento-enamel
# junction, 1.3 = root stub hidden in the gum). A / B: half width / half depth factors, C: labial shift of the ring
# centre (x D/2), P: superellipse exponent.
PROFILES = {
    "inc": dict(f=[0.0, 0.05, 0.18, 0.42, 0.74, 1.0, 1.3], A=[.86, .95, 1.0, .98, .86, .70, .60],
                B=[.13, .30, .54, .80, .98, .92, .80], C=[.42, .32, .18, .05, 0, 0, 0],
                P=[3.2, 3.0, 2.6, 2.3, 2.1, 2.0, 2.0], M=12),
    "can": dict(f=[0.0, 0.06, 0.20, 0.44, 0.74, 1.0, 1.3], A=[.80, .90, .98, .98, .86, .70, .60],
                B=[.30, .50, .74, .94, 1.0, .90, .80], C=[.30, .22, .12, .03, 0, 0, 0],
                P=[2.4, 2.4, 2.3, 2.2, 2.0, 2.0, 2.0], M=12),
    "pm": dict(f=[0.0, 0.10, 0.34, 0.70, 1.0, 1.3], A=[.78, .92, 1.0, .92, .72, .64],
               B=[.74, .90, 1.0, .98, .86, .78], C=[0, 0, 0, 0, 0, 0], P=[2.5, 2.5, 2.4, 2.2, 2.0, 2.0], M=12),
    "mol": dict(f=[0.0, 0.06, 0.16, 0.50, 1.0, 1.3], A=[.78, .90, .96, 1.0, .84, .76], B=[.76, .90, .96, 1.0, .88, .80],
                C=[0, 0, 0, 0, 0, 0], P=[3.0, 2.9, 2.7, 2.5, 2.2, 2.2], M=12),
}


# upper incisors / canines: fuller cervical halves (Wheeler: the maxillary central is ~7.0 mm wide at the cervical line
# for 8.6 mm at the contacts = 0.81, the canine 5.5 / 7.6 = 0.72). The shared profile's 0.70 / 0.86 tapered the upper
# necks into wide V embrasures, and a toothy smile (the upper lip hides the gum band) showed the papillae that fill them
# as long pink 'fangs' between the crowns (round 5, web viewer). The lower incisors keep the narrow-necked profile.
NECK_UPPER = {"inc": [.86, .95, 1.0, .99, .94, .82, .72], "can": [.80, .90, .98, .98, .91, .77, .66]}


def profile(tp, jaw=None):
    """Ring profile of a crown type (PROFILES), with the upper jaw's fuller anterior necks (NECK_UPPER)."""
    pr = PROFILES[tp]
    if jaw == "upper" and tp in NECK_UPPER:
        pr = dict(pr, A=NECK_UPPER[tp])
    return pr


def _superellipse(M, p):
    """M points of a unit superellipse |x|^p + |y|^p = 1, theta = 0 on +y (labial), counter-clockwise."""
    th = np.arange(M) / M * 2 * np.pi
    s, c = np.sin(th), np.cos(th)
    x = np.sign(s) * np.abs(s) ** (2 / p)
    y = np.sign(c) * np.abs(c) ** (2 / p)
    return th, x, y


def tooth_local(tp, W, D, H, jaw=None):
    """Local crown mesh: rows of rings (u = mesiodistal, v = labial, w = towards the root, w = 0 at the edge / tips)
    plus a centre vertex closing the occlusal / incisal end. Returns (verts (N,3), faces, per-vertex (f, theta))."""
    pr = profile(tp, jaw)
    M = pr["M"]
    V, FT = [], []
    rings = []
    for k, f in enumerate(pr["f"]):
        th, x, y = _superellipse(M, pr["P"][k])
        u = x * pr["A"][k] * W / 2
        v = y * pr["B"][k] * D / 2 + pr["C"][k] * D / 2
        w = np.full(M, f * H)
        un = x                                          # -1 .. 1 across the width
        vn = y
        if tp == "can" and k <= 3:                      # pointed cusp: the cusp slopes fade over 4 rings, and each
            # ring stays below the previous one (a smaller factor on ring 1 folded the slopes over the tip ring)
            w = w + np.abs(un) ** 1.3 * CANINE_CUSP * H * (1.0, 1.0, 0.55, 0.15)[k]
        if k <= 1:                                      # occlusal relief on the two top rings
            if tp == "inc":                             # square mesial corner, rounded distal corner (+u = distal)
                w = w + (np.abs(un) ** 4) * np.where(un > 0, 0.9, 0.2) * MM
            elif tp == "pm":                            # buccal and lingual cusp tips, marginal ridges lower
                w = w + (np.abs(un) ** 2) * 0.55 * MM + (1 - np.abs(vn)) * 0.35 * MM
            elif tp == "mol":                           # four cusp tips on the diagonals
                w = w + (np.abs(np.abs(un) - np.abs(vn))) * 0.55 * MM
        rings.append(len(V))
        for i in range(M):
            V.append((u[i], v[i], w[i])); FT.append((f, th[i] / (2 * np.pi)))
    centre = len(V)
    cdep = {"inc": 0.25, "can": 0.0, "pm": 1.6, "mol": 1.9}[tp] * MM
    cv = pr["C"][0] * D / 2
    V.append((0.0, cv, cdep)); FT.append((0.0, 0.5))
    F = []
    for k in range(len(rings) - 1):
        a, b = rings[k], rings[k + 1]
        for i in range(M):
            j = (i + 1) % M
            F.append((a + i, a + j, b + j, b + i))
    for i in range(M):
        F.append((centre, rings[0] + (i + 1) % M, rings[0] + i))
    return np.array(V), F, np.array(FT)


# ------------------------------------------------------------------------------------------------------------------
# arch curves


class Arch:
    """Symmetric dental arch in plan: x(s) odd, y(s) even polynomials of a signed parameter s (0 = midline), fitted
    to guide points ordered front -> back on the +X side. Arc length is tabulated so teeth can be laid out in mm."""

    def __init__(self, pts):
        P = np.asarray(pts, dtype=float)
        s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(P[:, :2], axis=0), axis=1))]) + P[0, 0]
        S = np.concatenate([-s[::-1], s]); X = np.concatenate([-P[::-1, 0], P[:, 0]]); Y = np.concatenate([P[::-1, 1], P[:, 1]])
        Z = np.concatenate([P[::-1, 2], P[:, 2]])
        self.ax = np.linalg.lstsq(np.stack([S, S ** 3, S ** 5], 1), X, rcond=None)[0]
        self.ay = np.linalg.lstsq(np.stack([S ** 0, S ** 2, S ** 4], 1), Y, rcond=None)[0]
        self.az = np.linalg.lstsq(np.stack([S ** 0, S ** 2, S ** 4], 1), Z, rcond=None)[0]
        self.smax = s[-1] * 1.15
        t = np.linspace(0, self.smax, 2000)
        xy = np.stack([self.x(t), self.y(t)], 1)
        self.tab_t = t
        self.tab_len = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))])

    def x(self, s):
        return self.ax[0] * s + self.ax[1] * s ** 3 + self.ax[2] * s ** 5

    def y(self, s):
        return self.ay[0] + self.ay[1] * s ** 2 + self.ay[2] * s ** 4

    def z(self, s):
        return self.az[0] + self.az[1] * s ** 2 + self.az[2] * s ** 4

    def s_at(self, length):
        return np.interp(length, self.tab_len, self.tab_t)

    def length_at(self, s):
        return np.interp(s, self.tab_t, self.tab_len)

    def frame(self, s, eps=1e-5):
        """point (x, y) on the arch, unit tangent (distal) and unit labial normal (outwards), both horizontal."""
        p = np.array([self.x(s), self.y(s), 0.0])
        d = np.array([self.x(s + eps) - self.x(s - eps), self.y(s + eps) - self.y(s - eps), 0.0])
        T = d / np.linalg.norm(d)
        N = np.array([T[1], -T[0], 0.0])                # tangent rotated -90 deg: points out of the arch
        if N[1] > 0 and abs(s) < 1e-4:
            N = -N
        return p, T, N


# ------------------------------------------------------------------------------------------------------------------
# guide analysis


def analyse_guide(guide):
    """Crowns / gums of the CC0 teeth proxy: per jaw the crown centroids of the +X side ordered front -> back, the
    vertex indices of the whole jaw (for the rigid key fit)."""
    co = verts_np(guide)
    lab = mesh_islands(guide)
    jw = group_weights(guide, "jaw")
    out = {"upper": dict(crowns=[], idx=[]), "lower": dict(crowns=[], idx=[])}
    for isl in np.unique(lab):
        sel = np.nonzero(lab == isl)[0]
        P = co[sel]
        ext = P.max(0) - P.min(0)
        jaw = "lower" if jw[sel].mean() > 0.5 else "upper"
        out[jaw]["idx"].extend(sel.tolist())
        if len(sel) >= 8 and ext.max() < 0.014:        # a single crown (gum halves span > 2.5 cm)
            out[jaw]["crowns"].append(P.mean(0))
    for jaw in out:
        C = np.array(out[jaw]["crowns"])
        C = C[C[:, 0] > 0.0005]
        C = C[np.argsort(C[:, 1])]                       # front (most -Y) first
        out[jaw]["crowns"] = C
        out[jaw]["idx"] = np.array(sorted(out[jaw]["idx"]))
    return out




# ------------------------------------------------------------------------------------------------------------------
# dentition build
N_SLOTS = 14                                             # texture slots: 7 upper + 7 lower tooth types
CANINE_CUSP = 0.15      # canine cusp slope height (x crown height); 0.26 read as fangs in a laugh (user item 10)
GUM_ROWS = 8                                             # vestibule .. lingual deep (see _gum_rows)


def _place_teeth(arch, jaw, z_edge, scale, rise, zfun=None):
    """Crowns laid along `arch` from the midline (both sides), in contact (each tooth's widest ring touches its
    neighbours). Each tooth: world verts, faces, (f, theta) per vertex, its frame (origin on the edge, T distal,
    N labial, ax towards the root) and arch lengths L0..L1. The edge / cusp-tip height is z_edge + rise x
    (L / total)^1.6 (the occlusal curve), or zfun(rank, arch fraction) when given."""
    teeth, L = [], 0.0
    total = sum(t[1] for t in DENT[jaw]) * scale * MM
    for ti, (name, W, D, H, incl, tp) in enumerate(DENT[jaw]):
        W, D, H = W * scale * MM, D * scale * MM, H * scale * MM
        Lc = L + W / 2
        ze = zfun(ti, Lc / total) if zfun else z_edge + rise * (Lc / total) ** 1.6
        for sg in (1, -1):
            s = float(arch.s_at(Lc)) * sg
            p, T, N = arch.frame(s)
            if sg < 0:
                T = -T                                   # distal is -s on the right side
            upv = np.array([0, 0, 1.0])
            tip = math.radians(incl)
            ax = (upv if jaw == "upper" else -upv) * math.cos(tip) - N * math.sin(tip)
            origin = np.array([p[0], p[1], ze])
            Vl, F, FT = tooth_local(tp, W, D, H, jaw)
            Vw = origin + np.outer(Vl[:, 0], T) + np.outer(Vl[:, 1], N) + np.outer(Vl[:, 2], ax)
            teeth.append(dict(name=name, jaw=jaw, side=sg, tp=tp, W=W, D=D, H=H, V=Vw, F=F, FT=FT, origin=origin,
                              T=T, N=N, ax=ax, L0=L, L1=L + W, slot=ti + (0 if jaw == "upper" else 7), rank=ti))
        L += W
    return teeth


def _inside_tooth(t, P):
    """Per point: inside the analytic crown volume of tooth t (superellipse rings interpolated along the axis)."""
    pr = profile(t["tp"], t["jaw"])
    d = P - t["origin"]
    u, v, w = d @ t["T"], d @ t["N"], d @ t["ax"]
    f = w / t["H"]
    A = np.interp(f, pr["f"], pr["A"]) * t["W"] / 2
    B = np.interp(f, pr["f"], pr["B"]) * t["D"] / 2
    C = np.interp(f, pr["f"], pr["C"]) * t["D"] / 2
    p = np.interp(f, pr["f"], pr["P"])
    return (f > 0.0) & (f < 1.3) & ((np.abs(u / A) ** p + np.abs((v - C) / B) ** p) < 1.0)


def _gum_side(arch, q, jaw, hub):
    """One side (+X) of a gum: columns along the arch, GUM_ROWS rows across the ridge, then rows converging on the
    `hub` (palate vault / floor of the mouth). Returns (verts, faces (quads/tris), uvs, is_hub_vertex)."""
    total = q[-1]["L1"]
    cols = []
    for t in q:
        n = 6 if t["tp"] in ("inc", "can") else 3
        for i in range(n):
            cols.append((t["L0"] + (t["L1"] - t["L0"]) * i / n, t, -1 + 2 * i / n))
    cols.append((total, q[-1], 1.0))
    cols.append((total + 2.0 * MM, q[-1], 1.0))
    V, UV, PUSH = [], [], []
    extra = [0.4, 0.75]                                  # fractions towards the hub (hub itself = 1)
    nrow = GUM_ROWS + len(extra)
    for ci, (L, t, un) in enumerate(cols):
        s = float(arch.s_at(min(L, arch.tab_len[-1])))
        p, T, N = arch.frame(s)
        ant = t["tp"] in ("inc", "can")
        # interdental papilla up to the contact area (user item 10: the old 3.3 / 1.9 mm left dark 'black triangle'
        # embrasures between every crown, the teeth read as separated pegs): tip at ~40 % of the crown height from the
        # incisal edge on the front teeth, ~55 % behind
        pap = (0.60 if ant else 0.45) * t["H"] - 0.7 * MM
        pr = PROFILES[t["tp"]]
        Bc = np.interp(1.0, pr["f"], pr["B"]) * t["D"] / 2
        a = min(abs(un), 0.985)
        half = Bc * (0.12 + 0.88 * math.sqrt(max(0.0, 1 - a ** 2.4)))
        if L > total:
            half = Bc * 0.5
        # this column's point on the edge line: arch point + the placement shift interpolated between tooth centres
        sh = [np.interp(L, [(u["L0"] + u["L1"]) / 2 for u in q], [u["_shift"][i] for u in q]) for i in range(3)]
        zc = np.interp(L, [(u["L0"] + u["L1"]) / 2 for u in q], [u["origin"][2] for u in q])
        base = np.array([p[0] + sh[0], p[1] + sh[1], zc])
        ax = t["ax"]
        H = t["H"]
        # the rows away from the crowns follow the crown axis / height / depth interpolated between tooth centres, so
        # the band has no zig-zag where the labial inclination changes from one tooth to the next
        cen = [(u["L0"] + u["L1"]) / 2 for u in q]
        axs = np.array([np.interp(L, cen, [u["ax"][i] for u in q]) for i in range(3)]); axs /= np.linalg.norm(axs)
        Hs = np.interp(L, cen, [u["H"] for u in q])
        Bcs = np.interp(L, cen, [np.interp(1.0, PROFILES[u["tp"]]["f"], PROFILES[u["tp"]]["B"]) * u["D"] / 2 for u in q])
        margin = (H - 0.7 * MM) - pap * (abs(un) ** 1.7)
        back = smoothstep(total - 4 * MM, total + 2 * MM, L)            # taper the band behind the last molar
        R = [base + axs * (Hs + (5.5 - 2.5 * back) * MM) + N * (Bcs + (1.2 - 0.8 * back) * MM),  # vestibule (behind the lip)
             base + axs * (Hs + (3.0 - 1.2 * back) * MM) + N * (Bcs + (1.2 - 0.6 * back) * MM),  # attached gingiva
             base + ax * (margin + 1.1 * MM) + N * (half + 0.75 * MM), # marginal bulge
             base + ax * margin + N * (half + 0.12 * MM),              # free margin hugging the crown
             base + ax * (H + 0.8 * MM),                               # crest (inside / between the crowns)
             base + ax * margin - N * (half + 0.12 * MM),              # lingual margin
             base + ax * (margin + 1.4 * MM) - N * (half + 0.9 * MM),
             base + axs * (Hs + 4.5 * MM) - N * (Bcs + 2.4 * MM)]      # lingual deep
        last = R[-1]
        for fr in extra:
            P_ = last * (1 - fr) + hub * fr
            P_[2] += (1 if jaw == "upper" else -1) * math.sin(fr * math.pi) * 2.5 * MM
            R.append(P_)
        d = min(1.0, L / (total + 2.0 * MM))
        for r, P_ in enumerate(R):
            V.append(P_)
            g = r / (GUM_ROWS - 1) * 0.7 if r < GUM_ROWS else 0.7 + 0.3 * extra[r - GUM_ROWS]
            UV.append((0.5 + 0.5 * g * 0.999, d))
            # direction a buried vertex is moved to get back inside the mouth bag: labial rows go lingual
            PUSH.append(-N if r <= 3 else (N if 5 <= r <= 7 else np.zeros(3)))
    ncol = len(cols)
    F = []
    for c in range(ncol - 1):
        for r in range(nrow - 1):
            a0, a1 = c * nrow + r, (c + 1) * nrow + r
            F.append((a0, a0 + 1, a1 + 1, a1))
    hub_i = len(V)
    V.append(hub); UV.append((1.0 - 1e-3, 0.6)); PUSH.append(np.zeros(3))
    for c in range(ncol - 1):
        F.append((c * nrow + nrow - 1, hub_i, (c + 1) * nrow + nrow - 1))
    return np.array(V), F, np.array(UV), ncol, nrow, np.array(PUSH)


def _mirror_merge(V, F, UV, ncol, nrow, PUSH):
    """Mirror a +X gum side to -X and weld the midline column (column 0) and the hub (last vertex)."""
    n = len(V)
    Vm = V.copy(); Vm[:, 0] *= -1
    remap = np.arange(n) + n
    remap[:nrow] = np.arange(nrow)                        # midline column shared
    remap[n - 1] = n - 1                                  # hub shared
    Fm = [tuple(int(remap[i]) for i in reversed(f)) for f in F]
    keep = np.ones(n, bool); keep[:nrow] = False; keep[n - 1] = False
    # compact: vertices of the mirrored copy that survive
    newidx = -np.ones(2 * n, dtype=np.int64); newidx[:n] = np.arange(n)
    k = n
    for i in range(n):
        if keep[i]:
            newidx[n + i] = k; k += 1
    Vall = np.concatenate([V, Vm[keep]]); UVall = np.concatenate([UV, UV[keep]])
    Pm = PUSH.copy(); Pm[:, 0] *= -1
    Pall = np.concatenate([PUSH, Pm[keep]])
    Pall[:nrow, 0] = 0.0                                  # midline column pushes stay in the mirror plane
    Fall = list(F) + [tuple(int(newidx[i]) for i in f) for f in Fm]
    return Vall, Fall, UVall, Pall


def body_tris(obj, co=None):
    me = obj.data
    me.calc_loop_triangles()
    t = np.empty(len(me.loop_triangles) * 3, dtype=np.int64); me.loop_triangles.foreach_get("vertices", t)
    return (verts_np(obj) if co is None else co), t.reshape(-1, 3)


def buried(Pb, Tb, Q):
    """Points of Q inside the body flesh (ray parity towards the face, Blender forward = -Y)."""
    from morph_check import inside_flesh
    if len(Q) == 0:
        return np.zeros(0, bool)
    lo, hi = Q.min(0) - 0.025, Q.max(0) + 0.025
    lo[1] = -np.inf                                      # keep everything AHEAD of the points: a ray from a molar
    return inside_flesh(Pb, Tb, Q, fwd=1, sign=-1.0, region=(lo, hi))   # must still meet the lips / face skin


def _bag_needs(q, Pb, Tb, cap=5.0 * MM):
    """Per tooth type of the +X side (front -> back): how far (m) the crown has to move lingually (-N) until the
    labial / buccal enamel of its occlusal 60 % clears the lip / cheek flesh of the body (only sideways contact is
    fixed this way: crown necks that reach above the mouth bag's roof are hidden by the gum anyway). Molars may rest
    1.5 mm into the cheek lining (real cheeks lie against the buccal faces of the molars)."""
    out = []
    for t in [t for t in q if t["side"] > 0]:
        P = t["V"][(t["FT"][:, 0] < 0.6) & ((t["V"] - t["origin"]) @ t["N"] > 0)]
        if t["tp"] == "mol":
            P = P - t["N"] * 1.5 * MM
        e = 0.0
        while e < cap and buried(Pb, Tb, P - t["N"] * e).any():
            e += 0.25 * MM
        out.append(e)
    return np.array(out)


def _arch_points(arch, lengths, offsets, shift=np.zeros(3)):
    """Points on `arch` at the given arch lengths, moved lingually by `offsets` (m) and by `shift`: control points
    for a narrower / derived arch."""
    pts = []
    for L, o in zip(lengths, offsets):
        p, T, N = arch.frame(float(arch.s_at(L)))
        pts.append(p - N * o + shift)
    return np.array(pts)


def _centres(q):
    return np.array([(t["L0"] + t["L1"]) / 2 for t in q if t["side"] > 0])


def build_dentition(kind, guide, body):
    """Lay out the crowns for this body. Returns the plan (teeth, arches, landmarks).
    Upper arch: the CC0 proxy's arch (fitted to any MPFB body), refitted narrower where the body's mouth bag is
    narrower (a smooth, front-to-back non-decreasing inset, so the arch stays a regular curve and the crowns stay in
    contact); upper incisors 1.1 mm behind the upper lip, edges 1.6 / 2.0 mm below the stomion (incisal display).
    Lower arch: derived from the upper one (centre line inset: the posterior buccal cusps sit in the upper fossae),
    overbite 1.3 mm on the incisors, the posterior teeth 0.3 mm apart (relaxed rest bite), then moved as ONE rigid
    set (back, then down) until no crown touches an upper crown: no per-tooth steps in the lower arch."""
    G = analyse_guide(guide)
    lm = lip_landmarks(body)
    Pb, Tb = body_tris(body)
    female = PRESETS[kind].get("tex_kind", kind) == "female"   # proportion variants (cust_lib) use their base kind
    a0 = Arch(G["upper"]["crowns"][:7])
    c5 = G["upper"]["crowns"][5]
    k = int(np.argmin((a0.x(a0.tab_t) - c5[0]) ** 2 + (a0.y(a0.tab_t) - c5[1]) ** 2))
    std_m1 = (sum(t[1] for t in DENT["upper"][:5]) + DENT["upper"][5][1] / 2) * MM
    scale = float(np.clip(a0.tab_len[k] / std_m1, 0.85, 1.12))
    drop = (2.0 if female else 1.6) * MM                 # upper incisal edge below the stomion
    z_up = lm["z_st"] - drop
    tot_u = sum(t[1] for t in DENT["upper"]) * scale * MM
    tot_l = sum(t[1] for t in DENT["lower"]) * scale * MM
    # occlusal curve: the proxy's crowns rise towards the back (the occlusal plane tilts up posteriorly, ~5 mm over
    # the arch); the new edges / cusp tips follow that rise, the central incisors' edge sits at z_up
    zg = lambda f: float(a0.z(a0.s_at(f * tot_u)))
    f_ci = DENT["upper"][0][1] * scale * MM / 2 / tot_u
    zfun_u = lambda rank, f: z_up + zg(f) - zg(f_ci)

    def shift(t, sh):
        t["_shift"] = t.get("_shift", np.zeros(3)) + sh
        t["V"] = t["V"] + sh; t["origin"] = t["origin"] + sh

    # ---- upper: fit into the mouth bag by narrowing the arch itself
    au, inset_u, pts = a0, np.zeros(7), np.array(G["upper"]["crowns"][:7], dtype=float)
    for it in range(7):
        up = _place_teeth(au, "upper", z_up, scale, 0.0, zfun_u)
        ci = [t for t in up if t["name"] == "CI"]
        fwd_u = (lm["y_inner_upper"] + 1.1 * MM) - min(t["V"][:, 1].min() for t in ci)
        for t in up:
            shift(t, np.array([0, fwd_u, 0.0]))
        m = _bag_needs(up, Pb, Tb) if it < 6 else np.zeros(7)
        if m.max() == 0:
            break
        m = np.maximum.accumulate(m)
        inset_u += m
        Lg = [au.tab_len[int(np.argmin((au.x(au.tab_t) - p[0]) ** 2 + (au.y(au.tab_t) - p[1]) ** 2))] for p in pts]
        pts = _arch_points(au, Lg, np.interp(Lg, _centres(up), m))
        au = Arch(pts)
    # ---- lower: centre line derived from the upper arch
    ob = np.array([1.3, 1.3, 0.9, -0.3, -0.3, -0.3, -0.3]) * MM             # lower edge above the upper edge (+)
    zfun = lambda rank, f: zfun_u(rank, f) + ob[rank]
    off = np.array([2.0, 2.0, 2.2, 2.4, 2.4, 2.4, 2.4]) * scale * MM        # centre-line inset under the upper arch
    extra = np.zeros(7)
    for it in range(7):
        Ll = np.cumsum([t[1] * scale * MM for t in DENT["lower"]]) - np.array([t[1] for t in DENT["lower"]]) * scale * MM / 2
        pl = _arch_points(au, Ll / tot_l * tot_u, off + extra, np.array([0, fwd_u, 0.0]))
        al = Arch(pl)
        lo = _place_teeth(al, "lower", 0.0, scale, 0.0, zfun)
        m = _bag_needs(lo, Pb, Tb) if it < 6 else np.zeros(7)
        if m.max() == 0:
            break
        extra += np.maximum.accumulate(m)
    # the whole lower set moves rigidly: back until the anterior crowns clear the upper ones, then down
    def hits(sel):
        for t in lo:
            if sel(t) and any(_inside_tooth(u_, t["V"]).any() or _inside_tooth(t, u_["V"]).any() for u_ in up
                              if np.linalg.norm(u_["origin"][:2] - t["origin"][:2]) < 0.02):
                return True
        return False
    back = down = 0.0
    while back < 4 * MM and hits(lambda t: t["tp"] in ("inc", "can")):
        for t in lo:
            shift(t, np.array([0, 0.1 * MM, 0]))
        back += 0.1 * MM
    while down < 4 * MM and hits(lambda t: True):
        for t in lo:
            shift(t, np.array([0, 0, -0.1 * MM]))
        down += 0.1 * MM
    for t in lo:                                         # the gum band follows the lower set's shift
        t["_shift"] = t.get("_shift", np.zeros(3))
    for t in up:
        t["_shift"] = np.array([0, fwd_u, 0.0])
    lo_bag = int(sum(buried(Pb, Tb, t["V"][t["FT"][:, 0] < 1.0]).sum() for t in lo if t["tp"] != "mol"))
    log("dentition %s: stomion z %.4f, scale %.3f, upper edge %.1f mm below the stomion, upper arch forward %.1f mm, "
        "inset into the mouth bag (mm, CI..M2) upper %s lower %s, lower set moved back %.1f mm / down %.1f mm to clear "
        "the upper crowns, lower non-molar enamel vertices in the flesh: %d" % (
            kind, lm["z_st"], scale, drop * 1e3, -fwd_u * 1e3, np.round(inset_u * 1e3, 1).tolist(),
            np.round(extra * 1e3, 1).tolist(), back * 1e3, down * 1e3, lo_bag))
    # the lower gum band is laid along the lower arch shifted like the crowns
    al_shift = np.array([0, back, -down])
    return dict(teeth=up + lo, arches={"upper": au, "lower": al}, guide=G, lm=lm, scale=scale, z_up=z_up,
                z_lo=z_up + ob[0], Pb=Pb, Tb=Tb, lower_shift=al_shift)


def dentition_mesh(plan):
    """Assemble crowns + gums into mesh arrays with UVs and a jaw flag per vertex (1 = lower)."""
    V, F, UV, JAW, PUSH = [], [], [], [], []
    for t in plan["teeth"]:
        o = len(V)
        V.extend(t["V"].tolist()); PUSH.extend([(0.0, 0.0, 0.0)] * len(t["V"]))
        f_, th = t["FT"][:, 0], t["FT"][:, 1]
        u = 0.5 * np.clip(f_ / 1.3, 0, 1) * 0.998
        v = (t["slot"] + 0.02 + 0.96 * th) / N_SLOTS
        UV.extend(zip(u, v))
        JAW.extend([1 if t["jaw"] == "lower" else 0] * len(t["V"]))
        F.extend([tuple(i + o for i in f) for f in t["F"]])
    for jaw in ("upper", "lower"):
        q = [t for t in plan["teeth"] if t["jaw"] == jaw and t["side"] > 0]
        m1 = [t for t in q if t["name"] == "M1"][0]
        if jaw == "upper":
            hub = np.array([0.0, m1["origin"][1], m1["origin"][2] + 15.0 * MM])
        else:
            hub = np.array([0.0, m1["origin"][1] - 4 * MM, m1["origin"][2] - 13.0 * MM])
        gv, gf, guv, ncol, nrow, gp = _gum_side(plan["arches"][jaw], q, jaw, hub)
        gv, gf, guv, gp = _mirror_merge(gv, gf, guv, ncol, nrow, gp)
        o = len(V)
        V.extend(gv.tolist()); UV.extend(guv.tolist()); JAW.extend([1 if jaw == "lower" else 0] * len(gv))
        PUSH.extend(gp.tolist())
        F.extend([tuple(i + o for i in f) for f in gf])
    V, PUSH = np.array(V), np.array(PUSH)
    # gum rows buried in the lip / cheek flesh are pulled back inside the mouth bag (smoothly along the band)
    Pb, Tb = plan["Pb"], plan["Tb"]
    has = np.linalg.norm(PUSH, axis=1) > 0.5
    E = set()
    for f in F:
        for i in range(len(f)):
            a, b = f[i], f[(i + 1) % len(f)]
            if has[a] and has[b]:
                E.add((min(a, b), max(a, b)))
    E = np.array(sorted(E))
    amt = np.zeros(len(V))
    for it in range(30):
        B = np.zeros(len(V), bool)
        B[has] = buried(Pb, Tb, (V + PUSH * amt[:, None])[has])
        if not B.any():
            break
        amt[B] += 0.25 * MM
        s_ = np.zeros(len(V)); c_ = np.zeros(len(V))
        np.add.at(s_, E[:, 0], amt[E[:, 1]]); np.add.at(s_, E[:, 1], amt[E[:, 0]])
        np.add.at(c_, E[:, 0], 1); np.add.at(c_, E[:, 1], 1)
        amt = np.maximum(amt, 0.85 * s_ / np.maximum(c_, 1))
    log("gums: %d vertices pulled back into the mouth bag (max %.1f mm, %d still buried)" % (
        int((amt > 0).sum()), amt.max() * 1e3, int(B.sum())))
    V = V + PUSH * amt[:, None]
    return V, F, np.array(UV), np.array(JAW)


def _write_png(path, rgb, alpha=None):
    """float RGB(A) array (h, w, 3) in 0..1, row 0 = top -> PNG via Blender (no PIL inside Blender)."""
    h, w = rgb.shape[:2]
    im = bpy.data.images.new(os.path.basename(path), w, h, alpha=alpha is not None)
    px = np.ones((h, w, 4), dtype=np.float32)
    px[:, :, :3] = rgb[::-1]
    if alpha is not None:
        px[:, :, 3] = alpha[::-1]
    im.pixels.foreach_set(px.ravel())
    im.filepath_raw = path + ".part"; im.file_format = 'PNG'        # temp file + rename: shared texture, atomic install
    im.save()
    os.replace(path + ".part", path)
    bpy.data.images.remove(im)


def _gltf_output_group(nt):
    """The glTF exporter's 'glTF Material Output' group node of node tree nt (created if missing)."""
    for n in nt.nodes:
        if n.type == 'GROUP' and n.node_tree and n.node_tree.name.startswith("glTF Material Output"):
            return n
    ng = bpy.data.node_groups.get("glTF Material Output")
    if ng is None:
        try:
            from io_scene_gltf2.blender.com.material_helpers import create_settings_group
            ng = create_settings_group("glTF Material Output")
        except Exception:
            ng = bpy.data.node_groups.new("glTF Material Output", 'ShaderNodeTree')
            ng.interface.new_socket("Occlusion", socket_type="NodeSocketFloat")
            ng.interface.new_socket("Thickness", socket_type="NodeSocketFloat")
            ng.nodes.new('NodeGroupOutput'); ng.nodes.new('NodeGroupInput')
    g = nt.nodes.new("ShaderNodeGroup"); g.node_tree = ng; g.label = "glTF Material Output"
    return g


def gltf_occlusion(mat):
    """Route R of the material's roughness image (the ORM map of pbr_material's rough_img) into the glTF exporter's
    'glTF Material Output' group: exported as occlusionTexture (same image, texCoord 0)."""
    nt = mat.node_tree
    sep = [n for n in nt.nodes if n.type == 'SEPARATE_COLOR']
    if not sep:
        return False
    nt.links.new(sep[0].outputs[0], _gltf_output_group(nt).inputs["Occlusion"])
    return True


def mouth_occlusion(body, kind):
    """glTF occlusionTexture for the skin (round 5, user item 10): engines / the web viewer have no occlusion inside
    the mouth, so an open mouth showed the inner cheeks and the palate as bright grey walls (the environment light
    and its reflections at grazing angles). R of bake_skin.py's mouth-shade map (<kind>_mouth_shade.png: 1 outside
    the mouth, darker with depth in the mouth bag) becomes the skin's occlusion (its own image: the skin ORM's R holds
    the body AO that the albedo already carries). Engine-stage body (export_face). Returns True when linked."""
    tk = PRESETS.get(kind, {}).get("tex_kind", kind)
    path = os.path.join(TEX, "%s_mouth_shade.png" % tk)
    mats = [m for m in body.data.materials if m and m.node_tree]
    if not os.path.exists(path) or not mats:
        return False
    nt = mats[0].node_tree
    g = _gltf_output_group(nt)
    if g.inputs["Occlusion"].is_linked:
        return False
    im = bpy.data.images.load(path, check_existing=True)
    im.colorspace_settings.name = "Non-Color"
    t = nt.nodes.new("ShaderNodeTexImage"); t.image = im; t.label = "mouth occlusion"; t.location = (-500, -700)
    sep = nt.nodes.new("ShaderNodeSeparateColor"); sep.location = (-200, -700)
    nt.links.new(t.outputs["Color"], sep.inputs[0])
    nt.links.new(sep.outputs[0], g.inputs["Occlusion"])
    log("mouth occlusion on %s: %s" % (mats[0].name, os.path.basename(path)))
    return True


def mouth_textures(size=1024):
    """Teeth + gums atlas (albedo sRGB, roughness in G). Left half: teeth, 14 slots (7 upper then 7 lower tooth types,
    front to back) along V, each slot = crown ring angle (0 labial, 0.25 / 0.75 proximal, 0.5 lingual), U = distance
    from the incisal edge (0) to the root stub (1.3 crown heights). Right half: gums, U = row across the ridge
    (vestibule -> margin -> lingual -> palate / floor), V = depth along the arch (front -> back)."""
    H = W = size
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    u = (xx + 0.5) / W
    v = 1.0 - (yy + 0.5) / H                        # UV v runs bottom -> top
    rgb = np.zeros((H, W, 3)); rough = np.zeros((H, W)); ao = np.ones((H, W))
    rng = np.random.default_rng(7)
    noise = rng.normal(0, 1, (H, W))
    for _ in range(3):                               # smooth value noise (no block / streak pattern)
        noise = (noise + np.roll(noise, 3, 0) + np.roll(noise, -3, 0) + np.roll(noise, 3, 1) + np.roll(noise, -3, 1)) / 5
    noise /= noise.std()
    # ---- teeth ----
    T = u < 0.5
    f = u / 0.5 * 1.3
    slot = np.clip((v * N_SLOTS).astype(int), 0, N_SLOTS - 1)
    th = np.clip(((v * N_SLOTS - slot) - 0.02) / 0.96, 0, 1)
    rank = slot % 7
    anterior = rank <= 2
    enamel = np.array([0.90, 0.87, 0.80])
    edge = np.array([0.70, 0.74, 0.78])             # translucent incisal third (reads blue-grey)
    neck = np.array([0.86, 0.76, 0.58])             # warmer dentine showing through at the neck
    root = np.array([0.72, 0.38, 0.38])
    c = np.broadcast_to(enamel, (H, W, 3)).copy()
    we = np.where(anterior, (1 - smoothstep(0.0, 0.16, f)) * 0.75, 0.0)[..., None]
    c = c * (1 - we) + edge * we
    # posterior occlusal tables: fissures / fossae read slightly darker and warmer (no blue-grey translucency)
    c = c * np.where(anterior, 1.0, 1 - 0.14 * (1 - smoothstep(0.0, 0.08, f)))[..., None]
    wn = (smoothstep(0.55, 1.0, f) * 0.7)[..., None]
    c = c * (1 - wn) + neck * wn
    wr = smoothstep(1.0, 1.06, f)[..., None]
    c = c * (1 - wr) + root * wr
    side = smoothstep(0.03, 0.22, f)                                # no side shading on the occlusal / incisal end
    lingual = 1 - side * 0.42 * (1 - np.abs(np.cos(np.pi * th)))  # th 0 / 1 labial, 0.5 lingual (faces the dark mouth)
    prox = 1 - side * (0.22 * np.exp(-((th - 0.25) / 0.11) ** 2) + 0.22 * np.exp(-((th - 0.75) / 0.11) ** 2))
    # depth in the mouth (engines have no occlusion in there): the buccal corridor and the far side of an open mouth
    # recede into shadow instead of reading as lit grey blocks
    # (user item 10: in the web viewer, which has no occlusion inside the mouth, the premolars / molars read as bright
    # denture blocks at the corners of an open mouth; the buccal corridor is in shadow in a real mouth)
    # (round 5: with the occlusion channel below on top, these made the posterior crowns charcoal grey with metallic
    # highlights in the web viewer: albedo x AO fell to 0.17 on the first molar; milder here and below, warmer)
    depth = np.array([1.0, 0.97, 0.92, 0.82, 0.74, 0.64, 0.57])[rank]
    warm = np.array([1.0, 0.84, 0.62])                             # shadowed enamel goes warm ivory, not steel grey
    shade = lingual * prox * depth * (1 + 0.015 * noise)
    c = c * (warm + (1 - warm) * shade[..., None] ** 0.5)
    rgb[T] = (c * shade[..., None])[T]
    # deeper teeth are rougher: a dark AND glossy albedo reads as metal once the environment reflects in it
    # (round 5: 0.17 + 0.055 x rank still showed chrome-grey premolars / molars at the corners of an open mouth in the
    # web viewer: rougher, and an OCCLUSION channel below)
    rough[T] = (0.2 + 0.075 * rank + 0.08 * smoothstep(0.8, 1.1, f) + 0.01 * noise)[T]
    # occlusion (glTF occlusionTexture = R of the roughness map, ORM packing): engines / three.js darken the ambient and
    # environment light (incl. its reflections) inside the mouth with it; the buccal corridor and the far side of the
    # arch sit in shadow, the lingual faces face the dark mouth
    ao[T] = (np.array([1.0, 0.94, 0.86, 0.72, 0.62, 0.52, 0.45])[rank] * (1 - side * 0.35 * (1 - np.abs(np.cos(np.pi * th))))
             * (1 - 0.3 * smoothstep(0.9, 1.2, f)))[T]
    # ---- gums / palate / floor ----
    Gm = ~T
    g = (u - 0.5) / 0.5
    d = v
    vest = np.array([0.50, 0.21, 0.22]); att = np.array([0.80, 0.53, 0.50]); marg = np.array([0.77, 0.46, 0.45])
    pal = np.array([0.62, 0.36, 0.37])                # coral pink attached gingiva, less saturated (read as red plastic)
    rm = 3 / (GUM_ROWS - 1) * 0.7                    # margin row in g
    c = np.where((g < rm * 0.6)[..., None], vest + (att - vest) * smoothstep(0.0, rm * 0.6, g)[..., None],
                 att + (marg - att) * smoothstep(rm * 0.6, rm, g)[..., None])
    c = np.where((g > rm)[..., None], marg + (pal - marg) * smoothstep(rm, 0.75, g)[..., None], c)
    dd = 1.0 - 0.72 * smoothstep(0.12, 1.0, d)
    deep = 1.0 - 0.55 * smoothstep(0.62, 1.0, g) - 0.25 * (1 - smoothstep(0.0, 0.2, g))
    rgb[Gm] = (c * (dd * deep * (1 + 0.04 * noise))[..., None])[Gm]
    rough[Gm] = (0.4 + 0.12 * smoothstep(0.6, 1.0, g) + 0.03 * noise)[Gm]
    ao[Gm] = ((1.0 - 0.7 * smoothstep(0.1, 0.9, d)) * (1.0 - 0.5 * smoothstep(0.55, 1.0, g)))[Gm]
    rgb = np.clip(rgb, 0, 1)
    base = os.path.join(TEX, "mouth_atlas_base.png"); rp = os.path.join(TEX, "mouth_atlas_rough.png")
    _write_png(base, rgb ** (1 / 1.0))
    R = np.zeros((H, W, 3)); R[:, :, 0] = np.clip(ao, 0.08, 1); R[:, :, 1] = np.clip(rough, 0.05, 1)
    _write_png(rp, R)
    return base, rp


def _ridge_affine(P, Q, W, c, lam=0.1):
    """Weighted least-squares affine map q ~ t + A (p - c), regularised towards A = I (lam x the region's spread), so
    a local, one-sided point set cannot shear or flatten the crowns. Returns X (4, 3): [p - c, 1] @ X."""
    M = np.c_[P - c, np.ones(len(P))]
    s2 = float((W[:, None] * (P - c) ** 2).sum() / max(W.sum(), 1e-12))
    mu = lam * W.sum() * s2
    D = np.diag([1.0, 1.0, 1.0, 0.0]); X0 = np.vstack([np.eye(3), np.zeros((1, 3))])
    return np.linalg.solve(M.T @ (W[:, None] * M) + mu * D, M.T @ (W[:, None] * Q) + mu * D @ X0)


def cust_dentition_follow(ob, body, rig, V, lower, eps=2e-4, sigma=0.008, reach=0.02):
    """Customisation morphs (cust_lib.py: mouth width / height, lips, jaw, chin, head width, age ...) reshape the mouth,
    not the jaw pose. Each set (upper on `head`, lower on `jaw`) RIDES WITH THE MOUTH THAT HOLDS IT: per cust key, a
    weighted, identity-regularised affine map is fitted to how the key moves the body's mouth bag + lips near that set
    (Gaussian weights, sigma 8 mm, within 2 cm of the crowns) and applied to the set. Smooth, linear in the key weight,
    the crowns never bend, and the teeth keep their place behind the customised lips. Returns {key: max move (mm)}."""
    from mathutils.kdtree import KDTree
    bk = body.data.shape_keys.key_blocks
    bb = verts_np(body, bk["Basis"])
    depth = mouth_cavity_depth(body, rig)
    region = (depth >= 0) | (group_weights(body, "lips") > 0.3)
    region[NBODY:] = False
    ridx = np.nonzero(region)[0]
    fits = []
    for sel in (~lower, lower):
        S = V[sel]
        kd = KDTree(len(S))
        for i, q in enumerate(S):
            kd.insert(q, i)
        kd.balance()
        d = np.array([kd.find(bb[i])[2] for i in ridx])
        w = np.exp(-(d / sigma) ** 2) * (d < reach)
        k = w > 1e-3
        fits.append((sel, ridx[k], w[k], S.mean(0)))
    log("dentition customisation: teeth follow %d / %d mouth-bag + lip vertices (upper / lower set)" % (
        len(fits[0][1]), len(fits[1][1])))
    out = {}
    for name in [k.name for k in bk if k.name.startswith("cust_")]:
        D = verts_np(body, bk[name]) - bb
        if np.abs(D[ridx]).max() < eps:
            continue
        Va = V.copy()
        for sel, idx, w, c in fits:
            X = _ridge_affine(bb[idx], bb[idx] + D[idx], w, c)
            Va[sel] = np.c_[V[sel] - c, np.ones(int(sel.sum()))] @ X
        mv = float(np.linalg.norm(Va - V, axis=1).max())
        if mv < 1e-5:
            continue
        sk = ob.shape_key_add(name=name, from_mix=False)
        sk.data.foreach_set("co", Va.ravel()); sk.value = 0.0
        out[name] = round(mv * 1e3, 2)
    return out


def make_teeth_object(rig, kind, guide, body):
    """Build the dentition object from the guide, skin it (upper -> head, lower -> jaw), give the lower set the guide's
    jaw motions as rigid shape keys (face keys) and both sets the customisation morphs (cust_dentition_follow), then
    delete the guide and take its name."""
    import bmesh
    plan = build_dentition(kind, guide, body)
    V, F, UV, JAW = dentition_mesh(plan)
    name = guide.name
    me = bpy.data.meshes.new(name + "_new")
    me.from_pydata(V.tolist(), [], [tuple(f) for f in F])
    me.validate()
    ob = bpy.data.objects.new(name + "_new", me)
    for c in guide.users_collection:
        c.objects.link(ob)
    # outward normals per island, smooth shading
    bm_ = bmesh.new(); bm_.from_mesh(me)
    bmesh.ops.recalc_face_normals(bm_, faces=bm_.faces[:])
    bm_.to_mesh(me); bm_.free()
    me.shade_smooth() if hasattr(me, "shade_smooth") else [setattr(p, "use_smooth", True) for p in me.polygons]
    # UVs per loop; the crown ring seam (theta 1 -> 0) is unwrapped per face
    uvl = me.uv_layers.new(name="UVMap")
    li = np.empty(len(me.loops), dtype=np.int64); me.loops.foreach_get("vertex_index", li)
    uv = UV[li].copy()
    for p in me.polygons:
        ls = list(p.loop_indices)
        vv = uv[ls, 1]
        if vv.max() < 1.0 and uv[ls, 0].max() <= 0.5:          # tooth faces (left half)
            sl = np.floor(vv * N_SLOTS)
            thv = (vv * N_SLOTS - sl - 0.02) / 0.96
            if thv.max() - thv.min() > 0.5:
                thv = np.where(thv < 0.5, thv + 1.0, thv)
                thv = np.minimum(thv, 1.0)
                uv[ls, 1] = (sl.min() + 0.02 + 0.96 * thv) / N_SLOTS
    uvl.data.foreach_set("uv", uv.ravel())
    # skin
    vg_h, vg_j = ob.vertex_groups.new(name="head"), ob.vertex_groups.new(name="jaw")
    vg_h.add(np.nonzero(JAW == 0)[0].tolist(), 1.0, 'REPLACE')
    vg_j.add(np.nonzero(JAW == 1)[0].tolist(), 1.0, 'REPLACE')
    ob.parent = rig
    mod = ob.modifiers.new("Armature", 'ARMATURE'); mod.object = rig
    # rigid jaw keys from the guide's lower set
    gidx = plan["guide"]["lower"]["idx"]
    kb = guide.data.shape_keys.key_blocks if guide.data.shape_keys else []
    gb = verts_np(guide)[gidx]
    lower = JAW == 1
    ob.shape_key_add(name="Basis", from_mix=False)
    fits = {}
    for k in list(kb)[1:]:
        if k.name.startswith("cust_"):
            continue                      # customisation morphs: cust_dentition_follow() below
        gk = verts_np(guide, k)[gidx]
        if np.abs(gk - gb).max() < 2e-4:
            continue
        R, t = kabsch(gb, gk)
        plan.setdefault("rigid", {})[k.name] = (R, t)
        res = np.linalg.norm((gb @ R.T + t) - gk, axis=1).max()
        Vk = V.copy()
        Vk[lower] = V[lower] @ R.T + t
        sk = ob.shape_key_add(name=k.name, from_mix=False)
        sk.data.foreach_set("co", Vk.ravel()); sk.value = 0.0
        ang = math.degrees(math.acos(max(-1, min(1, (np.trace(R) - 1) / 2))))
        fits[k.name] = "%.1fdeg/%.1fmm (res %.1fmm)" % (ang, np.linalg.norm(t + (R - np.eye(3)) @ gb.mean(0)) * 1e3, res * 1e3)
    log("dentition rigid jaw keys:", fits)
    cust = cust_dentition_follow(ob, body, rig, V, lower)
    if cust:
        log("dentition customisation keys (per-set affine, max mm):", cust)
    for key in [k for k in guide.keys() if k.startswith("rts_")]:      # part catalogue properties (cust_lib.name_parts)
        ob[key] = guide[key]
    # material
    base, rp = mouth_textures()
    mat = pbr_material("M_%s_teeth" % kind, base=base, rough=0.2, rough_img=rp, spec=0.6)
    gltf_occlusion(mat)
    set_material(ob, mat)
    tris = sum(len(p.vertices) - 2 for p in me.polygons)
    log("dentition %s: %d crowns, %d verts, %d tris (upper %d verts on head, lower %d on jaw)" % (
        kind, len(plan["teeth"]), len(V), tris, int((JAW == 0).sum()), int((JAW == 1).sum())))
    gme = guide.data
    bpy.data.objects.remove(guide, do_unlink=True)
    if gme.users == 0:                    # free the guide's mesh name too (the glTF mesh was '<kind>_teeth.001')
        bpy.data.meshes.remove(gme)
    ob.name = name; me.name = name
    return ob, plan


# ------------------------------------------------------------------------------------------------------------------
# expression shapes
NBODY = 13380                                            # MakeHuman basemesh body vertices come first (helpers after)
UNITS = os.path.join(bpy.utils.user_resource('EXTENSIONS'), "user_default", "mpfb", "data", "targets", "expression",
                     "units", "caucasian")
EXTRA_KEYS = EXTRA_FACE_KEYS                             # extra (non-ARKit) shapes, after ARKit + visemes


def read_target(path, n, scale=0.1):
    """MakeHuman .target(.gz) -> (n, 3) Blender-space deltas (MPFB's axis swap, decimetre -> metre)."""
    import gzip
    D = np.zeros((n, 3))
    txt = (gzip.open(path, "rt") if path.endswith(".gz") else open(path)).read()
    for line in txt.splitlines():
        line = line.strip()
        if not line or line[0] in "#\"":
            continue
        p = line.split()
        i = int(p[0])
        if i < n:
            D[i] = (float(p[1]) * scale, -float(p[3]) * scale, float(p[2]) * scale)
    return D


def unit(name, n):
    return read_target(os.path.join(UNITS, name + ".target.gz"), n)


def lr_split(D, co, width=0.010):
    """Split a symmetric delta into character-left (+X) / right halves with a smooth blend across the midline."""
    wl = smoothstep(-width, width, co[:, 0])[:, None]
    return D * wl, D * (1 - wl)


def key_delta(obj, name):
    kb = obj.data.shape_keys.key_blocks
    return verts_np(obj, kb[name]) - verts_np(obj, kb["Basis"])


def set_key_delta(obj, name, D, create=False):
    kb = obj.data.shape_keys.key_blocks
    if name not in kb:
        assert create, name
        k = obj.shape_key_add(name=name, from_mix=False); k.value = 0.0
    kb[name].data.foreach_set("co", (verts_np(obj, kb["Basis"]) + D).ravel())


# ------------------------------------------------------------------------------------------------------------------
# mouth cavity (the body's mouth bag behind the lips)


def mouth_cavity_depth(body, rig):
    """Per body vertex: geodesic depth (m) into the mouth bag, measured from the lips' vermilion, or -1 outside the
    bag. Flood fill over the mesh edges from the back wall of the mouth, never entering the 'lips' vertex group
    (MakeHuman's vermilion) or anything in front of the lips / beyond the mouth corners. Needs the MakeHuman
    'lips' group (live body, or the export body before clean_weights)."""
    import heapq
    me = body.data
    n = len(me.vertices)
    co = verts_np(body)
    nr = np.empty(n * 3); me.vertices.foreach_get("normal", nr); nr = nr.reshape(-1, 3)
    lips = group_weights(body, "lips")
    lm = lip_landmarks(body, co)
    st = lm["stomion"]
    xc = abs(rig.data.bones["mouth_corner_l"].head_local.x)
    e = np.empty(len(me.edges) * 2, dtype=np.int64); me.edges.foreach_get("vertices", e); e = e.reshape(-1, 2)
    adj = [[] for _ in range(n)]
    for a, b in e:
        adj[a].append(b); adj[b].append(a)
    allowed = (lips < 0.5) & (co[:, 1] > lm["y_outer_upper"] + 0.002) & (np.abs(co[:, 0]) < xc + 0.012) & \
              (np.abs(co[:, 2] - st[2]) < 0.045)
    allowed[NBODY:] = False
    seeds = np.nonzero(allowed & (co[:, 1] > st[1] + 0.05) & (np.abs(co[:, 0]) < 0.015) & (np.abs(co[:, 2] - st[2]) < 0.015)
                       & (nr[:, 1] < -0.5))[0]
    inbag = np.zeros(n, bool)
    stack = list(seeds)
    inbag[seeds] = True
    while stack:
        v = stack.pop()
        for w in adj[v]:
            if allowed[w] and not inbag[w]:
                inbag[w] = True; stack.append(w)
    # geodesic distance from the bag's rim (bag vertices next to non-bag vertices, i.e. the lips)
    dist = np.full(n, np.inf)
    rim = [v for v in np.nonzero(inbag)[0] if any(not inbag[w] for w in adj[v])]
    h = [(0.0, v) for v in rim]
    for v in rim:
        dist[v] = 0.0
    while h:
        d, v = heapq.heappop(h)
        if d > dist[v]:
            continue
        for w in adj[v]:
            if inbag[w]:
                nd = d + float(np.linalg.norm(co[v] - co[w]))
                if nd < dist[w]:
                    dist[w] = nd; heapq.heappush(h, (nd, w))
    depth = np.where(inbag, dist, -1.0)
    log("mouth bag: %d vertices (%d seeds, %d rim), depth up to %.1f mm, bbox y %.3f..%.3f" % (
        int(inbag.sum()), len(seeds), len(rim), float(depth.max()) * 1e3, co[inbag, 1].min(), co[inbag, 1].max()))
    return depth


# ------------------------------------------------------------------------------------------------------------------
# live-stage face edits (run on the MPFB body right after the face keys are loaded / repaired)


def modelled_coords(bm):
    """Current modelled shape of the live body (modelling keys mixed, face keys at 0), all vertices incl. helpers."""
    kb = bm.data.shape_keys.key_blocks
    saved = {k.name: k.value for k in kb if k.name in FACE_KEYS}
    for k in kb:
        if k.name in FACE_KEYS:
            k.value = 0.0
    tmp = bm.shape_key_add(name="__mix", from_mix=True)
    co = verts_np(bm, tmp)
    bm.shape_key_remove(tmp)
    for k, v in saved.items():
        kb[k].value = v
    return co


def vertex_normals(bm, co):
    """Area-weighted vertex normals of the body at positions co (numpy, so helper-free and key-independent)."""
    me = bm.data
    me.calc_loop_triangles()
    t = np.empty(len(me.loop_triangles) * 3, dtype=np.int64); me.loop_triangles.foreach_get("vertices", t)
    t = t.reshape(-1, 3)
    fn = np.cross(co[t[:, 1]] - co[t[:, 0]], co[t[:, 2]] - co[t[:, 0]])
    vn = np.zeros_like(co)
    for i in range(3):
        np.add.at(vn, t[:, i], fn)
    return vn / np.maximum(np.linalg.norm(vn, axis=1), 1e-12)[:, None]


def _fill(co, R, adj, p):
    """Flood fill over `adj` from the region vertex nearest to p."""
    cand = np.nonzero(R)[0]
    s0 = cand[np.argmin(np.linalg.norm(co[cand] - p, axis=1))]
    lab = np.zeros(len(co), bool); lab[s0] = True; st = [s0]
    while st:
        v = st.pop()
        for w in adj.get(v, []):
            if not lab[w]:
                lab[w] = True; st.append(w)
    return lab


def lip_seal_delta(bm, rig, co):
    """MakeHuman's closed lips leave a ~1 mm slit, so the teeth read as a white line between the lips at rest. The
    upper and lower lip are labelled by flood fill (they only connect at the corners and deep in the mouth bag, both
    excluded), the slit is measured per 1.5 mm column, and each lip is translated towards the other (gap / 2 +
    0.15 mm, fading over 3 mm from its edge and towards the corners): the mouth seals without flattening the lips."""
    n = len(co)
    me = bm.data
    body = np.zeros(n, bool); body[:NBODY] = True
    xc = abs(rig.data.bones["mouth_corner_l"].head_local.x)
    zu = rig.data.bones["lip_upper_c"].head_local.z; zl = rig.data.bones["lip_lower_c"].head_local.z
    zs0 = (zu + zl) / 2
    mid = body & (np.abs(co[:, 0]) < 0.0015)
    yf = co[mid & (np.abs(co[:, 2] - zs0) < 0.012), 1].min()               # most forward lip point
    ax = np.abs(co[:, 0])
    band = body & (np.abs(co[:, 2] - zs0) < 0.012)
    fb = np.arange(0.0, xc + 0.001, 0.001)                  # lip front at the seam per column (lips curve back)
    seamband = body & (np.abs(co[:, 2] - zs0) < 0.003)
    yfront = np.array([co[seamband & (np.abs(ax - b) < 0.0012), 1].min() for b in fb])
    R = None
    for cut, dep in ((1.0, 5.0), (2.5, 4.5), (4.0, 4.0), (5.5, 3.5)):   # stop before the lips join at the corner
        R = band & (ax < xc - cut * MM) & (co[:, 1] < np.interp(ax, fb, yfront) + dep * MM)
        e = np.empty(len(me.edges) * 2, dtype=np.int64); me.edges.foreach_get("vertices", e); e = e.reshape(-1, 2)
        e = e[R[e[:, 0]] & R[e[:, 1]]]
        adj = {}
        for a, b in e:
            adj.setdefault(a, []).append(b); adj.setdefault(b, []).append(a)
        up = _fill(co, R, adj, np.array([0.0, yf, zs0 + 0.006]))
        lo = _fill(co, R, adj, np.array([0.0, yf + 0.002, zs0 - 0.006]))
        if not (up & lo).any():
            break
    if (up & lo).any():
        log("lip seal: upper and lower lip connected inside the region, skipped")
        return np.zeros_like(co)
    bins = np.arange(0.0, xc - 1.0 * MM, 0.0015)
    xs, zue, zle = [], [], []
    for b in bins:
        col = np.abs(ax - b) < 0.0016
        U = np.nonzero(col & up)[0]; L = np.nonzero(col & lo)[0]
        if len(U) and len(L):
            xs.append(b); zue.append(co[U, 2].min()); zle.append(co[L, 2].max())
    xs, zue, zle = np.array(xs), np.array(zue), np.array(zle)
    gap = np.clip(zue - zle, 0.0, 1.5 * MM)
    gap = np.array([np.median(gap[max(0, i - 1):i + 2]) for i in range(len(gap))])   # sparse columns: 3-tap median
    g = np.interp(ax, xs, gap)
    off = g / 2 + 0.15 * MM
    fade = 1 - smoothstep(xc - 3.5 * MM, xc + 0.5 * MM, ax)
    D = np.zeros_like(co)
    wu = (1 - smoothstep(0.0, 3.0 * MM, co[:, 2] - np.interp(ax, xs, zue))) * fade * up
    wl = (1 - smoothstep(0.0, 3.0 * MM, np.interp(ax, xs, zle) - co[:, 2])) * fade * lo
    D[:, 2] = -off * wu + off * wl
    log("lip seal: slit (x mm: gap mm) %s; %d upper / %d lower lip vertices moved" % (
        " ".join("%.1f:%.2f" % (a * 1e3, b * 1e3) for a, b in zip(xs, gap)), int((wu > 0.01).sum()),
        int((wl > 0.01).sum())))
    return D


def add_model_fix(bm, D, name="rts_face_fix"):
    """Store geometric face fixes as a modelling shape key (value 1): the export bake folds it into the basis, the
    live file keeps it editable."""
    kb = bm.data.shape_keys.key_blocks
    k = kb.get(name) or bm.shape_key_add(name=name, from_mix=False)
    k.relative_key = kb["Basis"]
    k.data.foreach_set("co", (verts_np(bm, kb["Basis"]) + D).ravel())
    k.value = 1.0
    return k


def eye_spheres(eyes):
    """(centre, radius) of each eyeball (+X = character's left first) from the eye mesh's sclera (material 0)."""
    co = verts_np(eyes)
    me = eyes.data
    scl = np.zeros(len(co), bool)
    for p in me.polygons:
        if p.material_index == 0:
            scl[list(p.vertices)] = True
    out = []
    for sg in (1, -1):
        P = co[scl & (np.sign(co[:, 0]) == sg)]
        # least-squares sphere
        A = np.c_[2 * P, np.ones(len(P))]
        b = (P ** 2).sum(1)
        sol = np.linalg.lstsq(A, b, rcond=None)[0]
        c = sol[:3]; r = math.sqrt(sol[3] + (c ** 2).sum())
        out.append((c, r))
    return out


def eye_shells(eyes):
    """Per eye (+X first): [(sclera centre, r), (cornea shell centre, r)]. The split eye mesh's clear cornea (material 1)
    is a whole outer shell ~0.45 mm larger than the sclera and ~1.2 mm further forward: its front bulge stands ~2.5 mm
    proud of the sclera sphere, so a lid kept clear of the sclera alone shows the cornea through the closed lid."""
    co = verts_np(eyes)
    mi = np.array([p.material_index for p in eyes.data.polygons])
    out = []
    for sg, sph in zip((1, -1), eye_spheres(eyes)):
        shells = [sph]
        if (mi == 1).any():
            vs = np.unique(np.concatenate([list(eyes.data.polygons[i].vertices) for i in np.nonzero(mi == 1)[0]]))
            P = co[vs]; P = P[np.sign(P[:, 0]) == sg]
            sol = np.linalg.lstsq(np.c_[2 * P, np.ones(len(P))], (P ** 2).sum(1), rcond=None)[0]
            shells.append((sol[:3], math.sqrt(sol[3] + (sol[:3] ** 2).sum())))
        out.append(shells)
    return out


class EyeSurface:
    """The eye mesh (sclera + clear cornea: the cornea bulge is not a sphere) for 'how far is the outermost eye surface
    from the eyeball centre along this direction' queries (rays from the centre, last hit)."""

    def __init__(self, eyes, P=None):
        from mathutils.bvhtree import BVHTree
        me = eyes.data
        me.calc_loop_triangles()
        T = np.empty(len(me.loop_triangles) * 3, dtype=np.int64); me.loop_triangles.foreach_get("vertices", T)
        P = verts_np(eyes) if P is None else P
        self.bvh = BVHTree.FromPolygons(P.tolist(), T.reshape(-1, 3).tolist(), all_triangles=True)

    def radius(self, c, U, rmax=0.028):
        R = np.zeros(len(U))
        cv = Vector(c)
        for i, u in enumerate(U):
            d = Vector(u); o = cv.copy(); t = 0.0
            for _ in range(12):
                loc, nrm, fi, dist = self.bvh.ray_cast(o, d, rmax - t)
                if loc is None:
                    break
                t += dist + 1e-6
                o = loc + d * 1e-6
            R[i] = t
        return R


def eye_radius(c, surf, U):
    """Distance from c to the outermost eye surface (EyeSurface) along unit directions U (N,3)."""
    return surf.radius(c, U)


def lid_eyeball_correction(bm, eyes, co, keys_by_side, lining):
    """Lower-lid / cheek skin that the squint keys push INTO the eyeball (it then shows through the skin as a pale
    crescent under the eye) is pushed back out onto the sphere + 0.8 mm. Because blendshapes mix linearly and the
    outside of a sphere is not convex, the fix is solved over a grid of weight mixes (0..1 per key): each mix that
    still penetrates adds its push to the keys in proportion to their weights (minimum-norm split), 8 sweeps. Only
    lower-lid / cheek skin that is off the eyeball at rest is corrected; the lid margins and the red socket lining
    (which rest on / behind the eyeball) are left alone."""
    import itertools
    body = np.zeros(len(co), bool); body[:NBODY] = True
    rep = {}
    surf = EyeSurface(eyes)
    vn = vertex_normals(bm, co)
    lining = np.zeros(len(co), bool) if lining is None else lining
    for (c, r), keys in zip(eye_spheres(eyes), keys_by_side):
        shells = surf
        d0 = np.linalg.norm(co - c, axis=1)
        # clearance to the outer eye surface (sclera + cornea shell) along each vertex's direction from the centre
        cand = np.nonzero((np.arange(len(co)) < NBODY) & (d0 < 0.03))[0]
        R = np.full(len(co), 1.0)
        R[cand] = eye_radius(c, shells, (co[cand] - c) / np.maximum(d0[cand], 1e-9)[:, None]) + 0.8 * MM
        # lower lid / cheek skin in front of the eye; skin that already rests on / in the eye surface at rest (the
        # lid margins, the medial canthus) may not sink DEEPER than at rest
        lin_geo = (d0 < 0.03) & ((vn * (co - c)).sum(1) / np.maximum(d0, 1e-9) < 0.15)   # faces the eyeball
        # (geometric lining only: the albedo-red test also flags the caruncle / medial canthus skin, which the closing
        # lids pushed 0.6 mm into the eye)
        near = body & ~lin_geo & (d0 < 0.03) & (co[:, 1] < c[1]) & (co[:, 2] < c[2] + 0.002)
        rest_in = d0 < R
        keys = [k for k in keys if k in bm.data.shape_keys.key_blocks]
        grid = [w for w in itertools.product((0.0, 0.25, 0.5, 0.75, 1.0), repeat=len(keys)) if sum(w) > 0]
        Ds = [key_delta(bm, k) for k in keys]
        worst0 = worst = 0.0
        for sweep in range(8):
            worst = 0.0
            for w in grid:
                P = co + sum(wi * D for wi, D in zip(w, Ds))
                v = P - c; d = np.linalg.norm(v, axis=1)
                Rw = np.zeros(len(P))
                ni = np.nonzero(near)[0]
                Rw[ni] = eye_radius(c, shells, v[ni] / np.maximum(d[ni], 1e-9)[:, None]) + 0.8 * MM
                Rw = np.where(rest_in, np.minimum(Rw, d0 - 2e-5), Rw)
                bad = near & (d < Rw)
                if not bad.any():
                    continue
                C = np.zeros_like(P)
                C[bad] = v[bad] / d[bad][:, None] * (Rw[bad] - d[bad])[:, None]
                worst = max(worst, float((Rw[bad] - d[bad]).max()))
                nn = sum(wi * wi for wi in w)
                for i, wi in enumerate(w):
                    Ds[i] = Ds[i] + C * (wi / nn)
            if sweep == 0:
                worst0 = worst
            if worst == 0.0:
                break
        for k, D in zip(keys, Ds):
            set_key_delta(bm, k, D)
        rep["+".join(keys)] = "max penetration %.2f mm -> %.2f mm" % (worst0 * 1e3, worst * 1e3)
    log("lid / eyeball correction:", rep)


LID_CLEAR = 1.1 * MM          # closed upper lid: skin at least this far outside the eye (sclera + cornea shell)
LID_CLEAR_MID = 0.3 * MM      # ... and at blink 0.5 / 0.75 (linear blend of the rolled key: the chord)
LID_TOP = 35.0 * MM           # re-solved upper-lid region: skin within this distance of the eyeball centre (the
                              # pretarsal lid, the crease and the preseptal skin up to the underside of the brow)
_LID_DEBUG = {}
LIN_R = 2.5 * MM              # socket lining = faces the eyeball and lies within the eye radius + this
LID_IRLS = 6                  # stretch-equalising re-solves (edge weights x (stretch / mean stretch)^2)


def _cotan_edges(P, T):
    """Symmetric cotangent weights (clamped >= 0.05 of the mean) per unique edge of triangles T at positions P."""
    W = {}
    for a, b, c in T:
        for i, j, k in ((a, b, c), (b, c, a), (c, a, b)):
            u, v = P[i] - P[k], P[j] - P[k]
            ct = float(u @ v) / max(float(np.linalg.norm(np.cross(u, v))), 1e-12)
            e = (min(i, j), max(i, j))
            W[e] = W.get(e, 0.0) + 0.5 * ct
    E = np.array(list(W.keys()), dtype=np.int64); w = np.array(list(W.values()))
    w = np.maximum(w, 0.05 * np.abs(w).mean())
    return E, w


def _eye_frame(P, c):
    """Positions -> (x, rho, theta) about the eyeball's horizontal (x) axis through c: rho = distance from the axis,
    theta = elevation from straight ahead (-Y) towards +Z. A lid that closes ROLLS about this axis."""
    q = P - c
    return np.stack([q[:, 0], np.hypot(q[:, 1], q[:, 2]), np.arctan2(q[:, 2], -q[:, 1])], 1)


def _from_eye_frame(F, c):
    return c + np.stack([F[:, 0], -F[:, 1] * np.cos(F[:, 2]), F[:, 1] * np.sin(F[:, 2])], 1)


def lid_roll_blink(bm, eyes, co, lining, keys=("eyeBlinkLeft", "eyeBlinkRight"), log_=None):
    """User round-2 item 8 (closed lid texels stretched): the CC0 blink slides the upper-lid margin ~12 mm down over
    the eye (38 deg about the eyeball) but moves the skin above it only a little, so the 4-5 mm of pretarsal skin and
    the crease row stretch 3-6x and their texels smear into vertical streaks. The closed lid is re-solved as a ROLL:
    the blink displacement of the whole upper lid (margin -> crease -> preseptal skin up to the brow, LID_TOP from the
    eyeball centre) is written in the eyeball's rotational frame (x, distance from the eye's horizontal axis, elevation
    angle) and harmonically interpolated (cotangent weights of the rest mesh) between the margin (lid-edge vertices
    next to the red socket lining keep their authored path, so the eye closes exactly as before) and the unmoved skin
    around the region. Interpolating the ANGLE instead of the Cartesian offset makes every row turn down around the
    eyeball instead of cutting into it. LID_IRLS re-solves then stiffen the edges that stretch most (weights x
    (stretch / mean)^2), which spreads the stretch evenly over the lid; the closed lid is kept LID_CLEAR outside the
    eyeball sphere (projected vertices become constraints). The lining keeps its authored motion. Returns a report."""
    log_ = log_ or log
    me = bm.data
    n = len(co)
    nb = min(n, NBODY)
    me.calc_loop_triangles()
    Tt = np.empty(len(me.loop_triangles) * 3, dtype=np.int64); me.loop_triangles.foreach_get("vertices", Tt)
    Tt = Tt.reshape(-1, 3)
    Tt = Tt[(Tt < nb).all(1)]
    spheres = eye_spheres(eyes)
    surf = EyeSurface(eyes)
    vn = vertex_normals(bm, co)
    rep = {}
    for key, (c, r) in zip(keys, spheres):
        shells = surf
        if key not in me.shape_keys.key_blocks:
            continue
        sgn = 1 if key.endswith("Left") else -1
        D = key_delta(bm, key)
        mv = np.linalg.norm(D, axis=1)
        d0 = co - c
        dist = np.linalg.norm(d0, axis=1)
        # socket lining (conjunctiva / inner lid): faces the eyeball (geometric test; the albedo test also hits the
        # pinkish lid skin); only near this eye
        # (within r + 2.5 mm of the eye centre: the side of the nose also faces the eyeball, and taking it for lining
        # pinned the medial lid border 13 mm from the eye centre, a crease down the closed lid)
        lin = (dist < r + LIN_R) & ((vn * d0).sum(1) / np.maximum(dist, 1e-9) < 0.15)
        region = (np.arange(n) < nb) & (np.sign(co[:, 0]) == sgn) & ~lin & (d0[:, 2] > -0.003) & (d0[:, 1] < 0.006) & \
                 ((dist < LID_TOP) | ((mv > 1e-4) & (dist < LID_TOP + 0.006)))
        tri = Tt[region[Tt].all(1)]
        inreg = np.zeros(n, bool); inreg[np.unique(tri)] = True
        # lid edge = region vertices sharing an EDGE with the lining (they keep the authored closing path)
        ed = np.concatenate([Tt[:, [0, 1]], Tt[:, [1, 2]], Tt[:, [2, 0]]])
        ed = ed[lin[ed[:, 0]] ^ lin[ed[:, 1]]]
        adj_lin = np.zeros(n, bool)
        adj_lin[ed[~lin[ed[:, 0]], 0]] = True; adj_lin[ed[~lin[ed[:, 1]], 1]] = True
        margin = inreg & adj_lin
        touches_out = np.zeros(n, bool)
        touches_out[np.unique(Tt[inreg[Tt].any(1) & ~inreg[Tt].all(1)])] = True
        fixed = margin | (touches_out & inreg)
        free = inreg & ~fixed
        E, w0 = _cotan_edges(co, tri)
        L0 = np.linalg.norm(co[E[:, 0]] - co[E[:, 1]], axis=1)
        F0 = _eye_frame(co, c)
        F1 = _eye_frame(co + D, c)
        dF = F1 - F0
        dF[:, 2] = (dF[:, 2] + np.pi) % (2 * np.pi) - np.pi
        stretch0 = _lid_stretch(co, D, tri)
        w = w0.copy()
        X = dF.copy()
        lb = np.full(n, -np.inf)          # lower bound on the eye-axis distance change (eye clearance), per vertex
        fi_ = np.nonzero(free)[0]
        mg_ = np.nonzero(margin)[0]       # the lid edge keeps its authored path, only lifted off the eye if needed
        chk = np.nonzero(free | margin)[0]
        drest = dist
        for rw in range(LID_IRLS + 1):
            deg = np.zeros(n); np.add.at(deg, E[:, 0], w); np.add.at(deg, E[:, 1], w)
            for rnd in range(6):
                for it in range(600):
                    S = np.zeros_like(X)
                    np.add.at(S, E[:, 0], X[E[:, 1]] * w[:, None]); np.add.at(S, E[:, 1], X[E[:, 0]] * w[:, None])
                    X[fi_] = S[fi_] / deg[fi_][:, None]
                    X[fi_, 1] = np.maximum(X[fi_, 1], lb[fi_])
                    X[mg_, 1] = np.maximum(dF[mg_, 1], lb[mg_])
                P = _from_eye_frame(F0 + X, c)
                # the closed lid LID_CLEAR outside the eye surface (sclera + cornea), and the in-between blends (linear:
                # a rolled vertex travels the CHORD) LID_CLEAR_MID outside it; a violation raises that vertex's lower
                # bound on the eye-axis distance (its elevation / x stay free, so the roll stays smooth)
                need = np.zeros(n)
                for wgt, clr in ((1.0, LID_CLEAR), (0.5, LID_CLEAR_MID), (0.75, LID_CLEAR_MID)):
                    Pw = co[chk] + wgt * (P[chk] - co[chk])
                    v = Pw - c; dw = np.linalg.norm(v, axis=1)
                    Rw = eye_radius(c, shells, v / np.maximum(dw, 1e-9)[:, None])
                    # the lid edge rests ON the eye: it may not sink deeper than its rest distance / 0.2 mm off
                    tgt = np.where(margin[chk], np.minimum(Rw + 0.2 * MM, drest[chk]), Rw + clr)
                    need[chk] = np.maximum(need[chk], (tgt - dw) / wgt)
                bad = (free | margin) & (need > 1e-6)
                if not bad.any():
                    break
                dd = np.linalg.norm(P[bad] - c, axis=1)
                D2 = (dd + need[bad] + 2e-5) ** 2 - (P[bad, 0] - c[0]) ** 2
                lb[bad] = np.maximum(lb[bad], np.sqrt(np.maximum(D2, 0.0)) - F0[bad, 1])
            if rw == LID_IRLS:
                break
            lam = np.linalg.norm(P[E[:, 0]] - P[E[:, 1]], axis=1) / np.maximum(L0, 1e-9)
            w = w * np.clip(lam / lam.mean(), 0.5, 3.0) ** 2
        pinned = free & np.isfinite(lb)
        _LID_DEBUG[key] = dict(margin=margin, fixed=fixed, free=free, lining=lin, region=inreg)
        Dn = P - co
        Dn[~inreg] = D[~inreg]
        Dn[fixed & ~margin] = D[fixed & ~margin]
        set_key_delta(bm, key, Dn)
        stretch1 = _lid_stretch(co, Dn, tri)
        rep[key] = "%d lid verts re-solved (%d margin / boundary fixed, %d on the eyeball clearance), max stretch %.2f -> " \
                   "%.2f, p95 %.2f -> %.2f" % (int(free.sum()), int(fixed.sum()), int((pinned & free).sum()), stretch0[0],
                                              stretch1[0], stretch0[1], stretch1[1])
    log_("closed-lid roll:", rep)
    return rep


LID_RELAX_MAX = 2.0           # blink keys: outer lid-skin triangles stretched beyond this are relaxed (face_qa gate 2.5)
LINING_REF = "%s_skin_lining_ref.jpg"   # bake_skin.py: the skin albedo without the closed-lid treatment


def lining_ref_image(kind):
    """The albedo the eye-socket lining test reads (chr_lib.fix_eye_socket_keys: MakeHuman's red socket-lining texels):
    bake_skin.py's skin albedo WITHOUT the closed-lid treatment (lid_detail de-reddens the lid margin rows; the face keys
    must not depend on the lid texture, or every bake feeds the previous bake's lid back into the geometry). None
    before the first bake (callers fall back to the material's image)."""
    p = os.path.join(TEX, LINING_REF % kind)
    if not os.path.exists(p):
        return None
    return bpy.data.images.load(p, check_existing=True)


def lid_blink_relax(bm, eyes, co, keys=("eyeBlinkLeft", "eyeBlinkRight"), limit=LID_RELAX_MAX, log_=None):
    """Safety net after lid_roll_blink (user item 8). The roll re-solves the UPPER lid; the lower lid and the canthi
    keep the CC0 blink + the socket-lining repair (chr_lib.fix_eye_socket_keys), where a few small triangles next to
    the lining (medial / lateral canthus, lower lid) still open 2-3x in a blink. Per blink key: the delta of every
    vertex of an outer lid-skin triangle (face_qa.py's selection: eyelid_upper / eyelid_lower weighted, facing away
    from the eyeball at rest or closed, slivers excluded) stretched beyond `limit` is relaxed towards its neighbours'
    (damped umbrella steps) until none is; the lid edge (lid_roll_blink's margin) stays; a vertex whose relaxed path
    would come closer than 0.3 mm to the eye (blend 0.5 / 0.75 / 1: a linear key travels the chord) keeps its key.
    Returns a report."""
    log_ = log_ or log
    me = bm.data
    n = len(co); nb = min(n, NBODY)
    me.calc_loop_triangles()
    Tt = np.empty(len(me.loop_triangles) * 3, dtype=np.int64); me.loop_triangles.foreach_get("vertices", Tt)
    Tt = Tt.reshape(-1, 3); Tt = Tt[(Tt < nb).all(1)]
    e = np.empty(len(me.edges) * 2, dtype=np.int64); me.edges.foreach_get("vertices", e); e = e.reshape(-1, 2)
    e = e[(e < nb).all(1)]
    deg = np.zeros(n); np.add.at(deg, e[:, 0], 1); np.add.at(deg, e[:, 1], 1)
    surf = EyeSurface(eyes)
    rep = {}
    for key, (c, r) in zip(keys, eye_spheres(eyes)):
        if key not in me.shape_keys.key_blocks:
            continue
        s = "l" if key.endswith("Left") else "r"
        wu = group_weights(bm, "eyelid_upper_" + s); wl = group_weights(bm, "eyelid_lower_" + s)
        D = key_delta(bm, key)
        T = Tt[(wu[Tt] > 0.05).all(1) | (wl[Tt] > 0.05).all(1)]

        def outward(P):
            fn = np.cross(P[T[:, 1]] - P[T[:, 0]], P[T[:, 2]] - P[T[:, 0]]); cc = P[T].mean(1) - c
            return (fn * cc).sum(1) > 0.25 * np.linalg.norm(fn, axis=1) * np.linalg.norm(cc, axis=1)
        T = T[outward(co) | outward(co + D)]
        E0 = np.stack([co[T[:, 1]] - co[T[:, 0]], co[T[:, 2]] - co[T[:, 0]]], -1)
        ar2 = np.linalg.norm(np.cross(E0[:, :, 0], E0[:, :, 1]), axis=1)
        lmax = np.maximum(np.maximum(np.linalg.norm(E0[:, :, 0], axis=1), np.linalg.norm(E0[:, :, 1], axis=1)),
                          np.linalg.norm(E0[:, :, 1] - E0[:, :, 0], axis=1))
        keep = ~((ar2 / np.maximum(lmax, 1e-12) < 0.00025) | (ar2 / 2 < 0.25e-6))   # face_qa LID_SLIVER / LID_SMALL
        T, E0 = T[keep], E0[keep]
        E0i = np.linalg.pinv(E0)

        def fs(X):
            P = co + X
            E1 = np.stack([P[T[:, 1]] - P[T[:, 0]], P[T[:, 2]] - P[T[:, 0]]], -1)
            return np.linalg.svd(E1 @ E0i, compute_uv=False)[:, 0]

        def gap(X, vs, wgt):
            v = co[vs] + wgt * X[vs] - c; d = np.linalg.norm(v, axis=1)
            return d - eye_radius(c, surf, v / np.maximum(d, 1e-9)[:, None])
        frozen = _LID_DEBUG.get(key, {}).get("margin", np.zeros(n, bool)).copy()
        f0 = fs(D); X = D.copy(); moved = np.zeros(n, bool); it = 0
        for it in range(400):
            f = fs(X); hot = f > limit
            if not hot.any():
                break
            hv = np.unique(T[hot]); hv = hv[~frozen[hv]]
            if not len(hv):
                break
            S = np.zeros_like(X); np.add.at(S, e[:, 0], X[e[:, 1]]); np.add.at(S, e[:, 1], X[e[:, 0]])
            Xn = X.copy(); Xn[hv] += 0.5 * (S[hv] / np.maximum(deg[hv], 1)[:, None] - X[hv])
            bad = np.zeros(len(hv), bool)
            for wgt in (0.5, 0.75, 1.0):
                g1, g0 = gap(Xn, hv, wgt), gap(X, hv, wgt)
                bad |= (g1 < 0.3 * MM) & (g1 < g0 - 1e-6)
            frozen[hv[bad]] = True
            X[hv[~bad]] = Xn[hv[~bad]]; moved[hv[~bad]] = True
        f1 = fs(X)
        if moved.any():
            set_key_delta(bm, key, X)
        rep[key] = "%d verts relaxed (%d sweeps), max stretch %.2f -> %.2f, p95 %.2f -> %.2f, %d held by the eye clearance" % (
            int(moved.sum()), it, f0.max(), f1.max(), np.percentile(f0, 95), np.percentile(f1, 95),
            int((frozen & ~_LID_DEBUG.get(key, {}).get("margin", np.zeros(n, bool))).sum()))
    log_("blink stretch relax:", rep)
    return rep


def lid_vertex_mask(body, rig, co=None):
    """Per body vertex 0..1: the upper-lid skin that stretches when the eye closes (lid_roll_blink spreads the blink
    over it), plus the lower-lid rim at half strength (squints). Geometric, from the eye bones and the rig's
    eyelid_upper / eyelid_lower weights, so it works on any body (bake_skin.py has no eye mesh)."""
    co = verts_np(body) if co is None else co
    n = len(co)
    vn = vertex_normals(body, co)
    m = np.zeros(n)
    for s in ("l", "r"):
        c = np.array(rig.data.bones["eye_" + s].head_local)
        wu = group_weights(body, "eyelid_upper_" + s)
        wl = group_weights(body, "eyelid_lower_" + s)
        d0 = co - c; d = np.linalg.norm(d0, axis=1)
        dm = float(np.median(d[wu > 0.5])) if (wu > 0.5).any() else 0.019
        out = (vn * d0).sum(1) / np.maximum(d, 1e-9) > 0.15            # outer skin, not the socket lining
        front = d0[:, 1] < 0.004
        up = smoothstep(-0.004, 0.0, d0[:, 2]) * (1 - smoothstep(dm + 0.005, dm + 0.009, d))
        lo = 0.5 * (1 - smoothstep(-0.001, 0.002, d0[:, 2])) * (1 - smoothstep(dm + 0.002, dm + 0.005, d)) * (wl > 0.05)
        m = np.maximum(m, np.where(out & front, np.maximum(up, lo), 0.0))
        # the upper-lid MARGIN rows (they face the eye at rest; the rolled blink turns them forward, where their
        # socket-lining red showed as a peach band above the lashes of a closed eye)
        marg = ~out & front & (wu > 0.3) & (d0[:, 2] > -0.002) & (d < dm + 0.004)
        m = np.maximum(m, np.where(marg, 0.85, 0.0))
    m[NBODY:] = 0.0
    return m


# ---- closed-lid texture (user item 8, round 5 pass). The head is laid out ROTATED in MakeHuman's atlas (the eyes are
# stacked along V): the lid stretches along U when it closes, and the earlier lid_smooth()'s V blur smeared the pores along the lid
# margin instead, so the closed lid still showed vertical streaks. lid_detail() keeps the lid's colour and shading
# (isotropic low-pass) and replaces its fine detail (albedo mottling, pore / grain normals) by procedural detail that is
# evaluated on the lid surface PART-WAY CLOSED (lid_noise_domain: rest + LID_DETAIL_BETA x the rolled-blink motion):
# pores that are round on the closed lid are pre-compressed on the open lid, where the upper lid folds into its crease.
LID_DETAIL_BETA = 0.9         # noise domain: 0 = rest lid (stretches 1.6x when closed), 1 = closed lid (1.6x compressed open)
LID_LOW_SIGMA = 3.0           # px, isotropic low-pass kept from the baked maps (colour, AO, the subdivided-mesh shading)
# The lid detail must stay BAND-LIMITED: a texel-sized feature (pores are sub-texel at ~0.6 mm / texel on the face) is
# stretched with its texel whatever the noise domain, so a closed lid shows every 1-texel feature as a vertical streak
# (the first try of this pass with 0.9 mm Voronoi pores looked worse than the blur). Detail >= ~3 texels only:
LID_GRAIN_CELL = 0.0019       # m, fine skin grain (value noise; ~3 texels on the open lid)
LID_MOTT_CELL = 0.0045        # m, mottling
LID_AMP = {"albedo": 0.7, "normal": 0.35}   # detail amplitude vs the high-pass of the skin round the lid (thin lid skin)
LID_DECHROMA, LID_DEDARKEN = 0.55, 0.35     # lid albedo: chroma / darkening deviation from the lid's median colour removed


def _hash3(I, seed):
    h = np.sin(I[..., 0] * 127.1 + I[..., 1] * 311.7 + I[..., 2] * 74.7 + seed * 17.31) * 43758.5453
    return h - np.floor(h)


def value_noise3(P, seed=0):
    """Smooth 3D value noise in 0..1 at points P (N, 3) (unit lattice, quintic fade)."""
    I = np.floor(P); f = P - I
    u = f * f * f * (f * (f * 6 - 15) + 10)
    out = np.zeros(len(P))
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                w = (u[:, 0] if dx else 1 - u[:, 0]) * (u[:, 1] if dy else 1 - u[:, 1]) * (u[:, 2] if dz else 1 - u[:, 2])
                out += w * _hash3(I + np.array([dx, dy, dz]), seed)
    return out


def lid_noise_domain(low, lidv, co_rest, co_closed, res, beta=LID_DETAIL_BETA):
    """Noise-domain coordinates (m) per texel of the lid region: the lid surface rest + beta x (closed - rest), painted
    into UV space over the lid faces (lidv > 0). Returns dict(y0, x0, Q (h, w, 3), valid (h, w)) in Blender pixel
    order (row 0 = bottom), cropped to the lid texels' bounding box (+ margin)."""
    n = len(co_rest)
    vals = np.zeros((len(low.data.vertices), 3))
    vals[:n] = co_rest + beta * (co_closed - co_rest)
    fm = lambda tv: lidv[tv].max(1) > 0
    ch = [rasterize_uv(low, vals[:, i], res, fm, fill=np.nan)[::-1] for i in range(3)]
    Q = np.stack(ch, -1)
    valid = np.isfinite(Q).all(-1)
    ys, xs = np.nonzero(valid)
    y0, y1 = max(ys.min() - 16, 0), min(ys.max() + 17, res); x0, x1 = max(xs.min() - 16, 0), min(xs.max() + 17, res)
    return dict(y0=int(y0), x0=int(x0), Q=Q[y0:y1, x0:x1].astype(np.float32), valid=valid[y0:y1, x0:x1])


def _blur2(a, sigma):
    r = int(math.ceil(3 * sigma))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2); k /= k.sum()
    out = a
    for ax in (0, 1):
        acc = np.zeros_like(out)
        for i, w in enumerate(k):
            acc += w * np.roll(out, i - r, axis=ax)
        out = acc
    return out


def lid_detail(img, mask, dom, mode="albedo"):
    """Closed-lid texture treatment (see LID_DETAIL_BETA): inside `mask` (H, W; Blender pixel order like img) the map =
    its isotropic low-pass + procedural detail evaluated on the noise domain `dom` (lid_noise_domain), with the
    amplitude of the high-pass detail of the skin just round the lid (x LID_AMP). mode 'albedo' (RGB, luminance
    mottling + pores) or 'normal' (tangent-space RGB 0..1, OpenGL: pore pits + grain as a height field). Returns a new
    array."""
    out = img.copy()
    y0, x0, Q, valid = dom["y0"], dom["x0"], dom["Q"], dom["valid"]
    h, w = valid.shape
    A = img[y0:y0 + h, x0:x0 + w, :3].astype(np.float64)
    m = mask[y0:y0 + h, x0:x0 + w].astype(np.float64)
    L = _blur2(A, LID_LOW_SIGMA)
    Hp = A - L
    near = _blur2((m > 0.02).astype(float)[..., None], 6.0)[..., 0]
    ring = (m < 0.02) & (near > 0.02)
    P = Q[valid].astype(np.float64)
    grain = value_noise3(P / LID_GRAIN_CELL, 5) - 0.5
    mott = value_noise3(P / LID_MOTT_CELL, 7) - 0.5
    aa = lambda F: _blur2(F[..., None], 0.7)[..., 0]           # anti-alias: nothing left at the texel scale
    new = L.copy()
    if mode == "albedo":
        lum = lambda X: X @ np.array([0.2126, 0.7152, 0.0722])
        ref = float(np.std(lum(Hp)[ring])) if ring.any() else 0.01
        det = np.zeros((h, w)); det[valid] = 0.8 * mott + 0.6 * grain
        det = aa(det); det = det / max(float(det[valid].std()), 1e-9) * ref * LID_AMP["albedo"]
        # the CC0 albedo paints a dark red-brown lash-line band on the upper lid (and the margin rows carry the socket
        # lining's red); the rolled lid of a closed eye turns them forward as a peach / brown band above the lashes: even
        # the lid's chroma (and its darkening) towards the median colour of the skin round the lid
        sel = ring if ring.any() else (m > 0.5) & valid       # reference: the skin just round the lid
        if sel.any():
            Mc = np.median(L[sel], axis=0); ld = lum(L) - float(lum(Mc[None])[0])
            chroma = (L - Mc) - ld[..., None]
            L = L - chroma * (LID_DECHROMA * m)[..., None] - (ld * np.where(ld < 0, LID_DEDARKEN, 0.0) * m)[..., None]
        base = lum(L)
        new = L * (1 + det / np.maximum(base, 1e-3))[..., None]
    else:
        ht = np.zeros((h, w)); ht[valid] = grain + 0.5 * mott
        ht = aa(ht)
        gy, gx = np.gradient(ht)                    # rows = V (Blender order), columns = U
        nx, ny = -gx, -gy
        ref = float(np.sqrt(np.mean(Hp[..., 0][ring] ** 2 + Hp[..., 1][ring] ** 2))) * 2 if ring.any() else 0.05
        sc = ref * LID_AMP["normal"] / max(float(np.sqrt(np.mean((nx[valid] ** 2 + ny[valid] ** 2)))), 1e-9)
        Nn = L * 2 - 1
        Nn[..., 0] += nx * sc; Nn[..., 1] += ny * sc
        Nn /= np.maximum(np.linalg.norm(Nn, axis=-1), 1e-9)[..., None]
        new = Nn * 0.5 + 0.5
    mm = (m * valid)[..., None]
    out[y0:y0 + h, x0:x0 + w, :3] = (A * (1 - mm) + new * mm).astype(img.dtype)
    return out


def save_lid_domain(path, dom):
    np.savez_compressed(path, y0=dom["y0"], x0=dom["x0"], Q=dom["Q"], valid=dom["valid"])


def load_lid_domain(path):
    z = np.load(path)
    return dict(y0=int(z["y0"]), x0=int(z["x0"]), Q=z["Q"], valid=z["valid"])


def _lid_stretch(co, D, T):
    """(max, p95) of the largest singular value of the per-triangle deformation gradient rest -> rest + D."""
    E0 = np.stack([co[T[:, 1]] - co[T[:, 0]], co[T[:, 2]] - co[T[:, 0]]], -1)
    P = co + D
    E1 = np.stack([P[T[:, 1]] - P[T[:, 0]], P[T[:, 2]] - P[T[:, 0]]], -1)
    s = np.linalg.svd(E1 @ np.linalg.pinv(E0), compute_uv=False)[:, 0]
    return float(s.max()), float(np.percentile(s, 95))


def propagate_keys(bm, names, skip=("teeth", "tongue")):
    """Re-interpolate the given body keys onto the MPFB parts (eyebrows, eyelashes, hair, eyes) through their mhclo
    vertex correspondence (same maths as FaceService.interpolate_targets, but it replaces existing keys)."""
    from bl_ext.user_default.mpfb.services import ClothesService
    from bl_ext.user_default.mpfb.entities.clothes.mhclo import Mhclo
    rig = bm.parent
    for m in bm.modifiers:
        m.show_viewport = False
    base = verts_np(bm, bm.data.shape_keys.key_blocks["Basis"])
    deltas = {k: key_delta(bm, k) for k in names}
    for child in rig.children:
        if child == bm or child.type != 'MESH' or any(s in child.name for s in skip):
            continue
        path = ClothesService.find_clothes_absolute_path(child)
        if not path:
            continue
        mh = Mhclo(); mh.load(path)
        idx = [(ci, mh.verts[ci]["verts"], mh.verts[ci]["weights"]) for ci in mh.verts if ci < len(child.data.vertices)]
        ci = np.array([i for i, _, _ in idx]); vv = np.array([v for _, v, _ in idx]); ww = np.array([w for _, _, w in idx])
        if not child.data.shape_keys:
            child.shape_key_add(name="Basis", from_mix=False)
        cb = verts_np(child, child.data.shape_keys.key_blocks["Basis"])
        done = []
        for k, D in deltas.items():
            off = (D[vv] * ww[:, :, None]).sum(1)
            if np.abs(off).max() < 1e-4 and k not in child.data.shape_keys.key_blocks:
                continue
            X = cb.copy(); X[ci] += off
            sk = child.data.shape_keys.key_blocks.get(k) or child.shape_key_add(name=k, from_mix=False)
            sk.data.foreach_set("co", X.ravel()); sk.value = 0.0
            done.append(k)
        log("  propagated %d keys to %s" % (len(done), child.name))
    for m in bm.modifiers:
        m.show_viewport = True


def smooth_delta(bm, D, movable, iters=12, lam=0.5):
    """Umbrella (Laplacian) smoothing of a delta field over the mesh edges, only on `movable` vertices: removes the
    sharp shear in a sculpted delta that makes skin fold over itself, keeps its broad motion."""
    me = bm.data
    e = np.empty(len(me.edges) * 2, dtype=np.int64); me.edges.foreach_get("vertices", e); e = e.reshape(-1, 2)
    e = e[(e < NBODY).all(1)]
    deg = np.zeros(len(D)); np.add.at(deg, e[:, 0], 1); np.add.at(deg, e[:, 1], 1)
    X = D.copy()
    mv = movable & (deg > 0)
    for _ in range(iters):
        S = np.zeros_like(X); np.add.at(S, e[:, 0], X[e[:, 1]]); np.add.at(S, e[:, 1], X[e[:, 0]])
        X[mv] = (1 - lam) * X[mv] + lam * S[mv] / deg[mv][:, None]
    return X


def improve_face_keys(bm, rig, kind, skin_img=None):
    """Live-stage expression work on the MPFB body (after FaceService + fix_eye_socket_keys):
      1. rts_face_fix modelling key: lips sealed at rest (no teeth line between closed lips).
      2. mouthSmileLeft/Right  += 0.6 x MakeHuman 'mouth-corner-puller' (split L/R): corners go higher and further
         back (zygomaticus), the ARKit meaning is kept (closed-lip smile).
      3. mouthFrownLeft/Right  += 0.7 x 'mouth-depression' (split L/R): a readable frown (the CC0 one is very weak).
      4. extra shapes (after ARKit + visemes): smileOpenLeft/Right ('mouth-upward-retraction' split: the toothy smile,
         lips parted over the upper teeth), snarl ('mouth-part-later': teeth bared), grimace
         ('mouth-depression-retraction'). MakeHuman expression units (CC0), helpers untouched.
      5. squint / cheek-squint / blink keys and their mixes never push the lower lid into the eye (sclera + cornea
         mesh, lid_eyeball_correction).
      6. eyeBlinkLeft/Right: the closed lid ROLLS over the eye (lid_roll_blink, item 8): stretch spread over the whole
         upper lid, clear of the cornea at 1 / 0.75 / 0.5; lid_blink_relax() then limits the lower lid / canthi.
      7. every changed / new key is re-propagated to the eyebrows, eyelashes and hair (the export re-seats the brow /
         lash cards on the skin: seat_face_cards)."""
    co = modelled_coords(bm)
    n = len(co)
    body = np.zeros(n, bool); body[:NBODY] = True
    add_model_fix(bm, lip_seal_delta(bm, rig, co))
    U = {u: unit(u, n) * body[:, None] for u in ("mouth-corner-puller", "mouth-depression", "mouth-upward-retraction",
                                                 "mouth-part-later", "mouth-depression-retraction")}
    # the MakeHuman units carry a hard crease lateral to the mouth corner (skin folds over itself once added to the
    # ARKit smile): smooth their deltas everywhere except on the lips
    lips = group_weights(bm, "lips") > 0.3
    soft = body & ~lips
    for u in ("mouth-corner-puller", "mouth-upward-retraction"):
        U[u] = smooth_delta(bm, U[u], soft, iters=16)
    cpL, cpR = lr_split(U["mouth-corner-puller"], co)
    dpL, dpR = lr_split(U["mouth-depression"], co)
    urL, urR = lr_split(U["mouth-upward-retraction"], co)
    for k, D in (("mouthSmileLeft", 0.6 * cpL), ("mouthSmileRight", 0.6 * cpR), ("mouthFrownLeft", 0.7 * dpL),
                 ("mouthFrownRight", 0.7 * dpR)):
        set_key_delta(bm, k, key_delta(bm, k) + D)
    # the summed smile compresses the skin lateral to the corner to ~25 % (renders as a crack): relax the smile's
    # delta off the lips (Laplacian), which keeps the corner lift but spreads the bunching into a soft fold
    me = bm.data
    e = np.empty(len(me.edges) * 2, dtype=np.int64); me.edges.foreach_get("vertices", e); e = e.reshape(-1, 2)
    e = e[(e < NBODY).all(1)]
    L0 = np.linalg.norm(co[e[:, 0]] - co[e[:, 1]], axis=1)
    strain = lambda D: float((np.linalg.norm(co[e[:, 0]] + D[e[:, 0]] - co[e[:, 1]] - D[e[:, 1]], axis=1) / L0).min())
    for k in ("mouthSmileLeft", "mouthSmileRight"):
        D = key_delta(bm, k)
        D2 = smooth_delta(bm, D, soft, iters=10)
        set_key_delta(bm, k, D2)
        log("%s: min edge length ratio %.2f -> %.2f after relaxing" % (k, strain(D), strain(D2)))
    for k, D in (("smileOpenLeft", urL), ("smileOpenRight", urR), ("snarl", U["mouth-part-later"]),
                 ("grimace", U["mouth-depression-retraction"])):
        set_key_delta(bm, k, D, create=True)
    eyes = find_part(rig, ".high-poly") or find_part(rig, "_eyes")
    lining = np.zeros(n, bool)
    if skin_img is not None:                             # MakeHuman's red eye-socket lining (see fix_eye_socket_keys)
        col = _vertex_uv_colors(bm, skin_img)
        lining = (col[:, 0] > col[:, 1] * 1.5) & (col[:, 0] > 0.35)
    lid_eyeball_correction(bm, eyes, co, [("eyeSquintLeft", "cheekSquintLeft", "eyeBlinkLeft"),
                                          ("eyeSquintRight", "cheekSquintRight", "eyeBlinkRight")], lining)
    lid_roll_blink(bm, eyes, co, lining)                 # closed lid: the lid rolls over the eye (brows: rigid follow)
    lid_blink_relax(bm, eyes, co)                        # lower lid / canthi: no small triangle opens > LID_RELAX_MAX
    changed =["mouthSmileLeft", "mouthSmileRight", "mouthFrownLeft", "mouthFrownRight", "eyeSquintLeft",
               "eyeSquintRight", "cheekSquintLeft", "cheekSquintRight", "eyeBlinkLeft", "eyeBlinkRight"] + EXTRA_KEYS
    propagate_keys(bm, changed)
    log("face keys improved:", ", ".join(changed))


# ------------------------------------------------------------------------------------------------------------------
# export-stage mouth work (engine mesh): dentition, collision correctives, tongue look


def rasterize_uv(obj, values, size, faces_mask=None, fill=1.0):
    """Paint a per-vertex scalar into UV space (linear inside each triangle), then grow it 8 px over the UV island
    borders. Returns (size, size) float array, row 0 = top of the image; untouched pixels = fill."""
    me = obj.data
    me.calc_loop_triangles()
    lt = me.loop_triangles
    tl = np.empty(len(lt) * 3, dtype=np.int64); lt.foreach_get("loops", tl); tl = tl.reshape(-1, 3)
    tv = np.empty(len(lt) * 3, dtype=np.int64); lt.foreach_get("vertices", tv); tv = tv.reshape(-1, 3)
    uv = np.empty(len(me.loops) * 2); me.uv_layers.active.data.foreach_get("uv", uv); uv = uv.reshape(-1, 2)
    img = np.full((size, size), np.nan)
    keep = np.ones(len(tv), bool) if faces_mask is None else faces_mask(tv)
    for (l0, l1, l2), (v0, v1, v2) in zip(tl[keep], tv[keep]):
        P = uv[[l0, l1, l2]] * size
        x0, y0 = np.floor(P.min(0)).astype(int); x1, y1 = np.ceil(P.max(0)).astype(int)
        x0, y0 = max(x0, 0), max(y0, 0); x1, y1 = min(x1, size - 1), min(y1, size - 1)
        if x1 < x0 or y1 < y0:
            continue
        xs, ys = np.meshgrid(np.arange(x0, x1 + 1) + 0.5, np.arange(y0, y1 + 1) + 0.5)
        (ax, ay), (bx, by), (cx, cy) = P
        d = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy)
        if abs(d) < 1e-12:
            continue
        l1_ = ((by - cy) * (xs - cx) + (cx - bx) * (ys - cy)) / d
        l2_ = ((cy - ay) * (xs - cx) + (ax - cx) * (ys - cy)) / d
        l3_ = 1 - l1_ - l2_
        ins = (l1_ >= -0.02) & (l2_ >= -0.02) & (l3_ >= -0.02)
        val = l1_ * values[v0] + l2_ * values[v1] + l3_ * values[v2]
        sub = img[y0:y1 + 1, x0:x1 + 1]
        sub[ins] = val[ins]
    for _ in range(8):                                  # dilate over the seams
        m = np.isnan(img)
        if not m.any():
            break
        acc = np.zeros_like(img); cnt = np.zeros_like(img)
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            sh = np.roll(np.roll(img, dy, 0), dx, 1)
            ok = ~np.isnan(sh)
            acc[ok] += sh[ok]; cnt[ok] += 1
        grow = m & (cnt > 0)
        img[grow] = acc[grow] / cnt[grow]
    img[np.isnan(img)] = fill
    return img[::-1]                                     # image rows top -> bottom


def tongue_look(tongue, kind):
    """Richer tongue material: the CC0 photo texture graded redder / less milky, darker towards the root and on the
    underside (depth in the mouth), wet but not glossy."""
    from chr_lib import asset_file
    src = asset_file("tongue", "tongue01", "tongue01_diffuse.png")
    im = bpy.data.images.load(src, check_existing=True)
    w, h = im.size
    px = np.array(im.pixels[:], dtype=np.float32).reshape(h, w, 4)[::-1, :, :3]     # top row first
    co = verts_np(tongue)
    nr = vertex_normals(tongue, co)
    y = co[:, 1]
    t = (y - y.min()) / max(1e-6, (y.max() - y.min()))          # 0 = tip (front, -Y) .. 1 = root
    # (round 5: the open mouth in the web viewer, which has no occlusion inside the mouth, showed the dorsum as a flat,
    # evenly lit pink wall up to the upper teeth: the shade starts at the tip and reaches the mid-tongue)
    shade = (0.9 - 0.64 * smoothstep(0.04, 0.7, t)) * (1 - 0.3 * smoothstep(0.0, -0.6, nr[:, 2]))
    S = rasterize_uv(tongue, shade, w)
    lum = px.mean(2, keepdims=True)
    mean = px.reshape(-1, 3).mean(0)
    graded = np.clip((lum + (px - lum) * 1.05) * (np.array([0.62, 0.33, 0.34]) / mean), 0, 1)   # tip ~ (0.62, 0.33, 0.34)
    out = graded * S[:, :, None]
    path = os.path.join(TEX, "tongue_%s_base.png" % kind)
    _write_png(path, out)
    # rough 0.42 / spec 0.45 and no occlusion lit the lowered tongue's dorsum as a grey-white sheen inside an open mouth
    # (web viewer, round 5): an ORM map (R = occlusion, darker towards the root / underside; G = roughness, wet at the
    # tip, matt further back: engines without shadows inside the mouth lit the back of the dorsum with the key light's
    # specular, a grey veil over the dark tongue, where a real mouth is in the lips' / teeth's shadow)
    ao = (1 - 0.72 * smoothstep(0.06, 0.7, t)) * (1 - 0.4 * smoothstep(0.0, -0.6, nr[:, 2]))
    A = rasterize_uv(tongue, ao, w)
    Rg = rasterize_uv(tongue, 0.45 + 0.4 * smoothstep(0.1, 0.6, t), w)
    orm = np.zeros(A.shape + (3,)); orm[:, :, 0] = np.clip(A, 0.1, 1); orm[:, :, 1] = np.clip(Rg, 0.3, 0.9)
    opath = os.path.join(TEX, "tongue_%s_orm.png" % kind)
    _write_png(opath, orm)
    mat = pbr_material("M_%s_tongue" % kind, base=path, rough=0.5, rough_img=opath, spec=0.28, sss=0.1)
    gltf_occlusion(mat)
    set_material(tongue, mat)
    log("tongue: graded texture with depth shading ->", path)


def crown_inside(t, P, margin):
    """(inside mask, push vectors) of points P against the crown volume of tooth t grown by `margin`."""
    pr = profile(t["tp"], t["jaw"])
    d = P - t["origin"]
    u, v, w = d @ t["T"], d @ t["N"], d @ t["ax"]
    f = w / t["H"]
    A = np.interp(f, pr["f"], pr["A"]) * t["W"] / 2 + margin
    B = np.interp(f, pr["f"], pr["B"]) * t["D"] / 2 + margin
    C = np.interp(f, pr["f"], pr["C"]) * t["D"] / 2
    p = np.interp(f, pr["f"], pr["P"])
    r = (np.abs(u / A) ** p + np.abs((v - C) / B) ** p) ** (1 / p)
    ins = (f > -0.05) & (f < 1.05) & (r < 1.0)
    push = (np.outer(u, t["T"]) + np.outer(v - C, t["N"])) * ((1.0 / np.maximum(r, 0.05)) - 1.0)[:, None]
    return ins, push


def teeth_collision_correctives(body, plan, keys):
    """Per face key: body vertices (lips, corners, cheek lining) that the key drives INTO a crown are pushed back out
    of it (radially in the crown's cross-section, + 0.4 mm), and the correction is spread smoothly over 5 mm of
    skin, so lips drape over the teeth instead of the teeth showing through the lips. Lower crowns follow the key's
    rigid jaw motion. The correction becomes part of the key (engines need no extra driver)."""
    from mathutils.kdtree import KDTree
    kb = body.data.shape_keys.key_blocks
    base = verts_np(body, kb["Basis"])
    lo = np.min([t["V"].min(0) for t in plan["teeth"]], 0) - 0.03
    hi = np.max([t["V"].max(0) for t in plan["teeth"]], 0) + 0.03
    near = np.all((base > lo) & (base < hi), 1)
    near[NBODY:] = False
    # vertices already touching / inside a crown at rest (mucosa against the teeth, cheek lining at the last
    # molars) are the rest state: only NEW intersections caused by a key are corrected
    rest_in = np.zeros(len(base), bool)
    for t in plan["teeth"]:
        rest_in[near] |= crown_inside(t, base[near], 0.4 * MM)[0]
    near &= ~rest_in
    idx = np.nonzero(near)[0]
    rep = {"rest contacts (ignored)": int(rest_in.sum())}
    for k in keys:
        if k not in kb:
            continue
        D = key_delta(body, k)
        R, tt = plan.get("rigid", {}).get(k, (np.eye(3), np.zeros(3)))
        crowns = []
        for t in plan["teeth"]:
            if t["jaw"] == "lower":
                t = dict(t, origin=R @ t["origin"] + tt, T=R @ t["T"], N=R @ t["N"], ax=R @ t["ax"])
            crowns.append(t)
        total = np.zeros_like(D)
        for it in range(3):
            P = (base + D + total)[idx]
            Cv = np.zeros_like(P); hit = np.zeros(len(P), bool)
            for t in crowns:
                ins, push = crown_inside(t, P, 0.4 * MM)
                big = ins & (np.linalg.norm(push, axis=1) > np.linalg.norm(Cv, axis=1))
                Cv[big] = push[big]; hit |= ins
            if not hit.any():
                break
            # spread over 5 mm with a smooth falloff (normalised blend, scaled by the closest weight)
            src = P[hit]; C0 = Cv[hit]
            kd = KDTree(len(src))
            for i, q in enumerate(src):
                kd.insert(q, i)
            kd.balance()
            spread = np.zeros_like(P)
            for j, q in enumerate(P):
                found = kd.find_range(q, 0.005)
                if not found:
                    continue
                ws = np.array([(1 - (dd / 0.005) ** 2) ** 2 for _, _, dd in found])
                cs = np.array([C0[i] for _, i, _ in found])
                spread[j] = (cs * ws[:, None]).sum(0) / ws.sum() * ws.max()
            total[idx] += spread
        if np.abs(total).max() > 1e-5:
            set_key_delta(body, k, D + total)
            rep[k] = "%d v / %.1f mm" % (int((np.linalg.norm(total, axis=1) > 1e-4).sum()), np.linalg.norm(total, axis=1).max() * 1e3)
    log("teeth collision correctives:", rep)
    return rep


# ------------------------------------------------------------------------------------------------------------------
# Lip seal by rays (judge M18: 'the lips never seal'). The live-stage rts_face_fix closes the slit between the lips
# but fades out 1-5 mm before the corners, and the ARKit / viseme keys that should keep the mouth closed leak at the
# commissures (the dentition shows through a ~1 mm hole at x = +-23 mm). Here the mouth is looked at like the QA does
# (face_qa.py lips: orthographic ray grids from the front, 25 deg left / right and 15 deg from below): wherever a ray
# meets the dentition / tongue before the lips, the lip skin around that ray is pulled towards it (vertical, Gaussian
# 2 mm around the ray line, only lip / mouth-corner skin), iterated until no ray leaks. The result goes into the rest
# shape (basis + every key, deltas unchanged) or into the key(s) of the state: viseme_PP (P / B / M), mouthPress L/R
# (split by side), mouthRollUpper / Lower (split by lip), mouthClose (with jawOpen 0.3 / 0.6 / 1: the ARKit pairing).
SEAL_VIEWS = ((0.0, 0.0), (25.0, 0.0), (-25.0, 0.0), (0.0, -15.0))
SEAL_STATES = [("rest", {}, "basis"), ("PP", {"viseme_PP": 1.0}, "viseme_PP"),
               ("press", {"mouthPressLeft": 1.0, "mouthPressRight": 1.0}, "side:mouthPress"),
               ("press50", {"mouthPressLeft": 0.5, "mouthPressRight": 0.5}, "side:mouthPress"),
               ("roll", {"mouthRollLower": 1.0, "mouthRollUpper": 1.0}, "lip:mouthRoll"),
               ("close30", {"mouthClose": 0.3, "jawOpen": 0.3}, "mouthClose"),
               ("close60", {"mouthClose": 0.6, "jawOpen": 0.6}, "mouthClose"),
               ("close100", {"mouthClose": 1.0, "jawOpen": 1.0}, "mouthClose")] + \
              [("cust:" + c, {"cust_" + c: 1.0}, "cust_" + c) for c in
               ("mouth_width_pos", "mouth_width_neg", "lip_upper_pos", "lip_upper_neg", "lip_lower_pos", "lip_lower_neg",
                "mouth_corners_pos", "mouth_corners_neg", "mouth_cupidsbow_pos", "mouth_cupidsbow_neg")]


def _shape_arrays(o):
    kb = o.data.shape_keys.key_blocks if o.data.shape_keys else None
    B = verts_np(o, kb["Basis"]) if kb else verts_np(o)
    D = {k.name: verts_np(o, k) - B for k in list(kb)[1:]} if kb else {}
    return B, D


def lip_seal_rays(rig, body, parts, log_=None, step=0.4 * MM):
    """See the block comment above. parts: the dentition / tongue objects (rigid jaw keys). Returns a report."""
    from mathutils.bvhtree import BVHTree
    log_ = log_ or log
    PB, DB = _shape_arrays(body)
    me = body.data
    me.calc_loop_triangles()
    T = np.empty(len(me.loop_triangles) * 3, dtype=np.int64); me.loop_triangles.foreach_get("vertices", T)
    T = T.reshape(-1, 3)
    B = lambda n: np.array(rig.data.bones[n].head_local)
    st = (B("lip_upper_c") + B("lip_lower_c")) / 2
    xc = abs(B("mouth_corner_l")[0]) + 0.004
    T = T[(np.abs(PB[T].mean(1) - st) < np.array([0.06, 0.06, 0.05])).all(1)]
    PS = [(_shape_arrays(o), o) for o in parts if o is not None]
    PT = []
    for (Bp, Dp), o in PS:
        o.data.calc_loop_triangles()
        t = np.empty(len(o.data.loop_triangles) * 3, dtype=np.int64); o.data.loop_triangles.foreach_get("vertices", t)
        PT.append(t.reshape(-1, 3))
    lips = group_weights(body, "lips")
    region = (np.abs(PB[:, 0]) < xc + 0.006) & (np.abs(PB[:, 2] - st[2]) < 0.014) & (PB[:, 1] < st[1] + 0.012)
    region[NBODY:] = False
    ridx = np.nonzero(region)[0]
    xs = np.arange(-xc, xc + 1e-9, step); zs = np.arange(-0.009, 0.009 + 1e-9, step)

    def leaks(w, Cb, off=0.0):
        Pb = PB + Cb
        for k, v in w.items():
            if k in DB:
                Pb = Pb + v * DB[k]
        bb = BVHTree.FromPolygons([Vector(p) for p in Pb], T.tolist(), all_triangles=True)
        bt = []
        for ((Bp, Dp), o), t in zip(PS, PT):
            Pp = Bp.copy()
            for k, v in w.items():
                if k in Dp:
                    Pp += v * Dp[k]
            bt.append(BVHTree.FromPolygons([Vector(p) for p in Pp], t.tolist(), all_triangles=True))
        out = []
        for yaw, pitch in SEAL_VIEWS:
            a, p = math.radians(yaw), math.radians(pitch)
            d = np.array([-math.sin(a) * math.cos(p), math.cos(a) * math.cos(p), -math.sin(p)])
            ux = np.array([math.cos(a), math.sin(a), 0.0]); uz = np.cross(ux, d); uz /= np.linalg.norm(uz)
            if uz[2] < 0:
                uz = -uz
            dv = Vector(d)
            for x in xs + off:
                for z in zs + off:
                    o_ = st + x * ux + z * uz - d * 0.2
                    lb = bb.ray_cast(Vector(o_), dv, 1.0)
                    tb = lb[3] if lb[0] is not None else 9.0
                    for b in bt:
                        lt = b.ray_cast(Vector(o_), dv, 1.0)
                        if lt[0] is not None and lt[3] < tb:
                            out.append((o_, d)); break
        return out, Pb
    rep = {}
    Cbase = np.zeros_like(PB)
    for name, w, target in SEAL_STATES:
        if any(k not in DB for k in w):
            continue
        C = np.zeros_like(PB)
        n0 = None
        clean = 0
        for it in range(24):                                           # two grids (offset by half a step) must
            L, Pb = leaks(w, Cbase + C, (it % 2) * step / 2)           # both come out clean
            if n0 is None:
                n0 = len(L)
            clean = clean + 1 if not L else 0
            if clean >= 2:
                break
            if not L:
                continue
            step_c = np.zeros_like(PB)
            wsum = np.zeros(len(PB))
            for o_, d in L:
                q = Pb[ridx]
                rel = q - o_
                perp = rel - np.outer(rel @ d, d)                       # distance of the lip skin from the ray line
                dist = np.linalg.norm(perp, axis=1)
                g = np.exp(-(dist / (2.0 * MM)) ** 2) * (dist < 5.0 * MM)
                zr = o_[2] + (rel @ d) * d[2]
                dz = np.where(q[:, 2] > zr, -1.0, 1.0)                  # upper lip down, lower lip up
                step_c[ridx, 2] += dz * g * 0.18 * MM
                wsum[ridx] += g
            m = wsum > 1.0
            step_c[m] /= wsum[m][:, None]                              # many leaking rays: one step, not a sum
            C += step_c
        L_end = len(leaks(w, Cbase + C)[0]) + len(leaks(w, Cbase + C, step / 2)[0]) if n0 else 0
        mv = float(np.abs(C[:, 2]).max()) * 1e3
        if target == "basis":
            Cbase = Cbase + C
        else:
            wk = {k: v for k, v in w.items() if k != "jawOpen"}
            if target.startswith("side:"):
                base = target[5:]
                sL = smoothstep(-0.002, 0.002, PB[:, 0])[:, None]
                parts_c = {base + "Left": C * sL, base + "Right": C * (1 - sL)}
            elif target.startswith("lip:"):
                base = target[4:]
                up = (PB[:, 2] > st[2])[:, None]
                parts_c = {base + "Upper": C * up, base + "Lower": C * ~up}
            else:
                parts_c = {target: C}
            for k, Ck in parts_c.items():
                DB[k] = DB[k] + Ck / max(wk.get(k, 1.0), 1e-3)
        rep[name] = "%d -> %d leaking rays (lip move %.2f mm)" % (n0 or 0, L_end, mv)
    # write: basis + every key shifted by Cbase (deltas kept), corrected keys
    kb = me.shape_keys.key_blocks
    newB = PB + Cbase
    for k in kb:
        if k.name == "Basis":
            k.data.foreach_set("co", newB.ravel())
        else:
            k.data.foreach_set("co", (newB + DB[k.name]).ravel())
    me.vertices.foreach_set("co", newB.ravel())
    me.update()
    log_("lip seal by rays:", rep)
    return rep


TONGUE_DROP = 1.8 * MM    # front of the tongue below the lower incisal edge (0.8 mm hid the lower teeth in 'aa')


def fit_tongue(tongue, plan):
    """The CC0 tongue was fitted to the CC0 teeth; the new lower crowns sit lower and further back, so the tongue is
    moved back (tip 1 mm behind the lower incisors) and down (by the lower crowns' drop), tip fully, root 40 %. The
    same offset goes into the basis and every key (all key deltas unchanged)."""
    co = verts_np(tongue)
    t = (co[:, 1] - co[:, 1].min()) / max(1e-6, co[:, 1].max() - co[:, 1].min())
    w = 1 - 0.6 * smoothstep(0.0, 1.0, t)
    lo_ci = [u for u in plan["teeth"] if u["jaw"] == "lower" and u["name"] in ("CI", "LI")]
    y_back = max(u["V"][:, 1].max() for u in lo_ci)             # most lingual point of the lower incisors
    tip = co[np.argmin(co[:, 1])]
    s_back = max(0.0, (y_back + 1.0 * MM) - tip[1])
    g_ci = plan["guide"]["lower"]["crowns"][0]
    n_ci = [u for u in plan["teeth"] if u["jaw"] == "lower" and u["name"] == "CI" and u["side"] > 0][0]
    drop = max(0.0, g_ci[2] - n_ci["V"][n_ci["FT"][:, 0] < 1.0].mean(0)[2]) * 0.8
    # the front third of the dorsum must sit below the lower incisal edge, so an open mouth shows the lower teeth
    front = t < 0.35
    edge = n_ci["origin"][2]
    need = float(co[front, 2].max() - (edge - TONGUE_DROP)) / float(w[front].min())
    drop = float(np.clip(max(drop, need), 0.0, 6.0 * MM))
    D = np.zeros_like(co); D[:, 1] = s_back * w; D[:, 2] = -drop * w
    for k in tongue.data.shape_keys.key_blocks:
        k.data.foreach_set("co", (verts_np(tongue, k) + D).ravel())
    tongue.data.vertices.foreach_set("co", (co + D).ravel())
    log("tongue: tip moved back %.1f mm, down %.1f mm (root 40 %%)" % (s_back * 1e3, drop * 1e3))


TONGUE_FLAT = 0.4        # a fully open jaw lowers the mid / back dorsum by this fraction of its height above the underside


def tongue_jaw_flatten(tongue, amount=TONGUE_FLAT, ref="jawOpen"):
    """Round 5 (judge M22, user item 10): the CC0 tongue rides the jaw rigidly, so an open mouth (jawOpen, viseme_aa,
    a shout) showed its dorsum as a pink wall up to the upper teeth, filling the whole opening. A real tongue drops and
    flattens as the jaw opens (the dark oral cavity shows above it). Every tongue key that lowers the tip (a jaw
    opening; tongueOut excluded) gets, in proportion to its tip drop vs `ref`'s, the dorsum lowered by `amount` x its
    height above the tongue's underside, on the middle / back of the tongue (tip, sides of the tip and the root
    attachment unchanged; the underside never moves)."""
    me = tongue.data
    kb = me.shape_keys.key_blocks if me.shape_keys else None
    if not kb or ref not in kb:
        return
    from mathutils.kdtree import KDTree
    co = verts_np(tongue)
    n = len(co)
    t = (co[:, 1] - co[:, 1].min()) / max(1e-6, co[:, 1].max() - co[:, 1].min())   # 0 = tip (-Y) .. 1 = root
    kd = KDTree(n)
    for i, p in enumerate(co):
        kd.insert((p[0], p[1], 0.0), i)
    kd.balance()
    zbot = np.array([min(co[j, 2] for _, j, _ in kd.find_range((p[0], p[1], 0.0), 0.003)) for p in co])
    h = np.maximum(co[:, 2] - zbot, 0.0)
    w = smoothstep(0.1, 0.4, t) * (1 - smoothstep(0.78, 1.0, t))
    dz = -amount * h * w
    tip = int(np.argmin(co[:, 1]))
    drop_ref = -float((verts_np(tongue, kb[ref]) - co)[tip, 2])
    if drop_ref < 1e-4:
        return
    done = []
    for k in kb[1:]:
        if k.name == "tongueOut" or k.name.startswith(("cust_", "cor_")):      # jaw openings only, not face shapes
            continue
        Pk = verts_np(tongue, k)
        f = float(np.clip(-(Pk - co)[tip, 2] / drop_ref, 0.0, 1.0))
        if f < 0.05:
            continue
        Pk[:, 2] += f * dz
        k.data.foreach_set("co", Pk.ravel())
        done.append("%s %.2f" % (k.name, f))
    me.update()
    log("tongue: dorsum lowered with the jaw (max %.1f mm at %s): %s" % (-dz.min() * 1e3, ref, ", ".join(done)))


# ------------------------------------------------------------------------------------------------------------------
# Face cards seated on the skin (user round-2 items 7 / 9, judge M17). MakeHuman's brow / lash cards follow the face
# through the mhclo correspondence (3 reference vertices + a fixed offset), which does not rotate the offset with the
# skin and uses reference triangles that are not the ones under the card: brows sank up to 2 mm under the brow ridge
# (24 % of the vertices already at rest), lash tips went through the closed lids. Here every card vertex is re-attached
# to the skin it lies on and every morph target of the card is rebuilt from the body's own target:
#   anchor   brows: the closest point of the skin (triangle + barycentric); lashes: the closest point of the LID MARGIN
#            (the skin triangles under the lash roots of that ribbon), so a lash follows the margin it grows from
#   frame    a weighted Kabsch rotation of the 8 nearest skin vertices around the anchor (lashes: only lid skin of
#            that lid, eyelid_upper / eyelid_lower weights), so the lash fan rolls down with the lid in a blink
#   key      vertex = anchor(key) + R(key) (vertex - anchor)(rest): exact on the skin, rigid in the margin frame
#   clearance  after that, every vertex closer to the posed skin than CARD_MIN is pushed out along the smooth normal
#            of its closest skin point (push smoothed over the card), at weight 1 and at the in-between weights where
#            blendshape interpolation cuts corners (0.5 for every key; 0.25 / 0.5 / 0.75 for blinks / squints), and
#            over a grid of pair mixes for the lashes (blink x squint / cheek squint / look / wide / brow down) solved
#            like lid_eyeball_correction (minimum-norm split of the push between the two keys).
CARD_MIN = {"eyebrows": 0.35 * MM, "eyelashes": 0.15 * MM}      # clearance to the skin (rest and every key)
LASH_ROOT_MIN = -0.6 * MM      # lash ROOT row: may sit in the margin tissue this deep (real lashes emerge from ~2 mm
LASH_ROOT_DIST = 1.2 * MM      # deep follicles; closed lids press both margins together and squints bunch the lower
                               # margin over the roots); root = within LASH_ROOT_DIST of its margin anchor at rest
CARD_KEY_MIN = {"eyebrows": 0.25 * MM, "eyelashes": 0.40 * MM}
BROW_RIGID_KEYS = ("eyeBlink", "eyeSquint")   # brows follow these keys rigidly (see seat_face_cards.follow)
BROW_RIGID_MOVE = 1.0 * MM                    # ... weighted by exp(-(anchor motion / this)^2)
LASH_TIP_GAIN = 0.08      # extra lash clearance per metre of distance from the lash root (key states): 0.8 mm at 10 mm
LASH_EYE_R = 0.034        # lash solve: skin triangles within this distance of an eyeball centre
LASH_ROLL = 1.0           # fraction of the lid-skin normal swing the lash fan follows (expression keys)
LASH_SWING_MAX = 70.0     # deg; customisation keys use the rigid (Kabsch) fit of the lid skin instead
LASH_PAIRS = ("eyeSquint", "cheekSquint", "eyeLookDown", "eyeLookUp", "eyeLookIn", "eyeLookOut", "eyeWide", "browDown")
# customisation sliders that reshape the lids: their morphs are solved in pairs with eyeBlink / eyeSquint (a blink on a
# customised lid is not the rigid lash follow of either key alone)
LASH_CUST_PAIRS = True
LASH_CUST = ("eyes_", "brows_", "face_age", "face_weight", "cheek", "forehead", "head_width")


CARD_SWEEPS = int(os.environ.get("RTS_CARD_SWEEPS", "16"))
CARD_RANDOM = 300          # seeded random mixes per family in the card solve (card_mix_states)


def card_mix_states(keys, DB, seed=101, n_random=300):
    """Weight mixes the card solve must keep clear (singles at in-between weights, eye pairs, eye x lid-shaping
    customisation pairs, expression presets with and without closed eyes, seeded random preset + customisation +
    blink mixes; the QA gate (face_qa.py) samples its own random mixes with other seeds)."""
    ks = set(keys)
    S = []
    for k in keys:
        eyek = k.startswith(("eyeBlink", "eyeSquint", "eyeWide", "cheekSquint"))
        for w in ((0.25, 0.5, 0.75, 1.0) if eyek else (0.5, 1.0)):
            S.append({k: w})
    cust = [k for k in keys if k.startswith("cust_")]
    lidc = [k for k in cust if any(c in k for c in LASH_CUST)]
    for sd in ("Left", "Right"):
        for b in [x + sd for x in LASH_PAIRS]:
            for wa in (0.5, 1.0):
                for wb in (0.5, 1.0):
                    S.append({"eyeBlink" + sd: wa, b: wb})
        for e in ("eyeBlink", "eyeSquint"):
            for c in lidc:
                for we in (0.5, 1.0):
                    S.append({e + sd: we, c: 1.0})
    pp = os.path.join(os.path.dirname(os.path.abspath(__file__)), "expressions.json")
    pres = {k: {a: b for a, b in v.items() if not a.startswith("_")} for k, v in json.load(open(pp))["presets"].items()
            if k != "Talk"} if os.path.exists(pp) else {}
    for w in pres.values():
        if w:
            S.append(dict(w))
        S.append(dict(w, eyeBlinkLeft=1.0, eyeBlinkRight=1.0))
        if w:                              # expressions (cheek squints, smiles) on lid-shaping customised faces
            for c in lidc:
                S.append(dict(w, **{c: 1.0}))
    rng = np.random.default_rng(seed)
    sliders = sorted({k.rsplit("_", 1)[0] for k in cust})
    names = list(pres)
    for i in range(n_random):
        w = dict(pres[names[rng.integers(len(names))]]) if names else {}
        for sl in sliders:
            if rng.random() < 0.35:
                v = rng.uniform(-1, 1)
                k = sl + ("_pos" if v > 0 else "_neg")
                if k in ks:
                    w[k] = abs(v)
        w["eyeBlinkLeft"] = float(rng.choice([0.0, 0.5, 1.0])); w["eyeBlinkRight"] = float(rng.choice([0.0, 0.5, 1.0]))
        S.append(w)
    for i in range(n_random):                  # customised faces at rest (a character is a mix of sliders)
        w = {}
        for sl in sliders:
            if rng.random() < 0.4:
                v = rng.uniform(-1, 1)
                k = sl + ("_pos" if v > 0 else "_neg")
                if k in ks:
                    w[k] = abs(v)
        S.append(w)
    return [{k: v for k, v in w.items() if k in ks and v} for w in S]


def _card_role(o):
    r = o.get("rts_part") or ""
    if r in ("eyebrows", "eyelashes"):
        return r
    return "eyebrows" if "eyebrow" in o.name else ("eyelashes" if "eyelash" in o.name else None)


def _vnormals(P, T, n):
    fn = np.cross(P[T[:, 1]] - P[T[:, 0]], P[T[:, 2]] - P[T[:, 0]])
    vn = np.zeros((n, 3))
    for i in range(3):
        np.add.at(vn, T[:, i], fn)
    return vn / np.maximum(np.linalg.norm(vn, axis=1), 1e-12)[:, None]


def _bary_rows(Q, A, B, C):
    """Barycentric coordinates of points Q (N,3) in triangles (A, B, C) (N,3 each), clipped into the triangle."""
    v0, v1, v2 = B - A, C - A, Q - A
    d00 = (v0 * v0).sum(1); d01 = (v0 * v1).sum(1); d11 = (v1 * v1).sum(1)
    d20 = (v2 * v0).sum(1); d21 = (v2 * v1).sum(1)
    den = np.where(np.abs(d00 * d11 - d01 * d01) < 1e-24, 1e-24, d00 * d11 - d01 * d01)
    v = (d11 * d20 - d01 * d21) / den; w = (d00 * d21 - d01 * d20) / den
    b = np.clip(np.stack([1 - v - w, v, w], 1), 0, 1)
    return b / b.sum(1)[:, None]


class _Skin:
    """Posed skin (a triangle subset of the body) for closest-point / signed-distance queries."""

    def __init__(self, P, T, n):
        from mathutils.bvhtree import BVHTree
        self.P, self.T = P, T
        self.N = _vnormals(P, T, n)
        u, inv = np.unique(T, return_inverse=True)       # only the vertices of T (polygon ids = rows of T)
        self.bvh = BVHTree.FromPolygons(P[u].tolist(), inv.reshape(-1, 3).tolist(), all_triangles=True)

    def query(self, Q):
        """(closest points, smooth normals there, signed distances, triangle ids) for points Q (N,3)."""
        L = np.empty_like(Q); F = np.zeros(len(Q), dtype=np.int64); D = np.empty(len(Q))
        for i, q in enumerate(Q):
            loc, nrm, fi, d = self.bvh.find_nearest(Vector(q))
            L[i] = loc[:]; F[i] = fi; D[i] = d
        t = self.T[F]
        b = _bary_rows(L, self.P[t[:, 0]], self.P[t[:, 1]], self.P[t[:, 2]])
        Ns = (self.N[t] * b[:, :, None]).sum(1)
        Ns /= np.maximum(np.linalg.norm(Ns, axis=1), 1e-12)[:, None]
        sd = np.where(((Q - L) * Ns).sum(1) >= 0, D, -D)
        return L, Ns, sd, F


def _card_adjacency(o):
    me = o.data
    e = np.empty(len(me.edges) * 2, dtype=np.int64); me.edges.foreach_get("vertices", e)
    return e.reshape(-1, 2)


def _push_out(skin, X, hmin, E, iters=10):
    hmin = np.broadcast_to(np.asarray(hmin, dtype=float), (len(X),))
    """Push the card vertices X out of the skin to at least hmin (smooth normal of the closest point), the push
    field smoothed along the card edges E so a ribbon bends instead of kinking. Returns (new X, worst sd before)."""
    L, N, sd, _ = skin.query(X)
    worst = float((sd - hmin).min())
    if worst >= 0:
        return X, worst
    C = np.zeros_like(X)
    n = len(X)
    deg = np.zeros(n); np.add.at(deg, E[:, 0], 1); np.add.at(deg, E[:, 1], 1)
    for it in range(iters):
        bad = sd < hmin
        if not bad.any():
            break
        C[bad] += N[bad] * (hmin[bad] - sd[bad])[:, None] * 1.2
        if len(E):
            S = np.zeros_like(C); np.add.at(S, E[:, 0], C[E[:, 1]]); np.add.at(S, E[:, 1], C[E[:, 0]])
            avg = S / np.maximum(deg, 1)[:, None]
            m = np.linalg.norm(avg, axis=1) > np.linalg.norm(C, axis=1)
            C[m] = 0.5 * C[m] + 0.5 * avg[m]               # neighbours of pushed vertices follow part of the way
        L, N, sd, _ = skin.query(X + C)
    return X + C, worst


LASH_FOLD_DOT = 0.25      # adjacent lash triangles whose normals were within ~63 deg at rest (dot > LASH_FOLD_REST) and
LASH_FOLD_REST = 0.45     # bend past ~75 deg (dot < LASH_FOLD_DOT) in a state = a fold; sharper rest bends may bend
LASH_FOLD_BEND = 0.5      # LASH_FOLD_BEND (in dot) further


def _tri_normals(P, F):
    n = np.cross(P[F[:, 1]] - P[F[:, 0]], P[F[:, 2]] - P[F[:, 0]])
    return n / np.maximum(np.linalg.norm(n, axis=1), 1e-15)[:, None]


def ribbon_tri_pairs(o):
    """(triangles (m, 3), edge-adjacent triangle pairs (q, 2)) of a card mesh (quad diagonals included: a quad that
    creases across its diagonal is a fold too)."""
    me = o.data
    me.calc_loop_triangles()
    F = np.empty(len(me.loop_triangles) * 3, dtype=np.int64); me.loop_triangles.foreach_get("vertices", F)
    F = F.reshape(-1, 3)
    ed = {}
    for i, t in enumerate(F):
        for a, b in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0])):
            ed.setdefault((min(a, b), max(a, b)), []).append(i)
    pairs = np.array([v for v in ed.values() if len(v) == 2], dtype=np.int64).reshape(-1, 2)
    return F, pairs


def ribbon_folds(P, F, pairs, d0):
    """Boolean per pair: folded in positions P (rest normal dots d0)."""
    nn = _tri_normals(P, F)
    d = (nn[pairs[:, 0]] * nn[pairs[:, 1]]).sum(1)
    return np.where(d0 > LASH_FOLD_REST, d < LASH_FOLD_DOT, d < d0 - LASH_FOLD_BEND)


def _fold_setup(p):
    F, pairs = ribbon_tri_pairs(p["o"])
    nn = _tri_normals(p["X0"], F)
    p["fold"] = (F, pairs, (nn[pairs[:, 0]] * nn[pairs[:, 1]]).sum(1))


def _hmin(p, key=True):
    base = (CARD_KEY_MIN if key else CARD_MIN)[p["role"]]
    if p["role"] != "eyelashes":
        return np.full(len(p["X0"]), base)
    # expression / customisation states: the further a lash vertex is from its root, the more clearance it keeps
    # (blendshapes add the lid's rotations linearly, so mixes of many keys cut in by ~angle x angle x length)
    if not key:
        return np.where(p["root"], LASH_ROOT_MIN, base)
    # ... but never more than the lash has at rest (long curled lashes whose tips lie close to the lid / brow would
    # otherwise ask every near-rest customisation state for clearance the rest pose itself does not have)
    tip = np.maximum(CARD_MIN["eyelashes"], np.minimum(base + LASH_TIP_GAIN * p["tipd"], p["sd_rest"]))
    return np.where(p["root"], LASH_ROOT_MIN, tip)


LASH_SPEC, LASH_ROUGH = 0.08, 0.8
SEAT_PASSES = 2          # the second pass starts from the first pass's seated rest (roots on the margin, lashes clear
                         # of the lid): its constraint solve converges (female eyelashes03 / 04: 325 -> 0 violations)


def seat_face_cards(rig, body, log_=None, passes=SEAT_PASSES):
    """Re-seat every eyebrow / eyelash mesh of `rig` on the skin of `body` and rebuild all their morph targets from
    the body's face + cust_* targets (see the block comment above), `passes` times (each pass starts from the previous
    one's cards). Engine-stage (baked body: basis + deltas), after bake_body_for_export. Returns the last report."""
    rep = {}
    for i in range(passes):
        rep = _seat_face_cards_once(rig, body, log_)
    # lash cards: less specular and rougher (the seated lashes of a closed eye now lie outside the lid, in the light:
    # at specularFactor 0.6 they read as grey-white ribbons in the web viewer)
    for o in children_meshes(rig):
        if _card_role(o) == "eyelashes":
            for m in o.data.materials:
                for b in [n for n in (m.node_tree.nodes if m and m.node_tree else []) if n.type == 'BSDF_PRINCIPLED']:
                    b.inputs["Specular IOR Level"].default_value = LASH_SPEC
                    if not b.inputs["Roughness"].is_linked:
                        b.inputs["Roughness"].default_value = LASH_ROUGH
    return rep


def _seat_face_cards_once(rig, body, log_=None):
    from mathutils.kdtree import KDTree
    log_ = log_ or log
    cards = [o for o in children_meshes(rig) if _card_role(o)]
    if not cards or not body.data.shape_keys:
        return {}
    kb = body.data.shape_keys.key_blocks
    PB = verts_np(body, kb["Basis"])
    n = len(PB)
    keys = [k.name for k in kb if k.name != "Basis" and not k.name.startswith("cor_")]
    DB = {k: verts_np(body, kb[k]) - PB for k in keys}
    me = body.data
    me.calc_loop_triangles()
    Tall = np.empty(len(me.loop_triangles) * 3, dtype=np.int64); me.loop_triangles.foreach_get("vertices", Tall)
    Tall = Tall.reshape(-1, 3)
    ez = np.mean([rig.data.bones[b].head_local.z for b in ("eye_l", "eye_r")])
    T = Tall[PB[Tall].mean(1)[:, 2] > ez - 0.09]                     # face + scalp triangles
    rest = _Skin(PB, T, n)
    ecen = np.array([rig.data.bones["eye_" + s_].head_local for s_ in ("l", "r")])
    tc_ = PB[T].mean(1)
    T_eye = T[np.linalg.norm(tc_[:, None, :] - ecen[None], axis=2).min(1) < LASH_EYE_R]   # lash solve: periorbital skin
    kd = KDTree(n)
    for i in np.unique(T):
        kd.insert(PB[i], int(i))
    kd.balance()
    lidw = {g: group_weights(body, g) for g in ("eyelid_upper_l", "eyelid_upper_r", "eyelid_lower_l", "eyelid_lower_r")}
    eye = {s: np.array(rig.data.bones["eye_" + s].head_local) for s in ("l", "r")}
    plans = []
    for o in cards:
        rl = _card_role(o)
        ck = o.data.shape_keys
        X0 = verts_np(o, ck.key_blocks["Basis"]) if ck else verts_np(o)
        E = _card_adjacency(o)
        m = len(X0)
        tri = np.zeros(m, dtype=np.int64); bar = np.zeros((m, 3)); anc = np.zeros((m, 3))
        nb_i = np.zeros((m, 8), dtype=np.int64); nb_w = np.zeros((m, 8))
        if rl == "eyebrows":
            L, N, sd, F = rest.query(X0)
            tri[:] = F
            # rest clearance: vertices under / on the skin go out to CARD_MIN along the smooth normal
            X0 = np.where((sd < CARD_MIN[rl])[:, None], L + N * CARD_MIN[rl], X0)
            L, N, sd, F = rest.query(X0)
            tri[:] = F
            t = T[tri]
            bar[:] = _bary_rows(L, PB[t[:, 0]], PB[t[:, 1]], PB[t[:, 2]])
            anc[:] = L
            wsel = None
        else:
            lab = mesh_islands(o)
            L, N, sd, F = rest.query(X0)
            for isl in np.unique(lab):
                sel = np.nonzero(lab == isl)[0]
                c = X0[sel].mean(0)
                s = "l" if c[0] > 0 else "r"
                lid = "upper" if c[2] > eye[s][2] else "lower"
                root = sel[sd[sel] < sd[sel].min() + 1.0 * MM]
                mt = set(F[root].tolist())
                ring = set(T[list(mt)].ravel().tolist())
                mt |= {j for j in range(len(T)) if ring.intersection(T[j].tolist())} if len(ring) < 400 else set()
                mt = np.array(sorted(mt))
                sub = _Skin(PB, T[mt], n)
                Ls, Ns, sds, Fs = sub.query(X0[sel])
                tri[sel] = mt[Fs]; anc[sel] = Ls
                t = T[tri[sel]]
                bar[sel] = _bary_rows(Ls, PB[t[:, 0]], PB[t[:, 1]], PB[t[:, 2]])
                w = lidw["eyelid_%s_%s" % (lid, s)]
                for j, vi in enumerate(sel):
                    found = kd.find_n(anc[vi], 24)
                    cand = [(i, d) for _, i, d in found if w[i] > 0.05][:8]
                    if len(cand) < 4:
                        cand = [(i, d) for _, i, d in found][:8]
                    while len(cand) < 8:
                        cand.append(cand[-1])
                    ii = np.array([c_[0] for c_ in cand]); dd = np.array([c_[1] for c_ in cand])
                    sg = max(1.8 * MM, float(np.sort(dd)[3]))
                    nb_i[vi] = ii; nb_w[vi] = np.exp(-(dd / sg) ** 2) * np.maximum(w[ii], 0.05)
            wsel = True
        if rl == "eyebrows":
            for vi in range(m):
                found = kd.find_n(anc[vi], 8)
                ii = np.array([i for _, i, _ in found]); dd = np.array([d for _, _, d in found])
                sg = max(1.5 * MM, float(np.sort(dd)[3]))
                nb_i[vi] = ii; nb_w[vi] = np.exp(-(dd / sg) ** 2)
        # lashes: the whole ribbon out of the skin at rest (roots were buried up to 0.7 mm in the margin)
        # root row: per ribbon, the vertices within LASH_ROOT_DIST of that ribbon's closest approach to its margin
        # (some styles are fitted with the whole ribbon standing ~1-2 mm off the margin: an absolute threshold found
        # 9 roots instead of ~35 on the female's eyelashes03 / 04)
        root = np.zeros(m, bool)
        if rl == "eyelashes":
            da = np.linalg.norm(X0 - anc, axis=1)
            for isl in np.unique(lab):
                sel = lab == isl
                root[sel] = da[sel] < da[sel].min() + LASH_ROOT_DIST
        if rl == "eyelashes":             # rest: the root row CARD_MIN off the margin, the visible lash CARD_KEY_MIN (the
            # clearance every key state asks for: a rest pose with less makes the near-rest states unsatisfiable)
            X0, _ = _push_out(rest, X0, np.where(root, CARD_MIN[rl], CARD_KEY_MIN[rl]), E)
            L, N, sd, F = rest.query(X0)
        sd_rest = rest.query(X0)[2]
        side = np.where(X0[:, 0] > 0, 0, 1)
        ce = np.array([eye["l"], eye["r"]])[side]
        Nr = _vnormals(PB, T, n)[nb_i]; Pr = PB[nb_i] - ce[:, None]
        outward = (Nr * Pr).sum(2) > 0.3 * np.linalg.norm(Pr, axis=2)     # outer lid skin, not the socket lining
        nf_w = nb_w * np.where(outward, 1.0, 1e-3)
        plans.append(dict(o=o, role=rl, X0=X0, E=E, tri=tri, bar=bar, nb_i=nb_i, nb_w=nb_w, nf_w=nf_w, side=side,
                          root=root, tipd=np.linalg.norm(X0 - anc, axis=1), sd_rest=sd_rest, D={}))
    for p in plans:
        if p["role"] == "eyelashes":
            _fold_setup(p)
    # ---- rebuild every key from the body key
    # eyeball centres per key (the lash frame rolls about the eyeball like the lid margin it grows from)
    eyes_o = find_part(rig, "_eyes") or find_part(rig, ".high-poly")
    ES = None
    if eyes_o is not None:
        ek = eyes_o.data.shape_keys
        EB = verts_np(eyes_o, ek.key_blocks["Basis"]) if ek else verts_np(eyes_o)
        ED = {k.name: verts_np(eyes_o, k) - EB for k in list(ek.key_blocks)[1:]} if ek else {}
        scl = np.zeros(len(EB), bool)
        for poly in eyes_o.data.polygons:
            if poly.material_index == 0:
                scl[list(poly.vertices)] = True
        ES = (EB, ED, [scl & (EB[:, 0] > 0), scl & (EB[:, 0] < 0)])

    def eye_centres(w):
        if ES is None:
            return np.array([eye["l"], eye["r"]])
        EB, ED, sel = ES
        P = EB.copy()
        for k, v in w.items():
            if k in ED:
                P += v * ED[k]
        out = []
        for m_ in sel:
            Q = P[m_]
            sol = np.linalg.lstsq(np.c_[2 * Q, np.ones(len(Q))], (Q ** 2).sum(1), rcond=None)[0]
            out.append(sol[:3])
        return np.array(out)
    C0 = eye_centres({})

    def swing(a, b):
        """Per-row minimal rotations taking unit vectors a to b."""
        v = np.cross(a, b); c = (a * b).sum(1); s2 = (v * v).sum(1)
        K = np.zeros((len(a), 3, 3))
        K[:, 0, 1], K[:, 0, 2], K[:, 1, 0], K[:, 1, 2], K[:, 2, 0], K[:, 2, 1] = -v[:, 2], v[:, 1], v[:, 2], -v[:, 0], -v[:, 1], v[:, 0]
        f = np.where(s2 > 1e-16, (1 - c) / np.maximum(s2, 1e-16), 0.5)
        return np.eye(3)[None] + K + np.einsum("nij,njk->nik", K, K) * f[:, None, None]

    N0 = _vnormals(PB, T, n)
    NK = None

    def follow(p, P, k=None):
        t = T[p["tri"]]
        A = (P[t] * p["bar"][:, :, None]).sum(1)
        A0 = (PB[t] * p["bar"][:, :, None]).sum(1)
        if p["role"] == "eyelashes" and not (k or "").startswith("cust_"):   # keep the lash angle to the lid skin: swing of the outer
            # lid-skin normal around the anchor (rest -> key), so the fan hangs forward-down when the lid is closed
            N1 = NK if NK is not None else _vnormals(P, T, n)
            w = p["nf_w"][:, :, None]
            a = (N0[p["nb_i"]] * w).sum(1); b = (N1[p["nb_i"]] * w).sum(1)
            a /= np.linalg.norm(a, axis=1)[:, None]; b /= np.linalg.norm(b, axis=1)[:, None]
            b = a + LASH_ROLL * (b - a); b /= np.linalg.norm(b, axis=1)[:, None]
            # canthus normals are unstable: limit the swing to LASH_SWING_MAX
            ang = np.arccos(np.clip((a * b).sum(1), -1, 1)); lim = np.radians(LASH_SWING_MAX)
            big = ang > lim
            if big.any():
                f = (lim / ang[big])[:, None]
                b[big] = a[big] + f * (b[big] - a[big]); b[big] /= np.linalg.norm(b[big], axis=1)[:, None]
            R = swing(a, b)
            X = A + np.einsum("nij,nj->ni", R, p["X0"] - A0)
            # the root row (in the lid margin) translates with its anchor: its delta is the barycentric mix of the skin
            # triangle's deltas, so at ANY blend of keys it stays at its rest offset from that (blended) skin triangle
            rt = p["root"]
            X[rt] = A[rt] + (p["X0"][rt] - A0[rt])
            return X
        if p["role"] == "eyebrows" and (k or "").startswith(BROW_RIGID_KEYS):
            # lid keys roll the preseptal skin under the brows' lower edge (lid_roll_blink): each brow follows as ONE
            # rigid piece (Kabsch fit of its anchors) instead of stretching down with the lid; the clearance push
            # afterwards keeps it on the skin
            # (weighted: anchors on skin the lid key moves get little say, so the brow keeps to its stable part)
            X = p["X0"].copy()
            wa = np.exp(-(np.linalg.norm(A - A0, axis=1) / BROW_RIGID_MOVE) ** 2) + 1e-4
            for sd_ in (0, 1):
                m_ = p["side"] == sd_
                if m_.sum() >= 3:
                    w_ = wa[m_][:, None]
                    ca, cb = (A0[m_] * w_).sum(0) / w_.sum(), (A[m_] * w_).sum(0) / w_.sum()
                    H = ((A0[m_] - ca) * w_).T @ (A[m_] - cb)
                    U, S_, Vt = np.linalg.svd(H)
                    dd = np.sign(np.linalg.det(Vt.T @ U.T))
                    R = Vt.T @ np.diag([1, 1, dd]) @ U.T
                    X[m_] = (p["X0"][m_] - ca) @ R.T + cb
            return X
        R0 = PB[p["nb_i"]]; R1 = P[p["nb_i"]]; w = p["nb_w"][:, :, None]
        c0 = (R0 * w).sum(1) / w.sum(1); c1 = (R1 * w).sum(1) / w.sum(1)
        H = np.einsum("nki,nkj->nij", (R0 - c0[:, None]) * w, R1 - c1[:, None])
        U, S, Vt = np.linalg.svd(H)
        d = np.sign(np.linalg.det(np.einsum("nij,njk->nik", np.transpose(Vt, (0, 2, 1)), np.transpose(U, (0, 2, 1)))))
        Dm = np.zeros((len(d), 3, 3)); Dm[:, 0, 0] = 1; Dm[:, 1, 1] = 1; Dm[:, 2, 2] = d
        R = np.einsum("nij,njk,nkl->nil", np.transpose(Vt, (0, 2, 1)), Dm, np.transpose(U, (0, 2, 1)))
        return A + np.einsum("nij,nj->ni", R, p["X0"] - A0)
    t0 = time.time()
    moved = {}
    for k in keys:
        Dk = DB[k]
        P = PB + Dk
        active = [p for p in plans if np.abs(Dk[np.unique(np.concatenate([T[p["tri"]].ravel(), p["nb_i"].ravel()]))]).max() > 1e-5]
        if not active:
            continue
        skins = {}
        NK = _vnormals(P, T, n)
        for p in active:
            X = follow(p, P, k)
            p.setdefault("F", {})[k] = X - p["X0"]                   # the pure follow (for the report)
            if p["role"] not in skins:
                skins[p["role"]] = _Skin(P, T_eye if p["role"] == "eyelashes" else T, n)
            X, _ = _push_out(skins[p["role"]], X, _hmin(p), p["E"])
            p["D"][k] = X - p["X0"]
        moved[k] = len(active)
    # ---- mixes: blendshapes interpolate linearly (a rolled lid / rotated lash cuts the corner at in-between
    # weights) and add up (a blink on a squinting or customised lid is not the rigid follow of either key alone).
    # Constraint projection (Kaczmarz sweeps) over a state set: every card vertex closer than CARD_KEY_MIN to the
    # posed skin of a state is pushed out, and the push is split over the keys of that state in proportion to
    # weight x priority (minimum norm): expression keys take most of it, customisation keys little (their morphs
    # also show on an open, unexpressive face, where a lash pushed off the lid would float).
    states = card_mix_states(keys, DB, n_random=CARD_RANDOM)
    # the states face_qa.py gates: card_mix_states appends its 2 x n_random random mixes last; of the rest, the preset x
    # lid-shaping customisation combos (a preset + one cust_* key) are reported there, not gated
    gated = [w for w in states[:len(states) - 2 * CARD_RANDOM]
             if not any(k.startswith("cust_") for k in w) or sum(not k.startswith("cust_") for k in w) <= 1]
    prio = lambda k: 0.15 if k.startswith("cust_") else 1.0
    stats = dict(states=len(states), sweeps=0, violations=[], struct_sweeps=0, struct_violations=[])

    def sweep_states(sts, sweep, brows=True):
        viol = 0
        for w in sts:                     # brows (thin cards lying on the skin) take three passes, lashes every sweep
            act = [p for p in plans if ((brows and sweep < 3) or p["role"] == "eyelashes") and any(k in p["D"] for k in w)]
            if not act:
                continue
            P = PB.copy()
            for k, v in w.items():
                P += v * DB[k]
            skin = _Skin(P, T_eye if all(p["role"] == "eyelashes" for p in act) else T, n)
            for p in act:
                ks = [k for k in w if k in p["D"]]
                X = p["X0"] + sum(w[k] * p["D"][k] for k in ks)
                X2, worst = _push_out(skin, X, _hmin(p), p["E"], iters=4)
                if worst >= -1e-6:
                    continue
                viol += 1
                if os.environ.get("RTS_CARD_DEBUG") and sweep == CARD_SWEEPS - 1:
                    L_, N_, sd_, F_ = skin.query(X)
                    bad_ = np.nonzero(sd_ < _hmin(p))[0]
                    log_("CARDDBG %s %s worst %.2f mm verts %s keys %s" % (p["o"].name, "rand" if len(w) > 6 else "",
                         worst * 1e3, bad_[:6].tolist(), {k: round(v, 2) for k, v in w.items() if not k.startswith("cust_")}
                         if len(w) > 6 else w))
                C = X2 - X
                den = sum(prio(k) * w[k] * w[k] for k in ks)
                for k in ks:
                    p["D"][k] = p["D"][k] + C * (prio(k) * w[k] / den)
        return viol
    for sweep in range(CARD_SWEEPS):
        viol = sweep_states(states, sweep)
        stats["sweeps"] = sweep + 1; stats["violations"].append(viol)
        if viol == 0:
            break
    # the male's laugh / random mixes (cheek squint + smile on small-eyed, narrow faces) keep ~300 states unsatisfiable,
    # and their pushes fight the structured states: which of those end clean was luck (round 5: one lash vertex of
    # eyelashes04 3 mm inside the skin at eyeSquintLeft + cust_eyes_height_neg after an unrelated lid change). Finish
    # with lash sweeps over the STRUCTURED states only (the ones face_qa.py gates: singles, in-betweens, eye pairs,
    # customisation x blink / squint, presets, presets + blink); the random mixes stay reported.
    for sweep in range(CARD_SWEEPS if viol else 0):
        v2 = sweep_states(gated, CARD_SWEEPS + sweep, brows=False)
        stats["struct_sweeps"] = sweep + 1; stats["struct_violations"].append(v2)
        if v2 == 0:
            break
    log_("face cards: %d mix states, violations per sweep %s; then %d structured states: %s" % (
        stats["states"], stats["violations"], len(gated), stats["struct_violations"]))
    # ---- lash ribbon folds (reported): neighbouring lash strands can cross at the canthi of a closed eye, where the
    # lid margin bunches (per-strand rigid follows and a displacement-smoothing unfold pass were tried: more folds /
    # broken clearance in the mixes, so neither is used); face_qa.py reports them per state
    fold_log = ["%s %d" % (p["o"].name, sum(int(ribbon_folds(p["X0"] + p["D"][k], *p["fold"]).sum()) for k in p["D"]
                if not k.startswith("cust_"))) for p in plans if p["role"] == "eyelashes"]
    if fold_log:
        log_("lash ribbon folds summed over the single face keys: " + ", ".join(fold_log))
    # ---- write the cards: new basis + keys (old keys replaced; keys that no longer move the card removed)
    rep = {}
    for p in plans:
        o = p["o"]
        if o.data.shape_keys:
            names_old = [k.name for k in o.data.shape_keys.key_blocks]
            o.shape_key_clear()
        o.data.vertices.foreach_set("co", p["X0"].ravel())
        o.data.update()
        o.shape_key_add(name="Basis", from_mix=False)
        nk = 0
        for k in keys:                                                 # body key order
            D = p["D"].get(k)
            if D is None or np.abs(D).max() < 1e-5:
                continue
            sk = o.shape_key_add(name=k, from_mix=False)
            sk.data.foreach_set("co", (p["X0"] + D).ravel()); sk.value = 0.0
            nk += 1
        o.data.shape_keys.use_relative = True
        if p["role"] == "eyelashes":           # lash root row flag (QA: face_qa.py allows LASH_ROOT_MIN there)
            at = o.data.attributes.get("rts_lash_root") or o.data.attributes.new("rts_lash_root", 'FLOAT', 'POINT')
            at.data.foreach_set("value", p["root"].astype(np.float32))
        rest_sd = rest.query(p["X0"])[2]
        dev = {k: float(np.linalg.norm(p["D"][k] - p["F"][k], axis=1).max()) for k in p["D"] if k in p.get("F", {})}
        dc = {k: v for k, v in dev.items() if k.startswith("cust_")}; de = {k: v for k, v in dev.items() if k not in dc}
        rep[o.name] = dict(keys=nk, rest_min_mm=round(float(rest_sd.min()) * 1e3, 3),
                           push_expr_mm=round(max(de.values(), default=0) * 1e3, 2), push_expr_key=max(de, key=de.get) if de else None,
                           push_cust_mm=round(max(dc.values(), default=0) * 1e3, 2), push_cust_key=max(dc, key=dc.get) if dc else None)
    log_("face cards seated on the skin (%.1fs): %s" % (time.time() - t0, ", ".join(
        "%s %d keys (rest min %+.2f mm; clearance pushes beyond the follow: expression %.2f mm %s, cust %.2f mm %s)" % (
            k, v["keys"], v["rest_min_mm"], v["push_expr_mm"], v["push_expr_key"], v["push_cust_mm"], v["push_cust_key"])
        for k, v in rep.items())))
    return rep


TONGUE_SUBD = 1          # Catmull-Clark levels on the tongue (the CC0 tongue01's 448 tris showed a polygonal tip / rim)


def subdivide_keyed(o, levels=1):
    """Catmull-Clark subdivision of a skinned, shape-keyed mesh in place: every key is evaluated through the same
    subdivision (so each key stays the exact subdivided shape and the deltas stay consistent), vertex groups, UVs and
    materials are interpolated / kept by Blender's evaluated mesh. Other modifiers are ignored while evaluating."""
    if o is None or levels < 1:
        return
    me = o.data
    kb = me.shape_keys.key_blocks if me.shape_keys else None
    names = [k.name for k in kb] if kb else []
    saved = [(m, m.show_viewport) for m in o.modifiers]
    for m, _ in saved:
        m.show_viewport = False
    sd = o.modifiers.new("rts_subd", 'SUBSURF'); sd.levels = levels; sd.render_levels = levels; sd.quality = 3
    only, act = o.show_only_shape_key, o.active_shape_key_index
    coords = []
    dg = bpy.context.evaluated_depsgraph_get()
    if kb:
        o.show_only_shape_key = True
        for i in range(len(names)):
            o.active_shape_key_index = i
            me.update(); dg.update()
            ev = o.evaluated_get(dg); m2 = ev.to_mesh()
            a = np.empty(len(m2.vertices) * 3); m2.vertices.foreach_get("co", a); coords.append(a.reshape(-1, 3))
            ev.to_mesh_clear()
        o.active_shape_key_index = 0
    me.update(); dg.update()
    new = bpy.data.meshes.new_from_object(o.evaluated_get(dg), preserve_all_data_layers=True, depsgraph=dg)
    o.show_only_shape_key, o.active_shape_key_index = only, act
    o.modifiers.remove(sd)
    for m, v in saved:
        m.show_viewport = v
    nv0, nv = len(me.vertices), len(new.vertices)
    old_name = me.name
    o.data = new
    if coords:
        new.vertices.foreach_set("co", coords[0].ravel())
        o.shape_key_add(name=names[0], from_mix=False)
        for nm, c in zip(names[1:], coords[1:]):
            k = o.shape_key_add(name=nm, from_mix=False); k.data.foreach_set("co", c.ravel()); k.value = 0.0
    if me.users == 0:
        bpy.data.meshes.remove(me)
    new.name = old_name
    new.update()
    log("%s subdivided x%d: %d -> %d verts, %d keys, %d vertex groups" % (o.name, levels, nv0, nv, max(len(names) - 1, 0),
                                                                           len(o.vertex_groups)))


def export_face(rig, kind, body, seat_cards=True):
    """Export-stage mouth work on the baked engine body (call after bake_body_for_export, before clean_weights)."""
    guide = find_part(rig, "_teeth")
    teeth, plan = make_teeth_object(rig, kind, guide, body)
    depth = mouth_cavity_depth(body, rig)
    a = body.data.attributes.get("rts_mouth_depth") or body.data.attributes.new("rts_mouth_depth", 'FLOAT', 'POINT')
    a.data.foreach_set("value", depth.astype(np.float32))
    sc = group_weights(body, "scalp")              # smoothed hairline (as bake_skin / hair_gen): face_qa's scalp test
    if sc.any():
        a = body.data.attributes.get("rts_scalp") or body.data.attributes.new("rts_scalp", 'FLOAT', 'POINT')
        from chr_lib import smooth_scalp
        a.data.foreach_set("value", smooth_scalp(body.data, sc).astype(np.float32))
    teeth_collision_correctives(body, plan, [k for k in FACE_KEYS if k != "tongueOut"])
    fit_tongue(find_part(rig, "_tongue"), plan)
    tongue_jaw_flatten(find_part(rig, "_tongue"))                     # open mouth: the tongue drops (judge M22)
    subdivide_keyed(find_part(rig, "_tongue"), TONGUE_SUBD)          # smooth tongue silhouette (judge M22: 448 tris)
    lip_seal_rays(rig, body, [teeth, find_part(rig, "_tongue")])   # closed-mouth shapes sealed (judge M18)
    tongue_look(find_part(rig, "_tongue"), kind)
    mouth_occlusion(body, kind)                   # the mouth interior's occlusion for engines (round 5)
    if seat_cards:                                # brows / lashes on the skin in every key (base_humans re-runs it
        seat_face_cards(rig, body)                # after cust_lib.sparsify_card_keys)
    return teeth, plan
