"""ARCHER outfit (RTS ranged unit): quilted linen gambeson (mid-thigh, long sleeves), green wool hood with a shoulder
cowl, leather bracers over the sleeves, fingerless leather gloves, tall leather boots with a turned cuff, a belt with
pouch + knife on the gambeson, a back quiver with arrows on a baldric, and a yew longbow prop with its own bones and
sockets. The archer re-uses the PEASANT's linen shirt and wool trousers (mix-and-match: the same MPFB assets), and its
hood fits the peasant too (outfit 'peasant_hood' in outfit_civ.py).

Every garment is an MPFB clothes asset rts_archer_<slot>[_<kind>] authored on the live base human (MakeClothes), skinned
to the shared rts_human skeleton, hiding the skin it covers and following the cust_* body morphs. The garments grow from
the same tights lines as the peasant's shirt / trousers (outfit_civ_lib), so the layers keep their order under skinning.

run:  BLENDER_USER_RESOURCES=$PWD/blender_profile $BL -b out/base_<kind>.blend --python-exit-code 1 \\
        -P scripts/outfit_archer.py -- <mode> [pieces ...]
modes: preview   clay previews of the unwritten pieces (renders/civ/prev_archer_*.png)
       check     penetration gate on the unwritten pieces (renders/civ/check_archer_<kind>.json)
       author    write the MPFB clothes assets (assets/mpfb_assets/clothes/rts_archer_*) + the longbow prop GLB
       bow       only the longbow prop (out/civ/props/longbow.glb)
Dressing / export / QA: scripts/outfit_civ.py (scripts/outfit_civ.sh).
"""
import sys, os, math, time, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from outfit_civ_geo import *
import outfit_peasant as OP

OUTFIT = "archer"
ORDER = ["gambeson", "belt", "hood", "quiver", "bracers", "gloves", "boots"]


# ================================================================================================= gambeson
def gambeson(A, M, shirt_d, legs_d):
    """quilted linen gambeson: round neck, long sleeves over the shirt to the wrist, hangs from the chest and is
    gathered by the belt, mid-thigh skirt with front and back slits; leather binding on every edge. Padded: 2-3 cm off
    the skin, stiffer drape than the wool tunic (quilting channels along V in civ_quilt)."""
    cb = A.cb
    z_join = M.z_belt - 0.012
    z_hem = M.z_knee + 0.20
    t_sleeve = {s: 0.955 for s in SIDES}                   # ends at the wrist, clear of the gloved hand

    def keep(c, f):
        if c[2] < z_join - 0.012:
            return False
        s = side_of(c)
        hw = sum(sum(x for b, x in cb.W[i].items() if b.startswith(("hand", "thumb", "index", "middle", "ring", "pinky")))
                 for i in f) / len(f)
        if hw > 0.3:
            return False
        aw = sum(sum(x for b, x in cb.W[i].items() if b.startswith(("upperarm", "lowerarm", "hand"))) for i in f) / len(f)
        if aw > 0.45 and M.arm_t(c, s) > t_sleeve[s] + 0.02:
            return False
        if abs(c[0]) < 0.17 and c[2] > OP.neckline_z(M, phi_of(c), front=0.035, back=0.01) + 0.01:
            return False
        return True
    G = Garment(A, keep)
    L = classify_loops(G, M)
    for s in SIDES:
        O, Nn = arm_plane(M, s, t_sleeve[s])
        G.snap_loop_plane(L["cuff_" + s], O, Nn)
    G.snap_loop_plane(L["waist"], np.array([0, 0, z_join]), np.array([0, 0, 1.0]))
    G.snap_loop_zcurve(L["neck"], lambda p: OP.neckline_z(M, phi_of(p), front=0.035, back=0.01))
    T = G.P.copy()
    c = np.zeros(len(T))
    for i, p in enumerate(T):
        r = G.region[i]
        pit = smoothstep(0.03, 0.10, OP.armpit_dist(M, p))
        if r.startswith(("arm", "hand")):
            s = r[-1]; t = M.arm_t(p, s)
            c[i] = 0.016 + 0.004 * smoothstep(0.2, 0.6, t) - 0.004 * smoothstep(0.85, 0.97, t)
        else:
            blouse = smoothstep(z_join, z_join + 0.04, p[2]) * smoothstep(z_join + 0.16, z_join + 0.06, p[2])
            c[i] = 0.021 + 0.006 * blouse - 0.003 * smoothstep(z_join + 0.03, z_join, p[2])
        c[i] = lerp(0.017, c[i], pit)
    P, h = drape_lines(cb, G.idx, G.F, T, c, unders=[(shirt_d["h"], 0.0085), (legs_d["h"], 0.007)], iters=40,
                       verbose="gambeson", bridge=30)
    torso = np.array([not r.startswith(("arm", "hand")) for r in G.region])
    h = hang_torso(cb, G.idx, T, h, G.F, torso, z_top=M.z_sh - 0.075, z_join=z_join, gather=0.10, drop=0.06,
                   side_drop=0.22, fold_amp=0.18, fold_n=13, seed=71)
    P = T + cb.line_N(G.idx) * h[:, None]
    fn = fold_noise(P, 71, 4.0)
    for i, p in enumerate(P):
        r = G.region[i]
        if r.startswith(("arm", "hand")):
            s = r[-1]; t = M.arm_t(p, s); te = M.elbow_t(s)
            th = OP.ang(p, OP.line_frame(M, p, s))
            # padded sleeve: soft horizontal compression rings at the elbow crease, a looser upper arm
            h[i] += 0.0025 * smoothstep(te - 0.12, te, t) * smoothstep(te + 0.12, te, t) * \
                (0.5 + 0.5 * math.sin(t * 60 + 1.3 * th + 2 * fn[i]))
        else:
            bl = smoothstep(z_join, z_join + 0.03, p[2]) * smoothstep(z_join + 0.12, z_join + 0.04, p[2])
            h[i] += 0.0035 * bl * (0.5 + 0.5 * math.sin(phi_of(p) * 14 + 2.5 * fn[i]))
    P = T + cb.line_N(G.idx) * h[:, None]
    G.P = P
    m = G.to_mesh("cloth", axes=A.axes)
    comp = Composite([skin_target(cb, 0.0, exclude=("upperarm", "lowerarm", "hand", "thumb", "index", "middle",
                                                    "ring", "pinky")),
                      Target(legs_d["P"], legs_d["F"], 0.0), Target(shirt_d["P"], shirt_d["F"], 0.0)])
    hf = HangField(comp, z_hem - 0.06, z_join + 0.06)
    lm = legs_d["mesh"]; LP = lm.arr()
    kd = KDTree(len(LP))
    for i, p in enumerate(LP):
        kd.insert(p, i)
    kd.balance()

    sw_legs = SurfaceWeights(lm)

    def law(p, z, side):
        wu = sw_legs(p)
        wt = norm_w({"thigh_" + side: 0.85, "thigh_twist_01_" + side: 0.15})
        return mix_w(wu, wt, smoothstep(M.z_hip + 0.01, M.z_crotch - 0.06, z))
    hang_skirt(m, L["waist"], hf, z_hem, clear=0.022, flare=0.03, n_folds=10, fold_amp=0.006, row_step=0.034,
               slit_top=M.z_crotch - 0.01, back_drop=0.015, law=law,
               min_clear_fn=lambda z: 0.016 + 0.01 * smoothstep(M.z_knee + 0.35, z_hem, z))
    orient_outward(m, lambda c_: np.array([0.0, 0.012, c_[2]]))
    for lp in boundary_loops(m.F):
        cz = m.arr()[lp].mean(0)
        add_band(m, lp, 0.013, "trim:leather_edge", lift=0.0011, rings=3)
        add_hem(m, lp, thick=0.006, width=0.018)
    pc = Piece(OUTFIT, "gambeson", m, {"cloth": "civ_quilt", "trim": "civ_trim"}, layer=50,
               hide=["torso", "upperarm_l", "upperarm_r", "forearm_l", "forearm_r"],
               desc="quilted linen gambeson, mid-thigh, long sleeves",
               notes="over the peasant shirt and trousers; belt, bracers, hood cowl and quiver strap go over it",
               grime=lambda P: np.stack([1 - 0.16 * smoothstep(M.z_knee + 0.35, M.z_knee + 0.2, P[:, 2])] * 3, 1))
    return pc, dict(G=G, P=m.arr(), F=[f for f, hh in zip(m.F, m.HF) if not hh], mesh=m,
                    h={k: float(x) for k, x in zip(G.idx, h)})


# ================================================================================================= hood + cowl
def head_hull_targets(A):
    """ray targets for the hood: the head / neck skin + the hair (when the base file has it)"""
    cb, rig = A.cb, A.rig
    ts = [skin_target(cb, 0.0)]
    hair = next((o for o in rig.children if o.type == 'MESH' and o.name.endswith("_hair")), None)
    if hair is not None:
        dg = bpy.context.evaluated_depsgraph_get()
        ev = hair.evaluated_get(dg)
        me = ev.to_mesh()
        HP = np.array([(hair.matrix_world @ v.co)[:] for v in me.vertices])
        HF = [tuple(p.vertices) for p in me.polygons]
        ev.to_mesh_clear()
        ts.append(Target(HP, HF, 0.0, "hair"))
    return ts


def hood(A, M, under_d, ncol=64):
    """wool hood with a shoulder cowl (chaperon): rings around a vertical axis from the peak down over the shoulders.
    Head rows follow the head + hair hull (+2 cm), a gentle peak at the back; below the chin the cloth may only pull
    in slowly (it hangs from the occiput / chin instead of hugging the neck); the cowl hangs over the shoulders and
    the garment under it (non-shrinking downward) to its hem (longer in front). Face opening = an ellipse (brow to
    under the chin, +-62 deg at the cheeks), snapped exactly; rolled hems + a cord on the opening. Head rows are rigid
    to the head (never follow the jaw / face), the cowl takes the weights of the garment under it."""
    cb = A.cb
    H = cb.H
    z_eye = H["eye_l"][2] if "eye_l" in H else M.z_top - 0.10
    z_chin = z_eye - 0.125
    z_top = M.z_top
    z_nb = M.z_neck - 0.035                        # neck base (cowl starts)
    z_open_top, z_open_bot = z_eye + 0.058, z_chin - 0.012
    hy = np.mean(cb.co[cb.co[:, 2] > z_eye][:, 1])
    heads = head_hull_targets(A)
    under = Target(under_d["P"], under_d["F"], 0.0, "under")
    comp_head = Composite(heads)
    comp_body = Composite(heads + [under])
    sh_z = M.z_sh

    def z_hem(ph):
        # short over the shoulder points (the arms rise freely under it), longer down the chest and the back
        return sh_z - 0.055 - 0.065 * math.cos(ph) ** 2 - 0.04 * (0.5 + 0.5 * math.cos(ph)) ** 2

    # rows: (kind, z or t) from the top
    rows = []
    zz = z_top + 0.012
    while zz > z_nb + 0.005:
        rows.append(("z", zz)); zz -= 0.018 if zz > z_chin else 0.016
    rows.append(("z", z_nb))
    ncow = 7
    for k in range(1, ncow + 1):
        rows.append(("t", k / ncow))
    # face opening = an ellipse (brow to under the chin, +-62 deg at the cheeks). The columns of every row span
    # phi in [alpha, 2 pi - alpha]: alpha = the opening half-angle on the rows it crosses (the first / last columns ARE
    # the opening's edges), a hair's width elsewhere, where a seam face closes the ring across the front.
    zc, hz = (z_open_top + z_open_bot) / 2, (z_open_top - z_open_bot) / 2
    a0 = math.radians(62)
    eps = math.pi / ncol

    def alpha(z):
        u = (z - zc) / hz
        return max(eps, a0 * math.sqrt(max(0.0, 1.0 - u * u)))

    def centre(z):
        t = smoothstep(z_nb - 0.02, z_chin + 0.02, z)
        return np.array([0.0, lerp(0.012, hy, t)])

    def row_z(r, ph):
        return r[1] if r[0] == "z" else lerp(z_nb, z_hem(ph), r[1])

    row_alpha = [alpha(r[1]) if r[0] == "z" else eps for r in rows]
    PH = np.array([[al + (2 * math.pi - 2 * al) * j / (ncol - 1) for j in range(ncol)] for al in row_alpha])
    is_open = [al > eps * 1.01 for al in row_alpha]
    R = np.zeros((len(rows), ncol)); Z = np.zeros((len(rows), ncol))
    for i, r in enumerate(rows):
        for j in range(ncol):
            ph = PH[i, j]
            z = row_z(r, ph); Z[i, j] = z
            cx, cy = centre(z)
            d = np.array([math.sin(ph), -math.cos(ph), 0.0])
            comp = comp_head if (r[0] == "z" and z > z_chin - 0.01) else comp_body
            hit = comp.ray(np.array([cx, cy, z]) + d * 0.6, -d, 0.6)
            R[i, j] = (0.6 - hit) if hit is not None else np.nan
        ok = ~np.isnan(R[i])
        if ok.sum() >= 3:
            R[i, ~ok] = np.interp(PH[i][~ok], PH[i][ok], R[i, ok])
        else:
            R[i] = 0.01
    R = np.array([gauss1d(R[i][None], 1.2, 1, wrap=not is_open[i])[0] for i in range(len(rows))])
    # where the garment under the cowl follows the arm (deltoid / shoulder blade: upperarm + clavicle weights) the
    # cowl stands further off: the shoulder rises under it when the arm lifts (drawing the bow)
    sw_u = SurfaceWeights(under_d["mesh"])
    shoulder = np.zeros_like(R)
    for i, r in enumerate(rows):
        if r[0] != "t":
            continue
        for j in range(ncol):
            ph = PH[i, j]
            z = Z[i, j]; cx, cy = centre(z)
            q = np.array([cx + R[i, j] * math.sin(ph), cy - R[i, j] * math.cos(ph), z])
            w = sw_u(q, head=False)
            shoulder[i, j] = sum(x for b, x in w.items() if b.startswith(("upperarm", "clavicle")))
    shoulder = np.array([gauss1d(shoulder[i][None], 1.5, 1, wrap=not is_open[i])[0] for i in range(len(rows))])
    # clearance per row kind, peak at the back of the head
    for i, r in enumerate(rows):
        for j in range(ncol):
            ph = PH[i, j]
            z = Z[i, j]
            back = (0.5 - 0.5 * math.cos(ph))                    # 0 front .. 1 back
            if r[0] == "z" and z > z_chin:
                side = abs(math.sin(ph))
                c = 0.024 + 0.010 * side + 0.022 * smoothstep(z_eye - 0.04, z_top, z) * back ** 2 \
                    + 0.008 * smoothstep(z_chin, z_eye, z) * back
            elif r[0] == "z":
                c = 0.022
            else:
                c = 0.019 + 0.008 * r[1] + 0.016 * min(1.0, shoulder[i, j] * 1.5) + 0.008 * back ** 2
            R[i, j] += c
    # hanging: below the chin the radius may only shrink slowly going down (per unit drop)
    for i in range(1, len(rows)):
        if Z[i, 0] < z_chin or rows[i][0] == "t":
            dz = np.maximum(Z[i - 1] - Z[i], 1e-4)
            R[i] = np.maximum(R[i], R[i - 1] - 0.30 * dz)
    R = gauss1d(R, 0.8, 0)
    # the rows under the opening inherit the cheek radius at their front columns: smooth them round again
    R = np.array([gauss1d(R[i][None], 2.0, 1, wrap=True)[0] if not is_open[i] else R[i] for i in range(len(rows))])
    m = CMesh()
    grid = []
    for i, r in enumerate(rows):
        row = []
        for j in range(ncol):
            ph = PH[i, j]
            z = Z[i, j]
            cx, cy = centre(z)
            peak = smoothstep(z_eye + 0.02, z_top + 0.01, z)
            cy2 = cy + 0.045 * peak ** 2
            rr = R[i, j]
            p = np.array([cx + rr * math.sin(ph), cy2 - rr * math.cos(ph), z + 0.012 * peak ** 3 * (0.5 - 0.5 * math.cos(ph))])
            row.append(m.add_v(p, None, "body"))
        grid.append(row)
    top = m.add_v(np.array([0.0, hy + 0.09, z_top + 0.035]), None, "body")      # the peak points back
    for i in range(len(rows) - 1):
        for j in range(ncol - 1):
            m.add_f([grid[i][j], grid[i][j + 1], grid[i + 1][j + 1], grid[i + 1][j]],
                    [(j, i), (j + 1, i), (j + 1, i + 1), (j, i + 1)], "cloth")
        if not is_open[i] and not is_open[i + 1]:            # the front seam closes the ring outside the opening
            m.add_f([grid[i][ncol - 1], grid[i][0], grid[i + 1][0], grid[i + 1][ncol - 1]],
                    [(ncol - 1, i), (ncol, i), (ncol, i + 1), (ncol - 1, i + 1)], "cloth")
    for j in range(ncol):
        j2 = (j + 1) % ncol
        m.add_f([top, grid[0][j2], grid[0][j]], [(j + 0.5, -1), (j + 1, 0), (j, 0)], "cloth")
    # rows go down x columns around +phi: wound inward seen from outside: flip all
    orient_outward(m, lambda c_: np.array([0.0, 0.0, c_[2]]) if c_[2] < z_nb else np.array([0.0, hy, c_[2]]))
    P = m.arr()
    # weights: head rows rigid to the head, neck blends into the garment under the cowl
    sw_under = SurfaceWeights(under_d["mesh"])
    for i, p in enumerate(P):
        wg = sw_under(p)
        ws = norm_w(head_only(cb.weights_at(p)))
        wneck = mix_w(ws, wg, smoothstep(z_nb + 0.015, z_nb - 0.03, p[2]))
        m.W[i] = mix_w(wneck, {"head": 1.0}, smoothstep(z_chin - 0.02, z_chin + 0.03, p[2]))
        m.G[i] = "civ_match_head" if p[2] > z_chin + 0.01 else ("civ_match_neck" if p[2] > z_nb - 0.01 else "body")
    # smooth only the head -> neck -> cowl transition: the cowl keeps the exact weights of the garment under it
    mask = np.array([z_nb - 0.04 < p[2] < z_chin + 0.04 for p in P])
    smooth_weights(m, iters=2, lam=0.4, mask=mask)
    # UVs: metric cylindrical (u around, v down)
    P = m.arr()
    for fi, f in enumerate(m.F):
        Q = P[list(f)]
        c = Q.mean(0)
        cx, cy = centre(c[2])
        angs = np.array([math.atan2(q[0] - cx, -(q[1] - cy)) for q in Q])
        a_ = angs[0]
        angs = a_ + (angs - a_ + math.pi) % (2 * math.pi) - math.pi
        m.UV[fi] = [(a * 0.13, q[2]) for a, q in zip(angs, Q)]
    for lp in boundary_loops(m.F):
        cz = m.arr()[lp].mean(0)
        if cz[2] > z_nb + 0.03:
            add_band(m, lp, 0.012, "trim:cord", lift=0.0012, rings=3)
            add_hem(m, lp, thick=0.0035, width=0.02)
        else:
            add_band(m, lp, 0.016, "trim:stitch", lift=0.0010, rings=3)
            add_hem(m, lp, thick=0.0035, width=0.02)
    pc = Piece(OUTFIT, "hood", m, {"cloth": "civ_wool_green", "trim": "civ_trim"}, layer=80,
               hide=["scalp", "neck"], hides_hair=True, desc="wool hood with shoulder cowl",
               notes="over the gambeson / tunic; the hair is dropped under it; head part rigid to the head",
               grime=lambda P_: np.stack([1 - 0.12 * smoothstep(sh_z - 0.05, sh_z - 0.16, P_[:, 2])] * 3, 1))
    return pc, dict(P=m.arr(), F=[f for f, hh in zip(m.F, m.HF) if not hh], mesh=m, z_nb=z_nb, z_chin=z_chin)


# ================================================================================================= bracers
def bracers(A, M, gam_d):
    """stiff leather bracers over the gambeson sleeves (wrist to below the elbow) with a laced seam on the inside of the
    forearm; on the gambeson's own lines (+4 mm), so they move exactly with the sleeve"""
    cb = A.cb
    t0 = {s: M.elbow_t(s) + 0.10 for s in SIDES}
    t1 = 0.95

    def keep(c, f):
        s = side_of(c)
        aw = sum(sum(x for b, x in cb.W[i].items() if b.startswith(("lowerarm", "upperarm"))) for i in f) / len(f)
        return aw > 0.5 and t0[s] - 0.02 < M.arm_t(c, s) < t1 + 0.02
    G = Garment(A, keep)
    for lp in G.loops():
        c = G.P[lp].mean(0); s = side_of(c)
        t = M.arm_t(c, s)
        O, Nn = arm_plane(M, s, t0[s] if t < (t0[s] + t1) / 2 else t1)
        G.snap_loop_plane(lp, O, Nn)
    T = G.P.copy()
    c = np.full(len(T), 0.02)
    P, h = drape_lines(A.cb, G.idx, G.F, T, c, unders=[(gam_d["h"], 0.0055)], iters=10, verbose="bracers", bridge=6)
    G.P = P
    m = CMesh()
    for p, w in zip(P, G.W):
        m.add_v(p, dict(w))
    for f in G.F:
        s = side_of(P[f[0]])
        m.add_f(f, A.axes.uv(P, f, "arm_" + s), "leather")
    # lacing strip along the inner forearm (toward the body in the rest pose): stations along the arm axis, the
    # bracer surface found by a ray from the axis outward
    bv = BVHTree.FromPolygons([v3(p) for p in m.arr()], [tuple(f) for f in m.F])
    N_ = vertex_normals(m.arr(), m.F)
    kd = KDTree(len(m.V))
    for i, p in enumerate(m.V):
        kd.insert(p, i)
    kd.balance()
    for s in SIDES:
        sg = 1 if s == "l" else -1
        a_, b_ = arm_plane(M, s, t0[s] + 0.025)[0], arm_plane(M, s, t1 - 0.025)[0]
        ax_ = nrm(b_ - a_)
        d = np.array([-sg * 1.0, -0.4, 0.0]); d = nrm(d - ax_ * np.dot(d, ax_))
        path, up = [], []
        for t in np.linspace(0, 1, 12):
            o = lerp(a_, b_, t)
            hit = bv.ray_cast(v3(o + d * 0.2), v3(-d), 0.2)
            if hit[0] is None:
                continue
            q = np.array(hit[0][:]); n_ = np.array(hit[1][:])
            if np.dot(n_, d) < 0:
                n_ = -n_
            path.append(q + n_ * 0.0018)
            up.append(nrm(np.cross(ax_, n_)))
        if len(path) >= 3:
            band_along(m, np.array(path), np.array(up), 0.022, 0.0012, "trim:lacing",
                       lambda p: dict(m.W[kd.find(p)[1]]), closed=False,
                       prof=[(0.0, -0.5), (1.0, -0.45), (1.0, 0.45), (0.0, 0.5)])
    for lp in boundary_loops([f for f, s_ in zip(m.F, m.S) if not s_.startswith("trim")]):
        add_hem(m, lp, thick=0.002, width=0.006)
    return Piece(OUTFIT, "bracers", m, {"leather": "civ_leather_dark", "trim": "civ_trim"}, layer=65, hide=[],
                 desc="leather bracers, laced", notes="over the gambeson sleeves (same lines)")


# ================================================================================================= gloves
def gloves(A, M):
    """fingerless leather gloves: the hand skin offset 1.4 mm (fingers and thumb cut mid proximal phalanx, openings
    snapped to planes across each finger), cuff to 6 cm up the wrist under the gambeson sleeve; skin weights"""
    cb = A.cb
    H = cb.H
    fingers = ("index", "middle", "ring", "pinky")

    def cut_plane(s, f):
        if f == "thumb":
            a, b = H["thumb_02_" + s], H["thumb_03_" + s]; t = 0.45
        else:
            a, b = H[f + "_01_" + s], H[f + "_02_" + s]; t = 0.42
        return a + (b - a) * t, nrm(b - a)

    def chain(i):
        b = cb.dom[i]
        for f in fingers + ("thumb",):
            if b.startswith(f):
                return f
        return None
    faces = []
    for f in cb.faces:
        c = cb.co[list(f)].mean(0)
        s = "l" if c[0] > 0 else "r"
        doms = [cb.dom[i] for i in f]
        if not all(d.startswith(("hand", "thumb", "index", "middle", "ring", "pinky", "lowerarm")) for d in doms):
            continue
        if any(d.startswith("lowerarm") for d in doms):
            if M.arm_t(c, s) < 0.93:
                continue
        ch = [chain(i) for i in f]
        ok = True
        for fg in set(x for x in ch if x):
            if fg in fingers and all(cb.dom[i].startswith(fg + "_metacarpal") for i in f if chain(i) == fg):
                continue
            O, Nn = cut_plane(s, fg)
            if np.dot(c - O, Nn) > -0.002:
                ok = False
        if ok:
            faces.append(f)
    used = sorted(set(i for f in faces for i in f)); remap = {v: k for k, v in enumerate(used)}
    F = [tuple(remap[i] for i in f) for f in faces]
    P = cb.co[used].copy()
    N = cb.vn[used]
    W = [norm_w(dict(cb.W[i])) for i in used]
    G_ = type("G", (), {})()
    G_.P, G_.F = P, F
    G_.snap_loop_plane = lambda lp, O, Nn: Garment.snap_loop_plane(G_, lp, O, Nn, rings=1)
    for lp in boundary_loops(F):
        c = P[lp].mean(0); s = "l" if c[0] > 0 else "r"
        ch = max(fingers + ("thumb", "wrist"), key=lambda fg: -np.linalg.norm(c - (cut_plane(s, fg)[0] if fg != "wrist" else H["hand_" + s])))
        if ch == "wrist" or len(lp) > 30:
            O, Nn = arm_plane(M, s, 0.93)
        else:
            O, Nn = cut_plane(s, ch)
        Garment.snap_loop_plane(G_, lp, O, Nn, rings=1)
    P = G_.P
    off = np.full(len(P), 0.0014)
    # a touch more over the knuckles / back of the hand (leather over the tendons), cuff flares a little
    for k, i in enumerate(used):
        c = P[k]; s = "l" if c[0] > 0 else "r"
        t = M.arm_t(c, s)
        off[k] += 0.0025 * smoothstep(0.97, 0.93, t)
    P = P + N * off[:, None]
    m = CMesh()
    for p, w in zip(P, W):
        m.add_v(p, w)
    for f in F:
        s = "l" if P[f[0]][0] > 0 else "r"
        m.add_f(f, A.axes.uv(P, f, "hand_" + s), "leather")
    for lp in boundary_loops(m.F):
        big = len(lp) > 30                                  # the wrist cuff (under the sleeve); fingers: a thin edge
        add_hem(m, lp, thick=0.0012 if big else 0.0009, width=0.004 if big else 0.0012)
    pc = Piece(OUTFIT, "gloves", m, {"leather": "civ_leather_tan"}, layer=45, hide=["hand_l", "hand_r"],
               desc="fingerless leather gloves", notes="on the skin; the gambeson sleeve ends over the cuff")
    return pc, dict(P=P, F=F, mesh=m)


# ================================================================================================= boots
def boots(A, M, legs_d):
    """tall soft leather boots to below the knee over the trousers / leg wraps, flat stitched sole, a turned-down cuff
    (a wide leather band over the top edge)"""
    cb = A.cb
    z_top = M.z_knee - 0.095

    def keep(c, f):
        return c[2] < z_top + 0.012
    G = Garment(A, keep)
    for lp in G.loops():
        G.snap_loop_plane(lp, np.array([0, 0, z_top]), np.array([0, 0, 1.0]))
    T = G.P.copy()
    c = np.array([0.0055 + 0.004 * smoothstep(M.z_ankle + 0.02, M.z_ankle + 0.12, p[2]) for p in T])
    P, h = drape_lines(cb, G.idx, G.F, T, c, unders=[(legs_d["h"], 0.005)], iters=14, lam=0.4, verbose="boots",
                       bridge=4)
    fn = fold_noise(P, 91, 5.0)
    for i, p in enumerate(P):
        # soft ankle creases
        a = smoothstep(M.z_ankle + 0.02, M.z_ankle + 0.07, p[2]) * smoothstep(M.z_ankle + 0.16, M.z_ankle + 0.08, p[2])
        h[i] += 0.0022 * a * (0.5 + 0.5 * math.sin(p[2] / 0.018 * 2 * math.pi + 2 * fn[i]))
    P = T + cb.line_N(G.idx) * h[:, None]
    low = P[:, 2] < 0.016
    P[low, 2] = np.maximum(0.0008, P[low, 2] - 0.005)
    P[:, 2] = np.maximum(P[:, 2], 0.0008)
    G.P = P
    m = CMesh()
    for p, w in zip(P, G.W):
        m.add_v(p, dict(w))
    N = vertex_normals(P, G.F)
    for f in G.F:
        s = side_of(P[f[0]])
        sole = np.mean([N[i][2] for i in f]) < -0.6 and P[list(f)][:, 2].max() < 0.02
        reg = "foot_" + s if P[list(f)][:, 2].mean() < M.z_ankle + 0.03 else "leg_" + s
        m.add_f(f, A.axes.uv(P, f, reg), "sole" if sole else "leather")
    for lp in boundary_loops(m.F):
        add_band(m, lp, 0.055, "leather", lift=0.0025, rings=4)
        add_hem(m, lp, thick=0.0018, width=0.008)
    return Piece(OUTFIT, "boots", m, {"leather": "civ_leather_dark", "sole": "civ_leather_tan"}, layer=40,
                 hide=["foot_l", "foot_r", "calf_l", "calf_r"], desc="tall leather boots with turned cuffs",
                 grime=lambda P_: np.stack([1 - 0.25 * smoothstep(0.06, 0.0, P_[:, 2])] * 3, 1)), \
        dict(P=m.arr(), F=[f for f, hh in zip(m.F, m.HF) if not hh], mesh=m, h={k: float(x) for k, x in zip(G.idx, h)})


# ================================================================================================= quiver
QUIVER_ARROWS = 11


def quiver(A, M, gam_d, hood_d):
    """back quiver: a leather tube (stitched seam, rolled rim, wooden base) hanging diagonally from the left hip to
    behind the right shoulder, 11 arrows (ash shafts, nocks, three fletching cards each) standing out of it, on a
    baldric that runs from the quiver top over the right shoulder (over the hood cowl), across the chest to the left
    hip and round to the quiver base. Quiver + arrows rigid to spine_05 (MakeClothes RIGID triangle on the upper
    back); the baldric takes the weights of the surface under it."""
    cb = A.cb
    H = cb.H
    sp5 = H["spine_05"]
    under = Composite([Target(gam_d["P"], gam_d["F"], 0.0), Target(hood_d["P"], hood_d["F"], 0.0)])
    # tube axis: bottom behind the left hip, top behind the right shoulder blade
    rad = 0.046
    bot0 = np.array([0.10, 0.0, M.z_belt + 0.075])                  # the base clears the belt
    top0 = np.array([-0.13, 0.0, sp5[2] + 0.10])
    ax = nrm(top0 - bot0)
    L = np.linalg.norm(top0 - bot0)
    # push the tube back until it clears the gambeson + cowl by 6 mm along its whole length (rests on the back)
    ymin = 0.0
    for t in np.linspace(0, 1, 24):
        p = lerp(bot0, top0, t)
        for dx in np.linspace(-rad, rad, 7):
            q = p + np.array([dx, 0, 0]) * 1.0
            hit = under.ray(np.array([q[0], 0.8, q[2]]), np.array([0, -1.0, 0]), 1.0)
            if hit is not None:
                ymin = max(ymin, 0.8 - hit + math.sqrt(max(rad ** 2 - dx ** 2, 0)) + 0.006)
    bot = bot0 + np.array([0, ymin, 0]); top = top0 + np.array([0, ymin, 0])
    # a slight lean back of the top (the strap pulls the mouth out): keeps the arrows clear of the head / hood
    top = top + np.array([0, 0.025, 0])
    ax = nrm(top - bot); L = np.linalg.norm(top - bot)
    wq = {"spine_05": 0.8, "spine_04": 0.2}
    m = CMesh()
    g = "civ_rigid_back"
    ref = np.array([0, 1.0, 0])
    prof = [(rad * 0.8, -0.012), (rad * 1.02, -0.004), (rad, 0.02)] + \
           [(rad * (1.0 + 0.04 * math.sin(k * 1.3)), 0.02 + (L - 0.05) * k / 8) for k in range(1, 9)] + \
           [(rad * 1.08, L - 0.02), (rad * 1.1, L), (rad * 0.94, L + 0.004), (rad * 0.9, L - 0.03)]
    lathe(m, bot, ax, ref, prof[:2], 24, "wood", w=wq, g=g, close_bottom=True, r_uv=rad)
    lathe(m, bot, ax, ref, prof[1:], 24, "leather", w=wq, g=g, r_uv=rad)
    # arrows: shafts from inside the tube to 0.30-0.36 m above the mouth, nock + 3 fletching cards each
    rng = np.random.default_rng(12)
    X = nrm(np.cross(ax, ref)); Y = np.cross(X, ax)
    mouth = top
    for k in range(QUIVER_ARROWS):
        a = 2 * math.pi * k / QUIVER_ARROWS + rng.uniform(-0.2, 0.2)
        rr = rad * 0.55 * math.sqrt(rng.uniform(0.2, 1.0))
        tilt = X * rng.uniform(-0.05, 0.05) + Y * rng.uniform(-0.03, 0.06)
        d = nrm(ax + tilt)
        base = bot + ax * 0.06 + (X * math.cos(a) + Y * math.sin(a)) * rr
        tip = mouth + (X * math.cos(a) + Y * math.sin(a)) * rr + d * rng.uniform(0.28, 0.34)
        tube_path(m, np.array([base, tip]), 0.0043, 6, "wood", lambda p: dict(wq), cap=True)
        tube_path(m, np.array([tip, tip + d * 0.012]), [0.0048, 0.0040], 6, "trim:fletch_wrap", lambda p: dict(wq),
                  cap=True)
        for q in range(3):
            ang = a + q * 2 * math.pi / 3
            vdir = X * math.cos(ang) + Y * math.sin(ang)
            vdir = nrm(vdir - d * np.dot(vdir, d))
            f0 = tip - d * 0.022; f1 = tip - d * 0.13
            vs = [m.add_v(f0, dict(wq), g), m.add_v(f1, dict(wq), g),
                  m.add_v(f1 + vdir * 0.016, dict(wq), g), m.add_v(f0 + vdir * 0.013, dict(wq), g)]
            u0 = q / 3.0                                   # card 2 = the dyed cock feather
            m.add_f(vs, [(u0 + 0.027, 0.0), (u0 + 0.027, 1.0), (u0 + 0.30, 1.0), (u0 + 0.30, 0.0)], "atlas:fletch")
    # baldric: quiver top -> over the right shoulder -> across the chest -> left hip -> round the left side -> base
    strap = CMesh()
    cen = (0.0, 0.012)

    def surf(ph, z, off=0.0055):
        """strap point: the largest radius of the layers under it over a +-2.5 cm / +-5 deg patch (a stiff strap
        bridges the cowl hem step and the folds instead of dipping into them)"""
        best = 0.0
        for dz in (-0.025, -0.012, 0.0, 0.012, 0.025):
            for dp in (-0.09, 0.0, 0.09):
                d = np.array([math.sin(ph + dp), -math.cos(ph + dp), 0.0])
                hit = under.ray(np.array([cen[0], cen[1], z + dz]) + d * 0.8, -d, 0.8)
                if hit is not None:
                    best = max(best, 0.8 - hit)
        d = np.array([math.sin(ph), -math.cos(ph), 0.0])
        return np.array([cen[0], cen[1], z]) + d * ((best or 0.2) + off)
    x_sh = H["upperarm_r"][0] * 0.62                     # the strap crosses the right shoulder here
    pts = []
    # front diagonal: from the right shoulder top down to the left hip (phi from -0.55 to +1.4 rad)
    for t in np.linspace(0, 1, 36):
        ph = lerp(-0.62, 1.35, t)
        z = lerp(M.z_sh + 0.01, M.z_belt + 0.02, t)
        pts.append(surf(ph, z))
    front = np.array(pts)
    # over the shoulder top (from the front shoulder point to the back, via the top of the cowl / gambeson)
    over = []
    for t in np.linspace(0, 1, 7)[1:-1]:
        ph = lerp(-0.62, -2.35, t)
        zz = M.z_sh + 0.01 + 0.035 * math.sin(math.pi * t)
        o = np.array([x_sh, lerp(-0.06, 0.10, t), zz + 0.2])
        zs_ = []
        for dx in (-0.02, 0.0, 0.02):
            hit = under.ray(o + np.array([dx, 0, 0]), np.array([0, 0, -1.0]), 0.5)
            if hit is not None:
                zs_.append(o[2] - hit)
        z_s = max(zs_) if zs_ else zz
        over.append(np.array([o[0], o[1], z_s + 0.0055]))
    back_top = top + ax * -0.02 - Y * 0.0 + np.array([0, -0.0, 0])
    back = [back_top]
    path_up = [np.array(p) for p in front[::-1]] + [np.array(p) for p in over[::-1]] + back
    path_up = np.array(path_up)
    # smooth the path (Laplacian, ends fixed) and re-seat on the surface
    for _ in range(4):
        q = path_up.copy()
        q[1:-1] = 0.5 * path_up[1:-1] + 0.25 * (path_up[:-2] + path_up[2:])
        path_up = q
    sw_g = SurfaceWeights(gam_d["mesh"]); sw_h = SurfaceWeights(hood_d["mesh"])

    def wat(p):
        # the surface right under the strap: the cowl over the shoulder, else the gambeson
        dg = sw_g.bvh.find_nearest(v3(p))[3]; dh = sw_h.bvh.find_nearest(v3(p))[3]
        return sw_h(p) if (dh is not None and (dg is None or dh < dg)) else sw_g(p)
    ups = []
    for k in range(len(path_up)):
        p = path_up[k]
        nn_ = nrm(p - np.array([0.0, 0.0, p[2]]) + np.array([0, 0, 0.3 if p[2] > M.z_sh - 0.02 else 0.0]))
        tng = nrm(path_up[min(k + 1, len(path_up) - 1)] - path_up[max(k - 1, 0)])
        ups.append(nrm(np.cross(tng, nn_)))
    band_along(strap, path_up, np.array(ups), 0.040, 0.004, "strap", wat, closed=False)
    # lower strap: left hip -> round the left side -> quiver base
    low = []
    for t in np.linspace(0, 1, 14):
        ph = lerp(1.35, 2.35, t)
        z = lerp(M.z_belt + 0.02, bot[2] + 0.03, t)
        low.append(surf(ph, z, 0.009))
    low.append(bot + ax * 0.03 + Y * 0.0)
    low = np.array(low)
    upl = np.array([nrm(np.cross(nrm(low[min(k + 1, len(low) - 1)] - low[max(k - 1, 0)]),
                                 nrm(low[k] - np.array([0, 0.012, low[k][2]])))) for k in range(len(low))])
    band_along(strap, low, upl, 0.030, 0.0035, "strap", wat, closed=False)
    m.merge(strap)
    return Piece(OUTFIT, "quiver", m, {"leather": "civ_leather_dark", "wood": "civ_wood", "strap": "civ_leather_dark",
                                       "trim": "civ_trim", "atlas": "civ_fletch"}, layer=90, hide=[],
                 desc="back quiver with %d arrows on a baldric" % QUIVER_ARROWS,
                 notes="quiver rigid to spine_05 (MakeClothes rigid triangle), baldric over the hood cowl",
                 extra={"quiver_axis": [bot.tolist(), top.tolist()], "arrows": QUIVER_ARROWS})


# ================================================================================================= build
LAYERS = [("gambeson", [("skin", 0.004), ("p_shirt", 0.006), ("p_trousers", 0.005)]),
          ("belt", [("gambeson", 0.0015), ("p_trousers", 0.003), ("p_shirt", 0.003)]),
          ("hood", [("skin", 0.006), ("gambeson", 0.009), ("belt", 0.004)]),
          ("bracers", [("gambeson", 0.0025), ("skin", 0.004)]),
          ("gloves", [("skin", 0.0009)]),
          ("boots", [("skin", 0.003), ("p_trousers", 0.003)]),
          ]                                   # the quiver / baldric are placed clear by construction (no per-vertex push)


def build(A):
    M = Measures(A.cb)
    A.M = M
    t0 = time.time()
    # the peasant's shirt and trousers (not written again: they are the peasant's assets) as the under-layers
    sh, si = OP.shirt(A, M)
    tr, ti = OP.trousers(A, M, si)
    under = {"p_shirt": sh, "p_trousers": tr}
    out, info = {}, {}
    ga, info["gambeson"] = gambeson(A, M, si, ti); out["gambeson"] = ga
    out["belt"] = OP.belt(A, M, info["gambeson"], z=M.z_belt - 0.005, outfit=OUTFIT, pouch_side="r", knife=True)
    ho, info["hood"] = hood(A, M, info["gambeson"]); out["hood"] = ho
    out["quiver"] = quiver(A, M, info["gambeson"], info["hood"])
    out["bracers"] = bracers(A, M, info["gambeson"])
    gl, info["gloves"] = gloves(A, M); out["gloves"] = gl
    bo, info["boots"] = boots(A, M, ti); out["boots"] = bo
    OP.layer_pass(A, {**under, **out}, layers=LAYERS)
    clog("archer built in %.1fs: %s" % (time.time() - t0, {k: v.m.tris() for k, v in out.items()}))
    return out, info, M, under


def hides(A, pieces, info, M):
    cb = A.cb
    res = {}
    for name, pc in pieces.items():
        P = pc.m.arr(); F = pc.m.outer_faces()
        if name == "gambeson":
            res[name] = covered_verts(cb, P, F, 0.07, erode=1, zmin=M.z_crotch + 0.02)
        elif name == "hood":
            res[name] = covered_verts(cb, P, F, 0.08, erode=1, zmin=M.z_neck - 0.06)
        elif name == "gloves":
            res[name] = covered_verts(cb, P, F, 0.012, erode=1, tight=0.0065, open_dist=0.0025)
        elif name == "boots":
            top = P[:, 2].max()
            feet = [i for i in range(NB) if cb.dom[i].startswith(("foot", "ball", "calf")) and cb.co[i][2] < top - 0.03]
            res[name] = sorted(set(covered_verts(cb, P, F, 0.05, erode=1, tight=0.02, open_dist=0.02)) | set(feet))
        else:
            res[name] = []
    return res


def author(A, only=None):
    pieces, info, M, under = build(A)
    hid = hides(A, pieces, info, M)
    paths = {}
    for name in ORDER:
        if only and name not in only:
            continue
        paths[name] = write_piece(A, pieces[name], delete=hid[name])
    if not only or "bow" in only:
        import outfit_civ_bow as BOW
        BOW.write_prop()
    return paths


def preview(A, tag="prev_archer"):
    pieces, info, M, under = build(A)
    objs = preview_objects({**{k: v.m for k, v in under.items()}, **{k: v.m for k, v in pieces.items()}}, A.rig)
    hair = next((o for o in A.rig.children if o.name.endswith("_hair")), None)
    if hair is not None:
        hair.hide_render = True
    OP.render_clay(A.rig, tag + ("" if A.kind == "male" else "_" + A.kind))


def check(A, poses=None):
    import outfit_civ_qa as QA
    import outfit_civ_poses as PZ
    pieces, info, M, under = build(A)
    hid = hides(A, pieces, info, M)
    objs = preview_objects({**{k: v.m for k, v in under.items()}, **{k: v.m for k, v in pieces.items()}}, A.rig)
    union = sorted(set(i for v in hid.values() for i in v))
    body = QA.masked_body(A.bm, union)
    A.bm.hide_render = True
    res = QA.penetration(A.rig, objs, body, poses=poses or (PZ.STRESS + PZ.EXTRA), tag="archer", layers={**{k: v.layer for k, v in pieces.items()}, **{k: v.layer for k, v in under.items()}})
    os.makedirs(RENC, exist_ok=True)
    json.dump(res, open(os.path.join(RENC, "check_archer_%s.json" % A.kind), "w"), indent=1)
    return res


if __name__ == "__main__":
    args = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    mode = args[0] if args else "preview"
    kind = next((a.split("=")[1] for a in args if a.startswith("kind=")), None)
    rest = [a for a in args[1:] if not a.startswith("kind=")]
    t0 = time.time()
    if mode == "bow":
        import outfit_civ_bow as BOW
        BOW.write_prop()
    else:
        kind = kind or next((o.name[4:] for o in bpy.data.objects if o.type == 'ARMATURE' and o.name.startswith("rts_")), "male")
        A = Author(kind)
        if mode == "preview":
            preview(A)
        elif mode == "check":
            check(A, poses=rest or None)
        elif mode == "author":
            author(A, only=rest or None)
    clog("archer %s done in %.1fs" % (mode, time.time() - t0))
