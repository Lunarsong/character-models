"""PEASANT outfit (RTS civilian / worker unit): linen shirt, wool tunic (knee length, slit front + back, tablet-woven
trim), wool trousers with linen leg wraps, leather turnshoes, belt with iron buckle + pouch + knife, straw hat.
Every piece is an MPFB clothes asset rts_peasant_<slot> authored on the live male base human (MakeClothes), so it
fits the female / customised bodies, is skinned to the shared rts_human skeleton, hides the skin it covers and follows
the cust_* body morphs. Shared garments: the archer wears this shirt and these trousers (mix-and-match).

run:  BLENDER_USER_RESOURCES=$PWD/blender_profile $BL -b out/base_male.blend --python-exit-code 1 \
        -P scripts/outfit_peasant.py -- <mode> [pieces ...]
modes: preview   build the pieces and render clay previews (renders/civ/prev_peasant_*.png), nothing written
       author    build + write the MPFB clothes assets (assets/mpfb_assets/clothes/rts_peasant_*, installed into MPFB)
Dressing / export / QA: scripts/outfit_civ.py (see scripts/outfit_civ.sh).
"""
import sys, os, math, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from outfit_civ_geo import *

OUTFIT = "peasant"
ORDER = ["shirt", "trousers", "shoes", "tunic", "belt", "hat"]


# ================================================================================================= shirt
def line_frame(M, p, s, kind="arm"):
    """(origin, axis, X, Y) of a limb for angular fold patterns"""
    if kind == "arm":
        O, A_ = arm_plane(M, s, 0.0); ref = np.array([0, 0, 1.0])
    else:
        O = M.H["thigh_" + s]; A_ = nrm(M.H["foot_" + s] - O); ref = np.array([0, -1.0, 0])
    X = nrm(ref - A_ * np.dot(ref, A_)); Y = np.cross(A_, X)
    return O, A_, X, Y


def ang(p, fr):
    O, A_, X, Y = fr
    return math.atan2(np.dot(p - O, Y), np.dot(p - O, X))


def shirt(A, M):
    """linen shirt: round neck at the base of the neck, long loose sleeves gathered at the wrist cuffs, hem at the
    hips (tucked into the trousers)"""
    cb = A.cb
    z_hem = M.z_belt - 0.045                  # tucked 7-8 cm into the trousers (clear of the hip crease)
    t_cuff = 0.965

    def keep(c, f):
        if c[2] < z_hem - 0.012:
            return False
        hw = sum(sum(x for b, x in cb.W[i].items() if b.startswith(("hand", "thumb", "index", "middle", "ring", "pinky")))
                 for i in f) / len(f)
        if hw > 0.3:
            return False
        s = side_of(c)
        aw = sum(sum(x for b, x in cb.W[i].items() if b.startswith(("upperarm", "lowerarm"))) for i in f) / len(f)
        if aw > 0.5 and M.arm_t(c, s) > t_cuff + 0.02:
            return False
        if abs(c[0]) < 0.15 and c[2] > neckline_z(M, phi_of(c), front=0.028, back=0.004) + 0.008:
            return False
        return True
    G = Garment(A, keep)
    L = classify_loops(G, M)
    for s in SIDES:
        O, Nn = arm_plane(M, s, t_cuff)
        G.snap_loop_plane(L["cuff_" + s], O, Nn)
    G.snap_loop_plane(L["waist"], np.array([0, 0, z_hem]), np.array([0, 0, 1.0]))
    G.snap_loop_zcurve(L["neck"], lambda p: neckline_z(M, phi_of(p), front=0.028, back=0.004))
    T = G.P.copy()
    # clearance from the skin: 7.5 mm, fuller at the belly / lower back, sleeves billow on the forearm and gather
    # into the cuffs
    c = np.zeros(len(T))
    for i, p in enumerate(T):
        r = G.region[i]
        if r.startswith(("arm", "hand")):
            s = r[-1]; t = M.arm_t(p, s); te = M.elbow_t(s)
            billow = 0.012 * smoothstep(te + 0.04, te + 0.22, t) * (1 - smoothstep(0.86, t_cuff, t))
            c[i] = 0.0085 + billow + 0.003 * smoothstep(0.35, 0.0, t)
        else:
            c[i] = 0.0075 + 0.005 * smoothstep(M.z_belt + 0.14, M.z_belt + 0.02, p[2]) * (1 if p[1] < 0.03 else 0.7)
    P, h = drape_lines(cb, G.idx, G.F, T, c, iters=30, verbose="shirt", bridge=10)
    # folds along the lines: rings on the billowing forearm (diagonal, twisting) + soft vertical drape under the chest
    fn = fold_noise(P, 11, 5.0)
    for i, p in enumerate(P):
        r = G.region[i]
        if r.startswith(("arm", "hand")):
            s = r[-1]; t = M.arm_t(p, s); te = M.elbow_t(s)
            th = ang(p, line_frame(M, p, s))
            L_ = np.linalg.norm(M.H["hand_" + s] - M.H["upperarm_" + s])
            amp = 0.0045 * smoothstep(te - 0.02, te + 0.15, t) * (1 - smoothstep(0.88, t_cuff, t))
            amp += 0.0015 * smoothstep(te - 0.12, te, t) * smoothstep(te + 0.1, te, t)
            h[i] += amp * (0.6 + 0.4 * math.sin(t * L_ / 0.042 * 2 * math.pi + 1.6 * th + 2.0 * fn[i]))
        else:
            below = smoothstep(M.z_sh - 0.12, M.z_belt + 0.1, p[2]) * smoothstep(z_hem, z_hem + 0.08, p[2])
            h[i] += 0.0022 * below * (0.5 + 0.5 * math.sin(phi_of(p) * 11 + 2.5 * fn[i]))
    P = T + cb.line_N(G.idx) * h[:, None]
    G.P = P
    m = G.to_mesh("cloth", axes=A.axes)
    for key, (th_, wd) in (("cuff_l", (0.0022, 0.02)), ("cuff_r", (0.0022, 0.02)), ("waist", (0.0012, 0.004)),
                           ("neck", (0.0022, 0.012))):
        add_hem(m, L[key], thick=th_, width=wd)
    pc = Piece(OUTFIT, "shirt", m, {"cloth": "civ_linen"}, layer=20, hide=["torso", "upperarm_l", "upperarm_r",
                                                                        "forearm_l", "forearm_r"],
               desc="linen shirt, long sleeves, tucked in", notes="inner layer; tunic / gambeson go over it")
    return pc, dict(G=G, loops=L, P=P, F=G.F, h={k: float(x) for k, x in zip(G.idx, h)}, mesh=m)


# ================================================================================================= trousers
def trousers(A, M, under):
    """wool trousers (braies) from the waist to below the knee, loose at the seat and thighs, and linen leg wraps
    (winingas) spiralled from the ankle to below the knee over the trouser legs: the wraps are their own layer on the
    same normal lines (they cover the trouser ends), with a slanted top edge and a hem at both ends"""
    cb = A.cb
    z_top = M.z_belt + 0.03
    z_wrap0, z_wrap1 = M.z_ankle + 0.012, M.z_knee - 0.06
    z_low = z_wrap1 - 0.035                                    # the trouser legs end inside the wraps

    def keep(c, f):
        if c[2] > z_top + 0.012 or c[2] < z_low - 0.012:
            return False
        fw = sum(sum(x for b, x in cb.W[i].items() if b.startswith(("foot", "ball"))) for i in f) / len(f)
        aw = sum(sum(x for b, x in cb.W[i].items() if b.startswith(("upperarm", "lowerarm", "hand"))) for i in f) / len(f)
        return fw < 0.45 and aw < 0.3
    G = Garment(A, keep)
    for lp in G.loops():
        c = G.P[lp].mean(0)
        zz = z_top if c[2] > M.z_knee else z_low
        G.snap_loop_plane(lp, np.array([0, 0, zz]), np.array([0, 0, 1.0]))
    T = G.P.copy()
    c = np.zeros(len(T))
    for i, p in enumerate(T):
        z = p[2]
        seat = smoothstep(M.z_hip + 0.08, M.z_hip - 0.05, z) * smoothstep(M.z_knee + 0.02, M.z_knee + 0.2, z)
        c[i] = 0.0095 + 0.007 * seat - 0.002 * smoothstep(M.z_knee + 0.05, z_low, z)     # baggy seat and thighs
    unders = [(under["h"], 0.0065)] if under is not None else []
    P, h = drape_lines(cb, G.idx, G.F, T, c, unders=unders, iters=30, verbose="trousers", bridge=8)
    fn = fold_noise(P, 21, 4.0)
    for i, p in enumerate(P):
        z = p[2]; s = side_of(p)
        th = ang(p, line_frame(M, p, s, "leg"))
        if z < M.z_hip:
            knee = smoothstep(M.z_knee + 0.16, M.z_knee + 0.03, z)
            thigh = smoothstep(M.z_knee + 0.05, M.z_knee + 0.2, z) * smoothstep(M.z_hip, M.z_hip - 0.12, z)
            h[i] += 0.0035 * knee * (0.5 + 0.5 * math.sin(z / 0.03 * 2 * math.pi + 1.2 * th + 2 * fn[i])) \
                + 0.0028 * thigh * (0.5 + 0.5 * math.sin(th * 5 + 3 * fn[i]))
    P = T + cb.line_N(G.idx) * h[:, None]
    G.P = P
    m = CMesh()
    for p, w in zip(P, G.W):
        m.add_v(p, dict(w))
    for f in G.F:
        regs = [G.region[i] for i in f]
        r = max(set(regs), key=regs.count)
        r = r if r.startswith("leg") else "torso"
        m.add_f(f, A.axes.uv(P, f, r), "cloth")
    for lp in boundary_loops(m.F):
        add_hem(m, lp, thick=0.0015, width=0.008)
    hl = {k: float(x) for k, x in zip(G.idx, h)}
    # --- leg wraps
    def keep_w(c, f):
        if c[2] > z_wrap1 + 0.03 or c[2] < z_wrap0 - 0.012:
            return False
        fw = sum(sum(x for b, x in cb.W[i].items() if b.startswith(("foot", "ball"))) for i in f) / len(f)
        return fw < 0.45
    W_ = Garment(A, keep_w)
    for lp in W_.loops():
        cz = W_.P[lp].mean(0)
        sg = 1.0 if cz[0] > 0 else -1.0
        if cz[2] > M.z_knee - 0.2:
            O, Nn = leg_plane(M, "l" if sg > 0 else "r", M.leg_t(np.array([cz[0], cz[1], z_wrap1]), "l" if sg > 0 else "r"))
            W_.snap_loop_plane(lp, np.array([O[0], O[1], z_wrap1]), nrm(np.array([-sg * 0.16, 0.05, 1.0])))
        else:
            W_.snap_loop_plane(lp, np.array([0, 0, z_wrap0]), np.array([0, 0, 1.0]))
    TW = W_.P.copy()
    cw = np.full(len(TW), 0.0082)
    Pw, hw = drape_lines(cb, W_.idx, W_.F, TW, cw, unders=[(hl, 0.0021)], iters=20, verbose="wraps", bridge=4)
    for i, p in enumerate(Pw):
        s = side_of(p)
        th = ang(p, line_frame(M, p, s, "leg"))
        u = (th / (2 * math.pi)) % 1.0
        hw[i] += 0.0016 * (((p[2] - z_wrap0) / 0.045 - u) % 1.0)          # overlapping turns: saw-tooth ridge
    Pw = TW + cb.line_N(W_.idx) * hw[:, None]
    wm = CMesh()
    for p, w in zip(Pw, W_.W):
        wm.add_v(p, dict(w))
    for f in W_.F:
        cz = Pw[list(f)].mean(0)
        fr = line_frame(M, cz, side_of(cz), "leg")
        ths = np.array([ang(Pw[i], fr) for i in f])
        ths = ths[0] + (ths - ths[0] + math.pi) % (2 * math.pi) - math.pi
        wm.add_f(f, [(t / (2 * math.pi), (Pw[i][2] - z_wrap0) / 0.18) for t, i in zip(ths, f)], "atlas:wrap")
    for lp in boundary_loops(wm.F):
        add_hem(wm, lp, thick=0.0022, width=0.007)
    m.merge(wm)
    for k, x in zip(W_.idx, hw):
        hl[k] = max(hl.get(k, -1.0), float(x))
    pc = Piece(OUTFIT, "trousers", m, {"cloth": "civ_wool_brown", "atlas": "civ_legwrap"}, layer=30,
               hide=["hips", "thigh_l", "thigh_r", "calf_l", "calf_r"],
               desc="wool trousers with linen leg wraps", notes="over the shirt tail; shoes / boots go over the ankles",
               grime=lambda P: np.stack([1 - 0.28 * smoothstep(0.35, 0.05, P[:, 2])] * 3, 1) * np.array([1.0, 0.97, 0.93]) ** smoothstep(0.35, 0.05, P[:, 2])[:, None])
    return pc, dict(G=G, P=m.arr(), F=[f for f, hh in zip(m.F, m.HF) if not hh], mesh=m, h=hl)


# ================================================================================================= shoes
def shoes(A, M, legs=None):
    """leather turnshoes: soft uppers to just above the ankle bone with a loose collar (the trouser hems sit inside),
    flat sole on the floor"""
    cb = A.cb
    z_top = M.z_ankle + 0.034

    def keep(c, f):
        return c[2] < z_top + 0.01
    G = Garment(A, keep)
    for lp in G.loops():
        G.snap_loop_plane(lp, np.array([0, 0, z_top]), np.array([0, 0, 1.0]))
    T = G.P.copy()
    c = np.array([0.0048 + 0.006 * smoothstep(M.z_ankle - 0.02, z_top, p[2]) for p in T])
    unders = [(legs["h"], 0.0055)] if legs is not None else []
    P, h = drape_lines(cb, G.idx, G.F, T, c, unders=unders, iters=12, lam=0.4, verbose="shoes")
    # sole flat on the floor, 5 mm under the skin's sole
    low = P[:, 2] < 0.014
    P[low, 2] = np.maximum(0.0006, P[low, 2] - 0.004)
    P[:, 2] = np.maximum(P[:, 2], 0.0006)
    G.P = P
    m = CMesh()
    for p, w in zip(P, G.W):
        m.add_v(p, dict(w))
    N = vertex_normals(P, G.F)
    for f in G.F:
        s = side_of(P[f[0]])
        sole = np.mean([N[i][2] for i in f]) < -0.6 and P[list(f)][:, 2].max() < 0.02
        m.add_f(f, A.axes.uv(P, f, "foot_" + s), "sole" if sole else "leather")
    for lp in boundary_loops(m.F):
        add_hem(m, lp, thick=0.0022, width=0.006)
    pc = Piece(OUTFIT, "shoes", m, {"leather": "civ_leather_tan", "sole": "civ_leather_dark"}, layer=40,
               hide=["foot_l", "foot_r"], desc="leather turnshoes",
               grime=lambda P: np.stack([1 - 0.22 * smoothstep(0.05, 0.0, P[:, 2])] * 3, 1))
    return pc, dict(G=G, P=P, F=G.F, mesh=m)


# ================================================================================================= tunic
def neckline_z(M, ph, front=0.05, back=0.018):
    return M.z_neck - lerp(back, front, (1 + math.cos(ph)) / 2)


def armpit_dist(M, p):
    return min(np.linalg.norm(p - (M.H["upperarm_" + s] + np.array([-0.03 if s == "l" else 0.03, 0.0, -0.09])))
               for s in SIDES)


def tunic(A, M, shirt_d, legs_d):
    """wool tunic: round neck, sleeves to just above the elbow, bloused over the belt line, knee-length skirt with
    front and back slits (each half follows its thigh), tablet-woven trim at the neck, sleeve ends and hem"""
    cb = A.cb
    z_join = M.z_belt - 0.012
    z_hem = M.z_knee + 0.10
    t_sleeve = {s: M.elbow_t(s) - 0.045 for s in SIDES}

    def keep(c, f):
        if c[2] < z_join - 0.012:
            return False
        s = side_of(c)
        aw = sum(sum(x for b, x in cb.W[i].items() if b.startswith(("upperarm", "lowerarm", "hand"))) for i in f) / len(f)
        if aw > 0.45 and M.arm_t(c, s) > t_sleeve[s] + 0.02:
            return False
        if abs(c[0]) < 0.17 and c[2] > neckline_z(M, phi_of(c)) + 0.01:
            return False
        return True
    G = Garment(A, keep)
    L = classify_loops(G, M)
    for s in SIDES:
        O, Nn = arm_plane(M, s, t_sleeve[s])
        G.snap_loop_plane(L["cuff_" + s], O, Nn)
    G.snap_loop_plane(L["waist"], np.array([0, 0, z_join]), np.array([0, 0, 1.0]))
    G.snap_loop_zcurve(L["neck"], lambda p: neckline_z(M, phi_of(p)))
    T = G.P.copy()
    c = np.zeros(len(T))
    for i, p in enumerate(T):
        r = G.region[i]
        pit = smoothstep(0.03, 0.10, armpit_dist(M, p))              # thinner in the armpit (arms come down)
        if r.startswith(("arm", "hand")):
            s = r[-1]; t = M.arm_t(p, s)
            c[i] = 0.019 + 0.012 * smoothstep(t_sleeve[s] - 0.25, t_sleeve[s], t)
        else:
            blouse = smoothstep(z_join + 0.0, z_join + 0.04, p[2]) * smoothstep(z_join + 0.16, z_join + 0.06, p[2])
            c[i] = 0.016 + 0.008 * blouse - 0.003 * smoothstep(z_join + 0.03, z_join, p[2])
        c[i] = lerp(0.015, c[i], pit)
    P, h = drape_lines(cb, G.idx, G.F, T, c, unders=[(shirt_d["h"], 0.0065), (legs_d["h"], 0.0062)], iters=40,
                       verbose="tunic", bridge=30)
    # wool hangs from the chest / shoulder blades and is gathered by the belt (no leotard over pecs, bust and waist)
    torso = np.array([not r.startswith(("arm", "hand")) for r in G.region])
    h = hang_torso(cb, G.idx, T, h, G.F, torso, z_top=M.z_sh - 0.075, z_join=z_join, gather=0.12, fold_amp=0.28,
                   fold_n=15, seed=33)
    P = T + cb.line_N(G.idx) * h[:, None]
    fn = fold_noise(P, 31, 4.0)
    for i, p in enumerate(P):
        r = G.region[i]
        if r.startswith(("arm", "hand")):
            s = r[-1]; t = M.arm_t(p, s)
            th = ang(p, line_frame(M, p, s))
            h[i] += 0.003 * smoothstep(0.15, t_sleeve[s], t) * (0.5 + 0.5 * math.sin(th * 4 + 2 * fn[i]))
        else:
            ph = phi_of(p)
            bl = smoothstep(z_join, z_join + 0.03, p[2]) * smoothstep(z_join + 0.14, z_join + 0.05, p[2])
            chest = smoothstep(M.z_sh - 0.1, M.z_belt + 0.15, p[2]) * smoothstep(z_join + 0.05, z_join + 0.15, p[2])
            h[i] += 0.0045 * bl * (0.5 + 0.5 * math.sin(ph * 16 + 2.5 * fn[i])) \
                + 0.0025 * chest * (0.5 + 0.5 * math.sin(ph * 9 + 3 * fn[i]))
    P = T + cb.line_N(G.idx) * h[:, None]
    G.P = P
    m = G.to_mesh("cloth", axes=A.axes)
    # skirt over the trousers (+ the shirt tail): hull of the legs / hips without the arms
    comp = Composite([skin_target(cb, 0.0, exclude=("upperarm", "lowerarm", "hand", "thumb", "index", "middle",
                                                    "ring", "pinky")),
                      Target(legs_d["P"], legs_d["F"], 0.0), Target(shirt_d["P"], shirt_d["F"], 0.0)])
    hf = HangField(comp, z_hem - 0.08, z_join + 0.06)
    lm = legs_d["mesh"]; LP = lm.arr()
    kd = KDTree(len(LP))
    for i, p in enumerate(LP):
        kd.insert(p, i)
    kd.balance()

    sw_legs = SurfaceWeights(lm)

    def law(p, z, side):
        """above the crotch: the weights of the trouser surface under the skirt (moves with it); below: the thigh"""
        wu = sw_legs(p)
        wt = norm_w({"thigh_" + side: 0.85, "thigh_twist_01_" + side: 0.15})
        return mix_w(wu, wt, smoothstep(M.z_hip + 0.01, M.z_crotch - 0.06, z))
    hang_skirt(m, L["waist"], hf, z_hem, clear=0.02, flare=0.045, n_folds=13, fold_amp=0.009, row_step=0.034,
               slit_top=M.z_crotch - 0.02, back_drop=0.02, law=law,
               min_clear_fn=lambda z: 0.014 + 0.012 * smoothstep(M.z_knee + 0.25, z_hem, z))
    orient_outward(m, lambda c_: np.array([0.0, 0.012, c_[2]]))
    # trims: a sewn-on tablet-woven band along the neck, sleeve ends and hem (+ slits), then turned-in hems
    for lp in boundary_loops(m.F):
        cz = m.arr()[lp].mean(0)
        wid = 0.024 if cz[2] < M.z_hip else (0.017 if abs(cz[0]) < 0.2 else 0.016)
        add_band(m, lp, wid, "trim:tablet", lift=0.0009, rings=3)
        add_hem(m, lp, thick=0.003, width=0.016)
    pc = Piece(OUTFIT, "tunic", m, {"cloth": "civ_wool_russet", "trim": "civ_trim"}, layer=50,
               hide=["torso", "upperarm_l", "upperarm_r"], desc="wool tunic, knee length, slit front and back",
               notes="over the shirt and the trouser tops; belt goes over it",
               grime=lambda P: np.stack([1 - 0.18 * smoothstep(M.z_knee + 0.25, M.z_knee + 0.07, P[:, 2])] * 3, 1))
    return pc, dict(G=G, P=m.arr(), F=m.F, mesh=m, h={k: float(x) for k, x in zip(G.idx, h)})


def band_faces(m, loop, slot, rings=1):
    """faces touching a boundary loop become a trim band: u = arc length along the loop, v 0 (edge) -> 1 (inside)"""
    P = m.arr()
    ls = {v: k for k, v in enumerate(loop)}
    n = len(loop)
    al = [0.0]
    for k in range(1, n + 1):
        al.append(al[-1] + np.linalg.norm(P[loop[k % n]] - P[loop[k - 1]]))
    kd = KDTree(n)
    for k, v in enumerate(loop):
        kd.insert(P[v], k)
    kd.balance()
    for fi, f in enumerate(m.F):
        on = [v in ls for v in f]
        if not any(on) or m.S[fi].startswith("trim"):
            continue
        ks = [ls[v] if v in ls else kd.find(P[v])[1] for v in f]
        us = np.array([al[k] for k in ks])
        if us.max() - us.min() > al[-1] / 2:               # wraps around the loop start
            us = np.where(us < al[-1] / 2, us + al[-1], us)
        m.UV[fi] = [(u, 0.0 if o else 1.0) for u, o in zip(us, on)]
        m.S[fi] = slot


# ================================================================================================= belt, pouch, knife
def belt(A, M, tunic_d, z=None, tilt=-0.012, over=0.0015, outfit=OUTFIT, pouch_side="r", knife=True, slot_name="belt"):
    """leather belt seated on the tunic (sampled on its surface), iron buckle + tongue, strap tail, leather pouch on
    the right hip, a small knife in a sheath on the left. Belt vertices take the weights of the garment surface under
    them (it moves with the tunic); pouch / knife are rigid (pelvis + a quarter of the thigh)."""
    cb = A.cb
    z = z if z is not None else M.z_belt
    tm = tunic_d["mesh"]
    TP = tm.arr()
    kd = KDTree(len(TP))
    for i, p in enumerate(TP):
        kd.insert(p, i)
    kd.balance()
    sw_ = SurfaceWeights(tm)
    wat = lambda p: sw_(p)
    comp = Composite([Target(TP, tm.F, 0.0)])
    cen = (0.0, 0.012)
    nseg = 72
    ring = ring_on(comp, z, cen, nseg, over + 0.0025, tilt=tilt, half_h=0.02)
    m = CMesh()
    up = np.tile([0, 0, 1.0], (nseg, 1))
    band_along(m, ring, up, 0.036, 0.0045, "strap", wat, closed=True)
    # buckle: iron D-frame + bar + tongue at the front, a touch to the character's right
    k0 = int(round(nseg * (1 - 0.035)))
    p0 = ring[k0]; T = nrm(ring[(k0 + 1) % nseg] - ring[k0 - 1]); U = np.array([0, 0, 1.0]); O_ = np.cross(T, U)
    base = p0 + O_ * 0.0062
    wb = wat(p0)
    frame = []
    for k in range(20):
        a = math.pi * (k / 19.0) - math.pi / 2
        frame.append(base + T * (0.024 * max(0.0, math.cos(a)) + 0.004) + U * 0.024 * math.sin(a))
    frame.append(base + U * -0.024); frame.append(base + U * 0.024 * -1 + T * 0.0)
    path = [base + U * 0.024] + frame + [base + U * -0.024]
    tube_path(m, np.array(frame), 0.0028, 6, "iron", lambda p: dict(wb), cap=True)
    tube_path(m, np.array([base + U * 0.026, base - U * 0.026]), 0.0026, 6, "iron", lambda p: dict(wb), cap=True)
    tube_path(m, np.array([base + T * 0.001, base + T * 0.026 + O_ * 0.002]), 0.0018, 5, "iron", lambda p: dict(wb), cap=True)
    # strap tail: from the buckle, tucked under the belt and hanging down the front of the tunic
    tail = []
    for k in range(9):
        zz = z - 0.004 - k * 0.018
        ph = 2 * math.pi * (k0 - 2.5) / nseg - 0.02 * k
        d = np.array([math.sin(ph), -math.cos(ph), 0.0])
        h = comp.ray(np.array([cen[0], cen[1], zz]) + d * 0.75, -d, 0.75)
        r = (0.75 - h) if h else 0.2
        tail.append(np.array([cen[0], cen[1], zz]) + d * (r + 0.0085 - 0.002 * (k == 0)))
    tail = np.array(tail)
    upt = np.array([np.cross(np.array([math.cos(2 * math.pi * (k0 - 2.5) / nseg), math.sin(2 * math.pi * (k0 - 2.5) / nseg), 0.0]), [0, 0, 1.0])
                    for _ in tail])
    upt = np.array([nrm(np.cross(np.array([0, 0, 1.0]), nrm(p - np.array([cen[0], cen[1], p[2]])))) for p in tail])
    band_along(m, tail, upt, 0.030, 0.0035, "strap", wat, closed=False, uv_u0=0.3)
    # pouch on the right hip (rigid)
    sg = -1 if pouch_side == "r" else 1
    ph = (2 * math.pi - 2.15) if pouch_side == "r" else 2.15     # behind the hip: clear of the hanging hand and thigh
    d = np.array([math.sin(ph), -math.cos(ph), 0.0])
    r = hull_radius_span(comp, z - 0.15, z - 0.02, ph, cen) or 0.2
    X = nrm(np.cross(np.array([0, 0, 1.0]), d)); Z = np.array([0, 0.0, 1.0])
    pc_ = np.array([cen[0], cen[1], z - 0.075]) + d * (r + 0.027)
    wp = norm_w({"pelvis": 0.72, "thigh_" + pouch_side: 0.28})
    rounded_box(m, pc_, X, d, Z, (0.058, 0.022, 0.062), e=0.35, nu=20, nv=10, slot="pouch", w=wp)
    rounded_box(m, pc_ + Z * 0.036 + d * 0.004, X, d, Z, (0.062, 0.025, 0.028), e=0.25, nu=20, nv=8, slot="pouch", w=wp)
    # loops of the pouch over the belt
    for dx in (-0.03, 0.03):
        lp = np.array([pc_ + X * dx + Z * 0.06 + d * 0.0, pc_ + X * dx + Z * 0.085 + d * 0.004])
        tube_path(m, lp, 0.005, 5, "strap", lambda p: dict(wp), cap=True)
    if knife:
        ph = 2.5 if pouch_side == "r" else 2 * math.pi - 2.5     # back of the other hip
        d = np.array([math.sin(ph), -math.cos(ph), 0.0])
        r = hull_radius_span(comp, z - 0.17, z + 0.0, ph, cen) or 0.2
        X = nrm(np.cross(np.array([0, 0, 1.0]), d))
        ax = nrm(np.array([0.0, 0.0, -1.0]) + X * 0.25)
        top = np.array([cen[0], cen[1], z + 0.006]) + d * (r + 0.016)
        wk = norm_w({"pelvis": 0.85, ("thigh_l" if pouch_side == "r" else "thigh_r"): 0.15})
        # sheath (tapered), grip, pommel
        prof = [(0.0035, -0.17), (0.009, -0.15), (0.013, -0.09), (0.014, -0.02), (0.015, 0.0)]
        lathe(m, top, ax, d, prof, 10, "strap", w=wk, close_bottom=True)
        grip = [(0.013, 0.0), (0.0105, -0.012), (0.0095, -0.05), (0.0115, -0.085), (0.0125, -0.09)]
        lathe(m, top, -ax, d, [(r_, -h_) for r_, h_ in grip], 10, "wood", w=wk)
        lathe(m, top - ax * 0.09, -ax, d, [(0.0125, 0.0), (0.013, 0.004), (0.0, 0.009)], 10, "iron", w=wk, close_top=True)
    orient_outward_parts(m)
    return Piece(outfit, slot_name, m, {"strap": "civ_leather_dark", "iron": "civ_iron", "pouch": "civ_leather_tan",
                                        "wood": "civ_wood"}, layer=70, desc="belt, buckle, pouch" + (", knife" if knife else ""),
                 notes="seated on the %s surface" % ("tunic" if outfit == OUTFIT else "gambeson"))


def orient_outward_parts(m):
    """accessories are built with outward winding by construction; nothing to fix (kept as a hook)"""
    return m


# ================================================================================================= straw hat
def straw_hat(A, M):
    """wide-brimmed straw hat: plaited-straw crown sized over the hair, drooping brim, wool band; rigid to the head
    (MakeClothes RIGID triangle on the skull: it follows head-shape morphs as one unit, never the face)"""
    cb, rig = A.cb, A.rig
    hair = next((o for o in rig.children if o.name.endswith("_hair")), None)
    pts = [cb.co[cb.co[:, 2] > M.z_top - 0.2]]
    if hair is not None:
        dg = bpy.context.evaluated_depsgraph_get()
        me = hair.evaluated_get(dg).to_mesh()
        hp = np.array([v.co[:] for v in me.vertices]); hair.evaluated_get(dg).to_mesh_clear()
        pts.append(hp)
    allp = np.concatenate(pts)
    top = allp[:, 2].max()
    hy = cb.co[cb.co[:, 2] > M.z_top - 0.1][:, 1].mean()
    O = np.array([0.0, hy + 0.004, 0.0])
    # crown: radius of the head+hair hull at each height (+10 mm), band at the brow ring
    z_band = M.z_top - 0.085
    def hull_r(z):
        s = allp[np.abs(allp[:, 2] - z) < 0.006]
        if len(s) == 0:
            return 0.02
        d = np.hypot(s[:, 0] - O[0], s[:, 1] - O[1])
        return np.percentile(d, 99)
    r_band = hull_r(z_band) + 0.012
    prof = []
    ztop = top + 0.028
    for k, t in enumerate(np.linspace(0, 1, 9)):
        z = lerp(ztop, z_band, t)
        rz = hull_r(z) + 0.013 if t > 0.15 else 0.0
        rz = max(rz, r_band * math.sin(math.pi / 2 * min(1.0, t * 1.9)) * (0.92 + 0.08 * t))
        prof.append((max(rz, 0.004 if k else 0.0), z))
    prof[0] = (0.0, ztop)
    # a softly domed top: ease the first rings
    prof = [(r_, z_) for r_, z_ in prof]
    rb = [r_band + 0.004, r_band + 0.05, r_band + 0.10, r_band + 0.135]
    brim_top = [(rb[0], z_band - 0.004), (rb[1], z_band - 0.012), (rb[2], z_band - 0.028), (rb[3], z_band - 0.046)]
    roll = [(rb[3] + 0.004, z_band - 0.052), (rb[3], z_band - 0.056)]
    brim_bot = [(rb[2], z_band - 0.035), (rb[1], z_band - 0.019), (rb[0] - 0.004, z_band - 0.011), (r_band - 0.006, z_band - 0.009),
                (r_band - 0.008, z_band + 0.015)]
    full = prof + brim_top + roll + brim_bot
    m = CMesh()
    wh = {"head": 1.0}
    rows = lathe(m, O * np.array([1, 1, 0]), np.array([0, 0, 1.0]), np.array([0, -1.0, 0]),
                 [(r_, z_) for r_, z_ in full], 40, "straw", w=wh, g="civ_rigid_head", r_uv=0.12)
    # the first profile point is the apex (r = 0): its ring collapses; weld it
    m.weld(1e-7)
    # band: the crown rows just above the brim
    nb_rows = len(prof)
    P = m.arr()
    for fi, f in enumerate(m.F):
        zc = P[list(f)][:, 2].mean(); rc = np.hypot(P[list(f)][:, 0] - O[0], P[list(f)][:, 1] - O[1]).mean()
        if z_band - 0.002 < zc < z_band + 0.03 and rc < r_band + 0.012:
            m.S[fi] = "band"
    # tilt forward a little (brim shades the eyes) about the band centre
    ang = math.radians(4.0)
    R = np.array([[1, 0, 0], [0, math.cos(ang), -math.sin(ang)], [0, math.sin(ang), math.cos(ang)]])
    c0 = np.array([O[0], O[1], z_band])
    P = m.arr()
    m.set_pos((P - c0) @ R.T + c0)
    # the droop must not pass the face: lift the front of the brim a little
    return Piece(OUTFIT, "hat", m, {"straw": "civ_straw", "band": "civ_wool_brown"}, layer=90, hide=[],
                 desc="straw hat (rigid to the head)", notes="over the hair (crown sized to the hair hull)")


# ================================================================================================= build
def build(A, only=None):
    M = Measures(A.cb)
    A.M = M
    out, info = {}, {}
    t0 = time.time()
    sh, info["shirt"] = shirt(A, M); out["shirt"] = sh
    tr, info["trousers"] = trousers(A, M, info["shirt"]); out["trousers"] = tr
    so, info["shoes"] = shoes(A, M, info["trousers"]); out["shoes"] = so
    tu, info["tunic"] = tunic(A, M, info["shirt"], info["trousers"]); out["tunic"] = tu
    out["belt"] = belt(A, M, info["tunic"])
    out["hat"] = straw_hat(A, M)
    layer_pass(A, out)
    clog("peasant built in %.1fs: %s" % (time.time() - t0, {k: v.m.tris() for k, v in out.items()}))
    return out, info, M


LAYERS = [("shirt", [("skin", 0.0025)]),
          ("trousers", [("skin", 0.0025), ("shirt", 0.0035)]),
          ("shoes", [("skin", 0.0025), ("trousers", 0.0035)]),
          ("tunic", [("skin", 0.004), ("shirt", 0.0045), ("trousers", 0.0045)]),
          ("belt", [("tunic", 0.0015), ("trousers", 0.003), ("shirt", 0.003)])]


def layer_pass(A, pieces, layers=None):
    """enforce the layer order on the finished meshes (hems included): outer pieces pushed out of the inner ones"""
    sk = skin_target(A.cb, 0.0)
    for name, unders in (layers or LAYERS):
        if name not in pieces:
            continue
        tg = []
        for u, gap in unders:
            if u == "skin":
                t = skin_target(A.cb, gap)
            elif u in pieces:
                t = Target.of_mesh(pieces[u].m, gap, u)
            else:
                continue
            tg.append(t)
        resolve_layers(pieces[name].m, tg, verbose=name)


def hides(A, pieces, info, M):
    """skin each piece hides (MPFB delete group): normal ray + 4 tilted rays hit the piece within dmax, eroded"""
    cb = A.cb
    res = {}
    for name, pc in pieces.items():
        P = pc.m.arr(); F = pc.m.F
        if name == "shirt":
            res[name] = covered_verts(cb, P, F, 0.035, erode=1, tight=0.016, open_dist=0.03)
        elif name == "trousers":
            res[name] = covered_verts(cb, P, F, 0.04, erode=1, tight=0.018, open_dist=0.025)
        elif name == "shoes":
            # the whole foot inside the shoe (toes included: they bend at the ball more than the shoe does)
            top = P[:, 2].max()
            feet = [i for i in range(NB) if cb.dom[i].startswith(("foot", "ball")) and cb.co[i][2] < top - 0.012]
            res[name] = sorted(set(covered_verts(cb, P, F, 0.03, erode=1, tight=0.012, open_dist=0.012)) | set(feet))
        elif name == "tunic":
            res[name] = covered_verts(cb, P, F, 0.06, erode=1, zmin=M.z_crotch + 0.02)
        else:
            res[name] = []
    return res


def author(A, only=None):
    pieces, info, M = build(A)
    hid = hides(A, pieces, info, M)
    paths = {}
    for name in ORDER:
        if only and name not in only:
            continue
        pc = pieces[name]
        paths[name] = write_piece(A, pc, delete=hid[name])
    return paths


def preview(A, tag="prev_peasant"):
    pieces, info, M = build(A)
    rig = A.rig
    objs = preview_objects({k: v.m for k, v in pieces.items()}, rig)
    render_clay(rig, tag + ("" if A.kind == "male" else "_" + A.kind))


def render_clay(rig, tag, poses=("rest", "arms_up", "squat", "walk")):
    import outfit_civ_poses as PZ
    os.makedirs(RENC, exist_ok=True)
    render_setup("BLENDER_WORKBENCH", res=(700, 1000))
    sc = bpy.context.scene
    sc.display.shading.light = 'STUDIO'; sc.display.shading.color_type = 'MATERIAL'; sc.display.shading.show_cavity = True
    sc.view_settings.view_transform = 'Standard'
    for o in rig.children:
        if o.type == 'MESH' and o.name.endswith("_hair"):
            o.hide_render = False
    for pose in poses:
        pose_reset(rig)
        PZ.POSES[pose](rig)
        for nm, loc, tgt in (("front", (0, -4.3, 1.0), (0, 0, 0.93)), ("back", (0.0, 4.3, 1.0), (0, 0, 0.93)),
                             ("side", (4.3, -0.3, 1.0), (0, 0, 0.93)), ("34", (2.3, -3.4, 1.35), (0, 0, 0.93))):
            if pose != "rest" and nm in ("side",):
                continue
            camera(loc, tgt, lens=52)
            render(os.path.join(RENC, "%s_%s_%s.png" % (tag, pose, nm)))
    pose_reset(rig)


def check(A, poses=None):
    """penetration gate on the unwritten pieces (skinned previews + the skin they leave visible)"""
    import outfit_civ_qa as QA
    pieces, info, M = build(A)
    hid = hides(A, pieces, info, M)
    objs = preview_objects({k: v.m for k, v in pieces.items()}, A.rig)
    union = sorted(set(i for v in hid.values() for i in v))
    body = QA.masked_body(A.bm, union)
    A.bm.hide_render = True
    import outfit_civ_poses as PZ
    res = QA.penetration(A.rig, objs, body, poses=poses or (PZ.STRESS + PZ.EXTRA), tag="peasant", layers={k: v.layer for k, v in pieces.items()})
    os.makedirs(RENC, exist_ok=True)
    json.dump(res, open(os.path.join(RENC, "check_peasant_%s.json" % A.kind), "w"), indent=1)
    return res


if __name__ == "__main__":
    args = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    mode = args[0] if args else "preview"
    t0 = time.time()
    kind = next((o.name[4:] for o in bpy.data.objects if o.type == 'ARMATURE' and o.name.startswith("rts_")), "male")
    A = Author(kind)
    if mode == "preview":
        preview(A)
    elif mode == "check":
        check(A, poses=args[1:] or None)
    elif mode == "author":
        author(A, only=args[1:] or None)
    clog("peasant %s done in %.1fs" % (mode, time.time() - t0))
