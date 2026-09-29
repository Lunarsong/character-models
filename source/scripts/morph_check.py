"""Numeric morph-target test on an exported character GLB (what three.js / an engine actually receives).

For every face key it measures, per mesh, the largest vertex offset and the side of the face it moves, then checks:
  - the body carries all 52 ARKit names (exact spelling) + the 15 OVR visemes, and every part's targetNames are unique
  - every ARKit key except tongueOut moves the body; tongueOut moves the tongue
  - *Left keys move the character's LEFT (+X in glTF, the character faces +Z), *Right keys the right
  - eyeBlinkL/R move the body lids AND the eyelashes, and CLOSE the eye: rays shot at the iris from the front must hit
    lid skin, not the eyeball (open fraction < 3 %); the other eye must stay open (open fraction > 60 %)
  - eyeLook* rotate the eyeballs; brow keys move the eyebrow cards
  - teeth (skinned: upper set -> `head`, lower set -> `jaw`): no face key moves the upper set; jawOpen moves the lower
    set (> 3 mm) and every face key moves the lower set RIGIDLY (pairwise distances kept within 0.1 mm); customisation
    morphs (cust_*) move each set by one affine map (residual < 0.1 mm), pose correctives (cor_*) are body-only
  - mouth keys (smile, stretch, frown, press, upper up, extras, ...) never move the teeth or tongue (lip shapes)
  - the extra shapes (EXTRA) exist on the body, move it, and their Left / Right halves move the right side
  - every expression preset of scripts/expressions.json: 'Wink*' presets close exactly one eye; per preset the
    number of tooth-enamel vertices buried in the lips / cheeks (ray parity) is reported (info)
Exit code 1 on any failure.

usage: python3 scripts/morph_check.py out/base_male.glb [out/base_female.glb ...] [--json out/morph_report.json] [-v]
"""
import sys, os, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from check_glb import load, accessor

ARKIT = ["browDownLeft", "browDownRight", "browInnerUp", "browOuterUpLeft", "browOuterUpRight", "cheekPuff",
         "cheekSquintLeft", "cheekSquintRight", "eyeBlinkLeft", "eyeBlinkRight", "eyeLookDownLeft", "eyeLookDownRight",
         "eyeLookInLeft", "eyeLookInRight", "eyeLookOutLeft", "eyeLookOutRight", "eyeLookUpLeft", "eyeLookUpRight",
         "eyeSquintLeft", "eyeSquintRight", "eyeWideLeft", "eyeWideRight", "jawForward", "jawLeft", "jawOpen", "jawRight",
         "mouthClose", "mouthDimpleLeft", "mouthDimpleRight", "mouthFrownLeft", "mouthFrownRight", "mouthFunnel",
         "mouthLeft", "mouthLowerDownLeft", "mouthLowerDownRight", "mouthPressLeft", "mouthPressRight", "mouthPucker",
         "mouthRight", "mouthRollLower", "mouthRollUpper", "mouthShrugLower", "mouthShrugUpper", "mouthSmileLeft",
         "mouthSmileRight", "mouthStretchLeft", "mouthStretchRight", "mouthUpperUpLeft", "mouthUpperUpRight",
         "noseSneerLeft", "noseSneerRight", "tongueOut"]
VISEMES = ["viseme_sil", "viseme_PP", "viseme_FF", "viseme_TH", "viseme_DD", "viseme_kk", "viseme_CH", "viseme_SS",
           "viseme_nn", "viseme_RR", "viseme_aa", "viseme_E", "viseme_I", "viseme_O", "viseme_U"]
EXTRA = ["smileOpenLeft", "smileOpenRight", "snarl", "grimace"]
MOVE_EPS = 2e-4          # 0.2 mm: "moves this mesh"
PRESETS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "expressions.json")
# keys that are pure lip / cheek shapes: rigid teeth and the tongue must not follow them
LIP_ONLY = ["mouthSmile", "mouthStretch", "mouthFrown", "mouthPress", "mouthUpperUp", "mouthLowerDown", "mouthDimple",
            "mouthRoll", "mouthShrug", "mouthFunnel", "mouthPucker", "cheek", "nose", "brow", "eye", "smileOpen",
            "snarl", "grimace"]


def meshes_of(js, bin_):
    """{role: dict(pos, idx, prim_mat, targets{name: delta})} for every mesh node (primitives concatenated)."""
    out = {}
    for n in js["nodes"]:
        if "mesh" not in n:
            continue
        m = js["meshes"][n["mesh"]]
        names = (m.get("extras") or {}).get("targetNames", [])
        pos, idx, mats, tg, off, joint, uvs = [], [], [], {k: [] for k in names}, 0, [], []
        jn = [js["nodes"][j]["name"] for j in js["skins"][n["skin"]]["joints"]] if "skin" in n else []
        for p in m["primitives"]:
            P = accessor(js, bin_, p["attributes"]["POSITION"]).astype(np.float64)
            if "JOINTS_0" in p["attributes"]:
                J = accessor(js, bin_, p["attributes"]["JOINTS_0"]); W = accessor(js, bin_, p["attributes"]["WEIGHTS_0"])
                joint += [jn[J[i, np.argmax(W[i])]] for i in range(len(J))]
            uvs.append(accessor(js, bin_, p["attributes"]["TEXCOORD_0"]) if "TEXCOORD_0" in p["attributes"] else np.zeros((len(P), 2)))
            I = accessor(js, bin_, p["indices"]).ravel().astype(np.int64).reshape(-1, 3) + off
            pos.append(P); idx.append(I)
            mats += [js["materials"][p["material"]]["name"] if "material" in p else ""] * len(I)
            for k, t in zip(names, p.get("targets", [])):
                tg[k].append(accessor(js, bin_, t["POSITION"]).astype(np.float64) if "POSITION" in t else np.zeros_like(P))
            off += len(P)
        # '<kind>_<role>' (kind may contain '_': male_stocky_body), alternates '<kind>_<role>__<style>'
        nm = n["name"]
        role = nm.split("__")[0].rsplit("_", 1)[1] + ("__" + nm.split("__", 1)[1] if "__" in nm else "") if "_" in nm else nm
        out[role] = dict(name=n["name"], pos=np.concatenate(pos), idx=np.concatenate(idx), mats=np.array(mats),
                         targets={k: np.concatenate(v) for k, v in tg.items()}, names=names, joint=np.array(joint),
                         uv=np.concatenate(uvs))
    return out


def islands(n, tris):
    par = np.arange(n)

    def f(x):
        while par[x] != x:
            par[x] = par[par[x]]; x = par[x]
        return x
    for a, b, c in tris:
        for u, v in ((a, b), (b, c)):
            ru, rv = f(u), f(v)
            if ru != rv:
                par[ru] = rv
    return np.array([f(i) for i in range(n)])


def ortho_hits(P, T, xy):
    """Front-most (max z) hit depth of rays travelling -Z at points xy against triangles T of positions P.
    Returns z per ray (-inf = no hit)."""
    A, B, C = P[T[:, 0]], P[T[:, 1]], P[T[:, 2]]
    x, y = xy[:, 0][:, None], xy[:, 1][:, None]
    d = (B[:, 1] - C[:, 1]) * (A[:, 0] - C[:, 0]) + (C[:, 0] - B[:, 0]) * (A[:, 1] - C[:, 1])
    ok = np.abs(d) > 1e-14
    d = np.where(ok, d, 1.0)
    l1 = ((B[:, 1] - C[:, 1]) * (x - C[:, 0]) + (C[:, 0] - B[:, 0]) * (y - C[:, 1])) / d
    l2 = ((C[:, 1] - A[:, 1]) * (x - C[:, 0]) + (A[:, 0] - C[:, 0]) * (y - C[:, 1])) / d
    l3 = 1 - l1 - l2
    inside = (l1 >= 0) & (l2 >= 0) & (l3 >= 0) & ok
    z = l1 * A[:, 2] + l2 * B[:, 2] + l3 * C[:, 2]
    return np.where(inside, z, -np.inf).max(1)


def inside_flesh(Pb, Tb, Q, fwd=2, sign=1.0, region=None):
    """Ray parity: for each point Q, count crossings of a ray going forward (axis `fwd`, direction `sign`) with the
    body triangles Tb (positions Pb). The mouth cavity is outside the flesh (a ray from a tooth crosses the lip twice
    or leaves through the mouth opening), so an ODD count means the point is buried in the lips / chin / cheek."""
    ax = [a for a in range(3) if a != fwd]
    A, B, C = Pb[Tb[:, 0]], Pb[Tb[:, 1]], Pb[Tb[:, 2]]
    if region is not None:
        lo, hi = region
        c = (A + B + C) / 3
        k = np.all((c > lo) & (c < hi), axis=1)
        A, B, C = A[k], B[k], C[k]
    cnt = np.zeros(len(Q), dtype=np.int64)
    for s in range(0, len(Q), 512):
        q = Q[s:s + 512]
        x, y = q[:, ax[0]][:, None], q[:, ax[1]][:, None]
        d = (B[:, ax[1]] - C[:, ax[1]]) * (A[:, ax[0]] - C[:, ax[0]]) + (C[:, ax[0]] - B[:, ax[0]]) * (A[:, ax[1]] - C[:, ax[1]])
        ok = np.abs(d) > 1e-16
        d = np.where(ok, d, 1.0)
        l1 = ((B[:, ax[1]] - C[:, ax[1]]) * (x - C[:, ax[0]]) + (C[:, ax[0]] - B[:, ax[0]]) * (y - C[:, ax[1]])) / d
        l2 = ((C[:, ax[1]] - A[:, ax[1]]) * (x - C[:, ax[0]]) + (A[:, ax[0]] - C[:, ax[0]]) * (y - C[:, ax[1]])) / d
        l3 = 1 - l1 - l2
        inside = (l1 >= 0) & (l2 > 0) & (l3 > 0) & ok
        z = l1 * A[:, fwd] + l2 * B[:, fwd] + l3 * C[:, fwd]
        ahead = (z - q[:, fwd][:, None]) * sign > 0
        cnt[s:s + 512] = (inside & ahead).sum(1)
    return cnt % 2 == 1


def posed(m, weights):
    P = m["pos"].copy()
    for k, w in weights.items():
        if w and k in m["targets"]:
            P += w * m["targets"][k]
    return P


def mouth_penetration(M, weights, parts=("teeth", "tongue")):
    """Number of teeth / gum / tongue vertices buried in the body flesh for a weight mix (glTF: forward = +Z)."""
    body = M["body"]
    Pb = posed(body, weights)
    out = {}
    for part in parts:
        if part not in M:
            continue
        Q = posed(M[part], weights)
        lo, hi = Q.min(0) - 0.02, Q.max(0) + 0.02
        hi[2] = np.inf                  # keep all skin AHEAD (+Z): clipping it flips the parity of deep points
        out[part] = int(inside_flesh(Pb, body["idx"], Q, fwd=2, sign=1.0, region=(lo, hi)).sum())
    return out


def eye_open_fraction(M, weights, side):
    """Fraction of iris rays (front view) that reach the eyeball of eye `side` (+1 = character's left) with the given
    morph weights applied to the body (lids). 1 = fully open, 0 = closed."""
    body, eyes = M["body"], M["eyes"]
    Pb = body["pos"].copy()
    for k, w in weights.items():
        if k in body["targets"]:
            Pb += w * body["targets"][k]
    Pe = eyes["pos"]
    sel = np.sign(Pe[:, 0]) == side
    ez = Pe[sel]
    apex = ez[np.argmax(ez[:, 2])]
    r = 0.0045                                      # iris / pupil disc radius probed (m)
    g = np.linspace(-r, r, 21)
    xy = np.array([(apex[0] + a, apex[1] + b) for a in g for b in g if a * a + b * b <= r * r])
    # only triangles near the eye (speed)
    def near(P, T):
        c = P[T].mean(1)
        return T[(np.abs(c[:, 0] - apex[0]) < 0.03) & (np.abs(c[:, 1] - apex[1]) < 0.03) & (c[:, 2] > apex[2] - 0.04)]
    Tb, Te = near(Pb, body["idx"]), near(Pe, eyes["idx"])
    zb, ze = ortho_hits(Pb, Tb, xy), ortho_hits(Pe, Te, xy)
    return float(((ze > -np.inf) & (ze >= zb)).mean())


def check(path, verbose=False):
    js, bin_ = load(path)
    M = meshes_of(js, bin_)
    fails, rows = [], {}
    body = M["body"]
    for role, m in M.items():
        if len(set(m["names"])) != len(m["names"]):
            fails.append("%s: duplicate targetNames" % m["name"])
    missing = [k for k in ARKIT + VISEMES if k not in body["names"]]
    if missing:
        fails.append("body misses %s" % missing)
    allkeys = list(dict.fromkeys(body["names"] + [k for m in M.values() for k in m["names"]]))
    for k in allkeys:
        row = {}
        for role, m in M.items():
            if k not in m["targets"]:
                continue
            D = m["targets"][k]; mag = np.linalg.norm(D, axis=1)
            if mag.max() < MOVE_EPS:
                continue
            w = mag / mag.sum()
            row[role] = dict(max_mm=round(float(mag.max()) * 1e3, 2), cx_mm=round(float((m["pos"][:, 0] * w).sum()) * 1e3, 1))
        rows[k] = row
    # rules ------------------------------------------------------------------------------------------------------
    for k in EXTRA:
        if k not in body["names"]:
            fails.append("extra shape %s missing on the body" % k)
    for k in ARKIT + [e for e in EXTRA if e in body["names"]]:
        r = rows.get(k, {})
        if k == "tongueOut":
            if "tongue" not in r:
                fails.append("tongueOut does not move the tongue")
            continue
        if "body" not in r:
            fails.append("%s does not move the body" % k)
            continue
        cx = r["body"]["cx_mm"]
        if k.endswith("Left") and k not in ("mouthLeft", "jawLeft") and cx < 3:
            fails.append("%s moves the wrong side (body centroid x %.1f mm, character's left is +X)" % (k, cx))
        if k.endswith("Right") and k not in ("mouthRight", "jawRight") and cx > -3:
            fails.append("%s moves the wrong side (body centroid x %.1f mm)" % (k, cx))
        if k.startswith("eyeBlink") and r.get("eyelashes", {}).get("max_mm", 0) < 2:
            fails.append("%s does not move the eyelashes" % k)
        if k.startswith("eyeLook") and "eyes" not in r:
            fails.append("%s does not rotate the eyeball" % k)
        if k.startswith("brow") and "eyebrows" not in r:
            fails.append("%s does not move the eyebrow cards" % k)
        if any(k.startswith(p) for p in LIP_ONLY):
            for part in ("teeth", "tongue"):
                if part in r:
                    fails.append("%s moves the %s (%.2f mm)" % (k, part, r[part]["max_mm"]))
    for k, side in (("mouthLeft", 1), ("jawLeft", 1), ("mouthRight", -1), ("jawRight", -1)):
        r = rows.get(k, {}).get("body")
        if r:
            D = body["targets"][k]
            mv = D[np.linalg.norm(D, axis=1) > 1e-3]
            if len(mv) and np.sign(mv[:, 0].mean()) != side:
                fails.append("%s pushes the mouth to the wrong side (mean dx %.2f mm)" % (k, mv[:, 0].mean() * 1e3))
    # blink closure
    blink = {}
    for k, side in (("eyeBlinkLeft", 1), ("eyeBlinkRight", -1)):
        rest = eye_open_fraction(M, {}, side)
        closed = eye_open_fraction(M, {k: 1.0}, side)
        other = eye_open_fraction(M, {k: 1.0}, -side)
        half = eye_open_fraction(M, {k: 0.5}, side)
        blink[k] = dict(rest=round(rest, 3), at_1=round(closed, 3), at_half=round(half, 3), other_eye=round(other, 3))
        if closed > 0.03:
            fails.append("%s leaves %.0f %% of the iris visible (eye not closed)" % (k, closed * 100))
        if other < 0.6:
            fails.append("%s also closes the other eye (%.0f %% open)" % (k, other * 100))
    # expression presets as the viewer plays them: winks close one eye; buried enamel per preset (info)
    presets = json.load(open(PRESETS))["presets"] if os.path.exists(PRESETS) else {}
    wink, bury = {}, {}
    enamel = None
    if "teeth" in M:
        t = M["teeth"]
        enamel = (t["uv"][:, 0] < 0.5 / 1.3)                  # crown atlas: u < 0.385 = enamel (f < 1)
    for nm, ws in presets.items():
        ws = {k: v for k, v in ws.items() if not k.startswith("_")}
        if nm.startswith("Wink"):
            side = 1 if ws.get("eyeBlinkLeft", 0) > ws.get("eyeBlinkRight", 0) else -1
            wink[nm] = dict(closed_eye=round(eye_open_fraction(M, ws, side), 3), open_eye=round(eye_open_fraction(M, ws, -side), 3))
            if wink[nm]["closed_eye"] > 0.03:
                fails.append("%s: the winking eye is %.0f %% open" % (nm, wink[nm]["closed_eye"] * 100))
            if wink[nm]["open_eye"] < 0.4:
                fails.append("%s: the other eye is closed too" % nm)
        if enamel is not None:
            body_P = posed(body, ws); tq = posed(M["teeth"], ws)
            lo_, hi_ = tq.min(0) - 0.02, tq.max(0) + 0.02
            b = inside_flesh(body_P, body["idx"], tq, fwd=2, sign=1.0, region=(lo_, hi_))
            bury[nm] = int((b & enamel).sum())
    # teeth: the upper set (head) never moves; the lower set (jaw) moves rigidly
    teeth_rep, cust_rep = {}, {}
    if "teeth" in M:
        t = M["teeth"]
        up, lo = t["joint"] == "head", t["joint"] == "jaw"
        if not up.any() or not lo.any():
            fails.append("teeth: expected an upper set skinned to 'head' and a lower set skinned to 'jaw'")
        rng = np.random.default_rng(1)
        smp = rng.choice(np.nonzero(lo)[0], size=min(300, int(lo.sum())), replace=False) if lo.any() else []
        for k in t["names"]:
            mag = np.linalg.norm(t["targets"][k], axis=1)
            if k.startswith("cust_"):
                # customisation morphs reshape the mouth: each set follows by one affine map (no bent crowns)
                res = 0.0
                for sel in (up, lo):
                    if sel.any() and mag[sel].max() > 1e-5:
                        P0 = t["pos"][sel]; P1 = P0 + t["targets"][k][sel]
                        with np.errstate(all="ignore"):         # (spurious BLAS matmul warnings on macOS)
                            X = np.linalg.lstsq(np.c_[P0, np.ones(len(P0))], P1, rcond=None)[0]
                            res = max(res, float(np.linalg.norm(np.c_[P0, np.ones(len(P0))] @ X - P1, axis=1).max()))
                cust_rep[k] = dict(upper_mm=round(float(mag[up].max()) * 1e3 if up.any() else 0, 2),
                                   lower_mm=round(float(mag[lo].max()) * 1e3 if lo.any() else 0, 2), affine_err_mm=round(res * 1e3, 3))
                if res > 1e-4:
                    fails.append("teeth: %s distorts a tooth set (affine residual %.2f mm)" % (k, res * 1e3))
                continue
            if k.startswith("cor_"):
                continue
            P0 = t["pos"][smp]; P1 = P0 + t["targets"][k][smp]
            d0 = np.linalg.norm(P0[:, None] - P0[None], axis=2); d1 = np.linalg.norm(P1[:, None] - P1[None], axis=2)
            rig_err = float(np.abs(d1 - d0).max()) if len(smp) else 0.0
            teeth_rep[k] = dict(upper_mm=round(float(mag[up].max()) * 1e3 if up.any() else 0, 2),
                                lower_mm=round(float(mag[lo].max()) * 1e3 if lo.any() else 0, 2), rigid_err_mm=round(rig_err * 1e3, 3))
            if up.any() and mag[up].max() > 5e-4:
                fails.append("teeth: %s moves the UPPER teeth %.2f mm (they are fixed to the skull)" % (k, mag[up].max() * 1e3))
            if rig_err > 1e-4:
                fails.append("teeth: %s bends the lower teeth (pairwise distance error %.2f mm)" % (k, rig_err * 1e3))
        if teeth_rep.get("jawOpen", {}).get("lower_mm", 0) < 3:
            fails.append("teeth: jawOpen does not move the lower teeth")
    rep = dict(file=os.path.basename(path), fails=fails, blink=blink, wink=wink, teeth=teeth_rep, teeth_cust=cust_rep,
               keys=rows, bury=bury)
    return rep


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    jout = sys.argv[sys.argv.index("--json") + 1] if "--json" in sys.argv else None
    if jout:
        args = [a for a in args if a != jout]
    verbose = "-v" in sys.argv
    reps, bad = [], False
    for a in args:
        r = check(a, verbose)
        reps.append(r)
        print("MORPH %s: %d keys on the body, %d failures" % (r["file"], len(r["keys"]), len(r["fails"])))
        for k, b in r["blink"].items():
            print("   %-13s iris visible: rest %.0f%%  w=0.5 %.0f%%  w=1 %.1f%%  other eye %.0f%%" % (
                k, b["rest"] * 100, b["at_half"] * 100, b["at_1"] * 100, b["other_eye"] * 100))
        for k, b in r["wink"].items():
            print("   %-13s winking eye %.1f%% open, other eye %.0f%% open" % (k, b["closed_eye"] * 100, b["open_eye"] * 100))
        if r["teeth"]:
            print("   teeth: upper set static in all %d keys, lower set rigid (max pairwise error %.3f mm), jawOpen %.1f mm" % (
                len(r["teeth"]), max(v["rigid_err_mm"] for v in r["teeth"].values()), r["teeth"]["jawOpen"]["lower_mm"]))
        if r["teeth_cust"]:
            print("   teeth: %d customisation morphs move the sets affinely (max residual %.3f mm, max move %.1f mm)" % (
                len(r["teeth_cust"]), max(v["affine_err_mm"] for v in r["teeth_cust"].values()),
                max(max(v["upper_mm"], v["lower_mm"]) for v in r["teeth_cust"].values())))
        if r["bury"]:
            print("   enamel vertices inside lips/cheeks per preset (info):", ", ".join("%s %d" % kv for kv in r["bury"].items()))
        if verbose:
            for k, row in r["keys"].items():
                print("   %-20s %s" % (k, "  ".join("%s %.1fmm@x%+.0f" % (m, v["max_mm"], v["cx_mm"]) for m, v in row.items())))
            for k, v in r["teeth"].items():
                print("   teeth %-14s upper %.2f mm lower %.2f mm" % (k, v["upper_mm"], v["lower_mm"]))
        for f in r["fails"]:
            print("   FAIL", f); bad = True
    if jout:
        json.dump(reps, open(jout, "w"), indent=1)
    sys.exit(1 if bad else 0)
