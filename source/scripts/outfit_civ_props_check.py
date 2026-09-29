#!/usr/bin/env python3
"""Prop socket contract check for the archer's longbow (pure Python + numpy, no Blender):
the stand-alone prop out/civ/props/longbow.glb has its own skeleton whose root joint bow_root is the attach point.
Attach it the engine way (bow_root placed at the dressed character's socket_hand_l with an identity local transform,
the prop's other joints keep their local transforms), skin the prop's meshes at bind and compare with the bow / arrow
meshes inside the dressed archer GLB (nearest-vertex distance). Also checks that every civ socket / prop joint has
the same local rest rotation in the male and female GLBs (M15: canonical frames, body-independent).

usage: python3 scripts/outfit_civ_props_check.py [male female]     exit 1 if an error exceeds 0.5 mm / 0.05 deg
"""
import json, math, os, struct, sys
import numpy as np

CH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CT = {5120: np.int8, 5121: np.uint8, 5122: np.int16, 5123: np.uint16, 5125: np.uint32, 5126: np.float32}
NC = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}


def load(path):
    b = open(path, "rb").read()
    off, js, binc = 12, None, b""
    while off < len(b):
        ln, typ = struct.unpack_from("<II", b, off); off += 8
        if typ == 0x4E4F534A:
            js = json.loads(b[off:off + ln])
        elif typ == 0x004E4942:
            binc = b[off:off + ln]
        off += ln
    return js, binc


def acc(js, binc, i):
    a = js["accessors"][i]
    dt = CT[a["componentType"]]; nc = NC[a["type"]]
    bv = js["bufferViews"][a["bufferView"]]
    st = bv.get("byteStride", 0) or np.dtype(dt).itemsize * nc
    start = bv.get("byteOffset", 0) + a.get("byteOffset", 0)
    raw = np.frombuffer(binc, dtype=np.uint8, count=st * (a["count"] - 1) + np.dtype(dt).itemsize * nc, offset=start)
    out = np.lib.stride_tricks.as_strided(raw.view(dt), shape=(a["count"], nc), strides=(st, np.dtype(dt).itemsize)).copy()
    return out.astype(np.float64) if dt == np.float32 else out


def trs(n):
    M = np.eye(4)
    if "matrix" in n:
        return np.array(n["matrix"]).reshape(4, 4).T
    x, y, z, w = n.get("rotation", [0, 0, 0, 1])
    R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                  [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                  [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
    M[:3, :3] = R * np.array(n.get("scale", [1, 1, 1]))
    M[:3, 3] = n.get("translation", [0, 0, 0])
    return M


def globals_(js):
    par = {}
    for i, n in enumerate(js["nodes"]):
        for c in n.get("children", []):
            par[c] = i
    G = {}

    def g(i):
        if i in G:
            return G[i]
        M = trs(js["nodes"][i])
        if i in par:
            M = g(par[i]) @ M
        G[i] = M
        return M
    for i in range(len(js["nodes"])):
        g(i)
    return G, par


def skinned_positions(js, binc, node_idx, Gj):
    n = js["nodes"][node_idx]
    sk = js["skins"][n["skin"]]
    ibm = acc(js, binc, sk["inverseBindMatrices"]).reshape(-1, 4, 4).transpose(0, 2, 1)
    out = []
    for p in js["meshes"][n["mesh"]]["primitives"]:
        P = acc(js, binc, p["attributes"]["POSITION"])
        J = acc(js, binc, p["attributes"]["JOINTS_0"]).astype(int)
        W = acc(js, binc, p["attributes"]["WEIGHTS_0"])
        if W.dtype != np.float64:
            W = W.astype(np.float64)
        Ms = np.array([Gj(sk["joints"][k]) @ ibm[k] for k in range(len(sk["joints"]))])
        Ph = np.c_[P, np.ones(len(P))]
        V = np.zeros((len(P), 4))
        for c in range(4):
            V += W[:, c:c + 1] * np.einsum("nij,nj->ni", Ms[J[:, c]], Ph)
        out.append(V[:, :3])
    return np.concatenate(out)


def nearest(A, B):
    """max over A of the distance to the nearest point of B (brute force in chunks)"""
    worst = 0.0
    for i in range(0, len(A), 256):
        d = np.sqrt(((A[i:i + 256, None, :] - B[None, :, :]) ** 2).sum(-1)).min(1)
        worst = max(worst, float(d.max()))
    return worst


def check_kind(kind, prop):
    js, binc = load(os.path.join(CH, "out", "civ", "archer_%s.glb" % kind))
    G, par = globals_(js)
    name = {n.get("name"): i for i, n in enumerate(js["nodes"])}
    sock = G[name["socket_hand_l"]]
    pj, pb = prop
    pname = {n.get("name"): i for i, n in enumerate(pj["nodes"])}
    ppar = {}
    for i, n in enumerate(pj["nodes"]):
        for c in n.get("children", []):
            ppar[c] = i
    PG = {}

    def pg(i):                                   # the prop's joints re-rooted: bow_root := socket_hand_l
        if i in PG:
            return PG[i]
        nm = pj["nodes"][i].get("name")
        if nm == "bow_root":
            M = sock.copy()
        else:
            M = pg(ppar[i]) @ trs(pj["nodes"][i])
        PG[i] = M
        return M
    res = {}
    for mesh_name, dressed in (("longbow", "%s_longbow" % kind), ("arrow", "%s_arrow" % kind)):
        A = skinned_positions(pj, pb, pname[mesh_name], pg)
        B = skinned_positions(js, binc, name[dressed], lambda i: G[i])
        res[mesh_name] = round(nearest(A, B) * 1000, 3)
    return res


def local_rot(js, nm):
    for n in js["nodes"]:
        if n.get("name") == nm:
            return n.get("rotation", [0, 0, 0, 1])
    return None


def main(argv):
    kinds = [a for a in argv if not a.startswith("-")] or ["male", "female"]
    prop = load(os.path.join(CH, "out", "civ", "props", "longbow.glb"))
    ok = True
    report = {}
    for k in kinds:
        r = check_kind(k, prop)
        report[k] = r
        print("longbow prop attached to socket_hand_l (%s): max vertex error longbow %.3f mm, arrow %.3f mm" %
              (k, r["longbow"], r["arrow"]))
        ok &= r["longbow"] < 0.5 and r["arrow"] < 0.5
    if len(kinds) >= 2:
        a = load(os.path.join(CH, "out", "civ", "archer_%s.glb" % kinds[0]))[0]
        b = load(os.path.join(CH, "out", "civ", "archer_%s.glb" % kinds[1]))[0]
        worst = (0.0, "")
        for nm in ("socket_weapon_r", "socket_hand_l", "socket_back", "socket_hip_r", "bow_root", "bow_limb_u1",
                   "bow_limb_u2", "bow_limb_l1", "bow_limb_l2", "bow_string", "bow_socket_arrow", "bow_arrow"):
            qa, qb = local_rot(a, nm), local_rot(b, nm)
            if qa is None or qb is None:
                continue
            d = abs(sum(x * y for x, y in zip(qa, qb)))
            ang = math.degrees(2 * math.acos(min(1.0, d)))
            worst = max(worst, (ang, nm))
        print("civ socket / prop joints, local rest rotation %s vs %s: max %.4f deg (%s)" % (kinds[0], kinds[1], *worst))
        report["socket_rot_diff_deg"] = worst[0]
        ok &= worst[0] < 0.05
    json.dump(report, open(os.path.join(CH, "renders", "civ", "props_check.json"), "w"), indent=1)
    print("props check", "OK" if ok else "FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main(sys.argv[1:])
