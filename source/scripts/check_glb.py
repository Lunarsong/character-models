"""Inspect exported character GLBs: skeleton, skinning limits, morph targets, size, orientation, and run the Khronos
glTF validator (work/val/node_modules/gltf-validator). Exit code 1 if any file has validator errors or breaks a limit.

usage: python3 characters/scripts/check_glb.py out/base_male.glb [more.glb ...] [--json report.json]
"""
import json, struct, sys, os, subprocess
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
VALJS = os.path.join(HERE, "validate.js")
CT = {5120: np.int8, 5121: np.uint8, 5122: np.int16, 5123: np.uint16, 5125: np.uint32, 5126: np.float32}
NC = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}


def load(path):
    b = open(path, "rb").read()
    assert b[:4] == b"glTF"
    off = 12; js = None; bin_ = None
    while off < len(b):
        ln, typ = struct.unpack_from("<II", b, off); off += 8
        if typ == 0x4E4F534A: js = json.loads(b[off:off + ln])
        elif typ == 0x004E4942: bin_ = b[off:off + ln]
        off += ln
    return js, bin_


def accessor(js, bin_, i):
    a = js["accessors"][i]
    dt = CT[a["componentType"]]; nc = NC[a["type"]]
    out = np.zeros((a["count"], nc), dtype=dt)
    if "bufferView" in a:
        bv = js["bufferViews"][a["bufferView"]]
        start = bv.get("byteOffset", 0) + a.get("byteOffset", 0)
        stride = bv.get("byteStride", 0) or np.dtype(dt).itemsize * nc
        raw = np.frombuffer(bin_, dtype=np.uint8, count=stride * (a["count"] - 1) + np.dtype(dt).itemsize * nc, offset=start)
        out = np.lib.stride_tricks.as_strided(raw.view(dt), shape=(a["count"], nc),
                                              strides=(stride, np.dtype(dt).itemsize)).copy()
    if "sparse" in a:
        sp = a["sparse"]
        iv = js["bufferViews"][sp["indices"]["bufferView"]]; vv = js["bufferViews"][sp["values"]["bufferView"]]
        idx = np.frombuffer(bin_, dtype=CT[sp["indices"]["componentType"]], count=sp["count"],
                            offset=iv.get("byteOffset", 0) + sp["indices"].get("byteOffset", 0))
        val = np.frombuffer(bin_, dtype=dt, count=sp["count"] * nc,
                            offset=vv.get("byteOffset", 0) + sp["values"].get("byteOffset", 0)).reshape(-1, nc)
        out[idx] = val
    if a.get("normalized"):
        out = out.astype(np.float32) / np.iinfo(dt).max
    return out


def check(path):
    js, bin_ = load(path)
    rep = {"file": os.path.basename(path), "MB": round(os.path.getsize(path) / 1e6, 2), "problems": []}
    skins = js.get("skins", [])
    rep["skins"] = len(skins)
    joints = skins[0]["joints"] if skins else []
    rep["bones"] = len(joints)
    rep["bone_names"] = [js["nodes"][j]["name"] for j in joints]
    meshes = []
    tris_total = 0
    pos_all = []
    for ni, n in enumerate(js["nodes"]):
        if "mesh" not in n:
            continue
        m = js["meshes"][n["mesh"]]
        names = (m.get("extras") or {}).get("targetNames", [])
        info = {"node": n.get("name"), "skinned": "skin" in n, "morphs": len(names), "prims": len(m["primitives"]),
                "verts": 0, "tris": 0, "max_infl": 0, "weight_sum_err": 0.0, "materials": []}
        for p in m["primitives"]:
            at = p["attributes"]
            nv = js["accessors"][at["POSITION"]]["count"]
            info["verts"] += nv
            ntri = js["accessors"][p["indices"]]["count"] // 3 if "indices" in p else nv // 3
            info["tris"] += ntri
            pos_all.append(accessor(js, bin_, at["POSITION"]))
            if "JOINTS_1" in at:
                rep["problems"].append("%s has JOINTS_1 (>4 influences)" % n.get("name"))
            if "WEIGHTS_0" in at:
                w = accessor(js, bin_, at["WEIGHTS_0"]).astype(np.float64)
                info["max_infl"] = max(info["max_infl"], int((w > 1e-4).sum(1).max()))
                info["weight_sum_err"] = max(info["weight_sum_err"], float(np.abs(w.sum(1) - 1).max()))
                jn = accessor(js, bin_, at["JOINTS_0"])
                if jn.max() >= len(joints):
                    rep["problems"].append("%s joint index out of range" % n.get("name"))
            if "material" in p:
                info["materials"].append(js["materials"][p["material"]]["name"])
            if len(p.get("targets", [])) != len(names):
                rep["problems"].append("%s: targets %d != targetNames %d" % (n.get("name"), len(p.get("targets", [])), len(names)))
        info["morph_names"] = names
        tris_total += info["tris"]
        meshes.append(info)
        if info["skinned"] and info["max_infl"] > 4:
            rep["problems"].append("%s max influences %d" % (info["node"], info["max_infl"]))
        if info["skinned"] and info["weight_sum_err"] > 0.01:
            rep["problems"].append("%s weight sums off by %.3f" % (info["node"], info["weight_sum_err"]))
    P = np.concatenate(pos_all)
    rep["bbox_min"] = [round(float(x), 3) for x in P.min(0)]
    rep["bbox_max"] = [round(float(x), 3) for x in P.max(0)]
    rep["height_y"] = round(float(P[:, 1].max() - P[:, 1].min()), 3)
    rep["tris"] = tris_total
    rep["meshes"] = meshes
    rep["materials"] = [m["name"] for m in js.get("materials", [])]
    rep["images"] = [(im.get("name"), im.get("mimeType")) for im in js.get("images", [])]
    r = subprocess.run(["node", VALJS, path], capture_output=True, text=True)
    rep["validator"] = r.stdout.strip()
    try:
        rep["val_errors"] = int(r.stdout.split("errors")[1].split()[0])
    except Exception:
        rep["val_errors"] = -1
    if rep["val_errors"] != 0:
        rep["problems"].append("validator errors: " + r.stdout.strip()[:400])
    return rep


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    jout = sys.argv[sys.argv.index("--json") + 1] if "--json" in sys.argv else None
    if jout:
        args = [a for a in args if a != jout]
    reps = [check(a) for a in args]
    bad = False
    for r in reps:
        print("GLB %s  %.2f MB  bones %d  tris %d  height(Y) %.3f m  bbox %s..%s" % (r["file"], r["MB"], r["bones"], r["tris"],
              r["height_y"], r["bbox_min"], r["bbox_max"]))
        for m in r["meshes"]:
            print("   %-18s verts %6d tris %6d morphs %3d max_infl %d wsum_err %.4f mats %s" % (m["node"], m["verts"], m["tris"],
                  m["morphs"], m["max_infl"], m["weight_sum_err"], m["materials"]))
        print("   validator:", r["validator"].replace("\n", " | ")[:600])
        for p in r["problems"]:
            print("   PROBLEM", p); bad = True
    if jout:
        json.dump(reps, open(jout, "w"), indent=1)
    sys.exit(1 if bad else 0)
