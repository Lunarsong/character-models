"""Minimal GLB rewriter (pure python + numpy): replace morph-target NORMAL deltas of one mesh and re-pack the binary
chunk (unreferenced buffer views are dropped, every view stays 4-byte aligned).

Why: Blender's glTF exporter derives a shape key's morph normals from the shape key's own rest-space geometry. For a
pose-space corrective that geometry is meaningless (the delta was solved through the inverse skinning of an extreme
pose), so engines that skin the morphed normal (three.js, Unreal, Unity, Godot) would light the corrected skin with
nonsense normals. correctives.py solves the normal deltas that make the *skinned* normal match the target surface;
this module writes them into the GLB.
"""
import json, struct
import numpy as np

F32, U32 = 5126, 5125


def read_glb(path):
    b = open(path, "rb").read()
    assert b[:4] == b"glTF"
    off = 12; js = None; bin_ = b""
    while off < len(b):
        ln, typ = struct.unpack_from("<II", b, off); off += 8
        if typ == 0x4E4F534A:
            js = json.loads(b[off:off + ln])
        elif typ == 0x004E4942:
            bin_ = b[off:off + ln]
        off += ln
    return js, bytearray(bin_)


def write_glb(path, js, bin_):
    jb = json.dumps(js, separators=(",", ":")).encode()
    jb += b" " * ((4 - len(jb) % 4) % 4)
    bb = bytes(bin_) + b"\0" * ((4 - len(bin_) % 4) % 4)
    total = 12 + 8 + len(jb) + 8 + len(bb)
    with open(path, "wb") as f:
        f.write(struct.pack("<III", 0x46546C67, 2, total))
        f.write(struct.pack("<II", len(jb), 0x4E4F534A)); f.write(jb)
        f.write(struct.pack("<II", len(bb), 0x004E4942)); f.write(bb)


def _append(js, bin_, data, target=None):
    while len(bin_) % 4:
        bin_.append(0)
    bv = {"buffer": 0, "byteOffset": len(bin_), "byteLength": len(data)}
    if target:
        bv["target"] = target
    bin_ += data
    js["bufferViews"].append(bv)
    return len(js["bufferViews"]) - 1


def repack(js, bin_):
    """Drop buffer views nothing references and rebuild the binary chunk."""
    used = set()
    for a in js.get("accessors", []):
        if "bufferView" in a:
            used.add(a["bufferView"])
        if "sparse" in a:
            used.add(a["sparse"]["indices"]["bufferView"]); used.add(a["sparse"]["values"]["bufferView"])
    for im in js.get("images", []):
        if "bufferView" in im:
            used.add(im["bufferView"])
    new_bin = bytearray(); remap = {}; views = []
    for i, bv in enumerate(js["bufferViews"]):
        if i not in used:
            continue
        while len(new_bin) % 4:
            new_bin.append(0)
        start = bv.get("byteOffset", 0)
        data = bin_[start:start + bv["byteLength"]]
        nb = dict(bv); nb["byteOffset"] = len(new_bin)
        new_bin += data
        remap[i] = len(views); views.append(nb)
    js["bufferViews"] = views
    for a in js.get("accessors", []):
        if "bufferView" in a:
            a["bufferView"] = remap[a["bufferView"]]
        if "sparse" in a:
            a["sparse"]["indices"]["bufferView"] = remap[a["sparse"]["indices"]["bufferView"]]
            a["sparse"]["values"]["bufferView"] = remap[a["sparse"]["values"]["bufferView"]]
    for im in js.get("images", []):
        if "bufferView" in im:
            im["bufferView"] = remap[im["bufferView"]]
    js["buffers"][0]["byteLength"] = len(new_bin)
    return new_bin


def accessor_array(js, bin_, i):
    """Dense float/int array of accessor i (handles sparse)."""
    CT = {5120: np.int8, 5121: np.uint8, 5122: np.int16, 5123: np.uint16, 5125: np.uint32, 5126: np.float32}
    NC = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}
    a = js["accessors"][i]; dt = CT[a["componentType"]]; nc = NC[a["type"]]
    out = np.zeros((a["count"], nc), dtype=dt)
    if "bufferView" in a:
        bv = js["bufferViews"][a["bufferView"]]
        start = bv.get("byteOffset", 0) + a.get("byteOffset", 0)
        stride = bv.get("byteStride", 0) or np.dtype(dt).itemsize * nc
        raw = np.frombuffer(bytes(bin_), dtype=np.uint8, count=stride * (a["count"] - 1) + np.dtype(dt).itemsize * nc,
                            offset=start)
        out = np.lib.stride_tricks.as_strided(raw.view(dt), shape=(a["count"], nc),
                                              strides=(stride, np.dtype(dt).itemsize)).copy()
    if "sparse" in a:
        sp = a["sparse"]
        iv = js["bufferViews"][sp["indices"]["bufferView"]]; vv = js["bufferViews"][sp["values"]["bufferView"]]
        idx = np.frombuffer(bytes(bin_), dtype=CT[sp["indices"]["componentType"]], count=sp["count"],
                            offset=iv.get("byteOffset", 0) + sp["indices"].get("byteOffset", 0))
        val = np.frombuffer(bytes(bin_), dtype=dt, count=sp["count"] * nc,
                            offset=vv.get("byteOffset", 0) + sp["values"].get("byteOffset", 0)).reshape(-1, nc)
        out[idx] = val
    return out


def set_morph_normals(path, mesh_name, deltas, eps=1e-6):
    """deltas: {target name: (n_gltf_verts, 3) float array of NORMAL deltas in glTF axes} for mesh `mesh_name`
    (every primitive of the mesh must share the same vertex set, as Blender writes a single-material body).
    Each listed target's NORMAL becomes a sparse accessor holding the non-zero rows."""
    js, bin_ = read_glb(path)
    mesh = [m for m in js["meshes"] if m["name"] == mesh_name][0]
    names = mesh["extras"]["targetNames"]
    for prim in mesh["primitives"]:
        n = js["accessors"][prim["attributes"]["POSITION"]]["count"]
        for tname, d in deltas.items():
            d = np.asarray(d, dtype=np.float32)
            assert d.shape == (n, 3), (tname, d.shape, n)
            t = prim["targets"][names.index(tname)]
            nz = np.nonzero(np.abs(d).max(1) > eps)[0].astype(np.uint32)
            acc = {"componentType": F32, "count": int(n), "type": "VEC3"}
            if len(nz):
                vals = d[nz]
                lo, hi = vals.min(0), vals.max(0)
                if len(nz) < n:                  # the rows a sparse accessor does not list are zeros: bounds include 0
                    lo, hi = np.minimum(lo, 0.0), np.maximum(hi, 0.0)
                acc["min"] = [float(v) for v in lo]; acc["max"] = [float(v) for v in hi]
                acc["sparse"] = {"count": int(len(nz)),
                                 "indices": {"bufferView": _append(js, bin_, nz.tobytes()), "componentType": U32},
                                 "values": {"bufferView": _append(js, bin_, vals.tobytes())}}
            else:
                acc["min"] = [0.0, 0.0, 0.0]; acc["max"] = [0.0, 0.0, 0.0]
                acc["sparse"] = {"count": 1,
                                 "indices": {"bufferView": _append(js, bin_, np.zeros(1, np.uint32).tobytes()), "componentType": U32},
                                 "values": {"bufferView": _append(js, bin_, np.zeros(3, np.float32).tobytes())}}
            if "NORMAL" in t:
                js["accessors"][t["NORMAL"]] = acc
            else:
                js["accessors"].append(acc); t["NORMAL"] = len(js["accessors"]) - 1
    bin_ = repack(js, bin_)
    write_glb(path, js, bin_)
