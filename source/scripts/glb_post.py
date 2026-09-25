"""GLB post-process for the customisation layer (pure Python, no Blender needed; base_humans.py calls it right after
the glTF export). It adds:
  * KHR_materials_variants: one glTF variant per option, named '<category>:<option>' (e.g. 'skin:tan',
    'eyes:green', 'hair:blond'). A variant maps only the primitives of its category, so categories combine: an
    engine applies one variant per category (the viewer does). Plain KHR_materials_variants players that switch
    one variant at a time reset the other categories to the default look, which is also valid.
    Variant materials are clones of the default material with a new baseColorTexture and / or baseColorFactor;
    their textures are appended to the binary chunk (deduplicated by file).
  * scene extras 'rts_customise': slider table (cust_* morph pairs), part catalogue, variant categories, per-morph
    skeleton refit offsets (see cust_lib.bone_offsets) and notes; three.js exposes it as gltf.scene.userData.
  * optional: drops the morph-target NORMAL deltas of the meshes matched by spec["strip_morph_normals"] (a regex;
    used for hair / beard / brow / lash cards, whose shading comes from their own card normals) and compacts the
    binary chunk (unused accessors / buffer views removed).

spec = {"variants": [{"name": "skin:tan", "materials": {"M_male_skin": {"baseColorTexture": "/abs/tex.jpg",
         "baseColorFactor": [r, g, b, a]}}}, ...], "extras": {...}}
usage: python3 scripts/glb_post.py file.glb spec.json
"""
import json, struct, os, sys, copy

MIME = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png"}


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
    j = json.dumps(js, separators=(",", ":")).encode()
    j += b" " * ((4 - len(j) % 4) % 4)
    bin_ = bytes(bin_) + b"\0" * ((4 - len(bin_) % 4) % 4)
    total = 12 + 8 + len(j) + 8 + len(bin_)
    with open(path, "wb") as f:
        f.write(struct.pack("<III", 0x46546C67, 2, total))
        f.write(struct.pack("<II", len(j), 0x4E4F534A)); f.write(j)
        f.write(struct.pack("<II", len(bin_), 0x004E4942)); f.write(bin_)


def post(glb, spec):
    js, bin_ = read_glb(glb)
    js.setdefault("images", []); js.setdefault("textures", [])
    mats = js["materials"]
    by_name = {m["name"]: i for i, m in enumerate(mats)}
    img_cache, mat_cache = {}, {}

    def add_image(path):
        if path in img_cache:
            return img_cache[path]
        data = open(path, "rb").read()
        while len(bin_) % 8:
            bin_.append(0)
        off = len(bin_); bin_.extend(data)
        js["bufferViews"].append({"buffer": 0, "byteOffset": off, "byteLength": len(data)})
        js["images"].append({"name": os.path.splitext(os.path.basename(path))[0],
                             "mimeType": MIME[os.path.splitext(path)[1].lower()], "bufferView": len(js["bufferViews"]) - 1})
        img_cache[path] = len(js["images"]) - 1
        return img_cache[path]

    def variant_material(mi, ov, vname):
        if not ov:
            return mi
        key = (mi, json.dumps(ov, sort_keys=True))
        if key in mat_cache:
            return mat_cache[key]
        m = copy.deepcopy(mats[mi])
        m["name"] = "%s@%s" % (m["name"], vname)
        pbr = m.setdefault("pbrMetallicRoughness", {})
        if "baseColorTexture" in ov:
            old = pbr.get("baseColorTexture")
            tex = {"source": add_image(ov["baseColorTexture"])}
            if old is not None and "sampler" in js["textures"][old["index"]]:
                tex["sampler"] = js["textures"][old["index"]]["sampler"]
            js["textures"].append(tex)
            nt = {"index": len(js["textures"]) - 1}
            if old and "texCoord" in old:
                nt["texCoord"] = old["texCoord"]
            pbr["baseColorTexture"] = nt
        if "baseColorFactor" in ov:
            pbr["baseColorFactor"] = [float(x) for x in ov["baseColorFactor"]]
        mats.append(m)
        mat_cache[key] = len(mats) - 1
        return mat_cache[key]

    variants = []
    prim_maps = {}                               # (mesh, prim) -> {material: [variant indices]}
    for v in spec.get("variants", []):
        hits = [(mn, ov) for mn, ov in v["materials"].items() if mn in by_name]
        if not hits:
            continue
        vi = len(variants); variants.append({"name": v["name"]})
        for mn, ov in hits:
            mi = by_name[mn]
            tgt = variant_material(mi, ov, v["name"])
            for me_i, me in enumerate(js["meshes"]):
                for p_i, p in enumerate(me["primitives"]):
                    if p.get("material") == mi:
                        prim_maps.setdefault((me_i, p_i), {}).setdefault(tgt, []).append(vi)
    if variants:
        js.setdefault("extensions", {})["KHR_materials_variants"] = {"variants": variants}
        for (me_i, p_i), mp in prim_maps.items():
            p = js["meshes"][me_i]["primitives"][p_i]
            p.setdefault("extensions", {})["KHR_materials_variants"] = {
                "mappings": [{"material": m, "variants": sorted(set(vs))} for m, vs in sorted(mp.items())]}
        used = js.setdefault("extensionsUsed", [])
        if "KHR_materials_variants" not in used:
            used.append("KHR_materials_variants")
    stripped = 0
    if spec.get("strip_morph_normals"):
        import re
        rx = re.compile(spec["strip_morph_normals"])
        for me in js["meshes"]:
            if not rx.search(me.get("name", "")):
                continue
            for p in me["primitives"]:
                for t in p.get("targets", []):
                    if t.pop("NORMAL", None) is not None:
                        stripped += 1
                    t.pop("TANGENT", None)
        js, bin_ = compact(js, bin_)
    if spec.get("extras"):
        js["scenes"][js.get("scene", 0)].setdefault("extras", {})["rts_customise"] = spec["extras"]
    js["buffers"][0]["byteLength"] = len(bin_) + ((4 - len(bin_) % 4) % 4)
    write_glb(glb, js, bin_)
    return dict(variants=len(variants), materials_added=len(mat_cache), images_added=len(img_cache),
                morph_normals_stripped=stripped, MB=round(os.path.getsize(glb) / 1e6, 2))


def compact(js, bin_):
    """Drop unreferenced accessors and buffer views and repack the binary chunk (8-byte aligned views)."""
    used_acc = []
    seen = {}

    def acc(i):
        if i not in seen:
            seen[i] = len(used_acc); used_acc.append(i)
        return seen[i]
    for me in js["meshes"]:
        for p in me["primitives"]:
            p["attributes"] = {k: acc(v) for k, v in p["attributes"].items()}
            if "indices" in p:
                p["indices"] = acc(p["indices"])
            if "targets" in p:
                p["targets"] = [{k: acc(v) for k, v in t.items()} for t in p["targets"]]
    for sk in js.get("skins", []):
        if "inverseBindMatrices" in sk:
            sk["inverseBindMatrices"] = acc(sk["inverseBindMatrices"])
    for an in js.get("animations", []):
        for sm in an["samplers"]:
            sm["input"] = acc(sm["input"]); sm["output"] = acc(sm["output"])
    accessors = [js["accessors"][i] for i in used_acc]
    used_bv, bv_map = [], {}

    def bv(i):
        if i not in bv_map:
            bv_map[i] = len(used_bv); used_bv.append(i)
        return bv_map[i]
    for a in accessors:
        if "bufferView" in a:
            a["bufferView"] = bv(a["bufferView"])
        if "sparse" in a:
            a["sparse"]["indices"]["bufferView"] = bv(a["sparse"]["indices"]["bufferView"])
            a["sparse"]["values"]["bufferView"] = bv(a["sparse"]["values"]["bufferView"])
    for im in js.get("images", []):
        if "bufferView" in im:
            im["bufferView"] = bv(im["bufferView"])
    out = bytearray(); views = []
    for i in used_bv:
        v = dict(js["bufferViews"][i])
        while len(out) % 8:
            out.append(0)
        start = v.get("byteOffset", 0)
        data = bin_[start:start + v["byteLength"]]
        v["byteOffset"] = len(out); out.extend(data)
        views.append(v)
    js["accessors"] = accessors
    js["bufferViews"] = views
    return js, out


if __name__ == "__main__":
    print(post(sys.argv[1], json.load(open(sys.argv[2]))))
