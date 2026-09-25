#!/usr/bin/env python3
"""Knight GLB check + report (build_knight.py stage 'check'; system python3 + numpy + Pillow).

  1. corr_* morph targets get an all-zero NORMAL accessor (no bufferView). The correctives are rest-space deltas
     mapped through inverse skinning, so the normal deltas Blender writes for them are wrong after skinning and
     engines that add morph normals (UE, Unity, three.js) show shading lumps (found by the viewer agent). Omitting
     NORMAL is not an option (three r147 then substitutes the base normal as the delta).
  2. check_glb.py: Khronos glTF validator (0 errors required) + skin / morph limits.
  3. out/knight_report.json + a printed summary: triangles per piece and per look (game = helm, cutscene = bare
     head), bones, morphs per mesh, clips, sockets, texture memory (as stored and on the GPU: RGBA8 + mips and
     BC7 / BC5 block compression).

usage: python3 scripts/knight_report.py [male female]
"""
import io, json, os, struct, subprocess, sys
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
CH = os.path.dirname(HERE)
OUT = os.path.join(CH, "out")
sys.path.insert(0, HERE)
import check_glb


def read_glb(path):
    b = open(path, "rb").read()
    off = 12; js = binc = None
    while off < len(b):
        ln, typ = struct.unpack_from("<II", b, off); off += 8
        if typ == 0x4E4F534A: js = json.loads(b[off:off + ln])
        elif typ == 0x004E4942: binc = b[off:off + ln]
        off += ln
    return js, binc


def write_glb(path, js, binc):
    j = json.dumps(js, separators=(",", ":")).encode()
    j += b" " * (-len(j) % 4)
    binc = binc + b"\0" * (-len(binc) % 4)
    out = struct.pack("<III", 0x46546C67, 2, 12 + 8 + len(j) + 8 + len(binc))
    out += struct.pack("<II", len(j), 0x4E4F534A) + j + struct.pack("<II", len(binc), 0x004E4942) + binc
    open(path, "wb").write(out)


def zero_corr_normals(path):
    js, binc = read_glb(path)
    zero = {}
    n = 0
    for m in js.get("meshes", []):
        names = (m.get("extras") or {}).get("targetNames", [])
        for p in m["primitives"]:
            cnt = js["accessors"][p["attributes"]["POSITION"]]["count"]
            for i, t in enumerate(p.get("targets", [])):
                if i < len(names) and names[i].startswith("corr_") and "NORMAL" in t:
                    a = js["accessors"][t["NORMAL"]]
                    if "bufferView" not in a and "sparse" not in a:
                        continue                                  # already zero
                    if cnt not in zero:
                        js["accessors"].append({"componentType": 5126, "count": cnt, "type": "VEC3"})
                        zero[cnt] = len(js["accessors"]) - 1
                    t["NORMAL"] = zero[cnt]
                    n += 1
    if n:
        write_glb(path, js, binc)
    return n


def image_info(js, binc):
    views = js.get("bufferViews", [])
    out = []
    for i, im in enumerate(js.get("images", [])):
        bv = views[im["bufferView"]]
        data = binc[bv.get("byteOffset", 0): bv.get("byteOffset", 0) + bv["byteLength"]]
        w, h = Image.open(io.BytesIO(data)).size
        out.append(dict(name=im.get("name"), mime=im.get("mimeType"), w=w, h=h, bytes=len(data)))
    return out


def material_images(js):
    """material index -> {role: image index}"""
    res = []
    tex = js.get("textures", [])
    for m in js.get("materials", []):
        pbr = m.get("pbrMetallicRoughness", {})
        roles = {"base": pbr.get("baseColorTexture"), "mr": pbr.get("metallicRoughnessTexture"),
                 "normal": m.get("normalTexture"), "occlusion": m.get("occlusionTexture"), "emissive": m.get("emissiveTexture")}
        d = {}
        for r, t in roles.items():
            if t and "source" in tex[t["index"]]:
                d[r] = tex[t["index"]]["source"]
        for ext in (m.get("extensions") or {}).values():
            for k, v in ext.items():
                if isinstance(v, dict) and "index" in v and "source" in tex[v["index"]]:
                    d[k] = tex[v["index"]]["source"]
        res.append(d)
    return res


def gpu_bytes(w, h, role):
    raw = w * h * 4 * 4 / 3                         # RGBA8 + full mip chain
    bc = w * h * 1 * 4 / 3                          # BC7 (colour / ORM) or BC5 (normal): 1 byte per texel
    return raw, bc


def report(kind):
    path = os.path.join(OUT, "knight_%s.glb" % kind)
    nz = zero_corr_normals(path)
    chk = check_glb.check(path)
    js, binc = read_glb(path)
    imgs = image_info(js, binc)
    mats = material_images(js)
    skin = js["skins"][0]
    joints = [js["nodes"][j]["name"] for j in skin["joints"]]
    arm = next((n for n in js["nodes"] if (n.get("extras") or {}).get("rts_kind")), {})
    ex = arm.get("extras", {})
    meshes = []
    for n in js["nodes"]:
        if "mesh" not in n:
            continue
        m = js["meshes"][n["mesh"]]
        e = n.get("extras", {})
        tris = 0; mi = set()
        for p in m["primitives"]:
            tris += js["accessors"][p["indices"]]["count"] // 3
            if "material" in p:
                mi.add(p["material"])
        meshes.append(dict(name=n["name"], look=e.get("rts_look", "any"), slot=e.get("slot"), part=e.get("rts_part"),
                           variant=e.get("rts_variant_group"), default=e.get("rts_default"), tris=tris,
                           morphs=len((m.get("extras") or {}).get("targetNames", [])), materials=sorted(mi)))

    def look_meshes(look):
        out = []
        for m in meshes:
            if m["look"] not in ("any", look):
                continue
            if m["variant"] and not m["default"]:
                continue                               # alternative eyebrows: one of them is shown
            out.append(m)
        return out

    looks = {}
    for look in ("helm", "bare"):
        ms = look_meshes(look)
        used = set()
        for m in ms:
            for mi in m["materials"]:
                used |= set(mats[mi].values())
        tex_raw = sum(gpu_bytes(imgs[i]["w"], imgs[i]["h"], "")[0] for i in used)
        tex_bc = sum(gpu_bytes(imgs[i]["w"], imgs[i]["h"], "")[1] for i in used)
        looks[look] = dict(tris=sum(m["tris"] for m in ms), tris_no_props=sum(m["tris"] for m in ms if m["slot"] != "Props"),
                           meshes=len(ms), draws=sum(len(m["materials"]) for m in ms), images=len(used),
                           tex_gpu_rgba8_MB=round(tex_raw / 2 ** 20, 1), tex_gpu_bc_MB=round(tex_bc / 2 ** 20, 1))
    anims = []
    for a in js.get("animations", []):
        dur = max(js["accessors"][s["input"]]["max"][0] for s in a["samplers"])
        paths = {}
        for c in a["channels"]:
            paths[c["target"]["path"]] = paths.get(c["target"]["path"], 0) + 1
        anims.append(dict(name=a["name"], seconds=round(dur, 3), channels=paths))
    rep = dict(file=os.path.basename(path), MB=round(os.path.getsize(path) / 1e6, 1), validator=chk["validator"],
               val_errors=chk["val_errors"], problems=chk["problems"], corr_normals_zeroed=nz,
               bones=len(joints), sockets=[j for j in joints if j.startswith("socket_")],
               chains=[j for j in joints if j.startswith(("cape_", "tabard_"))], looks=looks, meshes=meshes,
               animations=anims, clips=json.loads(ex.get("rts_clips", "{}")),
               images=[dict(i, gpu_bc_MB=round(gpu_bytes(i["w"], i["h"], "")[1] / 2 ** 20, 2)) for i in imgs],
               max_influences=max((m["max_infl"] for m in chk["meshes"] if m["skinned"]), default=0),
               height_y=chk["height_y"])
    return rep


def main(kinds=None):
    kinds = kinds or [a for a in sys.argv[1:] if a in ("male", "female")] or ["male", "female"]
    reps = {}
    bad = False
    for k in kinds:
        r = report(k)
        reps[k] = r
        print("KNT %s: %s  %.1f MB  %s" % (k, r["file"], r["MB"], r["validator"]))
        print("KNT   bones %d (sockets %d, cape/tabard chain %d), max influences %d, corr normals zeroed %d"
              % (r["bones"], len(r["sockets"]), len(r["chains"]), r["max_influences"], r["corr_normals_zeroed"]))
        for look, L in r["looks"].items():
            print("KNT   look %-4s  %6d tris (%6d without props)  %2d meshes  %2d draws  %2d images  GPU %.0f MB RGBA8+mips / %.0f MB BC7-BC5"
                  % (look, L["tris"], L["tris_no_props"], L["meshes"], L["draws"], L["images"], L["tex_gpu_rgba8_MB"], L["tex_gpu_bc_MB"]))
        for a in r["animations"]:
            print("KNT   clip %-13s %5.2f s  %s" % (a["name"], a["seconds"], a["channels"]))
        for m in sorted(r["meshes"], key=lambda m: (m["look"], m["slot"] or "", m["name"])):
            print("KNT   %-5s %-26s %-6s %6d tris %4d morphs" % (m["look"], m["name"], m["slot"] or "", m["tris"], m["morphs"]))
        if r["val_errors"] != 0 or r["problems"]:
            bad = True
            print("KNT   PROBLEMS", r["problems"])
    old = {}
    p = os.path.join(OUT, "knight_report.json")
    if os.path.exists(p):
        try:
            old = json.load(open(p))
        except Exception:
            old = {}
    old.update(reps)
    json.dump(old, open(p, "w"), indent=1)
    print("KNT report", p)
    if bad:
        raise SystemExit("knight GLB check failed")


if __name__ == "__main__":
    main()
