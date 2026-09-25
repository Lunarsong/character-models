#!/usr/bin/env python3
"""Web copy of a GLB for the HTML viewers: every embedded image downscaled to --max px (default 1024); base-colour
images (used as baseColorTexture only) re-encoded as JPEG (quality --q), data maps (normal / metallicRoughness /
occlusion / alpha) kept lossless PNG. Geometry, skins, morphs and extras are copied byte for byte.
The engine GLB stays the full-resolution one; this is only to keep the self-contained viewer pages small.

usage: python3 scripts/glb_web.py IN.glb OUT.glb [--max 1024] [--q 88]
"""
import io, json, struct, sys, argparse
from PIL import Image


def load(path):
    b = open(path, "rb").read()
    assert b[:4] == b"glTF"
    off = 12; js = binc = None
    while off < len(b):
        ln, typ = struct.unpack_from("<II", b, off); off += 8
        if typ == 0x4E4F534A:
            js = json.loads(b[off:off + ln])
        elif typ == 0x004E4942:
            binc = b[off:off + ln]
        off += ln
    return js, binc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src"); ap.add_argument("dst")
    ap.add_argument("--max", type=int, default=1024); ap.add_argument("--q", type=int, default=88)
    a = ap.parse_args()
    js, binc = load(a.src)
    # which images are base colour only (safe for JPEG) and which carry alpha
    base_only = {}
    for mat in js.get("materials", []):
        pbr = mat.get("pbrMetallicRoughness", {})
        roles = [("base", pbr.get("baseColorTexture")), ("mr", pbr.get("metallicRoughnessTexture")),
                 ("n", mat.get("normalTexture")), ("o", mat.get("occlusionTexture")), ("e", mat.get("emissiveTexture"))]
        for role, t in roles:
            if not t:
                continue
            img = js["textures"][t["index"]].get("source")
            if img is None:
                continue
            ok = role == "base" and mat.get("alphaMode", "OPAQUE") == "OPAQUE"
            base_only[img] = base_only.get(img, True) and ok
    views = js["bufferViews"]
    new_bin = bytearray(); new_views = []
    remap = {}
    img_views = {im["bufferView"]: i for i, im in enumerate(js.get("images", [])) if "bufferView" in im}
    for vi, v in enumerate(views):
        data = binc[v.get("byteOffset", 0): v.get("byteOffset", 0) + v["byteLength"]]
        if vi in img_views:
            ii = img_views[vi]
            im = Image.open(io.BytesIO(data))
            im.load()
            if max(im.size) > a.max:
                s = a.max / max(im.size)
                im = im.resize((max(1, round(im.width * s)), max(1, round(im.height * s))), Image.LANCZOS)
            buf = io.BytesIO()
            if base_only.get(ii) and im.mode in ("RGB", "RGBA", "L", "P"):
                im.convert("RGB").save(buf, "JPEG", quality=a.q, optimize=True)
                js["images"][ii]["mimeType"] = "image/jpeg"
            else:
                im.save(buf, "PNG", optimize=True)
                js["images"][ii]["mimeType"] = "image/png"
            data = buf.getvalue()
        while len(new_bin) % 4:
            new_bin.append(0)
        nv = dict(v); nv["byteOffset"] = len(new_bin); nv["byteLength"] = len(data); nv["buffer"] = 0
        new_bin += data
        new_views.append(nv)
    while len(new_bin) % 4:
        new_bin.append(0)
    js["bufferViews"] = new_views
    js["buffers"] = [{"byteLength": len(new_bin)}]
    jb = json.dumps(js, separators=(",", ":")).encode()
    while len(jb) % 4:
        jb += b" "
    total = 12 + 8 + len(jb) + 8 + len(new_bin)
    out = struct.pack("<III", 0x46546C67, 2, total) + struct.pack("<II", len(jb), 0x4E4F534A) + jb + \
        struct.pack("<II", len(new_bin), 0x004E4942) + bytes(new_bin)
    open(a.dst, "wb").write(out)
    print("glb_web: %s -> %s  %.1f MB -> %.1f MB" % (a.src, a.dst, len(open(a.src, 'rb').read()) / 1e6, len(out) / 1e6))


if __name__ == "__main__":
    main()
