"""Self-contained HTML viewer (GLB embedded as base64, three.js r147 from jsDelivr) for a character GLB.
If out/parts_<kind>.glb exists next to out/base_<kind>.glb (customisation build), it is embedded too and the viewer's
Customise tab binds those alternate parts to the base skeleton (PARTS=0 in the environment skips it). The expression
presets come from scripts/expressions.json.
usage: python3 characters/scripts/mkviewer_char.py out/base_male.glb "Base male" ["subtitle"] [out.html]  -> viewer/base_male.html
"""
import sys, os, json, base64
CH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
glb = sys.argv[1]
title = sys.argv[2] if len(sys.argv) > 2 else os.path.basename(glb)
sub = sys.argv[3] if len(sys.argv) > 3 else "MPFB / MakeHuman CC0 base human on the rts_human skeleton."
name = os.path.splitext(os.path.basename(glb))[0]
H = 1.85
VIEWS = {"Body": {"p": [1.6, 1.25, 3.6], "t": [0, 0.95, 0]},
         "Face": {"p": [0.28, 1.74, 0.95], "t": [0, 1.69, 0]},
         "Back": {"p": [-1.2, 1.3, -3.6], "t": [0, 0.95, 0]},
         "RTS": {"p": [0, 9.5, 6.5], "t": [0, 0.6, 0]}}
if "female" in name:
    VIEWS["Face"] = {"p": [0.26, 1.61, 0.9], "t": [0, 1.57, 0]}
if name not in ("base_male", "base_female"):
    # other statures (proportion variants): scale the camera heights by the body mesh's height (glTF bounds)
    import struct
    b = open(glb, "rb").read(); ln = struct.unpack_from("<I", b, 12)[0]; js = json.loads(b[20:20 + ln])
    body = [m for m in js["meshes"] if m["name"].endswith("_body")]
    if body:
        hb = js["accessors"][body[0]["primitives"][0]["attributes"]["POSITION"]]["max"][1]
        k = hb / (1.719 if "female" in name else 1.85)
        for v in VIEWS.values():
            v["p"][1] *= k; v["t"][1] *= k
h = open(os.path.join(CH, "viewer", "char_viewer_template.html")).read()
pre = json.load(open(os.path.join(CH, "scripts", "expressions.json")))["presets"]   # shared with the Blender renders
h = h.replace("__TITLE__", title).replace("__SUB__", sub).replace("__VIEWS__", json.dumps(VIEWS))
h = h.replace("__PRESETS__", json.dumps(pre))
h = h.replace("__GLB_BASE64__", base64.b64encode(open(glb, "rb").read()).decode())
parts = os.path.join(os.path.dirname(glb), os.path.basename(glb).replace("base_", "parts_", 1))
use_parts = os.environ.get("PARTS", "1") != "0" and parts != glb and os.path.exists(parts)
h = h.replace("__PARTS_BASE64__", base64.b64encode(open(parts, "rb").read()).decode() if use_parts else "")
out = sys.argv[4] if len(sys.argv) > 4 else os.path.join(CH, "viewer", name + ".html")
open(out, "w").write(h)
print("VIEWER", out, "%.1f MB" % (len(h) / 1e6))
