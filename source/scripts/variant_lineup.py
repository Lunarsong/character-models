"""Line-up of the base humans and the proportion variants at true relative scale (render_look.py renders, which frame
each body proportionally to its height) + a face row. -> renders/variants_lineup.png
usage: python3 scripts/variant_lineup.py   (after: Blender -b out/base_<k>_export.blend -P scripts/render_look.py -- <k> var_<k>)
"""
import os, json, struct
from PIL import Image, ImageDraw, ImageFont

CH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REN = os.path.join(CH, "renders")
KINDS = [("male_stocky", "male_stocky (dwarf-ish)"), ("male", "male (base)"), ("male_slim", "male_slim (elf-ish)"),
         ("female", "female (base)"), ("female_slim", "female_slim")]


def body_height(k):
    b = open(os.path.join(CH, "out", "base_%s.glb" % k), "rb").read()
    js = json.loads(b[20:20 + struct.unpack_from("<I", b, 12)[0]])
    m = [m for m in js["meshes"] if m["name"].endswith("_body")][0]
    a = js["accessors"][m["primitives"][0]["attributes"]["POSITION"]]
    return a["max"][1] - a["min"][1]


font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 22)
cols = []
for k, label in KINDS:
    H = body_height(k)
    im = Image.open(os.path.join(REN, "var_%s_body_front.png" % k)).convert("RGB")
    s = H / 1.85 * 0.62
    im = im.resize((int(im.width * s), int(im.height * s)), Image.LANCZOS)
    face = Image.open(os.path.join(REN, "var_%s_face_34.png" % k)).convert("RGB").resize((450, 500), Image.LANCZOS)
    cols.append((label, H, im, face))
W = sum(max(c[2].width, c[3].width) for c in cols)
top = max(c[2].height for c in cols)
S = Image.new("RGB", (W, 40 + top + 500), (24, 24, 28))
d = ImageDraw.Draw(S)
x = 0
for label, H, im, face in cols:
    w = max(im.width, face.width)
    S.paste(im, (x + (w - im.width) // 2, 40 + top - im.height))
    S.paste(face, (x + (w - face.width) // 2, 40 + top))
    d.text((x + 10, 8), "%s  %.2f m" % (label, H), fill=(235, 235, 235), font=font)
    x += w
out = os.path.join(REN, "variants_lineup.png")
S.save(out)
print("LINEUP", out, S.size)
