"""Customisation QA renders on the ENGINE mesh (out/base_<kind>_export.blend: baked body, cust_* + face shape keys,
alternate parts present as hidden objects). Eevee, real materials.

  sheet  every cust slider at -1 and +1 (face sliders: 3/4 face close-up; body sliders: full body) ->
         renders/cust_<kind>_<nn>_<slider>_{neg,pos}.png   (+ cust_<kind>_00_base.png)
  grid   6 random characters (seeded: sliders, skin tone, eye colour, hair style / colour, brows, lashes, beard) x
         [neutral + 4 expressions] -> renders/custgrid_<kind>_<seed>_<expr>.png; the ARKit keys are added on top of
         the cust keys exactly as an engine does (additive deltas)
  The random recipe is written to renders/custgrid_<kind>_recipes.json (same schema as the viewer's 'Copy recipe').

run: Blender -b out/base_<kind>_export.blend --python-exit-code 1 -P scripts/cust_qa.py -- <kind> [sheet] [grid]
then: python3 scripts/sheet.py cust_<kind> 8 300 ; python3 scripts/cust_grid_sheet.py <kind>
"""
import sys, os, json, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cust_lib import *

args = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
kind = args[0] if args else "male"
modes = [a for a in args[1:] if a in ("sheet", "grid")] or ["sheet", "grid"]
tk = PRESETS[kind].get("tex_kind", kind)
rig = bpy.data.objects["rts_" + kind]
meshes = children_meshes(rig)
keyed = [o for o in meshes if o.data.shape_keys]
body = [o for o in meshes if o.name.endswith("_body")][0]
SL = sliders_for(kind)
# the same presets as the viewer and the face QA renders (scripts/expressions.json; '_' keys are viewer playback hints)
_PRE = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "expressions.json")))["presets"]
EXPR = {tag: {k: v for k, v in _PRE[name].items() if not k.startswith("_")}
        for tag, name in (("neutral", "Neutral"), ("smile", "Smile"), ("angry", "Angry"), ("surprise", "Surprised"),
                          ("battlecry", "Battle cry"))}


def set_weights(w):
    for o in keyed:
        for k in o.data.shape_keys.key_blocks[1:]:
            k.value = w.get(k.name, 0.0)


def slider_weights(vals):
    w = {}
    for sid, grp, lab, neg, pos, rnd in SL:
        v = vals.get(sid, 0.0)
        if v > 0:
            w[CUST + sid + "_pos"] = v
        elif v < 0 and neg is not None:
            w[CUST + sid + "_neg"] = -v
    return w


eye = (rig.pose.bones["eye_l"].head + rig.pose.bones["eye_r"].head) / 2
face_t = Vector((0, eye.y, eye.z - 0.045))
render_setup("BLENDER_EEVEE", (360, 420), 24, world=0.5)
studio_lights(0.25)


def show_parts(sel):
    """sel: {role: style or 'none'}; default parts are the objects without rts_alt."""
    for o in meshes:
        role = o.get("rts_part")
        if role not in ("hair", "beard", "eyebrows", "eyelashes"):
            continue
        want = sel.get(role)
        if want is None:
            vis = not o.get("rts_alt")
        else:
            vis = o.get("rts_style") == want
        o.hide_render = not vis


# ------------------------------------------------------------------------------------------------ material swaps
skin_mat = body.data.materials[0]
skin_node = [n for n in skin_mat.node_tree.nodes if n.type == 'TEX_IMAGE' and n.image and "skin_base" in n.image.name][0]
skin_default = skin_node.image
bsdf = skin_mat.node_tree.nodes["Principled BSDF"] if "Principled BSDF" in skin_mat.node_tree.nodes else \
    [n for n in skin_mat.node_tree.nodes if n.type == 'BSDF_PRINCIPLED'][0]
tint = skin_mat.node_tree.nodes.new("ShaderNodeMix"); tint.data_type = 'RGBA'; tint.blend_type = 'MULTIPLY'
tint.inputs["Factor"].default_value = 1.0; tint.inputs[7].default_value = (1, 1, 1, 1)
skin_mat.node_tree.links.new(skin_node.outputs["Color"], tint.inputs[6])
skin_mat.node_tree.links.new(tint.outputs[2], bsdf.inputs["Base Color"])


def set_skin(tone):
    tint.inputs[7].default_value = (1, 1, 1, 1)
    skin_node.image = skin_default
    for t, f in SKIN_TINTS:
        if t == tone:
            tint.inputs[7].default_value = (*f, 1)
            return
    if tone != "fair":
        skin_node.image = bpy.data.images.load(os.path.join(TEX, "%s_skin_%s_base.jpg" % (tk, tone)), check_existing=True)


eyes = [o for o in meshes if o.name.endswith("_eyes")][0]
eye_node = [n for n in eyes.data.materials[0].node_tree.nodes if n.type == 'TEX_IMAGE'][0]
eye_default = eye_node.image


def set_eyes(c):
    eye_node.image = eye_default if c == PRESETS[kind]["eye_color"] else \
        bpy.data.images.load(os.path.join(PARTS_TEX, "eye_%s.jpg" % c), check_existing=True)


# default hair: switch its coloured atlas to the neutral atlas + a multiply tint (what the hair:<colour> variants do)
hair_def = [o for o in meshes if o.name == kind + "_hair"][0]
hm = hair_def.data.materials[0]
hnode = [n for n in hm.node_tree.nodes if n.type == 'TEX_IMAGE' and n.outputs["Alpha"].is_linked][0]
hbsdf = [n for n in hm.node_tree.nodes if n.type == 'BSDF_PRINCIPLED'][0]
htint = hm.node_tree.nodes.new("ShaderNodeMix"); htint.data_type = 'RGBA'; htint.blend_type = 'MULTIPLY'
htint.inputs["Factor"].default_value = 1.0
hm.node_tree.links.new(hnode.outputs["Color"], htint.inputs[6]); hm.node_tree.links.new(htint.outputs[2], hbsdf.inputs["Base Color"])
hdef_img = hnode.image
dstyle = PRESETS[kind]["parts"]["hair"].split("/")[0]
hneutral = bpy.data.images.load(os.path.join(TEX, "hair_strands_neutral_base.png") if dstyle.startswith("rts_") else
                                os.path.join(PARTS_TEX, "hair_%s.png" % dstyle), check_existing=True)
colours = dict(HAIR_COLOURS)


def set_hair_colour(cname):
    rgb = colours[cname]
    f = hair_factor(rgb)
    if tuple(rgb) == tuple(PRESETS[kind]["hair_rgb"]):
        hnode.image = hdef_img; htint.inputs[7].default_value = (1, 1, 1, 1)
    else:
        hnode.image = hneutral; htint.inputs[7].default_value = (*f, 1)
    for o in meshes:
        if not o.get("rts_alt") or not o.data.materials:
            continue
        for mat in o.data.materials:                 # card beards: cards + shadow shell
            nt = mat.node_tree
            if o.get("rts_part") in ("hair", "beard"):
                for n in nt.nodes:
                    if n.type == 'MIX' and n.blend_type == 'MULTIPLY':
                        n.inputs[7].default_value = (*f, 1)
            elif o.get("rts_part") == "eyebrows":
                for n in nt.nodes:
                    if n.type == 'RGB':
                        n.outputs[0].default_value = (*brow_factor(rgb), 1)
    for n in [o for o in meshes if o.name == kind + "_eyebrows"][0].data.materials[0].node_tree.nodes:
        if n.type == 'RGB':
            n.outputs[0].default_value = (*brow_factor(rgb), 1)


def styles(role):
    return sorted({o.get("rts_style") for o in meshes if o.get("rts_part") == role})


def random_recipe(seed):
    rng = random.Random(seed)
    vals = {}
    for sid, grp, lab, neg, pos, rnd in SL:
        vals[sid] = round(rng.uniform(-rnd, rnd), 2) if rng.random() < 0.8 else 0.0
    if rng.random() < 0.2:
        vals["ears_pointed"] = round(rng.uniform(0.6, 1.0), 2)
    hs = styles("hair") + ["none"]
    beards = styles("beard")
    rec = {"sliders": vals,
           "parts": {"hair": rng.choice(hs), "eyebrows": rng.choice(styles("eyebrows")),
                     "eyelashes": rng.choice(styles("eyelashes")),
                     "beard": (rng.choice(beards) if beards and rng.random() < 0.6 else "none")},
           "variants": {"skin": rng.choice([t[0] for t in SKIN_TONES[tk]] + ([t for t, _ in SKIN_TINTS] if rng.random() < 0.3 else [])),
                        "eyes": rng.choice(EYE_COLOURS), "hair": rng.choice([n for n, _ in HAIR_COLOURS])}}
    return rec


def apply_recipe(rec):
    show_parts({r: s for r, s in rec["parts"].items()})
    set_skin(rec["variants"]["skin"]); set_eyes(rec["variants"]["eyes"]); set_hair_colour(rec["variants"]["hair"])


if "sheet" in modes:
    show_parts({})
    for o in meshes:
        if o.get("rts_part") == "hair":
            o.hide_render = o.get("rts_alt") or False
    def shots(tag):
        render(os.path.join(REN, "cust_%s_%s.png" % (kind, tag)))
    H = max(v.co.z for v in body.data.vertices)
    set_weights({})
    camera(face_t + Vector((-0.34, -0.5, 0.03)), face_t, lens=85); shots("00_base_face")
    bpy.context.scene.render.resolution_x, bpy.context.scene.render.resolution_y = 300, 520
    camera((-1.6, -3.4, H * 0.55), (0, 0, H * 0.5), lens=55); shots("00_base_body")
    for i, (sid, grp, lab, neg, pos, rnd) in enumerate(SL):
        isbody = grp == "Body"
        if isbody:
            bpy.context.scene.render.resolution_x, bpy.context.scene.render.resolution_y = 300, 520
            camera((-1.6, -3.4, H * 0.55), (0, 0, H * 0.5), lens=55)
        else:
            bpy.context.scene.render.resolution_x, bpy.context.scene.render.resolution_y = 360, 420
            camera(face_t + Vector((-0.34, -0.5, 0.03)), face_t, lens=85)
        for side, s in (("neg", neg), ("pos", pos)):
            if s is None:
                continue
            set_weights({CUST + sid + "_" + side: 1.0})
            shots("%02d_%s_%s" % (i + 1, sid, side))
    set_weights({})

if "grid" in modes:
    bpy.context.scene.render.resolution_x, bpy.context.scene.render.resolution_y = 330, 400
    camera(face_t + Vector((-0.3, -0.62, 0.04)), face_t + Vector((0, 0, 0.01)), lens=80)
    recipes = {}
    for seed in range(1, 7):
        rec = random_recipe(1000 * (2 if kind != "male" else 1) + seed)
        recipes[seed] = rec
        apply_recipe(rec)
        base = slider_weights(rec["sliders"])
        for ename, ew in EXPR.items():
            w = dict(base); w.update(ew)
            set_weights(w)
            render(os.path.join(REN, "custgrid_%s_%d_%s.png" % (kind, seed, ename)))
    json.dump(recipes, open(os.path.join(REN, "custgrid_%s_recipes.json" % kind), "w"), indent=1)
    set_weights({})
