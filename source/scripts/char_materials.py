"""Blender-side helpers for the character texture sets made by scripts/textures_char.py (import inside Blender).

    import sys; sys.path.insert(0, "<characters>/scripts"); import char_materials as cm
    mat = cm.material("steel_worn")                  # Principled BSDF wired for a 1:1 glTF metallic-roughness export
    cm.assign(obj, mat)
    cm.set_texel_density(obj, "steel_worn")          # scale the UVs so one texture tile = its physical size
    cm.map_strip(obj, "trim_gold", "filigree_wide", along="U")   # fit an unwrapped trim strip into a trim-sheet strip

Every material:  base (sRGB)  -> Base Color (+ Alpha for RGBA sets)
                 orm (Non-Color) -> Separate Color: R -> glTF Material Output.Occlusion, G -> Roughness, B -> Metallic
                 normal (Non-Color, OpenGL) -> Normal Map (tangent space) -> Normal
The glTF exporter then writes base / metallicRoughness (the ORM image itself) / occlusion (same image, R) / normal.
Optional: `uv=` name of the UV map for all textures; `ao_uv=` + `ao_image=` for a per-piece baked AO on a 2nd UV map
(glTF occlusionTexture with texCoord 1, replaces the ORM's R); `tint=` multiplies the base colour (e.g. dye variants).
"""
import bpy, os, json, math

HERE = os.path.dirname(os.path.abspath(__file__))
CH = os.path.dirname(HERE)
TEX = os.path.join(CH, "assets", "textures")
_MAN = None


def manifest():
    global _MAN
    if _MAN is None:
        with open(os.path.join(TEX, "textures_char.json")) as f:
            _MAN = json.load(f)["sets"]
    return _MAN


def info(set_name):
    return manifest()[set_name]


def tex_path(set_name, key):
    return os.path.join(TEX, info(set_name)["files"][key])


def _img(path, colorspace):
    im = bpy.data.images.load(path, check_existing=True)
    im.colorspace_settings.name = colorspace
    if colorspace == "Non-Color":
        im.alpha_mode = 'NONE'
    return im


def gltf_output_group():
    name = "glTF Material Output"
    ng = bpy.data.node_groups.get(name)
    if ng:
        return ng
    try:
        from io_scene_gltf2.blender.com.material_helpers import create_settings_group
        return create_settings_group(name)
    except Exception:
        ng = bpy.data.node_groups.new(name, 'ShaderNodeTree')
        ng.interface.new_socket("Occlusion", socket_type="NodeSocketFloat")
        ng.interface.new_socket("Thickness", socket_type="NodeSocketFloat")
        ng.nodes.new('NodeGroupOutput'); ng.nodes.new('NodeGroupInput')
        return ng


DEFAULTS = {
    # set: principled overrides (the maps carry roughness / metallic; these are extras)
    "cloth_blue": dict(sheen=0.3, sheen_tint=(0.25, 0.38, 0.95), sheen_rough=0.45),
    "tabard_lion": dict(sheen=0.25, sheen_tint=(0.25, 0.38, 0.95), sheen_rough=0.45),
    "cape_lion": dict(sheen=0.25, sheen_tint=(0.25, 0.38, 0.95), sheen_rough=0.45),
    "horsehair_blue": dict(alpha_mode="MASK", alpha_cut=0.35, double_sided=True, aniso=0.6, aniso_rot=0.25),
    "mail_riveted": dict(),
}


def material(set_name, name=None, uv=None, ao_uv=None, ao_image=None, normal_strength=1.0, tint=None,
             alpha_mode=None, alpha_cut=None, double_sided=None, replace=True, **extra):
    """Create (or replace) a material named `name` (default 'M_<set>') from a texture set."""
    name = name or f"M_{set_name}"
    opts = dict(DEFAULTS.get(set_name, {})); opts.update(extra)
    alpha_mode = alpha_mode or opts.get("alpha_mode")
    alpha_cut = alpha_cut if alpha_cut is not None else opts.get("alpha_cut", 0.5)
    double_sided = double_sided if double_sided is not None else opts.get("double_sided", False)
    m = bpy.data.materials.get(name)
    if m and not replace:
        return m
    if m:
        bpy.data.materials.remove(m)
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    m["rts_texture_set"] = set_name
    nt = m.node_tree; N = nt.nodes; L = nt.links
    for nd in list(N):
        N.remove(nd)
    out = N.new("ShaderNodeOutputMaterial"); out.location = (600, 0)
    b = N.new("ShaderNodeBsdfPrincipled"); b.location = (250, 0)
    L.new(b.outputs[0], out.inputs[0])
    meta = info(set_name)
    files = meta["files"]

    def uvnode(mapname, y):
        if not mapname:
            return None
        u = N.new("ShaderNodeUVMap"); u.uv_map = mapname; u.location = (-1100, y)
        return u

    uvn = uvnode(uv, 0)
    if "base" in files:
        t = N.new("ShaderNodeTexImage"); t.location = (-700, 300); t.image = _img(tex_path(set_name, "base"), "sRGB")
        t.label = "base"
        if uvn:
            L.new(uvn.outputs[0], t.inputs[0])
        src = t.outputs["Color"]
        if tint:
            mx = N.new("ShaderNodeMix"); mx.data_type = 'RGBA'; mx.blend_type = 'MULTIPLY'; mx.location = (-300, 350)
            mx.inputs["Factor"].default_value = 1.0
            L.new(src, mx.inputs[6]); mx.inputs[7].default_value = (*tint, 1.0)
            src = mx.outputs[2]
        L.new(src, b.inputs["Base Color"])
        if meta.get("alpha") and alpha_mode in ("MASK", "BLEND"):
            if alpha_mode == "MASK":
                gt = N.new("ShaderNodeMath"); gt.operation = 'GREATER_THAN'; gt.location = (-300, 150)
                gt.inputs[1].default_value = alpha_cut           # exporter reads (X > c) as MASK, alphaCutoff = c
                L.new(t.outputs["Alpha"], gt.inputs[0]); L.new(gt.outputs[0], b.inputs["Alpha"])
                m.surface_render_method = 'DITHERED'
            else:
                L.new(t.outputs["Alpha"], b.inputs["Alpha"])
                m.surface_render_method = 'BLENDED'
    if "orm" in files:
        t = N.new("ShaderNodeTexImage"); t.location = (-700, -50); t.image = _img(tex_path(set_name, "orm"), "Non-Color")
        t.label = "orm"
        if uvn:
            L.new(uvn.outputs[0], t.inputs[0])
        sep = N.new("ShaderNodeSeparateColor"); sep.location = (-350, -50)
        L.new(t.outputs["Color"], sep.inputs[0])
        L.new(sep.outputs[1], b.inputs["Roughness"])
        L.new(sep.outputs[2], b.inputs["Metallic"])
        grp = N.new("ShaderNodeGroup"); grp.node_tree = gltf_output_group(); grp.location = (250, -450)
        grp.label = "glTF Material Output"
        if ao_image:
            ta = N.new("ShaderNodeTexImage"); ta.location = (-700, -700)
            ta.image = ao_image if isinstance(ao_image, bpy.types.Image) else _img(ao_image, "Non-Color")
            ua = uvnode(ao_uv, -700)
            if ua:
                L.new(ua.outputs[0], ta.inputs[0])
            sa = N.new("ShaderNodeSeparateColor"); sa.location = (-350, -700)
            L.new(ta.outputs["Color"], sa.inputs[0]); L.new(sa.outputs[0], grp.inputs["Occlusion"])
        else:
            L.new(sep.outputs[0], grp.inputs["Occlusion"])
    if "normal" in files:
        t = N.new("ShaderNodeTexImage"); t.location = (-700, -350); t.image = _img(tex_path(set_name, "normal"), "Non-Color")
        t.label = "normal"
        if uvn:
            L.new(uvn.outputs[0], t.inputs[0])
        nm = N.new("ShaderNodeNormalMap"); nm.location = (-300, -350); nm.inputs["Strength"].default_value = normal_strength
        if uv:
            nm.uv_map = uv
        L.new(t.outputs["Color"], nm.inputs["Color"]); L.new(nm.outputs[0], b.inputs["Normal"])
    if opts.get("sheen"):
        b.inputs["Sheen Weight"].default_value = opts["sheen"]
        b.inputs["Sheen Tint"].default_value = (*opts.get("sheen_tint", (1, 1, 1)), 1)
        b.inputs["Sheen Roughness"].default_value = opts.get("sheen_rough", 0.5)
    if opts.get("aniso"):
        b.inputs["Anisotropic"].default_value = opts["aniso"]
        b.inputs["Anisotropic Rotation"].default_value = opts.get("aniso_rot", 0.0)
    if opts.get("coat"):
        b.inputs["Coat Weight"].default_value = opts["coat"]
    m.use_backface_culling = not double_sided
    return m


def assign(obj, mat, slot=None):
    """Append / set a material on obj. slot=None: replace all slots with mat."""
    me = obj.data
    if slot is None:
        me.materials.clear(); me.materials.append(mat)
        for p in me.polygons:
            p.material_index = 0
        return 0
    while len(me.materials) <= slot:
        me.materials.append(None)
    me.materials[slot] = mat
    return slot


# ------------------------------------------------------------------------------------------------ UV helpers
def _uv_area_3d_area(me, uvl, polys=None):
    import bmesh
    a3 = 0.0; a2 = 0.0
    loops = uvl.data
    for p in (polys if polys is not None else me.polygons):
        a3 += p.area
        uvs = [loops[li].uv for li in p.loop_indices]
        s = 0.0
        for i in range(len(uvs)):
            x0, y0 = uvs[i]; x1, y1 = uvs[(i + 1) % len(uvs)]
            s += x0 * y1 - x1 * y0
        a2 += abs(s) * 0.5
    return a2, a3


def set_texel_density(obj, set_name=None, uv_per_m=None, uv=None, polys=None, keep_center=True):
    """Uniformly scale the UVs of `polys` (default: all) so that 1 m on the surface = uv_per_m UV units
    (default: the set's manifest value: one tile = its physical size). Returns the scale factor applied."""
    me = obj.data
    uvl = me.uv_layers[uv] if uv else me.uv_layers.active
    if uv_per_m is None:
        inf = info(set_name)
        uv_per_m = inf.get("uv_per_m") or inf.get("uv_per_m_along") or 1.0 / inf.get("tile_m", [1.0])[0]
    k_target = uv_per_m
    polys = list(me.polygons) if polys is None else polys
    a2, a3 = _uv_area_3d_area(me, uvl, polys)
    if a2 <= 0 or a3 <= 0:
        return 0.0
    k = k_target / math.sqrt(a2 / a3)
    idx = [li for p in polys for li in p.loop_indices]
    cx = sum(uvl.data[i].uv[0] for i in idx) / len(idx); cy = sum(uvl.data[i].uv[1] for i in idx) / len(idx)
    for i in idx:
        u, v = uvl.data[i].uv
        uvl.data[i].uv = ((u - cx) * k + (cx if keep_center else 0), (v - cy) * k + (cy if keep_center else 0))
    return k


def strip_rect(set_name, strip):
    """(v0, v1) Blender-UV range of a trim-sheet strip (V up) and its record from the manifest."""
    st = info(set_name)["strips"][strip]
    return st["v"], st


def map_strip(obj, set_name, strip, polys=None, uv=None, along="U", width_m=None, u_offset=0.0, flip=False):
    """Fit already-unwrapped strip faces into a trim-sheet strip: the UV island's short side is scaled to the strip's
    V range, the long side keeps the aspect ratio (so the pattern is not stretched) and runs along U (tiles freely).
    width_m: physical width of the trim (default: measured from the mesh); only used for reporting."""
    me = obj.data
    uvl = me.uv_layers[uv] if uv else me.uv_layers.active
    polys = list(me.polygons) if polys is None else polys
    idx = [li for p in polys for li in p.loop_indices]
    us = [uvl.data[i].uv[0] for i in idx]; vs = [uvl.data[i].uv[1] for i in idx]
    u0, u1, v0, v1 = min(us), max(us), min(vs), max(vs)
    du, dv = u1 - u0, v1 - v0
    swap = (along == "U" and dv > du) or (along == "V" and du > dv)
    (sv0, sv1), st = strip_rect(set_name, strip)
    across = du if swap else dv
    k = (sv1 - sv0) / max(across, 1e-9)
    for i in idx:
        u, v = uvl.data[i].uv
        a, c = ((v - v0), (u - u0)) if swap else ((u - u0), (v - v0))
        if flip:
            c = across - c
        uvl.data[i].uv = (a * k + u_offset, sv0 + c * k)
    return k
