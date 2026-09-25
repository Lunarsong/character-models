"""GAME LODs with merged atlases + separable CUTSCENE face data (judge M16), on the engine knight
(out/knight_<kind>_export.blend: helm + bare looks, culled, clips baked):

  game <kind>   helm look only, three skinned meshes per LOD, each on ONE material with its own atlas:
                  armour  every opaque piece (plates, mail, cloth, belts, boots, fillers) merged, Smart-UV-unwrapped
                          into a new atlas and baked from the piece materials (base colour incl. the vertex AO,
                          ORM = occlusion / roughness / metallic, tangent-space normal) at 2048 px
                  plume   the horsehair cards on their own alpha-masked 1k atlas (LOD1 keeps half the cards, LOD2 a fifth)
                  props   sword + shield merged, baked to a 1024 px atlas (swap weapons by swapping this mesh)
                LOD1 / LOD2 = collapse-decimated copies of LOD0 (same atlases, weights re-normalised to <= 4).
                Game skeleton: the face joints are removed (their weights go to `head`); body clips only.
                -> out/game/knight_<kind>_lod0.glb, _lod1.glb, _lod2.glb + out/game/knight_<kind>_lods.json
  face <kind>   the bare-head look (head skin, hair, brows, lashes, eyes, teeth, tongue) with the face joints, the 112
                face / customisation morphs and the talk_emote clip -> out/cutscene/knight_<kind>_face.glb. Attach it
                to the game skeleton by joint name (the face joints graft under `head`) for cutscenes.
run: Blender -b out/knight_<kind>_export.blend --python-exit-code 1 -P scripts/knight_lod.py -- game <kind>
"""
import bpy, os, sys, json, time, math
import numpy as np
from mathutils import Matrix

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import chr_lib as C

GAME = os.path.join(C.OUT, "game")
CUT = os.path.join(C.OUT, "cutscene")
TEXOUT = os.path.join(GAME, "textures")
FACE_BONES = ("jaw", "eye_", "eyelid_", "brow_", "cheek_", "nose_", "mouth_corner_", "lip_", "tongue_")
BARE = ("head", "hair", "eyebrows", "eyelashes", "eyes", "teeth", "tongue")
LOD_TRIS = {0: 40000, 1: 12000, 2: 4000}         # total triangle targets (plan: hero LOD0 25-40k; judge M16: LOD1
#                                                  ~12k, LOD2 ~4k). LOD0 is only decimated when the merged helm look
#                                                  is over its budget (the armour pieces grew in iteration 2)
ATLAS = {"armour": 2048, "props": 1024}


def log(*a):
    print("LOD", *a, flush=True)


def tris(o):
    return sum(len(p.vertices) - 2 for p in o.data.polygons)


def select_only(objs, active=None):
    for o in list(bpy.context.scene.objects):
        o.select_set(o in objs)
    bpy.context.view_layer.objects.active = active or (objs[0] if objs else None)


def dup(o, name):
    c = o.copy(); c.data = o.data.copy(); c.name = c.data.name = name
    bpy.context.scene.collection.objects.link(c)
    if c.data.shape_keys:
        c.shape_key_clear()
    c.animation_data_clear()
    return c


def merge(objs, name, rig):
    """copies of objs joined into one mesh (shape keys dropped: a baked game unit; weights / UVs / materials kept)"""
    cs = [dup(o, "%s_tmp_%d" % (name, i)) for i, o in enumerate(objs)]
    for c in cs:
        for u in list(c.data.uv_layers)[1:]:
            c.data.uv_layers.remove(u)
        c.data.uv_layers[0].name = "UVMap"
        c.parent = None; c.matrix_world = Matrix.Identity(4)
    select_only(cs, cs[0])
    bpy.ops.object.join()
    m = bpy.context.view_layer.objects.active
    m.name = m.data.name = name
    m.parent = rig
    m.matrix_parent_inverse = Matrix.Identity(4)
    for md in list(m.modifiers):
        if md.type != 'ARMATURE':
            m.modifiers.remove(md)
    arms = [md for md in m.modifiers if md.type == 'ARMATURE']
    for md in arms[1:]:
        m.modifiers.remove(md)
    if not arms:
        md = m.modifiers.new("Armature", 'ARMATURE'); md.object = rig
    return m


# ------------------------------------------------------------------------------------------------------ baking
def _socket_source(nt, sock):
    """(from_socket or None, default value) feeding `sock`"""
    if sock.is_linked:
        return sock.links[0].from_socket, None
    v = sock.default_value
    return None, (tuple(v) if hasattr(v, "__len__") else v)


def _bsdf(nt):
    return next((n for n in nt.nodes if n.type == 'BSDF_PRINCIPLED'), None)


def _occlusion_socket(nt):
    for n in nt.nodes:
        if n.type == 'GROUP' and n.node_tree and "Occlusion" in n.inputs:
            return n.inputs["Occlusion"]
    return None


def bake_atlas(m, size, tag, kind):
    """Smart-UV atlas for merged mesh m + Cycles bakes of its materials -> one glTF material (returns it)"""
    me = m.data
    src_uv = me.uv_layers["UVMap"]
    # the atlas starts as a copy of the pieces' own UVs: their islands are the authored unwraps (plates along their
    # arc length, trims as strips, cloth panels whole); every island is rescaled to its true surface area (uniform
    # texel density; linings that sampled a small patch get their real size) and packed without overlaps
    at = me.uv_layers.new(name="atlas", do_init=True)
    me.uv_layers.active = at
    src_uv.active_render = True
    select_only([m], m)
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    ts = bpy.context.scene.tool_settings
    ts.use_uv_select_sync = True
    bpy.ops.uv.select_all(action='SELECT')
    bpy.ops.uv.average_islands_scale()
    bpy.ops.uv.pack_islands(margin=0.002, rotate=True)
    bpy.ops.object.mode_set(mode='OBJECT')
    sc = bpy.context.scene
    sc.render.engine = 'CYCLES'
    sc.cycles.device = 'CPU'
    sc.cycles.samples = 4
    sc.render.bake.margin = 6
    sc.render.bake.use_clear = True
    sc.render.bake.target = 'IMAGE_TEXTURES'
    mats = [s.material for s in m.material_slots if s.material]
    # work on material copies (the piece materials stay untouched for the other exports)
    copies = {}
    for i, s in enumerate(m.material_slots):
        if s.material:
            if s.material.name not in copies:
                copies[s.material.name] = s.material.copy()
            s.material = copies[s.material.name]
    mats = list(copies.values())
    # a single-sided source (plates) and a double-sided one (cloth panels, the shield face, cards) end up on one atlas
    # material: it must be double-sided if any source was, else those surfaces vanish from behind (LOD shield face)
    two_sided = any(not mt.use_backface_culling for mt in mats)
    os.makedirs(TEXOUT, exist_ok=True)
    imgs = {}

    def target(name, colorspace):
        im = bpy.data.images.new("%s_%s_%s" % (tag, kind, name), size, size, alpha=False, float_buffer=False)
        im.colorspace_settings.name = colorspace
        for mt in mats:
            nt = mt.node_tree
            n = nt.nodes.new("ShaderNodeTexImage"); n.image = im; n.name = "rts_bake_target"
            n.select = True; nt.nodes.active = n
        return im

    def clear_targets():
        for mt in mats:
            for n in [n for n in mt.node_tree.nodes if n.name.startswith("rts_bake_target")]:
                mt.node_tree.nodes.remove(n)

    def emit_bake(name, pick, colorspace):
        """rewire every material: its value for `pick` -> an Emission shader -> bake EMIT"""
        saved = []
        for mt in mats:
            nt = mt.node_tree
            out = next((n for n in nt.nodes if n.type == 'OUTPUT_MATERIAL' and n.is_active_output), None) or \
                next((n for n in nt.nodes if n.type == 'OUTPUT_MATERIAL'), None)
            if out is None:
                continue
            old = out.inputs["Surface"].links[0].from_socket if out.inputs["Surface"].is_linked else None
            em = nt.nodes.new("ShaderNodeEmission"); em.name = "rts_bake_emit"
            em.inputs["Strength"].default_value = 1.0
            src, val = pick(nt)
            if src is not None:
                nt.links.new(src, em.inputs["Color"])
            else:
                v = val if isinstance(val, tuple) else (val, val, val)
                em.inputs["Color"].default_value = (v[0], v[1], v[2], 1.0)
            nt.links.new(em.outputs[0], out.inputs["Surface"])
            saved.append((nt, out, old, em))
        im = target(name, colorspace)
        t0 = time.time()
        bpy.ops.object.bake(type='EMIT', uv_layer="atlas", margin=6)
        log("bake %-8s %-6s %dpx %.0fs" % (tag, name, size, time.time() - t0))
        clear_targets()
        for nt, out, old, em in saved:
            nt.nodes.remove(em)
            if old is not None:
                nt.links.new(old, out.inputs["Surface"])
        return im

    def base_pick(nt):
        b = _bsdf(nt)
        return _socket_source(nt, b.inputs["Base Color"]) if b else (None, (0.5, 0.5, 0.5))

    def rough_pick(nt):
        b = _bsdf(nt)
        return _socket_source(nt, b.inputs["Roughness"]) if b else (None, 0.6)

    def metal_pick(nt):
        b = _bsdf(nt)
        return _socket_source(nt, b.inputs["Metallic"]) if b else (None, 0.0)

    def occ_pick(nt):
        s = _occlusion_socket(nt)
        return (s.links[0].from_socket, None) if s is not None and s.is_linked else (None, 1.0)

    imgs["base"] = emit_bake("base", base_pick, "sRGB")
    imgs["rough"] = emit_bake("rough", rough_pick, "Non-Color")
    imgs["metal"] = emit_bake("metal", metal_pick, "Non-Color")
    imgs["occ"] = emit_bake("occ", occ_pick, "Non-Color")
    imgs["normal"] = target("normal", "Non-Color")
    t0 = time.time()
    bpy.ops.object.bake(type='NORMAL', normal_space='TANGENT', uv_layer="atlas", margin=6)
    log("bake %-8s normal %dpx %.0fs" % (tag, size, time.time() - t0))
    clear_targets()
    # pack ORM, save the maps
    px = lambda im: np.array(im.pixels[:], dtype=np.float32).reshape(size, size, 4)
    orm = np.ones((size, size, 4), np.float32)
    orm[..., 0] = px(imgs["occ"])[..., 0]
    orm[..., 1] = px(imgs["rough"])[..., 0]
    orm[..., 2] = px(imgs["metal"])[..., 0]
    im_orm = bpy.data.images.new("%s_%s_orm" % (tag, kind), size, size, alpha=False)
    im_orm.colorspace_settings.name = "Non-Color"
    im_orm.pixels = orm.ravel()
    files = {}
    for key, im, fmt in (("base", imgs["base"], 'JPEG'), ("orm", im_orm, 'PNG'), ("normal", imgs["normal"], 'PNG')):
        f = os.path.join(TEXOUT, "%s_%s_%s.%s" % (tag, kind, key, "jpg" if fmt == 'JPEG' else "png"))
        im.filepath_raw = f; im.file_format = fmt
        if fmt == 'JPEG':
            sc.render.image_settings.quality = 92
        im.save()
        files[key] = f
    for k in ("rough", "metal", "occ"):
        bpy.data.images.remove(imgs[k])
    # one glTF material on the atlas
    import armour_upper as AU
    mat = AU.orm_material("M_%s_%s_atlas" % (kind, tag), files["base"], files["orm"], files["normal"],
                          double_sided=two_sided)
    mat.use_backface_culling = not two_sided
    me.materials.clear(); me.materials.append(mat)
    me.polygons.foreach_set("material_index", np.zeros(len(me.polygons), dtype=np.int32))
    me.uv_layers.remove(me.uv_layers["UVMap"])
    me.uv_layers["atlas"].name = "UVMap"
    me.uv_layers["UVMap"].active_render = True
    for c in m.data.color_attributes[:]:
        m.data.color_attributes.remove(c)        # the vertex AO is baked into the base colour now
    return mat, files


# ------------------------------------------------------------------------------------------------------ LODs
def decimate_to(m, target, name):
    """collapse-decimated copy of m (atlas UVs and skin weights are interpolated by the modifier)"""
    c = dup(m, name)
    c.parent = m.parent; c.matrix_parent_inverse = Matrix.Identity(4)
    n0 = tris(c)
    if target and n0 > target:
        md = c.modifiers.new("dec", 'DECIMATE'); md.decimate_type = 'COLLAPSE'
        md.ratio = max(0.02, target / n0); md.use_collapse_triangulate = True
        select_only([c], c)
        bpy.ops.object.modifier_move_to_index(modifier="dec", index=0)
        bpy.ops.object.modifier_apply(modifier="dec")
    return c


def thin_cards(o, keep_every):
    """drop whole alpha cards (islands) to thin the plume for the lower LODs"""
    import rig_helpers as RH
    comps = RH.components(o.data)
    drop = [c for i, c in enumerate(comps) if i % keep_every]
    import bmesh
    b = bmesh.new(); b.from_mesh(o.data); b.verts.ensure_lookup_table()
    bmesh.ops.delete(b, geom=[b.verts[int(i)] for c in drop for i in c], context='VERTS')
    b.to_mesh(o.data); b.free()


def strip_face_joints(rig, meshes):
    """face joints out of the game skeleton: their weights merge into `head` first"""
    face = [b.name for b in rig.data.bones if b.name.startswith(FACE_BONES)]
    for o in meshes:
        idx = {g.index: g.name for g in o.vertex_groups}
        fg = [g for g in o.vertex_groups if g.name in face]
        if not fg:
            continue
        head = o.vertex_groups.get("head") or o.vertex_groups.new(name="head")
        fi = {g.index for g in fg}
        for v in o.data.vertices:
            w = sum(g.weight for g in v.groups if g.group in fi)
            if w > 0:
                cur = next((g.weight for g in v.groups if g.group == head.index), 0.0)
                head.add([v.index], cur + w, 'REPLACE')
        for g in fg:
            o.vertex_groups.remove(g)
    select_only([rig], rig)
    bpy.ops.object.mode_set(mode='EDIT')
    for n in face:
        eb = rig.data.edit_bones.get(n)
        if eb:
            rig.data.edit_bones.remove(eb)
    bpy.ops.object.mode_set(mode='OBJECT')
    return len(face)


def export(rig, meshes, path, animations=True):
    import build_knight as BK
    for o in meshes:
        un = C.clean_weights(o, rig)
        assert un == 0, (o.name, un)
    keep = set(meshes)
    for o in C.children_meshes(rig):
        if o not in keep:
            o.parent = None                                  # not exported (use_selection), not moved
    for o in meshes:
        o.parent = rig; o.matrix_parent_inverse = Matrix.Identity(4)
    BK.export_glb(rig, path, animations=animations)


def game(kind):
    t0 = time.time()
    import build_knight as BK
    rig = bpy.data.objects["rts_" + kind]
    C.pose_reset(rig)
    if rig.animation_data:
        rig.animation_data.action = None
    for a in [a for a in bpy.data.actions if a.name == "talk_emote"]:
        bpy.data.actions.remove(a)
    for o in [o for o in bpy.data.objects if o.get("rts_probe")]:
        bpy.data.objects.remove(o, do_unlink=True)
    helm = [o for o in C.children_meshes(rig) if o.get("rts_look", "any") in ("helm", "any")]
    plume = [o for o in helm if o.get("rts_part") == "plume"]
    props = [o for o in helm if o.get("rts_prop")]
    armour = [o for o in helm if o not in plume and o not in props]
    bare = [o for o in C.children_meshes(rig) if o.get("rts_look") == "bare"]
    src_tris = sum(tris(o) for o in helm)
    A = merge(armour, "%s_armour" % kind, rig)
    P = merge(props, "%s_props" % kind, rig) if props else None
    Pl = merge(plume, "%s_plume" % kind, rig) if plume else None
    for o in armour + props + plume + bare:
        bpy.data.objects.remove(o, do_unlink=True)
    for x, n in ((A, "armour"), (P, "props"), (Pl, "plume")):
        if x:
            x.name = x.data.name = "%s_%s" % (kind, n)
    rep = {"kind": kind, "source_helm_tris": src_tris, "lods": {}}
    ma, fa = bake_atlas(A, ATLAS["armour"], "armour", kind)
    if P:
        mp, fp = bake_atlas(P, ATLAS["props"], "props", kind)
    if Pl:                                               # plume: its own 1k alpha atlas, no bake
        for im in {n.image for m in Pl.data.materials if m for n in m.node_tree.nodes if n.type == 'TEX_IMAGE' and n.image}:
            if max(im.size) > 1024:
                im.scale(1024, 1024 * im.size[1] // im.size[0])
    nface = strip_face_joints(rig, [x for x in (A, P, Pl) if x])
    os.makedirs(GAME, exist_ok=True)
    A0 = A
    over = sum(tris(x) for x in (A, Pl, P) if x) - LOD_TRIS[0]
    if LOD_TRIS[0] and over > 0:
        A0 = decimate_to(A, tris(A) - over, "%s_armour_lod0" % kind)
        log("LOD0 over budget by %d tris: armour decimated %d -> %d" % (over, tris(A), tris(A0)))
    lods = {0: [x for x in (A0, Pl, P) if x]}
    fixed = (tris(Pl) if Pl else 0)
    for L in (1, 2):
        tgt = LOD_TRIS[L]
        pt = int(tgt * 0.12) if P else 0
        pl = None
        if Pl:
            # the plume is the unit's team-colour crest at RTS range: LOD1 keeps every 2nd card, LOD2 every 5th
            pl = dup(Pl, "%s_plume_lod%d" % (kind, L)); pl.parent = rig; thin_cards(pl, 2 if L == 1 else 5)
        a = decimate_to(A, tgt - pt - (tris(pl) if pl else 0), "%s_armour_lod%d" % (kind, L))
        p = decimate_to(P, pt, "%s_props_lod%d" % (kind, L)) if P else None
        lods[L] = [x for x in (a, pl, p) if x]
    for L, ms in lods.items():
        path = os.path.join(GAME, "knight_%s_lod%d.glb" % (kind, L))
        export(rig, ms, path, animations=True)
        rep["lods"][L] = {"file": os.path.relpath(path, C.CH), "tris": sum(tris(o) for o in ms),
                          "meshes": {o.name: tris(o) for o in ms}, "MB": round(os.path.getsize(path) / 1e6, 2)}
        log("LOD%d %6d tris %s -> %s (%.1f MB)" % (L, rep["lods"][L]["tris"], rep["lods"][L]["meshes"], path,
                                                   rep["lods"][L]["MB"]))
    rep["face_joints_removed"] = nface
    rep["joints"] = len(rig.data.bones)
    rep["atlases"] = {"armour": {k: os.path.relpath(v, C.CH) for k, v in fa.items()}}
    if P:
        rep["atlases"]["props"] = {k: os.path.relpath(v, C.CH) for k, v in fp.items()}
    json.dump(rep, open(os.path.join(GAME, "knight_%s_lods.json" % kind), "w"), indent=1)
    log("game LODs for %s in %.0fs" % (kind, time.time() - t0))


def face(kind):
    """cutscene face data: bare-look meshes + full skeleton + talk_emote only"""
    rig = bpy.data.objects["rts_" + kind]
    C.pose_reset(rig)
    for a in [a for a in bpy.data.actions if a.name != "talk_emote"]:
        bpy.data.actions.remove(a)
    bare = [o for o in C.children_meshes(rig) if o.get("rts_look") == "bare"]
    for o in [o for o in bpy.data.objects if o.type == 'MESH' and o not in bare]:
        bpy.data.objects.remove(o, do_unlink=True)
    os.makedirs(CUT, exist_ok=True)
    path = os.path.join(CUT, "knight_%s_face.glb" % kind)
    import build_knight as BK
    if rig.animation_data:
        rig.animation_data.action = bpy.data.actions.get("talk_emote")
    BK.export_glb(rig, path, animations=True)
    log("face", path, "%.1f MB" % (os.path.getsize(path) / 1e6), [o.name for o in bare])


if __name__ == "__main__":
    a = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    {"game": game, "face": face}[a[0]](a[1] if len(a) > 1 else "male")
