"""Test GLBs for the look-dev viewer (viewer/make_viewer.py), so its generic and modular paths are checked on real files.

  generic: a small non-MakeHuman creature built from scratch: its own 7-bone skeleton with Blender-style names
           (Arm.L -> Arm_L in glTF), 4 skinned meshes with node extras 'slot', shared shape keys (jawOpen, eyeBlinkLeft/Right,
           7 visemes on head + beard), a rigid hat under the Head bone, and three clips: 'Sway' + 'Wave' (bones) and
           'Blink' (shape-key weights).
    $BL -b --factory-startup --python-exit-code 1 -P scripts/viewer_test_glbs.py -- generic out/test/viewer_generic.glb
  piece:   outfit pieces exported WITHOUT the body, to be bound onto out/base_<kind>.glb by joint name in the viewer:
           the smoke-test worksuit + shoes (slots torso / feet), a test cape skinned to two extra bones the body does not
           have (cape_01/02 under spine_05, slot back) and a rigid test sword under hand_r (slot weapon), plus a clip.
    $BL -b out/test/outfit_smoke_male_export.blend --python-exit-code 1 -P scripts/viewer_test_glbs.py -- piece out/test/viewer_piece_test.glb
(BLENDER_USER_RESOURCES=$PWD/blender_profile as for every script here.) Nothing here is saved back to a .blend.
"""
import bpy, bmesh, sys, math
from mathutils import Vector, Matrix

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
MODE, OUT = argv[0], argv[1]


def mat(name, rgb, rough=0.6, metal=0.0):
    m = bpy.data.materials.new(name); m.use_nodes = True
    b = m.node_tree.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = (*rgb, 1); b.inputs["Roughness"].default_value = rough
    b.inputs["Metallic"].default_value = metal
    return m


def seg_dist(p, a, b):
    ab = b - a; t = max(0.0, min(1.0, (p - a).dot(ab) / max(ab.length_squared, 1e-12)))
    return (p - (a + ab * t)).length


def skin(obj, rig, names, sharp=6.0):
    """Weights from the distance to each bone segment (two strongest, normalised)."""
    segs = {n: (rig.data.bones[n].head_local, rig.data.bones[n].tail_local) for n in names}
    groups = {n: obj.vertex_groups.new(name=n) for n in names}
    for v in obj.data.vertices:
        w = sorted(((1.0 / max(seg_dist(v.co, *segs[n]), 0.01) ** sharp, n) for n in names), reverse=True)[:2]
        s = sum(x for x, _ in w)
        for x, n in w: groups[n].add([v.index], x / s, 'REPLACE')
    md = obj.modifiers.new("Armature", 'ARMATURE'); md.object = rig


def tube(name, rings, segs=20):
    """rings: [(z, r, x)]: a closed tube along +Z (quads + caps)."""
    bm = bmesh.new(); loops = []
    for z, r, x in rings:
        loops.append([bm.verts.new((x + r * math.cos(2 * math.pi * i / segs), r * 0.85 * math.sin(2 * math.pi * i / segs), z)) for i in range(segs)])
    for a, b in zip(loops, loops[1:]):
        for i in range(segs):
            bm.faces.new((a[i], a[(i + 1) % segs], b[(i + 1) % segs], b[i]))
    bm.faces.new(list(reversed(loops[0]))); bm.faces.new(loops[-1])
    me = bpy.data.meshes.new(name); bm.to_mesh(me); bm.free()
    for p in me.polygons: p.use_smooth = True
    o = bpy.data.objects.new(name, me); bpy.context.scene.collection.objects.link(o)
    return o


def export(objs, rig, path, anims):
    for o in bpy.context.view_layer.objects: o.select_set(False)
    for o in objs: o.select_set(True)
    rig.select_set(True); bpy.context.view_layer.objects.active = rig
    bpy.ops.export_scene.gltf(filepath=path, export_format='GLB', use_selection=True, export_yup=True, export_apply=False,
                              export_skins=True, export_all_influences=False, export_morph=True, export_morph_normal=True,
                              export_animations=anims, export_animation_mode='ACTIONS', export_extras=True,
                              export_def_bones=False, export_rest_position_armature=True, export_leaf_bone=False,
                              export_materials='EXPORT', export_image_format='AUTO')
    print("VTEST exported", path)


def key_pose(rig, action_name, frames):
    """frames: {frame: {bone: (axis, deg)}} -> a new action on the rig (keeps a fake user)."""
    rig.animation_data_create()
    act = bpy.data.actions.new(action_name); act.use_fake_user = True
    rig.animation_data.action = act
    for f, pose in sorted(frames.items()):
        for pb in rig.pose.bones: pb.rotation_mode = 'QUATERNION'
        for n, (ax, deg) in pose.items():
            pb = rig.pose.bones[n]
            pb.rotation_quaternion = Matrix.Rotation(math.radians(deg), 4, ax).to_quaternion()
            pb.keyframe_insert("rotation_quaternion", frame=f)
    rig.animation_data.action = None
    return act


def generic():
    for o in list(bpy.data.objects): bpy.data.objects.remove(o)
    arm = bpy.data.armatures.new("CreatureRig"); rig = bpy.data.objects.new("Creature", arm)
    bpy.context.scene.collection.objects.link(rig); bpy.context.view_layer.objects.active = rig
    bpy.ops.object.mode_set(mode='EDIT')
    B = {}
    for n, h, t, p in [("Root", (0, 0, 0), (0, 0, 0.35), None), ("Spine", (0, 0, 0.35), (0, 0, 0.9), "Root"),
                       ("Chest", (0, 0, 0.9), (0, 0, 1.3), "Spine"), ("Head", (0, 0, 1.3), (0, 0, 1.75), "Chest"),
                       ("Arm.L", (0.2, 0, 1.2), (0.62, 0, 0.95), "Chest"), ("Arm.R", (-0.2, 0, 1.2), (-0.62, 0, 0.95), "Chest"),
                       ("Jaw", (0, -0.02, 1.52), (0, -0.14, 1.45), "Head")]:
        e = arm.edit_bones.new(n); e.head, e.tail = h, t
        if p: e.parent = B[p]
        B[n] = e
    bpy.ops.object.mode_set(mode='OBJECT')
    body = tube("creature_body", [(0.0, 0.13, 0), (0.2, 0.2, 0), (0.45, 0.24, 0), (0.7, 0.22, 0), (0.95, 0.25, 0), (1.2, 0.2, 0), (1.32, 0.09, 0)])
    arms = tube("creature_arms_tmp", [(0, 0.07, 0), (0.25, 0.06, 0), (0.48, 0.045, 0)], 14)
    # arms: two tubes rotated into place, joined into the body mesh
    armL = arms; armL.name = "armL"; armL.rotation_euler = (0, math.radians(120), 0); armL.location = (0.2, 0, 1.2)
    armR = arms.copy(); armR.data = arms.data.copy(); armR.name = "armR"; bpy.context.scene.collection.objects.link(armR)
    armR.rotation_euler = (0, math.radians(-120), 0); armR.location = (-0.2, 0, 1.2)
    for o in (armL, armR, body): o.select_set(True)
    bpy.context.view_layer.objects.active = body
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    bpy.ops.object.join()
    body.data.materials.append(mat("M_Hide", (0.36, 0.52, 0.30), 0.7)); body["slot"] = "body"
    skin(body, rig, ["Root", "Spine", "Chest", "Arm.L", "Arm.R", "Head"])
    bpy.ops.mesh.primitive_uv_sphere_add(segments=32, ring_count=20, radius=0.17, location=(0, 0, 1.5))
    head = bpy.context.object; head.name = "creature_head"; bpy.ops.object.shade_smooth()
    head.data.materials.append(mat("M_Head", (0.42, 0.58, 0.34), 0.55)); head["slot"] = "head"
    bpy.ops.mesh.primitive_cone_add(vertices=24, radius1=0.2, radius2=0.02, depth=0.3, location=(0, 0, 1.78))
    hat = bpy.context.object; hat.name = "creature_hat"; hat.data.materials.append(mat("M_Hat", (0.12, 0.2, 0.55), 0.4))
    hat["slot"] = "helmet"; hat.parent = rig; hat.parent_type = 'BONE'; hat.parent_bone = "Head"
    hat.matrix_world = Matrix.Translation((0, 0, 1.78))
    bpy.ops.mesh.primitive_uv_sphere_add(segments=24, ring_count=12, radius=0.09, location=(0, -0.13, 1.4))
    beard = bpy.context.object; beard.name = "creature_beard"; beard.scale = (1.1, 0.6, 1.2); bpy.ops.object.shade_smooth()
    bpy.ops.object.transform_apply(scale=True)
    beard.data.materials.append(mat("M_Beard", (0.55, 0.42, 0.2), 0.8)); beard["slot"] = "face"
    bpy.ops.mesh.primitive_torus_add(major_radius=0.25, minor_radius=0.035, location=(0, 0, 0.62))
    belt = bpy.context.object; belt.name = "creature_belt"; belt.data.materials.append(mat("M_Belt", (0.35, 0.2, 0.1), 0.5))
    belt["slot"] = "waist"; bpy.ops.object.shade_smooth()
    for o in (head, beard, belt):          # vertices in rig space before skinning / shape keys
        for x in bpy.context.view_layer.objects: x.select_set(x == o)
        bpy.context.view_layer.objects.active = o; bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    skin(head, rig, ["Head", "Jaw"], 3.0); skin(beard, rig, ["Jaw", "Head"], 3.0); skin(belt, rig, ["Spine", "Root"])
    # shape keys shared by name on head + beard (jaw / blinks / 7 visemes / smile)
    def keys(o, fn):
        o.shape_key_add(name="Basis")
        for n in ["jawOpen", "eyeBlinkLeft", "eyeBlinkRight", "viseme_aa", "viseme_O", "viseme_U", "viseme_E", "viseme_I",
                  "viseme_PP", "viseme_FF", "mouthSmileLeft"]:
            k = o.shape_key_add(name=n, from_mix=False)
            k.value = 0.0                         # glTF default weight (Blender 5.2 creates keys at 1.0)
            for i, v in enumerate(o.data.vertices): k.data[i].co = v.co + Vector(fn(n, v.co))
    def head_fn(n, c):
        low = max(0.0, 1.46 - c.z) * 6; front = max(0.0, -c.y) * 6
        eye = lambda sx: max(0.0, 1 - ((c.x - sx) ** 2 + (c.z - 1.56) ** 2) ** 0.5 / 0.05) * front
        return {"jawOpen": (0, 0, -0.06 * low * front), "eyeBlinkLeft": (0, 0, -0.02 * eye(0.06)), "eyeBlinkRight": (0, 0, -0.02 * eye(-0.06)),
                "viseme_aa": (0, 0, -0.05 * low * front), "viseme_O": (-0.02 * c.x * low * 10, 0, -0.03 * low * front),
                "viseme_U": (-0.03 * c.x * low * 10, -0.02 * low * front, 0), "viseme_E": (0.02 * c.x * low * 10, 0, -0.02 * low * front),
                "viseme_I": (0.025 * c.x * low * 10, 0, -0.01 * low * front), "viseme_PP": (0, 0, 0.01 * low * front),
                "viseme_FF": (0, 0.01 * low * front, 0.005 * low), "mouthSmileLeft": (0.01 * low * front, 0, 0.01 * low * front * (c.x > 0))}[n]
    keys(head, head_fn)
    keys(beard, lambda n, c: (0, 0, -0.05) if n in ("jawOpen", "viseme_aa") else (0, 0, -0.03) if n in ("viseme_O", "viseme_E") else (0, 0, 0))
    X, Y, Z = 'X', 'Y', 'Z'
    key_pose(rig, "Sway", {1: {"Spine": (Z, 0), "Chest": (Z, 0)}, 13: {"Spine": (Z, 12), "Chest": (Z, 10)},
                           37: {"Spine": (Z, -12), "Chest": (Z, -10)}, 49: {"Spine": (Z, 0), "Chest": (Z, 0)}})
    key_pose(rig, "Wave", {1: {"Arm.R": (X, 0)}, 12: {"Arm.R": (X, 95)}, 20: {"Arm.R": (X, 75)}, 28: {"Arm.R": (X, 100)},
                           36: {"Arm.R": (X, 75)}, 48: {"Arm.R": (X, 0)}})
    # shape-key clip on the head
    sk = head.data.shape_keys; sk.animation_data_create()
    act = bpy.data.actions.new("Blink"); act.use_fake_user = True; sk.animation_data.action = act
    for f, v in [(1, 0), (4, 1), (7, 0), (20, 0), (23, 1), (26, 0), (36, 0)]:
        for n in ("eyeBlinkLeft", "eyeBlinkRight"):
            sk.key_blocks[n].value = v; sk.key_blocks[n].keyframe_insert("value", frame=f)
    bpy.context.scene.frame_start, bpy.context.scene.frame_end = 1, 48
    for o in (body, head, beard, belt): o.parent = None
    export([body, head, beard, belt, hat], rig, OUT, True)


def piece():
    rig = bpy.data.objects["rts_male"] if "rts_male" in bpy.data.objects else next(o for o in bpy.data.objects if o.type == 'ARMATURE')
    kind = rig.get("rts_kind", "male")
    keep = [o for o in bpy.data.objects if o.type == 'MESH' and o.get("rts_part")]
    for o in keep: o["slot"] = "feet" if "shoe" in o.name else "torso"
    # extra bones the body does not have: a 2-bone cape chain under spine_05
    bpy.context.view_layer.objects.active = rig; rig.select_set(True)
    bpy.ops.object.mode_set(mode='EDIT')
    eb = rig.data.edit_bones; sp = eb["spine_05"]
    top = sp.head + Vector((0, 0.16, 0.12))                     # Blender space: -Y is the character's front
    c1 = eb.new("cape_01"); c1.head = top; c1.tail = top + Vector((0, 0.05, -0.55)); c1.parent = sp
    c2 = eb.new("cape_02"); c2.head = c1.tail; c2.tail = c1.tail + Vector((0, 0.06, -0.6)); c2.parent = c1; c2.use_connect = True
    bpy.ops.object.mode_set(mode='OBJECT')
    rm = rig.matrix_world
    bm = bmesh.new(); W, N, M = 0.46, 6, 12
    vs = [[bm.verts.new((rm @ top) + Vector(((i / N - 0.5) * W * (1 + 0.5 * j / M), 0.02 + 0.11 * j / M, -1.15 * j / M))) for i in range(N + 1)] for j in range(M + 1)]
    for j in range(M):
        for i in range(N): bm.faces.new((vs[j][i], vs[j][i + 1], vs[j + 1][i + 1], vs[j + 1][i]))
    me = bpy.data.meshes.new("%s_test_cape" % kind); bm.to_mesh(me); bm.free()
    cape = bpy.data.objects.new(me.name, me); bpy.context.scene.collection.objects.link(cape)
    sol = cape.modifiers.new("Solidify", 'SOLIDIFY'); sol.thickness = 0.012
    cape.data.materials.append(mat("M_TestCape", (0.05, 0.12, 0.45), 0.8)); cape["slot"] = "back"
    bpy.context.view_layer.objects.active = cape; bpy.ops.object.modifier_apply(modifier="Solidify")
    skin(cape, rig, ["spine_05", "cape_01", "cape_02"], 4.0)
    # rigid sword under hand_r
    bpy.ops.mesh.primitive_cube_add(size=1)
    sw = bpy.context.object; sw.name = "%s_test_sword" % kind; sw.scale = (0.012, 0.05, 0.95)
    bpy.ops.object.transform_apply(scale=True); sw.data.materials.append(mat("M_TestSteel", (0.8, 0.8, 0.82), 0.25, 1.0))
    sw["slot"] = "weapon"
    hb = rig.data.bones["hand_r"]; hm = rm @ hb.matrix_local
    sw.parent = rig; sw.parent_type = 'BONE'; sw.parent_bone = "hand_r"
    sw.matrix_world = Matrix.Translation(hm @ Vector((0, hb.length * 0.9, 0))) @ Matrix.Translation((0, -0.05, 0.4))
    key_pose(rig, "Cape flutter", {1: {"cape_01": ('X', 0), "cape_02": ('X', 0)}, 13: {"cape_01": ('X', -18), "cape_02": ('X', -25)},
                                   25: {"cape_01": ('X', 0), "cape_02": ('X', 0)}})
    for o in keep + [cape]: o.parent = None
    export(keep + [cape, sw], rig, OUT, True)


{"generic": generic, "piece": piece}[MODE]()
