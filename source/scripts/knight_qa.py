"""Knight QA in Blender (build_knight.py stage 'qa'), on the ENGINE version out/knight_<kind>_export.blend (the
meshes, weights, morphs and clips that are in the GLB):

  exposure   poke-through / gap test: in every test pose (bind, clip frames, extreme poses) each remaining body
             vertex casts 5 rays (normal + 4 tilted 30 deg); a vertex whose rays escape the armour is visible skin.
             renders/knight_<kind>_exposure.json + Workbench renders with the skin in magenta (knight_<kind>_poke_*)
  look       EEVEE look-dev on a dark studio like the reference sheet: turnaround in the idle pose, close-ups,
             RTS camera (renders/knight_<kind>_look_*)
  clips      mid-poses of every clip (renders/knight_<kind>_clip_*)
  face       bare-head look: neutral, visemes, blink, smile, frown, talk frames (renders/knight_<kind>_face_*)
  cape       arms / legs / props passing through the cape in the clips (renders/knight_<kind>_cape_clearance.json)
  intersect  triangle-triangle intersections between every pair of pieces per clip frame
             (renders/knight_<kind>_intersections.json)

run: Blender -b out/knight_<kind>_export.blend --python-exit-code 1 -P scripts/build_knight.py -- qa <kind> [what ..]
"""
import bpy, os, sys, json, math
import numpy as np
from mathutils import Vector, Matrix
from mathutils.bvhtree import BVHTree

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import chr_lib as C


def log(*a):
    print("QA ", *a, flush=True)


def rig_of(kind):
    return bpy.data.objects["rts_" + kind]


def meshes(rig):
    return [o for o in bpy.data.objects if o.type == 'MESH' and not o.get("rts_probe")
            and any(m.type == 'ARMATURE' and m.object == rig for m in o.modifiers)]


def in_look(o, look):
    lk = o.get("rts_look", "any")
    if lk not in ("any", look):
        return False
    if o.get("rts_variant_group") and not o.get("rts_default"):
        return False
    return True


def set_look(rig, look, props=True):
    for o in meshes(rig):
        show = in_look(o, look) and (props or not o.get("rts_prop"))
        o.hide_render = not show
        o.hide_viewport = not show


def play(rig, clip, frame):
    rig.animation_data.action = bpy.data.actions[clip] if clip else None
    if clip is None:
        C.pose_reset(rig)
    bpy.context.scene.frame_set(int(frame))
    bpy.context.view_layer.update()


def face_keys_static(rig):
    """detach the face clip from the shape keys (so expressions can be set by hand)"""
    for o in meshes(rig):
        k = o.data.shape_keys
        if k and k.animation_data:
            k.animation_data.action = None
        if k:
            for kb in k.key_blocks[1:]:
                kb.value = 0.0


def face_keys_clip(rig, name="talk_emote"):
    act = bpy.data.actions.get(name)
    for o in meshes(rig):
        k = o.data.shape_keys
        if not k or not act:
            continue
        if k.animation_data is None:
            k.animation_data_create()
        k.animation_data.action = act
        for sl in act.slots:
            if sl.target_id_type == 'KEY' and sl.name_display == o.name:
                k.animation_data.action_slot = sl


# ------------------------------------------------------------------------------------------ extreme test poses
def _sg(s):
    return 1 if s == "_l" else -1


def P_arms_up(r):
    C._pose_arm_up(r)


def P_arms_forward(r):
    for s in ("_l", "_r"):
        C.aim(r, "upperarm" + s, (0.18 * _sg(s), -1, 0.05)); C.aim(r, "lowerarm" + s, (0.1 * _sg(s), -1, 0.12))


def P_squat(r):
    z = r.pose.bones["foot_l"].head.z
    C.rot(r, "spine_01", (1, 0, 0), 22); C.rot(r, "spine_03", (1, 0, 0), 10)
    for s in ("_l", "_r"):
        C.aim(r, "thigh" + s, (0.32 * _sg(s), -0.93, 0.12)); C.aim(r, "calf" + s, (0.12 * _sg(s), 0.38, -0.92))
        C.aim(r, "foot" + s, (0.12 * _sg(s), -1, -0.25))
    pb = r.pose.bones["pelvis"]
    low = min(r.pose.bones["foot_l"].head.z, r.pose.bones["foot_r"].head.z)
    M = pb.matrix.copy(); M.translation.z -= (low - z); pb.matrix = M
    bpy.context.view_layer.update()


def P_lunge(r):
    C.aim(r, "thigh_l", (0.12, -1.0, -0.55)); C.aim(r, "calf_l", (0.06, 0.05, -1.0)); C.aim(r, "foot_l", (0.05, -1.0, -0.3))
    C.aim(r, "thigh_r", (-0.08, 0.55, -1.0)); C.aim(r, "calf_r", (-0.04, 0.95, -0.6)); C.aim(r, "foot_r", (0, -0.1, -1.0))
    pb = r.pose.bones["pelvis"]; M = pb.matrix.copy(); M.translation.z -= 0.16; pb.matrix = M
    bpy.context.view_layer.update()


def P_kneel(r):
    C.aim(r, "thigh_l", (0.1, -1.0, -0.25)); C.aim(r, "calf_l", (0.05, 0.1, -1.0))
    C.aim(r, "thigh_r", (-0.08, -0.05, -1.0)); C.aim(r, "calf_r", (-0.03, 1.0, -0.12)); C.aim(r, "foot_r", (0, 0.2, -1.0))
    pb = r.pose.bones["pelvis"]; M = pb.matrix.copy(); M.translation.z -= 0.36; pb.matrix = M
    bpy.context.view_layer.update()


def P_twist(r):
    for b in ("spine_03", "spine_04", "spine_05"):
        C.rot(r, b, (0, 0, 1), 16)
    C.rot(r, "head", (0, 0, 1), 25)


def P_bend(r):
    C.rot(r, "spine_03", (1, 0, 0), 20); C.rot(r, "spine_04", (1, 0, 0), 20)
    C.rot(r, "neck_02", (1, 0, 0), 25)


def P_elbows(r):
    for s in ("_l", "_r"):
        C.aim(r, "upperarm" + s, (0.25 * _sg(s), -0.25, -1)); C.aim(r, "lowerarm" + s, (0.1 * _sg(s), -1, 0.35))


EXTREME = {"arms_up": P_arms_up, "arms_forward": P_arms_forward, "squat": P_squat, "lunge": P_lunge, "kneel": P_kneel,
           "torso_twist": P_twist, "bend": P_bend, "elbows": P_elbows}
CLIP_FRAMES = [("idle", 0), ("idle", 90), ("walk", 0), ("walk", 8), ("walk", 16), ("walk", 25), ("run", 0), ("run", 5),
               ("run", 10), ("run", 15), ("attack_sword", 11), ("attack_sword", 15), ("attack_sword", 19),
               ("attack_sword", 24), ("block_shield", 8), ("block_shield", 16), ("block_shield", 30)]


def test_poses(rig):
    """yields (name, setter) for every test pose"""
    yield "bind", lambda: play(rig, None, 0)
    for c, f in CLIP_FRAMES:
        if bpy.data.actions.get(c):
            yield "%s_f%02d" % (c, f), (lambda c=c, f=f: play(rig, c, f))
    for n, fn in EXTREME.items():
        def st(fn=fn):
            play(rig, None, 0); fn(rig)
            import rig_helpers
            rig_helpers.drive_helpers(rig)                 # plate helper joints follow the hand-set pose
            body = bpy.data.objects.get(rig.get("rts_kind", "male") + "_skin_probe")
            if body is not None:
                C.drive_correctives(rig, body)
        yield n, st


def _eval(o, dg):
    ev = o.evaluated_get(dg); me = ev.to_mesh()
    co = [o.matrix_world @ v.co for v in me.vertices]
    no = [(o.matrix_world.to_3x3() @ v.normal).normalized() for v in me.vertices]
    polys = [tuple(p.vertices) for p in me.polygons]
    ev.to_mesh_clear()
    return co, no, polys


def probe_of(kind):
    """the full-skin probe saved with the export (build_knight.make_probe): skin under the armour, not exported"""
    pr = bpy.data.objects.get(kind + "_skin_probe")
    me = pr.data
    if "orig_index" not in me.attributes:
        a = me.attributes.new("orig_index", 'INT', 'POINT')
        a.data.foreach_set("value", np.arange(len(me.vertices), dtype=np.int32))
    return pr


INSET = 0.004       # rays start 4 mm under the skin: skin that merely touches a low-poly layer (glove, mail) is not a gap


def gap_scan(rig, probe, pieces, poses=None, inset=INSET):
    """{pose: [probe vertex indices whose skin is visible past the pieces]} over the test poses; rays from 4 mm
    under the skin along the normal and 4 directions tilted 30 deg, >= 2 escaping = visible"""
    me0 = probe.data
    if "orig_index" not in me0.attributes:
        a = me0.attributes.new("orig_index", 'INT', 'POINT')
        a.data.foreach_set("value", np.arange(len(me0.vertices), dtype=np.int32))
    out = {}
    for pname, setter in test_poses(rig):
        if poses and pname not in poses:
            continue
        setter()
        dg = bpy.context.evaluated_depsgraph_get()
        V, P = [], []
        for o in pieces:
            co, no, polys = _eval(o, dg)
            off = len(V); V += co; P += [tuple(i + off for i in p) for p in polys]
        bvh = BVHTree.FromPolygons(V, P)
        ev = probe.evaluated_get(dg); me = ev.to_mesh()
        oi = np.empty(len(me.vertices), dtype=np.int32); me.attributes["orig_index"].data.foreach_get("value", oi)
        M = probe.matrix_world; M3 = M.to_3x3()
        exposed = []
        for i, v in enumerate(me.vertices):
            p = M @ v.co; n = (M3 @ v.normal).normalized()
            t = n.orthogonal().normalized(); b = n.cross(t)
            esc = 0
            for d in (n, (n + t * 0.58).normalized(), (n - t * 0.58).normalized(), (n + b * 0.58).normalized(), (n - b * 0.58).normalized()):
                if bvh.ray_cast(p - n * inset, d, 1.5)[0] is None:
                    esc += 1
            if esc >= 2:
                exposed.append(int(oi[i]))
        nv = len(me.vertices)
        ev.to_mesh_clear()
        out[pname] = (exposed, nv)
    return out


def dominant_bones(rig, obj):
    bones = {b.name for b in rig.data.bones}
    names = {g.index: g.name for g in obj.vertex_groups}
    dom = []
    for v in obj.data.vertices:
        gs = [g for g in v.groups if names[g.group] in bones]
        g = max(gs, key=lambda g: g.weight) if gs else None
        dom.append(names[g.group] if g else "?")
    return dom


def exposure(kind, render=True):
    """skin of the probe visible past the armour in every test pose (gaps between pieces or pieces pushed into the
    body); the knight export keeps no skin below the neck, the gaps found at build time are closed by the
    under-layer filler meshes, so this should report only face / finger false positives"""
    rig = rig_of(kind)
    probe = probe_of(kind)
    probe.hide_viewport = False
    face_keys_static(rig)
    set_look(rig, "helm", props=False)
    pieces = [o for o in meshes(rig) if in_look(o, "helm") and not o.get("rts_prop")]
    dom = dominant_bones(rig, probe)
    res = {}
    for pname, (exposed, nv) in gap_scan(rig, probe, pieces).items():
        regions = {}
        for i in exposed:
            regions[dom[i]] = regions.get(dom[i], 0) + 1
        real = [i for i in exposed if not IGNORE_BONE(dom[i])]
        res[pname] = dict(exposed=len(exposed), gaps=len(real), of=nv,
                          regions=dict(sorted(regions.items(), key=lambda kv: -kv[1])[:10]))
        log("exposure %-18s %4d / %d skin verts visible, %3d outside face / hands %s"
            % (pname, len(exposed), nv, len(real), res[pname]["regions"]))
    json.dump(res, open(os.path.join(C.REN, "knight_%s_exposure.json" % kind), "w"), indent=1)
    if render:
        poke_renders(kind, res)
    play(rig, None, 0)
    probe.hide_viewport = True
    return res


FACE_HAND = ("head", "eye", "eyelid", "brow", "cheek", "nose", "mouth", "lip", "tongue", "hand", "thumb",
             "index", "middle", "ring", "pinky")


def IGNORE_BONE(b):
    """skin that is invisible in the export whatever the rays say: the face inside the closed helm (seen only through
    the visor slits, lined dark) and the hand inside the glove (its low-poly shell lets curled skin fingers through)"""
    return b.startswith(FACE_HAND)


def poke_renders(kind, res):
    """Workbench clay with the skin in magenta: visible skin (gaps / poke-through) jumps out"""
    rig = rig_of(kind)
    body = probe_of(kind)
    body.hide_render = False; body.hide_viewport = False
    sc = C.render_setup("BLENDER_WORKBENCH", (360, 520))
    sc.display.shading.light = 'STUDIO'; sc.display.shading.color_type = 'OBJECT'
    sc.display.shading.show_cavity = False
    for o in meshes(rig):
        o.color = (0.62, 0.62, 0.64, 1)
    body.color = (1.0, 0.0, 0.85, 1)
    set_look(rig, "helm", props=False)
    for pname, setter in test_poses(rig):
        setter()
        for view, loc in (("front", (0.0, -3.6, 1.0)), ("back", (0.0, 3.6, 1.0)), ("side", (-3.6, 0.0, 1.0))):
            C.camera(loc, (0, 0, 0.92), lens=45)
            C.render(os.path.join(C.REN, "knight_%s_poke_%s_%s.png" % (kind, pname, view)))
    body.hide_render = True
    play(rig, None, 0)


# ------------------------------------------------------------------------------------------ look-dev
def studio(engine="BLENDER_EEVEE"):
    import armour_upper as AU
    sc = AU.lookdev_setup(engine, 64)
    sc.render.film_transparent = False
    return sc


VIEWS = {  # name: (camera, target, lens, res, yaw)
    "front": ((0, -3.45, 1.12), (0, 0, 1.02), 50, (900, 1300), 0),
    "right": ((0, -3.45, 1.12), (0, 0, 1.02), 50, (900, 1300), 90),
    "back": ((0, -3.45, 1.12), (0, 0, 1.02), 50, (900, 1300), 180),
    "left": ((0, -3.45, 1.12), (0, 0, 1.02), 50, (900, 1300), -90),
    "34": ((-2.2, -2.75, 1.35), (0, 0, 1.02), 50, (900, 1300), 0),
    "helm": ((-0.62, -0.8, 1.86), (0, -0.05, 1.72), 70, (900, 1000), 0),
    "shoulder": ((-0.95, -0.72, 1.62), (-0.24, -0.03, 1.42), 60, (900, 1000), 0),
    "torso": ((0.0, -1.55, 1.35), (0, -0.05, 1.2), 50, (900, 1000), 0),
    "shield": ((1.25, -1.5, 1.2), (0.35, -0.2, 0.95), 50, (900, 1000), 0),
    "sword_hand": ((-1.1, -1.05, 1.0), (-0.32, -0.25, 0.85), 55, (900, 1000), 0),
    "legs": ((-0.5, -1.7, 0.55), (0.0, 0, 0.45), 45, (900, 1000), 0),
    "cape": ((0.4, -2.4, 1.2), (0, 0, 0.95), 50, (900, 1200), 180),
    "rts": ((-5.5, -8.0, 10.5), (0, 0, 0.9), 60, (480, 480), 0),
}


def look(kind, shots=None, tag="look"):
    rig = rig_of(kind)
    face_keys_static(rig)
    studio()
    set_look(rig, "helm")
    play(rig, "idle", 0)
    for nm, (loc, tgt, lens, res, yaw) in VIEWS.items():
        if shots and nm not in shots:
            continue
        sc = bpy.context.scene
        sc.render.resolution_x, sc.render.resolution_y = res
        rig.rotation_euler.z = math.radians(yaw)
        bpy.context.view_layer.update()
        C.camera(loc, tgt, lens=lens)
        C.render(os.path.join(C.REN, "knight_%s_%s_%s.png" % (kind, tag, nm)))
    rig.rotation_euler.z = 0.0


def clips(kind):
    rig = rig_of(kind)
    face_keys_static(rig)
    studio()
    set_look(rig, "helm")
    sc = bpy.context.scene
    sc.render.resolution_x, sc.render.resolution_y = (800, 1000)
    shots = [("idle", 90, "34"), ("walk", 4, "side"), ("walk", 12, "34"), ("run", 3, "side"), ("run", 12, "34"),
             ("attack_sword", 11, "34"), ("attack_sword", 17, "34"), ("attack_sword", 19, "front"), ("attack_sword", 24, "34"),
             ("block_shield", 10, "34"), ("block_shield", 16, "front"), ("talk_emote", 110, "34")]
    cams = {"34": ((-2.6, -3.3, 1.55), (0, -0.1, 0.95)), "side": ((-4.0, -0.3, 1.2), (0, -0.1, 0.95)),
            "front": ((0.3, -4.1, 1.3), (0, -0.1, 0.95))}
    for c, f, v in shots:
        if not bpy.data.actions.get(c):
            continue
        play(rig, c, f)
        C.camera(cams[v][0], cams[v][1], lens=45)
        C.render(os.path.join(C.REN, "knight_%s_clip_%s_f%02d.png" % (kind, c, f)))


FACE_SHOTS = [
    ("neutral", {}), ("smile", {"mouthSmileLeft": 0.85, "mouthSmileRight": 0.8, "cheekSquintLeft": 0.45, "cheekSquintRight": 0.4,
                                "eyeSquintLeft": 0.3, "eyeSquintRight": 0.3, "mouthUpperUpLeft": 0.25, "mouthUpperUpRight": 0.25}),
    ("frown", {"mouthFrownLeft": 0.75, "mouthFrownRight": 0.7, "browDownLeft": 0.7, "browDownRight": 0.7, "mouthPressLeft": 0.35,
               "mouthPressRight": 0.35, "noseSneerLeft": 0.25, "noseSneerRight": 0.2}),
    ("blink", {"eyeBlinkLeft": 1.0, "eyeBlinkRight": 1.0}), ("viseme_aa", {"viseme_aa": 1.0}), ("viseme_O", {"viseme_O": 1.0}),
    ("viseme_FF", {"viseme_FF": 1.0}), ("viseme_PP", {"viseme_PP": 1.0}), ("battle_cry", {"jawOpen": 0.7, "mouthStretchLeft": 0.5,
        "mouthStretchRight": 0.5, "browDownLeft": 0.8, "browDownRight": 0.8, "noseSneerLeft": 0.5, "noseSneerRight": 0.5}),
]


def face(kind):
    rig = rig_of(kind)
    studio()
    set_look(rig, "bare")
    play(rig, "idle", 0)
    C.pose_reset(rig)
    rig.animation_data.action = None
    C.pose_reset(rig)
    sc = bpy.context.scene
    sc.render.resolution_x, sc.render.resolution_y = (700, 800)
    head = rig.pose.bones["head"].head
    eye = (rig.pose.bones["eye_l"].head + rig.pose.bones["eye_r"].head) / 2
    tgt = Vector((0, eye.y + 0.02, eye.z - 0.035))
    face_keys_static(rig)
    for nm, keys in FACE_SHOTS:
        for o in meshes(rig):
            k = o.data.shape_keys
            if not k:
                continue
            for kb in k.key_blocks[1:]:
                kb.value = keys.get(kb.name, 0.0)
        for view, off in (("front", (0.0, -0.55, 0.02)), ("34", (-0.36, -0.42, 0.04))):
            C.camera(tuple(tgt + Vector(off)), tuple(tgt), lens=85)
            C.render(os.path.join(C.REN, "knight_%s_face_%s_%s.png" % (kind, nm, view)))
    face_keys_static(rig)
    face_keys_clip(rig)
    for f in (15, 30, 57, 75, 118, 155, 175, 215):
        play(rig, "talk_emote", f)
        e = (rig.pose.bones["eye_l"].head + rig.pose.bones["eye_r"].head) / 2
        t2 = Vector((0, e.y + 0.02, e.z - 0.035))
        C.camera(tuple(t2 + Vector((-0.2, -0.52, 0.03))), tuple(t2), lens=85)
        C.render(os.path.join(C.REN, "knight_%s_face_talk_f%03d.png" % (kind, f)))


CAPE_HITTERS = ("pauldron", "rerebrace", "couter", "vambrace", "gauntlet", "shield", "sword", "cuisses", "poleyns",
                "greaves", "sabatons", "boots", "legs_mail", "tassets", "scabbard")


def cape_clearance(kind, step=2, tol=0.004, verbose=True):
    """Arms, legs and props passing through the cape in the clips: every vertex of those pieces casts a ray forward
    (-Y in the character's frame); if it hits the cape within 0.5 m, the vertex is behind (or inside) the cape.
    Returns {clip: {piece: [n verts, max depth mm, worst frame]}} (tol: grazing contacts under 4 mm ignored)."""
    rig = rig_of(kind)
    cape = bpy.data.objects.get(kind + "_cape")
    if cape is None:
        return {}
    face_keys_static(rig)
    hitters = [o for o in meshes(rig) if o.name[len(kind) + 1:].startswith(CAPE_HITTERS)]
    table = json.loads(rig.get("rts_clips", "{}"))
    out = {}
    for clip in [c for c in table if c != "talk_emote" and bpy.data.actions.get(c)]:
        act = bpy.data.actions[clip]
        nf = int(round(act.frame_range[1])) + 1
        res = {}
        for f in range(0, nf, step):
            play(rig, clip, f)
            dg = bpy.context.evaluated_depsgraph_get()
            co, no, polys = _eval(cape, dg)
            bvh = BVHTree.FromPolygons(co, polys)
            # the character's forward (-Y; the clips do not turn the character)
            fwd = (rig.matrix_world.to_3x3() @ Vector((0, -1, 0))).normalized()
            for o in hitters:
                sk = o.get("rts_socket")
                if sk and sk in rig.pose.bones:
                    # a prop whose grip is above the cape top or behind the cape lies outside it (sword wound up over
                    # the shoulder): correct layering, not a pass-through
                    g = rig.matrix_world @ rig.pose.bones[sk].head
                    if g.z > max(v.z for v in co) - 0.02 or bvh.ray_cast(g, fwd, 0.5)[0] is not None:
                        continue
                pc, _, _ = _eval(o, dg)
                n = 0; dmax = 0.0
                for p in pc:
                    hit = bvh.ray_cast(p, fwd, 0.5)
                    if hit[0] is not None and hit[3] > tol:
                        n += 1; dmax = max(dmax, hit[3])
                if n:
                    nm = o.name[len(kind) + 1:]
                    r = res.setdefault(nm, [0, 0.0, 0])
                    if n > r[0]:
                        r[0] = n; r[2] = f
                    r[1] = max(r[1], round(dmax * 1000, 1))
        out[clip] = res
        if verbose:
            log("cape clearance %-13s %s" % (clip, res or "clear"))
    return out


def intersections(kind, step=3, look="helm", poses=None, verbose=True):
    """Numeric penetration check (user feedback 19): per clip frame (every `step` frames, plus the extreme test
    poses) every pair of visible pieces is tested for triangle-triangle intersections (BVHTree.overlap). Returns
    {pair: {"max": n intersecting triangle pairs, "at": pose, "frames": frames with any}} and writes
    renders/knight_<kind>_intersections.json. Pieces that are layered by design (mail under plate, the under-layer
    filler, cloth over mail) still count: the numbers are for tracking regressions and finding the worst pairs."""
    rig = rig_of(kind)
    face_keys_static(rig)
    objs = [o for o in meshes(rig) if in_look(o, look) and o.get("rts_part")]
    names = [o.name[len(kind) + 1:] for o in objs]
    table = json.loads(rig.get("rts_clips", "{}"))
    frames = [("bind", None, 0)]
    for clip in [c for c in table if c != "talk_emote" and bpy.data.actions.get(c)]:
        nf = int(round(bpy.data.actions[clip].frame_range[1]))
        frames += [("%s_f%02d" % (clip, f), clip, f) for f in range(0, nf + 1, step)]
    res = {}
    per_pose = {}
    for pname, clip, f in frames:
        if poses and pname not in poses:
            continue
        play(rig, clip, f)
        dg = bpy.context.evaluated_depsgraph_get()
        bvh = {}
        for o, nm in zip(objs, names):
            co, _, polys = _eval(o, dg)
            bvh[nm] = BVHTree.FromPolygons(co, polys)
        tot = 0
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a, b = names[i], names[j]
                n = len(bvh[a].overlap(bvh[b]))
                if n:
                    key = "%s|%s" % (a, b)
                    r = res.setdefault(key, {"max": 0, "at": None, "frames": 0})
                    r["frames"] += 1
                    if n > r["max"]:
                        r["max"], r["at"] = n, pname
                    tot += n
        per_pose[pname] = tot
    out = {"pairs": dict(sorted(res.items(), key=lambda kv: -kv[1]["max"])), "total_per_pose": per_pose,
           "poses": len(per_pose)}
    json.dump(out, open(os.path.join(C.REN, "knight_%s_intersections.json" % kind), "w"), indent=1)
    if verbose:
        log("intersections: %d poses, %d piece pairs ever intersecting; worst:" % (len(per_pose), len(res)))
        for k, r in list(out["pairs"].items())[:25]:
            log("   %-28s max %5d tri pairs at %-18s (%d poses)" % (k, r["max"], r["at"], r["frames"]))
    return out


# ------------------------------------------------------------------------------------------ rig / animation gates
def _clip_frames(rig, step=1):
    table = json.loads(rig.get("rts_clips", "{}"))
    out = []
    for clip in [c for c in table if c != "talk_emote" and bpy.data.actions.get(c)]:
        nf = int(round(bpy.data.actions[clip].frame_range[1]))
        out += [(clip, f) for f in range(0, nf + 1, step)]
    return out


def strain(kind, step=2, res_mm=3.0, min_sv=0.97, verbose=True):
    """Plate-strain gate (judge C1 / M1): every connected component of every plate piece (rig_helpers.PLATE_SLOTS) is
    fitted rigidly (Kabsch) to its bind shape in every clip frame (every `step`) and the extreme test poses; a plate
    passes when the worst residual stays under `res_mm` and the best-fit linear map never squashes it below `min_sv`.
    Writes renders/knight_<kind>_strain.json; returns {slot: {...}}."""
    import rig_helpers as RH
    rig = rig_of(kind)
    face_keys_static(rig)
    objs = {o["rts_part"]: o for o in meshes(rig) if o.get("rts_part") in RH.PLATE_SLOTS}
    comps = {}
    for s, o in objs.items():
        cs = [c for c in RH.components(o.data) if len(c) >= 12]
        comps[s] = cs
    play(rig, None, 0)
    dg = bpy.context.evaluated_depsgraph_get()
    rest = {}
    for s, o in objs.items():
        co, _, _ = _eval(o, dg)
        rest[s] = np.array([v[:] for v in co])
    poses = [("%s_f%02d" % (c, f), (lambda c=c, f=f: play(rig, c, f))) for c, f in _clip_frames(rig, step)]
    for n, fn in EXTREME.items():
        def st(fn=fn):
            play(rig, None, 0); fn(rig); RH.drive_helpers(rig)
        poses.append((n, st))
    rep = {s: {"res_mm": 0.0, "min_sv": 1.0, "at": None, "comps": len(comps[s])} for s in objs}
    for pname, setter in poses:
        setter()
        dg = bpy.context.evaluated_depsgraph_get()
        for s, o in objs.items():
            co, _, _ = _eval(o, dg)
            P = np.array([v[:] for v in co])
            for c in comps[s]:
                r, sv = RH.kabsch(rest[s][c], P[c])
                R = rep[s]
                if r * 1000 > R["res_mm"]:
                    R["res_mm"] = round(r * 1000, 1); R["at"] = pname
                R["min_sv"] = round(min(R["min_sv"], sv), 3)
    play(rig, None, 0)
    ok = True
    for s, R in sorted(rep.items()):
        R["pass"] = R["res_mm"] <= res_mm and R["min_sv"] >= min_sv
        ok &= R["pass"]
        if verbose:
            log("strain %-12s %s  worst residual %6.1f mm at %-18s min singular value %.3f (%d components)"
                % (s, "PASS" if R["pass"] else "FAIL", R["res_mm"], R["at"], R["min_sv"], R["comps"]))
    out = {"gate": {"res_mm": res_mm, "min_sv": min_sv, "pass": ok}, "pieces": rep}
    json.dump(out, open(os.path.join(C.REN, "knight_%s_strain.json" % kind), "w"), indent=1)
    return out


FOOT_MESH = ("sabatons", "boots", "greaves")


def feet(kind, contact=0.005, max_slide=0.05, min_sole=-0.005, verbose=True):
    """Foot-contact gate (judge M13 / m4): per clip frame, three contact points per foot (heel under the ankle, ball,
    toe tip; fixed in the foot / toe frames from the rest pose at floor level) and the lowest boot / sabaton vertex.
    A point within `contact` m of the floor is planted; its horizontal speed, corrected for in-place travel
    (rts_clips speed_mps: the ground moves +Y under the character), must stay below `max_slide` m/s (planted in two
    consecutive frames). The sole must never sink below `min_sole` m. Writes renders/knight_<kind>_feet.json."""
    rig = rig_of(kind)
    face_keys_static(rig)
    table = json.loads(rig.get("rts_clips", "{}"))
    B = rig.data.bones
    pts = {}
    for s in ("l", "r"):
        ft, bl = B["foot_" + s], B["ball_" + s]
        ank = ft.head_local; ball = bl.head_local; toe = bl.tail_local
        heel = Vector((ank.x, ank.y + 0.035, 0.0))
        pts[s] = [("heel", "foot_" + s, ft.matrix_local.inverted() @ heel),
                  ("ball", "ball_" + s, bl.matrix_local.inverted() @ Vector((ball.x, ball.y, 0.0))),
                  ("toe", "ball_" + s, bl.matrix_local.inverted() @ Vector((toe.x, toe.y, 0.0)))]
    soles = [o for o in meshes(rig) if o.get("rts_part") in FOOT_MESH]
    rep = {}
    ok = True
    for clip in [c for c in table if c != "talk_emote" and bpy.data.actions.get(c)]:
        info = table[clip]
        nf = int(round(bpy.data.actions[clip].frame_range[1])) + 1
        v_ground = info.get("speed_mps", 0.0)
        fps = info.get("fps", 30)
        track = {s: {n: [] for n, _, _ in pts[s]} for s in pts}
        sole = []
        for f in range(nf):
            play(rig, clip, f)
            for s in pts:
                for n, b, lp in pts[s]:
                    track[s][n].append((rig.matrix_world @ rig.pose.bones[b].matrix) @ lp)
            if soles:
                dg = bpy.context.evaluated_depsgraph_get()
                zmin = 9.0
                for o in soles:
                    ev = o.evaluated_get(dg); me = ev.to_mesh()
                    z = np.empty(len(me.vertices) * 3); me.vertices.foreach_get("co", z)
                    zmin = min(zmin, float(z.reshape(-1, 3)[:, 2].min()))
                    ev.to_mesh_clear()
                sole.append(zmin)
        worst = (0.0, None)
        for s in pts:
            for n in track[s]:
                P = track[s][n]
                for f in range(nf - 1):
                    if P[f].z < contact and P[f + 1].z < contact:
                        d = P[f + 1] - P[f]
                        vx, vy = d.x * fps, d.y * fps - v_ground
                        v = math.hypot(vx, vy)
                        if v > worst[0]:
                            worst = (v, "%s_%s f%d" % (n, s, f))
        smin = min(sole) if sole else 0.0
        R = {"max_slide_mps": round(worst[0], 3), "at": worst[1], "min_sole_mm": round(smin * 1000, 1),
             "min_sole_frame": int(np.argmin(sole)) if sole else None}
        R["pass"] = worst[0] <= max_slide and smin >= min_sole
        ok &= R["pass"]
        rep[clip] = R
        if verbose:
            log("feet %-13s %s  max planted slide %.3f m/s (%s)  lowest sole %6.1f mm (f%s)"
                % (clip, "PASS" if R["pass"] else "FAIL", worst[0], worst[1], smin * 1000, R["min_sole_frame"]))
    out = {"gate": {"max_slide_mps": max_slide, "min_sole_m": min_sole, "pass": ok}, "clips": rep}
    json.dump(out, open(os.path.join(C.REN, "knight_%s_feet.json" % kind), "w"), indent=1)
    play(rig, None, 0)
    return out


def shield_carry(kind, verbose=True):
    """Shield attitude per clip frame (judge M13): angle of the shield's long axis (socket_shield_l +Y) from the
    vertical, and where its face points. Idle / walk / run should carry it within 10 deg of vertical."""
    rig = rig_of(kind)
    if "socket_shield_l" not in rig.pose.bones:
        return {}
    table = json.loads(rig.get("rts_clips", "{}"))
    rep = {}
    for clip in [c for c in table if c != "talk_emote" and bpy.data.actions.get(c)]:
        nf = int(round(bpy.data.actions[clip].frame_range[1])) + 1
        tilts = []
        for f in range(0, nf, 2):
            play(rig, clip, f)
            M = (rig.matrix_world @ rig.pose.bones["socket_shield_l"].matrix).to_3x3()
            ax = M @ Vector((0, 1, 0))
            tilts.append(math.degrees(ax.angle(Vector((0, 0, -1)))))
        play(rig, clip, 0)
        M = (rig.matrix_world @ rig.pose.bones["socket_shield_l"].matrix).to_3x3()
        face = M @ Vector((0, 0, 1))
        rep[clip] = {"tilt_max": round(max(tilts), 1), "tilt_mean": round(float(np.mean(tilts)), 1),
                     "face_f0": [round(x, 2) for x in face]}
        if verbose:
            log("shield %-13s long axis from vertical: mean %5.1f deg, max %5.1f deg; face at f0 %s"
                % (clip, rep[clip]["tilt_mean"], rep[clip]["tilt_max"], rep[clip]["face_f0"]))
    play(rig, None, 0)
    json.dump(rep, open(os.path.join(C.REN, "knight_%s_shield.json" % kind), "w"), indent=1)
    return rep


I33 = os.path.join(C.CH, "refs", "feedback", "user_block_guard_reference.png")
I33_CROP = (0, 960, 800, 1580)       # the lower-left fighter: half-shield guard, sword hand under the buckler (side view)
BLOCK_HOLD = 28                      # block_shield: guard held (impact recoil over), before the step back


def _world(o, dg):
    co, _, polys = _eval(o, dg)
    return np.array([v[:] for v in co]), polys


def block(kind, frame=BLOCK_HOLD, verbose=True):
    """Block guard vs the I.33 reference (user item 18): guard metrics at the hold frame (shield ahead of the chest,
    height, axis tilt, sword grip behind the shield face / beside its edge, blade direction, stance, elbow, triangle
    overlaps of the shield with the other pieces) -> renders/knight_<kind>_block.json, renders from the left side (the
    manuscript's view), 3/4 and front -> renders/knight_<kind>_block_{side,34,front}.png, and a sheet next to the
    manuscript figure -> renders/knight_<kind>_block_vs_i33.png"""
    rig = rig_of(kind)
    face_keys_static(rig)
    set_look(rig, "helm")
    B = rig.pose.bones
    W = rig.matrix_world
    play(rig, "idle", 0)
    pel0 = (W @ B["pelvis"].matrix).translation.z
    play(rig, "block_shield", frame)
    dg = bpy.context.evaluated_depsgraph_get()
    obj = {o.get("rts_part"): o for o in meshes(rig) if o.get("rts_part")}
    out = {"frame": frame}
    if "shield" in obj and "socket_shield_l" in B:
        sh, shp = _world(obj["shield"], dg)
        Ms = W @ B["socket_shield_l"].matrix
        face = (Ms.to_3x3() @ Vector((0, 0, 1))).normalized()
        axis = (Ms.to_3x3() @ Vector((0, 1, 0))).normalized()
        cen = Vector(sh.mean(0))
        chest = (W @ B["spine_05"].matrix).translation
        Mw = W @ B["socket_weapon_r"].matrix
        grip = Mw.translation
        blade = (Mw.to_3x3() @ Vector((0, 1, 0))).normalized()
        side = face.cross(Vector((0, 0, 1))).normalized()
        bvh = BVHTree.FromPolygons([Vector(v) for v in sh], shp)
        hits = {}
        for part, o in obj.items():
            if part == "shield":
                continue
            co, pp = _world(o, dg)
            n = len(bvh.overlap(BVHTree.FromPolygons([Vector(v) for v in co], pp)))
            if n:
                hits[part] = n
        up_l = B["upperarm_l"].matrix.to_3x3() @ Vector((0, 1, 0))
        lo_l = B["lowerarm_l"].matrix.to_3x3() @ Vector((0, 1, 0))
        feet = {s: (W @ B["foot_" + s].matrix).translation for s in "lr"}
        out.update(
            shield_ahead_of_chest_m=round(chest.y - cen.y, 3), shield_centre_z_m=round(cen.z, 3),
            shield_axis_from_vertical_deg=round(math.degrees(axis.angle(Vector((0, 0, -1)))), 1),
            shield_face=[round(x, 2) for x in face],
            grip_behind_shield_face_m=round(-(grip - cen).dot(face), 3),
            grip_from_shield_centre_sideways_m=round(abs((grip - cen).dot(side)), 3),
            blade_yaw_from_forward_deg=round(math.degrees(math.atan2(-blade.x, -blade.y)), 1),
            blade_elevation_deg=round(math.degrees(math.asin(max(-1, min(1, blade.z)))), 1),
            shield_elbow_flexion_deg=round(math.degrees(up_l.angle(lo_l)), 1),
            stance_width_m=round(abs(feet["l"].x - feet["r"].x), 3), stance_depth_m=round(feet["r"].y - feet["l"].y, 3),
            pelvis_drop_m=round(pel0 - (W @ B["pelvis"].matrix).translation.z, 3),
            shield_overlaps=hits)
    json.dump(out, open(os.path.join(C.REN, "knight_%s_block.json" % kind), "w"), indent=1)
    if verbose:
        log("block %s f%d: %s" % (kind, frame, json.dumps(out)))
    studio()
    sc = bpy.context.scene
    sc.render.resolution_x, sc.render.resolution_y = (800, 900)
    cams = {"side": ((3.9, -0.45, 1.1), (0, -0.25, 0.92)), "34": ((2.5, -3.1, 1.45), (0, -0.2, 0.92)),
            "front": ((0.25, -4.0, 1.25), (0, -0.2, 0.92))}
    files = []
    for v, (loc, tgt) in cams.items():
        C.camera(loc, tgt, lens=42)
        f = os.path.join(C.REN, "knight_%s_block_%s.png" % (kind, v))
        C.render(f)
        files.append(f)
    try:                                       # Blender's Python has no PIL: build_knight.block_sheet does it then
        from PIL import Image, ImageDraw
        ref = Image.open(I33).convert("RGB").crop(I33_CROP)
        ims = [Image.open(f).convert("RGB") for f in files]
        h = 600
        ref = ref.resize((round(ref.width * h / ref.height), h))
        ims = [i.resize((round(i.width * h / i.height), h)) for i in ims]
        sheet = Image.new("RGB", (ref.width + sum(i.width for i in ims), h + 28), (24, 24, 28))
        x = 0
        for im, lab in [(ref, "I.33 half-shield (user reference)")] + list(zip(ims, ["block f%d: side (left)" % frame,
                                                                                   "3/4", "front"])):
            sheet.paste(im, (x, 28))
            ImageDraw.Draw(sheet).text((x + 8, 8), lab, fill=(255, 220, 120))
            x += im.width
        sheet.save(os.path.join(C.REN, "knight_%s_block_vs_i33.png" % kind))
    except Exception as e:
        log("block sheet deferred to the orchestrator (%r)" % e)
    play(rig, None, 0)
    return out


STEEL_PIECES = ("greaves", "pauldron_l", "cuirass", "poleyns", "helmet", "vambrace_r")


def steel(kind, pieces=STEEL_PIECES, verbose=True):
    """M3 check (judge iteration 1): rendered value of the plate pieces under the studio light (EEVEE look-dev setup),
    front and 3/4 views. A piece's mask = the pixels where it is visible (every other object set to holdout,
    transparent film); its value = the mean luminance of the render under that mask. Each view is rendered twice:
    with the real materials, and with ONE plain metal material on everything (view-layer material override: base 0.53,
    metallic 1, roughness 0.5), which measures what the light and the shape alone do (the studio key falls off toward
    the floor). material factor = textured value / plain value. Gate (judge M3: greaves and pauldrons read as one
    metal, no near-black greaves next to blown-out pauldrons): greaves / pauldron material-factor ratio within 0.7..1/0.7
    in both views. -> renders/knight_<kind>_steel.json + renders/knight_<kind>_steel_<view>.png"""
    rig = rig_of(kind)
    face_keys_static(rig)
    set_look(rig, "helm")
    play(rig, "idle", 0)
    sc = studio()
    sc.render.resolution_x, sc.render.resolution_y = (600, 860)
    objs = {o.get("rts_part"): o for o in meshes(rig) if o.get("rts_part")}
    everything = [o for o in bpy.context.scene.objects if o.type == 'MESH']
    plain = bpy.data.materials.get("_qa_plain_metal") or bpy.data.materials.new("_qa_plain_metal")
    plain.use_nodes = True
    bs = next(n for n in plain.node_tree.nodes if n.type == 'BSDF_PRINCIPLED')
    bs.inputs["Base Color"].default_value = (0.53, 0.53, 0.53, 1.0)
    bs.inputs["Metallic"].default_value = 1.0
    bs.inputs["Roughness"].default_value = 0.5
    vl = bpy.context.view_layer
    tmp = os.path.join(C.REN, "_steel_tmp.png")

    def shot(path, h, w):
        C.render(path)
        img = bpy.data.images.load(path)
        a = np.array(img.pixels[:], dtype=np.float32).reshape(h, w, 4)
        bpy.data.images.remove(img)
        return a

    rep = {"views": {}}
    for view, (loc, tgt) in {"front": ((0, -3.45, 1.12), (0, 0, 1.0)), "34": ((-2.2, -2.75, 1.35), (0, 0, 1.0))}.items():
        C.camera(loc, tgt, lens=50)
        w, h = sc.render.resolution_x, sc.render.resolution_y
        masks = {}
        sc.render.film_transparent = True
        for p in pieces:
            o = objs.get(p)
            if o is None:
                continue
            for x in everything:
                x.is_holdout = x != o
            m = shot(tmp, h, w)[..., 3] > 0.9
            if m.sum() > 50:
                masks[p] = m
        for x in everything:
            x.is_holdout = False
        sc.render.film_transparent = False
        lum = lambda a: 0.2126 * a[..., 0] + 0.7152 * a[..., 1] + 0.0722 * a[..., 2]
        tex = lum(shot(os.path.join(C.REN, "knight_%s_steel_%s.png" % (kind, view)), h, w))
        vl.material_override = plain
        pl = lum(shot(tmp, h, w))
        vl.material_override = None
        vals = {}
        for p, m in masks.items():
            vals[p] = dict(value=round(float(tex[m].mean()), 3), plain=round(float(pl[m].mean()), 3), px=int(m.sum()))
            vals[p]["material_factor"] = round(vals[p]["value"] / max(vals[p]["plain"], 1e-6), 3)
        top = max((v["value"] for v in vals.values()), default=1.0)
        for v in vals.values():
            v["vs_brightest"] = round(v["value"] / max(top, 1e-6), 2)
        rep["views"][view] = vals
    if os.path.exists(tmp):
        os.remove(tmp)
    pair = []
    for vv in rep["views"].values():
        if "greaves" in vv and "pauldron_l" in vv:
            pair.append(vv["greaves"]["material_factor"] / max(vv["pauldron_l"]["material_factor"], 1e-6))
    rep["gate"] = {"greaves_vs_pauldron_material_factor": [round(x, 2) for x in pair], "range": [0.7, round(1 / 0.7, 2)],
                   "pass": bool(pair) and all(0.7 <= x <= 1 / 0.7 for x in pair)}
    json.dump(rep, open(os.path.join(C.REN, "knight_%s_steel.json" % kind), "w"), indent=1)
    if verbose:
        for view, vv in rep["views"].items():
            log("steel %-5s %s" % (view, "  ".join("%s %.3f (plain %.3f, factor %.2f)" % (p, v["value"], v["plain"],
                                                                                         v["material_factor"])
                                                    for p, v in vv.items())))
        log("steel gate %s (greaves / pauldron material factor %s, within 0.7..1.43)" % (
            "PASS" if rep["gate"]["pass"] else "FAIL", rep["gate"]["greaves_vs_pauldron_material_factor"]))
    play(rig, None, 0)
    return rep


def main(args):
    kind = args[0] if args else "male"
    what = args[1:] or ["exposure", "look", "clips", "face", "cape", "intersect", "strain", "feet", "shield", "block", "steel"]
    if "block" in what:
        block(kind)
    if "steel" in what:
        steel(kind)
    if "strain" in what:
        strain(kind)
    if "feet" in what:
        feet(kind)
    if "shield" in what:
        shield_carry(kind)
    if "exposure" in what:
        exposure(kind)
    if "look" in what:
        look(kind)
    if "clips" in what:
        clips(kind)
    if "face" in what:
        face(kind)
    if "intersect" in what:
        intersections(kind)
    if "cape" in what:
        r = cape_clearance(kind)
        json.dump(r, open(os.path.join(C.REN, "knight_%s_cape_clearance.json" % kind), "w"), indent=1)
