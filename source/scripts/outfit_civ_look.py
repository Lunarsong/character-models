"""Look-dev renders for the civilian outfits (EEVEE, the kit's dark studio: same lights as armour_upper.lookdev_setup,
copied here so this script never imports a module another agent is editing).

  look   on a dressed export file (out/civ/<outfit>_<kind>_export.blend): idle frame 0, turnaround + close-ups
         -> renders/civ/look_<outfit>_<kind>_<view>.png
  lineup (no file: builds a scene) knight / peasant / archer (+ the mix-and-match outfits) side by side, appended from
         their export files, hero scale (full body, 3/4, back) and RTS scale (the viewers' RTS camera: ~55 deg down,
         30 deg lens, units 1.4 m apart) -> renders/civ/lineup_<kind>_<view>.png

run: Blender -b out/civ/peasant_male_export.blend --python-exit-code 1 -P scripts/outfit_civ_look.py -- look male peasant
     Blender -b --python-exit-code 1 -P scripts/outfit_civ_look.py -- lineup male [female]
"""
import sys, os, math, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from chr_lib import *
from mathutils import Vector, Matrix

RENC = os.path.join(REN, "civ")


def studio(res=(900, 1300), samples=48, floor=True):
    sc = render_setup("BLENDER_EEVEE", res, samples, world=1.0)
    w = sc.world
    bg = w.node_tree.nodes.get("Background")
    bg.inputs[0].default_value = (0.055, 0.058, 0.066, 1)
    bg.inputs[1].default_value = 1.0
    sc.view_settings.look = 'AgX - Medium High Contrast'
    sc.view_settings.exposure = 0.0
    try:
        sc.eevee.use_raytracing = True
        sc.eevee.ray_tracing_options.resolution_scale = '1'
        sc.eevee.use_shadows = True
    except Exception:
        pass
    for o in [o for o in bpy.data.objects if o.type == 'LIGHT']:
        bpy.data.objects.remove(o, do_unlink=True)
    add_light("Key", 'AREA', (-2.4, -2.6, 3.2), (50, 0, -42), 700, 2.5, (1.0, 0.95, 0.88))
    add_light("Rim", 'AREA', (1.6, 2.8, 2.9), (-58, 0, 150), 900, 1.5, (0.75, 0.85, 1.0))
    add_light("Rim2", 'AREA', (-2.2, 2.2, 2.2), (-60, 0, -140), 400, 1.5, (1.0, 0.9, 0.8))
    add_light("Fill", 'AREA', (2.8, -2.6, 1.4), (75, 0, 48), 180, 3.0, (0.85, 0.9, 1.0))
    if floor and not bpy.data.objects.get("Floor"):
        bpy.ops.mesh.primitive_plane_add(size=60, location=(0, 0, 0))
        fl = bpy.context.object; fl.name = "Floor"
        set_material(fl, pbr_material("M_floor", base_color=(0.03, 0.03, 0.033, 1), rough=0.85))
    return sc


def daylight(res=(900, 900), samples=32):
    """outdoor sun + sky for the RTS shots (reads like the town viewer: warm sun, blue ambient, grass-ish ground)"""
    sc = render_setup("BLENDER_EEVEE", res, samples, world=1.0)
    bg = sc.world.node_tree.nodes.get("Background")
    bg.inputs[0].default_value = (0.42, 0.52, 0.66, 1)
    bg.inputs[1].default_value = 0.9
    sc.view_settings.look = 'AgX - Medium High Contrast'
    for o in [o for o in bpy.data.objects if o.type == 'LIGHT']:
        bpy.data.objects.remove(o, do_unlink=True)
    add_light("Sun", 'SUN', (0, 0, 10), (48, 0, -38), 4.2, 2.0, (1.0, 0.94, 0.84))
    fl = bpy.data.objects.get("Floor")
    if fl is None:
        bpy.ops.mesh.primitive_plane_add(size=60, location=(0, 0, 0))
        fl = bpy.context.object; fl.name = "Floor"
    set_material(fl, pbr_material("M_ground", base_color=(0.16, 0.15, 0.10, 1), rough=0.95))
    return sc


def idle(rig, frame=0):
    if rig.animation_data and bpy.data.actions.get("idle"):
        rig.animation_data.action = bpy.data.actions["idle"]
    bpy.context.scene.frame_set(frame)
    bpy.context.view_layer.update()


def hide_alts(rig):
    for o in rig.children:
        if o.type != 'MESH':
            continue
        if (o.get("rts_variant_group") and not o.get("rts_default")) or o.get("rts_alt"):
            o.hide_render = True


VIEWS = {  # name: (camera, target, lens, res, yaw)
    "front": ((0, -3.45, 1.08), (0, 0, 0.98), 50, (900, 1300), 0),
    "34": ((-2.2, -2.75, 1.35), (0, 0, 0.98), 50, (900, 1300), 0),
    "side": ((0, -3.45, 1.08), (0, 0, 0.98), 50, (900, 1300), 90),
    "back": ((0, -3.45, 1.08), (0, 0, 0.98), 50, (900, 1300), 180),
    "back34": ((-2.2, -2.75, 1.35), (0, 0, 0.98), 50, (900, 1300), 150),
    "upper": ((-0.75, -1.35, 1.6), (0, -0.02, 1.38), 55, (900, 1000), 0),
    "head": ((-0.55, -0.85, 1.8), (0, -0.02, 1.66), 70, (900, 1000), 0),
    "waist": ((0.55, -1.2, 1.05), (0.02, -0.04, 0.95), 55, (900, 1000), 0),
    "legs": ((-0.5, -1.7, 0.5), (0.0, 0, 0.42), 45, (900, 1000), 0),
    "backgear": ((0.9, 1.45, 1.55), (0.0, 0.05, 1.2), 50, (900, 1000), 0),
    "neck": ((-0.25, -0.62, 1.62), (0, -0.02, 1.52), 60, (900, 800), 0),
    "sleeve": ((-0.75, -0.55, 1.35), (-0.33, -0.05, 1.28), 60, (900, 800), 0),
    "rts": ((-5.5, -8.0, 10.5), (0, 0, 0.9), 60, (480, 480), 0),
}


def look(kind, outfit, shots=None, pose_clip="idle", frame=0):
    rig = bpy.data.objects["rts_" + kind]
    hide_alts(rig)
    studio()
    if pose_clip:
        if rig.animation_data and bpy.data.actions.get(pose_clip):
            rig.animation_data.action = bpy.data.actions[pose_clip]
        bpy.context.scene.frame_set(frame)
    z_scale = 1.0
    for nm, (loc, tgt, lens, res, yaw) in VIEWS.items():
        if shots and nm not in shots:
            continue
        sc = bpy.context.scene
        sc.render.resolution_x, sc.render.resolution_y = res
        rig.rotation_euler.z = math.radians(yaw)
        bpy.context.view_layer.update()
        camera(loc, tgt, lens=lens)
        render(os.path.join(RENC, "look_%s_%s_%s.png" % (outfit, kind, nm)))
    rig.rotation_euler.z = 0.0


def clip_strip(kind, outfit, clip, frames, cam="34", res=(520, 760)):
    """frames of a clip (EEVEE) -> renders/civ/clip_<outfit>_<kind>_<clip>_f###_<cam>.png (+ a sheet)"""
    rig = bpy.data.objects["rts_" + kind]
    hide_alts(rig)
    studio(res=res, samples=24)
    cams = {"34": ((-2.2, -2.75, 1.35), (0, 0, 0.98)), "left34": ((2.4, -2.6, 1.45), (0, -0.1, 1.05)),
            "side": ((3.3, -0.3, 1.2), (0, 0, 1.0)), "front": ((0, -3.45, 1.1), (0, 0, 0.98))}
    loc, tgt = cams[cam]
    camera(loc, tgt, lens=50)
    rig.animation_data.action = bpy.data.actions[clip]
    files = []
    for f in frames:
        bpy.context.scene.frame_set(f)
        out = os.path.join(RENC, "clip_%s_%s_%s_f%03d_%s.png" % (outfit, kind, clip, f, cam))
        render(out); files.append(out)
    return files


# ------------------------------------------------------------------------------------------------ lineup
def append_character(blend, tag, x, yaw=0.0, clip="idle", frame=0):
    """append every object of an export .blend (rig + meshes) and its actions, move the rig to x"""
    with bpy.data.libraries.load(blend, link=False) as (src, dst):
        dst.objects = list(src.objects)
        dst.actions = list(src.actions)
    rig = None
    for o in dst.objects:
        if o is None:
            continue
        if o.type in ('MESH', 'ARMATURE'):
            bpy.context.scene.collection.objects.link(o)
            o.name = tag + "_" + o.name
            if o.type == 'ARMATURE':
                rig = o
        # cameras / lights of the source file are dropped
    acts = {a.name: a for a in dst.actions if a}
    for o in bpy.context.scene.collection.objects:
        if o.type == 'MESH' and o.parent is None and any(m.type == 'ARMATURE' and m.object == rig for m in o.modifiers):
            o.parent = rig
    rig.location.x = x
    rig.rotation_euler.z = math.radians(yaw)
    if clip:
        cand = [a for n, a in acts.items() if n == clip or n.startswith(clip + ".")]
        if cand:
            rig.animation_data_create()
            rig.animation_data.action = cand[0]
    hide_alts(rig)
    for o in rig.children:
        if o.type == 'MESH' and o.get("rts_look") == "bare":
            o.hide_render = True                   # knight: game (helm) look
    return rig


LINEUP = [("knight", "out/{kind}_export"), ("peasant", "civ"), ("archer", "civ"), ("peasant_hood", "civ"),
          ("militia", "civ")]


def lineup(kinds):
    for o in list(bpy.data.objects):
        bpy.data.objects.remove(o, do_unlink=True)
    rigs = []
    k0 = kinds[0]
    ents = []
    for kind in kinds:
        for name, where in LINEUP:
            p = os.path.join(OUT, "knight_%s_export.blend" % kind) if name == "knight" else \
                os.path.join(OUT, "civ", "%s_%s_export.blend" % (name, kind))
            if os.path.exists(p):
                ents.append((name, kind, p))
    n = len(ents)
    gap = 1.25
    for i, (name, kind, p) in enumerate(ents):
        x = (i - (n - 1) / 2) * gap
        rigs.append((name, kind, append_character(p, "%s_%s" % (name, kind), x)))
    bpy.context.scene.frame_set(0)
    bpy.context.view_layer.update()
    tag = "_".join(kinds)
    width = n * gap
    studio(res=(int(360 * n), 1100))
    for nm, loc, tgt, lens in (("front", (0, -2.9 - width * 0.95, 1.1), (0, 0, 0.95), 45),
                               ("34", (-width * 0.55, -2.4 - width * 0.8, 1.45), (0, 0, 0.92), 45),
                               ("back", (0, 2.9 + width * 0.95, 1.1), (0, 0, 0.95), 45)):
        camera(loc, tgt, lens=lens)
        render(os.path.join(RENC, "lineup_%s_%s.png" % (tag, nm)))
    # RTS scale: the building viewers' RTS camera (work/mkviewer.py: ~55 deg down, ~37 m away, 30 deg vertical field of
    # view) over the village lane the viewers use, in daylight; + the 'RTS zoom' camera (half the distance)
    town = os.path.normpath(os.path.join(CH, "..", "work", "out", "village_lane.glb"))
    fl = bpy.data.objects.get("Floor")
    if os.path.exists(town):
        before = set(bpy.data.objects)
        bpy.ops.import_scene.gltf(filepath=town)
        new = [o for o in bpy.data.objects if o not in before]
        for o in new:
            if o.parent is None:
                o.location.z -= 0.8                    # the viewers stand the character on the lane (TOWN offset)
        if fl:
            fl.hide_render = True
    W_, H_ = 1600, 900
    daylight(res=(W_, H_))
    if fl and os.path.exists(town):
        fl.hide_render = True
    vfov = math.radians(30.0)
    lens = 18.0 * (H_ / W_) / math.tan(vfov / 2)       # 36 mm sensor fitted horizontally -> 30 deg vertical
    for nm, d in (("rts", 37.0), ("rts_zoom", 18.5)):
        el = math.radians(55); az = math.radians(-30)
        cam = (math.sin(az) * math.cos(el) * d, -math.cos(az) * math.cos(el) * d, math.sin(el) * d + 0.9)
        camera(cam, (0, 0, 0.9), lens=lens)
        f = os.path.join(RENC, "lineup_%s_%s.png" % (tag, nm))
        render(f)
        # (the 3x crops of the units are cut by outfit_civ.sh with system python / PIL)


if __name__ == "__main__":
    a = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    mode = a[0]
    if mode == "look":
        look(a[1], a[2], shots=a[3:] or None)
    elif mode == "clip":
        # clip <kind> <outfit> <clip> <cam> f0 f1 ...
        clip_strip(a[1], a[2], a[3], [int(x) for x in a[5:]], cam=a[4])
    elif mode == "lineup":
        lineup(a[1:] or ["male"])
