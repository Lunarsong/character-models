"""Look-dev renders of the character texture sets (Blender headless, Cycles).

BLENDER_USER_RESOURCES=$PWD/blender_profile $BL -b --python-exit-code 1 -P scripts/texture_preview.py -- [set ...] [--samples 96] [--res 1200x800]
-> renders/texprev_<set>.png  (a ball + a bent plate / a draped cloth / hair cards, depending on the set kind), lit by
   Blender's bundled studio HDRI + a key light, dark studio background like the reference sheet.
"""
import bpy, bmesh, sys, os, math
from mathutils import Vector, Matrix

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import char_materials as cm

CH = os.path.dirname(HERE)
RENDERS = os.path.join(CH, "renders")
argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []


def opt(name, default):
    if name in argv:
        i = argv.index(name); v = argv[i + 1]; del argv[i:i + 2]; return v
    return default


SAMPLES = int(opt("--samples", "96"))
RX, RY = [int(v) for v in opt("--res", "1200x800").split("x")]
SETS = argv or ["steel_worn", "steel_blued", "gold_worn", "cloth_blue", "leather_brown", "mail_riveted", "horsehair_blue"]


def clear():
    bpy.ops.wm.read_factory_settings(use_empty=True)


def setup_render():
    sc = bpy.context.scene
    sc.render.engine = 'CYCLES'
    sc.cycles.samples = SAMPLES
    sc.cycles.use_denoising = True
    try:
        prefs = bpy.context.preferences.addons['cycles'].preferences
        prefs.compute_device_type = 'METAL'
        prefs.get_devices()
        for d in prefs.devices:
            d.use = True
        sc.cycles.device = 'GPU'
    except Exception as e:
        print("GPU setup failed", e)
    sc.render.resolution_x, sc.render.resolution_y = RX, RY
    sc.render.resolution_percentage = 100
    sc.view_settings.view_transform = 'AgX'
    sc.view_settings.look = 'AgX - Medium High Contrast'
    sc.render.film_transparent = False
    w = bpy.data.worlds.new("W"); sc.world = w; w.use_nodes = True
    nt = w.node_tree; N = nt.nodes; L = nt.links
    for n in list(N):
        N.remove(n)
    out = N.new("ShaderNodeOutputWorld")
    env = N.new("ShaderNodeTexEnvironment")
    hdr = os.path.join(os.path.dirname(bpy.app.binary_path), "..", "Resources",
                       f"{bpy.app.version[0]}.{bpy.app.version[1]}", "datafiles", "studiolights", "world", "studio.exr")
    env.image = bpy.data.images.load(os.path.abspath(hdr))
    mapn = N.new("ShaderNodeMapping"); tc = N.new("ShaderNodeTexCoord")
    mapn.inputs["Rotation"].default_value = (0, 0, math.radians(110))
    L.new(tc.outputs["Generated"], mapn.inputs[0]); L.new(mapn.outputs[0], env.inputs[0])
    bg = N.new("ShaderNodeBackground"); bg.inputs[1].default_value = 0.55
    L.new(env.outputs[0], bg.inputs[0])
    # camera rays see a dark studio backdrop, lighting comes from the HDRI
    lp = N.new("ShaderNodeLightPath")
    bgd = N.new("ShaderNodeBackground"); bgd.inputs[0].default_value = (0.018, 0.018, 0.021, 1); bgd.inputs[1].default_value = 1
    mx = N.new("ShaderNodeMixShader")
    L.new(lp.outputs["Is Camera Ray"], mx.inputs[0]); L.new(bg.outputs[0], mx.inputs[1]); L.new(bgd.outputs[0], mx.inputs[2])
    L.new(mx.outputs[0], out.inputs[0])
    return sc


def light(name, kind, loc, target, energy, size, color=(1, 1, 1)):
    ld = bpy.data.lights.new(name, kind); ld.energy = energy; ld.color = color
    if kind == 'AREA':
        ld.size = size
    o = bpy.data.objects.new(name, ld); bpy.context.scene.collection.objects.link(o)
    o.location = loc
    o.rotation_euler = (Vector(target) - Vector(loc)).to_track_quat('-Z', 'Y').to_euler()
    return o


def camera(loc, target, lens=85):
    cd = bpy.data.cameras.new("Cam"); cd.lens = lens; cd.clip_start = 0.01
    o = bpy.data.objects.new("Cam", cd); bpy.context.scene.collection.objects.link(o)
    o.location = loc; o.rotation_euler = (Vector(target) - Vector(loc)).to_track_quat('-Z', 'Y').to_euler()
    bpy.context.scene.camera = o
    return o


def link(obj):
    bpy.context.scene.collection.objects.link(obj)
    return obj


def mesh_from_bm(name, bm):
    me = bpy.data.meshes.new(name); bm.to_mesh(me); bm.free()
    return link(bpy.data.objects.new(name, me))


def ball(name, r=0.13, loc=(0, 0, 0)):
    bm = bmesh.new(); bm.loops.layers.uv.new("UVMap")
    bmesh.ops.create_uvsphere(bm, u_segments=96, v_segments=48, radius=r, calc_uvs=True)
    o = mesh_from_bm(name, bm)
    for p in o.data.polygons:
        p.use_smooth = True
    o.location = loc
    return o


def bent_plate(name, w=0.34, h=0.24, bend=0.9, loc=(0, 0, 0), thick=0.004, rx=40, ry=28):
    """A curved rectangular plate (pauldron / cuirass-like), UV planar with U along the long side."""
    bm = bmesh.new()
    verts = {}
    for j in range(ry + 1):
        for i in range(rx + 1):
            u = i / rx; v = j / ry
            x = (u - 0.5) * w; y = (v - 0.5) * h
            ang = x / (w * 0.5) * bend * 0.5
            R = w / bend
            X = math.sin(ang) * R; Z = (math.cos(ang) - 1) * R
            Z += -((y / (h * 0.5)) ** 2) * 0.02
            verts[i, j] = bm.verts.new((X, 0, 0))
            verts[i, j].co = Vector((X, Z, y))
    uvl = bm.loops.layers.uv.new("UVMap")
    for j in range(ry):
        for i in range(rx):
            f = bm.faces.new((verts[i, j], verts[i + 1, j], verts[i + 1, j + 1], verts[i, j + 1]))
            for lp, (a, b) in zip(f.loops, ((i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1))):
                lp[uvl].uv = (a / rx * w, b / ry * h)          # 1 UV unit = 1 m, rescaled later
            f.smooth = True
    o = mesh_from_bm(name, bm)
    o.location = loc
    sol = o.modifiers.new("sol", 'SOLIDIFY'); sol.thickness = thick; sol.offset = -1
    bev = o.modifiers.new("bev", 'BEVEL'); bev.width = 0.0015; bev.segments = 3; bev.limit_method = 'ANGLE'
    return o


def drape(name, w=0.5, h=0.7, folds=5, amp=0.03, loc=(0, 0, 0), res=(80, 110), u_range=(0.0, 1.0)):
    bm = bmesh.new(); rx, ry = res; verts = {}
    for j in range(ry + 1):
        for i in range(rx + 1):
            u = i / rx; v = j / ry
            x = (u - 0.5) * w; z = (v - 0.5) * h
            depth = amp * (0.35 + 0.65 * (1 - v)) * math.sin(u * folds * 2 * math.pi + 0.6) \
                + amp * 0.3 * math.sin(u * folds * 4.3 * math.pi + v * 3)
            verts[i, j] = bm.verts.new((x, depth, z))
    uvl = bm.loops.layers.uv.new("UVMap")
    for j in range(ry):
        for i in range(rx):
            f = bm.faces.new((verts[i, j], verts[i + 1, j], verts[i + 1, j + 1], verts[i, j + 1]))
            for lp, (a, b) in zip(f.loops, ((i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1))):
                lp[uvl].uv = (u_range[0] + (u_range[1] - u_range[0]) * a / rx, b / ry)
            f.smooth = True
    o = mesh_from_bm(name, bm); o.location = loc
    sol = o.modifiers.new("sol", 'SOLIDIFY'); sol.thickness = 0.004
    return o


def kite(name, set_name, loc=(0, 0, 0), res=60):
    """Curved kite-shield face with planar UVs matching the shield atlas."""
    import numpy as np
    inf = cm.info(set_name); w, h = inf["panel_m"]
    ol = inf["kite_outline_uv"]
    bm = bmesh.new(); uvl = bm.loops.layers.uv.new("UVMap")
    # grid clipped to the kite outline (point-in-polygon on UV)
    poly = [(u, v) for u, v in ol]
    def inside(u, v):
        c = False
        for i in range(len(poly)):
            (x1, y1), (x2, y2) = poly[i], poly[(i + 1) % len(poly)]
            if (y1 > v) != (y2 > v) and u < (x2 - x1) * (v - y1) / (y2 - y1 + 1e-12) + x1:
                c = not c
        return c
    nx, ny = res, int(res * h / w)
    vid = {}
    for j in range(ny + 1):
        for i in range(nx + 1):
            u, v = i / nx, j / ny
            x = (u - 0.5) * w; z = (v - 0.5) * h
            y = -0.10 * (1 - (2 * u - 1) ** 2) * 0.6          # curved across
            vid[i, j] = (bm.verts.new((x, y, z)), (u, v))
    for j in range(ny):
        for i in range(nx):
            q = [vid[i, j], vid[i + 1, j], vid[i + 1, j + 1], vid[i, j + 1]]
            cu = sum(t[1][0] for t in q) / 4; cv = sum(t[1][1] for t in q) / 4
            if not inside(cu, cv):
                continue
            f = bm.faces.new([t[0] for t in q])
            for lp, t in zip(f.loops, q):
                lp[uvl].uv = t[1]
            f.smooth = True
    for v in [v for v in bm.verts if not v.link_faces]:
        bm.verts.remove(v)
    o = mesh_from_bm(name, bm); o.location = loc
    sol = o.modifiers.new("sol", 'SOLIDIFY'); sol.thickness = 0.012
    return o


def cards(name, set_name, loc=(0, 0, 0), n=70, length=0.46, width=0.06):
    """A horsehair plume: cards rise from a crest holder, arch backwards and fall (like the reference helm plume)."""
    import random
    bm = bmesh.new(); uvl = bm.loops.layers.uv.new("UVMap")
    rnd = random.Random(5)
    cols = cm.info(set_name)["columns"]
    for k in range(n):
        c = cols[1 + rnd.randrange(len(cols) - 1)]
        u0, u1 = c["u"]
        seg = 18
        yaw = rnd.uniform(-0.55, 0.55)
        L = length * rnd.uniform(0.75, 1.1)
        rise = rnd.uniform(0.05, 0.12); back = rnd.uniform(0.18, 0.30); drop = rnd.uniform(0.25, 0.42)
        w = width * rnd.uniform(0.8, 1.3)
        rows = []
        for sgi in range(seg + 1):
            t = sgi / seg
            # centreline: up and back in a fountain arc, then down
            x = math.sin(yaw) * back * t * 0.6
            y = back * math.sin(t * math.pi * 0.5) * (1 + 0.2 * t)
            z = rise * math.sin(t * math.pi) * 1.8 - drop * t * t
            pos = Vector((x, y, z)) * (L / length)
            side = Vector((math.cos(yaw), -math.sin(yaw) * 0.3, 0.15 * math.sin(yaw))) * w * 0.5 * (1 - 0.35 * t)
            rows.append((bm.verts.new(pos - side), bm.verts.new(pos + side), t))
        for sgi in range(seg):
            a0, a1, t0 = rows[sgi]; b0, b1, t1 = rows[sgi + 1]
            f = bm.faces.new((a0, a1, b1, b0))
            for lp, uv in zip(f.loops, ((u0, 1 - t0), (u1, 1 - t0), (u1, 1 - t1), (u0, 1 - t1))):
                lp[uvl].uv = uv
            f.smooth = True
    o = mesh_from_bm(name, bm); o.location = loc
    return o


def plate_point(u, v, w=0.34, h=0.24, bend=0.9):
    x = (u - 0.5) * w; y = (v - 0.5) * h
    ang = x / (w * 0.5) * bend * 0.5; R = w / bend
    return Vector((math.sin(ang) * R, (math.cos(ang) - 1) * R - ((y / (h * 0.5)) ** 2) * 0.02, y))


def plate_normal(u, v, **kw):
    e = 1e-3
    du = plate_point(u + e, v, **kw) - plate_point(u - e, v, **kw)
    dv = plate_point(u, v + e, **kw) - plate_point(u, v - e, **kw)
    return du.cross(dv).normalized()


def trim_band(name, path_uv, width_m, lift=0.0015, n=80, **kw):
    """A strip following the plate surface along path_uv (list of (u, v) params), width_m wide toward the inside.
    UV: U = arc length (m), V = across (m) -> fitted into a trim-sheet strip with char_materials.map_strip."""
    bm = bmesh.new(); uvl = bm.loops.layers.uv.new("UVMap")
    pts = []
    for i in range(n + 1):
        t = i / n
        k = t * (len(path_uv) - 1); j = min(int(k), len(path_uv) - 2); f = k - j
        u = path_uv[j][0] * (1 - f) + path_uv[j + 1][0] * f; v = path_uv[j][1] * (1 - f) + path_uv[j + 1][1] * f
        pts.append((u, v))
    rows = []; s = 0.0; prev = None
    for i, (u, v) in enumerate(pts):
        p = plate_point(u, v, **kw); nrm = plate_normal(u, v, **kw)
        if prev is not None:
            s += (p - prev).length
        prev = p
        # inward direction: toward the plate centre in param space
        cu, cv = 0.5 - u, 0.5 - v
        l = math.hypot(cu, cv) or 1
        q = plate_point(u + cu / l * 0.01, v + cv / l * 0.01, **kw)
        inward = (q - p); inward = (inward - nrm * inward.dot(nrm)).normalized()
        rows.append((bm.verts.new(p + nrm * lift), bm.verts.new(p + nrm * lift + inward * width_m), s))
    for i in range(n):
        a0, a1, s0 = rows[i]; b0, b1, s1 = rows[i + 1]
        fc = bm.faces.new((a0, b0, b1, a1))
        for lp, uv in zip(fc.loops, ((s0, 0), (s1, 0), (s1, width_m), (s0, width_m))):
            lp[uvl].uv = uv
        fc.smooth = True
    o = mesh_from_bm(name, bm)
    sol = o.modifiers.new("sol", 'SOLIDIFY'); sol.thickness = 0.0025; sol.offset = 1
    bev = o.modifiers.new("bev", 'BEVEL'); bev.width = 0.0008; bev.segments = 2
    return o


def combo():
    """Armour look-dev: steel plate + gold filigree trims (map_strip) + rope edge + lion inlay decal; mail sleeve with
    a stitched leather strap; a tabard drape behind."""
    steel = cm.material("steel_worn"); trim = cm.material("trim_gold"); mail = cm.material("mail_riveted")
    straps = cm.material("leather_straps"); tab = cm.material("tabard_lion")
    kw = dict(w=0.34, h=0.24, bend=0.9)
    p = bent_plate("plate", loc=(0, 0, 0)); cm.assign(p, steel); cm.set_texel_density(p, "steel_worn")
    border = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0), (0.0, 0.0)]
    outer = trim_band("trim_outer", border, 0.026, lift=0.0012, n=240, **kw)
    cm.assign(outer, trim); cm.map_strip(outer, "trim_gold", "filigree_wide")
    inner_path = [(0.13, 0.17), (0.87, 0.17), (0.87, 0.83), (0.13, 0.83), (0.13, 0.17)]
    rope = trim_band("trim_rope", inner_path, 0.010, lift=0.0010, n=240, **kw)
    cm.assign(rope, trim); cm.map_strip(rope, "trim_gold", "rope")
    # lion inlay decal panel in the middle
    bm = bmesh.new(); uvl = bm.loops.layers.uv.new("UVMap"); N = 24; vid = {}
    for j in range(N + 1):
        for i in range(N + 1):
            u = 0.36 + 0.28 * i / N; v = 0.24 + 0.52 * j / N
            vid[i, j] = bm.verts.new(plate_point(u, v, **kw) + plate_normal(u, v, **kw) * 0.0003)
    dec = cm.info("trim_gold")["decals"]["lion_inlay_steel"]
    for j in range(N):
        for i in range(N):
            f = bm.faces.new((vid[i, j], vid[i + 1, j], vid[i + 1, j + 1], vid[i, j + 1]))
            for lp, (a, b) in zip(f.loops, ((i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1))):
                lp[uvl].uv = (dec["u"][0] + (dec["u"][1] - dec["u"][0]) * a / N, dec["v"][0] + (dec["v"][1] - dec["v"][0]) * b / N)
            f.smooth = True
    d = mesh_from_bm("lion_decal", bm); cm.assign(d, trim)
    for o in (p, outer, rope, d):
        o.rotation_euler = (0, 0, math.radians(-12)); o.location = (0.16, 0.0, 0.05)
    # mail sleeve + strap
    bm = bmesh.new(); uvl = bm.loops.layers.uv.new("UVMap"); R = 0.055; H = 0.34; nu, nv = 64, 40; vid = {}
    for j in range(nv + 1):
        for i in range(nu + 1):
            a = 2 * math.pi * i / nu; z = H * j / nv - H / 2
            r = R * (1 + 0.12 * math.sin(j / nv * math.pi))
            vid[i, j] = bm.verts.new((math.cos(a) * r, math.sin(a) * r, z))
    for j in range(nv):
        for i in range(nu):
            f = bm.faces.new((vid[i, j], vid[i + 1, j], vid[i + 1, j + 1], vid[i, j + 1]))
            for lp, (a, b) in zip(f.loops, ((i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1))):
                lp[uvl].uv = (a / nu * 2 * math.pi * R, b / nv * H)
            f.smooth = True
    for v in bm.verts:
        pass
    sleeve = mesh_from_bm("sleeve", bm); cm.assign(sleeve, mail); cm.set_texel_density(sleeve, "mail_riveted")
    sleeve.location = (-0.24, 0.02, 0.0)
    bm = bmesh.new(); uvl = bm.loops.layers.uv.new("UVMap"); nu = 64; vid = {}; W_ = 0.047; R2 = R * 1.14 + 0.004
    for j in range(2):
        for i in range(nu + 1):
            a = 2 * math.pi * i / nu
            vid[i, j] = bm.verts.new((math.cos(a) * R2, math.sin(a) * R2, 0.04 + j * W_))
    for i in range(nu):
        f = bm.faces.new((vid[i, 0], vid[i + 1, 0], vid[i + 1, 1], vid[i, 1]))
        for lp, (a, b) in zip(f.loops, ((i, 0), (i + 1, 0), (i + 1, 1), (i, 1))):
            lp[uvl].uv = (a / nu * 2 * math.pi * R2, b * W_)
        f.smooth = True
    strap = mesh_from_bm("strap", bm); cm.assign(strap, straps); cm.map_strip(strap, "leather_straps", "belt_wide")
    sol = strap.modifiers.new("sol", 'SOLIDIFY'); sol.thickness = 0.004; sol.offset = 1
    strap.location = (-0.24, 0.02, 0.0)
    t = drape("tabard", w=0.40, h=1.05, folds=3, amp=0.02, loc=(0.02, 0.45, 0.0), res=(60, 140), u_range=(0.0, 0.5))
    cm.assign(t, tab)
    camera((-0.02, -1.05, 0.22), (-0.02, 0, 0.02), lens=60)


def preview(set_name):
    clear(); sc = setup_render()
    close = set_name.endswith("_close")
    if close:
        set_name = set_name[:-6]
    kind = cm.info(set_name).get("kind") if set_name != "combo" else "combo"
    mat = cm.material(set_name) if set_name != "combo" else None
    light("Key", 'AREA', (-0.9, -1.2, 1.1), (0, 0, 0.0), 120, 0.8, (1.0, 0.96, 0.9))
    light("Rim", 'AREA', (1.0, 1.0, 0.8), (0, 0, 0), 90, 0.6, (0.85, 0.9, 1.0))
    if kind == "combo":
        combo()
    elif kind == "hair_cards":
        o = cards("cards", set_name, loc=(0, -0.1, 0.12)); cm.assign(o, mat)
        camera((1.6, 0.35, 0.05), (0, 0.12, -0.02), lens=55)
    elif set_name == "tabard_lion":
        w, h = cm.info(set_name)["panel_m"]
        a = drape("front", w=w, h=h, folds=3, amp=0.02, loc=(-0.25, 0, 0), res=(60, 140), u_range=(0.0, 0.5))
        b = drape("back", w=w, h=h, folds=3, amp=0.02, loc=(0.25, 0, 0), res=(60, 140), u_range=(0.5, 1.0))
        cm.assign(a, mat); cm.assign(b, mat)
        camera((0.0, -3.2, 0.1), (0, 0, 0.0), lens=50)
        if close:        # cutscene close-up on the front emblem (~0.3 m field of view)
            camera((-0.25, -0.62, 0.26), (-0.25, 0, 0.26), lens=85)
    elif set_name == "cape_lion":
        mat = cm.material(set_name, alpha_mode="MASK")
        w, h = cm.info(set_name)["panel_m"]
        a = drape("cape", w=w, h=h, folds=4, amp=0.035, loc=(0, 0, 0), res=(120, 180)); cm.assign(a, mat)
        camera((0.0, -3.6, 0.1), (0, 0, 0.0), lens=50)
    elif set_name == "shield_lion":
        a = kite("shield", set_name, loc=(0, 0, 0)); cm.assign(a, mat)
        camera((0.35, -2.4, 0.25), (0, 0, 0.0), lens=50)
    elif kind == "trim_sheet":
        # the whole sheet at its native scale (0.5 m square), bent a little so the relief catches the light
        bm = bmesh.new(); uvl = bm.loops.layers.uv.new("UVMap"); N = 64; vid = {}
        for j in range(N + 1):
            for i in range(N + 1):
                u, v = i / N, j / N
                vid[i, j] = bm.verts.new(((u - 0.5) * 0.5, -0.06 * math.sin(u * math.pi), (v - 0.5) * 0.5))
        for j in range(N):
            for i in range(N):
                f = bm.faces.new((vid[i, j], vid[i + 1, j], vid[i + 1, j + 1], vid[i, j + 1]))
                for lp, (a, b) in zip(f.loops, ((i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1))):
                    lp[uvl].uv = (a / N, b / N)
                f.smooth = True
        o = mesh_from_bm("sheet", bm); cm.assign(o, mat)
        camera((0.0, -1.35, 0.05), (0, 0, 0.0), lens=50)
    elif set_name.startswith("cloth") or kind == "atlas":
        o = drape("cloth", loc=(0.02, 0.0, 0)); cm.assign(o, mat)
        if kind == "tileable":
            cm.set_texel_density(o, set_name)
        b = ball("ball", 0.1, loc=(-0.36, -0.05, -0.12)); cm.assign(b, mat); cm.set_texel_density(b, set_name)
        camera((-0.25, -1.55, 0.18), (-0.1, 0, -0.02), lens=55)
    else:
        b = ball("ball", 0.12, loc=(-0.2, 0.0, 0.0)); cm.assign(b, mat); cm.set_texel_density(b, set_name)
        p = bent_plate("plate", loc=(0.18, 0.02, 0.0)); cm.assign(p, mat); cm.set_texel_density(p, set_name)
        p.rotation_euler = (0, 0, math.radians(-15))
        camera((-0.05, -1.25, 0.28), (0.0, 0, 0.0), lens=65)
    # ground
    bm = bmesh.new(); bmesh.ops.create_grid(bm, x_segments=1, y_segments=1, size=3.0)
    g = mesh_from_bm("ground", bm); g.location = (0, 0, -0.26 if kind != "atlas" else -0.75)
    gm = bpy.data.materials.new("ground"); gm.use_nodes = True
    gb = gm.node_tree.nodes["Principled BSDF"]; gb.inputs["Base Color"].default_value = (0.02, 0.02, 0.022, 1)
    gb.inputs["Roughness"].default_value = 0.7
    g.data.materials.append(gm)
    out = os.path.join(RENDERS, f"texprev_{set_name}{'_close' if close else ''}.png")
    sc.render.filepath = out
    bpy.ops.render.render(write_still=True)
    print("PREVIEW", out)


for s in SETS:
    preview(s)
