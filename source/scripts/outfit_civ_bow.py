"""The archer's LONGBOW: a yew self-bow prop with its own bones and sockets (Blender Python + numpy).

Bow-local frame (= the frame of socket_hand_l, grip centre, M15-canonical): +Y along the stave (upper limb), +Z = the
bow's back (toward the target, the knuckle side of the bow hand), X = Y x Z. The string is on the -Z side.
  stave     1.80 m, D-section (flat back, round belly), 30 x 36 mm at the handle tapering to 13 x 11 mm, braced (tips
            14 cm toward the string side), leather grip wrap, horn nocks with a string groove
  string    linen, served (thicker, stitched) 10 cm round the nocking point, a nocking-point bead
Bones (all children of socket_hand_l when dressed; the prop's own skeleton otherwise):
  bow_root (grip = attach point, identity to socket_hand_l), bow_limb_u1/u2 and bow_limb_l1/l2 (the limbs bend toward
  the archer when drawn), bow_string (the nocking point: follows the drawing hand; = the arrow-nock socket),
  bow_socket_arrow (non-deform: the arrow rest just above the bow hand).
Draw rule (evaluated in the clips, stored in the armature extras rts_bow): pull = distance of bow_string from its rest
point along -Z; limb angles u1 = l1 = 9 deg * pull / 0.72 m, u2 = l2 = 14 deg * pull / 0.72 m (tips toward -Z).
"""
import os, math, json
import numpy as np
import bpy
from mathutils import Vector, Matrix, Quaternion
from outfit_civ_lib import CMesh, nrm, smoothstep, lerp, v3, norm_w, clog, material_for, apply_slots, OUT
from outfit_civ_geo import tube_path, lathe

LEN = 1.80
HALF = LEN / 2
TIP_Z = -0.14                   # brace: tips bent toward the string side
HANDLE = 0.07
DRAW_REF = 0.72                 # full draw pull (m) for the reference limb angles
BEND = {"bow_limb_u1": 9.0, "bow_limb_u2": 14.0, "bow_limb_l1": 9.0, "bow_limb_l2": 14.0}
STRING_Z = TIP_Z - 0.012         # string line (in the nock grooves)


def zc(y):
    return TIP_Z * (abs(y) / HALF) ** 2 * smoothstep(HANDLE * 0.5, HANDLE * 1.5, abs(y)) ** 0.5


def section(y):
    """(width along X, thickness along Z) of the stave at y"""
    a = abs(y)
    if a < HANDLE:
        return 0.030, 0.036
    t = (a - HANDLE) / (HALF - HANDLE)
    return lerp(0.030, 0.013, t ** 0.9), lerp(0.034, 0.011, t ** 0.8)


BONES = {  # name: (parent, head (bow-local), tail, deform)
    "bow_root": ("socket_hand_l", (0, 0, 0), (0, 0.10, 0), True),
    "bow_limb_u1": ("bow_root", (0, HANDLE, 0), (0, 0.45, None), True),
    "bow_limb_u2": ("bow_limb_u1", (0, 0.45, None), (0, HALF, None), True),
    "bow_limb_l1": ("bow_root", (0, -HANDLE, 0), (0, -0.45, None), True),
    "bow_limb_l2": ("bow_limb_l1", (0, -0.45, None), (0, -HALF, None), True),
    "bow_string": ("bow_root", (0, 0, STRING_Z), (0, 0.10, STRING_Z), True),
    "bow_socket_arrow": ("bow_root", (0, 0.034, 0.0), (0, 0.034, 0.10), False),
    # the nocked arrow: on the nocking point, aimed at the arrow rest (keyed in the clips: scale ~0 when not nocked)
    "bow_arrow": ("bow_string", (0, 0.004, STRING_Z), (0, 0.034, 0.0), True),
}
ARROW_LEN = 0.76


def _pt(p):
    x, y, z = p
    return np.array([x, y, zc(y) if z is None else z], float)


def bone_local():
    """{name: (parent, head, y axis, z axis, deform)} in bow-local coordinates"""
    out = {}
    for n, (par, h, t, dfm) in BONES.items():
        H, T = _pt(h), _pt(t)
        y = nrm(T - H)
        z = np.array([0, 0, 1.0]) if abs(y[2]) < 0.9 else np.array([0, -1.0, 0])
        z = nrm(z - y * np.dot(z, y))
        out[n] = (par, H, y, z, dfm, float(np.linalg.norm(T - H)))
    return out


def limb_weights(y):
    a = abs(y)
    s = "u" if y >= 0 else "l"
    w1 = smoothstep(HANDLE - 0.02, HANDLE + 0.03, a)
    w2 = smoothstep(0.40, 0.50, a)
    return norm_w({"bow_root": 1 - w1, "bow_limb_%s1" % s: w1 * (1 - w2), "bow_limb_%s2" % s: w1 * w2})


def bow_mesh(nring=10, nst=44):
    """the bow in bow-local coordinates (CMesh with bone weights and slots stave / grip / horn / string)"""
    m = CMesh()
    ys = np.concatenate([np.linspace(-HALF + 0.025, -HANDLE - 0.004, nst // 2),
                         np.linspace(-HANDLE, HANDLE, 7), np.linspace(HANDLE + 0.004, HALF - 0.025, nst // 2)])
    rings = []
    for y in ys:
        w, t = section(y)
        c = np.array([0, y, zc(y)])
        ring = []
        for k in range(nring):
            a = 2 * math.pi * k / nring
            # D-section: flat back (+Z half squashed), round belly (-Z)
            ca, sa = math.cos(a), math.sin(a)
            zz = t / 2 * (sa if sa < 0 else sa ** 3 * 0.55 + 0.45 * sa ** 0.5 * 0.0 + 0.45 * sa)
            xx = w / 2 * ca * (1.0 if sa < 0 else 0.92 + 0.08 * (1 - sa))
            ring.append(m.add_v(c + np.array([xx, 0, zz]), limb_weights(y)))
        rings.append(ring)
    al = np.concatenate([[0], np.cumsum(np.abs(np.diff(ys)))])
    for i in range(len(rings) - 1):
        slot = "grip" if abs(ys[i] + ys[i + 1]) / 2 < HANDLE - 0.002 else "stave"
        for k in range(nring):
            k2 = (k + 1) % nring
            u0, u1 = k / nring * 0.08, (k + 1) / nring * 0.08
            m.add_f([rings[i][k], rings[i][k2], rings[i + 1][k2], rings[i + 1][k]],
                    [(u0, al[i]), (u1, al[i]), (u1, al[i + 1]), (u0, al[i + 1])], slot)
    # horn nocks: bulb + groove + point, closing the stave ends
    for sgn, ring, y0 in ((1, rings[-1], ys[-1]), (-1, rings[0], ys[0])):
        prev = ring
        w, t = section(y0)
        prof = [(1.08, 0.006), (1.15, 0.012), (0.8, 0.016), (0.95, 0.019), (0.7, 0.024), (0.0, 0.03)]
        for sc_, dy in prof:
            y = y0 + sgn * dy
            c = np.array([0, y, zc(y0) + (zc(y0) - zc(y0 - sgn * 0.01)) * dy / 0.01])
            if sc_ == 0.0:
                tipv = m.add_v(c, limb_weights(y0))
                for k in range(nring):
                    k2 = (k + 1) % nring
                    vs = [prev[k], prev[k2], tipv] if sgn > 0 else [prev[k2], prev[k], tipv]
                    m.add_f(vs, [(0, 0), (0.01, 0), (0.005, 0.01)], "horn")
                break
            cur = []
            for k in range(nring):
                a = 2 * math.pi * k / nring
                cur.append(m.add_v(c + np.array([w / 2 * math.cos(a) * sc_, 0, t / 2 * math.sin(a) * sc_]),
                                   limb_weights(y0)))
            for k in range(nring):
                k2 = (k + 1) % nring
                vs = [prev[k], prev[k2], cur[k2], cur[k]] if sgn > 0 else [prev[k2], prev[k], cur[k], cur[k2]]
                m.add_f(vs, [(0, 0), (0.01, 0), (0.01, 0.01), (0, 0.01)], "horn")
            prev = cur
    if True:
        # the handle faces were wound with the stave; the lower nock loop above builds on rings[0] in reverse order
        pass
    # string: upper groove -> nocking point -> lower groove; weights blend tip bone -> bow_string
    y_n = HALF - 0.012
    top = np.array([0, y_n, STRING_Z]); bot = np.array([0, -y_n, STRING_Z]); mid = np.array([0, 0.0, STRING_Z])
    nseg = 14
    for a_, b_, tipb in ((top, mid, "bow_limb_u2"), (mid, bot, "bow_limb_l2")):
        path = [lerp(a_, b_, t) for t in np.linspace(0, 1, nseg + 1)]
        rad = [0.0016 if abs(p[1]) > 0.06 else 0.0022 for p in path]

        def wfn(p, tipb=tipb):
            u = 1 - abs(p[1]) / y_n                     # 0 at the nock, 1 at the nocking point
            return norm_w({tipb: 1 - u, "bow_string": u})
        tube_path(m, np.array(path), rad, 6, "trim:cord", wfn, cap=True, twist_ref=(0, 0, 1))
    # nocking point bead
    tube_path(m, np.array([mid + np.array([0, 0.006, 0]), mid + np.array([0, 0.014, 0])]), 0.0032, 6, "horn",
              lambda p: {"bow_string": 1.0}, cap=True)
    return m


SETS = {"stave": "civ_wood", "grip": "civ_leather_dark", "horn": "civ_leather_tan", "trim": "civ_trim",
        "iron": "civ_iron", "atlas": "civ_fletch"}


def final_uv(m):
    """bow UVs -> texture UVs: trim strips (tubes: u along the length, v around), atlas as is, tileables x uv_per_m"""
    import outfit_civ_lib as CL
    strips = CL.civ_manifest().get("civ_trim", {}).get("strips", {})
    out = []
    for u_, sl in zip(m.UV, m.S):
        if sl.startswith("trim:"):
            v0, v1 = strips[sl.split(":")[1]]["v"]; mu = strips[sl.split(":")[1]]["m_per_u"]
            out.append([(b / mu, v0 + (v1 - v0) * ((a / (2 * math.pi * 0.0022)) % 1.0)) for a, b in u_])
        elif sl.startswith("atlas:"):
            out.append(list(u_))
        else:
            k = CL.uv_scale(SETS[sl])
            out.append([(a * k, b * k) for a, b in u_])
    return out


def arrow_mesh():
    """the nocked arrow in bow-local coordinates along the bow_arrow bone (nock at the string, head past the bow)"""
    bl = bone_local()
    par, H, y, z, dfm, ln = bl["bow_arrow"]
    x = np.cross(y, z)
    m = CMesh()
    w = {"bow_arrow": 1.0}
    p0 = H; p1 = H + y * ARROW_LEN
    tube_path(m, np.array([p0 + y * 0.004, p1 - y * 0.03]), 0.0043, 6, "stave", lambda p: dict(w), cap=True,
              twist_ref=tuple(z))
    tube_path(m, np.array([p0 - y * 0.004, p0 + y * 0.012]), [0.0038, 0.0047], 6, "horn", lambda p: dict(w), cap=True,
              twist_ref=tuple(z))
    tube_path(m, np.array([p0 + y * 0.14, p0 + y * 0.155]), [0.0048, 0.0048], 6, "trim:fletch_wrap", lambda p: dict(w),
              cap=True, twist_ref=tuple(z))
    lathe(m, p1 - y * 0.032, y, x, [(0.0045, 0.0), (0.0062, 0.012), (0.0048, 0.024), (0.0, 0.034)], 6, "iron", w=w,
          close_bottom=True)
    for q in range(3):
        a = math.pi / 2 + q * 2 * math.pi / 3
        vd = x * math.cos(a) + z * math.sin(a)
        f0 = p0 + y * 0.018; f1 = p0 + y * 0.135
        vs = [m.add_v(f0, dict(w)), m.add_v(f1, dict(w)), m.add_v(f1 + vd * 0.016, dict(w)), m.add_v(f0 + vd * 0.013, dict(w))]
        u0 = q / 3.0
        m.add_f(vs, [(u0 + 0.027, 0.0), (u0 + 0.027, 1.0), (u0 + 0.30, 1.0), (u0 + 0.30, 0.0)], "atlas:fletch")
    return m


def bow_specs(rig):
    """extra bones for outfit_civ_lib.ensure_sockets: armature-space heads / axes from the socket_hand_l rest frame
    (canonical rotation, body-fitted position)"""
    import outfit_civ_lib as CL
    specs, frames = CL.civ_socket_specs(rig)
    s = specs["socket_hand_l"]
    Y = nrm(s["y"]); Z = nrm(s["z"] - Y * np.dot(s["z"], Y)); X = np.cross(Y, Z)
    R3 = np.stack([X, Y, Z], 1)                  # bow-local -> armature
    O = np.array(s["head"], float)
    out = {}
    for n, (par, H, y, z, dfm, ln) in bone_local().items():
        out[n] = dict(parent=par, head=O + R3 @ H, y=R3 @ y, z=R3 @ z, deform=dfm, length=ln)
    return out, (O, R3)


def _to_object(m, name, rig, O=None, R3=None):
    if O is not None:
        P = np.array(m.V) @ R3.T + O
        m.V = [tuple(p) for p in P]
    if m.mixed():
        m.triangulate()
    m.UV = final_uv(m)
    slots = sorted(set(s.split(":")[0] for s in m.S))
    ob = m.to_object(name, slots=sorted(set(m.S)))
    ob.data.materials.clear()
    for s in slots:
        ob.data.materials.append(material_for(SETS[s]))
    ob.data.polygons.foreach_set("material_index", np.array([slots.index(s.split(":")[0]) for s in m.S], np.int32))
    ob.data.update()
    for i, w in enumerate(m.W):
        for b, x in w.items():
            g = ob.vertex_groups.get(b) or ob.vertex_groups.new(name=b)
            g.add([i], x, 'REPLACE')
    md = ob.modifiers.new("Armature", 'ARMATURE'); md.object = rig
    ob.parent = rig
    ob.matrix_parent_inverse = Matrix.Identity(4)
    ca = ob.data.color_attributes.new("Color", 'BYTE_COLOR', 'POINT')
    ca.data.foreach_set("color", np.ones(len(ob.data.vertices) * 4))
    return ob


def add_bow(rig, kind):
    """the bow + the nocked arrow, skinned to the bow bones (already in the rig), materials from the civ sets"""
    specs, (O, R3) = bow_specs(rig)
    ob = _to_object(bow_mesh(), "%s_longbow" % kind, rig, O, R3)
    ob["rts_part"] = "longbow"; ob["rts_prop"] = "longbow"; ob["rts_asset"] = "rts_archer_longbow"
    ob["rts_outfit"] = "archer"; ob["rts_socket"] = "socket_hand_l"; ob["slot"] = "Props"
    ar = _to_object(arrow_mesh(), "%s_arrow" % kind, rig, O, R3)
    ar["rts_part"] = "arrow"; ar["rts_prop"] = "arrow"; ar["rts_asset"] = "rts_archer_arrow"
    ar["rts_outfit"] = "archer"; ar["rts_socket"] = "bow_arrow"; ar["slot"] = "Props"
    rig["rts_bow"] = json.dumps({"doc": "longbow draw rule (outfit_civ_bow.py): pull = distance of bow_string from its "
                                        "rest point along the bow's -Z; limb bone angle = deg * pull / draw_ref, tips "
                                        "toward -Z (the archer). bow_arrow: the nocked arrow (scale ~0 in the clips "
                                        "while no arrow is nocked; spawn the projectile at the clip's 'release' event)",
                                 "draw_ref_m": DRAW_REF, "deg": BEND, "string_rest_local": [0, 0, STRING_Z],
                                 "attach": "bow_root = socket_hand_l (identity)",
                                 "sockets": {"bow_string": "arrow nock (nocking point)", "bow_socket_arrow": "arrow rest"}})
    clog("longbow: %d tris, arrow %d tris" % (sum(len(p.vertices) - 2 for p in ob.data.polygons),
                                              sum(len(p.vertices) - 2 for p in ar.data.polygons)))
    return ob, ar


# ------------------------------------------------------------------------------------------------ clip keys
def _limb_rot(name, ang_deg, R3):
    """armature-space rotation of a limb about the bow X axis (tips toward -Z)"""
    X = R3[:, 0]
    sgn = -1.0 if name.startswith("bow_limb_u") else 1.0
    return Matrix.Rotation(math.radians(sgn * ang_deg), 3, Vector(X))


def bow_pose(E, pose, string_world=None, vib=0.0):
    """posed matrices of the bow bones for a solved body pose (dict of armature-space 4x4). string_world: where the
    nocking point is held (None = at rest / braced); vib: string vibration offset (m, along the bow Z) after release"""
    root = pose["bow_root"]
    rel = E.rel_rest
    R3 = np.array(root.to_3x3())
    out = {}
    s_rest = root @ rel["bow_string"]
    if string_world is not None:
        q = Vector(string_world)
    else:
        q = s_rest.translation + root.to_3x3() @ Vector((0, 0, vib))
    ms = Matrix.Translation(q) @ s_rest.to_3x3().to_4x4()
    out["bow_string"] = ms
    local = root.inverted() @ q
    pull = max(0.0, -(local.z - STRING_Z))
    for n in ("bow_limb_u1", "bow_limb_l1"):
        Mr = root @ rel[n]
        R = _limb_rot(n, BEND[n] * pull / DRAW_REF, R3)
        out[n] = Matrix.Translation(Mr.translation) @ (R @ Mr.to_3x3()).to_4x4()
        n2 = n[:-1] + "2"
        M2 = out[n] @ rel[n2]
        R2 = _limb_rot(n2, BEND[n2] * pull / DRAW_REF, R3)
        out[n2] = Matrix.Translation(M2.translation) @ (R2 @ M2.to_3x3()).to_4x4()
    out["bow_socket_arrow"] = root @ rel["bow_socket_arrow"]
    return out, pull


# ------------------------------------------------------------------------------------------------ prop GLB
def write_prop(path=None):
    """stand-alone prop: the bow on its own skeleton (bow_root = the root joint = the attach point: parent it to
    socket_hand_l with an identity transform), bind pose braced"""
    path = path or os.path.join(OUT, "civ", "props", "longbow.glb")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    for o in list(bpy.data.objects):
        bpy.data.objects.remove(o, do_unlink=True)
    arm = bpy.data.armatures.new("rts_longbow")
    rig = bpy.data.objects.new("rts_longbow", arm)
    bpy.context.scene.collection.objects.link(rig)
    bpy.context.view_layer.objects.active = rig
    rig.select_set(True)
    bpy.ops.object.mode_set(mode='EDIT')
    bl = bone_local()
    # the prop frame: bow-local = Blender armature space rotated so glTF (Y up) keeps the bow-local axes on the root
    # joint: bones are authored in bow-local coordinates; the exporter converts only the armature node
    for n, (par, H, y, z, dfm, ln) in bl.items():
        b = arm.edit_bones.new(n)
        b.head = v3(H); b.tail = v3(H + y * max(ln, 0.05))
        b.align_roll(v3(z))
        b.use_deform = dfm
    for n, (par, H, y, z, dfm, ln) in bl.items():
        if par in arm.edit_bones:
            arm.edit_bones[n].parent = arm.edit_bones[par]
    bpy.ops.object.mode_set(mode='OBJECT')
    ob = _to_object(bow_mesh(), "longbow", rig)
    ar = _to_object(arrow_mesh(), "arrow", rig)
    ar["rts_prop"] = "arrow"; ar["rts_socket"] = "bow_arrow"
    ob["rts_prop"] = "longbow"; ob["rts_socket"] = "socket_hand_l"
    rig["rts_bow"] = json.dumps({"attach": "bow_root -> socket_hand_l, identity", "draw_ref_m": DRAW_REF, "deg": BEND,
                                 "sockets": {"bow_string": "arrow nock (nocking point)",
                                             "bow_socket_arrow": "arrow rest"}})
    bpy.context.view_layer.update()
    for o in bpy.context.view_layer.objects:
        o.select_set(False)
    for o in (rig, ob, ar):
        o.select_set(True)
    for o in (ob, ar):
        o.parent = None                    # skinned meshes must be glTF root nodes (the armature modifier binds them)
    bpy.context.view_layer.objects.active = rig
    bpy.ops.export_scene.gltf(filepath=path, export_format='GLB', use_selection=True, export_yup=True,
                              export_skins=True, export_all_influences=False, export_materials='EXPORT',
                              export_extras=True, export_def_bones=False, export_animations=False,
                              export_tangents=True, export_image_format='AUTO')
    clog("longbow prop", path, "%.2f MB" % (os.path.getsize(path) / 1e6))
    return path
