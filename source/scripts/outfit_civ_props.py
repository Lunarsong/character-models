"""The peasant's HOE (work tool prop), rigid on its own deform bone prop_hoe under socket_weapon_r (identity frame:
+Y along the haft from the right fist toward the blade, the knight's socket axis convention). Ash haft 1.25 m,
forged iron blade with a socket eye, a wedge. Built in socket-local coordinates (Blender Python + numpy)."""
import os, math, json
import numpy as np
import bpy
from mathutils import Matrix
from outfit_civ_lib import CMesh, nrm, lerp, clog, material_for
from outfit_civ_geo import tube_path, lathe, rounded_box
import outfit_civ_bow as BOW

HAFT = (-0.10, 1.12)          # haft from 10 cm behind the right fist to the blade end
LEFT_GRIP = 0.34              # the left hand holds the haft here (m along +Y)
SETS = {"wood": "civ_wood", "iron": "civ_iron", "trim": "civ_trim"}


def hoe_specs(rig):
    import outfit_civ_lib as CL
    specs, frames = CL.civ_socket_specs(rig)
    s = specs["socket_weapon_r"]
    return {"prop_hoe": dict(parent="socket_weapon_r", head=np.array(s["head"], float), y=s["y"], z=s["z"],
                             deform=True, length=0.2)}


def hoe_mesh():
    m = CMesh()
    w = {"prop_hoe": 1.0}
    y0, y1 = HAFT
    path = np.array([[0, y, 0] for y in np.linspace(y0, y1, 8)])
    rad = [0.0155 if i == 0 else 0.0145 for i in range(len(path))]
    tube_path(m, path, rad, 10, "wood", lambda p: dict(w), cap=True, twist_ref=(0, 0, 1))
    # blade: a forged plate at the end, bent 70 deg down from the haft (toward -Z = the palm side), socket eye
    eye = np.array([0, y1 - 0.02, 0])
    lathe(m, eye - np.array([0, 0.04, 0]), np.array([0, 1.0, 0]), np.array([0, 0, 1.0]),
          [(0.022, 0.0), (0.024, 0.02), (0.024, 0.055), (0.02, 0.07)], 12, "iron", w=w, close_top=True,
          close_bottom=True)
    d = nrm(np.array([0, -0.34, -1.0]))                 # blade direction
    c = eye + d * 0.11
    X = np.array([1.0, 0, 0]); Y = nrm(np.cross(d, X))
    rounded_box(m, c, X, Y, d, (0.075, 0.006, 0.10), e=0.18, nu=16, nv=6, slot="iron", w=w)
    return m


def add_hoe(rig, kind):
    import outfit_civ_lib as CL
    s = hoe_specs(rig)["prop_hoe"]
    Y = nrm(s["y"]); Z = nrm(np.array(s["z"]) - Y * np.dot(s["z"], Y)); X = np.cross(Y, Z)
    R3 = np.stack([X, Y, Z], 1)
    m = hoe_mesh()
    BOW.SETS.update(SETS)
    ob = BOW._to_object(m, "%s_hoe" % kind, rig, np.array(s["head"], float), R3)
    ob["rts_part"] = "hoe"; ob["rts_prop"] = "hoe"; ob["rts_asset"] = "rts_peasant_hoe"
    ob["rts_outfit"] = "peasant"; ob["rts_socket"] = "socket_weapon_r"; ob["slot"] = "Props"
    rig["rts_hoe"] = json.dumps({"attach": "prop_hoe = socket_weapon_r (identity)", "left_grip_m": LEFT_GRIP})
    clog("hoe: %d tris" % sum(len(p.vertices) - 2 for p in ob.data.polygons))
    return ob
