"""Socket contract check for the knight props: the stand-alone out/props/knight_<prop>.glb attached with an IDENTITY
offset to its socket joint of out/knight_lower_<kind>.glb must land exactly on the prop inside the dressed GLB.
Also shows why the props are pre-rotated: joint frames stay in Blender bone axes (hypothesis block at the end).
run (from characters/): python3 scripts/armour_lower_sockets_check.py [knight]  (exit 1 if any prop is off by > 0.5 mm;
`knight` checks the assembled out/knight_<kind>.glb instead of the lower-armour test GLBs)
"""
import sys, numpy as np
sys.path.insert(0, "scripts")
import check_glb as C

def qmat(q):
    x, y, z, w = q
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)], [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)], [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])

def local(n):
    M = np.eye(4)
    if "matrix" in n: return np.array(n["matrix"]).reshape(4, 4).T
    S = np.diag(n.get("scale", [1, 1, 1])); R = qmat(n.get("rotation", [0, 0, 0, 1]))
    M[:3, :3] = R @ S; M[:3, 3] = n.get("translation", [0, 0, 0]); return M

def globals_(js):
    par = {}
    for i, n in enumerate(js["nodes"]):
        for c in n.get("children", []): par[c] = i
    G = {}
    def g(i):
        if i in G: return G[i]
        G[i] = (g(par[i]) if i in par else np.eye(4)) @ local(js["nodes"][i]); return G[i]
    for i in range(len(js["nodes"])): g(i)
    return G

def mesh_pos(js, b, name):
    for n in js["nodes"]:
        if n.get("mesh") is not None and js["meshes"][n["mesh"]]["name"] == name:
            P = [C.accessor(js, b, p["attributes"]["POSITION"]) for p in js["meshes"][n["mesh"]]["primitives"]]
            return np.vstack(P).astype(float), n
    raise KeyError(name)

# `knight` argument: check the assembled out/knight_<kind>.glb (sword + shield; the knight carries no scabbard)
KNIGHT = "knight" in sys.argv[1:]
for kind in ("male", "female"):
    js, b = C.load(("out/knight_%s.glb" if KNIGHT else "out/knight_lower_%s.glb") % kind)
    G = globals_(js)
    names = {n.get("name"): i for i, n in enumerate(js["nodes"])}
    for prop, sock in (("sword", "socket_weapon_r"), ("shield", "socket_shield_l")) + (() if KNIGHT else (("scabbard", "socket_scabbard_l"),)):
        pj, pb = C.load("out/props/knight_%s.glb" % prop)
        Pp, pn = mesh_pos(pj, pb, "knight_" + prop)
        Pp = (np.c_[Pp, np.ones(len(Pp))] @ local(pn).T)[:, :3]          # prop root transform
        try:
            Pd, dn = mesh_pos(js, b, "%s_%s" % (kind, prop))              # skinned in the dressed GLB (bind pose)
        except KeyError:
            print("%-6s %-8s -> %-18s (not worn in this GLB: skipped)" % (kind, prop, sock))
            continue
        Mw = G[names[sock]]
        Pa = (np.c_[Pp, np.ones(len(Pp))] @ Mw.T)[:, :3]
        # compare as point sets (vertex order can differ after export): nearest-neighbour distance
        d = np.array([np.min(np.linalg.norm(Pd - p, axis=1)) for p in Pa[::7]])
        print("%-6s %-8s -> %-18s max %.5f m  mean %.5f m" % (kind, prop, sock, d.max(), d.mean()))
        BAD = globals().setdefault("BAD", [])
        if not d.max() < 5e-4:
            BAD.append((kind, prop))
if globals().get("BAD"):
    print("FAIL", BAD); sys.exit(1)
print("OK: identity-attached props match the dressed GLBs")
sys.exit(0)
print("--- hypothesis: joint frames are Blender bone frames; undo the Y-up conversion on the prop vertices")
Cinv = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], float)     # glTF (x,y,z) -> Blender (x, -z, y)
for kind in ("male",):
    js, b = C.load("out/knight_lower_%s.glb" % kind)
    G = globals_(js); names = {n.get("name"): i for i, n in enumerate(js["nodes"])}
    for prop, sock in (("sword", "socket_weapon_r"), ("shield", "socket_shield_l"), ("scabbard", "socket_scabbard_l")):
        pj, pb = C.load("out/props/knight_%s.glb" % prop)
        Pp, pn = mesh_pos(pj, pb, "knight_" + prop)
        Pp = (np.c_[Pp.astype(float), np.ones(len(Pp))] @ local(pn).T)[:, :3]
        q = Pp @ Cinv.T
        Pd, dn = mesh_pos(js, b, "%s_%s" % (kind, prop)); Pd = Pd.astype(float)
        Pa = (np.c_[q, np.ones(len(q))] @ G[names[sock]].T)[:, :3]
        d = np.array([np.min(np.linalg.norm(Pd - p, axis=1)) for p in Pa[::7]])
        print("%-6s %-8s -> %-18s max %.5f m  mean %.5f m" % (kind, prop, sock, d.max(), d.mean()))
