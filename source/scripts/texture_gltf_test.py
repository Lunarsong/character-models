"""Export test for the character texture sets: one tile per set with a char_materials material -> out/test/textures_test.glb,
then check how the glTF exporter wrote the materials (ORM reuse, occlusion, alpha mode, sheen / anisotropy).

BLENDER_USER_RESOURCES=$PWD/blender_profile $BL -b --python-exit-code 1 -P scripts/texture_gltf_test.py
node scripts/validate.js out/test/textures_test.glb
"""
import bpy, bmesh, sys, os, json, struct

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import char_materials as cm

CH = os.path.dirname(HERE)
OUT = os.path.join(CH, "out", "test", "textures_test.glb")
bpy.ops.wm.read_factory_settings(use_empty=True)
sets = [k for k, v in cm.manifest().items() if v.get("kind") != "detail_normal"]
for i, sname in enumerate(sorted(sets)):
    bm = bmesh.new(); uvl = bm.loops.layers.uv.new("UVMap")
    vs = [bm.verts.new(p) for p in ((0, 0, 0), (0.3, 0, 0), (0.3, 0, 0.3), (0, 0, 0.3))]
    f = bm.faces.new(vs)
    for lp, uv in zip(f.loops, ((0, 0), (1, 0), (1, 1), (0, 1))):
        lp[uvl].uv = uv
    me = bpy.data.meshes.new(sname); bm.to_mesh(me); bm.free()
    o = bpy.data.objects.new(sname, me); bpy.context.scene.collection.objects.link(o)
    o.location = ((i % 5) * 0.35, 0, (i // 5) * 0.35)
    cm.assign(o, cm.material(sname))
os.makedirs(os.path.dirname(OUT), exist_ok=True)
bpy.ops.export_scene.gltf(filepath=OUT, export_format='GLB', export_image_format='AUTO', export_yup=True)
b = open(OUT, "rb").read()
ln = struct.unpack_from("<I", b, 12)[0]
js = json.loads(b[20:20 + ln])
print("GLB", OUT, round(len(b) / 1e6, 1), "MB, images", len(js.get("images", [])), "textures", len(js.get("textures", [])))
print("extensionsUsed", js.get("extensionsUsed"))
for m in js["materials"]:
    pbr = m.get("pbrMetallicRoughness", {})
    mr = pbr.get("metallicRoughnessTexture", {}).get("index"); oc = m.get("occlusionTexture", {}).get("index")
    src = lambda t: js["textures"][t]["source"] if t is not None else None
    print("MAT", m["name"], "alpha", m.get("alphaMode", "OPAQUE"), m.get("alphaCutoff", ""), "MR img", src(mr),
          "OCC img", src(oc), "same" if src(mr) == src(oc) and mr is not None else "", "ext", list(m.get("extensions", {}).keys()),
          "doubleSided", m.get("doubleSided", False))
