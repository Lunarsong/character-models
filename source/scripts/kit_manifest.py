#!/usr/bin/env python3
"""MODULAR KIT, orchestrator side (judge C4). After kit_build.py exported the pieces and the clips (Blender):

  body      out/base_<kind>.glb (the base pipeline's full character: body + default face parts + hair, face / cust /
            corrective morphs, material variants) -> out/kit/<kind>/body.glb with the body mesh split into one
            primitive per BODY REGION (kit_glb.split_regions; mesh extras rts_regions) and the MPFB bookkeeping extras
            stripped
  textures  every image of every kit GLB moved to out/kit/textures/ (content-deduplicated: the armour trim sheet is
            stored once for 20 pieces), GLBs reference them by relative uri
  manifest  out/kit/kit_manifest.json: bodies (+ region primitives), slots, pieces (per body: file, tris, covered
            regions + bitsets), outfits, hide rules, skeleton (core + extra joints, helper rules, sockets), clips
  validate  Khronos validator on every kit GLB (scripts/kit_validate.js resolves the shared textures) + skin limits
            (<= 4 influences, weight sums, joint indices) + a union-coverage report per outfit

usage: python3 scripts/kit_manifest.py [male female]      (run by build_knight.py stage 'kit')
Engine contract: see the manifest's "rules" and STATUS.md ('Iteration 2: rig / modularity ...').
"""
import os, sys, json, time, base64, subprocess
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import kit_glb as KG

CH = os.path.dirname(HERE)
OUT = os.path.join(CH, "out")
KIT = os.path.join(OUT, "kit")
TEXD = os.path.join(KIT, "textures")
HIDE_T = 0.95
OUTFITS = {
    "knight": {"doc": "the hero knight after refs/knight_sheet.png (every knight piece, gap fillers, sword and shield)",
               "sets": ["knight"], "exclude": ["scabbard", "gorget_top"]},
    # the civilian outfit agent's outfits (outfit_civ.OUTFITS: peasant, archer, ...) are added from pieces.json
    "civilian": {"doc": "CC0 MakeHuman work / casual suit and shoes (fallback when no civ outfit is installed)",
                 "pieces": ["civ_suit", "civ_shoes"]},
    "militia": {"doc": "mix-and-match proof: the peasant (or the CC0 suit) with the knight's helm, plume, mail "
                       "aventail, gauntlets, sword and shield: one skeleton, pieces from two sets",
                "pieces": ["peasant_shirt", "peasant_trousers", "peasant_shoes", "peasant_tunic", "peasant_belt",
                           "civ_suit", "civ_shoes", "helmet", "plume", "aventail", "gauntlet_l", "gauntlet_r", "sword",
                           "shield"]},
    "bare": {"doc": "the base body only", "pieces": []},
}
CIV_DOC = {"peasant": "peasant (outfit_civ.py): linen shirt, trousers, shoes, wool tunic, belt, straw hat",
           "archer": "archer (outfit_civ.py): the peasant's shirt and trousers + gambeson, boots, hood, bracers, "
                     "gloves, quiver", "peasant_hood": "peasant with the archer's hood instead of the straw hat"}
SLOT_GROUP = {"head": "Head", "crest": "Head", "neck_inner": "Head", "neck": "Torso", "neck_top": "Head",
              "torso_inner": "Torso", "underlayer": "Torso", "torso": "Torso", "torso_outer": "Cloth", "belt": "Belts",
              "back": "Cloth", "clasps": "Cloth", "suit": "Torso", "shoes": "Feet", "weapon_r": "Props", "shield_l": "Props",
              "hip_l": "Props", "legs_inner": "Legs", "hips": "Legs", "tassets": "Legs", "thighs": "Legs", "knees": "Legs",
              "shins": "Legs", "feet_inner": "Feet", "feet": "Feet"}
RULES = {
    "bind": "every piece GLB is skinned to the shared rts_human skeleton (+ the extra joints it lists); bind its skin "
            "to the body's joints BY NAME, graft extra joints (chains, sockets, helpers) under their listed parent",
    "slots": "one piece per slot; a piece also clears the slots in its 'occupies' list; 'requires' pieces must be worn",
    "layers": "layer 0 lies on the skin (mail, boots, gap fillers), higher layers lie over lower ones (plates 1-2, "
              "tabard 2, belts 3, cape 4, props 5)",
    "hide_regions": "hide a body region primitive when the equipped pieces cover >= %.2f of its vertices: OR the "
                    "pieces' cover_bits for that region (bitset over the region's vertex list) and count; single "
                    "pieces list the regions they hide alone in 'hides'" % HIDE_T,
    "hide_parts": "hide the body GLB's part meshes named in the equipped pieces' 'hides_parts' (a closed helm hides "
                  "hair, brows, lashes, eyes, teeth, tongue)",
    "helpers": "helper joints follow skeleton.helpers.rules (baked in the kit clips; evaluate the rule for your own "
               "clips / IK)",
    "correctives": "the body's cor_* corrective morphs (mesh extras rts_correctives) are driven from joint angles and "
                   "only exist on the skin; armour pieces do not carry them",
    "customisation": "cust_* morphs exist on the body and on every piece they move: drive them by name on all meshes",
}


def log(*a):
    print("KITM", *a, flush=True)


def bits_of(b64, n):
    return np.unpackbits(np.frombuffer(base64.b64decode(b64), dtype=np.uint8))[:n].astype(bool)


def union_cover(pieces, ids, regions):
    out = {}
    for r, n in regions.items():
        if not n:
            continue
        m = np.zeros(n, bool)
        for i in ids:
            b = pieces[i]["cover_bits"].get(r)
            if b:
                m |= bits_of(b, n)
        out[r] = round(float(m.mean()), 4)
    return out


def body(kind, reg):
    src = os.path.join(OUT, "base_%s.glb" % kind)
    dst = os.path.join(KIT, kind, "body.glb")
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    counts = KG.split_regions(src, dst, KG.region_of_bone, "%s_body" % kind, order=KG.REGIONS)
    n = KG.strip_extras(dst)
    KG.externalize(dst, TEXD, registry=reg)
    js, _ = KG.read_glb(dst)
    parts = [nd["name"] for nd in js["nodes"] if "mesh" in nd and nd["name"] != "%s_body" % kind]
    log("%s body: %d region primitives %s, %d MPFB extras stripped, parts %s" % (kind, len(counts), counts, n, parts))
    return counts, parts


def run_validator(files):
    r = subprocess.run(["node", os.path.join(HERE, "kit_validate.js")] + files, capture_output=True, text=True)
    res = {}
    for line in r.stdout.splitlines():
        if " errors " in line:
            f = line.split()[0]
            p = line.split()
            res[f] = dict(errors=int(p[p.index("errors") + 1]), warnings=int(p[p.index("warnings") + 1]),
                          tris=int(p[p.index("tris") + 1]))
    return res, r.stdout


def skin_limits(path):
    js, binc = KG.read_glb(path)
    probs = []
    nj = max([len(s["joints"]) for s in js.get("skins", [])] or [0])
    for m in js.get("meshes", []):
        for p in m["primitives"]:
            a = p["attributes"]
            if "JOINTS_1" in a:
                probs.append("JOINTS_1")
            if "WEIGHTS_0" in a:
                w = KG.accessor(js, binc, a["WEIGHTS_0"]).astype(float)
                j = KG.accessor(js, binc, a["JOINTS_0"])
                if abs(w.sum(1) - 1).max() > 0.01:
                    probs.append("weight sums off by %.3f" % abs(w.sum(1) - 1).max())
                if j.max() >= nj:
                    probs.append("joint index out of range")
    return probs


def build(kinds):
    t0 = time.time()
    os.makedirs(TEXD, exist_ok=True)
    reg = {}
    man = {"kit": "rts_hero_kit", "version": 1, "generated": time.strftime("%Y-%m-%d %H:%M"),
           "units": "metres, glTF +Y up, the character faces +Z", "textures": "textures/", "rules": RULES,
           "regions": {r: sorted({b for b in []}) for r in KG.REGIONS}, "bodies": {}, "slots": {}, "pieces": {},
           "outfits": {}, "anims": {}, "skeleton": {}}
    man["regions"] = {r: {"joints": "dominant joint in kit_glb.region_of_bone"} for r in KG.REGIONS}
    per_kind = {}
    for kind in kinds:
        pj = os.path.join(KIT, kind, "pieces.json")
        if not os.path.exists(pj):
            log("no pieces for", kind, "(run kit_build.py pieces)"); continue
        info = json.load(open(pj))
        per_kind[kind] = info
        counts, parts = body(kind, reg)
        man["bodies"][kind] = {"file": "%s/body.glb" % kind, "mesh": "%s_body" % kind,
                               "regions": {r: {"primitive": i, "tris": counts[r], "verts": info["regions"].get(r, 0)}
                                           for i, r in enumerate([r for r in KG.REGIONS if r in counts])},
                               "parts": {p[len(kind) + 1:]: p for p in parts},
                               "source": "out/base_%s.glb (scripts/base_humans.py)" % kind}
        for pid, p in info["pieces"].items():
            f = os.path.join(KIT, kind, p["file"])
            nt = KG.fix_tangents(f)
            if nt:
                log("%s %s: %d degenerate tangents repaired" % (kind, pid, nt))
            KG.externalize(f, TEXD, registry=reg)
            e = man["pieces"].setdefault(pid, {k: p[k] for k in ("slot", "occupies", "layer", "set", "requires",
                                                                  "asset", "socket", "prop", "hides_parts")})
            e.setdefault("bodies", {})[kind] = {k: p[k] for k in ("tris", "verts", "materials", "cust_morphs", "joints",
                                                                   "extra_joints", "covers", "cover_bits", "hides",
                                                                   "MB")}
            e["bodies"][kind]["file"] = "%s/%s" % (kind, p["file"])
            man["slots"].setdefault(p["slot"], {"layer": p["layer"], "group": SLOT_GROUP.get(p["slot"], "Outfit")})
        an = os.path.join(KIT, kind, "anims_knight.glb")
        if os.path.exists(an):
            js, _ = KG.read_glb(an)
            man["anims"][kind] = {"file": "%s/anims_knight.glb" % kind,
                                  "clips": [a.get("name") for a in js.get("animations", [])]}
        sk = info["skeleton"]
        man["skeleton"].setdefault("core", len(sk["core"]))
        man["skeleton"].setdefault("extra", {}).update(sk["extra"])
        man["skeleton"]["helpers"] = sk["helpers"]
        man["skeleton"]["sockets"] = sk["sockets"]
        man["skeleton"]["secondary"] = sk["secondary"]
    # outfits (piece ids present for every body)
    outfits = dict(OUTFITS)
    for kind, info in per_kind.items():
        for n, o in (info.get("civ_outfits") or {}).items():
            if o.get("missing"):                          # an outfit whose assets are not all installed yet
                log("civ outfit %s skipped for %s: missing %s" % (n, kind, o["missing"]))
                continue
            outfits.setdefault(n, {"doc": CIV_DOC.get(n, "civilian outfit %s (outfit_civ.py)" % n), "pieces": []})
            outfits[n]["pieces"] = sorted(set(outfits[n]["pieces"]) | set(o["pieces"]))
    for name, o in outfits.items():
        if "pieces" in o:
            ids = [i for i in o["pieces"] if i in man["pieces"]]
        else:
            ids = sorted(i for i, p in man["pieces"].items() if p["set"] in o["sets"] and i not in o.get("exclude", []))
        if not ids and name not in ("bare",):
            continue
        man["outfits"][name] = {"doc": o["doc"], "pieces": ids,
                                "compatible_bodies": sorted(k for k in per_kind
                                                            if all(k in man["pieces"][i]["bodies"] for i in ids))}
    # a piece fits every body it has a fitted GLB for; any other MakeHuman-basemesh body can refit its MPFB asset
    for pid, p in man["pieces"].items():
        p["compatible_bodies"] = sorted(p["bodies"])
        p["refit"] = ("MPFB clothes asset assets/mpfb_assets/clothes/%s (outfit_lib.add_piece on any MakeHuman body)"
                      % p["asset"]) if p.get("asset") and p["asset"] != "gap_filler" else None
    man["compatibility"] = ("a piece's 'compatible_bodies' are the bodies it was fitted to (one GLB per body under "
                            "'bodies'); all bodies share the rts_human core skeleton with identical rest rotations "
                            "(M15), so clips and helper rules are shared; cust_* morphs are per body mesh")
    # coverage report: union per outfit and body -> hidden regions
    report = {}
    for kind, info in per_kind.items():
        P = {pid: man["pieces"][pid]["bodies"][kind] for pid in man["pieces"] if kind in man["pieces"][pid]["bodies"]}
        for name, o in man["outfits"].items():
            u = union_cover(P, [i for i in o["pieces"] if i in P], info["regions"])
            hidden = sorted(r for r, f in u.items() if f >= HIDE_T)
            man["outfits"][name].setdefault("hidden_regions", {})[kind] = hidden
            report["%s/%s" % (kind, name)] = u
            log("outfit %-9s %-6s hides %2d regions %s; partly covered %s" % (
                name, kind, len(hidden), hidden, {r: f for r, f in u.items() if f < HIDE_T and f > 0.05}))
    json.dump(man, open(os.path.join(KIT, "kit_manifest.json"), "w"), indent=1)
    # validation
    files = []
    for kind in per_kind:
        files += [os.path.join(KIT, kind, "body.glb")] + [os.path.join(KIT, kind, "pieces", f) for f in
                                                          sorted(os.listdir(os.path.join(KIT, kind, "pieces")))
                                                          if f.endswith(".glb")]
        an = os.path.join(KIT, kind, "anims_knight.glb")
        if os.path.exists(an):
            files.append(an)
    res, txt = run_validator(files)
    errs = sum(r["errors"] for r in res.values())
    warns = sum(r["warnings"] for r in res.values())
    lim = {os.path.basename(f): skin_limits(f) for f in files}
    bad_lim = {k: v for k, v in lim.items() if v}
    tex_mb = sum(os.path.getsize(os.path.join(TEXD, f)) for f in os.listdir(TEXD)) / 1e6
    glb_mb = sum(os.path.getsize(f) for f in files) / 1e6
    summary = dict(files=len(files), validator_errors=errs, validator_warnings=warns, skin_limit_problems=bad_lim,
                   glb_MB=round(glb_mb, 1), textures_MB=round(tex_mb, 1), textures=len(os.listdir(TEXD)),
                   seconds=round(time.time() - t0, 1), coverage=report)
    json.dump(summary, open(os.path.join(KIT, "kit_report.json"), "w"), indent=1)
    for line in txt.splitlines():
        if "errors 0 warnings 0" not in line and line.strip():
            log("  " + line)
    log("kit: %d GLBs (%.1f MB) + %d shared textures (%.1f MB); validator %d errors, %d warnings; skin limits %s"
        % (len(files), glb_mb, summary["textures"], tex_mb, errs, warns, bad_lim or "OK"))
    if errs or bad_lim:
        raise SystemExit("kit validation failed")
    return man


if __name__ == "__main__":
    # every body whose pieces are on disk goes into the one manifest (a single-body kit stage keeps the other body)
    want = [a for a in sys.argv[1:] if not a.startswith("-")]
    have = [k for k in ("male", "female") if os.path.exists(os.path.join(KIT, k, "pieces.json"))]
    build([k for k in ("male", "female") if k in want or k in have] or ["male", "female"])
