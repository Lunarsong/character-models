#!/usr/bin/env python3
"""Shared-skeleton gate (judge item M15): every GLB on the rts_human skeleton must have IDENTICAL joint rest rotations,
so one animation clip plays on every body. Only joint translations (proportions) may differ.

Convention (chr_lib.canonical_rest): each rts_human joint's rest rotation is the frame of the rig JSON's default
positions (assets/rig/rts_human.json, the neutral MakeHuman basemesh), for every body, proportion variant and outfit.
In glTF that is the same node `rotation` for a joint in every file; the node `translation` is per body.

usage: python3 scripts/skeleton_check.py A.glb B.glb [C.glb ...] [--tol DEG] [--json out.json] [--warn-extra]
  Compares the local rest rotation of every joint the files share (first file = reference). Exit 1 if a core rts_human
  joint (the 97 of the rig JSON) differs by more than --tol degrees (default 1.0). Extra joints (outfit chains, sockets)
  are checked the same way; with --warn-extra they only warn (until their owners give them canonical frames too).
"""
import json, math, os, struct, sys

CH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RIG = os.path.join(CH, "assets", "rig", "rts_human.json")


def read_json(path):
    with open(path, "rb") as f:
        f.read(12)
        n, _ = struct.unpack("<II", f.read(8))
        return json.loads(f.read(n))


def joints(path):
    js = read_json(path)
    nodes = js["nodes"]
    if not js.get("skins"):
        return {}
    out = {}
    for sk in js["skins"]:
        for j in sk["joints"]:
            n = nodes[j]
            out[n.get("name", str(j))] = (n.get("rotation", [0.0, 0.0, 0.0, 1.0]), n.get("translation", [0.0, 0.0, 0.0]))
    return out


def qangle(a, b):
    d = abs(sum(x * y for x, y in zip(a, b))) / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))
    return math.degrees(2 * math.acos(min(1.0, d)))


def main():
    args = sys.argv[1:]
    tol, jout, warn_extra = 1.0, None, False
    if "--tol" in args:
        i = args.index("--tol"); tol = float(args[i + 1]); del args[i:i + 2]
    if "--json" in args:
        i = args.index("--json"); jout = args[i + 1]; del args[i:i + 2]
    if "--warn-extra" in args:
        args.remove("--warn-extra"); warn_extra = True
    files = [a for a in args if a.endswith(".glb")]
    assert len(files) >= 2, __doc__
    core = set(json.load(open(RIG))["bones"]) if os.path.exists(RIG) else set()
    ref = joints(files[0])
    report = {"reference": files[0], "tol_deg": tol, "files": {}}
    fail = False
    for f in files[1:]:
        J = joints(f)
        shared = [n for n in ref if n in J]
        rows = sorted(((qangle(ref[n][0], J[n][0]), n) for n in shared), reverse=True)
        bad_core = [(a, n) for a, n in rows if a > tol and n in core]
        bad_extra = [(a, n) for a, n in rows if a > tol and n not in core]
        dt = max((math.dist(ref[n][1], J[n][1]) for n in shared), default=0.0)
        report["files"][f] = {"shared": len(shared), "max_deg": round(rows[0][0], 4) if rows else 0.0,
                              "worst": [[n, round(a, 3)] for a, n in rows[:8]],
                              "core_over_tol": [[n, round(a, 3)] for a, n in bad_core],
                              "extra_over_tol": [[n, round(a, 3)] for a, n in bad_extra],
                              "max_translation_diff_m": round(dt, 4)}
        status = "OK"
        if bad_core or (bad_extra and not warn_extra):
            status = "FAIL"; fail = True
        elif bad_extra:
            status = "WARN (extra joints)"
        print("SKELETON %-5s %s vs %s: %d shared joints, max rest-rotation difference %.4f deg (%s); core > %.1f deg: %d, "
              "extra > %.1f deg: %d; translations differ up to %.3f m" % (
                  status, os.path.basename(f), os.path.basename(files[0]), len(shared), rows[0][0] if rows else 0,
                  rows[0][1] if rows else "-", tol, len(bad_core), tol, len(bad_extra), dt))
        for a, n in (bad_core + bad_extra)[:12]:
            print("    %-24s %7.2f deg%s" % (n, a, "" if n in core else "  (extra joint)"))
    if jout:
        json.dump(report, open(jout, "w"), indent=1)
    sys.exit(1 if fail else 0)


if __name__ == "__main__":
    main()
