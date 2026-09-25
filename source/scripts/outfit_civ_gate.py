#!/usr/bin/env python3
"""Penetration gate summary for the civilian outfits (reads renders/civ/qa_<outfit>_<kind>.json written by
`outfit_civ.py qa`, i.e. measured on exactly what is in the dressed GLBs).

Measures (outfit_civ_qa.penetration):
  pokes  VISIBLE poke-through: vertices of an inner layer (or the visible skin) that the outer layer covers at rest,
         that end up > 1 mm outside it (oriented ray test, turned-in hems / linings ignored) AND that can be seen (their
         outward ray is not blocked within 25 cm: pokes inside a crease or between the arm and the torso are counted
         separately as hidden_pokes_total)
  tri    intersecting triangle pairs between two pieces (strict: also counts contacts deep inside folds)
Classes: clips (every 3rd frame of every clip), everyday poses, extreme poses (deep flexion / full reach, listed below).
usage: python3 scripts/outfit_civ_gate.py [male female] [--strict]   (--strict: exit 1 unless every clip frame is 0)
"""
import json, os, sys, glob

CH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REN = os.path.join(CH, "renders", "civ")
EXTREME = {"squat", "viewer_squat", "deep_squat", "sitting", "high_kick", "kneel", "deep_lunge", "arms_up_180",
           "cross_body_reach", "arm_behind_back"}


def summarize(path):
    r = json.load(open(path))
    pp = r.get("visible_pokes_per_pose", r["pokes_per_pose"])
    allp = r["pokes_per_pose"]
    tri = r["per_pose"]
    clips, every, extreme = {}, {}, {}
    for k, v in pp.items():
        if k.startswith("pose:"):
            p = k[5:]
            (extreme if p in EXTREME else every)[p] = v
        else:
            c = k.rsplit("_f", 1)[0]
            d = clips.setdefault(c, {"frames": 0, "frames_with_pokes": 0, "max": 0, "sum": 0})
            d["frames"] += 1; d["frames_with_pokes"] += int(v > 0); d["max"] = max(d["max"], v); d["sum"] += v
    worst = sorted(r["pokes"].items(), key=lambda kv: -kv[1].get("visible_max", kv[1]["max"]))[:6]
    hidden = sum(allp.values()) - sum(pp.values())
    return dict(outfit=r.get("outfit"), kind=r.get("kind"), clips=clips, hidden_pokes_total=hidden,
                everyday=dict(max=max(every.values() or [0]), sum=sum(every.values()), poses=every),
                extreme=dict(max=max(extreme.values() or [0]), sum=sum(extreme.values()), poses=extreme),
                tri_at_rest=tri.get("pose:rest"), worst_pairs={k: dict(max=v["max"], visible_max=v.get("visible_max"),
                                                                         at=v["at"], mm=v["max_mm"],
                                                                         joints=dict(sorted(v.get("joints", {}).items(),
                                                                                            key=lambda kv: -kv[1])[:3]))
                                                               for k, v in worst})


def main(argv):
    strict = "--strict" in argv
    kinds = [a for a in argv if not a.startswith("--")] or ["male", "female"]
    out = {}
    ok = True
    print("%-14s %-7s | %-38s | %-15s | %-15s | %s" % ("outfit", "body", "clips: VISIBLE pokes max/frame (frames>0)",
                                                      "everyday max/sum", "extreme max/sum", "tri@rest"))
    for f in sorted(glob.glob(os.path.join(REN, "qa_*_*.json"))):
        base = os.path.basename(f)[3:-5]
        kind = base.rsplit("_", 1)[1]
        if kind not in kinds:
            continue
        s = summarize(f)
        out[base] = s
        cl = " ".join("%s %d(%d)" % (c, d["max"], d["frames_with_pokes"]) for c, d in s["clips"].items())
        print("%-14s %-7s | %-38s | %6d / %-6d | %6d / %-6d | %s" % (s["outfit"], kind, cl, s["everyday"]["max"],
                                                                s["everyday"]["sum"], s["extreme"]["max"],
                                                                s["extreme"]["sum"], s["tri_at_rest"]))
        if any(d["sum"] for d in s["clips"].values()):
            ok = False
    json.dump(out, open(os.path.join(REN, "qa_summary.json"), "w"), indent=1)
    print("summary -> renders/civ/qa_summary.json; clip gate %s" % ("PASS" if ok else "OPEN (see worst_pairs)"))
    if strict and not ok:
        sys.exit(1)


if __name__ == "__main__":
    main(sys.argv[1:])
