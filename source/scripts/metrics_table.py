#!/usr/bin/env python3
"""Totals of deform_test.py metrics files, side by side (regions summed over poses and L/R).

usage: python3 scripts/metrics_table.py LABEL=renders/deform_male_metrics.json LABEL2=other.json ... [--poses a,b]
Per region: fold_sum (crease growth beyond 30 deg), folds (> 60 deg), flipped faces, self-intersections, mean volume.
"""
import json, sys

REGS = ["shoulder", "groin", "hip", "elbow", "knee", "neck"]
args = sys.argv[1:]
poses = None
if "--poses" in args:
    i = args.index("--poses"); poses = set(args[i + 1].split(",")); del args[i:i + 2]
sets = [(a.split("=", 1)[0], json.load(open(a.split("=", 1)[1]))) for a in args]
print("region    " + " | ".join("%-30s" % n for n, _ in sets))
for r in REGS:
    cells = []
    for n, m in sets:
        t = [0, 0, 0, 0, 0.0, 0]
        for p, pm in m.items():
            if poses and p not in poses:
                continue
            for side in (["_l", "_r"] if r != "neck" else [""]):
                v = pm.get(r + side)
                if v:
                    t[0] += v["fold_sum"]; t[1] += v["folds"]; t[2] += v.get("flips", 0); t[3] += v["isect"]
                    t[4] += v["vol"]; t[5] += 1
        cells.append("%6.0f /%4d /%4d /%4d  v%.2f" % (t[0], t[1], t[2], t[3], t[4] / max(t[5], 1)))
    print("%-9s %s" % (r, " | ".join("%-30s" % c for c in cells)))
