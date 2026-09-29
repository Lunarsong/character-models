"""Viewer-level expression test: loads a character viewer page (viewer/base_<kind>.html) in headless Chrome with the
test hook of viewer/char_viewer_template.html, which clicks every expression button like a user (idle blink ON) and
records the morph influences three.js actually applies to every mesh for 6 s. Checks:
  - every weight of every preset reaches every mesh that has that morph target and HOLDS (min over time within 0.02
    of the preset value; the idle blink may only raise eyeBlink*, never lower it; '_loop' keys may oscillate)
  - 'Wink L' / 'Wink R': the winking eye's eyeBlink* stays >= 0.99 on every mesh (body lids, eyelashes, ...) for the
    whole run, while the idle blink still runs on the other eye (it reaches > 0.5 and returns to ~0)
  - 'Neutral': nothing but the idle blink is applied (pose correctives cor_* are skeleton-driven and ignored here;
    customisation cust_* stay at 0); 'Talk': visemes are played
  - with a customisation build the parts GLB is loaded too: every alternate part (brows, lashes, hair, beards) is
    checked the same way
This covers the original 'Wink' bug: the old viewer wrote the idle-blink curve into eyeBlinkLeft/Right every frame
(overwriting the preset), so the winking eye re-opened immediately.
usage: python3 scripts/viewer_check.py viewer/base_male.html [viewer/base_female.html ...] [--json out/viewer_report.json]
exit 1 on any failure.
"""
import sys, os, json, subprocess, tempfile

CH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORK = os.path.join(os.path.dirname(CH), "work")
PRESETS = json.load(open(os.path.join(CH, "scripts", "expressions.json")))["presets"]
CHROMES = [os.environ.get("CHROME", ""), "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
           "/Applications/Chromium.app/Contents/MacOS/Chromium", "/usr/bin/google-chrome", "/usr/bin/chromium"]


def run_page(src):
    h = open(src).read().replace("https://cdn.jsdelivr.net/npm/three@0.147.0", "file://%s/vendor/three" % WORK)
    h = h.replace('<script type="application/octet-stream" id="glb">',
                  '<script>window.__TEST__=true;window.__NOBLINK__=false;</script>'
                  '<script type="application/octet-stream" id="glb">', 1)
    chrome = next(c for c in CHROMES if c and os.path.exists(c))
    fd, page = tempfile.mkstemp(suffix=".html", prefix=".check_", dir=os.path.join(CH, "viewer"))
    os.write(fd, h.encode()); os.close(fd)
    try:
        # DevTools protocol (scripts/cdp_eval.mjs, Node >= 22): real time, polls until the test hook has written its
        # result (a --dump-dom virtual-time budget runs out before a 30 MB page has decoded its textures)
        for attempt in range(2):
            r = subprocess.run(["node", os.path.join(CH, "scripts", "cdp_eval.mjs"), chrome, "file://" + page,
                                "(document.getElementById('__test__') || {}).textContent || null", "300"],
                               capture_output=True, text=True, timeout=400)
            if r.returncode == 0:
                return json.loads(r.stdout)
        raise RuntimeError("test hook produced no result for %s (%s)" % (src, r.stdout.strip()))
    finally:
        os.remove(page)


def check(src):
    res = run_page(src)
    fails, info = [], {}
    has = {m: set(ks) for m, ks in res["meshes"].items()}
    body = next(m for m in has if m.endswith("_body"))
    for name, pre in PRESETS.items():
        got = res["presets"].get(name, {})
        loop = pre.get("_loop", {})
        for k, v in pre.items():
            if k.startswith("_"):
                continue
            meshes = [m for m in has if k in has[m]]
            if body not in meshes:
                fails.append("%s: %s is not a morph target of the body" % (name, k))
            for m in meshes:
                lo, hi = got.get(m + "|" + k, [0.0, 0.0])
                if k in loop:
                    a, b, _ = loop[k]
                    ok = lo >= a - 0.03 and hi <= b + 0.03
                elif k.startswith("eyeBlink"):
                    ok = lo >= v - 0.02                    # the idle blink may only close the eye further
                else:
                    ok = lo >= v - 0.02 and hi <= v + 0.02
                if not ok:
                    fails.append("%s: %s on %s ranges %.3f..%.3f, preset %.2f" % (name, k, m, lo, hi, v))
        # cor_* = pose-space correctives, driven by the skeleton (not by expressions; scripts/deform_test.py covers them)
        for k in [k for k in got if k.split("|")[1] not in pre and not k.split("|")[1].startswith(("eyeBlink", "cor_"))]:
            if name not in ("Talk",) and got[k][1] > 0.02:
                fails.append("%s: unexpected %s up to %.3f" % (name, k, got[k][1]))
        if name.startswith("Wink"):
            side = "Left" if name.endswith("L") else "Right"
            other = "Right" if side == "Left" else "Left"
            w = {m: got.get(m + "|eyeBlink" + side, [0, 0]) for m in has if "eyeBlink" + side in has[m]}
            o = got.get(body + "|eyeBlink" + other, [0, 0])
            info[name] = dict(winking_min={m: v[0] for m, v in w.items()}, other_eye_range=o)
            if min(v[0] for v in w.values()) < 0.99:
                fails.append("%s: the winking eye opens (eyeBlink%s min %s)" % (name, side, {m: v[0] for m, v in w.items()}))
            if not (o[1] > 0.5 and o[0] < 0.05):
                fails.append("%s: the idle blink no longer runs on the other eye (%s)" % (name, o))
    t = res["presets"].get("Talk", {})
    if not any(k.split("|")[1].startswith("viseme_") and v[1] > 0.5 for k, v in t.items()):
        fails.append("Talk: no viseme played")
    info["eyeBlinkLeft_meshes"] = sorted(m for m in has if "eyeBlinkLeft" in has[m])
    print("VIEWER %s: %d presets, %d failures" % (os.path.basename(src), len(res["presets"]), len(fails)))
    for f in fails:
        print("   FAIL", f)
    for k, v in info.items():
        print("   %s: %s" % (k, v))
    return dict(file=os.path.basename(src), fails=fails, info=info)


if __name__ == "__main__":
    args = sys.argv[1:]
    js = args[args.index("--json") + 1] if "--json" in args else None
    files = [a for a in args if a.endswith(".html")]
    rep = [check(f) for f in files]
    if js:
        json.dump(rep, open(js, "w"), indent=1)
    sys.exit(1 if any(r["fails"] for r in rep) else 0)
