"""Real-time headless Chrome screenshots of a character viewer through the DevTools protocol (stdlib only).

Why: viewer_shot.sh uses --virtual-time-budget, which does not wait for asynchronous texture decodes, so a page that
loads a second GLB (parts) and material variants can be captured half-loaded. This script loads the page once, waits
until window.__READY__ is true, then for each shot runs a JS snippet (the viewer's own functions: applyRecipe, goView,
setMorph, EXPR, showTab, ...), waits for the frames to settle and captures a PNG.

usage: python3 scripts/cdp_shot.py viewer/base_male.html shots.json
  shots.json: {"w": 1400, "h": 900, "shots": [{"out": "renders/x.png", "js": "goView('Face')", "wait": 1.5,
               "clip": [x, y, w, h] (optional), "print": "<JS expression to print, optional>"}, ...]}
Chrome: the one scripts/viewer_shot.sh finds (work/browser.sh); three.js from work/vendor/three (no network).
"""
import base64, json, os, shutil, socket, struct, subprocess, sys, tempfile, time, urllib.request

CH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORK = os.path.join(CH, "..", "work")
CHROMES = [os.environ.get("CHROME", ""), "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
           "/Applications/Chromium.app/Contents/MacOS/Chromium", shutil.which("google-chrome") or "", shutil.which("chromium") or ""]


class WS:
    """Minimal RFC 6455 client (text frames, client masking, fragmented / large messages)."""

    def __init__(self, url):
        assert url.startswith("ws://")
        hostport, path = url[5:].split("/", 1)
        host, port = hostport.split(":")
        self.s = socket.create_connection((host, int(port)))
        key = base64.b64encode(os.urandom(16)).decode()
        self.s.sendall(("GET /%s HTTP/1.1\r\nHost: %s\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n"
                        "Sec-WebSocket-Key: %s\r\nSec-WebSocket-Version: 13\r\n\r\n" % (path, hostport, key)).encode())
        buf = b""
        while b"\r\n\r\n" not in buf:
            buf += self.s.recv(4096)
        assert b" 101 " in buf.split(b"\r\n")[0], buf[:200]
        self.rest = buf.split(b"\r\n\r\n", 1)[1]
        self.id = 0

    def _read(self, n):
        while len(self.rest) < n:
            chunk = self.s.recv(1 << 20)
            if not chunk:
                raise EOFError
            self.rest += chunk
        out, self.rest = self.rest[:n], self.rest[n:]
        return out

    def send(self, obj):
        data = json.dumps(obj).encode()
        hdr = bytearray([0x81])
        n = len(data)
        if n < 126:
            hdr.append(0x80 | n)
        elif n < 65536:
            hdr.append(0x80 | 126); hdr += struct.pack(">H", n)
        else:
            hdr.append(0x80 | 127); hdr += struct.pack(">Q", n)
        mask = os.urandom(4)
        self.s.sendall(bytes(hdr) + mask + bytes(b ^ mask[i % 4] for i, b in enumerate(data)))

    def recv(self):
        msg = b""
        while True:
            b0, b1 = self._read(2)
            n = b1 & 0x7F
            if n == 126:
                n = struct.unpack(">H", self._read(2))[0]
            elif n == 127:
                n = struct.unpack(">Q", self._read(8))[0]
            payload = self._read(n)
            if (b0 & 0x0F) in (0x9, 0xA):            # ping / pong
                continue
            msg += payload
            if b0 & 0x80:
                return json.loads(msg)

    def call(self, method, **params):
        self.id += 1
        mid = self.id
        self.send({"id": mid, "method": method, "params": params})
        while True:
            m = self.recv()
            if m.get("id") == mid:
                if "error" in m:
                    raise RuntimeError("%s: %s" % (method, m["error"]))
                return m.get("result", {})


def prepare(src):
    h = open(src).read().replace("https://cdn.jsdelivr.net/npm/three@0.147.0", "file://%s/vendor/three" % os.path.abspath(WORK))
    fd, page = tempfile.mkstemp(suffix=".html", dir=os.path.join(CH, "viewer"), prefix=".cdp_")
    os.write(fd, h.replace('<script type="application/octet-stream" id="glb">',
                           '<script>window.__NOBLINK__=true;</script><script type="application/octet-stream" id="glb">', 1).encode())
    os.close(fd)
    return page


def main():
    src, spec = sys.argv[1], json.load(open(sys.argv[2]))
    chrome = next(c for c in CHROMES if c and os.path.exists(c))
    page = prepare(src)
    prof = tempfile.mkdtemp(prefix="cdpshot")
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]
    W, H = spec.get("w", 1400), spec.get("h", 900)
    angle = os.environ.get("ANGLE", "metal" if sys.platform == "darwin" else "swiftshader")
    p = subprocess.Popen([chrome, "--headless=new", "--no-sandbox", "--disable-gpu-sandbox", "--use-angle=" + angle,
                          "--enable-unsafe-swiftshader", "--user-data-dir=" + prof, "--no-first-run",
                          "--no-default-browser-check", "--hide-scrollbars", "--allow-file-access-from-files",
                          "--remote-debugging-port=%d" % port, "--window-size=%d,%d" % (W, H), "about:blank"],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(100):
            try:
                tabs = json.load(urllib.request.urlopen("http://127.0.0.1:%d/json" % port))
                tab = [t for t in tabs if t.get("type") == "page"][0]
                break
            except Exception:
                time.sleep(0.2)
        ws = WS(tab["webSocketDebuggerUrl"])
        ws.call("Page.enable")
        ws.call("Emulation.setDeviceMetricsOverride", width=W, height=H, deviceScaleFactor=1, mobile=False)
        ws.call("Page.navigate", url="file://" + page)
        t0 = time.time()
        while True:
            r = ws.call("Runtime.evaluate", expression="window.__READY__ === true", returnByValue=True)
            if r.get("result", {}).get("value"):
                break
            if time.time() - t0 > float(os.environ.get("SHOT_TIMEOUT", 180)):
                st = ws.call("Runtime.evaluate", expression="document.getElementById('status').textContent", returnByValue=True)
                raise TimeoutError("viewer not ready: %s" % st.get("result", {}).get("value"))
            time.sleep(0.3)
        print("CDP ready in %.1fs" % (time.time() - t0))
        for sh in spec["shots"]:
            if sh.get("js"):
                r = ws.call("Runtime.evaluate", expression="(async () => { %s })()" % sh["js"], awaitPromise=True, returnByValue=True)
                if "exceptionDetails" in r:
                    raise RuntimeError("js failed: %s" % json.dumps(r["exceptionDetails"])[:500])
            time.sleep(sh.get("wait", 1.2))
            if sh.get("print"):
                r = ws.call("Runtime.evaluate", expression=sh["print"], returnByValue=True)
                print("CDP print", json.dumps(r.get("result", {}).get("value"))[:2000])
            if not sh.get("out"):
                continue
            params = {"format": "png"}
            if sh.get("clip"):
                x, y, w, h = sh["clip"]; params["clip"] = {"x": x, "y": y, "width": w, "height": h, "scale": 1}
            img = ws.call("Page.captureScreenshot", **params)["data"]
            out = sh["out"] if os.path.isabs(sh["out"]) else os.path.join(CH, sh["out"])
            os.makedirs(os.path.dirname(out), exist_ok=True)
            open(out, "wb").write(base64.b64decode(img))
            print("CDP shot", out)
    finally:
        p.terminate()
        try:
            p.wait(5)
        except Exception:
            p.kill()
        shutil.rmtree(prof, ignore_errors=True)
        os.remove(page)


if __name__ == "__main__":
    main()
