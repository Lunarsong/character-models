// Open a page in headless Chrome and poll a JS expression over the DevTools protocol until it returns a non-null
// value (real time, no virtual-time budget: a 30 MB viewer page needs seconds to decode its textures).
// usage: node scripts/cdp_eval.mjs <chrome> <url> <expression> [timeout_s] [shot.png delay_ms WxH]
//   prints the value (JSON) on stdout; with shot.png it waits delay_ms more (real time, animations run) and saves a
//   screenshot of the WxH window instead
import { spawn } from 'node:child_process';
import { mkdtempSync, readFileSync, existsSync, rmSync, writeSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
const [chrome, url, expr, tmo = '240', shot = '', delay = '0', size = '800x600'] = process.argv.slice(2);
const prof = mkdtempSync(join(tmpdir(), 'rtscdp.'));
const angle = process.platform === 'darwin' ? 'metal' : 'swiftshader';
const ch = spawn(chrome, ['--headless=new', '--no-sandbox', `--use-angle=${angle}`, '--enable-unsafe-swiftshader',
  '--remote-debugging-port=0', `--user-data-dir=${prof}`, '--allow-file-access-from-files', `--window-size=${size.replace('x', ',')}`,
  '--hide-scrollbars',
  '--no-first-run', '--no-default-browser-check', url], { stdio: 'ignore' });
const sleep = ms => new Promise(r => setTimeout(r, ms));
// synchronous write: process.exit() would cut an asynchronous pipe write of a large result (> 64 KB) short
const done = (code, msg) => { if (msg) writeSync(1, msg + '\n'); ch.kill('SIGKILL'); try { rmSync(prof, { recursive: true, force: true }); } catch {} process.exit(code); };
const t0 = Date.now(), limit = +tmo * 1000;
let port;
while (!port) {
  if (Date.now() - t0 > 30000) done(2, 'no DevToolsActivePort');
  const f = join(prof, 'DevToolsActivePort');
  if (existsSync(f)) port = readFileSync(f, 'utf8').split('\n')[0].trim();
  await sleep(200);
}
let ws;
while (!ws) {
  const list = await (await fetch(`http://127.0.0.1:${port}/json/list`)).json();
  const t = list.find(x => x.type === 'page' && x.url.startsWith('file:'));
  if (t) ws = new WebSocket(t.webSocketDebuggerUrl); else await sleep(200);
}
await new Promise(r => ws.addEventListener('open', r));
let id = 0; const pending = {};
ws.addEventListener('message', e => { const m = JSON.parse(e.data); if (m.id && pending[m.id]) { pending[m.id](m); delete pending[m.id]; } });
const call = (method, params) => new Promise(r => { const i = ++id; pending[i] = r; ws.send(JSON.stringify({ id: i, method, params })); });
while (Date.now() - t0 < limit) {
  const r = await call('Runtime.evaluate', { expression: expr, returnByValue: true });
  const v = r.result && r.result.result ? r.result.result.value : undefined;
  if (v !== undefined && v !== null) {
    if (!shot) done(0, typeof v === 'string' ? v : JSON.stringify(v));
    await sleep(+delay);
    const [w, h] = size.split('x').map(Number);
    await call('Emulation.setDeviceMetricsOverride', { width: w, height: h, deviceScaleFactor: 1, mobile: false });
    await sleep(300);
    const r2 = await call('Page.captureScreenshot', { format: 'png' });
    (await import('node:fs')).writeFileSync(shot, Buffer.from(r2.result.data, 'base64'));
    done(0, shot);
  }
  await sleep(500);
}
done(1, 'timeout');
