// Khronos glTF validator for kit GLBs whose images live in a shared texture folder (external uris, resolved relative
// to each GLB). Same output format as validate.js. usage: node scripts/kit_validate.js a.glb [b.glb ...]
const path = require('path');
const fs = require('fs');
const v = require(path.resolve(__dirname, '../../work/val/node_modules/gltf-validator'));
(async () => {
  let bad = 0;
  for (const f of process.argv.slice(2)) {
    const dir = path.dirname(path.resolve(f));
    const r = await v.validateBytes(new Uint8Array(fs.readFileSync(f)), {
      maxIssues: 200, uri: path.basename(f),
      externalResourceFunction: uri => new Promise((res, rej) =>
        fs.readFile(path.resolve(dir, decodeURIComponent(uri)), (e, d) => e ? rej(e.toString()) : res(new Uint8Array(d)))),
    });
    if (r.issues.numErrors) bad++;
    console.log(path.basename(f), 'errors', r.issues.numErrors, 'warnings', r.issues.numWarnings, 'infos', r.issues.numInfos,
      'tris', r.info.totalTriangleCount, 'draws', r.info.drawCallCount, 'morphed', r.info.morphedPrimitivesCount || 0,
      'skinned', r.info.skinnedPrimitivesCount || 0);
    r.issues.messages.filter(m => m.severity < 2).slice(0, 12).forEach(m => console.log('  ', m.severity === 0 ? 'ERR' : 'WARN', m.code, m.message, m.pointer));
  }
  process.exit(bad ? 1 : 0);
})();
