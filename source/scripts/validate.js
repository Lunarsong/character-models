// Khronos glTF validator for character GLBs (uses the kit's install in work/val, never modifies it).
// usage: node characters/scripts/validate.js file.glb [more.glb ...]
const path = require('path');
const v = require(path.resolve(__dirname, '../../work/val/node_modules/gltf-validator'));
const fs = require('fs');
(async () => {
  for (const f of process.argv.slice(2)) {
    const r = await v.validateBytes(new Uint8Array(fs.readFileSync(f)), {maxIssues: 200});
    console.log(path.basename(f), 'errors', r.issues.numErrors, 'warnings', r.issues.numWarnings, 'infos', r.issues.numInfos,
      'tris', r.info.totalTriangleCount, 'draws', r.info.drawCallCount, 'morphed', r.info.morphedPrimitivesCount || 0,
      'skinned', r.info.skinnedPrimitivesCount || 0);
    r.issues.messages.filter(m => m.severity < 2).slice(0, 12).forEach(m => console.log('  ', m.severity === 0 ? 'ERR' : 'WARN', m.code, m.message, m.pointer));
  }
})();
