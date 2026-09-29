#!/usr/bin/env python3
"""Character look-dev viewer: embeds character GLB(s) into one self-contained HTML page (three.js r147 inlined from
work/vendor/three, GLBs as base64), from characters/viewer/lookdev_template.html.

usage:
  python3 viewer/make_viewer.py CHARACTER.glb [PIECE.glb ...] [-o OUT.html] [--title T] [--sub S]
         [--town | --context BUILDING.glb[@x,z[,rot]] ...] [--context-tex 512] [--view 'Name=px,py,pz:tx,ty,tz[:fov]']
         [--cdn] [--wait MINUTES]

  CHARACTER.glb  any skinned (or static) glTF binary; the first file is the character. Extra files are outfit pieces
                 skinned to the same skeleton: their skinned meshes are rebound to the character's bones by joint name
                 (bones the body lacks, e.g. cape chains, are grafted on), rigid meshes under a bone keep their offset.
  --town         embeds work/out/village_lane.glb (houses on a lane, with its ground) so the RTS cameras show the
                 character among the buildings at game scale; the character stands on the lane at the origin.
  --context      any building GLB, placed at x,z metres (rotated rot degrees about +Y). Its textures are shrunk to
                 --context-tex px (default 512: plenty for the RTS camera) to keep the page small.
  --cdn          reference three.js on jsDelivr instead of inlining it (smaller page, needs network).
  --wait N       wait up to N minutes (polling every minute) for CHARACTER.glb to appear.
  --kit MANIFEST DRESS-UP page of the modular kit (scripts/kit_manifest.py: out/kit/kit_manifest.json) for --kind
                 (male | female): the region-split body, every piece GLB of that body and the clip GLB; the kit's
                 shared textures are embedded once (shrunk to --kit-tex px) and resolved by uri. The page gets a
                 'Dress-up' tab (outfits, one piece per slot, body regions hidden by the worn pieces' coverage).

Output: viewer/<name>_lookdev.html by default. Open it in a browser; headless shots: viewer/shot.sh.
The page reads node extras: 'slot' (or rts_slot / rts_variant_group / option_group / rts_part / MPFB object type)
groups the piece toggles; 'rts_correctives' drives pose correctives; KHR_materials_variants gives a variant picker.
"""
import argparse, base64, html, io, json, os, re, struct, sys, time

VIEWER = os.path.dirname(os.path.abspath(__file__))
CH = os.path.dirname(VIEWER)
WORK = os.path.normpath(os.path.join(CH, '..', 'work'))
THREE = os.path.join(WORK, 'vendor', 'three')
THREE_FILES = ['build/three.min.js', 'examples/js/controls/OrbitControls.js', 'examples/js/loaders/GLTFLoader.js',
               'examples/js/environments/RoomEnvironment.js']
CDN = 'https://cdn.jsdelivr.net/npm/three@0.147.0/'
# village_lane.glb: the character stands on the lane between the houses (lane found from a top view of the scene)
TOWN = [(os.path.join(WORK, "out", "village_lane.glb"), (0.0, -0.8, 0.0))]


def read_glb(data):
    magic, ver, length = struct.unpack('<III', data[:12])
    if magic != 0x46546C67:
        raise ValueError('not a GLB')
    off, js, binc = 12, None, b''
    while off < length:
        clen, ctype = struct.unpack('<II', data[off:off + 8])
        chunk = data[off + 8:off + 8 + clen]
        if ctype == 0x4E4F534A: js = json.loads(chunk)
        elif ctype == 0x004E4942: binc = chunk
        off += 8 + clen
    return js, binc


def write_glb(js, binc):
    j = json.dumps(js, separators=(',', ':')).encode()
    j += b' ' * (-len(j) % 4)
    binc = binc + b'\0' * (-len(binc) % 4)
    out = struct.pack('<III', 0x46546C67, 2, 12 + 8 + len(j) + (8 + len(binc) if binc else 0))
    out += struct.pack('<II', len(j), 0x4E4F534A) + j
    if binc: out += struct.pack('<II', len(binc), 0x004E4942) + binc
    return out


def shrink_textures(data, maxpx):
    """Re-encode the embedded images of a GLB at most maxpx wide (JPEG when opaque, PNG when they carry alpha)."""
    from PIL import Image
    js, binc = read_glb(data)
    views = js.get('bufferViews', [])
    chunks = [binc[v.get('byteOffset', 0):v.get('byteOffset', 0) + v['byteLength']] for v in views]
    for img in js.get('images', []):
        if 'bufferView' not in img: continue
        im = Image.open(io.BytesIO(chunks[img['bufferView']]))
        if max(im.size) <= maxpx and img.get('mimeType') == 'image/jpeg': continue
        s = maxpx / max(im.size)
        if im.mode == 'P' and 'transparency' in im.info: im = im.convert('RGBA')
        if s < 1: im = im.resize((max(1, round(im.width * s)), max(1, round(im.height * s))), Image.LANCZOS)
        alpha = im.mode in ('RGBA', 'LA') and im.getchannel('A').getextrema()[0] < 255
        buf = io.BytesIO()
        if alpha: im.save(buf, 'PNG', optimize=True); img['mimeType'] = 'image/png'
        else: im.convert('RGB').save(buf, 'JPEG', quality=86, optimize=True); img['mimeType'] = 'image/jpeg'
        chunks[img['bufferView']] = buf.getvalue()
    out = bytearray()
    for v, c in zip(views, chunks):
        out += b'\0' * (-len(out) % 4)
        v['byteOffset'] = len(out); v['byteLength'] = len(c); out += c
    if js.get('buffers'): js['buffers'][0]['byteLength'] = len(out)
    return write_glb(js, bytes(out))


def glb_summary(data):
    js, _ = read_glb(data)
    return dict(meshes=len(js.get('meshes', [])), nodes=len(js.get('nodes', [])), skins=len(js.get('skins', [])),
                animations=[a.get('name', '') for a in js.get('animations', [])])


def kit_setup(manifest, kind, maxpx):
    """(CFG.kit, [(glb path, kit piece id | None)], {texture name: (name, bytes, mime)}) for a dress-up page"""
    from PIL import Image
    man = json.load(open(manifest))
    root = os.path.dirname(os.path.abspath(manifest))
    B = man['bodies'][kind]
    files = [(os.path.join(root, B['file']), None)]
    pieces = {}
    for pid, p in sorted(man['pieces'].items(), key=lambda kv: (kv[1]['layer'], kv[0])):
        b = p['bodies'].get(kind)
        if not b:
            continue
        files.append((os.path.join(root, b['file']), pid))
        pieces[pid] = dict(slot=p['slot'], occupies=p.get('occupies', []), requires=p.get('requires', []),
                           layer=p['layer'], set=p.get('set'), hides_parts=p.get('hides_parts', []),
                           cover_bits=b.get('cover_bits', {}), tris=b.get('tris'))
    if kind in man.get('anims', {}):
        files.append((os.path.join(root, man['anims'][kind]['file']), None))
    tex = {}
    for f, _ in files:
        js, _b = read_glb(open(f, 'rb').read())
        for im in js.get('images', []):
            u = im.get('uri')
            if not u or u.startswith('data:'):
                continue
            name = u.split('/')[-1]
            if name in tex:
                continue
            im_ = Image.open(os.path.join(os.path.dirname(f), u))
            if im_.mode == 'P' and 'transparency' in im_.info: im_ = im_.convert('RGBA')
            sc = maxpx / max(im_.size)
            if sc < 1: im_ = im_.resize((max(1, round(im_.width * sc)), max(1, round(im_.height * sc))), Image.LANCZOS)
            alpha = im_.mode in ('RGBA', 'LA') and im_.getchannel('A').getextrema()[0] < 255
            buf = io.BytesIO()
            if alpha: im_.save(buf, 'PNG', optimize=True); mime = 'image/png'
            else: im_.convert('RGB').save(buf, 'JPEG', quality=88, optimize=True); mime = 'image/jpeg'
            tex[name] = (name, buf.getvalue(), mime)
    regions = {r: v['verts'] for r, v in B['regions'].items()}
    cfg = dict(kind=kind, bodyMesh=B['mesh'], regions=regions, hideT=float(man['rules']['hide_regions'].split('>= ')[1].split()[0]),
               pieces=pieces, outfits={n: dict(doc=o['doc'], pieces=[i for i in o['pieces'] if i in pieces])
                                       for n, o in man['outfits'].items()},
               slots=man['slots'], default='knight')
    return cfg, files, tex


def parse_place(s):
    if '@' not in s: return s, (0.0, 0.0, 0.0)
    path, p = s.rsplit('@', 1)
    v = [float(x) for x in p.split(',')] + [0.0, 0.0, 0.0]
    return path, tuple(v[:3])


def parse_view(s):
    name, spec = s.split('=', 1)
    parts = spec.split(':')
    v = dict(p=[float(x) for x in parts[0].split(',')], t=[float(x) for x in parts[1].split(',')], fov=30.0)
    if len(parts) > 2: v['fov'] = float(parts[2])
    return name.strip(), v


def three_tags(cdn):
    if cdn:
        return '\n'.join('<script src="%s%s"></script>' % (CDN, f) for f in THREE_FILES)
    out = []
    for f in THREE_FILES:
        src = open(os.path.join(THREE, f), encoding='utf-8').read().replace('</script', '<\\/script')
        out.append('<script>/* three.js r147 (MIT): %s */\n%s\n</script>' % (f, src))
    return '\n'.join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('glbs', nargs='*')
    ap.add_argument('--kit'); ap.add_argument('--kind', default='male'); ap.add_argument('--kit-tex', type=int, default=1024)
    ap.add_argument('-o', '--out')
    ap.add_argument('--title'); ap.add_argument('--sub')
    ap.add_argument('--town', action='store_true')
    ap.add_argument('--context', action='append', default=[])
    ap.add_argument('--context-tex', type=int, default=512)
    ap.add_argument('--view', action='append', default=[])
    ap.add_argument('--cdn', action='store_true')
    ap.add_argument('--wait', type=float, default=0)
    a = ap.parse_args()

    kit_cfg, kit_files, kit_tex = None, [], {}
    if a.kit:
        kit_cfg, kit_files, kit_tex = kit_setup(a.kit, a.kind, a.kit_tex)
        a.glbs = [f for f, _ in kit_files]
        if not a.out: a.out = os.path.join(VIEWER, 'kit_%s_dressup.html' % a.kind)
        if not a.title: a.title = 'Modular kit (%s): dress-up' % a.kind
    if not a.glbs: ap.error('no GLB given')
    main_glb = a.glbs[0]
    t0 = time.time()
    while not os.path.exists(main_glb) and time.time() - t0 < a.wait * 60:
        print('waiting for', main_glb, flush=True); time.sleep(60)
    for g in a.glbs:
        if not os.path.exists(g): sys.exit('missing ' + g)

    name = os.path.splitext(os.path.basename(main_glb))[0]
    out = a.out or os.path.join(VIEWER, name + '_lookdev.html')
    title = a.title or name.replace('_', ' ').capitalize() + ' look-dev'
    files, data_tags = [], []

    def add(path, role, data, place=None):
        fid = 'glb%d' % len(files)
        label = os.path.splitext(os.path.basename(path))[0]
        f = dict(id=fid, label=label, role=role, bytes=len(data), summary=glb_summary(data))
        if place: f['place'] = list(place)
        files.append(f)
        data_tags.append('<script type="application/octet-stream" id="%s">%s</script>' % (fid, base64.b64encode(data).decode()))

    for i, g in enumerate(a.glbs):
        add(g, 'character' if i == 0 else 'piece', open(g, 'rb').read())
        if kit_files and kit_files[i][1]:
            files[-1]['kit'] = kit_files[i][1]; files[-1]['label'] = kit_files[i][1]
    ctx = [parse_place(c) for c in a.context] + (TOWN if a.town else [])
    for path, place in ctx:
        if not os.path.exists(path):
            sys.exit('missing context %s (the building kit writes it: cd work && python3 mkviewer.py <name>)' % path)
        raw = open(path, 'rb').read()
        small = shrink_textures(raw, a.context_tex) if a.context_tex > 0 else raw
        print('context %s: %.1f MB -> %.1f MB' % (os.path.basename(path), len(raw) / 1e6, len(small) / 1e6))
        add(path, 'context', small, place)

    s = files[0]['summary']
    sub = a.sub or ('%s · %d meshes · %s · built %s' % (os.path.basename(main_glb), s['meshes'],
                    '%d clips' % len(s['animations']) if s['animations'] else 'no clips in the GLB',
                    time.strftime('%Y-%m-%d %H:%M')))
    views = dict(parse_view(v) for v in a.view)
    cfg = dict(title=title, files=files, views=views)
    if kit_cfg:
        cfg['kit'] = kit_cfg; cfg['kitTex'] = {}
        for k, (name, data, mime) in enumerate(kit_tex.values()):
            tid = 'kt%d' % k
            cfg['kitTex'][name] = dict(id=tid, mime=mime)
            data_tags.append('<script type="application/octet-stream" id="%s">%s</script>' % (tid, base64.b64encode(data).decode()))
        print('kit: %d pieces, %d shared textures (%.1f MB at <= %d px)' % (len(kit_cfg['pieces']), len(kit_tex),
              sum(len(d) for _, d, _ in kit_tex.values()) / 1e6, a.kit_tex))

    h = open(os.path.join(VIEWER, 'lookdev_template.html'), encoding='utf-8').read()
    h = h.replace('__TITLE__', html.escape(title)).replace('__SUB__', html.escape(sub))
    h = h.replace('/*__CFG__*/null', json.dumps(cfg).replace('</', '<\\/'))
    h = h.replace('<!--__THREE__-->', three_tags(a.cdn))
    h = h.replace('<!--__DATA__-->', '\n'.join(data_tags))
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    open(out, 'w', encoding='utf-8').write(h)
    print('VIEWER %s  %.1f MB  (%s)' % (out, len(h) / 1e6, ', '.join('%s:%s' % (f['role'], f['label']) for f in files)))


if __name__ == '__main__':
    main()
