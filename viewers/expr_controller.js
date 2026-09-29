
(function () {
    const CFG = {
    "attack":  {"brow": 0.1, "eye": 0.075, "cheek": 0.17, "nose": 0.13, "mouth": 0.16, "jaw": 0.15, "tongue": 0.1, "viseme": 0.035, "other": 0.12},
    "release": {"brow": 0.18, "eye": 0.12, "cheek": 0.3, "nose": 0.22, "mouth": 0.3, "jaw": 0.24, "tongue": 0.14, "viseme": 0.05, "other": 0.22},
    "release_chan": {"^mouthSmile": 0.42, "^smileOpen": 0.42, "^cheekSquint": 0.42, "^eyeSquint": 0.46},
    "antagonist": {"smile": "^mouthSmile", "neg": "^(browDown|mouthFrown|mouthPress|noseSneer)", "hold": 0.2, "max_hold": 0.9,
                   "neg_rise": "^(browInnerUp|browOuterUp)", "pair": "^(smileOpen|mouthUpperUp)", "pair_lead": 0.1,
                   "neg_rise_wide": 0.3, "ease": "^browDown", "ease_share": 0.3, "ease_gate": 0.35,
                   "post_hold": 0.22},
    "speech": {"fade_in": 0.12, "fade_out": 0.3, "energy_lp": 0.18, "energy_hold": 0.5, "class_lp": 0.3, "open_jaw": 0.45,
               "class": {"spread": ["viseme_E", "viseme_I", "viseme_SS"], "neutral": ["viseme_DD", "viseme_nn", "viseme_kk", "viseme_TH"],
                         "round": ["viseme_O", "viseme_U", "viseme_CH", "viseme_RR"], "closed": ["viseme_PP", "viseme_FF"], "open": ["viseme_aa"]},
               "smile_keep": {"spread": 0.5, "neutral": 0.25, "round": 0.2, "closed": 0.45, "open": 0.4},
               "smileopen_keep": {"spread": 0.4, "neutral": 0.35, "round": 0.35, "closed": 0.0, "open": 0.4},
               "lip_lift": {"key": "mouthShrugLower", "max": 0.16, "teeth": [0.15, 0.4]}, "release": 0.16,
               "eye_smile_floor": 0.45, "smile_floor": 0.5,
               "keep": {"jaw": 0.2, "mouthFrown": 0.55, "mouthShrug": 0.5, "noseSneer": 0.6, "mouthDimple": 0.6, "mouthUpperUp": 0.25, "mouthLowerDown": 0.15, "mouthStretch": 0.25, "mouthPress": 0.4, "mouthClose": 0.0, "mouthRoll": 0.2, "mouthPucker": 0.2, "mouthFunnel": 0.2, "mouthLeft": 0.3, "mouthRight": 0.3, "snarl": 0.15, "grimace": 0.2, "tongue": 0.0, "cheekPuff": 0.0, "other": 0.3},
               "min_jaw": 0.14, "min_jaw_base": 0.07, "jaw_classes": ["spread", "open", "neutral", "round"], "shout_keep": 0.6},
    "gesture": {"attack": 0.06, "release": 0.16, "smile_gate": [0.2, 0.4], "down_to_raise": 0.5, "neg_gate": [0.1, 0.2], "raise_gate": [0.2, 0.4]},
    "loop_fade": 0.25,
    "blink": {"release_rate": 8.0, "wide_suppress": 0.0},
    "look": {"amp_x": 0.05, "amp_y": 0.035, "interval": [0.6, 2.5], "saccade": 0.045}
  }  ;
  const FACE = /^(brow|eye|cheek|jaw|mouth|nose|tongue|smileOpen|snarl|grimace|viseme_)/;
  const region = n => /^viseme_/.test(n) ? 'viseme' : /^brow/.test(n) ? 'brow' : /^eye/.test(n) ? 'eye' :
    /^cheek(Squint)/.test(n) ? 'cheek' : /^nose|^snarl/.test(n) ? 'nose' : /^jaw/.test(n) ? 'jaw' : /^tongue/.test(n) ? 'tongue' :
    /^(mouth|smileOpen|grimace|cheekPuff)/.test(n) ? 'mouth' : 'other';
  const S = CFG.speech, RX_SMILE = new RegExp(CFG.antagonist.smile), RX_NEG = new RegExp(CFG.antagonist.neg);
  const RX_NEG_RISE = new RegExp(CFG.antagonist.neg_rise || '^$');
  const RX_PAIR = new RegExp(CFG.antagonist.pair || '^$'), RX_EASE = new RegExp(CFG.antagonist.ease || '^$');
  const REL_CHAN = Object.entries(CFG.release_chan || {}).map(([k, v]) => [new RegExp(k), v]);
  const VCLASS = {}; for (const c in S.class) for (const v of S.class[c]) VCLASS[v] = c;
  const SPEECH_KIND = n => /^mouthSmile/.test(n) ? 'smile' : /^smileOpen/.test(n) ? 'smileopen' : /^(cheekSquint|eyeSquint)/.test(n) ? 'eyesmile' :
    /^(jaw|mouth|snarl|grimace|tongue|cheekPuff|noseSneer)/.test(n) ? 'yield' : null;
  const keepOf = n => { for (const k in S.keep) if (n.startsWith(k)) return S.keep[k]; return S.keep.other; };
  const CONFLICT = n => /^(mouthPress|mouthClose|mouthRoll)/.test(n) ? 'open' : /^(mouthPucker|mouthFunnel)/.test(n) ? 'spread' :
    /^mouthStretch/.test(n) ? 'round' : null;
  let names = [], base = null, overlay = null, blinkNames = [], VJ = {}, speedK = 1, speedT = 0;
  const cur = {}, vel = {}, out = {}, gcur = {}, gvel = {}, bOut = {}, holdT = {};
  const yk = {}, ykv = {}, eOut = {}, postH = {};
  let lastVis = -1e9;
  let loopW = 0, loopHint = null, lastLoop = null, loopT0 = 0, speechW = 0, talking = false, energy = 0;
  const share = {}; for (const c in S.class) share[c] = 0;
  const JAWC = S.jaw_classes || Object.keys(S.class).filter(c => c !== 'closed');
  const look = {idle: false, w: 0, x: 0, y: 0, x0: 0, y0: 0, x1: 0, y1: 0, t: 0, next: 1.0, dur: CFG.look.saccade};
  let clock = 0;

  function smoothDamp(c, target, v, st, dt) {
    const omega = 2 / Math.max(st, 1e-4), x = omega * dt, e = 1 / (1 + x + 0.48 * x * x + 0.235 * x * x * x);
    const change = c - target, temp = (v + omega * change) * dt;
    let nv = (v - omega * temp) * e, o = target + (change + temp) * e;
    if ((target - c > 0) === (o > target)) { o = target; nv = 0; }
    return [o, nv];
  }
  const tAR = (n, c, t, fast) => {
    if (Math.abs(t) > Math.abs(c) + 1e-6) return CFG.attack[region(n)] || CFG.attack.other;
    if (!fast) for (const [rx, v] of REL_CHAN) if (rx.test(n)) return v;
    return CFG.release[region(n)] || CFG.release.other;
  };
  function setup(o) {
    names = (o.names || []).filter(n => FACE.test(n) && !/^cmb_/.test(n));
    base = o.base; overlay = o.overlay; blinkNames = (o.blinkNames || []).filter(Boolean);
    VJ = Object.assign({viseme_aa: 0.55, viseme_E: 0.08, viseme_O: 0.15, viseme_I: 0.05, viseme_U: 0.05, viseme_CH: 0.05,
      viseme_SS: 0.03, viseme_RR: 0.05, viseme_kk: 0.05, viseme_DD: 0.03, viseme_nn: 0.03, viseme_TH: 0.05}, o.visemeJaw || {});
    for (const n of names) { cur[n] = base[n] || 0; vel[n] = 0; out[n] = cur[n]; gcur[n] = 0; gvel[n] = 0; holdT[n] = 0;
      yk[n] = 1; ykv[n] = 0; eOut[n] = cur[n]; postH[n] = 0; }
  }
  const has = n => n in cur;
  function setLoop(hint, t0) { if (hint) { loopHint = hint; lastLoop = hint; loopT0 = t0 || 0; } else loopHint = null; }
  function setTalking(on) { talking = !!on; }
  function setLook(o) { if (o && o.idle != null) look.idle = !!o.idle; }
  const mx2 = (a, b) => Math.max(a || 0, b || 0);
  function step(dt, nowMs) {
    if (!base) return;
    dt = Math.max(0, Math.min(dt, 0.1)); clock += dt;
    const lpk = tc => dt > 0 ? 1 - Math.exp(-dt / tc) : 1;
    speechW += ((talking ? 1 : 0) - speechW) * (1 - Math.exp(-dt / ((talking ? S.fade_in : S.fade_out) / 3)));
    loopW += ((loopHint && !talking ? 1 : 0) - loopW) * (1 - Math.exp(-dt / (CFG.loop_fade / 3)));
    let smC = 0, smT = 0, ngC = 0, ngT = 0, nrT = 0, wdT = 0;
    for (const n of names) {
      if (RX_SMILE.test(n)) { smC = Math.max(smC, cur[n]); smT = Math.max(smT, base[n] || 0); }
      else if (RX_NEG.test(n)) { ngC = Math.max(ngC, cur[n]); ngT = Math.max(ngT, base[n] || 0); }
      else if (RX_NEG_RISE.test(n)) nrT = Math.max(nrT, base[n] || 0);
      if (/^eyeWide/.test(n)) wdT = Math.max(wdT, base[n] || 0);
    }
    const H = CFG.antagonist.hold, HMAX = CFG.antagonist.max_hold || 0.3, antIn = ngT > H || nrT > H;
    const smFall = smT < H && smC > H && smC - smT > 0.02, ngFall = ngT < H && ngC > H && ngC - ngT > 0.02;
    const nrHold = ngT > H && wdT < (CFG.antagonist.neg_rise_wide || 0.3);
    const EG = CFG.antagonist.ease_gate || 0.4, ES = CFG.antagonist.ease_share || 0;
    let moving = false;
    for (const n of names) {
      const isVis = /^viseme_/.test(n);
      let tgt = isVis ? mx2(overlay[n], base[n]) : (base[n] || 0);
      const tgt0 = tgt;
      let held = false;
      if (tgt > cur[n] && (holdT[n] || 0) < HMAX) {
        if ((RX_NEG.test(n) || (RX_NEG_RISE.test(n) && nrHold)) && smFall) held = true;
        else if ((RX_SMILE.test(n) || RX_PAIR.test(n)) && ngFall) held = true;
      }
      if (held) { tgt = (RX_EASE.test(n) && smC < EG) ? Math.max(cur[n], ES * tgt0) : cur[n]; holdT[n] = (holdT[n] || 0) + dt; postH[n] = 1; }
      else if (!(tgt > cur[n])) { holdT[n] = 0; postH[n] = 0; }
      if (postH[n] && !held && Math.abs(tgt - cur[n]) < 0.02) postH[n] = 0;
      if (dt > 0) { let st = tAR(n, cur[n], tgt, antIn) * (isVis ? 1 : speedK); if (postH[n] && !held) st = Math.max(st, CFG.antagonist.post_hold || 0);
        const r = smoothDamp(cur[n], tgt, vel[n], st, dt); cur[n] = r[0]; vel[n] = r[1]; }
      else { cur[n] = tgt; vel[n] = 0; }
      if (!isVis && Math.abs(cur[n] - (base[n] || 0)) > 2e-3) moving = true;
    }
    if (smT > 0.05) for (const n of names) if (RX_PAIR.test(n)) {
      const side = /Left$/.test(n) ? 'Left' : /Right$/.test(n) ? 'Right' : '', sm = 'mouthSmile' + side;
      const sT = has(sm) ? (base[sm] || 0) : smT, sC = has(sm) ? cur[sm] : smC, T = base[n] || 0;
      if (T > cur[n] - 1e-6 && sT > 0.05) { const lim = T * Math.min(1, sC / sT + (CFG.antagonist.pair_lead || 0)); if (cur[n] > lim) { cur[n] = lim; vel[n] = Math.min(vel[n], 0); } }
    }
    if (speedK !== 1) { speedT += dt; if (!moving && speedT > 0.1) speedK = 1; }
    const L = loopHint || (loopW > 1e-3 ? lastLoop : null);
    const tl = ((nowMs || performance.now()) - loopT0) / 1000;
    const lo = {};
    if (L) for (const k in L) if (has(k)) { const [a, b, hz] = L[k]; lo[k] = loopW * (b - a) / 2 * -Math.cos(tl * hz * 2 * Math.PI); }
    let vsum = 0, open = 0; const cs = {}; for (const c in share) cs[c] = 0;
    for (const n of names) if (/^viseme_/.test(n)) {
      const v = cur[n]; vsum += v; open += v * (VJ[n] || 0); const c = VCLASS[n]; if (c) cs[c] += v;
    }
    open = Math.min(1, open / S.open_jaw);
    if (vsum > 0.05) lastVis = clock;
    const eT = Math.min(1, vsum);
    if (eT >= energy) energy += (eT - energy) * lpk(S.energy_lp * 0.4);
    else if (clock - lastVis > (S.energy_hold || 0)) energy += (eT - energy) * lpk(S.energy_lp);
    if (vsum > 0.05) for (const c in share) share[c] += (cs[c] / vsum - share[c]) * lpk(S.class_lp);
    const A = speechW * energy;
    let kSmile = 0, kOpen = 0, sumShare = 0;
    for (const c in share) { kSmile += share[c] * S.smile_keep[c]; kOpen += share[c] * S.smileopen_keep[c]; sumShare += share[c]; }
    const ks = sumShare > 0.01 ? kSmile / sumShare : S.smile_keep.spread, ko = sumShare > 0.01 ? kOpen / sumShare : S.smileopen_keep.spread;
    const smileRet = 1 - A * (1 - ks), smW = Math.min(1, Math.max(smT, smC) / 0.3);
    let jawT = 0, shapeT = 0;
    for (const n of names) {
      const bv = base[n] || 0;
      if (n === 'jawOpen') jawT = bv;
      else if (/^(mouthStretch|snarl|mouthUpperUp|noseSneer)/.test(n)) shapeT = Math.max(shapeT, bv);
    }
    const shout = Math.min(1, Math.max(0, (jawT - 0.25) / 0.15)) * Math.min(1, Math.max(0, (shapeT - 0.2) / 0.2));
    let b = 0; for (const bn of blinkNames) b = Math.max(b, overlay[bn] || 0);
    const G = CFG.gesture, gate = (v, g_) => Math.min(1, Math.max(0, (v - g_[0]) / (g_[1] - g_[0])));
    const sg = gate(smT, G.smile_gate);
    let bdT = 0, frT = 0, buT = 0;
    for (const n of names) {
      const bv = base[n] || 0;
      if (/^browDown/.test(n)) bdT = Math.max(bdT, bv); else if (/^mouthFrown/.test(n)) frT = Math.max(frT, bv);
      else if (/^(browInnerUp|browOuterUp|eyeWide)/.test(n)) buT = Math.max(buT, bv);
    }
    const ng = gate(Math.max(bdT, frT), G.neg_gate || [0.1, 0.2]), ug = gate(buT, G.raise_gate || [0.2, 0.4]);
    const gt = {};
    for (const n of names) {
      if (/^viseme_/.test(n) || blinkNames.includes(n)) continue;
      let g = overlay[n] || 0;
      if (/^(browDown|eyeSquint)/.test(n)) g *= (1 - sg) * (1 - ug);
      else if (/^(mouthSmile|smileOpen|cheekSquint)/.test(n)) g *= 1 - ng;
      else if (/^(browInnerUp|browOuterUp)/.test(n)) g *= 1 - gate(bdT, [0.2, 0.35]);
      gt[n] = g;
    }
    const down = Math.max(overlay.browDownLeft || 0, overlay.browDownRight || 0) * Math.max(sg, ug) * G.down_to_raise * (1 - gate(bdT, [0.2, 0.35]));
    for (const n of ['browInnerUp', 'browOuterUpLeft', 'browOuterUpRight']) if (n in gt) gt[n] = Math.max(gt[n], down);
    for (const n in gt) {
      if (dt > 0) { const r = smoothDamp(gcur[n], gt[n], gvel[n], gt[n] > gcur[n] ? G.attack : G.release, dt); gcur[n] = r[0]; gvel[n] = r[1]; }
      else { gcur[n] = gt[n]; gvel[n] = 0; }
    }
    let teeth = 0;
    const sRel = S.release || CFG.attack.mouth;
    for (const n of names) {
      let e = Math.max(0, cur[n] + (lo[n] || 0));
      if (/^viseme_/.test(n)) { out[n] = Math.min(1, cur[n]); continue; }
      const sk = SPEECH_KIND(n);
      let f = 1;
      if (sk && A > 1e-4) {
        let keep;
        if (sk === 'smile') keep = ks;
        else if (sk === 'smileopen') keep = ko;
        else if (sk === 'eyesmile') keep = null;
        else { keep = keepOf(n); const cf = CONFLICT(n);
          if (cf === 'open') keep *= 1 - open; else if (cf === 'spread') keep *= 1 - share.spread; else if (cf === 'round') keep *= 1 - share.round;
          if (shout > 0 && /^(jaw|mouthStretch|snarl|mouthUpperUp|noseSneer)/.test(n)) keep = keep + (Math.max(keep, S.shout_keep || 0.85) - keep) * shout; }
        if (keep === null) f = 1 - smW * (1 - (S.eye_smile_floor + (1 - S.eye_smile_floor) * Math.max(smileRet, S.smile_floor || 0)));
        else { f = 1 - A * (1 - keep); if (sk === 'smile') f = Math.max(f, S.smile_floor || 0); }
      }
      if (dt > 0) { if (f <= yk[n]) { yk[n] = f; ykv[n] = 0; } else { const r = smoothDamp(yk[n], f, ykv[n], sRel, dt); yk[n] = r[0]; ykv[n] = r[1]; } }
      else { yk[n] = f; ykv[n] = 0; }
      e *= yk[n];
      if (dt > 0 && e > eOut[n] && Math.max(0, (base[n] || 0) + (lo[n] || 0)) < eOut[n] - 1e-4) e = eOut[n];
      eOut[n] = e;
      if (/^smileOpen/.test(n)) teeth = Math.max(teeth, e);
      else if (/^(mouthUpperUp|snarl)/.test(n)) teeth = Math.max(teeth, 0.8 * e);
      else if (/^mouthSmile/.test(n)) teeth = Math.max(teeth, 0.5 * e);
      let v = Math.max(e, gcur[n] || 0);
      out[n] = Math.min(1, Math.max(0, v));
    }
    const LL = S.lip_lift;
    if (LL && has(LL.key) && speechW > 1e-3) {
      const tg = Math.min(1, Math.max(0, (teeth - LL.teeth[0]) / (LL.teeth[1] - LL.teeth[0])));
      out[LL.key] = Math.min(1, Math.max(out[LL.key] || 0, LL.max * A * tg * (1 - (share.closed || 0))));
    }
    if (has('jawOpen') && speechW > 1e-3) {
      let vo = 0; for (const n of names) if (/^viseme_/.test(n) && JAWC.includes(VCLASS[n])) vo += cur[n];
      const mb = S.min_jaw_base || 0, floor = speechW * Math.min(1, vo * 1.5) * (mb + (S.min_jaw - mb) * Math.min(1, Math.max(0, (teeth - 0.05) / 0.2)));
      out.jawOpen = Math.min(1, Math.max(out.jawOpen, floor));
    }
    let aim = 0; for (const n of names) if (/^eyeLook/.test(n)) aim = Math.max(aim, base[n] || 0);
    look.w += ((look.idle ? 1 : 0) * (1 - Math.min(1, aim * 4)) - look.w) * lpk(0.3);
    if (look.idle && dt > 0) {
      if (clock >= look.next) {
        const Lc = CFG.look, r1 = Math.sin(clock * 91.7) * 43758.5453, r2 = Math.sin(clock * 37.3) * 12345.678, r3 = Math.sin(clock * 13.1) * 9876.54;
        const f = z => z - Math.floor(z);
        look.x0 = look.x; look.y0 = look.y; look.x1 = (2 * f(r1) - 1) * Lc.amp_x; look.y1 = (2 * f(r2) - 1) * Lc.amp_y;
        look.t = 0; look.next = clock + Lc.interval[0] + (Lc.interval[1] - Lc.interval[0]) * f(r3);
      }
      look.t += dt; const u = Math.min(1, look.t / look.dur), s = u * u * (3 - 2 * u);
      look.x = look.x0 + (look.x1 - look.x0) * s; look.y = look.y0 + (look.y1 - look.y0) * s;
    }
    if (look.w > 1e-3) {
      const gx = look.x * look.w, gy = look.y * look.w;
      const add = (n, v) => { if (has(n) && v > 0) out[n] = Math.min(1, out[n] + v); };
      add('eyeLookOutLeft', gx); add('eyeLookInRight', gx); add('eyeLookInLeft', -gx); add('eyeLookOutRight', -gx);
      add('eyeLookUpLeft', gy); add('eyeLookUpRight', gy); add('eyeLookDownLeft', -gy); add('eyeLookDownRight', -gy);
    }
    for (const bn of blinkNames) {
      if (!has(bn)) continue;
      const bb = overlay[bn] || 0;
      bOut[bn] = dt > 0 ? Math.max(bb, (bOut[bn] || 0) - CFG.blink.release_rate * dt) : bb;
      const v = out[bn]; out[bn] = Math.min(1, v + bOut[bn] * (1 - v));
    }
    for (const bn of blinkNames) {
      const side = /Left$/.test(bn) ? 'Left' : 'Right', w = 'eyeWide' + side;
      if (has(w)) out[w] *= 1 - CFG.blink.wide_suppress * Math.min(1, out[bn]);
    }
  }
  function snap() {
    for (const n of names) { cur[n] = /^viseme_/.test(n) ? mx2(overlay[n], base[n]) : (base[n] || 0); vel[n] = 0; holdT[n] = 0; postH[n] = 0; ykv[n] = 0; eOut[n] = cur[n];
      const isB = blinkNames.includes(n); gcur[n] = isB ? 0 : (overlay[n] || 0); gvel[n] = 0; }
    speechW = talking ? 1 : 0; loopW = loopHint && !talking ? 1 : 0; energy = 0;
    for (const bn of blinkNames) bOut[bn] = overlay[bn] || 0;
    step(0);
    for (const n of names) eOut[n] = Math.min(1, out[n] || 0);
  }
  function speed(sec) {
    const mx = Math.max(...Object.entries(CFG.release).filter(([k]) => k !== 'viseme').map(([, v]) => v), ...Object.values(CFG.release_chan || {}));
    speedK = sec > 0 ? sec / (2.5 * mx) : 1; speedT = 0;
  }
  window.RTS_EXPR = {CFG, setup, step, snap, speed, has, val: n => out[n] || 0, setLoop, setTalking, setLook,
    state: () => ({speech: +speechW.toFixed(3), energy: +energy.toFixed(3), loop: +loopW.toFixed(3), look: +look.w.toFixed(3), speedK,
      share: Object.fromEntries(Object.entries(share).map(([k, v]) => [k, +v.toFixed(2)])),
      out: Object.fromEntries(Object.entries(out).filter(([, v]) => v > 1e-3).map(([k, v]) => [k, +v.toFixed(3)]))})};
})();
