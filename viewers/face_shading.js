
(function () {
  if (!window.THREE) return;
  const T = THREE;
  const SKIN = /_skin(\b|_|$)/i, NOT_SKIN = /(teeth|tongue|gum|lash|brow|hair|eye|nail)/i;
  const U = {
    rtsWrap: {value: 0.42},
    rtsScatter: {value: new T.Vector3(0.34, 0.14, 0.09)},
    rtsLobe2: {value: 0.15},
    rtsLobe2Rough: {value: 1.75},
    rtsSkinGain: {value: 0.66},
    rtsSkinAmb: {value: 0.66},
    rtsRimSpec: {value: 0.1},
    rtsRimDiff: {value: 0.3}, rtsRimTint: {value: new T.Vector3(1.0, 0.92, 0.82)},
    rtsBagFloor: {value: 0.2},
    rtsPore: {value: 0.35}, rtsPoreFreq: {value: 3200.0}, rtsPoreRough: {value: 0.18},
    rtsDetail: {value: null}, rtsPoreDet: {value: 1.875}, rtsLipFurrow: {value: 0.35}, rtsLipFreq: {value: 1150.0}, rtsTZone: {value: 0.14},
    rtsLashLine: {value: 0.8}, rtsLashTint: {value: new T.Vector3(0.28, 0.21, 0.18)},
    rtsLashRGB: {value: new T.Vector3(0.016, 0.011, 0.008)}, rtsLashBoost: {value: 0.2},
    rtsPitR: {value: 0.22}, rtsPitDepth: {value: 0.35}, rtsGrainDet: {value: 1.875},
    rtsDiffDetail: {value: 0.35}, rtsPit5: {value: 0.7}, rtsGrain5: {value: 0.6}, rtsSlopeMax: {value: 0.15}, rtsPoreVN: {value: 0.0}, rtsPitFade: {value: new T.Vector2(0.5, 1.0)}, rtsAge: {value: 0.0}, rtsYoung: {value: 0.8}, rtsAgeRough: {value: 0.12},
    rtsWrinkleMap: {value: null}, rtsMaskA: {value: null}, rtsMaskB: {value: null},
    rtsWA: {value: new T.Vector4()}, rtsWB: {value: new T.Vector4()},
    rtsWrinkleGain: {value: 1.0}, rtsCavity: {value: 0.35},
    rtsMouthOpen: {value: 0.0}, rtsTeethDark: {value: 0.55}, rtsTeethBox: {value: new T.Vector4(0, 0.025, 0, 0.03)},
  };
  const done = new WeakSet(), skinMats = [];
  let wrinkles = null, bodyMesh = null, detail = false;

  const DIFF = 'reflectedLight.directDiffuse += irradiance * BRDF_Lambert( material.diffuseColor );';
  const SPEC = 'reflectedLight.directSpecular += irradiance * BRDF_GGX( directLight.direction, geometry.viewDir, geometry.normal, material.specularColor, material.specularF90, material.roughness );';
  const skinChunk = () => {
    let c = T.ShaderChunk.lights_physical_pars_fragment;
    if (c.indexOf(DIFF) < 0) return null;
    c = c.replace(DIFF, `{
		
		
		
		vec3 rtsND = dot( rtsN0, rtsN0 ) > 0.5 ? normalize( mix( rtsN0, geometry.normal, rtsDiffDetail ) ) : geometry.normal;
		float rtsNL = dot( rtsND, directLight.direction );
		
		
		float rtsX = saturate( ( rtsNL + rtsWrap ) / ( 1.0 + rtsWrap ) ) - saturate( rtsNL );
		vec3 rtsD = vec3( saturate( rtsNL ) ) + rtsScatter * rtsX;
		
		
		float rtsBkD = saturate( - dot( directLight.direction, geometry.viewDir ) * 1.4 - 0.1 );
		vec3 rtsLc = directLight.color * mix( vec3( 1.0 ), rtsRimTint, rtsBkD ) * ( 1.0 - rtsRimDiff * rtsBkD );
		reflectedLight.directDiffuse += rtsD * rtsLc * BRDF_Lambert( material.diffuseColor ) * rtsSkinGain;
	}`);
    const AMB = 'reflectedLight.indirectDiffuse += irradiance * BRDF_Lambert( material.diffuseColor );';
    if (c.indexOf(AMB) >= 0) c = c.replace(AMB, 'reflectedLight.indirectDiffuse += irradiance * BRDF_Lambert( material.diffuseColor ) * rtsSkinAmb;');
    if (c.indexOf(SPEC) >= 0) c = c.replace(SPEC, `{
		float rtsNV = saturate( dot( geometry.normal, geometry.viewDir ) );
		float rtsBk = saturate( - dot( directLight.direction, geometry.viewDir ) * 1.4 );
		float rtsSk = 1.0 - rtsBk * ( 1.0 - rtsRimSpec * smoothstep( 0.05, 0.65, rtsNV ) );
		reflectedLight.directSpecular += irradiance * rtsSk * mix(
			BRDF_GGX( directLight.direction, geometry.viewDir, geometry.normal, material.specularColor, material.specularF90, material.roughness ),
			BRDF_GGX( directLight.direction, geometry.viewDir, geometry.normal, material.specularColor, material.specularF90, min( 1.0, material.roughness * rtsLobe2Rough + 0.12 ) ),
			rtsLobe2 );
	}`);
    return c;
  };
  const WRINKLE_N = 'mapN.xy *= normalScale;';

  function patchSkin(m) {
    const chunk = skinChunk(); if (!chunk) return;
    const prev = m.onBeforeCompile;
    m.onBeforeCompile = (sh, r) => {
      if (prev) prev(sh, r);
      for (const k in U) sh.uniforms[k] = U[k];
      let f = sh.fragmentShader;
      const wr = !!(wrinkles && m.normalMap);
      f = f.replace('#include <common>', `#include <common>
uniform float rtsWrap; uniform vec3 rtsScatter; uniform float rtsLobe2, rtsLobe2Rough, rtsSkinGain, rtsSkinAmb, rtsRimSpec, rtsRimDiff;
uniform vec3 rtsRimTint; uniform float rtsBagFloor;
uniform float rtsDiffDetail; vec3 rtsN0 = vec3( 0.0 );
uniform float rtsPore, rtsPoreFreq, rtsPoreRough;
${detail ? 'uniform sampler2D rtsDetail; uniform float rtsPoreDet, rtsLipFurrow, rtsLipFreq, rtsTZone, rtsLashLine, rtsPitR, rtsPitDepth, rtsGrainDet; uniform vec3 rtsLashTint;' : ''}
${detail ? 'uniform float rtsPit5, rtsGrain5, rtsSlopeMax, rtsAge, rtsYoung, rtsAgeRough, rtsPoreVN; uniform vec2 rtsPitFade;' : ''}
float rtsH2( vec2 p ) { return fract( sin( dot( p, vec2( 127.1, 311.7 ) ) ) * 43758.5453 ); }
float rtsVN( vec2 p ) { vec2 i = floor( p ), f = fract( p ); vec2 u = f * f * ( 3.0 - 2.0 * f );
  return mix( mix( rtsH2( i ), rtsH2( i + vec2( 1.0, 0.0 ) ), u.x ), mix( rtsH2( i + vec2( 0.0, 1.0 ) ), rtsH2( i + vec2( 1.0, 1.0 ) ), u.x ), u.y ); }


float rtsPoreN( vec2 uv ) { return 0.65 * rtsVN( uv ) + 0.35 * rtsVN( uv * 2.3 + 17.0 ); }
${detail ? `


float rtsCell( vec2 p ) { vec2 i = floor( p ), f = fract( p ); float d = 1e3;
  for ( int y = -1; y <= 1; y ++ ) for ( int x = -1; x <= 1; x ++ ) { vec2 g = vec2( float( x ), float( y ) );
    vec2 o = vec2( rtsH2( i + g ), rtsH2( i + g + 19.7 ) ); vec2 r = g + o - f; d = min( d, dot( r, r ) ); }
  return d; }
float rtsPoreC( vec2 p ) { return -exp( -rtsCell( p ) / ( rtsPitR * rtsPitR ) ) - 0.35 * exp( -rtsCell( p * 1.7 + 5.3 ) / ( 0.8 * rtsPitR * rtsPitR ) ); }` : ''}
${wr ? 'uniform sampler2D rtsWrinkleMap, rtsMaskA, rtsMaskB; uniform vec4 rtsWA, rtsWB; uniform float rtsWrinkleGain, rtsCavity; float rtsFold = 0.0;' : ''}`);
      f = f.replace('#include <lights_physical_pars_fragment>', chunk);
      if (m.normalMap && f.indexOf('#include <normal_fragment_maps>') >= 0) {
        f = f.replace('#include <normal_fragment_maps>', `#include <normal_fragment_maps>
	{
		
		
		
		rtsN0 = normal;                              
		vec2 rtsP = vUv * rtsPoreFreq, rtsQ = vUv * rtsPoreFreq * 0.42;
		float fwP = length( fwidth( rtsP ) ), fwQ = length( fwidth( rtsQ ) );
		float kP = rtsPore * 1.0e-4 * ( 1.0 - smoothstep( 0.5, 1.0, fwP ) ), kQ = rtsPore * 0.6e-4 * ( 1.0 - smoothstep( 0.5, 1.0, fwQ ) );
		float kL = 0.0; vec2 rtsL = vec2( 0.0 );
${detail ? `		vec4 rtsDt = texture2D( rtsDetail, vUv );
		
		float rtsAgeK = mix( rtsYoung, 1.15, rtsAge );
		float rtsWB = rtsPoreVN, rtsWD = 1.0 - smoothstep( rtsPitFade.x, rtsPitFade.y, fwP );
		kP = rtsPoreDet * rtsPit5 * rtsDt.r * 1.0e-4 * rtsAgeK * ( 1.0 - smoothstep( 0.6, 1.2, fwP ) );
		kQ = rtsGrainDet * rtsGrain5 * rtsDt.r * 0.55e-4 * ( 1.0 - smoothstep( 0.5, 1.0, fwQ ) ) * rtsAgeK;
		roughnessFactor = clamp( roughnessFactor * ( 1.0 + rtsAgeRough * rtsAge ), 0.05, 1.0 );
		
		rtsL = vec2( vUv.x * rtsLipFreq * 0.12, vUv.y * rtsLipFreq );
		kL = rtsLipFurrow * rtsDt.g * 0.6e-4 * ( 1.0 - smoothstep( 0.5, 1.0, length( fwidth( rtsL ) ) ) );
		roughnessFactor = clamp( roughnessFactor * ( 1.0 - rtsTZone * rtsDt.b ), 0.05, 1.0 );
		diffuseColor.rgb *= vec3( 1.0 ) - rtsLashLine * ( 1.0 - rtsDt.a ) * ( vec3( 1.0 ) - rtsLashTint );` : ''}
		if ( kP + kQ + kL > 1e-9 ) {
			float e = 0.3, n0 = rtsPoreN( rtsP ), m0 = rtsVN( rtsQ );
			vec2 gP = vec2( rtsPoreN( rtsP + vec2( e, 0.0 ) ) - n0, rtsPoreN( rtsP + vec2( 0.0, e ) ) - n0 ) / e;
${detail ? `			{ float ec = 0.08, c0 = rtsPoreC( rtsP );
			  vec2 gC = vec2( rtsPoreC( rtsP + vec2( ec, 0.0 ) ) - c0, rtsPoreC( rtsP + vec2( 0.0, ec ) ) - c0 ) / ec;
			  gP = gP * rtsWB + gC * rtsWD * rtsPitDepth / max( rtsPitDepth, 1e-3 ); }` : ''}
			vec2 gQ = vec2( rtsVN( rtsQ + vec2( e, 0.0 ) ) - m0, rtsVN( rtsQ + vec2( 0.0, e ) ) - m0 ) / e;
			
			float lw = 0.35 + 0.65 * rtsVN( rtsL * vec2( 1.0, 0.11 ) + 3.0 );
			float ph = 3.14159 * ( rtsL.y + 0.9 * rtsVN( rtsL * vec2( 0.6, 0.05 ) + 7.0 ) );      
			vec2 gL = vec2( 0.0, 3.14159 * cos( ph ) * sign( sin( ph ) ) ) * lw;
			vec2 dHdxy = vec2( kP * dot( gP, dFdx( rtsP ) ) + kQ * dot( gQ, dFdx( rtsQ ) ) + kL * dot( gL, dFdx( rtsL ) ),
			                   kP * dot( gP, dFdy( rtsP ) ) + kQ * dot( gQ, dFdy( rtsQ ) ) + kL * dot( gL, dFdy( rtsL ) ) );
			vec3 sp = - vViewPosition, vSx = dFdx( sp ), vSy = dFdy( sp );
			vec3 R1 = cross( vSy, normal ), R2 = cross( normal, vSx );
			float fDet = dot( vSx, R1 );
			vec3 vGrad = sign( fDet ) * ( dHdxy.x * R1 + dHdxy.y * R2 );
${detail ? `			{ float tl = length( vGrad ) / max( abs( fDet ), 1e-30 ); if ( tl > rtsSlopeMax ) vGrad *= rtsSlopeMax / tl; }   // face round 5: slope clamp` : ''}
			vec3 nb = abs( fDet ) * normal - vGrad;
			if ( dot( nb, nb ) > 1e-30 ) normal = normalize( nb );
		}
		vec2 rtsR = vUv * rtsPoreFreq * 0.2; float fwR = length( fwidth( rtsR ) );
		roughnessFactor = clamp( roughnessFactor * ( 1.0 + rtsPoreRough * rtsVN( rtsR ) * ( 1.0 - smoothstep( 0.6, 1.2, fwR ) ) ), 0.05, 1.0 );   
	}`);
      }
      if (wr && f.indexOf('#include <normal_fragment_maps>') >= 0) {
        f = f.replace('#include <normal_fragment_maps>', T.ShaderChunk.normal_fragment_maps.replace(WRINKLE_N, `${WRINKLE_N}
	{
		float rtsA = dot( texture2D( rtsMaskA, vUv ), rtsWA ) + dot( texture2D( rtsMaskB, vUv ), rtsWB );
		rtsA = clamp( rtsA * rtsWrinkleGain, 0.0, 1.0 );
		vec4 rtsWn = texture2D( rtsWrinkleMap, vUv );
		mapN = normalize( vec3( mapN.xy + ( rtsWn.xy * 2.0 - 1.0 ) * rtsA, mapN.z ) );   
		rtsFold = rtsA * ( 1.0 - rtsWn.a );
	}`));
        f = f.replace('#include <lights_fragment_begin>', 'diffuseColor.rgb *= 1.0 - rtsCavity * rtsFold;\n#include <lights_fragment_begin>');
      }
      if (m.aoMap && f.indexOf('#include <aomap_fragment>') >= 0) {
        f = f.replace('#include <aomap_fragment>', `#include <aomap_fragment>
#ifdef USE_AOMAP
	{ float rtsAOr = texture2D( aoMap, vUv2 ).r, rtsAOd = clamp( ( rtsAOr - 1.0 ) * aoMapIntensity + 1.0, 0.0, 1.0 );
	  rtsAOd = rtsAOd * rtsAOd;
	  reflectedLight.directDiffuse *= rtsAOd; reflectedLight.directSpecular *= rtsAOd;
	  
	  
	  reflectedLight.indirectDiffuse = max( reflectedLight.indirectDiffuse, diffuseColor.rgb * rtsBagFloor * ( 1.0 - rtsAOr ) ); }
#endif`);
      }
      sh.fragmentShader = f;
    };
    const key = m.customProgramCacheKey ? m.customProgramCacheKey.bind(m) : () => '';
    m.customProgramCacheKey = () => key() + '|rtsSkin' + (wrinkles ? 'W' : '') + (detail ? 'D' : '') + (m.aoMap ? 'A' : '');
    m.needsUpdate = true;
  }

  function patchTeeth(m, tongue) {
    const prev = m.onBeforeCompile;
    m.onBeforeCompile = (sh, r) => {
      if (prev) prev(sh, r);
      sh.uniforms.rtsMouthOpen = U.rtsMouthOpen; sh.uniforms.rtsTeethDark = U.rtsTeethDark; sh.uniforms.rtsTeethBox = U.rtsTeethBox;
      sh.vertexShader = sh.vertexShader.replace('#include <common>', '#include <common>\nvarying vec3 rtsObj;')
        .replace('#include <begin_vertex>', '#include <begin_vertex>\nrtsObj = position;');
      sh.fragmentShader = sh.fragmentShader.replace('#include <common>', `#include <common>
varying vec3 rtsObj; uniform float rtsMouthOpen, rtsTeethDark; uniform vec4 rtsTeethBox;`)
        .replace('#include <lights_fragment_begin>', `{
		
		float lat = abs( rtsObj.x - rtsTeethBox.x ) / max( rtsTeethBox.y, 1e-4 );
		float back = ( rtsTeethBox.z - rtsObj.z ) / max( rtsTeethBox.w, 1e-4 );
		
		
		float shade = smoothstep( 0.55, 1.3, max( lat, back * 1.1 ) );
		${tongue ? `
		
		shade = max( shade, 0.6 + 0.4 * smoothstep( 0.0, 1.0, back ) );
		diffuseColor.rgb *= 1.0 - 0.8 * shade * ( 1.0 - 0.25 * rtsMouthOpen );` :
		`diffuseColor.rgb *= 1.0 - rtsTeethDark * shade * ( 1.0 - 0.7 * rtsMouthOpen ) - 0.02;`}
	}
#include <lights_fragment_begin>`);
    };
    const key = m.customProgramCacheKey ? m.customProgramCacheKey.bind(m) : () => '';
    m.customProgramCacheKey = () => key() + (tongue ? '|rtsTongue' : '|rtsTeeth');
    m.needsUpdate = true;
  }

  function mat(m) {
    if (!m || done.has(m) || !m.name) return;
    const n = m.name;
    if (SKIN.test(n) && !NOT_SKIN.test(n) && (m.isMeshStandardMaterial || m.isMeshPhysicalMaterial)) { done.add(m); skinMats.push(m);
      if (m.aoMap) m.aoMapIntensity = Math.min(m.aoMapIntensity == null ? 1 : m.aoMapIntensity, 0.82); patchSkin(m); return; }
    if (/(teeth|gum)/i.test(n) && (m.isMeshStandardMaterial || m.isMeshPhysicalMaterial)) {
      done.add(m); patchTeeth(m); m.envMapIntensity = Math.min(m.envMapIntensity == null ? 1 : m.envMapIntensity, 0.5); return;
    }
    if (/tongue/i.test(n) && (m.isMeshStandardMaterial || m.isMeshPhysicalMaterial)) {
      done.add(m); patchTeeth(m, true); m.envMapIntensity = Math.min(m.envMapIntensity == null ? 1 : m.envMapIntensity, 0.3); return;
    }
    if (/_cornea$/i.test(n)) { done.add(m); m.envMapIntensity = 1.0; return; }
    if (/_tearline$/i.test(n)) { done.add(m); m.envMapIntensity = 0.3; if ('specularIntensity' in m) m.specularIntensity = 0.7; return; }
    if (/_eye$/i.test(n)) { done.add(m); m.envMapIntensity = 0.8; return; }
    if (/_eyeocc$/i.test(n)) { done.add(m); m.depthWrite = false; m.transparent = true; return; }
    if (/lash/i.test(n) && m.alphaTest > 0) { done.add(m); m.alphaToCoverage = true; m.alphaTest = Math.min(m.alphaTest, 0.2);
      m.flatShading = true; m.envMapIntensity = Math.min(m.envMapIntensity == null ? 1 : m.envMapIntensity, 0.35);
      const prev = m.onBeforeCompile;
      m.onBeforeCompile = (sh, r) => {
        if (prev) prev(sh, r);
        if (!m.map || sh.fragmentShader.indexOf('#include <alphatest_fragment>') < 0) return;
        const an = r && r.capabilities ? r.capabilities.getMaxAnisotropy() : 1;
        if (an > 1 && m.map.anisotropy !== an) { m.map.anisotropy = an; m.map.needsUpdate = true; }
        sh.uniforms.rtsLashRGB = U.rtsLashRGB; sh.uniforms.rtsLashBoost = U.rtsLashBoost;
        sh.fragmentShader = 'uniform vec3 rtsLashRGB; uniform float rtsLashBoost;\n' + sh.fragmentShader.replace('#include <alphatest_fragment>', `{
		
		
		vec2 rtsTs = vec2( textureSize( map, 0 ) ), rtsDx = dFdx( vUv ) * rtsTs, rtsDy = dFdy( vUv ) * rtsTs;
		float rtsMa = max( dot( rtsDx, rtsDx ), dot( rtsDy, rtsDy ) ), rtsMi = min( dot( rtsDx, rtsDx ), dot( rtsDy, rtsDy ) );
		float rtsLod = max( 0.0, 0.5 * log2( max( rtsMi, rtsMa / 256.0 ) ) );
		diffuseColor.a *= 1.0 + rtsLashBoost * rtsLod;
		diffuseColor.a = clamp( ( diffuseColor.a - 0.42 ) / max( fwidth( diffuseColor.a ), 1e-4 ) + 0.5, 0.0, 1.0 );
		if ( diffuseColor.a < 0.02 ) discard;
		diffuseColor.rgb = rtsLashRGB;
	}`).replace('#include <normal_fragment_begin>', '#include <normal_fragment_begin>\n\tnormal = normalize( vViewPosition ); geometryNormal = normal;   // face round 5: a strand seen from the front, not the card plane');
      };
      const key = m.customProgramCacheKey ? m.customProgramCacheKey.bind(m) : () => '';
      m.customProgramCacheKey = () => key() + '|rtsLash3';
      m.needsUpdate = true; return; }
    if (/(lash|brow)/i.test(n) && m.alphaTest > 0) { done.add(m); m.alphaToCoverage = true; m.alphaTest = /eyebrow/i.test(n) ? (/female/i.test(n) ? 0.3 : 0.34) : Math.min(m.alphaTest, 0.2); m.needsUpdate = true; return; }
  }

  function teethBox(o) {
    const g = o.geometry; if (!g) return;
    g.computeBoundingBox(); const b = g.boundingBox;
    const hw = 0.5 * (b.max.x - b.min.x), cx = 0.5 * (b.max.x + b.min.x);
    U.rtsTeethBox.value.set(cx, Math.max(hw * 0.62, 0.004), b.max.z, Math.max((b.max.z - b.min.z) * 0.55, 0.004));
  }

  const W = k => (bodyMesh && bodyMesh.morphTargetDictionary[k] !== undefined) ? bodyMesh.morphTargetInfluences[bodyMesh.morphTargetDictionary[k]] || 0 : 0;
  function drive() {
    const open = Math.min(1, 2.6 * W('jawOpen') + 0.45 * (W('smileOpenLeft') + W('smileOpenRight')) +
      0.25 * (W('mouthUpperUpLeft') + W('mouthUpperUpRight')) + 1.2 * W('viseme_aa') + 0.6 * W('viseme_O'));
    U.rtsMouthOpen.value = open;
    U.rtsAge.value = W('cust_age_lines_pos');
    if (wrinkles) {
      const r = wrinkles.regions, a = U.rtsWA.value, b = U.rtsWB.value, v = [0, 0, 0, 0, 0, 0, 0, 0];
      for (let i = 0; i < r.length && i < 8; i++) {
        let s = 0; const d = r[i].drivers || {};
        for (const k in d) s += d[k] * W(k);
        const g0 = r[i].gain == null ? 1 : r[i].gain, ag = r[i].age, a = ag ? W(ag.key) : 0;
        const st = (r[i].start || 0) + (ag ? (ag.start - (r[i].start || 0)) * a : 0), gn = g0 + (ag ? (ag.gain - g0) * a : 0);
        const t = Math.min(1, Math.max(0, (s - st) / Math.max(1e-3, (r[i].full || 1) - st)));
        v[i] = Math.min(1, t * t * (3 - 2 * t) * gn + (ag ? ag.rest * a : 0));
      }
      a.set(v[0], v[1], v[2], v[3]); b.set(v[4], v[5], v[6], v[7]);
    }
  }


  const LASH = {on: true, list: [], scene: null, hooked: false, n: 0, ms: 0};
  function lashFindBody(o) {
    let r = o; while (r.parent && !r.parent.isScene) r = r.parent;
    const out = []; r.traverse(x => { if (x.isMesh && /_body$/.test(x.name) && x.morphTargetInfluences) out.push(x); });
    return out;
  }
  function lashPrep(mesh) {
    let sp = mesh.userData.rts_lash_bind; if (typeof sp === 'string') { try { sp = JSON.parse(sp); } catch (e) { return null; } }
    if (!sp || !sp.cards) return null;
    const bodies = lashFindBody(mesh); if (!bodies.length) return null;
    const key = (x, y, z) => Math.round(x * 2e5) + ',' + Math.round(y * 2e5) + ',' + Math.round(z * 2e5);
    const hash = new Map();
    for (const b of bodies) { const pa = b.geometry.attributes.position;
      for (let i = 0; i < pa.count; i++) { const k = key(pa.getX(i), pa.getY(i), pa.getZ(i)); if (!hash.has(k)) hash.set(k, [b, i]); } }
    const nr = sp.refs.length / 3, refs = []; let miss = 0;
    const nearest = (x, y, z) => { let best = null, bd = 1e-8;
      for (const b of bodies) { const pa = b.geometry.attributes.position;
        for (let i = 0; i < pa.count; i++) { const d = (pa.getX(i) - x) ** 2 + (pa.getY(i) - y) ** 2 + (pa.getZ(i) - z) ** 2; if (d < bd) { bd = d; best = [b, i]; } } }
      return best; };
    for (let i = 0; i < nr; i++) { const x = sp.refs[3 * i], y = sp.refs[3 * i + 1], z = sp.refs[3 * i + 2];
      const h = hash.get(key(x, y, z)) || nearest(x, y, z); if (!h) miss++; refs.push(h || [bodies[0], 0]); }
    const la = mesh.geometry.attributes.position, lmap = new Map();
    for (let i = 0; i < sp.n; i++) lmap.set(key(sp.lp[3 * i], sp.lp[3 * i + 1], sp.lp[3 * i + 2]), i);
    const glb2b = new Int32Array(la.count); let lmiss = 0;
    for (let i = 0; i < la.count; i++) { let v = lmap.get(key(la.getX(i), la.getY(i), la.getZ(i)));
      if (v === undefined) { let bd = 1e-8; for (let b = 0; b < sp.n; b++) { const d = (sp.lp[3 * b] - la.getX(i)) ** 2 + (sp.lp[3 * b + 1] - la.getY(i)) ** 2 + (sp.lp[3 * b + 2] - la.getZ(i)) ** 2; if (d < bd) { bd = d; v = b; } } }
      glb2b[i] = v === undefined ? -1 : v; if (v === undefined) lmiss++; }
    if (miss || lmiss) console.warn('rts_lash_bind', mesh.name, 'unmatched body refs', miss, 'lash vertices', lmiss);
    const P = new Float64Array(nr * 3), X = new Float64Array(sp.n * 3);
    const L = {mesh, sp, refs, glb2b, P, X, miss, lmiss, rest: Float32Array.from(la.array), active: false, saved: null};
    const pb = mesh.onBeforeRender, pa = mesh.onAfterRender;
    mesh.onBeforeRender = function () { const inf = mesh.morphTargetInfluences;
      if (LASH.on && L.active && inf) { if (!L.saved || L.saved.length !== inf.length) L.saved = new Float32Array(inf.length);
        for (let i = 0; i < inf.length; i++) { L.saved[i] = inf[i]; inf[i] = 0; } }
      if (pb) pb.apply(this, arguments); };
    mesh.onAfterRender = function () { if (pa) pa.apply(this, arguments);
      const inf = mesh.morphTargetInfluences; if (LASH.on && L.active && inf && L.saved) { for (let i = 0; i < inf.length; i++) inf[i] = L.saved[i]; } };
    return L;
  }
  function lashPose(L) {
    const act = new Map();
    for (const [b] of L.refs) if (!act.has(b)) { const inf = b.morphTargetInfluences, a = [];
      for (let k = 0; k < inf.length; k++) if (inf[k] > 1e-5) a.push(k); act.set(b, a); }
    const P = L.P;
    L.refs.forEach(([b, i], r) => { const g = b.geometry, pa = g.attributes.position, ma = g.morphAttributes.position, inf = b.morphTargetInfluences, rel = g.morphTargetsRelative;
      let x = pa.getX(i), y = pa.getY(i), z = pa.getZ(i);
      for (const k of act.get(b)) { const m = ma[k], w = inf[k];
        if (rel) { x += w * m.getX(i); y += w * m.getY(i); z += w * m.getZ(i); } else { x += w * (m.getX(i) - pa.getX(i)); y += w * (m.getY(i) - pa.getY(i)); z += w * (m.getZ(i) - pa.getZ(i)); } }
      P[3 * r] = x; P[3 * r + 1] = y; P[3 * r + 2] = z; });
  }
  const V3 = {sub: (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]], add: (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]],
    mul: (a, s) => [a[0] * s, a[1] * s, a[2] * s], dot: (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2],
    cross: (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]],
    norm: a => { const l = Math.hypot(a[0], a[1], a[2]); return l > 1e-15 ? [a[0] / l, a[1] / l, a[2] / l] : [0, 0, 0]; },
    lerp: (a, b, t) => [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t]};
  function lashAvg(P, idx, w) { let x = 0, y = 0, z = 0; for (let k = 0; k < idx.length; k++) { const i = idx[k]; x += w[k] * P[3 * i]; y += w[k] * P[3 * i + 1]; z += w[k] * P[3 * i + 2]; } return [x, y, z]; }
  function lashStations(P, I, W) { return I.map((idx, j) => lashAvg(P, idx, W[j])); }
  function lashTan(M) { const K = M.length; return M.map((_, j) => V3.sub(M[Math.min(j + 2, K - 1)], M[Math.max(j - 2, 0)])); }
  function lashGS(e1, d) { e1 = V3.norm(e1); const e2 = V3.norm(V3.sub(d, V3.mul(e1, V3.dot(d, e1)))); return [e1, e2, V3.cross(e1, e2)]; }
  function lashCard(L, c, cs, age) {
    const P = L.P, C = L.sp.consts, K = c.K, n = c.verts.length;
    const M = lashStations(P, c.M_idx, c.M_w), U = lashStations(P, c.U_idx, c.U_w), T = lashTan(M), D = M.map((m, j) => V3.sub(U[j], m));
    let cj = null, th = null;
    if (c.C_idx) {
      const Cn = V3.mul(V3.add(lashAvg(P, c.C_idx[0], c.C_w[0]), lashAvg(P, c.C_idx[1], c.C_w[1])), 0.5);
      cj = M.map((m, j) => { const r0 = [c.R0[3 * j], c.R0[3 * j + 1], c.R0[3 * j + 2]], dd = [c.D[3 * j], c.D[3 * j + 1], c.D[3 * j + 2]];
        const v = V3.dot(V3.sub(V3.sub(m, Cn), r0), dd) / Math.max(V3.dot(dd, dd), 1e-18);
        return Math.min(1, Math.max(0, (v - C.DZ) / (1 - C.DZ))); });
      th = c.theta.map((t, j) => t * cj[j]);
    }
    const O = [], F = [];
    for (let v = 0; v < n; v++) { const j = c.j[v], f = c.t[v];
      O.push(V3.lerp(M[j], M[j + 1], f)); F.push(lashGS(V3.lerp(T[j], T[j + 1], f), V3.lerp(D[j], D[j + 1], f))); }
    if (th && c.Q_idx) {
      const Q = lashStations(P, c.Q_idx, c.Q_w), Dn = lashStations(P, c.N_idx, c.N_w), TQ = lashTan(Q);
      const N = Q.map((q, j) => V3.norm(V3.mul(V3.cross(TQ[j], V3.sub(Dn[j], q)), c.n_sign)));
      const st = new Array(K).fill(Infinity), clr = C.CLEAR_MM * 1e-3, sg = c.roll_sign;
      for (let v = 0; v < n; v++) { const rw = c.roll_w[v]; if (rw <= 0.05) continue;
        const j = c.j[v], f = c.t[v], q = V3.lerp(Q[j], Q[j + 1], f), nn = V3.norm(V3.lerp(N[j], N[j + 1], f));
        const pv = [c.pv[3 * v], c.pv[3 * v + 1], c.pv[3 * v + 2]], r = [c.l[3 * v] - pv[0], c.l[3 * v + 1] - pv[1], c.l[3 * v + 2] - pv[2]];
        const [e1, e2, e3] = F[v], n1 = V3.dot(e1, nn), n2 = V3.dot(e2, nn), n3 = V3.dot(e3, nn);
        const base = V3.add(O[v], V3.add(V3.add(V3.mul(e1, pv[0]), V3.mul(e2, pv[1])), V3.mul(e3, pv[2])));
        const Cc = V3.dot(V3.sub(base, q), nn) + r[0] * n1, A = r[1] * n2 + r[2] * n3, B = sg * (-r[2] * n2 + r[1] * n3);
        if (Cc + A < clr) continue;
        const R = Math.hypot(A, B) + 1e-15, k = Math.min(1, Math.max(-1, (clr - Cc) / R)); if (k <= -1) continue;
        let hi = Math.atan2(B, A) + Math.acos(k); if (hi < 0) hi += 2 * Math.PI;
        const am = hi / Math.max(rw, 1e-3);
        if (f < 0.999) st[j] = Math.min(st[j], am); if (f > 0.001) st[j + 1] = Math.min(st[j + 1], am);
      }
      th = th.map((t, j) => Math.min(t, st[j]));
    }
    const X = L.X, latR = c.lat_roll != null ? c.lat_roll : C.LAT_ROLL, latC = c.lat_closed != null ? c.lat_closed : C.LAT_CLOSED;
    for (let v = 0; v < n; v++) {
      let l0 = c.l[3 * v], l1 = c.l[3 * v + 1], l2 = c.l[3 * v + 2], cw = cs || 0;
      if (th) { const j = c.j[v], f = c.t[v], lat = c.lat_w[v], rw = c.roll_w[v];
        const a = (th[j] * (1 - f) + th[j + 1] * f) * rw * (1 - latR * lat) * c.roll_sign;
        const cv = cj[j] * (1 - f) + cj[j + 1] * f, sc = Math.max(0.2, 1 - (1 - latC + (c.lat_age || 0) * age) * cv * lat * rw);
        const p0 = c.pv[3 * v], p1 = c.pv[3 * v + 1], p2 = c.pv[3 * v + 2], r1 = l1 - p1, r2 = l2 - p2, ca = Math.cos(a), sa = Math.sin(a);
        l0 = p0 + (l0 - p0) * sc; l1 = p1 + (r1 * ca - r2 * sa) * sc; l2 = p2 + (r1 * sa + r2 * ca) * sc; cw = cv; }
      if (c.corr) {
        l0 += c.corr[3 * v] * cw; l1 += c.corr[3 * v + 1] * cw; l2 += c.corr[3 * v + 2] * cw;
        if (age && c.corr_age) { const k = cw * age; l0 += c.corr_age[3 * v] * k; l1 += c.corr_age[3 * v + 1] * k; l2 += c.corr_age[3 * v + 2] * k; } }
      const [e1, e2, e3] = F[v], o = O[v];
      let x = o[0] + e1[0] * l0 + e2[0] * l1 + e3[0] * l2, y = o[1] + e1[1] * l0 + e2[1] * l1 + e3[1] * l2, z = o[2] + e1[2] * l0 + e2[2] * l1 + e3[2] * l2;
      const ws = c.S_wt[v];
      if (ws > 0) {
        const t0 = c.S_tri[3 * v], t1 = c.S_tri[3 * v + 1], t2 = c.S_tri[3 * v + 2];
        const A = [P[3 * t0], P[3 * t0 + 1], P[3 * t0 + 2]], B = [P[3 * t1], P[3 * t1 + 1], P[3 * t1 + 2]], Cc = [P[3 * t2], P[3 * t2 + 1], P[3 * t2 + 2]];
        const b0 = c.S_bary[3 * v], b1 = c.S_bary[3 * v + 1], b2 = c.S_bary[3 * v + 2];
        const f1 = V3.norm(V3.sub(B, A)), f3 = V3.norm(V3.cross(V3.sub(B, A), V3.sub(Cc, A))), f2 = V3.cross(f3, f1);
        const s0 = c.S_l[3 * v], s1 = c.S_l[3 * v + 1], s2 = c.S_l[3 * v + 2];
        const sx = A[0] * b0 + B[0] * b1 + Cc[0] * b2 + f1[0] * s0 + f2[0] * s1 + f3[0] * s2,
          sy = A[1] * b0 + B[1] * b1 + Cc[1] * b2 + f1[1] * s0 + f2[1] * s1 + f3[1] * s2,
          sz = A[2] * b0 + B[2] * b1 + Cc[2] * b2 + f1[2] * s0 + f2[2] * s1 + f3[2] * s2;
        x = x * (1 - ws) + sx * ws; y = y * (1 - ws) + sy * ws; z = z * (1 - ws) + sz * ws; }
      if (c.K_tri) {
        const K = c.K_tri.length / (3 * n), ft = C.CLAMP_FOOT, win = C.CLAMP_WIN_MM * 1e-3, trig = (c.clamp_trig_mm != null ? c.clamp_trig_mm : 1e9) * 1e-3;
        const fade = c.clamp_close ? Math.min(1, Math.max(0, (c.clamp_close[1] - cw) / Math.max(c.clamp_close[1] - c.clamp_close[0], 1e-6))) : 1;
        for (let g = 0; g < K; g++) {
          const o3 = 3 * (v * K + g), i0 = c.K_tri[o3]; if (i0 < 0) continue;
          const i1 = c.K_tri[o3 + 1], i2 = c.K_tri[o3 + 2];
          const ax = P[3 * i0], ay = P[3 * i0 + 1], az = P[3 * i0 + 2];
          const e1x = P[3 * i1] - ax, e1y = P[3 * i1 + 1] - ay, e1z = P[3 * i1 + 2] - az, e2x = P[3 * i2] - ax, e2y = P[3 * i2 + 1] - ay, e2z = P[3 * i2 + 2] - az;
          let nx = e1y * e2z - e1z * e2y, ny = e1z * e2x - e1x * e2z, nz = e1x * e2y - e1y * e2x; const nl = Math.hypot(nx, ny, nz); if (nl < 1e-30) continue;
          nx /= nl; ny /= nl; nz /= nl;
          const qx = x - ax, qy = y - ay, qz = z - az, h = qx * nx + qy * ny + qz * nz, tg = c.K_h[v * K + g];
          if (!(h < Math.min(tg, trig) && h > -win)) continue;
          const px = qx - h * nx, py = qy - h * ny, pz = qz - h * nz;
          const d00 = e1x * e1x + e1y * e1y + e1z * e1z, d01 = e1x * e2x + e1y * e2y + e1z * e2z, d11 = e2x * e2x + e2y * e2y + e2z * e2z;
          const d20 = px * e1x + py * e1y + pz * e1z, d21 = px * e2x + py * e2y + pz * e2z, den = Math.max(d00 * d11 - d01 * d01, 1e-30);
          const bv = (d11 * d20 - d01 * d21) / den, bw = (d00 * d21 - d01 * d20) / den, bu = 1 - bv - bw;
          if (bu >= -ft && bv >= -ft && bw >= -ft) { const k = (tg - h) * fade; x += k * nx; y += k * ny; z += k * nz; }
        }
      }
      const bi = c.verts[v]; X[3 * bi] = x; X[3 * bi + 1] = y; X[3 * bi + 2] = z;
    }
    return cj ? cj.reduce((a, b) => a + b, 0) / cj.length : null;
  }
  function lashUpdate(L) {
    const pa = L.mesh.geometry.attributes.position, arr = pa.array;
    if (!LASH.on) { if (L.dirty) { arr.set(L.rest); pa.needsUpdate = true; L.dirty = false; } return; }
    lashPose(L);
    const b0 = L.refs[0][0], ai = b0.morphTargetDictionary ? b0.morphTargetDictionary[(L.sp.consts && L.sp.consts.AGE_KEY) || 'cust_age_lines_pos'] : undefined;
    const age = ai === undefined ? 0 : (b0.morphTargetInfluences[ai] || 0), cs = {};
    for (const c of L.sp.cards) if (c.upper) cs[c.side] = lashCard(L, c, 0, age);
    for (const c of L.sp.cards) if (!c.upper) lashCard(L, c, cs[c.side] || 0, age);
    const X = L.X;
    for (let i = 0; i < L.glb2b.length; i++) { const b = L.glb2b[i]; if (b < 0) continue; arr[3 * i] = X[3 * b]; arr[3 * i + 1] = X[3 * b + 1]; arr[3 * i + 2] = X[3 * b + 2]; }
    pa.needsUpdate = true; L.dirty = true;
    L.active = true;
  }
  function lashTick() {
    const t0 = performance.now();
    let top = LASH.root; while (top && top.parent && !top.parent.isScene) top = top.parent;
    if ((LASH.n++ % 30) === 0 && top) top.traverse(o => {
      if (o.isMesh && o.userData && o.userData.rts_lash_bind && !LASH.list.some(L => L.mesh === o)) { const L = lashPrep(o); if (L) { LASH.list.push(L); o.frustumCulled = false; } } });
    for (const L of LASH.list) { let vis = true; for (let o = L.mesh; o; o = o.parent) if (!o.visible) { vis = false; break; } if (vis) lashUpdate(L); }
    LASH.ms = 0.9 * LASH.ms + 0.1 * (performance.now() - t0);
  }
  function lashHook(obj) {
    let s = obj; while (s.parent) s = s.parent;
    if (!s.isScene || LASH.hooked) return;
    const prev = s.onBeforeRender; s.onBeforeRender = function () { if (prev) prev.apply(this, arguments); lashTick(); };
    LASH.hooked = true; LASH.scene = s;
  }
  function lashSetup(root) {
    LASH.root = root;
    let first = null; root.traverse(o => { if (!first && o.isMesh && /_body$/.test(o.name)) first = o; });
    if (!first) return;
    const prev = first.onBeforeRender;
    first.onBeforeRender = function () { lashHook(first); if (prev) prev.apply(this, arguments); };
  }

  function setup(gltf) {
    const root = gltf.scene || gltf;
    let body = null, teeth = [];
    root.traverse(o => {
      if (!o.isMesh) return;
      if (/_body$/.test(o.name) && o.morphTargetDictionary) body = o;
      if (/_teeth$/.test(o.name) || /teeth/i.test(o.name)) teeth.push(o);
    });
    if (teeth.length) teethBox(teeth[0]);
    if (!body) { root.traverse(o => { if (o.isMesh) for (const m of [].concat(o.material)) mat(m); }); return; }
    bodyMesh = body;
    lashSetup(root);
    if (/female/i.test(body.name)) { U.rtsSkinGain.value = 0.61; U.rtsSkinAmb.value = 0.63; U.rtsLashRGB.value.set(0.011, 0.0075, 0.0058); }
    let spec = body.userData.rts_wrinkles;
    if (typeof spec === 'string') { try { spec = JSON.parse(spec); } catch (e) { spec = null; } }
    const finish = () => {
      root.traverse(o => { if (o.isMesh) for (const m of [].concat(o.material)) mat(m); });
      const prev = body.onBeforeRender;
      body.onBeforeRender = function () { drive(); if (prev) prev.apply(this, arguments); };
    };
    if (spec && spec.textures && gltf.parser) {
      const t = spec.textures;
      Promise.all([t.normal, t.maskA, t.maskB, t.detail].map(i => i == null ? Promise.resolve(null) : gltf.parser.getDependency('texture', i)))
        .then(([n, a, b, dm]) => {
          if (n && a) {
            for (const x of [n, a, b, dm]) if (x) { x.encoding = T.LinearEncoding; x.flipY = false; x.needsUpdate = true; }
            U.rtsWrinkleMap.value = n; U.rtsMaskA.value = a; U.rtsMaskB.value = b || a;
            if (dm) { U.rtsDetail.value = dm; detail = true; }
            wrinkles = {regions: spec.regions || []};
            for (const m of skinMats) m.needsUpdate = true;
          }
          finish();
        }, e => { console.warn('rts_wrinkles', e); finish(); });
    } else finish();
  }

  window.RTS_FACE = {mat, setup, uniforms: U, drive,
    lashBind: on => { if (on !== undefined) LASH.on = !!on; return LASH.on; },
    lashSpec: (name, spec) => { let o = null, top = LASH.root; while (top && top.parent && !top.parent.isScene) top = top.parent;
      if (top) top.traverse(x => { if (!o && x.isMesh && x.name === name) o = x; });
      if (!o) return 'no mesh ' + name; const i = LASH.list.findIndex(L => L.mesh === o); if (i >= 0) { const L0 = LASH.list[i]; o.geometry.attributes.position.array.set(L0.rest); LASH.list.splice(i, 1); }
      o.userData.rts_lash_bind = spec; const L = lashPrep(o); if (L) { LASH.list.push(L); o.frustumCulled = false; } return L ? 'ok' : 'prep failed'; },
    lashInfo: () => ({on: LASH.on, hooked: LASH.hooked, ms: +LASH.ms.toFixed(3), meshes: LASH.list.map(L => ({name: L.mesh.name, n: L.sp.n,
      glb: L.glb2b.length, refs: L.refs.length, miss: L.miss, lmiss: L.lmiss}))}),
    lashBench: n => { const vis = L => { for (let o = L.mesh; o; o = o.parent) if (!o.visible) return false; return true; };
      const Ls = LASH.list.filter(vis), t0 = performance.now(); for (let i = 0; i < (n || 100); i++) for (const L of Ls) lashUpdate(L);
      return {ms: +((performance.now() - t0) / (n || 100)).toFixed(4), meshes: Ls.map(L => L.mesh.name)}; },
    lashPositions: name => { const L = LASH.list.find(x => x.mesh.name === name) || LASH.list[0]; return L ? Array.from(L.X) : null; },
    info: () => ({wrinkles: !!wrinkles, body: bodyMesh && bodyMesh.name,
    mouthOpen: U.rtsMouthOpen.value, wa: U.rtsWA.value.toArray(), wb: U.rtsWB.value.toArray()})};
})();
