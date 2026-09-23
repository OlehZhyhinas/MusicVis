// GLSL for V2: a copy of V1's shader library with V2 uniforms, the chain op
// snippets, the emitter fields, and the genome -> shader builders.
// Numeric genome parameters are uniforms; only structure changes the source,
// so compiled programs are cached by structuralKey().

import { FLAME_VARIATION_GLSL } from './variations';
import type { EmitterKind, Genome, OpGene } from './genome';

const HEAD = /* glsl */ `#version 300 es
precision highp float;
precision highp int;
precision highp sampler2D;
`;

const COMMON = /* glsl */ `
const float PI = 3.14159265359;
const float TAU = 6.28318530718;
float hash12(vec2 p) {
  vec3 p3 = fract(vec3(p.xyx) * 0.1031);
  p3 += dot(p3, p3.yzx + 33.33);
  return fract((p3.x + p3.y) * p3.z);
}
float hash11(float p) {
  p = fract(p * 0.1031);
  p *= p + 33.33;
  p *= p + p;
  return fract(p);
}
vec2 hash22(vec2 p) {
  vec3 p3 = fract(vec3(p.xyx) * vec3(0.1031, 0.1030, 0.0973));
  p3 += dot(p3, p3.yzx + 33.33);
  return fract((p3.xx + p3.yz) * p3.zy);
}
float vnoise(vec2 p) {
  vec2 i = floor(p), f = fract(p);
  vec2 u = f * f * (3.0 - 2.0 * f);
  float a = hash12(i), b = hash12(i + vec2(1, 0)), c = hash12(i + vec2(0, 1)), d = hash12(i + vec2(1, 1));
  return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}
float fbm2(vec2 p) { return vnoise(p) * 0.65 + vnoise(p * 2.03 + 17.1) * 0.35; }
vec2 curlNoise(vec2 p, float t) {
  const float e = 0.05;
  vec2 o = vec2(t * 0.13, -t * 0.09);
  float n1 = fbm2(p + o + vec2(0, e)), n2 = fbm2(p + o - vec2(0, e));
  float n3 = fbm2(p + o + vec2(e, 0)), n4 = fbm2(p + o - vec2(e, 0));
  return vec2(n1 - n2, -(n3 - n4)) / (2.0 * e);
}
mat2 rot2(float a) { float c = cos(a), s = sin(a); return mat2(c, s, -s, c); }
float luma(vec3 c) { return dot(c, vec3(0.2126, 0.7152, 0.0722)); }
vec3 hsv2rgb(vec3 c) {
  vec3 p = abs(fract(c.xxx + vec3(0.0, 2.0 / 3.0, 1.0 / 3.0)) * 6.0 - 3.0);
  return c.z * mix(vec3(1.0), clamp(p - 1.0, 0.0, 1.0), c.y);
}
vec3 hueRotate(vec3 c, float a) {
  const vec3 k = vec3(0.57735);
  float ca = cos(a);
  return c * ca + cross(k, c) * sin(a) + k * dot(k, c) * (1.0 - ca);
}
`;

const LIB = /* glsl */ `
uniform vec2 uRes;
uniform float uAspect, uTime, uPhase, uDt, uF60, uSpeed;
uniform float uBeat, uBar, uBars, uBeats, uBeatPulse;
uniform float uSpin, uSpinStep;
uniform vec4 uStem, uOnset, uPres;
uniform float uAct, uBuild, uDrop, uLoud, uMelody, uKeyHue;
uniform vec3 uColA, uColB, uColC;
uniform float uDecay;
uniform float uLayerK;   // 1 - decay in the feedback pass, 1 in the composite: static fields look the same on either layer
uniform float uAccum;    // 1 in the feedback pass, ~6 in the composite: accumulating emitters look the same on either layer
uniform vec4 uOpA[6], uOpB[6];
uniform vec4 uEm[12];
uniform vec4 uInk[6], uBlob[6];
uniform vec4 uSeg[48];
uniform vec4 uSegZ[12];
uniform int uSegN;
uniform float uChroma[12];
uniform sampler2D uWave, uSpec;

float waveAt(float x) { return texture(uWave, vec2(x, 0.5)).r; }
float specAt(float x) { return texture(uSpec, vec2(clamp(x, 0.0, 1.0), 0.5)).r; }
float chromaAt(float i) { return uChroma[int(mod(i, 12.0))]; }
vec3 pal(float t) {
  t = fract(t) * 3.0;
  if (t < 1.0) return mix(uColA, uColB, t);
  if (t < 2.0) return mix(uColB, uColC, t - 1.0);
  return mix(uColC, uColA, t - 2.0);
}
float glow(float d, float w) { return exp(-d * d / (w * w)); }
float sdSeg(vec2 p, vec2 a, vec2 b) {
  vec2 pa = p - a, ba = b - a;
  float h = clamp(dot(pa, ba) / max(dot(ba, ba), 1e-8), 0.0, 1.0);
  return length(pa - ba * h);
}
float px() { return 1.0 / uRes.y; }
float fbm4(vec2 p) {
  float s = 0.0, a = 0.5;
  for (int i = 0; i < 4; i++) { s += a * vnoise(p); p = rot2(0.6) * p * 2.03 + 11.7; a *= 0.5; }
  return s / 0.9375;
}
vec3 keyCol(float pc, float s, float v) { return hsv2rgb(vec3(fract(uKeyHue + pc * 7.0 / 12.0), s, v)); }
vec3 lin(vec3 c) { return c * c; }
`;

// ------------------------------------------------------------- chain ops
// Each snippet transforms `p` in place; OA / OB are the op's uniform slots,
// filled per frame by the engine (see engine.ts packOp).

const OP_GLSL: Record<string, string> = {
  zoom: `{ vec2 d = p - OA.xy; float r = length(d); float z = 1.0 + OA.z * mix(1.0, 0.5 + r * 1.2, OA.w); p = OA.xy + d / z; }`,
  rotate: `{ p = OA.xy + rot2(OA.z) * (p - OA.xy); }`,
  translate: `{ p += OA.xy; }`,
  swirl: `{ vec2 d = p - OA.xy; float r = length(d); p = OA.xy + rot2(OA.z / (r * OA.w + 0.4)) * d; }`,
  twist: `{ vec2 d = p - OA.xy; p = OA.xy + rot2(OA.z * length(d) * 8.0) * d; }`,
  ripple: `{ if (OA.w > 0.5) { float r = length(p); p += normalize(p + 1e-5) * OA.x * sin(r * OA.y - OA.z); } else { p.y += OA.x * sin(p.x * OA.y - OA.z); } }`,
  noise: `{ p += curlNoise(p * OA.y + 4.0, OA.z) * OA.x; }`,
  mirror: `{ if (OA.x < 0.5) p.x = abs(p.x); else if (OA.x < 1.5) p.y = abs(p.y); else p = abs(p); }`,
  tile: `{ vec2 h = vec2(uAspect, 1.0) * 0.5; vec2 q = mod(p * OA.x + h, 4.0 * h); p = abs(q - 2.0 * h) - h; }`,
  polar: `{ float a = mod(atan(p.y, p.x) + OA.y + PI, TAU) - PI; p = vec2(a / PI * uAspect * 0.5, length(p) * 2.0 * OA.x - 0.5); }`,
  kaleido: `{ float seg = TAU / OA.x; float a = mod(atan(p.y, p.x) + OA.y, seg); a = abs(a - seg * 0.5); p = length(p) * vec2(cos(a), sin(a)); }`,
};

function opCode(o: OpGene, i: number): string {
  const src = o.op.startsWith('v_') ? `{ p = mix(p, V_${o.op.slice(2)}(p * OA.y) / OA.y, OA.x); }` : OP_GLSL[o.op];
  return '  ' + src.replace(/OA/g, `uOpA[${i}]`).replace(/OB/g, `uOpB[${i}]`) + '\n';
}

// -------------------------------------------------------------- emitters
// Field emitters: vec3 em_<kind>(vec2 p) in p space. EA..ED are the emitter's
// four uniform vec4 slots (EA = gain, hue, ...). "Accumulating" emitters are
// tuned for the feedback layer and use uAccum; "static" fields use uLayerK.

const EMIT_GLSL: Partial<Record<EmitterKind, string>> = {
  spectrum: /* glsl */ `
vec3 em_spectrum(vec2 p) {
  vec4 A = EA, B = EB, C = EC;
  vec2 O = A.zw;
  int mode = int(B.x + 0.5);
  float N = B.y;
  float aa = 1.5 * px();
  float gw = C.x * 0.5;
  vec3 c = vec3(0.0);
  if (mode == 0 || mode == 3) {
    float hw = uAspect * 0.5;
    float s = mode == 0 ? (p.x + hw) / (2.0 * hw) : abs(p.x - O.x) / hw;
    float bin = (floor(s * N) + 0.5) / N;
    float gap = abs(fract(s * N) - 0.5);
    float lv = specAt(bin * 0.8 + 0.02);
    float h = 0.008 + B.w * lv * lv;
    float y = mode == 0 ? p.y - O.y : abs(p.y - O.y);
    float bar = smoothstep(gw + 0.03, gw - 0.03, gap) * smoothstep(-aa, 0.0, y) * smoothstep(h + aa, h, y);
    c = pal(bin * 0.5 + A.y) * bar * (0.4 + 0.8 * lv);
  } else {
    vec2 d = p - O;
    if (mode == 1 && d.y < 0.0) return vec3(0.0);
    float r = length(d);
    float a = atan(d.y, d.x);
    float s = mode == 1 ? abs(a / PI - 0.5) * 2.0 : abs(a / PI);
    float bin = (floor(s * N) + 0.5) / N;
    float gap = abs(fract(s * N) - 0.5);
    float lv = specAt(bin * 0.8 + 0.02);
    float r0 = B.z + 0.025 * uStem.y;
    float len = 0.015 + B.w * lv * lv;
    float bar = smoothstep(gw + 0.03, gw - 0.03, gap) * smoothstep(r0 + 0.012, r0 + 0.012 + aa, r) * smoothstep(r0 + 0.012 + len + aa, r0 + 0.012 + len, r);
    c = pal(bin * 0.5 + A.y) * bar;
    c += mix(uColA, vec3(1.0, 0.85, 0.6), 0.4) * smoothstep(r0, r0 - aa, r) * (0.12 + 0.2 * uStem.y);
  }
  return c * A.x * 0.1 * uAccum;
}`,

  stars: /* glsl */ `
vec2 starPos(vec2 cell) { return cell + 0.2 + 0.6 * hash22(cell * 1.7 + 0.3); }
bool hasStar(vec2 cell, float dens) { return hash12(cell * 1.3 + 7.1) < dens; }
vec3 em_stars(vec2 p) {
  vec4 A = EA, B = EB;
  float S = B.y;
  vec2 q = p * S + vec2(A.z, 0.0);
  vec2 cell = floor(q);
  vec3 c = vec3(0.0);
  float pw = S * px();
  float lvl = clamp(A.w * 1.5, 0.3, 1.0);
  for (int j = -1; j <= 1; j++) for (int i = -1; i <= 1; i++) {
    vec2 cc = cell + vec2(i, j);
    if (!hasStar(cc, B.x)) continue;
    vec2 sp = starPos(cc);
    float pc = floor(hash12(cc + 3.7) * 12.0);
    float e = pow(chromaAt(pc), 2.5) * lvl;
    vec3 col = keyCol(pc + A.y * 12.0, 0.3, 1.0);
    float tw = 1.0 - 0.3 * B.w + 0.3 * B.w * sin(uTime * (0.5 + hash12(cc)) + pc);
    float sd = length(q - sp);
    c += col * (0.4 * tw + 1.6 * e) * (glow(sd, pw * (1.8 + 2.5 * e)) + 0.08 * glow(sd, pw * 10.0));
    if (B.z > 0.01) {
      for (int k = 0; k < 2; k++) {
        vec2 nc = cc + (k == 0 ? vec2(1.0, 0.0) : vec2(0.0, 1.0));
        if (!hasStar(nc, B.x)) continue;
        float pc2 = floor(hash12(nc + 3.7) * 12.0);
        float link = min(e, pow(chromaAt(pc2), 2.5) * lvl);
        if (link < 0.04) continue;
        c += mix(col, keyCol(pc2 + A.y * 12.0, 0.3, 1.0), 0.5) * link * B.z * glow(sdSeg(q, sp, starPos(nc)), pw * 1.1) * 0.8;
      }
    }
  }
  return c * 0.08 * A.x * uAccum;
}`,

  ink: /* glsl */ `
vec3 em_ink(vec2 p) {
  vec3 c = vec3(0.0);
  int n = int(EA.z + 0.5);
  for (int i = 0; i < 6; i++) {
    if (i >= n) break;
    vec4 e = uInk[i];
    c += pal(float(i) * 0.25 + 0.1 + EA.y) * e.z * glow(length(p - e.xy), e.w);
  }
  return c * 0.4 * EA.x * uAccum;
}`,

  wire: /* glsl */ `
float segZ(int i) { return uSegZ[i / 4][i % 4]; }
vec3 em_wire(vec2 p) {
  vec3 c = vec3(0.0);
  float w = px() * EA.z;
  for (int i = 0; i < 48; i++) {
    if (i >= uSegN) break;
    vec4 s = uSeg[i];
    float d = sdSeg(p, s.xy, s.zw);
    float z = segZ(i);
    c += pal(z * 0.4 + 0.1 + EA.y) * z * (glow(d, w) + 0.12 * glow(d, min(w * 7.0, 0.012 * EA.z)));
  }
  // Lines are a pixel or two wide at any resolution: keep their energy resolution independent.
  return c * (0.12 + 0.08 * uLoud) * EA.x * uAccum * sqrt(clamp(uRes.y / 1080.0, 0.1, 1.0));
}`,

  plasma: /* glsl */ `
vec3 em_plasma(vec2 p) {
  vec4 A = EA, B = EB;
  float t = A.z;
  vec2 q = p * B.x;
  vec2 w1 = vec2(fbm4(q + vec2(0.0, t)), fbm4(q + vec2(5.2, -t * 0.8)));
  vec2 w2 = vec2(fbm4(q + B.y * w1 + vec2(1.7, 9.2) + A.w * 0.1), fbm4(q + B.y * w1 + vec2(8.3, 2.8)));
  float f = fbm4(q + (2.0 + 0.8 * uStem.y) * w2);
  float bands = f * B.z - A.w;
  float line = 1.0 - abs(fract(bands) - 0.5) * 2.0;
  float fw = fwidth(bands) * 2.0;
  float ln = smoothstep(1.0 - fw - 0.06, 1.0, line);
  vec3 base = pal(f * 1.2 + t * 0.15 + A.y) * pow(f, 3.0) * mix(0.35, 0.08, B.w);
  return (base + pal(f + 0.33 + A.y) * ln * B.w * (0.15 + 0.45 * uLoud) * (0.6 + 0.4 * f)) * A.x * uLayerK;
}`,

  aurora: /* glsl */ `
vec3 em_aurora(vec2 p) {
  vec4 A = EA, B = EB;
  float t = A.w;
  float y0 = B.x + 0.09 * sin(p.x * B.w + t) + 0.06 * (fbm2(vec2(p.x * 1.8 - t * 0.7, t)) - 0.5);
  float dy = p.y - y0;
  if (dy < -0.02) return vec3(0.0);
  float rays = 0.3 + 0.7 * fbm2(vec2(p.x * B.z + fbm2(vec2(p.x * 3.0, t)) * 4.0, t * 2.0));
  float body = exp(-max(dy, 0.0) * B.y) * smoothstep(-0.02, 0.005, dy);
  vec3 col = mix(pal(A.y), uColC, smoothstep(0.0, 0.3, dy));
  return col * body * rays * A.z * 0.03 * A.x * uAccum;
}`,

  blobs: /* glsl */ `
vec3 em_blobs(vec2 p) {
  float F = 0.0;
  vec2 g = vec2(0.0);
  int n = int(EA.z + 0.5);
  for (int i = 0; i < 6; i++) {
    if (i >= n) break;
    vec2 d = p - uBlob[i].xy;
    float r2 = uBlob[i].z * uBlob[i].z;
    float q = 1.0 / (dot(d, d) + 1e-4);
    F += r2 * q;
    g += -2.0 * r2 * q * q * d;
  }
  float e = fwidth(F) * 1.2;
  float m = smoothstep(1.0 - e, 1.0 + e, F);
  vec3 nn = normalize(vec3(-g * 0.05, 1.0));
  vec3 r = reflect(vec3(0.0, 0.0, -1.0), nn);
  vec3 env = mix(uColB * 0.03, pal(EA.y) * 0.7, smoothstep(-0.3, 0.9, r.y));
  env += vec3(1.0) * glow(r.y - 0.2 - 0.12 * sin(r.x * 3.0 + uPhase * 0.3), 0.06);
  env += uColC * pow(max(-r.x, 0.0), 5.0) * 0.8;
  float fres = pow(1.0 - nn.z, 2.0);
  vec3 chrome = env * (0.3 + 0.7 * fres) + vec3(0.02);
  vec3 soft = pal(EA.y + F * 0.1) * (0.25 + 0.5 * fres);
  vec3 surf = mix(soft, chrome, EA.w);
  vec3 bg = uColB * 0.003 + uColC * 0.015 * smoothstep(0.2, 1.0, F);
  return mix(bg, surf, m) * EA.x * uLayerK;
}`,

  edge: /* glsl */ `
vec2 edgeLocal(vec2 p, float side) {
  float hw = uAspect * 0.5;
  if (side < 0.5) return vec2(p.y, hw - p.x);
  if (side < 1.5) return vec2(p.x, 0.5 - p.y);
  if (side < 2.5) return vec2(p.x, p.y + 0.5);
  return vec2(p.y, p.x + hw);
}
vec3 em_edge(vec2 p) {
  vec4 A = EA, B = EB, C = EC, D = ED;
  vec2 L = edgeLocal(p, B.y);
  float u = L.x, v = L.y;
  float W = A.z;
  int mode = int(B.x + 0.5);
  vec3 c = vec3(0.0);
  if (mode == 0) {
    if (v > W) return c;
    float y = u - B.z;
    float h = D.x, id = D.y, lit = D.z, fr = D.w;
    if (fr < 0.06 || y < 0.0) return c;
    if (y < h) {
      c = uColB * 0.012 + uColC * 0.01 * (1.0 - y / max(h, 1e-3));
      float row = floor(y / 0.022);
      float inRow = step(0.35, fract(y / 0.022));
      float inCol = step(0.4, fract(fr * C.y));
      float on = step(hash12(vec2(id, row)), lit) * inRow * inCol * step(y, h - 0.014) * step(0.1, fr) * step(fr, 0.94);
      c += mix(pal(A.y), vec3(1.0, 0.8, 0.5), 0.5) * on * (0.25 + 0.5 * hash12(vec2(row, id * 3.1)));
    }
    c += uColC * glow(y - h, 0.003) * (0.5 + 1.2 * uBeatPulse * uPres.x) * step(0.08, fr);
    return c * A.x * uAccum;
  }
  if (mode == 1) {
    if (v > W * 3.0 + 0.05) return c;
    float d = sdSeg(vec2(u, v), vec2(C.w, W), vec2(C.z, 0.0));
    float w = 0.004 + 0.008 * uLoud;
    c = mix(pal(A.y), uColC, uMelody) * (glow(d, w) + 0.2 * glow(d, w * 4.0));
    c += vec3(1.0) * glow(length(vec2(u - C.z, v)), w * 0.9) * 0.6;
    return c * 0.3 * A.x * uAccum;
  }
  if (mode == 2) {
    if (v > W * 2.0 + 3.0 * px()) return c;
    float dens = 20.0 + 30.0 * C.x;
    float col = floor((u + 5.0) * dens);
    float xin = fract((u + 5.0) * dens) - 0.5;
    float m = 1.0 + floor(hash11(col * 1.37) * 2.0);
    float halfLen = (B.y < 0.5 || B.y > 2.5) ? 0.5 : uAspect * 0.5;
    float bin = abs(u) / halfLen;
    float lv = specAt(bin * 0.75 + 0.03);
    // Drops are born on an eighth-note grid; the first eighth of each bar fires a fuller curtain.
    float e8 = uBeats * 2.0;
    float win = step(fract(e8), 0.14 + 0.08 * m);
    float down = step(uBar, 0.125);
    float thr = mix(0.03, 0.45, hash12(vec2(col, floor(e8))));
    float fire = step(thr, lv * (0.35 + 0.65 * uAct) * (0.4 + C.x) * (1.0 + 0.8 * down));
    float bright = (0.6 + 1.6 * lv) * (0.7 + 0.9 * uOnset.x + 0.6 * down);
    c = pal(bin * 0.5 + 0.1 * m + A.y) * fire * win * bright * glow(xin, 0.055) * (0.5 + 0.35 * m);
    return c * 1.6 * A.x * uAccum;
  }
  if (v > W) return c;
  float y = u - B.z;
  float h = C.z * B.w;
  float aa = 1.5 * px();
  c = pal(A.y + 0.5) * 0.01 * smoothstep(h + aa, h, y) * smoothstep(-0.3, 0.0, y);
  c += mix(pal(A.y), vec3(1.0), 0.3) * glow(y - h, 0.003) * (0.4 + 0.8 * uLoud);
  return c * A.x * uAccum;
}`,

  tiles: /* glsl */ `
vec4 hexCell(vec2 p) {
  const vec2 s = vec2(1.0, 1.7320508);
  vec4 hC = floor(vec4(p, p - vec2(0.5, 1.0)) / s.xyxy) + 0.5;
  vec4 h = vec4(p - hC.xy * s, p - (hC.zw + 0.5) * s);
  return dot(h.xy, h.xy) < dot(h.zw, h.zw) ? vec4(h.xy, hC.xy) : vec4(h.zw, hC.zw + 0.5);
}
float hexD(vec2 q) { q = abs(q); return max(dot(q, vec2(0.5, 0.8660254)), q.x); }
vec4 tileCell(vec2 q, float shape) {
  if (shape < 0.5) return hexCell(q);
  if (shape < 1.5) return vec4(fract(q) - 0.5, floor(q));
  vec2 s = vec2(q.x - q.y * 0.57735, q.y * 1.1547);
  vec2 id = floor(s);
  vec2 f = fract(s);
  float up = step(f.x + f.y, 1.0);
  vec2 cc = up > 0.5 ? vec2(0.3333) : vec2(0.6667);
  vec2 d = f - cc;
  return vec4(vec2(d.x + d.y * 0.5, d.y * 0.866), id * 2.0 + vec2(up, 0.0));
}
float tileD(vec2 l, float shape) {
  if (shape < 0.5) return hexD(l);
  if (shape < 1.5) return max(abs(l.x), abs(l.y)) * 0.9;
  return length(l) * 1.5;
}
vec3 em_tiles(vec2 p) {
  vec4 A = EA, B = EB;
  vec2 q = rot2(A.w) * p * B.y;
  vec4 h = tileCell(q, B.x);
  float pc = floor(hash12(h.zw) * 12.0);
  float e = pow(chromaAt(pc), 2.0) * clamp(A.z * 1.5, 0.3, 1.0);
  float trig = step(hash12(h.zw + floor(uBeats)), B.z * (0.6 + 0.8 * uAct));
  float d = tileD(h.xy, B.x);
  float aa = fwidth(d) * 1.5;
  float fill = smoothstep(0.43, 0.43 - aa, d) * (0.5 + 0.5 * smoothstep(0.43, 0.0, d));
  vec3 c = lin(keyCol(pc + A.y * 12.0, 0.8, 1.0)) * fill * e * e * trig * (0.02 + 0.12 * uBeatPulse) * uAccum;
  float edge = smoothstep(aa, 0.0, abs(d - 0.47));
  c += mix(uColB, uColC, 0.5) * edge * 0.035 * (0.5 + uLoud) * B.w * uLayerK;
  return c * A.x;
}`,

  horizon: /* glsl */ `
vec3 em_horizon(vec2 p) {
  vec4 A = EA, B = EB;
  float hz = B.x;
  vec3 c = vec3(0.0);
  if (p.y < hz) {
    float dy = hz - p.y;
    float z = 0.35 / dy;
    float x = p.x * z;
    float gz = abs(fract(z * 0.5 + A.z) - 0.5);
    float gx = abs(fract(x * B.y) - 0.5);
    float wz = fwidth(z * 0.5) * 1.2, wx = fwidth(x * B.y) * 1.2;
    float line = max(smoothstep(wz, 0.0, gz), smoothstep(wx, 0.0, gx));
    c += pal(A.y) * line * exp(-z * 0.12) * (0.3 + 0.6 * uBeatPulse * uPres.x);
    c += pal(A.y) * 0.03 * exp(-dy * 25.0);
  } else if (B.z > 0.0) {
    float dy = p.y - hz;
    float bin = abs(p.x) / (uAspect * 0.5);
    float mh = (0.015 + 0.1 * specAt(bin * 0.6 + 0.02) * (0.4 + 0.6 * uAct)) * B.z;
    if (dy < mh) c = uColB * 0.004 + pal(A.y) * glow(dy - mh, 0.002) * 0.25;
    c += mix(uColC * 0.02, uColB * 0.002, smoothstep(0.0, 0.5, dy));
  }
  return c * A.x * uLayerK;
}`,

  orb: /* glsl */ `
vec3 em_orb(vec2 p) {
  vec4 A = EA, B = EB, C = EC;
  vec2 d = p - A.zw;
  float md = length(d);
  float R = B.x;
  vec3 moon = mix(vec3(1.0, 0.95, 0.85), pal(A.y), 0.3);
  vec3 sun = mix(uColC, uColB, smoothstep(-R, R, d.y));
  vec3 body = mix(moon, sun, B.z);
  float cut = d.y > 0.02 * R / 0.19 ? 1.0 : step(0.35 - d.y / R * 0.5, fract(d.y / R * 4.5 - uPhase * 0.2));
  cut = mix(1.0, cut, B.z);
  float tex = mix(1.0, 0.7 + 0.3 * fbm2(d * 1.8 / R), B.w);
  vec3 c = body * smoothstep(R, R - 0.0025, md) * cut * tex * 0.5;
  c += pal(A.y) * step(R, md) * B.y * (0.06 * exp(-(md - R) * 10.0) + 0.12 * C.x * exp(-(md - R) * 4.0));
  return c * A.x * uLayerK;
}`,
};

function emitterCode(kind: EmitterKind, slot: number): string {
  const src = EMIT_GLSL[kind];
  if (!src) return '';
  return src
    .replace(/\bEA\b/g, `uEm[${slot * 4}]`)
    .replace(/\bEB\b/g, `uEm[${slot * 4 + 1}]`)
    .replace(/\bEC\b/g, `uEm[${slot * 4 + 2}]`)
    .replace(/\bED\b/g, `uEm[${slot * 4 + 3}]`);
}

/** Emitters drawn as fullscreen fields (the rest are geometry passes). */
export function isFieldEmitter(kind: EmitterKind): boolean {
  return kind in EMIT_GLSL;
}

// ------------------------------------------------------------- builders

export interface Sources {
  feedback: string;
  composite: string;
}

export function buildSources(g: Genome): Sources {
  const defs: string[] = [];
  if (g.carrier.kind === 'none') defs.push('NO_FEEDBACK');
  if (g.carrier.kind === 'fluid') defs.push('USE_FLUID');
  if (g.carrier.kind === 'flow') defs.push('USE_FLOW');
  if (g.color.p.reflect > 0.5) defs.push('REFLECT');
  if (g.color.p.tonemap > 0.5) defs.push('LOG_TONE');
  const pre = HEAD + defs.map((d) => `#define ${d}\n`).join('') + COMMON + LIB + FLAME_VARIATION_GLSL;

  const warpOps = g.chain.map((o, i) => (o.stage === 'warp' ? opCode(o, i) : '')).join('');
  const viewOps = g.chain.map((o, i) => (o.stage === 'view' ? opCode(o, i) : '')).join('');

  const fbEm: string[] = [];
  const topEm: string[] = [];
  let fbCode = '';
  let topCode = '';
  g.emitters.forEach((e, slot) => {
    if (!isFieldEmitter(e.kind)) return;
    if (e.layer === 'fb') {
      fbCode += emitterCode(e.kind, slot) + '\n';
      fbEm.push(`em_${e.kind}(p)`);
    } else {
      topCode += emitterCode(e.kind, slot) + '\n';
      topEm.push(`em_${e.kind}(q)`);
    }
  });

  const feedback = pre + /* glsl */ `
in vec2 vUv;
uniform sampler2D uPrev, uVel;
uniform vec2 uSimTexel;
uniform float uFluidAmt, uBlur, uDecaySub, uFlowAmt, uFlowScale;
out vec4 o;
vec3 prevAt(vec2 uv) {
  vec3 c = texture(uPrev, uv).rgb;
  if (uBlur > 0.0) {
    vec2 q = 0.9 / uRes;
    vec3 b = texture(uPrev, uv + q).rgb + texture(uPrev, uv - q).rgb +
             texture(uPrev, uv + vec2(q.x, -q.y)).rgb + texture(uPrev, uv + vec2(-q.x, q.y)).rgb;
    c = mix(c, b * 0.25, uBlur);
  }
  vec2 e = min(uv, 1.0 - uv);
  c *= smoothstep(0.0, 0.002, min(e.x, e.y));
  return c;
}
vec2 warp(vec2 p) {
${warpOps}  return p;
}
${fbCode}
void main() {
  vec2 asp = vec2(uAspect, 1.0);
  vec2 p = (vUv - 0.5) * asp;
  vec3 c = vec3(0.0);
#ifndef NO_FEEDBACK
  vec2 w = warp(p);
#ifdef USE_FLOW
  w += curlNoise(p * uFlowScale + 3.0, uPhase * 0.3) * uFlowAmt;
#endif
  vec2 suv = w / asp + 0.5;
#ifdef USE_FLUID
  suv -= texture(uVel, vUv).xy * uSimTexel * uDt * uFluidAmt;
#endif
  c = max(prevAt(suv) * uDecay - uDecaySub, 0.0);
#endif
  c += ${fbEm.length ? fbEm.join(' + ') : 'vec3(0.0)'};
  o = vec4(clamp(c, vec3(0.0), vec3(64.0)), 1.0);
}`;

  const composite = pre + /* glsl */ `
in vec2 vUv;
uniform sampler2D uFb;
uniform float uWeight, uSat, uSweep, uReflectY;
out vec4 o;
vec3 fb(vec2 q) { return texture(uFb, q / vec2(uAspect, 1.0) + 0.5).rgb; }
vec2 view(vec2 p) {
${viewOps}  return p;
}
${topCode}
void main() {
  vec2 p = (vUv - 0.5) * vec2(uAspect, 1.0);
  float att = 1.0;
#ifdef REFLECT
  if (p.y < uReflectY) {
    float d = uReflectY - p.y;
    p = vec2(p.x + (0.002 + d * 0.02) * sin(d * 260.0 - uPhase * 3.0), uReflectY + d);
    att = 0.3 * exp(-d * 4.0);
  }
#endif
  vec2 q = view(p);
#ifdef LOG_TONE
  vec2 ox = vec2(0.5 / uRes.y, 0.0), oy = vec2(0.0, 0.5 / uRes.y);
  vec3 c = (fb(q + ox + oy) + fb(q - ox - oy) + fb(q + ox - oy) + fb(q - ox + oy)) * 0.25;
  c += ${topEm.length ? topEm.join(' + ') : 'vec3(0.0)'};
  float l = max(c.r, max(c.g, c.b));
  float b = log(1.0 + l * 24.0) / log(25.0);
  c = c / max(l, 1e-5) * pow(b, 1.1) * 0.55;
#else
  vec3 c = fb(q) + ${topEm.length ? topEm.join(' + ') : 'vec3(0.0)'};
#endif
  c *= att;
#ifdef REFLECT
  c += uColC * glow(p.y - uReflectY, 0.0012) * 0.06 * step(0.99, att);
#endif
  if (uSweep > 0.001) {
    float front = (1.0 - uSweep) * 1.2;
    float sw = uSweep * exp(-pow((length(p) - front) * 5.0, 2.0));
    c = max(hueRotate(c, sw * 2.4), 0.0) * (1.0 + sw * 0.8);
  }
  c = max(c, 0.0);
  float l2 = luma(c);
  c = max(mix(vec3(l2), c, uSat), 0.0);
  o = vec4(c * uWeight, 1.0);
}`;
  return { feedback, composite };
}

// ------------------------------------------------------ line geometry

export const WAVE_VS = HEAD + COMMON + LIB + /* glsl */ `
uniform float uN, uThick, uBright;
uniform vec4 uW[3]; // (shape, amp, x, y) (radius, turns, ra, rb) (phase, rotation, hue, -)
out float vSide;
out vec3 vCol;
vec2 curve(float k) {
  vec4 A = uW[0], B = uW[1], C = uW[2];
  int sh = int(A.x + 0.5);
  vec2 q;
  if (sh == 0) {
    float env = smoothstep(0.0, 0.12, k) * smoothstep(1.0, 0.88, k);
    q = vec2((k * 2.0 - 1.0) * uAspect * 0.42, waveAt(k) * A.y * env);
  } else if (sh == 1) {
    float a = k * TAU;
    q = (B.x + waveAt(abs(k * 2.0 - 1.0)) * A.y * 0.35) * vec2(cos(a), sin(a));
  } else if (sh == 2) {
    float a = k * TAU * B.y;
    float r = B.x * (0.12 + 0.88 * k) + waveAt(k) * A.y * 0.2 * k;
    q = r * vec2(cos(a), sin(a));
  } else {
    float t = k * 42.0;
    float damp = exp(-k * 2.0);
    float x = sin(B.z * t + C.x) + 0.5 * sin(B.z * 2.0 * t + 0.01 + C.x * 1.3);
    float y = sin(B.w * t + C.x * 0.7 + 1.3) + 0.5 * sin(B.w * 1.5 * t - 0.01);
    q = vec2(x, y) * damp * B.x * 0.62 * (1.0 + waveAt(k) * A.y * 0.4);
  }
  return rot2(C.y) * q + A.zw;
}
vec3 curveColor(float k) {
  int sh = int(uW[0].x + 0.5);
  float env = sh == 0 ? smoothstep(0.0, 0.15, k) * smoothstep(1.0, 0.85, k) : 1.0;
  float damp = sh == 3 ? 0.5 + 0.5 * exp(-k * 1.5) : 1.0;
  vec3 c = sh == 0 ? mix(pal(uW[2].z), vec3(1.0), 0.2) : mix(pal(uW[2].z), pal(uW[2].z + 0.66), k);
  return c * env * damp;
}
void main() {
  int i = gl_VertexID;
  float k = float(i / 2) / (uN - 1.0);
  float side = (i % 2 == 0) ? -1.0 : 1.0;
  float dk = 1.0 / uN;
  vec2 a = curve(k);
  vec2 t = curve(min(k + dk, 1.0)) - curve(max(k - dk, 0.0));
  vec2 n = normalize(vec2(-t.y, t.x) + 1e-6);
  vec2 pp = a + n * side * uThick / uRes.y;
  vSide = side;
  vCol = curveColor(k) * uBright;
  gl_Position = vec4(pp.x / (uAspect * 0.5), pp.y / 0.5, 0.0, 1.0);
}`;

export const WAVE_FS = HEAD + /* glsl */ `
in float vSide;
in vec3 vCol;
out vec4 o;
void main() {
  float s = 1.0 - vSide * vSide;
  o = vec4(vCol * (s * s + pow(s, 10.0) * 0.4), 1.0);
}`;

