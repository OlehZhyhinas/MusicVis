// GLSL for V2: the shared library, the chain op snippets, and the builders
// that assemble each body's shader from its sub-genes (shape distance field,
// placement fold or copy loop, deformation, material, emission). Numeric
// genome parameters are uniforms (packed per frame by engine.ts); only
// structure changes the source, so compiled programs are cached by
// structuralKey().

import { RELIEF_GLSL } from './genes/relief';
import { FLAME_VARIATION_GLSL } from './variations';
import { SUPERSCOPE_GLSL } from './genes/superscope';
import { CELLS_GLSL } from './genes/cells';
import { BEAMS_GLSL } from './genes/beams';
import { WATER_GLSL } from './genes/water';
import { BLEND_GLSL, blendCall } from './genes/blend';
import { CYMATICS_GLSL } from './genes/cymatics';
import {
  SHAPE_CLASS, STATIC_MATERIALS, bodyLayer, isFoldPlace, sdfCapable,
  type BodyGene, type Genome, type OpGene, type ShapeKind,
} from './genome';
import { SCENE_PASS, sceneField } from './genes/raymarch';

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

/** Per-body uniform vec4 slots (see engine.ts packBody for the layout). */
export const BODY_VEC4 = 21;
/** Explicit copies per body (uCp / uCq entries). */
export const COPY_SLOTS = 6;

function lib(nb: number): string {
  return /* glsl */ `
uniform vec2 uRes;
uniform float uAspect, uTime, uPhase, uDt, uF60, uSpeed;
uniform float uBeat, uBar, uBars, uBeats, uBeatPulse, uBarPulse;
uniform float uSpin, uSpinStep;
uniform vec4 uStem, uOnset, uPres;
uniform float uAct, uBuild, uDrop, uLoud, uMelody, uKeyHue;
uniform vec3 uColA, uColB, uColC;
uniform float uDecay;
uniform float uLayerK;   // 1 - decay in the feedback pass, 1 in the composite: steady materials look the same on either layer
uniform float uAccum;    // 1 in the feedback pass, ~6 in the composite: accumulating materials look the same on either layer
uniform vec4 uOpA[6], uOpB[6];
uniform vec4 uBd[${nb * BODY_VEC4}];   // per-body parameters
uniform vec4 uCp[${nb * COPY_SLOTS}];  // copies: (x, y, angle, scale)
uniform vec4 uCq[${nb * COPY_SLOTS}];  // copies: (level, hue offset, previous x, previous y)
uniform vec4 uWv[${nb * 4}];           // curve parameters per body (see WAVE_VS)
uniform vec4 uDrA[${nb * 3}], uDrB[${nb * 3}]; // draw-space ops per body: A = amounts, B.x = op type id
uniform vec4 uSeg[48];
uniform vec4 uSegZ[12];
uniform int uSegN;
uniform vec4 uChroma4[3];
uniform sampler2D uWave, uSpec;

float waveAt(float x) { return texture(uWave, vec2(x, 0.5)).r; }
float specAt(float x) { return texture(uSpec, vec2(clamp(x, 0.0, 1.0), 0.5)).r; }
float chromaAt(float i) { int k = int(mod(i, 12.0)); return uChroma4[k / 4][k % 4]; }
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
float segZ(int i) { return uSegZ[i / 4][i % 4]; }
float smin(float a, float b, float k) { float h = clamp(0.5 + 0.5 * (b - a) / k, 0.0, 1.0); return mix(b, a, h) - k * h * (1.0 - h); }
// View-stage ops may scale the sampled picture and add light (stretch); ignored in the feedback pass.
float vMul = 1.0;
vec3 vAdd = vec3(0.0);
// Grid cells pass their twinkle to the materials.
float gTw = 1.0;
// Metaball gradient for the chrome material (valid when gMetaOn > 0.5).
vec2 gMeta = vec2(0.0);
float gMetaOn = 0.0;
float gMetaF = 1.0;
`;
}

// ------------------------------------------------------------- chain ops
// Each snippet transforms `p` in place; OA / OB are the op's uniform slots.

const OP_GLSL: Record<string, string> = {
  zoom: `{ vec2 d = p - OA.xy; float r = length(d); float z = 1.0 + OA.z * mix(1.0, 0.5 + r * 1.2, OA.w); p = OA.xy + d / z; }`,
  rotate: `{ p = OA.xy + rot2(OA.z) * (p - OA.xy); }`,
  translate: `{ vec2 sh = OA.xy; if (OA.z > 1.0) { sh.y *= 1.0 + floor(hash11(floor((p.x + 5.0) * OA.z) * 1.37) * 2.0); sh.x *= 1.0 + floor(hash11(floor((p.y + 5.0) * OA.z) * 1.37) * 2.0); } p += sh; }`,
  swirl: `{ vec2 d = p - OA.xy; float r = length(d); p = OA.xy + rot2(OA.z / (r * OA.w + 0.4)) * d; }`,
  twist: `{ vec2 d = p - OA.xy; p = OA.xy + rot2(OA.z * length(d) * 8.0) * d; }`,
  ripple: `{ if (OA.w > 0.5) { float r = length(p); p += normalize(p + 1e-5) * OA.x * sin(r * OA.y - OA.z); } else { p.y += OA.x * sin(p.x * OA.y - OA.z); } }`,
  noise: `{ p += curlNoise(p * OA.y + 4.0, OA.z) * OA.x; }`,
  quad: `{ vec2 d = p - OA.xy; p += rot2(OA.w) * vec2(d.x * d.x - d.y * d.y, 2.0 * d.x * d.y) * OA.z; }`,
  push: `{ if (OA.y < 0.5) p.x -= sign(p.x) * OA.x * smoothstep(0.0, 0.03, abs(p.x)); else if (OA.y < 1.5) p.y -= sign(p.y) * OA.x * smoothstep(0.0, 0.03, abs(p.y)); else { float r = length(p); p -= p / max(r, 1e-4) * OA.x * smoothstep(0.0, 0.03, r); } }`,
  stretch: `{ float hw = uAspect * 0.5; float bin = clamp((p.x + hw) / (2.0 * hw), 0.0, 1.0); bin = (floor(bin * OA.w) + 0.5) / OA.w; float lv = specAt(bin * 0.8 + 0.03); float st = 1.0 + OA.y * lv * (0.4 + 0.6 * uAct) + OA.z * uBeatPulse * uPres.x; float above = step(OA.x, p.y); vAdd += OB.y * above * mix(uColC * (0.025 + 0.05 * uStem.y * uPres.y), uColB * 0.002, smoothstep(OA.x, 0.5, p.y)) * vMul; vMul *= 1.0 + OB.x * (1.4 * lv - 0.3); p.y = OA.x + (p.y - OA.x) / st; }`,
  mirror: `{ if (OA.x < 0.5) p.x = abs(p.x); else if (OA.x < 1.5) p.y = abs(p.y); else p = abs(p); }`,
  tile: `{ vec2 h = vec2(uAspect, 1.0) * 0.5; vec2 q = mod(p * OA.x + h, 4.0 * h); p = abs(q - 2.0 * h) - h; }`,
  polar: `{ float a = mod(atan(p.y, p.x) + OA.y + PI, TAU) - PI; p = vec2(a / PI * uAspect * 0.5, length(p) * 2.0 * OA.x - 0.5); }`,
  kaleido: `{ float seg = TAU / OA.x; float a = mod(atan(p.y, p.x) + OA.y, seg); a = abs(a - seg * 0.5); p = length(p) * vec2(cos(a), sin(a)); }`,
};

function opCode(o: OpGene, i: number): string {
  const src = o.op.startsWith('v_') ? `{ p = mix(p, V_${o.op.slice(2)}(p * OA.y) / OA.y, OA.x); }` : OP_GLSL[o.op];
  return '  ' + src.replace(/OA/g, `uOpA[${i}]`).replace(/OB/g, `uOpB[${i}]`) + '\n';
}

// ------------------------------------------------------- draw-space ops
// A uniform-driven loop: a body's deform ops change only uniforms. Applied in
// scene space before the placement, so they bend the whole arrangement.

const DRAW_GLSL = /* glsl */ `
vec2 varById(int i, vec2 p) {
  if (i == 1) return V_sinusoidal(p);
  if (i == 2) return V_spherical(p);
  if (i == 3) return V_swirl(p);
  if (i == 4) return V_horseshoe(p);
  if (i == 5) return V_polar(p);
  if (i == 6) return V_handkerchief(p);
  if (i == 7) return V_heart(p);
  if (i == 8) return V_disc(p);
  if (i == 9) return V_spiral(p);
  if (i == 10) return V_hyperbolic(p);
  if (i == 11) return V_julia(p);
  return p;
}
vec2 drawOp(int t, vec4 A, vec2 p) {
  if (t == 1) { vec2 d = p - A.xy; float r = length(d); return A.xy + rot2(A.z / (r * A.w + 0.4)) * d; }
  if (t == 2) { vec2 d = p - A.xy; return A.xy + rot2(A.z * length(d) * 8.0) * d; }
  if (t == 3) {
    if (A.w > 0.5) { float r = length(p); return p + normalize(p + 1e-5) * A.x * sin(r * A.y - A.z); }
    return vec2(p.x, p.y + A.x * sin(p.x * A.y - A.z));
  }
  if (t == 4) return p + curlNoise(p * A.y + 4.0, A.z) * A.x;
  if (t == 5) return A.xy + rot2(A.z) * (p - A.xy);
  if (t == 6) { vec2 d = p - A.xy; float r = length(d); return A.xy + d / max(0.2, mix(A.z, 1.0 + (A.z - 1.0) * (0.5 + r * 1.2), A.w)); }
  if (t == 7) { if (A.x < 0.5) return vec2(abs(p.x), p.y); if (A.x < 1.5) return vec2(p.x, abs(p.y)); return abs(p); }
  if (t == 8) { float seg = TAU / A.x; float a = mod(atan(p.y, p.x) + A.y, seg); a = abs(a - seg * 0.5); return length(p) * vec2(cos(a), sin(a)); }
  if (t >= 20) return mix(p, varById(t - 20, p * A.y) / A.y, A.x);
  return p;
}
vec2 drawWarp(vec2 p, int base, int n) {
  for (int i = 0; i < min(n, 3); i++) {
    p = drawOp(int(uDrB[base + i].x + 0.5), uDrA[base + i], p);
  }
  return p;
}
// Line geometry is moved forward, so only the continuous ops apply (reversed).
vec2 drawWarpFwd(vec2 p, int base, int n) {
  for (int i = 0; i < min(n, 3); i++) {
    int t = int(uDrB[base + i].x + 0.5);
    if (t >= 7) continue;
    vec4 A = uDrA[base + i];
    if (t == 6) A.z = 1.0 / max(A.z, 0.2);
    else if (t == 3 || t == 4) A.x = -A.x;
    else A.z = -A.z;
    p = drawOp(t, A, p);
  }
  return p;
}
vec4 hexCell(vec2 p) {
  const vec2 s = vec2(1.0, 1.7320508);
  vec4 hC = floor(vec4(p, p - vec2(0.5, 1.0)) / s.xyxy) + 0.5;
  vec4 h = vec4(p - hC.xy * s, p - (hC.zw + 0.5) * s);
  return dot(h.xy, h.xy) < dot(h.zw, h.zw) ? vec4(h.xy, hC.xy) : vec4(h.zw, hC.zw + 0.5);
}
vec4 triCell(vec2 q) {
  vec2 s = vec2(q.x - q.y * 0.57735, q.y * 1.1547);
  vec2 id = floor(s);
  vec2 f = fract(s);
  float up = step(f.x + f.y, 1.0);
  vec2 cc = up > 0.5 ? vec2(0.3333) : vec2(0.6667);
  vec2 d = f - cc;
  return vec4(vec2(d.x + d.y * 0.5, d.y * 0.866), id * 2.0 + vec2(up, 0.0));
}
float sdPoly(vec2 p, float r, float n) {
  float an = PI / n;
  float off = (mod(n, 2.0) > 0.5 || abs(n - 6.0) < 0.5) ? an : 0.0;
  float bn = mod(atan(p.x, p.y) + off, 2.0 * an) - an;
  p = length(p) * vec2(cos(bn), abs(sin(bn)));
  vec2 acs = vec2(cos(an), sin(an));
  p -= r * acs;
  p.y += clamp(-p.y, 0.0, r * acs.y);
  return length(p) * sign(p.x);
}
float sdStar(vec2 p, float r, float n, float m) {
  float an = PI / n;
  float en = PI / m;
  vec2 acs = vec2(cos(an), sin(an));
  vec2 ecs = vec2(cos(en), sin(en));
  float bn = mod(atan(p.x, p.y), 2.0 * an) - an;
  p = length(p) * vec2(cos(bn), abs(sin(bn)));
  p -= r * acs;
  p += ecs * clamp(-dot(p, ecs), 0.0, r * acs.y / ecs.y);
  return length(p) * sign(p.x);
}
`;

// --------------------------------------------------------------- shapes
// vec3 SHP(vec2 q): (signed distance in local units, colour shade, brightness
// shade). SA / SB are the shape's two parameter vec4s; WV the curve slots.

const SHAPE_SDF: Partial<Record<ShapeKind, string>> = {
  dot: `vec3 SHP(vec2 q) { return vec3(length(q) - SA.x, 0.0, 1.0); }`,
  polygon: `vec3 SHP(vec2 q) { float r = SA.y; float rd = SA.z * 0.35 * r; return vec3(sdPoly(q, r - rd, SA.x) - rd, 0.0, 1.0); }`,
  star: `vec3 SHP(vec2 q) { return vec3(sdStar(q, SA.y, SA.x, 2.0 + (SA.x - 2.0) * SA.z), 0.0, 1.0); }`,
  segment: `vec3 SHP(vec2 q) { return vec3(sdSeg(q, vec2(-SA.x * 0.5, 0.0), vec2(SA.x * 0.5, 0.0)) - SA.y, 0.0, 1.0); }`,
  solid: `vec3 SHP(vec2 q) {
  float d = 1e3, z = 0.0;
  // Dynamic bounds keep the compiler from unrolling (and predicating) every iteration.
  for (int i = 0; i < min(uSegN, 48); i++) {
    vec4 s = uSeg[i];
    float di = sdSeg(q, s.xy, s.zw);
    if (di < d) { d = di; z = segZ(i); }
  }
  return vec3(d, z * 0.4 + 0.1, z);
}`,
  // Spectrum bars with gaps: a box per bin (SA = mode, bins, radius, len; SB = fill, bass swell).
  bars: `vec3 SHP(vec2 q) {
  int mode = int(SA.x + 0.5);
  float N = SA.y;
  if (mode == 0 || mode == 3) {
    float hw = uAspect * 0.5;
    float s = mode == 0 ? (q.x + hw) / (2.0 * hw) : abs(q.x) / hw;
    float fb = s * N;
    float bin = (floor(fb) + 0.5) / N;
    float lv = specAt(bin * 0.8 + 0.02);
    float h = 0.008 + SA.w * lv * lv;
    float y = mode == 0 ? q.y : abs(q.y);
    float cw = (mode == 0 ? 2.0 * hw : hw) / N;
    vec2 dd = vec2(abs(fract(fb) - 0.5) * cw - SB.x * 0.5 * cw, abs(y - h * 0.5) - h * 0.5);
    float d = length(max(dd, 0.0)) + min(max(dd.x, dd.y), 0.0);
    d = max(d, abs(q.x) - hw);
    return vec3(d, bin * 0.5, 0.4 + 0.8 * lv);
  }
  float r = length(q);
  float a = atan(q.y, q.x);
  float s = mode == 1 ? abs(a / PI - 0.5) * 2.0 : abs(a / PI);
  float dsda = mode == 1 ? 2.0 / PI : 1.0 / PI;
  float fb = s * N;
  float bin = (floor(fb) + 0.5) / N;
  float lv = specAt(bin * 0.8 + 0.02);
  float r0 = SA.z + SB.y + 0.012;
  float len = 0.015 + SA.w * lv * lv;
  float tang = abs(fract(fb) - 0.5) / N / dsda * r;
  vec2 dd = vec2(tang - SB.x * 0.5 / N / dsda * r, abs(r - r0 - len * 0.5) - len * 0.5);
  float d = length(max(dd, 0.0)) + min(max(dd.x, dd.y), 0.0);
  if (mode == 1) d = max(d, -q.y);
  vec3 o = vec3(d, bin * 0.5, 0.4 + 0.8 * lv);
  float dh = r - (r0 - 0.012);
  if (dh < o.x) o = vec3(dh, 0.0, 0.25 + 0.4 * uStem.y);
  return o;
}`,
  // Waveform curves with a distance field (line, circle, spiral, arc); WV = the body's curve slots.
  curve: `vec3 SHP(vec2 q) {
  vec4 A = WV0, B = WV1, C = WV2;
  q = rot2(-C.y) * q;
  int sh = int(A.x + 0.5);
  float w = 0.004;
  if (sh == 0) {
    float hw = uAspect * 0.42;
    float k = clamp(q.x / (2.0 * hw) + 0.5, 0.0, 1.0);
    float env = smoothstep(0.0, 0.12, k) * smoothstep(1.0, 0.88, k);
    float d = abs(q.y - waveAt(k) * A.y * env) * 0.7;
    return vec3(max(d, abs(q.x) - hw) - w, k, 1.0);
  }
  float r = length(q);
  float a = atan(q.y, q.x);
  if (sh == 2) {
    float best = 1e3;
    for (int n = 0; n < 7; n++) {
      float k = (fract(a / TAU) + float(n)) / max(B.y, 1.0);
      if (k > 1.0) break;
      float rk = B.x * (0.12 + 0.88 * k) + waveAt(k) * A.y * 0.2 * k;
      best = min(best, abs(r - rk));
    }
    return vec3(best * 0.8 - w, 0.3, 1.0);
  }
  if (sh == 4) {
    float span = max(B.y * 0.37, 1e-3);
    float k = a / span + 0.5;
    float kc = clamp(k, 0.0, 1.0);
    float rk = B.x + waveAt(kc) * A.y * 0.4;
    float ae = (kc - 0.5) * span;
    return vec3((k == kc ? abs(r - rk) * 0.8 : length(q - rk * vec2(cos(ae), sin(ae)))) - w, kc * 0.66, 1.0);
  }
  float k = fract(a / TAU);
  return vec3(abs(r - (B.x + waveAt(abs(k * 2.0 - 1.0)) * A.y * 0.35)) * 0.8 - w, k * 0.66, 1.0);
}`,
  // A ribbon band (fused shapes only): SA = (-, fall, rays, wav), SB = (time, level).
  aurora: `vec3 SHP(vec2 q) {
  float t = SB.x;
  float y0 = 0.09 * sin(q.x * SA.w + t) + 0.06 * (fbm2(vec2(q.x * 1.8 - t * 0.7, t)) - 0.5);
  float hw = 0.02 + 0.03 * clamp(SB.y, 0.0, 1.5);
  return vec3((abs(q.y - y0 - hw) - hw) * 0.8, 0.2, 1.0);
}`,
};

// ---------------------------------------------------------------- fields
// Chunk shapes with their own look: vec3 FLD(vec2 p) in the body's local frame.
// EA..ED are the body's four field slots (EA = gain, hue, ...), packed as before.

const FIELD_GLSL: Partial<Record<ShapeKind, string>> = {
  plasma: /* glsl */ `
vec3 FLD(vec2 p) {
  vec4 A = EA, B = EB, C = EC;
  float t = A.z;
  vec2 q = p * B.x;
  vec2 w1 = vec2(fbm4(q + vec2(0.0, t)), fbm4(q + vec2(5.2, -t * 0.8)));
  vec2 w2 = vec2(fbm4(q + B.y * w1 + vec2(1.7, 9.2) + C.x), fbm4(q + B.y * w1 + vec2(8.3, 2.8)));
  float f = fbm4(q + (2.0 + 0.8 * C.y) * w2);
  float bands = f * B.z - A.w;
  float line = 1.0 - abs(fract(bands) - 0.5) * 2.0;
  float fw = fwidth(bands) * 2.0;
  float ln = smoothstep(1.0 - fw - 0.06 - 0.06 * C.z, 1.0, line);
  vec3 base = pal(f * 1.2 + t * 0.15 + A.y + C.w) * pow(f, 3.0) * mix(0.35, 0.08, B.w);
  return (base + pal(f + 0.33 + A.y + C.w) * ln * B.w * (0.15 + 0.45 * uLoud) * ED.x * (0.6 + 0.4 * f)) * A.x * uLayerK;
}`,
  aurora: /* glsl */ `
vec3 FLD(vec2 p) {
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
  edge: /* glsl */ `
vec2 edgeLocal(vec2 p, float side) {
  float hw = uAspect * 0.5;
  if (side < 0.5) return vec2(p.y, hw - p.x);
  if (side < 1.5) return vec2(p.x, 0.5 - p.y);
  if (side < 2.5) return vec2(p.x, p.y + 0.5);
  return vec2(p.y, p.x + hw);
}
vec3 FLD(vec2 p) {
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
  beams: BEAMS_GLSL,
  cells: CELLS_GLSL,
  cymatics: CYMATICS_GLSL,
  terrain: /* glsl */ `
float hzTerrain(vec2 w, float scroll, float amt) {
  float ax = abs(w.x);
  float z = w.y + scroll * 2.0;
  float side = smoothstep(0.35, 1.6, ax);
  float lv = specAt(clamp(ax / 5.0, 0.0, 1.0) * 0.7 + 0.03);
  float ridge = 0.6 + 0.4 * sin(z * 1.3 + ax * 0.9);
  float h = side * (0.03 + 0.34 * lv * (0.4 + 0.6 * uAct)) * ridge;
  float k = fract(z * 0.25 + 0.5);
  h += uBeatPulse * uPres.x * 0.05 * exp(-pow((k - 0.5) * 9.0, 2.0)) * (0.4 + side);
  h *= smoothstep(18.0, 6.0, w.y);
  return min(h, 0.32) * amt;
}
vec3 FLD(vec2 p) {
  vec4 A = EA, B = EB, C = EC;
  float hz = B.x;
  vec3 c = vec3(0.0);
  if (p.y < hz) {
    float dy = hz - p.y;
    float zFlat = 0.35 / dy;
    float z = zFlat, h = 0.0;
    if (C.x > 0.001) {
      float z0 = 0.2, st = 0.08;
      for (int k = 0; k < 40; k++) {
        float zt = z0 + st;
        if (zt >= zFlat) break;
        if (0.35 - dy * zt <= hzTerrain(vec2(p.x * zt, zt), A.z, C.x)) { z = zt; break; }
        z0 = zt;
        st *= 1.12;
      }
      for (int k = 0; k < 5; k++) {
        float zm = 0.5 * (z0 + z);
        if (0.35 - dy * zm <= hzTerrain(vec2(p.x * zm, zm), A.z, C.x)) z = zm; else z0 = zm;
      }
      h = hzTerrain(vec2(p.x * z, z), A.z, C.x);
    }
    float x = p.x * z;
    float gz = abs(fract(z * 0.5 + A.z) - 0.5);
    float gx = abs(fract(x * B.y) - 0.5);
    float wz = fwidth(z * 0.5) * 1.2, wx = fwidth(x * B.y) * 1.2;
    float line = max(smoothstep(wz, 0.0, gz), smoothstep(wx, 0.0, gx));
    vec3 lc = mix(pal(A.y), uColC, smoothstep(0.02, 0.22, h));
    lc = mix(lc, vec3(1.0), 0.35 * uBarPulse * C.y);
    c += lc * line * exp(-z * 0.12) * (0.3 + 0.6 * uBeatPulse * uPres.x + 0.8 * h);
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
};

// ------------------------------------------------------------- deform
// vec2 DFM(vec2 q, out float k): bends the shape in its local unit frame;
// k corrects the distance field for the stretch. D = BD(7).

const DEFORM_GLSL: Record<string, string> = {
  none: `vec2 DFM(vec2 q, out float k) { k = 1.0; return q; }`,
  // D = (count, reach, curl now, width); BD(8..9) = angles, BD(10..11) = lengths.
  arms: `float armA(int j) { return j < 4 ? BD(8)[j] : BD(9)[j - 4]; }
float armL(int j) { return j < 4 ? BD(10)[j] : BD(11)[j - 4]; }
vec2 DFM(vec2 q, out float k) {
  vec4 D = BD(7);
  float R = max(BD(4).w, 1e-3);
  float md = length(q);
  float a = atan(q.y, q.x);
  float rel = max(md / R - 1.0, 0.0);
  float ext = 0.0;
  int n = min(int(D.x + 0.5), 7);
  for (int j = 0; j < n; j++) {
    float ang = armA(j) + D.z * rel;
    float da = atan(sin(a - ang), cos(a - ang));
    float w = D.w / (1.0 + 1.9 * rel);
    ext += armL(j) * exp(-da * da / (w * w));
  }
  float s = 1.0 + ext * D.y;
  k = s * 0.6;
  return q / s;
}`,
  // D = (lobes, amp now, phase, -)
  wobble: `vec2 DFM(vec2 q, out float k) {
  vec4 D = BD(7);
  float s = 1.0 + D.y * sin(D.x * atan(q.y, q.x) + D.z);
  k = s * 0.8;
  return q / s;
}`,
  // D = (amp now, scale, phase, -)
  noise: `vec2 DFM(vec2 q, out float k) {
  vec4 D = BD(7);
  k = 0.8;
  return q + curlNoise(q * D.y + 4.0, D.z) * D.x;
}`,
  // D = (amount now, -, -, -)
  twist: `vec2 DFM(vec2 q, out float k) {
  k = 0.9;
  return rot2(BD(7).x * length(q)) * q;
}`,
};

// ------------------------------------------------------------ materials
// vec4 MAT(vec3 s, vec2 q, vec4 Q, float R, vec2 p, out vec3 ex): paint colour
// and coverage for one copy (s = distance in scene units, colour shade,
// brightness shade; q = local position; Q = copy level, hue; R = copy size),
// ex = extra light around it. M0 = BD(0) = (gain, hue, a, b), M1 = BD(1).

// Colour mapping: BD(0).y = the body's base hue, BD(20) = (mapping kind, detail, height amount, instrument
// amount). Q.y < -0.5 marks a pitch class (keyed colours around the song's key).
const BODY_COL = `vec3 COL(vec4 Q, float cs, float sat, vec2 p) {
  if (Q.y < -0.5) { vec3 k = keyCol(-Q.y - 1.0 + BD(0).y * 12.0, sat, 1.0); return sat > 0.7 ? lin(k) : k; }
  return pal(BD(0).y + Q.y + cs * BD(20).y + p.y * BD(20).z);
}
float CHUE(vec2 cc, float pc) {
  int k = int(BD(20).x + 0.5);
  if (k == 2) return -1.0 - pc;
  if (k == 1) return (floor(hash12(cc + 5.1) * 4.0) * 0.25 + 0.1) * BD(20).w;
  return 0.0;
}`;

const MATERIAL_GLSL: Record<string, string> = {
  // M0 = (gain, hue, width px, halo)
  line: `vec4 MAT(vec3 s, vec2 q, vec4 Q, float R, vec2 p, out vec3 ex) {
  ex = vec3(0.0);
  float w = px() * BD(0).z;
  float cov = glow(s.x, w) + BD(0).w * glow(s.x, min(w * 7.0, 0.012 * BD(0).z));
  vec3 col = COL(Q, s.y, 0.3, p) * s.z * Q.x * (0.12 + 0.08 * uLoud) * sqrt(clamp(uRes.y / 1080.0, 0.1, 1.0));
  return vec4(col, cov);
}`,
  // M0 = (gain, hue, soft, halo), M1 = (outline, core, clip, halo level)
  fill: `vec4 MAT(vec3 s, vec2 q, vec4 Q, float R, vec2 p, out vec3 ex) {
  float aa = px() * 1.2;
  float cov = smoothstep(aa, -max(aa, BD(0).z * R), s.x);
  cov *= 1.0 - 0.5 * BD(1).y + 0.5 * BD(1).y * smoothstep(0.0, -R, s.x);
  vec3 col = COL(Q, s.y, 0.8, p) * s.z * Q.x * 0.5;
  ex = COL(Q, 0.0, 0.8, p) * step(0.0, s.x) * BD(0).w * (0.06 * exp(-s.x * 10.0) + 0.12 * BD(1).w * exp(-s.x * 4.0));
  ex += mix(uColB, uColC, 0.5) * smoothstep(aa, 0.0, abs(s.x - 0.08 * R)) * 0.035 * (0.5 + uLoud) * BD(1).x;
  float clip = smoothstep(BD(1).z - 0.0015, BD(1).z + 0.0015, p.y);
  ex *= clip;
  return vec4(col, cov * clip);
}`,
  // M0 = (gain, hue, width, base), M1 = (halo, -, -, -)
  glow: `vec4 MAT(vec3 s, vec2 q, vec4 Q, float R, vec2 p, out vec3 ex) {
  ex = vec3(0.0);
  float w = max(BD(0).z * (0.7 + 0.8 * Q.x), px() * (1.5 + 2.0 * Q.x));
  float dd = max(s.x, 0.0);
  float cov = glow(dd, w) + BD(1).x * glow(dd, w * 6.0);
  // A glow is tuned for small sources: a large body glows dimmer, so its trail does not flood the screen.
  vec3 col = COL(Q, s.y, 0.3, p) * s.z * (BD(0).w * gTw + Q.x) * 0.4 * min(1.0, pow(max(BD(0).z, 0.012) / max(R, 1e-3), 2.0));
  return vec4(col, cov);
}`,
  // M0 = (gain, hue, spacing, size)
  dots: `vec4 MAT(vec3 s, vec2 q, vec4 Q, float R, vec2 p, out vec3 ex) {
  ex = vec3(0.0);
  float sp = BD(0).z;
  vec2 g = fract(q / sp + 0.5) - 0.5;
  float dm = smoothstep(BD(0).w * 0.5, BD(0).w * 0.5 - 0.12, length(g));
  float aa = px() * 1.2;
  float cov = max(smoothstep(aa, -aa, s.x), glow(s.x, sp * 0.6)) * dm;
  vec3 col = COL(Q, s.y, 0.5, p) * s.z * Q.x * 0.35;
  return vec4(col, cov);
}`,
  // M0 = (gain, hue, amount, halo), M1 = (tex, clip, halo level, -)
  textured: `vec4 MAT(vec3 s, vec2 q, vec4 Q, float R, vec2 p, out vec3 ex) {
  float amt = BD(0).z;
  float sd = s.x;
  vec3 moon = mix(vec3(1.0, 0.95, 0.85), COL(Q, 0.0, 0.5, p), 0.3);
  vec3 body = moon;
  float cut = 1.0, tex = 1.0;
#if TEX == 0
  tex = mix(1.0, 0.7 + 0.3 * fbm2(q * 1.8 / R), amt);
#elif TEX == 1
  vec3 sun = mix(uColC, uColB, smoothstep(-R, R, q.y));
  body = mix(moon, sun, amt);
  cut = q.y > 0.02 * R / 0.19 ? 1.0 : step(0.35 - q.y / R * 0.5, fract(q.y / R * 4.5 - uPhase * 0.2));
  cut = mix(1.0, cut, amt);
#else
  vec2 cell = floor(q / (R * 0.2));
  vec2 f = fract(q / (R * 0.2));
  float on = step(hash12(cell + floor(uBars) * 7.0), 0.25 + 0.5 * Q.x) * step(0.25, f.x) * step(0.3, f.y) * step(f.x, 0.75) * step(f.y, 0.8);
  body = mix(moon * 0.12, mix(vec3(1.0, 0.8, 0.5), COL(Q, 0.1, 0.5, p), 0.3) * (0.6 + 0.6 * hash12(cell)), on * amt);
#endif
  vec3 col = body * cut * tex * 0.5 * Q.x;
  float cov = smoothstep(0.0, -0.0025, sd);
  ex = COL(Q, 0.0, 0.5, p) * step(0.0, sd) * BD(0).w * (0.06 * exp(-sd * 10.0) + 0.12 * BD(1).z * exp(-sd * 4.0));
  float clip = smoothstep(BD(1).y - 0.0015, BD(1).y + 0.0015, p.y);
  ex *= clip;
  return vec4(col, cov * clip);
}`,
  // M0 = (gain, hue, chrome, -)
  chrome: `vec4 MAT(vec3 s, vec2 q, vec4 Q, float R, vec2 p, out vec3 ex) {
  vec3 nn;
  float F;
  if (gMetaOn > 0.5) {
    nn = normalize(vec3(-gMeta * 0.05, 1.0));
    F = gMetaF;
  } else {
    vec2 gd = vec2(dFdx(s.x), dFdy(s.x)) / px();
    float gl = length(gd);
    gd = gl > 1e-4 ? gd / gl : vec2(0.0);
    float h = R + s.x;
    float slope = clamp(h / sqrt(max(R * R - h * h, 1e-6)), 0.0, 6.0);
    nn = normalize(vec3(gd * slope, 1.0));
    F = 1.0 / max(1e-3, pow(1.0 + s.x / max(R, 1e-3), 2.0));
  }
  float aa = px() * 1.4;
  float cov = smoothstep(aa, -aa, s.x);
  vec3 r = reflect(vec3(0.0, 0.0, -1.0), nn);
  vec3 env = mix(uColB * 0.03, COL(Q, 0.0, 0.5, p) * 0.7, smoothstep(-0.3, 0.9, r.y));
  env += vec3(1.0) * glow(r.y - 0.2 - 0.12 * sin(r.x * 3.0 + uPhase * 0.3), 0.06);
  env += uColC * pow(max(-r.x, 0.0), 5.0) * 0.8;
  float fres = pow(1.0 - nn.z, 2.0);
  vec3 chrome = env * (0.3 + 0.7 * fres) + vec3(0.02);
  vec3 soft = COL(Q, F * 0.1, 0.5, p) * (0.25 + 0.5 * fres);
  vec3 col = mix(soft, chrome, BD(0).z) * Q.x;
  ex = (uColB * 0.003 + uColC * 0.015 * smoothstep(0.2, 1.0, F)) * (1.0 - cov);
  return vec4(col, cov);
}`,
};

// -------------------------------------------------------------- bodies

interface BodyCode {
  code: string;
  /** Statement drawing the body: `c = body_i(p, c);` (p is the name of the coordinate). */
  call: (coord: string) => string;
  /** Feedback-pass mask for point geometry (flame / sparks) shaped by a fused shape or a deformation. */
  fbMask: string | null;
}

/** Replaces the per-body macros with this body's uniform slots. */
function slot(src: string, bi: number, suffix: string): string {
  const b = bi * BODY_VEC4;
  return src
    .replace(/BD\((\d+)\)/g, (_m, k: string) => `uBd[${b + Number(k)}]`)
    .replace(/\bEA\b/g, `uBd[${b + 16}]`)
    .replace(/\bEB\b/g, `uBd[${b + 17}]`)
    .replace(/\bEC\b/g, `uBd[${b + 18}]`)
    .replace(/\bED\b/g, `uBd[${b + 19}]`)
    .replace(/\bWV(\d)\b/g, (_m, k: string) => `uWv[${bi * 4 + Number(k)}]`)
    .replace(/\b(SHP|FSH|FLD|DFM|MAT|COL|CHUE|armA|armL|edgeLocal|hzTerrain)\b/g, `$1_${suffix}`);
}

/** Shape distance field for the body (SA/SB = its parameter slots). */
function shapeCode(kind: ShapeKind, name: 'SHP' | 'FSH', slots: [number, number]): string {
  const src = SHAPE_SDF[kind];
  if (!src) throw new Error(`no distance field for ${kind}`);
  return src.replace(/\bSHP\b/g, name).replace(/\bSA\b/g, `BD(${slots[0]})`).replace(/\bSB\b/g, `BD(${slots[1]})`);
}

/** Does this body draw through its distance field in the full-screen passes? */
function drawsSdf(b: BodyGene): boolean {
  const cls = SHAPE_CLASS[b.shape.kind];
  return cls === 'sdf' || (cls === 'curve' && !!b.fuse);
}

export function isMetaball(b: BodyGene): boolean {
  return b.shape.kind === 'dot' && (b.place.p.fuse ?? 0) > 0 && !isFoldPlace(b.place.kind) && b.deform.kind === 'none' && !b.fuse;
}

function bodyCode(b: BodyGene, bi: number): BodyCode {
  const sfx = String(bi);
  const cls = SHAPE_CLASS[b.shape.kind];
  const layerK = STATIC_MATERIALS.includes(b.material.kind) ? 'uLayerK' : 'uAccum';
  const cover = b.emit.kind === 'cover';
  const C0 = bi * 6;
  const ops = b.deform.ops?.length ? `p = drawWarp(p, ${bi * 3}, int(BD(12).z + 0.5));` : '';
  const fuse = b.fuse;
  let pre = '';
  if (cls === 'field') pre += (b.shape.kind === 'scene' ? sceneField(b.material.kind) : FIELD_GLSL[b.shape.kind]) ?? '';
  if (drawsSdf(b)) pre += shapeCode(b.shape.kind, 'SHP', [2, 3]) + '\n';
  if (fuse) pre += shapeCode(fuse.shape.kind, 'FSH', [14, 15]) + '\n';
  pre += (DEFORM_GLSL[b.deform.kind] ?? DEFORM_GLSL.none) + '\n';
  if (drawsSdf(b)) pre += BODY_COL + '\n' + MATERIAL_GLSL[b.material.kind].replace(/#if TEX == (\d)/g, (_m, t: string) => `#if ${b.material.p.tex} == ${t}`).replace(/#elif TEX == (\d)/g, (_m, t: string) => `#elif ${b.material.p.tex} == ${t}`) + '\n';

  // Fuse: blend the second shape's field into s (same local frame).
  const fuseOp = !fuse
    ? ''
    : fuse.p.mode === 0
      ? `{ vec3 f = FSH(qd); f.x *= sc * dk; float kk = max(BD(13).y, 1e-3); float h = clamp(0.5 + 0.5 * (f.x - s.x) / kk, 0.0, 1.0); s = vec3(mix(f.x, s.x, h) - kk * h * (1.0 - h), mix(f.y + 0.4, s.y, h), mix(f.z, s.z, h)); }`
      : fuse.p.mode === 1
        ? `{ vec3 f = FSH(qd); f.x *= sc * dk; float t = BD(13).z; s = vec3(mix(s.x, f.x, t), mix(s.y, f.y + 0.4, t), mix(s.z, f.z, t)); }`
        : `{ vec3 f = FSH(qd); f.x *= sc * dk; float w = max(px() * 1.2, 0.0025); mreg = ${fuse.p.inside ? 'smoothstep(0.004, -0.004, f.x)' : 'glow(f.x, w * 6.0)'}; outl += glow(f.x, w) * 0.05; }`;
  const regionBoost = fuse?.p.mode === 2 ? ' * (0.05 + 1.8 * mreg)' : '';

  // One copy: local frame q (scene units), scale sc, copy info Q, centre T.
  const evalCopy = (q: string, Q: string, T: string) => {
    let s = `  {
    vec2 qq = ${q};
    float sc = max(${T}.w, 1e-3);
    float dk;
    vec2 qd = DFM(qq / sc, dk);
    vec3 s = SHP(qd);
    s.x *= sc * dk;
    float mreg = 1.0;
    ${fuseOp}
    vec3 ex;
    vec4 m = MAT(s, qq, ${Q}, max(BD(4).w, 1e-3) * sc, p, ex);
    m.a *= 1.0${regionBoost};
`;
    if (cover) {
      s += `    vec3 col = m.rgb * gain;
    c = mix(c + col * m.a * 0.3 * uAccum, mix(c, col, clamp(m.a, 0.0, 1.0)), BD(12).x);
    c += ex * gain * ${layerK};
    // The bright tip marks where a moving copy paints now (a still copy would pile it up in one spot).
    float tipW = clamp(length(${T}.xy - ${Q}.zw) / (max(BD(4).w, 1e-3) * sc * 0.15), 0.0, 1.0);
    c += mix(vec3(${Q}.x), col, 0.6) * glow(length(p - ${T}.xy), max(BD(4).w, 1e-3) * sc * 1.3) * 0.3 * BD(12).y * uAccum * tipW;
  }
`;
    } else {
      s += `    acc += m.rgb * m.a + ex;
    tip += glow(length(p - ${T}.xy), max(BD(4).w, 1e-3) * sc * 1.3) * ${Q}.x * clamp(length(${T}.xy - ${Q}.zw) / (max(BD(4).w, 1e-3) * sc * 0.15), 0.0, 1.0);
  }
`;
    }
    return s;
  };

  let placeCode = '';
  const k = b.place.kind;
  if (cls === 'field') {
    // A chunk shape: one frame (a copy's transform, or a fold).
    let q = `rot2(-uCp[${C0}].z) * (p - uCp[${C0}].xy) / max(uCp[${C0}].w, 1e-3)`;
    if (k === 'mirror') q = `rot2(-uCp[${C0}].z) * (mirrorFold(p, BD(5).x) - uCp[${C0}].xy)`;
    else if (k === 'ring') q = `ringFold(p, uCp[${C0}], BD(5).x, BD(5).y).xy`;
    else if (k === 'grid') q = `(fract(rot2(-uCp[${C0}].z) * (p - uCp[${C0}].xy) * BD(5).x + 0.5) - 0.5) / BD(5).x * 2.0`;
    placeCode = `  float dk;
  vec2 q = DFM(${q}, dk);
  vec3 col = FLD(q);
`;
    if (fuse) {
      placeCode += `  { vec3 f = FSH(q); float w = max(px() * 1.2, 0.0025); float mreg = ${fuse.p.inside ? 'smoothstep(0.004, -0.004, f.x)' : 'glow(f.x, w * 6.0)'}; col = col * (0.05 + 1.8 * mreg) + pal(BD(0).y + 0.4) * glow(f.x, w) * 0.05 * uAccum * BD(0).x; }
`;
    }
    placeCode += `  c += col;
`;
  } else if (drawsSdf(b)) {
    if (isMetaball(b)) {
      // Copies melt into one surface: an inverse-square field with an exact gradient.
      placeCode = `  int n = int(BD(4).x + 0.5);
  float F = 0.0, rs = 0.0;
  vec2 g = vec2(0.0);
  vec4 Qm = vec4(0.0);
  for (int i = 0; i < min(n, 6); i++) {
    vec4 T = uCp[${C0} + i];
    vec2 d = p - T.xy;
    float r = max(BD(2).x * T.w, 1e-3);
    float iq = 1.0 / (dot(d, d) + 1e-4);
    F += r * r * iq;
    g += -2.0 * r * r * iq * iq * d;
    rs += r;
    Qm += uCq[${C0} + i] * r * r * iq;
  }
  Qm /= max(F, 1e-5);
  float dm = rs / float(max(n, 1)) * (inversesqrt(max(F, 1e-4)) - 1.0);
  gMeta = g; gMetaOn = 1.0; gMetaF = F;
  vec3 ex;
  vec4 m = MAT(vec3(dm, 0.0, 1.0), p, Qm, max(BD(4).w, 1e-3), p, ex);
  gMetaOn = 0.0;
`;
      placeCode += cover
        ? `  vec3 col = m.rgb * gain;
  c = mix(c + col * m.a * 0.3 * uAccum, mix(c, col, clamp(m.a, 0.0, 1.0)), BD(12).x);
  c += ex * gain * ${layerK};
`
        : `  acc += m.rgb * m.a + ex;
`;
    } else if (k === 'grid') {
      placeCode = `  vec4 T0 = uCp[${C0}];
  float S = BD(5).x, jit = BD(5).y, dens = BD(5).z, lit = BD(5).w;
  float links = BD(6).x, lvl = BD(6).z, twk = BD(6).w;
  int lat = int(BD(6).y + 0.5);
  vec2 g0 = rot2(-T0.z) * (p - T0.xy) * S;
  if (lat == 0) {
    vec2 cell = floor(g0);
    int K = jit > 0.25 ? 1 : 0;
    for (int j = -K; j <= K; j++) for (int i = -K; i <= K; i++) {
      vec2 cc = cell + vec2(i, j);
      if (hash12(cc * 1.3 + 7.1) >= dens) continue;
      vec2 sp = cc + 0.5 + jit * (hash22(cc * 1.7 + 0.3) - 0.5);
      float pc = floor(hash12(cc + 3.7) * 12.0);
      float e = pow(chromaAt(pc), 2.5) * lvl;
      float gate = lit < 0.99 ? step(hash12(cc + BD(4).z), lit * (0.6 + 0.8 * uAct)) * (0.2 + 1.2 * uBeatPulse) : 1.0;
      gTw = 1.0 - 0.3 * twk + 0.3 * twk * sin(uTime * (0.5 + hash12(cc)) + pc);
      vec4 Qc = vec4(e * gate, CHUE(cc, pc), T0.xy);
${evalCopy('(g0 - sp) / S', 'Qc', 'T0').replace(/length\(p - T0\.xy\)/g, 'length(g0 - sp) / S')}
      if (links > 0.01) {
        for (int kk = 0; kk < 2; kk++) {
          vec2 nc = cc + (kk == 0 ? vec2(1.0, 0.0) : vec2(0.0, 1.0));
          if (hash12(nc * 1.3 + 7.1) >= dens) continue;
          float pc2 = floor(hash12(nc + 3.7) * 12.0);
          float link = min(e, pow(chromaAt(pc2), 2.5) * lvl) * gate;
          if (link < 0.04) continue;
          vec2 np = nc + 0.5 + jit * (hash22(nc * 1.7 + 0.3) - 0.5);
          acc += mix(COL(Qc, 0.0, 0.3, p), COL(vec4(0.0, CHUE(nc, pc2), 0.0, 0.0), 0.0, 0.3, p), 0.5) * link * links * glow(sdSeg(g0, sp, np) / S, px() * 1.1) * 0.8 * 0.2;
        }
      }
    }
  } else {
    vec4 h = lat == 1 ? hexCell(g0) : triCell(g0);
    vec2 cc = h.zw;
    if (hash12(cc * 1.3 + 7.1) < dens) {
      float pc = floor(hash12(cc) * 12.0);
      float e = pow(chromaAt(pc), 2.0) * lvl;
      float gate = 1.0;
      if (lit < 0.99) { gate = step(hash12(cc + BD(4).z), lit * (0.6 + 0.8 * uAct)) * (0.2 + 1.2 * uBeatPulse); e *= e; }
      gTw = 1.0 - 0.3 * twk + 0.3 * twk * sin(uTime * (0.5 + hash12(cc)) + pc);
      vec4 Qc = vec4(e * gate, CHUE(cc, pc), T0.xy);
${evalCopy('h.xy / S', 'Qc', 'T0').replace(/length\(p - T0\.xy\)/g, 'length(h.xy) / S')}
    }
  }
  gTw = 1.0;
`;
    } else if (k === 'ring') {
      placeCode = `  vec4 T0 = uCp[${C0}];
  vec3 rq = ringFold(p, T0, BD(5).x, BD(5).y);
  int ci = int(mod(rq.z, 4.0) + 4.0) % 4;
  vec4 Qc = uCq[${C0} + ci];
${evalCopy('rq.xy', 'Qc', 'T0').replace(/length\(p - T0\.xy\)/g, 'length(rq.xy)')}
`;
    } else if (k === 'mirror') {
      placeCode = `  vec4 T0 = uCp[${C0}];
  vec2 mp = mirrorFold(p, BD(5).x);
  vec4 Qc = uCq[${C0}];
${evalCopy('rot2(-T0.z) * (mp - T0.xy)', 'Qc', 'T0').replace(/length\(p - T0\.xy\)/g, 'length(mp - T0.xy)')}
`;
    } else if ((b.place.p.fuse ?? 0) > 0) {
      // Copies melt together (smooth union), then one material pass.
      placeCode = `  int n = int(BD(4).x + 0.5);
  float kf = max(BD(4).y, 1e-3);
  vec3 sF = vec3(1e3, 0.0, 1.0);
  vec4 QF = vec4(0.0);
  float wsum = 0.0;
  for (int i = 0; i < min(n, 6); i++) {
    vec4 T = uCp[${C0} + i];
    vec4 Q = uCq[${C0} + i];
    vec2 qq = rot2(-T.z) * (p - T.xy);
    float sc = max(T.w, 1e-3);
    float dk;
    vec2 qd = DFM(qq / sc, dk);
    vec3 s = SHP(qd);
    s.x *= sc * dk;
    float mreg = 1.0;
    ${fuseOp}
    float h = clamp(0.5 + 0.5 * (s.x - sF.x) / kf, 0.0, 1.0);
    sF = vec3(mix(s.x, sF.x, h) - kf * h * (1.0 - h), mix(s.y, sF.y, h), mix(s.z, sF.z, h));
    float wq = 1.0 / (1.0 + max(s.x, 0.0) * 40.0);
    QF += Q * wq;
    wsum += wq;
  }
  QF /= max(wsum, 1e-4);
  vec3 ex;
  vec4 m = MAT(sF, p - uCp[${C0}].xy, QF, max(BD(4).w, 1e-3), p, ex);
`;
      placeCode += cover
        ? `  vec3 col = m.rgb * gain;
  c = mix(c + col * m.a * 0.3 * uAccum, mix(c, col, clamp(m.a, 0.0, 1.0)), BD(12).x);
  c += ex * gain * ${layerK};
`
        : `  acc += m.rgb * m.a + ex;
`;
    } else {
      // Explicit copies; each sweeps from its previous position (a continuous stroke at any frame rate).
      placeCode = `  int n = int(BD(4).x + 0.5);
  for (int i = 0; i < min(n, 6); i++) {
    vec4 T = uCp[${C0} + i];
    vec4 Q = uCq[${C0} + i];
    vec2 pv = Q.zw;
    vec2 dv = T.xy - pv;
    float hh = clamp(dot(p - pv, dv) / max(dot(dv, dv), 1e-8), 0.0, 1.0);
${evalCopy('rot2(-T.z) * (p - (pv + dv * hh))', 'Q', 'T')}
  }
`;
    }
    if (fuse?.p.mode === 2) placeCode = `  float outl = 0.0;\n` + placeCode + `  acc += pal(BD(0).y + 0.4) * outl;\n`;
    if (!cover) placeCode += `  c += (acc * ${layerK} + vec3(1.0) * tip * 0.25 * BD(12).y * uAccum) * gain * BD(12).w;\n`;
  }

  let fbMask: string | null = null;
  if ((cls === 'flame' || (b.emit.kind === 'sparks' && !drawsSdf(b))) && (fuse || b.deform.kind !== 'none')) {
    // Point geometry fades fast outside the fused shape (or the deformed silhouette) and lingers inside.
    fbMask = `float fbMask_${sfx}(vec2 p0) {
  vec2 p = p0;
  ${ops}
  vec4 T0 = uCp[${C0}];
  vec2 q = rot2(-T0.z) * (p - T0.xy);
  float dk;
  vec2 qd = DFM(q, dk);
  float d = ${fuse ? 'FSH(qd).x * dk' : `(length(qd) - max(BD(4).w, 1e-3)) * dk`};
  return ${fuse && !fuse.p.inside ? 'glow(d, 0.09)' : 'smoothstep(0.004, -0.004, d)'};
}`;
  }

  const helpers = `vec2 mirrorFold(vec2 p, float axis) { if (axis < 0.5) return vec2(abs(p.x), p.y); if (axis < 1.5) return vec2(p.x, abs(p.y)); return abs(p); }
vec3 ringFold(vec2 p, vec4 T, float nr, float rad) {
  vec2 d = rot2(-T.z) * (p - T.xy);
  float seg = TAU / max(nr, 1.0);
  float a = atan(d.y, d.x);
  float id = floor(a / seg + 0.5);
  float ar = a - id * seg;
  vec2 q = length(d) * vec2(cos(ar), sin(ar)) - vec2(rad, 0.0);
  return vec3(q.y, q.x, id);
}`;
  const body = `vec3 body_${sfx}(vec2 p, vec3 c) {
  ${ops}
  float gain = BD(0).x;
  vec3 acc = vec3(0.0);
  float tip = 0.0;
${placeCode}  return c;
}`;
  const needHelpers = bi === 0 ? helpers + '\n' : '';
  const code = slot(needHelpers + pre + (fbMask ? fbMask + '\n' : '') + body, bi, sfx);
  const draws = cls === 'field' || drawsSdf(b);
  return {
    code,
    call: (coord: string) => blendCall(draws ? `  c = body_${sfx}(${coord}, c);\n` : '', b.material.p.blend ?? 0),
    fbMask: fbMask ? `fbMask_${sfx}` : null,
  };
}

// ------------------------------------------------------------- builders

export interface Sources {
  feedback: string;
  composite: string;
  /** The ray-marched scene pass (only when a body has a 'scene' shape). */
  scene?: string;
}

export function buildSources(g: Genome): Sources {
  const nb = g.bodies.length;
  const defs: string[] = [];
  if (g.carrier.kind === 'none') defs.push('NO_FEEDBACK');
  if (g.carrier.kind === 'fluid') defs.push('USE_FLUID');
  if (g.carrier.kind === 'flow') defs.push('USE_FLOW');
  if (g.tone.p.reflect > 0.5) defs.push('REFLECT');
  if (g.tone.p.tonemap > 0.5) defs.push('LOG_TONE');
  const pre = HEAD + defs.map((d) => `#define ${d}\n`).join('') + COMMON + lib(nb) + FLAME_VARIATION_GLSL + DRAW_GLSL + BLEND_GLSL;

  const warpOps = g.chain.map((o, i) => (o.stage === 'warp' ? opCode(o, i) : '')).join('');
  const viewOps = g.chain.map((o, i) => (o.stage === 'view' ? opCode(o, i) : '')).join('');

  let code = '';
  let fbDraw = '';
  let topDraw = '';
  let masks = '';
  g.bodies.forEach((b, bi) => {
    const bc = bodyCode(b, bi);
    code += bc.code + '\n';
    if (bodyLayer(b) === 'fb') fbDraw += bc.call('p');
    else topDraw += bc.call('q');
    if (bc.fbMask) masks += `  c *= mix(0.88, 1.0, ${bc.fbMask}(p));\n`;
  });

  const feedback = pre + /* glsl */ `
in vec2 vUv;
uniform sampler2D uPrev, uVel;
uniform vec2 uSimTexel;
uniform float uFluidAmt, uBlur, uDecaySub, uFlowAmt, uFlowScale;
uniform float uSharpen, uGrain, uSharpNoise, uBorder, uBorderW;
uniform vec3 uBorderCol;
out vec4 o;
${WATER_GLSL}
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
${code}
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
  // Water: the carried picture is read through the ripple field's slope (refraction).
  vec2 wsl = uWaterAmt > 0.0 ? waterSlope(vUv) : vec2(0.0);
  suv += clamp(wsl, -0.5, 0.5) * uWaterAmt * 0.005;
  vec3 pv = prevAt(suv);
  // Glint on the slopes facing the light, scaled by the per-frame fade so it holds steady instead of piling up.
  if (uWaterAmt > 0.0) pv += mix(uColA, vec3(1.0), 0.3) * min(max(dot(wsl, vec2(-0.6, 0.8)), 0.0), 0.5) * uWaterAmt * uLayerK * 0.6;
  // With a border, whatever the warp pulls in from beyond the edge is the border colour (clamped
  // sampling), so escaping flow fills with it and the fractal set stays dark.
  if (uBorder > 0.001) {
    vec2 eo = min(suv, 1.0 - suv);
    pv += uBorderCol * uBorder * (1.0 - smoothstep(0.0, 0.002, min(eo.x, eo.y)));
  }
  if (uSharpen > 0.001) {
    // Unsharp mask against a two-ring blur, then the 8-bit style clamp that keeps the patterns bistable.
    vec2 rr = vec2(uGrain / uAspect, uGrain);
    vec3 bl = vec3(0.0);
    for (int i = 0; i < 8; i++) {
      float a = float(i) * 0.7853982 + 0.3927;
      vec2 off = vec2(cos(a), sin(a)) * rr;
      bl += texture(uPrev, suv + off).rgb + texture(uPrev, suv - off * 0.5).rgb;
    }
    bl *= 0.0625;
    // Sharpened on the brightness, so the channels grow one pattern and keep its colours.
    float lv = luma(pv);
    // Grain only where there is light nearby, so patterns grow out of what is drawn, not out of black.
    float lb = luma(bl);
    float lg = max(lv + (lv - lb) * uSharpen + (hash12(vUv * uRes + fract(uTime * 7.31) * 97.0) - 0.5) * uSharpNoise * smoothstep(0.0, 0.06, lb), 0.0);
    // Growth out of darkness takes the palette's second colour.
    pv = lv > 0.004 ? pv * (lg / lv) : uColB * (lg / max(luma(uColB), 0.05));
    pv = mix(pv, vec3(luma(pv)), 0.04 * uSharpen);
    pv = clamp(pv / max(1.0, max(pv.r, max(pv.g, pv.b))), 0.0, 1.0);
  }
  c = max(pv * uDecay - uDecaySub, 0.0);
  if (uBorder > 0.001) {
    // Outer border in colour, a dark inner border just inside it (MilkDrop's ob / ib pair).
    vec2 e = min(vUv, 1.0 - vUv) * asp;
    float ed = min(e.x, e.y);
    c *= 1.0 - uBorder * smoothstep(uBorderW * 4.0, uBorderW * 3.0, ed);
    c = mix(c, uBorderCol, uBorder * smoothstep(uBorderW, uBorderW * 0.5, ed));
  }
${masks}#endif
${fbDraw}  o = vec4(clamp(c, vec3(0.0), vec3(64.0)), 1.0);
}`;

  const composite = pre + /* glsl */ `
in vec2 vUv;
uniform sampler2D uFb;
uniform float uWeight, uSat, uSweep, uReflectY;
out vec4 o;
vec3 fb(vec2 q) { return texture(uFb, q / vec2(uAspect, 1.0) + 0.5).rgb; }
${RELIEF_GLSL}vec2 view(vec2 p) {
${viewOps}  return p;
}
${code}
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
  vec3 c = reliefLit(q, (fb(q + ox + oy) + fb(q - ox - oy) + fb(q + ox - oy) + fb(q - ox + oy)) * 0.25) * vMul + vAdd;
${topDraw}  float l = max(c.r, max(c.g, c.b));
  float b = log(1.0 + l * 24.0) / log(25.0);
  c = c / max(l, 1e-5) * pow(b, 1.1) * 0.55;
#else
  vec3 c = reliefLit(q, fb(q)) * vMul + vAdd;
${topDraw}#endif
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
  const scene = g.bodies.some((b) => b.shape.kind === 'scene') ? HEAD + COMMON + lib(nb) + SCENE_PASS : undefined;
  return scene ? { feedback, composite, scene } : { feedback, composite };
}

/** True when the body is drawn by the curve geometry pass. */
export function drawsCurve(b: BodyGene): boolean {
  return SHAPE_CLASS[b.shape.kind] === 'curve' && !b.fuse;
}
export { sdfCapable };

// ------------------------------------------------------ line geometry

/**
 * Curve geometry: one draw per copy. uW: (form, amp, x, y) (radius, turns, ra, rb)
 * (phase, rotation, hue, -) (pendulum phases); uT: the copy (x, y, angle, scale);
 * uFlip: mirror scale; uDk: deform kind (0 none, 1 arms, 2 wobble, 3 noise, 4 twist)
 * with uDp and the arm slots; uDrBase / uDrN: the body's draw ops.
 */
export const WAVE_VS = HEAD + COMMON + lib(3) + FLAME_VARIATION_GLSL + DRAW_GLSL + /* glsl */ `
uniform float uN, uThick, uBright;
uniform vec4 uW[4];
uniform vec4 uT;
uniform vec2 uFlip;
uniform int uDk;
uniform vec4 uDp, uArmA0, uArmA1, uArmL0, uArmL1;
uniform int uDrBase, uDrN;
out float vSide;
out vec3 vCol;
out float vK;
vec2 deformFwd(vec2 q) {
  if (uDk == 1) {
    float a = atan(q.y, q.x);
    float ext = 0.0;
    int n = min(int(uDp.x + 0.5), 7);
    for (int j = 0; j < n; j++) {
      float ang = (j < 4 ? uArmA0[j] : uArmA1[j - 4]);
      float da = atan(sin(a - ang), cos(a - ang));
      ext += (j < 4 ? uArmL0[j] : uArmL1[j - 4]) * exp(-da * da / (uDp.w * uDp.w));
    }
    return q * (1.0 + ext * uDp.y);
  }
  if (uDk == 2) return q * (1.0 + uDp.y * sin(uDp.x * atan(q.y, q.x) + uDp.z));
  if (uDk == 3) return q + curlNoise(q * uDp.y + 4.0, uDp.z) * uDp.x;
  if (uDk == 4) return rot2(-uDp.x * length(q)) * q;
  return q;
}
${SUPERSCOPE_GLSL}
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
  } else if (sh == 3) {
    float t = k * 42.0;
    float damp = exp(-k * 2.0);
    float x = sin(B.z * t + C.x) + 0.5 * sin(B.z * 2.0 * t + 0.01 + C.x * 1.3);
    float y = sin(B.w * t + C.x * 0.7 + 1.3) + 0.5 * sin(B.w * 1.5 * t - 0.01);
    q = vec2(x, y) * damp * B.x * 0.62 * (1.0 + waveAt(k) * A.y * 0.4);
  } else if (sh == 4) {
    float a = (k - 0.5) * B.y * 0.37;
    q = (B.x + waveAt(k) * A.y * 0.4) * vec2(cos(a), sin(a));
  } else if (sh == 6) {
    q = superscopeAt(k);
  } else {
    float t = k * 42.0;
    float damp = exp(-k * 2.0);
    vec4 D = uW[3];
    float x = sin(B.x * t + D.x) + C.w * sin(B.y * t + D.y);
    float y = sin(B.z * t + D.z) + C.w * sin(B.w * t + D.w);
    q = vec2(x, y) * damp * C.x;
  }
  q = rot2(C.y) * deformFwd(q);
  q = rot2(uT.z) * q * uT.w * uFlip + uT.xy;
  return drawWarpFwd(q, uDrBase, uDrN);
}
vec3 curveColor(float k) {
  int sh = int(uW[0].x + 0.5);
  float env = sh == 0 ? smoothstep(0.0, 0.15, k) * smoothstep(1.0, 0.85, k) : 1.0;
  float damp = sh == 3 || sh == 5 ? 0.5 + 0.5 * exp(-k * 1.5) : 1.0;
  float t = sh == 4 ? 0.5 + 0.5 * sin(k * TAU) : k;
  vec3 c = sh == 0 ? mix(pal(uW[2].z), vec3(1.0), 0.2) : mix(pal(uW[2].z), pal(uW[2].z + 0.66), t * uW[2].w);
  if (sh == 6) return mix(pal(uW[2].z), pal(uW[2].z + 0.33), (0.5 + 0.5 * gScopeZ) * uW[2].w) * (0.35 + 0.65 * (0.5 + 0.5 * gScopeZ)) * superscopeFade(k);
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
  vK = k * uN;
  vCol = curveColor(k) * uBright;
  gl_Position = vec4(pp.x / (uAspect * 0.5), pp.y / 0.5, 0.0, 1.0);
}`;

/** uDash > 0: dotted (dash period in vertices); uHalo: wider soft falloff. */
export const WAVE_FS = HEAD + /* glsl */ `
in float vSide;
in vec3 vCol;
in float vK;
uniform float uDash, uSoft, uLines;
out vec4 o;
void main() {
  // Interlace blend: only every other scanline.
  if (uLines > 0.5 && fract(gl_FragCoord.y * 0.5) < 0.5) discard;
  float s = 1.0 - vSide * vSide;
  float prof = mix(s * s + pow(s, 10.0) * 0.4, sqrt(max(s, 0.0)) * 0.6, uSoft);
  float dash = uDash > 0.5 ? smoothstep(0.5, 0.2, abs(fract(vK / uDash) - 0.5) * 2.0) : 1.0;
  o = vec4(vCol * prof * dash, 1.0);
}`;
