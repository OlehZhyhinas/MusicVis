// Superscope shape gene (after Winamp AVS's Superscope component).
//
// An AVS superscope runs a small per-point script: for each of n points, i goes 0..1, v is the
// waveform or spectrum value at i, and the script sets x, y (and a colour). Most classic scopes are
// a 3D parametric curve, pushed outward by v, turned in 3D and perspective-projected. Rather than a
// free expression language (which breeds badly), this gene is a small expression vocabulary: a
// curve family plus its integer frequencies, the audio push, the 3D tumble and the perspective.
// Points are drawn by the curve geometry pass (dots with the dots material, lines otherwise), and
// nearer points are brighter, as AVS scopes usually colour by depth.
//
//   family  0 torus knot (p, q), 1 sphere spiral (p turns), 2 rose (p / q petals), 3 coiled ring
//           (p x q coils), 4 3D lissajous (p : q), 5 ring tunnel (p + 2 rings streaming toward you)
//   audio   how far v pushes each point outward; spec 0: v is the waveform, 1: the spectrum
//   spinX / spinY  tumble about the x / y axes, turns per bar; persp: perspective depth
//   n       points per scope

import type { Schema } from '../genome';

const TURNS = [-1, -0.5, -0.25, -0.125, -0.0625, 0, 0.0625, 0.125, 0.25, 0.5, 1];
const FREQ = [1, 2, 3, 4, 5, 6, 7, 8];

export const SUPERSCOPE_SCHEMA: Schema = {
  family: { min: 0, max: 5, def: 0, choices: [0, 1, 2, 3, 4, 5] },
  p: { min: 1, max: 8, def: 2, choices: FREQ },
  q: { min: 1, max: 8, def: 3, choices: FREQ },
  size: { min: 0.08, max: 0.45, def: 0.25 },
  audio: { min: 0, max: 1, def: 0.4 },
  spec: { min: 0, max: 1, def: 0, choices: [0, 1] },
  spinX: { min: -1, max: 1, def: 0.125, choices: TURNS },
  spinY: { min: -1, max: 1, def: 0.0625, choices: TURNS },
  persp: { min: 0, max: 1, def: 0.5 },
  n: { min: 256, max: 2048, def: 1024, choices: [256, 512, 1024, 2048] },
};

/** Curve-pass form id of a superscope (the curve shape's own forms are 0-5). */
export const SUPERSCOPE_FORM = 6;
/** Distance-field evaluation cost stand-in (a superscope has no distance field). */
export const SUPERSCOPE_COST = 0.25;

/**
 * Packs the 16 curve slots: A (form, audio, family, spec), B (size, p, q, persp),
 * C (phase, rotation, hue, detail), D (x angle, y angle, tunnel phase, beat kick).
 */
export function packSuperscope(
  u: Float32Array, w0: number, sp: Record<string, number>, P: (k: string) => number,
  spin: number, phase: number, hue: number, detail: number, beat: number, bass: number,
): void {
  u[w0] = SUPERSCOPE_FORM; u[w0 + 1] = P('audio') * (0.7 + 0.6 * bass); u[w0 + 2] = sp.family; u[w0 + 3] = sp.spec;
  u[w0 + 4] = P('size') * (1 + 0.06 * beat); u[w0 + 5] = sp.p; u[w0 + 6] = sp.q; u[w0 + 7] = P('persp');
  u[w0 + 8] = phase; u[w0 + 9] = 0; u[w0 + 10] = hue; u[w0 + 11] = detail;
  u[w0 + 12] = spin * sp.spinX + 0.35; u[w0 + 13] = spin * sp.spinY; u[w0 + 14] = phase * 3; u[w0 + 15] = beat;
}

/**
 * GLSL for the curve vertex shader: superscopeAt(k) returns the projected point (body space) and
 * sets gScopeZ (depth, -1 far .. 1 near); superscopeFade(k) hides the jumps between tunnel rings.
 * Needs uW, waveAt, specAt, TAU and rot2 from the curve pass.
 */
export const SUPERSCOPE_GLSL = /* glsl */ `
float gScopeZ = 0.0;
vec3 scopeBase(float k, int fam, float p, float q, float tph) {
  float t = k * TAU;
  if (fam == 0) {
    float r = 2.0 + cos(q * t);
    return vec3(r * cos(p * t), r * sin(p * t), -sin(q * t) * 1.4) / 3.0;
  }
  if (fam == 1) {
    float z = k * 2.0 - 1.0;
    float a = k * TAU * (4.0 + p * 3.0);
    float s = sqrt(max(1.0 - z * z, 0.0));
    return vec3(s * cos(a), s * sin(a), z);
  }
  if (fam == 2) {
    float th = t * q;
    float r = cos(p / q * th);
    return vec3(r * cos(th), r * sin(th), 0.35 * sin(th * 0.5 + tph));
  }
  if (fam == 3) {
    float nc = p * q * 2.0 + 2.0;
    float r = 0.3 * cos(nc * t);
    return vec3((0.68 + r) * cos(t), (0.68 + r) * sin(t), 0.3 * sin(nc * t));
  }
  if (fam == 4) {
    return vec3(sin(p * t + tph), sin(q * t), sin((p + q) * t * 0.5 + tph * 0.7));
  }
  float R = p + 2.0;
  float ring = floor(k * R * 0.9999);
  float a = fract(k * R) * TAU;
  float z = fract(ring / R + tph * 0.25) * 2.0 - 1.0;
  return vec3(cos(a) * 0.8, sin(a) * 0.8, z * 1.2);
}
vec2 superscopeAt(float k) {
  vec4 A = uW[0], B = uW[1], D = uW[3];
  int fam = int(A.z + 0.5);
  vec3 P = scopeBase(k, fam, B.y, B.z, D.z);
  float v = A.w > 0.5 ? specAt(0.02 + 0.6 * fract(k * (fam == 5 ? B.y + 2.0 : 1.0))) : waveAt(k);
  P *= 1.0 + A.y * v * (A.w > 0.5 ? 0.9 : 0.5) + 0.08 * D.w;
  P.yz = rot2(D.x) * P.yz;
  P.xz = rot2(D.y) * P.xz;
  gScopeZ = clamp(P.z, -1.0, 1.0);
  float d = 1.0 - 0.5 * B.w * P.z;
  return P.xy / d * B.x;
}
float superscopeFade(float k) {
  vec4 A = uW[0], B = uW[1];
  if (int(A.z + 0.5) != 5) return 1.0;
  float f = fract(k * (B.y + 2.0));
  return smoothstep(0.0, 0.03, f) * smoothstep(1.0, 0.97, f);
}
`;
