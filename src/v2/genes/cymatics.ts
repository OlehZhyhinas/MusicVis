// Cymatics: Chladni figures, a full-screen chunk shape ('field' class). A vibrating plate
// (square or round) sits at the body's placement; sand gathers on the nodal lines of its current
// standing-wave mode (m, n). The song picks the mode: the two strongest pitch classes (chords),
// the two loudest spectrum bands, or the section. A new mode is taken only on the body's clock
// grid (every `hold` bars on its feel clock), and then the sand scatters and settles onto the new
// lines over `settle` beats while the pattern morphs. Major keys use the symmetric mode
// combination, minor keys the antisymmetric one.
//
// Square plate: f = cos(n pi x) cos(m pi y) +- cos(m pi x) cos(n pi y) (the classic free-plate
// approximation). Round plate: f = cos(n theta) sin(m pi r): n nodal diameters, m nodal rings.

import type { ParamSpec, Schema } from '../genome';

const P = (min: number, max: number, def: number): ParamSpec => ({ min, max, def });
const I = (min: number, max: number, def: number): ParamSpec => ({ min, max, def, int: true });
const C = (choices: number[], def: number): ParamSpec => ({ min: Math.min(...choices), max: Math.max(...choices), def, choices });

/**
 * plate: 0 square, 1 round. size: plate half-size. modes: highest mode number used. source: what
 * picks the mode, 0 chroma (the two strongest pitch classes), 1 spectrum (the two loudest bands),
 * 2 section. hold: bars between possible mode changes. settle: beats the sand takes to gather.
 * sand: grains vs smooth lines. line: nodal line width. shake: grain jitter and swelling with the
 * bass. rim: plate edge glow.
 */
export const CYMATICS_SCHEMA: Schema = {
  plate: C([0, 1], 0),
  size: P(0.15, 0.5, 0.4),
  modes: I(3, 12, 7),
  source: C([0, 1, 2], 0),
  hold: C([1, 2, 4, 8], 1),
  settle: C([0.5, 1, 2, 4], 2),
  sand: P(0, 1, 0.75),
  line: P(0.2, 3, 1),
  shake: P(0, 1, 0.4),
  rim: P(0, 1, 0.3),
};

/**
 * Estimated GPU ms at 1440p: one full-screen pass, two mode evaluations and a grain hash per pixel
 * (timer-query median on Apple M5 Pro at 2560x1440: ~0.5-1 ms over a waveform seed).
 */
export function cymaticsCost(_p: Record<string, number>): number {
  return 1.1;
}

/**
 * vec3 FLD(vec2 p): EA = (gain, hue, level, time), EB = (plate, size, sand, line),
 * EC = (m0, n0, sign0, morph 0..1), ED = (m1, n1, sign1, shake now), BD(2) = (rim, grain seed, scatter, -).
 */
export const CYMATICS_GLSL = /* glsl */ `
float cymMode(vec2 u, float m, float n, float s, float round_) {
  if (round_ > 0.5) {
    float r = length(u);
    return cos(n * atan(u.y, u.x)) * sin(m * PI * r);
  }
  vec2 a = u * PI;
  return cos(n * a.x) * cos(m * a.y) + s * cos(m * a.x) * cos(n * a.y);
}
vec3 FLD(vec2 p) {
  vec4 A = EA, B = EB, M0 = EC, M1 = ED, X = BD(2);
  float R = max(B.y, 0.05);
  vec2 u = p / R;
  bool round_ = B.x > 0.5;
  float inside = round_ ? 1.0 - length(u) : 1.0 - max(abs(u.x), abs(u.y));
  float aa = 1.5 * px() / R;
  float mask = smoothstep(-aa, aa, inside);
  vec3 col = pal(A.y);
  vec3 c = pal(A.y + 0.5) * glow(inside * R, 0.004) * X.x * (0.4 + 0.6 * A.z);
  float w = clamp(M0.w, 0.0, 1.0);
  float f = mix(cymMode(u, M0.x, M0.y, M0.z, B.x), cymMode(u, M1.x, M1.y, M1.z, B.x), w);
  float gpx = length(vec2(dFdx(f), dFdy(f))) + 1e-5;
  float dpx = abs(f) / gpx;
  // Scatter while the plate changes mode, then settle back onto the nodal lines.
  float spread = 1.0 + 5.0 * X.z + 2.0 * M1.w;
  float lineW = max(B.w, 0.2) * 1.2 * spread;
  float ln = glow(dpx, lineW);
  vec2 cell = floor(p * uRes.y * 0.5);
  float h = hash12(cell + floor(X.y) * 17.0);
  vec2 jit = (hash22(cell + X.y) - 0.5) * M1.w;
  float dens = glow(max(dpx - length(jit) * 3.0, 0.0), lineW * 2.5);
  float grain = step(h, dens * 0.85) * (0.6 + 0.4 * hash12(cell * 1.7));
  float sandV = mix(ln, grain, B.z);
  vec3 sandC = mix(col, vec3(1.0, 0.95, 0.85), 0.45);
  c += sandC * sandV * (0.35 + 0.65 * A.z) * mask;
  c += pal(A.y + 0.33) * abs(f) * 0.015 * A.z * (1.0 + M1.w) * mask;
  c += col * 0.012 * mask;
  return c * A.x * uLayerK;
}`;

/** The engine state packCymatics needs. */
export interface CymaticsFrame {
  chroma: Float32Array;
  spec: Float32Array;
  keyTonic: number;
  minor: boolean;
  sectionIndex: number;
  bass: number;
  loud: number;
  bpm: number;
}

/** The mode pair (m, n, sign) the music asks for now; m != n. */
export function cymaticsTarget(p: Record<string, number>, f: CymaticsFrame, smooth: ArrayLike<number>): [number, number, number] {
  const top = p.modes;
  const sign = f.minor ? -1 : 1;
  let a = 0;
  let b = 1;
  if (p.source === 2) {
    const k = ((Math.floor(f.sectionIndex) % 64) + 64) % 64;
    a = (k * 5 + 1) % top;
    b = (k * 3 + 2) % top;
  } else {
    // The two strongest bins (pitch classes relative to the key, or spectrum bands).
    const n = smooth.length;
    let i1 = 0;
    let i2 = 1;
    for (let i = 0; i < n; i++) if (smooth[i] > smooth[i1]) i1 = i;
    if (i2 === i1) i2 = 0;
    for (let i = 0; i < n; i++) if (i !== i1 && smooth[i] > smooth[i2]) i2 = i;
    // Pitch classes go round the circle of fifths from the key: consonant chords give simple figures.
    const rel = (i: number) => (p.source === 0 ? (((((i - f.keyTonic) % 12) + 12) % 12) * 7) % 12 : i);
    a = rel(i1) % top;
    b = rel(i2) % top;
  }
  let m = 1 + a;
  let nn = 1 + b;
  if (m === nn) nn = nn >= top ? m - 1 : nn + 1;
  if (nn < 1) nn = 2;
  return [Math.min(m, nn), Math.max(m, nn), sign];
}

/**
 * Packs the field slots. E: uniforms; o: slot 16 (EA) of the body; o2: slot 2; P: a parameter after
 * reactions; mem / key: per-body memory; tick: true on the body's clock grid every `hold` bars;
 * resp: the body's response curve.
 */
export function packCymatics(
  E: Float32Array, o: number, o2: number, P: (k: string) => number, p: Record<string, number>, f: CymaticsFrame,
  mem: Record<string, number>, key: (k: string) => string, tick: boolean, resp: (k: string, raw: number) => number, sdt: number,
): void {
  const mm = (k: string, init = 0) => mem[key(k)] ?? (mem[key(k)] = init);
  const ease = (k: string, target: number, rate: number) => (mem[key(k)] = mm(k, target) + (target - mm(k, target)) * (1 - Math.exp(-rate * sdt)));
  // Smoothed bins: 12 pitch classes, or `modes` spectrum bands.
  const bins = p.source === 1 ? p.modes : 12;
  const sm: number[] = [];
  for (let i = 0; i < bins; i++) {
    let raw: number;
    if (p.source === 1) {
      const n = f.spec.length;
      const a = Math.floor((i / bins) * n * 0.8);
      const b = Math.max(a + 1, Math.floor(((i + 1) / bins) * n * 0.8));
      raw = 0;
      for (let j = a; j < b; j++) raw += f.spec[j];
      raw /= b - a;
    } else raw = f.chroma[i] ?? 0;
    sm.push(ease(`c${i}`, raw, 2));
  }
  const [tm, tn, ts] = cymaticsTarget(p, f, sm);
  if (mem[key('m1')] === undefined) {
    mem[key('m0')] = mem[key('m1')] = tm;
    mem[key('n0')] = mem[key('n1')] = tn;
    mem[key('s0')] = mem[key('s1')] = ts;
    mem[key('w')] = 1;
  }
  const changed = tm !== mem[key('m1')] || tn !== mem[key('n1')] || ts !== mem[key('s1')];
  if (tick && changed && mm('w') >= 1) {
    mem[key('m0')] = mem[key('m1')]; mem[key('n0')] = mem[key('n1')]; mem[key('s0')] = mem[key('s1')];
    mem[key('m1')] = tm; mem[key('n1')] = tn; mem[key('s1')] = ts;
    mem[key('w')] = 0;
    mem[key('seed')] = (mm('seed') + 1) % 4096;
  }
  const w = (mem[key('w')] = Math.min(1, mm('w', 1) + (sdt * f.bpm) / 60 / p.settle));
  const level = ease('lv', 0.3 + 0.7 * resp('cy', Math.max(f.loud, f.bass)), 5);
  const shake = P('shake') * resp('cb', f.bass);
  const t = (mem[key('t')] = (mm('t') + sdt) % 1024);
  E[o + 2] = level; E[o + 3] = t;
  E[o + 4] = p.plate; E[o + 5] = P('size'); E[o + 6] = P('sand'); E[o + 7] = P('line');
  E[o + 8] = mm('m0'); E[o + 9] = mm('n0'); E[o + 10] = mm('s0'); E[o + 11] = w * w * (3 - 2 * w);
  E[o + 12] = mm('m1'); E[o + 13] = mm('n1'); E[o + 14] = mm('s1'); E[o + 15] = shake;
  E[o2] = P('rim'); E[o2 + 1] = mm('seed'); E[o2 + 2] = Math.sin(Math.PI * w);
}
