// Volumetric beams: concert-lighting light shafts through haze (moving heads, lasers, gobo
// patterns), a full-screen chunk shape ('field' class). Kept in its own file so the gene's
// schema, shader and per-frame packing live together; genome.ts, glsl.ts and engine.ts only
// reference it.
//
// A rig of `count` fixtures sits on a truss (length `spread`) at the body's placement; each
// throws a cone (`width` = divergence) whose rest aims fan out by `fan` and swing by `sweep` in
// one of five patterns, on the body's own musical clock (its feel gene: bar-locked or free, and
// its clock unit scales the sweep). The beams light a shared haze field once per pixel, so the
// cost grows only a little per beam. Strobe-free by design: the level is smoothed and the beat
// accent lights one beam at a time (a chase), never the whole screen.

import type { ParamSpec, Schema } from '../genome';

const P = (min: number, max: number, def: number): ParamSpec => ({ min, max, def });
const I = (min: number, max: number, def: number): ParamSpec => ({ min, max, def, int: true });
const C = (choices: number[], def: number): ParamSpec => ({ min: Math.min(...choices), max: Math.max(...choices), def, choices });

/** Most fixtures a rig may have (the shader loop bound). */
export const MAX_BEAMS = 12;

/**
 * pattern: 0 unison (every head swings together), 1 scissor (the two halves swing mirrored),
 * 2 chase wave (the swing ripples along the truss), 3 alternate (odd and even heads opposite),
 * 4 step (each head jumps to a new aim every `period` beats and holds).
 * period: bars per sweep cycle. gobo: 0 open, 1 breakup (sub-beams), 2 ring (hollow cone), 3 textured.
 * hues: palette step from one head to the next. accent: how much the beat chase brightens one head.
 */
export const BEAMS_SCHEMA: Schema = {
  count: I(1, MAX_BEAMS, 6),
  spread: P(0, 1.4, 0.8),
  fan: P(0, 1, 0.35),
  sweep: P(0, 1, 0.4),
  pattern: C([0, 1, 2, 3, 4], 2),
  period: C([1, 2, 4, 8], 2),
  width: P(0.005, 0.2, 0.05),
  haze: P(0, 1, 0.5),
  gobo: C([0, 1, 2, 3], 0),
  hues: P(0, 0.5, 0.08),
  length: P(0.3, 2.5, 1.2),
  flare: P(0, 1, 0.5),
  accent: P(0, 1, 0.5),
};

/**
 * Estimated GPU ms at 1440p: the full-screen pass with its haze field plus each beam (textured gobos
 * add a noise lookup). Timer-query measurements on Apple M5 Pro at 2560x1440, whole frame: 8 open
 * beams 2.5 ms, 12 textured beams 2.55 ms (the model is slightly conservative for large rigs).
 */
export function beamsCost(p: Record<string, number>): number {
  return 1.4 + p.count * (p.gobo === 3 ? 0.11 : 0.08);
}

/**
 * vec3 FLD(vec2 p): EA = (gain, hue, level, haze time), EB = (count, spread, fan, sweep),
 * EC = (pattern, sweep phase in cycles, width, haze), ED = (gobo, hues, length, rest aim),
 * BD(2) = (step index, step fraction, flare, accent), BD(3) = (chase clock in beats, -, -, -).
 */
export const BEAMS_GLSL = /* glsl */ `
vec3 FLD(vec2 p) {
  vec4 A = EA, B = EB, C = EC, D = ED, S = BD(2), T = BD(3);
  int n = int(B.x + 0.5);
  int pat = int(C.x + 0.5);
  int gobo = int(D.x + 0.5);
  float ph = C.y * TAU;
  float hz = fbm2(p * 2.2 + vec2(A.w * 0.05, A.w * 0.021));
  float haze = mix(1.0, smoothstep(0.15, 0.85, hz) * 1.6, C.w) * (0.3 + 0.9 * C.w);
  float chase = mod(floor(T.x), max(float(n), 1.0));
  float cf = fract(T.x);
  float kick = smoothstep(0.0, 0.08, cf) * exp(-cf * 3.0);
  vec3 c = vec3(0.0);
  for (int i = 0; i < ${MAX_BEAMS}; i++) {
    if (i >= n) break;
    float fi = float(i);
    float u = n > 1 ? fi / float(n - 1) * 2.0 - 1.0 : 0.0;
    float off;
    if (pat == 0) off = sin(ph);
    else if (pat == 1) off = sin(ph) * (u < 0.0 ? -1.0 : 1.0);
    else if (pat == 2) off = sin(ph - (u * 0.5 + 0.5) * PI);
    else if (pat == 3) off = sin(ph) * (mod(fi, 2.0) < 0.5 ? 1.0 : -1.0);
    else {
      float a0 = hash11(S.x * 7.13 + fi * 1.37) * 2.0 - 1.0;
      float a1 = hash11((S.x - 1.0) * 7.13 + fi * 1.37) * 2.0 - 1.0;
      off = mix(a1, a0, smoothstep(0.0, 0.4, S.y));
    }
    float ang = D.w + u * B.z * PI * 0.5 + off * B.w * PI * 0.4;
    vec2 dir = vec2(cos(ang), sin(ang));
    vec2 d = p - vec2(u * B.y * 0.5, 0.0);
    float along = dot(d, dir);
    float lvl = A.z * (1.0 + S.w * 1.5 * kick * step(abs(fi - chase), 0.5));
    vec3 col = pal(A.y + D.y * fi);
    c += col * glow(length(d), 0.012) * S.z * lvl * 0.6;
    if (along <= 0.0) continue;
    float perp = d.x * dir.y - d.y * dir.x;
    float r = perp / (along * C.z + 0.004);
    if (abs(r) > 3.0) continue;
    float cone = exp(-r * r);
    float g = 1.0;
    if (gobo == 1) g = 0.2 + 0.8 * pow(0.5 + 0.5 * cos(r * PI * 2.5), 3.0);
    else if (gobo == 2) { cone = exp(-r * r * 0.35); g = smoothstep(0.3, 1.3, abs(r)) * 1.5; }
    else if (gobo == 3) g = 0.3 + 0.9 * vnoise(vec2(r * 2.5 + fi * 7.3, along * 3.0 - A.w * 0.4));
    float hot = 0.02 / (0.02 + along * C.z);
    float atten = exp(-along / D.z) * smoothstep(0.0, 0.03, along);
    float I = cone * g * atten * (0.3 + 0.7 * hot) * haze * lvl;
    c += mix(col, vec3(1.0), 0.35 * hot * cone) * I;
  }
  return c * 0.35 * A.x * uLayerK;
}`;

/** The engine state packBeams needs (a structural slice of the engine's per-body context). */
export interface BeamsFrame {
  /** Body clock bars (feel-scaled, grid-locked or free). */
  bars: number;
  loud: number;
  melodic: number;
  drop: number;
  speed: number;
}

/**
 * Packs the field slots. E: the uniform array; o: slot 16 (EA) of the body; o2: slot 2;
 * P: a parameter after reactions; mem / key: per-body memory; y: the rig's height (aim flips
 * upward when the rig sits low); resp: the body's response curve.
 */
export function packBeams(
  E: Float32Array, o: number, o2: number, P: (k: string) => number, p: Record<string, number>, f: BeamsFrame,
  mem: Record<string, number>, key: (k: string) => string, y: number, resp: (k: string, raw: number) => number, sdt: number,
): void {
  const mm = (k: string, init = 0) => mem[key(k)] ?? (mem[key(k)] = init);
  const ease = (k: string, target: number, rate: number) => (mem[key(k)] = mm(k, target) + (target - mm(k, target)) * (1 - Math.exp(-rate * sdt)));
  // Smoothed (never flashing) level: the body's response to loudness and the melodic parts, a lift on drops.
  const level = ease('lv', 0.3 + 0.7 * resp('bm', Math.max(f.loud, f.melodic)) + 0.3 * f.drop, 6);
  const t = (mem[key('t')] = (mm('t') + sdt * f.speed) % 1024);
  const aim = ease('aim', y < -0.1 ? Math.PI / 2 : -Math.PI / 2, 3);
  const period = p.period;
  const stepPos = (f.bars * 4) / period;
  E[o + 2] = level; E[o + 3] = t;
  E[o + 4] = p.count; E[o + 5] = P('spread'); E[o + 6] = P('fan'); E[o + 7] = P('sweep');
  E[o + 8] = p.pattern; E[o + 9] = (f.bars / period) % 1; E[o + 10] = P('width'); E[o + 11] = P('haze');
  E[o + 12] = p.gobo; E[o + 13] = P('hues'); E[o + 14] = P('length'); E[o + 15] = aim;
  E[o2] = Math.floor(stepPos) % 4096; E[o2 + 1] = stepPos % 1; E[o2 + 2] = P('flare'); E[o2 + 3] = P('accent');
  E[o2 + 4] = (f.bars * 4) % 4096;
}
