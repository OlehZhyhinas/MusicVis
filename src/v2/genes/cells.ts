// Cells: a Voronoi cell field (cell foam, membranes, scales), a full-screen chunk shape ('field'
// class), after the cell-foam looks of MilkDrop presets such as Flexi's mindblob family and
// alien fish pond. Schema, shader and per-frame packing live here; genome.ts, glsl.ts and
// engine.ts only reference it.
//
// `scale` cells per unit, each around a seed point that wanders on its own small orbit (`speed`,
// and further with the bass); `warp` bends the whole lattice with noise so walls go organic.
// mode 0 foam: thick bright walls around cells filled with the second palette colour;
// mode 1 veins: thin glowing walls on dark cells; mode 2 domes: each cell shaded as a lit dome
// with a dark nucleus. `wall` is the wall width (swelling with the loudness), `fill` how bright
// the cell interiors are, `var` how far each cell's hue strays, `pulse` how strongly a
// beat-chosen handful of cells flares on each beat.

import type { ParamSpec, Schema } from '../genome';

const P = (min: number, max: number, def: number): ParamSpec => ({ min, max, def });
const C = (choices: number[], def: number): ParamSpec => ({ min: Math.min(...choices), max: Math.max(...choices), def, choices });

export const CELLS_SCHEMA: Schema = {
  mode: C([0, 1, 2], 0),
  scale: P(1.5, 14, 4),
  speed: P(0, 1, 0.3),
  warp: P(0, 1, 0.3),
  wall: P(0, 1, 0.35),
  fill: P(0, 1, 0.5),
  var: P(0, 1, 0.2),
  pulse: P(0, 1, 0.3),
};

/** Estimated GPU ms at 1440p: a 3x3 cell search per pixel, plus the noise warp when used. */
export function cellsCost(p: Record<string, number>): number {
  return 0.7 + (p.warp > 0.001 ? 0.25 : 0);
}

/**
 * vec3 FLD(vec2 p): EA = (gain, hue, time, beat pulse), EB = (scale, wall, fill, warp),
 * EC = (mode, pulse, var, bass jiggle), ED = (beat index, loudness, -, -).
 */
export const CELLS_GLSL = /* glsl */ `
vec3 FLD(vec2 p) {
  vec4 A = EA, B = EB, C = EC, D = ED;
  float t = A.z;
  vec2 q = p * B.x;
  if (B.w > 0.001) q += B.w * 2.2 * (vec2(fbm2(q * 0.45 + vec2(0.0, t * 0.15)), fbm2(q * 0.45 + vec2(7.3, -t * 0.13))) - 0.5);
  vec2 ip = floor(q), fp = fract(q);
  float f1 = 9.0, f2 = 9.0;
  vec2 id = vec2(0.0), r1 = vec2(0.0);
  for (int j = -1; j <= 1; j++) {
    for (int i = -1; i <= 1; i++) {
      vec2 g = vec2(float(i), float(j));
      vec2 h = hash22(ip + g);
      vec2 o = 0.5 + (0.3 + 0.12 * C.w) * sin(t * (0.5 + h.yx) + 6.2831 * h);
      vec2 r = g + o - fp;
      float d = dot(r, r);
      if (d < f1) { f2 = f1; f1 = d; id = ip + g; r1 = r; }
      else if (d < f2) f2 = d;
    }
  }
  f1 = sqrt(f1);
  f2 = sqrt(f2);
  float e = f2 - f1;
  float h = hash12(id * 1.37 + 3.1);
  float aa = fwidth(e) + 1e-4;
  float w = (0.02 + 0.3 * B.y) * (0.8 + 0.4 * D.y);
  int mode = int(C.x + 0.5);
  // A few cells chosen per beat flare with the beat pulse.
  float chosen = step(hash12(id + floor(D.x) * 7.13), 0.25);
  float flare = 1.0 + C.y * A.w * chosen * 2.5;
  vec3 cellCol = pal(A.y + 0.33 + (h - 0.5) * C.z) * flare;
  vec3 wallCol = pal(A.y + (h - 0.5) * C.z * 0.3);
  vec3 c;
  if (mode == 0) {
    float inWall = 1.0 - smoothstep(w - aa, w + aa, e);
    c = mix(cellCol * B.z * (0.35 + 0.65 * smoothstep(0.0, 0.6, e)), wallCol * 1.4, inWall);
    // A dark rim where wall meets cell keeps the two colours crisp.
    c *= 0.2 + 0.8 * smoothstep(0.0, 0.03 + aa, abs(e - w));
  } else if (mode == 1) {
    float glowW = w * 0.35 + aa;
    c = wallCol * (exp(-e * e / (glowW * glowW)) * 1.6 + exp(-e / (w + 0.05)) * 0.25) + cellCol * B.z * 0.25 * smoothstep(0.0, 0.5, e);
  } else {
    float dome = smoothstep(0.0, 0.5, e);
    float lit = clamp(0.55 + 0.9 * dot(-r1, vec2(0.6, 0.8)), 0.0, 1.4);
    float nucleus = smoothstep(0.08, 0.14, length(r1 - 0.08 * vec2(sin(t + h * 6.2831), cos(t * 1.3 + h * 6.2831))));
    c = cellCol * (0.15 + B.z) * dome * lit * mix(0.25, 1.0, nucleus) + wallCol * 0.4 * (1.0 - smoothstep(0.0, w + aa, e));
  }
  return c * A.x * 0.55 * uLayerK;
}
`;

export interface CellsFrame {
  speed: number;
  beats: number;
  beatPulse: number;
  bass: number;
  loud: number;
}

/** Packs the four field slots (EA..ED at E[o..o+15]); the cell clock runs at `speed`, faster with the bass. */
export function packCells(
  E: Float32Array, o: number, P: (k: string) => number, p: Record<string, number>, f: CellsFrame,
  mem: Record<string, number>, key: (k: string) => string, resp: (k: string, raw: number) => number, sdt: number,
): void {
  const mm = (k: string, init = 0) => mem[key(k)] ?? (mem[key(k)] = init);
  const bass = (mem[key('b')] = mm('b') + (resp('cb', f.bass) - mm('b')) * (1 - Math.exp(-6 * sdt)));
  const t = (mem[key('t')] = (mm('t') + sdt * f.speed * (0.15 + 1.6 * P('speed')) * (0.6 + 0.8 * bass)) % 4096);
  E[o + 2] = t; E[o + 3] = resp('cp', f.beatPulse);
  E[o + 4] = P('scale'); E[o + 5] = P('wall'); E[o + 6] = P('fill'); E[o + 7] = P('warp');
  E[o + 8] = p.mode; E[o + 9] = P('pulse'); E[o + 10] = P('var'); E[o + 11] = bass;
  E[o + 12] = Math.floor(f.beats) % 4096; E[o + 13] = resp('cl', f.loud);
}
