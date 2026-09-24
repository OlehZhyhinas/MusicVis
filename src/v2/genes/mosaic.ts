// Mosaic: a fold op that breaks the picture into blocks (pixel art, LED walls, glossy tile grids),
// after MilkDrop presets such as Waltra's square orgy and the blocky pixel looks of suksma and
// Hexcollie. Every point takes the picture at the centre of its cell; `shape` picks square tiles,
// round LED dots or hexagons, `gap` opens dark grout between them (view stage only), `angle` and
// `lock` turn the grid, and `pulse` swells the cells with the bass. In the warp stage the carried
// picture itself is quantised every frame, so feedback grows blocky, pixel-automaton patterns.

import type { ParamSpec, Schema } from '../genome';

const P = (min: number, max: number, def: number): ParamSpec => ({ min, max, def });
const C = (choices: number[], def: number): ParamSpec => ({ min: Math.min(...choices), max: Math.max(...choices), def, choices });
const LOCKS = [-0.25, -0.125, -0.0625, 0, 0.0625, 0.125, 0.25];

export const MOSAIC_SCHEMA: Schema = {
  size: P(0.006, 0.15, 0.04),
  shape: C([0, 1, 2], 0),
  gap: P(0, 0.6, 0.1),
  angle: P(0, 0.25, 0),
  lock: C(LOCKS, 0),
  pulse: P(0, 1, 0),
};

/** OA = (cell size, shape, gap, grid angle rad). */
export const MOSAIC_GLSL = `{ float s = max(OA.x, 1e-3); vec2 q = rot2(-OA.w) * p; vec2 g, f; if (OA.y > 1.5) { vec2 r = vec2(1.0, 1.7320508) * s; vec2 h = r * 0.5; vec2 a = mod(q, r) - h; vec2 b = mod(q - h, r) - h; vec2 d = dot(a, a) < dot(b, b) ? a : b; g = q - d; f = d / s; } else { g = (floor(q / s) + 0.5) * s; f = (q - g) / s; } float r0 = 0.5 * (1.0 - OA.z); float fw = px() / s; float dist = OA.y > 1.5 ? max(abs(f.x), dot(abs(f), vec2(0.5, 0.8660254))) : OA.y > 0.5 ? length(f) : max(abs(f.x), abs(f.y)); vMul *= OA.z > 0.001 ? smoothstep(r0 + fw, r0 - fw, dist) : 1.0; p = rot2(OA.w) * g; }`;

/** Per-frame slots: the cells swell with the bass (pulse) and the grid turns bar-locked. */
export function packMosaic(a: Float32Array, j: number, P: (k: string) => number, p: Record<string, number>, bass: number, spin: number): void {
  a[j] = P('size') * (1 + 0.6 * P('pulse') * bass);
  a[j + 1] = p.shape;
  a[j + 2] = P('gap');
  a[j + 3] = P('angle') * Math.PI * 2 + spin * p.lock;
}
