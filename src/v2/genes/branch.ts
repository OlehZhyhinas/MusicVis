// A folded binary tree distance field. Each generation splits at the preceding tip;
// evaluating its nearer mirrored child makes up to 63 limbs cost only six segments.
// Growth reveals successive generations continuously, so a phrase can open a crown.
import type { Schema } from '../genome';

export const BRANCH_SCHEMA: Schema = {
  size: { min: 0.025, max: 0.4, def: 0.23 },
  width: { min: 0.008, max: 0.12, def: 0.035 },
  spread: { min: 0.2, max: 1.15, def: 0.65 },
  ratio: { min: 0.48, max: 0.7, def: 0.64 },
  levels: { min: 2, max: 6, def: 5, choices: [2, 3, 4, 5, 6] },
  grow: { min: 0.2, max: 1, def: 0.92 },
  bend: { min: 0, max: 0.45, def: 0.12 },
};

// SA = size, width, spread, ratio; SB = levels, grow, bend, musical phase.
export const BRANCH_GLSL = /* glsl */ `
vec3 SHP(vec2 p) {
  float size = max(SA.x, 0.001);
  vec2 q = p / size + vec2(0.0, 0.8);
  float limb = 0.65, width = SA.y, d = 1e3, shade = 0.0;
  for (int i = 0; i < min(int(SB.x + 0.5), 6); i++) {
    float reveal = clamp(SB.y * SB.x - float(i), 0.0, 1.0);
    if (reveal <= 0.001) break;
    float h = limb * reveal;
    float t = clamp(q.y / max(h, 0.001), 0.0, 1.0);
    float di = length(q - vec2(0.0, h * t)) - width * mix(1.0, 0.6, t);
    if (di < d) { d = di; shade = (float(i) + t) / SB.x; }
    q.y -= limb;
    q.x = abs(q.x);
    float angle = SA.z + SB.z * sin(SB.w + float(i) * 0.9);
    q = rot2(angle) * q;
    limb *= SA.w;
    width *= 0.66;
  }
  return vec3(d * size, shade * 0.65, 0.6 + 0.6 * shade);
}
`;

/** Pack into either the body's shape slots or a fused shape's slots. */
export function packBranch(E: Float32Array, o: number, P: (k: string) => number, phase: number): number {
  ['size', 'width', 'spread', 'ratio', 'levels', 'grow', 'bend'].forEach((k, i) => { E[o + i] = P(k); });
  E[o + 7] = phase;
  return P('size') * 1.6;
}
