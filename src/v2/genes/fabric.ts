// A finite draped sheet, not a full-screen effect. Its silhouette, pleat lighting
// and thread pattern travel with the body through placements, warps and fusion.
import type { Schema } from '../genome';

export const FABRIC_SCHEMA: Schema = {
  size: { min: 0.025, max: 0.4, def: 0.25 },
  aspect: { min: 0.25, max: 2, def: 1.2 },
  folds: { min: 2, max: 18, def: 7, int: true },
  depth: { min: 0, max: 1, def: 0.6 },
  drape: { min: 0, max: 1, def: 0.45 },
  weave: { min: 0, max: 1, def: 0.25 },
  flutter: { min: 0, max: 1, def: 0.35 },
};

// SA = size, aspect, folds, depth; SB = drape, weave, flutter, musical phase.
export const FABRIC_GLSL = /* glsl */ `
vec3 SHP(vec2 p) {
  float size = max(SA.x, 0.001);
  vec2 q = p / size;
  q.x += SB.z * 0.18 * sin(q.y * 3.0 + SB.w);
  float across = clamp(q.x / SA.y, -1.0, 1.0);
  q.y += SB.x * 0.55 * (1.0 - across * across);
  float pleat = q.x * SA.z * PI / SA.y + SB.w;
  q.y += SA.w * 0.12 * sin(pleat) + SB.z * 0.12 * sin(q.x * 3.0 - SB.w);
  vec2 edge = abs(q) - vec2(SA.y - 0.06, 0.74);
  float d = length(max(edge, 0.0)) + min(max(edge.x, edge.y), 0.0) - 0.06;
  // Broad directional highlights make folds legible with fill as well as chrome.
  float light = mix(1.0, 0.22 + 1.05 * pow(0.5 + 0.5 * cos(pleat), 2.0), SA.w);
  // Fade threads before they alias when the body shrinks or is repeated in a grid.
  vec2 thread = q * 100.0;
  float aa = max(fwidth(thread.x), fwidth(thread.y));
  light *= 1.0 - SB.y * 0.2 * (0.5 + 0.5 * sin(thread.x) * sin(thread.y)) * (1.0 - smoothstep(0.5, 2.0, aa));
  return vec3(d * size, 0.18 * sin(pleat) + q.y * 0.12, light);
}
`;

export function packFabric(E: Float32Array, o: number, P: (k: string) => number, phase: number): number {
  ['size', 'aspect', 'folds', 'depth', 'drape', 'weave', 'flutter'].forEach((k, i) => { E[o + i] = P(k); });
  E[o + 7] = phase;
  return P('size') * Math.max(P('aspect') + 0.2, 1.6);
}
