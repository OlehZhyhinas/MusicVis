// A finite feather blade: curved rachis, tapered vanes, and antialiased barbs.
import type { Schema } from '../genome';

export const PLUME_SCHEMA: Schema = {
  size: { min: 0.025, max: 0.4, def: 0.3 },
  width: { min: 0.12, max: 0.65, def: 0.38 },
  bend: { min: -0.7, max: 0.7, def: 0.18 },
  taper: { min: 0.4, max: 1.8, def: 0.75 },
  barbs: { min: 10, max: 70, def: 38, int: true },
  split: { min: 0, max: 1, def: 0.35 },
  sheen: { min: 0, max: 1, def: 0.7 },
};

// SA = size, width, bend, taper; SB = barb count, splitting, sheen, unused.
export const PLUME_GLSL = /* glsl */ `
vec3 SHP(vec2 p) {
  float size = max(SA.x, 0.001);
  vec2 q = p / size;
  float y = clamp(q.y, -1.0, 1.0);
  float profile = max(0.0, 1.0 - y*y);
  float axis = SA.z * profile * 0.5;
  float x = q.x - axis;
  float width = SA.y * pow(max(profile, 0.00001), SA.w);
  float across = clamp(x / max(width, 0.001), -1.0, 1.0);
  float barbPhase = (y - abs(x)*0.8) * SB.x * PI;
  float aa = 1.0 - smoothstep(0.5, 2.0, fwidth(barbPhase));
  float barb = 0.5 + 0.5 * cos(barbPhase);
  float edge = width * (1.0 - SB.y * 0.28 * pow(barb, 8.0) * aa);
  // Bound the blade at both tips. A local slope correction keeps the silhouette
  // crisp while width/taper/bend change, including very narrow vanes.
  float slope = -2.0*y*SA.y*SA.w*pow(max(profile, 0.001), SA.w-1.0);
  float side = (abs(x)-edge) / sqrt(1.0 + pow(abs(slope)+abs(SA.z*y), 2.0));
  float d = max(side, abs(q.y)-1.0);
  float z = sqrt(max(0.0, 1.0-across*across));
  float light = 0.32 + 0.68 * max(0.0, dot(vec3(across, 0.2, z), normalize(vec3(-0.5,0.3,0.9))));
  light *= 1.0 - 0.3 * barb * aa;
  float shaft = exp(-abs(x)*100.0);
  light += 0.28 * shaft;
  light += SB.z * 0.45 * pow(max(0.0, z*0.85-across*0.45), 12.0);
  return vec3(d*size, y*0.2 + SB.z*(0.28*across + 0.08*barb*aa), light);
}
`;

export function packPlume(E: Float32Array, o: number, P: (k: string) => number): number {
  ['size', 'width', 'bend', 'taper', 'barbs', 'split', 'sheen'].forEach((k, i) => { E[o+i] = P(k); });
  E[o+7] = 0;
  return P('size') * Math.max(1, P('width') + Math.abs(P('bend'))*0.5);
}
