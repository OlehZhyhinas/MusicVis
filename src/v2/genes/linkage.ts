// Forward kinematics in a distance field: each link starts at the preceding
// joint and inherits its angle. Flex propagates along the chain on the musical
// clock, giving a moving skeleton rather than a rigid body spinning as a whole.
import type { Schema } from '../genome';

export const LINKAGE_SCHEMA: Schema = {
  size: { min: 0.025, max: 0.4, def: 0.25 },
  joints: { min: 2, max: 8, def: 5, choices: [2, 3, 4, 5, 6, 7, 8] },
  width: { min: 0.015, max: 0.16, def: 0.055 },
  curl: { min: -0.9, max: 0.9, def: 0.15 },
  flex: { min: 0, max: 1.2, def: 0.55 },
  taper: { min: 0.6, max: 1, def: 0.92 },
  knuckle: { min: 0, max: 1, def: 0.65 },
};

// SA = size, joints, width, curl; SB = flex, taper, knuckle, musical phase.
export const LINKAGE_GLSL = /* glsl */ `
vec3 SHP(vec2 p) {
  float size = max(SA.x, 0.001);
  vec2 q = p / size;
  vec2 root = vec2(0.0, -0.8);
  float angle = 0.0, limb = 1.6 / SA.y, width = SA.z;
  float d = 1e3, shade = 0.0, light = 1.0;
  for (int i = 0; i < min(int(SA.y + 0.5), 8); i++) {
    angle += SA.w + SB.x * sin(SB.w - float(i) * 0.85);
    vec2 tip = root + limb * vec2(sin(angle), cos(angle));
    float bone = sdSeg(q, root, tip) - width;
    float joint = length(q - root) - width * (1.0 + SB.z * 1.2);
    float di = min(bone, joint);
    if (di < d) {
      d = di;
      shade = float(i) / SA.y * 0.65;
      light = joint < bone ? 1.25 : 0.7 + 0.25 * cos(angle - 0.7);
    }
    root = tip;
    limb *= SB.y;
    width *= SB.y;
  }
  float cap = length(q - root) - width * (1.0 + SB.z * 1.2);
  if (cap < d) { d = cap; shade = 0.75; light = 1.3; }
  return vec3(d * size, shade, light);
}
`;

export function packLinkage(E: Float32Array, o: number, P: (k: string) => number, phase: number): number {
  ['size', 'joints', 'width', 'curl', 'flex', 'taper', 'knuckle'].forEach((k, i) => { E[o + i] = P(k); });
  E[o + 7] = phase;
  // A curling chain can reach back past its root; keep crowd/fusion fitting conservative.
  return P('size') * 2.6;
}
