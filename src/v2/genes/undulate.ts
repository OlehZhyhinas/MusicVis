// An area-preserving travelling shear. Forward curves and inverse distance
// fields share the same map: its wave coordinate is unchanged by the shear.
import type { Schema } from '../genome';

export const UNDULATE_SCHEMA: Schema = {
  amp: { min: 0, max: 0.18, def: 0.06 },
  span: { min: 0.08, max: 1.2, def: 0.55 },
  rate: { min: -1, max: 1, def: 0.25, choices: [-1, -0.5, -0.25, 0, 0.25, 0.5, 1] },
  angle: { min: -0.5, max: 0.5, def: 0 },
  pin: { min: 0, max: 1, def: 1 },
};

// D = amplitude, wavelength, phase radians, axis radians; pin in the next vec4.
export function packUndulate(E: Float32Array, o: number, P: (k: string) => number, bars: number): void {
  E[o] = P('amp'); E[o + 1] = P('span');
  E[o + 2] = (bars * P('rate') % 1) * 2 * Math.PI;
  E[o + 3] = P('angle') * 2 * Math.PI; E[o + 4] = P('pin');
}

export const UNDULATE_GLSL = /* glsl */ `
vec2 undulateMap(vec2 q, vec4 D, float pin, float direction) {
  vec2 r = rot2(D.w) * q;
  r.x += direction * D.x * (sin(TAU * r.y / D.y - D.z) - pin * sin(-D.z));
  return rot2(-D.w) * r;
}
`;
