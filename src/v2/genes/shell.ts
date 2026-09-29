// A logarithmic coil with a rounded, ribbed surface and a flared aperture.
// Five angular branches cover the maximum winding count without a segment march.
import type { Schema } from '../genome';

export const SHELL_SCHEMA: Schema = {
  size: { min: 0.025, max: 0.4, def: 0.27 },
  turns: { min: 0.75, max: 4.5, def: 2.6 },
  growth: { min: 0.12, max: 0.38, def: 0.2 },
  width: { min: 0.06, max: 0.38, def: 0.2 },
  ribs: { min: 8, max: 48, def: 26, int: true },
  relief: { min: 0, max: 1, def: 0.65 },
  aperture: { min: 0, max: 1, def: 0.35 },
};

// SA = size, turns, growth, width; SB = ribs, relief, aperture, unused.
export const SHELL_GLSL = /* glsl */ `
vec3 SHP(vec2 p) {
  float size = max(SA.x, 0.001);
  vec2 q = p / size;
  float r = length(q), phi = r > 0.00001 ? mod(atan(q.y, q.x) + TAU, TAU) : 0.0;
  float end = SA.y * TAU, growth = SA.z;
  float d = 1e3, along = 0.0;
  vec2 surface = vec2(0.0);
  for (int j = 0; j < 5; j++) {
    float t = clamp(phi + TAU * float(j), 0.0, end);
    // Refine the closest point, including both rounded end caps. Radial distance
    // alone creates a visible angular seam where the outer opening ends.
    for (int k = 0; k < 3; k++) {
      float radius = exp(growth * (t - end));
      vec2 axis = vec2(cos(t), sin(t)), tangent = vec2(-axis.y, axis.x);
      vec2 delta = radius * axis - q;
      vec2 velocity = radius * (growth * axis + tangent);
      vec2 acceleration = radius * ((growth*growth - 1.0)*axis + 2.0*growth*tangent);
      float denom = max(0.25*dot(velocity, velocity), dot(velocity, velocity) + dot(delta, acceleration));
      t = clamp(t - clamp(dot(delta, velocity) / max(denom, 1e-9), -0.5, 0.5), 0.0, end);
    }
    float radius = exp(growth * (t - end));
    float width = max(0.012, radius * SA.w) * (1.0 + SB.z * smoothstep(end - 0.9, end, t));
    vec2 delta = q - radius * vec2(cos(t), sin(t));
    float di = length(delta) - width;
    if (di < d) { d = di; surface = delta / width; along = t; }
  }
  float z = sqrt(max(0.0, 1.0 - dot(surface, surface)));
  float light = 0.25 + 0.75 * max(0.0, dot(vec3(surface, z), normalize(vec3(-0.4, 0.6, 0.8))));
  float ribPhase = along * SB.x;
  float ribs = (0.5 + 0.5*cos(ribPhase)) * (1.0-smoothstep(0.5, 2.0, fwidth(ribPhase)));
  light *= 1.0 - SB.y * 0.45 * ribs;
  return vec3(d * size, along / max(end, 0.01) * 0.45 + surface.y * 0.08, light);
}
`;

export function packShell(E: Float32Array, o: number, P: (k: string) => number): number {
  ['size', 'turns', 'growth', 'width', 'ribs', 'relief', 'aperture'].forEach((k, i) => { E[o + i] = P(k); });
  E[o + 7] = 0;
  return P('size') * (1 + P('width') * (1 + P('aperture')));
}
