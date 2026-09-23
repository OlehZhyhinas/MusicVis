// The 12 fractal-flame variations, copied from src/render/flame.ts so the pure
// V2 modules (genome, operators, shader builders) load without the GL classes.
// Keep in the same order and definitions as flame.ts: the Flame class there
// reads weights in this slot order.

export const FLAME_VARIATIONS = [
  'linear', 'sinusoidal', 'spherical', 'swirl',
  'horseshoe', 'polar', 'handkerchief', 'heart',
  'disc', 'spiral', 'hyperbolic', 'julia',
] as const;
export type FlameVar = (typeof FLAME_VARIATIONS)[number];

export const FLAME_VARIATION_GLSL = /* glsl */ `
float flameHash(vec2 p) { return fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453); }
vec2 V_linear(vec2 p) { return p; }
vec2 V_sinusoidal(vec2 p) { return sin(p); }
vec2 V_spherical(vec2 p) { return p / (dot(p, p) + 1e-6); }
vec2 V_swirl(vec2 p) {
  float r2 = dot(p, p);
  float s = sin(r2), c = cos(r2);
  return vec2(p.x * s - p.y * c, p.x * c + p.y * s);
}
vec2 V_horseshoe(vec2 p) {
  float r = length(p) + 1e-6;
  return vec2((p.x - p.y) * (p.x + p.y), 2.0 * p.x * p.y) / r;
}
vec2 V_polar(vec2 p) { return vec2(atan(p.x, p.y) / 3.14159265, length(p) - 1.0); }
vec2 V_handkerchief(vec2 p) {
  float r = length(p), t = atan(p.x, p.y);
  return r * vec2(sin(t + r), cos(t - r));
}
vec2 V_heart(vec2 p) {
  float r = length(p), t = atan(p.x, p.y);
  return r * vec2(sin(t * r), -cos(t * r));
}
vec2 V_disc(vec2 p) {
  float r = length(p), t = atan(p.x, p.y) / 3.14159265;
  return t * vec2(sin(3.14159265 * r), cos(3.14159265 * r));
}
vec2 V_spiral(vec2 p) {
  float r = length(p) + 1e-6, t = atan(p.x, p.y);
  return vec2(cos(t) + sin(r), sin(t) - cos(r)) / r;
}
vec2 V_hyperbolic(vec2 p) {
  float r = length(p) + 1e-6, t = atan(p.x, p.y);
  return vec2(sin(t) / r, r * cos(t));
}
vec2 V_julia(vec2 p) {
  float r = sqrt(length(p)), t = atan(p.x, p.y) * 0.5 + step(0.5, flameHash(p * 311.7)) * 3.14159265;
  return r * vec2(cos(t), sin(t));
}
`;
