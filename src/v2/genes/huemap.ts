// Hue map: a per-pixel colour transform of the whole picture (the classic MilkDrop comp-shader
// trick of mapping brightness through a cycling colour ramp). Brightness picks a place on the
// palette ramp, repeated `bands` times from black to white, so smooth gradients become psychedelic
// contour bands; `poster` hardens the bands into flat posterized steps, `drift` scrolls them through
// the palette on the bar clock, and `solar` folds the brightness first (solarize: highlights turn
// dark again), which works on its own too. Applied at the end of the composite, after relief.
// Off (huemap 0 and solar 0) costs one uniform branch.

/** GLSL for the composite: vec3 hueMapped(vec3 c) using luma() and pal(). */
export const HUEMAP_GLSL = /* glsl */ `
uniform vec4 uHueMap;  // amount, bands, phase (palette turns), solar
uniform float uPoster;
vec3 hueMapped(vec3 c) {
  if (uHueMap.x < 0.001 && uHueMap.w < 0.001) return c;
  vec3 k = 1.0 - exp(-1.6 * max(c, 0.0));
  float v = luma(k);
  // Solarize: fold the brightness (and each channel) back down past the midpoint.
  vec3 ks = mix(k, 1.0 - abs(2.0 * k - 1.0), uHueMap.w);
  v = mix(v, 1.0 - abs(2.0 * v - 1.0), uHueMap.w);
  float x = v * uHueMap.y;
  float steps = mix(64.0, 4.0, uPoster);
  float xq = floor(x * steps) / steps;
  x = mix(x, xq + smoothstep(0.0, 1.0, fract(x * steps)) / steps * (1.0 - uPoster), step(0.001, uPoster));
  // Black stays black, the ramp brightens with the level and the very brightest parts burn white.
  vec3 m = mix(pal(x + uHueMap.z) * (0.45 + 0.9 * v), vec3(1.15), smoothstep(0.8, 1.0, v) * 0.8) * smoothstep(0.03, 0.22, v);
  // Back to scene brightness: expand the folded / mapped values out of the 0..1 range.
  vec3 base = -log(max(1.0 - ks * 0.98, 0.02)) / 1.6;
  vec3 mapped = -log(max(1.0 - clamp(m, 0.0, 1.0) * 0.98, 0.02)) / 1.6;
  return mix(base, mapped, uHueMap.x);
}
`;

/** Uniform values: `bars` is the song clock; drift scrolls one palette turn every 4 / drift bars. */
export function hueMapUniforms(get: (k: string) => number, bars: number): { v: [number, number, number, number]; poster: number } {
  const drift = get('drift');
  return { v: [get('huemap'), get('bands'), (bars * drift * 0.25) % 1, get('solar')], poster: get('poster') };
}
