// Relief: the carried picture lit as a height map (the classic MilkDrop emboss / bump-lit comp shader).
// Brightness is the height; its gradient gives a surface normal that a light from `light` (turns)
// shades, with a specular highlight (`gloss`) and, with `metal`, colour taken from the palette as if
// the surface reflected it (liquid chrome). Applied in the composite to the feedback layer only, so
// top-layer bodies stay flat on the embossed picture. Off (relief 0) costs one uniform branch.

/** GLSL for the composite: vec3 reliefLit(vec2 q, vec3 c) using fb(), luma(), pal(). */
export const RELIEF_GLSL = /* glsl */ `
uniform vec4 uRelief;  // amount, bump, light angle (rad), gloss
uniform float uMetal;
float reliefH(vec2 q) { return 1.0 - exp(-2.0 * luma(fb(q))); }
vec3 reliefLit(vec2 q, vec3 c) {
  if (uRelief.x < 0.001) return c;
  const float e = 0.0022;
  vec2 g = vec2(reliefH(q + vec2(e, 0.0)) - reliefH(q - vec2(e, 0.0)), reliefH(q + vec2(0.0, e)) - reliefH(q - vec2(0.0, e)));
  vec3 n = normalize(vec3(-g * (0.35 * uRelief.y) / (2.0 * e) * 0.02, 1.0));
  vec3 L = normalize(vec3(cos(uRelief.z) * 0.75, sin(uRelief.z) * 0.75, 0.66));
  float dif = max(dot(n, L), 0.0);
  float spec = pow(max(dot(reflect(-L, n), vec3(0.0, 0.0, 1.0)), 0.0), mix(6.0, 48.0, uRelief.w)) * uRelief.w * 1.6;
  float lv = luma(c);
  // Metal: the surface reflects the palette along its normal, keeping the picture's brightness.
  vec3 env = pal(0.5 + 0.4 * n.x + 0.3 * n.y) * 1.6 * sqrt(max(lv, 0.0));
  vec3 base = mix(c, env, uMetal);
  // Highlights only where there is a surface (light nearby), so flat black stays black.
  vec3 lit = base * (0.3 + 1.05 * dif) + spec * mix(vec3(1.0), uColC, 0.25) * smoothstep(0.0, 0.08, lv);
  return mix(c, max(lit, 0.0), uRelief.x);
}
`;

/** Uniform values from the (reaction-adjusted) tone params. */
export function reliefUniforms(get: (k: string) => number): { v: [number, number, number, number]; metal: number } {
  return { v: [get('relief'), get('bump'), get('light') * Math.PI * 2, get('gloss')], metal: get('metal') };
}
