// Water ripple carrier gene (after Winamp AVS's Water and Water Bump components).
//
// A coarse height field runs the classic two-buffer water recurrence (each cell becomes half the
// sum of its four neighbours minus its own previous height, slightly damped), so every drop spreads
// as rings that bounce and interfere. Drops fall on beats at random spots (bigger with the bass) and
// a large one lands in the middle on each drop section. The feedback pass reads the carried picture
// through the height field's slope (refraction), so the rings bend whatever light is travelling,
// and a faint glint on the slopes facing the light shows them even over black.
//
// Carrier params: water (0 = off) is the refraction strength, wsize the drop radius; the damping
// follows the carrier's half-life a little (long trails, long-lived ripples).

export const WATER_PARAMS = {
  water: { min: 0, max: 1, def: 0 },
  wsize: { min: 0.01, max: 0.08, def: 0.03 },
};
/** Estimated ms at 1440p: the coarse simulation plus four extra taps in the feedback pass. */
export const WATER_COST = 0.12;
export const MAX_DROPS = 4;

/**
 * Feedback-pass GLSL: waterSlope(uv) is the height field's gradient. Uniforms: uWater (the height
 * texture), uWaterAmt (0 = off), uWaterTexel.
 */
export const WATER_GLSL = /* glsl */ `
uniform sampler2D uWater;
uniform float uWaterAmt;
uniform vec2 uWaterTexel;
vec2 waterSlope(vec2 uv) {
  float l = texture(uWater, uv - vec2(uWaterTexel.x, 0.0)).r;
  float r = texture(uWater, uv + vec2(uWaterTexel.x, 0.0)).r;
  float b = texture(uWater, uv - vec2(0.0, uWaterTexel.y)).r;
  float t = texture(uWater, uv + vec2(0.0, uWaterTexel.y)).r;
  return vec2(r - l, t - b);
}
`;
