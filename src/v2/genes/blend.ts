// Blend-mode gene (after Winamp AVS's Effect List output modes and its Render Mode setting).
//
// In AVS every effect list (and every line/dot render) says how it goes onto the frame below it:
// additive, maximum, subtractive, XOR, 50/50, every other line... Here each body carries the same
// choice as a material parameter, `blend`, applied to what the body adds this frame:
//
//   0 add        light adds (the default, how every body drew before)
//   1 maximum    the brighter of the body and what is already there (no pile-up)
//   2 subtract   the body cuts dark into the light below it
//   3 xor        difference: where the body crosses light it inverts it
//   4 screen     soft add that never burns out
//   5 interlace  added on every other scanline only
//
// Distance-field and field bodies blend in their shader (the body's call is wrapped, so the mode is
// structural); curve bodies set the GL blend state per draw; flames and particles always add.

import type { Schema } from '../genome';

export const BLEND_MODES = ['add', 'maximum', 'subtract', 'xor', 'screen', 'interlace'] as const;
export const BLEND_SCHEMA: Schema = { blend: { min: 0, max: 5, def: 0, choices: [0, 1, 2, 3, 4, 5] } };
/** Extra ms at 1440p for a non-additive body (a few ALU per pixel, or a GL state change). */
export const BLEND_COST = 0.02;

/** GLSL: blendInto(before, after, mode), after = what the body's call returned. */
export const BLEND_GLSL = /* glsl */ `
vec3 blendInto(vec3 before, vec3 after, int mode) {
  vec3 a = after - before;
  if (mode == 1) return max(before, a);
  if (mode == 2) return max(before - max(a, 0.0), 0.0);
  if (mode == 3) return abs(before - a);
  if (mode == 4) return before + max(a, 0.0) * (1.0 - clamp(before, 0.0, 1.0));
  if (mode == 5) return before + a * step(0.5, fract(gl_FragCoord.y * 0.5));
  return after;
}
`;

/** Wraps a body's call so it blends by `mode` (additive calls stay as they are). */
export function blendCall(call: string, mode: number): string {
  if (!mode || !call) return call;
  const m = /^(\s*)c = (body_\w+\([^)]*\));\n$/.exec(call);
  return m ? `${m[1]}c = blendInto(c, ${m[2]}, ${mode});\n` : call;
}

/** Sets the GL blend state for a curve draw in this mode; returns true when the interlace mask is needed. */
export function setCurveBlend(gl: WebGL2RenderingContext, mode: number): boolean {
  gl.blendEquation(mode === 1 ? gl.MAX : mode === 2 ? gl.FUNC_REVERSE_SUBTRACT : gl.FUNC_ADD);
  if (mode === 3) gl.blendFunc(gl.ONE_MINUS_DST_COLOR, gl.ONE_MINUS_SRC_COLOR);
  else if (mode === 4) gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_COLOR);
  else gl.blendFunc(gl.ONE, gl.ONE);
  return mode === 5;
}

/** Back to plain additive blending after a curve draw. */
export function resetCurveBlend(gl: WebGL2RenderingContext): void {
  gl.blendEquation(gl.FUNC_ADD);
  gl.blendFunc(gl.ONE, gl.ONE);
}
