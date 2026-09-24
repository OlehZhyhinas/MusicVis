// Tunnel: a fold op that wraps the picture onto the wall of a tunnel seen head-on (the classic
// cylindrical texture mapping of MilkDrop tunnel presets). Screen radius becomes depth (depth /
// radius), angle goes round the wall; the picture repeats mirrored along the depth so it never
// shows a seam, and scrolls toward the viewer at `speed` (surging on the beat). `twist` spirals the
// wall, `sides` > 0 makes the cross-section a polygon, `rep` repeats the picture round the wall,
// `fog` darkens the far end and `lock` turns the tunnel bar-locked. Usually a view-stage fold (the
// carried picture is shown as a tunnel); in the warp it feeds the tunnel back into itself.

import type { ParamSpec, Schema } from '../genome';

const P = (min: number, max: number, def: number): ParamSpec => ({ min, max, def });
const I = (min: number, max: number, def: number): ParamSpec => ({ min, max, def, int: true });
const C = (choices: number[], def: number): ParamSpec => ({ min: Math.min(...choices), max: Math.max(...choices), def, choices });
const LOCKS = [-0.25, -0.125, -0.0625, 0, 0.0625, 0.125, 0.25];

export const TUNNEL_SCHEMA: Schema = {
  depth: P(0.05, 0.6, 0.2),
  speed: P(-1, 2, 0.5),
  twist: P(-1.5, 1.5, 0),
  sides: C([0, 3, 4, 5, 6, 8], 0),
  rep: I(1, 4, 1),
  fog: P(0, 1, 0.6),
  lock: C(LOCKS, 0),
};

/** OA = (depth, scroll, twist, spin), OB = (sides, rep, fog, -). */
export const TUNNEL_GLSL = `{ float r = length(p); float a = atan(p.y, p.x) + OA.w; if (OB.x > 2.5) { float sg = TAU / OB.x; r *= cos(mod(a, sg) - sg * 0.5); } float z = OA.x / max(r, 1e-3); a += OA.z * z; float u = abs(mod(a * OB.y / PI + 1.0, 2.0) - 1.0); float v = abs(mod(z + OA.y, 2.0) - 1.0); vMul *= mix(1.0, smoothstep(0.0, 0.25, r), OB.z); p = vec2((u - 0.5) * uAspect * 0.8, (v - 0.5) * 0.9); }`;

/** Per-frame slots; the scroll accumulates in the slot memory under `key`. */
export function packTunnel(
  a: Float32Array, b: Float32Array, j: number, P: (k: string) => number, p: Record<string, number>,
  mem: Record<string, number>, key: string, sdt: number, speed: number, beatPulse: number, spin: number,
): void {
  const v = P('speed');
  mem[key] = ((mem[key] ?? 0) + sdt * v * speed * (1 + 1.5 * beatPulse)) % 4096;
  a[j] = P('depth'); a[j + 1] = mem[key]; a[j + 2] = P('twist'); a[j + 3] = spin * p.lock;
  b[j] = p.sides; b[j + 1] = p.rep; b[j + 2] = P('fog');
}
