// Physarum (slime mould) emission: the gene side (schema, cost, budget fitting). No GL here, so the
// genome operators and the tests can use it; the GPU simulation lives in physarumGpu.ts.
//
// Many agents walk a trail map. Each one senses the trail ahead-left, ahead and ahead-right (sensor
// angle sa, distance sd), turns toward the strongest (turn), steps forward (step) and deposits
// (deposit); the trail diffuses (diffuse) and fades (decay) every frame, so the walkers grow vein
// networks. The body seeds them twice over: its light in the feedback feeds the trail (feed), and a
// share of the agents is re-born at its copies (birth). The trail is laid into the body's feedback in
// the palette colours, so the carrier and the space chain act on it like on any other light.

import type { ParamSpec, Schema } from '../genome';

const P = (min: number, max: number, def: number, extra: Partial<ParamSpec> = {}): ParamSpec => ({ min, max, def, ...extra });

export const SLIME_SCHEMA: Schema = {
  // Agents at 1280x720 trail resolution (scaled with the trail area on smaller stages).
  count: P(16384, 1048576, 262144, { log: true, int: true }),
  // Sensor angle (radians either side of the heading) and distance (scene units, screen height 1).
  sa: P(0.1, 1.4, 0.45),
  sd: P(0.003, 0.05, 0.015),
  // Turn per step (radians) and step length (scene units per frame at 60 fps).
  turn: P(0.05, 1.2, 0.4),
  step: P(0.0005, 0.006, 0.0015),
  // Trail laid per agent per frame; trail kept per frame (60 fps); blur mixed in per frame.
  deposit: P(0.05, 1, 0.3),
  decay: P(0.8, 0.995, 0.93),
  diffuse: P(0, 1, 0.6),
  // How visible the body itself stays (it still seeds the network when hidden).
  body: P(0, 1, 1),
  // Light drawn into the feedback above the network (the body, the carried picture) feeds the trail, so
  // shapes seed networks; birth: agents re-born at the body's copies, fraction of all agents per second.
  feed: P(0, 1, 0.5),
  birth: P(0, 0.5, 0.05),
};

/** Reference trail area the agent count is expressed at. */
export const SLIME_REF_AREA = 1280 * 720;
/** Largest trail map height (the trail is at most 720 rows; small stages use their own height). */
export const SLIME_MAX_H = 720;

/** Estimated GPU ms at 1440p: the agent update and deposit scale with the count, the trail passes are fixed. */
export function slimeCost(count: number): number {
  return 0.15 + (count / 262144) * 0.8;
}

/**
 * Display scale of the trail from the gene's own settings: the trail's steady level grows with the agent
 * density and the deposit and falls with the fade, so dividing them out keeps a network equally bright
 * whatever its settings (the material gain sets the brightness; reactions on the deposit still flash).
 */
export function slimeDisplayScale(p: Record<string, number>): number {
  return (0.3 * (262144 / p.count) * ((1 - p.decay) / 0.07)) / (p.deposit / 0.3);
}

/** Brightness of the laid trail per unit of material gain. */
export const SLIME_GAIN = 1.5;

/** Agents a stage of this trail area runs for a gene count (never below a few thousand). */
export function slimeAgents(count: number, trailArea: number): number {
  return Math.max(4096, Math.round(count * Math.min(1, trailArea / SLIME_REF_AREA)));
}
