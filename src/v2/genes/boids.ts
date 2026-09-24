// Boids (flocking) emission: the gene side (schema, cost). No GL here; the GPU simulation lives in
// boidsGpu.ts.
//
// Thousands of birds fly with a velocity each. Instead of comparing every pair, every bird is splatted
// into a coarse neighbourhood grid (count and summed velocity per cell, one cell per `radius`); each
// bird then reads the grid where it is: it turns toward the local average heading (align), toward
// denser air while the crowd is thin (cohere) and away from it once crowded (separate), wanders a
// little (wander) and is pulled toward the body's copies (home), all at about `speed`. The birds are
// drawn as soft points into the body's feedback, so the carrier turns their paths into trails.

import type { ParamSpec, Schema } from '../genome';

const P = (min: number, max: number, def: number, extra: Partial<ParamSpec> = {}): ParamSpec => ({ min, max, def, ...extra });
const C = (choices: number[], def: number): ParamSpec => ({ min: Math.min(...choices), max: Math.max(...choices), def, choices });

export const FLOCK_SCHEMA: Schema = {
  count: P(4096, 262144, 32768, { log: true, int: true }),
  // Cruising speed, screen heights per second; neighbourhood size, screen heights.
  speed: P(0.05, 0.6, 0.2),
  radius: P(0.015, 0.08, 0.035),
  align: P(0, 1, 0.6),
  cohere: P(0, 1, 0.4),
  separate: P(0, 1, 0.5),
  wander: P(0, 1, 0.2),
  // Pull toward the body's copies (the flock circles its lights).
  home: P(0, 1, 0.3),
  // Bird size in pixels at 1080p; how visible the body itself stays.
  size: P(1, 5, 2),
  body: P(0, 1, 1),
  // On a drop: 0 nothing, 1 the flock bursts outward from where each bird is, 2 every bird restarts at the body.
  onDrop: C([0, 1, 2], 1),
};

/**
 * Estimated GPU ms at 1440p: the grid passes are tiny; the update and above all the blended points
 * scale with the count (timer query: ~1 ms per 65k birds of ~3 px at 800 px, scaled up for 1440p).
 */
export function flockCost(count: number): number {
  return 0.1 + (count / 65536) * 1.2;
}

/** Brightness of a bird per unit of material gain. */
export const FLOCK_GAIN = 0.9;
