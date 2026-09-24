// Stem ecosystem emission: the gene side (schema, cost, budget, population dynamics). No GL here, so the
// genome operators and the tests can use it; the GPU simulation lives in ecosystemGpu.ts.
//
// Every instrument is a species of GPU agent sharing one world:
//   0 drums  = predators: fast, chase the plankton scent, dart and strike on drum hits;
//   1 bass   = grazers: slow and heavy, eat the growth field and leave trails the others follow;
//   2 vocals = pollinators: follow the grazer trails and make the growth field bloom where they pass;
//   3 other  = plankton: drift on a slow current, flee the predators, get eaten and multiply.
// The mix drives the populations: a species grows while its stem is present and starves while it is
// silent; predators feed on plankton (a drop is a feast), blooms feed the grazers and the plankton. The
// populations are a small ODE on the CPU (below); on the GPU an agent is alive while its fixed rank is
// under its species' population, and a newborn copies the position of a live parent, so colonies grow
// out of the clusters already there.

import type { ParamSpec, Schema } from '../genome';

const P = (min: number, max: number, def: number, extra: Partial<ParamSpec> = {}): ParamSpec => ({ min, max, def, ...extra });

export const ECO_SPECIES = ['predators', 'grazers', 'pollinators', 'plankton'] as const;

export const ECO_SCHEMA: Schema = {
  // Agents at 1280x720 (scaled with the stage area on smaller stages), shared by the species.
  count: P(4096, 262144, 32768, { log: true, int: true }),
  // Roster: each species' share of the agent pool (drums, bass, vocals, other); 0 leaves it out.
  wD: P(0, 1, 0.5),
  wB: P(0, 1, 0.5),
  wV: P(0, 1, 0.5),
  wO: P(0, 1, 0.8),
  // Look set: 0 creatures (streak predators, blob grazers, ring pollinators, dot plankton), 1 points,
  // 2 comets (all streaks), 3 sigils (diamonds, rings, stars, dots).
  glyph: { min: 0, max: 3, def: 0, choices: [0, 1, 2, 3] },
  // Sprite size (pixels at 1080p, scaled per species) and streak length along the heading.
  size: P(1, 8, 3),
  trail: P(0, 1, 0.5),
  // Interactions: plankton eaten by predators; flora grown by pollinators; flora eaten by grazers.
  predation: P(0, 1, 0.5),
  bloom: P(0, 1, 0.5),
  graze: P(0, 1, 0.5),
  // Population growth while a stem plays and starvation while it is silent (per second).
  growth: P(0.1, 2, 0.6),
  starve: P(0.05, 2, 0.4),
  // Growth field kept per frame (60 fps): low = blooms wilt fast, high = a meadow builds up.
  decay: P(0.9, 0.999, 0.985),
  // Movement speed of every species; flora visibility; species-to-palette rotation.
  speed: P(0.2, 2, 1),
  field: P(0, 1, 0.5),
  hues: { min: 0, max: 3, def: 0, choices: [0, 1, 2, 3] },
  // How visible the body itself stays.
  body: P(0, 1, 0.3),
};

/** Reference stage area the agent count is expressed at. */
export const ECO_REF_AREA = 1280 * 720;
/** Largest growth field height (the field runs at a third of the stage, at most this many rows). */
export const ECO_FIELD_MAX_H = 360;
/** Sprite size multiplier per species (grazers are big, plankton small). */
export const ECO_SIZE = [1.1, 2.2, 1.3, 0.7];
/** Glyph per species for each look set: 0 dot, 1 streak, 2 blob, 3 ring, 4 diamond, 5 star. */
export const ECO_GLYPHS: number[][] = [
  [1, 2, 3, 0],
  [0, 0, 0, 0],
  [1, 1, 1, 1],
  [4, 3, 5, 0],
];

/** Estimated GPU ms at 1440p: the agent update, field deposit and sprite fill scale with the count. */
export function ecoCost(p: { count: number; size: number }): number {
  return 0.35 + (p.count / 65536) * (0.35 + 0.012 * p.size * p.size);
}

/** Agents a stage of this area runs for a gene count (never below a couple of thousand). */
export function ecoAgents(count: number, area: number): number {
  return Math.max(2048, Math.round(count * Math.min(1, area / ECO_REF_AREA)));
}

/** Cumulative share of the agent index range per species (last is 1); a zero roster falls back to plankton. */
export function ecoCuts(p: Record<string, number>): [number, number, number, number] {
  const w = [p.wD, p.wB, p.wV, p.wO].map((x) => Math.max(0, x));
  const sum = w[0] + w[1] + w[2] + w[3];
  if (sum < 1e-6) return [0, 0, 0, 1];
  let acc = 0;
  return w.map((x) => (acc += x / sum)) as [number, number, number, number];
}

/** The ecosystem's CPU-side state: population of each species (share of its slots alive) and the flora level. */
export interface EcoState {
  pop: [number, number, number, number];
  flora: number;
}

/** Every species starts as a small colony; survivors never drop below this so a silent stem can come back. */
export const ECO_FLOOR = 0.03;

export function ecoInitial(): EcoState {
  return { pop: [0.2, 0.25, 0.2, 0.35], flora: 0.2 };
}

/**
 * Advances the populations by dt seconds. pres: each stem's presence (0..1, drums, bass, vocals, other);
 * drop: the drop envelope (a feast for the predators). Logistic growth while present, starvation while
 * silent, plus the couplings: predators feed on plankton and plankton are eaten, pollinators grow the
 * flora, grazers eat it and it feeds the grazers and plankton.
 */
export function ecoStep(s: EcoState, pres: ArrayLike<number>, drop: number, p: Record<string, number>, dt: number): void {
  dt = Math.min(0.1, Math.max(0, dt));
  const [D, B, V, O] = s.pop;
  const F = s.flora;
  const g = p.growth;
  const st = p.starve;
  const pr = (i: number) => Math.min(1, Math.max(0, pres[i] ?? 0));
  const grow = (x: number, i: number, food: number) => g * pr(i) * food * x * (1 - x) - st * (1 - pr(i)) * x;
  const feast = 1 + 2.5 * Math.max(0, drop);
  const dD = grow(D, 0, (0.35 + 1.3 * p.predation * O) * feast);
  const dB = grow(B, 1, 0.5 + 0.9 * F * (0.4 + p.graze));
  const dV = grow(V, 2, 0.7 + 0.3 * F);
  const dO = grow(O, 3, 1 + 0.5 * F) - 0.35 * p.predation * D * O * (0.5 + pr(0)) * (1 + 1.5 * Math.max(0, drop));
  const dF = 0.9 * p.bloom * V * (0.3 + pr(2)) * (1 - F) - 0.7 * p.graze * B * F - 0.04 * F;
  const lim = (x: number) => Math.min(1, Math.max(ECO_FLOOR, x));
  s.pop = [lim(D + dD * dt), lim(B + dB * dt), lim(V + dV * dt), lim(O + dO * dt)];
  s.flora = Math.min(1, Math.max(0, F + dF * dt));
}

/**
 * Display scale of the growth field from the gene's own settings: its steady level grows with the
 * pollinator density and the bloom and falls with the fade, so dividing them out keeps a meadow equally
 * bright whatever its settings (the field parameter and the material gain set the brightness).
 */
export function ecoFieldScale(p: Record<string, number>): number {
  return 0.6 * ((1 - p.decay) / 0.015) / Math.max(0.1, p.bloom / 0.5);
}
