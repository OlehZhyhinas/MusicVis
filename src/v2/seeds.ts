// The seed population: V1's 24 hand-made presets (E01-E24) re-expressed in the
// shared genome vocabulary. These are approximations of the originals built
// from the generic parts, so they can breed with each other.

import {
  CARRIER_SCHEMA, COLOR_SCHEMA, EMITTER_SCHEMAS, OP_SCHEMAS, defaultParams, repair,
  type CarrierKind, type EmitterGene, type EmitterKind, type FlameXformGene, type Genome, type Layer,
  type OpGene, type OpKind, type ReactionGene, type Scheme, type Signal,
} from './genome';

export interface Seed {
  origin: string; // V1 preset id, e.g. 'E07'
  name: string;
  genome: Genome;
}

const hl = (decayPerFrame: number) => Math.log(0.5) / (60 * Math.log(decayPerFrame));

function op(kind: OpKind, p: Record<string, number> = {}, w = 1, stage: 'warp' | 'view' = 'warp'): OpGene {
  return { op: kind, stage, w, p: { ...defaultParams(OP_SCHEMAS[kind]), ...p } };
}
function em(kind: EmitterKind, p: Record<string, number> = {}, layer: Layer = 'fb', xforms?: FlameXformGene[]): EmitterGene {
  const e: EmitterGene = { kind, layer, p: { ...defaultParams(EMITTER_SCHEMAS[kind]), ...p } };
  if (xforms) e.xforms = xforms;
  return e;
}
function xf(x: Partial<FlameXformGene> & Pick<FlameXformGene, 'aff' | 'weight' | 'color' | 'vars'>): FlameXformGene {
  return { spin: 0, bass: 0, drift: [0, 0], pulse: 0, ...x };
}
function rx(src: Signal, g: ReactionGene['g'], i: number, k: string, gain: number): ReactionGene {
  return { src, g, i, k, gain };
}

interface Def {
  origin: string;
  name: string;
  energy: [number, number];
  scheme: Scheme;
  hue: number;
  color?: Record<string, number>;
  carrier: CarrierKind;
  decay?: number;
  car?: Record<string, number>;
  chain?: OpGene[];
  emitters: EmitterGene[];
  reactions?: ReactionGene[];
}

function build(d: Def): Seed {
  const g: Genome = {
    v: 1,
    chain: d.chain ?? [],
    emitters: d.emitters,
    carrier: {
      kind: d.carrier,
      p: { ...defaultParams(CARRIER_SCHEMA), halfLife: d.decay ? hl(d.decay) : 0.5, floor: 1, ...d.car },
    },
    color: {
      scheme: d.scheme,
      p: { ...defaultParams(COLOR_SCHEMA), sat: 1, adapt: 0.6, bloom: 1, vignette: 0.45, hue: d.hue, ...d.color },
    },
    reactions: d.reactions ?? [],
    energy: d.energy,
  };
  return { origin: d.origin, name: d.name, genome: repair(g) };
}

const DEFS: Def[] = [
  {
    origin: 'E01', name: 'Night Skyline', energy: [0.3, 0.8], scheme: 'complementary', hue: 0.05,
    color: { adapt: 0.3, reflect: 1, reflectY: -0.16 }, carrier: 'warp', decay: 0.9995, car: { floor: 0.05 },
    chain: [op('translate', { vx: -0.2 })],
    emitters: [em('edge', { mode: 0, side: 0, base: -0.16, height: 0.3, density: 0.5 })],
  },
  {
    origin: 'E02', name: 'River of Light', energy: [0, 0.45], scheme: 'analogous', hue: 0.55,
    color: { adapt: 0.25, bloom: 1.2 }, carrier: 'warp', decay: 0.996, car: { floor: 0.4 },
    chain: [op('translate', { vx: -0.12 }), op('ripple', { amp: 0.00025, freq: 2, speed: 0.2, radial: 0 })],
    emitters: [em('edge', { mode: 1, side: 0, base: 0 })],
  },
  {
    origin: 'E03', name: 'Rain Curtains', energy: [0.4, 0.95], scheme: 'analogous', hue: 0.5,
    color: { bloom: 1.1 }, carrier: 'warp', decay: 0.962,
    chain: [op('translate', { vy: -0.4 })],
    emitters: [em('edge', { mode: 2, side: 1, density: 0.6 })],
    reactions: [rx('bass', 'op', 0, 'vy', -0.15)],
  },
  {
    origin: 'E04', name: 'Rising Smoke', energy: [0.1, 0.6], scheme: 'split', hue: 0.02,
    color: { adapt: 0.4 }, carrier: 'warp', decay: 0.9935, car: { blur: 0.3 },
    chain: [op('noise', { amp: 0.0014, scale: 1.8, speed: 0.3 }), op('translate', { vy: 0.29 })],
    emitters: [
      em('ink', { count: 4, orbit: 0, row: 1, size: 0.04, wander: 0.05, force: 0, gain: 0.15 }),
      em('particles', { spawn: 5, count: 1536, size: 5, speed: 0.6, curl: 0.1, life: 0.55, lift: 0.12, drag: 1.5, gain: 0.7 }, 'top'),
    ],
    reactions: [rx('beat', 'op', 1, 'vy', 0.48), rx('bass', 'op', 1, 'vy', 0.36)],
  },
  {
    origin: 'E05', name: 'Wandering Vortex', energy: [0.5, 1], scheme: 'triad', hue: 0.7,
    color: { bloom: 1.1 }, carrier: 'warp', decay: 0.986,
    chain: [
      op('zoom', { rate: -0.004, wander: 0.35 }),
      op('swirl', { amt: 0.01, k: 6, wander: 0.35 }),
      op('rotate', { lock: 0.25 }),
    ],
    emitters: [em('ink', { count: 2, orbit: 1, radius: 0.3, size: 0.018, gain: 0.9, force: 0 })],
    reactions: [rx('bass', 'op', 0, 'rate', -0.2), rx('beat', 'op', 0, 'rate', -0.33)],
  },
  {
    origin: 'E06', name: 'Ink Garden', energy: [0.15, 0.75], scheme: 'triad', hue: 0.15,
    color: { adapt: 0.35 }, carrier: 'fluid', decay: 0.993, car: { floor: 1.5, amount: 1, vort: 28, fnoise: 0.35 },
    emitters: [em('ink', { count: 4, orbit: 0, size: 0.02, wander: 0.1, force: 1 })],
  },
  {
    origin: 'E07', name: 'Oscilloscope', energy: [0, 0.4], scheme: 'mono', hue: 0.45,
    color: { sat: 0.7, adapt: 0.15, bloom: 0.9, vignette: 0.6 }, carrier: 'warp', decay: 0.955, car: { blur: 0.2 },
    chain: [op('zoom', { rate: -0.0008 }), op('translate', { vy: 0.084 })],
    emitters: [em('wave', { shape: 0, amp: 0.24, y: -0.1, thick: 1.5, gain: 1.1 })],
  },
  {
    origin: 'E08', name: 'Contour Plasma', energy: [0.3, 0.85], scheme: 'split', hue: 0.6,
    color: { bloom: 0.9, adapt: 0.4 }, carrier: 'none',
    emitters: [em('plasma', { scale: 1.7, warp: 3, bands: 8, lines: 1, speed: 0.15 }, 'top')],
  },
  {
    origin: 'E09', name: 'Rorschach', energy: [0.2, 0.7], scheme: 'complementary', hue: 0.85,
    color: { adapt: 0.35 }, carrier: 'warp', decay: 0.992, car: { blur: 0.25 },
    chain: [
      op('noise', { amp: 0.0016, scale: 2.5, speed: 0.35 }),
      op('zoom', { rate: 0.0012 }),
      op('mirror', { axis: 0 }),
      op('mirror', { axis: 0 }, 1, 'view'),
    ],
    emitters: [em('ink', { count: 3, orbit: 0, size: 0.03, wander: 0.08, force: 0, gain: 0.25 })],
  },
  {
    origin: 'E10', name: 'Hex Keys', energy: [0.4, 0.9], scheme: 'mono', hue: 0.6,
    color: { adapt: 0.35 }, carrier: 'warp', decay: 0.93,
    emitters: [em('tiles', { shape: 0, scale: 5, lock: 0.0625, lit: 0.5, edges: 1 })],
  },
  {
    origin: 'E11', name: 'Sunrise Arc', energy: [0.35, 0.85], scheme: 'analogous', hue: 0.02,
    color: { adapt: 0.35, reflect: 1, reflectY: -0.42 }, carrier: 'warp', decay: 0.925,
    chain: [op('zoom', { rate: 0.006, cy: -0.4 })],
    emitters: [em('spectrum', { mode: 1, y: -0.42, bins: 40, radius: 0.2, len: 0.3, fill: 0.6 })],
    reactions: [rx('beat', 'op', 0, 'rate', 0.3)],
  },
  {
    origin: 'E12', name: 'Warp Speed', energy: [0.55, 1], scheme: 'analogous', hue: 0.58,
    color: { sat: 0.35, bloom: 1.1 }, carrier: 'warp', decay: 0.8,
    chain: [op('rotate', { lock: 0.0625 }), op('zoom', { rate: 0.012 })],
    emitters: [em('particles', { spawn: 4, count: 16384, size: 2.6, gain: 1.15, speed: 0.1, curl: 0, life: 0.2, zoomFlow: 1.1, drag: 4, spread: 0.12 })],
  },
  {
    origin: 'E13', name: 'Constellations', energy: [0, 0.45], scheme: 'mono', hue: 0.6,
    color: { adapt: 0.15, vignette: 0.55 }, carrier: 'warp', decay: 0.9,
    emitters: [em('stars', { density: 0.5, scale: 6.5, links: 1, twinkle: 1, drift: 0.005 })],
  },
  {
    origin: 'E14', name: 'Polyhedra', energy: [0.35, 0.9], scheme: 'triad', hue: 0.5,
    color: { adapt: 0.3 }, carrier: 'warp', decay: 0.88,
    chain: [op('rotate', { lock: -0.0625 }), op('zoom', { rate: -0.004 })],
    emitters: [em('wire', { solid: 5, scale: 0.2, tilt: 0.45, lock: 0.25, inner: 1, thick: 1.3 })],
  },
  {
    origin: 'E15', name: 'Liquid Chrome', energy: [0.3, 0.8], scheme: 'complementary', hue: 0.55,
    color: { sat: 0.6, adapt: 0.4 }, carrier: 'none',
    emitters: [em('blobs', { count: 6, size: 0.06, spread: 0.55, speed: 0.122, chrome: 1 }, 'top')],
  },
  {
    origin: 'E16', name: 'Aurora', energy: [0.1, 0.6], scheme: 'analogous', hue: 0.35,
    color: { adapt: 0.3 }, carrier: 'warp', decay: 0.965, car: { blur: 0.1 },
    chain: [op('noise', { amp: 0.0006, scale: 1.5, speed: 0.5 }), op('translate', { vy: 0.03 })],
    emitters: [em('aurora', { y: 0, fall: 5, rays: 24, wav: 1.4 })],
  },
  {
    origin: 'E17', name: 'Hyperspace', energy: [0.6, 1], scheme: 'complementary', hue: 0.8,
    color: { bloom: 1.2 }, carrier: 'warp', decay: 0.94,
    chain: [op('zoom', { rate: 0.02, radial: 1 }), op('rotate', { lock: 0.25, alt: 1 })],
    emitters: [em('wire', { solid: 4, sides: 6, scale: 0.12, lock: 0.25, thick: 1.8, inner: 0, gain: 1.6 })],
    reactions: [rx('bass', 'op', 0, 'rate', 0.5), rx('beat', 'op', 0, 'rate', 0.8), rx('build', 'op', 0, 'rate', 0.5)],
  },
  {
    origin: 'E18', name: 'Mandala', energy: [0.5, 1], scheme: 'triad', hue: 0.1,
    color: { bloom: 1.1 }, carrier: 'warp', decay: 0.94,
    chain: [op('rotate', { lock: 0.125 }), op('zoom', { rate: 0.006 }), op('kaleido', { n: 8, lock: 0.0625 }, 1, 'view')],
    // An off-centre ring folded eight ways gives the petals.
    emitters: [em('wave', { shape: 1, x: 0.14, y: 0.05, radius: 0.2, amp: 0.29, thick: 1.8, gain: 0.8 })],
    reactions: [rx('beat', 'em', 0, 'gain', 0.5)],
  },
  {
    origin: 'E19', name: 'Harmonograph', energy: [0.05, 0.5], scheme: 'analogous', hue: 0.1,
    color: { adapt: 0.15, vignette: 0.55 }, carrier: 'warp', decay: 0.86,
    emitters: [em('wave', { shape: 3, radius: 0.36, amp: 0.2, ra: 2, rb: 3, thick: 1.2, gain: 0.6 })],
  },
  {
    origin: 'E20', name: 'Moonrise', energy: [0, 0.45], scheme: 'analogous', hue: 0.62,
    color: { sat: 0.5, adapt: 0.15, vignette: 0.5, reflect: 1, reflectY: -0.14 }, carrier: 'none',
    emitters: [em('orb', { x: 0.27, y: 0.1, radius: 0.065, halo: 0.5, stripes: 0, craters: 1 }, 'top')],
  },
  {
    origin: 'E21', name: 'Retro Grid', energy: [0.5, 1], scheme: 'split', hue: 0.85,
    color: { bloom: 1.2, adapt: 0.35 }, carrier: 'none',
    emitters: [
      em('horizon', { y: -0.06, speed: 0.5, density: 1.5, peaks: 1 }, 'top'),
      em('orb', { x: 0, y: 0.09, radius: 0.19, halo: 1, stripes: 1, craters: 0 }, 'top'),
    ],
  },
  {
    origin: 'E22', name: 'Silk Flame', energy: [0.05, 0.5], scheme: 'analogous', hue: 0.55,
    color: { adapt: 0.3, bloom: 0.9, tonemap: 1 }, carrier: 'warp', decay: 0.955,
    emitters: [em('flame', { count: 262144, zoom: 0.2, camSpin: 0.125, rounds: 2, flow: 1, breathe: 0.12 }, 'fb', [
      xf({ aff: [0.7, -0.3, 0.3, 0.7, 0.2, 0.1], weight: 1, color: 0, vars: { julia: 0.7, linear: 0.3 }, alt: { spiral: 0.6, heart: 0.4 }, spin: 0.125, drift: [0.25, 0.2] }),
      xf({ aff: [0.5, 0, 0, 0.5, -0.5, 0.2], weight: 0.6, color: 0.5, vars: { spherical: 0.5, swirl: 0.3 }, alt: { disc: 0.6, horseshoe: 0.4 }, bass: 0.2, drift: [0.3, 0.25] }),
      xf({ aff: [-0.4, 0.3, -0.3, -0.4, 0.3, -0.4], weight: 0.4, color: 0.9, vars: { sinusoidal: 1 }, alt: { handkerchief: 0.7, polar: 0.3 }, spin: -0.0625, pulse: 0.08, drift: [0.2, 0.3] }),
    ])],
  },
  {
    origin: 'E23', name: 'Ember Flame', energy: [0.55, 1], scheme: 'split', hue: 0.02,
    color: { adapt: 0.35, bloom: 1.2, tonemap: 1 }, carrier: 'warp', decay: 0.85,
    emitters: [em('flame', { count: 524288, zoom: 0.3, camSpin: -0.125, rounds: 2 }, 'fb', [
      xf({ aff: [0.8, 0.2, -0.2, 0.8, 0, 0.1], weight: 1, color: 0.1, vars: { swirl: 0.6, linear: 0.4 }, alt: { spiral: 0.5, heart: 0.3 }, spin: 0.25, bass: 0.25 }),
      xf({ aff: [0.4, -0.35, 0.35, 0.4, 0.6, 0], weight: 0.5, color: 0.6, vars: { horseshoe: 1 }, alt: { disc: 1 } }),
      xf({ aff: [0.5, 0, 0, -0.5, -0.4, -0.3], weight: 0.4, color: 0.95, vars: { handkerchief: 0.8 }, alt: { polar: 0.8 }, spin: -0.125 }),
    ])],
  },
  {
    origin: 'E24', name: 'Spiral Nebula', energy: [0.3, 0.75], scheme: 'triad', hue: 0.6,
    color: { adapt: 0.3, tonemap: 1 }, carrier: 'warp', decay: 0.88,
    emitters: [em('flame', { count: 262144, zoom: 0.22, camSpin: 0.0625, rounds: 2 }, 'fb', [
      xf({ aff: [0.6, -0.5, 0.5, 0.6, 0, 0], weight: 1, color: 0, vars: { spiral: 0.3, linear: 0.7 }, spin: 0.0625, bass: 0.1 }),
      xf({ aff: [0.3, 0, 0, 0.3, 0.7, 0], weight: 0.5, color: 0.5, vars: { spherical: 1 } }),
      xf({ aff: [0.3, 0, 0, 0.3, -0.35, 0.6], weight: 0.3, color: 0.85, vars: { disc: 0.6, linear: 0.4 } }),
    ])],
  },
];

export const SEEDS: Seed[] = DEFS.map(build);
