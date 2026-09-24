// The seed population: the original 24 hand-made presets (E01-E24) re-expressed in the
// sub-gene vocabulary, so breeding can mix their ideas (a snake's walk with a
// polyhedron's shape, the moon's arms on a flame, ink dye from a star grid).
//
// Decomposed: snake (dot x walker x fill x cover, turning on hits), ink (dot x
// row / orbit / instrument stations x glow x trail or dye), moon and sun (dot x
// point x textured, arms deformation, bob), stars (dot x grid x glow), hex keys
// (hexagon x hex grid x fill), polyhedra (wireframe solid x point x line,
// bar-locked 3D spin), chrome blobs (dot x float x chrome, melting copies),
// sunrise arc (spectrum bars x point x fill), waveforms (curve x point x line),
// sparks (the particle systems are an emission).
// Kept as larger chunk shapes (hand-written tricks that lose fidelity if split):
// plasma contours (E08), aurora (E16), ray-marched terrain (E21), the edge
// strips (E01 skyline with its view-stage spectrum stretch, E03 rain), and the
// fractal flames (E22-E24, which keep their transforms).
//
// M01-M10 re-create ten of the best-known classic MilkDrop presets in the same
// vocabulary (feedback warp as the chain and carrier, waveforms as curves,
// custom shapes as bodies, the blur-sharpen warp shader and the outer border
// as carrier genes). They are re-creations of each preset's look and music
// response, credited to the original authors in the name.
//
// SEED_VERSION goes up whenever the seeds are re-encoded or new seeds are
// added; stored populations and imported files with an older version get the
// new seed genomes and any missing seeds (votes, views and ids are kept, and
// bred children are never touched).

import {
  CARRIER_SCHEMA, TONE_SCHEMA, PALETTE_SCHEMAS, MAPPING_SCHEMAS, DEFORM_SCHEMAS, EMIT_SCHEMAS, FEEL_SCHEMAS, MATERIAL_SCHEMAS, MOTION_SCHEMAS, OP_SCHEMAS, PLACE_SCHEMAS,
  SHAPE_SCHEMAS, defaultParams, repair,
  type BodyGene, type CarrierKind, type DeformKind, type EmitKind, type FeelKind, type MappingKind, type FlameXformGene, type Genome, type MaterialKind,
  type MotionKind, type OpGene, type OpKind, type PlaceKind, type ReactionGene, type Scheme, type ShapeKind, type Signal,
} from './genome';
import { CHOREO_SCHEMA } from './genes/choreo';

export const SEED_VERSION = 28;

export interface Seed {
  origin: string; // source preset id, e.g. 'E07'
  name: string;
  genome: Genome;
}

const hl = (decayPerFrame: number) => Math.log(0.5) / (60 * Math.log(decayPerFrame));

function op(kind: OpKind, p: Record<string, number> = {}, w = 1, stage: 'warp' | 'view' = 'warp'): OpGene {
  return { op: kind, stage, w, p: { ...defaultParams(OP_SCHEMAS[kind]), ...p } };
}
function xf(x: Partial<FlameXformGene> & Pick<FlameXformGene, 'aff' | 'weight' | 'color' | 'vars'>): FlameXformGene {
  return { spin: 0, bass: 0, drift: [0, 0], pulse: 0, ...x };
}
function rx(src: Signal, g: ReactionGene['g'], i: number, k: string, gain: number, curve: Partial<ReactionGene> = {}): ReactionGene {
  return { src, g, i, k, gain, atk: 0.005, rel: 0.005, thr: 0, q: 0, div: 1, ...curve };
}

type P = Record<string, number>;
interface BodyDef {
  shape: [ShapeKind, P?, FlameXformGene[]?];
  place?: [PlaceKind, P?];
  motion?: [MotionKind, P?];
  deform?: [DeformKind, P?];
  material: [MaterialKind, P?];
  emit?: [EmitKind, P?];
  feel?: [FeelKind, P?];
  /** Colour mapping; by default what the placement implies (grid: pitch, copies: instrument, one shape: fixed). */
  color?: [MappingKind, P?];
}
function body(d: BodyDef): BodyGene {
  const [sk, sp = {}, xforms] = d.shape;
  const [pk, pp = {}] = d.place ?? ['point'];
  const [mk, mp = {}] = d.motion ?? ['none'];
  const [dk, dp = {}] = d.deform ?? ['none'];
  const [ak, ap = {}] = d.material;
  const [ek, ep = {}] = d.emit ?? ['trail'];
  const [fk, fp = {}] = d.feel ?? ['flow'];
  const implied: MappingKind = pk === 'grid' ? 'pitch' : ['orbit', 'stations', 'row', 'outline', 'ring', 'walker'].includes(pk) ? 'instrument' : 'fixed';
  const [ck, cp = {}] = d.color ?? [implied];
  const b: BodyGene = {
    shape: { kind: sk, p: { ...defaultParams(SHAPE_SCHEMAS[sk]), ...sp } },
    place: { kind: pk, p: { ...defaultParams(PLACE_SCHEMAS[pk]), ...pp } },
    motion: { kind: mk, p: { ...defaultParams(MOTION_SCHEMAS[mk]), ...mp } },
    deform: { kind: dk, p: { ...defaultParams(DEFORM_SCHEMAS[dk]), ...dp } },
    material: { kind: ak, p: { ...defaultParams(MATERIAL_SCHEMAS[ak]), ...ap } },
    emit: { kind: ek, p: { ...defaultParams(EMIT_SCHEMAS[ek]), ...ep } },
    feel: { kind: fk, p: { ...defaultParams(FEEL_SCHEMAS[fk]), ...fp } },
    color: { kind: ck, p: { ...defaultParams(MAPPING_SCHEMAS[ck]), amount: 1, ...cp } },
  };
  if (xforms) b.shape.xforms = xforms;
  return b;
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
  bodies: BodyGene[];
  reactions?: ReactionGene[];
  /** Choreography over the song timeline (src/v2/genes/choreo.ts); omitted = none. */
  choreo?: Record<string, number>;
}

function build(d: Def): Seed {
  const g: Genome = {
    v: 5,
    chain: d.chain ?? [],
    bodies: d.bodies,
    carrier: {
      kind: d.carrier,
      p: { ...defaultParams(CARRIER_SCHEMA), halfLife: d.decay ? hl(d.decay) : 0.5, floor: 1, ...d.car },
    },
    palette: { kind: d.scheme, p: { ...defaultParams(PALETTE_SCHEMAS[d.scheme]), hue: d.hue } },
    tone: { p: { ...defaultParams(TONE_SCHEMA), sat: 1, adapt: 0.6, bloom: 1, vignette: 0.45, ...d.color } },
    reactions: d.reactions ?? [],
    energy: d.energy,
  };
  if (d.choreo) g.choreo = { p: { ...defaultParams(CHOREO_SCHEMA), ...d.choreo } };
  return { origin: d.origin, name: d.name, genome: repair(g) };
}

const DEFS: Def[] = [
  {
    // Buildings scroll in from the right; each screen strip is a spectrum band that stretches taller and
    // brightens with its band, jumps on the beat; the reflection follows and the sky glows with the bass.
    origin: 'E01', name: 'Night Skyline', energy: [0.3, 0.8], scheme: 'complementary', hue: 0.05,
    color: { adapt: 0.3, reflect: 1, reflectY: -0.16 }, carrier: 'warp', decay: 0.9995, car: { floor: 0.05 },
    chain: [
      op('translate', { vx: -0.2 }),
      op('stretch', { base: -0.16, amt: 0.9, beat: 0.18, strips: 32, win: 1, sky: 1 }, 1, 'view'),
    ],
    bodies: [body({ shape: ['edge', { mode: 0, side: 0, base: -0.16, height: 0.3, density: 0.5 }], material: ['fill'] })],
  },
  {
    // Two walker heads (melody, bass) roam at free angles, turning on beats and downbeats and sharply on
    // hits; the newest body paints over older trail, which drifts on curl noise.
    origin: 'E02', name: 'River of Light', energy: [0, 0.45], scheme: 'analogous', hue: 0.55,
    color: { adapt: 0.25, bloom: 1.2, exposure: 0.85 }, carrier: 'warp', decay: 0.998, car: { floor: 0.4 },
    chain: [op('noise', { amp: 0.0015, scale: 1.6, speed: 0.15 })],
    bodies: [body({
      shape: ['dot', { r: 0.01 }],
      place: ['walker', { heads: 2, step: 0.11, every: 2, turn: 1, curve: 0.8 }],
      motion: ['hits', { amt: 1 }],
      material: ['fill', { gain: 2, soft: 0.6 }],
      emit: ['cover', { amt: 1, tip: 1 }],
      feel: ['flow', { atk: 0.01, rel: 0.12 }],
    })],
    reactions: [rx('beat', 'col', 0, 'exposure', 0.875), rx('bass', 'col', 0, 'exposure', 0.625)],
  },
  {
    // Rain born on the eighth-note grid, fuller on each downbeat; columns fall at 1x or 2x speed.
    origin: 'E03', name: 'Rain Curtains', energy: [0.4, 0.95], scheme: 'analogous', hue: 0.5,
    color: { bloom: 1.1 }, carrier: 'warp', decay: 0.962,
    chain: [op('translate', { vy: -0.4, lanes: 38 })],
    bodies: [body({ shape: ['edge', { mode: 2, side: 1, density: 0.6 }], material: ['fill'] })],
  },
  {
    // A row of smoke sources along the bottom, glowing with their instruments and throwing sparks on top.
    origin: 'E04', name: 'Rising Smoke', energy: [0.1, 0.6], scheme: 'split', hue: 0.02,
    color: { adapt: 0.4 }, carrier: 'warp', decay: 0.9935, car: { blur: 0.3 },
    chain: [op('noise', { amp: 0.0014, scale: 1.8, speed: 0.3 }), op('translate', { vy: 0.29 })],
    bodies: [body({
      shape: ['dot', { r: 0 }],
      place: ['row', { count: 4, y: -0.5, wander: 0.05 }],
      material: ['glow', { gain: 0.15, width: 0.04 }],
      emit: ['sparks', { count: 1536, size: 5, speed: 0.6, curl: 0.1, life: 0.55, lift: 0.12, drag: 1.5, top: 1, body: 1 }],
    })],
    reactions: [rx('beat', 'op', 1, 'vy', 0.48), rx('bass', 'op', 1, 'vy', 0.36)],
  },
  {
    // Zoom, swirl and spin all share the wandering centre; two dots orbit it.
    origin: 'E05', name: 'Wandering Vortex', energy: [0.5, 1], scheme: 'triad', hue: 0.7,
    color: { bloom: 1.1 }, carrier: 'warp', decay: 0.986,
    chain: [
      op('zoom', { rate: -0.004, wander: 0.35 }),
      op('swirl', { amt: -0.01, k: 6, wander: 0.35 }),
      op('rotate', { lock: 0.25, wander: 0.35 }),
    ],
    bodies: [body({
      shape: ['dot', { r: 0 }],
      place: ['orbit', { count: 2, radius: 0.3, follow: 0.35, rate: 0.5 }],
      material: ['glow', { gain: 0.9, width: 0.018 }],
    })],
    reactions: [rx('bass', 'op', 0, 'rate', -0.2), rx('beat', 'op', 0, 'rate', -0.33)],
  },
  {
    // Pure fluid; the sources move with their instruments (drums jump every bar, bass swings out, vocals
    // follow the melody), aims step on hits and dye goes out in beat pulses at a tempo-scaled speed.
    origin: 'E06', name: 'Ink Garden', energy: [0.15, 0.75], scheme: 'triad', hue: 0.15,
    color: { adapt: 0.35 }, carrier: 'fluid', decay: 0.993, car: { floor: 1.5, amount: 1, vort: 28, fnoise: 0.35 },
    bodies: [body({
      shape: ['dot', { r: 0 }],
      place: ['stations', { count: 4, inst: 1, xs: 1, jump: 0, wander: 0 }],
      material: ['glow', { gain: 1, width: 0.02 }],
      emit: ['dye', { force: 1 }],
      feel: ['flow', { atk: 0.01, rel: 0.2 }],
    })],
  },
  {
    origin: 'E07', name: 'Oscilloscope', energy: [0, 0.4], scheme: 'mono', hue: 0.45,
    color: { sat: 0.7, adapt: 0.15, bloom: 0.9, vignette: 0.6 }, carrier: 'warp', decay: 0.955, car: { blur: 0.2 },
    chain: [op('zoom', { rate: -0.0008 }), op('translate', { vy: 0.084 })],
    bodies: [body({
      shape: ['curve', { form: 0, amp: 0.24 }],
      place: ['point', { x: 0, y: -0.1 }],
      motion: ['spin', { rate: 0.0625 }],
      material: ['line', { gain: 1.1, width: 1.5 }],
    })],
  },
  {
    // Contours flow a quarter band per beat, the beat is a brightness pulse, a smoothed bass swells the
    // warp and thickens the lines, the melody shifts hue.
    origin: 'E08', name: 'Contour Plasma', energy: [0.3, 0.85], scheme: 'split', hue: 0.6,
    color: { bloom: 0.9, adapt: 0.4 }, carrier: 'none',
    bodies: [body({ shape: ['plasma', { scale: 1.7, warp: 3, bands: 8, lines: 1, speed: 0.15, tempo: 1, pulse: 1, melHue: 1 }], material: ['fill'], emit: ['none'] })],
  },
  {
    // A mirrored inkblot that breathes: beats push the ink outward from the spine, drum hits drop fresh
    // blots at new heights, the bass swells the base; colours swap every bar.
    origin: 'E09', name: 'Rorschach', energy: [0.2, 0.7], scheme: 'complementary', hue: 0.85,
    color: { adapt: 0.35 }, carrier: 'warp', decay: 0.986, car: { blur: 0.1 },
    chain: [
      op('noise', { amp: 0.0011, scale: 2.5, speed: 0.35 }),
      op('push', { amt: 0.0008, axis: 0 }),
      op('zoom', { rate: 0.0008 }),
      op('mirror', { axis: 0 }, 1, 'view'),
    ],
    bodies: [body({
      shape: ['dot', { r: 0 }],
      place: ['stations', { count: 3, inst: 1, xs: 0.04, jump: 1, wander: 0, swap: 1 }],
      material: ['glow', { gain: 0.07, width: 0.025 }],
    })],
    reactions: [
      rx('beat', 'op', 1, 'amt', 0.65), rx('bass', 'op', 1, 'amt', 0.3),
      rx('vocals', 'op', 0, 'amp', 1), rx('other', 'op', 0, 'amp', 1),
      rx('beat', 'ma', 0, 'gain', 0.085),
    ],
  },
  {
    // Hexagon keys on a turning hex grid: each key is a pitch class, lit by the chroma on a beat-random
    // schedule, with faint outlines breathing with the loudness.
    origin: 'E10', name: 'Hex Keys', energy: [0.4, 0.9], scheme: 'mono', hue: 0.6,
    color: { adapt: 0.35 }, carrier: 'warp', decay: 0.93,
    bodies: [body({
      shape: ['polygon', { n: 6, r: 0.4965 / 5 }],
      place: ['grid', { lattice: 1, scale: 5, jitter: 0, density: 1, lit: 0.5, links: 0, twinkle: 0, lock: 0.0625 }],
      material: ['fill', { gain: 2.4, soft: 0, outline: 1, core: 1 }],
    })],
  },
  {
    origin: 'E11', name: 'Sunrise Arc', energy: [0.35, 0.85], scheme: 'analogous', hue: 0.02,
    color: { adapt: 0.35, reflect: 1, reflectY: -0.42 }, carrier: 'warp', decay: 0.925,
    chain: [op('zoom', { rate: 0.006, cy: -0.4 })],
    bodies: [body({
      shape: ['bars', { mode: 1, bins: 40, radius: 0.2, len: 0.3, fill: 0.6 }],
      place: ['point', { x: 0, y: -0.42 }],
      material: ['fill', { gain: 2.6, soft: 0 }],
    })],
    reactions: [rx('beat', 'op', 0, 'rate', 0.3)],
  },
  {
    // Sparks from the centre; warp speed surges on every beat (fast attack, slow ease), cruises with the
    // loudness and jumps on a drop; the tunnel zoom follows the same surge.
    origin: 'E12', name: 'Warp Speed', energy: [0.55, 1], scheme: 'analogous', hue: 0.58,
    color: { sat: 0.35, bloom: 1.1 }, carrier: 'warp', decay: 0.86,
    chain: [op('rotate', { lock: 0.0625 }), op('zoom', { rate: 0.0082 })],
    bodies: [body({
      shape: ['dot', { r: 0.004 }],
      place: ['point'],
      material: ['glow', { gain: 1.57 }],
      emit: ['sparks', { count: 5120, size: 3.4, speed: 0.1, curl: 0, life: 0.2, zoomFlow: 1.1, drag: 4, spread: 0.12, surge: 1, top: 0, body: 0 }],
    })],
    reactions: [rx('surge', 'op', 1, 'rate', 0.4)],
  },
  {
    // Stars on a jittered grid; each is a pitch class that shines with the chroma, twinkles, drifts, and
    // links to lit neighbours.
    origin: 'E13', name: 'Constellations', energy: [0, 0.45], scheme: 'mono', hue: 0.6,
    color: { adapt: 0.15, vignette: 0.55 }, carrier: 'warp', decay: 0.9,
    bodies: [body({
      shape: ['dot', { r: 0 }],
      place: ['grid', { lattice: 0, scale: 6.5, jitter: 0.6, density: 0.5, lit: 1, links: 1, twinkle: 1, lock: 0 }],
      motion: ['drift', { vx: 0.0046 }],
      material: ['glow', { gain: 0.32, width: 0.0018, base: 0.25, halo: 0.08 }],
      feel: ['flow', { atk: 0.1, rel: 0.8 }],
    })],
  },
  {
    // A wireframe solid turning once per bar in 3D; its shape follows the song sections.
    origin: 'E14', name: 'Polyhedra', energy: [0.35, 0.9], scheme: 'triad', hue: 0.5,
    color: { adapt: 0.3 }, carrier: 'warp', decay: 0.88,
    chain: [op('rotate', { lock: -0.0625 }), op('zoom', { rate: -0.004 })],
    bodies: [body({
      shape: ['solid', { solid: 5, size: 0.2, tilt: 0.45, inner: 1 }],
      motion: ['spin', { rate: 1 }],
      material: ['line', { width: 1.3, halo: 0.12 }],
      feel: ['flow', { atk: 0.08, rel: 0.6 }],
    })],
  },
  {
    // Six chrome drops floating on slow paths, melting into each other; their sizes follow the instruments.
    origin: 'E15', name: 'Liquid Chrome', energy: [0.3, 0.8], scheme: 'complementary', hue: 0.55,
    color: { sat: 0.6, adapt: 0.4 }, carrier: 'none',
    bodies: [body({
      shape: ['dot', { r: 0.06 }],
      place: ['float', { count: 6, spread: 0.55, speed: 0.122, fuse: 0.1 }],
      material: ['chrome', { chrome: 1 }],
      feel: ['flow', { atk: 0.02, rel: 0.25 }],
      emit: ['none'],
    })],
  },
  {
    origin: 'E16', name: 'Aurora', energy: [0.1, 0.6], scheme: 'analogous', hue: 0.35,
    color: { adapt: 0.3 }, carrier: 'warp', decay: 0.965, car: { blur: 0.1 },
    chain: [op('noise', { amp: 0.0006, scale: 1.5, speed: 0.5 }), op('translate', { vy: 0.03 })],
    bodies: [body({ shape: ['aurora', { fall: 5, rays: 24, wav: 1.4 }], place: ['point', { y: 0 }], material: ['glow'] })],
  },
  {
    // A hexagon outline tunnelling toward the viewer, turning every bar.
    origin: 'E17', name: 'Hyperspace', energy: [0.6, 1], scheme: 'complementary', hue: 0.8,
    color: { bloom: 1.2 }, carrier: 'warp', decay: 0.94,
    chain: [op('zoom', { rate: 0.02, radial: 1 }), op('rotate', { lock: 0.25, alt: 1 })],
    bodies: [body({
      shape: ['solid', { solid: 4, sides: 6, size: 0.12, inner: 0 }],
      motion: ['spin', { rate: 1 }],
      material: ['line', { gain: 1.6, width: 1.8, halo: 0.12 }],
      feel: ['flow', { atk: 0.03, rel: 0.3 }],
    })],
    reactions: [rx('bass', 'op', 0, 'rate', 0.5), rx('beat', 'op', 0, 'rate', 0.8), rx('build', 'op', 0, 'rate', 0.5)],
  },
  {
    // A waveform arc folded eight ways, zooming out and turning.
    origin: 'E18', name: 'Mandala', energy: [0.5, 1], scheme: 'triad', hue: 0.1,
    color: { bloom: 1.1 }, carrier: 'warp', decay: 0.94,
    chain: [op('rotate', { lock: 0.125 }), op('zoom', { rate: 0.006 }), op('kaleido', { n: 8, lock: 0.0625 }, 1, 'view')],
    bodies: [body({
      shape: ['curve', { form: 4, radius: 0.27, amp: 0.25, turns: 3 }],
      motion: ['spin', { rate: 0.0625 }],
      material: ['line', { gain: 0.8, width: 1.8 }],
    })],
    reactions: [rx('beat', 'ma', 0, 'gain', 0.6)],
  },
  {
    // Ratios step every 2 bars through a circle-of-fifths order from the key and glide; phases advance
    // with the beat; downbeats swing it, the melody detunes it, drum hits widen the second pendulum.
    origin: 'E19', name: 'Harmonograph', energy: [0.05, 0.5], scheme: 'analogous', hue: 0.1,
    color: { adapt: 0.15, vignette: 0.55 }, carrier: 'warp', decay: 0.86,
    bodies: [body({ shape: ['curve', { form: 5, radius: 0.36, amp: 0.2, ra: 1, rb: 2 }], material: ['line', { gain: 0.6, width: 1.2 }] })],
  },
  {
    // A soft moon with five curling arms that reach and retract with the instruments, sway every 2 bars
    // and turn every 16; it bobs with the beat and the reflection matches.
    origin: 'E20', name: 'Moonrise', energy: [0, 0.45], scheme: 'analogous', hue: 0.62,
    color: { sat: 0.5, adapt: 0.15, vignette: 0.5, reflect: 1, reflectY: -0.14 }, carrier: 'none',
    bodies: [body({
      shape: ['dot', { r: 0.05 }],
      place: ['point', { x: 0.27, y: 0.1 }],
      motion: ['bob', { amp: 1 }],
      deform: ['arms', { count: 5, reach: 1 }],
      material: ['textured', { tex: 0, amount: 1, halo: 0.5 }],
      feel: ['flow', { atk: 0.05, rel: 0.5 }],
      emit: ['none'],
    })],
  },
  {
    // Ray-marched terrain: a valley in the middle, spectrum hills either side, a kick ridge rolling toward
    // the viewer, lines coloured by height, downbeat flash; a striped sun sets behind the horizon.
    origin: 'E21', name: 'Retro Grid', energy: [0.5, 1], scheme: 'split', hue: 0.85,
    color: { bloom: 1.2, adapt: 0.35 }, carrier: 'none',
    bodies: [
      body({ shape: ['terrain', { speed: 0.5, density: 1.5, peaks: 1, terrain: 1, flash: 1 }], place: ['point', { y: -0.06 }], material: ['fill'], emit: ['none'] }),
      body({
        shape: ['dot', { r: 0.19 }],
        place: ['point', { x: 0, y: 0.09 }],
        material: ['textured', { tex: 1, amount: 1, halo: 0.35, clip: -0.06 }],
        emit: ['none'],
      }),
    ],
  },
  {
    // Flows between two forms every 8 bars (a drop reverses it); the three transforms drift so it keeps
    // unfolding; bass breathes, beats kick.
    origin: 'E22', name: 'Silk Flame', energy: [0.05, 0.5], scheme: 'analogous', hue: 0.55,
    color: { adapt: 0.3, bloom: 0.9, tonemap: 1 }, carrier: 'warp', decay: 0.955,
    bodies: [body({
      shape: ['flame', { count: 262144, zoom: 0.2, rounds: 2, flow: 1, breathe: 0.12 }, [
        xf({ aff: [0.7, -0.3, 0.3, 0.7, 0.2, 0.1], weight: 1, color: 0, vars: { julia: 0.7, linear: 0.3 }, alt: { spiral: 0.6, heart: 0.4 }, spin: 0.125, drift: [0.25, 0.2] }),
        xf({ aff: [0.5, 0, 0, 0.5, -0.5, 0.2], weight: 0.6, color: 0.5, vars: { spherical: 0.5, swirl: 0.3 }, alt: { disc: 0.6, horseshoe: 0.4 }, bass: 0.2, drift: [0.3, 0.25] }),
        xf({ aff: [-0.4, 0.3, -0.3, -0.4, 0.3, -0.4], weight: 0.4, color: 0.9, vars: { sinusoidal: 1 }, alt: { handkerchief: 0.7, polar: 0.3 }, spin: -0.0625, pulse: 0.08, drift: [0.2, 0.3] }),
      ]],
      motion: ['spin', { rate: 0.125 }],
      material: ['glow'],
    })],
  },
  {
    // Energetic: flows every 4 bars (a drop reverses it), strong bass breathing, beat kicks on every
    // transform, wide drift, longer trails.
    origin: 'E23', name: 'Ember Flame', energy: [0.55, 1], scheme: 'split', hue: 0.02,
    color: { adapt: 0.35, bloom: 1.2, tonemap: 1 }, carrier: 'warp', decay: 0.92,
    bodies: [body({
      shape: ['flame', { count: 524288, zoom: 0.3, rounds: 2, flow: 2, breathe: 0.2 }, [
        xf({ aff: [0.8, 0.2, -0.2, 0.8, 0, 0.1], weight: 1, color: 0.1, vars: { swirl: 0.6, linear: 0.4 }, alt: { spiral: 0.5, heart: 0.3 }, spin: 0.25, bass: 0.25, pulse: 0.1, drift: [0.3, 0.25] }),
        xf({ aff: [0.4, -0.35, 0.35, 0.4, 0.6, 0], weight: 0.5, color: 0.6, vars: { horseshoe: 1 }, alt: { disc: 1 }, pulse: 0.12, drift: [0.35, 0.3] }),
        xf({ aff: [0.5, 0, 0, -0.5, -0.4, -0.3], weight: 0.4, color: 0.95, vars: { handkerchief: 0.8 }, alt: { polar: 0.8 }, spin: -0.125, pulse: 0.08, drift: [0.25, 0.35] }),
      ]],
      motion: ['spin', { rate: -0.125 }],
      material: ['glow'],
    })],
  },
  {
    // Flows between a spiral galaxy and a looser swirl every 8 bars (a drop reverses it); the arms drift,
    // bass breathes, beats kick the core.
    origin: 'E24', name: 'Spiral Nebula', energy: [0.3, 0.75], scheme: 'triad', hue: 0.6,
    color: { adapt: 0.3, tonemap: 1 }, carrier: 'warp', decay: 0.94,
    bodies: [body({
      shape: ['flame', { count: 262144, zoom: 0.22, rounds: 2, flow: 1, breathe: 0.15 }, [
        xf({ aff: [0.6, -0.5, 0.5, 0.6, 0, 0], weight: 1, color: 0, vars: { spiral: 0.3, linear: 0.7 }, alt: { swirl: 0.5, linear: 0.5 }, spin: 0.125, bass: 0.15, pulse: 0.08, drift: [0.12, 0.12] }),
        xf({ aff: [0.3, 0, 0, 0.3, 0.7, 0], weight: 0.5, color: 0.5, vars: { spherical: 1 }, alt: { julia: 0.7, spherical: 0.3 }, drift: [0.25, 0.3] }),
        xf({ aff: [0.3, 0, 0, 0.3, -0.35, 0.6], weight: 0.3, color: 0.85, vars: { disc: 0.6, linear: 0.4 }, alt: { heart: 0.6, linear: 0.4 }, drift: [0.3, 0.25] }),
      ]],
      motion: ['spin', { rate: 0.125 }],
      material: ['glow'],
    })],
  },
];

// ------------------------------------------------------------ MilkDrop classics
// Per-frame MilkDrop values map onto the chain as: zoom z -> zoom rate z - 1, rot -> rotate rate,
// cx / cy wandering -> the ops' shared wander path, warp -> a noise op, dx / dy -> translate,
// the z-squared per-pixel warp -> a quad op,
// decay -> half-life, per-pixel rotation growing toward the centre -> swirl, video echo -> a
// view-stage mirror or kaleido.

const MILKDROP: Def[] = [
  {
    // Geiss, Cosmic Dust 2: a dotted scope throws dust that streams out of a slowly wandering centre
    // (zoom 1.05) with a gentle warp; bass hits jolt the whole field sideways and the dust colour drifts.
    origin: 'M01', name: 'Cosmic Dust 2 (after Geiss)', energy: [0.35, 0.95], scheme: 'analogous', hue: 0.72,
    color: { sat: 0.45, adapt: 0.2, bloom: 1, vignette: 0.35, contrast: 0.06 }, carrier: 'warp', decay: 0.96,
    chain: [
      op('zoom', { rate: 0.05, wander: 0.3 }),
      op('rotate', { lock: 0, rate: 0.0015, wander: 0.3 }),
      op('noise', { amp: 0.0012, scale: 3.1, speed: 0.4 }),
      op('translate', { vx: 0, vy: 0 }),
    ],
    bodies: [
      body({
        shape: ['dot', { r: 0.002 }],
        place: ['point'],
        material: ['glow', { gain: 0.35, width: 0.004 }],
        emit: ['sparks', { count: 8192, size: 1.8, speed: 0.4, curl: 0.15, zoomFlow: 1.2, drag: 2, life: 0.3, spread: 0.1, surge: 0.6, top: 0, body: 0 }],
        color: ['age', { hue: 0, rate: 0.125, detail: 0.6 }],
      }),
      body({
        shape: ['curve', { form: 3, radius: 0.1, amp: 0.5, ra: 2, rb: 3 }],
        material: ['dots', { gain: 0.2, spacing: 0.03 }],
        color: ['age', { hue: 0.3, rate: 0.125, detail: 0.6 }],
      }),
    ],
    reactions: [
      rx('hit', 'op', 3, 'vx', 0.5, { rel: 0.25 }), rx('bass', 'op', 3, 'vy', -0.3, { atk: 0.02, rel: 0.3, thr: 0.5 }),
      rx('surge', 'op', 0, 'rate', 0.25), rx('loud', 'em', 0, 'speed', 0.4, { atk: 0.02, rel: 0.2 }),
    ],
  },
  {
    // Rovastar, Loadus + Geiss, FractalDrop (Triple Mix): discs that copy the picture into themselves
    // (an IFS of shrinking maps) turn faster the more bass there is, blurred and lightly sharpened,
    // while the zooming feedback is folded three ways into a radiating fractal flower.
    origin: 'M02', name: 'FractalDrop (after Rovastar & Loadus)', energy: [0.3, 0.85], scheme: 'mono', hue: 0.85,
    color: { adapt: 0.35, bloom: 1, tonemap: 1 }, carrier: 'warp', decay: 0.9, car: { sharpen: 0.08, grain: 0.003, blur: 0.3 },
    chain: [
      op('zoom', { rate: 0.0099 }),
      op('rotate', { lock: 0.0625, rate: 0 }),
      op('kaleido', { n: 3, lock: 0 }),
    ],
    bodies: [
      body({
        shape: ['flame', { count: 262144, zoom: 0.26, rounds: 2, flow: 0, breathe: 0.2 }, [
          xf({ aff: [0.5, 0, 0, 0.5, 0.37, 0], weight: 1, color: 0, vars: { linear: 0.6, spherical: 0.4 }, spin: 0.125, bass: 0.25, pulse: 0.05 }),
          xf({ aff: [0.5, 0, 0, 0.5, -0.18, 0.32], weight: 0.8, color: 0.5, vars: { linear: 0.7, disc: 0.3 }, spin: -0.25, bass: 0.2 }),
          xf({ aff: [0.5, 0, 0, 0.5, -0.18, -0.32], weight: 0.6, color: 0.9, vars: { linear: 0.8, spherical: 0.2 }, spin: 0.0625, pulse: 0.08 }),
        ]],
        motion: ['spin', { rate: 0.125 }],
        material: ['glow'],
      }),
    ],
    reactions: [rx('bass', 'op', 1, 'rate', 0.35, { atk: 0.05, rel: 0.8 }), rx('other', 'car', 0, 'sharpen', 0.1, { atk: 0.1, rel: 0.5 })],
  },
  {
    // Eo.S., glowsticks v2 05 and proton lights: glowing sticks swing along looping paths in the dark,
    // reversing every bar, and their fading light trails build ribbons; the loudness widens the loops,
    // the bass lengthens the sticks and the beat flares them.
    origin: 'M03', name: 'Glowsticks (after Eo.S.)', energy: [0.35, 0.9], scheme: 'analogous', hue: 0.5,
    color: { adapt: 0.3, bloom: 1.3, vignette: 0.5, contrast: 0.06 }, carrier: 'warp', car: { halfLife: 0.3 },
    chain: [op('zoom', { rate: 0.002 })],
    bodies: [body({
      shape: ['segment', { len: 0.3, w: 0.005 }],
      place: ['outline', { count: 3, path: 2, radius: 0.14, rate: 0.25 }],
      motion: ['spin', { rate: 0.5, alt: 1 }],
      material: ['line', { gain: 0.7, width: 2, halo: 0.15 }],
      emit: ['trail', { tip: 0.5 }],
      feel: ['flow', { atk: 0.02, rel: 0.3 }],
    })],
    reactions: [
      rx('loud', 'pl', 0, 'radius', 0.25, { atk: 0.05, rel: 0.4 }), rx('bass', 'sh', 0, 'len', 0.25, { atk: 0.02, rel: 0.3 }),
      rx('beat', 'ma', 0, 'gain', 0.3, { rel: 0.2 }),
    ],
  },
  {
    // fiShbRaiN, witchcraft: four pens wander the screen, steered by the bass one way and the treble the
    // other, scribbling glowing curls that a slow swirl twists further; a mirrored echo doubles them.
    origin: 'M04', name: 'Witchcraft (after fiShbRaiN)', energy: [0.2, 0.75], scheme: 'triad', hue: 0.8,
    color: { sat: 0.8, adapt: 0.3, bloom: 1.1 }, carrier: 'warp', car: { halfLife: 0.6 },
    chain: [
      op('zoom', { rate: -0.0005 }),
      op('noise', { amp: 0.0016, scale: 2.2, speed: 0.3 }),
      op('swirl', { amt: 0.008, k: 6, wander: 0.25 }),
      op('mirror', { axis: 0 }, 1, 'view'),
    ],
    bodies: [
      body({
        shape: ['dot', { r: 0.002 }],
        place: ['walker', { heads: 2, step: 0.13, every: 1, square: 0, wrap: 0, curve: 2, turn: 1.5 }],
        motion: ['hits', { amt: 0.8 }],
        material: ['glow', { gain: 1.3, width: 0.004 }],
        emit: ['trail', { tip: 0.3 }],
        feel: ['flow', { atk: 0.01, rel: 0.15 }],
      }),
      body({
        shape: ['dot', { r: 0.002 }],
        place: ['walker', { heads: 2, step: 0.09, every: 2, square: 0, wrap: 0, curve: 1.2, turn: 1.5 }],
        motion: ['hits', { amt: 0.8 }],
        material: ['glow', { gain: 1.3, width: 0.004 }],
        emit: ['trail', { tip: 0.3 }],
        feel: ['flow', { atk: 0.01, rel: 0.15 }],
        color: ['instrument', { hue: 0.5 }],
      }),
    ],
    reactions: [
      rx('bass', 'pl', 0, 'curve', 0.4, { atk: 0.02, rel: 0.2 }), rx('other', 'pl', 1, 'curve', -0.4, { atk: 0.02, rel: 0.2 }),
      rx('loud', 'pl', 0, 'step', 0.3), rx('loud', 'pl', 1, 'step', 0.3),
    ],
  },
  {
    // Geiss, Reaction Diffusion 2: the blur-difference warp grows worm-like Turing patterns out of a
    // faint waveform and treble grain; the centre wanders, the picture turns and bass kicks the zoom.
    origin: 'M05', name: 'Reaction Diffusion 2 (after Geiss)', energy: [0.25, 0.85], scheme: 'analogous', hue: 0.08,
    color: { sat: 1, exposure: 0.6, adapt: 0.2, bloom: 0.7, contrast: 0.05 }, carrier: 'warp', car: { halfLife: 0.09, sharpen: 0.15, grain: 0.05 },
    chain: [op('zoom', { rate: 0.009, wander: 0.3 }), op('rotate', { lock: 0, rate: 0.002, wander: 0.3 })],
    bodies: [body({
      shape: ['curve', { form: 0, amp: 0.3 }],
      motion: ['sway', { amp: 0.06, period: 4 }],
      material: ['line', { gain: 1.2, width: 1.6 }],
      color: ['age', { rate: 0.0625 }],
    })],
    reactions: [rx('bass', 'op', 0, 'rate', 0.5, { atk: 0.02, rel: 0.35, thr: 0.35 }), rx('hit', 'car', 0, 'sharpen', 0.25, { rel: 0.3 })],
  },
  {
    // Flexi, mindblob: two ink sources on springy circles, pulled around by their instruments, stir a
    // liquid of two contrasting colours; the bass thickens the flow and swells the sources.
    origin: 'M06', name: 'Mindblob (after Flexi)', energy: [0.15, 0.7], scheme: 'complementary', hue: 0.72,
    color: { exposure: 0.8, adapt: 0.3, bloom: 1.1 }, carrier: 'fluid', car: { halfLife: 1.2, floor: 1.2, amount: 1.2, vort: 34, fnoise: 0.5 },
    bodies: [body({
      shape: ['dot', { r: 0.015 }],
      place: ['stations', { count: 2, inst: 1, xs: 0.6, jump: 0, wander: 0.2 }],
      motion: ['circle', { radius: 0.1, period: 2 }],
      material: ['glow', { gain: 0.5, width: 0.02 }],
      emit: ['dye', { force: 1.3 }],
      feel: ['flow', { atk: 0.02, rel: 0.35 }],
    })],
    reactions: [
      rx('bass', 'car', 0, 'amount', 0.3, { atk: 0.05, rel: 0.5 }), rx('bass', 'sh', 0, 'r', 0.15, { atk: 0.03, rel: 0.3 }),
      rx('loud', 'em', 0, 'force', 0.5, { atk: 0.05, rel: 0.4 }),
    ],
  },
  {
    // Rovastar, Fractopia: a z-squared flow pulls the coloured border (and the dark band inside it) in
    // from the edges while the picture turns, growing spiralling fractal coastlines around a dotted
    // scope; the centre wanders with the bars and the bass strengthens the flow.
    origin: 'M07', name: 'Fractopia (after Rovastar)', energy: [0.2, 0.75], scheme: 'triad', hue: 0.6,
    color: { adapt: 0.35, bloom: 1, vignette: 0.2 }, carrier: 'warp', car: { halfLife: 3, border: 1, floor: 0 },
    chain: [
      op('quad', { amt: 2, turn: 0.25, cx: 0, cy: -0.1, wander: 0.15 }),
      op('rotate', { lock: -0.25, rate: -0.01, wander: 0.15 }),
      op('zoom', { rate: -0.01, wander: 0.15 }),
    ],
    bodies: [body({
      shape: ['curve', { form: 3, radius: 0.1, amp: 0.5, ra: 2, rb: 3 }],
      material: ['dots', { gain: 0.25, spacing: 0.03 }],
      color: ['age', { rate: 0.125 }],
    })],
    reactions: [rx('bass', 'op', 0, 'amt', 0.15, { atk: 0.1, rel: 1 }), rx('beat', 'car', 0, 'border', -0.3)],
  },
  {
    // Rovastar + Geiss, Hurricane Nightmare: a circular waveform feeds a vortex that spins hardest at
    // the eye and zooms hardest at the edges; loud bass winds it tighter.
    origin: 'M08', name: 'Hurricane Nightmare (after Rovastar & Geiss)', energy: [0.45, 1], scheme: 'analogous', hue: 0.55,
    color: { adapt: 0.35, bloom: 1.2, vignette: 0.55 }, carrier: 'warp', decay: 0.965,
    chain: [
      op('zoom', { rate: 0.012, radial: 1 }),
      op('swirl', { amt: -0.03, k: 3 }),
      op('swirl', { amt: -0.03, k: 6 }),
      op('rotate', { lock: 0, rate: -0.006 }),
    ],
    bodies: [body({
      shape: ['curve', { form: 1, radius: 0.16, amp: 0.3 }],
      place: ['point', { x: 0, y: -0.03 }],
      material: ['line', { gain: 1.1, width: 2.4, halo: 0.3 }],
      color: ['age', { rate: 0.125, detail: 0.5 }],
    })],
    reactions: [rx('bass', 'op', 1, 'amt', -0.35, { atk: 0.05, rel: 0.6 }), rx('bass', 'op', 2, 'amt', -0.2, { atk: 0.05, rel: 0.6 }), rx('surge', 'op', 0, 'rate', 0.3)],
  },
  {
    // Krash + Rovastar, Rainbow Orb: a small circular waveform, shifting sideways with the loudness, is
    // spun by a rotation strongest at the centre and streamed outward by a fast zoom, so its colour
    // history lays down rainbow rings; the echo mirrors it and the treble holds the zoom back.
    origin: 'M09', name: 'Rainbow Orb (after Krash & Rovastar)', energy: [0.4, 1], scheme: 'triad', hue: 0,
    color: { sat: 1, adapt: 0.35, bloom: 1.2 }, carrier: 'warp', decay: 0.975,
    chain: [
      op('zoom', { rate: 0.03 }),
      op('swirl', { amt: 0.03, k: 1 }),
      op('swirl', { amt: 0.03, k: 1 }),
      op('swirl', { amt: 0.03, k: 1 }),
      op('mirror', { axis: 0 }, 1, 'view'),
    ],
    bodies: [body({
      shape: ['curve', { form: 1, radius: 0.08, amp: 0.3 }],
      material: ['line', { gain: 1, width: 2 }],
      color: ['age', { rate: 0.5, detail: 1 }],
    })],
    reactions: [
      rx('loud', 'pl', 0, 'x', 0.25, { atk: 0.05, rel: 0.3 }), rx('other', 'op', 0, 'rate', -0.3, { atk: 0.05, rel: 0.4 }),
      rx('bass', 'op', 1, 'amt', -0.2, { atk: 0.05, rel: 0.4 }),
    ],
  },
  {
    // Geiss, Thumb Drum: a circular waveform seeds fingerprint-like sharpened stripes that two
    // counter-rotating vortices keep stirring over a grey field; the mids decide how hard they stir.
    origin: 'M10', name: 'Thumb Drum (after Geiss)', energy: [0.2, 0.7], scheme: 'mono', hue: 0.6,
    color: { sat: 0.25, exposure: 0.8, adapt: 0.3, bloom: 0.9, contrast: 0.05 }, carrier: 'warp', car: { halfLife: 0.2, sharpen: 0.3, grain: 0.006 },
    chain: [
      op('swirl', { amt: 0.02, k: 4, cx: -0.25, cy: 0.05, wander: 0.2 }),
      op('swirl', { amt: -0.02, k: 4, cx: 0.25, cy: -0.05, wander: 0.2 }),
      op('zoom', { rate: 0.005 }),
    ],
    bodies: [body({
      shape: ['curve', { form: 1, radius: 0.22, amp: 0.35 }],
      material: ['line', { gain: 0.9, width: 1.4 }],
    })],
    reactions: [
      rx('vocals', 'op', 0, 'amt', 0.3, { atk: 0.05, rel: 0.5 }), rx('vocals', 'op', 1, 'amt', -0.3, { atk: 0.05, rel: 0.5 }),
      rx('other', 'car', 0, 'sharpen', 0.2, { atk: 0.05, rel: 0.4 }),
    ],
  },
  {
    // Flexi, Martin + Geiss, dedicated to the sherwin maxawow: two wandering vortices, one stirred by the
    // bass and one by the mids, and a bass-driven turbulence pull a colour frame (its hue swinging every
    // bar) inward into layered ribbons that never fade; the picture is lit as a glossy embossed surface.
    origin: 'M11', name: 'Sherwin Maxawow (after Flexi, Martin & Geiss)', energy: [0.2, 0.8], scheme: 'triad', hue: 0.05,
    color: { sat: 1, exposure: 1.3, contrast: 0.005, adapt: 0.1, bloom: 0.6, vignette: 0.05, relief: 1, bump: 3, light: 0.375, gloss: 0.9, metal: 0.15 },
    carrier: 'warp', car: { halfLife: 25, floor: 0, border: 0.75 },
    chain: [
      op('swirl', { amt: 0.018, k: 1.5, cx: -0.3, cy: 0.1, wander: 0.3 }),
      op('swirl', { amt: -0.018, k: 1.5, cx: 0.3, cy: -0.1, wander: 0.35 }),
      op('noise', { amp: 0.002, scale: 1.4, speed: 0.35 }),
      op('zoom', { rate: -0.008 }),
    ],
    bodies: [body({
      shape: ['curve', { form: 1, radius: 0.1, amp: 0.3 }],
      place: ['walker', { heads: 1, step: 0.08, every: 2, curve: 1.2 }],
      material: ['line', { gain: 0.25, width: 3 }],
      color: ['age', { rate: 0.5, detail: 0.3 }],
    })],
    reactions: [
      rx('bass', 'op', 0, 'amt', 0.5, { atk: 0.03, rel: 0.4 }), rx('other', 'op', 1, 'amt', -0.5, { atk: 0.03, rel: 0.4 }),
      rx('bass', 'op', 2, 'amp', 0.5, { atk: 0.02, rel: 0.3, thr: 0.3 }), rx('bar', 'pal', 0, 'hue', 1),
      rx('bar', 'col', 0, 'light', 0.4),
    ],
  },
  {
    // Flexi, mindblob 2.0: a two-colour cell foam, hot walls around cool blobs, bent by a slow
    // inward flow so the membranes curl; the bass pushes the cells about and the beat swells a few.
    origin: 'M12', name: 'Mindblob Foam (after Flexi)', energy: [0.15, 0.7], scheme: 'analogous', hue: 0.92,
    color: { sat: 1, exposure: 1.1, contrast: 0.01, adapt: 0.3, bloom: 0.6, vignette: 0.15 },
    carrier: 'warp', car: { halfLife: 0.08, floor: 0.2 },
    chain: [op('swirl', { amt: 0.01, k: 3, wander: 0.3 })],
    bodies: [body({
      shape: ['cells', { mode: 0, scale: 2.2, speed: 0.35, warp: 0.9, wall: 0.6, fill: 1, var: 0.05, pulse: 0.3 }],
      material: ['fill', { gain: 1.1 }],
      emit: ['trail'],
      feel: ['flow', { atk: 0.02, rel: 0.4 }],
    })],
    reactions: [rx('bass', 'sh', 0, 'warp', 0.3, { atk: 0.05, rel: 0.6 }), rx('loud', 'sh', 0, 'wall', 0.3, { atk: 0.05, rel: 0.4 })],
  },
  {
    // Flexi, alien fish pond: a pond of shaded cells, each a lit dome with a dark nucleus drifting
    // inside, packed edge to edge; the colours roll slowly through the palette and the drums jostle them.
    origin: 'M13', name: 'Alien Fish Pond (after Flexi)', energy: [0.1, 0.6], scheme: 'triad', hue: 0.35,
    color: { sat: 0.95, exposure: 1.1, adapt: 0.3, bloom: 0.7, vignette: 0.3, relief: 0.5, bump: 1.2, gloss: 0.4 },
    carrier: 'warp', car: { halfLife: 0.15, floor: 0.1 },
    chain: [op('rotate', { lock: 0, rate: 0.0006 })],
    bodies: [body({
      shape: ['cells', { mode: 2, scale: 3.6, speed: 0.25, warp: 0.35, wall: 0.1, fill: 0.9, var: 0.25, pulse: 0.2 }],
      material: ['fill', { gain: 1 }],
      emit: ['trail'],
      color: ['height', { amount: 0.6, detail: 1 }],
    })],
    reactions: [rx('drums', 'sh', 0, 'speed', 0.3, { atk: 0.02, rel: 0.3 }), rx('bar', 'sh', 0, 'var', 0.3)],
  },
  {
    // Mig, COLORFUL9: a wandering, slowly turning zoom grows sharpened contour patterns out of a
    // drifting blob; the picture's brightness is mapped through the palette in hard neon bands that
    // scroll every bar; treble adds grain and bass hits jolt the flow sideways.
    origin: 'M14', name: 'Colorful 9 (after Mig)', energy: [0.3, 0.9], scheme: 'triad', hue: 0,
    color: { sat: 1, exposure: 1.05, adapt: 0.25, bloom: 0.5, vignette: 0.1, huemap: 1, bands: 3, drift: 0.5, poster: 0.85 },
    carrier: 'warp', car: { halfLife: 1.2, floor: 0.1, sharpen: 0.3, grain: 0.05 },
    chain: [
      op('zoom', { rate: 0.009, wander: 0.3 }),
      op('rotate', { lock: 0, rate: 0.004, wander: 0.3 }),
      op('translate', { vx: 0.02, vy: 0.015 }),
    ],
    bodies: [body({
      shape: ['dot', { r: 0.09 }],
      place: ['walker', { heads: 2, step: 0.16, every: 1, curve: 1.2 }],
      material: ['fill', { gain: 0.8, soft: 0.6 }],
      emit: ['trail'],
    })],
    reactions: [rx('hit', 'op', 2, 'vx', 0.5, { rel: 0.3 }), rx('other', 'car', 0, 'sharpen', 0.25, { atk: 0.05, rel: 0.4 }), rx('bass', 'col', 0, 'bands', 0.15, { atk: 0.05, rel: 0.5 })],
  },
  {
    // Zylot, Crossing Over (Paint Spatter mix): a zoom that pulls hardest at the rim streams sharpened
    // splashes outward from a waveform ring, and big discs splat on the bass; the contrast is pushed
    // until the colours fold over (solarized) into hard fringes on black.
    origin: 'M15', name: 'Crossing Over (after Zylot)', energy: [0.35, 0.95], scheme: 'split', hue: 0.55,
    color: { sat: 1, exposure: 0.95, contrast: 0.04, adapt: 0.2, bloom: 0.35, vignette: 0.1, huemap: 0.8, bands: 1.5, solar: 0.35, poster: 0.8 },
    carrier: 'warp', car: { halfLife: 0.3, floor: 0.8 },
    chain: [op('zoom', { rate: 0.012, radial: 1 })],
    bodies: [
      body({
        shape: ['curve', { form: 1, radius: 0.12, amp: 0.35 }],
        material: ['line', { gain: 1.2, width: 2.5 }],
        color: ['age', { rate: 0.25, detail: 0.5 }],
      }),
      body({
        shape: ['dot', { r: 0.12 }],
        place: ['float', { count: 2, spread: 0.4, speed: 0.2 }],
        material: ['fill', { gain: 1, soft: 0.2 }],
        emit: ['trail'],
        feel: ['flow', { atk: 0.01, rel: 0.2, thr: 0.35, sens: 1.8 }],
      }),
    ],
    reactions: [rx('bass', 'op', 0, 'rate', 0.15, { atk: 0.02, rel: 0.3 }), rx('bass', 'ma', 1, 'gain', 0.4, { atk: 0.01, rel: 0.25, thr: 0.5 })],
  },
  {
    // martin, tunnel race: bright waveform bands scroll up the carried picture, which is wrapped onto
    // the wall of a pale tunnel rushing toward you as rings, surging on every beat and turning with the
    // bar, with an orb racing round the wall; the far end is lost in haze.
    origin: 'M16', name: 'Tunnel Race (after martin)', energy: [0.35, 0.95], scheme: 'analogous', hue: 0.85,
    color: { sat: 0.8, exposure: 1.2, adapt: 0.25, bloom: 1.1, vignette: 0.15 },
    carrier: 'warp', car: { halfLife: 2.5, blur: 0.15 },
    chain: [
      op('translate', { vx: 0, vy: 0.3 }),
      op('tunnel', { depth: 0.16, speed: 0.8, twist: 0.1, sides: 0, rep: 2, fog: 0.5, lock: 0.0625 }, 1, 'view'),
    ],
    bodies: [
      body({
        shape: ['curve', { form: 0, amp: 0.2 }],
        place: ['point', { x: 0, y: -0.4 }],
        material: ['line', { gain: 1.3, width: 3, halo: 0.4 }],
        color: ['age', { rate: 0.25, detail: 0.5 }],
      }),
      body({
        shape: ['dot', { r: 0.04 }],
        place: ['orbit', { count: 1, radius: 0.3, rate: 0.5 }],
        material: ['glow', { gain: 1.2, width: 0.03 }],
        emit: ['none'],
      }),
    ],
    reactions: [rx('bass', 'op', 1, 'speed', 0.3, { atk: 0.03, rel: 0.4 }), rx('loud', 'op', 1, 'twist', 0.2, { atk: 0.1, rel: 0.8 })],
  },
  {
    // Flexi + Martin, tunnel of supraschismatika: a dark chrome pipe flown through at speed, glints
    // streaking along its polished wall toward you; the bass drives the flight and the twist.
    origin: 'M17', name: 'Tunnel of Supraschismatika (after Flexi & Martin)', energy: [0.3, 0.9], scheme: 'mono', hue: 0.6,
    color: { sat: 0.25, exposure: 0.9, adapt: 0.3, bloom: 1.2, vignette: 0.35, relief: 0.7, bump: 1.6, light: 0.25, gloss: 1, metal: 0.6 },
    carrier: 'warp', decay: 0.9,
    chain: [
      op('translate', { vx: 0, vy: -0.25 }),
      op('tunnel', { depth: 0.22, speed: 1, twist: -0.35, sides: 0, rep: 3, fog: 0.9, lock: 0 }, 1, 'view'),
    ],
    bodies: [body({
      shape: ['dot', { r: 0.01 }],
      place: ['float', { count: 6, spread: 0.6, speed: 0.2 }],
      material: ['glow', { gain: 1.6, width: 0.012, base: 0.4 }],
      emit: ['trail'],
    })],
    reactions: [rx('bass', 'op', 1, 'speed', 0.35, { atk: 0.03, rel: 0.4 }), rx('bass', 'op', 1, 'twist', -0.25, { atk: 0.05, rel: 0.6 }), rx('beat', 'col', 0, 'light', 0.2)],
  },
  {
    // Waltra, Square Orgy: a turning grid of glossy tiles, each lit by the colour behind it, as bright
    // blobs drift and bloom underneath; the grid swells with the bass and the tiles shine like enamel.
    origin: 'M18', name: 'Square Orgy (after Waltra)', energy: [0.3, 0.9], scheme: 'triad', hue: 0.08,
    color: { sat: 1, exposure: 1.05, adapt: 0.3, bloom: 0.8, vignette: 0.2, relief: 0.6, bump: 1.4, gloss: 0.9, light: 0.3 },
    carrier: 'warp', car: { halfLife: 1.4, floor: 0.2 },
    chain: [
      op('zoom', { rate: 0.006, wander: 0.2 }),
      op('mosaic', { size: 0.13, shape: 0, gap: 0.12, angle: 0.07, lock: 0.0625, pulse: 0.5 }, 1, 'view'),
    ],
    bodies: [body({
      shape: ['dot', { r: 0.1 }],
      place: ['float', { count: 6, spread: 0.7, speed: 0.18 }],
      material: ['fill', { gain: 1.2, soft: 0.5 }],
      color: ['height', { amount: 1.5, detail: 1 }],
    })],
    reactions: [rx('bass', 'op', 1, 'size', 0.2, { atk: 0.03, rel: 0.4 }), rx('beat', 'ma', 0, 'gain', 0.4, { rel: 0.25 })],
  },
  {
    // Goody + Flexi, Data Crusher: a field of points streams outward from the centre at speed and is
    // shown as a coarse wall of pixels, so the rush breaks into blocky streaks; the bass drives the
    // flight and the drums flash the field.
    origin: 'M19', name: 'Data Crusher (after Goody & Flexi)', energy: [0.35, 0.95], scheme: 'analogous', hue: 0.5,
    color: { sat: 0.7, exposure: 1.1, adapt: 0.3, bloom: 0.9, vignette: 0.3 },
    carrier: 'warp', car: { halfLife: 0.5, floor: 0.4 },
    chain: [
      op('zoom', { rate: 0.035 }),
      op('rotate', { lock: 0.0625 }),
      op('mosaic', { size: 0.02, shape: 0, gap: 0.25, angle: 0, lock: 0, pulse: 0.3 }, 1, 'view'),
    ],
    bodies: [body({
      shape: ['dot', { r: 0.004 }],
      place: ['grid', { lattice: 0, scale: 9, jitter: 0.6, density: 0.4, lit: 0.5, twinkle: 0.6 }],
      material: ['glow', { gain: 1.4, width: 0.006, base: 0.3 }],
    })],
    reactions: [rx('bass', 'op', 0, 'rate', 0.3, { atk: 0.02, rel: 0.3 }), rx('drums', 'ma', 0, 'gain', 0.5, { atk: 0.01, rel: 0.2 })],
  },
];

// C01.. showcase the choreography gene: the picture is composed over the song's timeline from the
// offline analysis (anticipation before a known drop, the release on it).
const CHOREO: Def[] = [
  {
    // A spinning waveform ring folded six ways and streamed outward. Over the last eight bars before
    // a drop the camera creeps in and leans, the colour drains and the light dims; on the drop it
    // snaps back with a slam of colour and light that settles over two bars.
    origin: 'C01', name: 'Drop Countdown', energy: [0.45, 1], scheme: 'split', hue: 0.08,
    color: { bloom: 1.15, vignette: 0.5 }, carrier: 'warp', decay: 0.955,
    chain: [op('zoom', { rate: 0.012, radial: 1 }), op('rotate', { lock: 0.0625 }), op('kaleido', { n: 6, lock: 0.0625 }, 1, 'view')],
    bodies: [body({
      shape: ['curve', { form: 1, radius: 0.14, amp: 0.3 }],
      material: ['line', { gain: 1, width: 2 }],
      color: ['age', { rate: 0.125 }],
    })],
    reactions: [rx('beat', 'ma', 0, 'gain', 0.5), rx('bass', 'op', 0, 'rate', 0.3, { atk: 0.05, rel: 0.4 })],
    choreo: { lead: 8, curve: 2.5, push: 0.25, roll: 0.015, drain: 0.8, dim: 0.35, punch: 1, relax: 2 },
  },
  {
    // A hexagon keyboard lit by the chroma, shot like a film: each section type has its own camera
    // set-up and colour (verses cool and wide, choruses warm and close, the drop hottest), the camera
    // glides into it over two bars and dollies slowly in across the section; a short build-up leans
    // in before a drop and it lands with a soft punch.
    origin: 'C02', name: 'Scene Director', energy: [0.3, 0.85], scheme: 'analogous', hue: 0.55,
    color: { adapt: 0.35, bloom: 1.05 }, carrier: 'warp', decay: 0.93,
    bodies: [body({
      shape: ['polygon', { n: 6, r: 0.4965 / 5 }],
      place: ['grid', { lattice: 1, scale: 5, jitter: 0, density: 1, lit: 0.6, links: 0, twinkle: 0.3, lock: 0.0625 }],
      material: ['fill', { gain: 2.2, soft: 0.1, outline: 1, core: 1 }],
    })],
    reactions: [rx('beat', 'ma', 0, 'gain', 0.3, { atk: 0.01, rel: 0.25 })],
    choreo: { lead: 4, curve: 1.5, push: 0.12, roll: -0.01, drain: 0.4, dim: 0.2, punch: 0.5, relax: 1, frame: 0.85, shot: 0.37, glide: 2, dolly: 0.08, scene: 0.35 },
  },
];

// V01.. showcase the physics-inspired shape genes (genes/*.ts), one seed per gene.
const PHYSICS: Def[] = [
  {
    // A concert rig on an overhead truss: eight moving heads throw shafts through drifting haze, the
    // swing rippling along the truss every two bars and the beat chasing from head to head.
    origin: 'V01', name: 'Stage Rig', energy: [0.35, 1], scheme: 'triad', hue: 0.62,
    color: { sat: 0.85, adapt: 0.35, bloom: 1.2, vignette: 0.5 }, carrier: 'warp', car: { halfLife: 0.12 },
    bodies: [body({
      shape: ['beams', { count: 8, spread: 1.1, fan: 0.45, sweep: 0.5, pattern: 2, period: 2, width: 0.035, haze: 0.7, gobo: 0, hues: 0.06, length: 1.4, flare: 0.6, accent: 0.6 }],
      place: ['point', { x: 0, y: 0.45 }],
      material: ['glow', { gain: 1 }],
      feel: ['flow', { atk: 0.03, rel: 0.4 }],
    })],
    reactions: [rx('build', 'sh', 0, 'sweep', 0.5, { atk: 0.2, rel: 1 }), rx('bass', 'sh', 0, 'width', 0.2, { atk: 0.05, rel: 0.5 })],
  },
  {
    // A Chladni plate: sand gathers on the nodal lines of the standing wave the chord asks for. On a
    // new chord (checked every bar) the sand scatters and settles onto the new figure over two beats;
    // minor keys give the antisymmetric figures. The plate turns slowly and the bass shakes the grains.
    origin: 'V02', name: 'Chladni Plate', energy: [0.15, 0.8], scheme: 'analogous', hue: 0.08,
    color: { sat: 0.7, adapt: 0.3, bloom: 1.1, vignette: 0.55 }, carrier: 'none',
    bodies: [body({
      shape: ['cymatics', { plate: 0, size: 0.42, modes: 7, source: 0, hold: 1, settle: 2, sand: 0.8, line: 1.6, shake: 0.4, rim: 0.35 }],
      place: ['point', { x: 0, y: 0 }],
      motion: ['spin', { rate: 0.0625 }],
      material: ['glow', { gain: 1 }],
      emit: ['none'],
      feel: ['flow', { atk: 0.02, rel: 0.3 }],
    })],
    reactions: [rx('bass', 'sh', 0, 'line', 0.25, { atk: 0.03, rel: 0.4 })],
  },
];

// ------------------------------------------------------------ Winamp AVS classics
// A01.. re-create presets from the AVS 'Community Picks' / 'Winamp 5 Picks' packs in the same
// vocabulary: superscopes as superscope bodies, Dynamic Movement as the chain, Water / Water Bump as
// the carrier's ripple field, Blur and Fade as carrier blur and half-life. Credited to the authors.

const AVS: Def[] = [
  {
    // UnConeD, Silk Strings: a ribbon of fine threads, each a smooth 3D curve, loops through space
    // while the camera drifts round it; the threads fan apart and twist around each other with the
    // instruments, and the blurred additive glow is mapped into a single dark wine hue.
    origin: 'A01', name: 'Silk Strings (after UnConeD)', energy: [0.15, 0.7], scheme: 'mono', hue: 0.9,
    color: { sat: 0.8, adapt: 0.3, bloom: 1.25, vignette: 0.5 }, carrier: 'warp', car: { halfLife: 0.35, blur: 0.25 },
    chain: [op('zoom', { rate: 0.0015 })],
    bodies: [body({
      shape: ['superscope', { family: 4, p: 1, q: 2, size: 0.28, audio: 0.25, spec: 0, spinX: 0.0625, spinY: 0.125, persp: 0.55, n: 1024 }],
      place: ['orbit', { count: 6, radius: 0.02, rate: 0.125 }],
      material: ['line', { gain: 0.7, width: 0.9, halo: 0.1 }],
      color: ['fixed', { hue: 0, detail: 1 }],
      feel: ['flow', { atk: 0.05, rel: 0.6 }],
    })],
    reactions: [
      rx('bass', 'sh', 0, 'audio', 0.4, { atk: 0.03, rel: 0.4 }), rx('melody', 'pl', 0, 'radius', 0.3, { atk: 0.1, rel: 0.8 }),
      rx('loud', 'ma', 0, 'gain', 0.3, { atk: 0.05, rel: 0.5 }),
    ],
  },
  {
    // Tonic, One More (black and white contest): a plain waveform line, heavily blurred and wiped on
    // every beat, sinks into a grey pool whose ripples (two stacked water passes) bend it into liquid
    // rings; the bass stirs the water harder.
    origin: 'A03', name: 'One More (after Tonic)', energy: [0.3, 0.85], scheme: 'mono', hue: 0.6,
    color: { sat: 0.08, exposure: 0.9, adapt: 0.3, bloom: 1, contrast: 0.06, vignette: 0.35 }, carrier: 'warp',
    car: { halfLife: 0.3, blur: 0.4, water: 0.85, wsize: 0.025 },
    chain: [op('zoom', { rate: 0.004, radial: 1 })],
    bodies: [body({
      shape: ['curve', { form: 0, amp: 0.32 }],
      material: ['line', { gain: 1.1, width: 1.8, halo: 0.2 }],
    })],
    reactions: [rx('hit', 'car', 0, 'floor', 0.6, { rel: 0.15 }), rx('bass', 'car', 0, 'water', 0.3, { atk: 0.03, rel: 0.4 })],
  },
  {
    // Duo, Brainstorm: two thick waveform bars drawn in XOR, so where they cross the trail they
    // punch negative holes in it, jump to new places on the beat, and a grid of cosine ripples,
    // re-rolled on the hits, tears the blue-and-red picture apart.
    origin: 'A07', name: 'Brainstorm (after Duo)', energy: [0.4, 1], scheme: 'complementary', hue: 0.6,
    color: { sat: 0.95, adapt: 0.35, bloom: 1.1, vignette: 0.4, contrast: 0.05 }, carrier: 'warp', car: { halfLife: 0.45, blur: 0.15 },
    chain: [op('ripple', { amp: 0.0022, freq: 12, speed: 1.2, radial: 0 }), op('swirl', { amt: 0.006, k: 3 }), op('zoom', { rate: 0.003 })],
    bodies: [body({
      shape: ['curve', { form: 0, amp: 0.45 }],
      place: ['mirror', { axis: 1, x: 0, y: 0.2 }],
      motion: ['hits', { amt: 1 }],
      material: ['line', { gain: 1.2, width: 3, halo: 0.1, blend: 3 }],
      feel: ['step', { div: 1 }],
      color: ['melody', { hue: 0, amount: 0.5 }],
    })],
    reactions: [rx('hit', 'op', 0, 'amp', 0.6, { rel: 0.3 }), rx('bass', 'op', 1, 'amt', 0.4, { atk: 0.03, rel: 0.5 }), rx('beat', 'ma', 0, 'gain', 0.3, { rel: 0.2 })],
  },
  {
    // NemoOrange, the Light of Speed ('long-exposure photographs of highways and cities'): a
    // handful of tiny tight coils in warm yellows and reds wander on slow looping paths while the
    // picture streams outward from a drifting centre under a long fade, so each coil draws a light
    // trail; the trails meet in a maximum blend (no burn-out) and the beat nudges the flow sideways.
    origin: 'A02', name: 'the Light of Speed (after NemoOrange)', energy: [0.25, 0.8], scheme: 'analogous', hue: 0.06,
    color: { sat: 0.9, adapt: 0.3, bloom: 1.3, vignette: 0.45 }, carrier: 'warp', car: { halfLife: 1.4, blur: 0.2 },
    chain: [op('zoom', { rate: 0.035, wander: 0.2 }), op('noise', { amp: 0.0006, scale: 1.5, speed: 0.3 }), op('translate', { vx: 0, vy: 0 })],
    bodies: [body({
      shape: ['curve', { form: 2, radius: 0.035, turns: 6, amp: 0.1 }],
      place: ['float', { count: 6, spread: 0.6, speed: 0.1 }],
      motion: ['spin', { rate: 0.5 }],
      material: ['line', { gain: 0.9, width: 2, halo: 0.25, blend: 1 }],
      color: ['instrument', { hue: 0, amount: 0.35 }],
    })],
    reactions: [rx('beat', 'op', 2, 'vx', 0.25, { rel: 0.4 }), rx('hit', 'op', 2, 'vy', -0.2, { rel: 0.4 }), rx('loud', 'op', 0, 'rate', 0.3, { atk: 0.05, rel: 0.6 })],
  },
  {
    // Yathosho (Jan T. Sott, movement by David Hansen), sakura: three soft blobs in blue, pink and
    // red swell on the beat and are copied into a turning ring of offset layers, then folded into a
    // six-petal blossom whose radius ripples in concentric bands, over water, all washed pastel.
    origin: 'A04', name: 'sakura (after Yathosho)', energy: [0.2, 0.75], scheme: 'triad', hue: 0.92,
    color: { sat: 0.5, exposure: 1.1, adapt: 0.35, bloom: 1.2, vignette: 0.3 }, carrier: 'warp', car: { halfLife: 0.7, water: 0.6, wsize: 0.04 },
    chain: [op('ripple', { amp: 0.004, freq: 9, speed: 1, radial: 1 }), op('rotate', { lock: 0.125 }), op('zoom', { rate: 0.004 }), op('kaleido', { n: 6, lock: -0.0625 }, 1, 'view')],
    bodies: [body({
      shape: ['dot', { r: 0.006 }],
      place: ['ring', { n: 5, radius: 0.13 }],
      motion: ['pulse', { amp: 0.4 }],
      material: ['glow', { gain: 0.45, width: 0.022, base: 0.15 }],
      color: ['instrument', { hue: 0, amount: 1 }],
    })],
    reactions: [rx('beat', 'pl', 0, 'radius', 0.35, { rel: 0.35 }), rx('bass', 'car', 0, 'water', 0.3, { atk: 0.03, rel: 0.5 }), rx('vocals', 'op', 0, 'amp', 0.4, { atk: 0.1, rel: 0.6 })],
  },
  {
    // Zevensoft, Ocean4: fireworks over a night sea. Spherical bursts of random-coloured sparks
    // blossom and fall above a rolling horizon line that follows the melody, and the sea below
    // mirrors it all; the hits set off the bursts and the bass lifts the swell.
    origin: 'A05', name: 'Ocean4 (after Zevensoft)', energy: [0.3, 0.9], scheme: 'complementary', hue: 0.62,
    color: { sat: 0.85, adapt: 0.35, bloom: 1.3, vignette: 0.4, reflect: 1, reflectY: -0.22 }, carrier: 'warp', car: { halfLife: 0.4, blur: 0.1 },
    bodies: [
      body({
        shape: ['edge', { mode: 1, side: 2, base: -0.22, height: 0.12, density: 0.6 }],
        material: ['fill', { gain: 1 }],
      }),
      body({
        shape: ['dot', { r: 0.004 }],
        place: ['point', { x: 0, y: 0 }],
        material: ['glow', { gain: 0.6, width: 0.006 }],
        emit: ['sparks', { count: 16384, size: 2.2, speed: 0.7, curl: 0.05, zoomFlow: 0, lift: -0.06, drag: 3, life: 0.55, spread: 0.12, surge: 1, top: 0, body: 0.1 }],
        color: ['age', { hue: 0.5, rate: 0.25, detail: 1 }],
      }),
    ],
    reactions: [rx('hit', 'em', 1, 'speed', 0.5, { rel: 0.3 }), rx('bass', 'sh', 0, 'height', 0.4, { atk: 0.05, rel: 0.5 }), rx('loud', 'ma', 1, 'gain', 0.3, { atk: 0.05, rel: 0.4 })],
  },
  {
    // El-vis, hubble002: spiral galaxies of a thousand spectrum-pushed dots, re-coloured and moved
    // on the beat, drift through a blue starfield while the feedback smears their arms into a
    // blurred nebula that slowly twists and draws you in.
    origin: 'A06', name: 'hubble002 (after El-vis)', energy: [0.2, 0.75], scheme: 'triad', hue: 0.6,
    color: { sat: 0.8, adapt: 0.3, bloom: 1.3, vignette: 0.5 }, carrier: 'warp', car: { halfLife: 0.8, blur: 0.3 },
    chain: [op('zoom', { rate: 0.006 }), op('twist', { amt: 0.004 }), op('noise', { amp: 0.0006, scale: 2, speed: 0.2 })],
    bodies: [
      body({
        shape: ['curve', { form: 2, radius: 0.14, turns: 6, amp: 0.4 }],
        place: ['float', { count: 3, spread: 0.45, speed: 0.06 }],
        motion: ['spin', { rate: 0.25 }],
        material: ['dots', { gain: 0.8, spacing: 0.01, size: 0.5 }],
        color: ['pitch', { hue: 0, detail: 1 }],
      }),
      body({
        shape: ['dot', { r: 0.002 }],
        material: ['glow', { gain: 0.3, width: 0.003 }],
        emit: ['sparks', { count: 8192, size: 1.6, speed: 0.2, curl: 0, zoomFlow: 1.3, drag: 1.5, life: 0.5, spread: 0.2, surge: 0.3, top: 0, body: 0 }],
        color: ['fixed', { hue: 0.33, detail: 0.5 }],
      }),
    ],
    reactions: [rx('bass', 'sh', 0, 'amp', 0.4, { atk: 0.03, rel: 0.4 }), rx('beat', 'pl', 0, 'spread', 0.2, { rel: 0.5 }), rx('other', 'op', 1, 'amt', 0.3, { atk: 0.1, rel: 0.8 })],
  },
  {
    // Degnic, Fractal (slo-mo metallic): a spectrum-pulsed ring echoes down a slowly turning,
    // sinking feedback tunnel, blurred and sharpened into nested rings, and the whole picture is lit
    // as brushed metal; onsets kick the turn and the bass pulls the tunnel in faster.
    origin: 'A08', name: 'Fractal slo-mo metallic (after Degnic)', energy: [0.1, 0.6], scheme: 'mono', hue: 0.58,
    color: { sat: 0.15, exposure: 1.05, adapt: 0.3, bloom: 0.8, vignette: 0.45, relief: 0.8, bump: 1.6, light: 0.3, gloss: 0.8, metal: 0.9 },
    carrier: 'warp', car: { halfLife: 1.2, blur: 0.2, sharpen: 0.05, grain: 0.004 },
    chain: [op('zoom', { rate: -0.008, wander: 0.15 }), op('rotate', { lock: 0.0625, rate: 0.002 }), op('mirror', { axis: 2 }, 1, 'view')],
    bodies: [body({
      shape: ['curve', { form: 1, radius: 0.22, amp: 0.3 }],
      material: ['line', { gain: 0.9, width: 1.6, halo: 0.2 }],
      feel: ['flow', { atk: 0.1, rel: 1 }],
    })],
    reactions: [rx('hit', 'op', 1, 'rate', 0.4, { rel: 0.8 }), rx('bass', 'op', 0, 'rate', -0.3, { atk: 0.1, rel: 1 }), rx('vocals', 'sh', 0, 'radius', 0.2, { atk: 0.1, rel: 0.6 })],
  },
  {
    // Jheriko, Alien Device (gallery remix by Zamuz): an icosahedron of pentagons turns in space
    // on the bar, interlaced line by line into the trail so the device flickers like an old
    // monitor, while the camera shakes on the hits and the bass swells the machine.
    origin: 'A09', name: 'Alien Device (after Jheriko & Zamuz)', energy: [0.3, 0.85], scheme: 'complementary', hue: 0.35,
    color: { sat: 0.85, adapt: 0.35, bloom: 1.2, vignette: 0.5, contrast: 0.05 }, carrier: 'warp', car: { halfLife: 0.3, blur: 0.1 },
    chain: [op('noise', { amp: 0.0008, scale: 1.2, speed: 0.6 }), op('zoom', { rate: 0.003 })],
    bodies: [body({
      shape: ['solid', { solid: 3, size: 0.3, tilt: 0.5, inner: 0 }],
      motion: ['spin', { rate: 0.25 }],
      material: ['line', { gain: 1.1, width: 1.8, halo: 0.2, blend: 5 }],
      color: ['height', { hue: 0, amount: 0.6 }],
    })],
    reactions: [rx('hit', 'op', 0, 'amp', 0.7, { rel: 0.25 }), rx('bass', 'sh', 0, 'size', 0.3, { atk: 0.03, rel: 0.4 }), rx('loud', 'ma', 0, 'gain', 0.3, { atk: 0.05, rel: 0.4 })],
  },
];

// R01-: ray-marched 3D scenes (the 'scene' shape, genes/raymarch.ts), fed through the same material,
// palette, carrier and reactions as every other body.
const RAYMARCH: Def[] = [
  {
    // Five primitives (spheres, tori, boxes, octahedra) melt into each other under an orbiting camera;
    // the bass swells them, drum hits jolt the camera in, each section reshuffles the arrangement, and a
    // slow zoom in the feedback leaves soft trails behind the moving surfaces.
    origin: 'R01', name: 'Melting Forms', energy: [0.2, 0.8], scheme: 'triad', hue: 0.6,
    color: { adapt: 0.35, bloom: 1.1 }, carrier: 'warp', decay: 0.9,
    chain: [op('zoom', { rate: 0.004 })],
    bodies: [body({
      shape: ['scene', { scene: 0, cam: 0, res: 0.7, size: 1, blend: 0.55, speed: 0.4, pulse: 0.5, kick: 0.5, vary: 1, rim: 0.6, ao: 0.7, fog: 0.4, glow: 0.3 }],
      material: ['fill', { gain: 1.2 }],
      color: ['fixed', { hue: 0, detail: 1 }],
    })],
    reactions: [
      rx('bass', 'sh', 0, 'size', 0.15, { atk: 0.02, rel: 0.4 }),
      rx('drums', 'sh', 0, 'kick', 0.3, { atk: 0.01, rel: 0.2 }),
      rx('section', 'sh', 0, 'vary', 0.3, { atk: 0.05, rel: 1.5 }),
    ],
  },
  {
    // The camera weaves between the melting shapes, banking into its turns and lunging forward on drum
    // hits; only the rims and the glow of near misses are lit, so the shapes read as neon outlines that
    // smear into long trails.
    origin: 'R02', name: 'Neon Passage', energy: [0.4, 1], scheme: 'split', hue: 0.8,
    color: { adapt: 0.3, bloom: 1.3 }, carrier: 'warp', decay: 0.95,
    bodies: [body({
      shape: ['scene', { scene: 0, cam: 1, res: 0.5, size: 0.9, blend: 0.35, speed: 0.55, pulse: 0.6, kick: 0.8, vary: 1, rim: 1, ao: 0.4, fog: 0.5, glow: 0.7, roam: 0.4 }],
      material: ['line', { gain: 1.1 }],
      color: ['age', { rate: 0.125, detail: 1 }],
    })],
    reactions: [
      rx('drums', 'sh', 0, 'kick', 0.25, { atk: 0.01, rel: 0.15 }),
      rx('bass', 'sh', 0, 'glow', 0.3, { atk: 0.02, rel: 0.3 }),
      rx('section', 'sh', 0, 'vary', 0.3, { atk: 0.05, rel: 1.5 }),
    ],
  },
  {
    // A dolly zoom on a chrome cluster: every four bars the camera pulls back while the lens zooms in,
    // so the shapes hold their size and the space around them stretches; drum hits punch it in and the
    // bass swells the melted chrome.
    origin: 'R03', name: 'Vertigo Chrome', energy: [0.15, 0.7], scheme: 'complementary', hue: 0.1,
    color: { adapt: 0.4, bloom: 1.1 }, carrier: 'warp', decay: 0.85,
    bodies: [body({
      shape: ['scene', { scene: 0, cam: 2, res: 0.7, size: 1.1, blend: 0.85, speed: 0.3, pulse: 0.7, kick: 0.5, vary: 0.8, rim: 0.7, ao: 0.8, fog: 0.3, glow: 0.15, roam: 0.8 }],
      material: ['chrome', { gain: 1.1, chrome: 1 }],
      color: ['fixed', { hue: 0, detail: 0.6 }],
    })],
    reactions: [
      rx('bass', 'sh', 0, 'size', 0.12, { atk: 0.02, rel: 0.4 }),
      rx('drums', 'sh', 0, 'kick', 0.3, { atk: 0.01, rel: 0.2 }),
      rx('section', 'sh', 0, 'vary', 0.3, { atk: 0.05, rel: 1.5 }),
    ],
  },
  {
    // Cruising down an endless corridor between two plates of shapes, one per cell, each rising with its
    // own band of the spectrum like a 3D equaliser city; the camera weaves and lunges on drum hits, each
    // section rebuilds the city, and fog swallows the far end.
    origin: 'R04', name: 'Resonant Grid', energy: [0.35, 1], scheme: 'analogous', hue: 0.5,
    color: { adapt: 0.35, bloom: 1.2 }, carrier: 'warp', decay: 0.7,
    bodies: [body({
      shape: ['scene', { scene: 1, cam: 1, res: 0.5, size: 0.9, blend: 0.3, speed: 0.5, pulse: 0.5, kick: 0.6, vary: 1, rim: 0.6, ao: 0.8, fog: 0.5, glow: 0.25, roam: 0.3, gap: 0.3, spec: 0.9 }],
      material: ['fill', { gain: 1.2 }],
      color: ['fixed', { hue: 0, detail: 1 }],
    })],
    reactions: [
      rx('drums', 'sh', 0, 'kick', 0.25, { atk: 0.01, rel: 0.2 }),
      rx('loud', 'sh', 0, 'spec', 0.2, { atk: 0.05, rel: 0.4 }),
      rx('section', 'sh', 0, 'vary', 0.3, { atk: 0.05, rel: 1.5 }),
    ],
  },
  {
    // Racing down a bending, ribbed tunnel: every rib closes in with its own band of the spectrum, the
    // wall ripples with the bass, drum hits surge the ride forward, and each section re-routes the bends.
    origin: 'R05', name: 'Wormhole Run', energy: [0.45, 1], scheme: 'triad', hue: 0.95,
    color: { adapt: 0.3, bloom: 1.25 }, carrier: 'warp', decay: 0.8,
    bodies: [body({
      shape: ['scene', { scene: 2, cam: 1, res: 0.5, size: 1, blend: 0.3, speed: 0.45, pulse: 0.8, kick: 0.7, vary: 1, rim: 0.7, ao: 0.6, fog: 0.55, glow: 0.4, roam: 0.6, gap: 0.5, spec: 0.8 }],
      material: ['glow', { gain: 1.1 }],
      color: ['age', { rate: 0.0625, detail: 1 }],
    })],
    reactions: [
      rx('drums', 'sh', 0, 'kick', 0.25, { atk: 0.01, rel: 0.2 }),
      rx('bass', 'sh', 0, 'pulse', 0.2, { atk: 0.02, rel: 0.3 }),
      rx('section', 'sh', 0, 'vary', 0.3, { atk: 0.05, rel: 1.5 }),
    ],
  },
];

// P01.. showcase the agent simulations (src/v2/genes/physarum.ts): physarum slime growing vein networks.
const AGENTS: Def[] = [
  {
    // Half a million slime-mould agents sense each other's trail and grow a meandering vein network that
    // keeps re-routing itself; the bass makes them steer harder, drum hits lay down bright bursts of
    // trail, the vocals soften it, and four faint instrument lights drift through it, each feeding the
    // trail and giving birth to fresh agents, so the veins thicken around them and stream out of them.
    // On a drop the network dissolves into a haze and regrows.
    origin: 'P01', name: 'Physarum Bloom', energy: [0.2, 0.8], scheme: 'analogous', hue: 0.3,
    color: { adapt: 0.35, bloom: 1.1, vignette: 0.4, contrast: 0.05 }, carrier: 'warp', car: { halfLife: 0.25, floor: 1 },
    bodies: [body({
      shape: ['dot', { r: 0.004 }],
      place: ['stations', { count: 4, inst: 1, xs: 0.8, wander: 0.15 }],
      material: ['glow', { gain: 1, width: 0.012 }],
      emit: ['slime', { count: 524288, sa: 0.6, sd: 0.008, steer: 0.3, step: 0.001, deposit: 0.3, decay: 0.93, diffuse: 0.5, body: 0.35, feed: 0.5, birth: 0.05, onDrop: 1 }],
      feel: ['flow', { atk: 0.02, rel: 0.3 }],
      color: ['height', { amount: 0.5 }],
    })],
    reactions: [
      rx('bass', 'em', 0, 'steer', 0.15, { atk: 0.05, rel: 0.4 }), rx('hit', 'em', 0, 'deposit', 0.6, { rel: 0.3 }),
      rx('vocals', 'em', 0, 'diffuse', 0.3, { atk: 0.1, rel: 0.8 }), rx('loud', 'ma', 0, 'gain', 0.3, { atk: 0.05, rel: 0.3 }),
    ],
  },
  {
    // A slowly turning five-point star feeds the trail and gives birth to agents inside itself, so a
    // mycelium blossoms out of the shape: fine radial veins inside, looping runners around it. Drum hits
    // release bursts of new agents, the bass drives them faster, the vocals make the star feed harder;
    // on a drop every agent bursts out of the star at once.
    origin: 'P02', name: 'Mycelial Star', energy: [0.25, 0.85], scheme: 'triad', hue: 0.1,
    color: { adapt: 0.35, bloom: 1.1, vignette: 0.45, contrast: 0.05 }, carrier: 'warp', car: { halfLife: 0.3, floor: 1 },
    bodies: [body({
      shape: ['star', { n: 5, r: 0.3, inner: 0.45 }],
      motion: ['spin', { rate: 0.0625 }],
      material: ['line', { gain: 1, width: 2, halo: 0.2 }],
      emit: ['slime', { count: 393216, sa: 0.5, sd: 0.01, steer: 0.35, step: 0.0012, deposit: 0.3, decay: 0.93, diffuse: 0.5, body: 0.7, feed: 1, birth: 0.15, onDrop: 2 }],
      color: ['age', { rate: 0.0625, detail: 0.6 }],
    })],
    reactions: [
      rx('hit', 'em', 0, 'birth', 0.8, { rel: 0.25 }), rx('bass', 'em', 0, 'step', 0.3, { atk: 0.05, rel: 0.4 }),
      rx('vocals', 'em', 0, 'feed', 0.4, { atk: 0.1, rel: 0.8 }), rx('loud', 'ma', 0, 'gain', 0.3, { atk: 0.05, rel: 0.3 }),
    ],
  },
  {
    // A murmuration: sixteen thousand boids wheel around three slowly orbiting lights in streaking
    // flocks that merge and split; each bird is coloured by its heading, so turning banks change colour.
    // The bass drives the flock faster, drum hits scatter it, the vocals pull it into long ribbons, and
    // on a drop the whole flock bursts outward before regrouping.
    origin: 'P03', name: 'Murmuration', energy: [0.25, 0.9], scheme: 'complementary', hue: 0.6,
    color: { adapt: 0.35, bloom: 1.1, vignette: 0.45 }, carrier: 'warp', car: { halfLife: 0.15, floor: 1 },
    chain: [op('zoom', { rate: 0.002 })],
    bodies: [body({
      shape: ['dot', { r: 0.004 }],
      place: ['orbit', { count: 3, radius: 0.28, rate: 0.125, follow: 0.1 }],
      material: ['glow', { gain: 0.5, width: 0.01 }],
      emit: ['flock', { count: 16384, speed: 0.25, radius: 0.025, align: 0.8, cohere: 0.8, separate: 0.35, wander: 0.15, home: 0.35, size: 1.6, body: 0.3, onDrop: 1 }],
      feel: ['flow', { atk: 0.02, rel: 0.3 }],
    })],
    reactions: [
      rx('bass', 'em', 0, 'speed', 0.4, { atk: 0.05, rel: 0.4 }), rx('hit', 'em', 0, 'separate', 0.7, { rel: 0.3 }),
      rx('vocals', 'em', 0, 'align', 0.3, { atk: 0.1, rel: 0.8 }), rx('loud', 'ma', 0, 'gain', 0.3, { atk: 0.05, rel: 0.3 }),
    ],
  },
];

// B01.. showcase the stem ecosystem (src/v2/genes/ecosystem.ts): each instrument is a species of agent,
// and the balance of the mix decides who thrives.
const ECOSYSTEM: Def[] = [
  {
    // A savanna food web: streaking drum predators hunt the drifting plankton of the other instruments,
    // fat bass grazers lumber after the flora, leaving trails the ring-shaped vocal pollinators follow,
    // and the meadow blooms wherever they pass. A vocal bridge fills the screen with flowers and
    // pollinators, a drop is a feast for the predators, and a silent stem lets its species starve away.
    origin: 'B01', name: 'Stem Savanna', energy: [0.2, 0.85], scheme: 'triad', hue: 0.12,
    color: { adapt: 0.35, bloom: 1, vignette: 0.4, contrast: 0.05 }, carrier: 'warp', car: { halfLife: 0.15, floor: 1 },
    bodies: [body({
      shape: ['dot', { r: 0.01 }],
      material: ['glow', { gain: 1, width: 0.01 }],
      emit: ['ecosystem', { count: 16384, wD: 0.35, wB: 0.3, wV: 0.45, wO: 0.9, glyph: 0, size: 3, trail: 0.6, predation: 0.6, bloom: 0.6, graze: 0.5, growth: 0.7, starve: 0.4, decay: 0.985, speed: 1, field: 0.6, hues: 0, body: 0 }],
    })],
    reactions: [rx('loud', 'ma', 0, 'gain', 0.25, { atk: 0.05, rel: 0.3 }), rx('drop', 'em', 0, 'speed', 0.4, { atk: 0.02, rel: 1 })],
  },
];

export const SEEDS: Seed[] = [...DEFS, ...MILKDROP, ...CHOREO, ...PHYSICS, ...AVS, ...RAYMARCH, ...AGENTS, ...ECOSYSTEM].map(build);
