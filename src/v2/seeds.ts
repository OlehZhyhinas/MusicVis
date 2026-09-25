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
import { DRIFT_SCHEMA } from './genes/drift';
import { HARMONY_SCHEMA } from './genes/harmony';
import { ACCENT_SCHEMA } from './genes/accent';
import { GROOVE_SCHEMA } from './genes/groove';
import { TIMBRE_SCHEMA } from './genes/timbre';
import { DEJAVU_SCHEMA } from './genes/dejavu';
import { LYRICS_SCHEMA } from './genes/lyrics';

export const SEED_VERSION = 105;

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
  /** Drift through gene space over the song (src/v2/genes/drift.ts); omitted = none. */
  drift?: Record<string, number>;
  /** Harmony: symmetry that follows the chord progression (src/v2/genes/harmony.ts); omitted = none. */
  harmony?: Record<string, number>;
  /** Timing feel for the motion (src/v2/genes/groove.ts); omitted = none. */
  groove?: Record<string, number>;
  /** The sound's timbre as the bodies' material (src/v2/genes/timbre.ts); omitted = none. */
  timbre?: Record<string, number>;
  /** Visual deja vu: returning sections recall their first appearance (src/v2/genes/dejavu.ts); omitted = none. */
  dejavu?: Record<string, number>;
  /** Built-in accents (hook gesture, section response): only set to tune them down; omitted = the defaults (src/v2/genes/accent.ts). */
  accent?: Record<string, number>;
  /** What the sung words are about steers the picture, and the line is shown (src/v2/genes/lyrics.ts); omitted = none. */
  lyrics?: Record<string, number>;
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
  if (d.drift) g.drift = { p: { ...defaultParams(DRIFT_SCHEMA), ...d.drift } };
  if (d.harmony) g.harmony = { p: { ...defaultParams(HARMONY_SCHEMA), ...d.harmony } };
  if (d.groove) g.groove = { p: { ...defaultParams(GROOVE_SCHEMA), ...d.groove } };
  if (d.timbre) g.timbre = { p: { ...defaultParams(TIMBRE_SCHEMA), ...d.timbre } };
  if (d.dejavu) g.dejavu = { p: { ...defaultParams(DEJAVU_SCHEMA), ...d.dejavu } };
  if (d.accent) g.accent = { p: { ...defaultParams(ACCENT_SCHEMA), ...d.accent } };
  if (d.lyrics) g.lyrics = { p: { ...defaultParams(LYRICS_SCHEMA), ...d.lyrics } };
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
    // Rain born on the eighth-note grid, fuller on each downbeat; columns fall at 1x or 2x speed, dim
    // and slow enough that the drops never strobe on fast songs.
    origin: 'E03', name: 'Rain Curtains', energy: [0.4, 0.95], scheme: 'analogous', hue: 0.5,
    color: { bloom: 1.1 }, carrier: 'warp', decay: 0.975,
    chain: [op('translate', { vy: -0.22, lanes: 38 })],
    bodies: [body({ shape: ['edge', { mode: 2, side: 1, density: 0.6 }], material: ['fill', { gain: 0.45 }] })],
    // Its thin streaks flash when magnified or brightened, so the hook gesture is kept small and the
    // built-in hit kick is off (strobe margin).
    accent: { hook: 0.15, kick: 0 },
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
    reactions: [rx('bass', 'op', 0, 'rate', -0.08, { atk: 0.03, rel: 0.3 }), rx('beat', 'op', 0, 'rate', -0.132, { atk: 0.03, rel: 0.3 })],
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
    reactions: [rx('beat', 'op', 0, 'rate', 0.12, { atk: 0.03, rel: 0.3 })],
  },
  {
    // Sparks from the centre; warp speed surges on every beat (fast attack, slow ease), cruises with the
    // loudness and jumps on a drop. (A zoom-rate surge on top pushed the streaks past the strobe limit.)
    origin: 'E12', name: 'Warp Speed', energy: [0.55, 1], scheme: 'analogous', hue: 0.58,
    color: { sat: 0.35, bloom: 1.1 }, carrier: 'warp', decay: 0.86,
    chain: [op('rotate', { lock: 0.0625 }), op('zoom', { rate: 0.0082 })],
    bodies: [body({
      shape: ['dot', { r: 0.004 }],
      place: ['point'],
      material: ['glow', { gain: 1.57 }],
      emit: ['sparks', { count: 5120, size: 3.4, speed: 0.1, curl: 0, life: 0.2, zoomFlow: 1.1, drag: 4, spread: 0.12, surge: 1, top: 0, body: 0 }],
    })],
    reactions: [rx('surge', 'ma', 0, 'gain', 0.2, { atk: 0.02, rel: 0.3 })],
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
    // Aurora curtains over a dark horizon, bright enough to always read (it used to sit near black),
    // flaring on the beat, their waves swelling with the bass.
    origin: 'E16', name: 'Aurora', energy: [0.1, 0.6], scheme: 'analogous', hue: 0.35,
    color: { adapt: 0.3 }, carrier: 'warp', decay: 0.965, car: { blur: 0.1, floor: 0.5 },
    chain: [op('noise', { amp: 0.0006, scale: 1.5, speed: 0.5 }), op('translate', { vy: 0.03 })],
    bodies: [body({ shape: ['aurora', { fall: 5, rays: 24, wav: 1.4 }], place: ['point', { y: 0 }], material: ['glow', { gain: 2.4 }] })],
    reactions: [rx('beat', 'ma', 0, 'gain', 0.3, { atk: 0.01, rel: 0.3 }), rx('bass', 'sh', 0, 'wav', 0.35, { atk: 0.03, rel: 0.4 })],
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
    reactions: [rx('bass', 'op', 0, 'rate', 0.2, { atk: 0.03, rel: 0.3 }), rx('beat', 'op', 0, 'rate', 0.32, { atk: 0.03, rel: 0.3 }), rx('build', 'op', 0, 'rate', 0.2, { atk: 0.03, rel: 0.3 })],
  },
  {
    // A waveform arc folded eight ways into a flower, zooming out and turning; it flares on the beat,
    // drum hits throw its petals outward and the bass roughens the waveform. The arc and the fold
    // stay put so the flower never slips out of the kaleidoscope's wedge and leaves the screen black.
    origin: 'E18', name: 'Mandala', energy: [0.5, 1], scheme: 'triad', hue: 0.1,
    color: { bloom: 1.1 }, carrier: 'warp', decay: 0.94,
    chain: [op('rotate', { lock: 0.125 }), op('zoom', { rate: 0.006 }), op('kaleido', { n: 8, lock: 0 }, 1, 'view')],
    bodies: [body({
      shape: ['curve', { form: 4, radius: 0.27, amp: 0.25, turns: 3 }],
      material: ['line', { gain: 0.8, width: 1.8 }],
    })],
    reactions: [rx('beat', 'ma', 0, 'gain', 0.6, { rel: 0.2 }), rx('bass', 'sh', 0, 'amp', 0.4, { atk: 0.02, rel: 0.25 }), rx('hit', 'sh', 0, 'radius', 0.25, { rel: 0.2 })],
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
    // with a gentle warp; drum hits jolt the whole field sideways and the dust colour drifts. The
    // stream runs a little slower than the original's zoom 1.05, which strobed on fast songs.
    origin: 'M01', name: 'Cosmic Dust 2 (after Geiss)', energy: [0.35, 0.95], scheme: 'analogous', hue: 0.72,
    color: { sat: 0.45, adapt: 0.2, bloom: 1, vignette: 0.35, contrast: 0.06 }, carrier: 'warp', decay: 0.96,
    chain: [
      op('zoom', { rate: 0.03, wander: 0.3 }),
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
      rx('hit', 'op', 3, 'vx', 0.3, { rel: 0.35 }), rx('bass', 'op', 3, 'vy', -0.3, { atk: 0.02, rel: 0.3, thr: 0.5 }),
      rx('surge', 'op', 0, 'rate', 0.05, { atk: 0.03, rel: 0.3 }), rx('loud', 'em', 0, 'speed', 0.4, { atk: 0.02, rel: 0.2 }),
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
    reactions: [rx('bass', 'op', 1, 'rate', 0.14, { atk: 0.05, rel: 0.8 }), rx('other', 'car', 0, 'sharpen', 0.1, { atk: 0.1, rel: 0.5 })],
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
    // The pens wrap round the edges instead of sliding along them, and their curls linger (a long,
    // low-floor trail) so the scribble fills the frame.
    origin: 'M04', name: 'Witchcraft (after fiShbRaiN)', energy: [0.2, 0.75], scheme: 'triad', hue: 0.8,
    color: { sat: 0.8, adapt: 0.3, bloom: 1.1 }, carrier: 'warp', car: { halfLife: 2.5, floor: 0.3 },
    chain: [
      op('zoom', { rate: -0.0005 }),
      op('noise', { amp: 0.0016, scale: 2.2, speed: 0.3 }),
      op('swirl', { amt: 0.008, k: 6, wander: 0.25 }),
      op('mirror', { axis: 0 }, 1, 'view'),
    ],
    bodies: [
      body({
        shape: ['dot', { r: 0.004 }],
        place: ['walker', { heads: 2, step: 0.13, every: 1, square: 0, wrap: 1, curve: 2, turn: 1.5 }],
        motion: ['hits', { amt: 0.8 }],
        material: ['glow', { gain: 1.3, width: 0.008 }],
        emit: ['trail', { tip: 0.3 }],
        feel: ['flow', { atk: 0.01, rel: 0.15 }],
      }),
      body({
        shape: ['dot', { r: 0.004 }],
        place: ['walker', { heads: 2, step: 0.09, every: 2, square: 0, wrap: 1, curve: 1.2, turn: 1.5 }],
        motion: ['hits', { amt: 0.8 }],
        material: ['glow', { gain: 1.3, width: 0.008 }],
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
    reactions: [rx('bass', 'op', 0, 'rate', 0.2, { atk: 0.03, rel: 0.35, thr: 0.35 }), rx('hit', 'car', 0, 'sharpen', 0.25, { rel: 0.3 })],
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
    reactions: [rx('bass', 'op', 1, 'amt', -0.35, { atk: 0.05, rel: 0.6 }), rx('bass', 'op', 2, 'amt', -0.2, { atk: 0.05, rel: 0.6 }), rx('surge', 'op', 0, 'rate', 0.06, { atk: 0.03, rel: 0.3 })],
  },
  {
    // Krash + Rovastar, Rainbow Orb: a small circular waveform, shifting sideways with the loudness, is
    // spun by a rotation strongest at the centre and streamed outward by a fast zoom, so its colour
    // history lays down rainbow rings; the echo mirrors it. (A treble hold on the zoom rate pushed the
    // rings past the strobe limit on fast songs, so the zoom runs free.)
    origin: 'M09', name: 'Rainbow Orb (after Krash & Rovastar)', energy: [0.4, 1], scheme: 'triad', hue: 0,
    color: { sat: 1, adapt: 0.35, bloom: 1.2 }, carrier: 'warp', decay: 0.975,
    chain: [
      op('zoom', { rate: 0.03 }),
      op('swirl', { amt: 0.01, k: 1 }),
      op('swirl', { amt: 0.01, k: 1 }),
      op('swirl', { amt: 0.01, k: 1 }),
      op('mirror', { axis: 0 }, 1, 'view'),
    ],
    bodies: [body({
      shape: ['curve', { form: 1, radius: 0.08, amp: 0.3 }],
      material: ['line', { gain: 1, width: 2 }],
      color: ['age', { rate: 0.5, detail: 1 }],
    })],
    reactions: [
      rx('loud', 'pl', 0, 'x', 0.25, { atk: 0.05, rel: 0.3 }),
      rx('bass', 'op', 1, 'amt', -0.07, { atk: 0.05, rel: 0.4 }),
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
    reactions: [rx('bass', 'op', 0, 'rate', 0.06, { atk: 0.03, rel: 0.3 }), rx('bass', 'ma', 1, 'gain', 0.4, { atk: 0.01, rel: 0.25, thr: 0.5 })],
  },
  {
    // martin, tunnel race: waveform bands scroll up the carried picture, which is wrapped onto the wall
    // of a tunnel coming toward you as rings, surging and deepening on every beat and turning with the
    // bar, with an orb racing round the wall; the far end is lost in haze. The bass speeds the flight
    // and drum hits flash the bands and the orb. Kept slow and dim enough that the passing rings never
    // strobe (under 3 flashes a second).
    origin: 'M16', name: 'Tunnel Race (after martin)', energy: [0.35, 0.95], scheme: 'analogous', hue: 0.85,
    color: { sat: 0.8, exposure: 0.95, adapt: 0.25, bloom: 1.1, vignette: 0.55 },
    carrier: 'warp', car: { halfLife: 0.6, blur: 0.05 },
    chain: [
      op('translate', { vx: 0, vy: 0.05 }),
      op('tunnel', { depth: 0.16, speed: 0.15, twist: 0.1, sides: 0, rep: 1, fog: 0.8, lock: 0.0625 }, 1, 'view'),
    ],
    bodies: [
      body({
        shape: ['curve', { form: 0, amp: 0.2 }],
        place: ['point', { x: 0, y: -0.4 }],
        material: ['line', { gain: 0.6, width: 2, halo: 0.2 }],
        color: ['age', { rate: 0.25, detail: 0.5 }],
      }),
      body({
        shape: ['dot', { r: 0.04 }],
        place: ['orbit', { count: 1, radius: 0.3, rate: 0.5 }],
        material: ['glow', { gain: 1.2, width: 0.03 }],
        emit: ['none'],
      }),
    ],
    reactions: [
      rx('bass', 'op', 1, 'speed', 0.3, { atk: 0.03, rel: 0.4 }), rx('hit', 'ma', 1, 'gain', 0.6, { rel: 0.2 }),
      rx('hit', 'ma', 0, 'gain', 0.7, { rel: 0.2 }), rx('beat', 'op', 1, 'depth', 0.3, { rel: 0.25 }),
    ],
  },
  {
    // Flexi + Martin, tunnel of supraschismatika: a dark chrome pipe flown through, glints streaking
    // along its polished wall toward you; the bass drives the flight and the twist, the pipe deepens on
    // every beat and the glints flash on drum hits. Flown slower than the original so the passing
    // rings never strobe.
    origin: 'M17', name: 'Tunnel of Supraschismatika (after Flexi & Martin)', energy: [0.3, 0.9], scheme: 'mono', hue: 0.6,
    color: { sat: 0.25, exposure: 0.9, adapt: 0.3, bloom: 1.2, vignette: 0.35, relief: 0.7, bump: 1.6, light: 0.25, gloss: 1, metal: 0.6 },
    carrier: 'warp', decay: 0.9,
    chain: [
      op('translate', { vx: 0, vy: -0.08 }),
      op('tunnel', { depth: 0.22, speed: 0.15, twist: -0.35, sides: 0, rep: 2, fog: 0.9, lock: 0 }, 1, 'view'),
    ],
    bodies: [body({
      shape: ['dot', { r: 0.01 }],
      place: ['float', { count: 6, spread: 0.6, speed: 0.2 }],
      material: ['glow', { gain: 1.6, width: 0.012, base: 0.4 }],
      emit: ['trail'],
    })],
    reactions: [rx('bass', 'op', 1, 'speed', 0.2, { atk: 0.03, rel: 0.4 }), rx('bass', 'op', 1, 'twist', -0.25, { atk: 0.05, rel: 0.6 }), rx('beat', 'op', 1, 'depth', 0.25, { rel: 0.25 }), rx('hit', 'ma', 0, 'gain', 0.5, { rel: 0.2 })],
  },
  {
    // Waltra, Square Orgy: a turning grid of glossy tiles, each lit by the colour behind it, as bright
    // blobs drift and bloom underneath; the grid swells with the bass and the tiles shine like enamel.
    origin: 'M18', name: 'Square Orgy (after Waltra)', energy: [0.3, 0.9], scheme: 'triad', hue: 0.08,
    color: { sat: 1, exposure: 1.05, adapt: 0.3, bloom: 0.8, vignette: 0.2, relief: 0.6, bump: 1.4, gloss: 0.9, light: 0.3 },
    carrier: 'warp', car: { halfLife: 1.4, floor: 0.2 },
    chain: [
      op('zoom', { rate: 0.006, wander: 0.2 }),
      op('mosaic', { size: 0.13, shape: 0, gap: 0.12, angle: 0.07, lock: 0.0625, pulse: 0.25 }, 1, 'view'),
    ],
    bodies: [body({
      shape: ['dot', { r: 0.1 }],
      place: ['float', { count: 6, spread: 0.7, speed: 0.18 }],
      material: ['fill', { gain: 1.2, soft: 0.5 }],
      color: ['height', { amount: 1.5, detail: 1 }],
    })],
    reactions: [rx('bass', 'op', 1, 'size', 0.12, { atk: 0.05, rel: 0.4 }), rx('beat', 'ma', 0, 'gain', 0.4, { rel: 0.25 })],
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
    reactions: [rx('bass', 'op', 0, 'rate', 0.12, { atk: 0.03, rel: 0.3 }), rx('drums', 'ma', 0, 'gain', 0.5, { atk: 0.01, rel: 0.2 })],
  },
  {
    // Geiss, Tokamak Plus 2: two waveforms are drawn across the screen and a slowly turning strain flow
    // (squeezed toward one axis, pulled from the other) stretches and folds their trails into silky grey
    // filaments, with a little turbulence; the waves swell with the volume and the bass strains harder.
    origin: 'M20', name: 'Tokamak (after Geiss)', energy: [0.2, 0.8], scheme: 'mono', hue: 0.6,
    color: { sat: 0.2, exposure: 1.05, adapt: 0.3, bloom: 0.8, vignette: 0.3, relief: 0.45, bump: 1, gloss: 0.6, metal: 0.3 },
    carrier: 'warp', decay: 0.98,
    chain: [
      op('push', { amt: -0.002, axis: 0 }),
      op('push', { amt: 0.002, axis: 1 }),
      op('rotate', { lock: 0.0625, rate: 0, wander: 0.3 }),
      op('noise', { amp: 0.0015, scale: 2.2, speed: 0.4 }),
    ],
    bodies: [body({
      shape: ['curve', { form: 0, amp: 0.45 }],
      place: ['mirror', { axis: 1, x: 0, y: 0.18 }],
      motion: ['sway', { amp: 0.12, period: 4, tilt: 0.6 }],
      material: ['line', { gain: 0.7, width: 1.2, halo: 0.2 }],
      color: ['age', { rate: 0.0625, detail: 0.3 }],
      feel: ['flow', { atk: 0.02, rel: 0.3 }],
    })],
    reactions: [rx('loud', 'sh', 0, 'amp', 0.3, { atk: 0.03, rel: 0.3 }), rx('bass', 'op', 0, 'amt', -0.2, { atk: 0.05, rel: 0.6 }), rx('bass', 'op', 1, 'amt', 0.2, { atk: 0.05, rel: 0.6 })],
  },
  {
    // martin, mandelbox explorer: the whole carved-stone Mandelbox seen from outside, in one hue with
    // light fog and bright rims, under a slow vertigo dolly every four bars so its depth keeps
    // shifting; the bass pushes the fold scale so the walls shift, drums lunge the camera in, and it
    // brightens on the beat. (The old deep dive spent most of its time with a wall filling the frame.)
    origin: 'M21', name: 'Mandelbox Explorer (after martin)', energy: [0.2, 0.8], scheme: 'mono', hue: 0.35,
    color: { sat: 0.7, exposure: 0.85, contrast: 0.04, adapt: 0.2, bloom: 0.6, vignette: 0.5 }, carrier: 'warp', decay: 0.8,
    bodies: [body({
      shape: ['scene', { scene: 3, cam: 2, res: 0.5, size: 1, blend: 0.3, speed: 0.35, pulse: 0.4, kick: 0.6, vary: 0.4, rim: 0.6, ao: 0.8, fog: 0.4, glow: 0.1, roam: 0.5, gap: 0.5, spec: 0.3, iter: 8, fscale: -2.2, fold: 1, power: 8 }],
      material: ['fill', { gain: 0.8 }],
      color: ['fixed', { hue: 0, detail: 1 }],
    })],
    reactions: [
      rx('bass', 'sh', 0, 'fscale', 0.08, { atk: 0.1, rel: 1 }),
      rx('beat', 'ma', 0, 'gain', 0.35, { atk: 0.005, rel: 0.2 }),
    ],
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
    reactions: [rx('beat', 'ma', 0, 'gain', 0.5), rx('bass', 'op', 0, 'rate', 0.12, { atk: 0.05, rel: 0.4 })],
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
    // heads swinging together once a bar and the beat chasing from head to head.
    origin: 'V01', name: 'Stage Rig', energy: [0.35, 1], scheme: 'triad', hue: 0.62,
    color: { sat: 0.85, adapt: 0.35, bloom: 1.2, vignette: 0.5 }, carrier: 'warp', car: { halfLife: 0.12 },
    bodies: [body({
      shape: ['beams', { count: 8, spread: 1.1, fan: 0.35, sweep: 0.55, pattern: 4, period: 1, width: 0.035, haze: 0.7, gobo: 0, hues: 0.06, length: 1.4, flare: 0.4, accent: 0.8 }],
      place: ['point', { x: 0, y: 0.45 }],
      material: ['glow', { gain: 1 }],
      feel: ['flow', { atk: 0.02, rel: 0.15 }],
    })],
    // Run like a lighting operator, not a metronome: every head jumps to a new aim on each beat and
    // holds, drum hits flare the lenses, the rig dims in quiet passages and blazes when it's loud, the
    // bass fattens the shafts, and a drop fans every head out wide. The choreography dims and drains the
    // rig through the build, punches it on the drop and gives each section type its own colour.
    reactions: [
      rx('hit', 'sh', 0, 'width', 0.9, { rel: 0.15 }), rx('loud', 'ma', 0, 'gain', 0.8, { atk: 0.05, rel: 0.4 }),
      rx('bass', 'sh', 0, 'haze', 0.5, { atk: 0.02, rel: 0.25 }), rx('drop', 'sh', 0, 'fan', 0.6, { atk: 0.02, rel: 2 }),
      rx('build', 'sh', 0, 'sweep', 0.3, { atk: 0.2, rel: 1 }),
    ],
    choreo: { lead: 4, push: 0.08, drain: 0.5, dim: 0.4, punch: 0.8, relax: 1, scene: 0.35, frame: 0.25, glide: 0 },
  },
  {
    // A Chladni plate: sand gathers on the nodal lines of the standing wave the chord asks for. On a
    // new chord (checked every bar) the sand scatters and snaps onto the new figure within half a beat;
    // minor keys give the antisymmetric figures. The plate turns slowly and the bass shakes the grains.
    origin: 'V02', name: 'Chladni Plate', energy: [0.15, 0.8], scheme: 'analogous', hue: 0.08,
    color: { sat: 0.7, adapt: 0.3, bloom: 1.1, vignette: 0.55 }, carrier: 'none',
    bodies: [body({
      shape: ['cymatics', { plate: 0, size: 0.42, modes: 7, source: 0, hold: 1, settle: 0.5, sand: 0.8, line: 1.6, shake: 0.3, rim: 0.35 }],
      place: ['point', { x: 0, y: 0 }],
      motion: ['spin', { rate: 0.0625 }],
      material: ['glow', { gain: 1 }],
      emit: ['none'],
      feel: ['flow', { atk: 0.01, rel: 0.12 }],
    })],
    // Tuned to dance with the beat: a new figure snaps in within half a beat of the downbeat, drum hits
    // make the grains jump, and the bass thickens the lines.
    reactions: [
      rx('bass', 'sh', 0, 'line', 0.35, { atk: 0.02, rel: 0.25 }), rx('hit', 'sh', 0, 'shake', 0.6, { rel: 0.15 }),
      rx('loud', 'ma', 0, 'gain', 0.25, { atk: 0.05, rel: 0.3 }),
    ],
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
    reactions: [rx('beat', 'op', 2, 'vx', 0.25, { rel: 0.4 }), rx('hit', 'op', 2, 'vy', -0.2, { rel: 0.4 }), rx('loud', 'op', 0, 'rate', 0.12, { atk: 0.05, rel: 0.6 })],
  },
  {
    // Yathosho (Jan T. Sott, movement by David Hansen), sakura: three soft blobs in blue, pink and
    // red swell on the beat and are copied into a turning ring of offset layers, then folded into a
    // six-petal blossom whose radius ripples in concentric bands, over water, all washed pastel. The
    // blossom flares on the beat and swells with the bass, drum hits open its ring, and the vocals
    // deepen the ripple.
    origin: 'A04', name: 'sakura (after Yathosho)', energy: [0.2, 0.75], scheme: 'triad', hue: 0.92,
    color: { sat: 0.5, exposure: 1.1, adapt: 0.35, bloom: 1.2, vignette: 0.3 }, carrier: 'warp', car: { halfLife: 0.7, water: 0.6, wsize: 0.04 },
    chain: [op('ripple', { amp: 0.0022, freq: 9, speed: 1, radial: 1 }), op('rotate', { lock: 0.125 }), op('zoom', { rate: 0.004 }), op('kaleido', { n: 6, lock: -0.0625 }, 1, 'view')],
    bodies: [body({
      shape: ['dot', { r: 0.006 }],
      place: ['ring', { n: 5, radius: 0.13 }],
      motion: ['pulse', { amp: 0.4 }],
      material: ['glow', { gain: 0.45, width: 0.022, base: 0.15 }],
      color: ['instrument', { hue: 0, amount: 1 }],
    })],
    reactions: [
      rx('beat', 'ma', 0, 'gain', 0.5, { rel: 0.25 }), rx('bass', 'sh', 0, 'r', 0.4, { atk: 0.03, rel: 0.3 }),
      rx('vocals', 'op', 0, 'amp', 0.6, { atk: 0.1, rel: 0.6 }), rx('hit', 'pl', 0, 'radius', 0.3, { rel: 0.25 }),
    ],
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
        place: ['stations', { count: 1, inst: 1, xs: 0.8, jump: 1, wander: 0 }],
        material: ['glow', { gain: 0.6, width: 0.006 }],
        emit: ['sparks', { count: 8192, size: 2.2, speed: 0.7, curl: 0.05, zoomFlow: 0, lift: -0.06, drag: 3, life: 0.55, spread: 0.12, surge: 1, top: 0, body: 0.1 }],
        color: ['age', { hue: 0.5, rate: 0.25, detail: 1 }],
      }),
    ],
    reactions: [
      rx('hit', 'em', 1, 'speed', 0.8, { rel: 0.3 }), rx('hit', 'ma', 1, 'gain', 0.6, { rel: 0.15 }),
      rx('beat', 'ma', 1, 'gain', 0.3, { rel: 0.25 }), rx('bass', 'em', 1, 'lift', 0.4, { atk: 0.03, rel: 0.4 }),
      rx('melody', 'em', 1, 'lift', 0.5, { atk: 0.1, rel: 0.3 }),
    ],
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
    reactions: [rx('hit', 'op', 1, 'rate', 0.16, { rel: 0.8, atk: 0.03 }), rx('bass', 'op', 0, 'rate', -0.12, { atk: 0.1, rel: 1 }), rx('vocals', 'sh', 0, 'radius', 0.2, { atk: 0.1, rel: 0.6 })],
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
      shape: ['scene', { scene: 2, cam: 1, res: 0.5, size: 1, blend: 0.3, speed: 0.3, pulse: 0.8, kick: 0.7, vary: 1, rim: 0.7, ao: 0.6, fog: 0.55, glow: 0.4, roam: 0.6, gap: 0.5, spec: 0.8 }],
      material: ['glow', { gain: 1.1 }],
      color: ['age', { rate: 0.0625, detail: 1 }],
    })],
    reactions: [
      rx('drums', 'sh', 0, 'kick', 0.25, { atk: 0.01, rel: 0.2 }),
      rx('bass', 'sh', 0, 'pulse', 0.2, { atk: 0.02, rel: 0.3 }),
      rx('section', 'sh', 0, 'vary', 0.3, { atk: 0.05, rel: 1.5 }),
    ],
  },
  {
    // A Mandelbulb turning under an orbiting camera: the bass raises its power so the bulbs sprout and
    // melt back, its folds breathe with the low end, drum hits push the camera in, each section turns it
    // to a new face; the spectrum tints its orbit-trap colours.
    origin: 'R06', name: 'Bulb Bloom', energy: [0.2, 0.8], scheme: 'triad', hue: 0.3,
    color: { adapt: 0.35, bloom: 1.15 }, carrier: 'warp', decay: 0.85,
    bodies: [body({
      shape: ['scene', { scene: 4, cam: 0, res: 0.35, size: 1, blend: 0.3, speed: 0.35, pulse: 0.6, kick: 0.5, vary: 1, rim: 0.7, ao: 0.45, fog: 0.2, glow: 0.35, roam: 0.4, gap: 0.5, spec: 0.6, iter: 5, fscale: -1.8, fold: 1, power: 7 }],
      material: ['fill', { gain: 1.6 }],
      color: ['fixed', { hue: 0, detail: 1 }],
    })],
    reactions: [
      rx('bass', 'sh', 0, 'power', 0.25, { atk: 0.05, rel: 0.6 }),
      rx('drums', 'sh', 0, 'kick', 0.3, { atk: 0.01, rel: 0.2 }),
      rx('section', 'sh', 0, 'vary', 0.3, { atk: 0.05, rel: 1.5 }),
    ],
  },
  {
    // Diving into a Menger sponge and back out every sixteen bars, steering around its walls: the bass
    // shifts where the holes open, drum hits lunge deeper, sections turn the sponge; neon rims only.
    origin: 'R07', name: 'Sponge Descent', energy: [0.3, 0.9], scheme: 'split', hue: 0.55,
    color: { adapt: 0.3, bloom: 1.3 }, carrier: 'warp', decay: 0.75,
    bodies: [body({
      shape: ['scene', { scene: 5, cam: 1, res: 0.5, size: 1, blend: 0.3, speed: 0.25, pulse: 0.5, kick: 0.6, vary: 1, rim: 1, ao: 0.6, fog: 0.4, glow: 0.5, roam: 0.9, gap: 0.5, spec: 0.5, iter: 4, fscale: -1.8, fold: 1, power: 8 }],
      material: ['line', { gain: 1.1 }],
      color: ['age', { rate: 0.0625, detail: 1 }],
    })],
    reactions: [
      rx('bass', 'sh', 0, 'fold', 0.12, { atk: 0.05, rel: 0.5 }),
      rx('drums', 'sh', 0, 'kick', 0.3, { atk: 0.01, rel: 0.2 }),
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
    // A murmuration: sixteen thousand boids wheel around three slowly orbiting lights in loose,
    // streaking flocks that merge and split; each bird is coloured by its heading, so turning banks
    // change colour. The flock itself keeps a calm, steady clock: it surges on every beat, loosens on
    // every downbeat and tightens into ribbons and back over each bar. On top of the trails the birds
    // flash on every drum hit and on the beat and swell with the bass (the flock overlay), and on a
    // drop the whole flock bursts outward before regrouping.
    origin: 'P03', name: 'Murmuration', energy: [0.25, 0.9], scheme: 'complementary', hue: 0.6,
    color: { adapt: 0.35, bloom: 1.1, vignette: 0.45 }, carrier: 'warp', car: { halfLife: 0.15, floor: 1 },
    chain: [op('zoom', { rate: 0.002 })],
    bodies: [body({
      shape: ['dot', { r: 0.004 }],
      place: ['orbit', { count: 3, radius: 0.36, rate: 0.125, follow: 0.1 }],
      material: ['glow', { gain: 0.5, width: 0.01 }],
      emit: ['flock', { count: 16384, speed: 0.3, radius: 0.035, align: 0.8, cohere: 0.45, separate: 0.5, wander: 0.15, home: 0.2, size: 1.6, body: 0.3, onDrop: 1, over: 0.15, osize: 2 }],
      feel: ['flow', { atk: 0.02, rel: 0.3 }],
    })],
    reactions: [
      rx('beat', 'em', 0, 'speed', 0.5, { rel: 0.3 }), rx('barpulse', 'em', 0, 'separate', 0.7, { rel: 0.4 }),
      rx('bar', 'em', 0, 'align', 0.3, { atk: 0.1, rel: 0.3 }), rx('hit', 'em', 0, 'over', 1, { atk: 0.005, rel: 0.15 }),
      rx('bass', 'em', 0, 'osize', 0.5, { atk: 0.02, rel: 0.25 }), rx('beat', 'em', 0, 'over', 0.4, { atk: 0.005, rel: 0.2 }),
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
  {
    // An underwater lagoon: clouds of tiny plankton points (the other instruments) drift on a slow current
    // through a glowing meadow of algae the vocal pollinators seed, while a few drum hunters cut through
    // the swarm and scatter it. Everything is carried by a soft fluid, so the plankton smears into
    // luminous currents; a vocal passage floods the water with bloom, a drum-heavy stretch thins the swarm.
    origin: 'B02', name: 'Plankton Lagoon', energy: [0.1, 0.65], scheme: 'analogous', hue: 0.5,
    color: { adapt: 0.3, bloom: 1.2, vignette: 0.5, contrast: 0.05 }, carrier: 'fluid', car: { halfLife: 0.6, floor: 1.2, amount: 0.8, vort: 20, fnoise: 0.4 },
    bodies: [body({
      shape: ['dot', { r: 0.01 }],
      material: ['glow', { gain: 0.9, width: 0.01 }],
      emit: ['ecosystem', { count: 49152, wD: 0.12, wB: 0.15, wV: 0.35, wO: 1, glyph: 1, size: 3, trail: 0.2, predation: 0.5, bloom: 0.9, graze: 0.3, growth: 0.8, starve: 0.3, decay: 0.994, speed: 0.6, field: 1, hues: 1, body: 0 }],
    })],
    reactions: [rx('vocals', 'em', 0, 'bloom', 0.3, { atk: 0.2, rel: 1.5 }), rx('hit', 'em', 0, 'speed', 0.5, { atk: 0.01, rel: 0.4 }), rx('loud', 'ma', 0, 'gain', 0.2, { atk: 0.05, rel: 0.4 })],
  },
  {
    // A hunt folded into a six-way kaleidoscope: diamond-shaped drum predators outnumber everything else and
    // lunge on every hit, rings of bass grazers and star pollinators scatter before them, and the plankton
    // dots they catch vanish and hatch again elsewhere. The mirror turns with the bars, so the chase reads
    // as a spinning heraldic pattern; a drop sends the hunters into a frenzy.
    origin: 'B03', name: 'Sigil Hunt', energy: [0.45, 1], scheme: 'complementary', hue: 0.02,
    color: { adapt: 0.4, bloom: 1.1, vignette: 0.55, contrast: 0.1 }, carrier: 'warp', car: { halfLife: 0.08, floor: 1.5 },
    chain: [op('rotate', { rate: 0.0008 }), op('kaleido', { n: 6, lock: 0.0625 }, 1, 'view')],
    bodies: [body({
      shape: ['dot', { r: 0.01 }],
      material: ['glow', { gain: 0.6, width: 0.01 }],
      emit: ['ecosystem', { count: 8192, wD: 1, wB: 0.3, wV: 0.3, wO: 0.8, glyph: 3, size: 4, trail: 0.3, predation: 1, bloom: 0.3, graze: 0.5, growth: 1, starve: 0.6, decay: 0.95, speed: 1.4, field: 0.25, hues: 0, body: 0 }],
    })],
    reactions: [rx('hit', 'em', 0, 'size', 0.35, { atk: 0.01, rel: 0.25 }), rx('drop', 'em', 0, 'speed', 0.5, { atk: 0.02, rel: 1.5 }), rx('beat', 'ma', 0, 'gain', 0.25, { atk: 0.01, rel: 0.2 })],
  },
  {
    // Bass herds: heavy grazers dominate the roster, drawn as long comets that plough slowly through the
    // meadow and leave furrows the vocal pollinators follow. The feedback streams gently outward, so each
    // herd's path becomes a curving wake that fans toward the edges; the bass drives the herds on, and a
    // drop makes them stampede. With the bass silent the herds thin out and the meadow grows back. Tuned for legibility: a calm
    // cruising pace that surges on each bass hit, and comets that flash bigger on every drum hit.
    origin: 'B04', name: 'Bass Herds', energy: [0.25, 0.9], scheme: 'split', hue: 0.08,
    color: { adapt: 0.35, bloom: 1, vignette: 0.45, contrast: 0.05 }, carrier: 'warp', car: { halfLife: 0.45, floor: 1.2 },
    chain: [op('zoom', { rate: 0.006 })],
    bodies: [body({
      shape: ['dot', { r: 0.01 }],
      material: ['glow', { gain: 0.7, width: 0.01 }],
      emit: ['ecosystem', { count: 8192, wD: 0.15, wB: 1, wV: 0.4, wO: 0.3, glyph: 2, size: 2, trail: 0.5, predation: 0.4, bloom: 0.7, graze: 1, growth: 0.7, starve: 0.5, decay: 0.99, speed: 0.8, field: 0.7, hues: 0, body: 0 }],
    })],
    reactions: [
      rx('bass', 'em', 0, 'speed', 0.9, { atk: 0.02, rel: 0.35 }), rx('hit', 'em', 0, 'size', 0.6, { rel: 0.2 }),
      rx('drop', 'em', 0, 'trail', 0.3, { atk: 0.02, rel: 2 }), rx('loud', 'ma', 0, 'gain', 0.2, { atk: 0.05, rel: 0.4 }),
    ],
  },
  {
    // A pollinator garden around a slowly turning six-petalled flower: the vocal pollinators are the
    // largest species, fluttering points that seed a soft meadow of bloom wherever they pass, with a few
    // drifting plankton and shy grazers and almost no predators. A sung phrase makes the meadows swell
    // out of their colonies across the screen; when the voice drops out the pollinators starve and the
    // flowers wilt, leaving the flower alone in the dark until the next phrase.
    origin: 'B05', name: 'Pollinator Garden', energy: [0.05, 0.6], scheme: 'analogous', hue: 0.85,
    color: { adapt: 0.3, bloom: 1.2, vignette: 0.5, contrast: 0.05 }, carrier: 'warp', car: { halfLife: 0.3, floor: 1 },
    bodies: [body({
      shape: ['star', { n: 6, r: 0.07, inner: 0.55 }],
      motion: ['spin', { rate: 0.0625 }],
      material: ['glow', { gain: 0.7, width: 0.012 }],
      emit: ['ecosystem', { count: 16384, wD: 0.05, wB: 0.2, wV: 1, wO: 0.5, glyph: 1, size: 3, trail: 0.2, predation: 0.2, bloom: 1, graze: 0.2, growth: 0.6, starve: 0.5, decay: 0.992, speed: 0.7, field: 1, hues: 0, body: 0.6 }],
      color: ['melody', { amount: 0.4 }],
    })],
    reactions: [rx('vocals', 'em', 0, 'field', 0.3, { atk: 0.2, rel: 1.5 }), rx('melody', 'em', 0, 'speed', 0.3, { atk: 0.1, rel: 0.8 }), rx('loud', 'ma', 0, 'gain', 0.2, { atk: 0.05, rel: 0.4 })],
  },
];

// W01.. showcase the drift gene: the preset travels through gene space as the song unfolds, one
// small mutation per section, returning to a section type's earlier look when it comes back.
const DRIFT: Def[] = [
  {
    // A pendulum harmonograph folded eight ways and streamed slowly outward. Every section nudges its
    // ratios, radius, trail and colours a little further along one journey through the song; the second
    // chorus comes back almost exactly to the first chorus' figure, the breakdown wanders furthest, and
    // each new look morphs in over four bars.
    origin: 'W01', name: 'Tidal Mandala', energy: [0.2, 0.8], scheme: 'split', hue: 0.6,
    color: { bloom: 1, vignette: 0.5, adapt: 0.35 }, carrier: 'warp', decay: 0.97,
    chain: [op('zoom', { rate: 0.014, radial: 0.5 }), op('swirl', { amt: 0.004, k: 4 }), op('kaleido', { n: 8, lock: 0.0625 }, 1, 'view')],
    bodies: [body({
      shape: ['curve', { form: 5, radius: 0.3, amp: 0.22, ra: 2, rb: 3 }],
      material: ['line', { gain: 0.8, width: 1.3, halo: 0.2 }],
      color: ['age', { rate: 0.0625, detail: 0.7 }],
    })],
    reactions: [rx('beat', 'ma', 0, 'gain', 0.35, { atk: 0.01, rel: 0.3 }), rx('bass', 'op', 1, 'amt', 0.3, { atk: 0.05, rel: 0.5 })],
    drift: { step: 0.55, kinds: 0, what: 0, ret: 0.85, morph: 4, bound: 0.2, seed: 0.31 },
  },
  {
    // Six stars spin on a ring and stream inward. Section by section their size, spikes, spin and glow
    // wander; on each drop the shapes themselves jump to a new kind (and back toward the old one when
    // the section type returns), so the drop reads as a costume change of the same dancers.
    origin: 'W02', name: 'Costume Change', energy: [0.35, 1], scheme: 'triad', hue: 0.05,
    color: { bloom: 1.2, vignette: 0.45, adapt: 0.4 }, carrier: 'warp', decay: 0.985,
    chain: [op('zoom', { rate: -0.012 }), op('rotate', { lock: -0.0625 })],
    bodies: [body({
      shape: ['star', { n: 6, r: 0.09, inner: 0.45 }],
      place: ['ring', { n: 6, radius: 0.28 }],
      motion: ['spin', { rate: 0.125 }],
      material: ['line', { gain: 0.9, width: 1.5, halo: 0.3 }],
      color: ['instrument', { amount: 0.8, detail: 0.5 }],
    })],
    reactions: [rx('drums', 'sh', 0, 'r', 0.3, { atk: 0.01, rel: 0.25 }), rx('bass', 'pl', 0, 'radius', 0.25, { atk: 0.05, rel: 0.4 })],
    drift: { step: 0.8, kinds: 1, what: 1, ret: 0.9, morph: 2, bound: 0.3, seed: 0.9 },
  },
  {
    // Stained glass: a slow Voronoi lattice of lead veins with glowing panes. The window itself never
    // changes; only the light through it does. Each section the palette, saturation, bloom and contrast
    // drift a long way over eight bars, like the day passing behind the glass, and a returning section
    // brings its own light back.
    origin: 'W03', name: 'Stained Glass Seasons', energy: [0.1, 0.7], scheme: 'analogous', hue: 0.1,
    color: { sat: 0.9, exposure: 1.05, adapt: 0.3, bloom: 0.8, vignette: 0.35, contrast: 0.04 },
    carrier: 'warp', car: { halfLife: 0.2, floor: 0.2 },
    chain: [op('rotate', { lock: 0, rate: 0.0004 })],
    bodies: [body({
      shape: ['cells', { mode: 1, scale: 3, speed: 0.15, warp: 0.4, wall: 0.55, fill: 0.8, var: 0.45, pulse: 0.25 }],
      material: ['fill', { gain: 1 }],
      emit: ['trail'],
      color: ['pitch', { detail: 0.8 }],
    })],
    reactions: [rx('loud', 'sh', 0, 'fill', 0.25, { atk: 0.05, rel: 0.5 }), rx('beat', 'sh', 0, 'pulse', 0.3, { atk: 0.01, rel: 0.3 })],
    drift: { step: 1, kinds: 0, what: 2, ret: 0.7, morph: 8, bound: 0.3, seed: 0.44 },
  },
  {
    // A glowing torus knot tumbles in 3D above a streaming tunnel. Its look stays put; what drifts is how
    // everything moves: the tumble, the flow of the tunnel, the swirl and the body's own dance change
    // from section to section (even the kind of dance, spin one section and sway or bob the next),
    // quickly, within a bar, so every section has its own choreography of the same object.
    origin: 'W04', name: 'Knot Dancer', energy: [0.3, 0.95], scheme: 'complementary', hue: 0.55,
    color: { sat: 0.9, adapt: 0.35, bloom: 1.2, vignette: 0.5 }, carrier: 'warp', car: { halfLife: 0.9 },
    chain: [op('zoom', { rate: 0.012 }), op('swirl', { amt: 0.006, k: 5 })],
    bodies: [body({
      shape: ['superscope', { family: 0, p: 2, q: 3, size: 0.26, audio: 0.3, spec: 0, spinX: 0.0625, spinY: 0.125, persp: 0.6, n: 1024 }],
      motion: ['spin', { rate: 0.125 }],
      material: ['line', { gain: 1, width: 1.4, halo: 0.3 }],
      color: ['age', { rate: 0.125, detail: 0.8 }],
      feel: ['flow', { atk: 0.03, rel: 0.4 }],
    })],
    reactions: [rx('bass', 'sh', 0, 'audio', 0.4, { atk: 0.03, rel: 0.4 }), rx('drums', 'op', 0, 'rate', 0.12, { atk: 0.03, rel: 0.3 })],
    drift: { step: 0.4, kinds: 2, what: 3, ret: 0.6, morph: 1, bound: 0.2, seed: 0.6 },
  },
  {
    // The far traveller: glowing polygons orbit and stream outward, and every section boundary cuts
    // (no morph) to a new relative of the section before: shapes, placements, materials and space ops
    // may all change, so the song reads as one family tree walked in order. A returning section type
    // lands near its first appearance and the outro comes home to the saved preset.
    origin: 'W05', name: 'Family Tree', energy: [0.3, 1], scheme: 'split', hue: 0.8,
    color: { sat: 0.95, adapt: 0.4, bloom: 1.15, vignette: 0.45 }, carrier: 'warp', car: { halfLife: 0.5 },
    chain: [op('zoom', { rate: 0.012 }), op('rotate', { lock: 0.0625 })],
    bodies: [body({
      shape: ['polygon', { n: 5, r: 0.06, round: 0.2 }],
      place: ['orbit', { count: 5, radius: 0.25, rate: 0.25 }],
      material: ['line', { gain: 1, width: 1.5, halo: 0.3 }],
      color: ['instrument', { amount: 0.9, detail: 0.5 }],
    })],
    reactions: [rx('drums', 'sh', 0, 'r', 0.3, { atk: 0.01, rel: 0.25 }), rx('bass', 'pl', 0, 'radius', 0.25, { atk: 0.05, rel: 0.4 })],
    drift: { step: 0.6, kinds: 2, what: 0, ret: 0.6, morph: 0, bound: 0.4, seed: 0.7 },
  },
];

// L01-: the song as a landscape (the 'landscape' shape, genes/landscape.ts): the analysed song is built
// into terrain before playback and the camera travels through it, so the drops are on the horizon
// long before they arrive.
const LANDSCAPE: Def[] = [
  {
    // A night road over rolling hills, the altitude following the song's energy: the road climbs through
    // each build toward a pass between two mountains, the drop, and plunges over its far side; every
    // section begins at a lit gate over the road, breakdowns sink into valleys, and the next drop glows
    // above its pass like a sun rising on the horizon. Drum hits bump the camera, the bass lights the road.
    origin: 'L01', name: 'Road to the Drop', energy: [0.2, 0.9], scheme: 'analogous', hue: 0.55,
    color: { adapt: 0.35, bloom: 1.15, vignette: 0.35 }, carrier: 'warp', decay: 0.7,
    bodies: [body({
      shape: ['landscape', { path: 0, ground: 0, mark: 1, res: 0.5, look: 24, height: 0.35, relief: 0.7, rough: 0.5, wind: 0.5, fog: 0.45, glow: 0.6, tint: 0.5, kick: 0.3, rim: 0.5 }],
      material: ['fill', { gain: 1.2 }],
      color: ['fixed', { hue: 0, detail: 1 }],
    })],
    reactions: [
      rx('bass', 'sh', 0, 'glow', 0.3, { atk: 0.02, rel: 0.3 }),
      rx('drums', 'sh', 0, 'kick', 0.3, { atk: 0.01, rel: 0.2 }),
      rx('loud', 'sh', 0, 'rough', 0.15, { atk: 0.1, rel: 1 }),
    ],
  },
  {
    // A river through a canyon of violet crystal: the water carries the camera between faceted banks
    // that rise with the music, a glowing ring floats over the river at every section start, and the
    // key stains the crystal (a change of key ahead shows as a new colour in the distant walls). A long
    // view (32 s ahead) lets the drop's pass hang on the horizon, ringed and sunlit, for a whole build
    // before the river runs into its gorge. The vocals brighten the rings, the bass the crystal edges.
    origin: 'L02', name: 'Crystal River', energy: [0.15, 0.8], scheme: 'split', hue: 0.72,
    color: { adapt: 0.4, bloom: 1.25, vignette: 0.4 }, carrier: 'warp', decay: 0.8,
    bodies: [body({
      shape: ['landscape', { path: 1, ground: 1, mark: 2, res: 0.5, look: 32, height: 0.6, relief: 0.5, rough: 0.6, wind: 0.4, fog: 0.5, glow: 0.7, tint: 0.8, kick: 0.2, rim: 0.7 }],
      material: ['fill', { gain: 1.1 }],
      color: ['fixed', { hue: 0, detail: 0.8 }],
    })],
    reactions: [
      rx('vocals', 'sh', 0, 'glow', 0.3, { atk: 0.05, rel: 0.5 }),
      rx('drums', 'sh', 0, 'kick', 0.3, { atk: 0.01, rel: 0.2 }),
      rx('bass', 'sh', 0, 'rim', 0.2, { atk: 0.02, rel: 0.4 }),
    ],
  },
  {
    // Flying low over an endless desert of dunes at dusk, banking with the wind's curves: the dunes swell
    // taller as the song gets louder and flatten into salt pans in breakdowns, pairs of light pillars
    // stand at every section start like beacons marking the route, and each drop rises from the horizon
    // as a sunlit pass the flight climbs toward through the whole build (40 s of the song in view).
    // The kick drum jolts the glider, the bass swells the beacons, the loudness raises the dunes.
    origin: 'L03', name: 'Dune Glider', energy: [0.1, 0.75], scheme: 'complementary', hue: 0.08,
    color: { adapt: 0.4, bloom: 1.3, vignette: 0.45 }, carrier: 'warp', decay: 0.75,
    bodies: [body({
      shape: ['landscape', { path: 2, ground: 2, mark: 3, res: 0.5, look: 40, height: 0.3, relief: 0.8, rough: 0.8, wind: 0.8, fog: 0.3, glow: 0.8, tint: 0.3, kick: 0.4, rim: 0.4 }],
      material: ['fill', { gain: 1.15 }],
      color: ['height', { amount: 0.3 }],
    })],
    reactions: [
      rx('bass', 'sh', 0, 'glow', 0.25, { atk: 0.02, rel: 0.4 }),
      rx('drums', 'sh', 0, 'kick', 0.3, { atk: 0.01, rel: 0.2 }),
      rx('loud', 'sh', 0, 'relief', 0.15, { atk: 0.2, rel: 2 }),
    ],
  },
  {
    // A night train on an elevated rail through a neon city: the blocks are drawn only by their lit
    // edges, towers grow taller in the loud sections and the city thins into open lots in breakdowns,
    // every chorus passes through the same streets (one layout per section type), black obelisks flank
    // the line at each section start, and the drop's pass towers over the skyline ahead with its sun.
    // The rails flash on the beat, the drums jolt the carriage, the bass brightens the edges.
    origin: 'L04', name: 'Neon Night Line', energy: [0.35, 1], scheme: 'triad', hue: 0.85,
    color: { adapt: 0.3, bloom: 1.4, vignette: 0.4 }, carrier: 'warp', decay: 0.85,
    bodies: [body({
      shape: ['landscape', { path: 3, ground: 3, mark: 0, res: 0.5, look: 16, height: 0.5, relief: 0.5, rough: 0.7, wind: 0.35, fog: 0.35, glow: 0.9, tint: 0.4, kick: 0.5, rim: 0.35 }],
      material: ['line', { gain: 1.1 }],
      color: ['age', { rate: 0.0625, detail: 0.8 }],
    })],
    reactions: [
      rx('bass', 'sh', 0, 'rim', 0.2, { atk: 0.02, rel: 0.3 }),
      rx('drums', 'sh', 0, 'kick', 0.3, { atk: 0.01, rel: 0.15 }),
      rx('loud', 'sh', 0, 'rough', 0.2, { atk: 0.2, rel: 1.5 }),
    ],
  },
  {
    // A fast run through an abstract world of terraced ribbons, mirrored top to bottom (a mirror
    // placement) so the road runs along the middle of the screen with terrain and horizon both above
    // and below it: only 10 s of the song is in view, so the world rushes by, each drop's pass looms up
    // quickly and the road plunges over its cliff, light pillars flash past at every section start and
    // the terraces trail behind in the feedback. The loudness raises the terraces, the drums jolt the
    // camera, the bass lights the road.
    origin: 'L05', name: 'Ribbon Rush', energy: [0.45, 1], scheme: 'triad', hue: 0.95,
    color: { adapt: 0.3, bloom: 1.3, vignette: 0.3 }, carrier: 'warp', decay: 0.9,
    bodies: [body({
      shape: ['landscape', { path: 0, ground: 4, mark: 3, res: 0.5, look: 10, height: 0.6, relief: 0.9, rough: 0.9, wind: 0.6, fog: 0.25, glow: 0.8, tint: 0.6, kick: 0.5, rim: 0.7 }],
      place: ['mirror', { axis: 1, x: 0, y: 0.2 }],
      material: ['fill', { gain: 1.1 }],
      color: ['fixed', { hue: 0, detail: 1 }],
    })],
    reactions: [
      rx('loud', 'sh', 0, 'relief', 0.1, { atk: 0.2, rel: 1.5 }),
      rx('drums', 'sh', 0, 'kick', 0.3, { atk: 0.01, rel: 0.15 }),
      rx('bass', 'sh', 0, 'glow', 0.2, { atk: 0.02, rel: 0.3 }),
    ],
  },
];

// H01.. showcase the harmony gene (genes/harmony.ts): the chord progression read on the Tonnetz,
// consonance as symmetry, tension breaking it, resolutions snapping it back.
const HARMONY: Def[] = [
  {
    // An eight-fold mandala of a slowly turning star, streaming outward. While the harmony sits on
    // the home chord the mandala is perfect; as the chords wander away its segments slide apart and
    // an off-centre swirl pulls at the frame; when the progression comes home (V-I) every segment
    // clicks back into place with a flash. Each chord change nudges the colours along the lattice.
    origin: 'H01', name: 'Cadence Mandala', energy: [0.2, 0.85], scheme: 'analogous', hue: 0.6,
    color: { bloom: 1.15, vignette: 0.5 }, carrier: 'warp', decay: 0.975,
    chain: [op('zoom', { rate: 0.016, radial: 1 }), op('rotate', { lock: 0.0625 }), op('kaleido', { n: 8, lock: 0.0625 }, 1, 'view')],
    bodies: [body({
      shape: ['star', { n: 5, r: 0.22, inner: 0.4 }],
      place: ['point'],
      motion: ['spin', { rate: 0.125 }],
      material: ['line', { gain: 1.1, width: 2, halo: 0.25 }],
      color: ['age', { rate: 0.0625, detail: 0.5 }],
    })],
    reactions: [
      rx('chordchange', 'ma', 0, 'gain', 0.5, { atk: 0.01, rel: 0.4 }),
      rx('tension', 'op', 0, 'rate', 0.14, { atk: 0.3, rel: 0.6 }),
      rx('bass', 'sh', 0, 'r', 0.2, { atk: 0.03, rel: 0.3 }),
    ],
    harmony: { brk: 0.85, warp: 0.3, style: 1, snap: 0.8, settle: 0.3, walk: 0.08, kick: 0.3, modHue: 0.1, modTurn: 0.008, calm: 0.4 },
  },
  {
    // A pendulum harmonograph (its frequency ratios walk the circle of fifths) drawn four ways in a
    // mirror, leaving long warm trails that sink inward. Consonant chords keep the four quarters in
    // perfect reflection; tension leans the reflections apart and bends the frame sideways, the trails
    // lengthen and the colours slide away from home. A resolution lets the quarters fall back
    // together with a slow, pendulum-like wobble.
    origin: 'H02', name: 'Suspended Mirror', energy: [0.1, 0.7], scheme: 'split', hue: 0.07,
    color: { bloom: 0.9, vignette: 0.55, adapt: 0.7 }, carrier: 'warp', decay: 0.975,
    chain: [op('zoom', { rate: -0.006, wander: 0.1 }), op('rotate', { lock: -0.0625 }), op('mirror', { axis: 2 }, 1, 'view')],
    bodies: [body({
      shape: ['curve', { form: 5, radius: 0.38, amp: 0.3, ra: 2, rb: 2 }],
      material: ['line', { gain: 0.4, width: 1.3, halo: 0.12 }],
      color: ['age', { rate: 0.03125, detail: 0.7 }],
    })],
    reactions: [
      rx('tension', 'car', 0, 'floor', -0.5, { atk: 0.4, rel: 1 }),
      rx('resolve', 'ma', 0, 'gain', 0.6, { atk: 0.01, rel: 0.8 }),
      rx('vocals', 'sh', 0, 'amp', 0.25, { atk: 0.1, rel: 0.6 }),
    ],
    harmony: { brk: 1, warp: 0.45, style: 0, snap: 0.45, settle: 1.2, walk: 0.12, kick: 0.1, modHue: 0.12, modTurn: -0.006, calm: 0.7 },
  },
  {
    // A hexagon keyboard lit by the chroma, folded into a wall of mirrored tiles. On the home chord the tiles meet
    // seamlessly like a tiled floor; as the harmony strays, each tile turns and slips on its own
    // (every chord shuffles them a different way) and the whole wall buckles; the resolution clicks
    // them back flush with a hard flash. Key changes swing the wall's hue far round the wheel and
    // tilt it, so a modulation reads as a new room.
    origin: 'H03', name: 'Modulation Tiles', energy: [0.3, 0.9], scheme: 'triad', hue: 0.35,
    color: { bloom: 1.05, vignette: 0.4, adapt: 0.35 }, carrier: 'warp', decay: 0.9,
    chain: [op('tile', { n: 2.4 }, 1, 'view')],
    bodies: [body({
      shape: ['polygon', { n: 6, r: 0.4965 / 5 }],
      place: ['grid', { lattice: 1, scale: 5, jitter: 0, density: 1, lit: 0.6, links: 0, twinkle: 0.3, lock: 0.0625 }],
      material: ['fill', { gain: 2, soft: 0.1, outline: 1, core: 1 }],
    })],
    reactions: [
      rx('chordchange', 'ma', 0, 'gain', 0.5, { atk: 0.01, rel: 0.4 }),
      rx('tension', 'pl', 0, 'jitter', 0.5, { atk: 0.4, rel: 0.8 }),
      rx('modulation', 'col', 0, 'exposure', 0.4, { atk: 0.02, rel: 1.5 }),
    ],
    harmony: { brk: 0.9, warp: 0.5, style: 2, snap: 1, settle: 0.15, walk: 0.1, kick: 0.4, modHue: 0.22, modTurn: 0.012, calm: 0.2 },
  },
  {
    // The harmony map itself: the tonal lattice (fifths across, thirds on the diagonals) with every
    // note node glowing as it sounds. The current chord's triangle burns, the last chords leave a
    // fading path, and the camera drifts after the walk. Tension bends the lattice off true; a
    // cadence home straightens it with a flash, and a key change turns the whole map.
    origin: 'H04', name: 'Tonnetz Walk', energy: [0.1, 0.8], scheme: 'analogous', hue: 0.52,
    color: { sat: 0.85, adapt: 0.3, bloom: 1.15, vignette: 0.5 }, carrier: 'warp', decay: 0.7,
    bodies: [body({
      shape: ['tonnetz', { scale: 0.14, follow: 0.6, tilt: 0, nodes: 0.5, lines: 0.25, fill: 0.85, echo: 0.3, trail: 0.55, pulse: 0.9 }],
      place: ['point', { x: 0, y: 0 }],
      material: ['glow', { gain: 1 }],
      emit: ['trail'],
      feel: ['flow', { atk: 0.01, rel: 0.15 }],
    })],
    // A beat layer under the chord walk (tuned after a driving synth-pop track read as drifting): drum
    // hits flash the lattice, the bass swells the note nodes with room to move, and each chord change
    // and cadence lands with a firmer kick.
    reactions: [
      rx('chordchange', 'sh', 0, 'lines', 0.6, { atk: 0.01, rel: 0.35 }), rx('bass', 'sh', 0, 'nodes', 0.45, { atk: 0.02, rel: 0.25 }),
      rx('hit', 'ma', 0, 'gain', 0.5, { rel: 0.2 }),
    ],
    harmony: { brk: 0.5, warp: 0.4, style: 0, snap: 0.8, settle: 0.3, walk: 0.05, kick: 0.6, modHue: 0.1, modTurn: 0.02, calm: 0.3 },
  },
  {
    // A spectrum skyline standing on a lake: the bars rise from the waterline and the lower half of
    // the frame mirrors them, while the trails drift slowly upward like heat haze. On the home chord
    // the reflection is exact; as the harmony strays the reflection slides and tilts out of register
    // and the whole scene buckles in waves, the colours drifting with each chord; a cadence snaps
    // the water still with a kick of light. Key changes swing the palette.
    origin: 'H05', name: 'Reflecting Pool', energy: [0.3, 1], scheme: 'complementary', hue: 0.78,
    color: { bloom: 1.2, vignette: 0.5, contrast: 0.05 }, carrier: 'warp', decay: 0.94,
    chain: [op('translate', { vy: 0.06 }), op('mirror', { axis: 1 }, 1, 'view')],
    bodies: [body({
      shape: ['bars', { mode: 0, bins: 48, radius: 0.18, len: 0.35, fill: 0.6 }],
      place: ['point', { x: 0, y: 0.02 }],
      material: ['line', { gain: 1, width: 1.6, halo: 0.2 }],
    })],
    reactions: [
      rx('tension', 'car', 0, 'floor', -0.4, { atk: 0.4, rel: 0.8 }),
      rx('resolve', 'col', 0, 'bloom', 0.6, { atk: 0.01, rel: 0.6 }),
      rx('drums', 'sh', 0, 'len', 0.3, { atk: 0.01, rel: 0.2 }),
    ],
    harmony: { brk: 0.9, warp: 0.55, style: 2, snap: 0.9, settle: 0.3, walk: 0.14, kick: 0.4, modHue: 0.15, modTurn: -0.01, calm: 0.5 },
  },
];

// Q01.. showcase the groove gene (src/v2/genes/groove.ts): the motion takes the music's timing feel,
// swung, laid back, machine-tight or loose, as measured from the song.
const GROOVE: Def[] = [
  {
    // A row of lanterns sways on the bar and their trails float upward, drawing the swing as a loping
    // wave: in swung music every sway lands late on the off-beat and a small off-beat pulse pops on
    // the "and"; a laid-back backbeat drags the whole row behind the grid; human playing nudges each
    // lantern on the hits, while a machine-tight track makes the row tick in crisp half-beat steps.
    origin: 'Q01', name: 'Shuffle Lanterns', energy: [0.2, 0.75], scheme: 'analogous', hue: 0.08,
    color: { adapt: 0.4, bloom: 1.1, vignette: 0.5 }, carrier: 'warp', car: { halfLife: 0.9, floor: 0.5 },
    chain: [op('translate', { vy: 0.32 }), op('noise', { amp: 0.0006, scale: 2, speed: 0.2 })],
    bodies: [body({
      shape: ['polygon', { n: 6, r: 0.03, round: 0.4 }],
      place: ['row', { count: 5, y: -0.38, wander: 0 }],
      motion: ['sway', { amp: 0.07, period: 1, tilt: 0.6 }],
      material: ['glow', { gain: 0.75, width: 0.007, base: 0.3, halo: 0.2 }],
      emit: ['trail', { tip: 0.5 }],
    })],
    reactions: [rx('bass', 'ma', 0, 'gain', 0.35, { atk: 0.03, rel: 0.3 }), rx('swing', 'mo', 0, 'amp', 0.4, { atk: 0.5, rel: 1.5 }), rx('hit', 'ma', 0, 'gain', 0.4, { rel: 0.2 })],
    groove: { swing: 1.2, sub: 8, sway: 0.03, off: 0.6, lean: 0.7, crisp: 0.7, tick: 2, jitter: 0.6, accent: 0.4 },
  },
  {
    // Three stars circle the centre and spin while the picture streams outward into a spiral tunnel.
    // The orbit runs on the groove's clock: in a swung song the stars rush through the downbeat and
    // hang back into the off-beat, and a laid-back backbeat drags the whole wheel a touch behind the
    // kick; the spiral trail records every lurch. Straight electronic tracks turn it into an even,
    // slightly ticking wheel.
    origin: 'Q02', name: 'Laid-Back Orbit', energy: [0.3, 0.85], scheme: 'split', hue: 0.62,
    color: { adapt: 0.35, bloom: 1.1, vignette: 0.55 }, carrier: 'warp', car: { halfLife: 0.35, floor: 1 },
    chain: [op('zoom', { rate: 0.006 }), op('rotate', { lock: 0.0625 })],
    bodies: [body({
      shape: ['star', { n: 5, r: 0.06, inner: 0.45 }],
      place: ['orbit', { count: 3, radius: 0.22, rate: 0.5 }],
      motion: ['spin', { rate: 0.5 }],
      material: ['line', { gain: 1.1, width: 1.6, halo: 0.25 }],
      emit: ['trail', { tip: 0.4 }],
      feel: ['flow', { atk: 0.02, rel: 0.25, sens: 1.3 }],
    })],
    reactions: [
      rx('hit', 'pl', 0, 'radius', 0.5, { atk: 0.005, rel: 0.25 }), rx('bass', 'pl', 0, 'radius', 0.3, { atk: 0.03, rel: 0.35 }),
      rx('melody', 'pl', 0, 'y', 0.6, { atk: 0.15, rel: 0.5 }),
    ],
    groove: { swing: 1, sub: 8, sway: 0.008, off: 0.5, lean: 1, crisp: 0.3, tick: 4, jitter: 0.3, accent: 0.5 },
  },
  {
    // A hexagonal lattice of six-point stars in one colour, each a pitch class that lights with the
    // chroma, the whole lattice turning on the bar. On a machine-tight track it ticks like a watch
    // movement: the lattice holds still, then snaps a notch forward on every beat, as every star kicks
    // in size and flares; the bass swells the stars. Loose, swung playing melts the ticks into a
    // lilting, rolling turn.
    origin: 'Q03', name: 'Clockwork Lattice', energy: [0.35, 0.95], scheme: 'triad', hue: 0.55,
    color: { adapt: 0.4, bloom: 1, vignette: 0.5, contrast: 0.05 }, carrier: 'warp', car: { halfLife: 0.12, floor: 1 },
    bodies: [body({
      shape: ['star', { n: 6, r: 0.075, inner: 0.4 }],
      place: ['grid', { lattice: 1, scale: 2.6, jitter: 0, density: 0.85, lit: 1, links: 0, twinkle: 0, lock: 0.125 }],
      motion: ['pulse', { amp: 0.2 }],
      material: ['fill', { gain: 1.2, soft: 0.1, outline: 1, core: 0.6 }],
      emit: ['none'],
      color: ['fixed', { hue: 0, detail: 0.5 }],
    })],
    reactions: [rx('beat', 'ma', 0, 'gain', 0.5, { rel: 0.2 }), rx('bass', 'sh', 0, 'r', 0.12, { atk: 0.03, rel: 0.3 })],
    groove: { swing: 0.8, sub: 16, sway: 0, off: 0.2, lean: 0.3, crisp: 1, tick: 1, jitter: 0, accent: 0.3 },
  },
  {
    // A jam session in ink: one small triangle per instrument stirs a fluid, each moving with its
    // player (drums jump on hits, bass swings out, vocals follow the melody). The groove makes them
    // play like people: every hit nudges the players a little, differently each time, so a loose,
    // human drummer scatters the ink in organic flicks, syncopated hits kick the players hard, and
    // the dye pushes on the (swung) off-beat. A quantized track keeps them tidy and still.
    origin: 'Q04', name: 'Loose Ensemble', energy: [0.2, 0.8], scheme: 'triad', hue: 0.3,
    color: { adapt: 0.35, bloom: 1.05, vignette: 0.45 }, carrier: 'fluid', decay: 0.992, car: { floor: 1.4, amount: 1.1, vort: 32, fnoise: 0.3 },
    bodies: [body({
      shape: ['polygon', { n: 3, r: 0.022, round: 0.3 }],
      place: ['stations', { count: 4, inst: 1, xs: 0.8, jump: 1, wander: 0.05 }],
      motion: ['hits', { amt: 0.6 }],
      material: ['glow', { gain: 0.9, width: 0.016, base: 0.2 }],
      emit: ['dye', { force: 1.2 }],
      feel: ['flow', { atk: 0.01, rel: 0.25 }],
    })],
    reactions: [rx('humanity', 'em', 0, 'force', 0.4, { atk: 0.3, rel: 1 }), rx('synco', 'ma', 0, 'gain', 0.3, { atk: 0.2, rel: 0.8 })],
    groove: { swing: 1, sub: 8, sway: 0.015, off: 0.6, lean: 0.4, crisp: 0.2, tick: 2, jitter: 1, accent: 0.9 },
  },
  {
    // A funk ring: a full circle of spectrum bars that pumps on the beat, mirrored into a mandala and
    // spun through a slow tunnel zoom. It is tuned for 16th-note grooves: the ring's pulse lands on
    // the swung 16th off-beats, syncopated hits (the ghost notes and pushes between the beats) kick
    // it wide, and the whole ring sways a little around the two-beat backbeat. Straight, busy
    // techno just pumps it evenly.
    origin: 'Q05', name: 'Ghost Note Ring', energy: [0.4, 1], scheme: 'complementary', hue: 0.9,
    color: { adapt: 0.4, bloom: 1, vignette: 0.55, ca: 0.003 }, carrier: 'warp', car: { halfLife: 0.1, floor: 1 },
    chain: [op('zoom', { rate: 0.025 }), op('mirror', { axis: 2 }, 1, 'view')],
    bodies: [body({
      shape: ['bars', { mode: 2, bins: 40, radius: 0.14, len: 0.3, fill: 0.5 }],
      motion: ['pulse', { amp: 0.25 }],
      material: ['line', { gain: 0.7, width: 1.4, halo: 0.15 }],
      emit: ['trail', { tip: 0 }],
      feel: ['flow', { atk: 0.01, rel: 0.12, sens: 1.2 }],
    })],
    reactions: [rx('synco', 'sh', 0, 'len', 0.4, { atk: 0.2, rel: 0.8 }), rx('drums', 'ma', 0, 'gain', 0.3, { atk: 0.01, rel: 0.2 })],
    groove: { swing: 1.3, sub: 16, sway: 0.02, off: 1, lean: 0.5, crisp: 0.4, tick: 4, jitter: 0.3, accent: 1 },
  },
];


// D01.. showcase visual deja vu (genes/dejavu.ts): a section that returns (the second chorus, a
// returning riff) recalls the picture, framing, colours and motion of its first appearance, evolved.
const DEJAVU: Def[] = [
  {
    // Two roaming glow heads paint a garden of trails that drifts slowly outward and turns, folded
    // five ways: over a section the drawing grows into something no other moment has. When a chorus
    // comes back, the garden it grew the first time resurfaces over two bars (turned a little
    // further each time), the colours swing back to that chorus's own and the turning rewinds to
    // where it was; then the heads keep painting and the old garden slowly overgrows.
    origin: 'D01', name: 'Chorus Garden', energy: [0.2, 0.85], scheme: 'triad', hue: 0.35,
    color: { adapt: 0.35, bloom: 1.1, vignette: 0.4 }, carrier: 'warp', car: { halfLife: 6, floor: 0.2 },
    chain: [op('zoom', { rate: 0.0008 }), op('rotate', { lock: 0, rate: 0.002 }), op('kaleido', { n: 5, lock: 0 }, 1, 'view')],
    bodies: [body({
      shape: ['dot', { r: 0.02 }],
      place: ['walker', { heads: 2, step: 0.15, every: 1, curve: 1.2 }],
      material: ['glow', { gain: 1, width: 0.014 }],
      emit: ['trail'],
      color: ['instrument', { amount: 1 }],
    })],
    reactions: [rx('beat', 'ma', 0, 'gain', 0.4, { atk: 0.01, rel: 0.25 })],
    dejavu: { recall: 0.85, blend: 2, snap: 0.85, frame: 0.5, hue: 0.8, motion: 0.6, evolve: 0.25, keep: 0, res: 0.5, cap: 3, min: 0.65 },
  },
  {
    // Ink in water: three sources ride their instruments through a swirling liquid and bleed dye on
    // the beats. Every appearance of a section that will return is remembered, so the memory is
    // rewritten each time: when the chorus comes back, the ink cloud it left last time wells up again
    // out of the current over four bars, softer and a turn further round, its colours restored, and
    // the living current tears it into something new that the next return will remember.
    origin: 'D02', name: 'Ink Recollection', energy: [0.15, 0.75], scheme: 'complementary', hue: 0.58,
    color: { exposure: 0.85, adapt: 0.3, bloom: 1.05, vignette: 0.35 }, carrier: 'fluid',
    car: { halfLife: 2.5, floor: 0.8, amount: 1.1, vort: 30, fnoise: 0.4 },
    bodies: [body({
      shape: ['dot', { r: 0.012 }],
      place: ['stations', { count: 3, inst: 1, xs: 0.7, jump: 0, wander: 0.25 }],
      motion: ['circle', { radius: 0.08, period: 4 }],
      material: ['glow', { gain: 0.6, width: 0.02 }],
      emit: ['dye', { force: 1.2 }],
      feel: ['flow', { atk: 0.02, rel: 0.35 }],
    })],
    reactions: [
      rx('bass', 'car', 0, 'amount', 0.3, { atk: 0.05, rel: 0.5 }), rx('loud', 'em', 0, 'force', 0.4, { atk: 0.05, rel: 0.4 }),
    ],
    dejavu: { recall: 0.7, blend: 4, snap: 0.75, frame: 0.3, hue: 0.6, motion: 0.2, evolve: 0.5, keep: 1, res: 0.25, cap: 4, min: 0.7 },
  },
  {
    // Six small stars drift on slow Lissajous paths through a gently stirred night, each drawing a long
    // luminous wake in its instrument's colour, so the sky fills with a constellation of looping trails
    // that no other moment repeats. When a section returns, its first constellation comes back in a
    // bar, a quick cut to a memory, framed as it was and slightly turned, and the stars go on drawing
    // over it while the old wakes fade.
    origin: 'D03', name: 'Constellation Return', energy: [0.1, 0.7], scheme: 'split', hue: 0.6,
    color: { adapt: 0.3, bloom: 1.2, vignette: 0.45, contrast: 0.04 }, carrier: 'warp', car: { halfLife: 8, floor: 0.15 },
    chain: [op('swirl', { amt: 0.004, k: 3 }), op('zoom', { rate: -0.0004 })],
    bodies: [body({
      shape: ['star', { n: 5, r: 0.012, inner: 0.45 }],
      place: ['float', { count: 6, spread: 0.6, speed: 0.15 }],
      motion: ['spin', { rate: 0.125 }],
      material: ['line', { gain: 0.9, width: 1.5, halo: 0.3 }],
      emit: ['trail', { tip: 1 }],
      color: ['instrument', { amount: 1 }],
    })],
    reactions: [rx('beat', 'ma', 0, 'gain', 0.35, { atk: 0.01, rel: 0.3 }), rx('vocals', 'op', 0, 'amt', 0.3, { atk: 0.1, rel: 0.6 })],
    dejavu: { recall: 0.9, blend: 1, snap: 0.85, frame: 0.7, hue: 0.7, motion: 0.5, evolve: 0.3, keep: 0, res: 0.5, cap: 3, min: 0.7 },
  },
  {
    // A ring of six neon hexagons spins on its own free clock and streams outward into a fast, turning
    // tunnel of afterimages in triad colours. Nothing lingers, so a return is a flashback: when a
    // section comes back, its first tunnel flashes over the picture for half a bar, the camera snaps to
    // the remembered framing and the ring's spin jumps back to the angle it had, so the shot rhymes
    // with the first one before it streams away again.
    origin: 'D04', name: 'Flashback Tunnel', energy: [0.35, 1], scheme: 'triad', hue: 0.9,
    color: { adapt: 0.3, bloom: 1.2, vignette: 0.5, ca: 0.3 }, carrier: 'warp', car: { halfLife: 0.6 },
    chain: [op('zoom', { rate: 0.02 }), op('rotate', { lock: 0.0625 })],
    bodies: [body({
      shape: ['polygon', { n: 6, r: 0.06 }],
      place: ['ring', { n: 6, radius: 0.22 }],
      motion: ['spin', { rate: 0.25 }],
      material: ['line', { gain: 1, width: 2, halo: 0.25 }],
      feel: ['flow', { lock: 0, atk: 0.02, rel: 0.25 }],
    })],
    reactions: [rx('beat', 'ma', 0, 'gain', 0.4, { atk: 0.01, rel: 0.25 }), rx('bass', 'op', 0, 'rate', 0.12, { atk: 0.03, rel: 0.3 })],
    dejavu: { recall: 1, blend: 0.5, snap: 0.6, frame: 1, hue: 0.8, motion: 1, evolve: 0.5, keep: 0, res: 0.25, cap: 3, min: 0.7 },
  },
  {
    // A glowing torus knot tumbles in perspective, pushed by the waveform and swelling with the bass,
    // laying a trail of silk-like loops in warm gold on violet; it flares on the beat and punches
    // outward on drum hits. Every appearance of a returning section is
    // remembered afresh and only the two latest memories are kept, so each return brings back the
    // sculpture the last one left, pushed in and turned a good deal further and shifted in hue: the
    // choruses grow into a chain of variations, each a memory of the one before.
    origin: 'D05', name: 'Haunted Knot', energy: [0.2, 0.8], scheme: 'complementary', hue: 0.12,
    color: { sat: 1, exposure: 0.85, adapt: 0.3, bloom: 0.9, vignette: 0.5 }, carrier: 'warp', car: { halfLife: 0.5, floor: 0.4 },
    chain: [op('zoom', { rate: 0.001 }), op('swirl', { amt: 0.003, k: 2 })],
    bodies: [body({
      shape: ['superscope', { family: 0, p: 2, q: 3, size: 0.34, audio: 0.25, spec: 0, spinX: 0.0625, spinY: 0.125, persp: 0.6, n: 1024 }],
      material: ['line', { gain: 0.4, width: 1, halo: 0.1 }],
      color: ['fixed', { hue: 0, detail: 1 }],
      feel: ['flow', { lock: 0, atk: 0.05, rel: 0.6 }],
    })],
    reactions: [
      rx('bass', 'sh', 0, 'size', 0.25, { atk: 0.03, rel: 0.4 }), rx('beat', 'ma', 0, 'gain', 0.5, { rel: 0.25 }),
      rx('hit', 'sh', 0, 'size', 0.3, { atk: 0.005, rel: 0.25 }),
    ],
    dejavu: { recall: 0.8, blend: 2, snap: 0.8, frame: 0.6, hue: 0.5, motion: 0.7, evolve: 0.8, keep: 1, res: 0.5, cap: 2, min: 0.65 },
  },
];

// U01..: presets the user evolved in the app and asked to keep as seeds, tuned where they fell short.
const EVOLVED: Def[] = [
  {
    // Bred from Mycelial Star and a 3D scene: melting shapes under a slow dolly zoom, drawn as neon rims
    // that feed a slime-mould network, masked through a five-point star. The veins meander like lazy
    // rivers, merging and splitting. Tuned so they answer the music: the veins pulse brighter on every
    // beat and flash on drum hits, wiggle with each new melody note (a view-stage noise warp, so the
    // wiggle never feeds back into the network), flow faster with the bass, and drops burst them out.
    origin: 'U01', name: 'Lazy Rivers', energy: [0.28, 0.95], scheme: 'triad', hue: 0.1,
    color: { adapt: 0.36, bloom: 1.1, vignette: 0.45, contrast: 0.05, ca: 0.0015, reflectY: -0.16 },
    carrier: 'warp', car: { halfLife: 0.3, floor: 1, vort: 28, fnoise: 0.35, fscale: 2, famt: 0.0012, grain: 0.006 },
    bodies: [{
      ...body({
        shape: ['scene', { scene: 0, cam: 2, res: 0.7, size: 1.1, blend: 0.85, speed: 0.3, pulse: 0.7, kick: 0.5, vary: 0.8, rim: 0.7, ao: 0.8, fog: 0.3, glow: 0.15, roam: 0.8, gap: 0.5, spec: 0.6 }],
        material: ['line', { gain: 0.8, width: 2, halo: 0.2 }],
        emit: ['slime', { count: 393216, sa: 0.5, sd: 0.01, steer: 0.35, step: 0.0012, deposit: 0.3, decay: 0.93, diffuse: 0.5, body: 0.7, feed: 1, birth: 0.15, onDrop: 2 }],
        feel: ['flow', { atk: 0.005, rel: 0.005, div: 4 }],
        color: ['age', { rate: 0.0625, detail: 0.6 }],
      }),
      fuse: { shape: { kind: 'star', p: { n: 5, r: 0.3, inner: 0.45 } }, p: { mode: 2, k: 0.165, t: 0.36, drive: 1, depth: 0.53, rate: 4, inside: 1 } },
    }],
    chain: [op('noise', { amp: 0, scale: 2.5, speed: 0.4 }, 1, 'view')],
    reactions: [
      rx('beat', 'ma', 0, 'gain', 0.45, { atk: 0.005, rel: 0.2 }),
      rx('hit', 'ma', 0, 'gain', 0.3, { atk: 0.005, rel: 0.15 }),
      rx('noteon', 'op', 0, 'amp', 0.8, { atk: 0.01, rel: 0.35 }),
      rx('bass', 'em', 0, 'step', 0.5, { atk: 0.01, rel: 0.3 }),
      rx('held', 'op', 0, 'amp', 0.35, { atk: 0.08, rel: 0.4 }),
    ],
  },
];


// T01.. showcase timbre as material (src/v2/genes/timbre.ts): what the sound is made of becomes what the
// bodies are made of: bright sound metallic, pure tones glass, noise and distortion grit, breath velvet.
const TIMBRE: Def[] = [
  {
    // Three liquid blobs orbit and melt into one another. Their surface is the sound: a bright synth
    // lead turns them to polished chrome with a white highlight, a pure sine pad to clear glass with a
    // glowing rim, distorted guitars and hi-hat noise to sparkling grit, a breathy vocal to soft velvet
    // with a bloom around it; every sharp pluck flashes their outline.
    origin: 'T01', name: 'Timbre Blobs', energy: [0.15, 0.8], scheme: 'analogous', hue: 0.55,
    color: { adapt: 0.4, bloom: 1.1, vignette: 0.5 }, carrier: 'warp', car: { halfLife: 0.08, floor: 1 },
    bodies: [body({
      shape: ['dot', { r: 0.09 }],
      place: ['orbit', { count: 3, radius: 0.1, rate: 0.25, follow: 0.1, fuse: 0.12 }],
      motion: ['pulse', { amp: 0.12 }],
      material: ['fill', { gain: 1.3, soft: 0.15, halo: 0.3, core: 0.4 }],
      emit: ['none'],
    })],
    reactions: [rx('loud', 'ma', 0, 'gain', 0.25, { atk: 0.05, rel: 0.4 }), rx('bright', 'pl', 0, 'radius', 0.3, { atk: 0.3, rel: 1 })],
    timbre: { src: 0, sheen: 0.9, glass: 0.8, grain: 0.7, scale: 22, velvet: 0.6, edge: 0.6, emboss: 0 },
  },
  {
    // A ring of eight rounded hexagons around the centre over a wheeling fan of their halos. Each
    // drum hit kicks the panes round by a new amount, clockwise or anticlockwise, the fan reverses
    // every other bar, and the ring breathes out with the bass and rises with the melody. They read the vocals: a pure sung note turns every pane to clear glass with a
    // bright rim, a bright belted line polishes them to chrome, a breathy phrase fogs them into velvet
    // with a soft bloom, and consonants and hard onsets flash their outlines.
    origin: 'T02', name: 'Glass Choir', energy: [0.1, 0.7], scheme: 'split', hue: 0.58,
    color: { adapt: 0.4, bloom: 1.15, vignette: 0.55 }, carrier: 'warp', car: { halfLife: 0.25, floor: 1 },
    chain: [op('rotate', { lock: 0.0625, alt: 1 }), op('zoom', { rate: -0.004 })],
    bodies: [body({
      shape: ['polygon', { n: 6, r: 0.07, round: 0.5 }],
      place: ['ring', { n: 8, radius: 0.27 }],
      motion: ['hits', { amt: 0.7 }],
      material: ['fill', { gain: 1.3, soft: 0.05, halo: 0.35, core: 0.5 }],
      emit: ['none'],
      feel: ['flow', { atk: 0.03, rel: 0.4 }],
    })],
    reactions: [
      rx('vocals', 'ma', 0, 'gain', 0.35, { atk: 0.05, rel: 0.5 }), rx('bass', 'pl', 0, 'radius', 0.45, { atk: 0.03, rel: 0.35 }),
      rx('melody', 'pl', 0, 'radius', 0.3, { atk: 0.15, rel: 0.4 }),
    ],
    timbre: { src: 3, sheen: 0.7, glass: 1, grain: 0.3, scale: 30, velvet: 0.8, edge: 0.5, emboss: 0 },
  },
  {
    // A square lattice of four-point stars, each a pitch class lit by the chroma, streaming outward
    // into a tunnel. It reads the drums: crisp, noisy hats and snares grind every star into sparkling
    // grit and emboss the whole streaming picture into a rough, lit relief (glossier on bright
    // cymbals), a soft kit leaves it smooth, and every sharp hit flashes the star outlines.
    origin: 'T03', name: 'Snare Grit', energy: [0.45, 1], scheme: 'complementary', hue: 0.05,
    color: { adapt: 0.4, bloom: 1, vignette: 0.5, contrast: 0.05 }, carrier: 'warp', car: { halfLife: 0.4, floor: 0.4 },
    chain: [op('zoom', { rate: 0.01 }), op('rotate', { lock: -0.0625 })],
    bodies: [body({
      shape: ['star', { n: 4, r: 0.06, inner: 0.35 }],
      place: ['grid', { lattice: 0, scale: 7, jitter: 0, density: 0.7, lit: 1, links: 0, twinkle: 0.3, lock: 0.0625 }],
      material: ['fill', { gain: 3, soft: 0, outline: 1, core: 0.6 }],
      emit: ['trail', { tip: 0 }],
    })],
    reactions: [rx('drums', 'ma', 0, 'gain', 0.3, { atk: 0.01, rel: 0.2 }), rx('attack', 'op', 0, 'rate', 0.16, { atk: 0.03, rel: 0.4 })],
    timbre: { src: 1, sheen: 0.3, glass: 0, grain: 1, scale: 14, velvet: 0, edge: 0.9, emboss: 0.8 },
  },
  {
    // Three soft moons float on slow drifting paths, sized by their instruments. They wear the other
    // instruments' timbre: breathy pads and airy synths turn them to plush velvet wrapped in a soft
    // bloom, a clean electric piano or bell to translucent glass, a bright lead to a satin sheen,
    // and a plucked or struck note lights a thin edge around each moon.
    origin: 'T04', name: 'Velvet Moons', energy: [0, 0.6], scheme: 'triad', hue: 0.72,
    color: { adapt: 0.3, bloom: 1.25, vignette: 0.6, sat: 0.8 }, carrier: 'warp', car: { halfLife: 0.1, floor: 1 },
    bodies: [body({
      shape: ['dot', { r: 0.075 }],
      place: ['float', { count: 3, spread: 0.5, speed: 0.07 }],
      material: ['fill', { gain: 1.1, soft: 0.35, halo: 0.6, core: 0.6 }],
      emit: ['none'],
      feel: ['flow', { atk: 0.2, rel: 1.2 }],
    })],
    reactions: [rx('other', 'ma', 0, 'gain', 0.3, { atk: 0.2, rel: 1 }), rx('noisy', 'ma', 0, 'soft', 0.4, { atk: 0.3, rel: 1.2 })],
    timbre: { src: 4, sheen: 0.4, glass: 0.7, grain: 0.15, scale: 40, velvet: 1, edge: 0.5, emboss: 0 },
  },
  {
    // A slow plasma of contour bands, lit as a surface by the bass's timbre: a clean sub bass leaves
    // it a soft, barely raised sheet, a growling, distorted bass hammers it into deep, rough relief,
    // and a bright, buzzy bass turns the relief to glossy liquid metal reflecting the palette.
    origin: 'T05', name: 'Molten Bass', energy: [0.35, 1], scheme: 'split', hue: 0.08,
    color: { adapt: 0.4, bloom: 1.05, vignette: 0.5, exposure: 1.15, gloss: 0.5, bump: 1.4, light: 0.3 }, carrier: 'warp', car: { halfLife: 0.2, floor: 1 },
    chain: [op('swirl', { amt: 0.006, k: 4 })],
    bodies: [body({
      shape: ['plasma', { scale: 1.4, warp: 2.2, bands: 7, lines: 0.35, speed: 0.12, tempo: 0.4, pulse: 0.5, melHue: 0 }],
      material: ['fill', { gain: 0.9 }],
      emit: ['trail', { tip: 0 }],
    })],
    reactions: [rx('bass', 'sh', 0, 'warp', 0.35, { atk: 0.03, rel: 0.4 }), rx('rough', 'sh', 0, 'lines', 0.5, { atk: 0.1, rel: 0.6 })],
    timbre: { src: 2, sheen: 0.5, glass: 0.3, grain: 0.5, scale: 18, velvet: 0.2, edge: 0.3, emboss: 1 },
  },
];

// Y01.. showcase the lyrics gene (genes/lyrics.ts): the sung words steer the picture and are shown.
// Both are complete presets without lyrics (instrumentals, live input): the words only add to them.
const LYRICS: Def[] = [
  {
    // One wireframe dodecahedron turning a full turn per bar, flying slowly outward through its own
    // trails. The line being sung fills in karaoke style beneath it and a faint copy is drawn into
    // the trails, so every line streams away behind the solid. Each new line (or, without lyrics,
    // each vocal entry) swells the solid; the words tint it (fire warm, night deep blue, love pink),
    // dim or brighten it, lift or sink the camera, and intense lines speed the flight.
    origin: 'Y01', name: 'Sung Prism', energy: [0.3, 0.9], scheme: 'triad', hue: 0.58,
    color: { adapt: 0.35, bloom: 1.15, vignette: 0.5 }, carrier: 'warp', decay: 0.9,
    chain: [op('zoom', { rate: 0.007 }), op('rotate', { lock: 0.0625 })],
    bodies: [body({
      shape: ['solid', { solid: 5, size: 0.19, tilt: 0.5, inner: 0.8 }],
      motion: ['spin', { rate: 1 }],
      material: ['line', { gain: 1.2, width: 1.5, halo: 0.18 }],
      feel: ['flow', { atk: 0.04, rel: 0.4 }],
    })],
    reactions: [
      rx('beat', 'ma', 0, 'gain', 0.35, { rel: 0.25 }),
      rx('line', 'sh', 0, 'size', 0.35, { atk: 0.02, rel: 0.6 }),
      rx('arousal', 'op', 0, 'rate', 0.16, { atk: 1, rel: 2 }),
    ],
    lyrics: { strength: 0.75, pal: 1, tone: 1, motion: 1, chain: 0, lag: 1.2, kick: 0.6, show: 2, smear: 0.45 },
  },
  {
    // Rain on the words: the waveform, one glowing line across the frame, sheds its trail upward
    // through rippling water; drops fall on the beats and the rings bend the rising lines, the line
    // flares on every beat and swells with the vocals. The sung line shows as a quiet caption and
    // melts upward into the trails. Water words swell the ripples, storms stir them, dream words
    // swirl and blur; the mood of the song (the words', or without lyrics the music's major or minor
    // feel) slowly turns the colours.
    origin: 'Y02', name: 'Rain on the Words', energy: [0.15, 0.8], scheme: 'analogous', hue: 0.55,
    color: { adapt: 0.35, bloom: 1.15, vignette: 0.4 }, carrier: 'warp', car: { halfLife: 1.4, blur: 0.04, water: 0.75, wsize: 0.04 },
    chain: [op('ripple', { amp: 0.0012, freq: 7, speed: 0.6, radial: 1 }), op('translate', { vy: 0.07 }), op('swirl', { amt: 0.002, k: 3 })],
    bodies: [body({
      shape: ['curve', { form: 0, amp: 0.2 }],
      place: ['point', { y: -0.12 }],
      material: ['line', { gain: 0.75, width: 1.4, halo: 0.2 }],
      feel: ['flow', { atk: 0.02, rel: 0.3 }],
    })],
    reactions: [
      rx('beat', 'ma', 0, 'gain', 0.35, { rel: 0.3 }),
      rx('vocals', 'sh', 0, 'amp', 0.35, { atk: 0.08, rel: 0.6 }),
      rx('valence', 'pal', 0, 'hue', 0.25, { atk: 2, rel: 3 }),
      rx('bass', 'car', 0, 'water', 0.3, { atk: 0.03, rel: 0.5 }),
    ],
    lyrics: { strength: 0.8, pal: 1, tone: 1, motion: 1, chain: 1, lag: 2.5, kick: 0.2, show: 1, smear: 0.6 },
  },
];

// N01.. showcase the notes shape (genes/notes.ts): the melody drawn as it is played, held and
// gliding notes as ribbons bending with the pitch, detached notes as marks born at each note start.
const NOTES: Def[] = [
  {
    // One melody line drawn across the frame as it is played, the present at the right. A sung or
    // held note is a glowing ribbon that rises and falls with the pitch as it wanders, beading with
    // light where the voice has vibrato; a staccato riff breaks into a row of bright dots, one per
    // note, each lit while its note sounds and fading after, all drifting left into the past.
    origin: 'N01', name: 'Legato Line', energy: [0.15, 0.85], scheme: 'triad', hue: 0.6,
    color: { adapt: 0.35, bloom: 1.25, vignette: 0.45 }, carrier: 'warp', decay: 0.85,
    bodies: [body({
      shape: ['notes', { mode: 0, span: 4, len: 1.55, height: 0.62, now: 0.32, ribbon: 0.95, thick: 0.009, marks: 0.95, form: 0, size: 0.028, fade: 0.45, shimmer: 0.45, hues: 0.7, glow: 0.6 }],
      place: ['point', { x: 0, y: 0 }],
      material: ['glow', { gain: 1.15 }],
      emit: ['trail'],
      feel: ['flow', { atk: 0.01, rel: 0.15 }],
    })],
    reactions: [
      rx('noteon', 'ma', 0, 'gain', 0.3, { atk: 0.005, rel: 0.2 }),
      rx('legato', 'sh', 0, 'thick', 0.4, { atk: 0.4, rel: 0.8 }),
      rx('hit', 'sh', 0, 'glow', 0.3, { atk: 0.005, rel: 0.25 }),
    ],
  },
  {
    // A music box seen from above: the last six seconds of melody wind round a clock face, the
    // present at twelve o'clock and pitch pushing outward. Plucked notes flare as four-pointed sparks
    // round the rim and linger a moment after their notes end; a held or sung line draws arcs of
    // light that bend in and out with the pitch, and the trails curl inward in a slow spiral.
    origin: 'N02', name: 'Music Box Clock', energy: [0.1, 0.8], scheme: 'split', hue: 0.62,
    color: { adapt: 0.35, bloom: 1.2, vignette: 0.55 }, carrier: 'warp', decay: 0.9,
    chain: [op('rotate', { lock: 0.0625 }), op('zoom', { rate: -0.004 })],
    bodies: [body({
      shape: ['notes', { mode: 1, span: 6, len: 1.1, height: 0.55, now: 0, tilt: 0, ribbon: 0.8, thick: 0.007, marks: 1, form: 1, size: 0.045, fade: 0.6, rise: 0.05, shimmer: 0.6, hues: 0.8, glow: 0.5 }],
      place: ['point', { x: 0, y: 0 }],
      material: ['glow', { gain: 1.1 }],
      emit: ['trail'],
      feel: ['flow', { atk: 0.01, rel: 0.15 }],
    })],
    reactions: [
      rx('noteon', 'sh', 0, 'size', 0.35, { atk: 0.005, rel: 0.25 }),
      rx('legato', 'sh', 0, 'ribbon', 0.3, { atk: 0.5, rel: 1 }),
      rx('bar', 'sh', 0, 'tilt', 0.15, { atk: 0.5, rel: 1 }),
    ],
  },
  {
    // A piano roll standing on end and mirrored: notes are born along the bottom edge, placed out
    // from the middle by pitch, and rise up the frame as time passes, each a rounded bar exactly as
    // long as the note was held, so a staccato riff climbs as a scatter of short beads and a sung
    // phrase as a staircase of long stems. The rising trails smoke upward and soften on drum hits.
    origin: 'N03', name: 'Rising Keys', energy: [0.2, 0.9], scheme: 'triad', hue: 0.08,
    color: { adapt: 0.35, bloom: 1.15, vignette: 0.4 }, carrier: 'warp', decay: 0.92, car: { blur: 0.05 },
    chain: [op('translate', { vy: 0.05 })],
    bodies: [body({
      shape: ['notes', { mode: 0, span: 3, len: 1.1, height: 0.8, now: 0.42, tilt: -0.25, ribbon: 0.55, thick: 0.006, marks: 1, form: 2, size: 0.03, fade: 0.8, shimmer: 0.3, hues: 1, glow: 0.4 }],
      place: ['mirror', { axis: 0, x: 0.42, y: 0 }],
      material: ['glow', { gain: 1.05 }],
      emit: ['trail'],
      feel: ['flow', { atk: 0.01, rel: 0.15 }],
    })],
    reactions: [
      rx('noteon', 'ma', 0, 'gain', 0.3, { atk: 0.005, rel: 0.25 }),
      rx('hit', 'car', 0, 'blur', 0.3, { atk: 0.01, rel: 0.3 }),
      rx('glide', 'sh', 0, 'glow', 0.4, { atk: 0.05, rel: 0.4 }),
    ],
  },
  {
    // A night pond: the melody skims across just above the waterline and is mirrored in it. Every
    // note start drops a ring that spreads and fades on the surface, so a staccato riff rains
    // circles; a sung line glides over the water as a thin shimmering thread, beading on vibrato.
    // Note starts stir the water under the picture, which blurs the reflection.
    origin: 'N04', name: 'Ripple Pond', energy: [0.1, 0.75], scheme: 'triad', hue: 0.5,
    color: { adapt: 0.35, bloom: 1.2, vignette: 0.5, reflect: 1, reflectY: -0.04 }, carrier: 'warp', car: { halfLife: 0.5, blur: 0.03, water: 0.35, wsize: 0.05 },
    bodies: [body({
      shape: ['notes', { mode: 0, span: 5, len: 1.7, height: 0.36, now: 0.15, ribbon: 0.75, thick: 0.005, marks: 1, form: 3, size: 0.05, fade: 0.8, shimmer: 0.7, hues: 0.9, glow: 0.6 }],
      place: ['point', { x: 0, y: 0.17 }],
      material: ['glow', { gain: 1.1 }],
      emit: ['trail'],
      feel: ['flow', { atk: 0.01, rel: 0.2 }],
    })],
    reactions: [
      rx('noteon', 'car', 0, 'water', 0.35, { atk: 0.01, rel: 0.4 }),
      rx('vibrato', 'sh', 0, 'shimmer', 0.3, { atk: 0.1, rel: 0.5 }),
      rx('legato', 'sh', 0, 'thick', 0.35, { atk: 0.4, rel: 0.8 }),
    ],
  },
  {
    // Fireflies over warm smoke: each note lights a firefly at its pitch that lingers after the note
    // and drifts up as it fades, so a staccato riff sets off a swarm and a pause lets it settle; a
    // sung line trails a ribbon of glowing smoke that the air curls and carries away. Note starts
    // puff the air, drum hits stir it harder.
    origin: 'N05', name: 'Firefly Smoke', energy: [0.15, 0.85], scheme: 'analogous', hue: 0.1,
    color: { adapt: 0.35, bloom: 1.2, vignette: 0.5 }, carrier: 'fluid', car: { halfLife: 1.1, floor: 0.7, amount: 0.9, vort: 24, fnoise: 0.4 },
    bodies: [body({
      shape: ['notes', { mode: 0, span: 6, len: 1.6, height: 0.55, now: 0.3, tilt: 0.03, ribbon: 0.7, thick: 0.007, marks: 1, form: 0, size: 0.032, fade: 1.2, rise: 0.08, shimmer: 0.5, hues: 0.5, glow: 0.9 }],
      place: ['point', { x: 0, y: -0.04 }],
      material: ['glow', { gain: 1.35 }],
      emit: ['dye', { force: 0.7 }],
      feel: ['flow', { atk: 0.01, rel: 0.2 }],
    })],
    reactions: [
      rx('noteon', 'em', 0, 'force', 0.4, { atk: 0.005, rel: 0.3 }),
      rx('noteon', 'ma', 0, 'gain', 0.3, { atk: 0.005, rel: 0.25 }),
      rx('hit', 'car', 0, 'amount', 0.3, { atk: 0.01, rel: 0.4 }),
      rx('legato', 'sh', 0, 'ribbon', 0.3, { atk: 0.4, rel: 0.8 }),
    ],
  },
];

const ALL_DEFS: Def[] = [...DEFS, ...MILKDROP, ...CHOREO, ...PHYSICS, ...AVS, ...RAYMARCH, ...AGENTS, ...ECOSYSTEM, ...DRIFT, ...LANDSCAPE, ...HARMONY, ...GROOVE, ...DEJAVU, ...EVOLVED, ...TIMBRE, ...LYRICS, ...NOTES];
export const SEEDS: Seed[] = ALL_DEFS.map(build);
/** The reactions each seed was written with, before repair (the tests check repair kept every one as written). */
export const SEED_DECLARED_REACTIONS: Record<string, readonly ReactionGene[]> = Object.fromEntries(ALL_DEFS.map((d) => [d.origin, d.reactions ?? []]));
