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
import { USER_GENOMES } from './seedsUser';
import { GROOVE_SCHEMA } from './genes/groove';
import { TIMBRE_SCHEMA } from './genes/timbre';
import { DEJAVU_SCHEMA } from './genes/dejavu';
import { LYRICS_SCHEMA } from './genes/lyrics';
import { part, type CompoundPart } from './genes/compound';
// Compound parts for seed bodies: part('capsule', 'union', { y: 0.5, sx: 0.1, sy: 1, m: 0.6 }).
export { part };

export const SEED_VERSION = 168;

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
  /** A compound shape's parts (shape 'compound'; see genes/compound.ts), made with part(). */
  parts?: CompoundPart[];
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
  if (d.parts) b.shape.parts = d.parts.map((x) => ({ ...x }));
  return b;
}

interface Def {
  origin: string;
  name: string;
  energy: [number, number];
  scheme: Scheme | 'free';
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
  /** Palette parameters beyond the hue (e.g. key 0: absolute colours, the same in every song key); omitted = defaults. */
  pal?: Record<string, number>;
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
    palette: { kind: d.scheme, p: { ...defaultParams(PALETTE_SCHEMAS[d.scheme]), hue: d.hue, ...d.pal } },
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
    // brightens with its band and lifts a touch on the beat (never the whole skyline jumping); the
    // reflection follows and the sky glows with the bass.
    origin: 'E01', name: 'Night Skyline', energy: [0.3, 0.8], scheme: 'complementary', hue: 0.05,
    color: { adapt: 0.3, reflect: 1, reflectY: -0.16 }, carrier: 'warp', decay: 0.9995, car: { floor: 0.05 },
    chain: [
      op('translate', { vx: -0.2 }),
      op('stretch', { base: -0.16, amt: 0.9, beat: 0.03, strips: 32, win: 1, sky: 1 }, 1, 'view'),
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
    bodies: [body({ shape: ['edge', { mode: 2, side: 1, density: 0.6 }], material: ['fill', { gain: 0.38 }] })],
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
    // of a tunnel coming toward you as rings, turning with the bar, with an orb racing round the wall
    // that swells on every beat; the far end is lost in haze. The bass eases the flight faster and drum
    // hits flash the bands and the orb; the tunnel itself never lurches on the beat. Kept slow and dim
    // enough that the passing rings never strobe (under 3 flashes a second).
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
      rx('bass', 'op', 1, 'speed', 0.3, { atk: 0.25, rel: 0.8 }), rx('hit', 'ma', 1, 'gain', 0.6, { rel: 0.2 }),
      rx('hit', 'ma', 0, 'gain', 0.7, { rel: 0.2 }), rx('beat', 'sh', 1, 'r', 0.3, { rel: 0.25 }),
    ],
  },
  {
    // Flexi + Martin, tunnel of supraschismatika: a dark chrome pipe flown through, glints streaking
    // along its polished wall toward you; the bass eases the flight and the twist up, the glints swell on
    // every beat and flash on drum hits while the pipe itself never lurches. Flown slower than the
    // original so the passing rings never strobe.
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
    reactions: [rx('bass', 'op', 1, 'speed', 0.2, { atk: 0.25, rel: 0.8 }), rx('bass', 'op', 1, 'twist', -0.25, { atk: 0.25, rel: 0.8 }), rx('beat', 'sh', 0, 'r', 0.25, { rel: 0.25 }), rx('hit', 'ma', 0, 'gain', 0.5, { rel: 0.2 })],
  },
  {
    // Waltra, Square Orgy: a turning grid of glossy tiles, each lit by the colour behind it, as bright
    // blobs drift and bloom underneath; the grid eases wider with the bass (a slow swell, never a pump on
    // every kick) and the tiles shine like enamel.
    origin: 'M18', name: 'Square Orgy (after Waltra)', energy: [0.3, 0.9], scheme: 'triad', hue: 0.08,
    color: { sat: 1, exposure: 1.05, adapt: 0.3, bloom: 0.8, vignette: 0.2, relief: 0.6, bump: 1.4, gloss: 0.9, light: 0.3 },
    carrier: 'warp', car: { halfLife: 1.4, floor: 0.2 },
    chain: [
      op('zoom', { rate: 0.006, wander: 0.2 }),
      op('mosaic', { size: 0.13, shape: 0, gap: 0.12, angle: 0.07, lock: 0.0625, pulse: 0 }, 1, 'view'),
    ],
    bodies: [body({
      shape: ['dot', { r: 0.1 }],
      place: ['float', { count: 6, spread: 0.7, speed: 0.18 }],
      material: ['fill', { gain: 1.2, soft: 0.5 }],
      color: ['height', { amount: 1.5, detail: 1 }],
    })],
    reactions: [rx('bass', 'op', 1, 'size', 0.2, { atk: 0.2, rel: 0.8 }), rx('beat', 'ma', 0, 'gain', 0.4, { rel: 0.25 })],
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
      shape: ['beams', { count: 8, spread: 1.1, fan: 0.35, sweep: 0.55, pattern: 4, period: 1, width: 0.035, haze: 0.7, gobo: 0, hues: 0.06, length: 1.4, flare: 0.4, accent: 0.8, trig: 1 }],
      place: ['point', { x: 0, y: 0.45 }],
      material: ['glow', { gain: 1 }],
      feel: ['flow', { atk: 0.02, rel: 0.15 }],
    })],
    // Run like a lighting operator, not a metronome: every head jumps to a new aim on each drum hit
    // and holds (the riff lights one head per riff note), drum hits flare the lenses, the rig dims
    // in quiet passages and blazes when it's loud, the bass fattens the shafts, and a drop fans every
    // head out wide. The choreography dims and drains the rig through the build, punches it on the
    // drop and gives each section type its own colour.
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
// same chord, same shape. The home chord is the clean symmetric picture, every other chord bends
// it its own fixed way, and each change glides to the new shape and holds it for the chord.
const HARMONY: Def[] = [
  {
    // An eight-fold mandala of a slowly turning star, streaming outward. On the home chord the
    // mandala is perfect; every other chord slides its segments apart in its own fixed pattern and
    // an off-centre swirl pulls at the frame, further for chords further from home, so a repeating
    // progression repeats its shapes. Coming home (V-I) the segments glide back into place with a
    // lift of light. Each chord change nudges the colours along the lattice.
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
      rx('bass', 'sh', 0, 'r', 0.2, { atk: 0.03, rel: 0.3 }),
    ],
    harmony: { brk: 0.85, warp: 0.3, style: 1, snap: 0.8, settle: 0.3, walk: 0.08, kick: 0.3, modHue: 0.1, modTurn: 0.008, calm: 0.4 },
  },
  {
    // A pendulum harmonograph (its frequency ratios walk the circle of fifths) drawn four ways in a
    // mirror, leaving long warm trails that sink inward. The home chord keeps the four quarters in
    // perfect reflection; each other chord leans the reflections apart and bends the frame sideways
    // by its own fixed amount, and the colours slide away from home. The glides between chords are
    // the slowest of the set, so a cadence home lets the quarters drift back together.
    origin: 'H02', name: 'Suspended Mirror', energy: [0.1, 0.7], scheme: 'split', hue: 0.07,
    color: { bloom: 0.9, vignette: 0.55, adapt: 0.7 }, carrier: 'warp', decay: 0.975,
    chain: [op('zoom', { rate: -0.006, wander: 0.1 }), op('rotate', { lock: -0.0625 }), op('mirror', { axis: 2 }, 1, 'view')],
    bodies: [body({
      shape: ['curve', { form: 5, radius: 0.38, amp: 0.3, ra: 2, rb: 2 }],
      material: ['line', { gain: 0.4, width: 1.3, halo: 0.12 }],
      color: ['age', { rate: 0.03125, detail: 0.7 }],
    })],
    reactions: [
      rx('resolve', 'ma', 0, 'gain', 0.6, { atk: 0.01, rel: 0.8 }),
      rx('vocals', 'sh', 0, 'amp', 0.25, { atk: 0.1, rel: 0.6 }),
    ],
    harmony: { brk: 1, warp: 0.45, style: 0, snap: 0.45, settle: 1.2, walk: 0.12, kick: 0.1, modHue: 0.12, modTurn: -0.006, calm: 0.7 },
  },
  {
    // A hexagon keyboard lit by the chroma, folded into a wall of mirrored tiles. On the home chord the tiles meet
    // seamlessly like a tiled floor; on every other chord each tile turns and slips on its own
    // (every chord shuffles them its own fixed way, so the same chord always gives the same wall)
    // and the whole wall buckles; coming home they glide back flush with a bright lift of light.
    // Key changes swing the wall's hue far round the wheel and tilt it, so a modulation reads as a new room.
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
      rx('modulation', 'col', 0, 'exposure', 0.4, { atk: 0.02, rel: 1.5 }),
    ],
    harmony: { brk: 0.9, warp: 0.5, style: 2, snap: 1, settle: 0.15, walk: 0.1, kick: 0.4, modHue: 0.22, modTurn: 0.012, calm: 0.2 },
  },
  {
    // The harmony map itself: the tonal lattice (fifths across, thirds on the diagonals) with every
    // note node glowing as it sounds. The current chord's triangle burns, the last chords leave a
    // fading path, and the camera drifts after the walk. Each chord away from home bends the lattice
    // off true in its own fixed way; a cadence home glides it straight with a lift of light, and a
    // key change turns the whole map.
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
    // the reflection is exact; each other chord tilts the reflection and buckles the scene in its
    // own fixed way, gliding there on the change and holding still for the whole chord, so the
    // water ripples in time with the progression; a cadence home glides it flat again with a lift
    // of light. Key changes swing the palette.
    origin: 'H05', name: 'Reflecting Pool', energy: [0.3, 1], scheme: 'complementary', hue: 0.78,
    color: { bloom: 1.2, vignette: 0.5, contrast: 0.05 }, carrier: 'warp', decay: 0.94,
    chain: [op('translate', { vy: 0.06 }), op('mirror', { axis: 1 }, 1, 'view')],
    bodies: [body({
      shape: ['bars', { mode: 0, bins: 48, radius: 0.18, len: 0.35, fill: 0.6 }],
      place: ['point', { x: 0, y: 0.02 }],
      material: ['line', { gain: 1, width: 1.6, halo: 0.2 }],
    })],
    reactions: [
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
    // chroma, the whole lattice turning on the bar. On a machine-tight track its turn catches a little
    // on every beat like a watch movement (a soft tick, never a snap of the whole lattice) as every star
    // kicks in size and flares; the bass swells the stars. Loose, swung playing melts the ticks into a
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
    groove: { swing: 0.8, sub: 16, sway: 0, off: 0.2, lean: 0.3, crisp: 0.25, tick: 1, jitter: 0, accent: 0.3 },
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
    // drum hit turns the panes by a new amount, clockwise or anticlockwise, easing round over a fifth
    // of a second (a turn of the shapes, not a jolt of the frame), the fan reverses every other bar, and the ring breathes out with the bass and rises with the melody. They read the vocals: a pure sung note turns every pane to clear glass with a
    // bright rim, a bright belted line polishes them to chrome, a breathy phrase fogs them into velvet
    // with a soft bloom, and consonants and hard onsets flash their outlines.
    origin: 'T02', name: 'Glass Choir', energy: [0.1, 0.7], scheme: 'split', hue: 0.58,
    color: { adapt: 0.4, bloom: 1.15, vignette: 0.55 }, carrier: 'warp', car: { halfLife: 0.25, floor: 1 },
    chain: [op('rotate', { lock: 0.0625, alt: 1 }), op('zoom', { rate: -0.004 })],
    bodies: [body({
      shape: ['polygon', { n: 6, r: 0.07, round: 0.5 }],
      place: ['ring', { n: 8, radius: 0.27 }],
      motion: ['hits', { amt: 0.45, glide: 0.2 }],
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

// X21.. ART3: glass paperweights (clear spheres as lenses with bubbles and vivid suspended designs) and
// pressed flowers on pale paper (the paper is a light field the ink cuts into with subtractive bodies).
const ART3: Def[] = [
  {
    // A bioluminescent jellyfish seen from the side, drifting slowly round the dark water: a soft
    // glowing bell that never holds a perfect circle (wobbling lobes), and below it long tentacles
    // that curl and sway and stream behind it as it moves. Light travels down the tentacles: each
    // melody note start sends a spark of light down them, held notes long waves, the colour
    // following the melody as it flows. Drum hits contract the bell and thrust the creature upward,
    // the bass deepens its glow; faint motes drift in the water around it.
    origin: 'X21', name: 'Pandora Jelly', energy: [0.1, 0.85], scheme: 'triad', hue: 0.5,
    color: { adapt: 0.15, bloom: 1.25, vignette: 0.55 }, carrier: 'warp', car: { halfLife: 0.7, floor: 0.3, blur: 0.18 },
    chain: [op('translate', { vy: -0.1 }), op('noise', { amp: 0.0014, scale: 2.5, speed: 0.35 })],
    bodies: [
      body({
        shape: ['dot', { r: 0.17 }],
        place: ['point', { x: 0, y: 0.14 }],
        motion: ['circle', { radius: 0.07, period: 16 }],
        deform: ['wobble', { lobes: 3, amp: 0.1, rate: 0.125 }],
        material: ['chrome', { gain: 1.1, chrome: 0.3 }],
        emit: ['none'],
        feel: ['flow', { atk: 0.02, rel: 0.4 }],
        color: ['fixed', { hue: 0.1 }],
      }),
      body({
        shape: ['dot', { r: 0.12 }],
        place: ['point', { x: 0, y: 0.1 }],
        motion: ['circle', { radius: 0.07, period: 16 }],
        deform: ['arms', { count: 6, reach: 1, width: 0.15, curl: 1, sway: 0.8, turn: 16 }],
        material: ['line', { gain: 0.42, width: 1.3, halo: 0.35 }],
        emit: ['trail'],
        feel: ['flow', { atk: 0.01, rel: 0.25 }],
        color: ['age', { hue: 0.3, rate: 0.25, detail: 0.5 }],
      }),
      body({
        shape: ['dot', { r: 0.002 }],
        place: ['grid', { lattice: 1, scale: 7, jitter: 0.6, density: 0.2, lit: 1, links: 0, twinkle: 0.8 }],
        motion: ['drift', { vx: 0.004, vy: 0.008 }],
        material: ['glow', { gain: 0.06, base: 0.3, width: 0.003 }],
        emit: ['none'],
        feel: ['flow', { atk: 0.1, rel: 0.8 }],
      }),
    ],
    timbre: { src: 0, sheen: 0.15, glass: 1, grain: 0, scale: 18, velvet: 0.2, edge: 0.3, emboss: 0 },
    reactions: [
      rx('hit', 'sh', 0, 'r', -0.7, { atk: 0.005, rel: 0.35 }),
      rx('hit', 'pl', 0, 'y', 0.3, { atk: 0.03, rel: 1.2 }),
      rx('hit', 'pl', 1, 'y', 0.3, { atk: 0.03, rel: 1.2 }),
      rx('surge', 'op', 0, 'vy', -0.5, { atk: 0.05, rel: 0.6 }),
      rx('noteon', 'ma', 1, 'gain', 0.8, { atk: 0.005, rel: 0.2 }),
      rx('legato', 'cm', 1, 'hue', 0.5, { atk: 0.4, rel: 1.2 }),
    ],
  },
  {
    // A bubble paperweight: a clear glass sphere packed with glinting bubbles, a cloud of pink ink
    // swirling in its lower half. The bass swells and brightens the bubbles, drum hits stir the ink,
    // each new section recolours it, and the riff makes a handful of bubbles flare.
    origin: 'X23', name: 'Bubble Glass', energy: [0.1, 0.85], scheme: 'triad', hue: 0.9,
    color: { adapt: 0.15, bloom: 1.1, vignette: 0.55, reflect: 1, reflectY: -0.31 }, carrier: 'warp', car: { halfLife: 1.8, floor: 0.4, blur: 0.02 },
    chain: [op('swirl', { amt: 0.012, k: 4 }), op('zoom', { rate: -0.006 }), op('noise', { amp: 0.0018, scale: 4, speed: 0.35 })],
    bodies: [
      body({
        shape: ['dot', { r: 0.25 }],
        material: ['fill', { gain: 0.13, soft: 0.02, outline: 1, core: 1, halo: 0.3 }],
        emit: ['none'],
        feel: ['flow', { atk: 0.03, rel: 0.4 }],
        color: ['height', { hue: 0.45, amount: 0.5, detail: 1 }],
      }),
      body({
        shape: ['dot', { r: 0 }],
        place: ['orbit', { count: 3, radius: 0.13, x: 0, y: -0.02, follow: 0, rate: 0, fuse: 0 }],
        material: ['glow', { gain: 0.4, width: 0.012 }],
        emit: ['trail'],
        feel: ['flow', { atk: 0.01, rel: 0.2 }],
        color: ['instrument', { hue: 0, amount: 1 }],
      }),
      body({
        shape: ['notes', { mode: 0, span: 2.2, len: 0.5, height: 0.36, now: 0.5, tilt: -0.25, ribbon: 0, marks: 1, form: 3, size: 0.006, fade: 1.6, rise: 0, shimmer: 0, hues: 0.2, glow: 0.3 }],
        material: ['glow', { gain: 1.3 }],
        emit: ['none'],
        feel: ['flow', { atk: 0.01, rel: 0.2 }],
        color: ['fixed', { hue: 0.5 }],
      }),
    ],
    timbre: { src: 0, sheen: 0.3, glass: 1, grain: 0, scale: 18, velvet: 0.2, edge: 0.2, emboss: 0 },
    reactions: [
      rx('bass', 'sh', 0, 'r', 0.3, { atk: 0.03, rel: 0.4 }),
      rx('bass', 'ma', 0, 'gain', 0.5, { atk: 0.03, rel: 0.4 }),
      rx('hit', 'op', 0, 'amt', 0.5, { atk: 0.02, rel: 0.5 }),
      rx('hit', 'pl', 1, 'radius', 0.4, { atk: 0.02, rel: 0.4 }),
      rx('noteon', 'ma', 1, 'gain', 0.4, { atk: 0.01, rel: 0.3 }),
      rx('other', 'op', 2, 'amp', 0.5, { atk: 0.1, rel: 0.6 }),
    ],
  },
  {
    // Millefiori: a slab of glass-cane slices, a honeycomb of little flowers in jewel colours, each
    // cane a glowing ring round a star-shaped flower. Each cane lights with its pitch class as the
    // melody and chords sound it, so the slab sparkles in the song's harmony; each chord change reshapes
    // the flowers (their petals open and narrow) and slides the canes out of register by that chord's
    // own fixed amount (home keeps them exact), the bass swells the rings, and each drum hit turns the slab a notch.
    // the flowers (their petals open and narrow), harmonic tension slides the canes out of register and
    // a resolution clicks them back, the bass swells the rings, and each drum hit eases the slab a little
    // further round (a glide, never a jolt of the whole slab).
    origin: 'X29', name: 'Millefiori', energy: [0.1, 0.85], scheme: 'triad', hue: 0.0,
    color: { adapt: 0.2, bloom: 1.1, vignette: 0.7, ca: 0 }, carrier: 'none',
    chain: [op('kaleido', { n: 8, lock: 0 }, 1, 'view')],
    bodies: [
      {
        ...body({
          shape: ['dot', { r: 0.05 }],
          place: ['grid', { lattice: 1, scale: 7, jitter: 0, density: 1, lit: 1, links: 0, twinkle: 0.2 }],
          material: ['line', { gain: 1, width: 1.8, halo: 0.08 }],
          emit: ['none'],
          feel: ['flow', { atk: 0.02, rel: 0.4 }],
          color: ['pitch', { hue: 0, detail: 0.5 }],
        }),
        deform: { kind: 'none', p: {}, ops: [op('rotate', { lock: 0, rate: 0 })] },
      },
      {
        ...body({
          shape: ['star', { n: 8, r: 0.042, inner: 0.35 }],
          place: ['grid', { lattice: 1, scale: 7, jitter: 0, density: 1, lit: 1, links: 0, twinkle: 0.2 }],
          material: ['fill', { gain: 1.6, soft: 0.2, core: 0.4 }],
          emit: ['none'],
          feel: ['flow', { atk: 0.02, rel: 0.4 }],
          color: ['pitch', { hue: 0.25, detail: 0.5 }],
        }),
        deform: { kind: 'none', p: {}, ops: [op('rotate', { lock: 0, rate: 0 })] },
      },
    ],
    harmony: { brk: 0.2, warp: 0.1, style: 1, snap: 0.5, settle: 0.6, walk: 0.1, kick: 0.2, modHue: 0.1, modTurn: 0.006, calm: 0.3 },
    reactions: [
      rx('hit', 'dr', 0, 'rate', 0.4, { atk: 0.25, rel: 1 }),
      rx('hit', 'dr', 3, 'rate', 0.4, { atk: 0.25, rel: 1 }),
      rx('chordchange', 'sh', 1, 'inner', 1, { atk: 0.05, rel: 0.8 }),
      rx('bass', 'sh', 1, 'r', 0.2, { atk: 0.03, rel: 0.4 }),
      rx('noteon', 'ma', 0, 'gain', 0.5, { atk: 0.01, rel: 0.3 }),
      rx('vocals', 'ma', 1, 'gain', 0.4, { atk: 0.05, rel: 0.4 }),
    ],
  },
];

// ------------------------------------------------------------ art direction, second set
// X11..X20: hand-directed presets built only from existing genes, each one clear focal idea with
// the music locked to it through specific events (hits, riff notes, chords, held notes).
const ART2: Def[] = [
  {
    // A smoke machine on the stage floor: four moving heads stand at the foot of the frame and throw
    // crossing shafts up through real simulated smoke. The rig moves only when the music tells it to:
    // on each drum hit every head snaps to a new aim and holds it, and the smoke machine puffs a curl
    // of smoke into the shafts that keeps rolling on its own. Each riff note flares one head (the same
    // head for the same note every repeat) and opens the fan, the bass fattens the shafts, the voice
    // thickens the haze, quiet passages dim the rig, and the gel colour follows the melody.
    origin: 'X11', name: 'Smoke Machine', energy: [0.3, 1], scheme: 'analogous', hue: 0.58,
    color: { sat: 0.8, adapt: 0.3, bloom: 1.3, vignette: 0.3 }, carrier: 'fluid', car: { halfLife: 0.8, floor: 0.55, amount: 1.2, vort: 24, fnoise: 0.2, blur: 0.02 },
    bodies: [body({
      shape: ['beams', { count: 4, spread: 1.6, fan: 0.55, sweep: 0.5, pattern: 4, period: 1, width: 0.045, haze: 0.9, gobo: 3, hues: 0.1, length: 2.2, flare: 0.5, accent: 1, trig: 1 }],
      place: ['point', { x: 0, y: -0.45 }],
      material: ['glow', { gain: 1.1 }],
      emit: ['dye', { force: 1.5 }],
      feel: ['flow', { atk: 0.02, rel: 0.4 }],
      color: ['melody', { amount: 0.45, detail: 0.3 }],
    })],
    reactions: [
      rx('vocals', 'sh', 0, 'haze', 0.5, { atk: 0.05, rel: 0.5 }),
      rx('bass', 'sh', 0, 'width', 0.9, { atk: 0.02, rel: 0.3 }),
      rx('hook', 'sh', 0, 'fan', 0.7, { atk: 0.005, rel: 0.4 }),
      rx('loud', 'ma', 0, 'gain', 0.9, { atk: 0.1, rel: 0.8 }),
    ],
  },
  {
    // A cratered moon hangs in the dark with five slow arms curling off its rim, and on its face the
    // sand of a round Chladni plate draws the chord: every chord change scatters the grains and they
    // settle into a new nodal mandala across the lunar surface. The bass pushes the arms out, drum
    // hits make the grains jump, the melody curls the arms, and a drop solarises the whole moon.
    origin: 'X12', name: 'Lunar Chladni', energy: [0.1, 0.8], scheme: 'split', hue: 0.1,
    color: { sat: 1, adapt: 0.1, exposure: 0.8, bloom: 0.9, vignette: 0.5, contrast: 0.06 }, carrier: 'none',
    bodies: [
      body({
        shape: ['dot', { r: 0.3 }],
        place: ['point', { x: 0, y: 0.02 }],
        motion: ['bob', { amp: 0.6 }],
        deform: ['arms', { count: 5, reach: 0.8, width: 0.2, curl: 0.9, sway: 1, turn: 16 }],
        material: ['fill', { gain: 1.1, soft: 0.35, halo: 0.5, core: 0.5 }],
        emit: ['none'],
        feel: ['flow', { atk: 0.05, rel: 0.5 }],
        color: ['fixed', { hue: 0.5 }],
      }),
      body({
        shape: ['cymatics', { plate: 1, size: 0.28, modes: 7, source: 0, hold: 1, settle: 0.5, sand: 0.7, line: 1.2, shake: 0.3, rim: 0.2 }],
        place: ['point', { x: 0, y: 0.02 }],
        material: ['glow', { gain: 0.8 }],
        emit: ['none'],
        feel: ['flow', { atk: 0.01, rel: 0.12 }],
        color: ['fixed', { hue: 0 }],
      }),
    ],
    reactions: [
      rx('bass', 'de', 0, 'reach', 0.6, { atk: 0.03, rel: 0.4 }),
      rx('held', 'de', 0, 'width', 0.6, { atk: 0.05, rel: 0.5 }),
      rx('hit', 'ma', 1, 'gain', 0.6, { atk: 0.005, rel: 0.2 }),
      rx('chordchange', 'sh', 1, 'line', 0.7, { atk: 0.005, rel: 0.5 }),
      rx('drop', 'ma', 0, 'gain', 0.5, { atk: 0.02, rel: 2 }),
    ],
  },
  {
    // Sumi ink in gold on black lacquer: the melody is written across the frame as brush strokes that
    // bleed into a slow fluid and keep spreading after the brush has passed, so a phrase leaves a
    // painted sheet behind it. Each drum hit presses the brush: the ink flares and a plume of pigment
    // curls off the stroke. The whole picture is lit as a raised surface, so the ink stands up like
    // wet lacquer. Note starts dab bigger marks, the bass thickens the stroke, held (legato) phrases
    // glow, the riff draws the same strokes every repeat, and a new section wipes to a fresh sheet.
    origin: 'X13', name: 'Sumi Calligraphy', energy: [0.1, 0.8], scheme: 'mono', hue: 0.075, pal: { key: 0, spread: 0.6 },
    color: { sat: 0.4, adapt: 0.25, bloom: 0.8, vignette: 0.5, relief: 0.9, bump: 2.5, gloss: 0.4, light: 0.3 },
    carrier: 'fluid', car: { halfLife: 6, floor: 0.12, amount: 0.8, vort: 16, fnoise: 0.15, blur: 0.01 },
    bodies: [body({
      shape: ['notes', { mode: 0, span: 5, len: 1.8, height: 0.75, now: 0.3, tilt: 0.02, ribbon: 1, thick: 0.016, marks: 0.9, form: 0, size: 0.022, fade: 0.8, rise: 0, shimmer: 0.25, hues: 0.15, glow: 0.12 }],
      place: ['point', { x: 0, y: 0 }],
      material: ['glow', { gain: 1 }],
      emit: ['dye', { force: 1 }],
      feel: ['flow', { atk: 0.01, rel: 0.2 }],
      color: ['melody', { amount: 0.3 }],
    })],
    reactions: [
      rx('bass', 'sh', 0, 'thick', 0.7, { atk: 0.03, rel: 0.4 }),
      rx('hit', 'ma', 0, 'gain', 0.5, { atk: 0.005, rel: 0.3 }),
      rx('noteon', 'sh', 0, 'size', 0.6, { atk: 0.005, rel: 0.3 }),
      rx('legato', 'sh', 0, 'glow', 0.5, { atk: 0.3, rel: 0.8 }),
      rx('section', 'car', 0, 'floor', 1, { atk: 0.05, rel: 1.5 }),
    ],
  },
  {
    // Moon tide: a glowing blue moon with five slow curling arms hangs over black water, and the
    // melody skims the surface below it: every note start drops a ring on the water that swells as
    // the note lands, a held note draws a thread across it, and the whole scene is mirrored in the
    // water. The arms reach with the instruments, the bass lifts the moon, drum hits brighten it, and
    // a drop swells it full. Colours are literal (moonlit blue and gold), not the song key.
    origin: 'X15', name: 'Moon Tide', energy: [0.05, 0.75], scheme: 'complementary', hue: 0.58, pal: { key: 0 },
    color: { sat: 0.75, adapt: 0.2, bloom: 1.0, vignette: 0.45, reflect: 1, reflectY: -0.24 },
    carrier: 'warp', car: { halfLife: 0.12, floor: 0.6, blur: 0.02, water: 0.35, wsize: 0.05 },
    bodies: [
      body({
        shape: ['notes', { mode: 0, span: 5, len: 1.75, height: 0.22, now: 0.2, tilt: 0, ribbon: 0.6, thick: 0.005, marks: 1, form: 3, size: 0.035, fade: 0.9, rise: 0, shimmer: 0.6, hues: 0.6, glow: 0.5 }],
        place: ['point', { x: 0, y: -0.15 }],
        material: ['glow', { gain: 1.1 }],
        emit: ['trail'],
        feel: ['flow', { atk: 0.01, rel: 0.2 }],
        color: ['fixed', { hue: 0.06 }],
      }),
      body({
        shape: ['dot', { r: 0.075 }],
        place: ['point', { x: 0, y: 0.13 }],
        deform: ['arms', { count: 5, reach: 0.9, width: 0.2, curl: 0.9, sway: 1, turn: 16 }],
        material: ['glow', { gain: 1.2, width: 0.03, base: 0.7, halo: 0.3 }],
        emit: ['none'],
        feel: ['flow', { atk: 0.05, rel: 0.5 }],
        color: ['fixed', { hue: 0 }],
      }),
    ],
    reactions: [
      rx('noteon', 'sh', 0, 'size', 0.5, { atk: 0.005, rel: 0.3 }),
      rx('bass', 'pl', 1, 'y', 0.12, { atk: 0.05, rel: 0.5 }),
      rx('hit', 'ma', 1, 'gain', 0.5, { atk: 0.005, rel: 0.25 }),
      rx('drop', 'sh', 1, 'r', 0.15, { atk: 0.05, rel: 2 }),
    ],
  },
  {
    // Chord rose: one rose drawn in fine stippled light, folded by a five-way kaleidoscope into a
    // flower whose shape is the chord. The home chord is the clean symmetric flower; every other chord
    // has its own fixed shape (its folds slide out of register and the flower leans by the chord's
    // place on the tonal lattice) and its own colour, so on each chord change the rose glides in about
    // a third of a second to that chord's shape and holds it while the chord lasts, and a repeating
    // progression repeats its flowers. Drum hits brighten it, the bass swells it, note starts tip it
    // in depth, and the dots drift slowly on a flow field like pollen in still air.
    origin: 'X16', name: 'Chord Rose', energy: [0.1, 0.8], scheme: 'triad', hue: 0.9,
    color: { adapt: 0.3, bloom: 1.15, vignette: 0.45 }, carrier: 'flow', car: { halfLife: 0.4, blur: 0.05, floor: 0.6 },
    chain: [op('kaleido', { n: 5, lock: 0 }, 1, 'view')],
    bodies: [body({
      shape: ['superscope', { family: 2, p: 3, q: 1, size: 0.42, audio: 0.12, spec: 0, spinX: 0, spinY: 0, persp: 0, n: 2048 }],
      place: ['point', { x: 0, y: 0 }],
      material: ['dots', { gain: 1.2, spacing: 0.01, size: 0.5 }],
      emit: ['trail', { tip: 0.3 }],
      feel: ['flow', { atk: 0.02, rel: 0.5 }],
      color: ['pitch', { amount: 0.6 }],
    })],
    harmony: { brk: 1, warp: 0.6, style: 1, snap: 0.3, settle: 0.3, walk: 0.2, kick: 0, calm: 0.4 },
    reactions: [
      rx('hit', 'ma', 0, 'gain', 0.4, { atk: 0.005, rel: 0.25 }),
      rx('bass', 'sh', 0, 'size', 0.15, { atk: 0.03, rel: 0.35 }),
      rx('noteon', 'sh', 0, 'persp', 0.4, { atk: 0.005, rel: 0.3 }),
    ],
  },
  {
    // Eclipse corona: the sun's corona as a living thing. A dark disc sits at the centre and from its
    // rim a crown of slime-mould filaments streams out across the whole frame, branching and looping
    // like the sun's magnetic field lines, gold and rose on black. The bass lengthens the streamers,
    // each drum hit flares the rim and lays bright new filament, the riff sends fresh prominences out
    // of the rim, the voice widens how far the filaments spread, and a drop is totality: every
    // filament bursts out from the disc at once and the corona regrows.
    origin: 'X18', name: 'Eclipse Corona', energy: [0.15, 0.9], scheme: 'analogous', hue: 0.08, pal: { key: 0 },
    color: { sat: 0.85, adapt: 0.1, exposure: 0.75, bloom: 0.8, vignette: 0.5, contrast: 0.06 }, carrier: 'warp', car: { halfLife: 0.3, floor: 1 },
    chain: [op('zoom', { rate: 0.003 })],
    accent: { hue: 0 },
    bodies: [
      body({
        shape: ['dot', { r: 0.2 }],
        place: ['point', { x: 0, y: 0 }],
        material: ['line', { gain: 1.2, width: 2.5, halo: 0.5 }],
        emit: ['slime', { count: 393216, sa: 0.25, sd: 0.03, steer: 0.12, step: 0.0015, deposit: 0.22, decay: 0.9, diffuse: 0.12, body: 0.8, feed: 1, birth: 0.35, onDrop: 2 }],
        feel: ['flow', { atk: 0.02, rel: 0.3 }],
        color: ['fixed', { hue: 0 }],
      }),
      body({
        shape: ['dot', { r: 0.195 }],
        place: ['point', { x: 0, y: 0 }],
        material: ['fill', { gain: 0.02, soft: 0.05 }],
        emit: ['cover', { amt: 1, tip: 0 }],
        color: ['fixed', { hue: 0 }],
      }),
    ],
    reactions: [
      rx('bass', 'em', 0, 'step', 0.4, { atk: 0.05, rel: 0.4 }),
      rx('hit', 'em', 0, 'deposit', 0.6, { atk: 0.005, rel: 0.25 }),
      rx('hit', 'ma', 0, 'gain', 0.5, { atk: 0.005, rel: 0.25 }),
      rx('hook', 'em', 0, 'birth', 0.7, { atk: 0.005, rel: 0.3 }),
      rx('vocals', 'em', 0, 'sa', 0.3, { atk: 0.2, rel: 0.8 }),
    ],
  },
  {
    // Polaron: a wireframe octahedron hangs at the centre like a crystal, and the melody winds round it
    // as a ring of light, the present at twelve o'clock: held notes draw arcs that bend with the
    // pitch, short notes flare as sparks round the rim. The crystal answers the tune: each note start
    // makes it swell, held notes tip it toward the viewer and the drums turn it to a new face (both
    // stepping on the beat and holding), and the bass breathes its light. Short crisp trails of its
    // edges linger as it moves.
    origin: 'X19', name: 'Polaron', energy: [0.1, 0.85], scheme: 'triad', hue: 0.6,
    color: { sat: 0.85, adapt: 0.3, bloom: 1.2, vignette: 0.45, contrast: 0.04 }, carrier: 'warp', car: { halfLife: 0.16, blur: 0.02 },
    bodies: [
      body({
        shape: ['notes', { mode: 1, span: 6, len: 1.55, height: 0.4, now: 0, tilt: 0, ribbon: 0.85, thick: 0.007, marks: 1, form: 1, size: 0.04, fade: 0.8, rise: 0, shimmer: 0.6, hues: 0.8, glow: 0.5 }],
        place: ['point', { x: 0, y: 0 }],
        material: ['glow', { gain: 1.3 }],
        emit: ['trail'],
        feel: ['flow', { atk: 0.01, rel: 0.2 }],
        color: ['fixed'],
      }),
      body({
        shape: ['solid', { solid: 2, size: 0.19, tilt: 0.3, inner: 0.6 }],
        place: ['point', { x: 0, y: 0 }],
        motion: ['none'],
        material: ['line', { gain: 1.2, width: 1.6, halo: 0.2 }],
        emit: ['trail'],
        feel: ['flow', { atk: 0.02, rel: 0.3 }],
        color: ['fixed', { hue: 0.33 }],
      }),
    ],
    reactions: [
      rx('noteon', 'sh', 1, 'size', 0.35, { atk: 0.005, rel: 0.25 }),
      rx('held', 'sh', 1, 'tilt', 0.5, { atk: 0.08, rel: 0.08, q: 1, div: 1 }),
      rx('bass', 'ma', 1, 'gain', 0.4, { atk: 0.03, rel: 0.4 }),
      rx('drums', 'pl', 1, 'angle', 0.5, { atk: 0.06, rel: 0.06, q: 1, div: 1 }),
    ],
  },
  {
    // Void pearl: one luminous pearl hangs in a slow, dark fluid, and four jets of light pour out of
    // its rim, one per instrument, marbling the fluid around it into blue and silver smoke that is lit
    // as a glossy raised surface. The jets fire with their instruments (the drum jet kicks on each
    // hit, the others flow with their stems) and aim anew on their hits, so the marbling is the
    // arrangement; drum hits also ripple the whole surface. The bass swells the pearl, the voice
    // pushes every jet harder, the pearl's colour follows the melody and glows on each note start,
    // and a drop floods it bright.
    origin: 'X20', name: 'Void Pearl', energy: [0.1, 0.85], scheme: 'analogous', hue: 0.55, pal: { key: 0 },
    color: { sat: 0.6, adapt: 0.2, bloom: 1.1, vignette: 0.5, relief: 0.5, bump: 1.5, gloss: 0.7, metal: 0.3 }, accent: { hue: 0 },
    carrier: 'fluid', car: { halfLife: 4, floor: 0.25, amount: 1.1, vort: 26, fnoise: 0.3, water: 0.35, wsize: 0.06 },
    bodies: [
      body({
        shape: ['dot', { r: 0 }],
        place: ['orbit', { count: 4, radius: 0.2, rate: 0, follow: 0 }],
        material: ['glow', { gain: 1, width: 0.02 }],
        emit: ['dye', { force: 1.2 }],
        feel: ['flow', { atk: 0.01, rel: 0.2 }],
        color: ['instrument'],
      }),
      body({
        shape: ['dot', { r: 0.17 }],
        place: ['point', { x: 0, y: 0 }],
        material: ['fill', { gain: 1.1, soft: 0.2, halo: 0.5, core: 0.8 }],
        emit: ['none'],
        feel: ['flow', { atk: 0.03, rel: 0.4 }],
        color: ['melody', { amount: 0.4 }],
      }),
    ],
    reactions: [
      rx('bass', 'sh', 1, 'r', 0.6, { atk: 0.04, rel: 0.4 }),
      rx('hit', 'car', 0, 'water', 0.5, { atk: 0.005, rel: 0.4 }),
      rx('vocals', 'em', 0, 'force', 0.5, { atk: 0.1, rel: 0.6 }),
      rx('drop', 'ma', 1, 'gain', 0.5, { atk: 0.05, rel: 2 }),
      rx('noteon', 'ma', 1, 'core', 0.4, { atk: 0.005, rel: 0.3 }),
    ],
  },
];

// X01.. new artistic presets built from existing genes, each around one focal idea tied to specific
// musical events.
const ART: Def[] = [
  {
    // Melody Mycelium: a living slime-mould network seen as a six-fold rose window. The melody is
    // written into it as food: each note and held phrase is laid down where the veins can reach it, so
    // they race to the tune and thicken into glowing spokes and rings, the notes lit only through the
    // veins that feed on them. The bass drives the agents faster so the whole window surges and
    // re-routes, drum hits make them lay down bright bursts of trail and flash the web, note starts
    // feed bigger knots, held notes wider roots, the riff widens their senses so the web opens into
    // bigger loops, and a drop scatters every agent into haze from which the window regrows.
    origin: 'X01', name: 'Melody Mycelium', energy: [0.15, 0.85], scheme: 'analogous', hue: 0.45,
    color: { adapt: 0.35, bloom: 1.15, vignette: 0.4, contrast: 0.05 }, chain: [op('kaleido', { n: 6, lock: 0 }, 1, 'view')], carrier: 'warp', car: { halfLife: 0.3, floor: 1 },
    bodies: [body({
      shape: ['notes', { mode: 0, span: 6, len: 1.8, height: 0.6, now: 0.32, ribbon: 1, thick: 0.016, marks: 1, form: 0, size: 0.055, fade: 1.5, rise: 0, shimmer: 0.3, hues: 0.4, glow: 0 }],
      place: ['point', { x: 0, y: 0.12 }],
      material: ['glow', { gain: 1 }],
      emit: ['slime', { count: 262144, sa: 0.45, sd: 0.022, steer: 0.7, step: 0.0015, deposit: 0.25, decay: 0.96, diffuse: 0.3, body: 0.6, feed: 1, birth: 0.03, onDrop: 1 }],
      feel: ['flow', { atk: 0.01, rel: 0.25 }],
      color: ['melody', { amount: 0.4 }],
    })],
    reactions: [
      rx('bass', 'em', 0, 'step', 0.45, { atk: 0.03, rel: 0.35 }),
      rx('hit', 'em', 0, 'deposit', 0.6, { atk: 0.005, rel: 0.25 }),
      rx('hook', 'em', 0, 'sa', 0.3, { atk: 0.01, rel: 0.4 }),
      rx('held', 'sh', 0, 'thick', 0.5, { atk: 0.05, rel: 0.4 }),
      rx('hit', 'ma', 0, 'gain', 0.5, { atk: 0.005, rel: 0.2 }),
      rx('noteon', 'sh', 0, 'size', 0.5, { atk: 0.005, rel: 0.3 }),
    ],
  },
  {
    // Rack Focus: hoops of light stream toward the viewer down a tunnel, folded four ways into one
    // great symmetric iris that fills the frame, its lens-shaped pupil dead centre, in a cool blue to
    // violet that stays the same in every key. Drum hits flash the hoops and punch them outward, the
    // bass breathes their depth, every melody note ripples them with the sound itself, held notes
    // thicken them like a lens pulling focus, and each chord change shifts their colour a step.
    origin: 'X08', name: 'Rack Focus', energy: [0.15, 0.95], scheme: 'analogous', hue: 0.55, pal: { key: 0, spread: 1.2 },
    color: { sat: 0.7, adapt: 0.3, bloom: 1.2, vignette: 0.4, ca: 0.0015 },
    chain: [op('mirror', { axis: 2 }, 1, 'view')], carrier: 'warp', car: { halfLife: 0.15, floor: 1, blur: 0.05, sharpen: 0.1 },
    bodies: [body({
      shape: ['superscope', { family: 5, p: 6, q: 1, size: 0.42, audio: 0.3, spec: 0, spinX: 0, spinY: 0, persp: 0.8, n: 2048 }],
      place: ['point', { x: 0, y: 0, angle: 0.5 }],
      material: ['line', { gain: 0.7, width: 1.3, halo: 0.35 }],
      emit: ['trail'],
      feel: ['flow', { atk: 0.02, rel: 0.4 }],
      color: ['height', { amount: 0.6, detail: 0.5 }],
    })],
    reactions: [
      rx('hit', 'ma', 0, 'gain', 1, { atk: 0.005, rel: 0.2 }),
      rx('hit', 'sh', 0, 'size', 0.35, { atk: 0.005, rel: 0.25 }),
      rx('bass', 'sh', 0, 'persp', 0.3, { atk: 0.03, rel: 0.3 }),
      rx('held', 'ma', 0, 'width', 0.6, { atk: 0.05, rel: 0.4 }),
      rx('noteon', 'sh', 0, 'audio', 0.5, { atk: 0.005, rel: 0.25 }),
      rx('chordchange', 'pal', 0, 'hue', 0.15, { atk: 0.02, rel: 1.5 }),
    ],
  },
];

// X30.. art direction from a photo of marbled clay / kinetic sand: thick layered ribbons of neon
// colour side by side, embossed with a sandy relief, flowing along one form into a spiral vortex.
const ART4: Def[] = [
  {
    // Marbled clay: four dye sources, one per instrument, pour thick strands of colour into a slow,
    // heavy fluid that folds them over and round each other; the whole mass is lit as glossy
    // embossed clay with a faint sandy grain, each strand keeping its own colour beside the next.
    // Drum hits shove a fresh strand in and brighten the clay, the bass stirs the mass and puts a
    // sheen on it, the melody steers where the vocal strand enters and turns the palette with its
    // pitch, the riff swings the light across the relief the same way each repeat, and each section
    // type moves the palette to another family.
    origin: 'X30', name: 'Marbled Clay', energy: [0.2, 0.9], scheme: 'triad', hue: 0.55,
    color: { sat: 1, exposure: 0.95, contrast: 0.05, adapt: 0.1, bloom: 0.45, vignette: 0.4, relief: 1, bump: 2.5, gloss: 0.6, light: 0.375 },
    carrier: 'fluid', car: { halfLife: 12, floor: 0.3, amount: 0.85, vort: 8, fnoise: 0.05, blur: 0.04, sharpen: 0.04, grain: 0.003 },
    bodies: [body({
      shape: ['dot', { r: 0.016 }],
      place: ['stations', { count: 4, inst: 1, xs: 0.9, jump: 0, wander: 0.05 }],
      material: ['glow', { gain: 0.4, width: 0.02 }],
      emit: ['dye', { force: 1.1 }],
      feel: ['flow', { atk: 0.01, rel: 0.25 }],
    })],
    reactions: [
      rx('hit', 'em', 0, 'force', 0.4, { atk: 0.005, rel: 0.3 }),
      rx('hit', 'col', 0, 'exposure', 0.6, { atk: 0.005, rel: 0.2 }),
      rx('bass', 'car', 0, 'amount', 0.35, { atk: 0.05, rel: 0.5 }),
      rx('bass', 'col', 0, 'gloss', 0.5, { atk: 0.03, rel: 0.4 }),
      rx('melody', 'pal', 0, 'hue', 1, { atk: 0.08, rel: 0.4 }),
      rx('hook', 'col', 0, 'light', 0.5, { atk: 0.01, rel: 0.5 }),
    ],
    accent: { section: 0.25, hue: 1 },
  },
  {
    // Clay vortex: one tight, hand-rolled spiral of glossy clay bands. The coil is laid as a thick
    // wobbling spiral and its trail is wound round and slowly pushed outward, so every band sits
    // beside the ones before it; each is coloured by the melody note it was laid on, so the vortex
    // keeps the tune as rings of colour, and the whole coil is lit as embossed clay. Drum hits
    // brighten the fresh band, the bass stirs the winding and twists the coil, each note start
    // re-coils the spiral, the riff pulls it in tight the same way every repeat, and drops unwind
    // it outward across the frame.
    origin: 'X31', name: 'Clay Vortex', energy: [0.15, 0.9], scheme: 'triad', hue: 0.1,
    color: { sat: 1, exposure: 1, contrast: 0.05, adapt: 0.1, bloom: 0.45, vignette: 0.45, relief: 1, bump: 2.5, gloss: 0.6, light: 0.375 },
    carrier: 'warp', car: { halfLife: 2, floor: 0.7, sharpen: 0.05, grain: 0.003 },
    chain: [op('swirl', { amt: 0.004, k: 2, cx: 0.05 }), op('zoom', { rate: 0.0018, cx: 0.05 }), op('noise', { amp: 0.0006, scale: 3, speed: 0.2 })],
    bodies: [{
      ...body({
        shape: ['curve', { form: 2, radius: 0.45, turns: 3.5, amp: 0.04 }],
        place: ['point', { x: 0.05, y: 0 }],
        motion: ['hits', { amt: 0.35 }],
        material: ['glow', { gain: 0.2, width: 0.045 }],
        emit: ['trail'],
        feel: ['flow', { atk: 0.01, rel: 0.2 }],
        color: ['melody', { amount: 1, detail: 1 }],
      }),
      deform: { kind: 'none', p: {}, ops: [op('twist', { amt: 0.002 }), op('noise', { amp: 0.0015, scale: 2.5, speed: 0.25 })] },
    }],
    reactions: [
      rx('bass', 'op', 0, 'amt', 0.4, { atk: 0.05, rel: 0.5 }),
      rx('bass', 'dr', 0, 'amt', 0.5, { atk: 0.03, rel: 0.4 }),
      rx('hit', 'ma', 0, 'gain', 0.5, { atk: 0.005, rel: 0.25 }),
      rx('noteon', 'sh', 0, 'turns', 0.6, { atk: 0.01, rel: 0.4 }),
      rx('drop', 'op', 1, 'rate', 0.25, { atk: 0.3, rel: 2 }),
      rx('hook', 'sh', 0, 'radius', -0.4, { atk: 0.02, rel: 0.4 }),
    ],
    accent: { hook: 1 },
  },
];

// X36: Toronto at night from the islands, across the harbour.
const WL36 = -0.15; // the waterline (tone reflect) the city stands on
const TX36 = -0.25; // the CN Tower's x
function landmarks36(): BodyGene {
  // The CN Tower and the Rogers Centre as one compound standing on the waterline: a
  // slender shaft tapering into the antenna, the wide main pod about two thirds up, the small SkyPod
  // near the top, and left of the tower's foot the low dome with its lit rim band where it meets the base
  // (units of 0.3 scene).
  // A dark groove cut round the dome's base holds its lit rim band; a ring halo hugs the whole outline.
  return body({
    shape: ['compound', { size: 0.3 }],
    parts: [
      part('capsule', 'union', { y: 0.9, sx: 0.06, sy: 1.1, m: 0.97 }),
      part('ellipse', 'smooth', { y: 1.26, sx: 0.2, sy: 0.06, k: 0.04, bright: 1.3 }),
      part('ellipse', 'smooth', { y: 1.56, sx: 0.06, sy: 0.035, k: 0.02, bright: 1.2 }),
      part('ellipse', 'union', { x: -0.5, y: 0, sx: 0.36, sy: 0.18, bright: 0.6 }),
      part('box', 'subtract', { x: -0.5, y: 0.045, sx: 0.4, sy: 0.022, k: 0.004 }),
      part('box', 'union', { x: -0.5, y: 0.045, sx: 0.35, sy: 0.012, m: 1, hue: 0.33, bright: 2 }),
    ],
    place: ['point', { x: TX36, y: WL36 }],
    material: ['fill', { gain: 1.5, soft: 0.02, halo: 0.1, outline: 0, rings: 1, rgap: 0.006, rfade: 0.4 }],
    emit: ['none'],
    color: ['fixed', { hue: 0, detail: 1 }],
  });
}
const TORONTO_X36: Def[] = [
  {
    // Night Toronto seen from the islands: the CN Tower tall and slender left of the downtown core, its
    // wide main pod two thirds up and the SkyPod near the top, the Rogers Centre dome with its lit rim
    // at its foot, downtown a strip of lit windows, the whole city mirrored and rippling in the lake.
    // The melody runs up the tower like the N presets' notes: each note start sends a coloured mark up
    // the shaft (coloured and nudged sideways by its pitch), held notes draw a glowing ribbon. Drum
    // hits flash the landmarks' lights, the beat swells their glow, the skyline's towers rise and fall
    // with their spectrum bands and the windows glow with the bass. Nothing moves the frame itself.
    origin: 'X36', name: 'Harbour Night', energy: [0.2, 0.9], scheme: 'complementary', hue: 0.6,
    color: { sat: 0.75, adapt: 0.3, bloom: 1.2, vignette: 0.4, reflect: 1, reflectY: WL36 },
    carrier: 'warp', car: { halfLife: 0.35, floor: 0.6 },
    bodies: [
      landmarks36(),
      body({
        // The melody lane laid exactly on the shaft's axis (same x, turned upright, its foot at the
        // waterline where the tower starts; the present at the foot, the past climbing to the antenna
        // tip), with no pitch range across it, so every light rides inside the tower's body and pitch
        // shows only as colour: each note start is a light that climbs the tower, a held note a
        // continuous glowing band up the shaft.
        shape: ['notes', { mode: 0, span: 2, len: 0.6, height: 0, now: 0.5, tilt: 0, ribbon: 0.9, thick: 0.005, marks: 1, form: 0, size: 0.009, fade: 1.2, shimmer: 0.4, hues: 0.9, glow: 0.25 }],
        place: ['point', { x: TX36, y: WL36 + 0.3, angle: -0.25 }],
        material: ['glow', { gain: 1.5 }],
        emit: ['none'],
        feel: ['flow', { atk: 0.01, rel: 0.15 }],
      }),
      body({
        // Downtown east of the tower: mirrored spectrum bars (the bass towers in the middle of the core,
        // the treble ones low at its edges; the mirrored half is under the water) stippled with windows.
        shape: ['bars', { mode: 3, bins: 20, len: 0.3, fill: 0.45 }],
        place: ['point', { x: 0.5, y: WL36 }],
        material: ['dots', { gain: 1, spacing: 0.0095, size: 0.42 }],
        emit: ['none'],
        color: ['melody', { hue: 0.33, amount: 0.3, detail: 0.3 }],
      }),
    ],
    reactions: [
      rx('hit', 'ma', 0, 'gain', 0.2, { atk: 0.005, rel: 0.25 }),
      rx('beat', 'ma', 0, 'halo', 0.3, { rel: 0.3 }),
      rx('noteon', 'ma', 1, 'gain', 0.3, { atk: 0.005, rel: 0.2 }),
      rx('legato', 'sh', 1, 'thick', 0.4, { atk: 0.4, rel: 0.8 }),
      rx('bass', 'ma', 2, 'gain', 0.6, { atk: 0.03, rel: 0.3 }),
    ],
    // Only the section light and hue steps: no hit kick, riff camera nudge or drop punch moves the frame.
    accent: { hue: 0.9, kick: 0, hook: 0, drop: 0, frame: 0 },
  },
];

// ------------------------------------------------------------ art direction, after Van Gogh's The Starry Night
// X33..X35: the painting's sky of short impasto dashes swirling in vortices, haloed stars, the dark cypress
// and the village, in literal blues and yellows (palette key 0: the same colours in every song key).
const ART5: Def[] = [
  {
    // The painting at a glance, framed like it (height fitted, the sky extended sideways): fine, soft
    // blue brush strokes streaming round one great spiral in the upper middle and a counter-swirl
    // below it, the paint kept so no black gaps open; a crescent moon top right and eight stars at the
    // painting's positions, each in concentric halo rings; one silhouette of the near-black flame
    // cypress rising from the bottom left almost to the top, rolling hills climbing right, a pale
    // steeple and church, and lit house windows along the village. Drum hits stir the great spiral
    // (eased, not jolted) and pulse every halo outward, the bass turns the counter-swirl, each melody
    // note lightens the sky.
    origin: 'X33', name: 'Starry Night', energy: [0.15, 0.85], scheme: 'free', hue: 0.63, pal: { key: 0, s1: 0.51, s2: 0.95 },
    color: { sat: 0.72, exposure: 1, adapt: 0.35, bloom: 0.6, vignette: 0.2, relief: 0, bump: 1.4, light: 0.375, gloss: 0.35 },
    carrier: 'flow', car: { halfLife: 4, floor: 0.02, famt: 0.0005, fscale: 1.6, blur: 0.012 },
    chain: [
      op('swirl', { amt: 0.011, k: 6, cx: 0.04, cy: 0.17 }),
      op('swirl', { amt: -0.008, k: 7, cx: 0.21, cy: 0.02 }),
    ],
    bodies: [
      body({
        shape: ['dot', { r: 0.0014 }],
        place: ['grid', { lattice: 2, scale: 14, jitter: 0.6, density: 1, lit: 1, twinkle: 0.4 }],
        material: ['glow', { gain: 0.09, width: 0.0015, base: 0.6 }],
        emit: ['trail'],
        feel: ['flow', { atk: 0.02, rel: 0.3 }],
        color: ['height', { hue: 0.9, amount: 0.35, detail: 1 }],
      }),
      body({
        shape: ['compound', { size: 0.5 }],
        parts: [
          part('ellipse', 'union', { x: 1.03, y: 0.65, sx: 0.15, sy: 0.15, bright: 2, hue: 0.36 }),
          part('ellipse', 'subtract', { x: 0.97, y: 0.7, sx: 0.135, sy: 0.135, k: 0.01 }),
          part('ellipse', 'union', { x: -0.99, y: 0.91, sx: 0.035, sy: 0.035, bright: 2, hue: 0.333 }),
          part('ellipse', 'union', { x: -0.68, y: 0.93, sx: 0.025, sy: 0.025, bright: 2, hue: 0.333 }),
          part('ellipse', 'union', { x: -0.39, y: 0.92, sx: 0.03, sy: 0.03, bright: 2, hue: 0.333 }),
          part('ellipse', 'union', { x: -0.67, y: 0.65, sx: 0.04, sy: 0.04, bright: 2, hue: 0.333 }),
          part('ellipse', 'union', { x: -0.43, y: 0.35, sx: 0.03, sy: 0.03, bright: 2, hue: 0.333 }),
          part('ellipse', 'union', { x: -0.93, y: 0.05, sx: 0.035, sy: 0.035, bright: 2, hue: 0.333 }),
          part('ellipse', 'union', { x: -0.37, y: -0.06, sx: 0.055, sy: 0.055, bright: 2, hue: 0.333 }),
          part('ellipse', 'union', { x: 0.51, y: 0.54, sx: 0.04, sy: 0.04, bright: 2, hue: 0.333 }),
        ],
        place: ['point', { x: 0, y: 0 }],
        material: ['fill', { gain: 2, soft: 0.02, halo: 0.2, rings: 3, rgap: 0.011, rfade: 0.65 }],
        emit: ['none'],
        feel: ['flow', { atk: 0.01, rel: 0.25 }],
        color: ['fixed', { hue: 0, detail: 1 }],
      }),
      body({
        shape: ['compound', { size: 0.5 }],
        parts: [
          part('triangle', 'union', { x: -0.6, y: -0.05, sx: 0.32, sy: 0.95, m: -0.5, bright: 0, hue: 0.5 }),
          part('triangle', 'smooth', { x: -0.3, y: -0.56, sx: 0.16, sy: 0.44, m: 0.3, k: 0.04, bright: 0, hue: 0.5 }),
          part('ellipse', 'union', { x: 1, y: -0.82, sx: 1.4, sy: 0.24, rot: 0.04, bright: 0.26 }),
          part('box', 'union', { x: 0.8, y: -0.88, sx: 1, sy: 0.14, m: 0.5, bright: 0.16 }),
          part('triangle', 'union', { x: 0.16, y: -0.5, sx: 0.02, sy: 0.26, bright: 0.6 }),
          part('box', 'union', { x: 0.16, y: -0.8, sx: 0.07, sy: 0.06, bright: 0.6 }),
          part('ellipse', 'union', { x: 0.35, y: -0.72, sx: 0.6, sy: 0.16, bright: 0.24 }),
          part('box', 'union', { x: 0.3, y: -0.57, sx: 0.028, sy: 0.022, bright: 2, hue: 0.36 }),
          part('box', 'union', { x: 0.5, y: -0.575, sx: 0.028, sy: 0.022, bright: 2, hue: 0.36 }),
          part('box', 'union', { x: 0.72, y: -0.6, sx: 0.028, sy: 0.022, bright: 2, hue: 0.36 }),
        ],
        place: ['point', { x: 0, y: 0 }],
        material: ['fill', { gain: 1.6, soft: 0, halo: 0.1 }],
        emit: ['cover', { amt: 1, tip: 0 }],
        feel: ['flow', { atk: 0.05, rel: 0.4 }],
        color: ['fixed', { hue: 0, detail: 1 }],
      }),
    ],
    accent: { hue: 0, hook: 0, kick: 0 },
    reactions: [
      rx('hit', 'op', 0, 'amt', 0.35, { atk: 0.2, rel: 0.6 }),
      rx('bass', 'op', 1, 'amt', -0.5, { atk: 0.03, rel: 0.5 }),
      rx('noteon', 'ma', 0, 'gain', 0.4, { atk: 0.005, rel: 0.25 }),
      rx('hit', 'ma', 1, 'rgap', 0.4, { atk: 0.03, rel: 0.4 }),
    ],
  },
  {
    // Close up on the painting's central double swirl as pure brushwork: interlocking vortices of paint
    // strokes in cerulean, ultramarine and white with flecks of yellow, the paint kept on the canvas (no
    // black) and embossed like impasto. The riff turns the left vortex the same way on every repeat, drum
    // hits jolt the right one and flare fresh yellow strokes, the bass tightens the left vortex's core,
    // held notes lengthen the strokes (the flow runs faster), each melody note lightens the paint; the
    // built-in drop punch flares the whole sky.
    origin: 'X34', name: 'Brushstroke Vortex', energy: [0.2, 0.9], scheme: 'free', hue: 0.58, pal: { key: 0, s1: 0.56, s2: 0.03 },
    color: { sat: 0.95, exposure: 0.85, contrast: 0.07, adapt: 0.35, bloom: 1.05, vignette: 0.3, relief: 1, bump: 2.6, light: 0.375, gloss: 0.4 },
    carrier: 'flow', car: { halfLife: 2, floor: 0.15, famt: 0.0008, fscale: 1.6, blur: 0 },
    chain: [
      op('swirl', { amt: 0.018, k: 4, cx: -0.3, cy: 0.05 }),
      op('swirl', { amt: -0.012, k: 4, cx: 0.3, cy: -0.06 }),
    ],
    bodies: [
      body({
        shape: ['dot', { r: 0.004 }],
        place: ['grid', { lattice: 1, scale: 24, jitter: 0.6, density: 0.9, lit: 1, twinkle: 0.3 }],
        material: ['glow', { gain: 0.1, width: 0.008, base: 0.5 }],
        emit: ['trail'],
        feel: ['flow', { atk: 0.02, rel: 0.3 }],
        color: ['height', { hue: 0.95, amount: 0.4, detail: 1 }],
      }),
      body({
        shape: ['dot', { r: 0.008 }],
        place: ['grid', { lattice: 2, scale: 9, jitter: 0.6, density: 0.3, lit: 0.5, twinkle: 0.6 }],
        material: ['glow', { gain: 0.3, width: 0.008, base: 0.4 }],
        emit: ['trail'],
        feel: ['flow', { atk: 0.01, rel: 0.3 }],
        color: ['fixed', { hue: 0.333, detail: 0.5 }],
      }),
    ],
    accent: { hue: 0.25 },
    reactions: [
      rx('hook', 'op', 0, 'amt', 0.45, { atk: 0.005, rel: 0.5 }),
      rx('hit', 'op', 1, 'amt', -0.6, { atk: 0.005, rel: 0.3 }),
      rx('hit', 'ma', 1, 'gain', 0.9, { atk: 0.005, rel: 0.3 }),
      rx('bass', 'op', 0, 'k', -0.3, { atk: 0.03, rel: 0.5 }),
      rx('held', 'car', 0, 'famt', 0.6, { atk: 0.1, rel: 0.6 }),
      rx('noteon', 'ma', 0, 'gain', 0.6, { atk: 0.005, rel: 0.25 }),
    ],
  },
  {
    // A village under the stars: rolling blue hills, a steeple and houses with lit windows along the
    // bottom, a bright moon at the heart of one great vortex of fine blue brush strokes that sweeps the
    // whole sky round it, and the melody written into the sky as stars: each note stamps a new one,
    // ringed in a halo that spreads and fades. Drum hits stir the vortex (eased) and brighten the
    // strokes, the bass swells the hills and the moon, chord changes lift the night's light, and each
    // section shifts the night's colour.
    origin: 'X35', name: 'Village Under Stars', energy: [0.1, 0.85], scheme: 'free', hue: 0.63, pal: { key: 0, s1: 0.51, s2: 0.95 },
    color: { sat: 0.75, exposure: 1, adapt: 0.35, bloom: 0.8, vignette: 0.25 },
    carrier: 'warp', car: { halfLife: 3, floor: 0.03, blur: 0.012 },
    chain: [
      op('swirl', { amt: -0.016, k: 4, cx: 0.3, cy: 0.28 }),
    ],
    bodies: [
      body({
        shape: ['dot', { r: 0.0014 }],
        place: ['grid', { lattice: 2, scale: 14, jitter: 0.6, density: 1, lit: 1, twinkle: 0.4 }],
        material: ['glow', { gain: 0.09, width: 0.0015, base: 0.6 }],
        emit: ['trail'],
        feel: ['flow', { atk: 0.02, rel: 0.3 }],
        color: ['height', { hue: 0.9, amount: 0.35, detail: 1 }],
      }),
      body({
        shape: ['notes', { mode: 0, span: 8, len: 1.6, height: 0.34, now: -0.1, tilt: 0, ribbon: 0, thick: 0.004, marks: 1, form: 3, size: 0.026, fade: 1.5, rise: 0.05, shimmer: 0.3, hues: 0, glow: 0.6 }],
        place: ['point', { x: -0.1, y: 0.2 }],
        material: ['glow', { gain: 1.8 }],
        emit: ['none'],
        feel: ['flow', { atk: 0.01, rel: 0.2 }],
        color: ['fixed', { hue: 0.333, detail: 0 }],
      }),
      body({
        shape: ['compound', { size: 0.5 }],
        parts: [
          part('ellipse', 'union', { x: 0, y: 0, sx: 0.12, sy: 0.12, bright: 2, hue: 0.36 }),
          part('ellipse', 'union', { x: -1.5, y: -1.72, sx: 1.3, sy: 0.55, bright: 0.2 }),
          part('ellipse', 'smooth', { x: 0.4, y: -1.6, sx: 1.3, sy: 0.55, k: 0.12, bright: 0.26 }),
          part('box', 'union', { x: -0.62, y: -1.12, sx: 0.12, sy: 0.07, m: 0.2, bright: 0.3 }),
          part('box', 'union', { x: -0.22, y: -1.1, sx: 0.1, sy: 0.08, m: 0.2, bright: 0.3 }),
          part('triangle', 'union', { x: -0.42, y: -0.92, sx: 0.03, sy: 0.3, bright: 0.35 }),
          part('box', 'union', { x: -0.62, y: -1.06, sx: 0.028, sy: 0.022, bright: 2, hue: 0.36 }),
          part('box', 'union', { x: -0.22, y: -1.03, sx: 0.028, sy: 0.022, bright: 2, hue: 0.36 }),
          part('box', 'union', { x: 0.2, y: -1.08, sx: 0.028, sy: 0.022, bright: 2, hue: 0.36 }),
        ],
        place: ['point', { x: 0.3, y: 0.28 }],
        material: ['fill', { gain: 1.6, soft: 0, halo: 0.2 }],
        emit: ['cover', { amt: 1, tip: 0 }],
        feel: ['flow', { atk: 0.05, rel: 0.4 }],
        color: ['fixed', { hue: 0, detail: 1 }],
      }),
    ],
    accent: { hook: 0, kick: 0, drop: 0 },
    reactions: [
      rx('bass', 'sh', 2, 'size', 0.45, { atk: 0.08, rel: 0.6 }),
      rx('hit', 'op', 0, 'amt', -0.35, { atk: 0.2, rel: 0.6 }),
      rx('hit', 'ma', 0, 'gain', 0.5, { atk: 0.08, rel: 0.4 }),
      rx('chordchange', 'col', 0, 'exposure', 0.4, { atk: 0.05, rel: 1 }),
    ],
  },
];

// ------------------------------------------------------------ art direction, neon lilies
// X39..X41: glowing rainbow lilies on black. A bloom is a compound of six long pointed petals (tapered
// capsules turned round the centre, each a little different in length, width, angle and palette hue),
// curled toward the tips by a twist deform, lit as a soft fill with two neon ring outlines. Inside it
// a small copy of the same bloom is drawn into the feedback and carried outward by a slow zoom (its
// twist scaled so the copy lands on the petals), its hue drifting with age, so bands of iridescent
// colour flow through the petals from base to tip. Palettes are absolute (key 0) magenta, teal and
// gold so the petals stay vivid in every key; the section accent turns the colour family.
/** Six lily petals: [turn, length, width, hue] each, as tapered capsules pointing out from the centre. */
const LILY_PETALS: [number, number, number, number][] = [
  [0.01, 1.0, 0.36, 0], [0.175, 0.9, 0.31, 0.3], [0.33, 1.06, 0.37, 0.62], [0.505, 0.93, 0.32, 0.15], [0.655, 1.02, 0.35, 0.47], [0.835, 0.88, 0.3, 0.8],
];
/** Petal hues sitting exactly on the three palette slots (no in-between mixes, which go pastel). */
const LILY_SLOTS = [0, 1 / 3, 2 / 3, 0, 1 / 3, 2 / 3];
function lilyParts(open = 0.92, k = 0.1, hues?: number[]): CompoundPart[] {
  return LILY_PETALS.map(([a, len, w, hue], i) => part('capsule', i ? 'smooth' : 'union', {
    x: -Math.sin(a * 2 * Math.PI) * len * open, y: Math.cos(a * 2 * Math.PI) * len * open,
    sx: w * 1.35, sy: len, m: 1, rot: a > 0.5 ? a - 1 : a, k, hue: hues ? hues[i] : hue,
  }));
}
/** Three stamens with glowing anthers: thin filaments and a bright bead at each tip. */
function stamenParts(tip = 2): CompoundPart[] {
  const st: [number, number][] = [[0.07, 1.0], [0.4, 0.85], [0.74, 1.1]];
  return [
    ...st.map(([a, len]) => part('capsule', 'union', { x: -Math.sin(a * 2 * Math.PI) * len, y: Math.cos(a * 2 * Math.PI) * len, sx: 0.05, sy: len, m: 0.5, rot: a > 0.5 ? a - 1 : a, bright: 0.7 })),
    ...st.map(([a, len]) => part('ellipse', 'union', { x: -Math.sin(a * 2 * Math.PI) * len * tip, y: Math.cos(a * 2 * Math.PI) * len * tip, sx: 0.12, sy: 0.08, rot: a > 0.5 ? a - 1 : a, bright: 2, hue: 0 })),
  ];
}
const LILIES: Def[] = [
  {
    // One hero lily filling the middle of the frame, six iridescent petals each its own neon hue, three
    // glowing stamens over the throat. Drum hits swell the petals' glow, the bass breathes the bloom's
    // size, the voice opens the petals (less curl while singing, curling closed in the gaps), each
    // melody note start sends a brighter wave of colour up the petals and legato passages speed the
    // flow, the melody's pitch tints the petals, and each section turns the palette.
    origin: 'X39', name: 'Neon Lily', energy: [0.1, 0.9], scheme: 'free', hue: 0.88, pal: { key: 0, s1: 0.62, s2: 0.24 },
    color: { sat: 1, exposure: 0.8, adapt: 0.2, bloom: 1.1, vignette: 0.35 }, carrier: 'warp', car: { halfLife: 0.5, floor: 1.4, blur: 0 },
    chain: [op('zoom', { rate: 0.008 })],
    bodies: [
      body({
        shape: ['compound', { size: 0.09 }],
        parts: lilyParts(),
        place: ['point', { x: 0, y: 0 }],
        deform: ['twist', { amt: 2.6 }],
        material: ['line', { gain: 0.2, width: 3.5, halo: 0.3 }],
        emit: ['trail'],
        feel: ['flow', { atk: 0.02, rel: 0.4 }],
        color: ['age', { hue: 0.3, rate: 0.5, detail: 0.5 }],
      }),
      body({
        shape: ['compound', { size: 0.235 }],
        parts: lilyParts(),
        place: ['point', { x: 0, y: 0 }],
        deform: ['twist', { amt: 1 }],
        material: ['fill', { gain: 0.45, soft: 0.06, core: 0, rings: 2, rgap: 0.009, rfade: 0.4 }],
        emit: ['none'],
        feel: ['flow', { atk: 0.02, rel: 0.4 }],
        color: ['melody', { hue: 0, amount: 1, detail: 1 }],
      }),
      body({
        shape: ['compound', { size: 0.13 }],
        parts: stamenParts(),
        place: ['point', { x: 0, y: 0 }],
        deform: ['twist', { amt: 0.6 }],
        material: ['line', { gain: 1.4, width: 1.6, halo: 0.25 }],
        emit: ['none'],
        feel: ['flow', { atk: 0.01, rel: 0.25 }],
        color: ['fixed', { hue: 0.55, detail: 1 }],
      }),
    ],
    accent: { kick: 0, hook: 0, hue: 1, drop: 0.3 },
    reactions: [
      rx('hit', 'ma', 1, 'gain', 0.6, { atk: 0.02, rel: 0.3 }),
      rx('bass', 'sh', 1, 'size', 0.35, { atk: 0.05, rel: 0.45 }),
      rx('vocals', 'de', 1, 'amt', -0.05, { atk: 0.4, rel: 1.2 }),
      rx('vocals', 'de', 0, 'amt', -0.13, { atk: 0.4, rel: 1.2 }),
      rx('legato', 'op', 0, 'rate', 0.15, { atk: 0.3, rel: 1 }),
      rx('noteon', 'ma', 0, 'gain', 0.2, { atk: 0.005, rel: 0.3 }),
    ],
  },
  {
    // A close-up inside one lily: the bloom sits low left and its petals run off the frame, each on one
    // of the three palette colours with a double neon edge, three stamens arching over the throat. A
    // small copy of the bloom is drawn into a slow outward zoom centred on the throat, so bands of
    // colour flow from the base out along every petal to the tips. Legato passages brighten and speed
    // that flow, drum hits flash the petals and swell them, the bass breathes the bloom, each melody
    // note start flicks the stamens, the melody's pitch tints the petals and sections turn the palette.
    origin: 'X41', name: 'Lily Close-up', energy: [0.1, 0.9], scheme: 'free', hue: 0.88, pal: { key: 0, s1: 0.62, s2: 0.24 },
    color: { sat: 1, exposure: 0.8, adapt: 0.2, bloom: 1.1, vignette: 0.3 }, carrier: 'warp', car: { halfLife: 0.4, floor: 2.2, blur: 0 },
    chain: [op('zoom', { rate: 0.01, cx: -0.3, cy: -0.22 })],
    bodies: [
      body({
        shape: ['compound', { size: 0.15 }],
        parts: lilyParts(0.92, 0.1, LILY_SLOTS),
        place: ['point', { x: -0.3, y: -0.22, angle: -0.04 }],
        deform: ['twist', { amt: 1.26 }],
        material: ['line', { gain: 0.3, width: 4, halo: 0.25 }],
        emit: ['trail'],
        feel: ['flow', { atk: 0.02, rel: 0.4 }],
        color: ['age', { hue: 0.3, rate: 0.5, detail: 0.5 }],
      }),
      body({
        shape: ['compound', { size: 0.42 }],
        parts: lilyParts(0.92, 0.1, LILY_SLOTS),
        place: ['point', { x: -0.3, y: -0.22, angle: -0.04 }],
        deform: ['twist', { amt: 0.45 }],
        material: ['fill', { gain: 0.8, soft: 0.04, core: 0, rings: 2, rgap: 0.01, rfade: 0.6 }],
        emit: ['none'],
        feel: ['flow', { atk: 0.02, rel: 0.4 }],
        color: ['melody', { hue: 0, amount: 0.35, detail: 1 }],
      }),
      body({
        shape: ['compound', { size: 0.22 }],
        parts: stamenParts(2.1),
        place: ['point', { x: -0.3, y: -0.22, angle: 0.08 }],
        deform: ['twist', { amt: 0.5 }],
        material: ['line', { gain: 1.4, width: 2.2, halo: 0.3 }],
        emit: ['none'],
        feel: ['flow', { atk: 0.01, rel: 0.25 }],
        color: ['fixed', { hue: 0.55, detail: 1 }],
      }),
    ],
    accent: { kick: 0, hook: 0, hue: 1, section: 0.15, drop: 0.3 },
    reactions: [
      rx('drums', 'ma', 1, 'gain', 0.9, { atk: 0.01, rel: 0.15 }),
      rx('bass', 'sh', 1, 'size', 0.08, { atk: 0.15, rel: 0.8 }),
      rx('legato', 'ma', 0, 'gain', 0.4, { atk: 0.1, rel: 0.6 }),
      rx('legato', 'op', 0, 'rate', 0.5, { atk: 0.3, rel: 1 }),
      rx('noteon', 'sh', 2, 'size', 0.15, { atk: 0.005, rel: 0.3 }),
      rx('hit', 'sh', 1, 'size', 0.12, { atk: 0.01, rel: 0.25 }),
    ],
  },
];

// X37: the CN Tower's LED light show as the visualizer.
const TORONTO_X37: Def[] = [
  {
    // A close, heroic view of the CN Tower alone against the night sky, built from LED dots (a compound
    // silhouette: tapered shaft rising from below the frame, the wide main pod two thirds up, SkyPod and
    // antenna reaching the top), an aurora glowing behind it. The tower's lights are the visualizer: each
    // drum hit flares the tower and lifts the bright LED band off the base, racing up the shaft to the
    // pod; the bass swells the LED dots; a held melody note brightens the band; each note start steps
    // the tower's colour; drops widen the band and settle; sections switch the colours (built-in
    // accents). All brightness changes stay on the tower; no whole-frame beat motion.
    origin: 'X37', name: 'Tower Light Show', energy: [0.2, 0.95], scheme: 'split', hue: 0.55,
    color: { adapt: 0.3, bloom: 1.0, vignette: 0.5, exposure: 0.85, sat: 1.2 },
    carrier: 'warp', car: { halfLife: 0.3, floor: 0 },
    bodies: [
      body({
        shape: ['aurora', { fall: 3, rays: 22, wav: 1 }],
        place: ['point', { x: 0, y: -0.45 }],
        material: ['fill', { gain: 0.8 }],
        emit: ['none'],
        color: ['fixed', { hue: 0 }],
      }),
      body({
        shape: ['compound', { size: 0.25 }],
        parts: [
          part('capsule', 'union', { y: -0.5, sx: 0.2, sy: 1.8, m: 0.75, bright: 0.3 }),
          part('ellipse', 'smooth', { y: 0.55, sx: 0.44, sy: 0.12, k: 0.05, bright: 1.6, hue: 0.33 }),
          part('box', 'union', { y: 0.41, sx: 0.26, sy: 0.06, m: 0.6, bright: 0.5 }),
          part('box', 'union', { y: 0.56, sx: 0.46, sy: 0.025, m: 1, bright: 2, hue: 0.5 }),
          part('ellipse', 'smooth', { y: 1.3, sx: 0.1, sy: 0.07, k: 0.03, bright: 1.4, hue: 0.33 }),
          part('box', 'union', { y: 1.68, sx: 0.016, sy: 0.4, bright: 0.8 }),
        ],
        place: ['point', { x: 0, y: -0.02 }],
        material: ['dots', { gain: 1.2, spacing: 0.0055, size: 0.7 }],
        emit: ['none'],
        color: ['fixed', { hue: 0.5, detail: 1 }],
      }),
      body({
        shape: ['segment', { len: 0.7, w: 0.012 }],
        place: ['point', { x: 0, y: -0.5, angle: 0.25 }],
        material: ['dots', { gain: 1.8, spacing: 0.0055, size: 0.75 }],
        emit: ['none'],
        color: ['fixed', { hue: 0.66 }],
      }),
    ],
    reactions: [
      rx('hit', 'pl', 2, 'y', 1, { atk: 0.03, rel: 0.5 }),
      rx('bass', 'ma', 1, 'size', 0.4, { atk: 0.05, rel: 0.4 }),
      rx('held', 'ma', 2, 'gain', 0.5, { atk: 0.1, rel: 0.5 }),
      rx('hit', 'ma', 1, 'gain', 0.9, { atk: 0.005, rel: 0.3 }),
      rx('noteon', 'cm', 1, 'hue', 0.25, { atk: 0.005, rel: 0.35 }),
      rx('drop', 'sh', 2, 'w', 0.4, { atk: 0.02, rel: 2.5 }),
    ],
    accent: { drop: 0.15, hook: 0.3 },
  },
];

// X38: a Toronto skyline preset built from existing genes.
const TORONTO_X38: Def[] = [
  {
    // Driving into Toronto at night along the Gardiner: a lit road runs dead straight across a dark
    // plain toward downtown, whose skyline stands along the horizon (a mirrored strip of towers that
    // rise and fall with the spectrum, tallest at the core), and the CN Tower looms in the middle of
    // it, nearly half the frame tall: tapered shaft, the main pod with its lit rim two thirds up, the
    // SkyPod and the antenna with a beacon. Through each build the tower grows as the car closes in;
    // the drop floods the horizon with light, the arrival downtown. The bass lights the lane lines,
    // drum hits flash the road's edges, every melody note lights the tower, the riff flares the
    // skyline's windows, the melody moves the city lights' colour. Colours are fixed night blue and
    // sodium amber, the same in every key.
    origin: 'X38', name: 'Gardiner Night Drive', energy: [0.3, 1], scheme: 'free', hue: 0.62, pal: { s1: 0.5, s2: 0.06, key: 0 },
    color: { adapt: 0.2, bloom: 1.1, vignette: 0.35, exposure: 0.9, contrast: 0.06, sat: 0.85 }, carrier: 'none',
    accent: { hue: 0 },
    bodies: [
      body({
        shape: ['landscape', { path: 0, ground: 0, mark: 3, res: 0.35, look: 24, height: 0.45, relief: 0, rough: 0.1, wind: 0, fog: 0.1, glow: 0.4, tint: 0, kick: 0, rim: 0.06 }],
        material: ['glow', { gain: 0.3 }],
        color: ['fixed', { hue: 0, detail: 0.1 }],
      }),
      body({
        shape: ['bars', { mode: 0, bins: 40, len: 0.26, fill: 0.9 }],
        place: ['point', { x: 0, y: 0.07 }],
        material: ['dots', { gain: 0.6, spacing: 0.014, size: 0.4 }],
        emit: ['none'],
        color: ['melody', { hue: 0.33, amount: 0.08, detail: 0.1 }],
      }),
      body({
        shape: ['compound', { size: 0.078 }],
        parts: [
          part('capsule', 'union', { y: -0.95, sx: 0.32, sy: 4.05, m: 0.65, bright: 0.9 }),
          part('ellipse', 'smooth', { y: 1.2, sx: 0.8, sy: 0.28, k: 0.1, bright: 1 }),
          part('box', 'union', { y: 0.88, sx: 0.5, sy: 0.12, m: 0.6, bright: 0.8 }),
          part('box', 'union', { y: 1.2, sx: 0.84, sy: 0.045, m: 1, bright: 2, hue: 0.33 }),
          part('ellipse', 'smooth', { y: 3.1, sx: 0.26, sy: 0.2, k: 0.06, bright: 1.3 }),
          part('box', 'union', { y: 4.2, sx: 0.06, sy: 0.8, bright: 1.6, hue: 0.33 }),
        ],
        place: ['point', { x: 0, y: 0.23 }],
        material: ['fill', { gain: 1.7, soft: 0.05, halo: 0.35, clip: 0.065 }],
        emit: ['none'],
        color: ['fixed', { hue: 0, detail: 1 }],
      }),
    ],
    reactions: [
      rx('build', 'sh', 2, 'size', 0.05, { atk: 0.3, rel: 1.5 }),
      rx('drop', 'sh', 0, 'fog', 0.5, { atk: 0.02, rel: 1.5 }),
      rx('hook', 'ma', 2, 'halo', 1, { atk: 0.01, rel: 0.3 }),
      rx('bass', 'sh', 0, 'glow', 1, { atk: 0.02, rel: 0.25 }),
      rx('hit', 'ma', 1, 'gain', 1, { atk: 0.005, rel: 0.18 }),
      rx('noteon', 'ma', 2, 'gain', 0.6, { atk: 0.01, rel: 0.2 }),
    ],
  },
];

// X25..X28 FLOWERS: after the owner's photos of pressed-flower pictures and glass paperweights; organic
// petals built from compound parts, moved only by the music (no clock sway, no camera moves).
// Pale paper ground (a fine warm cell field in the feedback, blurred flat) with ink drawn by
// subtraction: the visible colour is the complement of the body's palette colour.
function paper25(gain: number): BodyGene {
  // One rounded box bigger than the frame: an even sheet, a warm off-white between the palette's first two slots.
  return body({
    shape: ['compound', { size: 0.6 }],
    parts: [part('box', 'union', { sx: 1.7, sy: 1, m: 0.2, hue: 0.24 })],
    material: ['fill', { gain, soft: 0 }],
    emit: ['none'],
    color: ['fixed', { hue: 0, detail: 1 }],
  });
}
/** A pressed poppy: four broad overlapping petals (uneven sizes and turns) round a dark seed boss, and its stem. */
function poppy25(v: number, stem: number): CompoundPart[] {
  const j = (k: number) => (((Math.sin(v * 12.9898 + k * 78.233) * 43758.5453) % 1) + 1) % 1 - 0.5; // -0.5..0.5, fixed per poppy
  const petal = (a: number, k: number) => {
    const r = 0.62 + 0.12 * j(k);
    const ang = (a + 0.04 * j(k + 9)) * Math.PI * 2;
    return part('ellipse', k === 1 ? 'union' : 'smooth', {
      x: Math.cos(ang) * r, y: Math.sin(ang) * r, sx: 0.9 + 0.16 * j(k + 3), sy: 0.72 + 0.1 * j(k + 5), rot: a + 0.25 + 0.03 * j(k + 13), k: 0.14,
      hue: 0.01 * j(k + 7), bright: 1.25 + 0.7 * j(k + 11),
    });
  };
  return [
    petal(0.1, 1), petal(0.35, 2), petal(0.6, 3), petal(0.85, 4),
    part('capsule', 'union', { x: 0.1 * stem, y: -1.2 - stem, sx: 0.1, sy: 1 + stem, rot: 0.02 * stem, m: 0.4, hue: 2 / 3, bright: 2 }),
    part('ellipse', 'union', { sx: 0.46, sy: 0.42, hue: 1 / 6, bright: 2 }),
  ];
}
/** One stem of a spray (ring copy frame: +y points away from the gathered base) with a small flower at its tip. */
function spray26(v: number, petals: number): CompoundPart[] {
  const j = (k: number) => (((Math.sin(v * 12.9898 + k * 78.233) * 43758.5453) % 1) + 1) % 1 - 0.5;
  const out: CompoundPart[] = [part('capsule', 'union', { x: 0, y: 0.2, sx: 0.045, sy: 1.9, m: 0.4, hue: 0, bright: 0.75 })];
  for (let k = 0; k < petals; k++) {
    const a = (k / petals + 0.1 * j(k)) * Math.PI * 2;
    out.push(part('ellipse', k === 0 ? 'union' : 'smooth', {
      x: Math.cos(a) * 0.33, y: 2 + Math.sin(a) * 0.33, sx: 0.3 + 0.05 * j(k + 3), sy: 0.21 + 0.04 * j(k + 5), rot: k / petals + 0.02 * j(k + 13), k: 0.03,
      hue: 0.02 * j(k + 7), bright: 1 + 0.4 * j(k + 11),
    }));
  }
  if (out.length < 6) out.push(part('ellipse', 'union', { y: 2, sx: 0.14, sy: 0.13, hue: 0.33, bright: 1.6 }));
  return out;
}
/** A body with draw-space ops bending it (after its own deformation). */
function bent(b: BodyGene, ops: OpGene[]): BodyGene {
  b.deform.ops = ops;
  return b;
}
/** A flower head seen from above for the paperweight: n petals round a small boss, turned by `turn`. */
function bloom28(v: number, n: number, r: number, sx: number, sy: number, turn: number): CompoundPart[] {
  const j = (k: number) => (((Math.sin(v * 12.9898 + k * 78.233) * 43758.5453) % 1) + 1) % 1 - 0.5;
  const out: CompoundPart[] = [];
  for (let k = 0; k < n; k++) {
    const a = (k / n + turn + 0.03 * j(k)) * Math.PI * 2;
    out.push(part('ellipse', k === 0 ? 'union' : 'smooth', {
      x: Math.cos(a) * r, y: Math.sin(a) * r, sx: sx + 0.06 * j(k + 3), sy: sy + 0.04 * j(k + 5), rot: k / n + turn + 0.02 * j(k + 7), k: 0.06,
      hue: 0.04 * j(k + 9), bright: 1 + 0.3 * j(k + 11),
    }));
  }
  out.push(part('ellipse', 'union', { sx: r * 0.45, sy: r * 0.42, hue: 0.33, bright: 1.6 }));
  return out;
}
const FLOWERS: Def[] = [
  {
    // Pressed poppies on warm paper: three big orange-red flower heads, each five rounded, overlapping,
    // slightly uneven petals round a dark seed boss, the ink embossed a little into the paper. The bass
    // swells the petals, drum hits make them flutter and crinkle, the melody tilts the heads.
    origin: 'X25', name: 'Poppy Press', energy: [0.15, 0.9], scheme: 'free', hue: 0.555, pal: { key: 0, s1: 0.565, s2: 0.29 },
    color: { sat: 1, exposure: 0.9, adapt: 0.1, bloom: 0.4, vignette: 0.3, contrast: 0, relief: 0 },
    carrier: 'warp', car: { halfLife: 0.3, floor: 1 },
    bodies: [
      paper25(2),
      body({
        shape: ['compound', { size: 0.17 }],
        parts: poppy25(1, 1.2),
        place: ['point', { x: -0.08, y: 0.1, angle: 0.02 }],
        deform: ['twist', { amt: 0.6 }],
        material: ['fill', { gain: 2.4, soft: 0.012, core: 0.9, blend: 2 }],
        emit: ['none'],
        color: ['fixed', { hue: 0, detail: 1 }],
      }),
      body({
        shape: ['compound', { size: 0.125 }],
        parts: poppy25(2, 0.6),
        place: ['mirror', { axis: 0, x: 0.6, y: -0.1, angle: 0.06 }],
        deform: ['twist', { amt: -0.7 }],
        material: ['fill', { gain: 2.4, soft: 0.012, core: 0.9, blend: 2 }],
        emit: ['none'],
        color: ['fixed', { hue: 0, detail: 1 }],
      }),
    ],
    accent: { kick: 0, hook: 0, drop: 0, hue: 0, section: 0.3 },
    reactions: [
      rx('bass', 'sh', 1, 'size', 0.14, { atk: 0.03, rel: 0.4 }),
      rx('hook', 'sh', 2, 'size', 0.14, { atk: 0.01, rel: 0.35 }),
      rx('hit', 'de', 1, 'amt', 0.05, { atk: 0.005, rel: 0.3 }),
      rx('hit', 'de', 2, 'amt', 0.05, { atk: 0.005, rel: 0.3 }),
      rx('noteon', 'pl', 1, 'angle', 0.04, { atk: 0.04, rel: 0.4 }),
      rx('held', 'pl', 2, 'angle', -0.06, { atk: 0.15, rel: 0.6 }),
    ],
  },
  {
    // A pressed bouquet on paper: two interleaved sprays of stems fan up from one gathered base at the
    // foot of the frame, each ending in a small pastel flower (violet, pink, butter yellow), inked into
    // the sheet. Each flower belongs to an instrument and deepens when it plays (the drum flowers flush on
    // hits, the vocal ones while the voice sings); note starts open the flowers wider, the bass lengthens
    // the spray, and every section type turns the bouquet to a new colour family.
    origin: 'X26', name: 'Pressed Bouquet', energy: [0.15, 0.85], scheme: 'free', hue: 0.64, pal: { key: 0, s1: 0.78, s2: 0.48 },
    color: { sat: 0.75, exposure: 0.9, adapt: 0.1, bloom: 0.4, vignette: 0.3, contrast: 0 },
    carrier: 'warp', car: { halfLife: 0.3, floor: 1 },
    bodies: [
      body({
        shape: ['compound', { size: 0.6 }],
        parts: [part('box', 'union', { sx: 1.7, sy: 1, m: 0.2, hue: 0.765 })],
        material: ['fill', { gain: 2.4, soft: 0 }],
        emit: ['none'],
        color: ['fixed', { hue: 0, detail: 1 }],
      }),
      bent(body({
        shape: ['compound', { size: 0.2 }],
        parts: spray26(3, 5),
        place: ['ring', { n: 9, radius: 0.3, x: 0, y: -0.4, angle: 0 }],
        deform: ['twist', { amt: 0.12 }],
        material: ['fill', { gain: 1.4, soft: 0.02, core: 0.5, blend: 2 }],
        emit: ['none'],
        color: ['instrument', { hue: 0, amount: 0.75, detail: 1 }],
      }), [op('noise', { amp: 0.0016, scale: 1.6, speed: 0.05 })]),
      bent(body({
        shape: ['compound', { size: 0.15 }],
        parts: spray26(7, 4),
        place: ['ring', { n: 11, radius: 0.22, x: 0, y: -0.4, angle: 0.045 }],
        deform: ['twist', { amt: -0.15 }],
        material: ['fill', { gain: 1.4, soft: 0.02, core: 0.5, blend: 2 }],
        emit: ['none'],
        color: ['instrument', { hue: 0.2, amount: 0.6, detail: 1 }],
      }), [op('noise', { amp: 0.0016, scale: 1.9, speed: 0.05 })]),
    ],
    accent: { kick: 0, hook: 0, drop: 0, hue: 0, section: 0.3 },
    reactions: [
      rx('hit', 'sh', 1, 'size', 0.14, { atk: 0.005, rel: 0.25 }),
      rx('hit', 'ma', 1, 'gain', 0.5, { atk: 0.005, rel: 0.3 }),
      rx('noteon', 'ma', 2, 'gain', 0.5, { atk: 0.01, rel: 0.35 }),
      rx('bass', 'pl', 1, 'radius', 0.2, { atk: 0.03, rel: 0.35 }),
      rx('bass', 'pl', 2, 'radius', 0.2, { atk: 0.03, rel: 0.35 }),

    ],
  },
  {
    // A pressed flower sealed in a glass paperweight on a dark shelf: a clear dome of glass with a soft
    // highlight, resting on its own reflection, and inside it a flower of two petal rings (warm outer
    // petals, pale inner ones) that curl like drawn glass cane. The melody opens the flower: held notes
    // unfurl the petals and turn their spiral, note starts brighten them; the bass breathes the glass dome,
    // drum hits flare the inner ring; the glass takes the song's timbre (clear on pure tones).
    origin: 'X28', name: 'Glass Garden', energy: [0.15, 0.85], scheme: 'free', hue: 0.58, pal: { key: 0, s1: 0.45, s2: 0.56 },
    color: { sat: 1, exposure: 1, adapt: 0.3, bloom: 1.1, vignette: 0.5, contrast: 0.02, reflect: 1, reflectY: -0.36 },
    carrier: 'warp', car: { halfLife: 0.3, floor: 0.6, blur: 0.02 },
    bodies: [
      body({
        shape: ['compound', { size: 0.158 }],
        parts: bloom28(3, 5, 0.9, 0.72, 0.46, 0.05),
        place: ['point', { x: 0, y: -0.08 }],
        deform: ['twist', { amt: 0.35 }],
        material: ['fill', { gain: 1.3, soft: 0.06, core: 0.6, halo: 0 }],
        emit: ['none'],
        feel: ['flow', { atk: 0.02, rel: 0.3 }],
        color: ['fixed', { hue: 1 / 3, detail: 1 }],
      }),
      body({
        shape: ['compound', { size: 0.09 }],
        parts: bloom28(8, 5, 0.85, 0.65, 0.38, 0.15),
        place: ['point', { x: 0, y: -0.08 }],
        deform: ['twist', { amt: -0.6 }],
        material: ['fill', { gain: 1.4, soft: 0.06, core: 0.6, halo: 0 }],
        emit: ['none'],
        feel: ['flow', { atk: 0.02, rel: 0.3 }],
        color: ['fixed', { hue: 2 / 3, detail: 1 }],
      }),
      body({
        shape: ['dot', { r: 0.28 }],
        place: ['point', { x: 0, y: -0.08 }],
        deform: ['wobble', { lobes: 3, amp: 0.03, rate: 0 }],
        material: ['chrome', { gain: 0.4, chrome: 0.85 }],
        emit: ['none'],
        feel: ['flow', { atk: 0.03, rel: 0.4 }],
        color: ['fixed', { hue: 0, detail: 0 }],
      }),
    ],
    accent: { kick: 0, hook: 0, drop: 0, hue: 0.3, section: 0.4 },
    reactions: [
      rx('bass', 'sh', 2, 'r', 0.12, { atk: 0.04, rel: 0.45 }),
      rx('held', 'sh', 0, 'size', 0.08, { atk: 0.1, rel: 0.6 }),
      rx('legato', 'de', 0, 'amt', 0.2, { atk: 0.2, rel: 0.8 }),
      rx('noteon', 'ma', 0, 'gain', 0.5, { atk: 0.005, rel: 0.3 }),
      rx('hit', 'sh', 1, 'size', 0.1, { atk: 0.005, rel: 0.3 }),
      rx('hit', 'ma', 1, 'gain', 0.6, { atk: 0.005, rel: 0.3 }),
    ],
  },
];

// Four composed studies using only the existing gene vocabulary (X42..X45).
const STUDIES: Def[] = [
  {
    // A brass instrument suspended in darkness: engraved concentric dials, twelve rotating teeth,
    // and three enamel satellites. Bass opens the dial spacing; melody lights the satellites;
    // percussion lifts the teeth without moving the whole camera.
    origin: 'X42', name: 'Amber Orrery', energy: [0.1, 0.85], scheme: 'free', hue: 0.1,
    pal: { key: 0, s1: 0.43, s2: 0.94 },
    color: { exposure: 0.9, adapt: 0.2, bloom: 0.75, vignette: 0.55, sat: 0.8 },
    carrier: 'none',
    bodies: [
      body({
        shape: ['dot', { r: 0.085 }],
        material: ['line', { gain: 0.85, width: 1.6, halo: 0.08, rings: 5, rgap: 0.047, rfade: 0.86 }],
        emit: ['none'], color: ['fixed', { hue: 0, detail: 0 }],
      }),
      body({
        shape: ['compound', { size: 0.035 }],
        parts: [
          part('box', 'union', { sx: 0.28, sy: 0.9, m: 0.2 }),
          part('ellipse', 'union', { y: 1.4, sx: 0.35, sy: 0.35, hue: 1 / 3 }),
        ],
        place: ['ring', { n: 12, radius: 0.29 }], motion: ['spin', { rate: -0.0625 }],
        material: ['fill', { gain: 1.2, soft: 0.04, core: 0.5 }],
        emit: ['none'], color: ['fixed', { hue: 0, detail: 1 }],
      }),
      body({
        shape: ['dot', { r: 0.024 }],
        place: ['orbit', { count: 3, radius: 0.225, rate: 0.125 }],
        material: ['chrome', { gain: 1.2, chrome: 0.7 }],
        emit: ['none'], color: ['instrument', { hue: 1 / 3, amount: 0.25, detail: 0.25 }],
        feel: ['flow', { atk: 0.03, rel: 0.35 }],
      }),
    ],
    reactions: [
      rx('bass', 'ma', 0, 'rgap', 0.09, { atk: 0.08, rel: 0.5 }),
      rx('hit', 'ma', 1, 'gain', 0.45, { atk: 0.01, rel: 0.35 }),
      rx('noteon', 'sh', 2, 'r', 0.035, { atk: 0.02, rel: 0.4 }),
      rx('held', 'ma', 2, 'gain', 0.4, { atk: 0.12, rel: 0.5 }),
    ],
    accent: { hook: 0.15, kick: 0, drop: 0.1, hue: 0, section: 0.25 },
  },
  {
    // Two counter-rotating lattices of rounded diamond loops make moving interference patterns.
    // A sparse third layer of warm pinheads marks the crossings; bass bends one set of threads,
    // the sung line bends the other and percussion catches the pins.
    origin: 'X43', name: 'Prism Loom', energy: [0.2, 0.95], scheme: 'free', hue: 0.53,
    pal: { key: 0, s1: 0.35, s2: 0.56 },
    color: { exposure: 0.85, adapt: 0.2, bloom: 0.65, vignette: 0.4, sat: 0.85 },
    carrier: 'none',
    bodies: [
      body({
        shape: ['polygon', { n: 4, r: 0.075, round: 0.25 }],
        place: ['grid', { lattice: 0, scale: 4.5, jitter: 0, density: 1, lit: 1, twinkle: 0, angle: 0.12, lock: 0.0625 }],
        deform: ['twist', { amt: 0.7 }],
        material: ['line', { gain: 0.85, width: 1.3, halo: 0.04, rings: 2, rgap: 0.014, rfade: 0.8 }],
        emit: ['none'], color: ['fixed', { hue: 0, detail: 0 }],
      }),
      body({
        shape: ['polygon', { n: 4, r: 0.075, round: 0.25 }],
        place: ['grid', { lattice: 0, scale: 4.8, jitter: 0, density: 1, lit: 1, twinkle: 0, angle: -0.12, lock: -0.0625 }],
        deform: ['twist', { amt: -0.7 }],
        material: ['line', { gain: 0.75, width: 1.3, halo: 0.04, rings: 2, rgap: 0.014, rfade: 0.8 }],
        emit: ['none'], color: ['fixed', { hue: 1 / 3, detail: 0 }],
      }),
      body({
        shape: ['dot', { r: 0.003 }],
        place: ['grid', { lattice: 0, scale: 4.5, jitter: 0, density: 0.3, lit: 1, twinkle: 0, angle: 0.12, lock: 0.0625 }],
        material: ['glow', { gain: 0.45, width: 0.006, base: 0.25 }],
        emit: ['none'], color: ['fixed', { hue: 2 / 3, detail: 0 }],
      }),
    ],
    reactions: [
      rx('bass', 'de', 0, 'amt', 0.1, { atk: 0.1, rel: 0.5 }),
      rx('held', 'de', 1, 'amt', -0.12, { atk: 0.15, rel: 0.65 }),
      rx('hit', 'ma', 2, 'gain', 0.65, { atk: 0.01, rel: 0.25 }),
      rx('hook', 'ma', 1, 'gain', 0.25, { atk: 0.04, rel: 0.4 }),
    ],
    accent: { hook: 0, kick: 0, drop: 0.1, hue: 0, section: 0.25 },
  },
  {
    // A luminous Art Deco arcade above a black reflecting pool. Each arch is an ellipse joined
    // to a rectangle with its doorway cut out; offset ring halos extend the architectural ribs.
    // Instrument copies light independently, melody widens the ribs and bass swells the central sun.
    origin: 'X44', name: 'Echo Arcade', energy: [0.1, 0.85], scheme: 'free', hue: 0.57,
    pal: { key: 0, s1: 0.48, s2: 0.36 },
    color: { exposure: 0.85, adapt: 0.2, bloom: 0.85, vignette: 0.45, reflect: 1, reflectY: -0.27 },
    carrier: 'none',
    bodies: [
      body({
        shape: ['compound', { size: 0.11 }],
        parts: [
          part('ellipse', 'union', { y: 0.55, sx: 0.75, sy: 0.8 }),
          part('box', 'union', { y: -0.2, sx: 0.75, sy: 0.75 }),
          part('ellipse', 'subtract', { y: 0.55, sx: 0.6, sy: 0.65, k: 0 }),
          part('box', 'subtract', { y: -0.35, sx: 0.6, sy: 0.9, k: 0 }),
        ],
        place: ['row', { count: 4, y: -0.08, wander: 0 }],
        material: ['line', { gain: 0.7, width: 1.7, halo: 0.05, rings: 3, rgap: 0.014, rfade: 0.8 }],
        emit: ['none'], color: ['instrument', { amount: 0.75, detail: 0 }],
        feel: ['flow', { atk: 0.05, rel: 0.45 }],
      }),
      body({
        shape: ['dot', { r: 0.065 }], place: ['point', { y: 0.25 }],
        material: ['textured', { tex: 1, gain: 0.9, halo: 0.3 }],
        emit: ['none'], color: ['fixed', { hue: 1 / 3, detail: 0.25 }],
      }),
      body({
        shape: ['segment', { len: 0.8, w: 0.003 }], place: ['point', { y: -0.265 }],
        material: ['line', { gain: 0.5, width: 1.2, halo: 0.06 }],
        emit: ['none'], color: ['fixed', { hue: 0, detail: 0 }],
      }),
    ],
    reactions: [
      rx('held', 'ma', 0, 'rgap', 0.06, { atk: 0.2, rel: 0.7 }),
      rx('noteon', 'ma', 0, 'gain', 0.35, { atk: 0.02, rel: 0.35 }),
      rx('bass', 'sh', 1, 'r', 0.1, { atk: 0.08, rel: 0.45 }),
      rx('hit', 'ma', 2, 'gain', 0.45, { atk: 0.01, rel: 0.3 }),
    ],
    accent: { hook: 0, kick: 0, drop: 0.1, hue: 0, section: 0.3 },
  },
  {
    // Three enamel ribbons weave a figure-eight through slow sideways feedback. The fresh strokes
    // paint over the old ones instead of adding to white; the outer outlines stay like gold leaf.
    // Bass thickens the ribbons, vocal sustain curls them and note starts illuminate the edging.
    origin: 'X45', name: 'Lacquer Current', energy: [0.15, 0.9], scheme: 'free', hue: 0.97,
    pal: { key: 0, s1: 0.53, s2: 0.13 },
    color: { exposure: 0.85, adapt: 0.2, bloom: 0.6, vignette: 0.35, relief: 0.35, sat: 0.85 },
    carrier: 'warp', car: { halfLife: 3.5, floor: 0.5, blur: 0.03 },
    chain: [op('translate', { vx: 0.09 }), op('swirl', { amt: 0.0035, k: 3, cx: -0.2 })],
    bodies: [
      body({
        shape: ['segment', { len: 0.28, w: 0.023 }],
        place: ['outline', { count: 3, path: 2, radius: 0.3, rate: 0.125 }],
        motion: ['spin', { rate: -0.125 }], deform: ['twist', { amt: 3 }],
        material: ['fill', { gain: 1.1, soft: 0.12, core: 0.6 }],
        emit: ['cover', { amt: 1, tip: 0 }], color: ['instrument', { amount: 0.8, detail: 0.1 }],
        feel: ['flow', { atk: 0.04, rel: 0.4 }],
      }),
      body({
        shape: ['segment', { len: 0.28, w: 0.023 }],
        place: ['outline', { count: 3, path: 2, radius: 0.3, rate: 0.125 }],
        motion: ['spin', { rate: -0.125 }], deform: ['twist', { amt: 3 }],
        material: ['line', { gain: 0.4, width: 1.3, halo: 0.02 }],
        emit: ['trail'], color: ['fixed', { hue: 2 / 3, detail: 0 }],
        feel: ['flow', { atk: 0.04, rel: 0.4 }],
      }),
    ],
    reactions: [
      rx('bass', 'sh', 0, 'w', 0.16, { atk: 0.05, rel: 0.4 }),
      rx('bass', 'sh', 1, 'w', 0.16, { atk: 0.05, rel: 0.4 }),
      rx('held', 'de', 0, 'amt', 0.16, { atk: 0.15, rel: 0.6 }),
      rx('held', 'de', 1, 'amt', 0.16, { atk: 0.15, rel: 0.6 }),
      rx('noteon', 'ma', 1, 'gain', 0.4, { atk: 0.01, rel: 0.35 }),
    ],
    accent: { hook: 0.15, kick: 0, drop: 0.1, hue: 0, section: 0.3 },
  },
];

// New body families: starter parents whose structure survives crossover and mutation.
const BODY_STUDIES: Def[] = [
  {
    origin: 'X50', name: 'Clockwork Anemone', energy: [0.2, 0.95], scheme: 'free', hue: 0.09,
    pal: { key: 0, s1: 0.46, s2: 0.8 },
    color: { exposure: 0.95, bloom: 0.8, adapt: 0.15, vignette: 0.45 }, carrier: 'none',
    bodies: [body({
      shape: ['linkage', { size: 0.25, joints: 6, width: 0.045, curl: 0, flex: 0.3, taper: 0.95, knuckle: 0.8 }],
      place: ['ring', { n: 5, radius: 0.25 }],
      material: ['line', { gain: 1.1, width: 1.6, halo: 0.08 }], emit: ['none'],
      color: ['fixed', { hue: 0, detail: 1 }],
    })],
    reactions: [
      rx('bass', 'sh', 0, 'curl', 0.08, { atk: 0.08, rel: 0.45 }),
      rx('held', 'sh', 0, 'flex', 0.15, { atk: 0.18, rel: 0.65 }),
      rx('hit', 'sh', 0, 'knuckle', 0.25, { atk: 0.02, rel: 0.3 }),
      rx('noteon', 'ma', 0, 'gain', 0.3, { atk: 0.02, rel: 0.35 }),
    ],
    accent: { hook: 0, kick: 0, hue: 0, drop: 0.1, section: 0.25 },
  },
  {
    origin: 'X51', name: 'Marionette March', energy: [0.15, 0.9], scheme: 'free', hue: 0.55,
    pal: { key: 0, s1: 0.43, s2: 0.31 },
    color: { exposure: 1, bloom: 0.65, adapt: 0.2, vignette: 0.3, reflect: 1, reflectY: -0.26 }, carrier: 'none',
    bodies: [body({
      shape: ['linkage', { size: 0.23, joints: 5, width: 0.075, curl: -0.15, flex: 0.55, taper: 1, knuckle: 0.75 }],
      place: ['row', { count: 4, y: -0.015, wander: 0 }],
      material: ['fill', { gain: 1.5, soft: 0.015, core: 0.1, halo: 0.08 }], emit: ['none'],
      color: ['instrument', { amount: 0.85, detail: 0.65 }],
      feel: ['flow', { atk: 0.04, rel: 0.3, div: 8 }],
    })],
    reactions: [
      rx('bass', 'sh', 0, 'flex', 0.45, { atk: 0.08, rel: 0.45 }),
      rx('held', 'sh', 0, 'curl', 0.3, { atk: 0.2, rel: 0.7 }),
      rx('hit', 'ma', 0, 'gain', 0.4, { atk: 0.02, rel: 0.3 }),
    ],
    accent: { hook: 0, kick: 0, hue: 0, drop: 0.1, section: 0.25 },
  },

  {
    origin: 'X48', name: 'Iridescent Sail', energy: [0.05, 0.85], scheme: 'free', hue: 0.57,
    pal: { key: 0, s1: 0.3, s2: 0.94 },
    color: { exposure: 1, bloom: 0.45, adapt: 0.15, vignette: 0.35 }, carrier: 'none',
    bodies: [body({
      shape: ['fabric', { size: 0.3, aspect: 1.55, folds: 9, depth: 0.85, drape: 0.55, weave: 0.3, flutter: 0.4 }],
      place: ['point', { y: 0.1, angle: -0.06 }], motion: ['sway', { amp: 0.025, period: 8, tilt: 0.15 }],
      material: ['fill', { gain: 1.3, soft: 0.01 }], emit: ['none'],
      color: ['height', { hue: 0, amount: 0.65, detail: 1 }],
    })],
    reactions: [
      rx('bass', 'sh', 0, 'depth', 0.25, { atk: 0.1, rel: 0.55 }),
      rx('held', 'sh', 0, 'drape', 0.4, { atk: 0.2, rel: 0.7 }),
      rx('hit', 'sh', 0, 'flutter', 0.3, { atk: 0.04, rel: 0.45 }),
      rx('noteon', 'ma', 0, 'gain', 0.25, { atk: 0.03, rel: 0.4 }),
    ],
    accent: { hook: 0, kick: 0, hue: 0, drop: 0.1, section: 0.2 },
  },
  {
    origin: 'X49', name: 'Ribbon Assembly', energy: [0.2, 0.95], scheme: 'free', hue: 0.08,
    pal: { key: 0, s1: 0.46, s2: 0.79 },
    color: { exposure: 0.9, bloom: 0.6, adapt: 0.15, vignette: 0.4 }, carrier: 'none',
    bodies: [body({
      shape: ['fabric', { size: 0.19, aspect: 0.28, folds: 4, depth: 0.75, drape: 0.25, weave: 0.1, flutter: 0.65 }],
      place: ['ring', { n: 9, radius: 0.25 }], motion: ['spin', { rate: -0.0625 }],
      material: ['fill', { gain: 1.15, soft: 0.02, halo: 0.05 }], emit: ['none'],
      color: ['height', { hue: 0, amount: 1, detail: 0.75 }],
    })],
    reactions: [
      rx('bass', 'sh', 0, 'aspect', 0.12, { atk: 0.05, rel: 0.4 }),
      rx('held', 'sh', 0, 'flutter', 0.3, { atk: 0.15, rel: 0.6 }),
      rx('hit', 'ma', 0, 'gain', 0.35, { atk: 0.02, rel: 0.3 }),
    ],
    accent: { hook: 0, kick: 0, hue: 0, drop: 0.1, section: 0.25 },
  },

  {
    origin: 'X46', name: 'Coral Cathedral', energy: [0.1, 0.85], scheme: 'free', hue: 0.52,
    pal: { key: 0, s1: 0.38, s2: 0.55 },
    color: { exposure: 0.95, bloom: 0.85, adapt: 0.2, vignette: 0.4 }, carrier: 'none',
    bodies: [body({
      shape: ['branch', { size: 0.37, width: 0.06, levels: 6, spread: 0.72, ratio: 0.67, grow: 0.82, bend: 0.08 }],
      place: ['point', { y: -0.02 }], material: ['line', { gain: 1.25, width: 2, halo: 0.08 }],
      emit: ['none'], color: ['fixed', { hue: 0, detail: 1 }],
    })],
    reactions: [
      rx('build', 'sh', 0, 'grow', 0.4, { atk: 0.3, rel: 1.5 }),
      rx('held', 'sh', 0, 'spread', 0.25, { atk: 0.2, rel: 0.7 }),
      rx('bass', 'sh', 0, 'width', 0.2, { atk: 0.06, rel: 0.4 }),
      rx('noteon', 'ma', 0, 'gain', 0.4, { atk: 0.02, rel: 0.3 }),
    ],
    accent: { hook: 0, kick: 0, drop: 0.15, hue: 0, section: 0.3 },
  },
  {
    origin: 'X47', name: 'Copper Nervure', energy: [0.2, 0.95], scheme: 'free', hue: 0.075,
    pal: { key: 0, s1: 0.49, s2: 0.87 },
    color: { exposure: 0.9, bloom: 0.7, adapt: 0.15, vignette: 0.45 }, carrier: 'none',
    bodies: [body({
      shape: ['branch', { size: 0.12, width: 0.025, levels: 5, spread: 0.9, ratio: 0.63, grow: 0.88, bend: 0.15 }],
      place: ['ring', { n: 7, radius: 0.15 }], motion: ['spin', { rate: 0.0625 }],
      material: ['line', { gain: 1.1, width: 1.4, halo: 0.12 }], emit: ['none'],
      color: ['height', { amount: 0.5, detail: 0.6 }],
    })],
    reactions: [
      rx('bass', 'sh', 0, 'spread', -0.35, { atk: 0.08, rel: 0.45 }),
      rx('held', 'sh', 0, 'grow', 0.3, { atk: 0.2, rel: 0.6 }),
      rx('hit', 'ma', 0, 'gain', 0.35, { atk: 0.015, rel: 0.3 }),
    ],
    accent: { hook: 0, kick: 0, drop: 0.15, hue: 0, section: 0.3 },
  },
];

// Musical gestures composed from existing bodies, with pitch direction kept distinct.
const MUSIC_STUDIES: Def[] = [
  {
    origin: 'X52', name: 'Songbird Mobile', energy: [0.05, 0.85], scheme: 'free', hue: 0.08,
    pal: { key: 0, s1: 0.48, s2: 0.83 },
    color: { exposure: 1, bloom: 0.55, adapt: 0.12, vignette: 0.25 }, carrier: 'none',
    bodies: [
      body({
        shape: ['linkage', { size: 0.34, joints: 7, width: 0.065, curl: -0.24, flex: 0.08, taper: 0.91, knuckle: 0.95 }],
        place: ['point', { x: -0.26, y: -0.2, angle: -0.16 }],
        material: ['line', { gain: 1.3, width: 2, halo: 0.05 }], emit: ['none'],
        color: ['fixed', { hue: 0, detail: 0.5 }],
      }),
      body({
        shape: ['linkage', { size: 0.29, joints: 6, width: 0.1, curl: 0.3, flex: 0.08, taper: 0.93, knuckle: 0.75 }],
        place: ['point', { x: 0.28, y: -0.05, angle: 0.34 }],
        material: ['fill', { gain: 1.4, soft: 0.02, halo: 0.03 }], emit: ['none'],
        color: ['fixed', { hue: 0.48, detail: 0.8 }],
      }),
    ],
    reactions: [
      rx('register', 'pl', 0, 'y', 0.85, { atk: 0.15, rel: 0.25 }),
      rx('register', 'pl', 1, 'y', -0.65, { atk: 0.15, rel: 0.25 }),
      rx('rising', 'sh', 0, 'flex', 1, { atk: 0.06, rel: 0.4 }),
      rx('falling', 'sh', 1, 'flex', 1, { atk: 0.06, rel: 0.4 }),
      rx('hit', 'sh', 0, 'knuckle', -0.6, { atk: 0.02, rel: 0.25 }),
      rx('noteon', 'ma', 1, 'gain', 0.25, { atk: 0.02, rel: 0.25 }),
    ],
    accent: { hook: 0, kick: 0, hue: 0, drop: 0.1, section: 0.2 },
  },
  {
    origin: 'X53', name: 'Portamento Loom', energy: [0.05, 0.9], scheme: 'free', hue: 0.57,
    pal: { key: 0, s1: 0.38, s2: 0.91 },
    color: { exposure: 0.95, bloom: 0.35, adapt: 0.12, vignette: 0.3 }, carrier: 'none',
    bodies: [
      body({
        shape: ['fabric', { size: 0.29, aspect: 0.72, folds: 7, depth: 0.8, drape: 0.2, weave: 0.65, flutter: 0.08 }],
        place: ['point', { x: -0.25, y: 0.08, angle: -0.12 }],
        material: ['fill', { gain: 1.4, soft: 0.008 }], emit: ['none'],
        color: ['height', { hue: 0, amount: 0.55, detail: 1 }],
      }),
      body({
        shape: ['fabric', { size: 0.24, aspect: 0.62, folds: 5, depth: 0.85, drape: 0.2, weave: 0.2, flutter: 0.08 }],
        place: ['point', { x: 0.29, y: -0.08, angle: 0.15 }],
        material: ['line', { gain: 1.5, width: 1.8, halo: 0.08 }], emit: ['none'],
        color: ['height', { hue: 0.4, amount: 0.6, detail: 1 }],
      }),
    ],
    reactions: [
      rx('rising', 'sh', 0, 'drape', 1, { atk: 0.08, rel: 0.4 }),
      rx('falling', 'sh', 1, 'drape', 1, { atk: 0.08, rel: 0.4 }),
      rx('register', 'sh', 0, 'folds', 0.8, { atk: 0.15, rel: 0.3 }),
      rx('vibrato', 'sh', 1, 'flutter', 0.9, { atk: 0.1, rel: 0.4 }),
      rx('held', 'sh', 0, 'aspect', 0.35, { atk: 0.2, rel: 0.5 }),
      rx('noteon', 'ma', 1, 'gain', 0.3, { atk: 0.02, rel: 0.3 }),
    ],
    accent: { hook: 0, kick: 0, hue: 0, drop: 0.1, section: 0.2 },
  },
  {
    origin: 'X54', name: 'Cadence Conservatory', energy: [0.05, 0.9], scheme: 'free', hue: 0.1,
    pal: { key: 0, s1: 0.45, s2: 0.77 },
    color: { exposure: 1, bloom: 0.6, adapt: 0.15, vignette: 0.4 }, carrier: 'none',
    bodies: [
      body({
        shape: ['compound', { size: 0.29 }],
        parts: [
          part('box', 'union', { sx: 0.85, sy: 1.3, m: 0.65 }),
          part('box', 'subtract', { sx: 0.7, sy: 1.15, m: 0.65, k: 0 }),
          part('capsule', 'union', { y: -1.32, sx: 0.025, sy: 1.02, rot: 0.25, hue: 0.33 }),
        ],
        place: ['mirror', { axis: 0, x: 0.31, y: 0 }],
        material: ['line', { gain: 1.2, width: 1.3, halo: 0.04 }], emit: ['none'],
        color: ['fixed', { detail: 0.9 }],
      }),
      body({
        shape: ['branch', { size: 0.35, width: 0.045, levels: 6, ratio: 0.66, spread: 0.5, grow: 0.83, bend: 0.01 }],
        place: ['mirror', { axis: 0, x: 0.31, y: -0.03 }],
        material: ['line', { gain: 1.6, width: 1.7, halo: 0.08 }], emit: ['none'],
        color: ['height', { hue: 0.42, amount: 0.6, detail: 0.9 }],
      }),
    ],
    reactions: [
      rx('tension', 'sh', 1, 'spread', 0.65, { atk: 0.3, rel: 0.65 }),
      rx('resolve', 'sh', 1, 'grow', 0.4, { atk: 0.12, rel: 1.8 }),
      rx('register', 'pl', 1, 'y', 0.16, { atk: 0.15, rel: 0.3 }),
      rx('chordchange', 'ma', 0, 'gain', 0.3, { atk: 0.05, rel: 0.6 }),
      rx('bass', 'sh', 1, 'width', 0.15, { atk: 0.08, rel: 0.4 }),
    ],
    harmony: { brk: 0.4, warp: 0.22, style: 0, snap: 0.2, walk: 0.045, kick: 0.1, modTurn: 0, calm: 0.2 },
    accent: { hook: 0, kick: 0, hue: 0, drop: 0.1, section: 0.1 },
  },
  {
    origin: 'X55', name: 'Offbeat Letterpress', energy: [0.15, 0.95], scheme: 'free', hue: 0.075,
    pal: { key: 0, s1: 0.54, s2: 0.88 },
    color: { exposure: 0.85, bloom: 0.05, adapt: 0.1, vignette: 0.12, contrast: 0, sat: 0.8 }, carrier: 'none',
    bodies: [
      body({
        shape: ['compound', { size: 0.6 }],
        parts: [part('box', 'union', { sx: 2, sy: 1.2, hue: 0.72 })],
        material: ['fill', { gain: 1.5, soft: 0 }], emit: ['none'], color: ['fixed', { detail: 1 }],
      }),
      body({
        shape: ['compound', { size: 0.15 }],
        parts: [
          part('ellipse', 'union', { sx: 0.9, sy: 1.2 }),
          part('box', 'subtract', { x: 0.4, y: 0.25, sx: 0.8, sy: 0.36, rot: 0.04, k: 0 }),
          part('box', 'union', { x: -0.45, y: -0.75, sx: 0.28, sy: 0.9, rot: -0.04, hue: 0.08 }),
        ],
        place: ['row', { count: 4, y: 0.08, wander: 0 }],
        motion: ['sway', { amp: 0.008, period: 1, tilt: 0.06 }],
        material: ['fill', { gain: 2.1, soft: 0.01, blend: 2 }], emit: ['none'],
        color: ['instrument', { hue: 0.02, amount: 0.25, detail: 0.5 }],
      }),
      body({
        shape: ['segment', { len: 0.65, w: 0.009 }], place: ['point', { y: -0.29, angle: -0.015 }],
        material: ['fill', { gain: 1.8, soft: 0.005, blend: 2 }], emit: ['none'],
        color: ['fixed', { hue: 0.54 }],
      }),
    ],
    reactions: [
      rx('synco', 'pl', 1, 'angle', 0.22, { atk: 0.06, rel: 0.4 }),
      rx('swing', 'mo', 1, 'amp', 0.8, { atk: 0.2, rel: 0.7 }),
      rx('hit', 'pl', 1, 'y', 0.18, { atk: 0.015, rel: 0.3 }),
      rx('attack', 'ma', 1, 'soft', 0.18, { atk: 0.025, rel: 0.25 }),
      rx('bass', 'sh', 2, 'w', 0.5, { atk: 0.06, rel: 0.4 }),
    ],
    groove: { swing: 1.2, sway: 0.025, off: 0.18, lean: 0.7, crisp: 0.2, jitter: 0.2, accent: 0.55 },
    accent: { hook: 0, kick: 0, hue: 0, drop: 0, section: 0.1 },
  },
  {
    origin: 'X56', name: 'Refrain Lantern', energy: [0.05, 0.85], scheme: 'free', hue: 0.07,
    pal: { key: 0, s1: 0.49, s2: 0.8 },
    color: { exposure: 0.85, bloom: 0.5, adapt: 0.15, vignette: 0.4 },
    carrier: 'warp', car: { halfLife: 0.65, floor: 0.18, blur: 0.025 },
    chain: [op('translate', { vx: -0.09, vy: 0.015 }), op('ripple', { amp: 0.00025, freq: 12, speed: 0.4 })],
    bodies: [
      body({
        shape: ['compound', { size: 0.14 }],
        parts: [
          part('ellipse', 'union', { sx: 0.65, sy: 1.1 }),
          part('ellipse', 'subtract', { sx: 0.47, sy: 0.92, k: 0 }),
          part('capsule', 'union', { sx: 0.028, sy: 1.08, hue: 0.2 }),
          part('box', 'union', { y: 1.05, sx: 0.32, sy: 0.07, hue: 0.45 }),
          part('capsule', 'union', { y: -1.45, sx: 0.025, sy: 0.35, hue: 0.45 }),
        ],
        place: ['point', { x: -0.3, y: -0.12 }],
        material: ['line', { gain: 0.9, width: 1.4, halo: 0.03 }], emit: ['trail'],
        color: ['height', { hue: 0, amount: 0.5, detail: 1 }],
      }),
      body({
        shape: ['curve', { form: 0, amp: 0.06, radius: 0.35 }],
        place: ['point', { y: -0.33 }], material: ['line', { gain: 0.35, width: 0.9, halo: 0 }], emit: ['none'],
        color: ['fixed', { hue: 0.49 }],
      }),
    ],
    reactions: [
      rx('hookphase', 'pl', 0, 'x', 1, { atk: 0.05, rel: 0.08 }),
      rx('register', 'pl', 0, 'y', 0.65, { atk: 0.12, rel: 0.25 }),
      rx('hook', 'ma', 0, 'gain', 0.35, { atk: 0.02, rel: 0.3 }),
      rx('legato', 'car', 0, 'blur', 0.15, { atk: 0.2, rel: 0.5 }),
      rx('noteon', 'pl', 0, 'angle', 0.09, { atk: 0.025, rel: 0.35 }),
      rx('bass', 'sh', 1, 'amp', 0.2, { atk: 0.07, rel: 0.4 }),
    ],
    accent: { hook: 0, kick: 0, hue: 0, drop: 0.05, section: 0.15 },
  },
];

// Surface-led compositions: broad tonal shapes, finer detail, and restrained musical gestures.
const SURFACE_STUDIES: Def[] = [
  {
    origin: 'X57', name: 'Nacre Bloom', energy: [0.05, 0.85], scheme: 'analogous', hue: 0.66,
    pal: { key: 0, spread: 0.7 },
    color: { exposure: 0.9, sat: 0.65, bloom: 0.5, adapt: 0.15, vignette: 0.4, contrast: 0.01, ca: 0,
      relief: 0.45, bump: 0.8, gloss: 0.65, metal: 0.25 }, carrier: 'none',
    bodies: [
      body({
        shape: ['cells', { mode: 2, scale: 1.5, warp: 0.75, wall: 0, fill: 0.7, speed: 0.08, var: 0.1, pulse: 0 }],
        material: ['fill', { gain: 0.15 }], emit: ['none'], color: ['fixed', { hue: 0.6 }],
      }),
      body({
        shape: ['fabric', { size: 0.33, aspect: 0.38, folds: 12, depth: 0.8, drape: 0.3, weave: 0.12, flutter: 0.16 }],
        place: ['ring', { n: 11, radius: 0.13, x: -0.07, y: 0.02 }],
        motion: ['spin', { rate: 0.0625 }], deform: ['twist', { amt: 1.5 }],
        material: ['fill', { gain: 1.1, soft: 0.025, core: 0.25 }], emit: ['none'],
        color: ['height', { hue: 0.04, amount: 0.4, detail: 0.65 }],
      }),
      body({
        shape: ['fabric', { size: 0.19, aspect: 0.48, folds: 7, depth: 0.8, drape: 0.4, weave: 0.08, flutter: 0.12 }],
        place: ['ring', { n: 7, radius: 0.055, x: -0.07, y: 0.02, angle: 0.035 }],
        motion: ['spin', { rate: -0.0625 }], deform: ['twist', { amt: -2.4 }],
        material: ['fill', { gain: 1.3, soft: 0.025, core: 0.3 }], emit: ['none'],
        color: ['height', { hue: 0.35, amount: 0.3, detail: 0.8 }],
      }),
    ],
    reactions: [
      rx('held', 'sh', 1, 'drape', 0.45, { atk: 0.25, rel: 0.8 }),
      rx('register', 'de', 2, 'amt', 0.13, { atk: 0.2, rel: 0.4 }),
      rx('bass', 'pl', 1, 'radius', 0.15, { atk: 0.15, rel: 0.55 }),
      rx('vibrato', 'sh', 2, 'flutter', 0.5, { atk: 0.12, rel: 0.45 }),
      rx('bright', 'col', 0, 'gloss', 0.4, { atk: 0.12, rel: 0.6 }),
      rx('hit', 'ma', 2, 'gain', 0.16, { atk: 0.03, rel: 0.4 }),
    ],
    accent: { hook: 0, kick: 0, hue: 0, drop: 0.05, section: 0.1 },
  },
  {
    origin: 'X58', name: 'Tidal Agate', energy: [0.05, 0.9], scheme: 'free', hue: 0.1,
    pal: { key: 0, s1: 0.4, s2: 0.5 },
    color: { exposure: 1, sat: 0.6, bloom: 0.4, adapt: 0.15, vignette: 0.15, contrast: 0, ca: 0,
      relief: 0.4, bump: 0.6, gloss: 0.4, metal: 0.05, huemap: 0.85, bands: 5.5, drift: 0 },
    carrier: 'none',
    bodies: [body({
      shape: ['cells', { mode: 0, scale: 3.1, warp: 0.8, wall: 0.025, fill: 0.85, speed: 0.12, var: 0.08, pulse: 0.08 }],
      material: ['fill', { gain: 2.1 }], emit: ['none'], color: ['fixed', { hue: 0, detail: 0.4 }],
    })],
    reactions: [
      rx('vocals', 'sh', 0, 'warp', 0.3, { atk: 0.35, rel: 0.9 }),
      rx('bass', 'sh', 0, 'wall', 0.07, { atk: 0.15, rel: 0.55 }),
      rx('bright', 'col', 0, 'gloss', 0.4, { atk: 0.2, rel: 0.7 }),
      rx('tension', 'col', 0, 'bands', 0.15, { atk: 0.4, rel: 1.2 }),
      rx('noisy', 'col', 0, 'bump', 0.3, { atk: 0.25, rel: 0.7 }),
    ],
    accent: { hook: 0, kick: 0, hue: 0, drop: 0.04, section: 0.1 },
  },
  {
    origin: 'X59', name: 'Gossamer Atlas', energy: [0.05, 0.9], scheme: 'free', hue: 0.58,
    pal: { key: 0, s1: 0.1, s2: 0.51 },
    color: { exposure: 1.05, sat: 0.8, bloom: 0.55, adapt: 0.2, vignette: 0.25, contrast: 0.01, ca: 0, tonemap: 1 },
    carrier: 'warp', car: { halfLife: 0.16, floor: 0.05, blur: 0 },
    bodies: [body({
      shape: ['flame', { count: 524288, zoom: 0.42, rounds: 2, flow: 0.25, breathe: 0.025 }, [
        xf({ aff: [0.68, -0.32, 0.32, 0.68, 0.18, 0.06], weight: 1, color: 0.05,
          vars: { linear: 0.65, julia: 0.35 }, alt: { linear: 0.65, julia: 0.3, sinusoidal: 0.05 },
          spin: 0.0625, drift: [0.035, 0.025] }),
        xf({ aff: [0.44, 0.18, -0.18, 0.44, -0.65, 0.16], weight: 0.55, color: 0.38,
          vars: { swirl: 0.65, sinusoidal: 0.35 }, alt: { swirl: 0.55, handkerchief: 0.45 },
          drift: [0.055, 0.035], bass: 0.035 }),
        xf({ aff: [0.3, -0.36, 0.36, 0.3, 0.5, -0.35], weight: 0.4, color: 0.85,
          vars: { spherical: 0.3, linear: 0.7 }, alt: { spherical: 0.2, linear: 0.8 },
          spin: -0.0625, drift: [0.045, 0.06], pulse: 0.02 }),
      ]],
      material: ['glow', { gain: 1.1 }], emit: ['trail'],
      color: ['fixed', { hue: 0, detail: 1 }],
    })],
    reactions: [
      rx('held', 'sh', 0, 'zoom', 0.12, { atk: 0.35, rel: 1.1 }),
      rx('bright', 'col', 0, 'bloom', 0.25, { atk: 0.18, rel: 0.75 }),
      rx('hook', 'ma', 0, 'gain', 0.12, { atk: 0.05, rel: 0.5 }),
    ],
    accent: { hook: 0, kick: 0, hue: 0, drop: 0.08, section: 0.15 },
  },
];

// Every new capability ships with three compositions demonstrating different uses.
const SHELL_STUDIES: Def[] = [
  {
    origin: 'X60', name: 'Porcelain Nautilus', energy: [0.05, 0.85], scheme: 'analogous', hue: 0.085,
    pal: { key: 0, spread: 0.55 },
    color: { exposure: 1, sat: 0.5, bloom: 0.45, adapt: 0.12, vignette: 0.5, contrast: 0.01, ca: 0 },
    carrier: 'none',
    bodies: [
      body({
        shape: ['shell', { size: 0.34, turns: 3.1, growth: 0.17, width: 0.24, ribs: 34, relief: 0.7, aperture: 0.4 }],
        place: ['point', { x: -0.06, y: 0.02, angle: -0.09 }],
        motion: ['sway', { amp: 0.008, period: 8, tilt: 0.07 }],
        material: ['fill', { gain: 1.75, soft: 0.01, core: 0.12 }], emit: ['none'],
        color: ['fixed', { hue: 0.05, detail: 0.75 }],
      }),
      body({
        shape: ['dot', { r: 0.003 }], place: ['grid', { scale: 3, density: 0.14, jitter: 0.5, lit: 1, twinkle: 0 }],
        material: ['glow', { gain: 0.2, width: 0.012, base: 0.3 }], emit: ['none'],
        color: ['fixed', { hue: 0.7, detail: 0 }],
      }),
    ],
    reactions: [
      rx('held', 'sh', 0, 'aperture', 0.65, { atk: 0.25, rel: 0.8 }),
      rx('register', 'sh', 0, 'growth', 0.22, { atk: 0.2, rel: 0.5 }),
      rx('bright', 'sh', 0, 'relief', 0.35, { atk: 0.2, rel: 0.6 }),
      rx('bass', 'sh', 0, 'width', 0.12, { atk: 0.12, rel: 0.45 }),
      rx('noteon', 'ma', 1, 'gain', 0.2, { atk: 0.02, rel: 0.4 }),
    ],
    accent: { hook: 0, kick: 0, hue: 0, drop: 0.04, section: 0.1 },
  },
  {
    origin: 'X61', name: 'Malachite Tesserae', energy: [0.1, 0.9], scheme: 'free', hue: 0.38,
    pal: { key: 0, s1: 0.12, s2: 0.7 },
    color: { exposure: 0.9, sat: 0.75, bloom: 0.4, adapt: 0.15, vignette: 0.22, ca: 0,
      relief: 0.25, bump: 0.5, gloss: 0.55, metal: 0.15 }, carrier: 'none',
    bodies: [
      body({
        shape: ['shell', { size: 0.16, turns: 2.4, growth: 0.16, width: 0.3, ribs: 22, relief: 0.5, aperture: 0.1 }],
        place: ['grid', { lattice: 1, scale: 3.4, jitter: 0, density: 1, lit: 1, twinkle: 0, angle: 0.07 }],
        material: ['fill', { gain: 1.3, soft: 0.015 }], emit: ['none'],
        color: ['height', { hue: 0, amount: 0.25, detail: 0.8 }],
      }),
    ],
    reactions: [
      rx('tension', 'sh', 0, 'growth', 0.28, { atk: 0.3, rel: 0.8 }),
      rx('held', 'sh', 0, 'width', 0.18, { atk: 0.2, rel: 0.65 }),
      rx('rough', 'sh', 0, 'relief', 0.65, { atk: 0.2, rel: 0.6 }),
      rx('hit', 'ma', 0, 'gain', 0.18, { atk: 0.03, rel: 0.4 }),
    ],
    accent: { hook: 0, kick: 0, hue: 0, drop: 0.05, section: 0.15 },
  },
  {
    origin: 'X62', name: 'Conch Lanterns', energy: [0.05, 0.85], scheme: 'free', hue: 0.59,
    pal: { key: 0, s1: 0.12, s2: 0.5 },
    color: { exposure: 0.9, sat: 0.75, bloom: 0.65, adapt: 0.15, vignette: 0.4, contrast: 0.01, ca: 0 },
    carrier: 'warp', car: { halfLife: 0.45, floor: 0.25, blur: 0.015 },
    chain: [op('translate', { vx: -0.025, vy: 0.015 })],
    bodies: [body({
      shape: ['shell', { size: 0.13, turns: 1.8, growth: 0.3, width: 0.18, ribs: 40, relief: 0.75, aperture: 0.75 }],
      place: ['float', { count: 4, spread: 0.6, speed: 0.07, fuse: 0 }],
      motion: ['spin', { rate: -0.0625 }],
      material: ['fill', { gain: 1.1, soft: 0.025, halo: 0.04 }], emit: ['cover', { amt: 0.85, tip: 0 }],
      color: ['instrument', { hue: 0, amount: 0.45, detail: 0.75 }],
    })],
    reactions: [
      rx('vocals', 'sh', 0, 'aperture', 0.35, { atk: 0.2, rel: 0.7 }),
      rx('rising', 'sh', 0, 'growth', 0.3, { atk: 0.15, rel: 0.5 }),
      rx('bass', 'sh', 0, 'width', 0.25, { atk: 0.12, rel: 0.5 }),
      rx('noteon', 'ma', 0, 'gain', 0.25, { atk: 0.03, rel: 0.4 }),
    ],
    accent: { hook: 0, kick: 0, hue: 0, drop: 0.04, section: 0.15 },
  },
];

const RECOIL_STUDIES: Def[] = [
  {
    origin: 'X63', name: 'Pearl Bounce', energy: [0.1, 0.9], scheme: 'free', hue: 0.56,
    pal: { key: 0, s1: 0.1, s2: 0.48 },
    color: { exposure: 0.9, sat: 0.55, bloom: 0.5, adapt: 0.15, vignette: 0.4, contrast: 0.01, ca: 0 },
    carrier: 'none',
    bodies: [
      body({
        shape: ['cells', { mode: 2, scale: 1.5, warp: 0.6, fill: 0.5, wall: 0, speed: 0.06, var: 0.1, pulse: 0 }],
        material: ['fill', { gain: 0.12 }], emit: ['none'], color: ['fixed', { hue: 0.2 }],
      }),
      body({
        shape: ['dot', { r: 0.13 }], place: ['row', { count: 4, y: -0.02, wander: 0 }],
        deform: ['wobble', { lobes: 3, amp: 0.025, rate: 0 }],
        motion: ['recoil', { source: 0, distance: 0.13, direction: 0.25, tilt: 0.18, frequency: 1.2, damping: 0.2, fan: 0.32 }],
        material: ['chrome', { gain: 0.95, chrome: 0.65 }], emit: ['none'],
        color: ['instrument', { amount: 0.4, detail: 0.4 }],
      }),
    ],
    reactions: [
      rx('bass', 'sh', 1, 'r', 0.07, { atk: 0.12, rel: 0.45 }),
      rx('held', 'mo', 1, 'damping', 0.4, { atk: 0.3, rel: 0.7 }),
      rx('bright', 'ma', 1, 'chrome', 0.4, { atk: 0.2, rel: 0.65 }),
      rx('noisy', 'de', 1, 'amp', 0.08, { atk: 0.2, rel: 0.6 }),
    ],
    accent: { hook: 0, kick: 0, hue: 0, drop: 0.04, section: 0.1 },
  },
  {
    origin: 'X64', name: 'Kinetic Iris', energy: [0.15, 0.95], scheme: 'free', hue: 0.075,
    pal: { key: 0, s1: 0.5, s2: 0.08 },
    color: { exposure: 0.95, sat: 0.8, bloom: 0.5, adapt: 0.15, vignette: 0.4, contrast: 0.01, ca: 0 },
    carrier: 'none',
    bodies: [
      body({
        shape: ['fabric', { size: 0.27, aspect: 0.33, folds: 8, depth: 0.8, drape: 0.25, weave: 0.08, flutter: 0.06 }],
        place: ['ring', { n: 9, radius: 0.14 }], deform: ['twist', { amt: 1.5 }],
        motion: ['recoil', { source: 2, distance: 0, tilt: 0.7, frequency: 2.5, damping: 0.22 }],
        material: ['fill', { gain: 1.3, soft: 0.015, core: 0.2 }], emit: ['none'],
        color: ['height', { hue: 0, amount: 0.35, detail: 0.55 }],
      }),
      body({
        shape: ['polygon', { n: 8, r: 0.065, round: 0.15 }],
        motion: ['recoil', { source: 0, distance: 0.02, tilt: -0.6, frequency: 3, damping: 0.35 }],
        material: ['line', { gain: 0.9, width: 1.1, halo: 0.02 }], emit: ['none'],
        color: ['fixed', { hue: 0.33, detail: 0.2 }],
      }),
    ],
    reactions: [
      rx('tension', 'mo', 0, 'damping', 0.55, { atk: 0.25, rel: 0.7 }),
      rx('held', 'sh', 0, 'drape', 0.35, { atk: 0.2, rel: 0.6 }),
      rx('hit', 'ma', 1, 'gain', 0.25, { atk: 0.02, rel: 0.35 }),
    ],
    accent: { hook: 0, kick: 0, hue: 0, drop: 0.04, section: 0.1 },
  },
  {
    origin: 'X65', name: 'Afterimage Etude', energy: [0.05, 0.9], scheme: 'free', hue: 0.62,
    pal: { key: 0, s1: 0.28, s2: 0.46 },
    color: { exposure: 0.95, sat: 0.7, bloom: 0.55, adapt: 0.15, vignette: 0.3, contrast: 0.01, ca: 0 },
    carrier: 'warp', car: { halfLife: 1.4, floor: 0.15, blur: 0 },
    chain: [op('rotate', { lock: 0, rate: 0.0008, cx: 0.05, cy: -0.04 }), op('zoom', { rate: 0.0015 })],
    bodies: [body({
      shape: ['superscope', { family: 0, p: 3, q: 5, size: 0.35, audio: 0.08, spinX: 0.0625, spinY: -0.0625, persp: 0.3, n: 2048 }],
      motion: ['recoil', { source: 1, distance: 0.08, direction: 0.08, tilt: 0.55, frequency: 1.8, damping: 0.18 }],
      material: ['line', { gain: 0.75, width: 1.15, halo: 0.02 }], emit: ['trail'],
      color: ['height', { hue: 0, amount: 0.4, detail: 0.8 }],
    })],
    reactions: [
      rx('legato', 'mo', 0, 'damping', 0.5, { atk: 0.25, rel: 0.65 }),
      rx('vibrato', 'sh', 0, 'audio', 0.3, { atk: 0.15, rel: 0.5 }),
      rx('bass', 'sh', 0, 'size', 0.1, { atk: 0.1, rel: 0.5 }),
      rx('hook', 'ma', 0, 'gain', 0.2, { atk: 0.03, rel: 0.4 }),
    ],
    accent: { hook: 0, kick: 0, hue: 0, drop: 0.03, section: 0.1 },
  },
];

// One deformation, three compositions: shaded ribbons, branching fans, and curve trails.
const UNDULATE_STUDIES: Def[] = [
  {
    origin: 'X66', name: 'Silk Currents', energy: [0.05, 0.85], scheme: 'free', hue: 0.47,
    pal: { key: 0, s1: 0.13, s2: 0.45 },
    color: { exposure: 0.95, sat: 0.72, bloom: 0.4, adapt: 0.12, vignette: 0.45, contrast: 0.01, ca: 0 },
    carrier: 'none',
    bodies: [
      body({
        shape: ['cells', { mode: 2, scale: 1.4, warp: 0.45, fill: 0.5, wall: 0, speed: 0.05, var: 0.1, pulse: 0 }],
        material: ['fill', { gain: 0.09 }], emit: ['none'], color: ['fixed', { hue: 0.18 }],
      }),
      body({
        shape: ['fabric', { size: 0.38, aspect: 0.34, folds: 5, depth: 0.9, drape: 0.12, weave: 0.2, flutter: 0.03 }],
        place: ['row', { count: 3, y: 0, wander: 0 }],
        deform: ['undulate', { amp: 0.055, span: 0.66, rate: 0.25, angle: 0.04, pin: 0.5 }],
        material: ['fill', { gain: 1.4, soft: 0.01, core: 0.15 }], emit: ['none'],
        color: ['height', { hue: 0, amount: 0.35, detail: 0.7 }],
      }),
    ],
    reactions: [
      rx('vocals', 'de', 1, 'amp', 0.65, { atk: 0.25, rel: 0.8 }),
      rx('register', 'de', 1, 'span', -0.45, { atk: 0.4, rel: 0.7 }),
      rx('bright', 'sh', 1, 'depth', 0.2, { atk: 0.2, rel: 0.6 }),
    ],
    accent: { hook: 0, kick: 0, hue: 0, drop: 0.03, section: 0.1 },
  },
  {
    origin: 'X67', name: 'Tidal Filigree', energy: [0.1, 0.95], scheme: 'free', hue: 0.085,
    pal: { key: 0, s1: 0.12, s2: 0.52 },
    color: { exposure: 1, sat: 0.75, bloom: 0.5, adapt: 0.12, vignette: 0.3, contrast: 0.01, ca: 0 },
    carrier: 'none',
    bodies: [body({
      shape: ['aurora', { fall: 4, rays: 18, wav: 0.75 }],
      place: ['point', { y: -0.4 }],
      material: ['fill', { gain: 0.25 }], emit: ['none'], color: ['fixed', { hue: 0.48 }],
    }), body({
      shape: ['branch', { size: 0.35, width: 0.018, spread: 0.85, ratio: 0.68, levels: 6, grow: 1, bend: 0.025 }],
      place: ['row', { count: 3, y: -0.08, wander: 0 }],
      deform: ['undulate', { amp: 0.03, span: 0.72, rate: -0.25, angle: 0, pin: 1 }],
      material: ['fill', { gain: 1.8, soft: 0, core: 0, halo: 0.005 }], emit: ['none'],
      color: ['height', { hue: 0, amount: 0.35, detail: 0.65 }],
    })],
    reactions: [
      rx('bass', 'de', 1, 'amp', 0.4, { atk: 0.2, rel: 0.65 }),
      rx('held', 'sh', 1, 'spread', 0.15, { atk: 0.35, rel: 0.8 }),
      rx('tension', 'de', 1, 'span', -0.4, { atk: 0.4, rel: 0.9 }),
      rx('hook', 'ma', 1, 'gain', 0.15, { atk: 0.08, rel: 0.5 }),
    ],
    accent: { hook: 0, kick: 0, hue: 0, drop: 0.03, section: 0.1 },
  },
  {
    origin: 'X68', name: 'Violet Undertow', energy: [0.05, 0.9], scheme: 'free', hue: 0.72,
    pal: { key: 0, s1: 0.2, s2: 0.48 },
    color: { exposure: 0.95, sat: 0.7, bloom: 0.25, adapt: 0.1, vignette: 0.3, contrast: 0.01, ca: 0 },
    carrier: 'warp', car: { halfLife: 0.6, floor: 0.1, blur: 0 },
    chain: [op('zoom', { rate: 0.001 }), op('rotate', { lock: 0, rate: -0.0006 })],
    bodies: [body({
      shape: ['superscope', { family: 1, p: 8, q: 2, size: 0.3, audio: 0.04, spinX: 0.0625, spinY: 0, persp: 0.3, n: 2048 }],
      deform: ['undulate', { amp: 0.055, span: 0.56, rate: 0.5, angle: 0.22, pin: 1 }],
      material: ['line', { gain: 0.28, width: 0.85, halo: 0 }], emit: ['trail'],
      color: ['height', { hue: 0, amount: 0.45, detail: 0.85 }],
    })],
    reactions: [
      rx('held', 'de', 0, 'amp', 0.55, { atk: 0.3, rel: 0.7 }),
      rx('register', 'de', 0, 'span', -0.3, { atk: 0.4, rel: 0.8 }),
      rx('legato', 'de', 0, 'pin', -0.4, { atk: 0.4, rel: 0.7 }),
      rx('noteon', 'ma', 0, 'gain', 0.04, { atk: 0.04, rel: 0.45 }),
    ],
    accent: { hook: 0, kick: 0, hue: 0, drop: 0.02, section: 0.1 },
  },
];

const PLUME_STUDIES: Def[] = [
  {
    origin: 'X69', name: 'Peacock Quill', energy: [0.05, 0.85], scheme: 'free', hue: 0.43,
    pal: { key: 0, s1: 0.18, s2: 0.46 },
    color: { exposure: 1, sat: 0.8, bloom: 0.3, adapt: 0.1, vignette: 0.4, contrast: 0.01, ca: 0 },
    carrier: 'none',
    bodies: [body({
      shape: ['plume', { size: 0.39, width: 0.48, bend: 0.26, taper: 0.7, barbs: 46, split: 0.45, sheen: 0.85 }],
      place: ['point', { angle: -0.1 }], motion: ['sway', { amp: 0.025, tilt: 0.15, period: 8 }],
      material: ['fill', { gain: 1.55, soft: 0.005 }], emit: ['none'],
      color: ['height', { hue: 0, amount: 0.35, detail: 0.9 }],
    })],
    reactions: [
      rx('rising', 'sh', 0, 'bend', 0.45, { atk: 0.25, rel: 0.7 }),
      rx('falling', 'sh', 0, 'bend', -0.45, { atk: 0.25, rel: 0.7 }),
      rx('bright', 'sh', 0, 'sheen', 0.25, { atk: 0.2, rel: 0.6 }),
      rx('noisy', 'sh', 0, 'split', 0.4, { atk: 0.2, rel: 0.6 }),
    ],
    accent: { hook: 0, kick: 0, hue: 0, drop: 0.02, section: 0.1 },
  },
  {
    origin: 'X70', name: 'Gilded Plumage', energy: [0.1, 0.9], scheme: 'free', hue: 0.075,
    pal: { key: 0, s1: 0.16, s2: 0.5 },
    color: { exposure: 1, sat: 0.75, bloom: 0.35, adapt: 0.1, vignette: 0.3, contrast: 0.01, ca: 0 },
    carrier: 'none',
    bodies: [body({
      shape: ['plume', { size: 0.27, width: 0.48, bend: 0.4, taper: 0.65, barbs: 34, split: 0.3, sheen: 0.75 }],
      place: ['ring', { n: 9, radius: 0.13, angle: 0.1 }], motion: ['spin', { rate: 0.0625 }],
      material: ['fill', { gain: 1.4, soft: 0.005 }], emit: ['none'],
      color: ['height', { hue: 0, amount: 0.25, detail: 0.85 }],
    })],
    reactions: [
      rx('held', 'sh', 0, 'width', 0.25, { atk: 0.3, rel: 0.7 }),
      rx('tension', 'sh', 0, 'bend', -0.4, { atk: 0.4, rel: 0.8 }),
      rx('bass', 'pl', 0, 'radius', 0.1, { atk: 0.15, rel: 0.5 }),
      rx('attack', 'sh', 0, 'split', 0.45, { atk: 0.1, rel: 0.5 }),
    ],
    accent: { hook: 0, kick: 0, hue: 0, drop: 0.03, section: 0.1 },
  },
  {
    origin: 'X71', name: 'Kingfisher Brocade', energy: [0.1, 0.9], scheme: 'free', hue: 0.52,
    pal: { key: 0, s1: 0.12, s2: 0.48 },
    color: { exposure: 0.95, sat: 0.8, bloom: 0.3, adapt: 0.1, vignette: 0.25, contrast: 0.01, ca: 0,
      relief: 0.18, bump: 0.4, gloss: 0.4, metal: 0.1 },
    carrier: 'none',
    bodies: [body({
      shape: ['plume', { size: 0.135, width: 0.65, bend: -0.3, taper: 0.5, barbs: 24, split: 0.2, sheen: 0.8 }],
      place: ['grid', { lattice: 1, scale: 3.5, jitter: 0, density: 1, lit: 1, twinkle: 0, angle: -0.05 }],
      material: ['fill', { gain: 1.3, soft: 0.005 }], emit: ['none'],
      color: ['height', { hue: 0, amount: 0.25, detail: 0.9 }],
    })],
    reactions: [
      rx('register', 'sh', 0, 'taper', 0.5, { atk: 0.4, rel: 0.7 }),
      rx('rough', 'sh', 0, 'split', 0.55, { atk: 0.2, rel: 0.5 }),
      rx('vocals', 'sh', 0, 'bend', 0.35, { atk: 0.25, rel: 0.7 }),
      rx('noteon', 'ma', 0, 'gain', 0.12, { atk: 0.05, rel: 0.4 }),
    ],
    accent: { hook: 0, kick: 0, hue: 0, drop: 0.03, section: 0.1 },
  },
];

const WEAVE_STUDIES: Def[] = [
  {
    origin: 'X72', name: 'Swallow Waltz', energy: [0.05, 0.85], scheme: 'free', hue: 0.48,
    pal: { key: 0, s1: 0.16, s2: 0.5 },
    color: { exposure: 1, sat: 0.7, bloom: 0.3, adapt: 0.12, vignette: 0.4, contrast: 0.01, ca: 0 },
    carrier: 'none',
    bodies: [body({
      shape: ['aurora', { fall: 4, rays: 14, wav: 0.7 }], place: ['point', { y: -0.42 }],
      material: ['fill', { gain: 0.35 }], emit: ['none'], color: ['fixed', { hue: 0.25 }],
    }), body({
      shape: ['plume', { size: 0.2, width: 0.3, bend: 0.4, taper: 0.9, barbs: 26, split: 0.35, sheen: 0.7 }],
      place: ['row', { count: 3, y: 0.03, wander: 0, angle: -0.08 }],
      motion: ['weave', { radius: 0.14, height: 0.08, period: 4, stagger: 0.22, bank: 0.65, counter: 1 }],
      material: ['fill', { gain: 1.4, soft: 0.005 }], emit: ['none'],
      color: ['instrument', { hue: 0, amount: 0.3, detail: 0.8 }],
    })],
    reactions: [
      rx('swing', 'mo', 1, 'height', 0.5, { atk: 0.3, rel: 0.8 }),
      rx('bass', 'mo', 1, 'radius', 0.35, { atk: 0.2, rel: 0.6 }),
      rx('legato', 'mo', 1, 'bank', -0.25, { atk: 0.35, rel: 0.8 }),
      rx('rising', 'sh', 1, 'bend', 0.3, { atk: 0.2, rel: 0.6 }),
    ],
    accent: { hook: 0, kick: 0, hue: 0, drop: 0.02, section: 0.1 },
  },
  {
    origin: 'X73', name: 'Loom of Light', energy: [0.1, 0.95], scheme: 'free', hue: 0.57,
    pal: { key: 0, s1: 0.23, s2: 0.48 },
    color: { exposure: 0.95, sat: 0.75, bloom: 0.4, adapt: 0.1, vignette: 0.25, contrast: 0.01, ca: 0 },
    carrier: 'warp', car: { halfLife: 0.7, floor: 0.1, blur: 0 },
    chain: [op('zoom', { rate: 0.001 })],
    bodies: [body({
      shape: ['superscope', { family: 0, p: 2, q: 5, size: 0.16, audio: 0.04, spinX: 0.0625, spinY: -0.0625, persp: 0.25, n: 2048 }],
      place: ['orbit', { count: 3, radius: 0.05, rate: 0, follow: 0 }],
      motion: ['weave', { radius: 0.26, height: 0.14, period: 4, stagger: 0.33, bank: 0.3, counter: 1 }],
      material: ['line', { gain: 0.6, width: 0.9, halo: 0 }], emit: ['trail'],
      color: ['height', { hue: 0, amount: 0.5, detail: 0.9 }],
    })],
    reactions: [
      rx('held', 'mo', 0, 'height', 0.35, { atk: 0.3, rel: 0.7 }),
      rx('synco', 'mo', 0, 'bank', 0.4, { atk: 0.25, rel: 0.8 }),
      rx('register', 'sh', 0, 'size', 0.18, { atk: 0.35, rel: 0.7 }),
      rx('hook', 'ma', 0, 'gain', 0.08, { atk: 0.08, rel: 0.45 }),
    ],
    accent: { hook: 0, kick: 0, hue: 0, drop: 0.02, section: 0.1 },
  },
  {
    origin: 'X74', name: 'Porcelain Procession', energy: [0.05, 0.9], scheme: 'free', hue: 0.08,
    pal: { key: 0, s1: 0.12, s2: 0.48 },
    color: { exposure: 1, sat: 0.5, bloom: 0.3, adapt: 0.1, vignette: 0.35, contrast: 0.01, ca: 0 },
    carrier: 'none',
    bodies: [body({
      shape: ['cells', { mode: 2, scale: 1.6, warp: 0.5, fill: 0.5, wall: 0, speed: 0.05, var: 0.1, pulse: 0 }],
      material: ['fill', { gain: 0.1 }], emit: ['none'], color: ['fixed', { hue: 0.5 }],
    }), body({
      shape: ['shell', { size: 0.1, turns: 2.2, growth: 0.24, width: 0.25, ribs: 32, relief: 0.6, aperture: 0.5 }],
      place: ['orbit', { count: 4, radius: 0.13, rate: 0, follow: 0 }],
      motion: ['weave', { radius: 0.24, height: 0.12, period: 8, angle: 0.08, stagger: 0.25, bank: 0.45, counter: 0 }],
      material: ['fill', { gain: 1.6, soft: 0.005 }], emit: ['none'],
      color: ['height', { hue: 0, amount: 0.25, detail: 0.7 }],
    })],
    reactions: [
      rx('tension', 'mo', 1, 'radius', -0.5, { atk: 0.4, rel: 0.9 }),
      rx('resolve', 'mo', 1, 'height', 0.4, { atk: 0.25, rel: 0.8 }),
      rx('bright', 'sh', 1, 'relief', 0.35, { atk: 0.2, rel: 0.6 }),
      rx('noteon', 'ma', 1, 'gain', 0.1, { atk: 0.05, rel: 0.45 }),
    ],
    accent: { hook: 0, kick: 0, hue: 0, drop: 0.02, section: 0.1 },
  },
];

const ALL_DEFS: Def[] = [...DEFS, ...MILKDROP, ...CHOREO, ...PHYSICS, ...AVS, ...RAYMARCH, ...AGENTS, ...ECOSYSTEM, ...DRIFT, ...LANDSCAPE, ...HARMONY, ...GROOVE, ...DEJAVU, ...EVOLVED, ...TIMBRE, ...LYRICS, ...NOTES, ...ART, ...ART2, ...ART3, ...ART4, ...TORONTO_X36, ...ART5, ...LILIES, ...TORONTO_X37, ...TORONTO_X38, ...FLOWERS, ...STUDIES, ...BODY_STUDIES, ...MUSIC_STUDIES, ...SURFACE_STUDIES, ...SHELL_STUDIES, ...RECOIL_STUDIES, ...UNDULATE_STUDIES, ...PLUME_STUDIES, ...WEAVE_STUDIES];
export const SEEDS: Seed[] = [...ALL_DEFS.map(build), ...USER_GENOMES.map((u) => ({ origin: u.origin, name: u.name, genome: repair(u.genome) }))];
/** The reactions each seed was written with, before repair (the tests check repair kept every one as written). */
export const SEED_DECLARED_REACTIONS: Record<string, readonly ReactionGene[]> = Object.fromEntries([
  ...ALL_DEFS.map((d) => [d.origin, d.reactions ?? []] as const),
  ...USER_GENOMES.map((u) => [u.origin, ((u.genome as { reactions?: ReactionGene[] }).reactions ?? [])] as const),
]);
