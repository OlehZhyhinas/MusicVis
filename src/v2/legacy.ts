// The format-2 genome (one emitter object per light source), kept only to read
// old data: stored populations, exported files and pre-v3 children. repair()
// here canonicalises a format-1/2 genome exactly as the old code did, and
// upgradeV2() (bottom) re-expresses it in the format-3 body vocabulary.

import { FLAME_VARIATIONS, type FlameVar } from './variations';

export { FLAME_VARIATIONS };
export type { FlameVar };

export interface ParamSpec {
  min: number;
  max: number;
  def: number;
  int?: boolean;
  log?: boolean;
  choices?: number[];
}
export type Schema = Record<string, ParamSpec>;
export type Params = Record<string, number>;

const P = (min: number, max: number, def: number, extra: Partial<ParamSpec> = {}): ParamSpec => ({ min, max, def, ...extra });
const I = (min: number, max: number, def: number): ParamSpec => ({ min, max, def, int: true });
const C = (choices: number[], def: number): ParamSpec => ({ min: Math.min(...choices), max: Math.max(...choices), def, choices });
const LOCKS = [-0.25, -0.125, -0.0625, 0, 0.0625, 0.125, 0.25];

// ------------------------------------------------------------------ chain

export const MOTION_OPS = ['zoom', 'rotate', 'translate', 'swirl', 'twist', 'ripple', 'noise', 'push'] as const;
export const FOLD_OPS = ['mirror', 'tile', 'polar', 'kaleido', 'stretch'] as const;
export const VAR_OPS = FLAME_VARIATIONS.map((v) => `v_${v}`) as `v_${FlameVar}`[];
export type OpKind = (typeof MOTION_OPS)[number] | (typeof FOLD_OPS)[number] | `v_${FlameVar}`;
export const OP_KINDS: OpKind[] = [...MOTION_OPS, ...FOLD_OPS, ...VAR_OPS];
export type Stage = 'warp' | 'view';

const VAR_SCHEMA: Schema = { s: P(0.4, 3, 1.2) };

export const OP_SCHEMAS: Record<string, Schema> = {
  // rate > 0: content streams outward (flying in); < 0: inward.
  zoom: { rate: P(-0.02, 0.06, 0.008), cx: P(-0.4, 0.4, 0), cy: P(-0.4, 0.4, 0), radial: P(0, 1, 0), wander: P(0, 0.35, 0) },
  // lock: turns per bar (bar-locked spin); rate: free rotation, rad per frame.
  // wander: the centre follows the same bar-locked path as zoom / swirl wander.
  rotate: { lock: C(LOCKS, 0.0625), rate: P(-0.01, 0.01, 0), cx: P(-0.4, 0.4, 0), cy: P(-0.4, 0.4, 0), alt: C([0, 1], 0), wander: P(0, 0.35, 0) },
  // content velocity, p units per second (applied as whole-pixel shifts).
  // lanes > 1: columns (or rows) per unit that move at 1x or 2x speed, like rain.
  translate: { vx: P(-0.5, 0.5, 0), vy: P(-0.5, 0.5, 0), lanes: P(0, 50, 0) },
  swirl: { amt: P(-0.03, 0.03, 0.01), k: P(1, 12, 6), cx: P(-0.4, 0.4, 0), cy: P(-0.4, 0.4, 0), wander: P(0, 0.35, 0) },
  twist: { amt: P(-0.012, 0.012, 0.003), cx: P(-0.4, 0.4, 0), cy: P(-0.4, 0.4, 0) },
  ripple: { amp: P(0, 0.004, 0.0008), freq: P(2, 40, 8), speed: P(0.2, 4, 0.7), radial: C([0, 1], 0) },
  noise: { amp: P(0, 0.003, 0.0012), scale: P(0.8, 5, 2), speed: P(0.05, 1, 0.3) },
  // Constant outward shift (p units per frame) away from the spine (axis 0),
  // the horizon line (1) or the centre (2): ink pushed apart, beat pushes via reactions.
  push: { amt: P(-0.01, 0.01, 0.001), axis: C([0, 1, 2], 0) },
  mirror: { axis: C([0, 1, 2], 0) },
  tile: { n: P(1.2, 4, 2) },
  polar: { scale: P(0.4, 1.6, 1), lock: C(LOCKS, 0) },
  kaleido: { n: I(3, 12, 6), lock: C(LOCKS, 0.0625) },
  // Spectrum stretch: the screen is cut into vertical strips, one spectrum band
  // each; content above (and mirrored below) the base line stretches taller in
  // loud strips and on the beat. win brightens loud strips, sky adds a bass-lit
  // glow above the base. In the view stage it acts on the displayed picture
  // only; in the warp stage it slowly pulls content up the loud strips.
  stretch: { base: P(-0.4, 0.2, -0.16), amt: P(0, 1.5, 0.9), beat: P(0, 0.4, 0.18), strips: I(8, 64, 32), win: P(0, 1, 1), sky: P(0, 1, 0) },
};
for (const v of VAR_OPS) OP_SCHEMAS[v] = VAR_SCHEMA;

/**
 * Ops allowed in the draw-space chain. Their per-frame rates are reinterpreted
 * as absolute shaping amounts (see engine.ts packDraw); translate, push,
 * stretch, tile and polar stay carrier / view-stage only.
 */
export const DRAW_OPS: OpKind[] = ['swirl', 'twist', 'ripple', 'noise', 'rotate', 'zoom', 'mirror', 'kaleido', ...VAR_OPS];
export const MAX_DRAW = 3;
/** GLSL type id of a draw op (engine uniform uDrB[i].x); variations are 20 + their index. */
export function drawOpId(op: OpKind): number {
  const fixed = ['', 'swirl', 'twist', 'ripple', 'noise', 'rotate', 'zoom', 'mirror', 'kaleido'].indexOf(op);
  if (fixed > 0) return fixed;
  return isVarOp(op) ? 20 + FLAME_VARIATIONS.indexOf(op.slice(2) as FlameVar) : 0;
}
export function isDrawOp(op: string): boolean {
  return (DRAW_OPS as string[]).includes(op);
}

export function isFold(op: OpKind): boolean {
  return (FOLD_OPS as readonly string[]).includes(op);
}
export function isVarOp(op: OpKind): boolean {
  return op.startsWith('v_');
}
/** Ops that may live in either stage; motion ops are warp-only. */
export function stageFree(op: OpKind): boolean {
  return isFold(op) || isVarOp(op);
}

export interface OpGene {
  op: OpKind;
  stage: Stage;
  /** Strength 0..1 (blend amount for variations, rate multiplier for motion, unused by folds). */
  w: number;
  p: Params;
}

// --------------------------------------------------------------- emitters

export const EMITTER_KINDS = [
  'wave', 'spectrum', 'particles', 'stars', 'ink', 'wire', 'plasma', 'aurora', 'blobs', 'flame', 'edge', 'tiles', 'horizon', 'orb',
  'snake', 'merge',
] as const;
export type EmitterKind = (typeof EMITTER_KINDS)[number];
/** Every kind a plain (non-merge) emitter can have: random picks and merge parts. */
export const BASIC_KINDS: EmitterKind[] = EMITTER_KINDS.filter((k) => k !== 'merge');
/**
 * Kinds with a 2D signed distance field (glsl.ts SDF_GLSL): they can be one
 * side of a smooth union / morph, or the region a mask merge emits inside.
 * The others take part in merges only as the masked consumer.
 */
export const SDF_KINDS: EmitterKind[] = ['orb', 'wire', 'wave', 'spectrum', 'snake', 'aurora', 'blobs', 'ink'];
/** Consumers drawn as geometry: a mask merge masks their feedback trail. */
export const GEOMETRY_KINDS: EmitterKind[] = ['particles', 'flame'];
export const isSdfKind = (k: EmitterKind) => SDF_KINDS.includes(k);
export type Layer = 'fb' | 'top';

const COMMON_EMIT: Schema = { gain: P(0.05, 3, 1), hue: P(0, 1, 0) };

export const EMITTER_SCHEMAS: Record<EmitterKind, Schema> = {
  // shape: 0 line, 1 circle, 2 spiral, 3 lissajous, 4 arc (span turns * 0.37 rad),
  // 5 pendulum harmonograph: ratios walk the circle of fifths from the key every rb
  // bars (offset ra), phases advance with the beat, downbeats swing it.
  wave: { ...COMMON_EMIT, shape: C([0, 1, 2, 3, 4, 5], 0), amp: P(0.05, 0.5, 0.24), x: P(-0.3, 0.3, 0), y: P(-0.35, 0.35, 0), radius: P(0.08, 0.45, 0.27), turns: P(1, 6, 3), ra: I(1, 6, 2), rb: I(1, 6, 3), thick: P(0.8, 3, 1.5) },
  // mode: 0 bars on a baseline, 1 upper arc, 2 full ring, 3 mirrored bars
  spectrum: { ...COMMON_EMIT, mode: C([0, 1, 2, 3], 1), x: P(-0.3, 0.3, 0), y: P(-0.45, 0.2, -0.42), bins: I(12, 64, 40), radius: P(0.08, 0.35, 0.2), len: P(0.1, 0.5, 0.3), fill: P(0.3, 0.9, 0.6) },
  // spawn: 0 anywhere, 1 waveform, 2 rotating emitters, 3 ring, 4 centre, 5 bottom edge
  // surge: speed and zoom flow follow the beat-surge envelope (fast attack, slow ease, jumps on drops)
  particles: { ...COMMON_EMIT, spawn: C([0, 1, 2, 3, 4, 5], 4), count: P(1024, 65536, 16384, { log: true, int: true }), size: P(1.5, 6, 2.6), speed: P(0.05, 1, 0.3), curl: P(0, 0.5, 0.05), zoomFlow: P(-0.5, 1.5, 0.5), lift: P(-0.2, 0.2, 0), drag: P(1, 5, 2.5), life: P(0.1, 0.8, 0.3), spread: P(0.01, 0.2, 0.05), surge: P(0, 1, 0) },
  stars: { ...COMMON_EMIT, density: P(0.2, 0.8, 0.5), scale: P(3, 12, 6.5), links: P(0, 1, 0.8), twinkle: P(0, 1, 0.3), drift: P(0, 0.1, 0.03) },
  // orbit 0: blobs sit at stem stations; 1: they circle the centre, bar locked
  // row 1: blobs line up along the bottom edge (smoke sources)
  // inst 1: sources follow their instruments (drums jump to a new spot every bar,
  //   or on every hit with jump 1; bass swings out with the bass; vocals follow
  //   the melody), fluid pushes step their aim on hits and pulse on the beat at
  //   a tempo-scaled speed. xs scales how far they roam sideways.
  // follow: the orbit centre wanders on the bar-locked path. swap 1: colours trade places every bar.
  ink: { ...COMMON_EMIT, count: I(1, 6, 4), orbit: P(0, 1, 0), row: P(0, 1, 0), radius: P(0.1, 0.45, 0.3), size: P(0.008, 0.05, 0.02), wander: P(0, 0.2, 0.1), force: P(0, 1.5, 1), inst: P(0, 1, 0), xs: P(0.02, 1, 1), jump: C([0, 1], 0), follow: P(0, 0.35, 0), swap: C([0, 1], 0) },
  // solid: 0 tetra, 1 cube, 2 octa, 3 icosa, 4 polygon, 5 follow sections
  wire: { ...COMMON_EMIT, solid: C([0, 1, 2, 3, 4, 5], 5), sides: I(3, 8, 6), scale: P(0.06, 0.35, 0.2), tilt: P(0, 1, 0.45), lock: C(LOCKS, 0.25), inner: P(0, 1, 0.6), thick: P(0.8, 2.5, 1.3) },
  // tempo 1: contours flow a quarter band per beat (smooth, no bursts) and a smoothed bass swells the warp
  // and thickens the lines; pulse: lines brighten on the beat; melHue: the melody shifts the hue.
  plasma: { ...COMMON_EMIT, scale: P(0.8, 3, 1.7), warp: P(0.5, 3, 2), bands: P(3, 12, 8), lines: P(0, 1, 0.8), speed: P(0.05, 0.4, 0.15), tempo: P(0, 1, 0), pulse: P(0, 1, 0), melHue: P(0, 1, 0) },
  aurora: { ...COMMON_EMIT, y: P(-0.3, 0.25, 0), fall: P(2, 10, 5), rays: P(8, 40, 24), wav: P(0.5, 3, 1.4) },
  blobs: { ...COMMON_EMIT, count: I(3, 6, 6), size: P(0.03, 0.12, 0.06), spread: P(0.3, 0.7, 0.55), speed: P(0.05, 0.3, 0.12), chrome: P(0, 1, 1) },
  // flow: vars <-> alt morph cycles per 8 bars (0 = only drops morph); breathe: camera zoom per unit bass
  flame: { ...COMMON_EMIT, count: C([65536, 131072, 262144, 524288], 262144), zoom: P(0.1, 0.45, 0.22), camSpin: C(LOCKS, 0.0625), rounds: I(1, 2, 2), ox: P(-0.2, 0.2, 0), oy: P(-0.2, 0.2, 0), flow: P(0, 2, 0), breathe: P(0, 0.4, 0) },
  // mode: 0 skyline, 1 melody ribbon, 2 spectral rain, 3 terrain ridge. side: 0 right, 1 top, 2 bottom, 3 left
  edge: { ...COMMON_EMIT, mode: C([0, 1, 2, 3], 0), side: C([0, 1, 2, 3], 0), base: P(-0.3, 0.3, -0.16), height: P(0.1, 0.5, 0.3), density: P(0.2, 1, 0.6) },
  // shape: 0 hex, 1 square, 2 triangle
  tiles: { ...COMMON_EMIT, shape: C([0, 1, 2], 0), scale: P(3, 10, 5), lock: C(LOCKS, 0.0625), lit: P(0.2, 0.8, 0.5), edges: P(0, 1, 0.6) },
  // terrain: ray-marched hills either side of a valley, following the spectrum, with a kick ridge
  // rolling toward the viewer and lines coloured by height; flash: the grid flashes on downbeats.
  horizon: { ...COMMON_EMIT, y: P(-0.3, 0.1, -0.06), speed: P(0.2, 1, 0.5), density: P(0.5, 3, 1.5), peaks: P(0, 1, 1), terrain: P(0, 1, 0), flash: P(0, 1, 0) },
  // arms: five tapered, curling arms that reach with drums, bass, vocals, other and loudness plus a
  // breath of their own, sway every 2 bars and turn every 16. bob: drift per bar, bob on the beat.
  // clip: nothing is drawn below this height (a sun setting behind the horizon).
  orb: { ...COMMON_EMIT, x: P(-0.5, 0.5, 0), y: P(-0.3, 0.35, 0.1), radius: P(0.04, 0.25, 0.1), halo: P(0, 1, 0.5), stripes: P(0, 1, 0), craters: P(0, 1, 0), arms: P(0, 1, 0), bob: P(0, 1, 0), clip: P(-0.6, 0.35, -0.6) },
  // Two heads (melody, bass) roaming at free angles: they turn on beats / downbeats (sharply on heavy
  // hits), curve gently between turns, bounce off the edges and move step units per beat. cover 1:
  // the newest body paints over older trail; 0: it adds light.
  snake: { ...COMMON_EMIT, count: I(1, 2, 2), step: P(0.03, 0.2, 0.11), turn: P(0, 1.5, 1), curve: P(0, 2, 0.8), width: P(0.5, 3, 1), cover: P(0, 1, 1) },
  // Two parts fused into one shape. mode 0: smooth union (k = blend radius, grows with the bass);
  // 1: morph mix(sdA, sdB, t); 2: mask, part B only lights up inside (inside 1) or along (0) part A.
  // t drifts toward a music driver by depth: drive 0 none, 1 bar-locked sweep every rate bars,
  // 2 bass, 3 melody, 4 loudness, 5 beat surge. line / fill: outline and body brightness.
  merge: { ...COMMON_EMIT, mode: C([0, 1, 2], 0), k: P(0.02, 0.25, 0.08), t: P(0, 1, 0.5), drive: C([0, 1, 2, 3, 4, 5], 1), depth: P(0, 1, 0.6), rate: C([2, 4, 8, 16], 8), line: P(0, 1, 0.7), fill: P(0, 1, 0.5), width: P(0.5, 3, 1.2), inside: C([0, 1], 1) },
};

export interface FlameXformGene {
  /** [a, b, c, d, e, f]: x' = a x + b y + e, y' = c x + d y + f */
  aff: number[];
  weight: number; // 0.05..1
  color: number; // 0..1
  vars: Partial<Record<FlameVar, number>>; // each 0..1
  /** Variation mix the transform morphs toward (flow cycles / drops); omitted = no morph. */
  alt?: Partial<Record<FlameVar, number>>;
  spin: number; // turns per bar, from LOCKS
  bass: number; // 0..0.3 scale per unit bass
  drift: [number, number]; // 0..0.4, bar-locked drift of the translation
  pulse: number; // 0..0.3 scale kick per beat pulse
}
export const MAX_XFORMS = 4;
export const AFF_RANGE = [-1.2, 1.2];
export const XFORM_SPIN = LOCKS;
export const XFORM_DRIFT = 0.4;

export interface EmitterGene {
  kind: EmitterKind;
  layer: Layer;
  p: Params;
  xforms?: FlameXformGene[]; // flame only
  /** merge only: [shape A, shape B (or the masked consumer in mode 2)], plain kinds. */
  parts?: EmitterGene[];
}

// ---------------------------------------------------------------- carrier

export const CARRIER_KINDS = ['warp', 'fluid', 'flow', 'none'] as const;
export type CarrierKind = (typeof CARRIER_KINDS)[number];
export const CARRIER_SCHEMA: Schema = {
  halfLife: P(0.04, 25, 0.5, { log: true }), // seconds for the feedback to fade to half
  floor: P(0, 2, 1), // black-level subtraction
  blur: P(0, 0.4, 0),
  amount: P(0.4, 1.5, 1), // fluid advection amount
  vort: P(5, 40, 28),
  fnoise: P(0, 0.6, 0.35),
  fscale: P(1, 4, 2),
  famt: P(0.0004, 0.003, 0.0012),
};
export interface CarrierGene {
  kind: CarrierKind;
  p: Params;
}

// ----------------------------------------------------------------- colour

export const SCHEMES = ['analogous', 'complementary', 'triad', 'split', 'mono'] as const;
export type Scheme = (typeof SCHEMES)[number];
export const COLOR_SCHEMA: Schema = {
  hue: P(0, 1, 0.5),
  sat: P(0.2, 1, 0.9),
  exposure: P(0.6, 1.4, 1),
  contrast: P(0, 0.08, 0.03),
  bloom: P(0.4, 1.5, 1),
  adapt: P(0.1, 0.6, 0.3),
  vignette: P(0, 0.7, 0.45),
  ca: P(0, 0.008, 0.0015),
  reflect: C([0, 1], 0),
  reflectY: P(-0.45, 0, -0.16),
  tonemap: C([0, 1], 0), // 1 = flame log-density
};
export interface ColorGene {
  scheme: Scheme;
  p: Params;
}

// -------------------------------------------------------------- reactions

// surge: beat envelope with a fast attack and slow ease that cruises with loudness and jumps on drops.
export const SIGNALS = ['drums', 'bass', 'vocals', 'other', 'hit', 'beat', 'bar', 'complexity', 'drop', 'loud', 'melody', 'build', 'surge'] as const;
export type Signal = (typeof SIGNALS)[number];
// dr: the draw-space chain. em indexes flatEmitters() (a merge, then its two parts).
export type GeneGroup = 'op' | 'em' | 'car' | 'col' | 'dr';
export const GENE_GROUPS: GeneGroup[] = ['op', 'em', 'car', 'col', 'dr'];
export interface ReactionGene {
  src: Signal;
  g: GeneGroup;
  i: number; // index into chain / flat emitters / draw chain (0 for car / col)
  k: string; // parameter name
  gain: number; // -1..1, fraction of the parameter's half range per unit signal
}
export const MAX_REACTIONS = 6;
/** Parameters reactions may not touch (structural switches). */
const NO_REACT = new Set(['mode', 'shape', 'spawn', 'solid', 'side', 'axis', 'count', 'reflect', 'tonemap', 'alt', 'radial', 'rounds', 'lock', 'camSpin', 'sides', 'ra', 'rb', 'n', 'bins', 'halfLife', 'lanes', 'strips', 'drive', 'rate', 'inside']);

// ----------------------------------------------------------------- genome

/** Genome format. 1: before draw chains and merges (still loads unchanged). */
export const GENOME_VERSION = 2;

export interface Genome {
  v: 2;
  chain: OpGene[];
  /** Draw-space shaping ops (absent when empty, as in every format-1 genome). */
  draw?: OpGene[];
  emitters: EmitterGene[];
  carrier: CarrierGene;
  color: ColorGene;
  reactions: ReactionGene[];
  energy: [number, number]; // complexity range the preset suits
}

export const MAX_CHAIN = 6;
/** Hard cap for any genome (older children may have 3). */
export const MAX_EMITTERS = 3;
/** Cap for newly bred children: one body, a second layer only by a rare mutation. */
export const MAX_CHILD_EMITTERS = 2;
/** Emitters plus merge parts: the shaders have four emitter uniform slots. */
export const MAX_FLAT = 4;

/** Emitters with each merge followed by its two parts: reaction 'em' indices and shader slots. */
export function flatEmitters(g: Pick<Genome, 'emitters'>): EmitterGene[] {
  const out: EmitterGene[] = [];
  for (const e of g.emitters) {
    out.push(e);
    if (e.kind === 'merge' && e.parts) out.push(...e.parts);
  }
  return out;
}

export type Species =
  | 'terrain' | 'rain' | 'vortex' | 'ink' | 'scope' | 'plasma' | 'mirror'
  | 'spectrum' | 'stars' | 'wire' | 'chrome' | 'aurora' | 'flame';
export const SPECIES: Species[] = ['terrain', 'rain', 'vortex', 'ink', 'scope', 'plasma', 'mirror', 'spectrum', 'stars', 'wire', 'chrome', 'aurora', 'flame'];
export const SPECIES_LABEL: Record<Species, string> = {
  terrain: 'terrain/skyline', rain: 'rain/smoke', vortex: 'vortex/tunnel', ink: 'ink/fluid', scope: 'oscilloscope',
  plasma: 'plasma/field', mirror: 'mirror/tiled', spectrum: 'spectrum', stars: 'starfield', wire: 'wireframe',
  chrome: 'chrome/blobs', aurora: 'aurora', flame: 'fractal flame',
};
export type Energy = 'calm' | 'energetic';

// ---------------------------------------------------------------- helpers

export function clampParam(v: number, s: ParamSpec): number {
  if (!Number.isFinite(v)) v = s.def;
  if (s.choices) {
    let best = s.choices[0];
    for (const c of s.choices) if (Math.abs(c - v) < Math.abs(best - v)) best = c;
    return best;
  }
  v = Math.min(s.max, Math.max(s.min, v));
  return s.int ? Math.round(v) : v;
}

export function inRange(v: number, s: ParamSpec): boolean {
  if (typeof v !== 'number' || !Number.isFinite(v)) return false;
  if (s.choices) return s.choices.includes(v);
  if (v < s.min - 1e-9 || v > s.max + 1e-9) return false;
  return !s.int || Number.isInteger(v);
}

function repairParams(p: unknown, schema: Schema): Params {
  const src = (p && typeof p === 'object' ? p : {}) as Record<string, unknown>;
  const out: Params = {};
  for (const k of Object.keys(schema)) {
    const v = src[k];
    out[k] = clampParam(typeof v === 'number' ? v : schema[k].def, schema[k]);
  }
  return out;
}

export function defaultParams(schema: Schema): Params {
  const out: Params = {};
  for (const k of Object.keys(schema)) out[k] = schema[k].def;
  return out;
}

export function schemaFor(g: Genome, group: GeneGroup, i: number): Schema | null {
  if (group === 'op') return g.chain[i] ? OP_SCHEMAS[g.chain[i].op] : null;
  if (group === 'dr') return g.draw?.[i] ? OP_SCHEMAS[g.draw[i].op] : null;
  if (group === 'em') {
    const e = flatEmitters(g)[i];
    return e ? EMITTER_SCHEMAS[e.kind] : null;
  }
  if (group === 'car') return CARRIER_SCHEMA;
  return COLOR_SCHEMA;
}

export function reactable(schema: Schema): string[] {
  return Object.keys(schema).filter((k) => !NO_REACT.has(k) && !schema[k].choices);
}

const num = (v: unknown, d: number) => (typeof v === 'number' && Number.isFinite(v) ? v : d);
const clamp = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v));

export function repairXform(x: Partial<FlameXformGene> | undefined): FlameXformGene {
  const src = x ?? {};
  const aff = Array.isArray(src.aff) ? src.aff.slice(0, 6) : [];
  while (aff.length < 6) aff.push(aff.length === 0 || aff.length === 3 ? 0.5 : 0);
  const vars: Partial<Record<FlameVar, number>> = {};
  let any = false;
  for (const v of FLAME_VARIATIONS) {
    const w = num(src.vars?.[v], 0);
    if (w > 0.001) {
      vars[v] = clamp(w, 0, 1);
      any = true;
    }
  }
  if (!any) vars.linear = 1;
  let alt: Partial<Record<FlameVar, number>> | undefined;
  if (src.alt && typeof src.alt === 'object') {
    for (const v of FLAME_VARIATIONS) {
      const w = num(src.alt[v], 0);
      if (w > 0.001) (alt ??= {})[v] = clamp(w, 0, 1);
    }
  }
  const out: FlameXformGene = {
    aff: aff.map((a) => clamp(num(a, 0), AFF_RANGE[0], AFF_RANGE[1])),
    weight: clamp(num(src.weight, 0.5), 0.05, 1),
    color: clamp(num(src.color, 0.5), 0, 1),
    vars,
    spin: clampParam(num(src.spin, 0), C(LOCKS, 0)),
    bass: clamp(num(src.bass, 0), 0, 0.3),
    drift: [clamp(num(src.drift?.[0], 0), 0, XFORM_DRIFT), clamp(num(src.drift?.[1], 0), 0, XFORM_DRIFT)],
    pulse: clamp(num(src.pulse, 0), 0, 0.3),
  };
  if (alt) out.alt = alt;
  return out;
}

function repairOp(o: Partial<OpGene> | undefined): OpGene | null {
  if (!o || !OP_KINDS.includes(o.op as OpKind)) return null;
  const op = o.op as OpKind;
  const stage: Stage = stageFree(op) ? (o.stage === 'view' ? 'view' : 'warp') : 'warp';
  return { op, stage, w: clamp(num(o.w, 1), 0, 1), p: repairParams(o.p, OP_SCHEMAS[op]) };
}

function repairDrawOp(o: Partial<OpGene> | undefined): OpGene | null {
  if (!o || !isDrawOp(o.op as string)) return null;
  const op = o.op as OpKind;
  return { op, stage: 'warp', w: clamp(num(o.w, 1), 0, 1), p: repairParams(o.p, OP_SCHEMAS[op]) };
}

/** A merge whose parts break the rules degrades: parts swap roles, or it falls back to its first part. */
function repairMerge(e: Partial<EmitterGene>): EmitterGene | null {
  const parts = (Array.isArray(e.parts) ? e.parts : []).map((x) => (x && x.kind !== 'merge' ? repairEmitter(x) : null)).filter((x): x is EmitterGene => !!x);
  if (parts.length < 2 || parts[0].kind === parts[1].kind) return parts[0] ?? null;
  let [a, b] = parts;
  const p = repairParams(e.p, EMITTER_SCHEMAS.merge);
  if (p.mode !== 2 && !(isSdfKind(a.kind) && isSdfKind(b.kind))) p.mode = 2;
  if (p.mode === 2 && !isSdfKind(a.kind)) {
    if (!isSdfKind(b.kind)) return a;
    [a, b] = [b, a];
  }
  for (const x of p.mode === 2 ? [a] : [a, b]) {
    // Lissajous / pendulum curves have no distance field: a shaped wave becomes a circle.
    if (x.kind === 'wave' && (x.p.shape === 3 || x.p.shape === 5)) x.p.shape = 1;
  }
  a.layer = 'fb';
  b.layer = 'fb';
  const geo = GEOMETRY_KINDS.includes(b.kind);
  return { kind: 'merge', layer: geo ? 'fb' : e.layer === 'top' ? 'top' : 'fb', p, parts: [a, b] };
}

function repairEmitter(e: Partial<EmitterGene> | undefined): EmitterGene | null {
  if (!e || !EMITTER_KINDS.includes(e.kind as EmitterKind)) return null;
  const kind = e.kind as EmitterKind;
  if (kind === 'merge') return repairMerge(e);
  // Flames and particles need accumulation; field emitters may sit on either layer.
  const layer: Layer = kind === 'flame' ? 'fb' : e.layer === 'top' ? 'top' : 'fb';
  const out: EmitterGene = { kind, layer, p: repairParams(e.p, EMITTER_SCHEMAS[kind]) };
  if (kind === 'flame') {
    const xs = (Array.isArray(e.xforms) ? e.xforms : []).slice(0, MAX_XFORMS).map(repairXform);
    if (!xs.length) xs.push(repairXform({ vars: { linear: 0.5, spherical: 0.5 } }), repairXform({ aff: [0.5, 0, 0, 0.5, 0.5, 0], vars: { sinusoidal: 1 } }));
    out.xforms = xs;
  }
  return out;
}

/**
 * Returns a valid genome: every parameter clamped into its spec, unknown
 * fields dropped, caps enforced, dangling reactions retargeted or removed.
 * Idempotent: repair(repair(g)) deep-equals repair(g).
 */
export function repair(input: unknown): Genome {
  const g = (input && typeof input === 'object' ? input : {}) as Partial<Genome>;
  const chain = (Array.isArray(g.chain) ? g.chain : []).map(repairOp).filter((o): o is OpGene => !!o).slice(0, MAX_CHAIN);
  const draw = (Array.isArray(g.draw) ? g.draw : []).map(repairDrawOp).filter((o): o is OpGene => !!o).slice(0, MAX_DRAW);
  // Kinds are unique across emitters and merge parts (they share uniform arrays); one merge at most.
  const seen = new Set<EmitterKind>();
  const emitters: EmitterGene[] = [];
  let flat = 0;
  for (const e of Array.isArray(g.emitters) ? g.emitters : []) {
    let r = repairEmitter(e);
    if (r?.kind === 'merge' && (seen.has('merge') || r.parts!.some((x) => seen.has(x.kind)) || flat + 3 > MAX_FLAT)) {
      r = r.parts!.find((x) => !seen.has(x.kind)) ?? null;
    }
    if (!r || seen.has(r.kind) || flat + 1 > MAX_FLAT) continue;
    seen.add(r.kind);
    for (const x of r.parts ?? []) seen.add(x.kind);
    emitters.push(r);
    flat += 1 + (r.parts?.length ?? 0);
    if (emitters.length >= MAX_EMITTERS) break;
  }
  if (!emitters.length) emitters.push({ kind: 'wave', layer: 'fb', p: defaultParams(EMITTER_SCHEMAS.wave) });
  const ck = CARRIER_KINDS.includes(g.carrier?.kind as CarrierKind) ? (g.carrier!.kind as CarrierKind) : 'warp';
  const carrier: CarrierGene = { kind: ck, p: repairParams(g.carrier?.p, CARRIER_SCHEMA) };
  const scheme = SCHEMES.includes(g.color?.scheme as Scheme) ? (g.color!.scheme as Scheme) : 'analogous';
  const color: ColorGene = { scheme, p: repairParams(g.color?.p, COLOR_SCHEMA) };

  let lo = clamp(num(g.energy?.[0], 0.2), 0, 1);
  let hi = clamp(num(g.energy?.[1], 0.7), 0, 1);
  if (hi < lo) [lo, hi] = [hi, lo];
  if (hi - lo < 0.15) {
    const c = clamp((lo + hi) / 2, 0.075, 0.925);
    lo = c - 0.075;
    hi = c + 0.075;
  }
  // draw only appears when non-empty, so format-1 genomes repair to the same JSON (bar the version).
  const out: Genome = draw.length
    ? { v: 2, chain, draw, emitters, carrier, color, reactions: [], energy: [round4(lo), round4(hi)] }
    : { v: 2, chain, emitters, carrier, color, reactions: [], energy: [round4(lo), round4(hi)] };
  const flatLen = flatEmitters(out).length;

  for (const r of Array.isArray(g.reactions) ? g.reactions : []) {
    if (out.reactions.length >= MAX_REACTIONS) break;
    if (!r || !SIGNALS.includes(r.src as Signal)) continue;
    const group = GENE_GROUPS.includes(r.g) ? r.g : null;
    if (!group) continue;
    const len = group === 'op' ? chain.length : group === 'em' ? flatLen : group === 'dr' ? draw.length : 1;
    if (!len) continue;
    const i = group === 'op' || group === 'em' || group === 'dr' ? ((Math.floor(num(r.i, 0)) % len) + len) % len : 0;
    const keys = reactable(schemaFor(out, group, i)!);
    if (!keys.length) continue;
    const k = keys.includes(r.k) ? r.k : keys.includes('gain') ? 'gain' : keys[0];
    const gain = clamp(num(r.gain, 0.3), -1, 1);
    if (out.reactions.some((q) => q.g === group && q.i === i && q.k === k && q.src === r.src)) continue;
    out.reactions.push({ src: r.src as Signal, g: group, i, k, gain });
  }
  return out;
}

function round4(x: number): number {
  return Math.round(x * 1e4) / 1e4;
}

// ------------------------------------------------------- format 2 -> 3

type Raw = Record<string, unknown>;
interface RawBody {
  shape: Raw;
  place: Raw;
  motion: Raw;
  deform: Raw;
  material: Raw;
  emit: Raw;
  fuse?: Raw;
}
/** Where an old emitter parameter lives now: [body group, new key] (null: dropped). */
type ParamMap = Record<string, [string, string] | null>;

const LOCKS4 = [-1, -0.5, -0.25, -0.125, -0.0625, 0, 0.0625, 0.125, 0.25, 0.5, 1];
const nearest = (v: number, xs: number[]) => xs.reduce((a, b) => (Math.abs(b - v) < Math.abs(a - v) ? b : a), xs[0]);

function layerEmit(e: EmitterGene): Raw {
  return e.layer === 'top' ? { kind: 'none', p: {} } : { kind: 'trail', p: {} };
}

/** One old emitter as a body plus the map of its old parameter names. */
function convEmitter(e: EmitterGene, fluid: boolean): { body: RawBody; map: ParamMap } {
  const p = e.p;
  const mat = (kind: string, extra: Raw = {}): Raw => ({ kind, p: { gain: p.gain, hue: p.hue, ...extra } });
  const none: Raw = { kind: 'none', p: {} };
  const common: ParamMap = { gain: ['ma', 'gain'], hue: ['ma', 'hue'] };
  const b = (shape: Raw, place: Raw, material: Raw, emit: Raw, motion: Raw = none, deform: Raw = none): RawBody => ({ shape, place, motion, deform, material, emit });
  switch (e.kind) {
    case 'wave':
      return {
        body: b(
          { kind: 'curve', p: { form: p.shape, amp: p.amp, radius: p.radius, turns: p.turns, ra: p.ra, rb: p.rb } },
          { kind: 'point', p: { x: p.x, y: p.y } },
          mat('line', { width: p.thick, halo: 0.12 }),
          layerEmit(e),
          p.shape === 5 ? none : { kind: 'spin', p: { rate: 0.0625, alt: 0 } },
        ),
        map: { ...common, amp: ['sh', 'amp'], radius: ['sh', 'radius'], turns: ['sh', 'turns'], x: ['pl', 'x'], y: ['pl', 'y'], thick: ['ma', 'width'] },
      };
    case 'spectrum':
      return {
        body: b(
          { kind: 'bars', p: { mode: p.mode, bins: p.bins, radius: p.radius, len: p.len, fill: p.fill } },
          { kind: 'point', p: { x: p.x, y: p.y } },
          mat('fill', { soft: 0 }),
          layerEmit(e),
        ),
        map: { ...common, radius: ['sh', 'radius'], len: ['sh', 'len'], fill: ['sh', 'fill'], x: ['pl', 'x'], y: ['pl', 'y'] },
      };
    case 'particles': {
      // Spawn modes become placements: 0 anywhere (grid), 1 waveform (an unseen curve), 2 rotating
      // emitters (orbit), 3 ring, 4 centre, 5 bottom edge (row).
      const place: Raw[] = [
        { kind: 'grid', p: { density: 0.5 } },
        { kind: 'point', p: {} },
        { kind: 'orbit', p: { count: 3, radius: 0.3 } },
        { kind: 'ring', p: { n: 6, radius: 0.3 } },
        { kind: 'point', p: {} },
        { kind: 'row', p: { count: 4 } },
      ];
      const shape: Raw = p.spawn === 1 ? { kind: 'curve', p: { form: 0 } } : { kind: 'dot', p: { r: 0.004 } };
      return {
        body: b(shape, place[p.spawn] ?? place[4], mat(p.spawn === 1 ? 'line' : 'glow'), {
          kind: 'sparks',
          p: { count: p.count, size: p.size, speed: p.speed, curl: p.curl, zoomFlow: p.zoomFlow, lift: p.lift, drag: p.drag, life: p.life, spread: p.spread, surge: p.surge, top: e.layer === 'top' ? 1 : 0, body: 0 },
        }),
        map: { ...common, size: ['em', 'size'], speed: ['em', 'speed'], curl: ['em', 'curl'], zoomFlow: ['em', 'zoomFlow'], lift: ['em', 'lift'], drag: ['em', 'drag'], life: ['em', 'life'], spread: ['em', 'spread'], surge: ['em', 'surge'] },
      };
    }
    case 'stars':
      return {
        body: b(
          { kind: 'dot', p: { r: 0 } },
          { kind: 'grid', p: { lattice: 0, scale: p.scale, jitter: 0.6, density: p.density, lit: 1, links: p.links, twinkle: p.twinkle, lock: 0 } },
          { kind: 'glow', p: { gain: p.gain * 0.32, hue: p.hue, width: 0.0014, base: 0.25, halo: 0.08 } },
          layerEmit(e),
          p.drift > 0.0005 ? { kind: 'drift', p: { vx: Math.min(0.1, (p.drift * 6) / p.scale), vy: 0 } } : none,
        ),
        map: { ...common, density: ['pl', 'density'], scale: ['pl', 'scale'], links: ['pl', 'links'], twinkle: ['pl', 'twinkle'], drift: ['mo', 'vx'] },
      };
    case 'ink': {
      let place: Raw;
      if (p.orbit > 0.5) place = { kind: 'orbit', p: { count: p.count, radius: p.radius, follow: p.follow, rate: 0.5 } };
      else if (p.row > 0.5) place = { kind: 'row', p: { count: p.count, wander: p.wander } };
      else place = { kind: 'stations', p: { count: p.count, inst: p.inst, xs: p.xs, jump: p.jump, wander: p.wander, swap: p.swap } };
      const emit = fluid && p.force > 0 ? { kind: 'dye', p: { force: p.force } } : layerEmit(e);
      return {
        body: b({ kind: 'dot', p: { r: 0 } }, place, mat('glow', { width: p.size, base: 0 }), emit),
        map: { ...common, radius: ['pl', 'radius'], size: ['ma', 'width'], wander: ['pl', 'wander'], force: ['em', 'force'], inst: ['pl', 'inst'], xs: ['pl', 'xs'], follow: ['pl', 'follow'], orbit: null, row: null },
      };
    }
    case 'wire':
      return {
        body: b(
          { kind: 'solid', p: { solid: p.solid, sides: p.sides, size: p.scale, tilt: p.tilt, inner: p.inner } },
          { kind: 'point', p: {} },
          mat('line', { width: p.thick, halo: 0.12 }),
          layerEmit(e),
          p.lock !== 0 ? { kind: 'spin', p: { rate: nearest(p.lock * 4, LOCKS4), alt: 0 } } : none,
        ),
        map: { ...common, scale: ['sh', 'size'], tilt: ['sh', 'tilt'], inner: ['sh', 'inner'], thick: ['ma', 'width'] },
      };
    case 'plasma':
      return {
        body: b({ kind: 'plasma', p: { ...p } }, { kind: 'point', p: {} }, mat('fill'), layerEmit(e)),
        map: { ...common, scale: ['sh', 'scale'], warp: ['sh', 'warp'], bands: ['sh', 'bands'], lines: ['sh', 'lines'], speed: ['sh', 'speed'], tempo: ['sh', 'tempo'], pulse: ['sh', 'pulse'], melHue: ['sh', 'melHue'] },
      };
    case 'aurora':
      return {
        body: b({ kind: 'aurora', p: { fall: p.fall, rays: p.rays, wav: p.wav } }, { kind: 'point', p: { y: p.y } }, mat('glow'), layerEmit(e)),
        map: { ...common, y: ['pl', 'y'], fall: ['sh', 'fall'], rays: ['sh', 'rays'], wav: ['sh', 'wav'] },
      };
    case 'blobs':
      return {
        body: b(
          { kind: 'dot', p: { r: p.size } },
          { kind: 'float', p: { count: p.count, spread: p.spread, speed: p.speed, fuse: 0.1 } },
          mat('chrome', { chrome: p.chrome }),
          layerEmit(e),
        ),
        map: { ...common, size: ['sh', 'r'], spread: ['pl', 'spread'], speed: ['pl', 'speed'], chrome: ['ma', 'chrome'] },
      };
    case 'flame':
      return {
        body: b(
          { kind: 'flame', p: { count: p.count, zoom: p.zoom, rounds: p.rounds, flow: p.flow, breathe: p.breathe }, xforms: e.xforms },
          { kind: 'point', p: { x: p.ox, y: p.oy } },
          mat('glow'),
          { kind: 'trail', p: {} },
          p.camSpin !== 0 ? { kind: 'spin', p: { rate: p.camSpin, alt: 0 } } : none,
        ),
        map: { ...common, zoom: ['sh', 'zoom'], flow: ['sh', 'flow'], breathe: ['sh', 'breathe'], ox: ['pl', 'x'], oy: ['pl', 'y'] },
      };
    case 'edge':
      return {
        body: b({ kind: 'edge', p: { mode: p.mode, side: p.side, base: p.base, height: p.height, density: p.density } }, { kind: 'point', p: {} }, mat('fill'), layerEmit(e)),
        map: { ...common, base: ['sh', 'base'], height: ['sh', 'height'], density: ['sh', 'density'] },
      };
    case 'tiles':
      return {
        body: b(
          { kind: 'polygon', p: { n: [6, 4, 3][p.shape] ?? 6, r: 0.43 / p.scale, round: 0 } },
          { kind: 'grid', p: { lattice: p.shape === 0 ? 1 : p.shape === 1 ? 0 : 2, scale: p.scale, jitter: 0, density: 1, lit: p.lit, links: 0, twinkle: 0, lock: p.lock } },
          mat('fill', { soft: 0, outline: p.edges, core: 1 }),
          layerEmit(e),
        ),
        map: { ...common, scale: ['pl', 'scale'], lit: ['pl', 'lit'], edges: ['ma', 'outline'] },
      };
    case 'horizon':
      return {
        body: b({ kind: 'terrain', p: { speed: p.speed, density: p.density, peaks: p.peaks, terrain: p.terrain, flash: p.flash } }, { kind: 'point', p: { y: p.y } }, mat('fill'), layerEmit(e)),
        map: { ...common, y: ['pl', 'y'], speed: ['sh', 'speed'], density: ['sh', 'density'], peaks: ['sh', 'peaks'], terrain: ['sh', 'terrain'], flash: ['sh', 'flash'] },
      };
    case 'orb': {
      const stripes = p.stripes > 0.5;
      return {
        body: b(
          { kind: 'dot', p: { r: p.radius } },
          { kind: 'point', p: { x: p.x, y: p.y } },
          mat('textured', { tex: stripes ? 1 : 0, amount: stripes ? p.stripes : p.craters, halo: p.halo, clip: p.clip }),
          layerEmit(e),
          p.bob > 0.001 ? { kind: 'bob', p: { amp: p.bob } } : none,
          p.arms > 0.001 ? { kind: 'arms', p: { reach: p.arms } } : none,
        ),
        map: { ...common, x: ['pl', 'x'], y: ['pl', 'y'], radius: ['sh', 'r'], halo: ['ma', 'halo'], craters: ['ma', 'amount'], stripes: ['ma', 'amount'], arms: ['de', 'reach'], bob: ['mo', 'amp'], clip: ['ma', 'clip'] },
      };
    }
    case 'snake':
      return {
        body: b(
          { kind: 'dot', p: { r: 0.01 * p.width } },
          { kind: 'walker', p: { heads: p.count, step: p.step, every: 2, turn: p.turn, curve: p.curve } },
          mat('fill', { soft: 0.6 }),
          { kind: 'cover', p: { amt: p.cover, tip: 1 } },
          { kind: 'hits', p: { amt: 1 } },
        ),
        map: { ...common, step: ['pl', 'step'], turn: ['pl', 'turn'], curve: ['pl', 'curve'], width: ['sh', 'r'], cover: ['em', 'amt'] },
      };
    default:
      return { body: b({ kind: 'curve', p: {} }, { kind: 'point', p: {} }, mat('line'), layerEmit(e)), map: common };
  }
}

/**
 * Re-expresses a repaired format-2 genome in the format-3 body vocabulary (as a raw object for the
 * format-3 repair to finish). Each emitter becomes one body; a merge becomes its drawing body with
 * the other part fused in; the old genome-wide draw chain becomes every body's deform ops; reactions
 * follow their parameter to its new locus (unmappable ones are dropped).
 */
export function upgradeV2(g: Genome): Raw {
  const fluid = g.carrier.kind === 'fluid';
  const bodies: RawBody[] = [];
  const flatMap: ({ body: number; group: 'sh' | 'fs'; map: ParamMap } | null)[] = [];
  for (const e of g.emitters) {
    const bi = bodies.length;
    if (e.kind === 'merge' && e.parts) {
      const [a, b2] = e.parts;
      const mode = e.p.mode;
      const ca = convEmitter(a, fluid);
      const cb = convEmitter(b2, fluid);
      const [prim, other] = mode === 2 ? [cb, ca] : [ca, cb];
      const body = prim.body;
      if (mode !== 2) body.material = e.p.fill > 0.4 ? { kind: 'fill', p: { gain: e.p.gain, hue: e.p.hue, soft: 0.3, halo: e.p.line * 0.5 } } : { kind: 'line', p: { gain: e.p.gain, hue: e.p.hue, width: e.p.width * 1.2, halo: 0.18 } };
      body.fuse = { shape: other.body.shape, p: { mode, k: e.p.k, t: e.p.t, drive: e.p.drive, depth: e.p.depth, rate: e.p.rate, inside: e.p.inside } };
      if (e.layer === 'top' && body.emit.kind !== 'sparks') body.emit = { kind: 'none', p: {} };
      bodies.push(body);
      const mergeMap: ParamMap = { gain: ['ma', 'gain'], hue: ['ma', 'hue'], k: ['fu', 'k'], t: ['fu', 't'], depth: ['fu', 'depth'], width: ['ma', 'width'] };
      flatMap.push({ body: bi, group: 'sh', map: mergeMap });
      const shapeOnly = (m: ParamMap): ParamMap => Object.fromEntries(Object.entries(m).filter(([, v]) => v && v[0] === 'sh'));
      const aMap = mode === 2 ? { map: shapeOnly(ca.map), group: 'fs' as const } : { map: ca.map, group: 'sh' as const };
      const bMap = mode === 2 ? { map: cb.map, group: 'sh' as const } : { map: shapeOnly(cb.map), group: 'fs' as const };
      flatMap.push({ body: bi, ...aMap }, { body: bi, ...bMap });
    } else {
      const c = convEmitter(e, fluid);
      bodies.push(c.body);
      flatMap.push({ body: bi, group: 'sh', map: c.map });
    }
  }
  if (g.draw?.length) for (const b of bodies) b.deform.ops = g.draw.map((o) => ({ ...o, p: { ...o.p } }));
  const reactions: Raw[] = [];
  for (const r of g.reactions) {
    if (r.g === 'op' || r.g === 'car' || r.g === 'col') reactions.push({ ...r });
    else if (r.g === 'dr') reactions.push({ ...r, g: 'dr', i: r.i });
    else if (r.g === 'em') {
      const f = flatMap[r.i];
      const to = f?.map[r.k];
      if (!f || !to) continue;
      // A fused part's own shape keys go to the fused-shape locus.
      const group = to[0] === 'sh' && f.group === 'fs' ? 'fs' : to[0];
      reactions.push({ src: r.src, g: group, i: f.body, k: to[1], gain: r.gain });
    }
  }
  return { v: 3, chain: g.chain, bodies, carrier: g.carrier, color: g.color, reactions, energy: g.energy };
}
