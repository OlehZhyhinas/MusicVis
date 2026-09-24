// V2 genome, format 3: the unit of inheritance is an idea, not a whole object.
//
//   chain      ordered space transforms of the carrier (vec2 -> vec2). Motion
//              ops run in the feedback warp; fold ops and flame variations can
//              run in the warp or at display time ('view' stage).
//   bodies     1-2 light sources (3 only in converted old data). Each body is a
//              set of independent sub-genes, one per locus:
//                shape     what is drawn (dot, polygon, star, segment, wireframe
//                          solid, spectrum bars, a curve, or a larger chunk:
//                          plasma, aurora, terrain, edge strips, fractal flame)
//                place     where copies sit (a point, an orbit, a wandering
//                          walker, instrument stations, a row, floating copies,
//                          a path, or a grid / ring / mirror fold)
//                motion    how each copy moves (spin, sway, bob, drift, circle,
//                          turn on hits, pulse)
//                deform    how the shape is bent (arms, wobble lobes, noise,
//                          twist) plus optional draw-space ops
//                material  how it is lit (line, fill, glow, dots, textured, chrome)
//                emit      what it leaves behind (nothing, a trail, paint over
//                          older trail, dye into the fluid, sparks)
//              A body may also fuse a second shape into its own (smooth union,
//              music-driven morph, or a region the body lights up inside).
//   carrier    how existing light moves: feedback warp, fluid, flow field, none.
//   color      palette around the key hue plus post settings.
//   reactions  music signal -> parameter, with a gain.
//
// Every numeric parameter has a spec (range, integer, discrete choices) so
// mutation and crossover always stay in range; repair() enforces all of it and
// also lifts older formats (see legacy.ts).

import { FLAME_VARIATIONS, type FlameVar } from './variations';
import { repair as repairV2, upgradeV2 } from './legacy';
import { SUPERSCOPE_COST, SUPERSCOPE_SCHEMA } from './genes/superscope';
import { SLIME_SCHEMA, slimeCost } from './genes/physarum';
import { BEAMS_SCHEMA, beamsCost } from './genes/beams';
import { WATER_COST, WATER_PARAMS } from './genes/water';

import { CHOREO_COST_MS, repairChoreo, validateChoreo, type ChoreoGene } from './genes/choreo';
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
/** Bar-locked turn rates, turns per bar. */
export const TURNS = [-1, -0.5, -0.25, -0.125, -0.0625, 0, 0.0625, 0.125, 0.25, 0.5, 1];

// ------------------------------------------------------------------ chain

export const MOTION_OPS = ['zoom', 'rotate', 'translate', 'swirl', 'twist', 'ripple', 'noise', 'push', 'quad'] as const;
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
  rotate: { lock: C(LOCKS, 0.0625), rate: P(-0.01, 0.01, 0), cx: P(-0.4, 0.4, 0), cy: P(-0.4, 0.4, 0), alt: C([0, 1], 0), wander: P(0, 0.35, 0) },
  // content velocity, p units per second; lanes > 1: columns per unit moving at 1x or 2x.
  translate: { vx: P(-0.5, 0.5, 0), vy: P(-0.5, 0.5, 0), lanes: P(0, 50, 0) },
  swirl: { amt: P(-0.03, 0.03, 0.01), k: P(1, 12, 6), cx: P(-0.4, 0.4, 0), cy: P(-0.4, 0.4, 0), wander: P(0, 0.35, 0) },
  twist: { amt: P(-0.012, 0.012, 0.003), cx: P(-0.4, 0.4, 0), cy: P(-0.4, 0.4, 0) },
  ripple: { amp: P(0, 0.004, 0.0008), freq: P(2, 40, 8), speed: P(0.2, 4, 0.7), radial: C([0, 1], 0) },
  noise: { amp: P(0, 0.003, 0.0012), scale: P(0.8, 5, 2), speed: P(0.05, 1, 0.3) },
  push: { amt: P(-0.01, 0.01, 0.001), axis: C([0, 1, 2], 0) },
  // Quadratic flow: each frame samples from p + amt * turn(z^2) around the (wandering) centre, so the
  // feedback settles into a Julia-set coastline (the classic z-squared per-pixel warp).
  quad: { amt: P(0, 2.5, 1), turn: C([0, 0.25, 0.5, 0.75], 0.75), cx: P(-0.4, 0.4, 0), cy: P(-0.4, 0.4, 0), wander: P(0, 0.35, 0) },
  mirror: { axis: C([0, 1, 2], 0) },
  tile: { n: P(1.2, 4, 2) },
  polar: { scale: P(0.4, 1.6, 1), lock: C(LOCKS, 0) },
  kaleido: { n: I(3, 12, 6), lock: C(LOCKS, 0.0625) },
  // Spectrum stretch of the displayed picture (view) or a slow pull up loud strips (warp).
  stretch: { base: P(-0.4, 0.2, -0.16), amt: P(0, 1.5, 0.9), beat: P(0, 0.4, 0.18), strips: I(8, 64, 32), win: P(0, 1, 1), sky: P(0, 1, 0) },
};
for (const v of VAR_OPS) OP_SCHEMAS[v] = VAR_SCHEMA;

/** Ops allowed as draw-space shaping (a body's deform ops); rates become absolute amounts. */
export const DRAW_OPS: OpKind[] = ['swirl', 'twist', 'ripple', 'noise', 'rotate', 'zoom', 'mirror', 'kaleido', ...VAR_OPS];
export const MAX_DRAW = 3;
/** GLSL type id of a draw op; variations are 20 + their index. */
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

// ---------------------------------------------------------------- shapes

export const SHAPE_KINDS = ['dot', 'polygon', 'star', 'segment', 'solid', 'bars', 'curve', 'plasma', 'aurora', 'terrain', 'edge', 'flame', 'superscope', 'beams'] as const;
export type ShapeKind = (typeof SHAPE_KINDS)[number];
/**
 * sdf: a distance field in the body's local space (every material and placement applies).
 * curve: line geometry (a distance field too for forms 0, 1, 2, 4, used when fused).
 * field: a full-screen chunk with its own look (materials only set its gain, colour mapping its hue).
 * flame: the fractal-flame IFS (point geometry; placed by its camera offset).
 */
export type ShapeClass = 'sdf' | 'curve' | 'field' | 'flame';
export const SHAPE_CLASS: Record<ShapeKind, ShapeClass> = {
  dot: 'sdf', polygon: 'sdf', star: 'sdf', segment: 'sdf', solid: 'sdf', bars: 'sdf',
  curve: 'curve', plasma: 'field', aurora: 'field', terrain: 'field', edge: 'field', flame: 'flame',
  superscope: 'curve',
  beams: 'field',
};
/** Shapes the shared GPU state allows once per genome (wireframe segments, the flame sim). */
export const UNIQUE_SHAPES: ShapeKind[] = ['solid', 'flame'];

export const SHAPE_SCHEMAS: Record<ShapeKind, Schema> = {
  dot: { r: P(0, 0.3, 0.02) },
  polygon: { n: I(3, 8, 6), r: P(0.005, 0.45, 0.12), round: P(0, 1, 0) },
  star: { n: I(4, 9, 5), r: P(0.01, 0.45, 0.12), inner: P(0.2, 0.8, 0.45) },
  segment: { len: P(0.01, 0.8, 0.2), w: P(0.001, 0.05, 0.006) },
  // solid: 0 tetra, 1 cube, 2 octa, 3 icosa, 4 polygon (sides), 5 follows the song sections.
  solid: { solid: C([0, 1, 2, 3, 4, 5], 5), sides: I(3, 8, 6), size: P(0.06, 0.35, 0.2), tilt: P(0, 1, 0.45), inner: P(0, 1, 0.6) },
  // mode: 0 bars on a baseline, 1 upper arc, 2 full ring, 3 mirrored bars.
  bars: { mode: C([0, 1, 2, 3], 1), bins: I(12, 64, 40), radius: P(0.08, 0.35, 0.2), len: P(0.1, 0.5, 0.3), fill: P(0.3, 0.9, 0.6) },
  // form: 0 waveform line, 1 circle, 2 spiral, 3 lissajous, 4 arc, 5 pendulum harmonograph
  // (ratios walk the circle of fifths every rb bars, offset ra; phases advance with the beat).
  curve: { form: C([0, 1, 2, 3, 4, 5], 0), amp: P(0.05, 0.5, 0.24), radius: P(0.08, 0.45, 0.27), turns: P(1, 6, 3), ra: I(1, 6, 2), rb: I(1, 6, 3) },
  plasma: { scale: P(0.8, 3, 1.7), warp: P(0.5, 3, 2), bands: P(3, 12, 8), lines: P(0, 1, 0.8), speed: P(0.05, 0.4, 0.15), tempo: P(0, 1, 0), pulse: P(0, 1, 0), melHue: P(0, 1, 0) },
  aurora: { fall: P(2, 10, 5), rays: P(8, 40, 24), wav: P(0.5, 3, 1.4) },
  // terrain: ray-marched hills either side of a valley following the spectrum; flash: downbeat flash.
  terrain: { speed: P(0.2, 1, 0.5), density: P(0.5, 3, 1.5), peaks: P(0, 1, 1), terrain: P(0, 1, 0), flash: P(0, 1, 0) },
  // mode: 0 skyline, 1 melody ribbon, 2 spectral rain, 3 terrain ridge. side: 0 right, 1 top, 2 bottom, 3 left.
  edge: { mode: C([0, 1, 2, 3], 0), side: C([0, 1, 2, 3], 0), base: P(-0.3, 0.3, -0.16), height: P(0.1, 0.5, 0.3), density: P(0.2, 1, 0.6) },
  // flow: vars <-> alt morph cycles per 8 bars (0 = only drops morph); breathe: zoom per unit bass.
  flame: { count: C([65536, 131072, 262144, 524288], 262144), zoom: P(0.1, 0.45, 0.22), rounds: I(1, 2, 2), flow: P(0, 2, 0), breathe: P(0, 0.4, 0) },
  // AVS superscope: a 3D parametric point curve pushed by the audio, tumbling in perspective (genes/superscope.ts).
  superscope: SUPERSCOPE_SCHEMA,
  // Volumetric concert beams through haze (see genes/beams.ts).
  beams: BEAMS_SCHEMA,
};

/** True when this shape has a distance field (it can be fused, painted over, masked by). */
export function sdfCapable(s: Pick<ShapeGene, 'kind' | 'p'>): boolean {
  const c = SHAPE_CLASS[s.kind];
  if (c === 'sdf') return true;
  if (s.kind === 'curve') return s.p.form !== 3 && s.p.form !== 5;
  return s.kind === 'aurora';
}

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

// ------------------------------------------------------------ placement

export const PLACE_KINDS = ['point', 'orbit', 'walker', 'stations', 'row', 'float', 'outline', 'grid', 'ring', 'mirror'] as const;
export type PlaceKind = (typeof PLACE_KINDS)[number];
/** Folds: the shader repeats the shape by folding space (one evaluation); the rest are explicit copies. */
export const FOLD_PLACES: PlaceKind[] = ['grid', 'ring', 'mirror'];
export const isFoldPlace = (k: PlaceKind) => FOLD_PLACES.includes(k);
export const MAX_COPIES = 6;

export const PLACE_SCHEMAS: Record<PlaceKind, Schema> = {
  point: { x: P(-0.6, 0.6, 0), y: P(-0.45, 0.45, 0) },
  // Copies circling a centre (bar-locked, rate turns per bar); follow: the centre wanders. fuse: copies melt together.
  orbit: { count: I(1, 6, 2), radius: P(0.05, 0.45, 0.3), x: P(-0.4, 0.4, 0), y: P(-0.4, 0.4, 0), follow: P(0, 0.35, 0), rate: C(TURNS, 0.5), fuse: P(0, 0.2, 0) },
  // 1-2 heads roaming the screen, moving step units per beat: head 0 turns every `every` beats, head 1 every
  // 2 x `every`; square: 90 degree turns; wrap: wrap around the edges instead of bouncing; curve: steering.
  walker: { heads: I(1, 2, 2), step: P(0.03, 0.2, 0.11), every: C([1, 2, 4, 8], 2), square: C([0, 1], 0), wrap: C([0, 1], 0), curve: P(0, 2, 0.8), turn: P(0, 1.5, 1) },
  // One copy per instrument (drums, bass, vocals, other): inst 1 moves them with their instruments (drums jump
  // every bar, or on every hit with jump 1; bass swings out; vocals follow the melody). xs: sideways roam.
  stations: { count: I(1, 6, 4), inst: P(0, 1, 1), xs: P(0.02, 1, 1), jump: C([0, 1], 0), wander: P(0, 0.2, 0.1), swap: C([0, 1], 0), fuse: P(0, 0.2, 0) },
  // Copies in a row (smoke sources along the bottom).
  row: { count: I(1, 6, 4), y: P(-0.5, 0.3, -0.5), wander: P(0, 0.2, 0.05) },
  // Copies floating on slow Lissajous paths within `spread`; sizes follow their instruments.
  float: { count: I(1, 6, 6), spread: P(0.3, 0.7, 0.55), speed: P(0.05, 0.3, 0.12), fuse: P(0, 0.3, 0) },
  // Copies travelling along a path: 0 circle, 1 polygon corners, 2 figure eight.
  outline: { count: I(1, 6, 5), path: C([0, 1, 2], 0), radius: P(0.08, 0.45, 0.25), rate: C(TURNS, 0.125), x: P(-0.4, 0.4, 0), y: P(-0.4, 0.4, 0) },
  // Lattice fold: 0 square, 1 hex, 2 triangle; scale: cells per unit; density: fraction of cells used;
  // lit < 1: cells light on a beat-random schedule; links: lines to lit neighbours; lock: grid turns per bar.
  grid: { lattice: C([0, 1, 2], 0), scale: P(2, 14, 6.5), jitter: P(0, 0.6, 0.6), density: P(0.1, 1, 0.5), lit: P(0.1, 1, 1), links: P(0, 1, 0), twinkle: P(0, 1, 0.3), lock: C(TURNS, 0) },
  ring: { n: I(2, 12, 6), radius: P(0, 0.45, 0.25), x: P(-0.4, 0.4, 0), y: P(-0.4, 0.4, 0) },
  // Mirror fold across the vertical (0), horizontal (1) or both (2) axes; the copy sits at (x, y).
  mirror: { axis: C([0, 1, 2], 0), x: P(0, 0.6, 0.2), y: P(-0.45, 0.45, 0) },
};

// ---------------------------------------------------------------- motion

export const MOTION_KINDS = ['none', 'spin', 'sway', 'bob', 'drift', 'circle', 'hits', 'pulse'] as const;
export type MotionKind = (typeof MOTION_KINDS)[number];
export const MOTION_SCHEMAS: Record<MotionKind, Schema> = {
  none: {},
  // Bar-locked spin (turns per bar); alt: reverses every other bar. Solids turn in 3D.
  spin: { rate: C(TURNS, 0.25), alt: C([0, 1], 0) },
  // Sideways swing with a period in bars, tilting a little.
  sway: { amp: P(0, 0.15, 0.04), period: C([1, 2, 4, 8], 2), tilt: P(0, 1, 0.3) },
  // Drift once per bar and bob on the beat.
  bob: { amp: P(0, 2, 1) },
  // Slow drift (units per second), wrapping around the screen.
  drift: { vx: P(-0.1, 0.1, 0.005), vy: P(-0.1, 0.1, 0) },
  // A small circle around the placement, period in bars.
  circle: { radius: P(0, 0.12, 0.03), period: C([1, 2, 4, 8, 16], 4) },
  // Turn on drum hits (walker heads: a sharp turn; other copies: a jolt that stays).
  hits: { amt: P(0, 1.5, 1) },
  // Size kick on the beat.
  pulse: { amp: P(0, 0.5, 0.15) },
};

// ---------------------------------------------------------------- deform

export const DEFORM_KINDS = ['none', 'arms', 'wobble', 'noise', 'twist'] as const;
export type DeformKind = (typeof DEFORM_KINDS)[number];
export const DEFORM_SCHEMAS: Record<DeformKind, Schema> = {
  none: {},
  // Tapered curling arms that reach with drums, bass, vocals, other and loudness plus a breath of their
  // own; sway every 2 bars, turn once per `turn` bars.
  arms: { count: I(3, 7, 5), reach: P(0, 1, 1), width: P(0.15, 0.5, 0.29), curl: P(0, 1, 1), sway: P(0, 1, 1), turn: C([8, 16, 32], 16) },
  // Wobbling lobes around the outline, turning at `rate` turns per bar, swelling with the bass.
  wobble: { lobes: I(2, 9, 5), amp: P(0, 0.4, 0.12), rate: C(TURNS, 0.125) },
  noise: { amp: P(0, 0.08, 0.02), scale: P(1, 8, 3), speed: P(0.05, 1, 0.3) },
  // Rotation growing with the distance from the centre (radians per unit), breathing with loudness.
  twist: { amt: P(-8, 8, 3) },
};

// -------------------------------------------------------------- material

export const MATERIAL_KINDS = ['line', 'fill', 'glow', 'dots', 'textured', 'chrome'] as const;
export type MaterialKind = (typeof MATERIAL_KINDS)[number];
const MAT_COMMON: Schema = { gain: P(0.02, 3, 1) };
export const MATERIAL_SCHEMAS: Record<MaterialKind, Schema> = {
  // Thin glowing outline (width in pixels at 1080p), brighter with loudness.
  line: { ...MAT_COMMON, width: P(0.8, 3, 1.3), halo: P(0, 1, 0.12) },
  // Solid body: soft edge, outer halo, cell outline, core gradient, clip height (hidden below it).
  fill: { ...MAT_COMMON, soft: P(0, 1, 0.3), halo: P(0, 1, 0), outline: P(0, 1, 0), core: P(0, 1, 0), clip: P(-0.6, 0.35, -0.6) },
  // Gaussian glow (width in scene units) growing with the copy's level; base: brightness at silence.
  glow: { ...MAT_COMMON, width: P(0.001, 0.06, 0.02), base: P(0, 1, 0), halo: P(0, 1, 0) },
  // Stippled: dots filling the body.
  dots: { ...MAT_COMMON, spacing: P(0.006, 0.05, 0.016), size: P(0.2, 0.9, 0.45) },
  // tex: 0 craters (moon), 1 stripes with a warm gradient (setting sun), 2 lit windows.
  textured: { ...MAT_COMMON, tex: C([0, 1, 2], 0), amount: P(0, 1, 1), halo: P(0, 1, 0.5), clip: P(-0.6, 0.35, -0.6) },
  // Mirror-like surface reflecting the palette; chrome 0 = soft plastic.
  chrome: { ...MAT_COMMON, chrome: P(0, 1, 1) },
};
/** Materials whose brightness is a steady level (the rest accumulate into the trail). */
export const STATIC_MATERIALS: MaterialKind[] = ['fill', 'textured', 'chrome'];

// -------------------------------------------------------------- emission

export const EMIT_KINDS = ['none', 'trail', 'cover', 'dye', 'sparks', 'slime'] as const;
export type EmitKind = (typeof EMIT_KINDS)[number];
export const EMIT_SCHEMAS: Record<EmitKind, Schema> = {
  // Redrawn every frame on top of the picture.
  none: {},
  // Drawn into the feedback so it leaves a trail; tip: a bright point where each copy is now.
  trail: { tip: P(0, 1, 0) },
  // The newest stroke paints over older trail (amt), instead of adding light.
  cover: { amt: P(0, 1, 1), tip: P(0, 1, 1) },
  // Drawn into the feedback and pushed into the fluid (force) on hits and beats.
  dye: { force: P(0, 1.5, 1) },
  // Throws GPU particles from the copies; body: how visible the body itself stays; top: sparks drawn on top.
  sparks: {
    count: P(1024, 65536, 16384, { log: true, int: true }), size: P(1.5, 6, 2.6), speed: P(0.05, 1, 0.3), curl: P(0, 0.5, 0.05),
    zoomFlow: P(-0.5, 1.5, 0.5), lift: P(-0.2, 0.2, 0), drag: P(1, 5, 2.5), life: P(0.1, 0.8, 0.3), spread: P(0.01, 0.2, 0.05),
    surge: P(0, 1, 0), top: C([0, 1], 0), body: P(0, 1, 1),
  },
  // Physarum agents growing vein networks out of a trail the body seeds (genes/physarum.ts).
  slime: SLIME_SCHEMA,
};

// ------------------------------------------------------------------ fuse

/**
 * A second shape fused into the body. mode 0: smooth union (k = blend radius, grows with the bass);
 * 1: morph mix(body, other, t); 2: the body only lights up inside (inside 1) or along (0) the other shape.
 * t drifts toward a music driver by depth: drive 0 none, 1 bar-locked sweep every rate bars, 2 bass,
 * 3 melody, 4 loudness, 5 beat surge.
 */
export const FUSE_SCHEMA: Schema = {
  mode: C([0, 1, 2], 0), k: P(0.02, 0.25, 0.08), t: P(0, 1, 0.5), drive: C([0, 1, 2, 3, 4, 5], 1), depth: P(0, 1, 0.6),
  rate: C([2, 4, 8, 16], 8), inside: C([0, 1], 1),
};

// ------------------------------------------------------------------- feel

/**
 * How a body feels, separate from how it looks: its response curve and its musical clock.
 *   flow   levels follow the music through an envelope (attack / release times, threshold, sensitivity);
 *   step   the same, but sampled and held on the clock grid (quantised, staccato).
 * div: the clock unit in beats (half-beat .. phrase). The body's periodic motion (spins, sways, orbits,
 * arms) runs at 4 / div of its bar rate (clamped to 0.25x..2x) and its events (walker turns, station
 * jumps, grid refreshes, held steps) come div / 4 times as often as at the bar clock.
 * lock 1: phases follow the song position and events land on the grid; 0: free-running (organic).
 */
export const FEEL_KINDS = ['flow', 'step'] as const;
export type FeelKind = (typeof FEEL_KINDS)[number];
const FEEL_COMMON: Schema = {
  atk: P(0.005, 0.6, 0.005, { log: true }), rel: P(0.005, 3, 0.005, { log: true }), thr: P(0, 0.6, 0), sens: P(0.3, 2.5, 1),
  div: C([0.5, 1, 2, 4, 8, 16], 4), lock: C([0, 1], 1),
};
export const FEEL_SCHEMAS: Record<FeelKind, Schema> = { flow: FEEL_COMMON, step: FEEL_COMMON };

// ----------------------------------------------------------------- colour
// Colour in three parts: the palette (three hue slots relative to the song's key), each body's colour
// mapping (what drives its hue), and the tone (saturation, brightness, contrast, bloom, post).

export const SCHEMES = ['analogous', 'complementary', 'triad', 'split', 'mono'] as const;
export type Scheme = (typeof SCHEMES)[number];
/** Palette kinds: the classic schemes (slot offsets scaled by spread) or three free slots. */
export const PALETTE_KINDS = [...SCHEMES, 'free'] as const;
export type PaletteKind = (typeof PALETTE_KINDS)[number];
const PAL_SCHEME: Schema = { hue: P(0, 1, 0.5), spread: P(0.5, 1.5, 1) };
export const PALETTE_SCHEMAS: Record<PaletteKind, Schema> = {
  analogous: PAL_SCHEME, complementary: PAL_SCHEME, triad: PAL_SCHEME, split: PAL_SCHEME, mono: PAL_SCHEME,
  // hue: the first slot's offset from the key; s1 / s2: the other slots' offsets from the first.
  free: { hue: P(0, 1, 0.5), s1: P(0, 1, 0.33), s2: P(0, 1, 0.66) },
};
export interface PaletteGene {
  kind: PaletteKind;
  p: Params;
}
/**
 * Colour mapping (per body): what drives the hue. fixed: one hue; instrument: each copy its
 * instrument's slot; pitch: the pitch class (chroma, around the key); melody: the melody line; height:
 * the vertical position; age: the hue drifts with time, so older trail keeps older colours; speed: how
 * fast each copy moves. hue: offset into the palette; amount: how far the driver moves the hue;
 * detail: how much the shape's own shading (depth, bins, curve position) varies it; rate: age drift in
 * palette turns per bar.
 */
export const MAPPING_KINDS = ['fixed', 'instrument', 'pitch', 'melody', 'height', 'age', 'speed'] as const;
export type MappingKind = (typeof MAPPING_KINDS)[number];
const MAP_BASE: Schema = { hue: P(0, 1, 0), detail: P(0, 1, 1) };
export const MAPPING_SCHEMAS: Record<MappingKind, Schema> = {
  fixed: MAP_BASE,
  instrument: { ...MAP_BASE, amount: P(0, 1, 1) },
  pitch: MAP_BASE,
  melody: { ...MAP_BASE, amount: P(0, 1, 0.7) },
  height: { ...MAP_BASE, amount: P(-1.5, 1.5, 0.8) },
  age: { ...MAP_BASE, rate: C([0.03125, 0.0625, 0.125, 0.25, 0.5], 0.0625) },
  speed: { ...MAP_BASE, amount: P(0, 1, 0.5) },
};
export const TONE_SCHEMA: Schema = {
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
  // Relief: the carried picture lit as a height map (emboss / liquid chrome); bump: surface height,
  // light: light direction in turns, gloss: specular highlight, metal: palette reflections.
  relief: P(0, 1, 0),
  bump: P(0.2, 3, 1),
  light: P(0, 1, 0.375),
  gloss: P(0, 1, 0.5),
  metal: P(0, 1, 0),
};
export interface ToneGene {
  p: Params;
}

// ------------------------------------------------------------------ genes

export interface Gene<K extends string = string> {
  kind: K;
  p: Params;
}
export interface ShapeGene extends Gene<ShapeKind> {
  xforms?: FlameXformGene[]; // flame only
}
export interface DeformGene extends Gene<DeformKind> {
  /** Draw-space ops bending the body after its own deformation (absent when empty). */
  ops?: OpGene[];
}
export interface FuseGene {
  shape: ShapeGene;
  p: Params;
}
export interface BodyGene {
  shape: ShapeGene;
  place: Gene<PlaceKind>;
  motion: Gene<MotionKind>;
  deform: DeformGene;
  material: Gene<MaterialKind>;
  emit: Gene<EmitKind>;
  feel: Gene<FeelKind>;
  color: Gene<MappingKind>;
  fuse?: FuseGene;
  /**
   * Silent (recessive) alleles: a second gene some loci carry from an ancestor. It is not drawn; a
   * crossover may express it again in a later generation. Absent when the body carries none.
   */
  alt?: Partial<Record<Locus, Gene>>;
}
/** The loci of a body, in a fixed order (homologous slots for crossover): six of look, one of feel. */
export const LOCI = ['shape', 'place', 'motion', 'deform', 'material', 'emit', 'feel', 'color'] as const;
export type Locus = (typeof LOCI)[number];

export const LOCUS_KINDS: Record<Locus, readonly string[]> = {
  shape: SHAPE_KINDS, place: PLACE_KINDS, motion: MOTION_KINDS, deform: DEFORM_KINDS, material: MATERIAL_KINDS, emit: EMIT_KINDS,
  feel: FEEL_KINDS, color: MAPPING_KINDS,
};
export const LOCUS_SCHEMAS: Record<Locus, Record<string, Schema>> = {
  shape: SHAPE_SCHEMAS, place: PLACE_SCHEMAS, motion: MOTION_SCHEMAS, deform: DEFORM_SCHEMAS, material: MATERIAL_SCHEMAS, emit: EMIT_SCHEMAS,
  feel: FEEL_SCHEMAS, color: MAPPING_SCHEMAS,
};

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
  // Unsharp mask of the carried picture (the classic blur-difference warp trick): edges sharpen and
  // uniform areas fade, so the feedback grows reaction-diffusion patterns. grain: its blur radius.
  sharpen: P(0, 1, 0),
  grain: P(0.002, 0.05, 0.006),
  // A coloured frame drawn into the feedback every frame (MilkDrop's outer border), swelling with
  // the bass; the warp carries it inward.
  border: P(0, 1, 0),
  // AVS Water / Water Bump: a ripple height field that drops rings on the beats and refracts the
  // carried picture (water = strength, 0 off; wsize = drop radius). See genes/water.ts.
  ...WATER_PARAMS,
};
export interface CarrierGene {
  kind: CarrierKind;
  p: Params;
}


// -------------------------------------------------------------- reactions

// surge: beat envelope with a fast attack and slow ease that cruises with loudness and jumps on drops.
// barpulse: a pulse on each downbeat; section: a pulse on each section change; bar: a slow wave across the bar.
export const SIGNALS = ['drums', 'bass', 'vocals', 'other', 'hit', 'beat', 'bar', 'complexity', 'drop', 'loud', 'melody', 'build', 'surge', 'barpulse', 'section'] as const;
export type Signal = (typeof SIGNALS)[number];
/**
 * Reaction targets. op: chain[i]; car / col / pal: the carrier / tone / palette (i = 0); body loci,
 * i = body index: sh shape, pl place, mo motion, de deform, ma material, em emit, fe feel, cm colour
 * mapping, fu fuse, fs fused shape;
 * dr: a body's deform op, i = body * MAX_DRAW + op.
 */
export type GeneGroup = 'op' | 'car' | 'col' | 'pal' | 'sh' | 'pl' | 'mo' | 'de' | 'ma' | 'em' | 'fe' | 'cm' | 'fu' | 'fs' | 'dr';
export const GENE_GROUPS: GeneGroup[] = ['op', 'car', 'col', 'pal', 'sh', 'pl', 'mo', 'de', 'ma', 'em', 'fe', 'cm', 'fu', 'fs', 'dr'];
export const BODY_GROUPS: Record<Locus, GeneGroup> = { shape: 'sh', place: 'pl', motion: 'mo', deform: 'de', material: 'ma', emit: 'em', feel: 'fe', color: 'cm' };
/**
 * A reaction is a chain: signal source -> response curve (attack, release, threshold; q 1 holds the
 * value on the clock grid, every div beats) -> gain -> target parameter.
 */
export interface ReactionGene {
  src: Signal;
  g: GeneGroup;
  i: number;
  k: string; // parameter name
  gain: number; // -1..1, fraction of the parameter's half range per unit signal (the sensitivity)
  atk: number; // attack time, seconds
  rel: number; // release time, seconds
  thr: number; // threshold (signal below it is ignored)
  q: number; // 0 free, 1 quantised to the clock
  div: number; // clock unit in beats (quantised reactions)
}
export const REACTION_SCHEMA: Schema = {
  gain: P(-1, 1, 0.3), atk: P(0.005, 0.6, 0.005, { log: true }), rel: P(0.005, 3, 0.005, { log: true }), thr: P(0, 0.6, 0),
  q: C([0, 1], 0), div: C([0.5, 1, 2, 4, 8, 16], 1),
};
export const MAX_REACTIONS = 6;
/** Parameters reactions may not touch (structural switches, counts, clocks). */
const NO_REACT = new Set([
  'mode', 'form', 'solid', 'side', 'axis', 'count', 'reflect', 'tonemap', 'alt', 'radial', 'rounds', 'lock', 'sides', 'ra', 'rb', 'n',
  'bins', 'halfLife', 'lanes', 'strips', 'drive', 'rate', 'inside', 'heads', 'every', 'square', 'wrap', 'jump', 'swap', 'lattice',
  'path', 'period', 'lobes', 'turn', 'tex', 'top', 'fuse', 'div', 'q',
]);

// ----------------------------------------------------------------- genome

/**
 * Genome format. 1: before draw chains and merges; 2: one emitter object per light source
 * (legacy.ts); 3: bodies made of sub-genes; 4: feel genes (a body's response curve and clock, a
 * reaction's curve and clock). Older formats load through repair(); format-3 genomes get the neutral
 * feel (instant response, bar clock, grid-locked), which is exactly how they behaved; 5: colour in
 * parts (palette, per-body colour mapping, tone) replacing the single colour gene and material hues.
 */
export const GENOME_VERSION = 5;

export interface Genome {
  v: 5;
  chain: OpGene[];
  bodies: BodyGene[];
  carrier: CarrierGene;
  palette: PaletteGene;
  tone: ToneGene;
  reactions: ReactionGene[];
  energy: [number, number]; // complexity range the preset suits
  /** Optional: composes the picture over the song timeline (src/v2/genes/choreo.ts). */
  choreo?: ChoreoGene;
}

export const MAX_CHAIN = 6;
/** Hard cap for any genome (converted old children may have 3). */
export const MAX_BODIES = 3;
/** Cap for newly bred children: one body, a second layer only by a rare mutation. */
export const MAX_CHILD_BODIES = 2;

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

export function repairParams(p: unknown, schema: Schema): Params {
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

/** Schema of a body locus gene. */
export function locusSchema(locus: Locus, kind: string): Schema {
  return LOCUS_SCHEMAS[locus][kind] ?? {};
}

export function schemaFor(g: Genome, group: GeneGroup, i: number): Schema | null {
  if (group === 'op') return g.chain[i] ? OP_SCHEMAS[g.chain[i].op] : null;
  if (group === 'car') return CARRIER_SCHEMA;
  if (group === 'col') return TONE_SCHEMA;
  if (group === 'pal') return PALETTE_SCHEMAS[g.palette.kind];
  if (group === 'dr') {
    const o = g.bodies[Math.floor(i / MAX_DRAW)]?.deform.ops?.[i % MAX_DRAW];
    return o ? OP_SCHEMAS[o.op] : null;
  }
  const b = g.bodies[i];
  if (!b) return null;
  switch (group) {
    case 'sh': return SHAPE_SCHEMAS[b.shape.kind];
    case 'pl': return PLACE_SCHEMAS[b.place.kind];
    case 'mo': return MOTION_SCHEMAS[b.motion.kind];
    case 'de': return DEFORM_SCHEMAS[b.deform.kind];
    case 'ma': return MATERIAL_SCHEMAS[b.material.kind];
    case 'em': return EMIT_SCHEMAS[b.emit.kind];
    case 'fe': return FEEL_SCHEMAS[b.feel.kind];
    case 'cm': return MAPPING_SCHEMAS[b.color.kind];
    case 'fu': return b.fuse ? FUSE_SCHEMA : null;
    case 'fs': return b.fuse ? SHAPE_SCHEMAS[b.fuse.shape.kind] : null;
  }
  return null;
}

export function reactable(schema: Schema): string[] {
  return Object.keys(schema).filter((k) => !NO_REACT.has(k) && !schema[k].choices);
}

const num = (v: unknown, d: number) => (typeof v === 'number' && Number.isFinite(v) ? v : d);
const clamp = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v));
const isObj = (v: unknown): v is Record<string, unknown> => !!v && typeof v === 'object';

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

export function repairOp(o: Partial<OpGene> | undefined): OpGene | null {
  if (!o || !OP_KINDS.includes(o.op as OpKind)) return null;
  const op = o.op as OpKind;
  const stage: Stage = stageFree(op) ? (o.stage === 'view' ? 'view' : 'warp') : 'warp';
  return { op, stage, w: clamp(num(o.w, 1), 0, 1), p: repairParams(o.p, OP_SCHEMAS[op]) };
}

export function repairDrawOp(o: Partial<OpGene> | undefined): OpGene | null {
  if (!o || !isDrawOp(o.op as string)) return null;
  const op = o.op as OpKind;
  return { op, stage: 'warp', w: clamp(num(o.w, 1), 0, 1), p: repairParams(o.p, OP_SCHEMAS[op]) };
}

function repairGene<K extends string>(raw: unknown, kinds: readonly K[], schemas: Record<string, Schema>, fallback: K): Gene<K> {
  const r = isObj(raw) ? raw : {};
  const kind = kinds.includes(r.kind as K) ? (r.kind as K) : fallback;
  return { kind, p: repairParams(r.p, schemas[kind]) };
}

export function repairShape(raw: unknown, fallback: ShapeKind = 'dot'): ShapeGene {
  const g = repairGene(raw, SHAPE_KINDS, SHAPE_SCHEMAS, fallback) as ShapeGene;
  if (g.kind === 'flame') {
    const src = isObj(raw) && Array.isArray(raw.xforms) ? (raw.xforms as Partial<FlameXformGene>[]) : [];
    const xs = src.slice(0, MAX_XFORMS).map(repairXform);
    if (!xs.length) xs.push(repairXform({ vars: { linear: 0.5, spherical: 0.5 } }), repairXform({ aff: [0.5, 0, 0, 0.5, 0.5, 0], vars: { sinusoidal: 1 } }));
    g.xforms = xs;
  }
  return g;
}

/** Copy count a placement expresses (fold placements: 1 evaluation, reported as 1). */
export function copyCount(place: Gene<PlaceKind>): number {
  switch (place.kind) {
    case 'orbit': case 'stations': case 'row': case 'float': case 'outline': return place.p.count;
    case 'walker': return place.p.heads;
    default: return 1;
  }
}
/** Name of the copy-count parameter of a placement, if it has one. */
export function countKey(k: PlaceKind): string | null {
  if (k === 'walker') return 'heads';
  if (k === 'orbit' || k === 'stations' || k === 'row' || k === 'float' || k === 'outline') return 'count';
  return null;
}

/**
 * One body made valid: every locus kind known and its params in range, and the
 * combinations the renderer expresses: chunk shapes take a single copy and no
 * paint-over, a flame needs a placement with a position and a trail, a curve
 * cannot use a grid or the solid materials, a fused shape needs distance fields.
 */
export function repairBody(raw: unknown): BodyGene {
  const r = isObj(raw) ? raw : {};
  const shape = repairShape(r.shape);
  const place = repairGene(r.place, PLACE_KINDS, PLACE_SCHEMAS, 'point');
  const motion = repairGene(r.motion, MOTION_KINDS, MOTION_SCHEMAS, 'none');
  const deform = repairGene(r.deform, DEFORM_KINDS, DEFORM_SCHEMAS, 'none') as DeformGene;
  const material = repairGene(r.material, MATERIAL_KINDS, MATERIAL_SCHEMAS, 'glow');
  const emit = repairGene(r.emit, EMIT_KINDS, EMIT_SCHEMAS, 'trail');
  const feel = repairGene(r.feel, FEEL_KINDS, FEEL_SCHEMAS, 'flow');
  const color = repairGene(r.color, MAPPING_KINDS, MAPPING_SCHEMAS, 'fixed');
  const rawOps = isObj(r.deform) && Array.isArray(r.deform.ops) ? (r.deform.ops as Partial<OpGene>[]) : [];
  const ops = rawOps.map(repairDrawOp).filter((o): o is OpGene => !!o).slice(0, MAX_DRAW);
  if (ops.length) deform.ops = ops;

  const cls = SHAPE_CLASS[shape.kind];
  const setKind = <K extends string>(gene: Gene<K>, kind: K, schemas: Record<string, Schema>, keep: Params = {}) => {
    gene.kind = kind;
    gene.p = repairParams({ ...keep }, schemas[kind]);
  };
  // Chunk shapes are drawn once: every copy count becomes 1.
  if (cls === 'field' || cls === 'flame') {
    const ck = countKey(place.kind);
    if (ck) place.p[ck] = 1;
    if (place.kind === 'float') place.p.fuse = 0;
  }
  if (cls === 'flame' && isFoldPlace(place.kind)) setKind(place, 'point', PLACE_SCHEMAS, { x: place.p.x ?? 0, y: place.p.y ?? 0 });
  if (cls === 'flame' && emit.kind !== 'trail' && emit.kind !== 'sparks') setKind(emit, 'trail', EMIT_SCHEMAS);
  if (cls === 'curve' && place.kind === 'grid') setKind(place, 'point', PLACE_SCHEMAS);
  if (cls === 'curve' && (material.kind === 'fill' || material.kind === 'textured' || material.kind === 'chrome')) {
    setKind(material, 'line', MATERIAL_SCHEMAS, { gain: material.p.gain });
  }
  if (cls !== 'sdf' && emit.kind === 'cover' && !(cls === 'curve' && isObj(r.fuse))) setKind(emit, 'trail', EMIT_SCHEMAS, { tip: emit.p.tip });

  const body: BodyGene = { shape, place, motion, deform, material, emit, feel, color };
  if (isObj(r.fuse)) {
    const f = r.fuse as Record<string, unknown>;
    const fs = repairShape(f.shape, 'dot');
    const fp = repairParams(f.p, FUSE_SCHEMA);
    if (fp.mode !== 2 && !sdfCapable(shape)) fp.mode = 2;
    // The same kind twice only makes sense as a region (e.g. sparks born inside a disc).
    // A superscope has no distance field to light up inside, so it never fuses.
    const ok = sdfCapable(fs) && (fs.kind !== shape.kind || fp.mode === 2) && shape.kind !== 'superscope';
    if (ok) {
      body.fuse = { shape: fs, p: fp };
      // A fused curve is drawn through its distance field.
      if (shape.kind === 'curve' && (shape.p.form === 3 || shape.p.form === 5)) shape.p.form = 1;
    }
  }
  if (cls === 'curve' && !body.fuse && emit.kind === 'cover') setKind(emit, 'trail', EMIT_SCHEMAS, { tip: emit.p.tip });
  // Silent alleles: valid genes of a different kind than the expressed one (flames never go silent).
  if (isObj(r.alt)) {
    const alt: Partial<Record<Locus, Gene>> = {};
    for (const locus of LOCI) {
      const a = (r.alt as Record<string, unknown>)[locus];
      if (!isObj(a) || !LOCUS_KINDS[locus].includes(a.kind as string)) continue;
      const g = locus === 'shape' ? repairShape(a) : repairGene(a, LOCUS_KINDS[locus] as readonly string[], LOCUS_SCHEMAS[locus], a.kind as string);
      if (g.kind === (body[locus] as Gene).kind || g.kind === 'flame') continue;
      alt[locus] = g;
    }
    if (Object.keys(alt).length) body.alt = alt;
  }
  return body;
}

/**
 * Format 3/4 -> 5: the single colour gene splits into palette (scheme and hue) and tone (the rest);
 * each body's material hue becomes its colour mapping's hue, with the mapping its placement implied
 * (grids by pitch class, copies by instrument, single shapes one hue); hue reactions follow.
 */
export function upgradeColour(src: Record<string, unknown>): Record<string, unknown> {
  const color = src.color as { scheme?: unknown; p?: Record<string, unknown> };
  const cp = isObj(color.p) ? color.p : {};
  const out: Record<string, unknown> = { ...src };
  delete out.color;
  out.palette = { kind: color.scheme, p: { hue: cp.hue, spread: 1 } };
  const tp: Record<string, unknown> = { ...cp };
  delete tp.hue;
  out.tone = { p: tp };
  if (Array.isArray(src.bodies)) {
    out.bodies = src.bodies.map((b) => {
      if (!isObj(b)) return b;
      const mat = isObj(b.material) ? b.material : {};
      const mp = isObj(mat.p) ? (mat.p as Record<string, unknown>) : {};
      const place = isObj(b.place) ? String(b.place.kind) : 'point';
      const kind = place === 'grid' ? 'pitch' : ['orbit', 'stations', 'row', 'outline', 'ring', 'walker'].includes(place) ? 'instrument' : 'fixed';
      return { ...b, color: { kind, p: { hue: mp.hue ?? 0, detail: 1, amount: 1 } } };
    });
  }
  if (Array.isArray(src.reactions)) {
    out.reactions = src.reactions.map((r) => {
      if (!isObj(r)) return r;
      if (r.g === 'col' && r.k === 'hue') return { ...r, g: 'pal' };
      if (r.g === 'ma' && r.k === 'hue') return { ...r, g: 'cm' };
      return r;
    });
  }
  out.v = 5;
  return out;
}

/**
 * Returns a valid genome: every parameter clamped into its spec, unknown fields
 * dropped, caps enforced, dangling reactions retargeted or removed, and the GPU
 * cost brought under the budget by dropping copies. Older formats (1, 2) are
 * converted first. Idempotent: repair(repair(g)) deep-equals repair(g).
 */
export function repair(input: unknown): Genome {
  let src = (isObj(input) ? input : {}) as Record<string, unknown>;
  if (src.v !== 3 && src.v !== 4 && src.v !== 5 && (Array.isArray(src.emitters) || src.v === 1 || src.v === 2)) src = upgradeV2(repairV2(src)) as Record<string, unknown>;
  if (src.v !== 5 && isObj(src.color) && !isObj(src.palette)) src = upgradeColour(src);
  const g = src as Partial<Genome>;
  const chain = (Array.isArray(g.chain) ? g.chain : []).map(repairOp).filter((o): o is OpGene => !!o).slice(0, MAX_CHAIN);

  const bodies: BodyGene[] = [];
  const used = new Set<ShapeKind>();
  let sparks = false;
  let slime = false;
  for (const raw of Array.isArray(g.bodies) ? g.bodies : []) {
    if (bodies.length >= MAX_BODIES) break;
    const b = repairBody(raw);
    if (UNIQUE_SHAPES.includes(b.shape.kind) && used.has(b.shape.kind)) continue;
    if (b.fuse && UNIQUE_SHAPES.includes(b.fuse.shape.kind) && (used.has(b.fuse.shape.kind) || b.fuse.shape.kind === b.shape.kind)) dropFuse(b);
    // One particle system per genome: a second sparks body leaves a plain trail.
    if (b.emit.kind === 'sparks') {
      if (sparks) b.emit = { kind: 'trail', p: defaultParams(EMIT_SCHEMAS.trail) };
      sparks = true;
    }
    // One agent simulation per genome as well.
    if (b.emit.kind === 'slime') {
      if (slime) b.emit = { kind: 'trail', p: defaultParams(EMIT_SCHEMAS.trail) };
      slime = true;
    }
    used.add(b.shape.kind);
    if (b.fuse) used.add(b.fuse.shape.kind);
    if (b.alt) {
      for (const k of Object.keys(b.alt) as Locus[]) if (b.alt[k]!.kind === (b[k] as Gene).kind) delete b.alt[k];
      if (!Object.keys(b.alt).length) delete b.alt;
    }
    bodies.push(b);
  }
  if (!bodies.length) bodies.push(repairBody({ shape: { kind: 'curve' }, place: { kind: 'point' }, material: { kind: 'line' }, emit: { kind: 'trail' } }));
  const ck = CARRIER_KINDS.includes(g.carrier?.kind as CarrierKind) ? (g.carrier!.kind as CarrierKind) : 'warp';
  const carrier: CarrierGene = { kind: ck, p: repairParams(g.carrier?.p, CARRIER_SCHEMA) };
  const pk = PALETTE_KINDS.includes(g.palette?.kind as PaletteKind) ? (g.palette!.kind as PaletteKind) : 'analogous';
  const palette: PaletteGene = { kind: pk, p: repairParams(g.palette?.p, PALETTE_SCHEMAS[pk]) };
  const tone: ToneGene = { p: repairParams(g.tone?.p, TONE_SCHEMA) };

  let lo = clamp(num(g.energy?.[0], 0.2), 0, 1);
  let hi = clamp(num(g.energy?.[1], 0.7), 0, 1);
  if (hi < lo) [lo, hi] = [hi, lo];
  if (hi - lo < 0.15) {
    const c = clamp((lo + hi) / 2, 0.075, 0.925);
    lo = c - 0.075;
    hi = c + 0.075;
  }
  const out: Genome = { v: 5, chain, bodies, carrier, palette, tone, reactions: [], energy: [round4(lo), round4(hi)] };
  if (isObj(g.choreo)) out.choreo = repairChoreo(g.choreo);
  fitBudget(out);

  for (const r of Array.isArray(g.reactions) ? g.reactions : []) {
    if (out.reactions.length >= MAX_REACTIONS) break;
    if (!r || !SIGNALS.includes(r.src as Signal)) continue;
    const group = GENE_GROUPS.includes(r.g) ? r.g : null;
    if (!group) continue;
    const i = reactionIndex(out, group, num(r.i, 0));
    if (i < 0) continue;
    const keys = reactable(schemaFor(out, group, i) ?? {});
    if (!keys.length) continue;
    const k = keys.includes(r.k) ? r.k : keys.includes('gain') ? 'gain' : keys[0];
    if (out.reactions.some((q) => q.g === group && q.i === i && q.k === k && q.src === r.src)) continue;
    const c = repairParams(r, REACTION_SCHEMA);
    out.reactions.push({ src: r.src as Signal, g: group, i, k, gain: c.gain, atk: c.atk, rel: c.rel, thr: c.thr, q: c.q, div: c.div });
  }
  return out;
}

/** A reaction index wrapped onto an existing target of the group, or -1 when the group has none. */
function reactionIndex(g: Genome, group: GeneGroup, raw: number): number {
  const wrap = (n: number) => (n ? ((Math.floor(raw) % n) + n) % n : -1);
  if (group === 'car' || group === 'col' || group === 'pal') return 0;
  if (group === 'op') return wrap(g.chain.length);
  if (group === 'dr') {
    const slots: number[] = [];
    g.bodies.forEach((b, bi) => b.deform.ops?.forEach((_o, j) => slots.push(bi * MAX_DRAW + j)));
    if (!slots.length) return -1;
    return slots.includes(Math.floor(raw)) ? Math.floor(raw) : slots[wrap(slots.length)];
  }
  if (group === 'fu' || group === 'fs') {
    const withFuse = g.bodies.map((b, bi) => (b.fuse ? bi : -1)).filter((x) => x >= 0);
    if (!withFuse.length) return -1;
    return withFuse.includes(Math.floor(raw)) ? Math.floor(raw) : withFuse[0];
  }
  return wrap(g.bodies.length);
}

/** Drops copies (most expensive body first) until the estimated cost fits the budget. */
function fitBudget(g: Genome): void {
  // Slime agents go first (halved, down to a still-connected network), then copies, then more agents.
  const fewerAgents = (floor: number) => {
    for (const b of g.bodies) {
      while (b.emit.kind === 'slime' && b.emit.p.count > floor && estimateCost(g) > COST_BUDGET_MS * 0.95) {
        b.emit.p.count = Math.max(floor, Math.round(b.emit.p.count / 2));
      }
    }
  };
  fewerAgents(65536);
  for (let guard = 0; guard < 24 && estimateCost(g) > COST_BUDGET_MS * 0.95; guard++) {
    let best: BodyGene | null = null;
    let bc = 0;
    for (const b of g.bodies) {
      const ck = countKey(b.place.kind);
      if (!ck || b.place.p[ck] <= 1) continue;
      const c = bodyCost(b);
      if (c > bc) {
        bc = c;
        best = b;
      }
    }
    if (!best) break;
    best.place.p[countKey(best.place.kind)!] -= 1;
  }
  fewerAgents(SLIME_SCHEMA.count.min);
  // Still over: a grid evaluates one cell instead of its 3x3 neighbourhood, then fused shapes go.
  for (const b of g.bodies) {
    if (estimateCost(g) <= COST_BUDGET_MS * 0.95) return;
    if (b.place.kind === 'grid' && evalCount(b) > 1) b.place.p.jitter = 0.25;
  }
  for (const b of g.bodies) {
    if (estimateCost(g) <= COST_BUDGET_MS * 0.95) return;
    if (b.fuse) dropFuse(b);
  }
  // Physics fields with a fixture count shed fixtures last.
  for (const b of g.bodies) {
    while (b.shape.kind === 'beams' && b.shape.p.count > 1 && estimateCost(g) > COST_BUDGET_MS * 0.95) b.shape.p.count--;
  }
}

/** Removes a body's fused shape; a curve that painted over the trail through it goes back to a plain trail. */
function dropFuse(b: BodyGene): void {
  delete b.fuse;
  if (SHAPE_CLASS[b.shape.kind] !== 'sdf' && b.emit.kind === 'cover') b.emit = { kind: 'trail', p: repairParams({ tip: b.emit.p.tip }, EMIT_SCHEMAS.trail) };
}

function round4(x: number): number {
  return Math.round(x * 1e4) / 1e4;
}

/** Lists every rule a genome breaks (empty when valid). Does not modify it. */
export function validate(g: Genome): string[] {
  const errs: string[] = [];
  const chk = (p: Params | undefined, s: Schema, where: string) => {
    if (!p || typeof p !== 'object') {
      errs.push(`${where} params`);
      return;
    }
    for (const k of Object.keys(s)) if (!inRange(p[k], s[k])) errs.push(`${where}.${k}=${p[k]} out of range`);
    for (const k of Object.keys(p)) if (!(k in s)) errs.push(`${where}.${k} unknown`);
  };
  if (g.v !== 5) errs.push('version');
  if (!Array.isArray(g.chain) || g.chain.length > MAX_CHAIN) errs.push('chain length');
  g.chain?.forEach((o, i) => {
    if (!OP_KINDS.includes(o.op)) errs.push(`chain[${i}] op ${o.op}`);
    else chk(o.p, OP_SCHEMAS[o.op], `chain[${i}]`);
    if (!(o.w >= 0 && o.w <= 1)) errs.push(`chain[${i}].w`);
    if (o.stage === 'view' && !stageFree(o.op)) errs.push(`chain[${i}] stage`);
  });
  if (!Array.isArray(g.bodies) || g.bodies.length < 1 || g.bodies.length > MAX_BODIES) errs.push('body count');
  const used = new Set<string>();
  let sparks = 0;
  let slimes = 0;
  g.bodies?.forEach((b, i) => {
    const w = `bodies[${i}]`;
    for (const locus of LOCI) {
      const gene = b[locus] as Gene;
      if (!gene || !LOCUS_KINDS[locus].includes(gene.kind)) {
        errs.push(`${w}.${locus} kind`);
        continue;
      }
      chk(gene.p, locusSchema(locus, gene.kind), `${w}.${locus}`);
    }
    if (errs.length) return;
    const cls = SHAPE_CLASS[b.shape.kind];
    const checkXforms = (s: ShapeGene, where: string) => {
      if (s.kind !== 'flame') {
        if (s.xforms) errs.push(`${where} stray xforms`);
        return;
      }
      if (!s.xforms || s.xforms.length < 1 || s.xforms.length > MAX_XFORMS) errs.push(`${where} xforms count`);
      s.xforms?.forEach((x, j) => {
        if (x.aff.length !== 6 || x.aff.some((a) => !(a >= AFF_RANGE[0] && a <= AFF_RANGE[1]))) errs.push(`xform[${j}].aff`);
        if (!(x.weight >= 0.05 && x.weight <= 1)) errs.push(`xform[${j}].weight`);
        if (!(x.color >= 0 && x.color <= 1)) errs.push(`xform[${j}].color`);
        const vs = Object.entries(x.vars);
        if (!vs.length || vs.some(([k, v]) => !FLAME_VARIATIONS.includes(k as FlameVar) || !(v! >= 0 && v! <= 1))) errs.push(`xform[${j}].vars`);
        if (!LOCKS.includes(x.spin)) errs.push(`xform[${j}].spin`);
        if (!(x.bass >= 0 && x.bass <= 0.3)) errs.push(`xform[${j}].bass`);
        if (!(x.pulse >= 0 && x.pulse <= 0.3)) errs.push(`xform[${j}].pulse`);
        if (!Array.isArray(x.drift) || x.drift.length !== 2 || x.drift.some((d) => !(d >= 0 && d <= XFORM_DRIFT))) errs.push(`xform[${j}].drift`);
        if (x.alt && Object.entries(x.alt).some(([k, v]) => !FLAME_VARIATIONS.includes(k as FlameVar) || !(v! >= 0 && v! <= 1))) errs.push(`xform[${j}].alt`);
      });
    };
    checkXforms(b.shape, `${w}.shape`);
    if (b.deform.ops !== undefined) {
      if (!Array.isArray(b.deform.ops) || !b.deform.ops.length || b.deform.ops.length > MAX_DRAW) errs.push(`${w} deform ops length`);
      b.deform.ops?.forEach?.((o, j) => {
        if (!isDrawOp(o.op)) errs.push(`${w}.ops[${j}] op ${o.op}`);
        else chk(o.p, OP_SCHEMAS[o.op], `${w}.ops[${j}]`);
        if (!(o.w >= 0 && o.w <= 1)) errs.push(`${w}.ops[${j}].w`);
        if (o.stage !== 'warp') errs.push(`${w}.ops[${j}] stage`);
      });
    }
    const ck = countKey(b.place.kind);
    if ((cls === 'field' || cls === 'flame') && ck && b.place.p[ck] !== 1) errs.push(`${w} chunk shape with copies`);
    if (cls === 'flame' && isFoldPlace(b.place.kind)) errs.push(`${w} flame fold placement`);
    if (cls === 'flame' && b.emit.kind !== 'trail' && b.emit.kind !== 'sparks') errs.push(`${w} flame emission`);
    if (cls === 'curve' && b.place.kind === 'grid') errs.push(`${w} curve on a grid`);
    if (cls === 'curve' && (b.material.kind === 'fill' || b.material.kind === 'textured' || b.material.kind === 'chrome')) errs.push(`${w} curve material`);
    if (cls !== 'sdf' && b.emit.kind === 'cover' && !(cls === 'curve' && b.fuse)) errs.push(`${w} cover without a distance field`);
    for (const k of [b.shape.kind, ...(b.fuse?.shape ? [b.fuse.shape.kind] : [])]) {
      if (!UNIQUE_SHAPES.includes(k)) continue;
      if (used.has(k)) errs.push(`${w} duplicate ${k}`);
      used.add(k);
    }
    if (b.emit.kind === 'sparks') sparks++;
    if (b.emit.kind === 'slime') slimes++;
    if (b.fuse) {
      const f = b.fuse;
      if (!f.shape || !SHAPE_KINDS.includes(f.shape.kind)) errs.push(`${w}.fuse shape kind`);
      else {
        chk(f.shape.p, SHAPE_SCHEMAS[f.shape.kind], `${w}.fuse.shape`);
        checkXforms(f.shape, `${w}.fuse.shape`);
        if (!sdfCapable(f.shape)) errs.push(`${w}.fuse shape has no distance field`);
        if (f.shape.kind === b.shape.kind && f.p.mode !== 2) errs.push(`${w}.fuse same kind`);
      }
      chk(f.p, FUSE_SCHEMA, `${w}.fuse`);
      if (f.p.mode !== 2 && !sdfCapable(b.shape)) errs.push(`${w}.fuse needs a distance field`);
      if (b.shape.kind === 'curve' && (b.shape.p.form === 3 || b.shape.p.form === 5)) errs.push(`${w}.fuse curve form`);
      if (b.shape.kind === 'superscope') errs.push(`${w}.fuse on a superscope`);
    }
  });
  g.bodies?.forEach((b, i) => {
    if (b.alt === undefined) return;
    const keys = Object.keys(b.alt);
    if (!keys.length) errs.push(`bodies[${i}].alt empty`);
    for (const k of keys) {
      const locus = k as Locus;
      const a = b.alt[locus];
      if (!LOCI.includes(locus) || !a || !LOCUS_KINDS[locus].includes(a.kind)) {
        errs.push(`bodies[${i}].alt.${k} kind`);
        continue;
      }
      chk(a.p, locusSchema(locus, a.kind), `bodies[${i}].alt.${k}`);
      if (a.kind === (b[locus] as Gene).kind || a.kind === 'flame') errs.push(`bodies[${i}].alt.${k} not silent`);
    }
  });
  if (sparks > 1) errs.push('more than one sparks emission');
  if (slimes > 1) errs.push('more than one slime emission');
  if (!CARRIER_KINDS.includes(g.carrier?.kind)) errs.push('carrier kind');
  else chk(g.carrier.p, CARRIER_SCHEMA, 'carrier');
  if (!PALETTE_KINDS.includes(g.palette?.kind)) errs.push('palette kind');
  else chk(g.palette.p, PALETTE_SCHEMAS[g.palette.kind], 'palette');
  chk(g.tone?.p, TONE_SCHEMA, 'tone');
  if (g.choreo !== undefined) errs.push(...validateChoreo(g.choreo));
  if (!(g.energy?.[0] >= 0 && g.energy[1] <= 1 && g.energy[0] < g.energy[1])) errs.push('energy');
  if (g.reactions?.length > MAX_REACTIONS) errs.push('reaction count');
  g.reactions?.forEach((r, i) => {
    const s = schemaFor(g, r.g, r.i);
    if (!s || !reactable(s).includes(r.k)) errs.push(`reactions[${i}] target ${r.g}${r.i}.${r.k}`);
    if (!SIGNALS.includes(r.src)) errs.push(`reactions[${i}] src`);
    for (const key of Object.keys(REACTION_SCHEMA)) if (!inRange((r as unknown as Params)[key], REACTION_SCHEMA[key])) errs.push(`reactions[${i}].${key}`);
  });
  return errs;
}

// ------------------------------------------------------------- identity

export function fnv1a(s: string): number {
  let h = 0x811c9dc5;
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i);
    h = Math.imul(h, 0x01000193);
  }
  return h >>> 0;
}

/** Shader layer a body draws on: 'fb' accumulates into the feedback, 'top' is redrawn in the composite. */
export function bodyLayer(b: BodyGene): 'fb' | 'top' {
  return b.emit.kind === 'none' ? 'top' : 'fb';
}

/** Structural key of one body (what changes its shader code). Motion and all numbers are uniforms. */
export function bodyKey(b: BodyGene): string {
  const place = isFoldPlace(b.place.kind) ? b.place.kind : 'loop';
  const metaball = b.shape.kind === 'dot' && (b.place.p.fuse ?? 0) > 0 ? 'm' : '';
  const fuse = b.fuse ? `+${b.fuse.p.mode}${b.fuse.shape.kind}${b.fuse.p.inside}` : '';
  const emit = b.emit.kind === 'cover' ? 'c' : b.emit.kind === 'sparks' ? (b.emit.p.top > 0.5 ? 'st' : 's') : '';
  return `${b.shape.kind}:${place}${metaball}:${b.deform.kind}:${b.material.kind}${b.material.kind === 'textured' ? b.material.p.tex : ''}:${bodyLayer(b)}${emit}${fuse}`;
}

/** Everything that changes the compiled shaders (numeric params are uniforms). */
export function structuralKey(g: Genome): string {
  const ops = g.chain.map((o) => `${o.op}${o.stage === 'view' ? '@v' : ''}`).join(',');
  const bodies = g.bodies.map(bodyKey).join(',');
  return `${ops}|${bodies}|${g.carrier.kind}|r${g.tone.p.reflect}t${g.tone.p.tonemap}`;
}

export function genomeHash(g: Genome): number {
  return fnv1a(JSON.stringify(g, (_k, v) => (typeof v === 'number' ? Math.round(v * 1000) / 1000 : v)));
}

export function cloneGenome(g: Genome): Genome {
  return JSON.parse(JSON.stringify(g)) as Genome;
}

export function cloneBody(b: BodyGene): BodyGene {
  return JSON.parse(JSON.stringify(b)) as BodyGene;
}

// -------------------------------------------------------- classification

export function speciesScores(g: Genome): Record<Species, number> {
  const s = Object.fromEntries(SPECIES.map((k) => [k, 0])) as Record<Species, number>;
  for (const b of g.bodies) {
    const sp = b.shape.p;
    const w = 0.6 + 0.4 * Math.min(1, b.material.p.gain);
    const pk = b.place.kind;
    const hidden = b.emit.kind === 'sparks' && b.emit.p.body < 0.3;
    const shapeW = hidden ? 0.3 : 1;
    switch (b.shape.kind) {
      case 'flame': s.flame += 3.2 * w; break;
      case 'curve': s.scope += 2.4 * w * shapeW; break;
      case 'superscope': s.scope += 2 * w * shapeW; s.wire += 0.8 * w * shapeW; break;
      case 'bars': s.spectrum += 2.6 * w * shapeW; break;
      case 'solid': s.wire += 2.6 * w * shapeW; break;
      case 'plasma': s.plasma += 2.6 * w; break;
      case 'aurora': s.aurora += 2.6 * w; break;
      case 'beams': s.aurora += 2.4 * w; s.spectrum += 0.6; break;
      case 'terrain': s.terrain += 2.6 * w; break;
      case 'edge': (sp.mode === 2 ? (s.rain += 2.6) : (s.terrain += 2.6)); if (sp.side === 1 || sp.side === 2) s.rain += 0.6; break;
      case 'dot':
        if (pk === 'grid') s.stars += 2.6 * w * shapeW;
        else if (pk === 'walker') s.scope += 2.4 * w * shapeW;
        else if (pk === 'float' || b.material.kind === 'chrome') s.chrome += 2.6 * w * shapeW;
        else if (pk === 'orbit') { s.vortex += 0.8 * shapeW; s.ink += 0.4 * shapeW; }
        else if (pk === 'stations' || pk === 'row') s.ink += 1.8 * shapeW;
        else s.terrain += 1.2 * w * shapeW;
        break;
      case 'polygon': case 'star':
        if (pk === 'grid') s.mirror += 2.4 * w * shapeW;
        else s.wire += 1.4 * w * shapeW;
        break;
      case 'segment':
        if (pk === 'walker') s.scope += 2 * w * shapeW;
        else s.wire += 1.2 * w * shapeW;
        break;
    }
    if (b.material.kind === 'chrome' && b.shape.kind !== 'dot') s.chrome += 1;
    if (b.emit.kind === 'dye') s.ink += 1;
    if (b.emit.kind === 'sparks') {
      const e = b.emit.p;
      s.stars += 1.8;
      if (pk === 'row' && e.lift > 0.03) s.rain += 1.6;
      if (pk === 'point' && e.zoomFlow > 0.5) s.stars += 1;
    }
    if (pk === 'ring' || pk === 'mirror') s.mirror += 1.2;
    if (b.deform.kind === 'twist') s.vortex += 0.6;
    if (b.deform.kind === 'noise') s.ink += 0.3;
    for (const o of b.deform.ops ?? []) {
      if (o.op === 'swirl' || o.op === 'twist' || o.op === 'rotate') s.vortex += 0.6;
      else if (o.op === 'mirror' || o.op === 'kaleido') s.mirror += 1.2;
      else if (isVarOp(o.op)) s.flame += 0.4;
    }
    if (b.fuse) {
      const f = b.fuse.shape;
      if (f.kind === 'solid') s.wire += 1;
      else if (f.kind === 'curve') s.scope += 1;
      else if (f.kind === 'bars') s.spectrum += 1;
      else if (f.kind === 'aurora') s.aurora += 1;
    }
  }
  if (g.carrier.kind === 'fluid') s.ink += 2.6;
  if (g.carrier.kind === 'flow') s.rain += 0.4;
  // A sharpened carrier grows its own pattern field; a border feeds one from the edges.
  if (g.carrier.kind !== 'none') s.plasma += 3 * g.carrier.p.sharpen + 1.6 * g.carrier.p.border;
  if (g.carrier.kind !== 'none') s.ink += 1.2 * g.carrier.p.water;
  // An embossed picture reads as a lit surface: liquid metal when it reflects the palette.
  s.chrome += g.tone.p.relief * (2.8 + 1.4 * g.tone.p.metal);
  for (const o of g.chain) {
    const p = o.p;
    switch (o.op) {
      case 'zoom': s.vortex += Math.min(2.2, Math.abs(p.rate) * o.w * 110); break;
      case 'rotate': s.vortex += Math.min(1.2, (Math.abs(p.lock) * 4 + Math.abs(p.rate) * 60) * o.w); break;
      case 'swirl': s.vortex += Math.min(1.8, Math.abs(p.amt) * o.w * 120); break;
      case 'twist': s.vortex += Math.min(1, Math.abs(p.amt) * o.w * 150); break;
      case 'translate':
        if (Math.abs(p.vy) > Math.abs(p.vx) && Math.abs(p.vy) > 0.08) s.rain += 1.2;
        else if (Math.abs(p.vx) > 0.08) s.terrain += 1;
        break;
      case 'noise': s.ink += 0.3; s.rain += 0.2; break;
      case 'mirror': s.mirror += 1.6; break;
      case 'tile': s.mirror += 1.8; break;
      case 'kaleido': s.mirror += 2.2; break;
      case 'polar': s.vortex += 1; break;
      case 'push': if (p.axis === 0) s.mirror += 0.5; else s.vortex += 0.5; break;
      case 'quad': s.flame += Math.min(1.6, p.amt * o.w); s.plasma += Math.min(1.2, p.amt * o.w * 0.6); break;
      case 'stretch': s.terrain += 1.2; break;
      default: s.flame += 0.5 * o.w; // variations
    }
  }
  return s;
}

export interface Classification {
  primary: Species;
  secondary: Species | null;
  label: string; // "vortex × flame" style
}

export function classify(g: Genome): Classification {
  const s = speciesScores(g);
  const ranked = SPECIES.slice().sort((a, b) => s[b] - s[a]);
  const primary = ranked[0];
  const second = ranked[1];
  const secondary = s[second] >= 1.5 && s[second] >= 0.6 * s[primary] ? second : null;
  const short = (k: Species) => SPECIES_LABEL[k].split('/')[0];
  return { primary, secondary, label: secondary ? `${short(primary)} × ${short(secondary)}` : SPECIES_LABEL[primary] };
}

export function energyOf(g: Genome): Energy {
  return (g.energy[0] + g.energy[1]) / 2 >= 0.5 ? 'energetic' : 'calm';
}

// -------------------------------------------------------------- cost

/**
 * Estimated GPU milliseconds per frame at 2560x1440 on Apple Silicon, calibrated against timer-query
 * measurements of isolated parts (seeds and single-part genomes, median of 30-40 frames): e.g. the
 * frame's fixed passes ~0.45 ms, a distance-field body ~0.9 ms plus its evaluations (a dot 0.3, a
 * wireframe solid ~4 ms), a 3x3 star grid ~3.5 ms, plasma ~5 ms, a 262k-point flame round ~1.35 ms.
 */
export function estimateCost(g: Genome): number {
  let ms = 0.45; // feedback + composite + bloom + exposure + final
  for (const o of g.chain) ms += o.op === 'noise' ? 1.1 : isVarOp(o.op) ? 0.15 : o.op === 'stretch' ? 0.1 : 0.03;
  if (g.carrier.kind === 'fluid') ms += 0.5;
  if (g.carrier.kind === 'flow') ms += 0.35;
  if (g.carrier.p.blur > 0) ms += 0.2;
  if (g.carrier.kind !== 'none' && g.carrier.p.sharpen > 0.001) ms += 0.45;
  if (g.carrier.kind !== 'none' && g.carrier.p.border > 0.001) ms += 0.03;
  if (g.carrier.kind !== 'none' && g.carrier.p.water > 0.001) ms += WATER_COST;
  if (g.tone.p.relief > 0.001) ms += 0.12; // four extra feedback samples in the composite
  for (const b of g.bodies) ms += bodyCost(b);
  if (g.choreo) ms += CHOREO_COST_MS;
  return ms;
}

/** One distance-field evaluation of a shape, per full-screen pass. */
const SDF_COST: Record<ShapeKind, number> = {
  dot: 0.3, polygon: 0.3, star: 0.35, segment: 0.3, solid: 4.0, bars: 0.1, curve: 0.25, aurora: 0.6,
  plasma: 0.5, terrain: 0.5, edge: 0.3, flame: 0.3, superscope: SUPERSCOPE_COST,
  beams: 0.3,
};
const FIELD_COST: Partial<Record<ShapeKind, number>> = { plasma: 6.3, aurora: 1.5, edge: 0.05 };
const MATERIAL_COST: Record<MaterialKind, number> = { line: 0.05, fill: 0.05, glow: 0.05, dots: 0.1, textured: 0.35, chrome: 0.45 };
/** One evaluation of a shape; a wireframe costs by its segment count (plus the inner solid). */
function shapeEvalCost(sh: ShapeGene): number {
  if (sh.kind !== 'solid') return SDF_COST[sh.kind];
  const segs = sh.p.solid === 4 ? sh.p.sides : [6, 12, 12, 30, 0, 30][sh.p.solid] + (sh.p.inner > 0.01 ? 12 : 0);
  return 0.4 + 0.11 * segs;
}
/** A distance-field body's fixed cost (its pass, loop, material and tip), before its evaluations. */
const BODY_OVERHEAD = 0.9;

function deformCost(b: BodyGene): number {
  let ms = 0;
  switch (b.deform.kind) {
    case 'arms': ms += 0.15 * b.deform.p.count; break;
    case 'wobble': ms += 0.1; break;
    case 'noise': ms += 1.5; break;
    case 'twist': ms += 0.05; break;
  }
  for (const o of b.deform.ops ?? []) ms += o.op === 'noise' ? 1.2 : isVarOp(o.op) ? 0.2 : 0.1;
  return ms;
}

/** Evaluations of the shape per pixel a placement costs. */
export function evalCount(b: BodyGene): number {
  if (b.place.kind === 'grid') return b.place.p.lattice === 0 && b.place.p.jitter > 0.25 ? 9 : 1;
  if (isFoldPlace(b.place.kind)) return 1;
  return copyCount(b.place);
}

export function bodyCost(b: BodyGene): number {
  const cls = SHAPE_CLASS[b.shape.kind];
  let ms = 0;
  if (b.emit.kind === 'sparks') ms += 0.25 + (b.emit.p.count / 65536) * 0.6;
  if (b.emit.kind === 'slime') ms += slimeCost(b.emit.p.count);
  if (cls === 'flame') {
    ms += (b.shape.p.count / 262144) * b.shape.p.rounds * 1.35;
    if (b.fuse || b.deform.kind !== 'none') ms += 0.3 + deformCost(b) + (b.fuse ? SDF_COST[b.fuse.shape.kind] : 0);
    return ms;
  }
  if (cls === 'field') {
    const base = b.shape.kind === 'terrain' ? 0.25 + (b.shape.p.terrain > 0.001 ? 0.3 : 0) : b.shape.kind === 'beams' ? beamsCost(b.shape.p) : FIELD_COST[b.shape.kind] ?? 0.3;
    ms += base + deformCost(b) + (b.fuse ? 0.2 + SDF_COST[b.fuse.shape.kind] : 0);
    return ms;
  }
  if (cls === 'curve' && !b.fuse) {
    const draws = b.place.kind === 'ring' ? b.place.p.n : b.place.kind === 'mirror' ? 2 : copyCount(b.place);
    const perDraw = b.shape.kind === 'superscope' ? 0.03 * Math.max(1, b.shape.p.n / 1024) : 0.03;
    return ms + 0.05 + perDraw * draws + (b.deform.kind === 'noise' ? 0.1 : 0);
  }
  const n = evalCount(b);
  let per = shapeEvalCost(b.shape) + deformCost(b) + (b.fuse ? SDF_COST[b.fuse.shape.kind] + 0.05 : 0) + 0.02;
  if (b.place.kind === 'grid' && b.place.p.links > 0.01) per += 0.1;
  ms += BODY_OVERHEAD + n * per + MATERIAL_COST[b.material.kind] * (b.place.p.fuse > 0 ? 1 : Math.min(n, 3)) + (b.emit.kind === 'cover' ? 0.03 : 0);
  return ms;
}
export const COST_BUDGET_MS = 8;
