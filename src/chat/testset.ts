// The gene chat test set: everyday requests against seed presets, each with the property the
// edited genome should have (not an exact edit: any sensible way to get there passes). Run in the
// browser with the model loaded: await __geneChatTest() (see testRunner.ts).

import { paletteHue, type Genome } from '../v2/genome';
import * as E from '../v2/geneEdit';
import { COLOUR_HUES, parseKindPath, parsePath, paramPaths } from './edits';
import type { LookMetrics } from './prompt';

export type Check = (before: Genome, after: Genome) => string | null;

export interface TestCase {
  id: string;
  /** Seed origin (E01..E24, M01..) the request starts from. */
  seed: string;
  request: string;
  expect: Check;
  /** Screen metrics the model is told (defaults to a medium, half-filled screen). */
  look?: LookMetrics;
  /** Written after the prompt was tuned and never tuned against: the honest pass rate. */
  heldOut?: boolean;
}

// ------------------------------------------------------------ probes

const EPS = 1e-6;
export function val(g: Genome, path: string): number {
  const pp = parsePath(path);
  const s = pp && E.schemaAt(g, pp.target)?.[pp.key];
  return pp && s ? E.getParam(g, pp.target, pp.key) : NaN;
}
function kindAt(g: Genome, path: string): string {
  const t = parseKindPath(path);
  if (!t) return '';
  if (t.t === 'locus') return (g.bodies[t.b]?.[t.locus] as { kind: string } | undefined)?.kind ?? '';
  if (t.t === 'palette') return g.palette.kind;
  if (t.t === 'carrier') return g.carrier.kind;
  if (t.t === 'op') return g.chain[t.j]?.op ?? '';
  return '';
}
/** Every path of the genome matching a pattern like "*.material.gain" or "op*.rate". */
function paths(g: Genome, pattern: string): string[] {
  const re = new RegExp(`^${pattern.replace(/\./g, '\\.').replace(/\*/g, '[A-Za-z0-9]*')}$`);
  return paramPaths(g).map((p) => p.path).filter((p) => re.test(p));
}
function moved(pattern: string, dir: 1 | -1, abs = false): Check {
  return (b, a) => {
    const ps = paths(b, pattern);
    for (const p of ps) {
      let x = val(b, p), y = val(a, p);
      if (!Number.isFinite(x) || !Number.isFinite(y)) continue;
      if (abs) [x, y] = [Math.abs(x), Math.abs(y)];
      if (dir > 0 ? y > x + EPS : y < x - EPS) return null;
    }
    return `no ${pattern} ${dir > 0 ? 'rose' : 'fell'}`;
  };
}
export const up = (pattern: string) => moved(pattern, 1);
export const down = (pattern: string) => moved(pattern, -1);
export const absDown = (pattern: string) => moved(pattern, -1, true);
export const absUp = (pattern: string) => moved(pattern, 1, true);
/** The gene at a kind path became one of these kinds (it was not one before). */
export const kindIs = (path: string, ...kinds: string[]): Check => (b, a) =>
  kinds.includes(kindAt(a, path)) && !kinds.includes(kindAt(b, path)) ? null : `${path} ${kindAt(b, path)} -> ${kindAt(a, path)}, wanted ${kinds.join('/')}`;
/** More bodies than before carry one of these kinds at a locus. */
export const anyBodyKind = (locus: string, ...kinds: string[]): Check => (b, a) => {
  const n = (g: Genome) => g.bodies.filter((x) => kinds.includes((x as unknown as Record<string, { kind: string }>)[locus].kind)).length;
  return n(a) > n(b) ? null : `no new body with ${locus} ${kinds.join('/')}`;
};
export const bodiesDelta = (d: number): Check => (b, a) => (a.bodies.length - b.bodies.length === d ? null : `bodies ${b.bodies.length} -> ${a.bodies.length}`);
export const chainGains = (...ops: string[]): Check => (b, a) => {
  const n = (g: Genome) => g.chain.filter((o) => ops.includes(o.op)).length + g.bodies.reduce((s, x) => s + (x.deform.ops ?? []).filter((o) => ops.includes(o.op)).length, 0);
  return n(a) > n(b) ? null : `no new ${ops.join('/')} op`;
};
export const newReaction = (...signals: string[]): Check => (b, a) => {
  const key = (r: Genome['reactions'][number]) => `${r.src}|${r.g}|${r.i}|${r.k}`;
  const old = new Set(b.reactions.map(key));
  const fresh = a.reactions.filter((r) => !old.has(key(r)) && (!signals.length || signals.includes(r.src)));
  if (fresh.length) return null;
  // Strengthening an existing reaction on that signal counts too.
  const stronger = a.reactions.some((r) => (!signals.length || signals.includes(r.src)) && b.reactions.some((q) => key(q) === key(r) && Math.abs(r.gain) > Math.abs(q.gain) + EPS));
  return stronger ? null : `no new or stronger ${signals.join('/') || ''} reaction`;
};
/** The first palette colour on screen (with the test key hue) is near a colour, or a body's colour is. */
export const hueNear = (colour: string, keyHue: number): Check => (_b, a) => {
  const target = COLOUR_HUES[colour];
  const near = (h: number) => Math.min(Math.abs(h - target), 1 - Math.abs(h - target)) < 0.09;
  const wrap = (h: number) => ((h % 1) + 1) % 1;
  // What is drawn: the palette's first slot, and each fixed-colour body's own hue (an offset from it).
  const hues = (g: Genome) => [wrap(paletteHue(g.palette.p, keyHue)), ...g.bodies.filter((x) => x.color.kind === 'fixed').map((x) => wrap(paletteHue(g.palette.p, keyHue) + (x.color.p.hue ?? 0)))];
  const h = hues(a);
  // It must move there (a preset already that colour proves nothing) and every drawn hue must agree.
  if (h.every(near) && JSON.stringify(h) !== JSON.stringify(hues(_b))) return null;
  return `hues on screen ${h.map((x) => x.toFixed(2)).join(', ')}, wanted ${colour} (${target})`;
};
export const unchanged: Check = (b, a) => (JSON.stringify(a) === JSON.stringify(b) ? null : 'the genome changed');
export const changed: Check = (b, a) => (JSON.stringify(a) !== JSON.stringify(b) ? null : 'nothing changed');
export function any(...cs: Check[]): Check {
  return (b, a) => {
    const why: string[] = [];
    for (const c of cs) {
      const r = c(b, a);
      if (!r) return null;
      why.push(r);
    }
    return why.join('; ');
  };
}
export function all(...cs: Check[]): Check {
  return (b, a) => {
    for (const c of cs) {
      const r = c(b, a);
      if (r) return r;
    }
    return null;
  };
}

// Composite looks.
const slower = any(absDown('op*.rate'), absDown('op*.lock'), absDown('op*.amt'), absDown('op*.speed'), absDown('op*.v*'), absDown('*.motion.rate'), absDown('*.place.rate'),
  absDown('*.place.step'), absDown('*.place.speed'), absDown('*.shape.speed'), up('*.motion.period'), absDown('*.motion.amp'), absDown('op*.w'), up('*.feel.div'), absDown('*.shape.flow'));
const faster = any(absUp('op*.rate'), absUp('op*.lock'), absUp('op*.amt'), absUp('op*.speed'), absUp('op*.v*'), absUp('*.motion.rate'), absUp('*.place.rate'),
  absUp('*.place.step'), absUp('*.place.speed'), absUp('*.shape.speed'), down('*.motion.period'), absUp('op*.w'), down('*.feel.div'), absUp('*.shape.flow'), kindIs('b0.motion', 'spin', 'circle', 'sway', 'drift'));
const brighter = any(up('tone.exposure'), up('*.material.gain'), up('tone.bloom'), down('carrier.floor'), up('*.material.base'), up('*.material.halo'), up('carrier.halfLife'));
const darker = any(down('tone.exposure'), down('*.material.gain'), down('tone.bloom'), up('carrier.floor'), up('tone.vignette'), down('carrier.halfLife'));
const longerTrails = any(up('carrier.halfLife'), kindIs('b0.emit', 'trail', 'cover'));
const shorterTrails = any(down('carrier.halfLife'), kindIs('b0.emit', 'none'));
const moreColour = any(up('tone.sat'), up('palette.spread'), kindIs('palette', 'triad', 'complementary', 'split', 'free'), up('*.color.amount'), anyBodyKind('color', 'instrument', 'pitch', 'melody', 'age', 'height', 'speed'));
const lessColour = any(down('tone.sat'), down('palette.spread'), kindIs('palette', 'mono', 'analogous'), down('*.color.amount'));
const bigger = any(up('*.shape.r'), up('*.shape.size'), up('*.shape.len'), up('*.shape.radius'), up('*.shape.amp'), up('*.shape.height'), up('*.shape.zoom'), up('*.material.width'), up('*.place.radius'), up('*.place.spread'));
const smaller = any(down('*.shape.r'), down('*.shape.size'), down('*.shape.len'), down('*.shape.radius'), down('*.shape.amp'), down('*.shape.height'), down('*.shape.zoom'), down('*.material.width'), down('*.place.radius'), down('*.place.spread'));
const fuller = any(up('*.shape.r'), up('*.shape.size'), up('*.shape.radius'), up('*.shape.zoom'), up('*.place.count'), up('*.place.radius'), up('*.place.spread'), up('carrier.halfLife'), up('*.material.width'),
  up('*.material.halo'), bodiesDelta(1), chainGains('tile', 'kaleido', 'mirror', 'polar'), kindIs('b0.place', 'grid', 'float', 'row', 'stations', 'orbit'), up('*.place.density'), up('*.shape.height'));
const emptier = any(down('*.place.count'), down('*.place.heads'), down('*.shape.r'), down('*.shape.size'), down('*.shape.radius'), down('carrier.halfLife'), bodiesDelta(-1), down('*.place.density'), down('*.material.halo'), down('*.material.gain'));
const sharper = any(up('carrier.sharpen'), down('carrier.blur'), down('*.material.halo'), down('*.material.width'), down('*.material.soft'), down('tone.bloom'), up('tone.contrast'));
const softer = any(up('carrier.blur'), up('*.material.halo'), up('*.material.soft'), up('tone.bloom'), down('carrier.sharpen'), kindIs('b0.material', 'glow'), up('*.material.width'));
const symmetric = any(chainGains('kaleido', 'mirror', 'polar', 'tile'), kindIs('b0.place', 'mirror', 'ring', 'grid'), up('op*.n'));
const lessSymmetric = (b: Genome, a: Genome): string | null => {
  const n = (g: Genome) => g.chain.filter((o) => ['kaleido', 'mirror', 'polar', 'tile'].includes(o.op)).length + g.bodies.filter((x) => ['mirror', 'ring'].includes(x.place.kind)).length;
  return n(a) < n(b) ? null : any(down('op*.n'), down('op*.w'))(b, a);
};
const punchier = any(newReaction('beat', 'drums', 'hit', 'barpulse', 'surge', 'bass'), up('reaction*.gain'), kindIs('b0.motion', 'pulse', 'hits', 'bob'), down('*.feel.atk'), up('*.feel.sens'), kindIs('b0.feel', 'step'));
const calmer = any(slower, up('carrier.halfLife'), down('reaction*.gain'), down('*.material.gain'), up('*.feel.rel'), down('*.feel.sens'));
const busier = any(faster, bodiesDelta(1), up('*.place.count'), chainGains(...['swirl', 'twist', 'ripple', 'noise', 'zoom', 'rotate']), newReaction(), up('*.emit.count'), kindIs('b0.emit', 'sparks'));

const KEY = 0.1;
const DIM: LookMetrics = { brightness: 0.04, coverage: 0.08, motion: 0.01, colourfulness: 0.5, hue: 0.6 };
const WASHED: LookMetrics = { brightness: 0.62, coverage: 0.9, motion: 0.02, colourfulness: 0.2, hue: 0.1 };
const SPARSE: LookMetrics = { brightness: 0.1, coverage: 0.06, motion: 0.01, colourfulness: 0.6, hue: 0.3 };

export const TEST_KEY_HUE = KEY;

export const TEST_SET: TestCase[] = [
  { id: 'calmer', seed: 'E05', request: 'make it calmer', expect: calmer },
  { id: 'chill', seed: 'M08', request: 'more chill please', expect: calmer },
  { id: 'busier', seed: 'E07', request: 'busier, more going on', expect: busier },
  { id: 'faster', seed: 'E14', request: 'make it faster', expect: faster },
  { id: 'slower', seed: 'E05', request: 'slow it down', expect: slower },
  { id: 'brighter', seed: 'E09', request: 'brighter', expect: brighter },
  { id: 'darker', seed: 'E16', request: 'a bit darker', expect: darker },
  { id: 'too-dark', seed: 'E13', request: "it's too dark", expect: brighter, look: DIM },
  { id: 'washed-out', seed: 'E08', request: 'this is washed out', expect: any(darker, lessColour, down('tone.exposure'), up('tone.contrast'), up('tone.sat')), look: WASHED },
  { id: 'fill-screen', seed: 'E13', request: 'fill more of the screen', expect: fuller, look: SPARSE },
  { id: 'emptier', seed: 'M04', request: 'less cluttered', expect: emptier },
  { id: 'more-colour', seed: 'E07', request: 'more colourful', expect: moreColour },
  { id: 'muted', seed: 'M09', request: 'more muted colours', expect: lessColour },
  { id: 'warmer', seed: 'E12', request: 'warmer colours', expect: any(hueNear('orange', KEY), hueNear('red', KEY), hueNear('amber', KEY), hueNear('yellow', KEY)) },
  { id: 'colder', seed: 'E23', request: 'make it colder', expect: any(hueNear('blue', KEY), hueNear('cyan', KEY), hueNear('teal', KEY)) },
  { id: 'blue', seed: 'E05', request: 'make it blue', expect: hueNear('blue', KEY) },
  { id: 'red', seed: 'E13', request: 'turn it red', expect: hueNear('red', KEY) },
  { id: 'green', seed: 'E18', request: 'green please', expect: hueNear('green', KEY) },
  { id: 'purple', seed: 'E11', request: 'purple', expect: any(hueNear('purple', KEY), hueNear('violet', KEY), hueNear('magenta', KEY)) },
  { id: 'bigger', seed: 'E14', request: 'make the shape bigger', expect: bigger },
  { id: 'smaller', seed: 'E10', request: 'smaller', expect: smaller },
  { id: 'sharper', seed: 'E16', request: 'sharper, crisper lines', expect: sharper },
  { id: 'softer', seed: 'E18', request: 'softer and blurrier', expect: softer },
  { id: 'longer-trails', seed: 'E13', request: 'longer trails', expect: longerTrails },
  { id: 'shorter-trails', seed: 'E02', request: 'shorter trails', expect: shorterTrails },
  { id: 'no-trails', seed: 'E07', request: 'no trails at all', expect: any(shorterTrails, kindIs('b0.emit', 'none')) },
  { id: 'dreamy', seed: 'E14', request: 'dreamy', expect: any(softer, longerTrails, slower) },
  { id: 'underwater', seed: 'E09', request: 'make it feel underwater', expect: any(hueNear('blue', KEY), hueNear('teal', KEY), hueNear('cyan', KEY), chainGains('ripple', 'noise'), up('op*.amp'), kindIs('carrier', 'fluid', 'flow')) },
  { id: 'fire', seed: 'E04', request: 'like fire', expect: any(hueNear('orange', KEY), hueNear('red', KEY), hueNear('yellow', KEY)) },
  { id: 'space', seed: 'E07', request: 'more like outer space with stars', expect: any(bodiesDelta(1), anyBodyKind('emit', 'sparks'), anyBodyKind('place', 'grid'), chainGains('zoom')) },
  { id: 'trippy', seed: 'E09', request: 'more psychedelic', expect: any(symmetric, moreColour, chainGains('swirl', 'twist', 'quad', 'ripple'), anyBodyKind('color', 'age', 'melody', 'pitch')) },
  { id: 'minimal', seed: 'M01', request: 'more minimal and clean', expect: any(emptier, lessColour, bodiesDelta(-1), (b, a) => (a.chain.length < b.chain.length ? null : 'chain not shorter')) },
  { id: 'aggressive', seed: 'E06', request: 'more aggressive', expect: any(punchier, faster, sharper, up('tone.contrast')) },
  { id: 'punchier', seed: 'E13', request: 'hit harder on the beat', expect: punchier },
  { id: 'bass', seed: 'E14', request: 'react to the bass', expect: any(newReaction('bass'), kindIs('b0.motion', 'pulse')) },
  { id: 'vocals', seed: 'E16', request: 'follow the vocals', expect: any(newReaction('vocals', 'melody'), anyBodyKind('color', 'melody')) },
  { id: 'drop', seed: 'E05', request: 'go wild on the drop', expect: any(newReaction('drop', 'build', 'surge', 'section')) },
  { id: 'kaleido', seed: 'E07', request: 'kaleidoscope', expect: any(chainGains('kaleido'), kindIs('b0.place', 'mirror', 'ring')) },
  { id: 'symmetry', seed: 'E02', request: 'more symmetry', expect: symmetric },
  { id: 'less-symmetry', seed: 'E18', request: 'less symmetric', expect: lessSymmetric },
  { id: 'spin', seed: 'E10', request: 'make it spin', expect: any(kindIs('b0.motion', 'spin'), chainGains('rotate'), absUp('op*.lock'), absUp('op*.rate'), absUp('b0.motion.rate')) },
  { id: 'stop-spin', seed: 'E14', request: 'stop spinning', expect: any(kindIs('b0.motion', 'none', 'bob', 'pulse', 'sway'), absDown('b0.motion.rate'), absDown('op*.lock'), absDown('op*.rate'), (b, a) => (a.chain.filter((o) => o.op === 'rotate').length < b.chain.filter((o) => o.op === 'rotate').length ? null : 'rotate kept')) },
  { id: 'sparks', seed: 'E05', request: 'add sparks', expect: any(anyBodyKind('emit', 'sparks'), up('*.emit.count')) },
  { id: 'glow', seed: 'E10', request: 'make it glow', expect: any(kindIs('b0.material', 'glow'), up('*.material.halo'), up('tone.bloom'), up('*.material.base')) },
  { id: 'star-shape', seed: 'E10', request: 'change the shape to a star', expect: anyBodyKind('shape', 'star') },
  { id: 'add-layer', seed: 'E16', request: 'add a second layer of little dots', expect: all(bodiesDelta(1), anyBodyKind('shape', 'dot')) },
  { id: 'remove-layer', seed: 'E21', request: 'remove the second layer', expect: bodiesDelta(-1) },
  { id: 'swirl', seed: 'E13', request: 'add a swirl', expect: any(chainGains('swirl'), chainGains('twist')) },
  { id: 'fluid', seed: 'E07', request: 'make the light flow like liquid', expect: any(kindIs('carrier', 'fluid', 'flow'), chainGains('ripple', 'noise', 'swirl'), anyBodyKind('emit', 'dye')) },
  { id: 'staccato', seed: 'E17', request: 'jerkier, more staccato', expect: any(kindIs('b0.feel', 'step'), down('*.feel.atk'), down('*.feel.rel'), up('reaction*.q')) },
  { id: 'smoother', seed: 'M04', request: 'smoother motion', expect: any(up('*.feel.rel'), up('*.feel.atk'), kindIs('b0.feel', 'flow'), slower, up('carrier.halfLife')) },
  { id: 'nonsense', seed: 'E05', request: "what's the weather like tomorrow?", expect: unchanged },
  { id: 'sliders', seed: 'E08', request: 'set the exposure to 1.2', expect: (_b, a) => (Math.abs(a.tone.p.exposure - 1.2) < 1e-6 ? null : `exposure ${a.tone.p.exposure}`) },

  // Held out: phrasings the prompt and lexicon were never tuned on.
  { id: 'h-kick', seed: 'E13', request: 'can you make it pulse with the kick drum', expect: punchier, heldOut: true },
  { id: 'h-less-busy', seed: 'M08', request: 'this is way too busy', expect: any(calmer, emptier), heldOut: true },
  { id: 'h-slow-blue', seed: 'E12', request: 'slower and bluer', expect: all(slower, any(hueNear('blue', KEY), hueNear('cyan', KEY), hueNear('indigo', KEY))), heldOut: true },
  { id: 'h-contrast', seed: 'E16', request: 'I want more contrast', expect: any(up('tone.contrast'), sharper), heldOut: true },
  { id: 'h-sunset', seed: 'E14', request: 'make it look like a sunset', expect: any(hueNear('orange', KEY), hueNear('red', KEY), hueNear('pink', KEY), hueNear('amber', KEY)), heldOut: true },
  { id: 'h-too-big', seed: 'E15', request: 'the blobs are too big', expect: smaller, heldOut: true },
  { id: 'h-rotation', seed: 'E13', request: 'add some rotation', expect: any(chainGains('rotate'), kindIs('b0.motion', 'spin'), absUp('op*.lock'), absUp('op*.rate')), heldOut: true },
  { id: 'h-fade-faster', seed: 'E02', request: 'make the trails fade faster', expect: shorterTrails, heldOut: true },
  { id: 'h-hypnotic', seed: 'E10', request: 'make it more hypnotic', expect: changed, heldOut: true },
  { id: 'h-thanks', seed: 'E07', request: 'thanks, that looks great!', expect: unchanged, heldOut: true },
  { id: 'h-wider', seed: 'E05', request: 'spread the dots out wider', expect: any(up('*.place.radius'), up('*.place.spread'), up('*.place.xs'), up('*.place.follow')), heldOut: true },
  { id: 'h-less-glow', seed: 'E16', request: 'tone down the glow', expect: any(down('tone.bloom'), down('*.material.gain'), down('*.material.halo'), down('*.material.width'), down('*.material.base')), heldOut: true },
];
