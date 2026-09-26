// Which seeds use which signals and genome inputs, to weight each parity gap by the presets it
// affects. Reactions name their signal directly; genes are mapped to the MusicState fields they
// read (from src/v2/genes/*.ts and the engine's Signals.update). Every preset also reads a core
// set through the engine itself (activity, speed, spin, hits, surge, the built-in accents).

import type { EngineBundle } from './bundle';

type Genome = EngineBundle['SEEDS'][number]['genome'];

const TIMBRE = ['mix', 'drums', 'bass', 'vocals', 'other'].flatMap((p) => ['bright', 'noise', 'rough', 'attack'].map((k) => `timbre.${p}.${k}`));
const GROOVE = ['swing', 'push', 'humanity', 'synco'].map((k) => `groove.${k}`);
const NOTES = ['notes.on', 'notes.held', 'notes.legato', 'notes.pitch', 'notes.height', 'notes.voice', 'notes.vibrato', 'notes.glide'];
const HARMONY = ['chord', 'tension', 'chordPulse', 'resolvePulse', 'modulationPulse', 'key'];
const STEMS = ['drums', 'bass', 'vocals', 'other'];

/** Fields every preset reads through the engine (Signals.update, accents, speed, spin, hit). */
export const CORE_INPUTS = new Set([
  'complexity', 'buildIntensity', 'dropPulse', 'beatPhase', 'barPhase', 'onBeat', 'loudness',
  ...STEMS.map((s) => `stems.${s}`), ...STEMS.map((s) => `stemPresence.${s}`), 'stemOnsets.drums',
  // Built-in accents (src/v2/genes/accent.ts): the hook gesture and the section cue.
  'hookOn', 'hookPulse', 'hookNotePulse', 'section.class', 'sectionChanged',
]);

/** Genome inputs a seed reads through its genes (not counting the core set). */
export function geneInputs(g: Genome): Set<string> {
  const out = new Set<string>();
  const add = (...k: string[]) => k.forEach((x) => out.add(x));
  if (g.choreo) add('section.class', 'section.label', 'sectionChanged', 'timeToDrop', 'prevSectionLabel', 'barPhase', 'dropPulse');
  if (g.drift) add('section.label', 'sectionChanged');
  if (g.harmony) add(...HARMONY);
  if (g.groove) add(...GROOVE, 'beatPhase');
  if (g.timbre) add(...TIMBRE);
  if (g.dejavu) add('repeat (dejavu)', 'section.label');
  if (g.lyrics) add('lyrics');
  if (g.accent) add('hookOn', 'hookPulse', 'hookNotePulse', 'hookPhase');
  for (const b of g.bodies) {
    const kind = b.shape.kind as string;
    if (kind === 'notes') add(...NOTES);
    if (kind === 'tonnetz') add('chord', 'tension', 'chordPulse', 'resolvePulse', 'key');
    if (kind === 'cymatics') add('key');
    if (kind === 'beams') add('hookNotePulse', 'dropPulse');
    if (kind === 'landscape') add('timeToDrop', ...STEMS.map((s) => `stemOnsets.${s}`));
    if (kind === 'scene') add(...STEMS.map((s) => `stemOnsets.${s}`));
    if ((b.motion.kind as string) === 'hits') add('onBeat');
    if ((b.motion.kind as string) === 'pulse' || (b.feel.kind as string) === 'step') add('beatPhase');
  }
  return out;
}

export interface Usage {
  /** Signal -> number of seeds with at least one reaction on it, and total reactions. */
  signalSeeds: Map<string, number>;
  signalReactions: Map<string, number>;
  /** Genome input -> number of seeds reading it through genes. */
  inputSeeds: Map<string, number>;
  seeds: number;
  /** Per seed: its reaction signals and gene inputs. */
  perSeed: Map<string, { signals: string[]; inputs: string[] }>;
}

export function usage(E: EngineBundle): Usage {
  const signalSeeds = new Map<string, number>();
  const signalReactions = new Map<string, number>();
  const inputSeeds = new Map<string, number>();
  const perSeed = new Map<string, { signals: string[]; inputs: string[] }>();
  const inc = (m: Map<string, number>, k: string) => m.set(k, (m.get(k) ?? 0) + 1);
  for (const s of E.SEEDS) {
    const sigs = new Set<string>();
    for (const r of s.genome.reactions) {
      inc(signalReactions, r.src);
      sigs.add(r.src);
    }
    for (const x of sigs) inc(signalSeeds, x);
    const ins = geneInputs(s.genome);
    for (const x of ins) inc(inputSeeds, x);
    perSeed.set(s.origin, { signals: [...sigs], inputs: [...ins] });
  }
  return { signalSeeds, signalReactions, inputSeeds, seeds: E.SEEDS.length, perSeed };
}

/** The genome input a signal is derived from (to attribute signal gaps to analysis fields). */
export const SIGNAL_SOURCE: Record<string, string> = {
  drums: 'stems', bass: 'stems', vocals: 'stems', other: 'stems', hit: 'stemOnsets / beat', beat: 'beat grid', bar: 'barPhase',
  complexity: 'complexity', drop: 'structure (drop)', loud: 'loudness', melody: 'spectrum (live analyser)', build: 'structure (build)',
  surge: 'beat + loudness + drop', barpulse: 'beat grid (downbeat)', section: 'structure (sections)', tension: 'harmony', resolve: 'harmony',
  chordchange: 'harmony', modulation: 'harmony (key)', swing: 'groove', push: 'groove', humanity: 'groove', synco: 'groove',
  line: 'lyrics / vocals fallback', valence: 'lyrics / key fallback', arousal: 'lyrics / activity fallback', bright: 'timbre', noisy: 'timbre',
  rough: 'timbre', attack: 'timbre', noteon: 'notes', held: 'notes', legato: 'notes', glide: 'notes', vibrato: 'notes', voice: 'notes',
  hook: 'hooks', hookphase: 'hooks', hookon: 'hooks',
};
