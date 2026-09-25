// Tests for the harmony map (src/analysis/harmony.ts): chord recognition accuracy on synthetic
// progressions with known answers (chroma and audio), tension ordering, cadence and modulation
// detection, the realtime tracker. Called from analysis-test.ts.

import {
  CADENCE, HarmonyTracker, MOVE, analyzeHarmony, cadenceKind, chordName, chordTension, keyFifths, latticeSteps, makeChord,
  moveKind, fifthDirection, roughness, tonnetzXY, triadDistance, type ChordIndex, type HarmonyInput,
} from '../src/analysis/harmony';
import { computeChroma, resampleChroma } from '../src/analysis/key';
import type { AnalysisResult, KeySegment, LiveAudioFrame } from '../src/types';
import { TimelineSampler } from '../src/analysis/TimelineSampler';

type Check = (name: string, ok: boolean, detail: string) => void;

function rng(seed: number): () => number {
  let s = seed >>> 0;
  return () => {
    s = (s + 0x6d2b79f5) >>> 0;
    let t = s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const NOTE: Record<string, number> = { C: 0, 'C#': 1, Db: 1, D: 2, Eb: 3, E: 4, F: 5, 'F#': 6, Gb: 6, G: 7, Ab: 8, A: 9, Bb: 10, B: 11 };
/** 'Am' -> chord index. */
function ch(name: string): ChordIndex {
  const minor = name.endsWith('m');
  return makeChord(NOTE[minor ? name.slice(0, -1) : name], minor);
}

const BEAT = 0.5;
const FR = 50;

interface Prog {
  name: string;
  chords: string[]; // one per step
  beats: number[]; // beats per step (cycled)
  key: KeySegment[] | { tonic: number; mode: 'major' | 'minor' };
  sevenths?: boolean;
}

/** Frame chroma of a progression: chord tones with random levels, the root doubled, noise, passing tones. */
function chromaSong(p: Prog, seed: number, noise: number): { inp: HarmonyInput; truth: ChordIndex[] } {
  const r = rng(seed);
  const truth: ChordIndex[] = [];
  p.chords.forEach((c, i) => {
    for (let b = 0; b < p.beats[i % p.beats.length]; b++) truth.push(ch(c));
  });
  const nb = truth.length;
  const duration = nb * BEAT;
  const T = Math.round(duration * FR);
  const chroma = new Float32Array(T * 12);
  const loud = new Float32Array(T).fill(0.7);
  const v = new Float64Array(12);
  for (let f = 0; f < T; f++) {
    const bi = Math.min(nb - 1, Math.floor(f / FR / BEAT));
    const c = truth[bi];
    const root = c % 12;
    const tones = [root, (root + (c >= 12 ? 3 : 4)) % 12, (root + 7) % 12];
    for (let k = 0; k < 12; k++) v[k] = noise * r();
    v[tones[0]] += 0.9 + 0.3 * r();
    v[tones[1]] += 0.55 + 0.4 * r();
    v[tones[2]] += 0.55 + 0.4 * r();
    if (p.sevenths) v[(root + (c >= 12 ? 10 : 10)) % 12] += 0.45;
    // A passing melody note now and then.
    if (r() < 0.3) v[Math.floor(r() * 12)] += 0.5 * r();
    let m = 0;
    for (let k = 0; k < 12; k++) m = Math.max(m, v[k]);
    for (let k = 0; k < 12; k++) chroma[f * 12 + k] = v[k] / m;
  }
  const beats = Float32Array.from({ length: nb }, (_, i) => i * BEAT);
  const keys: KeySegment[] = Array.isArray(p.key) ? p.key : [{ start: 0, end: duration, tonic: p.key.tonic, mode: p.key.mode, confidence: 1 }];
  return { inp: { duration, frameRate: FR, numFrames: T, chroma, loudness: loud, beats, keys }, truth };
}

/** Share of beats whose recognized chord matches the truth. */
function accuracy(inp: HarmonyInput, truth: ChordIndex[]): { acc: number; got: string } {
  const h = analyzeHarmony(inp);
  let ok = 0;
  const got: string[] = [];
  for (let i = 0; i < truth.length; i++) {
    const t = i * BEAT + 1e-3;
    let k = 0;
    while (k + 1 < h.times.length && h.times[k + 1] <= t) k++;
    if (h.chords[k] === truth[i]) ok++;
    if (i % 2 === 0) got.push(chordName(h.chords[k]));
  }
  return { acc: ok / truth.length, got: got.join(' ') };
}

const PROGS: Prog[] = [
  { name: 'pop I-V-vi-IV in C', chords: ['C', 'G', 'Am', 'F', 'C', 'G', 'Am', 'F'], beats: [4], key: { tonic: 0, mode: 'major' } },
  { name: 'jazz ii-V-I in Bb (sevenths)', chords: ['Cm', 'F', 'Bb', 'Bb', 'Cm', 'F', 'Bb', 'Gm'], beats: [4, 4, 8, 4], key: { tonic: 10, mode: 'major' }, sevenths: true },
  { name: 'minor i-iv-V-i in A minor', chords: ['Am', 'Dm', 'E', 'Am', 'F', 'G', 'E', 'Am'], beats: [2, 2, 4], key: { tonic: 9, mode: 'minor' } },
  { name: 'all 24 triads, 2 beats each', chords: ['C', 'Cm', 'Db', 'C#m', 'D', 'Dm', 'Eb', 'Ebm', 'E', 'Em', 'F', 'Fm', 'F#', 'F#m', 'G', 'Gm', 'Ab', 'Abm', 'A', 'Am', 'Bb', 'Bbm', 'B', 'Bm'], beats: [2], key: { tonic: 0, mode: 'major' } },
  { name: 'chromatic mediants', chords: ['C', 'E', 'Ab', 'C', 'Eb', 'B', 'G', 'C'], beats: [4], key: { tonic: 0, mode: 'major' } },
];

// ------------------------------------------------------------------ audio

const SR = 22050;
const midiHz = (m: number) => 440 * Math.pow(2, (m - 69) / 12);

/** A pad (three chord tones, a few harmonics) over a bass root, one chord per bar at 120 BPM. */
function audioSong(chords: string[]): { x: Float32Array; duration: number; truth: ChordIndex[] } {
  const BAR = 4 * BEAT;
  const duration = chords.length * BAR;
  const x = new Float32Array(Math.ceil(duration * SR));
  const r = rng(99);
  chords.forEach((name, ci) => {
    const c = ch(name);
    const root = c % 12;
    const notes = [60 + root, 60 + root + (c >= 12 ? 3 : 4), 60 + root + 7].map((m) => (m > 71 ? m - 12 : m));
    const s0 = Math.round(ci * BAR * SR);
    const s1 = Math.round((ci + 1) * BAR * SR);
    const ph = notes.map(() => r() * 6.28);
    for (let i = s0; i < s1 && i < x.length; i++) {
      const t = (i - s0) / SR;
      const env = Math.min(1, t / 0.02) * Math.min(1, (s1 - i) / SR / 0.02);
      let v = 0;
      notes.forEach((m, k) => {
        const w = 2 * Math.PI * midiHz(m) * t + ph[k];
        v += 0.18 * (Math.sin(w) + 0.4 * Math.sin(2 * w) + 0.2 * Math.sin(3 * w));
      });
      const wb = 2 * Math.PI * midiHz(36 + root) * t;
      v += 0.25 * (Math.sin(wb) + 0.5 * Math.sin(2 * wb));
      // A little noise (hats) on the off-beats.
      if (((t / BEAT) % 1) > 0.5 && ((t / BEAT) % 1) < 0.53) v += (r() * 2 - 1) * 0.15;
      x[i] = v * env;
    }
  });
  const truth: ChordIndex[] = [];
  for (const name of chords) for (let b = 0; b < 4; b++) truth.push(ch(name));
  return { x, duration, truth };
}

export function harmonyTests(check: Check): void {
  // --- Chord recognition accuracy on synthetic chroma ---
  let total = 0;
  let n = 0;
  for (const [i, p] of PROGS.entries()) {
    for (const noise of [0.15, 0.35, 0.7]) {
      const { inp, truth } = chromaSong(p, 11 + i * 7 + noise * 100, noise);
      const { acc, got } = accuracy(inp, truth);
      total += acc;
      n++;
      check(`harmony: ${p.name}, noise ${noise}: accuracy ${(acc * 100).toFixed(0)}%`, acc >= (noise > 0.5 ? 0.7 : noise > 0.3 ? 0.85 : 0.92), got);
    }
  }
  check(`harmony: mean chroma accuracy ${((total / n) * 100).toFixed(1)}% >= 90%`, total / n >= 0.9, '');

  // --- Chord recognition on synthesized audio (the real chroma front end) ---
  {
    const chords = ['C', 'Am', 'F', 'G', 'Em', 'Am', 'Dm', 'G', 'C', 'E', 'Am', 'C'];
    const { x, duration, truth } = audioSong(chords);
    const hop = 256;
    const c = computeChroma(x, SR, 4096, 1024);
    const T = Math.floor((x.length - 1) / hop) + 1;
    const frameRate = SR / hop;
    const chroma = resampleChroma(c, T, frameRate);
    const beats = Float32Array.from({ length: truth.length }, (_, i) => i * BEAT);
    const inp: HarmonyInput = { duration, frameRate, numFrames: T, chroma, beats, keys: [{ start: 0, end: duration, tonic: 0, mode: 'major', confidence: 1 }] };
    const { acc, got } = accuracy(inp, truth);
    check(`harmony: audio progression accuracy ${(acc * 100).toFixed(0)}% >= 85%`, acc >= 0.85, got);
  }

  // --- Silence and noise give no chord ---
  {
    const T = 500;
    const inp: HarmonyInput = { duration: 10, frameRate: 50, numFrames: T, chroma: new Float32Array(T * 12), beats: Float32Array.from({ length: 20 }, (_, i) => i * 0.5), keys: [] };
    const h = analyzeHarmony(inp);
    check('harmony: silence is no chord, zero tension', h.chords.every((c) => c === -1) && h.tension.every((t) => t === 0), [...h.chords].join(','));
    const flat = new Float32Array(T * 12);
    const r = rng(5);
    for (let i = 0; i < flat.length; i++) flat[i] = 0.8 + 0.2 * r();
    const h2 = analyzeHarmony({ ...inp, chroma: flat });
    const none = h2.chords.filter((c) => c === -1).length / h2.chords.length;
    check('harmony: flat (atonal) chroma is mostly no chord', none >= 0.8, `${(none * 100).toFixed(0)}%`);
  }

  // --- Tension ordering ---
  {
    const C = { tonic: 0, mode: 'major' as const };
    const t = (c: string) => chordTension(ch(c), C, 0);
    const order = ['C', 'Am', 'F', 'G', 'F#'].map(t);
    check('harmony: tension I < vi < IV < V < tritone chord', order.every((v, i) => i === 0 || v > order[i - 1]), order.map((v) => v.toFixed(2)).join(' < '));
    check('harmony: tonic triad has zero tension', t('C') === 0, String(t('C')));
    const am = { tonic: 9, mode: 'minor' as const };
    check('harmony: minor key: i < iv < V', chordTension(ch('Am'), am, 0) < chordTension(ch('Dm'), am, 0) && chordTension(ch('Dm'), am, 0) < chordTension(ch('E'), am, 0), '');
    // Roughness: a clean triad vs the same triad with semitone clashes.
    const clean = [1, 0, 0, 0, 0.8, 0, 0, 0.8, 0, 0, 0, 0];
    const rough = [1, 0.7, 0, 0, 0.8, 0.6, 0, 0.8, 0.6, 0, 0, 0];
    const rc = roughness(clean, ch('C'));
    const rr = roughness(rough, ch('C'));
    check('harmony: roughness of a clean triad ~0, a clashing one high', rc < 0.01 && rr > 0.3, `${rc.toFixed(2)} ${rr.toFixed(2)}`);
    check('harmony: dissonance raises tension', chordTension(ch('C'), C, rr) > chordTension(ch('C'), C, rc) + 0.05, '');
  }

  // --- Lattice ---
  {
    check('harmony: lattice steps: fifth 1, major third 1, minor third 1, whole tone 2, tritone 3', latticeSteps(0, 7) === 1 && latticeSteps(0, 4) === 1 && latticeSteps(0, 3) === 1 && latticeSteps(0, 2) === 2 && latticeSteps(0, 6) === 3, '');
    check('harmony: PLR neighbours of C: Cm, Am, Em; G is 2 away', triadDistance(ch('C'), ch('Cm')) === 1 && triadDistance(ch('C'), ch('Am')) === 1 && triadDistance(ch('C'), ch('Em')) === 1 && triadDistance(ch('C'), ch('G')) === 2, '');
    const a = tonnetzXY(ch('C'), 0);
    const g = tonnetzXY(ch('G'), 0);
    const f = tonnetzXY(ch('F'), 0);
    check('harmony: on the Tonnetz G is one step right of C, F one step left', Math.abs(g[0] - a[0] - 1) < 1e-6 && Math.abs(a[0] - f[0] - 1) < 1e-6 && Math.abs(g[1] - a[1]) < 1e-6, `${a} ${g} ${f}`);
    check('harmony: minor triads point down (centre below the major one on the same root)', tonnetzXY(ch('Cm'), 0)[1] < a[1], '');
    check('harmony: motion kinds', moveKind(ch('G'), ch('C')) === MOVE.fifth && moveKind(ch('C'), ch('Am')) === MOVE.third && moveKind(ch('C'), ch('Dm')) === MOVE.step && moveKind(ch('C'), ch('F#')) === MOVE.tritone && moveKind(ch('C'), ch('Cm')) === MOVE.mode, '');
    check('harmony: G -> C falls a fifth (+1), C -> G rises (-1)', fifthDirection(ch('G'), ch('C')) === 1 && fifthDirection(ch('C'), ch('G')) === -1, '');
    check('harmony: key fifths C->G +1, C->F -1, C->Am 0, C->D +2', keyFifths({ tonic: 0, mode: 'major' }, { tonic: 7, mode: 'major' }) === 1 && keyFifths({ tonic: 0, mode: 'major' }, { tonic: 5, mode: 'major' }) === -1 && keyFifths({ tonic: 0, mode: 'major' }, { tonic: 9, mode: 'minor' }) === 0 && keyFifths({ tonic: 0, mode: 'major' }, { tonic: 2, mode: 'major' }) === 2, '');
  }

  // --- Cadences ---
  {
    const C = { tonic: 0, mode: 'major' as const };
    check('harmony: cadence kinds', cadenceKind(ch('G'), ch('C'), C) === CADENCE.authentic && cadenceKind(ch('F'), ch('C'), C) === CADENCE.plagal && cadenceKind(ch('Am'), ch('C'), C) === CADENCE.return && cadenceKind(ch('C'), ch('G'), C) === CADENCE.none, '');
    const res = (p: Prog) => analyzeHarmony(chromaSong(p, 3, 0.15).inp).resolutions;
    const auth = res({ name: '', chords: ['C', 'F', 'Dm', 'G', 'C'], beats: [4], key: C });
    check('harmony: V-I at 16 beats is detected as an authentic resolution', auth.length === 1 && auth[0].kind === CADENCE.authentic && Math.abs(auth[0].time - 8) < 0.01 && auth[0].strength >= 0.5, JSON.stringify(auth));
    const plag = res({ name: '', chords: ['C', 'Am', 'F', 'C'], beats: [4], key: C });
    check('harmony: IV-I is a plagal resolution, weaker than V-I', plag.length === 1 && plag[0].kind === CADENCE.plagal && plag[0].strength < auth[0].strength, JSON.stringify(plag));
    const none = res({ name: '', chords: ['C', 'Am', 'F', 'Am', 'F', 'Am'], beats: [4], key: C });
    check('harmony: a progression that never returns home has no resolutions', none.length === 0, JSON.stringify(none));
    const minor = res({ name: '', chords: ['Am', 'Dm', 'E', 'Am'], beats: [4], key: { tonic: 9, mode: 'minor' } });
    check('harmony: minor V-i resolves', minor.length === 1 && minor[0].kind === CADENCE.authentic, JSON.stringify(minor));
    const h = analyzeHarmony(chromaSong({ name: '', chords: ['C', 'F', 'G', 'C'], beats: [4], key: C }, 4, 0.15).inp);
    const segT = h.segments.map((s) => s.tension);
    check('harmony: segment tension rises to V and falls on the return', segT.length === 4 && segT[2] > segT[1] && segT[1] > segT[0] && segT[3] < segT[2], segT.map((v) => v.toFixed(2)).join(' '));
    check('harmony: segment motion G->C is a falling fifth', h.segments[3].move === MOVE.fifth && h.segments[3].fifths === 1, '');
  }

  // --- Modulations ---
  {
    const keys: KeySegment[] = [
      { start: 0, end: 8, tonic: 0, mode: 'major', confidence: 1 },
      { start: 8, end: 16, tonic: 2, mode: 'major', confidence: 1 },
    ];
    const { inp } = chromaSong({ name: '', chords: ['C', 'G', 'A', 'D'], beats: [8], key: keys }, 8, 0.15);
    const h = analyzeHarmony(inp);
    check('harmony: a key change is a modulation two fifths up', h.modulations.length === 1 && h.modulations[0].fifths === 2 && h.modulations[0].time === 8, JSON.stringify(h.modulations));
    // A-> D in the new key is V-I there.
    check('harmony: cadences follow the new key after a modulation', h.resolutions.some((r) => Math.abs(r.time - 12) < 0.01), JSON.stringify(h.resolutions));
  }

  // --- Key vote: a mis-detected key is corrected by the chords ---
  {
    const wrong: KeySegment[] = [{ start: 0, end: 32, tonic: 6, mode: 'minor', confidence: 0.5 }];
    const { inp } = chromaSong({ name: '', chords: ['D', 'F#m', 'G', 'A', 'D', 'Bm', 'G', 'A'], beats: [8], key: wrong }, 9, 0.2);
    const h = analyzeHarmony(inp);
    const k = h.keys[0];
    check('harmony: chords vote D major over a detected F# minor', h.keys.length === 1 && k.tonic === 2 && k.mode === 'major', JSON.stringify(h.keys));
    check('harmony: after the vote A -> D resolves (authentic)', h.resolutions.some((r) => r.kind === CADENCE.authentic && Math.abs(r.time - 16) < 0.01), JSON.stringify(h.resolutions));
  }

  // --- Realtime tracker ---
  {
    const p: Prog = { name: '', chords: ['C', 'F', 'G', 'C', 'Am', 'F', 'G', 'C'], beats: [8], key: { tonic: 0, mode: 'major' } };
    const { inp, truth } = chromaSong(p, 21, 0.2);
    const trk = new HarmonyTracker();
    trk.setKey(0, 'major');
    let ok = 0;
    let counted = 0;
    const dt = 1 / FR;
    for (let f = 0; f < inp.numFrames; f++) {
      trk.push(inp.chroma.subarray(f * 12, f * 12 + 12), 0.8, dt);
      const bi = Math.floor(f * dt / BEAT);
      // Skip the first beat of each chord (the tracker needs ~0.4 s to switch).
      if (bi % 8 === 0) continue;
      counted++;
      if (trk.chord === truth[bi]) ok++;
    }
    const acc = ok / counted;
    check(`harmony: realtime tracker accuracy ${(acc * 100).toFixed(0)}% >= 90%`, acc >= 0.9, '');
    check('harmony: realtime tracker counts 8 chord arrivals and 2 resolutions', trk.changes === 8 && trk.resolves === 2, `changes ${trk.changes} resolves ${trk.resolves}`);
    trk.setKey(7, 'major');
    check('harmony: realtime tracker counts a key change', trk.modulations === 1, String(trk.modulations));
  }

  // --- MusicState through the TimelineSampler ---
  {
    const C = { tonic: 0, mode: 'major' as const };
    const { inp } = chromaSong({ name: '', chords: ['C', 'F', 'G', 'C', 'Am', 'Dm', 'G', 'C'], beats: [4], key: C }, 6, 0.2);
    const T = inp.numFrames;
    const z = () => new Float32Array(T);
    const st = () => ({ drums: z(), bass: z(), vocals: z(), other: z() });
    const r: AnalysisResult = {
      duration: inp.duration, frameRate: inp.frameRate, numFrames: T, stems: st(), stemOnsets: st(), stemPresence: st(), complexity: z(), songComplexity: 0.5,
      loudness: inp.loudness!, chroma: inp.chroma, bpm: 120, beats: Float32Array.from(inp.beats), downbeats: Float32Array.from(inp.beats).filter((_b, i) => i % 4 === 0), beatsPerBar: 4,
      sections: [{ start: 0, end: inp.duration, label: 'verse', energy: 0.5 }], keys: inp.keys,
    };
    const smp = new TimelineSampler(r);
    const live: LiveAudioFrame = { bass: 1, mid: 1, treb: 1, bassAtt: 1, midAtt: 1, trebAtt: 1, waveform: new Float32Array(1024), spectrum: new Float32Array(512) };
    let changes = 0;
    const resolveAt: number[] = [];
    let maxT = 0;
    let tAtG = 0;
    let prevRes = 0;
    let prevCh = 0;
    for (let t = 0; t < inp.duration; t += 1 / 60) {
      const s = smp.sample(t, 1 / 60, true, live);
      if ((s.chordPulse ?? 0) > prevCh + 0.5) changes++;
      if ((s.resolvePulse ?? 0) > prevRes + 0.3) resolveAt.push(t);
      prevRes = s.resolvePulse ?? 0;
      prevCh = s.chordPulse ?? 0;
      maxT = Math.max(maxT, s.tension ?? 0);
      if (Math.abs(t - 5.9) < 0.01) tAtG = s.tension ?? 0;
    }
    check('harmony: sampler fires a chord pulse on each of the 7 changes', changes === 7, String(changes));
    check('harmony: sampler fires resolve pulses at the two V-I arrivals (6 s, 14 s)', resolveAt.length === 2 && Math.abs(resolveAt[0] - 6) < 0.05 && Math.abs(resolveAt[1] - 14) < 0.05, resolveAt.map((v) => v.toFixed(2)).join(','));
    check('harmony: sampler tension is high on the V, in 0..1', tAtG > 0.3 && maxT <= 1, `${tAtG.toFixed(2)} max ${maxT.toFixed(2)}`);
    const s0 = smp.sample(3, 0, true, live);
    check('harmony: after a seek the sampler lands on the chord without firing', s0.chord === ch('F') && (s0.chordPulse ?? 0) < 1, `${s0.chord} ${s0.chordPulse}`);
  }

  // --- Cost ---
  {
    const p: Prog = { name: '', chords: PROGS[3].chords, beats: [2], key: { tonic: 0, mode: 'major' } };
    const long: string[] = [];
    for (let i = 0; i < 20; i++) long.push(...p.chords);
    const { inp } = chromaSong({ ...p, chords: long }, 2, 0.2);
    const t0 = performance.now();
    analyzeHarmony(inp);
    const ms = performance.now() - t0;
    check(`harmony: ${inp.duration.toFixed(0)} s song analysed in ${ms.toFixed(1)} ms (< 150 ms)`, ms < 150, '');
  }
}
