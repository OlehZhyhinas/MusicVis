// Synthetic-audio tests for the real-time (live input) analysis.
// Feeds generated songs block by block through RealtimeAnalyzer + RealtimeSampler.
// Run: node --import ./scripts/analysis-test.hooks.mjs scripts/realtime-test.ts

import { RealtimeAnalyzer } from '../src/analysis/RealtimeAnalyzer';
import { RealtimeSampler } from '../src/analysis/RealtimeSampler';
import type { LiveAudioFrame, MusicState } from '../src/types';

let SR = 44100;
let BPM = 128;
let BEAT = 60 / BPM;
let BAR = BEAT * 4;
const T0 = 0.25; // music starts here

function setTempo(bpm: number, sr: number): void {
  SR = sr;
  BPM = bpm;
  BEAT = 60 / bpm;
  BAR = BEAT * 4;
}

let failures = 0;
function check(name: string, ok: boolean, detail: string): void {
  console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}: ${detail}`);
  if (!ok) failures++;
}

function rng(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const MUTE = new Set((process.env.MUTE ?? '').split(',').filter(Boolean));
const midiHz = (m: number) => 440 * Math.pow(2, (m - 69) / 12);

type Kind = 'intro' | 'verse' | 'build' | 'drop' | 'chorus' | 'breakdown' | 'outro';
interface Part {
  kind: Kind;
  bars: number;
  tonic: number; // pitch class of the minor tonic
}

interface Song {
  left: Float32Array;
  right: Float32Array;
  duration: number;
  vocalGate: Float32Array;
  parts: { kind: Kind; start: number; end: number; tonic: number }[];
}

// Same generator as scripts/analysis-test.ts (kick / snare / hats, pad, bass,
// vocal-like melody, snare-roll + riser builds), parameterized by tempo and rate.

function synth(parts: Part[]): Song {
  const totalBars = parts.reduce((s, p) => s + p.bars, 0);
  const duration = T0 + totalBars * BAR + 1.0;
  const N = Math.ceil(duration * SR);
  const L = new Float32Array(N);
  const R = new Float32Array(N);
  const gate = new Float32Array(Math.ceil(duration * 100));
  const rnd = rng(1234);
  const noise = () => rnd() * 2 - 1;

  const add = (buf: Float32Array, i: number, v: number) => {
    if (i >= 0 && i < N) buf[i] += v;
  };

  function kick(t: number, g: number) {
    const s0 = Math.round(t * SR);
    let ph = 0;
    for (let i = 0; i < 0.35 * SR; i++) {
      const tt = i / SR;
      const f = 48 + 110 * Math.exp(-tt / 0.03);
      ph += (2 * Math.PI * f) / SR;
      const v = g * Math.sin(ph) * Math.exp(-tt / 0.16) * Math.min(1, i / 20);
      add(L, s0 + i, v);
      add(R, s0 + i, v);
    }
  }
  function snare(t: number, g: number) {
    const s0 = Math.round(t * SR);
    let lp = 0;
    for (let i = 0; i < 0.25 * SR; i++) {
      const tt = i / SR;
      const nz = noise();
      lp += 0.5 * (nz - lp);
      const v = g * (0.8 * lp * Math.exp(-tt / 0.07) + 0.5 * Math.sin(2 * Math.PI * 185 * tt) * Math.exp(-tt / 0.05));
      add(L, s0 + i, v);
      add(R, s0 + i, v);
    }
  }
  function hat(t: number, g: number) {
    const s0 = Math.round(t * SR);
    let prev = 0;
    for (let i = 0; i < 0.06 * SR; i++) {
      const tt = i / SR;
      const nz = noise();
      const hp = nz - prev;
      prev = nz;
      const v = g * 0.5 * hp * Math.exp(-tt / 0.018);
      add(L, s0 + i, v * 0.9);
      add(R, s0 + i, v * 1.1);
    }
  }
  function tone(t: number, dur: number, midi: number, g: number, pan: number, harmonics: number[], vibrato = 0, attack = 0.01, release = 0.05) {
    const s0 = Math.round(t * SR);
    const n = Math.round((dur + release) * SR);
    const f0 = midiHz(midi);
    let ph = 0;
    const gl = g * Math.sqrt(0.5 * (1 - pan));
    const gr = g * Math.sqrt(0.5 * (1 + pan));
    for (let i = 0; i < n; i++) {
      const tt = i / SR;
      const f = f0 * (vibrato > 0 ? Math.pow(2, (vibrato * Math.sin(2 * Math.PI * 5.5 * tt)) / 12) : 1);
      ph += (2 * Math.PI * f) / SR;
      let env = Math.min(1, tt / attack);
      if (tt > dur) env *= Math.max(0, 1 - (tt - dur) / release);
      let v = 0;
      for (let h = 0; h < harmonics.length; h++) v += harmonics[h] * Math.sin((h + 1) * ph);
      v *= env;
      add(L, s0 + i, v * gl);
      add(R, s0 + i, v * gr);
    }
  }

  // Chords relative to minor tonic: i, iv, V, i
  const chordRoots = [0, 5, 7, 0];
  const chordTones = [
    [0, 3, 7],
    [5, 8, 12],
    [7, 11, 14],
    [0, 3, 7],
  ];
  // Vocal melody (scale degrees relative to tonic) per half-bar, 8 notes per 4 bars.
  const melody = [12, 15, 17, 15, 19, 17, 14, 12];

  let bar0 = 0;
  const outParts: Song['parts'] = [];
  for (const part0 of parts) {
    const part: Part = part0.kind === 'chorus' ? { ...part0, kind: 'drop' } : part0;
    const pStart = T0 + bar0 * BAR;
    outParts.push({ kind: part0.kind, start: pStart, end: pStart + part.bars * BAR, tonic: part.tonic });
    const tm = part.tonic === 9 ? 57 : 48 + part.tonic; // A3 or C3..B3
    for (let b = 0; b < part.bars; b++) {
      const tb = pStart + b * BAR;
      const ci = b % 4;
      const lvl = part.kind === 'intro' || part.kind === 'outro' || part.kind === 'breakdown' ? 0.35 : part.kind === 'verse' ? 0.6 : part.kind === 'build' ? 0.45 + 0.35 * (b / part.bars) : 1.0;
      // Pad: chord tones, hard-panned alternately (wide).
      if (!MUTE.has('pad')) chordTones[ci].forEach((iv, k) => {
        tone(tb, BAR - 0.02, tm + iv, 0.05 * lvl + 0.03, k % 2 === 0 ? -1 : 1, [1, 0.5, 0.33, 0.25, 0.2], 0, 0.08, 0.1);
      });
      // Bass.
      const bassMidi = tm - 24 + chordRoots[ci];
      if (part.kind !== 'build' && !MUTE.has('bass')) {
        const per = part.kind === 'intro' || part.kind === 'outro' || part.kind === 'breakdown' ? 4 : 8;
        const bg = part.kind === 'drop' ? 0.3 : part.kind === 'verse' ? 0.2 : 0.12;
        for (let k = 0; k < per; k++) tone(tb + (k * BAR) / per, BAR / per - 0.04, bassMidi, bg, 0, [1, 0.4, 0.2], 0, 0.005, 0.02);
      }
      // Drums.
      if ((part.kind === 'verse' || part.kind === 'drop') && !MUTE.has('drums')) {
        const dg = part.kind === 'drop' ? 1.0 : 0.6;
        for (let k = 0; k < 4; k++) {
          kick(tb + k * BEAT, 0.8 * dg);
          if (k === 1 || k === 3) snare(tb + k * BEAT, 0.5 * dg);
        }
        for (let k = 0; k < 8; k++) hat(tb + (k * BEAT) / 2, 0.25 * dg);
      }
      if (part.kind === 'build' && !MUTE.has('roll')) {
        const sub = b < part.bars / 2 ? 2 : 4;
        for (let k = 0; k < 4 * sub; k++) snare(tb + (k * BEAT) / sub, 0.15 + 0.35 * ((b + k / (4 * sub)) / part.bars));
      }
      // Vocal melody: center, vibrato, two notes per bar, 85% duty.
      if ((part.kind === 'verse' || part.kind === 'drop') && !MUTE.has('vocal')) {
        for (let k = 0; k < 2; k++) {
          const m = tm + melody[(b * 2 + k) % 8];
          const tn = tb + k * (BAR / 2);
          const dur = (BAR / 2) * 0.85;
          tone(tn, dur, m, part.kind === 'drop' ? 0.16 : 0.13, 0, [1, 0.6, 0.4, 0.25], 0.3, 0.03, 0.05);
          for (let q = Math.floor(tn * 100); q < Math.floor((tn + dur) * 100); q++) gate[q] = 1;
        }
      }
    }
    // Riser: wide band-passed noise sweeping up.
    if (part.kind === 'build' && !MUTE.has('riser')) {
      const s0 = Math.round(pStart * SR);
      const n = Math.round(part.bars * BAR * SR);
      const st = [
        { lp: 0, bp: 0 },
        { lp: 0, bp: 0 },
      ];
      for (let i = 0; i < n; i++) {
        const x = i / n;
        const fc = 400 * Math.pow(20, x);
        const f = 2 * Math.sin((Math.PI * fc) / SR);
        const q = 0.5;
        const g = 0.05 + 0.25 * x * x;
        for (let c = 0; c < 2; c++) {
          const s = st[c];
          const inp = noise();
          const hp = inp - s.lp - q * s.bp;
          s.bp += f * hp;
          s.lp += f * s.bp;
          add(c === 0 ? L : R, s0 + i, g * s.bp);
        }
      }
    }
    bar0 += part.bars;
  }
  // Master gain per part (the drop is the loudest part of the song).
  const gainOf = (kind: Kind, x: number) =>
    kind === 'drop' ? 1.5 : kind === 'chorus' ? 1.15 : kind === 'verse' ? 0.8 : kind === 'build' ? 0.7 + 0.4 * x : 0.6;
  for (const p of outParts) {
    const a = Math.round(p.start * SR);
    const b = Math.min(N, Math.round(p.end * SR));
    for (let i = a; i < b; i++) {
      const g = gainOf(p.kind, (i - a) / (b - a));
      L[i] *= g;
      R[i] *= g;
    }
  }
  return { left: L, right: R, duration, vocalGate: gate, parts: outParts };
}

function soloMelody(seconds: number, gain: number): Float32Array {
  const N = Math.ceil(seconds * SR);
  const x = new Float32Array(N);
  const tune = [72, 74, 76, 77, 79, 77, 76, 74, 72, 76, 79, 84, 79, 76, 74, 72];
  const step = 0.45;
  for (let k = 0; T0 + k * step < seconds - 1; k++) {
    const f0 = midiHz(tune[k % tune.length]);
    const s0 = Math.round((T0 + k * step) * SR);
    const len = Math.round(1.2 * SR);
    for (let i = 0; i < len && s0 + i < N; i++) {
      const t = i / SR;
      let v = 0;
      for (let h = 1; h <= 6; h++) v += (Math.sin(2 * Math.PI * f0 * h * t) / h) * Math.exp((-t * (1 + h)) / 0.6);
      x[s0 + i] += gain * v * Math.min(1, i / 40);
    }
  }
  return x;
}

// ---------------------------------------------------------------- harness

const LIVE: LiveAudioFrame = {
  bass: 1,
  mid: 1,
  treb: 1,
  bassAtt: 1,
  midAtt: 1,
  trebAtt: 1,
  waveform: new Float32Array(1024),
  spectrum: new Float32Array(512),
};

interface Trace {
  t: number;
  bpm: number;
  pos: number; // beat clock position at t (no lookahead)
  locked: boolean;
  cx: number;
  cxFast: number;
  pd: number;
  pb: number;
  kick: number;
  db: number;
  label: string;
  key: number;
  onBeat: boolean;
  onBar: boolean;
  sectionChanged: boolean;
  dropPulse: number;
  build: number;
}

interface RunResult {
  a: RealtimeAnalyzer;
  trace: Trace[];
  nonFinite: string[];
  ms: number;
}

function finiteState(s: MusicState): string | null {
  const nums: [string, number][] = [
    ['bpm', s.bpm],
    ['beatPhase', s.beatPhase],
    ['barPhase', s.barPhase],
    ['beatPulse', s.beatPulse],
    ['barPulse', s.barPulse],
    ['loudness', s.loudness],
    ['complexity', s.complexity],
    ['songComplexity', s.songComplexity],
    ['keyHue', s.keyHue],
    ['keyChangePulse', s.keyChangePulse],
    ['sectionProgress', s.sectionProgress],
    ['dropPulse', s.dropPulse],
    ['buildIntensity', s.buildIntensity],
    ['beatIndex', s.beatIndex],
    ['barIndex', s.barIndex],
  ];
  for (const k of ['drums', 'bass', 'vocals', 'other'] as const) {
    nums.push([`stems.${k}`, s.stems[k]], [`onsets.${k}`, s.stemOnsets[k]], [`presence.${k}`, s.stemPresence[k]]);
  }
  for (let k = 0; k < 12; k++) nums.push([`chroma${k}`, s.chroma[k]]);
  for (const [n, v] of nums) if (!Number.isFinite(v)) return n;
  if (!(s.beatPhase >= 0 && s.beatPhase < 1) || !(s.barPhase >= 0 && s.barPhase < 1)) return 'phase range';
  if (!(s.complexity >= 0 && s.complexity <= 1)) return `complexity range ${s.complexity}`;
  return null;
}

/** Feed stereo PCM in 128-sample blocks (AudioWorklet render quantum); sample at 60 fps. */
function run(left: Float32Array, right: Float32Array, sr: number, block = 128): RunResult {
  const a = new RealtimeAnalyzer(sr);
  const smp = new RealtimeSampler(a);
  const trace: Trace[] = [];
  const nonFinite: string[] = [];
  const frameDt = 1 / 60;
  let nextFrame = 0;
  let ms = 0;
  for (let i = 0; i < left.length; i += block) {
    const j = Math.min(left.length, i + block);
    const t0 = performance.now();
    a.process(left.subarray(i, j), right === left ? left.subarray(i, j) : right.subarray(i, j));
    ms += performance.now() - t0;
    while (a.streamTime >= nextFrame) {
      const t = nextFrame;
      const s = smp.sample(t, frameDt, true, LIVE);
      const bad = finiteState(s);
      if (bad && nonFinite.length < 5) nonFinite.push(`${bad} @ ${t.toFixed(2)}`);
      trace.push({
        t,
        bpm: s.bpm,
        pos: a.beat.positionAt(t),
        locked: a.beat.locked,
        cx: a.complexity,
        cxFast: a.complexityFast,
        pd: a.stemPresence.drums,
        pb: a.stemPresence.bass,
        kick: a.kickShare,
        db: a.dbfs,
        label: s.section.label,
        key: a.key.key,
        onBeat: s.onBeat,
        onBar: s.onBar,
        sectionChanged: s.sectionChanged,
        dropPulse: s.dropPulse,
        build: s.buildIntensity,
      });
      nextFrame += frameDt;
    }
  }
  return { a, trace, nonFinite, ms };
}

const NAMES = ['C', 'C#', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B'];
const keyName = (k: number) => (k < 0 ? 'none' : `${NAMES[k % 12]} ${k >= 12 ? 'minor' : 'major'}`);
const meanOf = (tr: Trace[], a: number, b: number, f: (x: Trace) => number) => {
  const xs = tr.filter((x) => x.t >= a && x.t < b);
  return xs.length ? xs.reduce((s, x) => s + f(x), 0) / xs.length : NaN;
};
const at = (tr: Trace[], t: number) => tr[Math.min(tr.length - 1, Math.max(0, Math.round(t * 60)))];
const wrapHalf = (x: number) => x - Math.round(x);
const DEBUG = process.env.DEBUG === '1';

// ---------------------------------------------------------------- EDM form at several tempi

function edmCase(bpm: number, sr: number): void {
  setTempo(bpm, sr);
  const song = synth([
    { kind: 'intro', bars: 16, tonic: 9 },
    { kind: 'verse', bars: 16, tonic: 9 },
    { kind: 'build', bars: 8, tonic: 9 },
    { kind: 'drop', bars: 16, tonic: 0 },
  ]);
  const r = run(song.left, song.right, sr);
  const { a, trace } = r;
  const P = song.parts;
  const tag = `${bpm} BPM @ ${sr / 1000}k`;
  console.log(`\n${tag}: ${song.duration.toFixed(1)} s of audio analyzed in ${r.ms.toFixed(0)} ms (${((100 * r.ms) / 1000 / song.duration).toFixed(2)}% of real time)`);
  const secs = trace.filter((x) => x.sectionChanged).map((x) => `${x.label}@${x.t.toFixed(1)}`);
  console.log(`  sections: ${secs.join(' ')}`);
  console.log(`  truth:    ${P.map((p) => `${p.kind}@${p.start.toFixed(1)}`).join(' ')}`);
  if (process.env.ZOOM) {
    const [z0, z1] = process.env.ZOOM.split(',').map(Number);
    for (let t = z0; t < z1; t += 0.1) {
      const x = at(trace, t);
      console.log(`   t=${t.toFixed(1)} cx=${x.cx.toFixed(2)} fast=${x.cxFast.toFixed(2)} pd=${x.pd.toFixed(2)} pb=${x.pb.toFixed(2)} kick=${x.kick.toFixed(3)} db=${x.db.toFixed(1)} ${x.label} bpm=${x.bpm.toFixed(1)}`);
    }
  }
  if (DEBUG) {
    for (let t = 1; t < song.duration; t += 1) {
      const x = at(trace, t);
      const perr = wrapHalf(x.pos - (x.t - T0) / BEAT) * BEAT * 1000;
      console.log(`   t=${t.toFixed(0).padStart(3)} err=${perr.toFixed(0).padStart(4)}ms bpm=${x.bpm.toFixed(1)} lock=${x.locked ? 1 : 0} cx=${x.cx.toFixed(2)} fast=${x.cxFast.toFixed(2)} pd=${x.pd.toFixed(2)} pb=${x.pb.toFixed(2)} db=${x.db.toFixed(1)} ${x.label} build=${x.build.toFixed(2)} key=${keyName(x.key)}`);
    }
  }

  check(`${tag} finite`, r.nonFinite.length === 0, r.nonFinite.length ? r.nonFinite.join(', ') : 'all MusicState fields finite and in range');

  // Lock
  const lockT = a.beat.firstLockTime - T0;
  check(`${tag} lock time`, Number.isFinite(lockT) && lockT < 6, `locked ${lockT.toFixed(2)} s after the music started`);

  // Tempo after lock (+2 s settle)
  const from = (Number.isFinite(lockT) ? lockT + T0 : 6) + 2;
  let maxErr = 0;
  let maxErrT = 0;
  let unlocked = 0;
  let n = 0;
  for (const x of trace) {
    if (x.t < from || x.t > song.duration - 1.5) continue;
    if (Math.abs(x.bpm - bpm) > maxErr) {
      maxErr = Math.abs(x.bpm - bpm);
      maxErrT = x.t;
    }
    if (!x.locked) unlocked++;
    n++;
  }
  check(`${tag} tempo`, maxErr <= 2, `max |bpm - ${bpm}| after lock ${maxErr.toFixed(2)} (at ${maxErrT.toFixed(1)} s); final ${a.beat.bpm.toFixed(2)} (clock ${(60 / a.beat.period).toFixed(2)}); unlocked ${((100 * unlocked) / Math.max(1, n)).toFixed(1)}% of frames`);

  // Beat phase: clock position vs truth
  const errs: number[] = [];
  for (const x of trace) {
    if (x.t < from || x.t > song.duration - 1.5) continue;
    const truth = (x.t - T0) / BEAT;
    errs.push(wrapHalf(x.pos - truth) * BEAT * 1000);
  }
  const abs = errs.map(Math.abs).sort((p, q) => p - q);
  const med = abs[abs.length >> 1] ?? NaN;
  const p90 = abs[Math.floor(abs.length * 0.9)] ?? NaN;
  const bias = errs.reduce((s, v) => s + v, 0) / Math.max(1, errs.length);
  check(`${tag} beat phase`, med < 25 && p90 < 50, `|error| median ${med.toFixed(1)} ms, p90 ${p90.toFixed(1)} ms, mean ${bias.toFixed(1)} ms (clock vs true beats, analysis frame time)`);

  // onBeat / onBar events
  let hits = 0;
  let total = 0;
  let barHits = 0;
  let barTotal = 0;
  for (const x of trace) {
    if (x.t < from + 8 || x.t > song.duration - 1.5) continue;
    if (x.onBeat) {
      total++;
      const ph = wrapHalf((x.t - T0) / BEAT) * BEAT;
      if (Math.abs(ph) <= 0.045) hits++;
    }
    if (x.onBar) {
      barTotal++;
      const ph = wrapHalf((x.t - T0) / BAR) * BAR;
      if (Math.abs(ph) <= 0.045) barHits++;
    }
  }
  const expectBeats = (song.duration - 1.5 - from - 8) / BEAT;
  check(`${tag} onBeat events`, hits / Math.max(1, total) > 0.9 && Math.abs(total - expectBeats) <= 3, `${hits}/${total} within 45 ms of a true beat (expected ~${expectBeats.toFixed(0)} beats)`);
  check(`${tag} downbeats`, barHits / Math.max(1, barTotal) > 0.8, `${barHits}/${barTotal} onBar events on true downbeats`);

  // Drop
  const dropStart = P[3].start;
  const drops = a.structure.dropTimes;
  const hit = drops.find((t) => Math.abs(t - dropStart) <= 1.5);
  const falseDrops = drops.filter((t) => t < dropStart - 1.5);
  const pulseT = trace.find((x) => x.sectionChanged && x.label === 'drop')?.t;
  check(
    `${tag} drop`,
    hit !== undefined && falseDrops.length === 0,
    `drops at [${drops.map((t) => t.toFixed(2)).join(', ')}], truth ${dropStart.toFixed(2)}; delay ${hit !== undefined ? (hit - dropStart).toFixed(2) : '-'} s (sectionChanged at ${pulseT?.toFixed(2) ?? '-'}), false before: ${falseDrops.length}`,
  );
  const buildT = trace.find((x) => x.sectionChanged && x.label === 'build')?.t;
  const maxBuild = Math.max(...trace.filter((x) => x.t < dropStart + 0.5).map((x) => x.build));
  check(`${tag} build`, buildT !== undefined && buildT > P[1].start && buildT < dropStart && maxBuild > 0.5, `build section at ${buildT?.toFixed(2) ?? '-'} (truth ${P[2].start.toFixed(2)}), max buildIntensity ${maxBuild.toFixed(2)}`);

  // Key (window filled well before the build ends)
  const kBuild = at(trace, dropStart - 0.5).key;
  const kEnd = at(trace, song.duration - 1).key;
  check(`${tag} key`, kBuild === 9 + 12, `before drop ${keyName(kBuild)} (truth A minor, first estimate ${(a.key.firstKeyTime - T0).toFixed(1)} s in); end ${keyName(kEnd)} (truth C minor from ${dropStart.toFixed(1)} s)`);

  // Complexity
  const cIntro = meanOf(trace, P[0].start + 4, P[0].end, (x) => x.cx);
  const cVerse = meanOf(trace, P[1].start + 3, P[1].end, (x) => x.cx);
  const cDrop = meanOf(trace, P[3].start + 3, P[3].end, (x) => x.cx);
  check(`${tag} complexity`, cIntro < 0.35 && cDrop > 0.6 && cDrop > cVerse && cVerse > cIntro, `intro ${cIntro.toFixed(3)}, verse ${cVerse.toFixed(3)}, build ${meanOf(trace, P[2].start + 2, P[2].end, (x) => x.cx).toFixed(3)}, drop ${cDrop.toFixed(3)}; songComplexity ${a.songComplexity.toFixed(3)}`);
}

const ONLY = process.env.CASE ? Number(process.env.CASE) : 0;
if (!ONLY || ONLY === 128) edmCase(128, 44100);
if (!ONLY || ONLY === 96) edmCase(96, 48000);
if (!ONLY || ONLY === 174) edmCase(174, 44100);
if (ONLY) process.exit(failures === 0 ? 0 : 1);

// ---------------------------------------------------------------- solo melody
{
  setTempo(120, 44100);
  const x = soloMelody(40, 0.35);
  const r = run(x, x, SR);
  let cMax = 0;
  let dMax = 0;
  for (const t of r.trace) {
    if (t.t < 2) continue;
    cMax = Math.max(cMax, t.cx);
    dMax = Math.max(dMax, t.pd);
  }
  console.log('');
  check('solo melody finite', r.nonFinite.length === 0, r.nonFinite.join(', ') || 'ok');
  check('solo melody simple', cMax < 0.3 && dMax < 0.1 && r.a.structure.dropTimes.length === 0, `complexity max ${cMax.toFixed(3)}, drums presence max ${dMax.toFixed(3)}, drops ${r.a.structure.dropTimes.length}, songComplexity ${r.a.songComplexity.toFixed(3)}`);
}

// ---------------------------------------------------------------- silence, gaps, odd input
{
  const sil = new Float32Array(10 * 44100);
  const r = run(sil, sil, 44100);
  const beats = r.trace.filter((x) => x.onBeat).length;
  const cMax = Math.max(...r.trace.map((x) => x.cx));
  check('silence', r.nonFinite.length === 0 && beats === 0 && cMax === 0 && r.a.structure.dropTimes.length === 0, `non-finite: ${r.nonFinite.join(', ') || 'none'}; onBeat ${beats}, complexity max ${cMax}, level ${r.a.levelDb.toFixed(0)} dBFS`);

  // Song, 8 s of silence, song again: beats stop in the gap, a new song is detected.
  setTempo(128, 44100);
  const s1 = synth([
    { kind: 'verse', bars: 8, tonic: 9 },
    { kind: 'drop', bars: 4, tonic: 9 },
  ]);
  const gap = 8 * SR;
  const L = new Float32Array(s1.left.length * 2 + gap);
  const R = new Float32Array(L.length);
  L.set(s1.left, 0);
  R.set(s1.right, 0);
  L.set(s1.left, s1.left.length + gap);
  R.set(s1.right, s1.left.length + gap);
  const g = run(L, R, SR);
  const gapA = s1.duration + 2;
  const gapB = s1.duration + 8 - 0.5;
  const gapBeats = g.trace.filter((x) => x.onBeat && x.t > gapA && x.t < gapB).length;
  check('silence gap', g.nonFinite.length === 0 && gapBeats === 0 && g.a.structure.newSongs === 1, `onBeat in the gap ${gapBeats}, new songs detected ${g.a.structure.newSongs}, non-finite: ${g.nonFinite.join(', ') || 'none'}`);

  // Mono input (same array), odd block sizes, NaN / Inf samples, clipping noise.
  const r1 = rng(3);
  const nz = Float32Array.from({ length: 6 * 48000 }, (_, i) => (i % 9973 === 0 ? NaN : i % 7919 === 0 ? Infinity : (r1() * 2 - 1) * 1.5));
  const m = run(nz, nz, 48000, 333);
  check('noise / NaN input', m.nonFinite.length === 0, m.nonFinite.join(', ') || 'ok');
}

console.log(`\n${failures === 0 ? 'ALL PASSED' : `${failures} FAILED`}`);
process.exitCode = failures === 0 ? 0 : 1;
