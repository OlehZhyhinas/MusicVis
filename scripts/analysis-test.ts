// Synthetic-audio tests for the offline analysis pipeline.
// Run: node --import ./scripts/analysis-test.hooks.mjs scripts/analysis-test.ts

import { analyzePcm } from '../src/analysis/analyzePcm';
import { TimelineSampler } from '../src/analysis/TimelineSampler';
import type { AnalysisResult, LiveAudioFrame, StemName } from '../src/types';

const SR = 44100;
const BPM = Number(process.env.BPM ?? 128);
const BEAT = 60 / BPM;
const BAR = BEAT * 4;
const T0 = 0.25; // music starts here

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

// Debugging aid: MUTE=riser,roll,pad,bass,drums,vocal silences instruments.
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
  vocalGate: Float32Array; // 1 where the vocal sounds, per 10 ms
  parts: { kind: Kind; start: number; end: number; tonic: number }[];
}

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

function validate(name: string, r: AnalysisResult): void {
  const errs: string[] = [];
  const contiguous = (segs: { start: number; end: number }[], what: string) => {
    if (segs.length === 0) errs.push(`${what} empty`);
    if (segs.length && Math.abs(segs[0].start) > 1e-6) errs.push(`${what} does not start at 0`);
    if (segs.length && Math.abs(segs[segs.length - 1].end - r.duration) > 1e-3 && r.duration > 0) errs.push(`${what} does not end at duration`);
    for (let i = 1; i < segs.length; i++) if (Math.abs(segs[i].start - segs[i - 1].end) > 1e-6) errs.push(`${what} gap at ${i}`);
    for (const s of segs) if (!(s.end > s.start)) errs.push(`${what} empty segment`);
  };
  contiguous(r.sections, 'sections');
  contiguous(r.keys, 'keys');
  for (let i = 1; i < r.beats.length; i++) if (!(r.beats[i] > r.beats[i - 1])) errs.push('beats not ascending');
  if (r.beats.length === 0) errs.push('no beats');
  const bs = new Set(Array.from(r.beats));
  for (const d of r.downbeats) if (!bs.has(d)) errs.push('downbeat not a beat');
  const arrs: [string, Float32Array][] = [['loudness', r.loudness]];
  for (const s of Object.keys(r.stems) as StemName[]) {
    arrs.push([`stem ${s}`, r.stems[s]]);
    arrs.push([`onset ${s}`, r.stemOnsets[s]]);
  }
  for (const [n, a] of arrs) {
    if (a.length !== r.numFrames) errs.push(`${n} length`);
    for (let i = 0; i < a.length; i++)
      if (!(a[i] >= 0 && a[i] <= 1)) {
        errs.push(`${n} out of range`);
        break;
      }
  }
  if (r.chroma.length !== r.numFrames * 12) errs.push('chroma length');
  if (!(r.bpm > 0 && Number.isFinite(r.bpm))) errs.push('bpm');
  check(`${name} validity`, errs.length === 0, errs.length ? errs.slice(0, 5).join('; ') : `ok (${r.beats.length} beats, ${r.sections.length} sections, ${r.keys.length} keys)`);
}

function meanRange(x: Float32Array, fr: number, a: number, b: number): number {
  const i0 = Math.max(0, Math.floor(a * fr));
  const i1 = Math.min(x.length, Math.floor(b * fr));
  let s = 0;
  for (let i = i0; i < i1; i++) s += x[i];
  return i1 > i0 ? s / (i1 - i0) : 0;
}

const NAMES = ['C', 'C#', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B'];
const fmtKey = (k: { tonic: number; mode: string }) => `${NAMES[k.tonic]} ${k.mode}`;

function printSummary(r: AnalysisResult): void {
  console.log(`  bpm=${r.bpm.toFixed(2)} beats=${r.beats.length} downbeats=${r.downbeats.length} frames=${r.numFrames} @ ${r.frameRate.toFixed(2)} fps`);
  console.log('  sections: ' + r.sections.map((s) => `${s.label}[${s.start.toFixed(1)}-${s.end.toFixed(1)} e=${s.energy.toFixed(2)}]`).join(' '));
  console.log('  keys: ' + r.keys.map((k) => `${fmtKey(k)}[${k.start.toFixed(1)}-${k.end.toFixed(1)} c=${k.confidence.toFixed(2)}]`).join(' '));
}

// ---------------------------------------------------------------- main test
{
  const song = synth([
    { kind: 'intro', bars: 16, tonic: 9 },
    { kind: 'verse', bars: 16, tonic: 9 },
    { kind: 'build', bars: 8, tonic: 9 },
    { kind: 'drop', bars: 16, tonic: 0 },
  ]);
  const t0 = performance.now();
  const r = analyzePcm(song.left, song.right, SR);
  const ms = performance.now() - t0;
  console.log(`\nMain synthetic song (${song.duration.toFixed(1)} s): analyzed in ${ms.toFixed(0)} ms`);
  printSummary(r);
  validate('main', r);

  // Tempo
  const bpmErr = r.bpm - BPM;
  let octave = '';
  if (Math.abs(r.bpm / BPM - 2) < 0.03) octave = ' (double-time error)';
  if (Math.abs(r.bpm / BPM - 0.5) < 0.03) octave = ' (half-time error)';
  check('tempo', Math.abs(bpmErr) <= 1, `bpm ${r.bpm.toFixed(2)} (truth ${BPM})${octave}`);

  // Beats
  const totalBars = 56;
  const truth: number[] = [];
  for (let k = 0; k < totalBars * 4; k++) truth.push(T0 + k * BEAT);
  const nearest = (arr: Float32Array, t: number) => {
    let lo = 0,
      hi = arr.length - 1;
    while (lo < hi) {
      const m = (lo + hi) >> 1;
      if (arr[m] < t) lo = m + 1;
      else hi = m;
    }
    let best = arr[lo];
    if (lo > 0 && Math.abs(arr[lo - 1] - t) < Math.abs(best - t)) best = arr[lo - 1];
    return best - t;
  };
  let hits = 0;
  let sumErr = 0;
  const hitsByPart: Record<string, [number, number]> = {};
  for (const t of truth) {
    const e = nearest(r.beats, t);
    const ok = Math.abs(e) <= 0.03;
    if (ok) {
      hits++;
      sumErr += e;
    }
    const part = song.parts.find((p) => t >= p.start - 1e-6 && t < p.end - 1e-6)!;
    hitsByPart[part.kind] ??= [0, 0];
    hitsByPart[part.kind][0] += ok ? 1 : 0;
    hitsByPart[part.kind][1]++;
  }
  const byPart = Object.entries(hitsByPart)
    .map(([k, [h, n]]) => `${k} ${h}/${n}`)
    .join(', ');
  check('beat accuracy', hits / truth.length > 0.9, `${hits}/${truth.length} within 30 ms (${((100 * hits) / truth.length).toFixed(1)}%), mean offset ${((1000 * sumErr) / Math.max(1, hits)).toFixed(1)} ms; ${byPart}`);

  // Downbeats
  let dbHits = 0;
  const truthDb = truth.filter((_, i) => i % 4 === 0);
  for (const t of truthDb) if (Math.abs(nearest(r.downbeats, t)) <= 0.03) dbHits++;
  check('downbeat phase', dbHits / truthDb.length > 0.9, `${dbHits}/${truthDb.length} truth downbeats matched`);

  // Structure
  const dropStart = song.parts[3].start;
  const buildStart = song.parts[2].start;
  const idx = r.sections.findIndex((s) => Math.abs(s.start - dropStart) <= BAR);
  const dropSec = idx >= 0 ? r.sections[idx] : null;
  const prevSec = idx > 0 ? r.sections[idx - 1] : null;
  check(
    'drop section',
    !!dropSec && (dropSec.label === 'drop' || dropSec.label === 'chorus'),
    dropSec ? `section at ${dropSec.start.toFixed(2)} s (truth ${dropStart.toFixed(2)}) labeled ${dropSec.label}` : 'no boundary within 1 bar of the drop',
  );
  check('build before drop', prevSec?.label === 'build', prevSec ? `previous section ${prevSec.label} starting ${prevSec.start.toFixed(2)} s (truth build ${buildStart.toFixed(2)})` : 'none');
  check('intro first', r.sections[0].label === 'intro', `first section ${r.sections[0].label} ends ${r.sections[0].end.toFixed(2)} (truth ${song.parts[1].start.toFixed(2)})`);

  // Keys
  const k0 = r.keys[0];
  const kN = r.keys[r.keys.length - 1];
  const keyBoundary = r.keys.length >= 2 ? r.keys[r.keys.length - 1].start : NaN;
  check(
    'key segments',
    k0.tonic === 9 && k0.mode === 'minor' && kN.tonic === 0 && kN.mode === 'minor' && r.keys.length <= 3 && Math.abs(keyBoundary - dropStart) <= 2 * BAR,
    `${r.keys.map(fmtKey).join(' -> ')}, change at ${keyBoundary.toFixed(2)} s (truth ${dropStart.toFixed(2)})`,
  );

  // Stems
  const fr = r.frameRate;
  const P = song.parts;
  const drIntro = meanRange(r.stems.drums, fr, P[0].start + 1, P[0].end - 1);
  const drDrop = meanRange(r.stems.drums, fr, P[3].start + 1, P[3].end - 1);
  check('drums stem', drIntro < 0.15 && drDrop > 0.6, `intro ${drIntro.toFixed(3)}, verse ${meanRange(r.stems.drums, fr, P[1].start + 1, P[1].end - 1).toFixed(3)}, build ${meanRange(r.stems.drums, fr, P[2].start + 1, P[2].end - 1).toFixed(3)}, drop ${drDrop.toFixed(3)}`);
  const vIntro = meanRange(r.stems.vocals, fr, P[0].start + 1, P[0].end - 1);
  const vVerse = meanRange(r.stems.vocals, fr, P[1].start + 1, P[1].end - 1);
  const vBuild = meanRange(r.stems.vocals, fr, P[2].start + 1, P[2].end - 1);
  const vDrop = meanRange(r.stems.vocals, fr, P[3].start + 1, P[3].end - 1);
  // Correlation with the vocal gate.
  const gate = song.vocalGate;
  let sx = 0,
    sy = 0,
    sxx = 0,
    syy = 0,
    sxy = 0,
    cnt = 0;
  for (let q = 0; q < gate.length; q++) {
    const t = q / 100;
    const v = r.stems.vocals[Math.min(r.numFrames - 1, Math.round(t * fr))];
    const g = gate[q];
    sx += v;
    sy += g;
    sxx += v * v;
    syy += g * g;
    sxy += v * g;
    cnt++;
  }
  const corr = (sxy - (sx * sy) / cnt) / Math.sqrt((sxx - (sx * sx) / cnt) * (syy - (sy * sy) / cnt));
  check('vocals stem', vIntro < 0.2 && vBuild < 0.3 && vVerse > 0.45 && vDrop > 0.45 && corr > 0.6, `intro ${vIntro.toFixed(3)}, verse ${vVerse.toFixed(3)}, build ${vBuild.toFixed(3)}, drop ${vDrop.toFixed(3)}, corr with vocal gate ${corr.toFixed(3)}`);
  const bIntro = meanRange(r.stems.bass, fr, P[0].start + 1, P[0].end - 1);
  const bBuild = meanRange(r.stems.bass, fr, P[2].start + 1, P[2].end - 1);
  const bDrop = meanRange(r.stems.bass, fr, P[3].start + 1, P[3].end - 1);
  console.log(`  info  bass stem: intro ${bIntro.toFixed(3)}, build ${bBuild.toFixed(3)}, drop ${bDrop.toFixed(3)}; other: intro ${meanRange(r.stems.other, fr, P[0].start + 1, P[0].end - 1).toFixed(3)}, drop ${meanRange(r.stems.other, fr, P[3].start + 1, P[3].end - 1).toFixed(3)}`);

  // Sampler
  const live: LiveAudioFrame = {
    bass: 1,
    mid: 1,
    treb: 1,
    bassAtt: 1,
    midAtt: 1,
    trebAtt: 1,
    waveform: new Float32Array(1024),
    spectrum: new Float32Array(512),
  };
  const smp = new TimelineSampler(r);
  let beatsSeen = 0,
    barsSeen = 0,
    secChanges = 0,
    dropPulses = 0,
    maxBuild = 0,
    phaseJumps = 0;
  let lastPhase = -1;
  const dt = 1 / 60;
  for (let t = 0; t < song.duration; t += dt) {
    const s = smp.sample(t, dt, true, live);
    if (s.onBeat) beatsSeen++;
    if (s.onBar) barsSeen++;
    if (s.sectionChanged) secChanges++;
    if (s.dropPulse === 1) dropPulses++;
    maxBuild = Math.max(maxBuild, s.buildIntensity);
    if (lastPhase >= 0 && !(s.barPhase >= 0 && s.barPhase < 1)) phaseJumps++;
    lastPhase = s.barPhase;
  }
  // Seek: reset + sample far away must not produce events.
  smp.reset();
  const s1 = smp.sample(60, dt, true, live);
  const s2 = smp.sample(60 + dt, dt, true, live);
  check(
    'sampler',
    Math.abs(beatsSeen - r.beats.length) <= 1 && Math.abs(barsSeen - r.downbeats.length) <= 1 && secChanges === r.sections.length - 1 && dropPulses >= 1 && maxBuild > 0.8 && phaseJumps === 0 && !s1.onBeat && !s1.sectionChanged && !s2.sectionChanged,
    `onBeat ${beatsSeen}/${r.beats.length}, onBar ${barsSeen}/${r.downbeats.length}, sectionChanged ${secChanges}/${r.sections.length - 1}, dropPulses ${dropPulses}, max buildIntensity ${maxBuild.toFixed(2)}, keyHue ${s2.keyHue.toFixed(3)}`,
  );
}

// ---------------------------------------------------------------- 4-minute timing
{
  const song = synth([
    { kind: 'intro', bars: 16, tonic: 9 },
    { kind: 'verse', bars: 16, tonic: 9 },
    { kind: 'build', bars: 8, tonic: 9 },
    { kind: 'drop', bars: 16, tonic: 0 },
    { kind: 'breakdown', bars: 16, tonic: 0 },
    { kind: 'build', bars: 8, tonic: 0 },
    { kind: 'drop', bars: 32, tonic: 0 },
    { kind: 'outro', bars: 15, tonic: 0 },
  ]);
  const t0 = performance.now();
  let lastStage = '';
  const r = analyzePcm(song.left, song.right, SR, (stage) => {
    lastStage = stage;
  });
  const ms = performance.now() - t0;
  console.log(`\n4-minute synthetic song (${song.duration.toFixed(1)} s): analyzed in ${ms.toFixed(0)} ms (last stage "${lastStage}")`);
  printSummary(r);
  console.log('  truth: ' + song.parts.map((p) => `${p.kind}@${p.start.toFixed(1)}`).join(' '));
  validate('4-minute', r);
  check('4-minute runtime', ms < 8000, `${(ms / 1000).toFixed(2)} s`);
}

// ---------------------------------------------------------------- pop form (informational)
{
  const song = synth([
    { kind: 'intro', bars: 8, tonic: 9 },
    { kind: 'verse', bars: 16, tonic: 9 },
    { kind: 'chorus', bars: 16, tonic: 9 },
    { kind: 'verse', bars: 16, tonic: 9 },
    { kind: 'chorus', bars: 16, tonic: 9 },
    { kind: 'outro', bars: 8, tonic: 9 },
  ]);
  const r = analyzePcm(song.left, song.right, SR);
  console.log(`\nPop-form synthetic song (${song.duration.toFixed(1)} s), labels informational:`);
  printSummary(r);
  console.log('  truth: ' + song.parts.map((p) => `${p.kind}@${p.start.toFixed(1)}`).join(' '));
  validate('pop', r);
}

// ---------------------------------------------------------------- robustness
{
  const cases: [string, Float32Array, Float32Array, number][] = [];
  const sil = new Float32Array(5 * SR);
  cases.push(['silence 5 s', sil, sil, SR]);
  const r1 = rng(7);
  const nz = Float32Array.from({ length: 3 * 48000 }, () => (r1() * 2 - 1) * 0.3);
  cases.push(['mono noise 3 s @48k', nz, nz, 48000]);
  cases.push(['tiny 0.2 s', new Float32Array(0.2 * SR).map((_, i) => Math.sin(i * 0.05)), new Float32Array(0.2 * SR), SR]);
  cases.push(['empty', new Float32Array(0), new Float32Array(0), SR]);
  // 3/4 waltz at 90 BPM with a click on each beat, accent on 1.
  const w = new Float32Array(20 * 22050);
  for (let k = 0; k * (60 / 90) < 20; k++) {
    const s0 = Math.round(k * (60 / 90) * 22050);
    for (let i = 0; i < 2000 && s0 + i < w.length; i++) w[s0 + i] += (k % 3 === 0 ? 0.8 : 0.4) * Math.sin(i * 0.1) * Math.exp(-i / 400);
  }
  cases.push(['waltz 20 s @22.05k', w, w, 22050]);
  for (const [name, l, rr, sr] of cases) {
    try {
      const r = analyzePcm(l, rr, sr);
      validate(name, r);
      const smp = new TimelineSampler(r);
      const live: LiveAudioFrame = { bass: 1, mid: 1, treb: 1, bassAtt: 1, midAtt: 1, trebAtt: 1, waveform: new Float32Array(1024), spectrum: new Float32Array(512) };
      for (let t = -0.5; t < r.duration + 2; t += 0.05) {
        const s = smp.sample(t, 0.05, true, live);
        if (!Number.isFinite(s.beatPhase + s.barPhase + s.keyHue + s.sectionProgress)) throw new Error('non-finite sampler output');
      }
    } catch (e) {
      check(`${name} no crash`, false, String((e as Error).stack ?? e));
    }
  }
}

console.log(`\n${failures === 0 ? 'ALL PASSED' : `${failures} FAILED`}`);
process.exitCode = failures === 0 ? 0 : 1;
