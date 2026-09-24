// Groove analysis on synthetic click tracks: straight vs swung, quantized vs humanized, a laid-back
// backbeat and an off-beat (syncopated) pattern, run through the full offline pipeline, plus the
// realtime tracker on the same signals.

import { analyzePcm } from '../src/analysis/analyzePcm';
import { RealtimeAnalyzer } from '../src/analysis/RealtimeAnalyzer';
import { analyzeGroove, fracOf, slotOf, swingOf } from '../src/analysis/groove';
import type { GrooveStats } from '../src/types';

type Check = (name: string, ok: boolean, detail: string) => void;

const SR = 22050;

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

interface Pattern {
  bpm?: number;
  bars?: number;
  /** Off-8th position in the beat (0.5 straight, 2/3 triplet). */
  frac?: number;
  /** Gaussian timing jitter, seconds (std). */
  jitter?: number;
  /** Backbeat (beats 2 and 4) delay, seconds. */
  lateBackbeat?: number;
  /** A syncopated pattern: stabs on the off-beats, silent beats after them. */
  offOnly?: boolean;
}

/** Kick on 1 and 3, snare on 2 and 4, hat on the off-8ths. */
function clickTrack(p: Pattern): Float32Array {
  const bpm = p.bpm ?? 110;
  const bars = p.bars ?? 24;
  const beat = 60 / bpm;
  const N = Math.ceil((0.5 + bars * 4 * beat + 1) * SR);
  const x = new Float32Array(N);
  const r = rng(99);
  const g = () => {
    let u = 0;
    while (u === 0) u = r();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * r());
  };
  const hit = (t: number, amp: number, kind: 'kick' | 'snare' | 'hat') => {
    const s0 = Math.round(t * SR);
    const len = kind === 'kick' ? 0.25 : kind === 'snare' ? 0.15 : 0.05;
    let ph = 0;
    for (let i = 0; i < len * SR && s0 + i < N; i++) {
      if (s0 + i < 0) continue;
      const tt = i / SR;
      let v: number;
      if (kind === 'kick') {
        ph += (2 * Math.PI * (50 + 100 * Math.exp(-tt / 0.03))) / SR;
        v = Math.sin(ph) * Math.exp(-tt / 0.1);
      } else if (kind === 'snare') v = (r() * 2 - 1) * Math.exp(-tt / 0.04) + 0.4 * Math.sin(2 * Math.PI * 190 * tt) * Math.exp(-tt / 0.05);
      else v = (r() * 2 - 1) * Math.exp(-tt / 0.012);
      x[s0 + i] += amp * v * Math.min(1, i / 8);
    }
  };
  const j = () => (p.jitter ? g() * p.jitter : 0);
  for (let b = 0; b < bars * 4; b++) {
    const t = 0.5 + b * beat;
    const pos = b % 4;
    const back = pos === 1 || pos === 3;
    if (p.offOnly) {
      // A kick on the one, a quiet hat on every beat, and stabs on the off-beats of 1, 2 and 3
      // (each tied over the silent beat after it).
      if (pos === 0) hit(t, 0.8, 'kick');
      hit(t, 0.12, 'hat');
      if (pos < 3) hit(t + beat * 0.5, 0.6, 'snare');
      continue;
    }
    if (!back) hit(t + j(), 0.8, 'kick');
    else hit(t + (p.lateBackbeat ?? 0) + j(), 0.6, 'snare');
    hit(t + beat * (p.frac ?? 0.5) + j(), 0.3, 'hat');
  }
  return x;
}

function measure(p: Pattern): { off: GrooveStats; live: GrooveStats; ms: number } {
  const x = clickTrack(p);
  const t0 = performance.now();
  const r = analyzePcm(x, x, SR);
  const ms = performance.now() - t0;
  const g = r.groove!;
  // Realtime: the live analyzer on the same audio, in 512-sample blocks (its own beat clock).
  const a = new RealtimeAnalyzer(SR);
  for (let i = 0; i < x.length; i += 512) a.process(x.subarray(i, i + 512), x.subarray(i, i + 512));
  const tr = a.groove;
  return { off: g.song, live: { ...tr.out }, ms };
}

const f2 = (x: number) => x.toFixed(2);
const fmt = (s: GrooveStats) => `swing ${f2(s.swing)} push ${f2(s.push)} human ${f2(s.humanity)} synco ${f2(s.synco)}`;

export function grooveTests(check: Check): void {
  check('groove.maps', Math.abs(swingOf(fracOf(0.6)) - 0.6) < 1e-9 && swingOf(0.5) === 0 && Math.abs(swingOf(2 / 3) - 1) < 1e-9, 'swing <-> off-beat fraction');
  check('groove.slots', slotOf(0) === 0 && slotOf(0.5) === 2 && slotOf(0.667) === 2 && slotOf(0.25) === 1 && slotOf(0.78) === 3, 'swung off-beats stay on slot 2');
  const empty = analyzeGroove(new Float32Array(100), 86, [], []);
  check('groove.empty', empty.song.swing === 0 && empty.swing.length === 100, 'no beats -> neutral');

  const straight = measure({});
  const swung = measure({ frac: 2 / 3 });
  const medium = measure({ frac: 0.6 });
  const human = measure({ jitter: 0.015 });
  const late = measure({ lateBackbeat: 0.03 });
  const synco = measure({ offOnly: true });
  console.log(`  groove straight: ${fmt(straight.off)} | live ${fmt(straight.live)}`);
  console.log(`  groove triplet:  ${fmt(swung.off)} | live ${fmt(swung.live)}`);
  console.log(`  groove 0.6:      ${fmt(medium.off)} | live ${fmt(medium.live)}`);
  console.log(`  groove human:    ${fmt(human.off)} | live ${fmt(human.live)}`);
  console.log(`  groove late bb:  ${fmt(late.off)} | live ${fmt(late.live)}`);
  console.log(`  groove off-only: ${fmt(synco.off)} | live ${fmt(synco.live)}`);

  check('groove.straight', straight.off.swing < 0.15 && straight.off.humanity < 0.3 && Math.abs(straight.off.push) < 0.25, fmt(straight.off));
  check('groove.triplet-swing', swung.off.swing > 0.8, fmt(swung.off));
  check('groove.medium-swing', medium.off.swing > 0.4 && medium.off.swing < 0.8, fmt(medium.off));
  check('groove.humanity', human.off.humanity > 0.45 && human.off.humanity > straight.off.humanity + 0.3, `${f2(human.off.humanity)} vs straight ${f2(straight.off.humanity)}`);
  check('groove.push', late.off.push > 0.4, `30 ms late backbeat -> push ${f2(late.off.push)}`);
  check('groove.synco', synco.off.synco > straight.off.synco + 0.2, `off-beats ${f2(synco.off.synco)} vs straight ${f2(straight.off.synco)}`);
  // The live estimate reads a wobbling clock: only its swing is asserted (see groove.ts).
  check('groove.live-swing', medium.live.swing > straight.live.swing + 0.3 && swung.live.swing > straight.live.swing + 0.2, `live 0.6 ${f2(medium.live.swing)}, triplet ${f2(swung.live.swing)} vs straight ${f2(straight.live.swing)}`);
  const all = [straight, swung, medium, human, late, synco].map((m) => m.live);
  check('groove.live-range', all.every((g) => g.swing >= 0 && g.swing <= 1 && g.humanity >= 0 && g.humanity <= 1 && g.synco >= 0 && g.synco <= 1 && Math.abs(g.push) <= 1), 'live stats in range');
}
