// Note tracking on synthetic signals with known notes: staccato notes, plucks ringing into the
// next one, tied legato notes with glides, a sung line (vibrato, scoops, drift, off the grid) and
// the same line as a steady synth, each over a drum loop and bass, some under sustained pad chords.
// Checks onset F-measure and pitch accuracy, the legato / glide / vibrato / voice cues, the live
// tracker, and the pipeline (analyzePcm -> TimelineSampler -> MusicState.notes).

import { analyzePcm } from '../src/analysis/analyzePcm';
import { TimelineSampler } from '../src/analysis/TimelineSampler';
import { stftPower } from '../src/analysis/stft';
import { NoteRecorder, NoteTracker } from '../src/analysis/notes';
import type { LiveAudioFrame, NoteTrack } from '../src/types';

type Check = (name: string, ok: boolean, detail: string) => void;

const SR = 22050;
const N = 2048;
const HOP = 256;
const FR = SR / HOP;

interface Truth { start: number; midi: number }
interface Case { x: Float32Array; truth: Truth[] }

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

const mtof = (m: number) => 440 * Math.pow(2, (m - 69) / 12);

/** Adds a tone following pitch(t) (MIDI) and amp(t) over [t0, t1): harmonics 1/h^tilt. */
function tone(x: Float32Array, pitch: (t: number) => number, amp: (t: number) => number, t0: number, t1: number, harm = 10, tilt = 1): void {
  let ph = 0;
  const i0 = Math.max(0, Math.floor(t0 * SR)), i1 = Math.min(x.length, Math.ceil(t1 * SR));
  for (let i = i0; i < i1; i++) {
    const t = i / SR;
    const f = mtof(pitch(t));
    ph += (2 * Math.PI * f) / SR;
    const a = amp(t);
    if (a <= 0) continue;
    let v = 0;
    for (let h = 1; h <= harm && h * f < SR / 2; h++) v += Math.sin(h * ph) / Math.pow(h, tilt);
    x[i] += a * v * 0.25;
  }
}

/** Drum loop and bass under the melody: a kick per beat, off-beat hats, a low A. */
function band(x: Float32Array): void {
  const r = rng(3);
  const beat = 60 / 124;
  for (let i = 0; i < x.length; i++) {
    const t = i / SR;
    const bt = t % beat;
    x[i] += 0.5 * Math.sin(2 * Math.PI * (50 + 80 * Math.exp(-bt * 30)) * bt) * Math.exp(-bt * 12);
    x[i] += 0.08 * (r() * 2 - 1) * Math.exp(-((t + beat / 2) % beat) * 60);
    x[i] += 0.12 * Math.sin(2 * Math.PI * 55 * t) + 0.01 * (r() * 2 - 1);
  }
}

/** Sustained triads in the melody's register, changing every 2 s (each voice at amp). */
function pads(x: Float32Array, amp = 0.35): void {
  const chords = [[57, 60, 64], [53, 57, 60], [55, 59, 62], [52, 55, 59]];
  for (let t = 0, k = 0; t < x.length / SR; t += 2, k++) {
    for (const m of chords[k % 4]) tone(x, () => m, (u) => amp * Math.min(1, (u - t) / 0.05) * Math.min(1, (t + 2 - u) / 0.05), t, t + 2, 8, 1.3);
  }
}

const DUR = 8;
const SCALE = [60, 62, 64, 67, 69, 72, 71, 67, 64, 62];
const empty = () => new Float32Array(DUR * SR);

/** Short detached notes: 0.1 s on every 0.25 s ("tu tu tu"). */
function staccato(): Case {
  const x = empty();
  const truth: Truth[] = [];
  for (let t = 0.5, k = 0; t + 0.25 < DUR; t += 0.25, k++) {
    const m = SCALE[k % SCALE.length];
    tone(x, () => m, (u) => Math.min(1, (u - t) / 0.005) * Math.min(1, (t + 0.1 - u) / 0.01), t, t + 0.1);
    truth.push({ start: t, midi: m });
  }
  band(x);
  return { x, truth };
}

/** Plucks on one pitch, each decaying into the next (no silence between them). */
function plucks(): Case {
  const x = empty();
  const truth: Truth[] = [];
  for (let t = 0.5; t + 0.25 < DUR; t += 0.25) {
    tone(x, () => 67, (u) => Math.min(1, (u - t) / 0.004) * Math.exp(-(u - t) / 0.06), t, t + 0.249);
    truth.push({ start: t, midi: 67 });
  }
  band(x);
  return { x, truth };
}

/** Tied notes, one continuous tone: steps at 1.5, 2.5 and 4.5 s, 0.25 s glides at 3.2 and 5.5 s. */
function legato(): Case {
  const x = empty();
  const segs: [number, number][] = [[0.5, 62], [1.5, 65], [2.5, 69], [3.2, 67], [4.5, 64], [5.5, 62]];
  const pitch = (t: number) => {
    let i = 0;
    while (i + 1 < segs.length && t >= segs[i + 1][0]) i++;
    const [a, m] = segs[i];
    if ((i === 3 || i === 5) && t < a + 0.25) return segs[i - 1][1] + ((m - segs[i - 1][1]) * (t - a)) / 0.25;
    return m;
  };
  tone(x, pitch, (u) => Math.min(1, (u - 0.5) / 0.02) * Math.min(1, (7.5 - u) / 0.05), 0.5, 7.5);
  band(x);
  // A glide does not start a note: the notes are the four steps, at their median pitch.
  return { x, truth: [0.5, 1.5, 2.5, 4.5].map((start, i) => ({ start, midi: [62, 65, 67, 62][i] })) };
}

/** Long notes sung (vibrato 5.8 Hz +-0.5 st, a scoop into each, slow drift, off the grid) or played steady. */
function line(voice: boolean): Case {
  const x = empty();
  const r = rng(9);
  const truth: Truth[] = [];
  for (const [a, b, m] of [[0.5, 2.0, 64], [2.1, 3.6, 67], [3.7, 5.5, 69], [5.6, 7.5, 64]]) {
    const off = voice ? (r() - 0.5) * 0.5 : 0;
    const pitch = (t: number) => {
      const u = t - a;
      if (!voice) return m;
      return m + off - 0.6 * Math.exp(-u / 0.05) + 0.12 * Math.sin(u * 1.3 + a) + 0.5 * Math.min(1, u / 0.3) * Math.sin(2 * Math.PI * 5.8 * u);
    };
    tone(x, pitch, (u) => Math.min(1, (u - a) / 0.03) * Math.min(1, (b - u) / 0.05), a, b);
    truth.push({ start: a, midi: m + off });
  }
  band(x);
  return { x, truth };
}

function withPads(c: Case): Case {
  pads(c.x);
  return c;
}

interface Run { tr: NoteTrack; liveLegato: number; liveVoice: number; liveOns: number }

function run(c: Case): Run {
  const T = Math.floor((c.x.length - 1) / HOP) + 1;
  const rec = new NoteRecorder(N, SR, FR, T);
  const live = new NoteTracker(N, SR, FR);
  let ll = 0, lv = 0, n = 0, ons = 0, lastOn = 0;
  stftPower(c.x, N, HOP, T, (f, pow) => {
    rec.frame(f, pow);
    live.push(pow);
    if (live.out.on > lastOn + 0.2) ons++;
    lastOn = live.out.on;
    if (f >= FR && f < 7 * FR) {
      ll += live.out.legato;
      lv += live.out.voice;
      n++;
    }
  });
  return { tr: rec.build(), liveLegato: ll / n, liveVoice: lv / n, liveOns: ons };
}

/** Onset F-measure (50 ms) and median pitch error (cents) of the matched notes. */
function score(tr: NoteTrack, truth: Truth[]): { F: number; cents: number } {
  const used = new Set<number>();
  const errs: number[] = [];
  for (const g of truth) {
    let best = -1, bd = 0.05;
    tr.notes.forEach((n, i) => {
      const d = Math.abs(n.start - g.start);
      if (!used.has(i) && d <= bd) {
        bd = d;
        best = i;
      }
    });
    if (best >= 0) {
      used.add(best);
      errs.push(Math.abs(tr.notes[best].pitch - g.midi) * 100);
    }
  }
  const P = used.size / Math.max(1, tr.notes.length), R = used.size / Math.max(1, truth.length);
  errs.sort((a, b) => a - b);
  return { F: (2 * P * R) / Math.max(1e-9, P + R), cents: errs[errs.length >> 1] ?? Infinity };
}

/** Mean of a track over 1..7 s, or its max |value|. */
function mean(a: Float32Array): number {
  let s = 0, n = 0;
  for (let i = Math.round(FR); i < Math.round(7 * FR); i++, n++) s += a[i];
  return s / n;
}
function peak(a: Float32Array): number {
  let m = 0;
  for (let i = Math.round(FR); i < Math.round(7 * FR); i++) m = Math.max(m, Math.abs(a[i]));
  return m;
}

const f2 = (x: number) => x.toFixed(2);

export function notesTests(check: Check): void {
  const cases: Record<string, Case> = {
    staccato: staccato(), plucks: plucks(), legato: legato(), sung: line(true), synth: line(false),
    'staccato+pads': withPads(staccato()), 'legato+pads': withPads(legato()), 'sung+pads': withPads(line(true)),
  };
  const R: Record<string, Run> = {};
  const S: Record<string, { F: number; cents: number }> = {};
  for (const [k, c] of Object.entries(cases)) {
    R[k] = run(c);
    S[k] = score(R[k].tr, c.truth);
    const tr = R[k].tr;
    console.log(`  ${k.padEnd(14)} notes ${String(tr.notes.length).padStart(2)}/${c.truth.length} F ${f2(S[k].F)} pitch err ${S[k].cents.toFixed(1)} c | legato ${f2(mean(tr.legato))} live ${f2(R[k].liveLegato)} | glide ${peak(tr.glide).toFixed(1)} st/s vibrato ${f2(peak(tr.vibrato))} st | voice ${f2(mean(tr.voice))} live ${f2(R[k].liveVoice)}`);
  }
  const clean = ['staccato', 'plucks', 'legato', 'sung', 'synth'];
  check('notes.onsets', clean.every((k) => S[k].F >= 0.95), clean.map((k) => `${k} ${f2(S[k].F)}`).join(', '));
  check('notes.pitch', clean.every((k) => S[k].cents < 12), clean.map((k) => `${k} ${S[k].cents.toFixed(1)} c`).join(', '));
  check('notes.pads', S['staccato+pads'].F >= 0.85 && S['legato+pads'].F >= 0.85 && S['sung+pads'].F >= 0.7, `staccato ${f2(S['staccato+pads'].F)}, legato ${f2(S['legato+pads'].F)}, sung ${f2(S['sung+pads'].F)}`);
  const L = (k: string) => mean(R[k].tr.legato);
  check('notes.legato', L('staccato') < 0.4 && L('plucks') < 0.4 && L('staccato+pads') < 0.4 && L('legato') > 0.8 && L('sung') > 0.8 && L('legato+pads') > 0.8,
    `staccato ${f2(L('staccato'))}, plucks ${f2(L('plucks'))}, staccato+pads ${f2(L('staccato+pads'))} vs legato ${f2(L('legato'))}, sung ${f2(L('sung'))}, legato+pads ${f2(L('legato+pads'))}`);
  check('notes.glide', peak(R.legato.tr.glide) > 4 && peak(R.synth.tr.glide) < 1, `glides ${peak(R.legato.tr.glide).toFixed(1)} st/s, steady ${peak(R.synth.tr.glide).toFixed(1)}`);
  check('notes.vibrato', peak(R.sung.tr.vibrato) > 0.3 && peak(R.synth.tr.vibrato) < 0.1 && peak(R.staccato.tr.vibrato) < 0.1, `sung ${f2(peak(R.sung.tr.vibrato))}, steady ${f2(peak(R.synth.tr.vibrato))}, staccato ${f2(peak(R.staccato.tr.vibrato))}`);
  const V = (k: string) => mean(R[k].tr.voice);
  check('notes.voice', V('sung') > 0.45 && V('sung+pads') > 0.4 && V('synth') < 0.15 && V('staccato') < 0.15, `sung ${f2(V('sung'))}, sung+pads ${f2(V('sung+pads'))} vs synth ${f2(V('synth'))}, staccato ${f2(V('staccato'))}`);
  check('notes.live', R.staccato.liveLegato < 0.45 && R.plucks.liveLegato < 0.45 && R.legato.liveLegato > 0.8 && R.sung.liveLegato > 0.8
    && Math.abs(R.staccato.liveOns - 29) <= 3 && R.sung.liveVoice > R.synth.liveVoice + 0.15,
    `legato staccato ${f2(R.staccato.liveLegato)}, plucks ${f2(R.plucks.liveLegato)} vs legato ${f2(R.legato.liveLegato)}, sung ${f2(R.sung.liveLegato)}; onsets ${R.staccato.liveOns}/29; voice sung ${f2(R.sung.liveVoice)} vs synth ${f2(R.synth.liveVoice)}`);

  // Pipeline: analyzePcm keeps the track, the sampler turns it into MusicState.notes.
  const c = staccato();
  const res = analyzePcm(c.x, c.x, SR);
  const sm = new TimelineSampler(res);
  const live: LiveAudioFrame = { bass: 1, mid: 1, treb: 1, bassAtt: 1, midAtt: 1, trebAtt: 1, waveform: new Float32Array(1024), spectrum: new Float32Array(512) };
  let ons = 0, marks = 0, heights = new Set<string>();
  let held = 0, gap = 0;
  for (let t = 1; t < 7; t += 1 / 60) {
    const s = sm.sample(t, 1 / 60, true, live);
    const n = s.notes;
    if (!n) break;
    if (n.on > 0.8) ons++;
    marks = Math.max(marks, n.recent.length);
    for (const m of n.recent) heights.add(m.height.toFixed(2));
    const phase = (t - 0.5) % 0.25;
    if (phase > 0.03 && phase < 0.08) held += n.held;
    if (phase > 0.16 && phase < 0.22) gap += n.held;
  }
  check('notes.pipeline', !!res.notes && ons >= 20 && marks >= 8 && heights.size >= 5 && held > 3 * gap,
    `track ${res.notes?.notes.length ?? 0} notes, on-pulse frames ${ons}, marks ${marks}, heights ${heights.size}, held ${held.toFixed(1)} vs gaps ${gap.toFixed(1)}`);
}
