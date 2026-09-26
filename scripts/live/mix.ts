// Synthetic DJ mixes from the test songs, with a known truth: beat grid, track takeovers, keys,
// builds and drops. Decks are beat-locked to a master clock through each song's own offline beat
// grid (like a DJ app's beatgrid sync) and resampled vinyl-style (pitch follows tempo; rates stay
// within 3 % so the key stays the source key). Transitions: beatmatched crossfades with an
// EQ-style bass swap, a tempo ramp, a hard cut to a new tempo, and a looped build with a filter
// sweep, riser and snare roll into a drop. No silence anywhere. Rendered to PCM only (never played).

import { createHash } from 'node:crypto';
import type { AnalysisResult } from '../../src/types';
import { decode, offlineCached, slugOf, testSongs, type Pcm, SR } from './common';
import { events, matchEvents, type Recording } from './parity';

interface Src {
  slug: string;
  pcm: Pcm;
  res: AnalysisResult;
  bpm: number;
}

interface DeckPlay {
  src: Src;
  /** Master beat where source beat `anchorBeat` sits. */
  mbAnchor: number;
  anchorBeat: number;
  /** Master beats where the deck starts / stops sounding. */
  mbStart: number;
  mbEnd: number;
  /** Band gains over master beats. */
  lo: (mb: number) => number;
  hi: (mb: number) => number;
  /** Loop the `len` beats before `from` once the master passes `from` (a DJ loop for a build). */
  loop?: { from: number; len: number };
}

export interface Transition {
  kind: 'xfade' | 'xfade+bass-swap' | 'cut' | 'build-drop' | 'ramp';
  /** Mix seconds. */
  start: number;
  end: number;
  /** When the new track takes over (bass swap, cut, drop); NaN for a ramp. */
  takeover: number;
  from: string;
  to: string;
}

export interface Mix {
  name: string;
  hash: string;
  pcm: Pcm;
  /** Truth beat and downbeat times (the master clock). */
  beats: number[];
  downbeats: number[];
  /** Master tempo per truth beat. */
  bpmAt: (t: number) => number;
  transitions: Transition[];
  /** Truth drops: the build's drop, and source drops while that deck leads. */
  drops: number[];
  builds: { start: number; end: number }[];
  /** Leading track and its key (tonic + 12 for minor) over time. */
  lead: (t: number) => { slug: string; key: number };
  describe: string;
}

// ------------------------------------------------------------------ builder

class MixBuilder {
  readonly decks: DeckPlay[] = [];
  /** Master tempo changes: at master beat mb, bpm ramps linearly to `bpm` over `beats`. */
  readonly tempo: { mb: number; bpm: number; beats: number }[] = [];
  readonly marks: { kind: Transition['kind']; mb0: number; mb1: number; take: number; from: string; to: string }[] = [];
  readonly buildSpans: { mb0: number; mb1: number }[] = [];
  readonly dropBeats: number[] = [];
  mb = 0;
  bpm: number;
  cur: DeckPlay | null = null;
  constructor(bpm: number) {
    this.bpm = bpm;
    this.tempo.push({ mb: 0, bpm, beats: 0 });
  }
  private deck(src: Src, srcAt: number, mbAt: number): DeckPlay {
    const dbs = src.res.downbeats;
    let db = dbs.find((d) => d >= srcAt) ?? dbs[dbs.length - 1];
    const beats = src.res.beats;
    let bi = 0;
    while (bi < beats.length - 1 && beats[bi] < db - 1e-3) bi++;
    return { src, mbAnchor: mbAt, anchorBeat: bi, mbStart: mbAt, mbEnd: Infinity, lo: () => 1, hi: () => 1 };
  }
  start(src: Src, srcAt: number): this {
    const d = this.deck(src, srcAt, 0);
    this.decks.push(d);
    this.cur = d;
    return this;
  }
  play(bars: number): this {
    this.mb += bars * 4;
    return this;
  }
  xfade(src: Src, srcAt: number, bars: number, bassSwap: boolean): this {
    const a = this.cur!;
    const m0 = this.mb, len = bars * 4, m1 = m0 + len, mid = m0 + len / 2;
    const b = this.deck(src, srcAt, m0);
    const ramp = (x: number, p: number, q: number) => Math.max(0, Math.min(1, (x - p) / (q - p)));
    const aLo0 = a.lo, aHi0 = a.hi;
    if (bassSwap) {
      // B's mids/highs fade in over the first half with its bass cut; at the midpoint the basses swap
      // over one beat; A's mids/highs fade out over the second half.
      b.hi = (x) => ramp(x, m0, mid);
      b.lo = (x) => ramp(x, mid - 0.5, mid + 0.5);
      a.lo = (x) => (x < m0 ? aLo0(x) : 1 - ramp(x, mid - 0.5, mid + 0.5));
      a.hi = (x) => (x < m0 ? aHi0(x) : 1 - ramp(x, mid, m1));
    } else {
      // Equal-power crossfade over the whole span.
      b.hi = b.lo = (x) => Math.sin((Math.PI / 2) * ramp(x, m0, m1));
      a.hi = a.lo = (x) => (x < m0 ? aHi0(x) : Math.cos((Math.PI / 2) * ramp(x, m0, m1)));
    }
    a.mbEnd = m1;
    b.mbStart = m0;
    this.decks.push(b);
    this.marks.push({ kind: bassSwap ? 'xfade+bass-swap' : 'xfade', mb0: m0, mb1: m1, take: mid, from: a.src.slug, to: src.slug });
    this.cur = b;
    this.mb = m1;
    return this;
  }
  ramp(bpm: number, bars: number): this {
    this.tempo.push({ mb: this.mb, bpm, beats: bars * 4 });
    this.marks.push({ kind: 'ramp', mb0: this.mb, mb1: this.mb + bars * 4, take: NaN, from: this.cur!.src.slug, to: this.cur!.src.slug });
    this.mb += bars * 4;
    this.bpm = bpm;
    return this;
  }
  /** Hard cut on the current downbeat; native = the new track at its own tempo (the master jumps). */
  cut(src: Src, srcAt: number, native: boolean): this {
    const a = this.cur!;
    a.mbEnd = this.mb;
    if (native) {
      this.tempo.push({ mb: this.mb, bpm: src.bpm, beats: 0 });
      this.bpm = src.bpm;
    }
    const b = this.deck(src, srcAt, this.mb);
    this.decks.push(b);
    this.marks.push({ kind: 'cut', mb0: this.mb, mb1: this.mb, take: this.mb, from: a.src.slug, to: src.slug });
    this.cur = b;
    return this;
  }
  /** Loop the last `loopBars` of the current deck for `bars`, sweeping its bass out; the render adds a riser and snare roll. */
  build(bars: number, loopBars: number): this {
    const a = this.cur!;
    const m0 = this.mb, m1 = m0 + bars * 4;
    a.loop = { from: m0, len: loopBars * 4 };
    const lo0 = a.lo, hi0 = a.hi;
    a.lo = (x) => (x < m0 ? lo0(x) : Math.max(0, 1 - (x - m0) / ((m1 - m0) * 0.6)));
    a.hi = (x) => (x < m0 ? hi0(x) : 1 - 0.3 * ((x - m0) / (m1 - m0)));
    this.buildSpans.push({ mb0: m0, mb1: m1 });
    this.mb = m1;
    return this;
  }
  drop(src: Src, srcAt: number): this {
    const from = this.cur!.src.slug;
    const m0 = this.buildSpans[this.buildSpans.length - 1]?.mb0 ?? this.mb;
    this.cut(src, srcAt, false);
    const last = this.marks.pop()!;
    this.marks.push({ ...last, kind: 'build-drop', mb0: m0, from });
    this.dropBeats.push(this.mb);
    return this;
  }
}

// ------------------------------------------------------------------ render

function bpmAtBeat(tempo: MixBuilder['tempo'], mb: number): number {
  let bpm = tempo[0].bpm;
  let prev = bpm;
  for (const c of tempo) {
    if (mb < c.mb) break;
    if (c.beats <= 0) bpm = c.bpm;
    else bpm = mb >= c.mb + c.beats ? c.bpm : prev + (c.bpm - prev) * ((mb - c.mb) / c.beats);
    prev = bpm;
  }
  return bpm;
}

/** Source time of fractional source beat index x (the song's own beat grid, extrapolated at its tempo). */
function srcTimeOfBeat(res: AnalysisResult, x: number): number {
  const b = res.beats;
  const per = 60 / (res.bpm || 120);
  if (b.length < 2) return x * per;
  if (x <= 0) return b[0] + x * per;
  if (x >= b.length - 1) return b[b.length - 1] + (x - (b.length - 1)) * per;
  const i = Math.floor(x);
  return b[i] + (b[i + 1] - b[i]) * (x - i);
}

function keyAt(res: AnalysisResult, t: number): number {
  const k = res.keys.find((s) => t >= s.start && t < s.end) ?? res.keys[res.keys.length - 1];
  return k ? k.tonic + (k.mode === 'minor' ? 12 : 0) : -1;
}

function render(b: MixBuilder, name: string, spec: string): Mix {
  const totalBeats = b.mb;
  // Master clock: beat position per sample.
  const beatTimes: number[] = [];
  let mb = 0, t = 0;
  const dt = 1 / SR;
  const mbOfSample: number[] = [];
  // First pass: sample count and master beat per 64-sample block.
  const BLK = 64;
  while (mb < totalBeats) {
    mbOfSample.push(mb);
    const bpm = bpmAtBeat(b.tempo, mb);
    const nextMb = mb + (bpm / 60) * dt * BLK;
    for (let k = Math.ceil(mb); k < nextMb && k <= totalBeats; k++) if (k >= mb) beatTimes.push(t + ((k - mb) / (nextMb - mb)) * dt * BLK);
    mb = nextMb;
    t += dt * BLK;
  }
  const N = mbOfSample.length * BLK;
  const L = new Float32Array(N);
  const R = new Float32Array(N);
  for (const d of b.decks) {
    const src = d.src;
    const sl = src.pcm.left, sr = src.pcm.right;
    // Two cascaded one-pole low-passes at 180 Hz split the deck into bass and the rest (EQ-style).
    const a = 1 - Math.exp((-2 * Math.PI * 180) / SR);
    let l1 = 0, l2 = 0, r1 = 0, r2 = 0;
    for (let blk = 0; blk < mbOfSample.length; blk++) {
      const m0 = mbOfSample[blk];
      const m1 = blk + 1 < mbOfSample.length ? mbOfSample[blk + 1] : m0 + (m0 - (mbOfSample[blk - 1] ?? m0));
      if (m1 < d.mbStart || m0 >= d.mbEnd) continue;
      for (let j = 0; j < BLK; j++) {
        const m = m0 + ((m1 - m0) * j) / BLK;
        if (m < d.mbStart || m >= d.mbEnd) continue;
        let mm = m;
        if (d.loop && m >= d.loop.from) mm = d.loop.from - d.loop.len + ((m - d.loop.from) % d.loop.len);
        const st = srcTimeOfBeat(src.res, d.anchorBeat + (mm - d.mbAnchor)) * SR;
        const i0 = Math.floor(st);
        if (i0 < 0 || i0 + 1 >= sl.length) continue;
        const f = st - i0;
        const xl = sl[i0] + (sl[i0 + 1] - sl[i0]) * f;
        const xr = sr[i0] + (sr[i0 + 1] - sr[i0]) * f;
        l1 += a * (xl - l1); l2 += a * (l1 - l2);
        r1 += a * (xr - r1); r2 += a * (r1 - r2);
        const gl = d.lo(m), gh = d.hi(m);
        const i = blk * BLK + j;
        L[i] += gl * l2 + gh * (xl - l2);
        R[i] += gl * r2 + gh * (xr - r2);
      }
    }
  }
  // Build effects: a noise riser (brighter and louder) and an accelerating snare roll.
  let seed = 12345;
  const rnd = () => ((seed = (seed * 1664525 + 1013904223) >>> 0) / 4294967296) * 2 - 1;
  for (const s of b.buildSpans) {
    const t0 = timeOfBeat(beatTimes, s.mb0), t1 = timeOfBeat(beatTimes, s.mb1);
    let prev = 0;
    for (let i = Math.floor(t0 * SR); i < Math.min(N, t1 * SR); i++) {
      const p = (i / SR - t0) / (t1 - t0);
      const n = rnd();
      const bright = n - (1 - p) * prev; // more high-passed as it rises
      prev = n;
      const v = 0.18 * p * p * bright;
      L[i] += v;
      R[i] += v;
    }
    const beats = s.mb1 - s.mb0;
    for (let k = 0; k < beats * 4; k++) {
      const x = s.mb0 + k / 4; // 16th grid
      const frac = (x - s.mb0) / beats;
      const every = frac < 0.5 ? 4 : frac < 0.75 ? 2 : 1; // quarters, then 8ths, then 16ths
      if (k % every) continue;
      const ts = timeOfBeat(beatTimes, x);
      const g = 0.15 + 0.35 * frac;
      for (let j = 0; j < 0.08 * SR; j++) {
        const i = Math.floor(ts * SR) + j;
        if (i >= N) break;
        const env = Math.exp(-j / (0.018 * SR));
        const v = g * env * (0.7 * rnd() + 0.3 * Math.sin((2 * Math.PI * 190 * j) / SR));
        L[i] += v;
        R[i] += v;
      }
    }
  }
  // Soft limiter so overlaps do not clip.
  let peak = 0;
  for (let i = 0; i < N; i++) peak = Math.max(peak, Math.abs(L[i]), Math.abs(R[i]));
  const g = peak > 0.98 ? 0.98 / peak : 1;
  for (let i = 0; i < N; i++) {
    L[i] = Math.tanh(L[i] * g * 1.1) / Math.tanh(1.1);
    R[i] = Math.tanh(R[i] * g * 1.1) / Math.tanh(1.1);
  }

  const tb = (x: number) => timeOfBeat(beatTimes, x);
  const downbeats = beatTimes.filter((_, i) => i % 4 === 0);
  const transitions: Transition[] = b.marks.map((m) => ({ kind: m.kind, start: tb(m.mb0), end: tb(m.mb1), takeover: Number.isFinite(m.take) ? tb(m.take) : NaN, from: m.from, to: m.to }));
  // Leading deck: the louder of the sounding decks (bass counted like the rest).
  const leadDeck = (mbx: number) => {
    let best: DeckPlay | null = null, bw = -1;
    for (const d of b.decks) {
      if (mbx < d.mbStart || mbx >= d.mbEnd) continue;
      const w = d.lo(mbx) + d.hi(mbx);
      if (w > bw) (bw = w), (best = d);
    }
    return best;
  };
  const beatOfTime = (tt: number) => {
    let lo = 0, hi = beatTimes.length - 1;
    if (tt <= beatTimes[0]) return 0;
    if (tt >= beatTimes[hi]) return hi;
    while (hi - lo > 1) {
      const m = (lo + hi) >> 1;
      if (beatTimes[m] <= tt) lo = m;
      else hi = m;
    }
    return lo + (tt - beatTimes[lo]) / (beatTimes[hi] - beatTimes[lo]);
  };
  const lead = (tt: number) => {
    const mbx = beatOfTime(tt);
    const d = leadDeck(mbx);
    if (!d) return { slug: '', key: -1 };
    let mm = mbx;
    if (d.loop && mbx >= d.loop.from) mm = d.loop.from - d.loop.len + ((mbx - d.loop.from) % d.loop.len);
    const st = srcTimeOfBeat(d.src.res, d.anchorBeat + (mm - d.mbAnchor));
    const rate = bpmAtBeat(b.tempo, mbx) / d.src.bpm;
    const shift = Math.round(12 * Math.log2(rate));
    const k = keyAt(d.src.res, st);
    return { slug: d.src.slug, key: k < 0 ? -1 : ((k % 12) + shift + 12) % 12 + (k >= 12 ? 12 : 0) };
  };
  // Source drops while that deck leads (TimelineSampler's rule: a 'drop', or a chorus after a build).
  const drops = b.dropBeats.map(tb);
  for (const d of b.decks) {
    const secs = d.src.res.sections;
    for (let i = 1; i < secs.length; i++) {
      if (!(secs[i].label === 'drop' || (secs[i].label === 'chorus' && secs[i - 1].label === 'build'))) continue;
      // Find the master beat where the deck plays this source time (no loops here).
      const srcBeatIdx = beatIndexOf(d.src.res, secs[i].start);
      const mbx = d.mbAnchor + (srcBeatIdx - d.anchorBeat);
      if (mbx > d.mbStart + 4 && mbx < d.mbEnd && leadDeck(mbx) === d && !(d.loop && mbx >= d.loop.from)) drops.push(tb(mbx));
    }
  }
  drops.sort((p, q) => p - q);
  const bpmAt = (tt: number) => bpmAtBeat(b.tempo, beatOfTime(tt));
  const hash = createHash('sha1').update(spec + ':v1').digest('hex').slice(0, 8);
  return {
    name, hash, pcm: { sr: SR, left: L, right: R }, beats: beatTimes, downbeats, bpmAt, transitions, drops,
    builds: b.buildSpans.map((s) => ({ start: tb(s.mb0), end: tb(s.mb1) })), lead, describe: spec,
  };
}

function beatIndexOf(res: AnalysisResult, t: number): number {
  const b = res.beats;
  let i = 0;
  while (i < b.length - 1 && b[i + 1] <= t) i++;
  return i + (b.length > i + 1 ? (t - b[i]) / (b[i + 1] - b[i]) : 0);
}

function timeOfBeat(beatTimes: number[], x: number): number {
  const i = Math.floor(x);
  if (i < 0) return beatTimes[0];
  if (i >= beatTimes.length - 1) return beatTimes[beatTimes.length - 1] + (x - (beatTimes.length - 1)) * (beatTimes[beatTimes.length - 1] - beatTimes[beatTimes.length - 2]);
  return beatTimes[i] + (beatTimes[i + 1] - beatTimes[i]) * (x - i);
}

// ------------------------------------------------------------------ the mixes

function src(part: string): Src {
  const path = testSongs().find((p) => slugOf(p).includes(part));
  if (!path) throw new Error('no test song ' + part);
  const slug = slugOf(path);
  const pcm = decode(path);
  const res = offlineCached(slug, pcm);
  return { slug, pcm, res, bpm: res.bpm };
}

export function buildMixes(): Mix[] {
  const out: Mix[] = [];
  {
    const sax = src('saxobeat'), wfl = src('waiting-for-love'), tatu = src('t-a-t-u'), ghosts = src('ghosts');
    const b = new MixBuilder(128)
      .start(sax, 16).play(32)
      .xfade(wfl, 30, 32, true).play(16)
      .ramp(131, 32)
      .xfade(tatu, 65, 16, false).play(16)
      .build(16, 2)
      .drop(ghosts, 67).play(32);
    out.push(render(b, 'mix-club', 'saxobeat@16 32bars > xfade+bassswap 32 wfl@30 > 16 > ramp 128-131 over 32 > xfade 16 tatu@65 > 16 > loop-build 16 (2-bar loop) > drop ghosts@67 > 32'));
  }
  {
    const st = src('stromae'), ff = src('fireflies'), ed = src('thinking-out-loud'), tdfw = src('turn-down'), an = src('animals');
    const b = new MixBuilder(118)
      .start(st, 23).play(32)
      .xfade(ff, 23, 16, true)
      .ramp(121, 32).play(16)
      .xfade(ed, 32, 32, true).play(16)
      .cut(tdfw, 78, true).play(32)
      .xfade(an, 114, 16, true).play(32);
    out.push(render(b, 'mix-pop', 'stromae@23 32bars > xfade+bassswap 16 fireflies@23 > ramp 118-121 over 32 > 16 > xfade+bassswap 32 thinking-out-loud@32 > 16 > hard cut to turn-down@78 at its own 125 bpm > 32 > xfade+bassswap 16 animals@114 > 32'));
  }
  return out;
}

// ------------------------------------------------------------------ truth report

const f2 = (x: number) => (Number.isFinite(x) ? x.toFixed(2) : 'n/a');

function col(rec: Recording, key: string, side: 'off' | 'live'): Float32Array {
  const i = rec.channels.findIndex((c) => c.key === key && c.group !== 'signal');
  if (i < 0) throw new Error('no channel ' + key);
  return side === 'off' ? rec.off[i] : rec.live[i];
}

/** The mix-specific measurements against the construction truth, as markdown. */
export function mixTruthReport(mix: Mix, rec: Recording, _offMix: AnalysisResult): string {
  const fps = rec.fps;
  const n = rec.n;
  const lines: string[] = [];
  lines.push(`Mix: ${mix.describe} (${(n / fps).toFixed(0)} s)`);
  lines.push('');
  // Truth beat phase per frame.
  const truthPhase = new Float32Array(n).fill(NaN);
  {
    let j = 0;
    for (let k = 0; k < n; k++) {
      const t = (k + 1) / fps;
      while (j < mix.beats.length - 2 && mix.beats[j + 1] <= t) j++;
      if (t >= mix.beats[0] && t < mix.beats[mix.beats.length - 1]) truthPhase[k] = (t - mix.beats[j]) / (mix.beats[j + 1] - mix.beats[j]);
    }
  }
  const windows: { label: string; a: number; b: number }[] = [];
  for (const tr of mix.transitions) windows.push({ label: `${tr.kind} ${tr.from.slice(0, 12)}>${tr.to.slice(0, 12)}`, a: tr.start - 2, b: tr.end + (tr.kind === 'cut' ? 20 : 8) });
  const inAny = (t: number) => windows.some((w) => t >= w.a && t < w.b);
  const phaseStats = (side: 'off' | 'live', a: number, b: number) => {
    const ph = col(rec, 'beatPhase', side);
    let ok = 0, c = 0;
    for (let k = Math.max(0, Math.floor(a * fps)); k < Math.min(n, b * fps); k++) {
      if (!Number.isFinite(truthPhase[k])) continue;
      c++;
      if (!Number.isFinite(ph[k])) continue;
      let e = ph[k] - truthPhase[k];
      e -= Math.round(e);
      if (Math.abs(e) < 0.1) ok++;
    }
    return c ? ok / c : NaN;
  };
  const beatF1 = (side: 'off' | 'live', a: number, b: number) => {
    const ob = col(rec, 'onBeat', side);
    const est = events(ob, 0.5, fps, 0.05).filter((t) => t >= a && t < b);
    const ref = mix.beats.filter((t) => t >= a && t < b);
    const { tp } = matchEvents(ref, est, 0.07);
    const P = est.length ? tp / est.length : 0, R = ref.length ? tp / ref.length : 0;
    return P + R > 0 ? (2 * P * R) / (P + R) : 0;
  };
  // Steady = frames outside every transition window.
  const steady = (side: 'off' | 'live') => {
    const ph = col(rec, 'beatPhase', side);
    let ok = 0, c = 0;
    for (let k = 0; k < n; k++) {
      const t = (k + 1) / fps;
      if (t < 15 || inAny(t) || !Number.isFinite(truthPhase[k])) continue;
      c++;
      if (!Number.isFinite(ph[k])) continue;
      let e = ph[k] - truthPhase[k];
      e -= Math.round(e);
      if (Math.abs(e) < 0.1) ok++;
    }
    return c ? ok / c : NaN;
  };
  lines.push('**Beat and phase vs the true grid** (in-phase = |phase error| < 0.1 beat; beat F1 at 70 ms). "offline" is the offline analysis of the whole mix file, which knows the future.');
  lines.push('');
  lines.push('| window | live in-phase | offline in-phase | live beat F1 | offline beat F1 |');
  lines.push('|---|---|---|---|---|');
  lines.push(`| steady (outside transitions, after 15 s) | ${f2(steady('live'))} | ${f2(steady('off'))} | | |`);
  for (const w of windows) lines.push(`| ${w.label} (${w.a.toFixed(0)}-${w.b.toFixed(0)} s) | ${f2(phaseStats('live', w.a, w.b))} | ${f2(phaseStats('off', w.a, w.b))} | ${f2(beatF1('live', w.a, w.b))} | ${f2(beatF1('off', w.a, w.b))} |`);
  // Tempo tracking through the ramp.
  const bpmLive = col(rec, 'bpm', 'live');
  for (const tr of mix.transitions.filter((x) => x.kind === 'ramp' || x.kind === 'cut')) {
    const errs: number[] = [];
    for (let k = Math.floor(tr.start * fps); k < Math.min(n, (tr.end + 16) * fps); k++) errs.push(Math.abs(bpmLive[k] - mix.bpmAt((k + 1) / fps)));
    errs.sort((p, q) => p - q);
    // Re-lock: first time after the change starts that live BPM stays within 1.5 % of the truth for 4 s
    // (searched from 1 s in, so a tracker that was already right before the change is not counted as locked).
    let relock = NaN;
    for (let k = Math.floor((tr.start + 1) * fps); k < n - 4 * fps; k += fps / 4) {
      let ok = true;
      for (let j = k; j < k + 4 * fps; j += 6) if (Math.abs(bpmLive[j] / mix.bpmAt((j + 1) / fps) - 1) > 0.015) {
        ok = false;
        break;
      }
      if (ok) {
        relock = k / fps - tr.start;
        break;
      }
    }
    lines.push('');
    lines.push(`Tempo during ${tr.kind} at ${tr.start.toFixed(0)} s (true ${mix.bpmAt(tr.start - 1).toFixed(1)} -> ${mix.bpmAt(tr.end + 1).toFixed(1)} BPM): live BPM median error ${f2(errs[errs.length >> 1])}, 90th pct ${f2(errs[Math.floor(errs.length * 0.9)])} over the change and 16 s after; live tempo within 1.5 % for 4 s straight from ${Number.isFinite(relock) ? relock.toFixed(1) + ' s after the change started' : 'never'}.`);
  }

  // Key adaptation after each takeover.
  lines.push('');
  lines.push('**Adaptation after a new track takes over** (seconds from the takeover until the value is right and stays right 4 s; key = tonic and mode of the leading track).');
  lines.push('');
  lines.push('| takeover | key before > after | live key | offline key | live chord = offline-of-mix chord: 0-10 s / 10-30 s after (steady) | live notes: height pinned at 0/1 in the 10 s after (steady) |');
  lines.push('|---|---|---|---|---|---|');
  const chL = col(rec, 'chord', 'live'), chO = col(rec, 'chord', 'off');
  const chordAgree = (a: number, b: number) => {
    let ok = 0, c = 0;
    for (let k = Math.max(0, Math.floor(a * fps)); k < Math.min(n, b * fps); k++) {
      if (!Number.isFinite(chO[k])) continue;
      c++;
      if (chL[k] === chO[k]) ok++;
    }
    return c ? ok / c : NaN;
  };
  let steadyChord = 0, scc = 0;
  for (let k = 15 * fps; k < n; k += fps) {
    const t = k / fps;
    if (inAny(t)) continue;
    const v = chordAgree(t, t + 1);
    if (Number.isFinite(v)) (steadyChord += v), scc++;
  }
  steadyChord = scc ? steadyChord / scc : NaN;
  const keyOf = (side: 'off' | 'live') => col(rec, 'key', side);
  const heightL = col(rec, 'notes.height', 'live');
  const heldL = col(rec, 'notes.held', 'live');
  const pinned = (a: number, b: number) => {
    let p = 0, c = 0;
    for (let k = Math.max(0, Math.floor(a * fps)); k < Math.min(n, b * fps); k++) {
      if (!(heldL[k] > 0.1)) continue;
      c++;
      if (heightL[k] <= 0.02 || heightL[k] >= 0.98) p++;
    }
    return c ? p / c : NaN;
  };
  let steadyPin = 0, sc = 0;
  for (let k = 15 * fps; k < n; k += fps) {
    const t = k / fps;
    if (inAny(t)) continue;
    const v = pinned(t, t + 1);
    if (Number.isFinite(v)) (steadyPin += v), sc++;
  }
  steadyPin = sc ? steadyPin / sc : NaN;
  for (const tr of mix.transitions.filter((x) => Number.isFinite(x.takeover))) {
    const before = mix.lead(tr.takeover - 3).key, after = mix.lead(tr.takeover + 3).key;
    const settle = (side: 'off' | 'live') => {
      const k0 = Math.floor(tr.takeover * fps);
      const ks = keyOf(side);
      for (let k = k0; k < Math.min(n, k0 + 90 * fps); k += fps / 4) {
        let ok = true;
        for (let j = k; j < Math.min(n, k + 4 * fps); j += 6) if (ks[j] !== after) {
          ok = false;
          break;
        }
        if (ok) return (k - k0) / fps;
      }
      return NaN;
    };
    const nm = (k: number) => (k < 0 ? '?' : ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'][k % 12] + (k >= 12 ? 'm' : ''));
    const sl = settle('live'), so = settle('off');
    lines.push(`| ${tr.kind} ${tr.to.slice(0, 16)} @${tr.takeover.toFixed(0)} s | ${nm(before)} > ${nm(after)} | ${Number.isFinite(sl) ? sl.toFixed(1) + ' s' : 'not within 90 s'} | ${Number.isFinite(so) ? so.toFixed(1) + ' s' : 'not within 90 s'} | ${f2(chordAgree(tr.takeover, tr.takeover + 10))} / ${f2(chordAgree(tr.takeover + 10, tr.takeover + 30))} (${f2(steadyChord)}) | ${f2(pinned(tr.takeover, tr.takeover + 10))} (${f2(steadyPin)}) |`);
  }

  // Structure: drops, builds, section changes, false triggers inside blends.
  lines.push('');
  const dropL = events(col(rec, 'dropPulse', 'live'), 0.3, fps, 0.5);
  const dropO = events(col(rec, 'dropPulse', 'off'), 0.3, fps, 0.5);
  const mDrop = matchEvents(mix.drops, dropL, 2.0);
  const mDropO = matchEvents(mix.drops, dropO, 2.0);
  lines.push(`**Drops** (truth ${mix.drops.length} at ${mix.drops.map((x) => x.toFixed(0)).join(', ')} s): live fires ${dropL.length} (${dropL.map((x) => x.toFixed(0)).join(', ') || 'none'}), ${mDrop.tp} within 2 s of a true drop, median latency ${mDrop.offsets.length ? (mDrop.offsets.sort((p, q) => p - q)[mDrop.offsets.length >> 1] * 1000).toFixed(0) + ' ms' : 'n/a'}. Offline-of-mix fires ${dropO.length}, ${mDropO.tp} true.`);
  const build = col(rec, 'buildIntensity', 'live');
  const buildO = col(rec, 'buildIntensity', 'off');
  for (const bs of mix.builds) {
    const m = (arr: Float32Array, a: number, b: number) => {
      let s = 0, c = 0;
      for (let k = Math.max(0, Math.floor(a * fps)); k < Math.min(n, b * fps); k++) (s += arr[k]), c++;
      return c ? s / c : NaN;
    };
    lines.push(`**Build** ${bs.start.toFixed(0)}-${bs.end.toFixed(0)} s: live buildIntensity mean ${f2(m(build, bs.start, bs.end))} in the build, ${f2(m(build, bs.end - 8, bs.end))} in its last 8 s, vs ${f2(m(build, 15, n / fps))} over the mix; offline-of-mix ${f2(m(buildO, bs.start, bs.end))} / ${f2(m(buildO, bs.end - 8, bs.end))}.`);
  }
  const secL = events(col(rec, 'sectionChanged', 'live'), 0.5, fps, 0.5);
  const keyPL = events(col(rec, 'keyChangePulse', 'live'), 0.3, fps, 0.5);
  const blends = mix.transitions.filter((x) => x.kind.startsWith('xfade'));
  const inBlend = (t: number) => blends.some((x) => t > x.start + 1 && t < x.end - 1);
  const nearTake = (t: number) => mix.transitions.some((x) => Number.isFinite(x.takeover) && Math.abs(t - x.takeover) < 4) || mix.drops.some((d) => Math.abs(t - d) < 2);
  const falseDrops = dropL.filter((t) => inBlend(t) && !mix.drops.some((d) => Math.abs(t - d) < 2));
  const blendSecs = secL.filter((t) => inBlend(t) && !nearTake(t));
  lines.push('');
  const falseAll = dropL.filter((t) => !mix.drops.some((d) => Math.abs(t - d) < 2));
  lines.push(`**False drops anywhere**: ${falseAll.length} of ${dropL.length} live drop pulses are not within 2 s of a true drop (${(falseAll.length / (n / fps / 60)).toFixed(2)} per minute).`);
  lines.push(`**False triggers inside blends** (${blends.length} crossfades, ${blends.reduce((s, x) => s + x.end - x.start, 0).toFixed(0)} s): false drops ${falseDrops.length}${falseDrops.length ? ' at ' + falseDrops.map((x) => x.toFixed(0)).join(', ') + ' s' : ''}; section changes not at a takeover ${blendSecs.length}; key-change pulses ${keyPL.filter(inBlend).length}.`);
  const takes = mix.transitions.filter((x) => Number.isFinite(x.takeover));
  const detected = takes.filter((x) => secL.some((t) => Math.abs(t - x.takeover) < 8));
  lines.push(`**Track changes**: the live path only announces a new song after 5 s of silence (rtStructure newSongs), which a mix never has (see the new-song table); a section change within 8 s of the takeover happened for ${detected.length} of ${takes.length} takeovers. Live section changes over the whole mix: ${secL.length} (${(secL.length / (n / fps / 60)).toFixed(1)} per minute).`);
  return lines.join('\n');
}
