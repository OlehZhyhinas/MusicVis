// Live-vs-offline signal parity: runs one PCM buffer through the offline analysis
// (TimelineSampler) and the live path (RealtimeAnalyzer + RealtimeSampler, fed like LiveInput)
// on the same render clock, turns both MusicStates into the engine's preset signals with the
// real Signals class, records every signal and genome input per frame and scores each one.

import { TimelineSampler } from '../../src/analysis/TimelineSampler';
import type { AnalysisResult, MusicState, SectionLabel } from '../../src/types';
import { OfflineLive } from '../avq/audio';
import { FAKE_GL, type EngineBundle } from './bundle';
import { LivePath, mono, type Pcm } from './common';

export const FPS = 60;

export type Kind =
  | { t: 'cont' }
  | { t: 'event'; tol: number; jump: number }
  | { t: 'phase'; tol: number }
  | { t: 'cat' }
  | { t: 'binary' }
  | { t: 'pitch' }
  | { t: 'missing' };

export interface Channel {
  key: string;
  kind: Kind;
  /** Reads the value from a MusicState (after Signals.update) or a signal. */
  read: (s: MusicState, sig: (name: string) => number) => number;
  group: string;
}

const cont: Kind = { t: 'cont' };
const ev = (tol: number, jump = 0.25): Kind => ({ t: 'event', tol, jump });
const STEMS = ['drums', 'bass', 'vocals', 'other'] as const;
const TIMBRE_KEYS = ['bright', 'noise', 'rough', 'attack'] as const;

/** Energy class of a section label (labels are ambiguous; the class is what presets feel). */
export const LABEL_CLASS: Record<SectionLabel, number> = { intro: 0, breakdown: 0, outro: 0, verse: 1, build: 1, chorus: 2, drop: 2 };
const LABEL_ID: Record<SectionLabel, number> = { intro: 0, verse: 1, build: 2, chorus: 3, drop: 4, breakdown: 5, outro: 6 };

/** How each preset signal is scored (event tolerances in seconds). */
export const SIGNAL_KIND: Record<string, Kind> = {
  drums: cont, bass: cont, vocals: cont, other: cont,
  hit: ev(0.07, 0.1), beat: ev(0.07, 0.2), bar: { t: 'phase', tol: 0.1 },
  complexity: cont, drop: ev(1.0, 0.3), loud: cont, melody: cont, build: cont, surge: cont,
  barpulse: ev(0.1, 0.2), section: ev(2.0, 0.3), tension: cont, resolve: ev(0.5, 0.15), chordchange: ev(0.3, 0.3),
  modulation: ev(4, 0.3), swing: cont, push: cont, humanity: cont, synco: cont,
  line: ev(1.0, 0.5), valence: cont, arousal: cont,
  bright: cont, noisy: cont, rough: cont, attack: cont,
  noteon: ev(0.07, 0.15), held: cont, legato: cont, glide: cont, vibrato: cont, voice: cont,
  hook: ev(0.5, 0.3), hookphase: cont, hookon: { t: 'binary' },
};

/** The genome inputs: MusicState fields the genes (not only the reactions) read. */
export function stateChannels(): Channel[] {
  const ch: Channel[] = [];
  const add = (key: string, group: string, kind: Kind, read: Channel['read']) => ch.push({ key, group, kind, read });
  add('beatPhase', 'beat', { t: 'phase', tol: 0.1 }, (s) => (s.beatIndex >= 0 ? s.beatPhase : NaN));
  add('barPhase', 'beat', { t: 'phase', tol: 0.1 }, (s) => (s.barIndex >= 0 ? s.barPhase : NaN));
  add('onBeat', 'beat', ev(0.07, 0.5), (s) => (s.onBeat ? 1 : 0));
  add('onBar', 'beat', ev(0.1, 0.5), (s) => (s.onBar ? 1 : 0));
  add('bpm', 'beat', cont, (s) => s.bpm);
  for (const n of STEMS) add(`stems.${n}`, 'stems', cont, (s) => s.stems[n]);
  for (const n of STEMS) add(`stemOnsets.${n}`, 'stems', cont, (s) => s.stemOnsets[n]);
  for (const n of STEMS) add(`stemPresence.${n}`, 'stems', cont, (s) => s.stemPresence[n]);
  add('loudness', 'level', cont, (s) => s.loudness);
  add('complexity', 'level', cont, (s) => s.complexity);
  add('section.class', 'structure', { t: 'cat' }, (s) => LABEL_CLASS[s.section.label]);
  add('section.label', 'structure', { t: 'cat' }, (s) => LABEL_ID[s.section.label]);
  add('sectionChanged', 'structure', ev(2.0, 0.5), (s) => (s.sectionChanged ? 1 : 0));
  add('buildIntensity', 'structure', cont, (s) => s.buildIntensity);
  add('dropPulse', 'structure', ev(1.0, 0.3), (s) => s.dropPulse);
  add('timeToDrop', 'structure', { t: 'missing' }, (s) => (s.timeToDrop === undefined ? NaN : Math.min(60, s.timeToDrop)));
  add('prevSectionLabel', 'structure', { t: 'missing' }, (s) => (s.prevSectionLabel === undefined ? NaN : LABEL_ID[s.prevSectionLabel]));
  add('repeat (dejavu)', 'structure', { t: 'missing' }, (s) => (s.repeatIndex === undefined ? NaN : s.repeatIndex));
  add('key', 'harmony', { t: 'cat' }, (s) => s.keyTonic + (s.keyMode === 'minor' ? 12 : 0));
  add('keyChangePulse', 'harmony', ev(4, 0.3), (s) => s.keyChangePulse);
  add('chord', 'harmony', { t: 'cat' }, (s) => (s.chord === undefined || s.chord < 0 ? NaN : s.chord));
  add('tension', 'harmony', cont, (s) => s.tension ?? NaN);
  add('chordPulse', 'harmony', ev(0.3, 0.3), (s) => s.chordPulse ?? 0);
  add('resolvePulse', 'harmony', ev(0.5, 0.15), (s) => s.resolvePulse ?? 0);
  add('modulationPulse', 'harmony', ev(4, 0.3), (s) => s.modulationPulse ?? 0);
  add('hookOn', 'hooks', { t: 'binary' }, (s) => s.hookOn ?? 0);
  add('hookPhase', 'hooks', cont, (s) => s.hookPhase ?? 0);
  add('hookPulse', 'hooks', ev(0.5, 0.3), (s) => s.hookPulse ?? 0);
  add('hookNotePulse', 'hooks', ev(0.1, 0.3), (s) => s.hookNotePulse ?? 0);
  add('notes.on', 'notes', ev(0.07, 0.15), (s) => s.notes?.on ?? 0);
  add('notes.held', 'notes', cont, (s) => s.notes?.held ?? 0);
  add('notes.legato', 'notes', cont, (s) => s.notes?.legato ?? NaN);
  add('notes.pitch', 'notes', { t: 'pitch' }, (s) => (s.notes && s.notes.held > 0.1 ? s.notes.pitch : NaN));
  add('notes.height', 'notes', cont, (s) => s.notes?.height ?? NaN);
  add('notes.voice', 'notes', cont, (s) => s.notes?.voice ?? NaN);
  add('notes.vibrato', 'notes', cont, (s) => s.notes?.vibrato ?? NaN);
  add('notes.glide', 'notes', cont, (s) => s.notes?.glide ?? NaN);
  for (const part of ['mix', ...STEMS] as const) for (const k of TIMBRE_KEYS) add(`timbre.${part}.${k}`, 'timbre', cont, (s) => s.timbre?.[part][k] ?? NaN);
  for (const k of ['swing', 'push', 'humanity', 'synco'] as const) add(`groove.${k}`, 'groove', cont, (s) => s.groove?.[k] ?? NaN);
  // Lyrics come from src/lyrics (LRCLIB), not the audio analysis: offline in the app when the song has lyrics, never live.
  add('lyrics', 'lyrics', { t: 'missing' }, (s) => (s.lyricPresence === undefined ? NaN : s.lyricPresence));
  return ch;
}

export function signalChannels(signals: readonly string[]): Channel[] {
  return signals.map((n) => ({ key: n, group: 'signal', kind: SIGNAL_KIND[n] ?? cont, read: (_s, sig) => sig(n) }));
}

export interface Recording {
  n: number;
  fps: number;
  channels: Channel[];
  off: Float32Array[];
  live: Float32Array[];
  liveProcessMs: number;
  seconds: number;
}

/** Record every channel for offline and live over [0, duration) at FPS. */
/**
 * opts.lag: LiveInput's visual lag (default the app's 0.1 s, with the look-ahead note tracker).
 * opts.align (default true): compare the live state drawn at wall time t + lag with the offline
 * state at t, i.e. score the analysis itself; the fixed lag is a presentation delay reported
 * separately (it adds to every latency for tab capture, where the sound cannot be delayed).
 */
export function record(E: EngineBundle, pcm: Pcm, result: AnalysisResult, channels: Channel[], opts: { t0?: number; t1?: number; lag?: number; align?: boolean } = {}): Recording {
  const dur = pcm.left.length / pcm.sr;
  const t0 = opts.t0 ?? 0;
  const t1 = Math.min(dur, opts.t1 ?? dur);
  const n = Math.max(0, Math.floor((t1 - t0) * FPS));
  const dt = 1 / FPS;
  const m = mono(pcm);
  const liveA = new OfflineLive(m, pcm.sr);
  const liveB = new OfflineLive(m, pcm.sr);
  const tl = new TimelineSampler(result);
  const sigOff = new E.Signals(FAKE_GL as never);
  const sigLive = new E.Signals(FAKE_GL as never);
  const off = channels.map(() => new Float32Array(n));
  const live = channels.map(() => new Float32Array(n));
  // Live listening starts at t0 (the analyzer hears nothing before); offline is sampled from the song start.
  if (t0 > 0) {
    // Skip the offline sampler over the lead-in so its pulses are settled.
    for (let t = Math.max(0, t0 - 4); t < t0; t += dt) sigOff.update(tl.sample(t, dt, true, liveA.read(t, dt)), dt, 16 / 9);
  }
  const i0 = Math.round(t0 * pcm.sr);
  const lpAt = new LivePath(i0 > 0 ? { sr: pcm.sr, left: pcm.left.subarray(i0), right: pcm.right.subarray(i0) } : pcm, opts.lag);
  const shift = opts.align === false ? 0 : lpAt.lag;
  for (let k = 0; k < n; k++) {
    const t = t0 + (k + 1) * dt;
    const so = tl.sample(t, dt, true, liveA.read(t, dt));
    sigOff.update(so, dt, 16 / 9);
    const offSig = (x: string) => sigOff.signal(x as never);
    for (let c = 0; c < channels.length; c++) off[c][k] = channels[c].read(so, offSig);
    // The app's waveform analyser trails the sound by the lag too (LiveInput's DelayNode).
    const sl = lpAt.sample(t - t0 + shift, dt, liveB.read(Math.max(0, t - (lpAt.lag - shift)), dt));
    sigLive.update(sl, dt, 16 / 9);
    const liveSig = (x: string) => sigLive.signal(x as never);
    for (let c = 0; c < channels.length; c++) live[c][k] = channels[c].read(sl, liveSig);
  }
  return { n, fps: FPS, channels, off, live, liveProcessMs: lpAt.processMs, seconds: t1 - t0 };
}

// ------------------------------------------------------------------ metrics

export interface Score {
  key: string;
  group: string;
  kind: Kind['t'];
  /** 0..1 how well live matches offline (1 = parity). NaN when the song gives no evidence. */
  quality: number;
  /** Seconds, positive = live late (events: median offset of matches; continuous: best cross-correlation lag). */
  latency: number;
  /** Short human summary of the numbers. */
  detail: string;
  flags: string[];
  /** Raw numbers for aggregation. */
  num: Record<string, number>;
}

const mean = (a: ArrayLike<number>) => {
  let s = 0, c = 0;
  for (let i = 0; i < a.length; i++) if (Number.isFinite(a[i])) (s += a[i]), c++;
  return c ? s / c : NaN;
};

function std(a: ArrayLike<number>, m = mean(a)): number {
  let s = 0, c = 0;
  for (let i = 0; i < a.length; i++) if (Number.isFinite(a[i])) (s += (a[i] - m) ** 2), c++;
  return c > 1 ? Math.sqrt(s / c) : NaN;
}

function pearsonLag(x: Float32Array, y: Float32Array, lag: number, step: number): number {
  // corr(x[i], y[i + lag])
  let sx = 0, sy = 0, sxx = 0, syy = 0, sxy = 0, c = 0;
  for (let i = Math.max(0, -lag); i < x.length && i + lag < y.length; i += step) {
    const a = x[i], b = y[i + lag];
    if (!Number.isFinite(a) || !Number.isFinite(b)) continue;
    sx += a; sy += b; sxx += a * a; syy += b * b; sxy += a * b; c++;
  }
  if (c < 10) return NaN;
  const vx = sxx / c - (sx / c) ** 2;
  const vy = syy / c - (sy / c) ** 2;
  if (vx <= 1e-12 || vy <= 1e-12) return NaN;
  return (sxy / c - (sx / c) * (sy / c)) / Math.sqrt(vx * vy);
}

export function events(x: Float32Array, jump: number, fps: number, refractory: number): number[] {
  const out: number[] = [];
  let last = -Infinity;
  for (let i = 1; i < x.length; i++) {
    const a = x[i - 1], b = x[i];
    if (!Number.isFinite(b)) continue;
    if (b - (Number.isFinite(a) ? a : 0) > jump) {
      const t = i / fps;
      if (t - last >= refractory) out.push(t);
      last = t;
    }
  }
  return out;
}

/** Greedy one-to-one matching of event times within tol. */
export function matchEvents(ref: number[], est: number[], tol: number): { tp: number; offsets: number[] } {
  const used = new Uint8Array(est.length);
  const offsets: number[] = [];
  let j0 = 0;
  for (const r of ref) {
    while (j0 < est.length && est[j0] < r - tol) j0++;
    let best = -1, bd = Infinity;
    for (let j = j0; j < est.length && est[j] <= r + tol; j++) {
      if (used[j]) continue;
      const d = Math.abs(est[j] - r);
      if (d < bd) (bd = d), (best = j);
    }
    if (best >= 0) {
      used[best] = 1;
      offsets.push(est[best] - r);
    }
  }
  return { tp: offsets.length, offsets };
}

const median = (a: number[]) => {
  if (!a.length) return NaN;
  const s = [...a].sort((p, q) => p - q);
  return s.length % 2 ? s[s.length >> 1] : 0.5 * (s[s.length / 2 - 1] + s[s.length / 2]);
};

const f2 = (x: number) => (Number.isFinite(x) ? x.toFixed(2) : 'n/a');

export function score(ch: Channel, off: Float32Array, live: Float32Array, fps: number): Score {
  const k = ch.kind;
  const flags: string[] = [];
  const base = { key: ch.key, group: ch.group, kind: k.t };
  const activeFrac = (a: Float32Array, thr: number) => {
    let c = 0, n = 0;
    for (let i = 0; i < a.length; i++) if (Number.isFinite(a[i])) (n++, a[i] > thr && c++);
    return n ? c / n : NaN;
  };
  if (k.t === 'missing') {
    const offHas = activeFrac(off, -Infinity);
    const liveHas = activeFrac(live, -Infinity);
    return { ...base, quality: Number.isFinite(liveHas) && liveHas > 0.5 ? 1 : 0, latency: NaN, detail: `present offline ${f2(offHas)} of frames, live ${f2(Number.isFinite(liveHas) ? liveHas : 0)}`, flags: ['missing live'], num: { offHas, liveHas: Number.isFinite(liveHas) ? liveHas : 0 } };
  }
  if (k.t === 'event') {
    const refr = Math.min(k.tol, 0.08);
    const eo = events(off, k.jump, fps, refr);
    const el = events(live, k.jump, fps, refr);
    const { tp, offsets } = matchEvents(eo, el, k.tol);
    const P = el.length ? tp / el.length : NaN;
    const R = eo.length ? tp / eo.length : NaN;
    const F = eo.length === 0 && el.length === 0 ? NaN : P + R > 0 ? (2 * P * R) / (P + R) : 0;
    if (el.length === 0 && eo.length > 0) flags.push('never fires');
    if (eo.length === 0 && el.length > 0) flags.push('fires only live');
    const rate = eo.length ? el.length / eo.length : NaN;
    if (rate > 2) flags.push('over-fires');
    const lat = median(offsets);
    return { ...base, quality: eo.length === 0 ? (el.length ? 0 : NaN) : F, latency: lat, detail: `F1 ${f2(F)} (P ${f2(P)} R ${f2(R)}) off ${eo.length} live ${el.length} lat ${Number.isFinite(lat) ? (lat * 1000).toFixed(0) + 'ms' : 'n/a'}`, flags, num: { F, P, R, nOff: eo.length, nLive: el.length, lat, tp } };
  }
  if (k.t === 'phase') {
    let ok = 0, n = 0, errSum = 0, liveMissing = 0;
    for (let i = 0; i < off.length; i++) {
      if (!Number.isFinite(off[i])) continue;
      n++;
      if (!Number.isFinite(live[i])) {
        liveMissing++;
        continue;
      }
      let e = live[i] - off[i];
      e -= Math.round(e);
      errSum += Math.abs(e);
      if (Math.abs(e) < k.tol) ok++;
    }
    const q = n ? ok / n : NaN;
    // Lock time: first frame after which the phase stays within tol for 4 s (90 % of frames).
    const W = 4 * fps;
    let lock = NaN;
    for (let i = 0; i + W < off.length; i += fps / 4) {
      let good = 0, cnt = 0;
      for (let j = i; j < i + W; j++) {
        if (!Number.isFinite(off[j])) continue;
        cnt++;
        if (!Number.isFinite(live[j])) continue;
        let e = live[j] - off[j];
        e -= Math.round(e);
        if (Math.abs(e) < k.tol) good++;
      }
      if (cnt > W / 2 && good / cnt > 0.9) {
        lock = i / fps;
        break;
      }
    }
    if (!Number.isFinite(lock)) flags.push('never locks');
    return { ...base, quality: q, latency: NaN, detail: `in phase ${f2(q)} of frames, mean |err| ${f2(n ? errSum / Math.max(1, n - liveMissing) : NaN)} cycle, lock ${Number.isFinite(lock) ? lock.toFixed(1) + 's' : 'never'}`, flags, num: { inPhase: q, err: n ? errSum / Math.max(1, n - liveMissing) : NaN, lock } };
  }
  if (k.t === 'cat') {
    let agree = 0, n = 0, liveNone = 0;
    for (let i = 0; i < off.length; i++) {
      if (!Number.isFinite(off[i])) continue;
      n++;
      if (!Number.isFinite(live[i])) {
        liveNone++;
        continue;
      }
      if (live[i] === off[i]) agree++;
    }
    const q = n ? agree / n : NaN;
    if (n && liveNone / n > 0.9) flags.push('never set live');
    let changesOff = 0, changesLive = 0;
    for (let i = 1; i < off.length; i++) {
      if (off[i] !== off[i - 1] && Number.isFinite(off[i]) && Number.isFinite(off[i - 1])) changesOff++;
      if (live[i] !== live[i - 1] && Number.isFinite(live[i]) && Number.isFinite(live[i - 1])) changesLive++;
    }
    return { ...base, quality: q, latency: NaN, detail: `agree ${f2(q)} of frames, changes off ${changesOff} live ${changesLive}`, flags, num: { agree: q, changesOff, changesLive } };
  }
  if (k.t === 'binary') {
    let inter = 0, uni = 0, onOff = 0, onLive = 0;
    for (let i = 0; i < off.length; i++) {
      const a = off[i] > 0.5, b = live[i] > 0.5;
      if (a) onOff++;
      if (b) onLive++;
      if (a && b) inter++;
      if (a || b) uni++;
    }
    const q = uni ? inter / uni : NaN;
    if (onLive === 0 && onOff > 0) flags.push('never fires');
    if (onLive > 0.9 * off.length) flags.push('always on');
    return { ...base, quality: onOff === 0 ? (onLive ? 0 : NaN) : q, latency: NaN, detail: `IoU ${f2(q)}, on offline ${f2(onOff / off.length)} live ${f2(onLive / off.length)}`, flags, num: { iou: q, onOff: onOff / off.length, onLive: onLive / off.length } };
  }
  if (k.t === 'pitch') {
    let both = 0, offOn = 0, good = 0, pcGood = 0;
    const errs: number[] = [];
    for (let i = 0; i < off.length; i++) {
      if (!Number.isFinite(off[i])) continue;
      offOn++;
      if (!Number.isFinite(live[i])) continue;
      both++;
      const e = live[i] - off[i];
      errs.push(Math.abs(e));
      if (Math.abs(e) < 0.5) good++;
      const pc = ((e % 12) + 12) % 12;
      if (pc < 0.5 || pc > 11.5) pcGood++;
    }
    const cover = offOn ? both / offOn : NaN;
    const acc = both ? good / both : NaN;
    return { ...base, quality: offOn ? good / offOn : NaN, latency: NaN, detail: `within 0.5 st ${f2(acc)} of shared frames (pitch class ${f2(both ? pcGood / both : NaN)}), coverage ${f2(cover)}, median err ${f2(median(errs))} st`, flags, num: { acc, cover, pcAcc: both ? pcGood / both : NaN, med: median(errs) } };
  }
  // continuous
  const mo = mean(off), ml = mean(live);
  const so = std(off, mo), sl = std(live, ml);
  const maxLag = Math.round(0.5 * fps);
  const step = off.length > 20000 ? 2 : 1;
  let best = -Infinity, bestLag = 0;
  for (let lag = -maxLag; lag <= maxLag; lag += 2) {
    const r = pearsonLag(off, live, lag, step);
    if (Number.isFinite(r) && r > best) (best = r), (bestLag = lag);
  }
  const r0 = pearsonLag(off, live, 0, step);
  let mae = 0, c = 0;
  for (let i = 0; i < off.length; i++) if (Number.isFinite(off[i]) && Number.isFinite(live[i])) (mae += Math.abs(off[i] - live[i])), c++;
  mae = c ? mae / c : NaN;
  if (!Number.isFinite(so) || so < 1e-4) {
    // No offline variation: score the level only.
    const lvl = Number.isFinite(mae) ? Math.max(0, 1 - mae / 0.1) : NaN;
    return { ...base, quality: lvl, latency: NaN, detail: `offline flat at ${f2(mo)}, live mean ${f2(ml)}`, flags, num: { r: NaN, r0: NaN, lag: NaN, mo, ml, so, sl, mae } };
  }
  const rb = Number.isFinite(best) ? best : NaN;
  if (!Number.isFinite(sl) || sl < 0.1 * so) flags.push('flat live');
  const level = Number.isFinite(mae) ? Math.max(0, 1 - mae / (2 * so + 0.02)) : 0;
  const q = Number.isFinite(sl) && sl >= 0.1 * so ? 0.6 * Math.max(0, Number.isFinite(rb) ? rb : 0) + 0.4 * level : 0;
  const lat = bestLag / fps;
  return { ...base, quality: q, latency: lat, detail: `r ${f2(rb)} @${(lat * 1000).toFixed(0)}ms (r0 ${f2(r0)}), mean off ${f2(mo)} live ${f2(ml)}, sd off ${f2(so)} live ${f2(sl)}`, flags, num: { r: rb, r0, lag: lat, mo, ml, so, sl, mae, level } };
}

export function scoreAll(rec: Recording): Score[] {
  return rec.channels.map((ch, i) => score(ch, rec.off[i], rec.live[i], rec.fps));
}
