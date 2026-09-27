// Held-out evaluation of the ML prototypes against the offline analysis, with the same metrics
// as scripts/live-parity.ts (scripts/live/parity.ts score()), next to the DSP live path.
// Each song is scored with the model of the fold that held it out. Audio is never played.
//
//   node --import ./scripts/analysis-test.hooks.mjs scripts/ml/eval.ts [--model gru128] [--folds 3] [--songs a,b] [--wasm] [--pretrained]
//
// Variants:
//   dsp        RealtimeAnalyzer + RealtimeSampler exactly as LiveInput drives them (scripts/live/common.ts LivePath)
//   ml         the GRU's own outputs: beat / bar phase regressed directly (smoothed by a tiny PLL), stems, onsets
//   ml+pll     the GRU's beat activation (instead of spectral flux) and downbeat activation (as the kick accent)
//              fed to the existing BeatTracker (src/analysis/rtBeat.ts, imported, unchanged)
//   <pretrained> beat / downbeat times from .testdata/ml/pretrained/<model>/<slug>.json (causal models only)
// Writes .testdata/ml/eval-<model>.json and prints the table.

import { existsSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { createRequire } from 'node:module';
import { BeatTracker } from '../../src/analysis/rtBeat';
import { TimelineSampler } from '../../src/analysis/TimelineSampler';
import type { MusicState } from '../../src/types';
import { OfflineLive } from '../avq/audio';
import { decode, LivePath, mono, offlineCached, slugOf, testSongs } from '../live/common';
import { FPS, score, type Channel, type Score } from '../live/parity';
import { drumOnsets, labelsOf, LABELS, OUT_ML } from './dump';
import { frameTime, ML_FEAT, ML_FPS, ML_HOP, StreamFeatures } from './features';

const require = createRequire(import.meta.url);
const argv = process.argv.slice(2);
const arg = (k: string, d: string) => (argv.includes('--' + k) ? argv[argv.indexOf('--' + k) + 1] : d);
const MODEL = arg('model', 'gru128');
const FOLDS = Number(arg('folds', '3'));
const USE_WASM = argv.includes('--wasm');
// --ref beat-this: score the beat channels against Beat This! (strong offline tracker) instead of our offline analysis.
const REF = arg('ref', 'offline');
const LEAD = 0.02; // RealtimeSampler's LOOKAHEAD_S, applied to every beat clock alike
const OUTS = ['beat', 'downbeat', 'beatCos', 'beatSin', 'barCos', 'barSin', 'stems.drums', 'stems.bass', 'stems.vocals', 'stems.other', 'onset.drums', 'onset.bass', 'onset.vocals', 'onset.other', 'kick', 'snare', 'hat'];
const O = Object.fromEntries(OUTS.map((n, i) => [n, i]));
const NO = OUTS.length;

type Ort = typeof import('onnxruntime-node');
const ort: Ort = USE_WASM ? require('onnxruntime-web') : require('onnxruntime-node');

const wrapHalf = (x: number) => x - Math.round(x);
const frac = (x: number) => x - Math.floor(x);

/** Run the streaming model over every 512-sample block; returns T x NO outputs and ms per step. */
async function runModel(path: string, hidden: number, layers: number, left: Float32Array, right: Float32Array): Promise<{ y: Float32Array; msPerStep: number; featMs: number }> {
  const sess = await ort.InferenceSession.create(USE_WASM ? readFileSync(path) : path, USE_WASM ? { executionProviders: ['wasm'] } : { intraOpNumThreads: 1, interOpNumThreads: 1 });
  const T = Math.floor(left.length / ML_HOP);
  const fx = new StreamFeatures();
  const y = new Float32Array(T * NO);
  let h = new ort.Tensor('float32', new Float32Array(layers * hidden), [layers, 1, hidden]);
  let ms = 0, fms = 0;
  for (let k = 0; k < T; k++) {
    const t0 = performance.now();
    const f = fx.push(left.subarray(k * ML_HOP, (k + 1) * ML_HOP), right.subarray(k * ML_HOP, (k + 1) * ML_HOP));
    const t1 = performance.now();
    const out = await sess.run({ x: new ort.Tensor('float32', new Float32Array(f), [1, 1, ML_FEAT]), h });
    ms += performance.now() - t1;
    fms += t1 - t0;
    y.set(out.y.data as Float32Array, k * NO);
    h = out.h_out as typeof h;
  }
  return { y, msPerStep: ms / T, featMs: fms / T };
}

/** A beat clock sampled at the render clock: position (beats) and bar position (bars) at time t. */
interface Clock {
  beatPos(t: number): number;
  barPhase(t: number): number;
}

/** Direct phase regression, smoothed by a small PLL so the clock advances evenly between frames. */
function mlPhaseClock(y: Float32Array, T: number): Clock {
  const pos = new Float64Array(T), rate = new Float64Array(T), bar = new Float64Array(T), conf = new Float32Array(T);
  let p = 0, r = 2, b = 0; // beats per second
  const dt = 1 / ML_FPS;
  for (let k = 0; k < T; k++) {
    const c = y[k * NO + O.beatCos], s = y[k * NO + O.beatSin];
    const ph = frac(Math.atan2(s, c) / (2 * Math.PI));
    const m = Math.min(1, Math.hypot(c, s));
    const pred = p + r * dt;
    const e = wrapHalf(ph - frac(pred));
    p = pred + 0.25 * m * e;
    r = Math.min(3.2, Math.max(1.0, r + 0.02 * m * e * ML_FPS * dt * 4));
    const bc = y[k * NO + O.barCos], bs = y[k * NO + O.barSin];
    const bph = frac(Math.atan2(bs, bc) / (2 * Math.PI));
    const bpred = b + (r / 4) * dt;
    b = bpred + 0.25 * Math.min(1, Math.hypot(bc, bs)) * wrapHalf(bph - frac(bpred));
    pos[k] = p; rate[k] = r; bar[k] = b; conf[k] = m;
  }
  const at = (t: number) => Math.min(T - 1, Math.floor(t * ML_FPS) - 1);
  return {
    beatPos: (t) => {
      const k = at(t);
      if (k < 0) return NaN;
      return pos[k] + rate[k] * Math.min(0.05, t + LEAD - frameTime(k));
    },
    barPhase: (t) => {
      const k = at(t);
      if (k < 0) return NaN;
      return frac(bar[k] + (rate[k] / 4) * Math.min(0.05, t + LEAD - frameTime(k)));
    },
  };
}

/** The GRU's activations through the existing BeatTracker (PLL, tempo ACF, downbeat slot). */
function mlPllClock(y: Float32Array, T: number, left: Float32Array): Clock {
  const beat = new Float32Array(T), down = new Float32Array(T);
  for (let k = 0; k < T; k++) (beat[k] = y[k * NO + O.beat]), (down[k] = y[k * NO + O.downbeat]);
  return actPllClock(beat, down, T, left);
}

/** Any causal beat activation (one value per 512-sample frame) through the existing BeatTracker. */
function actPllClock(beat: Float32Array, down: Float32Array | null, T: number, left: Float32Array): Clock {
  const bt = new BeatTracker(ML_FPS);
  const refPos = new Float64Array(T), refTime = new Float64Array(T), period = new Float64Array(T), slot = new Int8Array(T);
  for (let k = 0; k < T; k++) {
    let e = 0;
    for (let i = k * ML_HOP; i < (k + 1) * ML_HOP; i += 4) e += left[i] * left[i];
    const active = e / (ML_HOP / 4) > 1e-6;
    const t = frameTime(k);
    bt.push(4 * beat[k], t, active, { kick: down ? down[k] : 0, bass: 0, snare: 0 }, null);
    refPos[k] = bt.refPos; refTime[k] = bt.refTime; period[k] = bt.period; slot[k] = bt.downbeatSlot;
  }
  const at = (t: number) => Math.min(T - 1, Math.floor(t * ML_FPS) - 1);
  const pos = (t: number) => {
    const k = at(t);
    if (k < 0) return NaN;
    const tt = Math.min(t + LEAD, frameTime(k) + 0.05 + LEAD);
    return refPos[k] + (tt - refTime[k]) / period[k];
  };
  return { beatPos: pos, barPhase: (t) => frac((pos(t) - slot[Math.max(0, at(t))]) / 4) };
}

/** A clock from reported beat / downbeat times (pretrained causal trackers): phase between the last two reports, extrapolated. */
function timesClock(beats: number[], downbeats: number[]): Clock {
  const beatPos = (t: number) => {
    // index of the last beat reported by t
    let i = -1;
    let lo = 0, hi = beats.length - 1;
    while (lo <= hi) {
      const m = (lo + hi) >> 1;
      if (beats[m] <= t) (i = m), (lo = m + 1);
      else hi = m - 1;
    }
    if (i < 1) return NaN;
    const per = Math.min(1.2, Math.max(0.25, beats[i] - beats[i - 1]));
    return i + Math.min(0.999, (t - beats[i]) / per);
  };
  const dset = downbeats.map((d) => beats.reduce((bi, b, j) => (Math.abs(b - d) < Math.abs(beats[bi] - d) ? j : bi), 0));
  return {
    beatPos,
    barPhase: (t) => {
      const p = beatPos(t);
      if (!Number.isFinite(p)) return NaN;
      let last = -1;
      for (const j of dset) if (j <= Math.floor(p)) last = j;
      if (last < 0) return NaN;
      return frac((p - last) / 4);
    },
  };
}

const ph = (tol: number) => ({ t: 'phase' as const, tol });
const ev = (tol: number, jump: number) => ({ t: 'event' as const, tol, jump });
const cont = { t: 'cont' as const };
const CH: Channel[] = [
  { key: 'beatPhase', group: 'beat', kind: ph(0.1), read: () => 0 },
  { key: 'barPhase', group: 'beat', kind: ph(0.1), read: () => 0 },
  { key: 'onBeat', group: 'beat', kind: ev(0.07, 0.5), read: () => 0 },
  { key: 'onBar', group: 'beat', kind: ev(0.1, 0.5), read: () => 0 },
  ...['drums', 'bass', 'vocals', 'other'].map((s) => ({ key: `stems.${s}`, group: 'stems', kind: cont, read: () => 0 })),
  ...['drums', 'bass', 'vocals', 'other'].map((s) => ({ key: `stemOnsets.${s}`, group: 'stems', kind: cont, read: () => 0 })),
  ...['kick', 'snare', 'hat'].map((s) => ({ key: `drum.${s}`, group: 'drums', kind: cont, read: () => 0 })),
  ...['kick', 'snare', 'hat'].map((s) => ({ key: `drum.${s}.events`, group: 'drums', kind: ev(0.05, 0.3), read: () => 0 })),
];
const CI = Object.fromEntries(CH.map((c, i) => [c.key, i]));

/** Fill beat channels of a variant from a clock (events: floor of the position advancing, never re-firing backwards). */
function clockChannels(cl: Clock, n: number, rows: Float32Array[]) {
  let lastB = -Infinity, lastBar = NaN;
  for (let i = 0; i < n; i++) {
    const t = (i + 1) / FPS;
    const p = cl.beatPos(t);
    const bp = cl.barPhase(t);
    rows[CI.beatPhase][i] = Number.isFinite(p) ? frac(p) : NaN;
    rows[CI.barPhase][i] = bp;
    let on = 0, onBar = 0;
    if (Number.isFinite(p) && Math.floor(p) > lastB) {
      if (lastB > -Infinity) on = 1;
      lastB = Math.floor(p);
      if (Number.isFinite(bp) && Number.isFinite(lastBar) && bp < lastBar - 0.5) onBar = 1;
    }
    if (Number.isFinite(bp) && bp < 0.25 && Number.isFinite(lastBar) && lastBar > 0.75) onBar = on ? 1 : onBar || 1;
    lastBar = bp;
    rows[CI.onBeat][i] = on;
    rows[CI.onBar][i] = onBar;
  }
}

/** Exact (non-causal) beat channels from reference beat / downbeat times. */
function refChannels(beats: number[], downbeats: number[], n: number, rows: Float32Array[]) {
  const phase = (g: number[], t: number) => {
    if (g.length < 2 || t < g[0] || t >= g[g.length - 1]) return NaN;
    let lo = 0, hi = g.length - 1;
    while (hi - lo > 1) { const m = (lo + hi) >> 1; if (g[m] <= t) lo = m; else hi = m; }
    return (t - g[lo]) / (g[hi] - g[lo]);
  };
  let bi = 0, di = 0;
  for (let i = 0; i < n; i++) {
    const t = (i + 1) / FPS;
    rows[CI.beatPhase][i] = phase(beats, t);
    rows[CI.barPhase][i] = phase(downbeats, t);
    let on = 0, onb = 0;
    while (bi < beats.length && beats[bi] <= t) (on = 1), bi++;
    while (di < downbeats.length && downbeats[di] <= t) (onb = 1), di++;
    rows[CI.onBeat][i] = on;
    rows[CI.onBar][i] = onb;
  }
}

function peakEvents(x: Float32Array, thr: number): Float32Array {
  // 0/1 train: local maxima above thr (causal-ish: a peak is known one frame later, 17 ms at 60 fps).
  const o = new Float32Array(x.length);
  for (let i = 1; i < x.length - 1; i++) if (x[i] > thr && x[i] >= x[i - 1] && x[i] > x[i + 1]) o[i + 1] = 1;
  return o;
}

async function main() {
  const songs = testSongs().map(slugOf);
  const want = arg('songs', '');
  const results: { slug: string; fold: number; variants: Record<string, Score[]>; cost: Record<string, number> }[] = [];
  const pretrained = argv.includes('--pretrained') ? ['madmom-online', 'beatnet-online', 'madmom-downbeat-offline', 'beat-this'].filter((m) => existsSync(join(OUT_ML, 'pretrained', m))) : [];
  for (let si = 0; si < songs.length; si++) {
    const slug = songs[si];
    if (want && !want.split(',').some((w) => slug.includes(w))) continue;
    const fold = si % FOLDS;
    const mname = `${MODEL}-f${fold}of${FOLDS}`;
    const meta = JSON.parse(readFileSync(join(OUT_ML, 'models', mname + '.json'), 'utf8'));
    if (!meta.held.includes(slug)) throw new Error(`${slug} not held out by ${mname}`);
    const path = testSongs().find((p) => slugOf(p) === slug)!;
    const pcm = decode(path);
    const res = offlineCached(slug, pcm);
    const dur = pcm.left.length / pcm.sr;
    const n = Math.floor(dur * FPS) - 2;
    const T = Math.floor(pcm.left.length / ML_HOP);

    // Offline reference and DSP live on the render clock (as scripts/live/parity.ts record()).
    const m = mono(pcm);
    const liveA = new OfflineLive(m, pcm.sr), liveB = new OfflineLive(m, pcm.sr);
    const tl = new TimelineSampler(res);
    const lp = new LivePath(pcm);
    const mk = () => CH.map(() => new Float32Array(n));
    const off = mk(), dsp = mk();
    const lab = labelsOf(res, drumOnsets(pcm), T);
    const L = LABELS.length;
    const readState = (s: MusicState, rows: Float32Array[], i: number) => {
      rows[CI.beatPhase][i] = s.beatIndex >= 0 ? s.beatPhase : NaN;
      rows[CI.barPhase][i] = s.barIndex >= 0 ? s.barPhase : NaN;
      rows[CI.onBeat][i] = s.onBeat ? 1 : 0;
      rows[CI.onBar][i] = s.onBar ? 1 : 0;
      for (const st of ['drums', 'bass', 'vocals', 'other'] as const) {
        rows[CI[`stems.${st}`]][i] = s.stems[st];
        rows[CI[`stemOnsets.${st}`]][i] = s.stemOnsets[st];
      }
    };
    const dt = 1 / FPS;
    for (let i = 0; i < n; i++) {
      const t = (i + 1) * dt;
      readState(tl.sample(t, dt, true, liveA.read(t, dt)), off, i);
      readState(lp.sample(t, dt, liveB.read(t, dt)), dsp, i);
      // Offline drum references: the offline kick / snare / hat onset strengths at t.
      const k = Math.min(T - 1, Math.max(0, Math.round(t * ML_FPS) - 1));
      for (const d of ['kick', 'snare', 'hat'] as const) off[CI[`drum.${d}`]][i] = lab[k * L + LABELS.indexOf(d)];
      // DSP live has no per-drum output; its closest signal is the drum stem's onsets.
      for (const d of ['kick', 'snare', 'hat'] as const) dsp[CI[`drum.${d}`]][i] = dsp[CI['stemOnsets.drums']][i];
    }
    for (const d of ['kick', 'snare', 'hat'] as const) {
      off[CI[`drum.${d}.events`]] = peakEvents(off[CI[`drum.${d}`]], 0.3);
      dsp[CI[`drum.${d}.events`]] = peakEvents(dsp[CI[`drum.${d}`]], 0.3);
    }

    // ML
    const { y, msPerStep, featMs } = await runModel(join(OUT_ML, 'models', mname + '.onnx'), meta.hidden, meta.layers, pcm.left, pcm.right);
    const ml = mk(), mlp = mk();
    clockChannels(mlPhaseClock(y, T), n, ml);
    clockChannels(mlPllClock(y, T, pcm.left), n, mlp);
    for (let i = 0; i < n; i++) {
      const t = (i + 1) * dt;
      const k = Math.min(T - 1, Math.floor(t * ML_FPS) - 1);
      for (const rows of [ml, mlp]) {
        for (const st of ['drums', 'bass', 'vocals', 'other'] as const) {
          rows[CI[`stems.${st}`]][i] = k >= 0 ? y[k * NO + O[`stems.${st}`]] : 0;
          rows[CI[`stemOnsets.${st}`]][i] = k >= 0 ? y[k * NO + O[`onset.${st}`]] : 0;
        }
        for (const d of ['kick', 'snare', 'hat'] as const) rows[CI[`drum.${d}`]][i] = k >= 0 ? y[k * NO + O[d]] : 0;
      }
    }
    for (const rows of [ml, mlp]) for (const d of ['kick', 'snare', 'hat'] as const) rows[CI[`drum.${d}.events`]] = peakEvents(rows[CI[`drum.${d}`]], 0.3);

    const variants: Record<string, Float32Array[]> = { dsp, ml, 'ml+pll': mlp };
    for (const pm of pretrained) {
      const f = join(OUT_ML, 'pretrained', pm, slug + '.json');
      if (!existsSync(f)) continue;
      const j = JSON.parse(readFileSync(f, 'utf8')) as { beats: number[]; downbeats: number[] };
      const rows = mk();
      clockChannels(timesClock(j.beats, j.downbeats), n, rows);
      for (let c = 4; c < CH.length; c++) rows[c].fill(NaN);
      variants[pm] = rows;
      // The model's raw causal activation (100 fps) through the existing BeatTracker.
      const af = join(OUT_ML, 'pretrained', pm, slug + '.act.f32');
      if (existsSync(af)) {
        const b = readFileSync(af);
        const act = new Float32Array(b.buffer, b.byteOffset, b.byteLength / 4);
        const beat = new Float32Array(T);
        for (let k = 0; k < T; k++) beat[k] = act[Math.min(act.length - 1, Math.floor(frameTime(k) * 100))] ?? 0;
        const r2 = mk();
        clockChannels(actPllClock(beat, null, T, pcm.left), n, r2);
        for (let c = 4; c < CH.length; c++) r2[c].fill(NaN);
        variants[pm + '+pll'] = r2;
      }
    }
    if (REF !== 'offline') {
      const j = JSON.parse(readFileSync(join(OUT_ML, 'pretrained', REF, slug + '.json'), 'utf8')) as { beats: number[]; downbeats: number[] };
      variants.offline = off.map((r) => r.slice());
      for (let c = 4; c < CH.length; c++) variants.offline[c].fill(NaN);
      delete variants[REF];
      refChannels(j.beats, j.downbeats, n, off);
      // Drum events against ADTOF's transcription (a trained drum transcriber) when present.
      const af = join(OUT_ML, 'pretrained', 'adtof', slug + '.json');
      if (existsSync(af)) {
        const ad = JSON.parse(readFileSync(af, 'utf8')) as Record<string, number[]>;
        for (const d of ['kick', 'snare', 'hat'] as const) {
          const c = CI[`drum.${d}.events`];
          variants.offline[c] = off[c].slice();
          off[c].fill(0);
          for (const t of ad[d] ?? []) { const i = Math.round(t * FPS) - 1; if (i >= 0 && i < n) off[c][i] = 1; }
        }
      }
    }
    const scored: Record<string, Score[]> = {};
    for (const [v, rows] of Object.entries(variants)) scored[v] = CH.map((ch, c) => (rows[c].every((x) => Number.isNaN(x)) ? { key: ch.key, group: ch.group, kind: ch.kind.t, quality: NaN, latency: NaN, detail: 'n/a', flags: [], num: {} } : score(ch, off[c], rows[c], FPS)));
    results.push({ slug, fold, variants: scored, cost: { msPerStep, featMs } });
    const q = (v: string, k: string) => scored[v]?.[CI[k]]?.quality;
    console.error(`${slug} (fold ${fold}): beatPhase dsp ${q('dsp', 'beatPhase')?.toFixed(2)} ml ${q('ml', 'beatPhase')?.toFixed(2)} ml+pll ${q('ml+pll', 'beatPhase')?.toFixed(2)} | onBeat dsp ${q('dsp', 'onBeat')?.toFixed(2)} ml ${q('ml', 'onBeat')?.toFixed(2)} ml+pll ${q('ml+pll', 'onBeat')?.toFixed(2)} | ${msPerStep.toFixed(3)} ms/step`);
  }
  writeFileSync(join(OUT_ML, `eval-${MODEL}${USE_WASM ? '-wasm' : ''}${REF !== 'offline' ? '-ref-' + REF : ''}.json`), JSON.stringify(results, (_, v) => (typeof v === 'number' && !Number.isFinite(v) ? null : v)));
  // Table
  const vnames = [...new Set(results.flatMap((r) => Object.keys(r.variants)))];
  const mean = (a: number[]) => { const b = a.filter(Number.isFinite); return b.length ? b.reduce((p, q) => p + q, 0) / b.length : NaN; };
  const med = (a: number[]) => { const b = a.filter(Number.isFinite).sort((p, q) => p - q); return b.length ? b[b.length >> 1] : NaN; };
  console.log(`\n${results.length} held-out songs, model ${MODEL} (${FOLDS}-fold), beat reference: ${REF}`);
  console.log('| channel | ' + vnames.map((v) => `${v} q | ${v} lat ms`).join(' | ') + ' |');
  console.log('|---|' + vnames.map(() => '---|---').join('|') + '|');
  for (const ch of CH) {
    const cells = vnames.map((v) => {
      const s = results.map((r) => r.variants[v]?.[CI[ch.key]]).filter(Boolean) as Score[];
      const q = mean(s.map((x) => x.quality)), l = med(s.map((x) => x.latency));
      return `${Number.isFinite(q) ? q.toFixed(2) : 'n/a'} | ${Number.isFinite(l) ? Math.round(l * 1000) : ''}`;
    });
    console.log(`| ${ch.key} | ${cells.join(' | ')} |`);
  }
  console.log(`model step ${mean(results.map((r) => r.cost.msPerStep)).toFixed(3)} ms, features ${mean(results.map((r) => r.cost.featMs)).toFixed(3)} ms per 11.6 ms block (${USE_WASM ? 'onnxruntime-web wasm' : 'onnxruntime-node cpu, 1 thread'})`);
}

await main();
