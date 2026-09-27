// Stem energies against a real separator: htdemucs (offline, strong) stem RMS envelopes from
// .testdata/ml/stems/demucs/<slug>.<stem>.f32 (one value per 512-sample hop, causal 2048 window).
// Scores, per stem, the Pearson correlation of dB envelopes (scale-free, 0.25 s smoothing) at the
// best lag in +-300 ms, and that lag, for: the live DSP stems (LivePath, as LiveInput drives them),
// the offline analysis' stems, and HS-TasNet streaming (causal separator, .testdata/ml/stems/hstasnet/).
//
//   node --import ./scripts/analysis-test.hooks.mjs scripts/ml/stems-eval.ts [--seconds N]

import { existsSync, readFileSync } from 'node:fs';
import { join } from 'node:path';
import { OfflineLive } from '../avq/audio';
import { decode, LivePath, mono, offlineCached, slugOf, testSongs } from '../live/common';

const STEMS = ['drums', 'bass', 'vocals', 'other'] as const;
const ML = join(import.meta.dirname, '../../.testdata/ml');
const HOP = 512, SR = 44100, FR = SR / HOP;
const ai = process.argv.indexOf('--seconds');
const SECONDS = ai > 0 ? Number(process.argv[ai + 1]) : Infinity;

function readF32(p: string): Float32Array {
  const b = readFileSync(p);
  return new Float32Array(b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength));
}
/** RMS envelope -> dB, floored 60 dB under the song's 99th percentile, smoothed ~0.25 s. */
function db(x: Float32Array): Float32Array {
  const d = Float32Array.from(x, (v) => 20 * Math.log10(v + 1e-9));
  const s = [...d].sort((a, b) => a - b);
  const top = s[Math.floor(0.99 * (s.length - 1))];
  for (let i = 0; i < d.length; i++) d[i] = Math.max(d[i], top - 60);
  return smooth(d);
}
function smooth(x: Float32Array, w = Math.round(0.25 * FR)): Float32Array {
  const o = new Float32Array(x.length);
  let a = x[0] ?? 0;
  const k = 1 / w;
  for (let i = 0; i < x.length; i++) o[i] = a += (x[i] - a) * k;
  return o;
}
function corrLag(ref: Float32Array, y: Float32Array): { r: number; lag: number } {
  const n = Math.min(ref.length, y.length);
  let best = -2, bl = 0;
  const L = Math.round(0.3 * FR);
  for (let lag = -L; lag <= L; lag++) {
    let sx = 0, sy = 0, sxx = 0, syy = 0, sxy = 0, c = 0;
    for (let i = Math.max(0, -lag); i < n && i + lag < n; i++) {
      const a = ref[i], b = y[i + lag];
      sx += a; sy += b; sxx += a * a; syy += b * b; sxy += a * b; c++;
    }
    const r = (sxy / c - (sx / c) * (sy / c)) / Math.sqrt((sxx / c - (sx / c) ** 2) * (syy / c - (sy / c) ** 2) || 1);
    if (r > best) (best = r), (bl = lag);
  }
  return { r: best, lag: bl / FR };
}

const acc: Record<string, { r: number[]; lag: number[] }> = {};
const add = (k: string, v: { r: number; lag: number }) => {
  const a = (acc[k] ??= { r: [], lag: [] });
  a.r.push(v.r);
  a.lag.push(v.lag);
};
for (const path of testSongs()) {
  const slug = slugOf(path);
  if (!existsSync(join(ML, 'stems/demucs', `${slug}.drums.f32`))) continue;
  const pcm = decode(path);
  const res = offlineCached(slug, pcm);
  const T = Math.min(Math.floor(pcm.left.length / HOP), Math.floor(SECONDS * FR));
  // Live DSP stems sampled once per hop.
  const lp = new LivePath(pcm);
  const live = new OfflineLive(mono(pcm), pcm.sr);
  const dsp: Record<string, Float32Array> = Object.fromEntries(STEMS.map((s) => [s, new Float32Array(T)]));
  const dt = HOP / SR;
  for (let k = 0; k < T; k++) {
    const t = (k + 1) * dt;
    const st = lp.sample(t, dt, live.read(t, dt));
    for (const s of STEMS) dsp[s][k] = st.stems[s];
  }
  const line: string[] = [];
  for (const s of STEMS) {
    const ref = db(readF32(join(ML, 'stems/demucs', `${slug}.${s}.f32`)).subarray(0, T));
    const off = new Float32Array(T);
    for (let k = 0; k < T; k++) off[k] = res.stems[s][Math.min(res.stems[s].length - 1, Math.round(((k + 1) * dt) * res.frameRate))];
    const cD = corrLag(ref, smooth(dsp[s])), cO = corrLag(ref, smooth(off));
    add(`dsp.${s}`, cD);
    add(`offline.${s}`, cO);
    const hp = join(ML, 'stems/hstasnet', `${slug}.${s}.f32`);
    let hs = '';
    if (existsSync(hp)) {
      const cH = corrLag(ref, db(readF32(hp).subarray(0, T)));
      add(`hstasnet.${s}`, cH);
      hs = ` hs ${cH.r.toFixed(2)}`;
    }
    line.push(`${s} dsp ${cD.r.toFixed(2)} off ${cO.r.toFixed(2)}${hs}`);
  }
  console.error(`${slug.slice(0, 24).padEnd(24)} ${line.join(' | ')}`);
}
const mean = (a: number[]) => a.reduce((p, q) => p + q, 0) / a.length;
const med = (a: number[]) => [...a].sort((p, q) => p - q)[a.length >> 1];
console.log('\nstem energy vs htdemucs (dB envelopes): mean r, median lag ms (+ = late)');
console.log('| stem | DSP live | offline analysis | HS-TasNet streaming |');
console.log('|---|---|---|---|');
for (const s of STEMS) {
  const c = (k: string) => (acc[k] ? `${mean(acc[k].r).toFixed(2)} @ ${Math.round(med(acc[k].lag) * 1000)}` : 'n/a');
  console.log(`| ${s} | ${c('dsp.' + s)} | ${c('offline.' + s)} | ${c('hstasnet.' + s)} |`);
}
