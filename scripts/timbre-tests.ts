// Timbre analysis on synthetic signals: sine vs saw vs noise vs a detuned pair (brightness,
// noisiness, roughness), plucked vs swelling notes (attack), a low vs high tone per stem region,
// the full pipeline's track and the live analyzer's running estimate.

import { analyzePcm } from '../src/analysis/analyzePcm';
import { RealtimeAnalyzer } from '../src/analysis/RealtimeAnalyzer';
import { stftPower } from '../src/analysis/stft';
import { TimbreFrames, type TimbreKey } from '../src/analysis/timbre';
import type { TimbreStats } from '../src/types';

type Check = (name: string, ok: boolean, detail: string) => void;

const SR = 22050;
const N = 2048;
const HOP = 256;

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

function signal(seconds: number, f: (t: number, i: number) => number): Float32Array {
  const x = new Float32Array(Math.round(seconds * SR));
  for (let i = 0; i < x.length; i++) x[i] = f(i / SR, i);
  return x;
}

const saw = (f0: number, t: number) => {
  let v = 0;
  for (let k = 1; k * f0 < SR / 2; k++) v += Math.sin(2 * Math.PI * k * f0 * t) / k;
  return v * 0.3;
};

/** Mean timbre of one key over the middle of the signal (or the max attack over it). */
function measure(x: Float32Array, key: TimbreKey = 'mix'): TimbreStats {
  const T = Math.floor((x.length - 1) / HOP) + 1;
  const tf = new TimbreFrames(N, SR, SR / HOP);
  const acc = { bright: 0, noise: 0, rough: 0, attack: 0 };
  let n = 0;
  stftPower(x, N, HOP, T, (f, pow) => {
    tf.push(pow);
    const o = tf.out[key];
    if (f > T * 0.3 && f < T * 0.9) {
      acc.bright += o.bright;
      acc.noise += o.noise;
      acc.rough += o.rough;
      acc.attack += o.attack;
      n++;
    }
  });
  return { bright: acc.bright / n, noise: acc.noise / n, rough: acc.rough / n, attack: acc.attack / n };
}

const f2 = (x: number) => x.toFixed(2);
const fmt = (s: TimbreStats) => `bright ${f2(s.bright)} noise ${f2(s.noise)} rough ${f2(s.rough)} attack ${f2(s.attack)}`;

export function timbreTests(check: Check): void {
  const r = rng(5);
  const sine = measure(signal(2, (t) => 0.4 * Math.sin(2 * Math.PI * 330 * t)));
  const sawS = measure(signal(2, (t) => saw(220, t)));
  const noise = measure(signal(2, () => (r() * 2 - 1) * 0.3));
  const beat = measure(signal(2, (t) => 0.3 * Math.sin(2 * Math.PI * 440 * t) + 0.3 * Math.sin(2 * Math.PI * 466 * t)));
  const dull = measure(signal(2, (t) => 0.4 * Math.sin(2 * Math.PI * 150 * t)));
  const bright = measure(signal(2, (t) => 0.4 * Math.sin(2 * Math.PI * 4000 * t)));
  console.log(`  timbre sine 330:  ${fmt(sine)}`);
  console.log(`  timbre saw 220:   ${fmt(sawS)}`);
  console.log(`  timbre noise:     ${fmt(noise)}`);
  console.log(`  timbre 440+466:   ${fmt(beat)}`);
  console.log(`  timbre sine 150 / 4k bright: ${f2(dull.bright)} / ${f2(bright.bright)}`);
  check('timbre.noise', noise.noise > 0.6 && sine.noise < 0.15 && sawS.noise < 0.35 && noise.noise > sawS.noise + 0.3, `noise ${f2(noise.noise)}, saw ${f2(sawS.noise)}, sine ${f2(sine.noise)}`);
  check('timbre.bright', bright.bright > dull.bright + 0.5 && sawS.bright > sine.bright - 0.05 && noise.bright > sawS.bright, `4k ${f2(bright.bright)} vs 150 ${f2(dull.bright)}; noise ${f2(noise.bright)} > saw ${f2(sawS.bright)}`);
  check('timbre.rough', sine.rough < 0.1 && sawS.rough > sine.rough + 0.2 && beat.rough > sine.rough + 0.2, `sine ${f2(sine.rough)}, saw ${f2(sawS.rough)}, beating pair ${f2(beat.rough)}`);

  // Attack: plucks (instant rise, decay) every half second vs swells (300 ms rise).
  const pluck = measure(signal(3, (t) => {
    const u = t % 0.5;
    return 0.4 * Math.sin(2 * Math.PI * 330 * t) * Math.exp(-u / 0.12);
  }));
  const swell = measure(signal(3, (t) => {
    const u = t % 0.5;
    return 0.4 * Math.sin(2 * Math.PI * 330 * t) * Math.min(1, u / 0.3) * (u < 0.45 ? 1 : (0.5 - u) / 0.05);
  }));
  check('timbre.attack', pluck.attack > 0.5 && pluck.attack > swell.attack + 0.3, `pluck ${f2(pluck.attack)} vs swell ${f2(swell.attack)}`);

  // Stem regions: a low sine lives in the bass, a 1 kHz tone in the vocals / other.
  const low = signal(2, (t) => 0.4 * Math.sin(2 * Math.PI * 45 * t));
  const bassLow = measure(low, 'bass');
  const bassHigh = measure(signal(2, (t) => 0.4 * Math.sin(2 * Math.PI * 200 * t)), 'bass');
  const vocLow = measure(low, 'vocals');
  const hiSaw = signal(2, (t) => saw(880, t));
  const vocSaw = measure(hiSaw, 'vocals');
  check('timbre.stems', bassHigh.bright > bassLow.bright + 0.4 && vocSaw.rough > 0.05 && vocLow.bright < 0.1, `bass region bright 45 Hz ${f2(bassLow.bright)} vs 200 Hz ${f2(bassHigh.bright)}, vocal region of a low sine ${f2(vocLow.bright)} (gated), vocal saw rough ${f2(vocSaw.rough)}`);

  // Full pipeline: a noise burst track vs a sine track through analyzePcm, and the live analyzer.
  const pipe = (x: Float32Array) => {
    const res = analyzePcm(x, x, SR);
    const t = res.timbre!;
    const mid = Math.floor(res.numFrames / 2);
    return { noise: t.mix.noise[mid], rough: t.mix.rough[mid], ok: res.timbre!.drums.attack.length === res.numFrames };
  };
  const pn = pipe(signal(4, () => (r() * 2 - 1) * 0.3));
  const ps = pipe(signal(4, (t) => 0.4 * Math.sin(2 * Math.PI * 330 * t)));
  check('timbre.pipeline', pn.ok && pn.noise > ps.noise + 0.4, `analyzePcm noise: noise ${f2(pn.noise)} vs sine ${f2(ps.noise)}`);
  const live = (x: Float32Array) => {
    const a = new RealtimeAnalyzer(SR);
    for (let i = 0; i < x.length; i += 512) a.process(x.subarray(i, i + 512), x.subarray(i, i + 512));
    return a.timbreLive.out.mix;
  };
  const ln = live(signal(3, () => (r() * 2 - 1) * 0.3));
  const ls = live(signal(3, (t) => saw(220, t)));
  check('timbre.live', ln.noise > ls.noise + 0.3 && ls.rough > 0.1, `live noise ${f2(ln.noise)} vs saw ${f2(ls.noise)}, saw rough ${f2(ls.rough)}`);
}
