// New-song detection lab: records the live analyzer's features at 10 Hz for the test songs, songs
// back to back (no gap, 2 s, a 15 s ad) and the DJ mixes, then scores a track-change rule on them
// (false triggers inside single songs, hits / latency at true song starts or DJ takeovers).
//   node --import ./scripts/analysis-test.hooks.mjs scripts/live/nsLab.ts
// Features are cached in .testdata/live/ns/.

import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { deserialize, serialize } from 'node:v8';
import { decode, LivePath, OUT, slugOf, SR, testSongs, type Pcm } from './common';
import { buildMixes } from './mix';
import { NewSongDetector } from '../../src/analysis/rtNewSong';

const RATE = 10;

interface Feat {
  name: string;
  truth: number[];
  /** Per 0.1 s: chroma (12), timbre (16), dbfs, gate, bpm, conf, barPhase. */
  n: number;
  chroma: Float32Array;
  timbre: Float32Array;
  db: Float32Array;
  gate: Float32Array;
  bpm: Float32Array;
  conf: Float32Array;
  bar: Float32Array;
  /** Times the in-analyzer detector (if any) fired. */
  fired: number[];
}

function record(name: string, pcm: Pcm, truth: number[]): Feat {
  const lp = new LivePath(pcm);
  const a = lp.analyzer;
  const dur = pcm.left.length / pcm.sr;
  const n = Math.floor(dur * RATE);
  const f: Feat = { name, truth, n, chroma: new Float32Array(n * 12), timbre: new Float32Array(n * 16), db: new Float32Array(n), gate: new Float32Array(n), bpm: new Float32Array(n), conf: new Float32Array(n), bar: new Float32Array(n), fired: [] };
  for (let i = 0; i < n; i++) {
    const t = (i + 1) / RATE;
    lp.feed(t);
    f.chroma.set(a.chroma, i * 12);
    f.timbre.set(a.timbre, i * 16);
    f.db[i] = a.dbfs;
    f.gate[i] = a.gate;
    f.bpm[i] = a.beat.bpm;
    f.conf[i] = a.beat.confidence;
    f.bar[i] = ((((a.beat.positionAt(a.streamTime) - a.beat.downbeatSlot) / 4) % 1) + 1) % 1;
  }
  return f;
}

function cached(name: string, make: () => { pcm: Pcm; truth: number[] }): Feat {
  const p = join(OUT, 'ns', name + '.v8');
  if (existsSync(p)) return deserialize(readFileSync(p)) as Feat;
  const { pcm, truth } = make();
  const f = record(name, pcm, truth);
  mkdirSync(join(OUT, 'ns'), { recursive: true });
  writeFileSync(p, serialize(f));
  return f;
}

function concat(parts: Pcm[]): Pcm {
  const n = parts.reduce((s, p) => s + p.left.length, 0);
  const left = new Float32Array(n), right = new Float32Array(n);
  let o = 0;
  for (const p of parts) {
    left.set(p.left, o);
    right.set(p.right, o);
    o += p.left.length;
  }
  return { sr: SR, left, right };
}
const silence = (s: number): Pcm => ({ sr: SR, left: new Float32Array(Math.round(s * SR)), right: new Float32Array(Math.round(s * SR)) });
function excerpt(p: Pcm, a: number, b: number, gain = 1): Pcm {
  const i0 = Math.round(a * SR), i1 = Math.min(p.left.length, Math.round(b * SR));
  const l = p.left.slice(i0, i1), r = p.right.slice(i0, i1);
  for (let i = 0; i < l.length; i++) {
    const g = Math.min(1, i / (0.3 * SR), (l.length - i) / (0.3 * SR)) * gain;
    l[i] *= g;
    r[i] *= g;
  }
  return { sr: SR, left: l, right: r };
}
const find = (part: string) => testSongs().find((p) => slugOf(p).includes(part))!;

export function labInputs(): { singles: Feat[]; seqs: Feat[] } {
  const singles = testSongs().map((p) => cached('song-' + slugOf(p), () => ({ pcm: decode(p), truth: [] })));
  const seqs: Feat[] = [];
  const sets: [string, string[]][] = [
    ['A', ['inna-morenito', 'lean-on', 'ghosts', 'saxobeat']],
    ['B', ['stromae', 'fireflies', 'thinking-out-loud', 'radioactive', 'thrift-shop', 't-a-t-u', 'waiting-for-love', 'turn-down', 'animals']],
  ];
  const gaps: [string, () => Pcm[]][] = [
    ['gap0', () => []],
    ['gap2', () => [silence(2)]],
    ['ad', () => [silence(0.5), excerpt(decode(find('thrift-shop')), 60, 75, 0.5), silence(0.5)]],
  ];
  for (const [set, songs] of sets) {
    for (const [g, gap] of gaps) {
      if (set === 'B' && g === 'ad') continue;
      seqs.push(cached(`seq-${set}-${g}`, () => {
        const parts: Pcm[] = [];
        const truth: number[] = [];
        let t = 0;
        songs.forEach((s, i) => {
          if (i > 0) {
            for (const x of gap()) {
              parts.push(x);
              t += x.left.length / SR;
            }
            truth.push(t);
          }
          const p = decode(find(s));
          parts.push(p);
          t += p.left.length / SR;
        });
        return { pcm: concat(parts), truth };
      }));
    }
  }
  let mixes: ReturnType<typeof buildMixes> | null = null;
  for (const name of ['mix-club', 'mix-pop']) {
    seqs.push(cached(name, () => {
      mixes ??= buildMixes();
      const m = mixes.find((x) => x.name === name)!;
      return { pcm: m.pcm, truth: m.transitions.filter((x) => Number.isFinite(x.takeover)).map((x) => x.takeover) };
    }));
  }
  return { singles, seqs };
}

/** Run the detector over recorded features; returns firing times. */
export function runDetector(f: Feat, params: Record<string, number> = {}): number[] {
  const d = new NewSongDetector(params);
  const out: number[] = [];
  for (let i = 0; i < f.n; i++) {
    const t = (i + 1) / RATE;
    const fired = d.push(t, f.chroma.subarray(i * 12, i * 12 + 12), f.timbre.subarray(i * 16, i * 16 + 16), f.db[i], f.gate[i], f.bpm[i], f.conf[i], f.bar[i]);
    if (fired) out.push(t);
  }
  return out;
}

export function evaluate(params: Record<string, number> = {}, verbose = true): { falses: number; hits: number; total: number; lat: number[] } {
  const { singles, seqs } = labInputs();
  let falses = 0, hits = 0, total = 0;
  const lat: number[] = [];
  const log = (s: string) => verbose && console.log(s);
  for (const f of singles) {
    const det = runDetector(f, params);
    falses += det.length;
    if (det.length) log(`FALSE ${f.name}: ${det.map((x) => x.toFixed(0)).join(', ')}`);
  }
  for (const f of seqs) {
    const det = runDetector(f, params);
    const used = new Set<number>();
    const ls: string[] = [];
    for (const b of f.truth) {
      total++;
      const j = det.findIndex((d, k) => !used.has(k) && d >= b - 5 && d <= b + 25);
      if (j >= 0) {
        used.add(j);
        hits++;
        lat.push(det[j] - b);
        ls.push((det[j] - b).toFixed(1));
      } else ls.push('miss');
    }
    const extra = det.filter((_, k) => !used.has(k));
    falses += extra.length;
    log(`${f.name.padEnd(14)} truth ${f.truth.length} hits ${used.size} lat [${ls.join(' ')}] false ${extra.length}${extra.length ? ' at ' + extra.map((x) => x.toFixed(0)).join(',') : ''}`);
  }
  lat.sort((p, q) => p - q);
  log(`TOTAL false ${falses}, hits ${hits}/${total}, median latency ${lat.length ? lat[lat.length >> 1].toFixed(1) : 'n/a'} s, max ${lat.length ? lat[lat.length - 1].toFixed(1) : 'n/a'} s`);
  return { falses, hits, total, lat };
}

if (import.meta.url === `file://${process.argv[1]}` && !process.argv.includes("diag") && !process.argv.includes("search")) {
  const params: Record<string, number> = {};
  for (const a of process.argv.slice(2)) {
    const [k, v] = a.split('=');
    if (v !== undefined) params[k] = Number(v);
  }
  evaluate(params);
}

/** Novelty traces with the detector never firing: per song the highest sustained novelty, per boundary the peak after it. */
export function diag(params: Record<string, number> = {}): void {
  const { singles, seqs } = labInputs();
  const comp = process.argv.find((a) => a.startsWith('comp:'))?.slice(5);
  const trace = (f: Feat) => {
    const d = new NewSongDetector({ ...params, thr: 1e9 });
    const nov = new Float32Array(f.n);
    const resets = f.truth.map((b) => b + 25);
    for (let i = 0; i < f.n; i++) {
      if (resets.length && (i + 1) / RATE >= resets[0]) {
        resets.shift();
        d.forget();
      }
      d.push((i + 1) / RATE, f.chroma.subarray(i * 12, i * 12 + 12), f.timbre.subarray(i * 16, i * 16 + 16), f.db[i], f.gate[i], f.bpm[i], f.conf[i], f.bar[i]);
      nov[i] = comp ? (d.comp as Record<string, number>)[comp] : d.novelty;
    }
    return nov;
  };
  // Sustained novelty: the min over a 3 s window, maximised over time.
  const sus = (nov: Float32Array, a: number, b: number) => {
    let best = 0;
    for (let i = Math.max(0, Math.floor(a * RATE)); i + 3 * RATE < Math.min(nov.length, b * RATE); i++) {
      let m = Infinity;
      for (let j = i; j < i + 3 * RATE; j++) m = Math.min(m, nov[j]);
      best = Math.max(best, m);
    }
    return best;
  };
  const w: string[] = [];
  for (const f of singles) {
    const nov = trace(f);
    w.push(sus(nov, 0, f.n / RATE).toFixed(2));
  }
  console.log('within songs (max sustained): ' + w.join(' '));
  for (const f of seqs) {
    const nov = trace(f);
    console.log(`${f.name.padEnd(14)} boundaries ${f.truth.map((b) => sus(nov, b, b + 20).toFixed(2)).join(' ')}`);
  }
}
if (process.argv.includes('diag')) {
  const params: Record<string, number> = {};
  for (const a of process.argv.slice(2)) {
    const [k, v] = a.split('=');
    if (v !== undefined) params[k] = Number(v);
  }
  diag(params);
}

/** Grid search over conjunction rules on component traces (oracle memory resets at the truth). */
export function search(): void {
  const { singles, seqs } = labInputs();
  type Tr = { name: string; truth: number[]; c: Float32Array[]; gap: Uint8Array; single: boolean };
  const trs: Tr[] = [];
  for (const [f, single] of [...singles.map((x) => [x, true] as const), ...seqs.map((x) => [x, false] as const)]) {
    const d = new NewSongDetector({ thr: 1e9 });
    const c = [0, 1, 2, 3].map(() => new Float32Array(f.n));
    const gap = new Uint8Array(f.n);
    const resets = f.truth.map((b) => b + 25);
    let silent = 0, lastGap = -1e9;
    for (let i = 0; i < f.n; i++) {
      const t = (i + 1) / RATE;
      if (resets.length && t >= resets[0]) {
        resets.shift();
        d.forget();
      }
      d.push(t, f.chroma.subarray(i * 12, i * 12 + 12), f.timbre.subarray(i * 16, i * 16 + 16), f.db[i], f.gate[i], f.bpm[i], f.conf[i], f.bar[i]);
      c[0][i] = d.comp.tempo; c[1][i] = d.comp.key; c[2][i] = d.comp.timbre; c[3][i] = d.comp.level;
      if (f.gate[i] <= 0.5) silent += 0.1;
      else {
        if (silent >= 1) lastGap = t;
        silent = 0;
      }
      gap[i] = t - lastGap < 12 ? 1 : 0;
    }
    trs.push({ name: f.name, truth: f.truth, c, gap, single });
  }
  const TH = [[0.5, 0.99], [0.06, 0.1, 0.15], [0.8, 1.0, 1.3], [1.0, 1.5, 2.0]];
  const results: string[] = [];
  const combos: number[][] = [];
  for (const a of TH[0]) for (const b of TH[1]) for (const c of TH[2]) for (const e of TH[3]) combos.push([a, b, c, e]);
  for (const th of combos) for (const k of [2, 3]) for (const kGap of [1, 2]) for (const sustain of [3, 5]) {
    let falses = 0, hits = 0, total = 0;
    const lat: number[] = [];
    for (const tr of trs) {
      const fires: number[] = [];
      let above = 0, last = -1e9;
      for (let i = 0; i < tr.c[0].length; i += 10) {
        const t = (i + 1) / RATE;
        let cnt = 0;
        for (let j = 0; j < 4; j++) if (tr.c[j][i] > th[j]) cnt++;
        const need = tr.gap[i] ? kGap : k;
        above = cnt >= need ? above + 1 : 0;
        if (above >= sustain && t - last > 20) {
          fires.push(t);
          last = t;
          above = 0;
        }
      }
      const used = new Set<number>();
      for (const b of tr.truth) {
        total++;
        const j = fires.findIndex((d, q) => !used.has(q) && d >= b - 5 && d <= b + 25);
        if (j >= 0) {
          used.add(j);
          hits++;
          lat.push(fires[j] - b);
        }
      }
      falses += fires.length - used.size;
    }
    lat.sort((p, q) => p - q);
    results.push(`${falses}\t${hits}/${total}\tmed ${lat.length ? lat[lat.length >> 1].toFixed(1) : '-'}\tth ${th.join(',')} k ${k} kGap ${kGap} sus ${sustain}`);
  }
  results.sort((x, y) => Number(x.split('\t')[0]) - Number(y.split('\t')[0]) || Number(y.split('\t')[1].split('/')[0]) - Number(x.split('\t')[1].split('/')[0]));
  console.log(results.slice(0, 25).join('\n'));
}
if (process.argv.includes('search')) search();
