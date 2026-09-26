// Live vs offline: runs real songs through the live path (RealtimeAnalyzer +
// RealtimeSampler, rendered at 60 fps) with the greedy note tracker (visual
// lag 0) and with LiveLookahead at visual lags V, and scores each against the
// offline analysis (analyzePcm) of the same audio.
//
// Decodes with ffmpeg to raw PCM in memory; nothing is ever played.
//
// Run: node --import ./scripts/analysis-test.hooks.mjs scripts/live-vs-offline.ts [options]
//   --dir <folder>      songs (default ~/Downloads/YoutubeToMp3)
//   --songs a,b         only files whose name contains one of these (case-insensitive)
//   --lags 0.05,0.1     visual lags for LiveLookahead, seconds (default 0,0.05,0.1,0.15)
//   --secs N            only the first N seconds of each song
//
// Metrics (on the display clock, which runs V behind the sound):
//   onset F1 seen   note starts as the visuals show them, within +-50 ms of the offline note starts
//   onset F1 pos    note starts as the tracker places them (latency removed), +-50 ms
//   lat             median delay of a shown note start after the offline one (matched within 150 ms)
//   pitch           share of matched notes (pos) within 0.5 semitone of the offline pitch
//   legato r        correlation of the per-frame legato with the offline one (after 5 s)
//   beat F1         onBeat events vs offline beats, +-70 ms
//   sect F1         section changes vs offline section starts, +-3 s

import { spawnSync } from 'node:child_process';
import { readdirSync } from 'node:fs';
import { homedir } from 'node:os';
import { join } from 'node:path';
import { analyzePcm } from '../src/analysis/analyzePcm';
import { RealtimeAnalyzer } from '../src/analysis/RealtimeAnalyzer';
import { RealtimeSampler } from '../src/analysis/RealtimeSampler';
import { LiveLookahead } from '../src/analysis/LiveLookahead';
import { neutralNotes, sampleNoteTrack } from '../src/analysis/notes';
import type { AnalysisResult, LiveAudioFrame } from '../src/types';

const SR = 44100;
const BLOCK = 512;
const FPS = 60;

function arg(name: string, def: string): string {
  const i = process.argv.indexOf(`--${name}`);
  return i >= 0 && i + 1 < process.argv.length ? process.argv[i + 1] : def;
}

const dir = arg('dir', join(homedir(), 'Downloads', 'YoutubeToMp3'));
const only = arg('songs', '').split(',').map((s) => s.trim().toLowerCase()).filter(Boolean);
const lags = arg('lags', '0,0.05,0.1,0.15').split(',').map(Number).filter((x) => Number.isFinite(x) && x >= 0);
const secs = Number(arg('secs', '0'));

function decode(file: string): { left: Float32Array; right: Float32Array } {
  const args = ['-v', 'error', '-i', file, '-f', 'f32le', '-ac', '2', '-ar', String(SR)];
  if (secs > 0) args.push('-t', String(secs));
  args.push('pipe:1');
  const r = spawnSync('ffmpeg', args, { maxBuffer: 1 << 30 });
  if (r.status !== 0) throw new Error(`ffmpeg failed on ${file}: ${r.stderr?.toString()}`);
  const buf = r.stdout;
  const all = new Float32Array(buf.buffer, buf.byteOffset, Math.floor(buf.byteLength / 4));
  const N = all.length >> 1;
  const left = new Float32Array(N), right = new Float32Array(N);
  for (let i = 0; i < N; i++) {
    left[i] = all[2 * i];
    right[i] = all[2 * i + 1];
  }
  return { left, right };
}

interface Onset {
  seen: number;
  pos: number;
  pitch: number;
}

interface LiveRun {
  onsets: Onset[];
  legato: { t: number; v: number }[];
  beats: number[];
  sections: number[];
  ms: number;
}

const silentFrame = (): LiveAudioFrame => ({
  bass: 0, mid: 0, treb: 0, bassAtt: 0, midAtt: 0, trebAtt: 0, waveform: new Float32Array(1024), spectrum: new Float32Array(512),
});

/** lag null: the greedy tracker; a number: LiveLookahead with that visual lag. */
function runLive(left: Float32Array, right: Float32Array, lag: number | null): LiveRun {
  const a = new RealtimeAnalyzer(SR);
  const sampler = new RealtimeSampler(a);
  const la = lag === null ? null : new LiveLookahead(a, lag);
  if (la) sampler.noteSource = la.sampleNotes;
  const frame = silentFrame();
  const out: LiveRun = { onsets: [], legato: [], beats: [], sections: [], ms: 0 };
  let tick = 0;
  let lastStart = -Infinity;
  let cur: { o: Onset; ps: number[] } | null = null;
  const finish = () => {
    if (!cur) return;
    const ps = cur.ps.sort((x, y) => x - y);
    cur.o.pitch = ps.length ? ps[ps.length >> 1] : NaN;
    out.onsets.push(cur.o);
    cur = null;
  };
  const t0 = performance.now();
  for (let i = 0; i < left.length; i += BLOCK) {
    const l = left.subarray(i, Math.min(left.length, i + BLOCK));
    const r = right.subarray(i, Math.min(right.length, i + BLOCK));
    if (la) la.process(l, r);
    else a.process(l, r);
    while (tick + 1 / FPS <= a.streamTime) {
      tick += 1 / FPS;
      const s = sampler.sample(tick, 1 / FPS, true, frame);
      if (s.onBeat) out.beats.push(tick);
      if (s.sectionChanged) out.sections.push(tick);
      const ns = s.notes;
      if (!ns) continue;
      if (tick > 5) out.legato.push({ t: tick, v: ns.legato });
      const m = ns.recent[ns.recent.length - 1];
      if (!m) continue;
      let pos: number;
      if (la) pos = tick - m.age;
      else {
        // Greedy marks age on the analyzer's frame clock; place the start at its frame's centre.
        const fStart = Math.round(a.frames - m.age * a.frameRate);
        pos = (fStart * a.hop - a.n / 2) / a.sr;
      }
      if (pos > lastStart + 0.03) {
        finish();
        lastStart = pos;
        cur = { o: { seen: tick, pos, pitch: NaN }, ps: [] };
      }
      if (cur && !m.ended && Math.abs(pos - lastStart) < 0.03) cur.ps.push(ns.pitch);
    }
  }
  finish();
  out.ms = performance.now() - t0;
  return out;
}

/** Greedy one-to-one matching within tol: [ref index, est index] pairs. */
function match(ref: number[], est: number[], tol: number): [number, number][] {
  const used = new Uint8Array(est.length);
  const pairs: [number, number][] = [];
  let j0 = 0;
  for (let i = 0; i < ref.length; i++) {
    while (j0 < est.length && est[j0] < ref[i] - tol) j0++;
    let best = -1, bd = Infinity;
    for (let j = j0; j < est.length && est[j] <= ref[i] + tol; j++) {
      if (used[j]) continue;
      const d = Math.abs(est[j] - ref[i]);
      if (d < bd) {
        bd = d;
        best = j;
      }
    }
    if (best >= 0) {
      used[best] = 1;
      pairs.push([i, best]);
    }
  }
  return pairs;
}

function f1(ref: number[], est: number[], tol: number): number {
  if (!ref.length && !est.length) return 1;
  const m = match(ref, est, tol).length;
  const p = est.length ? m / est.length : 0, r = ref.length ? m / ref.length : 0;
  return p + r > 0 ? (2 * p * r) / (p + r) : 0;
}

function pearson(x: number[], y: number[]): number {
  const n = x.length;
  if (n < 3) return NaN;
  let mx = 0, my = 0;
  for (let i = 0; i < n; i++) {
    mx += x[i];
    my += y[i];
  }
  mx /= n;
  my /= n;
  let sxy = 0, sxx = 0, syy = 0;
  for (let i = 0; i < n; i++) {
    const dx = x[i] - mx, dy = y[i] - my;
    sxy += dx * dy;
    sxx += dx * dx;
    syy += dy * dy;
  }
  return sxx > 0 && syy > 0 ? sxy / Math.sqrt(sxx * syy) : NaN;
}

interface Score {
  f1Seen: number;
  f1Pos: number;
  lat: number;
  pitch: number;
  legato: number;
  beat: number;
  sect: number;
  notes: number;
}

function score(off: AnalysisResult, live: LiveRun): Score {
  const tr = off.notes!;
  const ref = tr.notes.map((n) => n.start);
  const refP = tr.notes.map((n) => n.pitch);
  const seen = live.onsets.map((o) => o.seen).sort((x, y) => x - y);
  const byPos = [...live.onsets].sort((x, y) => x.pos - y.pos);
  const pos = byPos.map((o) => o.pos);
  const pairs = match(ref, pos, 0.05);
  let good = 0;
  for (const [i, j] of pairs) if (Math.abs(byPos[j].pitch - refP[i]) <= 0.5) good++;
  const latPairs = match(ref, seen, 0.15).map(([i, j]) => seen[j] - ref[i]).sort((x, y) => x - y);
  const ns = neutralNotes();
  const xs: number[] = [], ys: number[] = [];
  for (const { t, v } of live.legato) {
    sampleNoteTrack(tr, off.frameRate, t, ns);
    xs.push(v);
    ys.push(ns.legato);
  }
  const secStarts = off.sections.slice(1).map((s) => s.start);
  return {
    f1Seen: f1(ref, seen, 0.05),
    f1Pos: f1(ref, pos, 0.05),
    lat: latPairs.length ? latPairs[latPairs.length >> 1] : NaN,
    pitch: pairs.length ? good / pairs.length : NaN,
    legato: pearson(xs, ys),
    beat: f1(Array.from(off.beats), live.beats, 0.07),
    sect: f1(secStarts, live.sections, 3),
    notes: live.onsets.length,
  };
}

const fmt = (x: number, d = 2) => (Number.isFinite(x) ? x.toFixed(d) : ' -  ');

function main(): void {
  const files = readdirSync(dir)
    .filter((f) => /\.(mp3|m4a|wav|flac|ogg)$/i.test(f))
    .filter((f) => !/\(1\)\.\w+$/.test(f)) // skip duplicate downloads
    .filter((f) => !only.length || only.some((o) => f.toLowerCase().includes(o)))
    .sort();
  if (!files.length) {
    console.log(`No songs in ${dir}`);
    return;
  }
  const modes: { name: string; lag: number | null }[] = [{ name: 'greedy V=0', lag: null }, ...lags.map((l) => ({ name: `look V=${Math.round(l * 1000)}ms`, lag: l }))];
  const sums = modes.map(() => ({ f1Seen: 0, f1Pos: 0, lat: 0, pitch: 0, legato: 0, beat: 0, sect: 0, n: 0 }));
  const head = 'mode              notes  F1seen  F1pos   lat(ms)  pitch  legato r  beatF1  sectF1   x-real';
  for (const f of files) {
    const { left, right } = decode(join(dir, f));
    const dur = left.length / SR;
    const tOff = performance.now();
    const off = analyzePcm(left, right, SR);
    const offMs = performance.now() - tOff;
    console.log(`\n${f}  (${dur.toFixed(0)} s, offline ${off.notes!.notes.length} notes, ${(offMs / 1000).toFixed(1)} s)`);
    console.log(head);
    modes.forEach((m, k) => {
      const live = runLive(left, right, m.lag);
      const s = score(off, live);
      const su = sums[k];
      su.f1Seen += s.f1Seen;
      su.f1Pos += s.f1Pos;
      su.lat += Number.isFinite(s.lat) ? s.lat : 0;
      su.pitch += Number.isFinite(s.pitch) ? s.pitch : 0;
      su.legato += Number.isFinite(s.legato) ? s.legato : 0;
      su.beat += s.beat;
      su.sect += s.sect;
      su.n++;
      console.log(
        `${m.name.padEnd(17)} ${String(s.notes).padStart(5)}  ${fmt(s.f1Seen).padStart(6)}  ${fmt(s.f1Pos).padStart(5)}  ${fmt(s.lat * 1000, 0).padStart(7)}  ${fmt(s.pitch).padStart(6)}  ${fmt(s.legato).padStart(8)}  ${fmt(s.beat).padStart(6)}  ${fmt(s.sect).padStart(6)}   ${fmt(dur / (live.ms / 1000), 0).padStart(5)}`,
      );
    });
  }
  console.log(`\nMean over ${files.length} songs`);
  console.log(head);
  modes.forEach((m, k) => {
    const su = sums[k];
    const d = Math.max(1, su.n);
    console.log(
      `${m.name.padEnd(17)}        ${fmt(su.f1Seen / d).padStart(6)}  ${fmt(su.f1Pos / d).padStart(5)}  ${fmt((su.lat / d) * 1000, 0).padStart(7)}  ${fmt(su.pitch / d).padStart(6)}  ${fmt(su.legato / d).padStart(8)}  ${fmt(su.beat / d).padStart(6)}  ${fmt(su.sect / d).padStart(6)}`,
    );
  });
}

main();
