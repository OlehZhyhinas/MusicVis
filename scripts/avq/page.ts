// Headless render page for the AV quality harness (served by scripts/avq/vite.avq.config.ts,
// driven over CDP by scripts/avq/render.ts). Exposes window.avq:
//   await avq.song(path)             decode + real analysis (cached in IndexedDB), hooks, moments
//   await avq.render(opts)           render a preset over clip windows at a fixed frame clock,
//                                    write <clips>/<preset>/<song>__<label>.{json,bin} (+ JPEG frames)
// One render job per page load: stages keep GPU state, so the driver reloads between jobs.

import { Engine, Stage } from '../../src/v2/engine';
import { SEEDS } from '../../src/v2/seeds';
import type { Genome } from '../../src/v2/genome';
import { analyzeAudio } from '../../src/analysis/analyze';
import { TimelineSampler } from '../../src/analysis/TimelineSampler';
import type { AnalysisResult, MusicState } from '../../src/types';
import { MelodyProbe, OfflineLive, SPEC_BANDS, monoMix } from './audio';
import { VIS_FIELDS, VisualFeatures } from './features';
import { FIELDS, MUSIC_FIELDS, THUMB_BYTES, THUMB_H, THUMB_W, type ClipHeader } from './format';
import { autoWindows, findHooks, songMoments, type ClipWindow, type Hook, type Moment, type SongData } from './music';

// ------------------------------------------------------------------ io

async function save(rel: string, data: BodyInit): Promise<void> {
  const r = await fetch('/avq/save?p=' + encodeURIComponent(rel), { method: 'POST', body: data });
  if (!r.ok) throw new Error('save failed ' + rel + ': ' + (await r.text()));
}

async function saveParts(pattern: string, first: number, parts: Uint8Array[]): Promise<void> {
  const head = new Uint32Array(1 + parts.length);
  head[0] = parts.length;
  parts.forEach((p, i) => (head[1 + i] = p.length));
  const r = await fetch(`/avq/save?split=1&first=${first}&p=${encodeURIComponent(pattern)}`, { method: 'POST', body: new Blob([head, ...parts] as BlobPart[]) });
  if (!r.ok) throw new Error('save failed ' + pattern);
}

export function slugOf(path: string): string {
  const base = path.split('/').pop()!.replace(/\.[^.]+$/, '');
  return base.toLowerCase().replace(/\(.*?\)/g, '').replace(/[^a-z0-9]+/g, '-').replace(/^-+|-+$/g, '').slice(0, 40) || 'song';
}

// Analysis cache (IndexedDB, structured clone keeps the typed arrays).
const DB = 'avq';
function idb(): Promise<IDBDatabase> {
  return new Promise((ok, fail) => {
    const r = indexedDB.open(DB, 1);
    r.onupgradeneeded = () => r.result.createObjectStore('analysis');
    r.onsuccess = () => ok(r.result);
    r.onerror = () => fail(r.error);
  });
}
async function cacheGet(key: string): Promise<AnalysisResult | undefined> {
  const db = await idb();
  return new Promise((ok) => {
    const r = db.transaction('analysis').objectStore('analysis').get(key);
    r.onsuccess = () => ok(r.result as AnalysisResult | undefined);
    r.onerror = () => ok(undefined);
  });
}
async function cachePut(key: string, v: AnalysisResult): Promise<void> {
  const db = await idb();
  await new Promise<void>((ok) => {
    const tx = db.transaction('analysis', 'readwrite');
    tx.objectStore('analysis').put(v, key);
    tx.oncomplete = () => ok();
    tx.onerror = () => ok();
  });
}

// ------------------------------------------------------------------ song

interface Song {
  path: string;
  slug: string;
  sr: number;
  pcm: Float32Array;
  result: AnalysisResult;
  data: SongData;
  hooks: Hook[];
  moments: Moment[];
  analysisMs: number;
}

const songs = new Map<string, Song>();

function b64(a: Float32Array): string {
  const u = new Uint8Array(a.buffer, a.byteOffset, a.byteLength);
  let s = '';
  for (let i = 0; i < u.length; i += 0x8000) s += String.fromCharCode(...u.subarray(i, i + 0x8000));
  return btoa(s);
}

function songData(r: AnalysisResult): SongData {
  return {
    duration: r.duration, frameRate: r.frameRate, numFrames: r.numFrames, bpm: r.bpm,
    beats: Array.from(r.beats), downbeats: Array.from(r.downbeats), beatsPerBar: r.beatsPerBar,
    sections: r.sections.map((s) => ({ ...s })), repeats: r.repeats?.map((x) => ({ group: x.group, of: x.of, n: x.n, sim: x.sim })),
    chroma: r.chroma, loudness: r.loudness, stems: r.stems, stemOnsets: r.stemOnsets, stemPresence: r.stemPresence,
  };
}

async function loadSong(path: string): Promise<Song> {
  const have = songs.get(path);
  if (have) return have;
  const buf = await (await fetch('/avq/file?p=' + encodeURIComponent(path))).arrayBuffer();
  const key = path + ':' + buf.byteLength;
  const ctx = new OfflineAudioContext(2, 1, 44100);
  const audio = await ctx.decodeAudioData(buf);
  const t0 = performance.now();
  let result = await cacheGet(key);
  if (!result) {
    result = await analyzeAudio(audio);
    await cachePut(key, result);
  }
  const analysisMs = performance.now() - t0;
  const left = audio.getChannelData(0);
  const right = audio.numberOfChannels > 1 ? audio.getChannelData(1) : left;
  const data = songData(result);
  const s: Song = {
    path, slug: slugOf(path), sr: audio.sampleRate, pcm: monoMix(left, right), result, data,
    hooks: findHooks(data), moments: songMoments(data), analysisMs,
  };
  songs.set(path, s);
  // Song dump for the Node tools (metrics, sheets): analysis subset with base64 float32 arrays.
  const enc = (rec: Record<string, Float32Array>) => Object.fromEntries(Object.entries(rec).map(([k, v]) => [k, b64(v)]));
  await save(`songs/${s.slug}.json`, JSON.stringify({
    path, slug: s.slug, duration: data.duration, frameRate: data.frameRate, numFrames: data.numFrames, bpm: data.bpm,
    beats: data.beats, downbeats: data.downbeats, beatsPerBar: data.beatsPerBar, sections: data.sections, repeats: data.repeats,
    keys: result.keys, hooks: s.hooks, moments: s.moments,
    chroma: b64(data.chroma), loudness: b64(data.loudness), complexity: b64(result.complexity),
    stems: enc(data.stems), stemOnsets: enc(data.stemOnsets), stemPresence: enc(data.stemPresence),
  }));
  return s;
}

// ------------------------------------------------------------------ render

function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export interface RenderOpts {
  song: string;
  /** Seed origin id (e.g. 'E14'), or pass genome. */
  preset?: string;
  genome?: Genome;
  name?: string;
  /** 'auto' (drop + hook windows), 'song' (whole song), or explicit windows. */
  clips?: 'auto' | 'song' | ClipWindow[];
  w?: number;
  h?: number;
  fps?: number;
  seed?: number;
  /** Seconds rendered before each window (feedback and exposure settle); not recorded. */
  warm?: number;
  /** Save every recorded frame as JPEG (for MP4 review clips and filmstrips). */
  frames?: boolean;
  jpegQuality?: number;
  particleCap?: number;
  flameCap?: number;
}

let eng: Engine | null = null;
function engine(): Engine {
  return (eng ??= new Engine(document.getElementById('c') as HTMLCanvasElement));
}

const tick = () => new Promise((r) => setTimeout(r, 0));

async function programs(e: Engine, g: Genome) {
  for (let i = 0; i < 2000; i++) {
    const p = e.cache.get(g, i > 20);
    if (p) return p;
    if (e.cache.failed(g)) throw new Error('compile failed: ' + e.cache.failed(g));
    await new Promise((r) => setTimeout(r, 10));
  }
  throw new Error('compile timeout');
}

function writeMusic(row: Float32Array, s: MusicState, mel: { midi: number; salience: number }): void {
  const v: number[] = [
    s.time,
    s.stems.drums, s.stems.bass, s.stems.vocals, s.stems.other,
    s.stemOnsets.drums, s.stemOnsets.bass, s.stemOnsets.vocals, s.stemOnsets.other,
    s.stemPresence.drums, s.stemPresence.bass, s.stemPresence.vocals, s.stemPresence.other,
    s.loudness, s.bass, s.mid, s.treb,
    s.beatPhase, s.beatPulse, s.barPhase, s.barPulse, s.onBeat ? 1 : 0, s.onBar ? 1 : 0, s.beatIndex, s.barIndex,
    s.sectionIndex, s.sectionChanged ? 1 : 0, s.dropPulse, s.buildIntensity, s.timeToDrop ?? Infinity,
    s.repeatGroup ?? -1, s.repeatIndex ?? -1, s.repeatSim ?? 0,
    s.chord ?? -1, s.tension ?? 0, s.chordPulse ?? 0, s.keyHue,
    mel.midi, mel.salience,
    ...Array.from(s.chroma),
  ];
  if (v.length !== MUSIC_FIELDS.length) throw new Error(`music row ${v.length} != ${MUSIC_FIELDS.length}`);
  row.set(v, 0);
}

async function renderWindow(song: Song, g: Genome, presetId: string, presetName: string, win: ClipWindow, o: Required<Omit<RenderOpts, 'genome' | 'preset' | 'clips' | 'name' | 'song'>>): Promise<{ base: string; frames: number; ms: number }> {
  const e = engine();
  const progs = await programs(e, g);
  Math.random = mulberry32(o.seed);
  const st = new Stage(e, { offscreen: true, particleCap: o.particleCap, flameCap: o.flameCap });
  st.resize(o.w, o.h);
  const slot = st.makeSlot(g, progs);
  st.slots = [slot];
  st.resetHistory();
  const sampler = new TimelineSampler(song.result);
  const live = new OfflineLive(song.pcm, song.sr);
  const melody = new MelodyProbe(song.pcm, song.sr);
  const vis = new VisualFeatures(o.w, o.h);
  const dt = 1 / o.fps;
  const t0 = Math.max(0, win.start - o.warm);
  const kStart = Math.round((win.start - t0) * o.fps); // frames before the window
  const kEnd = Math.round((win.end - t0) * o.fps);
  const n = kEnd - kStart;
  const F = FIELDS.length;
  const rows = new Float32Array(n * F);
  const thumbs = new Uint8Array(n * THUMB_BYTES);
  const specs = new Uint8Array(n * SPEC_BANDS);
  const px = new Uint8Array(o.w * o.h * 4);
  const row = new Float32Array(F);
  const base = `${presetId}/${song.slug}__${win.label}`;
  const framesDir = `frames/${base}`;
  const canvas = o.frames ? new OffscreenCanvas(o.w, o.h) : null;
  const ctx2 = canvas?.getContext('2d') ?? null;
  const img = ctx2 ? ctx2.createImageData(o.w, o.h) : null;
  let batch: Promise<Blob>[] = [];
  let batchFirst = 0;
  const flush = async () => {
    if (!batch.length) return;
    const blobs = await Promise.all(batch);
    const parts = await Promise.all(blobs.map(async (b) => new Uint8Array(await b.arrayBuffer())));
    await saveParts(`${framesDir}/%06d.jpg`, batchFirst, parts);
    batch = [];
  };
  const tStart = performance.now();
  for (let k = 1; k <= kEnd; k++) {
    const t = t0 + k * dt;
    const state = sampler.sample(t, dt, true, live.read(t, dt));
    st.render(state, dt, 'out');
    if (k <= kStart) continue;
    const i = k - kStart - 1;
    st.readPixels(px);
    const mel = melody.read(t, specs.subarray(i * SPEC_BANDS, (i + 1) * SPEC_BANDS));
    writeMusic(row, state, mel);
    row.set(vis.update(px), MUSIC_FIELDS.length);
    rows.set(row, i * F);
    thumbs.set(vis.thumb, i * THUMB_BYTES);
    if (ctx2 && img && canvas) {
      const W4 = o.w * 4;
      for (let y = 0; y < o.h; y++) img.data.set(px.subarray((o.h - 1 - y) * W4, (o.h - y) * W4), y * W4);
      ctx2.putImageData(img, 0, 0);
      if (!batch.length) batchFirst = i;
      batch.push(canvas.convertToBlob({ type: 'image/jpeg', quality: o.jpegQuality }));
      if (batch.length >= 60) await flush();
    }
    if (i % 90 === 0) await tick();
  }
  await flush();
  const ms = performance.now() - tStart;
  st.disposeSlot(slot);
  const header: ClipHeader = {
    version: 1,
    preset: { id: presetId, name: presetName },
    song: { slug: song.slug, path: song.path, duration: song.data.duration, bpm: song.data.bpm, beatsPerBar: song.data.beatsPerBar },
    clip: { label: win.label, start: t0 + kStart * dt, end: t0 + kEnd * dt, warm: o.warm },
    render: { w: o.w, h: o.h, fps: o.fps, seed: o.seed, ms: Math.round(ms), hq: e.hq },
    fields: [...FIELDS],
    frames: n,
    thumb: { w: THUMB_W, h: THUMB_H, offset: rows.byteLength },
    spec: { bands: SPEC_BANDS, offset: rows.byteLength + thumbs.byteLength },
    beats: song.data.beats, downbeats: song.data.downbeats, sections: song.data.sections,
    moments: song.moments, hooks: song.hooks,
    framesDir: o.frames ? framesDir : undefined,
  };
  await save(`clips/${base}.bin`, new Blob([rows, thumbs, specs] as BlobPart[]));
  await save(`clips/${base}.json`, JSON.stringify(header));
  return { base, frames: n, ms };
}

async function render(opts: RenderOpts) {
  const song = await loadSong(opts.song);
  let g = opts.genome;
  let id = opts.name ?? 'custom';
  let name = opts.name ?? 'custom';
  if (!g) {
    const seed = SEEDS.find((s) => s.origin === opts.preset);
    if (!seed) throw new Error('no seed ' + opts.preset);
    g = seed.genome;
    id = seed.origin;
    name = seed.name;
  }
  const o = {
    w: opts.w ?? 320, h: opts.h ?? 180, fps: opts.fps ?? 30, seed: opts.seed ?? 1, warm: opts.warm ?? 3,
    frames: opts.frames ?? false, jpegQuality: opts.jpegQuality ?? 0.85, particleCap: opts.particleCap ?? 262144, flameCap: opts.flameCap ?? 524288,
  };
  const clips = opts.clips ?? 'auto';
  const wins: ClipWindow[] = clips === 'auto' ? autoWindows(song.data, song.hooks) : clips === 'song' ? [{ label: 'song', start: 0, end: song.data.duration }] : clips;
  const out = [];
  for (const w of wins) out.push(await renderWindow(song, g, id, name, w, o));
  return { preset: id, name, song: song.slug, analysisMs: Math.round(song.analysisMs), hooks: song.hooks.map((h) => ({ bars: h.bars, n: h.occurrences.length, len: +h.len.toFixed(2), score: +h.score.toFixed(3) })), clips: out };
}

async function songInfo(path: string) {
  const s = await loadSong(path);
  return {
    slug: s.slug, duration: s.data.duration, bpm: s.data.bpm, analysisMs: Math.round(s.analysisMs),
    sections: s.data.sections.map((x) => `${x.start.toFixed(1)} ${x.label}`),
    moments: s.moments, hooks: s.hooks.map((h) => ({ bars: h.bars, len: h.len, score: h.score, distinct: h.distinct, salience: h.salience, at: h.occurrences.map((o) => +o.start.toFixed(2)) })),
    windows: autoWindows(s.data, s.hooks),
  };
}

const api = { song: songInfo, render, seeds: () => SEEDS.map((s) => s.origin), fields: () => [...FIELDS], vis: () => [...VIS_FIELDS] };
(window as unknown as { avq: typeof api }).avq = api;
document.getElementById('log')!.textContent = 'avq ready';
