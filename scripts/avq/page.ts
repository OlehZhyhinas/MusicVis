// Headless render page for the AV quality harness (served by scripts/avq/vite.avq.config.ts,
// driven over CDP by scripts/avq/render.ts). Exposes window.avq:
//   await avq.song(path)             decode + real analysis (cached in IndexedDB), hooks, moments
//   await avq.render(opts)           render a preset over clip windows at a fixed frame clock,
//                                    write <clips>/<preset>/<song>__<label>.{json,bin} (+ JPEG frames)
// One render job per page load: stages keep GPU state, so the driver reloads between jobs.

import { Engine, Stage } from '../../src/v2/engine';
import { SEEDS } from '../../src/v2/seeds';
import { cloneGenome, schemaFor, type Genome } from '../../src/v2/genome';
import { paramsFor } from '../../src/v2/engine';
import { Embedder, EMB_DIM } from '../../src/v2/embedding';
import { analyzeAudio } from '../../src/analysis/analyze';
import { TimelineSampler } from '../../src/analysis/TimelineSampler';
import type { AnalysisResult, MusicState } from '../../src/types';
import { MelodyProbe, OfflineLive, SPEC_BANDS, monoMix, stemRegion, type StemId } from './audio';
import { VIS_FIELDS, VisualFeatures } from './features';
import { FIELDS, MUSIC_FIELDS, THUMB_BYTES, THUMB_H, THUMB_W, parseClip, type CfResult, type ClipHeader } from './format';
import { drawSheet } from './sheet';
import type { ReportCard } from './metrics';
import { autoWindows, findHooks, songMoments, type ClipWindow, type Hook, type Moment, type SongData } from './music';

// ------------------------------------------------------------------ io

/** Progress marker the driver can poll (avq.status()). */
let statusText = 'idle';
function status(s: string): void {
  statusText = s;
}

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
  status('fetch ' + path);
  const buf = await (await fetch('/avq/file?p=' + encodeURIComponent(path))).arrayBuffer();
  status('decode');
  const key = path + ':' + buf.byteLength;
  const ctx = new OfflineAudioContext(2, 1, 44100);
  const audio = await ctx.decodeAudioData(buf);
  const t0 = performance.now();
  status('cache');
  let result = await cacheGet(key);
  if (!result) {
    status('analyse');
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
  /** Embed every Nth recorded frame with DINOv2-small (0 = off). Written to <base>.emb.json. */
  embedEvery?: number;
}

let embedder: Embedder | null = null;
async function getEmbedder(): Promise<Embedder> {
  if (!embedder) embedder = new Embedder();
  if (!(await embedder.load())) throw new Error('embedder: ' + embedder.status.detail);
  return embedder;
}

let eng: Engine | null = null;
function engine(): Engine {
  return (eng ??= new Engine(document.getElementById('c') as HTMLCanvasElement));
}

// Yield to the event loop without timers (background tabs throttle setTimeout).
const chan = new MessageChannel();
const waiters: (() => void)[] = [];
chan.port1.onmessage = () => waiters.shift()?.();
const tick = () =>
  new Promise<void>((r) => {
    waiters.push(r);
    chan.port2.postMessage(0);
  });

async function programs(e: Engine, g: Genome) {
  for (let i = 0; i < 2000; i++) {
    const p = e.cache.get(g, i > 20);
    if (p) return p;
    if (e.cache.failed(g)) throw new Error('compile failed: ' + e.cache.failed(g));
    const t = performance.now();
    while (performance.now() - t < 10) await tick();
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
    s.timbre?.mix.bright ?? 0, s.timbre?.mix.noise ?? 0, s.complexity,
    ...Array.from(s.chroma),
  ];
  if (v.length !== MUSIC_FIELDS.length) throw new Error(`music row ${v.length} != ${MUSIC_FIELDS.length}`);
  row.set(v, 0);
}

async function renderWindow(song: Song, g: Genome, presetId: string, presetName: string, win: ClipWindow, o: Required<Omit<RenderOpts, 'genome' | 'preset' | 'clips' | 'name' | 'song'>>): Promise<{ base: string; frames: number; ms: number }> {
  const e = engine();
  status('compile');
  const progs = await programs(e, g);
  status('render ' + win.label);
  Math.random = mulberry32(o.seed);
  const st = new Stage(e, { offscreen: true, particleCap: o.particleCap, flameCap: o.flameCap });
  st.resize(o.w, o.h);
  const slot = st.makeSlot(g, progs);
  st.slots = [slot];
  st.resetHistory();
  const inst = new Instrument(st, slot, g);
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
  const emb = o.embedEvery > 0 ? await getEmbedder() : null;
  const embCanvas = emb ? document.createElement('canvas') : null;
  if (embCanvas) {
    embCanvas.width = o.w;
    embCanvas.height = o.h;
  }
  const embCtx = embCanvas?.getContext('2d') ?? null;
  const embImg = embCtx ? embCtx.createImageData(o.w, o.h) : null;
  const embIdx: number[] = [];
  const embVecs: number[][] = [];
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
    if (k <= kStart) {
      if (k % 30 === 0) {
        status(`warm ${win.label} ${k}/${kStart}`);
        await tick();
      }
      continue;
    }
    const i = k - kStart - 1;
    st.readPixels(px);
    const mel = melody.read(t, specs.subarray(i * SPEC_BANDS, (i + 1) * SPEC_BANDS));
    writeMusic(row, state, mel);
    row.set(vis.update(px), MUSIC_FIELDS.length);
    rows.set(row, i * F);
    inst.record();
    if (emb && embCanvas && embCtx && embImg && i % o.embedEvery === 0) {
      const W4 = o.w * 4;
      for (let y = 0; y < o.h; y++) embImg.data.set(px.subarray((o.h - 1 - y) * W4, (o.h - y) * W4), y * W4);
      embCtx.putImageData(embImg, 0, 0);
      const v = await emb.embed([embCanvas]);
      if (v) {
        embIdx.push(i);
        embVecs.push(v);
      }
    }
    thumbs.set(vis.thumb, i * THUMB_BYTES);
    if (ctx2 && img && canvas) {
      const W4 = o.w * 4;
      for (let y = 0; y < o.h; y++) img.data.set(px.subarray((o.h - 1 - y) * W4, (o.h - y) * W4), y * W4);
      ctx2.putImageData(img, 0, 0);
      if (!batch.length) batchFirst = i;
      batch.push(canvas.convertToBlob({ type: 'image/jpeg', quality: o.jpegQuality }));
      if (batch.length >= 60) await flush();
    }
    if (i % 90 === 0) {
      status(`render ${win.label} ${i}/${n}`);
      await tick();
    }
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
  await save(`clips/${base}.inst.json`, inst.json());
  if (emb) await save(`clips/${base}.emb.json`, JSON.stringify({ model: 'Xenova/dinov2-small', dim: EMB_DIM, every: o.embedEvery, device: emb.status.device, idx: embIdx, vecs: embVecs }));
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
    embedEvery: opts.embedEvery ?? 0,
  };
  const clips = opts.clips ?? 'auto';
  const wins: ClipWindow[] = clips === 'auto' ? autoWindows(song.data, song.hooks) : clips === 'song' ? [{ label: 'song', start: 0, end: song.data.duration }] : clips;
  const out = [];
  for (const w of wins) out.push(await renderWindow(song, g, id, name, w, o));
  return { preset: id, name, song: song.slug, analysisMs: Math.round(song.analysisMs), hooks: song.hooks.map((h) => ({ bars: h.bars, n: h.occurrences.length, len: +h.len.toFixed(2), score: +h.score.toFixed(3) })), clips: out };
}



// ------------------------------------------------------------------ engine instrumentation

/**
 * Per-frame readout from the engine itself (not pixels): each reaction's source signal, its
 * response after the curve and the driven parameter's value; the engine's frame signals
 * (activity, speed, spin, hit, surge, melody, harmony pulses); the harmony gene's motor output;
 * the blended camera / choreography pose. Stage internals are read through `any` so the app
 * stays untouched.
 */
export class Instrument {
  readonly names: string[] = [];
  readonly cols: number[][] = [];
  /** Per reaction: the driven parameter's spec range and genome value (to tell a pinned parameter). */
  readonly reactions: { src: string; target: string; gain: number; min: number; max: number; base: number }[] = [];
  private slot: any;
  private stage: any;
  private g: Genome;
  constructor(stage: Stage, slot: unknown, g: Genome) {
    this.stage = stage;
    this.slot = slot;
    this.g = g;
    g.reactions.slice(0, 6).forEach((r, j) => {
      const tag = `rx${j}:${r.src}>${r.g}${r.i}.${r.k}`;
      this.names.push(`${tag}:src`, `${tag}:resp`, `${tag}:value`);
      const spec = schemaFor(g, r.g, r.i)?.[r.k];
      const p = paramsFor(g, r.g, r.i);
      this.reactions.push({ src: r.src, target: `${r.g}${r.i}.${r.k}`, gain: r.gain, min: spec?.min ?? NaN, max: spec?.max ?? NaN, base: p?.[r.k] ?? NaN });
    });
    this.names.push(
      'F.act', 'F.speed', 'F.spin', 'F.hit', 'F.hitPulse', 'F.surge', 'F.melody', 'F.loud', 'F.drop', 'F.build',
      'F.tension', 'F.resolve', 'F.chordPulse', 'F.modPulse', 'F.chord', 'F.tonnetzX', 'F.tonnetzY',
      'harm.brk', 'harm.warp', 'harm.zoom', 'harm.roll', 'harm.hue', 'harm.sat', 'harm.exposure',
      'pose.zoom', 'pose.roll', 'pose.tx', 'pose.ty', 'pose.hue', 'pose.sat', 'pose.exposure', 'flash',
    );
    for (let i = 0; i < this.names.length; i++) this.cols.push([]);
  }
  record(): void {
    const v: number[] = [];
    const s = this.slot;
    this.g.reactions.slice(0, 6).forEach((r, j) => {
      const schema = schemaFor(this.g, r.g, r.i);
      const p = paramsFor(this.g, r.g, r.i);
      const value = schema && p && r.k in schema ? s.P(r.g, r.i, p, r.k, schema) : NaN;
      v.push(s.meters[j * 2], s.meters[j * 2 + 1], value);
    });
    const F = this.stage.sig.F;
    v.push(F.act, F.speed, F.spin, F.hit, F.hitPulse, F.surge, F.melody, F.loud, F.drop, F.build, F.tension, F.resolve, F.chordPulse, F.modPulse, F.chord, F.tonnetzX, F.tonnetzY);
    const h = this.stage.harmOut?.get(s);
    v.push(h?.brk ?? 0, h?.warp ?? 0, h?.zoom ?? 1, h?.roll ?? 0, h?.hue ?? 0, h?.sat ?? 1, h?.exposure ?? 1);
    const q = this.stage.pose;
    v.push(q.zoom, q.roll, q.tx, q.ty, q.hue, q.sat, q.exposure, this.stage.flash);
    for (let i = 0; i < v.length; i++) this.cols[i].push(Math.round(v[i] * 1e5) / 1e5);
  }
  json(): string {
    return JSON.stringify({ reactions: this.reactions, names: this.names, cols: this.cols.map((c) => c.map((x) => (Number.isFinite(x) ? x : null))) });
  }
}

// ------------------------------------------------------------------ counterfactuals

/**
 * A counterfactual variant of the base render. All variants render in lockstep with the base
 * (own Stage, own seeded Math.random stream, same frame clock), so any divergence comes from
 * the change alone:
 *   shift   the music timeline moved by `beats` (the animation clock stays; only MusicState
 *           and the live frame come from t + shift): does the picture follow the timing?
 *   offset  the music from `seconds` later in the song (content, not just timing)
 *   mute    one stem removed: its MusicState fields zeroed (stems, onsets, presence, timbre)
 *           and its spectral region (audio.ts stemRegion, scaled by the stem's presence) taken
 *           out of the emulated live spectrum / levels / waveform
 *   ablate  one reaction's gain set to 0 (same programs, same indices)
 *   gain    every level-like music value scaled by `factor` (0.98: an inaudible change). The
 *           divergence it causes is the chaos floor: how much the preset amplifies a
 *           perturbation that carries no musical meaning.
 */
export type Variant =
  | { kind: 'shift'; beats: number }
  | { kind: 'offset'; seconds: number }
  | { kind: 'mute'; stem: StemId }
  | { kind: 'ablate'; reaction: number }
  | { kind: 'gain'; factor: number };

export const variantId = (v: Variant): string =>
  v.kind === 'shift' ? `shift${v.beats}b` : v.kind === 'offset' ? `offset${Math.round(v.seconds)}s` : v.kind === 'mute' ? `mute-${v.stem}` : v.kind === 'gain' ? `gain${v.factor}` : `ablate-r${v.reaction}`;

class MusicSource {
  private sampler: TimelineSampler;
  private live: OfflineLive;
  private gains: Float32Array | null = null;
  constructor(private song: Song, private v: Variant | null) {
    this.sampler = new TimelineSampler(song.result);
    this.live = new OfflineLive(song.pcm, song.sr);
    if (v?.kind === 'mute') {
      this.gains = new Float32Array(1024);
      this.live.gain = this.gains;
    }
  }
  state(t: number, dt: number): MusicState {
    const v = this.v;
    const dur = this.song.data.duration;
    let tm = t;
    if (v?.kind === 'shift') tm = t + (v.beats * 60) / (this.song.data.bpm || 120);
    else if (v?.kind === 'offset') tm = (((t + v.seconds) % dur) + dur) % dur;
    if (v?.kind === 'mute' && this.gains) {
      const r = this.song.result;
      const f = Math.max(0, Math.min(r.numFrames - 1, Math.round(tm * r.frameRate)));
      const pres = r.stemPresence[v.stem][f];
      const hz = this.song.sr / 2048;
      for (let k = 0; k < 1024; k++) this.gains[k] = Math.max(0, 1 - pres * stemRegion(v.stem, k * hz));
    }
    const s = this.sampler.sample(Math.max(0, tm), dt, true, this.live.read(Math.max(0, tm), dt));
    s.time = t;
    if (v?.kind === 'gain') {
      const f = v.factor;
      for (const k of ['drums', 'bass', 'vocals', 'other'] as StemId[]) {
        s.stems[k] *= f;
        s.stemOnsets[k] *= f;
      }
      s.loudness *= f;
      s.bass *= f;
      s.mid *= f;
      s.treb *= f;
      s.bassAtt *= f;
      s.midAtt *= f;
      s.trebAtt *= f;
      for (let i = 0; i < s.spectrum.length; i++) s.spectrum[i] *= f;
      for (let i = 0; i < s.waveform.length; i++) s.waveform[i] *= f;
    }
    if (v?.kind === 'mute') {
      s.stems[v.stem] = 0;
      s.stemOnsets[v.stem] = 0;
      s.stemPresence[v.stem] = 0;
      if (s.timbre) s.timbre[v.stem] = { bright: 0, noise: 0, rough: 0, attack: 0 };
    }
    return s;
  }
}

/** Box-downsample an RGBA bottom-up frame to gw x gh RGB floats 0..1. */
function shrink(px: Uint8Array, w: number, h: number, gw: number, gh: number, out: Float32Array): void {
  out.fill(0);
  const sx = gw / w;
  const sy = gh / h;
  for (let y = 0; y < h; y++) {
    const gy = Math.min(gh - 1, Math.floor(y * sy));
    for (let x = 0; x < w; x++) {
      const i = (y * w + x) * 4;
      const k = (gy * gw + Math.min(gw - 1, Math.floor(x * sx))) * 3;
      out[k] += px[i];
      out[k + 1] += px[i + 1];
      out[k + 2] += px[i + 2];
    }
  }
  const norm = 1 / (255 * (w / gw) * (h / gh));
  for (let k = 0; k < out.length; k++) out[k] *= norm;
}

function meanAbs(a: Float32Array, b: Float32Array): number {
  let s = 0;
  for (let k = 0; k < a.length; k++) s += Math.abs(a[k] - b[k]);
  return s / a.length;
}

export interface CfOpts {
  song: string;
  preset?: string;
  genome?: Genome;
  name?: string;
  clips?: 'auto' | ClipWindow[];
  /** Default: half-bar shift, a far offset, each stem muted, each reaction ablated. */
  variants?: Variant[];
  w?: number;
  h?: number;
  fps?: number;
  seed?: number;
  warm?: number;
  particleCap?: number;
  flameCap?: number;
}

async function counterfactual(opts: CfOpts) {
  const song = await loadSong(opts.song);
  let g = opts.genome;
  let id = opts.name ?? 'custom';
  let name = opts.name ?? 'custom';
  if (!g) {
    const seed = SEEDS.find((x) => x.origin === opts.preset);
    if (!seed) throw new Error('no seed ' + opts.preset);
    g = seed.genome;
    id = seed.origin;
    name = seed.name;
  }
  const bpb = Math.max(2, song.data.beatsPerBar || 4);
  const variants: Variant[] = opts.variants ?? [
    { kind: 'shift', beats: bpb / 2 },
    // Chaos floor: an inaudible 2 % level change; divergence here is amplification, not meaning.
    { kind: 'gain', factor: 0.98 },
    { kind: 'offset', seconds: Math.round(song.data.duration * 0.37) },
    ...(['drums', 'bass', 'vocals', 'other'] as StemId[]).map((stem) => ({ kind: 'mute' as const, stem })),
    ...g.reactions.slice(0, 6).map((_, reaction) => ({ kind: 'ablate' as const, reaction })),
  ];
  const w = opts.w ?? 320, h = opts.h ?? 180, fps = opts.fps ?? 30, seed = opts.seed ?? 1, warm = opts.warm ?? 3;
  const wins = !opts.clips || opts.clips === 'auto' ? autoWindows(song.data, song.hooks) : opts.clips;
  const e = engine();
  const results = [];
  for (const win of wins) {
    const tStart = performance.now();
    const all: (Variant | null)[] = [null, ...variants];
    const lanes = [];
    for (const v of all) {
      const gv = v?.kind === 'ablate' ? cloneGenome(g) : g;
      if (v?.kind === 'ablate') gv.reactions[v.reaction].gain = 0;
      status('compile ' + (v ? variantId(v) : 'base'));
      const progs = await programs(e, gv);
      const st = new Stage(e, { offscreen: true, particleCap: opts.particleCap ?? 262144, flameCap: opts.flameCap ?? 524288 });
      st.resize(w, h);
      const slot = st.makeSlot(gv, progs);
      st.slots = [slot];
      st.resetHistory();
      lanes.push({ v, st, slot, src: new MusicSource(song, v), rand: mulberry32(seed), px: new Uint8Array(w * h * 4), small: new Float32Array(80 * 45 * 3) });
    }
    const dt = 1 / fps;
    const t0 = Math.max(0, win.start - warm);
    const kStart = Math.round((win.start - t0) * fps);
    const kEnd = Math.round((win.end - t0) * fps);
    const n = kEnd - kStart;
    const lagF = Math.max(1, Math.round((bpb / 2) * (60 / (song.data.bpm || 120)) * fps)); // half a bar
    const hist: Float32Array[] = [];
    const motion = new Float32Array(n); // |base(t) - base(t - half bar)|
    const step = new Float32Array(n); // |base(t) - base(t - 1)|
    const div = variants.map(() => new Float32Array(n));
    const realRandom = Math.random;
    for (let k = 1; k <= kEnd; k++) {
      const t = t0 + k * dt;
      for (const L of lanes) {
        Math.random = L.rand;
        L.st.render(L.src.state(t, dt), dt, 'out');
      }
      if (k <= kStart) {
        if (k % 30 === 0) await tick();
        continue;
      }
      const i = k - kStart - 1;
      for (const L of lanes) {
        L.st.readPixels(L.px);
        shrink(L.px, w, h, 80, 45, L.small);
      }
      const base = lanes[0].small;
      for (let j = 1; j < lanes.length; j++) div[j - 1][i] = meanAbs(base, lanes[j].small);
      hist.push(base.slice());
      if (hist.length > lagF + 1) hist.shift();
      step[i] = hist.length >= 2 ? meanAbs(base, hist[hist.length - 2]) : 0;
      motion[i] = hist.length > lagF ? meanAbs(base, hist[0]) : NaN;
      if (i % 30 === 0) {
        status(`cf ${win.label} ${i}/${n}`);
        await tick();
      }
    }
    Math.random = realRandom;
    for (const L of lanes) L.st.disposeSlot(L.slot);
    let mSum = 0, mN = 0, sSum = 0;
    for (let i = 0; i < n; i++) {
      if (Number.isFinite(motion[i])) (mSum += motion[i]), mN++;
      sSum += step[i];
    }
    const M = mN ? mSum / mN : 0;
    const r4 = (x: number) => Math.round(x * 1e5) / 1e5;
    const out = {
      version: 1,
      preset: { id, name },
      song: { slug: song.slug, path: song.path, bpm: song.data.bpm, beatsPerBar: bpb },
      clip: { label: win.label, start: t0 + kStart * dt, end: t0 + kEnd * dt, warm },
      render: { w, h, fps, seed, ms: Math.round(performance.now() - tStart) },
      halfBarFrames: lagF,
      /** Mean |base(t) - base(t - half bar)| on an 80x45 RGB grid: the preset's own motion scale. */
      motion: r4(M),
      /** Mean |base(t) - base(t - 1 frame)|. */
      step: r4(sSum / Math.max(1, n)),
      reactions: g.reactions.slice(0, 6).map((r) => ({ src: r.src, target: `${r.g}${r.i}.${r.k}`, gain: r.gain })),
      variants: variants.map((v, j) => {
        const m = div[j].reduce((a, b) => a + b, 0) / Math.max(1, n);
        return { id: variantId(v), ...v, mean: r4(m), rel: r4(M > 1e-6 ? m / M : 0), series: Array.from(div[j], r4) };
      }),
      motionSeries: Array.from(motion, (x) => (Number.isFinite(x) ? r4(x) : null)),
    };
    const base = `${id}/${song.slug}__${win.label}`;
    await save(`cf/${base}.json`, JSON.stringify(out));
    results.push({ base, frames: n, ms: out.render.ms, motion: out.motion, variants: out.variants.map((v) => `${v.id} ${v.rel.toFixed(2)}`) });
  }
  return { preset: id, name, song: song.slug, clips: results };
}

// ------------------------------------------------------------------ sheets

async function fetchJson<T>(path: string): Promise<T | null> {
  const r = await fetch('/avq/file?p=' + encodeURIComponent(path));
  return r.ok ? ((await r.json()) as T) : null;
}

/** Draw the timeline sheet for clip `base` from the files under `out` (.testdata/avq) and save sheets/<base>.png. */
async function sheet(opts: { base: string; out: string }) {
  const { base, out } = opts;
  const header = await fetchJson<ClipHeader>(`${out}/clips/${base}.json`);
  if (!header) throw new Error('no clip ' + base);
  const bin = new Uint8Array(await (await fetch('/avq/file?p=' + encodeURIComponent(`${out}/clips/${base}.bin`))).arrayBuffer());
  const clip = parseClip(header, bin);
  const card = await fetchJson<ReportCard>(`${out}/cards/${base}.json`);
  const cf = await fetchJson<CfResult>(`${out}/cf/${base}.json`);
  const inst = await fetchJson<{ names: string[]; cols: (number | null)[][] }>(`${out}/clips/${base}.inst.json`);
  const frame = async (i: number) => {
    if (!header.framesDir) return null;
    const r = await fetch('/avq/file?p=' + encodeURIComponent(`${out}/${header.framesDir}/${String(i).padStart(6, '0')}.jpg`));
    return r.ok ? createImageBitmap(await r.blob()) : null;
  };
  const cv = await drawSheet({ clip, card, cf, inst, frame });
  const blob = await new Promise<Blob>((ok) => cv.toBlob((b) => ok(b!), 'image/png'));
  await save(`sheets/${base}.png`, blob);
  return { base, w: cv.width, h: cv.height, bytes: blob.size };
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

const api = { status: () => statusText, song: songInfo, render, counterfactual, sheet, seeds: () => SEEDS.map((s) => s.origin), fields: () => [...FIELDS], vis: () => [...VIS_FIELDS] };
(window as unknown as { avq: typeof api }).avq = api;
document.getElementById('log')!.textContent = 'avq ready';
