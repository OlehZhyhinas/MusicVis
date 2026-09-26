// Shared helpers for the live-parity tools: decode audio files to PCM with ffmpeg (never played,
// so nothing is audible), cache the offline analysis on disk, and drive the live analysis path
// in Node exactly as src/audio/LiveInput.ts does in the browser (512-sample blocks from the
// capture worklet into RealtimeAnalyzer.process, RealtimeSampler.sample at the render clock
// with the stream time extrapolated like LiveInput.streamTimeNow).

import { execFileSync } from 'node:child_process';
import { existsSync, mkdirSync, readdirSync, readFileSync, statSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { deserialize, serialize } from 'node:v8';
import { analyzePcm } from '../../src/analysis/analyzePcm';
import { RealtimeAnalyzer } from '../../src/analysis/RealtimeAnalyzer';
import { RealtimeSampler } from '../../src/analysis/RealtimeSampler';
import { DEFAULT_VISUAL_LAG, LiveLookahead } from '../../src/analysis/LiveLookahead';
import type { AnalysisResult, LiveAudioFrame, MusicState } from '../../src/types';

export const TEST_DIR = '/Users/oleh/Downloads/YoutubeToMp3';
export const OUT = join(import.meta.dirname, '../../.testdata/live');
export const SR = 44100;

export interface Pcm {
  sr: number;
  left: Float32Array;
  right: Float32Array;
}

export function ffmpegPath(): string {
  for (const p of ['/opt/homebrew/bin/ffmpeg', '/usr/local/bin/ffmpeg']) if (existsSync(p)) return p;
  return execFileSync('which', ['ffmpeg']).toString().trim();
}

export function slugOf(path: string): string {
  const base = path.split('/').pop()!.replace(/\.[^.]+$/, '');
  return base.toLowerCase().replace(/\(.*?\)/g, '').replace(/[^a-z0-9]+/g, '-').replace(/^-+|-+$/g, '').slice(0, 40) || 'song';
}

/** The test songs (one per distinct title: "(1)" duplicates are skipped). */
export function testSongs(): string[] {
  return readdirSync(TEST_DIR)
    .filter((f) => /\.(mp3|m4a|wav|flac|ogg)$/i.test(f) && !/\(\d+\)\.[a-z0-9]+$/i.test(f))
    .sort()
    .map((f) => join(TEST_DIR, f));
}

/** Decode to 44.1 kHz stereo float PCM (ffmpeg to a pipe; nothing is played). */
export function decode(path: string): Pcm {
  mkdirSync(OUT, { recursive: true });
  const cache = join(OUT, 'pcm', slugOf(path) + '.f32');
  let buf: Buffer;
  if (existsSync(cache) && statSync(cache).mtimeMs > statSync(path).mtimeMs) buf = readFileSync(cache);
  else {
    buf = execFileSync(ffmpegPath(), ['-v', 'error', '-i', path, '-f', 'f32le', '-ac', '2', '-ar', String(SR), '-'], { maxBuffer: 1 << 30 });
    mkdirSync(join(OUT, 'pcm'), { recursive: true });
    writeFileSync(cache, buf);
  }
  const all = new Float32Array(buf.buffer, buf.byteOffset, buf.byteLength / 4);
  const n = all.length >> 1;
  const left = new Float32Array(n);
  const right = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    left[i] = all[2 * i];
    right[i] = all[2 * i + 1];
  }
  return { sr: SR, left, right };
}

/** Bump when the offline analysis gains outputs. */
const OFFLINE_VER = 'o1';

/** Offline analysis of a PCM buffer, cached on disk under `key`. */
export function offlineCached(key: string, pcm: Pcm): AnalysisResult {
  const p = join(OUT, 'offline', `${OFFLINE_VER}-${key}.v8`);
  if (existsSync(p)) return deserialize(readFileSync(p)) as AnalysisResult;
  const r = analyzePcm(pcm.left, pcm.right, pcm.sr);
  mkdirSync(join(OUT, 'offline'), { recursive: true });
  writeFileSync(p, serialize(r));
  return r;
}

export function mono(pcm: Pcm): Float32Array {
  const out = new Float32Array(pcm.left.length);
  for (let i = 0; i < out.length; i++) out[i] = 0.5 * (pcm.left[i] + pcm.right[i]);
  return out;
}

/** Capture worklet batch (src/audio/LiveInput.ts processorOptions.batch). */
export const BLOCK = 512;

/**
 * The live path over a PCM buffer, advanced by wall-clock time. Blocks arrive as the capture
 * worklet posts them (each once its last sample has been captured); sample(t) renders a frame
 * at wall time t with the stream time extrapolated up to 50 ms past the last block, like
 * LiveInput.streamTimeNow. With a visual lag the notes are drawn `lag` behind (their look-ahead);
 * everything else describes the sound at t.
 */
export class LivePath {
  readonly analyzer: RealtimeAnalyzer;
  readonly sampler: RealtimeSampler;
  private pos = 0;
  private lastBlockEnd = 0;
  /** Milliseconds spent in RealtimeAnalyzer.process (and the look-ahead notes). */
  processMs = 0;
  private pcm: Pcm;
  /** The delay line with the look-ahead note tracker (LiveInput's visual lag), or null for lag 0. */
  readonly lookahead: LiveLookahead | null;
  /** Seconds the visuals trail the sound (LiveInput visualLag; the app's default is 0.1). */
  readonly lag: number;
  constructor(pcm: Pcm, lag = DEFAULT_VISUAL_LAG) {
    this.pcm = pcm;
    this.lag = lag;
    this.analyzer = new RealtimeAnalyzer(pcm.sr);
    this.sampler = new RealtimeSampler(this.analyzer);
    this.lookahead = lag > 0 ? new LiveLookahead(this.analyzer, lag) : null;
    if (this.lookahead) this.sampler.noteSource = this.lookahead.sampleNotes;
  }
  /** Feed every block that has fully arrived by wall time t. */
  feed(t: number): void {
    const { left, right, sr } = this.pcm;
    const end = Math.min(left.length, Math.floor(t * sr));
    while (this.pos + BLOCK <= end) {
      const t0 = performance.now();
      const l = left.subarray(this.pos, this.pos + BLOCK), r = right.subarray(this.pos, this.pos + BLOCK);
      if (this.lookahead) this.lookahead.process(l, r);
      else this.analyzer.process(l, r);
      this.processMs += performance.now() - t0;
      this.pos += BLOCK;
      this.lastBlockEnd = this.pos / sr;
    }
  }
  /** LiveInput.streamTimeNow at wall time t. */
  streamTime(t: number): number {
    const a = this.analyzer.streamTime;
    const ahead = Math.min(0.05, Math.max(0, t - this.lastBlockEnd));
    return a + ahead;
  }
  sample(t: number, dt: number, live: LiveAudioFrame): MusicState {
    this.feed(t);
    return this.sampler.sample(this.streamTime(t), dt, true, live);
  }
}
