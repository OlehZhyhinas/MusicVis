// Per-track lyrics for the playlist: reads each added file's tags, looks the song up on LRCLIB one
// track at a time (automatically, in the background), and turns what it finds into a timed
// LyricTrack once the song's analysis (duration, vocal presence) is known. Lookups that failed
// because the network was down are retried when the browser comes back online. Once a song is
// decoded its real duration is known (files rarely carry it in their tags); a result timed for a
// different version of the song is then looked up again with it.

import type { AnalysisResult } from '../types';
import type { LyricResult, LyricStatus, LyricTrack, TrackMeta } from './types';
import { readFileTags } from './tags';
import { mergeMeta } from './filename';
import { idbCache, lookupLyrics, RECHECK_AFTER, type LyricsCache } from './lrclib';
import { parseLrc, spreadPlain, vocalRegions } from './lrc';
import { alignLines, shiftTrack } from './align';

export interface TrackLyrics {
  status: LyricStatus;
  meta?: TrackMeta;
  result?: LyricResult;
}

interface Item extends TrackLyrics {
  file: Blob & { name?: string };
  /** The decoded audio's duration (seconds), once known. */
  decoded?: number;
  built?: { result: AnalysisResult; track: LyricTrack | null };
}

export class LyricsLibrary {
  private items = new Map<string, Item>();
  private queue: string[] = [];
  private busy = false;
  private cache: LyricsCache;
  private fetchFn: typeof fetch | undefined;
  /** Called when a track's lyric status changes. */
  onChange: ((id: string) => void) | null = null;

  constructor(opts: { cache?: LyricsCache; fetch?: typeof fetch } = {}) {
    this.cache = opts.cache ?? idbCache();
    this.fetchFn = opts.fetch;
    if (typeof window !== 'undefined') window.addEventListener('online', () => this.retryOffline());
  }

  add(id: string, file: Blob & { name?: string }): void {
    if (this.items.has(id)) return;
    this.items.set(id, { status: 'pending', file });
    this.queue.push(id);
    void this.pump();
  }

  remove(id: string): void {
    this.items.delete(id);
    this.queue = this.queue.filter((q) => q !== id);
  }

  clear(): void {
    this.items.clear();
    this.queue = [];
  }

  get(id: string): TrackLyrics | undefined {
    return this.items.get(id);
  }

  status(id: string): LyricStatus | undefined {
    return this.items.get(id)?.status;
  }

  /** Looks up again every track whose lookup failed for want of a network. */
  retryOffline(): void {
    for (const [id, it] of this.items) {
      if (it.status === 'offline' && !this.queue.includes(id)) {
        it.status = 'pending';
        this.queue.push(id);
        this.onChange?.(id);
      }
    }
    void this.pump();
  }

  /**
   * The decoded audio's duration: used by the lookups from now on, and a found result whose record
   * is more than RECHECK_AFTER seconds longer or shorter (another version of the song, timed
   * differently) is looked up again, first in the queue.
   */
  setDuration(id: string, seconds: number): void {
    const it = this.items.get(id);
    if (!it || !(seconds > 0) || it.decoded === seconds) return;
    it.decoded = seconds;
    if (it.meta) it.meta = { ...it.meta, duration: seconds };
    const r = it.result;
    const found = r && (r.status === 'synced' || r.status === 'plain' || r.status === 'instrumental');
    if (!found || this.queue.includes(id)) return;
    if (r.duration !== undefined && Math.abs(r.duration - seconds) <= RECHECK_AFTER) return;
    this.queue.unshift(id);
    void this.pump();
  }

  /** Moves a track to the front of the queue (the one about to play). */
  prioritize(id: string): void {
    if (!this.queue.includes(id)) return;
    this.queue = [id, ...this.queue.filter((q) => q !== id)];
  }

  /**
   * The timed lyrics of an analysed track (null: none, or not looked up yet). Synced lyrics are
   * matched to the song's sung onsets (moved when the match is confident: the LRC was timed for
   * another release); plain lyrics are spread over the song's vocal stretches.
   */
  lyricTrack(id: string, result: AnalysisResult): LyricTrack | null {
    const it = this.items.get(id);
    if (!it?.result) return null;
    if (it.built?.result === result) return it.built.track;
    let track: LyricTrack | null = null;
    const r = it.result;
    if (r.status === 'synced' && r.synced) {
      track = parseLrc(r.synced, result.duration);
      const onsets = result.stemOnsets?.vocals;
      if (onsets && track.lines.length) {
        const a = alignLines(track.lines, { onsets, frameRate: result.frameRate, duration: result.duration });
        const moved = a.applied ? shiftTrack(track, a.offset, a.scale, result.duration) : track;
        track = { ...moved, align: { offset: a.offset, scale: a.scale, confidence: a.confidence, applied: a.applied } };
      }
    } else if (r.status === 'plain' && r.plain) {
      const voc = result.stemPresence?.vocals;
      track = spreadPlain(r.plain, result.duration, voc ? vocalRegions(voc, result.frameRate) : []);
    }
    if (track && !track.lines.length) track = null;
    it.built = { result, track };
    return track;
  }

  private async pump(): Promise<void> {
    if (this.busy) return;
    this.busy = true;
    try {
      while (this.queue.length) {
        const id = this.queue.shift()!;
        const it = this.items.get(id);
        if (!it) continue;
        try {
          if (!it.meta) {
            let tags: TrackMeta = {};
            try {
              tags = await readFileTags(it.file);
            } catch {
              // Unreadable: the file name has to do.
            }
            it.meta = mergeMeta(tags, it.file.name ?? '');
            if (it.decoded) it.meta.duration = it.decoded;
          }
          const res = await lookupLyrics(it.meta, { cache: this.cache, fetch: this.fetchFn });
          if (!this.items.has(id)) continue;
          it.result = res;
          it.status = res.status;
          it.built = undefined;
        } catch {
          it.status = 'offline';
        }
        this.onChange?.(id);
      }
    } finally {
      this.busy = false;
    }
  }
}

/** Short playlist label and tooltip for a lyric status. */
export function lyricStatusLabel(s: LyricStatus | undefined): { label: string; title: string } | null {
  switch (s) {
    case 'pending': return { label: 'lyrics…', title: 'Looking up lyrics' };
    case 'synced': return { label: 'synced', title: 'Time-synced lyrics found' };
    case 'plain': return { label: 'lyrics', title: 'Lyrics found (no timing: spread over the vocals)' };
    case 'instrumental': return { label: 'instr.', title: 'Instrumental: no lyrics' };
    case 'missing': return { label: 'no lyrics', title: 'No lyrics found' };
    case 'offline': return { label: 'offline', title: 'Lyrics lookup failed (offline?); retried when back online' };
    default: return null;
  }
}
