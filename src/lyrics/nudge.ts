// Manual lyric timing: when the automatic alignment is off (or not confident), [ and ] move the
// lyrics a quarter second earlier or later. The nudge is remembered per song (artist, title and
// length, so another version of the song starts clean) in localStorage.

import type { TrackMeta } from './types';
import { norm } from './lrclib';

export const NUDGE_STEP = 0.25;
/** Nudges beyond this are clamped (the automatic alignment covers larger shifts). */
export const NUDGE_MAX = 30;
const STORE_KEY = 'musicvis:lyricNudge';

/** Where the nudges live (localStorage in the browser; a Map in tests). */
export interface NudgeStore {
  getItem(key: string): string | null;
  setItem(key: string, value: string): void;
}

/** The song's nudge key: artist | title | length in whole seconds (null: no title). */
export function nudgeKey(meta: TrackMeta | undefined, duration: number): string | null {
  if (!meta?.title || !(duration > 0)) return null;
  return `${norm(meta.artist)}|${norm(meta.title)}|${Math.round(duration)}`;
}

export class LyricNudges {
  private map: Record<string, number> = {};
  private store: NudgeStore | null;

  constructor(store?: NudgeStore | null) {
    this.store = store !== undefined ? store : typeof localStorage !== 'undefined' ? localStorage : null;
    try {
      const raw = this.store?.getItem(STORE_KEY);
      const m = raw ? JSON.parse(raw) : null;
      if (m && typeof m === 'object') for (const [k, v] of Object.entries(m)) if (typeof v === 'number' && Number.isFinite(v)) this.map[k] = v;
    } catch {
      // Unreadable: start empty.
    }
  }

  /** The song's nudge in seconds (positive: lyrics later); 0 when none. */
  get(key: string | null): number {
    return key ? (this.map[key] ?? 0) : 0;
  }

  /** Sets (0 clears) and saves; returns the value kept (rounded to 0.01 s, clamped). */
  set(key: string | null, seconds: number): number {
    const v = Math.max(-NUDGE_MAX, Math.min(NUDGE_MAX, Math.round(seconds * 100) / 100));
    if (!key) return v;
    if (v === 0) delete this.map[key];
    else this.map[key] = v;
    try {
      this.store?.setItem(STORE_KEY, JSON.stringify(this.map));
    } catch {
      // Quota or private mode: kept for this session only.
    }
    return v;
  }
}

/** "+0.25 s" / "-1.50 s" / "0.00 s". */
export function formatNudge(v: number): string {
  return `${v > 0 ? '+' : v < 0 ? '−' : ''}${Math.abs(v).toFixed(2)} s`;
}
