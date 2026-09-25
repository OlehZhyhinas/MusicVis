// Lyrics lookup on LRCLIB (https://lrclib.net: free, no key, time-synced LRC lyrics). An exact
// match first (/api/get), then a search (/api/search) scored by title, artist and duration. Results
// are cached by artist + title (IndexedDB in the browser); a failed request (offline, server busy)
// is not cached, so the next play tries again.
//
// The duration matters: the same song exists in several versions (album, radio edit, music video
// with a longer intro) and synced lyrics are timed for one of them. The lookup sends the song's
// duration (the decoded audio's, once known) and prefers records within DURATION_MATCH of it; a
// cached result whose record is more than RECHECK_AFTER away from the file is looked up again.

import type { LyricResult, TrackMeta } from './types';

export const LRCLIB_URL = 'https://lrclib.net';
/** LRCLIB asks clients to identify themselves; browsers may not set User-Agent, so this header. */
export const LRCLIB_CLIENT = 'MusicVis (https://github.com/OlehZhyhinas/MusicVis)';
/** Not-found results are trusted this long, then asked again (lyrics get added over time). */
export const MISSING_TTL_MS = 7 * 24 * 3600 * 1000;
/** A record this close to the file's duration (seconds) is the same version of the song. */
export const DURATION_MATCH = 2;
/** A cached record further than this from the file's duration (seconds) is looked up again. */
export const RECHECK_AFTER = 3;

export interface CacheEntry {
  result: LyricResult;
  at: number;
  /** The duration the lookup asked for (seconds; absent: asked without one). */
  forDuration?: number;
}
export interface LyricsCache {
  get(key: string): Promise<CacheEntry | undefined>;
  set(key: string, entry: CacheEntry): Promise<void>;
}

/** One LRCLIB record (only the fields used). */
interface LrclibRecord {
  trackName?: string;
  artistName?: string;
  albumName?: string;
  duration?: number;
  instrumental?: boolean;
  plainLyrics?: string | null;
  syncedLyrics?: string | null;
}

/** Lowercase, accents, punctuation and "the" folded, for matching names. */
export function norm(s: string | undefined): string {
  return (s ?? '')
    .normalize('NFKD')
    .replace(/[\u0300-\u036f]/g, '')
    .toLowerCase()
    .replace(/&/g, ' and ')
    .replace(/[^\p{L}\p{N}]+/gu, ' ')
    .replace(/^the /, '')
    .trim();
}

export function cacheKey(meta: TrackMeta): string {
  return `${norm(meta.artist)}|${norm(meta.title)}`;
}

function toResult(r: LrclibRecord, meta: TrackMeta): LyricResult {
  const source = [r.artistName ?? meta.artist, r.trackName ?? meta.title].filter(Boolean).join(' - ');
  const dur = r.duration && r.duration > 0 ? { duration: Math.round(r.duration * 10) / 10 } : {};
  if (r.instrumental) return { status: 'instrumental', source, ...dur };
  const synced = r.syncedLyrics?.trim();
  const plain = r.plainLyrics?.trim();
  if (synced) return { status: 'synced', synced, ...(plain ? { plain } : {}), source, ...dur };
  if (plain) return { status: 'plain', plain, source, ...dur };
  return { status: 'missing' };
}

/**
 * Whether a cached result still fits the file: its record's duration is within RECHECK_AFTER of
 * the file's, or it was already looked up for (about) this duration and that was the best there
 * was. Without a known file duration any result fits.
 */
export function fitsDuration(entry: CacheEntry, duration: number | undefined): boolean {
  if (!duration || !(duration > 0)) return true;
  const r = entry.result;
  if (r.status === 'missing' || r.status === 'offline') return true;
  if (entry.forDuration !== undefined && Math.abs(entry.forDuration - duration) <= RECHECK_AFTER) return true;
  return r.duration !== undefined && Math.abs(r.duration - duration) <= RECHECK_AFTER;
}

/** How well a search record fits (negative: reject). */
export function score(r: LrclibRecord, meta: TrackMeta): number {
  const t = norm(meta.title);
  const rt = norm(r.trackName);
  if (!t || !rt) return -1;
  let s = 0;
  if (rt === t) s += 4;
  else if (rt.startsWith(t) || t.startsWith(rt)) s += 2;
  else if (rt.includes(t) || t.includes(rt)) s += 1;
  else return -1;
  const a = norm(meta.artist);
  const ra = norm(r.artistName);
  if (a && ra) {
    if (ra === a) s += 3;
    else if (ra.includes(a) || a.includes(ra)) s += 1.5;
    else s -= 3;
  }
  if (meta.duration && r.duration) {
    // The version matters for synced lyrics: a record of the same length is timed for this file.
    const d = Math.abs(meta.duration - r.duration);
    if (d <= DURATION_MATCH) s += 3;
    else if (d <= 5) s += 1;
    else if (d > 20) s -= 3;
    else if (d > 10) s -= 1;
  }
  if (r.syncedLyrics) s += 1;
  else if (!r.plainLyrics && !r.instrumental) s -= 5;
  return s;
}

export interface LookupOptions {
  fetch?: typeof fetch;
  cache?: LyricsCache;
  signal?: AbortSignal;
  now?: number;
  /** Waits before retrying a busy server or a dropped request (ms); default 1.5 s then 4 s. */
  retryDelays?: number[];
}

class Transient extends Error {}

const sleep = (ms: number, signal?: AbortSignal) =>
  new Promise<void>((resolve, reject) => {
    const t = setTimeout(resolve, ms);
    signal?.addEventListener('abort', () => {
      clearTimeout(t);
      reject(new DOMException('aborted', 'AbortError'));
    });
  });

/** A request, retried after the given delays while the server is busy (503, 429) or unreachable. */
async function getJson(f: typeof fetch, path: string, q: Record<string, string | number | undefined>, signal: AbortSignal | undefined, delays: number[]): Promise<unknown | null> {
  for (let i = 0; ; i++) {
    try {
      return await getJsonOnce(f, path, q, signal);
    } catch (e) {
      if (!(e instanceof Transient) || i >= delays.length) throw e;
      await sleep(delays[i], signal);
    }
  }
}

async function getJsonOnce(f: typeof fetch, path: string, q: Record<string, string | number | undefined>, signal?: AbortSignal): Promise<unknown | null> {
  const qs = Object.entries(q)
    .filter(([, v]) => v !== undefined && v !== '')
    .map(([k, v]) => `${k}=${encodeURIComponent(String(v))}`)
    .join('&');
  let res: Response;
  try {
    res = await f(`${LRCLIB_URL}${path}?${qs}`, { headers: { 'Lrclib-Client': LRCLIB_CLIENT }, signal });
  } catch (e) {
    if ((e as Error)?.name === 'AbortError') throw e;
    throw new Transient('network');
  }
  if (res.status === 404) return null;
  if (res.status === 503 || res.status === 429 || res.status >= 500) throw new Transient(`http ${res.status}`);
  // Other client errors (a 400 for a malformed query): nothing to find this way.
  if (!res.ok) return null;
  return res.json();
}

/**
 * Lyrics for a song: synced (LRC), plain, instrumental, missing, or offline (the lookup failed and
 * nothing is cached). Never throws, except when aborted.
 */
export async function lookupLyrics(meta: TrackMeta, opts: LookupOptions = {}): Promise<LyricResult> {
  const f = opts.fetch ?? fetch.bind(globalThis);
  const now = opts.now ?? Date.now();
  if (!meta.title) return { status: 'missing' };
  const key = cacheKey(meta);
  const dur = meta.duration && meta.duration > 0 ? Math.round(meta.duration) : undefined;
  try {
    const hit = await opts.cache?.get(key);
    if (hit && (hit.result.status !== 'missing' || now - hit.at < MISSING_TTL_MS) && fitsDuration(hit, meta.duration)) return hit.result;
  } catch {
    // An unreadable cache is a miss.
  }
  let failed = false;
  let result: LyricResult = { status: 'missing' };
  const tryStep = async (step: () => Promise<LyricResult | null>): Promise<LyricResult | null> => {
    try {
      return await step();
    } catch (e) {
      if ((e as Error)?.name === 'AbortError') throw e;
      failed = true;
      return null;
    }
  };
  const delays = opts.retryDelays ?? [1500, 4000];

  // 1. Exact match (needs an artist; with a duration LRCLIB only returns a record within 2 s).
  let found: LyricResult | null = null;
  if (meta.artist) {
    found = await tryStep(async () => {
      const r = (await getJson(f, '/api/get', { artist_name: meta.artist, track_name: meta.title, album_name: meta.album, duration: dur }, opts.signal, delays)) as LrclibRecord | null;
      if (!r) return null;
      const out = toResult(r, meta);
      return out.status === 'missing' ? null : out;
    });
  }
  // 2. Search: by fields, then free text; the best-scoring record wins. Also asked when the exact
  // match has no timing, for another record of the song that has.
  const exact = found;
  if (!found || found.status === 'plain') {
    found = null;
    const queries: Record<string, string | undefined>[] = meta.artist
      ? [{ track_name: meta.title, artist_name: meta.artist }, { q: `${meta.artist} ${meta.title}` }]
      : [{ q: meta.title }];
    for (const q of queries) {
      found = await tryStep(async () => {
        const list = (await getJson(f, '/api/search', q, opts.signal, delays)) as LrclibRecord[] | null;
        if (!Array.isArray(list) || !list.length) return null;
        let best: LrclibRecord | null = null;
        let bestScore = -Infinity;
        const gap = (r: LrclibRecord) => (meta.duration && r.duration ? Math.abs(meta.duration - r.duration) : 0);
        for (const r of list) {
          const s = score(r, meta);
          // Equal scores: the record closest in length.
          if (s > bestScore || (s === bestScore && best && gap(r) < gap(best))) {
            bestScore = s;
            best = r;
          }
        }
        // Without an artist only a close duration match is trusted.
        const need = meta.artist ? 3 : 6;
        if (!best || bestScore < need) return null;
        const out = toResult(best, meta);
        if (out.status === 'missing' || (exact && out.status !== 'synced')) return null;
        return out;
      });
      if (found) break;
    }
    found ??= exact;
  }
  if (found) result = found;
  else if (failed) return { status: 'offline' };
  try {
    await opts.cache?.set(key, { result, at: now, ...(dur ? { forDuration: dur } : {}) });
  } catch {
    // Not cached: looked up again next time.
  }
  return result;
}

// ------------------------------------------------------------------ caches

export function memoryCache(): LyricsCache & { map: Map<string, CacheEntry> } {
  const map = new Map<string, CacheEntry>();
  return {
    map,
    get: async (k) => map.get(k),
    set: async (k, v) => void map.set(k, v),
  };
}

/** IndexedDB cache (falls back to memory when IndexedDB is unavailable, e.g. private mode). */
export function idbCache(dbName = 'musicvis-lyrics'): LyricsCache {
  const mem = memoryCache();
  let dbp: Promise<IDBDatabase | null> | null = null;
  const open = () =>
    (dbp ??= new Promise((resolve) => {
      try {
        const req = indexedDB.open(dbName, 1);
        req.onupgradeneeded = () => req.result.createObjectStore('lyrics');
        req.onsuccess = () => resolve(req.result);
        req.onerror = () => resolve(null);
        req.onblocked = () => resolve(null);
      } catch {
        resolve(null);
      }
    }));
  const run = <T>(mode: IDBTransactionMode, fn: (s: IDBObjectStore) => IDBRequest): Promise<T | undefined> =>
    open().then(
      (db) =>
        new Promise<T | undefined>((resolve) => {
          if (!db) return resolve(undefined);
          try {
            const req = fn(db.transaction('lyrics', mode).objectStore('lyrics'));
            req.onsuccess = () => resolve(req.result as T);
            req.onerror = () => resolve(undefined);
          } catch {
            resolve(undefined);
          }
        }),
    );
  return {
    async get(k) {
      return mem.map.get(k) ?? (await run<CacheEntry>('readonly', (s) => s.get(k)));
    },
    async set(k, v) {
      mem.map.set(k, v);
      await run('readwrite', (s) => s.put(v, k));
    },
  };
}
