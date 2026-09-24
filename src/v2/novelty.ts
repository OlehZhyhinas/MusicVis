// Novelty search in phenotype space. Novelty = mean fingerprint distance to
// the k nearest neighbours in an ARCHIVE of every fingerprint ever shown or
// bred (disliked, hidden and culled members included, so the search doesn't
// wander back to looks it has already tried). Exploration mode weights
// novelty against fitness; the weight tapers as a member collects votes, so
// votes still decide in the end. Pure logic (no DOM) for the Node tests.

import { FP_VERSION, validFingerprint, zDistance, type FeatureNorm, type GroupWeights, EQUAL_WEIGHTS } from './fingerprint';

export const ARCHIVE_CAP = 3000;
export const NOVELTY_K = 10;

export interface ArchiveEntry {
  id: string;
  fp: number[];
  /** ms since epoch when it entered the archive. */
  t: number;
}

export interface ArchiveData {
  format: 'musicvis-v2-novelty-archive';
  fpVersion: number;
  entries: ArchiveEntry[];
}

export class NoveltyArchive {
  entries: ArchiveEntry[] = [];
  /** Bumped on every change (novelty caches key on it). */
  version = 0;
  cap: number;

  constructor(cap = ARCHIVE_CAP) {
    this.cap = cap;
  }

  get size(): number {
    return this.entries.length;
  }

  has(id: string): boolean {
    return this.entries.some((e) => e.id === id);
  }

  /** Add (or refresh) an id's fingerprint. The oldest entries go past the cap. */
  add(id: string, fp: number[], now = Date.now()): void {
    if (!validFingerprint(fp)) return;
    const i = this.entries.findIndex((e) => e.id === id);
    if (i >= 0) this.entries.splice(i, 1);
    this.entries.push({ id, fp, t: now });
    if (this.entries.length > this.cap) this.entries.splice(0, this.entries.length - this.cap);
    this.version++;
  }

  toJSON(): ArchiveData {
    return { format: 'musicvis-v2-novelty-archive', fpVersion: FP_VERSION, entries: this.entries };
  }

  /** Entries of another fingerprint version are dropped (their members are re-fingerprinted and re-added). */
  static fromJSON(data: unknown, cap = ARCHIVE_CAP): NoveltyArchive {
    const a = new NoveltyArchive(cap);
    const d = data as Partial<ArchiveData> | undefined;
    if (!d || d.format !== 'musicvis-v2-novelty-archive' || d.fpVersion !== FP_VERSION || !Array.isArray(d.entries)) return a;
    for (const e of d.entries) {
      if (e && typeof e.id === 'string' && validFingerprint(e.fp)) a.entries.push({ id: e.id, fp: e.fp, t: typeof e.t === 'number' ? e.t : 0 });
    }
    if (a.entries.length > cap) a.entries.splice(0, a.entries.length - cap);
    return a;
  }
}

/** Mean distance to the k nearest z-scored fingerprints (excluding `selfId`). */
export function knnNovelty(z: Float32Array, others: { id: string; z: Float32Array }[], k = NOVELTY_K, selfId?: string, w: GroupWeights = EQUAL_WEIGHTS): number {
  const d: number[] = [];
  for (const o of others) if (o.id !== selfId) d.push(zDistance(z, o.z, w));
  if (!d.length) return 0;
  d.sort((a, b) => a - b);
  const n = Math.min(k, d.length);
  let s = 0;
  for (let i = 0; i < n; i++) s += d[i];
  return s / n;
}

/**
 * Novelty of every member at once against the archive, plus a relative
 * value: the fraction of archived looks that are themselves less novel
 * (a percentile, 0..1), which is what exploration weights. 0.5 is "as novel
 * as a typical archived look", 0.9 "more novel than 90% of them".
 */
export function noveltyTable(
  members: { id: string; fp?: number[] }[], archive: NoveltyArchive, norm: FeatureNorm, w: GroupWeights = EQUAL_WEIGHTS, k = NOVELTY_K,
): { table: Map<string, { nov: number; rel: number }>; typical: number[] } {
  const az = archive.entries.map((e) => ({ id: e.id, z: norm.z(e.fp) }));
  // The archive's typical novelty (a sample, for speed), the scale for `rel`.
  const step = Math.max(1, Math.floor(az.length / 200));
  const typ: number[] = [];
  for (let i = 0; i < az.length; i += step) typ.push(knnNovelty(az[i].z, az, k, az[i].id, w));
  typ.sort((a, b) => a - b);
  const out = new Map<string, { nov: number; rel: number }>();
  for (const m of members) {
    if (!m.fp || !validFingerprint(m.fp)) continue;
    const nov = knnNovelty(norm.z(m.fp), az, k, m.id, w);
    out.set(m.id, { nov, rel: relNovelty(nov, typ) });
  }
  return { table: out, typical: typ };
}

/** Percentile of `nov` among the archive's own (sorted) novelties, interpolated; 0.5 without an archive. */
export function relNovelty(nov: number, sortedTypical: number[]): number {
  const t = sortedTypical;
  if (t.length < 2) return 0.5;
  if (nov <= t[0]) return 0;
  if (nov >= t[t.length - 1]) return 1;
  let lo = 0, hi = t.length - 1;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (t[mid] <= nov) lo = mid;
    else hi = mid;
  }
  const f = t[hi] > t[lo] ? (nov - t[lo]) / (t[hi] - t[lo]) : 0;
  return (lo + f) / (t.length - 1);
}

// ------------------------------------------------------ exploration mode

export type ExploreMode = 'off' | 'gentle' | 'explore' | 'wild';
export const EXPLORE_MODES: ExploreMode[] = ['off', 'gentle', 'explore', 'wild'];
export const EXPLORE_LABEL: Record<ExploreMode, string> = { off: 'Off', gentle: 'Gentle', explore: 'Explore', wild: 'Wild' };

/** Novelty weight per mode, against fitness in 0..1. */
export const EXPLORE_WEIGHT: Record<ExploreMode, number> = { off: 0, gentle: 0.08, explore: 0.3, wild: 0.6 };

/** Children less novel than this (relative) are turned away in breeding, per mode. */
export const EXPLORE_ACCEPT: Record<ExploreMode, number> = { off: 0, gentle: 0, explore: 0.18, wild: 0.3 };

/** Votes (likes + dislikes, implicit signals at 0.3) after which a member's novelty weight has halved. */
export const VOTE_HALF = 3;

export function voteCount(m: { likes: number; dislikes: number; weakLikes: number; softDislikes: number }): number {
  return m.likes + m.dislikes + 0.3 * (m.weakLikes + m.softDislikes);
}

/** The novelty weight for a member: the mode's weight, tapering as votes come in. */
export function noveltyWeight(mode: ExploreMode, votes: number): number {
  return EXPLORE_WEIGHT[mode] / (1 + votes / VOTE_HALF);
}

/** score = fitness + w * novelty, w from the mode and tapering with votes. */
export function exploreScore(fit: number, rel: number, mode: ExploreMode, votes: number): number {
  return fit + noveltyWeight(mode, votes) * rel;
}

export function parseExploreMode(v: unknown): ExploreMode {
  return EXPLORE_MODES.includes(v as ExploreMode) ? (v as ExploreMode) : 'gentle';
}
