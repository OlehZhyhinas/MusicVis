import type { AnalysisResult } from '../types';

export type TrackStatus = 'queued' | 'analyzing' | 'ready' | 'error';
export type RepeatMode = 'off' | 'all' | 'one';

export interface Track {
  readonly id: string;
  readonly file: File;
  readonly title: string;
  duration: number | null;
  status: TrackStatus;
  progress: number; // 0..1, meaningful while status === 'analyzing'
  error?: string;
}

interface Loaded {
  buffer: AudioBuffer;
  result: AnalysisResult;
}

function titleFromFilename(name: string): string {
  const withoutExt = name.replace(/\.[^./\\]+$/, '');
  return withoutExt || name;
}

function makeId(): string {
  if ('randomUUID' in crypto) return crypto.randomUUID();
  return `t${Date.now()}-${Math.random().toString(36).slice(2)}`;
}

function shuffleArray<T>(arr: T[]): T[] {
  const out = arr.slice();
  for (let i = out.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [out[i], out[j]] = [out[j], out[i]];
  }
  return out;
}

/**
 * Owns the track list, play order (shuffle/repeat), and lazy decode +
 * analysis of each track, with in-memory caching so switching back to an
 * already-analyzed track is instant. Only one track is decoded+analyzed at
 * a time; decoded AudioBuffers for tracks that are neither current nor
 * "next up" are dropped to save memory (AnalysisResults stay cached, they
 * are small).
 */
export class Playlist {
  private tracks: Track[] = [];
  /** Playback order, as a list of track ids. Reshuffled when shuffle toggles on. */
  private order: string[] = [];
  private posInOrder = -1;
  private _shuffle = false;
  private _repeat: RepeatMode = 'off';

  private results = new Map<string, AnalysisResult>();
  private buffers = new Map<string, AudioBuffer>();
  private analyzingId: string | null = null;
  private queuedAnalysis: Promise<Loaded> | null = null;

  onChange: (() => void) | null = null;

  get shuffle(): boolean {
    return this._shuffle;
  }

  get repeat(): RepeatMode {
    return this._repeat;
  }

  get all(): readonly Track[] {
    return this.tracks;
  }

  get isEmpty(): boolean {
    return this.tracks.length === 0;
  }

  get currentTrack(): Track | undefined {
    const id = this.order[this.posInOrder];
    return id ? this.byId(id) : undefined;
  }

  private byId(id: string): Track | undefined {
    return this.tracks.find((t) => t.id === id);
  }

  private notify(): void {
    this.onChange?.();
  }

  addFiles(files: File[]): Track[] {
    const added: Track[] = files.map((file) => ({
      id: makeId(),
      file,
      title: titleFromFilename(file.name),
      duration: null,
      status: 'queued',
      progress: 0,
    }));
    this.tracks.push(...added);
    for (const t of added) this.order.push(t.id);
    if (this._shuffle) {
      // Keep already-played history stable-ish; simplest correct behavior:
      // reshuffle the not-yet-played tail including the new tracks.
      this.reshuffleTail();
    }
    if (this.posInOrder === -1 && this.order.length > 0) {
      this.posInOrder = 0;
    }
    this.notify();
    return added;
  }

  removeTrack(id: string): void {
    const currentId = this.order[this.posInOrder];
    this.tracks = this.tracks.filter((t) => t.id !== id);
    this.order = this.order.filter((tid) => tid !== id);
    this.results.delete(id);
    this.buffers.delete(id);
    if (this.analyzingId === id) this.analyzingId = null;
    if (currentId === id) {
      this.posInOrder = Math.min(this.posInOrder, this.order.length - 1);
    } else {
      this.posInOrder = this.order.indexOf(currentId);
    }
    this.notify();
  }

  clear(): void {
    this.tracks = [];
    this.order = [];
    this.posInOrder = -1;
    this.results.clear();
    this.buffers.clear();
    this.analyzingId = null;
    this.notify();
  }

  selectTrack(id: string): boolean {
    const idx = this.order.indexOf(id);
    if (idx === -1) return false;
    this.posInOrder = idx;
    this.notify();
    return true;
  }

  setShuffle(on: boolean): void {
    if (this._shuffle === on) return;
    this._shuffle = on;
    const currentId = this.order[this.posInOrder];
    if (on) {
      this.reshuffleTail();
    } else {
      this.order = this.tracks.map((t) => t.id);
    }
    this.posInOrder = currentId ? this.order.indexOf(currentId) : -1;
    this.notify();
  }

  setRepeat(mode: RepeatMode): void {
    this._repeat = mode;
    this.notify();
  }

  /** Reshuffles everything after the current position, keeping history in place. */
  private reshuffleTail(): void {
    const currentId = this.order[this.posInOrder];
    const remaining = this.tracks.map((t) => t.id).filter((id) => id !== currentId);
    const shuffled = shuffleArray(remaining);
    this.order = currentId ? [currentId, ...shuffled] : shuffled;
    this.posInOrder = currentId ? 0 : -1;
  }

  /** Advances to the next track per shuffle/repeat rules. Returns undefined at the end of the list. */
  next(): Track | undefined {
    if (this.order.length === 0) return undefined;
    if (this._repeat === 'one') {
      return this.currentTrack;
    }
    if (this.posInOrder + 1 < this.order.length) {
      this.posInOrder++;
      this.notify();
      return this.currentTrack;
    }
    // End of the order.
    if (this._repeat === 'all') {
      if (this._shuffle) {
        this.order = shuffleArray(this.tracks.map((t) => t.id));
      }
      this.posInOrder = 0;
      this.notify();
      return this.currentTrack;
    }
    return undefined;
  }

  /** True when the current position is the last one and repeat won't continue. */
  get isAtEnd(): boolean {
    return this._repeat === 'off' && this.posInOrder >= this.order.length - 1;
  }

  /** currentTimeSec > 3 restarts the track instead of moving back, per standard player UX. */
  previous(currentTimeSec: number): Track | undefined {
    if (this.order.length === 0) return undefined;
    if (currentTimeSec > 3) {
      return this.currentTrack;
    }
    if (this.posInOrder > 0) {
      this.posInOrder--;
      this.notify();
      return this.currentTrack;
    }
    return this.currentTrack;
  }

  peekNextId(): string | undefined {
    if (this.order.length === 0) return undefined;
    if (this._repeat === 'one') return this.order[this.posInOrder];
    if (this.posInOrder + 1 < this.order.length) return this.order[this.posInOrder + 1];
    if (this._repeat === 'all') return this.order[0];
    return undefined;
  }

  getResult(id: string): AnalysisResult | undefined {
    return this.results.get(id);
  }

  /**
   * Decodes (if needed) and analyzes (if needed) a track, reporting progress
   * through the track's own `progress`/`status` fields (and onChange). Only
   * one decode+analyze runs at a time; a second caller for a different track
   * while one is in flight is queued behind it.
   */
  async ensureLoaded(ctx: AudioContext, track: Track): Promise<Loaded> {
    const cachedResult = this.results.get(track.id);
    if (cachedResult) {
      const buffer = this.buffers.get(track.id) ?? (await this.decode(ctx, track));
      return { buffer, result: cachedResult };
    }

    // Serialize analysis: wait for anything already running first.
    while (this.analyzingId && this.analyzingId !== track.id) {
      await this.queuedAnalysis?.catch(() => {});
    }

    const already = this.results.get(track.id);
    if (already) {
      const buffer = this.buffers.get(track.id) ?? (await this.decode(ctx, track));
      return { buffer, result: already };
    }

    const work = this.analyzeTrack(ctx, track);
    this.analyzingId = track.id;
    this.queuedAnalysis = work;
    try {
      return await work;
    } finally {
      if (this.analyzingId === track.id) this.analyzingId = null;
      this.queuedAnalysis = null;
    }
  }

  private async decode(ctx: AudioContext, track: Track): Promise<AudioBuffer> {
    const cached = this.buffers.get(track.id);
    if (cached) return cached;
    const arrayBuffer = await track.file.arrayBuffer();
    const buffer = await ctx.decodeAudioData(arrayBuffer);
    this.buffers.set(track.id, buffer);
    track.duration = buffer.duration;
    return buffer;
  }

  private async analyzeTrack(ctx: AudioContext, track: Track): Promise<Loaded> {
    track.status = 'analyzing';
    track.progress = 0;
    track.error = undefined;
    this.notify();
    try {
      const buffer = await this.decode(ctx, track);
      const { analyzeAudio } = await import('../analysis/analyze');
      const result = await analyzeAudio(buffer, (_stage, p) => {
        track.progress = Math.max(0, Math.min(1, p));
        this.notify();
      });
      this.results.set(track.id, result);
      track.status = 'ready';
      track.progress = 1;
      track.duration = result.duration;
      this.notify();
      return { buffer, result };
    } catch (err) {
      track.status = 'error';
      track.error = err instanceof Error ? err.message : 'Analysis failed';
      this.notify();
      throw err;
    }
  }

  /** Kicks off background analysis of the "next up" track, if idle and not already analyzed. */
  prefetchNext(ctx: AudioContext): void {
    if (this.analyzingId) return;
    const nextId = this.peekNextId();
    if (!nextId || this.results.has(nextId)) return;
    const track = this.byId(nextId);
    if (!track || track.status === 'analyzing') return;
    void this.ensureLoaded(ctx, track).catch(() => {
      // Surfaced via track.status/error; nothing else to do here.
    });
  }

  /** Drops decoded buffers for tracks that are not current or next, to save memory. */
  evictStaleBuffers(): void {
    const keep = new Set<string>();
    const current = this.currentTrack;
    if (current) keep.add(current.id);
    const nextId = this.peekNextId();
    if (nextId) keep.add(nextId);
    for (const id of Array.from(this.buffers.keys())) {
      if (!keep.has(id)) this.buffers.delete(id);
    }
  }
}
