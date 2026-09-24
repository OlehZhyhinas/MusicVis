// Timed lyrics -> the lyric fields of a MusicState, once per frame: the line playing, its progress,
// a pulse on each new line, and what the words are about (imagery tags, valence, arousal), read
// from the line and its neighbours so the picture drifts with the verse instead of flickering.

import type { MusicState } from '../types';
import type { LyricTrack } from './types';
import { lineAt } from './lrc';
import { readLine, TAG_COUNT, type LineMeaning } from './lexicon';

/** Weight of the current line against its neighbours (the rest is the context of +-2 lines). */
const LINE_WEIGHT = 0.6;
/** Seconds the words' meaning takes to settle (and to fade after the last line of a passage). */
const SETTLE = 1.2;
const FADE = 4;

export interface LyricFrame {
  index: number;
  text: string;
  next: string;
  progress: number;
  tags: Float32Array;
  valence: number;
  arousal: number;
  pulse: number;
  presence: number;
}

export class LyricSampler {
  readonly track: LyricTrack;
  /** Per-line meaning (the lexicon's reading; may be refined later, e.g. by the chat model). */
  readonly meanings: LineMeaning[];
  /** Per-line blend of the line and its neighbours. */
  private blended: LineMeaning[];
  private last = -2;
  private held = -1;
  private frame: LyricFrame = {
    index: -1, text: '', next: '', progress: 0, tags: new Float32Array(TAG_COUNT), valence: 0.5, arousal: 0.5, pulse: 0, presence: 0,
  };

  constructor(track: LyricTrack) {
    this.track = track;
    this.meanings = track.lines.map((l) => readLine(l.text));
    this.blended = this.blend();
  }

  /** Replaces some lines' meanings (the refinement hook); takes effect from the next frame. */
  refine(updates: { index: number; meaning: LineMeaning }[]): void {
    for (const u of updates) if (u.index >= 0 && u.index < this.meanings.length) this.meanings[u.index] = u.meaning;
    this.blended = this.blend();
  }

  private blend(): LineMeaning[] {
    const M = this.meanings;
    return M.map((m, i) => {
      const tags = new Float32Array(TAG_COUNT);
      let v = 0;
      let a = 0;
      let w = 0;
      let wsum = 0;
      for (let d = -2; d <= 2; d++) {
        const n = M[i + d];
        if (!n) continue;
        const k = d === 0 ? LINE_WEIGHT : ((1 - LINE_WEIGHT) / 4) * (Math.abs(d) === 1 ? 1.3 : 0.7);
        for (let t = 0; t < TAG_COUNT; t++) tags[t] += n.tags[t] * k;
        // Lines the lexicon did not understand pull valence and arousal less.
        const kk = k * (0.3 + 0.7 * n.weight);
        v += n.valence * kk;
        a += n.arousal * kk;
        w += n.weight * k;
        wsum += kk;
      }
      return { tags, valence: wsum ? v / wsum : 0.5, arousal: wsum ? a / wsum : 0.5, weight: w };
    });
  }

  /** The lyric frame at `time`; call once per frame with the frame's dt (smooths the meaning). */
  sample(time: number, dt: number): LyricFrame {
    const f = this.frame;
    const L = this.track.lines;
    const i = lineAt(this.track, time);
    // A seek (a jump of more than one line) snaps instead of easing.
    const jump = i >= 0 && (this.last === -2 || (this.held >= 0 && Math.abs(i - this.held) > 1));
    f.pulse *= Math.exp(-Math.max(0, dt) * 4);
    if (i >= 0 && i !== this.held) f.pulse = 1;
    if (i >= 0) this.held = i;
    this.last = i;
    f.index = i;
    f.text = i >= 0 ? L[i].text : '';
    // The line coming next (for a karaoke display): the following one, or the first ahead of a gap.
    let nx = i >= 0 ? i + 1 : L.findIndex((l) => l.t > time);
    if (nx < 0) nx = L.length;
    f.next = nx < L.length ? L[nx].text : '';
    f.progress = i >= 0 ? Math.min(1, Math.max(0, (time - L[i].t) / Math.max(0.05, L[i].end - L[i].t))) : 0;
    const src = this.held >= 0 ? this.blended[this.held] : null;
    const k = jump ? 1 : 1 - Math.exp(-Math.max(0, dt) / SETTLE);
    const presTarget = i >= 0 ? 1 : 0;
    f.presence += (presTarget - f.presence) * (presTarget > f.presence ? k : 1 - Math.exp(-Math.max(0, dt) / FADE));
    if (src) {
      for (let t = 0; t < TAG_COUNT; t++) f.tags[t] += (src.tags[t] - f.tags[t]) * k;
      f.valence += (src.valence - f.valence) * k;
      f.arousal += (src.arousal - f.arousal) * k;
    }
    return f;
  }

  /** Seeks: the next sample snaps to its line. */
  reset(): void {
    this.last = -2;
    this.held = -1;
    this.frame.pulse = 0;
  }

  /** Writes the frame into a MusicState's lyric fields. */
  apply(state: MusicState, dt: number): void {
    const f = this.sample(state.time, dt);
    state.lyricLine = f.text;
    state.lyricNext = f.next;
    state.lyricIndex = f.index;
    state.lyricProgress = f.progress;
    state.lyricTags = f.tags;
    state.lyricValence = f.valence;
    state.lyricArousal = f.arousal;
    state.lyricPulse = f.pulse;
    state.lyricPresence = f.presence;
    state.lyricSynced = this.track.synced;
  }
}
