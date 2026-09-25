// Live clip recorder: collects the per-frame MusicState fields and visual features the AV
// metrics need while presets render in the app (the clip-duel page), and exposes them as a
// ClipLike so the same report card code runs in the app and in the offline harness
// (scripts/avq). Melody comes from the analysis' note line (MusicState.notes) instead of the
// harness's pitch probe; without notes, melody-based scores come out as unknown (NaN).

import type { MusicState } from '../../types';
import { THUMB_H, THUMB_W, VIS_FIELDS, VisualFeatures } from './features';
import type { ClipLike } from './metrics';
import type { Hook, Moment, SectionLite } from './music';

const MUSIC = ['t', 'drums', 'bass', 'vocals', 'other', 'onDrums', 'onBass', 'onVocals', 'onOther', 'prDrums', 'prBass', 'prVocals', 'prOther', 'loud', 'onBeat', 'melMidi', 'melSal', 'tBright', 'barPhase', 'beatPhase', 'beatIndex'] as const;

export interface SongContext {
  fps: number;
  bpm: number;
  beats: number[];
  downbeats: number[];
  beatsPerBar: number;
  sections: SectionLite[];
  moments: Moment[];
  hooks: Hook[];
}

export class LiveRecorder {
  private vis: VisualFeatures;
  private cols = new Map<string, number[]>();
  private thumbs: Uint8Array[] = [];
  /** Song time of the first recorded frame minus one frame (clipT0 for the metrics). */
  t0 = NaN;

  constructor(w: number, h: number) {
    this.vis = new VisualFeatures(w, h);
    for (const k of [...MUSIC, ...VIS_FIELDS]) this.cols.set(k, []);
  }

  get n(): number {
    return this.thumbs.length;
  }

  /** Record one frame: the state it was rendered from and its RGBA readback (rows bottom-up). */
  push(s: MusicState, px: Uint8Array, fps: number): void {
    if (!this.thumbs.length) this.t0 = s.time - 1 / fps;
    const m: Record<(typeof MUSIC)[number], number> = {
      t: s.time, drums: s.stems.drums, bass: s.stems.bass, vocals: s.stems.vocals, other: s.stems.other,
      onDrums: s.stemOnsets.drums, onBass: s.stemOnsets.bass, onVocals: s.stemOnsets.vocals, onOther: s.stemOnsets.other,
      prDrums: s.stemPresence.drums, prBass: s.stemPresence.bass, prVocals: s.stemPresence.vocals, prOther: s.stemPresence.other,
      loud: s.loudness, onBeat: s.onBeat ? 1 : 0, tBright: s.timbre?.mix.bright ?? 0,
      // The analysis' melody line (notes gene) stands in for the harness's pitch probe.
      melMidi: s.notes && s.notes.held > 0.1 ? s.notes.pitch : NaN, melSal: s.notes ? Math.min(1, s.notes.held * 1.2) : 0,
      barPhase: s.barPhase, beatPhase: s.beatPhase, beatIndex: s.beatIndex,
    };
    for (const k of MUSIC) this.cols.get(k)!.push(m[k]);
    const v = this.vis.update(px);
    VIS_FIELDS.forEach((k, i) => this.cols.get(k)!.push(v[i]));
    this.thumbs.push(this.vis.thumb.slice());
  }

  clip(ctx: SongContext): ClipLike {
    const cache = new Map<string, Float32Array>();
    const n = this.n;
    return {
      fps: ctx.fps, n,
      col: (name) => {
        let c = cache.get(name);
        if (!c) {
          const src = this.cols.get(name);
          c = src ? Float32Array.from(src) : new Float32Array(n);
          cache.set(name, c);
        }
        return c;
      },
      has: (name) => this.cols.has(name),
      thumb: (i) => this.thumbs[i],
      thumbW: THUMB_W, thumbH: THUMB_H,
      beats: ctx.beats, downbeats: ctx.downbeats, beatsPerBar: ctx.beatsPerBar, bpm: ctx.bpm,
      sections: ctx.sections, moments: ctx.moments, hooks: ctx.hooks,
    };
  }
}
