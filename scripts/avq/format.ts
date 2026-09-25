// On-disk format of one rendered clip (preset x song x window), shared by the page that
// writes it and the Node tools that read it.
//
//   <base>.json  header: meta, field names, song summary, hooks, moments, byte offsets
//   <base>.bin   [frames*fields float32 row-major][frames*THUMB bytes RGB][frames*SPEC bytes]
//
// Row i is the frame rendered at time t = start + (i + 1) / fps (the MusicState time).

import { VIS_FIELDS, THUMB_W, THUMB_H } from './features';
import { SPEC_BANDS } from './audio';
import type { Hook, Moment, SectionLite } from './music';
import type { ClipLike } from './metrics';

export const STEMS = ['drums', 'bass', 'vocals', 'other'] as const;

export const MUSIC_FIELDS = [
  't',
  'drums', 'bass', 'vocals', 'other', // stem envelopes 0..1
  'onDrums', 'onBass', 'onVocals', 'onOther', // stem onsets 0..1
  'prDrums', 'prBass', 'prVocals', 'prOther', // stem presence 0..1
  'loud', 'lBass', 'lMid', 'lTreb', // loudness, live band levels (~1 = usual)
  'beatPhase', 'beatPulse', 'barPhase', 'barPulse', 'onBeat', 'onBar', 'beatIndex', 'barIndex',
  'section', 'sectionChanged', 'dropPulse', 'build', 'timeToDrop',
  'repeatGroup', 'repeatIndex', 'repeatSim',
  'chord', 'tension', 'chordPulse', 'keyHue',
  'melMidi', 'melSal', // melody probe: MIDI pitch (NaN when unpitched) and salience
  'tBright', 'tNoise', 'complexity', // mix timbre brightness / noisiness, musical density
  'c0', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9', 'c10', 'c11', // chroma
] as const;

export const FIELDS = [...MUSIC_FIELDS, ...VIS_FIELDS] as const;
export type Field = (typeof FIELDS)[number];
export const THUMB_BYTES = THUMB_W * THUMB_H * 3;
export { THUMB_W, THUMB_H, SPEC_BANDS };

export interface ClipHeader {
  version: 1;
  preset: { id: string; name: string };
  song: { slug: string; path: string; duration: number; bpm: number; beatsPerBar: number };
  clip: { label: string; start: number; end: number; warm: number };
  render: { w: number; h: number; fps: number; seed: number; ms: number; hq: boolean };
  fields: string[];
  frames: number;
  thumb: { w: number; h: number; offset: number };
  spec: { bands: number; offset: number };
  /** Song-level context (whole song, not just the window). */
  beats: number[];
  downbeats: number[];
  sections: SectionLite[];
  moments: Moment[];
  hooks: Hook[];
  /** JPEG frames directory relative to .testdata/avq/, when frames were saved (%06d.jpg, index = row). */
  framesDir?: string;
}

export interface Clip {
  header: ClipHeader;
  n: number;
  /** Column accessor. */
  col(f: Field): Float32Array;
  thumb(i: number): Uint8Array;
  spec(i: number): Uint8Array;
}

export function parseClip(header: ClipHeader, bin: Uint8Array): Clip {
  const F = header.fields.length;
  const n = header.frames;
  const data = new Float32Array(bin.buffer.slice(bin.byteOffset, bin.byteOffset + n * F * 4));
  const cols = new Map<string, Float32Array>();
  return {
    header,
    n,
    col(f: Field) {
      let c = cols.get(f);
      if (!c) {
        const k = header.fields.indexOf(f);
        if (k < 0) throw new Error('no field ' + f);
        c = new Float32Array(n);
        for (let i = 0; i < n; i++) c[i] = data[i * F + k];
        cols.set(f, c);
      }
      return c;
    },
    thumb: (i) => bin.subarray(header.thumb.offset + i * THUMB_BYTES, header.thumb.offset + (i + 1) * THUMB_BYTES),
    spec: (i) => bin.subarray(header.spec.offset + i * header.spec.bands, header.spec.offset + (i + 1) * header.spec.bands),
  };
}

/** Adapter from a parsed clip to the metrics' ClipLike. */
export function asClipLike(c: Clip): ClipLike {
  const h = c.header;
  return {
    fps: h.render.fps, n: c.n,
    col: (name) => c.col(name as Field),
    has: (name) => h.fields.includes(name),
    thumb: (i) => c.thumb(i),
    thumbW: h.thumb.w, thumbH: h.thumb.h,
    beats: h.beats, downbeats: h.downbeats, beatsPerBar: h.song.beatsPerBar, bpm: h.song.bpm,
    sections: h.sections, moments: h.moments, hooks: h.hooks,
  };
}

/** Counterfactual result file (page.ts counterfactual()), .testdata/avq/cf/<base>.json. */
export interface CfResult {
  preset: { id: string; name: string };
  song: { slug: string; bpm: number; beatsPerBar: number };
  clip: { label: string; start: number; end: number };
  halfBarFrames: number;
  motion: number;
  step: number;
  reactions: { src: string; target: string; gain: number }[];
  variants: { id: string; kind: string; mean: number; rel: number; motionRel?: number; series: number[]; stem?: string; reaction?: number }[];
  motionSeries: (number | null)[];
}
