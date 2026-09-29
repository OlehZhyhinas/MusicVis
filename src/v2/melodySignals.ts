import type { NoteStats } from '../types';

export type MelodySignal = 'register' | 'rising' | 'falling';
const unit = (v: number | undefined, fallback = 0): number => Number.isFinite(v) ? Math.max(0, Math.min(1, v!)) : fallback;

/** Reuse the note tracker's range and vibrato-filtered slope; no second pitch detector. */
export function melodySignal(source: MelodySignal, notes?: NoteStats): number {
  // The tracker retains the last height between notes. Missing analysis rests at centre.
  if (source === 'register') return unit(notes?.height, 0.5);
  // A stale pitch slope must not keep bending a body after a note has ended.
  if (!(unit(notes?.held) > 0)) return 0;
  const slope = notes?.glide;
  if (!Number.isFinite(slope)) return 0;
  return unit((source === 'rising' ? slope! : -slope!) / 12);
}
