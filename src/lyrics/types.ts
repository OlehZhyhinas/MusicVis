// Shared lyric types: the song's metadata (from its tags or file name), the lyrics found for it and
// the lookup status the playlist shows. Pure types, no runtime code.

/** What a song is, as read from its tags (or guessed from its file name). */
export interface TrackMeta {
  artist?: string;
  title?: string;
  album?: string;
  /** Seconds, when the tags carry it (ID3 TLEN, MP4 mvhd, FLAC STREAMINFO). */
  duration?: number;
}

export interface LyricLine {
  /** Start, seconds. */
  t: number;
  /** End, seconds (the next line's start, or a few seconds after the last one). */
  end: number;
  text: string;
}

/** A song's lyrics as timed lines. */
export interface LyricTrack {
  lines: LyricLine[];
  /** true: times come from the lyrics (LRC); false: plain lyrics spread over the vocals. */
  synced: boolean;
}

/**
 * Where a lookup stands: pending (not asked yet or in flight), synced / plain (found), instrumental
 * (the song has no words), missing (not found), offline (the lookup failed; tried again later).
 */
export type LyricStatus = 'pending' | 'synced' | 'plain' | 'instrumental' | 'missing' | 'offline';

/** A lookup's outcome: lyrics text as the service gave it, or why there is none. */
export interface LyricResult {
  status: LyricStatus;
  /** LRC text (status synced). */
  synced?: string;
  /** Untimed lines (status plain; also kept with synced when the service had both). */
  plain?: string;
  /** How the song was matched (artist - title), for the HUD. */
  source?: string;
}
