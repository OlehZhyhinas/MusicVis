// Artist and title from a file name when the tags have none: "Artist - Title (Official Video).mp3"
// -> { artist: 'Artist', title: 'Title' }. Also cleans tag titles of video-site noise.

import type { TrackMeta } from './types';

/** Bracketed or trailing words that are about the upload, not the song. */
const NOISE = /\b(official|video|music|audio|lyrics?|lyric video|visuali[sz]er|hd|hq|4k|1080p|720p|remaster(ed)?|explicit|clean|full|version|mv|m\/v|live|clip|original mix|radio edit|extended|color coded|prod\.?|produced by)\b/i;
const FEAT = /\s*[([]?\s*\b(feat\.?|ft\.?|featuring)\s+[^)\]]*[)\]]?/gi;

/** A title without "(Official Video)", "[Lyrics]", "(feat. X)", file junk and extra spaces. */
export function cleanTitle(raw: string): string {
  let s = raw.replace(/_/g, ' ');
  // Bracketed groups made of upload words go; anything else in brackets (a remix name) stays.
  s = s.replace(/\s*[([{【]([^)\]}】]*)[)\]}】]/g, (m, inner: string) => (NOISE.test(inner) || /^\s*\d{4}\s*$/.test(inner) ? '' : m));
  s = s.replace(FEAT, '');
  // A trailing "- Official Video" / "| Lyrics".
  s = s.replace(/\s+[-|]\s+[^-|]*$/, (m) => (NOISE.test(m) && m.split(/\s+/).length <= 5 ? '' : m));
  return s.replace(/\s+/g, ' ').replace(/^[\s\-–—.]+|[\s\-–—.]+$/g, '').trim();
}

/** A clean artist name (a trailing " - Topic" and "VEVO" go; featured artists stay out). */
export function cleanArtist(raw: string): string {
  return raw
    .replace(/_/g, ' ')
    .replace(/\s*-\s*topic$/i, '')
    .replace(/vevo$/i, '')
    .replace(FEAT, '')
    // Several artists in one tag ("A; B", "A / B", ID3v2.4's NUL separator): the first leads.
    .split(/\s*(?:;|\u0000|\s\/\s)\s*/)[0]
    .replace(/\s+/g, ' ')
    .trim();
}

/** Artist and title guessed from a file name (either may be missing). */
export function metaFromFilename(name: string): TrackMeta {
  let base = name.replace(/^.*[\\/]/, '').replace(/\.[a-z0-9]{2,4}$/i, '').replace(/_/g, ' ');
  // Leading track numbers: "01 - ", "01. ", "1 ".
  base = base.replace(/^\s*\d{1,3}\s*[-.)]\s*/, '');
  const parts = base
    .split(/\s+[-–—]\s+|\s*[–—]\s*/)
    .map((p) => p.trim())
    .filter((p) => p && !/^\d{1,3}$/.test(p));
  if (parts.length === 0) return {};
  if (parts.length === 1) {
    const title = cleanTitle(parts[0]);
    return title ? { title } : {};
  }
  // Two parts: "Artist - Title"; more: the last two ("Genre - Artist - Title", "Artist - Album - Title"
  // take the one before the title as the artist).
  const title = cleanTitle(parts[parts.length - 1]);
  const artist = cleanArtist(parts[parts.length - 2]);
  const out: TrackMeta = {};
  if (artist) out.artist = artist;
  if (title) out.title = title;
  return out;
}

/** Tags first, the file name for whatever they lack; titles and artists cleaned. */
export function mergeMeta(tags: TrackMeta, fileName: string): TrackMeta {
  const f = metaFromFilename(fileName);
  const out: TrackMeta = {};
  const artist = tags.artist ? cleanArtist(tags.artist) : f.artist;
  const title = tags.title ? cleanTitle(tags.title) : f.title;
  if (artist) out.artist = artist;
  if (title) out.title = title;
  if (tags.album) out.album = tags.album.trim();
  if (tags.duration && tags.duration > 0) out.duration = tags.duration;
  return out;
}
