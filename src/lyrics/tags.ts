// A small audio tag reader: artist, title, album and (when stored) duration from ID3v2 (MP3, also
// ID3v1 at the end), FLAC and Ogg Vorbis/Opus comments, and MP4/M4A atoms. Reads only the bytes it
// needs (the head of the file, the tail for ID3v1, the moov atom for MP4). No dependencies.

import type { TrackMeta } from './types';

const latin1 = (b: Uint8Array) => {
  let s = '';
  for (let i = 0; i < b.length; i++) s += String.fromCharCode(b[i]);
  return s;
};
const utf8 = (b: Uint8Array) => new TextDecoder('utf-8').decode(b);
const strictUtf8 = new TextDecoder('utf-8', { fatal: true });
/** Latin-1 text that is really UTF-8 (a common tagger bug) is read as UTF-8. */
const latin1OrUtf8 = (b: Uint8Array) => {
  if (b.some((x) => x >= 0x80)) {
    try {
      return strictUtf8.decode(b);
    } catch {
      // Real Latin-1.
    }
  }
  return latin1(b);
};
/** Tag durations beyond this are junk (some taggers store samples or bytes in TLEN). */
const MAX_DURATION = 3 * 3600;
const ascii = (b: Uint8Array, at: number, n: number) => latin1(b.subarray(at, at + n));
const be32 = (b: Uint8Array, at: number) => ((b[at] << 24) | (b[at + 1] << 16) | (b[at + 2] << 8) | b[at + 3]) >>> 0;
const le32 = (b: Uint8Array, at: number) => (b[at] | (b[at + 1] << 8) | (b[at + 2] << 16) | (b[at + 3] << 24)) >>> 0;
const syncsafe = (b: Uint8Array, at: number) => ((b[at] & 0x7f) << 21) | ((b[at + 1] & 0x7f) << 14) | ((b[at + 2] & 0x7f) << 7) | (b[at + 3] & 0x7f);
const trimNul = (s: string) => s.replace(/\u0000+$/g, '').replace(/^\uFEFF/, '').trim();

// ------------------------------------------------------------------ ID3v2

/** Undoes ID3 unsynchronisation (every 0xFF 0x00 becomes 0xFF). */
export function unsync(b: Uint8Array): Uint8Array {
  const out = new Uint8Array(b.length);
  let n = 0;
  for (let i = 0; i < b.length; i++) {
    out[n++] = b[i];
    if (b[i] === 0xff && b[i + 1] === 0x00) i++;
  }
  return out.subarray(0, n);
}

function utf16(b: Uint8Array, bigEndian: boolean): string {
  let s = '';
  for (let i = 0; i + 1 < b.length; i += 2) s += String.fromCharCode(bigEndian ? (b[i] << 8) | b[i + 1] : b[i] | (b[i + 1] << 8));
  return s;
}

/** An ID3 text frame's value (encoding byte, then text; several values are NUL-separated). */
export function id3Text(b: Uint8Array): string {
  if (!b.length) return '';
  const enc = b[0];
  const body = b.subarray(1);
  let s: string;
  if (enc === 1) {
    // UTF-16 with a BOM (each value may carry its own; little endian without one).
    const be = body[0] === 0xfe && body[1] === 0xff;
    const bom = (body[0] === 0xff && body[1] === 0xfe) || be;
    s = utf16(bom ? body.subarray(2) : body, be);
  } else if (enc === 2) s = utf16(body, true);
  else if (enc === 3) s = utf8(body);
  else s = latin1OrUtf8(body);
  // Several values: keep the first, joined the way the tags show them.
  const vals = s.split('\u0000').map((v) => v.replace(/^\uFEFF|^\uFFFE/, '').trim()).filter(Boolean);
  return vals.join('; ');
}

const ID3_KEYS: Record<string, keyof TrackMeta> = {
  TIT2: 'title', TT2: 'title', TPE1: 'artist', TP1: 'artist', TALB: 'album', TAL: 'album', TLEN: 'duration', TLE: 'duration',
  TPE2: 'artist', TP2: 'artist',
};

/** Tags from an ID3v2 block at the start of `b` (empty when there is none). */
export function readId3v2(b: Uint8Array): TrackMeta {
  const out: TrackMeta = {};
  if (b.length < 10 || ascii(b, 0, 3) !== 'ID3') return out;
  const ver = b[3];
  const flags = b[5];
  const size = syncsafe(b, 6);
  let tag = b.subarray(10, Math.min(b.length, 10 + size));
  if (ver < 4 && flags & 0x80) tag = unsync(tag);
  let pos = 0;
  if (flags & 0x40 && ver >= 3) {
    // Extended header: v2.4 counts its own size field (syncsafe), v2.3 does not.
    pos = ver === 4 ? syncsafe(tag, 0) : be32(tag, 0) + 4;
  }
  const idLen = ver === 2 ? 3 : 4;
  const hdrLen = ver === 2 ? 6 : 10;
  let albumArtist: string | undefined;
  while (pos + hdrLen <= tag.length) {
    const id = ascii(tag, pos, idLen);
    if (!/^[A-Z0-9]+$/.test(id)) break; // padding
    const fsize = ver === 2 ? (tag[pos + 3] << 16) | (tag[pos + 4] << 8) | tag[pos + 5] : ver === 4 ? syncsafe(tag, pos + 4) : be32(tag, pos + 4);
    const fflags = ver === 2 ? 0 : tag[pos + 9];
    let data = tag.subarray(pos + hdrLen, pos + hdrLen + fsize);
    pos += hdrLen + fsize;
    if (fsize <= 0) continue;
    const key = ID3_KEYS[id];
    if (!key) continue;
    if (ver === 4) {
      if (fflags & 0x0c) continue; // compressed or encrypted
      if (fflags & 0x01) data = data.subarray(4); // data length indicator
      if (fflags & 0x02) data = unsync(data);
    } else if (ver === 3 && fflags & 0xc0) continue;
    const v = id3Text(data);
    if (!v) continue;
    if (id === 'TPE2' || id === 'TP2') albumArtist = v;
    else if (key === 'duration') {
      const ms = parseFloat(v);
      if (ms > 0 && ms / 1000 < MAX_DURATION) out.duration = ms / 1000;
    } else (out as Record<string, string>)[key] = v;
  }
  if (!out.artist && albumArtist) out.artist = albumArtist;
  return out;
}

/** Tags from a 128-byte ID3v1 block ("TAG" + title 30 + artist 30 + album 30 ...). */
export function readId3v1(b: Uint8Array): TrackMeta {
  const out: TrackMeta = {};
  if (b.length < 128) return out;
  const t = b.subarray(b.length - 128);
  if (ascii(t, 0, 3) !== 'TAG') return out;
  const field = (at: number) => {
    const f = t.subarray(at, at + 30);
    const z = f.indexOf(0);
    return trimNul(latin1OrUtf8(z >= 0 ? f.subarray(0, z) : f));
  };
  const title = field(3);
  const artist = field(33);
  const album = field(63);
  if (title) out.title = title;
  if (artist) out.artist = artist;
  if (album) out.album = album;
  return out;
}

// --------------------------------------------------------- Vorbis comments

/** A Vorbis comment block (vendor, then count of "KEY=value" strings, little endian). */
export function readVorbisComments(b: Uint8Array, at = 0): TrackMeta {
  const out: TrackMeta = {};
  if (at + 8 > b.length) return out;
  const vendor = le32(b, at);
  let pos = at + 4 + vendor;
  if (pos + 4 > b.length) return out;
  const n = le32(b, pos);
  pos += 4;
  let albumArtist: string | undefined;
  for (let i = 0; i < n && pos + 4 <= b.length; i++) {
    const len = le32(b, pos);
    pos += 4;
    const s = utf8(b.subarray(pos, Math.min(b.length, pos + len)));
    pos += len;
    const eq = s.indexOf('=');
    if (eq < 0) continue;
    const k = s.slice(0, eq).toUpperCase();
    const v = s.slice(eq + 1).trim();
    if (!v) continue;
    if (k === 'TITLE' && !out.title) out.title = v;
    else if (k === 'ARTIST' && !out.artist) out.artist = v;
    else if (k === 'ALBUMARTIST' || k === 'ALBUM ARTIST') albumArtist ??= v;
    else if (k === 'ALBUM' && !out.album) out.album = v;
  }
  if (!out.artist && albumArtist) out.artist = albumArtist;
  return out;
}

/** FLAC: metadata blocks after "fLaC" (STREAMINFO gives the duration, VORBIS_COMMENT the tags). */
export function readFlac(b: Uint8Array): TrackMeta {
  let out: TrackMeta = {};
  let pos = ascii(b, 0, 3) === 'ID3' ? 10 + syncsafe(b, 6) : 0;
  if (ascii(b, pos, 4) !== 'fLaC') return out;
  pos += 4;
  let duration: number | undefined;
  for (let guard = 0; guard < 64 && pos + 4 <= b.length; guard++) {
    const last = b[pos] & 0x80;
    const type = b[pos] & 0x7f;
    const len = (b[pos + 1] << 16) | (b[pos + 2] << 8) | b[pos + 3];
    const at = pos + 4;
    if (type === 0 && at + 18 <= b.length) {
      const rate = (b[at + 10] << 12) | (b[at + 11] << 4) | (b[at + 12] >> 4);
      const total = (b[at + 13] & 0x0f) * 2 ** 32 + be32(b, at + 14);
      if (rate > 0 && total > 0 && total / rate < MAX_DURATION) duration = total / rate;
    } else if (type === 4) out = readVorbisComments(b, at);
    pos = at + len;
    if (last) break;
  }
  if (duration) out.duration = duration;
  return out;
}

/** Ogg (Vorbis or Opus): the comment header in the first pages (packets joined across pages). */
export function readOgg(b: Uint8Array): TrackMeta {
  // Concatenate the payload of the first few pages, then look for the comment packet's magic.
  const chunks: Uint8Array[] = [];
  let pos = 0;
  let total = 0;
  for (let page = 0; page < 16 && pos + 27 <= b.length && ascii(b, pos, 4) === 'OggS'; page++) {
    const segs = b[pos + 26];
    let len = 0;
    for (let i = 0; i < segs; i++) len += b[pos + 27 + i];
    const start = pos + 27 + segs;
    chunks.push(b.subarray(start, Math.min(b.length, start + len)));
    total += len;
    pos = start + len;
  }
  const all = new Uint8Array(total);
  let o = 0;
  for (const c of chunks) {
    all.set(c, o);
    o += c.length;
  }
  for (let i = 0; i + 8 < all.length; i++) {
    if (all[i] === 3 && ascii(all, i + 1, 6) === 'vorbis') return readVorbisComments(all, i + 7);
    if (all[i] === 0x4f && ascii(all, i, 8) === 'OpusTags') return readVorbisComments(all, i + 8);
  }
  return {};
}

// ---------------------------------------------------------------------- MP4

const MP4_CONTAINERS = new Set(['moov', 'udta', 'meta', 'ilst', 'trak', 'mdia']);
const MP4_KEYS: Record<string, keyof TrackMeta> = { '©nam': 'title', '©ART': 'artist', '©alb': 'album', aART: 'artist' };

/** Tags from an MP4 atom tree (b holds the atoms from `start` to `end`: a file head or a moov atom). */
export function readMp4(b: Uint8Array, start = 0, end = b.length, out: TrackMeta = {}): TrackMeta {
  let pos = start;
  while (pos + 8 <= end) {
    let size = be32(b, pos);
    const type = ascii(b, pos + 4, 4);
    let hdr = 8;
    if (size === 1 && pos + 16 <= end) {
      size = be32(b, pos + 8) * 2 ** 32 + be32(b, pos + 12);
      hdr = 16;
    } else if (size === 0) size = end - pos;
    if (size < hdr) break;
    const body = pos + hdr;
    const stop = Math.min(end, pos + size);
    if (type === 'mvhd' && body + 20 <= stop) {
      const v = b[body];
      const scale = v === 1 ? be32(b, body + 20) : be32(b, body + 12);
      const dur = v === 1 ? be32(b, body + 24) * 2 ** 32 + be32(b, body + 28) : be32(b, body + 16);
      if (scale > 0 && dur > 0 && dur / scale < MAX_DURATION) out.duration = dur / scale;
    } else if (MP4_KEYS[type] !== undefined) {
      // ilst item: a 'data' atom (size, 'data', type 4 bytes, locale 4 bytes, value).
      let p = body;
      while (p + 16 <= stop) {
        const s = be32(b, p);
        if (s < 16) break;
        if (ascii(b, p + 4, 4) === 'data') {
          const v = utf8(b.subarray(p + 16, Math.min(stop, p + s))).trim();
          const key = MP4_KEYS[type];
          if (v && !(type === 'aART' && out.artist)) (out as Record<string, string>)[key] = v;
          break;
        }
        p += s;
      }
    } else if (MP4_CONTAINERS.has(type)) {
      // 'meta' is a full atom: version and flags before its children (QuickTime writes it without).
      const skip = type === 'meta' && ascii(b, body + 4, 4) !== 'hdlr' ? 4 : 0;
      readMp4(b, body + skip, stop, out);
    }
    pos += size;
  }
  return out;
}

// --------------------------------------------------------------- dispatch

export type TagFormat = 'id3' | 'flac' | 'ogg' | 'mp4' | 'unknown';

/** The container of a file from its first bytes. */
export function sniff(b: Uint8Array): TagFormat {
  if (ascii(b, 0, 4) === 'fLaC') return 'flac';
  if (ascii(b, 0, 3) === 'ID3') {
    const after = 10 + syncsafe(b, 6);
    return ascii(b, after, 4) === 'fLaC' ? 'flac' : 'id3';
  }
  if (ascii(b, 0, 4) === 'OggS') return 'ogg';
  if (ascii(b, 4, 4) === 'ftyp') return 'mp4';
  return 'unknown';
}

/** Tags from a buffer holding the start of a file (and, for ID3v1, possibly its end). */
export function readTags(head: Uint8Array, tail?: Uint8Array): TrackMeta {
  let out: TrackMeta = {};
  try {
    switch (sniff(head)) {
      case 'flac': out = readFlac(head); break;
      case 'ogg': out = readOgg(head); break;
      case 'mp4': out = readMp4(head); break;
      case 'id3': out = readId3v2(head); break;
    }
    if (!out.title && tail) {
      out = { ...readId3v1(tail), ...out };
    }
  } catch {
    // A broken tag gives what was read so far (or nothing); the file name fills in.
  }
  for (const k of ['artist', 'title', 'album'] as const) {
    const v = out[k];
    if (v !== undefined) {
      const t = trimNul(v);
      if (t) out[k] = t;
      else delete out[k];
    }
  }
  return out;
}

const HEAD_BYTES = 512 * 1024;

/** Tags of a File (or Blob): reads the head, the ID3v1 tail, and an MP4 moov atom placed further in. */
export async function readFileTags(file: Blob): Promise<TrackMeta> {
  const read = async (a: number, b: number) => new Uint8Array(await file.slice(a, b).arrayBuffer());
  const head = await read(0, Math.min(file.size, HEAD_BYTES));
  const fmt = sniff(head);
  if (fmt === 'id3' && head.length >= 10) {
    // A tag bigger than the head (cover art first): read all of it.
    const need = 10 + syncsafe(head, 6);
    if (need > head.length && need < 16 * 1024 * 1024) return readTags(await read(0, need), await read(Math.max(0, file.size - 128), file.size));
  }
  if (fmt === 'mp4') {
    // Walk the top-level atoms by their headers only; parse the moov wherever it sits.
    let pos = 0;
    const out: TrackMeta = {};
    for (let guard = 0; guard < 64 && pos + 8 <= file.size; guard++) {
      const h = pos + 16 <= head.length ? head.subarray(pos, pos + 16) : await read(pos, pos + 16);
      let size = be32(h, 0);
      const type = ascii(h, 4, 4);
      if (size === 1) size = be32(h, 8) * 2 ** 32 + be32(h, 12);
      else if (size === 0) size = file.size - pos;
      if (size < 8) break;
      if (type === 'moov') {
        if (size > 32 * 1024 * 1024) break;
        const moov = pos + size <= head.length ? head.subarray(pos, pos + size) : await read(pos, pos + size);
        return readMp4(moov, 0, moov.length, out);
      }
      pos += size;
    }
    return out;
  }
  const tail = fmt === 'id3' || fmt === 'unknown' ? await read(Math.max(0, file.size - 128), file.size) : undefined;
  return readTags(head, tail);
}
