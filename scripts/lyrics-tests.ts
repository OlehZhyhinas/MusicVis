// Tests for lyrics (src/lyrics/**): tag reader on crafted byte buffers, file name cleaning, LRC
// parsing, plain-lyric spreading, and the LRCLIB lookup against a mocked fetch. Called from v2-test.ts.

import { readTags, readId3v2, readId3v1, readFlac, readOgg, readMp4, sniff, unsync } from '../src/lyrics/tags';
import { cleanTitle, metaFromFilename, mergeMeta } from '../src/lyrics/filename';
import { parseLrc, spreadPlain, vocalRegions, lineAt } from '../src/lyrics/lrc';
import { lookupLyrics, memoryCache, cacheKey, MISSING_TTL_MS, fitsDuration } from '../src/lyrics/lrclib';
import { LyricsLibrary } from '../src/lyrics/library';
import { alignLines, shiftTrack, lineOnsetFit } from '../src/lyrics/align';
import { readLine, topTags, lookupWord, LYRIC_TAGS, TAG_COUNT, tagIndex } from '../src/lyrics/lexicon';
import { LyricSampler } from '../src/lyrics/sampler';
import { COST_BUDGET_MS, cloneGenome, estimateCost, repair, validate, type Genome } from '../src/v2/genome';
import { crossover, mulberry32, mutate, MUTATION_NAMES } from '../src/v2/ops';
import { SEEDS } from '../src/v2/seeds';
import { ADJ_POOLS, nameFor } from '../src/v2/naming';
import { genomeGene } from '../src/v2/geneRegistry';
import {
  LYRICS_SCHEMA, NEUTRAL_NUDGE, TAG_EFFECTS, crossLyrics, easeNudge, lineKick, lyricTarget, lyricsCost, randomLyrics, repairLyrics, validateLyrics,
  type LyricNudge,
} from '../src/v2/genes/lyrics';

type Check = (name: string, ok: boolean, detail: string) => void;

const enc = new TextEncoder();
const bytes = (...parts: (number[] | Uint8Array | string)[]): Uint8Array => {
  const arrs = parts.map((p) => (typeof p === 'string' ? Uint8Array.from([...p].map((c) => c.charCodeAt(0) & 0xff)) : Uint8Array.from(p)));
  const out = new Uint8Array(arrs.reduce((s, a) => s + a.length, 0));
  let o = 0;
  for (const a of arrs) {
    out.set(a, o);
    o += a.length;
  }
  return out;
};
const be32 = (n: number) => [(n >>> 24) & 255, (n >>> 16) & 255, (n >>> 8) & 255, n & 255];
const le32 = (n: number) => [n & 255, (n >>> 8) & 255, (n >>> 16) & 255, (n >>> 24) & 255];
const ss = (n: number) => [(n >> 21) & 127, (n >> 14) & 127, (n >> 7) & 127, n & 127];
const zeros = (n: number) => new Array<number>(n).fill(0);
const pad30 = (s: string) => bytes(s, zeros(30 - s.length));
const utf16le = (s: string) => {
  const out: number[] = [0xff, 0xfe];
  for (const c of s) out.push(c.charCodeAt(0) & 255, c.charCodeAt(0) >> 8);
  return out;
};

function id3v23(frames: [string, Uint8Array][]): Uint8Array {
  const body = bytes(...frames.map(([id, d]) => bytes(id, be32(d.length), [0, 0], d)), zeros(20));
  return bytes('ID3', [3, 0, 0], ss(body.length), body, [0xff, 0xfb, 0x90, 0x00]);
}
function id3v24(frames: [string, Uint8Array][]): Uint8Array {
  const body = bytes(...frames.map(([id, d]) => bytes(id, ss(d.length), [0, 0], d)));
  return bytes('ID3', [4, 0, 0], ss(body.length), body);
}
function id3v22(frames: [string, Uint8Array][]): Uint8Array {
  const body = bytes(...frames.map(([id, d]) => bytes(id, [(d.length >> 16) & 255, (d.length >> 8) & 255, d.length & 255], d)));
  return bytes('ID3', [2, 0, 0], ss(body.length), body);
}
const txt = (s: string, e = 0) => (e === 3 ? bytes([3], enc.encode(s)) : e === 1 ? bytes([1], utf16le(s), [0, 0]) : bytes([0], s));

function vorbis(comments: string[]): Uint8Array {
  const v = enc.encode('test vendor');
  return bytes(le32(v.length), v, le32(comments.length), ...comments.map((c) => bytes(le32(enc.encode(c).length), enc.encode(c))));
}
function atom(type: string, ...kids: Uint8Array[]): Uint8Array {
  const body = bytes(...kids);
  return bytes(be32(8 + body.length), type, body);
}
const dataAtom = (s: string) => atom('data', bytes([0, 0, 0, 1, 0, 0, 0, 0], enc.encode(s)));
const COPY = String.fromCharCode(0xa9);

export async function lyricsTests(check: Check): Promise<void> {
  // ---------------------------------------------------------------- tags
  {
    const t = readTags(id3v23([['TIT2', txt('Hello World')], ['TPE1', txt('Björk', 1)], ['TALB', txt('Album', 3)], ['TLEN', txt('215000')]]));
    check('lyrics.tags.id3v23', t.title === 'Hello World' && t.artist === 'Björk' && t.album === 'Album' && t.duration === 215, JSON.stringify(t));
    const t4 = readId3v2(id3v24([['TPE1', txt('Sigur Rós', 3)], ['TIT2', bytes([3], enc.encode('Hoppípolla'), [0], enc.encode('alt'))]]));
    check('lyrics.tags.id3v24-utf8-multi', t4.artist === 'Sigur Rós' && t4.title === 'Hoppípolla; alt', JSON.stringify(t4));
    const t2 = readId3v2(id3v22([['TT2', txt('Old')], ['TP1', txt('Timer')]]));
    check('lyrics.tags.id3v22', t2.title === 'Old' && t2.artist === 'Timer', JSON.stringify(t2));
    // Album artist fills a missing artist; ID3v1 fills a missing ID3v2.
    const aa = readId3v2(id3v23([['TIT2', txt('X')], ['TPE2', txt('Band')]]));
    const v1 = bytes('TAG', pad30('Tail Title'), pad30('Tail Artist'), pad30('Tail Album'), zeros(35));
    const onlyV1 = readTags(bytes([0xff, 0xfb, 0x90, 0x00], zeros(100)), bytes(zeros(50), v1));
    check('lyrics.tags.album-artist+id3v1', aa.artist === 'Band' && onlyV1.title === 'Tail Title' && onlyV1.artist === 'Tail Artist' && readId3v1(v1).album === 'Tail Album', JSON.stringify({ aa, onlyV1 }));
    // Latin-1 frames holding UTF-8 bytes (a tagger bug) read as UTF-8; a TLEN of samples is ignored.
    const mis = readId3v2(id3v23([['TPE1', bytes([0], enc.encode('Björn'))], ['TLEN', txt('2465310208')]]));
    check('lyrics.tags.latin1-utf8+bad-tlen', mis.artist === 'Björn' && mis.duration === undefined, JSON.stringify(mis));
    const us = unsync(Uint8Array.from([1, 0xff, 0, 0xe0, 0xff, 0, 0])).join();
    check('lyrics.tags.unsync', us === '1,255,224,255,0', us);
    // Garbage and truncated tags give what they can, never throw.
    const g = readTags(id3v23([['TIT2', txt('Cut Off Title')], ['TPE1', txt('Artist')]]).subarray(0, 30));
    const junk = readTags(Uint8Array.from({ length: 64 }, (_, i) => (i * 37) & 255));
    check('lyrics.tags.robust', typeof g === 'object' && !g.artist && Object.keys(junk).length === 0, JSON.stringify({ g, junk }));

    // FLAC: STREAMINFO (44100 Hz, 441000 samples = 10 s) + VORBIS_COMMENT.
    const si = new Uint8Array(34);
    const rate = 44100;
    si[10] = (rate >> 12) & 255;
    si[11] = (rate >> 4) & 255;
    si[12] = ((rate & 15) << 4) | (1 << 1);
    si.set(be32(441000), 14);
    const vc = vorbis(['TITLE=Flac Song', 'artist=Flac Artist', 'ALBUM=FA']);
    const flac = bytes('fLaC', [0, 0, 0, 34], si, [0x84, (vc.length >> 16) & 255, (vc.length >> 8) & 255, vc.length & 255], vc);
    const f = readTags(flac);
    check('lyrics.tags.flac', sniff(flac) === 'flac' && f.title === 'Flac Song' && f.artist === 'Flac Artist' && f.album === 'FA' && Math.abs((f.duration ?? 0) - 10) < 1e-9 && readFlac(flac).title === 'Flac Song', JSON.stringify(f));

    // Ogg Vorbis: id header page, then a comment packet split over two pages; and Opus.
    const oggPage = (payload: Uint8Array, seq: number) => bytes('OggS', [0, seq === 0 ? 2 : 0], zeros(8), le32(1), le32(seq), le32(0), [1, payload.length], payload);
    const idh = bytes([1], 'vorbis', zeros(23));
    const com = bytes([3], 'vorbis', vorbis(['TITLE=Ogg Song', 'ARTIST=Ogg Artist']), [1]);
    const half = Math.floor(com.length / 2);
    const ogg = bytes(oggPage(idh, 0), oggPage(com.subarray(0, half), 1), oggPage(com.subarray(half), 2));
    const o = readTags(ogg);
    const opus = readOgg(bytes(oggPage(bytes('OpusHead', zeros(11)), 0), oggPage(bytes('OpusTags', vorbis(['TITLE=Opus Song'])), 1)));
    check('lyrics.tags.ogg', sniff(ogg) === 'ogg' && o.title === 'Ogg Song' && o.artist === 'Ogg Artist' && opus.title === 'Opus Song', JSON.stringify({ o, opus }));

    // MP4: ftyp, mdat, moov { mvhd (scale 1000, 187500 = 187.5 s), udta { meta (full) { hdlr, ilst } } }.
    const mvhd = atom('mvhd', bytes([0, 0, 0, 0], be32(0), be32(0), be32(1000), be32(187500), zeros(80)));
    const ilst = atom('ilst', atom(COPY + 'nam', dataAtom('M4A Song')), atom(COPY + 'ART', dataAtom('M4A Artist')), atom(COPY + 'alb', dataAtom('M4A Album')));
    const meta = atom('meta', bytes([0, 0, 0, 0]), atom('hdlr', new Uint8Array(25)), ilst);
    const mp4 = bytes(atom('ftyp', bytes('M4A ', [0, 0, 0, 0])), atom('mdat', new Uint8Array(40)), atom('moov', mvhd, atom('udta', meta)));
    const m = readTags(mp4);
    check('lyrics.tags.mp4', sniff(mp4) === 'mp4' && m.title === 'M4A Song' && m.artist === 'M4A Artist' && m.album === 'M4A Album' && m.duration === 187.5 && readMp4(mp4).title === 'M4A Song', JSON.stringify(m));
  }

  // ------------------------------------------------------------ file names
  {
    const cases: [string, string | undefined, string | undefined][] = [
      ['Artist - Title (Official Video).mp3', 'Artist', 'Title'],
      ['Daft Punk - Get Lucky [Lyrics] (feat. Pharrell Williams).mp3', 'Daft Punk', 'Get Lucky'],
      ['01 - The Band - Song Name (Official Music Video) [HD].m4a', 'The Band', 'Song Name'],
      ['Electropop - Garmisch - 01 - Facing the Sea.mp3', 'Garmisch', 'Facing the Sea'],
      ['Artist_Name_-_Some_Song_(Audio).flac', 'Artist Name', 'Some Song'],
      ['Just A Title.mp3', undefined, 'Just A Title'],
      ['Queen - Bohemian Rhapsody (Remastered 2011).mp3', 'Queen', 'Bohemian Rhapsody'],
      ['Artist - Song (Club Remix) (Official Audio).mp3', 'Artist', 'Song (Club Remix)'],
    ];
    const bad = cases.filter(([n, a, t]) => {
      const mm = metaFromFilename(n);
      return mm.artist !== a || mm.title !== t;
    });
    check('lyrics.filename', !bad.length, bad.map(([n]) => `${n} -> ${JSON.stringify(metaFromFilename(n))}`).join(' | ') || `${cases.length} names split and cleaned`);
    const merged = mergeMeta({ title: 'Tag Title (Official Video)' }, 'File Artist - File Title.mp3');
    check('lyrics.filename.merge', merged.title === 'Tag Title' && merged.artist === 'File Artist' && cleanTitle('Song [4K] | Lyrics') === 'Song', JSON.stringify({ merged, c: cleanTitle('Song [4K] | Lyrics') }));
  }

  // ------------------------------------------------------------------ LRC
  {
    const lrc = '[ar:Someone]\n[offset:+500]\n[00:01.00]First line\n[00:04.50][01:00.00]Chorus <00:04.80>word\n[00:08.25]\n[00:20.123]Late line';
    const tr = parseLrc(lrc, 70);
    const L = tr.lines;
    const ok =
      tr.synced && L.length === 4 && L[0].text === 'First line' && Math.abs(L[0].t - 0.5) < 1e-9 && Math.abs(L[0].end - 4.0) < 1e-9 &&
      L[1].text === 'Chorus word' && Math.abs(L[1].end - 7.75) < 1e-9 && Math.abs(L[2].t - 19.623) < 1e-9 && L[2].text === 'Late line' &&
      Math.abs(L[3].t - 59.5) < 1e-9 && Math.abs(L[3].end - 65.5) < 1e-9;
    check('lyrics.lrc.parse', ok, JSON.stringify(L));
    check('lyrics.lrc.lineAt', lineAt(tr, 0.2) === -1 && lineAt(tr, 1) === 0 && lineAt(tr, 7) === 1 && lineAt(tr, 10) === -1 && lineAt(tr, 60) === 3, [0.2, 1, 7, 10, 60].map((x) => lineAt(tr, x)).join(','));
    const capped = parseLrc('[00:01.00]a\n[01:00.00]b');
    check('lyrics.lrc.max-line', capped.lines[0].end <= 13.01 && parseLrc('no stamps here').lines.length === 0, JSON.stringify(capped.lines));

    // Vocals at 10..30 s and 50..70 s (10 fps): plain lines land inside them only.
    const pres = new Float32Array(900);
    for (let i = 100; i < 300; i++) pres[i] = 0.6;
    for (let i = 500; i < 700; i++) pres[i] = 0.6;
    pres[400] = 0.9; // a blip
    const regs = vocalRegions(pres, 10);
    check('lyrics.plain.regions', regs.length === 2 && Math.abs(regs[0][0] - 10) < 0.5 && Math.abs(regs[1][1] - 70) < 0.5, JSON.stringify(regs));
    const pl = spreadPlain('one two three\nfour five six\n\nseven eight nine\nten eleven twelve', 90, regs);
    const inside = pl.lines.every((l) => regs.some((r) => l.t >= r[0] - 0.01 && l.end <= r[1] + 0.01));
    const ordered = pl.lines.every((l, i) => i === 0 || l.t >= pl.lines[i - 1].t);
    check('lyrics.plain.spread', !pl.synced && pl.lines.length === 4 && inside && ordered && Math.abs(pl.lines[2].t - 50) < 1, JSON.stringify(pl.lines.map((l) => [l.t.toFixed(1), l.end.toFixed(1)])));
    const none = spreadPlain('a\nb', 100, []);
    check('lyrics.plain.no-vocals', none.lines[0].t >= 5 && none.lines[1].end <= 95.01, JSON.stringify(none.lines));
  }

  // --------------------------------------------------------- LRCLIB (mocked)
  {
    const calls: string[] = [];
    const rec = (o: Record<string, unknown>) => ({ id: 1, trackName: 'Song', artistName: 'Band', albumName: 'LP', duration: 200, instrumental: false, plainLyrics: 'a\nb', syncedLyrics: '[00:01.00]a\n[00:02.00]b', ...o });
    let mode: 'get' | 'search' | 'down' | 'busy' | 'instr' | 'none' = 'get';
    const json = (b: unknown) => new Response(JSON.stringify(b), { status: 200, headers: { 'content-type': 'application/json' } });
    const mock = (async (url: string | URL, init?: RequestInit) => {
      const u = String(url);
      calls.push(u);
      const hdr = (init?.headers as Record<string, string> | undefined)?.['Lrclib-Client'];
      if (!hdr) return new Response('no client header', { status: 400 });
      if (mode === 'down') throw new TypeError('Failed to fetch');
      if (mode === 'busy') return new Response('{"statusCode":503}', { status: 503 });
      if (u.includes('/api/get')) {
        if (mode === 'get') return json(rec({}));
        if (mode === 'instr') return json(rec({ instrumental: true, plainLyrics: null, syncedLyrics: null }));
        return new Response('{"statusCode":404}', { status: 404 });
      }
      if (u.includes('/api/search') && mode === 'search')
        return json([
          rec({ trackName: 'Other Song', artistName: 'Else', syncedLyrics: '[00:01.00]wrong' }),
          rec({ trackName: 'Song (Live)', duration: 260, syncedLyrics: null }),
          rec({ trackName: 'Song', duration: 201, syncedLyrics: '[00:03.00]right' }),
        ]);
      return json([]);
    }) as typeof fetch;
    const cache = memoryCache();
    const meta = { artist: 'Band', title: 'Song', album: 'LP', duration: 200.4 };
    const r1 = await lookupLyrics(meta, { fetch: mock, cache });
    const u1 = calls[0] ?? '';
    check('lyrics.lrclib.get', r1.status === 'synced' && !!r1.synced && r1.plain === 'a\nb' && u1.includes('/api/get?artist_name=Band&track_name=Song&album_name=LP&duration=200'), `${r1.status} ${u1}`);
    calls.length = 0;
    const r1b = await lookupLyrics(meta, { fetch: mock, cache });
    check('lyrics.lrclib.cached', r1b.status === 'synced' && calls.length === 0 && cache.map.has(cacheKey(meta)), `${calls.length} calls on the second lookup`);

    mode = 'search';
    const r2 = await lookupLyrics({ artist: 'Band', title: 'Song', duration: 200 }, { fetch: mock, cache: memoryCache() });
    check('lyrics.lrclib.search-fallback', r2.status === 'synced' && r2.synced === '[00:03.00]right', `${r2.status} ${r2.synced}`);

    mode = 'none';
    const c3 = memoryCache();
    const r3 = await lookupLyrics({ artist: 'Nobody', title: 'Nothing' }, { fetch: mock, cache: c3, now: 1000 });
    calls.length = 0;
    const r3b = await lookupLyrics({ artist: 'Nobody', title: 'Nothing' }, { fetch: mock, cache: c3, now: 2000 });
    const cachedCalls = calls.length;
    await lookupLyrics({ artist: 'Nobody', title: 'Nothing' }, { fetch: mock, cache: c3, now: 1000 + MISSING_TTL_MS + 1 });
    check('lyrics.lrclib.missing-ttl', r3.status === 'missing' && r3b.status === 'missing' && cachedCalls === 0 && calls.length > 0, `${r3.status}, ${cachedCalls} calls within TTL, ${calls.length} after`);

    mode = 'instr';
    const r4 = await lookupLyrics({ artist: 'Band', title: 'Song' }, { fetch: mock, cache: memoryCache() });
    check('lyrics.lrclib.instrumental', r4.status === 'instrumental', r4.status);

    for (const md of ['down', 'busy'] as const) {
      mode = md;
      const c = memoryCache();
      calls.length = 0;
      const r = await lookupLyrics({ artist: 'Band', title: 'Song' }, { fetch: mock, cache: c, retryDelays: [1] });
      check(`lyrics.lrclib.${md}`, r.status === 'offline' && c.map.size === 0 && calls.length === 6, `${r.status}, cached ${c.map.size}, ${calls.length} requests (3 steps x 2 tries)`);
    }
    // A busy server that recovers: the retry finds the lyrics.
    let busyOnce = true;
    const flaky = (async (url: string | URL, init?: RequestInit) => {
      if (busyOnce) {
        busyOnce = false;
        return new Response('{"statusCode":503}', { status: 503 });
      }
      mode = 'get';
      return mock(url, init);
    }) as typeof fetch;
    const rf = await lookupLyrics({ artist: 'Band', title: 'Song' }, { fetch: flaky, cache: memoryCache(), retryDelays: [1] });
    check('lyrics.lrclib.retry', rf.status === 'synced', rf.status);
    mode = 'search';
    // Without an artist, a result needs a close duration to be trusted.
    const r5 = await lookupLyrics({ title: 'Song' }, { fetch: mock, cache: memoryCache() });
    const r6 = await lookupLyrics({ title: 'Song', duration: 201 }, { fetch: mock, cache: memoryCache() });
    const r7 = await lookupLyrics({}, { fetch: mock, cache: memoryCache() });
    check('lyrics.lrclib.title-only', r5.status === 'missing' && r6.status === 'synced' && r7.status === 'missing', `${r5.status} ${r6.status} ${r7.status}`);
  }

  // ------------------------------------------- LRCLIB: the song's version (duration)
  {
    // A mocked LRCLIB with three versions of one song: the album (209 s, lyrics from 7 s), a music
    // video (235 s, lyrics from 22 s) and a radio edit (180 s). /api/get honours the duration the
    // way LRCLIB does (a record within 2 s, else 404); without one it returns the first record, which
    // here has plain lyrics only.
    const recs = [
      { id: 1, trackName: 'Song', artistName: 'Band', duration: 209, plainLyrics: 'a', syncedLyrics: null },
      { id: 2, trackName: 'Song', artistName: 'Band', duration: 209, plainLyrics: 'a', syncedLyrics: '[00:07.00]album' },
      { id: 3, trackName: 'Song', artistName: 'Band', duration: 235, plainLyrics: 'a', syncedLyrics: '[00:22.00]video' },
      { id: 4, trackName: 'Song', artistName: 'Band', duration: 180, plainLyrics: 'a', syncedLyrics: '[00:05.00]radio' },
    ];
    const calls: string[] = [];
    const json = (b: unknown) => new Response(JSON.stringify(b), { status: 200, headers: { 'content-type': 'application/json' } });
    let only: number[] | null = null;
    const mock = (async (url: string | URL) => {
      const u = new URL(String(url));
      calls.push(u.pathname + u.search);
      const list = recs.filter((r) => !only || only.includes(r.id));
      if (u.pathname === '/api/get') {
        const d = u.searchParams.get('duration');
        const r = d ? list.find((x) => Math.abs(x.duration - Number(d)) <= 2) : list[0];
        return r ? json(r) : new Response('{"statusCode":404}', { status: 404 });
      }
      return json(list);
    }) as typeof fetch;
    const video = await lookupLyrics({ artist: 'Band', title: 'Song', duration: 234.8 }, { fetch: mock, cache: memoryCache() });
    check('lyrics.lrclib.duration-get', video.synced === '[00:22.00]video' && video.duration === 235 && calls[0].includes('duration=235'), `${video.synced} ${video.duration} ${calls[0]}`);
    // Search scoring: with /api/get missing the version, the closest duration wins among equal matches.
    only = [2, 3, 4];
    calls.length = 0;
    const byScore = await lookupLyrics({ artist: 'Band', title: 'Song', duration: 181 }, { fetch: mock, cache: memoryCache() });
    only = [2, 4];
    const near = await lookupLyrics({ artist: 'Band', title: 'Song', duration: 200 }, { fetch: mock, cache: memoryCache() });
    only = null;
    check('lyrics.lrclib.duration-search', byScore.synced === '[00:05.00]radio' && near.synced === '[00:07.00]album', `181 s -> ${byScore.synced}, 200 s -> ${near.synced}`);
    // Without a duration /api/get gives the plain-only record; the search finds a synced one.
    const noDur = await lookupLyrics({ artist: 'Band', title: 'Song' }, { fetch: mock, cache: memoryCache() });
    check('lyrics.lrclib.plain-then-synced', noDur.status === 'synced', `${noDur.status} ${noDur.synced}`);

    // The cache: an album result is looked up again for the 235 s video, once.
    const cache = memoryCache();
    only = [2];
    const first = await lookupLyrics({ artist: 'Band', title: 'Song' }, { fetch: mock, cache });
    only = null;
    calls.length = 0;
    const again = await lookupLyrics({ artist: 'Band', title: 'Song', duration: 234.8 }, { fetch: mock, cache });
    const requery = calls.length;
    calls.length = 0;
    const third = await lookupLyrics({ artist: 'Band', title: 'Song', duration: 234.8 }, { fetch: mock, cache });
    check('lyrics.lrclib.duration-recheck', first.synced === '[00:07.00]album' && again.synced === '[00:22.00]video' && requery > 0 && third.synced === again.synced && calls.length === 0,
      `${first.synced} -> ${again.synced} (${requery} requests), then ${calls.length} requests`);
    // Nothing closer exists: the lookup for this duration is remembered and not repeated.
    const c2 = memoryCache();
    only = [2];
    await lookupLyrics({ artist: 'Band', title: 'Song', duration: 234.8 }, { fetch: mock, cache: c2 });
    calls.length = 0;
    const kept = await lookupLyrics({ artist: 'Band', title: 'Song', duration: 235.3 }, { fetch: mock, cache: c2 });
    only = null;
    const entry = c2.map.get(cacheKey({ artist: 'Band', title: 'Song' }))!;
    check('lyrics.lrclib.duration-best-available', kept.synced === '[00:07.00]album' && calls.length === 0 && entry.forDuration === 235 && fitsDuration(entry, 210) && !fitsDuration(entry, 300),
      `${kept.synced}, ${calls.length} requests, for ${entry.forDuration}`);

    // The library: a song added without tags is looked up by name, then again once decoded.
    const lib = new LyricsLibrary({ fetch: mock, cache: memoryCache() });
    only = [2, 3];
    calls.length = 0;
    const done = () => new Promise<void>((res) => (lib.onChange = () => res()));
    let wait = done();
    lib.add('t1', Object.assign(new Blob([new Uint8Array(64)]), { name: 'Band - Song (Official Video).mp3' }));
    await wait;
    const before = lib.get('t1')?.result?.synced;
    wait = done();
    lib.setDuration('t1', 234.8);
    await wait;
    const after = lib.get('t1')?.result?.synced;
    const n = calls.length;
    lib.setDuration('t1', 234.9);
    only = null;
    check('lyrics.library.decoded-duration', before === '[00:07.00]album' && after === '[00:22.00]video' && calls.some((c) => c.includes('duration=235')) && calls.length === n,
      `${before} -> ${after}; ${calls.join(' ')}`);
  }

  // --------------------------------------------- alignment of synced lyrics to the audio
  {
    const FR = 86.1328125;
    const DUR = 200;
    const rng = mulberry32(7);
    // An LRC: three blocks of lines 3..4.5 s apart with instrumental gaps between them.
    const lrcLines: string[] = [];
    let t = 6;
    for (const [count, gap] of [[10, 18], [12, 22], [8, 0]]) {
      for (let i = 0; i < count; i++) {
        const s = Math.floor(t);
        lrcLines.push(`[${String(Math.floor(s / 60)).padStart(2, '0')}:${String(s % 60).padStart(2, '0')}.${String(Math.round((t - s) * 100)).padStart(2, '0')}]line ${lrcLines.length}`);
        t += 3 + rng() * 1.5;
      }
      t += gap;
    }
    const track = parseLrc(lrcLines.join('\n'), DUR);
    // Vocal onsets for the song as sung at t * scale + offset: a strong onset at each line start and
    // weaker syllables through the line, over noise and off-beat instrument onsets.
    const song = (offset: number, scale: number, voice = true) => {
      const r = mulberry32(11);
      const o = new Float32Array(Math.floor(DUR * FR));
      for (let i = 0; i < o.length; i++) o[i] = r() * 0.08;
      const hit = (sec: number, a: number) => {
        const i = Math.round(sec * FR);
        for (let k = 0; k < 4; k++) if (i + k >= 0 && i + k < o.length) o[i + k] = Math.max(o[i + k], a * (1 - k / 4));
      };
      for (let x = 0.3; x < DUR; x += 0.5 + r()) hit(x, 0.35 + r() * 0.3);
      if (voice)
        for (const l of track.lines) {
          const a = l.t * scale + offset;
          hit(a + 0.03, 0.9);
          for (let y = a + 0.35; y < Math.min(l.end * scale + offset - 0.3, a + 3); y += 0.3 + r() * 0.3) hit(y, 0.3 + r() * 0.25);
        }
      return { onsets: o, frameRate: FR, duration: DUR };
    };
    const a1 = alignLines(track.lines, song(5.3, 1));
    check('lyrics.align.offset', a1.applied && Math.abs(a1.offset - 5.3) <= 0.1 && a1.scale === 1 && a1.confidence >= 0.8, JSON.stringify(a1));
    const a2 = alignLines(track.lines, song(-3.4, 1));
    check('lyrics.align.negative-offset', a2.applied && Math.abs(a2.offset + 3.4) <= 0.1 && a2.scale === 1, JSON.stringify(a2));
    // A release running 2% slower: the speed is found and every line lands within 0.1 s.
    const f3 = song(-2, 1.02);
    const a3 = alignLines(track.lines, f3);
    const moved = shiftTrack(track, a3.offset, a3.scale, DUR);
    const worst = Math.max(...track.lines.map((l, i) => Math.abs(moved.lines[i].t - (l.t * 1.02 - 2))));
    check('lyrics.align.stretch', a3.applied && Math.abs(a3.scale - 1.02) < 0.003 && worst <= 0.1, `${JSON.stringify(a3)}, worst line ${worst.toFixed(3)} s`);
    // Already in time: nothing to move.
    const a4 = alignLines(track.lines, song(0, 1));
    check('lyrics.align.in-time', !a4.applied && Math.abs(a4.offset) < 0.1 && a4.scale === 1, JSON.stringify(a4));
    // No voice in the onsets (instruments only): not confident, left as timed.
    const a5 = alignLines(track.lines, song(4, 1, false));
    check('lyrics.align.low-confidence', !a5.applied && a5.confidence < 0.5, JSON.stringify(a5));
    const silent = alignLines(track.lines, { onsets: new Float32Array(1000), frameRate: FR, duration: DUR });
    check('lyrics.align.no-onsets', !silent.applied && silent.confidence === 0 && alignLines(track.lines.slice(0, 3), song(5, 1)).confidence === 0, JSON.stringify(silent));
    const fit0 = lineOnsetFit(track.lines, song(5.3, 1));
    const fit1 = lineOnsetFit(shiftTrack(track, a1.offset).lines, song(5.3, 1));
    check('lyrics.align.fit-measure', fit1.strength > fit0.strength + 0.25 && fit1.residual <= 0.15 && fit0.residual >= 1, `strength ${fit0.strength.toFixed(2)} -> ${fit1.strength.toFixed(2)}, residual ${fit0.residual.toFixed(2)} -> ${fit1.residual.toFixed(2)} s`);
    // Shifting clips to the song and drops lines pushed before its start.
    const cut = track.lines[1].end + 0.1;
    const early = shiftTrack(track, -cut, 1, DUR);
    const late = shiftTrack(track, 0, 1.5, DUR);
    check('lyrics.align.shift-clip', early.lines.length === track.lines.length - 2 && early.lines[0].t === 0 && early.lines[0].text === track.lines[2].text && late.lines.every((l) => l.end <= DUR) && late.lines.length < track.lines.length && shiftTrack(track, 0, 1) === track,
      `${early.lines.length} of ${track.lines.length} after -${cut.toFixed(2)} s, first at ${early.lines[0].t}; ${late.lines.length} left at x1.5`);

    // The library aligns synced lyrics against the analysis' vocal onsets.
    const lib = new LyricsLibrary({ fetch: (async () => new Response(JSON.stringify({ trackName: 'Song', artistName: 'Band', duration: 200, syncedLyrics: lrcLines.join('\n') }), { status: 200 })) as typeof fetch, cache: memoryCache() });
    await new Promise<void>((res) => {
      lib.onChange = () => res();
      lib.add('a', Object.assign(new Blob([new Uint8Array(16)]), { name: 'Band - Song.mp3' }));
    });
    const fr = song(5.3, 1);
    const result = { duration: DUR, frameRate: FR, stemOnsets: { vocals: fr.onsets }, stemPresence: {} } as unknown as import('../src/types').AnalysisResult;
    const lt = lib.lyricTrack('a', result);
    check('lyrics.align.library', !!lt?.align?.applied && Math.abs(lt.lines[0].t - (track.lines[0].t + 5.3)) <= 0.1 && lib.lyricTrack('a', result) === lt, JSON.stringify(lt?.align));
  }

  // --------------------------------------------------------------- lexicon
  {
    const cases: [string, string[]][] = [
      ['We are burning in the fire tonight', ['fire', 'night']],
      ['Drowning in the ocean of your tears', ['water']],
      ['Rising up above the clouds', ['rise', 'sky']],
      ['Falling down, breaking to the ground', ['fall']],
      ['Racing down the highway, faster', ['speed', 'city']],
      ['Frozen in the winter snow', ['cold']],
      ['Diamonds and gold, a golden crown', ['gold']],
      ['Stars in the galaxy, lost in space', ['space']],
      ['Thunder and lightning, the storm is here', ['storm']],
      ["Dreamin' of you", ['dream']],
    ];
    const bad = cases.filter(([line, want]) => {
      const got = topTags(readLine(line), 3, 0.3);
      return !want.every((w) => got.includes(w as never));
    });
    check('lyrics.lexicon.tags', !bad.length, bad.map(([l]) => `${l} -> ${topTags(readLine(l)).join(',')}`).join(' | ') || `${cases.length} lines tagged as expected`);
    const happy = readLine('I feel so happy, dancing in the sun with a smile');
    const sad = readLine('I cry alone, broken and lost in the pain');
    const notHappy = readLine('I am not happy');
    const calm = readLine('Sleep now, slow and quiet, gentle rest');
    const wild = readLine('RUN! Fight the fire, scream and shout!');
    check('lyrics.lexicon.valence-arousal', happy.valence > 0.75 && sad.valence < 0.25 && notHappy.valence < 0.5 && calm.arousal < 0.25 && wild.arousal > 0.8,
      `happy ${happy.valence.toFixed(2)} sad ${sad.valence.toFixed(2)} not-happy ${notHappy.valence.toFixed(2)} calm ${calm.arousal.toFixed(2)} wild ${wild.arousal.toFixed(2)}`);
    const neutral = readLine('la la la hmm');
    const bounded = [happy, sad, calm, wild, neutral].every((m) => m.tags.length === TAG_COUNT && m.tags.every((x) => x >= 0 && x <= 1) && m.valence >= 0 && m.valence <= 1 && m.arousal >= 0 && m.arousal <= 1);
    check('lyrics.lexicon.neutral+bounded', neutral.valence === 0.5 && neutral.arousal === 0.5 && neutral.weight === 0 && neutral.tags.every((x) => x === 0) && bounded && LYRIC_TAGS.length === 17,
      JSON.stringify({ v: neutral.valence, a: neutral.arousal, w: neutral.weight }));
    check('lyrics.lexicon.stemming', !!lookupWord('flames') && !!lookupWord("burnin'") && !!lookupWord('skies') && !!lookupWord('Oceans,') && !lookupWord('xylophonic'), 'plural, -in\', -ies, punctuation folded');
  }

  // --------------------------------------------------------------- sampler
  {
    const track = parseLrc('[00:02.00]Burning fire in the night\n[00:06.00]Burning, burning, fire\n[00:10.00]\n[00:30.00]Ocean waves and rain\n[00:34.00]Cold water, deep sea', 40);
    const ls = new LyricSampler(track);
    const fire = tagIndex('fire');
    const water = tagIndex('water');
    const f0 = { ...ls.sample(1, 1 / 60) };
    let pulses = 0;
    let maxFire = 0;
    let t = 1;
    for (; t < 12; t += 1 / 60) {
      const f = ls.sample(t, 1 / 60);
      if (f.pulse === 1) pulses++;
      maxFire = Math.max(maxFire, f.tags[fire]);
    }
    const mid = ls.sample(t, 1 / 60);
    const between = { presence: mid.presence, line: mid.text, next: mid.next, fire: mid.tags[fire] };
    for (; t < 29; t += 1 / 60) ls.sample(t, 1 / 60);
    const late = ls.sample(29, 1 / 60);
    const lateP = late.presence;
    for (t = 29; t < 38; t += 1 / 60) ls.sample(t, 1 / 60);
    const w = ls.sample(38, 1 / 60);
    check('lyrics.sampler.lines', f0.index === -1 && f0.next === 'Burning fire in the night' && pulses === 2 && between.line === '' && between.next === 'Ocean waves and rain' && w.index === 3,
      JSON.stringify({ f0: f0.index, pulses, between, w: w.index }));
    check('lyrics.sampler.meaning', maxFire > 0.4 && w.tags[water] > 0.4 && w.tags[fire] < 0.2 && between.presence > 0.3 && lateP < 0.05,
      `fire peak ${maxFire.toFixed(2)}, water ${w.tags[water].toFixed(2)}, fire later ${w.tags[fire].toFixed(2)}, presence between ${between.presence.toFixed(2)} late ${lateP.toFixed(2)}`);
    // A seek snaps the meaning to the new line at once.
    ls.reset();
    const s = ls.sample(31, 1 / 60);
    check('lyrics.sampler.seek', s.index === 2 && s.tags[water] > 0.4 && s.progress > 0.2 && s.progress < 0.3, `water ${s.tags[water].toFixed(2)}, progress ${s.progress.toFixed(2)}`);
    const state = {} as Parameters<LyricSampler['apply']>[0];
    (state as { time: number }).time = 7;
    ls.apply(state, 1 / 60);
    check('lyrics.sampler.state', state.lyricLine === 'Burning, burning, fire' && state.lyricSynced === true && state.lyricTags!.length === TAG_COUNT && typeof state.lyricValence === 'number', JSON.stringify({ l: state.lyricLine, p: state.lyricProgress }));
  }

  // ------------------------------------------------------------ the gene
  {
    const def = repairLyrics({});
    const bad = repairLyrics({ p: { strength: 7, pal: 0.4, show: 1.7, lag: -1, smear: NaN, junk: 3 } });
    check('lyrics.gene.repair', !validateLyrics(def).length && bad.p.strength === 1 && bad.p.pal === 0 && bad.p.show === 2 && bad.p.lag === LYRICS_SCHEMA.lag.min && bad.p.smear === 0 && !('junk' in bad.p) && !validateLyrics(bad).length,
      JSON.stringify(bad.p));
    const spec = genomeGene('lyrics');
    check('lyrics.gene.registered', !!spec && spec.optional && !!spec.glossary && Object.keys(TAG_EFFECTS).join() === LYRIC_TAGS.join(), spec?.title ?? 'missing');
    const base = SEEDS[0].genome;
    const withG = repair({ ...cloneGenome(base), lyrics: { p: { strength: 0.8, show: 2, smear: 0.5 } } });
    const broken = { ...cloneGenome(withG), lyrics: { p: { ...withG.lyrics!.p, strength: 3 } } } as Genome;
    check('lyrics.gene.genome', !validate(withG).length && withG.lyrics!.p.show === 2 && validate(broken).some((e) => e.startsWith('lyrics.')) && !repair(cloneGenome(base)).lyrics,
      validate(withG).join(',') || 'kept through repair, validated');
    const dc = estimateCost(withG) - estimateCost(base);
    check('lyrics.gene.cost', Math.abs(dc - lyricsCost(withG.lyrics!.p)) < 1e-9 && lyricsCost({ ...def.p, smear: 0 }) === 0.01 && lyricsCost({ ...def.p, smear: 0.4 }) > 0.01, `+${dc.toFixed(3)} ms`);
    // Crossover: both parents blend; one parent passes it on about half the time; none draws no rng.
    const rng = mulberry32(7);
    let carried = 0;
    let validAll = true;
    for (let i = 0; i < 200; i++) {
      const c = crossLyrics(def, undefined, rng);
      if (c) carried++;
      const both = crossLyrics(randomLyrics(rng), randomLyrics(rng), rng)!;
      if (validateLyrics(both).length) validAll = false;
    }
    let draws = 0;
    const counting = () => (draws++, 0.5);
    const none = crossLyrics(undefined, undefined, counting);
    check('lyrics.gene.crossover', carried > 60 && carried < 140 && validAll && none === undefined && draws === 0, `${carried}/200 carried from one parent, ${draws} draws with none`);
    const other = SEEDS[5].genome;
    let kids = 0;
    let kidsWith = 0;
    let probs: string[] = [];
    const r2 = mulberry32(11);
    for (let i = 0; i < 40; i++) {
      const c = crossover(withG, other, r2);
      kids++;
      if (c.lyrics) kidsWith++;
      const v = validate(c);
      if (v.length) probs.push(v[0]);
    }
    check('lyrics.gene.breeds', !probs.length && kidsWith > 5 && kidsWith < kids, probs[0] ?? `${kidsWith}/${kids} children inherit it`);
    const r3 = mulberry32(5);
    let gained = 0;
    let mutProbs = 0;
    for (let i = 0; i < 400; i++) {
      const m = mutate(base, r3, 1);
      if (m.lyrics) gained++;
      if (validate(m).length) mutProbs++;
      const m2 = mutate(withG, r3, 2);
      if (validate(m2).length) mutProbs++;
    }
    check('lyrics.gene.mutation', MUTATION_NAMES.includes('lyrics') && MUTATION_NAMES.includes('jitter-lyrics') && gained > 0 && gained < 60 && !mutProbs, `${gained}/400 mutations gained it, ${mutProbs} invalid`);
    const lyrical = SEEDS.slice(0, 24).filter((x) => ADJ_POOLS.lyrical.includes(nameFor(repair({ ...cloneGenome(x.genome), lyrics: { p: { strength: 1, show: 2, smear: 0.6 } } })).split(' ')[0])).length;
    check('lyrics.gene.name', lyrical >= 3, `${lyrical}/24 originals named lyrical with a strong lyrics gene`);

    // Nudges: neutral without words, bounded with any words, and they lean the right way.
    const tags = (o: Partial<Record<(typeof LYRIC_TAGS)[number], number>>) => Float32Array.from(LYRIC_TAGS.map((t) => o[t] ?? 0));
    const w = (o: Partial<Record<(typeof LYRIC_TAGS)[number], number>>, extra: Partial<{ valence: number; arousal: number; presence: number }> = {}) => ({ tags: tags(o), valence: 0.5, arousal: 0.5, presence: 1, pulse: 0, ...extra });
    const same = (n: LyricNudge) => (Object.keys(NEUTRAL_NUDGE) as (keyof LyricNudge)[]).every((k) => Math.abs(n[k] - NEUTRAL_NUDGE[k]) < 1e-9);
    const g1 = repairLyrics({ p: { strength: 1, chain: 1 } });
    check('lyrics.nudge.neutral', same(lyricTarget(g1, { valence: 0.9, arousal: 0.9, presence: 1, pulse: 1 }, 0.3, 1)) && same(lyricTarget(g1, w({ fire: 1 }, { presence: 0 }), 0.3, 1)) && same(lyricTarget(undefined, w({ fire: 1 }), 0.3, 1)) && same(lyricTarget(repairLyrics({ p: { strength: 0 } }), w({ fire: 1 }), 0.3, 1)),
      'no lyrics, no presence, no gene, zero strength: identity');
    const fire = lyricTarget(g1, w({ fire: 0.9 }), 0.5, 1);
    const water = lyricTarget(g1, w({ water: 0.9 }), 0.2, 1);
    const night = lyricTarget(g1, w({ night: 0.9, dark: 0.6 }), 0.5, 1);
    const rise = lyricTarget(g1, w({ rise: 0.9 }), 0.5, 1);
    const fall = lyricTarget(g1, w({ fall: 0.9 }), 0.5, 1);
    const fast = lyricTarget(g1, w({ speed: 0.9 }, { arousal: 0.9 }), 0.5, 1);
    const calm = lyricTarget(g1, w({ dream: 0.9 }, { arousal: 0.1 }), 0.5, 1);
    const hue = (base: number, n: LyricNudge) => (((base + n.hue) % 1) + 1) % 1;
    check('lyrics.nudge.direction',
      hue(0.5, fire) < 0.5 && fire.lift > 0 && water.hue > 0 && water.ripple > 0 && water.water > 0 && night.exposure < 0.9 && rise.lift > 0 && rise.zoom < fall.zoom && fall.lift < 0 && fast.speed > 1.3 && calm.speed < 0.85 && calm.blur > 0,
      JSON.stringify({ fireHue: hue(0.5, fire).toFixed(2), waterHue: hue(0.2, water).toFixed(2), nightExp: night.exposure.toFixed(2), riseZoom: rise.zoom.toFixed(3), fallZoom: fall.zoom.toFixed(3), fast: fast.speed.toFixed(2), calm: calm.speed.toFixed(2) }));
    const rr = mulberry32(99);
    let outOfBounds = 0;
    for (let i = 0; i < 500; i++) {
      const tg = Float32Array.from(LYRIC_TAGS.map(() => (rr() < 0.4 ? rr() : 0)));
      const n = lyricTarget(randomLyrics(rr), { tags: tg, valence: rr(), arousal: rr(), presence: rr(), pulse: rr() }, rr(), rr() * 100);
      if (Math.abs(n.hue) > 0.5 || n.sat < 0.65 || n.sat > 1.3 || n.exposure < 0.7 || n.exposure > 1.25 || n.zoom < 1 || n.zoom > 1.2 || Math.abs(n.lift) > 0.035 || Math.abs(n.roll) > 0.03 || n.speed < 0.6 || n.speed > 1.6 || [n.ripple, n.swirl, n.noise, n.water, n.blur].some((x) => x < 0 || x > 1)) outOfBounds++;
    }
    const off = repairLyrics({ p: { strength: 1, pal: 0, tone: 0, motion: 0, chain: 0 } });
    check('lyrics.nudge.bounded+groups', !outOfBounds && same(lyricTarget(off, w({ fire: 1, storm: 1 }, { arousal: 1 }), 0.2, 3)), `${outOfBounds}/500 random nudges out of bounds; all groups off = identity`);
    // Easing follows over the lag and never overshoots; the kick is separate and decays with the pulse.
    const cur = { ...NEUTRAL_NUDGE };
    for (let i = 0; i < 90; i++) easeNudge(cur, fast, 1 / 60, 1.5);
    const half = cur.speed;
    for (let i = 0; i < 900; i++) easeNudge(cur, fast, 1 / 60, 1.5);
    check('lyrics.nudge.ease+kick', half > 1 && half < fast.speed && Math.abs(cur.speed - fast.speed) < 1e-3 && lineKick(g1, 1) > 1 && lineKick(g1, 0) === 1 && lineKick(undefined, 1) === 1, `after 1.5 s ${half.toFixed(3)} of ${fast.speed.toFixed(3)}; kick ${lineKick(g1, 1).toFixed(3)}`);
    // The nudges never touch the genome: computing them leaves it byte-identical.
    const before = JSON.stringify(withG);
    for (let i = 0; i < 20; i++) lyricTarget(withG.lyrics, w({ fire: 1, water: 1 }), 0.4, i);
    check('lyrics.nudge.not-saved', JSON.stringify(withG) === before, 'genome unchanged');
  }

  // --------------------------------------------------------------- seeds
  {
    const ys = SEEDS.filter((x) => /^Y\d\d$/.test(x.origin));
    const bad = ys.filter((x) => !x.genome.lyrics || validate(x.genome).length || estimateCost(x.genome) >= COST_BUDGET_MS || !x.genome.reactions.some((r) => ['line', 'valence', 'arousal'].includes(r.src)));
    const shows = ys.map((x) => x.genome.lyrics?.p.show).join(',');
    check('lyrics.seeds', ys.map((x) => x.origin).join() === 'Y01,Y02' && !bad.length && shows === '2,1' && ys.every((x) => (x.genome.lyrics?.p.smear ?? 0) > 0 && x.genome.carrier.kind !== 'none'),
      `${ys.map((x) => `${x.origin} ${x.name} ${estimateCost(x.genome).toFixed(2)} ms`).join('; ')}${bad.length ? ' bad: ' + bad.map((x) => x.origin + ':' + validate(x.genome)[0]).join(',') : ''}`);
  }
}
