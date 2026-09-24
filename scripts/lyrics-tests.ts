// Tests for lyrics (src/lyrics/**): tag reader on crafted byte buffers, file name cleaning, LRC
// parsing, plain-lyric spreading, and the LRCLIB lookup against a mocked fetch. Called from v2-test.ts.

import { readTags, readId3v2, readId3v1, readFlac, readOgg, readMp4, sniff, unsync } from '../src/lyrics/tags';
import { cleanTitle, metaFromFilename, mergeMeta } from '../src/lyrics/filename';
import { parseLrc, spreadPlain, vocalRegions, lineAt } from '../src/lyrics/lrc';
import { lookupLyrics, memoryCache, cacheKey, MISSING_TTL_MS } from '../src/lyrics/lrclib';
import { readLine, topTags, lookupWord, LYRIC_TAGS, TAG_COUNT, tagIndex } from '../src/lyrics/lexicon';
import { LyricSampler } from '../src/lyrics/sampler';

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
}
