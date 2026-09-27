// New-song detection on the live path: whole songs played back to back (no gap, a 2 s gap, a
// 6 s gap, and a 15 s "ad" between songs, like YouTube autoplay), and the DJ mixes. Emulates
// src/ui/liveMode.ts: a new song is announced when rtStructure.newSongs moves and the analyzer
// has since heard NEW_SONG_MUSIC_S of music. Scores latency, misses and false triggers.

import { decode, LivePath, slugOf, SR, testSongs, type Pcm } from './common';
import type { Mix } from './mix';

/** src/ui/liveMode.ts NEW_SONG_MUSIC_S. */
const NEW_SONG_MUSIC_S = 6;
const TOL = 25;

function song(part: string): { slug: string; pcm: Pcm } {
  const path = testSongs().find((p) => slugOf(p).includes(part));
  if (!path) throw new Error('no test song ' + part);
  return { slug: slugOf(path), pcm: decode(path) };
}

function concat(parts: Pcm[]): Pcm {
  const n = parts.reduce((s, p) => s + p.left.length, 0);
  const left = new Float32Array(n), right = new Float32Array(n);
  let o = 0;
  for (const p of parts) {
    left.set(p.left, o);
    right.set(p.right, o);
    o += p.left.length;
  }
  return { sr: SR, left, right };
}

const silence = (s: number): Pcm => ({ sr: SR, left: new Float32Array(Math.round(s * SR)), right: new Float32Array(Math.round(s * SR)) });

function excerpt(p: Pcm, a: number, b: number, gain = 1): Pcm {
  const i0 = Math.round(a * SR), i1 = Math.min(p.left.length, Math.round(b * SR));
  const l = p.left.slice(i0, i1), r = p.right.slice(i0, i1);
  for (let i = 0; i < l.length; i++) {
    // 0.3 s fades so the excerpt does not click.
    const f = Math.min(1, i / (0.3 * SR), (l.length - i) / (0.3 * SR)) * gain;
    l[i] *= f;
    r[i] *= f;
  }
  return { sr: SR, left: l, right: r };
}

/** Run the live path and return the times liveMode would announce a new song. */
export function detectNewSongs(pcm: Pcm): number[] {
  const lp = new LivePath(pcm);
  const a = lp.analyzer;
  const out: number[] = [];
  let seen = 0, pending = false;
  const dur = pcm.left.length / pcm.sr;
  for (let t = 0; t < dur; t += 1 / 30) {
    lp.feed(t);
    const ns = a.structure.newSongs;
    if (ns !== seen) {
      seen = ns;
      pending = true;
    }
    if (pending && a.musicSeconds >= NEW_SONG_MUSIC_S) {
      pending = false;
      out.push(t);
    }
  }
  return out;
}

function score(truth: number[], det: number[]): { hits: number; lat: number[]; misses: number; falses: number[] } {
  const used = new Set<number>();
  const lat: number[] = [];
  for (const b of truth) {
    const j = det.findIndex((d, k) => !used.has(k) && d >= b - 1 && d <= b + TOL);
    if (j >= 0) {
      used.add(j);
      lat.push(det[j] - b);
    }
  }
  const falses = det.filter((_, k) => !used.has(k));
  return { hits: lat.length, lat, misses: truth.length - lat.length, falses };
}

export function newSongReport(mixes: Mix[]): string {
  const songs = ['inna-morenito', 'lean-on', 'ghosts', 'saxobeat'].map(song);
  const ad = song('thrift-shop');
  const lines: string[] = [];
  lines.push(`New-song detection: ${songs.map((s) => s.slug).join(' > ')} played whole, back to back, as liveMode announces it (rtStructure.newSongs: music after a 5 s silence, or a sustained fingerprint change from rtNewSong.ts; then ${NEW_SONG_MUSIC_S} s of music). Hit = announced within ${TOL} s after the true start.`);
  lines.push('');
  lines.push('| sequence | true song starts | announced | hits | median latency | misses | false (mid-song / mid-blend) |');
  lines.push('|---|---|---|---|---|---|---|');
  const variants: { name: string; gap: () => Pcm[] }[] = [
    { name: 'no gap', gap: () => [] },
    { name: '2 s silence', gap: () => [silence(2)] },
    { name: '6 s silence', gap: () => [silence(6)] },
    { name: '15 s ad (other music at -6 dB) + 0.5 s silences', gap: () => [silence(0.5), excerpt(ad.pcm, 60, 75, 0.5), silence(0.5)] },
  ];
  for (const v of variants) {
    const parts: Pcm[] = [];
    const starts: number[] = [];
    let t = 0;
    songs.forEach((s, i) => {
      if (i > 0) for (const g of v.gap()) {
        parts.push(g);
        t += g.left.length / SR;
      }
      if (i > 0) starts.push(t);
      parts.push(s.pcm);
      t += s.pcm.left.length / SR;
    });
    const det = detectNewSongs(concat(parts));
    const sc = score(starts, det);
    const med = sc.lat.length ? [...sc.lat].sort((p, q) => p - q)[sc.lat.length >> 1] : NaN;
    lines.push(`| ${v.name} | ${starts.map((x) => x.toFixed(0)).join(', ')} s | ${det.map((x) => x.toFixed(0)).join(', ') || 'none'} | ${sc.hits}/${starts.length} | ${Number.isFinite(med) ? med.toFixed(1) + ' s' : 'n/a'} | ${sc.misses} | ${sc.falses.length}${sc.falses.length ? ' at ' + sc.falses.map((x) => x.toFixed(0)).join(', ') + ' s' : ''} |`);
  }
  {
    const falses: string[] = [];
    let n = 0;
    for (const p of testSongs()) {
      const det = detectNewSongs(decode(p));
      n += det.length;
      if (det.length) falses.push(`${slugOf(p)} at ${det.map((x) => x.toFixed(0)).join(', ')} s`);
    }
    lines.push(`| each test song alone (${testSongs().length}) | none | ${n} | | | | ${n}${falses.length ? ': ' + falses.join('; ') : ''} |`);
  }
  for (const m of mixes) {
    const takes = m.transitions.filter((x) => Number.isFinite(x.takeover)).map((x) => x.takeover);
    const det = detectNewSongs(m.pcm);
    const sc = score(takes, det);
    const med = sc.lat.length ? [...sc.lat].sort((p, q) => p - q)[sc.lat.length >> 1] : NaN;
    lines.push(`| DJ ${m.name} (takeovers) | ${takes.map((x) => x.toFixed(0)).join(', ')} s | ${det.map((x) => x.toFixed(0)).join(', ') || 'none'} | ${sc.hits}/${takes.length} | ${Number.isFinite(med) ? med.toFixed(1) + ' s' : 'n/a'} | ${sc.misses} | ${sc.falses.length} |`);
  }
  return lines.join('\n');
}
