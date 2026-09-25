// Review clips (CLI): MP4s of rendered frames at the render frame rate with the song audio
// muxed in, one per clip window -> .testdata/avq/mp4/<preset>/<song>__<label>.mp4
//   node --import ./scripts/analysis-test.hooks.mjs scripts/avq/clips.ts --preset E14 --song "<mp3>" [--clips auto|A-B:label]
//   node --import ./scripts/analysis-test.hooks.mjs scripts/avq/clips.ts --existing [--preset E14] [--song slug]
// Without --existing it renders the windows with --frames first. Needs ffmpeg on PATH
// (Homebrew: brew install ffmpeg). Row i shows song time start + (i + 1) / fps, so the audio
// is cut from start + 1 / fps.

import { execFileSync } from 'node:child_process';
import { existsSync, mkdirSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { ensureServers, OUT } from './cdp';
import { listClips, loadClip } from './load';
import { parseArgs, parseClips, runJobs, testSongs, type Job } from './render';

export function ffmpegPath(): string {
  for (const p of ['/opt/homebrew/bin/ffmpeg', '/usr/local/bin/ffmpeg']) if (existsSync(p)) return p;
  try {
    return execFileSync('which', ['ffmpeg']).toString().trim();
  } catch {
    throw new Error('ffmpeg not found (brew install ffmpeg)');
  }
}

/** Encode one clip's JPEG frames + audio to MP4; returns the output path. */
export function encodeClip(base: string, ffmpeg = ffmpegPath()): string {
  const c = loadClip(base);
  const h = c.header;
  if (!h.framesDir) throw new Error(`${base} has no frames (render with --frames)`);
  const out = join(OUT, 'mp4', base + '.mp4');
  mkdirSync(dirname(out), { recursive: true });
  const fps = h.render.fps;
  const a0 = h.clip.start + 1 / fps;
  const dur = c.n / fps;
  execFileSync(ffmpeg, [
    '-y', '-loglevel', 'error',
    '-framerate', String(fps), '-i', join(OUT, h.framesDir, '%06d.jpg'),
    '-ss', a0.toFixed(3), '-t', dur.toFixed(3), '-i', h.song.path,
    '-map', '0:v', '-map', '1:a',
    '-c:v', 'libx264', '-preset', 'medium', '-crf', '22', '-maxrate', '2500k', '-bufsize', '5000k', '-pix_fmt', 'yuv420p',
    '-vf', 'scale=640:-2:flags=lanczos',
    '-c:a', 'aac', '-b:a', '160k', '-shortest', '-movflags', '+faststart',
    out,
  ]);
  return out;
}

async function main() {
  const a = parseArgs(process.argv.slice(2));
  const ffmpeg = ffmpegPath();
  let bases: string[];
  if (a.existing) {
    bases = listClips({ preset: a.preset as string, song: a.song as string, label: a.label as string }).filter((b) => !!loadClip(b).header.framesDir);
  } else {
    const songs = a.song === 'test' || !a.song ? testSongs() : String(a.song).split(',');
    const presets = String(a.preset ?? 'E14').split(',');
    const servers = await ensureServers();
    const jobs: Job[] = [];
    for (const s of songs) for (const p of presets) jobs.push({ preset: p, song: s, opts: { clips: parseClips(a.clips), frames: true } });
    bases = [];
    try {
      await runJobs(jobs, Number(a.jobs ?? 1), (j, r, err) => {
        if (err) console.log(`FAIL ${j.preset} ${j.song}: ${err.message}`);
        else for (const cl of (r as { clips: { base: string }[] }).clips) bases.push(cl.base);
      });
    } finally {
      if (!a.keep) servers.stop();
    }
  }
  for (const b of bases) console.log(encodeClip(b, ffmpeg));
}

if (import.meta.url === `file://${process.argv[1]}`) await main();
