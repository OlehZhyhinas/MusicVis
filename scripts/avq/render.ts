// Deterministic song renderer (CLI). Renders presets over real songs at a fixed frame clock
// in headless Chrome and writes per-frame visual + music records to .testdata/avq/clips/.
//
//   node --import ./scripts/analysis-test.hooks.mjs scripts/avq/render.ts \
//     --preset E14 --song "/Users/oleh/Downloads/YoutubeToMp3/Stromae - Alors on danse (Official Video).mp3"
//
// Options:
//   --preset ID[,ID...]|all     seed origin ids (default E14)
//   --song PATH[,PATH...]|test  song files; 'test' = every mp3 in ~/Downloads/YoutubeToMp3
//   --clips auto|song|A-B[:label][,A-B...]   windows in seconds (default auto: drop +-10 s, densest 20 s of the top hook)
//   --w 320 --h 180 --fps 30 --seed 1 --warm 3
//   --frames                    also save every frame as JPEG (for review clips / filmstrips)
//   --embed [N]                 DINOv2-small CLS embedding of every Nth frame (default 5 = 6 per s)
//   --jobs N                    parallel tabs (default 1, max 3)
//   --info                      only analyse the songs and print sections, moments, hooks, windows
//   --cf                        counterfactual lockstep renders instead (half-bar shift, far offset,
//                               each stem muted, each reaction ablated) -> .testdata/avq/cf/
//   --keep                      leave vite/chrome running afterwards

import { readdirSync } from 'node:fs';
import { join } from 'node:path';
import { ensureServers, Tab } from './cdp';

export const TEST_DIR = '/Users/oleh/Downloads/YoutubeToMp3';

export function parseArgs(argv: string[]): Record<string, string | boolean> {
  const a: Record<string, string | boolean> = {};
  for (let i = 0; i < argv.length; i++) {
    const k = argv[i];
    if (!k.startsWith('--')) continue;
    const next = argv[i + 1];
    if (next === undefined || next.startsWith('--')) a[k.slice(2)] = true;
    else {
      a[k.slice(2)] = next;
      i++;
    }
  }
  return a;
}

export function testSongs(): string[] {
  return readdirSync(TEST_DIR).filter((f) => /\.(mp3|m4a|wav|flac|ogg)$/i.test(f)).sort().map((f) => join(TEST_DIR, f));
}

export function parseClips(s: string | boolean | undefined): unknown {
  if (!s || s === true || s === 'auto') return 'auto';
  if (s === 'song') return 'song';
  return String(s).split(',').map((w, i) => {
    const [range, label] = w.split(':');
    const [a, b] = range.split('-').map(Number);
    return { label: label ?? `w${i}`, start: a, end: b };
  });
}

export interface Job {
  preset: string;
  song: string;
  opts: Record<string, unknown>;
  /** Page method: 'render' (default) or 'counterfactual'. */
  method?: string;
}

/** Run jobs over up to `jobs` tabs, one page load per job. */
export async function runJobs(jobs: Job[], parallel: number, onDone: (j: Job, r: unknown, err?: Error) => void): Promise<void> {
  const queue = [...jobs];
  const worker = async () => {
    const tab = await Tab.open();
    try {
      while (queue.length) {
        const j = queue.shift()!;
        try {
          tab.logs.length = 0;
          await tab.load();
          const r = await tab.eval(`avq.${j.method ?? 'render'}(${JSON.stringify({ preset: j.preset, song: j.song, ...j.opts })})`);
          onDone(j, r);
        } catch (e) {
          onDone(j, null, new Error(String((e as Error).message) + (tab.logs.length ? ' | ' + tab.logs.slice(-3).join(' | ') : '')));
        }
      }
    } finally {
      await tab.close();
    }
  };
  await Promise.all(Array.from({ length: Math.max(1, Math.min(3, parallel)) }, worker));
}

async function main() {
  const a = parseArgs(process.argv.slice(2));
  const songs = a.song === 'test' || !a.song ? testSongs() : String(a.song).split(',');
  const servers = await ensureServers();
  try {
    if (a.info) {
      const tab = await Tab.open();
      await tab.load();
      for (const s of songs) console.log(JSON.stringify(await tab.eval(`avq.song(${JSON.stringify(s)})`), null, 1));
      await tab.close();
      return;
    }
    let presets = String(a.preset ?? 'E14').split(',');
    if (presets[0] === 'all') {
      const tab = await Tab.open();
      await tab.load();
      presets = await tab.eval<string[]>('avq.seeds()');
      await tab.close();
    }
    const opts: Record<string, unknown> = { clips: parseClips(a.clips) };
    for (const k of ['w', 'h', 'fps', 'seed', 'warm']) if (a[k] !== undefined) opts[k] = Number(a[k]);
    if (a.embed) opts.embedEvery = a.embed === true ? 5 : Number(a.embed);
    if (a.frames) opts.frames = true;
    const jobs: Job[] = [];
    const method = a.cf ? 'counterfactual' : 'render';
    if (a.cf) delete opts.frames;
    for (const s of songs) for (const p of presets) jobs.push({ preset: p, song: s, opts, method });
    const t0 = Date.now();
    let done = 0;
    await runJobs(jobs, Number(a.jobs ?? 1), (j, r, err) => {
      done++;
      if (err) console.log(`[${done}/${jobs.length}] FAIL ${j.preset} ${j.song.split('/').pop()}: ${err.message}`);
      else console.log(`[${done}/${jobs.length}] ${JSON.stringify(r)}`);
    });
    console.error(`done ${jobs.length} jobs in ${((Date.now() - t0) / 1000).toFixed(1)} s`);
  } finally {
    if (!a.keep) servers.stop();
  }
}

if (import.meta.url === `file://${process.argv[1]}`) await main();
