// Timeline sheets (CLI): computes report cards, then draws one tall PNG per clip in the
// headless page -> .testdata/avq/sheets/<preset>/<song>__<label>.png
//   node --import ./scripts/analysis-test.hooks.mjs scripts/avq/sheets.ts [--preset E14] [--song slug] [--label hook]
// Filmstrips use the clip's JPEG frames when it was rendered with --frames, else its thumbnails.

import { ensureServers, OUT, Tab } from './cdp';
import { listClips } from './load';
import { cardFor } from './report';
import { parseArgs } from './render';

export async function drawSheets(bases: string[], tab: Tab): Promise<string[]> {
  const out: string[] = [];
  for (const b of bases) {
    cardFor(b);
    const r = await tab.eval<{ base: string; h: number; bytes: number }>(`avq.sheet(${JSON.stringify({ base: b, out: OUT })})`);
    out.push(`${OUT}/sheets/${r.base}.png`);
    console.log(`${OUT}/sheets/${r.base}.png  ${r.h}px  ${(r.bytes / 1024).toFixed(0)} KB`);
  }
  return out;
}

async function main() {
  const a = parseArgs(process.argv.slice(2));
  const bases = listClips({ preset: a.preset as string, song: a.song as string, label: a.label as string });
  const servers = await ensureServers();
  const tab = await Tab.open();
  try {
    await tab.load();
    await drawSheets(bases, tab);
  } finally {
    await tab.close();
    if (!a.keep) servers.stop();
  }
}

if (import.meta.url === `file://${process.argv[1]}`) await main();
