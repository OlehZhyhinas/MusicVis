// Page-side: runs the app's Screener (src/v2/screen.ts) over seeds, to calibrate its cheap
// reactivity score against the harness report cards.

import type { Engine } from '../../src/v2/engine';
import { Screener, type ScreenMetrics } from '../../src/v2/screen';
import { SEEDS } from '../../src/v2/seeds';

export async function screenSeeds(eng: Engine, ids?: string[]): Promise<{ id: string; ok: boolean; reason?: string; m: Pick<ScreenMetrics, 'events' | 'hitLift' | 'reactivity' | 'motion' | 'beatCorr' | 'msPerFrame'> }[]> {
  const scr = new Screener(eng);
  const out = [];
  for (const s of SEEDS.filter((x) => !ids || ids.includes(x.origin))) {
    let done = false;
    const p = scr.screen(s.genome).then((r) => {
      done = true;
      return r;
    });
    while (!done) {
      scr.runner.pump(8);
      await new Promise<void>((ok) => {
        const ch = new MessageChannel();
        ch.port1.onmessage = () => ok();
        ch.port2.postMessage(0);
      });
    }
    const r = await p;
    const m = r.metrics;
    out.push({ id: s.origin, ok: r.ok, reason: r.reason, m: { events: +m.events.toFixed(3), hitLift: +m.hitLift.toFixed(3), reactivity: +m.reactivity.toFixed(3), motion: +m.motion.toFixed(4), beatCorr: +m.beatCorr.toFixed(3), msPerFrame: +m.msPerFrame.toFixed(2) } });
  }
  return out;
}
