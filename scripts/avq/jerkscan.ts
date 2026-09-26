// Static scan of the seeds for whole-frame beat motion: reactions whose signal is percussive and
// whose target moves the whole picture (a chain op's geometry, the carrier's advection), the camera
// parts of the choreo and accent genes, and landscape / tunnel camera kicks.
//
//   node --import ./scripts/analysis-test.hooks.mjs scripts/avq/jerkscan.ts [--preset ID,...]

import { SEEDS } from '../../src/v2/seeds';
import { accentPlan } from '../../src/v2/genes/accent';
import type { Genome } from '../../src/v2/genome';
import { parseArgs } from './render';

/** Signals that pulse on drums, beats or drops. */
export const PERC = new Set(['hit', 'beat', 'drums', 'surge', 'barpulse', 'bar', 'drop', 'attack', 'hook']);
/** Chain-op params that move the picture geometrically (per op kind; '*' = any param of that kind). */
const GEOM: Record<string, string[] | '*'> = {
  zoom: '*', rotate: '*', translate: '*', swirl: '*', twist: '*', ripple: '*', noise: '*', push: '*', quad: '*',
  tunnel: '*', polar: '*', kaleido: ['rot', 'spin', 'turn', 'angle'], tile: ['off', 'scroll', 'shift'], mirror: ['angle', 'rot'],
  stretch: '*', mosaic: [],
};
const CAR = new Set(['vort', 'famt', 'fscale', 'fnoise', 'amount']);

export function scan(g: Genome): string[] {
  const out: string[] = [];
  for (const r of g.reactions) {
    if (!PERC.has(r.src)) continue;
    if (r.g === 'op') {
      const op = g.chain[r.i];
      const kind = op?.op ?? '?';
      const geo = kind.startsWith('v_') ? '*' : GEOM[kind];
      if (geo === '*' || (geo && geo.includes(r.k))) out.push(`${r.src}->op${r.i}.${kind}.${r.k} gain ${r.gain.toFixed(2)} atk ${r.atk.toFixed(3)} rel ${r.rel.toFixed(2)}`);
    } else if (r.g === 'car' && CAR.has(r.k)) out.push(`${r.src}->car.${r.k} gain ${r.gain.toFixed(2)} atk ${r.atk.toFixed(3)} rel ${r.rel.toFixed(2)}`);
  }
  const a = accentPlan(g);
  if (a.kick > 0) out.push(`accent.kick ${a.kick}`);
  if (a.hook > 0) out.push(`accent.hook ${a.hook.toFixed(2)}`);
  if (a.drop > 0) out.push(`accent.drop ${a.drop.toFixed(2)}`);
  if (a.frame > 0) out.push(`accent.frame ${a.frame.toFixed(2)}`);
  const c = g.choreo?.p;
  if (c && c.punch > 0) out.push(`choreo.punch ${c.punch}`);
  for (const b of g.bodies) {
    const s = b.shape as { kind: string; p?: Record<string, number> };
    if (s.kind === 'landscape' && (s.p?.kick ?? 0) > 0) out.push(`landscape.kick ${s.p!.kick}`);
  }
  return out;
}

if (import.meta.url === `file://${process.argv[1]}`) {
  const a = parseArgs(process.argv.slice(2));
  const want = a.preset ? String(a.preset).split(',') : null;
  for (const s of SEEDS) if (!want || want.includes(s.origin)) console.log(`${s.origin.padEnd(4)} ${s.name.padEnd(24)} ${scan(s.genome).join('; ')}`);
}
