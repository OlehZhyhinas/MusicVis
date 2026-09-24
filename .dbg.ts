import { crossoverTagged, mulberry32 } from './src/v2/ops';
import { SEEDS } from './src/v2/seeds';
const rng = mulberry32(1);
const rate: Record<string, [number, number]> = {};
for (const a of SEEDS) for (const b of SEEDS) {
  if (a === b) continue;
  for (let k = 0; k < 6; k++) {
    const c = crossoverTagged(a.genome, b.genome, rng, (rng() - 0.5) * 2);
    for (const o of [a.origin, b.origin]) {
      rate[o] ??= [0, 0];
      rate[o][1]++;
      if (c.tag === 'layered') rate[o][0]++;
    }
  }
}
console.log(Object.entries(rate).map(([o, [l, n]]) => `${o}:${(l / n * 100).toFixed(1)}`).join(' '));
