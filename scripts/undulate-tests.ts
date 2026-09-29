import { UNDULATE_SCHEMA, packUndulate } from '../src/v2/genes/undulate';
import { SEEDS } from '../src/v2/seeds';
import { cloneGenome, validate, repair } from '../src/v2/genome';
import { crossover, mutate, mulberry32 } from '../src/v2/ops';

type Check = (name: string, ok: boolean, detail: string) => void;
export function undulateTests(check: Check): void {
  const p = Object.fromEntries(Object.entries(UNDULATE_SCHEMA).map(([k, s]) => [k, s.def]));
  const packed = (bars: number) => {
    const E = new Float32Array(8); E.fill(123);
    packUndulate(E, 2, k => p[k], bars); return E;
  };
  for (const rate of UNDULATE_SCHEMA.rate.choices!) {
    p.rate = rate;
    const a = packed(8.25), b = packed(8.25 + (rate === 0 ? 1000 : 1 / Math.abs(rate)));
    check(`undulate.bar-period.${rate}`, Math.abs(a[4] - b[4]) < 1e-6 && (rate !== 0 || a[4] === 0),
      'travels once per selected bar period; zero rate holds a static bend');
  }
  p.angle = 0.25; const E = packed(12345678.25);
  check('undulate.packing', E[0] === 123 && E[1] === 123 && E[7] === 123 && Math.abs(E[5]-Math.PI/2) < 1e-6 && E[6] === 1 && Math.abs(E[4]) < Math.PI*2,
    'packing preserves neighbouring slots, converts turns, and bounds long-running phase');
  const seeds = ['X66', 'X67', 'X68'].map(id => SEEDS.find(s => s.origin === id)!);
  check('undulate.three-presets', seeds.every(s => s && !validate(s.genome).length && s.genome.bodies.some(b => b.deform.kind === 'undulate')),
    'all three starter presets retain travelling deformation');
  const rng = mulberry32(6601); let retained = 0; const errors: string[] = [];
  for (let i = 0; i < 90; i++) {
    const g = crossover(cloneGenome(seeds[i % 3].genome), SEEDS[i].genome, rng);
    if (g.bodies.some(b => b.deform.kind === 'undulate')) retained++;
    for (const child of [g, mutate(g, rng)]) {
      errors.push(...validate(child));
      if (JSON.stringify(child) !== JSON.stringify(repair(child))) errors.push('repair changed valid child');
    }
  }
  check('undulate.inherits', retained > 15 && !errors.length, `${retained}/90 children retained undulate; ${errors.length} invalid children`);
}
