// New body types must remain usable as shapes, fusion partners and inherited genes.
import { cloneGenome, repair, repairBody, validate, SHAPE_SCHEMAS, defaultParams, sdfCapable, estimateCost, COST_BUDGET_MS, type ShapeKind } from '../src/v2/genome';
import { crossover, mutate, mulberry32, randomBody, makeFuse } from '../src/v2/ops';
import { SEEDS } from '../src/v2/seeds';
import { buildSources } from '../src/v2/glsl';
import { packBranch } from '../src/v2/genes/branch';

type Check = (name: string, ok: boolean, detail: string) => void;
export const BODY_FAMILIES: ShapeKind[] = ['branch'];
export function bodyFamilyTests(check: Check): void {
  for (const kind of BODY_FAMILIES) {
    const seeds = SEEDS.filter(s => s.genome.bodies.some(b => b.shape.kind === kind));
    check(`${kind}.parents`, seeds.length >= 2 && seeds.every(s => !validate(s.genome).length && estimateCost(s.genome) < COST_BUDGET_MS), 'two distinct, valid starter parents within the rendering budget');
    const base = cloneGenome(seeds[0].genome);
    const rng = mulberry32(1841);
    let inherited = 0;
    const errors: string[] = [];
    for (let i = 0; i < 80; i++) {
      const child = crossover(base, SEEDS[i % SEEDS.length].genome, rng);
      if (child.bodies.some(b => b.shape.kind === kind || b.fuse?.shape.kind === kind)) inherited++;
      for (const g of [child, mutate(child, rng), repair({ ...base, bodies: [randomBody(rng, kind)] })]) {
        errors.push(...validate(g));
        if (JSON.stringify(repair(g)) !== JSON.stringify(g)) errors.push('not idempotent');
        const source = buildSources(g);
        if (/undefined|NaN/.test(source.feedback + source.composite)) errors.push('bad shader');
      }
    }
    check(`${kind}.breeding`, !errors.length && inherited >= 20, `${inherited}/80 crossovers retained the new shape; ${errors.slice(0, 2).join(';') || 'mutations and random placements valid'}`);
    const b = repairBody({ shape: { kind, p: { size: -9 } }, material: { kind: 'fill' } });
    const fused = makeFuse(repairBody({ shape: { kind: 'dot' } }), { kind, p: defaultParams(SHAPE_SCHEMAS[kind]) }, rng);
    check(`${kind}.fusion`, sdfCapable(b.shape) && b.shape.p.size === SHAPE_SCHEMAS[kind].size.min && fused?.fuse?.shape.kind === kind, 'repairs invalid dimensions and works as a fusion partner');
  }
  // Packing must respect the destination offset, and use live reaction values, including on fused shapes.
  const E = new Float32Array(24).fill(-77);
  const p = defaultParams(SHAPE_SCHEMAS.branch);
  packBranch(E, 8, k => k === 'grow' ? 0.37 : p[k], 1.25);
  check('branch.packing', E.slice(0,8).every(x=>x===-77) && E.slice(16).every(x=>x===-77) && Math.abs(E[13]-0.37)<1e-6 && E[15]===1.25, 'reaction-driven growth reaches the right uniform without overwriting adjacent genes');
}
