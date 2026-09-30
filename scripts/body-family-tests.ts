// New body types must remain usable as shapes, fusion partners and inherited genes.
import { cloneGenome, repair, repairBody, validate, SHAPE_SCHEMAS, defaultParams, sdfCapable, estimateCost, COST_BUDGET_MS, type ShapeKind } from '../src/v2/genome';
import { crossover, mutate, mulberry32, randomBody, makeFuse } from '../src/v2/ops';
import { SEEDS } from '../src/v2/seeds';
import { buildSources } from '../src/v2/glsl';
import { atlasDistance } from '../src/v2/genes/lilyDistance';
import { packLily } from '../src/v2/genes/lily';
import { packPlume } from '../src/v2/genes/plume';
import { packShell } from '../src/v2/genes/shell';
import { packLinkage } from '../src/v2/genes/linkage';
import { packFabric } from '../src/v2/genes/fabric';
import { packBranch } from '../src/v2/genes/branch';

type Check = (name: string, ok: boolean, detail: string) => void;
export const BODY_FAMILIES: ShapeKind[] = ['branch', 'fabric', 'linkage', 'shell', 'plume', 'lily'];
export function bodyFamilyTests(check: Check): void {
  for (const kind of BODY_FAMILIES) {
    const seeds = SEEDS.filter(s => s.genome.bodies.some(b => b.shape.kind === kind));
    check(`${kind}.parents`, seeds.length >= 3 && seeds.every(s => !validate(s.genome).length && estimateCost(s.genome) < COST_BUDGET_MS), 'three distinct, valid starter parents within the rendering budget');
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
  const alpha = new Uint8Array(18*12*4);
  for(let y=1;y<5;y++) for(let x=1;x<5;x++) alpha[(y*18+x)*4+3]=255;
  const distances=atlasDistance(alpha,18,12);
  check('lily.atlas-distance', distances[(2*18+2)*4]<128 && distances[0]>128
    && distances[(2*18+6)*4]===255 && distances[(8*18+2)*4]===255,
    'painted silhouette has negative interior, positive exterior and isolated neighbouring cells');
  const lily = new Float32Array(16).fill(-77);
  const yl = defaultParams(SHAPE_SCHEMAS.lily);
  const lr = packLily(lily,4,k=>k==='open'?0.42:yl[k],2.5);
  check('lily.packing', lily.slice(0,4).every(x=>x===-77) && lily.slice(12).every(x=>x===-77)
    && Math.abs(lily[6]-0.42)<1e-6 && lily[11]===2.5 && lr>=yl.size,
    'opening reactions, petal phase and stem bounds pack without touching adjacent genes');
  const plume = new Float32Array(16).fill(-77);
  const pp = defaultParams(SHAPE_SCHEMAS.plume);
  const pr = packPlume(plume, 4, k => k === 'bend' ? -0.6 : pp[k]);
  check('plume.packing', plume.slice(0,4).every(x=>x===-77) && plume.slice(12).every(x=>x===-77)
    && Math.abs(plume[6]+0.6)<1e-6 && plume[11]===0 && pr >= pp.size,
    'live curvature packs in both primary and fused shape slots without touching neighbours');
  const shell = new Float32Array(16).fill(-77);
  const sp = defaultParams(SHAPE_SCHEMAS.shell);
  const radius = packShell(shell, 4, k => k === 'aperture' ? 0.9 : sp[k]);
  check('shell.packing', shell.slice(0,4).every(x=>x===-77) && shell.slice(12).every(x=>x===-77)
    && Math.abs(shell[10]-0.9)<1e-6 && shell[11]===0 && radius > sp.size,
    'aperture reactions reach both shape slots and bounds without overwriting adjacent genes');
  const links = new Float32Array(16).fill(-77);
  const lp = defaultParams(SHAPE_SCHEMAS.linkage);
  packLinkage(links, 4, k => k === 'flex' ? 0.91 : lp[k], 3.5);
  check('linkage.packing', links.slice(0,4).every(x=>x===-77) && links.slice(12).every(x=>x===-77) && Math.abs(links[8]-0.91)<1e-6 && links[11]===3.5, 'joint reactions and phase reach the shape and fusion uniforms');
  const cloth = new Float32Array(16).fill(-77);
  const cp = defaultParams(SHAPE_SCHEMAS.fabric);
  packFabric(cloth, 4, k => k === 'depth' ? 0.83 : cp[k], 2.5);
  check('fabric.packing', cloth.slice(0,4).every(x=>x===-77) && cloth.slice(12).every(x=>x===-77) && Math.abs(cloth[7]-0.83)<1e-6 && cloth[11]===2.5, 'live pleat depth and musical phase pack without overwriting neighbouring uniforms');
  // Packing must respect the destination offset, and use live reaction values, including on fused shapes.
  const E = new Float32Array(24).fill(-77);
  const p = defaultParams(SHAPE_SCHEMAS.branch);
  packBranch(E, 8, k => k === 'grow' ? 0.37 : p[k], 1.25);
  check('branch.packing', E.slice(0,8).every(x=>x===-77) && E.slice(16).every(x=>x===-77) && Math.abs(E[13]-0.37)<1e-6 && E[15]===1.25, 'reaction-driven growth reaches the right uniform without overwriting adjacent genes');
}
