// Checks for the ray-marched 'scene' shape gene (src/v2/genes/raymarch.ts); run from v2-test.ts.

import {
  COST_BUDGET_MS, MATERIAL_KINDS, SHAPE_SCHEMAS, classify, cloneGenome, estimateCost, reactable, repair, validate,
  type Genome,
} from '../src/v2/genome';
import { crossover, mulberry32, mutate, randomBody, randomGene } from '../src/v2/ops';
import { SEEDS } from '../src/v2/seeds';
import { Population } from '../src/v2/population';
import { nameFor, nounKind, NOUN_POOLS } from '../src/v2/naming';
import { buildSources } from '../src/v2/glsl';
import { SCENE_SCHEMA, SCENE_VEC4, packScene, sceneCost } from '../src/v2/genes/raymarch';
import type { Frame } from '../src/v2/engine';

type Check = (name: string, ok: boolean, detail: string) => void;

export function raymarchChecks(check: Check): void {
  const rs = SEEDS.filter((s) => s.origin.startsWith('R'));
  const r01 = rs.find((s) => s.origin === 'R01')!.genome;

  // Seeds: valid, stable, under budget, their own species and noun.
  const seedBad: string[] = [];
  for (const s of rs) {
    const g = s.genome;
    if (validate(g).length) seedBad.push(`${s.origin}:${validate(g)[0]}`);
    if (JSON.stringify(repair(JSON.parse(JSON.stringify(g)))) !== JSON.stringify(g)) seedBad.push(`${s.origin}:round-trip`);
    if (!(estimateCost(g) < COST_BUDGET_MS)) seedBad.push(`${s.origin}:cost ${estimateCost(g).toFixed(2)}`);
    if (classify(g).primary !== 'depth') seedBad.push(`${s.origin}:species ${classify(g).primary}`);
    if (!g.bodies.some((b) => b.shape.kind === 'scene')) seedBad.push(`${s.origin}:no scene`);
    if (g.reactions.length < 3) seedBad.push(`${s.origin}:reactions`);
  }
  check('scene.seeds', !seedBad.length && rs.length >= 1, seedBad.join(' | ') || `${rs.length} R seeds valid, stable, under budget, species 3D scene`);
  const name = nameFor(r01);
  check('scene.name', nounKind(r01.bodies[0]) === 'depth' && NOUN_POOLS.depth.includes(name.split(/\s+/).slice(1).join(' ')), name);

  // Shaders: the scene pass exists only with a scene body; every material samples the scene texture.
  const src = buildSources(r01);
  check('scene.glsl-pass', !!src.scene && src.scene.includes('rmMap') && src.scene.includes('uScn') && src.feedback.includes('uScene') && !buildSources(SEEDS[0].genome).scene,
    'scene pass built for scene genomes only; the body samples uScene');
  const matBad: string[] = [];
  for (const m of MATERIAL_KINDS) {
    const g = cloneGenome(r01);
    g.bodies[0].material = { kind: m, p: {} };
    const r = repair(g);
    const s = buildSources(r);
    if (validate(r).length || !s.scene || !(s.feedback + s.composite).includes('texture(uScene')) matBad.push(m);
  }
  check('scene.every-material', !matBad.length, matBad.join(',') || `${MATERIAL_KINDS.length} materials light the scene`);

  // Structure: one scene per genome, copies collapse to one, reactions skip the switches.
  const two = cloneGenome(r01);
  two.bodies.push(cloneGenome(r01).bodies[0]);
  const orb = repair({ ...cloneGenome(r01), bodies: [{ ...cloneGenome(r01).bodies[0], place: { kind: 'orbit', p: { count: 4 } } }] });
  const keys = reactable(SHAPE_SCHEMAS.scene);
  check('scene.structure', repair(two).bodies.length === 1 && orb.bodies[0].place.p.count === 1 && !keys.includes('scene') && !keys.includes('res') && keys.includes('size') && keys.includes('kick') && keys.includes('vary'),
    `one scene per genome, single copy, reactable: ${keys.join(',')}`);

  // Cost: grows with the internal resolution; over budget, repair lowers the resolution first.
  const lo = sceneCost({ ...r01.bodies[0].shape.p, res: 0.35 });
  const hi = sceneCost({ ...r01.bodies[0].shape.p, res: 0.7 });
  const heavy = cloneGenome(r01);
  heavy.bodies[0].shape.p.res = 0.7;
  heavy.chain = Array.from({ length: 6 }, () => ({ op: 'noise' as const, stage: 'warp' as const, w: 1, p: { amp: 0.001, scale: 2, speed: 0.3 } }));
  const fixed = repair(heavy);
  check('scene.cost', hi > lo * 3 && fixed.bodies[0].shape.p.res < 0.7 && estimateCost(repair({ ...cloneGenome(r01), chain: [] })) < COST_BUDGET_MS,
    `res 0.35 ${lo.toFixed(2)} ms, 0.7 ${hi.toFixed(2)} ms; heavy genome repaired to res ${fixed.bodies[0].shape.p.res}`);

  // Cameras: each mode moves differently, stays finite, and never ends up inside a primitive's bound.
  const camBad: string[] = [];
  const paths: number[][] = [];
  for (const cam of SCENE_SCHEMA.cam.choices!) for (const [roam, size] of [[1, 1.6], [0, 1.6], [0, 0.4]]) {
    const p = { ...r01.bodies[0].shape.p, cam, roam, size, pulse: 1, kick: 1 };
    const out = new Float32Array(SCENE_VEC4 * 4);
    const mem: Record<string, number> = {};
    const F = { speed: 1, act: 1, loud: 1, beatPulse: 0, gate: new Float32Array([1, 1, 1, 1]), stem: new Float32Array(4), onset: new Float32Array(4), sectionIndex: 0, bars: 0 } as unknown as Frame;
    let minGap = 1e9;
    const path: number[] = [];
    for (let i = 0; i < 1800; i++) {
      F.bars = i / 120;
      F.sectionIndex = Math.floor(i / 400);
      F.stem[1] = i % 50 < 10 ? 1 : 0;
      F.onset[0] = i % 30 === 0 ? 1 : 0;
      F.beatPulse = (i % 30) / 30;
      packScene(out, { F, sdt: 1 / 60, P: (k) => p[k], raw: p, mem, key: 'b0.' });
      if (![...out].every(Number.isFinite)) { camBad.push(`cam ${cam}: not finite`); break; }
      for (let j = 0; j < 5; j++) {
        const o = (8 + j) * 4;
        minGap = Math.min(minGap, Math.hypot(out[0] - out[o], out[1] - out[o + 1], out[2] - out[o + 2]) - 1.35 * out[o + 3]);
      }
      if (i % 300 === 0) path.push(+out[0].toFixed(2), +out[2].toFixed(2), +out[3].toFixed(2));
    }
    if (!(minGap > 0.05)) camBad.push(`cam ${cam} roam ${roam} size ${size}: inside a shape (gap ${minGap.toFixed(2)})`);
    paths.push(path);
  }
  const distinct = new Set(paths.filter((_, i) => i % 3 === 0).map((x) => x.join())).size === SCENE_SCHEMA.cam.choices!.length;
  check('scene.cameras', !camBad.length && distinct, camBad.join(' | ') || `${SCENE_SCHEMA.cam.choices!.length} camera modes move differently and keep clear of the shapes (3 settings each)`);

  // Random scene genes are in range; scene bodies breed with every species (both ways) and mutate validly.
  const rng = mulberry32(31337);
  let rangeBad = 0;
  for (let i = 0; i < 200; i++) {
    const s = randomGene('shape', rng, 'scene');
    for (const [k, sp] of Object.entries(SCENE_SCHEMA)) if (!(s.p[k] >= sp.min && s.p[k] <= sp.max)) rangeBad++;
  }
  const bySpecies = new Map<string, Genome>();
  for (const s of SEEDS) {
    const sp = classify(s.genome).primary;
    if (!bySpecies.has(sp) && sp !== 'depth') bySpecies.set(sp, s.genome);
  }
  const crossBad: string[] = [];
  let crosses = 0, kept = 0;
  for (const [sp, other] of bySpecies) for (const [a, b] of [[r01, other], [other, r01]]) {
    for (let k = 0; k < 6; k++) {
      const c = crossover(a, b, rng);
      crosses++;
      if (c.bodies.some((x) => x.shape.kind === 'scene')) kept++;
      if (validate(c).length) crossBad.push(`${sp}:${validate(c)[0]}`);
      if (!(estimateCost(c) < COST_BUDGET_MS)) crossBad.push(`${sp}:cost ${estimateCost(c).toFixed(2)}`);
      const mu = mutate(c, rng, 1.5);
      if (validate(mu).length) crossBad.push(`${sp}:mutant ${validate(mu)[0]}`);
    }
  }
  check('scene.breeds-every-species', !crossBad.length && !rangeBad && kept > crosses * 0.2,
    crossBad.slice(0, 6).join(' | ') || `${crosses} crossovers with ${bySpecies.size} species valid and under budget (${kept} keep the scene), random genes in range`);
  let appeared = 0;
  for (let i = 0; i < 300; i++) if (randomBody(rng).shape.kind === 'scene') appeared++;
  check('scene.random-bodies', appeared > 3 && appeared < 60, `${appeared}/300 random bodies are scenes`);

  // Migration: a seed-version-6 population gains the R seeds exactly once; the rest stays as it was.
  const base = Population.seeded(1);
  for (const s of rs) base.members.delete(`G0-${s.origin}`);
  base.get('G0-E07')!.likes = 2;
  const kid = base.addChild(crossover(SEEDS[4].genome, SEEDS[13].genome, mulberry32(9)), [base.get('G0-E05')!, base.get('G0-E14')!], 2);
  base.vote(kid.id, true);
  const loaded = Population.fromJSON(JSON.parse(JSON.stringify({ ...base.toJSON(), seedVersion: 6 })));
  const before = JSON.stringify(loaded.list().sort((a, b) => a.id.localeCompare(b.id)));
  const added = loaded.upgradeSeeds(5);
  const after = JSON.stringify(loaded.list().filter((m) => !m.id.startsWith('G0-R')).sort((a, b) => a.id.localeCompare(b.id)));
  check('scene.migrate', JSON.stringify(added) === JSON.stringify(rs.map((s) => `G0-${s.origin}`)) && before === after && loaded.upgradeSeeds(6).length === 0 && loaded.get(kid.id)!.likes === 1 && loaded.get('G0-E07')!.likes === 2,
    `added ${added.join(',')} once; votes and children untouched`);
}
