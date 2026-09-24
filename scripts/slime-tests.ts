// Tests for the physarum (slime) emission gene (src/v2/genes/physarum.ts). Called from v2-test.ts.

import {
  COST_BUDGET_MS, EMIT_KINDS, EMIT_SCHEMAS, SPECIES, classify, cloneGenome, defaultParams, estimateCost, repair, validate,
  type Genome,
} from '../src/v2/genome';
import { crossover, mulberry32, mutate, randomGenome } from '../src/v2/ops';
import { SEEDS } from '../src/v2/seeds';
import { Population } from '../src/v2/population';
import { ADJ_POOLS, NOUN_POOLS, nameFor, nounKind } from '../src/v2/naming';
import { buildSources } from '../src/v2/glsl';
import { SLIME_SCHEMA, slimeAgents, slimeCost, slimeDisplayScale } from '../src/v2/genes/physarum';

type Check = (name: string, ok: boolean, detail: string) => void;

const slimeEmit = (p: Record<string, number> = {}) => ({ kind: 'slime' as const, p: { ...defaultParams(SLIME_SCHEMA), ...p } });

/** A dot at the centre growing a network: the simplest slime genome. */
export function slimeGenome(p: Record<string, number> = {}): Genome {
  return repair({
    v: 5, chain: [], carrier: { kind: 'warp', p: { halfLife: 0.4 } }, palette: { kind: 'analogous', p: {} }, tone: { p: {} },
    bodies: [{ shape: { kind: 'dot', p: { r: 0.02 } }, place: { kind: 'point' }, material: { kind: 'glow' }, emit: slimeEmit(p) }],
    reactions: [], energy: [0.2, 0.8],
  });
}

export function slimeTests(check: Check): void {
  // Schema: a kind of emission with every parameter's default inside its range.
  const specOk = Object.entries(SLIME_SCHEMA).every(([, s]) => s.min < s.max && s.def >= s.min && s.def <= s.max);
  check('slime.schema', EMIT_KINDS.includes('slime') && EMIT_SCHEMAS.slime === SLIME_SCHEMA && specOk && !!SLIME_SCHEMA.count.log && !!SLIME_SCHEMA.count.int,
    `emit kinds ${EMIT_KINDS.join(',')}; params ${Object.keys(SLIME_SCHEMA).join(',')}`);

  // Every seed's first body can take the emission (flames keep their trail), valid and stable.
  const bad: string[] = [];
  let took = 0;
  for (const s of SEEDS) {
    const g = cloneGenome(s.genome);
    g.bodies[0].emit = slimeEmit();
    const r = repair(g);
    const errs = validate(r);
    if (errs.length) bad.push(`${s.origin}:${errs[0]}`);
    if (JSON.stringify(repair(JSON.parse(JSON.stringify(r)))) !== JSON.stringify(r)) bad.push(`${s.origin}:round-trip`);
    if (!(estimateCost(r) <= COST_BUDGET_MS)) bad.push(`${s.origin}:cost ${estimateCost(r).toFixed(2)}`);
    if (!buildSources(r).feedback) bad.push(`${s.origin}:glsl`);
    if (r.bodies[0].emit.kind === 'slime') took++;
    else if (r.bodies[0].shape.kind !== 'flame') bad.push(`${s.origin}:lost to ${r.bodies[0].emit.kind}`);
  }
  check('slime.every-seed-valid', !bad.length && took > SEEDS.length * 0.8, bad.slice(0, 6).join(' | ') || `${took}/${SEEDS.length} seeds grow a network (flames keep their trail), valid, stable, under budget`);

  // One agent simulation per genome.
  const two = repair({ ...slimeGenome(), bodies: [slimeGenome().bodies[0], { ...slimeGenome().bodies[0], shape: { kind: 'star', p: {} } }] });
  check('slime.one-per-genome', two.bodies.filter((b) => b.emit.kind === 'slime').length === 1 && !validate(two).length, two.bodies.map((b) => b.emit.kind).join(','));

  // Cost grows with the agents; the budget halves the agents first on a costly body, never below the minimum.
  const c1 = estimateCost(slimeGenome({ count: 65536 }));
  const c2 = estimateCost(slimeGenome({ count: 1048576 }));
  const plasma = cloneGenome(SEEDS.find((s) => s.origin === 'E08')!.genome);
  plasma.bodies[0].emit = slimeEmit({ count: 1048576 });
  const fitted = repair(plasma);
  const fc = fitted.bodies[0].emit.p.count;
  check('slime.cost-and-budget', c2 > c1 && slimeCost(1048576) > 3 && slimeGenome({ count: 1048576 }).bodies[0].emit.p.count === 1048576
    && estimateCost(fitted) <= COST_BUDGET_MS && fc < 1048576 && fc >= SLIME_SCHEMA.count.min,
    `65k ${c1.toFixed(2)} ms, 1M ${c2.toFixed(2)} ms; plasma + 1M agents -> ${fc} agents, ${estimateCost(fitted).toFixed(2)} ms`);

  // Agents scale with the trail area; the display scale divides out density, deposit and fade.
  const sc = (p: Record<string, number>) => slimeDisplayScale({ ...defaultParams(SLIME_SCHEMA), ...p });
  check('slime.scaling', slimeAgents(262144, 1280 * 720) === 262144 && slimeAgents(262144, 320 * 180) === 16384 && slimeAgents(16384, 100) === 4096
    && Math.abs(sc({ count: 524288 }) * 2 - sc({})) < 1e-9 && sc({ deposit: 0.6 }) < sc({}) && sc({ decay: 0.99 }) < sc({}),
    'agents follow the trail area (min 4096); brightness normalised');

  // Crossed with one seed of every species both ways: valid, under budget, mutants valid; the idea is inherited.
  const bySpecies = new Map<string, Genome>();
  for (const s of SEEDS) {
    const sp = classify(s.genome).primary;
    if (!bySpecies.has(sp)) bySpecies.set(sp, s.genome);
  }
  const rng = mulberry32(7171);
  const crossBad: string[] = [];
  let inherited = 0;
  let n = 0;
  const sg = slimeGenome();
  for (const [sp, e] of bySpecies) for (const [a, b] of [[sg, e], [e, sg]]) {
    for (let k = 0; k < 6; k++) {
      const c = crossover(a, b, rng);
      n++;
      const errs = validate(c);
      if (errs.length) crossBad.push(`${sp}:${errs[0]}`);
      if (!(estimateCost(c) <= COST_BUDGET_MS)) crossBad.push(`${sp}:cost`);
      if (c.bodies.some((x) => x.emit.kind === 'slime')) inherited++;
      const mu = mutate(c, rng, 1.5);
      if (validate(mu).length) crossBad.push(`${sp}:mutant ${validate(mu)[0]}`);
    }
  }
  check('slime.crossover-every-species', !crossBad.length && inherited > n * 0.2 && bySpecies.size >= SPECIES.length / 2,
    crossBad.slice(0, 6).join(' | ') || `${n} crossovers with ${bySpecies.size} species valid; ${inherited} children grow a network`);

  // Random genomes and mutation reach the emission.
  const r2 = mulberry32(5150);
  let randomHits = 0;
  let randomBad = 0;
  for (let i = 0; i < 400; i++) {
    const g = randomGenome(r2);
    if (validate(g).length) randomBad++;
    if (g.bodies.some((b) => b.emit.kind === 'slime')) randomHits++;
  }
  check('slime.random-reachable', randomHits > 0 && !randomBad, `${randomHits}/400 random genomes grow a network, ${randomBad} invalid`);

  // Names: a hidden body reads as a network; a slime body earns a veined adjective.
  const hidden = slimeGenome({ body: 0 });
  const nounOk = nounKind(hidden.bodies[0]) === 'network' && NOUN_POOLS.network.includes(nameFor(hidden).split(' ').pop()!);
  let veined = 0;
  for (let i = 0; i < 40; i++) {
    const g = slimeGenome({ sa: 0.1 + i * 0.03, body: 0.5 + (i % 5) * 0.1 });
    if (ADJ_POOLS.veined.some((w) => nameFor(g).split(' ').includes(w))) veined++;
  }
  check('slime.names', nounOk && veined > 10, `${nameFor(hidden)}; ${veined}/40 slime names use a veined adjective`);

  // Showcase seeds: P01.. grow a network; a population saved before them gains them exactly once.
  {
    const ps = SEEDS.filter((x) => /^P\d\d$/.test(x.origin));
    check('slime.seeds', ps.length >= 1 && ps.every((x) => x.genome.bodies.some((b) => b.emit.kind === 'slime') && !validate(x.genome).length && estimateCost(x.genome) < COST_BUDGET_MS
      && JSON.stringify(repair(JSON.parse(JSON.stringify(x.genome)))) === JSON.stringify(x.genome)),
      ps.map((x) => `${x.origin} ${x.name} ${estimateCost(x.genome).toFixed(2)}ms (${nameFor(x.genome)})`).join(', '));
    const base = Population.seeded(1);
    for (const x of ps) base.members.delete(`G0-${x.origin}`);
    base.get('G0-E07')!.likes = 2;
    const kid = base.addChild(crossover(SEEDS[4].genome, SEEDS[21].genome, mulberry32(6)), [base.get('G0-E05')!, base.get('G0-E22')!], 2);
    base.vote(kid.id, true);
    const loaded = Population.fromJSON(JSON.parse(JSON.stringify({ ...base.toJSON(), seedVersion: 7 })));
    const before = JSON.stringify(loaded.list().sort((a, b) => a.id.localeCompare(b.id)));
    const added = loaded.upgradeSeeds(9);
    const rest = JSON.stringify(loaded.list().filter((m) => !ps.some((x) => m.id === `G0-${x.origin}`)).sort((a, b) => a.id.localeCompare(b.id)));
    check('slime.migrate-once', JSON.stringify(added) === JSON.stringify(ps.map((x) => `G0-${x.origin}`)) && loaded.upgradeSeeds(10).length === 0 && before === rest
      && loaded.get(kid.id)!.likes === 1 && loaded.get('G0-E07')!.likes === 2 && ps.every((x) => JSON.stringify(loaded.get(`G0-${x.origin}`)!.genome) === JSON.stringify(x.genome)),
      `added ${added.join(',')}; votes, seeds and bred children untouched`);
  }
}
