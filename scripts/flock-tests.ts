// Tests for the boids (flock) emission gene (src/v2/genes/boids.ts). Called from v2-test.ts.

import {
  COST_BUDGET_MS, EMIT_KINDS, EMIT_SCHEMAS, classify, cloneGenome, defaultParams, estimateCost, reactable, repair, validate,
  type Genome,
} from '../src/v2/genome';
import { crossover, mulberry32, mutate, randomGenome } from '../src/v2/ops';
import { SEEDS } from '../src/v2/seeds';
import { ADJ_POOLS, NOUN_POOLS, nameFor, nounKind } from '../src/v2/naming';
import { buildSources } from '../src/v2/glsl';
import { FLOCK_SCHEMA, flockCost } from '../src/v2/genes/boids';

type Check = (name: string, ok: boolean, detail: string) => void;

const flockEmit = (p: Record<string, number> = {}) => ({ kind: 'flock' as const, p: { ...defaultParams(FLOCK_SCHEMA), ...p } });

/** Three orbiting lights leading a flock: the simplest flock genome. */
function flockGenome(p: Record<string, number> = {}): Genome {
  return repair({
    v: 5, chain: [], carrier: { kind: 'warp', p: { halfLife: 0.2 } }, palette: { kind: 'analogous', p: {} }, tone: { p: {} },
    bodies: [{ shape: { kind: 'dot', p: { r: 0.004 } }, place: { kind: 'orbit', p: { count: 3 } }, material: { kind: 'glow' }, emit: flockEmit(p) }],
    reactions: [], energy: [0.2, 0.8],
  });
}

export function flockTests(check: Check): void {
  const specOk = Object.values(FLOCK_SCHEMA).every((s) => s.min < s.max && s.def >= s.min && s.def <= s.max);
  check('flock.schema', EMIT_KINDS.includes('flock') && EMIT_SCHEMAS.flock === FLOCK_SCHEMA && specOk && !!FLOCK_SCHEMA.count.log
    && JSON.stringify(FLOCK_SCHEMA.onDrop.choices) === '[0,1,2]' && ['speed', 'align', 'cohere', 'separate', 'home'].every((k) => reactable(FLOCK_SCHEMA).includes(k)),
    `params ${Object.keys(FLOCK_SCHEMA).join(',')}; reactable ${reactable(FLOCK_SCHEMA).join(',')}`);

  // Every seed's first body can lead a flock (flames keep their trail), valid, stable, under budget.
  const bad: string[] = [];
  let took = 0;
  for (const s of SEEDS) {
    const g = cloneGenome(s.genome);
    g.bodies[0].emit = flockEmit();
    const r = repair(g);
    const errs = validate(r);
    if (errs.length) bad.push(`${s.origin}:${errs[0]}`);
    if (JSON.stringify(repair(JSON.parse(JSON.stringify(r)))) !== JSON.stringify(r)) bad.push(`${s.origin}:round-trip`);
    if (!(estimateCost(r) <= COST_BUDGET_MS)) bad.push(`${s.origin}:cost ${estimateCost(r).toFixed(2)}`);
    if (!buildSources(r).feedback) bad.push(`${s.origin}:glsl`);
    if (r.bodies[0].emit.kind === 'flock') took++;
    else if (r.bodies[0].shape.kind !== 'flame') bad.push(`${s.origin}:lost to ${r.bodies[0].emit.kind}`);
  }
  check('flock.every-seed-valid', !bad.length && took > SEEDS.length * 0.8, bad.slice(0, 6).join(' | ') || `${took}/${SEEDS.length} seeds lead a flock, valid, stable, under budget`);

  // One flock per genome; a flock and a slime network may share one.
  const one = flockGenome().bodies[0];
  const two = repair({ ...flockGenome(), bodies: [one, { ...one, shape: { kind: 'star', p: {} } }] });
  const both = repair({ ...flockGenome(), bodies: [one, { ...one, emit: { kind: 'slime', p: {} } }] });
  check('flock.one-per-genome', two.bodies.filter((b) => b.emit.kind === 'flock').length === 1 && !validate(two).length
    && both.bodies.map((b) => b.emit.kind).join(',') === 'flock,slime' && !validate(both).length,
    `${two.bodies.map((b) => b.emit.kind).join(',')}; ${both.bodies.map((b) => b.emit.kind).join(',')}`);

  // Cost grows with the birds; the budget halves a big flock on a costly body.
  const plasma = cloneGenome(SEEDS.find((s) => s.origin === 'E08')!.genome);
  plasma.bodies[0].emit = flockEmit({ count: 262144 });
  const fitted = repair(plasma);
  check('flock.cost-and-budget', flockCost(262144) > flockCost(16384) && estimateCost(fitted) <= COST_BUDGET_MS && fitted.bodies[0].emit.p.count < 262144
    && fitted.bodies[0].emit.p.count >= FLOCK_SCHEMA.count.min && flockGenome({ count: 131072 }).bodies[0].emit.p.count === 131072,
    `16k ${flockCost(16384).toFixed(2)} ms, 262k ${flockCost(262144).toFixed(2)} ms; plasma + 262k birds -> ${fitted.bodies[0].emit.p.count}`);

  // Crossed with one seed of every species both ways: valid, under budget, mutants valid; inherited.
  const bySpecies = new Map<string, Genome>();
  for (const s of SEEDS) {
    const sp = classify(s.genome).primary;
    if (!bySpecies.has(sp)) bySpecies.set(sp, s.genome);
  }
  const rng = mulberry32(6262);
  const crossBad: string[] = [];
  let inherited = 0;
  let n = 0;
  const fg = flockGenome();
  for (const [sp, e] of bySpecies) for (const [a, b] of [[fg, e], [e, fg]]) {
    for (let k = 0; k < 6; k++) {
      const c = crossover(a, b, rng);
      n++;
      const errs = validate(c);
      if (errs.length) crossBad.push(`${sp}:${errs[0]}`);
      if (!(estimateCost(c) <= COST_BUDGET_MS)) crossBad.push(`${sp}:cost`);
      if (c.bodies.some((x) => x.emit.kind === 'flock')) inherited++;
      const mu = mutate(c, rng, 1.5);
      if (validate(mu).length) crossBad.push(`${sp}:mutant ${validate(mu)[0]}`);
    }
  }
  check('flock.crossover-every-species', !crossBad.length && inherited > n * 0.2,
    crossBad.slice(0, 6).join(' | ') || `${n} crossovers with ${bySpecies.size} species valid; ${inherited} children lead a flock`);

  const r2 = mulberry32(4242);
  let hits = 0;
  let invalid = 0;
  for (let i = 0; i < 400; i++) {
    const g = randomGenome(r2);
    if (validate(g).length) invalid++;
    if (g.bodies.some((b) => b.emit.kind === 'flock')) hits++;
  }
  check('flock.random-reachable', hits > 0 && !invalid, `${hits}/400 random genomes lead a flock, ${invalid} invalid`);

  // Names: a hidden body reads as a flock; a flock body earns a flocking adjective.
  const hidden = flockGenome({ body: 0 });
  let adj = 0;
  for (let i = 0; i < 40; i++) {
    const g = flockGenome({ speed: 0.1 + i * 0.01, body: 0.5 + (i % 5) * 0.1 });
    if (ADJ_POOLS.flocking.some((w) => nameFor(g).split(' ').includes(w))) adj++;
  }
  check('flock.names', nounKind(hidden.bodies[0]) === 'flock' && NOUN_POOLS.flock.includes(nameFor(hidden).split(' ').pop()!) && adj > 10,
    `${nameFor(hidden)}; ${adj}/40 flock names use a flocking adjective`);

  // Showcase seed: a P seed leads a flock with its reactions on the flock's own params.
  const ps = SEEDS.filter((x) => /^P\d\d$/.test(x.origin) && x.genome.bodies.some((b) => b.emit.kind === 'flock'));
  const keys = ps.flatMap((x) => x.genome.reactions.filter((r) => r.g === 'em').map((r) => r.k));
  check('flock.seed', ps.length >= 1 && ps.every((x) => !validate(x.genome).length && estimateCost(x.genome) < COST_BUDGET_MS) && ['speed', 'separate', 'align'].every((k) => keys.includes(k)),
    ps.map((x) => `${x.origin} ${x.name} ${estimateCost(x.genome).toFixed(2)}ms (${nameFor(x.genome)}); reactions ${keys.join(',')}`).join(', ') || 'none');
}
