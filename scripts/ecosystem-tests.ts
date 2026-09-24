// Tests for the stem ecosystem emission gene (src/v2/genes/ecosystem.ts). Called from v2-test.ts.

import {
  COST_BUDGET_MS, EMIT_KINDS, EMIT_SCHEMAS, SPECIES, classify, cloneGenome, defaultParams, estimateCost, repair, validate,
  type Genome,
} from '../src/v2/genome';
import { crossover, mulberry32, mutate, randomGenome } from '../src/v2/ops';
import { SEEDS } from '../src/v2/seeds';
import { ADJ_POOLS, NOUN_POOLS, nameFor, nounKind } from '../src/v2/naming';
import { buildSources } from '../src/v2/glsl';
import { ECO_FLOOR, ECO_GLYPHS, ECO_SCHEMA, ecoAgents, ecoCost, ecoCuts, ecoInitial, ecoStep, type EcoState } from '../src/v2/genes/ecosystem';

type Check = (name: string, ok: boolean, detail: string) => void;

const ecoEmit = (p: Record<string, number> = {}) => ({ kind: 'ecosystem' as const, p: { ...defaultParams(ECO_SCHEMA), ...p } });

/** A small hidden dot running an ecosystem: the simplest ecosystem genome. */
export function ecoGenome(p: Record<string, number> = {}): Genome {
  return repair({
    v: 5, chain: [], carrier: { kind: 'warp', p: { halfLife: 0.3 } }, palette: { kind: 'analogous', p: {} }, tone: { p: {} },
    bodies: [{ shape: { kind: 'dot', p: { r: 0.02 } }, place: { kind: 'point' }, material: { kind: 'glow' }, emit: ecoEmit(p) }],
    reactions: [], energy: [0.2, 0.8],
  });
}

/** Runs the population model for secs seconds at 60 fps under a fixed stem presence. */
function run(s: EcoState, pres: number[], secs: number, drop = 0, p: Record<string, number> = defaultParams(ECO_SCHEMA)): EcoState {
  for (let i = 0; i < secs * 60; i++) ecoStep(s, pres, drop, p, 1 / 60);
  return s;
}

export function ecosystemTests(check: Check): void {
  const specOk = Object.entries(ECO_SCHEMA).every(([, s]) => s.min < s.max && s.def >= s.min && s.def <= s.max);
  check('eco.schema', EMIT_KINDS.includes('ecosystem') && EMIT_SCHEMAS.ecosystem === ECO_SCHEMA && specOk && !!ECO_SCHEMA.count.log && !!ECO_SCHEMA.count.int
    && ECO_GLYPHS.length === ECO_SCHEMA.glyph.choices!.length && ECO_GLYPHS.every((g) => g.length === 4),
    `params ${Object.keys(ECO_SCHEMA).join(',')}`);

  // Population dynamics: a species grows while its stem plays and starves while it is silent.
  {
    const full = run(ecoInitial(), [1, 1, 1, 1], 20);
    const silent = run(ecoInitial(), [0, 0, 0, 0], 20);
    const vocal = run(ecoInitial(), [0.1, 0.2, 1, 0.3], 20);
    const bare = run(ecoInitial(), [0.1, 0.2, 0, 0.3], 20);
    const feast = run(ecoInitial(), [1, 0.5, 0.3, 1], 4, 1);
    const plain = run(ecoInitial(), [1, 0.5, 0.3, 1], 4, 0);
    const ok = full.pop.every((x) => x > 0.4) && silent.pop.every((x) => Math.abs(x - ECO_FLOOR) < 1e-6)
      && vocal.pop[2] > 0.6 && vocal.flora > bare.flora + 0.2 && vocal.pop[2] > bare.pop[2] * 5
      && feast.pop[0] > plain.pop[0] && feast.pop[3] < plain.pop[3];
    const back = run(run(ecoInitial(), [0, 0, 0, 0], 20), [1, 1, 1, 1], 30);
    const f = (s: EcoState) => `${s.pop.map((x) => x.toFixed(2)).join('/')} flora ${s.flora.toFixed(2)}`;
    check('eco.populations', ok && back.pop.every((x) => x > 0.3),
      `full ${f(full)}; silent ${f(silent)}; vocal bridge ${f(vocal)} vs no vocals ${f(bare)}; drop predators ${feast.pop[0].toFixed(2)} > ${plain.pop[0].toFixed(2)}, plankton ${feast.pop[3].toFixed(2)} < ${plain.pop[3].toFixed(2)}; starved then revived ${f(back)}`);
    // Predation strength eats plankton; pollinators without grazers grow a meadow, grazers eat it down.
    const hungry = run(ecoInitial(), [1, 0, 0, 1], 20, 0, { ...defaultParams(ECO_SCHEMA), predation: 1 });
    const tame = run(ecoInitial(), [1, 0, 0, 1], 20, 0, { ...defaultParams(ECO_SCHEMA), predation: 0 });
    const meadow = run(ecoInitial(), [0, 0, 1, 0], 20, 0, { ...defaultParams(ECO_SCHEMA), bloom: 1, graze: 1 });
    const grazed = run(ecoInitial(), [0, 1, 1, 0], 20, 0, { ...defaultParams(ECO_SCHEMA), bloom: 1, graze: 1 });
    check('eco.interactions', hungry.pop[3] < tame.pop[3] && meadow.flora > grazed.flora,
      `plankton with predation 1 ${hungry.pop[3].toFixed(2)} < 0 ${tame.pop[3].toFixed(2)}; flora ungrazed ${meadow.flora.toFixed(2)} > grazed ${grazed.flora.toFixed(2)}`);
  }

  // Roster: cuts are cumulative and a species with no weight gets no agents.
  const c = ecoCuts({ wD: 0, wB: 1, wV: 1, wO: 2 });
  check('eco.roster', c[0] === 0 && Math.abs(c[1] - 0.25) < 1e-9 && Math.abs(c[2] - 0.5) < 1e-9 && c[3] === 1
    && ecoCuts({ wD: 0, wB: 0, wV: 0, wO: 0 }).join() === '0,0,0,1' && ecoAgents(32768, 1280 * 720) === 32768 && ecoAgents(32768, 320 * 180) === 2048,
    `cuts ${c.join(',')}`);

  // Every seed's first body can take the emission (flames keep their trail), valid and stable.
  const bad: string[] = [];
  let took = 0;
  for (const s of SEEDS) {
    const g = cloneGenome(s.genome);
    g.bodies[0].emit = ecoEmit();
    const r = repair(g);
    const errs = validate(r);
    if (errs.length) bad.push(`${s.origin}:${errs[0]}`);
    if (JSON.stringify(repair(JSON.parse(JSON.stringify(r)))) !== JSON.stringify(r)) bad.push(`${s.origin}:round-trip`);
    if (!(estimateCost(r) <= COST_BUDGET_MS)) bad.push(`${s.origin}:cost ${estimateCost(r).toFixed(2)}`);
    if (!buildSources(r).feedback) bad.push(`${s.origin}:glsl`);
    if (r.bodies[0].emit.kind === 'ecosystem') took++;
    else if (r.bodies[0].shape.kind !== 'flame') bad.push(`${s.origin}:lost to ${r.bodies[0].emit.kind}`);
  }
  check('eco.every-seed-valid', !bad.length && took > SEEDS.length * 0.8, bad.slice(0, 6).join(' | ') || `${took}/${SEEDS.length} seeds host an ecosystem, valid, stable, under budget`);

  // One ecosystem per genome.
  const two = repair({ ...ecoGenome(), bodies: [ecoGenome().bodies[0], { ...ecoGenome().bodies[0], shape: { kind: 'star', p: {} } }] });
  check('eco.one-per-genome', two.bodies.filter((b) => b.emit.kind === 'ecosystem').length === 1 && !validate(two).length, two.bodies.map((b) => b.emit.kind).join(','));

  // Cost grows with the agents and the sprite size; the budget halves the agents on a costly genome.
  const c1 = estimateCost(ecoGenome({ count: 16384 }));
  const c2 = estimateCost(ecoGenome({ count: 262144, size: 8 }));
  const heavy = cloneGenome(SEEDS.find((s) => s.origin === 'E08')!.genome);
  heavy.bodies[0].emit = ecoEmit({ count: 262144, size: 8 });
  const fitted = repair(heavy);
  const fc = fitted.bodies[0].emit.p.count;
  check('eco.cost-and-budget', c2 > c1 && ecoCost({ count: 262144, size: 8 }) > 4 && ecoCost({ count: 65536, size: 3 }) > ecoCost({ count: 65536, size: 1 })
    && estimateCost(fitted) <= COST_BUDGET_MS && fc < 262144 && fc >= ECO_SCHEMA.count.min,
    `16k ${c1.toFixed(2)} ms, 262k size 8 ${c2.toFixed(2)} ms; plasma + 262k -> ${fc} agents, ${estimateCost(fitted).toFixed(2)} ms`);

  // Crossed with one seed of every species both ways: valid, under budget, mutants valid; the idea is inherited.
  const bySpecies = new Map<string, Genome>();
  for (const s of SEEDS) {
    const sp = classify(s.genome).primary;
    if (!bySpecies.has(sp)) bySpecies.set(sp, s.genome);
  }
  const rng = mulberry32(8181);
  const crossBad: string[] = [];
  let inherited = 0;
  let n = 0;
  const eg = ecoGenome();
  for (const [sp, e] of bySpecies) for (const [a, b] of [[eg, e], [e, eg]]) {
    for (let k = 0; k < 6; k++) {
      const ch = crossover(a, b, rng);
      n++;
      const errs = validate(ch);
      if (errs.length) crossBad.push(`${sp}:${errs[0]}`);
      if (!(estimateCost(ch) <= COST_BUDGET_MS)) crossBad.push(`${sp}:cost`);
      if (ch.bodies.some((x) => x.emit.kind === 'ecosystem')) inherited++;
      const mu = mutate(ch, rng, 1.5);
      if (validate(mu).length) crossBad.push(`${sp}:mutant ${validate(mu)[0]}`);
    }
  }
  check('eco.crossover-every-species', !crossBad.length && inherited > n * 0.2 && bySpecies.size >= SPECIES.length / 2,
    crossBad.slice(0, 6).join(' | ') || `${n} crossovers with ${bySpecies.size} species valid; ${inherited} children keep an ecosystem`);

  // Random genomes reach the emission.
  const r2 = mulberry32(6262);
  let randomHits = 0;
  let randomBad = 0;
  for (let i = 0; i < 400; i++) {
    const g = randomGenome(r2);
    if (validate(g).length) randomBad++;
    if (g.bodies.some((b) => b.emit.kind === 'ecosystem')) randomHits++;
  }
  check('eco.random-reachable', randomHits > 0 && !randomBad, `${randomHits}/400 random genomes host an ecosystem, ${randomBad} invalid`);

  // Names: a hidden body reads as a habitat; a visible one earns a teeming adjective.
  const hidden = ecoGenome({ body: 0 });
  const nounOk = nounKind(hidden.bodies[0]) === 'habitat' && NOUN_POOLS.habitat.includes(nameFor(hidden).split(' ').pop()!);
  let teeming = 0;
  for (let i = 0; i < 40; i++) {
    const g = ecoGenome({ predation: i / 40, body: 0.5 + (i % 5) * 0.1 });
    if (ADJ_POOLS.teeming.some((w) => nameFor(g).split(' ').includes(w))) teeming++;
  }
  check('eco.names', nounOk && teeming > 10, `${nameFor(hidden)}; ${teeming}/40 ecosystem names use a teeming adjective`);

  // Showcase seeds B01..: valid, under budget, stable, each with an ecosystem.
  const bs = SEEDS.filter((x) => /^B\d\d$/.test(x.origin));
  check('eco.seeds', bs.length >= 1 && bs.every((x) => x.genome.bodies.some((b) => b.emit.kind === 'ecosystem') && !validate(x.genome).length && estimateCost(x.genome) < COST_BUDGET_MS
    && JSON.stringify(repair(JSON.parse(JSON.stringify(x.genome)))) === JSON.stringify(x.genome)),
    bs.map((x) => `${x.origin} ${x.name} ${estimateCost(x.genome).toFixed(2)}ms (${nameFor(x.genome)})`).join(', '));
}
