// Tests for the V2 genome operators (repair, validate, crossover, mutate,
// classify, population lineage/fitness/serialization, cull, glsl builders).
// Run: node --import ./scripts/analysis-test.hooks.mjs scripts/v2-test.ts

import {
  classify, cloneGenome, energyOf, estimateCost, structuralKey, validate, repair,
  EMITTER_KINDS, FLAME_VARIATIONS,
  type Genome,
} from '../src/v2/genome';
import { addLayer, crossover, crossoverTagged, makeMerge, morphParams, mulberry32, mutate, randomEmitter, randomGenome, randomOp } from '../src/v2/ops';
import { COST_BUDGET_MS, EMITTER_SCHEMAS, SDF_KINDS, BASIC_KINDS, flatEmitters, isSdfKind } from '../src/v2/genome';
import { SEEDS, SEED_VERSION } from '../src/v2/seeds';
import { Population, fitness, POPULATION_VERSION, uniqueName } from '../src/v2/population';
import { nameFor, NOUN_POOLS, ADJ_POOLS, HUE_WORDS } from '../src/v2/naming';
import { buildSources } from '../src/v2/glsl';

let failures = 0;
function check(name: string, ok: boolean, detail: string): void {
  console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}: ${detail}`);
  if (!ok) failures++;
}

function seedByOrigin(origin: string): Genome {
  const s = SEEDS.find((x) => x.origin === origin);
  if (!s) throw new Error(`missing seed ${origin}`);
  return s.genome;
}

// -------------------------------------------------------------- 1. seeds

{
  const origins = SEEDS.map((s) => s.origin);
  const expected = Array.from({ length: 24 }, (_, i) => `E${String(i + 1).padStart(2, '0')}`);
  check('seeds.count', SEEDS.length === 24, `${SEEDS.length} seeds`);
  check('seeds.order', JSON.stringify(origins) === JSON.stringify(expected), origins.join(','));

  let allValid = true;
  let allIdempotent = true;
  const badValid: string[] = [];
  const badIdempotent: string[] = [];
  for (const s of SEEDS) {
    const errs = validate(s.genome);
    if (errs.length) {
      allValid = false;
      badValid.push(`${s.origin}:${errs.join(';')}`);
    }
    const repaired = repair(s.genome);
    if (JSON.stringify(repaired) !== JSON.stringify(s.genome)) {
      allIdempotent = false;
      badIdempotent.push(s.origin);
    }
  }
  check('seeds.validate', allValid, badValid.join(' | ') || 'all 24 valid');
  check('seeds.repair-idempotent', allIdempotent, badIdempotent.join(',') || 'repair(seed) === seed for all 24');
}

// ------------------------------------------------------ 2. repair idempotence

{
  const rng = mulberry32(1001);
  let idemFails = 0;
  let n = 0;
  for (let i = 0; i < 2000; i++) {
    n++;
    const g = randomGenome(rng);
    const r1 = repair(g);
    const r2 = repair(r1);
    if (JSON.stringify(r1) !== JSON.stringify(r2)) idemFails++;
  }
  check('repair.idempotent-random', idemFails === 0, `${idemFails}/${n} mismatches`);

  const garbageInputs: unknown[] = [
    null,
    undefined,
    {},
    { chain: 'x' },
    { chain: [{ op: 'zoom', stage: 'warp', w: NaN, p: { rate: 1e9, cx: -Infinity, cy: NaN } }] },
    { emitters: [{ kind: 'nope', p: {} }] },
    { emitters: [{ kind: 'wave', p: {} }, { kind: 'wave', p: { gain: 5 } }, { kind: 'stars', p: {} }] },
    { emitters: Array.from({ length: 10 }, (_, i) => ({ kind: EMITTER_KINDS[i % EMITTER_KINDS.length], p: {} })) },
    { chain: Array.from({ length: 12 }, () => ({ op: 'zoom', stage: 'warp', w: 1, p: {} })) },
    {
      chain: [{ op: 'zoom', stage: 'warp', w: 0.5, p: {} }],
      reactions: [
        { src: 'bass', g: 'op', i: 99, k: 'rate', gain: 0.5 },
        { src: 'beat', g: 'em', i: 50, k: 'gain', gain: -2 },
      ],
    },
  ];
  let garbageOk = true;
  const garbageBad: string[] = [];
  garbageInputs.forEach((input, i) => {
    try {
      const r = repair(input);
      const errs = validate(r);
      if (errs.length) {
        garbageOk = false;
        garbageBad.push(`#${i}:${errs.join(';')}`);
      }
    } catch (e) {
      garbageOk = false;
      garbageBad.push(`#${i}:threw ${(e as Error).message}`);
    }
  });
  check('repair.garbage-inputs-valid', garbageOk, garbageBad.join(' | ') || `${garbageInputs.length} garbage inputs all repaired to valid genomes`);
}

// ------------------------------------------------------------- 3. crossover

{
  const rng = mulberry32(2002);
  const pool: Genome[] = SEEDS.map((s) => s.genome);
  let bad: string[] = [];
  const N = 3000;
  for (let i = 0; i < N; i++) {
    const useSeeds = rng() < 0.5;
    const a = useSeeds ? pool[Math.floor(rng() * pool.length)] : randomGenome(rng);
    const b = useSeeds ? pool[Math.floor(rng() * pool.length)] : randomGenome(rng);
    const child = crossover(a, b, rng);
    const errs = validate(child);
    if (errs.length) bad.push(`#${i}:${errs.join(';')}`);
    if (child.chain.length > 6) bad.push(`#${i}:chain-too-long(${child.chain.length})`);
    if (child.emitters.length < 1 || child.emitters.length > 3) bad.push(`#${i}:emitter-count(${child.emitters.length})`);
    const kinds = new Set(child.emitters.map((e) => e.kind));
    if (kinds.size !== child.emitters.length) bad.push(`#${i}:duplicate-emitter-kinds`);
    const flame = child.emitters.find((e) => e.kind === 'flame');
    if (flame && (!flame.xforms || flame.xforms.length < 1 || flame.xforms.length > 4)) bad.push(`#${i}:flame-xforms(${flame.xforms?.length})`);
  }
  check('crossover.always-valid', bad.length === 0, bad.slice(0, 5).join(' | ') || `${N} crossovers all valid`);

  // Explicitly spot-check a few numeric params stay within spec.
  const spotBad: string[] = [];
  for (let i = 0; i < 200; i++) {
    const a = pool[Math.floor(rng() * pool.length)];
    const b = pool[Math.floor(rng() * pool.length)];
    const child = crossover(a, b, rng);
    if (!(child.color.p.hue >= 0 && child.color.p.hue <= 1)) spotBad.push(`hue=${child.color.p.hue}`);
    if (!(child.carrier.p.halfLife >= 0.04 && child.carrier.p.halfLife <= 25)) spotBad.push(`halfLife=${child.carrier.p.halfLife}`);
    if (!(child.energy[0] >= 0 && child.energy[1] <= 1 && child.energy[0] < child.energy[1])) spotBad.push(`energy=${child.energy}`);
  }
  check('crossover.spot-params-in-range', spotBad.length === 0, spotBad.slice(0, 5).join(' | ') || '200 spot checks fine');
}

// --------------------------------------------------------------- 4. mutate

{
  const rng = mulberry32(3003);
  const pool: Genome[] = SEEDS.map((s) => s.genome);
  let invalid = 0;
  let changed = 0;
  const N = 3000;
  const invalidDetails: string[] = [];
  for (let i = 0; i < N; i++) {
    const base = rng() < 0.5 ? pool[Math.floor(rng() * pool.length)] : randomGenome(rng);
    const amt = 0.5 + rng() * 2.5; // 0.5..3
    const child = mutate(base, rng, amt);
    const errs = validate(child);
    if (errs.length) {
      invalid++;
      if (invalidDetails.length < 5) invalidDetails.push(`#${i}:${errs.join(';')}`);
    }
    if (JSON.stringify(child) !== JSON.stringify(base)) changed++;
  }
  check('mutate.always-valid', invalid === 0, invalidDetails.join(' | ') || `${N} mutations all valid`);
  const frac = changed / N;
  check('mutate.usually-changes', frac >= 0.9, `${changed}/${N} differed (${(frac * 100).toFixed(1)}%)`);
}

// ---------------------------------------------------- 5. cross-type specifics

function xIsVarOpChainOnly(g: Genome): boolean {
  return g.chain.some((o) => o.op.startsWith('v_'));
}

{
  const flameSeed = seedByOrigin('E22'); // flame
  const vortexSeed = seedByOrigin('E05'); // vortex
  const scopeSeed = seedByOrigin('E07'); // oscilloscope
  const inkSeed = seedByOrigin('E06'); // ink
  const starsSeed = seedByOrigin('E12'); // starfield (Warp Speed classifies as stars/vortex-ish; used for structure mixing)

  function structureTransferHappens(a: Genome, b: Genome, seedBase: number, N: number): { hits: number; N: number } {
    let hits = 0;
    for (let i = 0; i < N; i++) {
      const rng = mulberry32(seedBase + i);
      const child = crossover(a, b, rng);
      const hasFlame = child.emitters.some((e) => e.kind === 'flame');
      const hasNonFlame = child.emitters.some((e) => e.kind !== 'flame');
      const chainNonEmpty = child.chain.length > 0;
      const flameDroppedWithVarOp = !hasFlame && xIsVarOpChainOnly(child);
      if ((hasFlame && hasNonFlame && chainNonEmpty) || flameDroppedWithVarOp) hits++;
    }
    return { hits, N };
  }

  const r1 = structureTransferHappens(flameSeed, vortexSeed, 10000, 400);
  check('crossover.flame-x-vortex-structure-transfer', r1.hits >= 1, `${r1.hits}/${r1.N} children showed structure transfer`);

  const r2 = structureTransferHappens(flameSeed, scopeSeed, 20000, 400);
  check('crossover.flame-x-scope-structure-transfer', r2.hits >= 1, `${r2.hits}/${r2.N} children showed structure transfer`);

  let inkStarsHits = 0;
  const N3 = 400;
  for (let i = 0; i < N3; i++) {
    const rng = mulberry32(30000 + i);
    const child = crossover(inkSeed, starsSeed, rng);
    const hasInkOrFluid = child.emitters.some((e) => e.kind === 'ink') || child.carrier.kind === 'fluid';
    const hasParticles = child.emitters.some((e) => e.kind === 'particles' || e.kind === 'stars');
    if (hasInkOrFluid && hasParticles) inkStarsHits++;
  }
  check('crossover.ink-x-stars-both-present', inkStarsHits >= 1, `${inkStarsHits}/${N3} children had both ink/fluid and particles/stars`);
}

// -------------------------------------------------------------- 6. classify

{
  const expectations: [string, string][] = [
    ['E22', 'flame'], ['E23', 'flame'], ['E24', 'flame'],
    ['E07', 'scope'], ['E06', 'ink'], ['E17', 'vortex'],
    ['E12', 'stars'], ['E15', 'chrome'], ['E16', 'aurora'],
    ['E08', 'plasma'], ['E14', 'wire'], ['E11', 'spectrum'],
  ];
  for (const [origin, expected] of expectations) {
    const s = SEEDS.find((x) => x.origin === origin)!;
    const c = classify(s.genome);
    check(`classify.${origin}->${expected}`, c.primary === expected, `got primary=${c.primary} secondary=${c.secondary} label=${c.label}`);
  }
}

// ------------------------------------------------------- 7. ids and lineage

{
  const pop = Population.seeded();
  check('population.seeded-size', pop.size === 24, `size=${pop.size}`);
  const ids = pop.list().map((m) => m.id).sort();
  const expectedIds = SEEDS.map((s) => `G0-${s.origin}`).sort();
  check('population.seeded-ids', JSON.stringify(ids) === JSON.stringify(expectedIds), ids.join(','));

  const p1 = pop.get('G0-E01')!;
  const p2 = pop.get('G0-E02')!;
  const child1 = pop.addChild(randomGenome(mulberry32(99)), [p1, p2]);
  check('population.child1-id', child1.id === 'G1-0001', child1.id);
  check('population.child1-parents', JSON.stringify(child1.parents.sort()) === JSON.stringify([p1.id, p2.id].sort()), child1.parents.join(','));
  check('population.child1-gen', child1.gen === 1, `gen=${child1.gen}`);

  const p0 = pop.get('G0-E03')!;
  const child2 = pop.addChild(randomGenome(mulberry32(100)), [child1, p0]);
  check('population.child2-id', child2.id === 'G2-0002', child2.id);
  check('population.child2-gen', child2.gen === 2, `gen=${child2.gen}`);
}

// ------------------------------------------------------------- 8. fitness

{
  const pop = Population.seeded();
  const [likedId, dislikedId] = ['G0-E01', 'G0-E02'];
  for (let i = 0; i < 5; i++) pop.vote(likedId, true);
  for (let i = 0; i < 5; i++) pop.vote(dislikedId, false);
  const liked = pop.get(likedId)!;
  const disliked = pop.get(dislikedId)!;
  const fitLiked = fitnessOf(pop, likedId);
  const fitDisliked = fitnessOf(pop, dislikedId);
  check('fitness.liked-beats-disliked', fitLiked > fitDisliked, `liked=${fitLiked.toFixed(4)} disliked=${fitDisliked.toFixed(4)}`);

  const before = fitnessOf(pop, likedId);
  pop.vote(likedId, true);
  pop.vote(likedId, true);
  const after = fitnessOf(pop, likedId);
  check('fitness.more-likes-raises', after > before, `before=${before.toFixed(4)} after=${after.toFixed(4)}`);

  const neutralId = 'G0-E04';
  const beforeView = fitnessOf(pop, neutralId);
  pop.recordView(neutralId, 3, true);
  const m = pop.get(neutralId)!;
  check('recordView.soft-dislike-increments', m.softDislikes === 1, `softDislikes=${m.softDislikes}`);
  const afterView = fitnessOf(pop, neutralId);
  check('recordView.soft-dislike-lowers-fitness', afterView < beforeView, `before=${beforeView.toFixed(4)} after=${afterView.toFixed(4)}`);

  const neutral2 = 'G0-E05';
  pop.recordView(neutral2, 70, false);
  const m2 = pop.get(neutral2)!;
  check('recordView.weak-like-increments', m2.weakLikes === 1, `weakLikes=${m2.weakLikes}`);
}

function fitnessOf(pop: Population, id: string): number {
  return fitness(pop.get(id)!);
}

// -------------------------------------------------------- 9. serialization

{
  const pop = Population.seeded();
  const p1 = pop.get('G0-E01')!;
  const p2 = pop.get('G0-E02')!;
  pop.addChild(randomGenome(mulberry32(7)), [p1, p2]); // G1-0001
  pop.addChild(randomGenome(mulberry32(8)), [p1, p2]); // G1-0002

  const j1 = pop.toJSON();
  const str = JSON.stringify(j1);
  const parsed = JSON.parse(str);
  const pop2 = Population.fromJSON(parsed);
  const j2 = pop2.toJSON();

  // Compare with member order normalized (Map iteration order should match insertion, but sort defensively).
  const norm = (d: ReturnType<Population['toJSON']>) => ({
    ...d,
    members: [...d.members].sort((a, b) => a.id.localeCompare(b.id)),
  });
  check('serialization.roundtrip-equal', JSON.stringify(norm(j1)) === JSON.stringify(norm(j2)), 'toJSON->stringify->parse->fromJSON->toJSON matches');

  const child3 = pop2.addChild(randomGenome(mulberry32(9)), [p1, p2]);
  check('serialization.counter-preserved', child3.id === 'G1-0003', child3.id);

  let threw = false;
  try {
    Population.fromJSON({ not: 'a population' });
  } catch {
    threw = true;
  }
  check('serialization.rejects-garbage', threw, threw ? 'threw as expected' : 'did not throw');

  const withBadMember = {
    format: 'musicvis-v2-population',
    version: 1,
    counter: 0,
    votesSinceBreed: 0,
    members: [
      ...j1.members,
      { id: 'G0-BAD', gen: 0, parents: [], created: Date.now(), name: 'Bad', genome: { v: 1, chain: 'not-an-array' }, likes: 0, dislikes: 0, softDislikes: 0, weakLikes: 0, views: 0, watch: 0, hidden: false },
    ],
  };
  const pop3 = Population.fromJSON(withBadMember);
  check('serialization.drops-invalid-members', pop3.size === j1.members.length, `size=${pop3.size} expected=${j1.members.length}`);
}

// -------------------------------------------------------------------- 10. cull

{
  const pop = Population.seeded();
  const p1 = pop.get('G0-E01')!;
  const p2 = pop.get('G0-E02')!;
  const oldNow = Date.now() - 700_000; // past the 10-minute grace period
  for (let i = 0; i < 200; i++) {
    pop.addChild(randomGenome(mulberry32(50000 + i)), [p1, p2], oldNow);
  }
  const beforeSize = pop.size;
  const removed = pop.cull(150);
  const afterSize = pop.size;
  check('cull.leaves-150', afterSize === 150, `before=${beforeSize} after=${afterSize} removed=${removed.length}`);
  const seedsGone = SEEDS.filter((s) => !pop.get(`G0-${s.origin}`)).map((s) => s.origin);
  check('cull.never-removes-seeds', seedsGone.length === 0, seedsGone.join(',') || 'all 24 seeds still present');
}

// ------------------------------------------------------ 11. glsl / structuralKey

{
  const rng = mulberry32(4004);
  let bad: string[] = [];
  const genomes: Genome[] = [...SEEDS.map((s) => s.genome)];
  for (let i = 0; i < 200; i++) genomes.push(randomGenome(rng));
  for (let i = 0; i < genomes.length; i++) {
    const src = buildSources(genomes[i]);
    if (!src.feedback.includes('void main')) bad.push(`#${i}:feedback missing void main`);
    if (!src.composite.includes('void main')) bad.push(`#${i}:composite missing void main`);
  }
  check('glsl.buildSources-void-main', bad.length === 0, bad.slice(0, 5).join(' | ') || `${genomes.length} sources all contain void main in both stages`);

  // structuralKey equal for genomes differing only in numeric params.
  const base = cloneGenome(seedByOrigin('E05'));
  const jittered = cloneGenome(base);
  if (jittered.chain.length) {
    const op0 = jittered.chain[0];
    const keys = Object.keys(op0.p);
    if (keys.length) op0.p[keys[0]] = op0.p[keys[0]] + 0.001;
  }
  jittered.color.p.hue = (jittered.color.p.hue + 0.05) % 1;
  const k1 = structuralKey(base);
  const k2 = structuralKey(jittered);
  check('structuralKey.param-jitter-same-key', k1 === k2, `base=${k1} jittered=${k2}`);
}

// -------------------------------------------------------------- 12. cost

{
  let over: string[] = [];
  for (const s of SEEDS) {
    const cost = estimateCost(s.genome);
    if (!(cost < 8)) over.push(`${s.origin}=${cost.toFixed(2)}`);
  }
  check('estimateCost.under-budget', over.length === 0, over.join(',') || 'all 24 seeds under 8ms');
}

// ------------------------------------------------- 13. seeds: range, round trip

{
  const bad: string[] = [];
  for (const s of SEEDS) {
    const back = repair(JSON.parse(JSON.stringify(s.genome)));
    if (JSON.stringify(back) !== JSON.stringify(s.genome)) bad.push(`${s.origin}:roundtrip`);
    if (validate(back).length) bad.push(`${s.origin}:${validate(back).join(';')}`);
  }
  check('seeds.serialization-roundtrip', bad.length === 0, bad.join(' | ') || 'all 24 seeds survive JSON + repair unchanged and in range');

  // The reworked seeds use the parts that express their V1 behaviour.
  const uses: [string, (g: Genome) => boolean][] = [
    ['E01', (g) => g.chain.some((o) => o.op === 'stretch' && o.stage === 'view')],
    ['E02', (g) => g.emitters.some((e) => e.kind === 'snake' && e.p.count === 2 && e.p.cover === 1)],
    ['E03', (g) => g.chain.some((o) => o.op === 'translate' && o.p.lanes > 1)],
    ['E06', (g) => g.emitters.some((e) => e.kind === 'ink' && e.p.inst === 1) && g.carrier.kind === 'fluid'],
    ['E08', (g) => g.emitters.some((e) => e.kind === 'plasma' && e.p.tempo === 1 && e.p.pulse === 1)],
    ['E09', (g) => g.chain.some((o) => o.op === 'push') && g.emitters.some((e) => e.kind === 'ink' && e.p.jump === 1 && e.p.swap === 1)],
    ['E12', (g) => g.emitters.some((e) => e.kind === 'particles' && e.p.surge === 1) && g.reactions.some((r) => r.src === 'surge')],
    ['E19', (g) => g.emitters.some((e) => e.kind === 'wave' && e.p.shape === 5)],
    ['E20', (g) => g.emitters.some((e) => e.kind === 'orb' && e.p.arms > 0)],
    ['E21', (g) => g.emitters.some((e) => e.kind === 'horizon' && e.p.terrain > 0)],
    ['E23', (g) => g.emitters.some((e) => e.kind === 'flame' && e.p.flow === 2)],
  ];
  const miss = uses.filter(([o, f]) => !f(seedByOrigin(o))).map(([o]) => o);
  check('seeds.reworked-parts', miss.length === 0, miss.join(',') || `${uses.length} reworked seeds use their new parts`);

  const src = buildSources(seedByOrigin('E02'));
  check('glsl.cover-emitter', src.feedback.includes('c = em_snake(p, c);'), 'snake paints over the trail in the feedback pass');
  const top = cloneGenome(seedByOrigin('E02'));
  top.emitters[0].layer = 'top';
  check('glsl.cover-emitter-top', buildSources(top).composite.includes('c = em_snake(q, c);'), 'snake on the top layer paints over the composite');
  check('glsl.stretch-view', buildSources(seedByOrigin('E01')).composite.includes('vMul *='), 'stretch scales the displayed picture');
}

// ------------------------------------------------ 14. new parts breed validly

{
  const rng = mulberry32(5005);
  const NEW_EMITTERS = ['snake', 'orb', 'horizon', 'ink', 'plasma', 'particles', 'wave'] as const;
  const withNewParts = (): Genome => {
    const g = randomGenome(rng);
    const e = randomEmitter(rng, NEW_EMITTERS[Math.floor(rng() * NEW_EMITTERS.length)]);
    // Switch the new behaviours on.
    if (e.kind === 'orb') e.p.arms = 0.3 + 0.7 * rng();
    if (e.kind === 'horizon') e.p.terrain = 0.3 + 0.7 * rng();
    if (e.kind === 'ink') e.p.inst = rng();
    if (e.kind === 'plasma') e.p.tempo = rng();
    if (e.kind === 'particles') e.p.surge = rng();
    if (e.kind === 'wave') e.p.shape = rng() < 0.5 ? 4 : 5;
    g.emitters = [e, ...g.emitters.filter((x) => x.kind !== e.kind)].slice(0, 3);
    g.chain = [randomOp(rng, rng() < 0.5 ? 'push' : 'stretch'), ...g.chain].slice(0, 6);
    if (rng() < 0.5) g.reactions.push({ src: 'surge', g: 'op', i: 0, k: 'amt', gain: 0.5 });
    return repair(g);
  };
  const flames = SEEDS.filter((s) => s.genome.emitters.some((e) => e.kind === 'flame')).map((s) => s.genome);
  const reworked = ['E01', 'E02', 'E06', 'E08', 'E09', 'E12', 'E19', 'E20', 'E21'].map(seedByOrigin);
  const bad: string[] = [];
  let newInChild = 0;
  const N = 2000;
  for (let i = 0; i < N; i++) {
    const a = rng() < 0.5 ? withNewParts() : reworked[Math.floor(rng() * reworked.length)];
    const b = rng() < 0.4 ? flames[Math.floor(rng() * flames.length)] : rng() < 0.5 ? withNewParts() : randomGenome(rng);
    const child = rng() < 0.3 ? mutate(crossover(a, b, rng), rng, 0.4) : crossover(a, b, rng);
    const errs = validate(child);
    if (errs.length) bad.push(`#${i}:${errs.join(';')}`);
    if (!(estimateCost(child) > 0)) bad.push(`#${i}:cost`);
    if (!buildSources(child).composite.includes('void main')) bad.push(`#${i}:glsl`);
    if (child.emitters.some((e) => e.kind === 'snake') || child.chain.some((o) => o.op === 'push' || o.op === 'stretch')) newInChild++;
  }
  check('breed.new-parts-valid', bad.length === 0, bad.slice(0, 5).join(' | ') || `${N} crossovers / mutations with new parts all valid`);
  check('breed.new-parts-inherited', newInChild > N * 0.3, `${newInChild}/${N} children carry a new part`);

  const mbad: string[] = [];
  for (let i = 0; i < 1500; i++) {
    const child = mutate(rng() < 0.5 ? withNewParts() : reworked[i % reworked.length], rng, 0.5 + 2.5 * rng());
    const errs = validate(child);
    if (errs.length) mbad.push(`#${i}:${errs.join(';')}`);
  }
  check('mutate.new-parts-valid', mbad.length === 0, mbad.slice(0, 5).join(' | ') || '1500 mutations of genomes with new parts all valid');

  // Flame crossed with the reworked seeds keeps the flame's extended ranges.
  const fx = crossover(seedByOrigin('E23'), seedByOrigin('E02'), mulberry32(77));
  check('breed.flame-x-snake-valid', validate(fx).length === 0, validate(fx).join(';') || 'E23 x E02 child valid');
}

// ------------------------------------------------------ 15. seed migration

{
  // A population saved by the previous seed encoding: no seedVersion, stale
  // seed genomes, votes on seeds, and children bred from them with old params.
  const pop = Population.seeded(1000);
  const data = pop.toJSON();
  delete (data as { seedVersion?: number }).seedVersion;
  const oldSeed = (id: string) => data.members.find((m) => m.id === id)!;
  // Old encodings: E02 was a melody ribbon, E12 a dense particle field.
  oldSeed('G0-E02').genome = repair({ ...oldSeed('G0-E02').genome, chain: [{ op: 'translate', stage: 'warp', w: 1, p: { vx: -0.12 } }], emitters: [{ kind: 'edge', layer: 'fb', p: { mode: 1 } }], reactions: [] });
  oldSeed('G0-E12').genome.emitters[0].p.count = 16384;
  Object.assign(oldSeed('G0-E02'), { likes: 4, dislikes: 1, views: 9, watch: 321.5, weakLikes: 2, softDislikes: 1 });
  Object.assign(oldSeed('G0-E12'), { likes: 2, views: 3, watch: 40, hidden: true });
  // Children with old-format genomes (parameters that no longer exist, missing new ones, old flame range).
  const oldChild = {
    id: 'G1-0001', gen: 1, parents: ['G0-E02', 'G0-E22'], created: 2000, name: 'Velvet Tide',
    genome: {
      v: 1,
      chain: [{ op: 'zoom', stage: 'warp', w: 1, p: { rate: 0.01, cx: 0, cy: 0, radial: 0, wander: 0.1 } }, { op: 'translate', stage: 'warp', w: 1, p: { vx: -0.1, vy: 0 } }],
      emitters: [
        { kind: 'edge', layer: 'fb', p: { gain: 1, hue: 0, mode: 1, side: 0, base: 0, height: 0.3, density: 0.6 } },
        { kind: 'flame', layer: 'fb', p: { gain: 1, hue: 0, count: 262144, zoom: 0.2, camSpin: 0.125, rounds: 2, ox: 0, oy: 0, flow: 1, breathe: 0.1, retired: 3 }, xforms: [{ aff: [0.5, 0, 0, 0.5, 0, 0], weight: 1, color: 0, vars: { julia: 1 }, spin: 0, bass: 0, drift: [0.3, 0.3], pulse: 0 }] },
        { kind: 'plasma', layer: 'top', p: { gain: 1, hue: 0, scale: 1.7, warp: 2, bands: 8, lines: 0.8, speed: 0.15 } },
      ],
      carrier: { kind: 'warp', p: { halfLife: 0.5, floor: 1, blur: 0, amount: 1, vort: 28, fnoise: 0.35, fscale: 2, famt: 0.0012 } },
      color: { scheme: 'analogous', p: { hue: 0.5, sat: 0.9, exposure: 1, contrast: 0.03, bloom: 1, adapt: 0.3, vignette: 0.45, ca: 0.0015, reflect: 0, reflectY: -0.16, tonemap: 0 } },
      reactions: [{ src: 'beat', g: 'op', i: 0, k: 'rate', gain: 0.3 }],
      energy: [0.2, 0.7],
    },
    likes: 3, dislikes: 0, softDislikes: 0, weakLikes: 1, views: 5, watch: 99, hidden: false,
  };
  const grandChild = { ...JSON.parse(JSON.stringify(oldChild)), id: 'G2-0002', gen: 2, parents: ['G1-0001', 'G0-E12'], likes: 1 };
  data.members.push(oldChild as never, grandChild as never);
  data.counter = 2;
  const json = JSON.parse(JSON.stringify(data));

  const loaded = Population.fromJSON(json);
  check('migration.old-version-detected', loaded.seedVersion === 1, `seedVersion=${loaded.seedVersion}`);
  const childBefore = JSON.stringify(loaded.get('G1-0001')!.genome);
  const changed = loaded.upgradeSeeds();
  check('migration.version-bumped', loaded.seedVersion === SEED_VERSION, `seedVersion=${loaded.seedVersion}`);
  check('migration.all-seeds-upgraded', changed.length === 24, `${changed.length} seeds upgraded`);
  const e02 = loaded.get('G0-E02')!;
  const e12 = loaded.get('G0-E12')!;
  check('migration.seed-genome-new', JSON.stringify(e02.genome) === JSON.stringify(seedByOrigin('E02')) && JSON.stringify(e12.genome) === JSON.stringify(seedByOrigin('E12')), `E02 emitters=${e02.genome.emitters.map((e) => e.kind)}`);
  check('migration.votes-kept', e02.likes === 4 && e02.dislikes === 1 && e02.views === 9 && e02.watch === 321.5 && e02.weakLikes === 2 && e02.softDislikes === 1 && e12.likes === 2 && e12.hidden, `E02 ${e02.likes}/${e02.dislikes} views=${e02.views} watch=${e02.watch}; E12 likes=${e12.likes} hidden=${e12.hidden}`);
  check('migration.ids-kept', e02.id === 'G0-E02' && e02.gen === 0 && e02.origin === 'E02' && loaded.size === 26, `size=${loaded.size}`);
  const c1 = loaded.get('G1-0001')!;
  const c2 = loaded.get('G2-0002')!;
  check('migration.children-untouched', JSON.stringify(c1.genome) === childBefore && c1.likes === 3 && c2.likes === 1, 'bred children keep genomes and votes');
  check('migration.parent-links-valid', [...c1.parents, ...c2.parents].every((id) => loaded.get(id)), `${c1.parents},${c2.parents}`);
  check('migration.old-child-valid', validate(c1.genome).length === 0 && c1.genome.emitters.length === 3, validate(c1.genome).join(';') || 'old-format child repaired to a valid genome');
  const plasma = c1.genome.emitters.find((e) => e.kind === 'plasma')!;
  check('migration.old-child-defaults', plasma.p.tempo === 0 && plasma.p.pulse === 0 && c1.genome.chain[1].p.lanes === 0, 'new parameters default to the old behaviour');
  check('migration.old-child-renders', buildSources(c1.genome).feedback.includes('void main') && estimateCost(c1.genome) < 8, 'old child still builds shaders');
  const second = loaded.upgradeSeeds();
  check('migration.idempotent', second.length === 0, `${second.length} seeds changed on the second run`);

  // Round trip keeps the version, so the next load does not migrate again.
  const again = Population.fromJSON(JSON.parse(JSON.stringify(loaded.toJSON())));
  check('migration.version-persisted', again.seedVersion === SEED_VERSION && again.upgradeSeeds().length === 0, `seedVersion=${again.seedVersion}`);

  // Seed missing from an old file comes back; a newer file is left alone.
  const missing = Population.fromJSON({ ...json, members: json.members.filter((m: { id: string }) => m.id !== 'G0-E20') });
  missing.upgradeSeeds();
  check('migration.missing-seed-restored', !!missing.get('G0-E20'), 'G0-E20 restored');
  const future = Population.fromJSON({ ...json, seedVersion: SEED_VERSION + 1 });
  check('migration.newer-left-alone', future.upgradeSeeds().length === 0 && future.seedVersion === SEED_VERSION + 1, 'newer seed version untouched');
}


// ------------------------------------------------ 16. fused crossover

{
  // Every ordered pair of seed species: valid, at most two emitters, normally one.
  const rng = mulberry32(6006);
  const bad: string[] = [];
  const tags: Record<string, number> = {};
  let one = 0, n = 0;
  for (const a of SEEDS) for (const b of SEEDS) {
    if (a === b) continue;
    for (let k = 0; k < 3; k++) {
      const { genome: g, tag } = crossoverTagged(a.genome, b.genome, rng);
      n++;
      tags[tag] = (tags[tag] ?? 0) + 1;
      const errs = validate(g);
      if (errs.length) bad.push(`${a.origin}x${b.origin}:${errs.join(';')}`);
      if (g.emitters.length > 2) bad.push(`${a.origin}x${b.origin}:${g.emitters.length} emitters`);
      if (g.emitters.length === 1) one++;
      if (tag === 'merged' && !g.emitters.some((e) => e.kind === 'merge')) bad.push(`${a.origin}x${b.origin}:merged without merge`);
      if (tag === 'layered' && g.emitters.length !== 2) bad.push(`${a.origin}x${b.origin}:layered with ${g.emitters.length}`);
      if (!(estimateCost(g) < COST_BUDGET_MS)) bad.push(`${a.origin}x${b.origin}:cost ${estimateCost(g).toFixed(2)}`);
    }
  }
  check('fuse.all-seed-pairs-valid', bad.length === 0, bad.slice(0, 5).join(' | ') || `${n} children valid, <= 2 emitters, under budget`);
  check('fuse.normally-one-emitter', one / n > 0.85, `${one}/${n} single-emitter children; tags ${JSON.stringify(tags)}`);
  check('fuse.layering-rare', (tags.layered ?? 0) / n < 0.08, `${tags.layered ?? 0}/${n} layered`);
  check('fuse.all-operators-used', ['fused', 'morph', 'merged', 'layered'].every((t) => (tags[t] ?? 0) > 0), JSON.stringify(tags));

  // Recessive shaping: vortex x moon children bend the moon with the vortex (draw chain) or fuse with it.
  let shaped = 0;
  for (let i = 0; i < 200; i++) {
    const { genome: g } = crossoverTagged(seedByOrigin('E20'), seedByOrigin('E05'), mulberry32(7000 + i), 1);
    if (g.draw?.some((o) => o.op === 'swirl' || o.op === 'rotate' || o.op === 'zoom') || g.emitters.some((e) => e.kind === 'merge')) shaped++;
  }
  check('fuse.recessive-shapes-body', shaped > 150, `${shaped}/200 moon-dominant children shaped by the vortex or merged with it`);
}

// ------------------------------------------------ 17. same-kind morph

{
  const rng = mulberry32(8008);
  const bad: string[] = [];
  for (let i = 0; i < 300; i++) {
    const kind = BASIC_KINDS[i % BASIC_KINDS.length];
    const a = randomEmitter(rng, kind);
    const b = randomEmitter(rng, kind);
    const sch = EMITTER_SCHEMAS[kind];
    const m = morphParams(a.p, b.p, sch, rng);
    for (const k of Object.keys(sch)) {
      const s = sch[k];
      const v = m[k];
      if (s.choices) {
        if (v !== a.p[k] && v !== b.p[k]) bad.push(`${kind}.${k} not from a parent`);
      } else if (v < Math.min(a.p[k], b.p[k]) - (s.int ? 1 : 1e-9) || v > Math.max(a.p[k], b.p[k]) + (s.int ? 1 : 1e-9)) bad.push(`${kind}.${k}=${v} outside parents`);
      if (v < s.min || v > s.max) bad.push(`${kind}.${k}=${v} out of range`);
    }
  }
  check('morph.params-between-parents', bad.length === 0, bad.slice(0, 5).join(' | ') || '300 morphs interpolate within the parents and the spec');

  // Flame x flame: one flame whose transforms blend both parents.
  let morphs = 0, oneFlame = 0, blended = 0;
  const e22 = seedByOrigin('E22').emitters[0], e23 = seedByOrigin('E23').emitters[0];
  for (let i = 0; i < 100; i++) {
    const { genome: g, tag } = crossoverTagged(seedByOrigin('E22'), seedByOrigin('E23'), mulberry32(9000 + i));
    const flames = flatEmitters(g).filter((e) => e.kind === 'flame');
    if (flames.length === 1 && g.emitters.length === 1) oneFlame++;
    if (tag !== 'morph') continue;
    morphs++;
    const f = flames[0];
    const z = f.p.zoom;
    const x0 = f.xforms![0];
    const vars = Object.keys(x0.vars);
    if (z >= Math.min(e22.p.zoom, e23.p.zoom) && z <= Math.max(e22.p.zoom, e23.p.zoom) && vars.length >= 2 && validate(g).length === 0) blended++;
  }
  check('morph.flame-x-flame', morphs > 70 && oneFlame > 85 && blended === morphs, `${morphs} morphs, ${oneFlame}/100 single flame, ${blended} blended in range`);
}

// ------------------------------------------------ 18. merged emitters

{
  const rng = mulberry32(10010);
  const bad: string[] = [];
  let n = 0;
  for (const a of SDF_KINDS) for (const b of BASIC_KINDS) {
    if (a === b) continue;
    for (const mode of [0, 1, 2]) {
      if (mode < 2 && !isSdfKind(b)) continue;
      const m = makeMerge(randomEmitter(rng, a), randomEmitter(rng, b), rng, mode);
      if (!m) {
        bad.push(`${a}+${b}:null`);
        continue;
      }
      const g = repair({ v: 2, chain: [randomOp(rng, 'swirl')], draw: [randomOp(rng, 'twist')], emitters: [m], carrier: { kind: 'warp', p: {} }, color: { scheme: 'triad', p: {} }, reactions: [{ src: 'bass', g: 'em', i: 0, k: 'k', gain: 0.5 }, { src: 'beat', g: 'dr', i: 0, k: 'amt', gain: 0.4 }], energy: [0.2, 0.8] });
      n++;
      const errs = validate(g);
      if (errs.length) bad.push(`${a}+${b}/${mode}:${errs.join(';')}`);
      if (!g.emitters.some((e) => e.kind === 'merge')) bad.push(`${a}+${b}/${mode}:merge lost`);
      const back = repair(JSON.parse(JSON.stringify(g)));
      if (JSON.stringify(back) !== JSON.stringify(g)) bad.push(`${a}+${b}/${mode}:roundtrip`);
      if (!(estimateCost(g) < COST_BUDGET_MS)) bad.push(`${a}+${b}/${mode}:cost ${estimateCost(g).toFixed(2)}`);
      const src = buildSources(g);
      if (!src.feedback.includes('em_merge') && !src.composite.includes('em_merge')) bad.push(`${a}+${b}/${mode}:no em_merge`);
      if (g.reactions.length !== 2) bad.push(`${a}+${b}/${mode}:reactions ${g.reactions.length}`);
    }
  }
  check('merge.valid-roundtrip-budget', bad.length === 0, bad.slice(0, 5).join(' | ') || `${n} merges valid, round-trip, under budget, build shaders`);

  // Merges breed: mutation and crossover keep them valid.
  const mbad: string[] = [];
  let kept = 0;
  const E20 = seedByOrigin('E20'), E14 = seedByOrigin('E14');
  for (let i = 0; i < 1000; i++) {
    let g = crossover(E20, E14, rng);
    if (!g.emitters.some((e) => e.kind === 'merge')) continue;
    g = mutate(g, rng, 0.5 + 2 * rng());
    const c = crossover(g, rng() < 0.5 ? E20 : g, rng);
    for (const x of [g, c]) {
      const errs = validate(x);
      if (errs.length) mbad.push(errs.join(';'));
      if (x.emitters.some((e) => e.kind === 'merge')) kept++;
    }
  }
  check('merge.breedable', mbad.length === 0 && kept > 200, mbad.slice(0, 3).join(' | ') || `${kept} merges survived mutation / crossover, all valid`);

  // Broken merges repair into valid genomes (degrade rather than fail).
  const broken = repair({ emitters: [{ kind: 'merge', layer: 'top', p: { mode: 0 }, parts: [{ kind: 'plasma', p: {} }, { kind: 'flame', p: {} }] }] });
  check('merge.repair-broken', validate(broken).length === 0, `${broken.emitters.map((e) => e.kind).join(',')}`);
}

// ------------------------------------------------ 19. add-layer cap

{
  const rng = mulberry32(11011);
  const g1 = cloneGenome(seedByOrigin('E07'));
  const ok1 = addLayer(g1, rng, seedByOrigin('E13').emitters);
  const ok2 = addLayer(g1, rng);
  check('layer.cap-two', ok1 && !ok2 && g1.emitters.length === 2 && g1.emitters[1].kind === 'stars', `first=${ok1} second=${ok2} kinds=${g1.emitters.map((e) => e.kind)}`);
  let over = 0;
  for (let i = 0; i < 2000; i++) {
    let g = crossover(SEEDS[i % 24].genome, SEEDS[(i * 7 + 3) % 24].genome, rng);
    for (let k = 0; k < 4; k++) g = mutate(g, rng, 1 + rng());
    if (g.emitters.length > 2) over++;
  }
  check('layer.mutation-respects-cap', over === 0, `${over}/2000 mutated children above two emitters`);
}

// ------------------------------------------------ 20. old formats load

{
  // A format-1 file (population version 1, genome v 1) with a three-emitter union child.
  const oldGenome = {
    v: 1,
    chain: [{ op: 'rotate', stage: 'warp', w: 1, p: { lock: 0.25, rate: 0, cx: 0, cy: 0, alt: 0, wander: 0 } }],
    emitters: [
      { kind: 'ink', layer: 'fb', p: { ...Object.fromEntries(Object.entries(EMITTER_SCHEMAS.ink).map(([k, s]) => [k, s.def])) } },
      { kind: 'orb', layer: 'top', p: { ...Object.fromEntries(Object.entries(EMITTER_SCHEMAS.orb).map(([k, s]) => [k, s.def])), arms: 1 } },
      { kind: 'stars', layer: 'fb', p: { ...Object.fromEntries(Object.entries(EMITTER_SCHEMAS.stars).map(([k, s]) => [k, s.def])) } },
    ],
    carrier: { kind: 'warp', p: { halfLife: 0.5, floor: 1, blur: 0, amount: 1, vort: 28, fnoise: 0.35, fscale: 2, famt: 0.0012 } },
    color: { scheme: 'triad', p: { hue: 0.5, sat: 0.9, exposure: 1, contrast: 0.03, bloom: 1, adapt: 0.3, vignette: 0.45, ca: 0.0015, reflect: 0, reflectY: -0.16, tonemap: 0 } },
    reactions: [{ src: 'bass', g: 'em', i: 2, k: 'density', gain: 0.4 }],
    energy: [0.2, 0.7],
  };
  const file = { format: 'musicvis-v2-population', version: 1, seedVersion: SEED_VERSION, counter: 1, votesSinceBreed: 0, members: [...Population.seeded(1).toJSON().members, { id: 'G1-0001', gen: 1, parents: ['G0-E05', 'G0-E20'], created: 5, name: 'Old Union', genome: oldGenome, likes: 2, dislikes: 0, softDislikes: 0, weakLikes: 0, views: 1, watch: 10, hidden: false }] };
  const pop = Population.fromJSON(JSON.parse(JSON.stringify(file)));
  const m = pop.get('G1-0001')!;
  const { v: _v1, ...oldRest } = oldGenome;
  const { v: _v2, ...newRest } = m.genome;
  check('format.v1-child-loads', !!m && validate(m.genome).length === 0 && m.genome.emitters.length === 3 && m.likes === 2, validate(m.genome).join(';') || 'three-emitter union child kept');
  check('format.v1-child-unchanged', JSON.stringify(newRest) === JSON.stringify(oldRest) && m.genome.draw === undefined && m.genome.v === 2, 'genome identical apart from the version number');
  check('format.v1-structural-key', structuralKey(m.genome) === 'rotate|ink@fb,orb@top,stars@fb|warp|r0t0', structuralKey(m.genome));
  const out = pop.toJSON();
  const again = Population.fromJSON(JSON.parse(JSON.stringify(out)));
  check('format.v2-roundtrip', out.version === POPULATION_VERSION && JSON.stringify(again.get('G1-0001')!.genome) === JSON.stringify(m.genome), `version=${out.version}`);
  const child = again.addChild(crossover(seedByOrigin('E05'), seedByOrigin('E20'), mulberry32(5)), [again.get('G0-E05')!, again.get('G0-E20')!], 10, 'merged');
  const re = Population.fromJSON(JSON.parse(JSON.stringify(again.toJSON())));
  check('format.cross-tag-persisted', re.get(child.id)!.cross === 'merged', `${child.id} cross=${re.get(child.id)!.cross}`);
  let threw = false;
  try {
    Population.fromJSON({ ...file, version: 99 });
  } catch {
    threw = true;
  }
  check('format.newer-file-rejected', threw, threw ? 'refused' : 'accepted a newer file');
}

// ------------------------------------------------------ 21. descriptive names

function bodyGenome(kind: string, extra: Record<string, unknown> = {}): Genome {
  return repair({
    chain: [], emitters: [{ kind, p: {} }], carrier: { kind: 'warp', p: {} }, color: { scheme: 'analogous', p: {} },
    reactions: [], energy: [0.2, 0.7], ...extra,
  });
}
const nounOf = (name: string) => name.split(/\s+/).slice(1).join(' ');
const adjOf = (name: string) => name.split(/\s+/)[0];

{
  // No fixed hue words, ever: the palette hue is a runtime offset from the
  // song's key, so a name baked around a colour would be wrong on the next song.
  const rng = mulberry32(4242);
  let hueHit = '';
  for (let i = 0; i < 500 && !hueHit; i++) {
    const g = i % 2 === 0 ? randomGenome(rng) : mutate(randomGenome(rng), rng, 1 + rng());
    const words = nameFor(g).split(/\s+/);
    const bad = words.find((w) => HUE_WORDS.includes(w));
    if (bad) hueHit = `${bad} in ${words.join(' ')}`;
  }
  check('naming.no-hue-words', !hueHit, hueHit || '500 random/mutated names, none hue-based');

  // Body-derived nouns.
  const inkName = nameFor(bodyGenome('ink'));
  check('naming.ink-body-noun', NOUN_POOLS.ink.includes(nounOf(inkName)), inkName);
  const orbName = nameFor(bodyGenome('orb'));
  check('naming.orb-body-noun', NOUN_POOLS.orb.includes(nounOf(orbName)), orbName);
  const wireName = nameFor(bodyGenome('wire'));
  check('naming.wire-body-noun', NOUN_POOLS.wire.includes(nounOf(wireName)), wireName);

  // Swirl-heavy: a strong swirl op should win the adjective.
  const swirlGenome = repair({
    chain: [{ op: 'swirl', stage: 'warp', w: 1, p: { amt: 0.03, k: 6, cx: 0, cy: 0, wander: 0 } }],
    emitters: [{ kind: 'wave', p: {} }], carrier: { kind: 'warp', p: {} }, color: { scheme: 'analogous', p: {} },
    reactions: [], energy: [0.2, 0.7],
  });
  const swirlName = nameFor(swirlGenome);
  check('naming.swirl-heavy-adjective', ADJ_POOLS.spiral.includes(adjOf(swirlName)), swirlName);

  // Deterministic: same genome (and same parent names) always names the same.
  const detG = mutate(randomGenome(mulberry32(555)), mulberry32(556), 1);
  const n1 = nameFor(detG, ['Rising Moon', 'Hushed Serpent']);
  const n2 = nameFor(detG, ['Rising Moon', 'Hushed Serpent']);
  check('naming.deterministic', n1 === n2, `${n1} / ${n2}`);

  // Fused crossover: when the child's body kind is unchanged, the dominant
  // parent's noun survives (their noun word is still valid for that kind).
  const pA = bodyGenome('ink');
  const pB = bodyGenome('wire');
  const nameA = nameFor(pA);
  const nameB = nameFor(pB);
  let fusedOk = false;
  let fusedDetail = 'no fused, same-kind child found in 400 tries';
  for (let seed = 0; seed < 400 && !fusedOk; seed++) {
    const res = crossoverTagged(pA, pB, mulberry32(seed), 0.9);
    if (res.tag !== 'fused') continue;
    const kind = res.genome.emitters[0].kind;
    if (kind !== 'ink' && kind !== 'wire') continue;
    const parentNoun = kind === 'ink' ? nounOf(nameA) : nounOf(nameB);
    const childName = nameFor(res.genome, [nameA, nameB]);
    if (nounOf(childName) === parentNoun) {
      fusedOk = true;
      fusedDetail = `seed=${seed} kind=${kind} child=${childName}`;
    }
  }
  check('naming.fused-keeps-dominant-noun', fusedOk, fusedDetail);

  // Duplicate resolution: same base name gets a roman numeral.
  const used = new Set(['Calm Orb']);
  const dup1 = uniqueName(used, 'Calm Orb');
  check('naming.dedupe-first-numeral', dup1 === 'Calm Orb II', dup1);
  used.add(dup1);
  const dup2 = uniqueName(used, 'Calm Orb');
  check('naming.dedupe-second-numeral', dup2 === 'Calm Orb III', dup2);
  check('naming.dedupe-no-collision', uniqueName(used, 'Rising Moon') === 'Rising Moon', 'unrelated base name untouched');

  // Seeds keep their hand-written V1 names exactly.
  check('naming.seeds-unchanged', SEEDS.find((s) => s.origin === 'E01')!.name === 'Night Skyline'
    && SEEDS.find((s) => s.origin === 'E02')!.name === 'River of Light', 'seed names untouched');

  // Migration: a version-1 file's bred children get renamed (votes / ids / parents kept).
  const seededOld = Population.seeded(1);
  const p1 = seededOld.get('G0-E06')!; // Ink Garden
  const p2 = seededOld.get('G0-E14')!; // Polyhedra
  const oldChild = seededOld.addChild(crossover(p1.genome, p2.genome, mulberry32(3)), [p1, p2], 100);
  oldChild.name = 'Random Junk'; // simulate the old scheme's unrelated name
  oldChild.likes = 5;
  const oldGrandchild = seededOld.addChild(crossover(oldChild.genome, p1.genome, mulberry32(4)), [oldChild, p1], 200);
  oldGrandchild.name = 'Other Junk';
  oldGrandchild.dislikes = 2;
  const oldData = seededOld.toJSON();
  (oldData as { version: number }).version = 1;
  const migrated = Population.fromJSON(JSON.parse(JSON.stringify(oldData)));
  const mChild = migrated.get(oldChild.id)!;
  const mGrand = migrated.get(oldGrandchild.id)!;
  check('naming.migration-renamed', mChild.name !== 'Random Junk' && mGrand.name !== 'Other Junk', `${mChild.name} / ${mGrand.name}`);
  check('naming.migration-kept-votes-and-parents', mChild.likes === 5 && mGrand.dislikes === 2
    && JSON.stringify(mChild.parents.sort()) === JSON.stringify([p1.id, p2.id].sort())
    && JSON.stringify(mGrand.parents.sort()) === JSON.stringify([oldChild.id, p1.id].sort()), `${mChild.name}/${mGrand.name}`);
  check('naming.migration-kept-genomes', JSON.stringify(mChild.genome) === JSON.stringify(oldChild.genome)
    && JSON.stringify(mGrand.genome) === JSON.stringify(oldGrandchild.genome), 'genomes untouched by the rename pass');
  // A grandchild's parent name (the renamed child) is resolved before the grandchild is named:
  // if the child's inherited noun survived, the grandchild's inheritance check saw the *new* name.
  const again2 = Population.fromJSON(JSON.parse(JSON.stringify(migrated.toJSON())));
  check('naming.migration-current-version-stable', again2.get(oldChild.id)!.name === mChild.name
    && again2.get(oldGrandchild.id)!.name === mGrand.name, 'a file already at the current version is not renamed again');
}

// -------------------------------------------------- 22. example crossovers

{
  console.log('\n--- 30 example crossover names ---');
  const rng = mulberry32(99009);
  for (let i = 0; i < 30; i++) {
    const a = SEEDS[Math.floor(rng() * 24)];
    let b = SEEDS[Math.floor(rng() * 24)];
    if (b.origin === a.origin) b = SEEDS[(SEEDS.indexOf(b) + 1) % 24];
    const child = crossoverTagged(a.genome, b.genome, rng, (rng() - 0.5) * 2);
    const childName = nameFor(child.genome, [a.name, b.name]);
    console.log(`${a.name} x ${b.name} -> ${childName}  [${child.tag}]`);
  }
}

console.log(`\n${failures ? 'FAILED' : 'PASSED'}: ${failures} failing check(s)`);
process.exit(failures ? 1 : 0);
