// Tests for the V2 genome operators (repair, validate, crossover, mutate,
// classify, population lineage/fitness/serialization, cull, glsl builders).
// Run: node --import ./scripts/analysis-test.hooks.mjs scripts/v2-test.ts

import {
  classify, cloneGenome, energyOf, estimateCost, structuralKey, validate, repair,
  EMITTER_KINDS, FLAME_VARIATIONS,
  type Genome,
} from '../src/v2/genome';
import { crossover, mulberry32, mutate, randomGenome } from '../src/v2/ops';
import { SEEDS } from '../src/v2/seeds';
import { Population, fitness } from '../src/v2/population';
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

console.log(`\n${failures ? 'FAILED' : 'PASSED'}: ${failures} failing check(s)`);
process.exit(failures ? 1 : 0);
