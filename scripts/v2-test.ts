// Tests for the V2 genome operators (repair, validate, crossover, mutate,
// classify, population lineage/fitness/serialization, cull, glsl builders,
// migration of older formats, names).
// Run: node --import ./scripts/analysis-test.hooks.mjs scripts/v2-test.ts

import {
  COST_BUDGET_MS, DEFORM_KINDS, EMIT_KINDS, LOCI, LOCUS_KINDS, MATERIAL_KINDS, MOTION_KINDS, PLACE_KINDS, SHAPE_CLASS,
  SHAPE_KINDS, SHAPE_SCHEMAS, UNIQUE_SHAPES,
  bodyCost, classify, cloneBody, cloneGenome, estimateCost, locusSchema, repair, repairBody, sdfCapable, structuralKey, validate,
  type BodyGene, type Gene, type Genome, type Locus,
} from '../src/v2/genome';
import {
  addLayer, crossover, crossoverTagged, makeFuse, morphParams, mulberry32, mutate, randomBody, randomGene, randomGenome,
  randomOp, MUTATION_NAMES,
} from '../src/v2/ops';
import { SEEDS, SEED_VERSION } from '../src/v2/seeds';
import { Population, fitness, POPULATION_VERSION, uniqueName } from '../src/v2/population';
import {
  TONE_SCHEMA, CARRIER_KINDS, CARRIER_SCHEMA, DRAW_OPS, MAX_CHAIN, MAX_DRAW, MAX_REACTIONS, OP_KINDS, OP_SCHEMAS, PALETTE_KINDS, REACTION_SCHEMA,
  reactable, schemaFor,
  type ParamSpec, type Schema, type Signal,
} from '../src/v2/genome';
import {
  addDrawOp, addOp, addReaction, buildModel, editParam, expressAllele, freeTargets, fromSlider, moveOp, parseGenome,
  reactionTargets, reactKey, removeDrawOp, removeOp, removeReaction, saveEdited, setParam, setReactionTarget, setStage,
  switchKind, toSlider, SLIDER_STEPS,
  type ParamControl, type Target,
} from '../src/v2/geneEdit';
import { nameFor, nounKind, NOUN_POOLS, ADJ_POOLS, HUE_WORDS } from '../src/v2/naming';
import { buildSources, WAVE_VS } from '../src/v2/glsl';
import { SUPERSCOPE_SCHEMA } from '../src/v2/genes/superscope';
import { CELLS_SCHEMA } from '../src/v2/genes/cells';
import { TUNNEL_SCHEMA } from '../src/v2/genes/tunnel';
import { repair as repairV2, upgradeV2, EMITTER_SCHEMAS as V2_SCHEMAS } from '../src/v2/legacy';
import { choreoTests } from './choreo-tests';
import { driftTests } from './drift-tests';
import { slimeTests } from './slime-tests';
import { flockTests } from './flock-tests';
import { ecosystemTests } from './ecosystem-tests';
import { physicsChecks } from './v2-physics';
import { raymarchChecks } from './raymarch-checks';
import { noveltyTests, noveltyTestsAsync } from './novelty-tests';
import { lyricsTests } from './lyrics-tests';

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
const b0 = (origin: string): BodyGene => seedByOrigin(origin).bodies[0];
/** MilkDrop re-creations present (M01.. in order). */
const M_COUNT = SEEDS.filter((s) => s.origin.startsWith('M')).length;
const defs = (schema: Record<string, { def: number }>) => Object.fromEntries(Object.entries(schema).map(([k, s]) => [k, s.def]));

/** A minimal valid genome (one grid of dots, empty chain) to build up structural edits from scratch. */
function freshGenome(): Genome {
  return repair({
    v: 5, chain: [], bodies: [{ shape: { kind: 'dot' }, place: { kind: 'grid' } }],
    carrier: { kind: 'warp', p: {} }, palette: { kind: 'triad', p: {} }, tone: { p: {} }, reactions: [], energy: [0.2, 0.8],
  });
}

// -------------------------------------------------------------- 1. seeds

{
  const origins = SEEDS.map((s) => s.origin);
  const expected = [
    ...Array.from({ length: 24 }, (_, i) => `E${String(i + 1).padStart(2, '0')}`),
    ...Array.from({ length: M_COUNT }, (_, i) => `M${String(i + 1).padStart(2, '0')}`),
  ];
  // Feature seeds (C01.. choreography, and other prefixes) follow the original E and M seeds.
  const base = origins.filter((o) => /^[EM]\d/.test(o));
  check('seeds.count', base.length === 24 + M_COUNT && M_COUNT >= 11, `${SEEDS.length} seeds (${M_COUNT} MilkDrop)`);
  check('seeds.order', JSON.stringify(origins.slice(0, base.length)) === JSON.stringify(expected) && new Set(origins).size === origins.length, origins.join(','));
  const badValid: string[] = [];
  const badIdem: string[] = [];
  const badTrip: string[] = [];
  const over: string[] = [];
  for (const s of SEEDS) {
    const errs = validate(s.genome);
    if (errs.length) badValid.push(`${s.origin}:${errs.join(';')}`);
    if (JSON.stringify(repair(s.genome)) !== JSON.stringify(s.genome)) badIdem.push(s.origin);
    const back = repair(JSON.parse(JSON.stringify(s.genome)));
    if (JSON.stringify(back) !== JSON.stringify(s.genome)) badTrip.push(s.origin);
    if (!(estimateCost(s.genome) < COST_BUDGET_MS)) over.push(`${s.origin}=${estimateCost(s.genome).toFixed(2)}`);
  }
  check('seeds.validate', !badValid.length, badValid.join(' | ') || `all ${SEEDS.length} valid`);
  check('seeds.repair-idempotent', !badIdem.length, badIdem.join(',') || `repair(seed) === seed for all ${SEEDS.length}`);
  check('seeds.serialization-roundtrip', !badTrip.length, badTrip.join(',') || `all ${SEEDS.length} survive JSON + repair unchanged`);
  check('seeds.under-budget', !over.length, over.join(',') || `all under ${COST_BUDGET_MS} ms`);
  check('seeds.version', SEED_VERSION >= 8, `SEED_VERSION=${SEED_VERSION}`);

  // The seeds are combinations of sub-genes (the decomposition the design names).
  const is = (o: string, f: (b: BodyGene, g: Genome) => boolean) => [o, f] as const;
  const uses = [
    is('E01', (b, g) => b.shape.kind === 'edge' && b.shape.p.mode === 0 && g.chain.some((o) => o.op === 'stretch' && o.stage === 'view')),
    is('E02', (b) => b.shape.kind === 'dot' && b.place.kind === 'walker' && b.place.p.heads === 2 && b.motion.kind === 'hits' && b.emit.kind === 'cover'),
    is('E03', (b, g) => b.shape.kind === 'edge' && b.shape.p.mode === 2 && g.chain.some((o) => o.op === 'translate' && o.p.lanes > 1)),
    is('E04', (b) => b.shape.kind === 'dot' && b.place.kind === 'row' && b.emit.kind === 'sparks' && b.emit.p.top === 1),
    is('E05', (b) => b.shape.kind === 'dot' && b.place.kind === 'orbit' && b.material.kind === 'glow'),
    is('E06', (b, g) => b.shape.kind === 'dot' && b.place.kind === 'stations' && b.place.p.inst === 1 && b.emit.kind === 'dye' && g.carrier.kind === 'fluid'),
    is('E07', (b) => b.shape.kind === 'curve' && b.shape.p.form === 0 && b.material.kind === 'line'),
    is('E08', (b) => b.shape.kind === 'plasma' && b.shape.p.tempo === 1 && b.shape.p.pulse === 1),
    is('E09', (b, g) => b.place.kind === 'stations' && b.place.p.jump === 1 && b.place.p.swap === 1 && g.chain.some((o) => o.op === 'push')),
    is('E10', (b) => b.shape.kind === 'polygon' && b.shape.p.n === 6 && b.place.kind === 'grid' && b.place.p.lattice === 1 && b.material.kind === 'fill'),
    is('E11', (b) => b.shape.kind === 'bars' && b.shape.p.mode === 1),
    is('E12', (b, g) => b.emit.kind === 'sparks' && b.emit.p.surge === 1 && b.emit.p.body === 0 && g.reactions.some((r) => r.src === 'surge')),
    is('E13', (b) => b.shape.kind === 'dot' && b.place.kind === 'grid' && b.material.kind === 'glow' && b.place.p.links > 0),
    is('E14', (b) => b.shape.kind === 'solid' && b.motion.kind === 'spin' && b.material.kind === 'line'),
    is('E15', (b) => b.shape.kind === 'dot' && b.place.kind === 'float' && b.place.p.fuse > 0 && b.material.kind === 'chrome'),
    is('E19', (b) => b.shape.kind === 'curve' && b.shape.p.form === 5),
    is('E20', (b) => b.shape.kind === 'dot' && b.deform.kind === 'arms' && b.material.kind === 'textured' && b.motion.kind === 'bob'),
    is('E21', (b, g) => b.shape.kind === 'terrain' && b.shape.p.terrain > 0 && g.bodies[1]?.material.kind === 'textured' && g.bodies[1].material.p.tex === 1),
    is('E23', (b) => b.shape.kind === 'flame' && b.shape.p.flow === 2 && (b.shape.xforms?.length ?? 0) === 3),
  ];
  const miss = uses.filter(([o, f]) => !f(b0(o), seedByOrigin(o))).map(([o]) => o);
  check('seeds.sub-gene-decomposition', !miss.length, miss.join(',') || `${uses.length} seeds use their sub-genes`);

  const src2 = buildSources(seedByOrigin('E02'));
  check('glsl.cover-emission', /c = mix\(c \+ col \* m\.a \* 0\.3 \* uAccum/.test(src2.feedback) && src2.feedback.includes('c = body_0(p, c);'), 'the walker paints over the trail in the feedback pass');
  check('glsl.stretch-view', buildSources(seedByOrigin('E01')).composite.includes('vMul *='), 'stretch scales the displayed picture');
  check('glsl.grid-fold', buildSources(seedByOrigin('E13')).feedback.includes('hash22(cc * 1.7 + 0.3)'), 'stars are a grid fold');
  check('glsl.metaball', buildSources(seedByOrigin('E15')).composite.includes('gMetaOn = 1.0'), 'chrome drops melt as metaballs');
}

// ------------------------------------------------------ 2. repair / validate

{
  const rng = mulberry32(1001);
  const bad: string[] = [];
  for (let i = 0; i < 400; i++) {
    const g = randomGenome(rng);
    const errs = validate(g);
    if (errs.length) bad.push(`random#${i}:${errs.join(';')}`);
    if (JSON.stringify(repair(repair(g))) !== JSON.stringify(repair(g))) bad.push(`random#${i}:not idempotent`);
    if (!(estimateCost(g) > 0)) bad.push(`random#${i}:cost`);
  }
  check('repair.random-valid-idempotent', !bad.length, bad.slice(0, 4).join(' | ') || '400 random genomes valid and idempotent');

  // Garbage never crashes and always repairs to a valid genome.
  const junk: unknown[] = [
    null, 5, 'x', [], {}, { v: 3 }, { v: 3, bodies: 'no' }, { v: 3, bodies: [null, 3, { shape: { kind: 'nope' } }] },
    { v: 3, bodies: [{ shape: { kind: 'flame' }, place: { kind: 'grid' }, emit: { kind: 'cover' } }] },
    { v: 3, bodies: [{ shape: { kind: 'curve', p: { form: 3 } }, material: { kind: 'chrome' }, place: { kind: 'grid' }, fuse: { shape: { kind: 'plasma' } } }] },
    { v: 3, bodies: [{ shape: { kind: 'plasma' }, place: { kind: 'orbit', p: { count: 6 } }, emit: { kind: 'cover' } }] },
    { v: 3, bodies: Array.from({ length: 5 }, () => ({ shape: { kind: 'solid' }, emit: { kind: 'sparks' } })) },
    { v: 2, emitters: [{ kind: 'merge', parts: [{ kind: 'orb' }, { kind: 'orb' }] }] },
    { v: 1, emitters: [{ kind: 'wat' }], chain: [{ op: 'nope' }] },
    { v: 3, bodies: [{ shape: { kind: 'dot', p: { r: NaN } }, place: { kind: 'stations', p: { count: 99 } } }], reactions: [{ src: 'beat', g: 'zz', i: 7, k: 'q', gain: 5 }] },
  ];
  const jbad: string[] = [];
  junk.forEach((j, i) => {
    try {
      const g = repair(j);
      const errs = validate(g);
      if (errs.length) jbad.push(`#${i}:${errs.join(';')}`);
      if (JSON.stringify(repair(g)) !== JSON.stringify(g)) jbad.push(`#${i}:not idempotent`);
    } catch (e) {
      jbad.push(`#${i}:threw ${(e as Error).message}`);
    }
  });
  check('repair.garbage', !jbad.length, jbad.join(' | ') || `${junk.length} malformed inputs repair to valid genomes`);

  // Renderer rules are enforced (and caught by validate when broken by hand).
  const flame = repair({ v: 3, bodies: [{ shape: { kind: 'flame' }, place: { kind: 'ring' }, emit: { kind: 'none' } }] }).bodies[0];
  check('repair.flame-rules', flame.place.kind === 'point' && flame.emit.kind === 'trail', `${flame.place.kind}/${flame.emit.kind}`);
  const field = repair({ v: 3, bodies: [{ shape: { kind: 'plasma' }, place: { kind: 'stations', p: { count: 4 } }, emit: { kind: 'cover' } }] }).bodies[0];
  check('repair.chunk-single-copy', field.place.p.count === 1 && field.emit.kind === 'trail', `count=${field.place.p.count} emit=${field.emit.kind}`);
  const curve = repair({ v: 3, bodies: [{ shape: { kind: 'curve' }, material: { kind: 'textured' }, place: { kind: 'grid' } }] }).bodies[0];
  check('repair.curve-rules', curve.material.kind === 'line' && curve.place.kind === 'point', `${curve.material.kind}/${curve.place.kind}`);
  const two = repair({ v: 3, bodies: [{ shape: { kind: 'solid' } }, { shape: { kind: 'solid' } }, { shape: { kind: 'dot' }, emit: { kind: 'sparks' } }, { shape: { kind: 'dot' }, emit: { kind: 'sparks' } }] });
  check('repair.unique-resources', two.bodies.filter((b) => b.shape.kind === 'solid').length === 1 && two.bodies.filter((b) => b.emit.kind === 'sparks').length === 1, two.bodies.map((b) => `${b.shape.kind}/${b.emit.kind}`).join(','));
  const broken = cloneGenome(seedByOrigin('E14'));
  broken.bodies[0].place = { kind: 'stations', p: { count: 9 } } as never;
  check('validate.catches-range', validate(broken).some((e) => e.includes('count')), validate(broken).join(';'));

  // Budget: repair drops copies of an expensive shape.
  const heavy = repair({ v: 3, bodies: [{ shape: { kind: 'solid' }, place: { kind: 'stations', p: { count: 6 } }, deform: { kind: 'noise' }, material: { kind: 'chrome' } }], carrier: { kind: 'fluid' } });
  check('repair.fits-budget', heavy.bodies[0].place.p.count === 1, `count=${heavy.bodies[0].place.p.count} cost=${estimateCost(heavy).toFixed(2)}`);
}

// ------------------------------------------------ 3. every locus kind is buildable

{
  const rng = mulberry32(2002);
  const bad: string[] = [];
  let n = 0;
  for (const locus of LOCI) {
    for (const kind of LOCUS_KINDS[locus]) {
      for (let k = 0; k < 6; k++) {
        const base = randomBody(rng);
        (base as unknown as Record<string, Gene>)[locus] = randomGene(locus, rng, kind);
        const g = repair({ v: 5, chain: [randomOp(rng)], bodies: [base], carrier: { kind: 'warp', p: {} }, palette: { kind: 'triad', p: {} }, tone: { p: {} }, reactions: [], energy: [0.2, 0.8] });
        n++;
        const errs = validate(g);
        if (errs.length) bad.push(`${locus}=${kind}:${errs.join(';')}`);
        const src = buildSources(g);
        if (!src.feedback.includes('void main') || !src.composite.includes('void main')) bad.push(`${locus}=${kind}:glsl`);
        if (!(estimateCost(g) < 20)) bad.push(`${locus}=${kind}:cost`);
      }
    }
  }
  check('loci.every-kind-builds', !bad.length, bad.slice(0, 5).join(' | ') || `${n} bodies (every kind of every locus) valid and build shaders`);
  const counts = `${SHAPE_KINDS.length} shapes, ${PLACE_KINDS.length} placements, ${MOTION_KINDS.length} motions, ${DEFORM_KINDS.length} deformations, ${MATERIAL_KINDS.length} materials, ${EMIT_KINDS.length} emissions`;
  check('loci.vocabulary', SHAPE_KINDS.length >= 12 && PLACE_KINDS.length >= 10 && MOTION_KINDS.length >= 7 && MATERIAL_KINDS.length >= 6 && EMIT_KINDS.length >= 5, counts);

  // structuralKey: numbers and motion are uniforms (same key); shape / placement class / material are structure.
  const e02 = seedByOrigin('E02');
  const j = cloneGenome(e02);
  j.bodies[0].place.p.step += 0.01;
  j.bodies[0].motion = { kind: 'sway', p: { amp: 0.05, period: 2, tilt: 0.3 } };
  j.palette.p.hue = (j.palette.p.hue + 0.05) % 1;
  check('structuralKey.uniform-changes', structuralKey(repair(j)) === structuralKey(e02), structuralKey(e02));
  const m = cloneGenome(e02);
  m.bodies[0].material = { kind: 'glow', p: { gain: 1, hue: 0, width: 0.02, base: 0, halo: 0 } };
  check('structuralKey.material-is-structure', structuralKey(repair(m)) !== structuralKey(e02), structuralKey(repair(m)));
}

// ------------------------------------------------------------- 4. crossover

{
  // Every ordered pair of seeds, three times: valid, at most two bodies, under budget, buildable.
  const rng = mulberry32(6006);
  const bad: string[] = [];
  const tags: Record<string, number> = {};
  let one = 0, n = 0, mixed = 0;
  const lociFrom = { dom: 0, rec: 0 };
  for (const a of SEEDS) for (const b of SEEDS) {
    if (a === b) continue;
    for (let k = 0; k < 3; k++) {
      const { genome: g, tag } = crossoverTagged(a.genome, b.genome, rng);
      n++;
      tags[tag] = (tags[tag] ?? 0) + 1;
      const errs = validate(g);
      if (errs.length) bad.push(`${a.origin}x${b.origin}:${errs.join(';')}`);
      if (g.bodies.length > 2) bad.push(`${a.origin}x${b.origin}:${g.bodies.length} bodies`);
      if (g.bodies.length === 1) one++;
      if (tag === 'merged' && !g.bodies[0].fuse) bad.push(`${a.origin}x${b.origin}:merged without fuse`);
      if (tag === 'layered' && g.bodies.length !== 2) bad.push(`${a.origin}x${b.origin}:layered with ${g.bodies.length}`);
      if (!(estimateCost(g) < COST_BUDGET_MS)) bad.push(`${a.origin}x${b.origin}:cost ${estimateCost(g).toFixed(2)}`);
      if (n % 7 === 0 && !buildSources(g).composite.includes('void main')) bad.push(`${a.origin}x${b.origin}:glsl`);
      // Which parent each locus of the main body came from (by kind, where the parents differ).
      const cb = g.bodies[0];
      const [pa, pb] = [a.genome.bodies[0], b.genome.bodies[0]];
      let fromA = 0, fromB = 0;
      for (const l of LOCI) {
        const ck = (cb[l] as Gene).kind, ak = (pa[l] as Gene).kind, bk = (pb[l] as Gene).kind;
        if (ak === bk) continue;
        if (ck === ak) fromA++;
        else if (ck === bk) fromB++;
      }
      if (fromA && fromB) mixed++;
      lociFrom.dom += Math.max(fromA, fromB);
      lociFrom.rec += Math.min(fromA, fromB);
    }
  }
  check('cross.all-seed-pairs-valid', !bad.length, bad.slice(0, 5).join(' | ') || `${n} children valid, <= 2 bodies, under budget`);
  check('cross.normally-one-body', one / n > 0.85, `${one}/${n} single-body children; tags ${JSON.stringify(tags)}`);
  check('cross.layering-rare', (tags.layered ?? 0) / n < 0.08, `${tags.layered ?? 0}/${n} layered`);
  check('cross.all-operators-used', ['fused', 'morph', 'merged', 'layered'].every((t) => (tags[t] ?? 0) > 0), JSON.stringify(tags));
  check('cross.mixes-ideas', mixed / n > 0.45, `${mixed}/${n} children take distinct loci from both parents (dominant ${lociFrom.dom}, recessive ${lociFrom.rec} loci)`);

  // Same-shape morph: params between the parents and in range.
  const mbad: string[] = [];
  for (let i = 0; i < 300; i++) {
    const kind = SHAPE_KINDS[i % SHAPE_KINDS.length];
    const a = randomGene('shape', rng, kind);
    const bb = randomGene('shape', rng, kind);
    const sch = SHAPE_SCHEMAS[kind];
    const mm = morphParams(a.p, bb.p, sch, rng);
    for (const k of Object.keys(sch)) {
      const s = sch[k];
      const v = mm[k];
      if (s.choices) {
        if (v !== a.p[k] && v !== bb.p[k]) mbad.push(`${kind}.${k} not from a parent`);
      } else if (v < Math.min(a.p[k], bb.p[k]) - (s.int ? 1 : 1e-9) || v > Math.max(a.p[k], bb.p[k]) + (s.int ? 1 : 1e-9)) mbad.push(`${kind}.${k}=${v} outside parents`);
      if (v < s.min || v > s.max) mbad.push(`${kind}.${k}=${v} out of range`);
    }
  }
  check('morph.params-between-parents', !mbad.length, mbad.slice(0, 5).join(' | ') || '300 shape morphs interpolate within the parents and the spec');

  // Flame x flame: one flame whose transforms blend both parents.
  let morphs = 0, oneFlame = 0, blended = 0;
  const z22 = b0('E22').shape.p.zoom, z23 = b0('E23').shape.p.zoom;
  for (let i = 0; i < 100; i++) {
    const { genome: g, tag } = crossoverTagged(seedByOrigin('E22'), seedByOrigin('E23'), mulberry32(9000 + i));
    const flames = g.bodies.filter((x) => x.shape.kind === 'flame');
    if (flames.length === 1 && g.bodies.length === 1) oneFlame++;
    if (tag !== 'morph') continue;
    morphs++;
    const f = flames[0].shape;
    if (f.p.zoom >= Math.min(z22, z23) && f.p.zoom <= Math.max(z22, z23) && Object.keys(f.xforms![0].vars).length >= 2 && !validate(g).length) blended++;
  }
  check('morph.flame-x-flame', morphs > 70 && oneFlame > 85 && blended === morphs, `${morphs} morphs, ${oneFlame}/100 single flame, ${blended} blended in range`);
}

// ------------------------------------------------ 5. ideas cross between species

{
  // The design's examples appear among real crossover children.
  const find = (a: string, b: string, want: (g: Genome) => boolean, tries = 600): Genome | null => {
    for (let i = 0; i < tries; i++) {
      const g = crossover(seedByOrigin(a), seedByOrigin(b), mulberry32(31000 + i), 1);
      if (want(g)) return g;
    }
    return null;
  };
  const cage = find('E14', 'E02', (g) => g.bodies[0].shape.kind === 'solid' && g.bodies[0].place.kind === 'walker');
  check('ideas.wandering-cage', !!cage && !validate(cage).length, cage ? `solid x walker, emit ${cage.bodies[0].emit.kind}, motion ${cage.bodies[0].motion.kind}` : 'not found');
  const armedFlame = find('E22', 'E20', (g) => g.bodies[0].shape.kind === 'flame' && g.bodies[0].deform.kind === 'arms');
  check('ideas.moon-arms-on-flame', !!armedFlame && buildSources(armedFlame).feedback.includes('fbMask_0'), armedFlame ? 'flame with arms deformation (feedback silhouette mask)' : 'not found');
  const dyeStars = find('E13', 'E06', (g) => g.bodies[0].place.kind === 'grid' && g.bodies[0].emit.kind === 'dye');
  check('ideas.ink-from-star-grid', !!dyeStars && !validate(dyeStars).length, dyeStars ? `grid x dye, carrier ${dyeStars.carrier.kind}` : 'not found');
  const polyInk = find('E06', 'E14', (g) => g.bodies[0].place.kind === 'stations' && g.bodies[0].motion.kind === 'spin');
  check('ideas.polyhedra-turning-on-ink', !!polyInk, polyInk ? `ink stations with bar-locked spin (rate ${polyInk.bodies[0].motion.p.rate})` : 'not found');

  // Loci only move when the child's shape can express them.
  let refused = 0;
  for (let i = 0; i < 300; i++) {
    const g = crossover(seedByOrigin('E22'), seedByOrigin('E13'), mulberry32(41000 + i), 1);
    if (g.bodies[0].shape.kind === 'flame' && g.bodies[0].place.kind !== 'grid') refused++;
  }
  check('ideas.incompatible-loci-refused', refused > 0, `${refused}/300 flame children kept a placement a flame can express`);

  // Fusion at the shape level: every sdf pair in every mode.
  const rng = mulberry32(10010);
  const bad: string[] = [];
  let n = 0;
  for (const a of SHAPE_KINDS) for (const b of SHAPE_KINDS) {
    if (a === b) continue;
    for (const mode of [0, 1, 2]) {
      const body = randomBody(rng, a);
      delete body.fuse;
      body.deform = { kind: 'none', p: {} };
      body.place = { kind: 'point', p: { x: 0, y: 0 } };
      // No particle system either: its cost is not what this checks, and a costly body with sparks sheds the fuse to fit the budget.
      if (body.emit.kind === 'sparks' || body.emit.kind === 'slime') body.emit = { kind: 'trail', p: { tip: 0 } };
      const other = randomGene('shape', rng, b) as BodyGene['shape'];
      if (!sdfCapable(other) || UNIQUE_SHAPES.includes(b)) continue;
      const f = makeFuse(body, other, rng, mode);
      if (!f) continue;
      const g = repair({ v: 5, chain: [randomOp(rng, 'swirl')], bodies: [f], carrier: { kind: 'warp', p: {} }, palette: { kind: 'triad', p: {} }, tone: { p: {} }, reactions: [{ src: 'bass', g: 'fu', i: 0, k: 'k', gain: 0.5 }], energy: [0.2, 0.8] });
      // The budget may legitimately drop a fused shape from an expensive random body.
      if (!g.bodies[0].fuse && estimateCost({ ...g, bodies: [f] }) > COST_BUDGET_MS * 0.95) continue;
      n++;
      const errs = validate(g);
      if (errs.length) bad.push(`${a}+${b}/${mode}:${errs.join(';')}`);
      if (!g.bodies[0].fuse) bad.push(`${a}+${b}/${mode}:fuse lost`);
      if (JSON.stringify(repair(JSON.parse(JSON.stringify(g)))) !== JSON.stringify(g)) bad.push(`${a}+${b}/${mode}:roundtrip`);
      const src = buildSources(g);
      if (SHAPE_CLASS[a] !== 'flame' && !src.feedback.includes('FSH_0') && !src.composite.includes('FSH_0')) bad.push(`${a}+${b}/${mode}:no fused field`);
    }
  }
  check('fuse.valid-roundtrip', !bad.length && n > 100, bad.slice(0, 5).join(' | ') || `${n} fused bodies valid, round-trip, build shaders`);
}

// --------------------------------------------------------------- 6. mutate

{
  const rng = mulberry32(3003);
  const bad: string[] = [];
  const seen = new Set<string>();
  for (let i = 0; i < 3000; i++) {
    const src = i % 3 === 0 ? randomGenome(rng) : SEEDS[i % 24].genome;
    const log: string[] = [];
    const g = mutate(src, rng, 0.3 + 2.5 * rng(), log);
    log.forEach((x) => seen.add(x));
    const errs = validate(g);
    if (errs.length) bad.push(`#${i}(${log.join(',')}):${errs.join(';')}`);
    if (g.bodies.length > 2 && src.bodies.length <= 2) bad.push(`#${i}:${g.bodies.length} bodies`);
  }
  check('mutate.valid', !bad.length, bad.slice(0, 4).join(' | ') || '3000 mutations valid, body cap kept');
  const swaps = ['swap-shape', 'swap-place', 'swap-motion', 'swap-deform', 'swap-material', 'swap-emit'];
  const unused = MUTATION_NAMES.filter((m) => !seen.has(m));
  check('mutate.sub-gene-swaps', swaps.every((s) => seen.has(s)), `unused: ${unused.join(',') || 'none'}`);

  // add-layer respects the cap.
  const g1 = cloneGenome(seedByOrigin('E07'));
  const ok1 = addLayer(g1, rng, seedByOrigin('E13').bodies);
  const ok2 = addLayer(g1, rng);
  check('layer.cap-two', ok1 && !ok2 && g1.bodies.length === 2 && g1.bodies[1].place.kind === 'grid', `first=${ok1} second=${ok2}`);
}

// -------------------------------------------------------------- 7. classify

{
  const expectations: [string, string][] = [
    ['E22', 'flame'], ['E23', 'flame'], ['E24', 'flame'], ['E07', 'scope'], ['E06', 'ink'], ['E17', 'vortex'],
    ['E12', 'stars'], ['E13', 'stars'], ['E15', 'chrome'], ['E16', 'aurora'], ['E08', 'plasma'], ['E14', 'wire'],
    ['E11', 'spectrum'], ['E10', 'mirror'], ['E02', 'scope'],
  ];
  const bad = expectations.filter(([o, e]) => classify(seedByOrigin(o)).primary !== e).map(([o, e]) => `${o}: ${classify(seedByOrigin(o)).primary} != ${e}`);
  check('classify.seeds', !bad.length, bad.join(', ') || `${expectations.length} seeds classified as their species`);
}

// ------------------------------------------------------- 8. ids, lineage, fitness

{
  const pop = Population.seeded();
  check('population.seeded-size', pop.size === SEEDS.length, `size=${pop.size}`);
  const p1 = pop.get('G0-E01')!;
  const p2 = pop.get('G0-E02')!;
  const child1 = pop.addChild(randomGenome(mulberry32(99)), [p1, p2]);
  check('population.child1', child1.id === 'G1-0001' && child1.gen === 1 && child1.parents.length === 2, child1.id);
  const child2 = pop.addChild(randomGenome(mulberry32(100)), [child1, pop.get('G0-E03')!]);
  check('population.child2', child2.id === 'G2-0002' && child2.gen === 2, child2.id);

  for (let i = 0; i < 5; i++) pop.vote('G0-E01', true);
  for (let i = 0; i < 5; i++) pop.vote('G0-E02', false);
  check('fitness.liked-beats-disliked', fitness(pop.get('G0-E01')!) > fitness(pop.get('G0-E02')!), 'liked > disliked');
  const before = fitness(pop.get('G0-E04')!);
  pop.recordView('G0-E04', 3, true);
  check('fitness.soft-dislike', pop.get('G0-E04')!.softDislikes === 1 && fitness(pop.get('G0-E04')!) < before, 'skip within 5 s lowers fitness');
  pop.recordView('G0-E05', 70, false);
  check('fitness.weak-like', pop.get('G0-E05')!.weakLikes === 1, 'watch > 60 s counts');
}

// -------------------------------------------------------- 9. serialization, cull

{
  const pop = Population.seeded();
  const p1 = pop.get('G0-E01')!;
  const p2 = pop.get('G0-E02')!;
  pop.addChild(crossover(p1.genome, p2.genome, mulberry32(7)), [p1, p2], Date.now(), 'fused');
  pop.addChild(randomGenome(mulberry32(8)), [p1, p2]);
  const j1 = pop.toJSON();
  const pop2 = Population.fromJSON(JSON.parse(JSON.stringify(j1)));
  const norm = (d: ReturnType<Population['toJSON']>) => ({ ...d, members: [...d.members].sort((a, b) => a.id.localeCompare(b.id)) });
  check('serialization.roundtrip-equal', JSON.stringify(norm(j1)) === JSON.stringify(norm(pop2.toJSON())), 'toJSON -> JSON -> fromJSON -> toJSON matches');
  check('serialization.version', j1.version === POPULATION_VERSION && POPULATION_VERSION === 6, `version=${j1.version}`);
  check('serialization.counter', pop2.addChild(randomGenome(mulberry32(9)), [p1, p2]).id === 'G1-0003', 'counter preserved');
  let threw = false;
  try {
    Population.fromJSON({ not: 'a population' });
  } catch {
    threw = true;
  }
  check('serialization.rejects-garbage', threw, 'non-population refused');
  const withBad = { ...j1, members: [...j1.members, { id: 'G0-BAD', gen: 0, parents: [], created: 1, name: 'Bad', genome: { v: 3, chain: 'no' }, likes: 0 }, { id: 'G5-0009', genome: { v: 3, chain: [], bodies: [], carrier: {}, color: {} } }, { id: 'nope' }, null] };
  const pop3 = Population.fromJSON(JSON.parse(JSON.stringify(withBad)));
  check('serialization.drops-invalid-members', pop3.size === j1.members.length, `size=${pop3.size} expected=${j1.members.length}`);

  const big = Population.seeded();
  const oldNow = Date.now() - 700_000;
  for (let i = 0; i < 200; i++) big.addChild(randomGenome(mulberry32(50000 + i)), [p1, p2], oldNow);
  big.cull(150);
  check('cull.leaves-150-keeps-seeds', big.size === 150 && SEEDS.every((s) => big.get(`G0-${s.origin}`)), `size=${big.size}`);
}

// ------------------------------------------------------ 10. migration (format 2 -> 3)

/** A format-2 genome as the previous code stored it. */
function v2Genome(emitters: unknown[], extra: Record<string, unknown> = {}): Record<string, unknown> {
  return {
    v: 2, chain: [{ op: 'rotate', stage: 'warp', w: 1, p: { lock: 0.25, rate: 0, cx: 0, cy: 0, alt: 0, wander: 0 } }], emitters,
    carrier: { kind: 'warp', p: { halfLife: 0.5, floor: 1, blur: 0, amount: 1, vort: 28, fnoise: 0.35, fscale: 2, famt: 0.0012 } },
    color: { scheme: 'triad', p: { hue: 0.5, sat: 0.9, exposure: 1, contrast: 0.03, bloom: 1, adapt: 0.3, vignette: 0.45, ca: 0.0015, reflect: 0, reflectY: -0.16, tonemap: 0 } },
    reactions: [], energy: [0.2, 0.7], ...extra,
  };
}
const em = (kind: string, p: Record<string, number> = {}, layer = 'fb') => ({ kind, layer, p: { ...defs(V2_SCHEMAS[kind as keyof typeof V2_SCHEMAS]), ...p } });

{
  // Every old emitter kind converts to a valid body expressing the same idea.
  const expect: Record<string, (b: BodyGene) => boolean> = {
    wave: (b) => b.shape.kind === 'curve' && b.material.kind === 'line',
    spectrum: (b) => b.shape.kind === 'bars',
    particles: (b) => b.emit.kind === 'sparks' && b.emit.p.body === 0,
    stars: (b) => b.shape.kind === 'dot' && b.place.kind === 'grid' && b.material.kind === 'glow',
    ink: (b) => b.shape.kind === 'dot' && (b.place.kind === 'stations' || b.place.kind === 'orbit' || b.place.kind === 'row'),
    wire: (b) => b.shape.kind === 'solid' && b.motion.kind === 'spin',
    plasma: (b) => b.shape.kind === 'plasma',
    aurora: (b) => b.shape.kind === 'aurora',
    blobs: (b) => b.place.kind === 'float' && b.material.kind === 'chrome',
    flame: (b) => b.shape.kind === 'flame' && (b.shape.xforms?.length ?? 0) > 0,
    edge: (b) => b.shape.kind === 'edge',
    tiles: (b) => b.shape.kind === 'polygon' && b.place.kind === 'grid',
    horizon: (b) => b.shape.kind === 'terrain',
    orb: (b) => b.shape.kind === 'dot' && b.material.kind === 'textured' && b.deform.kind === 'arms',
    snake: (b) => b.place.kind === 'walker' && b.emit.kind === 'cover',
  };
  const bad: string[] = [];
  for (const [kind, ok] of Object.entries(expect)) {
    const p = kind === 'orb' ? { arms: 1 } : kind === 'ink' ? { orbit: 1 } : {};
    const g = repair(v2Genome([em(kind, p)]));
    const errs = validate(g);
    if (errs.length) bad.push(`${kind}:${errs.join(';')}`);
    else if (!ok(g.bodies[0])) bad.push(`${kind}:${JSON.stringify(g.bodies[0]).slice(0, 120)}`);
  }
  check('migrate.every-emitter-kind', !bad.length, bad.join(' | ') || `${Object.keys(expect).length} old emitter kinds convert to their sub-genes`);

  // Merges become fused bodies; the old draw chain becomes deform ops; reactions follow their parameter.
  const merged = repair(v2Genome([{ kind: 'merge', layer: 'fb', p: { ...defs(V2_SCHEMAS.merge), mode: 0 }, parts: [em('orb', { arms: 1 }), em('wire')] }], {
    draw: [{ op: 'swirl', stage: 'warp', w: 1, p: defs({ amt: { def: 0.01 }, k: { def: 6 }, cx: { def: 0 }, cy: { def: 0 }, wander: { def: 0 } }) }],
    reactions: [{ src: 'bass', g: 'em', i: 0, k: 'k', gain: 0.5 }, { src: 'beat', g: 'em', i: 1, k: 'radius', gain: 0.3 }, { src: 'loud', g: 'em', i: 2, k: 'scale', gain: 0.2 }, { src: 'drums', g: 'dr', i: 0, k: 'amt', gain: 0.4 }],
  }));
  const mb = merged.bodies[0];
  check('migrate.merge-to-fuse', !validate(merged).length && mb.shape.kind === 'dot' && mb.fuse?.shape.kind === 'solid' && mb.fuse.p.mode === 0 && mb.deform.kind === 'arms', validate(merged).join(';') || `${mb.shape.kind}+${mb.fuse?.shape.kind}`);
  check('migrate.draw-to-deform-ops', mb.deform.ops?.[0]?.op === 'swirl', `${mb.deform.ops?.map((o) => o.op)}`);
  const rs = merged.reactions.map((r) => `${r.g}${r.i}.${r.k}`).sort().join(',');
  check('migrate.reactions-follow', rs === 'dr0.amt,fs0.size,fu0.k,sh0.r', rs);
  const mask = repair(v2Genome([{ kind: 'merge', layer: 'fb', p: { ...defs(V2_SCHEMAS.merge), mode: 2 }, parts: [em('orb'), em('particles')] }]));
  check('migrate.mask-merge', !validate(mask).length && mask.bodies[0].emit.kind === 'sparks' && mask.bodies[0].fuse?.shape.kind === 'dot' && mask.bodies[0].fuse.p.mode === 2, JSON.stringify(mask.bodies[0].fuse?.p));

  // upgradeV2 is what repair() uses; a format-1 genome converts too.
  const v1 = { ...v2Genome([em('ink'), em('orb', { arms: 1 }, 'top'), em('stars')]), v: 1 };
  const g1 = repair(v1);
  check('migrate.format-1', !validate(g1).length && g1.bodies.length === 3 && g1.bodies[1].emit.kind === 'none', g1.bodies.map((b) => `${b.shape.kind}/${b.place.kind}/${b.emit.kind}`).join(','));
  check('migrate.upgrade-direct', JSON.stringify(repair(upgradeV2(repairV2(v1)))) === JSON.stringify(g1), 'repair(old) === repair(upgradeV2(repairV2(old)))');

  // A version-3 population file: seeds replaced by the new genomes keeping ids / votes / views, bred children
  // converted keeping ids / votes / parents / names, nothing crashes on junk.
  const seeds = SEEDS.map((s) => ({
    id: `G0-${s.origin}`, gen: 0, origin: s.origin, parents: [], created: 1, name: s.name,
    genome: v2Genome([em('wave')]), likes: s.origin === 'E02' ? 4 : 0, dislikes: s.origin === 'E02' ? 1 : 0, softDislikes: 0, weakLikes: 0,
    views: s.origin === 'E02' ? 9 : 0, watch: s.origin === 'E02' ? 321.5 : 0, hidden: s.origin === 'E12', descriptor: [0.1, 0.2],
  }));
  const kids = [
    { id: 'G1-0001', gen: 1, parents: ['G0-E02', 'G0-E14'], created: 2, name: 'Wheeling Serpent', genome: v2Genome([em('snake')], { draw: [{ op: 'twist', stage: 'warp', w: 1, p: {} }] }), likes: 3, dislikes: 0, softDislikes: 0, weakLikes: 1, views: 5, watch: 99, hidden: false, cross: 'fused', descriptor: [0.3, 0.3] },
    { id: 'G2-0002', gen: 2, parents: ['G1-0001', 'G0-E20'], created: 3, name: 'Starfish Pearl Cage', genome: v2Genome([{ kind: 'merge', layer: 'top', p: { ...defs(V2_SCHEMAS.merge), mode: 1 }, parts: [em('orb', { arms: 1 }), em('wire')] }]), likes: 1, dislikes: 2, softDislikes: 0, weakLikes: 0, views: 2, watch: 10, hidden: false, cross: 'merged' },
    { id: 'G3-0003', gen: 3, parents: ['G2-0002'], created: 4, name: 'Old Flame', genome: { ...v2Genome([em('flame'), em('particles', {}, 'top')]), v: 1 }, likes: 0, dislikes: 0, softDislikes: 0, weakLikes: 0, views: 0, watch: 0, hidden: true },
    { id: 'G3-0004', gen: 3, parents: [], created: 4, name: 'Junk', genome: { v: 2, chain: [], emitters: 'x', carrier: {}, color: {} } },
  ];
  const file = { format: 'musicvis-v2-population', version: 3, seedVersion: 2, counter: 4, votesSinceBreed: 2, members: [...seeds, ...kids] };
  let pop: Population | null = null;
  let err = '';
  try {
    pop = Population.fromJSON(JSON.parse(JSON.stringify(file)));
  } catch (e) {
    err = (e as Error).message;
  }
  check('migrate.v3-file-loads', !!pop && pop.size === SEEDS.length + 3, err || `size=${pop?.size}`);
  if (pop) {
    const all = pop.list();
    check('migrate.all-valid', all.every((m) => !validate(m.genome).length), all.filter((m) => validate(m.genome).length).map((m) => m.id).join(',') || 'every member valid');
    const k1 = pop.get('G1-0001')!, k2 = pop.get('G2-0002')!, k3 = pop.get('G3-0003')!;
    check('migrate.children-kept', k1.name === 'Wheeling Serpent' && k2.name === 'Starfish Pearl Cage' && k3.name === 'Old Flame' && k1.likes === 3 && k2.dislikes === 2 && k3.hidden && k1.cross === 'fused' && JSON.stringify(k2.parents) === JSON.stringify(['G1-0001', 'G0-E20']), `${k1.name} / ${k2.name} / ${k3.name}`);
    check('migrate.children-faithful', k1.genome.bodies[0].place.kind === 'walker' && k1.genome.bodies[0].deform.ops?.[0].op === 'twist' && k2.genome.bodies[0].fuse?.shape.kind === 'solid' && k2.genome.bodies[0].emit.kind === 'none' && k3.genome.bodies.length === 2, structuralKey(k2.genome));
    check('migrate.descriptors-dropped', k1.descriptor === undefined, 'converted children are measured again');
    const changed = pop.upgradeSeeds();
    const e02 = pop.get('G0-E02')!;
    check('migrate.seeds-replaced', changed.length === SEEDS.length && JSON.stringify(e02.genome) === JSON.stringify(seedByOrigin('E02')) && pop.seedVersion === SEED_VERSION, `${changed.length} seeds upgraded`);
    check('migrate.seed-votes-kept', e02.likes === 4 && e02.dislikes === 1 && e02.views === 9 && e02.watch === 321.5 && pop.get('G0-E12')!.hidden, `E02 ${e02.likes}/${e02.dislikes} views=${e02.views}`);
    check('migrate.parent-links', pop.list().every((m) => m.parents.every((id) => pop!.get(id))), 'parents resolve');
    check('migrate.idempotent', pop.upgradeSeeds().length === 0, 'second upgrade changes nothing');
    const again = Population.fromJSON(JSON.parse(JSON.stringify(pop.toJSON())));
    check('migrate.stable', JSON.stringify(again.get('G2-0002')!.genome) === JSON.stringify(k2.genome) && again.get('G2-0002')!.name === k2.name, 'a saved v4 file reloads unchanged');
    check('migrate.renders', all.every((m) => buildSources(m.genome).feedback.includes('void main') && estimateCost(m.genome) < COST_BUDGET_MS), 'every converted member builds shaders under budget');
  }
  // Version 1-2 files still get the descriptive rename of their bred children.
  const oldFile = { ...file, version: 1 };
  const popOld = Population.fromJSON(JSON.parse(JSON.stringify(oldFile)));
  check('migrate.v1-file-renamed', popOld.get('G1-0001')!.name !== 'Wheeling Serpent' && popOld.get('G1-0001')!.likes === 3, popOld.get('G1-0001')!.name);
  let threw = false;
  try {
    Population.fromJSON({ ...file, version: 99 });
  } catch {
    threw = true;
  }
  check('migrate.newer-file-rejected', threw, 'refused');
}

// ------------------------------------------------------ 11. names

const nounOf = (name: string) => name.split(/\s+/).slice(1).join(' ');
const adjOf = (name: string) => name.split(/\s+/)[0];

{
  const rng = mulberry32(4242);
  let hueHit = '';
  for (let i = 0; i < 600 && !hueHit; i++) {
    const g = i % 2 === 0 ? randomGenome(rng) : mutate(crossover(SEEDS[i % 24].genome, SEEDS[(i * 5) % 24].genome, rng), rng, 1);
    const bad = nameFor(g).split(/\s+/).find((w) => HUE_WORDS.includes(w));
    if (bad) hueHit = `${bad} in ${nameFor(g)}`;
  }
  check('naming.no-hue-words', !hueHit, hueHit || '600 names, none hue-based');

  // Nouns come from the body's shape / placement / emission.
  const expectNoun: [string, string][] = [['E02', 'snake'], ['E13', 'stars'], ['E10', 'tiles'], ['E14', 'wire'], ['E15', 'blobs'], ['E20', 'orb'], ['E06', 'ink'], ['E12', 'particles'], ['E22', 'flame'], ['E07', 'wave'], ['E11', 'spectrum']];
  const nbad = expectNoun.filter(([o, k]) => nounKind(b0(o)) !== k || !NOUN_POOLS[k as keyof typeof NOUN_POOLS].includes(nounOf(nameFor(seedByOrigin(o)))));
  check('naming.body-nouns', !nbad.length, nbad.map(([o]) => `${o}:${nounKind(b0(o))}:${nameFor(seedByOrigin(o))}`).join(', ') || `${expectNoun.length} seeds name their body`);
  // A walker placement makes a dot a serpent, a grid makes it a starfield.
  const cage = cloneGenome(seedByOrigin('E14'));
  cage.bodies[0].place = { kind: 'walker', p: defs(locusSchema('place', 'walker')) } as never;
  check('naming.placement-changes-noun', nounKind(repair(cage).bodies[0]) === 'wire', nameFor(repair(cage)));
  const walkerDot = cloneGenome(seedByOrigin('E20'));
  walkerDot.bodies[0].place = { kind: 'walker', p: defs(locusSchema('place', 'walker')) } as never;
  check('naming.walking-dot-serpent', nounKind(repair(walkerDot).bodies[0]) === 'snake', nameFor(repair(walkerDot)));

  // Adjectives come from motion / deformation / material traits.
  const armed = nameFor(seedByOrigin('E20'));
  check('naming.arms-adjective', ADJ_POOLS.reaching.includes(adjOf(armed)) || ADJ_POOLS.breathing.includes(adjOf(armed)), armed);
  const spin = cloneGenome(seedByOrigin('E06'));
  spin.bodies[0].motion = { kind: 'hits', p: { amt: 1.5 } };
  spin.bodies[0].deform = { kind: 'wobble', p: { lobes: 5, amp: 0.4, rate: 0.125 } };
  const spinName = nameFor(repair(spin));
  check('naming.motion-deform-adjective', ADJ_POOLS.darting.includes(adjOf(spinName)) || ADJ_POOLS.rippling.includes(adjOf(spinName)), spinName);
  const swirl = repair({ v: 3, chain: [{ op: 'swirl', stage: 'warp', w: 1, p: { amt: 0.03, k: 6, cx: 0, cy: 0, wander: 0 } }], bodies: [{ shape: { kind: 'curve' }, material: { kind: 'line' } }] });
  check('naming.swirl-adjective', ADJ_POOLS.spiral.includes(adjOf(nameFor(swirl))), nameFor(swirl));
  const fused = makeFuse(b0('E20'), b0('E14').shape, mulberry32(3), 0)!;
  const fusedName = nameFor(repair({ ...cloneGenome(seedByOrigin('E20')), bodies: [fused] }));
  check('naming.fused-two-nouns', fusedName.split(/\s+/).length === 3 && NOUN_POOLS.wire.includes(fusedName.split(/\s+/)[2]), fusedName);

  const detG = mutate(randomGenome(mulberry32(555)), mulberry32(556), 1);
  check('naming.deterministic', nameFor(detG, ['Rising Moon', 'Hushed Serpent']) === nameFor(detG, ['Rising Moon', 'Hushed Serpent']), nameFor(detG));
  // A parent's noun survives while the child's body still reads that way.
  const serp = 'Hushed Viper';
  let kept = 0, tot = 0;
  for (let i = 0; i < 200; i++) {
    const c = crossover(seedByOrigin('E02'), seedByOrigin('E13'), mulberry32(700 + i), 1);
    if (nounKind(c.bodies[0]) !== 'snake') continue;
    tot++;
    if (nounOf(nameFor(c, [serp, 'Quiet Starfield'])) === 'Viper') kept++;
  }
  check('naming.inherits-noun', tot > 0 && kept === tot, `${kept}/${tot} serpent children kept "Viper"`);
  const used = new Set(['Calm Orb']);
  const dup1 = uniqueName(used, 'Calm Orb');
  used.add(dup1);
  check('naming.dedupe', dup1 === 'Calm Orb II' && uniqueName(used, 'Calm Orb') === 'Calm Orb III', dup1);
}

/** A format-5 genome written back as format 3 (one colour gene, material hues, no feel / mapping loci). */
function toV3(g: Genome): Record<string, unknown> & { bodies: Record<string, unknown>[]; reactions: Record<string, unknown>[] } {
  const o = JSON.parse(JSON.stringify(g));
  o.v = 3;
  o.color = { scheme: o.palette.kind, p: { ...o.tone.p, hue: o.palette.p.hue } };
  delete o.palette;
  delete o.tone;
  for (const b of o.bodies) {
    b.material.p.hue = b.color.p.hue;
    delete b.color;
  }
  return o;
}

// ------------------------------------------------------ 12. feel genes

{
  // A format-3 genome (no feel, reactions without curves) gets the neutral feel: instant response,
  // bar clock, grid-locked, i.e. exactly how it behaved.
  const v3 = toV3(seedByOrigin('E09'));
  for (const b of v3.bodies) delete b.feel;
  v3.reactions = v3.reactions.map((r) => ({ src: r.src, g: r.g, i: r.i, k: r.k, gain: r.gain }));
  const g = repair(v3);
  const f = g.bodies[0].feel;
  check('feel.format3-neutral', !validate(g).length && g.v === 5 && f.kind === 'flow' && f.p.atk === 0.005 && f.p.rel === 0.005 && f.p.thr === 0 && f.p.sens === 1 && f.p.div === 4 && f.p.lock === 1
    && g.reactions.every((r) => r.atk === 0.005 && r.rel === 0.005 && r.q === 0 && r.thr === 0), JSON.stringify(f.p));
  check('feel.format3-reactions-kept', JSON.stringify(g.reactions.map((r) => [r.src, r.g, r.i, r.k, r.gain])) === JSON.stringify(seedByOrigin('E09').reactions.map((r) => [r.src, r.g, r.i, r.k, r.gain])), `${g.reactions.length} reactions`);

  // One parent's look with the other's feel (polyhedra's calm clock on ink, and the reverse).
  const e14f = b0('E14').feel.p, e06f = b0('E06').feel.p;
  let inkCalm = 0, polyLively = 0;
  for (let i = 0; i < 400; i++) {
    const c = crossover(seedByOrigin('E06'), seedByOrigin('E14'), mulberry32(60000 + i));
    const b = c.bodies[0];
    if (b.shape.kind === 'dot' && b.place.kind === 'stations' && b.feel.p.rel === e14f.rel && b.feel.p.atk === e14f.atk) inkCalm++;
    if (b.shape.kind === 'solid' && b.feel.p.rel === e06f.rel && b.feel.p.atk === e06f.atk) polyLively++;
  }
  check('feel.look-and-feel-split', inkCalm > 10 && polyLively > 10, `${inkCalm} ink children with polyhedra's feel, ${polyLively} polyhedra children with ink's feel (of 400)`);

  // Feel mutations and reaction curves stay valid and in range.
  const rng = mulberry32(61000);
  const seen = new Set<string>();
  const bad: string[] = [];
  for (let i = 0; i < 1500; i++) {
    const log: string[] = [];
    const m = mutate(SEEDS[i % 24].genome, rng, 1 + rng(), log);
    log.forEach((x) => seen.add(x));
    if (validate(m).length) bad.push(validate(m).join(';'));
  }
  check('feel.mutations', !bad.length && ['swap-feel', 'change-clock', 'rewire-reaction', 'reaction-curve'].every((x) => seen.has(x)), bad[0] ?? 'feel swaps, clock changes, rewired sources and curve changes all valid');
  const rg = randomGenome(mulberry32(62000));
  check('feel.random-curves', rg.reactions.every((r) => r.atk >= 0.005 && r.rel >= 0.005 && (r.q === 0 || r.q === 1)), JSON.stringify(rg.reactions[0] ?? {}));

  // Names read the feel: a stepped body, a calm clock.
  const st = cloneGenome(seedByOrigin('E06'));
  st.bodies[0].feel = { kind: 'step', p: { atk: 0.005, rel: 0.005, thr: 0, sens: 1, div: 4, lock: 1 } };
  const stName = nameFor(repair(st));
  const calm = cloneGenome(seedByOrigin('E06'));
  calm.bodies[0].feel = { kind: 'flow', p: { atk: 0.2, rel: 2.5, thr: 0, sens: 1, div: 16, lock: 1 } };
  const calmName = nameFor(repair(calm));
  check('feel.names', ADJ_POOLS.stepped.includes(adjOf(stName)) || stName !== nameFor(seedByOrigin('E06')), `${stName} / ${calmName}`);
}

// ------------------------------------------------------ 13. colour in parts

{
  // Format 3 / 4 colour converts: palette = scheme + hue, tone = the rest, mapping from the placement.
  const v3 = toV3(seedByOrigin('E13'));
  v3.reactions = [{ src: 'melody', g: 'col', i: 0, k: 'hue', gain: 0.3 }, { src: 'bass', g: 'ma', i: 0, k: 'hue', gain: 0.2 }, { src: 'beat', g: 'col', i: 0, k: 'bloom', gain: 0.3 }];
  const g = repair(v3);
  check('colour.format3-converts', !validate(g).length && JSON.stringify(g.palette) === JSON.stringify(seedByOrigin('E13').palette) && JSON.stringify(g.tone) === JSON.stringify(seedByOrigin('E13').tone) && g.bodies[0].color.kind === 'pitch', validate(g).join(';') || JSON.stringify(g.palette));
  check('colour.format3-reactions', g.reactions.map((r) => `${r.g}.${r.k}`).join(',') === 'pal.hue,cm.hue,col.bloom', g.reactions.map((r) => `${r.g}.${r.k}`).join(','));
  // The original seeds' mappings are the ones their placement implies (the M seeds choose theirs).
  const back = [SEEDS.filter((s) => s.origin.startsWith('E')).every((s) => JSON.stringify(repair(toV3(s.genome)).bodies.map((b) => b.color.kind)) === JSON.stringify(s.genome.bodies.map((b) => b.color.kind)))];
  check('colour.seed-mappings-implied', back[0], SEEDS.map((s) => `${s.origin}:${s.genome.bodies[0].color.kind}`).join(' '));

  // Palette, mapping and tone travel separately in crossover.
  let palFromR = 0, mapFromR = 0, both = 0;
  for (let i = 0; i < 300; i++) {
    const c = crossover(seedByOrigin('E13'), seedByOrigin('E02'), mulberry32(70000 + i), 1);
    const shapeDot = c.bodies[0].place.kind === 'grid';
    if (!shapeDot) continue;
    const pr = c.palette.kind === seedByOrigin('E02').palette.kind && c.palette.kind !== seedByOrigin('E13').palette.kind;
    const mr = c.bodies[0].color.kind === 'instrument';
    if (pr) palFromR++;
    if (mr) mapFromR++;
    if (pr !== mr) both++;
  }
  check('colour.parts-split', palFromR > 10 && mapFromR > 10 && both > 10, `grid children: ${palFromR} with the walker's palette, ${mapFromR} with its instrument mapping, ${both} with one but not the other`);

  // Every mapping kind builds and names without hue words.
  const rng = mulberry32(71000);
  const bad: string[] = [];
  for (const kind of LOCUS_KINDS.color) {
    for (let k = 0; k < 8; k++) {
      const b = randomBody(rng);
      b.color = randomGene('color', rng, kind) as BodyGene['color'];
      const gg = repair({ ...cloneGenome(seedByOrigin('E05')), bodies: [b] });
      if (validate(gg).length) bad.push(`${kind}:${validate(gg).join(';')}`);
      if (!buildSources(gg).feedback.includes('CHUE_0') && !buildSources(gg).composite.includes('CHUE_0') && SHAPE_CLASS[b.shape.kind] === 'sdf') bad.push(`${kind}:no mapping code`);
      if (nameFor(gg).split(/\s+/).some((w) => HUE_WORDS.includes(w))) bad.push(`${kind}:hue word`);
    }
  }
  check('colour.every-mapping', !bad.length, bad.slice(0, 3).join(' | ') || `${LOCUS_KINDS.color.length} mapping kinds build, validate and name cleanly`);
  const hn = cloneGenome(seedByOrigin('E07'));
  hn.bodies[0].color = { kind: 'height', p: { hue: 0, detail: 1, amount: 1.5 } };
  check('colour.mapping-name', ADJ_POOLS.graded.includes(adjOf(nameFor(repair(hn)))) || nameFor(repair(hn)) !== nameFor(seedByOrigin('E07')), nameFor(repair(hn)));
}

// ------------------------------------------------------ 14. homologous loci and linkage

{
  // Placement and motion travel together; a split lets them part.
  let pr = 0, prm = 0, pd = 0, pdm = 0, sr = 0, sdm = 0;
  const rng = mulberry32(80000);
  for (let i = 0; i < 4000; i++) {
    const a = SEEDS[i % 24].genome, b = SEEDS[(i * 7 + 5) % 24].genome;
    const [ab, bb] = [a.bodies[0], b.bodies[0]];
    if (ab.place.kind === bb.place.kind || ab.motion.kind === bb.motion.kind) continue;
    const c = crossover(a, b, rng, 1).bodies[0];
    const domA = c.shape.kind === ab.shape.kind;
    const [D, R] = domA ? [ab, bb] : [bb, ab];
    if (c.place.kind === R.place.kind) {
      pr++;
      if (c.motion.kind === R.motion.kind) prm++;
    } else if (c.place.kind === D.place.kind) {
      pd++;
      if (c.motion.kind === R.motion.kind) pdm++;
    }
    if (c.material.kind === R.material.kind && ab.material.kind !== bb.material.kind) {
      sr++;
      if (c.color.kind === R.color.kind || R.color.kind === D.color.kind) sdm++;
    }
  }
  const linked = prm / Math.max(1, pr), unlinked = pdm / Math.max(1, pd);
  check('linkage.place-motion', linked > 0.6 && unlinked < 0.25 && prm < pr, `motion follows a recessive placement ${(linked * 100).toFixed(0)}% (${prm}/${pr}) vs ${(unlinked * 100).toFixed(0)}% otherwise; splits still happen`);
  check('linkage.material-colour', sr > 0 && sdm / sr > 0.6, `${sdm}/${sr} recessive materials brought their colour mapping`);

  // Reactions pair by target: never two reactions on the same parameter slot from crossover.
  let dup = 0, n = 0;
  for (let i = 0; i < 1500; i++) {
    const c = crossover(SEEDS[i % 24].genome, SEEDS[(i * 11 + 3) % 24].genome, rng);
    n += c.reactions.length;
    const keys = c.reactions.map((r) => `${r.g}${r.i}.${r.k}`);
    if (new Set(keys).size !== keys.length) dup++;
  }
  check('homology.reactions', dup === 0, `${n} inherited reactions, no target driven twice`);
  const e09 = seedByOrigin('E09');
  let keep = 0;
  for (let i = 0; i < 200; i++) {
    const c = crossover(e09, seedByOrigin('E04'), mulberry32(81000 + i), 3);
    if (c.bodies[0].place.kind === 'stations' && c.reactions.some((r) => r.g === 'ma' && r.k === 'gain')) keep++;
  }
  check('homology.reaction-follows-locus', keep > 0, `${keep} E09-bodied children kept the beat -> material gain reaction`);
}

// ------------------------------------------------------ 15. silent (recessive) alleles

{
  // Many generations stay valid; alleles are carried but stay compact.
  const rng = mulberry32(90000);
  let pop: Genome[] = SEEDS.map((s) => s.genome);
  const bad: string[] = [];
  let carriers = 0, alleles = 0, bodies = 0, bytesAlt = 0, bytes = 0;
  for (let gen = 0; gen < 6; gen++) {
    const next: Genome[] = [];
    for (let i = 0; i < 60; i++) {
      let c = crossover(pop[Math.floor(rng() * pop.length)], pop[Math.floor(rng() * pop.length)], rng);
      if (rng() < 0.3) c = mutate(c, rng, 0.6);
      if (validate(c).length) bad.push(validate(c).join(';'));
      for (const b of c.bodies) {
        bodies++;
        if (b.alt) {
          carriers++;
          alleles += Object.keys(b.alt).length;
          bytesAlt += JSON.stringify(b.alt).length;
        }
      }
      bytes += JSON.stringify(c).length;
      next.push(c);
    }
    pop = next;
  }
  check('allele.generations-valid', !bad.length, bad[0] ?? '6 generations x 60 children valid');
  check('allele.compact', carriers > bodies * 0.3 && bytesAlt / bytes < 0.35, `${carriers}/${bodies} bodies carry ${alleles} silent alleles, ${(100 * bytesAlt / bytes).toFixed(0)}% of the serialized size`);
  const mseen = new Set<string>();
  const mbad: string[] = [];
  for (let i = 0; i < 1500; i++) {
    const log: string[] = [];
    const m = mutate(pop[i % pop.length], rng, 1.5, log);
    log.forEach((x) => mseen.add(x));
    if (validate(m).length) mbad.push(validate(m).join(';'));
  }
  check('allele.mutations', !mbad.length && mseen.has('express-allele') && mseen.has('drop-allele'), mbad[0] ?? 'express-allele / drop-allele keep carriers valid');
  const rt = pop.find((g) => g.bodies[0].alt)!;
  check('allele.roundtrip', JSON.stringify(repair(JSON.parse(JSON.stringify(rt)))) === JSON.stringify(rt) && structuralKey(rt) === structuralKey({ ...rt, bodies: rt.bodies.map((b) => ({ ...b, alt: undefined })) } as Genome), 'alleles survive JSON + repair and never change the shaders');
  const same = repair({ ...cloneGenome(seedByOrigin('E02')), bodies: [{ ...cloneGenome(seedByOrigin('E02')).bodies[0], alt: { place: { kind: 'walker', p: {} }, motion: { kind: 'spin', p: { rate: 9 } }, shape: { kind: 'flame', p: {} } } }] });
  check('allele.repair', JSON.stringify(Object.keys(same.bodies[0].alt ?? {})) === '["motion"]' && same.bodies[0].alt!.motion!.p.rate === 1, JSON.stringify(same.bodies[0].alt));

  // A trait silent in both parents resurfaces in grandchildren: a still polyhedron child that carries
  // the walk, crossed back with polyhedra, sometimes walks again.
  let carrier: Genome | null = null;
  for (let i = 0; i < 2000 && !carrier; i++) {
    const c = crossover(seedByOrigin('E14'), seedByOrigin('E02'), mulberry32(91000 + i), 3);
    if (c.bodies[0].place.kind !== 'walker' && c.bodies[0].alt?.place?.kind === 'walker') carrier = c;
  }
  let back = 0;
  if (carrier) {
    for (let i = 0; i < 600; i++) {
      const gc = crossover(carrier, seedByOrigin('E14'), mulberry32(92000 + i), 0);
      if (gc.bodies[0].place.kind === 'walker') back++;
    }
  }
  check('allele.resurfaces', !!carrier && back > 0, carrier ? `${back}/600 grandchildren walk again (neither parent shows the walk)` : 'no carrier found');
}

// ------------------------------------------------------------- 17. gene editor

{
  // 1. Every locus/kind combo is reachable from at least one seed and modeled with the right controls.
  const bad: string[] = [];
  let n = 0;
  for (const locus of LOCI) {
    for (const kind of LOCUS_KINDS[locus]) {
      n++;
      let found: Genome | null = null;
      for (const s of SEEDS) {
        const r = switchKind(s.genome, { t: 'locus', b: 0, locus }, kind);
        if (r.ok) { found = r.genome; break; }
      }
      if (!found) { bad.push(`${locus}=${kind}: no seed accepted this switch`); continue; }
      const model = buildModel(found);
      const sec = model.find((m) => m.id === `b0.${locus}`);
      const expectedKeys = Object.keys(locusSchema(locus, kind));
      if (!sec) { bad.push(`${locus}=${kind}: no section b0.${locus} in the model`); continue; }
      if (sec.kind?.value !== kind) bad.push(`${locus}=${kind}: section kind.value=${sec.kind?.value}`);
      const gotKeys = sec.params.map((p) => p.key);
      if (JSON.stringify(gotKeys) !== JSON.stringify(expectedKeys)) bad.push(`${locus}=${kind}: params ${gotKeys.join(',')} != ${expectedKeys.join(',')}`);
    }
  }
  check('editor.model-covers-loci', !bad.length, bad.slice(0, 12).join(' | ') || `all ${n} locus/kind combos reachable from a seed and modeled`);
}

{
  // 2. Every control across every seed's model has the right widget, and genome-wide sections exist.
  const bad: string[] = [];
  const checkParams = (params: ParamControl[], where: string) => {
    for (const p of params) {
      if (p.spec.choices) {
        const wantWidget = p.spec.choices.length <= 4 ? 'segmented' : 'select';
        if (p.widget !== wantWidget) bad.push(`${where}.${p.key}: widget=${p.widget} expected ${wantWidget} (${p.spec.choices.length} choices)`);
        if (!p.options || p.options.length !== p.spec.choices.length) bad.push(`${where}.${p.key}: options.length=${p.options?.length} != ${p.spec.choices.length}`);
      } else if (p.widget !== 'slider') bad.push(`${where}.${p.key}: widget=${p.widget} expected slider`);
    }
  };
  for (const s of SEEDS) {
    const model = buildModel(s.genome);
    const ids = new Set(model.map((m) => m.id));
    for (const id of ['palette', 'tone', 'carrier', 'chain', 'reactions', 'energy']) {
      if (!ids.has(id)) bad.push(`${s.origin}: missing genome-wide section "${id}"`);
    }
    for (const sec of model) {
      checkParams(sec.params, `${s.origin}.${sec.id}`);
      sec.items?.forEach((it) => checkParams(it.params, `${s.origin}.${sec.id}.${it.id}`));
    }
  }
  check('editor.model-widgets', !bad.length, bad.slice(0, 12).join(' | ') || `every control across ${SEEDS.length} seeds has the right widget; genome-wide sections present`);
}

{
  // 3. switchKind stays valid (or leaves the genome untouched with a reason) across seeds, loci, palette, carrier, chain ops.
  const bad: string[] = [];
  for (const s of SEEDS) {
    for (const locus of LOCI) {
      for (const kind of LOCUS_KINDS[locus]) {
        const r = switchKind(s.genome, { t: 'locus', b: 0, locus }, kind);
        if (r.ok) {
          if (validate(r.genome).length) bad.push(`${s.origin}.${locus}=${kind}: invalid ${validate(r.genome).join(';')}`);
          if (r.genome.bodies.length !== s.genome.bodies.length) bad.push(`${s.origin}.${locus}=${kind}: bodies length changed`);
          if ((r.genome.bodies[0][locus] as Gene).kind !== kind) bad.push(`${s.origin}.${locus}=${kind}: kind not applied`);
        } else {
          if (JSON.stringify(r.genome) !== JSON.stringify(s.genome)) bad.push(`${s.origin}.${locus}=${kind}: genome changed on failure`);
          if (!r.reason) bad.push(`${s.origin}.${locus}=${kind}: no reason on failure`);
        }
      }
    }
    for (const kind of PALETTE_KINDS) {
      const r = switchKind(s.genome, { t: 'palette' }, kind);
      if (r.ok) {
        if (validate(r.genome).length || r.genome.palette.kind !== kind) bad.push(`${s.origin}.palette=${kind}: bad result`);
      } else if (!r.reason) bad.push(`${s.origin}.palette=${kind}: no reason on failure`);
    }
    for (const kind of CARRIER_KINDS) {
      const r = switchKind(s.genome, { t: 'carrier' }, kind);
      if (r.ok) {
        if (validate(r.genome).length || r.genome.carrier.kind !== kind) bad.push(`${s.origin}.carrier=${kind}: bad result`);
      } else if (!r.reason) bad.push(`${s.origin}.carrier=${kind}: no reason on failure`);
    }
    if (s.genome.chain.length) {
      for (const kind of OP_KINDS) {
        const r = switchKind(s.genome, { t: 'op', j: 0 }, kind);
        if (r.ok) {
          if (validate(r.genome).length || r.genome.chain[0].op !== kind) bad.push(`${s.origin}.op0=${kind}: bad result`);
        } else if (!r.reason) bad.push(`${s.origin}.op0=${kind}: no reason on failure`);
      }
    }
  }
  check('editor.kind-switch-valid', !bad.length, bad.slice(0, 12).join(' | ') || `switchKind valid (or clean-refused) across ${SEEDS.length} seeds x loci/kinds, palette, carrier, chain ops`);
}

{
  // 4. Slider round-trip within one step for every non-choice ParamSpec, including log specs.
  const bad: string[] = [];
  const schemas: [string, Schema][] = [];
  for (const locus of LOCI) for (const kind of LOCUS_KINDS[locus]) schemas.push([`${locus}.${kind}`, locusSchema(locus, kind)]);
  schemas.push(['reaction', REACTION_SCHEMA], ['carrier', CARRIER_SCHEMA]);
  for (const [name, schema] of schemas) {
    for (const [key, spec] of Object.entries(schema) as [string, ParamSpec][]) {
      if (spec.choices) continue;
      const linStep = (spec.max - spec.min) / SLIDER_STEPS;
      const logRatio = spec.log && spec.min > 0 ? Math.pow(spec.max / spec.min, 1 / SLIDER_STEPS) - 1 : 0;
      for (const v of [spec.def, spec.min, spec.max]) {
        const back = fromSlider(toSlider(v, spec), spec);
        const localStep = spec.log && spec.min > 0 ? Math.max(v, spec.min) * logRatio : linStep;
        const tol = Math.max(localStep, spec.int ? 1 : 0) + 1e-6;
        if (Math.abs(back - v) > tol) bad.push(`${name}.${key}: v=${v} back=${back} tol=${tol.toFixed(5)}`);
        if (spec.int && !Number.isInteger(back)) bad.push(`${name}.${key}: int spec gave ${back}`);
      }
      if (spec.log && spec.min > 0) {
        const pMin = toSlider(spec.min, spec);
        const pMax = toSlider(spec.max, spec);
        if (pMin !== 0) bad.push(`${name}.${key}: log toSlider(min)=${pMin} != 0`);
        if (pMax !== SLIDER_STEPS) bad.push(`${name}.${key}: log toSlider(max)=${pMax} != ${SLIDER_STEPS}`);
        const pDef = toSlider(spec.def, spec);
        if (!(pDef >= 0 && pDef <= SLIDER_STEPS)) bad.push(`${name}.${key}: log toSlider(def)=${pDef} out of [0, ${SLIDER_STEPS}]`);
      }
    }
  }
  check('editor.slider-roundtrip', !bad.length, bad.slice(0, 12).join(' | ') || 'fromSlider(toSlider(v)) within one step for def/min/max of every non-choice spec, incl. log halfLife/atk/rel');
}

{
  // 5. setParam clamps out-of-range and non-integer input; energy keeps its gap; editParam repairs a broken count.
  const bad: string[] = [];
  for (const s of SEEDS) {
    for (const locus of LOCI) {
      const gene = s.genome.bodies[0][locus] as Gene;
      const schema = locusSchema(locus, gene.kind);
      const g = cloneGenome(s.genome);
      const t: Target = { t: 'locus', b: 0, locus };
      for (const key of Object.keys(schema)) {
        const spec = schema[key];
        const above = setParam(g, t, key, spec.max + 1000);
        if (above > spec.max + 1e-9 || above < spec.min - 1e-9) bad.push(`${s.origin}.${locus}.${key}: above-max not clamped (${above})`);
        const below = setParam(g, t, key, spec.min - 1000);
        if (below < spec.min - 1e-9 || below > spec.max + 1e-9) bad.push(`${s.origin}.${locus}.${key}: below-min not clamped (${below})`);
        if (spec.int) {
          const frac = setParam(g, t, key, spec.def + 0.37);
          if (!Number.isInteger(frac)) bad.push(`${s.origin}.${locus}.${key}: int spec gave ${frac}`);
        }
        if (validate(g).length) bad.push(`${s.origin}.${locus}.${key}: invalid after setParam ${validate(g).join(';')}`);
      }
    }
  }
  const eg = cloneGenome(SEEDS[0].genome);
  setParam(eg, { t: 'energy' }, 'lo', 0.95);
  const [elo, ehi] = eg.energy;
  if (!(ehi - elo >= 0.15 - 1e-9) || ehi > 1 + 1e-9) bad.push(`energy: setting lo=0.95 gave lo=${elo} hi=${ehi}`);
  if (validate(eg).length) bad.push(`energy: invalid after clamp ${validate(eg).join(';')}`);

  // A field shape (plasma) given a counted placement beyond its single-copy rule must be repaired back.
  const fieldBase = repair({
    v: 5, chain: [], bodies: [{ shape: { kind: 'plasma' }, place: { kind: 'stations', p: { count: 1 } }, emit: { kind: 'trail' } }],
    carrier: { kind: 'warp', p: {} }, palette: { kind: 'triad', p: {} }, tone: { p: {} }, reactions: [], energy: [0.2, 0.8],
  });
  if (validate(fieldBase).length) bad.push(`field base invalid: ${validate(fieldBase).join(';')}`);
  const countEdit = editParam(fieldBase, { t: 'locus', b: 0, locus: 'place' }, 'count', 6);
  if (validate(countEdit.genome).length) bad.push(`editParam count=6 on a field shape left it invalid: ${validate(countEdit.genome).join(';')}`);
  if (countEdit.genome.bodies[0].place.p.count !== 1) bad.push(`editParam count=6 on a field shape did not repair back to a single copy (count=${countEdit.genome.bodies[0].place.p.count})`);
  if (!countEdit.repaired) bad.push('editParam count=6 on a field shape did not report repaired=true');
  check('editor.setParam-clamps', !bad.length, bad.slice(0, 12).join(' | ') || 'setParam clamps range/int for every locus param, energy keeps its gap, editParam repairs a broken field-shape count');
}

{
  // 6. Chain edits: fill to MAX_CHAIN and cap, moveOp swaps and carries reactions, removeOp drops/reindexes, setStage rules.
  const bad: string[] = [];
  let g = freshGenome();
  let i = 0;
  while (g.chain.length < MAX_CHAIN) {
    const r = addOp(g, OP_KINDS[i % OP_KINDS.length]);
    if (!r.ok) { bad.push(`addOp failed before the cap at length ${g.chain.length}: ${r.reason}`); break; }
    if (validate(r.genome).length) bad.push(`chain.addOp: invalid ${validate(r.genome).join(';')}`);
    g = r.genome;
    i++;
  }
  if (g.chain.length !== MAX_CHAIN) bad.push(`chain length ${g.chain.length} != MAX_CHAIN ${MAX_CHAIN}`);
  const overOp = addOp(g, OP_KINDS[0]);
  if (overOp.ok) bad.push('addOp beyond MAX_CHAIN unexpectedly succeeded');

  let g2 = addOp(addOp(freshGenome(), 'zoom').genome, 'kaleido').genome;
  const addedReaction = addReaction(g2, 'bass');
  if (!addedReaction.ok) bad.push(`could not add a reaction to set up the moveOp test: ${addedReaction.reason}`);
  else {
    g2 = addedReaction.genome;
    const j2 = g2.reactions.length - 1;
    const op0Targets = reactionTargets(g2).filter((x) => x.g === 'op' && x.i === 0);
    if (op0Targets.length) {
      const rewired = setReactionTarget(g2, j2, op0Targets[0]);
      if (rewired.ok) g2 = rewired.genome;
    }
    const beforeReaction = g2.reactions.find((r) => r.g === 'op' && r.i === 0);
    if (!beforeReaction) bad.push('chain.moveOp: could not get a reaction driving op index 0 for the test');
    else {
      const mv = moveOp(g2, 0, 1);
      if (!mv.ok) bad.push(`moveOp failed: ${mv.reason}`);
      else {
        if (validate(mv.genome).length) bad.push(`moveOp: invalid ${validate(mv.genome).join(';')}`);
        if (JSON.stringify(mv.genome.chain[0]) !== JSON.stringify(g2.chain[1]) || JSON.stringify(mv.genome.chain[1]) !== JSON.stringify(g2.chain[0])) {
          bad.push('moveOp did not swap the first two ops');
        }
        const after = mv.genome.reactions.find((r) => r.g === 'op' && r.k === beforeReaction.k && r.i === 1);
        if (!after) bad.push('moveOp did not carry the reaction targeting op 0 to op 1');

        const rem = removeOp(mv.genome, 1);
        if (!rem.ok) bad.push(`removeOp failed: ${rem.reason}`);
        else {
          if (validate(rem.genome).length) bad.push(`removeOp: invalid ${validate(rem.genome).join(';')}`);
          if (rem.genome.chain.length !== mv.genome.chain.length - 1) bad.push('removeOp did not shrink the chain');
          if (rem.genome.reactions.some((r) => r.g === 'op' && r.k === beforeReaction.k)) bad.push('removeOp did not drop the reaction that targeted the removed op');
        }
      }
    }
  }

  const gZoom = addOp(freshGenome(), 'zoom').genome;
  const stZoom = setStage(gZoom, gZoom.chain.length - 1, 'view');
  if (stZoom.ok) bad.push('setStage(view) unexpectedly succeeded for the motion op "zoom"');
  const gKal = addOp(freshGenome(), 'kaleido').genome;
  const stKal = setStage(gKal, gKal.chain.length - 1, 'view');
  if (!stKal.ok) bad.push(`setStage(view) failed for the fold op "kaleido": ${stKal.reason}`);
  else if (validate(stKal.genome).length) bad.push(`setStage(kaleido, view): invalid ${validate(stKal.genome).join(';')}`);

  check('editor.chain-ops', !bad.length, bad.slice(0, 12).join(' | ') || `chain fills to MAX_CHAIN=${MAX_CHAIN} and caps; moveOp swaps and carries reactions; removeOp reindexes/drops; setStage enforces warp-only motion ops`);
}

{
  // 7. Draw (deform) ops: fill to MAX_DRAW and cap; removing the last op drops deform.ops entirely.
  const bad: string[] = [];
  let g = freshGenome();
  let i = 0;
  while ((g.bodies[0].deform.ops?.length ?? 0) < MAX_DRAW) {
    const r = addDrawOp(g, 0, DRAW_OPS[i % DRAW_OPS.length]);
    if (!r.ok) { bad.push(`addDrawOp failed before the cap at length ${g.bodies[0].deform.ops?.length ?? 0}: ${r.reason}`); break; }
    if (validate(r.genome).length) bad.push(`addDrawOp: invalid ${validate(r.genome).join(';')}`);
    g = r.genome;
    i++;
  }
  if ((g.bodies[0].deform.ops?.length ?? 0) !== MAX_DRAW) bad.push(`draw ops length ${g.bodies[0].deform.ops?.length} != MAX_DRAW ${MAX_DRAW}`);
  const overDraw = addDrawOp(g, 0, DRAW_OPS[0]);
  if (overDraw.ok) bad.push('addDrawOp beyond MAX_DRAW unexpectedly succeeded');
  while (g.bodies[0].deform.ops?.length) {
    const j = g.bodies[0].deform.ops.length - 1;
    const r = removeDrawOp(g, 0, j);
    if (!r.ok) { bad.push(`removeDrawOp failed: ${r.reason}`); break; }
    if (validate(r.genome).length) bad.push(`removeDrawOp: invalid ${validate(r.genome).join(';')}`);
    g = r.genome;
  }
  if (g.bodies[0].deform.ops !== undefined) bad.push('deform.ops was not deleted after removing the last op');
  check('editor.draw-ops', !bad.length, bad.slice(0, 12).join(' | ') || `deform ops fill to MAX_DRAW=${MAX_DRAW} and cap; removing the last op clears deform.ops`);
}

{
  // 8. Reactions: fill to MAX_REACTIONS with distinct targets and cap; rewiring never double-drives a target;
  // removeReaction shrinks; reactionTargets() reports only reactable keys with a group label.
  const bad: string[] = [];
  let g = cloneGenome(seedByOrigin('E21')); // two bodies -> plenty of free targets
  g.reactions = [];
  if (validate(g).length) bad.push(`base genome invalid: ${validate(g).join(';')}`);
  const srcs: Signal[] = ['bass', 'beat', 'drums', 'vocals', 'other', 'melody'];
  let i = 0;
  while (g.reactions.length < MAX_REACTIONS) {
    const r = addReaction(g, srcs[i % srcs.length]);
    if (!r.ok) { bad.push(`addReaction failed before the cap at ${g.reactions.length}: ${r.reason}`); break; }
    if (validate(r.genome).length) bad.push(`addReaction: invalid ${validate(r.genome).join(';')}`);
    g = r.genome;
    i++;
  }
  if (g.reactions.length !== MAX_REACTIONS) bad.push(`reactions length ${g.reactions.length} != MAX_REACTIONS ${MAX_REACTIONS}`);
  const keys = g.reactions.map(reactKey);
  if (new Set(keys).size !== keys.length) bad.push('two reactions drive the same target');
  const overReaction = addReaction(g, 'bass');
  if (overReaction.ok) bad.push('addReaction beyond MAX_REACTIONS unexpectedly succeeded');

  const dup = setReactionTarget(g, 1, { g: g.reactions[0].g, i: g.reactions[0].i, k: g.reactions[0].k });
  if (dup.ok) bad.push('setReactionTarget allowed driving a target twice');
  const free = freeTargets(g, 1);
  if (!free.length) bad.push('no free targets left to test a valid rewire');
  else {
    const rewired = setReactionTarget(g, 1, free[0]);
    if (!rewired.ok) bad.push(`setReactionTarget to a free target failed: ${rewired.reason}`);
    else if (validate(rewired.genome).length) bad.push(`setReactionTarget: invalid ${validate(rewired.genome).join(';')}`);
  }
  const beforeCount = g.reactions.length;
  const removed = removeReaction(g, 0);
  if (!removed.ok || removed.genome.reactions.length !== beforeCount - 1) bad.push('removeReaction did not reduce the reaction count by one');

  const targets = reactionTargets(g);
  if (!targets.length) bad.push('reactionTargets() returned nothing');
  for (const t of targets) {
    const s = schemaFor(g, t.g, t.i);
    if (!s || !reactable(s).includes(t.k)) bad.push(`reactionTargets: ${t.g}${t.i}.${t.k} is not actually reactable`);
    if (!t.group) bad.push(`reactionTargets: ${t.g}${t.i}.${t.k} has an empty group label`);
  }
  check('editor.reactions', !bad.length, bad.slice(0, 12).join(' | ') || `reactions fill to MAX_REACTIONS=${MAX_REACTIONS} with distinct targets and cap; rewiring never double-drives; reactionTargets() stays reactable`);
}

{
  // 9. Expressing a silent allele swaps it into place and files the old kind as silent (flame never lingers as one).
  const bad: string[] = [];
  let carrier: { g: Genome; b: number; locus: Locus } | null = null;
  outer:
  for (const s of SEEDS) {
    for (let bi = 0; bi < s.genome.bodies.length; bi++) {
      const alt = s.genome.bodies[bi].alt;
      if (alt) {
        const locus = (Object.keys(alt) as Locus[])[0];
        carrier = { g: s.genome, b: bi, locus };
        break outer;
      }
    }
  }
  if (!carrier) {
    // Silent alleles are a crossover artifact (see section 15): a plain randomGenome() never carries
    // one, so breed random seed pairs (occasionally mutating) to reproduce how they actually arise.
    const rng = mulberry32(123456);
    for (let k = 0; k < 500 && !carrier; k++) {
      let g = randomGenome(rng);
      if (k % 2 === 0) g = crossover(SEEDS[Math.floor(rng() * SEEDS.length)].genome, SEEDS[Math.floor(rng() * SEEDS.length)].genome, rng);
      if (rng() < 0.3) g = mutate(g, rng, 0.6);
      for (let bi = 0; bi < g.bodies.length; bi++) {
        const alt = g.bodies[bi].alt;
        // A silent allele the body cannot express (e.g. a fill material on a curve) is a valid state, not this check's case.
        if (alt && expressAllele(g, bi, (Object.keys(alt) as Locus[])[0]).ok) {
          const locus = (Object.keys(alt) as Locus[])[0];
          // Some silent kinds cannot be expressed on this body (a fill material on a curve); keep looking.
          if (!expressAllele(g, bi, locus).ok) continue;
          carrier = { g, b: bi, locus };
          break;
        }
      }
    }
  }
  if (!carrier) {
    bad.push('no genome with a silent allele found among the 24 seeds or 500 random/crossed genomes');
  } else {
    const { g, b, locus } = carrier;
    const oldKind = (g.bodies[b][locus] as Gene).kind;
    const silentKind = g.bodies[b].alt![locus]!.kind;
    const r = expressAllele(g, b, locus);
    if (!r.ok) bad.push(`expressAllele failed: ${r.reason}`);
    else {
      if (validate(r.genome).length) bad.push(`expressAllele: invalid ${validate(r.genome).join(';')}`);
      if ((r.genome.bodies[b][locus] as Gene).kind !== silentKind) bad.push(`locus ${locus} not switched to the previously silent kind ${silentKind}`);
      if (oldKind === 'flame') {
        if (r.genome.bodies[b].alt?.[locus]) bad.push('a flame kind should not linger on as a new silent allele');
      } else if (r.genome.bodies[b].alt?.[locus]?.kind !== oldKind) {
        bad.push(`old kind ${oldKind} was not filed as the new silent allele for ${locus}`);
      }
    }
  }
  check('editor.express-allele', !bad.length, bad.join(' | ') || `expressAllele swaps ${carrier?.locus} into place and files the old kind as silent`);
}

{
  // 10. saveEdited files an edit as a new generation, named uniquely, surviving population round-trip.
  const bad: string[] = [];
  const pop = Population.seeded();
  const parent = pop.get('G0-E07') ?? pop.list()[0];
  const curMotion = parent.genome.bodies[0].motion.kind;
  const altMotion = MOTION_KINDS.find((k) => k !== curMotion)!;
  const sw = switchKind(parent.genome, { t: 'locus', b: 0, locus: 'motion' }, altMotion);
  if (!sw.ok) bad.push(`setup: switchKind(motion -> ${altMotion}) failed: ${sw.reason}`);
  const child = saveEdited(pop, parent, sw.genome, 12345);
  if (child.cross !== 'edited') bad.push(`child.cross=${child.cross} expected "edited"`);
  if (JSON.stringify(child.parents) !== JSON.stringify([parent.id])) bad.push(`child.parents=${JSON.stringify(child.parents)} expected [${parent.id}]`);
  if (child.gen !== parent.gen + 1) bad.push(`child.gen=${child.gen} expected ${parent.gen + 1}`);
  if (!/^G1-\d{4}$/.test(child.id)) bad.push(`child.id=${child.id} does not match /^G1-\\d{4}$/`);
  if (pop.get(child.id) !== child) bad.push('pop.get(child.id) !== child');
  if (validate(child.genome).length) bad.push(`child.genome invalid: ${validate(child.genome).join(';')}`);
  if (!child.name) bad.push('child has no name');
  if (child.name === parent.name) bad.push(`child.name "${child.name}" is not unique from the parent's`);

  const grandchild = saveEdited(pop, child, child.genome, 12346);
  if (grandchild.gen !== 2 || !/^G2-\d{4}$/.test(grandchild.id)) bad.push(`grandchild gen/id wrong: ${grandchild.gen}/${grandchild.id}`);
  const n1 = parseInt(child.id.split('-')[1], 10);
  const n2 = parseInt(grandchild.id.split('-')[1], 10);
  if (!(n2 > n1)) bad.push(`id counter did not increment: ${child.id} -> ${grandchild.id}`);

  const pop2 = Population.fromJSON(JSON.parse(JSON.stringify(pop.toJSON())));
  if (pop2.get(child.id)?.cross !== 'edited') bad.push('Population.fromJSON(toJSON()) lost cross="edited" on the saved edit');

  check('editor.save-as-new', !bad.length, bad.join(' | ') || `saveEdited makes ${child.id} "${child.name}" (gen ${child.gen}) from ${parent.id}, then ${grandchild.id} at gen 2, surviving round-trip`);
}

{
  // 11. parseGenome round-trips a bare genome and a {genome} wrapper, and rejects garbage with a reason.
  const bad: string[] = [];
  const seed = SEEDS[0].genome;
  const r1 = parseGenome(JSON.stringify(seed));
  if (!r1.ok) bad.push(`parseGenome(seed) failed: ${r1.reason}`);
  else if (JSON.stringify(r1.genome) !== JSON.stringify(seed)) bad.push('parseGenome(seed) did not deep-equal the seed');
  const r2 = parseGenome(JSON.stringify({ genome: seed }));
  if (!r2.ok) bad.push(`parseGenome({genome: seed}) failed: ${r2.reason}`);
  else if (JSON.stringify(r2.genome) !== JSON.stringify(seed)) bad.push('parseGenome({genome: seed}) did not deep-equal the seed');
  const r3 = parseGenome('nope');
  if (r3.ok) bad.push('parseGenome("nope") unexpectedly succeeded');
  else if (!r3.reason) bad.push('parseGenome("nope") failed without a reason');
  check('editor.parse-genome', !bad.length, bad.join(' | ') || 'parseGenome round-trips a genome and a {genome} wrapper, rejects garbage with a reason');
}

// ------------------------------------------- 18. MilkDrop seeds and carrier extensions

{
  const ms = SEEDS.filter((x) => x.origin.startsWith('M'));
  const es = SEEDS.filter((x) => x.origin.startsWith('E'));
  const ids = ms.map((x) => `G0-${x.origin}`);
  check('milkdrop.ids', JSON.stringify(ids) === JSON.stringify(Array.from({ length: M_COUNT }, (_, i) => `G0-M${String(i + 1).padStart(2, '0')}`)), ids.join(','));
  const badMeta = ms.filter((x) => !/\(after [^)]+\)$/.test(x.name) || x.name.length > 60 || x.genome.reactions.length < 1 || !(x.genome.energy[1] - x.genome.energy[0] >= 0.15));
  check('milkdrop.meta', !badMeta.length, badMeta.map((x) => x.origin).join(',') || 'credited names, reactions and an energy range on every M seed');
  const inRangeBad: string[] = [];
  for (const x of ms) {
    const errs = validate(x.genome);
    if (errs.length) inRangeBad.push(`${x.origin}:${errs[0]}`);
    if (!(estimateCost(x.genome) < COST_BUDGET_MS)) inRangeBad.push(`${x.origin}:cost ${estimateCost(x.genome).toFixed(2)}`);
    if (JSON.stringify(repair(JSON.parse(JSON.stringify(x.genome)))) !== JSON.stringify(x.genome)) inRangeBad.push(`${x.origin}:round-trip`);
    if (!buildSources(x.genome).feedback.includes('uSharpen')) inRangeBad.push(`${x.origin}:glsl`);
  }
  check('milkdrop.valid-in-range-budget', !inRangeBad.length, inRangeBad.join(' | ') || 'valid, in range, round-trip, under budget, buildable');
  // Every M seed crossed with one seed of every species present among the originals, both ways.
  const bySpecies = new Map<string, (typeof SEEDS)[number]>();
  for (const e of es) {
    const sp = classify(e.genome).primary;
    if (!bySpecies.has(sp)) bySpecies.set(sp, e);
  }
  const rng = mulberry32(8080);
  const crossBad: string[] = [];
  let crosses = 0;
  for (const m of ms) for (const e of bySpecies.values()) for (const [a, b] of [[m, e], [e, m]]) {
    for (let k = 0; k < 4; k++) {
      const c = crossover(a.genome, b.genome, rng);
      crosses++;
      const errs = validate(c);
      if (errs.length) crossBad.push(`${a.origin}x${b.origin}:${errs[0]}`);
      if (!(estimateCost(c) < COST_BUDGET_MS)) crossBad.push(`${a.origin}x${b.origin}:cost`);
      const mu = mutate(c, rng, 1.5);
      if (validate(mu).length) crossBad.push(`${a.origin}x${b.origin}:mutant ${validate(mu)[0]}`);
    }
  }
  check('milkdrop.crossover-every-species', !crossBad.length, crossBad.slice(0, 8).join(' | ') || `${crosses} crossovers with ${bySpecies.size} species (and their mutants) valid and under budget`);

  // The carrier genes: old genomes get them switched off, they cost when on, they breed in range.
  const old = cloneGenome(seedByOrigin('E05')) as unknown as { carrier: { p: Record<string, number> } };
  delete old.carrier.p.sharpen; delete old.carrier.p.grain; delete old.carrier.p.border;
  const fixed = repair(old);
  check('carrier.new-genes-default-off', fixed.carrier.p.sharpen === 0 && fixed.carrier.p.border === 0 && fixed.carrier.p.grain === CARRIER_SCHEMA.grain.def, JSON.stringify({ s: fixed.carrier.p.sharpen, b: fixed.carrier.p.border }));
  const on = cloneGenome(fixed);
  on.carrier.p.sharpen = 0.4;
  on.carrier.p.border = 0.8;
  check('carrier.new-genes-cost', Math.abs(estimateCost(on) - estimateCost(fixed) - 0.48) < 1e-9, `${estimateCost(fixed).toFixed(2)} -> ${estimateCost(on).toFixed(2)} ms`);
  const offNone = cloneGenome(on);
  offNone.carrier.kind = 'none';
  check('carrier.new-genes-free-without-feedback', estimateCost(offNone) < estimateCost(on), 'no feedback, no sharpen / border cost');
  const r2 = mulberry32(5150);
  let strayed = 0, sharpened = 0;
  for (let i = 0; i < 400; i++) {
    const g = mutate(i % 2 ? on : seedByOrigin('M02'), r2, 2);
    for (const k of ['sharpen', 'grain', 'border']) {
      const sp = CARRIER_SCHEMA[k];
      if (!(g.carrier.p[k] >= sp.min && g.carrier.p[k] <= sp.max)) strayed++;
    }
    if (g.carrier.p.sharpen > 0) sharpened++;
  }
  const rnd = mulberry32(777);
  let randomOn = 0;
  for (let i = 0; i < 300; i++) if (randomGenome(rnd).carrier.p.sharpen > 0) randomOn++;
  check('carrier.new-genes-breed', strayed === 0 && sharpened > 300 && randomOn > 5 && randomOn < 100, `strayed=${strayed} sharpened=${sharpened}/400 random-on=${randomOn}/300`);

  // The quad op (z-squared flow): a breedable chain op with its shader snippet.
  const qr = mulberry32(4455);
  let quads = 0;
  const quadBad: string[] = [];
  for (let i = 0; i < 600; i++) {
    const o = randomOp(qr);
    if (o.op !== 'quad') continue;
    quads++;
    const g = repair({ ...cloneGenome(seedByOrigin('E05')), chain: [o] });
    if (validate(g).length) quadBad.push(validate(g)[0]);
    if (!buildSources(g).feedback.includes('d.x * d.x - d.y * d.y')) quadBad.push('no quad glsl');
  }
  const qg = repair({ ...cloneGenome(seedByOrigin('E05')), chain: [{ op: 'quad', stage: 'view', w: 2, p: { amt: 9, turn: 0.3 } }] });
  check('ops.quad', quads > 10 && !quadBad.length && qg.chain[0].stage === 'warp' && qg.chain[0].w === 1 && qg.chain[0].p.amt === 2.5 && qg.chain[0].p.turn === 0.25, quadBad.join(' | ') || `${quads}/600 random ops were quad, all valid and built; a bad one is repaired into range`);

  // Migration: a seed-version-5 population (24 seeds, votes, a hidden seed, bred children) gains
  // M01-M10 exactly once; the original seeds and every child stay exactly as they were.
  const base = Population.seeded(1);
  for (const x of ms) base.members.delete(`G0-${x.origin}`);
  base.seedVersion = 5;
  base.get('G0-E07')!.likes = 3;
  base.get('G0-E07')!.descriptor = [0.1, 0.2, 0.3];
  base.get('G0-E12')!.hidden = true;
  const kid = base.addChild(crossover(seedByOrigin('E05'), seedByOrigin('E22'), mulberry32(3)), [base.get('G0-E05')!, base.get('G0-E22')!], 2);
  base.vote(kid.id, true);
  const file = JSON.parse(JSON.stringify({ ...base.toJSON(), seedVersion: 5 }));
  const loaded = Population.fromJSON(file);
  const before = JSON.stringify(loaded.list().sort((a, b) => a.id.localeCompare(b.id)));
  const added = loaded.upgradeSeeds(9);
  const afterOld = JSON.stringify(loaded.list().filter((m) => !m.id.startsWith('G0-M')).sort((a, b) => a.id.localeCompare(b.id)));
  check('migrate.adds-milkdrop-once', JSON.stringify(added) === JSON.stringify(ids) && loaded.size === base.size + M_COUNT && loaded.upgradeSeeds(10).length === 0, `added ${added.join(',')}`);
  check('migrate.milkdrop-leaves-rest', before === afterOld && loaded.get('G0-E07')!.likes === 3 && loaded.get('G0-E07')!.descriptor?.length === 3 && loaded.get('G0-E12')!.hidden && loaded.get(kid.id)!.likes === 1, 'existing seeds (votes, descriptors, hidden) and bred children untouched');
  check('migrate.milkdrop-genomes', ms.every((x) => JSON.stringify(loaded.get(`G0-${x.origin}`)!.genome) === JSON.stringify(x.genome) && loaded.get(`G0-${x.origin}`)!.name === x.name), 'new seeds arrive with their genomes and names');
  const reload = Population.fromJSON(JSON.parse(JSON.stringify(loaded.toJSON())));
  check('migrate.milkdrop-stable', reload.upgradeSeeds().length === 0 && reload.size === loaded.size, 'a saved migrated population reloads without further changes');
}

// -------------------------------------------------- AVS genes: superscope shape

{
  const bad: string[] = [];
  const rng = mulberry32(6106);
  const scopeGenome = (p: Record<string, number> = {}): Genome => repair({
    ...cloneGenome(seedByOrigin('E07')),
    bodies: [{ ...cloneGenome(seedByOrigin('E07')).bodies[0], shape: { kind: 'superscope', p: { ...p } }, material: { kind: 'dots', p: { gain: 1, spacing: 0.01, size: 0.5 } } }],
  });
  const sg = scopeGenome({ family: 5, p: 4, n: 2048 });
  check('superscope.schema', SHAPE_KINDS.includes('superscope') && SHAPE_CLASS.superscope === 'curve' && SHAPE_SCHEMAS.superscope === SUPERSCOPE_SCHEMA && !sdfCapable(sg.bodies[0].shape), 'a curve-class shape with its own schema, no distance field');
  check('superscope.repair', !validate(sg).length && sg.bodies[0].shape.p.family === 5 && JSON.stringify(repair(JSON.parse(JSON.stringify(sg)))) === JSON.stringify(sg), validate(sg).join(';') || 'valid and idempotent');
  const wild = scopeGenome({ family: 9, p: 0, q: 11, size: 3, audio: -1, spinX: 0.3, n: 5000 });
  const wp = wild.bodies[0].shape.p;
  check('superscope.clamps', !validate(wild).length && wp.family === 5 && wp.p === 1 && wp.q === 8 && wp.size === 0.45 && wp.audio === 0 && wp.spinX === 0.25 && wp.n === 2048, JSON.stringify(wp));
  // Never fused (no distance field to light up inside), even when a fuse is supplied.
  const fused = repair({ ...cloneGenome(sg), bodies: [{ ...sg.bodies[0], fuse: { shape: { kind: 'dot', p: { r: 0.1 } }, p: { mode: 2 } } }] });
  check('superscope.no-fuse', !fused.bodies[0].fuse && !validate(fused).length && makeFuse(sg.bodies[0], { kind: 'dot', p: { r: 0.1 } }, rng) === null, 'fuse dropped');
  check('superscope.glsl', WAVE_VS.includes('superscopeAt(k)') && WAVE_VS.includes('sh == 6') && buildSources(sg).feedback.includes('void main'), 'curve pass carries the superscope family code');
  const cheap = scopeGenome({ n: 256 });
  check('superscope.cost', estimateCost(sg) > estimateCost(cheap) && estimateCost(sg) < COST_BUDGET_MS, `${estimateCost(cheap).toFixed(3)} -> ${estimateCost(sg).toFixed(3)} ms`);
  check('superscope.named', nounKind(sg.bodies[0]) === 'scope' && NOUN_POOLS.scope.includes(nameFor(sg).split(/\s+/).pop()!) && classify(sg).primary === 'scope', `${nameFor(sg)} (${classify(sg).label})`);
  // Breeds with every species both ways; mutants stay valid; random bodies sometimes pick it.
  const bySpecies = new Map<string, Genome>();
  for (const e of SEEDS) if (!bySpecies.has(classify(e.genome).primary)) bySpecies.set(classify(e.genome).primary, e.genome);
  let crosses = 0, kept = 0;
  for (const e of bySpecies.values()) for (const [a, b] of [[sg, e], [e, sg]]) for (let k = 0; k < 6; k++) {
    const c = crossover(a, b, rng);
    crosses++;
    if (c.bodies.some((x) => x.shape.kind === 'superscope')) kept++;
    if (validate(c).length) bad.push(`cross:${validate(c)[0]}`);
    if (!(estimateCost(c) < COST_BUDGET_MS)) bad.push('cross:cost');
    const mu = mutate(c, rng, 2);
    if (validate(mu).length) bad.push(`mutant:${validate(mu)[0]}`);
    if (!buildSources(mu).feedback.includes('void main')) bad.push('mutant:glsl');
  }
  let random = 0;
  for (let i = 0; i < 400; i++) if (randomBody(rng).shape.kind === 'superscope') random++;
  check('superscope.breeds', !bad.length && kept > 10 && random > 5, bad.slice(0, 6).join(' | ') || `${crosses} crossovers over ${bySpecies.size} species valid, ${kept} kept the scope; ${random}/400 random bodies are scopes`);
}

// -------------------------------------------------- AVS genes: water ripple carrier

{
  const old = cloneGenome(seedByOrigin('E05')) as unknown as { carrier: { p: Record<string, number> } };
  delete old.carrier.p.water; delete old.carrier.p.wsize;
  const fixed = repair(old);
  check('water.default-off', fixed.carrier.p.water === 0 && fixed.carrier.p.wsize === CARRIER_SCHEMA.wsize.def && !validate(fixed).length, JSON.stringify({ w: fixed.carrier.p.water, s: fixed.carrier.p.wsize }));
  const on = cloneGenome(fixed);
  on.carrier.p.water = 0.7;
  const wild = repair({ ...cloneGenome(on), carrier: { kind: 'warp', p: { ...on.carrier.p, water: 5, wsize: -1 } } });
  check('water.clamps', wild.carrier.p.water === 1 && wild.carrier.p.wsize === CARRIER_SCHEMA.wsize.min, JSON.stringify({ w: wild.carrier.p.water, s: wild.carrier.p.wsize }));
  const offNone = cloneGenome(on);
  offNone.carrier.kind = 'none';
  check('water.cost', estimateCost(on) > estimateCost(fixed) && estimateCost(offNone) < estimateCost(on) && estimateCost(on) < COST_BUDGET_MS, `${estimateCost(fixed).toFixed(2)} -> ${estimateCost(on).toFixed(2)} ms`);
  const src = buildSources(on).feedback;
  check('water.glsl', src.includes('waterSlope(vUv)') && src.includes('uniform sampler2D uWater'), 'feedback reads the ripple slope');
  const r = mulberry32(2718);
  let strayed = 0, randomOn = 0, reacts = 0;
  for (let i = 0; i < 300; i++) {
    const g = mutate(i % 2 ? on : seedByOrigin('M07'), r, 2);
    for (const k of ['water', 'wsize']) if (!(g.carrier.p[k] >= CARRIER_SCHEMA[k].min && g.carrier.p[k] <= CARRIER_SCHEMA[k].max)) strayed++;
    if (randomGenome(r).carrier.p.water > 0) randomOn++;
  }
  if (reactable(CARRIER_SCHEMA).includes('water')) reacts = 1;
  // Crossed with a seed of every species, both ways, the ripple survives in range and under budget.
  const bad: string[] = [];
  const bySpecies = new Map<string, Genome>();
  for (const e of SEEDS) if (!bySpecies.has(classify(e.genome).primary)) bySpecies.set(classify(e.genome).primary, e.genome);
  for (const e of bySpecies.values()) for (const [a, b] of [[on, e], [e, on]]) {
    const c = crossover(a, b, r);
    if (validate(c).length) bad.push(validate(c)[0]);
    if (!(estimateCost(c) < COST_BUDGET_MS)) bad.push('cost');
  }
  check('water.breeds', strayed === 0 && randomOn > 3 && randomOn < 60 && reacts === 1 && !bad.length, bad.slice(0, 4).join(' | ') || `strayed=${strayed} random-on=${randomOn}/300, reactable, ${bySpecies.size * 2} crossovers valid`);
}

// -------------------------------------------------- AVS seeds (A01..)

{
  const as = SEEDS.filter((x) => x.origin.startsWith('A'));
  const badMeta = as.filter((x) => !/^A\d\d$/.test(x.origin) || !/\(after [^)]+\)$/.test(x.name) || x.name.length > 60 || x.genome.reactions.length < 1 || !(x.genome.energy[1] - x.genome.energy[0] >= 0.15));
  check('avs.meta', as.length >= 2 && !badMeta.length, badMeta.map((x) => x.origin).join(',') || `${as.length} AVS seeds: ${as.map((x) => x.name).join(', ')}`);
  const bad: string[] = [];
  for (const x of as) {
    const errs = validate(x.genome);
    if (errs.length) bad.push(`${x.origin}:${errs[0]}`);
    if (!(estimateCost(x.genome) < COST_BUDGET_MS)) bad.push(`${x.origin}:cost ${estimateCost(x.genome).toFixed(2)}`);
    if (JSON.stringify(repair(JSON.parse(JSON.stringify(x.genome)))) !== JSON.stringify(x.genome)) bad.push(`${x.origin}:round-trip`);
    if (!buildSources(x.genome).feedback.includes('void main')) bad.push(`${x.origin}:glsl`);
  }
  // The AVS genes are on the seeds that showcase them.
  const uses = (o: string, f: (g: Genome) => boolean) => { const x = as.find((y) => y.origin === o); return !x || f(x.genome); };
  if (!uses('A01', (g) => g.bodies.some((b) => b.shape.kind === 'superscope'))) bad.push('A01 without a superscope');
  if (!uses('A03', (g) => g.carrier.p.water > 0)) bad.push('A03 without water');
  check('avs.valid-in-range-budget', !bad.length, bad.join(' | ') || 'valid, round-trip, under budget, buildable, genes present');
  const bySpecies = new Map<string, Genome>();
  for (const e of SEEDS.filter((x) => x.origin.startsWith('E'))) if (!bySpecies.has(classify(e.genome).primary)) bySpecies.set(classify(e.genome).primary, e.genome);
  const rng = mulberry32(4242);
  const cbad: string[] = [];
  let crosses = 0;
  for (const x of as) for (const e of bySpecies.values()) for (const [a, b] of [[x.genome, e], [e, x.genome]]) for (let k = 0; k < 3; k++) {
    const c = crossover(a, b, rng);
    crosses++;
    if (validate(c).length) cbad.push(`${x.origin}:${validate(c)[0]}`);
    if (!(estimateCost(c) < COST_BUDGET_MS)) cbad.push(`${x.origin}:cost`);
    if (validate(mutate(c, rng, 1.5)).length) cbad.push(`${x.origin}:mutant`);
  }
  check('avs.crossover-every-species', !cbad.length, cbad.slice(0, 6).join(' | ') || `${crosses} crossovers (and mutants) valid and under budget`);
  // Migration: a population from before the AVS seeds gains them exactly once; nothing else changes.
  const base = Population.seeded(1);
  for (const x of as) base.members.delete(`G0-${x.origin}`);
  base.seedVersion = SEED_VERSION - 1;
  base.get('G0-M03')!.likes = 2;
  const kid = base.addChild(crossover(seedByOrigin('E05'), seedByOrigin('M07'), mulberry32(5)), [base.get('G0-E05')!, base.get('G0-M07')!], 2);
  base.vote(kid.id, true);
  const loaded = Population.fromJSON(JSON.parse(JSON.stringify({ ...base.toJSON(), seedVersion: SEED_VERSION - 1 })));
  const before = JSON.stringify(loaded.list().sort((a, b) => a.id.localeCompare(b.id)));
  const added = loaded.upgradeSeeds(11);
  const after = JSON.stringify(loaded.list().filter((m) => !/^G0-A\d/.test(m.id)).sort((a, b) => a.id.localeCompare(b.id)));
  check('migrate.adds-avs-once', JSON.stringify(added) === JSON.stringify(as.map((x) => `G0-${x.origin}`)) && loaded.upgradeSeeds(12).length === 0, `added ${added.join(',')}`);
  check('migrate.avs-leaves-rest', before === after && loaded.get('G0-M03')!.likes === 2 && loaded.get(kid.id)!.likes === 1 && as.every((x) => JSON.stringify(loaded.get(`G0-${x.origin}`)!.genome) === JSON.stringify(x.genome)), 'existing members untouched, new seeds arrive with their genomes');
}
// -------------------------------------------------- 19. ray-marched scenes

raymarchChecks(check);

// -------------------------------------------------- AVS genes: blend mode (material.blend)

{
  const seed = seedByOrigin('E13');
  const noField = cloneGenome(seed) as unknown as { bodies: { material: { p: Record<string, number> } }[] };
  for (const b of noField.bodies) delete b.material.p.blend;
  const fixed = repair(noField);
  check('blend.default-add', fixed.bodies.every((b) => b.material.p.blend === 0) && buildSources(fixed).feedback === buildSources(seed).feedback && !buildSources(fixed).feedback.includes('c = blendInto('), 'old bodies add, shaders unchanged');
  const bad: string[] = [];
  for (const mode of [1, 2, 3, 4, 5]) {
    const g = cloneGenome(seed);
    g.bodies[0].material.p.blend = mode;
    const r = repair(g);
    const src = buildSources(r);
    if (validate(r).length) bad.push(`${mode}:${validate(r)[0]}`);
    if (!(src.feedback + src.composite).includes(`blendInto(c, body_`)) bad.push(`${mode}:no wrap`);
    if (structuralKey(r) === structuralKey(seed)) bad.push(`${mode}:same key`);
    if (!(estimateCost(r) > estimateCost(seed))) bad.push(`${mode}:cost`);
  }
  const wild = repair({ ...cloneGenome(seed), bodies: [{ ...cloneGenome(seed).bodies[0], material: { kind: 'glow', p: { blend: 9 } } }] });
  check('blend.modes', !bad.length && wild.bodies[0].material.p.blend === 5, bad.join(' | ') || 'modes 1-5 wrap the body call, change the structure, cost a little; out-of-range clamps');
  const rng = mulberry32(1357);
  let on = 0;
  for (let i = 0; i < 400; i++) if (randomBody(rng).material.p.blend > 0) on++;
  const a07 = SEEDS.find((x) => x.origin === 'A07');
  const cbad: string[] = [];
  const bySpecies = new Map<string, Genome>();
  for (const e of SEEDS.filter((x) => x.origin.startsWith('E'))) if (!bySpecies.has(classify(e.genome).primary)) bySpecies.set(classify(e.genome).primary, e.genome);
  const xg = cloneGenome(seed);
  xg.bodies[0].material.p.blend = 2;
  for (const e of bySpecies.values()) for (const [a, b] of [[xg, e], [e, xg]]) {
    const c = crossover(a, b, rng);
    if (validate(c).length) cbad.push(validate(c)[0]);
    if (!buildSources(c).feedback.includes('void main')) cbad.push('glsl');
  }
  check('blend.breeds', on > 20 && on < 110 && !cbad.length && !!a07 && a07.genome.bodies[0].material.p.blend === 3, cbad.slice(0, 4).join(' | ') || `${on}/400 random bodies blend non-additively; crossovers valid; A07 draws in xor`);
}

// -------------------------------------------------- 16. example crossovers

{
  console.log('\n--- 30 example crossovers ---');
  const rng = mulberry32(99009);
  const short = (b: BodyGene) => `${b.shape.kind}/${b.place.kind}/${b.motion.kind}/${b.deform.kind}/${b.material.kind}/${b.emit.kind}${b.fuse ? `+${b.fuse.shape.kind}` : ''}`;
  for (let i = 0; i < 30; i++) {
    const a = SEEDS[Math.floor(rng() * 24)];
    let b = SEEDS[Math.floor(rng() * 24)];
    if (b.origin === a.origin) b = SEEDS[(SEEDS.indexOf(b) + 1) % 24];
    const child = crossoverTagged(a.genome, b.genome, rng, (rng() - 0.5) * 2);
    console.log(`${a.name} x ${b.name} -> ${nameFor(child.genome, [a.name, b.name])}  [${child.tag}] ${short(child.genome.bodies[0])}`);
  }
}

// -------------------------------------------------- choreography

choreoTests(check);
driftTests(check);
slimeTests(check);
flockTests(check);
ecosystemTests(check);
physicsChecks(check);

// -------------------------------------------------- MilkDrop mining: relief (emboss lighting)

{
  const base = seedByOrigin('E07');
  const off = repair(JSON.parse(JSON.stringify({ ...base, tone: { p: { ...base.tone.p, relief: undefined, bump: undefined, light: undefined, gloss: undefined, metal: undefined } } })));
  check('relief.default-off', off.tone.p.relief === 0 && TONE_SCHEMA.relief.max === 1 && !validate(off).length, JSON.stringify({ relief: off.tone.p.relief, bump: off.tone.p.bump }));
  const on = cloneGenome(off);
  on.tone.p.relief = 0.8;
  check('relief.cost', Math.abs(estimateCost(on) - estimateCost(off) - 0.12) < 1e-9 && estimateCost(seedByOrigin('M11')) < COST_BUDGET_MS, `${estimateCost(off).toFixed(2)} -> ${estimateCost(on).toFixed(2)} ms; M11 ${estimateCost(seedByOrigin('M11')).toFixed(2)} ms`);
  const bad = repair({ ...cloneGenome(on), tone: { p: { ...on.tone.p, relief: 7, bump: -3, light: 2, gloss: NaN, metal: 0.5 } } });
  check('relief.repair', bad.tone.p.relief === 1 && bad.tone.p.bump === TONE_SCHEMA.bump.min && bad.tone.p.light === 1 && bad.tone.p.gloss === TONE_SCHEMA.gloss.def && !validate(bad).length, JSON.stringify(bad.tone.p));
  const src = buildSources(on);
  check('relief.glsl', src.composite.includes('reliefLit(q, fb(q))') && src.composite.includes('uniform vec4 uRelief') && !src.feedback.includes('reliefLit'), 'composite lights the feedback layer; the feedback pass is untouched');
  // Breeds with every species: M11 crossed both ways with one seed per species stays valid and in range.
  const m11 = seedByOrigin('M11');
  const rng = mulberry32(8181);
  const probs: string[] = [];
  let kept = 0, n = 0;
  const bySp = new Map<string, Genome>();
  for (const x of SEEDS) if (!bySp.has(classify(x.genome).primary)) bySp.set(classify(x.genome).primary, x.genome);
  for (const [sp, other] of bySp) {
    for (let k = 0; k < 6; k++) {
      const c = k % 2 ? crossover(m11, other, rng) : crossover(other, m11, rng);
      n++;
      if (validate(c).length) probs.push(`${sp}:${validate(c)[0]}`);
      if (!(c.tone.p.relief >= 0 && c.tone.p.relief <= 1)) probs.push(`${sp}: relief ${c.tone.p.relief}`);
      if (c.tone.p.relief > 0) kept++;
      const m = mutate(c, rng, 2);
      if (validate(m).length) probs.push(`${sp} mutated:${validate(m)[0]}`);
    }
  }
  check('relief.breeds-with-every-species', !probs.length && kept > n * 0.3, probs.slice(0, 3).join(' | ') || `${bySp.size} species x 6 crossovers (+ mutation) valid; relief inherited in ${kept}/${n}`);
  const rr = mulberry32(99);
  let randomOn = 0;
  for (let i = 0; i < 300; i++) if (randomGenome(rr).tone.p.relief > 0) randomOn++;
  check('relief.rare-in-random', randomOn > 5 && randomOn < 80, `${randomOn}/300 random genomes embossed`);
  const nm = nameFor(on);
  const embossedHits = SEEDS.slice(0, 24).filter((x) => ADJ_POOLS.embossed.includes(nameFor(repair({ ...cloneGenome(x.genome), tone: { p: { ...x.genome.tone.p, relief: 1 } } })).split(' ')[0])).length;
  check('relief.name', embossedHits >= 4 && classify(m11).label.includes('chrome'), `${embossedHits}/24 originals named embossed when relief is on (e.g. ${nm}); M11 is ${classify(m11).label}`);
  // Migration: a seed-version-6 population (E01-E24 + M01-M10, votes, a child) gains M11 exactly once.
  const pop = Population.seeded(1);
  for (const x of SEEDS) if (!/^(E\d\d|M0\d|M10)$/.test(x.origin)) pop.members.delete(`G0-${x.origin}`);
  pop.get('G0-M05')!.likes = 2;
  const kid = pop.addChild(crossover(seedByOrigin('M05'), seedByOrigin('E22'), mulberry32(4)), [pop.get('G0-M05')!, pop.get('G0-E22')!], 2);
  const file = JSON.parse(JSON.stringify({ ...pop.toJSON(), seedVersion: 6 }));
  const loaded = Population.fromJSON(file);
  const kidBefore = JSON.stringify(loaded.get(kid.id));
  const added = loaded.upgradeSeeds(9);
  check('migrate.adds-m11-once', added.includes('G0-M11') && JSON.stringify(loaded.get('G0-M11')!.genome) === JSON.stringify(m11) && loaded.upgradeSeeds(10).length === 0 && JSON.stringify(loaded.get(kid.id)) === kidBefore && loaded.get('G0-M05')!.likes === 2, `added ${added.filter((x) => /M1[1-9]|M[2-9]\d/.test(x)).join(',')}; child and votes untouched`);
}

// -------------------------------------------------- phenotype fingerprints and novelty

noveltyTests(check);
await noveltyTestsAsync(check);
await lyricsTests(check);

// -------------------------------------------------- MilkDrop mining: cells (Voronoi field)

{
  const m12 = seedByOrigin('M12'), m13 = seedByOrigin('M13');
  const cb = m12.bodies[0];
  check('cells.schema', SHAPE_KINDS.includes('cells') && SHAPE_CLASS.cells === 'field' && SHAPE_SCHEMAS.cells === CELLS_SCHEMA && !sdfCapable(cb.shape), 'a field-class shape with its own schema');
  const wild = repair({ ...cloneGenome(m12), bodies: [{ ...cloneBody(cb), shape: { kind: 'cells', p: { mode: 7, scale: 99, speed: -1, warp: 2, wall: NaN } }, place: { kind: 'orbit', p: { count: 5 } } }] });
  const wp = wild.bodies[0].shape.p;
  check('cells.repair', !validate(wild).length && wp.mode === 2 && wp.scale === 14 && wp.speed === 0 && wp.warp === 1 && wp.wall === CELLS_SCHEMA.wall.def && wild.bodies[0].place.p.count === 1, JSON.stringify(wp));
  const src = buildSources(m12).feedback;
  check('cells.glsl', /vec3 FLD_0\(vec2 p\)/.test(src) && src.includes('f2 - f1') && !buildSources(m13).composite.includes('undefined'), 'the Voronoi field is built into the body pass');
  const plain = cloneGenome(m12);
  plain.bodies[0].shape.p.warp = 0;
  check('cells.cost', Math.abs(bodyCost(m12.bodies[0]) - bodyCost(plain.bodies[0]) - 0.25) < 1e-9 && estimateCost(m12) < COST_BUDGET_MS && estimateCost(m13) < COST_BUDGET_MS, `M12 ${estimateCost(m12).toFixed(2)} ms (measured 1.51 at 1440p), M13 ${estimateCost(m13).toFixed(2)} ms (1.63)`);
  const rng = mulberry32(7171);
  const probs: string[] = [];
  let kept = 0, n = 0;
  const bySp = new Map<string, Genome>();
  for (const x of SEEDS) if (!bySp.has(classify(x.genome).primary)) bySp.set(classify(x.genome).primary, x.genome);
  for (const [sp, other] of bySp) {
    for (let k = 0; k < 6; k++) {
      const c = k % 2 ? crossover(m12, other, rng) : crossover(other, m13, rng);
      n++;
      if (validate(c).length) probs.push(`${sp}:${validate(c)[0]}`);
      if (c.bodies.some((b) => b.shape.kind === 'cells')) kept++;
      const m = mutate(c, rng, 2);
      if (validate(m).length) probs.push(`${sp} mutated:${validate(m)[0]}`);
    }
  }
  check('cells.breeds-with-every-species', !probs.length && kept > n * 0.25, probs.slice(0, 3).join(' | ') || `${bySp.size} species x 6 crossovers (+ mutation) valid; cells inherited in ${kept}/${n}`);
  const rr = mulberry32(2024);
  let rc = 0;
  for (let i = 0; i < 600; i++) if (randomBody(rr).shape.kind === 'cells') rc++;
  check('cells.random', rc > 5 && rc < 120, `${rc}/600 random bodies are cell fields`);
  check('cells.name', nounKind(cb) === 'cells' && NOUN_POOLS.cells.includes(nameFor(repair({ ...cloneGenome(m12), chain: [] })).split(' ').pop()!), nameFor(m12));
}

// -------------------------------------------------- MilkDrop mining: hue map (brightness -> palette bands)

{
  const base = seedByOrigin('E07');
  const m14 = seedByOrigin('M14'), m15 = seedByOrigin('M15');
  const off = repair(JSON.parse(JSON.stringify({ ...base, tone: { p: { ...base.tone.p, huemap: undefined, bands: undefined, drift: undefined, solar: undefined, poster: undefined } } })));
  check('huemap.default-off', off.tone.p.huemap === 0 && off.tone.p.solar === 0 && off.tone.p.bands === TONE_SCHEMA.bands.def && !validate(off).length, JSON.stringify({ h: off.tone.p.huemap, s: off.tone.p.solar }));
  const on = cloneGenome(off);
  on.tone.p.huemap = 1;
  const sol = cloneGenome(off);
  sol.tone.p.solar = 0.5;
  check('huemap.cost', Math.abs(estimateCost(on) - estimateCost(off) - 0.03) < 1e-9 && Math.abs(estimateCost(sol) - estimateCost(off) - 0.03) < 1e-9 && estimateCost(m14) < COST_BUDGET_MS && estimateCost(m15) < COST_BUDGET_MS, `${estimateCost(off).toFixed(2)} -> ${estimateCost(on).toFixed(2)} ms; M14 ${estimateCost(m14).toFixed(2)}, M15 ${estimateCost(m15).toFixed(2)}`);
  const bad = repair({ ...cloneGenome(on), tone: { p: { ...on.tone.p, huemap: 3, bands: 0, drift: -1, solar: 9, poster: NaN } } });
  check('huemap.repair', bad.tone.p.huemap === 1 && bad.tone.p.bands === 0.5 && bad.tone.p.drift === 0 && bad.tone.p.solar === 1 && bad.tone.p.poster === 0 && !validate(bad).length, JSON.stringify(bad.tone.p));
  const src = buildSources(on);
  check('huemap.glsl', src.composite.includes('c = hueMapped(max(c, 0.0));') && src.composite.includes('uniform vec4 uHueMap') && !src.feedback.includes('hueMapped'), 'the composite maps the finished picture; the feedback pass is untouched');
  const rng = mulberry32(6161);
  const probs: string[] = [];
  let kept = 0, n = 0;
  const bySp = new Map<string, Genome>();
  for (const x of SEEDS) if (!bySp.has(classify(x.genome).primary)) bySp.set(classify(x.genome).primary, x.genome);
  for (const [sp, other] of bySp) {
    for (let k = 0; k < 6; k++) {
      const c = k % 2 ? crossover(m14, other, rng) : crossover(other, m15, rng);
      n++;
      if (validate(c).length) probs.push(`${sp}:${validate(c)[0]}`);
      if (c.tone.p.huemap > 0) kept++;
      const m = mutate(c, rng, 2);
      if (validate(m).length) probs.push(`${sp} mutated:${validate(m)[0]}`);
    }
  }
  check('huemap.breeds-with-every-species', !probs.length && kept > n * 0.3, probs.slice(0, 3).join(' | ') || `${bySp.size} species x 6 crossovers (+ mutation) valid; hue map inherited in ${kept}/${n}`);
  const rr = mulberry32(31);
  let hOn = 0, sOn = 0;
  for (let i = 0; i < 300; i++) { const g = randomGenome(rr); if (g.tone.p.huemap > 0) hOn++; if (g.tone.p.solar > 0) sOn++; }
  check('huemap.rare-in-random', hOn > 3 && hOn < 70 && sOn > 1 && sOn < 60, `${hOn}/300 hue-mapped, ${sOn}/300 solarized random genomes`);
  const hits = SEEDS.slice(0, 24).filter((x) => ADJ_POOLS.psychedelic.includes(nameFor(repair({ ...cloneGenome(x.genome), tone: { p: { ...x.genome.tone.p, huemap: 1 } } })).split(' ')[0])).length;
  check('huemap.name', hits >= 4 && classify(m14).label.includes('plasma'), `${hits}/24 originals named psychedelic when hue-mapped; M14 is ${classify(m14).label}`);
}

// -------------------------------------------------- MilkDrop mining: tunnel fold op

{
  const m16 = seedByOrigin('M16'), m17 = seedByOrigin('M17');
  check('tunnel.schema', OP_KINDS.includes('tunnel') && OP_SCHEMAS.tunnel === TUNNEL_SCHEMA && m16.chain.some((o) => o.op === 'tunnel' && o.stage === 'view'), 'a stage-free fold op with its own schema');
  const bad = repair({ ...cloneGenome(m16), chain: [{ op: 'tunnel', stage: 'view', w: 3, p: { depth: 9, speed: -7, twist: 0.2, sides: 7, rep: 2.6, fog: NaN, lock: 0.3 } }] });
  const bp = bad.chain[0].p;
  check('tunnel.repair', !validate(bad).length && bad.chain[0].stage === 'view' && bad.chain[0].w === 1 && bp.depth === 0.6 && bp.speed === -1 && bp.sides === 6 && bp.rep === 3 && bp.fog === TUNNEL_SCHEMA.fog.def && bp.lock === 0.25, JSON.stringify(bp));
  const view = buildSources(m16);
  const warpG = repair({ ...cloneGenome(m16), chain: [{ ...m16.chain[1], stage: 'warp' }] });
  check('tunnel.glsl', view.composite.includes('OB') === false && view.composite.includes('float z = uOpA[1].x / max(r, 1e-3)') && buildSources(warpG).feedback.includes('float z = uOpA[0].x / max(r, 1e-3)'), 'the tunnel mapping builds into the view (composite) or warp (feedback) stage');
  check('tunnel.cost', estimateCost(m16) < COST_BUDGET_MS && estimateCost(m17) < COST_BUDGET_MS, `M16 ${estimateCost(m16).toFixed(2)} ms, M17 ${estimateCost(m17).toFixed(2)} ms`);
  const qr = mulberry32(5511);
  let tunnels = 0;
  const rbad: string[] = [];
  for (let i = 0; i < 800; i++) {
    const o = randomOp(qr);
    if (o.op !== 'tunnel') continue;
    tunnels++;
    const g = repair({ ...cloneGenome(seedByOrigin('E05')), chain: [o] });
    if (validate(g).length) rbad.push(validate(g)[0]);
  }
  check('tunnel.random', tunnels > 5 && !rbad.length, rbad[0] ?? `${tunnels}/800 random ops were tunnels, all valid`);
  const rng = mulberry32(4141);
  const probs: string[] = [];
  let kept = 0, n = 0;
  const bySp = new Map<string, Genome>();
  for (const x of SEEDS) if (!bySp.has(classify(x.genome).primary)) bySp.set(classify(x.genome).primary, x.genome);
  for (const [sp, other] of bySp) {
    for (let k = 0; k < 6; k++) {
      const c = k % 2 ? crossover(m16, other, rng) : crossover(other, m17, rng);
      n++;
      if (validate(c).length) probs.push(`${sp}:${validate(c)[0]}`);
      if (c.chain.some((o) => o.op === 'tunnel') || c.bodies.some((b) => b.shape.kind === 'flame')) kept++;
      const m = mutate(c, rng, 2);
      if (validate(m).length) probs.push(`${sp} mutated:${validate(m)[0]}`);
    }
  }
  check('tunnel.breeds-with-every-species', !probs.length && kept > n * 0.25, probs.slice(0, 3).join(' | ') || `${bySp.size} species x 6 crossovers (+ mutation) valid; tunnel (or its flame-transform form) inherited in ${kept}/${n}`);
  check('tunnel.species-name', classify(m16).primary === 'vortex' && classify(m17).label.includes('vortex'), `M16 ${classify(m16).label} (${nameFor(m16)}), M17 ${classify(m17).label}`);
}

void (repairBody as unknown);
void (PLACE_KINDS as unknown as Locus);
console.log(`\n${failures ? 'FAILED' : 'PASSED'}: ${failures} failing check(s)`);
process.exit(failures ? 1 : 0);
