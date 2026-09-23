// Tests for the V2 genome operators (repair, validate, crossover, mutate,
// classify, population lineage/fitness/serialization, cull, glsl builders,
// migration of older formats, names).
// Run: node --import ./scripts/analysis-test.hooks.mjs scripts/v2-test.ts

import {
  COST_BUDGET_MS, DEFORM_KINDS, EMIT_KINDS, LOCI, LOCUS_KINDS, MATERIAL_KINDS, MOTION_KINDS, PLACE_KINDS, SHAPE_CLASS,
  SHAPE_KINDS, SHAPE_SCHEMAS, UNIQUE_SHAPES,
  classify, cloneGenome, estimateCost, locusSchema, repair, repairBody, sdfCapable, structuralKey, validate,
  type BodyGene, type Gene, type Genome, type Locus,
} from '../src/v2/genome';
import {
  addLayer, crossover, crossoverTagged, makeFuse, morphParams, mulberry32, mutate, randomBody, randomGene, randomGenome,
  randomOp, MUTATION_NAMES,
} from '../src/v2/ops';
import { SEEDS, SEED_VERSION } from '../src/v2/seeds';
import { Population, fitness, POPULATION_VERSION, uniqueName } from '../src/v2/population';
import { nameFor, nounKind, NOUN_POOLS, ADJ_POOLS, HUE_WORDS } from '../src/v2/naming';
import { buildSources } from '../src/v2/glsl';
import { repair as repairV2, upgradeV2, EMITTER_SCHEMAS as V2_SCHEMAS } from '../src/v2/legacy';

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
const defs = (schema: Record<string, { def: number }>) => Object.fromEntries(Object.entries(schema).map(([k, s]) => [k, s.def]));

// -------------------------------------------------------------- 1. seeds

{
  const origins = SEEDS.map((s) => s.origin);
  const expected = Array.from({ length: 24 }, (_, i) => `E${String(i + 1).padStart(2, '0')}`);
  check('seeds.count', SEEDS.length === 24, `${SEEDS.length} seeds`);
  check('seeds.order', JSON.stringify(origins) === JSON.stringify(expected), origins.join(','));
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
  check('seeds.validate', !badValid.length, badValid.join(' | ') || 'all 24 valid');
  check('seeds.repair-idempotent', !badIdem.length, badIdem.join(',') || 'repair(seed) === seed for all 24');
  check('seeds.serialization-roundtrip', !badTrip.length, badTrip.join(',') || 'all 24 survive JSON + repair unchanged');
  check('seeds.under-budget', !over.length, over.join(',') || `all under ${COST_BUDGET_MS} ms`);
  check('seeds.version', SEED_VERSION === 5, `SEED_VERSION=${SEED_VERSION}`);

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
  check('repair.fits-budget', estimateCost(heavy) <= COST_BUDGET_MS * 0.95 && heavy.bodies[0].place.p.count < 6, `count=${heavy.bodies[0].place.p.count} cost=${estimateCost(heavy).toFixed(2)}`);
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
      const other = randomGene('shape', rng, b) as BodyGene['shape'];
      if (!sdfCapable(other) || UNIQUE_SHAPES.includes(b)) continue;
      const f = makeFuse(body, other, rng, mode);
      if (!f) continue;
      const g = repair({ v: 5, chain: [randomOp(rng, 'swirl')], bodies: [f], carrier: { kind: 'warp', p: {} }, palette: { kind: 'triad', p: {} }, tone: { p: {} }, reactions: [{ src: 'bass', g: 'fu', i: 0, k: 'k', gain: 0.5 }], energy: [0.2, 0.8] });
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
  check('population.seeded-size', pop.size === 24, `size=${pop.size}`);
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
  check('migrate.v3-file-loads', !!pop && pop.size === 27, err || `size=${pop?.size}`);
  if (pop) {
    const all = pop.list();
    check('migrate.all-valid', all.every((m) => !validate(m.genome).length), all.filter((m) => validate(m.genome).length).map((m) => m.id).join(',') || 'every member valid');
    const k1 = pop.get('G1-0001')!, k2 = pop.get('G2-0002')!, k3 = pop.get('G3-0003')!;
    check('migrate.children-kept', k1.name === 'Wheeling Serpent' && k2.name === 'Starfish Pearl Cage' && k3.name === 'Old Flame' && k1.likes === 3 && k2.dislikes === 2 && k3.hidden && k1.cross === 'fused' && JSON.stringify(k2.parents) === JSON.stringify(['G1-0001', 'G0-E20']), `${k1.name} / ${k2.name} / ${k3.name}`);
    check('migrate.children-faithful', k1.genome.bodies[0].place.kind === 'walker' && k1.genome.bodies[0].deform.ops?.[0].op === 'twist' && k2.genome.bodies[0].fuse?.shape.kind === 'solid' && k2.genome.bodies[0].emit.kind === 'none' && k3.genome.bodies.length === 2, structuralKey(k2.genome));
    check('migrate.descriptors-dropped', k1.descriptor === undefined, 'converted children are measured again');
    const changed = pop.upgradeSeeds();
    const e02 = pop.get('G0-E02')!;
    check('migrate.seeds-replaced', changed.length === 24 && JSON.stringify(e02.genome) === JSON.stringify(seedByOrigin('E02')) && pop.seedVersion === SEED_VERSION, `${changed.length} seeds upgraded`);
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
  const back = [SEEDS.every((s) => JSON.stringify(repair(toV3(s.genome)).bodies.map((b) => b.color.kind)) === JSON.stringify(s.genome.bodies.map((b) => b.color.kind)))];
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

// -------------------------------------------------- 14. example crossovers

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

void (repairBody as unknown);
void (PLACE_KINDS as unknown as Locus);
console.log(`\n${failures ? 'FAILED' : 'PASSED'}: ${failures} failing check(s)`);
process.exit(failures ? 1 : 0);
