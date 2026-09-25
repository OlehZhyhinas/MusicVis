// Checks for the physics-inspired shape genes (volumetric beams, ...). Called from v2-test.ts,
// kept in its own file so parallel gene work does not collide in the main test script.

import {
  COST_BUDGET_MS, SHAPE_SCHEMAS, bodyCost, estimateCost, inRange, repair, validate,
  type BodyGene, type Genome, type ShapeKind,
} from '../src/v2/genome';
import { crossover, mulberry32, mutate, randomBody, randomGene } from '../src/v2/ops';
import { SEEDS } from '../src/v2/seeds';
import { Population } from '../src/v2/population';
import { buildSources } from '../src/v2/glsl';
import { nameFor, nounKind } from '../src/v2/naming';
import { MAX_BEAMS, beamsCost } from '../src/v2/genes/beams';
import { cymaticsTarget } from '../src/v2/genes/cymatics';

type Check = (name: string, ok: boolean, detail: string) => void;

/** A one-body genome around the given body (warp carrier, short trail). */
function around(b: unknown): Genome {
  return repair({ v: 5, chain: [], bodies: [b], carrier: { kind: 'warp', p: { halfLife: 0.3 } }, palette: { kind: 'triad', p: {} }, tone: { p: {} }, reactions: [], energy: [0.3, 0.9] });
}

/** Shared gene checks for a physics shape kind: repair, random bodies, crossover with every seed, mutation, cost, name. */
export function shapeGeneChecks(check: Check, kind: ShapeKind, marker: string, noun: string, shed: string | null): void {
  const schema = SHAPE_SCHEMAS[kind];
  const rng = mulberry32(4242 + kind.length);

  // Repair clamps every parameter, and a chunk shape keeps a single copy.
  const wild: Record<string, number> = {};
  for (const k of Object.keys(schema)) wild[k] = schema[k].max * 3 + 7;
  const fixed = around({ shape: { kind, p: wild }, place: { kind: 'orbit', p: { count: 5 } }, material: { kind: 'glow' }, emit: { kind: 'cover' } });
  const fb = fixed.bodies[0];
  check(`${kind}.repair`, !validate(fixed).length && fb.shape.kind === kind && Object.keys(schema).every((k) => inRange(fb.shape.p[k], schema[k])) && fb.place.p.count === 1 && fb.emit.kind === 'trail',
    validate(fixed).join(';') || `${JSON.stringify(fb.shape.p)} copies=${fb.place.p.count} emit=${fb.emit.kind}`);

  // Random bodies of this kind: valid, idempotent, build the field shader.
  const bad: string[] = [];
  for (let i = 0; i < 120; i++) {
    const g = around(randomBody(rng, kind));
    const errs = validate(g);
    if (errs.length) bad.push(errs.join(';'));
    if (JSON.stringify(repair(JSON.parse(JSON.stringify(g)))) !== JSON.stringify(g)) bad.push('roundtrip');
    const src = buildSources(g);
    const s = g.bodies[0].emit.kind === 'none' ? src.composite : src.feedback;
    if (!s.includes(marker)) bad.push('no field code');
    if (!(estimateCost(g) < COST_BUDGET_MS)) bad.push(`cost ${estimateCost(g).toFixed(2)}`);
  }
  check(`${kind}.random-bodies`, !bad.length, bad.slice(0, 4).join(' | ') || '120 random bodies valid, idempotent, under budget, build shaders');

  // Crossover with every seed (both orders) and mutation of the children stay valid and in budget.
  const host = around({ ...randomBody(rng, kind), shape: randomGene('shape', rng, kind) });
  const xbad: string[] = [];
  let carried = 0;
  let n = 0;
  for (const s of SEEDS) {
    for (let k = 0; k < 4; k++) {
      const child = k % 2 ? crossover(host, s.genome, rng) : crossover(s.genome, host, rng);
      const mut = mutate(child, rng, 1.5);
      for (const g of [child, mut]) {
        n++;
        const errs = validate(g);
        if (errs.length) xbad.push(`${s.origin}:${errs.join(';')}`);
        // Where this gene is expressed the genome is under budget, or the gene already gave up all it
        // can (its shed parameter at the minimum) and the rest is another layer's cost, which
        // screening rejects as too slow.
        const mine = g.bodies.find((b) => b.shape.kind === kind);
        const heavierLayer = mine && g.bodies.some((b) => b !== mine && bodyCost(b) > bodyCost(mine));
        if (mine && !(estimateCost(g) < COST_BUDGET_MS) && !(shed && mine.shape.p[shed] === schema[shed].min) && !heavierLayer) xbad.push(`${s.origin}:cost ${estimateCost(g).toFixed(2)}`);
        if (g.bodies.some((b) => b.shape.kind === kind)) carried++;
        buildSources(g);
      }
    }
  }
  check(`${kind}.crossover-every-seed`, !xbad.length && carried > n * 0.15, xbad.slice(0, 4).join(' | ') || `${n} children and mutants valid and under budget, ${carried} express ${kind}`);

  // Mutation of a genome of this kind keeps it valid; parameters move.
  let g = host;
  const mbad: string[] = [];
  let moved = 0;
  for (let i = 0; i < 300; i++) {
    const m = mutate(g, rng, 1);
    const errs = validate(m);
    if (errs.length) mbad.push(errs.join(';'));
    const a = g.bodies.find((b) => b.shape.kind === kind);
    const b = m.bodies.find((x) => x.shape.kind === kind);
    if (a && b && JSON.stringify(a.shape.p) !== JSON.stringify(b.shape.p)) moved++;
    if (m.bodies.some((x) => x.shape.kind === kind)) g = m;
  }
  check(`${kind}.mutation`, !mbad.length && moved >= 3, mbad.slice(0, 3).join(' | ') || `300 mutations valid, ${moved} moved the ${kind} parameters`);

  // A hidden-body slime emission names the preset after its network, so name a trail-emitting copy.
  const nb = { ...(host.bodies.find((b) => b.shape.kind === kind) as BodyGene), emit: { kind: 'trail', p: {} } } as BodyGene;
  const named = nameFor(host);
  check(`${kind}.name`, nounKind(nb) === noun, `${named} (${noun})`);
}

export function physicsChecks(check: Check): void {
  shapeGeneChecks(check, 'beams', 'hot = 0.02 / (0.02 + along', 'beams', 'count');
  const maxed = around({ shape: { kind: 'beams', p: { count: MAX_BEAMS, gobo: 3 } }, material: { kind: 'glow' } });
  check('beams.cost', Math.abs(beamsCost(maxed.bodies[0].shape.p) - (1.2 + MAX_BEAMS * 0.24)) < 1e-9 && estimateCost(maxed) < COST_BUDGET_MS * 0.75,
    `12 textured beams ${beamsCost(maxed.bodies[0].shape.p).toFixed(2)} ms, genome ${estimateCost(maxed).toFixed(2)} ms`);

  seedChecks(check, 'V01', 'beams');

  shapeGeneChecks(check, 'cymatics', 'float cymMode(', 'cymatics', null);
  // Modes follow the music: chords (relative to the key), bands, sections; m != n; major / minor sign.
  const base = { chroma: new Float32Array(12), spec: new Float32Array(64), keyTonic: 2, minor: false, sectionIndex: 0, bass: 0, loud: 0, bpm: 120 };
  const p = { ...around({ shape: { kind: 'cymatics' } }).bodies[0].shape.p };
  const ch = new Float32Array(12);
  ch[2] = 1; ch[9] = 0.8; // D and A in D: the tonic and the fifth
  const [m1, n1, s1] = cymaticsTarget(p, { ...base, chroma: ch }, ch);
  const [m2, n2, s2] = cymaticsTarget(p, { ...base, minor: true }, [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0.9]);
  const secs = new Set(Array.from({ length: 8 }, (_, k) => cymaticsTarget({ ...p, source: 2 }, { ...base, sectionIndex: k }, []).join(',')));
  let pairsOk = true;
  for (let top = 3; top <= 12; top++) for (let a = 0; a < 12; a++) for (let b = 0; b < 12; b++) {
    const s = new Array(12).fill(0);
    s[a] = 1;
    s[b] = a === b ? 1 : 0.5;
    const [m, n] = cymaticsTarget({ ...p, modes: top }, base, s);
    if (!(m >= 1 && n <= top && m < n)) pairsOk = false;
  }
  check('cymatics.modes', m1 === 1 && n1 === 2 && s1 === 1 && s2 === -1 && m2 !== n2 && secs.size >= 4 && pairsOk,
    `D+A in D -> (${m1},${n1}), minor sign ${s2}, ${secs.size} figures over 8 sections, every pair 1 <= m < n <= modes: ${pairsOk}`);
  seedChecks(check, 'V02', 'cymatics');
}

/** A showcase seed: valid, under budget, uses its gene; a population saved before it gains it exactly once. */
export function seedChecks(check: Check, origin: string, kind: ShapeKind): void {
  const seed = SEEDS.find((x) => x.origin === origin);
  if (!seed) {
    check(`${kind}.seed`, false, `no seed ${origin}`);
    return;
  }
  const g = seed.genome;
  check(`${kind}.seed`, !validate(g).length && estimateCost(g) < COST_BUDGET_MS && g.bodies.some((b) => b.shape.kind === kind) && JSON.stringify(repair(JSON.parse(JSON.stringify(g)))) === JSON.stringify(g),
    `${origin} ${seed.name}: ${estimateCost(g).toFixed(2)} ms, ${validate(g).join(';') || 'valid'}`);
  const base = Population.seeded(1);
  base.members.delete(`G0-${origin}`);
  base.get('G0-E07')!.likes = 2;
  base.get('G0-E12')!.hidden = true;
  const kid = base.addChild(crossover(SEEDS[4].genome, SEEDS[21].genome, mulberry32(5)), [base.get('G0-E05')!, base.get('G0-E22')!], 2);
  base.vote(kid.id, false);
  const loaded = Population.fromJSON(JSON.parse(JSON.stringify(base.toJSON())));
  loaded.seedVersion -= 1;
  const before = JSON.stringify(loaded.list().sort((a, b) => a.id.localeCompare(b.id)));
  const added = loaded.upgradeSeeds(9);
  const rest = JSON.stringify(loaded.list().filter((m) => m.id !== `G0-${origin}`).sort((a, b) => a.id.localeCompare(b.id)));
  const again = Population.fromJSON(JSON.parse(JSON.stringify(loaded.toJSON())));
  check(`${kind}.seed-migrates-once`, JSON.stringify(added) === JSON.stringify([`G0-${origin}`]) && before === rest && loaded.get(kid.id)!.dislikes === 1
    && loaded.get('G0-E07')!.likes === 2 && loaded.get('G0-E12')!.hidden && again.upgradeSeeds(10).length === 0 && JSON.stringify(loaded.get(`G0-${origin}`)!.genome) === JSON.stringify(g),
    `added ${added.join(',')}; votes, hidden seeds and bred children untouched; reload stable`);
}
