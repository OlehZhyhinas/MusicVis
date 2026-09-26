// Checks for the compound shape (several primitives blended into one distance field) and the ring
// halo material option (src/v2/genes/compound.ts); run from v2-test.ts.

import {
  bodyBxCount, bodyCost, cloneGenome, estimateCost, repair, repairBody, structuralKey, validate, type BodyGene, type Genome,
} from '../src/v2/genome';
import { crossover, mulberry32, mutate, randomBody } from '../src/v2/ops';
import { SEEDS, part } from '../src/v2/seeds';
import { buildSources } from '../src/v2/glsl';
import { nounKind } from '../src/v2/naming';
import {
  COMPOUND_PART_COST, MAX_PARTS, PART_SCHEMA, PART_STEP, RINGS_COST, compoundField, type CompoundPart,
} from '../src/v2/genes/compound';

type Check = (name: string, ok: boolean, detail: string) => void;

function withBody(b: unknown): Genome {
  const g = cloneGenome(SEEDS[0].genome);
  g.bodies = [repairBody(b)];
  g.reactions = [];
  delete g.drift;
  return repair(g);
}
const compoundBody = (parts: CompoundPart[], size = 0.2, material: Record<string, number> = {}): unknown => ({
  shape: { kind: 'compound', p: { size }, parts },
  place: { kind: 'point' },
  material: { kind: 'fill', p: material },
  emit: { kind: 'none' },
});
const both = (g: Genome) => { const s = buildSources(g); return s.feedback + s.composite; };

export function compoundChecks(check: Check): void {
  // ---------------------------------------------------------------- default / identity
  {
    const g = withBody({ shape: { kind: 'compound' }, place: { kind: 'point' }, material: { kind: 'fill' }, emit: { kind: 'none' } });
    const sh = g.bodies[0].shape;
    const parts = sh.parts ?? [];
    let err = 0;
    for (let i = 0; i < 200; i++) {
      const x = Math.sin(i * 1.7) * 0.4, y = Math.cos(i * 2.3) * 0.4;
      err = Math.max(err, Math.abs(compoundField(sh.p.size, parts, x, y).d - (Math.hypot(x, y) - sh.p.size * 0.5)));
    }
    // Identity: a part moved and turned is the default part's field moved and turned.
    const moved = [part('box', 'union', { x: 0.5, y: -0.3, sx: 0.4, sy: 0.2, rot: 0.125 })];
    const base = [part('box', 'union', { sx: 0.4, sy: 0.2 })];
    let err2 = 0;
    const a = Math.PI / 4;
    for (let i = 0; i < 200; i++) {
      const lx = Math.sin(i * 1.3) * 0.5, ly = Math.cos(i * 0.7) * 0.5;
      const wx = (0.5 + Math.cos(a) * lx - Math.sin(a) * ly) * 0.2, wy = (-0.3 + Math.sin(a) * lx + Math.cos(a) * ly) * 0.2;
      err2 = Math.max(err2, Math.abs(compoundField(0.2, moved, wx, wy).d - compoundField(0.2, base, lx * 0.2, ly * 0.2).d));
    }
    const src = both(g);
    check('compound.default',
      parts.length === 1 && parts[0].prim === 0 && parts[0].op === 0 && err < 1e-9 && err2 < 1e-9 && !validate(g).length && src.includes('uniform vec4 uBx0[4];') && src.includes('cpEll(') && nounKind(g.bodies[0]) === 'figure',
      `a bare compound is one disc of radius size/2 (max error ${err.toExponential(1)}); a moved, turned part is the same field moved and turned (${err2.toExponential(1)}); valid; 4 extra uniforms`);
  }

  // ---------------------------------------------------------------- existing genomes unchanged
  {
    const bad: string[] = [];
    for (const e of SEEDS) {
      // Seeds written with the new genes (a compound shape or ring halos) are not "existing" genomes.
      if (e.genome.bodies.some((b) => b.shape.kind === 'compound' || (b.material.p.rings ?? 0) > 0)) continue;
      const raw = cloneGenome(e.genome) as unknown as { bodies: { material: { p: Record<string, number> } }[] };
      for (const b of raw.bodies) { delete b.material.p.rings; delete b.material.p.rgap; delete b.material.p.rfade; }
      const r = repair(raw);
      if (!r.bodies.every((b) => b.material.p.rings === 0)) bad.push(`${e.origin}:rings`);
      const s0 = both(e.genome);
      if (both(r) !== s0) bad.push(`${e.origin}:glsl`);
      if (/uBx|cpEll|RINGS/.test(s0)) bad.push(`${e.origin}:extra code`);
      if (structuralKey(r) !== structuralKey(e.genome) || /@r|compound/.test(structuralKey(r))) bad.push(`${e.origin}:key`);
    }
    check('compound.seeds', !bad.length, bad.slice(0, 4).join(' | ') || `${SEEDS.length} seeds (those using compounds or rings skipped): no compound or ring code, identical shaders without the new material params`);
  }

  // ---------------------------------------------------------------- max parts, clamping
  {
    const six = [0, 1, 2, 3, 0, 1].map((prim, i) => ({ ...part('ellipse', 'union', { x: i * 0.2 - 0.5, sx: 0.2, sy: 0.3, rot: 0.1 * i, m: 0.4 }), prim, op: i % 4 }));
    const g = withBody(compoundBody(six));
    const src = both(g);
    const n = g.bodies[0].shape.parts!.length;
    const opens = (src.match(/\{/g) ?? []).length, closes = (src.match(/\}/g) ?? []).length;
    const extra = withBody(compoundBody([...six, ...six], 0.2));
    const wild = withBody(compoundBody([{ prim: 9, op: 2, x: 7, sx: -1, rot: 3, hue: 4, bright: -1, junk: 1 } as unknown as CompoundPart]));
    const wp = wild.bodies[0].shape.parts![0];
    const stray = withBody({ shape: { kind: 'dot', parts: six }, place: { kind: 'point' }, material: { kind: 'fill' } });
    const fused = repairBody({ ...(compoundBody(six) as object), shape: { kind: 'dot' }, fuse: { shape: { kind: 'compound', parts: six }, p: { mode: 0 } } });
    const ok = n === MAX_PARTS && !validate(g).length && src.includes(`uniform vec4 uBx0[${1 + 2 * MAX_PARTS + MAX_PARTS / 2}];`)
      && ['cpEll(pp', 'cpCap(pp', 'cpBox(pp', 'cpTri(pp'].every((f) => src.includes(f)) && (src.match(/\bdi = cp/g) ?? []).length === MAX_PARTS * 2
      && !/\bBX\(|\bSA\b|\bSHP\b[^_]/.test(src) && opens === closes && bodyBxCount(g.bodies[0]) === 1 + 2 * MAX_PARTS + 3
      && extra.bodies[0].shape.parts!.length === MAX_PARTS && !validate(extra).length
      && wp.prim === 3 && wp.op === 0 && wp.x === PART_SCHEMA.x.max && wp.sx === PART_SCHEMA.sx.min && wp.rot === 0.5 && wp.hue === 1 && wp.bright === 0 && !('junk' in wp) && !validate(wild).length
      && !stray.bodies[0].shape.parts && !fused.fuse;
    check('compound.max', ok, `6 parts of every primitive and op build (both passes, all slots substituted, braces balanced), 12 clamp to 6, wild values clamp, the first part is the base (op 0), stray parts and a compound fuse are dropped`);
  }

  // ---------------------------------------------------------------- subtract (crescent) and cost
  {
    const moon = [part('ellipse', 'union', { sx: 1, sy: 1, hue: 0.1, bright: 1.5 }), part('ellipse', 'subtract', { x: 0.45, y: 0.2, sx: 0.9, sy: 0.9, k: 0 })];
    const f = (x: number, y: number) => compoundField(0.2, moon, x, y);
    const g = withBody(compoundBody(moon));
    const src = both(g);
    const cost2 = bodyCost(g.bodies[0]);
    const cost6 = bodyCost(withBody(compoundBody([...moon, ...moon, ...moon.slice(0, 2)])).bodies[0]);
    const ok = f(-0.15, 0).d < 0 && f(0.05, 0.02).d > 0 && f(0.3, 0).d > 0 && f(-0.15, 0).hue === 0.1 && f(-0.15, 0).bright === 1.5
      && src.includes('mix(d, -di, h)') && !validate(g).length && Math.abs(cost6 - cost2 - 4 * COMPOUND_PART_COST) < 1e-9;
    check('compound.subtract', ok, `a disc minus an offset disc is a crescent (horn inside, bite and outside out), keeps the base part's warm hue; each part costs ${COMPOUND_PART_COST} ms per evaluation (2 -> 6 parts: +${(cost6 - cost2).toFixed(2)} ms)`);
  }

  // ---------------------------------------------------------------- mutation bounds, breeding, rarity
  {
    const tower = [
      part('capsule', 'union', { y: 0, sx: 0.12, sy: 1.2, m: 0.6 }), part('ellipse', 'smooth', { y: 0.5, sx: 0.35, sy: 0.12, k: 0.05 }),
      part('box', 'union', { y: 1.4, sx: 0.015, sy: 0.3 }),
    ];
    const g0 = withBody(compoundBody(tower));
    const rng = mulberry32(7171);
    let invalid = 0, bigStep = 0, recount = 0, opChange = 0, lost = 0, touched = 0;
    for (let i = 0; i < 1500; i++) {
      const c = mutate(g0, rng, 1);
      if (validate(c).length) invalid++;
      const b = c.bodies.find((x) => x.shape.kind === 'compound');
      if (!b) { lost++; continue; }
      const ps = b.shape.parts!;
      if (ps.length < 1 || ps.length > MAX_PARTS) invalid++;
      if (ps.length !== tower.length) { recount++; continue; }
      let moved = false;
      ps.forEach((p, j) => {
        const q = tower[j];
        if (p.prim !== q.prim) return;
        if (p.op !== q.op) opChange++;
        const d = Math.max(Math.abs(p.x - q.x), Math.abs(p.y - q.y));
        if (d > 1e-9) moved = true;
        if (d > PART_STEP * 1.5 + 1e-9 && b.shape.p.size === g0.bodies[0].shape.p.size) bigStep++;
      });
      if (moved) touched++;
    }
    // Crossover of two compounds: parts come from the parents.
    const other = withBody(compoundBody([part('triangle', 'union', { sx: 0.5, sy: 0.8 }), part('box', 'subtract', { y: -0.5, sx: 0.6, sy: 0.2 })]));
    let crossBad = 0, mixed = 0;
    const keys = new Set([...tower, ...other.bodies[0].shape.parts!].map((p) => `${p.prim}${p.op}`));
    for (let i = 0; i < 200; i++) {
      const c = crossover(g0, other, rng);
      if (validate(c).length) crossBad++;
      const ps = c.bodies[0].shape.parts;
      if (!ps) continue;
      if (!ps.every((p, j) => keys.has(`${p.prim}${p.op}`) || (j === 0 && p.op === 0))) crossBad++;
      const ks = ps.map((p) => `${p.prim}${p.op}`);
      if (ks.some((k) => k === '10' || k === '01') && ks.some((k) => k === '30' || k === '22')) mixed++;
    }
    // Random genomes: compound rarely, with few parts.
    let comp = 0, many = 0;
    for (let i = 0; i < 3000; i++) {
      const b: BodyGene = randomBody(rng);
      if (b.shape.kind === 'compound') {
        comp++;
        if (b.shape.parts!.length > 3) many++;
      }
    }
    check('compound.mutation', !invalid && !bigStep && touched > 30 && recount < 120 && opChange < 60 && crossBad === 0 && mixed > 0 && comp > 3 && comp < 90 && !many,
      `1500 mutations: ${invalid} invalid, ${touched} nudge parts, none by more than ${PART_STEP * 1.5} units, ${recount} change the part count, ${opChange} a combine op, ${lost} swap the shape; crossovers ${crossBad} bad, ${mixed}/200 mix parts of both parents; ${comp}/3000 random bodies compound, ${many} with more than 3 parts`);
  }

  // ---------------------------------------------------------------- ring halos
  {
    const star = { shape: { kind: 'star', p: { r: 0.05 } }, place: { kind: 'point' }, material: { kind: 'glow', p: {} as Record<string, number> }, emit: { kind: 'none' } };
    const off = withBody(star);
    star.material.p = { rings: 4, rgap: 0.03, rfade: 0.7 };
    const on = withBody(star);
    const srcOn = both(on);
    const rng = mulberry32(99);
    let ringed = 0;
    for (let i = 0; i < 2000; i++) if ((randomBody(rng).material.p.rings ?? 0) > 0) ringed++;
    const comp = withBody(compoundBody([part('ellipse', 'union')], 0.2, { rings: 3 }));
    const ok = !/RINGS|uBx/.test(both(off)) && srcOn.includes('ex += RINGS_0(s, Q, p, sc);') && srcOn.includes('uniform vec4 uBx0[1];')
      && structuralKey(on) !== structuralKey(off) && Math.abs(estimateCost(on) - estimateCost(off) - RINGS_COST) < 1e-9 && !validate(on).length
      && both(comp).includes('uniform vec4 uBx0[4];') && ringed > 20 && ringed < 160;
    check('rings.halo', ok, `ring halos add no code when off; on: one nearest-ring lookup per evaluation (+${RINGS_COST} ms), a compound shares the extra array; ${ringed}/2000 random materials ringed`);
    // Rings obey the material's clip height: a clipped fill star shows no rings below the clip line either.
    const clipped = withBody({ ...star, material: { kind: 'fill', p: { rings: 3, rgap: 0.03, rfade: 0.7, clip: -0.1 } } });
    const srcClip = both(clipped);
    check('rings.clip', /ex \+= RINGS_0\(s, Q, p, sc\) \* smoothstep\(BD_0\(1\)\.z|ex \+= RINGS_0\(s, Q, p, sc\) \* smoothstep/.test(srcClip) && srcOn.includes('ex += RINGS_0(s, Q, p, sc);'), 'fill and textured rings are multiplied by the clip step; materials without a clip unchanged');
  }
}
