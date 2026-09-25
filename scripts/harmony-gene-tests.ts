// Tests for the harmony gene (src/v2/genes/harmony.ts): repair, validation, breeding with every
// species, cost, naming, shader wiring, and the motor (tension breaks symmetry, a resolution snaps
// it back). Called from v2-test.ts.

import { COST_BUDGET_MS, SPECIES, cloneGenome, estimateCost, repair, speciesScores, validate, type Genome } from '../src/v2/genome';
import { crossover, mulberry32, mutate } from '../src/v2/ops';
import { SEEDS } from '../src/v2/seeds';
import { ADJ_POOLS, nameFor } from '../src/v2/naming';
import { buildSources } from '../src/v2/glsl';
import { genomeGene } from '../src/v2/geneRegistry';
import { seedChecks, shapeGeneChecks } from './v2-physics';
import { packTonnetz, type TonnetzFrame } from '../src/v2/genes/tonnetz';
import {
  HARMONY_COST_MS, HARMONY_SCHEMA, HarmonyMotor, IDLE_HARMONY, repairHarmony, tensionShape, validateHarmony,
  type HarmonyGene, type HarmonyInputs, type HarmonyOut,
} from '../src/v2/genes/harmony';

type Check = (name: string, ok: boolean, detail: string) => void;

const withHarmony = (g: Genome, h: HarmonyGene = repairHarmony({})): Genome => repair({ ...cloneGenome(g), harmony: h });
const inputs = (x: Partial<HarmonyInputs> = {}): HarmonyInputs => ({ tension: 0, resolve: 0, chordPulse: 0, modPulse: 0, tonnetzX: 0.5, tonnetzY: 0.2887, keyWalk: 0, ...x });
const out = (): HarmonyOut => ({ ...IDLE_HARMONY });

export function harmonyGeneTests(check: Check): void {
  // --- Schema, repair, validation ---
  {
    const d = repairHarmony({});
    check('harmony.repair-defaults', Object.keys(HARMONY_SCHEMA).every((k) => d.p[k] === HARMONY_SCHEMA[k].def) && !validateHarmony(d).length, JSON.stringify(d.p));
    const bad = repairHarmony({ p: { brk: 7, style: 1.4, settle: 0.5, walk: NaN, junk: 3 } });
    check('harmony.repair-clamps', bad.p.brk === 1 && bad.p.style === 1 && bad.p.settle === 0.6 && bad.p.walk === HARMONY_SCHEMA.walk.def && !('junk' in bad.p), JSON.stringify(bad.p));
    check('harmony.validate-catches', validateHarmony({ p: { ...d.p, brk: 2, extra: 1 } }).length === 2, validateHarmony({ p: { ...d.p, brk: 2, extra: 1 } }).join(','));
    const g = withHarmony(SEEDS[0].genome);
    check('harmony.genome-repair-keeps', !!g.harmony && !validate(g).length && JSON.stringify(repair(g)) === JSON.stringify(g), validate(g).join(';'));
    check('harmony.registered', genomeGene('harmony')?.optional === true && !!genomeGene('harmony')?.glossary, '');
    check('harmony.cost', Math.abs(estimateCost(g) - estimateCost(SEEDS[0].genome) - HARMONY_COST_MS) < 1e-9, `${HARMONY_COST_MS} ms`);
  }

  // --- Seeds ---
  {
    const hs = SEEDS.filter((s) => /^H\d\d$/.test(s.origin));
    check('harmony.seeds', hs.length >= 1 && hs.every((s) => !!s.genome.harmony && !validate(s.genome).length && estimateCost(s.genome) < COST_BUDGET_MS), hs.map((s) => `${s.origin} ${s.name} ${estimateCost(s.genome).toFixed(2)}ms`).join(', '));
  }

  // --- Crossover with every species; carry; determinism without the gene ---
  {
    const rng = mulberry32(4242);
    const donor = SEEDS.find((s) => s.genome.harmony)!.genome;
    const plain = SEEDS.filter((x) => !x.genome.harmony);
    const bySpecies = new Map<string, Genome>();
    for (const s of plain) {
      const sc = speciesScores(s.genome);
      const top = SPECIES.reduce((a, b) => (sc[b] > sc[a] ? b : a));
      if (!bySpecies.has(top)) bySpecies.set(top, s.genome);
    }
    const bad: string[] = [];
    let carried = 0, one = 0;
    for (const [sp, g] of bySpecies) {
      for (let i = 0; i < 20; i++) {
        const kid = i % 2 ? crossover(donor, g, rng, (rng() - 0.5) * 2) : crossover(withHarmony(g), withHarmony(donor, repairHarmony({ p: { brk: 0.2 } })), rng);
        const errs = validate(kid);
        if (errs.length) bad.push(`${sp}:${errs.join(';')}`);
        if (i % 2) {
          one++;
          if (kid.harmony) carried++;
        } else if (!kid.harmony) bad.push(`${sp}: both parents had harmony, child none`);
      }
    }
    check('harmony.crossover-species', !bad.length && bySpecies.size >= 8, bad.slice(0, 3).join(' | ') || `${bySpecies.size} species crossed with a harmony parent, all valid`);
    check('harmony.crossover-carry', carried > one * 0.3 && carried < one * 0.7, `${carried}/${one}`);
    const x = crossover(plain[1].genome, plain[5].genome, mulberry32(7));
    const y = crossover(plain[1].genome, plain[5].genome, mulberry32(7));
    check('harmony.crossover-deterministic', JSON.stringify(x) === JSON.stringify(y) && !x.harmony, 'harmony-less parents give harmony-less children');
  }

  // --- Mutation ---
  {
    const rng = mulberry32(777);
    let gained = 0, lost = 0;
    const bad: string[] = [];
    for (let i = 0; i < 2000; i++) {
      const base = SEEDS[i % SEEDS.length].genome;
      const src = i % 2 && !base.harmony ? base : withHarmony(base);
      const g = mutate(src, rng, 0.3 + 2 * rng());
      const errs = validate(g);
      if (errs.length) bad.push(`#${i}:${errs.join(';')}`);
      if (!src.harmony && g.harmony) gained++;
      if (src.harmony && !g.harmony) lost++;
    }
    check('harmony.mutate', !bad.length && gained > 5 && gained < 200 && lost > 0, bad.slice(0, 3).join(' | ') || `gained ${gained}/1000, lost ${lost}/1000`);
  }

  // --- Naming ---
  {
    const pool = ADJ_POOLS.harmonic ?? [];
    const strong = withHarmony(SEEDS.find((s) => s.origin === 'E02')?.genome ?? SEEDS[1].genome, repairHarmony({ p: { brk: 1, snap: 1, warp: 1 } }));
    const name = nameFor(strong);
    check('harmony.naming', pool.length >= 8 && pool.includes(name.split(/\s+/)[0]), name);
  }

  // --- Shader wiring: fold ops read the loosening only when it is set ---
  {
    const g = SEEDS.find((s) => s.genome.harmony && s.genome.chain.some((o) => ['mirror', 'tile', 'polar', 'kaleido'].includes(o.op)));
    const src = g ? buildSources(g.genome) : null;
    const i = g ? g.genome.chain.findIndex((o) => ['mirror', 'tile', 'polar', 'kaleido'].includes(o.op)) : -1;
    check('harmony.shader-folds', !!src && (src.feedback + src.composite).includes(`uOpB[${i}].w != 0.0`), g?.origin ?? 'no harmony seed with a fold');
  }

  // --- Motor ---
  {
    const h = repairHarmony({ p: { brk: 1, warp: 1, snap: 1, settle: 0.3, walk: 0.1, modTurn: 0.01, calm: 0.5, kick: 0.5 } });
    const m = new HarmonyMotor();
    const o = out();
    const none = m.update(undefined, inputs({ tension: 1 }), 0.016, out());
    check('harmony.motor-identity-without-gene', JSON.stringify(none) === JSON.stringify(IDLE_HARMONY), JSON.stringify(none));
    m.update(h, inputs({ tension: 0 }), 0.016, o);
    check('harmony.motor-consonance-is-order', o.brk === 0 && o.warp === 0 && Math.abs(o.hue) < 1e-6 && o.zoom === 1 && o.sat < 1, JSON.stringify(o));
    const rise: number[] = [];
    for (let i = 0; i < 180; i++) {
      m.update(h, inputs({ tension: 0.8 }), 1 / 60, o);
      if (i % 30 === 29) rise.push(o.brk);
    }
    check('harmony.motor-tension-breaks', rise.every((v, i) => i === 0 || v >= rise[i - 1]) && rise[rise.length - 1] > 0.85 && o.warp > 0.85 && o.sat > 1, rise.map((v) => v.toFixed(2)).join(' '));
    check('harmony.tension-shape', tensionShape(0.1) === 0 && tensionShape(0.8) > 0.99 && tensionShape(0.4) > 0.2 && tensionShape(0.4) < 0.8, '');
    // The resolution: tension drops and the resolve pulse fires; the fold snaps back fast, overshooting a little.
    let minBrk = 1;
    let snapped = -1;
    let flash = 0;
    let r = 0.9;
    for (let i = 0; i < 60; i++) {
      m.update(h, inputs({ tension: 0.05, resolve: r }), 1 / 60, o);
      r *= Math.exp(-1 / 60 / 0.7);
      minBrk = Math.min(minBrk, o.brk);
      if (snapped < 0 && Math.abs(o.brk) < 0.1) snapped = i;
      if (i === 1) flash = o.exposure;
    }
    check('harmony.motor-resolve-snaps', snapped >= 0 && snapped <= 12 && minBrk < -0.02 && minBrk > -0.5 && Math.abs(o.brk) < 0.05, `back in ${snapped} frames, overshoot ${minBrk.toFixed(2)}, end ${o.brk.toFixed(3)}`);
    check('harmony.motor-resolve-flash', flash > 1.2, flash.toFixed(2));
    // Chord walk and modulation.
    const m2 = new HarmonyMotor();
    const a = m2.update(h, inputs({ tonnetzX: 1.5 }), 0.016, out()).hue; // G major: a fifth right of home
    const b = m2.update(h, inputs({ tonnetzX: -0.5 }), 0.016, out()).hue; // F major
    check('harmony.motor-walk', a > 0 && b < 0 && Math.abs(a + b) < 1e-9, `${a.toFixed(3)} ${b.toFixed(3)}`);
    for (let i = 0; i < 600; i++) m2.update(h, inputs({ keyWalk: 2 }), 1 / 60, o);
    check('harmony.motor-modulation-turns', Math.abs(o.roll - 0.02 * Math.PI * 2) < 0.01 && o.hue > 0.1, `roll ${o.roll.toFixed(3)} hue ${o.hue.toFixed(3)}`);
    const k = m2.update(h, inputs({ keyWalk: 2, chordPulse: 1 }), 0.016, out());
    check('harmony.motor-chord-kick', k.zoom > 1.01, k.zoom.toFixed(3));
  }

  // --- Tonnetz shape: the lattice walk ---
  shapeGeneChecks(check, 'tonnetz', 'vec2 tzPlane(', 'lattice', null);
  seedChecks(check, 'H04', 'tonnetz');
  {
    const E = new Float32Array(64 + 16);
    const mem: Record<string, number> = {};
    const p = { scale: 0.14, follow: 1, tilt: 0, nodes: 1, lines: 1, fill: 1, echo: 1, trail: 1, pulse: 1 };
    const P = (k: string) => p[k as keyof typeof p];
    const fr = (chord: number, x: number, y: number): TonnetzFrame => ({ chord, tonnetzX: x, tonnetzY: y, keyTonic: 0, tension: 0.3, chordPulse: 0, resolve: 0, loud: 0.5 });
    const key = (k: string) => 't.' + k;
    packTonnetz(E, 64, 8, 12, P, p, fr(0, 0.5, 0.2887), mem, key, 1 / 60); // C
    packTonnetz(E, 64, 8, 12, P, p, fr(7, 1.5, 0.2887), mem, key, 1 / 60); // G
    packTonnetz(E, 64, 8, 12, P, p, fr(9 + 12, 0.1667, 0.5774), mem, key, 1 / 60); // Am
    check('tonnetz.pack-chord', E[72] === 9 && E[73] === 1 && E[74] === 0, `${E[72]} ${E[73]} ${E[74]}`);
    check('tonnetz.trail', Math.abs(E[8] - 1.5) < 1e-6 && Math.abs(E[10] - 0.5) < 1e-6 && Math.abs(E[14] - 0.1667) < 1e-6, `${[...E.slice(8, 16)].map((v) => v.toFixed(2))}`);
    for (let i = 0; i < 300; i++) packTonnetz(E, 64, 8, 12, P, p, fr(9 + 12, 0.1667, 0.5774), mem, key, 1 / 60);
    check('tonnetz.camera-follows', Math.abs(E[69] - 0.1667) < 0.01 && Math.abs(E[70] - 0.5774) < 0.01, `${E[69].toFixed(3)} ${E[70].toFixed(3)}`);
  }
}
