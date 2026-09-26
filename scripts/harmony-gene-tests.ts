// Tests for the harmony gene (src/v2/genes/harmony.ts): repair, validation, breeding with every
// species, cost, naming, shader wiring, and the motor (same chord, same shape: home is clean,
// every chord holds its own fixed shape, changes glide, flickers are ignored). Called from v2-test.ts.

import { COST_BUDGET_MS, SPECIES, cloneGenome, estimateCost, repair, speciesScores, validate, type Genome } from '../src/v2/genome';
import { crossover, mulberry32, mutate } from '../src/v2/ops';
import { SEEDS } from '../src/v2/seeds';
import { ADJ_POOLS, nameFor } from '../src/v2/naming';
import { buildSources } from '../src/v2/glsl';
import { genomeGene } from '../src/v2/geneRegistry';
import { seedChecks, shapeGeneChecks } from './v2-physics';
import { packTonnetz, type TonnetzFrame } from '../src/v2/genes/tonnetz';
import {
  HARMONY_COST_MS, HARMONY_SCHEMA, HarmonyMotor, IDLE_HARMONY, chordShape, glideTime, repairHarmony, validateHarmony,
  type HarmonyGene, type HarmonyInputs, type HarmonyOut,
} from '../src/v2/genes/harmony';

type Check = (name: string, ok: boolean, detail: string) => void;

const withHarmony = (g: Genome, h: HarmonyGene = repairHarmony({})): Genome => repair({ ...cloneGenome(g), harmony: h });
const inputs = (x: Partial<HarmonyInputs> = {}): HarmonyInputs => ({ chord: 0, keyTonic: 0, minor: false, keyWalk: 0, bpm: 120, ...x });
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

  // --- Motor: same chord, same shape ---
  {
    const h = repairHarmony({ p: { brk: 1, warp: 1, snap: 1, settle: 0.3, walk: 0.1, modHue: 0.1, modTurn: 0.01, calm: 0.5, kick: 0.5 } });
    const dt = 1 / 60;
    const G = 7, F = 5, Am = 21, C = 0;
    const none = new HarmonyMotor().update(undefined, inputs({ chord: G }), dt, out());
    check('harmony.motor-identity-without-gene', JSON.stringify(none) === JSON.stringify(IDLE_HARMONY), JSON.stringify(none));
    const geo = (o: HarmonyOut) => [o.brk, o.bend, o.warp, o.phase, o.seed];
    const same = (a: number[], b: number[]) => a.every((v, i) => Math.abs(v - b[i]) < 1e-9);
    // Runs a chord for some seconds; returns the frames' outputs.
    const run = (m: HarmonyMotor, x: Partial<HarmonyInputs>, secs: number): HarmonyOut[] => {
      const r: HarmonyOut[] = [];
      for (let i = 0; i < Math.round(secs / dt); i++) r.push({ ...m.update(h, inputs(x), dt, out()) });
      return r;
    };
    const m = new HarmonyMotor();
    const home = run(m, { chord: C }, 1).at(-1)!;
    check('harmony.motor-home-is-clean', home.brk === 0 && home.warp === 0 && Math.abs(home.hue) < 1e-9 && home.zoom === 1 && home.sat < 1, JSON.stringify(home));
    const g1 = run(m, { chord: G }, 1.5);
    const f1 = run(m, { chord: F }, 1.5);
    const a1 = run(m, { chord: Am }, 1.5);
    const g2 = run(m, { chord: G }, 1.5);
    const f2 = run(m, { chord: F }, 1.5);
    const back = run(m, { chord: C }, 1.5);
    check('harmony.motor-chords-deform', g1.at(-1)!.brk > 0.3 && g1.at(-1)!.warp > 0.3 && f1.at(-1)!.brk > 0.3 && a1.at(-1)!.brk > 0.3, `${g1.at(-1)!.brk.toFixed(2)} ${f1.at(-1)!.brk.toFixed(2)} ${a1.at(-1)!.brk.toFixed(2)}`);
    check('harmony.motor-same-chord-same-shape', same(geo(g1.at(-1)!), geo(g2.at(-1)!)) && same(geo(f1.at(-1)!), geo(f2.at(-1)!)) && !same(geo(g1.at(-1)!), geo(f1.at(-1)!)),
      `${geo(g1.at(-1)!).map((v) => v.toFixed(3))} / ${geo(g2.at(-1)!).map((v) => v.toFixed(3))}`);
    // Held still for the whole chord once the glide is done.
    const settleF = Math.ceil((1.2 * glideTime(0.3) + 0.25 + 0.05) / dt);
    let drift = 0;
    for (const seg of [g1, f1, a1, g2, f2, back]) for (let i = settleF + 1; i < seg.length; i++) drift = Math.max(drift, ...geo(seg[i]).map((v, k) => Math.abs(v - geo(seg[i - 1])[k])));
    check('harmony.motor-holds-within-chord', drift === 0, `max frame change after the glide ${drift}`);
    // Glides: no single frame moves far (smoothstep over ~0.3 s), and the new shape is reached in time.
    let step = 0;
    const all = [...g1, ...f1, ...a1, ...g2, ...f2, ...back];
    for (let i = 1; i < all.length; i++) step = Math.max(step, Math.abs(all[i].warp - all[i - 1].warp), Math.abs(all[i].brk - all[i - 1].brk), Math.abs(all[i].bend - all[i - 1].bend));
    check('harmony.motor-glides', step < 0.12 && step > 0, `largest frame step ${step.toFixed(3)}`);
    const reached = g1.findIndex((o) => Math.abs(o.warp - g1.at(-1)!.warp) < 1e-9);
    check('harmony.motor-glide-time', reached > 0.25 / dt && reached * dt < 0.25 + glideTime(0.3) + 0.05, `${(reached * dt).toFixed(3)} s`);
    // A cadence home: a smooth glide back to the clean picture, with a lift of light and no zoom.
    const lift = Math.max(...back.map((o) => o.exposure));
    check('harmony.motor-cadence-glides-home', back.at(-1)!.brk === 0 && back.at(-1)!.warp === 0 && lift > 1.15 && back.every((o) => o.zoom === 1), `lift ${lift.toFixed(2)}`);
    check('harmony.motor-kick-is-light', g1.some((o) => o.exposure > 1.02) && all.every((o) => o.zoom === 1), '');
    // Flicker: a chord that does not hold for half a beat never changes the shape.
    const m3 = new HarmonyMotor();
    run(m3, { chord: G }, 1);
    const before = geo(m3.update(h, inputs({ chord: G }), dt, out()));
    for (let i = 0; i < 6; i++) {
      run(m3, { chord: F }, 0.1);
      run(m3, { chord: G }, 0.1);
    }
    const flick = run(m3, { chord: -1 }, 1); // unknown chord: hold
    check('harmony.motor-ignores-flicker', flick.every((o) => same(geo(o), before)) && m3.heldChord === G, `${geo(flick.at(-1)!).map((v) => v.toFixed(3))} vs ${before.map((v) => v.toFixed(3))}`);
    // Key relative: V in C major has the same shape as V in G major; i is clean in a minor key.
    const s1 = chordShape(G, 0, false);
    const s2 = chordShape(2, 7, false);
    check('harmony.shape-key-relative', s1.amt === s2.amt && s1.seed === s2.seed && s1.phase === s2.phase && s1.amt > 0, `${s1.amt} ${s2.amt}`);
    check('harmony.shape-minor-home', chordShape(Am, 9, true).home && chordShape(Am, 9, true).amt === 0 && !chordShape(9, 9, true).home, '');
    // Palette walk: a fifth up and a fifth down go opposite ways; modulations swing the hue and roll smoothly.
    const mw = new HarmonyMotor();
    const hg = run(mw, { chord: G }, 1).at(-1)!.hue;
    const hf = run(mw, { chord: F }, 1).at(-1)!.hue;
    check('harmony.motor-walk', hg > 0 && hf < 0, `${hg.toFixed(3)} ${hf.toFixed(3)}`);
    const mod = run(mw, { chord: F, keyWalk: 2 }, 10);
    let hueStep = 0;
    for (let i = 1; i < mod.length; i++) hueStep = Math.max(hueStep, Math.abs(mod[i].hue - mod[i - 1].hue));
    check('harmony.motor-modulation-swings', Math.abs(mod.at(-1)!.roll - 0.02 * Math.PI * 2) < 0.01 && mod.at(-1)!.hue - hf > 0.15 && hueStep < 0.01, `roll ${mod.at(-1)!.roll.toFixed(3)} hue step ${hueStep.toFixed(4)}`);
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
