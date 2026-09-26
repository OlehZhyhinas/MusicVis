// Tests for the groove gene (src/v2/genes/groove.ts). Called from v2-test.ts.

import { COST_BUDGET_MS, SIGNALS, SPECIES, classify, cloneGenome, estimateCost, repair, validate, type Genome } from '../src/v2/genome';
import { crossover, mulberry32, mutate } from '../src/v2/ops';
import { SEEDS } from '../src/v2/seeds';
import { ADJ_POOLS, nameFor } from '../src/v2/naming';
import { genomeGene } from '../src/v2/geneRegistry';
import {
  GROOVE_COST_MS, GROOVE_SCHEMA, grooveBeat, grooveClock, grooveOffset, offFrac, repairGroove, validateGroove,
  type GrooveGene, type GrooveOffset,
} from '../src/v2/genes/groove';
import type { GrooveStats } from '../src/types';

type Check = (name: string, ok: boolean, detail: string) => void;

const defaults = (): GrooveGene => repairGroove({});
const withGroove = (g: Genome, x: GrooveGene = defaults()): Genome => repair({ ...cloneGenome(g), groove: x });
const ST = (swing: number, push = 0, humanity = 0, synco = 0): GrooveStats => ({ swing, push, humanity, synco });

export function grooveGeneTests(check: Check): void {
  // Repair / validate / registry.
  {
    const d = defaults();
    check('groove.defaults-valid', validateGroove(d).length === 0 && Object.keys(d.p).length === Object.keys(GROOVE_SCHEMA).length && !!genomeGene('groove'), JSON.stringify(d.p));
    const broken = repairGroove({ p: { swing: 9, sub: 11, tick: 3.2, sway: NaN, junk: 1 } });
    check('groove.repair-clamps', validateGroove(broken).length === 0 && broken.p.swing === 1.5 && broken.p.sub === 8 && broken.p.tick === 4 && !('junk' in broken.p), JSON.stringify(broken.p));
    const g = withGroove(SEEDS[0].genome, broken);
    check('groove.genome-roundtrip', validate(g).length === 0 && JSON.stringify(repair(g)) === JSON.stringify(g), 'repair is idempotent with a groove gene');
    const bad = cloneGenome(g);
    bad.groove!.p.crisp = 4;
    check('groove.genome-validate', validate(bad).some((e) => e.startsWith('groove')), validate(bad).join(','));
    check('groove.seeds-untouched', SEEDS.every((s) => (/^[UX]/.test(s.origin) ? true : s.origin.startsWith('Q') ? !!s.genome.groove : !('groove' in s.genome))), 'only Q seeds carry a groove');
    check('groove.signals', ['swing', 'push', 'humanity', 'synco'].every((s) => (SIGNALS as readonly string[]).includes(s)), SIGNALS.join(','));
  }

  // Cost.
  {
    const worst = SEEDS.filter((s) => !s.genome.drift).map((s) => s.genome).sort((a, b) => estimateCost(b) - estimateCost(a))[0];
    const g = withGroove(worst);
    const d = estimateCost(g) - estimateCost(repair(cloneGenome(worst)));
    check('groove.cost', Math.abs(d - GROOVE_COST_MS) < 1e-9 && estimateCost(g) <= COST_BUDGET_MS, `+${d.toFixed(3)} ms, costliest seed with groove ${estimateCost(g).toFixed(2)} ms`);
  }

  // The feel: straight and quantized-free music leaves the clock alone; swing lands the off-beat late.
  {
    const g = repairGroove({ p: { swing: 1, crisp: 0, lean: 0 } });
    let maxd = 0;
    for (let b = 0; b < 8; b += 0.01) maxd = Math.max(maxd, Math.abs(grooveBeat(g, ST(0), b) - b));
    check('groove.straight-identity', maxd < 1e-9, `max shift ${maxd}`);
    const f = offFrac(1);
    const atOff = grooveBeat(g, ST(1), 3 + f);
    const beforeOff = grooveBeat(g, ST(1), 3.5);
    check('groove.swing-late', Math.abs(atOff - 3.5) < 1e-9 && beforeOff < 3.5 - 0.05, `motion half-beat reached at ${f.toFixed(3)} (value ${atOff.toFixed(3)}), at 0.5 still ${beforeOff.toFixed(3)}`);
    // Monotone and continuous for any setting.
    const rng = mulberry32(3);
    let ok = true;
    for (let t = 0; t < 40 && ok; t++) {
      const gg = repairGroove({ p: { swing: rng() * 1.5, sub: rng() < 0.5 ? 8 : 16, lean: rng(), crisp: rng(), tick: [1, 2, 4][t % 3] } });
      const st = ST(rng(), rng() * 2 - 1, rng(), rng());
      let prev = grooveBeat(gg, st, 0);
      for (let b = 0.002; b < 4; b += 0.002) {
        const v = grooveBeat(gg, st, b);
        if (v < prev - 1e-9 || v - prev > 0.2) ok = false;
        prev = v;
      }
    }
    check('groove.monotone', ok, 'warped beat never runs backwards or jumps');
    // Crisp: quantized music holds most of each tick, humanized music moves evenly.
    const c = repairGroove({ p: { swing: 0, crisp: 1, tick: 1, lean: 0 } });
    const held = grooveBeat(c, ST(0, 0, 0), 0.25);
    const loose = grooveBeat(c, ST(0, 0, 1), 0.25);
    check('groove.crisp-ticks', held < 0.02 && Math.abs(loose - 0.25) < 1e-9, `quarter beat in: quantized ${held.toFixed(4)}, human ${loose.toFixed(4)}`);
    const lean = grooveBeat(repairGroove({ p: { swing: 0, crisp: 0, lean: 1 } }), ST(0, 1), 2);
    check('groove.lean-drags', lean < 2 - 0.1, `laid back: beat 2 motion at ${lean.toFixed(3)}`);
    const clk = { bars: 1.1, spin: 5, barPhase: 0.1, beatPhase: 0.4 };
    grooveClock(g, ST(1), clk);
    check('groove.clock', Math.abs(clk.bars * 4 - grooveBeat(g, ST(1), 4.4)) < 1e-9 && Math.abs(clk.spin - (5 + 2 * Math.PI * (clk.bars - 1.1))) < 1e-9, JSON.stringify(clk));
  }

  // Copy offsets: jitter only when human, locked to onsets; accents only on weak slots.
  {
    const g = repairGroove({ p: { sway: 0, off: 0, crisp: 0, jitter: 1, accent: 1 } });
    const run = (st: GrooveStats, onsetAt: number): { moved: number; kick: number } => {
      const mem: Record<string, number> = {};
      const out: GrooveOffset = { dx: 0, dy: 0, da: 0, s: 1 };
      let moved = 0, kick = 0;
      for (let i = 0; i < 60; i++) {
        const beat = 1 + i / 60;
        grooveOffset(g, st, { beat, period: 0.5, dt: 1 / 60, onset: Math.abs(beat - onsetAt) < 0.02 ? 1 : 0 }, mem, 'b0.g', out);
        moved = Math.max(moved, Math.hypot(out.dx, out.dy));
        kick = Math.max(kick, out.s - 1);
      }
      return { moved, kick };
    };
    const tight = run(ST(0, 0, 0, 1), 1.5);
    const human = run(ST(0, 0, 1, 1), 1.5);
    const onBeat = run(ST(0, 0, 1, 1), 1.02);
    check('groove.jitter-human', tight.moved < 1e-9 && human.moved > 0.005, `tight ${tight.moved.toFixed(4)}, human ${human.moved.toFixed(4)}`);
    check('groove.accent-weak-slots', human.kick > 0.1 && onBeat.kick < 1e-9, `off-beat hit kick ${human.kick.toFixed(3)}, on-beat ${onBeat.kick.toFixed(3)}`);
  }

  // Crossover: every species pair stays valid; nothing drawn from the rng without a parent groove.
  {
    const rng = mulberry32(777);
    const bad: string[] = [];
    let none = 0, carried = 0, one = 0;
    const plain = SEEDS.filter((x) => !x.genome.groove);
    const bySpecies = SPECIES.map((sp) => plain.filter((s) => classify(s.genome).primary === sp)).filter((l) => l.length);
    for (let i = 0; i < 600; i++) {
      const la = bySpecies[i % bySpecies.length];
      const lb = bySpecies[Math.floor(rng() * bySpecies.length)];
      const a = la[Math.floor(rng() * la.length)].genome;
      const b = lb[Math.floor(rng() * lb.length)].genome;
      const mode = i % 3;
      const pa = mode >= 1 ? withGroove(a) : a;
      const pb = mode === 2 ? withGroove(b, repairGroove({ p: { swing: 0.2, sway: 0.06 } })) : b;
      const child = crossover(pa, pb, rng, (rng() - 0.5) * 2);
      const errs = validate(child);
      if (errs.length) bad.push(`#${i}:${errs.join(';')}`);
      if (!(estimateCost(child) < COST_BUDGET_MS)) bad.push(`#${i}: cost ${estimateCost(child).toFixed(2)}`);
      if (mode === 0 && child.groove) none++;
      if (mode === 1) {
        one++;
        if (child.groove) carried++;
      }
      if (mode === 2) {
        if (!child.groove) bad.push(`#${i}: both parents had a groove, child none`);
        else if (child.groove.p.swing < 0.2 - 1e-9 || child.groove.p.swing > 1 + 1e-9) bad.push(`#${i}: swing ${child.groove.p.swing} outside parents`);
      }
    }
    check('groove.crossover-valid', !bad.length && none === 0, bad.slice(0, 4).join(' | ') || `600 children across ${bySpecies.length} species, none invented`);
    check('groove.crossover-carry', carried > one * 0.3 && carried < one * 0.7, `${carried}/${one} single-parent children inherited it`);
  }

  // Mutation: gained rarely, lost, nudged, always valid.
  {
    const rng = mulberry32(8080);
    let gained = 0, lost = 0;
    const bad: string[] = [];
    for (let i = 0; i < 2000; i++) {
      const base = SEEDS[i % SEEDS.length].genome;
      const src = i % 2 && !base.groove ? base : withGroove(base);
      const g = mutate(src, rng, 0.3 + 2 * rng());
      const errs = validate(g);
      if (errs.length) bad.push(`#${i}:${errs.join(';')}`);
      if (!src.groove && g.groove) gained++;
      if (src.groove && !g.groove) lost++;
    }
    check('groove.mutate', !bad.length && gained > 5 && gained < 200 && lost > 0, bad.slice(0, 3).join(' | ') || `gained ${gained}/1000, lost ${lost}/1000`);
  }

  // Naming.
  {
    const swungG = withGroove(SEEDS[3].genome, repairGroove({ p: { swing: 1.4, sway: 0.07, crisp: 0.1, jitter: 0.8 } }));
    const tightG = withGroove(SEEDS[3].genome, repairGroove({ p: { swing: 0, sway: 0, off: 0, jitter: 0, crisp: 1 } }));
    const pools = [...(ADJ_POOLS.swung ?? []), ...(ADJ_POOLS.lockstep ?? [])];
    const others = Object.entries(ADJ_POOLS).filter(([k]) => k !== 'swung' && k !== 'lockstep').flatMap(([, v]) => v);
    check('groove.naming-pools', pools.length === 24 && pools.every((w) => !others.includes(w)) && new Set(pools).size === pools.length, 'swung / lockstep words are their own');
    check('groove.named', !!nameFor(swungG) && !!nameFor(tightG), `${nameFor(swungG)} / ${nameFor(tightG)}`);
  }
}
