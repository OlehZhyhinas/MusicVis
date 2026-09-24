// Tests for the drift gene (src/v2/genes/drift.ts) and its planned path (src/v2/genes/driftPath.ts).
// Called from v2-test.ts.

import { COST_BUDGET_MS, cloneGenome, estimateCost, repair, structuralKey, validate, type Genome } from '../src/v2/genome';
import { crossover, mulberry32, mutate } from '../src/v2/ops';
import { SEEDS } from '../src/v2/seeds';
import { ADJ_POOLS, nameFor } from '../src/v2/naming';
import { genomeGene } from '../src/v2/geneRegistry';
import type { Section } from '../src/types';
import { DRIFT_COST_MS, DRIFT_HEADROOM, DRIFT_SCHEMA, crossDrift, repairDrift, validateDrift, type DriftGene } from '../src/v2/genes/drift';
import { blendToward, driftDistance, morphGenome, performedAt, planDrift, stopAt, stripDrift } from '../src/v2/genes/driftPath';

type Check = (name: string, ok: boolean, detail: string) => void;

const withDrift = (g: Genome, p: Partial<DriftGene['p']> = {}): Genome => repair({ ...cloneGenome(g), drift: { p: { ...repairDrift({}).p, ...p } } });

/** A song with a verse / chorus form, a breakdown (the bridge), a drop and an outro; 120 bpm (2 s bars). */
const SONG: Section[] = [
  { start: 0, end: 16, label: 'intro', energy: 0.2 },
  { start: 16, end: 48, label: 'verse', energy: 0.4 },
  { start: 48, end: 64, label: 'build', energy: 0.6 },
  { start: 64, end: 96, label: 'chorus', energy: 0.8 },
  { start: 96, end: 128, label: 'verse', energy: 0.45 },
  { start: 128, end: 160, label: 'chorus', energy: 0.85 },
  { start: 160, end: 192, label: 'breakdown', energy: 0.3 },
  { start: 192, end: 208, label: 'build', energy: 0.7 },
  { start: 208, end: 240, label: 'drop', energy: 1 },
  { start: 240, end: 256, label: 'outro', energy: 0.2 },
];
const BAR = 2;

export function driftTests(check: Check): void {
  const seed = (o: string) => SEEDS.find((s) => s.origin === o)!.genome;
  const plain = SEEDS.filter((s) => !s.genome.drift).map((s) => s.genome);

  // Schema, registry, repair, validation.
  {
    const d = repairDrift({});
    check('drift.defaults-valid', validateDrift(d).length === 0 && Object.keys(d.p).length === Object.keys(DRIFT_SCHEMA).length, JSON.stringify(d.p));
    const broken = repairDrift({ p: { step: 9, kinds: 1.4, morph: 3, bound: -1, junk: 1, seed: NaN } });
    check('drift.repair-clamps', !validateDrift(broken).length && broken.p.step === 1 && broken.p.kinds === 1 && broken.p.bound === DRIFT_SCHEMA.bound.min && !('junk' in broken.p), JSON.stringify(broken.p));
    check('drift.registered', genomeGene('drift')?.optional === true && !!genomeGene('drift')?.glossary, 'registered, optional, with a glossary');
    const g = withDrift(plain[0]);
    check('drift.genome-roundtrip', !validate(g).length && JSON.stringify(repair(g)) === JSON.stringify(g), validate(g).join(',') || 'idempotent');
    const bad = cloneGenome(g);
    bad.drift!.p.ret = 4;
    check('drift.genome-validate', validate(bad).some((e) => e.startsWith('drift')), validate(bad).join(','));
    check('drift.absent-stays-absent', !('drift' in repair(cloneGenome(plain[3]))), 'repair does not invent a drift');
  }

  // Cost: the worst case over the path is what estimateCost charges.
  {
    const base = plain[5];
    const g = withDrift(base);
    const expect = estimateCost(base) * (1 + DRIFT_HEADROOM) + DRIFT_COST_MS;
    check('drift.cost', Math.abs(estimateCost(g) - expect) < 1e-6, `${estimateCost(base).toFixed(3)} -> ${estimateCost(g).toFixed(3)}`);
    let bad = '';
    for (const s of SEEDS.slice(0, 40)) {
      const h = withDrift(s.genome, { kinds: 2, step: 1, bound: 0.5 });
      const plan = planDrift(h, SONG);
      if (plan.worstCost > estimateCost(h) + 1e-6 || plan.stops.some((st) => estimateCost(st.genome) > COST_BUDGET_MS)) bad += `${s.origin}=${plan.worstCost.toFixed(2)}>${estimateCost(h).toFixed(2)} `;
    }
    check('drift.path-within-cost', !bad, bad || 'every stop within the charged worst case and the budget (40 seeds, wildest settings)');
  }

  // Determinism, seeking, bounds, return to parent.
  {
    const home = withDrift(seed('E05'), { step: 0.6, ret: 1, bound: 0.4, morph: 2 });
    const a = planDrift(home, SONG);
    const b = planDrift(cloneGenome(home), SONG.map((s) => ({ ...s })));
    check('drift.plan-deterministic', JSON.stringify(a) === JSON.stringify(b), `${a.stops.length} stops`);
    const other = planDrift(withDrift(seed('E05'), { step: 0.6, ret: 1, bound: 0.4, morph: 2, seed: 0.9 }), SONG);
    check('drift.seed-changes-path', JSON.stringify(other.stops.map((s) => s.genome)) !== JSON.stringify(a.stops.map((s) => s.genome)), 'another seed, another journey');
    check('drift.one-stop-per-section', a.stops.length === SONG.length && a.stops.every((s, i) => s.start === SONG[i].start && s.label === SONG[i].label), a.stops.map((s) => s.label).join(','));
    check('drift.opens-home', JSON.stringify(a.stops[0].genome) === JSON.stringify(stripDrift(home)), 'the first section plays the saved preset');
    const moved = a.stops.filter((s) => s.dist > 0.001).length;
    check('drift.moves', moved >= 4, `${moved} of ${a.stops.length} sections differ from home: ${a.stops.map((s) => s.dist.toFixed(3)).join(' ')}`);
    // Return to parent: with ret 1 the second chorus and verse replay the first ones exactly.
    const c1 = a.stops[3], v2 = a.stops[4], c2 = a.stops[5];
    check('drift.return-exact', c2.parent === 3 && v2.parent === 1 && JSON.stringify(c2.genome) === JSON.stringify(c1.genome) && JSON.stringify(v2.genome) === JSON.stringify(a.stops[1].genome), `chorus 2 parent ${c2.parent}, verse 2 parent ${v2.parent}`);
    // Partial return: closer to the parent than without it, on average over seeds.
    let closer = 0, n = 0;
    for (const s of SEEDS.slice(0, 24)) {
      const r0 = planDrift(withDrift(s.genome, { step: 0.7, ret: 0, bound: 0.5, seed: 0.3 }), SONG);
      const r1 = planDrift(withDrift(s.genome, { step: 0.7, ret: 0.8, bound: 0.5, seed: 0.3 }), SONG);
      const d0 = driftDistance(r0.stops[5].genome, r0.stops[3].genome);
      const d1 = driftDistance(r1.stops[5].genome, r1.stops[3].genome);
      if (d0 === 0 && d1 === 0) continue;
      n++;
      if (d1 <= d0) closer++;
    }
    check('drift.return-partial', n > 0 && closer >= n * 0.8, `${closer}/${n} seeds: chorus 2 at least as close to chorus 1 with ret 0.8 as with 0`);
    // Bounds: every stop inside the wander bound, valid, and nothing breaks with extreme settings.
    let out = '';
    for (const s of SEEDS) {
      for (const p of [{ step: 1, bound: 0.05 }, { step: 1, bound: 0.5, kinds: 2 }, { step: 0.3, what: 2, kinds: 1 }]) {
        const h = withDrift(s.genome, p);
        const plan = planDrift(h, SONG);
        for (const st of plan.stops) {
          if (st.dist > h.drift!.p.bound + 1e-9) out += `${s.origin}:${st.label} dist ${st.dist.toFixed(3)} `;
          const errs = validate(st.genome);
          if (errs.length) out += `${s.origin}:${st.label} ${errs[0]} `;
          if (st.genome.drift) out += `${s.origin} path carries drift `;
        }
      }
    }
    check('drift.bounds', !out, out.slice(0, 400) || `every stop of ${SEEDS.length} seeds x 3 settings valid and within its bound`);
    // Breakdown wanders furthest (on average), outro heads back home.
    let bd = 0, vs = 0, outroHome = 0, cnt = 0;
    for (const s of SEEDS.slice(0, 30)) {
      const plan = planDrift(withDrift(s.genome, { step: 0.5, ret: 0.5, bound: 0.5, seed: 0.7 }), SONG);
      bd += plan.stops[6].dist;
      vs += plan.stops[1].dist;
      if (plan.stops[9].dist <= plan.stops[8].dist + 1e-9) outroHome++;
      cnt++;
    }
    check('drift.bridge-furthest', bd > vs, `mean breakdown ${(bd / cnt).toFixed(3)} vs verse ${(vs / cnt).toFixed(3)}`);
    check('drift.outro-home', outroHome >= cnt * 0.8, `${outroHome}/${cnt} outros closer to home than the drop`);
    // Kinds: parameters-only keeps the home structure on every stop.
    let structs = 0;
    for (const s of SEEDS.slice(0, 30)) {
      const h = withDrift(s.genome, { step: 1, kinds: 0, bound: 0.5 });
      const plan = planDrift(h, SONG);
      if (plan.stops.some((st) => st.key !== structuralKey(stripDrift(h)))) structs++;
    }
    check('drift.params-only-keeps-structure', structs === 0, `${structs} seeds changed structure with kinds 0`);
    let changed = 0;
    for (const s of SEEDS.slice(0, 30)) {
      const plan = planDrift(withDrift(s.genome, { step: 1, kinds: 2, bound: 0.5 }), SONG);
      if (new Set(plan.stops.map((st) => st.key)).size > 1) changed++;
    }
    check('drift.kinds-change-structure', changed >= 5, `${changed}/30 seeds change structure with kinds 2`);
    // Colour-only drift leaves every body's form alone.
    let formMoved = 0;
    for (const s of SEEDS.slice(0, 30)) {
      const h = withDrift(s.genome, { step: 1, what: 2, bound: 0.5 });
      const base = stripDrift(h);
      for (const st of planDrift(h, SONG).stops) if (JSON.stringify(st.genome.bodies.map((x) => x.shape)) !== JSON.stringify(base.bodies.map((x) => x.shape))) formMoved++;
    }
    check('drift.colour-only', formMoved === 0, `${formMoved} stops moved a shape with what=colour`);
  }

  // Performance layer: morphs, cuts and seeking.
  {
    const home = withDrift(seed('E07'), { step: 0.8, morph: 4, ret: 0.5, bound: 0.5 });
    const plan = planDrift(home, SONG);
    check('drift.stop-at', stopAt(plan, -5) === 0 && stopAt(plan, 0) === 0 && stopAt(plan, 16) === 1 && stopAt(plan, 250) === 9 && stopAt(plan, 999) === 9, 'binary search over section starts');
    const at = (t: number) => JSON.stringify(performedAt(plan, t, BAR).genome);
    // Seeking: the genome at a time does not depend on what was asked before.
    const times = [3, 17, 20, 70, 130, 170, 209, 250];
    const fwd = times.map(at);
    const rev = [...times].reverse().map(at).reverse();
    check('drift.seek-stable', JSON.stringify(fwd) === JSON.stringify(rev), 'same genome at each time in either order');
    // Morph: starts at the previous section's genome, arrives at this one's after `morph` bars.
    const s1 = plan.stops[1], s0 = plan.stops[0];
    const start = performedAt(plan, 16, BAR);
    const mid = performedAt(plan, 16 + 4, BAR);
    const end = performedAt(plan, 16 + 8, BAR);
    const d0 = driftDistance(start.genome, morphGenome(s0.genome, s1.genome, 0));
    check('drift.morph-start', d0 < 1e-9 && start.key === s1.key, `distance to the morph origin ${d0}`);
    check('drift.morph-arrives', end.genome === s1.genome, 'after morph bars the stop genome itself');
    const dm = driftDistance(mid.genome, s1.genome), ds = driftDistance(start.genome, s1.genome);
    check('drift.morph-continuous', ds === 0 || (dm < ds && dm > 0), `start ${ds.toFixed(4)} mid ${dm.toFixed(4)}`);
    check('drift.morph-valid', !validate(repair(mid.genome)).length && JSON.stringify(repair(mid.genome).bodies.map((b) => b.shape.kind)) === JSON.stringify(s1.genome.bodies.map((b) => b.shape.kind)), 'mid-morph genome valid, structure of the new section');
    check('drift.drop-cuts', plan.stops[8].cut && performedAt(plan, 208.01, BAR).genome === plan.stops[8].genome, 'a drop jumps straight to its genome');
    const noDrift = planDrift(stripDrift(home), SONG);
    check('drift.none-single-stop', noDrift.stops.length === 1 && performedAt(noDrift, 100, BAR).genome === noDrift.stops[0].genome, 'without the gene, home all song');
    const emptySong = planDrift(home, []);
    check('drift.no-sections', emptySong.stops.length === 1 && JSON.stringify(emptySong.stops[0].genome) === JSON.stringify(stripDrift(home)), 'an unanalysed song plays home');
    const x = blendToward(plan.stops[3].genome, stripDrift(home), 1);
    check('drift.blend-full', JSON.stringify(x) === JSON.stringify(stripDrift(home)), 'blendToward(t=1) is the target');
  }

  // Breeding: crossover and mutation keep the gene valid; children of drift-less parents stay drift-less.
  {
    const rng = mulberry32(99);
    let bad = 0, carried = 0, spurious = 0;
    const a = withDrift(SEEDS[1].genome, { step: 0.2 }), b = withDrift(SEEDS[2].genome, { step: 0.8, kinds: 2 });
    for (let i = 0; i < 40; i++) {
      const c = crossover(a, b, rng);
      if (validate(c).length) bad++;
      if (c.drift) carried++;
      const m = mutate(a, rng, 1);
      if (validate(m).length) bad++;
      if (crossover(SEEDS[3].genome, SEEDS[4].genome, rng).drift) spurious++;
    }
    check('drift.breeding-valid', bad === 0 && carried === 40, `${bad} invalid, ${carried}/40 children of two drifting parents drift`);
    check('drift.no-spurious', spurious === 0, `${spurious} children of drift-less parents drifted`);
    const one = crossDrift(a.drift, undefined, () => 0.1);
    const mix = crossDrift(a.drift, b.drift, () => 0.5);
    check('drift.cross', !!one && !!mix && !validateDrift(mix).length && mix.p.step > 0.2 && mix.p.step < 0.8, JSON.stringify(mix?.p));
    const nm = nameFor(withDrift(plain[0], { step: 1, bound: 0.5, kinds: 2 }));
    const pool = ADJ_POOLS.shifting ?? [];
    const others = Object.entries(ADJ_POOLS).filter(([k]) => k !== 'shifting').flatMap(([, v]) => v);
    check('drift.name-pool', pool.length >= 10 && pool.every((w) => !others.includes(w)), 'shifting adjectives are their own');
    check('drift.named', typeof nm === 'string' && nm.split(' ').length >= 2, nm);
  }
}
