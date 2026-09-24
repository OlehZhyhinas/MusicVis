// Tests for the choreography gene (src/v2/genes/choreo.ts). Called from v2-test.ts.

import { COST_BUDGET_MS, cloneGenome, estimateCost, repair, validate, type Genome } from '../src/v2/genome';
import { crossover, mulberry32, mutate } from '../src/v2/ops';
import { SEEDS } from '../src/v2/seeds';
import { ADJ_POOLS, nameFor } from '../src/v2/naming';
import { TimelineSampler } from '../src/analysis/TimelineSampler';
import type { AnalysisResult, LiveAudioFrame, Section } from '../src/types';
import {
  CHOREO_COST_MS, CHOREO_SCHEMA, IDENTITY_POSE, buildRamp, choreoPose, cueOf, releaseEnv, repairChoreo, validateChoreo,
  type ChoreoCue, type ChoreoGene,
} from '../src/v2/genes/choreo';

type Check = (name: string, ok: boolean, detail: string) => void;

const defaults = (): ChoreoGene => repairChoreo({});
const withChoreo = (g: Genome, c: ChoreoGene = defaults()): Genome => repair({ ...cloneGenome(g), choreo: c });

/** A song with a build into a drop at 20 s, a verse, then a build into a chorus (a second drop) at 60 s. */
function songResult(): AnalysisResult {
  const duration = 90;
  const frameRate = 10;
  const numFrames = duration * frameRate;
  const f = () => new Float32Array(numFrames);
  const stems = { drums: f(), bass: f(), vocals: f(), other: f() };
  const bpm = 120;
  const beats = new Float32Array(Math.floor(duration * 2));
  for (let i = 0; i < beats.length; i++) beats[i] = i * 0.5;
  const downbeats = beats.filter((_b, i) => i % 4 === 0);
  const sections: Section[] = [
    { start: 0, end: 10, label: 'intro', energy: 0.2 },
    { start: 10, end: 20, label: 'build', energy: 0.5 },
    { start: 20, end: 40, label: 'drop', energy: 0.9 },
    { start: 40, end: 52, label: 'verse', energy: 0.4 },
    { start: 52, end: 60, label: 'build', energy: 0.6 },
    { start: 60, end: 80, label: 'chorus', energy: 0.8 },
    { start: 80, end: 90, label: 'outro', energy: 0.2 },
  ];
  return {
    duration, frameRate, numFrames, stems, stemOnsets: { ...stems }, stemPresence: { ...stems }, complexity: f(), songComplexity: 0.5,
    loudness: f(), chroma: new Float32Array(numFrames * 12), bpm, beats, downbeats, beatsPerBar: 4, sections,
    keys: [{ start: 0, end: duration, tonic: 0, mode: 'major', confidence: 1 }],
  };
}

const LIVE: LiveAudioFrame = {
  bass: 1, mid: 1, treb: 1, bassAtt: 1, midAtt: 1, trebAtt: 1, waveform: new Float32Array(1024), spectrum: new Float32Array(512),
};

export function choreoTests(check: Check): void {
  // Schema, repair, validation.
  {
    const d = defaults();
    check('choreo.defaults-valid', validateChoreo(d).length === 0 && Object.keys(d.p).length === Object.keys(CHOREO_SCHEMA).length, JSON.stringify(d.p));
    const broken = repairChoreo({ p: { lead: 5, push: 9, roll: -1, drain: NaN, junk: 3 } });
    check('choreo.repair-clamps', validateChoreo(broken).length === 0 && broken.p.lead === 4 && broken.p.push === CHOREO_SCHEMA.push.max && !('junk' in broken.p), JSON.stringify(broken.p));
    check('choreo.validate-catches', validateChoreo({ p: { ...d.p, push: 5 } }).includes('choreo.push'), 'out-of-range push reported');
    const g = withChoreo(SEEDS[0].genome, broken);
    check('choreo.genome-roundtrip', validate(g).length === 0 && JSON.stringify(repair(g)) === JSON.stringify(g), 'repair is idempotent with a choreo gene');
    const bad = cloneGenome(g);
    bad.choreo!.p.dim = 7;
    check('choreo.genome-validate', validate(bad).some((e) => e.startsWith('choreo')), validate(bad).join(','));
    check('choreo.seeds-untouched', SEEDS.every((s) => !s.origin.startsWith('C') ? !('choreo' in s.genome) : true), 'existing seeds carry no choreo');
    check('choreo.absent-stays-absent', !('choreo' in repair(cloneGenome(SEEDS[3].genome))), 'repair does not invent a choreo gene');
  }

  // Cost.
  {
    const worst = SEEDS.map((s) => s.genome).sort((a, b) => estimateCost(b) - estimateCost(a))[0];
    const g = withChoreo(worst);
    const d = estimateCost(g) - estimateCost(repair(cloneGenome(worst)));
    check('choreo.cost', Math.abs(d - CHOREO_COST_MS) < 1e-9 && estimateCost(g) <= COST_BUDGET_MS, `+${d.toFixed(3)} ms, costliest seed with choreo ${estimateCost(g).toFixed(2)} ms`);
  }

  // Crossover: every species pair stays valid; no rng drawn without a parent choreo.
  {
    const rng = mulberry32(4242);
    const bad: string[] = [];
    let none = 0, carried = 0, one = 0, both = 0;
    for (let i = 0; i < 600; i++) {
      const a = SEEDS[Math.floor(rng() * SEEDS.length)].genome;
      const b = SEEDS[Math.floor(rng() * SEEDS.length)].genome;
      const mode = i % 3;
      const pa = mode >= 1 ? withChoreo(a) : a;
      const pb = mode === 2 ? withChoreo(b, repairChoreo({ p: { push: 0.3, lead: 2, drain: 0 } })) : b;
      const child = crossover(pa, pb, rng, (rng() - 0.5) * 2);
      const errs = validate(child);
      if (errs.length) bad.push(`#${i}:${errs.join(';')}`);
      if (mode === 0 && child.choreo) none++;
      if (mode === 1) {
        one++;
        if (child.choreo) carried++;
      }
      if (mode === 2) {
        if (!child.choreo) bad.push(`#${i}: both parents had choreo, child none`);
        else if (child.choreo.p.push < 0.12 - 1e-9 || child.choreo.p.push > 0.3 + 1e-9) bad.push(`#${i}: push ${child.choreo.p.push} outside parents`);
        both++;
      }
    }
    check('choreo.crossover-valid', !bad.length && none === 0, bad.slice(0, 4).join(' | ') || `${both} both-parent children blended, none invented`);
    check('choreo.crossover-carry', carried > one * 0.3 && carried < one * 0.7, `${carried}/${one} single-parent children inherited it`);
    const r1 = mulberry32(7), r2 = mulberry32(7);
    const x = crossover(SEEDS[1].genome, SEEDS[5].genome, r1);
    const y = crossover(SEEDS[1].genome, SEEDS[5].genome, r2);
    check('choreo.crossover-deterministic', JSON.stringify(x) === JSON.stringify(y) && !x.choreo, 'choreo-less parents give choreo-less children');
  }

  // Mutation: existing seeds gain it (rarely), lose it, and stay valid.
  {
    const rng = mulberry32(5150);
    let gained = 0, lost = 0;
    const bad: string[] = [];
    for (let i = 0; i < 2000; i++) {
      const src = i % 2 ? SEEDS[i % SEEDS.length].genome : withChoreo(SEEDS[i % SEEDS.length].genome);
      const g = mutate(src, rng, 0.3 + 2 * rng());
      const errs = validate(g);
      if (errs.length) bad.push(`#${i}:${errs.join(';')}`);
      if (!src.choreo && g.choreo) gained++;
      if (src.choreo && !g.choreo) lost++;
    }
    check('choreo.mutate', !bad.length && gained > 5 && gained < 200 && lost > 0, bad.slice(0, 3).join(' | ') || `gained ${gained}/1000, lost ${lost}/1000`);
  }

  // Naming.
  {
    const others = Object.entries(ADJ_POOLS).filter(([k]) => k !== 'staged').flatMap(([, v]) => v);
    const pool = ADJ_POOLS.staged ?? [];
    check('choreo.name-pool', pool.length >= 8 && pool.every((w) => !others.includes(w)), pool.join(','));
    const plain = cloneGenome(SEEDS[6].genome);
    check('choreo.name-deterministic', nameFor(withChoreo(plain)) === nameFor(withChoreo(plain)), nameFor(withChoreo(plain)));
  }

  // Look-ahead from the offline analysis.
  {
    const smp = new TimelineSampler(songResult());
    const at = (t: number) => {
      smp.reset();
      const s = smp.sample(t, 0.016, true, LIVE);
      return { ttd: s.timeToDrop!, since: s.sinceDrop!, bar: s.barSeconds! };
    };
    const a = at(15), b = at(25), c = at(55), d = at(85), e = at(5);
    check('choreo.lookahead', Math.abs(a.ttd - 5) < 1e-6 && a.since === Infinity && Math.abs(b.since - 5) < 1e-6 && Math.abs(b.ttd - 35) < 1e-6
      && Math.abs(c.ttd - 5) < 1e-6 && d.ttd === Infinity && Math.abs(d.since - 25) < 1e-6 && Math.abs(e.ttd - 15) < 1e-6 && a.bar === 2,
      `t15 ttd=${a.ttd} t25 since=${b.since} ttd=${b.ttd} t55 ttd=${c.ttd} t85 ttd=${d.ttd} since=${d.since} bar=${a.bar}s`);
  }

  // Envelopes.
  {
    const c = repairChoreo({ p: { lead: 4, curve: 1, relax: 2 } });
    const cue = (ttd: number, since = Infinity): ChoreoCue => ({ timeToDrop: ttd, sinceDrop: since, barSeconds: 2 });
    const r = [buildRamp(c, cue(9)), buildRamp(c, cue(8)), buildRamp(c, cue(4)), buildRamp(c, cue(0.001))];
    check('choreo.build-ramp', r[0] === 0 && r[1] === 0 && Math.abs(r[2] - 0.5) < 1e-9 && r[3] > 0.99, r.map((x) => x.toFixed(3)).join(','));
    const e = [releaseEnv(c, cue(Infinity, 0)), releaseEnv(c, cue(Infinity, 2)), releaseEnv(c, cue(Infinity, 4)), releaseEnv(c, cue(Infinity, Infinity))];
    check('choreo.release-env', e[0] === 1 && Math.abs(e[1] - 0.25) < 1e-9 && e[2] === 0 && e[3] === 0, e.map((x) => x.toFixed(3)).join(','));
    const none = choreoPose(undefined, cue(1));
    const live = cueOf({ bpm: 120 } as never);
    check('choreo.no-gene-identity', JSON.stringify(none) === JSON.stringify(IDENTITY_POSE) && live.timeToDrop === Infinity && live.barSeconds === 2, 'no gene or no look-ahead: identity pose');
  }
}
