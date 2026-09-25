// Tests for the choreography gene (src/v2/genes/choreo.ts). Called from v2-test.ts.

import { COST_BUDGET_MS, cloneGenome, estimateCost, repair, validate, type Genome } from '../src/v2/genome';
import { crossover, mulberry32, mutate } from '../src/v2/ops';
import { SEEDS } from '../src/v2/seeds';
import { Population } from '../src/v2/population';
import { ADJ_POOLS, nameFor } from '../src/v2/naming';
import { TimelineSampler } from '../src/analysis/TimelineSampler';
import type { AnalysisResult, LiveAudioFrame, Section } from '../src/types';
import {
  CHOREO_COST_MS, CHOREO_SCHEMA, IDENTITY_POSE, blendPoses, glideAmount, phraseArc, sceneFraming, buildRamp, cameraUniforms, choreoPose, cueOf, releaseEnv, repairChoreo, validateChoreo,
  type ChoreoCue, type ChoreoGene, type ChoreoPose,
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
    const CHOREO_TUNED = new Set(['V01']);
    check('choreo.genome-roundtrip', validate(g).length === 0 && JSON.stringify(repair(g)) === JSON.stringify(g), 'repair is idempotent with a choreo gene');
    const bad = cloneGenome(g);
    bad.choreo!.p.dim = 7;
    check('choreo.genome-validate', validate(bad).some((e) => e.startsWith('choreo')), validate(bad).join(','));
    check('choreo.seeds-untouched', SEEDS.every((s) => (s.origin.startsWith('C') || CHOREO_TUNED.has(s.origin)) ? true : !('choreo' in s.genome)), 'only C seeds and seeds tuned with choreography carry it');
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
    const plain = SEEDS.filter((x) => !x.genome.choreo);
    for (let i = 0; i < 600; i++) {
      const a = plain[Math.floor(rng() * plain.length)].genome;
      const b = plain[Math.floor(rng() * plain.length)].genome;
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
      const base = SEEDS[i % SEEDS.length].genome;
      const src = i % 2 && !base.choreo ? base : withChoreo(base);
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

  // Showcase seeds: C01.. carry a choreography; a population saved before them gains them exactly once.
  {
    const cs = SEEDS.filter((x) => /^C\d\d$/.test(x.origin));
    check('choreo.seeds', cs.length >= 1 && cs.every((x) => !!x.genome.choreo && validate(x.genome).length === 0 && estimateCost(x.genome) < COST_BUDGET_MS),
      cs.map((x) => `${x.origin} ${x.name} ${estimateCost(x.genome).toFixed(2)}ms`).join(', '));
    const base = Population.seeded(1);
    for (const x of cs) base.members.delete(`G0-${x.origin}`);
    base.get('G0-E07')!.likes = 2;
    const kid = base.addChild(crossover(SEEDS[4].genome, SEEDS[21].genome, mulberry32(5)), [base.get('G0-E05')!, base.get('G0-E22')!], 2);
    base.vote(kid.id, true);
    const loaded = Population.fromJSON(JSON.parse(JSON.stringify({ ...base.toJSON(), seedVersion: 6 })));
    const before = JSON.stringify(loaded.list().sort((a, b) => a.id.localeCompare(b.id)));
    const added = loaded.upgradeSeeds(9);
    const rest = JSON.stringify(loaded.list().filter((m) => !cs.some((x) => m.id === `G0-${x.origin}`)).sort((a, b) => a.id.localeCompare(b.id)));
    check('choreo.migrate-once', JSON.stringify(added) === JSON.stringify(cs.map((x) => `G0-${x.origin}`)) && loaded.upgradeSeeds(10).length === 0 && before === rest
      && loaded.get(kid.id)!.likes === 1 && loaded.get('G0-E07')!.likes === 2 && cs.every((x) => JSON.stringify(loaded.get(`G0-${x.origin}`)!.genome) === JSON.stringify(x.genome)),
      `added ${added.join(',')}; votes, seeds and bred children untouched`);
    // A seed-version-7 population (C01 already there, with a vote) gains only C02.
    const v7 = Population.seeded(1);
    v7.members.delete('G0-C02');
    v7.get('G0-C01')!.likes = 4;
    const l7 = Population.fromJSON(JSON.parse(JSON.stringify({ ...v7.toJSON(), seedVersion: 7 })));
    const add7 = l7.upgradeSeeds(9);
    check('choreo.migrate-v7', JSON.stringify(add7) === '["G0-C02"]' && l7.get('G0-C01')!.likes === 4 && l7.upgradeSeeds(10).length === 0, `added ${add7.join(',')}`);
  }

  // Look-ahead from the offline analysis.
  {
    const smp = new TimelineSampler(songResult());
    const at = (t: number) => {
      smp.reset();
      const s = smp.sample(t, 0.016, true, LIVE);
      return { ttd: s.timeToDrop!, since: s.sinceDrop!, bar: s.barSeconds!, prev: s.prevSectionLabel };
    };
    const a = at(15), b = at(25), c = at(55), d = at(85), e = at(5);
    check('choreo.lookahead', Math.abs(a.ttd - 5) < 1e-6 && a.since === Infinity && Math.abs(b.since - 5) < 1e-6 && Math.abs(b.ttd - 35) < 1e-6
      && b.prev === 'build' && e.prev === undefined && c.prev === 'verse'
      && Math.abs(c.ttd - 5) < 1e-6 && d.ttd === Infinity && Math.abs(d.since - 25) < 1e-6 && Math.abs(e.ttd - 15) < 1e-6 && a.bar === 2,
      `t15 ttd=${a.ttd} t25 since=${b.since} ttd=${b.ttd} t55 ttd=${c.ttd} t85 ttd=${d.ttd} since=${d.since} bar=${a.bar}s`);
  }

  // Envelopes.
  {
    const c = repairChoreo({ p: { lead: 4, curve: 1, relax: 2 } });
    const cue = (ttd: number, since = Infinity): ChoreoCue => ({ timeToDrop: ttd, sinceDrop: since, barSeconds: 2, label: 'verse', prevLabel: null, sinceSection: 0, sectionLen: 30, bars: 0 });
    const r = [buildRamp(c, cue(9)), buildRamp(c, cue(8)), buildRamp(c, cue(4)), buildRamp(c, cue(0.001))];
    check('choreo.build-ramp', r[0] === 0 && r[1] === 0 && Math.abs(r[2] - 0.5) < 1e-9 && r[3] > 0.99, r.map((x) => x.toFixed(3)).join(','));
    const e = [releaseEnv(c, cue(Infinity, 0)), releaseEnv(c, cue(Infinity, 2)), releaseEnv(c, cue(Infinity, 4)), releaseEnv(c, cue(Infinity, Infinity))];
    check('choreo.release-env', e[0] === 1 && Math.abs(e[1] - 0.25) < 1e-9 && e[2] === 0 && e[3] === 0, e.map((x) => x.toFixed(3)).join(','));
    const none = choreoPose(undefined, cue(1));
    const live = cueOf({ bpm: 120 } as never);
    // Anticipation and release poses.
    const g = repairChoreo({ p: { lead: 4, curve: 1, push: 0.2, roll: 0.01, drain: 1, dim: 0.5, punch: 1, relax: 1 } });
    const far = choreoPose(g, cue(100));
    const mid = choreoPose(g, cue(4));
    const peak = choreoPose(g, cue(1e-6));
    const hit = choreoPose(g, cue(100, 0));
    const after = choreoPose(g, cue(100, 2));
    check('choreo.pose-far', JSON.stringify(far) === JSON.stringify(IDENTITY_POSE), 'far from a drop: identity');
    check('choreo.pose-build', Math.abs(mid.zoom - 1.1) < 1e-6 && Math.abs(peak.zoom - 1.2) < 1e-4 && peak.sat < 0.2 && Math.abs(peak.exposure - 0.5) < 1e-4
      && Math.abs(peak.roll - 0.01 * 2 * Math.PI) < 1e-4 && mid.sat > peak.sat,
      `mid zoom ${mid.zoom.toFixed(3)} sat ${mid.sat.toFixed(2)}; peak zoom ${peak.zoom.toFixed(3)} sat ${peak.sat.toFixed(2)} exp ${peak.exposure.toFixed(2)}`);
    check('choreo.pose-release', Math.abs(hit.zoom - 1.15) < 1e-6 && hit.sat > 1.3 && hit.exposure > 1.3 && hit.roll === 0 && JSON.stringify(after) === JSON.stringify(IDENTITY_POSE),
      `drop zoom ${hit.zoom.toFixed(3)} sat ${hit.sat.toFixed(2)} exp ${hit.exposure.toFixed(2)}, settled after relax`);
    const bl = blendPoses([peak, far], [1, 1]);
    check('choreo.blend', Math.abs(bl.zoom - (peak.zoom + 1) / 2) < 1e-9 && JSON.stringify(blendPoses([], [])) === JSON.stringify(IDENTITY_POSE), `half-way zoom ${bl.zoom.toFixed(3)}`);

    // The camera never shows the scene's edge.
    const ident = cameraUniforms(IDENTITY_POSE, 16 / 9);
    const rng = mulberry32(31337);
    let worst = 0;
    for (let i = 0; i < 2000; i++) {
      const pose: ChoreoPose = { zoom: 1 + rng() * 0.5, roll: (rng() - 0.5) * 0.3, tx: (rng() - 0.5), ty: (rng() - 0.5), sat: 1, exposure: 1, hue: 0 };
      const a = [16 / 9, 9 / 16, 1, 21 / 9, 4 / 3][i % 5];
      const m = cameraUniforms(pose, a);
      for (const [dx, dy] of [[-0.5, -0.5], [0.5, -0.5], [-0.5, 0.5], [0.5, 0.5]]) {
        const u = 0.5 + (1 + m[0]) * dx + m[1] * dy + m[4];
        const v = 0.5 + m[2] * dx + (1 + m[3]) * dy + m[5];
        worst = Math.max(worst, -u, u - 1, -v, v - 1);
      }
    }
    check('choreo.camera-covers', ident.every((x) => Math.abs(x) < 1e-12) && worst <= 1e-9, `identity uniforms zero; worst corner overshoot ${worst.toExponential(1)}`);
    // Scenes: a framing per section type, reached by a cut or a glide; a dolly across each section.
    const sc = repairChoreo({ p: { frame: 1, shot: 0.37, glide: 2, dolly: 0.1, scene: 0.5, push: 0, punch: 0, drain: 0, dim: 0 } });
    const at = (label: 'verse' | 'chorus' | 'drop', prevLabel: 'verse' | 'chorus' | 'build' | null, sinceSection: number, sectionLen = 32): ChoreoPose =>
      choreoPose(sc, { timeToDrop: Infinity, sinceDrop: Infinity, barSeconds: 2, label, prevLabel, sinceSection, sectionLen, bars: 0 });
    const ch1 = sceneFraming(sc, 'chorus'), ch2 = sceneFraming(sc, 'chorus'), vs = sceneFraming(sc, 'verse');
    const off = sceneFraming(defaults(), 'drop');
    const shots = new Set([0.1, 0.3, 0.5, 0.7, 0.9].map((shot) => sceneFraming(repairChoreo({ p: { frame: 1, shot } }), 'chorus').tx.toFixed(4)));
    check('choreo.scene-framing', JSON.stringify(ch1) === JSON.stringify(ch2) && ch1.tx !== vs.tx && ch1.hue === 0.2 && vs.hue === 0 && ch1.zoom >= 1 && ch1.zoom <= 1.35
      && JSON.stringify(off) === JSON.stringify(IDENTITY_POSE) && shots.size === 5,
      `chorus zoom ${ch1.zoom.toFixed(3)} pan ${ch1.tx.toFixed(3)},${ch1.ty.toFixed(3)} hue ${ch1.hue}; verse pan ${vs.tx.toFixed(3)}; frame 0 identity; 5 shots differ`);
    const g0 = at('chorus', 'verse', 0), g1 = at('chorus', 'verse', 2), g2 = at('chorus', 'verse', 4), g3 = at('chorus', 'verse', 20);
    check('choreo.glide', Math.abs(g0.tx - vs.tx) < 1e-9 && Math.abs(g2.tx - ch1.tx) < 1e-9 && Math.abs(g1.tx - (vs.tx + ch1.tx) / 2) < 1e-9 && g3.tx === ch1.tx
      && glideAmount(repairChoreo({ p: { glide: 0 } }), { ...cue(1), sinceSection: 0 }) === 1,
      `pan ${g0.tx.toFixed(3)} -> ${g1.tx.toFixed(3)} -> ${g2.tx.toFixed(3)} over 2 bars; glide 0 is a cut`);
    const d0 = at('drop', 'build', 4), d1 = at('drop', 'build', 16), d2 = at('drop', 'build', 32);
    check('choreo.dolly', d1.zoom > d0.zoom && Math.abs(d2.zoom / d0.zoom - 1.1 / (1 + 0.1 * 4 / 32)) < 1e-9, `zoom ${d0.zoom.toFixed(3)} -> ${d1.zoom.toFixed(3)} -> ${d2.zoom.toFixed(3)} across the section`);
    // Phrase arc: swells across each phrase and is back to rest on the next phrase's downbeat.
    const ar = repairChoreo({ p: { arc: 0.1, phrase: 8, push: 0, punch: 0, drain: 0, dim: 0 } });
    const pa = (bars: number) => phraseArc(ar, { ...cue(Infinity), bars });
    const zs = [0.01, 4, 6.8, 7.5, 8, 12].map((b) => choreoPose(ar, { ...cue(Infinity), bars: b }).zoom);
    check('choreo.phrase-arc', pa(8) === 0 && pa(16) === 0 && Math.abs(pa(6.8) - 1) < 1e-9 && pa(4) > 0.4 && pa(7.5) < 1 && pa(7.5) > 0 && Math.abs(zs[2] - 1.1) < 1e-9
      && phraseArc(defaults(), { ...cue(Infinity), bars: 5 }) === 0, zs.map((z) => z.toFixed(3)).join(' '));
    check('choreo.no-gene-identity', JSON.stringify(none) === JSON.stringify(IDENTITY_POSE) && live.timeToDrop === Infinity && live.barSeconds === 2, 'no gene or no look-ahead: identity pose');
  }
}
