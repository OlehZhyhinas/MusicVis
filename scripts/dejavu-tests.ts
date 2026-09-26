// Tests for the visual deja vu gene (src/v2/genes/dejavu.ts): gene rules, breeding, cost, naming,
// seeds and migration, and the planner's snapshot / recall timing on synthetic songs with repeated
// sections (the GPU side only executes the planner's decisions). Called from v2-test.ts.

import { COST_BUDGET_MS, cloneGenome, estimateCost, repair, validate, type Genome } from '../src/v2/genome';
import { crossover, mulberry32, mutate } from '../src/v2/ops';
import { SEEDS } from '../src/v2/seeds';
import { Population } from '../src/v2/population';
import { ADJ_POOLS, nameFor } from '../src/v2/naming';
import { TimelineSampler } from '../src/analysis/TimelineSampler';
import type { LiveAudioFrame, MusicState } from '../src/types';
import {
  DEJAVU_SCHEMA, DejaVuPlanner, dejavuCost, dejavuCueOf, dejavuMemoryBytes, evolution, imageAlpha, isPhaseKey, phaseOffset, recallEnvelopes, repairDejaVu, validateDejaVu,
  type DejaVuGene,
} from '../src/v2/genes/dejavu';
import { featureSong, RB, RC, RI, RV, FEATURE_BAR } from './repetition-tests';

type Check = (name: string, ok: boolean, detail: string) => void;

const defaults = (): DejaVuGene => repairDejaVu({});
const withDejaVu = (g: Genome, d: DejaVuGene = defaults()): Genome => repair({ ...cloneGenome(g), dejavu: d });
const LIVE: LiveAudioFrame = { bass: 0, mid: 0, treb: 0, bassAtt: 0, midAtt: 0, trebAtt: 0, waveform: new Float32Array(1024), spectrum: new Float32Array(512) };

interface Event {
  t: number;
  kind: 'snap' | 'evict' | 'recall-start' | 'recall-end';
  group: number;
  section: number;
  k?: number;
}

/**
 * Plays a song through the sampler and a planner at 60 fps, committing every planned snapshot (as the
 * GPU would), and logs snapshots, evictions and recall windows. `seekAt` jumps the playhead once.
 */
function play(song: ReturnType<typeof featureSong>, p: Partial<Record<string, number>>, seekAt?: [number, number]) {
  const gene = repairDejaVu({ p });
  const pl = new DejaVuPlanner(gene);
  const smp = new TimelineSampler(song);
  const ev: Event[] = [];
  let recalling = -1;
  let maxHeld = 0;
  const dt = 1 / 60;
  let seeked = false;
  const mem: Record<string, number> = { 'b0.csp': 0 };
  for (let t = 0; t < song.duration; t += dt) {
    if (seekAt && !seeked && t >= seekAt[0]) {
      seeked = true;
      t = seekAt[1];
      smp.reset();
    }
    const s = smp.sample(t, dt, true, LIVE);
    const cue = dejavuCueOf(s);
    const plan = pl.plan(cue);
    const r = plan.recall;
    if (r && r.image > 0 && recalling !== s.sectionIndex) {
      recalling = s.sectionIndex;
      ev.push({ t, kind: 'recall-start', group: r.group, section: s.sectionIndex, k: r.k });
    }
    if (recalling === s.sectionIndex && (!r || r.t >= 1)) {
      ev.push({ t, kind: 'recall-end', group: r?.group ?? -1, section: s.sectionIndex });
      recalling = -2;
    }
    if (plan.snap >= 0) {
      if (plan.evict >= 0) ev.push({ t, kind: 'evict', group: plan.evict, section: s.sectionIndex });
      pl.remember(cue, plan, { zoom: 1, roll: 0, tx: 0, ty: 0 }, s.keyHue, mem);
      ev.push({ t, kind: 'snap', group: plan.snap, section: s.sectionIndex });
    }
    mem['b0.csp'] += 0.05;
    maxHeld = Math.max(maxHeld, pl.records.size);
  }
  return { ev, maxHeld, pl };
}

const fmt = (ev: Event[]) => ev.map((e) => `${e.kind}:g${e.group}s${e.section}${e.k ? `k${e.k}` : ''}@${e.t.toFixed(1)}`).join(' ');

export function dejavuTests(check: Check): void {
  // Gene rules.
  {
    const d = defaults();
    check('dejavu.defaults-valid', validateDejaVu(d).length === 0 && Object.keys(d.p).length === Object.keys(DEJAVU_SCHEMA).length, JSON.stringify(d.p));
    const broken = repairDejaVu({ p: { recall: 9, blend: 3.5, res: 0.3, cap: 9, min: -1, keep: 0.7, evolve: NaN, junk: 1 } });
    check('dejavu.repair-clamps', validateDejaVu(broken).length === 0 && broken.p.recall === 1 && broken.p.blend === 4 && broken.p.res === 0.25 && broken.p.cap === 4
      && broken.p.min === DEJAVU_SCHEMA.min.min && broken.p.keep === 1 && broken.p.evolve === DEJAVU_SCHEMA.evolve.def && !('junk' in broken.p), JSON.stringify(broken.p));
    check('dejavu.validate-catches', validateDejaVu({ p: { ...d.p, res: 0.3 } }).includes('dejavu.res'), 'off-choice res reported');
    const g = withDejaVu(SEEDS[0].genome, broken);
    check('dejavu.genome-roundtrip', validate(g).length === 0 && JSON.stringify(repair(JSON.parse(JSON.stringify(g)))) === JSON.stringify(g), 'repair is idempotent with a dejavu gene');
    const bad = cloneGenome(g);
    bad.dejavu!.p.recall = 7;
    check('dejavu.genome-validate', validate(bad).some((e) => e.startsWith('dejavu')), validate(bad).join(','));
    check('dejavu.seeds-untouched', SEEDS.every((s) => (/^D\d\d$/.test(s.origin) || /^[UX]/.test(s.origin) ? true : !('dejavu' in s.genome))), 'only D seeds carry a dejavu gene');
    check('dejavu.absent-stays-absent', !('dejavu' in repair(cloneGenome(SEEDS[3].genome))), 'repair does not invent a dejavu gene');
    // Migration: a genome saved before the gene existed loads unchanged; a garbled gene is repaired.
    const old = JSON.parse(JSON.stringify(SEEDS[2].genome));
    const garbled = repair({ ...old, dejavu: { p: { recall: 'x', cap: 17 } } });
    check('dejavu.migration', JSON.stringify(repair(old)) === JSON.stringify(SEEDS[2].genome) && validate(garbled).length === 0 && garbled.dejavu!.p.cap === 4, JSON.stringify(garbled.dejavu));
  }

  // Cost and memory.
  {
    const worst = SEEDS.map((s) => s.genome).sort((a, b) => estimateCost(b) - estimateCost(a))[0];
    const plain = SEEDS.find((s) => s.origin === 'E07')!.genome;
    const d = estimateCost(withDejaVu(plain)) - estimateCost(plain);
    const w = withDejaVu(worst);
    check('dejavu.cost', Math.abs(d - dejavuCost(defaults().p)) < 1e-9 && estimateCost(w) <= COST_BUDGET_MS && validate(w).length === 0,
      `+${d.toFixed(3)} ms; costliest seed with dejavu ${estimateCost(w).toFixed(2)} ms`);
    const most = dejavuMemoryBytes({ ...defaults().p, res: 0.5, cap: 4 }, 2560, 1440);
    const dflt = dejavuMemoryBytes(defaults().p, 2560, 1440);
    check('dejavu.memory-bounded', most <= 32 * 1024 * 1024 && dflt <= 8 * 1024 * 1024, `worst ${(most / 1048576).toFixed(1)} MB, default ${(dflt / 1048576).toFixed(1)} MB at 1440p`);
  }

  // Crossover: every species pair stays valid; no rng drawn without a parent dejavu.
  {
    const rng = mulberry32(9191);
    const bad: string[] = [];
    let none = 0, carried = 0, one = 0, both = 0;
    const plain = SEEDS.filter((x) => !x.genome.dejavu);
    for (let i = 0; i < 600; i++) {
      const a = plain[Math.floor(rng() * plain.length)].genome;
      const b = plain[Math.floor(rng() * plain.length)].genome;
      const mode = i % 3;
      const pa = mode >= 1 ? withDejaVu(a) : a;
      const pb = mode === 2 ? withDejaVu(b, repairDejaVu({ p: { recall: 1, evolve: 0.9 } })) : b;
      const child = crossover(pa, pb, rng, (rng() - 0.5) * 2);
      const errs = validate(child);
      if (errs.length) bad.push(`#${i}:${errs.join(';')}`);
      if (mode === 0 && child.dejavu) none++;
      if (mode === 1) {
        one++;
        if (child.dejavu) carried++;
      }
      if (mode === 2) {
        if (!child.dejavu) bad.push(`#${i}: both parents had dejavu, child none`);
        else if (child.dejavu.p.recall < 0.6 - 1e-9 || child.dejavu.p.recall > 1 + 1e-9) bad.push(`#${i}: recall ${child.dejavu.p.recall} outside parents`);
        both++;
      }
    }
    check('dejavu.crossover-valid', !bad.length && none === 0, bad.slice(0, 4).join(' | ') || `${both} both-parent children blended, none invented`);
    check('dejavu.crossover-carry', carried > one * 0.3 && carried < one * 0.7, `${carried}/${one} single-parent children inherited it`);
    const x = crossover(SEEDS[1].genome, SEEDS[5].genome, mulberry32(7));
    const y = crossover(SEEDS[1].genome, SEEDS[5].genome, mulberry32(7));
    check('dejavu.crossover-deterministic', JSON.stringify(x) === JSON.stringify(y) && !x.dejavu, 'dejavu-less parents give dejavu-less children');
  }

  // Mutation: gained rarely, lost sometimes, always valid, across every seed.
  {
    const rng = mulberry32(6262);
    let gained = 0, lost = 0;
    const bad: string[] = [];
    for (let i = 0; i < 2000; i++) {
      const base = SEEDS[i % SEEDS.length].genome;
      const src = i % 2 && !base.dejavu ? base : withDejaVu(base);
      const g = mutate(src, rng, 0.3 + 2 * rng());
      const errs = validate(g);
      if (errs.length) bad.push(`#${i}:${errs.join(';')}`);
      if (!src.dejavu && g.dejavu) gained++;
      if (src.dejavu && !g.dejavu) lost++;
    }
    check('dejavu.mutate', !bad.length && gained > 3 && gained < 200 && lost > 0, bad.slice(0, 3).join(' | ') || `gained ${gained}/1000, lost ${lost}/1000`);
  }

  // Naming.
  {
    const others = Object.entries(ADJ_POOLS).filter(([k]) => k !== 'remembered').flatMap(([, v]) => v);
    const pool = ADJ_POOLS.remembered ?? [];
    check('dejavu.name-pool', pool.length >= 8 && pool.every((w) => !others.includes(w)), pool.join(','));
    const plain = cloneGenome(SEEDS[6].genome);
    check('dejavu.name-deterministic', nameFor(withDejaVu(plain)) === nameFor(withDejaVu(plain)), nameFor(withDejaVu(plain)));
  }

  // Showcase seeds D01..: carry the gene, valid, under budget; an older population gains them once.
  {
    const ds = SEEDS.filter((x) => /^D\d\d$/.test(x.origin));
    check('dejavu.seeds', ds.length >= 1 && ds.every((x) => !!x.genome.dejavu && validate(x.genome).length === 0 && estimateCost(x.genome) < COST_BUDGET_MS),
      ds.map((x) => `${x.origin} ${x.name} ${estimateCost(x.genome).toFixed(2)}ms`).join(', '));
    const base = Population.seeded(1);
    for (const x of ds) base.members.delete(`G0-${x.origin}`);
    base.get('G0-E07')!.likes = 3;
    const loaded = Population.fromJSON(JSON.parse(JSON.stringify({ ...base.toJSON(), seedVersion: 22 })));
    const before = JSON.stringify(loaded.list().sort((a, b) => a.id.localeCompare(b.id)));
    const added = loaded.upgradeSeeds(9);
    const rest = JSON.stringify(loaded.list().filter((m) => !ds.some((x) => m.id === `G0-${x.origin}`)).sort((a, b) => a.id.localeCompare(b.id)));
    check('dejavu.migrate-once', JSON.stringify(added) === JSON.stringify(ds.map((x) => `G0-${x.origin}`)) && loaded.upgradeSeeds(10).length === 0 && before === rest
      && loaded.get('G0-E07')!.likes === 3, `added ${added.join(',')}; everything else untouched`);
  }

  // Timeline: snapshot and recall timing on a pop song I V C V C B C' (last chorus a step up).
  const song = featureSong([RI(), RV(), RC(), RV(), RC(), RB(), RC(16, 2)]);
  const secs = song.sections;
  const bar = FEATURE_BAR;
  {
    const { ev } = play(song, { snap: 0.8, blend: 2, keep: 0, min: 0.65 });
    const snaps = ev.filter((e) => e.kind === 'snap');
    const at = (i: number) => secs[i].start + 0.8 * (secs[i].end - secs[i].start);
    check('dejavu.snap-first-appearances', snaps.length === 2 && snaps[0].section === 1 && snaps[1].section === 2
      && Math.abs(snaps[0].t - at(1)) < 0.02 && Math.abs(snaps[1].t - at(2)) < 0.02, fmt(ev));
    const starts = ev.filter((e) => e.kind === 'recall-start');
    check('dejavu.recall-on-return', starts.length === 3 && starts.map((e) => e.section).join() === '3,4,6' && starts.map((e) => e.k).join() === '1,1,2'
      && starts.every((e) => Math.abs(e.t - secs[e.section].start) < 0.05), fmt(ev));
    const ends = ev.filter((e) => e.kind === 'recall-end');
    check('dejavu.recall-window', ends.length === 3 && ends.every((e) => Math.abs(e.t - (secs[e.section].start + 2 * bar)) < 0.05), fmt(ends));
    check('dejavu.no-recall-novel', !starts.some((e) => [0, 1, 2, 5].includes(e.section)), 'first appearances and the bridge are never recalled');
    const again = play(song, { snap: 0.8, blend: 2, keep: 0, min: 0.65 });
    check('dejavu.deterministic', fmt(again.ev) === fmt(ev), 'same song, same events');
  }
  // keep=1: every appearance that returns later is re-remembered; the last chorus recalls the second.
  {
    const { ev } = play(song, { keep: 1, min: 0.65 });
    const snaps = ev.filter((e) => e.kind === 'snap').map((e) => e.section).join();
    const last = ev.filter((e) => e.kind === 'recall-start' && e.section === 6)[0];
    check('dejavu.keep-latest', snaps === '1,2,4' && !!last && last.k === 1, fmt(ev));
  }
  // min gates both remembering and recalling (the transposed last chorus is a weaker match).
  {
    const sim6 = new TimelineSampler(song).sample(secs[6].start + 1, 0.016, true, LIVE).repeatSim!;
    const { ev } = play(song, { min: Math.min(0.95, sim6 + 0.01) });
    check('dejavu.min-gate', sim6 < 0.95 && !ev.some((e) => e.kind === 'recall-start' && e.section === 6) && ev.some((e) => e.kind === 'recall-start' && e.section === 4), `last chorus sim ${sim6.toFixed(2)}: ${fmt(ev)}`);
  }
  // Cap: three returning groups with room for two; the least recently used is forgotten.
  {
    const abc = featureSong([RV(), RC(), RB(), RV(), RC(), RB()]);
    const { ev, maxHeld } = play(abc, { cap: 2, min: 0.6 });
    const evict = ev.filter((e) => e.kind === 'evict');
    const recalled = ev.filter((e) => e.kind === 'recall-start').map((e) => e.section).join();
    check('dejavu.cap', maxHeld <= 2 && evict.length === 1 && evict[0].group === 0 && recalled === '4,5', `held ≤ ${maxHeld}: ${fmt(ev)}`);
  }
  // Seeking: jumping back into the first chorus does not re-remember it (keep=0), and a jump past a
  // first appearance never recalls a memory that was never taken.
  {
    const { ev } = play(song, {}, [secs[4].start + 1, secs[2].start + 1]);
    const snapsC = ev.filter((e) => e.kind === 'snap' && e.section === 2).length;
    check('dejavu.seek-back', snapsC === 1, fmt(ev));
    const skip = play(song, {}, [1, secs[3].start - 0.5]);
    // Landing late in the first chorus remembers it on the spot; the verse, never seen, is not recalled.
    check('dejavu.seek-forward', !skip.ev.some((e) => e.kind === 'recall-start' && e.section === 3) && skip.ev.some((e) => e.kind === 'recall-start' && e.section === 4),
      fmt(skip.ev));
  }
  // Live input carries no repetition: nothing happens.
  {
    const pl = new DejaVuPlanner(defaults());
    const cue = dejavuCueOf({ time: 10, bpm: 120, section: { start: 0, end: 20, label: 'verse', energy: 0.5 } } as unknown as MusicState);
    const plan = pl.plan(cue);
    check('dejavu.live-idle', !cue.valid && plan.snap < 0 && !plan.recall, 'no repetition data, no snapshots, no recall');
  }

  // Recall effects: framing, colour, phases, image blend, evolution.
  {
    const g = repairDejaVu({ p: { frame: 1, hue: 1, motion: 1, evolve: 0 } });
    const pl = new DejaVuPlanner(g);
    const cueAt = (start: number, since: number, index: number, returnSim: number) =>
      ({ valid: true, group: 3, index, sim: index ? 1 : 0, returnSim, sectionStart: start, sinceSection: since, sectionLen: 32, barSeconds: 2, time: start + since });
    const c0 = cueAt(0, 26, 0, 1);
    const p0 = pl.plan(c0);
    pl.remember(c0, p0, { zoom: 1.3, roll: 0.2, tx: 0.05, ty: -0.02 }, 0.9, { 'b0.csp': 1, 'b0.mph': 5, junk: 3 });
    const rec = pl.records.get(3)!;
    check('dejavu.phase-keys', 'b0.csp' in rec.phases && !('junk' in rec.phases) && isPhaseKey('b1.ctb') && isPhaseKey('rp2') && !isPhaseKey('b0.hm'), Object.keys(rec.phases).join(','));
    check('dejavu.phase-offset', Math.abs(phaseOffset('b0.csp', 0, 2 * Math.PI + 0.1) - 0.1) < 1e-9 && phaseOffset('b0.ph', 0, 30) === 0 && phaseOffset('b0.ph', 0, 3) === 3
      && Math.abs(phaseOffset('b0.ctb', 0, 17) - 1) < 1e-9, 'angles wrap, bars wrap at 16, free phases cap');
    const mem = { 'b0.csp': 3, 'b0.mph': 5.5 };
    let pose = { zoom: 1, roll: 0, tx: 0, ty: 0, hue: 0 };
    for (let s = 0; s <= 7.01; s += 0.05) {
      const plan = pl.plan(cueAt(64, s, 1, 0));
      pose = { zoom: 1, roll: 0, tx: 0, ty: 0, hue: 0 };
      if (plan.recall) pl.apply(plan.recall, pose, 0.1, mem, 64 + s);
      if (Math.abs(s - 1) < 0.026) {
        check('dejavu.framing-recall', pose.zoom > 1.2 && pose.roll > 0.1 && pose.tx > 0.03 && Math.abs(pose.hue - (-0.2)) < 0.05, JSON.stringify(pose));
      }
    }
    check('dejavu.motion-rewind', Math.abs(mem['b0.csp'] - 1) < 1e-6 && Math.abs(mem['b0.mph'] - 5) < 1e-6, JSON.stringify(mem));
    check('dejavu.framing-releases', Math.abs(pose.zoom - 1) < 1e-9 && Math.abs(pose.hue - -0.2) < 1e-6, `after the window the camera is free, the colour holds: ${JSON.stringify(pose)}`);
    const e0 = recallEnvelopes(0), e1 = recallEnvelopes(0.1), e9 = recallEnvelopes(1);
    check('dejavu.envelopes', e0.image === 0 && e1.image > 0.95 && e9.image === 0 && e9.colour === 1 && e9.motion === 1, JSON.stringify([e0, e1, e9]));
    const a1 = imageAlpha(repairDejaVu({ p: { recall: 0.7 } }), 1, 1 / 30);
    const a2 = imageAlpha(repairDejaVu({ p: { recall: 0.7 } }), 1, 1 / 60);
    check('dejavu.alpha-framerate', Math.abs(1 - (1 - a2) ** 2 - a1) < 1e-9 && imageAlpha(repairDejaVu({ p: { recall: 0 } }), 1, 1 / 60) === 0, `30 fps ${a1.toFixed(4)} = two 60 fps steps`);
    const ev1 = evolution(repairDejaVu({ p: { evolve: 0.5 } }), 2, 1);
    const ev2 = evolution(repairDejaVu({ p: { evolve: 0.5 } }), 2, 2);
    const ev0 = evolution(repairDejaVu({ p: { evolve: 0 } }), 2, 3);
    check('dejavu.evolution', ev2.zoom > ev1.zoom && Math.abs(ev2.rot) > Math.abs(ev1.rot) && ev0.zoom === 1 && ev0.rot === 0 && ev0.hue === 0, JSON.stringify([ev1, ev2]));
  }
}
