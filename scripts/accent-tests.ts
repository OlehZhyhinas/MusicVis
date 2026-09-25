// Tests for the accents gene (src/v2/genes/accent.ts). Called from v2-test.ts.

import { cloneGenome, repair, validate, type Genome } from '../src/v2/genome';
import { crossover, mulberry32, mutate } from '../src/v2/ops';
import { SEEDS } from '../src/v2/seeds';
import { IDENTITY_POSE, type ChoreoCue, type ChoreoPose } from '../src/v2/genes/choreo';
import { ACCENT_SCHEMA, accentPlan, applyAccents, crossAccent, hasHitResponse, repairAccent, setAccentOverride, validateAccent, type AccentInput } from '../src/v2/genes/accent';
import { genomeGene } from '../src/v2/geneRegistry';

type Check = (name: string, ok: boolean, detail: string) => void;

const cue = (over: Partial<ChoreoCue> = {}): ChoreoCue => ({
  timeToDrop: Infinity, sinceDrop: Infinity, barSeconds: 2, label: 'verse', prevLabel: null, sinceSection: 10, sectionLen: 30, bars: 5, ...over,
});
const input = (over: Partial<AccentInput> = {}): AccentInput => ({ cue: cue(), hookOn: 0, hookPulse: 0, hookNotePulse: 0, hookNote: -1, hookId: -1, hit: 0, ...over });
const pose = (): ChoreoPose => ({ ...IDENTITY_POSE });
const same = (a: ChoreoPose, b: ChoreoPose) => (Object.keys(a) as (keyof ChoreoPose)[]).every((k) => Math.abs(a[k] - b[k]) < 1e-9);
const fmt = (q: ChoreoPose) => JSON.stringify(Object.fromEntries(Object.entries(q).map(([k, v]) => [k, +v.toFixed(4)])));

export function accentTests(check: Check): void {
  const base = SEEDS[0].genome;
  const spec = genomeGene('accent');
  check('accent.registered', !!spec && spec.optional && !!spec.glossary && Object.keys(ACCENT_SCHEMA).every((k) => (spec.schemas as Record<string, unknown>)[k]), spec?.title ?? 'missing');

  // Repair / validate.
  const r = repairAccent({ p: { hook: 7, junk: 1 } });
  check('accent.repair', r.p.hook === 1 && !('junk' in r.p) && validateAccent(r).length === 0, JSON.stringify(r));
  const g = repair({ ...cloneGenome(base), accent: { p: { hook: 0.2 } } } as Genome);
  check('accent.genome-repair', g.accent?.p.hook === 0.2 && validate(g).length === 0, validate(g).join(','));
  check('accent.validate-bad', validate({ ...cloneGenome(base), accent: { p: { hook: 3 } } } as Genome).some((e) => e.startsWith('accent')), 'out of range hook');

  // Plan: defaults without the gene, the gene's values with it, off under the harness switch.
  const plain = cloneGenome(base);
  delete plain.accent;
  const p0 = accentPlan(plain);
  check('accent.default-on', p0.hook === ACCENT_SCHEMA.hook.def, JSON.stringify(p0));
  const off = accentPlan({ ...cloneGenome(plain), accent: { p: { ...repairAccent({}).p, hook: 0 } } });
  check('accent.gene-off', off.hook === 0, JSON.stringify(off));
  const hooked = cloneGenome(plain);
  hooked.reactions = [{ src: 'hook', g: 'col', i: 0, k: 'exposure', gain: 0.3, atk: 0.005, rel: 0.2, thr: 0, q: 0, div: 1 }];
  check('accent.hook-reaction-opts-out', accentPlan(hooked).hook === 0 && accentPlan({ ...hooked, accent: repairAccent({}) }).hook === 0, 'a preset reacting to the hook signals keeps only its own response');
  setAccentOverride(false);
  const forced = accentPlan(plain);
  setAccentOverride(null);
  check('accent.override', forced.hook === 0 && accentPlan(plain).hook > 0, JSON.stringify(forced));

  // Pose: identity with no hook; the same pose for the same moment of two repeats (the rhyme);
  // different notes nudge different ways.
  const plan = { ...accentPlan(plain), section: 0, hue: 0, drop: 0, kick: 0 };
  const q0 = applyAccents(plan, input(), pose());
  check('accent.identity-outside-hooks', same(q0, IDENTITY_POSE as ChoreoPose), fmt(q0));
  const a = applyAccents(plan, input({ hookOn: 1, hookNotePulse: 0.8, hookNote: 2, hookId: 0, cue: cue({ bars: 12.3 }) }), pose());
  const b = applyAccents(plan, input({ hookOn: 1, hookNotePulse: 0.8, hookNote: 2, hookId: 0, cue: cue({ bars: 40.3, label: 'chorus' }) }), pose());
  check('accent.rhymes', same(a, b) && a.zoom > 1, `${fmt(a)} vs ${fmt(b)}`);
  const c = applyAccents(plan, input({ hookOn: 1, hookNotePulse: 0.8, hookNote: 1, hookId: 0 }), pose());
  check('accent.notes-differ', Math.hypot(a.tx - c.tx, a.ty - c.ty) > 0.005 && Math.sign(a.roll) !== Math.sign(c.roll), `${fmt(a)} vs ${fmt(c)}`);
  const start = applyAccents(plan, input({ hookOn: 1, hookPulse: 1, hookNotePulse: 1, hookNote: 0, hookId: 0 }), pose());
  check('accent.bounded', start.zoom < 1.15 && Math.abs(start.hue) < 0.1 && start.exposure === 1, fmt(start));

  // Sections: each type its own framing and hue, the same every time it comes back; a quick glide
  // at the boundary; a punch on drops that settles within a bar; parts the choreography does are its own.
  const sp = accentPlan(plain);
  const verse = applyAccents(sp, input({ cue: cue({ label: 'verse', sinceSection: 5 }) }), pose());
  const verse2 = applyAccents(sp, input({ cue: cue({ label: 'verse', sinceSection: 9, prevLabel: 'chorus', bars: 50 }) }), pose());
  const chorus = applyAccents(sp, input({ cue: cue({ label: 'chorus', sinceSection: 5, prevLabel: 'verse' }) }), pose());
  const dz = (x: ChoreoPose, y: ChoreoPose) => Math.abs(x.zoom - y.zoom) + Math.abs(x.tx - y.tx) + Math.abs(x.ty - y.ty) + Math.abs(x.hue - y.hue);
  check('accent.section-consistent', same(verse, verse2), `${fmt(verse)} vs ${fmt(verse2)}`);
  check('accent.section-differs', dz(verse, chorus) > 0.05 && Math.abs(chorus.hue - verse.hue) > 0.05, `${fmt(verse)} vs ${fmt(chorus)}`);
  const edge = applyAccents(sp, input({ cue: cue({ label: 'chorus', sinceSection: 0, prevLabel: 'verse' }) }), pose());
  const half = applyAccents(sp, input({ cue: cue({ label: 'chorus', sinceSection: 0.17, prevLabel: 'verse' }) }), pose());
  check('accent.section-glide', same(edge, verse) && dz(half, verse) > 0.01 && dz(half, chorus) > 0.01, `${fmt(edge)} / ${fmt(half)}`);
  const d0 = applyAccents(sp, input({ cue: cue({ label: 'drop', sinceSection: 2, sinceDrop: 0 }) }), pose());
  const d1 = applyAccents(sp, input({ cue: cue({ label: 'drop', sinceSection: 2, sinceDrop: 2.5 }) }), pose());
  check('accent.drop-punch', d0.zoom > d1.zoom * 1.05 && d0.exposure > 1.1 && d1.exposure === 1 && d0.exposure < 1.2, `${fmt(d0)} vs ${fmt(d1)}`);
  const choreo = repair({ ...cloneGenome(plain), choreo: { p: { frame: 0.5, scene: 0, punch: 0.6 } }, accent: repairAccent({ p: { frame: 0.5 } }) } as Genome);
  const cp = accentPlan(choreo);
  check('accent.choreo-not-doubled', cp.frame === 0 && cp.drop === 0 && cp.hue > 0 && cp.section > 0 && cp.hook > 0, JSON.stringify(cp));
  check('accent.section-no-push', verse.zoom === 1 && chorus.zoom === 1 && verse.exposure < 1 && chorus.exposure === 1 && chorus.sat > verse.sat, `default sections change light and colour, not the framing: ${fmt(verse)} / ${fmt(chorus)}`);
  const framed = applyAccents({ ...sp, frame: 1 }, input({ cue: cue({ label: 'chorus', sinceSection: 5 }) }), pose());
  check('accent.frame-opt-in', framed.zoom > 1, fmt(framed));
  setAccentOverride(['hook']);
  const only = accentPlan(plain);
  setAccentOverride(null);
  check('accent.override-parts', only.hook > 0 && only.section === 0 && only.hue === 0 && only.frame === 0 && only.drop === 0 && only.kick === 0, JSON.stringify(only));

  // Kick: a small punch on drum hits, by default only for presets that do not answer hits themselves.
  const noHits = cloneGenome(plain);
  noHits.reactions = noHits.reactions.filter((x) => x.src !== 'hit' && x.src !== 'drums');
  noHits.bodies.forEach((b) => { if (b.motion.kind === 'hits') b.motion = { kind: 'none', p: {} }; });
  const withHits = cloneGenome(noHits);
  withHits.reactions = [{ src: 'hit', g: 'col', i: 0, k: 'exposure', gain: 0.3, atk: 0.005, rel: 0.2, thr: 0, q: 0, div: 1 }];
  check('accent.kick-auto', !hasHitResponse(noHits) && accentPlan(noHits).kick > 0 && accentPlan(withHits).kick === 0, `${accentPlan(noHits).kick} / ${accentPlan(withHits).kick}`);
  const tuned = { ...cloneGenome(noHits), accent: repairAccent({ p: { kick: 0.8 } }) };
  const tunedHits = { ...cloneGenome(withHits), accent: repairAccent({ p: { hook: 0.25 } }) };
  check('accent.kick-gene', accentPlan(tuned).kick === 0.8 && accentPlan(tunedHits).kick === 0 && accentPlan(tunedHits).hook === 0.25, 'a gene tunes the kick; a preset that answers hits gets none even when its gene carries the default');
  const kp = { ...accentPlan(noHits), hook: 0, section: 0, hue: 0, frame: 0, drop: 0 };
  const k0 = applyAccents(kp, input({ hit: 1.3 }), pose());
  const k1 = applyAccents(kp, input({ hit: 0 }), pose());
  check('accent.kick-subtle', k0.zoom > 1.01 && k0.zoom < 1.025 && k0.exposure > 1.05 && k0.exposure <= 1.08 && same(k1, IDENTITY_POSE as ChoreoPose), fmt(k0));

  // Breeding: parents without the gene give children without it; a child keeps valid accents otherwise.
  const rng = mulberry32(5);
  let kids = 0, bad = 0;
  for (let i = 0; i < 20; i++) {
    const d = SEEDS[i % SEEDS.length].genome;
    const e = SEEDS[(i * 7 + 3) % SEEDS.length].genome;
    const kid = crossover(d, e, rng);
    if (kid.accent && !d.accent && !e.accent) bad++;
    kids++;
  }
  check('accent.no-spontaneous', bad === 0, `${bad}/${kids} children of accent-less parents got one`);
  check('accent.cross-none', crossAccent(undefined, undefined, () => { throw new Error('drew'); }) === undefined, 'no rng draw without a parent gene');
  let gained = 0, invalid = 0;
  for (let i = 0; i < 200; i++) {
    const m = mutate(cloneGenome(base), mulberry32(100 + i));
    if (m.accent) {
      gained++;
      if (validateAccent(m.accent).length) invalid++;
    }
  }
  check('accent.mutation', invalid === 0, `${gained} of 200 mutants carry a tuned accent gene, ${invalid} invalid`);
}
