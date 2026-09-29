import { Population } from '../src/v2/population';
import { Evolution } from '../src/v2/evolve';
import { mulberry32 } from '../src/v2/ops';
import { cloneGenome } from '../src/v2/genome';
import type { Screener } from '../src/v2/screen';
import type { Store } from '../src/v2/store';
import type { ExploreMode } from '../src/v2/novelty';

type Check = (name: string, ok: boolean, detail: string) => void;
export async function breedingPenTests(check: Check): Promise<void> {
  const pop = Population.seeded(1);
  const a = pop.get('G0-E13')!, b = pop.get('G0-X46')!, outsider = pop.get('G0-X48')!;
  const rng = mulberry32(813);
  check('pen.starts-empty', pop.breedingPool().length === 0 && pop.pickParents('calm', rng) === null,
    'an empty pen never falls back to the general population');
  pop.setBreeding([a.id], true);
  check('pen.single-parent', pop.pickParents('energetic', rng) === null,
    'one member cannot silently cross with an outsider');
  pop.setBreeding([b.id], true);
  outsider.likes = 10000;
  let confined = true;
  for (const mode of ['off', 'gentle', 'explore', 'wild'] as ExploreMode[]) {
    for (const niche of ['calm', 'energetic'] as const) {
      for (let i = 0; i < 40; i++) {
        const pair = pop.pickParents(niche, rng, 3, (m) => m === outsider ? 100 : 0, mode);
        confined &&= !!pair && pair[0] !== pair[1] && pair.every((m) => m === a || m === b);
      }
    }
  }
  check('pen.all-selection-paths', confined, 'niche, family, novelty and fitness fallbacks use only the two pen members');
  b.hidden = true;
  check('pen.hidden', pop.pickParents('calm', rng) === null && !pop.canBreed(b), 'hidden pen members cannot breed');
  b.hidden = false;
  const child = pop.addChild(cloneGenome(a.genome), [a, b], 2);
  check('pen.offspring', !child.breeding && pop.visible().includes(child) && pop.breedingPool().length === 2,
    'children remain playable in the general pool without inheriting membership');
  pop.setBreeding([child.id], true);
  pop.setBreeding([a.id], false);
  const restored = Population.fromJSON(JSON.parse(JSON.stringify(pop.toJSON())));
  restored.upgradeSeeds();
  check('pen.persistence', !restored.get(a.id)!.breeding && restored.get(b.id)!.breeding && restored.get(child.id)!.breeding,
    'both groups survive export/import and seed upgrades');
  const legacy = pop.toJSON();
  for (const m of legacy.members) delete (m as Partial<typeof m>).breeding;
  const loadedLegacy = Population.fromJSON(legacy);
  check('pen.legacy', loadedLegacy.size === pop.size && loadedLegacy.breedingPool().length === 0,
    'older populations keep all presets and start with an empty pen');
  restored.remove(b.id);
  check('pen.deleted', !restored.breedingPool().some((m) => m.id === b.id), 'deleted members leave the pen');

  let screens = 0;
  let duringScreen = () => {};
  const evo = new Evolution({} as Store, {
    screen: async () => {
      screens++;
      duringScreen();
      return { ok: true, descriptor: [], metrics: { reactivity: 0.8, hitLift: 0.4, events: 0.4 } };
    },
  } as unknown as Screener);
  evo.pop = Population.seeded(1);
  evo.rng = mulberry32(813);
  evo.changed = () => {};
  const p = evo.pop.get(a.id)!, q = evo.pop.get(b.id)!;
  const outside = await evo.breed([p, q], 1, 'cross');
  const noAuto = await evo.autoBreed('calm');
  const noMutation = await evo.breed([p], 1, 'mutate');
  check('pen.controller-guard', !outside.length && !noAuto.length && !noMutation.length && screens === 0,
    'manual, automatic and mutation requests cannot bypass the pen');
  evo.pop.setBreeding([p.id, q.id], true);
  const invalid = await evo.breed([p, p], 1, 'cross');
  const missing = await evo.breed([], 1, 'mutate');
  check('pen.invalid-parents', !invalid.length && !missing.length && screens === 0, 'invalid parent lists are rejected before screening');
  const born = await evo.breed([p, q], 1, 'cross');
  check('pen.manual-birth', born.length === 1 && !born[0].breeding && born[0].parents.join() === [p.id, q.id].join(),
    'screened offspring record only pen parents and enter the general pool');
  const automatic = await evo.autoBreed('energetic');
  check('pen.automatic-birth', automatic.length === 2 && automatic.every((m) => !m.breeding && m.parents.every((id) => id === p.id || id === q.id)),
    'both automatic rounds stay inside the pen, excluding their new offspring');
  evo.pop.setBreeding([q.id], false);
  const mutant = await evo.breed([p], 1, 'mutate');
  check('pen.mutation', mutant.length === 1 && !mutant[0].breeding && mutant[0].parents.join() === p.id,
    'one pen member can mutate and its child stays in the general pool');
  evo.pop.setBreeding([q.id], true);
  duringScreen = () => evo.pop.setBreeding([q.id], false);
  const interrupted = await evo.breed([p, q], 1, 'cross');
  check('pen.removed-during-screen', !interrupted.length && evo.breeding === 0,
    'removing a parent while screening stops the in-flight birth');
  evo.pop.setBreeding([q.id], true);
  duringScreen = () => { evo.pop = Population.fromJSON(evo.pop.toJSON()); };
  const replaced = await evo.breed([p, q], 1, 'cross');
  check('pen.replaced-population', !replaced.length && evo.breeding === 0,
    'importing or resetting during screening cannot add an old batch to the new population');
}
