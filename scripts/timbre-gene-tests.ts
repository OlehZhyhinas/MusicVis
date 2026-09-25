// Tests for the timbre gene (src/v2/genes/timbre.ts). Called from v2-test.ts.

import { COST_BUDGET_MS, SIGNALS, SPECIES, classify, cloneGenome, estimateCost, repair, structuralKey, validate, type Genome } from '../src/v2/genome';
import { crossover, mulberry32, mutate } from '../src/v2/ops';
import { SEEDS } from '../src/v2/seeds';
import { buildSources } from '../src/v2/glsl';
import { ADJ_POOLS, nameFor } from '../src/v2/naming';
import { genomeGene } from '../src/v2/geneRegistry';
import { TIMBRE_SCHEMA, repairTimbre, timbreLook, timbreTone, timbreWrap, validateTimbre, type TimbreGene } from '../src/v2/genes/timbre';
import type { TimbreStats } from '../src/types';

type Check = (name: string, ok: boolean, detail: string) => void;

const defaults = (): TimbreGene => repairTimbre({});
const withTimbre = (g: Genome, x: TimbreGene = defaults()): Genome => repair({ ...cloneGenome(g), timbre: x });
const T = (bright: number, noise: number, rough: number, attack: number): TimbreStats => ({ bright, noise, rough, attack });

export function timbreGeneTests(check: Check): void {
  // Repair / validate / registry / signals.
  {
    const d = defaults();
    check('timbre.defaults-valid', validateTimbre(d).length === 0 && Object.keys(d.p).length === Object.keys(TIMBRE_SCHEMA).length && !!genomeGene('timbre'), JSON.stringify(d.p));
    const broken = repairTimbre({ p: { src: 3.4, sheen: 4, scale: -1, glass: NaN, junk: 1 } });
    check('timbre.repair-clamps', validateTimbre(broken).length === 0 && broken.p.src === 3 && broken.p.sheen === 1 && broken.p.scale === 4 && !('junk' in broken.p), JSON.stringify(broken.p));
    const g = withTimbre(SEEDS[0].genome, broken);
    check('timbre.genome-roundtrip', validate(g).length === 0 && JSON.stringify(repair(g)) === JSON.stringify(g), 'repair is idempotent with a timbre gene');
    const bad = cloneGenome(g);
    bad.timbre!.p.velvet = 3;
    check('timbre.genome-validate', validate(bad).some((e) => e.startsWith('timbre')), validate(bad).join(','));
    check('timbre.seeds-untouched', SEEDS.every((s) => (s.origin.startsWith('T') ? !!s.genome.timbre : !('timbre' in s.genome))), 'only T seeds carry a timbre gene');
    check('timbre.signals', ['bright', 'noisy', 'rough', 'attack'].every((s) => (SIGNALS as readonly string[]).includes(s)), SIGNALS.join(','));
    check('timbre.structural', structuralKey(g) !== structuralKey(repair(cloneGenome(SEEDS[0].genome))), 'the surface is compiled only with the gene');
  }

  // The look: each quality of the sound drives its own surface.
  {
    const g = repairTimbre({ p: { sheen: 1, glass: 1, grain: 1, velvet: 1, edge: 1 } });
    const sine = timbreLook(g, T(0.3, 0, 0, 0.1));
    const lead = timbreLook(g, T(0.85, 0.05, 0.3, 0.6));
    const noise = timbreLook(g, T(0.8, 1, 0.7, 0.2));
    const breath = timbreLook(g, T(0.5, 0.6, 0.3, 0));
    check('timbre.look-glass', sine.glass > 0.9 && noise.glass < 0.05, `sine glass ${sine.glass.toFixed(2)}, noise ${noise.glass.toFixed(2)}`);
    check('timbre.look-sheen', lead.sheen > 0.7 && sine.sheen < 0.2, `bright lead sheen ${lead.sheen.toFixed(2)}, dull sine ${sine.sheen.toFixed(2)}`);
    check('timbre.look-grain', noise.grain > 0.9 && sine.grain < 0.05, `noise grain ${noise.grain.toFixed(2)}, sine ${sine.grain.toFixed(2)}`);
    check('timbre.look-velvet', breath.velvet > 0.7 && lead.velvet < 0.1 && sine.velvet === 0, `breath velvet ${breath.velvet.toFixed(2)}, lead ${lead.velvet.toFixed(2)}`);
    check('timbre.look-edge', lead.edge > sine.edge + 0.4, `pluck edge ${lead.edge.toFixed(2)} vs ${sine.edge.toFixed(2)}`);
    const e = repairTimbre({ p: { emboss: 1 } });
    const get = (k: string) => ({ relief: 0, gloss: 0.5, metal: 0, bump: 1 })[k] ?? 0;
    check('timbre.emboss', timbreTone(e, T(0.5, 0.9, 0.8, 0), get, 'relief') > 0.6 && timbreTone(defaults(), T(0.5, 0.9, 0.8, 0), get, 'relief') === 0, 'rough sound embosses only with emboss > 0');
  }

  // Shader: the surface wraps distance-field materials and leaves the rest alone.
  {
    const t01 = SEEDS.find((s) => s.origin === 'T01')!.genome;
    const src = buildSources(t01);
    check('timbre.glsl', src.composite.includes('timbreSurface') && src.composite.includes('MATB_0(s, q, Q, R, p, ex)') && src.composite.includes('uniform vec4 uTmb[2]'), 'T01 wraps MAT_0');
    const plain = buildSources(repair(cloneGenome(t01)));
    check('timbre.glsl-optional', buildSources(withTimbre(SEEDS[0].genome)).composite.includes('void main') && !buildSources(SEEDS[0].genome).composite.includes('timbreSurface') && plain.composite.includes('timbreSurface'), 'no surface without the gene');
    check('timbre.wrap-noop', timbreWrap('vec3 body_0(vec2 p, vec3 c) { return c; }', 0) === 'vec3 body_0(vec2 p, vec3 c) { return c; }', 'bodies without a material are unchanged');
  }

  // Cost: counted per distance-field evaluation, plus the emboss.
  {
    const worst = SEEDS.filter((s) => !s.genome.drift).map((s) => s.genome).sort((a, b) => estimateCost(b) - estimateCost(a))[0];
    const g = withTimbre(worst, repairTimbre({ p: { emboss: 1 } }));
    check('timbre.cost', estimateCost(g) > estimateCost(repair(cloneGenome(worst))) && estimateCost(g) <= COST_BUDGET_MS, `costliest seed with timbre ${estimateCost(g).toFixed(2)} ms`);
  }

  // Crossover across species, mutation, naming.
  {
    const rng = mulberry32(991);
    const bad: string[] = [];
    let none = 0, carried = 0, one = 0;
    const plain = SEEDS.filter((x) => !x.genome.timbre);
    const bySpecies = SPECIES.map((sp) => plain.filter((s) => classify(s.genome).primary === sp)).filter((l) => l.length);
    for (let i = 0; i < 600; i++) {
      const la = bySpecies[i % bySpecies.length];
      const lb = bySpecies[Math.floor(rng() * bySpecies.length)];
      const a = la[Math.floor(rng() * la.length)].genome;
      const b = lb[Math.floor(rng() * lb.length)].genome;
      const mode = i % 3;
      const pa = mode >= 1 ? withTimbre(a) : a;
      const pb = mode === 2 ? withTimbre(b, repairTimbre({ p: { sheen: 0.1, grain: 0.9 } })) : b;
      const child = crossover(pa, pb, rng, (rng() - 0.5) * 2);
      const errs = validate(child);
      if (errs.length) bad.push(`#${i}:${errs.join(';')}`);
      if (!(estimateCost(child) < COST_BUDGET_MS)) bad.push(`#${i}: cost ${estimateCost(child).toFixed(2)}`);
      if (child.timbre && i % 5 === 0 && !buildSources(child).composite.includes('void main')) bad.push(`#${i}: glsl`);
      if (mode === 0 && child.timbre) none++;
      if (mode === 1) {
        one++;
        if (child.timbre) carried++;
      }
      if (mode === 2 && !child.timbre) bad.push(`#${i}: both parents had timbre, child none`);
    }
    check('timbre.crossover-valid', !bad.length && none === 0, bad.slice(0, 4).join(' | ') || `600 children across ${bySpecies.length} species`);
    check('timbre.crossover-carry', carried > one * 0.3 && carried < one * 0.7, `${carried}/${one} single-parent children inherited it`);

    const mr = mulberry32(4040);
    let gained = 0, lost = 0;
    const mbad: string[] = [];
    for (let i = 0; i < 2000; i++) {
      const base = SEEDS[i % SEEDS.length].genome;
      const src = i % 2 && !base.timbre ? base : withTimbre(base);
      const g = mutate(src, mr, 0.3 + 2 * mr());
      const errs = validate(g);
      if (errs.length) mbad.push(`#${i}:${errs.join(';')}`);
      if (!src.timbre && g.timbre) gained++;
      if (src.timbre && !g.timbre) lost++;
    }
    check('timbre.mutate', !mbad.length && gained > 5 && gained < 200 && lost > 0, mbad.slice(0, 3).join(' | ') || `gained ${gained}/1000, lost ${lost}/1000`);

    const pools = [...(ADJ_POOLS.lustrous ?? []), ...(ADJ_POOLS.gritty ?? [])];
    const others = Object.entries(ADJ_POOLS).filter(([k]) => k !== 'lustrous' && k !== 'gritty').flatMap(([, v]) => v);
    check('timbre.naming', pools.length === 24 && pools.every((w) => !others.includes(w)) && new Set(pools).size === 24, `${nameFor(withTimbre(SEEDS[4].genome))} / ${nameFor(withTimbre(SEEDS[4].genome, repairTimbre({ p: { sheen: 0, glass: 0, grain: 1, velvet: 1 } })))}`);
  }
}
