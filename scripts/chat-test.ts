// Tests for the gene chat's pure parts: edit paths, applying edits through the gene editor's
// functions (limits, repair, cost), the reply schema, the prompt text and reply parsing.
// Run: node --import ./scripts/analysis-test.hooks.mjs scripts/chat-test.ts

import { COST_BUDGET_MS, MAX_BODIES, MAX_REACTIONS, cloneGenome, estimateCost, validate, type Genome } from '../src/v2/genome';
import * as E from '../src/v2/geneEdit';
import { SEEDS } from '../src/v2/seeds';
import { applyEdits, paramPaths, parseKindPath, parsePath, replySchema, settablePaths, type Edit } from '../src/chat/edits';
import { systemPrompt, genomeDiff, genomeText, glossaryGaps, lookText } from '../src/chat/prompt';
import { parseReply, partialSay } from '../src/chat/geneChat';
import { registerGenomeGene, repairGenomeGenes, validateGenomeGenes } from '../src/v2/geneRegistry';

let failures = 0;
function check(name: string, ok: boolean, detail = ''): void {
  console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}${detail ? `: ${detail}` : ''}`);
  if (!ok) failures++;
}
const ctx = { keyHue: 0.1 };
const seeds = SEEDS.map((s) => s.genome);
const byShape = (k: string) => cloneGenome(seeds.find((g) => g.bodies.some((b) => b.shape.kind === k)) ?? seeds[0]);

// ------------------------------------------------------------ paths

{
  let bad = 0;
  let n = 0;
  for (const g of seeds) {
    for (const p of paramPaths(g)) {
      n++;
      const pp = parsePath(p.path);
      if (!pp || !E.schemaAt(g, pp.target)?.[pp.key] || E.getParam(g, pp.target, pp.key) !== p.value) bad++;
    }
  }
  check('every parameter path of every seed parses back to its value', bad === 0, `${n} paths, ${bad} bad`);
  const g = seeds[0];
  check('kind paths parse', !!parseKindPath('b0.shape') && !!parseKindPath('palette') && !!parseKindPath('op0') && !parseKindPath('b0.shape.r'));
  const sp = settablePaths(g);
  check('settable paths include other kinds\' params at each locus', sp.includes('b0.shape.inner') && sp.includes('b0.material.chrome'), `${sp.length} paths`);
}

// ------------------------------------------------------------ applying

{
  const g = byShape('dot');
  const before = JSON.stringify(g);
  const a = applyEdits(g, [{ op: 'set', path: 'tone.exposure', value: 1.3 }, { op: 'mul', path: 'carrier.halfLife', by: 2 }], ctx);
  check('set + mul apply', a.errors.length === 0 && a.genome.tone.p.exposure === 1.3 && Math.abs(a.genome.carrier.p.halfLife - g.carrier.p.halfLife * 2) < 1e-9, a.changes.join('; '));
  check('the input genome is untouched', JSON.stringify(g) === before);
  check('touched lists the paths', a.touched.includes('tone.exposure') && a.touched.includes('carrier.halfLife'));

  const c = applyEdits(g, [{ op: 'set', path: 'tone.exposure', value: 9 }], ctx);
  check('out-of-range values clamp with a note', c.genome.tone.p.exposure === 1.4 && c.errors.some((e) => e.startsWith('note:')));

  const d = applyEdits(g, [{ op: 'set', path: 'b0.shape.nope', value: 1 }, { op: 'set', path: 'zz.q', value: 1 }], ctx);
  check('unknown paths fail with a reason', d.errors.length === 2 && /no parameter "nope"/.test(d.errors[0]) && /no such path/.test(d.errors[1]), d.errors.join(' | '));

  const hue = applyEdits(g, [{ op: 'set', path: 'palette.hue', value: 'blue' }], ctx);
  check('colour names set absolute hues against the key', Math.abs(hue.genome.palette.p.hue - (0.62 - 0.1)) < 1e-6, String(hue.genome.palette.p.hue));

  const k = applyEdits(g, [{ op: 'kind', path: 'b0.material', kind: 'glow' }, { op: 'set', path: 'b0.material.width', value: 0.03 }], ctx);
  check('a kind switch then a param of the new kind', k.genome.bodies[0].material.kind === 'glow' && k.genome.bodies[0].material.p.width === 0.03, k.errors.join(' | '));
}

{
  let g = byShape('dot');
  while (g.bodies.length < MAX_BODIES) g = applyEdits(g, [{ op: 'add_body', shape: 'star' }], ctx).genome;
  const r = applyEdits(g, [{ op: 'add_body', shape: 'polygon' }], ctx);
  check('bodies are capped', r.genome.bodies.length === MAX_BODIES && /at most/.test(r.errors[0] ?? ''), r.errors[0]);
  const rm = applyEdits(g, [{ op: 'remove_body', body: 1 }], ctx);
  check('remove a body', rm.genome.bodies.length === MAX_BODIES - 1 && validate(rm.genome).length === 0);
  const one = byShape('dot');
  one.bodies = one.bodies.slice(0, 1);
  one.reactions = one.reactions.filter((x) => !['sh', 'pl', 'mo', 'de', 'ma', 'em', 'fe', 'cm', 'fu', 'fs', 'dr'].includes(x.g) || x.i === 0);
  const last = applyEdits(one, [{ op: 'remove_body', body: 0 }], ctx);
  check('the last body stays', last.genome.bodies.length === 1 && last.errors.length === 1);
}

{
  const g = byShape('dot');
  g.reactions = [];
  const r = applyEdits(g, [
    { op: 'add_reaction', signal: 'bass', path: 'tone.exposure', gain: 0.4 },
    { op: 'add_reaction', signal: 'drums', path: 'tone.exposure', gain: 0.4 },
  ], ctx);
  check('a parameter is never driven twice', r.genome.reactions.length === 1 && /already driven/.test(r.errors[0] ?? ''), r.errors[0]);
  let full = g;
  const targets = ['tone.exposure', 'tone.sat', 'tone.bloom', 'tone.contrast', 'tone.vignette', 'tone.ca', 'tone.adapt'];
  full = applyEdits(full, targets.map((path) => ({ op: 'add_reaction', signal: 'bass', path, gain: 0.3 }) as Edit), ctx).genome;
  check('reactions are capped', full.reactions.length === MAX_REACTIONS);
}

{
  const g = byShape('dot');
  const sparks = applyEdits(g, [{ op: 'kind', path: 'b0.emit', kind: 'sparks' }, { op: 'set', path: 'b0.emit.count', value: 65536 }, { op: 'set', path: 'b0.place.count', value: 6 }], ctx);
  const over = estimateCost(sparks.genome) > COST_BUDGET_MS;
  check('over-budget edits are reported', !over || sparks.errors.some((e) => /budget/.test(e)), `${estimateCost(sparks.genome).toFixed(1)} ms`);
}

{
  let ok = true;
  for (const g of seeds) {
    const edits: Edit[] = [{ op: 'kind', path: 'b0.shape', kind: 'flame' }, { op: 'add_op', kind: 'kaleido', stage: 'view' }, { op: 'fuse', body: 0, shape: 'star' }];
    const a = applyEdits(g, edits, ctx);
    if (validate(a.genome).length) ok = false;
  }
  check('structural edits on every seed leave valid genomes', ok);
}

// ------------------------------------------------------------ schema and prompt

{
  const g = seeds[0];
  const s = JSON.stringify(replySchema(g));
  check('reply schema is JSON with path enums', s.includes('"tone.exposure"') && s.includes('"add_body"'), `${s.length} chars`);
  const t = genomeText(g, 0.2);
  check('genome text lists every body and the palette', t.includes('b0.shape=') && t.includes('palette=') && t.includes('tone '), `${t.length} chars`);
  check('system prompt within budget', systemPrompt().length < 16000, `${systemPrompt().length} chars`);
  const gaps = glossaryGaps();
  check('every gene kind has a glossary entry', gaps.length === 0, gaps.join(', ') || 'none missing');
  check('look text words', /dark/.test(lookText({ brightness: 0.08, coverage: 0.1, motion: 0.01, colourfulness: 0.5, hue: 0.6 })));
  const g2 = cloneGenome(g);
  g2.tone.p.exposure = 1.2;
  const d = genomeDiff(g, g2, 0);
  check('diff after a manual edit', !!d && d.length === 1 && d[0].startsWith('tone.exposure'), d?.join(' | '));
  const g3 = applyEdits(g, [{ op: 'add_op', kind: 'swirl' }], ctx).genome;
  check('structural change sends the whole preset', genomeDiff(g, g3, 0) === null);
}

// ------------------------------------------------------------ replies

{
  const r = parseReply('<think>\n\n</think>\n\n{"say": "Bluer now.", "edits": [{"op":"set","path":"palette.hue","value":"blue"}]}');
  check('replies parse after a think block', !!r && r.say === 'Bluer now.' && r.edits.length === 1);
  check('broken replies are null', parseReply('{"say": "x", "edits": [') === null);
  check('partial say streams', partialSay('{"say": "Making it cal') === 'Making it cal');
}

// ------------------------------------------------------------ registered genome-wide genes

{
  registerGenomeGene({
    key: 'testWave', title: 'Test wave', kinds: ['sine', 'saw'], optional: true, glossary: 'a test gene',
    schemas: { sine: { amp: { min: 0, max: 1, def: 0.5 }, rate: { min: 0.1, max: 4, def: 1 } }, saw: { amp: { min: 0, max: 1, def: 0.5 }, teeth: { min: 2, max: 9, def: 4, int: true } } },
  });
  const g = seeds[0];
  const model = E.buildModel(g).find((x) => x.id === 'testWave');
  check('an absent optional gene shows as an addable section', !!model && model.gene?.present === false && model.gene.optional);
  const a = applyEdits(g, [{ op: 'add_gene', gene: 'testWave' }, { op: 'set', path: 'testWave.amp', value: 0.8 }, { op: 'kind', path: 'testWave', kind: 'saw' }, { op: 'set', path: 'testWave.teeth', value: 7 }], ctx);
  const v = E.geneValue(a.genome, 'testWave');
  check('chat adds, sets and switches a registered gene', !!v && v.kind === 'saw' && v.p.amp === 0.8 && v.p.teeth === 7, `${JSON.stringify(v)} ${a.errors.join(' | ')}`);
  const errs: string[] = [];
  validateGenomeGenes(a.genome as unknown as Record<string, unknown>, errs);
  check('the registered gene validates', errs.length === 0, errs.join(', '));
  check('a later structural edit keeps it', !!E.geneValue(applyEdits(a.genome, [{ op: 'add_op', kind: 'swirl' }], ctx).genome, 'testWave'));
  check('its paths are in the reply schema and the genome text', JSON.stringify(replySchema(a.genome)).includes('"testWave.teeth"') && genomeText(a.genome, 0).includes('testWave=saw'));
  const r = applyEdits(a.genome, [{ op: 'remove_gene', gene: 'testWave' }], ctx);
  check('remove it again', !E.geneValue(r.genome, 'testWave') && r.errors.length === 0);
  const repaired: Record<string, unknown> = {};
  repairGenomeGenes({ testWave: { kind: 'saw', p: { amp: 5, teeth: 3.4, junk: 1 } } }, repaired);
  check('repairGenomeGenes clamps and drops unknown params', JSON.stringify(repaired.testWave) === JSON.stringify({ kind: 'saw', p: { amp: 1, teeth: 3 } }), JSON.stringify(repaired.testWave));
}

void (null as unknown as Genome);
console.log(`\n${failures ? 'FAILED' : 'PASSED'}: ${failures} failing check(s)`);
process.exit(failures ? 1 : 0);
