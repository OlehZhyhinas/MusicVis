// Runs the gene chat test set (testset.ts) against the loaded model, the way the chat pane uses it:
// one conversation, each case a new preset. From the console: await __geneChatTest() or
// await __geneChatTest(['calmer', 'blue']). Resolves with per-case results and the pass rate.

import { cloneGenome, type Genome } from '../v2/genome';
import { SEEDS } from '../v2/seeds';
import { GeneChat } from './geneChat';
import type { LocalLLM } from './llm';
import type { LookMetrics } from './prompt';
import { TEST_KEY_HUE, TEST_SET } from './testset';

const MEDIUM: LookMetrics = { brightness: 0.2, coverage: 0.4, motion: 0.015, colourfulness: 0.5, hue: 0.55 };

export interface CaseResult {
  id: string;
  request: string;
  pass: boolean;
  why: string | null;
  say: string;
  changes: string[];
  problems: string[];
  repaired: boolean;
  ms: number;
  tokens: number;
  /** Model calls this case made: new prompt tokens, prefill ms, reply tokens, decode ms. */
  calls: { promptTokens: number; prefillMs: number; completionTokens: number; decodeMs: number }[];
}

export async function runChatTests(llm: LocalLLM, ids?: string[]): Promise<{ passed: number; total: number; rate: number; avgMs: number; results: CaseResult[] }> {
  let genome: Genome | null = null;
  let preset = '';
  let look: LookMetrics = MEDIUM;
  const chat = new GeneChat(llm, { genome: () => genome, presetId: () => preset, presetLabel: () => preset, keyHue: () => TEST_KEY_HUE, look: () => look });
  const results: CaseResult[] = [];
  for (const c of TEST_SET.filter((x) => !ids || ids.includes(x.id))) {
    const seed = SEEDS.find((s) => s.origin === c.seed);
    if (!seed) throw new Error(`no seed ${c.seed}`);
    genome = cloneGenome(seed.genome);
    preset = `${c.seed} "${seed.name}" (${c.id})`;
    look = c.look ?? MEDIUM;
    const t0 = performance.now();
    try {
      const r = await chat.send(c.request);
      const why = c.expect(r.before, r.after);
      results.push({ id: c.id, request: c.request, pass: !why, why, say: r.say, changes: r.changes, problems: r.problems, repaired: r.repaired, ms: Math.round(performance.now() - t0), tokens: r.stats.completionTokens, calls: chat.trace.map(({ promptTokens, prefillMs, completionTokens, decodeMs }) => ({ promptTokens, prefillMs, completionTokens, decodeMs })) });
    } catch (err) {
      results.push({ id: c.id, request: c.request, pass: false, why: `error: ${err instanceof Error ? err.message : String(err)}`, say: '', changes: [], problems: [], repaired: false, ms: Math.round(performance.now() - t0), tokens: 0, calls: [] });
    }
    const last = results[results.length - 1];
    (globalThis as unknown as { __chatTestLog?: CaseResult[] }).__chatTestLog = results;
    console.info(`[chat-test] ${last.pass ? 'PASS' : 'FAIL'} ${c.id}: ${last.why ?? ''} | ${last.say} | ${last.changes.join('; ')}${last.problems.length ? ` | problems: ${last.problems.join('; ')}` : ''} (${last.ms} ms)`);
  }
  const passed = results.filter((r) => r.pass).length;
  const calls = results.flatMap((r) => r.calls);
  const sum = (f: (c: CaseResult['calls'][number]) => number) => calls.reduce((s, c) => s + f(c), 0);
  const median = (xs: number[]) => (xs.length ? [...xs].sort((a, b) => a - b)[Math.floor(xs.length / 2)] : 0);
  const out = {
    passed, total: results.length, rate: results.length ? passed / results.length : 0,
    avgMs: Math.round(results.reduce((s, r) => s + r.ms, 0) / Math.max(1, results.length)),
    medianMs: median(results.map((r) => r.ms)),
    repaired: results.filter((r) => r.repaired).length,
    decodeTokS: Math.round((sum((c) => c.completionTokens) / Math.max(1, sum((c) => c.decodeMs) / 1000)) * 10) / 10,
    medianPromptTokens: median(calls.map((c) => c.promptTokens)),
    medianPrefillMs: median(calls.map((c) => c.prefillMs)),
    heldOut: (() => {
      const h = results.filter((r) => TEST_SET.find((c) => c.id === r.id)?.heldOut);
      return { passed: h.filter((r) => r.pass).length, total: h.length };
    })(),
    results,
  };
  console.info(`[chat-test] ${passed}/${results.length} passed (${Math.round(out.rate * 100)}%), ${out.avgMs} ms per request (median ${out.medianMs}), decode ${out.decodeTokS} tok/s, median prefill ${out.medianPrefillMs} ms for ${out.medianPromptTokens} new tokens, ${out.repaired} repair rounds; held out ${out.heldOut.passed}/${out.heldOut.total}`);
  return out;
}
