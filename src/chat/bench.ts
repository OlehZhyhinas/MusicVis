// Runtime measurements for the gene chat (load, prefill, cache reuse, decode, visualizer fps
// while generating). Run from the console: await __geneChatBench().

import { LocalLLM, type ChatMessage } from './llm';
import { LOCUS_SCHEMAS, OP_SCHEMAS, CARRIER_SCHEMA, TONE_SCHEMA, PALETTE_SCHEMAS } from '../v2/genome';

function fpsMeter(): { stop: () => { fps: number; worstMs: number; long: string } } {
  let n = 0;
  let worst = 0;
  const long: string[] = [];
  let last = performance.now();
  const t0 = last;
  let on = true;
  const tick = () => {
    if (!on) return;
    const now = performance.now();
    worst = Math.max(worst, now - last);
    if (now - last > 50) long.push(`${Math.round(last - t0)}+${Math.round(now - last)}`);
    last = now;
    n++;
    requestAnimationFrame(tick);
  };
  requestAnimationFrame(tick);
  return {
    stop: () => {
      on = false;
      return { fps: (n * 1000) / (performance.now() - t0), worstMs: worst, long: long.join(' ') };
    },
  };
}

async function idleFps(ms: number): Promise<number> {
  const m = fpsMeter();
  await new Promise((r) => setTimeout(r, ms));
  return m.stop().fps;
}

/** A realistic ~3.5K-token system prompt: instructions plus every gene schema. */
function systemPrompt(): string {
  const schemas = JSON.stringify({ loci: LOCUS_SCHEMAS, ops: OP_SCHEMAS, carrier: CARRIER_SCHEMA, tone: TONE_SCHEMA, palette: PALETTE_SCHEMAS }, (_k, v) =>
    v && typeof v === 'object' && 'min' in v && 'max' in v ? `${v.min}..${v.max}` : v);
  return `You edit a music visualizer preset. Reply with JSON {"say": string, "edits": [...]}. Each edit is {"op":"set","path":string,"value":number}. Paths look like body0.shape.r or tone.exposure. Parameter ranges:\n${schemas}`;
}

const SCHEMA = {
  type: 'object',
  properties: {
    say: { type: 'string' },
    edits: {
      type: 'array',
      items: { type: 'object', properties: { op: { type: 'string', enum: ['set'] }, path: { type: 'string' }, value: { type: 'number' } }, required: ['op', 'path', 'value'] },
    },
  },
  required: ['say', 'edits'],
};

export async function runBench(): Promise<Record<string, unknown>> {
  const out: Record<string, unknown> = {};
  const llm = new LocalLLM();
  out.fpsIdle = await idleFps(3000);
  let lastText = '';
  const loadFps = fpsMeter();
  await llm.load((p) => {
    if (p.text !== lastText) console.info(`[bench] ${Math.round(p.progress * 100)}% ${p.text}`);
    lastText = p.text;
  });
  out.fpsDuringLoad = loadFps.stop().fps;
  out.plan = llm.plan;
  out.loadMs = Math.round(llm.loadMs);
  const sys = systemPrompt();
  out.systemChars = sys.length;

  // Warm-up (shader compile).
  await llm.generate([{ role: 'user', content: 'hi' }], { maxTokens: 2 });

  // 1. Cold turn: system prompt + user message, free text.
  const msgs: ChatMessage[] = [{ role: 'system', content: sys }, { role: 'user', content: 'Make it calmer. Reply in one short sentence.' }];
  let m = fpsMeter();
  let r = await llm.generate(msgs, { maxTokens: 60 });
  let f = m.stop();
  out.turn1 = { promptTokens: r.promptTokens, prefillMs: Math.round(r.prefillMs), prefillTokS: Math.round(r.promptTokens / (r.prefillMs / 1000)), completion: r.completionTokens, decodeTokS: +(r.completionTokens / (r.decodeMs / 1000)).toFixed(1), fps: +f.fps.toFixed(1), worstFrameMs: Math.round(f.worstMs) };

  // 2. Same conversation extended by one turn: WebLLM's multi-round path keeps the cache.
  msgs.push({ role: 'assistant', content: r.text }, { role: 'user', content: 'Now brighter. One short sentence.' });
  r = await llm.generate(msgs, { maxTokens: 60 });
  out.turn2Extended = { promptTokens: r.promptTokens, prefillMs: Math.round(r.prefillMs), completion: r.completionTokens, decodeTokS: +(r.completionTokens / (r.decodeMs / 1000)).toFixed(1) };

  // 3. Same system prompt, fresh history: WebLLM resets and prefills everything again.
  r = await llm.generate([{ role: 'system', content: sys }, { role: 'user', content: 'Make it busier. One short sentence.' }], { maxTokens: 60 });
  out.turn3Fresh = { promptTokens: r.promptTokens, prefillMs: Math.round(r.prefillMs) };

  // 4. JSON-schema constrained reply (grammar mask on every token).
  m = fpsMeter();
  r = await llm.generate([{ role: 'system', content: sys }, { role: 'user', content: 'Make the dots bigger and the picture darker.' }], { schema: SCHEMA, maxTokens: 200 });
  f = m.stop();
  out.turn4Json = { promptTokens: r.promptTokens, prefillMs: Math.round(r.prefillMs), completion: r.completionTokens, decodeTokS: +(r.completionTokens / (r.decodeMs / 1000)).toFixed(1), fps: +f.fps.toFixed(1), worstFrameMs: Math.round(f.worstMs), text: r.text };

  // 5. Long free decode for a clean tok/s and fps figure.
  m = fpsMeter();
  r = await llm.generate([{ role: 'user', content: 'Count from one to eighty in words, comma separated.' }], { maxTokens: 300 });
  f = m.stop();
  out.turn5Decode = { completion: r.completionTokens, decodeTokS: +(r.completionTokens / (r.decodeMs / 1000)).toFixed(1), fps: +f.fps.toFixed(1), worstFrameMs: Math.round(f.worstMs) };
  out.fpsIdleAfter = await idleFps(3000);
  (globalThis as unknown as { __geneChatLLM: LocalLLM }).__geneChatLLM = llm;
  console.info('[bench]', JSON.stringify(out, null, 1));
  return out;
}

/** Prefill speed and the longest visualizer frame for several prefill chunk sizes. */
export async function runChunkBench(sizes = [64, 128, 256, 512, 1024], yieldMs = -1): Promise<Record<string, unknown>[]> {
  const sys = systemPrompt();
  const rows: Record<string, unknown>[] = [];
  for (const chunk of sizes) {
    const llm = new LocalLLM(chunk, yieldMs);
    await llm.load(() => {});
    await llm.generate([{ role: 'user', content: 'hi' }], { maxTokens: 2 });
    const m = fpsMeter();
    const r = await llm.generate([{ role: 'system', content: sys }, { role: 'user', content: 'Make it calmer. One short sentence.' }], { maxTokens: 1 });
    const f = m.stop();
    rows.push({ chunk, yieldMs, promptTokens: r.promptTokens, prefillMs: Math.round(r.prefillMs), tokS: Math.round(r.promptTokens / (r.prefillMs / 1000)), fps: +f.fps.toFixed(1), worstFrameMs: Math.round(f.worstMs), long: f.long });
    console.info('[bench]', JSON.stringify(rows[rows.length - 1]));
    await llm.dispose();
  }
  return rows;
}
