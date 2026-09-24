// On-device language model for the gene chat: Qwen3.5 4B (q4f16_1, MLC) run by a
// patched WebLLM 0.2.84 in a worker. Everything lives under public/llm/ as
// plain static files (the runtime bundle, the worker, the weight prefetcher, a
// catalog naming the tuned model lib); the weights stream from Hugging Face into
// the browser's OPFS store once and are reused from there.
//
// On an Apple GPU exposing WGSL subgroups behind Chromium 152+, the tuned model
// lib and decode knobs apply (device-resident greedy argmax, batched command
// encoding, prompt-lookup drafting forked for the recurrent layers). Anywhere
// else the same bundle runs with its stock decoding and WebLLM's stock model lib.
// Nothing is fetched until load() is called.

export const MODEL_ID = 'Qwen3.5-4B-q4f16_1-MLC';
/** Rough download size, for the confirmation prompt. */
export const MODEL_BYTES = 2.37e9;
/** Context the engine is created with: the system prompt plus a few turns of history. */
export const CONTEXT_TOKENS = 8192;
/**
 * Tokens per prefill pass. A pass occupies the GPU the visualizer shares, so its length is the
 * longest frame the visualizer can drop while a prompt is read.
 */
export const PREFILL_CHUNK = 256;
/** Pause between prefill passes (ms, -1 = none) so the visualizer's frames get the GPU in between. */
export const PREFILL_YIELD_MS = 0;

const base = () => new URL('llm/', document.baseURI).href;

export interface ChatMessage {
  role: 'system' | 'user' | 'assistant';
  content: string;
}
export interface Progress {
  /** 0..1 */
  progress: number;
  text: string;
}
export interface GenerateOptions {
  /** JSON schema the reply must follow (grammar-constrained decoding). */
  schema?: object;
  maxTokens?: number;
  onText?: (textSoFar: string) => void;
  signal?: AbortSignal;
}
export interface GenerateResult {
  text: string;
  promptTokens: number;
  completionTokens: number;
  /** Time to first token (prefill), ms. */
  prefillMs: number;
  /** Decode time after the first token, ms. */
  decodeMs: number;
  aborted: boolean;
}
export interface Plan {
  kind: 'tuned' | 'stock';
  why: string;
  label: string;
}

// Minimal shapes of the parts of WebLLM used here (the bundle is untyped JS).
interface WebLLMModule {
  prebuiltAppConfig: { model_list: ModelRecord[] };
  CreateWebWorkerMLCEngine(worker: Worker, model: string, cfg: object, chatOpts?: object): Promise<WebLLMEngine>;
}
interface ModelRecord {
  model: string;
  model_id: string;
  model_lib: string;
  overrides?: Record<string, unknown>;
  [k: string]: unknown;
}
interface Chunk {
  choices: { delta?: { content?: string } }[];
  usage?: { prompt_tokens: number; completion_tokens: number; extra?: Record<string, number> };
}
interface WebLLMEngine {
  chat: { completions: { create(req: object): Promise<AsyncIterable<Chunk> & object> } };
  interruptGenerate(): void;
  resetChat(): Promise<void>;
  unload(): Promise<void>;
}
interface Catalog {
  target: { gpu: { vendor: string; features: string[] }; browser: { minMajor: number } };
  models: Record<string, {
    modelLib: { url: string };
    default: string;
    variants: Record<string, { tuning: Record<string, boolean | number | string> }>;
  }>;
}

let modPromise: Promise<WebLLMModule> | null = null;
function webllm(): Promise<WebLLMModule> {
  modPromise ??= import(/* @vite-ignore */ `${base()}web-llm-0.2.84-qwen-m5-prompt-lookup.js`) as Promise<WebLLMModule>;
  return modPromise;
}

function chromeMajor(): number {
  const nav = navigator as Navigator & { userAgentData?: { brands?: { brand: string; version: string }[] } };
  for (const b of nav.userAgentData?.brands ?? []) {
    if (/Chromium|Google Chrome/i.test(b.brand)) return Number.parseInt(b.version, 10);
  }
  const m = /Chrom(?:e|ium)\/(\d+)/.exec(navigator.userAgent);
  return m ? Number(m[1]) : 0;
}

/** Why the chat cannot run here (null when it can): no WebGPU, no f16 shaders, a phone. */
export async function unsupportedReason(): Promise<string | null> {
  const gpu = (navigator as Navigator & { gpu?: { requestAdapter(): Promise<GPUAdapterLike | null> } }).gpu;
  if (!gpu) return 'This browser has no WebGPU (try desktop Chrome or Edge).';
  if (matchMedia('(pointer: coarse)').matches && !matchMedia('(pointer: fine)').matches) return 'The gene chat runs on desktop only.';
  let adapter: GPUAdapterLike | null = null;
  try {
    adapter = await gpu.requestAdapter();
  } catch {
    adapter = null;
  }
  if (!adapter) return 'No WebGPU adapter is available.';
  if (!adapter.features.has('shader-f16')) return 'This GPU lacks 16-bit shader support, which the model needs.';
  return null;
}
interface GPUAdapterLike {
  features: { has(f: string): boolean };
  info?: { vendor?: string };
}

async function plan(catalog: Catalog | null): Promise<{ plan: Plan; tuning: Record<string, boolean | number | string>; modelLib?: string }> {
  const stock = (why: string) => ({ plan: { kind: 'stock' as const, why, label: 'stock WebLLM' }, tuning: {} });
  if (new URLSearchParams(location.search).get('webllm') === 'stock') return stock('forced');
  const m = catalog?.models[MODEL_ID];
  if (!catalog || !m) return stock('no catalog');
  const gpu = (navigator as Navigator & { gpu?: { requestAdapter(): Promise<GPUAdapterLike | null> } }).gpu;
  const adapter = await gpu?.requestAdapter().catch(() => null);
  if (!adapter) return stock('no adapter');
  const vendor = String(adapter.info?.vendor ?? '').toLowerCase();
  if (vendor !== catalog.target.gpu.vendor) return stock(`gpu vendor ${vendor || 'unknown'}`);
  const missing = catalog.target.gpu.features.filter((f) => !adapter.features.has(f));
  if (missing.length) return stock(`missing ${missing.join(', ')}`);
  if (chromeMajor() < catalog.target.browser.minMajor) return stock(`chromium ${chromeMajor()} < ${catalog.target.browser.minMajor}`);
  const variant = new URLSearchParams(location.search).get('llmVariant') ?? m.default;
  const v = m.variants[variant] ?? m.variants[m.default];
  // Qwen3.5 has recurrent layers: never over-fill the decode queue (no bursts, no lookahead).
  const tuning: Record<string, boolean | number | string> = { ...v.tuning, greedyBurst: 1 };
  // ?llmTune=batchPass:0,flushEvery:16 overrides single knobs (for measuring).
  for (const kv of (new URLSearchParams(location.search).get('llmTune') ?? '').split(',').filter(Boolean)) {
    const [k, val] = kv.split(':');
    if (k === 'greedyBurst') continue;
    if (val === undefined || val === '') delete tuning[k];
    else tuning[k] = /^\d+$/.test(val) && k !== 'batchPass' && k !== 'greedyArgmax' && k !== 'bindGroupCache' ? Number(val) : val === '1';
  }
  return { plan: { kind: 'tuned', why: 'ok', label: `tuned ${variant}` }, tuning, modelLib: m.modelLib.url };
}

function workerUrl(tuning: Record<string, boolean | number | string>): URL {
  const url = new URL(`${base()}webllm-worker.js`);
  for (const [k, v] of Object.entries(tuning)) url.searchParams.set(k, typeof v === 'boolean' ? (v ? '1' : '0') : String(v));
  return url;
}

export class LocalLLM {
  private engine: WebLLMEngine | null = null;
  private worker: Worker | null = null;
  private loading: Promise<void> | null = null;
  plan: Plan | null = null;
  loadMs = 0;

  /** prefillChunk: tokens per prefill pass (smaller passes let visualizer frames through sooner). */
  private readonly prefillChunk: number;
  private readonly prefillYieldMs: number;

  constructor(prefillChunk = PREFILL_CHUNK, prefillYieldMs = PREFILL_YIELD_MS) {
    this.prefillChunk = prefillChunk;
    this.prefillYieldMs = prefillYieldMs;
  }

  get ready(): boolean {
    return !!this.engine;
  }

  /** Downloads (first time) and compiles the model. Safe to call twice. */
  load(onProgress: (p: Progress) => void): Promise<void> {
    this.loading ??= this.doLoad(onProgress).catch((err) => {
      this.loading = null;
      throw err;
    });
    return this.loading;
  }

  private async doLoad(onProgress: (p: Progress) => void): Promise<void> {
    const t0 = performance.now();
    onProgress({ progress: 0, text: 'Loading the runtime…' });
    const [mod, catalog] = await Promise.all([
      webllm(),
      fetch(`${base()}catalog.json`, { cache: 'no-cache' }).then((r) => (r.ok ? (r.json() as Promise<Catalog>) : null)).catch(() => null),
    ]);
    const p = await plan(catalog);
    const stockRec = mod.prebuiltAppConfig.model_list.find((r) => r.model_id === MODEL_ID);
    if (!stockRec) throw new Error(`${MODEL_ID} is missing from the runtime's model list`);
    const record: ModelRecord = { ...stockRec, ...(p.modelLib ? { model_lib: p.modelLib } : {}), overrides: { ...stockRec.overrides, context_window_size: CONTEXT_TOKENS, prefill_chunk_size: this.prefillChunk } };
    const opfs = typeof navigator.storage?.getDirectory === 'function';
    const appConfig = { model_list: [record], cacheBackend: opfs ? 'opfs' : 'cache' };
    // An 8-way largest-first prefetch into the same OPFS layout WebLLM reads; WebLLM
    // then finds every shard present. A failure is only logged (WebLLM fetches the rest).
    if (opfs) {
      try {
        const store = (await import(/* @vite-ignore */ `${base()}webllm-store.js`)) as {
          prefetchModel(r: ModelRecord, o: object): Promise<{ downloaded: number; bytes: number; ms: number }>;
        };
        const mb = (b: number) => (b / 2 ** 20).toFixed(0);
        const r = await store.prefetchModel(record, {
          concurrency: 8,
          onProgress: (x: { loaded: number; total: number }) =>
            onProgress({ progress: x.total ? (0.9 * x.loaded) / x.total : 0, text: `Downloading weights: ${mb(x.loaded)} / ${mb(x.total)} MB` }),
        });
        if (r.downloaded) console.info(`[chat] prefetched ${r.downloaded} shards, ${mb(r.bytes)} MB in ${(r.ms / 1000).toFixed(1)} s`);
      } catch (err) {
        console.warn('[chat] weight prefetch failed; WebLLM fetches the rest', err);
      }
    }
    const worker = new Worker(workerUrl({ ...p.tuning, prefillChunk: this.prefillChunk, ...(this.prefillYieldMs >= 0 ? { prefillYield: this.prefillYieldMs } : {}) }), { type: 'module', name: 'gene-chat-llm' });
    try {
      this.engine = await mod.CreateWebWorkerMLCEngine(worker, MODEL_ID, {
        appConfig,
        initProgressCallback: (x: { progress: number; text: string }) => onProgress({ progress: 0.9 + 0.1 * (x.progress ?? 0), text: x.text }),
      }, { context_window_size: CONTEXT_TOKENS, prefill_chunk_size: this.prefillChunk });
    } catch (err) {
      worker.terminate();
      throw err;
    }
    this.worker = worker;
    this.plan = p.plan;
    this.loadMs = performance.now() - t0;
    onProgress({ progress: 1, text: `Ready (${p.plan.label})` });
  }

  /**
   * One completion over `messages`. When the messages extend the previous call's conversation
   * (same system prompt and history plus one new user turn), WebLLM keeps its cache and only the
   * new turn is prefilled.
   */
  async generate(messages: ChatMessage[], opts: GenerateOptions = {}): Promise<GenerateResult> {
    const eng = this.engine;
    if (!eng) throw new Error('model not loaded');
    const req: Record<string, unknown> = {
      messages,
      stream: true,
      stream_options: { include_usage: true },
      temperature: 0,
      max_tokens: opts.maxTokens ?? 400,
      extra_body: { enable_thinking: false },
    };
    if (opts.schema) req.response_format = { type: 'json_object', schema: JSON.stringify(opts.schema) };
    const t0 = performance.now();
    let first = 0;
    let text = '';
    let usage: Chunk['usage'];
    let aborted = false;
    const onAbort = () => {
      aborted = true;
      eng.interruptGenerate();
    };
    opts.signal?.addEventListener('abort', onAbort);
    try {
      const stream = await eng.chat.completions.create(req);
      for await (const c of stream) {
        const d = c.choices[0]?.delta?.content;
        if (d) {
          if (!first) first = performance.now();
          text += d;
          opts.onText?.(text);
        }
        if (c.usage) usage = c.usage;
      }
    } finally {
      opts.signal?.removeEventListener('abort', onAbort);
    }
    const end = performance.now();
    return {
      text,
      promptTokens: usage?.prompt_tokens ?? 0,
      completionTokens: usage?.completion_tokens ?? 0,
      prefillMs: (first || end) - t0,
      decodeMs: first ? end - first : 0,
      aborted,
    };
  }

  async reset(): Promise<void> {
    await this.engine?.resetChat();
  }

  async dispose(): Promise<void> {
    try {
      await this.engine?.unload();
    } catch {
      /* the worker goes away regardless */
    }
    this.worker?.terminate();
    this.engine = null;
    this.worker = null;
    this.loading = null;
  }
}
