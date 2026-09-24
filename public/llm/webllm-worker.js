// WebLLM runs here, off the main thread, so the multi-GB model load and token
// generation never stall the visualizer's frame loop. The runtime is the
// patched WebLLM 0.2.84 bundle next to this file; its decode knobs arrive as
// query parameters and become the globals the bundle reads.
const HOOKS = {
  greedyArgmax: "__webllmGreedyArgmax",
  greedyBurst: "__webllmGreedyBurst",
  batchPass: "__tvmjsWebGPUBatchPass",
  flushEvery: "__tvmjsWebGPUFlushEvery",
  bindGroupCache: "__tvmjsWebGPUBindGroupCache",
  promptLookup: "__webllmPromptLookup",
  // Local additions to the bundle: prefill pass size and a pause between passes (ms).
  prefillChunk: "__musicvisPrefillChunk",
  prefillYield: "__musicvisPrefillYieldMs",
};
const NUMERIC = new Set(["greedyBurst", "flushEvery", "prefillChunk", "prefillYield"]);
const BUNDLE = "./web-llm-0.2.84-qwen-m5-prompt-lookup.js";

const queue = [];
let forward = null;
let initError = null;

self.onmessage = (msg) => {
  if (forward) forward(msg);
  else if (initError) reject(msg);
  else queue.push(msg);
};

// "5:3:2:fork" -> { k: 5, nMax: 3, nMin: 2, hybrid: "fork" }
function promptLookup(spec) {
  if (!/^[1-9]\d*(?::[1-9]\d*){0,2}(?::fork)?$/.test(spec)) return undefined;
  const parts = spec.split(":");
  const fork = parts[parts.length - 1] === "fork";
  const [k, nMax = 3, nMin = 2] = (fork ? parts.slice(0, -1) : parts).map(Number);
  const out = { k, nMax, nMin: Math.min(nMin, nMax) };
  if (fork) out.hybrid = "fork";
  return out;
}

function applyTuning(params) {
  for (const [key, hook] of Object.entries(HOOKS)) {
    if (!params.has(key)) continue;
    const v = params.get(key);
    if (key === "promptLookup") globalThis[hook] = promptLookup(v);
    else if (NUMERIC.has(key)) { if (/^\d+$/.test(v)) globalThis[hook] = Number(v); }
    else if (v === "1" || v === "0") globalThis[hook] = v === "1";
  }
}

async function boot() {
  const params = new URLSearchParams(self.location.search);
  const mod = await import(BUNDLE);
  applyTuning(params);
  const handler = new mod.WebWorkerMLCEngineHandler();
  forward = (msg) => handler.onmessage(msg);
  for (const msg of queue.splice(0)) handler.onmessage(msg);
}

// WebLLM's worker protocol answers every request by uuid; answering a failed
// boot with {kind:"throw"} makes engine creation reject instead of hanging.
function reject(msg) {
  const uuid = msg?.data?.uuid;
  if (uuid) self.postMessage({ kind: "throw", uuid, content: `webllm-worker init failed: ${initError}` });
}

boot().catch((err) => {
  initError = String(err?.message || err);
  console.error("webllm-worker init failed:", err);
  for (const msg of queue.splice(0)) reject(msg);
});
