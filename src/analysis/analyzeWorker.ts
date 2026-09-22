// Module Web Worker: AnalysisRequest -> progress messages + final AnalysisResult.

import type { AnalysisRequest, AnalysisResponse, AnalysisResult } from '../types';
import { STEM_NAMES } from '../types';
import { analyzePcm } from './analyzePcm';

const ctx = self as unknown as DedicatedWorkerGlobalScope;

function transferables(r: AnalysisResult): Transferable[] {
  const set = new Set<ArrayBuffer>();
  const add = (a: Float32Array) => {
    if (a.buffer instanceof ArrayBuffer) set.add(a.buffer);
  };
  for (const s of STEM_NAMES) {
    add(r.stems[s]);
    add(r.stemOnsets[s]);
  }
  add(r.loudness);
  add(r.chroma);
  add(r.beats);
  add(r.downbeats);
  return [...set];
}

ctx.onmessage = (ev: MessageEvent<AnalysisRequest>) => {
  const msg = ev.data;
  if (!msg || msg.type !== 'analyze') return;
  try {
    let lastPost = 0;
    const result = analyzePcm(msg.left, msg.right, msg.sampleRate, (stage, progress) => {
      const now = performance.now();
      if (progress < 1 && now - lastPost < 50) return;
      lastPost = now;
      const m: AnalysisResponse = { type: 'progress', stage, progress };
      ctx.postMessage(m);
    });
    const done: AnalysisResponse = { type: 'done', result };
    ctx.postMessage(done, transferables(result));
  } catch (err) {
    const m: AnalysisResponse = { type: 'error', message: err instanceof Error ? err.message : String(err) };
    ctx.postMessage(m);
  }
};
