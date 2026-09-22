// Main-thread entry: runs the analysis in a module worker.

import type { AnalysisRequest, AnalysisResponse, AnalysisResult } from '../types';

export function analyzeAudio(buf: AudioBuffer, onProgress?: (stage: string, p: number) => void): Promise<AnalysisResult> {
  return new Promise<AnalysisResult>((resolve, reject) => {
    const worker = new Worker(new URL('./analyzeWorker.ts', import.meta.url), { type: 'module' });
    const finish = () => worker.terminate();
    worker.onmessage = (ev: MessageEvent<AnalysisResponse>) => {
      const m = ev.data;
      if (m.type === 'progress') onProgress?.(m.stage, m.progress);
      else if (m.type === 'done') {
        finish();
        onProgress?.('Done', 1);
        resolve(m.result);
      } else if (m.type === 'error') {
        finish();
        reject(new Error(m.message));
      }
    };
    worker.onerror = (ev) => {
      finish();
      reject(new Error(ev.message || 'Analysis worker failed'));
    };
    // Copy channel data (AudioBuffer storage must not be detached), then transfer the copies.
    const left = new Float32Array(buf.getChannelData(0));
    const right = buf.numberOfChannels > 1 ? new Float32Array(buf.getChannelData(1)) : left;
    const req: AnalysisRequest = { type: 'analyze', sampleRate: buf.sampleRate, left, right };
    const transfer: Transferable[] = [left.buffer];
    if (right !== left) transfer.push(right.buffer);
    worker.postMessage(req, transfer);
  });
}
