// GPU side of the phenotype fingerprint: renders a genome offscreen at 160x90
// on the reference clip, a few frames per animation frame through the
// screener's job runner (so the main view never stalls), and feeds sampled
// frames to FingerprintAcc. Also records a short strip of small frames for
// live previews (the similarity page).

import type { Engine, Stage } from './engine';
import { Stage as StageClass } from './engine';
import type { Genome } from './genome';
import type { JobRunner } from './screen';
import { CLIP, FingerprintAcc, ReferenceClip } from './fingerprint';

export const FP_W = 160;
export const FP_H = 90;
const SETTLE = 20; // frames before sampling starts (feedback and exposure settle)
const SAMPLE_EVERY = 4; // 15 Hz
const STILL_EVERY = 8; // full still analysis at 7.5 Hz
const PAIR_EVERY = 24; // a flow pair (this frame and the next) every 0.4 s

export interface FingerprintResult {
  fp: number[] | null;
  /** Wall-clock ms the job took (queued time excluded). */
  ms: number;
}

export class Fingerprinter {
  private stage: Stage;
  private px = new Uint8Array(FP_W * FP_H * 4);
  private clip = new ReferenceClip();
  private eng: Engine;
  private runner: JobRunner;
  /** Jobs run so far and their average cost (debug / HUD). */
  stats = { jobs: 0, avgMs: 0, lastMs: 0 };

  constructor(eng: Engine, runner: JobRunner) {
    this.eng = eng;
    this.runner = runner;
    this.stage = new StageClass(eng, { offscreen: true, particleCap: 16384, flameCap: 32768 });
    this.stage.resize(FP_W, FP_H);
  }

  fingerprint(g: Genome): Promise<FingerprintResult> {
    return new Promise((resolve) => {
      const st = this.stage;
      const acc = new FingerprintAcc(FP_W, FP_H);
      const total = Math.round(CLIP.end * CLIP.fps);
      let slot: ReturnType<Stage['makeSlot']> | null = null;
      let frame = 0;
      let waited = 0;
      let pairA: Uint8Array | null = null;
      let busyMs = 0;
      const done = (fp: number[] | null) => {
        if (slot) st.disposeSlot(slot);
        slot = null;
        this.stats.jobs++;
        this.stats.lastMs = busyMs;
        this.stats.avgMs += (busyMs - this.stats.avgMs) / Math.min(this.stats.jobs, 20);
        resolve({ fp, ms: busyMs });
      };
      this.runner.enqueue(() => {
        const t0 = performance.now();
        try {
          if (!slot) {
            const progs = this.eng.cache.get(g, waited > 240);
            if (!progs) {
              if (this.eng.cache.failed(g) || ++waited > 600) {
                done(null);
                return true;
              }
              return false;
            }
            slot = st.makeSlot(g, progs);
            st.slots = [slot];
            st.resetHistory();
            this.clip.reset();
            return false;
          }
          const state = this.clip.next();
          st.render(state, 1 / CLIP.fps, 'out');
          frame++;
          if (pairA) {
            st.readPixels(this.px);
            acc.pair(pairA, this.px);
            pairA = null;
          } else if (frame >= SETTLE && frame % SAMPLE_EVERY === 0) {
            st.readPixels(this.px);
            acc.sample(this.px, state.time, frame % STILL_EVERY === 0);
            if (frame % PAIR_EVERY === 0) pairA = this.px.slice();
          }
          if (frame >= total) {
            done(acc.samples > 8 ? acc.finish() : null);
            return true;
          }
          return false;
        } finally {
          busyMs += performance.now() - t0;
        }
      });
    });
  }
}
