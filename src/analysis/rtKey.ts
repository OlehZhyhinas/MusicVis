// Real-time key estimation: chroma accumulated over a rolling ~12 s window
// (exponential), Krumhansl-Schmuckler correlation for all 24 keys, and
// hysteresis so the key only changes when another one wins clearly and
// persistently.

import { keyCorrelations } from './key';

const WINDOW_TAU_S = 8; // exponential window, ~8-16 s of effective memory
const MIN_FILL_S = 5; // weighted seconds of chroma before the first estimate
const EVAL_S = 0.5;
const SWITCH_MARGIN = 0.04;
const SWITCH_EVALS = 5; // ~2.5 s of consistent evidence

export class KeyTracker {
  private readonly acc = new Float64Array(12);
  private readonly corr = new Float64Array(24);
  private fill = 0; // weighted seconds in the window
  private sinceEval = 0;
  private pending = -1;
  private pendingCount = 0;

  /** Current key index: tonic + 12 * minor, or -1 before the window has filled. */
  key = -1;
  confidence = 0;
  /** Increments on every key change (after the first estimate). */
  changes = 0;
  /** Stream time of the first estimate (NaN until then). */
  firstKeyTime = NaN;

  get tonic(): number {
    return this.key < 0 ? 0 : this.key % 12;
  }

  get mode(): 'major' | 'minor' {
    return this.key >= 12 ? 'minor' : 'major';
  }

  reset(): void {
    this.acc.fill(0);
    this.fill = 0;
    this.sinceEval = 0;
    this.pending = -1;
    this.pendingCount = 0;
    this.key = -1;
    this.confidence = 0;
    this.changes = 0;
    this.firstKeyTime = NaN;
  }

  /**
   * @param chroma 12 values, frame-normalized to max 1
   * @param weight 0..1 loudness weight of this chroma frame
   * @param dt     seconds covered by this chroma frame
   * @param t      stream time
   */
  push(chroma: ArrayLike<number>, weight: number, dt: number, t: number): boolean {
    const k = Math.exp(-dt / WINDOW_TAU_S);
    for (let i = 0; i < 12; i++) this.acc[i] = this.acc[i] * k + weight * chroma[i] * dt;
    this.fill = this.fill * k + weight * dt;
    this.sinceEval += dt;
    if (this.sinceEval < EVAL_S) return false;
    this.sinceEval = 0;
    // Enough signal in the window? (a full window of loud music fills to ~WINDOW_TAU_S)
    if (this.fill < Math.min(MIN_FILL_S, WINDOW_TAU_S * 0.6)) {
      return false;
    }
    keyCorrelations(this.acc, this.corr);
    let best = 0;
    for (let s = 1; s < 24; s++) if (this.corr[s] > this.corr[best]) best = s;
    if (this.key < 0) {
      this.key = best;
      this.confidence = Math.max(0, this.corr[best]);
      this.firstKeyTime = t;
      return true;
    }
    this.confidence += (Math.max(0, this.corr[this.key]) - this.confidence) * 0.3;
    if (best !== this.key && this.corr[best] - this.corr[this.key] > SWITCH_MARGIN) {
      if (best === this.pending) this.pendingCount++;
      else {
        this.pending = best;
        this.pendingCount = 1;
      }
      if (this.pendingCount >= SWITCH_EVALS) {
        this.key = best;
        this.pending = -1;
        this.pendingCount = 0;
        this.changes++;
        return true;
      }
    } else {
      this.pendingCount = 0;
      this.pending = -1;
    }
    return false;
  }
}
