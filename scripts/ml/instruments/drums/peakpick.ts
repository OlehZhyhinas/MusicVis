// Streaming causal peak picker, the TS twin of peakpick.py's peak_pick() (same decisions, frame by
// frame). Push one activation per hop; an onset for frame k is emitted when frame k+lookahead
// arrives. Fires at k when act[k] >= threshold, act[k] > max(previous `past` frames),
// act[k] >= max(next `lookahead` frames), act[k] >= mean(previous avgWin frames) + delta (if delta > 0),
// and at least refractory+1 frames have passed since the last onset.

export interface PeakParams { threshold: number; refractory: number; lookahead: number; past: number; delta: number; avgWin?: number }

export class StreamPeakPicker {
  private readonly buf: Float64Array; // ring of the last past+lookahead+avgWin+1 values
  private n = 0; // frames pushed
  private last = -1e9;
  private readonly W: number;
  private readonly p: PeakParams;
  constructor(p: PeakParams) {
    this.p = p;
    this.W = Math.max(p.past, p.avgWin ?? 8) + p.lookahead + 1;
    this.buf = new Float64Array(this.W);
  }
  private at(k: number): number { // value of frame k (-Inf if out of range)
    if (k < 0 || k >= this.n || k <= this.n - 1 - this.W) return -Infinity;
    return this.buf[k % this.W];
  }
  /** Push frame n's activation. Returns the frame index of an onset decided now, or -1. */
  push(a: number): number {
    this.buf[this.n % this.W] = a;
    this.n++;
    const k = this.n - 1 - this.p.lookahead;
    if (k < 0) return -1;
    const v = this.at(k);
    if (v < this.p.threshold) return -1;
    for (let j = 1; j <= this.p.past; j++) if (!(v > this.at(k - j))) return -1;
    for (let j = 1; j <= this.p.lookahead; j++) if (v < this.at(k + j)) return -1;
    if (this.p.delta > 0) {
      const w = this.p.avgWin ?? 8;
      const lo = Math.max(0, k - w);
      let s = 0;
      for (let j = lo; j < k; j++) s += this.at(j);
      if (v < s / Math.max(1, k - lo) + this.p.delta) return -1;
    }
    if (k - this.last <= this.p.refractory) return -1;
    this.last = k;
    return k;
  }
}
