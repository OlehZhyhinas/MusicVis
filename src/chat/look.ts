// What the visualizer looks like right now, for the gene chat ("too dark", "fill more of the
// screen"): brightness, coverage, motion and colourfulness of the live canvas, sampled a few times
// a second at 48x27 right after a frame is drawn and smoothed over a few seconds.

import type { LookMetrics } from './prompt';

const W = 48;
const H = 27;
const EVERY_S = 0.25;
const TAU_S = 2.5;

export class LookSampler {
  private ctx: CanvasRenderingContext2D | null = null;
  private clock = 0;
  private prev: Float32Array | null = null;
  private m: LookMetrics | null = null;
  private samples = 0;
  on = false;

  /** Call right after the frame is rendered (the WebGL drawing buffer is still valid then). */
  tick(dt: number, canvas: HTMLCanvasElement): void {
    if (!this.on) return;
    this.clock -= dt;
    if (this.clock > 0) return;
    this.clock = EVERY_S;
    try {
      if (!this.ctx) {
        const c = document.createElement('canvas');
        c.width = W;
        c.height = H;
        this.ctx = c.getContext('2d', { willReadFrequently: true });
        if (!this.ctx) return;
      }
      this.ctx.drawImage(canvas, 0, 0, W, H);
      this.add(this.ctx.getImageData(0, 0, W, H).data);
    } catch {
      /* a lost context or a tainted canvas: no measurement */
    }
  }

  private add(px: Uint8ClampedArray): void {
    const n = W * H;
    const luma = new Float32Array(n);
    let sum = 0, lit = 0, sat = 0, hx = 0, hy = 0, motion = 0;
    for (let i = 0; i < n; i++) {
      const r = px[i * 4] / 255, g = px[i * 4 + 1] / 255, b = px[i * 4 + 2] / 255;
      const l = 0.2126 * r + 0.7152 * g + 0.0722 * b;
      luma[i] = l;
      sum += l;
      if (this.prev) motion += Math.abs(l - this.prev[i]);
      if (l > 0.08) {
        lit++;
        const mx = Math.max(r, g, b), mn = Math.min(r, g, b);
        const s = mx > 0 ? (mx - mn) / mx : 0;
        sat += s;
        if (mx - mn > 0.02) {
          let h: number;
          if (mx === r) h = ((g - b) / (mx - mn) + 6) % 6;
          else if (mx === g) h = (b - r) / (mx - mn) + 2;
          else h = (r - g) / (mx - mn) + 4;
          const a = (h / 6) * Math.PI * 2;
          hx += Math.cos(a) * s * l;
          hy += Math.sin(a) * s * l;
        }
      }
    }
    const now: LookMetrics = {
      brightness: sum / n,
      coverage: lit / n,
      motion: this.prev ? motion / n : 0,
      colourfulness: lit ? sat / lit : 0,
      hue: Math.hypot(hx, hy) > 1e-3 ? (((Math.atan2(hy, hx) / (Math.PI * 2)) % 1) + 1) % 1 : -1,
    };
    this.prev = luma;
    this.samples++;
    if (!this.m || this.samples < 3) {
      this.m = now;
      return;
    }
    const k = 1 - Math.exp(-EVERY_S / TAU_S);
    const m = this.m;
    m.brightness += (now.brightness - m.brightness) * k;
    m.coverage += (now.coverage - m.coverage) * k;
    m.motion += (now.motion - m.motion) * k;
    m.colourfulness += (now.colourfulness - m.colourfulness) * k;
    m.hue = now.hue;
  }

  /** Smoothed metrics (null until a few samples are in). */
  get metrics(): LookMetrics | null {
    return this.samples >= 3 && this.m ? { ...this.m } : null;
  }
}
