// Per-frame visual features from an RGBA8 readback (rows bottom-up, as gl.readPixels gives
// them). Pure and allocation-free per frame after construction.
//
// All positions are in 0..1 with y pointing UP (0 bottom, 1 top), so "visual height" reads
// naturally. Flow is block matching on a 4x-downsampled luminance grid: 8x8 blocks on an
// 8-cell step, +-3 cells search (+-12 px at full res), SAD. Divergence and curl are the mean
// radial and tangential flow components around the frame centre (zoom-out / zoom-in and
// counter-clockwise / clockwise rotation), in grid cells per frame.

export const VIS_FIELDS = [
  'lum', // mean luminance 0..1
  'lumStd', // luminance standard deviation (contrast)
  'colorful', // Hasler-Susstrunk colourfulness, 0..~1
  'coverage', // share of pixels with luminance > 0.1
  'hueX', // saturation-weighted mean hue as a vector (cos, sin); its angle is the dominant hue
  'hueY',
  'sat', // mean saturation 0..1
  'cx', // luminance-weighted centroid x 0..1
  'cy', // luminance-weighted centroid y 0..1 (up)
  'spread', // luminance-weighted radial spread around the centroid, 0..~0.7
  'diff', // mean |dY| against the previous frame (0..1)
  'edge', // mean luminance gradient magnitude (detail)
  'flowMag', // mean block flow magnitude, grid cells per frame
  'flowX', // mean flow (translation), cells per frame, +x right
  'flowY', // +y up
  'div', // mean radial flow (+ outward / zooming in on the viewer)
  'curl', // mean tangential flow (+ counter-clockwise)
] as const;
export type VisField = (typeof VIS_FIELDS)[number];

export const THUMB_W = 32;
export const THUMB_H = 18;

export class VisualFeatures {
  readonly gw: number;
  readonly gh: number;
  private grid: Float32Array;
  private prev: Float32Array;
  private hasPrev = false;
  readonly out = new Float32Array(VIS_FIELDS.length);
  readonly thumb = new Uint8Array(THUMB_W * THUMB_H * 3);
  /** Last flow field (bw*bh*2), for debugging and the sheet. */
  readonly flow: Float32Array;
  readonly bw: number;
  readonly bh: number;

  readonly w: number;
  readonly h: number;

  constructor(w: number, h: number) {
    this.w = w;
    this.h = h;
    this.gw = Math.max(8, Math.floor(w / 4));
    this.gh = Math.max(8, Math.floor(h / 4));
    this.grid = new Float32Array(this.gw * this.gh);
    this.prev = new Float32Array(this.gw * this.gh);
    this.bw = Math.max(1, Math.floor((this.gw - 8) / 8) + 1);
    this.bh = Math.max(1, Math.floor((this.gh - 8) / 8) + 1);
    this.flow = new Float32Array(this.bw * this.bh * 2);
  }

  reset(): void {
    this.hasPrev = false;
  }

  /** Compute this frame's features from RGBA bottom-up pixels. */
  update(px: Uint8Array): Float32Array {
    const { w, h, gw, gh } = this;
    let sumY = 0, sumY2 = 0, cover = 0, sumRg = 0, sumRg2 = 0, sumYb = 0, sumYb2 = 0;
    let hx = 0, hy = 0, sumS = 0, wx = 0, wy = 0;
    const n = w * h;
    const grid = this.grid;
    grid.fill(0);
    const sx = gw / w;
    const sy = gh / h;
    for (let y = 0; y < h; y++) {
      const fy = (y + 0.5) / h; // bottom-up rows: y=0 is the bottom, already "up" oriented
      const gyi = Math.min(gh - 1, Math.floor(y * sy));
      let row = y * w * 4;
      for (let x = 0; x < w; x++, row += 4) {
        const r = px[row] / 255;
        const g = px[row + 1] / 255;
        const b = px[row + 2] / 255;
        const Y = 0.2126 * r + 0.7152 * g + 0.0722 * b;
        sumY += Y;
        sumY2 += Y * Y;
        if (Y > 0.1) cover++;
        const rg = r - g;
        const yb = 0.5 * (r + g) - b;
        sumRg += rg; sumRg2 += rg * rg; sumYb += yb; sumYb2 += yb * yb;
        const mx = Math.max(r, g, b);
        const mn = Math.min(r, g, b);
        const c = mx - mn;
        if (c > 1e-3) {
          let hue: number;
          if (mx === r) hue = ((g - b) / c + 6) % 6;
          else if (mx === g) hue = (b - r) / c + 2;
          else hue = (r - g) / c + 4;
          const a = (hue / 6) * 2 * Math.PI;
          const s = c / (mx + 1e-6);
          hx += Math.cos(a) * s * mx;
          hy += Math.sin(a) * s * mx;
          sumS += s;
        }
        wx += Y * ((x + 0.5) / w);
        wy += Y * fy;
        grid[gyi * gw + Math.min(gw - 1, Math.floor(x * sx))] += Y;
      }
    }
    const cellN = (w / gw) * (h / gh);
    for (let i = 0; i < grid.length; i++) grid[i] /= cellN;
    const o = this.out;
    const mY = sumY / n;
    o[0] = mY;
    o[1] = Math.sqrt(Math.max(0, sumY2 / n - mY * mY));
    const mRg = sumRg / n, mYb = sumYb / n;
    const sdRg = Math.sqrt(Math.max(0, sumRg2 / n - mRg * mRg));
    const sdYb = Math.sqrt(Math.max(0, sumYb2 / n - mYb * mYb));
    o[2] = Math.hypot(sdRg, sdYb) + 0.3 * Math.hypot(mRg, mYb);
    o[3] = cover / n;
    o[4] = hx / n;
    o[5] = hy / n;
    o[6] = sumS / n;
    const cx = sumY > 1e-6 ? wx / sumY : 0.5;
    const cy = sumY > 1e-6 ? wy / sumY : 0.5;
    o[7] = cx;
    o[8] = cy;
    // Spread and edges from the grid (cheap).
    let spr = 0, sw = 0, edge = 0, diff = 0;
    for (let y = 0; y < gh; y++) {
      for (let x = 0; x < gw; x++) {
        const v = grid[y * gw + x];
        const dx = (x + 0.5) / gw - cx;
        const dy = (y + 0.5) / gh - cy;
        spr += v * Math.hypot(dx, dy);
        sw += v;
        if (x + 1 < gw && y + 1 < gh) edge += Math.hypot(grid[y * gw + x + 1] - v, grid[(y + 1) * gw + x] - v);
        if (this.hasPrev) diff += Math.abs(v - this.prev[y * gw + x]);
      }
    }
    o[9] = sw > 1e-6 ? spr / sw : 0;
    o[10] = this.hasPrev ? diff / (gw * gh) : 0;
    o[11] = edge / ((gw - 1) * (gh - 1));
    if (this.hasPrev) this.blockFlow();
    else {
      o[12] = o[13] = o[14] = o[15] = o[16] = 0;
      this.flow.fill(0);
    }
    this.prev.set(grid);
    this.hasPrev = true;
    this.makeThumb(px);
    return o;
  }

  private blockFlow(): void {
    const { gw, gh, bw, bh } = this;
    const cur = this.grid;
    const prev = this.prev;
    const R = 3;
    let mag = 0, fx = 0, fy = 0, dv = 0, cu = 0, cnt = 0;
    for (let by = 0; by < bh; by++) {
      for (let bx = 0; bx < bw; bx++) {
        const x0 = bx * 8;
        const y0 = by * 8;
        // Skip flat blocks (no texture to track).
        let mean = 0;
        for (let y = 0; y < 8; y++) for (let x = 0; x < 8; x++) mean += cur[(y0 + y) * gw + x0 + x];
        mean /= 64;
        let tex = 0;
        for (let y = 0; y < 8; y++) for (let x = 0; x < 8; x++) tex += Math.abs(cur[(y0 + y) * gw + x0 + x] - mean);
        const k = (by * bw + bx) * 2;
        if (tex / 64 < 0.004) {
          this.flow[k] = this.flow[k + 1] = 0;
          continue;
        }
        let best = Infinity, bdx = 0, bdy = 0, zero = Infinity;
        for (let dy = -R; dy <= R; dy++) {
          for (let dx = -R; dx <= R; dx++) {
            // Content that moved by (dx, dy) came from prev at (x - dx, y - dy).
            let sad = 0;
            for (let y = 0; y < 8 && sad < best; y++) {
              const py = y0 + y - dy;
              const pyc = py < 0 ? 0 : py >= gh ? gh - 1 : py;
              for (let x = 0; x < 8; x++) {
                const pxx = x0 + x - dx;
                const pxc = pxx < 0 ? 0 : pxx >= gw ? gw - 1 : pxx;
                sad += Math.abs(cur[(y0 + y) * gw + x0 + x] - prev[pyc * gw + pxc]);
              }
            }
            if (dx === 0 && dy === 0) zero = sad;
            // Prefer small motion on ties.
            const pen = sad + 1e-4 * (dx * dx + dy * dy);
            if (pen < best) {
              best = pen;
              bdx = dx;
              bdy = dy;
            }
          }
        }
        if (zero <= best + 1e-6) bdx = bdy = 0;
        this.flow[k] = bdx;
        this.flow[k + 1] = bdy;
        const px = (x0 + 4) / gw - 0.5;
        const py = (y0 + 4) / gh - 0.5;
        const r = Math.hypot(px, py) || 1;
        mag += Math.hypot(bdx, bdy);
        fx += bdx;
        fy += bdy;
        dv += (bdx * px + bdy * py) / r;
        cu += (px * bdy - py * bdx) / r;
        cnt++;
      }
    }
    const o = this.out;
    const d = Math.max(1, bw * bh);
    o[12] = mag / d;
    o[13] = fx / d;
    o[14] = fy / d;
    o[15] = cnt ? dv / cnt : 0;
    o[16] = cnt ? cu / cnt : 0;
  }

  private makeThumb(px: Uint8Array): void {
    const { w, h } = this;
    const t = this.thumb;
    for (let ty = 0; ty < THUMB_H; ty++) {
      // Thumb rows are top-down.
      const y0 = Math.floor(((THUMB_H - 1 - ty) * h) / THUMB_H);
      const y1 = Math.max(y0 + 1, Math.floor(((THUMB_H - ty) * h) / THUMB_H));
      for (let tx = 0; tx < THUMB_W; tx++) {
        const x0 = Math.floor((tx * w) / THUMB_W);
        const x1 = Math.max(x0 + 1, Math.floor(((tx + 1) * w) / THUMB_W));
        let r = 0, g = 0, b = 0, c = 0;
        for (let y = y0; y < y1; y += 2) {
          for (let x = x0; x < x1; x += 2) {
            const i = (y * w + x) * 4;
            r += px[i]; g += px[i + 1]; b += px[i + 2]; c++;
          }
        }
        const k = (ty * THUMB_W + tx) * 3;
        t[k] = r / c; t[k + 1] = g / c; t[k + 2] = b / c;
      }
    }
  }
}
