// Beat-locked global jerk (CLI): how much the WHOLE picture jumps, zooms, shakes or turns on
// drum hits, measured from rendered frames. Reads clips rendered with --frames (render.ts).
//
//   node --import ./scripts/analysis-test.hooks.mjs scripts/avq/jerk.ts [--preset ID,...] [--song slug] [--label drop] [--json out.json]
//
// Per frame pair the picture's global similarity motion (translation, scale, rotation, plus a
// brightness gain / offset so flashes are not read as motion) is fitted to the whole frame by
// robust (Huber-weighted) Gauss-Newton, coarse to fine on 160x90 greyscale. A body moving on its
// own is a local residual the robust fit ignores; a camera or chain-level zoom / pan / roll moves
// every textured pixel and is what the fit reports.
//
// Units: motion is the displacement it causes at half the frame width from the centre, as a
// fraction of the frame width (so zoom and translation compare). jerk = |v(t) - v(t-1)|, the
// frame-to-frame change of that velocity, in % of the frame width per frame.
//
// Scores per clip (events = the engine's drum hits, F.hit rising edges from the .inst.json, and the
// song's beats; a jerk above SPIKE reads as a visible jump):
//   beat    beat-locked global jerk: jerk above SPIKE inside the windows [event - 1, event + 5]
//           beyond those windows' share of the frames, summed per second (drop moments excluded).
//   jerk    all jerk above SPIKE summed per second (drop moments excluded).
//   spk/s   frames per second with a jerk above SPIKE.
//   drop    the largest jerk within 8 frames of a drop moment (a snap; a swell stays low).
//   cam     the same on the engine's camera pose alone (pose.zoom / tx / ty / roll), so the camera's
//           share (choreo, accents) can be told apart from the chain's (zoom / swirl ops in the feedback).

import { execFileSync } from 'node:child_process';
import { existsSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { OUT } from './cdp';
import { listClips, loadInst } from './load';
import type { ClipHeader } from './format';
import { parseArgs } from './render';

const W = 160;
const H = 90;
/** A jerk above this (% of frame width per frame) reads as a visible jump. */
export const SPIKE = 0.5;
const POST = 5;

type Img = { w: number; h: number; d: Float32Array };

function down(a: Img): Img {
  const w = a.w >> 1;
  const h = a.h >> 1;
  const d = new Float32Array(w * h);
  for (let y = 0; y < h; y++)
    for (let x = 0; x < w; x++) {
      const i = 2 * y * a.w + 2 * x;
      d[y * w + x] = 0.25 * (a.d[i] + a.d[i + 1] + a.d[i + a.w] + a.d[i + a.w + 1]);
    }
  return { w, h, d };
}

function blur(a: Img): Img {
  const d = new Float32Array(a.d.length);
  const { w, h } = a;
  const t = new Float32Array(a.d.length);
  for (let y = 0; y < h; y++)
    for (let x = 0; x < w; x++) {
      const l = a.d[y * w + Math.max(0, x - 1)];
      const r = a.d[y * w + Math.min(w - 1, x + 1)];
      t[y * w + x] = 0.25 * l + 0.5 * a.d[y * w + x] + 0.25 * r;
    }
  for (let y = 0; y < h; y++)
    for (let x = 0; x < w; x++) {
      const u = t[Math.max(0, y - 1) * w + x];
      const b = t[Math.min(h - 1, y + 1) * w + x];
      d[y * w + x] = 0.25 * u + 0.5 * t[y * w + x] + 0.25 * b;
    }
  return { w, h, d };
}

function sample(a: Img, x: number, y: number): number {
  if (x < 0 || y < 0 || x > a.w - 1 || y > a.h - 1) return NaN;
  const x0 = Math.floor(x);
  const y0 = Math.floor(y);
  const x1 = Math.min(a.w - 1, x0 + 1);
  const y1 = Math.min(a.h - 1, y0 + 1);
  const fx = x - x0;
  const fy = y - y0;
  const r0 = a.d[y0 * a.w + x0] * (1 - fx) + a.d[y0 * a.w + x1] * fx;
  const r1 = a.d[y1 * a.w + x0] * (1 - fx) + a.d[y1 * a.w + x1] * fx;
  return r0 * (1 - fy) + r1 * fy;
}

/** Solves the n x n system A x = b in place (Gaussian elimination with partial pivoting). */
function solve(A: number[][], b: number[]): number[] | null {
  const n = b.length;
  for (let c = 0; c < n; c++) {
    let p = c;
    for (let r = c + 1; r < n; r++) if (Math.abs(A[r][c]) > Math.abs(A[p][c])) p = r;
    if (Math.abs(A[p][c]) < 1e-9) return null;
    [A[c], A[p]] = [A[p], A[c]];
    [b[c], b[p]] = [b[p], b[c]];
    for (let r = c + 1; r < n; r++) {
      const f = A[r][c] / A[c][c];
      for (let k = c; k < n; k++) A[r][k] -= f * A[c][k];
      b[r] -= f * b[c];
    }
  }
  const x = new Array(n).fill(0);
  for (let r = n - 1; r >= 0; r--) {
    let s = b[r];
    for (let k = r + 1; k < n; k++) s -= A[r][k] * x[k];
    x[r] = s / A[r][r];
  }
  return x;
}

/**
 * Global similarity motion from a to b: b(W(x)) ~ (1 + g) a(x) + c with
 * W(x) = ((1 + s) x - r y + tx, r x + (1 + s) y + ty) about the centre. Returns [tx, ty, s, r] with
 * tx, ty in pixels of the finest level.
 */
function fit(pa: Img[], pb: Img[]): number[] {
  let p = [0, 0, 0, 0, 0, 0]; // tx ty s r g c
  for (let L = pa.length - 1; L >= 0; L--) {
    const A = pa[L];
    const B = pb[L];
    const cx = (A.w - 1) / 2;
    const cy = (A.h - 1) / 2;
    for (let it = 0; it < 6; it++) {
      const res: number[] = [];
      const rows: { j: number[]; e: number }[] = [];
      for (let y = 1; y < A.h - 1; y++)
        for (let x = 1; x < A.w - 1; x++) {
          const X = x - cx;
          const Y = y - cy;
          const wx = (1 + p[2]) * X - p[3] * Y + p[0] + cx;
          const wy = p[3] * X + (1 + p[2]) * Y + p[1] + cy;
          const v = sample(B, wx, wy);
          if (Number.isNaN(v)) continue;
          const gx = 0.5 * (sample(B, wx + 1, wy) - sample(B, wx - 1, wy));
          const gy = 0.5 * (sample(B, wx, wy + 1) - sample(B, wx, wy - 1));
          if (Number.isNaN(gx) || Number.isNaN(gy)) continue;
          const a = A.d[y * A.w + x];
          const e = v - (1 + p[4]) * a - p[5];
          rows.push({ j: [gx, gy, gx * X + gy * Y, -gx * Y + gy * X, -a, -1], e });
          res.push(Math.abs(e));
        }
      if (rows.length < 50) break;
      res.sort((u, v) => u - v);
      const sc = Math.max(1e-3, 1.4826 * res[res.length >> 1]) * 1.5;
      const M = Array.from({ length: 6 }, () => new Array(6).fill(0));
      const bb = new Array(6).fill(0);
      for (const { j, e } of rows) {
        const ae = Math.abs(e);
        const w = ae <= sc ? 1 : sc / ae;
        for (let r = 0; r < 6; r++) {
          bb[r] -= w * j[r] * e;
          for (let c = 0; c < 6; c++) M[r][c] += w * j[r] * j[c];
        }
      }
      for (let r = 0; r < 6; r++) M[r][r] += 1e-6 * (1 + M[r][r]);
      const dp = solve(M, bb);
      if (!dp) break;
      for (let r = 0; r < 6; r++) p[r] += dp[r];
      if (Math.abs(dp[0]) + Math.abs(dp[1]) < 0.01 && Math.abs(dp[2]) + Math.abs(dp[3]) < 1e-4) break;
    }
    if (L > 0) p = [p[0] * 2, p[1] * 2, p[2], p[3], p[4], p[5]];
  }
  return p.slice(0, 4);
}

function pyramid(d: Float32Array): Img[] {
  const l0 = blur({ w: W, h: H, d });
  return [l0, blur(down(l0)), blur(down(down(l0)))];
}

/** Greyscale 160x90 frames of a clip (ffmpeg decodes the saved JPEGs). */
function frames(h: ClipHeader): Float32Array[] {
  const dir = join(OUT, h.framesDir!);
  const raw = execFileSync('ffmpeg', ['-v', 'error', '-i', join(dir, '%06d.jpg'), '-vf', `scale=${W}:${H}:flags=area`, '-f', 'rawvideo', '-pix_fmt', 'gray', '-'], { maxBuffer: 1 << 30 });
  const n = Math.floor(raw.length / (W * H));
  const out: Float32Array[] = [];
  for (let i = 0; i < n; i++) {
    const d = new Float32Array(W * H);
    for (let k = 0; k < W * H; k++) d[k] = raw[i * W * H + k] / 255;
    out.push(d);
  }
  return out;
}

export interface JerkResult {
  base: string;
  preset: string;
  frames: number;
  /** Rhythmic events in the clip (drum hits and beats). */
  events: number;
  /** Mean global motion speed (% W / frame): steady drift, fine on its own. */
  speed: number;
  /** Global jerk: the jerk above SPIKE summed per second, outside the drop moments. */
  jerk: number;
  /** Beat-locked global jerk: the part of `jerk` inside the event windows beyond their share of the frames. */
  beat: number;
  /** Frames per second whose jerk exceeds SPIKE, outside the drop moments. */
  spikes: number;
  /** Largest jerk within DROP_WIN frames of a drop moment. */
  drop: number;
  /** The same measured on the engine's camera pose alone (instrumentation). */
  camJerk: number;
  camBeat: number;
  camDrop: number;
}

const DROP_WIN = 8;
const mean = (a: number[]) => (a.length ? a.reduce((x, y) => x + y, 0) / a.length : 0);
const jerkOf = (v: number[][]) => v.map((x, i) => (i < 2 ? 0 : Math.hypot(...x.map((y, k) => y - v[i - 1][k]))));

/** Global jerk, its beat-locked part, spike rate and drop jerk of a per-frame jerk series. */
function score(J: number[], ev: number[], drops: number[], fps: number): { jerk: number; beat: number; spikes: number; drop: number } {
  const n = J.length;
  const inDrop = (i: number) => drops.some((d) => Math.abs(i - d) <= DROP_WIN);
  const win = new Uint8Array(n);
  for (const e of ev) for (let k = e - 1; k <= e + POST; k++) if (k >= 0 && k < n) win[k] = 1;
  let tot = 0;
  let inW = 0;
  let cnt = 0;
  let spk = 0;
  let cov = 0;
  for (let i = 2; i < n; i++) {
    if (inDrop(i)) continue;
    const ex = Math.max(0, J[i] - SPIKE);
    cnt++;
    tot += ex;
    if (J[i] > SPIKE) spk++;
    if (win[i]) {
      inW += ex;
      cov++;
    }
  }
  const sec = Math.max(1, cnt) / fps;
  const share = cnt ? cov / cnt : 0;
  let drop = 0;
  for (const d of drops) for (let k = d - DROP_WIN; k <= d + DROP_WIN; k++) drop = Math.max(drop, J[k] ?? 0);
  return { jerk: tot / sec, beat: Math.max(0, inW - share * tot) / sec, spikes: spk / sec, drop };
}

/** Per-frame global velocity [tx, ty, zoom, roll] (%W / frame) and jerk of a clip. */
export function analyseSeries(base: string): { h: ClipHeader; v: number[][]; J: number[] } | null {
  const h = JSON.parse(readFileSync(join(OUT, 'clips', base + '.json'), 'utf8')) as ClipHeader;
  if (!h.framesDir || !existsSync(join(OUT, h.framesDir))) return null;
  const fr = frames(h);
  const pyr = fr.map(pyramid);
  const v: number[][] = [[0, 0, 0, 0]];
  for (let i = 1; i < fr.length; i++) {
    const [tx, ty, s, r] = fit(pyr[i - 1], pyr[i]);
    // displacement at half the frame width, as % of the frame width
    v.push([(100 * tx) / W, (100 * ty) / W, 50 * s, 50 * r]);
  }
  return { h, v, J: jerkOf(v) };
}

export function analyse(base: string): JerkResult | null {
  const S = analyseSeries(base);
  if (!S) return null;
  const { h, v, J } = S;
  const inst = loadInst(base);
  const col = (name: string) => {
    const k = inst?.names.indexOf(name) ?? -1;
    return k < 0 || !inst ? null : inst.cols[k].map((x) => Number(x ?? 0));
  };
  const fps = h.render.fps;
  const hit = col('F.hit');
  const hitEv: number[] = [];
  if (hit) for (let i = 1; i < hit.length; i++) if (hit[i] > 0 && !(hit[i - 1] > 0)) hitEv.push(i);
  // events: the engine's drum hits and the song's beats (the frame whose music time first passes each)
  const beatEv = h.beats.filter((t) => t > h.clip.start && t < h.clip.end).map((t) => Math.ceil((t - h.clip.start) * fps) - 1);
  const ev = [...new Set([...hitEv, ...beatEv])].sort((x, y) => x - y).filter((x, i, a) => i === 0 || x - a[i - 1] > 2);
  const drops = h.moments.filter((m) => m.kind === 'drop' && m.t > h.clip.start && m.t < h.clip.end).map((m) => Math.ceil((m.t - h.clip.start) * fps) - 1);
  const img = score(J, ev, drops, fps);
  let cam = { jerk: 0, beat: 0, spikes: 0, drop: 0 };
  const pz = col('pose.zoom');
  if (pz) {
    // the camera pose in the same units: zoom factor z -> 50 ln z, pan (scene uv) -> 100 t, roll (rad) -> 50 roll
    const tx = col('pose.tx')!;
    const ty = col('pose.ty')!;
    const ro = col('pose.roll')!;
    const p = pz.map((z, i) => [100 * tx[i], 100 * ty[i], 50 * Math.log(Math.max(1e-3, z)), 50 * ro[i]]);
    cam = score(jerkOf(p.map((x, i) => (i ? x.map((y, k) => y - p[i - 1][k]) : [0, 0, 0, 0]))), ev, drops, fps);
  }
  return {
    base, preset: h.preset.id, frames: v.length, events: ev.length,
    speed: mean(v.map((x) => Math.hypot(...x))),
    jerk: img.jerk, beat: img.beat, spikes: img.spikes, drop: img.drop,
    camJerk: cam.jerk, camBeat: cam.beat, camDrop: cam.drop,
  };
}

async function main() {
  const a = parseArgs(process.argv.slice(2));
  const presets = a.preset ? String(a.preset).split(',') : null;
  const bases = listClips({ song: a.song ? String(a.song) : undefined, label: a.label ? String(a.label) : undefined }).filter((b) => !presets || presets.includes(b.split('/')[0]));
  const out: JerkResult[] = [];
  const f = (x: number) => x.toFixed(2).padStart(6);
  for (const b of bases) {
    const r = analyse(b);
    if (!r) continue;
    out.push(r);
    console.log(`${r.base.padEnd(48)} beat${f(r.beat)} jerk${f(r.jerk)} spk/s${f(r.spikes)} drop${f(r.drop)} | cam beat${f(r.camBeat)} jerk${f(r.camJerk)} drop${f(r.camDrop)} | speed${f(r.speed)}`);
  }
  if (a.json) writeFileSync(String(a.json), JSON.stringify(out, null, 1));
}

if (import.meta.url === `file://${process.argv[1]}`) await main();
