// Fractal distance fields for the ray-marched scene gene (genes/raymarch.ts): a Mandelbox, a
// Mandelbulb and a Menger sponge, with low iteration counts (the iter gene) so they fit the frame
// budget at a reduced internal resolution. The same estimators run in JS once per frame so the camera
// can dive into the fractal without ever ending up inside it.
//
// uScn slots used by the fractal scenes (see raymarch.ts for the shared ones):
//   5  world scale, fold scale (Mandelbox), fold limit (Mandelbox; Menger offset), power (Mandelbulb)
//   6  iterations, variation seed, time, spectrum colouring
//   13-15  rotation rows of the fractal (turned by the section seed and slowly by time)

import type { Frame } from '../engine';
import type { Params } from '../genome';

/** Scene kind numbers of the fractals (after primitives, lattice, tunnel). */
export const FRACTAL_FIRST = 3;
export type FractalKind = 'mandelbox' | 'mandelbulb' | 'menger';
export const FRACTAL_KINDS: FractalKind[] = ['mandelbox', 'mandelbulb', 'menger'];

export const FRACTAL_GLSL = /* glsl */ `
vec3 rmFracRot(vec3 p) { return vec3(dot(uScn[13].xyz, p), dot(uScn[14].xyz, p), dot(uScn[15].xyz, p)); }
// Mandelbox: box fold, sphere fold, scale and translate; orbit trap = closest approach.
vec2 rmMandelbox(vec3 p) {
  vec4 A = uScn[5];
  vec3 c = rmFracRot(p) / A.x;
  vec3 z = c;
  float dr = 1.0, trap = 1e5;
  int n = int(uScn[6].x + 0.5);
  for (int i = 0; i < 12; i++) {
    if (i >= n) break;
    z = clamp(z, -A.z, A.z) * 2.0 - z;
    float r2 = dot(z, z);
    float m = r2 < 0.25 ? 4.0 : (r2 < 1.0 ? 1.0 / r2 : 1.0);
    z *= m;
    dr *= m;
    z = A.y * z + c;
    dr = dr * abs(A.y) + 1.0;
    trap = min(trap, r2);
  }
  return vec2(length(z) / abs(dr) * A.x, 0.1 + 0.8 * clamp(sqrt(trap) * 0.5, 0.0, 1.0));
}
// Mandelbulb: the power-n spherical-coordinates map; trap = smallest radius.
vec2 rmBulb(vec3 p) {
  vec4 A = uScn[5];
  vec3 c = rmFracRot(p) / A.x;
  vec3 z = c;
  float dr = 1.0, r = length(z), trap = 1e5;
  int n = int(uScn[6].x + 0.5);
  for (int i = 0; i < 12; i++) {
    if (i >= n || r > 2.0) break;
    float th = acos(clamp(z.z / r, -1.0, 1.0)) * A.w;
    float ph = atan(z.y, z.x) * A.w;
    dr = pow(r, A.w - 1.0) * A.w * dr + 1.0;
    z = pow(r, A.w) * vec3(sin(th) * cos(ph), sin(th) * sin(ph), cos(th)) + c;
    r = length(z);
    trap = min(trap, r);
  }
  return vec2(0.5 * log(max(r, 1e-6)) * r / dr * A.x, 0.1 + 0.8 * clamp(trap, 0.0, 1.0));
}
// Menger sponge: a cube with crosses cut out at every scale; the fold limit offsets the holes.
vec2 rmMenger(vec3 p) {
  vec4 A = uScn[5];
  vec3 c = rmFracRot(p) / A.x;
  vec3 q = abs(c) - 1.0;
  float d = length(max(q, 0.0)) + min(max(q.x, max(q.y, q.z)), 0.0);
  float s = 1.0, sh = 0.1;
  int n = min(int(uScn[6].x + 0.5), 5);
  for (int i = 0; i < 5; i++) {
    if (i >= n) break;
    vec3 a = mod(c * s + (A.z - 1.0) * 0.5, 2.0) - 1.0;
    s *= 3.0;
    vec3 r = abs(1.0 - 3.0 * abs(a));
    float cut = (min(max(r.x, r.y), min(max(r.y, r.z), max(r.z, r.x))) - 1.0) / s;
    if (cut > d) { d = cut; sh = 0.2 + 0.15 * float(i); }
  }
  return vec2(d * A.x, sh);
}
vec2 rmFractal(vec3 p) {
  float k = uScn[4].x;
  vec2 h = k > 4.5 ? rmMenger(p) : k > 3.5 ? rmBulb(p) : rmMandelbox(p);
  // Spectrum colouring: bands shift the trap colour.
  h.y += uScn[6].w * specAt(fract(h.y * 3.0) * 0.7 + 0.03) * 0.4;
  return h;
}
`;

/** One frame of fractal geometry, shared by the shader uniforms and the JS camera. */
export interface FractalFrame {
  kind: FractalKind;
  /** World scale: the fractal's natural radius maps to `radius`. */
  S: number;
  radius: number;
  scale: number;
  fold: number;
  power: number;
  iter: number;
  rot: number[];
}

/** Natural bounding radius of a fractal at these settings (before the world scale). */
function naturalRadius(kind: FractalKind, scale: number): number {
  if (kind === 'mandelbulb') return 1.25;
  if (kind === 'menger') return Math.sqrt(3);
  // The Mandelbox fills a cube of half-width 2 (s + 1) / (s - 1); its corners reach sqrt 3 further.
  const s = Math.abs(scale);
  return ((2 * (s + 1)) / (s - 1)) * Math.sqrt(3);
}

/** Mandelbox fold scale kept away from the degenerate |s| < 1.5. */
export function effectiveScale(s: number): number {
  const a = Math.min(3, Math.max(1.5, Math.abs(s)));
  return s < 0 ? -a : a;
}

export function fractalFrame(kind: FractalKind, P: (k: string) => number, raw: Params, size: number, bass: number, seed: number, T: number): FractalFrame {
  const scale = effectiveScale(P('fscale'));
  const radius = 2.2 * size;
  const S = radius / naturalRadius(kind, scale);
  // The bass breathes the fold, so the structure itself swells and folds with the low end.
  const fold = P('fold') * (1 + 0.06 * P('pulse') * bass) + 0.08 * seed;
  const rot = rotation(seed * Math.PI * 2 + T * 0.03, seed * 3 + T * 0.02);
  return { kind, S, radius, scale, fold, power: P('power'), iter: raw.iter, rot };
}

/** Rows of a rotation about z by a, then about x by b. */
function rotation(a: number, b: number): number[] {
  const ca = Math.cos(a), sa = Math.sin(a), cb = Math.cos(b), sb = Math.sin(b);
  return [ca, sa, 0, -sa * cb, ca * cb, sb, sa * sb, -ca * sb, cb];
}

/** The fractal's distance estimate at a world point (mirrors the GLSL). */
export function fractalDE(f: FractalFrame, x: number, y: number, z: number): number {
  const R = f.rot;
  const cx = (R[0] * x + R[1] * y + R[2] * z) / f.S;
  const cy = (R[3] * x + R[4] * y + R[5] * z) / f.S;
  const cz = (R[6] * x + R[7] * y + R[8] * z) / f.S;
  if (f.kind === 'mandelbox') {
    let zx = cx, zy = cy, zz = cz, dr = 1;
    const fl = f.fold, s = f.scale;
    for (let i = 0; i < f.iter; i++) {
      zx = Math.min(fl, Math.max(-fl, zx)) * 2 - zx;
      zy = Math.min(fl, Math.max(-fl, zy)) * 2 - zy;
      zz = Math.min(fl, Math.max(-fl, zz)) * 2 - zz;
      const r2 = zx * zx + zy * zy + zz * zz;
      const m = r2 < 0.25 ? 4 : r2 < 1 ? 1 / r2 : 1;
      zx = zx * m * s + cx; zy = zy * m * s + cy; zz = zz * m * s + cz;
      dr = dr * m * Math.abs(s) + 1;
    }
    return (Math.hypot(zx, zy, zz) / Math.abs(dr)) * f.S;
  }
  if (f.kind === 'mandelbulb') {
    let zx = cx, zy = cy, zz = cz, dr = 1, r = Math.hypot(zx, zy, zz);
    const n = f.power;
    for (let i = 0; i < f.iter && r <= 2; i++) {
      const th = Math.acos(Math.min(1, Math.max(-1, zz / Math.max(r, 1e-9)))) * n;
      const ph = Math.atan2(zy, zx) * n;
      dr = Math.pow(r, n - 1) * n * dr + 1;
      const zr = Math.pow(r, n);
      zx = zr * Math.sin(th) * Math.cos(ph) + cx;
      zy = zr * Math.sin(th) * Math.sin(ph) + cy;
      zz = zr * Math.cos(th) + cz;
      r = Math.hypot(zx, zy, zz);
    }
    return ((0.5 * Math.log(Math.max(r, 1e-6)) * r) / dr) * f.S;
  }
  // Menger sponge.
  const qx = Math.abs(cx) - 1, qy = Math.abs(cy) - 1, qz = Math.abs(cz) - 1;
  let d = Math.hypot(Math.max(qx, 0), Math.max(qy, 0), Math.max(qz, 0)) + Math.min(Math.max(qx, qy, qz), 0);
  let s = 1;
  const mod = (a: number, m: number) => a - m * Math.floor(a / m);
  for (let i = 0; i < Math.min(f.iter, 5); i++) {
    const off = (f.fold - 1) * 0.5;
    const ax = mod(cx * s + off, 2) - 1, ay = mod(cy * s + off, 2) - 1, az = mod(cz * s + off, 2) - 1;
    s *= 3;
    const rx = Math.abs(1 - 3 * Math.abs(ax)), ry = Math.abs(1 - 3 * Math.abs(ay)), rz = Math.abs(1 - 3 * Math.abs(az));
    const cut = (Math.min(Math.max(rx, ry), Math.max(ry, rz), Math.max(rz, rx)) - 1) / s;
    d = Math.max(d, cut);
  }
  return d * f.S;
}

/** Pushes a point out of the fractal until it is at least `clear` from the surface. */
export function keepClear(f: FractalFrame, p: [number, number, number], clear: number): [number, number, number] {
  let [x, y, z] = p;
  for (let k = 0; k < 24; k++) {
    const d = fractalDE(f, x, y, z);
    if (d >= clear) break;
    const e = 0.02 * f.radius;
    let nx = fractalDE(f, x + e, y, z) - fractalDE(f, x - e, y, z);
    let ny = fractalDE(f, x, y + e, z) - fractalDE(f, x, y - e, z);
    let nz = fractalDE(f, x, y, z + e) - fractalDE(f, x, y, z - e);
    let len = Math.hypot(nx, ny, nz);
    if (!(len > 1e-9)) {
      // Flat estimate (deep inside): head straight out from the centre.
      [nx, ny, nz] = [x, y, z];
      len = Math.hypot(x, y, z) || 1;
    }
    const step = Math.max(clear - d, 0.02 * f.radius);
    x += (nx / len) * step; y += (ny / len) * step; z += (nz / len) * step;
  }
  return [x, y, z];
}

export interface FractalCamCtx {
  F: Frame;
  sdt: number;
  P: (k: string) => number;
  cam: number;
  mem: Record<string, number>;
  key: string;
}

/**
 * Fractal cameras. orbit: circles outside the fractal; fly: the explorer, diving from outside deep
 * into the structure and back out over 16 bars along a slowly turning heading (roam = how deep),
 * steering around the surface; dolly: a vertigo zoom from outside. Drum hits lunge inward. Every
 * position is pushed clear of the surface with the JS estimator.
 */
export function fractalCamera(out: Float32Array, c: FractalCamCtx, f: FractalFrame, T: number, kick: number): void {
  const { F, sdt, P, mem } = c;
  const k = (s: string) => `${c.key}${s}`;
  const get = (s: string, init = 0) => mem[k(s)] ?? (mem[k(s)] = init);
  const set = (s: string, v: number) => (mem[k(s)] = v);
  const speed = P('speed'), roam = P('roam');
  const base = 0.6 + 0.4 * P('size');
  const ang = set('ang', (get('ang') + sdt * F.speed * (0.05 + 0.5 * speed)) % (Math.PI * 128));
  const R = f.radius;
  let pos: [number, number, number];
  let tgt: [number, number, number] = [0, 0, 0];
  let fl = 1.6;
  if (c.cam === 1) {
    const w = 0.5 - 0.5 * Math.cos((F.bars / 16) * Math.PI * 2);
    const r = R * 1.35 + 0.4 - (R * 1.35 + 0.4 - R * (0.95 - 0.85 * roam)) * w - 0.25 * R * kick;
    const el = 0.5 * Math.sin(T * 0.07);
    const dir: [number, number, number] = [Math.cos(ang * 0.5) * Math.cos(el), Math.sin(el), Math.sin(ang * 0.5) * Math.cos(el)];
    const want: [number, number, number] = [dir[0] * r, dir[1] * r, dir[2] * r];
    // Ease toward the path so pushes around the surface stay smooth.
    const e = 1 - Math.exp(-sdt * 4);
    const px = get('fx', want[0]), py = get('fy', want[1]), pz = get('fz', want[2]);
    pos = [px + (want[0] - px) * e, py + (want[1] - py) * e, pz + (want[2] - pz) * e];
    // Look inward, drifting a little sideways so the dive curves.
    tgt = [pos[0] * 0.2 - dir[2] * 0.3 * R, pos[1] * 0.2, pos[2] * 0.2 + dir[0] * 0.3 * R];
    fl = 1.3;
  } else if (c.cam === 2) {
    const w = 0.5 - 0.5 * Math.cos((F.bars / 4) * Math.PI * 2);
    const d = R + 1 + (1.2 + 4.5 * roam * w) * base;
    const dist = d - 0.8 * kick * base;
    const a = ang * 0.3;
    pos = [Math.cos(a) * dist, 0.5 * base, Math.sin(a) * dist];
    fl = (1.6 * d) / (R + 2.2 * base);
  } else {
    const el = 0.35 + 0.6 * roam * Math.sin(T * 0.11);
    const dist = R + 1.2 + 1.6 * base * (1 - 0.6 * kick);
    pos = [Math.cos(ang) * Math.cos(el) * dist, Math.sin(el) * dist, Math.sin(ang) * Math.cos(el) * dist];
  }
  pos = keepClear(f, pos, 0.06 * R);
  if (c.cam === 1) { set('fx', pos[0]); set('fy', pos[1]); set('fz', pos[2]); }
  out[0] = pos[0]; out[1] = pos[1]; out[2] = pos[2]; out[3] = fl;
  out[4] = tgt[0]; out[5] = tgt[1]; out[6] = tgt[2]; out[7] = 0.1 * kick * Math.sin(ang * 7);
}

/** Fractal uniforms: slots 5, 6 and the rotation rows 13-15. */
export function packFractal(out: Float32Array, f: FractalFrame, seed: number, T: number, spec: number): void {
  out[20] = f.S; out[21] = f.scale; out[22] = f.fold; out[23] = f.power;
  out[24] = f.iter; out[25] = seed; out[26] = T; out[27] = spec;
  for (let r = 0; r < 3; r++) for (let j = 0; j < 3; j++) out[(13 + r) * 4 + j] = f.rot[r * 3 + j];
}

/** Full-resolution march cost (model ms) of a fractal at an iteration count. */
export function fractalFullMs(kind: FractalKind, iter: number): number {
  if (kind === 'menger') return 4 + 1.6 * Math.min(iter, 5);
  if (kind === 'mandelbulb') return 4 + 3.2 * iter;
  return 4 + 2.2 * iter;
}
