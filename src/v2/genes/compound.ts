// Compound shape: one body built from a few primitives blended into one distance field (a CN Tower
// from a tapered shaft, pods and an antenna; a crescent moon as a disc minus an offset disc; a
// village of hills, box houses and a steeple). A 'sdf' class shape, so every material, placement,
// motion, deformation and emission applies to the whole figure.
//
// The shape gene carries `size` (the figure's scale, scene units per part unit; reactable) and
// `parts` (1..MAX_PARTS). Each part is a small param record in the body's own frame, in units of
// `size`, so the parts move, react and turn with the body:
//   prim    0 ellipse (disc when sx = sy), 1 capsule (vertical; m tapers it: the top radius is
//           sx * (1 - m), so m = 1 is a cone, a flame or cypress; m < 0 flares), 2 box (half
//           extents sx, sy; m > 0 rounds the corners), 3 triangle (base 2 sx at y = -sy, apex at
//           (m * sx, +sy): m = 0 isosceles, +-1 a right-angled wedge)
//   op      how the part joins the figure so far: 0 union, 1 smooth union (radius k), 2 subtract,
//           3 intersect (both smooth by k when k > 0); the first part's op is ignored (always 0)
//   x, y    offset; sx, sy the size (non-uniform); rot its own static turn (turns)
//   hue     palette offset of the part (1/3 = the next palette slot, scaled by the colour mapping's
//           detail), bright a brightness weight (0 = a dark silhouette), so a warm moon can sit
//           next to a blue sky in one body
// Structure (each part's prim and op, the part count) is shader source; the numbers are uniforms in a
// per-body array (`uBx<i>`) declared only when a body has a compound or ring halos, so other genomes
// compile exactly as before.

import type { ParamSpec, Params, Schema } from '../genome';

type Rng = () => number;

const P = (min: number, max: number, def: number): ParamSpec => ({ min, max, def });
const C = (choices: number[], def: number): ParamSpec => ({ min: Math.min(...choices), max: Math.max(...choices), def, choices });

export const PRIMS = ['ellipse', 'capsule', 'box', 'triangle'] as const;
export type PrimName = (typeof PRIMS)[number];
export const COMBINE_OPS = ['union', 'smooth', 'subtract', 'intersect'] as const;
export type CombineName = (typeof COMBINE_OPS)[number];
export const MAX_PARTS = 10;

export const COMPOUND_SCHEMA: Schema = { size: P(0.02, 0.6, 0.2) };
export const PART_SCHEMA: Schema = {
  prim: C([0, 1, 2, 3], 0),
  op: C([0, 1, 2, 3], 0),
  x: P(-2, 2, 0),
  y: P(-2, 2, 0),
  sx: P(0.005, 2, 0.5),
  sy: P(0.005, 2, 0.5),
  rot: P(-0.5, 0.5, 0),
  k: P(0, 0.5, 0.1),
  m: P(-1, 1, 0),
  hue: P(0, 1, 0),
  bright: P(0, 2, 1),
};
/** Parameters of a compound part that are structure (shader source), not numbers. */
export const PART_STRUCT = ['prim', 'op'];

export type CompoundPart = Params;

/** Estimated ms at 1440p per evaluation: the figure's fixed part and each part (rotate, primitive, combine). */
export const COMPOUND_BASE_COST = 0.15;
export const COMPOUND_PART_COST = 0.1;
export function compoundCost(parts: CompoundPart[] | undefined): number {
  return COMPOUND_BASE_COST + COMPOUND_PART_COST * Math.max(1, parts?.length ?? 1);
}

const num = (v: unknown, d: number) => (typeof v === 'number' && Number.isFinite(v) ? v : d);
const clamp = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v));

function clampSpec(v: number, s: ParamSpec): number {
  if (s.choices) {
    let best = s.choices[0];
    for (const c of s.choices) if (Math.abs(c - v) < Math.abs(best - v)) best = c;
    return best;
  }
  const x = clamp(v, s.min, s.max);
  return s.int ? Math.round(x) : x;
}

export function defaultPart(): CompoundPart {
  const out: CompoundPart = {};
  for (const k of Object.keys(PART_SCHEMA)) out[k] = PART_SCHEMA[k].def;
  return out;
}

export function repairPart(raw: unknown): CompoundPart {
  const r = raw && typeof raw === 'object' ? (raw as Record<string, unknown>) : {};
  const out: CompoundPart = {};
  for (const k of Object.keys(PART_SCHEMA)) out[k] = clampSpec(num(r[k], PART_SCHEMA[k].def), PART_SCHEMA[k]);
  return out;
}

/** 1..MAX_PARTS valid parts; the first always joins as a plain union (it is the base). */
export function repairParts(raw: unknown): CompoundPart[] {
  const src = Array.isArray(raw) ? raw : [];
  const parts = src.slice(0, MAX_PARTS).map(repairPart);
  if (!parts.length) parts.push(defaultPart());
  parts[0].op = 0;
  return parts;
}

export function validateParts(parts: unknown, where: string): string[] {
  const errs: string[] = [];
  if (!Array.isArray(parts) || parts.length < 1 || parts.length > MAX_PARTS) return [`${where} parts count`];
  parts.forEach((p: Record<string, number>, j) => {
    if (!p || typeof p !== 'object') {
      errs.push(`${where}.parts[${j}]`);
      return;
    }
    for (const k of Object.keys(PART_SCHEMA)) {
      const s = PART_SCHEMA[k];
      const v = p[k];
      const ok = typeof v === 'number' && Number.isFinite(v) && (s.choices ? s.choices.includes(v) : v >= s.min && v <= s.max);
      if (!ok) errs.push(`${where}.parts[${j}].${k}=${v} out of range`);
    }
    for (const k of Object.keys(p)) if (!(k in PART_SCHEMA)) errs.push(`${where}.parts[${j}].${k} unknown`);
  });
  if (parts[0]?.op !== 0) errs.push(`${where}.parts[0].op not a union`);
  return errs;
}

/** Rough radius of the figure in part units (what the material's scale and the fit rules use). */
export function compoundExtent(parts: CompoundPart[] | undefined): number {
  let r = 0;
  for (const p of parts ?? []) {
    if (p.op === 2) continue;
    const reach = p.prim === 1 ? p.sy + p.sx * Math.max(1, 1 - p.m) : Math.hypot(p.sx, p.sy);
    r = Math.max(r, Math.hypot(p.x, p.y) + reach);
  }
  return Math.max(r, 0.05);
}

/** Uniform vec4s a body needs in its extra array: slot 0 the ring halo, then the compound's parts. */
export function bxCount(parts: CompoundPart[] | undefined, rings: number): number {
  if (parts?.length) return 1 + 2 * parts.length + Math.ceil(parts.length / 2);
  return rings > 0 ? 1 : 0;
}

/** Packs the parts into a body's extra array (after slot 0): A = (x, y, sx, sy), B = (cos, sin, k, m), then (hue, bright) pairs. */
export function packParts(parts: CompoundPart[], out: Float32Array): void {
  const n = parts.length;
  parts.forEach((p, i) => {
    const a = (1 + 2 * i) * 4;
    out[a] = p.x; out[a + 1] = p.y; out[a + 2] = p.sx; out[a + 3] = p.sy;
    const ang = p.rot * Math.PI * 2;
    out[a + 4] = Math.cos(ang); out[a + 5] = Math.sin(ang); out[a + 6] = p.k; out[a + 7] = p.m;
    const c = (1 + 2 * n + (i >> 1)) * 4 + (i & 1) * 2;
    out[c] = p.hue; out[c + 1] = p.bright;
  });
}

// ------------------------------------------------------------- shader

/** Primitive distance fields, emitted once per program when a body has a compound shape. */
export const COMPOUND_LIB = /* glsl */ `
float cpEll(vec2 p, vec2 r) {
  r = max(r, vec2(1e-4));
  float k0 = length(p / r), k1 = length(p / (r * r));
  return k1 > 1e-6 ? k0 * (k0 - 1.0) / k1 : -min(r.x, r.y);
}
float cpCap(vec2 p, vec2 s, float m) {
  float r1 = s.x, r2 = s.x * clamp(1.0 - m, 0.0, 2.0), h = 2.0 * max(s.y, 1e-4);
  p.x = abs(p.x);
  p.y += s.y;
  float b = clamp((r1 - r2) / h, -0.999, 0.999), a = sqrt(1.0 - b * b);
  float k = dot(p, vec2(-b, a));
  if (k < 0.0) return length(p) - r1;
  if (k > a * h) return length(p - vec2(0.0, h)) - r2;
  return dot(p, vec2(a, b)) - r1;
}
float cpBox(vec2 p, vec2 s, float m) {
  float r = clamp(m, 0.0, 1.0) * min(s.x, s.y);
  vec2 d = abs(p) - s + r;
  return length(max(d, 0.0)) + min(max(d.x, d.y), 0.0) - r;
}
float cpTri(vec2 p, vec2 s, float m) {
  vec2 p0 = vec2(-s.x, -s.y), p1 = vec2(s.x, -s.y), p2 = vec2(m * s.x, s.y);
  vec2 e0 = p1 - p0, e1 = p2 - p1, e2 = p0 - p2;
  vec2 v0 = p - p0, v1 = p - p1, v2 = p - p2;
  vec2 q0 = v0 - e0 * clamp(dot(v0, e0) / dot(e0, e0), 0.0, 1.0);
  vec2 q1 = v1 - e1 * clamp(dot(v1, e1) / dot(e1, e1), 0.0, 1.0);
  vec2 q2 = v2 - e2 * clamp(dot(v2, e2) / dot(e2, e2), 0.0, 1.0);
  float o = sign(e0.x * e2.y - e0.y * e2.x);
  vec2 d = min(min(vec2(dot(q0, q0), o * (v0.x * e0.y - v0.y * e0.x)), vec2(dot(q1, q1), o * (v1.x * e1.y - v1.y * e1.x))), vec2(dot(q2, q2), o * (v2.x * e2.y - v2.y * e2.x)));
  return -sqrt(d.x) * sign(d.y);
}
`;

const PRIM_CALL = ['cpEll(pp, A.zw)', 'cpCap(pp, A.zw, B.w)', 'cpBox(pp, A.zw, B.w)', 'cpTri(pp, A.zw, B.w)'];
const OP_CODE = [
  // union: the nearer part gives the colour
  '  if (di < d) { cs = C.x; br = C.y; }\n  d = min(d, di);\n',
  // smooth union
  '  kk = max(B.z, 1e-4); h = clamp(0.5 + 0.5 * (di - d) / kk, 0.0, 1.0); d = mix(di, d, h) - kk * h * (1.0 - h); cs = mix(C.x, cs, h); br = mix(C.y, br, h);\n',
  // subtract (smooth by k)
  '  kk = max(B.z, 1e-4); h = clamp(0.5 - 0.5 * (d + di) / kk, 0.0, 1.0); d = mix(d, -di, h) + kk * h * (1.0 - h);\n',
  // intersect (smooth by k)
  '  kk = max(B.z, 1e-4); h = clamp(0.5 - 0.5 * (di - d) / kk, 0.0, 1.0); d = mix(di, d, h) + kk * h * (1.0 - h);\n',
];

/**
 * `vec3 SHP(vec2 q)` for a compound: (distance, colour shade, brightness shade). SA.x is the size;
 * BX(k) the body's extra uniform slots (glsl.ts substitutes both).
 */
export function compoundShp(parts: CompoundPart[]): string {
  const n = parts.length;
  let s = `vec3 SHP(vec2 q) {
  float S = max(SA.x, 1e-3);
  vec2 u = q / S;
  float d = 1e3, cs = 0.0, br = 1.0, di, kk, h;
  vec4 A, B;
  vec2 C, pp;
`;
  parts.forEach((p, i) => {
    s += `  A = BX(${1 + 2 * i}); B = BX(${2 + 2 * i}); C = BX(${1 + 2 * n + (i >> 1)}).${i & 1 ? 'zw' : 'xy'};
  pp = mat2(B.x, -B.y, B.y, B.x) * (u - A.xy);
  di = ${PRIM_CALL[p.prim] ?? PRIM_CALL[0]};
`;
    s += i === 0 ? '  d = di; cs = C.x; br = C.y;\n' : OP_CODE[p.op] ?? OP_CODE[0];
  });
  return s + `  return vec3(d * S, cs, br);
}`;
}

// --------------------------------------------------- reference field (tests, tools)

function ell(px: number, py: number, rx: number, ry: number): number {
  rx = Math.max(rx, 1e-4);
  ry = Math.max(ry, 1e-4);
  const k0 = Math.hypot(px / rx, py / ry);
  const k1 = Math.hypot(px / (rx * rx), py / (ry * ry));
  return k1 > 1e-6 ? (k0 * (k0 - 1)) / k1 : -Math.min(rx, ry);
}
function cap(px: number, py: number, sx: number, sy: number, m: number): number {
  const r1 = sx, r2 = sx * clamp(1 - m, 0, 2), h = 2 * Math.max(sy, 1e-4);
  px = Math.abs(px);
  py += sy;
  const b = clamp((r1 - r2) / h, -0.999, 0.999), a = Math.sqrt(1 - b * b);
  const k = -b * px + a * py;
  if (k < 0) return Math.hypot(px, py) - r1;
  if (k > a * h) return Math.hypot(px, py - h) - r2;
  return a * px + b * py - r1;
}
function box(px: number, py: number, sx: number, sy: number, m: number): number {
  const r = clamp(m, 0, 1) * Math.min(sx, sy);
  const dx = Math.abs(px) - sx + r, dy = Math.abs(py) - sy + r;
  return Math.hypot(Math.max(dx, 0), Math.max(dy, 0)) + Math.min(Math.max(dx, dy), 0) - r;
}
function tri(px: number, py: number, sx: number, sy: number, m: number): number {
  const P0 = [-sx, -sy], P1 = [sx, -sy], P2 = [m * sx, sy];
  const V = [P0, P1, P2];
  const E = [[P1[0] - P0[0], P1[1] - P0[1]], [P2[0] - P1[0], P2[1] - P1[1]], [P0[0] - P2[0], P0[1] - P2[1]]];
  const o = Math.sign(E[0][0] * E[2][1] - E[0][1] * E[2][0]);
  let dd = Infinity, sg = Infinity;
  for (let i = 0; i < 3; i++) {
    const vx = px - V[i][0], vy = py - V[i][1];
    const [ex, ey] = E[i];
    const t = clamp((vx * ex + vy * ey) / (ex * ex + ey * ey), 0, 1);
    const qx = vx - ex * t, qy = vy - ey * t;
    dd = Math.min(dd, qx * qx + qy * qy);
    sg = Math.min(sg, o * (vx * ey - vy * ex));
  }
  return -Math.sqrt(dd) * Math.sign(sg);
}

/** The compound's field at (x, y) in the body's frame, as the shader computes it: distance, hue offset, brightness. */
export function compoundField(size: number, parts: CompoundPart[], x: number, y: number): { d: number; hue: number; bright: number } {
  const S = Math.max(size, 1e-3);
  const ux = x / S, uy = y / S;
  let d = 1e3, cs = 0, br = 1;
  parts.forEach((p, i) => {
    const a = p.rot * Math.PI * 2, c = Math.cos(a), s = Math.sin(a);
    const dx = ux - p.x, dy = uy - p.y;
    const px = c * dx + s * dy, py = -s * dx + c * dy;
    const di = p.prim === 1 ? cap(px, py, p.sx, p.sy, p.m) : p.prim === 2 ? box(px, py, p.sx, p.sy, p.m) : p.prim === 3 ? tri(px, py, p.sx, p.sy, p.m) : ell(px, py, p.sx, p.sy);
    if (i === 0) {
      d = di; cs = p.hue; br = p.bright;
      return;
    }
    const kk = Math.max(p.k, 1e-4);
    if (p.op === 0) {
      if (di < d) { cs = p.hue; br = p.bright; }
      d = Math.min(d, di);
    } else if (p.op === 1) {
      const h = clamp(0.5 + (0.5 * (di - d)) / kk, 0, 1);
      d = di + (d - di) * h - kk * h * (1 - h);
      cs = p.hue + (cs - p.hue) * h;
      br = p.bright + (br - p.bright) * h;
    } else if (p.op === 2) {
      const h = clamp(0.5 - (0.5 * (d + di)) / kk, 0, 1);
      d = d + (-di - d) * h + kk * h * (1 - h);
    } else {
      const h = clamp(0.5 - (0.5 * (di - d)) / kk, 0, 1);
      d = di + (d - di) * h + kk * h * (1 - h);
    }
  });
  return { d: d * S, hue: cs, bright: br };
}

// ----------------------------------------------------------- breeding

const pick = <T>(rng: Rng, xs: readonly T[]): T => xs[Math.floor(rng() * xs.length) % xs.length];
function gauss(rng: Rng): number {
  const u = Math.max(1e-9, rng());
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * rng());
}

/** A random part near the figure: mostly small unions, sometimes a smooth join or a cut. */
export function randomPart(rng: Rng, first = false): CompoundPart {
  const r = rng();
  return repairPart({
    prim: pick(rng, [0, 0, 1, 1, 2, 3]),
    op: first ? 0 : r < 0.45 ? 0 : r < 0.8 ? 1 : r < 0.95 ? 2 : 3,
    x: first ? 0 : (rng() - 0.5) * 1.2,
    y: first ? 0 : (rng() - 0.5) * 1.2,
    sx: 0.15 + 0.5 * rng(),
    sy: 0.15 + 0.5 * rng(),
    rot: rng() < 0.6 ? 0 : (rng() - 0.5) * 0.5,
    k: 0.05 + 0.2 * rng(),
    m: rng() < 0.5 ? 0 : (rng() - 0.5) * 1.6,
    hue: rng() < 0.75 ? 0 : pick(rng, [1 / 3, 2 / 3, 0.1]),
    bright: rng() < 0.8 ? 1 : 0.4 + 1.2 * rng(),
  });
}

/** Random genomes: few parts (1-3, mostly 2). */
export function randomParts(rng: Rng): CompoundPart[] {
  const r = rng();
  const n = r < 0.25 ? 1 : r < 0.8 ? 2 : 3;
  return repairParts(Array.from({ length: n }, (_x, i) => randomPart(rng, i === 0)));
}

/** Largest step a jitter moves a part's offset or size (part units, at amt 1). */
export const PART_STEP = 0.25;
/** Largest turn a jitter gives a part (turns, at amt 1). */
export const PART_TURN = 0.08;

/**
 * Small nudges of one or two parts: offsets and sizes a few hundredths of the figure, the static turn a
 * few degrees; the blend radius, taper, hue and brightness only now and then.
 */
export function jitterParts(parts: CompoundPart[], rng: Rng, amt = 1): void {
  const touch = rng() < 0.3 && parts.length > 1 ? 2 : 1;
  for (let t = 0; t < touch; t++) {
    const p = pick(rng, parts);
    const step = (sd: number, lim: number) => clamp(gauss(rng) * sd * amt, -lim * amt, lim * amt);
    if (rng() < 0.6) { p.x += step(0.06, PART_STEP); p.y += step(0.06, PART_STEP); }
    if (rng() < 0.5) { p.sx *= Math.exp(step(0.08, 0.2)); p.sy *= Math.exp(step(0.08, 0.2)); }
    if (rng() < 0.3) p.rot += step(0.02, PART_TURN);
    if (rng() < 0.15) p.k += step(0.03, 0.1);
    if (rng() < 0.15) p.m += step(0.1, 0.3);
    if (rng() < 0.1) p.hue = (p.hue + step(0.05, 0.15) + 1) % 1;
    if (rng() < 0.1) p.bright += step(0.1, 0.3);
    Object.assign(p, repairPart(p));
  }
  parts[0].op = 0;
}

/** Rare structural change: add, remove or reorder a part, or change a part's combine op or primitive. */
export function restructureParts(parts: CompoundPart[], rng: Rng): boolean {
  const r = rng();
  if (r < 0.3 && parts.length < MAX_PARTS) {
    const p = randomPart(rng);
    const near = pick(rng, parts);
    p.x = clamp(near.x + (rng() - 0.5) * near.sx * 2, -2, 2);
    p.y = clamp(near.y + (rng() - 0.5) * near.sy * 2, -2, 2);
    parts.push(p);
  } else if (r < 0.55 && parts.length > 1) {
    parts.splice(1 + Math.floor(rng() * (parts.length - 1)), 1);
  } else if (r < 0.7 && parts.length > 1) {
    const i = Math.floor(rng() * parts.length);
    let j = Math.floor(rng() * (parts.length - 1));
    if (j >= i) j++;
    [parts[i], parts[j]] = [parts[j], parts[i]];
  } else if (r < 0.9 && parts.length > 1) {
    const p = parts[1 + Math.floor(rng() * (parts.length - 1))];
    p.op = pick(rng, [0, 1, 2, 3].filter((o) => o !== p.op));
  } else {
    const p = pick(rng, parts);
    p.prim = pick(rng, [0, 1, 2, 3].filter((o) => o !== p.prim));
  }
  parts[0].op = 0;
  return true;
}

/**
 * Crossover of two compounds: parts paired by position; a pair of the same primitive and op may blend,
 * otherwise one parent's part is taken whole (parts swap); unpaired parts join sometimes.
 */
export function crossParts(a: CompoundPart[], b: CompoundPart[], rng: Rng): CompoundPart[] {
  const out: CompoundPart[] = [];
  for (let i = 0; i < Math.max(a.length, b.length); i++) {
    const x = a[i], y = b[i];
    if (x && y) {
      if (x.prim === y.prim && x.op === y.op && rng() < 0.5) {
        const t = 0.3 + 0.4 * rng();
        const m: CompoundPart = {};
        for (const k of Object.keys(PART_SCHEMA)) m[k] = PART_STRUCT.includes(k) ? x[k] : x[k] + (y[k] - x[k]) * t;
        out.push(m);
      } else out.push({ ...(rng() < 0.6 ? x : y) });
    } else if (x ? rng() < 0.75 : rng() < 0.35) out.push({ ...(x ?? y)! });
  }
  return repairParts(out);
}

/** Seed helper: a part by primitive and combine name, e.g. part('capsule', 'union', { y: 0.5, sx: 0.1, sy: 1, m: 0.6 }). */
export function part(prim: PrimName, op: CombineName, p: Partial<CompoundPart> = {}): CompoundPart {
  return repairPart({ ...defaultPart(), ...p, prim: PRIMS.indexOf(prim), op: COMBINE_OPS.indexOf(op) });
}

// ----------------------------------------------------------- ring halos

/**
 * Ring halos (a material option, every material): `rings` concentric rings outside a distance-field
 * shape, `rgap` apart (scene units, scaled with the copy), each `rfade` times as bright as the one inside
 * it; Van Gogh's ringed stars. rings 0 draws nothing and adds no shader code.
 */
export const RINGS_SCHEMA: Schema = { rings: { min: 0, max: 8, def: 0, int: true }, rgap: P(0.004, 0.15, 0.03), rfade: P(0.1, 1, 0.6) };
/** Extra ms per shape evaluation with rings on (one nearest ring, no loop). */
export const RINGS_COST = 0.03;
/** `vec3 RINGS(vec3 s, vec4 Q, vec2 p, float sc)`: light to add outside the shape (slot BX(0) = count, gap, fade). */
export const RINGS_GLSL = /* glsl */ `vec3 RINGS(vec3 s, vec4 Q, vec2 p, float sc) {
  vec4 R = BX(0);
  float gap = max(R.y * sc, 1e-4);
  float i = floor(s.x / gap + 0.5);
  if (s.x <= 0.0 || i < 1.0 || i > R.x) return vec3(0.0);
  float w = max(gap * 0.16, px() * 1.2);
  return COL(Q, s.y, 0.6, p) * s.z * glow(s.x - i * gap, w) * pow(R.z, i - 1.0) * (0.3 + 0.7 * Q.x) * 0.35;
}`;
