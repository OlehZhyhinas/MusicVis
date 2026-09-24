// Similarity judgements: "which of these two looks more like the reference?"
// Answers teach the phenotype distance how much each feature group matters.
// The distance is a weighted mean of per-group squared z differences (plus,
// later, an embedding term), so an answer (ref, a, b, picked a) says
// w . (t(ref,b) - t(ref,a)) > 0. The weights are fitted by a logistic
// ranking model with non-negative weights and a pull toward equal weights;
// agreement is reported cross-validated, so the number means something.
// Pure logic (no DOM) for the Node tests.

import { GROUPS, groupTerms, type FeatureNorm } from './fingerprint';

export interface SimilarityAnswer {
  ref: string;
  a: string;
  b: string;
  /** Which candidate looked more like the reference. */
  pick: 'a' | 'b';
  t: number;
  /** Fingerprints at answer time (members may be culled later). */
  fps: [number[], number[], number[]];
  /** Optional embedding term values [ref-a, ref-b] (perceptual embedding, when it was on). */
  emb?: [number, number];
}

export interface SimilarityData {
  format: 'musicvis-v2-similarity';
  answers: SimilarityAnswer[];
}

export const TERM_NAMES = [...GROUPS] as string[];

/** Term vector for one pair: per-group mean squared z difference. */
export function pairTerms(norm: FeatureNorm, x: number[], y: number[]): number[] {
  return groupTerms(norm.z(x), norm.z(y));
}

/** Rows for fitting: x = terms(ref, b) - terms(ref, a); y = 1 when a was picked. extra: appended terms (embedding). */
export function answerRows(answers: SimilarityAnswer[], norm: FeatureNorm, withEmb = false): { x: number[][]; y: number[] } {
  const x: number[][] = [], y: number[] = [];
  for (const an of answers) {
    if (withEmb && !an.emb) continue;
    const ta = pairTerms(norm, an.fps[0], an.fps[1]);
    const tb = pairTerms(norm, an.fps[0], an.fps[2]);
    const row = tb.map((v, i) => v - ta[i]);
    if (withEmb) row.push(an.emb![1] - an.emb![0]);
    x.push(row);
    y.push(an.pick === 'a' ? 1 : 0);
  }
  return { x, y };
}

const sig = (v: number) => 1 / (1 + Math.exp(-v));

/**
 * Logistic fit of non-negative weights: P(pick a) = sigmoid(w . x). An L2 pull
 * toward `prior` (equal weights by default) keeps a handful of answers from
 * swinging the metric. Projected gradient descent, deterministic.
 */
export function fitWeights(x: number[][], y: number[], prior?: number[], lambda = 1, iters = 400): number[] {
  const d = x[0]?.length ?? prior?.length ?? GROUPS.length;
  const p0 = prior ?? new Array(d).fill(1);
  const w = [...p0];
  if (!x.length) return w;
  const n = x.length;
  // Step 1/L, L bounding the curvature of the mean logistic loss plus the pull.
  let r2 = 0;
  for (const r of x) r2 = Math.max(r2, r.reduce((s, v) => s + v * v, 0));
  const lr = 1 / (0.25 * r2 + lambda / n + 1e-9);
  for (let it = 0; it < iters; it++) {
    const g = new Array(d).fill(0);
    for (let i = 0; i < n; i++) {
      let s = 0;
      for (let k = 0; k < d; k++) s += w[k] * x[i][k];
      const e = sig(s) - y[i];
      for (let k = 0; k < d; k++) g[k] += e * x[i][k];
    }
    for (let k = 0; k < d; k++) w[k] = Math.max(0, w[k] - lr * (g[k] / n + (lambda / n) * (w[k] - p0[k])));
  }
  return w;
}

/** Fraction of rows where the weighted distance agrees with the pick (ties count half). */
export function agreement(x: number[][], y: number[], w: number[]): number {
  if (!x.length) return NaN;
  let ok = 0;
  x.forEach((r, i) => {
    let s = 0;
    for (let k = 0; k < r.length; k++) s += w[k] * r[k];
    if (Math.abs(s) < 1e-12) ok += 0.5;
    else if ((s > 0 ? 1 : 0) === y[i]) ok++;
  });
  return ok / x.length;
}

/** Cross-validated agreement of the fit (leave-one-out up to 120 answers, else 10-fold). */
export function crossValidated(x: number[][], y: number[], prior?: number[], lambda = 1): number {
  const n = x.length;
  if (n < 4) return NaN;
  const folds = n <= 120 ? n : 10;
  let ok = 0;
  for (let f = 0; f < folds; f++) {
    const trX: number[][] = [], trY: number[] = [], teX: number[][] = [], teY: number[] = [];
    for (let i = 0; i < n; i++) {
      if (i % folds === f) {
        teX.push(x[i]);
        teY.push(y[i]);
      } else {
        trX.push(x[i]);
        trY.push(y[i]);
      }
    }
    const w = fitWeights(trX, trY, prior, lambda, 250);
    ok += agreement(teX, teY, w) * teX.length;
  }
  return ok / n;
}

export interface SimilarityFit {
  n: number;
  /** Fitted weights, normalised to mean 1 (GROUPS order, then the embedding when present). */
  weights: number[];
  /** Agreement of plain equal weights with the answers. */
  agreeEqual: number;
  /** Cross-validated agreement of the fitted weights. */
  agreeFit: number;
}

export function fitAnswers(answers: SimilarityAnswer[], norm: FeatureNorm, withEmb = false, embPrior = 1): SimilarityFit {
  const { x, y } = answerRows(answers, norm, withEmb);
  const d = GROUPS.length + (withEmb ? 1 : 0);
  const prior = new Array(d).fill(1);
  if (withEmb) prior[d - 1] = embPrior;
  const w = fitWeights(x, y, prior);
  const mean = w.reduce((s, v) => s + v, 0) / d || 1;
  return {
    n: x.length,
    weights: w.map((v) => v / mean),
    agreeEqual: agreement(x, y, prior),
    agreeFit: crossValidated(x, y, prior),
  };
}

// ------------------------------------------------------ triplet choice

export interface TripletCandidate {
  id: string;
  fp: number[];
}

export interface Triplet {
  ref: string;
  a: string;
  b: string;
  /** Why it was chosen: 0..1 uncertainty of the current metric and group disagreement. */
  uncertainty: number;
  disagreement: number;
}

/**
 * An informative triplet: a reference and two of its near neighbours for
 * which the current metric is least sure (the two distances nearly equal),
 * preferring pairs where the feature groups disagree about which is closer
 * (their answer moves the weights most). `extraDisagree(ref, a, b)` adds a
 * disagreement signal from elsewhere (hand features vs the embedding).
 */
export function chooseTriplet(
  cands: TripletCandidate[], norm: FeatureNorm, weights: number[], rng: () => number,
  opts: { refCounts?: Map<string, number>; seen?: Set<string>; neighbours?: number; extraDisagree?: (ref: string, a: string, b: string) => number } = {},
): Triplet | null {
  if (cands.length < 3) return null;
  const counts = opts.refCounts ?? new Map<string, number>();
  const zs = new Map(cands.map((c) => [c.id, norm.z(c.fp)]));
  // Reference: the least-used, randomly among ties.
  const byUse = [...cands].sort((p, q) => (counts.get(p.id) ?? 0) - (counts.get(q.id) ?? 0) || rng() - 0.5);
  const ref = byUse[Math.floor(rng() * Math.min(byUse.length, Math.max(3, Math.ceil(byUse.length / 4))))];
  const zr = zs.get(ref.id)!;
  const terms = new Map<string, number[]>();
  const dist = (t: number[]) => {
    let s = 0, ws = 0;
    t.forEach((v, i) => {
      s += weights[i] * v;
      ws += weights[i];
    });
    return ws > 0 ? s / ws : 0;
  };
  const others = cands.filter((c) => c.id !== ref.id).map((c) => {
    const t = groupTerms(zr, zs.get(c.id)!);
    terms.set(c.id, t);
    return { id: c.id, d: dist(t) };
  });
  others.sort((p, q) => p.d - q.d);
  const near = others.slice(0, opts.neighbours ?? 14);
  let best: Triplet | null = null;
  let bs = -Infinity;
  for (let i = 0; i < near.length; i++) for (let j = i + 1; j < near.length; j++) {
    const a = near[i], b = near[j];
    const key = [ref.id, ...[a.id, b.id].sort()].join('|');
    if (opts.seen?.has(key)) continue;
    const unc = 1 - Math.abs(a.d - b.d) / Math.max(1e-6, a.d + b.d);
    const ta = terms.get(a.id)!, tb = terms.get(b.id)!;
    const overall = Math.sign(b.d - a.d);
    let dis = 0, ws = 0;
    ta.forEach((v, g) => {
      ws += weights[g];
      if (Math.sign(tb[g] - v) !== overall) dis += weights[g];
    });
    dis = ws > 0 ? dis / ws : 0;
    const extra = opts.extraDisagree ? opts.extraDisagree(ref.id, a.id, b.id) : 0;
    const s = unc + 0.5 * dis + 0.5 * extra + 0.08 * rng();
    if (s > bs) {
      bs = s;
      // Random left / right order.
      best = rng() < 0.5 ? { ref: ref.id, a: a.id, b: b.id, uncertainty: unc, disagreement: dis } : { ref: ref.id, a: b.id, b: a.id, uncertainty: unc, disagreement: dis };
    }
  }
  return best;
}

export function parseAnswers(data: unknown): SimilarityAnswer[] {
  const d = data as Partial<SimilarityData> | undefined;
  if (!d || d.format !== 'musicvis-v2-similarity' || !Array.isArray(d.answers)) return [];
  return d.answers.filter(
    (a) => a && typeof a.ref === 'string' && typeof a.a === 'string' && typeof a.b === 'string' && (a.pick === 'a' || a.pick === 'b') &&
      Array.isArray(a.fps) && a.fps.length === 3 && a.fps.every((f) => Array.isArray(f) && f.length === a.fps[0].length),
  );
}
