// Learned AV judge: a Bradley-Terry (logistic) model over audio-visual quality metrics, fit to
// clip-duel answers ("which feels more in sync?", "which do you like more?") and, as weaker
// evidence, to the population's behaviour (likes, watch time, quick skips).
//
// Pure maths so it runs in the Node tests; the duel page (duelUi.ts) stores answers and calls
// it. The metric vectors come from the harness report cards (scripts/avq), one per clip.

import type { ReportCard } from './avq/metrics';

/** Feature order of a metric vector (scripts/avq report card fields). */
export const JUDGE_FEATURES = [
  'sync', 'coupling', 'hookRhyme', 'melody', 'structure', 'flow', 'interest', 'correspond',
  'cfSync', 'stillness', 'chaos', 'activity', 'events',
] as const;
export type JudgeFeature = (typeof JUDGE_FEATURES)[number];

/** A report card (src/v2/avq/metrics.ts) as a judge feature vector (NaN where unknown). */
export function judgeVector(c: ReportCard): number[] {
  const h = c.headline as unknown as Record<string, number>;
  return JUDGE_FEATURES.map((f) => {
    const x = f === 'cfSync' ? c.counterfactual?.score : f === 'stillness' ? c.flow.stillness : f === 'chaos' ? c.counterfactual?.chaos : f === 'activity' ? c.interest.activity : h[f];
    return x !== undefined && Number.isFinite(x) ? Math.round(x * 1e4) / 1e4 : NaN;
  });
}

/** Answers needed before the judge's score is offered to the screener / fitness. */
export const MIN_DUELS_TO_APPLY = 30;

export interface Duel {
  /** Metric vectors of the two clips (JUDGE_FEATURES order; NaN = unknown). */
  va: number[];
  vb: number[];
  /** 1: a preferred, 0: b preferred. */
  y: number;
  /** Sample weight (1 for a duel answer, less for behaviour-derived pairs). */
  w?: number;
}

export interface JudgeModel {
  w: number[];
  mu: number[];
  sd: number[];
  n: number;
  /** k-fold cross-validated agreement on held-out answers (NaN below 8 answers). */
  cv: number;
  /** Agreement on the training answers. */
  fitAgree: number;
}

/** Column means / sds over every vector seen (NaN ignored); sd floor keeps constant columns harmless. */
export function standardizer(vs: number[][]): { mu: number[]; sd: number[] } {
  const d = JUDGE_FEATURES.length;
  const mu = new Array(d).fill(0);
  const sd = new Array(d).fill(1);
  for (let k = 0; k < d; k++) {
    const col = vs.map((v) => v[k]).filter(Number.isFinite);
    if (!col.length) continue;
    const m = col.reduce((a, b) => a + b, 0) / col.length;
    const s = Math.sqrt(col.reduce((a, b) => a + (b - m) ** 2, 0) / col.length);
    mu[k] = m;
    sd[k] = Math.max(s, 0.05);
  }
  return { mu, sd };
}

function z(v: number[], mu: number[], sd: number[]): number[] {
  return v.map((x, k) => (Number.isFinite(x) ? (x - mu[k]) / sd[k] : 0));
}

const sigmoid = (x: number) => 1 / (1 + Math.exp(-x));

/** Fit w by L2-regularised logistic regression on za - zb (no intercept: the model is symmetric). */
export function fitBT(duels: Duel[], opts: { l2?: number; iters?: number; mu?: number[]; sd?: number[] } = {}): JudgeModel {
  const d = JUDGE_FEATURES.length;
  const std = opts.mu && opts.sd ? { mu: opts.mu, sd: opts.sd } : standardizer(duels.flatMap((x) => [x.va, x.vb]));
  const X = duels.map((x) => {
    const a = z(x.va, std.mu, std.sd);
    const b = z(x.vb, std.mu, std.sd);
    return a.map((v, k) => v - b[k]);
  });
  const l2 = opts.l2 ?? 1;
  const w = new Array(d).fill(0);
  // Newton's method (d = 12): a few iterations converge.
  for (let it = 0; it < (opts.iters ?? 25); it++) {
    const g = new Array(d).fill(0);
    const H = Array.from({ length: d }, (_, i) => Array.from({ length: d }, (_, j) => (i === j ? l2 : 0)));
    for (let k = 0; k < d; k++) g[k] = l2 * w[k];
    X.forEach((x, i) => {
      const s = duels[i].w ?? 1;
      const p = sigmoid(x.reduce((a, v, k) => a + v * w[k], 0));
      const e = p - duels[i].y;
      for (let a = 0; a < d; a++) {
        g[a] += s * e * x[a];
        for (let b = 0; b < d; b++) H[a][b] += s * p * (1 - p) * x[a] * x[b];
      }
    });
    const step = solve(H, g);
    let moved = 0;
    for (let k = 0; k < d; k++) {
      w[k] -= step[k];
      moved += Math.abs(step[k]);
    }
    if (moved < 1e-6) break;
  }
  const model: JudgeModel = { w, mu: std.mu, sd: std.sd, n: duels.length, cv: NaN, fitAgree: agreement(duels, { w, mu: std.mu, sd: std.sd }) };
  return model;
}

function solve(A: number[][], b: number[]): number[] {
  const n = b.length;
  const M = A.map((r, i) => [...r, b[i]]);
  for (let p = 0; p < n; p++) {
    let piv = p;
    for (let q = p + 1; q < n; q++) if (Math.abs(M[q][p]) > Math.abs(M[piv][p])) piv = q;
    [M[p], M[piv]] = [M[piv], M[p]];
    const dd = M[p][p] || 1e-12;
    for (let q = p; q <= n; q++) M[p][q] /= dd;
    for (let r = 0; r < n; r++) {
      if (r === p) continue;
      const f = M[r][p];
      for (let q = p; q <= n; q++) M[r][q] -= f * M[p][q];
    }
  }
  return M.map((r) => r[n]);
}

/** Model score of one metric vector (higher = preferred); comparable across vectors. */
export function judgeScore(m: Pick<JudgeModel, 'w' | 'mu' | 'sd'>, v: number[]): number {
  return z(v, m.mu, m.sd).reduce((a, x, k) => a + x * m.w[k], 0);
}

/** P(a preferred over b). */
export function judgeProb(m: Pick<JudgeModel, 'w' | 'mu' | 'sd'>, va: number[], vb: number[]): number {
  return sigmoid(judgeScore(m, va) - judgeScore(m, vb));
}

export function agreement(duels: Duel[], m: Pick<JudgeModel, 'w' | 'mu' | 'sd'>): number {
  const hard = duels.filter((x) => (x.w ?? 1) >= 1);
  if (!hard.length) return NaN;
  let ok = 0;
  for (const x of hard) if ((judgeProb(m, x.va, x.vb) > 0.5 ? 1 : 0) === x.y) ok++;
  return ok / hard.length;
}

/** k-fold cross-validated agreement on the duel answers (behaviour pairs only ever train). */
export function crossValidate(duels: Duel[], k = 5, opts: { l2?: number } = {}): number {
  const hard = duels.map((x, i) => ({ x, i })).filter((e) => (e.x.w ?? 1) >= 1);
  if (hard.length < 8) return NaN;
  const std = standardizer(duels.flatMap((x) => [x.va, x.vb]));
  let ok = 0;
  let tot = 0;
  for (let f = 0; f < k; f++) {
    const test = new Set(hard.filter((_, j) => j % k === f).map((e) => e.i));
    const train = duels.filter((_, i) => !test.has(i));
    const m = fitBT(train, { ...opts, mu: std.mu, sd: std.sd });
    for (const i of test) {
      tot++;
      if ((judgeProb(m, duels[i].va, duels[i].vb) > 0.5 ? 1 : 0) === duels[i].y) ok++;
    }
  }
  return tot ? ok / tot : NaN;
}

export function trainJudge(duels: Duel[], opts: { l2?: number } = {}): JudgeModel {
  const m = fitBT(duels, opts);
  m.cv = crossValidate(duels, 5, opts);
  return m;
}

export interface Candidate {
  key: string;
  v: number[];
}

/**
 * Active learning: the next pair to ask about is the one the model is least sure of
 * (|p - 0.5| smallest), preferring clips shown less often; with no model yet, pairs far
 * apart in metric space (informative from the start).
 */
export function chooseDuel(cands: Candidate[], m: JudgeModel | null, shown: Map<string, number>, asked: Set<string>, rng: () => number): [Candidate, Candidate] | null {
  if (cands.length < 2) return null;
  let best: [Candidate, Candidate] | null = null;
  let bestV = -Infinity;
  const tries = Math.min(400, (cands.length * (cands.length - 1)) / 2);
  for (let t = 0; t < tries; t++) {
    const a = cands[Math.floor(rng() * cands.length)];
    const b = cands[Math.floor(rng() * cands.length)];
    if (a === b) continue;
    const key = [a.key, b.key].sort().join('|');
    if (asked.has(key)) continue;
    let v: number;
    if (m && m.n >= 4) v = -Math.abs(judgeProb(m, a.v, b.v) - 0.5);
    else v = a.v.reduce((s, x, k) => s + (Number.isFinite(x) && Number.isFinite(b.v[k]) ? (x - b.v[k]) ** 2 : 0), 0);
    v -= 0.05 * ((shown.get(a.key) ?? 0) + (shown.get(b.key) ?? 0));
    v += 1e-6 * rng();
    if (v > bestV) {
      bestV = v;
      best = rng() < 0.5 ? [a, b] : [b, a];
    }
  }
  return best;
}

/**
 * Weak preference pairs from the population's behaviour: for two presets with metric vectors
 * and enough exposure, the one with the higher implicit liking is preferred. Weighted well
 * below a duel answer (default 0.25).
 */
export function behaviourPairs(items: { v: number[]; likes: number; dislikes: number; weakLikes: number; softDislikes: number; views: number; watch: number }[], weight = 0.25): Duel[] {
  const score = (m: (typeof items)[number]) => (m.likes + 0.3 * m.weakLikes + 0.02 * m.watch / 60 + 1) / (m.likes + m.dislikes + 0.3 * (m.weakLikes + m.softDislikes) + 2);
  const seen = items.filter((m) => m.views >= 2);
  const out: Duel[] = [];
  for (let i = 0; i < seen.length; i++) {
    for (let j = i + 1; j < seen.length; j++) {
      const a = score(seen[i]), b = score(seen[j]);
      if (Math.abs(a - b) < 0.1) continue;
      out.push({ va: seen[i].v, vb: seen[j].v, y: a > b ? 1 : 0, w: weight });
    }
  }
  return out;
}

/**
 * Simulated answers for validating the pipeline before real answers exist: a hidden "true"
 * preference over the features plus logistic noise. Clearly labelled as simulated wherever shown.
 */
export function simulateDuels(cands: Candidate[], n: number, trueW: number[], rng: () => number, noise = 1): Duel[] {
  const std = standardizer(cands.map((c) => c.v));
  const out: Duel[] = [];
  for (let i = 0; i < n; i++) {
    const a = cands[Math.floor(rng() * cands.length)];
    let b = cands[Math.floor(rng() * cands.length)];
    if (a === b) b = cands[(cands.indexOf(a) + 1) % cands.length];
    const sa = judgeScore({ w: trueW, ...std }, a.v);
    const sb = judgeScore({ w: trueW, ...std }, b.v);
    const u = rng();
    const p = sigmoid((sa - sb) / noise);
    out.push({ va: a.v, vb: b.v, y: u < p ? 1 : 0 });
  }
  return out;
}

// ------------------------------------------------------------------ saved judge (duel page)

export const JUDGE_KEY = 'v2.avqJudge';

export interface SavedJudge {
  version: 1;
  features: string[];
  n: number;
  /** True once MIN_DUELS_TO_APPLY answers exist: the screener / fitness may use it. */
  applied: boolean;
  sync: JudgeModel | null;
  like: JudgeModel | null;
}

export function loadJudge(): SavedJudge | null {
  try {
    if (typeof localStorage === 'undefined') return null;
    const j = JSON.parse(localStorage.getItem(JUDGE_KEY) ?? 'null') as SavedJudge | null;
    return j && j.version === 1 ? j : null;
  } catch {
    return null;
  }
}

/**
 * The like-judge's preference (0..1) from the screener's cheap metrics (onset hit lift ->
 * sync, metronome motion divergence -> events; the rest unknown), or undefined until the judge applies.
 */
export function screenJudge(m: { hitLift: number; events: number }, j: SavedJudge | null = loadJudge()): number | undefined {
  if (!j?.applied || !j.like) return undefined;
  const c = (x: number) => (Number.isFinite(x) ? Math.max(0, Math.min(1, x)) : NaN);
  const v = JUDGE_FEATURES.map((f) => (f === 'sync' ? c(m.hitLift / 0.5) : f === 'events' ? c(m.events / 0.5) : NaN));
  return sigmoid(judgeScore(j.like, v));
}
