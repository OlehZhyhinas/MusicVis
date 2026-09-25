// Tests for phenotype fingerprints (reference clip, features, normalisation,
// distance), visual duplicate detection and fingerprint persistence.
// Called from v2-test.ts.

import {
  CLIP, DUP_FP_DIST, FEATURES, FEATURE_COUNT, FP_VERSION, FeatureNorm, FingerprintAcc, GROUPS, ReferenceClip,
  blockFlow, clipPart, fpDistance, frameFeatures, validFingerprint,
} from '../src/v2/fingerprint';
import { Population, fitness } from '../src/v2/population';
import { SpringLayout } from '../src/v2/springLayout';
import { agreement, answerRows, chooseTriplet, fitAnswers, parseAnswers, type SimilarityAnswer } from '../src/v2/similarity';
import { MIN_ANSWERS_TO_APPLY } from '../src/v2/phenotype';
import { groupTerms } from '../src/v2/fingerprint';
import { EMB_DIM, EMB_MODEL, EmbeddingStore, cosineDistance, normalize, poolFrames } from '../src/v2/embedding';
import { Phenotype } from '../src/v2/phenotype';
import {
  EXPLORE_ACCEPT, EXPLORE_MODES, EXPLORE_WEIGHT, NoveltyArchive, exploreScore, knnNovelty, noveltyTable, noveltyWeight, parseExploreMode,
  relNovelty, voteCount,
} from '../src/v2/novelty';

type Check = (name: string, ok: boolean, detail: string) => void;

const W = 160, H = 90;
// Pixel centres: the picture's centre lies between pixels, as in GL.
const CX = (W - 1) / 2, CY = (H - 1) / 2;

/** Procedural test pictures, RGBA8. */
type Pattern = (x: number, y: number, t: number, beat: number) => [number, number, number];
const PATTERNS: Record<string, Pattern> = {
  rings: (x, y, t, beat) => {
    const r = Math.hypot(x - CX, y - CY);
    const v = 0.5 + 0.5 * Math.cos(r * 0.8 - t * 4);
    return [v * (0.5 + 0.5 * beat), v * 0.3, v];
  },
  stripes: (x, _y, t) => {
    const v = 0.5 + 0.5 * Math.sin((x + t * 40) * 0.35);
    return [v, v * 0.9, 0.2 * v];
  },
  petals: (x, y, t, beat) => {
    const a = Math.atan2(y - CY, x - CX);
    const r = Math.hypot(x - CX, y - CY);
    const v = Math.max(0, Math.cos(a * 6 + t)) * Math.exp(-r / 30) * (0.6 + 0.4 * beat);
    return [v * 0.2, v, v * 0.6];
  },
  blobs: (x, y, t) => {
    let v = 0;
    for (let i = 0; i < 7; i++) {
      const cx = W * (0.15 + 0.7 * ((i * 0.37 + t * 0.05) % 1)), cy = H * (0.2 + 0.6 * ((i * 0.61) % 1));
      v += Math.exp(-((x - cx) ** 2 + (y - cy) ** 2) / 30);
    }
    return [Math.min(1, v), Math.min(1, v * 0.5), 0.1];
  },
};

function paint(p: Pattern, t: number, beat: number, out = new Uint8Array(W * H * 4)): Uint8Array {
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    const [r, g, b] = p(x, y, t, beat);
    const i = (y * W + x) * 4;
    out[i] = Math.round(Math.max(0, Math.min(1, r)) * 255);
    out[i + 1] = Math.round(Math.max(0, Math.min(1, g)) * 255);
    out[i + 2] = Math.round(Math.max(0, Math.min(1, b)) * 255);
    out[i + 3] = 255;
  }
  return out;
}

/** Fingerprint of a pattern "rendered" on the reference clip with the same schedule as fingerprintRender.ts (coarser). */
function fingerprintOf(p: Pattern): number[] {
  const acc = new FingerprintAcc(W, H);
  const clip = new ReferenceClip();
  const total = Math.round(CLIP.end * CLIP.fps);
  for (let f = 1; f <= total; f++) {
    const s = clip.next();
    if (f % 8 !== 0) continue;
    acc.sample(paint(p, s.time, s.beatPulse), s.time, f % 32 === 0);
    if (f % 48 === 0) acc.pair(paint(p, s.time, s.beatPulse), paint(p, s.time + 1 / 60, s.beatPulse));
  }
  return acc.finish();
}

/** In-memory stand-in for the IndexedDB key-value store. */
class MemKV {
  data = new Map<string, unknown>();
  async get<T>(k: string): Promise<T | undefined> {
    return this.data.get(k) as T | undefined;
  }
  async set(k: string, v: unknown): Promise<void> {
    this.data.set(k, JSON.parse(JSON.stringify(v)));
  }
}

/** A random but reproducible fingerprint cloud: `n` points around `centre` with spread `s` (in feature floors). */
function cloud(n: number, seed: number, centre: number, s: number): number[][] {
  let x = seed >>> 0;
  const r = () => ((x = (x * 1664525 + 1013904223) >>> 0) / 2 ** 32) - 0.5;
  return Array.from({ length: n }, () => FEATURES.map((f) => centre * f.floor * 10 + r() * s * f.floor * 10));
}

export async function noveltyTestsAsync(check: Check): Promise<void> {
  await similarityTests(check);
  // ------------------------------------------------ archive
  {
    const a = new NoveltyArchive(5);
    const pts = cloud(8, 1, 0, 1);
    pts.forEach((p, i) => a.add(`G1-000${i}`, p, i));
    a.add('G1-0007', pts[7], 99); // refresh keeps one entry
    a.add('bad', [1, 2, 3]);
    const back = NoveltyArchive.fromJSON(JSON.parse(JSON.stringify(a.toJSON())), 5);
    const other = NoveltyArchive.fromJSON({ ...a.toJSON(), fpVersion: FP_VERSION + 1 });
    const junk = NoveltyArchive.fromJSON({ format: 'nope' });
    check('novelty.archive', a.size === 5 && a.entries[0].id === 'G1-0003' && !a.has('bad') && back.size === 5 && back.entries[4].t === 99 && other.size === 0 && junk.size === 0,
      `capped at 5 (oldest dropped), refresh keeps one entry, invalid vectors refused, save/load keeps ${back.size}; another fingerprint version or a foreign file loads empty`);
  }
  {
    // Persistence + migration: a population fingerprinted before the archive existed fills it; culled members stay in it.
    const pop = Population.seeded(5);
    const pts = cloud(pop.size, 7, 0, 1);
    pop.list().forEach((m, i) => {
      m.fp = pts[i];
      m.fpv = FP_VERSION;
    });
    const kv = new MemKV();
    const ph = new Phenotype(null, () => pop);
    await ph.attachStore(kv);
    const migrated = ph.archive.size;
    await ph.flush();
    const gone = pop.list()[3];
    pop.members.delete(gone.id);
    const ph2 = new Phenotype(null, () => pop);
    await ph2.attachStore(kv);
    const stored = (kv.data.get('novelty-archive') as { entries: unknown[] }).entries.length;
    check('novelty.archive-persist', migrated === pts.length && stored === pts.length && ph2.archive.has(gone.id) && ph2.archive.size === pts.length,
      `${migrated} old fingerprints migrated into the archive and saved; after ${gone.id} is culled it is still archived (${ph2.archive.size})`);
  }

  // ------------------------------------------------ k-NN novelty
  {
    const norm = FeatureNorm.identity();
    const crowd = cloud(40, 3, 0, 1);
    const a = new NoveltyArchive();
    crowd.forEach((p, i) => a.add(`C${i}`, p));
    const inside = { id: 'X1', fp: cloud(1, 11, 0, 0.5)[0] };
    const outside = { id: 'X2', fp: cloud(1, 12, 3, 0.5)[0] };
    const { table, typical } = noveltyTable([inside, outside, { id: 'C0', fp: crowd[0] }], a, norm);
    const zs = crowd.map((p, i) => ({ id: `C${i}`, z: norm.z(p) }));
    const k1 = knnNovelty(norm.z(crowd[0]), zs, 1, 'C0');
    const self = knnNovelty(norm.z(crowd[0]), zs, 1);
    const ri = table.get('X1')!.rel, ro = table.get('X2')!.rel;
    const pct = relNovelty(typical[Math.floor(typical.length / 2)], typical);
    check('novelty.knn', ro === 1 && ri < 0.6 && table.get('X2')!.nov > table.get('X1')!.nov && self === 0 && k1 > 0 && Math.abs(pct - 0.5) < 0.05 && relNovelty(-1, typical) === 0,
      `a look inside the crowd rel ${ri.toFixed(2)}, far outside rel ${ro.toFixed(2)}; a member's own archive entry is excluded; the archive median maps to ${pct.toFixed(2)}`);
  }

  // ------------------------------------------------ exploration scoring
  {
    const bad: string[] = [];
    // Off: pure fitness. Novelty never helps in off mode.
    if (exploreScore(0.4, 1, 'off', 0) !== 0.4) bad.push('off adds novelty');
    // Modes are ordered.
    const ws = EXPLORE_MODES.map((m) => noveltyWeight(m, 0));
    if (!ws.every((w, i) => i === 0 || w > ws[i - 1])) bad.push(`weights ${ws}`);
    // The weight tapers with votes: halved after VOTE_HALF votes. A well-liked preset
    // beats an unvoted maximally novel one in explore mode; in wild mode the novel one
    // gets its airtime, but once it has three dislikes the liked one wins again.
    if (Math.abs(noveltyWeight('wild', 3) - EXPLORE_WEIGHT.wild / 2) > 1e-9) bad.push('taper');
    const liked = { likes: 12, dislikes: 1, weakLikes: 0, softDislikes: 0 };
    const unvoted = { likes: 0, dislikes: 0, weakLikes: 0, softDislikes: 0 };
    const disliked = { likes: 0, dislikes: 3, weakLikes: 0, softDislikes: 0 };
    const sc = (v: typeof liked, rel: number, m: 'explore' | 'wild') => exploreScore(fitness({ ...v } as never), rel, m, voteCount(v));
    const likedScore = sc(liked, 0.1, 'explore'), novelScore = sc(unvoted, 1, 'explore');
    if (!(likedScore > novelScore)) bad.push(`explore: liked ${likedScore.toFixed(3)} vs novel ${novelScore.toFixed(3)}`);
    if (!(sc(unvoted, 1, 'wild') > sc(liked, 0.1, 'wild') && sc(disliked, 1, 'wild') < sc(liked, 0.1, 'wild'))) bad.push('wild ordering');
    // Between two unvoted presets, the novel one wins in every mode but off.
    for (const m of EXPLORE_MODES.slice(1)) if (!(exploreScore(0.2, 0.9, m, 0) > exploreScore(0.2, 0.1, m, 0))) bad.push(`${m} ignores novelty`);
    if (!(EXPLORE_ACCEPT.off === 0 && EXPLORE_ACCEPT.gentle === 0 && EXPLORE_ACCEPT.wild > EXPLORE_ACCEPT.explore)) bad.push('accept floors');
    if (parseExploreMode('wild') !== 'wild' || parseExploreMode('bogus') !== 'gentle') bad.push('parse');
    check('novelty.explore-score', !bad.length, bad.join(', ') || `weights ${ws.join(' / ')}; halves after 3 votes; explore: a liked preset (12:1) outranks an unvoted maximally novel one (${likedScore.toFixed(2)} > ${novelScore.toFixed(2)}); wild: the novel one first, until 3 dislikes`);
  }
  {
    // Phenotype: score / acceptance use the mode; unfingerprinted members count as average.
    const pop = Population.seeded(9);
    const ms = pop.list();
    const crowd = cloud(ms.length - 1, 5, 0, 1);
    ms.slice(0, -1).forEach((m, i) => {
      m.fp = crowd[i];
      m.fpv = FP_VERSION;
    });
    const ph = new Phenotype(null, () => pop);
    ph.syncArchive();
    const odd = ms[ms.length - 1];
    ph.mode = 'off';
    const offBonus = ph.bonus(odd);
    ph.mode = 'wild';
    const unknown = ph.bonus(odd);
    const far = cloud(1, 99, 4, 0.2)[0];
    const near = crowd[0].map((v, i) => v + FEATURES[i].floor * 0.2);
    const accFar = ph.acceptNovelty(far), accNear = ph.acceptNovelty(near);
    ph.mode = 'gentle';
    const gentleNear = ph.acceptNovelty(near);
    check('novelty.phenotype', offBonus === 0 && Math.abs(unknown - EXPLORE_WEIGHT.wild * 0.5) < 1e-9 && accFar.ok && !accNear.ok && gentleNear.ok,
      `off adds nothing; an unmeasured member counts as rel 0.5; wild keeps a far-away child (rel ${accFar.rel.toFixed(2)}) and turns away a near copy (rel ${accNear.rel.toFixed(2)}); gentle turns nothing away`);
  }
}

function layoutTests(check: Check): void {
  // Three clusters of looks (distance ~0.3 inside, 2.5 across) become tight, separated clusters on the map.
  const n = 60;
  const ids = Array.from({ length: n }, (_, i) => `G1-${String(i).padStart(4, '0')}`);
  const group = (i: number) => i % 3;
  const dist = (i: number, j: number) => (group(i) === group(j) ? 0.3 + ((i * 7 + j * 3) % 5) * 0.03 : 2.5 + ((i + j) % 3) * 0.1);
  const L = new SpringLayout();
  L.setGraph(ids, dist);
  const t0 = performance.now();
  L.settle(3000);
  const ms = performance.now() - t0;
  const d = (a: number, b: number) => Math.hypot(L.nodes[a].x - L.nodes[b].x, L.nodes[a].y - L.nodes[b].y);
  const edge = new Set(L.edges.map((e) => `${e.a}:${e.b}`));
  let el = 0, en = 0, nl = 0, nn = 0;
  let gap = Infinity;
  const nnInside: number[] = [];
  for (let i = 0; i < n; i++) {
    let best = Infinity;
    for (let j = 0; j < n; j++) {
      if (j === i) continue;
      if (group(i) === group(j)) best = Math.min(best, d(i, j));
      else gap = Math.min(gap, d(i, j));
      if (j > i) {
        if (edge.has(`${i}:${j}`)) { el += d(i, j); en++; } else { nl += d(i, j); nn++; }
      }
    }
    nnInside.push(best);
  }
  nnInside.sort((a, b) => a - b);
  const nnMed = nnInside[n >> 1];
  el /= en;
  nl /= nn;
  const finite = L.nodes.every((p) => Number.isFinite(p.x) && Number.isFinite(p.y));
  let overlaps = 0;
  for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) if (d(i, j) < 26) overlaps++;
  // Incremental: a child that looks like node 5 appears next to it; the old nodes keep their places.
  const before = L.nodes.map((p) => [p.x, p.y]);
  const ids2 = [...ids, 'G2-0100'];
  const dist2 = (i: number, j: number) => (i === n || j === n ? ((i === n ? j : i) === 5 ? 0.05 : dist(5, i === n ? j : i) + 0.05) : dist(i, j));
  const { added } = L.setGraph(ids2, dist2);
  const child = L.get('G2-0100')!;
  const nearest = Math.hypot(child.x - L.nodes[5].x, child.y - L.nodes[5].y);
  const kept = before.every(([x, y], i) => L.nodes[i].x === x && L.nodes[i].y === y);
  const reheated = !L.settled && L.alpha <= 0.25;
  // A node without a fingerprint links to its parents.
  const ids3 = [...ids2, 'G3-0200'];
  const dist3 = (i: number, j: number) => (i === n + 1 || j === n + 1 ? NaN : dist2(i, j));
  L.setGraph(ids3, dist3, (i) => (i === n + 1 ? [40] : []));
  const orphanEdges = L.edges.filter((e) => e.a === n + 1 || e.b === n + 1).length;
  L.settle(400);
  check('map.layout', finite && el < nl * 0.35 && gap > nnMed * 1.5 && overlaps === 0 && added.length === 1 && nearest < 60 && kept && reheated && orphanEdges === 1,
    `3 clusters of 20: mean spring length ${el.toFixed(0)} vs ${nl.toFixed(0)} between unlinked nodes; closest pair across clusters ${gap.toFixed(0)} vs median neighbour spacing ${nnMed.toFixed(0)} inside; no overlapping discs; settled in ${ms.toFixed(0)} ms; a look-alike child appears ${nearest.toFixed(0)} px from its twin, old nodes stay put, gentle re-heat; unfingerprinted nodes hang off their parents`);
  // Scale: 500 nodes step fast enough to keep the visualizer smooth.
  const big = Array.from({ length: 500 }, (_, i) => `G1-${String(i).padStart(4, '0')}`);
  const B = new SpringLayout();
  B.setGraph(big, (i, j) => 0.4 + (Math.abs(Math.sin(i * 12.9898 + j * 78.233)) + Math.abs(Math.sin(j * 12.9898 + i * 78.233))) * 0.8);
  const t1 = performance.now();
  for (let i = 0; i < 50; i++) B.step();
  const per = (performance.now() - t1) / 50;
  check('map.layout-scale', per < 6, `500 nodes, ${B.edges.length} springs: ${per.toFixed(2)} ms per step`);
}

async function similarityTests(check: Check): Promise<void> {
  // A simulated user who judges with hidden group weights (colour and motion matter, detail and response barely).
  const hidden = [3, 0.4, 1, 2, 0.2];
  const pts = cloud(80, 21, 0, 2);
  const ids = pts.map((_, i) => `G1-${String(i).padStart(4, '0')}`);
  const norm = FeatureNorm.fit(pts);
  let seed = 5;
  const rnd = () => ((seed = (seed * 1664525 + 1013904223) >>> 0) / 2 ** 32);
  const wd = (w: number[], t: number[]) => t.reduce((s2, v, i) => s2 + w[i] * v, 0);
  const answers: SimilarityAnswer[] = [];
  const seen = new Set<string>();
  const cands = pts.map((fp, i) => ({ id: ids[i], fp }));
  let uncertainSum = 0;
  for (let k = 0; k < 90; k++) {
    const t = chooseTriplet(cands, norm, [1, 1, 1, 1, 1], rnd, { seen });
    if (!t) break;
    seen.add([t.ref, ...[t.a, t.b].sort()].join('|'));
    uncertainSum += t.uncertainty;
    const fp = (id: string) => pts[ids.indexOf(id)];
    const ta = groupTerms(norm.z(fp(t.ref)), norm.z(fp(t.a))), tb = groupTerms(norm.z(fp(t.ref)), norm.z(fp(t.b)));
    let pick: 'a' | 'b' = wd(hidden, ta) < wd(hidden, tb) ? 'a' : 'b';
    if (rnd() < 0.1) pick = pick === 'a' ? 'b' : 'a';
    answers.push({ ref: t.ref, a: t.a, b: t.b, pick, t: k, fps: [fp(t.ref), fp(t.a), fp(t.b)] });
  }
  const fit = fitAnswers(answers, norm);
  const { x, y } = answerRows(answers, norm);
  const ceiling = agreement(x, y, hidden);
  const hn = hidden.map((v) => v / (hidden.reduce((s2, q) => s2 + q, 0) / hidden.length));
  const order = (w: number[]) => w.map((v, i) => [v, i]).sort((p, q) => q[0] - p[0]).map((p) => p[1]).join('');
  check('similarity.fit', answers.length === 90 && fit.agreeFit > fit.agreeEqual + 0.1 && order(fit.weights).slice(0, 2) === order(hidden).slice(0, 2) && fit.weights[0] > fit.weights[1] * 2,
    `90 simulated answers (10% noise) on uncertain triplets (mean uncertainty ${(uncertainSum / answers.length).toFixed(2)}): equal weights ${Math.round(fit.agreeEqual * 100)}%, fitted ${Math.round(fit.agreeFit * 100)}% cross-validated (the hidden weights themselves ${Math.round(ceiling * 100)}%); weights ${fit.weights.map((v) => v.toFixed(2)).join('/')} vs hidden ${hn.map((v) => v.toFixed(2)).join('/')}`);

  // Triplets: never repeat a seen one; a reference's candidates are its near neighbours.
  const t1 = chooseTriplet(cands, norm, [1, 1, 1, 1, 1], () => 0.3, { seen: new Set() })!;
  const t2 = chooseTriplet(cands, norm, [1, 1, 1, 1, 1], () => 0.3, { seen: new Set([[t1.ref, ...[t1.a, t1.b].sort()].join('|')]) })!;
  const few = chooseTriplet(cands.slice(0, 2), norm, [1, 1, 1, 1, 1], rnd);
  check('similarity.triplet', !!t1 && !!t2 && [t1.ref, t1.a, t1.b].join() !== [t2.ref, t2.a, t2.b].join() && new Set([t1.ref, t1.a, t1.b]).size === 3 && few === null,
    `distinct presets, a seen triplet is not asked again, none with fewer than 3 fingerprints (uncertainty ${t1.uncertainty.toFixed(2)}, group disagreement ${t1.disagreement.toFixed(2)})`);

  // Phenotype: answers persist, fitted weights apply after MIN_ANSWERS_TO_APPLY.
  const pop = Population.seeded(3);
  const ms = pop.list();
  ms.forEach((m, i) => {
    m.fp = pts[i % pts.length];
    m.fpv = FP_VERSION;
  });
  const kv = new MemKV();
  const ph = new Phenotype(null, () => pop);
  await ph.attachStore(kv);
  let appliedAt = -1;
  for (let k = 0; k < 12; k++) {
    const [r, a, b] = [ms[k], ms[k + 1], ms[k + 2]];
    const ta = groupTerms(ph.norm.z(r.fp!), ph.norm.z(a.fp!)), tb = groupTerms(ph.norm.z(r.fp!), ph.norm.z(b.fp!));
    ph.addAnswer(r, a, b, wd(hidden, ta) < wd(hidden, tb) ? 'a' : 'b');
    if (appliedAt < 0 && ph.metricVersion > 0) appliedAt = k + 1;
  }
  const ph2 = new Phenotype(null, () => pop);
  await ph2.attachStore(kv);
  const bad = parseAnswers({ format: 'musicvis-v2-similarity', answers: [{ ref: 'x', a: 'y', b: 'z', pick: 'c', fps: [] }, answers[0]] });
  check('similarity.persist', appliedAt === MIN_ANSWERS_TO_APPLY && ph2.answers.length === 12 && JSON.stringify(ph2.weights) === JSON.stringify(ph.weights) && bad.length === 1 && parseAnswers(null).length === 0,
    `fitted weights take over at answer ${appliedAt}; 12 answers reload with the same weights; malformed answers are dropped`);
}

function embeddingTests(check: Check): void {
  let seed = 3;
  const rnd = () => ((seed = (seed * 1664525 + 1013904223) >>> 0) / 2 ** 32) - 0.5;
  const vec = (base?: number[], noise = 1) => Array.from({ length: EMB_DIM }, (_, i) => (base ? base[i] : 0) + rnd() * noise);
  const a = normalize(vec()), a2 = normalize(vec(a, 0.02)), b = normalize(vec());
  const pooled = poolFrames([a, a2]);
  const st = new EmbeddingStore();
  st.set('A', a);
  st.set('A2', a2);
  st.set('B', b);
  st.set('bad', [1, 2]);
  const back = EmbeddingStore.fromJSON(JSON.parse(JSON.stringify(st.toJSON())));
  const other = EmbeddingStore.fromJSON({ ...st.toJSON(), model: 'another/model' });
  const ok = cosineDistance(a, a) < 1e-6 && cosineDistance(a, a2) < 0.05 && cosineDistance(a, b) > 0.5 && Math.abs(Math.hypot(...pooled) - 1) < 1e-3 &&
    st.size === 3 && st.term('A', 'A2') < st.term('A', 'B') && Number.isNaN(st.term('A', 'zz')) && back.size === 3 && other.size === 0;
  check('embedding.store', ok, `cosine distances ${cosineDistance(a, a2).toFixed(3)} (near copy) vs ${cosineDistance(a, b).toFixed(3)} (unrelated); pooled frames unit length; terms scaled by the median distance; save/load keeps ${back.size}; another model's vectors (not ${EMB_MODEL}) are dropped`);

  // Blend weight: answers from a user who sees what the embedding sees raise its weight; a user who ignores it lowers it.
  const pts = cloud(40, 31, 0, 2);
  const norm = FeatureNorm.fit(pts);
  const embs = pts.map(() => normalize(vec()));
  const est = new EmbeddingStore();
  embs.forEach((e, i) => est.set(`P${i}`, e));
  const make = (byEmb: boolean): SimilarityAnswer[] => {
    const out: SimilarityAnswer[] = [];
    for (let k = 0; k < 80; k++) {
      const r = k % 40, x = (k * 7 + 3) % 40, y = (k * 13 + 5) % 40;
      if (r === x || r === y || x === y) continue;
      const ea = est.term(`P${r}`, `P${x}`), eb = est.term(`P${r}`, `P${y}`);
      const ha = groupTerms(norm.z(pts[r]), norm.z(pts[x])).reduce((s2, v) => s2 + v, 0), hb = groupTerms(norm.z(pts[r]), norm.z(pts[y])).reduce((s2, v) => s2 + v, 0);
      const pick: 'a' | 'b' = (byEmb ? ea < eb : ha < hb) ? 'a' : 'b';
      out.push({ ref: `P${r}`, a: `P${x}`, b: `P${y}`, pick, t: k, fps: [pts[r], pts[x], pts[y]], emb: [ea, eb] });
    }
    return out;
  };
  const fe = fitAnswers(make(true), norm, true), fh = fitAnswers(make(false), norm, true);
  const he = fitAnswers(make(true), norm, false);
  const we = fe.weights[5], wh = fh.weights[5];
  check('embedding.blend', we > 2 && wh < 1 && fe.agreeFit > he.agreeFit + 0.15,
    `embedding weight ${we.toFixed(2)} when answers follow the embedding (agreement ${Math.round(fe.agreeFit * 100)}% with it vs ${Math.round(he.agreeFit * 100)}% without), ${wh.toFixed(2)} when they follow the hand features`);
}

export function noveltyTests(check: Check): void {
  embeddingTests(check);
  layoutTests(check);
  // ------------------------------------------------ reference clip
  {
    const a = new ReferenceClip(), b = new ReferenceClip();
    let same = true;
    let changes = 0;
    const parts = new Set<string>();
    let maxDrop = 0, maxBuild = 0;
    for (let i = 0; i < CLIP.end * CLIP.fps; i++) {
      const x = a.next(), y = b.next();
      if (x.bass !== y.bass || x.beatPulse !== y.beatPulse || x.spectrum[3] !== y.spectrum[3] || x.section.label !== y.section.label) same = false;
      if (x.sectionChanged) changes++;
      parts.add(x.section.label);
      maxDrop = Math.max(maxDrop, x.dropPulse);
      maxBuild = Math.max(maxBuild, x.buildIntensity);
    }
    a.reset();
    const first = a.next().bass;
    b.reset();
    check('fingerprint.clip', same && changes === 2 && parts.size === 3 && maxDrop > 0.9 && maxBuild > 0.9 && first === b.next().bass && clipPart(0.1) === 'calm' && clipPart(CLIP.end - 0.1) === 'drop',
      `deterministic, ${changes} section changes over ${[...parts].join('/')}, dropPulse ${maxDrop.toFixed(2)}, build ${maxBuild.toFixed(2)}, ${CLIP.end.toFixed(2)} s`);
  }

  // ------------------------------------------------ features
  const fps: Record<string, number[]> = {};
  for (const k of Object.keys(PATTERNS)) fps[k] = fingerprintOf(PATTERNS[k]);
  {
    const again = fingerprintOf(PATTERNS.rings);
    check('fingerprint.determinism', JSON.stringify(again) === JSON.stringify(fps.rings) && validFingerprint(again) && again.length === FEATURE_COUNT,
      `${FEATURE_COUNT} features in ${GROUPS.length} groups, identical on a second run`);
  }
  {
    const at = (fp: number[], key: string) => fp[FEATURES.findIndex((f) => f.key === key)];
    const bad: string[] = [];
    if (!(at(fps.rings, 'rings') > at(fps.stripes, 'rings'))) bad.push('rings');
    if (!(at(fps.petals, 'kfold') > at(fps.stripes, 'kfold') && at(fps.petals, 'kfold') > at(fps.blobs, 'kfold'))) bad.push('kfold');
    if (!(at(fps.blobs, 'blobs') > at(fps.petals, 'blobs'))) bad.push('blobs');
    if (!(at(fps.stripes, 'trans') > 0.5)) bad.push(`trans ${at(fps.stripes, 'trans')}`);
    if (!(at(fps.rings, 'rot') > 0.9 && at(fps.stripes, 'symY') > 0.9)) bad.push('symmetry');
    if (!(at(fps.rings, 'beatLum') > 0.3 && at(fps.stripes, 'beatLum') < 0.2)) bad.push(`beatLum ${at(fps.rings, 'beatLum')} / ${at(fps.stripes, 'beatLum')}`);
    if (!(at(fps.stripes, 'hueX') > at(fps.petals, 'hueX'))) bad.push('hue');
    check('fingerprint.traits', !bad.length, bad.join(', ') || 'rings, n-fold petals, blobs, drift, symmetry, beat flash and hue each show on the matching picture');
  }
  {
    // Flow: a picture shifted right by 2 px reads as drift; a zoom reads as divergence.
    const a = paint(PATTERNS.blobs, 1, 0);
    const shift = new Uint8Array(a.length);
    for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) for (let c = 0; c < 4; c++) shift[(y * W + x) * 4 + c] = a[(y * W + Math.max(0, x - 2)) * 4 + c];
    const L = (px: Uint8Array) => Float32Array.from({ length: W * H }, (_, i) => px[i * 4] / 255);
    const f1 = blockFlow(L(a), L(shift), W, H);
    const tex: Pattern = (x, y) => { const v = 0.5 + 0.5 * Math.sin(x * 0.5) * Math.cos(y * 0.6); return [v, v, v]; };
    const zoomed = paint((x, y) => tex(W / 2 + (x - W / 2) / 1.05, H / 2 + (y - H / 2) / 1.05, 0, 0), 0, 0);
    const f2 = blockFlow(L(paint(tex, 0, 0)), L(zoomed), W, H);
    check('fingerprint.flow', !!f1 && f1.trans > 1.5 && !!f2 && f2.div > 0.5, `shift: drift ${f1?.trans.toFixed(2)}; zoom: divergence ${f2?.div.toFixed(2)}`);
  }
  {
    const s = frameFeatures(paint(PATTERNS.rings, 0, 1), W, H);
    const black = frameFeatures(new Uint8Array(W * H * 4), W, H);
    const finite = Object.values(black).every(Number.isFinite) && Object.values(s).every(Number.isFinite);
    check('fingerprint.black-frame', finite && black.lum === 0 && black.blobs === 0, 'an all-black frame gives finite, zero traits');
  }

  // ------------------------------------------------ normalisation and distance
  const all = Object.values(fps);
  const norm = FeatureNorm.fit(all);
  {
    const ks = Object.keys(fps);
    let selfZero = true, symmetric = true, triangle = true, positive = true;
    for (const a of ks) for (const b of ks) {
      const d = fpDistance(fps[a], fps[b], norm);
      if (a === b && d !== 0) selfZero = false;
      if (Math.abs(d - fpDistance(fps[b], fps[a], norm)) > 1e-9) symmetric = false;
      if (a !== b && !(d > DUP_FP_DIST)) positive = false;
      for (const c of ks) if (d > fpDistance(fps[a], fps[c], norm) + fpDistance(fps[c], fps[b], norm) + 1e-9) triangle = false;
    }
    // A slightly perturbed fingerprint stays within the duplicate distance.
    const near = fps.rings.map((v, i) => v + norm.scale[i] * 0.05 * ((i % 3) - 1));
    const dn = fpDistance(fps.rings, near, norm);
    check('fingerprint.distance', selfZero && symmetric && triangle && positive && dn < DUP_FP_DIST,
      `self 0, symmetric, triangle inequality, distinct pictures > ${DUP_FP_DIST}, a 5%-of-spread wobble ${dn.toFixed(3)}`);
    const id = FeatureNorm.identity();
    const few = FeatureNorm.fit(all.slice(0, 2));
    const scalesOk = norm.scale.every((s, i) => s >= FEATURES[i].floor);
    const clipped = norm.z(fps.rings.map((v) => v + 1e6)).every((z) => z <= 4);
    check('fingerprint.norm', scalesOk && clipped && few.n === 0 && id.n === 0 && norm.n === all.length,
      `robust scales never below the floors, z clipped at 4, identity until 3 fingerprints (fitted on ${norm.n})`);
  }

  // ------------------------------------------------ duplicates and persistence
  {
    const pop = Population.seeded(1000);
    const ms = pop.list();
    const ks = Object.keys(fps);
    ms.slice(0, ks.length).forEach((m, i) => {
      m.fp = fps[ks[i]];
      m.fpv = FP_VERSION;
    });
    const ph = new Phenotype(null, () => pop);
    ph.refit(true);
    const near = fps.petals.map((v, i) => v + ph.norm.scale[i] * 0.03);
    const hit = ph.duplicateOf(near);
    const miss = ph.duplicateOf(fingerprintOf((x, y, t) => [0.5 + 0.5 * Math.sin(x * 0.9 + y * 0.9 + t * 9), 0, 0.5]));
    check('fingerprint.duplicate', hit?.id === ms[ks.indexOf('petals')].id && !miss && ph.missing().length === ms.length - ks.length,
      `near-copy of petals flagged as ${hit?.id} (${hit?.dist.toFixed(3)}); a new picture passes; ${ph.missing().length} members still to fingerprint`);

    const back = Population.fromJSON(JSON.parse(JSON.stringify(pop.toJSON())));
    const kept = back.list().filter((m) => validFingerprint(m.fp)).length;
    const stale = pop.toJSON();
    stale.members[0].fpv = FP_VERSION + 1;
    (stale.members[1] as { fp?: unknown }).fp = [1, 2, 3];
    const back2 = Population.fromJSON(JSON.parse(JSON.stringify(stale)));
    const noFp = pop.toJSON();
    for (const m of noFp.members) {
      delete m.fp;
      delete m.fpv;
    }
    const back3 = Population.fromJSON(JSON.parse(JSON.stringify(noFp)));
    // A re-encoded seed drops its fingerprint (it looks different now).
    const up = Population.fromJSON(JSON.parse(JSON.stringify(pop.toJSON())));
    const s0 = up.list()[0];
    s0.genome = { ...s0.genome, energy: [0.01, 0.02] };
    const changed = up.upgradeSeeds();
    check('fingerprint.persist', kept === ks.length && back2.list().filter((m) => validFingerprint(m.fp)).length === ks.length - 2 &&
      back3.list().every((m) => m.fp === undefined) && changed.includes(s0.id) && s0.fp === undefined,
      `${kept} fingerprints survive save/load; another version or a malformed vector is dropped; old files load without; a re-encoded seed is re-measured`);
  }
}
