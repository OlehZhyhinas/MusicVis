// Phenotype controller: fingerprints members in the background (idle work,
// one at a time, only while nothing else is rendering offscreen), keeps the
// robust normalisation fitted to every fingerprint seen, and answers "does
// this child look like something we already have?" for breeding. Owns the
// novelty archive (every fingerprint ever shown or bred, persisted) and the
// exploration mode.

import type { Genome } from './genome';
import type { Member, Population } from './population';
import { DUP_FP_DIST, FP_VERSION, FeatureNorm, validFingerprint, zDistance, type GroupWeights, EQUAL_WEIGHTS } from './fingerprint';
import type { Fingerprinter } from './fingerprintRender';
import {
  NoveltyArchive, exploreScore, knnNovelty, noveltyTable, relNovelty, voteCount, EXPLORE_ACCEPT,
  type ExploreMode,
} from './novelty';
import { fitness } from './population';
import { GROUPS } from './fingerprint';
import { fitAnswers, parseAnswers, type SimilarityAnswer, type SimilarityFit } from './similarity';
import { EmbeddingStore, type Embedder } from './embedding';

export interface KV {
  get<T>(key: string): Promise<T | undefined>;
  set(key: string, value: unknown): Promise<void>;
}

const ARCHIVE_KEY = 'novelty-archive';
const ANSWERS_KEY = 'similarity-answers';
const EMB_KEY = 'embeddings';
/** Answers needed before the fitted weights replace equal weights. */
export const MIN_ANSWERS_TO_APPLY = 8;

export interface DuplicateHit {
  id: string;
  dist: number;
}

export class Phenotype {
  norm = FeatureNorm.identity();
  weights: GroupWeights = { ...EQUAL_WEIGHTS };
  private fitN = 0;
  private failed = new Set<string>();
  private running = false;
  private timer = 0;
  private zCache = new WeakMap<number[], Float32Array>();
  /** Called after a member got its fingerprint (the evolution saves, the map refreshes). */
  onFingerprint: ((m: Member) => void) | null = null;
  archive = new NoveltyArchive();
  mode: ExploreMode = 'gentle';
  private store: KV | null = null;
  private saveTimer = 0;
  private table: Map<string, { nov: number; rel: number }> = new Map();
  private tableKey = '';
  private typical: number[] = [];
  /** Similarity judgements and the current fit of the group weights to them. */
  answers: SimilarityAnswer[] = [];
  fit: SimilarityFit | null = null;
  /** Bumped when the metric (weights) changes: the map re-lays out. */
  metricVersion = 0;
  /** Agreement history (cross-validated, after each answer), for the similarity page. */
  agreeHistory: number[] = [];
  /** Perceptual embeddings (opt-in): vectors by member id, the loader, and the blend weight the answers decide. */
  emb = new EmbeddingStore();
  embedder: Embedder | null = null;
  embOn = false;
  embWeight = 0;
  /** Agreement on the answers that have embeddings: hand features only vs blended (both cross-validated). */
  embReport: { n: number; without: number; with: number } | null = null;
  private embTimer = 0;
  private embAdded = 0;
  private embFailed = new Set<string>();

  readonly fper: Fingerprinter | null;
  private popRef: () => Population;

  constructor(fper: Fingerprinter | null, popRef: () => Population) {
    this.fper = fper;
    this.popRef = popRef;
  }

  private get pop(): Population {
    return this.popRef();
  }

  /** Load the archive; members fingerprinted before the archive existed are added to it. */
  async attachStore(store: KV): Promise<void> {
    this.store = store;
    try {
      this.archive = NoveltyArchive.fromJSON(await store.get<unknown>(ARCHIVE_KEY));
    } catch {
      this.archive = new NoveltyArchive();
    }
    try {
      this.answers = parseAnswers(await store.get<unknown>(ANSWERS_KEY));
    } catch {
      this.answers = [];
    }
    try {
      this.emb = EmbeddingStore.fromJSON(await store.get<unknown>(EMB_KEY));
    } catch {
      this.emb = new EmbeddingStore();
    }
    this.syncArchive();
    this.refitWeights();
  }

  // --------------------------------------------------------- similarity

  addAnswer(ref: Member, a: Member, b: Member, pick: 'a' | 'b'): void {
    if (!validFingerprint(ref.fp) || !validFingerprint(a.fp) || !validFingerprint(b.fp)) return;
    this.answers.push({ ref: ref.id, a: a.id, b: b.id, pick, t: Date.now(), fps: [ref.fp, a.fp, b.fp] });
    void this.store?.set(ANSWERS_KEY, { format: 'musicvis-v2-similarity', answers: this.answers });
    this.refitWeights();
    if (Number.isFinite(this.fit?.agreeFit)) this.agreeHistory.push(this.fit!.agreeFit);
  }

  /**
   * Fit the group weights to the answers; applied once there are enough of
   * them. With the embedding on, answers whose three presets all have
   * embeddings also fit its blend weight, and agreement with and without it is
   * reported on that same subset.
   */
  refitWeights(): void {
    this.fit = this.answers.length ? fitAnswers(this.answers, this.norm) : null;
    const w = { ...EQUAL_WEIGHTS };
    let we = this.embOn ? 1 : 0;
    this.embReport = null;
    if (this.embOn) {
      for (const a of this.answers) {
        const ta = this.emb.term(a.ref, a.a), tb = this.emb.term(a.ref, a.b);
        a.emb = Number.isFinite(ta) && Number.isFinite(tb) ? [ta, tb] : undefined;
      }
      const sub = this.answers.filter((a) => a.emb);
      if (sub.length >= 4) {
        const both = fitAnswers(sub, this.norm, true);
        const hand = fitAnswers(sub, this.norm, false);
        this.embReport = { n: sub.length, without: hand.agreeFit, with: both.agreeFit };
        if (sub.length >= MIN_ANSWERS_TO_APPLY) {
          GROUPS.forEach((g, i) => (w[g] = Math.max(0.02, both.weights[i])));
          we = both.weights[GROUPS.length];
        }
      }
    }
    if (!this.embReport && this.fit && this.fit.n >= MIN_ANSWERS_TO_APPLY) GROUPS.forEach((g, i) => (w[g] = Math.max(0.02, this.fit!.weights[i])));
    if (JSON.stringify(w) !== JSON.stringify(this.weights) || we !== this.embWeight) {
      this.weights = w;
      this.embWeight = we;
      this.tableKey = '';
      this.metricVersion++;
    }
  }

  // ---------------------------------------------------------- embedding

  /** Turn the perceptual embedding on (loads the model) or off. Resolves false when it can't load. */
  async setEmbedding(on: boolean): Promise<boolean> {
    this.embOn = on;
    if (on && this.embedder && !(await this.embedder.load())) {
      this.embOn = false;
      return false;
    }
    this.refitWeights();
    return true;
  }

  /** The embedding term between two members (NaN unless both are embedded and the embedding is on). */
  embTerm(a: string, b: string): number {
    return this.embOn ? this.emb.term(a, b) : NaN;
  }

  private blend(): { w: number; term: (a: string, b: string) => number } | undefined {
    return this.embOn && this.embWeight > 0 && this.emb.size > 2 ? { w: this.embWeight, term: (a, b) => this.emb.term(a, b) } : undefined;
  }

  /** Members still without an embedding (visible first). */
  missingEmbeddings(): Member[] {
    return this.pop.list().filter((m) => !m.hidden && !this.emb.get(m.id)).concat(this.pop.list().filter((m) => m.hidden && !this.emb.get(m.id)));
  }

  /** Embed one member: a strip of the drop, three frames through DINOv2, pooled. */
  async embedOne(m: Member): Promise<boolean> {
    if (!this.fper || !this.embedder || this.embedder.status.state !== 'ready') return false;
    const frames = await this.fper.strip(m.genome, 12);
    if (frames.length < 12) return false;
    const v = await this.embedder.embed([frames[0], frames[5], frames[11]]);
    if (!v) return false;
    this.emb.set(m.id, v);
    this.embAdded++;
    // Throttled save (a debounce would never fire while embedding runs back to back).
    if (!this.embTimer) {
      this.embTimer = setTimeout(() => {
        this.embTimer = 0;
        void this.store?.set(EMB_KEY, this.emb.toJSON());
      }, 2000) as unknown as number;
    }
    // Re-lay out and refit every few embeddings, not on each one.
    if (this.embAdded % 8 === 0 || !this.missingEmbeddings().length) {
      this.tableKey = '';
      this.refitWeights();
      this.metricVersion++;
    }
    return true;
  }

  /** Every current member's fingerprint is in the archive (migration and imports). */
  syncArchive(): void {
    const before = this.archive.version;
    const known = new Set(this.archive.entries.map((e) => e.id));
    for (const m of this.pop.list()) if (validFingerprint(m.fp) && m.fpv === FP_VERSION && !known.has(m.id)) this.archive.add(m.id, m.fp);
    if (this.archive.version !== before) {
      this.refit(true);
      this.saveArchive();
    }
  }

  /** Write the archive now (tests; the app saves debounced). */
  async flush(): Promise<void> {
    clearTimeout(this.saveTimer);
    await this.store?.set(ARCHIVE_KEY, this.archive.toJSON());
  }

  private saveArchive(): void {
    if (!this.store) return;
    clearTimeout(this.saveTimer);
    this.saveTimer = setTimeout(() => void this.store?.set(ARCHIVE_KEY, this.archive.toJSON()), 1500) as unknown as number;
  }

  /** Every valid fingerprint the normalisation is fitted on: the archive (every look seen) plus current members. */
  protected corpus(): number[][] {
    const inArchive = new Set(this.archive.entries.map((e) => e.id));
    return [...this.archive.entries.map((e) => e.fp), ...this.pop.list().filter((m) => !inArchive.has(m.id)).map((m) => m.fp).filter(validFingerprint)];
  }

  // ------------------------------------------------------------ novelty

  private ensureTable(): void {
    const key = `${this.archive.version}:${this.fitN}:${this.pop.size}:${JSON.stringify(this.weights)}:${this.embOn ? this.embWeight : 0}`;
    if (key === this.tableKey) return;
    this.tableKey = key;
    const t = noveltyTable(this.pop.list(), this.archive, this.norm, this.weights, undefined, this.blend());
    this.table = t.table;
    this.typical = t.typical;
  }

  /** A member's novelty (mean distance to its k nearest archived looks) and its relative value 0..1; null without a fingerprint. */
  novelty(m: Member): { nov: number; rel: number } | null {
    if (!validFingerprint(m.fp)) return null;
    this.ensureTable();
    return this.table.get(m.id) ?? null;
  }

  /** Novelty of a candidate fingerprint (not yet a member). */
  noveltyOf(fp: number[]): { nov: number; rel: number } {
    this.ensureTable();
    const az = this.archive.entries.map((e) => ({ id: e.id, z: this.z(e.fp) }));
    const nov = knnNovelty(this.z(fp), az, undefined, undefined, this.weights);
    return { nov, rel: relNovelty(nov, this.typical) };
  }

  /** Exploration score: fitness + mode weight x relative novelty, the weight tapering with votes. Unfingerprinted members count as averagely novel. */
  score(m: Member): number {
    return exploreScore(fitness(m), this.novelty(m)?.rel ?? 0.5, this.mode, voteCount(m));
  }

  /** The novelty bonus alone (for parent selection). */
  bonus(m: Member): number {
    return this.score(m) - fitness(m);
  }

  /** In explore / wild mode, a child less novel than the mode's floor is turned away. */
  acceptNovelty(fp: number[]): { ok: boolean; rel: number } {
    const floor = EXPLORE_ACCEPT[this.mode];
    if (!floor || this.archive.size < 8) return { ok: true, rel: floor ? this.noveltyOf(fp).rel : 0 };
    const { rel } = this.noveltyOf(fp);
    return { ok: rel >= floor, rel };
  }

  /** Refit the robust z-score normalisation when the corpus has grown by 10% (or on force). */
  refit(force = false): void {
    const c = this.corpus();
    if (!force && c.length < Math.max(3, this.fitN * 1.1)) return;
    this.norm = FeatureNorm.fit(c);
    this.fitN = c.length;
    this.zCache = new WeakMap();
    if (this.answers.length) this.refitWeights();
  }

  z(fp: number[]): Float32Array {
    let z = this.zCache.get(fp);
    if (!z) {
      z = this.norm.z(fp);
      this.zCache.set(fp, z);
    }
    return z;
  }

  distance(a: number[], b: number[]): number {
    return zDistance(this.z(a), this.z(b), this.weights);
  }

  /** Distance between two members: the fingerprint distance, blended with the embedding when it is on. */
  memberDistance(a: Member, b: Member): number {
    if (!validFingerprint(a.fp) || !validFingerprint(b.fp)) return NaN;
    const bl = this.blend();
    return zDistance(this.z(a.fp), this.z(b.fp), this.weights, bl ? { w: bl.w, t: bl.term(a.id, b.id) } : undefined);
  }

  /** Nearest member whose fingerprint is within the duplicate distance, or null. */
  duplicateOf(fp: number[], extra: { id: string; fp?: number[] }[] = []): DuplicateHit | null {
    let best: DuplicateHit | null = null;
    for (const m of [...this.pop.list(), ...extra]) {
      if (!m.fp || !validFingerprint(m.fp)) continue;
      const d = this.distance(fp, m.fp);
      if (d < DUP_FP_DIST && (!best || d < best.dist)) best = { id: m.id, dist: d };
    }
    return best;
  }

  async fingerprint(g: Genome): Promise<number[] | null> {
    if (!this.fper) return null;
    return (await this.fper.fingerprint(g)).fp;
  }

  /** Members still without a current fingerprint (visible first, then newest). */
  missing(): Member[] {
    return this.pop
      .list()
      .filter((m) => !(m.fpv === FP_VERSION && validFingerprint(m.fp)) && !this.failed.has(m.id))
      .sort((a, b) => Number(a.hidden) - Number(b.hidden) || b.views - a.views || b.created - a.created);
  }

  /**
   * Background work: every `everyMs`, when `idle()` says the offscreen runner
   * is free, fingerprint one member that has none.
   */
  startIdle(idle: () => boolean, everyMs = 1200): void {
    window.clearInterval(this.timer);
    this.refit(true);
    this.timer = window.setInterval(() => {
      if (this.running || document.hidden || !idle()) return;
      this.syncArchive();
      const m = this.missing()[0];
      if (!m) {
        // Fingerprints done: embeddings next (when turned on and loaded).
        const e = this.embOn && this.embedder?.status.state === 'ready' ? this.missingEmbeddings().find((x) => !this.embFailed.has(x.id)) : undefined;
        if (!e) return;
        this.running = true;
        void this.embedOne(e)
          .then((ok) => {
            if (!ok) this.embFailed.add(e.id);
          })
          .catch(() => this.embFailed.add(e.id))
          .finally(() => (this.running = false));
        return;
      }
      this.running = true;
      const genome = m.genome;
      void this.fingerprint(genome)
        .then((fp) => {
          // The member may have been culled or re-encoded meanwhile.
          if (this.pop.get(m.id) !== m || m.genome !== genome) return;
          if (!fp) {
            this.failed.add(m.id);
            return;
          }
          this.adopt(m, fp);
        })
        .finally(() => (this.running = false));
    }, everyMs);
  }

  stopIdle(): void {
    window.clearInterval(this.timer);
  }

  adopt(m: Member, fp: number[]): void {
    m.fp = fp;
    m.fpv = FP_VERSION;
    this.archive.add(m.id, fp);
    this.saveArchive();
    this.refit();
    this.onFingerprint?.(m);
  }

  get busy(): boolean {
    return this.running;
  }
}
