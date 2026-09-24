// Phenotype controller: fingerprints members in the background (idle work,
// one at a time, only while nothing else is rendering offscreen), keeps the
// robust normalisation fitted to every fingerprint seen, and answers "does
// this child look like something we already have?" for breeding.

import type { Genome } from './genome';
import type { Member, Population } from './population';
import { DUP_FP_DIST, FP_VERSION, FeatureNorm, validFingerprint, zDistance, type GroupWeights, EQUAL_WEIGHTS } from './fingerprint';
import type { Fingerprinter } from './fingerprintRender';

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

  readonly fper: Fingerprinter | null;
  private popRef: () => Population;

  constructor(fper: Fingerprinter | null, popRef: () => Population) {
    this.fper = fper;
    this.popRef = popRef;
  }

  private get pop(): Population {
    return this.popRef();
  }

  /** Every valid fingerprint the normalisation is fitted on. */
  protected corpus(): number[][] {
    return this.pop.list().map((m) => m.fp).filter(validFingerprint);
  }

  /** Refit the robust z-score normalisation when the corpus has grown by 10% (or on force). */
  refit(force = false): void {
    const c = this.corpus();
    if (!force && c.length < Math.max(3, this.fitN * 1.1)) return;
    this.norm = FeatureNorm.fit(c);
    this.fitN = c.length;
    this.zCache = new WeakMap();
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
      const m = this.missing()[0];
      if (!m) return;
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
    this.refit();
    this.onFingerprint?.(m);
  }

  get busy(): boolean {
    return this.running;
  }
}
