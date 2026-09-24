// Evolution controller: owns the population, persists it, breeds and screens
// children, chooses what to show next, and tracks votes / views.

import { cloneGenome, energyOf, type Energy, type Genome } from './genome';
import { crossoverTagged, mulberry32, mutate, sameGenome, type CrossTag, type Rng } from './ops';
import { BREED_EVERY, POP_CAP, Population, fitness, type Member } from './population';
import type { ScreenResult, Screener } from './screen';
import type { Store } from './store';
import type { Phenotype } from './phenotype';

export type BreedMode = 'cross' | 'mutate';
export type ChooseReason = 'evolve' | 'new' | 'drop' | 'next';

export interface BreedEvent {
  kind: 'child' | 'reject' | 'done';
  member?: Member;
  reason?: string;
  tried: number;
}

const MAX_TRIES_PER_CHILD = 6;

export class Evolution {
  pop = Population.seeded();
  rng: Rng = mulberry32((Date.now() ^ (Math.random() * 1e9)) >>> 0);
  onChange: (() => void) | null = null;
  breeding = 0;
  lastRejects: string[] = [];
  /** Phenotype fingerprints (visual duplicate rejection, novelty); null without a GPU (tests). */
  pheno: Phenotype | null = null;
  private saveTimer = 0;
  private history: string[] = [];
  private view: { id: string; start: number } | null = null;

  constructor(private store: Store, readonly screener: Screener) {}

  async load(): Promise<void> {
    let data: unknown;
    try {
      data = await this.store.get<unknown>('population');
      if (data) {
        this.pop = Population.fromJSON(data);
        // Seeds re-encoded since this population was saved get the new genomes
        // (votes and children kept); their thumbnails are redrawn.
        const stale = this.pop.upgradeSeeds();
        if (stale.length) {
          void this.store.deleteThumbs(stale);
          void this.store.set('population', this.pop.toJSON());
        }
      }
    } catch (err) {
      console.warn('[v2] stored population unreadable, starting from the seeds (the old data is kept as population-unreadable)', err);
      // Keep the unreadable data so a later version (or an export by hand) can still recover it.
      if (data) void this.store.set('population-unreadable', data);
      this.pop = Population.seeded();
    }
  }

  changed(): void {
    this.onChange?.();
    window.clearTimeout(this.saveTimer);
    this.saveTimer = window.setTimeout(() => void this.store.set('population', this.pop.toJSON()), 400);
  }

  // ------------------------------------------------------------ votes

  vote(id: string, like: boolean, niche: Energy): void {
    this.pop.vote(id, like);
    this.changed();
    if (this.pop.votesSinceBreed >= BREED_EVERY && !this.breeding) {
      this.pop.votesSinceBreed = 0;
      void this.autoBreed(niche);
    }
  }

  /** A preset starts showing. The previous one's watch time is recorded. */
  startView(id: string, skippedPrev: boolean): void {
    this.endView(skippedPrev);
    const m = this.pop.get(id);
    if (m) m.views++;
    this.view = { id, start: performance.now() };
    this.history.push(id);
    if (this.history.length > 8) this.history.shift();
    this.changed();
  }

  endView(skipped: boolean): void {
    if (!this.view) return;
    const secs = (performance.now() - this.view.start) / 1000;
    this.pop.recordView(this.view.id, secs, skipped);
    this.view = null;
  }

  // ---------------------------------------------------------- choosing

  choose(reason: ChooseReason, songCx: number, currentId: string | null): Member | null {
    const pool = this.pop.visible().filter((m) => m.id !== currentId);
    if (!pool.length) return this.pop.visible()[0] ?? null;
    const dist = (m: Member) => {
      const [lo, hi] = m.genome.energy;
      const target = reason === 'drop' ? Math.max(0.55, Math.min(1, songCx + 0.3)) : songCx;
      return target < lo ? lo - target : target > hi ? target - hi : 0;
    };
    const recent = new Set(this.history);
    if (reason === 'evolve' && this.rng() < 0.3) {
      // Exploration: something nobody has seen yet, newest first.
      // With exploration on, the most novel-looking first.
      const nov = (m: Member) => (this.pheno && this.pheno.mode !== 'off' ? this.pheno.novelty(m)?.rel ?? 0.5 : 0);
      const fresh = pool.filter((m) => m.views === 0 && dist(m) < 0.25).sort((a, b) => nov(b) - nov(a) || b.created - a.created);
      if (fresh.length) return fresh[Math.floor(this.rng() * Math.min(4, fresh.length))];
    }
    let cands = pool.filter((m) => dist(m) < 0.1 && !recent.has(m.id) && (reason !== 'drop' || m.genome.energy[1] >= 0.8));
    if (cands.length < 6) cands = pool.filter((m) => dist(m) < 0.25 && !recent.has(m.id));
    if (!cands.length) cands = pool;
    // Score = fitness plus an exploration bonus for rarely seen presets (so
    // unvoted children get airtime until votes decide), with random
    // tie-breaking so the seeds at the front of the list don't win every tie.
    // Exploration mode adds phenotype novelty (tapering as votes come in).
    const totalViews = pool.reduce((s, m) => s + m.views, 0) + 1;
    const base = (m: Member) => (this.pheno ? this.pheno.score(m) : fitness(m));
    const score = (m: Member) => base(m) + 0.15 * Math.sqrt(Math.log(totalViews + 1) / (m.views + 1)) + 0.02 * this.rng();
    // Weighted pick among the best dozen.
    const ranked = cands.map((m) => ({ m, f: score(m) })).sort((a, b) => b.f - a.f).slice(0, reason === 'evolve' ? 10 : 14);
    let total = 0;
    for (const r of ranked) total += r.f * r.f + 0.01;
    let x = this.rng() * total;
    for (const r of ranked) {
      x -= r.f * r.f + 0.01;
      if (x <= 0) return r.m;
    }
    return ranked[0].m;
  }

  // ---------------------------------------------------------- breeding

  /**
   * Breed `n` screened children. cross: from two parents; mutate: from one.
   * Candidates failing screening are retried (up to 6 attempts per child).
   */
  async breed(parents: Member[], n: number, mode: BreedMode, onEvent?: (e: BreedEvent) => void): Promise<Member[]> {
    const out: Member[] = [];
    let tried = 0;
    this.breeding++;
    this.lastRejects = [];
    this.onChange?.();
    try {
      while (out.length < n && tried < n * MAX_TRIES_PER_CHILD) {
        tried++;
        let g: Genome;
        let tag: CrossTag | undefined;
        if (mode === 'cross' && parents.length >= 2) {
          // The fitter parent is likelier to supply the drawing; the other shapes it.
          const res = crossoverTagged(parents[0].genome, parents[1].genome, this.rng, (fitness(parents[0]) - fitness(parents[1])) * 3);
          g = res.genome;
          tag = res.tag;
          if (this.rng() < 0.35) {
            const n = g.bodies.length;
            g = mutate(g, this.rng, 0.4, undefined, true);
            if (g.bodies.length > n) tag = 'layered';
            else if (tag === 'merged' && !g.bodies.some((b) => b.fuse)) tag = 'fused';
          }
        } else {
          const n = parents[0].genome.bodies.length;
          g = mutate(parents[0].genome, this.rng, 1 + Math.min(2, tried * 0.12));
          if (g.bodies.length > n) tag = 'layered';
        }
        if (parents.some((p) => sameGenome(p.genome, g)) || this.pop.hasDuplicate(g) || out.some((c) => sameGenome(c.genome, g))) continue;
        const res: ScreenResult = await this.screener.screen(g);
        if (!res.ok) {
          this.lastRejects.push(res.reason ?? 'rejected');
          onEvent?.({ kind: 'reject', reason: res.reason, tried });
          continue;
        }
        // Visual duplicates: a child that looks like an existing member is rejected, whatever its genes say.
        const fp = this.pheno ? await this.pheno.fingerprint(g) : null;
        const dup = fp && this.pheno!.duplicateOf(fp);
        if (dup) {
          const reason = `looks like an existing preset (${dup.id}, distance ${dup.dist.toFixed(2)})`;
          this.lastRejects.push(reason);
          onEvent?.({ kind: 'reject', reason, tried });
          continue;
        }
        // Explore / wild: turn away children that look too familiar (never in the last half of the attempts).
        if (fp && tried <= (n * MAX_TRIES_PER_CHILD) / 2) {
          const acc = this.pheno!.acceptNovelty(fp);
          if (!acc.ok) {
            const reason = `too familiar for ${this.pheno!.mode} mode (novelty ${acc.rel.toFixed(2)})`;
            this.lastRejects.push(reason);
            onEvent?.({ kind: 'reject', reason, tried });
            continue;
          }
        }
        const child = this.pop.addChild(cloneGenome(g), parents, Date.now(), tag);
        child.descriptor = res.descriptor;
        if (fp) this.pheno!.adopt(child, fp);
        out.push(child);
        onEvent?.({ kind: 'child', member: child, tried });
        this.changed();
      }
      const removed = this.pop.cull(POP_CAP);
      if (removed.length) void this.store.deleteThumbs(removed.map((m) => m.id));
    } finally {
      this.breeding--;
      onEvent?.({ kind: 'done', tried });
      this.changed();
    }
    return out;
  }

  /** Background breeding after every few votes: tournament parents within the niche. */
  async autoBreed(niche: Energy): Promise<Member[]> {
    const born: Member[] = [];
    for (let i = 0; i < 2; i++) {
      const pair = this.pop.pickParents(niche, this.rng, 3, this.pheno ? (m) => this.pheno!.bonus(m) : undefined);
      if (!pair) break;
      born.push(...(await this.breed(pair, 1, this.rng() < 0.8 ? 'cross' : 'mutate')));
    }
    if (born.length) console.info(`[v2] auto-bred ${born.map((m) => m.id).join(', ')} in the ${niche} niche`);
    return born;
  }

  /** Screening descriptor for seeds and imported members that have none (used for novelty). */
  async describeMissing(limit = 4): Promise<void> {
    const todo = this.pop.list().filter((m) => !m.descriptor).slice(0, limit);
    for (const m of todo) {
      const r = await this.screener.screen(m.genome);
      m.descriptor = r.descriptor;
    }
    if (todo.length) this.changed();
  }

  // ------------------------------------------------------- thumbnails

  private thumbJobs = new Map<string, Promise<string>>();

  thumb(id: string): Promise<string> {
    const running = this.thumbJobs.get(id);
    if (running) return running;
    const job = (async () => {
      const cached = await this.store.getThumb(id);
      if (cached) return cached;
      const m = this.pop.get(id);
      if (!m) return '';
      const url = await this.screener.thumbnail(m.genome);
      if (url) await this.store.setThumb(id, url);
      return url;
    })();
    this.thumbJobs.set(id, job);
    void job.finally(() => this.thumbJobs.delete(id));
    return job;
  }

  // ---------------------------------------------------------- import / export

  exportJSON(): string {
    return JSON.stringify(this.pop.toJSON(), null, 1);
  }

  async importJSON(text: string): Promise<number> {
    const p = Population.fromJSON(JSON.parse(text));
    p.upgradeSeeds();
    this.pop = p;
    await this.store.clearThumbs();
    this.changed();
    return p.size;
  }

  async reset(): Promise<void> {
    this.pop = Population.seeded();
    this.history = [];
    this.view = null;
    await this.store.clearThumbs();
    this.changed();
  }

  nicheFor(songCx: number): Energy {
    return songCx >= 0.5 ? 'energetic' : 'calm';
  }

  energyOf(m: Member): Energy {
    return energyOf(m.genome);
  }
}
