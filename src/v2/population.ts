// Population: members, lineage, votes, fitness, selection and culling.
// Pure logic (no DOM, no GL) so it runs in the Node tests.

import {
  classify, cloneGenome, energyOf, genomeHash, repair, structuralKey, validate,
  type Energy, type Genome, type Species,
} from './genome';
import { nameFor } from './naming';
import { SEEDS, SEED_VERSION } from './seeds';
import type { CrossTag, Rng } from './ops';

const CROSS_TAGS: CrossTag[] = ['fused', 'morph', 'merged', 'layered'];

export interface Member {
  id: string; // "G0-E07" for seeds, "G{gen}-{nnnn}" for children
  gen: number;
  origin?: string; // V1 preset id for seeds
  parents: string[];
  created: number; // ms since epoch
  name: string;
  genome: Genome;
  species: Species;
  species2: Species | null;
  type: string; // display label, e.g. "vortex × flame"
  energy: Energy;
  likes: number;
  dislikes: number;
  softDislikes: number; // skipped within 5 s
  weakLikes: number; // watched more than 60 s
  views: number;
  watch: number; // seconds watched in total
  hidden: boolean;
  /** Screening render descriptor: mean rgb, motion, mirror symmetry, radial symmetry, detail, coverage. */
  descriptor?: number[];
  /** How a crossover child combined its parents (absent for seeds, mutants and older children). */
  cross?: CrossTag;
}

/**
 * File / storage format. 1: genomes of format 1 (no draw chains or merges);
 * 2: genomes of format 2 and the per-child crossover tag; 3: descriptive,
 * inherited names (fromJSON renames every bred child of a version 1-2 file, in
 * generation order, keeping ids / votes / parents / genomes); 4: genomes of
 * format 3 (bodies made of sub-genes). Older genomes are converted on load by
 * repair() (legacy.ts), keeping each member's id, votes, parents and name; a
 * converted child's screening descriptor is dropped so it is measured again.
 */
export const POPULATION_VERSION = 4;

export interface PopulationData {
  format: 'musicvis-v2-population';
  version: 1 | 2 | 3 | 4;
  /** Seed encoding the G0 genomes come from (missing in files from before versioning: 1). */
  seedVersion?: number;
  counter: number;
  votesSinceBreed: number;
  members: Member[];
}

export const POP_CAP = 150;
export const BREED_EVERY = 5; // votes between automatic breeding rounds

// ---------------------------------------------------------------- names

const ROMAN: [number, string][] = [
  [1000, 'M'], [900, 'CM'], [500, 'D'], [400, 'CD'], [100, 'C'], [90, 'XC'], [50, 'L'], [40, 'XL'],
  [10, 'X'], [9, 'IX'], [5, 'V'], [4, 'IV'], [1, 'I'],
];
function toRoman(n: number): string {
  let s = '';
  for (const [v, sym] of ROMAN) while (n >= v) { s += sym; n -= v; }
  return s;
}

/** `base`, or `base II`, `base III`, ... the first form not already in `used`. */
export function uniqueName(used: ReadonlySet<string>, base: string): string {
  if (!used.has(base)) return base;
  let i = 2;
  while (used.has(`${base} ${toRoman(i)}`)) i++;
  return `${base} ${toRoman(i)}`;
}

// -------------------------------------------------------------- fitness

const Z = 1.28; // ~80% one-sided confidence

export function wilson(pos: number, n: number): number {
  if (n <= 0) return 0;
  const p = pos / n;
  const z2 = Z * Z;
  return (p + z2 / (2 * n) - Z * Math.sqrt((p * (1 - p)) / n + z2 / (4 * n * n))) / (1 + z2 / n);
}

/** Wilson lower bound of liking, with implicit signals as partial votes and a 1:1 prior. */
export function fitness(m: Member): number {
  const pos = m.likes + 0.3 * m.weakLikes + 1;
  const n = m.likes + m.dislikes + 0.3 * (m.weakLikes + m.softDislikes) + 2;
  return wilson(pos, n);
}

export function descriptorDistance(a?: number[], b?: number[]): number {
  if (!a || !b || a.length !== b.length) return 1;
  let s = 0;
  for (let i = 0; i < a.length; i++) s += (a[i] - b[i]) ** 2;
  return Math.sqrt(s / a.length);
}

const DUP_DIST = 0.035;

// ----------------------------------------------------------- population

function describe(g: Genome): Pick<Member, 'species' | 'species2' | 'type' | 'energy'> {
  const c = classify(g);
  return { species: c.primary, species2: c.secondary, type: c.label, energy: energyOf(g) };
}

function seedMember(s: (typeof SEEDS)[number], now: number): Member {
  const g = cloneGenome(s.genome);
  return {
    id: `G0-${s.origin}`, gen: 0, origin: s.origin, parents: [], created: now, name: s.name, genome: g,
    ...describe(g), likes: 0, dislikes: 0, softDislikes: 0, weakLikes: 0, views: 0, watch: 0, hidden: false,
  };
}

export class Population {
  members = new Map<string, Member>();
  counter = 0;
  votesSinceBreed = 0;
  seedVersion = SEED_VERSION;

  static seeded(now = Date.now()): Population {
    const p = new Population();
    for (const s of SEEDS) {
      const m = seedMember(s, now);
      p.members.set(m.id, m);
    }
    return p;
  }

  /**
   * Seed migration. When the population's seed version is older than the
   * code's (or a seed genome differs from the current encoding), every G0 seed
   * gets the current genome while keeping its id, votes, views, watch time and
   * hidden flag; missing seeds come back. Bred children (G1+) are never touched,
   * so their parent links stay valid. A population from a newer seed version is
   * left alone. Returns the ids of the seeds that changed.
   */
  upgradeSeeds(now = Date.now()): string[] {
    if (this.seedVersion > SEED_VERSION) return [];
    const older = this.seedVersion < SEED_VERSION;
    const changed: string[] = [];
    for (const s of SEEDS) {
      const fresh = seedMember(s, now);
      const cur = this.members.get(fresh.id);
      if (!cur) {
        this.members.set(fresh.id, fresh);
        changed.push(fresh.id);
      } else if (older || JSON.stringify(cur.genome) !== JSON.stringify(fresh.genome)) {
        Object.assign(cur, {
          gen: 0, origin: s.origin, parents: [], genome: fresh.genome, name: s.name,
          species: fresh.species, species2: fresh.species2, type: fresh.type, energy: fresh.energy, descriptor: undefined,
        });
        changed.push(cur.id);
      }
    }
    this.seedVersion = SEED_VERSION;
    return changed;
  }

  get size(): number {
    return this.members.size;
  }

  list(): Member[] {
    return [...this.members.values()];
  }

  visible(): Member[] {
    return this.list().filter((m) => !m.hidden);
  }

  get(id: string): Member | undefined {
    return this.members.get(id);
  }

  /** Add a bred child. generation = max(parent gen) + 1; id counter is per population. */
  addChild(genome: Genome, parents: Member[], now = Date.now(), cross?: CrossTag): Member {
    const g = repair(genome);
    const gen = parents.length ? Math.max(...parents.map((p) => p.gen)) + 1 : 1;
    const nnnn = String(++this.counter).padStart(4, '0');
    const base = nameFor(g, parents.map((p) => p.name));
    const used = new Set(this.list().map((x) => x.name));
    const m: Member = {
      id: `G${gen}-${nnnn}`, gen, parents: parents.map((p) => p.id), created: now, name: uniqueName(used, base), genome: g,
      ...describe(g), likes: 0, dislikes: 0, softDislikes: 0, weakLikes: 0, views: 0, watch: 0, hidden: false,
    };
    if (cross) m.cross = cross;
    this.members.set(m.id, m);
    return m;
  }

  /** True when an identical genome (or same structure with near-identical params) is already present. */
  hasDuplicate(g: Genome): boolean {
    const h = genomeHash(g);
    for (const m of this.members.values()) if (genomeHash(m.genome) === h) return true;
    return false;
  }

  vote(id: string, like: boolean): void {
    const m = this.members.get(id);
    if (!m) return;
    if (like) m.likes++;
    else m.dislikes++;
    this.votesSinceBreed++;
  }

  /** Called when a preset stops showing. skipped: the user moved on manually. */
  recordView(id: string, seconds: number, skipped: boolean): void {
    const m = this.members.get(id);
    if (!m) return;
    m.watch += seconds;
    if (skipped && seconds < 5) m.softDislikes++;
    if (seconds > 60) m.weakLikes++;
  }

  /** Nearest-neighbour descriptor distance (novelty); 1 when unknown. */
  novelty(m: Member): number {
    let best = 1;
    for (const o of this.members.values()) {
      if (o === m || o.hidden) continue;
      best = Math.min(best, descriptorDistance(m.descriptor, o.descriptor));
    }
    return best;
  }

  /**
   * Tournament selection within a niche. The second parent is often of a
   * different type (cross-type mating) and sometimes from the other niche;
   * near-duplicates of the first parent are penalised.
   */
  pickParents(niche: Energy, rng: Rng, tournament = 3): [Member, Member] | null {
    const pool = this.visible();
    if (pool.length < 2) return null;
    const inNiche = pool.filter((m) => m.energy === niche);
    const base = inNiche.length >= 2 ? inNiche : pool;
    const score = (m: Member, ref?: Member) => {
      let s = fitness(m) + 0.04 * Math.min(1, this.novelty(m) / 0.2);
      if (ref && descriptorDistance(m.descriptor, ref.descriptor) < DUP_DIST * 2) s -= 0.15;
      if (ref && structuralKey(m.genome) === structuralKey(ref.genome)) s -= 0.05;
      return s;
    };
    const tour = (cands: Member[], ref?: Member) => {
      let best: Member | null = null;
      let bs = -Infinity;
      for (let i = 0; i < tournament; i++) {
        const c = cands[Math.floor(rng() * cands.length)];
        const s = score(c, ref);
        if (s > bs) {
          bs = s;
          best = c;
        }
      }
      return best!;
    };
    const a = tour(base);
    const r = rng();
    let cands = base.filter((m) => m !== a);
    if (r < 0.15) cands = pool.filter((m) => m !== a && m.energy !== niche);
    else if (r < 0.55) cands = base.filter((m) => m !== a && m.species !== a.species);
    if (!cands.length) cands = pool.filter((m) => m !== a);
    return [a, tour(cands, a)];
  }

  /** Remove the weakest non-seed members until the population fits the cap. */
  cull(cap = POP_CAP, now = Date.now()): Member[] {
    const removed: Member[] = [];
    while (this.members.size > cap) {
      const cands = this.list().filter((m) => m.gen > 0);
      if (!cands.length) break;
      // Unwatched children younger than 10 minutes get a grace period.
      const graced = cands.filter((m) => !(m.views === 0 && now - m.created < 600_000));
      const pool = graced.length ? graced : cands;
      let worst: Member | null = null;
      let ws = Infinity;
      for (const m of pool) {
        let s = fitness(m) + (m.hidden ? -0.2 : 0);
        // A near-duplicate of a fitter member is the first to go.
        for (const o of this.members.values()) {
          if (o !== m && fitness(o) >= fitness(m) && descriptorDistance(m.descriptor, o.descriptor) < DUP_DIST) {
            s -= 0.1;
            break;
          }
        }
        if (s < ws) {
          ws = s;
          worst = m;
        }
      }
      if (!worst) break;
      this.members.delete(worst.id);
      removed.push(worst);
    }
    return removed;
  }

  // -------------------------------------------------------- serialization

  toJSON(): PopulationData {
    return {
      format: 'musicvis-v2-population',
      version: POPULATION_VERSION,
      seedVersion: this.seedVersion,
      counter: this.counter,
      votesSinceBreed: this.votesSinceBreed,
      members: this.list().map((m) => ({ ...m, genome: cloneGenome(m.genome) })),
    };
  }

  static fromJSON(data: unknown): Population {
    const d = data as Partial<PopulationData>;
    if (!d || d.format !== 'musicvis-v2-population' || !Array.isArray(d.members)) throw new Error('Not a MusicVis V2 population file.');
    if (typeof d.version === 'number' && d.version > POPULATION_VERSION) throw new Error('This population file was saved by a newer version of MusicVis.');
    const p = new Population();
    const n = (v: unknown, def = 0) => (typeof v === 'number' && Number.isFinite(v) && v >= 0 ? v : def);
    for (const raw of d.members) {
      if (!raw || typeof raw.id !== 'string' || !/^G\d+-[A-Z0-9]+$/.test(raw.id)) continue;
      // Structurally broken genomes are dropped; only parameter values get repaired.
      const rg = raw.genome as unknown as Record<string, unknown> | undefined;
      if (!rg || !Array.isArray(rg.chain) || !rg.carrier || !rg.color) continue;
      const old = rg.v === 1 || rg.v === 2;
      const parts = old ? rg.emitters : rg.bodies;
      if ((!old && rg.v !== 3) || !Array.isArray(parts) || !parts.length) continue;
      let g: Genome;
      try {
        g = repair(rg);
      } catch {
        continue;
      }
      if (validate(g).length) continue;
      const m: Member = {
        id: raw.id,
        gen: Math.floor(n(raw.gen)),
        origin: typeof raw.origin === 'string' ? raw.origin : undefined,
        parents: Array.isArray(raw.parents) ? raw.parents.filter((x) => typeof x === 'string') : [],
        created: n(raw.created, Date.now()),
        name: typeof raw.name === 'string' && raw.name ? raw.name.slice(0, 60) : nameFor(g, []),
        genome: g,
        ...describe(g),
        likes: Math.floor(n(raw.likes)),
        dislikes: Math.floor(n(raw.dislikes)),
        softDislikes: Math.floor(n(raw.softDislikes)),
        weakLikes: Math.floor(n(raw.weakLikes)),
        views: Math.floor(n(raw.views)),
        watch: n(raw.watch),
        hidden: !!raw.hidden,
        descriptor: !old && Array.isArray(raw.descriptor) && raw.descriptor.every((x) => typeof x === 'number') ? raw.descriptor : undefined,
      };
      if (CROSS_TAGS.includes(raw.cross as CrossTag)) m.cross = raw.cross;
      p.members.set(m.id, m);
    }
    if (!p.members.size) throw new Error('The file has no valid presets.');
    let maxN = 0;
    for (const id of p.members.keys()) {
      const mm = /^G\d+-(\d{4,})$/.exec(id);
      if (mm) maxN = Math.max(maxN, Number(mm[1]));
    }
    p.counter = Math.max(Math.floor(n(d.counter)), maxN);
    p.votesSinceBreed = Math.floor(n(d.votesSinceBreed));
    p.seedVersion = Math.max(1, Math.floor(n(d.seedVersion, 1)));

    // Version-3 migration: rename every bred child of a version 1-2 file with the
    // descriptive, inherited scheme (those files kept random adjective-noun
    // pairs). Seeds (gen 0) keep their name as loaded. Parents are renamed before
    // children (gen order) so inheritance sees each parent's *new* name. Ids and
    // votes are untouched. Version 3+ names are kept as they are.
    const fileVersion = typeof d.version === 'number' ? d.version : 1;
    if (fileVersion < 3) {
      const used = new Set(p.list().filter((m) => m.gen === 0).map((m) => m.name));
      const order = p.list().filter((m) => m.gen > 0).sort((a, b) => a.gen - b.gen || a.id.localeCompare(b.id));
      for (const m of order) {
        const parentNames = m.parents.map((id) => p.get(id)?.name).filter((x): x is string => !!x);
        const base = nameFor(m.genome, parentNames);
        m.name = uniqueName(used, base);
        used.add(m.name);
      }
    }
    return p;
  }
}
