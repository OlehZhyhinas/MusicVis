// Force-directed layout for the preset map, in the style of a network graph
// (Obsidian-like): every node is linked to its k nearest neighbours by
// phenotype distance (symmetrised) with a short spring whose pull grows with
// similarity, so look-alikes collapse into tight clusters; a weak short-range
// repulsion and disc collision keep nodes readable and open gaps between
// clusters; communities found on the mutual links pull together and push
// other communities away harder, which opens visible gaps between clusters; a
// gentle pull to the centre keeps clusters from drifting off.
// Incremental: new nodes start next to their nearest look and the layout
// re-heats a little, so a bred child appears next to what it looks like and
// the map re-settles. Pure logic (no DOM) so it runs in the Node tests.

export interface LayoutNode {
  id: string;
  x: number;
  y: number;
  vx: number;
  vy: number;
  /** Pinned while dragged. */
  fixed?: boolean;
}

export interface LayoutEdge {
  a: number;
  b: number;
  rest: number;
  k: number;
  /** Each end is among the other's k nearest. */
  mutual?: boolean;
}

export const LAYOUT = {
  knn: 6,
  /** Spring rest length: about one disc diameter, only mildly longer for less similar pairs. */
  restBase: 30,
  restPerUnit: 6,
  spring: 0.12,
  /** Pull multiplier (dRef / d)^2 is clamped to this range (dRef: median neighbour distance). */
  simMin: 0.02,
  simMax: 4,
  /** Mutual neighbours (each in the other's k nearest) pull this much harder; one-way links this much softer. */
  mutual: 2,
  oneWay: 0.05,
  repulse: 2500,
  repulseCut: 240,
  /** Pull toward the centre of the node's community (label propagation on the spring graph). */
  cohesion: 0.06,
  /** Repulsion between nodes of different communities is this much stronger (opens gaps between clusters). */
  apart: 6,
  /** Discs never overlap: minimum centre distance. */
  collide: 30,
  gravity: 0.003,
  damping: 0.8,
  alphaDecay: 0.99,
  alphaMin: 0.004,
  maxStep: 30,
};

/**
 * Communities by weighted label propagation over the springs (deterministic:
 * fixed visiting order, ties to the smaller label). Strong, mutual links end
 * up sharing a label; weak one-way links rarely carry one across.
 */
export function communities(n: number, edges: LayoutEdge[]): Int32Array {
  const label = Int32Array.from({ length: n }, (_, i) => i);
  const adj: [number, number][][] = Array.from({ length: n }, () => []);
  for (const e of edges) {
    adj[e.a].push([e.b, e.k]);
    adj[e.b].push([e.a, e.k]);
  }
  for (let it = 0; it < 30; it++) {
    let changed = 0;
    for (let i = 0; i < n; i++) {
      if (!adj[i].length) continue;
      const score = new Map<number, number>();
      for (const [j, w] of adj[i]) score.set(label[j], (score.get(label[j]) ?? 0) + w);
      let best = label[i], bs = score.get(label[i]) ?? 0;
      for (const [l, v] of score) if (v > bs + 1e-12 || (Math.abs(v - bs) <= 1e-12 && l < best)) {
        best = l;
        bs = v;
      }
      if (best !== label[i]) {
        label[i] = best;
        changed++;
      }
    }
    if (!changed) break;
  }
  return label;
}

/** Deterministic pseudo-random position from an id (stable first layouts). */
export function hashPos(id: string, spread = 300): [number, number] {
  let h = 2166136261;
  for (let i = 0; i < id.length; i++) h = Math.imul(h ^ id.charCodeAt(i), 16777619);
  const a = ((h >>> 0) / 2 ** 32) * Math.PI * 2;
  const r = spread * Math.sqrt((((h >>> 16) & 0xffff) / 0xffff) * 0.9 + 0.1);
  return [Math.cos(a) * r, Math.sin(a) * r];
}

export class SpringLayout {
  nodes: LayoutNode[] = [];
  edges: LayoutEdge[] = [];
  /** Community of each node (label propagation over the springs, weighted by pull). */
  community: Int32Array = new Int32Array(0);
  alpha = 1;
  private index = new Map<string, number>();

  get settled(): boolean {
    return this.alpha < LAYOUT.alphaMin;
  }

  indexOf(id: string): number {
    return this.index.get(id) ?? -1;
  }

  get(id: string): LayoutNode | undefined {
    const i = this.index.get(id);
    return i === undefined ? undefined : this.nodes[i];
  }

  /**
   * Replace the node set and the edges. `dist(i, j)` is the phenotype distance
   * between ids[i] and ids[j] (NaN when unknown: such nodes link to `fallback`
   * ids instead, e.g. their parents). Existing nodes keep their positions;
   * new ones start at the mean of their neighbours (plus a little jitter).
   */
  setGraph(ids: string[], dist: (i: number, j: number) => number, fallback: (i: number) => number[] = () => []): { added: string[]; removed: string[] } {
    const old = new Map(this.nodes.map((n) => [n.id, n]));
    const keep = new Set(ids);
    const removed = this.nodes.filter((n) => !keep.has(n.id)).map((n) => n.id);
    const n = ids.length;
    // k nearest neighbours (symmetrised).
    const pairs = new Map<string, LayoutEdge>();
    const neigh: number[][] = ids.map(() => []);
    const nearestOf = new Int32Array(n).fill(-1);
    for (let i = 0; i < n; i++) {
      const ds: [number, number][] = [];
      for (let j = 0; j < n; j++) {
        if (j === i) continue;
        const d = dist(i, j);
        if (Number.isFinite(d)) ds.push([d, j]);
      }
      let links: [number, number][];
      if (ds.length) {
        ds.sort((a, b) => a[0] - b[0]);
        links = ds.slice(0, LAYOUT.knn);
      } else {
        links = fallback(i).filter((j) => j >= 0 && j !== i).map((j) => [1.5, j] as [number, number]);
      }
      if (links.length) nearestOf[i] = links[0][1];
      for (const [d, j] of links) {
        const key = i < j ? `${i}:${j}` : `${j}:${i}`;
        const e = pairs.get(key);
        if (e) e.mutual = true;
        else {
          pairs.set(key, { a: Math.min(i, j), b: Math.max(i, j), rest: LAYOUT.restBase + LAYOUT.restPerUnit * Math.min(d, 4), k: d, mutual: false });
          neigh[i].push(j);
          neigh[j].push(i);
        }
      }
    }
    // Pull by similarity: (dRef / d)^2, dRef the median neighbour distance (k held d until now).
    const ds = [...pairs.values()].map((e) => e.k).sort((x, y) => x - y);
    const dRef = ds.length ? Math.max(1e-3, ds[ds.length >> 1]) : 1;
    for (const e of pairs.values()) {
      e.k = LAYOUT.spring * Math.max(LAYOUT.simMin, Math.min(LAYOUT.simMax, (dRef / Math.max(1e-3, e.k)) ** 2)) * (e.mutual ? LAYOUT.mutual : LAYOUT.oneWay);
    }
    const added: string[] = [];
    const nodes: LayoutNode[] = ids.map((id) => {
      const o = old.get(id);
      if (o) return o;
      added.push(id);
      return { id, x: NaN, y: NaN, vx: 0, vy: 0 };
    });
    // Place new nodes: the mean of placed neighbours, else a stable hash position.
    for (let pass = 0; pass < 3; pass++) {
      nodes.forEach((nd, i) => {
        if (Number.isFinite(nd.x)) return;
        const near = nearestOf[i] >= 0 ? nodes[nearestOf[i]] : undefined;
        const placed = neigh[i].map((j) => nodes[j]).filter((m) => Number.isFinite(m.x));
        if (near && Number.isFinite(near.x)) {
          // Right next to the look it is closest to.
          const [jx, jy] = hashPos(nd.id, 1);
          const len = Math.hypot(jx, jy) || 1;
          nd.x = near.x + (jx / len) * LAYOUT.collide;
          nd.y = near.y + (jy / len) * LAYOUT.collide;
        } else if (placed.length || pass === 2) {
          if (placed.length) {
            const [jx, jy] = hashPos(nd.id, 18);
            nd.x = placed.reduce((s, m) => s + m.x, 0) / placed.length + jx;
            nd.y = placed.reduce((s, m) => s + m.y, 0) / placed.length + jy;
          } else [nd.x, nd.y] = hashPos(nd.id, 60 + 22 * Math.sqrt(n));
        }
      });
    }
    this.nodes = nodes;
    this.edges = [...pairs.values()];
    this.index = new Map(ids.map((id, i) => [id, i]));
    // Communities from mutual links only (one-way links would chain everything into one blob).
    this.community = communities(n, this.edges.filter((e) => e.mutual));
    // Re-heat: a lot for a fresh graph, a little for an incremental change.
    const fresh = old.size === 0 || added.length > n * 0.5;
    if (fresh) this.alpha = 1;
    else if (added.length || removed.length) this.alpha = Math.max(this.alpha, 0.25);
    return { added, removed };
  }

  /** One integration step. Returns the mean speed (px / step). */
  step(): number {
    const ns = this.nodes;
    const n = ns.length;
    if (!n) return 0;
    const a = this.alpha;
    const fx = new Float64Array(n), fy = new Float64Array(n);
    // Springs.
    for (const e of this.edges) {
      const p = ns[e.a], q = ns[e.b];
      const dx = q.x - p.x, dy = q.y - p.y;
      const d = Math.hypot(dx, dy) || 0.01;
      const f = (e.k * (d - e.rest)) / d;
      fx[e.a] += f * dx;
      fy[e.a] += f * dy;
      fx[e.b] -= f * dx;
      fy[e.b] -= f * dy;
    }
    // Short-range repulsion (grid buckets, so it stays near O(n)).
    const cut = LAYOUT.repulseCut, cut2 = cut * cut;
    const grid = this.buckets(cut);
    const com = this.community;
    this.eachNear(grid, cut, (i, j, dx, dy, d2) => {
      if (d2 > cut2) return;
      const f = (LAYOUT.repulse / d2) * (com[i] !== com[j] ? LAYOUT.apart : 1);
      const d = Math.sqrt(d2);
      fx[i] -= (f * dx) / d;
      fy[i] -= (f * dy) / d;
      fx[j] += (f * dx) / d;
      fy[j] += (f * dy) / d;
    });
    // Cohesion: each node is pulled toward its community's centre.
    if (LAYOUT.cohesion > 0 && com.length === n) {
      const cx = new Map<number, [number, number, number]>();
      for (let i = 0; i < n; i++) {
        const c = cx.get(com[i]) ?? [0, 0, 0];
        c[0] += ns[i].x;
        c[1] += ns[i].y;
        c[2]++;
        cx.set(com[i], c);
      }
      for (let i = 0; i < n; i++) {
        const c = cx.get(com[i])!;
        if (c[2] < 2) continue;
        fx[i] += LAYOUT.cohesion * (c[0] / c[2] - ns[i].x);
        fy[i] += LAYOUT.cohesion * (c[1] / c[2] - ns[i].y);
      }
    }
    let speed = 0;
    for (let i = 0; i < n; i++) {
      const nd = ns[i];
      if (nd.fixed) {
        nd.vx = nd.vy = 0;
        continue;
      }
      fx[i] -= LAYOUT.gravity * nd.x;
      fy[i] -= LAYOUT.gravity * nd.y;
      nd.vx = (nd.vx + fx[i] * a) * LAYOUT.damping;
      nd.vy = (nd.vy + fy[i] * a) * LAYOUT.damping;
      const v = Math.hypot(nd.vx, nd.vy);
      if (v > LAYOUT.maxStep) {
        nd.vx *= LAYOUT.maxStep / v;
        nd.vy *= LAYOUT.maxStep / v;
      }
      nd.x += nd.vx;
      nd.y += nd.vy;
      speed += Math.min(v, LAYOUT.maxStep);
    }
    // Collision: overlapping discs are pushed apart (half the overlap each).
    const c = LAYOUT.collide, c2 = c * c;
    this.eachNear(this.buckets(c), c, (i, j, dx, dy, d2) => {
      if (d2 >= c2) return;
      const d = Math.sqrt(d2);
      const push = (c - d) / 2 / d;
      const pi = ns[i].fixed ? 0 : ns[j].fixed ? 2 : 1, pj = ns[j].fixed ? 0 : ns[i].fixed ? 2 : 1;
      ns[i].x -= dx * push * pi;
      ns[i].y -= dy * push * pi;
      ns[j].x += dx * push * pj;
      ns[j].y += dy * push * pj;
    });
    this.alpha *= LAYOUT.alphaDecay;
    return speed / n;
  }

  private buckets(cell: number): Map<string, number[]> {
    const grid = new Map<string, number[]>();
    this.nodes.forEach((nd, i) => {
      const key = `${Math.floor(nd.x / cell)},${Math.floor(nd.y / cell)}`;
      const b = grid.get(key);
      if (b) b.push(i);
      else grid.set(key, [i]);
    });
    return grid;
  }

  /** Every pair of nodes in neighbouring cells, once: fn(i, j, dx, dy, d2) with d = j - i (coincident nodes are nudged apart deterministically). */
  private eachNear(grid: Map<string, number[]>, cell: number, fn: (i: number, j: number, dx: number, dy: number, d2: number) => void): void {
    const ns = this.nodes;
    for (let i = 0; i < ns.length; i++) {
      const gx = Math.floor(ns[i].x / cell), gy = Math.floor(ns[i].y / cell);
      for (let ox = -1; ox <= 1; ox++) for (let oy = -1; oy <= 1; oy++) {
        const b = grid.get(`${gx + ox},${gy + oy}`);
        if (!b) continue;
        for (const j of b) {
          if (j <= i) continue;
          let dx = ns[j].x - ns[i].x, dy = ns[j].y - ns[i].y;
          let d2 = dx * dx + dy * dy;
          if (d2 < 0.01) {
            dx = ((i * 7919 + j) % 13) - 6 || 1;
            dy = ((j * 104729 + i) % 11) - 5 || 1;
            d2 = dx * dx + dy * dy;
          }
          fn(i, j, dx, dy, d2);
        }
      }
    }
  }

  /** Run up to `steps` steps now (a fresh graph is pre-settled before it is first drawn). */
  settle(steps: number, maxMs = Infinity): void {
    const t0 = performance.now();
    for (let i = 0; i < steps && !this.settled && performance.now() - t0 < maxMs; i++) this.step();
  }

  /** Nudge the layout awake (a node was dragged). */
  reheat(alpha = 0.2): void {
    this.alpha = Math.max(this.alpha, alpha);
  }

  bounds(): { x0: number; y0: number; x1: number; y1: number } {
    let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
    for (const nd of this.nodes) {
      x0 = Math.min(x0, nd.x);
      y0 = Math.min(y0, nd.y);
      x1 = Math.max(x1, nd.x);
      y1 = Math.max(y1, nd.y);
    }
    return Number.isFinite(x0) ? { x0, y0, x1, y1 } : { x0: -100, y0: -100, x1: 100, y1: 100 };
  }
}
