// Force-directed layout for the preset map: every node is linked to its k
// nearest neighbours by phenotype distance with a spring whose rest length
// grows with that distance, all nodes repel each other at short range, and a
// weak pull keeps loose components near the middle. Incremental: new nodes
// start at the mean of their neighbours and the layout re-heats a little, so
// a bred child appears next to what it looks like and the map re-settles.
// Pure logic (no DOM) so it runs in the Node tests.

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
}

export const LAYOUT = {
  knn: 4,
  /** Extra weak long-range springs per node (to deterministic far picks), so clusters keep their mutual distances. */
  global: 3,
  globalK: 0.012,
  restBase: 36,
  restPerUnit: 70,
  spring: 0.06,
  repulse: 2600,
  repulseCut: 220,
  gravity: 0.0025,
  damping: 0.82,
  alphaDecay: 0.985,
  alphaMin: 0.004,
  maxStep: 30,
};

function hashUnit(s: string): number {
  let h = 2166136261;
  for (let i = 0; i < s.length; i++) h = Math.imul(h ^ s.charCodeAt(i), 16777619);
  return (h >>> 0) / 2 ** 32;
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
      for (const [d, j] of links) {
        const key = i < j ? `${i}:${j}` : `${j}:${i}`;
        if (!pairs.has(key)) pairs.set(key, { a: Math.min(i, j), b: Math.max(i, j), rest: LAYOUT.restBase + LAYOUT.restPerUnit * d, k: LAYOUT.spring });
        neigh[i].push(j);
        neigh[j].push(i);
      }
    }
    // A few weak long-range springs give the map its global shape (kNN alone only knows neighbourhoods).
    for (let i = 0; i < n && n > LAYOUT.knn + 2; i++) {
      for (let g = 0; g < LAYOUT.global; g++) {
        const j = (i + 1 + Math.floor(hashUnit(`${ids[i]}/${g}`) * (n - 1))) % n;
        const d = dist(i, j);
        if (j === i || !Number.isFinite(d)) continue;
        const key = i < j ? `${i}:${j}` : `${j}:${i}`;
        if (!pairs.has(key)) pairs.set(key, { a: Math.min(i, j), b: Math.max(i, j), rest: LAYOUT.restBase + LAYOUT.restPerUnit * d, k: LAYOUT.globalK });
      }
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
        const placed = neigh[i].map((j) => nodes[j]).filter((m) => Number.isFinite(m.x));
        if (placed.length || pass === 2) {
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
    const grid = new Map<string, number[]>();
    for (let i = 0; i < n; i++) {
      const key = `${Math.floor(ns[i].x / cut)},${Math.floor(ns[i].y / cut)}`;
      const b = grid.get(key);
      if (b) b.push(i);
      else grid.set(key, [i]);
    }
    for (let i = 0; i < n; i++) {
      const gx = Math.floor(ns[i].x / cut), gy = Math.floor(ns[i].y / cut);
      for (let ox = -1; ox <= 1; ox++) for (let oy = -1; oy <= 1; oy++) {
        const b = grid.get(`${gx + ox},${gy + oy}`);
        if (!b) continue;
        for (const j of b) {
          if (j <= i) continue;
          let dx = ns[j].x - ns[i].x, dy = ns[j].y - ns[i].y;
          let d2 = dx * dx + dy * dy;
          if (d2 > cut2) continue;
          if (d2 < 1) {
            // Coincident: separate deterministically.
            dx = ((i * 7919 + j) % 13) - 6 || 1;
            dy = ((j * 104729 + i) % 11) - 5 || 1;
            d2 = dx * dx + dy * dy;
          }
          const f = LAYOUT.repulse / d2 / Math.sqrt(d2);
          fx[i] -= f * dx;
          fy[i] -= f * dy;
          fx[j] += f * dx;
          fy[j] += f * dy;
        }
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
    this.alpha *= LAYOUT.alphaDecay;
    return speed / n;
  }

  /** Run up to `steps` steps now (a fresh graph is pre-settled before it is first drawn). */
  settle(steps: number): void {
    for (let i = 0; i < steps && !this.settled; i++) this.step();
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
