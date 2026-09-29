// Balance body families before selecting an individual. A family with a thousand
// descendants must not get a thousand tickets against one newly introduced shape.
import type { Genome, ShapeGene, ShapeKind } from './genome';
import type { ExploreMode } from './novelty';

export const FAMILY_EXPLORATION: Record<ExploreMode, number> = { off: 0, gentle: 0.25, explore: 0.6, wild: 0.85 };

const FAMILY_VARIANT: Partial<Record<ShapeKind, string>> = { curve: 'form', superscope: 'family', scene: 'scene', edge: 'mode', cells: 'mode' };

function shapeFamily(s: ShapeGene): string {
  // Discrete families inside multi-purpose shape genes are visually distinct too.
  const variant = FAMILY_VARIANT[s.kind];
  return variant ? `${s.kind}:${s.p[variant]}` : s.kind;
}

export function bodyFamily(g: Genome): string {
  return [...new Set(g.bodies.flatMap(b => [shapeFamily(b.shape), ...(b.fuse ? [shapeFamily(b.fuse.shape)] : [])]))].sort().join('+');
}

interface Candidate {
  id: string;
  parents: string[];
  genome: Genome;
  likes: number;
  dislikes: number;
}

export function closeRelatives(a: Candidate, b: Candidate): boolean {
  return a.parents.includes(b.id) || b.parents.includes(a.id) || a.parents.some(id => b.parents.includes(id));
}

/** One equally likely family unlike the anchor, preserving fitness selection within that family. */
export function diverseFamily<T extends Candidate>(pool: T[], anchor: T, rng: () => number): T[] | null {
  const home = bodyFamily(anchor.genome);
  const other = pool.filter(m => m.id !== anchor.id && bodyFamily(m.genome) !== home && !(m.dislikes >= 3 && m.dislikes > m.likes));
  // Prefer an unrelated partner when available, but allow a small family to breed.
  const unrelated = other.filter(m => !closeRelatives(anchor, m));
  const groups = new Map<string, T[]>();
  for (const m of unrelated.length ? unrelated : other) {
    const k = bodyFamily(m.genome);
    const group = groups.get(k);
    if (group) group.push(m); else groups.set(k, [m]);
  }
  const families = [...groups.values()];
  return families.length ? families[Math.min(families.length - 1, Math.floor(rng() * families.length))] : null;
}
