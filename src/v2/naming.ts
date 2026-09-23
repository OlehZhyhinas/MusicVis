// Descriptive, inherited names for V2 population members.
//
// A name is "Adjective Noun" (a merge's noun may itself be two words: "Serpent
// Lattice"). The noun says what the preset draws (its dominant emitter kind, a
// hybrid for a fused pair, or a structural word when a fold chain dominates a
// weak emitter). The adjective says how it moves or looks, picked from the
// traits actually present in the genome, weighted by their strength. Neither
// ever names a fixed hue: the palette hue is an offset from the song's key at
// runtime, so the same preset shows a different hue on every song.
//
// Inheritance: when a child is bred, a parent's existing word (noun or
// adjective) is kept whenever it still describes the child (its word is a
// valid choice for one of the child's candidate pools). This is why a family
// name persists across generations until the trait it names actually
// disappears. The final pick (parent word, or a fresh one when none fits) is
// otherwise deterministic: a hash of the genome selects among the valid
// synonyms, so export/import and re-runs reproduce the same name.

import { flatEmitters, genomeHash, isVarOp, type EmitterGene, type EmitterKind, type Genome } from './genome';

// -------------------------------------------------------------- helpers

function hashPick(pool: readonly string[], seed: number): string {
  const i = Math.abs(seed) % pool.length;
  return pool[i];
}

/** The first parent word (in order given) that is a member of `pool`; the pool's own pick otherwise. */
function pickWord(pool: readonly string[], seed: number, parentNames: readonly string[], fallback?: string): string {
  for (const name of parentNames) {
    for (const word of name.split(/\s+/)) {
      if (pool.includes(word)) return word;
    }
  }
  return fallback ?? hashPick(pool, seed);
}

// ---------------------------------------------------------------- nouns

export const NOUN_POOLS: Record<Exclude<EmitterKind, 'merge'>, string[]> = {
  wave: ['Line', 'Trace', 'Thread', 'Strand', 'Current', 'Signal', 'Wavelet', 'Course', 'Sinew', 'Skein', 'Waveform', 'Tremor'],
  spectrum: ['Skyline', 'Equalizer', 'Spires', 'Towers', 'Bars', 'Pillars', 'Palisade', 'Battlement', 'Cityscape', 'Columns', 'Facade', 'Metropolis'],
  particles: ['Swarm', 'Storm', 'Spray', 'Haze', 'Dust', 'Flurry', 'Shower', 'Cluster', 'Scatter', 'Drift', 'Cyclone', 'Blizzard'],
  stars: ['Constellation', 'Starfield', 'Galaxy', 'Cosmos', 'Firmament', 'Stardust', 'Night', 'Heavens', 'Starscape', 'Sparks', 'Asterism', 'Zenith'],
  ink: ['Inkblot', 'Plume', 'Bloom', 'Stain', 'Blot', 'Wash', 'Cloud', 'Tendril', 'Smoke', 'Billow', 'Wisp', 'Smudge'],
  wire: ['Lattice', 'Polyhedron', 'Frame', 'Crystal', 'Cage', 'Scaffold', 'Mesh', 'Girder', 'Truss', 'Framework', 'Armature', 'Web'],
  plasma: ['Contour', 'Marble', 'Topography', 'Isobar', 'Currents', 'Gradient', 'Strata', 'Landscape', 'Relief', 'Landform', 'Isoline', 'Ripples'],
  aurora: ['Curtain', 'Aurora', 'Veil', 'Shimmer', 'Streamer', 'Drape', 'Corona', 'Radiance', 'Skyglow', 'Glow', 'Luminance', 'Borealis'],
  blobs: ['Droplets', 'Mercury', 'Beads', 'Globules', 'Bubbles', 'Pebbles', 'Marbles', 'Dew', 'Jelly', 'Gems', 'Blobs', 'Spheres'],
  flame: ['Flame', 'Nebula', 'Filament', 'Cinder', 'Ember', 'Blaze', 'Inferno', 'Wildfire', 'Spark', 'Firestorm', 'Bonfire', 'Torch'],
  edge: ['Fringe', 'Rim', 'Silhouette', 'Ridgeline', 'Threshold', 'Border', 'Margin', 'Outline', 'Verge', 'Brink', 'Crest', 'Profile'],
  tiles: ['Mosaic', 'Grid', 'Checkerboard', 'Tessellation', 'Honeycomb', 'Patchwork', 'Quilt', 'Weave', 'Parquet', 'Squares', 'Tilework', 'Fretwork'],
  horizon: ['Ridge', 'Horizon', 'Dunes', 'Bluffs', 'Highlands', 'Foothills', 'Mesa', 'Escarpment', 'Terrain', 'Valley', 'Plateau', 'Badlands'],
  orb: ['Moon', 'Orb', 'Pearl', 'Lantern', 'Globe', 'Sphere', 'Bulb', 'Halo', 'Satellite', 'Beacon', 'Sun', 'Planet'],
  snake: ['Serpent', 'Trail', 'Ribbon', 'Wake', 'Coil', 'Sidewinder', 'Viper', 'Python', 'Eel', 'Adder', 'Slither', 'Curl'],
};
/** Fallback for a kind that somehow has no pool (should not happen; every EMITTER_KIND but merge is covered). */
const GENERIC_NOUN = ['Signal', 'Glow', 'Pulse', 'Figure', 'Form', 'Shape'];

/** A preset dominated by its fold chain (kaleido / mirror) over a faint emitter takes a structural noun. */
const STRUCTURAL_NOUNS = ['Mandala', 'Kaleidoscope', 'Rosette', 'Medallion', 'Tunnel', 'Wheel', 'Rose', 'Prism', 'Labyrinth', 'Spiral', 'Portal', 'Rings'];

/** Hand-picked pair names for common merges, keyed by each part's own word (still one of NOUN_POOLS[kind]). */
const HYBRID_PAIRS: Partial<Record<string, Partial<Record<Exclude<EmitterKind, 'merge'>, string>>>> = {
  'snake+wire': { snake: 'Serpent', wire: 'Lattice' },
  'flame+orb': { flame: 'Flame', orb: 'Moon' },
  'ink+orb': { ink: 'Bloom', orb: 'Moon' },
  'orb+wire': { orb: 'Pearl', wire: 'Cage' },
  'aurora+orb': { aurora: 'Curtain', orb: 'Lantern' },
  'blobs+orb': { blobs: 'Mercury', orb: 'Globe' },
  'ink+wire': { ink: 'Inkblot', wire: 'Frame' },
  'spectrum+wire': { spectrum: 'Spires', wire: 'Crystal' },
  'wave+wire': { wave: 'Current', wire: 'Polyhedron' },
  'flame+ink': { flame: 'Ember', ink: 'Plume' },
  'orb+snake': { orb: 'Moon', snake: 'Serpent' },
  'orb+stars': { orb: 'Lantern', stars: 'Constellation' },
  'orb+plasma': { orb: 'Globe', plasma: 'Contour' },
};

function pairKey(a: EmitterKind, b: EmitterKind): string {
  return [a, b].sort().join('+');
}

/** A preset whose look comes from a strong fold chain over a faint emitter (kaleido / mirror). */
function isStructural(g: Genome): boolean {
  const body = g.emitters[0];
  if (!body) return false;
  const gain = body.kind === 'merge' ? Math.max(body.parts?.[0]?.p.gain ?? 0, body.parts?.[1]?.p.gain ?? 0) : (body.p.gain ?? 1);
  if (gain >= 0.4) return false;
  for (const o of [...g.chain, ...(g.draw ?? [])]) {
    if ((o.op === 'kaleido' || o.op === 'mirror') && o.w >= 0.5) return true;
  }
  return false;
}

function pickNoun(g: Genome, parentNames: readonly string[]): string {
  const h = genomeHash(g);
  if (isStructural(g)) return pickWord(STRUCTURAL_NOUNS, h, parentNames);
  const body: EmitterGene | undefined = g.emitters[0];
  if (!body) return pickWord(GENERIC_NOUN, h, parentNames);
  if (body.kind === 'merge' && body.parts) {
    // Merge parts are never themselves 'merge' (repair()/validate() forbid nesting).
    const a = body.parts[0].kind as Exclude<EmitterKind, 'merge'>;
    const b = body.parts[1].kind as Exclude<EmitterKind, 'merge'>;
    const poolA = NOUN_POOLS[a] ?? GENERIC_NOUN;
    const poolB = NOUN_POOLS[b] ?? GENERIC_NOUN;
    const curated = HYBRID_PAIRS[pairKey(a, b)];
    const wordA = pickWord(poolA, h, parentNames, curated?.[a]);
    const wordB = pickWord(poolB, h >>> 5, parentNames, curated?.[b]);
    return `${wordA} ${wordB}`;
  }
  const pool = body.kind === 'merge' ? GENERIC_NOUN : (NOUN_POOLS[body.kind] ?? GENERIC_NOUN);
  return pickWord(pool, h, parentNames);
}

// ------------------------------------------------------------ adjectives

interface Trait {
  id: string;
  pool: readonly string[];
  weight: number;
}

const SPIRAL = ['Spiraling', 'Coiling', 'Whirling', 'Swirling', 'Corkscrewed', 'Curling', 'Vortical', 'Winding', 'Twisting', 'Looping', 'Gyrating', 'Helical', 'Twirling'];
const MIRRORED = ['Mirrored', 'Kaleidoscopic', 'Symmetric', 'Bilateral', 'Faceted', 'Radial', 'Fractured', 'Segmented', 'Paneled', 'Mirrorlike', 'Repeating', 'Sixfold'];
const TURNING = ['Turning', 'Wheeling', 'Spinning', 'Revolving', 'Circling', 'Rotating', 'Pivoting', 'Swiveling', 'Waltzing', 'Churning', 'Veering', 'Tilting'];
const ZOOM_OUT = ['Blooming', 'Flaring', 'Expanding', 'Unfurling', 'Billowing', 'Ballooning', 'Radiating', 'Dilating', 'Swelling', 'Surfacing', 'Emerging', 'Erupting'];
const ZOOM_IN = ['Tunneling', 'Receding', 'Plunging', 'Diving', 'Sinking', 'Collapsing', 'Contracting', 'Vanishing', 'Withdrawing', 'Funneling', 'Telescoping', 'Deepening'];
const FALLING = ['Falling', 'Cascading', 'Tumbling', 'Dropping', 'Showering', 'Plummeting', 'Pouring', 'Drooping', 'Settling', 'Trickling', 'Raining', 'Sifting'];
const RISING = ['Rising', 'Ascending', 'Lifting', 'Climbing', 'Soaring', 'Wafting', 'Levitating', 'Buoyant', 'Elevating', 'Uplifting', 'Floating', 'Skyward'];
const DRIFT = ['Drifting', 'Scrolling', 'Sliding', 'Gliding', 'Sweeping', 'Panning', 'Streaming', 'Coursing', 'Traversing', 'Meandering', 'Roaming', 'Wandering'];
const RIPPLING = ['Rippling', 'Undulating', 'Wavering', 'Quivering', 'Lapping', 'Rolling', 'Wobbling', 'Fluttering', 'Corrugated', 'Shivering', 'Pulsating', 'Lilting'];
const WANDERING = ['Errant', 'Roiling', 'Turbulent', 'Restive', 'Wispy', 'Fraying', 'Grainy', 'Jittery', 'Skittish', 'Unsettled', 'Flickering', 'Shifting'];
const BREATHING = ['Breathing', 'Throbbing', 'Heaving', 'Rhythmic', 'Ebbing', 'Tidal', 'Inflating', 'Deflating', 'Sighing', 'Pumping', 'Rocking', 'Swaying'];
const ORBITAL = ['Orbital', 'Rotational', 'Planetary', 'Gyroscopic', 'Celestial', 'Ringed', 'Encircling', 'Cycling', 'Moonlike', 'Annular', 'Concentric', 'Haloed'];
const TILED = ['Tiled', 'Patterned', 'Gridded', 'Partitioned', 'Modular', 'Checkered', 'Quilted', 'Paved', 'Banded', 'Striped', 'Sectioned', 'Latticed'];
const FRACTAL = ['Fractal', 'Feathered', 'Recursive', 'Filigreed', 'Ornate', 'Intricate', 'Lacelike', 'Branching', 'Dendritic', 'Fernlike', 'Iterative', 'Crystalline'];
const LIQUID = ['Liquid', 'Flowing', 'Marbled', 'Fluid', 'Molten', 'Viscous', 'Syrupy', 'Watery', 'Silken', 'Glossy', 'Oozing', 'Sloshing'];
const TRAILING = ['Lingering', 'Trailing', 'Persistent', 'Ghosting', 'Smoldering', 'Lasting', 'Enduring', 'Smeared', 'Streaked', 'Cometlike', 'Residual', 'Fading'];
const SURGING = ['Surging', 'Pulsing', 'Kicking', 'Pounding', 'Rushing', 'Bursting', 'Beating', 'Thumping', 'Driving', 'Percussive', 'Charging', 'Punchy'];
const CALM = ['Hushed', 'Slow', 'Gentle', 'Still', 'Serene', 'Quiet', 'Placid', 'Tranquil', 'Soft', 'Languid', 'Dreamy', 'Sedate'];
const ENERGETIC = ['Frenzied', 'Restless', 'Manic', 'Feverish', 'Wild', 'Volatile', 'Riotous', 'Furious', 'Hyperactive', 'Kinetic', 'Explosive', 'Storming'];
const PALE = ['Pale', 'Muted', 'Dusky', 'Smoky', 'Faded', 'Washed', 'Ashen', 'Ghostly', 'Milky', 'Powdery', 'Wan', 'Bleached'];
const VIVID = ['Vivid', 'Neon', 'Luminous', 'Glowing', 'Radiant', 'Electric', 'Blazing', 'Brilliant', 'Incandescent', 'Dazzling', 'Saturated', 'Fluorescent'];
const MONO = ['Monochrome', 'Grayscale', 'Uniform', 'Achromatic', 'Tonal', 'Tinted', 'Minimal', 'Sparse', 'Unified', 'Monotone', 'Plain', 'Understated'];
const TWO_TONE = ['Two-Tone', 'Dual', 'Contrasted', 'Split', 'Paired', 'Opposed', 'Bichrome', 'Duotone', 'Balanced', 'Complementary', 'Bicolor', 'Polarized'];
const PRISMATIC = ['Iridescent', 'Prismatic', 'Rainbow', 'Multicolored', 'Opalescent', 'Chromatic', 'Spectral', 'Variegated', 'Tri-Tone', 'Pearlescent', 'Nacreous', 'Shimmering'];
const SHADOWED = ['Shadowed', 'Dim', 'Vignetted', 'Darkened', 'Eclipsed', 'Obscured', 'Umbral', 'Hooded', 'Cloaked', 'Murky', 'Occluded', 'Sombre'];
const REFLECTED = ['Reflected', 'Pooled', 'Doubled', 'Inverted', 'Twinned', 'Glassy', 'Lucent', 'Specular', 'Echoed', 'Echoing', 'Glimmering', 'Lakeside'];
const GENERIC_ADJ = ['Drifting', 'Quiet', 'Restless', 'Steady', 'Roaming', 'Vagrant'];

function clamp01(x: number): number {
  return Math.min(1, Math.max(0, x));
}

function traits(g: Genome): Trait[] {
  const list: Trait[] = [];
  const add = (id: string, pool: readonly string[], weight: number) => {
    if (weight <= 0.02) return;
    const ex = list.find((t) => t.id === id);
    if (ex) ex.weight = Math.max(ex.weight, weight);
    else list.push({ id, pool, weight });
  };

  for (const o of [...g.chain, ...(g.draw ?? [])]) {
    const w = o.w;
    switch (o.op) {
      case 'swirl': add('spiral', SPIRAL, w * clamp01(0.5 + Math.abs(o.p.amt) * 40)); break;
      case 'twist': add('spiral', SPIRAL, w * clamp01(0.4 + Math.abs(o.p.amt) * 100)); break;
      case 'mirror': add('mirrored', MIRRORED, w * 0.9); break;
      case 'kaleido': add('mirrored', MIRRORED, w * 1); break;
      case 'rotate': add('turning', TURNING, w * clamp01(0.3 + Math.abs(o.p.lock) * 4 + Math.abs(o.p.rate) * 60)); break;
      case 'zoom': {
        const strength = clamp01(Math.abs(o.p.rate) * 30);
        if (o.p.rate > 0.0015) add('zoomOut', ZOOM_OUT, w * strength);
        else if (o.p.rate < -0.0015) add('zoomIn', ZOOM_IN, w * strength);
        break;
      }
      case 'translate': {
        const { vx, vy } = o.p;
        if (Math.abs(vy) > Math.abs(vx) && Math.abs(vy) > 0.05) add(vy > 0 ? 'falling' : 'rising', vy > 0 ? FALLING : RISING, w * clamp01(Math.abs(vy) * 3));
        else if (Math.abs(vx) > 0.05) add('drift', DRIFT, w * clamp01(Math.abs(vx) * 3));
        break;
      }
      case 'ripple': add('rippling', RIPPLING, w * clamp01(o.p.amp * 400)); break;
      case 'noise': add('wandering', WANDERING, w * clamp01(o.p.amp * 500)); break;
      case 'push': add('breathing', BREATHING, w * clamp01(Math.abs(o.p.amt) * 150)); break;
      case 'polar': add('orbital', ORBITAL, w * 0.6); break;
      case 'tile': add('tiled', TILED, w * 0.7); break;
      default:
        if (isVarOp(o.op)) add('fractal', FRACTAL, w * 0.7);
    }
  }
  if (flatEmitters(g).some((e) => e.kind === 'flame')) add('fractal', FRACTAL, 0.5);

  if (g.carrier.kind === 'fluid') add('liquid', LIQUID, 0.8);
  if (g.carrier.p.halfLife > 3) add('trailing', TRAILING, clamp01((g.carrier.p.halfLife - 3) / 10));

  const surgeSignals = new Set(['bass', 'drums', 'beat', 'surge', 'drop', 'hit']);
  if (g.reactions.some((r) => surgeSignals.has(r.src))) add('surging', SURGING, 0.6);

  const mid = (g.energy[0] + g.energy[1]) / 2;
  add(mid >= 0.5 ? 'energetic' : 'calm', mid >= 0.5 ? ENERGETIC : CALM, 0.4);

  const c = g.color.p;
  if (c.sat < 0.45 || c.exposure < 0.85) add('pale', PALE, clamp01(0.5 + (0.45 - Math.min(0.45, c.sat))));
  if (c.sat > 0.85 && c.exposure > 1.05) add('vivid', VIVID, 0.5);
  if (g.color.scheme === 'mono') add('mono', MONO, 0.9);
  if (g.color.scheme === 'complementary') add('twoTone', TWO_TONE, 0.9);
  if (g.color.scheme === 'triad' || g.color.scheme === 'split') add('prismatic', PRISMATIC, 0.9);
  if (c.vignette > 0.55) add('shadowed', SHADOWED, clamp01((c.vignette - 0.55) * 3));
  if (c.reflect === 1) add('reflected', REFLECTED, 0.8);

  list.sort((a, b) => b.weight - a.weight);
  return list;
}

function pickAdjective(g: Genome, parentNames: readonly string[]): string {
  const list = traits(g);
  const h = genomeHash(g);
  for (const name of parentNames) {
    const word = name.split(/\s+/)[0];
    if (list.some((t) => t.pool.includes(word))) return word;
  }
  if (!list.length) return hashPick(GENERIC_ADJ, h);
  return hashPick(list[0].pool, h >>> 3);
}

// ----------------------------------------------------------------- name

/**
 * Builds a descriptive name for the genome. `parentNames` (each an existing
 * "Adjective Noun[...]" name) lets an inherited word survive into the child
 * whenever it still applies; pass none for a genome with no lineage (seeds
 * use their own V1 names and never call this). Deterministic in the genome
 * and parent names alone: repeated calls, or a re-import, give the same name
 * (duplicate resolution against a live population is layered on separately).
 */
export function nameFor(g: Genome, parentNames: readonly string[] = []): string {
  const noun = pickNoun(g, parentNames);
  const adj = pickAdjective(g, parentNames);
  return `${adj} ${noun}`;
}

// ------------------------------------------------------------ for tests

/** Every adjective trait pool, keyed by trait id (for tests: membership checks, vocabulary size). */
export const ADJ_POOLS: Record<string, readonly string[]> = {
  spiral: SPIRAL, mirrored: MIRRORED, turning: TURNING, zoomOut: ZOOM_OUT, zoomIn: ZOOM_IN, falling: FALLING,
  rising: RISING, drift: DRIFT, rippling: RIPPLING, wandering: WANDERING, breathing: BREATHING, orbital: ORBITAL,
  tiled: TILED, fractal: FRACTAL, liquid: LIQUID, trailing: TRAILING, surging: SURGING, calm: CALM,
  energetic: ENERGETIC, pale: PALE, vivid: VIVID, mono: MONO, twoTone: TWO_TONE, prismatic: PRISMATIC,
  shadowed: SHADOWED, reflected: REFLECTED, generic: GENERIC_ADJ,
};

/**
 * Fixed hue names that must never appear in a generated name: the palette hue
 * is an offset from the song's key at runtime, so a preset that named itself
 * after a colour would be wrong on the very next song.
 */
export const HUE_WORDS = [
  'Jade', 'Crimson', 'Cobalt', 'Emerald', 'Amber', 'Coral', 'Ivory', 'Violet', 'Gold', 'Golden', 'Silver', 'Teal',
  'Magenta', 'Turquoise', 'Scarlet', 'Indigo', 'Azure', 'Ruby', 'Sapphire', 'Ebony', 'Bronze', 'Cyan', 'Maroon',
  'Lavender', 'Cerulean', 'Vermilion', 'Chartreuse', 'Ochre', 'Fuchsia', 'Burgundy', 'Tangerine', 'Lilac',
];
