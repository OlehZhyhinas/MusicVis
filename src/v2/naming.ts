// Descriptive, inherited names for V2 population members.
//
// A name is "Adjective Noun" (a fused body's noun may itself be two words:
// "Serpent Lattice"). The noun says what the preset draws: the main body's noun
// family, from its shape, placement, material and emission (a dot walking is a
// serpent, a dot on a grid a starfield, sparks from a hidden body a swarm), a
// hybrid for a fused shape, or a structural word when a fold chain dominates a
// faint body. The adjective says how it moves or looks, picked from the traits
// actually present (motion, deformation, placement, material, emission, chain,
// carrier, colour), weighted by their strength. Neither
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

import { genomeHash, isVarOp, type BodyGene, type Genome, type ShapeGene } from './genome';

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

/** What a body reads as (the noun family): derived from its shape, placement, material and emission. */
export type NounKind =
  | 'wave' | 'spectrum' | 'particles' | 'stars' | 'ink' | 'wire' | 'plasma' | 'aurora' | 'blobs' | 'flame' | 'edge'
  | 'tiles' | 'horizon' | 'orb' | 'snake' | 'polygon' | 'star' | 'segment' | 'scope' | 'network' | 'flock' | 'beams' | 'depth' | 'cells' | 'cymatics' | 'habitat' | 'journey';

export const NOUN_POOLS: Record<NounKind, string[]> = {
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
  polygon: ['Prism', 'Facet', 'Shard', 'Hexagon', 'Gem', 'Tablet', 'Plate', 'Keystone', 'Tile', 'Emblem', 'Sigil', 'Lozenge'],
  star: ['Star', 'Asterisk', 'Pinwheel', 'Starburst', 'Compass', 'Spur', 'Rowel', 'Sunburst', 'Blossom', 'Burr', 'Thistle', 'Urchin'],
  flock: ['Murmuration', 'Flock', 'Starlings', 'Shoal', 'Swallows', 'Rookery', 'Covey', 'Wheel', 'Gyre', 'Volery', 'Squadron', 'Exaltation'],
  habitat: ['Ecosystem', 'Tidepool', 'Menagerie', 'Habitat', 'Reef', 'Biome', 'Savanna', 'Terrarium', 'Aviary', 'Wetland', 'Lagoon', 'Wilderness'],
  network: ['Mycelium', 'Plexus', 'Rhizome', 'Veins', 'Capillaries', 'Delta', 'Tracery', 'Filigree', 'Roots', 'Mould', 'Hyphae', 'Reticulum'],
  cells: ['Cells', 'Foam', 'Membrane', 'Froth', 'Tissue', 'Colony', 'Crackle', 'Scales', 'Vesicles', 'Cytoplasm', 'Lacework', 'Hive'],
  cymatics: ['Chladni', 'Resonance', 'Nodes', 'Overtone', 'Harmonic', 'Sandplate', 'Standing Wave', 'Figure', 'Vibration', 'Drumhead', 'Soundplate', 'Mandorla'],
  journey: ['Journey', 'Voyage', 'Odyssey', 'Pilgrimage', 'Expedition', 'Crossing', 'Wayfaring', 'Sojourn', 'Traverse', 'Trek', 'Frontier', 'Passage'],
  beams: ['Searchlight', 'Floodlight', 'Spotlight', 'Lightshow', 'Beacons', 'Shafts', 'Rays', 'Laser', 'Stagelight', 'Limelight', 'Footlights', 'Lighthouse'],
  depth: ['Sculpture', 'Cavern', 'Chamber', 'Abyss', 'Vault', 'Hollow', 'Monolith', 'Reliquary', 'Atrium', 'Void', 'Depths', 'Diorama'],
  segment: ['Stroke', 'Dash', 'Needle', 'Streak', 'Stitch', 'Splinter', 'Rod', 'Baton', 'Quill', 'Sliver', 'Spoke', 'Wand'],
  scope: ['Scope', 'Knot', 'Gyroscope', 'Armillary', 'Spirograph', 'Torus', 'Orrery', 'Astrolabe', 'Whorl', 'Rotor', 'Loop', 'Oscillograph'],
};

/** The noun family of a body. */
export function nounKind(b: BodyGene): NounKind {
  const hidden = b.emit.kind === 'sparks' && b.emit.p.body < 0.3;
  if (hidden && b.shape.kind !== 'flame') return 'particles';
  if (b.emit.kind === 'slime' && b.emit.p.body < 0.3) return 'network';
  if (b.emit.kind === 'flock' && b.emit.p.body < 0.3) return 'flock';
  if (b.emit.kind === 'ecosystem' && b.emit.p.body < 0.3) return 'habitat';
  return shapeNoun(b.shape, b);
}

function shapeNoun(s: ShapeGene, b: BodyGene | null): NounKind {
  const pk = b?.place.kind;
  switch (s.kind) {
    case 'flame': return 'flame';
    case 'plasma': return 'plasma';
    case 'aurora': return 'aurora';
    case 'terrain': return 'horizon';
    case 'scene': return 'depth';
    case 'edge': return 'edge';
    case 'beams': return 'beams';
    case 'cells': return 'cells';
    case 'cymatics': return 'cymatics';
    case 'landscape': return 'journey';
    case 'curve': return 'wave';
    case 'superscope': return 'scope';
    case 'bars': return 'spectrum';
    case 'solid': return 'wire';
    case 'segment': return pk === 'walker' ? 'snake' : 'segment';
    case 'polygon': return pk === 'grid' ? 'tiles' : 'polygon';
    case 'star': return pk === 'grid' ? 'stars' : 'star';
    case 'dot':
      if (!b) return 'orb';
      if (pk === 'walker') return 'snake';
      if (pk === 'grid') return 'stars';
      if (pk === 'float' || b.material.kind === 'chrome') return 'blobs';
      if (pk === 'orbit' || pk === 'stations' || pk === 'row' || b.emit.kind === 'dye') return 'ink';
      return 'orb';
  }
  return 'orb';
}
/** Fallback for a kind that somehow has no pool (should not happen; every noun kind is covered). */
const GENERIC_NOUN = ['Signal', 'Glow', 'Pulse', 'Figure', 'Form', 'Shape'];

/** A preset dominated by its fold chain (kaleido / mirror) over a faint emitter takes a structural noun. */
const STRUCTURAL_NOUNS = ['Mandala', 'Kaleidoscope', 'Rosette', 'Medallion', 'Tunnel', 'Wheel', 'Rose', 'Prism', 'Labyrinth', 'Spiral', 'Portal', 'Rings'];

/** Hand-picked pair names for common fusions, keyed by each part's own word (still one of NOUN_POOLS[kind]). */
const HYBRID_PAIRS: Partial<Record<string, Partial<Record<NounKind, string>>>> = {
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

function pairKey(a: NounKind, b: NounKind): string {
  return [a, b].sort().join('+');
}

/** A preset whose look comes from a strong fold chain over a faint body (kaleido / mirror). */
function isStructural(g: Genome): boolean {
  const body = g.bodies[0];
  if (!body || body.material.p.gain >= 0.4) return false;
  for (const o of [...g.chain, ...(body.deform.ops ?? [])]) {
    if ((o.op === 'kaleido' || o.op === 'mirror' || o.op === 'tunnel') && o.w >= 0.5) return true;
  }
  return false;
}

function pickNoun(g: Genome, parentNames: readonly string[]): string {
  const h = genomeHash(g);
  if (isStructural(g)) return pickWord(STRUCTURAL_NOUNS, h, parentNames);
  const body: BodyGene | undefined = g.bodies[0];
  if (!body) return pickWord(GENERIC_NOUN, h, parentNames);
  const a = nounKind(body);
  if (body.fuse) {
    let b = shapeNoun(body.fuse.shape, null);
    if (b === a) b = body.fuse.shape.kind === 'dot' ? 'orb' : b;
    const poolA = NOUN_POOLS[a] ?? GENERIC_NOUN;
    const poolB = NOUN_POOLS[b] ?? GENERIC_NOUN;
    if (a !== b) {
      const curated = HYBRID_PAIRS[pairKey(a, b)];
      const wordA = pickWord(poolA, h, parentNames, curated?.[a]);
      const wordB = pickWord(poolB.filter((w) => w !== wordA), h >>> 5, parentNames, curated?.[b]);
      return `${wordA} ${wordB}`;
    }
  }
  return pickWord(NOUN_POOLS[a] ?? GENERIC_NOUN, h, parentNames);
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
const LAYERED = ['Collaged', 'Overlaid', 'Superimposed', 'Stacked', 'Composited', 'Interlaced', 'Negative', 'Solarized', 'Stenciled', 'Cutout', 'Sandwiched', 'Laminated'];
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
const REACHING = ['Reaching', 'Tentacled', 'Grasping', 'Sprawling', 'Starfish', 'Clutching', 'Groping', 'Octopoid', 'Stretching', 'Flailing', 'Beckoning', 'Waving'];
const DARTING = ['Darting', 'Lurching', 'Jolting', 'Zigzag', 'Swerving', 'Dodging', 'Jerking', 'Sidestepping', 'Careening', 'Bounding', 'Pouncing', 'Skipping'];
const STIPPLED = ['Stippled', 'Dotted', 'Pointillist', 'Speckled', 'Freckled', 'Pebbled', 'Spotted', 'Granular', 'Sequined', 'Dappled', 'Flecked', 'Beaded'];
const MOTTLED = ['Cratered', 'Pitted', 'Mottled', 'Weathered', 'Rugged', 'Scarred', 'Pockmarked', 'Worn', 'Stony', 'Dusty', 'Etched', 'Carved'];
const FLOCKING = ['Flocking', 'Wheeling', 'Swooping', 'Schooling', 'Banking', 'Soaring', 'Circling', 'Murmuring', 'Gliding', 'Veering', 'Winging', 'Migrating'];
const TEEMING = ['Teeming', 'Swarming', 'Grazing', 'Hunting', 'Burrowing', 'Brooding', 'Feral', 'Predatory', 'Pollinated', 'Thriving', 'Symbiotic', 'Nesting'];
const VEINED = ['Veined', 'Branching', 'Creeping', 'Mycelial', 'Foraging', 'Sprawling', 'Rooting', 'Tendrilled', 'Reticulate', 'Threaded', 'Questing', 'Spreading'];
const SPARKING = ['Sparking', 'Crackling', 'Fizzing', 'Spitting', 'Sputtering', 'Scintillating', 'Effervescent', 'Popping', 'Sparkling', 'Glinting', 'Twinkling', 'Spangled'];
const GRADED = ['Graded', 'Layered', 'Tiered', 'Stratified', 'Terraced', 'Shaded', 'Ombre', 'Banked', 'Tapered', 'Sloped', 'Ranked', 'Scaled'];
const STEPPED = ['Stepping', 'Ticking', 'Clockwork', 'Staccato', 'Metered', 'Marching', 'Pulsed', 'Chopped', 'Stuttering', 'Measured', 'Tapping', 'Syncopated'];
const PAINTED = ['Painted', 'Brushed', 'Inked', 'Daubed', 'Lacquered', 'Glazed', 'Enameled', 'Varnished', 'Stroked', 'Scrawled', 'Scribbled', 'Penned'];
const STAGED = ['Cinematic', 'Staged', 'Choreographed', 'Theatrical', 'Scripted', 'Plotted', 'Framed', 'Directed', 'Scenic', 'Dramatic', 'Orchestrated', 'Rehearsed'];
const EMBOSSED = ['Embossed', 'Burnished', 'Chromed', 'Sculpted', 'Chiseled', 'Polished', 'Hammered', 'Beaten', 'Gilded', 'Mercurial', 'Pewter', 'Repousse'];
const PSYCHEDELIC = ['Psychedelic', 'Acid', 'Lysergic', 'Trippy', 'Dayglo', 'Technicolor', 'Hallucinatory', 'Kandy', 'Lava-Lamp', 'Tie-Dyed', 'Op-Art', 'Blacklight'];
const SHIFTING = ['Journeying', 'Morphing', 'Mutating', 'Evolving', 'Shapeshifting', 'Protean', 'Metamorphic', 'Transforming', 'Changeling', 'Chameleon', 'Fluxing', 'Migrant'];
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

  for (const o of [...g.chain, ...g.bodies.flatMap((b) => b.deform.ops ?? [])]) {
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
      case 'tunnel': add(o.p.speed >= 0 ? 'zoomOut' : 'zoomIn', o.p.speed >= 0 ? ZOOM_OUT : ZOOM_IN, clamp01(0.5 + Math.abs(o.p.speed) * 0.4)); break;
      case 'tile': add('tiled', TILED, w * 0.7); break;
      case 'mosaic': add('tiled', TILED, clamp01(0.6 + o.p.gap * 0.5)); break;
      default:
        if (isVarOp(o.op)) add('fractal', FRACTAL, w * 0.7);
    }
  }
  if (g.bodies.some((b) => b.shape.kind === 'flame')) add('fractal', FRACTAL, 0.5);

  // Body sub-genes: how each copy moves, how the shape is bent, where copies sit, how it is lit.
  for (const [bi, b] of g.bodies.entries()) {
    const k = bi === 0 ? 1 : 0.6;
    const mp = b.motion.p;
    switch (b.motion.kind) {
      case 'spin': add('turning', TURNING, k * clamp01(0.4 + Math.abs(mp.rate) * 2)); break;
      case 'sway': add('breathing', BREATHING, k * clamp01(0.4 + mp.amp * 6)); break;
      case 'bob': add('breathing', BREATHING, k * clamp01(0.3 + mp.amp * 0.3)); break;
      case 'drift': add('drift', DRIFT, k * clamp01(0.3 + (Math.abs(mp.vx) + Math.abs(mp.vy)) * 20)); break;
      case 'circle': add('orbital', ORBITAL, k * clamp01(0.4 + mp.radius * 8)); break;
      case 'hits': add('darting', DARTING, k * clamp01(0.4 + mp.amt * 0.4)); break;
      case 'pulse': add('surging', SURGING, k * clamp01(0.4 + mp.amp * 2)); break;
    }
    const dp = b.deform.p;
    switch (b.deform.kind) {
      case 'arms': add('reaching', REACHING, k * clamp01(0.5 + dp.reach * 0.6)); break;
      case 'wobble': add('rippling', RIPPLING, k * clamp01(0.4 + dp.amp * 2)); break;
      case 'noise': add('wandering', WANDERING, k * clamp01(0.4 + dp.amp * 10)); break;
      case 'twist': add('spiral', SPIRAL, k * clamp01(0.4 + Math.abs(dp.amt) * 0.12)); break;
    }
    switch (b.place.kind) {
      case 'walker': add('drift', DRIFT, k * 0.6); break;
      case 'grid': add('tiled', TILED, k * 0.55); break;
      case 'ring': case 'mirror': add('mirrored', MIRRORED, k * 0.6); break;
      case 'orbit': case 'outline': add('orbital', ORBITAL, k * 0.55); break;
    }
    switch (b.material.kind) {
      case 'chrome': add('liquid', LIQUID, k * 0.6); break;
      case 'dots': add('stippled', STIPPLED, k * 0.7); break;
      case 'textured': if (b.material.p.tex === 0) add('mottled', MOTTLED, k * 0.5); break;
    }
    if (b.emit.kind === 'sparks') add('sparking', SPARKING, k * 0.55);
    if (b.emit.kind === 'slime') add('veined', VEINED, k * 0.7);
    if (b.emit.kind === 'flock') add('flocking', FLOCKING, k * 0.7);
    if (b.emit.kind === 'ecosystem') add('teeming', TEEMING, k * 0.7);
    if (b.emit.kind === 'cover') add('painted', PAINTED, k * 0.45);
    if (b.emit.kind === 'dye') add('liquid', LIQUID, k * 0.5);
    // Feel: stepped responses, slow clocks and long releases, sharp sensitive attacks.
    const fp = b.feel.p;
    if (b.feel.kind === 'step') add('stepped', STEPPED, k * 0.65);
    if (fp.rel >= 1 || fp.div >= 8) add('calm', CALM, k * clamp01(0.3 + fp.rel * 0.2 + (fp.div >= 8 ? 0.25 : 0)));
    if (fp.atk <= 0.02 && fp.sens >= 1.5) add('surging', SURGING, k * clamp01(0.3 + (fp.sens - 1.5) * 0.4));
  }

  if (g.carrier.kind === 'fluid') add('liquid', LIQUID, 0.8);
  if (g.bodies.some((b) => b.material.p.blend > 0)) add('layered', LAYERED, 0.5);
  if (g.carrier.kind !== 'none' && g.carrier.p.water > 0.05) add('rippling', RIPPLING, 0.4 + 0.5 * g.carrier.p.water);
  if (g.carrier.p.halfLife > 3) add('trailing', TRAILING, clamp01((g.carrier.p.halfLife - 3) / 10));

  const surgeSignals = new Set(['bass', 'drums', 'beat', 'surge', 'drop', 'hit']);
  if (g.reactions.some((r) => surgeSignals.has(r.src))) add('surging', SURGING, 0.6);

  const mid = (g.energy[0] + g.energy[1]) / 2;
  add(mid >= 0.5 ? 'energetic' : 'calm', mid >= 0.5 ? ENERGETIC : CALM, 0.4);

  const c = g.tone.p;
  if (c.sat < 0.45 || c.exposure < 0.85) add('pale', PALE, clamp01(0.5 + (0.45 - Math.min(0.45, c.sat))));
  if (c.sat > 0.85 && c.exposure > 1.05) add('vivid', VIVID, 0.5);
  const pk = g.palette.kind;
  if (pk === 'mono') add('mono', MONO, 0.9);
  if (pk === 'complementary') add('twoTone', TWO_TONE, 0.9);
  if (pk === 'triad' || pk === 'split' || pk === 'free') add('prismatic', PRISMATIC, pk === 'free' ? 0.7 : 0.9);
  if (c.vignette > 0.55) add('shadowed', SHADOWED, clamp01((c.vignette - 0.55) * 3));
  if (c.reflect === 1) add('reflected', REFLECTED, 0.8);
  if (c.relief > 0.25) add('embossed', EMBOSSED, clamp01(0.55 + c.relief * 0.45));
  if (c.huemap > 0.3 || c.solar > 0.4) add('psychedelic', PSYCHEDELIC, clamp01(0.5 + Math.max(c.huemap, c.solar) * 0.45));
  // Colour mapping: what drives each body's hue.
  for (const b of g.bodies) {
    const k = b === g.bodies[0] ? 1 : 0.6;
    switch (b.color.kind) {
      case 'pitch': add('prismatic', PRISMATIC, k * 0.55); break;
      case 'age': add('prismatic', PRISMATIC, k * 0.5); break;
      case 'height': add('graded', GRADED, k * clamp01(0.3 + Math.abs(b.color.p.amount) * 0.4)); break;
      case 'fixed': if (b.color.p.detail < 0.3) add('mono', MONO, k * 0.4); break;
    }
  }

  // Choreography: a picture composed over the song timeline, named by how strongly it is staged.
  if (g.choreo) {
    const cp = g.choreo.p;
    add('staged', STAGED, clamp01(0.35 + cp.push * 1.5 + cp.drain * 0.2 + cp.punch * 0.2));
  }
  // Drift: a preset that travels through gene space over the song, named by how far it roams.
  if (g.drift) {
    const dp = g.drift.p;
    add('shifting', SHIFTING, clamp01(0.4 + dp.step * 0.4 + dp.bound * 0.5 + dp.kinds * 0.1));
  }

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
 * use their own hand-given names and never call this). Deterministic in the genome
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
  shadowed: SHADOWED, reflected: REFLECTED, reaching: REACHING, darting: DARTING, stippled: STIPPLED, mottled: MOTTLED,
  sparking: SPARKING, veined: VEINED, flocking: FLOCKING, teeming: TEEMING, painted: PAINTED, stepped: STEPPED, graded: GRADED, generic: GENERIC_ADJ,
  staged: STAGED, embossed: EMBOSSED, psychedelic: PSYCHEDELIC, shifting: SHIFTING, layered: LAYERED,
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
