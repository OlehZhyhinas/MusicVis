// What a lyric line is about, without a language model: a hand-made lexicon maps words to imagery
// tags (fire, water, night...) and to valence (sad .. happy) and arousal (calm .. intense). Fast
// and deterministic, so every line of a song is read in well under a millisecond before it plays.

/** Imagery tags, in the order of the tag vector. */
export const LYRIC_TAGS = [
  'fire', 'water', 'sky', 'night', 'love', 'fall', 'rise', 'speed', 'cold', 'gold', 'dark', 'light', 'dream', 'city', 'nature', 'space', 'storm',
] as const;
export type LyricTag = (typeof LYRIC_TAGS)[number];
export const TAG_COUNT = LYRIC_TAGS.length;
export const tagIndex = (t: LyricTag): number => LYRIC_TAGS.indexOf(t);

/** Words per tag (base forms; plurals, -ing, -ed, -in' and -ly are folded onto them). */
const TAG_WORDS: Record<LyricTag, string> = {
  fire: 'fire flame flames burn burning burnt blaze blazing ember embers spark sparks smoke smoking heat hot ash ashes inferno ignite lit torch candle wildfire firework fireworks hell lava volcano scorch sear kindle',
  water: 'water ocean sea wave waves river rain raining rainy tear tears cry crying drown drowning flood swim swimming lake shore tide tides deep rivers stream flow flowing wet sink sinking underwater pool bay beach sail harbor harbour island',
  sky: 'sky skies cloud clouds heaven heavens fly flying flew wing wings bird birds air wind breeze blue horizon above high kite angel angels float floating soar',
  night: 'night nights midnight moon moonlight tonight evening dusk late sleep asleep bed dark shadows shadow nocturnal owl insomnia',
  love: 'love lover lovers loving kiss kisses kissing heart hearts baby darling honey sweet sweetheart hold embrace touch together forever romance romantic desire want need adore beloved valentine hug mine yours babe',
  fall: 'fall falling fell down drop dropping sink collapse crash crashing break broken breaking lose losing lost gone ground fade fading bury buried grave low under beneath descend tumble',
  rise: 'rise rising rose up higher lift lifting climb climbing grow growing stand raise risen ascend mountain peak summit top tower wake awake alive hope',
  speed: 'run running ran fast faster race racing rush rushing speed chase chasing drive driving car cars go gone quick hurry wild jump dance dancing move moving spin spinning roll rolling fly highway engine motor ride riding escape',
  cold: 'cold ice icy frozen freeze freezing snow snowing winter chill chilly frost numb cool shiver glacier arctic december',
  gold: 'gold golden money cash rich diamond diamonds shine shining crown king queen treasure jewel jewels silver luxury glitter glitters champagne dollar dollars fortune throne',
  dark: 'dark darkness black shadow shadows death dead die dying kill blood demon devil ghost ghosts grave hollow void evil fear scared nightmare doom pain hurt poison sin sinner',
  light: 'light lights shine shining bright brighter sun sunny sunshine sunlight glow glowing day daylight morning dawn sunrise white clear gleam flash beam lamp radiant',
  dream: 'dream dreams dreaming dreamer dreamy sleep asleep wish wishes imagine fantasy magic memory memories remember haze hazy daydream paradise fairy illusion vision visions',
  city: 'city cities street streets town downtown lights neon traffic highway road roads building buildings avenue club party subway train corner block concrete skyline taxi car',
  nature: 'tree trees forest woods flower flowers rose roses garden field fields grass leaf leaves earth mountain mountains valley spring summer autumn bloom blossom wild green seed root roots river meadow desert',
  space: 'space star stars galaxy universe planet planets moon mars rocket orbit cosmic cosmos astronaut sun comet satellite infinity gravity alien meteor nebula',
  storm: 'storm storms thunder lightning hurricane tornado wind winds rain chaos war fight fighting battle scream screaming rage angry wild shake shaking explode explosion bomb gun guns riot fury',
};

const POSITIVE = 'love happy happiness joy smile smiling laugh laughing sweet good great beautiful free freedom alive dance dancing sun sunshine bright light shine hope heaven dream fun party kiss best glad gold golden paradise celebrate peace yes wonderful amazing better friend friends home warm safe fine perfect win winning shine together forever';
const NEGATIVE = 'sad cry crying tears tear pain hurt broken break lonely alone lost lose die dead death kill hate cold dark darkness fear afraid scared sorry wrong bad goodbye gone empty nothing never nobody sick blood war fight lie lies liar cheat grave hell ghost regret miss missing bleed bleeding tired numb worst sin';
const AROUSED = 'run fire burn fight scream shout dance jump fast wild crazy rush explode storm thunder rage war gun kill party loud power fly race chase shake break crash blood energy rock go now ready electric faster higher boom bang hit';
const CALM = 'sleep dream slow quiet calm soft gentle rest peace still silence silent whisper float breathe lullaby easy lazy sunday wait stay lay lie drift drifting fade gently tender hush';

const LEX = new Map<string, { tags: number[]; v: number; a: number }>();
function entry(w: string) {
  let e = LEX.get(w);
  if (!e) LEX.set(w, (e = { tags: [], v: 0, a: 0 }));
  return e;
}
LYRIC_TAGS.forEach((t, i) => {
  for (const w of TAG_WORDS[t].split(/\s+/)) {
    const e = entry(w);
    if (!e.tags.includes(i)) e.tags.push(i);
  }
});
for (const w of POSITIVE.split(/\s+/)) entry(w).v = 1;
for (const w of NEGATIVE.split(/\s+/)) entry(w).v = -1;
for (const w of AROUSED.split(/\s+/)) entry(w).a = 1;
for (const w of CALM.split(/\s+/)) entry(w).a = -1;

const NEGATORS = new Set(['not', "don't", 'dont', 'no', 'never', "can't", 'cant', "won't", 'wont', 'nothing', "ain't", 'aint', 'without']);
const INTENSIFIERS = new Set(['so', 'very', 'too', 'really', 'always', 'all', 'forever', 'oh', 'yeah']);

/** A word as the lexicon knows it (lowercase, suffixes folded), or null when unknown. */
export function lookupWord(raw: string): { tags: number[]; v: number; a: number } | null {
  let w = raw.toLowerCase().replace(/[^a-z']/g, '');
  // Dropped g's: "burnin'" is "burning".
  if (w.endsWith("in'")) w = w.slice(0, -1) + 'g';
  w = w.replace(/^'+|'+$/g, '');
  if (!w) return null;
  const tries = [w];
  if (w.endsWith("'s")) tries.push(w.slice(0, -2));
  if (w.endsWith('ies')) tries.push(w.slice(0, -3) + 'y');
  if (w.endsWith('es')) tries.push(w.slice(0, -2));
  if (w.endsWith('s')) tries.push(w.slice(0, -1));
  if (w.endsWith('ing')) tries.push(w.slice(0, -3), w.slice(0, -3) + 'e', w.slice(0, -4));
  if (w.endsWith('ed')) tries.push(w.slice(0, -2), w.slice(0, -1), w.slice(0, -3));
  if (w.endsWith('ly')) tries.push(w.slice(0, -2));
  if (w.endsWith('er')) tries.push(w.slice(0, -2), w.slice(0, -1));
  for (const t of tries) {
    const e = LEX.get(t);
    if (e) return e;
  }
  return null;
}

/** What one line is about. */
export interface LineMeaning {
  /** Imagery tags, 0..1 each (LYRIC_TAGS order); a line naming nothing is all zeros. */
  tags: Float32Array;
  /** 0 sad .. 0.5 neutral .. 1 happy. */
  valence: number;
  /** 0 calm .. 0.5 neutral .. 1 intense. */
  arousal: number;
  /** How much the lexicon understood (0 none .. 1 several known words). */
  weight: number;
}

/** A line's meaning from its words (negation flips the valence of the next few words). */
export function readLine(text: string): LineMeaning {
  const tags = new Float32Array(TAG_COUNT);
  let v = 0;
  let a = 0;
  let nv = 0;
  let na = 0;
  let known = 0;
  let neg = 0;
  let boost = 1;
  const words = text.split(/[\s,.;:!?()"\-–—]+/).filter(Boolean);
  for (const w of words) {
    const lw = w.toLowerCase();
    if (NEGATORS.has(lw)) {
      neg = 3;
      continue;
    }
    if (INTENSIFIERS.has(lw)) {
      boost = 1.5;
      continue;
    }
    const e = lookupWord(w);
    if (e) {
      known++;
      for (const t of e.tags) tags[t] += boost;
      if (e.v) {
        v += (neg > 0 ? -0.6 : 1) * e.v * boost;
        nv++;
      }
      if (e.a) {
        a += e.a * boost;
        na++;
      }
    }
    boost = 1;
    if (neg > 0) neg--;
  }
  // "!" and shouted words raise the arousal a little.
  const bangs = (text.match(/!/g) ?? []).length;
  const caps = words.filter((w) => w.length > 2 && w === w.toUpperCase() && /[A-Z]/.test(w)).length;
  a += 0.5 * Math.min(2, bangs) + 0.5 * Math.min(2, caps);
  if (bangs || caps) na++;
  // Squash: one mention of a tag is 0.6, two about 0.85.
  for (let i = 0; i < TAG_COUNT; i++) tags[i] = tags[i] > 0 ? 1 - Math.exp(-tags[i] * 0.9) : 0;
  const sq = (x: number, n: number) => (n ? Math.tanh(x / Math.max(1, n * 0.8)) : 0);
  return {
    tags,
    valence: 0.5 + 0.5 * sq(v, nv),
    arousal: 0.5 + 0.5 * sq(a, na),
    weight: Math.min(1, known / 3),
  };
}

/** The dominant tags of a meaning, strongest first (for the HUD and tests). */
export function topTags(m: { tags: Float32Array }, n = 3, min = 0.3): LyricTag[] {
  return LYRIC_TAGS.map((t, i) => [t, m.tags[i]] as const)
    .filter(([, x]) => x >= min)
    .sort((x, y) => y[1] - x[1])
    .slice(0, n)
    .map(([t]) => t);
}
