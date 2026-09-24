// Tests for phenotype fingerprints (reference clip, features, normalisation,
// distance), visual duplicate detection and fingerprint persistence.
// Called from v2-test.ts.

import {
  CLIP, DUP_FP_DIST, FEATURES, FEATURE_COUNT, FP_VERSION, FeatureNorm, FingerprintAcc, GROUPS, ReferenceClip,
  blockFlow, clipPart, fpDistance, frameFeatures, validFingerprint,
} from '../src/v2/fingerprint';
import { Population } from '../src/v2/population';
import { Phenotype } from '../src/v2/phenotype';

type Check = (name: string, ok: boolean, detail: string) => void;

const W = 160, H = 90;
// Pixel centres: the picture's centre lies between pixels, as in GL.
const CX = (W - 1) / 2, CY = (H - 1) / 2;

/** Procedural test pictures, RGBA8. */
type Pattern = (x: number, y: number, t: number, beat: number) => [number, number, number];
const PATTERNS: Record<string, Pattern> = {
  rings: (x, y, t, beat) => {
    const r = Math.hypot(x - CX, y - CY);
    const v = 0.5 + 0.5 * Math.cos(r * 0.8 - t * 4);
    return [v * (0.5 + 0.5 * beat), v * 0.3, v];
  },
  stripes: (x, _y, t) => {
    const v = 0.5 + 0.5 * Math.sin((x + t * 40) * 0.35);
    return [v, v * 0.9, 0.2 * v];
  },
  petals: (x, y, t, beat) => {
    const a = Math.atan2(y - CY, x - CX);
    const r = Math.hypot(x - CX, y - CY);
    const v = Math.max(0, Math.cos(a * 6 + t)) * Math.exp(-r / 30) * (0.6 + 0.4 * beat);
    return [v * 0.2, v, v * 0.6];
  },
  blobs: (x, y, t) => {
    let v = 0;
    for (let i = 0; i < 7; i++) {
      const cx = W * (0.15 + 0.7 * ((i * 0.37 + t * 0.05) % 1)), cy = H * (0.2 + 0.6 * ((i * 0.61) % 1));
      v += Math.exp(-((x - cx) ** 2 + (y - cy) ** 2) / 30);
    }
    return [Math.min(1, v), Math.min(1, v * 0.5), 0.1];
  },
};

function paint(p: Pattern, t: number, beat: number, out = new Uint8Array(W * H * 4)): Uint8Array {
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    const [r, g, b] = p(x, y, t, beat);
    const i = (y * W + x) * 4;
    out[i] = Math.round(Math.max(0, Math.min(1, r)) * 255);
    out[i + 1] = Math.round(Math.max(0, Math.min(1, g)) * 255);
    out[i + 2] = Math.round(Math.max(0, Math.min(1, b)) * 255);
    out[i + 3] = 255;
  }
  return out;
}

/** Fingerprint of a pattern "rendered" on the reference clip with the same schedule as fingerprintRender.ts (coarser). */
function fingerprintOf(p: Pattern): number[] {
  const acc = new FingerprintAcc(W, H);
  const clip = new ReferenceClip();
  const total = Math.round(CLIP.end * CLIP.fps);
  for (let f = 1; f <= total; f++) {
    const s = clip.next();
    if (f % 8 !== 0) continue;
    acc.sample(paint(p, s.time, s.beatPulse), s.time, f % 32 === 0);
    if (f % 48 === 0) acc.pair(paint(p, s.time, s.beatPulse), paint(p, s.time + 1 / 60, s.beatPulse));
  }
  return acc.finish();
}

export function noveltyTests(check: Check): void {
  // ------------------------------------------------ reference clip
  {
    const a = new ReferenceClip(), b = new ReferenceClip();
    let same = true;
    let changes = 0;
    const parts = new Set<string>();
    let maxDrop = 0, maxBuild = 0;
    for (let i = 0; i < CLIP.end * CLIP.fps; i++) {
      const x = a.next(), y = b.next();
      if (x.bass !== y.bass || x.beatPulse !== y.beatPulse || x.spectrum[3] !== y.spectrum[3] || x.section.label !== y.section.label) same = false;
      if (x.sectionChanged) changes++;
      parts.add(x.section.label);
      maxDrop = Math.max(maxDrop, x.dropPulse);
      maxBuild = Math.max(maxBuild, x.buildIntensity);
    }
    a.reset();
    const first = a.next().bass;
    b.reset();
    check('fingerprint.clip', same && changes === 2 && parts.size === 3 && maxDrop > 0.9 && maxBuild > 0.9 && first === b.next().bass && clipPart(0.1) === 'calm' && clipPart(CLIP.end - 0.1) === 'drop',
      `deterministic, ${changes} section changes over ${[...parts].join('/')}, dropPulse ${maxDrop.toFixed(2)}, build ${maxBuild.toFixed(2)}, ${CLIP.end.toFixed(2)} s`);
  }

  // ------------------------------------------------ features
  const fps: Record<string, number[]> = {};
  for (const k of Object.keys(PATTERNS)) fps[k] = fingerprintOf(PATTERNS[k]);
  {
    const again = fingerprintOf(PATTERNS.rings);
    check('fingerprint.determinism', JSON.stringify(again) === JSON.stringify(fps.rings) && validFingerprint(again) && again.length === FEATURE_COUNT,
      `${FEATURE_COUNT} features in ${GROUPS.length} groups, identical on a second run`);
  }
  {
    const at = (fp: number[], key: string) => fp[FEATURES.findIndex((f) => f.key === key)];
    const bad: string[] = [];
    if (!(at(fps.rings, 'rings') > at(fps.stripes, 'rings'))) bad.push('rings');
    if (!(at(fps.petals, 'kfold') > at(fps.stripes, 'kfold') && at(fps.petals, 'kfold') > at(fps.blobs, 'kfold'))) bad.push('kfold');
    if (!(at(fps.blobs, 'blobs') > at(fps.petals, 'blobs'))) bad.push('blobs');
    if (!(at(fps.stripes, 'trans') > 0.5)) bad.push(`trans ${at(fps.stripes, 'trans')}`);
    if (!(at(fps.rings, 'rot') > 0.9 && at(fps.stripes, 'symY') > 0.9)) bad.push('symmetry');
    if (!(at(fps.rings, 'beatLum') > 0.3 && at(fps.stripes, 'beatLum') < 0.2)) bad.push(`beatLum ${at(fps.rings, 'beatLum')} / ${at(fps.stripes, 'beatLum')}`);
    if (!(at(fps.stripes, 'hueX') > at(fps.petals, 'hueX'))) bad.push('hue');
    check('fingerprint.traits', !bad.length, bad.join(', ') || 'rings, n-fold petals, blobs, drift, symmetry, beat flash and hue each show on the matching picture');
  }
  {
    // Flow: a picture shifted right by 2 px reads as drift; a zoom reads as divergence.
    const a = paint(PATTERNS.blobs, 1, 0);
    const shift = new Uint8Array(a.length);
    for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) for (let c = 0; c < 4; c++) shift[(y * W + x) * 4 + c] = a[(y * W + Math.max(0, x - 2)) * 4 + c];
    const L = (px: Uint8Array) => Float32Array.from({ length: W * H }, (_, i) => px[i * 4] / 255);
    const f1 = blockFlow(L(a), L(shift), W, H);
    const tex: Pattern = (x, y) => { const v = 0.5 + 0.5 * Math.sin(x * 0.5) * Math.cos(y * 0.6); return [v, v, v]; };
    const zoomed = paint((x, y) => tex(W / 2 + (x - W / 2) / 1.05, H / 2 + (y - H / 2) / 1.05, 0, 0), 0, 0);
    const f2 = blockFlow(L(paint(tex, 0, 0)), L(zoomed), W, H);
    check('fingerprint.flow', !!f1 && f1.trans > 1.5 && !!f2 && f2.div > 0.5, `shift: drift ${f1?.trans.toFixed(2)}; zoom: divergence ${f2?.div.toFixed(2)}`);
  }
  {
    const s = frameFeatures(paint(PATTERNS.rings, 0, 1), W, H);
    const black = frameFeatures(new Uint8Array(W * H * 4), W, H);
    const finite = Object.values(black).every(Number.isFinite) && Object.values(s).every(Number.isFinite);
    check('fingerprint.black-frame', finite && black.lum === 0 && black.blobs === 0, 'an all-black frame gives finite, zero traits');
  }

  // ------------------------------------------------ normalisation and distance
  const all = Object.values(fps);
  const norm = FeatureNorm.fit(all);
  {
    const ks = Object.keys(fps);
    let selfZero = true, symmetric = true, triangle = true, positive = true;
    for (const a of ks) for (const b of ks) {
      const d = fpDistance(fps[a], fps[b], norm);
      if (a === b && d !== 0) selfZero = false;
      if (Math.abs(d - fpDistance(fps[b], fps[a], norm)) > 1e-9) symmetric = false;
      if (a !== b && !(d > DUP_FP_DIST)) positive = false;
      for (const c of ks) if (d > fpDistance(fps[a], fps[c], norm) + fpDistance(fps[c], fps[b], norm) + 1e-9) triangle = false;
    }
    // A slightly perturbed fingerprint stays within the duplicate distance.
    const near = fps.rings.map((v, i) => v + norm.scale[i] * 0.05 * ((i % 3) - 1));
    const dn = fpDistance(fps.rings, near, norm);
    check('fingerprint.distance', selfZero && symmetric && triangle && positive && dn < DUP_FP_DIST,
      `self 0, symmetric, triangle inequality, distinct pictures > ${DUP_FP_DIST}, a 5%-of-spread wobble ${dn.toFixed(3)}`);
    const id = FeatureNorm.identity();
    const few = FeatureNorm.fit(all.slice(0, 2));
    const scalesOk = norm.scale.every((s, i) => s >= FEATURES[i].floor);
    const clipped = norm.z(fps.rings.map((v) => v + 1e6)).every((z) => z <= 4);
    check('fingerprint.norm', scalesOk && clipped && few.n === 0 && id.n === 0 && norm.n === all.length,
      `robust scales never below the floors, z clipped at 4, identity until 3 fingerprints (fitted on ${norm.n})`);
  }

  // ------------------------------------------------ duplicates and persistence
  {
    const pop = Population.seeded(1000);
    const ms = pop.list();
    const ks = Object.keys(fps);
    ms.slice(0, ks.length).forEach((m, i) => {
      m.fp = fps[ks[i]];
      m.fpv = FP_VERSION;
    });
    const ph = new Phenotype(null, () => pop);
    ph.refit(true);
    const near = fps.petals.map((v, i) => v + ph.norm.scale[i] * 0.03);
    const hit = ph.duplicateOf(near);
    const miss = ph.duplicateOf(fingerprintOf((x, y, t) => [0.5 + 0.5 * Math.sin(x * 0.9 + y * 0.9 + t * 9), 0, 0.5]));
    check('fingerprint.duplicate', hit?.id === ms[ks.indexOf('petals')].id && !miss && ph.missing().length === ms.length - ks.length,
      `near-copy of petals flagged as ${hit?.id} (${hit?.dist.toFixed(3)}); a new picture passes; ${ph.missing().length} members still to fingerprint`);

    const back = Population.fromJSON(JSON.parse(JSON.stringify(pop.toJSON())));
    const kept = back.list().filter((m) => validFingerprint(m.fp)).length;
    const stale = pop.toJSON();
    stale.members[0].fpv = FP_VERSION + 1;
    (stale.members[1] as { fp?: unknown }).fp = [1, 2, 3];
    const back2 = Population.fromJSON(JSON.parse(JSON.stringify(stale)));
    const noFp = pop.toJSON();
    for (const m of noFp.members) {
      delete m.fp;
      delete m.fpv;
    }
    const back3 = Population.fromJSON(JSON.parse(JSON.stringify(noFp)));
    // A re-encoded seed drops its fingerprint (it looks different now).
    const up = Population.fromJSON(JSON.parse(JSON.stringify(pop.toJSON())));
    const s0 = up.list()[0];
    s0.genome = { ...s0.genome, energy: [0.01, 0.02] };
    const changed = up.upgradeSeeds();
    check('fingerprint.persist', kept === ks.length && back2.list().filter((m) => validFingerprint(m.fp)).length === ks.length - 2 &&
      back3.list().every((m) => m.fp === undefined) && changed.includes(s0.id) && s0.fp === undefined,
      `${kept} fingerprints survive save/load; another version or a malformed vector is dropped; old files load without; a re-encoded seed is re-measured`);
  }
}
