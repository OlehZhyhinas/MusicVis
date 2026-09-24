// Checks for the song-as-landscape shape gene (src/v2/genes/landscape.ts) and its world map
// (src/analysis/songWorld.ts); run from v2-test.ts.

import {
  COST_BUDGET_MS, MATERIAL_KINDS, SHAPE_CLASS, SHAPE_KINDS, SHAPE_SCHEMAS, UNIQUE_SHAPES, classify, cloneGenome, estimateCost, reactable, repair,
  sdfCapable, validate, type Genome,
} from '../src/v2/genome';
import { crossover, mulberry32, mutate, randomBody, randomGene } from '../src/v2/ops';
import { SEEDS } from '../src/v2/seeds';
import { nameFor, nounKind, NOUN_POOLS } from '../src/v2/naming';
import { buildSources } from '../src/v2/glsl';
import { LAND_SCHEMA, LAND_VEC4, LAND_ALT, groundAt, landCost, landField, landK, packLandscape } from '../src/v2/genes/landscape';
import { LIVE_BACK, LIVE_SPAN, LiveWorld, SECTION_LABELS, buildSongWorld, worldAt, type SongWorld } from '../src/analysis/songWorld';
import type { AnalysisResult, MusicState, Section } from '../src/types';
import type { Frame } from '../src/v2/engine';

type Check = (name: string, ok: boolean, detail: string) => void;

/** A synthetic 3-minute song: intro, verse, build, drop, breakdown, chorus, build, drop, outro. */
export function landSong(): AnalysisResult {
  const dur = 180, rate = 20, n = dur * rate;
  const plan: [number, number, Section['label'], number][] = [
    [0, 16, 'intro', 0.2], [16, 48, 'verse', 0.45], [48, 64, 'build', 0.6], [64, 96, 'drop', 0.95], [96, 112, 'breakdown', 0.15],
    [112, 128, 'chorus', 0.7], [128, 144, 'build', 0.6], [144, 168, 'drop', 0.95], [168, 180, 'outro', 0.25],
  ];
  const sections: Section[] = plan.map(([start, end, label, energy]) => ({ start, end, label, energy }));
  const loudness = new Float32Array(n);
  const complexity = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const t = i / rate;
    const s = sections.find((x) => t >= x.start && t < x.end) ?? sections[sections.length - 1];
    const ramp = s.label === 'build' ? (t - s.start) / (s.end - s.start) : 0;
    loudness[i] = s.energy * (0.9 + 0.1 * Math.sin(t * 6)) + 0.2 * ramp;
    complexity[i] = s.energy;
  }
  const z = () => new Float32Array(n);
  const st = { drums: z(), bass: z(), vocals: z(), other: z() };
  return {
    duration: dur, frameRate: rate, numFrames: n, stems: st, stemOnsets: st, stemPresence: st, complexity, songComplexity: 0.5,
    loudness, chroma: new Float32Array(n * 12), bpm: 120, beats: new Float32Array(0), downbeats: new Float32Array(0), beatsPerBar: 4,
    sections, keys: [{ start: 0, end: 112, tonic: 9, mode: 'minor', confidence: 1 }, { start: 112, end: 180, tonic: 2, mode: 'major', confidence: 1 }],
  };
}

function frame(): Frame {
  return { speed: 1, act: 1, loud: 0.5, beatPulse: 0, gate: new Float32Array([1, 1, 1, 1]), stem: new Float32Array(4), onset: new Float32Array(4), keyHue: 0.2 } as unknown as Frame;
}

export function landscapeChecks(check: Check): void {
  const ls = SEEDS.filter((s) => /^L\d/.test(s.origin));
  const l01 = ls.find((s) => s.origin === 'L01')!.genome;

  // Seeds: valid, stable, under budget, terrain species, the journey noun.
  const seedBad: string[] = [];
  for (const s of ls) {
    const g = s.genome;
    if (validate(g).length) seedBad.push(`${s.origin}:${validate(g)[0]}`);
    if (JSON.stringify(repair(JSON.parse(JSON.stringify(g)))) !== JSON.stringify(g)) seedBad.push(`${s.origin}:round-trip`);
    if (!(estimateCost(g) < COST_BUDGET_MS)) seedBad.push(`${s.origin}:cost ${estimateCost(g).toFixed(2)}`);
    if (!g.bodies.some((b) => b.shape.kind === 'landscape')) seedBad.push(`${s.origin}:no landscape`);
    if (classify(g).primary !== 'terrain') seedBad.push(`${s.origin}:species ${classify(g).primary}`);
    if (g.reactions.length < 3) seedBad.push(`${s.origin}:reactions`);
  }
  check('landscape.seeds', !seedBad.length && ls.length >= 1, seedBad.join(' | ') || `${ls.length} L seeds valid, stable, under budget, terrain species (${ls.map((s) => `${s.origin} ${estimateCost(s.genome).toFixed(2)} ms`).join(', ')})`);
  const name = nameFor(repair({ ...cloneGenome(l01), chain: [] }));
  check('landscape.name', nounKind(l01.bodies[0]) === 'journey' && NOUN_POOLS.journey.some((w) => name.endsWith(w)), name);

  // Schema and structure: a unique field shape, one copy, reactions skip the switches.
  const b0 = l01.bodies[0];
  const two = cloneGenome(l01);
  two.bodies.push(cloneGenome(l01).bodies[0]);
  const orb = repair({ ...cloneGenome(l01), bodies: [{ ...cloneGenome(l01).bodies[0], place: { kind: 'orbit', p: { count: 4 } } }] });
  const keys = reactable(SHAPE_SCHEMAS.landscape);
  check('landscape.schema', SHAPE_KINDS.includes('landscape') && SHAPE_CLASS.landscape === 'field' && SHAPE_SCHEMAS.landscape === LAND_SCHEMA && UNIQUE_SHAPES.includes('landscape') && !sdfCapable(b0.shape)
    && repair(two).bodies.filter((b) => b.shape.kind === 'landscape').length === 1 && orb.bodies[0].place.p.count === 1
    && !['path', 'ground', 'mark', 'res', 'look'].some((k) => keys.includes(k)) && ['relief', 'glow', 'kick', 'rough', 'fog'].every((k) => keys.includes(k)),
  `unique field shape, single copy, reactable: ${keys.join(',')}`);
  const wild = repair({ ...cloneGenome(l01), bodies: [{ ...cloneGenome(l01).bodies[0], shape: { kind: 'landscape', p: { path: 9, ground: -3, look: 1000, relief: 7, fog: NaN, res: 0.1 } } }] });
  const wp = wild.bodies[0].shape.p;
  check('landscape.repair', !validate(wild).length && wp.path === 3 && wp.ground === 0 && wp.look === 40 && wp.relief === 1 && wp.fog === LAND_SCHEMA.fog.def && wp.res === 0.35 && wp.mark === LAND_SCHEMA.mark.def, JSON.stringify(wp));

  // Shaders: the pass exists only with a landscape body; every material samples the landscape texture.
  const src = buildSources(l01);
  check('landscape.glsl', !!src.land && src.land.includes('lsH(') && src.land.includes('uWorld') && src.land.includes('uLs[') && src.feedback.includes('uLand') && !src.scene
    && !buildSources(SEEDS[0].genome).land && !landField('fill').includes('uScene') && landField('fill').includes('texture(uLand'),
  'land pass built for landscape genomes only; the body samples uLand');
  const matBad: string[] = [];
  for (const m of MATERIAL_KINDS) {
    const g = cloneGenome(l01);
    g.bodies[0].material = { kind: m, p: {} };
    const r = repair(g);
    const s = buildSources(r);
    if (validate(r).length || !s.land || !(s.feedback + s.composite).includes('texture(uLand')) matBad.push(m);
  }
  check('landscape.every-material', !matBad.length, matBad.join(',') || `${MATERIAL_KINDS.length} materials light the landscape`);

  // Cost: grows with the internal resolution; over budget, repair lowers the resolution.
  const lo = landCost({ ...b0.shape.p, res: 0.35 });
  const hi = landCost({ ...b0.shape.p, res: 0.7 });
  const heavy = cloneGenome(l01);
  heavy.bodies[0].shape.p.res = 0.7;
  heavy.bodies[0].shape.p.ground = 3;
  heavy.chain = Array.from({ length: 6 }, () => ({ op: 'noise' as const, stage: 'warp' as const, w: 1, p: { amp: 0.001, scale: 2, speed: 0.3 } }));
  const fixed = repair(heavy);
  check('landscape.cost', hi > lo * 3 && fixed.bodies[0].shape.p.res < 0.7 && estimateCost(repair({ ...cloneGenome(l01), chain: [] })) < COST_BUDGET_MS,
    `res 0.35 ${lo.toFixed(2)} ms, 0.5 ${landCost({ ...b0.shape.p, res: 0.5 }).toFixed(2)} ms, 0.7 ${hi.toFixed(2)} ms; heavy genome repaired to res ${fixed.bodies[0].shape.p.res}`);

  // The world map of a song: climbs to each drop, a pass at the drop, a cliff after it, valleys in
  // breakdowns, one motif per section type, the key there.
  const song = landSong();
  const w = buildSongWorld(song);
  const alt = (t: number) => worldAt(w, t, 0);
  const worldOk = alt(63) > alt(50) + 0.2 && worldAt(w, 64, 1) > 0.95 && worldAt(w, 80, 1) < 0.01 && alt(66) < alt(63.5) && alt(104) < alt(80) - 0.4
    && worldAt(w, 120, 2) === SECTION_LABELS.indexOf('chorus') && worldAt(w, 150, 2) === worldAt(w, 70, 2) && Math.abs(worldAt(w, 150, 3) - 6) < 0.3
    && worldAt(w, 50, 7) < worldAt(w, 70, 7) && worldAt(w, 30, 5) === 1 && worldAt(w, 130, 5) === 0 && worldAt(w, 30, 4) !== worldAt(w, 130, 4)
    && w.drops.join() === '64,144' && w.marks.length === 8 && [...w.data].every(Number.isFinite);
  check('landscape.world', worldOk,
    `alt verse ${alt(30).toFixed(2)}, build ${alt(50).toFixed(2)} -> pass ${alt(63.5).toFixed(2)}, after the cliff ${alt(66).toFixed(2)}, drop ${alt(80).toFixed(2)}, breakdown ${alt(104).toFixed(2)}; drops ${w.drops.join(',')}`);

  // Camera: finite and above the ground over the whole song for every path style; the next drop's
  // sun rises before it and the landmarks within sight are passed.
  const camBad: string[] = [];
  let sunBefore = 0, sunFar = 1, marksSeen = 0;
  for (const path of LAND_SCHEMA.path.choices!) {
    const p = { ...b0.shape.p, path, height: 0, relief: 1, wind: 1 };
    const out = new Float32Array(LAND_VEC4 * 4);
    const mem: Record<string, number> = {};
    const F = frame();
    let lastZ = -Infinity;
    for (let i = 0; i < 180 * 30; i++) {
      const now = i / 30;
      F.onset[0] = i % 15 === 0 ? 1 : 0;
      packLandscape(out, { F, sdt: 1 / 30, P: (k) => p[k], raw: p, mem, key: 'b0.', world: w, now, keyHue: 0.2 });
      if (![...out].every(Number.isFinite)) { camBad.push(`path ${path}: not finite at ${now}`); break; }
      const gy = groundAt(w, now, 1) - (path === 1 ? 0.3 : 0);
      if (out[1] < gy + 0.05) { camBad.push(`path ${path}: under ground at ${now.toFixed(1)} (${out[1].toFixed(2)} < ${gy.toFixed(2)})`); break; }
      if (out[2] < lastZ) { camBad.push(`path ${path}: went backwards`); break; }
      lastZ = out[2];
      if (path === 0 && Math.abs(now - 60) < 1e-6) sunBefore = out[31];
      if (path === 0 && Math.abs(now - 20) < 1e-6) sunFar = out[31];
      if (path === 0 && out[27] > 0) marksSeen++;
    }
    if (Math.abs(out[2] - 179.97 * landK(p.look)) > 0.01) camBad.push(`path ${path}: z ${out[2]} not song time * k`);
  }
  check('landscape.camera', !camBad.length && sunBefore > 0.5 && sunFar === 0 && marksSeen > 1000,
    camBad.join(' | ') || `4 path styles ride above the ground for the whole song; drop sun 4 s before ${sunBefore.toFixed(2)}, 44 s before ${sunFar}; landmarks in sight ${marksSeen} frames`);

  // Live input: no future, so a moving window generated from history; samples keep their value once made.
  const live = new LiveWorld();
  const st = { time: 0, loudness: 0.4, buildIntensity: 0, dropPulse: 0, sectionIndex: 0, section: { start: 0, end: 30, label: 'verse', energy: 0.4 }, keyHue: 0.3, keyMode: 'major', complexity: 0.5 } as unknown as MusicState;
  const snap = (lw: SongWorld, t: number) => worldAt(lw, t, 0);
  let stableBad = 0;
  let ref = NaN;
  for (let i = 0; i < 60 * 40; i++) {
    st.time = i / 60;
    st.loudness = 0.3 + 0.3 * Math.sin(st.time * 0.7);
    live.update(st, 1 / 60);
    // A sample 30 s ahead at t = 10 s (generated then) must read the same when the camera reaches it.
    if (Math.abs(st.time - 10) < 1e-6) ref = snap(live.world, 40);
    if (st.time > 10 && st.time < 36 && Math.abs(snap(live.world, 40) - ref) > 1e-5) stableBad++;
  }
  const lw = live.world;
  const winOk = lw.live && Math.abs(lw.t0 - (st.time - LIVE_BACK)) < 0.3 && lw.span === LIVE_SPAN && [...lw.data].every(Number.isFinite);
  const flat = worldAt(lw, st.time + 6, 0);
  st.buildIntensity = 1;
  for (let i = 0; i < 180; i++) { st.time += 1 / 60; live.update(st, 1 / 60); }
  const risen = worldAt(lw, st.time + 2.9, 0);
  check('landscape.live', winOk && !stableBad && risen > flat + 0.2,
    `window ${lw.t0.toFixed(1)}..${(lw.t0 + lw.span).toFixed(1)} at ${st.time.toFixed(1)} s; ground ahead unchanged as the camera arrives (${stableBad} drifts); a build raises the ground ahead ${flat.toFixed(2)} -> ${risen.toFixed(2)}`);

  // Breeding: random genes in range; landscape bodies breed with every species (both ways) and mutate validly.
  const rng = mulberry32(4242);
  let rangeBad = 0;
  for (let i = 0; i < 200; i++) {
    const s = randomGene('shape', rng, 'landscape');
    for (const [k, sp] of Object.entries(LAND_SCHEMA)) if (!(s.p[k] >= sp.min && s.p[k] <= sp.max)) rangeBad++;
  }
  const bySpecies = new Map<string, Genome>();
  for (const s of SEEDS) {
    const sp = classify(s.genome).primary;
    if (!bySpecies.has(sp) && !s.genome.bodies.some((b) => b.shape.kind === 'landscape')) bySpecies.set(sp, s.genome);
  }
  const crossBad: string[] = [];
  let crosses = 0, kept = 0;
  for (const [sp, other] of bySpecies) for (const [a, b] of [[l01, other], [other, l01]]) {
    for (let k = 0; k < 6; k++) {
      const c = crossover(a, b, rng);
      crosses++;
      if (c.bodies.some((x) => x.shape.kind === 'landscape')) kept++;
      if (validate(c).length) crossBad.push(`${sp}:${validate(c)[0]}`);
      if (!(estimateCost(c) < COST_BUDGET_MS)) crossBad.push(`${sp}:cost ${estimateCost(c).toFixed(2)}`);
      const mu = mutate(c, rng, 1.5);
      if (validate(mu).length) crossBad.push(`${sp}:mutant ${validate(mu)[0]}`);
    }
  }
  check('landscape.breeds-every-species', !crossBad.length && !rangeBad && kept > crosses * 0.2,
    crossBad.slice(0, 6).join(' | ') || `${crosses} crossovers with ${bySpecies.size} species valid and under budget (${kept} keep the landscape), random genes in range`);
  let appeared = 0;
  for (let i = 0; i < 600; i++) if (randomBody(rng).shape.kind === 'landscape') appeared++;
  check('landscape.random-bodies', appeared > 3 && appeared < 100, `${appeared}/600 random bodies are landscapes`);
  void LAND_ALT;
}
