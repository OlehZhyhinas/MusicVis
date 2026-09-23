// Enhanced-engine presets. Each preset carries its own GLSL (MilkDrop style):
//   warp(p)            where to read last frame from (feedback motion)
//   draw(p, uv, prev)  what to add into the feedback buffer this frame
//   comp(uv, p)        how the feedback buffer becomes the displayed image
//   curve(k, inst)     optional line geometry (vertex shader)
// plus an optional JS hook that fills preset uniforms (uV, uSeg) each frame.
// Programs are compiled lazily per preset; see programs.ts and shaders.ts for
// the uniforms and helpers available to the snippets.
//
// Stable IDs: never reuse a retired ID; new presets take the next number.
//
//   ID   Name               Archetype                         Energy (complexity range)
//   E01  Night Skyline      horizontal scroll (city skyline)   0.30-0.80
//   E02  River of Light     two wandering snakes (melody, bass) 0.00-0.45
//   E03  Rain Curtains      vertical fall (spectral rain)      0.40-0.95
//   E04  Rising Smoke       vertical rise (smoke plumes)       0.10-0.60
//   E05  Wandering Vortex   off-centre vortex (Lissajous)      0.50-1.00
//   E06  Ink Garden         pure fluid ink                     0.15-0.75
//   E07  Oscilloscope       single waveform line, minimal      0.00-0.40
//   E08  Contour Plasma     domain-warped noise, no feedback   0.30-0.85
//   E09  Rorschach          left/right mirror inkblot          0.20-0.70
//   E10  Hex Keys           hex tiles lit by chroma            0.40-0.90
//   E11  Sunrise Arc        bottom-anchored radial spectrum    0.35-0.85
//   E12  Warp Speed         starfield (the particle preset)    0.55-1.00
//   E13  Constellations     stars linked by lines, dark        0.00-0.45
//   E14  Polyhedra          wireframe solids, 1 turn per bar   0.35-0.90
//   E15  Liquid Chrome      metaballs, bass driven             0.30-0.80
//   E16  Aurora             vocal-driven curtains              0.10-0.60
//   E17  Hyperspace         zoom tunnel                        0.60-1.00
//   E18  Mandala            radial kaleidoscope                0.50-1.00
//   E19  Harmonograph       damped Lissajous figure, minimal   0.05-0.50
//   E20  Moonrise           moon over water, dark              0.00-0.45
//   E21  Retro Grid         perspective grid and sun           0.50-1.00
//   E22  Silk Flame         fractal flame, julia/spherical     0.05-0.50
//   E23  Ember Flame        fractal flame, drop morphs         0.55-1.00
//   E24  Spiral Nebula      fractal flame, spiral/disc         0.30-0.75

import { FLAME_COMP, type FlameSpec } from './flame';

export type PaletteScheme = 'analogous' | 'complementary' | 'triad' | 'split' | 'mono';

export const SPAWN = { anywhere: 0, waveform: 1, emitters: 2, ring: 3, center: 4, bottom: 5 } as const;

/** Per-frame music values for preset JS hooks (computed once per frame). */
export interface Frame {
  time: number;
  dt: number;
  phase: number; // activity-scaled motion clock
  act: number; // activity budget 0..1 (from complexity)
  cx: number;
  speed: number;
  spin: number; // accumulated bar-locked angle (1 turn per bar at full activity)
  bars: number; // continuous bar count, wraps at 48
  beats: number; // continuous beat count, wraps at 256
  barIndex: number;
  beatIndex: number;
  barPhase: number;
  beatPhase: number;
  beatPulse: number;
  onBeat: boolean;
  onBar: boolean;
  /** drums, bass, vocals, other: stem level gated by absolute presence. */
  stem: Float32Array;
  onset: Float32Array;
  gate: Float32Array;
  loud: number;
  melody: number; // 0..1 pitch height of the lead line
  build: number;
  drop: number;
  keyTonic: number;
  minor: boolean;
  sectionIndex: number;
  aspect: number;
  /** Strength of a drum hit this frame (0 when none). */
  hit: number;
  /** True on the frame a drop section starts. */
  dropStart: boolean;
}

/** Per-preset instance state that its hook writes and its shaders read. */
export interface Runtime {
  v: Float32Array; // uV[8]
  seg: Float32Array; // uSeg[48]
  segZ: Float32Array; // 48 values
  segN: number;
  ringCenter: [number, number];
  mem: Record<string, number>;
  curveBright: number; // multiplier on the curve layer this frame
}

export interface Effects {
  /** Fluid splat, uv coordinates. type: 0 directional, 1 radial, 2 swirl. */
  splat(x: number, y: number, fx: number, fy: number, radius: number, type: number): void;
}

export interface CurveSpec {
  glsl: string;
  color?: string;
  n: number;
  instances?: number;
  thick: number; // half-width in pixels at 1080p
  bright: number;
  target?: 'fb' | 'top';
}

export interface ParticleSpec {
  count: number;
  size: number; // pixels at 1080p
  bright: number;
  spawn: number;
  target: 'fb' | 'top';
  speed: number;
  curl: number;
  life: number;
  lift?: [number, number];
  zoomFlow?: number;
  fluid?: number;
  drag?: number;
  spread?: number;
  minAct?: number; // activity at which particles start to appear
}

export interface Preset {
  id: string;
  name: string;
  kind: string;
  energy: [number, number];
  palette: PaletteScheme;
  hue?: number;
  sat?: number;
  feedback?: boolean; // default true
  decay?: number; // per 60 fps frame
  floor?: number; // black-level subtraction multiplier
  blur?: number;
  wrap?: boolean;
  /** Content velocity in p units per second (whole-pixel shifts, no resample blur). */
  scroll?: [number, number];
  warp?: string;
  draw?: string;
  comp?: string;
  curve?: CurveSpec;
  fluid?: { amount: number; vorticity: number; noise: number };
  rings?: number;
  particles?: ParticleSpec;
  /** Fractal flame plotted into the feedback buffer (see flame.ts). */
  flame?: FlameSpec;
  bloom?: number;
  exposure?: number;
  vignette?: number;
  adapt?: number; // auto-exposure strength
  js?: (f: Frame, r: Runtime, fx: Effects) => void;
}

const TAU = Math.PI * 2;
const clamp01 = (x: number) => Math.min(1, Math.max(0, x));
const approach = (cur: number, target: number, rate: number, dt: number) => cur + (target - cur) * (1 - Math.exp(-rate * dt));
function h11(n: number): number {
  const x = Math.sin(n * 127.1 + 311.7) * 43758.5453;
  return x - Math.floor(x);
}
function mem(r: Runtime, k: string, init = 0): number {
  const v = r.mem[k];
  return v === undefined ? (r.mem[k] = init) : v;
}
/** Melodic level: vocals, else "other", with a little loudness so solo pieces still draw. */
function melodic(f: Frame): number {
  return Math.max(f.stem[2], f.stem[3] * 0.85, f.loud * 0.35);
}

// Hex lattice helpers shared by E10's draw and comp.
const HEX = /* glsl */ `
vec4 hexCell(vec2 p) {
  const vec2 s = vec2(1.0, 1.7320508);
  vec4 hC = floor(vec4(p, p - vec2(0.5, 1.0)) / s.xyxy) + 0.5;
  vec4 h = vec4(p - hC.xy * s, p - (hC.zw + 0.5) * s);
  return dot(h.xy, h.xy) < dot(h.zw, h.zw) ? vec4(h.xy, hC.xy) : vec4(h.zw, hC.zw + 0.5);
}
float hexD(vec2 q) { q = abs(q); return max(dot(q, vec2(0.5, 0.8660254)), q.x); }
vec2 hexP(vec2 p) { return rot2(uSpin * 0.0625) * p * 5.0; }
`;

// Wireframe solids for E14.
type Solid = { v: number[][]; e: [number, number][] };
function edgesByLength(v: number[][]): [number, number][] {
  let min = Infinity;
  const d = (a: number[], b: number[]) => Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
  for (let i = 0; i < v.length; i++) for (let j = i + 1; j < v.length; j++) min = Math.min(min, d(v[i], v[j]));
  const e: [number, number][] = [];
  for (let i = 0; i < v.length; i++) for (let j = i + 1; j < v.length; j++) if (d(v[i], v[j]) < min * 1.01) e.push([i, j]);
  return e;
}
function norm(v: number[][]): number[][] {
  return v.map((p) => {
    const l = Math.hypot(p[0], p[1], p[2]);
    return [p[0] / l, p[1] / l, p[2] / l];
  });
}
const PHI = (1 + Math.sqrt(5)) / 2;
const SOLIDS: Solid[] = (() => {
  const cube: number[][] = [];
  for (let i = 0; i < 8; i++) cube.push([i & 1 ? 1 : -1, i & 2 ? 1 : -1, i & 4 ? 1 : -1]);
  const oct = [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]];
  const ico: number[][] = [];
  for (const a of [-1, 1]) for (const b of [-PHI, PHI]) ico.push([0, a, b], [a, b, 0], [b, 0, a]);
  const tet = [[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]];
  return [cube, ico, oct, tet].map((v) => {
    const n = norm(v);
    return { v: n, e: edgesByLength(n) };
  });
})();
const solidTmp = new Float32Array(3 * 12);

function projectSolid(
  s: Solid, r: Runtime, start: number, ang: number, tilt: number, scale: number, weight: number,
): number {
  const ca = Math.cos(ang), sa = Math.sin(ang), ct = Math.cos(tilt), st = Math.sin(tilt);
  for (let i = 0; i < s.v.length; i++) {
    const [x0, y0, z0] = s.v[i];
    // spin about Y, then tilt about X
    const x1 = x0 * ca + z0 * sa;
    const z1 = -x0 * sa + z0 * ca;
    const y2 = y0 * ct - z1 * st;
    const z2 = y0 * st + z1 * ct;
    const k = 1 / (1 + z2 * 0.3);
    solidTmp[i * 3] = x1 * scale * k;
    solidTmp[i * 3 + 1] = y2 * scale * k;
    solidTmp[i * 3 + 2] = z2;
  }
  let n = start;
  for (const [a, b] of s.e) {
    if (n >= 48) break;
    r.seg[n * 4] = solidTmp[a * 3];
    r.seg[n * 4 + 1] = solidTmp[a * 3 + 1];
    r.seg[n * 4 + 2] = solidTmp[b * 3];
    r.seg[n * 4 + 3] = solidTmp[b * 3 + 1];
    const z = (solidTmp[a * 3 + 2] + solidTmp[b * 3 + 2]) * 0.5;
    r.segZ[n] = weight * (0.35 + 0.65 * clamp01(0.5 - z * 0.5));
    n++;
  }
  return n;
}

const RATIOS: [number, number][] = [[2, 3], [3, 4], [3, 5], [4, 5], [2, 5], [5, 6]];

export const PRESETS: Preset[] = [
  // ------------------------------------------------------------ E01
  {
    id: 'E01', name: 'Night Skyline', kind: 'horizontal scroll', energy: [0.3, 0.8],
    palette: 'complementary', hue: 0.05, decay: 0.9995, floor: 0.05, scroll: [-0.2, 0], adapt: 0.3,
    js(f, r) {
      // A new building every beat, sometimes two beats wide.
      const b = f.beatIndex >= 0 ? f.beatIndex : Math.floor(f.time * 2);
      let start = mem(r, 'start', b);
      let wide = mem(r, 'wide', 1);
      if (b >= start + wide || b < start) {
        start = r.mem.start = b;
        wide = r.mem.wide = h11(b * 1.7) < 0.3 ? 2 : 1;
        const lv = 0.6 * f.loud + 0.4 * Math.max(f.stem[1], f.stem[3]);
        r.mem.h = (0.04 + 0.3 * Math.pow(h11(b * 3.3), 1.6) + 0.12 * lv) * (0.6 + 0.4 * f.act);
        r.mem.cols = 3 + Math.floor(h11(b * 5.1) * 3) * wide;
        r.mem.lit = 0.08 + 0.3 * Math.max(f.stem[3], f.stem[2]) + 0.15 * f.act;
      }
      const frac = (b - start + (f.beatIndex >= 0 ? f.beatPhase : (f.time * 2) % 1)) / wide;
      r.v[0] = mem(r, 'h', 0.1);
      r.v[1] = start;
      r.v[2] = mem(r, 'lit', 0.3);
      r.v[3] = frac;
      r.v[4] = mem(r, 'cols', 4);
    },
    draw: /* glsl */ `
vec3 draw(vec2 p, vec2 uv, vec3 prev) {
  float A = uAspect * 0.5;
  if (p.x < A - abs(uShift.x) - 2.0 * px()) return vec3(0.0);
  float base = -0.16;
  float h = uV[0].x, id = uV[0].y, lit = uV[0].z, fr = uV[0].w;
  float y = p.y - base;
  if (fr < 0.06 || y < 0.0) return vec3(0.0);
  vec3 c = vec3(0.0);
  if (y < h) {
    c = uColB * 0.012 + uColC * 0.01 * (1.0 - y / max(h, 1e-3));
    float row = floor(y / 0.022);
    float inRow = step(0.35, fract(y / 0.022));
    float inCol = step(0.4, fract(fr * uV[1].x));
    float on = step(hash12(vec2(id, row)), lit) * inRow * inCol * step(y, h - 0.014) * step(0.1, fr) * step(fr, 0.94);
    c += mix(uColA, vec3(1.0, 0.8, 0.5), 0.5) * on * (0.25 + 0.5 * hash12(vec2(row, id * 3.1)));
  }
  c += uColC * glow(y - h, 0.003) * (0.5 + 1.2 * uBeatPulse * uPres.x) * step(0.08, fr);
  return c;
}`,
    comp: /* glsl */ `
vec3 comp(vec2 uv, vec2 p) {
  float base = -0.16;
  // The whole skyline is live: each screen column is a spectrum band, so
  // buildings stretch taller as they scroll through loud bands, jump on the
  // beat, and their windows brighten with that band. Composite only, so the
  // stretch never accumulates in the feedback.
  float A = uAspect * 0.5;
  float bin = clamp((p.x + A) / (2.0 * A), 0.0, 1.0);
  bin = (floor(bin * 32.0) + 0.5) / 32.0; // whole strips stretch, so windows stay square
  float lv = specAt(bin * 0.8 + 0.03);
  float stretch = 1.0 + 0.9 * lv * (0.4 + 0.6 * uAct) + 0.18 * uBeatPulse * uPres.x;
  float win = 0.7 + 1.4 * lv;
  vec3 sky = mix(uColC * (0.025 + 0.05 * uStem.y * uPres.y), uColB * 0.002, smoothstep(base, 0.5, p.y));
  vec3 c;
  if (p.y >= base) {
    c = fb(vec2(uv.x, base + (p.y - base) / stretch + 0.5)) * win + sky;
  } else {
    float d = base - p.y;
    float rx = uv.x + (0.002 + d * 0.02) * sin(d * 260.0 - uPhase * 3.0);
    c = fb(vec2(rx, base + d / stretch + 0.5)) * win * 0.3 * exp(-d * 4.0) + uColB * 0.004;
  }
  c += uColC * glow(p.y - base, 0.0012) * 0.08;
  return c;
}`,
  },
  // ------------------------------------------------------------ E02
  {
    id: 'E02', name: 'River of Light', kind: 'wandering trails', energy: [0, 0.45],
    palette: 'analogous', hue: 0.55, decay: 0.998, floor: 0.4, adapt: 0.25, bloom: 1.2,
    js(f, r) {
      // Two independent snakes roam the screen. The melody snake turns every
      // second beat and sharply on heavy drum hits; the bass snake turns on
      // each downbeat and sharply on strong bass hits. Turns are at any angle,
      // they curve gently in between and bounce off the edges. Both move a
      // fixed distance per beat, so their speed follows the tempo.
      const A = f.aspect * 0.5;
      const M = 0.07;
      const prevBeats = mem(r, 'lastBeats', f.beats);
      let db = f.beats - prevBeats;
      if (db < 0) db += 256;
      if (db > 2) db = f.dt * 2;
      r.mem.lastBeats = f.beats;
      const hitRise = f.hit > 0.7 && mem(r, 'hitPrev') <= 0.7;
      r.mem.hitPrev = f.hit;
      const bassRise = f.onset[1] > 0.6 && mem(r, 'bassPrev') <= 0.6;
      r.mem.bassPrev = f.onset[1];
      const snakes: [string, boolean, boolean, number, number, number][] = [
        // key, turn now, heavy turn, start x, start y, distance per beat
        ['m', f.onBeat && f.beatIndex % 2 === 0, hitRise, -0.3 * A, 0.15, 0.13 * (0.7 + 0.5 * f.act)],
        ['b', f.onBar, bassRise, 0.3 * A, -0.2, 0.09 * (0.7 + 0.5 * f.act)],
      ];
      for (const [k, turnNow, heavy, sx, sy, perBeat] of snakes) {
        const x = mem(r, k + 'x', sx);
        const y = mem(r, k + 'y', sy);
        let ang = mem(r, k + 'a', k === 'm' ? 0.3 : 2.6);
        r.mem[k + 'px'] = x;
        r.mem[k + 'py'] = y;
        if (turnNow || heavy) {
          const n = (r.mem[k + 'n'] = mem(r, k + 'n') + 1);
          const side = h11(n * 7.3 + (k === 'm' ? 0 : 50)) < 0.5 ? -1 : 1;
          // Any angle; heavy hits turn sharper.
          const amt = (heavy ? 1.3 : 0.45) + (heavy ? 1.2 : 0.9) * h11(n * 3.7 + (k === 'm' ? 1 : 60));
          ang += side * amt;
        }
        // Gentle curving between turns, steered by the melody (or the bass).
        ang += ((k === 'm' ? f.melody : f.stem[1]) - 0.5) * 0.8 * f.dt;
        const step = db * perBeat;
        let nx = x + Math.cos(ang) * step;
        let ny = y + Math.sin(ang) * step;
        // Bounce off the edges.
        if (nx < -A + M || nx > A - M) { ang = Math.PI - ang; nx = Math.min(A - M, Math.max(-A + M, nx)); }
        if (ny < -0.5 + M || ny > 0.5 - M) { ang = -ang; ny = Math.min(0.5 - M, Math.max(-0.5 + M, ny)); }
        r.mem[k + 'a'] = ang;
        r.mem[k + 'x'] = nx;
        r.mem[k + 'y'] = ny;
      }
      r.v[0] = r.mem.mx;
      r.v[1] = r.mem.my;
      r.v[2] = r.mem.mpx;
      r.v[3] = r.mem.mpy;
      r.v[4] = 0.006 + 0.008 * f.loud;
      r.v[5] = 0.25 + melodic(f) * 0.55;
      r.v[6] = r.mem.bx;
      r.v[7] = r.mem.by;
      r.v[8] = r.mem.bpx;
      r.v[9] = r.mem.bpy;
      r.v[10] = 0.011 + 0.02 * f.stem[1];
      r.v[11] = 0.25 + 0.6 * f.stem[1] * (0.3 + 0.7 * f.act);
    },
    warp: /* glsl */ `
vec2 warp(vec2 p) {
  // The whole trail drifts slowly like smoke on a current.
  vec2 c = curlNoise(p * 1.6, uPhase * 0.15);
  return p + c * (0.0006 + 0.0012 * uAct) * uF60;
}`,
    draw: /* glsl */ `
vec3 draw(vec2 p, vec2 uv, vec3 prev) {
  // Each snake's newest stretch of body replaces whatever older trail it
  // crosses (the melody snake draws last, so it lies on top).
  vec3 base = prev;
  float wb = uV[2].z;
  vec3 cb = pal(0.5 + 0.4 * uStem.y + 0.15 * uBarPulse) * uV[2].w;
  float kb = smoothstep(wb, wb * 0.4, sdSeg(p, uV[2].xy, uV[1].zw));
  base = mix(base, cb, kb);
  float wm = uV[1].x;
  vec3 cm = pal(uMelody * 0.7 + 0.2 * uBeatPulse) * uV[1].y;
  float km = smoothstep(wm, wm * 0.4, sdSeg(p, uV[0].zw, uV[0].xy));
  base = mix(base, cm, km);
  base += vec3(1.0) * glow(length(p - uV[0].xy), wm * 1.3) * uV[1].y * 0.25;
  base += cb * glow(length(p - uV[1].zw), wb * 1.3) * 0.3;
  return base - prev;
}`,
    comp: /* glsl */ `
vec3 comp(vec2 uv, vec2 p) {
  vec3 bg = uColB * 0.003 * (1.0 - abs(p.y) * 1.6);
  // The trail pulses with the beat and the bass.
  float pulse = 0.85 + 0.35 * uBeatPulse + 0.25 * uStem.y * uPres.y;
  return fb(uv) * pulse + max(bg, 0.0);
}`,
  },
  // ------------------------------------------------------------ E03
  {
    id: 'E03', name: 'Rain Curtains', kind: 'vertical fall', energy: [0.4, 0.95],
    palette: 'analogous', hue: 0.5, decay: 0.962, scroll: [0, -0.4], bloom: 1.1,
    js(f, r) {
      const fl = mem(r, 'flash', 0);
      const rising = f.hit > 0.6 && mem(r, 'hitPrev', 0) <= 0.6;
      r.mem.hitPrev = f.hit;
      r.mem.flash = rising && f.act > 0.5 ? 1 : fl * Math.exp(-f.dt * 6);
      r.v[0] = r.mem.flash;
      r.v[1] = 0.35 + 0.65 * f.act;
    },
    warp: /* glsl */ `
vec2 warp(vec2 p) {
  float m = 1.0 + floor(hash11(floor((p.x + 5.0) * 30.0) * 1.37) * 2.0);
  return p + uShift * m;
}`,
    draw: /* glsl */ `
vec3 draw(vec2 p, vec2 uv, vec3 prev) {
  if (p.y < 0.5 - abs(uShift.y) * 2.0 - 3.0 * px()) return vec3(0.0);
  float col = floor((p.x + 5.0) * 30.0);
  float xin = fract((p.x + 5.0) * 30.0) - 0.5;
  float m = 1.0 + floor(hash11(col * 1.37) * 2.0);
  float bin = abs(p.x) / (uAspect * 0.5);
  float lv = specAt(bin * 0.75 + 0.03);
  // Drops are born on an eighth-note grid so rows of rain fall together in
  // time; the first eighth of each bar fires a fuller, brighter curtain.
  float e = uBeats * 2.0;
  float slot = floor(e);
  float win = step(fract(e), 0.14 + 0.08 * m);
  float down = step(uBar, 0.125);
  float thr = mix(0.03, 0.45, hash12(vec2(col, slot)));
  float fire = step(thr, lv * uV[0].y * (1.0 + 0.8 * down));
  float bright = (0.6 + 1.6 * lv) * (0.7 + 0.9 * uOnset.x + 0.6 * down);
  vec3 c = pal(bin * 0.5 + 0.1 * m) * fire * win * bright * glow(xin, 0.055) * (0.5 + 0.35 * m);
  return c * 1.6;
}`,
    comp: /* glsl */ `
vec3 comp(vec2 uv, vec2 p) {
  vec3 c = fb(uv);
  // Splash line at the bottom pulses with the bass; drops flash the whole curtain.
  c += uColC * glow(p.y + 0.5, 0.012) * (uLoud * 0.1 + uStem.y * uPres.y * 0.35 + uBeatPulse * 0.12);
  c += mix(uColC, vec3(1.0), 0.6) * uV[0].x * 0.05 * smoothstep(-0.5, 0.5, p.y);
  return c;
}`,
  },
  // ------------------------------------------------------------ E04
  {
    id: 'E04', name: 'Rising Smoke', kind: 'vertical rise', energy: [0.1, 0.6],
    palette: 'split', hue: 0.02, decay: 0.9935, blur: 0.3, adapt: 0.4,
    particles: {
      count: 1536, size: 5, bright: 1, spawn: SPAWN.bottom, target: 'top', speed: 0.6, curl: 0.1, life: 0.55,
      lift: [0, 0.12], drag: 1.5, minAct: 0.35,
    },
    warp: /* glsl */ `
vec2 warp(vec2 p) {
  vec2 c = curlNoise(p * 1.8 + vec2(0.0, -uPhase * 0.25), uPhase * 0.3);
  // Rise fast enough to reach the top before fading; beats and bass push it up.
  float rise = (0.0048 + 0.004 * uBeatPulse + 0.003 * uStem.y * uPres.y) * uSpeed;
  return p - vec2(p.x * 0.0012, rise) * uF60 + c * 0.0014 * uF60;
}`,
    draw: /* glsl */ `
vec3 draw(vec2 p, vec2 uv, vec3 prev) {
  vec3 c = vec3(0.0);
  for (int i = 0; i < 4; i++) {
    float fi = float(i);
    float x = (fi - 1.5) * 0.3 * uAspect * 0.5 + 0.05 * sin(uPhase * 0.4 + fi * 2.1);
    float lv = i == 0 ? uOnset.x : uStem[i];
    vec2 d = p - vec2(x, -0.5);
    c += pal(fi * 0.27 + 0.05) * lv * glow(length(d * vec2(1.0, 0.5)), 0.045 + 0.03 * lv);
  }
  return c * 0.06;
}`,
    comp: /* glsl */ `
vec3 comp(vec2 uv, vec2 p) {
  vec3 c = fb(uv);
  return c * (0.8 + 0.2 * smoothstep(0.5, 0.2, p.y));
}`,
  },
  // ------------------------------------------------------------ E05
  {
    id: 'E05', name: 'Wandering Vortex', kind: 'off-centre vortex', energy: [0.5, 1],
    palette: 'triad', hue: 0.7, decay: 0.986, rings: 0.3, bloom: 1.1,
    js(f, r) {
      const cx = 0.32 * f.aspect * 0.5 * Math.sin((TAU * f.bars) / 8);
      const cy = 0.22 * Math.sin((TAU * f.bars) / 6 + 1);
      r.v[0] = cx;
      r.v[1] = cy;
      r.v[2] = Math.max(f.stem[3], f.stem[2]) * 0.8 + 0.2 * f.loud;
      r.ringCenter[0] = cx;
      r.ringCenter[1] = cy;
    },
    warp: /* glsl */ `
vec2 warp(vec2 p) {
  vec2 c = uV[0].xy;
  vec2 d = p - c;
  float r = length(d);
  float z = 1.0 + (0.004 + 0.006 * uStem.y + 0.01 * uBeatPulse * uPres.x) * uSpeed * uF60;
  d = rot2(-uSpinStep * 0.25 - 0.01 * uF60 * uSpeed / (r * 6.0 + 0.4)) * d * z;
  return c + d;
}`,
    draw: /* glsl */ `
vec3 draw(vec2 p, vec2 uv, vec3 prev) {
  vec2 c = uV[0].xy;
  vec3 col = vec3(0.0);
  for (int i = 0; i < 2; i++) {
    float a = uSpin * 0.5 + float(i) * PI;
    vec2 cp = c + 0.3 * vec2(cos(a), sin(a));
    col += (i == 0 ? uColA : uColC) * glow(length(p - cp), 0.014 + 0.01 * uStem.y) * uV[0].z;
  }
  float r = length(p - c);
  col -= prev * smoothstep(0.05, 0.0, r) * 0.3;
  return col * 0.35;
}`,
  },
  // ------------------------------------------------------------ E06
  {
    id: 'E06', name: 'Ink Garden', kind: 'fluid ink', energy: [0.15, 0.75],
    palette: 'triad', hue: 0.15, decay: 0.993, floor: 1.5, fluid: { amount: 1, vorticity: 28, noise: 0.35 }, adapt: 0.35,
    warp: /* glsl */ `vec2 warp(vec2 p) { return p; }`,
    js(f, r, fx) {
      const A = f.aspect * 0.5;
      const base = [[-0.55 * A, -0.2], [0, -0.28], [0, 0.18], [0.55 * A, -0.05]];
      for (let i = 0; i < 4; i++) {
        const x = base[i][0] + 0.1 * Math.sin(f.phase * 0.23 + i * 1.9);
        const y = base[i][1] + 0.07 * Math.sin(f.phase * 0.31 + i * 2.7);
        const s = i === 0 ? f.onset[0] : f.stem[i];
        r.v[i * 4] = x;
        r.v[i * 4 + 1] = y;
        r.v[i * 4 + 2] = s;
        r.v[i * 4 + 3] = 0.014 + 0.016 * s;
        if (s > 0.04 && (i > 0 || f.hit > 0)) {
          const a = f.spin * 0.5 + i * (TAU / 4);
          const F = (i === 0 ? 1400 * f.hit : 420 * s) * (0.4 + 0.6 * f.act);
          fx.splat(x / f.aspect + 0.5, y + 0.5, Math.cos(a) * F, Math.sin(a) * F, 0.002, 0);
        }
      }
    },
    draw: /* glsl */ `
vec3 draw(vec2 p, vec2 uv, vec3 prev) {
  vec3 c = vec3(0.0);
  for (int i = 0; i < 4; i++) {
    vec4 e = uV[i];
    c += pal(float(i) * 0.25 + 0.1) * e.z * glow(length(p - e.xy), e.w);
  }
  return c * 0.4;
}`,
  },
  // ------------------------------------------------------------ E07
  {
    id: 'E07', name: 'Oscilloscope', kind: 'minimal line', energy: [0, 0.4],
    palette: 'mono', hue: 0.45, sat: 0.7, decay: 0.955, blur: 0.2, adapt: 0.15, bloom: 0.9, vignette: 0.6,
    js(f, r) {
      r.curveBright = 0.35 + 0.9 * f.loud;
    },
    warp: /* glsl */ `vec2 warp(vec2 p) { return p * vec2(1.0015, 1.0) - vec2(0.0, 0.0014 * uF60 * uSpeed); }`,
    curve: {
      n: 512, thick: 1.5, bright: 0.55, target: 'fb',
      glsl: /* glsl */ `
vec2 curve(float k, float inst) {
  float env = smoothstep(0.0, 0.12, k) * smoothstep(1.0, 0.88, k);
  return vec2((k * 2.0 - 1.0) * uAspect * 0.42, -0.1 + waveAt(k) * 0.24 * env);
}`,
      color: /* glsl */ `
vec3 curveColor(float k, float inst) {
  float env = smoothstep(0.0, 0.15, k) * smoothstep(1.0, 0.85, k);
  return mix(uColA, vec3(1.0), 0.2) * env;
}`,
    },
  },
  // ------------------------------------------------------------ E08
  {
    id: 'E08', name: 'Contour Plasma', kind: 'noise field', energy: [0.3, 0.85],
    palette: 'split', hue: 0.6, feedback: false, bloom: 0.9, adapt: 0.4,
    js(f, r) {
      r.mem.b = mem(r, 'b') + f.dt * f.stem[1] * 0.25;
      r.mem.ph = mem(r, 'ph') + f.dt * (0.15 + 1.6 * f.beatPulse * f.gate[0]) * f.speed;
      r.v[0] = r.mem.b % 64;
      r.v[1] = r.mem.ph % 256;
    },
    comp: /* glsl */ `
vec3 comp(vec2 uv, vec2 p) {
  float t = uPhase * 0.1;
  vec2 q = p * 1.7;
  vec2 w1 = vec2(fbm4(q + vec2(0.0, t)), fbm4(q + vec2(5.2, -t * 0.8)));
  vec2 w2 = vec2(fbm4(q + 3.0 * w1 + vec2(1.7, 9.2) + uV[0].x), fbm4(q + 3.0 * w1 + vec2(8.3, 2.8)));
  float f = fbm4(q + (2.0 + 0.8 * uStem.y) * w2);
  float bands = f * 8.0 - uV[0].y;
  float line = 1.0 - abs(fract(bands) - 0.5) * 2.0;
  float fw = fwidth(bands) * 2.0;
  float ln = smoothstep(1.0 - fw - 0.06, 1.0, line);
  vec3 base = pal(f * 1.2 + uPhase * 0.015) * pow(f, 3.0) * 0.08;
  return base + pal(f + 0.33) * ln * (0.15 + 0.45 * uLoud) * (0.6 + 0.4 * f);
}`,
  },
  // ------------------------------------------------------------ E09
  {
    id: 'E09', name: 'Rorschach', kind: 'mirror split', energy: [0.2, 0.7],
    palette: 'complementary', hue: 0.85, decay: 0.986, blur: 0.1, adapt: 0.35,
    js(f, r) {
      const y = (f.melody - 0.5) * 0.6;
      r.mem.y = approach(mem(r, 'y', y), y, 4, f.dt);
      r.mem.dh = f.hit > 0 ? f.hit : mem(r, 'dh') * Math.exp(-f.dt * 5);
      if (f.hit > 0) r.mem.dy = (h11(f.beatIndex * 1.3) - 0.5) * 0.7;
      r.v[0] = 0;
      r.v[1] = r.mem.y;
      r.v[2] = melodic(f);
      r.v[3] = r.mem.dh;
      r.v[4] = mem(r, 'dy');
      r.v[5] = ((f.barIndex % 2) + 2) % 2; // ink colours trade places every bar
    },
    warp: /* glsl */ `
vec2 warp(vec2 p) {
  vec2 c = curlNoise(vec2(abs(p.x), p.y) * 2.5 + 4.0, uPhase * 0.35);
  // Each beat pushes the ink outward, then it settles; bass adds a steady push.
  float push = 0.0008 + 0.0065 * uBeatPulse + 0.003 * uStem.y * uPres.y;
  float out1 = push * uF60 * uSpeed * smoothstep(0.0, 0.03, abs(p.x));
  // Vocals and the other instruments stir the ink.
  float stir = 0.0008 + 0.0028 * (uStem.z * uPres.z + uStem.w * uPres.w);
  return vec2(p.x - sign(p.x) * out1, p.y * (1.0 - 0.0008 * uF60)) + vec2(sign(p.x) * c.x, c.y) * stir * uF60;
}`,
    draw: /* glsl */ `
vec3 draw(vec2 p, vec2 uv, vec3 prev) {
  float ax = abs(p.x);
  vec3 ca = mix(uColA, uColB, uV[1].y);
  vec3 cb = mix(uColB, uColA, uV[1].y);
  // Melody blot along the spine, a bass blot swelling at the base, and a
  // fresh blot at a new height on every drum hit.
  vec3 c = ca * uV[0].z * glow(length(vec2(ax - 0.03, (p.y - uV[0].y) * 0.6)), 0.015 + 0.015 * uLoud);
  c += cb * uStem.y * uPres.y * glow(length(vec2(ax, p.y + 0.24)), 0.025 + 0.025 * uStem.y) * 0.6;
  c += uColC * uV[0].w * glow(length(vec2(ax - 0.05, p.y - uV[1].x)), 0.02 + 0.03 * uV[0].w);
  return c * (0.025 + 0.05 * uBeatPulse);
}`,
    comp: /* glsl */ `
vec3 comp(vec2 uv, vec2 p) {
  vec3 c = fb(vec2(0.5 + abs(uv.x - 0.5), uv.y));
  float l = luma(c);
  return c * smoothstep(0.0, 0.05, l) + uColC * 0.004 * (1.0 - abs(p.y));
}`,
  },
  // ------------------------------------------------------------ E10
  {
    id: 'E10', name: 'Hex Keys', kind: 'tiled grid', energy: [0.4, 0.9],
    palette: 'mono', hue: 0.6, decay: 0.93, adapt: 0.35,
    js(f, r) {
      r.v[0] = Math.max(melodic(f), 0.3 * f.stem[1]);
    },
    warp: /* glsl */ `vec2 warp(vec2 p) { return p; }`,
    draw: HEX + /* glsl */ `
vec3 draw(vec2 p, vec2 uv, vec3 prev) {
  vec4 h = hexCell(hexP(p));
  float id = hash12(h.zw);
  float pc = floor(id * 12.0);
  float e = pow(chromaAt(pc), 2.0) * clamp(uV[0].x * 1.5, 0.3, 1.0);
  float trig = step(hash12(h.zw + floor(uBeats)), 0.3 + 0.4 * uAct);
  float d = hexD(h.xy);
  float aa = fwidth(d) * 1.5;
  float fill = smoothstep(0.43, 0.43 - aa, d) * (0.5 + 0.5 * smoothstep(0.43, 0.0, d));
  return lin(keyCol(pc, 0.8, 1.0)) * fill * e * e * trig * (0.02 + 0.12 * uBeatPulse);
}`,
    comp: HEX + /* glsl */ `
vec3 comp(vec2 uv, vec2 p) {
  vec4 h = hexCell(hexP(p));
  float d = hexD(h.xy);
  float edge = smoothstep(fwidth(d) * 1.5, 0.0, abs(d - 0.47));
  return fb(uv) + mix(uColB, uColC, 0.5) * edge * 0.035 * (0.5 + uLoud);
}`,
  },
  // ------------------------------------------------------------ E11
  {
    id: 'E11', name: 'Sunrise Arc', kind: 'radial spectrum', energy: [0.35, 0.85],
    palette: 'analogous', hue: 0.02, decay: 0.925, adapt: 0.35,
    warp: /* glsl */ `
vec2 warp(vec2 p) {
  vec2 O = vec2(0.0, -0.42);
  return O + (p - O) / (1.0 + 0.006 * uF60 * uSpeed * (1.0 + 1.5 * uBeatPulse * uPres.x));
}`,
    draw: /* glsl */ `
vec3 draw(vec2 p, vec2 uv, vec3 prev) {
  vec2 O = vec2(0.0, -0.42);
  vec2 d = p - O;
  if (d.y < 0.0) return vec3(0.0);
  float r = length(d);
  float a = atan(d.y, d.x);
  float s = abs(a / PI - 0.5) * 2.0;
  float N = 40.0;
  float bin = (floor(s * N) + 0.5) / N;
  float gap = abs(fract(s * N) - 0.5);
  float lv = specAt(bin * 0.8 + 0.02);
  float r0 = 0.2 + 0.025 * uStem.y;
  float len = 0.015 + 0.3 * lv * lv;
  float aa = 1.5 * px();
  float bar = smoothstep(0.34, 0.28, gap) * smoothstep(r0 + 0.012, r0 + 0.012 + aa, r) * smoothstep(r0 + 0.012 + len + aa, r0 + 0.012 + len, r);
  vec3 c = pal(bin * 0.5) * bar * 0.1;
  c += mix(uColA, vec3(1.0, 0.85, 0.6), 0.4) * smoothstep(r0, r0 - aa, r) * (0.012 + 0.02 * uStem.y);
  return c;
}`,
    comp: /* glsl */ `
vec3 comp(vec2 uv, vec2 p) {
  float hz = -0.42;
  if (p.y >= hz) return fb(uv) + uColC * 0.01 * exp(-(p.y - hz) * 6.0);
  float d = hz - p.y;
  vec3 c = fb(vec2(uv.x + 0.003 * sin(d * 300.0 + uPhase * 2.0), hz + d + 0.5)) * 0.25 * exp(-d * 8.0);
  return c + uColA * glow(d, 0.002) * 0.1;
}`,
  },
  // ------------------------------------------------------------ E12
  {
    id: 'E12', name: 'Warp Speed', kind: 'starfield', energy: [0.55, 1],
    palette: 'analogous', hue: 0.58, sat: 0.35, decay: 0.8, bloom: 1.1,
    particles: {
      count: 16384, size: 2.6, bright: 1.6, spawn: SPAWN.center, target: 'fb', speed: 0.1, curl: 0, life: 0.2,
      zoomFlow: 1.1, drag: 4, spread: 0.12,
    },
    warp: /* glsl */ `vec2 warp(vec2 p) { return rot2(-uSpinStep * 0.0625) * p / (1.0 + 0.012 * uF60 * uSpeed); }`,
    draw: /* glsl */ `
vec3 draw(vec2 p, vec2 uv, vec3 prev) {
  return uColC * glow(length(p), 0.06) * (0.002 + 0.01 * uStem.y);
}`,
  },
  // ------------------------------------------------------------ E13
  {
    id: 'E13', name: 'Constellations', kind: 'dark minimal', energy: [0, 0.45],
    palette: 'mono', hue: 0.6, decay: 0.9, adapt: 0.15, vignette: 0.55,
    js(f, r) {
      r.v[0] = Math.max(melodic(f), 0.4 * f.stem[1]) * 1.2;
    },
    warp: /* glsl */ `vec2 warp(vec2 p) { return p; }`,
    draw: /* glsl */ `
vec2 starPos(vec2 cell) { return cell + 0.2 + 0.6 * hash22(cell * 1.7 + 0.3); }
bool hasStar(vec2 cell) { return hash12(cell * 1.3 + 7.1) < 0.5; }
vec3 draw(vec2 p, vec2 uv, vec3 prev) {
  float S = 6.5;
  vec2 q = p * S + vec2(uPhase * 0.03, 0.0);
  vec2 cell = floor(q);
  vec3 c = vec3(0.0);
  float pw = S * px();
  for (int j = -1; j <= 1; j++) for (int i = -1; i <= 1; i++) {
    vec2 cc = cell + vec2(i, j);
    if (!hasStar(cc)) continue;
    vec2 sp = starPos(cc);
    float pc = floor(hash12(cc + 3.7) * 12.0);
    float e = pow(chromaAt(pc), 2.5) * clamp(uV[0].x * 1.5, 0.3, 1.0);
    vec3 col = keyCol(pc, 0.3, 1.0);
    float tw = 0.7 + 0.3 * sin(uTime * (0.5 + hash12(cc)) + pc);
    float sd = length(q - sp);
    c += col * (0.4 * tw + 1.6 * e) * (glow(sd, pw * (1.8 + 2.5 * e)) + 0.08 * glow(sd, pw * 10.0));
    for (int k = 0; k < 2; k++) {
      vec2 nc = cc + (k == 0 ? vec2(1.0, 0.0) : vec2(0.0, 1.0));
      if (!hasStar(nc)) continue;
      float pc2 = floor(hash12(nc + 3.7) * 12.0);
      float link = min(e, pow(chromaAt(pc2), 2.5) * clamp(uV[0].x * 1.5, 0.3, 1.0));
      if (link < 0.04) continue;
      c += mix(col, keyCol(pc2, 0.3, 1.0), 0.5) * link * glow(sdSeg(q, sp, starPos(nc)), pw * 1.1) * 0.8;
    }
  }
  return c * 0.08;
}`,
  },
  // ------------------------------------------------------------ E14
  {
    id: 'E14', name: 'Polyhedra', kind: 'geometric wireframe', energy: [0.35, 0.9],
    palette: 'triad', hue: 0.5, decay: 0.88, adapt: 0.3,
    js(f, r) {
      const shape = SOLIDS[((f.sectionIndex % 4) + 4) % 4];
      const scale = 0.2 + 0.07 * f.stem[1] + 0.02 * f.beatPulse * f.gate[0];
      const tilt = 0.45 + 0.25 * Math.sin(f.phase * 0.07);
      let n = projectSolid(shape, r, 0, f.spin, tilt, scale, 1);
      const inner = clamp01((f.act - 0.45) / 0.25);
      if (inner > 0.01) {
        const dual = SOLIDS[((f.sectionIndex % 4) + 6) % 4];
        n = projectSolid(dual, r, n, -f.spin * 0.5, -tilt, scale * 0.45, inner * 0.7);
      }
      r.segN = n;
    },
    warp: /* glsl */ `vec2 warp(vec2 p) { return rot2(uSpinStep * 0.0625) * p * (1.0 + 0.004 * uF60 * uSpeed); }`,
    draw: /* glsl */ `
float segZ(int i) { return uSegZ[i / 4][i % 4]; }
vec3 draw(vec2 p, vec2 uv, vec3 prev) {
  vec3 c = vec3(0.0);
  float w = px() * 1.3;
  for (int i = 0; i < 48; i++) {
    if (i >= uSegN) break;
    vec4 s = uSeg[i];
    float d = sdSeg(p, s.xy, s.zw);
    float z = segZ(i);
    c += pal(z * 0.4 + 0.1) * z * (glow(d, w) + 0.12 * glow(d, w * 7.0));
  }
  return c * (0.12 + 0.08 * uLoud);
}`,
  },
  // ------------------------------------------------------------ E15
  {
    id: 'E15', name: 'Liquid Chrome', kind: 'metaballs', energy: [0.3, 0.8],
    palette: 'complementary', hue: 0.55, sat: 0.6, feedback: false, adapt: 0.4,
    js(f, r) {
      const A = f.aspect * 0.5;
      const src = [1, 1, 0, 2, 3, 3];
      for (let i = 0; i < 6; i++) {
        const t = f.phase * (0.11 + 0.03 * i);
        const lv = src[i] === 0 ? f.onset[0] * 0.6 + f.stem[0] * 0.4 : f.stem[src[i]];
        const tr = 0.045 + 0.075 * lv + 0.015 * f.beatPulse * f.gate[0];
        const rr = approach(mem(r, 'r' + i, tr), tr, 8, f.dt);
        r.mem['r' + i] = rr;
        r.v[i * 4] = 0.55 * A * Math.sin(t + i * 1.7) * (0.6 + 0.4 * Math.sin(t * 0.37 + i));
        r.v[i * 4 + 1] = 0.26 * Math.sin(t * 1.31 + i * 2.3);
        r.v[i * 4 + 2] = rr;
      }
    },
    comp: /* glsl */ `
vec3 comp(vec2 uv, vec2 p) {
  float F = 0.0;
  vec2 g = vec2(0.0);
  for (int i = 0; i < 6; i++) {
    vec2 d = p - uV[i].xy;
    float r2 = uV[i].z * uV[i].z;
    float q = 1.0 / (dot(d, d) + 1e-4);
    F += r2 * q;
    g += -2.0 * r2 * q * q * d;
  }
  float e = fwidth(F) * 1.2;
  float m = smoothstep(1.0 - e, 1.0 + e, F);
  vec3 n = normalize(vec3(-g * 0.05, 1.0));
  vec3 r = reflect(vec3(0.0, 0.0, -1.0), n);
  vec3 env = mix(uColB * 0.03, uColA * 0.7, smoothstep(-0.3, 0.9, r.y));
  env += vec3(1.0) * glow(r.y - 0.2 - 0.12 * sin(r.x * 3.0 + uPhase * 0.3), 0.06);
  env += uColC * pow(max(-r.x, 0.0), 5.0) * 0.8;
  float fres = pow(1.0 - n.z, 2.0);
  vec3 chrome = env * (0.3 + 0.7 * fres) + vec3(0.02);
  vec3 bg = uColB * 0.003 + uColC * 0.015 * smoothstep(0.2, 1.0, F);
  return mix(bg, chrome, m);
}`,
  },
  // ------------------------------------------------------------ E16
  {
    id: 'E16', name: 'Aurora', kind: 'curtains', energy: [0.1, 0.6],
    palette: 'analogous', hue: 0.35, decay: 0.965, blur: 0.1, adapt: 0.3,
    js(f, r) {
      const voc = f.stem[2];
      r.v[0] = Math.max(voc, f.stem[3] * 0.7 * (1 - f.gate[2])) + 0.08 * f.loud;
    },
    warp: /* glsl */ `
vec2 warp(vec2 p) {
  float n = vnoise(vec2(p.x * 3.0, uPhase * 0.5)) - 0.5;
  return p + vec2(n * 0.0012, -0.0005) * uF60;
}`,
    draw: /* glsl */ `
vec3 draw(vec2 p, vec2 uv, vec3 prev) {
  float t = uPhase * 0.15;
  float y0 = 0.0 + 0.09 * sin(p.x * 1.4 + t) + 0.06 * (fbm2(vec2(p.x * 1.8 - t * 0.7, t)) - 0.5);
  float dy = p.y - y0;
  if (dy < -0.02) return vec3(0.0);
  float rays = 0.3 + 0.7 * fbm2(vec2(p.x * 24.0 + fbm2(vec2(p.x * 3.0, t)) * 4.0, t * 2.0));
  float body = exp(-max(dy, 0.0) * 5.0) * smoothstep(-0.02, 0.005, dy);
  vec3 col = mix(uColA, uColC, smoothstep(0.0, 0.3, dy));
  return col * body * rays * uV[0].x * 0.03;
}`,
    comp: /* glsl */ `
vec3 comp(vec2 uv, vec2 p) {
  vec3 c = fb(uv) + uColB * 0.006 * smoothstep(0.5, -0.3, p.y);
  float m = -0.38 + 0.07 * fbm2(vec2(p.x * 2.2, 1.0)) + 0.025 * fbm2(vec2(p.x * 9.0, 3.0));
  return mix(c, vec3(0.0), smoothstep(m + 0.002, m - 0.002, p.y));
}`,
  },
  // ------------------------------------------------------------ E17
  {
    id: 'E17', name: 'Hyperspace', kind: 'zoom tunnel', energy: [0.6, 1],
    palette: 'complementary', hue: 0.8, decay: 0.94, rings: 0.35, bloom: 1.2,
    js(f, r) {
      r.v[0] = f.barIndex % 2 === 1 ? -1 : 1;
    },
    warp: /* glsl */ `
vec2 warp(vec2 p) {
  float r = length(p);
  float z = 1.0 + (0.02 + 0.015 * uStem.y + 0.025 * uBeatPulse * uPres.x + 0.03 * uBuild * uBuild) * uSpeed * uF60 * (0.5 + r * 1.2);
  return rot2(-uSpinStep * 0.25 * uV[0].x) * p / z;
}`,
    draw: /* glsl */ `
vec3 draw(vec2 p, vec2 uv, vec3 prev) {
  vec2 q = rot2(uSpin) * p;
  float R = 0.06 + 0.1 * uStem.y + 0.02 * uBeatPulse;
  float d = sdPoly(q, 6.0, R);
  float e = glow(d, px() * 1.8) + 0.2 * glow(d, 0.012);
  return mix(uColA, uColC, 0.3) * e * (0.08 + 0.4 * max(uStem.y, 0.4 * uLoud));
}`,
  },
  // ------------------------------------------------------------ E18
  {
    id: 'E18', name: 'Mandala', kind: 'radial kaleido', energy: [0.5, 1],
    palette: 'triad', hue: 0.1, decay: 0.94, bloom: 1.1,
    js(f, r) {
      r.curveBright = 0.4 + 0.8 * f.loud + 0.8 * f.beatPulse * f.gate[0];
    },
    warp: /* glsl */ `vec2 warp(vec2 p) { return rot2(-uSpinStep * 0.125) * p / (1.0 + 0.006 * uF60 * uSpeed); }`,
    curve: {
      n: 512, thick: 1.8, bright: 0.35, target: 'fb',
      glsl: /* glsl */ `
vec2 curve(float k, float inst) {
  float a = (k - 0.5) * 1.1;
  float r = 0.27 + waveAt(k) * 0.1;
  return r * vec2(cos(a), sin(a));
}`,
    },
    draw: /* glsl */ `
vec3 draw(vec2 p, vec2 uv, vec3 prev) { return uColB * glow(length(p), 0.05) * uStem.y * 0.02; }`,
    comp: /* glsl */ `
vec3 comp(vec2 uv, vec2 p) {
  float seg = TAU / 8.0;
  float a = mod(atan(p.y, p.x) + uSpin * 0.0625, seg);
  a = abs(a - seg * 0.5);
  vec2 q = length(p) * vec2(cos(a), sin(a));
  return fb(q / vec2(uAspect, 1.0) + 0.5);
}`,
  },
  // ------------------------------------------------------------ E19
  {
    id: 'E19', name: 'Harmonograph', kind: 'minimal figure', energy: [0.05, 0.5],
    palette: 'analogous', hue: 0.1, decay: 0.86, adapt: 0.15, vignette: 0.55,
    js(f, r) {
      // The figure changes every 2 bars, stepping through the ratio table in
      // circle-of-fifths order from the song's key, and glides between shapes.
      // Phases advance with the beat, each downbeat pushes the pendulums, the
      // melody detunes the figure and drum hits widen the second pendulum.
      const L = RATIOS.length;
      const stepN = Math.floor(Math.max(0, f.barIndex) / 2);
      const [a, b] = RATIOS[(((f.keyTonic + stepN * 5) % L) + L) % L];
      const fx = approach(mem(r, 'fx', a), a, 1.2, f.dt);
      const fy = approach(mem(r, 'fy', b), b, 1.2, f.dt);
      r.mem.fx = fx;
      r.mem.fy = fy;
      r.mem.swing = f.onBar ? 1 : mem(r, 'swing') * Math.exp(-f.dt * 1.2);
      const det = 0.012 * (f.melody - 0.5) + 0.004 * Math.sin(f.phase * 0.05);
      r.v[0] = fx;
      r.v[1] = fx * 2 + 0.01 + det;
      r.v[2] = fy;
      r.v[3] = fy * (f.minor ? 1.5 : 2) - 0.01;
      r.v[4] = f.spin * 0.25;
      r.v[5] = f.phase * 0.07 + f.beats * 0.25;
      r.v[6] = 1.3 + f.phase * 0.05 - f.beats * 0.18;
      r.v[7] = 0.5 - f.phase * 0.04 + f.beats * 0.11;
      r.v[8] = 0.75 + 0.2 * f.loud + 0.3 * r.mem.swing;
      r.v[9] = 0.3 + 0.35 * melodic(f) + 0.35 * f.onset[0];
      r.curveBright = 0.3 + 0.9 * Math.max(f.loud, melodic(f));
    },
    warp: /* glsl */ `vec2 warp(vec2 p) { return p; }`,
    curve: {
      n: 1024, thick: 1.2, bright: 0.3, target: 'fb',
      glsl: /* glsl */ `
vec2 curve(float k, float inst) {
  float t = k * 42.0;
  float damp = exp(-k * 2.0);
  vec4 f = uV[0], ph = uV[1];
  float x = sin(f.x * t + ph.x) + uV[2].y * sin(f.y * t + ph.y);
  float y = sin(f.z * t + ph.z) + uV[2].y * sin(f.w * t + ph.w);
  return vec2(x, y) * damp * 0.26 * uV[2].x;
}`,
      color: /* glsl */ `vec3 curveColor(float k, float inst) { return mix(uColA, uColC, k) * (0.5 + 0.5 * exp(-k * 1.5)); }`,
    },
  },
  // ------------------------------------------------------------ E20
  {
    id: 'E20', name: 'Moonrise', kind: 'dark minimal', energy: [0, 0.45],
    palette: 'analogous', hue: 0.62, sat: 0.5, feedback: false, adapt: 0.15, vignette: 0.5,
    js(f, r) {
      r.mem.halo = approach(mem(r, 'halo'), Math.max(f.loud, melodic(f)), 3, f.dt);
      r.v[0] = r.mem.halo;
      r.v[1] = 0.1 + 0.06 * Math.sin(f.phase * 0.01);
    },
    comp: /* glsl */ `
vec3 comp(vec2 uv, vec2 p) {
  float hz = -0.14;
  vec2 moon = vec2(0.3 * uAspect * 0.5, uV[0].y);
  float R = 0.065;
  vec3 mcol = mix(vec3(1.0, 0.95, 0.85), uColA, 0.3);
  vec2 q = p;
  float d = hz - p.y;
  if (d > 0.0) q = vec2(p.x + (0.002 + d * 0.02) * sin(d * 420.0 / (0.4 + d * 5.0) - uPhase * 1.5), hz + d);
  float md = length(q - moon);
  vec3 c = mcol * smoothstep(R, R - 0.0025, md) * (0.35 + 0.15 * fbm2((q - moon) * 28.0));
  c += uColA * step(R, md) * (0.03 * exp(-(md - R) * 10.0) + 0.06 * uV[0].x * exp(-(md - R) * 4.0));
  c += uColB * 0.006 * smoothstep(0.5, hz, q.y);
  if (d > 0.0) {
    float row = floor(d * 160.0);
    float st = specAt(fract(row * 0.137) * 0.7);
    float colm = glow((q.x - moon.x) / (0.04 + d * 0.6), 1.0);
    c = c * 0.35 * exp(-d * 3.0) + mcol * colm * st * st * 0.08 * step(0.45, fract(d * 160.0)) * exp(-d * 2.0);
  }
  c += vec3(0.6, 0.7, 1.0) * glow(p.y - hz, 0.0012) * 0.03;
  return c;
}`,
  },
  // ------------------------------------------------------------ E21
  {
    id: 'E21', name: 'Retro Grid', kind: 'perspective grid', energy: [0.5, 1],
    palette: 'split', hue: 0.85, feedback: false, bloom: 1.2, adapt: 0.35,
    js(f, r) {
      const prev = mem(r, 'lastBeats', f.beats);
      let d = f.beats - prev;
      if (d < 0) d += 256;
      if (d > 2) d = f.dt * 2;
      r.mem.lastBeats = f.beats;
      r.mem.scroll = (mem(r, 'scroll') + d * 0.5 * (0.5 + 0.5 * f.act)) % 1;
      r.v[0] = r.mem.scroll;
    },
    comp: /* glsl */ `
// Terrain height at floor point w = (x, z): a valley down the middle, hills on
// both sides whose ridges follow the spectrum (frequency maps outward), and a
// ridge that rolls toward the viewer on every kick. Scrolls with the grid.
float terrain(vec2 w) {
  float ax = abs(w.x);
  float z = w.y + uV[0].x * 2.0;
  float side = smoothstep(0.35, 1.6, ax);
  float lv = specAt(clamp(ax / 5.0, 0.0, 1.0) * 0.7 + 0.03);
  float ridge = 0.6 + 0.4 * sin(z * 1.3 + ax * 0.9);
  float h = side * (0.03 + 0.34 * lv * (0.4 + 0.6 * uAct)) * ridge;
  float k = fract(z * 0.25 + 0.5);
  h += uBeatPulse * uPres.x * 0.05 * exp(-pow((k - 0.5) * 9.0, 2.0)) * (0.4 + side);
  // Flatten toward the horizon so distant terrain stays clean.
  h *= smoothstep(18.0, 6.0, w.y);
  return min(h, 0.32);
}

vec3 comp(vec2 uv, vec2 p) {
  float hz = -0.06;
  vec3 c = vec3(0.0);
  if (p.y < hz) {
    float dy = hz - p.y;
    // Ray-march the heightfield: the eye is 0.35 above the floor, and at depth
    // z this pixel's ray is at height 0.35 - dy * z.
    float zFlat = 0.35 / dy;
    float z0 = 0.2, z = zFlat, h = 0.0;
    float st = 0.08;
    for (int k = 0; k < 40; k++) {
      float zt = z0 + st;
      if (zt >= zFlat) break;
      if (0.35 - dy * zt <= terrain(vec2(p.x * zt, zt))) { z = zt; break; }
      z0 = zt;
      st *= 1.12;
    }
    // Refine the hit between z0 and z.
    for (int k = 0; k < 5; k++) {
      float zm = 0.5 * (z0 + z);
      if (0.35 - dy * zm <= terrain(vec2(p.x * zm, zm))) z = zm; else z0 = zm;
    }
    h = terrain(vec2(p.x * z, z));
    float x = p.x * z;
    float gz = abs(fract(z * 0.5 + uV[0].x) - 0.5);
    float gx = abs(fract(x * 1.5) - 0.5);
    float wz = fwidth(z * 0.5) * 1.2, wx = fwidth(x * 1.5) * 1.2;
    float line = max(smoothstep(wz, 0.0, gz), smoothstep(wx, 0.0, gx));
    // Colour by height (valleys A, peaks C); every downbeat flashes the grid.
    vec3 lc = mix(uColA, uColC, smoothstep(0.02, 0.22, h));
    lc = mix(lc, vec3(1.0), 0.35 * uBarPulse);
    c += lc * line * exp(-z * 0.12) * (0.35 + 0.5 * uBeatPulse * uPres.x + 0.8 * h);
    c += uColA * 0.03 * exp(-dy * 25.0);
  } else {
    float dy = p.y - hz;
    vec2 sp = p - vec2(0.0, hz + 0.15);
    float R = 0.19 + 0.015 * uStem.y;
    float sd = length(sp);
    float cut = sp.y > 0.02 ? 1.0 : step(0.35 - sp.y / R * 0.5, fract(sp.y * 24.0 - uPhase * 0.2));
    vec3 sun = mix(uColC, uColB, smoothstep(-R, R, sp.y));
    c += sun * smoothstep(R, R - 0.003, sd) * cut * 0.5;
    c += uColC * 0.08 * exp(-(sd - R) * 7.0) * step(R, sd);
    float bin = abs(p.x) / (uAspect * 0.5);
    float mh = 0.015 + 0.1 * specAt(bin * 0.6 + 0.02) * (0.4 + 0.6 * uAct);
    if (dy < mh) c = uColB * 0.004 + uColA * glow(dy - mh, 0.002) * 0.25;
    c += mix(uColC * 0.02, uColB * 0.002, smoothstep(0.0, 0.5, dy));
  }
  return c;
}`,
  },
  // ------------------------------------------------------------ E22
  {
    id: 'E22', name: 'Silk Flame', kind: 'fractal flame', energy: [0.05, 0.5],
    palette: 'analogous', hue: 0.55, decay: 0.955, adapt: 0.3, bloom: 0.9,
    warp: /* glsl */ `vec2 warp(vec2 p) { return p; }`,
    comp: FLAME_COMP,
    flame: {
      // Flows between two forms every 8 bars (a drop reverses it); the three
      // transforms drift so it keeps unfolding; bass breathes, beats kick.
      count: 262144, iters: 4, rounds: 2, zoom: 0.2, camSpin: 0.125, gain: 1, flow: 1, breathe: 0.12,
      xforms: [
        { aff: [0.7, -0.3, 0.3, 0.7, 0.2, 0.1], weight: 1, color: 0, vars: { julia: 0.7, linear: 0.3 }, alt: { spiral: 0.6, heart: 0.4 }, spin: 0.125, drift: [0.25, 0.2] },
        { aff: [0.5, 0, 0, 0.5, -0.5, 0.2], weight: 0.6, color: 0.5, vars: { spherical: 0.5, swirl: 0.3 }, alt: { disc: 0.6, horseshoe: 0.4 }, bass: 0.2, drift: [0.3, 0.25] },
        { aff: [-0.4, 0.3, -0.3, -0.4, 0.3, -0.4], weight: 0.4, color: 0.9, vars: { sinusoidal: 1 }, alt: { handkerchief: 0.7, polar: 0.3 }, spin: -0.0625, pulse: 0.08, drift: [0.2, 0.3] },
      ],
    },
  },
  // ------------------------------------------------------------ E23
  {
    id: 'E23', name: 'Ember Flame', kind: 'fractal flame', energy: [0.55, 1],
    palette: 'split', hue: 0.02, decay: 0.92, adapt: 0.35, bloom: 1.2,
    warp: /* glsl */ `vec2 warp(vec2 p) { return p; }`,
    comp: FLAME_COMP,
    flame: {
      // Energetic: flows every 4 bars (a drop reverses it), strong bass
      // breathing, beat kicks on every transform, wide drift.
      count: 524288, iters: 4, rounds: 2, zoom: 0.3, camSpin: -0.125, gain: 1, flow: 2, breathe: 0.2,
      xforms: [
        { aff: [0.8, 0.2, -0.2, 0.8, 0, 0.1], weight: 1, color: 0.1, vars: { swirl: 0.6, linear: 0.4 }, alt: { spiral: 0.5, heart: 0.3 }, spin: 0.25, bass: 0.25, pulse: 0.1, drift: [0.3, 0.25] },
        { aff: [0.4, -0.35, 0.35, 0.4, 0.6, 0], weight: 0.5, color: 0.6, vars: { horseshoe: 1 }, alt: { disc: 1 }, pulse: 0.12, drift: [0.35, 0.3] },
        { aff: [0.5, 0, 0, -0.5, -0.4, -0.3], weight: 0.4, color: 0.95, vars: { handkerchief: 0.8 }, alt: { polar: 0.8 }, spin: -0.125, pulse: 0.08, drift: [0.25, 0.35] },
      ],
    },
  },
  // ------------------------------------------------------------ E24
  {
    id: 'E24', name: 'Spiral Nebula', kind: 'fractal flame', energy: [0.3, 0.75],
    palette: 'triad', hue: 0.6, decay: 0.94, adapt: 0.3,
    warp: /* glsl */ `vec2 warp(vec2 p) { return p; }`,
    comp: FLAME_COMP,
    flame: {
      // Flows between a spiral galaxy and a looser swirl every 8 bars (a drop
      // reverses it); the arms drift, bass breathes, beats kick the core.
      count: 262144, iters: 4, rounds: 2, zoom: 0.22, camSpin: 0.125, gain: 1, flow: 1, breathe: 0.15,
      xforms: [
        { aff: [0.6, -0.5, 0.5, 0.6, 0, 0], weight: 1, color: 0, vars: { spiral: 0.3, linear: 0.7 }, alt: { swirl: 0.5, linear: 0.5 }, spin: 0.125, bass: 0.15, pulse: 0.08, drift: [0.12, 0.12] },
        { aff: [0.3, 0, 0, 0.3, 0.7, 0], weight: 0.5, color: 0.5, vars: { spherical: 1 }, alt: { julia: 0.7, spherical: 0.3 }, drift: [0.25, 0.3] },
        { aff: [0.3, 0, 0, 0.3, -0.35, 0.6], weight: 0.3, color: 0.85, vars: { disc: 0.6, linear: 0.4 }, alt: { heart: 0.6, linear: 0.4 }, drift: [0.3, 0.25] },
      ],
    },
  },
];

// ------------------------------------------------------------ palettes

const SCHEMES: Record<PaletteScheme, [number, number, number, number, number, number]> = {
  // hue offsets for colours A, B, C then value multipliers
  analogous: [0, 0.13, -0.11, 1, 0.9, 1],
  complementary: [0, 0.5, 0.06, 1, 0.9, 1],
  triad: [0, 0.333, 0.667, 1, 0.9, 0.9],
  split: [0, 0.42, 0.58, 1, 0.9, 1],
  mono: [0, 0.03, -0.03, 1, 0.6, 1.2],
};

function hsv(h: number, s: number, v: number, out: Float32Array, o: number): void {
  h = ((h % 1) + 1) % 1;
  const i = Math.floor(h * 6);
  const f = h * 6 - i;
  const p = v * (1 - s);
  const q = v * (1 - f * s);
  const t = v * (1 - (1 - f) * s);
  let r = 0, g = 0, b = 0;
  switch (i % 6) {
    case 0: r = v; g = t; b = p; break;
    case 1: r = q; g = v; b = p; break;
    case 2: r = p; g = v; b = t; break;
    case 3: r = p; g = q; b = v; break;
    case 4: r = t; g = p; b = v; break;
    default: r = v; g = p; b = q;
  }
  // to linear
  out[o] = Math.pow(r, 2.2);
  out[o + 1] = Math.pow(g, 2.2);
  out[o + 2] = Math.pow(b, 2.2);
}

/** Writes three linear RGB colours (9 floats) for a palette around `hue`. */
export function paletteColors(scheme: PaletteScheme, hue: number, sat: number, out: Float32Array, offset = 0): void {
  const s = SCHEMES[scheme];
  const sa = Math.min(1, Math.max(0, sat));
  hsv(hue + s[0], sa, s[3], out, offset);
  hsv(hue + s[1], sa * (scheme === 'mono' ? 0.5 : 1), s[4], out, offset + 3);
  hsv(hue + s[2], sa * 0.85, s[5], out, offset + 6);
}

export function makeRuntime(): Runtime {
  return {
    v: new Float32Array(32),
    seg: new Float32Array(48 * 4),
    segZ: new Float32Array(48),
    segN: 0,
    ringCenter: [0, 0],
    mem: {},
    curveBright: 1,
  };
}
