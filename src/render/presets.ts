// Enhanced-engine presets. Numeric parameters are interpolated during a
// blend; discrete choices (warp function, waveform shape, palette, kaleido
// segments, particle spawn mode) are crossfaded by running both variants.

export type PaletteScheme = 'analogous' | 'complementary' | 'triad' | 'split' | 'mono';
export type EmitterPattern = 'radial' | 'swirl' | 'orbit' | 'jets' | 'random';

export const WARP = { sine: 0, spiral: 1, tunnel: 2, petals: 3, twirl: 4, fractal: 5, lens: 6, flow: 7 } as const;
export const WAVE = { line: 0, circle: 1, spiral: 2, dual: 3 } as const;
export const SPAWN = { anywhere: 0, waveform: 1, emitters: 2, ring: 3, center: 4 } as const;

export interface NumParams {
  zoom: number; // per-frame (60 fps) base zoom, 1 = none
  zoomExp: number; // radial dependence of zoom
  zoomBass: number; // extra zoom per unit bass stem
  zoomBeat: number; // extra zoom on beatPulse
  rot: number; // feedback rotation in turns per bar
  bRot: number; // buffer B rotation relative to A
  bZoom: number; // buffer B zoom relative to A
  trans: number; // slow drift amplitude
  warpAmt: number;
  warpSpeed: number;
  warpScale: number;
  fluid: number; // fluid displacement of feedback
  vorticity: number;
  splat: number; // splat force scale
  noise: number; // continuous curl forcing
  couple: number;
  decayA: number;
  decayB: number;
  blur: number;
  bMix: number;
  hueDrift: number; // per-frame hue rotation of trails (radians)
  sat: number;
  hueOffset: number;
  waveW: number;
  waveAmp: number;
  waveThick: number;
  waveMirror: number;
  drumsW: number;
  bassW: number;
  bassSize: number;
  vocalsW: number;
  chromaW: number;
  sparkle: number;
  partW: number;
  partSize: number;
  partSpeed: number;
  partCurl: number;
  partLife: number;
  partFluid: number;
  bloom: number;
  exposure: number;
  vignette: number;
  kalRot: number; // kaleido orientation turns per bar
}

export interface Preset extends NumParams {
  name: string;
  energy: 'high' | 'calm';
  warpFn: number;
  waveStyle: number;
  palette: PaletteScheme;
  kaleido: number; // segments, 0 = off
  mirror: boolean;
  emitter: EmitterPattern;
  emitterCount: number;
  spawn: number;
  bassSides: number;
  rotFlip: boolean;
}

export const NUM_KEYS: (keyof NumParams)[] = [
  'zoom', 'zoomExp', 'zoomBass', 'zoomBeat', 'rot', 'bRot', 'bZoom', 'trans',
  'warpAmt', 'warpSpeed', 'warpScale', 'fluid', 'vorticity', 'splat', 'noise',
  'couple', 'decayA', 'decayB', 'blur', 'bMix', 'hueDrift', 'sat', 'hueOffset',
  'waveW', 'waveAmp', 'waveThick', 'waveMirror', 'drumsW', 'bassW', 'bassSize',
  'vocalsW', 'chromaW', 'sparkle', 'partW', 'partSize', 'partSpeed', 'partCurl',
  'partLife', 'partFluid', 'bloom', 'exposure', 'vignette', 'kalRot',
];

const DEFAULTS: Omit<Preset, 'name'> = {
  energy: 'calm',
  zoom: 1.004,
  zoomExp: 0,
  zoomBass: 0.008,
  zoomBeat: 0.012,
  rot: 0.125,
  bRot: -1,
  bZoom: 0.6,
  trans: 0,
  warpAmt: 0.6,
  warpSpeed: 1,
  warpScale: 1,
  fluid: 0.7,
  vorticity: 25,
  splat: 1,
  noise: 1,
  couple: 0.5,
  decayA: 0.975,
  decayB: 0.96,
  blur: 0.35,
  bMix: 0.7,
  hueDrift: 0,
  sat: 1,
  hueOffset: 0,
  waveW: 1,
  waveAmp: 1,
  waveThick: 2.2,
  waveMirror: 0.5,
  drumsW: 1,
  bassW: 0.7,
  bassSize: 1,
  vocalsW: 1,
  chromaW: 0.8,
  sparkle: 0.6,
  partW: 1,
  partSize: 1.4,
  partSpeed: 0.5,
  partCurl: 0.12,
  partLife: 0.35,
  partFluid: 1,
  bloom: 1,
  exposure: 1,
  vignette: 0.45,
  kalRot: 0.25,
  warpFn: WARP.sine,
  waveStyle: WAVE.circle,
  palette: 'analogous',
  kaleido: 0,
  mirror: false,
  emitter: 'random',
  emitterCount: 3,
  spawn: SPAWN.waveform,
  bassSides: 6,
  rotFlip: false,
};

function preset(name: string, p: Partial<Omit<Preset, 'name'>>): Preset {
  return { ...DEFAULTS, ...p, name };
}

export const PRESETS: Preset[] = [
  preset('Nebula Loom', {
    warpFn: WARP.sine, warpAmt: 0.7, zoom: 1.004, rot: 0.125, fluid: 0.8,
    emitter: 'random', palette: 'analogous', waveStyle: WAVE.circle, bassSides: 6,
    spawn: SPAWN.waveform, decayA: 0.978, vocalsW: 1.1,
  }),
  preset('Cathedral of Glass', {
    energy: 'high', warpFn: WARP.petals, warpAmt: 0.8, kaleido: 6, rot: 1, rotFlip: true,
    emitter: 'orbit', emitterCount: 6, palette: 'triad', waveStyle: WAVE.spiral, bassSides: 3,
    spawn: SPAWN.emitters, zoom: 1.008, decayA: 0.965, decayB: 0.95, bloom: 1.2, kalRot: 1,
  }),
  preset('Ink Tide', {
    warpFn: WARP.flow, warpAmt: 0.6, zoom: 1.0, rot: 0, fluid: 1.6, vorticity: 35, splat: 1.3,
    emitter: 'swirl', palette: 'complementary', waveStyle: WAVE.line, spawn: SPAWN.anywhere,
    decayA: 0.988, decayB: 0.975, partW: 1.2, partFluid: 1.5, bassW: 0.4, couple: 0.3,
  }),
  preset('Solar Vortex', {
    energy: 'high', warpFn: WARP.spiral, warpAmt: 1, zoom: 1.02, zoomExp: 1, rot: 0.5,
    emitter: 'radial', palette: 'split', waveStyle: WAVE.circle, bassSides: 5,
    spawn: SPAWN.center, partSpeed: 0.9, decayA: 0.965, decayB: 0.955, hueDrift: 0.004, bloom: 1.2,
  }),
  preset('Aurora Drift', {
    warpFn: WARP.fractal, warpAmt: 0.8, zoom: 0.998, rot: 0, trans: 1, vocalsW: 1.6,
    palette: 'analogous', waveStyle: WAVE.dual, waveMirror: 1, spawn: SPAWN.waveform,
    decayA: 0.984, decayB: 0.97, bassW: 0.5, hueOffset: 0.35, fluid: 0.6,
  }),
  preset('Hyperspace Bloom', {
    energy: 'high', warpFn: WARP.tunnel, warpAmt: 1, zoom: 1.03, zoomExp: 1.5, rot: 0.25, rotFlip: true,
    emitter: 'jets', palette: 'complementary', waveStyle: WAVE.circle, spawn: SPAWN.ring,
    bloom: 1.4, decayA: 0.96, decayB: 0.95, partSpeed: 0.8, zoomBeat: 0.03,
  }),
  preset('Mandala Engine', {
    energy: 'high', warpFn: WARP.petals, warpAmt: 0.6, kaleido: 8, rot: 1, bassSides: 8,
    chromaW: 1.4, palette: 'triad', emitter: 'orbit', emitterCount: 4, spawn: SPAWN.emitters,
    decayA: 0.97, decayB: 0.96, kalRot: -0.5, waveStyle: WAVE.circle,
  }),
  preset('Deep Current', {
    warpFn: WARP.twirl, warpAmt: 0.9, fluid: 1.3, emitter: 'orbit', emitterCount: 3, rot: 0.125,
    decayA: 0.988, decayB: 0.975, waveStyle: WAVE.spiral, palette: 'mono', spawn: SPAWN.anywhere,
    sat: 1.1, hueOffset: 0.55, bassSides: 4,
  }),
  preset('Prism Storm', {
    energy: 'high', warpFn: WARP.fractal, warpAmt: 1.1, couple: 1.2, mirror: true, emitter: 'random',
    rot: 0.5, rotFlip: true, palette: 'split', waveStyle: WAVE.dual, decayA: 0.962, decayB: 0.955,
    spawn: SPAWN.waveform, splat: 1.4, hueDrift: -0.006,
  }),
  preset('Liquid Chrome', {
    warpFn: WARP.lens, warpAmt: 0.9, couple: 0.9, palette: 'mono', sat: 0.55, waveStyle: WAVE.line,
    spawn: SPAWN.center, decayA: 0.982, decayB: 0.972, rot: 0.25, bassSides: 7, exposure: 1.1,
  }),
  preset('Event Horizon', {
    energy: 'high', warpFn: WARP.spiral, warpAmt: 1.3, zoom: 1.028, zoomExp: 2, emitter: 'swirl',
    palette: 'complementary', waveStyle: WAVE.circle, bassSides: 4, spawn: SPAWN.ring,
    rot: 0.25, decayA: 0.968, decayB: 0.958, bloom: 1.3, partCurl: 0.2,
  }),
  preset('Coral Reverie', {
    warpFn: WARP.flow, warpAmt: 0.8, kaleido: 5, fluid: 1, decayA: 0.982, decayB: 0.97,
    palette: 'analogous', waveStyle: WAVE.circle, hueOffset: -0.08, spawn: SPAWN.anywhere,
    rot: 0.125, kalRot: 0.125, emitter: 'orbit', emitterCount: 5,
  }),
  preset('Neon Monsoon', {
    energy: 'high', warpFn: WARP.sine, warpAmt: 1.2, zoom: 1.015, fluid: 1.3, splat: 1.5,
    emitter: 'jets', palette: 'triad', waveStyle: WAVE.dual, mirror: true, spawn: SPAWN.anywhere,
    decayA: 0.97, decayB: 0.96, rot: 1, rotFlip: true, hueDrift: 0.003,
  }),
];

/** Scratch object holding the interpolated parameters for the current frame. */
export function makeParams(): NumParams {
  const o = {} as NumParams;
  for (let i = 0; i < NUM_KEYS.length; i++) o[NUM_KEYS[i]] = 0;
  return o;
}

export function lerpParams(out: NumParams, a: NumParams, b: NumParams, t: number): void {
  for (let i = 0; i < NUM_KEYS.length; i++) {
    const k = NUM_KEYS[i];
    out[k] = a[k] + (b[k] - a[k]) * t;
  }
}

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
