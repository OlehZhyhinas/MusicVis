// Tonnetz: a full-screen chunk shape ('field' class) that draws the harmony map itself. The tonal
// lattice fills the frame: a step right is a perfect fifth, a step up-right a major third, so every
// small triangle is a triad (pointing up: major, down: minor). Each note node glows with how much
// of that pitch class is sounding (the chroma), the current chord's triangle is lit, other places
// on the lattice holding the same chord echo it faintly, and the last few chords leave a glowing
// path, so the progression is drawn as a walk across the lattice. Home (the tonic triad) keeps a
// faint ring; the camera follows the walk as far as `follow` allows.
//
// Positions come from the harmony map (MusicState.tonnetzX/Y, relative to the key's tonic; see
// src/analysis/harmony.ts tonnetzXY), so the lattice is always drawn relative to home.

import type { ParamSpec, Schema } from '../genome';
import { tonnetzXY } from '../../analysis/harmony';

const P = (min: number, max: number, def: number): ParamSpec => ({ min, max, def });

/**
 * scale: lattice step (fraction of the screen height); follow: how far the camera follows the
 * walk away from home (0 = fixed on home); tilt: lattice rotation (turns); nodes: note nodes lit by
 * the chroma; lines: lattice line brightness; fill: the current chord's triangle; echo: other copies
 * of the same chord on the lattice; trail: the path of the last chords; pulse: a flash of the
 * triangle on each chord change and resolution.
 */
export const TONNETZ_SCHEMA: Schema = {
  scale: P(0.07, 0.3, 0.14),
  follow: P(0, 1, 0.6),
  tilt: P(-0.25, 0.25, 0),
  nodes: P(0, 1, 0.7),
  lines: P(0, 1, 0.35),
  fill: P(0, 1, 0.8),
  echo: P(0, 1, 0.25),
  trail: P(0, 1, 0.7),
  pulse: P(0, 1, 0.5),
};

/** Estimated GPU ms at 1440p: one full-screen pass, three node and three trail evaluations per pixel. */
export const TONNETZ_COST = 0.7;

/**
 * vec3 FLD(vec2 p): EA = (gain, hue, tension, time), EB = (scale, camera x, camera y, tilt rad),
 * EC = (chord root pc or -1, minor, tonic pc, fill now), ED = (nodes, lines, echo, trail),
 * BD(2) = (previous chord centre xy, the one before xy), BD(3) = (third previous xy, walk xy).
 * Lattice-plane coordinates: x along fifths, y up (a lattice step is 1).
 */
export const TONNETZ_GLSL = /* glsl */ `
vec2 tzPlane(vec2 ft) { return vec2(ft.x + 0.5 * ft.y, 0.8660254 * ft.y); }
vec3 FLD(vec2 p) {
  vec4 A = EA, B = EB, C = EC, D = ED, T0 = BD(2), T1 = BD(3);
  float S = max(B.x, 0.02);
  vec2 q = rot2(B.w) * p / S + B.yz;
  float t = q.y / 0.8660254;
  float f = q.x - 0.5 * t;
  vec2 ci = floor(vec2(f, t));
  vec2 fr = vec2(f, t) - ci;
  float up = step(fr.x + fr.y, 1.0);
  float aa = px() / S;
  vec3 c = vec3(0.0);
  // Lattice lines: fifths (t = const), major thirds (f = const), minor thirds (f + t = const).
  float dl = min(min(abs(t - floor(t + 0.5)), abs(f - floor(f + 0.5))), abs(f + t - floor(f + t + 0.5))) * 0.8660254;
  c += pal(A.y + 0.5) * glow(dl, 0.012 + aa) * D.y * 0.5;
  // The triangle under the pixel: its triad, centre and how deep inside we are.
  float root = up > 0.5 ? mod(7.0 * ci.x + 4.0 * ci.y + C.z, 12.0) : mod(7.0 * ci.x + 4.0 * (ci.y + 1.0) + C.z, 12.0);
  vec2 cen = tzPlane(ci + (up > 0.5 ? vec2(1.0 / 3.0) : vec2(2.0 / 3.0)));
  float inside = (up > 0.5 ? min(min(fr.x, fr.y), 1.0 - fr.x - fr.y) : min(min(1.0 - fr.x, 1.0 - fr.y), fr.x + fr.y - 1.0)) * 0.8660254;
  float same = step(0.0, C.x) * step(abs(root - C.x), 0.5) * step(abs((1.0 - up) - C.y), 0.5);
  float here = smoothstep(0.5, 0.15, length(cen - T1.zw));
  vec3 fillC = mix(pal(A.y), pal(A.y + 0.25), A.z);
  float body = smoothstep(0.0, 0.25, inside) * 1.1 + glow(inside, 0.03 + aa) * 1.4;
  c += fillC * body * C.w * here;
  c += fillC * body * D.z * same * (1.0 - here) * 0.4;
  // Home: a faint ring around the tonic triangle.
  c += pal(A.y + 0.1) * glow(abs(length(q - vec2(0.5, 0.2887)) - 0.36), 0.015 + aa) * 0.35 * D.y;
  // Note nodes at the triangle's corners, lit by the chroma.
  for (int k = 0; k < 3; k++) {
    vec2 v = up > 0.5 ? (k == 0 ? vec2(0.0) : k == 1 ? vec2(1.0, 0.0) : vec2(0.0, 1.0)) : (k == 0 ? vec2(1.0, 0.0) : k == 1 ? vec2(0.0, 1.0) : vec2(1.0));
    vec2 n = ci + v;
    float pc = mod(7.0 * n.x + 4.0 * n.y + C.z, 12.0);
    float lit = chromaAt(pc);
    float d = length(q - tzPlane(n));
    vec3 nc = mix(pal(A.y + lit * 0.3), keyCol(pc, 0.6, 1.0), 0.35);
    c += nc * (glow(d, 0.09 + aa) * (0.08 + 1.5 * lit * lit) + glow(abs(d - 0.11), 0.012 + aa) * 0.3 * lit) * D.x;
  }
  // The walk: the path through the last chords to the current one.
  float tr = glow(sdSeg(q, T1.zw, T0.xy), 0.035 + aa) + 0.6 * glow(sdSeg(q, T0.xy, T0.zw), 0.03 + aa) + 0.35 * glow(sdSeg(q, T0.zw, T1.xy), 0.025 + aa);
  tr += glow(length(q - T1.zw), 0.09 + aa) * 0.8;
  c += pal(A.y + 0.15) * tr * D.w;
  return c * A.x * uLayerK;
}`;

/** The engine state packTonnetz needs. */
export interface TonnetzFrame {
  chord: number;
  tonnetzX: number;
  tonnetzY: number;
  keyTonic: number;
  tension: number;
  chordPulse: number;
  resolve: number;
  loud: number;
}

const HOME: [number, number] = [0.5, 0.2887];

/**
 * Packs the field slots. E: uniforms; o: slot 16 (EA) of the body; o2 / o3: slots 2 and 3; P: a
 * parameter after reactions; mem / key: per-body memory.
 */
export function packTonnetz(
  E: Float32Array, o: number, o2: number, o3: number, P: (k: string) => number, p: Record<string, number>, f: TonnetzFrame,
  mem: Record<string, number>, key: (k: string) => string, sdt: number,
): void {
  const mm = (k: string, init = 0) => mem[key(k)] ?? (mem[key(k)] = init);
  const set = (k: string, v: number) => (mem[key(k)] = v);
  const wx = Number.isFinite(f.tonnetzX) ? f.tonnetzX : HOME[0];
  const wy = Number.isFinite(f.tonnetzY) ? f.tonnetzY : HOME[1];
  // Trail: when the chord changes, the previous chord's centre joins the path.
  const chord = f.chord >= 0 ? f.chord : -1;
  if (mem[key('ch')] === undefined) {
    set('ch', chord);
    for (const k of ['1', '2', '3']) {
      set('x' + k, wx);
      set('y' + k, wy);
    }
  }
  if (chord >= 0 && chord !== mm('ch')) {
    const prev = mm('ch');
    const [px, py] = prev >= 0 ? tonnetzXY(prev, f.keyTonic) : [wx, wy];
    set('x3', mm('x2')); set('y3', mm('y2'));
    set('x2', mm('x1')); set('y2', mm('y1'));
    set('x1', px); set('y1', py);
    set('ch', chord);
  }
  // Camera: follows the walk away from home as far as `follow` allows, easing.
  const fo = P('follow');
  const tx = HOME[0] + (wx - HOME[0]) * fo;
  const ty = HOME[1] + (wy - HOME[1]) * fo;
  const k = 1 - Math.exp(-sdt * 2.5);
  const cx = set('cx', mm('cx', tx) + (tx - mm('cx', tx)) * k);
  const cy = set('cy', mm('cy', ty) + (ty - mm('cy', ty)) * k);
  const flash = P('pulse') * Math.min(1, Math.max(f.chordPulse, f.resolve));
  const t = set('t', (mm('t') + sdt) % 1024);
  E[o] *= 0.6 + 0.4 * Math.min(1, f.loud * 1.5 + 0.3);
  E[o + 2] = Math.min(1, Math.max(0, f.tension));
  E[o + 3] = t;
  E[o + 4] = P('scale'); E[o + 5] = cx; E[o + 6] = cy; E[o + 7] = P('tilt') * Math.PI * 2;
  E[o + 8] = chord >= 0 ? chord % 12 : -1; E[o + 9] = chord >= 12 ? 1 : 0; E[o + 10] = ((f.keyTonic % 12) + 12) % 12; E[o + 11] = P('fill') * (1 + 1.5 * flash);
  E[o + 12] = P('nodes'); E[o + 13] = P('lines'); E[o + 14] = P('echo'); E[o + 15] = P('trail');
  E[o2] = mm('x1'); E[o2 + 1] = mm('y1'); E[o2 + 2] = mm('x2'); E[o2 + 3] = mm('y2');
  E[o3] = mm('x3'); E[o3 + 1] = mm('y3'); E[o3 + 2] = wx; E[o3 + 3] = wy;
}
