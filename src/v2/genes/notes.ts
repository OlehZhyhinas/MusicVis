// Notes: a full-screen chunk shape ('field' class) that draws the melody's articulation, the way
// the notes are played. Time runs along a lane (or round a ring), pitch across it:
//
//   ribbon   a held or gliding note is a continuous ribbon that bends with its pitch as it wanders
//            (rising and falling with a glide, wiggling with vibrato, which also makes it shimmer),
//            its width and brightness following the held note's strength; in a gap it breaks off.
//   marks    every note start stamps a mark at its pitch (a dot, a spark, a piano-roll bar as long
//            as the note, or a ripple ring) that stays lit while the note is held and fades once it
//            ends, so "tu tu tu" notes read as a row of distinct marks.
//
// The legato signal moves the weight between the two: while the melody is played legato the
// ribbon carries it and the marks step back; played staccato the marks take over and the ribbon
// thins. The picture shifts between the two as the song does.
//
// The data come from MusicState.notes (src/analysis/notes.ts) through a small texture that
// NoteHistory fills each frame (engine.ts Signals): row 0 is the last NOTE_SEC seconds of the
// melody sampled every NOTE_SEC / NOTE_W s (height, held strength, vibrato, legato; newest first),
// row 1 the recent notes (age, length, height, strength, negative once ended).

import type { ParamSpec, Schema } from '../genome';
import type { NoteStats } from '../../types';

const P = (min: number, max: number, def: number): ParamSpec => ({ min, max, def });
const C = (choices: number[], def: number): ParamSpec => ({ min: Math.min(...choices), max: Math.max(...choices), def, choices });

/** History texture width (samples) and the seconds it spans. */
export const NOTE_W = 512;
export const NOTE_SEC = 8;
/** Marks held in row 1 of the texture (the shader reads NOTE_MARKS of them). */
export const NOTE_MARKS = 12;

/**
 * mode: 0 a lane (time along x), 1 a ring (time round the clock, pitch as radius); span: seconds of
 * melody in view; len: lane length / ring size (screen heights); height: the pitch range across
 * (screen heights); now: where the present sits along the lane (-0.5 left .. 0.5 right); tilt:
 * rotation (turns); ribbon: brightness of held notes; thick: ribbon half width; marks: brightness of
 * the note-start marks; form: 0 dot, 1 spark, 2 bar as long as the note, 3 ripple ring; size: mark
 * size; fade: seconds a mark takes to fade once its note ends; rise: marks drift across the lane
 * as they age (screen heights / s, either way); shimmer: vibrato sparkle on the ribbon; hues: hue
 * spread over the pitch range; glow: halo.
 */
export const NOTES_SCHEMA: Schema = {
  mode: C([0, 1], 0),
  span: P(1, 8, 4),
  len: P(0.4, 1.8, 1.2),
  height: P(0.15, 0.9, 0.55),
  now: P(-0.5, 0.5, 0.3),
  tilt: P(-0.25, 0.25, 0),
  ribbon: P(0, 1, 0.8),
  thick: P(0.002, 0.03, 0.008),
  marks: P(0, 1, 0.8),
  form: C([0, 1, 2, 3], 0),
  size: P(0.005, 0.08, 0.025),
  fade: P(0.05, 1.5, 0.35),
  rise: P(-0.3, 0.3, 0),
  shimmer: P(0, 1, 0.5),
  hues: P(0, 1, 0.35),
  glow: P(0, 1, 0.5),
};

/** Estimated GPU ms at 1440p: one full-screen pass, three history taps and twelve marks per pixel. */
export const NOTES_COST = 0.6;

/**
 * vec3 FLD(vec2 p): EA = (gain, hue, legato now, time), EB = (mode, span, len, height),
 * EC = (now, tilt rad, ribbon, thick), ED = (marks, form, size, fade), BD(2) = (rise, shimmer, hues,
 * glow), BD(3) = (vibrato now, held now, -, -). uNote: the history texture.
 */
export const NOTES_GLSL = /* glsl */ `
vec4 ntAt(float age) {
  float u = (clamp(age / ${NOTE_SEC.toFixed(1)}, 0.0, 1.0) * ${(NOTE_W - 1).toFixed(1)} + 0.5) / ${NOTE_W.toFixed(1)};
  return texture(uNote, vec2(u, 0.25));
}
// The present, the time axis and the pitch axis: a point of the lane / ring at (age, across).
vec2 ntPos(float age, float across, vec4 B, vec4 C) {
  if (B.x < 0.5) return vec2(C.x * B.z - age * B.z / B.y, across);
  float R = B.z * 0.3;
  float a = age / B.y * TAU;
  return (R + across) * vec2(sin(a), cos(a));
}
vec3 FLD(vec2 p) {
  vec4 A = EA, B = EB, C = EC, D = ED, X = BD(2), Y = BD(3);
  float span = max(B.y, 0.5);
  vec2 q = rot2(-C.y) * p;
  float aa = px();
  bool ring = B.x > 0.5;
  float R = B.z * 0.3;
  // Screen units per second along the axis, and across per unit of pitch height.
  float k = ring ? R * TAU / span : B.z / span;
  float ys = ring ? B.w * 0.5 : B.w;
  float age, across;
  if (ring) {
    float an = atan(q.x, q.y);
    age = mod(an, TAU) / TAU * span;
    across = length(q) - R;
  } else {
    age = (C.x * B.z - q.x) / k;
    across = q.y;
  }
  vec3 c = vec3(0.0);
  // Ribbon: the held melody, bending with its pitch.
  if (age >= 0.0 && age <= span) {
    float dA = ${((NOTE_SEC / NOTE_W) * 2).toFixed(4)};
    vec4 s0 = ntAt(age), sp = ntAt(age + dA), sm = ntAt(max(age - dA, 0.0));
    float slope = (sp.r - sm.r) * ys / (2.0 * dA * k);
    float d = abs(across - (s0.r - 0.5) * ys) / sqrt(1.0 + slope * slope);
    // A jump between notes is not drawn (the texture would interpolate a wall across it); a glide is.
    float held = s0.g * smoothstep(0.06, 0.025, abs(sp.r - sm.r));
    float w = C.w * (0.35 + 0.65 * held) + aa;
    float old = smoothstep(span, span * 0.55, age);
    vec3 col = pal(A.y + s0.r * X.z);
    // The halo takes the held strength around this point, so a note's ends are rounded off.
    float soft = (held + sp.g + sm.g) / 3.0 * smoothstep(0.06, 0.025, abs(sp.r - sm.r));
    float art = C.z * (0.3 + 0.7 * s0.a) * old;
    c += col * (glow(d, w) * held + X.w * 0.3 * glow(d, w * 2.2) * soft) * art;
    // Vibrato: small beads of light twinkling along the ribbon where the note wavers.
    float ph = age * k / max(C.w * 6.0, 0.01);
    float bead = glow(fract(ph) - 0.5, 0.12) * (0.5 + 0.5 * sin(floor(ph) * 2.7 + A.w * 9.0));
    c += mix(col, vec3(1.0), 0.6) * glow(d, w * 0.9) * bead * X.y * smoothstep(0.15, 0.6, s0.b) * held * old * 2.0;
  }
  // The head: where the melody is now.
  {
    vec4 s0 = ntAt(0.0);
    vec2 h = ntPos(0.0, (s0.r - 0.5) * ys, B, C);
    float w = C.w * 2.0 + aa;
    c += pal(A.y + s0.r * X.z) * (glow(length(q - h), w) * 1.2 + glow(length(q - h), w * 2.5) * 0.3 * X.w) * Y.y * C.z;
  }
  // Marks: one per note start, lit while the note is held, fading once it ends.
  float mw = D.x * (1.0 - 0.65 * A.z);
  for (int i = 0; i < ${NOTE_MARKS}; i++) {
    vec4 m = texelFetch(uNote, ivec2(i, 1), 0);
    float st = abs(m.a);
    if (st <= 0.0 || m.r > span) continue;
    float mage = m.r;
    float env = m.a < 0.0 ? exp(-max(0.0, mage - m.g) / max(D.w, 0.02)) : 0.75 + 0.25 * exp(-mage / 0.12);
    float flash = exp(-mage / 0.06);
    float bright = (st * env * mw + flash * D.x * 0.8) * smoothstep(span, span * 0.7, mage);
    if (bright < 0.003) continue;
    float acr = (m.b - 0.5) * ys + X.x * mage;
    vec2 mp = ntPos(mage, acr, B, C);
    float r = D.z * (0.6 + 0.6 * st) * (1.0 + 0.6 * flash);
    vec2 dq = q - mp;
    float dd = length(dq);
    float v;
    int form = int(D.y + 0.5);
    if (form == 1) {
      // Spark: four rays and a hot core, turned a little per note.
      vec2 e = rot2(m.b * 7.0 + mage * 2.0) * dq;
      v = glow(dd, r * 0.35 + aa) + (glow(abs(e.x), aa + r * 0.06) * glow(e.y, r * 1.8) + glow(abs(e.y), aa + r * 0.06) * glow(e.x, r * 1.8)) * 0.8;
    } else if (form == 2) {
      // Bar: a piano-roll capsule from the note's start to its end (or to now while held).
      vec2 me = ntPos(max(mage - m.g, 0.0), acr, B, C);
      float ds = sdSeg(q, mp, me);
      v = glow(ds, r * 0.45 + aa) + glow(ds, r * 1.4) * 0.25;
    } else if (form == 3) {
      // Ripple: a ring spreading from the note start.
      float rr = r * (0.4 + mage * 3.0);
      v = glow(abs(dd - rr), aa + r * 0.12) * exp(-mage * 1.5) * 1.3 + glow(dd, r * 0.3 + aa) * env;
    } else {
      v = glow(dd, r * 0.5 + aa) + glow(dd, r * 1.8) * 0.3;
    }
    v += glow(dd, r * 3.5) * 0.25 * X.w;
    c += mix(pal(A.y + m.b * X.z + 0.08), vec3(1.0), 0.35 * flash) * v * bright;
  }
  return c * A.x * uLayerK;
}`;

/**
 * The note texture's contents: row 0 the melody history, row 1 the recent notes. Engine Signals
 * pushes each frame's MusicState.notes and uploads `data` (NOTE_W x 2 RGBA).
 */
export class NoteHistory {
  readonly data = new Float32Array(NOTE_W * 2 * 4);
  private acc = 0;
  private clock = 0;
  private lastStart = -1e9;
  private pendingBreak = false;
  private last = [0.5, 0, 0, 0.5];

  push(n: NoteStats | undefined, dt: number): void {
    const d = this.data;
    const step = NOTE_SEC / NOTE_W;
    this.acc += Math.max(0, dt);
    const steps = Math.min(NOTE_W, Math.floor(this.acc / step));
    this.acc -= steps * step;
    if (this.acc > step) this.acc = 0;
    const cur = n
      ? [clamp01(n.height), clamp01(n.held), clamp01(n.vibrato / 0.6), clamp01(n.legato)]
      : [this.last[0], 0, 0, 0.5];
    // A new note breaks the ribbon for one sample (tied notes stepping in pitch do not join up).
    this.clock += Math.max(0, dt);
    const newest = n?.recent.length ? n.recent[n.recent.length - 1] : null;
    if (newest) {
      const start = this.clock - newest.age;
      if (Math.abs(start - this.lastStart) > 0.02) {
        if (this.lastStart > -1e8) this.pendingBreak = true;
        this.lastStart = start;
      }
    }
    const broke = this.pendingBreak && steps > 0;
    if (steps > 0) this.pendingBreak = false;
    for (let s = 1; s <= steps; s++) {
      d.copyWithin(4, 0, (NOTE_W - 1) * 4);
      // Samples between frames are interpolated, except across a jump in pitch or a new note.
      const f = s / steps;
      const jump = broke || Math.abs(cur[0] - this.last[0]) > 0.04;
      for (let j = 0; j < 4; j++) d[j] = jump ? cur[j] : this.last[j] + (cur[j] - this.last[j]) * f;
      if (broke && s === 1) d[1] = 0;
    }
    if (steps > 0) this.last = cur;
    const o = NOTE_W * 4;
    d.fill(0, o, o + NOTE_MARKS * 4);
    const rec = n?.recent ?? [];
    const k0 = Math.max(0, rec.length - NOTE_MARKS);
    for (let i = k0; i < rec.length; i++) {
      const m = rec[i];
      const j = o + (i - k0) * 4;
      const st = Math.max(0.05, clamp01(m.strength));
      d[j] = m.age;
      d[j + 1] = m.len;
      d[j + 2] = clamp01(m.height);
      d[j + 3] = m.ended ? -st : st;
    }
  }

  reset(): void {
    this.data.fill(0);
    this.acc = 0;
    this.clock = 0;
    this.lastStart = -1e9;
    this.pendingBreak = false;
    this.last = [0.5, 0, 0, 0.5];
  }
}

const clamp01 = (x: number) => (Number.isFinite(x) ? (x < 0 ? 0 : x > 1 ? 1 : x) : 0);

/**
 * Packs the field slots. E: uniforms; o: slot 16 (EA) of the body; o2 / o3: slots 2 and 3; P: a
 * parameter after reactions; mem / key: per-body memory.
 */
export function packNotes(
  E: Float32Array, o: number, o2: number, o3: number, P: (k: string) => number, p: Record<string, number>, n: NoteStats | undefined,
  mem: Record<string, number>, key: (k: string) => string, sdt: number,
): void {
  const mm = (k: string, init = 0) => mem[key(k)] ?? (mem[key(k)] = init);
  const set = (k: string, v: number) => (mem[key(k)] = v);
  // Legato now, eased so the weight between ribbon and marks shifts over about a second.
  const leg = set('leg', mm('leg', 0.5) + (clamp01(n?.legato ?? 0.5) - mm('leg', 0.5)) * (1 - Math.exp(-sdt * 1.5)));
  const t = set('t', (mm('t') + sdt) % 1024);
  E[o + 2] = leg;
  E[o + 3] = t;
  E[o + 4] = p.mode; E[o + 5] = P('span'); E[o + 6] = P('len'); E[o + 7] = P('height');
  E[o + 8] = P('now'); E[o + 9] = P('tilt') * Math.PI * 2; E[o + 10] = P('ribbon'); E[o + 11] = P('thick');
  E[o + 12] = P('marks'); E[o + 13] = p.form; E[o + 14] = P('size'); E[o + 15] = P('fade');
  E[o2] = P('rise'); E[o2 + 1] = P('shimmer'); E[o2 + 2] = P('hues'); E[o2 + 3] = P('glow');
  E[o3] = clamp01((n?.vibrato ?? 0) / 0.6); E[o3 + 1] = clamp01(n?.held ?? 0); E[o3 + 2] = 0; E[o3 + 3] = 0;
}
