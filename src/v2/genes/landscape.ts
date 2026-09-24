// The song as a landscape: the 'landscape' shape gene (a full-screen field shape, one per genome).
//
// Before playback the song's analysis becomes a world map of its timeline (src/analysis/songWorld.ts:
// altitude from energy, a climb and a mountain pass at every drop, valleys in breakdowns, one terrain
// motif per section type, the key there). The shape ray-marches that world as a heightfield in its
// own reduced-resolution pass: along-track distance is song time, so the camera travels through the
// song and what it sees on the horizon is the music still to come (`look` seconds of it). Each section
// start gets a landmark beside the path; the next drop glows above its pass like a rising sun.
//
// Like the 'scene' shape, the pass writes lit / shade / glow / rim channels that the body's field
// function maps through its material, colour mapping and the palette; from there it is an ordinary
// field body (placement folds, deformation, the chain, the carrier and the tone all apply).
//
// This file has no runtime imports from genome.ts or engine.ts (genome.ts imports the schema from here).

import type { MaterialKind, Params, Schema } from '../genome';
import type { Frame } from '../engine';
import { worldAt, type SongWorld } from '../../analysis/songWorld';
import { sceneField } from './raymarch';

const P = (min: number, max: number, def: number) => ({ min, max, def });
const C = (choices: number[], def: number) => ({ min: Math.min(...choices), max: Math.max(...choices), def, choices });

/** Path styles: how the camera travels. */
export const LAND_PATHS = ['road', 'river', 'flight', 'rail'] as const;
/** Terrain styles. */
export const LAND_TERRAINS = ['hills', 'crystal', 'dunes', 'city', 'ribbons'] as const;
/** Landmarks at each section start. */
export const LAND_MARKS = ['obelisks', 'gates', 'rings', 'beacons'] as const;

/**
 * path: LAND_PATHS; ground: LAND_TERRAINS; mark: LAND_MARKS; res: internal resolution (cost goes
 * with its square); look: seconds of the song visible ahead (how far away the horizon is in time);
 * height: camera height; relief: how strongly energy maps to altitude; rough: terrain detail;
 * wind: how much the path meanders; fog: horizon haze; glow: path lights, landmarks and the drop's
 * sun; tint: how strongly key and mode tint the ground; kick: camera bump on drum hits; rim: ridge light.
 */
export const LAND_SCHEMA: Schema = {
  path: C([0, 1, 2, 3], 0),
  ground: C([0, 1, 2, 3, 4], 0),
  mark: C([0, 1, 2, 3], 0),
  res: C([0.35, 0.5, 0.7], 0.5),
  look: C([10, 16, 24, 32, 40], 24),
  height: P(0, 1, 0.4),
  relief: P(0, 1, 0.6),
  rough: P(0, 1, 0.5),
  wind: P(0, 1, 0.5),
  fog: P(0, 1, 0.4),
  glow: P(0, 1, 0.5),
  tint: P(0, 1, 0.5),
  kick: P(0, 1, 0.3),
  rim: P(0, 1, 0.5),
};
/** Structural switches reactions may not touch (look sets the world scale: reacting it would jump the camera). */
export const LAND_NO_REACT = ['path', 'ground', 'mark', 'res', 'look'];

/** Full-resolution (2560x1440) march cost per ground style, ms (benchmarked against the scene pass); the pass costs this times res squared. */
const LAND_FULL_MS = [17, 14, 14, 15, 12];

/** Estimated GPU ms of a landscape body: the reduced-resolution march plus the upsampling field lookup. */
export function landCost(p: Params): number {
  const full = LAND_FULL_MS[p.ground] ?? LAND_FULL_MS[0];
  return 0.3 + full * p.res * p.res;
}

// ------------------------------------------------------------ world geometry (shared with GLSL)

/** World units of altitude per unit of the map's altitude at relief 1. */
export const LAND_ALT = 3.0;
/** Height of the mountains flanking a drop's pass. */
export const LAND_PEAK = 4.5;
/** Far clip of the march (world units). */
export const LAND_TMAX = 36;
/** Half width of the flat corridor per path style. */
const PATH_W = [0.45, 0.55, 0.9, 0.35];
/** World units per second of song: the horizon is `look` seconds away. */
export const landK = (look: number) => 24 / Math.max(look, 1);
/** Sideways position of the path at along-track z. */
export const pathX = (z: number, wind: number) => wind * (1.8 * Math.sin(0.13 * z) + 0.8 * Math.sin(0.31 * z + 1.7));
/** Ground altitude on the path at song time t (the corridor is flat, so this is exact there). */
export const groundAt = (w: SongWorld, t: number, relief: number) => relief * LAND_ALT * worldAt(w, t, 0);

// ------------------------------------------------------------------ GLSL

/** vec4 slots of the landscape pass (uLs). */
export const LAND_VEC4 = 24;
/** Landmarks passed per frame. */
export const LAND_MAX_MARKS = 6;

/**
 * The landscape pass fragment body (after the shared library). uLs:
 *   0 camera position xyz, focal     1 camera target xyz, roll
 *   2 k (units per second), t0, span, samples of the world map
 *   3 relief, rough, wind, fog       4 path, terrain, mark, glow
 *   5 tint, rim, beat pulse, bass    6 anim time, song time now, key hue now, mark count
 *   7 next drop: z, x, ground y, strength
 *   8-13 landmarks: z, x, ground y, label + scale (fract)
 */
export const LAND_PASS = /* glsl */ `
in vec2 vUv;
uniform vec4 uLs[${LAND_VEC4}];
uniform sampler2D uWorld;
out vec4 o;
const float LALT = ${LAND_ALT.toFixed(2)}, LPEAK = ${LAND_PEAK.toFixed(2)}, LTMAX = ${LAND_TMAX.toFixed(1)};
float lsK, lsT0, lsSpan, lsN, lsCw, lsFine;
int lsPath, lsTer;
float lsPathX(float z) { return uLs[3].z * (1.8 * sin(0.13 * z) + 0.8 * sin(0.31 * z + 1.7)); }
// The world map at song time t: row r interpolated; nb = the nearer sample, df = t minus its time.
vec4 lsW(int r, float t, out vec4 nb, out float df) {
  float dts = lsSpan / (lsN - 1.0);
  float u = clamp((t - lsT0) / dts, 0.0, lsN - 1.0);
  float i = min(floor(u), lsN - 2.0);
  float f = u - i;
  vec4 a = texelFetch(uWorld, ivec2(int(i), r), 0);
  vec4 b = texelFetch(uWorld, ivec2(int(i) + 1, r), 0);
  nb = f < 0.5 ? a : b;
  df = (t - lsT0) - (f < 0.5 ? i : i + 1.0) * dts;
  return mix(a, b, f);
}
float lsDetail(vec2 p, float motif) {
  p += motif * vec2(37.1, 11.3);
  if (lsTer == 0) return 0.6 * vnoise(p * 0.35) + 0.3 * vnoise(p * 0.8 + 5.0) + (lsFine > 0.0 ? 0.1 * vnoise(p * 2.1) : 0.05);
  if (lsTer == 1) {
    float r = 1.0 - abs(vnoise(p * 0.45) * 2.0 - 1.0), r2 = lsFine > 0.0 ? 1.0 - abs(vnoise(p * 1.1 + 3.0) * 2.0 - 1.0) : 0.6;
    float h = r * r * 0.75 + r2 * r2 * 0.25;
    return mix(h, floor(h * 5.0) / 5.0, 0.35);
  }
  if (lsTer == 2) {
    float d = fract(p.y * 0.55 + 1.6 * vnoise(p * 0.18));
    return smoothstep(0.0, 0.75, d) * (1.0 - smoothstep(0.75, 1.0, d)) * (0.6 + 0.4 * vnoise(p * 0.3));
  }
  if (lsTer == 3) {
    vec2 c = floor(p * 0.9), f = fract(p * 0.9) - 0.5;
    float r = hash12(c + motif * 7.0);
    return (r > 0.3 ? r * r * 1.6 : 0.0) * step(max(abs(f.x), abs(f.y)), 0.36);
  }
  float q = (vnoise(p * 0.25) * 0.7 + (lsFine > 0.0 ? vnoise(p * 0.6) * 0.3 : 0.15)) * 6.0;
  return (floor(q) + smoothstep(0.8, 1.0, fract(q))) / 6.0;
}
// Terrain height at ground point q (x across, y along the track = song time * k). A point at height
// py well above the terrain's upper bound gets the bound instead (no noise evaluated): still a safe
// step for the march, and most samples along a ray are such points.
float lsH(vec2 q, float py) {
  vec4 nb; float df;
  vec4 w = lsW(0, q.y / lsK, nb, df);
  float adx = abs(q.x - lsPathX(q.y));
  float side = smoothstep(lsCw, lsCw * 3.0 + 0.6, adx);
  float local = nb.w + df;
  float detA = uLs[3].y * (lsTer == 3 ? 1.8 : 1.3) * side * (0.35 + 0.65 * smoothstep(0.0, 1.5, local));
  float pkA = w.y * LPEAK * smoothstep(lsCw, lsCw + 2.5, adx) * exp(-max(adx - 3.5, 0.0) * 0.25);
  float h = uLs[3].x * LALT * w.x + 0.35 * side * smoothstep(0.0, 8.0, adx) * (0.5 + uLs[3].y);
  if (lsPath == 1) h -= 0.3 * (1.0 - side);
  float hb = h + detA * 1.6 + pkA;
  if (py > hb + 0.05) return hb;
  return h + lsDetail(vec2(q.x, local * lsK), nb.z) * detA + pkA * (0.7 + 0.3 * vnoise(q * 0.9));
}
// Ray against an axis-aligned box (centre c, half size b): entry distance and normal, or -1.
float lsBox(vec3 ro, vec3 rd, vec3 c, vec3 b, out vec3 n) {
  vec3 m = 1.0 / rd, k = abs(m) * b, q = m * (ro - c);
  vec3 t1 = -q - k, t2 = -q + k;
  float tn = max(max(t1.x, t1.y), t1.z), tf = min(min(t2.x, t2.y), t2.z);
  if (tn > tf || tf < 0.0) return -1.0;
  n = -sign(rd) * step(t1.yzx, t1.xyz) * step(t1.zxy, t1.xyz);
  return tn;
}
void main() {
  vec4 C0 = uLs[0], C1 = uLs[1];
  lsK = uLs[2].x; lsT0 = uLs[2].y; lsSpan = uLs[2].z; lsN = uLs[2].w;
  lsPath = int(uLs[4].x + 0.5); lsTer = int(uLs[4].y + 0.5);
  lsCw = lsPath == 0 ? ${PATH_W[0]} : lsPath == 1 ? ${PATH_W[1]} : lsPath == 2 ? ${PATH_W[2]} : ${PATH_W[3]};
  float glowP = uLs[4].w, fogP = uLs[3].w, beat = uLs[5].z;
  vec2 sp = (vUv - 0.5) * vec2(uAspect, 1.0);
  vec3 ro = C0.xyz;
  vec3 cw = normalize(C1.xyz - ro);
  vec3 cu = normalize(cross(cw, vec3(sin(C1.w), cos(C1.w), 0.0)));
  vec3 cv = cross(cu, cw);
  vec3 rd = normalize(sp.x * cu + sp.y * cv + C0.w * cw);
  // Heightfield march: conservative steps (city blocks cap the step so rays cannot skip a tower).
  float maxStep = lsTer == 3 ? 0.3 : 1.2;
  // Nothing is higher than hmax: rays start at that ceiling and leave once back above it going up.
  float hmax = uLs[14].x;
  float t = ro.y > hmax && rd.y < 0.0 ? (ro.y - hmax) / -rd.y : 0.05, tp = t;
  bool hit = false;
  for (int i = 0; i < 80; i++) {
    vec3 p = ro + rd * t;
    lsFine = 8.0 - t;
    if (t > LTMAX || (rd.y >= 0.0 && p.y > hmax)) break;
    float d = p.y - lsH(p.xz, p.y);
    if (d < 0.003 * t) { hit = true; break; }
    tp = t;
    // Grazing rays close in on the ground slowly by the height gap alone, so the step never falls
    // below ~1.2% of the distance; an overshoot is found by the bisection below.
    t += clamp(d * (lsTer == 3 ? 0.5 : 0.8), 0.015 + 0.02 * t, maxStep);
  }
  lsFine = 8.0 - t;
  if (!hit && t <= LTMAX && tp < t) {
    vec3 p = ro + rd * t;
    hit = p.y - lsH(p.xz, p.y) < 0.0;
  }
  if (hit) {
    for (int i = 0; i < 5; i++) {
      float tm = 0.5 * (t + tp);
      vec3 p = ro + rd * tm;
      if (p.y - lsH(p.xz, p.y) < 0.0) t = tm; else tp = tm;
    }
  } else t = LTMAX;
  hit = hit && t < LTMAX;
  // Sky: horizon haze, and the next drop's pass glowing like a rising sun.
  vec4 D = uLs[7];
  vec3 sunDir = normalize(vec3(D.y, D.z + 4.5, D.x) - ro);
  float sun = D.w * (pow(max(dot(rd, sunDir), 0.0), 40.0) * 1.6 + pow(max(dot(rd, sunDir), 0.0), 6.0) * 0.25);
  float haze = (0.12 + 0.5 * fogP) * exp(-abs(rd.y) * 7.0);
  float sky = (haze + sun) * (0.4 + glowP);
  float lit = 0.0, shade = 0.5, glow = sky, rim = 0.0;
  if (hit) {
    vec3 pos = ro + rd * t;
    float e = 0.012 + 0.002 * t;
    float h0 = lsH(pos.xz, -1e9);
    vec3 n = normalize(vec3(h0 - lsH(pos.xz + vec2(e, 0.0), -1e9), e, h0 - lsH(pos.xz + vec2(0.0, e), -1e9)));
    vec4 nb, nb1; float df, df1;
    vec4 w0 = lsW(0, pos.z / lsK, nb, df);
    vec4 w1 = lsW(1, pos.z / lsK, nb1, df1);
    float dx = pos.x - lsPathX(pos.z), adx = abs(dx);
    float corr = 1.0 - smoothstep(lsCw * 0.8, lsCw * 1.4, adx);
    vec3 ld = normalize(vec3(-0.5, 0.65, 0.55));
    float dif = clamp(dot(n, ld), 0.0, 1.0);
    lit = 0.8 * dif + 0.2;
    float dk = w1.x - uLs[6].z;
    dk -= floor(dk + 0.5);
    float relH = (h0 - uLs[3].x * LALT * w0.x) / (LPEAK * 0.6 + 1.0);
    shade = 0.11 * nb.z + 0.2 * relH + uLs[5].x * (dk + 0.08 * nb1.y);
    lit *= 1.0 - 0.25 * uLs[5].x * nb1.y;
    rim = pow(1.0 - clamp(dot(n, -rd), 0.0, 1.0), 3.0) * uLs[5].y;
    float lw = 0.02 + 0.004 * t;
    float pg = 0.0;
    if (lsPath == 0) {
      pg = (1.0 - smoothstep(lw * 0.5, lw, adx)) * step(fract(pos.z * 1.5), 0.5) * (0.5 + beat)
        + 0.4 * exp(-pow((adx - lsCw * 0.85) / lw, 2.0));
      lit *= mix(1.0, 0.45, corr);
    } else if (lsPath == 1) {
      float fres = pow(1.0 - clamp(-rd.y, 0.0, 1.0), 4.0);
      float flow = vnoise(vec2(dx * 6.0, pos.z * 2.0 - uLs[6].x * 1.5));
      lit *= mix(1.0, 0.3, corr);
      rim += corr * (0.3 + 0.7 * fres) * (0.6 + 0.4 * flow);
      pg = corr * (haze * 2.0 + 0.25 * smoothstep(0.75, 0.95, flow) * (0.5 + beat));
    } else if (lsPath == 3) {
      pg = exp(-pow((adx - 0.22) / lw, 2.0)) * (0.6 + 0.6 * beat);
      lit *= mix(1.0, 0.5 + 0.5 * step(0.3, fract(pos.z * 3.0)), corr);
    }
    glow = pg * glowP;
  }
  // Landmarks at section starts (analytic, drawn over the terrain when nearer).
  int mk = int(uLs[4].z + 0.5);
  int nm = int(uLs[6].w + 0.5);
  for (int j = 0; j < ${LAND_MAX_MARKS}; j++) {
    if (j >= nm) break;
    vec4 M = uLs[8 + j];
    float lab = floor(M.w), sc = fract(M.w);
    if (mk == 0 || mk == 1) {
      float H = mk == 0 ? 1.5 + 3.0 * sc : 1.2 + 1.5 * sc;
      float off = mk == 0 ? lsCw + 0.7 : lsCw + 0.2;
      float wid = mk == 0 ? 0.18 : 0.12;
      for (int s = 0; s < 3; s++) {
        vec3 bn;
        vec3 c = s < 2 ? vec3(M.y + (s == 0 ? -off : off), M.z + H * 0.5 - 0.5, M.x) : vec3(M.y, M.z + H, M.x);
        vec3 b = s < 2 ? vec3(wid, H * 0.5 + 0.5, wid) : vec3(off + wid, 0.12, 0.12);
        if (s == 2 && mk == 0) break;
        float tb = lsBox(ro, rd, c, b, bn);
        if (tb > 0.0 && tb < t) {
          t = tb;
          hit = true;
          float dif = clamp(dot(bn, normalize(vec3(-0.5, 0.65, 0.55))), 0.0, 1.0);
          lit = 0.7 * dif + 0.15;
          shade = 0.11 * lab + 0.5;
          rim = 0.6 * uLs[5].y + 0.2;
          vec3 bp = ro + rd * tb;
          glow = glowP * (0.3 + beat) * smoothstep(c.y + b.y - 0.25, c.y + b.y, bp.y);
        }
      }
    } else if (mk == 2) {
      if (abs(rd.z) > 1e-4) {
        float tr = (M.x - ro.z) / rd.z;
        if (tr > 0.0 && tr < t) {
          vec3 p = ro + rd * tr;
          float R = 0.9 + 1.2 * sc;
          float r = length(p.xy - vec2(M.y, M.z + R * 0.9));
          float wr = 0.05 + 0.006 * tr;
          glow += glowP * (0.6 + beat) * 1.5 * exp(-pow((r - R) / wr, 2.0));
        }
      }
    } else {
      for (int s = 0; s < 2; s++) {
        vec2 c = vec2(M.y + (s == 0 ? -1.0 : 1.0) * (lsCw + 1.0), M.x);
        vec2 dr = normalize(rd.xz + 1e-5);
        float tc = dot(c - ro.xz, dr) / length(rd.xz + 1e-5);
        if (tc > 0.0 && tc < t) {
          float dd = length(ro.xz + rd.xz * tc - c);
          float y = ro.y + rd.y * tc - M.z;
          glow += glowP * (0.5 + beat) * 1.2 * exp(-pow(dd / (0.04 + 0.01 * tc), 2.0)) * smoothstep(-0.2, 0.3, y) * exp(-max(y, 0.0) * 0.12) * (0.6 + 0.4 * sc);
        }
      }
    }
  }
  // Horizon fog: far ground (and anything near the far clip) melts into the sky's glow.
  float fogv = hit ? exp(-t * t * fogP * 0.0025) * (1.0 - smoothstep(LTMAX * 0.75, LTMAX, t)) : 0.0;
  if (hit) glow = glow * fogv + sky * (1.0 - fogv);
  o = vec4(lit * fogv, shade, glow, rim * fogv);
}`;

/** The field function of a landscape body: samples the landscape texture, lit through the material. */
export function landField(material: MaterialKind): string {
  return sceneField(material).replace(/uScene/g, 'uLand');
}

// ------------------------------------------------------------ packing

export interface LandCtx {
  F: Frame;
  sdt: number;
  /** Parameter with reactions applied. */
  P: (k: string) => number;
  raw: Params;
  mem: Record<string, number>;
  /** Prefix for this body's persistent state in mem. */
  key: string;
  world: SongWorld;
  /** Song time now (seconds): playback position, or the live clock. */
  now: number;
  /** Key hue now (0..1). */
  keyHue: number;
}

const CAM_H = [[0.25, 0.8], [0.15, 0.5], [1.0, 3.0], [0.45, 1.2]];

/** Fills the landscape pass uniforms (LAND_VEC4 vec4s) for this frame. */
export function packLandscape(out: Float32Array, c: LandCtx): void {
  const { F, sdt, P, mem, world: w, now } = c;
  const k = (s: string) => `${c.key}${s}`;
  const get = (s: string, init = 0) => mem[k(s)] ?? (mem[k(s)] = init);
  const set = (s: string, v: number) => (mem[k(s)] = v);
  const path = c.raw.path;
  const look = c.raw.look;
  const K = landK(look);
  const relief = P('relief');
  const wind = P('wind');
  const height = P('height');
  // Camera: on the path at the song's position, riding above the ground there (and just ahead, so a
  // climb never clips it), easing down over a drop's cliff.
  const z = now * K;
  const ch = CAM_H[path] ?? CAM_H[0];
  const g = Math.max(groundAt(w, now, relief), groundAt(w, now + 0.6, relief)) - (path === 1 ? 0.3 : 0);
  const targetY = g + ch[0] + ch[1] * height;
  const y0 = get('y', targetY);
  const y = set('y', Number.isFinite(y0) && Math.abs(y0 - targetY) < 20 ? approach(y0, targetY, path === 2 ? 1.5 : 5, path === 2 ? 1 : 2.5, sdt) : targetY);
  const kick = set('kick', Math.max(get('kick') * Math.exp(-sdt * 8), F.onset[0] * F.gate[0] * P('kick')));
  const x = pathX(z, wind);
  const ahead = path === 2 ? 7 : 5;
  const zt = z + ahead;
  const gt = groundAt(w, now + ahead / K, relief);
  const yt = path === 2 ? y - 1.4 - height : Math.min(y, gt + ch[0] + ch[1] * height * 0.6) - 0.25;
  // Bank into the path's turns (flight most, rail not at all).
  const curv = pathX(z + 1.5, wind) - 2 * x + pathX(z - 1.5, wind);
  const bankK = [0.25, 0.15, 1.1, 0][path] ?? 0;
  const roll = set('roll', approach(get('roll'), -curv * bankK, 2, 2, sdt));
  out.fill(0);
  out[0] = x; out[1] = y - 0.12 * kick; out[2] = z;
  out[3] = (path === 2 ? 1.3 : 1.5) * (1 + 0.08 * kick);
  out[4] = pathX(zt, wind); out[5] = yt; out[6] = zt; out[7] = roll;
  out[8] = K; out[9] = w.t0; out[10] = w.span; out[11] = w.n;
  out[12] = relief; out[13] = P('rough'); out[14] = wind; out[15] = P('fog');
  out[16] = path; out[17] = c.raw.ground; out[18] = c.raw.mark; out[19] = P('glow');
  out[20] = P('tint'); out[21] = P('rim'); out[22] = F.beatPulse * F.gate[0]; out[23] = F.stem[1];
  const T = set('t', (get('t') + sdt) % 4096);
  out[24] = T; out[25] = now; out[26] = c.keyHue;
  // The next drop within sight: its sun rises as it comes over the horizon and sets once passed.
  let dz = 0, ds = 0;
  for (const d of w.drops) {
    const ahead = d - now;
    if (ahead < -2 || ahead > look * 1.3) continue;
    dz = d;
    ds = ahead < 0 ? 1 + ahead / 2 : smoothstep((look * 1.3 - ahead) / (look * 0.4));
    break;
  }
  if (ds > 0) {
    out[28] = dz * K; out[29] = pathX(dz * K, wind); out[30] = groundAt(w, dz, relief); out[31] = ds;
  }
  // Landmarks: the section starts from just behind the camera to past the horizon.
  let m = 0;
  for (const mk of w.marks) {
    if (m >= LAND_MAX_MARKS) break;
    if (mk.t < now - 1.5 || mk.t > now + look * 1.15) continue;
    const o = (8 + m) * 4;
    const mz = mk.t * K;
    out[o] = mz; out[o + 1] = pathX(mz, wind); out[o + 2] = groundAt(w, mk.t, relief) - (path === 1 ? 0.3 : 0);
    out[o + 3] = mk.label + Math.min(0.99, Math.max(0, mk.energy));
    m++;
  }
  out[27] = m;
  out[56] = landCeiling(w, relief);
}

const CEIL = new WeakMap<SongWorld, { v: number; hi: number }>();
/** Highest terrain anywhere in the world (a ceiling for the march): the top altitude plus every extra. */
function landCeiling(w: SongWorld, relief: number): number {
  let c = CEIL.get(w);
  if (!c || c.v !== w.version) {
    let hi = 0;
    for (let i = 0; i < w.n; i++) hi = Math.max(hi, w.data[i * 4]);
    CEIL.set(w, (c = { v: w.version, hi }));
  }
  return relief * LAND_ALT * c.hi + LAND_PEAK + 1.8 * 1.6 + 1.2 + 0.05;
}

function smoothstep(x: number): number {
  const t = Math.min(1, Math.max(0, x));
  return t * t * (3 - 2 * t);
}

/** Eases x toward y with separate rise / fall rates (per second). */
function approach(x: number, y: number, up: number, down: number, dt: number): number {
  return x + (y - x) * (1 - Math.exp(-dt * (y > x ? up : down)));
}
