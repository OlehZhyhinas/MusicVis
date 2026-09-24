// Ray-marched 3D scenes: the 'scene' shape gene (a full-screen chunk shape, one per genome).
//
// The scene is ray-marched in its own pass at a reduced internal resolution (the res gene, a
// fraction of the stage size) into a half-float texture; the body's field function then samples
// that texture (bilinear upsample) and maps its channels through the body's material, colour
// mapping and the palette. From there it is an ordinary field body: placement folds, deformation,
// the chain, the feedback carrier and the tone all apply to the picture.
//
// Scene texture channels: r = lit surface (diffuse, ambient, occlusion, fog), g = shade (a hue
// coordinate: which primitive, facing), b = glow (near misses along the ray), a = rim light.
//
// This file has no runtime imports from genome.ts (genome.ts imports the schema from here).

import type { MaterialKind, Params, Schema } from '../genome';
import type { Frame } from '../engine';

const P = (min: number, max: number, def: number) => ({ min, max, def });
const C = (choices: number[], def: number) => ({ min: Math.min(...choices), max: Math.max(...choices), def, choices });

/** Scene kinds: 0 smooth-union primitives, 1 an endless lattice of shapes between two plates, 2 a tunnel. */
export const SCENE_KINDS = ['primitives', 'lattice', 'tunnel'] as const;
/** Camera kinds: 0 orbit, 1 flythrough, 2 dolly zoom. */
export const SCENE_CAMS = ['orbit', 'fly', 'dolly'] as const;

/**
 * scene: what is ray-marched (SCENE_KINDS); cam: how the camera moves (SCENE_CAMS);
 * res: internal resolution as a fraction of the stage (cost goes with its square);
 * size: object scale (bass pulses it by `pulse`); blend: smooth-union radius; speed: animation and
 * camera speed; roam: how far the camera travels (orbit swing, flight path, dolly depth); kick: camera jolt on drum hits; vary: how far each song section reshuffles the scene;
 * rim / ao / fog / glow: lighting terms. Lattice: gap = height of the corridor between the plates,
 * spec = how far each cell's shape rises with its spectrum band. Tunnel: gap = how much it bends,
 * spec = how far its ribs close in with their spectrum bands.
 */
export const SCENE_SCHEMA: Schema = {
  scene: C([0, 1, 2], 0),
  cam: C([0, 1, 2], 0),
  res: C([0.35, 0.5, 0.7], 0.5),
  size: P(0.4, 1.6, 1),
  blend: P(0, 1, 0.5),
  speed: P(0, 1, 0.4),
  pulse: P(0, 1, 0.5),
  kick: P(0, 1, 0.4),
  vary: P(0, 1, 0.5),
  rim: P(0, 1, 0.5),
  ao: P(0, 1, 0.6),
  fog: P(0, 1, 0.4),
  glow: P(0, 1, 0.3),
  roam: P(0, 1, 0.5),
  gap: P(0, 1, 0.5),
  spec: P(0, 1, 0.6),
};
/** Structural switches reactions may not touch. */
export const SCENE_NO_REACT = ['scene', 'cam', 'res'];

/**
 * Full-resolution (2560x1440) march cost per scene kind in the cost model's units (ms on the
 * calibration machine); the pass costs this times res squared. Scaled from timer-query readings on
 * an M-series Mac against the wireframe solid (E14) as reference; kept on the high side because the
 * readings were noisy on a shared GPU.
 */
const SCENE_FULL_MS = [8.0, 8.0, 7.0];

/** Estimated GPU ms of a scene body: the reduced-resolution march plus the upsampling field lookup. */
export function sceneCost(p: Params): number {
  const full = SCENE_FULL_MS[p.scene] ?? SCENE_FULL_MS[0];
  return 0.25 + full * p.res * p.res;
}

// ------------------------------------------------------------------ GLSL

/** vec4 slots of the scene pass (uScn). */
export const SCENE_VEC4 = 32;

/**
 * The scene pass fragment body (after the shared library). uScn:
 *   0 camera position xyz, focal length    1 camera target xyz, roll
 *   2 time, size, blend, variation seed      3 rim, ao, fog, glow
 *   4 scene kind, camera kind, -, -
 *   5 lattice: cell size, floor y, ceiling y, -   6 lattice: shape radius, seed, spectrum lift, time
 *   5 tunnel: radius, rib spacing, bend, seed      6 tunnel: rib lift, wall wobble, time, -
 *   8-12 primitives: centre xyz, radius   13-27 primitives: rotation rows (3 per primitive)
 *   28 primitive kinds (0 sphere, 1 torus, 2 box, 3 octahedron) for 0-3, 29.x kind of 4
 */
export const SCENE_PASS = /* glsl */ `
in vec2 vUv;
uniform vec4 uScn[${SCENE_VEC4}];
out vec4 o;
float rmBox(vec3 p, vec3 b) { vec3 q = abs(p) - b; return length(max(q, 0.0)) + min(max(q.x, max(q.y, q.z)), 0.0); }
float rmTorus(vec3 p, vec2 t) { vec2 q = vec2(length(p.xz) - t.x, p.y); return length(q) - t.y; }
float rmOcta(vec3 p, float s) { p = abs(p); return (p.x + p.y + p.z - s) * 0.57735; }
// Smooth-union primitives: five shapes on slow paths melting into each other (placed per frame in JS).
float rmPrim(int kind, vec3 q, float r) {
  if (kind == 0) return length(q) - r;
  if (kind == 1) return rmTorus(q, vec2(r, r * 0.33));
  if (kind == 2) return rmBox(q, vec3(r * 0.72)) - r * 0.08;
  return rmOcta(q, r * 1.25);
}
vec2 rmPrims(vec3 p) {
  float k = max(uScn[2].z, 1e-3);
  float d = 1e5, sh = 0.0;
  for (int i = 0; i < 5; i++) {
    vec4 c = uScn[8 + i];
    vec3 q = p - c.xyz;
    q = vec3(dot(uScn[13 + i * 3].xyz, q), dot(uScn[14 + i * 3].xyz, q), dot(uScn[15 + i * 3].xyz, q));
    int kind = int((i < 4 ? uScn[28][i] : uScn[29].x) + 0.5);
    float di = rmPrim(kind, q, c.w);
    float h = clamp(0.5 + 0.5 * (di - d) / k, 0.0, 1.0);
    d = mix(di, d, h) - k * h * (1.0 - h);
    sh = mix(float(i) * 0.2, sh, h);
  }
  return vec2(d, sh);
}
// An endless lattice: a plate below and a plate above, each carrying one shape per cell; every cell
// rises with its own spectrum band and turns at its own pace.
vec2 rmLattice(vec3 p) {
  vec4 L = uScn[5], Q = uScn[6];
  float c = L.x;
  bool top = p.y > 0.5 * (L.y + L.z);
  float yy = top ? L.z - p.y : p.y - L.y;
  vec2 id = floor(p.xz / c);
  vec2 q2 = (fract(p.xz / c) - 0.5) * c;
  float h = hash12(id + (top ? 37.0 : 0.0) + floor(Q.y * 8.0) * 13.0);
  float band = specAt(fract(h * 7.31) * 0.7 + 0.03);
  float r = Q.x * (0.7 + 0.5 * h);
  vec3 q = vec3(q2.x, yy - r * 1.2 - Q.z * band * c, q2.y);
  q.xz = rot2(Q.w * (0.3 + h) + h * 6.0) * q.xz;
  float d = rmPrim(int(h * 4.0), q, r);
  // Never step past the cell wall (the neighbour's shape may be nearer).
  d = min(d, 0.5 * c - max(abs(q2.x), abs(q2.y)) + 0.1 * c);
  float sh = 0.15 + 0.7 * h;
  if (yy < d) { d = yy; sh = 0.95; }
  return vec2(d, sh);
}
// A tunnel bending along z: a ribbed, panelled wall; each rib closes in with its own spectrum band.
vec2 rmTunnelPath(float z) {
  vec4 A = uScn[5];
  return A.z * vec2(sin(z * 0.2 + A.w * TAU), 0.7 * cos(z * 0.16 + A.w * 4.0));
}
vec2 rmTunnel(vec3 p) {
  vec4 A = uScn[5], B = uScn[6];
  vec2 q = p.xy - rmTunnelPath(p.z);
  float r = length(q);
  float a = atan(q.y, q.x);
  float R = A.x * (1.0 + B.y * sin(a * 3.0 + B.z * 2.0) * 0.5);
  // Panels: a shallow angular relief on the wall.
  float wall = (R - r) + 0.04 * A.x * smoothstep(0.35, 0.5, abs(fract(a * 1.9099) - 0.5));
  float id = floor(p.z / A.y);
  float rz = (fract(p.z / A.y) - 0.5) * A.y;
  float band = specAt(fract(id * 0.137) * 0.7 + 0.03);
  float rib = length(vec2(r - (R - B.x * band - 0.06 * A.x), rz)) - 0.06 * A.x;
  float d = min(wall, rib) * 0.8;
  float sh = rib < wall ? 0.55 + 0.3 * fract(id * 0.37) : 0.1 + 0.15 * fract(a / TAU + id * 0.05);
  return vec2(d, sh);
}
vec2 rmMap(vec3 p) {
  if (uScn[4].x > 1.5) return rmTunnel(p);
  if (uScn[4].x > 0.5) return rmLattice(p);
  return rmPrims(p);
}
vec3 rmNormal(vec3 p, float t) {
  vec2 e = vec2(1.0, -1.0) * 0.0008 * (1.0 + t);
  return normalize(e.xyy * rmMap(p + e.xyy).x + e.yyx * rmMap(p + e.yyx).x + e.yxy * rmMap(p + e.yxy).x + e.xxx * rmMap(p + e.xxx).x);
}
float rmAO(vec3 p, vec3 n) {
  float occ = 0.0, w = 1.0;
  for (int i = 1; i <= 3; i++) {
    float h = 0.06 * float(i * i);
    occ += (h - rmMap(p + n * h).x) * w;
    w *= 0.6;
  }
  return clamp(1.0 - 2.2 * occ, 0.0, 1.0);
}
void main() {
  vec4 C0 = uScn[0], C1 = uScn[1], L = uScn[3];
  vec2 sp = (vUv - 0.5) * vec2(uAspect, 1.0);
  vec3 ro = C0.xyz;
  vec3 cw = normalize(C1.xyz - ro);
  vec3 cu = normalize(cross(cw, vec3(sin(C1.w), cos(C1.w), 0.0)));
  vec3 cv = cross(cu, cw);
  vec3 rd = normalize(sp.x * cu + sp.y * cv + C0.w * cw);
  float t = 0.02, glw = 0.0;
  vec2 h = vec2(1e5, 0.0);
  bool hit = false;
  const float TMAX = 24.0;
  for (int i = 0; i < 64; i++) {
    h = rmMap(ro + rd * t);
    glw += exp(-max(h.x, 0.0) * 14.0);
    if (h.x < 0.0012 * t) { hit = true; break; }
    t += h.x * 0.9;
    if (t > TMAX) break;
  }
  float glow = glw * 0.012 * L.w;
  if (!hit) { o = vec4(0.0, 0.0, glow, 0.0); return; }
  vec3 pos = ro + rd * t;
  vec3 n = rmNormal(pos, t);
  vec3 ld = normalize(vec3(0.6, 0.8, -0.3));
  // Key light from above plus a headlight along the view, so corridors and undersides still read.
  float dif = clamp(0.65 * dot(n, ld) + 0.45 * dot(n, -rd), 0.0, 1.0);
  float ao = mix(1.0, rmAO(pos, n), L.y);
  float fogv = exp(-t * t * L.z * 0.012);
  float rim = pow(1.0 - clamp(dot(n, -rd), 0.0, 1.0), 3.0) * L.x;
  float lit = (0.85 * dif + 0.18) * ao * fogv;
  float shade = h.y + 0.12 * dot(n, vec3(0.5, 0.3, -0.4));
  o = vec4(lit, shade, glow, rim * fogv * (0.4 + 0.6 * ao));
}`;

/** The field function of a scene body: samples the scene texture, lit through the material. */
export function sceneField(material: MaterialKind): string {
  const look: Record<MaterialKind, string> = {
    fill: 'base * s.x + mix(base, vec3(1.0), 0.5) * s.w * 0.8 + alt * s.z',
    line: 'base * s.w * 2.2 + base * s.x * 0.1 + alt * s.z',
    glow: 'base * s.x * 0.55 + alt * s.z * 2.5 + base * s.w * 0.6',
    dots: 'base * smoothstep(0.02, -0.02, length(fract(uv * uRes / (5.0 * uRes.y / 1080.0)) - 0.5) - 0.5 * sqrt(clamp(s.x, 0.0, 1.0))) * 1.3 + base * s.w * 0.5 + alt * s.z',
    textured: 'base * s.x * (0.55 + 0.45 * sin(s.y * 40.0)) + base * s.w + alt * s.z',
    chrome: 'pal(h + s.w * 0.8 + s.x * 0.3) * (s.x * 0.7 + s.w * 1.2) + vec3(pow(clamp(s.x, 0.0, 1.0), 8.0)) * 0.7 + alt * s.z',
  };
  return /* glsl */ `
uniform sampler2D uScene;
vec3 FLD(vec2 p) {
  vec2 uv = p / vec2(uAspect, 1.0) + 0.5;
  if (uv.x < 0.0 || uv.y < 0.0 || uv.x > 1.0 || uv.y > 1.0) return vec3(0.0);
  vec4 s = texture(uScene, uv);
  vec4 A = EA;
  float h = A.y + s.y * BD(20).y;
  vec3 base = pal(h), alt = pal(h + 0.33);
  vec3 c = ${look[material]};
  return max(c, 0.0) * A.x * uLayerK;
}`;
}

// ------------------------------------------------------------ packing

export interface SceneCtx {
  F: Frame;
  sdt: number;
  /** Parameter with reactions applied. */
  P: (k: string) => number;
  raw: Params;
  mem: Record<string, number>;
  /** Prefix for this body's persistent state in mem. */
  key: string;
}

const TAU = Math.PI * 2;
const hash = (n: number) => {
  const x = Math.sin(n * 127.1 + 311.7) * 43758.5453;
  return x - Math.floor(x);
};

/** Fills the scene pass uniforms (SCENE_VEC4 vec4s) for this frame. */
export function packScene(out: Float32Array, c: SceneCtx): void {
  const { F, sdt, P, mem } = c;
  const k = (s: string) => `${c.key}${s}`;
  const get = (s: string, init = 0) => mem[k(s)] ?? (mem[k(s)] = init);
  const set = (s: string, v: number) => (mem[k(s)] = v);
  const speed = P('speed');
  // Scene time: runs with the tempo and the activity, a little faster when loud.
  const T = set('t', (get('t') + sdt * F.speed * (0.15 + 0.85 * speed) * (0.6 + 0.4 * F.act + 0.3 * F.loud)) % 4096);
  // Bass swells the objects; the beat adds a small kick.
  const bass = set('bass', approach(get('bass'), F.stem[1], 10, 3, sdt));
  const pulse = P('pulse');
  const size = P('size') * (1 + pulse * (0.3 * bass + 0.08 * F.beatPulse * F.gate[0]));
  // Section changes reshuffle the arrangement (seed drifts to the new section's value over about a bar).
  const target = hash(F.sectionIndex + 1) * P('vary');
  const seed = set('seed', approach(get('seed', target), target, 1.5, 1.5, sdt));
  // Drum hits jolt the camera toward the scene and roll it slightly.
  const kick = set('kick', Math.max(get('kick') * Math.exp(-sdt * 7), F.onset[0] * F.gate[0] * P('kick')));
  const side = set('side', F.onset[0] > 0.6 && get('kick') < 0.05 ? -get('side', 1) : get('side', 1));
  out.fill(0);
  if (c.raw.scene === 2) tunnelCamera(out, c, T, kick, seed);
  else camera(out, c, T, kick, side);
  out[8] = T; out[9] = size; out[10] = P('blend'); out[11] = seed;
  out[12] = P('rim'); out[13] = P('ao'); out[14] = P('fog'); out[15] = P('glow');
  out[16] = c.raw.scene; out[17] = c.raw.cam;
  out[10] = (0.03 + 1.07 * P('blend')) * size;
  if (c.raw.scene === 1) {
    const L = latticeFrame(P);
    out[20] = L.cell; out[21] = L.floor; out[22] = L.ceil;
    out[24] = L.cell * 0.22 * (size / P('size')); out[25] = seed; out[26] = P('spec') * 0.6; out[27] = T;
  } else if (c.raw.scene === 2) {
    const U = tunnelFrame(P, seed);
    out[20] = U.R; out[21] = U.ribs; out[22] = U.bend; out[23] = seed;
    out[24] = U.lift; out[25] = P('pulse') * 0.16 * bass; out[26] = T;
  } else placePrims(out, T, size, seed);
}

/** The tunnel's path repeats over TUNNEL_WRAP (its frequencies and rib spacing divide it), so travel can wrap. */
const TUNNEL_WRAP = (TAU * 100) / 0.2;

/** Tunnel geometry: radius, rib spacing (dividing the wrap), bend amplitude, rib lift and the free radius. */
function tunnelFrame(P: (k: string) => number, seed: number): { R: number; ribs: number; bend: number; lift: number; free: number; seed: number } {
  const R = 1.3 * P('size');
  const ribs = TUNNEL_WRAP / Math.round(TUNNEL_WRAP / (1.1 * R));
  const lift = 0.35 * R * P('spec');
  // Wall wobble (0.08 R at most), rib lift and rib thickness leave this radius clear.
  const free = R * (1 - 0.08 - 0.06 - 0.12) - lift;
  return { R, ribs, bend: R * (0.3 + 2.2 * P('gap')), lift, free, seed };
}
function tunnelPath(U: { bend: number; seed: number }, z: number): [number, number] {
  return [U.bend * Math.sin(z * 0.2 + U.seed * TAU), U.bend * 0.7 * Math.cos(z * 0.16 + U.seed * 4)];
}

/**
 * Tunnel cameras, all travelling down the tunnel on its bending path: orbit circles the axis as it goes;
 * fly weaves and banks; dolly holds the axis while the lens breathes over four bars (a travelling vertigo
 * shot). Drum hits surge the travel forward. The camera stays inside the free radius.
 */
function tunnelCamera(out: Float32Array, c: SceneCtx, T: number, kick: number, seed: number): void {
  const { F, sdt, P, mem } = c;
  const k = (s: string) => `${c.key}${s}`;
  const get = (s: string, init = 0) => mem[k(s)] ?? (mem[k(s)] = init);
  const set = (s: string, v: number) => (mem[k(s)] = v);
  const speed = P('speed'), roam = P('roam');
  const U = tunnelFrame(P, seed);
  const z = set('travel', (get('travel') + sdt * F.speed * (0.8 + 4 * speed) * U.R * (1 + 1.5 * kick)) % TUNNEL_WRAP);
  const off = 0.55 * Math.max(0, U.free) * roam;
  let ox = 0, oy = 0, roll = 0, f = 1.2;
  if (c.raw.cam === 0) {
    const a = T * 0.35;
    ox = Math.cos(a) * off; oy = Math.sin(a) * off;
    roll = a * 0.25;
  } else if (c.raw.cam === 1) {
    ox = Math.sin(T * 0.7) * off; oy = Math.cos(T * 0.53) * off * 0.7;
    roll = 0.5 * Math.cos(T * 0.7) * roam;
  } else {
    f = 0.8 + 1.4 * (0.5 - 0.5 * Math.cos((F.bars / 4) * TAU)) * (0.4 + 0.6 * roam) + 0.3 * kick;
  }
  const [cx, cy] = tunnelPath(U, z);
  const [ax, ay] = tunnelPath(U, z + 2.5 * U.R);
  out[0] = cx + ox; out[1] = cy + oy; out[2] = z; out[3] = f;
  out[4] = ax + ox * 0.3; out[5] = ay + oy * 0.3; out[6] = z + 2.5 * U.R; out[7] = roll;
}

/** Lattice geometry: cell size, the two plates and the free corridor between their tallest shapes. */
function latticeFrame(P: (k: string) => number): { cell: number; floor: number; ceil: number; lo: number; hi: number } {
  const cell = 1.4 * P('size');
  // Tallest shape: a lifted, pulsed, largest-hash shape (radius 0.22 cell x 1.2 x pulse, reaching 2.6 radii up; lift 0.6 cell).
  const top = cell * (0.22 * 1.2 * 2.6 * (1 + 0.38 * P('pulse')) + 0.6 * P('spec'));
  const half = top + (0.6 + 2.5 * P('gap')) * P('size');
  return { cell, floor: -half, ceil: half, lo: -half + top + 0.25, hi: half - top - 0.25 };
}

/**
 * The camera (uScn 0-1: position, focal length, target, roll).
 *   orbit  circles the scene, rising and sinking by roam; drum hits push it in.
 *   fly    weaves a looping path around and between the objects looking along it, banking into the turns;
 *          drum hits lunge it forward along the path.
 *   dolly  faces the scene and dollies in and out over four bars while the focal length follows, so
 *          the subject holds its size and the space behind it stretches (the vertigo shot); drum hits
 *          punch it in.
 */
function camera(out: Float32Array, c: SceneCtx, T: number, kick: number, side: number): void {
  const { F, sdt, P, mem } = c;
  const k = (s: string) => `${c.key}${s}`;
  const get = (s: string, init = 0) => mem[k(s)] ?? (mem[k(s)] = init);
  const set = (s: string, v: number) => (mem[k(s)] = v);
  const speed = P('speed'), roam = P('roam');
  const base = 0.6 + 0.4 * P('size');
  // How far the objects reach from the centre at their largest (paths keep clear of it).
  const ext = P('size') * 1.75 * (1 + 0.38 * P('pulse'));
  const ang = set('ang', (get('ang') + sdt * F.speed * (0.05 + 0.5 * speed)) % (TAU * 64));
  let px: number, py: number, pz: number, tx = 0, ty = 0, tz = 0, f = 1.6, roll = 0.12 * kick * side;
  if (c.raw.cam === 1) {
    // The path phase runs with the tempo; kicks add a burst of travel.
    const s = set('fly', (get('fly') + sdt * F.speed * (0.08 + 0.4 * speed) + kick * sdt * 2.5) % (TAU * 64));
    const R = (ext + 0.6) / 0.7 + 1.5 * roam * base;
    const at = (u: number): [number, number, number] => [R * Math.sin(u), 0.35 * R * Math.sin(3 * u + 0.7), 0.75 * R * Math.cos(u) + 0.25 * R * Math.sin(2 * u)];
    [px, py, pz] = at(s);
    const [ax, ay, az] = at(s + 0.35);
    // Look ahead, drawn a little toward the centre so the objects stay in view.
    tx = ax * 0.3; ty = ay * 0.3; tz = az * 0.3;
    const [bx, , bz] = at(s + 0.7);
    roll += 0.5 * Math.max(-1, Math.min(1, ((bx - ax) * (az - pz) - (bz - az) * (ax - px)) / (R * R * 0.05)));
    f = 1.3 + 0.4 * kick;
  } else if (c.raw.cam === 2) {
    const w = 0.5 - 0.5 * Math.cos((F.bars / 4) * TAU);
    const d = ext + 1 + (1.2 + 4.5 * roam * w) * base;
    const dist = d - 0.8 * kick * base;
    const a = ang * 0.3;
    px = Math.cos(a) * dist; py = 0.8 * base; pz = Math.sin(a) * dist;
    f = (1.6 * d) / (ext + 2.2 * base);
  } else {
    const elev = 0.35 + 0.6 * roam * Math.sin(T * 0.11);
    const dist = ext + 1.6 + 1.6 * base * (1 - 0.6 * kick);
    px = Math.cos(ang) * Math.cos(elev) * dist;
    py = Math.sin(elev) * dist;
    pz = Math.sin(ang) * Math.cos(elev) * dist;
  }
  if (c.raw.scene === 1) {
    // The lattice is endless: the camera cruises forward and stays in the corridor between the plates.
    const L = latticeFrame(P);
    const D = set('travel', (get('travel') + sdt * F.speed * (0.4 + 2.2 * speed) * L.cell) % (L.cell * 512));
    const mid = 0.5 * (L.lo + L.hi), span = Math.max(0, 0.5 * (L.hi - L.lo));
    const squash = (y: number) => mid + span * Math.tanh(y / Math.max(span, 1e-3));
    py = squash(py); ty = squash(ty) * 0.6;
    pz += D; tz += D;
  }
  out[0] = px; out[1] = py; out[2] = pz; out[3] = f;
  out[4] = tx; out[5] = ty; out[6] = tz; out[7] = roll;
}

/** Five primitives on slow Lissajous paths, each tumbling; the section seed picks kinds and phases. */
function placePrims(out: Float32Array, T: number, sz: number, seed: number): void {
  for (let i = 0; i < 5; i++) {
    const ph = i * 1.2566 + seed * TAU * (0.3 + 0.13 * i);
    const o = (8 + i) * 4;
    out[o] = Math.sin(T * 0.7 * (1 + 0.1 * i) + ph) * 1.05 * sz;
    out[o + 1] = 0.7 * Math.sin(T * 0.53 * (1 + 0.07 * i) + ph * 1.7) * 1.05 * sz;
    out[o + 2] = Math.cos(T * 0.61 + ph * 2.3) * 1.05 * sz;
    out[o + 3] = sz * (0.42 + 0.1 * Math.sin(i * 3.1 + seed * 5));
    rotRows(out, (13 + i * 3) * 4, T * 0.4 + i, T * 0.3 + i * 2);
    const kind = (i + Math.floor(seed * 4)) % 4;
    if (i < 4) out[28 * 4 + i] = kind;
    else out[29 * 4] = kind;
  }
}

/** Rows of a rotation (about z by a, then about x by b) into three vec4 slots starting at o. */
function rotRows(out: Float32Array, o: number, a: number, b: number): void {
  const ca = Math.cos(a), sa = Math.sin(a), cb = Math.cos(b), sb = Math.sin(b);
  out[o] = ca; out[o + 1] = sa; out[o + 2] = 0;
  out[o + 4] = -sa * cb; out[o + 5] = ca * cb; out[o + 6] = sb;
  out[o + 8] = sa * sb; out[o + 9] = -ca * sb; out[o + 10] = cb;
}

/** Eases x toward y with separate rise / fall rates (per second). */
function approach(x: number, y: number, up: number, down: number, dt: number): number {
  return x + (y - x) * (1 - Math.exp(-dt * (y > x ? up : down)));
}
