// Fractal flames (Draves' flame algorithm) on the GPU. Every point runs the
// chaos game: pick a weighted affine transform, apply it, then a weighted sum
// of variations, and blend its colour coordinate toward the transform's
// colour. Points are plotted additively into the owning preset's feedback
// buffer, whose composite applies log-density tonemapping.
//
// The variations are standalone GLSL functions `vec2 V_<name>(vec2 p)`, listed
// in FLAME_VARIATIONS in the same order as their weight slots, so they can be
// reused as generic space transforms elsewhere.

import { Fullscreen, GL, PingPong, Program, formats } from './gl';

/** Variation names, in weight-slot order (4 per vec4, 3 vec4 per transform). */
export const FLAME_VARIATIONS = [
  'linear', 'sinusoidal', 'spherical', 'swirl',
  'horseshoe', 'polar', 'handkerchief', 'heart',
  'disc', 'spiral', 'hyperbolic', 'julia',
] as const;
export type FlameVar = (typeof FLAME_VARIATIONS)[number];
export const MAX_XFORMS = 4;

/** One GLSL function per variation (flam3 definitions; theta = atan(x, y)). */
export const FLAME_VARIATION_GLSL = /* glsl */ `
float flameHash(vec2 p) { return fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453); }
vec2 V_linear(vec2 p) { return p; }
vec2 V_sinusoidal(vec2 p) { return sin(p); }
vec2 V_spherical(vec2 p) { return p / (dot(p, p) + 1e-6); }
vec2 V_swirl(vec2 p) {
  float r2 = dot(p, p);
  float s = sin(r2), c = cos(r2);
  return vec2(p.x * s - p.y * c, p.x * c + p.y * s);
}
vec2 V_horseshoe(vec2 p) {
  float r = length(p) + 1e-6;
  return vec2((p.x - p.y) * (p.x + p.y), 2.0 * p.x * p.y) / r;
}
vec2 V_polar(vec2 p) { return vec2(atan(p.x, p.y) / 3.14159265, length(p) - 1.0); }
vec2 V_handkerchief(vec2 p) {
  float r = length(p), t = atan(p.x, p.y);
  return r * vec2(sin(t + r), cos(t - r));
}
vec2 V_heart(vec2 p) {
  float r = length(p), t = atan(p.x, p.y);
  return r * vec2(sin(t * r), -cos(t * r));
}
vec2 V_disc(vec2 p) {
  float r = length(p), t = atan(p.x, p.y) / 3.14159265;
  return t * vec2(sin(3.14159265 * r), cos(3.14159265 * r));
}
vec2 V_spiral(vec2 p) {
  float r = length(p) + 1e-6, t = atan(p.x, p.y);
  return vec2(cos(t) + sin(r), sin(t) - cos(r)) / r;
}
vec2 V_hyperbolic(vec2 p) {
  float r = length(p) + 1e-6, t = atan(p.x, p.y);
  return vec2(sin(t) / r, r * cos(t));
}
vec2 V_julia(vec2 p) {
  float r = sqrt(length(p)), t = atan(p.x, p.y) * 0.5 + step(0.5, flameHash(p * 311.7)) * 3.14159265;
  return r * vec2(cos(t), sin(t));
}
`;

const HEAD = `#version 300 es
precision highp float;
precision highp int;
precision highp sampler2D;
`;

const UPDATE_FS = HEAD + FLAME_VARIATION_GLSL + /* glsl */ `
uniform sampler2D uS;
uniform float uFrame;
uniform int uN, uIters;
uniform vec4 uA[4];    // a b c d : x' = a x + b y + e, y' = c x + d y + f
uniform vec4 uB[4];    // e f colour cumulative-weight
uniform vec4 uVar[12]; // 12 variation weights per transform
out vec4 o;

float h(vec2 p) {
  vec3 p3 = fract(vec3(p.xyx) * 0.1031);
  p3 += dot(p3, p3.yzx + 33.33);
  return fract((p3.x + p3.y) * p3.z);
}

vec2 vars(int i, vec2 p) {
  vec4 a = uVar[i * 3], b = uVar[i * 3 + 1], c = uVar[i * 3 + 2];
  vec2 s = vec2(0.0);
  if (a.x != 0.0) s += a.x * V_linear(p);
  if (a.y != 0.0) s += a.y * V_sinusoidal(p);
  if (a.z != 0.0) s += a.z * V_spherical(p);
  if (a.w != 0.0) s += a.w * V_swirl(p);
  if (b.x != 0.0) s += b.x * V_horseshoe(p);
  if (b.y != 0.0) s += b.y * V_polar(p);
  if (b.z != 0.0) s += b.z * V_handkerchief(p);
  if (b.w != 0.0) s += b.w * V_heart(p);
  if (c.x != 0.0) s += c.x * V_disc(p);
  if (c.y != 0.0) s += c.y * V_spiral(p);
  if (c.z != 0.0) s += c.z * V_hyperbolic(p);
  if (c.w != 0.0) s += c.w * V_julia(p);
  return s;
}

void main() {
  vec2 ij = gl_FragCoord.xy;
  vec4 s = texelFetch(uS, ivec2(ij), 0);
  vec2 p = s.xy;
  float col = s.z;
  for (int k = 0; k < 8; k++) {
    if (k >= uIters) break;
    float r = h(ij * 0.7131 + vec2(uFrame * 1.37 + float(k) * 7.31, uFrame * 0.719 + float(k) * 3.1));
    int i = uN - 1;
    for (int j = 3; j >= 0; j--) if (j < uN && r < uB[j].w) i = j;
    vec4 A = uA[i], B = uB[i];
    vec2 q = vec2(A.x * p.x + A.y * p.y + B.x, A.z * p.x + A.w * p.y + B.y);
    p = vars(i, q);
    col = (col + B.z) * 0.5;
  }
  // Escaped or NaN points restart at a random spot (NaN fails every comparison).
  if (!(abs(p.x) < 50.0 && abs(p.y) < 50.0)) {
    p = vec2(h(ij + uFrame * 0.37), h(ij.yx + uFrame * 0.91)) * 2.0 - 1.0;
    col = h(ij * 1.3 + uFrame);
  }
  o = vec4(p, col, s.w);
}`;

const DRAW_VS = HEAD + /* glsl */ `
uniform sampler2D uS;
uniform int uW;
uniform vec4 uCam;      // cos*zoom, sin*zoom, offset x, offset y (p space)
uniform float uAspect, uWeight, uHueShift;
uniform vec3 uColA, uColB, uColC;
out vec3 vCol;
vec3 pal(float t) {
  t = fract(t) * 3.0;
  if (t < 1.0) return mix(uColA, uColB, t);
  if (t < 2.0) return mix(uColB, uColC, t - 1.0);
  return mix(uColC, uColA, t - 2.0);
}
void main() {
  vec4 s = texelFetch(uS, ivec2(gl_VertexID % uW, gl_VertexID / uW), 0);
  vec2 p = mat2(uCam.x, uCam.y, -uCam.y, uCam.x) * s.xy + uCam.zw;
  gl_Position = vec4(p.x / (uAspect * 0.5), p.y / 0.5, 0.0, 1.0);
  gl_PointSize = 1.0;
  vCol = pal(s.z * 0.999 + uHueShift) * uWeight;
}`;

const DRAW_FS = HEAD + /* glsl */ `
in vec3 vCol;
out vec4 o;
void main() { o = vec4(vCol, 1.0); }`;

const FULL_VS = HEAD + /* glsl */ `
void main() {
  vec2 p = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2));
  gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}`;

export interface FlameXform {
  /** Affine [a, b, c, d, e, f]: x' = a x + b y + e, y' = c x + d y + f. */
  aff: [number, number, number, number, number, number];
  weight: number;
  color: number; // colour coordinate 0..1
  vars: Partial<Record<FlameVar, number>>;
  /** Variation mix a drop morphs toward. */
  alt?: Partial<Record<FlameVar, number>>;
  /** Rotation of the linear part in turns per bar (bar-locked). */
  spin?: number;
  /** Scale added per unit of the bass stem. */
  bass?: number;
  /** Amplitude of a slow bar-locked drift of the translation (e, f). */
  drift?: [number, number];
  /** Scale kick per unit of beat pulse. */
  pulse?: number;
}

export interface FlameSpec {
  xforms: FlameXform[];
  count: number;
  iters: number; // chaos-game iterations per plotted round
  rounds: number; // update + plot rounds per frame
  zoom: number; // flame units to p space
  offset?: [number, number];
  camSpin?: number; // camera turns per bar
  gain: number;
  /** Continuous vars <-> alt morph cycles per 8 bars (0 or unset: drop-driven only). A drop reverses it. */
  flow?: number;
  /** Camera zoom added per unit of the bass stem (the flame breathes). */
  breathe?: number;
}

/** Music inputs for one flame frame. */
export interface FlameDrive {
  spin: number; // bar-locked angle, radians (1 turn per bar)
  bass: number;
  vocals: number;
  morph: number; // 0 = base variations, 1 = alt variations
  hue: number;
  bars?: number; // continuous musical time in bars (wraps at 48)
  beat?: number; // beat pulse 0..1
}

export class Flame {
  private state: PingPong | null = null;
  private pUpdate: Program;
  private pDraw: Program;
  private vao: WebGLVertexArrayObject;
  private side = 0;
  private frameNo = 0;
  private uA = new Float32Array(16);
  private uB = new Float32Array(16);
  private uVar = new Float32Array(48);
  private n = 1;
  private iters = 4;
  private cam = [1, 0, 0, 0];
  private hue = 0;

  constructor(
    private gl: GL,
    private fs: Fullscreen,
  ) {
    this.pUpdate = new Program(gl, FULL_VS, UPDATE_FS, 'flame-update');
    this.pDraw = new Program(gl, DRAW_VS, DRAW_FS, 'flame-draw');
    const v = gl.createVertexArray();
    if (!v) throw new Error('[render] cannot create VAO');
    this.vao = v;
  }

  get count(): number {
    return this.side * this.side;
  }

  setCount(n: number): void {
    const side = Math.max(64, Math.round(Math.sqrt(n)));
    if (side === this.side && this.state) return;
    this.state?.dispose();
    this.side = side;
    const total = side * side;
    const s = new Float32Array(total * 4);
    for (let i = 0; i < total; i++) {
      s[i * 4] = Math.random() * 2 - 1;
      s[i * 4 + 1] = Math.random() * 2 - 1;
      s[i * 4 + 2] = Math.random();
      s[i * 4 + 3] = Math.random();
    }
    this.state = new PingPong(this.gl, side, side, [formats(this.gl).rgba32f], this.gl.NEAREST, [s]);
  }

  /** Build this frame's transform uniforms from the spec and the music. */
  configure(spec: FlameSpec, d: FlameDrive): void {
    const xs = spec.xforms.slice(0, MAX_XFORMS);
    this.n = xs.length;
    this.iters = spec.iters;
    let total = 0;
    for (const x of xs) total += x.weight;
    let cum = 0;
    this.uVar.fill(0);
    const bars = d.bars ?? 0;
    const TAU = Math.PI * 2;
    let morph = d.morph;
    if (spec.flow) morph = Math.abs(0.5 - 0.5 * Math.cos((TAU * bars * spec.flow) / 8) - d.morph);
    xs.forEach((x, i) => {
      const ang = d.spin * (x.spin ?? 0);
      const ca = Math.cos(ang), sa = Math.sin(ang);
      const sc = 1 + (x.bass ?? 0) * d.bass + (x.pulse ?? 0) * (d.beat ?? 0);
      const [a, b, c, dd, e0, f0] = x.aff;
      // Periods of 16 and 12 bars divide the 48-bar wrap, so the drift is seamless.
      const e = e0 + (x.drift ? x.drift[0] * Math.sin((TAU * bars) / 16 + i * 2.1) : 0);
      const f = f0 + (x.drift ? x.drift[1] * Math.cos((TAU * bars) / 12 + i * 1.3) : 0);
      // Rotate the linear part: R * [a b; c d]
      this.uA[i * 4] = (ca * a - sa * c) * sc;
      this.uA[i * 4 + 1] = (ca * b - sa * dd) * sc;
      this.uA[i * 4 + 2] = (sa * a + ca * c) * sc;
      this.uA[i * 4 + 3] = (sa * b + ca * dd) * sc;
      cum += x.weight / total;
      this.uB[i * 4] = e;
      this.uB[i * 4 + 1] = f;
      this.uB[i * 4 + 2] = (x.color + 0.3 * d.vocals) % 1;
      this.uB[i * 4 + 3] = i === xs.length - 1 ? 1.01 : cum;
      FLAME_VARIATIONS.forEach((name, k) => {
        const w0 = x.vars[name] ?? 0;
        const w1 = x.alt ? x.alt[name] ?? 0 : w0;
        this.uVar[i * 12 + k] = w0 + (w1 - w0) * morph;
      });
    });
    const zoom = spec.zoom * (1 + (spec.breathe ?? 0) * d.bass);
    const ca = Math.cos(d.spin * (spec.camSpin ?? 0)) * zoom;
    const sa = Math.sin(d.spin * (spec.camSpin ?? 0)) * zoom;
    this.cam = [ca, sa, spec.offset?.[0] ?? 0, spec.offset?.[1] ?? 0];
    this.hue = d.hue;
  }

  update(): void {
    if (!this.state) return;
    const gl = this.gl;
    gl.disable(gl.BLEND);
    this.state.write.bind();
    this.pUpdate
      .use()
      .tex('uS', this.state.read.t)
      .f1('uFrame', (this.frameNo++ % 4096) + 0.5)
      .i1('uN', this.n)
      .i1('uIters', this.iters)
      .f4v('uA', this.uA)
      .f4v('uB', this.uB)
      .f4v('uVar', this.uVar);
    this.fs.draw();
    this.state.swap();
  }

  /** Plot every point once into the bound framebuffer (additive). */
  draw(weight: number, aspect: number, cols: Float32Array): void {
    if (!this.state) return;
    const gl = this.gl;
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.ONE, gl.ONE);
    this.pDraw
      .use()
      .tex('uS', this.state.read.t)
      .i1('uW', this.side)
      .f4('uCam', this.cam[0], this.cam[1], this.cam[2], this.cam[3])
      .f1('uAspect', aspect)
      .f1('uWeight', weight)
      .f1('uHueShift', this.hue)
      .f3('uColA', cols[0], cols[1], cols[2])
      .f3('uColB', cols[3], cols[4], cols[5])
      .f3('uColC', cols[6], cols[7], cols[8]);
    gl.bindVertexArray(this.vao);
    gl.drawArrays(gl.POINTS, 0, this.count);
  }

  dispose(): void {
    this.state?.dispose();
    this.state = null;
    this.pUpdate.dispose();
    this.pDraw.dispose();
    this.gl.deleteVertexArray(this.vao);
  }
}

/**
 * Composite for flame presets: box-filtered density, log tonemapped with the
 * colour ratio preserved (flam3 style).
 */
export const FLAME_COMP = /* glsl */ `
vec3 comp(vec2 uv, vec2 p) {
  vec2 o = 0.5 / uRes;
  vec3 c = (fb(uv + o) + fb(uv - o) + fb(uv + vec2(o.x, -o.y)) + fb(uv + vec2(-o.x, o.y))) * 0.25;
  float l = max(c.r, max(c.g, c.b));
  float b = log(1.0 + l * 24.0) / log(25.0);
  return c / max(l, 1e-5) * pow(b, 1.1) * 0.55;
}`;
