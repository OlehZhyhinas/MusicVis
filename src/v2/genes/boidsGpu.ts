// Boids on the GPU (WebGL2). Birds live in an RGBA32F ping-pong texture (x, y in uv, velocity in screen
// heights per second). Each frame:
//   1. splat: every bird is drawn as a point into a coarse grid (one cell per neighbourhood radius),
//      adding (1, vx, vy): the count and summed velocity of each cell;
//   2. update: a fragment pass over the birds reads the grid (bilinear, so neighbouring cells blend) and
//      its gradient, then aligns, coheres / separates, wanders, homes toward the body and moves.
// draw() adds the birds into the bound framebuffer (the slot's feedback) as soft points.

import { Fullscreen, GL, PingPong, Program, Target, formats, type TexFormat } from '../../render/gl';

const HEAD = /* glsl */ `#version 300 es
precision highp float;
precision highp int;
precision highp sampler2D;
`;
const FULL_VS = HEAD + /* glsl */ `
void main() {
  vec2 p = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2));
  gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}`;
const HASH = /* glsl */ `
const float TAU = 6.28318530718;
float hash12(vec2 p) {
  vec3 p3 = fract(vec3(p.xyx) * 0.1031);
  p3 += dot(p3, p3.yzx + 33.33);
  return fract((p3.x + p3.y) * p3.z);
}`;

const SPLAT_VS = HEAD + /* glsl */ `
uniform sampler2D uBirds;
uniform int uW;
out vec3 vV;
void main() {
  vec4 s = texelFetch(uBirds, ivec2(gl_VertexID % uW, gl_VertexID / uW), 0);
  gl_Position = vec4(s.xy * 2.0 - 1.0, 0.0, 1.0);
  gl_PointSize = 1.0;
  vV = vec3(1.0, s.zw);
}`;
const SPLAT_FS = HEAD + /* glsl */ `
in vec3 vV;
out vec4 o;
void main() { o = vec4(vV, 1.0); }`;

const UPDATE_FS = HEAD + HASH + /* glsl */ `
uniform sampler2D uBirds, uGrid;
uniform vec2 uGridTexel;
uniform float uAspect, uDt, uTime, uSpeed, uAlign, uCohere, uSeparate, uWander, uHome, uExpect;
uniform vec3 uCopy[6];
uniform int uN, uBurst;
out vec4 o;
void main() {
  ivec2 ij = ivec2(gl_FragCoord.xy);
  vec4 s = texelFetch(uBirds, ij, 0);
  vec2 pos = s.xy, v = s.zw;
  float rnd = hash12(vec2(ij) + fract(uTime * 0.13) * 1000.0);
  if (uBurst == 2 && uN > 0) {
    // Restart at one of the body's copies, flying outward.
    vec3 c = uCopy[min(int(rnd * float(uN)), uN - 1)];
    float a = hash12(vec2(ij) * 1.7 + uTime) * TAU;
    pos = fract(c.xy + vec2(cos(a) / uAspect, sin(a)) * c.z * hash12(vec2(a, rnd)));
    o = vec4(pos, vec2(cos(a), sin(a)) * uSpeed * 2.0);
    return;
  }
  vec4 g = texture(uGrid, pos);
  float rho = g.r / uExpect;
  vec2 avg = g.r > 1e-3 ? g.gb / g.r : v;
  vec2 dx = vec2(uGridTexel.x, 0.0), dy = vec2(0.0, uGridTexel.y);
  vec2 grad = vec2(texture(uGrid, pos + dx).r - texture(uGrid, pos - dx).r, texture(uGrid, pos + dy).r - texture(uGrid, pos - dy).r) / uExpect;
  float glen = length(grad);
  vec2 gdir = glen > 1e-4 ? grad / glen : vec2(0.0);
  vec2 acc = uAlign * (avg - v) * 3.0;
  // Toward denser air while thin, away from it once crowded.
  acc += gdir * uSpeed * 4.0 * (uCohere - uSeparate * clamp(rho - 1.0, 0.0, 3.0) * 0.6);
  float a = (hash12(vec2(ij) * 0.37 + floor(uTime * 4.0)) - 0.5) * TAU;
  acc += vec2(cos(a), sin(a)) * uWander * uSpeed * 3.0;
  if (uN > 0) {
    // Pull toward the nearest copy (wrapped), stronger far away.
    vec2 best = vec2(0.0);
    float bd = 1e9;
    for (int i = 0; i < 6; i++) {
      if (i >= uN) break;
      vec2 d = uCopy[i].xy - pos;
      d -= floor(d + 0.5);
      d.x *= uAspect;
      float l = length(d);
      if (l < bd) { bd = l; best = d; }
    }
    acc += best / max(bd, 1e-3) * uHome * uSpeed * 2.5 * smoothstep(0.02, 0.4, bd);
  }
  if (uBurst == 1) acc += normalize(v + vec2(1e-4)) * uSpeed * 40.0;
  v += acc * uDt;
  float sp = length(v);
  // Cruise between half and 1.6x the speed; a burst may reach 4x and eases back.
  float lo = uSpeed * 0.5, hi = uSpeed * 4.0;
  if (sp > hi) v *= hi / sp;
  if (sp > uSpeed * 1.6) v *= 1.0 - min(1.0, uDt * 2.0);
  else if (sp < lo) v = (sp > 1e-5 ? v / sp : vec2(cos(a), sin(a))) * lo;
  pos = fract(pos + vec2(v.x / uAspect, v.y) * uDt);
  o = vec4(pos, v);
}`;

const DRAW_VS = HEAD + /* glsl */ `
uniform sampler2D uBirds;
uniform int uW;
uniform float uSize, uBright, uSpeed;
uniform vec3 uColA, uColB, uColC;
out vec3 vCol;
void main() {
  vec4 s = texelFetch(uBirds, ivec2(gl_VertexID % uW, gl_VertexID / uW), 0);
  gl_Position = vec4(s.xy * 2.0 - 1.0, 0.0, 1.0);
  gl_PointSize = uSize;
  // Colour by heading (two palette slots), the third slot for the fastest birds.
  float h = atan(s.w, s.z) / 6.28318530718 + 0.5;
  vec3 c = mix(uColA, uColB, 0.5 + 0.5 * cos(h * 6.28318530718));
  c = mix(c, uColC * 1.3, smoothstep(1.1, 1.8, length(s.zw) / max(uSpeed, 1e-3)));
  vCol = c * uBright;
}`;
const DRAW_FS = HEAD + /* glsl */ `
in vec3 vCol;
out vec4 o;
void main() {
  vec2 d = gl_PointCoord * 2.0 - 1.0;
  o = vec4(vCol * exp(-dot(d, d) * 3.0), 1.0);
}`;

export interface FlockStep {
  dt: number;
  time: number;
  aspect: number;
  /** Birds to run (already scaled for the stage). */
  count: number;
  speed: number;
  radius: number;
  align: number;
  cohere: number;
  separate: number;
  wander: number;
  home: number;
  /** Home places (uv x, uv y, radius; up to 6). */
  copies: Float32Array;
  nCopies: number;
  /** A drop this frame: 0 none, 1 burst outward, 2 restart at the body. */
  burst: number;
}

export class Boids {
  private birds: PingPong | null = null;
  private grid: Target | null = null;
  private pSplat: Program;
  private pUpdate: Program;
  private pDraw: Program;
  private vao: WebGLVertexArrayObject;
  side = 0;
  count = 0;
  private gw = 0;
  private gh = 0;

  constructor(
    private gl: GL,
    private fs: Fullscreen,
    private gridFmt: TexFormat,
  ) {
    this.pSplat = new Program(gl, SPLAT_VS, SPLAT_FS, 'flock-splat');
    this.pUpdate = new Program(gl, FULL_VS, UPDATE_FS, 'flock-update');
    this.pDraw = new Program(gl, DRAW_VS, DRAW_FS, 'flock-draw');
    const v = gl.createVertexArray();
    if (!v) throw new Error('[flock] cannot create VAO');
    this.vao = v;
  }

  /** Scatters the flock again (a new genome takes the simulation over). */
  reseed(): void {
    this.birds?.dispose();
    this.birds = null;
    this.side = 0;
  }

  private setCount(n: number): void {
    const side = Math.max(32, Math.ceil(Math.sqrt(n)));
    if (side === this.side && this.birds) return;
    this.birds?.dispose();
    this.side = side;
    const total = side * side;
    const s0 = new Float32Array(total * 4);
    for (let i = 0; i < total; i++) {
      const a = Math.random() * Math.PI * 2;
      s0[i * 4] = Math.random();
      s0[i * 4 + 1] = Math.random();
      s0[i * 4 + 2] = Math.cos(a) * 0.1;
      s0[i * 4 + 3] = Math.sin(a) * 0.1;
    }
    this.birds = new PingPong(this.gl, side, side, [formats(this.gl).rgba32f], this.gl.NEAREST, [s0]);
  }

  step(u: FlockStep): void {
    const gl = this.gl;
    if (!this.birds || Math.abs(this.count - u.count) > u.count * 0.15) {
      this.setCount(u.count);
      this.count = u.count;
    }
    const gh = Math.max(4, Math.round(1 / u.radius));
    const gw = Math.max(4, Math.round(gh * u.aspect));
    if (gw !== this.gw || gh !== this.gh || !this.grid) {
      this.grid?.dispose();
      this.grid = new Target(gl, gw, gh, [this.gridFmt], gl.LINEAR);
      this.gw = gw;
      this.gh = gh;
    }
    const b = this.birds!;
    this.grid.bind();
    gl.clearColor(0, 0, 0, 0);
    gl.clear(gl.COLOR_BUFFER_BIT);
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.ONE, gl.ONE);
    this.pSplat.use().tex('uBirds', b.read.t).i1('uW', this.side);
    gl.bindVertexArray(this.vao);
    gl.drawArrays(gl.POINTS, 0, this.count);
    gl.disable(gl.BLEND);

    b.write.bind();
    this.pUpdate
      .use()
      .tex('uBirds', b.read.t)
      .tex('uGrid', this.grid.t)
      .f2('uGridTexel', 1 / gw, 1 / gh)
      .f1('uAspect', u.aspect)
      .f1('uDt', Math.min(0.05, u.dt))
      .f1('uTime', u.time)
      .f1('uSpeed', u.speed)
      .f1('uAlign', u.align)
      .f1('uCohere', u.cohere)
      .f1('uSeparate', u.separate)
      .f1('uWander', u.wander)
      .f1('uHome', u.home)
      .f1('uExpect', Math.max(1e-3, this.count / (gw * gh)))
      .i1('uN', Math.min(6, u.nCopies))
      .i1('uBurst', u.burst);
    if (u.nCopies > 0) gl.uniform3fv(this.pUpdate.loc('uCopy'), u.copies);
    this.fs.draw();
    b.swap();
  }

  /** Adds the birds into the bound framebuffer as soft points (additive). */
  draw(size: number, bright: number, speed: number, cols: Float32Array): void {
    if (!this.birds) return;
    const gl = this.gl;
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.ONE, gl.ONE);
    this.pDraw
      .use()
      .tex('uBirds', this.birds.read.t)
      .i1('uW', this.side)
      .f1('uSize', size)
      .f1('uBright', bright)
      .f1('uSpeed', speed)
      .f3('uColA', cols[0], cols[1], cols[2])
      .f3('uColB', cols[3], cols[4], cols[5])
      .f3('uColC', cols[6], cols[7], cols[8]);
    gl.bindVertexArray(this.vao);
    gl.drawArrays(gl.POINTS, 0, this.count);
  }

  dispose(): void {
    this.birds?.dispose();
    this.grid?.dispose();
    this.birds = null;
    this.grid = null;
    this.pSplat.dispose();
    this.pUpdate.dispose();
    this.pDraw.dispose();
    this.gl.deleteVertexArray(this.vao);
  }
}
