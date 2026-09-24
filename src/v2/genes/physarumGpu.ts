// Physarum on the GPU (WebGL2). Agents live in an RGBA32F ping-pong texture (x, y in trail uv, heading,
// a per-agent random seed); the trail map is a ping-pong at up to 1280x720. Each frame:
//   1. sense / turn / move: a fragment pass over the agent texture reads the trail at three sensors;
//   2. deposit: every agent is drawn as a one-pixel point, added into the trail;
//   3. diffuse + decay: a 3x3 blur mixed in, then the fade.
// draw() lays the trail into the bound framebuffer (the slot's feedback) in the palette colours.

import { Fullscreen, GL, PingPong, Program, canRenderTo, formats, type TexFormat } from '../../render/gl';
import { SLIME_MAX_H, slimeAgents } from './physarum';

const HEAD = /* glsl */ `#version 300 es
precision highp float;
precision highp int;
precision highp sampler2D;
`;
const FULL_VS = HEAD + /* glsl */ `
out vec2 vUv;
void main() {
  vec2 p = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2));
  vUv = p;
  gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}`;
const HASH = /* glsl */ `
const float TAU = 6.28318530718;
float hash12(vec2 p) {
  vec3 p3 = fract(vec3(p.xyx) * 0.1031);
  p3 += dot(p3, p3.yzx + 33.33);
  return fract((p3.x + p3.y) * p3.z);
}`;

const UPDATE_FS = HEAD + HASH + /* glsl */ `
uniform sampler2D uAgents, uTrail;
uniform float uAspect, uSA, uSD, uTurn, uStep, uTime;
out vec4 o;
float sense(vec2 pos, float a) {
  return texture(uTrail, fract(pos + vec2(cos(a) / uAspect, sin(a)) * uSD)).r;
}
void main() {
  ivec2 ij = ivec2(gl_FragCoord.xy);
  vec4 s = texelFetch(uAgents, ij, 0);
  vec2 pos = s.xy;
  float h = s.z;
  float f = sense(pos, h), l = sense(pos, h + uSA), r = sense(pos, h - uSA);
  float rnd = hash12(vec2(ij) + vec2(s.w * 911.7, fract(uTime * 0.37) * 1733.1));
  if (f >= l && f >= r) {
    // Straight on (a little jitter keeps the veins from freezing).
    h += (rnd - 0.5) * uTurn * 0.2;
  } else if (f < l && f < r) {
    h += (rnd < 0.5 ? -1.0 : 1.0) * uTurn;
  } else {
    h += (l > r ? 1.0 : -1.0) * uTurn * (0.6 + 0.4 * rnd);
  }
  pos = fract(pos + vec2(cos(h) / uAspect, sin(h)) * uStep);
  o = vec4(pos, mod(h, TAU), s.w);
}`;

const DEPOSIT_VS = HEAD + /* glsl */ `
uniform sampler2D uAgents;
uniform int uW;
void main() {
  vec4 s = texelFetch(uAgents, ivec2(gl_VertexID % uW, gl_VertexID / uW), 0);
  gl_Position = vec4(s.xy * 2.0 - 1.0, 0.0, 1.0);
  gl_PointSize = 1.0;
}`;
const DEPOSIT_FS = HEAD + /* glsl */ `
uniform float uDep;
out vec4 o;
void main() { o = vec4(uDep, 0.0, 0.0, 1.0); }`;

const DIFFUSE_FS = HEAD + /* glsl */ `
uniform sampler2D uTrail;
uniform vec2 uTexel;
uniform float uDecay, uDiffuse;
in vec2 vUv;
out vec4 o;
void main() {
  float c = texture(uTrail, vUv).r;
  float b = 0.0;
  for (int y = -1; y <= 1; y++)
    for (int x = -1; x <= 1; x++) b += texture(uTrail, fract(vUv + vec2(x, y) * uTexel)).r;
  o = vec4(mix(c, b / 9.0, uDiffuse) * uDecay, 0.0, 0.0, 1.0);
}`;

const SHOW_FS = HEAD + /* glsl */ `
uniform sampler2D uTrail;
uniform vec3 uColA, uColB, uColC;
uniform float uGain, uScale;
in vec2 vUv;
out vec4 o;
void main() {
  float t = texture(uTrail, vUv).r * uScale;
  float k = 1.0 - exp(-t);
  vec3 c = mix(uColA, uColB, smoothstep(0.0, 0.55, k));
  c = mix(c, uColC * 1.4, smoothstep(0.6, 1.0, k));
  o = vec4(c * k * uGain, 1.0);
}`;

export interface SlimeStep {
  dt: number;
  time: number;
  aspect: number;
  /** Gene count (agents at the reference trail area). */
  count: number;
  sa: number;
  sd: number;
  turn: number;
  step: number;
  deposit: number;
  decay: number;
  diffuse: number;
}

export class Physarum {
  private agents: PingPong | null = null;
  private trail: PingPong | null = null;
  private pUpdate: Program;
  private pDeposit: Program;
  private pDiffuse: Program;
  private pShow: Program;
  private vao: WebGLVertexArrayObject;
  private trailFmt: TexFormat;
  side = 0;
  count = 0;
  tw = 0;
  th = 0;

  constructor(
    private gl: GL,
    private fs: Fullscreen,
    hdr: TexFormat,
  ) {
    this.pUpdate = new Program(gl, FULL_VS, UPDATE_FS, 'slime-update');
    this.pDeposit = new Program(gl, DEPOSIT_VS, DEPOSIT_FS, 'slime-deposit');
    this.pDiffuse = new Program(gl, FULL_VS, DIFFUSE_FS, 'slime-diffuse');
    this.pShow = new Program(gl, FULL_VS, SHOW_FS, 'slime-show');
    const v = gl.createVertexArray();
    if (!v) throw new Error('[slime] cannot create VAO');
    this.vao = v;
    const r16 = formats(gl).r16f;
    this.trailFmt = canRenderTo(gl, r16) ? r16 : hdr;
  }

  /** Trail map size for a stage (at most SLIME_MAX_H rows, the stage's own size when smaller). */
  resize(stageW: number, stageH: number): void {
    const th = Math.max(16, Math.min(SLIME_MAX_H, Math.round(stageH)));
    const tw = Math.max(16, Math.round((th * stageW) / Math.max(1, stageH)));
    if (tw === this.tw && th === this.th && this.trail) return;
    this.trail?.dispose();
    this.tw = tw;
    this.th = th;
    this.trail = new PingPong(this.gl, tw, th, [this.trailFmt], this.gl.LINEAR);
  }

  private setCount(n: number): void {
    const side = Math.max(64, Math.ceil(Math.sqrt(n)));
    if (side === this.side && this.agents) return;
    this.agents?.dispose();
    this.side = side;
    const total = side * side;
    const s0 = new Float32Array(total * 4);
    for (let i = 0; i < total; i++) {
      // Scattered over the whole screen with random headings: the network condenses out of the haze.
      s0[i * 4] = Math.random();
      s0[i * 4 + 1] = Math.random();
      s0[i * 4 + 2] = Math.random() * Math.PI * 2;
      s0[i * 4 + 3] = Math.random();
    }
    this.agents = new PingPong(this.gl, side, side, [formats(this.gl).rgba32f], this.gl.NEAREST, [s0]);
  }

  step(u: SlimeStep): void {
    if (!this.trail) return;
    const n = slimeAgents(u.count, this.tw * this.th);
    if (!this.agents || Math.abs(this.count - n) > n * 0.15) {
      this.setCount(n);
      this.count = n;
    }
    const gl = this.gl;
    const ag = this.agents!;
    const tr = this.trail;
    const f60 = Math.min(3, u.dt * 60);
    // Scene units -> trail pixels: the step never exceeds a couple of pixels per frame.
    gl.disable(gl.BLEND);
    ag.write.bind();
    this.pUpdate
      .use()
      .tex('uAgents', ag.read.t)
      .tex('uTrail', tr.read.t)
      .f1('uAspect', u.aspect)
      .f1('uSA', u.sa)
      .f1('uSD', u.sd)
      .f1('uTurn', Math.min(1.5, u.turn * f60))
      .f1('uStep', u.step * f60)
      .f1('uTime', u.time);
    this.fs.draw();
    ag.swap();

    tr.read.bind();
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.ONE, gl.ONE);
    this.pDeposit.use().tex('uAgents', ag.read.t).i1('uW', this.side).f1('uDep', u.deposit * f60);
    gl.bindVertexArray(this.vao);
    gl.drawArrays(gl.POINTS, 0, this.count);
    gl.disable(gl.BLEND);

    tr.write.bind();
    this.pDiffuse
      .use()
      .tex('uTrail', tr.read.t)
      .f2('uTexel', 1 / this.tw, 1 / this.th)
      .f1('uDecay', Math.pow(u.decay, f60))
      .f1('uDiffuse', Math.min(1, u.diffuse * f60));
    this.fs.draw();
    tr.swap();
  }

  /**
   * Lays the trail into the bound framebuffer (the slot's feedback), coloured by the palette. MAX blending:
   * the feedback holds at least the network's own light every frame (the trail already persists, so adding
   * it would pile up), and the carrier smears the older copies it moves away.
   */
  draw(gain: number, scale: number, cols: Float32Array): void {
    if (!this.trail) return;
    const gl = this.gl;
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.ONE, gl.ONE);
    gl.blendEquation(gl.MAX);
    this.pShow
      .use()
      .tex('uTrail', this.trail.read.t)
      .f1('uGain', gain)
      .f1('uScale', scale)
      .f3('uColA', cols[0], cols[1], cols[2])
      .f3('uColB', cols[3], cols[4], cols[5])
      .f3('uColC', cols[6], cols[7], cols[8]);
    this.fs.draw();
    gl.blendEquation(gl.FUNC_ADD);
  }

  dispose(): void {
    this.agents?.dispose();
    this.trail?.dispose();
    this.agents = this.trail = null;
    this.pUpdate.dispose();
    this.pDeposit.dispose();
    this.pDiffuse.dispose();
    this.pShow.dispose();
    this.gl.deleteVertexArray(this.vao);
  }
}
