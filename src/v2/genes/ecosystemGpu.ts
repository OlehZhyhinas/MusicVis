// Stem ecosystem on the GPU (WebGL2). Agents of four species live in one RGBA32F ping-pong texture
// (x, y in stage uv, heading, rank; rank + 2 marks an agent that was dead last frame); the species is
// fixed by the agent's index range (the roster). A shared RGBA16F world field at a third of the stage:
//   r = growth / flora (pollinators bloom it, grazers eat it), g = grazer trail, b = predator scent,
//   a = plankton density.
// Each frame:
//   1. update: births (a newborn copies a live parent of its species), plankton eaten inside dense
//      predator scent (reborn at another plankton), then each species senses the field and moves;
//   2. deposit: every live agent adds its species' marks into the field as a one-pixel point;
//   3. diffuse + decay: the flora spreads slowly and keeps (the gene's decay), the scents blur and fade.
// draw() lays the flora into the bound framebuffer (the slot's feedback) and the agents as sprites.

import { Fullscreen, GL, PingPong, Program, canRenderTo, formats, type TexFormat } from '../../render/gl';
import { ECO_FIELD_MAX_H, ECO_GLYPHS, ECO_SIZE, ecoAgents, ecoInitial, ecoStep, type EcoState } from './ecosystem';

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
const COMMON = /* glsl */ `
const float TAU = 6.28318530718;
float hash12(vec2 p) {
  vec3 p3 = fract(vec3(p.xyx) * 0.1031);
  p3 += dot(p3, p3.yzx + 33.33);
  return fract((p3.x + p3.y) * p3.z);
}
uniform vec4 uCut;
uniform int uW, uN;
int speciesOf(int idx) {
  float u = (float(idx) + 0.5) / float(uN);
  return u < uCut.x ? 0 : u < uCut.y ? 1 : u < uCut.z ? 2 : 3;
}`;

const UPDATE_FS = HEAD + COMMON + /* glsl */ `
uniform sampler2D uAgents, uField;
uniform vec4 uPop, uEnv;
uniform float uAspect, uTime, uSpeed, uStrike, uPred, uF60;
out vec4 o;
vec2 dirOf(float a) { return vec2(cos(a) / uAspect, sin(a)); }
float ch(vec4 f, int c) { return c == 0 ? f.r : c == 1 ? f.g : c == 2 ? f.b : f.a; }
float sense(vec2 p, float a, float d, int c) { return ch(texture(uField, fract(p + dirOf(a) * d)), c); }
// Physarum-style steering: straight on, toward the stronger side, or a random pick when both beat ahead.
float steer(float f, float l, float r, float rnd) {
  if (f >= l && f >= r) return (rnd - 0.5) * 0.2;
  if (f < l && f < r) return rnd < 0.5 ? -1.0 : 1.0;
  return (l > r ? 1.0 : -1.0) * (0.6 + 0.4 * rnd);
}
float vnoise(vec2 p) {
  vec2 i = floor(p), f = fract(p);
  f = f * f * (3.0 - 2.0 * f);
  return mix(mix(hash12(i), hash12(i + vec2(1, 0)), f.x), mix(hash12(i + vec2(0, 1)), hash12(i + vec2(1, 1)), f.x), f.y);
}
float adiff(float a, float b) { return mod(a - b + 3.0 * TAU * 0.5, TAU) - TAU * 0.5; }
void main() {
  ivec2 ij = ivec2(gl_FragCoord.xy);
  int idx = ij.y * uW + ij.x;
  vec4 s = texelFetch(uAgents, ij, 0);
  if (idx >= uN) { o = s; return; }
  int sp = speciesOf(idx);
  bool was = s.w < 2.0;
  float rank = was ? s.w : s.w - 2.0;
  float pop = uPop[sp];
  bool alive = rank < pop;
  float rnd = hash12(vec2(ij) + vec2(rank * 911.7, fract(uTime * 0.37) * 1733.1));
  float rnd2 = hash12(vec2(ij).yx * 1.37 + vec2(fract(uTime * 0.71) * 417.3, rank * 57.1));
  vec2 pos = s.xy;
  float h = s.z;
  // Plankton inside dense predator scent get eaten (and are reborn beside another plankton).
  bool eaten = sp == 3 && alive && was && rnd2 < uPred * 0.06 * uF60 * smoothstep(0.4, 3.0, texture(uField, pos).b);
  if (alive && (!was || eaten)) {
    float lo = sp == 0 ? 0.0 : uCut[sp - 1];
    int a0 = int(lo * float(uN));
    int a1 = max(a0 + 1, int(uCut[sp] * float(uN)));
    vec2 np = vec2(hash12(vec2(rnd, rnd2) * 97.1 + 3.1), hash12(vec2(rnd2, rnd) * 53.7 + 7.7));
    for (int k = 0; k < 4; k++) {
      float r = hash12(vec2(float(k) * 13.1 + rnd * 71.3, rnd2 * 33.9 + float(k)));
      int j = min(a1 - 1, a0 + int(r * float(a1 - a0)));
      vec4 q = texelFetch(uAgents, ivec2(j % uW, j / uW), 0);
      if (j != idx && q.w < pop) {
        np = q.xy + (vec2(rnd, rnd2) - 0.5) * vec2(0.02 / uAspect, 0.02);
        break;
      }
    }
    pos = fract(np);
    h = rnd * TAU;
  }
  if (alive) {
    float tf = min(1.5, uF60);
    float spd;
    if (sp == 0) {
      // Predators: chase the plankton, darting on drum hits.
      float d = 0.035;
      h += steer(sense(pos, h, d, 3), sense(pos, h + 0.6, d, 3), sense(pos, h - 0.6, d, 3), rnd) * 0.35 * tf;
      spd = 0.0035 * (0.4 + 0.6 * uEnv.x) * (1.0 + 5.0 * uStrike);
    } else if (sp == 1) {
      // Grazers: lumber toward the richest flora.
      float d = 0.05;
      h += steer(sense(pos, h, d, 0), sense(pos, h + 0.5, d, 0), sense(pos, h - 0.5, d, 0), rnd) * 0.1 * tf + (rnd2 - 0.5) * 0.08;
      spd = 0.0009 * (0.5 + uEnv.y);
    } else if (sp == 2) {
      // Pollinators: follow the grazer trails, fluttering.
      float d = 0.03;
      h += steer(sense(pos, h, d, 1), sense(pos, h + 0.7, d, 1), sense(pos, h - 0.7, d, 1), rnd) * 0.3 * tf;
      h += sin(uTime * 4.0 + rank * 60.0) * 0.12 * tf;
      spd = 0.0022 * (0.4 + 0.8 * uEnv.z);
    } else {
      // Plankton: drift on a slow current, turning away from predator scent.
      float flow = TAU * 2.0 * vnoise(pos * vec2(uAspect, 1.0) * 2.5 + uTime * 0.03);
      h += adiff(flow, h) * 0.05 * tf;
      float d = 0.04;
      float l = sense(pos, h + 0.8, d, 2), r = sense(pos, h - 0.8, d, 2);
      h += clamp((r - l) * 1.5, -0.6, 0.6) * tf;
      spd = 0.0012 * (0.5 + 0.8 * uEnv.w);
    }
    pos = fract(pos + dirOf(h) * spd * uSpeed * uF60);
  }
  o = vec4(pos, mod(h, TAU), alive ? rank : rank + 2.0);
}`;

const DEPOSIT_VS = HEAD + COMMON + /* glsl */ `
uniform sampler2D uAgents;
uniform float uDep, uBloom, uGraze;
out vec4 vDep;
void main() {
  vec4 s = texelFetch(uAgents, ivec2(gl_VertexID % uW, gl_VertexID / uW), 0);
  int sp = speciesOf(gl_VertexID);
  vDep = uDep * (sp == 0 ? vec4(0, 0, 1, 0) : sp == 1 ? vec4(-0.6 * uGraze, 1, 0, 0) : sp == 2 ? vec4(uBloom, 0, 0, 0) : vec4(0, 0, 0, 1));
  gl_Position = s.w < 2.0 ? vec4(s.xy * 2.0 - 1.0, 0.0, 1.0) : vec4(2.0, 2.0, 2.0, 1.0);
  gl_PointSize = 1.0;
}`;
const DEPOSIT_FS = HEAD + /* glsl */ `
in vec4 vDep;
out vec4 o;
void main() { o = vDep; }`;

const DIFFUSE_FS = HEAD + /* glsl */ `
uniform sampler2D uField;
uniform vec2 uTexel;
uniform float uDecayR, uDecayS, uMix;
in vec2 vUv;
out vec4 o;
void main() {
  vec4 c = texture(uField, vUv);
  vec4 b = vec4(0.0);
  for (int y = -1; y <= 1; y++)
    for (int x = -1; x <= 1; x++) b += texture(uField, fract(vUv + vec2(x, y) * uTexel));
  vec4 m = mix(c, b / 9.0, vec4(0.15, 0.5, 0.5, 0.5) * uMix);
  o = clamp(m * vec4(uDecayR, uDecayS, uDecayS, uDecayS), 0.0, 8.0);
}`;

const FLORA_FS = HEAD + /* glsl */ `
uniform sampler2D uField;
uniform vec3 uFlora, uTrailCol;
uniform float uGain, uScale;
in vec2 vUv;
out vec4 o;
void main() {
  vec4 f = texture(uField, vUv);
  float k = 1.0 - exp(-f.r * uScale);
  vec3 c = mix(uFlora * 0.5, uFlora * 1.2 + 0.25, smoothstep(0.45, 1.0, k)) * k;
  c += uTrailCol * (1.0 - exp(-f.g * 0.15)) * 0.35;
  o = vec4(c * uGain, 1.0);
}`;

const SPRITE_VS = HEAD + COMMON + /* glsl */ `
uniform sampler2D uAgents;
uniform vec3 uC0, uC1, uC2, uC3;
uniform vec4 uGlyphs, uSizes;
uniform float uSize, uTrail, uBright, uStrike;
out vec3 vCol;
out vec2 vDir;
flat out int vGlyph;
out float vR;
void main() {
  vec4 s = texelFetch(uAgents, ivec2(gl_VertexID % uW, gl_VertexID / uW), 0);
  int sp = speciesOf(gl_VertexID);
  int g = int(uGlyphs[sp] + 0.5);
  float base = max(1.0, uSize * uSizes[sp]);
  float grow = g == 1 ? 1.0 + 3.0 * uTrail : 1.0;
  vGlyph = g;
  vR = 1.0 / grow;
  vDir = vec2(cos(s.z), sin(s.z));
  vec3 c = sp == 0 ? uC0 : sp == 1 ? uC1 : sp == 2 ? uC2 : uC3;
  vCol = c * uBright * (sp == 0 ? 1.0 + 2.0 * uStrike : 1.0) * (g == 2 ? 0.6 : 1.0);
  gl_Position = s.w < 2.0 && gl_VertexID < uN ? vec4(s.xy * 2.0 - 1.0, 0.0, 1.0) : vec4(2.0, 2.0, 2.0, 1.0);
  gl_PointSize = clamp(base * grow, 1.0, 64.0);
}`;
const SPRITE_FS = HEAD + /* glsl */ `
in vec3 vCol;
in vec2 vDir;
flat in int vGlyph;
in float vR;
out vec4 o;
void main() {
  vec2 p = gl_PointCoord * 2.0 - 1.0;
  p.y = -p.y;
  float r2 = dot(p, p);
  float a;
  if (vGlyph == 0) a = exp(-r2 * 5.0);
  else if (vGlyph == 1) {
    // Streak: a bright head trailing back along the heading.
    vec2 A = vDir * (1.0 - vR);
    float t = clamp(dot(p + A, A) / max(1e-4, dot(2.0 * A, A)), 0.0, 1.0);
    float d = length(p + A - 2.0 * A * t);
    a = exp(-d * d / (vR * vR) * 3.0) * mix(0.15, 1.0, t * t);
  }
  else if (vGlyph == 2) a = exp(-r2 * 2.5);
  else if (vGlyph == 3) { float d = (sqrt(r2) - 0.62) / 0.16; a = exp(-d * d); }
  else if (vGlyph == 4) a = smoothstep(1.0, 0.65, abs(p.x) + abs(p.y)) * (0.6 + 0.4 * smoothstep(0.7, 0.2, abs(p.x) + abs(p.y)));
  else a = exp(-min(abs(p.x), abs(p.y)) * 14.0) * max(0.0, 1.0 - sqrt(r2)) + exp(-r2 * 10.0);
  if (a < 0.004) discard;
  o = vec4(vCol * a, 1.0);
}`;

export interface EcoStepIn {
  dt: number;
  time: number;
  aspect: number;
  /** Gene count (agents at the reference stage area). */
  count: number;
  cuts: [number, number, number, number];
  /** Presence of each stem (0..1) and its envelope; the drum strike; the drop envelope. */
  pres: ArrayLike<number>;
  env: ArrayLike<number>;
  strike: number;
  drop: number;
  /** The gene's parameters (with reactions applied). */
  p: Record<string, number>;
}

export interface EcoDrawIn {
  gain: number;
  fieldScale: number;
  field: number;
  size: number;
  trail: number;
  glyph: number;
  hues: number;
  bright: number;
  strike: number;
  cols: Float32Array;
}

export class Ecosystem {
  private agents: PingPong | null = null;
  private field: PingPong | null = null;
  private pUpdate: Program;
  private pDeposit: Program;
  private pDiffuse: Program;
  private pFlora: Program;
  private pSprite: Program;
  private vao: WebGLVertexArrayObject;
  private fieldFmt: TexFormat;
  private cuts: [number, number, number, number] = [0.25, 0.5, 0.75, 1];
  readonly state: EcoState = ecoInitial();
  side = 0;
  count = 0;
  fw = 0;
  fh = 0;
  sw = 1;
  sh = 1;

  constructor(
    private gl: GL,
    private fs: Fullscreen,
    hdr: TexFormat,
  ) {
    this.pUpdate = new Program(gl, FULL_VS, UPDATE_FS, 'eco-update');
    this.pDeposit = new Program(gl, DEPOSIT_VS, DEPOSIT_FS, 'eco-deposit');
    this.pDiffuse = new Program(gl, FULL_VS, DIFFUSE_FS, 'eco-diffuse');
    this.pFlora = new Program(gl, FULL_VS, FLORA_FS, 'eco-flora');
    this.pSprite = new Program(gl, SPRITE_VS, SPRITE_FS, 'eco-sprite');
    const v = gl.createVertexArray();
    if (!v) throw new Error('[eco] cannot create VAO');
    this.vao = v;
    const f16 = formats(gl).rgba16f;
    this.fieldFmt = canRenderTo(gl, f16) ? f16 : hdr;
  }

  /** World field size for a stage (a third of its rows, at most ECO_FIELD_MAX_H). */
  resize(stageW: number, stageH: number): void {
    this.sw = Math.max(1, stageW);
    this.sh = Math.max(1, stageH);
    const fh = Math.max(16, Math.min(ECO_FIELD_MAX_H, Math.round(stageH / 3)));
    const fw = Math.max(16, Math.round((fh * stageW) / Math.max(1, stageH)));
    if (fw === this.fw && fh === this.fh && this.field) return;
    this.field?.dispose();
    this.fw = fw;
    this.fh = fh;
    this.field = new PingPong(this.gl, fw, fh, [this.fieldFmt], this.gl.LINEAR);
  }

  /** Scatters the agents again, clears the world and restarts the populations (a new genome takes over). */
  reseed(): void {
    this.agents?.dispose();
    this.agents = null;
    this.side = 0;
    Object.assign(this.state, ecoInitial());
    for (const t of this.field ? [this.field.read, this.field.write] : []) {
      t.bind();
      this.gl.clearColor(0, 0, 0, 0);
      this.gl.clear(this.gl.COLOR_BUFFER_BIT);
    }
  }

  private setCount(n: number): void {
    const side = Math.max(32, Math.ceil(Math.sqrt(n)));
    if (side === this.side && this.agents) return;
    this.agents?.dispose();
    this.side = side;
    const total = side * side;
    const s0 = new Float32Array(total * 4);
    for (let i = 0; i < total; i++) {
      s0[i * 4] = Math.random();
      s0[i * 4 + 1] = Math.random();
      s0[i * 4 + 2] = Math.random() * Math.PI * 2;
      // Every agent starts unborn (rank + 2): the first frame gives birth to the starting colonies.
      s0[i * 4 + 3] = 2 + 0.001 + 0.998 * Math.random();
    }
    this.agents = new PingPong(this.gl, side, side, [formats(this.gl).rgba32f], this.gl.NEAREST, [s0]);
  }

  private common(p: Program): Program {
    return p.f4('uCut', this.cuts[0], this.cuts[1], this.cuts[2], this.cuts[3]).i1('uW', this.side).i1('uN', this.count);
  }

  step(u: EcoStepIn): void {
    if (!this.field) return;
    const n = Math.min(ecoAgents(u.count, this.sw * this.sh), 262144);
    if (!this.agents || Math.abs(this.count - n) > n * 0.15) {
      this.setCount(n);
      this.count = n;
    }
    this.cuts = u.cuts;
    ecoStep(this.state, u.pres, u.drop, u.p, u.dt);
    const pop = this.state.pop;
    const gl = this.gl;
    const ag = this.agents!;
    const fd = this.field;
    const f60 = Math.min(3, u.dt * 60);
    gl.disable(gl.BLEND);
    ag.write.bind();
    this.common(this.pUpdate.use())
      .tex('uAgents', ag.read.t)
      .tex('uField', fd.read.t)
      .f4('uPop', pop[0], pop[1], pop[2], pop[3])
      .f4('uEnv', u.env[0], u.env[1], u.env[2], u.env[3])
      .f1('uAspect', u.aspect)
      .f1('uTime', u.time)
      .f1('uSpeed', u.p.speed)
      .f1('uStrike', Math.min(1, u.strike))
      .f1('uPred', u.p.predation)
      .f1('uF60', f60);
    this.fs.draw();
    ag.swap();

    // Deposit: one mark per live agent, normalised by the agents per field pixel.
    fd.read.bind();
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.ONE, gl.ONE);
    const dep = (0.35 * (this.fw * this.fh)) / Math.max(1, this.count) * f60;
    this.common(this.pDeposit.use()).tex('uAgents', ag.read.t).f1('uDep', dep).f1('uBloom', u.p.bloom).f1('uGraze', u.p.graze);
    gl.bindVertexArray(this.vao);
    gl.drawArrays(gl.POINTS, 0, this.count);
    gl.disable(gl.BLEND);

    fd.write.bind();
    this.pDiffuse
      .use()
      .tex('uField', fd.read.t)
      .f2('uTexel', 1 / this.fw, 1 / this.fh)
      .f1('uDecayR', Math.pow(u.p.decay, f60))
      .f1('uDecayS', Math.pow(0.9, f60))
      .f1('uMix', Math.min(1, f60));
    this.fs.draw();
    fd.swap();
  }

  /** The four species' colours (predators, grazers, pollinators, plankton) from the palette, rotated by hues. */
  static speciesColours(cols: Float32Array, hues: number): number[][] {
    const A = [cols[0], cols[1], cols[2]];
    const B = [cols[3], cols[4], cols[5]];
    const C = [cols[6], cols[7], cols[8]];
    const D = A.map((x, i) => 0.3 * (x + B[i]) + 0.35);
    const base = [C, A, B, D];
    const r = Math.round(hues) & 3;
    return [0, 1, 2, 3].map((i) => base[(i + r) & 3]);
  }

  /**
   * Lays the flora into the bound framebuffer (MAX blending, like the physarum trail: it persists by
   * itself) and then adds the agents as sprites in their species' colours.
   */
  draw(d: EcoDrawIn): void {
    if (!this.field || !this.agents) return;
    const gl = this.gl;
    const sc = Ecosystem.speciesColours(d.cols, d.hues);
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.ONE, gl.ONE);
    if (d.field > 0.001) {
      gl.blendEquation(gl.MAX);
      this.pFlora
        .use()
        .tex('uField', this.field.read.t)
        .f1('uGain', d.gain * d.field * 1.5)
        .f1('uScale', d.fieldScale)
        .f3('uFlora', sc[2][0], sc[2][1], sc[2][2])
        .f3('uTrailCol', sc[1][0], sc[1][1], sc[1][2]);
      this.fs.draw();
      gl.blendEquation(gl.FUNC_ADD);
    }
    const gl4 = ECO_GLYPHS[Math.round(d.glyph) & 3];
    const px = this.sh / 1080;
    this.common(this.pSprite.use())
      .tex('uAgents', this.agents.read.t)
      .f3('uC0', sc[0][0], sc[0][1], sc[0][2])
      .f3('uC1', sc[1][0], sc[1][1], sc[1][2])
      .f3('uC2', sc[2][0], sc[2][1], sc[2][2])
      .f3('uC3', sc[3][0], sc[3][1], sc[3][2])
      .f4('uGlyphs', gl4[0], gl4[1], gl4[2], gl4[3])
      .f4('uSizes', ECO_SIZE[0], ECO_SIZE[1], ECO_SIZE[2], ECO_SIZE[3])
      .f1('uSize', d.size * px)
      .f1('uTrail', d.trail)
      .f1('uBright', d.bright)
      .f1('uStrike', Math.min(1, d.strike));
    gl.bindVertexArray(this.vao);
    gl.drawArrays(gl.POINTS, 0, this.count);
  }

  dispose(): void {
    this.agents?.dispose();
    this.field?.dispose();
    this.agents = this.field = null;
    this.pUpdate.dispose();
    this.pDeposit.dispose();
    this.pDiffuse.dispose();
    this.pFlora.dispose();
    this.pSprite.dispose();
    this.gl.deleteVertexArray(this.vao);
  }
}
