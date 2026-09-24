// GPU side of the water ripple carrier gene (see water.ts): the coarse wave simulation.

import { Fullscreen, GL, PingPong, Program, formats } from '../../render/gl';
import { FULLSCREEN_VS } from '../../render/shaders';
import { MAX_DROPS } from './water';

const STEP_FS = /* glsl */ `#version 300 es
precision highp float;
in vec2 vUv, vL, vR, vT, vB;
uniform sampler2D uH;
uniform float uDamp, uAspect;
uniform vec4 uDrop[${MAX_DROPS}];
out vec4 o;
void main() {
  vec2 h = texture(uH, vUv).rg;
  float s = texture(uH, vL).r + texture(uH, vR).r + texture(uH, vT).r + texture(uH, vB).r;
  // The two-buffer water recurrence, with a slight leak so no flat plateau builds up.
  float n = (s * 0.5 - h.g) * uDamp - 0.006 * h.r;
  for (int i = 0; i < ${MAX_DROPS}; i++) {
    vec4 d = uDrop[i];
    if (d.w == 0.0) continue;
    vec2 q = (vUv - d.xy) * vec2(uAspect, 1.0);
    // Zero-mean drop (a crest in a trough) so drops add rings, not a raised floor.
    float rr = dot(q, q) / (d.z * d.z);
    n += d.w * (exp(-rr) - 0.25 * exp(-rr * 0.25));
  }
  o = vec4(clamp(n, -4.0, 4.0), h.r, 0.0, 1.0);
}`;

/** The coarse wave simulation (one per Stage, owned by the heaviest slot with water). */
export class Water {
  private hf: PingPong | null = null;
  private readonly prog: Program;
  private readonly drops = new Float32Array(MAX_DROPS * 4);
  private n = 0;
  w = 0;
  h = 0;

  private readonly gl: GL;
  private readonly fs: Fullscreen;

  constructor(gl: GL, fs: Fullscreen) {
    this.gl = gl;
    this.fs = fs;
    this.prog = new Program(gl, FULLSCREEN_VS, STEP_FS, 'v2-water');
  }

  get tex(): WebGLTexture | null {
    return this.hf ? this.hf.read.t : null;
  }

  resize(aspect: number): void {
    const h = 200;
    const w = Math.max(64, Math.round(h * aspect));
    if (w === this.w && h === this.h) return;
    this.hf?.dispose();
    this.w = w;
    this.h = h;
    this.hf = new PingPong(this.gl, w, h, [formats(this.gl).rg16f], this.gl.LINEAR);
  }

  /** Queues a drop at (x, y) in uv, radius in screen heights, strength (negative: a dent). */
  drop(x: number, y: number, r: number, s: number): void {
    if (this.n >= MAX_DROPS) return;
    this.drops.set([x, y, r, s], this.n++ * 4);
  }

  step(damp: number, aspect: number): void {
    const hf = this.hf;
    if (!hf) return;
    hf.write.bind();
    this.gl.disable(this.gl.BLEND);
    this.prog.use().f2('uTexel', 1 / this.w, 1 / this.h).tex('uH', hf.read.t).f1('uDamp', damp).f1('uAspect', aspect).f4v('uDrop', this.drops);
    this.fs.draw();
    hf.swap();
    this.drops.fill(0);
    this.n = 0;
  }

  dispose(): void {
    this.hf?.dispose();
    this.hf = null;
  }
}

