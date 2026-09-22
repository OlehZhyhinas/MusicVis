// Stable Fluids velocity solver on a coarse grid. There is no dye field: the
// feedback buffers are the "ink", displaced by this velocity each frame.

import { Fullscreen, GL, PingPong, Program, Target, formats } from './gl';
import {
  ADVECT_FS, CURL_FS, DIVERGENCE_FS, FULLSCREEN_VS, GRADIENT_FS, PRESSURE_FS, SCALE_FS, SPLAT_FS, VORTICITY_FS,
} from './shaders';

export const MAX_SPLATS = 16;
export const SPLAT_DIR = 0;
export const SPLAT_RADIAL = 1;
export const SPLAT_SWIRL = 2;

export class Fluid {
  private velocity!: PingPong;
  private pressure!: PingPong;
  private divergence!: Target;
  private curl!: Target;
  private pSplat: Program;
  private pCurl: Program;
  private pVort: Program;
  private pDiv: Program;
  private pScale: Program;
  private pPressure: Program;
  private pGrad: Program;
  private pAdvect: Program;
  w = 0;
  h = 0;

  private splatA = new Float32Array(MAX_SPLATS * 4);
  private splatB = new Float32Array(MAX_SPLATS * 4);
  private splatCount = 0;

  iterations = 20;
  pressureDecay = 0.8;
  dissipation = 0.25;

  constructor(
    private gl: GL,
    private fs: Fullscreen,
  ) {
    this.pSplat = new Program(gl, FULLSCREEN_VS, SPLAT_FS, 'splat');
    this.pCurl = new Program(gl, FULLSCREEN_VS, CURL_FS, 'curl');
    this.pVort = new Program(gl, FULLSCREEN_VS, VORTICITY_FS, 'vorticity');
    this.pDiv = new Program(gl, FULLSCREEN_VS, DIVERGENCE_FS, 'divergence');
    this.pScale = new Program(gl, FULLSCREEN_VS, SCALE_FS, 'scale');
    this.pPressure = new Program(gl, FULLSCREEN_VS, PRESSURE_FS, 'pressure');
    this.pGrad = new Program(gl, FULLSCREEN_VS, GRADIENT_FS, 'gradient');
    this.pAdvect = new Program(gl, FULLSCREEN_VS, ADVECT_FS, 'advect');
  }

  get velocityTex(): WebGLTexture {
    return this.velocity.read.t;
  }
  get texelX(): number {
    return 1 / this.w;
  }
  get texelY(): number {
    return 1 / this.h;
  }

  resize(aspect: number): void {
    const h = 180;
    const w = Math.max(64, Math.round(h * aspect));
    if (w === this.w && h === this.h) return;
    this.disposeTargets();
    const gl = this.gl;
    const f = formats(gl);
    this.w = w;
    this.h = h;
    this.velocity = new PingPong(gl, w, h, [f.rg16f], gl.LINEAR);
    this.pressure = new PingPong(gl, w, h, [f.r16f], gl.NEAREST);
    this.divergence = new Target(gl, w, h, [f.r16f], gl.NEAREST);
    this.curl = new Target(gl, w, h, [f.r16f], gl.NEAREST);
  }

  /** Queue a splat. x, y in uv; for SPLAT_DIR (fx, fy) is the force, otherwise fx is the magnitude. */
  splat(x: number, y: number, fx: number, fy: number, radius: number, type: number): void {
    if (this.splatCount >= MAX_SPLATS) return;
    const i = this.splatCount++ * 4;
    this.splatA[i] = x;
    this.splatA[i + 1] = y;
    this.splatA[i + 2] = fx;
    this.splatA[i + 3] = fy;
    this.splatB[i] = radius;
    this.splatB[i + 1] = type;
    this.splatB[i + 2] = fx;
    this.splatB[i + 3] = 0;
  }

  step(dt: number, time: number, noise: number, vorticity: number, aspect: number): void {
    const gl = this.gl;
    const tx = 1 / this.w;
    const ty = 1 / this.h;
    gl.disable(gl.BLEND);

    // Forces: queued splats plus continuous curl noise.
    this.velocity.write.bind();
    this.pSplat
      .use()
      .f2('uTexel', tx, ty)
      .tex('uTarget', this.velocity.read.t)
      .f1('uAspect', aspect)
      .f1('uTime', time)
      .f1('uNoise', noise)
      .f1('uNoiseScale', 2.2)
      .i1('uCount', this.splatCount)
      .f4v('uSplatA', this.splatA)
      .f4v('uSplatB', this.splatB);
    this.fs.draw();
    this.velocity.swap();
    this.splatCount = 0;

    this.curl.bind();
    this.pCurl.use().f2('uTexel', tx, ty).tex('uVelocity', this.velocity.read.t);
    this.fs.draw();

    this.velocity.write.bind();
    this.pVort
      .use()
      .f2('uTexel', tx, ty)
      .tex('uVelocity', this.velocity.read.t)
      .tex('uCurl', this.curl.t)
      .f1('uCurlStrength', vorticity)
      .f1('uDt', dt);
    this.fs.draw();
    this.velocity.swap();

    this.divergence.bind();
    this.pDiv.use().f2('uTexel', tx, ty).tex('uVelocity', this.velocity.read.t);
    this.fs.draw();

    this.pressure.write.bind();
    this.pScale.use().tex('uTex', this.pressure.read.t).f1('uValue', this.pressureDecay);
    this.fs.draw();
    this.pressure.swap();

    this.pPressure.use().f2('uTexel', tx, ty);
    for (let i = 0; i < this.iterations; i++) {
      this.pressure.write.bind();
      this.pPressure.use().f2('uTexel', tx, ty).tex('uDivergence', this.divergence.t).tex('uPressure', this.pressure.read.t);
      this.fs.draw();
      this.pressure.swap();
    }

    this.velocity.write.bind();
    this.pGrad.use().f2('uTexel', tx, ty).tex('uPressure', this.pressure.read.t).tex('uVelocity', this.velocity.read.t);
    this.fs.draw();
    this.velocity.swap();

    this.velocity.write.bind();
    this.pAdvect
      .use()
      .f2('uTexel', tx, ty)
      .tex('uVelocity', this.velocity.read.t)
      .tex('uSource', this.velocity.read.t)
      .f2('uSimTexel', tx, ty)
      .f1('uDt', dt)
      .f1('uDissipation', this.dissipation);
    this.fs.draw();
    this.velocity.swap();
  }

  private disposeTargets(): void {
    if (!this.velocity) return;
    this.velocity.dispose();
    this.pressure.dispose();
    this.divergence.dispose();
    this.curl.dispose();
  }

  dispose(): void {
    this.disposeTargets();
    for (const p of [this.pSplat, this.pCurl, this.pVort, this.pDiv, this.pScale, this.pPressure, this.pGrad, this.pAdvect]) {
      p.dispose();
    }
  }
}
