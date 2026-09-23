// GPU particle system: state in RGBA32F textures, updated in a fragment
// shader, drawn as GL_POINTS via gl_VertexID + texelFetch.

import { Fullscreen, GL, PingPong, Program, formats } from './gl';
import { FULLSCREEN_VS, PARTICLE_FS, PARTICLE_UPDATE_FS, PARTICLE_VS } from './shaders';

export interface ParticleUpdate {
  dt: number;
  time: number;
  aspect: number;
  velocity: WebGLTexture;
  simTexelX: number;
  simTexelY: number;
  wave: WebGLTexture;
  fluidAmt: number;
  curl: number;
  zoomFlow: number;
  rotFlow: number;
  converge: number;
  drag: number;
  lifeRate: number;
  speed: number;
  spawnFrom: number;
  spawnTo: number;
  spawnMix: number;
  emitCount: number;
  emitAngle: number;
  emitRadius: number;
  burst: number;
  burstSeed: number;
  burstSpeed: number;
  liftX: number;
  liftY: number;
  spread: number;
}

export class Particles {
  private state: PingPong | null = null;
  private pUpdate: Program;
  private pDraw: Program;
  private vao: WebGLVertexArrayObject;
  side = 0;

  constructor(
    private gl: GL,
    private fs: Fullscreen,
  ) {
    this.pUpdate = new Program(gl, FULLSCREEN_VS, PARTICLE_UPDATE_FS, 'particle-update');
    this.pDraw = new Program(gl, PARTICLE_VS, PARTICLE_FS, 'particle-draw');
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
    const gl = this.gl;
    const f = formats(gl);
    const total = side * side;
    const s0 = new Float32Array(total * 4);
    const s1 = new Float32Array(total * 4);
    for (let i = 0; i < total; i++) {
      s0[i * 4] = Math.random();
      s0[i * 4 + 1] = Math.random();
      s1[i * 4] = Math.random(); // life
      s1[i * 4 + 1] = Math.random(); // seed
    }
    this.state = new PingPong(gl, side, side, [f.rgba32f, f.rgba32f], gl.NEAREST, [s0, s1]);
  }

  update(u: ParticleUpdate): void {
    if (!this.state) return;
    const gl = this.gl;
    gl.disable(gl.BLEND);
    this.state.write.bind();
    this.pUpdate
      .use()
      .tex('uS0', this.state.read.tex[0])
      .tex('uS1', this.state.read.tex[1])
      .tex('uVel', u.velocity)
      .tex('uWave', u.wave)
      .f1('uDt', u.dt)
      .f1('uTime', u.time)
      .f1('uAspect', u.aspect)
      .f2('uSimTexel', u.simTexelX, u.simTexelY)
      .f1('uFluidAmt', u.fluidAmt)
      .f1('uCurl', u.curl)
      .f1('uZoomFlow', u.zoomFlow)
      .f1('uRotFlow', u.rotFlow)
      .f1('uConverge', u.converge)
      .f1('uDrag', u.drag)
      .f1('uLifeRate', u.lifeRate)
      .f1('uSpeed', u.speed)
      .f3('uSpawn', u.spawnFrom, u.spawnTo, u.spawnMix)
      .f4('uEmit', u.emitCount, u.emitAngle, u.emitRadius, u.spread)
      .f2('uLift', u.liftX, u.liftY)
      .f1('uBurst', u.burst)
      .f1('uBurstSeed', u.burstSeed)
      .f1('uBurstSpeed', u.burstSpeed);
    this.fs.draw();
    this.state.swap();
  }

  /** Draw soft point sprites into the currently bound framebuffer (additive). */
  draw(size: number, bright: number, alive: number, cols: Float32Array): void {
    if (!this.state) return;
    const gl = this.gl;
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.ONE, gl.ONE);
    this.pDraw
      .use()
      .tex('uS0', this.state.read.tex[0])
      .tex('uS1', this.state.read.tex[1])
      .i1('uW', this.side)
      .f1('uSize', size)
      .f1('uBright', bright)
      .f1('uAlive', alive)
      .f1('uStreak', 0)
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
