// Dual-filter bloom: soft-knee prefilter, progressive downsample, additive
// tent upsample back to the half-resolution level.

import { Fullscreen, GL, Program, Target, TexFormat } from './gl';
import { BLOOM_DOWN_FS, BLOOM_PREFILTER_FS, BLOOM_UP_FS, FULLSCREEN_VS } from './shaders';

export class Bloom {
  private mips: Target[] = [];
  private pPre: Program;
  private pDown: Program;
  private pUp: Program;
  levels = 6;

  constructor(
    private gl: GL,
    private fs: Fullscreen,
  ) {
    this.pPre = new Program(gl, FULLSCREEN_VS, BLOOM_PREFILTER_FS, 'bloom-prefilter');
    this.pDown = new Program(gl, FULLSCREEN_VS, BLOOM_DOWN_FS, 'bloom-down');
    this.pUp = new Program(gl, FULLSCREEN_VS, BLOOM_UP_FS, 'bloom-up');
  }

  get output(): WebGLTexture {
    return this.mips[0].t;
  }

  resize(w: number, h: number, fmt: TexFormat): void {
    this.disposeTargets();
    let cw = Math.max(1, w >> 1);
    let ch = Math.max(1, h >> 1);
    for (let i = 0; i < this.levels; i++) {
      this.mips.push(new Target(this.gl, cw, ch, [fmt], this.gl.LINEAR));
      cw = Math.max(1, cw >> 1);
      ch = Math.max(1, ch >> 1);
    }
  }

  run(src: Target, threshold: number, knee: number): void {
    const gl = this.gl;
    gl.disable(gl.BLEND);
    const m = this.mips;
    m[0].bind();
    this.pPre.use().tex('uSrc', src.t).f2('uSrcTexel', src.texelX, src.texelY).f1('uThreshold', threshold).f1('uKnee', knee);
    this.fs.draw();
    for (let i = 1; i < m.length; i++) {
      m[i].bind();
      this.pDown.use().tex('uSrc', m[i - 1].t).f2('uSrcTexel', m[i - 1].texelX, m[i - 1].texelY);
      this.fs.draw();
    }
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.ONE, gl.ONE);
    for (let i = m.length - 1; i > 0; i--) {
      m[i - 1].bind();
      this.pUp.use().tex('uSrc', m[i].t).f2('uSrcTexel', m[i].texelX, m[i].texelY).f1('uWeight', 1);
      this.fs.draw();
    }
    gl.disable(gl.BLEND);
  }

  private disposeTargets(): void {
    for (const t of this.mips) t.dispose();
    this.mips.length = 0;
  }

  dispose(): void {
    this.disposeTargets();
    this.pPre.dispose();
    this.pDown.dispose();
    this.pUp.dispose();
  }
}
