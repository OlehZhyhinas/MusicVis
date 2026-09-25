// The lyrics gene's smeared caption on the GPU: the line being sung, drawn by the overlay into a
// transparent 2D canvas, uploaded as a texture and added into a slot's feedback buffer each frame,
// tinted by the slot's palette. The feedback chain then carries it (zoom, swirl, ripple...), so the
// words melt into the visual behind the crisp caption on top.

import { Fullscreen, GL, Program } from '../../render/gl';
import { FULLSCREEN_VS } from '../../render/shaders';

const CAPTION_FS = /* glsl */ `#version 300 es
precision highp float;
in vec2 vUv;
uniform sampler2D uTex;
uniform float uValue;
uniform vec3 uColor;
out vec4 o;
void main() {
  float a = texture(uTex, vec2(vUv.x, 1.0 - vUv.y)).a;
  o = vec4(uColor * (uValue * a), 0.0);
}`;

export class CaptionLayer {
  private prog: Program;
  private tex: WebGLTexture | null = null;
  private version = -1;
  /** 0..1 how visible the caption is this frame (set by the app from the line's fade). */
  alpha = 0;
  has = false;

  constructor(private gl: GL, private fs: Fullscreen) {
    this.prog = new Program(gl, FULLSCREEN_VS, CAPTION_FS, 'v2-lyric-caption');
  }

  /** New caption pixels (only re-uploaded when `version` changes); null clears it. */
  set(source: TexImageSource | null, version: number, alpha: number): void {
    this.alpha = alpha;
    if (!source) {
      this.has = false;
      return;
    }
    this.has = true;
    if (version === this.version && this.tex) return;
    this.version = version;
    const gl = this.gl;
    if (!this.tex) {
      this.tex = gl.createTexture();
      gl.bindTexture(gl.TEXTURE_2D, this.tex);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    }
    gl.bindTexture(gl.TEXTURE_2D, this.tex);
    gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, source);
  }

  /** Adds the caption into the bound target (additive blending must be on). */
  draw(value: number, r: number, g: number, b: number): void {
    if (!this.tex || !this.has || value <= 1e-4) return;
    this.prog.use().tex('uTex', this.tex).f1('uValue', value).f3('uColor', r, g, b);
    this.fs.draw();
  }

  dispose(): void {
    if (this.tex) this.gl.deleteTexture(this.tex);
    this.tex = null;
    this.prog.dispose();
  }
}
