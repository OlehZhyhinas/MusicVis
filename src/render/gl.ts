// Minimal WebGL2 helper layer: programs with cached uniforms, render targets,
// ping-pong pairs and a fullscreen triangle.

export type GL = WebGL2RenderingContext;

export interface TexFormat {
  internal: number;
  format: number;
  type: number;
}

export function formats(gl: GL) {
  return {
    rgba16f: { internal: gl.RGBA16F, format: gl.RGBA, type: gl.HALF_FLOAT } as TexFormat,
    rg16f: { internal: gl.RG16F, format: gl.RG, type: gl.HALF_FLOAT } as TexFormat,
    r16f: { internal: gl.R16F, format: gl.RED, type: gl.HALF_FLOAT } as TexFormat,
    rgba32f: { internal: gl.RGBA32F, format: gl.RGBA, type: gl.FLOAT } as TexFormat,
    rgba8: { internal: gl.RGBA8, format: gl.RGBA, type: gl.UNSIGNED_BYTE } as TexFormat,
  };
}

function numbered(src: string): string {
  return src
    .split('\n')
    .map((l, i) => `${String(i + 1).padStart(4)}  ${l}`)
    .join('\n');
}

function compile(gl: GL, type: number, src: string, name: string): WebGLShader {
  const sh = gl.createShader(type);
  if (!sh) throw new Error(`[render] cannot create shader for ${name}`);
  gl.shaderSource(sh, src);
  gl.compileShader(sh);
  if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) {
    const log = gl.getShaderInfoLog(sh);
    gl.deleteShader(sh);
    throw new Error(`[render] ${name} ${type === gl.VERTEX_SHADER ? 'VS' : 'FS'} compile error:\n${log}\n${numbered(src)}`);
  }
  return sh;
}

export class Program {
  readonly prog: WebGLProgram;
  private locs = new Map<string, WebGLUniformLocation | null>();
  private texUnit = 0;

  constructor(
    private gl: GL,
    vs: string,
    fs: string,
    readonly name: string,
  ) {
    const v = compile(gl, gl.VERTEX_SHADER, vs, name);
    const f = compile(gl, gl.FRAGMENT_SHADER, fs, name);
    const p = gl.createProgram();
    if (!p) throw new Error(`[render] cannot create program ${name}`);
    gl.attachShader(p, v);
    gl.attachShader(p, f);
    gl.linkProgram(p);
    gl.deleteShader(v);
    gl.deleteShader(f);
    if (!gl.getProgramParameter(p, gl.LINK_STATUS)) {
      throw new Error(`[render] ${name} link error: ${gl.getProgramInfoLog(p)}`);
    }
    this.prog = p;
  }

  /** Bind the program and reset the texture unit counter. */
  use(): this {
    this.gl.useProgram(this.prog);
    this.texUnit = 0;
    return this;
  }

  loc(name: string): WebGLUniformLocation | null {
    let l = this.locs.get(name);
    if (l === undefined) {
      l = this.gl.getUniformLocation(this.prog, name);
      this.locs.set(name, l);
    }
    return l;
  }

  f1(n: string, x: number): this {
    const l = this.loc(n);
    if (l) this.gl.uniform1f(l, x);
    return this;
  }
  f2(n: string, x: number, y: number): this {
    const l = this.loc(n);
    if (l) this.gl.uniform2f(l, x, y);
    return this;
  }
  f3(n: string, x: number, y: number, z: number): this {
    const l = this.loc(n);
    if (l) this.gl.uniform3f(l, x, y, z);
    return this;
  }
  f4(n: string, x: number, y: number, z: number, w: number): this {
    const l = this.loc(n);
    if (l) this.gl.uniform4f(l, x, y, z, w);
    return this;
  }
  v3(n: string, v: ArrayLike<number>): this {
    const l = this.loc(n);
    if (l) this.gl.uniform3f(l, v[0], v[1], v[2]);
    return this;
  }
  f1v(n: string, v: Float32Array): this {
    const l = this.loc(n);
    if (l) this.gl.uniform1fv(l, v);
    return this;
  }
  f4v(n: string, v: Float32Array): this {
    const l = this.loc(n);
    if (l) this.gl.uniform4fv(l, v);
    return this;
  }
  i1(n: string, x: number): this {
    const l = this.loc(n);
    if (l) this.gl.uniform1i(l, x);
    return this;
  }
  /** Bind a texture to the next free unit and point the sampler at it. */
  tex(n: string, t: WebGLTexture | null): this {
    const unit = this.texUnit++;
    this.gl.activeTexture(this.gl.TEXTURE0 + unit);
    this.gl.bindTexture(this.gl.TEXTURE_2D, t);
    const l = this.loc(n);
    if (l) this.gl.uniform1i(l, unit);
    return this;
  }

  dispose(): void {
    this.gl.deleteProgram(this.prog);
  }
}

export function createTexture(
  gl: GL,
  w: number,
  h: number,
  fmt: TexFormat,
  filter: number,
  data: ArrayBufferView | null = null,
): WebGLTexture {
  const t = gl.createTexture();
  if (!t) throw new Error('[render] cannot create texture');
  gl.bindTexture(gl.TEXTURE_2D, t);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, filter);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, filter);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.texImage2D(gl.TEXTURE_2D, 0, fmt.internal, w, h, 0, fmt.format, fmt.type, data);
  return t;
}

/** A framebuffer with one or more colour attachments. */
export class Target {
  readonly fbo: WebGLFramebuffer;
  readonly tex: WebGLTexture[] = [];
  readonly texelX: number;
  readonly texelY: number;

  constructor(
    private gl: GL,
    readonly w: number,
    readonly h: number,
    fmts: TexFormat[],
    filter: number,
    init: (ArrayBufferView | null)[] = [],
  ) {
    const fbo = gl.createFramebuffer();
    if (!fbo) throw new Error('[render] cannot create framebuffer');
    this.fbo = fbo;
    this.texelX = 1 / w;
    this.texelY = 1 / h;
    gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
    const bufs: number[] = [];
    fmts.forEach((f, i) => {
      const t = createTexture(gl, w, h, f, filter, init[i] ?? null);
      gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0 + i, gl.TEXTURE_2D, t, 0);
      this.tex.push(t);
      bufs.push(gl.COLOR_ATTACHMENT0 + i);
    });
    gl.drawBuffers(bufs);
    const status = gl.checkFramebufferStatus(gl.FRAMEBUFFER);
    if (status !== gl.FRAMEBUFFER_COMPLETE) {
      this.dispose();
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
      throw new Error(`[render] framebuffer incomplete (0x${status.toString(16)})`);
    }
    gl.viewport(0, 0, w, h);
    gl.clearColor(0, 0, 0, 0);
    if (!init.length) gl.clear(gl.COLOR_BUFFER_BIT);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  }

  get t(): WebGLTexture {
    return this.tex[0];
  }

  bind(): void {
    this.gl.bindFramebuffer(this.gl.FRAMEBUFFER, this.fbo);
    this.gl.viewport(0, 0, this.w, this.h);
  }

  dispose(): void {
    for (const t of this.tex) this.gl.deleteTexture(t);
    this.gl.deleteFramebuffer(this.fbo);
  }
}

/** Ping-pong pair of identical targets. */
export class PingPong {
  read: Target;
  write: Target;
  constructor(gl: GL, w: number, h: number, fmts: TexFormat[], filter: number, init: (ArrayBufferView | null)[] = []) {
    this.read = new Target(gl, w, h, fmts, filter, init);
    this.write = new Target(gl, w, h, fmts, filter, init);
  }
  get w(): number {
    return this.read.w;
  }
  get h(): number {
    return this.read.h;
  }
  swap(): void {
    const t = this.read;
    this.read = this.write;
    this.write = t;
  }
  dispose(): void {
    this.read.dispose();
    this.write.dispose();
  }
}

/** Draws a single oversized triangle covering the viewport (no attributes). */
export class Fullscreen {
  private vao: WebGLVertexArrayObject;
  constructor(private gl: GL) {
    const v = gl.createVertexArray();
    if (!v) throw new Error('[render] cannot create VAO');
    this.vao = v;
  }
  bindVao(): void {
    this.gl.bindVertexArray(this.vao);
  }
  draw(): void {
    this.gl.bindVertexArray(this.vao);
    this.gl.drawArrays(this.gl.TRIANGLES, 0, 3);
  }
  dispose(): void {
    this.gl.deleteVertexArray(this.vao);
  }
}

/** Check that a float format is renderable (and blendable) on this device. */
export function canRenderTo(gl: GL, fmt: TexFormat): boolean {
  try {
    const t = new Target(gl, 4, 4, [fmt], gl.NEAREST);
    t.dispose();
    return true;
  } catch {
    return false;
  }
}
