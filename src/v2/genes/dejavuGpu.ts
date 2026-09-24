// Visual deja vu on the GPU (WebGL2): the snapshot textures and the recall blend of one slot.
//   snapshot: the slot's feedback image, box-downsampled into a small texture per remembered group;
//   recall:   the remembered image, evolved (turned, pushed, hue-shifted), blended into the feedback
//             buffer before the feedback pass, so the chain carries it on and the bodies draw over it.
// The decisions (when to remember, how strongly to recall) come from the pure planner in dejavu.ts.

import { Fullscreen, GL, Program, Target, type TexFormat } from '../../render/gl';
import type { MusicState } from '../../types';
import { DejaVuPlanner, dejavuCueOf, evolution, imageAlpha, type DejaVuGene, type DejaVuPlan, type Evolution, type Framing } from './dejavu';

const HEAD = /* glsl */ `#version 300 es
precision highp float;
precision highp sampler2D;
`;
const FULL_VS = HEAD + /* glsl */ `
out vec2 vUv;
void main() {
  vec2 p = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2));
  vUv = p;
  gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}`;

/** 4-tap box downsample (each bilinear tap averages 2x2 source texels). */
const SNAP_FS = HEAD + /* glsl */ `
in vec2 vUv;
uniform sampler2D uSrc;
uniform vec2 uStep; // a quarter of a destination texel, in uv
out vec4 o;
void main() {
  vec3 c = texture(uSrc, vUv + vec2(-uStep.x, -uStep.y)).rgb + texture(uSrc, vUv + vec2(uStep.x, -uStep.y)).rgb
         + texture(uSrc, vUv + vec2(-uStep.x, uStep.y)).rgb + texture(uSrc, vUv + vec2(uStep.x, uStep.y)).rgb;
  o = vec4(max(c * 0.25, 0.0), 1.0);
}`;

/** The remembered image, turned and pushed about the centre (aspect-correct) and hue-rotated. */
const RECALL_FS = HEAD + /* glsl */ `
in vec2 vUv;
uniform sampler2D uMem;
uniform float uAspect, uZoom, uRot, uHue;
out vec4 o;
vec3 hueRotate(vec3 c, float turns) {
  // Rotation about the grey axis (luma-preserving enough for a memory).
  float a = turns * 6.28318530718;
  const vec3 k = vec3(0.57735);
  float ca = cos(a);
  return c * ca + cross(k, c) * sin(a) + k * dot(k, c) * (1.0 - ca);
}
void main() {
  vec2 d = (vUv - 0.5) * vec2(uAspect, 1.0);
  float c = cos(uRot), s = sin(uRot);
  d = mat2(c, s, -s, c) * d / uZoom;
  vec2 uv = d / vec2(uAspect, 1.0) + 0.5;
  // Mirror at the edges so a turned memory has no hard border.
  uv = 1.0 - abs(1.0 - abs(mod(uv, 2.0)));
  vec3 m = texture(uMem, uv).rgb;
  o = vec4(max(hueRotate(m, uHue), 0.0), 1.0);
}`;

/** Programs shared by every slot of an engine (compiled on first use). */
class Programs {
  readonly snap: Program;
  readonly recall: Program;
  constructor(gl: GL) {
    this.snap = new Program(gl, FULL_VS, SNAP_FS, 'dejavu-snap');
    this.recall = new Program(gl, FULL_VS, RECALL_FS, 'dejavu-recall');
  }
}
const PROGRAMS = new WeakMap<GL, Programs>();

/** The slot state deja vu reads and writes each frame. */
export interface DejaVuSlotView {
  /** The slot's pose this frame (choreography or identity), modified in place by a recall. */
  pose: Framing & { hue: number };
  /** Palette hue the slot would use without deja vu (key hue plus palette offset). */
  hue: number;
  /** The slot's free-running memory (phases). */
  mem: Record<string, number>;
}

/** One slot's deja vu: the planner plus its snapshot textures. */
export class DejaVu {
  readonly planner: DejaVuPlanner;
  private tex = new Map<number, Target>();
  private plan: DejaVuPlan = { snap: -1, evict: -1, recall: null };
  private cue = dejavuCueOf({ time: 0 } as MusicState);
  private ev: Evolution = { zoom: 1, rot: 0, hue: 0 };
  private alpha = 0;
  private framing: Framing = { zoom: 1, roll: 0, tx: 0, ty: 0 };
  private hue = 0;
  private phaseMem: Record<string, number> = {};

  constructor(
    private gl: GL,
    private fs: Fullscreen,
    private fmt: TexFormat,
    gene: DejaVuGene,
  ) {
    this.planner = new DejaVuPlanner(gene);
  }

  /** The gene may be edited in place (same slot): new params apply from the next frame. */
  retarget(gene: DejaVuGene): void {
    this.planner.gene = gene;
  }

  private programs(): Programs {
    let p = PROGRAMS.get(this.gl);
    if (!p) PROGRAMS.set(this.gl, (p = new Programs(this.gl)));
    return p;
  }

  /** CPU part of the frame: plan, then apply a recall's framing, colour and phases (before the slot ticks). */
  update(state: MusicState, sdt: number, view: DejaVuSlotView): void {
    this.cue = dejavuCueOf(state);
    this.plan = this.planner.plan(this.cue);
    // What a snapshot would remember: the framing and colour before any recall this frame.
    this.framing = { zoom: view.pose.zoom, roll: view.pose.roll, tx: view.pose.tx, ty: view.pose.ty };
    this.hue = view.hue + view.pose.hue;
    this.phaseMem = view.mem;
    const r = this.plan.recall;
    this.alpha = 0;
    if (r && this.tex.has(r.group)) {
      this.planner.apply(r, view.pose, view.hue + view.pose.hue, view.mem, this.cue.time);
      evolution(this.planner.gene, r.group, r.k, this.ev);
      this.alpha = imageAlpha(this.planner.gene, r.image, sdt);
    }
  }

  /** Blends the remembered image into the slot's feedback (bind nothing; call before the feedback pass). */
  recall(target: Target): void {
    const r = this.plan.recall;
    if (!r || this.alpha < 1e-4) return;
    const mem = this.tex.get(r.group);
    if (!mem) return;
    const gl = this.gl;
    target.bind();
    gl.enable(gl.BLEND);
    gl.blendColor(0, 0, 0, this.alpha);
    gl.blendFunc(gl.CONSTANT_ALPHA, gl.ONE_MINUS_CONSTANT_ALPHA);
    this.programs()
      .recall.use()
      .tex('uMem', mem.t)
      .f1('uAspect', target.w / target.h)
      .f1('uZoom', this.ev.zoom)
      .f1('uRot', this.ev.rot)
      .f1('uHue', this.ev.hue);
    this.fs.draw();
    gl.blendFunc(gl.ONE, gl.ONE);
    gl.disable(gl.BLEND);
  }

  /** Takes this frame's planned snapshot from the slot's feedback image (call after the feedback pass). */
  snapshot(src: Target): void {
    const plan = this.plan;
    if (plan.snap < 0) return;
    const p = this.planner.gene.p;
    if (plan.evict >= 0) {
      this.tex.get(plan.evict)?.dispose();
      this.tex.delete(plan.evict);
    }
    const w = Math.max(2, Math.round(src.w * p.res));
    const h = Math.max(2, Math.round(src.h * p.res));
    let t = this.tex.get(plan.snap);
    if (!t || t.w !== w || t.h !== h) {
      t?.dispose();
      t = new Target(this.gl, w, h, [this.fmt], this.gl.LINEAR);
      this.tex.set(plan.snap, t);
    }
    const gl = this.gl;
    t.bind();
    gl.disable(gl.BLEND);
    this.programs()
      .snap.use()
      .tex('uSrc', src.t)
      .f2('uStep', 0.25 / w, 0.25 / h);
    this.fs.draw();
    this.planner.remember(this.cue, plan, this.framing, this.hue, this.phaseMem);
    // Textures of groups the planner no longer holds (cap lowered by an edit) are freed.
    for (const g of [...this.tex.keys()]) {
      if (!this.planner.records.has(g)) {
        this.tex.get(g)!.dispose();
        this.tex.delete(g);
      }
    }
    plan.snap = -1;
  }

  /** Snapshots held (for tests and the HUD). */
  held(): number {
    return this.tex.size;
  }

  dispose(): void {
    for (const t of this.tex.values()) t.dispose();
    this.tex.clear();
    this.planner.reset();
  }
}

/** Every slot's deja vu for one stage (keyed by slot; created, retargeted and freed with the gene). */
export class DejaVuBank {
  private m = new WeakMap<object, DejaVu>();

  constructor(
    private gl: GL,
    private fs: Fullscreen,
    private fmt: TexFormat,
  ) {}

  /** The CPU part of a slot's frame (no-op, and frees the memory, when the genome has no deja vu). */
  update(slot: object, gene: DejaVuGene | undefined, state: MusicState, sdt: number, view: DejaVuSlotView): void {
    let d = this.m.get(slot);
    if (!gene) {
      if (d) this.drop(slot);
      return;
    }
    if (!d) this.m.set(slot, (d = new DejaVu(this.gl, this.fs, this.fmt, gene)));
    else d.retarget(gene);
    d.update(state, sdt, view);
  }

  recall(slot: object, target: Target): void {
    this.m.get(slot)?.recall(target);
  }

  snapshot(slot: object, src: Target): void {
    this.m.get(slot)?.snapshot(src);
  }

  get(slot: object): DejaVu | undefined {
    return this.m.get(slot);
  }

  drop(slot: object): void {
    this.m.get(slot)?.dispose();
    this.m.delete(slot);
  }
}
