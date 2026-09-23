// Lazily compiled per-preset programs. With KHR_parallel_shader_compile the
// driver compiles in the background and a preset only becomes selectable once
// its programs are linked, so switching never stalls a frame.

import { GL, PendingProgram, Program } from './gl';
import type { Preset } from './presets';
import { CURVE_FS, FULLSCREEN_VS, compositeFS, curveVS, feedbackFS } from './shaders';

export interface PresetPrograms {
  feedback: Program | null;
  composite: Program;
  curve: Program | null;
}

interface Entry {
  pending: (PendingProgram | null)[]; // feedback, composite, curve
  done: PresetPrograms | null;
  failed: boolean;
}

function defines(p: Preset): string[] {
  const d: string[] = [];
  if (p.fluid) d.push('USE_FLUID');
  if (p.rings) d.push('USE_RINGS');
  if (p.wrap) d.push('EDGE_WRAP');
  if (p.feedback === false) d.push('NO_FEEDBACK');
  return d;
}

export class ProgramCache {
  private entries = new Map<string, Entry>();
  private parallel: boolean;

  constructor(private gl: GL) {
    this.parallel = !!gl.getExtension('KHR_parallel_shader_compile');
  }

  /** Start compiling (no-op if already started). */
  request(p: Preset): void {
    if (this.entries.has(p.id)) return;
    const gl = this.gl;
    const src = { warp: p.warp, draw: p.draw, comp: p.comp, defines: defines(p) };
    const pending: (PendingProgram | null)[] = [
      p.feedback === false ? null : new PendingProgram(gl, FULLSCREEN_VS, feedbackFS(src), `${p.id}-feedback`, this.parallel),
      new PendingProgram(gl, FULLSCREEN_VS, compositeFS(src), `${p.id}-composite`, this.parallel),
      p.curve ? new PendingProgram(gl, curveVS(p.curve), CURVE_FS, `${p.id}-curve`, this.parallel) : null,
    ];
    this.entries.set(p.id, { pending, done: null, failed: false });
  }

  /** Programs if ready, null while compiling. Compiles synchronously when `now`. */
  get(p: Preset, now = false): PresetPrograms | null {
    this.request(p);
    const e = this.entries.get(p.id)!;
    if (e.done || e.failed) return e.done;
    let ready = true;
    for (const pp of e.pending) {
      if (!pp) continue;
      if (!pp.poll(now)) ready = false;
    }
    if (!ready) return null;
    const err = e.pending.find((pp) => pp?.error);
    if (err) {
      console.error(err.error);
      e.failed = true;
      for (const pp of e.pending) pp?.program?.dispose();
      return null;
    }
    e.done = {
      feedback: e.pending[0]?.program ?? null,
      composite: e.pending[1]!.program!,
      curve: e.pending[2]?.program ?? null,
    };
    return e.done;
  }

  failed(p: Preset): boolean {
    return !!this.entries.get(p.id)?.failed;
  }

  has(p: Preset): boolean {
    return this.entries.has(p.id);
  }

  dispose(): void {
    for (const e of this.entries.values()) {
      if (e.done) {
        e.done.feedback?.dispose();
        e.done.composite.dispose();
        e.done.curve?.dispose();
      }
    }
    this.entries.clear();
  }
}
