// The drift as the engine plays it (genes/drift.ts, genes/driftPath.ts): keeps the plan for the
// home genome on screen and the loaded song, and says each frame which genome to render. No GL here;
// the engine retargets its slot in place when the structure matches and crossfades otherwise.

import { structuralKey, type Genome } from '../genome';
import type { MusicState, Section } from '../../types';
import { performedAt, planDrift, type DriftPlan } from './driftPath';

/** Seconds ahead of a section boundary at which the next structure's shaders start compiling. */
export const DRIFT_LOOKAHEAD = 8;
/** Crossfade (seconds) hiding a recompile at a morphing boundary, and at a cut (a drop). */
export const DRIFT_XFADE = 0.45;
export const DRIFT_XFADE_CUT = 0.12;

export interface DriftFrame {
  /** The genome to render this frame (the home genome itself when nothing drifts). */
  genome: Genome;
  key: string;
  /** Crossfade to use when the structure changes. */
  xfade: number;
  /** The next section's genome when its boundary is near and its structure differs (compile it now). */
  upcoming: Genome | null;
}

export class DriftDriver {
  private song: readonly Section[] | null = null;
  private plan: DriftPlan | null = null;
  private planHome: Genome | null = null;
  private planSong: readonly Section[] | null = null;
  private homeKey = new WeakMap<Genome, string>();

  /** The loaded song's sections (null: no analysed song, e.g. live input). */
  setSong(sections: readonly Section[] | null): void {
    this.song = sections && sections.length ? sections : null;
  }

  /** The current plan (for tests and the HUD), made for `home` on the loaded song. */
  planFor(home: Genome): DriftPlan | null {
    if (!home.drift || !this.song) return null;
    if (this.plan && this.planHome === home && this.planSong === this.song) return this.plan;
    this.plan = planDrift(home, this.song);
    this.planHome = home;
    this.planSong = this.song;
    return this.plan;
  }

  /** What to render for `home` at this music state. */
  frame(home: Genome, state: MusicState): DriftFrame {
    const song = this.song;
    // Only a sampled song whose sections the plan was made from drifts (live input and idle play home).
    const onSong = !!song && !!state.section && song[state.sectionIndex] === state.section;
    const plan = onSong ? this.planFor(home) : null;
    if (!plan) {
      let key = this.homeKey.get(home);
      if (key === undefined) this.homeKey.set(home, (key = structuralKey(home)));
      return { genome: home, key, xfade: DRIFT_XFADE, upcoming: null };
    }
    const bar = state.barSeconds && state.barSeconds > 0 ? state.barSeconds : 240 / (state.bpm > 0 ? state.bpm : 120);
    const t = state.time;
    const perf = performedAt(plan, t, bar);
    const next = plan.stops[perf.index + 1];
    const upcoming = next && next.start - t < DRIFT_LOOKAHEAD && next.key !== perf.key ? next.genome : null;
    return { genome: perf.genome, key: perf.key, xfade: plan.stops[perf.index].cut ? DRIFT_XFADE_CUT : DRIFT_XFADE, upcoming };
  }
}
