// Online song structure for live input. Runs at ~10 Hz on smoothed features.
//
// - Drops: a large jump in complexity / drum + bass presence / loudness right
//   after a build or a quieter passage, with absolute minimums in the spirit of
//   the offline strict drop rule (structure.ts). Checked on a short window so
//   a drop registers within ~0.5-1 s.
// - Builds: rising loudness / onset density over the last few seconds while
//   the low end (bass) is thin.
// - Other changes: novelty between the last ~2 s and the preceding ~8 s of
//   timbre + energy features; labels from energy and complexity relative to
//   what has been heard so far (intro / verse / chorus / breakdown).

import type { Section, SectionLabel } from '../types';
import { History, histMean, histSlope } from './rtUtil';

export const STRUCT_RATE = 10; // updates per second
const TIMBRE_DIMS = 16;
const HIST_S = 24;

// Drop rule (absolute parts mirror structure.ts: DROP_MIN_COMPLEXITY 0.65, DROP_MIN_PRESENCE 0.4,
// DROP_MIN_CX_JUMP 0.15; slightly relaxed because the live measures are causal and short-windowed).
const DROP_MIN_CX = 0.68;
const DROP_MIN_PRESENCE = 0.4;
const DROP_MIN_CX_JUMP = 0.15;
const DROP_MIN_LOW_JUMP = 0.35; // rise of drums + bass presence after a build
const DROP_MIN_DB_JUMP = 3;
const DROP_MIN_SUDDEN_DB = 2.5; // the lift must be sudden (last 0.5 s vs 0.7-1.5 s ago), unlike a build's ramp
const DROP_MIN_KICK = 0.12; // share of the mix that is kick (low percussive energy)
const DROP_MIN_KICK_JUMP = 0.1;
const DROP_STRONG_KICK_JUMP = 0.25;
const DROP_STRONG_DB_JUMP = 5;
const DROP_CONFIRM_S = 2; // presence / complexity may confirm up to this long after the lift
const DROP_COOLDOWN_S = 12;

const NOVELTY_MIN_GAP_S = 6;
const SILENCE_SECTION_S = 1.5;
const NEW_SONG_SILENCE_S = 5;

export interface StructFrame {
  t: number;
  db: number; // mix level, dBFS
  cx: number; // fast complexity, 0..1
  pd: number; // drums presence
  pb: number; // bass presence
  kick: number; // share of the mix that is low percussive energy (kick), 0..1
  rate: number; // onsets per second
  gate: number; // 0..1, 0 = silent
  timbre: ArrayLike<number>; // TIMBRE_DIMS log band energies (dB)
}

export class StructureTracker {
  section: Section = { start: 0, end: 30, label: 'intro', energy: 0 };
  sectionIndex = 0;
  /** Increments on every section change. */
  changes = 0;
  prevLabel: SectionLabel = 'intro';
  buildIntensity = 0;
  /** Increments when music starts after a long silence (a new song). */
  newSongs = 0;
  /** Stream times of detected drops (for tests / diagnostics). */
  readonly dropTimes: number[] = [];
  /** Debug: last novelty value. */
  novelty = 0;

  private readonly db = new History(HIST_S * STRUCT_RATE);
  private readonly cx = new History(HIST_S * STRUCT_RATE);
  private readonly pd = new History(HIST_S * STRUCT_RATE);
  private readonly pb = new History(HIST_S * STRUCT_RATE);
  private readonly rate = new History(HIST_S * STRUCT_RATE);
  private readonly kick = new History(HIST_S * STRUCT_RATE);
  private readonly timbre: History[] = [];
  private readonly tMean = new Float64Array(TIMBRE_DIMS);
  private readonly tVar = new Float64Array(TIMBRE_DIMS).fill(25);
  private statsN = 0;

  private topDb = -60;
  private maxE = 0;
  private hadHigh = false;
  private silentFor = 0;
  private musicFor = 0;
  private lastDropT = -Infinity;
  private buildStart = 0;
  private buildStartDb = 0;
  private relabelled = false;
  private started = false;
  private liftT = -Infinity;
  private readonly liftPre = { cx: 0, pd: 0, pb: 0, kick: 0, db: 0 };
  private liftAfterBuild = false;

  constructor() {
    for (let k = 0; k < TIMBRE_DIMS; k++) this.timbre.push(new History(HIST_S * STRUCT_RATE));
  }

  reset(): void {
    this.section = { start: 0, end: 30, label: 'intro', energy: 0 };
    this.sectionIndex = 0;
    this.changes = 0;
    this.prevLabel = 'intro';
    this.buildIntensity = 0;
    this.dropTimes.length = 0;
    this.newSongs = 0;
    this.resetSong();
    this.silentFor = 0;
    this.musicFor = 0;
    this.started = false;
  }

  private resetSong(): void {
    for (const h of [this.db, this.cx, this.pd, this.pb, this.rate, this.kick, ...this.timbre]) h.reset();
    this.tMean.fill(0);
    this.tVar.fill(25);
    this.statsN = 0;
    this.topDb = -60;
    this.maxE = 0;
    this.hadHigh = false;
    this.lastDropT = -Infinity;
  }

  private energyOf(db: number, pd: number, pb: number): number {
    const loudRel = Math.max(0, Math.min(1, (db - (this.topDb - 30)) / 30));
    return 0.6 * loudRel + 0.25 * pd + 0.15 * pb;
  }

  private begin(t: number, label: SectionLabel, barSeconds: number): void {
    this.prevLabel = this.section.label;
    this.section = { start: t, end: t + 16 * barSeconds, label, energy: this.section.energy };
    this.sectionIndex++;
    this.changes++;
    this.relabelled = false;
    if (label === 'drop') {
      this.dropTimes.push(t);
      this.lastDropT = t;
    }
    if (label === 'build') {
      this.buildStart = t;
      this.buildStartDb = histMean(this.db, 0, 2 * STRUCT_RATE);
    }
    if (label === 'drop' || label === 'chorus') this.hadHigh = true;
  }

  private labelFor(cx: number, e: number): SectionLabel {
    if (this.maxE > 0 && e >= 0.8 * this.maxE && cx >= 0.5) return 'chorus';
    if (e <= 0.45 * this.maxE || cx < 0.3) return this.hadHigh ? 'breakdown' : 'intro';
    return 'verse';
  }

  update(f: StructFrame, barSeconds: number): void {
    const R = STRUCT_RATE;
    const dt = 1 / R;
    const t = f.t;
    if (!this.started) {
      this.started = true;
      this.section = { start: t, end: t + 16 * barSeconds, label: 'intro', energy: 0 };
    }
    const active = f.gate > 0.5;

    // --- Silence / new song ---
    if (!active) {
      this.silentFor += dt;
      if (this.silentFor >= SILENCE_SECTION_S && this.section.label !== 'intro' && this.section.label !== 'outro') {
        this.begin(t, 'outro', barSeconds);
      }
      if (this.silentFor >= NEW_SONG_SILENCE_S) this.musicFor = 0;
      this.buildIntensity *= Math.exp(-dt / 0.4);
      this.section.end = Math.max(this.section.end, t + 4 * barSeconds);
      return;
    }
    if (this.silentFor >= NEW_SONG_SILENCE_S && this.musicFor === 0 && this.changes > 0) {
      // Music again after a long silence: treat as a new song.
      this.resetSong();
      this.newSongs++;
      this.begin(t, 'intro', barSeconds);
    }
    this.silentFor = 0;
    this.musicFor += dt;

    this.db.push(f.db);
    this.cx.push(f.cx);
    this.pd.push(f.pd);
    this.pb.push(f.pb);
    this.rate.push(f.rate);
    this.kick.push(f.kick);
    for (let k = 0; k < TIMBRE_DIMS; k++) this.timbre[k].push(f.timbre[k]);
    // Running per-dimension timbre statistics (~30 s memory).
    this.statsN++;
    const a = Math.max(1 / this.statsN, 1 / (30 * R));
    for (let k = 0; k < TIMBRE_DIMS; k++) {
      const d = f.timbre[k] - this.tMean[k];
      this.tMean[k] += a * d;
      this.tVar[k] += a * (d * d - this.tVar[k]);
    }

    const db2 = histMean(this.db, 0, 2 * R);
    this.topDb = Math.max(db2, this.topDb - 0.02 * dt);
    const cx2 = histMean(this.cx, 0, 2 * R);
    const pd2 = histMean(this.pd, 0, 2 * R);
    const pb2 = histMean(this.pb, 0, 2 * R);
    const e2 = this.energyOf(db2, pd2, pb2);
    if (this.musicFor > 2) this.maxE = Math.max(this.maxE * Math.exp(-dt / 120), e2);
    const age = t - this.section.start;
    this.section.energy += (Math.max(0, Math.min(1, (db2 - (this.topDb - 30)) / 30)) - this.section.energy) * 0.05;
    this.section.end = Math.max(this.section.end, t + 4 * barSeconds);

    // --- Drop (short window) ---
    const n = this.db.count;
    if (n >= 3 * R) {
      const cur = {
        cx: histMean(this.cx, 0, 5),
        pd: histMean(this.pd, 0, 5),
        pb: histMean(this.pb, 0, 5),
        kick: histMean(this.kick, 0, 5),
        db: histMean(this.db, 0, 5),
      };
      const recentDb = histMean(this.db, 7, 15);
      const recentKick = histMean(this.kick, 7, 15);
      const lab = this.section.label;
      // A sudden lift (level or kick) opens a short window in which the slower
      // presence / complexity measures may confirm the drop. What came before is
      // frozen at the lift (the 4.5 s ending 1 s before it).
      if (cur.db - recentDb >= DROP_MIN_SUDDEN_DB || cur.kick - recentKick >= DROP_MIN_KICK_JUMP) {
        if (t - this.liftT > DROP_CONFIRM_S) {
          const p = this.liftPre;
          p.cx = histMean(this.cx, 10, 55);
          p.pd = histMean(this.pd, 10, 55);
          p.pb = histMean(this.pb, 10, 55);
          p.kick = histMean(this.kick, 10, 55);
          p.db = histMean(this.db, 10, 55);
          this.liftAfterBuild = lab === 'build' || (this.prevLabel === 'build' && age < 4);
        }
        this.liftT = t;
      }
      const sudden = t - this.liftT <= DROP_CONFIRM_S;
      const pre = this.liftPre;
      const absOk = cur.cx >= DROP_MIN_CX && cur.pd + cur.pb >= 2 * DROP_MIN_PRESENCE && cur.kick >= DROP_MIN_KICK;
      const kickJump = cur.kick - pre.kick;
      const lowJump = cur.pd + cur.pb - (pre.pd + pre.pb);
      const dbJump = cur.db - pre.db;
      // After a build the build carries the lift: the kick / low end must come
      // back and the level must jump. Otherwise (after a quieter passage) the
      // complexity must also jump, as in the offline rule.
      const liftOk = this.liftAfterBuild
        ? dbJump >= DROP_MIN_DB_JUMP && (kickJump >= DROP_MIN_KICK_JUMP || lowJump >= DROP_MIN_LOW_JUMP)
        : dbJump >= DROP_MIN_DB_JUMP && kickJump >= DROP_MIN_KICK_JUMP && cur.cx - pre.cx >= DROP_MIN_CX_JUMP;
      // Fast path: right after a build, a big kick + level jump is a drop even before
      // the slower complexity / presence measures have caught up.
      const strong = this.liftAfterBuild && kickJump >= DROP_STRONG_KICK_JUMP && dbJump >= DROP_STRONG_DB_JUMP && cur.kick >= 2 * DROP_MIN_KICK;
      if ((strong || (absOk && liftOk)) && sudden && lab !== 'drop' && t - this.lastDropT >= DROP_COOLDOWN_S) {
        this.begin(t, 'drop', barSeconds);
        this.buildIntensity *= 0.5;
        return;
      }
    }

    // --- Build ---
    if (this.section.label === 'build') {
      const elapsed = t - this.buildStart;
      const rise = histMean(this.db, 0, 5) - this.buildStartDb;
      const p = Math.max(0, Math.min(1, 0.7 * (elapsed / (8 * barSeconds)) + 0.3 * (rise / 8)));
      this.buildIntensity = Math.max(this.buildIntensity * Math.exp(-dt / 0.4), p * p);
      // A build that never drops is not a build after all.
      if (elapsed > 24 * barSeconds && age >= 4) {
        this.begin(t, this.labelFor(cx2, e2), barSeconds);
        return;
      }
    } else {
      this.buildIntensity *= Math.exp(-dt / 0.4);
      if (n >= 4 * R && age >= 2) {
        // Slope over the section so far (2..4 s), so a build is seen right after its boundary.
        const w = Math.round(Math.max(2, Math.min(4, age)) * R);
        const dbSlope = histSlope(this.db, w) * R; // dB per second
        const rateNow = histMean(this.rate, 0, 2 * R);
        const ratePrev = histMean(this.rate, 2 * R, 4 * R);
        const pbNow = histMean(this.pb, 0, 2 * R);
        // A ramp, not a step (music coming back after a short break): the level
        // rises through each of the last four seconds.
        const m0 = histMean(this.db, 0, R);
        const m1 = histMean(this.db, R, 2 * R);
        const m2 = histMean(this.db, 2 * R, 3 * R);
        const m3 = histMean(this.db, 3 * R, 4 * R);
        const ramp = m0 > m1 && m1 > m2 && m2 > m3 - 0.3 && m0 - m3 >= 1.5 && m0 - m1 < 4;
        const rising = ramp && dbSlope > 0.35 && histSlope(this.db, 2 * R) * R > 0.2;
        const denser = rateNow >= ratePrev - 0.25;
        if (rising && denser && (pbNow < 0.25 || histMean(this.kick, 0, 2 * R) < 0.04) && this.section.label !== 'intro') {
          this.begin(t - 2, 'build', barSeconds);
          return;
        }
      }
    }

    // --- Novelty ---
    if (n >= 8 * R && age >= NOVELTY_MIN_GAP_S) {
      let d = 0;
      for (let k = 0; k < TIMBRE_DIMS; k++) {
        const sd = Math.sqrt(Math.max(1, this.tVar[k]));
        const z = (histMean(this.timbre[k], 0, 2 * R) - histMean(this.timbre[k], 25, 100)) / sd;
        d += z * z;
      }
      d /= TIMBRE_DIMS;
      const dCx = histMean(this.cx, 0, 2 * R) - histMean(this.cx, 25, 100);
      const dPd = histMean(this.pd, 0, 2 * R) - histMean(this.pd, 25, 100);
      const dPb = histMean(this.pb, 0, 2 * R) - histMean(this.pb, 25, 100);
      const dDb = histMean(this.db, 0, 2 * R) - histMean(this.db, 25, 100);
      const nov = Math.sqrt(d) + 3 * Math.abs(dCx) + 1.5 * (Math.abs(dPd) + Math.abs(dPb)) + Math.abs(dDb) / 6;
      this.novelty = nov;
      if (nov > 1.4) {
        // Label from the last ~0.7 s: the 2 s means still straddle the boundary.
        const dbS = histMean(this.db, 0, 7);
        const label = this.labelFor(histMean(this.cx, 0, 7), this.energyOf(dbS, histMean(this.pd, 0, 7), histMean(this.pb, 0, 7)));
        const lab = this.section.label;
        // A build keeps its label until the drop (or the novelty says otherwise and it is not rising).
        const same = label === lab && (lab === 'intro' || lab === 'outro');
        if (!same && !(lab === 'build' && label !== 'breakdown' && label !== 'intro')) {
          this.begin(t - 1, label, barSeconds);
          return;
        }
      }
    }

    // One relabel ~3 s into a section, when the first estimate straddled the boundary.
    if (!this.relabelled && age >= 3 && this.section.label !== 'drop' && this.section.label !== 'build' && this.section.label !== 'outro') {
      this.relabelled = true;
      const label = this.labelFor(cx2, e2);
      if (label !== this.section.label && !(this.section.label === 'intro' && label === 'breakdown')) this.section.label = label;
      if (label === 'chorus') this.hadHigh = true;
    }
  }
}
