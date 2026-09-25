// The clip-duel page: two presets rendered live, side by side, on the same moment of the song
// the user has loaded (a hook stretch, the drop, a section change, picked from the analysis
// with the AV harness's logic, src/v2/avq/music.ts), with the song playing once, audibly.
// Two questions: "Which feels more in sync?" then "Which do you like more?" (1 / 2, S skip,
// R replay). While the clip plays, both renders are measured with the harness's metrics
// (src/v2/avq), so every answer is stored in IndexedDB with the two presets' metric vectors.
// A Bradley-Terry judge (judge.ts) is refit after each answer, the next pair is the one it is
// least sure about, and the page shows its cross-validated agreement (before 8 answers: the
// agreement on SIMULATED answers, labelled as such). The judge is saved as `v2.avqJudge` and
// marked usable for the screener / fitness from MIN_DUELS_TO_APPLY answers.

import type { AnalysisResult, MusicState } from '../types';
import type { Member } from './population';
import type { Engine, Slot } from './engine';
import { Stage } from './engine';
import type { Genome } from './genome';
import { icon } from '../ui/icons';
import { LiveRecorder } from './avq/live';
import { reportCard } from './avq/metrics';
import { findHooks, reviewWindows, songDataOf, songMoments, type ClipWindow, type Hook, type Moment, type SongData } from './avq/music';
import {
  JUDGE_FEATURES, JUDGE_KEY, MIN_DUELS_TO_APPLY, behaviourPairs, chooseDuel, judgeScore, judgeVector, simulateDuels, trainJudge,
  type Candidate, type Duel, type JudgeModel, type SavedJudge,
} from './judge';

/** Seed metric vectors from the offline harness sweep (optional; public/avq/seed-metrics.json). */
export const SEED_METRICS_URL = 'avq/seed-metrics.json';

const DW = 480;
const DH = 270;

export interface DuelAnswer {
  t: number;
  song: string;
  label: string;
  start: number;
  a: { id: string; v: number[] };
  b: { id: string; v: number[] };
  sync: 'a' | 'b' | null;
  like: 'a' | 'b' | null;
}


export interface DuelSong {
  id: string;
  result: AnalysisResult;
  seek(t: number): void;
  play(): void;
  pause(): void;
  time(): number;
  playing(): boolean;
}

export interface DuelDeps {
  eng: Engine;
  members(): Member[];
  /** The loaded song (null when none, or live input). */
  song(): DuelSong | null;
  toast(msg: string, detail?: string): void;
}

// ------------------------------------------------------------------ storage

function db(): Promise<IDBDatabase> {
  return new Promise((ok, fail) => {
    const r = indexedDB.open('musicvis-duels', 1);
    r.onupgradeneeded = () => r.result.createObjectStore('answers', { autoIncrement: true });
    r.onsuccess = () => ok(r.result);
    r.onerror = () => fail(r.error);
  });
}

async function loadAnswers(): Promise<DuelAnswer[]> {
  try {
    const d = await db();
    return await new Promise((ok) => {
      const r = d.transaction('answers').objectStore('answers').getAll();
      r.onsuccess = () => ok(r.result as DuelAnswer[]);
      r.onerror = () => ok([]);
    });
  } catch {
    return [];
  }
}

async function addAnswer(a: DuelAnswer): Promise<void> {
  try {
    const d = await db();
    await new Promise<void>((ok) => {
      const tx = d.transaction('answers', 'readwrite');
      tx.objectStore('answers').add(a);
      tx.oncomplete = () => ok();
      tx.onerror = () => ok();
    });
  } catch {
    /* private mode: answers live for this visit only */
  }
}


/** Duels for one question from the stored answers. */
export function duelsFor(answers: DuelAnswer[], q: 'sync' | 'like'): Duel[] {
  return answers.filter((a) => a[q]).map((a) => ({ va: a.a.v, vb: a.b.v, y: a[q] === 'a' ? 1 : 0 }));
}

const pct = (x: number) => (Number.isFinite(x) ? `${Math.round(x * 100)}%` : '--');
const NAN_V = () => JUDGE_FEATURES.map(() => NaN);

interface Side {
  member: Member;
  stage: Stage;
  slot: Slot | null;
  rec: LiveRecorder;
  canvas: HTMLCanvasElement;
  ctx: CanvasRenderingContext2D;
  img: ImageData;
  v: number[] | null;
}

// ------------------------------------------------------------------ page

export class DuelPage {
  private root: HTMLElement;
  private canvases: [HTMLCanvasElement, HTMLCanvasElement];
  private q: HTMLElement;
  private stats: HTMLElement;
  private answers: DuelAnswer[] = [];
  private stages: [Stage, Stage] | null = null;
  private px = new Uint8Array(DW * DH * 4);
  private sides: [Side, Side] | null = null;
  private win: ClipWindow | null = null;
  private song: DuelSong | null = null;
  private data: SongData | null = null;
  private hooks: Hook[] = [];
  private moments: Moment[] = [];
  private windows: ClipWindow[] = [];
  private phase: 'sync' | 'like' = 'sync';
  private play: 'idle' | 'compile' | 'playing' | 'done' = 'idle';
  private frames = 0;
  private playSecs = 0;
  private recAcc = 0;
  private pending: { sync?: 'a' | 'b' | null } = {};
  private shown = new Map<string, number>();
  private asked = new Set<string>();
  /** Latest metric vector per member id (from duels this visit and earlier ones, and the seed file). */
  private vectors = new Map<string, number[]>();
  private seedVectors = new Map<string, number[]>();
  private models: { sync: JudgeModel | null; like: JudgeModel | null } = { sync: null, like: null };
  private simulated = NaN;
  private lastFocus: HTMLElement | null = null;
  private rng = Math.random;

  constructor(private deps: DuelDeps) {
    this.root = document.createElement('div');
    this.root.className = 'sim duel';
    this.root.hidden = true;
    this.root.setAttribute('role', 'dialog');
    this.root.setAttribute('aria-modal', 'true');
    this.root.setAttribute('aria-labelledby', 'duel-q');
    this.root.innerHTML = `
      <div class="scrim" data-close></div>
      <div class="sim-card duel-card">
        <div class="sim-h">
          <div class="col grow" style="gap:2px"><h2 id="duel-q">Which feels more in sync?</h2>
          <span class="dim" data-moment>Same song moment, two presets. Your answers train the AV judge.</span></div>
          <button class="ib" data-close aria-label="Close" title="Close (Esc)">${icon('x', 16)}</button>
        </div>
        <div class="sim-pair duel-pair">
          <button class="sim-cand" data-pick="a" aria-label="The left render (1)"><canvas width="${DW}" height="${DH}"></canvas><span class="sim-cap"><span class="kbd">1</span></span></button>
          <button class="sim-cand" data-pick="b" aria-label="The right render (2)"><canvas width="${DW}" height="${DH}"></canvas><span class="sim-cap"><span class="kbd">2</span></span></button>
        </div>
        <div class="sim-actions">
          <button class="btn sm ghost" data-replay title="Play the moment again">Replay <span class="kbd">R</span></button>
          <button class="btn sm ghost" data-skip title="Can't tell">Skip <span class="kbd">S</span></button>
        </div>
        <div class="sim-stats" aria-live="polite"></div>
      </div>`;
    document.body.appendChild(this.root);
    const cv = this.root.querySelectorAll('canvas');
    this.canvases = [cv[0], cv[1]];
    this.q = this.root.querySelector('#duel-q')!;
    this.stats = this.root.querySelector('.sim-stats')!;
    this.root.addEventListener('click', (ev) => {
      const t = ev.target as HTMLElement;
      if (t.closest('[data-close]')) this.close();
      const p = t.closest<HTMLElement>('[data-pick]');
      if (p) void this.answer(p.dataset.pick === 'a' ? 'a' : 'b');
      if (t.closest('[data-skip]')) void this.answer(null);
      if (t.closest('[data-replay]')) this.start();
    });
    window.addEventListener('keydown', (ev) => {
      if (this.root.hidden) return;
      const k = ev.key;
      if (k === 'Escape') this.close();
      else if (k === '1') void this.answer('a');
      else if (k === '2') void this.answer('b');
      else if (k === 's' || k === 'S' || k === '0') void this.answer(null);
      else if (k === 'r' || k === 'R') this.start();
      else if (k === 'Tab') return;
      else return void ev.stopImmediatePropagation();
      ev.preventDefault();
      ev.stopImmediatePropagation();
    }, true);
  }

  get isOpen(): boolean {
    return !this.root.hidden;
  }

  async open(): Promise<void> {
    const song = this.deps.song();
    if (!song) {
      this.deps.toast('Load a song first', 'Duels play two presets on the same moment of the song you have loaded.');
      return;
    }
    if (this.visible().length < 2) {
      this.deps.toast('Not enough presets to compare');
      return;
    }
    if (this.song?.id !== song.id) {
      this.song = song;
      this.data = songDataOf(song.result);
      this.hooks = findHooks(this.data);
      this.moments = songMoments(this.data);
      this.windows = reviewWindows(this.data, this.hooks);
    }
    if (!this.windows.length) {
      this.deps.toast('No clear moment found in this song');
      return;
    }
    if (!this.seedVectors.size) await this.loadSeedVectors();
    this.answers = await loadAnswers();
    for (const a of this.answers) {
      this.asked.add([a.a.id, a.b.id].sort().join('|'));
      this.vectors.set(a.a.id, a.a.v);
      this.vectors.set(a.b.id, a.b.v);
    }
    this.refit();
    this.stages ??= [0, 1].map(() => {
      const st = new Stage(this.deps.eng, { offscreen: true, particleCap: 65536, flameCap: 131072 });
      st.resize(DW, DH);
      return st;
    }) as [Stage, Stage];
    this.lastFocus = document.activeElement as HTMLElement | null;
    this.root.hidden = false;
    this.next();
    (this.root.querySelector('[data-pick=a]') as HTMLElement).focus();
  }

  close(): void {
    this.root.hidden = true;
    this.song?.pause();
    this.disposeSides();
    this.play = 'idle';
    this.lastFocus?.focus?.();
  }

  private async loadSeedVectors(): Promise<void> {
    try {
      const r = await fetch(SEED_METRICS_URL, { cache: 'no-cache' });
      if (!r.ok) return;
      const j = (await r.json()) as { seeds: { preset: string; v: (number | null)[] }[] };
      for (const s of j.seeds) this.seedVectors.set(s.preset, s.v.map((x) => (x === null ? NaN : x)));
    } catch {
      /* optional */
    }
  }

  private visible(): Member[] {
    return this.deps.members().filter((m) => !m.hidden);
  }

  private vectorOf(m: Member): number[] {
    return this.vectors.get(m.id) ?? (m.origin ? this.seedVectors.get(m.origin) : undefined) ?? NAN_V();
  }

  private disposeSides(): void {
    if (!this.sides) return;
    for (const s of this.sides) if (s.slot) s.stage.disposeSlot(s.slot);
    this.sides = null;
  }

  /** Pick a moment and the pair the like-judge is least sure about, then start it. */
  private next(): void {
    this.win = this.windows[Math.floor(this.rng() * this.windows.length)];
    const members = this.visible();
    const byId = new Map(members.map((m) => [m.id, m]));
    const cands: Candidate[] = members.map((m) => ({ key: m.id, v: this.vectorOf(m) }));
    const pair = chooseDuel(cands, this.models.like, this.shown, this.asked, this.rng);
    if (!pair) {
      this.stats.textContent = 'No new pairs left to ask about.';
      return;
    }
    this.disposeSides();
    this.sides = pair.map((c, i) => {
      const canvas = this.canvases[i];
      const ctx = canvas.getContext('2d')!;
      return { member: byId.get(c.key)!, stage: this.stages![i], slot: null, rec: new LiveRecorder(DW, DH), canvas, ctx, img: ctx.createImageData(DW, DH), v: null };
    }) as [Side, Side];
    for (const s of this.sides) this.shown.set(s.member.id, (this.shown.get(s.member.id) ?? 0) + 1);
    this.phase = 'sync';
    this.pending = {};
    this.q.textContent = 'Which feels more in sync?';
    (this.root.querySelector('[data-moment]') as HTMLElement).textContent = `Moment: ${this.win.label} at ${this.win.start.toFixed(1)}-${this.win.end.toFixed(1)} s. Same music, two presets.`;
    this.start();
  }

  /** (Re)start the current pair from the window start: fresh slots, fresh measurements. */
  private start(): void {
    if (!this.sides || !this.win || !this.song) return;
    for (const s of this.sides) {
      if (s.slot) s.stage.disposeSlot(s.slot);
      s.slot = null;
      s.rec = new LiveRecorder(DW, DH);
      s.v = null;
    }
    this.play = 'compile';
    this.song.pause();
    this.renderStats();
  }

  /** Called by the app's frame loop instead of the main render while the page is open. */
  frame(state: MusicState, dt: number): void {
    if (!this.sides || !this.win || !this.song) return;
    if (this.play === 'compile') {
      const progs = this.sides.map((s) => this.deps.eng.cache.get(s.member.genome as Genome, true));
      const failed = this.sides.some((s) => this.deps.eng.cache.failed(s.member.genome));
      if (failed) {
        this.next();
        return;
      }
      if (progs.some((p) => !p)) return;
      this.sides.forEach((s, i) => {
        s.slot = s.stage.makeSlot(s.member.genome, progs[i]!);
        s.stage.slots = [s.slot];
        s.stage.resetHistory();
      });
      this.song.seek(this.win.start);
      this.song.play();
      this.play = 'playing';
      this.frames = 0;
      this.playSecs = 0;
      this.recAcc = 1;
      this.renderStats();
      return;
    }
    if (this.play !== 'playing') return;
    const t = this.song.time();
    if (t < this.win.start - 0.5) return; // seek not applied yet
    this.playSecs += dt;
    // Render every display frame, measure and show at ~30 Hz (the metrics' frame rate).
    this.recAcc += dt;
    const sample = this.recAcc >= 1 / 30 - 1e-3;
    if (sample) {
      this.recAcc = Math.min(this.recAcc - 1 / 30, 1 / 30);
      this.frames++;
    }
    for (const s of this.sides) {
      s.stage.render(state, dt, 'out');
      if (!sample) continue;
      s.stage.readPixels(this.px);
      s.rec.push(state, this.px, 30);
      const W4 = DW * 4;
      for (let y = 0; y < DH; y++) s.img.data.set(this.px.subarray((DH - 1 - y) * W4, (DH - y) * W4), y * W4);
      s.ctx.putImageData(s.img, 0, 0);
    }
    if (t >= this.win.end || !this.song.playing()) {
      this.song.pause();
      this.play = 'done';
      this.measure();
      this.renderStats();
    }
  }

  /** Report cards of both renders -> metric vectors (recorded at the display rate). */
  private measure(): void {
    if (!this.sides || !this.data) return;
    const fps = this.frames / Math.max(1e-3, this.playSecs);
    for (const s of this.sides) {
      if (s.rec.n < fps * 3) continue;
      const clip = s.rec.clip({ fps, bpm: this.data.bpm, beats: this.data.beats, downbeats: this.data.downbeats, beatsPerBar: this.data.beatsPerBar, sections: this.data.sections, moments: this.moments, hooks: this.hooks });
      const card = reportCard(clip, { preset: s.member.id, song: this.song?.id ?? '', clip: this.win?.label ?? '', clipT0: s.rec.t0 });
      s.v = judgeVector(card);
      this.vectors.set(s.member.id, s.v);
    }
  }

  private async answer(pick: 'a' | 'b' | null): Promise<void> {
    if (!this.sides || !this.win) return;
    if (this.phase === 'sync') {
      this.pending.sync = pick;
      this.phase = 'like';
      this.q.textContent = 'Which do you like more?';
      this.renderStats();
      return;
    }
    if (this.play === 'playing') {
      this.song?.pause();
      this.play = 'done';
      this.measure();
    }
    const [a, b] = this.sides;
    const ans: DuelAnswer = {
      t: Date.now(), song: this.song?.id ?? '', label: this.win.label, start: this.win.start,
      a: { id: a.member.id, v: a.v ?? this.vectorOf(a.member) }, b: { id: b.member.id, v: b.v ?? this.vectorOf(b.member) },
      sync: this.pending.sync ?? null, like: pick,
    };
    this.asked.add([a.member.id, b.member.id].sort().join('|'));
    if (ans.sync || ans.like) {
      this.answers.push(ans);
      await addAnswer(ans);
      this.refit();
    }
    this.next();
  }

  /** Refit both judges; behaviour-derived pairs join the like-judge as weak evidence. */
  private refit(): void {
    const withV = this.deps.members().map((m) => ({ ...m, v: this.vectorOf(m) })).filter((m) => m.v.some(Number.isFinite));
    const behaviour = behaviourPairs(withV);
    const sync = duelsFor(this.answers, 'sync');
    const like = [...duelsFor(this.answers, 'like'), ...behaviour];
    this.models = { sync: sync.length >= 4 ? trainJudge(sync) : null, like: like.length >= 4 ? trainJudge(like) : null };
    // Before real answers exist, show what the pipeline achieves on simulated ones (a made-up
    // preference for sync and hook rhyme, against stillness), labelled so it is never taken for yours.
    if (sync.length < 8 && Number.isNaN(this.simulated) && this.seedVectors.size > 8) {
      const cands = [...this.seedVectors].map(([key, v]) => ({ key, v }));
      const trueW = JUDGE_FEATURES.map((f) => (f === 'sync' ? 1.5 : f === 'hookRhyme' ? 1 : f === 'stillness' ? -1 : 0));
      let seed = 7;
      const r = () => (seed = (seed * 16807) % 2147483647) / 2147483647;
      this.simulated = trainJudge(simulateDuels(cands, 60, trueW, r, 0.7)).cv;
    }
    const n = this.answers.length;
    const saved: SavedJudge = { version: 1, features: [...JUDGE_FEATURES], n, applied: n >= MIN_DUELS_TO_APPLY, sync: this.models.sync, like: this.models.like };
    try {
      localStorage.setItem(JUDGE_KEY, JSON.stringify(saved));
    } catch {
      /* storage blocked: the judge still works for this visit */
    }
  }

  private renderStats(): void {
    const n = this.answers.length;
    const s = this.models.sync, l = this.models.like;
    const lines: string[] = [];
    if (this.play === 'compile') lines.push('Preparing both presets…');
    else if (this.play === 'playing') lines.push('Playing the moment once. Answer any time.');
    lines.push(`${n} duel${n === 1 ? '' : 's'} answered${n < MIN_DUELS_TO_APPLY ? ` (the judge starts guiding breeding from ${MIN_DUELS_TO_APPLY})` : ' (judge active)'}.`);
    if (s && Number.isFinite(s.cv)) lines.push(`Sync judge agrees with you on held-out duels ${pct(s.cv)}; like judge ${pct(l?.cv ?? NaN)}.`);
    else lines.push(`Held-out agreement appears after 8 answers.${Number.isFinite(this.simulated) ? ` Pipeline check on 60 simulated duels (a made-up preference, not yours): ${pct(this.simulated)}.` : ''}`);
    if (l && n >= 8 && this.seedVectors.size) {
      const ranked = [...this.seedVectors].map(([id, v]) => ({ id, s: judgeScore(l, v) })).sort((a, b) => b.s - a.s);
      lines.push(`Judge's top seeds: ${ranked.slice(0, 5).map((x) => x.id).join(', ')}; bottom: ${ranked.slice(-5).map((x) => x.id).join(', ')}.`);
    }
    if (this.sides) lines.push(`<span class="dim">${this.phase === 'sync' ? 'Step 1 of 2' : 'Step 2 of 2'}. Preset names stay hidden so they cannot bias you.</span>`);
    this.stats.innerHTML = lines.join('<br>');
  }
}
