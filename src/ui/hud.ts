// HUD (top-left): tempo, key, section with its time range and progress, beat
// dots, stem meters, and a footer with the preset, fps and frame stats. Built
// once; per-frame updates only touch values that changed.

import { STEM_NAMES, type MusicState } from '../types';
import { formatTime } from './transport';

const PITCH_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'];

export interface HudExtra {
  presetName: string;
  fps: number;
}

export class Hud {
  private bpm: HTMLElement;
  private key: HTMLElement;
  private section: HTMLElement;
  private range: HTMLElement;
  private progress: HTMLElement;
  private dots: HTMLElement[];
  private stems: HTMLElement[];
  private preset: HTMLElement;
  private fps: HTMLElement;
  private stats: HTMLElement;
  private shown: Record<string, string> = {};
  private slow = 0;

  constructor(el: HTMLElement) {
    el.classList.add('hudbox', 'glass');
    el.setAttribute('aria-label', 'HUD');
    el.innerHTML = `
      <div class="row hud-top">
        <div class="col" style="gap:1px"><span class="hud-bpm" data-k="bpm">–</span><span class="k">BPM</span></div>
        <div class="col" style="gap:1px;align-items:flex-end"><span class="hud-key" data-k="key">–</span><span class="k">KEY</span></div>
      </div>
      <div class="row hud-sec">
        <span class="tag solid" data-k="section"></span>
        <span class="hud-range" data-k="range"></span>
        <div class="beats">${'<i></i>'.repeat(4)}</div>
      </div>
      <div class="meter h3 hud-prog"><i data-k="progress" style="--v:0%"></i></div>
      <div class="col hud-stems">${STEM_NAMES.map((n) => `<div class="row"><span class="hud-stem">${n}</span><div class="meter h5"><i data-stem="${n}" style="--v:0%"></i></div></div>`).join('')}</div>
      <div class="hud-rule"></div>
      <div class="row hud-line"><span class="k">preset</span><span class="ell" data-k="preset"></span></div>
      <div class="row hud-line"><span class="k">fps</span><span data-k="fps"></span></div>
      <div class="hud-stats" data-k="stats"></div>`;
    const q = (k: string) => el.querySelector<HTMLElement>(`[data-k="${k}"]`)!;
    this.bpm = q('bpm');
    this.key = q('key');
    this.section = q('section');
    this.range = q('range');
    this.progress = q('progress');
    this.dots = [...el.querySelectorAll<HTMLElement>('.beats i')];
    this.stems = STEM_NAMES.map((n) => el.querySelector<HTMLElement>(`[data-stem="${n}"]`)!);
    this.preset = q('preset');
    this.fps = q('fps');
    this.stats = q('stats');
  }

  private set(el: HTMLElement, id: string, text: string): void {
    if (this.shown[id] === text) return;
    this.shown[id] = text;
    el.textContent = text;
  }

  update(state: MusicState, extra: HudExtra): void {
    // Beat dots every frame (they must stay in time); the rest ~12 times a second.
    const beat = state.beatIndex >= 0 ? state.beatIndex % 4 : -1;
    if (this.shown.beat !== String(beat)) {
      this.shown.beat = String(beat);
      this.dots.forEach((d, i) => (d.className = i === beat ? (i === 0 ? 'on down' : 'on') : ''));
    }
    this.slow -= state.dt;
    if (this.slow > 0) return;
    this.slow = 0.08;
    this.set(this.bpm, 'bpm', state.bpm > 0 ? state.bpm.toFixed(state.bpm >= 100 ? 0 : 1) : '–');
    this.set(this.key, 'key', `${PITCH_NAMES[state.keyTonic % 12]} ${state.keyMode}`);
    const s = state.section;
    if (this.shown.label !== s.label) {
      this.shown.label = s.label;
      this.section.textContent = s.label;
      this.section.style.setProperty('--c', `var(--s-${s.label})`);
      this.progress.style.setProperty('--c', `var(--s-${s.label})`);
    }
    const end = isFinite(s.end) ? formatTime(s.end) : '…';
    this.set(this.range, 'range', `${formatTime(s.start)} → ${end}`);
    const p = isFinite(s.end) && s.end > s.start ? Math.max(0, Math.min(1, (state.time - s.start) / (s.end - s.start))) : 0;
    this.progress.style.setProperty('--v', `${(p * 100).toFixed(1)}%`);
    STEM_NAMES.forEach((n, i) => this.stems[i].style.setProperty('--v', `${Math.round(Math.max(0, Math.min(1, state.stems[n])) * 100)}%`));
    this.set(this.preset, 'preset', extra.presetName);
    this.set(this.fps, 'fps', extra.fps.toFixed(0));
    this.fps.style.color = extra.fps >= 50 ? 'var(--ok)' : extra.fps >= 30 ? 'var(--warn)' : 'var(--neg)';
  }

  /** Frame stats line (frame · cpu · gpu · size · type). */
  setStats(text: string): void {
    this.set(this.stats, 'stats', text);
  }
}
