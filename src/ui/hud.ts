import { STEM_NAMES, type MusicState } from '../types';

const PITCH_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'];

export interface HudExtra {
  presetName: string;
  fps: number;
}

function formatTime(t: number): string {
  if (!isFinite(t) || t < 0) t = 0;
  const m = Math.floor(t / 60);
  const s = Math.floor(t % 60);
  return `${m}:${s.toString().padStart(2, '0')}`;
}

export class Hud {
  private el: HTMLElement;

  constructor(el: HTMLElement) {
    this.el = el;
  }

  update(state: MusicState, extra: HudExtra): void {
    const key = `${PITCH_NAMES[state.keyTonic % 12]} ${state.keyMode}`;
    const sectionEnd = isFinite(state.section.end) ? formatTime(state.section.end) : '—';
    const beatDots = Array.from({ length: 4 }, (_, i) => {
      const lit = state.beatIndex >= 0 && state.beatIndex % 4 === i;
      const downbeat = i === 0;
      return `<span class="hud-dot${lit ? ' lit' : ''}${downbeat ? ' downbeat' : ''}"></span>`;
    }).join('');

    const stemRows = STEM_NAMES.map((name) => {
      const v = Math.max(0, Math.min(1, state.stems[name]));
      const pct = Math.round(v * 100);
      return `
        <div class="hud-meter">
          <span class="hud-dim">${name.padEnd(6)}</span>
          <span class="hud-meter-track"><span class="hud-meter-fill" style="width:${pct}%"></span></span>
        </div>`;
    }).join('');

    this.el.innerHTML = `
      <div class="hud-row"><span>BPM</span><span>${state.bpm.toFixed(1)}</span></div>
      <div class="hud-row"><span>Key</span><span>${key}</span></div>
      <div class="hud-row"><span class="hud-dim">${state.section.label}</span><span class="hud-dim">${formatTime(state.time)} / ${sectionEnd}</span></div>
      <div class="hud-beats">${beatDots}</div>
      ${stemRows}
      <div class="hud-row"><span class="hud-dim">preset</span><span class="hud-dim">${extra.presetName}</span></div>
      <div class="hud-row"><span class="hud-dim">fps</span><span class="hud-dim">${extra.fps.toFixed(0)}</span></div>
    `;
  }
}
