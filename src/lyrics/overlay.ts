// The on-screen lyric line for presets whose lyrics gene shows it: a subtle caption (show 1) or a
// karaoke line that fills as it is sung with the next line beneath (show 2). With smear > 0 the same
// line is also drawn into a transparent canvas that the engine adds into the feedback, so the visual
// carries and melts it behind the crisp caption.

import type { MusicState } from '../types';
import type { Params } from '../v2/genome';

export interface CaptionFrame {
  source: HTMLCanvasElement | null;
  version: number;
  alpha: number;
}

const FADE_IN = 8; // 1/s
const FADE_OUT = 4;

export class LyricOverlay {
  private el: HTMLElement;
  private lineEl: HTMLElement;
  private fillEl: HTMLElement;
  private nextEl: HTMLElement;
  private text = '';
  private next = '';
  private mode = -1;
  private vis = 0;
  private canvas: HTMLCanvasElement | null = null;
  private version = 0;
  private drawn = '';
  private out: CaptionFrame = { source: null, version: 0, alpha: 0 };

  constructor(host: HTMLElement, private viz: HTMLElement) {
    this.el = document.createElement('div');
    this.el.className = 'lyr-cap';
    this.el.setAttribute('aria-live', 'off');
    this.el.hidden = true;
    this.el.innerHTML = '<div class="lyr-line"><span class="lyr-fill"></span></div><div class="lyr-next"></div>';
    this.lineEl = this.el.querySelector('.lyr-line')!;
    this.fillEl = this.el.querySelector('.lyr-fill')!;
    this.nextEl = this.el.querySelector('.lyr-next')!;
    host.insertBefore(this.el, viz.nextSibling);
  }

  /**
   * Updates the caption for this frame from the state's lyric fields and the shown preset's lyrics
   * gene (undefined: no gene, nothing shown). Returns the smear canvas for the engine.
   */
  update(state: MusicState, gene: Params | undefined, dt: number): CaptionFrame {
    const show = gene && state.lyricLine !== undefined ? gene.show : 0;
    const o = this.out;
    if (!show) {
      if (!this.el.hidden) {
        this.el.hidden = true;
        this.text = '';
        this.lineEl.classList.remove('in');
      }
      this.vis = 0;
      o.source = null;
      o.alpha = 0;
      return o;
    }
    this.el.hidden = false;
    if (show !== this.mode) {
      this.mode = show;
      this.el.dataset.mode = String(show);
    }
    const line = state.lyricLine ?? '';
    if (line !== this.text) {
      this.text = line;
      this.lineEl.classList.remove('in');
      if (line) {
        this.fillEl.textContent = line;
        void this.lineEl.offsetWidth; // restart the fade-in
        this.lineEl.classList.add('in');
      }
    }
    const next = show === 2 ? (state.lyricNext ?? '') : '';
    if (next !== this.next) {
      this.next = next;
      this.nextEl.textContent = next;
    }
    const synced = state.lyricSynced !== false;
    const p = show === 2 && synced ? Math.min(1, Math.max(0, state.lyricProgress ?? 0)) : 1;
    this.fillEl.style.setProperty('--p', `${(p * 100).toFixed(1)}%`);

    // The smeared copy: redrawn when the line changes, faded with the line.
    const target = line ? 1 : 0;
    this.vis += (target - this.vis) * (1 - Math.exp(-Math.max(0, dt) * (target > this.vis ? FADE_IN : FADE_OUT)));
    const smear = gene?.smear ?? 0;
    if (smear > 0 && line) {
      if (line !== this.drawn) this.drawCanvas(line, show);
      o.source = this.canvas;
      o.version = this.version;
    } else if (smear <= 0) o.source = null;
    // Released mostly as the line starts, so the words stream off into the visual and the crisp
    // caption on top stays readable (a faint steady copy keeps the trail fed while it is sung).
    o.alpha = smear > 0 ? this.vis * Math.max(0.12, Math.min(1, state.lyricPulse ?? 0)) : 0;
    return o;
  }

  /** Draws the line into the smear canvas where the caption sits on screen. */
  private drawCanvas(line: string, show: number): void {
    const vr = this.viz.getBoundingClientRect();
    const scale = Math.min(1, 960 / Math.max(1, vr.width));
    const w = Math.max(2, Math.round(vr.width * scale));
    const h = Math.max(2, Math.round(vr.height * scale));
    const c = (this.canvas ??= document.createElement('canvas'));
    if (c.width !== w || c.height !== h) {
      c.width = w;
      c.height = h;
    }
    const ctx = c.getContext('2d');
    if (!ctx) return;
    ctx.clearRect(0, 0, w, h);
    const lr = this.fillEl.getBoundingClientRect();
    const fs = parseFloat(getComputedStyle(this.fillEl).fontSize) || (show === 2 ? 30 : 18);
    ctx.font = `600 ${fs * scale}px Inter, system-ui, -apple-system, 'Segoe UI', sans-serif`;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillStyle = '#fff';
    const x = lr.width > 0 ? (lr.left + lr.width / 2 - vr.left) * scale : w / 2;
    const y = lr.height > 0 ? (lr.top + lr.height / 2 - vr.top) * scale : h * 0.8;
    ctx.fillText(line, x, y, w * 0.9);
    this.drawn = line;
    this.version++;
  }
}
