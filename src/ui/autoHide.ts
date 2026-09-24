// Auto-hide: after 2.5 s without pointer or key input the whole UI fades and
// the cursor hides; a 2 px section-coloured progress line and a faint preset ID
// stay (also in fullscreen). The first few hides show a reveal hint.

import { loadSetting, saveSetting } from './storage';

const AUTO_HIDE_MS = 2500;
const HINT_TIMES = 3;

export class AutoHide {
  private timer: ReturnType<typeof setTimeout> | null = null;
  private hidden = false;
  private overChrome = false;
  private hints = loadSetting<number>('revealHints', 0);
  private hintEl = document.getElementById('reveal-hint');

  /** Returns true while something needs the UI (a drag, an open menu, typing). */
  constructor(private app: HTMLElement, private hold: () => boolean) {
    const reveal = () => this.reveal();
    for (const ev of ['mousemove', 'pointerdown', 'wheel', 'keydown']) document.addEventListener(ev, reveal, { passive: true, capture: true });
    document.addEventListener('touchstart', reveal, { passive: true });
    app.addEventListener('pointerover', (ev) => {
      this.overChrome = !!(ev.target as Element).closest?.('.chrome, .pop-layer, .toast');
    });
    document.addEventListener('mouseleave', () => (this.overChrome = false));
    this.reveal();
  }

  get isHidden(): boolean {
    return this.hidden;
  }

  reveal(): void {
    if (this.hidden) {
      this.hidden = false;
      this.app.classList.remove('ui-hidden', 'cursor-hidden');
      if (this.hintEl) this.hintEl.hidden = true;
    }
    this.schedule();
  }

  private schedule(): void {
    if (this.timer !== null) clearTimeout(this.timer);
    this.timer = setTimeout(() => this.tryHide(), AUTO_HIDE_MS);
  }

  private tryHide(): void {
    const ae = document.activeElement;
    const typing = ae instanceof HTMLInputElement && ae.type !== 'range' && ae.type !== 'checkbox' || ae instanceof HTMLTextAreaElement || ae instanceof HTMLSelectElement;
    if (this.overChrome || typing || this.hold()) {
      this.schedule();
      return;
    }
    this.hidden = true;
    this.app.classList.add('ui-hidden', 'cursor-hidden');
    if (this.hintEl && this.hints < HINT_TIMES) {
      this.hintEl.textContent = this.app.classList.contains('phone') ? 'Tap for controls' : 'Move the mouse or press any key for controls';
      this.hints++;
      saveSetting('revealHints', this.hints);
      this.hintEl.hidden = false;
      this.hintEl.classList.remove('fade-in');
      void this.hintEl.offsetWidth;
      this.hintEl.classList.add('fade-in');
      window.setTimeout(() => {
        if (this.hintEl) this.hintEl.hidden = true;
      }, 4000);
    }
  }
}
