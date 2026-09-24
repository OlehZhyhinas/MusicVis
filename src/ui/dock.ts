// The right-hand dock: Playlist, Presets and Genes as tabs of one panel (a
// bottom sheet on phones), so panels never collide. Remembers the last tab.

import { loadSetting, saveSetting } from './storage';

export type DockTab = 'playlist' | 'presets' | 'genes';
const TABS: DockTab[] = ['playlist', 'presets', 'genes'];

export class Dock {
  private cur: DockTab | null = null;
  private last: DockTab;
  private root = document.getElementById('dock')!;
  onChange: ((tab: DockTab | null) => void) | null = null;

  constructor() {
    const saved = loadSetting<string>('dockTab', 'playlist');
    this.last = (TABS as string[]).includes(saved) ? (saved as DockTab) : 'playlist';
    for (const t of TABS) {
      document.getElementById(`dt-${t}`)!.addEventListener('click', () => this.open(t));
    }
    document.getElementById('dock-close')!.addEventListener('click', () => this.close());
    // Arrow keys move between tabs (tablist pattern).
    this.root.querySelector('[role=tablist]')!.addEventListener('keydown', (ev) => {
      const e = ev as KeyboardEvent;
      if (e.key !== 'ArrowLeft' && e.key !== 'ArrowRight') return;
      e.preventDefault();
      e.stopPropagation();
      const i = TABS.indexOf(this.cur ?? this.last);
      const t = TABS[(i + (e.key === 'ArrowRight' ? 1 : TABS.length - 1)) % TABS.length];
      this.open(t);
      document.getElementById(`dt-${t}`)!.focus();
    });
  }

  get tab(): DockTab | null {
    return this.cur;
  }

  get lastTab(): DockTab {
    return this.last;
  }

  get isOpen(): boolean {
    return this.cur !== null;
  }

  open(tab: DockTab = this.last): void {
    if (this.cur === tab) return;
    this.cur = tab;
    this.last = tab;
    saveSetting('dockTab', tab);
    this.apply();
  }

  close(): void {
    if (!this.cur) return;
    this.cur = null;
    this.apply();
  }

  /** Opens the tab, or closes the dock when that tab is already showing. */
  toggle(tab: DockTab): void {
    if (this.cur === tab) this.close();
    else this.open(tab);
  }

  private apply(): void {
    const t = this.cur;
    this.root.hidden = !t;
    this.root.dataset.tab = t ?? '';
    for (const x of TABS) {
      const b = document.getElementById(`dt-${x}`)!;
      b.setAttribute('aria-selected', String(x === t));
      b.tabIndex = x === t ? 0 : -1;
      document.getElementById(`tab-${x}`)!.hidden = x !== t;
    }
    const app = document.getElementById('app')!;
    app.classList.toggle('dock-open', !!t);
    this.onChange?.(t);
  }
}
