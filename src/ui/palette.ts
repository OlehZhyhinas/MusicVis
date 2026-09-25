// One overlay, two tabs. "/" opens Commands: every action with its shortcut,
// searchable; typing a preset ID (G3-…, or a bare seed ID like E07) turns it
// into Go to preset, and a miss shows the error inline. "?" opens Shortcuts:
// the keyboard map grouped by task.

import { icon, type IconName } from './icons';

export interface Command {
  group: string;
  icon: IconName;
  label: string;
  keys?: string[];
  run?: () => void;
  href?: string;
  bug?: boolean;
  /** Keep the overlay open after running (commands that switch its mode). */
  stay?: boolean;
}

export interface PresetHit {
  id: string;
  name: string;
  hint: string;
}

export interface PaletteDeps {
  commands: () => Command[];
  /** Presets whose ID matches the typed text. */
  findPresets: (q: string) => PresetHit[];
  gotoPreset: (id: string) => boolean;
  thumb: (id: string) => Promise<string>;
}

type Mode = 'cmd' | 'goto' | 'help';

const SHORTCUTS: [string, [string[], string][]][] = [
  ['Playback', [[['Space'], 'Play / pause'], [['←', '→'], 'Seek -5 s / +5 s'], [['Shift', '←', '→'], 'Previous / next track'], [['S'], 'Shuffle on / off'], [['M'], 'Mute'], [['F'], 'Fullscreen'], [['[', ']'], 'Lyrics 0.25 s earlier / later']]],
  ['Presets', [[['L'], 'Like'], [['D'], 'Dislike'], [['N'], 'Next preset'], [['G'], 'Go to preset by ID'], [['E'], 'Evolve mode on / off']]],
  ['Panels', [[['P'], 'Playlist'], [['B'], 'Preset browser'], [['K'], 'Gene editor (opens the HUD too)'], [['H'], 'HUD'], [['/'], 'All commands'], [['?'], 'This help'], [['Esc'], 'Close the top panel']]],
];

const esc = (s: string) => s.replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c]!);
const kbd = (k: string) => `<span class="kbd">${esc(k)}</span>`;
/** Looks like a preset ID: G3-…, G12…, or a bare seed ID (E07, M01). */
const looksLikeId = (q: string) => /^(g\d|[em]\d)/i.test(q.trim());

export class Palette {
  private wrap = document.getElementById('overlay')!;
  private box = document.getElementById('ov-box')!;
  private input = document.getElementById('cmdq') as HTMLInputElement;
  private listEl = document.getElementById('cmdlist')!;
  private cmdView = document.getElementById('ov-cmd')!;
  private helpView = document.getElementById('ov-help')!;
  private hints = document.getElementById('ov-hints')!;
  private mode: Mode = 'cmd';
  private hl = 0;
  private items: { el: HTMLElement; run: () => void }[] = [];
  private returnFocus: HTMLElement | null = null;

  constructor(private deps: PaletteDeps) {
    this.helpView.querySelector('.keys')!.innerHTML = SHORTCUTS.map(
      ([h, rows]) => `<div><h4 class="overline">${h}</h4><dl>${rows.map(([k, d]) => `<dt>${k.map(kbd).join('')}</dt><dd>${d}</dd>`).join('')}</dl></div>`,
    ).join('');
    document.getElementById('ov-close')!.addEventListener('click', () => this.close());
    this.wrap.querySelector('.scrim')!.addEventListener('click', () => this.close());
    for (const b of this.box.querySelectorAll<HTMLButtonElement>('.ov-head [data-v]')) {
      b.addEventListener('click', () => this.open(b.dataset.v as Mode));
    }
    this.input.addEventListener('input', () => {
      this.hl = 0;
      this.renderList();
    });
    this.box.addEventListener('keydown', (ev) => this.onKey(ev));
    this.listEl.addEventListener('mousemove', (ev) => {
      const el = (ev.target as HTMLElement).closest<HTMLElement>('.cmd');
      const i = this.items.findIndex((x) => x.el === el);
      if (i >= 0 && i !== this.hl) this.highlight(i);
    });
  }

  get isOpen(): boolean {
    return !this.wrap.hidden;
  }

  get currentMode(): Mode | null {
    return this.isOpen ? this.mode : null;
  }

  toggle(mode: Mode): void {
    if (this.isOpen && (this.mode === mode || (mode === 'cmd' && this.mode === 'goto'))) this.close();
    else this.open(mode);
  }

  open(mode: Mode): void {
    if (!this.isOpen) this.returnFocus = document.activeElement as HTMLElement | null;
    const wasCmd = this.isOpen && this.mode !== 'help';
    this.mode = mode;
    this.wrap.hidden = false;
    const isHelp = mode === 'help';
    this.cmdView.hidden = isHelp;
    this.helpView.hidden = !isHelp;
    this.hints.hidden = isHelp;
    this.box.classList.toggle('wide', isHelp);
    this.box.setAttribute('aria-label', isHelp ? 'Keyboard shortcuts' : 'Commands');
    for (const b of this.box.querySelectorAll<HTMLButtonElement>('.ov-head [data-v]')) {
      b.setAttribute('aria-pressed', String((b.dataset.v === 'help') === isHelp));
    }
    if (!isHelp) {
      if (mode === 'goto') {
        this.input.value = '';
        this.input.placeholder = 'Preset ID, e.g. G0-E07 or G3-0012';
      } else {
        if (!wasCmd) this.input.value = '';
        this.input.placeholder = 'Type a command, or a preset ID like G3-0142';
      }
      this.hl = 0;
      this.renderList();
      this.input.focus();
    } else {
      (this.box.querySelector('.ov-head [data-v=help]') as HTMLElement).focus();
    }
  }

  close(): void {
    if (!this.isOpen) return;
    this.wrap.hidden = true;
    this.input.blur();
    const r = this.returnFocus;
    this.returnFocus = null;
    if (r && document.contains(r)) r.focus({ preventScroll: true });
  }

  private onKey(ev: KeyboardEvent): void {
    // Keys inside the overlay never reach the page shortcuts.
    ev.stopPropagation();
    if (ev.key === 'Escape') {
      ev.preventDefault();
      this.close();
      return;
    }
    if (ev.key === 'Tab') {
      // Keep focus inside the dialog.
      const f = [...this.box.querySelectorAll<HTMLElement>('button, a[href], input')].filter((x) => x.offsetParent !== null);
      const i = f.indexOf(document.activeElement as HTMLElement);
      const next = ev.shiftKey ? (i <= 0 ? f.length - 1 : i - 1) : i === f.length - 1 ? 0 : i + 1;
      ev.preventDefault();
      f[next]?.focus();
      return;
    }
    if (this.mode === 'help') {
      if (ev.key === '/' ) {
        ev.preventDefault();
        this.open('cmd');
      } else if (ev.key === '?') this.close();
      return;
    }
    if (ev.key === 'ArrowDown' || ev.key === 'ArrowUp') {
      ev.preventDefault();
      if (!this.items.length) return;
      this.highlight((this.hl + (ev.key === 'ArrowDown' ? 1 : this.items.length - 1)) % this.items.length);
    } else if (ev.key === 'Enter' && ev.target === this.input) {
      ev.preventDefault();
      this.items[this.hl]?.run();
    }
  }

  private highlight(i: number): void {
    this.items[this.hl]?.el.classList.remove('hl');
    this.hl = i;
    const el = this.items[i]?.el;
    if (!el) return;
    el.classList.add('hl');
    el.scrollIntoView({ block: 'nearest' });
    this.input.setAttribute('aria-activedescendant', el.id);
  }

  private renderList(): void {
    const q = this.input.value.trim();
    this.items = [];
    this.listEl.textContent = '';
    let n = 0;
    const add = (el: HTMLElement, run: () => void) => {
      el.id = `cmd-${n++}`;
      el.setAttribute('role', 'option');
      this.items.push({ el, run });
      this.listEl.append(el);
    };
    const head = (t: string) => {
      const h = document.createElement('div');
      h.className = 'mh overline';
      h.textContent = t;
      this.listEl.append(h);
    };
    if (this.mode === 'goto' || looksLikeId(q)) {
      head('Go to preset');
      const hits = q ? this.deps.findPresets(q) : [];
      for (const p of hits) {
        const b = document.createElement('button');
        b.className = 'cmd';
        b.innerHTML = `<img class="th" alt="" width="32" height="20" /><span class="mono cmd-id">${esc(p.id)}</span><span class="lbl ell">${esc(p.name)}</span><span class="hint">${esc(p.hint)}</span>${kbd('↵')}`;
        const img = b.querySelector('img')!;
        void this.deps.thumb(p.id).then((url) => {
          if (url) img.src = url;
        });
        const run = () => {
          this.close();
          this.deps.gotoPreset(p.id);
        };
        b.addEventListener('click', run);
        add(b, run);
      }
      if (q && !hits.length) {
        const e = document.createElement('div');
        e.className = 'banner';
        e.setAttribute('role', 'alert');
        e.style.setProperty('--c', 'var(--neg)');
        e.innerHTML = `${icon('alert', 14)}<span class="sp"></span>`;
        e.querySelector('.sp')!.textContent = `No preset “${q}” in the population.`;
        this.listEl.append(e);
      }
      const p = document.createElement('p');
      p.className = 'dim cmd-note';
      p.textContent = 'IDs look like G0-E07 or G3-0012. A bare seed ID (E07) means G0-E07.';
      this.listEl.append(p);
    } else {
      const ql = q.toLowerCase();
      let group = '';
      for (const c of this.deps.commands()) {
        if (ql && !c.label.toLowerCase().includes(ql) && !c.group.toLowerCase().includes(ql)) continue;
        if (c.group !== group) {
          group = c.group;
          head(group);
        }
        const el = document.createElement(c.href ? 'a' : 'button') as HTMLButtonElement | HTMLAnchorElement;
        el.className = 'cmd';
        el.innerHTML = `${icon(c.icon, 16)}<span class="lbl"></span>${c.href ? icon('link', 13) : (c.keys ?? []).map(kbd).join('')}`;
        el.querySelector('.lbl')!.textContent = c.label;
        let run: () => void;
        if (c.href && el instanceof HTMLAnchorElement) {
          el.href = c.href;
          el.target = '_blank';
          el.rel = 'noopener';
          if (c.bug) el.dataset.bug = '';
          run = () => el.click();
          el.addEventListener('click', () => this.close());
        } else {
          run = () => {
            if (!c.stay) this.close();
            c.run?.();
          };
          el.addEventListener('click', run);
        }
        add(el, run);
      }
      if (!this.items.length) {
        const p = document.createElement('p');
        p.className = 'dim cmd-empty';
        p.textContent = `No command matches “${q}”.`;
        this.listEl.append(p);
      }
    }
    if (this.items.length) this.highlight(Math.min(this.hl, this.items.length - 1));
    else this.input.removeAttribute('aria-activedescendant');
  }
}
