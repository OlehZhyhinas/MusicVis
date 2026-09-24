// Popovers and menus: one open at a time, anchored above the control that
// opened it (a bottom sheet on phones). Click outside or Esc closes.

import { icon, type IconName } from './icons';

export type MenuItem =
  | string
  | {
      icon: IconName;
      label: string;
      kbd?: string;
      /** Shows a switch reflecting this state. */
      on?: () => boolean;
      run?: () => void;
      href?: string;
      bug?: boolean;
      /** Keep the menu open after running (switches). */
      keep?: boolean;
    };

const isPhone = () => document.getElementById('app')?.classList.contains('phone') ?? false;

export class Popovers {
  private cur: { el: HTMLElement; anchor: HTMLElement | null; onClose?: () => void } | null = null;
  onChange: (() => void) | null = null;

  constructor() {
    document.addEventListener('pointerdown', (ev) => {
      if (!this.cur) return;
      const t = ev.target as Node;
      if (this.cur.el.contains(t) || this.cur.anchor?.contains(t)) return;
      this.close();
    }, true);
    window.addEventListener('resize', () => this.place());
  }

  get openEl(): HTMLElement | null {
    return this.cur?.el ?? null;
  }

  isOpen(el?: HTMLElement): boolean {
    return el ? this.cur?.el === el : !!this.cur;
  }

  toggle(el: HTMLElement, anchor: HTMLElement | null, onClose?: () => void): boolean {
    if (this.cur?.el === el) {
      this.close();
      return false;
    }
    this.show(el, anchor, onClose);
    return true;
  }

  show(el: HTMLElement, anchor: HTMLElement | null, onClose?: () => void): void {
    if (this.cur && this.cur.el !== el) this.close();
    this.cur = { el, anchor, onClose };
    el.hidden = false;
    el.classList.remove('fade-in');
    void el.offsetWidth;
    el.classList.add('fade-in');
    anchor?.classList.add('on');
    anchor?.setAttribute('aria-expanded', 'true');
    this.place();
    this.onChange?.();
  }

  close(): boolean {
    const c = this.cur;
    if (!c) return false;
    this.cur = null;
    c.el.hidden = true;
    c.anchor?.classList.remove('on');
    c.anchor?.setAttribute('aria-expanded', 'false');
    c.onClose?.();
    this.onChange?.();
    return true;
  }

  /** Above the anchor's panel (transport / preset bar), right-aligned to the anchor. */
  place(): void {
    const c = this.cur;
    if (!c) return;
    const el = c.el;
    const a = c.anchor;
    const panel = a?.closest<HTMLElement>('.tp, .pbar, .dock, .first-run') ?? null;
    const inSheetDock = !!panel?.classList.contains('dock') && !!document.getElementById('app')?.classList.contains('dock-sheet');
    const phone = isPhone() || inSheetDock;
    el.classList.toggle('sheet', phone);
    if (phone) {
      el.style.left = el.style.right = el.style.bottom = el.style.top = '';
      return;
    }
    const W = window.innerWidth;
    const H = window.innerHeight;
    if (!a) {
      el.style.left = el.style.top = '';
      el.style.right = '16px';
      el.style.bottom = '16px';
      return;
    }
    const ar = a.getBoundingClientRect();
    const pr = (panel ?? a).getBoundingClientRect();
    el.style.top = '';
    el.style.left = '';
    const wide = el.dataset.align === 'panel' && panel?.classList.contains('tp');
    let right = W - (wide ? pr.right : ar.right);
    if (panel?.classList.contains('dock')) {
      // From inside the dock: open to its left, level with the anchor.
      el.style.right = `${W - pr.left + 12}px`;
      el.style.bottom = `${Math.max(16, H - ar.bottom)}px`;
      return;
    }
    const w = el.offsetWidth || 300;
    right = Math.max(16, Math.min(right, W - w - 16));
    el.style.right = `${right}px`;
    const above = panel?.classList.contains('first-run') ? ar.top : pr.top;
    el.style.bottom = `${Math.max(16, H - above + 8)}px`;
  }
}

export function renderMenu(el: HTMLElement, items: MenuItem[], close: () => void): void {
  el.textContent = '';
  if (isPhone()) {
    const g = document.createElement('div');
    g.className = 'grab';
    el.append(g);
  }
  for (const it of items) {
    if (typeof it === 'string') {
      const h = document.createElement('div');
      h.className = 'mh overline';
      h.textContent = it;
      el.append(h);
      continue;
    }
    const b = document.createElement(it.href ? 'a' : 'button') as HTMLAnchorElement | HTMLButtonElement;
    b.className = 'mi';
    b.setAttribute('role', 'menuitem');
    const on = it.on?.();
    b.innerHTML = `${icon(it.icon, 16)}<span class="lbl">${it.label}</span>${on !== undefined ? `<span class="tog${on ? ' is-on' : ''}" aria-hidden="true"></span>` : ''}${it.kbd ? `<span class="kbd">${it.kbd}</span>` : ''}${it.href ? icon('link', 13) : ''}`;
    if (on !== undefined) b.setAttribute('aria-checked', String(on));
    if (it.href && b instanceof HTMLAnchorElement) {
      b.href = it.href;
      b.target = '_blank';
      b.rel = 'noopener';
      if (it.bug) b.dataset.bug = '';
      b.addEventListener('click', () => close());
    } else {
      b.addEventListener('click', () => {
        if (!it.keep) close();
        it.run?.();
        if (it.keep) renderMenu(el, items, close);
      });
    }
    el.append(b);
  }
}
