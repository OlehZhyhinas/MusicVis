// Toasts: one style with an icon per kind, stacked top-right (left of the dock
// when it is open). Errors get a red edge and stay until dismissed.

import { icon, type IconName } from './icons';

export type ToastKind = 'error' | 'info' | 'ok' | 'live' | 'evolve';

const KIND: Record<ToastKind, [IconName, string]> = {
  info: ['info', 'var(--tx2)'],
  ok: ['check', 'var(--ok)'],
  error: ['alert', 'var(--neg)'],
  live: ['radio', 'var(--live)'],
  evolve: ['evolve', 'var(--acc)'],
};
const MAX = 4;

let container: HTMLDivElement | null = null;

function ensureContainer(): HTMLDivElement {
  if (container) return container;
  container = document.createElement('div');
  container.className = 'toasts';
  container.setAttribute('aria-live', 'polite');
  (document.getElementById('app') ?? document.body).appendChild(container);
  return container;
}

function dismiss(el: HTMLElement): void {
  if (el.classList.contains('out')) return;
  el.classList.add('out');
  setTimeout(() => el.remove(), 250);
}

/**
 * Shows a toast: a title, an optional detail line; errors stay until dismissed. A toast with an `id`
 * replaces the one before it with the same id (a readout that updates, like the lyric nudge).
 */
export function showToast(title: string, kind: ToastKind = 'info', durationMs = 5000, detail?: string, id?: string): void {
  const root = ensureContainer();
  if (id) for (const old of root.querySelectorAll<HTMLElement>('.toast')) if (old.dataset.id === id) old.remove();
  const el = document.createElement('div');
  if (id) el.dataset.id = id;
  el.className = `toast glass strong fade-in${kind === 'error' ? ' err' : ''}`;
  el.setAttribute('role', kind === 'error' ? 'alert' : 'status');
  const [ic, c] = KIND[kind];
  el.innerHTML = `<span class="ti" style="--c:${c}">${icon(ic, 16)}</span><div class="grow"><b></b>${detail ? '<p></p>' : ''}</div><button class="ib xs" aria-label="Dismiss" title="Dismiss">${icon('x', 14)}</button>`;
  el.querySelector('b')!.textContent = title;
  if (detail) el.querySelector('p')!.textContent = detail;
  el.querySelector('button')!.addEventListener('click', () => dismiss(el));
  root.prepend(el);
  const live = [...root.children].filter((x) => !x.classList.contains('out')) as HTMLElement[];
  for (const old of live.slice(MAX)) dismiss(old);
  if (kind !== 'error' && durationMs > 0) setTimeout(() => dismiss(el), durationMs);
}
