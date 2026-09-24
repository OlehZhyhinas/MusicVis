// Minimal toast notifications for errors and status messages.

let container: HTMLDivElement | null = null;

function ensureContainer(): HTMLDivElement {
  if (container) return container;
  container = document.createElement('div');
  container.className = 'toast-container';
  (document.getElementById('app') ?? document.body).appendChild(container);
  return container;
}

export type ToastKind = 'error' | 'info' | 'ok' | 'live' | 'evolve';

export function showToast(message: string, kind: ToastKind = 'info', durationMs = 5000): void {
  const root = ensureContainer();
  const el = document.createElement('div');
  el.className = `toast toast-${kind}`;
  el.textContent = message;
  root.appendChild(el);
  requestAnimationFrame(() => el.classList.add('toast-visible'));
  setTimeout(() => {
    el.classList.remove('toast-visible');
    setTimeout(() => el.remove(), 300);
  }, durationMs);
}
