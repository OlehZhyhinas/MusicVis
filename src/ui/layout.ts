// Geometry shared by every floating layer: the dock on the right, the
// transport and preset bar centred in the space left of it, and the width
// classes (phone below 600 px, compact on laptops or next to a wide dock).

export interface LayoutState {
  /** Dock width in px when open (0 when closed or on phones, where it is a sheet). */
  dockW: number;
  phone: boolean;
  compact: boolean;
  /** Content width left of the dock. */
  cw: number;
  /** Transport width and left edge. */
  tw: number;
  tx: number;
}

const GAP = 16;

export function dockWidth(tab: string | null, W: number): number {
  if (!tab || W < 600) return 0;
  if (tab === 'presets') return W >= 1440 ? 660 : 580;
  return W >= 1440 ? 400 : 360;
}

export function computeLayout(dockTab: string | null): LayoutState {
  const W = window.innerWidth;
  const phone = W < 600;
  // Never let the dock take more than the window minus a usable canvas.
  const dockW = Math.min(dockWidth(dockTab, W), Math.max(0, W - 2 * GAP - 320));
  const cw = W - dockW - (dockW ? GAP : 0);
  const tw = phone ? W - 2 * GAP : Math.min(cw - 2 * GAP, 1120);
  const tx = phone ? GAP : GAP + (cw - 2 * GAP - tw) / 2;
  const compact = !phone && (W < 1400 || (dockW > 0 && tw < 980));
  return { dockW, phone, compact, cw, tw, tx };
}

export function applyLayout(app: HTMLElement, L: LayoutState): void {
  app.classList.toggle('phone', L.phone);
  app.classList.toggle('compact', L.compact);
  const s = app.style;
  s.setProperty('--dock-w', `${L.dockW}px`);
  s.setProperty('--cw', `${L.cw}px`);
  s.setProperty('--tw', `${L.tw}px`);
  s.setProperty('--tx', `${L.tx}px`);
}
