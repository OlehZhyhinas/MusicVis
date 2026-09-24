// Geometry shared by every floating layer: the dock on the right, the
// transport and preset bar centred in the space left of it, and the width
// classes (phone below 600 px, compact on laptops or next to a wide dock).

export interface LayoutState {
  /** Dock width in px when open (0 when closed or when it is a bottom sheet). */
  dockW: number;
  phone: boolean;
  /** The dock opens as a bottom sheet (phones, and windows too narrow to give it a column). */
  sheet: boolean;
  compact: boolean;
  /** Transport too narrow for the extras (mic, fullscreen, score, next preset). */
  narrow: boolean;
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
  // The dock never squeezes the canvas below ~600 px; when even a slim dock would, it is a sheet.
  let dockW = Math.min(dockWidth(dockTab, W), W - GAP - 600);
  const sheet = phone || W - GAP - 600 < 340;
  if (sheet || dockW < 340) dockW = 0;
  const cw = W - dockW - (dockW ? GAP : 0);
  const tw = phone ? W - 2 * GAP : Math.min(cw - 2 * GAP, 1120);
  const tx = phone ? GAP : GAP + (cw - 2 * GAP - tw) / 2;
  const compact = !phone && (W < 1400 || (dockW > 0 && tw < 980));
  const narrow = !phone && tw < 760;
  return { dockW, phone, sheet, compact, narrow, cw, tw, tx };
}

export function applyLayout(app: HTMLElement, L: LayoutState): void {
  app.classList.toggle('phone', L.phone);
  app.classList.toggle('compact', L.compact);
  app.classList.toggle('narrow', L.narrow);
  app.classList.toggle('dock-sheet', L.sheet);
  const s = app.style;
  s.setProperty('--dock-w', `${L.dockW}px`);
  s.setProperty('--dock-gap', L.dockW ? `${GAP}px` : '0px');
  s.setProperty('--cw', `${L.cw}px`);
  s.setProperty('--tw', `${L.tw}px`);
  s.setProperty('--tx', `${L.tx}px`);
}
