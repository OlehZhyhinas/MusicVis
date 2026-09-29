// Onsets excite a damped spring; the exact solution keeps its timing independent
// of display frame rate. Shared by translation and tilt, once per body/frame.
import type { Schema } from '../genome';

export const RECOIL_SCHEMA: Schema = {
  source: { min: 0, max: 4, def: 0, choices: [0, 1, 2, 3, 4] },
  distance: { min: 0, max: 0.18, def: 0.06 },
  direction: { min: -0.5, max: 0.5, def: 0.25 },
  tilt: { min: -0.8, max: 0.8, def: 0.18 },
  frequency: { min: 0.6, max: 4, def: 1.6 },
  damping: { min: 0.12, max: 1, def: 0.3 },
  fan: { min: 0, max: 1, def: 0.35 },
};

/** One kick on a fresh onset, not one per frame of a decaying note-on pulse. */
export function recoilImpulse(level: number, previous: number): number {
  return level > 0.2 && (previous <= 0.2 || level - previous > 0.2) ? Math.min(1, level) : 0;
}

export function stepRecoil(x: number, v: number, dt: number, frequency: number, damping: number, impulse: number): { x: number; v: number } {
  if (!(dt > 0) || !Number.isFinite(dt)) return { x, v };
  const omega = 2 * Math.PI * frequency;
  const a = damping * omega;
  const b = omega * Math.sqrt(Math.max(0, 1 - damping * damping));
  // Bound repeated kicks so a dense drum roll cannot fling a body out of view.
  v = Math.max(-4 * omega, Math.min(4 * omega, v + impulse * omega * 1.7));
  const decay = Math.exp(-a * dt), c = Math.cos(b * dt);
  const s = b < 1e-6 ? dt : Math.sin(b * dt) / b;
  return { x: decay * (x * c + (v + a * x) * s), v: decay * (v * c - (a * v + omega * omega * x) * s) };
}
