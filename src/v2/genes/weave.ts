// A closed figure-eight flight in musical time, with phased, counter-travelling
// copies. Analytic positions avoid integration drift and frame-rate dependence.
import type { Schema } from '../genome';

export const WEAVE_SCHEMA: Schema = {
  radius: { min: 0, max: 0.3, def: 0.12 },
  height: { min: 0, max: 0.24, def: 0.07 },
  period: { min: 1, max: 16, def: 4, choices: [1, 2, 4, 8, 16] },
  angle: { min: -0.5, max: 0.5, def: 0 },
  stagger: { min: 0, max: 0.5, def: 0.125 },
  bank: { min: -1.2, max: 1.2, def: 0.35 },
  counter: { min: 0, max: 1, def: 0, choices: [0, 1] },
};

export function weaveOffset(P: (k: string) => number, bars: number, copy: number): { x: number; y: number; tilt: number } {
  const direction = P('counter') > 0.5 && copy % 2 === 1 ? -1 : 1;
  const phase = 2*Math.PI*((bars/P('period')*direction + copy*P('stagger')) % 1);
  const x = P('radius')*Math.sin(phase), y = P('height')*Math.sin(2*phase);
  const angle = P('angle')*2*Math.PI, c = Math.cos(angle), s = Math.sin(angle);
  // A periodic bank, not a wrapped atan2 heading: fractional banking cannot snap
  // when the path crosses the negative axis or returns through the centre.
  return { x: c*x-s*y, y: s*x+c*y, tilt: P('bank')*Math.cos(phase)*direction };
}
