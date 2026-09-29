import { recoilImpulse, stepRecoil } from '../src/v2/genes/recoil';
import { SEEDS } from '../src/v2/seeds';
import { cloneGenome, validate, repair } from '../src/v2/genome';
import { crossover, mutate, mulberry32 } from '../src/v2/ops';

type Check = (name: string, ok: boolean, detail: string) => void;
export function recoilTests(check: Check): void {
  const run = (fps: number, damping = 0.25) => {
    let state = { x: 0, v: 0 }, min = 0, peak = 0;
    for (let i = 0; i < fps * 3; i++) {
      state = stepRecoil(state.x, state.v, 1 / fps, 1.6, damping, i === 0 || i === fps / 2 ? 1 : 0);
      min = Math.min(min, state.x); peak = Math.max(peak, Math.abs(state.x));
    }
    return { ...state, min, peak };
  };
  const slow = run(30), fast = run(120);
  check('recoil.frame-rate', Math.abs(slow.x-fast.x) < 1e-9 && Math.abs(slow.v-fast.v) < 1e-9,
    'identically timed kicks settle to the same state at 30 and 120 FPS');
  check('recoil.overshoot', slow.min < -0.05 && Math.abs(slow.x) < slow.peak * 0.01,
    'a kick overshoots, then settles at its original position');
  const critical = run(60, 1);
  check('recoil.critical', critical.min >= 0 && Math.abs(critical.x) < 1e-7,
    'critical damping returns without crossing the rest position');
  let previous = 0, kicks = 0;
  for (const level of [0, 0.8, 0.8, 0.7, 0.5, 0.3, 0.1, 0.9, 0.8]) {
    if (recoilImpulse(level, previous)) kicks++;
    previous = level;
  }
  check('recoil.onset-edge', kicks === 2, 'sustained and decaying pulses do not retrigger each frame');
  let dense = { x: 0, v: 0 }, peak = 0;
  for (let i = 0; i < 6000; i++) {
    dense = stepRecoil(dense.x, dense.v, 1 / 60, 4, 0.12, 1);
    peak = Math.max(peak, Math.abs(Math.tanh(dense.x)));
  }
  const rested = stepRecoil(dense.x, dense.v, 60, 4, 0.12, 0);
  check('recoil.dense-and-gap', Number.isFinite(dense.x+dense.v) && peak <= 1 && Math.abs(rested.x) < 1e-10,
    'dense kicks stay bounded and a long gap settles without integration instability');
  const paused = stepRecoil(0.4, 0.2, 0, 1.6, 0.3, 1);
  check('recoil.zero-time', paused.x === 0.4 && paused.v === 0.2, 'a zero-time update cannot integrate a kick');

  const seeds = ['X63', 'X64', 'X65'].map(id => SEEDS.find(s => s.origin === id)!);
  check('recoil.three-presets', seeds.every(s => s && !validate(s.genome).length && s.genome.bodies.some(b => b.motion.kind === 'recoil')),
    'three distinct starter presets retain the new motion');
  const rng = mulberry32(6401); let retained = 0; const errors: string[] = [];
  for (let i = 0; i < 90; i++) {
    const g = crossover(cloneGenome(seeds[i % 3].genome), SEEDS[i].genome, rng);
    if (g.bodies.some(b => b.motion.kind === 'recoil')) retained++;
    for (const child of [g, mutate(g, rng)]) {
      errors.push(...validate(child));
      if (JSON.stringify(child) !== JSON.stringify(repair(child))) errors.push('repair changed valid child');
    }
  }
  check('recoil.inherits', retained > 15 && !errors.length, `${retained}/90 children retained recoil; ${errors.length} invalid children`);
}
