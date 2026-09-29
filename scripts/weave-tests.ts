import { WEAVE_SCHEMA, weaveOffset } from '../src/v2/genes/weave';
import { SEEDS } from '../src/v2/seeds';
import { cloneGenome, defaultParams, repair, validate } from '../src/v2/genome';
import { crossover, mutate, mulberry32 } from '../src/v2/ops';

type Check = (name: string, ok: boolean, detail: string) => void;
export function weaveTests(check: Check): void {
  const p = defaultParams(WEAVE_SCHEMA), P = (k: string) => p[k];
  const distance = (a: ReturnType<typeof weaveOffset>, b: ReturnType<typeof weaveOffset>) =>
    Math.hypot(a.x-b.x, a.y-b.y, a.tilt-b.tilt);
  let periodic = true, bounded = true, continuous = true;
  for (const period of WEAVE_SCHEMA.period.choices!) for (const counter of [0,1]) {
    p.period = period; p.counter = counter;
    for (let copy = 0; copy < 6; copy++) for (const time of [0,0.13,2.7,123456.25]) {
      const a = weaveOffset(P,time,copy), b = weaveOffset(P,time+period,copy);
      periodic &&= distance(a,b) < 1e-8;
      bounded &&= Math.hypot(a.x,a.y) <= Math.hypot(p.radius,p.height)+1e-9 && Math.abs(a.tilt) <= Math.abs(p.bank)+1e-9;
    }
    for (let copy = 0; copy < 6; copy++) {
      const seam = period*(1-copy*p.stagger);
      continuous &&= distance(weaveOffset(P,seam-1e-8,copy),weaveOffset(P,seam+1e-8,copy)) < 1e-6;
    }
  }
  check('weave.closed-and-bounded', periodic && bounded, 'all bar periods and copy directions close exactly without drifting beyond the path bounds');
  check('weave.continuous-bank', continuous, 'positions and banking remain continuous across phase wraps');
  p.counter=1; p.stagger=0;
  const a=weaveOffset(P,0.37,0), b=weaveOffset(P,0.37,1);
  check('weave.counter', Math.abs(a.x+b.x)+Math.abs(a.y+b.y)+Math.abs(a.tilt+b.tilt)<1e-9,
    'alternate copies traverse the same figure-eight in opposite directions');
  p.counter=0; p.angle=0;
  const original=weaveOffset(P,0.37,0); p.angle=0.25;
  const turned=weaveOffset(P,0.37,0);
  check('weave.path-angle', Math.abs(turned.x+original.y)+Math.abs(turned.y-original.x)<1e-9,
    'rotating the path changes its orientation without changing its scale');
  p.radius=0; p.height=0; p.bank=0;
  check('weave.zero', distance(weaveOffset(P,0.37,4),{x:0,y:0,tilt:0})===0, 'zero excursions and bank leave a copy at rest');

  const seeds=['X72','X73','X74'].map(id=>SEEDS.find(s=>s.origin===id)!);
  check('weave.three-presets',seeds.every(s=>s && !validate(s.genome).length && s.genome.bodies.some(b=>b.motion.kind==='weave')),
    'three distinct starter presets retain the new movement');
  const rng=mulberry32(7201); let retained=0; const errors: string[]=[];
  for(let i=0;i<90;i++) {
    const g=crossover(cloneGenome(seeds[i%3].genome),SEEDS[i].genome,rng);
    if(g.bodies.some(b=>b.motion.kind==='weave')) retained++;
    for(const child of [g,mutate(g,rng)]) {
      errors.push(...validate(child));
      if(JSON.stringify(child)!==JSON.stringify(repair(child))) errors.push('repair changed valid child');
    }
  }
  check('weave.inherits',retained>15 && !errors.length,`${retained}/90 children retained weave; ${errors.length} invalid children`);
}
