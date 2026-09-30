import { LightningSimulation, lightningBranch, type LightningParams, type LightningInput } from '../src/v2/genes/lightning';
import { SEEDS } from '../src/v2/seeds';
import { validate, repair, cloneGenome, EMIT_SCHEMAS, defaultParams, structuralKey } from '../src/v2/genome';
import { crossover,mutate,mulberry32 } from '../src/v2/ops';
type Check=(name:string,ok:boolean,detail:string)=>void;
export function lightningTests(check:Check):void {
  const p:LightningParams={channels:6,branches:1,jagged:1,speed:3,decay:0.2,width:0.004,spread:0.8,fan:0.5};
  const silent:LightningInput={hit:false,onsets:[0,0,0,0],noteOn:0,pitch:60,tonic:0};
  const sim=new LightningSimulation();for(let i=0;i<60;i++)sim.step(silent,p,1/60);
  check('lightning.silence',sim.strikes===0,'silence does not create free-running strikes');
  sim.step({...silent,noteOn:1},p,1/60);const first=sim.bolts[0];
  for(let i=0;i<8;i++)sim.step({...silent,noteOn:1},p,1/60);
  check('lightning.note-edge',sim.strikes===1 && first.hue===0,'a held onset launches only once, with pitch-class colour');
  sim.step(silent,p,1/60);sim.step({...silent,noteOn:1,pitch:64,hit:true,onsets:[0,1,1,1]},p,1/60);
  const hues=new Set(sim.bolts.map(b=>b.hue)),channels=new Set(sim.bolts.map(b=>b.channel));
  check('lightning.channels',sim.strikes===6 && hues.has(4/12) && channels.size>=4,'kick, note and three instrument onsets can strike independently; E differs from C');
  const branches=lightningBranch(123,2,p),spine=lightningBranch(123,2,{...p,branches:0});
  check('lightning.fractal',branches.length>spine.length*2 && branches.some(s=>s.weight<0.5) && branches.every(s=>Object.values(s).every(Number.isFinite))
    && JSON.stringify(branches)===JSON.stringify(lightningBranch(123,2,p)),`${branches.length} deterministic segments include secondary and tertiary branches`);
  for(let i=0;i<600;i++)sim.step({...silent,hit:true,noteOn:i%2,pitch:48+i%12},p,1/60);
  check('lightning.budget',sim.bolts.length<=p.channels*3,'dense rolls have a bounded live-strike count');
  for(let i=0;i<180;i++)sim.step(silent,p,1/60);
  check('lightning.decay',sim.bolts.length===0,'all strikes decay completely after music stops');
  const emitter={origins:[{x:0.2,y:-0.1,angle:0,scale:1,radius:0}],reach:0.5,sustain:1,held:1,legato:1};
  const arcs=new LightningSimulation();arcs.step({...silent,noteOn:1},p,1/60,emitter);
  const start=arcs.bolts[0].segments.find(s=>s.start===0)!;
  check('lightning.emission-anchor',Math.abs(start.ax-0.2)<1e-6 && Math.abs(start.ay+0.1)<1e-6,'strikes start at the host copy, not a full-screen lane');
  for(let i=0;i<180;i++)arcs.step(silent,p,1/60,emitter);
  check('lightning.emission-legato',arcs.bolts.length===1 && arcs.strikes===1,'one continuous legato arc survives a long held note without retriggering');
  for(let i=0;i<180;i++)arcs.step(silent,p,1/60,{...emitter,held:0});
  check('lightning.emission-release',arcs.bolts.length===0,'the sustained arc fades completely after release');
  const emitSeeds=SEEDS.filter(s=>s.genome.bodies.some(b=>b.emit.kind==='lightning'));
  check('lightning.emission-presets',emitSeeds.length===3 && emitSeeds.every(s=>!validate(s.genome).length),'three new host-body presets');
  const allKinds=new Map(SEEDS.flatMap(s=>s.genome.bodies.map(b=>[b.shape.kind,s.genome] as const)));
  let accepted=0;
  for(const [kind,source] of allKinds){
    const g=cloneGenome(source),bi=g.bodies.findIndex(b=>b.shape.kind===kind);
    g.bodies[bi].emit={kind:'none',p:{}};const old=structuralKey(g);g.bodies[bi].emit={kind:'lightning',p:defaultParams(EMIT_SCHEMAS.lightning)};
    const fixed=repair(g);
    if(fixed.bodies.some(b=>b.shape.kind===kind && b.emit.kind==='lightning') && !validate(fixed).length && structuralKey(fixed)!==old)accepted++;
  }
  check('lightning.emission-shapes',accepted===allKinds.size,`independent emission retained on ${accepted}/${allKinds.size} shape families`);
  let transferred=0;const erng=mulberry32(943);
  for(let i=0;i<100;i++){
    const child=crossover(emitSeeds[0].genome,SEEDS[i].genome,erng);
    if(child.bodies.some(b=>b.emit.kind==='lightning' && b.shape.kind!=='polygon'))transferred++;
  }
  check('lightning.emission-breeds',transferred>8,`${transferred}/100 children combine electrical emissions with a different body`);
  const seeds=SEEDS.filter(s=>s.genome.bodies.some(b=>b.shape.kind==='lightning'));
  check('lightning.presets',seeds.length>=3 && seeds.every(s=>!validate(s.genome).length),'three valid musical lightning arrangements');
  const rng=mulberry32(563),errors:string[]=[];let inherited=0;
  for(let i=0;i<80;i++){const child=crossover(seeds[0].genome,SEEDS[i].genome,rng);if(child.bodies.some(b=>b.shape.kind==='lightning'))inherited++;errors.push(...validate(child),...validate(mutate(child,rng)));}
  check('lightning.breeding',!errors.length && inherited>15,`${inherited}/80 offspring retained lightning; all mutations valid`);
}
