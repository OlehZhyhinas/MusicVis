import { FerrofluidSimulation, FERROFLUID_SCHEMA } from '../src/v2/genes/ferrofluid';
import { OrigamiSimulation, ORIGAMI_SCHEMA } from '../src/v2/genes/origami';
import { FractureSimulation, FRACTURE_SCHEMA, type SculptureMusic } from '../src/v2/genes/fracture';
import { MESH_STRIDE } from '../src/v2/genes/meshGeometry';
import { defaultParams, validate, repair } from '../src/v2/genome';
import { SEEDS } from '../src/v2/seeds';
type Check=(name:string,ok:boolean,detail:string)=>void;
export function sculptureTests(check:Check):void {
  const p={...defaultParams(FRACTURE_SCHEMA),open:0,freedom:0,drift:0};
  const quiet:SculptureMusic={hit:false,onset:0,noteOn:0,held:0,legato:0,bass:0,vocals:0,pitch:60};
  const sim=new FractureSimulation();const rest=sim.step(p,quiet,1/60);
  sim.step(p,{...quiet,hit:true,onset:1},1/60);for(let i=0;i<30;i++)sim.step(p,quiet,1/60);
  const distance=(a:number[],b:number[])=>Math.hypot(...a.map((v,i)=>v-b[i]));
  const opened=Math.max(...sim.shards.map(s=>distance(s.position,s.center)));
  check('fracture.impulse',opened>0.02 && sim.shards.length>80,'a drum impulse separates individual solid shards');
  for(let i=0;i<500;i++)sim.step(p,quiet,1/60);
  const returned=Math.max(...sim.shards.map(s=>distance(s.position,s.center)));
  check('fracture.reassemble',returned<0.002,'shards return to their original slab after the impact');
  const mesh=sim.step(p,quiet,0.8);let normalError=0;
  for(let i=0;i<mesh.length;i+=MESH_STRIDE)normalError=Math.max(normalError,Math.abs(Math.hypot(mesh[i+3],mesh[i+4],mesh[i+5])-1));
  check('fracture.solid-mesh',mesh.every(Number.isFinite)&&normalError<0.001&&mesh.length===rest.length,'closed triangular prisms have finite positions and unit face normals');
  const freeParams={...p,freedom:1,drift:0.7},free=new FractureSimulation();
  free.step(freeParams,quiet,1/60);const start=free.shards.map(s=>[...s.position]);
  for(let i=0;i<360;i++)free.step(freeParams,{...quiet,hit:i%45===0,onset:i%45===0?1:0},1/60);
  let rearranged=0,reordered=0;
  for(let i=1;i<start.length;i++){
    if(Math.abs(distance(start[i],start[i-1])-distance(free.shards[i].position,free.shards[i-1].position))>0.15)rearranged++;
    if((start[i][0]-start[i-1][0])*(free.shards[i].position[0]-free.shards[i-1].position[0])<0)reordered++;
  }
  check('fracture.independent-pieces',rearranged>start.length*0.5&&reordered>start.length*0.2,`${rearranged} neighbour distances change; ${reordered} pairs exchange horizontal order`);
  const heldGate=new FractureSimulation(),singleGate=new FractureSimulation();
  for(let i=0;i<120;i++){
    heldGate.step(freeParams,{...quiet,hit:true,onset:i===0?1:0},1/60);
    singleGate.step(freeParams,{...quiet,hit:i===0,onset:i===0?1:0},1/60);
  }
  check('fracture.single-impulse',heldGate.shards.every((s,i)=>distance(s.position,singleGate.shards[i].position)<1e-6),'a sustained drum gate delivers one impulse per shard');
  const gathered=new FractureSimulation(),loose=new FractureSimulation(),still={...freeParams,drift:0,impact:0};
  for(let i=0;i<360;i++){
    gathered.step(still,{...quiet,held:1,legato:1},1/60);loose.step(still,quiet,1/60);
  }
  const radius=(simulation:FractureSimulation)=>simulation.shards.reduce((sum,s)=>sum+Math.hypot(...s.position),0)/simulation.shards.length;
  check('fracture.legato-gather',radius(gathered)<radius(loose)*0.7,'held legato gathers the free cloud without restoring fixed shard slots');
  for(let i=0;i<600;i++)free.step({...freeParams,impact:1,recovery:4,tumble:1},{...quiet,hit:i%2===0,onset:i%2===0?1:0},i%13===0?0.8:1/60);
  check('fracture.stable-rotations',free.shards.every(s=>s.position.every(Number.isFinite)&&Math.hypot(...s.position)<=1.651&&Math.abs(Math.hypot(...s.orientation)-1)<1e-6),'dense impacts and slow frames keep free positions bounded and rotations normalized');
  for(const form of [0,1,2]) {
    const paper=new OrigamiSimulation(),params={...defaultParams(ORIGAMI_SCHEMA),form};
    let vertices=new Float32Array() as Float32Array;
    for(let i=0;i<240;i++)vertices=paper.step(params,{...quiet,hit:i%40===0,held:i>120?1:0,legato:1},1/60);
    let error=0;
    for(const face of paper.faces)for(let i=0;i<face.length;i++)for(let j=i+1;j<face.length;j++){
      const a=face[i],b=face[j],distance=(points:number[][])=>Math.hypot(...points[a].map((v,k)=>v-points[b][k]));
      error=Math.max(error,Math.abs(distance(paper.rest)-distance(paper.points)));
    }
    check('origami.rigid-panels-'+form,error<1e-6&&vertices.every(Number.isFinite),`all panel edges and diagonals keep their length while hinges move; max error ${error.toExponential(2)}`);
  }
  for(const form of [0,1,2]) {
    const liquid=new FerrofluidSimulation(),params={...defaultParams(FERROFLUID_SCHEMA),form};
    liquid.step(params,quiet,1/60);const rest=Array.from(liquid.drops);
    for(let i=0;i<180;i++)liquid.step(params,{...quiet,hit:i%30===0,onset:i%30===0?1:0,bass:0.8,pitch:72},1/60);
    const distance=Math.max(...liquid.drops.map((v,i)=>Math.abs(v-rest[i]))),field=liquid.field;
    for(let i=0;i<600;i++)liquid.step({...params,drift:0},quiet,i%20===0?0.8:1/60);
    check('ferrofluid.music-'+form,distance>0.05&&field>0.3,'musical forces move the liquid poles and excite the magnetic field');
    check('ferrofluid.stable-'+form,liquid.drops.every(Number.isFinite)&&Math.max(...liquid.drops.map(Math.abs))<2&&liquid.velocity.every(v=>Math.abs(v)<0.001)&&liquid.field<0.001,'finite bounded droplets settle after music stops, including slow frames');
  }
  const liquidSeeds=SEEDS.filter(s=>s.genome.bodies.some(b=>b.shape.kind==='ferrofluid'));
  check('ferrofluid.presets',liquidSeeds.length===3&&liquidSeeds.every(s=>!validate(s.genome).length&&JSON.stringify(repair(s.genome))===JSON.stringify(s.genome)&&s.genome.carrier.kind==='none'&&!s.genome.chain.length),'three stable liquid forms with no feedback');
  const paperSeeds=SEEDS.filter(s=>s.genome.bodies.some(b=>b.shape.kind==='origami'));
  check('origami.presets',paperSeeds.length===3&&paperSeeds.every(s=>!validate(s.genome).length&&s.genome.carrier.kind==='none'&&!s.genome.chain.length),'three connected-paper forms without feedback');
  const presets=SEEDS.filter(s=>s.genome.bodies.some(b=>b.shape.kind==='fracture'));
  check('fracture.presets',presets.length===3&&presets.every(s=>!validate(s.genome).length&&JSON.stringify(repair(s.genome))===JSON.stringify(s.genome)&&s.genome.carrier.kind==='none'&&!s.genome.chain.length),'three independently lit, serializable presets with no tunnel or feedback');
}
