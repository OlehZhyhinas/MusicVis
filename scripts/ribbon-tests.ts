import { RibbonRipples } from '../src/v2/genes/ribbonRipples';
import { repairRibbonPath, sampleRibbon, parseRibbonPath } from '../src/v2/genes/ribbon';
import { RibbonSimulation, RIBBON_STEPS, RIBBON_ACROSS, type RibbonDrive } from '../src/v2/genes/ribbonSim';
import { SEEDS } from '../src/v2/seeds';
import { cloneGenome, repair, validate, structuralKey } from '../src/v2/genome';
import { editRibbonPath } from '../src/v2/geneEdit';
import { crossover, mutate, mulberry32 } from '../src/v2/ops';
type Check=(name:string,ok:boolean,detail:string)=>void;
export function ribbonTests(check:Check):void {
  const seeds=SEEDS.filter(s=>s.genome.bodies.some(b=>b.shape.kind==='ribbon'));
  check('ribbon.presets',seeds.length>=3 && seeds.every(s=>!validate(s.genome).length),'three valid freeform mesh presets');
  const rain=new RibbonRipples();
  const impact={hit:false,onsets:[1,0,0,0],noteOn:1,pitch:64,height:0.4};
  for(let i=0;i<30;i++)rain.step(impact,1/60);
  check('ribbon.ripple-edge',rain.impacts===2 && rain.drops.length===2,'held onset envelopes trigger once, with independent drum/note colours');
  for(let i=0;i<80;i++)rain.step({...impact,noteOn:i%2,onsets:[i%2,1-i%2,0,0]},1/60);
  check('ribbon.ripple-cap',rain.drops.length<=8 && rain.uniforms.every(Number.isFinite),'dense accents remain bounded');
  for(let i=0;i<160;i++)rain.step({...impact,noteOn:0,onsets:[0,0,0,0]},1/60);
  check('ribbon.ripple-decay',rain.drops.length===0,'waves expire during silence');
  check('ribbon.new-dark-presets',['X87','X88','X89'].every(id=>seeds.find(s=>s.origin===id)?.genome.bodies[0].shape.p.ink===1)
    && ['X81','X82','X83'].every(id=>seeds.find(s=>s.origin===id)?.genome.bodies[0].shape.p.ink===0),'three new dark presets leave original ribbon surfaces intact');
  const path=parseRibbonPath('-1 0 0 1 -2\n0 0.5 1 0.5 3\n1 -0.5 -1 1.5 0');
  const midpoint=sampleRibbon(path,0.5,false);
  check('ribbon.path',midpoint.z===1 && midpoint.twist===3 && midpoint.width===0.5 && repairRibbonPath(path).length===3,'arbitrary depth, local width and multi-turn twist survive interpolation');
  const before=seeds[0].genome, edited=editRibbonPath(before,0,'-1 0 1 1 0\n0 1 -1 0.5 2\n1 0 0 1 -2');
  check('ribbon.edit',edited.ok && edited.genome.bodies[0].shape.path?.[1].z===-1 && structuralKey(before)===structuralKey(edited.genome)
    && !editRibbonPath(before,0,'1 nope 2').ok && JSON.stringify(repair(edited.genome))===JSON.stringify(edited.genome), 'path edits retain genes, reject bad input and serialize without shader recompilation');
  const drive:RibbonDrive={wind:0,stiffness:0.5,twist:1,width:0.12,speed:1,closed:false,bass:0,vocals:0,onset:0};
  const quiet=new RibbonSimulation(),loud=new RibbonSimulation();
  for(let i=0;i<600;i++){quiet.step(path,drive,1/60);loud.step(path,{...drive,wind:1,bass:1,vocals:1,onset:i%30===0?1:0},1/60);}
  const delta=loud.position.reduce((sum,v,i)=>sum+Math.abs(v-quiet.position[i]),0)/loud.position.length;
  check('ribbon.simulation',delta>0.01 && loud.vertices.every(Number.isFinite) && loud.position.every(v=>Math.abs(v)<4),`musical forces displace the 3D spring chain (${delta.toFixed(3)}); finite and bounded after 10 seconds`);
  const loop=new RibbonSimulation();
  const knot=seeds.find(s=>s.origin==='X82')!.genome.bodies[0].shape.path!;
  loop.step(knot,{...drive,closed:true,wind:0.8,twist:0.43},1/60);
  let seam=0;const stride=(RIBBON_ACROSS+1)*8;
  for(let j=0;j<=RIBBON_ACROSS;j++) for(let k=0;k<6;k++) seam=Math.max(seam,Math.abs(loop.vertices[j*8+k]-loop.vertices[RIBBON_STEPS*stride+j*8+k]));
  check('ribbon.closed-seam',seam<1e-4,`closed surface joins positions and normals (max seam ${seam.toExponential(2)})`);
  const degenerate=new RibbonSimulation();degenerate.step(repairRibbonPath([{x:0},{x:0}]),drive,0.05);
  check('ribbon.degenerate',degenerate.vertices.every(Number.isFinite),'coincident control points retain a finite frame');
  const rng=mulberry32(983),errors:string[]=[];let inherited=0;
  for(let i=0;i<80;i++) {
    const child=crossover(before,SEEDS[i%SEEDS.length].genome,rng), changed=mutate(cloneGenome(child),rng);
    errors.push(...validate(child),...validate(changed));if(child.bodies.some(b=>b.shape.kind==='ribbon' && b.shape.path?.length))inherited++;
  }
  check('ribbon.breeding',!errors.length && inherited>15,`${inherited}/80 offspring retain editable paths; ${errors.slice(0,2).join(';')||'all mutations valid'}`);
}
