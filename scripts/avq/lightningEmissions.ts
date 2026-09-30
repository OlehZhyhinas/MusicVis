// Compile/render independent electrical emissions across every existing host family.
import { ensureServers, Tab } from './cdp';
const servers=await ensureServers(),tab=await Tab.open();
try {
  await tab.load();
  const result=await tab.eval<{checked:string[];errors:string[]}>(`(async()=>{
    const {Engine,Stage}=await import('/src/v2/engine.ts');
    const {SEEDS}=await import('/src/v2/seeds.ts');
    const {cloneGenome,repair,EMIT_SCHEMAS,defaultParams}=await import('/src/v2/genome.ts');
    const {SyntheticMusic}=await import('/src/v2/screen.ts');
    const eng=new Engine(document.createElement('canvas'));await eng.lilyAtlas.ready;
    const errors=[],checked=[],families=new Map();
    SEEDS.forEach(s=>s.genome.bodies.forEach(b=>{if(!families.has(b.shape.kind))families.set(b.shape.kind,s.genome)}));
    for(const [kind,source] of families){
      const g=cloneGenome(source),bi=g.bodies.findIndex(b=>b.shape.kind===kind);
      g.bodies[bi].emit={kind:'lightning',p:defaultParams(EMIT_SCHEMAS.lightning)};
      const fixed=repair(g),index=fixed.bodies.findIndex(b=>b.shape.kind===kind && b.emit.kind==='lightning');
      if(index<0){errors.push(kind+' lost emission');continue;}
      const programs=eng.cache.get(fixed,true);if(!programs)throw Error(kind+' failed '+eng.cache.failed(fixed));
      if(!programs.composite.loc('uArcs'+index))errors.push(kind+' emission sampler missing');
      const stage=new Stage(eng,{offscreen:true,particleCap:65536,flameCap:65536});stage.resize(240,135);
      const slot=stage.makeSlot(fixed,programs);stage.slots=[slot];const music=new SyntheticMusic();
      for(let i=0;i<24;i++)stage.render(music.next(1/60),1/60,'out');
      const arc=slot.arcs.get(index);
      if(!arc?.texture || !arc.simulation.strikes)errors.push(kind+' no emitted strikes');
      if(eng.gl.getError()!==eng.gl.NO_ERROR)errors.push(kind+' WebGL error');
      checked.push(kind);stage.dispose();
      if(slot.arcs.size)errors.push(kind+' retained emissions after disposal');
    }
    return {checked,errors};
  })()`);
  console.log(JSON.stringify(result,null,2));
  const errors=tab.logs.filter(s=>/error|EXC/i.test(s));if(errors.length)console.error(errors.join('\n'));
  if(result.errors.length||errors.length)process.exitCode=1;
}finally{await tab.close();servers.stop();}
