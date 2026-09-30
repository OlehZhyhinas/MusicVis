// Live source rendering, replacement and resource teardown on the real WebGL device.
import { ensureServers, Tab } from './cdp';
const servers=await ensureServers(),tab=await Tab.open();
try {
  await tab.load();
  const result=await tab.eval<{checks:string[];errors:string[]}>(`(async()=>{
    const {Engine,Stage}=await import('/src/v2/engine.ts');
    const {SEEDS}=await import('/src/v2/seeds.ts');
    const {cloneGenome}=await import('/src/v2/genome.ts');
    const {SyntheticMusic}=await import('/src/v2/screen.ts');
    const eng=new Engine(document.createElement('canvas')); await eng.lilyAtlas.ready;
    const errors=[],checks=[],music=new SyntheticMusic();
    for(const id of ['X90','X91','X92']) {
      const g=cloneGenome(SEEDS.find(s=>s.origin===id).genome),st=new Stage(eng,{offscreen:true,flameCap:65536,particleCap:65536});st.resize(320,180);
      const programs=eng.cache.get(g,true);if(!programs)throw Error('source compile failed: '+eng.cache.failed(g));
      const slot=st.makeSlot(g,programs);st.slots=[slot];
      let first=null,delta=0;
      for(let i=0;i<120;i++) {
        st.render(music.next(1/60),1/60,'out');
        if(i===30||i===119){
          const source=slot.displays.get(0).stage,p=new Uint8Array(512*256*4);source.readPixels(p);
          if(!first)first=p;else delta=p.reduce((sum,v,j)=>sum+(j%4===3?0:Math.abs(v-first[j])),0);
        }
      }
      if(delta<1000)errors.push(id+' source does not animate');
      if(slot.displays.get(0).stage.frame!==120)errors.push(id+' source did not advance with main music');
      const old=slot.displays.get(0).stage;
      delete slot.genome.bodies[0].shape.display;st.render(music.next(1/60),1/60,'out');
      if(slot.displays.size || old.slots.length)errors.push(id+' source resources retained after removal');
      if(eng.gl.getError()!==eng.gl.NO_ERROR)errors.push(id+' WebGL error');
      checks.push(id+' live source delta '+delta);st.dispose();
    }
    return {checks,errors};
  })()`);
  console.log(JSON.stringify(result,null,2));
  const errors=tab.logs.filter(s=>/error|EXC/i.test(s));if(errors.length)console.error(errors.join('\n'));
  if(result.errors.length||errors.length)process.exitCode=1;
} finally {await tab.close();servers.stop();}
