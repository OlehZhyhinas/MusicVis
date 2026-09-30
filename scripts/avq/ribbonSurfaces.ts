// Real mesh/depth rendering, materials and musical spring forces.
import { ensureServers, Tab } from './cdp';
const servers=await ensureServers(),tab=await Tab.open();
try {
  await tab.load();
  const result=await tab.eval<{checked:string[];errors:string[]}>(`(async()=>{
    const {Engine,Stage}=await import('/src/v2/engine.ts');
    const {SEEDS}=await import('/src/v2/seeds.ts');
    const {cloneGenome,MATERIAL_KINDS,MATERIAL_SCHEMAS,defaultParams}=await import('/src/v2/genome.ts');
    const {ReferenceClip}=await import('/src/v2/fingerprint.ts');
    const eng=new Engine(document.createElement('canvas')),errors=[],checked=[];
    async function render(g,level,frames=90){
      const st=new Stage(eng,{offscreen:true,flameCap:65536,particleCap:65536});st.resize(320,180);
      const progs=eng.cache.get(g,true);if(!progs)throw Error('compile failed');
      const slot=st.makeSlot(g,progs);st.slots=[slot];const clip=new ReferenceClip();
      for(let i=0;i<frames;i++){
        const state=clip.next();state.stems.bass=level;state.stems.vocals=level;
        st.render(state,1/60,'out');
      }
      const pixels=new Uint8Array(320*180*4);st.readPixels(pixels);
      const positions=slot.ribbons.get(0).simulation.position.slice();
      if(eng.gl.getError()!==eng.gl.NO_ERROR)throw Error('WebGL error');
      st.disposeSlot(slot);st.dispose();return {pixels,positions};
    }
    for(const id of ['X81','X82','X83']){
      const g=cloneGenome(SEEDS.find(s=>s.origin===id).genome);g.reactions=[];
      for(const material of MATERIAL_KINDS){
        const m=cloneGenome(g);m.bodies[0].material={kind:material,p:{...defaultParams(MATERIAL_SCHEMAS[material]),gain:1.5}};
        const r=await render(m,0.7,12);
        if(!r.pixels.some((v,i)=>i%4!==3 && v>12))errors.push(id+'/'+material+' blank');
        checked.push(id+'/'+material);
      }
      const a=await render(g,0),b=await render(g,1);
      const motion=b.positions.reduce((sum,v,i)=>sum+Math.abs(v-a.positions[i]),0)/b.positions.length;
      if(motion<0.001)errors.push(id+' music did not move 3D vertices');
      checked.push(id+' music displacement '+motion.toFixed(4));
    }
    return {checked,errors};
  })()`);
  console.log(JSON.stringify(result,null,2));
  const errors=tab.logs.filter(s=>/error|EXC/i.test(s));if(errors.length)console.error(errors.join('\n'));
  if(result.errors.length||errors.length)process.exitCode=1;
}finally{await tab.close();servers.stop();}
