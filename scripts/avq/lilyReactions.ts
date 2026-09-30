// Ensure the reference-inspired flowers change geometry with musical phrasing.
// Run: node --import ./scripts/analysis-test.hooks.mjs scripts/avq/lilyReactions.ts
import { ensureServers, Tab } from './cdp';
const servers = await ensureServers();
const tab = await Tab.open();
try {
  await tab.load();
  const result = await tab.eval<{ checked: string[]; errors: string[] }>(`(async () => {
    const {Engine,Stage}=await import('/src/v2/engine.ts');
    const {SEEDS}=await import('/src/v2/seeds.ts');
    const {cloneGenome}=await import('/src/v2/genome.ts');
    const {ReferenceClip}=await import('/src/v2/fingerprint.ts');
    const eng=new Engine(document.createElement('canvas')); await eng.lilyAtlas.ready;
    if (!eng.lilyAtlas.loaded) throw Error('lily artwork did not load');
    const checked=[],errors=[];
    async function render(g,level) {
      const progs=eng.cache.get(g,true); if(!progs) throw Error('compile failed');
      const st=new Stage(eng,{offscreen:true,flameCap:65536,particleCap:65536});st.resize(320,180);
      const slot=st.makeSlot(g,progs);st.slots=[slot];
      const clip=new ReferenceClip();
      for(let i=0;i<180;i++) {
        const state=clip.next();
        state.stems.vocals=level;state.stemPresence.vocals=1;
        state.notes={height:level,held:level,pitch:60,on:0,glide:0,vibrato:0,voice:0.5,legato:0.5,recent:[]};
        st.render(state,1/60,'out');
      }
      const pixels=new Uint8Array(320*180*4);st.readPixels(pixels);
      const opening=slot.bd[10];
      if(eng.gl.getError()!==eng.gl.NO_ERROR) throw Error('WebGL error');
      st.disposeSlot(slot);st.slots=[];st.dispose();
      return {pixels,opening};
    }
    for(const id of ['X75','X76','X77']) {
      const g=cloneGenome(SEEDS.find(s=>s.origin===id).genome);
      g.reactions=g.reactions.filter(r=>r.g==='sh');
      const quiet=await render(g,0), singing=await render(g,1);
      let delta=0;for(let i=0;i<quiet.pixels.length;i++) if(i%4!==3) delta+=Math.abs(quiet.pixels[i]-singing.pixels[i]);
      delta/=320*180*3*255;
      if(singing.opening-quiet.opening<0.15) errors.push(id+': note/vocal envelope did not open the flower');
      if(delta<0.002) errors.push(id+': musical geometry change was not visible');
      checked.push(id+' opening '+quiet.opening.toFixed(2)+' -> '+singing.opening.toFixed(2)+', image delta '+delta.toFixed(4));
    }
    return {checked,errors};
  })()`);
  console.log(JSON.stringify(result,null,2));
  const gpuErrors=tab.logs.filter(s=>/error|EXC/i.test(s));
  if(gpuErrors.length) console.error(gpuErrors.join('\n'));
  if(result.errors.length || gpuErrors.length) process.exitCode=1;
} finally { await tab.close();servers.stop(); }
