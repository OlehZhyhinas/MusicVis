// Check the live motion path on SDF and curve bodies, including music reactions.
// Run: node --import ./scripts/analysis-test.hooks.mjs scripts/avq/weaveMotion.ts
import { ensureServers, Tab } from './cdp';
const servers = await ensureServers();
const tab = await Tab.open();
try {
  await tab.load();
  const result = await tab.eval<{ checked: string[]; errors: string[] }>(`(async () => {
    const { Engine, Stage } = await import('/src/v2/engine.ts');
    const { SEEDS } = await import('/src/v2/seeds.ts');
    const { cloneGenome } = await import('/src/v2/genome.ts');
    const { ReferenceClip } = await import('/src/v2/fingerprint.ts');
    const eng = new Engine(document.createElement('canvas'));
    const checked = [], errors = [];
    async function render(g, drive = 0) {
      const programs = eng.cache.get(g,true); if (!programs) throw Error('compile failed');
      const st = new Stage(eng,{offscreen:true,flameCap:65536,particleCap:65536}); st.resize(320,180);
      const slot = st.makeSlot(g,programs); st.slots = [slot];
      const clip = new ReferenceClip(), frames = [], copies = [];
      for(let f=0;f<180;f++) {
        const state = clip.next();
        state.notes={height:0.5,held:drive,pitch:60,on:0,glide:0,vibrato:0,voice:0.5,legato:0.5,recent:[]};
        state.groove={swing:drive,push:0,humanity:0,synco:drive}; state.tension=drive;
        st.render(state,1/60,'out');
        if ([59,119,179].includes(f)) {
          const pixels = new Uint8Array(320*180*4); st.readPixels(pixels); frames.push(pixels);
          copies.push([...slot.cp]);
        }
      }
      if(eng.gl.getError()!==eng.gl.NO_ERROR) throw Error('WebGL error');
      st.disposeSlot(slot); st.slots=[]; st.dispose(); return {frames,copies};
    }
    function delta(a,b) {
      let sum=0;
      for(let f=0;f<a.frames.length;f++) for(let i=0;i<a.frames[f].length;i++)
        if(i%4!==3) sum+=Math.abs(a.frames[f][i]-b.frames[f][i]);
      return sum/(a.frames.length*320*180*3*255);
    }
    for(const id of ['X72','X73','X74']) {
      const g=cloneGenome(SEEDS.find(s=>s.origin===id).genome);
      // Isolate motion reactions; differences in material/shape cannot hide a dead movement target.
      g.reactions=g.reactions.filter(r=>r.g==='mo');
      const low=await render(g,0), high=await render(g,1);
      let travel=0;
      high.copies.forEach((a,f)=>a.forEach((v,i)=>{if(i%4<2) travel+=Math.abs(v-low.copies[f][i]);}));
      if(travel<0.05) errors.push(id+': music did not change the motion path');
      g.reactions=[];
      const active=await render(g);
      for(const b of g.bodies) if(b.motion.kind==='weave') Object.assign(b.motion.p,{radius:0,height:0,bank:0});
      const zero=await render(g);
      for(const b of g.bodies) if(b.motion.kind==='weave') b.motion={kind:'none',p:{}};
      const idle=await render(g);
      if(delta(zero,idle)>0.0001) errors.push(id+': zero weave differs from no motion');
      if(delta(active,idle)<0.001) errors.push(id+': movement not visible');
      checked.push(id);
    }
    return {checked,errors};
  })()`);
  console.log(JSON.stringify(result,null,2));
  const gpuErrors=tab.logs.filter(s=>/error|EXC/i.test(s));
  if(gpuErrors.length) console.error(gpuErrors.join('\n'));
  if(result.errors.length || gpuErrors.length) process.exitCode=1;
} finally { await tab.close(); servers.stop(); }
