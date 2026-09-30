import { ensureServers, Tab } from './cdp';
const servers=await ensureServers(),tab=await Tab.open();
try{
  await tab.load();
  const result=await tab.eval<{checked:string[];errors:string[]}>(`(async()=>{
    const {Engine,Stage}=await import('/src/v2/engine.ts');
    const {SyntheticMusic}=await import('/src/v2/screen.ts');
    const {SEEDS}=await import('/src/v2/seeds.ts');
    const eng=new Engine(document.createElement('canvas')),checked=[],errors=[];
    const sheet=document.createElement('canvas');sheet.width=1920;sheet.height=3*394;
    const ctx=sheet.getContext('2d');ctx.fillStyle='#111';ctx.fillRect(0,0,sheet.width,sheet.height);
    for(const [row,id] of ['X84','X85','X86'].entries()){
      const seed=SEEDS.find(s=>s.origin===id),g=seed.genome;
      const st=new Stage(eng,{offscreen:true,flameCap:65536,particleCap:65536});st.resize(640,360);
      const slot=st.makeSlot(g,eng.cache.get(g,true));st.slots=[slot];const music=new SyntheticMusic(),silent=new SyntheticMusic(true);
      const pixels=new Uint8Array(640*360*4),hues=new Set(),channels=new Set();let lit=0;
      for(let f=0;f<120;f++){
        const state=music.next(1/60),t=f/60,step=Math.floor(t/0.25),pitch=48+[0,2,4,7,9][step%5];
        state.notes={height:(pitch-48)/12,pitch,on:Math.exp(-(t%0.25)*30),held:0.5,glide:0,vibrato:0,voice:0.5,legato:0.5,recent:[]};
        st.render(state,1/60,'out');
        const bolts=slot.lightning.get(0).simulation;
        for(const b of bolts.bolts){hues.add(b.hue);channels.add(b.channel);}
        if([8,38,98].includes(f)){
          st.readPixels(pixels);if(pixels.some((v,i)=>i%4!==3&&v>20))lit++;
          const img=ctx.createImageData(640,360);for(let y=0;y<360;y++)img.data.set(pixels.subarray((359-y)*640*4,(360-y)*640*4),y*640*4);
          const col=[8,38,98].indexOf(f);ctx.putImageData(img,col*640,row*394+34);ctx.fillStyle='#eee';ctx.font='18px sans-serif';ctx.fillText(id+' '+seed.name+' / strike '+(col+1),col*640+12,row*394+23);
        }
      }
      if(lit!==3||hues.size<4||channels.size<2)errors.push(id+' missing visible note colours/channels');
      for(let f=0;f<180;f++)st.render(silent.next(1/60),1/60,'out');
      st.readPixels(pixels);
      if(slot.lightning.get(0).simulation.bolts.length || pixels.some((v,i)=>i%4!==3&&v>12))errors.push(id+' did not decay in silence');
      if(eng.gl.getError()!==eng.gl.NO_ERROR)errors.push(id+' WebGL error');
      checked.push(id+' '+hues.size+' note colours, '+channels.size+' channels; decay verified');st.disposeSlot(slot);st.dispose();
    }
    await fetch('/avq/save?p=lightning-studies-preview.png',{method:'POST',body:await new Promise(r=>sheet.toBlob(r,'image/png'))});
    return {checked,errors};
  })()`);
  console.log(JSON.stringify(result,null,2));const errors=tab.logs.filter(s=>/error|EXC/i.test(s));if(errors.length)console.error(errors.join('\n'));if(result.errors.length||errors.length)process.exitCode=1;
}finally{await tab.close();servers.stop();}
