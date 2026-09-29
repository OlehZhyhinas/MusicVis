// Exercise both shader directions, material combinations, and musical reactions.
// Run: node --import ./scripts/analysis-test.hooks.mjs scripts/avq/undulateDeform.ts
import { ensureServers, Tab } from './cdp';
const servers = await ensureServers();
const tab = await Tab.open();
try {
  await tab.load();
  const result = await tab.eval<{ checked: string[]; errors: string[] }>(`(async () => {
    const { Engine, Stage } = await import('/src/v2/engine.ts');
    const { SEEDS } = await import('/src/v2/seeds.ts');
    const { cloneGenome, defaultParams, MATERIAL_SCHEMAS } = await import('/src/v2/genome.ts');
    const { UNDULATE_GLSL } = await import('/src/v2/genes/undulate.ts');
    const { ReferenceClip } = await import('/src/v2/fingerprint.ts');
    const errors = [], checked = [];
    // Execute the production GLSL map directly, including its inverse and pinned origin.
    const gl = document.createElement('canvas').getContext('webgl2');
    const program = gl.createProgram();
    const vertex = '#version 300 es\\nprecision highp float;\\nconst float TAU=6.28318530718;\\n' +
      'mat2 rot2(float a){float c=cos(a),s=sin(a);return mat2(c,s,-s,c);}\\n' + UNDULATE_GLSL +
      'uniform vec4 D; uniform float pin; out vec4 result; void main(){' +
      'vec2 q=vec2(float(gl_VertexID%13)-6.,float(gl_VertexID/13)-6.)*.071;' +
      'vec2 f=undulateMap(q,D,pin,1.);vec2 b=undulateMap(f,D,pin,-1.);' +
      'result=vec4(b-q,undulateMap(vec2(0),D,pin,1.));gl_Position=vec4(0,0,0,1); }';
    for(const [type,source] of [[gl.VERTEX_SHADER,vertex],[gl.FRAGMENT_SHADER,'#version 300 es\\nprecision highp float;out vec4 c;void main(){c=vec4(1); }']]) {
      const shader=gl.createShader(type);gl.shaderSource(shader,source);gl.compileShader(shader);
      if(!gl.getShaderParameter(shader,gl.COMPILE_STATUS)) throw Error(gl.getShaderInfoLog(shader));
      gl.attachShader(program,shader);gl.deleteShader(shader);
    }
    gl.transformFeedbackVaryings(program,['result'],gl.INTERLEAVED_ATTRIBS);gl.linkProgram(program);
    if(!gl.getProgramParameter(program,gl.LINK_STATUS)) throw Error(gl.getProgramInfoLog(program));
    gl.useProgram(program);
    const buffer=gl.createBuffer();gl.bindBuffer(gl.TRANSFORM_FEEDBACK_BUFFER,buffer);
    const data=new Float32Array(169*4);gl.bufferData(gl.TRANSFORM_FEEDBACK_BUFFER,data.byteLength,gl.STREAM_READ);
    gl.bindBufferBase(gl.TRANSFORM_FEEDBACK_BUFFER,0,buffer);gl.enable(gl.RASTERIZER_DISCARD);
    for(const amp of [0,0.06,0.18]) for(const span of [0.08,0.55,1.2]) for(const angle of [-3.14,0,1.57]) for(const pin of [0,1]) {
      gl.uniform4f(gl.getUniformLocation(program,'D'),amp,span,1.1,angle);
      gl.uniform1f(gl.getUniformLocation(program,'pin'),pin);
      gl.beginTransformFeedback(gl.POINTS);gl.drawArrays(gl.POINTS,0,169);gl.endTransformFeedback();
      gl.getBufferSubData(gl.TRANSFORM_FEEDBACK_BUFFER,0,data);
      for(let i=0;i<data.length;i++) if(!Number.isFinite(data[i]) || ((i%4<2 || pin===1) && Math.abs(data[i])>0.00001)) {
        errors.push('inverse or pin failed at '+[amp,span,angle,pin]); break;
      }
    }
    if(gl.getError()!==gl.NO_ERROR) errors.push('map WebGL error');
    gl.deleteBuffer(buffer);gl.deleteProgram(program);checked.push('54 GPU inverse/pin configurations');

    const eng = new Engine(document.createElement('canvas'));
    async function render(g, excite = false) {
      const programs = eng.cache.get(g, true); if (!programs) throw Error('compile failed');
      const st = new Stage(eng, { offscreen: true, flameCap: 65536, particleCap: 65536 }); st.resize(320, 180);
      const slot = st.makeSlot(g, programs); st.slots = [slot];
      const clip = new ReferenceClip(); const pixels = new Uint8Array(320*180*4);
      for(let f=0;f<120;f++) {
        const state=clip.next();
        state.notes={height:excite?0.9:0.1,held:excite?0.9:0.1,pitch:60,on:0,glide:0,vibrato:0,voice:0.5,legato:0.5,recent:[]};
        state.stems.vocals=excite?0.95:0.05;state.stemPresence.vocals=1;
        state.tension=excite?0.9:0.1;
        st.render(state,1/60,'out');
      }
      st.readPixels(pixels);
      if(eng.gl.getError()!==eng.gl.NO_ERROR) throw Error('render WebGL error');
      st.disposeSlot(slot);st.slots=[];st.dispose();return pixels;
    }
    function delta(a,b) {
      let sum=0;for(let i=0;i<a.length;i++) if(i%4!==3) sum+=Math.abs(a[i]-b[i]);
      return sum/(320*180*3*255);
    }
    for(const id of ['X66','X67','X68']) {
      const seed=SEEDS.find(s=>s.origin===id).genome;
      const active=await render(seed,true), idle=await render(seed,false);
      if(delta(active,idle)<0.001) errors.push(id+': musical bend not visible');
      const g=cloneGenome(seed);g.reactions=[];
      const bodies=g.bodies.filter(b=>b.deform.kind==='undulate');
      for(const b of bodies) b.deform.p.amp=0;
      const zero=await render(g);
      for(const b of bodies) b.deform={kind:'none',p:{}};
      if(delta(zero,await render(g))>0.0001) errors.push(id+': zero amplitude changed silhouette');
      for(const material of ['line','fill','glow','dots','textured','chrome']) {
        const variant=cloneGenome(seed);
        for(const b of variant.bodies) if(b.deform.kind==='undulate') {
          b.material={kind:material,p:defaultParams(MATERIAL_SCHEMAS[material])};
          b.deform.p.amp=0.18;b.deform.p.span=0.08;
        }
        await render(variant);checked.push(id+'/'+material);
      }
    }
    return {checked,errors};
  })()`);
  console.log(JSON.stringify(result,null,2));
  const gpuErrors=tab.logs.filter(s=>/error|EXC/i.test(s));
  if(gpuErrors.length) console.error(gpuErrors.join('\n'));
  if(result.errors.length || gpuErrors.length) process.exitCode=1;
} finally { await tab.close(); servers.stop(); }
