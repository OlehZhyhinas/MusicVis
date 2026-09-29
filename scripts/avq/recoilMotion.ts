// Onset routing and motion must work through the live renderer, once per body/frame.
// Run: node --import ./scripts/analysis-test.hooks.mjs scripts/avq/recoilMotion.ts
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
    const errors = [], checked = [];
    async function render(g, excite) {
      const programs = eng.cache.get(g, true);
      if (!programs) throw Error('compile failed');
      const st = new Stage(eng, { offscreen: true, flameCap: 65536, particleCap: 65536 }); st.resize(320, 180);
      const slot = st.makeSlot(g, programs); st.slots = [slot];
      const bi = g.bodies.findIndex(b => b.motion.kind === 'recoil');
      const key = 'b' + bi + '.mrecoil.x';
      const clip = new ReferenceClip();
      let peak = 0, samples = [];
      const shot = new Uint8Array(320*180*4);
      for (let f = 0; f < 240; f++) {
        const state = clip.next(), kick = excite && f === 60 ? 1 : 0;
        state.onBeat = false; state.beatPulse = 0;
        state.stems = { drums: 0.7, bass: 0.7, vocals: 0.7, other: 0.7 };
        state.stemPresence = { drums: 1, bass: 1, vocals: 1, other: 1 };
        state.stemOnsets = { drums: kick, bass: kick, vocals: kick, other: kick };
        state.notes = { on: kick, held: 0.7, height: 0.5, pitch: 60, glide: 0, vibrato: 0, voice: 0.5, legato: 0.5, recent: [] };
        st.render(state, 1/60, 'out');
        const x = slot.mem[key] ?? 0;
        if (!Number.isFinite(x)) throw Error('non-finite spring');
        peak = Math.max(peak, Math.abs(x)); samples.push(x);
        if (f === 67) st.readPixels(shot);
      }
      if (eng.gl.getError() !== eng.gl.NO_ERROR) throw Error('WebGL error');
      st.disposeSlot(slot); st.slots = []; st.dispose();
      return { peak, samples, shot };
    }
    for (const id of ['X63', 'X64', 'X65']) {
      const g = SEEDS.find(s => s.origin === id).genome;
      const active = await render(g, true), idle = await render(g, false);
      if (active.peak < 0.1 || idle.peak > 1e-8) errors.push(id + ': onset not routed correctly');
      if (Math.abs(active.samples[239]) > active.peak * 0.05) errors.push(id + ': did not settle');
      let delta = 0;
      for (let i=0; i<active.shot.length; i++) if (i%4 !== 3) delta += Math.abs(active.shot[i]-idle.shot[i]);
      if (delta / (320*180*3*255) < 0.001) errors.push(id + ': recoil not visible');
      if (id === 'X63') {
        const single = cloneGenome(g); single.bodies[1].place.p.count = 1;
        const one = await render(single, true);
        if (one.samples.some((x,i) => Math.abs(x-active.samples[i]) > 1e-8)) errors.push('copy count changed spring timing');
      }
      checked.push(id);
    }
    return { checked, errors };
  })()`);
  console.log(JSON.stringify(result, null, 2));
  const gpuErrors = tab.logs.filter(s => /error|EXC/i.test(s));
  if (gpuErrors.length) console.error(gpuErrors.join('\n'));
  if (result.errors.length || gpuErrors.length) process.exitCode = 1;
} finally { await tab.close(); servers.stop(); }
