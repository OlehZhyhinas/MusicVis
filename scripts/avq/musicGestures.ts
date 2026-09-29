// Controlled pitch gestures through the real Signals -> reactions -> WebGL path.
// The standard ReferenceClip has no note analysis, so supply it explicitly here.
// Run: node --import ./scripts/analysis-test.hooks.mjs scripts/avq/musicGestures.ts
import { ensureServers, Tab } from './cdp';
const servers = await ensureServers();
const tab = await Tab.open();
try {
  await tab.load();
  const result = await tab.eval<{ checked: string[]; errors: string[] }>(`(async () => {
    const { Engine, Stage } = await import('/src/v2/engine.ts');
    const { SEEDS } = await import('/src/v2/seeds.ts');
    const { ReferenceClip } = await import('/src/v2/fingerprint.ts');
    const eng = new Engine(document.createElement('canvas'));
    const ids = ['X52', 'X53'];
    const gestures = [
      { label: 'low / falling', height: 0.15, glide: -9 },
      { label: 'middle / steady', height: 0.5, glide: 0 },
      { label: 'high / rising', height: 0.85, glide: 9 },
    ];
    const sheet = document.createElement('canvas'); sheet.width = 1920; sheet.height = ids.length * 394;
    const ctx = sheet.getContext('2d'); ctx.fillStyle = '#111'; ctx.fillRect(0, 0, sheet.width, sheet.height);
    const errors = [], checked = [];
    for (const [row, id] of ids.entries()) {
      const seed = SEEDS.find(s => s.origin === id), g = seed.genome;
      const programs = eng.cache.get(g, true);
      if (!programs) { errors.push(id + ': compile failed'); continue; }
      const shots = [];
      for (const [col, gesture] of gestures.entries()) {
        const st = new Stage(eng, { offscreen: true }); st.resize(640, 360);
        const slot = st.makeSlot(g, programs); st.slots = [slot];
        const clip = new ReferenceClip(); clip.seek(8);
        for (let f = 0; f < 90; f++) {
          const state = clip.next();
          state.notes = { on: 0, held: 0.8, legato: 0.7, glide: gesture.glide, vibrato: 0.1,
            pitch: 48 + gesture.height * 24, height: gesture.height, voice: 0.7, recent: [] };
          st.render(state, 1 / 60, 'out');
        }
        const expected = [gesture.height, Math.max(0, gesture.glide) / 12, Math.max(0, -gesture.glide) / 12];
        for (const [i, source] of ['register', 'rising', 'falling'].entries()) {
          if (Math.abs(st.sig.signal(source) - expected[i]) > 1e-6) errors.push(id + ': wrong ' + source);
        }
        const pixels = new Uint8Array(640 * 360 * 4); st.readPixels(pixels); shots.push(pixels);
        if (eng.gl.getError() !== eng.gl.NO_ERROR) errors.push(id + ': WebGL error');
        if (!pixels.some((v, i) => i % 4 !== 3 && v > 20)) errors.push(id + ': blank');
        const img = ctx.createImageData(640, 360);
        for (let y = 0; y < 360; y++) img.data.set(pixels.subarray((359-y)*640*4, (360-y)*640*4), y*640*4);
        ctx.putImageData(img, col*640, row*394+34);
        ctx.fillStyle = '#eee'; ctx.font = '18px sans-serif';
        ctx.fillText(id + ' ' + seed.name + ' / ' + gesture.label, col*640+12, row*394+23);
        st.disposeSlot(slot); st.slots = [];
        st.dispose();
      }
      for (const i of [0, 2]) {
        let delta = 0;
        for (let p = 0; p < shots[i].length; p++) if (p % 4 !== 3) delta += Math.abs(shots[i][p] - shots[1][p]);
        if (delta / (640*360*3*255) < 0.002) errors.push(id + ': gesture has no visible effect');
      }
      checked.push(id);
    }
    await fetch('/avq/save?p=music-gestures-preview.png', { method: 'POST', body: await new Promise(r => sheet.toBlob(r, 'image/png')) });
    return { checked, errors };
  })()`);
  console.log(JSON.stringify(result, null, 2));
  const gpuErrors = tab.logs.filter(s => /error|EXC/i.test(s));
  if (gpuErrors.length) console.error(gpuErrors.join('\n'));
  if (result.errors.length || gpuErrors.length) process.exitCode = 1;
} finally { await tab.close(); servers.stop(); }
