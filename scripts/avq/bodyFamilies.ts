// GPU integration check for the new composable body families. Checks every material,
// two copies of the same gene, and the gene as a fused secondary shape.
// Run: node --import ./scripts/analysis-test.hooks.mjs scripts/avq/bodyFamilies.ts
import { ensureServers, Tab } from './cdp';

const servers = await ensureServers();
const tab = await Tab.open();
try {
  await tab.load();
  const result = await tab.eval<{ compiled: number; errors: string[] }>(`(async () => {
    const { Engine, Stage } = await import('/src/v2/engine.ts');
    const { SEEDS } = await import('/src/v2/seeds.ts');
    const { cloneGenome, repair, MATERIAL_KINDS, defaultParams, MATERIAL_SCHEMAS, FUSE_SCHEMA } = await import('/src/v2/genome.ts');
    const { ReferenceClip } = await import('/src/v2/fingerprint.ts');
    const eng = new Engine(document.createElement('canvas'));
    const st = new Stage(eng, { offscreen: true }); st.resize(320, 180);
    const pixels = new Uint8Array(320 * 180 * 4);
    const errors = []; let compiled = 0;
    for (const kind of ['branch', 'fabric', 'linkage']) {
      const seed = SEEDS.find(s => s.genome.bodies[0].shape.kind === kind);
      for (const material of MATERIAL_KINDS) for (const mode of ['body', 'pair', 'fused']) {
        const raw = cloneGenome(seed.genome);
        raw.reactions = [];
        const b = raw.bodies[0];
        b.material = { kind: material, p: defaultParams(MATERIAL_SCHEMAS[material]) };
        // Glow is normalized for small emitters; large filled-body defaults intentionally
        // dim it almost to black. Use a source-sized body and a visible glow width.
        if (material === 'glow') { b.shape.p.size = 0.1; Object.assign(b.material.p, { width: 0.06, base: 0.4, gain: 2 }); }
        if (mode === 'pair') raw.bodies.push(structuredClone(b));
        if (mode === 'fused') {
          b.fuse = { shape: b.shape, p: { ...defaultParams(FUSE_SCHEMA), mode: 1, t: 0.5 } };
          b.shape = { kind: 'dot', p: { r: 0.18 } };
        }
        const g = repair(raw);
        const programs = eng.cache.get(g, true);
        const label = kind + '/' + material + '/' + mode;
        if (!programs) { errors.push(label + ': ' + eng.cache.failed(g)); continue; }
        const slot = st.makeSlot(g, programs); st.slots = [slot]; st.resetHistory();
        const clip = new ReferenceClip(); clip.seek(8);
        for (let f = 0; f < 12; f++) st.render(clip.next(), 1 / 60, 'out');
        st.readPixels(pixels);
        if (eng.gl.getError() !== eng.gl.NO_ERROR) errors.push(label + ': WebGL error');
        if (!pixels.some((x, i) => i % 4 !== 3 && x > 12)) errors.push(label + ': blank');
        st.disposeSlot(slot); st.slots = [];
        compiled++;
      }
    }
    return { compiled, errors };
  })()`);
  console.log(JSON.stringify(result, null, 2));
  const gpuErrors = tab.logs.filter(s => /error|EXC/i.test(s));
  if (gpuErrors.length) console.error(gpuErrors.join('\n'));
  if (result.errors.length || gpuErrors.length) process.exitCode = 1;
} finally {
  await tab.close();
  servers.stop();
}
