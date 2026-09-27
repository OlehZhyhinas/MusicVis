// For the stems.* normalisation study: records, per analyzer frame, the power each stems.* DbEnvelope
// receives, the running `top` it is normalised against, and the 0..1 value it returns, for the DSP
// split (mode dsp) or the stem network (mode neural), by wrapping the analyzer's private env pushes.
// Writes .testdata/instr/presence/norm/<mode>/<slug>.f32 (T x 12: p[4], top[4], out[4]).
//   node --import ./scripts/analysis-test.hooks.mjs scripts/ml/instruments/presence/norm_dump.ts dsp|neural
import { mkdirSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { RealtimeAnalyzer } from '../../../../src/analysis/RealtimeAnalyzer';
import { StemNetSource } from '../../../../src/analysis/stemNet';
import { decode, slugOf, stemNetModelSync, testSongs } from '../../../live/common';

const mode = process.argv[2] ?? 'dsp';
const OUT = join(import.meta.dirname, '../../../../.testdata/instr/presence/norm', mode);
mkdirSync(OUT, { recursive: true });
const S = ['drums', 'bass', 'vocals', 'other'] as const;
for (const path of testSongs()) {
  const slug = slugOf(path);
  const pcm = decode(path);
  const an = new RealtimeAnalyzer(pcm.sr);
  const src = mode === 'neural' ? new StemNetSource(stemNetModelSync(), pcm.sr) : null;
  if (src) an.stemSource = src.at;
  const rows: number[][] = [];
  let cur: number[] = new Array(12).fill(0);
  let pushed = false;
  const env = (an as any).env as Record<string, { push: (p: number, top: number, a: number, b: number) => number }>;
  S.forEach((s, i) => {
    const e = env[s];
    const orig = e.push.bind(e);
    e.push = (p: number, top: number, a: number, b: number) => {
      const v = orig(p, top, a, b);
      cur[i] = p;
      cur[4 + i] = top;
      cur[8 + i] = v;
      if (i === 3) pushed = true;
      return v;
    };
  });
  for (let p = 0; p + 512 <= pcm.left.length; p += 512) {
    const l = pcm.left.subarray(p, p + 512), r = pcm.right.subarray(p, p + 512);
    src?.push(l, r);
    an.process(l, r);
    // one analyzer frame per 512-sample block; silent frames skip the pushes (stems = 0)
    rows.push(pushed ? cur : [0, 0, 0, 0, -150, -150, -150, -150, 0, 0, 0, 0]);
    cur = new Array(12).fill(0);
    pushed = false;
  }
  const out = new Float32Array(rows.length * 12);
  rows.forEach((r, k) => out.set(r, k * 12));
  writeFileSync(join(OUT, slug + '.f32'), Buffer.from(out.buffer));
  console.error(slug, rows.length);
}
