// Harmony shape check (CLI): reads the engine readout (.inst.json) of rendered clips and reports
// whether the harmony gene's geometry holds still within chords and moves only at chord changes.
//
//   node --import ./scripts/analysis-test.hooks.mjs scripts/avq/harmonyShape.ts [--preset H01,H05] [--song slug]
//
// Per frame the shape change is d = |d fold| + |d warp| + 0.5 warp |d phase| + a seed term, with the
// fold loosening the ops actually read (tile / kaleido: harm.brk and a hashed seed; mirror / polar:
// harm.bend with the phase as seed). Reported per clip:
//   inChordShare   share of all shape change more than 0.9 s from any chord change (0 = chord-locked)
//   stillInChord   share of frames inside chords with no shape change at all
//   maxFrame       the largest single-frame change
//   snaps          one-frame spikes: d > 0.02 and over 3x both neighbours
//   zoomJump       the largest frame-to-frame change of the gene's zoom

import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { OUT } from './cdp';
import { listClips } from './load';
import { parseArgs } from './render';
import { SEEDS } from '../../src/v2/seeds';

const a = parseArgs(process.argv.slice(2));
const presets = a.preset ? String(a.preset).split(',') : SEEDS.filter((s) => s.genome.harmony).map((s) => s.origin);
for (const preset of presets) {
  const seed = SEEDS.find((s) => s.origin === preset.split('@')[0]);
  const hashed = !!seed?.genome.chain.some((o) => o.op === 'tile' || o.op === 'kaleido');
  for (const base of listClips({ preset, song: a.song ? String(a.song) : undefined })) {
    let j: { names: string[]; cols: (number | null)[][] };
    try {
      j = JSON.parse(readFileSync(join(OUT, 'clips', base + '.inst.json'), 'utf8'));
    } catch {
      continue;
    }
    const col = (n: string) => {
      const k = j.names.indexOf(n);
      return k < 0 ? [] : j.cols[k].map((x) => x ?? 0);
    };
    const chord = col('F.chord'), brk = col('harm.brk'), bend = col('harm.bend'), warp = col('harm.warp');
    const ph = col('harm.phase'), sd = col('harm.seed'), zoom = col('harm.zoom');
    if (!ph.length) {
      console.log(`${base}: no harmony shape columns (re-render the clip)`);
      continue;
    }
    const fps = JSON.parse(readFileSync(join(OUT, 'clips', base + '.json'), 'utf8')).render.fps as number;
    const n = chord.length;
    const bounds: number[] = [];
    let last = -1;
    for (let i = 0; i < n; i++) if (chord[i] >= 0 && chord[i] !== last) {
      if (last >= 0) bounds.push(i);
      last = chord[i];
    }
    const near = (i: number) => bounds.some((b) => Math.abs(b - i) <= 0.9 * fps);
    const L = hashed ? brk : bend;
    const S = hashed ? sd : ph;
    const ds = [0];
    let inS = 0, nearS = 0, inF = 0, still = 0, zj = 0;
    for (let i = 1; i < n; i++) {
      const dS = Math.abs(S[i] - S[i - 1]);
      const seedTerm = hashed ? (dS > 1e-6 ? Math.abs(L[i]) : 0) : 0.3 * Math.abs(L[i]) * Math.min(1, dS);
      const d = Math.abs(L[i] - L[i - 1]) + Math.abs(warp[i] - warp[i - 1]) + 0.5 * warp[i] * Math.abs(ph[i] - ph[i - 1]) + seedTerm;
      ds.push(d);
      if (near(i)) nearS += d;
      else {
        inS += d;
        inF++;
        if (d < 1e-5) still++;
      }
      zj = Math.max(zj, Math.abs(zoom[i] - zoom[i - 1]));
    }
    let snaps = 0;
    for (let i = 1; i + 1 < n; i++) if (ds[i] > 0.02 && ds[i] > 3 * Math.max(ds[i - 1], ds[i + 1])) snaps++;
    const r = (x: number, k = 3) => +x.toFixed(k);
    console.log(JSON.stringify({ clip: base, chordChanges: bounds.length, inChordShare: r(inS / Math.max(1e-9, inS + nearS)), stillInChord: r(still / Math.max(1, inF)), maxFrame: r(Math.max(...ds)), snaps, zoomJump: r(zj) }));
  }
}
