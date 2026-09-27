import { readdirSync } from 'node:fs';
import { RealtimeAnalyzer } from '../../../../src/analysis/RealtimeAnalyzer';
import { decode } from '../../../live/common';

const TEST_DIR = '/Users/oleh/Downloads/YoutubeToMp3';
const file = readdirSync(TEST_DIR).find((f) => /\.(mp3|m4a)$/i.test(f))!;
const pcm = decode(TEST_DIR + '/' + file);
const analyzer = new RealtimeAnalyzer(pcm.sr);
const BLOCK = 512;
const T = Math.floor(pcm.left.length / BLOCK);
const t0 = performance.now();
for (let k = 0; k < T; k++) {
  analyzer.process(pcm.left.subarray(k * BLOCK, (k + 1) * BLOCK), pcm.right.subarray(k * BLOCK, (k + 1) * BLOCK));
}
const t1 = performance.now();
const msPerHop = (t1 - t0) / T;
const hopMs = (512 / 44100) * 1000;
console.log(JSON.stringify({ T, ms_per_hop: msPerHop, hop_ms: hopMs, cpu_pct: (100 * msPerHop) / hopMs }, null, 1));
