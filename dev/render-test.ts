// Dev harness for src/render: a synthetic MusicState at 128 bpm with a
// section cycle intro -> verse -> build -> drop (8 s each) and key changes
// every 20 s. "Simple" mode fakes a solo melody (complexity 0.15, only vocals
// and other present, no drop); "dense" fakes a full EDM mix (0.9, all stems).
// Keys: 1/2/3 = enhanced/classic/hybrid, N = next preset, P = pause,
// S = simple/dense toggle, G = newSong(), L = lock toggle, H = hide HUD.
// Query: ?mode=hybrid&particles=262144&scale=0.5&size=2560x1440&preset=E07&lock=1&simple=1

import { Visualizer } from '../src/render/Visualizer';
import type { MusicState, SectionLabel, StemName, VisualMode } from '../src/types';

const params = new URLSearchParams(location.search);
const canvas = document.getElementById('c') as HTMLCanvasElement;
const hud = document.getElementById('hud') as HTMLDivElement;

const ctx = new AudioContext();
const osc = ctx.createOscillator();
osc.type = 'sawtooth';
osc.frequency.value = 110;
const lfo = ctx.createOscillator();
lfo.frequency.value = 2.13;
const lfoGain = ctx.createGain();
lfoGain.gain.value = 40;
lfo.connect(lfoGain).connect(osc.frequency);
const noiseBuf = ctx.createBuffer(1, ctx.sampleRate * 2, ctx.sampleRate);
const nd = noiseBuf.getChannelData(0);
for (let i = 0; i < nd.length; i++) nd[i] = (Math.random() * 2 - 1) * (i % 11025 < 1500 ? 1 : 0.05);
const noise = ctx.createBufferSource();
noise.buffer = noiseBuf;
noise.loop = true;
const mix = ctx.createGain();
mix.gain.value = 0.5;
osc.connect(mix);
noise.connect(mix);
osc.start();
lfo.start();
noise.start();
const resume = () => void ctx.resume();
window.addEventListener('pointerdown', resume);

const viz = new Visualizer(canvas, { context: ctx, source: mix }, {
  particleCount: Number(params.get('particles') ?? 1048576),
  renderScale: Number(params.get('scale') ?? 1),
});
const forced = params.get('size')?.split('x').map(Number);
function doResize(): void {
  if (forced && forced.length === 2) viz.resize(forced[0], forced[1], 1);
  else viz.resize(window.innerWidth, window.innerHeight, window.devicePixelRatio || 1);
}
doResize();
window.addEventListener('resize', doResize);
const startMode = params.get('mode') as VisualMode | null;
if (startMode) viz.setMode(startMode);
const presetParam = params.get('preset');
if (presetParam !== null) viz.selectPreset(/^\d+$/.test(presetParam) ? Number(presetParam) : presetParam, 0.01);

// ---------------------------------------------------------------- fake state
const BPM = 128;
const BEATS_PER_BAR = 4;
const SECTION_LEN = 8;
const SEQ: SectionLabel[] = ['intro', 'verse', 'build', 'drop'];
const SEQ_SIMPLE: SectionLabel[] = ['intro', 'verse', 'chorus', 'verse'];
const stems: Record<StemName, number> = { drums: 0, bass: 0, vocals: 0, other: 0 };
const stemOnsets: Record<StemName, number> = { drums: 0, bass: 0, vocals: 0, other: 0 };
const stemPresence: Record<StemName, number> = { drums: 0.8, bass: 0.8, vocals: 0.8, other: 0.8 };
let simple = params.get('simple') === '1';
const SCALE = [0, 2, 4, 7, 9, 12, 11, 7];
const waveform = new Float32Array(1024);
const spectrum = new Float32Array(512);
const chroma = new Float32Array(12);
const section = { start: 0, end: SECTION_LEN, label: 'intro' as SectionLabel, energy: 0.3 };

const state: MusicState = {
  time: 0, dt: 1 / 60, playing: true,
  bass: 1, mid: 1, treb: 1, bassAtt: 1, midAtt: 1, trebAtt: 1,
  waveform, spectrum,
  bpm: BPM, beatIndex: -1, barIndex: -1, beatPhase: 0, barPhase: 0, beatPulse: 0, barPulse: 0, onBeat: false, onBar: false,
  stems, stemOnsets, stemPresence, loudness: 0.5, complexity: 0.9, songComplexity: 0.9,
  chroma, keyTonic: 0, keyMode: 'major', keyHue: 0, keyChangePulse: 0,
  section, sectionIndex: 0, sectionProgress: 0, sectionChanged: false, dropPulse: 0, buildIntensity: 0,
};

let lockPreset = params.has('lock');
let songTime = Number(params.get('t') ?? 0);
let playing = true;
let lastBeat = -1;
let lastBar = -1;
let lastSection = -1;
let lastKey = -1;
let hueTarget = 0;

function fakeFrame(dt: number): void {
  state.dt = dt;
  state.playing = playing;
  state.onBeat = false;
  state.onBar = false;
  state.sectionChanged = false;
  if (playing) songTime += dt;
  const t = songTime;
  state.time = t;

  const beatF = (t * BPM) / 60;
  const beatIndex = Math.floor(beatF);
  const barIndex = Math.floor(beatIndex / BEATS_PER_BAR);
  state.beatPhase = beatF - beatIndex;
  state.barPhase = (beatF / BEATS_PER_BAR) % 1;
  if (beatIndex !== lastBeat) {
    state.onBeat = playing;
    state.beatPulse = 1;
    lastBeat = beatIndex;
  }
  if (barIndex !== lastBar) {
    state.onBar = playing;
    state.barPulse = 1;
    lastBar = barIndex;
  }
  state.beatIndex = beatIndex;
  state.barIndex = barIndex;
  state.beatPulse *= Math.exp(-dt * 8);
  state.barPulse *= Math.exp(-dt * 4);

  const si = Math.floor(t / SECTION_LEN);
  const label = (simple ? SEQ_SIMPLE : SEQ)[si % SEQ.length];
  if (si !== lastSection) {
    state.sectionChanged = lastSection !== -1 || si > 0;
    lastSection = si;
    section.start = si * SECTION_LEN;
    section.end = section.start + SECTION_LEN;
    section.label = label;
    section.energy = label === 'drop' ? 1 : label === 'build' ? 0.7 : label === 'verse' ? 0.5 : 0.3;
    if (label === 'drop') state.dropPulse = 1;
  }
  state.sectionIndex = si;
  state.sectionProgress = (t - section.start) / SECTION_LEN;
  state.dropPulse *= Math.exp(-dt * 1.2);
  state.buildIntensity = label === 'build' ? state.sectionProgress : Math.max(0, state.buildIntensity - dt * 4);

  const ki = Math.floor(t / 20);
  if (ki !== lastKey) {
    lastKey = ki;
    state.keyTonic = (ki * 7) % 12;
    state.keyMode = ki % 2 ? 'minor' : 'major';
    hueTarget = ((ki * 7) % 12) / 12;
    if (ki > 0) state.keyChangePulse = 1;
  }
  state.keyChangePulse *= Math.exp(-dt * 1.5);
  let dh = hueTarget - state.keyHue;
  dh -= Math.round(dh);
  state.keyHue = (state.keyHue + dh * Math.min(1, dt * 1.5) + 1) % 1;

  const e = section.energy;
  const bp = state.beatPhase;
  const kick = Math.exp(-bp * 7);
  const snare = beatIndex % 2 === 1 ? Math.exp(-bp * 9) : 0;
  const hat = Math.exp(-((beatF * 2) % 1) * 12);
  // Raw stems stay per-song normalized (full range) in both modes, exactly the
  // case the renderer must gate with stemPresence.
  stemOnsets.drums = playing ? Math.max(kick, snare * 0.8) * (0.4 + 0.6 * e) : 0;
  stemOnsets.bass = kick * e;
  stemOnsets.other = hat * 0.5;
  stemOnsets.vocals = 0;
  stems.drums = playing ? (0.3 + 0.5 * kick) * e + 0.1 : 0;
  stems.bass = playing ? (0.4 + 0.5 * Math.exp(-bp * 3)) * (0.3 + 0.7 * e) : 0;
  stems.vocals = playing ? (label === 'verse' || label === 'drop' ? 0.5 + 0.4 * Math.sin(t * 0.9) : 0.15) : 0;
  stems.other = playing ? 0.3 + 0.25 * Math.sin(t * 0.37) + 0.3 * e : 0;
  // Melody: one note per beat from a small scale.
  const note = SCALE[(Math.floor(beatIndex / 2) * 3 + (beatIndex % 2)) % SCALE.length] + state.keyTonic;
  const noteHz = 261.6 * Math.pow(2, note / 12);
  const noteEnv = Math.exp(-bp * 1.5);
  if (simple) {
    state.complexity = 0.15;
    state.songComplexity = 0.15;
    stemPresence.drums = 0;
    stemPresence.bass = 0.02;
    stemPresence.vocals = 0.35;
    stemPresence.other = 0.6;
    stems.other = playing ? 0.35 + 0.5 * noteEnv : 0;
    state.loudness = playing ? 0.2 + 0.2 * noteEnv : 0;
    state.bass = 0.5;
    state.mid = 0.8 + 0.3 * noteEnv;
    state.treb = 0.5;
  } else {
    state.complexity = 0.9;
    state.songComplexity = 0.9;
    stemPresence.drums = stemPresence.bass = stemPresence.vocals = stemPresence.other = 0.8;
    state.loudness = playing ? 0.3 + 0.6 * e * (0.6 + 0.4 * kick) : 0;
    state.bass = 0.6 + 1.2 * kick * e;
    state.mid = 0.8 + 0.4 * Math.sin(t * 2.1);
    state.treb = 0.7 + 0.6 * hat;
  }
  state.bassAtt += (state.bass - state.bassAtt) * 0.1;
  state.midAtt += (state.mid - state.midAtt) * 0.1;
  state.trebAtt += (state.treb - state.trebAtt) * 0.1;

  const amp = playing ? 0.25 + 0.55 * state.loudness : 0.02;
  const cyc = (noteHz / 44100) * 1024; // cycles across the 1024-sample window
  for (let i = 0; i < 1024; i++) {
    const x = i / 1024;
    if (simple) {
      waveform[i] = amp * noteEnv * (0.8 * Math.sin(TAU * (x * cyc + t * 3)) + 0.2 * Math.sin(TAU * (x * cyc * 2 + t * 5)));
    } else {
      waveform[i] =
        amp *
        (0.6 * Math.sin(TAU * (x * 3 + t * 0.9)) +
          0.3 * Math.sin(TAU * (x * 11 - t * 2.3)) * (0.5 + kick) +
          0.15 * Math.sin(TAU * (x * 37 + t * 5.1)) * hat);
    }
  }
  const peak = noteHz / 43;
  for (let i = 0; i < 512; i++) {
    const tone = 0.75 * noteEnv * (Math.exp(-((i - peak) ** 2) / 2) + 0.5 * Math.exp(-((i - peak * 2) ** 2) / 3));
    const band = simple ? 0.25 * Math.exp(-i / 60) : (1 - i / 512) * (0.5 + 0.5 * kick) * 0.85;
    spectrum[i] = Math.min(1, Math.max(0, 0.3 + band + tone - Math.random() * 0.06));
  }
  const notePc = ((note % 12) + 12) % 12;
  for (let i = 0; i < 12; i++) {
    const inKey = [0, 2, 4, 5, 7, 9, 11].includes((i - state.keyTonic + 12) % 12);
    const base = simple ? 0.1 : (inKey ? 0.4 : 0.05) + (inKey ? 0.6 : 0) * Math.max(0, Math.sin(t * 1.7 + i * 1.3));
    chroma[i] = i === notePc ? Math.max(base, 0.6 + 0.4 * noteEnv) : base;
  }
}
const TAU = Math.PI * 2;

// ---------------------------------------------------------------- loop
let last = performance.now();
let acc = 0;
let frames = 0;
let lastLog = last;
const w = window as unknown as Record<string, unknown>;
w.__viz = viz;
w.__state = state;
// Dev-only frame statistics (readPixels right after render).
let measureReq: ((v: object) => void) | null = null;
function measure(): object {
  const gl = canvas.getContext('webgl2')!;
  const W = canvas.width, H = canvas.height;
  const px = new Uint8Array(W * H * 4);
  gl.readPixels(0, 0, W, H, gl.RGBA, gl.UNSIGNED_BYTE, px);
  let sum = 0, clip = 0, dark = 0, satSum = 0, n = 0;
  for (let i = 0; i < px.length; i += 4 * 7) {
    const r = px[i], g = px[i + 1], b = px[i + 2];
    const l = 0.2126 * r + 0.7152 * g + 0.0722 * b;
    const mx = Math.max(r, g, b), mn = Math.min(r, g, b);
    sum += l;
    satSum += mx > 0 ? (mx - mn) / mx : 0;
    if (mx >= 250) clip++;
    if (l < 12) dark++;
    n++;
  }
  return { mean: +(sum / n).toFixed(1), clipPct: +((clip / n) * 100).toFixed(1), darkPct: +((dark / n) * 100).toFixed(1), sat: +(satSum / n).toFixed(2), preset: viz.getPresetName(), section: section.label };
}
w.__measure = () => new Promise((res) => (measureReq = res));
// Measure every preset at a verse and a drop moment: __runSweep([0,1,2]) then read __sweep.
w.__runSweep = async (ids: number[], tv = 9, td = 25) => {
  const sl = (ms: number) => new Promise((r) => setTimeout(r, ms));
  const out: string[] = [];
  w.__sweep = out;
  lockPreset = true;
  viz.setAutoSwitch(false);
  for (const i of ids) {
    viz.selectPreset(i, 0.01);
    songTime = tv;
    await sl(2500);
    const a = (await (w.__measure as () => Promise<Record<string, unknown>>)()) as Record<string, number | string>;
    songTime = td;
    await sl(1500);
    const b = (await (w.__measure as () => Promise<Record<string, unknown>>)()) as Record<string, number | string>;
    out.push(`${i} ${String(a.preset).padEnd(20)} verse m${a.mean} c${a.clipPct} d${a.darkPct} s${a.sat} | drop m${b.mean} c${b.clipPct} d${b.darkPct} s${b.sat}`);
  }
  out.push('done');
};
w.__lock = (v: boolean) => {
  lockPreset = v;
  viz.setAutoSwitch(!v);
};
w.__simple = (v: boolean) => {
  simple = v;
};
viz.setAutoSwitch(!lockPreset);
// Contact sheet: __sheet(['E01','E02'], {t: 9, wait: 3000, simple: false, cols: 4}) snapshots each
// preset after warming up and shows them as a grid overlay (click it to close).
let snapReq: ((v: string) => void) | null = null;
w.__sheet = async (ids: string[], o: { t?: number; wait?: number; simple?: boolean; cols?: number } = {}) => {
  const sl = (ms: number) => new Promise((r) => setTimeout(r, ms));
  lockPreset = true;
  viz.setAutoSwitch(false);
  if (o.simple !== undefined) simple = o.simple;
  const shots: [string, string][] = [];
  for (const id of ids) {
    viz.selectPreset(id, 0.01);
    songTime = o.t ?? 9;
    await sl(o.wait ?? 3000);
    const url = await new Promise<string>((r) => (snapReq = r));
    shots.push([viz.getPresetName(), url]);
  }
  document.getElementById('sheet')?.remove();
  const d = document.createElement('div');
  d.id = 'sheet';
  const cols = o.cols ?? 4;
  d.style.cssText = `position:fixed;inset:0;background:#222;display:grid;grid-template-columns:repeat(${cols},1fr);gap:2px;z-index:9;align-content:start`;
  for (const [name, url] of shots) {
    const f = document.createElement('div');
    f.style.cssText = 'position:relative';
    f.innerHTML = `<img src="${url}" style="width:100%;display:block"><span style="position:absolute;left:3px;top:2px;font:10px monospace;color:#ff0;background:#0008">${name}</span>`;
    d.appendChild(f);
  }
  d.onclick = () => d.remove();
  document.body.appendChild(d);
  return shots.length;
};
w.__seek = (s: number) => {
  songTime = s;
};

function loop(now: number): void {
  const dt = Math.min(0.1, (now - last) / 1000);
  last = now;
  fakeFrame(dt);
  viz.render(state);
  if (snapReq) {
    const r = snapReq;
    snapReq = null;
    r(canvas.toDataURL('image/jpeg', 0.85));
  }
  if (measureReq) {
    const r = measureReq;
    measureReq = null;
    r(measure());
  }
  acc += dt;
  frames++;
  if (now - lastLog > 2000) {
    const ms = (acc / frames) * 1000;
    const s = viz.stats;
    const line = `[render-test] avg frame ${ms.toFixed(2)} ms (${(1000 / ms).toFixed(0)} fps) cpu ${s.cpuMs.toFixed(2)} ms gpu ${s.gpuMs.toFixed(2)} ms | ${s.width}x${s.height} particles ${s.particles} | ${viz.getMode()} | ${viz.getPresetName()} | ${section.label}`;
    console.log(line);
    w.__lastLog = line;
    acc = 0;
    frames = 0;
    lastLog = now;
  }
  hud.textContent = `${viz.getMode()}  ${viz.getPresetName()}\n${section.label}  bar ${state.barIndex}  build ${state.buildIntensity.toFixed(2)}  ${simple ? 'SIMPLE' : 'DENSE'}${lockPreset ? '  LOCK' : ''}`;
  schedule();
}
// rAF normally; when the page is hidden (e.g. an embedded preview pane) fall
// back to a MessageChannel pump paced at ~60 Hz so the harness keeps running.
const pump = new MessageChannel();
pump.port1.onmessage = () => {
  const now = performance.now();
  if (now - last >= 16.6) loop(now);
  else schedule();
};
function schedule(): void {
  if (document.hidden) pump.port2.postMessage(0);
  else requestAnimationFrame(loop);
}
schedule();

window.addEventListener('keydown', (e) => {
  resume();
  if (e.key === '1') viz.setMode('enhanced');
  else if (e.key === '2') viz.setMode('classic');
  else if (e.key === '3') viz.setMode('hybrid');
  else if (e.key === 'n' || e.key === 'N') viz.nextPreset();
  else if (e.key === 'p' || e.key === 'P') playing = !playing;
  else if (e.key === 'h' || e.key === 'H') hud.style.display = hud.style.display === 'none' ? '' : 'none';
  else if (e.key === 's' || e.key === 'S') simple = !simple;
  else if (e.key === 'g' || e.key === 'G') viz.newSong(simple ? 0.15 : 0.9);
  else if (e.key === 'l' || e.key === 'L') {
    lockPreset = !lockPreset;
    viz.setAutoSwitch(!lockPreset);
  }
});
