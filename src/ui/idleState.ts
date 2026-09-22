import { STEM_NAMES, type MusicState, type Section } from '../types';

// Produces a gently animated MusicState before any song is loaded, so the
// visualizer has something alive to render behind the start screen.

const IDLE_BPM = 90;
const IDLE_SECTION: Section = { start: 0, end: Infinity, label: 'intro', energy: 0 };

const zeroStems = () => {
  const stems: Record<string, number> = {};
  for (const name of STEM_NAMES) stems[name] = 0;
  return stems as MusicState['stems'];
};

const zeroWaveform = new Float32Array(1024);
const zeroSpectrum = new Float32Array(512);
const zeroChroma = new Float32Array(12);

export function idleState(time: number, dt: number): MusicState {
  const beatDur = 60 / IDLE_BPM;
  const beatPos = time / beatDur;
  const beatIndex = Math.floor(beatPos);
  const beatPhase = beatPos - beatIndex;
  const barPos = beatPos / 4;
  const barIndex = Math.floor(barPos);
  const barPhase = barPos - barIndex;

  // Slow hue drift, full circle every ~2 minutes.
  const keyHue = (time / 120) % 1;

  return {
    time,
    dt,
    playing: false,

    bass: 0,
    mid: 0,
    treb: 0,
    bassAtt: 0,
    midAtt: 0,
    trebAtt: 0,
    waveform: zeroWaveform,
    spectrum: zeroSpectrum,

    bpm: IDLE_BPM,
    beatIndex,
    barIndex,
    beatPhase,
    barPhase,
    beatPulse: Math.max(0, 1 - beatPhase * 6),
    barPulse: Math.max(0, 1 - barPhase * 6),
    onBeat: false,
    onBar: false,

    stems: zeroStems(),
    stemOnsets: zeroStems(),
    loudness: 0,

    chroma: zeroChroma,
    keyTonic: 0,
    keyMode: 'major',
    keyHue,
    keyChangePulse: 0,

    section: IDLE_SECTION,
    sectionIndex: 0,
    sectionProgress: 0,
    sectionChanged: false,
    dropPulse: 0,
    buildIntensity: 0,
  };
}
