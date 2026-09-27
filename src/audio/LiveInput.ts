// Live audio input (microphone, audio interface, a virtual loopback device
// such as BlackHole, or another browser tab shared with getDisplayMedia)
// captured into the app's AudioContext and analysed in real time. The captured
// signal is never connected to the speakers.

import { LiveAnalyser } from './LiveAnalyser';
import { RealtimeAnalyzer } from '../analysis/RealtimeAnalyzer';
import { RealtimeSampler } from '../analysis/RealtimeSampler';
import { LiveLookahead } from '../analysis/LiveLookahead';
import { BeatRnnOnset, DEFAULT_BEAT_RNN, loadBeatRnn, type BeatRnnModel } from '../analysis/beatRnn';
import { DEFAULT_NEURAL_STEMS, loadStemNet, StemNetSource, type StemNetModel } from '../analysis/stemNet';
import { DEFAULT_NEURAL_DOWNBEATS, LiveDownbeats, loadDownbeat, type DownbeatModel } from '../analysis/downbeatBlstm';

export type LiveDeviceKind = 'default' | 'builtin' | 'virtual' | 'external' | 'test' | 'tab';

export interface LiveDevice {
  deviceId: string;
  label: string;
  kind: LiveDeviceKind;
}

/** Device id used for the dev-only test source (an audio file played through the live path). */
export const TEST_DEVICE_PREFIX = 'test:';

/** Device id of a shared browser tab (not an input device: the stream comes from getDisplayMedia). */
export const TAB_DEVICE_ID = 'tab';

const VIRTUAL_RE = /blackhole|loopback|soundflower|vb-?cable|vb-?audio|voicemeeter|aggregate|multi-?output|background music|ishowu|virtual/i;
const BUILTIN_RE = /built-?in|internal|macbook|imac|mac mini|mac studio|default microphone/i;

function cleanLabel(label: string): string {
  // Drop USB vendor:product suffixes like "(1234:abcd)".
  return label.replace(/\s*\([0-9a-f]{4}:[0-9a-f]{4}\)\s*$/i, '').trim();
}

function kindOf(label: string): LiveDeviceKind {
  if (VIRTUAL_RE.test(label)) return 'virtual';
  if (BUILTIN_RE.test(label)) return 'builtin';
  return 'external';
}

export function liveInputSupported(): boolean {
  return typeof navigator !== 'undefined' && !!navigator.mediaDevices?.getUserMedia && typeof AudioWorkletNode !== 'undefined';
}

/**
 * Audio inputs. Labels are only available once the page has microphone
 * permission; `labelled` tells the UI whether to offer to ask for it.
 */
export async function listInputDevices(): Promise<{ devices: LiveDevice[]; labelled: boolean }> {
  if (!navigator.mediaDevices?.enumerateDevices) return { devices: [], labelled: false };
  const all = await navigator.mediaDevices.enumerateDevices();
  const inputs = all.filter((d) => d.kind === 'audioinput');
  const labelled = inputs.some((d) => d.label);
  const devices: LiveDevice[] = [];
  let n = 0;
  for (const d of inputs) {
    if (d.deviceId === 'communications') continue; // Windows duplicate of a real device
    n++;
    const raw = cleanLabel(d.label);
    if (d.deviceId === 'default' || d.deviceId === '') {
      const inner = raw.replace(/^default\s*-\s*/i, '');
      devices.push({ deviceId: 'default', label: inner ? `System default (${inner})` : 'System default input', kind: 'default' });
    } else {
      devices.push({ deviceId: d.deviceId, label: raw || `Audio input ${n}`, kind: kindOf(raw) });
    }
  }
  if (!devices.some((d) => d.deviceId === 'default')) devices.unshift({ deviceId: 'default', label: 'System default input', kind: 'default' });
  return { devices, labelled };
}

/** Ask for microphone permission once (so device labels become visible), then release it. */
export async function requestInputPermission(): Promise<void> {
  const s = await navigator.mediaDevices.getUserMedia({ audio: true });
  for (const t of s.getTracks()) t.stop();
}

/** Human-readable message for a getUserMedia failure. */
export function describeInputError(err: unknown): string {
  const name = err instanceof DOMException || err instanceof Error ? err.name : '';
  switch (name) {
    case 'NotAllowedError':
    case 'SecurityError':
      return 'Microphone access was blocked. Allow it for this site in the browser settings to use live input.';
    case 'NotFoundError':
    case 'OverconstrainedError':
      return 'That audio input is not available. Is it plugged in?';
    case 'NotReadableError':
    case 'AbortError':
      return 'The audio input is busy or could not be opened. Close other apps using it and try again.';
    default:
      return err instanceof Error && err.message ? `Could not start live input: ${err.message}` : 'Could not start live input.';
  }
}

// ------------------------------------------------------------------ tab capture

/** Why sharing a tab's audio failed; `message` is ready to show. */
export class TabCaptureError extends Error {
  constructor(
    readonly code: 'cancelled' | 'no-audio' | 'unsupported' | 'blocked' | 'failed',
    message: string,
  ) {
    super(message);
    this.name = 'TabCaptureError';
  }
}

/** Chrome, Edge, Opera, Brave, Arc...: the browsers that can share a tab's audio. */
function isChromium(): boolean {
  const brands = (navigator as Navigator & { userAgentData?: { brands?: { brand: string }[] } }).userAgentData?.brands;
  if (brands?.length) return brands.some((b) => /chromium/i.test(b.brand));
  return /\b(Chrome|Chromium|CriOS|Edg)\//.test(navigator.userAgent) && !/Firefox|FxiOS/.test(navigator.userAgent);
}

/** The browser has screen / tab sharing at all (it may still not share audio). */
export function tabCaptureSupported(): boolean {
  return liveInputSupported() && typeof navigator.mediaDevices?.getDisplayMedia === 'function';
}

/** Null when this browser can share a tab's audio, else why not (for the UI). */
export function tabCaptureUnsupportedReason(): string | null {
  if (!tabCaptureSupported()) {
    return window.isSecureContext ? 'This browser cannot share a tab. Use Chrome or Edge on a computer.' : 'Sharing a tab needs a secure (https) page.';
  }
  if (!isChromium()) return 'This browser can share a tab but not its sound. Use Chrome or Edge, or a loopback device above.';
  return null;
}

/**
 * Asks the user to pick a tab and share its audio. Call straight from a click
 * (the browser needs the user gesture). The video track Chrome insists on is
 * stopped at once; only the audio track is kept. The tab keeps playing its
 * own sound as usual. Throws TabCaptureError.
 */
export async function captureTab(): Promise<MediaStream> {
  const why = tabCaptureUnsupportedReason();
  if (why) throw new TabCaptureError('unsupported', why);
  const audio: MediaTrackConstraints & Record<string, unknown> = {
    echoCancellation: false,
    noiseSuppression: false,
    autoGainControl: false,
  };
  // Newer options Chrome understands (older browsers ignore unknown keys).
  const opts: DisplayMediaStreamOptions & Record<string, unknown> = {
    video: { displaySurface: 'browser' } as MediaTrackConstraints,
    audio,
    preferCurrentTab: false,
    selfBrowserSurface: 'exclude',
    systemAudio: 'exclude',
  };
  let stream: MediaStream;
  try {
    stream = await navigator.mediaDevices.getDisplayMedia(opts);
  } catch (err) {
    throw tabError(err);
  }
  const video = stream.getVideoTracks()[0];
  const surface = (video?.getSettings() as MediaTrackSettings & { displaySurface?: string } | undefined)?.displaySurface;
  for (const t of stream.getVideoTracks()) {
    t.stop();
    stream.removeTrack(t);
  }
  const track = stream.getAudioTracks()[0];
  if (!track || track.readyState === 'ended') {
    for (const t of stream.getTracks()) t.stop();
    throw new TabCaptureError(
      'no-audio',
      surface && surface !== 'browser'
        ? 'Only a browser tab can share its sound. Pick a tab (not a window or screen) and tick "Also share tab audio".'
        : 'No sound was shared. Try again and tick "Also share tab audio" in the sharing dialog.',
    );
  }
  return stream;
}

function tabError(err: unknown): TabCaptureError {
  const name = err instanceof DOMException || err instanceof Error ? err.name : '';
  const msg = err instanceof Error ? err.message : '';
  switch (name) {
    case 'NotAllowedError':
      // Chrome reports a dismissed picker as a plain "Permission denied".
      if (/system/i.test(msg)) return new TabCaptureError('blocked', 'Screen sharing is blocked in the system settings. Allow it for this browser and try again.');
      return new TabCaptureError('cancelled', 'Tab sharing was cancelled.');
    case 'SecurityError':
      return new TabCaptureError('blocked', 'Tab sharing is blocked on this page.');
    case 'InvalidStateError':
      return new TabCaptureError('failed', 'The browser did not allow sharing just now. Click the button again.');
    case 'TypeError':
    case 'NotSupportedError':
      return new TabCaptureError('unsupported', 'This browser cannot share a tab\'s sound. Use Chrome or Edge.');
    default:
      return new TabCaptureError('failed', msg ? `Could not share the tab: ${msg}` : 'Could not share the tab.');
  }
}

export interface LiveInputOptions {
  /** An already opened stream (a shared tab); used instead of getUserMedia. */
  stream?: MediaStream;
  /**
   * Seconds the notes trail the sound (0: none). The note tracker uses them
   * as look-ahead (LiveLookahead); the beat clock, hits, stems and levels
   * stay at the sound's time.
   */
  visualLag?: number;
  /**
   * Beat tracking from the beat RNN (src/analysis/beatRnn.ts) instead of the spectral flux
   * (default DEFAULT_BEAT_RNN). Until its weights have loaded, and if they fail to, the flux is used.
   */
  beatRnn?: boolean;
  /**
   * Stem levels from the stem network (src/analysis/stemNet.ts) instead of the DSP stem split
   * (default DEFAULT_NEURAL_STEMS). Until its weights have loaded, and if they fail to, the DSP is used.
   */
  neuralStems?: boolean;
}

export interface LiveInputEvents {
  /** The input stopped on its own (device unplugged, permission revoked, test file ended). */
  onEnded?(reason: string): void;
}

const workletLoaded = new WeakSet<BaseAudioContext>();

let beatRnnModel: Promise<BeatRnnModel> | null = null;
/** The beat RNN weights, fetched once (public/models/beat-lstm.bin, ~0.95 MB). */
function beatRnnWeights(): Promise<BeatRnnModel> {
  if (!beatRnnModel) {
    beatRnnModel = loadBeatRnn(import.meta.env.BASE_URL + 'models/beat-lstm.bin');
    beatRnnModel.catch(() => (beatRnnModel = null));
  }
  return beatRnnModel;
}

let stemNetModel: Promise<StemNetModel> | null = null;
/** The stem network weights, fetched once (public/models/stems.bin). */
function stemNetWeights(): Promise<StemNetModel> {
  if (!stemNetModel) {
    stemNetModel = loadStemNet(import.meta.env.BASE_URL + 'models/stems.bin');
    stemNetModel.catch(() => (stemNetModel = null));
  }
  return stemNetModel;
}

let downbeatModel: Promise<DownbeatModel> | null = null;
/** The downbeat network weights, fetched once (public/models/downbeat-blstm.bin, ~3.2 MB). */
function downbeatWeights(): Promise<DownbeatModel> {
  if (!downbeatModel) {
    downbeatModel = loadDownbeat(import.meta.env.BASE_URL + 'models/downbeat-blstm.bin');
    downbeatModel.catch(() => (downbeatModel = null));
  }
  return downbeatModel;
}

/** Milliseconds per capture block (11.6 ms) the downbeat network may use; its analyses are sliced to fit. */
const DOWNBEAT_BUDGET_MS = 2;

export class LiveInput {
  readonly context: AudioContext;
  readonly analyzer: RealtimeAnalyzer;
  readonly sampler: RealtimeSampler;
  /** MilkDrop-style bass / mid / treb levels, waveform and spectrum of the input. */
  liveAnalyser!: LiveAnalyser;
  readonly device: LiveDevice;

  private stream: MediaStream | null = null;
  private element: HTMLAudioElement | null = null;
  private source: AudioNode | null = null;
  private node: AudioWorkletNode | null = null;
  private sink: GainNode | null = null;
  private monitor: GainNode | null = null;
  /** In front of the waveform / spectrum analyser (no delay: the analyser stays at the sound's time). */
  private lagNode: DelayNode | null = null;
  private lookahead: LiveLookahead | null = null;
  /** The beat RNN onset source while "Neural beats" is on and its weights have loaded. */
  private beatSrc: BeatRnnOnset | null = null;
  /** The downbeat network voting into the beat tracker's bar slot, while neural beats are on. */
  private downbeats: LiveDownbeats | null = null;
  private beatRnnWanted = false;
  /** The stem network while "Neural stems" is on and its weights have loaded. */
  private stemSrc: StemNetSource | null = null;
  private neuralStemsWanted = false;
  private lastMsgAt = 0;
  private stopped = false;
  private readonly events: LiveInputEvents;

  private constructor(ctx: AudioContext, device: LiveDevice, events: LiveInputEvents) {
    this.context = ctx;
    this.device = device;
    this.events = events;
    this.analyzer = new RealtimeAnalyzer(ctx.sampleRate);
    this.sampler = new RealtimeSampler(this.analyzer);
  }

  /** Open a capture device (or, for a `test:` id, an audio file) and start analysing. */
  static async open(ctx: AudioContext, device: LiveDevice, events: LiveInputEvents = {}, opts: LiveInputOptions = {}): Promise<LiveInput> {
    const li = new LiveInput(ctx, device, events);
    if (opts.stream) li.stream = opts.stream; // released by stop() even if opening fails below
    try {
      if (!workletLoaded.has(ctx)) {
        await ctx.audioWorklet.addModule(new URL('./captureWorklet.js?no-inline', import.meta.url));
        workletLoaded.add(ctx);
      }
    } catch (err) {
      li.stop();
      throw err;
    }
    if (opts.stream) {
      const stream = opts.stream;
      for (const t of stream.getAudioTracks()) {
        t.addEventListener('ended', () => li.endedExternally('The tab is no longer shared.'));
      }
      li.source = ctx.createMediaStreamSource(stream);
    } else if (device.deviceId.startsWith(TEST_DEVICE_PREFIX)) {
      const el = new Audio();
      el.src = device.deviceId.slice(TEST_DEVICE_PREFIX.length);
      el.crossOrigin = 'anonymous';
      el.loop = true;
      li.element = el;
      const src = ctx.createMediaElementSource(el);
      li.source = src;
      // The test file is audible (it is not a microphone, so no feedback).
      li.monitor = ctx.createGain();
      li.monitor.gain.value = 0.8;
      src.connect(li.monitor).connect(ctx.destination);
      await el.play();
    } else {
      const audio: MediaTrackConstraints = {
        echoCancellation: false,
        noiseSuppression: false,
        autoGainControl: false,
        channelCount: 2,
      };
      if (device.deviceId && device.deviceId !== 'default') audio.deviceId = { exact: device.deviceId };
      const stream = await navigator.mediaDevices.getUserMedia({ audio });
      li.stream = stream;
      for (const t of stream.getAudioTracks()) {
        t.addEventListener('ended', () => li.endedExternally('The audio input was disconnected.'));
      }
      li.source = ctx.createMediaStreamSource(stream);
    }
    li.wire();
    li.setVisualLag(opts.visualLag ?? 0);
    void li.setBeatRnn(opts.beatRnn ?? DEFAULT_BEAT_RNN);
    void li.setNeuralStems(opts.neuralStems ?? DEFAULT_NEURAL_STEMS);
    return li;
  }

  private wire(): void {
    const ctx = this.context;
    const src = this.source!;
    const node = new AudioWorkletNode(ctx, 'musicvis-capture', {
      numberOfInputs: 1,
      numberOfOutputs: 1,
      outputChannelCount: [1],
      channelCount: 2,
      channelCountMode: 'explicit',
      channelInterpretation: 'speakers',
      processorOptions: { batch: 512 },
    });
    node.port.onmessage = (e: MessageEvent<{ l: Float32Array; r: Float32Array | null }>) => {
      if (this.stopped) return;
      const { l, r } = e.data;
      // The beat RNN hears each block first, so the analyzer's frames of this block find its activations.
      this.beatSrc?.push(l, r ?? l);
      this.stemSrc?.push(l, r ?? l);
      if (this.lookahead) this.lookahead.process(l, r ?? l);
      else this.analyzer.process(l, r ?? l);
      if (this.downbeats) {
        this.downbeats.push(l, r ?? l);
        this.downbeats.work(DOWNBEAT_BUDGET_MS);
      }
      this.lastMsgAt = performance.now();
    };
    src.connect(node);
    // Keep the worklet pulled by the graph with a silent path to the output.
    this.sink = ctx.createGain();
    this.sink.gain.value = 0;
    node.connect(this.sink).connect(ctx.destination);
    this.node = node;
    this.lagNode = ctx.createDelay(1);
    this.lagNode.delayTime.value = 0;
    src.connect(this.lagNode);
    this.liveAnalyser = new LiveAnalyser(ctx, this.lagNode);
  }

  /** Seconds the notes trail the sound. */
  get visualLag(): number {
    return this.lookahead?.lag ?? 0;
  }

  /**
   * Turns the notes' lag (and with it the look-ahead note tracker) on or off
   * while running; everything else is unaffected.
   */
  setVisualLag(lag: number): void {
    lag = Math.max(0, Math.min(0.5, lag));
    if (lag === this.visualLag) return;
    if (this.lookahead) {
      this.lookahead.flush();
      this.lookahead = null;
      this.sampler.noteSource = null;
      this.analyzer.notesLiveOn = true;
    }
    if (lag > 0) {
      this.lookahead = new LiveLookahead(this.analyzer, lag);
      this.sampler.noteSource = this.lookahead.sampleNotes;
    }
  }

  /** Whether the beat clock currently follows the beat RNN. */
  get beatRnn(): boolean {
    return this.beatSrc !== null;
  }

  /**
   * Turns the beat RNN on or off while running (off: the spectral flux drives the beat tracker).
   * Turning it on loads the weights first; the flux keeps the beat until then, or if loading fails.
   */
  async setBeatRnn(on: boolean): Promise<void> {
    this.beatRnnWanted = on;
    if (!on) {
      this.beatSrc = null;
      this.downbeats = null;
      this.analyzer.beatOnset = null;
      this.analyzer.beat.dspAccentWeight = 1;
      return;
    }
    if (DEFAULT_NEURAL_DOWNBEATS && !this.downbeats) void this.startDownbeats();
    if (this.beatSrc) return;
    let model: BeatRnnModel;
    try {
      model = await beatRnnWeights();
    } catch (err) {
      console.warn('Beat RNN unavailable, using the spectral flux:', err);
      return;
    }
    if (!this.beatRnnWanted || this.stopped || this.beatSrc) return;
    this.beatSrc = new BeatRnnOnset(model, this.context.sampleRate, this.analyzer.streamTime);
    this.analyzer.beatOnset = this.beatSrc.at;
  }

  /** Whether the stems.* envelopes currently come from the stem network. */
  get neuralStems(): boolean {
    return this.stemSrc !== null;
  }

  /**
   * Turns the stem network on or off while running (off: the DSP stem split). Turning it on loads
   * the weights first; the DSP keeps the stems until then, or if loading fails.
   */
  async setNeuralStems(on: boolean): Promise<void> {
    this.neuralStemsWanted = on;
    if (!on) {
      this.stemSrc = null;
      this.analyzer.stemSource = null;
      return;
    }
    if (this.stemSrc) return;
    let model: StemNetModel;
    try {
      model = await stemNetWeights();
    } catch (err) {
      console.warn('Stem network unavailable, using the DSP stems:', err);
      return;
    }
    if (!this.neuralStemsWanted || this.stopped || this.stemSrc) return;
    this.stemSrc = new StemNetSource(model, this.context.sampleRate, this.analyzer.streamTime);
    this.analyzer.stemSource = this.stemSrc.at;
  }

  /** Loads the downbeat network (with neural beats); the DSP accents keep the bar until it reports. */
  private async startDownbeats(): Promise<void> {
    let model: DownbeatModel;
    try {
      model = await downbeatWeights();
    } catch (err) {
      console.warn('Downbeat network unavailable, using the onset accents:', err);
      return;
    }
    if (!this.beatRnnWanted || this.stopped || this.downbeats) return;
    this.downbeats = new LiveDownbeats(model, this.context.sampleRate, this.analyzer);
  }

  private endedExternally(reason: string): void {
    if (this.stopped) return;
    this.stop();
    this.events.onEnded?.(reason);
  }

  /** Analysis stream time extrapolated to now (for rendering between audio blocks). */
  streamTimeNow(): number {
    const a = this.analyzer;
    if (this.lastMsgAt === 0) return a.streamTime;
    const ahead = Math.min(0.05, Math.max(0, (performance.now() - this.lastMsgAt) / 1000));
    return a.streamTime + ahead;
  }

  /** Input level for a meter, dBFS (peak, ~300 ms release). */
  get levelDb(): number {
    // Decay the meter when no audio is arriving at all.
    const idle = this.lastMsgAt > 0 ? (performance.now() - this.lastMsgAt) / 1000 : 0;
    return idle > 0.3 ? -120 : this.analyzer.levelDb;
  }

  get active(): boolean {
    return !this.stopped;
  }

  stop(): void {
    if (this.stopped) return;
    this.stopped = true;
    try {
      this.node?.port.postMessage('stop');
    } catch {
      // ignore
    }
    for (const n of [this.source, this.node, this.sink, this.monitor, this.lagNode]) {
      try {
        n?.disconnect();
      } catch {
        // ignore
      }
    }
    if (this.stream) for (const t of this.stream.getTracks()) t.stop();
    if (this.element) {
      this.element.pause();
      this.element.removeAttribute('src');
      this.element.load();
    }
    this.stream = null;
    this.element = null;
  }
}
