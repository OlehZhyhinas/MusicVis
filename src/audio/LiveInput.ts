// Live audio input (microphone, audio interface, a virtual loopback device
// such as BlackHole, or another browser tab shared with getDisplayMedia)
// captured into the app's AudioContext and analysed in real time. The captured
// signal is never connected to the speakers.

import { LiveAnalyser } from './LiveAnalyser';
import { RealtimeAnalyzer } from '../analysis/RealtimeAnalyzer';
import { RealtimeSampler } from '../analysis/RealtimeSampler';
import { LiveLookahead } from '../analysis/LiveLookahead';

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
   * Seconds the visuals trail the sound (0: none). The note tracker uses them
   * as look-ahead (LiveLookahead); every other visual is delayed to match.
   */
  visualLag?: number;
}

export interface LiveInputEvents {
  /** The input stopped on its own (device unplugged, permission revoked, test file ended). */
  onEnded?(reason: string): void;
}

const workletLoaded = new WeakSet<BaseAudioContext>();

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
  /** Delays the waveform / spectrum analyser by the visual lag. */
  private lagNode: DelayNode | null = null;
  private lookahead: LiveLookahead | null = null;
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
      if (this.lookahead) this.lookahead.process(l, r ?? l);
      else this.analyzer.process(l, r ?? l);
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

  /** Seconds the visuals trail the sound. */
  get visualLag(): number {
    return this.lookahead?.lag ?? 0;
  }

  /**
   * Turns the visual lag (and with it the look-ahead note tracker) on or off
   * while running. Turning it on holds the visuals for `lag` seconds while
   * the delay fills; turning it off catches them up at once.
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
    if (this.lagNode) this.lagNode.delayTime.value = lag;
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
