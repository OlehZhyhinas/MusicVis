// Live input mode: the "Live input" buttons, the device
// picker and the "Listen to a tab" button, starting / stopping capture, the transport readout, and the policy
// for telling the visualizer that a new song started.

import type { MusicState } from '../types';
import {
  LiveInput,
  TAB_DEVICE_ID,
  TEST_DEVICE_PREFIX,
  TabCaptureError,
  captureTab,
  describeInputError,
  listInputDevices,
  liveInputSupported,
  requestInputPermission,
  tabCaptureUnsupportedReason,
  type LiveDevice,
} from '../audio/LiveInput';
import type { Transport } from './transport';
import { showToast } from './toast';
import { loadSetting, saveSetting } from './storage';
import { icon } from './icons';
import type { Popovers } from './popover';
import { DEFAULT_VISUAL_LAG } from '../analysis/LiveLookahead';
import { DEFAULT_BEAT_RNN } from '../analysis/beatRnn';
import { DEFAULT_NEURAL_STEMS } from '../analysis/stemNet';

const DEVICE_KEY = 'liveDevice';
const LAG_KEY = 'liveSharpNotes';
const BEAT_RNN_KEY = 'liveNeuralBeats';
const STEMS_KEY = 'liveNeuralStems';
/** Seconds of music after a long silence before the new song's complexity is trusted. */
const NEW_SONG_MUSIC_S = 6;

export interface LiveModeHost {
  /** The page's AudioContext (created on demand, inside the click gesture). */
  ensureContext(): AudioContext;
  /** Live input started: pause file playback, show the transport. */
  onStart(): void;
  /** Live input stopped (by the user or because the device went away). */
  onStop(): void;
  /** Pick a preset for a new song (live start, or music after a long silence). */
  onNewSong(songComplexity: number): void;
}

const KIND_COLOR: Record<string, string> = {
  virtual: '#B79CFF',
  external: 'var(--warn)',
  test: 'var(--s-build)',
  tab: 'var(--acc)',
};

const KIND_TAG: Record<string, string> = {
  default: '',
  builtin: 'built-in',
  virtual: 'virtual',
  external: 'external',
  test: 'test',
  tab: 'tab',
};

const TAB_DEVICE: LiveDevice = { deviceId: TAB_DEVICE_ID, label: 'Browser tab', kind: 'tab' };
const TAB_HELP = 'Pick the tab and tick "Also share tab audio".';

export class LiveMode {
  private readonly host: LiveModeHost;
  private readonly transport: Transport;
  private input: LiveInput | null = null;
  private starting = false;
  private readonly panel: HTMLElement;
  private readonly listEl: HTMLElement;
  private readonly grantBtn: HTMLButtonElement;
  private readonly startBtn: HTMLButtonElement;
  private readonly stopBtn: HTMLButtonElement;
  private readonly meterFill: HTMLElement;
  private readonly levelRow: HTMLElement;
  private readonly levelDb: HTMLElement;
  private readonly cancelBtn: HTMLButtonElement;
  private readonly grantBox: HTMLElement;
  private anchor: HTMLElement | null = null;
  private readonly statusEl: HTMLElement;
  private readonly buttons: HTMLElement[];
  private devices: LiveDevice[] = [];
  private selected: string;
  private seenNewSongs = 0;
  private pendingNewSong = false;
  private readonly testUrl: string | null;
  private readonly tabBtn: HTMLButtonElement;
  private readonly tabHelp: HTMLElement;
  private readonly lagBtn: HTMLButtonElement;
  private sharpNotes: boolean;
  private readonly beatBtn: HTMLButtonElement;
  private neuralBeats: boolean;
  private readonly stemsBtn: HTMLButtonElement;
  private neuralStems: boolean;
  /** A one-off status line shown while idle (e.g. "Tab sharing was cancelled."). */
  private note = '';

  constructor(host: LiveModeHost, transport: Transport, buttons: HTMLElement[], private popovers: Popovers) {
    this.host = host;
    this.transport = transport;
    this.buttons = buttons;
    this.selected = loadSetting<string>(DEVICE_KEY, 'default');

    // Dev-only: ?liveTest=1 (or ?liveTest=/path/to/file.mp3) adds an audio file
    // played through the live-input path, for testing without a microphone.
    const flag = new URLSearchParams(location.search).get('liveTest');
    this.testUrl = flag ? (flag === '1' ? '/.testdata/dubstep.mp3' : flag) : null;
    if (this.testUrl) (window as unknown as Record<string, unknown>).musicvisLive = this;

    const panel = document.createElement('div');
    panel.id = 'live-picker';
    panel.className = 'lp glass strong pop-layer';
    panel.hidden = true;
    panel.setAttribute('role', 'dialog');
    panel.setAttribute('aria-label', 'Live input');
    panel.innerHTML = `
      <div class="grab"></div>
      <div class="row"><span class="lp-mic">${icon('mic', 18)}</span><b class="grow lp-title">Live input</b><button class="ib sm lp-close" aria-label="Close" title="Close (Esc)">${icon('x', 16)}</button></div>
      <div class="lp-list" role="listbox" aria-label="Input devices"></div>
      <div class="sub lp-tab"><button class="btn sm lp-tab-btn">${icon('music', 14)}<span>Listen to a tab (YouTube, Spotify…)</span></button><span class="muted lp-tab-help">${TAB_HELP}</span></div>
      <div class="sub lp-grant" hidden><span class="muted">Device names are hidden until the browser grants microphone access.</span><button class="btn sm lp-grant-btn">${icon('eye', 14)}<span>Show device names</span></button></div>
      <div class="row lp-level" hidden><span class="muted">${icon('volume', 14)}</span><div class="meter level h6"><i class="lp-meter-fill" style="--v:0%"></i></div><span class="mono dim lp-db">–</span></div>
      <label class="row lp-lag"><button class="tog lp-lag-btn" aria-pressed="false" aria-label="Sharper notes"></button><span class="grow">Sharper notes <span class="muted">· notes trail the sound by ${Math.round(DEFAULT_VISUAL_LAG * 1000)} ms</span></span></label>
      <label class="row lp-lag lp-beat"><button class="tog lp-beat-btn" aria-pressed="false" aria-label="Neural beats"></button><span class="grow">Neural beats <span class="muted">· small networks find the beat and the bar</span></span></label>
      <label class="row lp-lag lp-stems"><button class="tog lp-stems-btn" aria-pressed="false" aria-label="Neural stems"></button><span class="grow">Neural stems <span class="muted">· a small network splits drums, bass, vocals and the rest</span></span></label>
      <div class="status lp-status" role="status" aria-live="polite"></div>
      <div class="row"><button class="btn primary lp-start">${icon('play', 16)}<span>Start</span></button><button class="btn danger lp-stop" hidden>${icon('stop', 14)}<span>Stop</span></button><span class="grow"></span><button class="btn ghost lp-cancel">Cancel</button></div>
      <details class="lp-help"><summary>${icon('info', 14)}Play audio from other apps (BlackHole, Loopback)</summary><p>Visualizes a microphone or audio interface in real time. To visualize audio already playing on this Mac (Spotify, YouTube, a DJ app), install a virtual device such as BlackHole, create a Multi-Output Device in Audio MIDI Setup that sends to your speakers and to BlackHole (so you still hear it), pick it as the system output, then choose BlackHole here. For sound playing in another Chrome or Edge tab, "Listen to a tab" needs none of this.</p></details>
    `;
    (document.getElementById('app') ?? document.body).appendChild(panel);
    this.panel = panel;
    this.listEl = panel.querySelector('.lp-list')!;
    this.grantBox = panel.querySelector('.lp-grant')!;
    this.grantBtn = panel.querySelector('.lp-grant-btn')!;
    this.startBtn = panel.querySelector('.lp-start')!;
    this.stopBtn = panel.querySelector('.lp-stop')!;
    this.cancelBtn = panel.querySelector('.lp-cancel')!;
    this.meterFill = panel.querySelector('.lp-meter-fill')!;
    this.levelRow = panel.querySelector('.lp-level')!;
    this.levelDb = panel.querySelector('.lp-db')!;
    this.statusEl = panel.querySelector('.lp-status')!;
    this.tabBtn = panel.querySelector('.lp-tab-btn')!;
    this.tabHelp = panel.querySelector('.lp-tab-help')!;
    this.lagBtn = panel.querySelector('.lp-lag-btn')!;
    this.sharpNotes = loadSetting<boolean>(LAG_KEY, true);
    this.lagBtn.setAttribute('aria-pressed', String(this.sharpNotes));
    this.lagBtn.title = 'Tracks the melody notes with a little look-ahead: note starts land on time and match the analysed-file quality more closely';
    this.lagBtn.addEventListener('click', (ev) => {
      ev.preventDefault();
      this.sharpNotes = !this.sharpNotes;
      saveSetting(LAG_KEY, this.sharpNotes);
      this.lagBtn.setAttribute('aria-pressed', String(this.sharpNotes));
      this.input?.setVisualLag(this.visualLag);
    });

    this.beatBtn = panel.querySelector('.lp-beat-btn')!;
    this.neuralBeats = loadSetting<boolean>(BEAT_RNN_KEY, DEFAULT_BEAT_RNN);
    this.beatBtn.setAttribute('aria-pressed', String(this.neuralBeats));
    this.beatBtn.title = 'Finds the beat and the first beat of the bar with small recurrent networks (4 MB, run in the page) instead of the plain onset detector: the beat and bar lock on more songs';
    this.beatBtn.addEventListener('click', (ev) => {
      ev.preventDefault();
      this.neuralBeats = !this.neuralBeats;
      saveSetting(BEAT_RNN_KEY, this.neuralBeats);
      this.beatBtn.setAttribute('aria-pressed', String(this.neuralBeats));
      void this.input?.setBeatRnn(this.neuralBeats);
    });

    this.stemsBtn = panel.querySelector('.lp-stems-btn')!;
    this.neuralStems = loadSetting<boolean>(STEMS_KEY, DEFAULT_NEURAL_STEMS);
    this.stemsBtn.setAttribute('aria-pressed', String(this.neuralStems));
    this.stemsBtn.title = 'Follows the drums, bass, vocals and other instruments with a small network trained on a studio stem separator (1 MB, runs in the page) instead of the plain frequency-band split';
    this.stemsBtn.addEventListener('click', (ev) => {
      ev.preventDefault();
      this.neuralStems = !this.neuralStems;
      saveSetting(STEMS_KEY, this.neuralStems);
      this.stemsBtn.setAttribute('aria-pressed', String(this.neuralStems));
      void this.input?.setNeuralStems(this.neuralStems);
    });

    panel.querySelector('.lp-close')!.addEventListener('click', () => this.setOpen(false));
    this.cancelBtn.addEventListener('click', () => this.setOpen(false));
    this.grantBtn.addEventListener('click', () => void this.grant());
    this.startBtn.addEventListener('click', () => void this.start(this.selected));
    this.tabBtn.addEventListener('click', () => void this.start(TAB_DEVICE_ID));
    this.stopBtn.addEventListener('click', () => this.stop());
    this.listEl.addEventListener('click', (ev) => {
      const item = (ev.target as HTMLElement).closest<HTMLElement>('.dev');
      const id = item?.dataset.id;
      if (!id) return;
      this.selected = id;
      this.renderList();
      this.listEl.querySelector<HTMLElement>(`.dev[data-id="${CSS.escape(id)}"]`)?.focus();
      // Switching devices while live restarts capture on the new one.
      if (this.input) void this.start(id);
    });
    this.listEl.addEventListener('keydown', (ev) => {
      if (ev.key !== 'ArrowDown' && ev.key !== 'ArrowUp') return;
      ev.preventDefault();
      const items = [...this.listEl.querySelectorAll<HTMLElement>('.dev')];
      const i = items.indexOf(document.activeElement as HTMLElement);
      items[Math.max(0, Math.min(items.length - 1, i + (ev.key === 'ArrowDown' ? 1 : -1)))]?.focus();
    });
    panel.addEventListener('keydown', (ev) => {
      if (ev.key === 'Escape') {
        ev.stopPropagation();
        this.setOpen(false);
      }
    });
    for (const b of buttons) {
      b.addEventListener('click', (ev) => {
        ev.stopPropagation();
        this.toggle(b);
      });
    }
    navigator.mediaDevices?.addEventListener?.('devicechange', () => {
      if (!this.panel.hidden) void this.refresh();
    });
  }

  get active(): boolean {
    return !!this.input;
  }

  /** Seconds the notes trail the sound: the look-ahead the note tracker gets. */
  private get visualLag(): number {
    return this.sharpNotes ? DEFAULT_VISUAL_LAG : 0;
  }

  /** The running input (diagnostics / tests). */
  get current(): LiveInput | null {
    return this.input;
  }

  get isOpen(): boolean {
    return !this.panel.hidden;
  }

  toggle(anchor?: HTMLElement | null): void {
    this.setOpen(!this.isOpen, anchor);
  }

  /** Opens as a popover above the button that asked (a sheet on phones). */
  setOpen(open: boolean, anchor?: HTMLElement | null): void {
    if (open) {
      this.anchor = anchor && anchor.offsetParent ? anchor : this.defaultAnchor();
      this.popovers.show(this.panel, this.anchor, () => this.renderButtons());
      void this.refresh();
      this.listEl.querySelector<HTMLElement>('.dev[aria-selected=true]')?.focus();
    } else if (this.popovers.isOpen(this.panel)) {
      this.popovers.close();
    }
    this.renderButtons();
  }

  private defaultAnchor(): HTMLElement | null {
    // The transport mic when the transport shows, else the first visible live button.
    return this.buttons.find((b) => b.offsetParent !== null) ?? null;
  }

  private renderButtons(): void {
    const open = this.isOpen;
    for (const b of this.buttons) {
      b.classList.toggle('active', this.active);
      b.classList.toggle('on', open && b === this.anchor && !this.active);
      b.setAttribute('aria-expanded', String(open && b === this.anchor));
    }
  }

  private async refresh(): Promise<void> {
    if (!liveInputSupported()) {
      this.devices = [];
      this.renderList();
      this.statusEl.textContent = window.isSecureContext
        ? 'This browser does not support live audio input.'
        : 'Live input needs a secure (https) page.';
      this.startBtn.disabled = true;
      return;
    }
    try {
      const { devices, labelled } = await listInputDevices();
      this.devices = devices;
      this.grantBox.hidden = labelled;
    } catch {
      this.devices = [{ deviceId: 'default', label: 'System default input', kind: 'default' }];
    }
    if (this.testUrl) {
      const name = this.testUrl.split('/').pop() || this.testUrl;
      this.devices.push({ deviceId: TEST_DEVICE_PREFIX + this.testUrl, label: `Test file: ${name}`, kind: 'test' });
    }
    if (!this.devices.some((d) => d.deviceId === this.selected)) {
      // A remembered device that is not plugged in: fall back to the default for now
      // (keep the remembered id so it is picked again when it comes back).
      const remembered = loadSetting<string>(DEVICE_KEY, 'default');
      this.selected = this.devices.some((d) => d.deviceId === remembered) ? remembered : 'default';
    }
    this.renderList();
    this.renderState();
  }

  private renderList(): void {
    this.listEl.innerHTML = '';
    const current = this.input?.device.deviceId;
    for (const d of this.devices) {
      const b = document.createElement('button');
      b.className = 'dev';
      b.dataset.id = d.deviceId;
      b.setAttribute('role', 'option');
      const sel = d.deviceId === this.selected;
      b.setAttribute('aria-selected', String(sel));
      b.tabIndex = sel ? 0 : -1;
      const tag = KIND_TAG[d.kind];
      const color = KIND_COLOR[d.kind];
      b.innerHTML = `<span class="radio"></span><span class="grow ell lp-name"></span>${tag ? `<span class="tag sm"${color ? ` style="--c:${color}"` : ''}>${tag}</span>` : ''}${d.deviceId === current ? '<span class="dot lp-live-dot" title="Live"></span>' : ''}`;
      b.querySelector('.lp-name')!.textContent = d.label;
      this.listEl.appendChild(b);
    }
    if (!this.listEl.querySelector('[tabindex="0"]')) this.listEl.querySelector<HTMLElement>('.dev')?.setAttribute('tabindex', '0');
  }

  private renderState(): void {
    const live = !!this.input;
    this.startBtn.hidden = live;
    this.startBtn.disabled = this.starting || !liveInputSupported();
    this.stopBtn.hidden = !live;
    this.cancelBtn.hidden = live;
    this.levelRow.hidden = !live;
    const tabWhy = liveInputSupported() ? tabCaptureUnsupportedReason() : null;
    const onTab = this.input?.device.deviceId === TAB_DEVICE_ID;
    this.tabBtn.disabled = this.starting || !!tabWhy || onTab;
    this.tabHelp.textContent = tabWhy ?? (onTab ? 'Listening to the shared tab. Stop sharing in the browser bar or press Stop.' : TAB_HELP);
    this.panel.classList.toggle('lp-live', live);
    if (!this.starting && liveInputSupported()) {
      if (live) {
        const rate = this.host.ensureContext().sampleRate;
        this.statusEl.innerHTML = `<span class="dot" style="color:var(--live)"></span><span></span>`;
        this.statusEl.lastElementChild!.textContent = `Listening to ${this.input!.device.label} · ${Math.round(rate / 100) / 10} kHz`;
      } else this.statusEl.textContent = this.note || 'Pick an input, then Start. Nothing is recorded or uploaded.';
    }
    this.renderButtons();
  }

  private async grant(): Promise<void> {
    try {
      await requestInputPermission();
    } catch (err) {
      showToast('Microphone access failed', 'error', 0, describeInputError(err));
    }
    await this.refresh();
  }

  async start(deviceId: string): Promise<void> {
    if (this.starting) return;
    if (!liveInputSupported()) {
      showToast('Live input unavailable', 'error', 0, window.isSecureContext ? 'This browser does not support live audio input.' : 'Live input needs a secure (https) page.');
      return;
    }
    const tab = deviceId === TAB_DEVICE_ID;
    this.starting = true;
    this.note = '';
    this.statusEl.textContent = tab ? 'Pick a tab in the browser dialog…' : 'Starting…';
    this.renderState();
    const ctx = this.host.ensureContext();
    const device = tab
      ? TAB_DEVICE
      : (this.devices.find((d) => d.deviceId === deviceId) ?? { deviceId, label: deviceId === 'default' ? 'System default input' : 'Audio input', kind: 'default' as const });
    try {
      // The sharing dialog first: it needs the click's user activation.
      const stream = tab ? await captureTab() : undefined;
      await ctx.resume().catch(() => {});
      const next = await LiveInput.open(ctx, device, {
        onEnded: (reason) => {
          if (this.input !== next) return;
          this.input = null;
          this.finishStop();
          // Stopping the share from the browser bar is a normal way to end a tab.
          if (tab) showToast('Stopped listening to the tab', 'info', 5000, reason);
          else showToast('Live input stopped', 'error', 0, reason);
        },
      }, { stream, visualLag: this.visualLag, beatRnn: this.neuralBeats, neuralStems: this.neuralStems });
      const prev = this.input;
      this.input = next;
      prev?.stop();
      if (!tab && !device.deviceId.startsWith(TEST_DEVICE_PREFIX)) saveSetting(DEVICE_KEY, device.deviceId);
      this.seenNewSongs = 0;
      this.pendingNewSong = false;
      this.transport.setLive(true, device.label);
      this.host.onStart();
      this.host.onNewSong(next.analyzer.songComplexity);
      // Device names become available once permission was granted.
      if (!prev) void this.refresh();
      showToast(`Live input: ${device.label}`, 'live');
    } catch (err) {
      if (err instanceof TabCaptureError) {
        // Cancelling the dialog is not an error; the rest are explained where the button is.
        this.note = err.message;
        if (err.code !== 'cancelled') showToast('Could not listen to the tab', 'error', 0, err.message);
      } else {
        console.error(err);
        showToast('Live input failed', 'error', 0, describeInputError(err));
      }
    } finally {
      this.starting = false;
      this.statusEl.textContent = '';
      this.renderList();
      this.renderState();
    }
  }

  stop(): void {
    if (!this.input) return;
    const inp = this.input;
    this.input = null;
    inp.stop();
    this.finishStop();
  }

  private finishStop(): void {
    this.transport.setLive(false);
    this.renderList();
    this.renderState();
    this.host.onStop();
    // Opened from a transport that is gone now: close rather than float unanchored.
    if (this.isOpen && !this.anchor?.offsetParent) this.setOpen(false);
  }

  /**
   * The live MusicState for this frame, or null when live mode is off. Also
   * updates the transport / picker readouts and the new-song policy.
   */
  sample(dt: number): MusicState | null {
    const inp = this.input;
    if (!inp) return null;
    const a = inp.analyzer;
    const state = inp.sampler.sample(inp.streamTimeNow(), dt, true, inp.liveAnalyser.read(dt));
    const level = inp.levelDb;
    this.transport.updateLive(level, state.section.label, state.bpm, a.beat.locked);
    if (!this.panel.hidden) {
      const w = Math.round(Math.max(0, Math.min(1, (level + 60) / 60)) * 100);
      this.meterFill.style.setProperty('--v', `${w}%`);
      const db = Number.isFinite(level) ? `${Math.round(level)} dB` : '–';
      if (this.levelDb.textContent !== db) this.levelDb.textContent = db;
    }
    // Music after a long silence is a new song: once enough of it has been
    // heard to judge its complexity, let the visualizer pick a fitting preset.
    const ns = a.structure.newSongs;
    if (ns !== this.seenNewSongs) {
      this.seenNewSongs = ns;
      this.pendingNewSong = true;
    }
    if (this.pendingNewSong && a.musicSeconds >= NEW_SONG_MUSIC_S) {
      this.pendingNewSong = false;
      this.host.onNewSong(a.songComplexity);
    }
    return state;
  }
}
