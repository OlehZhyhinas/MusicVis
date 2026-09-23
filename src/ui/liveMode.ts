// Live input mode, shared by V1 and V2: the "Live input" buttons, the device
// picker, starting / stopping capture, the transport readout, and the policy
// for telling the visualizer that a new song started.

import type { MusicState } from '../types';
import {
  LiveInput,
  TEST_DEVICE_PREFIX,
  describeInputError,
  listInputDevices,
  liveInputSupported,
  requestInputPermission,
  type LiveDevice,
} from '../audio/LiveInput';
import type { Transport } from './transport';
import { showToast } from './toast';
import { loadSetting, saveSetting } from './storage';

const DEVICE_KEY = 'liveDevice';
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

const KIND_TAG: Record<string, string> = {
  default: '',
  builtin: 'built-in',
  virtual: 'virtual',
  external: 'external',
  test: 'test',
};

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
  private readonly statusEl: HTMLElement;
  private readonly buttons: HTMLElement[];
  private devices: LiveDevice[] = [];
  private selected: string;
  private seenNewSongs = 0;
  private pendingNewSong = false;
  private readonly testUrl: string | null;

  constructor(host: LiveModeHost, transport: Transport, buttons: HTMLElement[]) {
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
    panel.className = 'panel';
    panel.hidden = true;
    panel.setAttribute('role', 'dialog');
    panel.setAttribute('aria-label', 'Live input');
    panel.innerHTML = `
      <div class="lp-header">
        <span class="lp-title">Live input</span>
        <button class="tp-btn lp-close" aria-label="Close">&times;</button>
      </div>
      <div class="lp-list" role="listbox" aria-label="Audio inputs"></div>
      <button class="tp-btn-text lp-grant" hidden>Show device names</button>
      <div class="lp-meter" aria-hidden="true"><div class="lp-meter-fill"></div></div>
      <div class="lp-status" aria-live="polite"></div>
      <div class="lp-actions">
        <button class="tp-btn-text lp-start">Start</button>
        <button class="tp-btn-text lp-stop" hidden>Stop</button>
      </div>
      <p class="lp-help">Visualizes a microphone or audio interface in real time. To visualize audio already playing on this Mac (Spotify, YouTube, a DJ app), route it through a virtual device such as BlackHole and pick it here. Tip: a Multi-Output Device in Audio MIDI Setup lets you hear it at the same time.</p>
    `;
    (document.getElementById('app') ?? document.body).appendChild(panel);
    this.panel = panel;
    this.listEl = panel.querySelector('.lp-list')!;
    this.grantBtn = panel.querySelector('.lp-grant')!;
    this.startBtn = panel.querySelector('.lp-start')!;
    this.stopBtn = panel.querySelector('.lp-stop')!;
    this.meterFill = panel.querySelector('.lp-meter-fill')!;
    this.statusEl = panel.querySelector('.lp-status')!;

    panel.querySelector('.lp-close')!.addEventListener('click', () => this.setOpen(false));
    this.grantBtn.addEventListener('click', () => void this.grant());
    this.startBtn.addEventListener('click', () => void this.start(this.selected));
    this.stopBtn.addEventListener('click', () => this.stop());
    this.listEl.addEventListener('click', (ev) => {
      const item = (ev.target as HTMLElement).closest<HTMLElement>('.lp-item');
      const id = item?.dataset.id;
      if (!id) return;
      this.selected = id;
      this.renderList();
      // Switching devices while live restarts capture on the new one.
      if (this.input) void this.start(id);
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
        this.toggle();
      });
    }
    navigator.mediaDevices?.addEventListener?.('devicechange', () => {
      if (!this.panel.hidden) void this.refresh();
    });
  }

  get active(): boolean {
    return !!this.input;
  }

  /** The running input (diagnostics / tests). */
  get current(): LiveInput | null {
    return this.input;
  }

  get isOpen(): boolean {
    return !this.panel.hidden;
  }

  toggle(): void {
    this.setOpen(!this.isOpen);
  }

  setOpen(open: boolean): void {
    this.panel.hidden = !open;
    for (const b of this.buttons) b.classList.toggle('active', open || this.active);
    if (open) void this.refresh();
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
      this.grantBtn.hidden = labelled;
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
      b.className = 'lp-item';
      b.dataset.id = d.deviceId;
      b.setAttribute('role', 'option');
      const sel = d.deviceId === this.selected;
      b.setAttribute('aria-selected', String(sel));
      if (sel) b.classList.add('selected');
      if (d.deviceId === current) b.classList.add('live');
      const name = document.createElement('span');
      name.className = 'lp-name';
      name.textContent = d.label;
      b.appendChild(name);
      const tag = KIND_TAG[d.kind];
      if (tag) {
        const t = document.createElement('span');
        t.className = `lp-tag lp-tag-${d.kind}`;
        t.textContent = tag;
        b.appendChild(t);
      }
      this.listEl.appendChild(b);
    }
  }

  private renderState(): void {
    const live = !!this.input;
    this.startBtn.hidden = live;
    this.startBtn.disabled = this.starting;
    this.stopBtn.hidden = !live;
    this.panel.classList.toggle('lp-live', live);
    if (!this.starting) this.statusEl.textContent = live ? `Listening to ${this.input!.device.label}` : '';
    for (const b of this.buttons) b.classList.toggle('active', live || !this.panel.hidden);
  }

  private async grant(): Promise<void> {
    try {
      await requestInputPermission();
    } catch (err) {
      showToast(describeInputError(err), 'error', 8000);
    }
    await this.refresh();
  }

  async start(deviceId: string): Promise<void> {
    if (this.starting) return;
    if (!liveInputSupported()) {
      showToast(window.isSecureContext ? 'This browser does not support live audio input.' : 'Live input needs a secure (https) page.', 'error');
      return;
    }
    this.starting = true;
    this.statusEl.textContent = 'Starting…';
    this.renderState();
    const ctx = this.host.ensureContext();
    const device = this.devices.find((d) => d.deviceId === deviceId) ?? { deviceId, label: deviceId === 'default' ? 'System default input' : 'Audio input', kind: 'default' as const };
    try {
      await ctx.resume().catch(() => {});
      const next = await LiveInput.open(ctx, device, {
        onEnded: (reason) => {
          if (this.input !== next) return;
          this.input = null;
          this.finishStop();
          showToast(`${reason} Live input stopped.`, 'error', 7000);
        },
      });
      const prev = this.input;
      this.input = next;
      prev?.stop();
      if (!device.deviceId.startsWith(TEST_DEVICE_PREFIX)) saveSetting(DEVICE_KEY, device.deviceId);
      this.seenNewSongs = 0;
      this.pendingNewSong = false;
      this.transport.setLive(true);
      this.host.onStart();
      this.host.onNewSong(next.analyzer.songComplexity);
      // Device names become available once permission was granted.
      if (!prev) void this.refresh();
      showToast(`Live input: ${device.label}`);
    } catch (err) {
      console.error(err);
      showToast(describeInputError(err), 'error', 8000);
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
      this.meterFill.style.width = `${w}%`;
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
