// Live input mode: the "Live input" buttons, the device
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
import { icon } from './icons';
import type { Popovers } from './popover';

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

const KIND_COLOR: Record<string, string> = {
  virtual: '#B79CFF',
  external: 'var(--warn)',
  test: 'var(--s-build)',
};

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
      <div class="sub lp-grant" hidden><span class="muted">Device names are hidden until the browser grants microphone access.</span><button class="btn sm lp-grant-btn">${icon('eye', 14)}<span>Show device names</span></button></div>
      <div class="row lp-level" hidden><span class="muted">${icon('volume', 14)}</span><div class="meter level h6"><i class="lp-meter-fill" style="--v:0%"></i></div><span class="mono dim lp-db">–</span></div>
      <div class="status lp-status" role="status" aria-live="polite"></div>
      <div class="row"><button class="btn primary lp-start">${icon('play', 16)}<span>Start</span></button><button class="btn danger lp-stop" hidden>${icon('stop', 14)}<span>Stop</span></button><span class="grow"></span><button class="btn ghost lp-cancel">Cancel</button></div>
      <details class="lp-help"><summary>${icon('info', 14)}Play audio from other apps (BlackHole, Loopback)</summary><p>Visualizes a microphone or audio interface in real time. To visualize audio already playing on this Mac (Spotify, YouTube, a DJ app), install a virtual device such as BlackHole, create a Multi-Output Device in Audio MIDI Setup that sends to your speakers and to BlackHole (so you still hear it), pick it as the system output, then choose BlackHole here.</p></details>
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

    panel.querySelector('.lp-close')!.addEventListener('click', () => this.setOpen(false));
    this.cancelBtn.addEventListener('click', () => this.setOpen(false));
    this.grantBtn.addEventListener('click', () => void this.grant());
    this.startBtn.addEventListener('click', () => void this.start(this.selected));
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
    this.panel.classList.toggle('lp-live', live);
    if (!this.starting && liveInputSupported()) {
      if (live) {
        const rate = this.host.ensureContext().sampleRate;
        this.statusEl.innerHTML = `<span class="dot" style="color:var(--live)"></span><span></span>`;
        this.statusEl.lastElementChild!.textContent = `Listening to ${this.input!.device.label} · ${Math.round(rate / 100) / 10} kHz`;
      } else this.statusEl.textContent = 'Pick an input, then Start. Nothing is recorded or uploaded.';
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
          showToast('Live input stopped', 'error', 0, reason);
        },
      });
      const prev = this.input;
      this.input = next;
      prev?.stop();
      if (!device.deviceId.startsWith(TEST_DEVICE_PREFIX)) saveSetting(DEVICE_KEY, device.deviceId);
      this.seenNewSongs = 0;
      this.pendingNewSong = false;
      this.transport.setLive(true, device.label);
      this.host.onStart();
      this.host.onNewSong(next.analyzer.songComplexity);
      // Device names become available once permission was granted.
      if (!prev) void this.refresh();
      showToast(`Live input: ${device.label}`, 'live');
    } catch (err) {
      console.error(err);
      showToast('Live input failed', 'error', 0, describeInputError(err));
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
