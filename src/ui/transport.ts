// The transport: a full-width seek row (section colours, hover label) above
// now-playing / playback / options. In live mode the seek row becomes the LIVE
// badge, input level, detected section and tempo. It also drives the 2 px
// progress line that stays visible while the UI is auto-hidden.

import type { Section, SectionLabel } from '../types';
import type { RepeatMode } from '../audio/Playlist';
import { setIcon } from './icons';

export interface TransportCallbacks {
  onPlayPause(): void;
  onPrev(): void;
  onNext(): void;
  onShuffleToggle(): void;
  onRepeatCycle(): void;
  onSeek(time: number): void;
  onVolumeChange(v: number): void;
  onMuteToggle(): void;
  onFullscreen(): void;
  onPlaylistToggle(): void;
  onMore(anchor: HTMLElement): void;
  onVolumePopover(anchor: HTMLElement): void;
}

export function formatTime(t: number): string {
  if (!isFinite(t) || t < 0) t = 0;
  const m = Math.floor(t / 60);
  const s = Math.floor(t % 60);
  return `${m}:${s.toString().padStart(2, '0')}`;
}

const q = <T extends HTMLElement>(root: ParentNode, sel: string) => root.querySelector(sel) as T;

export class Transport {
  private root: HTMLElement;
  private playBtn: HTMLButtonElement;
  private seek: HTMLElement;
  private segs: HTMLElement;
  private segsPlayed: HTMLElement;
  private tip: HTMLElement;
  private timeEl: HTMLElement;
  private durationEl: HTMLElement;
  private titleEl: HTMLElement;
  private subEl: HTMLElement;
  private volume: HTMLInputElement;
  private volumePop: HTMLInputElement | null;
  private volumeNum: HTMLElement | null;
  private muteBtns: HTMLElement[];
  private prevBtn: HTMLButtonElement;
  private nextBtn: HTMLButtonElement;
  private shuffleBtn: HTMLButtonElement;
  private repeatBtn: HTMLButtonElement;
  private playlistBtn: HTMLButtonElement;
  private badge: HTMLElement;
  private line: HTMLElement | null;
  private lineSegs: HTMLElement | null;
  private lineWrap: HTMLElement | null;

  private liveRow: HTMLElement;
  private liveMeter: HTMLElement;
  private liveSection: HTMLElement;
  private liveBpm: HTMLElement;
  private liveShown = { level: -1, label: '', bpm: '' };

  private duration = 0;
  private sections: Section[] = [];
  private dragging = false;
  private dragFrac = 0;
  private loading = false;
  private live = false;
  private playing = false;
  private shuffleOn = false;
  private repeatMode: RepeatMode = 'off';
  private lastFrac = -1;
  private trackTitle = '';
  private trackSub = '';

  constructor(root: HTMLElement, private cb: TransportCallbacks) {
    this.root = root;
    this.playBtn = q(root, '#tp-playpause');
    this.seek = q(root, '#tp-seek');
    this.segs = q(root, '#tp-segs');
    this.segsPlayed = q(root, '#tp-segs-played');
    this.tip = q(root, '#tp-seek-tip');
    this.timeEl = q(root, '#tp-time');
    this.durationEl = q(root, '#tp-duration');
    this.titleEl = q(root, '#tp-title');
    this.subEl = q(root, '#tp-sub');
    this.volume = q(root, '#tp-volume');
    this.volumePop = document.getElementById('vol-range') as HTMLInputElement | null;
    this.volumeNum = document.getElementById('vol-num');
    this.muteBtns = [q(root, '#tp-mute'), document.getElementById('vol-mute')].filter((x): x is HTMLElement => !!x);
    this.prevBtn = q(root, '#tp-prev');
    this.nextBtn = q(root, '#tp-next');
    this.shuffleBtn = q(root, '#tp-shuffle');
    this.repeatBtn = q(root, '#tp-repeat');
    this.playlistBtn = q(root, '#tp-playlist');
    this.badge = q(root, '#tp-badge');
    this.liveRow = q(root, '#tp-live');
    this.liveMeter = q(root, '#tp-live-level');
    this.liveSection = q(root, '#tp-live-section');
    this.liveBpm = q(root, '#tp-live-bpm');
    this.lineWrap = document.getElementById('progress-line');
    this.line = this.lineWrap;
    this.lineSegs = document.getElementById('progress-line-segs');

    this.playBtn.addEventListener('click', () => cb.onPlayPause());
    this.prevBtn.addEventListener('click', () => cb.onPrev());
    this.nextBtn.addEventListener('click', () => cb.onNext());
    this.shuffleBtn.addEventListener('click', () => cb.onShuffleToggle());
    this.repeatBtn.addEventListener('click', () => cb.onRepeatCycle());
    this.playlistBtn.addEventListener('click', () => cb.onPlaylistToggle());
    q(root, '#tp-fullscreen').addEventListener('click', () => cb.onFullscreen());
    const more = q(root, '#tp-more');
    more.addEventListener('click', (ev) => {
      ev.stopPropagation();
      cb.onMore(more);
    });
    const volBtn = q(root, '#tp-volbtn');
    volBtn.addEventListener('click', (ev) => {
      ev.stopPropagation();
      cb.onVolumePopover(volBtn);
    });
    for (const b of this.muteBtns) b.addEventListener('click', () => cb.onMuteToggle());
    for (const r of [this.volume, this.volumePop]) r?.addEventListener('input', () => cb.onVolumeChange(Number(r.value)));

    // Seek: pointer drag on the section bar, keys when focused.
    const fracAt = (x: number) => {
      const r = this.seek.getBoundingClientRect();
      return Math.max(0, Math.min(1, (x - r.left) / Math.max(1, r.width)));
    };
    this.seek.addEventListener('pointerdown', (ev) => {
      if (this.loading || this.duration <= 0) return;
      this.dragging = true;
      this.seek.setPointerCapture(ev.pointerId);
      this.dragFrac = fracAt(ev.clientX);
      this.paint(this.dragFrac);
      this.timeEl.textContent = formatTime(this.dragFrac * this.duration);
    });
    this.seek.addEventListener('pointermove', (ev) => {
      const f = fracAt(ev.clientX);
      this.showTip(f);
      if (!this.dragging) return;
      this.dragFrac = f;
      this.paint(f);
      this.timeEl.textContent = formatTime(f * this.duration);
    });
    const end = () => {
      if (!this.dragging) return;
      this.dragging = false;
      if (this.duration > 0) cb.onSeek(this.dragFrac * this.duration);
    };
    this.seek.addEventListener('pointerup', end);
    this.seek.addEventListener('pointercancel', end);
    this.seek.addEventListener('pointerleave', () => {
      if (!this.dragging) this.tip.hidden = true;
    });
    this.seek.addEventListener('keydown', (ev) => {
      if (this.duration <= 0 || this.loading) return;
      const cur = Math.max(0, this.lastFrac) * this.duration;
      let t: number | null = null;
      if (ev.key === 'ArrowLeft' && !ev.shiftKey) t = cur - 5;
      else if (ev.key === 'ArrowRight' && !ev.shiftKey) t = cur + 5;
      else if (ev.key === 'Home') t = 0;
      else if (ev.key === 'End') t = this.duration - 1;
      if (t === null) return;
      ev.preventDefault();
      ev.stopPropagation();
      cb.onSeek(Math.max(0, Math.min(this.duration, t)));
    });
  }

  /** True while the seek knob is held (auto-hide waits). */
  get busy(): boolean {
    return this.dragging;
  }

  private showTip(f: number): void {
    if (this.duration <= 0 || this.live) return;
    const t = f * this.duration;
    const s = this.sections.find((x) => t >= x.start && t < x.end);
    this.tip.hidden = false;
    this.tip.innerHTML = s ? `<span class="dot" style="color:var(--s-${s.label})"></span>${s.label} · ${formatTime(t)}` : formatTime(t);
    this.tip.style.left = `${f * 100}%`;
  }

  private paint(f: number): void {
    const p = `${(f * 100).toFixed(2)}%`;
    this.seek.style.setProperty('--p', p);
    this.lineWrap?.style.setProperty('--p', p);
    this.seek.setAttribute('aria-valuenow', String(Math.round(f * this.duration)));
    this.seek.setAttribute('aria-valuetext', formatTime(f * this.duration));
  }

  show(): void {
    this.root.hidden = false;
    document.getElementById('app')?.classList.add('has-transport');
  }

  hide(): void {
    this.root.hidden = true;
    document.getElementById('app')?.classList.remove('has-transport');
  }

  get visible(): boolean {
    return !this.root.hidden;
  }

  setSections(sections: Section[], duration: number): void {
    this.sections = sections;
    this.duration = duration;
    this.durationEl.textContent = formatTime(duration);
    this.seek.setAttribute('aria-valuemax', String(Math.round(duration)));
    const html = sections
      .map((s) => {
        const e = isFinite(s.end) ? s.end : duration;
        const w = duration > 0 ? Math.max(0, e - s.start) / duration : 0;
        return `<i style="flex:${w.toFixed(4)};--c:var(--s-${s.label})"></i>`;
      })
      .join('');
    this.segs.innerHTML = html;
    this.segsPlayed.innerHTML = html;
    if (this.lineSegs) this.lineSegs.innerHTML = `<div class="segs">${html}</div><div class="segs played">${html}</div>`;
    this.line?.classList.toggle('ready', sections.length > 0);
  }

  setNowPlaying(title: string, sub: string): void {
    this.trackTitle = title;
    this.trackSub = sub;
    if (!this.live) {
      this.titleEl.textContent = title;
      this.titleEl.title = title;
      if (!this.loading) this.subEl.textContent = sub;
    }
  }

  /** Track count shown on the playlist button (null hides it). */
  setPlaylistBadge(n: number | null): void {
    this.badge.hidden = !n;
    this.badge.textContent = n ? String(n) : '';
  }

  setPlaylistOpen(open: boolean): void {
    this.playlistBtn.classList.toggle('on', open);
    this.playlistBtn.setAttribute('aria-pressed', String(open));
  }

  updatePlayback(currentTime: number, duration: number, playing: boolean): void {
    if (this.live) return;
    this.duration = duration;
    if (playing !== this.playing) {
      this.playing = playing;
      setIcon(this.playBtn, playing ? 'pause' : 'play');
      this.playBtn.setAttribute('aria-label', playing ? 'Pause' : 'Play');
    }
    if (!this.dragging && !this.loading) {
      const f = duration > 0 ? currentTime / duration : 0;
      if (Math.abs(f - this.lastFrac) > 0.0004) {
        this.lastFrac = f;
        this.paint(f);
      }
      const txt = formatTime(currentTime);
      if (this.timeEl.textContent !== txt) this.timeEl.textContent = txt;
    }
  }

  /** Shows load/analysis progress (0..1) in place of the time readout, or clears it when null. */
  setTrackLoading(progress: number | null): void {
    const was = this.loading;
    this.loading = progress !== null;
    if (was && !this.loading && !this.live) this.timeEl.textContent = formatTime(Math.max(0, this.lastFrac) * this.duration);
    this.seek.classList.toggle('disabled', this.loading);
    if (progress !== null && !this.live) {
      this.timeEl.textContent = `${Math.round(progress * 100)}%`;
      this.subEl.textContent = `Analyzing ${Math.round(progress * 100)}%`;
    } else if (!this.live) {
      this.subEl.textContent = this.trackSub;
    }
  }

  /**
   * Live-input mode: the seek row gives way to the LIVE badge, input level,
   * detected section and tempo; track controls are disabled and play becomes Stop.
   */
  setLive(on: boolean, device = 'Live input'): void {
    this.live = on;
    this.root.classList.toggle('tp-live', on);
    document.getElementById('app')?.classList.toggle('live-on', on);
    this.liveRow.hidden = !on;
    for (const b of [this.shuffleBtn, this.prevBtn, this.nextBtn, this.repeatBtn]) b.disabled = on;
    if (on) {
      setIcon(this.playBtn, 'stop', 14);
      this.playBtn.setAttribute('aria-label', 'Stop live input');
      this.playBtn.title = 'Stop live input';
      this.titleEl.textContent = device;
      this.subEl.textContent = 'Live input';
      this.liveShown = { level: -1, label: '', bpm: '' };
    } else {
      this.playing = false;
      setIcon(this.playBtn, 'play', 18);
      this.playBtn.setAttribute('aria-label', 'Play');
      this.playBtn.title = 'Play / pause (Space)';
      this.titleEl.textContent = this.trackTitle;
      this.subEl.textContent = this.trackSub;
      this.durationEl.textContent = formatTime(this.duration);
    }
    this.setShuffleUi(this.shuffleOn);
    this.setRepeatUi(this.repeatMode);
  }

  get isLive(): boolean {
    return this.live;
  }

  /** Per-frame live readout (cheap: only touches the DOM when something visible changed). */
  updateLive(levelDb: number, section: SectionLabel, bpm: number, locked: boolean): void {
    if (!this.live) return;
    const level = Math.round(Math.max(0, Math.min(1, (levelDb + 60) / 60)) * 100);
    if (level !== this.liveShown.level) {
      this.liveShown.level = level;
      this.liveMeter.style.setProperty('--v', `${level}%`);
    }
    if (section !== this.liveShown.label) {
      this.liveShown.label = section;
      this.liveSection.textContent = section;
      this.liveSection.style.setProperty('--c', `var(--s-${section})`);
    }
    const b = level <= 0 ? '' : locked ? `${Math.round(bpm)} BPM` : `~${Math.round(bpm)} BPM`;
    if (b !== this.liveShown.bpm) {
      this.liveShown.bpm = b;
      this.liveBpm.textContent = b;
      this.liveBpm.classList.toggle('dim', !locked);
      this.liveBpm.title = locked ? 'Tempo, locked' : 'Tempo, still locking';
    }
  }

  setVolumeUi(v: number, muted: boolean): void {
    const shown = muted ? 0 : v;
    for (const r of [this.volume, this.volumePop]) {
      if (!r) continue;
      r.value = String(shown);
      r.style.setProperty('--v', `${Math.round(shown * 100)}%`);
    }
    if (this.volumeNum) this.volumeNum.textContent = String(Math.round(shown * 100));
    const ic = muted || v === 0 ? 'mute' : 'volume';
    for (const b of [...this.muteBtns, q(this.root, '#tp-volbtn')]) {
      setIcon(b, ic);
      if (b.id !== 'tp-volbtn') {
        b.setAttribute('aria-label', muted ? 'Unmute' : 'Mute');
        b.title = `${muted ? 'Unmute' : 'Mute'} (M)`;
      }
    }
  }

  setShuffleUi(on: boolean): void {
    this.shuffleOn = on;
    this.shuffleBtn.classList.toggle('act', on && !this.live);
    this.shuffleBtn.setAttribute('aria-pressed', String(on));
    this.shuffleBtn.title = `Shuffle ${on ? 'on' : 'off'} (S)`;
    this.shuffleBtn.setAttribute('aria-label', `Shuffle ${on ? 'on' : 'off'}`);
  }

  setRepeatUi(mode: RepeatMode): void {
    this.repeatMode = mode;
    this.repeatBtn.classList.toggle('act', mode !== 'off' && !this.live);
    setIcon(this.repeatBtn, mode === 'one' ? 'repeat1' : 'repeat');
    this.repeatBtn.title = `Repeat: ${mode}`;
    this.repeatBtn.setAttribute('aria-label', `Repeat: ${mode}`);
  }
}
