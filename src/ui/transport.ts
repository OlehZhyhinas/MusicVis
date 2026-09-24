import type { Section, SectionLabel } from '../types';
import type { RepeatMode } from '../audio/Playlist';

export interface TransportCallbacks {
  onPlayPause(): void;
  onPrev(): void;
  onNext(): void;
  onShuffleToggle(): void;
  onRepeatCycle(): void;
  onSeek(time: number): void;
  onVolumeChange(v: number): void;
  onMuteToggle(): void;
  onNextPreset(): void;
  onFullscreen(): void;
  onHudToggle(): void;
  onPlaylistToggle(): void;
  onHelpToggle(): void;
}

const SECTION_COLORS: Record<SectionLabel, string> = {
  intro: '#5b6b8c',
  verse: '#5f8fd1',
  build: '#c9a24b',
  chorus: '#8bd17f',
  drop: '#ff6b6b',
  breakdown: '#a06bd1',
  outro: '#6b7280',
};

const SEEK_RESOLUTION = 1000;
const AUTO_HIDE_MS = 2500;

function formatTime(t: number): string {
  if (!isFinite(t) || t < 0) t = 0;
  const m = Math.floor(t / 60);
  const s = Math.floor(t % 60);
  return `${m}:${s.toString().padStart(2, '0')}`;
}

export class Transport {
  private root: HTMLElement;
  private appRoot: HTMLElement;
  private playBtn: HTMLButtonElement;
  private seek: HTMLInputElement;
  private seekWrap: HTMLElement;
  private sectionsEl: HTMLElement;
  private seekHover: HTMLElement;
  private timeEl: HTMLElement;
  private durationEl: HTMLElement;
  private volume: HTMLInputElement;
  private muteBtn: HTMLButtonElement;
  private presetBtn: HTMLButtonElement;
  private hudBtn: HTMLButtonElement;
  private fsBtn: HTMLButtonElement;
  private prevBtn: HTMLButtonElement;
  private nextBtn: HTMLButtonElement;
  private shuffleBtn: HTMLButtonElement;
  private repeatBtn: HTMLButtonElement;
  private playlistBtn: HTMLButtonElement;
  private helpBtn: HTMLButtonElement;

  private duration = 0;
  private sections: Section[] = [];
  private dragging = false;
  private hideTimer: ReturnType<typeof setTimeout> | null = null;
  private loading = false;
  private live = false;
  private liveEl: HTMLElement | null = null;
  private liveMeter: HTMLElement | null = null;
  private liveSection: HTMLElement | null = null;
  private liveBpm: HTMLElement | null = null;
  private liveShown = { level: -1, label: '', bpm: '' };

  constructor(root: HTMLElement, appRoot: HTMLElement, callbacks: TransportCallbacks) {
    this.root = root;
    this.appRoot = appRoot;

    this.playBtn = root.querySelector('#tp-playpause')!;
    this.seek = root.querySelector('#tp-seek')!;
    this.seekWrap = root.querySelector('#tp-seek-wrap')!;
    this.sectionsEl = root.querySelector('#tp-sections')!;
    this.seekHover = root.querySelector('#tp-seek-hover')!;
    this.timeEl = root.querySelector('#tp-time')!;
    this.durationEl = root.querySelector('#tp-duration')!;
    this.volume = root.querySelector('#tp-volume')!;
    this.muteBtn = root.querySelector('#tp-mute')!;
    this.presetBtn = root.querySelector('#tp-preset')!;
    this.hudBtn = root.querySelector('#tp-hud')!;
    this.fsBtn = root.querySelector('#tp-fullscreen')!;
    this.prevBtn = root.querySelector('#tp-prev')!;
    this.nextBtn = root.querySelector('#tp-next')!;
    this.shuffleBtn = root.querySelector('#tp-shuffle')!;
    this.repeatBtn = root.querySelector('#tp-repeat')!;
    this.playlistBtn = root.querySelector('#tp-playlist')!;
    this.helpBtn = root.querySelector('#tp-help')!;

    this.playBtn.addEventListener('click', () => callbacks.onPlayPause());
    this.prevBtn.addEventListener('click', () => callbacks.onPrev());
    this.nextBtn.addEventListener('click', () => callbacks.onNext());
    this.shuffleBtn.addEventListener('click', () => callbacks.onShuffleToggle());
    this.repeatBtn.addEventListener('click', () => callbacks.onRepeatCycle());
    this.playlistBtn.addEventListener('click', () => callbacks.onPlaylistToggle());
    this.helpBtn.addEventListener('click', () => callbacks.onHelpToggle());

    this.seek.addEventListener('mousedown', () => {
      this.dragging = true;
    });
    this.seek.addEventListener('input', () => {
      if (this.duration > 0) {
        this.timeEl.textContent = formatTime(this.fractionToTime(Number(this.seek.value)));
      }
    });
    const commitSeek = () => {
      this.dragging = false;
      if (this.duration > 0) {
        callbacks.onSeek(this.fractionToTime(Number(this.seek.value)));
      }
    };
    this.seek.addEventListener('change', commitSeek);
    this.seek.addEventListener('mouseup', commitSeek);

    this.seekWrap.addEventListener('mousemove', (ev) => this.onSeekHover(ev));
    this.seekWrap.addEventListener('mouseleave', () => {
      this.seekHover.hidden = true;
    });

    this.volume.addEventListener('input', () => callbacks.onVolumeChange(Number(this.volume.value)));
    this.muteBtn.addEventListener('click', () => callbacks.onMuteToggle());
    this.presetBtn.addEventListener('click', () => callbacks.onNextPreset());
    this.hudBtn.addEventListener('click', () => callbacks.onHudToggle());
    this.fsBtn.addEventListener('click', () => callbacks.onFullscreen());

    this.installAutoHide();
  }

  private fractionToTime(v: number): number {
    return (v / SEEK_RESOLUTION) * this.duration;
  }

  private onSeekHover(ev: MouseEvent): void {
    const rect = this.seekWrap.getBoundingClientRect();
    const frac = Math.max(0, Math.min(1, (ev.clientX - rect.left) / rect.width));
    const t = frac * this.duration;
    const section = this.sections.find((s) => t >= s.start && t < s.end);
    this.seekHover.hidden = false;
    this.seekHover.style.left = `${frac * 100}%`;
    this.seekHover.textContent = section ? `${section.label} · ${formatTime(t)}` : formatTime(t);
  }

  private installAutoHide(): void {
    const reveal = () => {
      this.root.classList.remove('tp-hidden');
      this.appRoot.classList.remove('cursor-hidden');
      if (this.hideTimer !== null) clearTimeout(this.hideTimer);
      this.hideTimer = setTimeout(() => {
        if (!this.dragging) {
          this.root.classList.add('tp-hidden');
          this.appRoot.classList.add('cursor-hidden');
        }
      }, AUTO_HIDE_MS);
    };
    document.addEventListener('mousemove', reveal);
    document.addEventListener('touchstart', reveal, { passive: true });
    reveal();
  }

  show(): void {
    this.root.hidden = false;
  }

  hide(): void {
    this.root.hidden = true;
  }

  setSections(sections: Section[], duration: number): void {
    this.sections = sections;
    this.duration = duration;
    this.durationEl.textContent = formatTime(duration);
    this.sectionsEl.innerHTML = '';
    for (const s of sections) {
      const end = isFinite(s.end) ? s.end : duration;
      const widthPct = duration > 0 ? ((end - s.start) / duration) * 100 : 0;
      const seg = document.createElement('div');
      seg.className = 'tp-section-seg';
      seg.style.width = `${widthPct}%`;
      seg.style.background = SECTION_COLORS[s.label] ?? '#666';
      seg.title = s.label;
      this.sectionsEl.appendChild(seg);
    }
  }

  updatePlayback(currentTime: number, duration: number, playing: boolean): void {
    if (this.live) return;
    this.duration = duration;
    this.playBtn.innerHTML = playing ? '&#10074;&#10074;' : '&#9658;';
    if (!this.dragging && !this.loading) {
      this.seek.value = duration > 0 ? String((currentTime / duration) * SEEK_RESOLUTION) : '0';
      this.timeEl.textContent = formatTime(currentTime);
    }
  }

  /** Shows load/analysis progress (0..1) in place of the time readout, or clears it when null. */
  setTrackLoading(progress: number | null): void {
    this.loading = progress !== null;
    this.seek.disabled = this.loading;
    if (progress !== null && !this.live) {
      this.timeEl.textContent = `${Math.round(progress * 100)}%`;
    }
  }

  /**
   * Live-input mode: the seek bar / section strip give way to a live indicator
   * (input level, current section, tempo) and the time readout says "Live".
   */
  setLive(on: boolean): void {
    this.live = on;
    this.root.classList.toggle('tp-live', on);
    if (on && !this.liveEl) {
      const el = document.createElement('div');
      el.id = 'tp-live-info';
      el.innerHTML =
        '<span class="tp-live-badge">LIVE</span>' +
        '<span class="tp-live-meter" title="Input level"><span class="tp-live-meter-fill"></span></span>' +
        '<span class="tp-live-section" title="Section (detected live)"></span>' +
        '<span class="tp-live-bpm" title="Tempo (detected live)"></span>';
      this.seekWrap.appendChild(el);
      this.liveEl = el;
      this.liveMeter = el.querySelector('.tp-live-meter-fill');
      this.liveSection = el.querySelector('.tp-live-section');
      this.liveBpm = el.querySelector('.tp-live-bpm');
    }
    if (on) {
      this.timeEl.textContent = 'Live';
      this.durationEl.textContent = '';
      this.playBtn.innerHTML = '&#9632;';
      this.playBtn.setAttribute('aria-label', 'Stop live input');
      this.playBtn.title = 'Stop live input';
      this.liveShown = { level: -1, label: '', bpm: '' };
    } else {
      this.playBtn.setAttribute('aria-label', 'Play/Pause');
      this.playBtn.title = '';
      this.durationEl.textContent = formatTime(this.duration);
      this.timeEl.textContent = formatTime(0);
    }
  }

  get isLive(): boolean {
    return this.live;
  }

  /** Per-frame live readout (cheap: only touches the DOM when something visible changed). */
  updateLive(levelDb: number, section: SectionLabel, bpm: number, locked: boolean): void {
    if (!this.live || !this.liveMeter || !this.liveSection || !this.liveBpm) return;
    const level = Math.round(Math.max(0, Math.min(1, (levelDb + 60) / 60)) * 100);
    if (level !== this.liveShown.level) {
      this.liveShown.level = level;
      this.liveMeter.style.width = `${level}%`;
      this.liveMeter.classList.toggle('hot', levelDb > -3);
    }
    if (section !== this.liveShown.label) {
      this.liveShown.label = section;
      this.liveSection.textContent = section;
      this.liveSection.style.background = SECTION_COLORS[section] ?? '#666';
    }
    const b = level <= 0 ? '' : locked ? `${Math.round(bpm)} BPM` : `~${Math.round(bpm)} BPM`;
    if (b !== this.liveShown.bpm) {
      this.liveShown.bpm = b;
      this.liveBpm.textContent = b;
      this.liveBpm.classList.toggle('tp-live-unlocked', !locked);
    }
  }

  setVolumeUi(v: number, muted: boolean): void {
    this.volume.value = String(v);
    this.muteBtn.innerHTML = muted || v === 0 ? '&#128263;' : '&#128266;';
  }

  setShuffleUi(on: boolean): void {
    this.shuffleBtn.classList.toggle('active', on);
  }

  setRepeatUi(mode: RepeatMode): void {
    this.repeatBtn.classList.toggle('active', mode !== 'off');
    this.repeatBtn.innerHTML = mode === 'one' ? '&#128257;1' : '&#128257;';
    this.repeatBtn.title = `Repeat: ${mode}`;
  }
}
