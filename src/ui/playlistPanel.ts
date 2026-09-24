import type { Playlist, Track } from '../audio/Playlist';
import { icon } from './icons';

export interface PlaylistPanelCallbacks {
  onSelect(id: string): void;
  onRemove(id: string): void;
  onClear(): void;
  onAdd(): void;
}

function formatTime(t: number | null): string {
  if (t === null || !isFinite(t) || t < 0) return '';
  const m = Math.floor(t / 60);
  const s = Math.floor(t % 60);
  return `${m}:${s.toString().padStart(2, '0')}`;
}

function statusText(track: Track): string {
  switch (track.status) {
    case 'queued':
      return 'queued';
    case 'analyzing':
      return `analyzing ${Math.round(track.progress * 100)}%`;
    case 'ready':
      return 'ready';
    case 'error':
      return track.error ?? 'error';
  }
}

export class PlaylistPanel {
  private panelEl: HTMLElement;
  private listEl: HTMLElement;
  private collapsed = true;

  constructor(panelEl: HTMLElement, playlist: Playlist, callbacks: PlaylistPanelCallbacks) {
    this.panelEl = panelEl;
    this.listEl = panelEl.querySelector('#playlist-items')!;

    const addBtn = panelEl.querySelector<HTMLButtonElement>('#pl-add')!;
    const clearBtn = panelEl.querySelector<HTMLButtonElement>('#pl-clear')!;
    const closeBtn = panelEl.querySelector<HTMLButtonElement>('#pl-close')!;

    addBtn.addEventListener('click', () => callbacks.onAdd());
    clearBtn.addEventListener('click', () => callbacks.onClear());
    closeBtn.addEventListener('click', () => this.setCollapsed(true));

    this.listEl.addEventListener('click', (ev) => {
      const target = ev.target as HTMLElement;
      const removeBtn = target.closest<HTMLElement>('.pl-item-remove');
      const item = target.closest<HTMLElement>('.pl-item');
      if (!item) return;
      const id = item.dataset.id;
      if (!id) return;
      if (removeBtn) {
        callbacks.onRemove(id);
      } else {
        callbacks.onSelect(id);
      }
    });

    this.render(playlist);
  }

  get isCollapsed(): boolean {
    return this.collapsed;
  }

  setCollapsed(collapsed: boolean): void {
    this.collapsed = collapsed;
    this.panelEl.hidden = false;
    this.panelEl.classList.toggle('pl-collapsed', collapsed);
  }

  toggle(): void {
    this.setCollapsed(!this.collapsed);
  }

  render(playlist: Playlist): void {
    const current = playlist.currentTrack;
    if (playlist.isEmpty) {
      this.listEl.innerHTML = '';
      return;
    }
    this.listEl.innerHTML = '';
    for (const track of playlist.all) {
      const li = document.createElement('li');
      li.className = `pl-item${current?.id === track.id ? ' pl-current' : ''}`;
      li.dataset.id = track.id;

      const main = document.createElement('div');
      main.className = 'pl-item-main';
      const title = document.createElement('div');
      title.className = 'pl-item-title';
      title.textContent = track.title;
      const status = document.createElement('div');
      status.className = `pl-item-status${track.status === 'error' ? ' pl-error' : ''}`;
      status.textContent = statusText(track);
      main.append(title, status);

      const duration = document.createElement('div');
      duration.className = 'pl-item-duration';
      duration.textContent = formatTime(track.duration);

      const remove = document.createElement('button');
      remove.className = 'pl-item-remove';
      remove.setAttribute('aria-label', `Remove ${track.title}`);
      remove.innerHTML = icon('x', 14);

      li.append(main, duration, remove);
      this.listEl.appendChild(li);
    }
  }
}
