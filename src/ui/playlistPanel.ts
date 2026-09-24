// The Playlist tab of the dock: header summary and actions, track rows
// (number or playing bars, title, status, duration, remove on hover).

import type { Playlist, Track } from '../audio/Playlist';
import { icon } from './icons';
import { formatTime } from './transport';
import { lyricStatusLabel } from '../lyrics/library';
import type { LyricStatus } from '../lyrics/types';

export interface PlaylistPanelCallbacks {
  onSelect(id: string): void;
  onRemove(id: string): void;
  onClear(): void;
  onAdd(): void;
  /** The track's lyrics lookup status (shown as a small tag in its row). */
  lyricStatus?(id: string): LyricStatus | undefined;
}

function lyricHtml(s: LyricStatus | undefined): string {
  const l = lyricStatusLabel(s);
  return l ? `<span class="trk-lyr" data-s="${s}" title="${esc(l.title)}">${esc(l.label)}</span>` : '';
}

function esc(s: string): string {
  return s.replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c]!);
}

function statusHtml(track: Track, current: boolean): string {
  switch (track.status) {
    case 'queued':
      return '<span class="dim">Queued</span>';
    case 'analyzing': {
      const p = Math.round(track.progress * 100);
      return `<span style="color:var(--acc)">${icon('loader', 12, 'spin')}</span><span>Analyzing ${p}%</span><div class="meter h3"><i style="--v:${p}%"></i></div>`;
    }
    case 'ready':
      return `<span style="color:var(--ok)">${icon('check', 12)}</span><span>${current ? 'Playing' : 'Ready'}</span>`;
    case 'error':
      return `<span style="color:var(--neg)">${icon('alert', 12)}</span><span class="ell" style="color:var(--neg)">${esc(track.error ?? 'error')}</span>`;
  }
}

export class PlaylistPanel {
  private listEl: HTMLElement;
  private summaryEl: HTMLElement;
  private callbacks: PlaylistPanelCallbacks;

  constructor(panelEl: HTMLElement, playlist: Playlist, callbacks: PlaylistPanelCallbacks) {
    this.callbacks = callbacks;
    this.listEl = panelEl.querySelector('#playlist-items')!;
    this.summaryEl = panelEl.querySelector('#pl-summary')!;
    panelEl.querySelector('#pl-add')!.addEventListener('click', () => callbacks.onAdd());
    panelEl.querySelector('#pl-clear')!.addEventListener('click', () => callbacks.onClear());

    const pick = (ev: Event) => {
      const target = ev.target as HTMLElement;
      const item = target.closest<HTMLElement>('.trk');
      const id = item?.dataset.id;
      if (!id) return;
      if (target.closest('.trk-remove')) callbacks.onRemove(id);
      else callbacks.onSelect(id);
    };
    this.listEl.addEventListener('click', pick);
    this.listEl.addEventListener('keydown', (ev) => {
      if ((ev.key === 'Enter' || ev.key === ' ') && (ev.target as HTMLElement).classList.contains('trk')) {
        ev.preventDefault();
        pick(ev);
      }
    });
    this.render(playlist);
  }

  render(playlist: Playlist): void {
    const current = playlist.currentTrack;
    const all = playlist.all;
    const focusedId = (document.activeElement as HTMLElement | null)?.closest?.<HTMLElement>('.trk')?.dataset.id;
    const scroll = this.listEl.scrollTop;
    this.listEl.innerHTML = all
      .map((t, i) => {
        const cur = current?.id === t.id;
        const lead = cur ? '<span class="bars" aria-hidden="true"><i></i><i></i><i></i></span>' : String(i + 1);
        return `<li class="trk${cur ? ' cur' : ''}" role="listitem" tabindex="0" data-id="${esc(t.id)}"${cur ? ' aria-current="true"' : ''}>
          <span class="lead">${lead}</span>
          <div class="grow"><b>${esc(t.title)}</b><div class="st">${statusHtml(t, cur)}${lyricHtml(this.callbacks.lyricStatus?.(t.id))}</div></div>
          <span class="dur">${t.duration != null && isFinite(t.duration) ? formatTime(t.duration) : '–:––'}</span>
          <button class="ib xs trk-remove" aria-label="Remove ${esc(t.title)}" title="Remove">${icon('x', 14)}</button>
        </li>`;
      })
      .join('');
    this.listEl.scrollTop = scroll;
    if (focusedId) this.listEl.querySelector<HTMLElement>(`.trk[data-id="${CSS.escape(focusedId)}"]`)?.focus();
    const analysed = all.filter((t) => t.status === 'ready').reduce((s, t) => s + (t.duration ?? 0), 0);
    const queued = all.filter((t) => t.status === 'queued').length;
    const parts = [`${all.length} track${all.length === 1 ? '' : 's'}`];
    if (analysed > 0) parts.push(`${formatTime(analysed)} analysed`);
    if (queued) parts.push(`${queued} queued`);
    this.summaryEl.textContent = all.length ? parts.join(' · ') : 'No songs yet';
  }
}
