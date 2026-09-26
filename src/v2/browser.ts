// Preset browser (the dock's Presets tab): every member of the population with
// lineage, votes and a lazily rendered still thumbnail. Select two to breed, one
// to mutate; the screened children appear as thumbnails and join the population.

import { SPECIES, SPECIES_LABEL, type Species } from './genome';
import type { Evolution } from './evolve';
import { fitness, type Member } from './population';
import { SEEDS } from './seeds';
import { icon } from '../ui/icons';

export interface BrowserCallbacks {
  play(id: string): void;
  currentId(): string | null;
  toast(msg: string, kind?: 'info' | 'error' | 'ok', detail?: string): void;
  /** A child was picked: close the browser (the dock). */
  onClose?(): void;
  /** Phenotype novelty (rel 0..1 against the archive), null before the preset is fingerprinted. */
  novelty?(m: Member): { nov: number; rel: number } | null;
}

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;

const TAG_COLOR: Record<NonNullable<Member['cross']>, string> = {
  fused: '#B79CFF',
  morph: 'var(--acc)',
  merged: '#FF9FD2',
  layered: 'var(--warn)',
  edited: '',
};

export class PresetBrowser {
  private root = $<HTMLElement>('v2-browser');
  private list = $<HTMLElement>('v2b-list');
  private scroller = $<HTMLElement>('v2b-scroll');
  private typeSel = $<HTMLSelectElement>('v2b-type');
  private energySeg = $<HTMLElement>('v2b-energy');
  private sortSel = $<HTMLSelectElement>('v2b-sort');
  private hiddenTog = $<HTMLButtonElement>('v2b-hidden');
  private selInfo = $<HTMLElement>('v2b-selected');
  private breedBtn = $<HTMLButtonElement>('v2b-breed');
  private mutateBtn = $<HTMLButtonElement>('v2b-mutate');
  private hideBtn = $<HTMLButtonElement>('v2b-hide');
  private clearBtn = $<HTMLButtonElement>('v2b-clear');
  private results = $<HTMLElement>('v2b-results');
  private resultsGrid = $<HTMLElement>('v2b-results-grid');
  private resultsTitle = $<HTMLElement>('v2b-results-title');
  private resultsStatus = $<HTMLElement>('v2b-results-status');
  private confirmBox = $<HTMLElement>('v2b-confirm');
  private resetBtn = $<HTMLButtonElement>('v2b-reset');
  private energy = '';
  private showHidden = false;
  private selected = new Set<string>();
  private observer: IntersectionObserver;
  private renderQueued = false;
  private highlight: string | null = null;
  private shown = false;

  constructor(private evo: Evolution, private cb: BrowserCallbacks) {
    this.typeSel.innerHTML = `<option value="">All types</option>` + SPECIES.map((s) => `<option value="${s}">${SPECIES_LABEL[s]}</option>`).join('');
    if (cb.novelty) this.sortSel.insertAdjacentHTML('beforeend', '<option value="novel">Most novel</option>');
    for (const el of [this.typeSel, this.sortSel]) el.addEventListener('change', () => this.render());
    this.energySeg.addEventListener('click', (ev) => {
      const b = (ev.target as HTMLElement).closest<HTMLButtonElement>('button[data-v]');
      if (!b) return;
      this.setEnergy(b.dataset.v ?? '');
      this.render();
    });
    this.hiddenTog.addEventListener('click', () => {
      this.setShowHidden(!this.showHidden);
      this.render();
    });
    $('v2b-results-close').addEventListener('click', () => (this.results.hidden = true));
    this.clearBtn.addEventListener('click', () => {
      this.selected.clear();
      this.render();
    });
    this.breedBtn.addEventListener('click', () => void this.breed('cross'));
    this.mutateBtn.addEventListener('click', () => void this.breed('mutate'));
    this.hideBtn.addEventListener('click', () => {
      for (const id of this.selected) {
        const m = this.evo.pop.get(id);
        if (m) m.hidden = !m.hidden;
      }
      this.selected.clear();
      this.evo.changed();
    });
    $('v2b-export').addEventListener('click', () => this.exportFile());
    const imp = $<HTMLInputElement>('v2-import-input');
    $('v2b-import').addEventListener('click', () => imp.click());
    imp.addEventListener('change', () => {
      const f = imp.files?.[0];
      imp.value = '';
      if (f) void this.importFile(f);
    });
    // Reset asks inline (no browser dialog).
    $('v2b-confirm-q').textContent = `Reset the population to the ${SEEDS.length} seed presets?`;
    this.resetBtn.addEventListener('click', () => this.setConfirm(this.confirmBox.hidden === true));
    $('v2b-reset-no').addEventListener('click', () => this.setConfirm(false));
    $('v2b-reset-yes').addEventListener('click', () => {
      this.setConfirm(false);
      void this.evo.reset().then(() => {
        this.selected.clear();
        this.cb.toast('Population reset to the seed presets.', 'ok');
      });
    });
    this.confirmBox.addEventListener('keydown', (ev) => {
      if (ev.key === 'Escape') {
        ev.stopPropagation();
        this.setConfirm(false);
        this.resetBtn.focus();
      }
    });

    this.list.addEventListener('click', (ev) => {
      const t = ev.target as HTMLElement;
      const row = t.closest<HTMLElement>('.prs');
      if (!row) return;
      const id = row.dataset.id!;
      const parent = t.closest<HTMLElement>('[data-parent]');
      if (parent) {
        ev.preventDefault();
        this.reveal(parent.dataset.parent!);
        return;
      }
      const chk = t.closest<HTMLInputElement>('input.chk');
      if (chk) {
        if (chk.checked) this.selected.add(id);
        else this.selected.delete(id);
        row.classList.toggle('sel', chk.checked);
        this.updateSelection();
        return;
      }
      this.cb.play(id);
    });
    this.list.addEventListener('keydown', (ev) => {
      const t = ev.target as HTMLElement;
      if ((ev.key === 'Enter' || ev.key === ' ') && t.classList.contains('prs')) {
        ev.preventDefault();
        this.cb.play(t.dataset.id!);
      }
    });

    this.observer = new IntersectionObserver((entries) => {
      for (const e of entries) {
        if (!e.isIntersecting) continue;
        const img = e.target as HTMLImageElement;
        this.observer.unobserve(img);
        const id = img.dataset.thumb!;
        void this.evo.thumb(id).then((url) => {
          if (url) img.src = url;
          img.classList.toggle('failed', !url);
        });
      }
    }, { root: this.scroller, rootMargin: '120px' });
  }

  get open(): boolean {
    return this.shown;
  }

  /** The Presets tab became visible / hidden (the dock owns that). */
  setOpen(open: boolean): void {
    this.shown = open;
    this.root.hidden = !open;
    if (open) this.render();
    else this.setConfirm(false);
  }

  private setConfirm(on: boolean): void {
    this.confirmBox.hidden = !on;
    this.resetBtn.setAttribute('aria-expanded', String(on));
    this.resetBtn.classList.toggle('active', on);
    if (on) ($('v2b-reset-no') as HTMLButtonElement).focus();
  }

  private setEnergy(v: string): void {
    this.energy = v;
    for (const b of this.energySeg.querySelectorAll<HTMLButtonElement>('button')) b.setAttribute('aria-pressed', String((b.dataset.v ?? '') === v));
  }

  private setShowHidden(on: boolean): void {
    this.showHidden = on;
    this.hiddenTog.setAttribute('aria-pressed', String(on));
  }

  /** Re-render soon (coalesces bursts of population changes). */
  refresh(): void {
    if (!this.open || this.renderQueued) return;
    this.renderQueued = true;
    requestAnimationFrame(() => {
      this.renderQueued = false;
      this.render();
    });
  }

  markCurrent(_id: string | null): void {
    this.refresh();
  }

  private reveal(id: string): void {
    const m = this.evo.pop.get(id);
    if (!m) {
      this.cb.toast(`${id} is no longer in the population (culled).`, 'error');
      return;
    }
    this.typeSel.value = '';
    this.setEnergy('');
    if (m.hidden) this.setShowHidden(true);
    this.highlight = id;
    this.render();
    const row = this.list.querySelector<HTMLElement>(`.prs[data-id="${CSS.escape(id)}"]`);
    row?.scrollIntoView({ block: 'center', behavior: 'smooth' });
  }

  /** The presets the current filters show (the map view uses the same set). */
  members(): Member[] {
    return this.filtered();
  }

  private filtered(): Member[] {
    const type = this.typeSel.value as Species | '';
    const energy = this.energy;
    let ms = this.evo.pop.list().filter((m) => (this.showHidden || !m.hidden) && (!type || m.species === type || m.species2 === type) && (!energy || m.energy === energy));
    const sort = this.sortSel.value;
    if (sort === 'newest') ms = ms.sort((a, b) => b.created - a.created || b.id.localeCompare(a.id));
    else if (sort === 'likes') ms = ms.sort((a, b) => b.likes - a.likes || a.dislikes - b.dislikes || b.created - a.created);
    else if (sort === 'gen') ms = ms.sort((a, b) => b.gen - a.gen || a.id.localeCompare(b.id));
    else if (sort === 'novel') ms = ms.sort((a, b) => (this.cb.novelty?.(b)?.rel ?? -1) - (this.cb.novelty?.(a)?.rel ?? -1) || b.created - a.created);
    else ms = ms.sort((a, b) => fitness(b) - fitness(a) || b.created - a.created);
    return ms;
  }

  render(): void {
    if (!this.open) return;
    const all = this.evo.pop.list();
    for (const id of [...this.selected]) if (!this.evo.pop.get(id)) this.selected.delete(id);
    const ms = this.filtered();
    const classics = SEEDS.filter((s) => s.origin.startsWith('M')).length;
    $('v2b-count').textContent = `${ms.length} shown · ${all.length} in population · ${SEEDS.length} seeds${classics ? ` incl. ${classics} MilkDrop classics` : ''}`;
    const cur = this.cb.currentId();
    const focused = (document.activeElement as HTMLElement | null)?.closest?.<HTMLElement>('.prs')?.dataset.id;
    const keepScroll = this.scroller.scrollTop;
    if (ms.length) {
      const frag = document.createDocumentFragment();
      for (const m of ms) frag.appendChild(this.row(m, m.id === cur));
      this.list.replaceChildren(frag);
      // Re-rendering (thumbnails, votes, views arriving) must never move the list under the user.
      this.scroller.scrollTop = keepScroll;
    } else {
      this.list.innerHTML = '<p class="dim v2b-none">No presets match these filters.</p>';
    }
    for (const img of this.list.querySelectorAll<HTMLImageElement>('img[data-thumb]')) this.observer.observe(img);
    if (focused) this.list.querySelector<HTMLElement>(`.prs[data-id="${CSS.escape(focused)}"]`)?.focus({ preventScroll: true });
    this.updateSelection();
    if (this.highlight) {
      this.list.querySelector(`.prs[data-id="${CSS.escape(this.highlight)}"]`)?.classList.add('flash');
      this.highlight = null;
    }
  }

  private row(m: Member, current: boolean): HTMLElement {
    const sel = this.selected.has(m.id);
    const r = document.createElement('div');
    r.className = `prs${current ? ' cur' : ''}${sel && !current ? ' sel' : ''}${m.hidden ? ' hid' : ''}`;
    r.dataset.id = m.id;
    r.tabIndex = 0;
    r.setAttribute('role', 'listitem');
    r.title = `Play ${m.id}`;
    const score = fitness(m);
    const lineage = m.parents.length
      ? `${icon('link', 12)}${m.parents.map((p) => `<button data-parent="${esc(p)}" title="Show ${esc(p)}">${esc(p)}</button>`).join('×')}`
      : m.origin ? `${icon('link', 12)}<span>seed from ${esc(m.origin)}</span>` : '';
    const tags = [
      m.cross === 'edited' ? `<span class="tag sm" title="${esc(TAG_TITLE.edited)}">edited</span>` : '',
      current ? '<span class="tag sm" style="--c:var(--acc)">playing</span>' : '',
      m.hidden ? '<span class="tag sm" style="--c:var(--tx3)">hidden</span>' : '',
    ].join('');
    r.innerHTML = `
      <input type="checkbox" class="chk" ${sel ? 'checked' : ''} aria-label="Select ${esc(m.id)}" />
      <img class="th" alt="" data-thumb="${esc(m.id)}" width="96" height="54" />
      <div class="col grow" style="gap:4px">
        <div class="row" style="gap:8px"><span class="id">${esc(m.id)}</span><b class="ell">${esc(m.name)}</b>${tags}</div>
        <div class="meta ell">${esc(m.type)} · ${m.energy} · gen ${m.gen}</div>
        <div class="lin">${lineage}</div>
      </div>
      <div class="sc">
        <b title="Score: Wilson lower bound of liking">${(score * 100).toFixed(0)}%</b>
        <div class="votes" title="likes / dislikes · views"><span style="color:var(--ok)">${icon('up', 12)}</span>${m.likes}<span style="color:var(--neg)">${icon('down', 12)}</span>${m.dislikes}<span class="dim v2b-views">· ${m.views}v</span></div>${novBadge(this.cb.novelty?.(m))}
      </div>`;
    return r;
  }

  private updateSelection(): void {
    const n = this.selected.size;
    const b = (t: string) => `<b style="color:var(--tx)">${t}</b>`;
    this.selInfo.innerHTML = n === 0 ? 'Select two to breed, one to mutate' : n === 1 ? `${b('1 selected')} · ready to mutate` : n === 2 ? `${b('2 selected')} · ready to breed` : `${b(`${n} selected`)} · pick two to breed`;
    const busy = this.evo.breeding > 0;
    this.breedBtn.disabled = n !== 2 || busy;
    this.breedBtn.classList.toggle('primary', n === 2 && !busy);
    this.mutateBtn.disabled = n !== 1 || busy;
    this.hideBtn.disabled = n === 0;
    this.clearBtn.disabled = n === 0;
  }

  private async breed(mode: 'cross' | 'mutate'): Promise<void> {
    const parents = [...this.selected].map((id) => this.evo.pop.get(id)).filter((m): m is Member => !!m);
    if ((mode === 'cross' && parents.length !== 2) || (mode === 'mutate' && parents.length !== 1)) return;
    this.results.hidden = false;
    this.scroller.scrollTop = 0;
    const label = mode === 'cross' ? `${parents[0].id} × ${parents[1].id}` : parents[0].id;
    const title = mode === 'cross' ? `Breeding ${label}` : `Mutating ${label}`;
    this.resultsTitle.textContent = `${title}…`;
    this.resultsStatus.innerHTML = `${icon('loader', 13, 'spin')} rendering`;
    const cells: HTMLButtonElement[] = [];
    this.resultsGrid.replaceChildren();
    for (let i = 0; i < 4; i++) {
      const c = document.createElement('button');
      c.className = 'child';
      c.disabled = true;
      c.innerHTML = `<div class="th loading"></div><span class="dim" style="font-size:11px">rendering…</span>`;
      this.resultsGrid.appendChild(c);
      cells.push(c);
    }
    let filled = 0;
    let rejected = 0;
    this.updateSelection();
    const children = await this.evo.breed(parents, 4, mode, (e) => {
      if (e.kind === 'reject') {
        rejected++;
        this.resultsStatus.innerHTML = `${icon('loader', 13, 'spin')} rendering · ${rejected} rejected`;
        this.resultsStatus.title = e.reason ?? '';
      }
      if (e.kind === 'child' && e.member) {
        const m = e.member;
        const cell = cells[filled++];
        if (!cell) return;
        cell.disabled = false;
        cell.dataset.id = m.id;
        cell.title = `Play ${m.id}`;
        const tag = m.cross ? `<span class="tag sm" style="--c:${TAG_COLOR[m.cross] || '#A9ABBD'}" title="${esc(TAG_TITLE[m.cross])}">${m.cross}</span>` : '';
        cell.innerHTML = `<div class="th loading"></div><div class="row" style="gap:6px"><span class="id">${esc(m.id)}</span>${tag}</div><b>${esc(m.name)}</b><span class="dim" style="font-size:11px">${esc(m.type)}</span>`;
        void this.evo.thumb(m.id).then((url) => {
          const box = cell.querySelector('.th')!;
          box.classList.remove('loading');
          box.innerHTML = url ? `<img src="${url}" alt="${esc(m.name)}" width="320" height="180" />` : '<span>no image</span>';
          box.classList.toggle('noimg', !url);
        });
        cell.addEventListener('click', () => {
          this.cb.play(m.id);
          this.cb.onClose?.();
        });
      }
    });
    for (let i = filled; i < cells.length; i++) {
      cells[i].innerHTML = `<div class="th noimg"><span>no valid child</span></div>`;
    }
    const nRej = this.evo.lastRejects.length;
    this.resultsTitle.textContent = title;
    this.resultsStatus.textContent = `${children.length} ${children.length === 1 ? 'child' : 'children'}${nRej ? ` · ${nRej} rejected` : ''}`;
    this.resultsStatus.title = nRej ? `Rejected: ${summarize(this.evo.lastRejects)}` : '';
    this.selected.clear();
    this.render();
  }

  private exportFile(): void {
    const blob = new Blob([this.evo.exportJSON()], { type: 'application/json' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    const d = new Date();
    a.download = `musicvis-v2-population-${d.getFullYear()}${String(d.getMonth() + 1).padStart(2, '0')}${String(d.getDate()).padStart(2, '0')}.json`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(a.href), 2000);
  }

  async importFile(f: File): Promise<void> {
    try {
      const n = await this.evo.importJSON(await f.text());
      this.selected.clear();
      this.cb.toast(`Imported ${n} presets.`, 'ok');
    } catch (err) {
      this.cb.toast('Import failed', 'error', err instanceof Error ? err.message : String(err));
    }
  }
}

const TAG_TITLE: Record<NonNullable<Member['cross']>, string> = {
  fused: "One parent's shape with the other's ideas (placement, motion, material, trail)",
  morph: 'Same shape in both parents: one shape with blended parameters',
  merged: 'Two shapes fused into one through their distance fields',
  layered: 'Rare: a second, separate layer on top',
  edited: 'Edited by hand in the gene editor and saved as a new preset',
};

function novBadge(n: { nov: number; rel: number } | null | undefined): string {
  if (n === undefined) return '';
  if (!n) return '<span class="nov" title="Novelty: not measured yet (fingerprinting in the background)">nov …</span>';
  return `<span class="nov${n.rel >= 0.6 ? ' hi' : ''}" title="Novelty: how unlike every look seen so far this preset is (mean distance ${n.nov.toFixed(2)} to its 10 nearest looks in the archive; 50 = typical)">nov ${Math.round(n.rel * 100)}</span>`;
}

function esc(s: string): string {
  return s.replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c]!);
}

function summarize(reasons: string[]): string {
  const counts = new Map<string, number>();
  for (const r of reasons) {
    const k = r.split(' (')[0];
    counts.set(k, (counts.get(k) ?? 0) + 1);
  }
  return [...counts.entries()].map(([k, n]) => (n > 1 ? `${k} ×${n}` : k)).join(', ');
}
