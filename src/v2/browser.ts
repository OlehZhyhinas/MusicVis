// Preset browser: every member of the population with lineage, votes and a
// lazily rendered still thumbnail. Select two to breed, one to mutate; the
// screened children appear as thumbnails and join the population.

import { SPECIES, SPECIES_LABEL, type Species } from './genome';
import type { Evolution } from './evolve';
import { fitness, type Member } from './population';

export interface BrowserCallbacks {
  play(id: string): void;
  currentId(): string | null;
  toast(msg: string, kind?: 'info' | 'error'): void;
  onOpen?(): void;
  onClose?(): void;
}

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;

export class PresetBrowser {
  private root = $<HTMLElement>('v2-browser');
  private list = $<HTMLElement>('v2b-list');
  private typeSel = $<HTMLSelectElement>('v2b-type');
  private energySel = $<HTMLSelectElement>('v2b-energy');
  private sortSel = $<HTMLSelectElement>('v2b-sort');
  private hiddenChk = $<HTMLInputElement>('v2b-hidden');
  private selInfo = $<HTMLElement>('v2b-selected');
  private breedBtn = $<HTMLButtonElement>('v2b-breed');
  private mutateBtn = $<HTMLButtonElement>('v2b-mutate');
  private hideBtn = $<HTMLButtonElement>('v2b-hide');
  private clearBtn = $<HTMLButtonElement>('v2b-clear');
  private results = $<HTMLElement>('v2b-results');
  private resultsGrid = $<HTMLElement>('v2b-results-grid');
  private resultsTitle = $<HTMLElement>('v2b-results-title');
  private selected = new Set<string>();
  private observer: IntersectionObserver;
  private renderQueued = false;
  private highlight: string | null = null;

  constructor(private evo: Evolution, private cb: BrowserCallbacks) {
    this.typeSel.innerHTML = `<option value="">All types</option>` + SPECIES.map((s) => `<option value="${s}">${SPECIES_LABEL[s]}</option>`).join('');
    for (const el of [this.typeSel, this.energySel, this.sortSel, this.hiddenChk]) el.addEventListener('change', () => this.render());
    $('v2b-close').addEventListener('click', () => this.setOpen(false));
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
    $('v2b-reset').addEventListener('click', () => {
      if (!window.confirm('Reset to the 24 seed presets? Every bred preset, vote and lineage is discarded (export first to keep them).')) return;
      void this.evo.reset().then(() => {
        this.selected.clear();
        this.cb.toast('Population reset to the seed presets.');
      });
    });

    this.list.addEventListener('click', (ev) => {
      const t = ev.target as HTMLElement;
      const row = t.closest<HTMLElement>('.v2b-row');
      if (!row) return;
      const id = row.dataset.id!;
      const parent = t.closest<HTMLElement>('[data-parent]');
      if (parent) {
        ev.preventDefault();
        this.reveal(parent.dataset.parent!);
        return;
      }
      if (t.closest('.v2b-sel')) {
        const chk = row.querySelector<HTMLInputElement>('.v2b-sel input')!;
        if (t.tagName !== 'INPUT') chk.checked = !chk.checked;
        if (chk.checked) this.selected.add(id);
        else this.selected.delete(id);
        this.updateSelection();
        return;
      }
      this.cb.play(id);
      this.markCurrent(id);
    });

    this.observer = new IntersectionObserver((entries) => {
      for (const e of entries) {
        if (!e.isIntersecting) continue;
        const img = e.target as HTMLImageElement;
        this.observer.unobserve(img);
        const id = img.dataset.thumb!;
        void this.evo.thumb(id).then((url) => {
          if (url) img.src = url;
          img.classList.toggle('v2b-thumb-failed', !url);
        });
      }
    }, { root: this.list, rootMargin: '120px' });
  }

  get open(): boolean {
    return !this.root.hidden;
  }

  setOpen(open: boolean): void {
    const was = this.open;
    this.root.hidden = !open;
    if (open) {
      this.cb.onOpen?.();
      this.render();
    } else if (was) {
      this.cb.onClose?.();
    }
  }

  toggle(): void {
    this.setOpen(!this.open);
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

  markCurrent(id: string | null): void {
    for (const el of this.list.querySelectorAll<HTMLElement>('.v2b-row')) el.classList.toggle('v2b-current', el.dataset.id === id);
  }

  private reveal(id: string): void {
    const m = this.evo.pop.get(id);
    if (!m) {
      this.cb.toast(`${id} is no longer in the population (culled).`, 'error');
      return;
    }
    this.typeSel.value = '';
    this.energySel.value = '';
    if (m.hidden) this.hiddenChk.checked = true;
    this.highlight = id;
    this.render();
    const row = this.list.querySelector<HTMLElement>(`.v2b-row[data-id="${CSS.escape(id)}"]`);
    row?.scrollIntoView({ block: 'center', behavior: 'smooth' });
  }

  private filtered(): Member[] {
    const type = this.typeSel.value as Species | '';
    const energy = this.energySel.value;
    const showHidden = this.hiddenChk.checked;
    let ms = this.evo.pop.list().filter((m) => (showHidden || !m.hidden) && (!type || m.species === type || m.species2 === type) && (!energy || m.energy === energy));
    const sort = this.sortSel.value;
    if (sort === 'newest') ms = ms.sort((a, b) => b.created - a.created || b.id.localeCompare(a.id));
    else if (sort === 'gen') ms = ms.sort((a, b) => b.gen - a.gen || a.id.localeCompare(b.id));
    else ms = ms.sort((a, b) => fitness(b) - fitness(a) || b.created - a.created);
    return ms;
  }

  render(): void {
    if (!this.open) return;
    const all = this.evo.pop.list();
    for (const id of [...this.selected]) if (!this.evo.pop.get(id)) this.selected.delete(id);
    const ms = this.filtered();
    $('v2b-count').textContent = `${ms.length} / ${all.length}`;
    const cur = this.cb.currentId();
    const frag = document.createDocumentFragment();
    for (const m of ms) frag.appendChild(this.row(m, m.id === cur));
    this.list.replaceChildren(frag);
    for (const img of this.list.querySelectorAll<HTMLImageElement>('img[data-thumb]')) this.observer.observe(img);
    this.updateSelection();
    if (this.highlight) {
      this.list.querySelector(`.v2b-row[data-id="${CSS.escape(this.highlight)}"]`)?.classList.add('v2b-flash');
      this.highlight = null;
    }
  }

  private row(m: Member, current: boolean): HTMLElement {
    const r = document.createElement('div');
    r.className = `v2b-row${current ? ' v2b-current' : ''}${m.hidden ? ' v2b-hidden-row' : ''}`;
    r.dataset.id = m.id;
    r.setAttribute('role', 'listitem');
    const score = fitness(m);
    const parents = m.parents.length
      ? m.parents.map((p) => `<a href="#" data-parent="${esc(p)}">${esc(p)}</a>`).join(' × ')
      : m.origin ? `seed from ${esc(m.origin)}` : '';
    r.innerHTML = `
      <label class="v2b-sel"><input type="checkbox" ${this.selected.has(m.id) ? 'checked' : ''} aria-label="Select ${esc(m.id)}" /></label>
      <img class="v2b-thumb" alt="" data-thumb="${esc(m.id)}" width="96" height="54" />
      <div class="v2b-main">
        <div class="v2b-name"><span class="v2b-id">${esc(m.id)}</span> ${esc(m.name)}</div>
        <div class="v2b-meta"><span class="v2b-type">${esc(m.type)}</span> · ${m.energy} · gen ${m.gen}</div>
        <div class="v2b-lineage">${parents}</div>
      </div>
      <div class="v2b-stats">
        <div class="v2b-score" title="Wilson lower bound of liking">${(score * 100).toFixed(0)}</div>
        <div class="v2b-votes" title="likes / dislikes · views">&#9650;${m.likes} &#9660;${m.dislikes} · ${m.views}v</div>
      </div>`;
    return r;
  }

  private updateSelection(): void {
    const n = this.selected.size;
    this.selInfo.textContent = n ? `${n} selected${n === 2 ? ' · ready to breed' : n === 1 ? ' · ready to mutate' : ''}` : 'Select two to breed, one to mutate';
    this.breedBtn.disabled = n !== 2 || this.evo.breeding > 0;
    this.mutateBtn.disabled = n !== 1 || this.evo.breeding > 0;
    this.hideBtn.disabled = n === 0;
    this.clearBtn.disabled = n === 0;
  }

  private async breed(mode: 'cross' | 'mutate'): Promise<void> {
    const parents = [...this.selected].map((id) => this.evo.pop.get(id)).filter((m): m is Member => !!m);
    if ((mode === 'cross' && parents.length !== 2) || (mode === 'mutate' && parents.length !== 1)) return;
    this.results.hidden = false;
    const label = mode === 'cross' ? `${parents[0].id} × ${parents[1].id}` : `mutants of ${parents[0].id}`;
    this.resultsTitle.textContent = `Breeding ${label}…`;
    const cells: HTMLElement[] = [];
    this.resultsGrid.replaceChildren();
    for (let i = 0; i < 4; i++) {
      const c = document.createElement('div');
      c.className = 'v2b-child v2b-pending';
      c.innerHTML = `<div class="v2b-child-img"></div><div class="v2b-child-label">screening…</div>`;
      this.resultsGrid.appendChild(c);
      cells.push(c);
    }
    let filled = 0;
    let rejected = 0;
    this.updateSelection();
    const children = await this.evo.breed(parents, 4, mode, (e) => {
      if (e.kind === 'reject') {
        rejected++;
        this.resultsTitle.textContent = `Breeding ${label}… ${rejected} rejected (${e.reason})`;
      }
      if (e.kind === 'child' && e.member) {
        const m = e.member;
        const cell = cells[filled++];
        if (!cell) return;
        cell.classList.remove('v2b-pending');
        cell.dataset.id = m.id;
        cell.querySelector('.v2b-child-label')!.innerHTML = `<b>${esc(m.id)}</b> ${esc(m.name)}<br><span>${esc(m.type)}</span>`;
        void this.evo.thumb(m.id).then((url) => {
          const box = cell.querySelector('.v2b-child-img')!;
          box.innerHTML = url ? `<img src="${url}" alt="${esc(m.name)}" width="320" height="180" />` : '<span>no image</span>';
        });
        cell.addEventListener('click', () => {
          this.cb.play(m.id);
          this.setOpen(false);
        });
      }
    });
    for (let i = filled; i < cells.length; i++) {
      cells[i].classList.remove('v2b-pending');
      cells[i].classList.add('v2b-empty');
      cells[i].querySelector('.v2b-child-label')!.textContent = 'no valid child';
    }
    const why = this.evo.lastRejects.length ? ` · rejected ${this.evo.lastRejects.length}: ${summarize(this.evo.lastRejects)}` : '';
    this.resultsTitle.textContent = `${children.length} ${mode === 'cross' ? `children of ${label}` : label}${why}`;
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
      this.cb.toast(`Imported ${n} presets.`);
    } catch (err) {
      this.cb.toast(`Import failed: ${err instanceof Error ? err.message : String(err)}`, 'error');
    }
  }
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
