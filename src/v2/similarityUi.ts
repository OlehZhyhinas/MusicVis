// The similarity page: "Which looks more like this one?" A reference preset
// and two candidates as looping live previews (short strips rendered on the
// reference clip), answered with 1 / 2 or skipped with S. Answers fit the
// fingerprint distance's group weights; the page shows how well the metric
// now agrees with the user.

import type { Member } from './population';
import type { Phenotype } from './phenotype';
import { MIN_ANSWERS_TO_APPLY } from './phenotype';
import type { Fingerprinter } from './fingerprintRender';
import { GROUPS, FEATURES, validFingerprint } from './fingerprint';
import { chooseTriplet, type Triplet } from './similarity';
import { icon } from '../ui/icons';

export interface SimilarityDeps {
  pheno: Phenotype;
  fper: Fingerprinter;
  members(): Member[];
  thumb(id: string): Promise<string>;
  toast(msg: string, detail?: string): void;
  /** Turn the perceptual embedding on / off (persists the choice). */
  setEmbedding(on: boolean): Promise<boolean>;
}

const GROUP_LABEL: Record<string, string> = { colour: 'Colour', detail: 'Detail', structure: 'Structure', motion: 'Motion', response: 'Music response', embedding: 'Perceptual' };
const STRIP_CACHE = 36;

interface Preview {
  id: string;
  frames: HTMLCanvasElement[];
  thumb: HTMLImageElement | null;
}

export class SimilarityPage {
  private root: HTMLElement;
  private canvases: Record<'ref' | 'a' | 'b', HTMLCanvasElement>;
  private caps: Record<'ref' | 'a' | 'b', HTMLElement>;
  private stats: HTMLElement;
  private cur: Triplet | null = null;
  private next: Triplet | null = null;
  private strips = new Map<string, Promise<HTMLCanvasElement[]>>();
  private previews: Record<'ref' | 'a' | 'b', Preview | null> = { ref: null, a: null, b: null };
  private raf = 0;
  private t0 = 0;
  private seen = new Set<string>();
  private refCounts = new Map<string, number>();
  private rng = Math.random;
  private answeredThisVisit = 0;
  private startAgree = NaN;
  private lastFocus: HTMLElement | null = null;

  constructor(private deps: SimilarityDeps) {
    this.root = document.createElement('div');
    this.root.id = 'sim-overlay';
    this.root.className = 'sim';
    this.root.hidden = true;
    this.root.setAttribute('role', 'dialog');
    this.root.setAttribute('aria-modal', 'true');
    this.root.setAttribute('aria-labelledby', 'sim-title');
    this.root.innerHTML = `
      <div class="scrim" data-close></div>
      <div class="sim-card">
        <div class="sim-h">
          <div class="col grow" style="gap:2px"><h2 id="sim-title">Which looks more like this one?</h2>
          <span class="dim">Teach the look metric: your answers weight colour, detail, structure, motion and music response.</span></div>
          <button class="ib" data-close aria-label="Close" title="Close (Esc)">${icon('x', 16)}</button>
        </div>
        <figure class="sim-ref"><canvas width="256" height="144"></canvas><figcaption class="sim-cap" data-cap="ref"></figcaption></figure>
        <div class="sim-pair">
          <button class="sim-cand" data-pick="a" aria-label="The first candidate looks more like it (1)"><canvas width="256" height="144"></canvas><span class="sim-cap"><span class="kbd">1</span><span data-cap="a"></span></span></button>
          <button class="sim-cand" data-pick="b" aria-label="The second candidate looks more like it (2)"><canvas width="256" height="144"></canvas><span class="sim-cap"><span class="kbd">2</span><span data-cap="b"></span></span></button>
        </div>
        <div class="sim-actions">
          <button class="btn sm ghost" data-skip title="Neither, or can't tell">Skip <span class="kbd">S</span></button>
        </div>
        <div class="sim-stats" aria-live="polite"></div>
        <div class="sim-emb">
          <label class="row" style="gap:8px"><button class="tog" data-emb aria-pressed="false" aria-label="Perceptual embedding"></button><b>Perceptual embedding</b></label>
          <span class="dim" data-emb-status></span>
        </div>
      </div>`;
    document.body.appendChild(this.root);
    const cv = this.root.querySelectorAll<HTMLCanvasElement>('canvas');
    this.canvases = { ref: cv[0], a: cv[1], b: cv[2] };
    this.caps = { ref: this.root.querySelector('[data-cap=ref]')!, a: this.root.querySelector('[data-cap=a]')!, b: this.root.querySelector('[data-cap=b]')! };
    this.stats = this.root.querySelector('.sim-stats')!;
    this.root.addEventListener('click', (ev) => {
      const t = ev.target as HTMLElement;
      if (t.closest('[data-close]')) this.close();
      const p = t.closest<HTMLElement>('[data-pick]');
      if (p) this.answer(p.dataset.pick === 'a' ? 'a' : 'b');
      if (t.closest('[data-skip]')) this.skip();
      if (t.closest('[data-emb]')) void this.toggleEmbedding();
    });
    const prev = this.deps.pheno.embedder?.onStatus;
    if (this.deps.pheno.embedder) this.deps.pheno.embedder.onStatus = (st) => {
      prev?.(st);
      if (!this.root.hidden) this.renderStats();
    };
    // Capture phase: while open, 1 / 2 / S / Esc are ours and nothing else sees keys.
    window.addEventListener('keydown', (ev) => {
      if (this.root.hidden) return;
      const k = ev.key;
      if (k === 'Escape') this.close();
      else if (k === '1') this.answer('a');
      else if (k === '2') this.answer('b');
      else if (k === 's' || k === 'S' || k === '0') this.skip();
      else if (k === 'Tab') return; // focus moves inside the dialog
      else return void ev.stopImmediatePropagation();
      ev.preventDefault();
      ev.stopImmediatePropagation();
    }, true);
  }

  get isOpen(): boolean {
    return !this.root.hidden;
  }

  open(): void {
    const ready = this.candidates();
    if (ready.length < 3) {
      this.deps.toast('Not enough fingerprinted presets yet', 'Fingerprints are computed in the background; try again in a minute.');
      return;
    }
    this.lastFocus = document.activeElement as HTMLElement | null;
    this.root.hidden = false;
    this.answeredThisVisit = 0;
    this.startAgree = this.deps.pheno.fit?.agreeFit ?? NaN;
    for (const a of this.deps.pheno.answers) {
      this.seen.add([a.ref, ...[a.a, a.b].sort()].join('|'));
      this.refCounts.set(a.ref, (this.refCounts.get(a.ref) ?? 0) + 1);
    }
    this.cur = null;
    this.next = null;
    this.advance();
    this.renderStats();
    (this.root.querySelector('[data-pick=a]') as HTMLElement).focus();
    this.t0 = performance.now();
    this.loop();
  }

  close(): void {
    this.root.hidden = true;
    cancelAnimationFrame(this.raf);
    this.raf = 0;
    this.lastFocus?.focus?.();
    if (this.answeredThisVisit) {
      const f = this.deps.pheno.fit;
      this.deps.toast(`${this.answeredThisVisit} similarity answer${this.answeredThisVisit === 1 ? '' : 's'} saved`, f && Number.isFinite(f.agreeFit) ? `The look metric now agrees with you ${Math.round(f.agreeFit * 100)}% of the time.` : undefined);
    }
  }

  private candidates(): { id: string; fp: number[] }[] {
    return this.deps.members().filter((m) => validFingerprint(m.fp)).map((m) => ({ id: m.id, fp: m.fp! }));
  }

  private pick(): Triplet | null {
    const ph = this.deps.pheno;
    const w = GROUPS.map((g) => ph.weights[g]);
    const byId = new Map(this.candidates().map((c) => [c.id, c.fp]));
    // Where the hand features and the embedding disagree about which candidate is closer.
    const extraDisagree = ph.embOn
      ? (r: string, a: string, b: string) => {
          const ea = ph.embTerm(r, a), eb = ph.embTerm(r, b);
          if (!Number.isFinite(ea) || !Number.isFinite(eb)) return 0;
          const ha = ph.distance(byId.get(r)!, byId.get(a)!), hb = ph.distance(byId.get(r)!, byId.get(b)!);
          return Math.sign(ea - eb) !== Math.sign(ha - hb) ? 1 : 0;
        }
      : undefined;
    return chooseTriplet(this.candidates(), ph.norm, w, this.rng, { refCounts: this.refCounts, seen: this.seen, extraDisagree });
  }

  private async toggleEmbedding(): Promise<void> {
    const ph = this.deps.pheno;
    const on = !ph.embOn;
    const ok = await this.deps.setEmbedding(on);
    if (on && !ok) this.deps.toast('Perceptual embedding could not load', ph.embedder?.status.detail);
    this.renderStats();
  }

  /** Show the prefetched triplet (or a fresh one) and prefetch the one after. */
  private advance(): void {
    this.cur = this.next ?? this.pick();
    if (!this.cur) {
      this.stats.textContent = 'No more informative comparisons right now.';
      return;
    }
    this.seen.add([this.cur.ref, ...[this.cur.a, this.cur.b].sort()].join('|'));
    const t = this.cur;
    for (const slot of ['ref', 'a', 'b'] as const) this.show(slot, t[slot]);
    this.next = this.pick();
    if (this.next) for (const id of [this.next.ref, this.next.a, this.next.b]) void this.strip(id);
  }

  private strip(id: string): Promise<HTMLCanvasElement[]> {
    let p = this.strips.get(id);
    if (!p) {
      const m = this.deps.members().find((x) => x.id === id);
      p = m ? this.deps.fper.strip(m.genome) : Promise.resolve([]);
      this.strips.set(id, p);
      if (this.strips.size > STRIP_CACHE) this.strips.delete(this.strips.keys().next().value!);
    }
    return p;
  }

  private show(slot: 'ref' | 'a' | 'b', id: string): void {
    const m = this.deps.members().find((x) => x.id === id);
    const pv: Preview = { id, frames: [], thumb: null };
    this.previews[slot] = pv;
    this.caps[slot].innerHTML = m ? `<span class="id">${esc(m.id)}</span> <b>${esc(m.name)}</b> <span class="dim">${esc(m.type)}</span>` : esc(id);
    void this.deps.thumb(id).then((url) => {
      if (!url || this.previews[slot] !== pv) return;
      const img = new Image();
      img.onload = () => (pv.thumb = img);
      img.src = url;
    });
    void this.strip(id).then((frames) => {
      if (this.previews[slot] === pv) pv.frames = frames;
    });
  }

  private loop(): void {
    if (this.root.hidden) return;
    const f = Math.floor(((performance.now() - this.t0) / 1000) * 10);
    for (const slot of ['ref', 'a', 'b'] as const) {
      const pv = this.previews[slot];
      const c = this.canvases[slot];
      const ctx = c.getContext('2d')!;
      if (pv?.frames.length) ctx.drawImage(pv.frames[f % pv.frames.length], 0, 0, c.width, c.height);
      else if (pv?.thumb) ctx.drawImage(pv.thumb, 0, 0, c.width, c.height);
      else {
        ctx.fillStyle = '#0b0c12';
        ctx.fillRect(0, 0, c.width, c.height);
      }
      c.classList.toggle('loading', !pv?.frames.length);
    }
    this.raf = requestAnimationFrame(() => this.loop());
  }

  private answer(pick: 'a' | 'b'): void {
    const t = this.cur;
    if (!t) return;
    const get = (id: string) => this.deps.members().find((m) => m.id === id);
    const ref = get(t.ref), a = get(t.a), b = get(t.b);
    if (ref && a && b) {
      this.deps.pheno.addAnswer(ref, a, b, pick);
      this.refCounts.set(t.ref, (this.refCounts.get(t.ref) ?? 0) + 1);
      this.answeredThisVisit++;
    }
    const btn = this.root.querySelector<HTMLElement>(`.sim-cand[data-pick=${pick}]`);
    btn?.classList.add('chosen');
    setTimeout(() => btn?.classList.remove('chosen'), 180);
    this.advance();
    this.renderStats();
  }

  private skip(): void {
    this.advance();
  }

  private renderStats(): void {
    const ph = this.deps.pheno;
    const f = ph.fit;
    const n = ph.answers.length;
    const pct = (v: number) => (Number.isFinite(v) ? `${Math.round(v * 100)}%` : '–');
    const bars = GROUPS.map((g) => {
      const w = ph.weights[g];
      return `<div class="sim-w" title="${GROUP_LABEL[g]}: weight ${w.toFixed(2)} (${FEATURES.filter((x) => x.group === g).map((x) => x.label).join(', ')})"><span>${GROUP_LABEL[g]}</span><i style="--w:${Math.min(100, (w / 3) * 100)}%"></i><b>${w.toFixed(2)}</b></div>`;
    }).join('');
    let head: string;
    if (!f || n < 4) head = `<b>${n}</b> answer${n === 1 ? '' : 's'} · agreement shows after 4`;
    else {
      const since = Number.isFinite(this.startAgree) && this.answeredThisVisit ? ` (was ${pct(this.startAgree)} when you opened this)` : '';
      head = `<b>${n}</b> answers · the metric agrees with you <b class="sim-agree">${pct(f.agreeFit)}</b>${since} · equal weights ${pct(f.agreeEqual)}`;
    }
    const applied = n >= MIN_ANSWERS_TO_APPLY ? 'Fitted weights are in use for novelty, duplicates and the map.' : `Fitted weights apply after ${MIN_ANSWERS_TO_APPLY} answers.`;
    const embBar = ph.embOn
      ? `<div class="sim-w" title="Perceptual embedding (DINOv2) weight ${ph.embWeight.toFixed(2)}"><span>${GROUP_LABEL.embedding}</span><i style="--w:${Math.min(100, (ph.embWeight / 3) * 100)}%"></i><b>${ph.embWeight.toFixed(2)}</b></div>`
      : '';
    this.stats.innerHTML = `<div>${head}</div><div class="sim-ws">${bars}${embBar}</div><div class="dim">${applied} Agreement is cross-validated (each answer predicted by a fit without it).</div>`;
    this.renderEmb();
  }

  private renderEmb(): void {
    const ph = this.deps.pheno;
    const tog = this.root.querySelector<HTMLButtonElement>('[data-emb]')!;
    tog.setAttribute('aria-pressed', String(ph.embOn));
    const st = ph.embedder?.status;
    const el = this.root.querySelector<HTMLElement>('[data-emb-status]')!;
    const pct = (v: number) => (Number.isFinite(v) ? `${Math.round(v * 100)}%` : '–');
    if (!ph.embOn) {
      el.textContent = st?.state === 'error'
        ? `Could not load: ${st.detail ?? 'unknown error'}`
        : 'DINOv2-small in your browser (WebGPU): judges looks like a vision model. Downloads about 45 MB once; your answers decide how much it counts.';
      return;
    }
    const total = this.deps.members().length;
    const done = total - ph.missingEmbeddings().length;
    const state = st?.state === 'loading' ? `loading${st.progress !== undefined ? ` ${Math.round(st.progress * 100)}%` : ''}…` : st?.state === 'ready' ? st.detail ?? 'ready' : st?.state ?? '';
    const rep = ph.embReport;
    const agree = rep ? ` · on ${rep.n} answers with all three embedded: with it ${pct(rep.with)}, without ${pct(rep.without)}` : ' · agreement with vs without shows after 4 answers on embedded presets';
    el.textContent = `${state} · ${done}/${total} presets embedded${agree}`;
  }
}

function esc(s: string): string {
  return s.replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c]!);
}
