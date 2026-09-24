// Gene editor (the dock's Genes tab): the playing preset's genes as live
// controls generated from the parameter schemas (geneEdit.ts). Edits go to a
// scratch copy of the genome: parameter changes are swapped into the running
// program in place (uniforms, no recompile); structural changes (a kind switch,
// an op added, a new structure from a slider) go through the program cache,
// debounced while a slider is dragged, and a failed compile falls back to the
// last genome that worked.

import {
  COST_BUDGET_MS, DRAW_OPS, FLAME_VARIATIONS, OP_KINDS, SIGNALS, cloneGenome, estimateCost, repair, schemaFor, structuralKey,
  type FlameVar, type Genome, type OpKind,
} from './genome';
import * as E from './geneEdit';
import type { Engine } from './engine';
import type { Evolution } from './evolve';
import type { Member } from './population';
import { loadSetting, saveSetting } from '../ui/storage';
import { icon, type IconName } from '../ui/icons';

export interface GeneEditorDeps {
  eng: Engine;
  evo: Evolution;
  /** Show a newly saved member (same picture, so in place). */
  adopt: (m: Member) => void;
  /** Unsaved edits appeared / went away. */
  onDirty: (dirty: boolean) => void;
}

interface Bound {
  el: HTMLElement;
  sync: () => void;
}

type Attrs = Record<string, string | number | boolean | undefined>;
function h<K extends keyof HTMLElementTagNameMap>(tag: K, attrs: Attrs = {}, ...kids: (Node | string | null | undefined)[]): HTMLElementTagNameMap[K] {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v === undefined || v === false) continue;
    if (k === 'class') el.className = String(v);
    else if (k === 'text') el.textContent = String(v);
    else el.setAttribute(k, v === true ? '' : String(v));
  }
  for (const c of kids) if (c !== null && c !== undefined) el.append(c);
  return el;
}

const DRAG_DEBOUNCE_MS = 220;
const METER_EVERY_S = 0.1;

export class GeneEditor {
  private member: Member | null = null;
  private original: Genome | null = null;
  private originalJson = '';
  private scratch: Genome | null = null;
  private lastGood: Genome | null = null;
  /** Structure last handed to the engine. */
  private liveKey = '';
  private watch: { key: string; genome: Genome; anchor: string; fallback: Genome } | null = null;
  private debounce = 0;
  private dirtyNow = false;
  private saving = false;
  private shown = false;
  private forced = false;
  private openState = new Map<string, boolean>(Object.entries(loadSetting<Record<string, boolean>>('v2.genesOpen', {})));
  private msgText = '';
  private errors = new Map<string, string>();
  private bound: Bound[] = [];
  private meters: { src: HTMLElement; resp: HTMLElement; val: HTMLElement }[] = [];
  private meterClock = 0;
  private confirmAction: (() => void) | null = null;

  private readonly who: HTMLElement;
  private readonly costMeter: HTMLElement;
  private readonly costFill: HTMLElement;
  private readonly costNum: HTMLElement;
  private readonly revertBtn: HTMLButtonElement;
  private readonly saveBtn: HTMLButtonElement;
  private readonly note: HTMLElement;
  private readonly prompt: HTMLElement;
  private readonly promptText: HTMLElement;
  private readonly pasteBox: HTMLElement;
  private readonly pasteText: HTMLTextAreaElement;
  private readonly msg: HTMLElement;
  private readonly jump: HTMLElement;
  private readonly body: HTMLElement;

  constructor(private root: HTMLElement, private deps: GeneEditorDeps) {
    root.textContent = '';
    this.who = h('span', { class: 'mono vg-who ell' });
    this.revertBtn = h('button', { class: 'btn sm ghost', title: 'Back to the preset as saved' });
    this.revertBtn.innerHTML = `${icon('undo', 14)}<span class="lbl">Revert</span>`;
    this.saveBtn = h('button', { class: 'btn sm primary', title: 'Add the edited genome to the population as a child of this preset' });
    this.saveBtn.innerHTML = `${icon('save', 14)}<span>Save as new</span>`;
    root.append(h('div', { class: 'sec-h' }, h('div', { class: 'col grow', style: 'gap:1px' }, h('h2', { text: 'Genes' }), this.who), this.revertBtn, this.saveBtn));

    this.promptText = h('b', { text: 'Discard unsaved edits?' });
    const discard = h('button', { class: 'btn danger sm' }, 'Discard');
    const keep = h('button', { class: 'btn ghost sm' }, 'Keep editing');
    this.prompt = h('div', { class: 'confirm warn', role: 'alertdialog', hidden: true }, this.promptText, h('div', { class: 'row' }, discard, keep));
    this.note = h('div', { class: 'banner', style: '--c:var(--warn)', hidden: true });
    this.note.innerHTML = `${icon('alert', 14)}<span class="sp">Editing · auto-switch paused until you save or revert</span>`;
    this.msg = h('div', { class: 'banner vg-msg', role: 'status', hidden: true });

    this.costFill = h('i');
    this.costMeter = h('div', { class: 'meter h6' }, this.costFill);
    this.costNum = h('span', { class: 'mono vg-cost-num' });
    const copyBtn = h('button', { class: 'ib sm', title: 'Copy genome JSON', 'aria-label': 'Copy genome JSON' });
    copyBtn.innerHTML = icon('copy', 16);
    const pasteBtn = h('button', { class: 'ib sm', title: 'Paste genome JSON', 'aria-label': 'Paste genome JSON', 'aria-expanded': 'false' });
    pasteBtn.innerHTML = icon('paste', 16);
    const cost = h('div', { class: 'row vg-cost', title: `Estimated GPU cost per frame at 1440p against the ${COST_BUDGET_MS} ms budget` },
      h('span', { class: 'muted vg-cpu' }), h('span', { class: 'muted vg-cost-label', text: 'GPU cost' }), this.costMeter, this.costNum, h('span', { class: 'div', style: 'height:18px' }), copyBtn, pasteBtn);
    (cost.querySelector('.vg-cpu') as HTMLElement).innerHTML = icon('cpu', 15);

    this.pasteText = h('textarea', { class: 'txt mono', rows: 4, spellcheck: 'false', placeholder: 'Paste genome JSON here', 'aria-label': 'Genome JSON' });
    const apply = h('button', { class: 'btn sm primary' }, 'Apply');
    const cancel = h('button', { class: 'btn sm ghost' }, 'Close');
    this.pasteBox = h('div', { class: 'sub vg-paste', hidden: true }, this.pasteText, h('div', { class: 'row' }, apply, cancel));
    this.jump = h('div', { class: 'seg full', role: 'group', 'aria-label': 'Jump to' });
    root.append(h('div', { class: 'vg-top' }, this.prompt, this.note, this.msg, cost, this.pasteBox, this.jump));
    this.body = h('div', { class: 'vg-body scroll' });
    root.append(this.body);

    this.revertBtn.addEventListener('click', () => this.askRevert());
    this.saveBtn.addEventListener('click', () => void this.save());
    copyBtn.addEventListener('click', () => this.copy());
    pasteBtn.addEventListener('click', () => {
      if (this.pasteBox.hidden) this.openPaste('');
      else this.pasteBox.hidden = true;
      pasteBtn.setAttribute('aria-expanded', String(!this.pasteBox.hidden));
    });
    apply.addEventListener('click', () => this.applyPaste());
    cancel.addEventListener('click', () => (this.pasteBox.hidden = true));
    discard.addEventListener('click', () => {
      const act = this.confirmAction;
      this.hidePrompt();
      this.revert();
      act?.();
    });
    keep.addEventListener('click', () => this.hidePrompt());
    this.jump.addEventListener('click', (ev) => {
      const b = (ev.target as HTMLElement).closest<HTMLButtonElement>('button[data-g]');
      if (!b) return;
      for (const x of this.jump.querySelectorAll('button')) x.setAttribute('aria-pressed', String(x === b));
      this.body.querySelector<HTMLElement>(`#vg-g${b.dataset.g}`)?.scrollIntoView({ block: 'start', behavior: 'smooth' });
    });
    // Keys typed into the panel never reach the page shortcuts (Esc still closes the dock
    // from anywhere but a text field).
    root.addEventListener('keydown', (ev) => {
      const t = ev.target as HTMLElement;
      if (ev.key === 'Escape' && !(t instanceof HTMLInputElement && t.type === 'text') && !(t instanceof HTMLTextAreaElement) && !(t instanceof HTMLSelectElement)) {
        if (!this.prompt.hidden) {
          ev.stopPropagation();
          this.hidePrompt();
        }
        return;
      }
      ev.stopPropagation();
    });
  }

  /** Called when the close button is pressed (main hides the panel). */
  onClose: (() => void) | null = null;
  /** A preset started showing (its genome is now the one edited). */
  onLoad: ((m: Member | null) => void) | null = null;

  /** The genome as edited now (the scratch copy; do not mutate). */
  current(): Genome | null {
    return this.scratch;
  }

  get memberId(): string {
    return this.member?.id ?? '';
  }

  get memberLabel(): string {
    return this.member ? `${this.member.id} "${this.member.name}"` : '';
  }

  /**
   * Replaces the edited genome from outside the editor (the gene chat): swapped in place when the
   * structure is unchanged, else compiled and crossfaded; the controls it touched (parameter paths
   * or section ids) flash and slide from their old values.
   */
  applyExternal(g: Genome, touched: string[] = []): void {
    if (!this.scratch) return;
    const before = this.scratch;
    this.errors.clear();
    this.scratch = cloneGenome(g);
    this.render();
    this.push(false, 'chat');
    this.afterChange();
    this.flash(touched, before);
  }

  private flash(touched: string[], before: Genome): void {
    const seen = new Set<string>();
    for (const t of touched) {
      if (seen.has(t)) continue;
      seen.add(t);
      const row = this.body.querySelector<HTMLElement>(`.prow[data-path="${CSS.escape(t)}"]`);
      const hit = row ?? this.body.querySelector<HTMLElement>(`[data-sec="${CSS.escape(t)}"]`);
      const sec = hit?.closest<HTMLDetailsElement>('details') ?? null;
      if (sec && !sec.open) sec.open = true;
      const el = row ?? (hit instanceof HTMLDetailsElement ? hit.querySelector<HTMLElement>('summary') : hit);
      if (!el) continue;
      el.classList.remove('vg-flash');
      void el.offsetWidth;
      el.classList.add('vg-flash');
      window.setTimeout(() => el.classList.remove('vg-flash'), 1600);
      // Sliders glide from the old value to the new one.
      const range = row?.querySelector<HTMLInputElement>('input[type=range]');
      const pp = row?.dataset.path;
      if (range && pp) {
        const i = pp.lastIndexOf('.');
        const t0 = this.targetOf(pp.slice(0, i));
        const key = pp.slice(i + 1);
        const spec = t0 && E.schemaAt(before, t0)?.[key];
        if (t0 && spec) {
          const from = E.toSlider(E.getParam(before, t0, key), spec);
          const to = Number(range.value);
          if (from !== to) this.glide(range, from, to);
        }
      }
    }
    const first = this.body.querySelector<HTMLElement>('.vg-flash');
    first?.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
  }

  /** The target whose section id is `id` in the current genome (null when gone). */
  private targetOf(id: string): E.Target | null {
    const g = this.scratch;
    if (!g) return null;
    for (const sec of E.buildModel(g)) {
      if (sec.target && E.targetId(sec.target) === id) return sec.target;
      for (const it of sec.items ?? []) if (E.targetId(it.target) === id) return it.target;
    }
    return null;
  }

  private glide(range: HTMLInputElement, from: number, to: number): void {
    const t0 = performance.now();
    const dur = 450;
    const step = (now: number) => {
      const k = Math.min(1, (now - t0) / dur);
      const e = 1 - Math.pow(1 - k, 3);
      range.value = String(Math.round(from + (to - from) * e));
      range.style.setProperty('--v', `${(Number(range.value) / E.SLIDER_STEPS) * 100}%`);
      if (k < 1 && range.isConnected) requestAnimationFrame(step);
    };
    range.value = String(from);
    requestAnimationFrame(step);
  }
  /** Asked to be seen (a discard question is waiting): main opens the Genes tab. */
  onAttention: (() => void) | null = null;

  get dirty(): boolean {
    return this.dirtyNow;
  }

  /** Visible with the HUD (and the panel toggle); forced visible while asking to discard. */
  setShown(on: boolean): void {
    this.shown = on;
    this.applyVisibility();
  }

  private applyVisibility(): void {
    const vis = this.shown || this.forced;
    this.root.hidden = !vis;
    if (vis) this.meterClock = 0;
  }

  /** A preset starts showing: its genome becomes the one being edited. */
  load(m: Member | null): void {
    window.clearTimeout(this.debounce);
    this.member = m;
    this.hidePrompt();
    this.pasteBox.hidden = true;
    this.errors.clear();
    this.watch = null;
    this.say('');
    if (!m) {
      this.original = this.scratch = this.lastGood = null;
      this.body.textContent = '';
      this.who.textContent = '';
      this.setDirty(false);
      this.onLoad?.(null);
      return;
    }
    this.original = cloneGenome(m.genome);
    this.originalJson = JSON.stringify(this.original);
    this.scratch = cloneGenome(m.genome);
    this.lastGood = cloneGenome(m.genome);
    this.liveKey = structuralKey(m.genome);
    this.who.textContent = `${m.id} · ${m.name}`;
    this.who.title = `${m.id} · ${m.name} · ${m.type}`;
    this.render();
    this.afterChange();
    this.onLoad?.(m);
  }

  /** Asks before discarding unsaved edits; runs the action on "Discard". */
  confirmDiscard(action: () => void): void {
    this.confirmAction = action;
    this.promptText.textContent = 'Discard unsaved edits?';
    this.prompt.hidden = false;
    this.forced = true;
    this.applyVisibility();
    this.updateBanners();
    this.onAttention?.();
    (this.prompt.querySelector('.btn.ghost') as HTMLElement | null)?.focus();
  }

  /** Revert asks inline first. */
  private askRevert(): void {
    if (!this.dirtyNow) return;
    this.confirmAction = null;
    this.promptText.textContent = 'Discard unsaved edits?';
    this.prompt.hidden = false;
    this.updateBanners();
    (this.prompt.querySelector('.btn.ghost') as HTMLElement | null)?.focus();
  }

  private updateBanners(): void {
    this.note.hidden = !this.dirtyNow || !this.prompt.hidden;
    this.msg.hidden = !this.msgText || !this.prompt.hidden;
  }

  private hidePrompt(): void {
    this.confirmAction = null;
    this.prompt.hidden = true;
    if (this.forced) {
      this.forced = false;
      this.applyVisibility();
    }
    this.updateBanners();
  }

  /** Per frame: compile status of structural edits, live meters (when visible). */
  tick(dt: number): void {
    const w = this.watch;
    if (w) {
      const eng = this.deps.eng;
      const err = eng.failed(w.genome);
      if (err) {
        this.watch = null;
        this.errors.set(w.anchor, `shader failed to compile, kept the last working genome (${err.split('\n').find((l) => /error/i.test(l))?.trim().slice(0, 120) ?? 'see console'})`);
        this.scratch = cloneGenome(w.fallback);
        this.liveKey = structuralKey(this.scratch);
        eng.edit(this.scratch);
        this.render();
        this.afterChange();
      } else if (!eng.compiling) {
        this.watch = null;
        const on = eng.playing();
        if (on && structuralKey(on) === w.key) this.lastGood = cloneGenome(this.scratch!);
      }
    }
    if (this.root.hidden || !this.meters.length) return;
    this.meterClock -= dt;
    if (this.meterClock > 0) return;
    this.meterClock = METER_EVERY_S;
    const on = this.deps.eng.playing();
    if (!on || structuralKey(on) !== this.liveKey || !this.scratch) return;
    const ms = this.deps.eng.reactionMeters();
    ms.forEach((m, j) => {
      const el = this.meters[j];
      const r = this.scratch!.reactions[j];
      if (!el || !r) return;
      el.src.style.setProperty('--v', `${Math.round(Math.min(1, Math.max(0, m.src)) * 100)}%`);
      el.resp.style.setProperty('--v', `${Math.round(Math.min(1, Math.max(0, m.resp)) * 100)}%`);
      const spec = schemaFor(this.scratch!, r.g, r.i)?.[r.k];
      el.val.textContent = spec && Number.isFinite(m.value) ? E.formatValue(m.value, spec) : '–';
    });
  }

  // ------------------------------------------------------------ edits

  private applyParam(t: E.Target, key: string, v: number, drag: boolean, anchor: string): void {
    if (!this.scratch) return;
    const res = E.editParam(this.scratch, t, key, v);
    this.scratch = res.genome;
    if (res.repaired) this.syncAll();
    this.errors.delete(anchor);
    this.push(drag, anchor);
    this.afterChange();
  }

  private structural(fn: (g: Genome) => E.EditResult, anchor: string): void {
    if (!this.scratch) return;
    const r = fn(this.scratch);
    if (!r.ok) {
      this.errors.set(anchor, r.reason ?? 'not possible');
      this.render();
      return;
    }
    this.errors.delete(anchor);
    this.scratch = r.genome;
    this.render();
    this.push(false, anchor);
    this.afterChange();
  }

  /** Hands the scratch genome to the engine: in place when the structure is live, else (debounced) a compile. */
  private push(drag: boolean, anchor: string): void {
    const g = this.scratch!;
    if (structuralKey(g) === this.liveKey) {
      window.clearTimeout(this.debounce);
      this.deps.eng.edit(g);
      if (!this.watch) this.lastGood = cloneGenome(g);
      return;
    }
    window.clearTimeout(this.debounce);
    if (drag) this.debounce = window.setTimeout(() => this.compile(anchor), DRAG_DEBOUNCE_MS);
    else this.compile(anchor);
  }

  private compile(anchor: string): void {
    const g = this.scratch;
    if (!g) return;
    const key = structuralKey(g);
    const fallback = this.watch?.fallback ?? this.lastGood ?? cloneGenome(g);
    const how = this.deps.eng.edit(g);
    this.liveKey = key;
    if (how === 'compile') this.watch = { key, genome: cloneGenome(g), anchor, fallback };
    else {
      this.watch = null;
      this.lastGood = cloneGenome(g);
    }
  }

  private revert(): void {
    if (!this.original) return;
    this.errors.clear();
    this.scratch = cloneGenome(this.original);
    this.render();
    this.push(false, 'revert');
    this.afterChange();
    this.say('Reverted to the saved preset.');
  }

  private async save(): Promise<void> {
    const parent = this.member;
    if (!parent || !this.scratch || this.saving || !this.dirtyNow) return;
    this.saving = true;
    this.saveBtn.disabled = true;
    const g = repair(this.scratch);
    const adjusted = JSON.stringify(g) !== JSON.stringify(this.scratch);
    this.say('Checking the edited preset…');
    let warn = '';
    let descriptor: number[] | undefined;
    try {
      const res = await Promise.race([
        this.deps.evo.screener.screen(g),
        new Promise<null>((resolve) => window.setTimeout(() => resolve(null), 12000)),
      ]);
      if (!res) warn = 'screening timed out';
      else if (!res.ok) warn = res.reason ?? 'rejected';
      else descriptor = res.descriptor;
    } catch (err) {
      warn = `screening failed: ${err instanceof Error ? err.message : String(err)}`;
    }
    this.saving = false;
    if (this.member !== parent) {
      this.say('The preset changed while saving; nothing was saved.', 'warn');
      this.afterChange();
      return;
    }
    const child = E.saveEdited(this.deps.evo.pop, this.deps.evo.pop.get(parent.id) ?? null, g);
    if (descriptor) child.descriptor = descriptor;
    this.deps.evo.changed();
    this.deps.adopt(child);
    const notes = [warn && `screener: ${warn}`, adjusted && 'repair adjusted some values'].filter(Boolean).join('; ');
    this.say(`Saved as ${child.id} · ${child.name}${notes ? ` (${notes})` : ''}`, warn ? 'warn' : 'ok');
  }

  private copy(): void {
    if (!this.scratch) return;
    const text = JSON.stringify(this.scratch);
    const fallback = () => {
      this.openPaste(text);
      this.pasteText.select();
      this.say('Clipboard unavailable: the JSON is selected below, copy it with Cmd/Ctrl+C.', 'warn');
    };
    if (navigator.clipboard?.writeText) navigator.clipboard.writeText(text).then(() => this.say('Genome JSON copied.'), fallback);
    else fallback();
  }

  private openPaste(text: string): void {
    this.pasteBox.hidden = false;
    this.pasteText.value = text;
    this.pasteText.focus();
  }

  private applyPaste(): void {
    const r = E.parseGenome(this.pasteText.value);
    if (!r.ok) {
      this.say(`Paste failed: ${r.reason}`, 'err');
      return;
    }
    this.pasteBox.hidden = true;
    this.errors.clear();
    this.scratch = r.genome;
    this.render();
    this.push(false, 'paste');
    this.afterChange();
    this.say('Pasted genome loaded (unsaved).');
  }

  private say(text: string, kind: 'ok' | 'warn' | 'err' = 'ok'): void {
    this.msgText = text;
    const c = kind === 'ok' ? 'var(--ok)' : kind === 'warn' ? 'var(--warn)' : 'var(--neg)';
    this.msg.style.setProperty('--c', c);
    this.msg.innerHTML = text ? `${icon(kind === 'ok' ? 'check' : 'alert', 14)}<span class="sp"></span>` : '';
    const sp = this.msg.querySelector('.sp');
    if (sp) sp.textContent = text;
    this.updateBanners();
  }

  private setDirty(d: boolean): void {
    if (d === this.dirtyNow) return;
    this.dirtyNow = d;
    this.root.classList.toggle('vg-dirty', d);
    this.updateBanners();
    this.deps.onDirty(d);
  }

  private afterChange(): void {
    const g = this.scratch;
    if (!g) return;
    const cost = estimateCost(g);
    const f = cost / COST_BUDGET_MS;
    const c = f > 1 ? 'var(--neg)' : f >= 0.85 ? 'var(--warn)' : 'var(--ok)';
    this.costFill.style.setProperty('--v', `${Math.min(100, (f / 1.25) * 100).toFixed(1)}%`);
    this.costMeter.style.setProperty('--c', c);
    this.costNum.style.color = c;
    this.costNum.textContent = `${cost.toFixed(1)} / ${COST_BUDGET_MS} ms${f > 1 ? ' over' : ''}`;
    const d = JSON.stringify(g) !== this.originalJson;
    this.setDirty(d);
    this.revertBtn.disabled = !d;
    this.saveBtn.disabled = !d || this.saving;
  }

  private syncAll(): void {
    for (const b of this.bound) if (b.el !== document.activeElement) b.sync();
  }

  // ------------------------------------------------------------ render

  private render(): void {
    const g = this.scratch;
    const scroll = this.body.scrollTop;
    this.body.textContent = '';
    this.bound = [];
    this.meters = [];
    if (!g) return;
    let group = -2;
    const groups: { id: number; label: string }[] = [];
    for (const sec of E.buildModel(g)) {
      if (sec.body !== group) {
        group = sec.body;
        const label = sec.body < 0 ? 'Genome' : g.bodies.length > 1 ? `Body ${sec.body + 1}` : 'Body';
        groups.push({ id: sec.body, label });
        this.body.append(h('div', { class: 'grp-h', id: `vg-g${sec.body}` }, h('span', { class: 'overline', text: label })));
      }
      this.body.append(this.section(sec));
    }
    this.body.scrollTop = scroll;
    // Jump links, one per group.
    const pressed = this.jump.querySelector('button[aria-pressed=true]')?.getAttribute('data-g');
    this.jump.textContent = '';
    for (const gr of groups) {
      const b = h('button', { 'data-g': gr.id, 'aria-pressed': String(pressed === String(gr.id) || (!pressed && gr === groups[0])) }, gr.label);
      this.jump.append(b);
    }
    this.jump.hidden = groups.length < 2;
  }

  private errLine(anchor: string): HTMLElement | null {
    const e = this.errors.get(anchor);
    if (!e) return null;
    const el = h('div', { class: 'banner vg-err', role: 'alert', style: '--c:var(--neg)' });
    el.innerHTML = `${icon('alert', 14)}<span class="sp"></span>`;
    el.querySelector('.sp')!.textContent = e;
    return el;
  }

  private section(sec: E.SectionModel): HTMLElement {
    const open = this.openState.get(sec.id) ?? sec.open;
    const d = h('details', { class: 'gx', open: open || this.errors.has(sec.id), 'data-sec': sec.id });
    const m = /^(.*) \((\d+)\/(\d+)\)$/.exec(sec.title);
    const title = m ? m[1] : sec.title;
    const count = m ? `${m[2]} / ${m[3]}` : sec.alleles ? String(sec.alleles.length) : '';
    const chev = h('span', { class: 'chev' });
    chev.innerHTML = icon('cright', 14);
    const sum = h('summary', {}, chev, h('b', { text: title }), count ? h('span', { class: 'dim mono vg-count', text: count }) : null, h('span', { class: 'sp' }));
    if (sec.kind && sec.target) {
      const t = sec.target;
      const sel = this.select(sec.kind.options.map((o) => ({ value: o, label: o })), sec.kind.value, (v) => this.structural((g) => E.switchKind(g, t, v), sec.id), `${title} kind`);
      sel.classList.add('vg-kind');
      // The dropdown lives in the summary: keep clicks and keys from folding the section.
      for (const ev of ['click', 'keydown', 'keyup']) sel.addEventListener(ev, (e) => e.stopPropagation());
      sum.append(sel);
    } else if (sec.kind) {
      sum.append(h('span', { class: 'tag', text: sec.kind.value }));
    }
    d.append(sum);
    d.addEventListener('toggle', () => {
      this.openState.set(sec.id, d.open);
      saveSetting('v2.genesOpen', Object.fromEntries(this.openState));
    });
    const gb = h('div', { class: 'gb' });
    d.append(gb);
    const err = this.errLine(sec.id);
    if (err) gb.append(err);
    if (sec.target) for (const c of sec.params) gb.append(this.paramRow(sec.target, c, sec.id));
    if (sec.target?.t === 'fuse') {
      const b = sec.target.b;
      gb.append(h('div', { class: 'row' }, this.button('Remove fused shape', () => this.structural((g) => E.removeFuse(g, b), sec.id), undefined, false, 'trash', 'ghost')));
    }
    if (sec.list) gb.append(this.list(sec));
    if (sec.alleles) {
      const b = sec.body;
      gb.append(h('p', { class: 'dim vg-hint', text: 'Dormant genes this body carries. Express swaps one in; the gene showing goes silent.' }));
      for (const a of sec.alleles) {
        const anchor = `${sec.id}.${a.locus}`;
        gb.append(h('div', { class: 'allele' },
          h('div', { class: 'grow' }, h('b', { text: `${E.LOCUS_TITLE[a.locus]}: ${a.kind}` }), h('small', { text: a.summary })),
          this.button('Express', () => this.structural((g) => E.expressAllele(g, b, a.locus), anchor), 'Show this gene; the one showing goes silent', false, 'sparkle')));
        const el = this.errLine(anchor);
        if (el) gb.append(el);
      }
    }
    return d;
  }

  private list(sec: E.SectionModel): HTMLElement {
    const wrap = h('div', { class: 'vg-list' });
    const body = sec.body;
    const items = sec.items ?? [];
    for (const [j, it] of items.entries()) {
      const t = it.target;
      if (sec.list === 'reactions') {
        const item = this.reaction(j);
        const el = this.errLine(it.id);
        if (el) item.append(el);
        for (const c of it.params) item.append(this.paramRow(t, c, it.id));
        wrap.append(item);
        continue;
      }
      const item = h('div', { class: 'sub', 'data-sec': it.id });
      const head = h('div', { class: 'sh' });
      head.append(h('span', { class: sec.list === 'xforms' ? '' : 'dim mono vg-n', text: it.title }));
      if (it.kind) {
        const sel = this.select(it.kind.options.map((o) => ({ value: o, label: o })), it.kind.value, (v) => this.structural((g) => E.switchKind(g, t, v), it.id), 'Op kind');
        sel.classList.add('vg-opkind');
        head.append(sel);
      }
      if (it.stage && sec.list === 'chain') {
        head.append(this.segmented([{ value: 'warp', label: 'warp' }, { value: 'view', label: 'view' }], it.stage, (v) => this.structural((g) => E.setStage(g, j, v as 'warp' | 'view'), it.id), 'Stage: warp (feedback) or view (display)'));
      }
      head.append(h('span', { class: 'sp' }));
      if (sec.list === 'chain' || sec.list === 'drawOps') {
        const move = (dir: -1 | 1) => this.structural((g) => (sec.list === 'chain' ? E.moveOp(g, j, dir) : E.moveDrawOp(g, body, j, dir)), it.id);
        head.append(this.iconButton('aup', () => move(-1), 'Move up', !it.canUp), this.iconButton('adown', () => move(1), 'Move down', !it.canDown));
      }
      head.append(this.iconButton('x', () => this.structural((g) => {
        if (sec.list === 'chain') return E.removeOp(g, j);
        if (sec.list === 'drawOps') return E.removeDrawOp(g, body, j);
        return E.removeXform(g, body, j);
      }, it.id), sec.list === 'xforms' ? 'Remove transform' : 'Remove'));
      item.append(head);
      const el = this.errLine(it.id);
      if (el) item.append(el);
      for (const c of it.params) item.append(this.paramRow(t, c, it.id));
      if (sec.list === 'xforms') {
        const present = new Set(it.params.filter((c) => c.key.startsWith('var.')).map((c) => c.key.slice(4)));
        const opts = FLAME_VARIATIONS.filter((v) => !present.has(v));
        const sel = this.select([{ value: '', label: 'add variation…' }, ...opts.map((v) => ({ value: v, label: v }))], '', (v) => {
          if (v) this.structural((g) => E.addVariation(g, body, j, v as FlameVar), it.id);
        }, 'Add a variation');
        item.append(h('div', { class: 'prow' }, h('span', { class: 'pl', text: 'Variation' }), sel));
      }
      wrap.append(item);
    }
    // Add row.
    const addAnchor = `${sec.id}.add`;
    if (sec.list === 'chain' || sec.list === 'drawOps') {
      const kinds = sec.list === 'chain' ? OP_KINDS : DRAW_OPS;
      let pick: string = 'swirl';
      const sel = this.select(kinds.map((k) => ({ value: k, label: k })), pick, (v) => (pick = v), 'Op to add');
      sel.classList.add('grow');
      const btn = this.button('Add op', () => this.structural((g) => (sec.list === 'chain' ? E.addOp(g, pick as OpKind) : E.addDrawOp(g, body, pick as OpKind)), addAnchor), undefined, !sec.canAdd, 'plus');
      wrap.append(h('div', { class: 'row' }, sel, btn));
    } else if (sec.list === 'reactions') {
      wrap.append(h('div', { class: 'row' }, this.button('Add reaction', () => this.structural((g) => E.addReaction(g), addAnchor), 'Disabled once every target is driven or the list is full', !sec.canAdd, 'plus', 'ghost')));
    } else if (sec.list === 'xforms') {
      wrap.append(h('div', { class: 'row' }, this.button('Add transform', () => this.structural((g) => E.addXform(g, body), addAnchor), undefined, !sec.canAdd, 'plus', 'ghost'),
        h('span', { class: 'dim vg-hint', text: `${items.length} transform${items.length === 1 ? '' : 's'}` })));
    }
    const el = this.errLine(addAnchor);
    if (el) wrap.append(el);
    return wrap;
  }

  private reaction(j: number): HTMLElement {
    const g = this.scratch!;
    const r = g.reactions[j];
    const anchor = `reaction${j}`;
    const src = this.select(SIGNALS.map((s) => ({ value: s, label: s })), r.src, (v) => this.structural((x) => E.switchKind(x, { t: 'reaction', j }, v), anchor), 'Source signal');
    const tsel = h('select', { class: 'dd', 'aria-label': 'Target parameter' });
    const used = new Set(g.reactions.filter((_x, i) => i !== j).map(E.reactKey));
    const groups = new Map<string, HTMLOptGroupElement>();
    for (const t of E.reactionTargets(g)) {
      let og = groups.get(t.group);
      if (!og) {
        og = h('optgroup', { label: t.group });
        groups.set(t.group, og);
        tsel.append(og);
      }
      const key = E.reactKey(t);
      og.append(h('option', { value: key, disabled: used.has(key), selected: key === E.reactKey(r), text: E.labelFor(t.k) + (used.has(key) ? ' (driven)' : '') }));
    }
    tsel.addEventListener('change', () => {
      const [gg, i, k] = tsel.value.split('|');
      this.structural((x) => E.setReactionTarget(x, j, { g: gg as typeof r.g, i: Number(i), k }), anchor);
    });
    const srcBar = h('i', { style: '--v:0%' });
    const respBar = h('i', { style: '--v:0%' });
    const val = h('span', { class: 'mono vg-m-val', text: '–' });
    this.meters[j] = { src: srcBar, resp: respBar, val };
    const arrow = h('span', { class: 'dim' });
    arrow.innerHTML = icon('cright', 14);
    const rm = this.iconButton('x', () => this.structural((x) => E.removeReaction(x, j), anchor), 'Remove reaction');
    return h('div', { class: 'rx' },
      h('div', { class: 'row' }, src, arrow, tsel, rm),
      h('div', { class: 'mm', title: 'Live: source signal (in), response after the curve (out), driven value' },
        h('span', { text: 'in' }), h('div', { class: 'meter h3' }, srcBar), h('span', { text: 'out' }), h('div', { class: 'meter h3', style: '--c:var(--ok)' }, respBar), val));
  }

  // ---------------------------------------------------------- widgets

  private button(label: string, fn: () => void, title?: string, disabled = false, ic?: IconName, kind = ''): HTMLButtonElement {
    const b = h('button', { class: `btn sm ${kind}`.trim(), title, disabled });
    b.innerHTML = `${ic ? icon(ic, 14) : ''}<span></span>`;
    b.querySelector('span')!.textContent = label;
    b.addEventListener('click', fn);
    return b;
  }

  private iconButton(name: IconName, fn: () => void, title: string, disabled = false): HTMLButtonElement {
    const b = h('button', { class: 'ib xs', title, 'aria-label': title, disabled });
    b.innerHTML = icon(name, 14);
    b.addEventListener('click', fn);
    return b;
  }

  private select(options: { value: string; label: string }[], value: string, fn: (v: string) => void, label: string): HTMLSelectElement {
    const s = h('select', { class: 'dd', 'aria-label': label });
    for (const o of options) s.append(h('option', { value: o.value, selected: o.value === value, text: o.label }));
    s.addEventListener('change', () => fn(s.value));
    return s;
  }

  private segmented(options: { value: string; label: string }[], value: string, fn: (v: string) => void, title: string): HTMLElement {
    const wrap = h('span', { class: 'seg', role: 'group', title });
    for (const o of options) {
      const b = h('button', { 'aria-pressed': String(o.value === value) }, o.label);
      b.addEventListener('click', () => fn(o.value));
      wrap.append(b);
    }
    return wrap;
  }

  private paramRow(t: E.Target, c: E.ParamControl, anchor: string): HTMLElement {
    const spec = c.spec;
    const title = `${c.key} · ${spec.choices ? spec.choices.join(' / ') : `${spec.min} … ${spec.max}${spec.log ? ' (log)' : ''}${spec.int ? ' (integer)' : ''}`} · default ${spec.def}`;
    const row = h('div', { class: 'prow', title, 'data-path': `${E.targetId(t)}.${c.key}` });
    row.append(h('span', { class: 'pl', text: c.label }));
    const current = () => {
      const g = this.scratch;
      return g && E.schemaAt(g, t)?.[c.key] ? E.getParam(g, t, c.key) : NaN;
    };
    if (c.widget === 'slider') {
      const range = h('input', { type: 'range', class: 'rng', min: 0, max: E.SLIDER_STEPS, step: 1, value: E.toSlider(c.value, spec), 'aria-label': c.label });
      const num = h('input', { type: 'text', class: 'val', value: E.formatValue(c.value, spec), 'aria-label': `${c.label} value`, inputmode: 'decimal' });
      const fill = () => range.style.setProperty('--v', `${(Number(range.value) / E.SLIDER_STEPS) * 100}%`);
      fill();
      range.addEventListener('input', () => {
        fill();
        const v = E.fromSlider(Number(range.value), spec);
        num.value = E.formatValue(v, spec);
        this.applyParam(t, c.key, v, true, anchor);
      });
      range.addEventListener('change', () => {
        // Drag released: a pending structural change compiles now.
        if (this.scratch && structuralKey(this.scratch) !== this.liveKey) {
          window.clearTimeout(this.debounce);
          this.compile(anchor);
        }
      });
      range.addEventListener('dblclick', () => {
        range.value = String(E.toSlider(spec.def, spec));
        fill();
        num.value = E.formatValue(spec.def, spec);
        this.applyParam(t, c.key, spec.def, false, anchor);
      });
      num.addEventListener('change', () => {
        const v = Number.parseFloat(num.value.replace(',', '.'));
        if (Number.isFinite(v)) this.applyParam(t, c.key, v, false, anchor);
        const now = current();
        num.value = E.formatValue(now, spec);
        range.value = String(E.toSlider(now, spec));
        fill();
      });
      num.addEventListener('keydown', (ev) => {
        if (ev.key === 'Enter') num.blur();
      });
      row.append(range, num);
      const sync = () => {
        const v = current();
        if (!Number.isFinite(v)) return;
        range.value = String(E.toSlider(v, spec));
        fill();
        num.value = E.formatValue(v, spec);
      };
      this.bound.push({ el: range, sync });
    } else {
      const opts = c.options!.map((o) => ({ value: String(o.value), label: o.label }));
      const holder = h('span', { class: 'vg-ctl' });
      const build = (v: number) => {
        holder.textContent = '';
        const pick = (s: string) => {
          this.applyParam(t, c.key, Number(s), false, anchor);
          build(current());
        };
        holder.append(c.widget === 'segmented' ? this.segmented(opts, String(v), pick, title) : this.select(opts, String(v), pick, c.label));
      };
      build(c.value);
      row.append(holder);
      this.bound.push({ el: holder, sync: () => build(current()) });
    }
    return row;
  }
}
