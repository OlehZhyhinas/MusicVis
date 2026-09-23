// Gene editor panel (HUD view, left side): the playing preset's genes as live
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
  private collapsed = loadSetting<boolean>('v2.genesCollapsed', false);
  private openState = new Map<string, boolean>();
  private errors = new Map<string, string>();
  private bound: Bound[] = [];
  private meters: { src: HTMLElement; resp: HTMLElement; val: HTMLElement }[] = [];
  private meterClock = 0;
  private confirmAction: (() => void) | null = null;

  private readonly who: HTMLElement;
  private readonly costFill: HTMLElement;
  private readonly costNum: HTMLElement;
  private readonly costWrap: HTMLElement;
  private readonly revertBtn: HTMLButtonElement;
  private readonly saveBtn: HTMLButtonElement;
  private readonly note: HTMLElement;
  private readonly prompt: HTMLElement;
  private readonly pasteBox: HTMLElement;
  private readonly pasteText: HTMLTextAreaElement;
  private readonly msg: HTMLElement;
  private readonly body: HTMLElement;

  constructor(private root: HTMLElement, private deps: GeneEditorDeps) {
    root.textContent = '';
    const fold = h('button', { class: 'tp-btn vg-fold', 'aria-label': 'Collapse the gene editor', title: 'Collapse' }, '▾');
    const close = h('button', { class: 'tp-btn vg-close', 'aria-label': 'Hide the gene editor (K)', title: 'Hide (K)' }, '×');
    this.who = h('span', { class: 'vg-who' });
    root.append(h('div', { class: 'vg-head' }, fold, h('span', { class: 'vg-title', text: 'Genes' }), this.who, close));

    this.costFill = h('i');
    this.costNum = h('span', { class: 'vg-cost-num' });
    this.costWrap = h('div', { class: 'vg-cost', title: `Estimated GPU cost per frame at 1440p against the ${COST_BUDGET_MS} ms budget` },
      h('span', { class: 'vg-cost-label', text: 'GPU' }), h('span', { class: 'vg-cost-bar' }, this.costFill, h('b')), this.costNum);
    this.revertBtn = h('button', { class: 'tp-btn-text', title: 'Back to the preset as saved' }, 'Revert');
    this.saveBtn = h('button', { class: 'tp-btn-text vg-save', title: 'Add the edited genome to the population as a child of this preset' }, 'Save as new');
    const copyBtn = h('button', { class: 'tp-btn-text', title: 'Copy the edited genome as JSON' }, 'Copy JSON');
    const pasteBtn = h('button', { class: 'tp-btn-text', title: 'Load a genome from JSON' }, 'Paste JSON');
    this.note = h('div', { class: 'vg-note', hidden: true }, 'Editing: auto-switch paused until you save or revert');
    const discard = h('button', { class: 'tp-btn-text' }, 'Discard');
    const keep = h('button', { class: 'tp-btn-text' }, 'Keep editing');
    this.prompt = h('div', { class: 'vg-prompt', hidden: true }, h('span', { text: 'Discard unsaved edits?' }), discard, keep);
    this.pasteText = h('textarea', { rows: 4, spellcheck: 'false', placeholder: 'Paste genome JSON here' });
    const apply = h('button', { class: 'tp-btn-text' }, 'Apply');
    const cancel = h('button', { class: 'tp-btn-text' }, 'Close');
    this.pasteBox = h('div', { class: 'vg-paste', hidden: true }, this.pasteText, h('div', { class: 'vg-paste-actions' }, apply, cancel));
    this.msg = h('div', { class: 'vg-msg' });
    root.append(h('div', { class: 'vg-top' }, this.costWrap, h('div', { class: 'vg-actions' }, this.revertBtn, this.saveBtn, copyBtn, pasteBtn), this.note, this.prompt, this.pasteBox, this.msg));
    this.body = h('div', { class: 'vg-body' });
    root.append(this.body);

    fold.addEventListener('click', () => {
      this.collapsed = !this.collapsed;
      saveSetting('v2.genesCollapsed', this.collapsed);
      this.root.classList.toggle('vg-collapsed', this.collapsed);
      fold.textContent = this.collapsed ? '▸' : '▾';
    });
    root.classList.toggle('vg-collapsed', this.collapsed);
    fold.textContent = this.collapsed ? '▸' : '▾';
    close.addEventListener('click', () => this.onClose?.());
    this.revertBtn.addEventListener('click', () => this.revert());
    this.saveBtn.addEventListener('click', () => void this.save());
    copyBtn.addEventListener('click', () => this.copy());
    pasteBtn.addEventListener('click', () => this.openPaste(''));
    apply.addEventListener('click', () => this.applyPaste());
    cancel.addEventListener('click', () => (this.pasteBox.hidden = true));
    discard.addEventListener('click', () => {
      const act = this.confirmAction;
      this.hidePrompt();
      this.revert();
      act?.();
    });
    keep.addEventListener('click', () => this.hidePrompt());
    // Keys typed into the panel never reach the page shortcuts.
    root.addEventListener('keydown', (ev) => ev.stopPropagation());
  }

  /** Called when the close button is pressed (main hides the panel). */
  onClose: (() => void) | null = null;

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
    document.getElementById('app')?.classList.toggle('vg-open', vis);
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
    if (!m) {
      this.original = this.scratch = this.lastGood = null;
      this.body.textContent = '';
      this.who.textContent = '';
      this.setDirty(false);
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
  }

  /** Asks before discarding unsaved edits; runs the action on "Discard". */
  confirmDiscard(action: () => void): void {
    this.confirmAction = action;
    this.prompt.hidden = false;
    this.forced = true;
    if (this.collapsed) this.root.classList.remove('vg-collapsed');
    this.applyVisibility();
  }

  private hidePrompt(): void {
    this.confirmAction = null;
    this.prompt.hidden = true;
    if (this.forced) {
      this.forced = false;
      this.root.classList.toggle('vg-collapsed', this.collapsed);
      this.applyVisibility();
    }
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
    if (this.root.hidden || this.collapsed || !this.meters.length) return;
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
      el.src.style.width = `${Math.round(Math.min(1, Math.max(0, m.src)) * 100)}%`;
      el.resp.style.width = `${Math.round(Math.min(1, Math.max(0, m.resp)) * 100)}%`;
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
    this.msg.textContent = text;
    this.msg.className = `vg-msg vg-${kind}`;
  }

  private setDirty(d: boolean): void {
    if (d === this.dirtyNow) return;
    this.dirtyNow = d;
    this.note.hidden = !d;
    this.root.classList.toggle('vg-dirty', d);
    this.deps.onDirty(d);
  }

  private afterChange(): void {
    const g = this.scratch;
    if (!g) return;
    const cost = estimateCost(g);
    const f = cost / COST_BUDGET_MS;
    this.costFill.style.width = `${Math.min(100, (f / 1.25) * 100).toFixed(1)}%`;
    this.costWrap.classList.toggle('vg-amber', f >= 0.85 && f <= 1);
    this.costWrap.classList.toggle('vg-red', f > 1);
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
    for (const sec of E.buildModel(g)) {
      if (sec.body !== group) {
        group = sec.body;
        const label = sec.body < 0 ? 'Preset' : g.bodies.length > 1 ? `Body ${sec.body + 1}` : 'Body';
        this.body.append(h('div', { class: 'vg-group', text: label }));
      }
      this.body.append(this.section(sec));
    }
    this.body.scrollTop = scroll;
  }

  private errLine(anchor: string): HTMLElement | null {
    const e = this.errors.get(anchor);
    return e ? h('div', { class: 'vg-err', role: 'alert', text: e }) : null;
  }

  private section(sec: E.SectionModel): HTMLElement {
    const open = this.openState.get(sec.id) ?? sec.open;
    const d = h('details', { class: 'vg-sec', open: open || this.errors.has(sec.id) });
    const sum = h('summary', {}, h('span', { text: sec.title }), sec.kind ? h('em', { text: sec.kind.value }) : null);
    d.append(sum);
    d.addEventListener('toggle', () => this.openState.set(sec.id, d.open));
    if (sec.kind && sec.target) {
      const t = sec.target;
      d.append(this.kindRow('kind', sec.kind, (v) => this.structural((g) => E.switchKind(g, t, v), sec.id)));
    }
    d.append(...[this.errLine(sec.id)].filter((x): x is HTMLElement => !!x));
    if (sec.target) for (const c of sec.params) d.append(this.paramRow(sec.target, c, sec.id));
    if (sec.target?.t === 'fuse') {
      const b = sec.target.b;
      d.append(h('div', { class: 'vg-row-actions' }, this.button('Remove fused shape', () => this.structural((g) => E.removeFuse(g, b), sec.id))));
    }
    if (sec.list) d.append(this.list(sec));
    if (sec.alleles) {
      const b = sec.body;
      for (const a of sec.alleles) {
        const anchor = `${sec.id}.${a.locus}`;
        d.append(h('div', { class: 'vg-allele' },
          h('span', {}, h('b', { text: `${E.LOCUS_TITLE[a.locus]}: ${a.kind}` }), h('small', { text: a.summary })),
          this.button('Express', () => this.structural((g) => E.expressAllele(g, b, a.locus), anchor), 'Show this gene; the one showing goes silent')));
        const el = this.errLine(anchor);
        if (el) d.append(el);
      }
    }
    return d;
  }

  private list(sec: E.SectionModel): HTMLElement {
    const wrap = h('div', { class: 'vg-list' });
    const body = sec.body;
    for (const [j, it] of (sec.items ?? []).entries()) {
      const item = h('div', { class: 'vg-item' });
      const head = h('div', { class: 'vg-item-head' });
      const t = it.target;
      if (sec.list === 'reactions') {
        head.append(this.reactionHead(j));
      } else {
        head.append(h('span', { class: 'vg-item-title', text: it.title }));
        if (it.kind) {
          head.append(this.select(it.kind.options.map((o) => ({ value: o, label: o })), it.kind.value, (v) => this.structural((g) => E.switchKind(g, t, v), it.id), 'Op kind'));
        }
        if (it.stage && sec.list === 'chain') {
          head.append(this.segmented([{ value: 'warp', label: 'warp' }, { value: 'view', label: 'view' }], it.stage, (v) => this.structural((g) => E.setStage(g, j, v as 'warp' | 'view'), it.id), 'Stage: warp (feedback) or view (display)'));
        }
        const acts = h('span', { class: 'vg-item-acts' });
        if (sec.list === 'chain' || sec.list === 'drawOps') {
          const move = (dir: -1 | 1) => this.structural((g) => (sec.list === 'chain' ? E.moveOp(g, j, dir) : E.moveDrawOp(g, body, j, dir)), it.id);
          acts.append(this.button('↑', () => move(-1), 'Move up', !it.canUp), this.button('↓', () => move(1), 'Move down', !it.canDown));
        }
        acts.append(this.button('×', () => this.structural((g) => {
          if (sec.list === 'chain') return E.removeOp(g, j);
          if (sec.list === 'drawOps') return E.removeDrawOp(g, body, j);
          return E.removeXform(g, body, j);
        }, it.id), 'Remove'));
        head.append(acts);
      }
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
        item.append(h('div', { class: 'vg-row-actions' }, sel));
      }
      wrap.append(item);
    }
    // Add row.
    const addAnchor = `${sec.id}.add`;
    if (sec.list === 'chain' || sec.list === 'drawOps') {
      const kinds = sec.list === 'chain' ? OP_KINDS : DRAW_OPS;
      let pick: string = 'swirl';
      const sel = this.select(kinds.map((k) => ({ value: k, label: k })), pick, (v) => (pick = v), 'Op to add');
      const btn = this.button('Add op', () => this.structural((g) => (sec.list === 'chain' ? E.addOp(g, pick as OpKind) : E.addDrawOp(g, body, pick as OpKind)), addAnchor), undefined, !sec.canAdd);
      wrap.append(h('div', { class: 'vg-row-actions' }, sel, btn));
    } else if (sec.list === 'reactions') {
      wrap.append(h('div', { class: 'vg-row-actions' }, this.button('Add reaction', () => this.structural((g) => E.addReaction(g), addAnchor), undefined, !sec.canAdd)));
    } else if (sec.list === 'xforms') {
      wrap.append(h('div', { class: 'vg-row-actions' }, this.button('Add transform', () => this.structural((g) => E.addXform(g, body), addAnchor), undefined, !sec.canAdd)));
    }
    const el = this.errLine(addAnchor);
    if (el) wrap.append(el);
    return wrap;
  }

  private reactionHead(j: number): HTMLElement {
    const g = this.scratch!;
    const r = g.reactions[j];
    const anchor = `reaction${j}`;
    const src = this.select(SIGNALS.map((s) => ({ value: s, label: s })), r.src, (v) => this.structural((x) => E.switchKind(x, { t: 'reaction', j }, v), anchor), 'Source signal');
    const tsel = h('select', { class: 'vg-select vg-target', 'aria-label': 'Target parameter' });
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
    const srcBar = h('i', { class: 'vg-m-src' });
    const respBar = h('i', { class: 'vg-m-resp' });
    const val = h('span', { class: 'vg-m-val', text: '–' });
    this.meters[j] = { src: srcBar, resp: respBar, val };
    const meter = h('span', { class: 'vg-meter', title: 'Live: source signal (top), response after the curve (bottom), driven value' }, h('span', { class: 'vg-m-bars' }, srcBar, respBar), val);
    const rm = this.button('×', () => this.structural((x) => E.removeReaction(x, j), anchor), 'Remove reaction');
    return h('div', { class: 'vg-react-head' }, h('div', { class: 'vg-react-line' }, src, h('span', { class: 'vg-arrow', text: '→' }), tsel, rm), h('div', { class: 'vg-react-line' }, meter));
  }

  // ---------------------------------------------------------- widgets

  private button(label: string, fn: () => void, title?: string, disabled = false): HTMLButtonElement {
    const b = h('button', { class: 'tp-btn-text vg-btn', title, disabled }, label);
    b.addEventListener('click', fn);
    return b;
  }

  private select(options: { value: string; label: string }[], value: string, fn: (v: string) => void, label: string): HTMLSelectElement {
    const s = h('select', { class: 'vg-select', 'aria-label': label });
    for (const o of options) s.append(h('option', { value: o.value, selected: o.value === value, text: o.label }));
    s.addEventListener('change', () => fn(s.value));
    return s;
  }

  private segmented(options: { value: string; label: string }[], value: string, fn: (v: string) => void, title: string): HTMLElement {
    const wrap = h('span', { class: 'vg-seg', role: 'group', title });
    for (const o of options) {
      const b = h('button', { class: o.value === value ? 'on' : '', 'aria-pressed': String(o.value === value) }, o.label);
      b.addEventListener('click', () => fn(o.value));
      wrap.append(b);
    }
    return wrap;
  }

  private kindRow(label: string, k: E.KindControl, fn: (v: string) => void): HTMLElement {
    return h('div', { class: 'vg-row vg-kind' }, h('label', { text: label }), this.select(k.options.map((o) => ({ value: o, label: o })), k.value, fn, label));
  }

  private paramRow(t: E.Target, c: E.ParamControl, anchor: string): HTMLElement {
    const spec = c.spec;
    const title = `${c.key} · ${spec.choices ? spec.choices.join(' / ') : `${spec.min} … ${spec.max}${spec.log ? ' (log)' : ''}${spec.int ? ' (integer)' : ''}`} · default ${spec.def}`;
    const row = h('div', { class: 'vg-row', title });
    row.append(h('label', { text: c.label }));
    const current = () => {
      const g = this.scratch;
      return g && E.schemaAt(g, t)?.[c.key] ? E.getParam(g, t, c.key) : NaN;
    };
    if (c.widget === 'slider') {
      const range = h('input', { type: 'range', min: 0, max: E.SLIDER_STEPS, step: 1, value: E.toSlider(c.value, spec), 'aria-label': c.label });
      const num = h('input', { type: 'text', class: 'vg-num', value: E.formatValue(c.value, spec), 'aria-label': `${c.label} value`, inputmode: 'decimal' });
      range.addEventListener('input', () => {
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
        num.value = E.formatValue(spec.def, spec);
        this.applyParam(t, c.key, spec.def, false, anchor);
      });
      num.addEventListener('change', () => {
        const v = Number.parseFloat(num.value.replace(',', '.'));
        if (Number.isFinite(v)) this.applyParam(t, c.key, v, false, anchor);
        const now = current();
        num.value = E.formatValue(now, spec);
        range.value = String(E.toSlider(now, spec));
      });
      num.addEventListener('keydown', (ev) => {
        if (ev.key === 'Enter') num.blur();
      });
      row.append(range, num);
      const sync = () => {
        const v = current();
        if (!Number.isFinite(v)) return;
        range.value = String(E.toSlider(v, spec));
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
