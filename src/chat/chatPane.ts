// The gene chat pane under the gene editor in the Genes tab: describe a change in words, the local
// model edits the playing preset in place (the editor's usual dirty banner, Save as new, Revert), and
// the controls it touched flash. Opt-in: nothing downloads until the user starts it. Each turn is
// one undo step.

import { cloneGenome, type Genome } from '../v2/genome';
import type { GeneEditor } from '../v2/geneEditor';
import { icon } from '../ui/icons';
import { loadSetting, saveSetting } from '../ui/storage';
import { GeneChat, type TurnResult } from './geneChat';
import { LocalLLM, MODEL_BYTES, unsupportedReason, type Progress } from './llm';
import type { LookSampler } from './look';

export interface ChatPaneDeps {
  editor: GeneEditor;
  look: LookSampler;
  keyHue: () => number;
}

type State = 'off' | 'unsupported' | 'loading' | 'warming' | 'ready' | 'busy' | 'failed';

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

const MIN_H = 150;
const EXAMPLES = ['calmer, with longer trails', 'make it more blue', 'fill more of the screen', 'pulse harder on the bass'];

export class ChatPane {
  private state: State = 'off';
  private llm = new LocalLLM();
  readonly chat: GeneChat;
  private abort: AbortController | null = null;
  private undo: { before: Genome; after: Genome; btn: HTMLButtonElement }[] = [];
  private collapsed = loadSetting<boolean>('chat.collapsed', false);
  private shown = false;

  private readonly status: HTMLElement;
  private readonly stopBtn: HTMLButtonElement;
  private readonly toggle: HTMLButtonElement;
  private readonly log: HTMLElement;
  private readonly intro: HTMLElement;
  private readonly progress: HTMLElement;
  private readonly progFill: HTMLElement;
  private readonly progText: HTMLElement;
  private readonly input: HTMLTextAreaElement;
  private readonly send: HTMLButtonElement;
  private readonly bodyEl: HTMLElement;

  constructor(private root: HTMLElement, private deps: ChatPaneDeps) {
    this.chat = new GeneChat(this.llm, {
      genome: () => deps.editor.current(),
      presetId: () => deps.editor.memberId,
      presetLabel: () => deps.editor.memberLabel,
      keyHue: deps.keyHue,
      look: () => deps.look.metrics,
    });
    root.textContent = '';
    root.classList.add('gc');
    const grip = h('div', { class: 'gc-grip', role: 'separator', 'aria-orientation': 'horizontal', 'aria-label': 'Resize the chat', title: 'Drag to resize' });
    this.toggle = h('button', { class: 'gc-toggle', 'aria-expanded': 'true', title: 'Show or hide the chat' });
    this.toggle.innerHTML = `<span class="chev">${icon('cright', 14)}</span><b>Chat</b>`;
    this.status = h('span', { class: 'tag sm gc-status', text: 'off' });
    this.stopBtn = h('button', { class: 'btn sm ghost', hidden: true, title: 'Stop generating' });
    this.stopBtn.innerHTML = `${icon('stop', 13)}<span>Stop</span>`;
    const head = h('div', { class: 'gc-head' }, this.toggle, this.status, h('span', { class: 'grow' }), this.stopBtn);

    this.log = h('div', { class: 'gc-log scroll', role: 'log', 'aria-live': 'polite', 'aria-label': 'Chat messages' });
    this.intro = h('div', { class: 'gc-intro' });
    this.log.append(this.intro);
    this.progFill = h('i', { style: '--v:0%' });
    this.progText = h('span', { class: 'dim ell' });
    this.progress = h('div', { class: 'gc-progress', hidden: true }, h('div', { class: 'meter h5' }, this.progFill), this.progText);
    this.input = h('textarea', { class: 'txt gc-in', rows: 2, placeholder: 'Describe a change, e.g. calmer, more blue, fill the screen', 'aria-label': 'Describe a change to the preset', disabled: true });
    this.send = h('button', { class: 'btn sm primary', title: 'Send (Enter)', disabled: true });
    this.send.innerHTML = `<span>Send</span>`;
    const form = h('form', { class: 'gc-form' }, this.input, this.send);
    this.bodyEl = h('div', { class: 'gc-body' }, this.log, this.progress, form);
    root.append(grip, head, this.bodyEl);

    this.toggle.addEventListener('click', () => this.setCollapsed(!this.collapsed));
    this.stopBtn.addEventListener('click', () => this.abort?.abort());
    form.addEventListener('submit', (ev) => {
      ev.preventDefault();
      void this.submit();
    });
    this.input.addEventListener('keydown', (ev) => {
      if (ev.key === 'Enter' && !ev.shiftKey && !ev.isComposing) {
        ev.preventDefault();
        void this.submit();
      } else if (ev.key === 'Escape') {
        this.input.blur();
      }
    });
    // Keys pressed in the pane never reach the page shortcuts (Esc still closes the dock).
    root.addEventListener('keydown', (ev) => {
      if (ev.key === 'Escape' && ev.target !== this.input) return;
      ev.stopPropagation();
    });
    this.dragResize(grip);
    const saved = loadSetting<number>('chat.h', 0);
    if (saved) root.style.setProperty('--gc-h', `${saved}px`);
    this.setCollapsed(this.collapsed);
    this.renderIntro();
    void unsupportedReason().then((why) => {
      if (why) {
        this.setState('unsupported');
        this.renderIntro(why);
      }
    });
    deps.editor.onLoad = () => this.presetChanged();
  }

  /** The local model (loaded once the chat has started). */
  get model(): LocalLLM {
    return this.llm;
  }

  /** The Genes tab became visible (or hidden): a previously enabled chat starts again. */
  setShown(on: boolean): void {
    this.shown = on;
    this.deps.look.on = on && this.llm.ready;
    if (on && this.state === 'off' && !this.collapsed && loadSetting<boolean>('chat.enabled', false)) void this.start();
  }

  // ------------------------------------------------------------ state

  private setState(s: State): void {
    this.state = s;
    this.root.dataset.state = s;
    const label: Record<State, string> = { off: 'off', unsupported: 'unavailable', loading: 'loading', warming: 'reading preset', ready: 'ready', busy: 'thinking', failed: 'failed' };
    this.status.textContent = label[s];
    this.status.classList.toggle('on', s === 'ready' || s === 'busy' || s === 'warming');
    const canType = s === 'ready';
    this.input.disabled = !(s === 'ready' || s === 'warming');
    this.send.disabled = !canType;
    this.stopBtn.hidden = s !== 'busy';
    this.progress.hidden = s !== 'loading';
    this.deps.look.on = this.llm.ready;
  }

  private setCollapsed(c: boolean): void {
    this.collapsed = c;
    saveSetting('chat.collapsed', c);
    this.root.classList.toggle('collapsed', c);
    this.toggle.setAttribute('aria-expanded', String(!c));
    if (!c && this.shown && this.state === 'off' && loadSetting<boolean>('chat.enabled', false)) void this.start();
  }

  private renderIntro(unsupported?: string): void {
    const el = this.intro;
    el.textContent = '';
    if (unsupported) {
      el.append(h('p', { class: 'muted', text: `Describe a change in words and a language model running on your GPU edits the genes. ${unsupported}` }));
      return;
    }
    const gb = (MODEL_BYTES / 1e9).toFixed(1);
    el.append(
      h('p', { class: 'muted', text: 'Describe a change in words ("calmer", "more blue", "fill the screen") and a language model running on your GPU edits the playing preset. Nothing leaves your machine.' }),
      h('p', { class: 'dim', text: `The model (Qwen3.5 4B) is a one-time download of about ${gb} GB from Hugging Face, kept in the browser afterwards. It needs a desktop GPU with WebGPU.` }),
    );
    const go = h('button', { class: 'btn sm primary' });
    go.innerHTML = `${icon('download', 14)}<span>Download and start (${gb} GB)</span>`;
    go.addEventListener('click', () => void this.start());
    el.append(h('div', { class: 'row' }, go));
  }

  private async start(): Promise<void> {
    if (this.state !== 'off' && this.state !== 'failed') return;
    this.setState('loading');
    this.intro.hidden = true;
    const t0 = performance.now();
    try {
      await this.llm.load((p: Progress) => {
        this.progFill.style.setProperty('--v', `${Math.round(p.progress * 100)}%`);
        this.progText.textContent = p.text;
      });
      saveSetting('chat.enabled', true);
      this.note(`Model ready in ${((performance.now() - t0) / 1000).toFixed(1)} s (${this.llm.plan?.label ?? ''}).`, 'dim');
      this.setState('warming');
      await this.warm();
      this.setState('ready');
      this.greet();
    } catch (err) {
      console.error('[chat] load failed', err);
      this.setState('failed');
      this.intro.hidden = false;
      this.note(`Could not start the model: ${err instanceof Error ? err.message : String(err)}`, 'err');
      this.renderIntro();
    }
  }

  /** Reads the system prompt and the current preset ahead of the first request. */
  private async warm(): Promise<void> {
    if (!this.deps.editor.current()) return;
    const t0 = performance.now();
    try {
      await this.chat.prewarm();
      console.info(`[chat] prewarm ${((performance.now() - t0) / 1000).toFixed(1)} s`);
    } catch (err) {
      console.warn('[chat] prewarm failed', err);
      this.chat.reset();
    }
  }

  private greet(): void {
    const ex = h('div', { class: 'gc-ex' });
    for (const e of EXAMPLES) {
      const b = h('button', { class: 'tag', text: e });
      b.addEventListener('click', () => {
        this.input.value = e;
        void this.submit();
      });
      ex.append(b);
    }
    this.log.append(h('div', { class: 'gc-msg bot' }, h('p', { text: 'What should change? For example:' }), ex));
    this.scroll();
  }

  private presetChanged(): void {
    // Undo steps belong to the preset they were made on.
    for (const u of this.undo) u.btn.disabled = true;
    this.undo = [];
  }

  // ------------------------------------------------------------ turns

  private async submit(): Promise<void> {
    const text = this.input.value.trim();
    if (!text || this.state !== 'ready') return;
    if (!this.deps.editor.current()) {
      this.note('No preset is playing yet.', 'warn');
      return;
    }
    this.input.value = '';
    this.log.append(h('div', { class: 'gc-msg me' }, h('p', { text })));
    const bot = h('div', { class: 'gc-msg bot pending' });
    const say = h('p', { class: 'gc-say' });
    const dots = h('span', { class: 'gc-dots', 'aria-label': 'thinking' }, h('i'), h('i'), h('i'));
    bot.append(say, dots);
    this.log.append(bot);
    this.scroll();
    this.setState('busy');
    this.abort = new AbortController();
    let res: TurnResult | null = null;
    try {
      res = await this.chat.send(text, { signal: this.abort.signal, onText: (s) => (say.textContent = s) });
    } catch (err) {
      console.error('[chat] turn failed', err);
      say.textContent = `Something went wrong: ${err instanceof Error ? err.message : String(err)}`;
      bot.classList.add('err');
    }
    this.abort = null;
    dots.remove();
    bot.classList.remove('pending');
    if (res) this.showResult(bot, say, res);
    this.scroll();
    // A nearly full conversation starts over now, while the user reads the reply.
    if (this.chat.full) {
      this.setState('warming');
      try {
        await this.chat.maintain();
      } catch (err) {
        console.warn('[chat] refresh failed', err);
        this.chat.reset();
      }
    }
    this.setState('ready');
    this.input.focus();
  }

  private showResult(bot: HTMLElement, say: HTMLElement, r: TurnResult): void {
    say.textContent = r.say || (r.aborted ? 'Stopped.' : r.changes.length ? 'Done.' : 'No changes.');
    const changed = JSON.stringify(r.after) !== JSON.stringify(r.before);
    if (changed && !r.aborted) this.deps.editor.applyExternal(r.after, r.touched);
    if (r.changes.length) {
      const list = h('ul', { class: 'gc-changes mono' });
      for (const c of r.changes) list.append(h('li', { text: c }));
      bot.append(list);
    }
    if (r.problems.length) {
      const p = h('ul', { class: 'gc-problems' });
      for (const c of r.problems) p.append(h('li', { text: c.replace(/^\{.*?\} failed: /, '') }));
      bot.append(p);
    }
    const s = r.stats;
    const tps = s.decodeMs > 0 ? s.completionTokens / (s.decodeMs / 1000) : 0;
    const foot = h('div', { class: 'gc-foot' }, h('span', { class: 'dim mono', text: `${(s.totalMs / 1000).toFixed(1)} s · ${Math.round(tps)} tok/s${r.repaired ? ' · repaired' : ''}` }), h('span', { class: 'grow' }));
    if (changed && !r.aborted) {
      const btn = h('button', { class: 'btn sm ghost', title: 'Undo this turn' });
      btn.innerHTML = `${icon('undo', 13)}<span>Undo</span>`;
      const step = { before: cloneGenome(r.before), after: cloneGenome(r.after), btn };
      btn.addEventListener('click', () => this.undoTo(step));
      this.undo.push(step);
      foot.append(btn);
    }
    bot.append(foot);
  }

  /** Undoes the turns back to (and including) this one, newest first. */
  private undoTo(step: { before: Genome; after: Genome; btn: HTMLButtonElement }): void {
    const i = this.undo.indexOf(step);
    if (i < 0) return;
    const cur = this.deps.editor.current();
    const last = this.undo[this.undo.length - 1];
    if (!cur || JSON.stringify(cur) !== JSON.stringify(last.after)) {
      this.note('The genes changed by hand since then, so undo is off for earlier turns.', 'warn');
      for (const u of this.undo) u.btn.disabled = true;
      this.undo = [];
      return;
    }
    this.deps.editor.applyExternal(step.before, []);
    for (const u of this.undo.splice(i)) {
      u.btn.disabled = true;
      u.btn.querySelector('span')!.textContent = 'Undone';
    }
  }

  private note(text: string, kind: 'dim' | 'warn' | 'err'): void {
    this.log.append(h('div', { class: `gc-note ${kind}`, text }));
    this.scroll();
  }

  private scroll(): void {
    this.log.scrollTop = this.log.scrollHeight;
  }

  private dragResize(grip: HTMLElement): void {
    let startY = 0;
    let startH = 0;
    const move = (ev: PointerEvent) => {
      const tab = this.root.parentElement?.getBoundingClientRect().height ?? 800;
      const hgt = Math.max(MIN_H, Math.min(tab * 0.75, startH - (ev.clientY - startY)));
      this.root.style.setProperty('--gc-h', `${Math.round(hgt)}px`);
    };
    const up = () => {
      window.removeEventListener('pointermove', move);
      window.removeEventListener('pointerup', up);
      saveSetting('chat.h', Math.round(this.root.getBoundingClientRect().height));
    };
    grip.addEventListener('pointerdown', (ev) => {
      if (this.collapsed) return;
      ev.preventDefault();
      startY = ev.clientY;
      startH = this.root.getBoundingClientRect().height;
      window.addEventListener('pointermove', move);
      window.addEventListener('pointerup', up);
    });
  }
}
