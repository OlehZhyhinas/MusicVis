// The gene chat's conversation: one growing conversation with the local model, so the engine keeps
// the system prompt and history cached and each turn only reads the new message. A turn sends the
// preset (in full when it is new to the conversation, as a diff after manual edits) with the screen
// metrics and the request, decodes a grammar-constrained JSON reply, applies it through the editor's
// edit functions, and gives the model one repair round with the exact errors when edits fail.

import { COST_BUDGET_MS, estimateCost, type Genome } from '../v2/genome';
import { repairKeeping } from '../v2/geneEdit';
import { applyEdits, replySchema, type Edit, type Reply } from './edits';
import { CONTEXT_TOKENS, type ChatMessage, type GenerateResult, type LocalLLM } from './llm';
import { entryKey, genomeDiff, genomeText, lookText, mentionedKeys, notesText, presentKeys, systemPrompt, type LookMetrics } from './prompt';

export interface ChatContext {
  /** The genome being edited now (the editor's scratch copy). */
  genome(): Genome | null;
  /** Identity of the preset being edited (a new one means a new preset for the model). */
  presetId(): string;
  presetLabel(): string;
  keyHue(): number;
  look(): LookMetrics | null;
}

export interface TurnResult {
  say: string;
  /** The genome before and after the turn (after === before when nothing applied). */
  before: Genome;
  after: Genome;
  changes: string[];
  /** Problems left after the repair round (shown to the user). */
  problems: string[];
  touched: string[];
  repaired: boolean;
  aborted: boolean;
  stats: { promptTokens: number; completionTokens: number; prefillMs: number; decodeMs: number; totalMs: number };
}

/** Tokens reserved for a turn's message, reply and repair round before the context counts as full. */
const TURN_RESERVE = 1400;
const MAX_REPLY_TOKENS = 360;
/** More manual changes than this and the whole preset is sent again instead of a diff. */
const MAX_DIFF_LINES = 10;

/** The JSON object in a reply (the model may emit an empty think block first). */
export function parseReply(text: string): Reply | null {
  const i = text.indexOf('{');
  const j = text.lastIndexOf('}');
  if (i < 0 || j < i) return null;
  try {
    const r = JSON.parse(text.slice(i, j + 1)) as Partial<Reply>;
    return { say: typeof r.say === 'string' ? r.say : '', edits: Array.isArray(r.edits) ? (r.edits as Edit[]) : [] };
  } catch {
    return null;
  }
}

/** The say text streamed so far (for showing the reply while it decodes). */
export function partialSay(text: string): string {
  const m = /"say"\s*:\s*"((?:[^"\\]|\\.)*)/.exec(text);
  if (!m) return '';
  try {
    return JSON.parse(`"${m[1].replace(/\\$/, '')}"`) as string;
  } catch {
    return m[1];
  }
}

/** Entry keys of the kinds a reply's edits name (their notes go with the repair round). */
function failedKeys(edits: Edit[], g: Genome): string[] {
  const out: string[] = [];
  for (const e of edits) {
    if (!e || typeof e !== 'object') continue;
    if (e.op === 'kind') {
      const m = /^b\d\.(\w+)$/.exec(e.path);
      if (m) out.push(entryKey(m[1], e.kind));
      else if (e.path === 'carrier' || e.path === 'palette') out.push(entryKey(e.path, e.kind));
      else if (/^op\d$/.test(e.path)) out.push(entryKey('op', e.kind));
    } else if (e.op === 'add_op' || e.op === 'add_deform_op') out.push(entryKey('op', e.kind));
    else if (e.op === 'add_body' || e.op === 'fuse') out.push(entryKey('shape', e.shape));
    else if (e.op === 'add_gene') out.push(`gene.${e.gene}`);
  }
  return [...out, ...presentKeys(g)];
}

export class GeneChat {
  private messages: ChatMessage[] = [{ role: 'system', content: systemPrompt() }];
  /** The genome as the model last saw it (after its own edits), and whose preset it was. */
  private seen: Genome | null = null;
  private seenPreset = '';
  private usedTokens = 0;
  private busyNow = false;
  /** Glossary entries the conversation has already been given (sent once, then remembered). */
  private described = new Set<string>();
  /** The last turn's model calls (message sent, raw reply, token counts), for debugging and the test set. */
  trace: { sent: string; raw: string; promptTokens: number; completionTokens: number; prefillMs: number; decodeMs: number }[] = [];

  private readonly llm: LocalLLM;
  private readonly ctx: ChatContext;

  constructor(llm: LocalLLM, ctx: ChatContext) {
    this.llm = llm;
    this.ctx = ctx;
  }

  get busy(): boolean {
    return this.busyNow;
  }

  /** Forgets the conversation (the next turn reads the system prompt again). */
  reset(): void {
    this.messages = [{ role: 'system', content: systemPrompt() }];
    this.seen = null;
    this.seenPreset = '';
    this.usedTokens = 0;
    this.described.clear();
  }

  /** "Gene notes" for the entries among `keys` the conversation has not seen yet (marks them seen). */
  private notes(keys: string[]): string {
    const fresh = keys.filter((k) => !this.described.has(k));
    for (const k of fresh) this.described.add(k);
    const text = notesText(fresh);
    return text ? `Gene notes:\n${text}\n` : '';
  }

  private stateText(g: Genome): string {
    const preset = this.ctx.presetId();
    const look = lookText(this.ctx.look());
    if (this.seen && preset === this.seenPreset) {
      const d = genomeDiff(this.seen, g, this.ctx.keyHue());
      if (d && d.length === 0) return look;
      if (d && d.length <= MAX_DIFF_LINES) return `The user changed these by hand since your last reply:\n${d.join('\n')}\n${look}`;
    }
    const intro = this.seenPreset && preset !== this.seenPreset ? 'The preset changed. ' : '';
    return `${intro}Preset ${this.ctx.presetLabel()}:\n${genomeText(g, this.ctx.keyHue())}\n${look}`;
  }

  private remember(r: GenerateResult): void {
    // WebLLM reports only the newly read tokens of a continued conversation.
    this.usedTokens += r.promptTokens + r.completionTokens;
  }

  /** True when the next turn would overflow the context (the conversation starts over then). */
  get full(): boolean {
    return this.usedTokens + TURN_RESERVE > CONTEXT_TOKENS;
  }

  /** Between turns: when the context is nearly full, start over and read the prompt again now. */
  async maintain(): Promise<boolean> {
    if (!this.full || this.busyNow) return false;
    this.reset();
    await this.prewarm();
    return true;
  }

  /**
   * Reads the system prompt and the current preset ahead of the first request, so the first real
   * turn only reads the request. Resolves when done.
   */
  async prewarm(signal?: AbortSignal): Promise<void> {
    const g = this.ctx.genome();
    if (!g || this.busyNow || this.messages.length > 1) return;
    this.busyNow = true;
    try {
      const msg = `${this.notes(presentKeys(g))}${this.stateText(g)}\nRequest: nothing yet, I will ask next. Reply with no edits.`;
      const msgs: ChatMessage[] = [...this.messages, { role: 'user', content: msg }];
      const r = await this.llm.generate(msgs, { schema: replySchema(g), maxTokens: 40, signal });
      if (r.aborted) return;
      this.messages = [...msgs, { role: 'assistant', content: r.text }];
      this.seen = g;
      this.seenPreset = this.ctx.presetId();
      this.usedTokens = r.promptTokens + r.completionTokens;
    } finally {
      this.busyNow = false;
    }
  }

  async send(request: string, opts: { signal?: AbortSignal; onText?: (say: string) => void } = {}): Promise<TurnResult> {
    const before = this.ctx.genome();
    if (!before) throw new Error('no preset to edit');
    if (this.busyNow) throw new Error('busy');
    this.busyNow = true;
    const t0 = performance.now();
    const stats = { promptTokens: 0, completionTokens: 0, prefillMs: 0, decodeMs: 0, totalMs: 0 };
    this.trace = [];
    const add = (r: GenerateResult, sent = '') => {
      this.trace.push({ sent, raw: r.text, promptTokens: r.promptTokens, completionTokens: r.completionTokens, prefillMs: Math.round(r.prefillMs), decodeMs: Math.round(r.decodeMs) });
      stats.promptTokens += r.promptTokens;
      stats.completionTokens += r.completionTokens;
      stats.prefillMs += r.prefillMs;
      stats.decodeMs += r.decodeMs;
    };
    try {
      if (this.usedTokens + TURN_RESERVE > CONTEXT_TOKENS) this.reset();
      const preset = this.ctx.presetId();
      const user = `${this.notes([...presentKeys(before), ...mentionedKeys(request)])}${this.stateText(before)}\nRequest: ${request.trim()}`;
      const msgs: ChatMessage[] = [...this.messages, { role: 'user', content: user }];
      const r1 = await this.llm.generate(msgs, { schema: replySchema(before), maxTokens: MAX_REPLY_TOKENS, signal: opts.signal, onText: (t) => opts.onText?.(partialSay(t)) });
      add(r1, user);
      const done = (res: Omit<TurnResult, 'before' | 'stats'>): TurnResult => {
        stats.totalMs = performance.now() - t0;
        return { ...res, before, stats };
      };
      if (r1.aborted) {
        // The conversation state inside the engine is now partial: start clean next time.
        this.reset();
        return done({ say: partialSay(r1.text), after: before, changes: [], problems: [], touched: [], repaired: false, aborted: true });
      }
      this.messages = [...msgs, { role: 'assistant', content: r1.text }];
      this.remember(r1);
      const reply = parseReply(r1.text);
      let g = before;
      let touched: string[] = [];
      let changes: string[] = [];
      let errors: string[] = [];
      let say = reply?.say ?? '';
      if (!reply) errors = ['your reply was not valid JSON'];
      else {
        const a = applyEdits(before, reply.edits, { keyHue: this.ctx.keyHue() });
        ({ genome: g, touched, changes } = a);
        errors = a.errors;
      }
      const hard = errors.filter((e) => !e.startsWith('note:'));
      let repaired = false;
      if (hard.length && !opts.signal?.aborted) {
        repaired = true;
        const fix = `${this.notes(failedKeys(reply?.edits ?? [], g))}These edits did not work:\n${hard.map((e) => `- ${e}`).join('\n')}\nThe other edits were applied. Reply with JSON holding only replacement edits that do what was asked within the rules (or no edits if it cannot be done), and a "say" for the user.`;
        const msgs2: ChatMessage[] = [...this.messages, { role: 'user', content: fix }];
        const r2 = await this.llm.generate(msgs2, { schema: replySchema(g), maxTokens: MAX_REPLY_TOKENS, signal: opts.signal, onText: (t) => opts.onText?.(partialSay(t)) });
        add(r2, fix);
        if (r2.aborted) this.reset();
        else {
          this.messages = [...msgs2, { role: 'assistant', content: r2.text }];
          this.remember(r2);
          const reply2 = parseReply(r2.text);
          if (reply2) {
            const b = applyEdits(g, reply2.edits, { keyHue: this.ctx.keyHue() });
            g = b.genome;
            touched = [...touched, ...b.touched];
            changes = [...changes, ...b.changes];
            errors = b.errors;
            if (reply2.say) say = reply2.say;
          }
        }
      }
      const problems = errors.filter((e) => !e.startsWith('note:'));
      if (estimateCost(g) > COST_BUDGET_MS) {
        g = repairKeeping(g);
        changes.push(`reduced copies to fit the ${COST_BUDGET_MS} ms budget`);
      }
      if (this.ctx.presetId() !== preset) {
        return done({ say: 'The preset changed while I was working, so nothing was applied.', after: this.ctx.genome() ?? before, changes: [], problems: [], touched: [], repaired, aborted: false });
      }
      this.seen = g;
      this.seenPreset = preset;
      return done({ say, after: g, changes, problems, touched, repaired, aborted: false });
    } finally {
      this.busyNow = false;
    }
  }
}
