// Tiny Chrome DevTools Protocol client plus server management for the AVQ harness:
// one Vite dev server (scripts/avq/vite.avq.config.ts) on AVQ_VITE_PORT (5231) and one
// headless Chrome on AVQ_CDP_PORT (5232); parallel jobs run in separate tabs of that Chrome.

import { spawn, type ChildProcess } from 'node:child_process';
import { mkdirSync, openSync } from 'node:fs';
import { join, resolve } from 'node:path';

export const ROOT = resolve(import.meta.dirname, '../..');
export const OUT = join(ROOT, '.testdata/avq');
export const VITE_PORT = Number(process.env.AVQ_VITE_PORT ?? 5231);
export const CDP_PORT = Number(process.env.AVQ_CDP_PORT ?? 5232);
const CHROME = process.env.AVQ_CHROME ?? '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
export const PAGE_URL = `http://127.0.0.1:${VITE_PORT}/scripts/avq/page.html`;

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

async function up(url: string): Promise<boolean> {
  try {
    const r = await fetch(url, { signal: AbortSignal.timeout(1500) });
    return r.ok;
  } catch {
    return false;
  }
}

export interface Servers {
  stop(): void;
}

/** Start Vite and headless Chrome unless they already answer; stop() kills only what was started here. */
export async function ensureServers(log = (s: string) => console.error(s)): Promise<Servers> {
  const started: ChildProcess[] = [];
  mkdirSync(join(OUT, 'logs'), { recursive: true });
  if (!(await up(PAGE_URL))) {
    log(`starting vite on ${VITE_PORT}`);
    const out = openSync(join(OUT, 'logs/vite.log'), 'a');
    started.push(spawn(join(ROOT, 'node_modules/.bin/vite'), ['--config', 'scripts/avq/vite.avq.config.ts', '--port', String(VITE_PORT), '--host', '127.0.0.1'], { cwd: ROOT, stdio: ['ignore', out, out], detached: false }));
    for (let i = 0; i < 60 && !(await up(PAGE_URL)); i++) await sleep(500);
    if (!(await up(PAGE_URL))) throw new Error('vite did not come up');
  }
  if (!(await up(`http://127.0.0.1:${CDP_PORT}/json/version`))) {
    log(`starting headless chrome on ${CDP_PORT}`);
    const out = openSync(join(OUT, 'logs/chrome.log'), 'a');
    const prof = join(OUT, 'chrome-profile');
    started.push(spawn(CHROME, [
      '--headless=new', `--remote-debugging-port=${CDP_PORT}`, `--user-data-dir=${prof}`,
      '--use-angle=metal', '--enable-gpu', '--ignore-gpu-blocklist', '--enable-unsafe-webgpu',
      '--disable-background-timer-throttling', '--disable-renderer-backgrounding', '--disable-backgrounding-occluded-windows',
      '--no-first-run', '--no-default-browser-check', '--mute-audio', 'about:blank',
    ], { stdio: ['ignore', out, out] }));
    for (let i = 0; i < 60 && !(await up(`http://127.0.0.1:${CDP_PORT}/json/version`)); i++) await sleep(500);
    if (!(await up(`http://127.0.0.1:${CDP_PORT}/json/version`))) throw new Error('chrome did not come up');
  }
  return {
    stop() {
      for (const p of started) p.kill('SIGTERM');
    },
  };
}

export class Tab {
  private ws!: WebSocket;
  private id = 0;
  private pending = new Map<number, (m: { result?: unknown; error?: { message: string } }) => void>();
  readonly logs: string[] = [];
  targetId = '';
  /** Node's WebSocket does not keep the event loop alive while we wait on Chrome. */
  private keepAlive: ReturnType<typeof setInterval> | null = null;

  static async open(): Promise<Tab> {
    const t = new Tab();
    t.keepAlive = setInterval(() => undefined, 1000);
    const target = (await (await fetch(`http://127.0.0.1:${CDP_PORT}/json/new?about:blank`, { method: 'PUT' })).json()) as { id: string; webSocketDebuggerUrl: string };
    t.targetId = target.id;
    t.ws = new WebSocket(target.webSocketDebuggerUrl);
    await new Promise((r, f) => {
      t.ws.addEventListener('open', r, { once: true });
      t.ws.addEventListener('error', f, { once: true });
    });
    t.ws.addEventListener('message', (ev) => {
      const m = JSON.parse(String(ev.data));
      if (m.id && t.pending.has(m.id)) {
        t.pending.get(m.id)!(m);
        t.pending.delete(m.id);
      } else if (m.method === 'Runtime.consoleAPICalled') {
        const line = `${m.params.type}: ${m.params.args.map((a: { value?: unknown; description?: string }) => a.value ?? a.description ?? '').join(' ')}`;
        if (m.params.type === 'error' || m.params.type === 'warning') t.logs.push(line);
      } else if (m.method === 'Runtime.exceptionThrown') {
        t.logs.push('EXC: ' + (m.params.exceptionDetails.exception?.description ?? m.params.exceptionDetails.text));
      }
    });
    await t.send('Runtime.enable');
    await t.send('Page.enable');
    return t;
  }

  send<T = unknown>(method: string, params: Record<string, unknown> = {}): Promise<T> {
    return new Promise((ok, fail) => {
      const i = ++this.id;
      this.pending.set(i, (m) => (m.error ? fail(new Error(`${method}: ${m.error.message}`)) : ok(m.result as T)));
      this.ws.send(JSON.stringify({ id: i, method, params }));
    });
  }

  /** Evaluate an expression (awaiting promises); page-side rejections come back as thrown Errors. */
  async eval<T = unknown>(expr: string): Promise<T> {
    const wrapped = `Promise.resolve().then(() => (${expr})).then((v) => ({ ok: v }), (e) => ({ err: String(e && e.stack || e) }))`;
    const out = await this.evalRaw<{ ok?: T; err?: string }>(wrapped);
    if (out && out.err) throw new Error(out.err);
    return out?.ok as T;
  }

  async evalRaw<T = unknown>(expr: string): Promise<T> {
    const r = await this.send<{ result: { value?: T }; exceptionDetails?: { exception?: { description?: string }; text: string } }>('Runtime.evaluate', { expression: expr, awaitPromise: true, returnByValue: true });
    if (r.exceptionDetails) throw new Error(r.exceptionDetails.exception?.description ?? r.exceptionDetails.text);
    return r.result.value as T;
  }

  /** (Re)load the harness page and wait for window.avq. */
  async load(): Promise<void> {
    await this.send('Page.navigate', { url: PAGE_URL });
    for (let i = 0; i < 200; i++) {
      await sleep(100);
      try {
        if (await this.evalRaw<boolean>('!!window.avq')) return;
      } catch {
        /* navigating */
      }
    }
    throw new Error('harness page did not load: ' + this.logs.slice(-5).join(' | '));
  }

  async close(): Promise<void> {
    if (this.keepAlive) clearInterval(this.keepAlive);
    this.ws.close();
    await fetch(`http://127.0.0.1:${CDP_PORT}/json/close/${this.targetId}`).catch(() => undefined);
  }
}
