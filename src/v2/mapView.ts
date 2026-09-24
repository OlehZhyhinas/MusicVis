// Map view of the Presets tab: a spring graph of looks. Each preset is a
// thumbnail disc; springs link it to its nearest neighbours by phenotype
// distance, so presets that look alike sit together. Canvas 2D; the
// simulation runs only while the map is on screen and unsettled, and redraws
// happen on demand, so a settled map costs nothing per frame.

import type { Member } from './population';
import { fitness } from './population';
import { SpringLayout } from './springLayout';
import { loadSetting, saveSetting } from '../ui/storage';

export interface MapCallbacks {
  /** The presets to show (the browser's filters, including Show hidden). */
  members(): Member[];
  currentId(): string | null;
  /** Phenotype distance, NaN when either has no fingerprint yet. */
  distance(a: Member, b: Member): number;
  novelty(m: Member): { nov: number; rel: number } | null;
  thumb(id: string): Promise<string>;
  /** Changes when the distance metric changes (fitted weights): the graph is rebuilt. */
  metricVersion?(): number;
  /** Play the preset and open it in the Genes tab with the HUD on. */
  open(id: string): void;
}

type RingMode = 'score' | 'energy';
const VIEW_KEY = 'v2.presetView';
const RING_KEY = 'v2.mapRing';
const STEP_BUDGET_MS = 3;

export type PresetView = 'list' | 'map';

export function loadPresetView(): PresetView {
  return loadSetting<string>(VIEW_KEY, 'list') === 'map' ? 'map' : 'list';
}

export class PresetMap {
  readonly host: HTMLElement;
  private canvas: HTMLCanvasElement;
  private ctx: CanvasRenderingContext2D;
  private tip: HTMLElement;
  private layout = new SpringLayout();
  private members = new Map<string, Member>();
  private imgs = new Map<string, HTMLImageElement | 'loading' | 'failed'>();
  private shown = false;
  private raf = 0;
  private dirty = true;
  private camX = 0;
  private camY = 0;
  private zoom = 1;
  private fitted = false;
  private hover: string | null = null;
  private ring: RingMode = loadSetting<string>(RING_KEY, 'score') === 'energy' ? 'energy' : 'score';
  private sig = '';
  private syncTimer = 0;
  private lastSync = 0;
  private drag: { kind: 'pan' | 'node'; id?: string; x: number; y: number; moved: number } | null = null;
  private dpr = 1;
  private w = 0;
  private h = 0;
  /** Debug / perf: last sync ms, steps run, average step ms. */
  stats = { syncMs: 0, steps: 0, stepMs: 0, drawMs: 0, nodes: 0, edges: 0 };

  constructor(private cb: MapCallbacks) {
    this.host = document.createElement('div');
    this.host.className = 'v2b-map';
    this.host.hidden = true;
    this.host.innerHTML = `
      <canvas aria-label="Map of presets: similar looks sit together" role="img"></canvas>
      <div class="map-ctl">
        <div class="seg" role="group" aria-label="Ring colour">
          <button data-ring="score" aria-pressed="false" title="Ring colour: score (grey = no votes yet)">Score</button><button data-ring="energy" aria-pressed="false" title="Ring colour: calm / energetic">Energy</button>
        </div>
        <button class="btn sm" data-fit title="Fit every preset in view">Fit</button>
      </div>
      <div class="map-legend dim">Springs link each preset to its nearest looks · drag to pan · scroll to zoom · click to open in Genes</div>
      <div class="map-tip" hidden></div>`;
    this.canvas = this.host.querySelector('canvas')!;
    this.ctx = this.canvas.getContext('2d')!;
    this.tip = this.host.querySelector('.map-tip')!;
    this.setRing(this.ring);
    this.host.querySelector('.map-ctl .seg')!.addEventListener('click', (ev) => {
      const b = (ev.target as HTMLElement).closest<HTMLButtonElement>('button[data-ring]');
      if (b) this.setRing(b.dataset.ring === 'energy' ? 'energy' : 'score');
    });
    this.host.querySelector('[data-fit]')!.addEventListener('click', () => this.fit());
    new ResizeObserver(() => this.resize()).observe(this.host);
    this.canvas.addEventListener('pointerdown', (e) => this.onDown(e));
    this.canvas.addEventListener('pointermove', (e) => this.onMove(e));
    this.canvas.addEventListener('pointerup', (e) => this.onUp(e));
    this.canvas.addEventListener('pointerleave', () => this.setHover(null));
    this.canvas.addEventListener('wheel', (e) => this.onWheel(e), { passive: false });
    document.addEventListener('visibilitychange', () => this.kick());
  }

  get isShown(): boolean {
    return this.shown;
  }

  setShown(on: boolean): void {
    this.shown = on;
    this.host.hidden = !on;
    if (on) {
      this.resize();
      this.sync(true);
    } else {
      cancelAnimationFrame(this.raf);
      this.raf = 0;
      this.setHover(null);
    }
  }

  private setRing(r: RingMode): void {
    this.ring = r;
    saveSetting(RING_KEY, r);
    for (const b of this.host.querySelectorAll<HTMLButtonElement>('button[data-ring]')) b.setAttribute('aria-pressed', String(b.dataset.ring === r));
    this.redraw();
  }

  /** The population or the filters changed: rebuild the graph (throttled; the layout keeps positions). */
  sync(now = false): void {
    if (!this.shown) return;
    const wait = 600 - (performance.now() - this.lastSync);
    if (!now && wait > 0) {
      if (!this.syncTimer) this.syncTimer = window.setTimeout(() => {
        this.syncTimer = 0;
        this.sync(true);
      }, wait);
      return;
    }
    this.lastSync = performance.now();
    const ms = this.cb.members();
    const sig = `${this.cb.metricVersion?.() ?? 0}:` + ms.map((m) => `${m.id}${m.fp ? '+' : ''}${m.hidden ? 'h' : ''}`).join(',');
    this.members = new Map(ms.map((m) => [m.id, m]));
    if (sig === this.sig) {
      this.redraw();
      return;
    }
    this.sig = sig;
    const t0 = performance.now();
    const n = ms.length;
    const D = new Float32Array(n * n);
    for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) D[i * n + j] = D[j * n + i] = this.cb.distance(ms[i], ms[j]);
    const idx = new Map(ms.map((m, i) => [m.id, i]));
    this.layout.setGraph(
      ms.map((m) => m.id),
      (i, j) => D[i * n + j],
      (i) => ms[i].parents.map((p) => idx.get(p) ?? -1),
    );
    if (!this.fitted) {
      this.layout.settle(260);
      this.fit();
    }
    this.stats.syncMs = performance.now() - t0;
    this.stats.nodes = n;
    this.stats.edges = this.layout.edges.length;
    for (const id of this.imgs.keys()) if (!this.members.has(id)) this.imgs.delete(id);
    this.kick();
  }

  markCurrent(): void {
    this.redraw();
  }

  private redraw(): void {
    this.dirty = true;
    this.kick();
  }

  /** Make sure a frame is coming when there is something to do. */
  private kick(): void {
    if (!this.shown || this.raf || document.hidden) return;
    if (!this.dirty && this.layout.settled && !this.drag) return;
    this.raf = requestAnimationFrame(() => this.frame());
  }

  private frame(): void {
    this.raf = 0;
    if (!this.shown || document.hidden) return;
    if (!this.layout.settled || this.drag?.kind === 'node') {
      const t0 = performance.now();
      let k = 0;
      do {
        this.layout.step();
        k++;
      } while (performance.now() - t0 < STEP_BUDGET_MS * 0.5 && k < 3 && !this.layout.settled);
      const dt = performance.now() - t0;
      this.stats.steps += k;
      this.stats.stepMs += (dt / k - this.stats.stepMs) * 0.1;
      this.dirty = true;
    }
    if (this.dirty) this.draw();
    this.kick();
  }

  private resize(): void {
    const r = this.host.getBoundingClientRect();
    this.dpr = Math.min(2, window.devicePixelRatio || 1);
    this.w = Math.max(1, r.width);
    this.h = Math.max(1, r.height);
    this.canvas.width = Math.round(this.w * this.dpr);
    this.canvas.height = Math.round(this.h * this.dpr);
    this.canvas.style.width = `${this.w}px`;
    this.canvas.style.height = `${this.h}px`;
    this.redraw();
  }

  fit(): void {
    const b = this.layout.bounds();
    const pad = 40;
    this.camX = (b.x0 + b.x1) / 2;
    this.camY = (b.y0 + b.y1) / 2;
    this.zoom = Math.max(0.15, Math.min(2.5, Math.min((this.w - 2 * pad) / Math.max(1, b.x1 - b.x0), (this.h - 2 * pad) / Math.max(1, b.y1 - b.y0))));
    this.fitted = this.w > 1;
    this.redraw();
  }

  private toScreen(x: number, y: number): [number, number] {
    return [(x - this.camX) * this.zoom + this.w / 2, (y - this.camY) * this.zoom + this.h / 2];
  }

  private toWorld(sx: number, sy: number): [number, number] {
    return [(sx - this.w / 2) / this.zoom + this.camX, (sy - this.h / 2) / this.zoom + this.camY];
  }

  private radius(): number {
    return Math.max(6, Math.min(28, 13 * Math.sqrt(this.zoom)));
  }

  private ringColour(m: Member): string {
    if (this.ring === 'energy') return m.energy === 'energetic' ? '#ffc766' : '#7ee2a8';
    const votes = m.likes + m.dislikes + m.weakLikes + m.softDislikes;
    if (!votes) return 'rgba(169,171,189,0.55)';
    const f = Math.max(0, Math.min(1, (fitness(m) - 0.1) / 0.6));
    return `hsl(${Math.round(350 + f * 140) % 360}, 80%, 64%)`;
  }

  private image(id: string): HTMLImageElement | null {
    const v = this.imgs.get(id);
    if (v instanceof HTMLImageElement) return v;
    if (v) return null;
    this.imgs.set(id, 'loading');
    void this.cb.thumb(id).then((url) => {
      if (!url) {
        this.imgs.set(id, 'failed');
        return;
      }
      const img = new Image();
      img.onload = () => {
        this.imgs.set(id, img);
        this.redraw();
      };
      img.onerror = () => this.imgs.set(id, 'failed');
      img.src = url;
    });
    return null;
  }

  private draw(): void {
    const t0 = performance.now();
    this.dirty = false;
    const c = this.ctx;
    c.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
    c.clearRect(0, 0, this.w, this.h);
    const ns = this.layout.nodes;
    const pos = ns.map((n) => this.toScreen(n.x, n.y));
    const cur = this.cb.currentId();
    const hv = this.hover;
    // Edges.
    c.lineWidth = 1;
    for (const e of this.layout.edges) {
      const [ax, ay] = pos[e.a], [bx, by] = pos[e.b];
      const lit = hv && (ns[e.a].id === hv || ns[e.b].id === hv);
      c.strokeStyle = lit ? 'rgba(127,216,255,0.55)' : 'rgba(255,255,255,0.07)';
      c.beginPath();
      c.moveTo(ax, ay);
      c.lineTo(bx, by);
      c.stroke();
    }
    const r = this.radius();
    const order = ns.map((_, i) => i).sort((a, b) => Number(ns[a].id === cur || ns[a].id === hv) - Number(ns[b].id === cur || ns[b].id === hv));
    for (const i of order) {
      const nd = ns[i];
      const m = this.members.get(nd.id);
      if (!m) continue;
      const [x, y] = pos[i];
      if (x < -r * 2 || y < -r * 2 || x > this.w + r * 2 || y > this.h + r * 2) continue;
      const big = nd.id === hv ? 1.35 : nd.id === cur ? 1.2 : 1;
      const rr = r * big;
      c.globalAlpha = m.hidden ? 0.35 : 1;
      c.save();
      c.beginPath();
      c.arc(x, y, rr, 0, Math.PI * 2);
      c.fillStyle = '#10121a';
      c.fill();
      const img = this.image(nd.id);
      if (img) {
        c.clip();
        // Centre-crop the 16:9 still into the disc.
        const s = (2 * rr) / Math.min(img.width, img.height);
        c.drawImage(img, x - (img.width * s) / 2, y - (img.height * s) / 2, img.width * s, img.height * s);
      }
      c.restore();
      c.beginPath();
      c.arc(x, y, rr + 1, 0, Math.PI * 2);
      c.lineWidth = nd.id === cur ? 3 : 2;
      c.strokeStyle = nd.id === cur ? '#7fd8ff' : this.ringColour(m);
      c.stroke();
      if (nd.id === cur) {
        c.beginPath();
        c.arc(x, y, rr + 6, 0, Math.PI * 2);
        c.lineWidth = 1.5;
        c.strokeStyle = 'rgba(127,216,255,0.45)';
        c.stroke();
      }
      c.globalAlpha = 1;
    }
    this.stats.drawMs += (performance.now() - t0 - this.stats.drawMs) * 0.1;
  }

  // ------------------------------------------------------ interaction

  private hit(sx: number, sy: number): string | null {
    const r = this.radius() * 1.15;
    let best: string | null = null;
    let bd = r * r;
    for (const nd of this.layout.nodes) {
      const [x, y] = this.toScreen(nd.x, nd.y);
      const d = (x - sx) ** 2 + (y - sy) ** 2;
      if (d < bd) {
        bd = d;
        best = nd.id;
      }
    }
    return best;
  }

  private local(e: PointerEvent | WheelEvent): [number, number] {
    const r = this.canvas.getBoundingClientRect();
    return [e.clientX - r.left, e.clientY - r.top];
  }

  private onDown(e: PointerEvent): void {
    const [x, y] = this.local(e);
    const id = this.hit(x, y);
    try {
      this.canvas.setPointerCapture(e.pointerId);
    } catch {
      // synthetic events have no active pointer
    }
    this.drag = { kind: id ? 'node' : 'pan', id: id ?? undefined, x, y, moved: 0 };
    if (id) {
      const nd = this.layout.get(id);
      if (nd) nd.fixed = true;
    }
  }

  private onMove(e: PointerEvent): void {
    const [x, y] = this.local(e);
    const d = this.drag;
    if (!d) {
      this.setHover(this.hit(x, y), x, y);
      return;
    }
    const dx = x - d.x, dy = y - d.y;
    d.moved += Math.abs(dx) + Math.abs(dy);
    d.x = x;
    d.y = y;
    if (d.kind === 'pan') {
      this.camX -= dx / this.zoom;
      this.camY -= dy / this.zoom;
      this.redraw();
    } else if (d.moved > 4 && d.id) {
      const nd = this.layout.get(d.id);
      if (nd) {
        [nd.x, nd.y] = this.toWorld(x, y);
        this.layout.reheat(0.12);
        this.setHover(d.id, x, y);
        this.redraw();
      }
    }
  }

  private onUp(e: PointerEvent): void {
    const d = this.drag;
    this.drag = null;
    try {
      this.canvas.releasePointerCapture(e.pointerId);
    } catch {
      // not captured
    }
    if (!d) return;
    if (d.id) {
      const nd = this.layout.get(d.id);
      if (nd) nd.fixed = false;
    }
    if (d.kind === 'node' && d.moved <= 4 && d.id) this.cb.open(d.id);
    this.kick();
  }

  private onWheel(e: WheelEvent): void {
    e.preventDefault();
    const [x, y] = this.local(e);
    const [wx, wy] = this.toWorld(x, y);
    this.zoom = Math.max(0.1, Math.min(5, this.zoom * Math.exp(-e.deltaY * 0.0015)));
    // Keep the point under the cursor fixed.
    this.camX = wx - (x - this.w / 2) / this.zoom;
    this.camY = wy - (y - this.h / 2) / this.zoom;
    this.setHover(this.hit(x, y), x, y);
    this.redraw();
  }

  private setHover(id: string | null, x = 0, y = 0): void {
    if (id !== this.hover) {
      this.hover = id;
      this.canvas.style.cursor = id ? 'pointer' : 'grab';
      this.redraw();
    }
    const m = id ? this.members.get(id) : undefined;
    if (!m) {
      this.tip.hidden = true;
      return;
    }
    const nv = this.cb.novelty(m);
    this.tip.innerHTML = `<div class="row" style="gap:6px"><span class="id">${esc(m.id)}</span><b class="ell">${esc(m.name)}</b></div><div class="dim ell">${esc(m.type)} · ${m.energy}</div><div class="dim">${nv ? `novelty ${Math.round(nv.rel * 100)}` : 'novelty not measured yet'} · score ${Math.round(fitness(m) * 100)}% · ${m.likes}↑ ${m.dislikes}↓${m.hidden ? ' · hidden' : ''}</div>`;
    this.tip.hidden = false;
    const tw = this.tip.offsetWidth, th = this.tip.offsetHeight;
    this.tip.style.left = `${Math.max(4, Math.min(this.w - tw - 4, x + 14))}px`;
    this.tip.style.top = `${y + 16 + th > this.h ? y - th - 12 : y + 16}px`;
  }
}

/** The List | Map switch, added to the Presets tab's tools row. */
export class ViewSwitch {
  private seg: HTMLElement;
  view: PresetView;

  constructor(tools: HTMLElement, view: PresetView, private onView: (v: PresetView) => void) {
    this.view = view;
    const wrap = document.createElement('div');
    wrap.className = 'seg';
    wrap.setAttribute('role', 'group');
    wrap.setAttribute('aria-label', 'View');
    wrap.innerHTML = `<button data-view="list" aria-pressed="${view === 'list'}">List</button><button data-view="map" aria-pressed="${view === 'map'}" title="Map: similar-looking presets sit together">Map</button>`;
    tools.append(wrap);
    this.seg = wrap;
    wrap.addEventListener('click', (ev) => {
      const b = (ev.target as HTMLElement).closest<HTMLButtonElement>('button[data-view]');
      if (b) this.set(b.dataset.view === 'map' ? 'map' : 'list');
    });
  }

  set(v: PresetView): void {
    this.view = v;
    saveSetting(VIEW_KEY, v);
    for (const b of this.seg.querySelectorAll<HTMLButtonElement>('button')) b.setAttribute('aria-pressed', String(b.dataset.view === v));
    this.onView(v);
  }
}

function esc(s: string): string {
  return s.replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c]!);
}
