// Viewer mode: the phone UI. Below 700 px, or on a touch-only device, the desktop
// chrome (dock, transport, preset bar, HUD, breeding, exploration, genes) is hidden
// by `#app.viewer` and this module shows its own minimal layer instead: a start card,
// a bottom bar (preset name, previous / next, like / dislike, more), a small menu and
// a thumbnail grid of presets. Nothing is built until a device qualifies, so desktop
// never gets this markup. "Advanced (desktop features)" switches back to the full UI.

import { icon, type IconName } from './icons';
import { loadSetting, saveSetting } from './storage';
import './viewer.css';

export interface ViewerPreset {
  id: string;
  name: string;
  liked: boolean;
}

export interface ViewerHost {
  current(): ViewerPreset | null;
  presets(): ViewerPreset[];
  play(id: string): void;
  next(): void;
  vote(like: boolean): void;
  thumb(id: string): Promise<string>;
  addSongs(): void;
  startMic(): void;
  stopMic(): void;
  micActive(): boolean;
  hasSongs(): boolean;
  playing(): boolean;
  togglePlay(): void;
  toggleFullscreen(): void;
  /** The viewer turned on or off: close the desktop panels and relayout. */
  onModeChange(on: boolean): void;
}

const VIEWER_MAX_W = 700;
const ADVANCED_KEY = 'viewer.advanced';
const HISTORY_MAX = 50;

/** Phone-sized window, or a device with touch and no fine pointer at all. */
export function viewerEligible(): boolean {
  const touchOnly = matchMedia('(pointer: coarse)').matches && !matchMedia('(any-pointer: fine)').matches;
  return window.innerWidth < VIEWER_MAX_W || touchOnly;
}

export class Viewer {
  private app = document.getElementById('app')!;
  private advanced = loadSetting<boolean>(ADVANCED_KEY, false);
  private on = false;
  private built = false;
  private history: string[] = [];
  private favOnly = false;
  private thumbObs: IntersectionObserver | null = null;

  private start!: HTMLElement;
  private bar!: HTMLElement;
  private nameEl!: HTMLElement;
  private likeBtn!: HTMLButtonElement;
  private dislikeBtn!: HTMLButtonElement;
  private menu!: HTMLElement;
  private sheet!: HTMLElement;
  private grid!: HTMLElement;
  private favBtn!: HTMLButtonElement;
  private scrim!: HTMLElement;

  constructor(private host: ViewerHost) {
    const cur = host.current();
    if (cur) this.history.push(cur.id);
    window.addEventListener('resize', () => this.sync());
    matchMedia('(any-pointer: fine)').addEventListener?.('change', () => this.sync());
    this.sync();
  }

  get active(): boolean {
    return this.on;
  }

  /** Something of ours is open (keeps the controls from auto-hiding). */
  get isOpen(): boolean {
    return this.on && (!this.menu.hidden || !this.sheet.hidden);
  }

  /** Offered in the desktop "More" menu on devices that would get the viewer. */
  get canReturn(): boolean {
    return this.advanced && viewerEligible();
  }

  setAdvanced(advanced: boolean): void {
    this.advanced = advanced;
    saveSetting(ADVANCED_KEY, advanced);
    this.sync();
  }

  private sync(): void {
    const on = !this.advanced && viewerEligible();
    if (on === this.on) return;
    this.on = on;
    if (on && !this.built) this.build();
    this.app.classList.toggle('viewer', on);
    if (!on && this.built) this.closeAll();
    this.host.onModeChange(on);
    if (on) this.update();
  }

  /** A preset started showing (remembered for Previous). */
  noteShown(id: string): void {
    if (this.history[this.history.length - 1] === id) return;
    this.history.push(id);
    if (this.history.length > HISTORY_MAX) this.history.shift();
    if (this.on) this.update();
  }

  /** Refreshes the bar, the start card and the open menu or grid. */
  update(): void {
    if (!this.on) return;
    const m = this.host.current();
    this.nameEl.textContent = m?.name ?? '';
    this.likeBtn.classList.toggle('on', !!m?.liked);
    const empty = !this.host.hasSongs() && !this.host.micActive();
    this.start.hidden = !empty;
    this.bar.hidden = empty;
    if (!this.menu.hidden) this.renderMenu();
    if (!this.sheet.hidden) this.renderGrid();
  }

  voted(like: boolean): void {
    const b = like ? this.likeBtn : this.dislikeBtn;
    b.classList.remove('pop');
    void b.offsetWidth;
    b.classList.add('pop');
    this.update();
  }

  private prev(): void {
    // The last entry is the preset on screen; step back to the one before it.
    if (this.history.length < 2) return;
    this.history.pop();
    const id = this.history.pop()!;
    this.host.play(id);
  }

  // ------------------------------------------------------------- markup

  private build(): void {
    this.built = true;
    const btn = (cls: string, ic: IconName, label: string, size = 22) =>
      `<button class="vw-ib ${cls}" aria-label="${label}" title="${label}">${icon(ic, size)}</button>`;

    this.start = el('div', 'vw-start', `
      <div class="vw-logo">${icon('radio', 28)}</div>
      <h1>MusicVis</h1>
      <p>Visuals that move with your music, following every beat, build and drop.</p>
      <button class="vw-big primary vw-add">${icon('plus', 22)}<span>Add songs</span></button>
      <button class="vw-big vw-mic">${icon('mic', 22)}<span>Use microphone</span></button>`);
    this.start.hidden = true;

    this.bar = el('div', 'vw-bar glass strong chrome', `
      <b class="vw-name ell"></b>
      <div class="vw-ctl">
        ${btn('vw-prev', 'cleft', 'Previous preset')}
        ${btn('vw-dislike', 'down', 'Dislike')}
        ${btn('vw-like', 'up', 'Like')}
        ${btn('vw-next', 'cright', 'Next preset')}
        ${btn('vw-more', 'more', 'Menu')}
      </div>`);
    this.bar.setAttribute('aria-label', 'Preset controls');
    this.bar.hidden = true;
    this.nameEl = this.bar.querySelector('.vw-name')!;
    this.likeBtn = this.bar.querySelector('.vw-like')!;
    this.dislikeBtn = this.bar.querySelector('.vw-dislike')!;

    this.scrim = el('div', 'vw-scrim', '');
    this.scrim.hidden = true;

    this.menu = el('div', 'vw-menu glass strong', '');
    this.menu.setAttribute('role', 'menu');
    this.menu.setAttribute('aria-label', 'Menu');
    this.menu.hidden = true;

    this.sheet = el('div', 'vw-sheet', `
      <div class="vw-sheet-h">
        <h2>Presets</h2>
        <button class="vw-chip vw-fav" aria-pressed="false">${icon('up', 16)}<span>Favourites</span></button>
        ${btn('vw-close', 'x', 'Close')}
      </div>
      <div class="vw-grid" role="list" aria-label="Presets"></div>`);
    this.sheet.setAttribute('role', 'dialog');
    this.sheet.setAttribute('aria-label', 'Presets');
    this.sheet.hidden = true;
    this.grid = this.sheet.querySelector('.vw-grid')!;
    this.favBtn = this.sheet.querySelector('.vw-fav')!;

    this.app.append(this.start, this.bar, this.scrim, this.menu, this.sheet);

    const on = (root: HTMLElement, sel: string, fn: () => void) => root.querySelector(sel)!.addEventListener('click', fn);
    on(this.start, '.vw-add', () => this.host.addSongs());
    on(this.start, '.vw-mic', () => this.host.startMic());
    on(this.bar, '.vw-prev', () => this.prev());
    on(this.bar, '.vw-next', () => this.host.next());
    on(this.bar, '.vw-like', () => this.host.vote(true));
    on(this.bar, '.vw-dislike', () => this.host.vote(false));
    on(this.bar, '.vw-more', () => (this.menu.hidden ? this.openMenu() : this.closeAll()));
    on(this.sheet, '.vw-close', () => this.closeAll());
    this.favBtn.addEventListener('click', () => {
      this.favOnly = !this.favOnly;
      this.favBtn.setAttribute('aria-pressed', String(this.favOnly));
      this.renderGrid();
    });
    this.scrim.addEventListener('click', () => this.closeAll());
    this.grid.addEventListener('click', (ev) => {
      const tile = (ev.target as Element).closest<HTMLElement>('.vw-tile');
      if (!tile?.dataset.id) return;
      this.host.play(tile.dataset.id);
      this.closeAll();
    });
    document.addEventListener('keydown', (ev) => {
      if (ev.key === 'Escape' && this.isOpen) this.closeAll();
    });
    this.thumbObs = new IntersectionObserver((entries) => {
      for (const e of entries) {
        if (!e.isIntersecting) continue;
        const img = e.target as HTMLImageElement;
        this.thumbObs!.unobserve(img);
        const id = img.dataset.id!;
        void this.host.thumb(id).then((url) => {
          if (url && img.dataset.id === id) img.src = url;
        });
      }
    }, { root: this.grid, rootMargin: '200px' });
  }

  private openMenu(): void {
    this.sheet.hidden = true;
    this.renderMenu();
    this.menu.hidden = false;
    this.scrim.hidden = false;
  }

  private openPresets(): void {
    this.menu.hidden = true;
    this.renderGrid();
    this.sheet.hidden = false;
    this.scrim.hidden = true;
    this.grid.querySelector<HTMLElement>('.vw-tile.cur')?.scrollIntoView({ block: 'center' });
  }

  private closeAll(): void {
    this.menu.hidden = true;
    this.sheet.hidden = true;
    this.scrim.hidden = true;
  }

  private renderMenu(): void {
    const mic = this.host.micActive();
    const items: { ic: IconName; label: string; run: () => void }[] = [
      { ic: 'plus', label: 'Add songs', run: () => this.host.addSongs() },
      mic ? { ic: 'stop', label: 'Stop microphone', run: () => this.host.stopMic() } : { ic: 'mic', label: 'Microphone', run: () => this.host.startMic() },
    ];
    if (this.host.hasSongs() && !mic) {
      const playing = this.host.playing();
      items.push({ ic: playing ? 'pause' : 'play', label: playing ? 'Pause music' : 'Play music', run: () => this.host.togglePlay() });
    }
    const rest: typeof items = [
      { ic: 'grid', label: 'Presets', run: () => this.openPresets() },
      { ic: 'fullscreen', label: document.fullscreenElement ? 'Exit fullscreen' : 'Fullscreen', run: () => this.host.toggleFullscreen() },
    ];
    const row = (it: (typeof items)[number], i: number, group: string) =>
      `<button class="vw-mi" role="menuitem" data-g="${group}" data-i="${i}">${icon(it.ic, 20)}<span>${it.label}</span></button>`;
    this.menu.innerHTML = `
      <div class="vw-mh">Music source</div>
      ${items.map((it, i) => row(it, i, 'a')).join('')}
      <div class="vw-sep"></div>
      ${rest.map((it, i) => row(it, i, 'b')).join('')}
      <button class="vw-adv">Advanced (desktop features)</button>`;
    this.menu.querySelectorAll<HTMLElement>('.vw-mi').forEach((b) => {
      b.addEventListener('click', () => {
        const it = (b.dataset.g === 'a' ? items : rest)[Number(b.dataset.i)];
        if (it.label !== 'Presets') this.closeAll();
        it.run();
      });
    });
    this.menu.querySelector('.vw-adv')!.addEventListener('click', () => this.setAdvanced(true));
  }

  private renderGrid(): void {
    const cur = this.host.current()?.id;
    const list = this.host.presets().filter((p) => !this.favOnly || p.liked);
    this.thumbObs?.disconnect();
    this.grid.innerHTML = '';
    if (!list.length) {
      this.grid.innerHTML = `<p class="vw-none">${this.favOnly ? 'Like a preset and it shows up here.' : 'No presets yet.'}</p>`;
      return;
    }
    const frag = document.createDocumentFragment();
    for (const p of list) {
      const tile = document.createElement('button');
      tile.className = 'vw-tile' + (p.id === cur ? ' cur' : '');
      tile.dataset.id = p.id;
      tile.setAttribute('role', 'listitem');
      const img = document.createElement('img');
      img.alt = '';
      img.dataset.id = p.id;
      const name = document.createElement('span');
      name.className = 'ell';
      name.textContent = p.name;
      tile.append(img, name);
      frag.append(tile);
    }
    this.grid.append(frag);
    this.grid.querySelectorAll('img').forEach((img) => this.thumbObs?.observe(img));
  }
}

function el(tag: string, cls: string, html: string): HTMLElement {
  const e = document.createElement(tag);
  e.className = cls;
  e.innerHTML = html;
  return e;
}
