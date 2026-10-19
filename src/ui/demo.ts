// Demo mode: the simplified first-run UI. Its one idea is "watch visuals, keep the
// ones you like, breed any of them into new children". While `#app.demo` is set the
// desktop chrome (dock, preset bar, transport, HUD, menus) is hidden by demo.css and
// this module shows its own layer instead: a welcome card, a watch layer (current
// visual, a glass bar with song / keepers / Skip, Keep, Breed), and one sheet that
// shows the breed picker, the children of a breed, or the family tree. Nothing is
// built until demo turns on, and the module talks to the app only through DemoHost.

import { icon, type IconName } from './icons';
import { loadSetting, saveSetting } from './storage';
import './demo.css';

export type DemoTag = 'fused' | 'morph' | 'merged' | 'layered' | 'edited';

export interface DemoVisual {
  id: string;
  name: string;
  gen: number;
  parents: string[];
  /** The user liked it. */
  kept: boolean;
  /** A gen 0 starter. */
  seed: boolean;
  /** How a crossover child combined its parents. */
  tag?: DemoTag;
  /** ms epoch */
  created: number;
}

export interface DemoSong {
  title: string;
  time: number;
  duration: number;
  playing: boolean;
}
export interface DemoTrack {
  id: string;
  title: string;
  current: boolean;
  status: 'queued' | 'analyzing' | 'ready' | 'error';
  /** 0..1 while analyzing. */
  progress: number;
}

export interface DemoHost {
  current(): DemoVisual | null;
  /** Every visible visual (seeds and children). */
  visuals(): DemoVisual[];
  get(id: string): DemoVisual | null;
  play(id: string): void;
  /** Dislike the current visual and show another. */
  skip(): void;
  /** Keep a visual, or unkeep it if it is kept (the current one when id is omitted). */
  keep(id?: string): void;
  /** data/blob URL ('' if none) */
  thumb(id: string): Promise<string>;
  /** Breeds 4 screened children; onChild fires as each arrives (they take a few seconds each). amount 0..1 = how different. */
  breed(parents: string[], mode: 'cross' | 'mutate', amount: number, onChild: (v: DemoVisual) => void): Promise<DemoVisual[]>;
  breeding(): boolean;
  /** Starts the bundled demo song (resolves when it is loading/playing). */
  playDemoSong(): Promise<void>;
  /** Opens the file picker; the first picked song plays straight away. */
  addSongs(): void;
  startMic(): void;
  /** Shares another browser tab's audio (Chrome / Edge). */
  startTab(): void;
  tabSupported(): boolean;
  stopMic(): void;
  micActive(): boolean;
  /** Music is loaded or playing, or the mic is on (the welcome card is not needed). */
  started(): boolean;
  song(): DemoSong | null;
  togglePlay(): void;
  /** The playlist, in play order. */
  tracks(): DemoTrack[];
  playTrack(id: string): void;
  /** The demo turned on or off: main closes panels / relayouts. */
  onModeChange(on: boolean): void;
}

type Mode = 'cross' | 'mutate';
type View = 'breed' | 'children' | 'family' | 'music';
type Filter = 'all' | 'kept' | 'seed' | 'family';

/** One breed run shown in the children view. */
interface Brood {
  parents: string[];
  mode: Mode;
  amount: number;
  items: DemoVisual[];
  /** How many cells to show (4 while running, the real count once it resolves). */
  expected: number;
  running: boolean;
  failed: boolean;
}

const COACH_KEY = 'demo.coach';
const AMOUNT_KEY = 'demo.amount';
const BROOD = 4;
const COACH_DELAY = 12000;
const TRAY_MAX = 4;
const PILL_MAX = 3;

const TAG: Partial<Record<DemoTag, { label: string; color: string }>> = {
  fused: { label: 'Fused', color: '#B79CFF' },
  morph: { label: 'Morphed', color: 'var(--acc)' },
  merged: { label: 'Merged', color: '#FF9FD2' },
  layered: { label: 'Layered', color: 'var(--warn)' },
};

export class Demo {
  private app = document.getElementById('app')!;
  private on = false;
  private built = false;
  private tickTimer = 0;
  private coachTimer = 0;
  private coachDone = loadSetting<string>(COACH_KEY, '') === 'done';
  private thumbs = new Map<string, string>();
  private thumbObs: IntersectionObserver | null = null;
  private lastFocus: HTMLElement | null = null;
  private wasStarted: boolean | null = null;
  private trayKey = '';
  private songKey = '';

  // sheet state
  private view: View | null = null;
  private mode: Mode = 'cross';
  private slots: [string | null, string | null] = [null, null];
  private activeSlot: 0 | 1 = 0;
  private filter: Filter = 'all';
  private amount = clamp01(loadSetting<number>(AMOUNT_KEY, 0.5));
  private pickerKey = '';
  private familyKey = '';
  private familySel: string[] = [];
  private brood: Brood | null = null;
  private broodRun = 0;
  private edgeFrame = 0;

  // markup
  private welcome!: HTMLElement;
  private top!: HTMLElement;
  private bottom!: HTMLElement;
  private info!: HTMLElement;
  private bar!: HTMLElement;
  private prow!: HTMLElement;
  private coach!: HTMLElement;
  private scrim!: HTMLElement;
  private sheet!: HTMLElement;

  /** Demo is the app at the root URL; `?full` (or the older `?demo=0`) opens Full mode. */
  constructor(private host: DemoHost) {
    const q = new URLSearchParams(location.search);
    this.apply(!(q.has('full') || q.get('demo') === '0'));
  }

  get active(): boolean {
    return this.on;
  }

  /** One of our sheets is open (keeps controls from auto-hiding). */
  get isOpen(): boolean {
    return this.on && this.view !== null;
  }

  /** Turn demo on/off; the URL follows (`?full`), so a reload or a shared link keeps the mode. */
  setOn(on: boolean): void {
    const url = new URL(location.href);
    url.searchParams.delete('demo');
    if (on) url.searchParams.delete('full');
    else url.searchParams.set('full', '');
    history.replaceState(history.state, '', url.toString().replace(/([?&]full)=(?=&|$)/, '$1'));
    this.apply(on);
  }

  /** A visual started showing. */
  noteShown(_id: string): void {
    this.update();
  }

  /** Re-renders the bar, the tray and the open sheet from host state. */
  update(): void {
    if (!this.on) return;
    const started = this.host.started();
    this.wasStarted = started;
    this.welcome.hidden = started;
    this.top.hidden = !started;
    this.bottom.hidden = !started;
    if (!started) {
      this.coach.hidden = true;
    } else {
      this.scheduleCoach();
    }
    this.renderInfo();
    this.renderSong();
    this.renderActions();
    this.renderTray();
    if (this.view === 'breed') this.renderBreed();
    else if (this.view === 'children') this.renderChildren();
    else if (this.view === 'family') this.renderFamily();
    else if (this.view === 'music') this.renderMusic();
  }

  // ------------------------------------------------------------- mode

  private apply(on: boolean): void {
    if (on === this.on) return;
    this.on = on;
    if (on && !this.built) this.build();
    this.app.classList.toggle('demo', on);
    if (on) {
      this.tickTimer = window.setInterval(() => this.tick(), 500);
    } else {
      window.clearInterval(this.tickTimer);
      window.clearTimeout(this.coachTimer);
      this.coachTimer = 0;
      if (this.built) {
        this.closeSheet(false);
        this.coach.hidden = true;
      }
    }
    this.host.onModeChange(on);
    if (on) this.update();
  }

  /** Twice a second: song time and play state, plus the welcome card going away. */
  private tick(): void {
    if (!this.on) return;
    if (this.host.started() !== this.wasStarted) {
      this.update();
      return;
    }
    this.renderSong();
  }

  // ------------------------------------------------------------- markup

  private build(): void {
    this.built = true;
    const mark = `<div class="dm-mark">${icon('radio', 16)}<b>MusicVis</b><span class="dm-pill-demo">Demo</span></div>`;

    this.welcome = el('div', 'dm-welcome', `
      <div class="dm-wcard glass strong" role="dialog" aria-labelledby="dm-w-title">
        ${mark}
        <h1 id="dm-w-title">Grow your own music visualizers</h1>
        <p class="dm-wbody">Every visual here has DNA. Keep the ones you love, breed any two together, and their children inherit a bit of each. A few rounds in, you have visuals nobody else has seen.</p>
        <div class="dm-steps">
          ${step('eye', 'Watch', "Visuals react to the song. Skip any you don't like.")}
          ${step('heart', 'Keep', 'Tap Keep on the ones you love.')}
          ${step('breed', 'Breed', 'Mix any two visuals into four new children.')}
        </div>
        <div class="dm-wbtns">
          <button class="btn primary dm-xl dm-demo-song">${icon('play', 18)}<span>Play a demo song</span></button>
          <div class="dm-wsrc">
            <button class="btn dm-src dm-upload">${icon('upload', 17)}<span>Upload my songs</span></button>
            <button class="btn dm-src dm-mic">${icon('mic', 17)}<span>Use my microphone</span></button>
            ${this.host.tabSupported() ? `<button class="btn dm-src dm-tab" title="Play music in another tab (Spotify, YouTube…) and share its audio">${icon('radio', 17)}<span>Listen to a tab</span></button>` : ''}
          </div>
        </div>
        <div class="dm-wfoot">
          <span>You can also drop songs anywhere.</span>
          <button class="btn ghost sm dm-full">Been here before? Full mode</button>
        </div>
      </div>`);

    this.top = el('div', 'dm-top chrome', `
      ${mark}
      <button class="btn ghost dm-full" aria-label="Switch to full mode">Full mode</button>`);

    this.bottom = el('div', 'dm-bottom', '');
    this.info = el('div', 'dm-info chrome', `
      <div class="dm-iname"><b class="dm-name ell"></b><span class="dm-gen"></span></div>
      <div class="dm-line ell"></div>`);
    this.info.setAttribute('aria-live', 'polite');

    // phone only: play button, song title and the keepers pill above the bar
    this.prow = el('div', 'dm-prow chrome', `
      <button class="dm-pp dm-round sm" aria-label="Play">${icon('play', 16)}</button>
      <button class="dm-ptitle dm-musicb ell" aria-label="Music: playlist, microphone or a tab"></button>
      <button class="dm-add dm-round sm" aria-label="Play my own songs">${icon('upload', 15)}</button>
      <button class="dm-kpill" aria-label="Open your family"><span class="dm-kcount"></span><span class="dm-kthumbs"></span></button>`);

    this.bar = el('div', 'dm-bar glass strong chrome', `
      <div class="dm-song">
        <button class="dm-pp dm-round" aria-label="Play">${icon('play', 18)}</button>
        <button class="dm-stext dm-musicb" aria-label="Music: playlist, microphone or a tab" title="Playlist, microphone or a tab">
          <b class="dm-stitle ell"></b>
          <span class="dm-stime mono"></span>
        </button>
        <button class="dm-add dm-round sm" aria-label="Play my own songs" title="Play my own songs">${icon('upload', 15)}</button>
      </div>
      <div class="dm-div"></div>
      <div class="dm-tray">
        <span class="dm-tlabel">Keepers</span>
        <div class="dm-tthumbs"></div>
        <button class="btn ghost sm dm-famb">Family</button>
      </div>
      <div class="dm-div"></div>
      <div class="dm-act">
        <button class="btn lg dm-skip" aria-label="Skip this visual">${icon('x', 16)}<span>Skip</span></button>
        <button class="btn lg dm-keep" aria-label="Keep this visual">${icon('heart', 16)}<span>Keep</span></button>
        <button class="btn lg primary dm-breed" aria-label="Breed new visuals">${icon('breed', 16)}<span>Breed</span></button>
        <div class="dm-coach" role="status" hidden>
          <div><b>Love this one? Keep it.</b><span>Breed mixes any two visuals into four new ones.</span></div>
          <button class="dm-coach-x" aria-label="Dismiss tip">${icon('x', 14)}</button>
        </div>
      </div>`);
    this.bar.setAttribute('role', 'toolbar');
    this.bar.setAttribute('aria-label', 'Visual controls');
    this.coach = this.bar.querySelector('.dm-coach')!;
    this.bottom.append(this.info, this.prow, this.bar);

    this.scrim = el('div', 'dm-scrim', '');
    this.scrim.hidden = true;
    this.sheet = el('div', 'dm-sheet glass strong', '');
    this.sheet.setAttribute('role', 'dialog');
    this.sheet.setAttribute('aria-modal', 'true');
    this.sheet.hidden = true;

    this.welcome.hidden = true;
    this.top.hidden = true;
    this.bottom.hidden = true;
    this.app.append(this.welcome, this.top, this.bottom, this.scrim, this.sheet);

    const demoBtn = this.welcome.querySelector<HTMLButtonElement>('.dm-demo-song')!;
    demoBtn.addEventListener('click', () => {
      demoBtn.disabled = true;
      this.host.playDemoSong().then(
        () => {
          demoBtn.disabled = false;
          this.update();
        },
        () => {
          demoBtn.disabled = false;
        },
      );
    });
    this.welcome.querySelector('.dm-mic')!.addEventListener('click', () => {
      this.host.startMic();
      this.update();
    });
    this.welcome.querySelector('.dm-tab')?.addEventListener('click', () => {
      this.host.startTab();
      this.update();
    });
    this.app.querySelectorAll('.dm-upload, .dm-add').forEach((b) => b.addEventListener('click', () => this.host.addSongs()));
    this.app.querySelectorAll('.dm-full').forEach((b) => b.addEventListener('click', () => this.setOn(false)));

    this.bottom.querySelectorAll('.dm-pp').forEach((b) =>
      b.addEventListener('click', () => {
        if (!this.host.song() && this.host.micActive()) this.host.stopMic();
        else this.host.togglePlay();
        this.renderSong();
      }),
    );
    this.bar.querySelector('.dm-skip')!.addEventListener('click', () => this.host.skip());
    this.bar.querySelector('.dm-keep')!.addEventListener('click', () => {
      this.dismissCoach();
      this.host.keep();
      this.update();
    });
    this.bar.querySelector('.dm-breed')!.addEventListener('click', (ev) => {
      this.dismissCoach();
      this.lastFocus = ev.currentTarget as HTMLElement;
      if (this.brood?.running) this.openView('children');
      else this.openBreed([this.host.current()?.id ?? null, null], 'cross');
    });
    this.bar.querySelector('.dm-coach-x')!.addEventListener('click', () => this.dismissCoach());
    const openFamily = (ev: Event) => {
      this.lastFocus = ev.currentTarget as HTMLElement;
      this.familySel = [];
      this.openView('family');
    };
    this.bar.querySelector('.dm-famb')!.addEventListener('click', openFamily);
    this.app.querySelectorAll('.dm-musicb').forEach((b) =>
      b.addEventListener('click', (ev) => {
        this.lastFocus = ev.currentTarget as HTMLElement;
        this.openView('music');
      }),
    );
    this.prow.querySelector('.dm-kpill')!.addEventListener('click', openFamily);
    this.bar.querySelector('.dm-tthumbs')!.addEventListener('click', (ev) => {
      const b = (ev.target as Element).closest<HTMLElement>('[data-id]');
      if (b?.dataset.id) this.host.play(b.dataset.id);
    });

    this.scrim.addEventListener('click', () => this.closeSheet());
    this.sheet.addEventListener('click', (ev) => this.onSheetClick(ev));
    this.sheet.addEventListener('input', (ev) => {
      const t = ev.target as HTMLInputElement;
      if (!t.classList.contains('dm-amount')) return;
      this.amount = clamp01(Number(t.value));
      saveSetting(AMOUNT_KEY, this.amount);
    });
    document.addEventListener('keydown', (ev) => {
      if (ev.key === 'Escape' && this.isOpen) {
        ev.preventDefault();
        ev.stopPropagation();
        this.closeSheet();
      }
    });
    window.addEventListener('resize', () => {
      if (this.view === 'family') this.queueEdges();
    });
    this.thumbObs = new IntersectionObserver(
      (entries) => {
        for (const e of entries) {
          if (!e.isIntersecting) continue;
          const img = e.target as HTMLImageElement;
          this.thumbObs!.unobserve(img);
          this.loadThumb(img);
        }
      },
      { rootMargin: '200px' },
    );
  }

  // ------------------------------------------------------------- watch layer

  private renderInfo(): void {
    const v = this.host.current();
    const name = this.info.querySelector<HTMLElement>('.dm-name')!;
    const gen = this.info.querySelector<HTMLElement>('.dm-gen')!;
    const line = this.info.querySelector<HTMLElement>('.dm-line')!;
    this.info.hidden = !v;
    if (!v) return;
    name.textContent = v.name;
    gen.textContent = `Gen ${v.gen}`;
    line.textContent = this.lineage(v);
  }

  private lineage(v: DemoVisual): string {
    if (v.seed || !v.parents.length) return 'Starter visual';
    const names = v.parents.map((p) => this.host.get(p)?.name ?? 'a lost visual');
    return names.length === 1 ? `Mutant of ${names[0]}` : `Child of ${names.join(' × ')}`;
  }

  private renderSong(): void {
    if (!this.built) return;
    const s = this.host.song();
    const mic = !s && this.host.micActive();
    const playing = s ? s.playing : mic;
    const title = s ? s.title : mic ? 'Listening to your microphone' : 'No song';
    const time = s ? `${fmtTime(s.time)} / ${fmtTime(s.duration)}` : mic ? 'Live' : '';
    const ic: IconName = mic ? 'stop' : playing ? 'pause' : 'play';
    const key = `${ic}|${title}|${time}`;
    if (key === this.songKey) return;
    this.songKey = key;
    const label = mic ? 'Stop microphone' : playing ? 'Pause' : 'Play';
    this.bottom.querySelectorAll<HTMLButtonElement>('.dm-pp').forEach((b) => {
      const size = b.classList.contains('sm') ? 16 : 18;
      if (b.dataset.ic !== ic) {
        b.dataset.ic = ic;
        b.innerHTML = icon(ic, size);
      }
      b.setAttribute('aria-label', label);
      b.title = label;
      b.disabled = !s && !mic;
    });
    this.bar.querySelector('.dm-stitle')!.textContent = title;
    this.bar.querySelector('.dm-stime')!.textContent = time;
    this.prow.querySelector('.dm-ptitle')!.textContent = title;
  }

  private renderActions(): void {
    const v = this.host.current();
    const keep = this.bar.querySelector<HTMLButtonElement>('.dm-keep')!;
    const kept = !!v?.kept;
    if (keep.dataset.kept !== String(kept)) {
      keep.dataset.kept = String(kept);
      keep.classList.toggle('on', kept);
      keep.innerHTML = kept ? `${icon('check', 16)}<span>Kept</span>` : `${icon('heart', 16)}<span>Keep</span>`;
      keep.setAttribute('aria-label', kept ? 'Kept: tap to unkeep' : 'Keep this visual');
      keep.title = kept ? 'Tap to unkeep' : '';
      keep.setAttribute('aria-pressed', String(kept));
    }
    keep.disabled = !v;
    this.bar.querySelector<HTMLButtonElement>('.dm-skip')!.disabled = !v;
  }

  private keepers(): DemoVisual[] {
    return this.host
      .visuals()
      .filter((v) => v.kept)
      .sort((a, b) => b.created - a.created);
  }

  private renderTray(): void {
    const kept = this.keepers();
    const key = kept.map((v) => v.id).join(',');
    if (key === this.trayKey) return;
    this.trayKey = key;
    const tray = this.bar.querySelector<HTMLElement>('.dm-tthumbs')!;
    tray.innerHTML = '';
    if (!kept.length) {
      tray.innerHTML = '<span class="dm-tnone">None yet</span>';
    } else {
      for (const v of kept.slice(0, TRAY_MAX)) {
        const b = document.createElement('button');
        b.className = 'dm-tthumb';
        b.dataset.id = v.id;
        b.setAttribute('aria-label', `Play ${v.name}`);
        b.title = v.name;
        b.append(this.img(v.id, false));
        tray.append(b);
      }
    }
    const n = kept.length;
    this.prow.querySelector('.dm-kcount')!.textContent = n ? `${n} keeper${n === 1 ? '' : 's'}` : 'Family';
    const pt = this.prow.querySelector<HTMLElement>('.dm-kthumbs')!;
    pt.innerHTML = '';
    for (const v of kept.slice(0, PILL_MAX)) pt.append(this.img(v.id, false));
  }

  private scheduleCoach(): void {
    if (this.coachDone || this.coachTimer) return;
    this.coachTimer = window.setTimeout(() => {
      if (!this.on || this.coachDone || !this.host.started()) {
        this.coachTimer = 0;
        return;
      }
      this.coach.hidden = false;
    }, COACH_DELAY);
  }

  private dismissCoach(): void {
    window.clearTimeout(this.coachTimer);
    this.coach.hidden = true;
    if (this.coachDone) return;
    this.coachDone = true;
    saveSetting(COACH_KEY, 'done');
  }

  // ------------------------------------------------------------- sheet

  private openBreed(slots: (string | null)[], mode: Mode): void {
    this.mode = mode;
    const a = slots[0] ?? null;
    const b = mode === 'cross' && slots[1] !== a ? slots[1] ?? null : null;
    this.slots = [a, b];
    // the slot to fill next: B once A is set (it replaces B when both are set)
    this.activeSlot = mode === 'mutate' || !a ? 0 : 1;
    this.openView('breed');
  }

  private openView(view: View): void {
    if (this.view === null) this.lastFocus ??= document.activeElement as HTMLElement | null;
    this.view = view;
    this.pickerKey = '';
    this.familyKey = '';
    this.coach.hidden = true;
    this.sheet.className = `dm-sheet glass strong v-${view}`;
    this.scrim.className = `dm-scrim${view === 'children' ? ' light' : ''}`;
    this.sheet.innerHTML = '';
    this.thumbObs?.disconnect();
    if (view === 'breed') this.buildBreed();
    else if (view === 'children') this.buildChildren();
    else if (view === 'music') this.buildMusic();
    else this.buildFamily();
    this.sheet.hidden = false;
    this.scrim.hidden = false;
    this.update();
    this.sheet.scrollTop = 0;
    const first = this.sheet.querySelector<HTMLElement>('.dm-sclose, button:not([disabled])');
    first?.focus({ preventScroll: true });
  }

  private closeSheet(restore = true): void {
    if (this.view === null) return;
    this.view = null;
    cancelAnimationFrame(this.edgeFrame);
    this.thumbObs?.disconnect();
    this.sheet.hidden = true;
    this.scrim.hidden = true;
    this.sheet.innerHTML = '';
    const f = this.lastFocus;
    this.lastFocus = null;
    if (restore && f?.isConnected) f.focus({ preventScroll: true });
  }

  private onSheetClick(ev: Event): void {
    const t = ev.target as Element;
    const act = t.closest<HTMLElement>('[data-act]');
    if (!act || !this.sheet.contains(act)) return;
    if ((act as HTMLButtonElement).disabled) return;
    const id = act.dataset.id ?? '';
    switch (act.dataset.act) {
      case 'close':
        this.closeSheet();
        break;
      case 'mode':
        this.setMode(act.dataset.mode === 'mutate' ? 'mutate' : 'cross');
        break;
      case 'slot':
        this.activeSlot = act.dataset.slot === '1' ? 1 : 0;
        this.renderBreed();
        break;
      case 'filter':
        this.filter = (act.dataset.filter as Filter) ?? 'all';
        this.renderBreed();
        break;
      case 'pick':
        this.pick(id);
        break;
      case 'go':
        this.startBreed(this.mode === 'mutate' ? [this.slots[0]!] : [this.slots[0]!, this.slots[1]!], this.mode);
        break;
      case 'back':
        this.openBreed(this.brood?.parents ?? this.slots, this.brood?.mode ?? this.mode);
        break;
      case 'again':
        if (this.brood) this.startBreed(this.brood.parents, this.brood.mode);
        break;
      case 'mutate-cur': {
        const cur = this.host.current();
        if (cur) this.openBreed([cur.id], 'mutate');
        break;
      }
      case 'play':
        this.host.play(id);
        this.update();
        break;
      case 'keep':
        this.host.keep(id);
        this.update();
        break;
      case 'select':
        this.toggleFamilySel(id);
        break;
      case 'breed-sel':
        this.openBreed(this.familySel.slice(0, 2), 'cross');
        break;
      case 'track':
        this.host.playTrack(id);
        this.closeSheet();
        break;
      case 'src-upload':
        this.closeSheet();
        this.host.addSongs();
        break;
      case 'src-mic':
        this.closeSheet();
        this.host.startMic();
        break;
      case 'src-tab':
        this.closeSheet();
        this.host.startTab();
        break;
      case 'src-stop':
        this.host.stopMic();
        this.update();
        break;
    }
  }

  private header(title: string, sub: string, extra = '', lead = ''): string {
    return `
      <div class="dm-sh">
        ${lead}
        <div class="dm-sht">
          <h2 id="dm-s-title">${esc(title)}</h2>
          <p class="dm-sub">${esc(sub)}</p>
        </div>
        ${extra}
        <button class="ib dm-sclose" data-act="close" aria-label="Close">${icon('x', 18)}</button>
      </div>`;
  }

  // ------------------------------------------------------------- breed view

  private buildBreed(): void {
    this.sheet.setAttribute('aria-labelledby', 'dm-s-title');
    const seg = `
      <div class="dm-seg" role="group" aria-label="Breeding mode">
        <button data-act="mode" data-mode="cross" aria-pressed="true">${icon('breed', 15)}<span>Breed two</span></button>
        <button data-act="mode" data-mode="mutate" aria-pressed="false">${icon('mutate', 15)}<span>Mutate one</span></button>
      </div>`;
    this.sheet.innerHTML = `
      ${this.header('Breed', '', seg)}
      <div class="dm-sb">
        <div class="dm-slots"></div>
        <div class="dm-ph">
          <div><h3>Choose parents</h3><p class="dm-hint">Tap any visual to fill the highlighted slot</p></div>
          <div class="dm-chips" role="group" aria-label="Filter visuals"></div>
        </div>
        <div class="dm-pgrid" role="list" aria-label="Visuals"></div>
      </div>
      <div class="dm-sf">
        <div class="dm-amt">
          <label for="dm-amount">How different should the children be?</label>
          <input id="dm-amount" class="dm-amount" type="range" min="0" max="1" step="0.01" value="${this.amount}">
          <div class="dm-amt-l"><span>Like the parents</span><span>Surprise me</span><span>Wild</span></div>
        </div>
        <button class="btn lg primary dm-go" data-act="go"></button>
      </div>`;
  }

  private setMode(mode: Mode): void {
    if (mode === this.mode) return;
    this.mode = mode;
    if (mode === 'mutate') this.activeSlot = 0;
    else this.activeSlot = this.slots[0] && !this.slots[1] ? 1 : 0;
    this.renderBreed();
  }

  private pick(id: string): void {
    if (!id) return;
    const other: 0 | 1 = this.activeSlot === 0 ? 1 : 0;
    if (this.slots[other] === id) {
      if (this.mode === 'cross') return;
      this.slots[other] = null;
    }
    this.slots[this.activeSlot] = id;
    if (this.mode === 'cross' && !this.slots[other]) this.activeSlot = other;
    this.renderBreed();
  }

  private renderBreed(): void {
    const all = this.host.visuals();
    // drop parents that are no longer visible
    for (const i of [0, 1] as const) {
      const s = this.slots[i];
      if (s && !this.host.get(s)) this.slots[i] = null;
    }
    const mutate = this.mode === 'mutate';
    this.sheet.querySelector('.dm-sub')!.textContent = mutate
      ? 'Pick one visual. Its children are variations on it.'
      : 'Pick any two visuals. Their children get a mix of both: shapes, motion and colour.';
    this.sheet.querySelectorAll<HTMLElement>('.dm-seg button').forEach((b) => b.setAttribute('aria-pressed', String(b.dataset.mode === this.mode)));

    // slots
    const slotHtml = (i: 0 | 1) => {
      const id = this.slots[i];
      const v = id ? this.host.get(id) : null;
      const act = this.activeSlot === i ? ' active' : '';
      const label = i === 0 ? 'Parent A' : 'Parent B';
      if (!v) {
        return `<button class="dm-slot empty${act}" data-act="slot" data-slot="${i}" aria-label="${label}: empty${act ? ', selected' : ''}">
          <span class="dm-slot-ph">${icon('plus', 20)}<span>${i === 0 && mutate ? 'Pick a visual' : i === 0 ? 'Pick a parent' : 'Pick a second parent'}</span></span></button>`;
      }
      return `<button class="dm-slot${act}" data-act="slot" data-slot="${i}" aria-label="${label}: ${esc(v.name)}${act ? ', selected' : ''}">
        <span class="dm-slot-img" data-thumb="${esc(v.id)}"></span>
        <span class="dm-slot-t"><span class="dm-slot-k">${mutate ? 'Parent' : label}</span><b class="ell">${esc(v.name)}</b><span class="dm-meta">${esc(meta(v))}</span></span></button>`;
    };
    const slots = this.sheet.querySelector<HTMLElement>('.dm-slots')!;
    const slotKey = `${this.mode}|${this.activeSlot}|${this.slots.map((s) => (s ? s + ':' + (this.host.get(s)?.kept ? 1 : 0) : '')).join(',')}`;
    if (slots.dataset.key !== slotKey) {
      slots.dataset.key = slotKey;
      slots.classList.toggle('one', mutate);
      slots.innerHTML = mutate ? slotHtml(0) : `${slotHtml(0)}<span class="dm-plus" aria-hidden="true">${icon('plus', 18)}</span>${slotHtml(1)}`;
      slots.querySelectorAll<HTMLElement>('[data-thumb]').forEach((s) => s.append(this.img(s.dataset.thumb!, false)));
    }

    // filter chips
    const counts: Record<Filter, number> = { all: all.length, kept: 0, seed: 0, family: 0 };
    for (const v of all) {
      if (v.kept) counts.kept++;
      if (v.seed) counts.seed++;
      else counts.family++;
    }
    const chipDefs: [Filter, string][] = [
      ['all', 'All'],
      ['kept', '♥ Keepers'],
      ['seed', 'Starters'],
      ['family', 'Your family'],
    ];
    const chips = this.sheet.querySelector<HTMLElement>('.dm-chips')!;
    const chipKey = `${this.filter}|${counts.all},${counts.kept},${counts.seed},${counts.family}`;
    if (chips.dataset.key !== chipKey) {
      chips.dataset.key = chipKey;
      chips.innerHTML = chipDefs
        .map(([f, l]) => `<button class="dm-chip" data-act="filter" data-filter="${f}" aria-pressed="${f === this.filter}">${l}<span>${counts[f]}</span></button>`)
        .join('');
    }

    // picker grid: rebuilt only when the list changes
    const list = all
      .filter((v) => (this.filter === 'kept' ? v.kept : this.filter === 'seed' ? v.seed : this.filter === 'family' ? !v.seed : true))
      .sort((a, b) => Number(b.kept) - Number(a.kept) || b.created - a.created);
    const key = this.filter + '|' + list.map((v) => v.id + (v.kept ? '+' : '')).join(',');
    const grid = this.sheet.querySelector<HTMLElement>('.dm-pgrid')!;
    if (key !== this.pickerKey) {
      this.pickerKey = key;
      grid.innerHTML = '';
      if (!list.length) {
        const msg: Record<Filter, string> = {
          all: 'No visuals yet.',
          kept: 'Tap Keep on a visual and it shows up here.',
          seed: 'No starter visuals.',
          family: 'Breed something and its children show up here.',
        };
        grid.innerHTML = `<p class="dm-none">${msg[this.filter]}</p>`;
      } else {
        const frag = document.createDocumentFragment();
        for (const v of list) {
          const b = document.createElement('button');
          b.className = 'dm-card';
          b.dataset.act = 'pick';
          b.dataset.id = v.id;
          b.setAttribute('role', 'listitem');
          b.innerHTML = `<span class="dm-cthumb"><span class="dm-badge">${icon('check', 12)}<span></span></span></span>
            <b class="ell"></b><span class="dm-meta">Gen ${v.gen}${v.kept ? ' · ♥ kept' : ''}</span>`;
          b.querySelector('b')!.textContent = v.name;
          b.querySelector('.dm-cthumb')!.prepend(this.img(v.id, true));
          frag.append(b);
        }
        grid.append(frag);
      }
    }
    // selection marks, in place
    grid.querySelectorAll<HTMLElement>('.dm-card').forEach((c) => {
      const i = this.slots.indexOf(c.dataset.id!);
      const sel = i === 0 || (i === 1 && !mutate);
      c.classList.toggle('sel', sel);
      c.setAttribute('aria-pressed', String(sel));
      c.querySelector('.dm-badge span')!.textContent = sel ? (mutate ? 'Parent' : i === 0 ? 'Parent A' : 'Parent B') : '';
      const name = this.host.get(c.dataset.id!)?.name ?? '';
      c.setAttribute('aria-label', sel ? `${name}, ${i === 0 ? 'parent A' : 'parent B'}` : name);
    });

    const go = this.sheet.querySelector<HTMLButtonElement>('.dm-go')!;
    const ready = !!this.slots[0] && (mutate || !!this.slots[1]);
    const busy = this.host.breeding();
    go.disabled = !ready || busy;
    const text = busy ? 'Breeding…' : mutate ? 'Mutate into 4' : 'Breed 4 children';
    go.innerHTML = `${icon(mutate ? 'mutate' : 'breed', 16)}<span>${text}</span>`;
  }

  // ------------------------------------------------------------- children view

  private startBreed(parents: string[], mode: Mode): void {
    if (this.host.breeding()) return;
    const run = ++this.broodRun;
    const brood: Brood = { parents: parents.slice(), mode, amount: this.amount, items: [], expected: BROOD, running: true, failed: false };
    this.brood = brood;
    this.openView('children');
    const live = () => run === this.broodRun;
    this.host
      .breed(brood.parents, mode, brood.amount, (v) => {
        if (!live() || brood.items.some((x) => x.id === v.id)) return;
        brood.items.push(v);
        if (this.view === 'children') this.update();
      })
      .then(
        (list) => {
          if (!live()) return;
          const seen = new Set(list.map((v) => v.id));
          brood.items = [...list, ...brood.items.filter((v) => !seen.has(v.id))].slice(0, BROOD);
          brood.expected = brood.items.length;
          brood.running = false;
          this.update();
        },
        () => {
          if (!live()) return;
          brood.running = false;
          brood.failed = true;
          brood.expected = brood.items.length;
          this.update();
        },
      );
  }

  private buildChildren(): void {
    this.sheet.setAttribute('aria-labelledby', 'dm-s-title');
    const back = `<button class="ib dm-back" data-act="back" aria-label="Back to parents">${icon('undo', 18)}</button>`;
    this.sheet.innerHTML = `
      ${this.header('4 new children', '', '<span class="dm-count mono"></span>', back)}
      <div class="dm-sb">
        <div class="dm-kgrid" role="list" aria-label="Children"></div>
        <p class="dm-note" hidden></p>
        <p class="dm-tip">${icon('sparkle', 16)}<span>Tap a child to watch it with your music. Kept children are easy to find later, and any of them can be a parent next round.</span></p>
      </div>
      <div class="dm-sf">
        <button class="btn lg dm-again" data-act="again">${icon('reset', 16)}<span>Breed again</span></button>
        <button class="btn lg dm-mutcur" data-act="mutate-cur" hidden>${icon('mutate', 16)}<span class="ell"></span></button>
        <span class="dm-grow"></span>
        <span class="dm-kept-n"></span>
        <button class="btn lg primary" data-act="close">Back to watching</button>
      </div>`;
  }

  private renderChildren(): void {
    const b = this.brood;
    if (!b) return;
    const cur = this.host.current()?.id;
    const n = b.items.length;
    const title = this.sheet.querySelector<HTMLElement>('#dm-s-title')!;
    title.textContent = b.running ? 'Breeding…' : n === 0 ? 'No new children' : `${n} new ${n === 1 ? 'child' : 'children'}`;
    this.sheet.querySelector('.dm-count')!.textContent = b.running ? `${n} of ${BROOD}` : '';

    // lineage line: "of [thumb] A × [thumb] B · Generation N"
    const sub = this.sheet.querySelector<HTMLElement>('.dm-sub')!;
    const pv = b.parents.map((p) => this.host.get(p));
    const gen = b.items[0]?.gen ?? Math.max(0, ...pv.map((v) => v?.gen ?? 0)) + 1;
    const subKey = b.parents.join(',') + '|' + gen + '|' + pv.map((v) => v?.name ?? '').join(',');
    if (sub.dataset.key !== subKey) {
      sub.dataset.key = subKey;
      sub.innerHTML = '';
      const who = (i: number) => {
        const s = document.createElement('span');
        s.className = 'dm-who';
        s.append(this.img(b.parents[i], false), document.createTextNode(pv[i]?.name ?? 'a lost visual'));
        return s;
      };
      if (b.mode === 'mutate') sub.append('variations on ', who(0));
      else sub.append('of ', who(0), ' × ', who(1));
      sub.append(` · Generation ${gen}`);
    }

    // cells: filled ones, then shimmer placeholders until the run ends
    const grid = this.sheet.querySelector<HTMLElement>('.dm-kgrid')!;
    // Kept state comes from the host: the items are snapshots taken when each child arrived.
    const cellKey = b.items.map((v) => `${v.id}:${this.host.get(v.id)?.kept ? 1 : 0}:${v.id === cur ? 1 : 0}`).join(',') + '|' + b.expected;
    if (grid.dataset.key !== cellKey) {
      grid.dataset.key = cellKey;
      grid.innerHTML = '';
      for (const v0 of b.items) {
        const v = this.host.get(v0.id) ?? v0;
        const playing = v.id === cur;
        const tag = v.tag ? TAG[v.tag] : undefined;
        const cell = document.createElement('div');
        cell.className = 'dm-kcell' + (playing ? ' playing' : '');
        cell.setAttribute('role', 'listitem');
        cell.innerHTML = `
          <button class="dm-card" data-act="play" data-id="${esc(v.id)}" aria-label="Watch ${esc(v.name)}" aria-pressed="${playing}">
            <span class="dm-cthumb">${playing ? `<span class="dm-playing">${icon('play', 10)}Playing</span>` : ''}</span>
            <b class="ell">${esc(v.name)}</b><span class="dm-meta ell">Gen ${v.gen} · <span class="mono">${esc(v.id)}</span></span>
          </button>
          <div class="dm-krow">
            ${tag ? `<span class="dm-tag" style="--tc:${tag.color}">${tag.label}</span>` : '<span></span>'}
            <button class="btn sm dm-kbtn${v.kept ? ' on' : ''}" data-act="keep" data-id="${esc(v.id)}" aria-label="${v.kept ? 'Unkeep ' + esc(v.name) : 'Keep ' + esc(v.name)}" aria-pressed="${v.kept}">
              ${icon(v.kept ? 'check' : 'heart', 14)}<span>${v.kept ? 'Kept' : 'Keep'}</span></button>
          </div>`;
        cell.querySelector('.dm-cthumb')!.prepend(this.img(v.id, false));
        grid.append(cell);
      }
      for (let i = n; i < b.expected; i++) {
        const ph = document.createElement('div');
        ph.className = 'dm-kcell wait';
        ph.setAttribute('aria-hidden', 'true');
        ph.innerHTML = '<span class="dm-shimmer"></span><span class="dm-shimmer line"></span><span class="dm-shimmer line short"></span>';
        grid.append(ph);
      }
    }

    const note = this.sheet.querySelector<HTMLElement>('.dm-note')!;
    if (b.failed) {
      note.hidden = false;
      note.textContent = n ? `Something went wrong after ${n}. Try breeding again.` : 'Something went wrong. Try breeding again.';
    } else if (!b.running && n < BROOD) {
      note.hidden = false;
      note.textContent = n === 0 ? 'None came out different enough this time.' : `Only ${n} came out different enough this time.`;
    } else {
      note.hidden = true;
    }

    const busy = this.host.breeding() || b.running;
    this.sheet.querySelector<HTMLButtonElement>('.dm-again')!.disabled = busy;
    const playingChild = b.items.find((v) => v.id === cur);
    const mut = this.sheet.querySelector<HTMLButtonElement>('.dm-mutcur')!;
    mut.hidden = !playingChild;
    mut.disabled = busy;
    if (playingChild) {
      const name = this.host.get(playingChild.id)?.name ?? playingChild.name;
      mut.querySelector('span')!.textContent = `Mutate ${name}`;
    }
    const kept = b.items.filter((v) => (this.host.get(v.id) ?? v).kept).length;
    this.sheet.querySelector('.dm-kept-n')!.textContent = `${kept} kept`;
  }

  // ------------------------------------------------------------- family view

  private buildFamily(): void {
    this.sheet.setAttribute('aria-labelledby', 'dm-s-title');
    const extra = `
      <div class="dm-legend" aria-hidden="true"><span class="bred">Bred</span><span class="mut">Mutated</span></div>
      <button class="btn lg primary dm-bsel" data-act="breed-sel">${icon('breed', 16)}<span>Breed selected</span></button>`;
    this.sheet.innerHTML = `
      ${this.header('Your family', '', extra)}
      <div class="dm-sb dm-fscroll">
        <div class="dm-tree"></div>
      </div>`;
  }

  // ------------------------------------------------------------- music view

  private buildMusic(): void {
    this.sheet.setAttribute('aria-labelledby', 'dm-s-title');
    const tab = this.host.tabSupported()
      ? `<button class="btn dm-src" data-act="src-tab" title="Play music in another tab (Spotify, YouTube…) and share its audio">${icon('radio', 17)}<span>Listen to a tab</span></button>`
      : '';
    this.sheet.innerHTML = `
      ${this.header('Music', '')}
      <div class="dm-sb">
        <div class="dm-live" hidden>
          <span class="dm-live-dot"></span><span class="dm-live-t ell"></span>
          <button class="btn sm" data-act="src-stop">${icon('stop', 14)}<span>Stop</span></button>
        </div>
        <div class="dm-tracks" role="list" aria-label="Playlist"></div>
        <div class="dm-wsrc">
          <button class="btn dm-src" data-act="src-upload">${icon('upload', 17)}<span>Upload my songs</span></button>
          <button class="btn dm-src" data-act="src-mic">${icon('mic', 17)}<span>Use my microphone</span></button>
          ${tab}
        </div>
      </div>`;
  }

  private renderMusic(): void {
    const tracks = this.host.tracks();
    const live = this.host.micActive();
    const liveEl = this.sheet.querySelector<HTMLElement>('.dm-live')!;
    liveEl.hidden = !live;
    if (live) liveEl.querySelector('.dm-live-t')!.textContent = this.host.song()?.title ?? 'Live input';
    this.sheet.querySelector('.dm-sub')!.textContent = live
      ? 'Live input is on. Pick a song to switch back.'
      : `${tracks.length} song${tracks.length === 1 ? '' : 's'} · tap one to play it`;
    const list = this.sheet.querySelector<HTMLElement>('.dm-tracks')!;
    const key = tracks.map((t) => `${t.id}:${t.current && !live ? 1 : 0}:${t.status}:${Math.round(t.progress * 20)}`).join(',');
    if (list.dataset.key === key) return;
    list.dataset.key = key;
    list.innerHTML = tracks.length ? '' : '<p class="dm-none"><span>No songs yet. Upload some, or use the microphone.</span></p>';
    tracks.forEach((t, i) => {
      const cur = t.current && !live;
      const state = t.status === 'analyzing' ? `Getting ready ${Math.round(t.progress * 100)}%` : t.status === 'error' ? 'Could not load' : cur ? 'Playing' : '';
      const row = document.createElement('button');
      row.className = 'dm-track' + (cur ? ' cur' : '') + (t.status === 'error' ? ' err' : '');
      row.dataset.act = 'track';
      row.dataset.id = t.id;
      row.setAttribute('role', 'listitem');
      row.setAttribute('aria-current', String(cur));
      row.innerHTML = `<span class="dm-tn mono">${cur ? icon('play', 12) : i + 1}</span><span class="dm-tt ell"></span><span class="dm-ts">${state}</span>`;
      row.querySelector('.dm-tt')!.textContent = t.title;
      list.append(row);
    });
  }

  private familyMembers(): DemoVisual[] {
    const out = new Map<string, DemoVisual>();
    const walk = (v: DemoVisual) => {
      if (out.has(v.id)) return;
      out.set(v.id, v);
      for (const p of v.parents) {
        const pv = this.host.get(p);
        if (pv) walk(pv);
      }
    };
    for (const v of this.host.visuals()) if (!v.seed) walk(v);
    return [...out.values()];
  }

  private renderFamily(): void {
    const members = this.familyMembers();
    const ids = new Set(members.map((v) => v.id));
    this.familySel = this.familySel.filter((id) => ids.has(id));
    const tree = this.sheet.querySelector<HTMLElement>('.dm-tree')!;
    const key = members.map((v) => `${v.id}:${v.kept ? 1 : 0}:${v.parents.join('/')}`).join(',');
    const gens = [...new Set(members.map((v) => v.gen))].sort((a, b) => a - b);
    this.sheet.querySelector('.dm-sub')!.textContent = members.length
      ? `${members.length} visual${members.length === 1 ? '' : 's'} across ${gens.length} generation${gens.length === 1 ? '' : 's'}`
      : '';

    if (key !== this.familyKey) {
      this.familyKey = key;
      tree.innerHTML = '';
      if (!members.length) {
        tree.innerHTML = `<p class="dm-none">${icon('breed', 28)}<span>Breed something and its family tree grows here.</span></p>`;
      } else {
        const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
        svg.classList.add('dm-edges');
        svg.setAttribute('aria-hidden', 'true');
        tree.append(svg);
        const rows = familyRows(members, gens);
        for (const [i, g] of gens.entries()) {
          const col = document.createElement('div');
          col.className = 'dm-col';
          col.setAttribute('role', 'list');
          const label = g === 0 ? 'Starters' : `Generation ${g}`;
          col.setAttribute('aria-label', label);
          col.innerHTML = `<h4>${label}</h4>`;
          for (const v of rows[i]) {
            const card = document.createElement('div');
            card.className = 'dm-fcard';
            card.dataset.id = v.id;
            card.setAttribute('role', 'listitem');
            card.innerHTML = `
              <button class="dm-card" data-act="play" data-id="${esc(v.id)}" aria-label="Watch ${esc(v.name)}">
                <span class="dm-cthumb"></span>
                <b class="ell">${esc(v.name)}${v.kept ? ` <span class="dm-heart">${icon('heart', 12)}</span>` : ''}</b>
                <span class="dm-meta">${esc(familyMeta(v))}</span>
              </button>
              <button class="dm-fsel" data-act="select" data-id="${esc(v.id)}" aria-pressed="false" aria-label="Select ${esc(v.name)} as a parent">${icon('check', 14)}</button>`;
            card.querySelector('.dm-cthumb')!.append(this.img(v.id, true));
            col.append(card);
          }
          tree.append(col);
        }
        this.queueEdges();
      }
    }

    const cur = this.host.current()?.id;
    tree.querySelectorAll<HTMLElement>('.dm-fcard').forEach((c) => {
      const id = c.dataset.id!;
      const sel = this.familySel.includes(id);
      c.classList.toggle('sel', sel);
      c.classList.toggle('playing', id === cur);
      c.querySelector('.dm-fsel')!.setAttribute('aria-pressed', String(sel));
    });
    const bsel = this.sheet.querySelector<HTMLButtonElement>('.dm-bsel')!;
    bsel.disabled = !this.familySel.length;
    bsel.querySelector('span')!.textContent = this.familySel.length ? `Breed selected (${this.familySel.length})` : 'Breed selected';
  }

  private toggleFamilySel(id: string): void {
    const i = this.familySel.indexOf(id);
    if (i >= 0) this.familySel.splice(i, 1);
    else {
      this.familySel.push(id);
      if (this.familySel.length > 2) this.familySel.shift();
    }
    this.renderFamily();
  }

  private queueEdges(): void {
    cancelAnimationFrame(this.edgeFrame);
    this.edgeFrame = requestAnimationFrame(() => this.drawEdges());
  }

  /**
   * Family-tree connectors, top down: two parents are joined by a line just under
   * them, a stem drops from its middle to a bar over their children, and each child
   * hangs from that bar. A mutant's single parent drops straight to its bar (dashed).
   */
  private drawEdges(): void {
    if (this.view !== 'family') return;
    const tree = this.sheet.querySelector<HTMLElement>('.dm-tree');
    const svg = tree?.querySelector<SVGSVGElement>('.dm-edges');
    if (!tree || !svg) return;
    const base = tree.getBoundingClientRect();
    const w = tree.scrollWidth;
    const h = tree.scrollHeight;
    svg.setAttribute('width', String(w));
    svg.setAttribute('height', String(h));
    svg.setAttribute('viewBox', `0 0 ${w} ${h}`);
    const rects = new Map<string, DOMRect>();
    tree.querySelectorAll<HTMLElement>('.dm-fcard').forEach((c) => rects.set(c.dataset.id!, c.getBoundingClientRect()));
    // One family unit per parent set: its children share a sibling bar.
    const units = new Map<string, { parents: DOMRect[]; kids: DOMRect[] }>();
    for (const c of tree.querySelectorAll<HTMLElement>('.dm-fcard')) {
      const v = this.host.get(c.dataset.id!);
      const cr = rects.get(c.dataset.id!);
      if (!v || !cr || !v.parents.length) continue;
      const pr = v.parents.map((p) => rects.get(p)).filter((r): r is DOMRect => !!r);
      if (!pr.length) continue;
      const k = v.parents.slice().sort().join('+');
      const u = units.get(k) ?? { parents: pr, kids: [] };
      u.kids.push(cr);
      units.set(k, u);
    }
    // Units over the same gap get their own lanes, so their bars don't run together.
    const lanes = new Map<number, number>();
    const f = (n: number) => n.toFixed(1);
    let bred = '';
    let mut = '';
    for (const u of units.values()) {
      const px = u.parents.map((r) => r.left + r.width / 2 - base.left);
      const pBottom = Math.max(...u.parents.map((r) => r.bottom - base.top));
      const kTop = Math.min(...u.kids.map((r) => r.top - base.top));
      const lane = lanes.get(Math.round(kTop)) ?? 0;
      lanes.set(Math.round(kTop), lane + 1);
      const gap = Math.max(16, kTop - pBottom);
      const step = Math.min(6, gap / 8);
      const joinY = pBottom + Math.min(14, gap / 3) + (lane % 3) * step;
      const barY = kTop - Math.min(14, gap / 3) - (lane % 3) * step;
      let d = '';
      for (const [i, r] of u.parents.entries()) d += `M${f(px[i])} ${f(r.bottom - base.top)}V${f(joinY)}`;
      const mx = px.reduce((a, b) => a + b, 0) / px.length;
      if (px.length > 1) d += `M${f(Math.min(...px))} ${f(joinY)}H${f(Math.max(...px))}`;
      const kx = u.kids.map((r) => r.left + r.width / 2 - base.left);
      d += `M${f(mx)} ${f(joinY)}V${f(barY)}`;
      d += `M${f(Math.min(mx, ...kx))} ${f(barY)}H${f(Math.max(mx, ...kx))}`;
      for (const [i, r] of u.kids.entries()) d += `M${f(kx[i])} ${f(barY)}V${f(r.top - base.top)}`;
      if (u.parents.length > 1) bred += d;
      else mut += d;
    }
    svg.innerHTML = `${bred ? `<path class="bred" d="${bred}"/>` : ''}${mut ? `<path class="mut" d="${mut}"/>` : ''}`;
  }

  // ------------------------------------------------------------- thumbnails

  /** An <img> for a visual's thumbnail: cached, lazy (observed) or loaded now. */
  private img(id: string, lazy: boolean): HTMLImageElement {
    const img = document.createElement('img');
    img.alt = '';
    img.decoding = 'async';
    img.dataset.id = id;
    const cached = this.thumbs.get(id);
    if (cached) img.src = cached;
    else if (lazy && this.thumbObs) this.thumbObs.observe(img);
    else this.loadThumb(img);
    return img;
  }

  private loadThumb(img: HTMLImageElement): void {
    const id = img.dataset.id!;
    void this.host.thumb(id).then(
      (url) => {
        if (!url) return;
        this.thumbs.set(id, url);
        if (img.dataset.id === id) img.src = url;
      },
      () => undefined,
    );
  }
}

// ------------------------------------------------------------- helpers

function el(tag: string, cls: string, html: string): HTMLElement {
  const e = document.createElement(tag);
  e.className = cls;
  e.innerHTML = html;
  return e;
}

function step(ic: IconName, title: string, text: string): string {
  return `<div class="dm-step"><span class="dm-step-ic">${icon(ic, 20)}</span><b>${title}</b><span>${esc(text)}</span></div>`;
}

function esc(s: string): string {
  return s.replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c]!);
}

function meta(v: DemoVisual): string {
  return `Gen ${v.gen}${v.seed ? ' · starter' : ''}${v.kept ? ' · ♥ kept' : ''}`;
}

function familyMeta(v: DemoVisual): string {
  if (v.seed || !v.parents.length) return `Gen ${v.gen} · starter`;
  if (v.parents.length === 1) return `Gen ${v.gen} · mutated`;
  const t = v.tag && TAG[v.tag] ? TAG[v.tag]!.label.toLowerCase() : 'bred';
  return `Gen ${v.gen} · ${t}`;
}

function fmtTime(s: number): string {
  if (!Number.isFinite(s) || s < 0) s = 0;
  const m = Math.floor(s / 60);
  const r = Math.floor(s % 60);
  return `${m}:${r < 10 ? '0' : ''}${r}`;
}

function clamp01(n: number): number {
  return Number.isFinite(n) ? Math.min(1, Math.max(0, n)) : 0.5;
}

/**
 * Generation rows for the family tree, ordered to keep lines short: each row follows
 * the average position of its parents (siblings side by side), then the top row is
 * reordered under its children.
 */
function familyRows(members: DemoVisual[], gens: number[]): DemoVisual[][] {
  const rows = gens.map((g) => members.filter((m) => m.gen === g).sort((a, b) => a.created - b.created));
  const pos = new Map<string, number>();
  const place = (row: DemoVisual[]) => row.forEach((v, i) => pos.set(v.id, (i + 0.5) / row.length));
  const mean = (xs: number[]) => (xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : NaN);
  const byParents = (row: DemoVisual[]) => {
    const at = new Map(row.map((v, i) => [v.id, i] as const));
    const key = (v: DemoVisual) => mean(v.parents.map((p) => pos.get(p)).filter((x): x is number => x !== undefined));
    return row.slice().sort((a, b) => {
      const ka = key(a), kb = key(b);
      if (Number.isNaN(ka) || Number.isNaN(kb)) return Number.isNaN(ka) === Number.isNaN(kb) ? at.get(a.id)! - at.get(b.id)! : Number.isNaN(ka) ? 1 : -1;
      return ka - kb || a.parents.slice().sort().join().localeCompare(b.parents.slice().sort().join()) || at.get(a.id)! - at.get(b.id)!;
    });
  };
  place(rows[0]);
  for (let i = 1; i < rows.length; i++) {
    rows[i] = byParents(rows[i]);
    place(rows[i]);
  }
  if (rows.length > 1) {
    // Top row under its children, then the rows below follow again.
    const kids = new Map<string, number[]>();
    for (const row of rows.slice(1)) for (const v of row) for (const p of v.parents) kids.set(p, [...(kids.get(p) ?? []), pos.get(v.id)!]);
    const at = new Map(rows[0].map((v, i) => [v.id, i] as const));
    rows[0] = rows[0].slice().sort((a, b) => (mean(kids.get(a.id) ?? []) || 2) - (mean(kids.get(b.id) ?? []) || 2) || at.get(a.id)! - at.get(b.id)!);
    place(rows[0]);
    for (let i = 1; i < rows.length; i++) {
      rows[i] = byParents(rows[i]);
      place(rows[i]);
    }
  }
  return rows;
}
