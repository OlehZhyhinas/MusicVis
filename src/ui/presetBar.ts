// Preset bar (above the transport): the playing preset with its thumbnail,
// like / dislike with counts, score, next preset, Evolve and the Presets tab.

export interface PresetInfo {
  id: string;
  name: string;
  type: string;
  energy: string;
  likes: number;
  dislikes: number;
  score: number; // 0..1
}

export interface PresetBarCallbacks {
  onLike(): void;
  onDislike(): void;
  onNext(): void;
  onEvolve(): void;
  onPresets(): void;
  thumb(id: string): Promise<string>;
}

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;

export class PresetBar {
  private thumb = $<HTMLImageElement>('pb-thumb');
  private idEl = $<HTMLElement>('pb-id');
  private nameEl = $<HTMLElement>('pb-name');
  private metaEl = $<HTMLElement>('pb-meta');
  private likeBtn = $<HTMLButtonElement>('v2-like');
  private dislikeBtn = $<HTMLButtonElement>('v2-dislike');
  private likesEl = $<HTMLElement>('v2-likes');
  private dislikesEl = $<HTMLElement>('v2-dislikes');
  private scoreEl = $<HTMLElement>('v2-score');
  private evolveBtn = $<HTMLButtonElement>('v2-evolve');
  private statusEl = $<HTMLElement>('v2-status');
  private presetsBtn = $<HTMLButtonElement>('v2-open-browser');
  private shownId = '';

  constructor(private cb: PresetBarCallbacks) {
    this.likeBtn.addEventListener('click', () => cb.onLike());
    this.dislikeBtn.addEventListener('click', () => cb.onDislike());
    $('pb-next').addEventListener('click', () => cb.onNext());
    this.evolveBtn.addEventListener('click', () => cb.onEvolve());
    this.presetsBtn.addEventListener('click', () => cb.onPresets());
  }

  update(p: PresetInfo | null, status: string, evolveOn: boolean): void {
    if (p) {
      this.idEl.textContent = p.id;
      this.nameEl.textContent = p.name;
      this.nameEl.title = `${p.id} · ${p.name}`;
      this.metaEl.textContent = `${p.type} · ${p.energy}`;
      this.likesEl.textContent = String(p.likes);
      this.dislikesEl.textContent = String(p.dislikes);
      this.scoreEl.textContent = `score ${Math.round(p.score * 100)}`;
      if (p.id !== this.shownId) {
        this.shownId = p.id;
        this.thumb.removeAttribute('src');
        const id = p.id;
        void this.cb.thumb(id).then((url) => {
          if (this.shownId === id && url) this.thumb.src = url;
        });
      }
    }
    this.evolveBtn.classList.toggle('on', evolveOn);
    this.evolveBtn.setAttribute('aria-pressed', String(evolveOn));
    this.evolveBtn.querySelector('.tog')?.classList.toggle('is-on', evolveOn);
    this.statusEl.textContent = status;
  }

  /** Pop animation on the vote button that was pressed. */
  voted(like: boolean): void {
    const b = like ? this.likeBtn : this.dislikeBtn;
    b.classList.remove('pop');
    void b.offsetWidth;
    b.classList.add('pop');
  }

  setPresetsOpen(open: boolean): void {
    this.presetsBtn.classList.toggle('active', open);
    this.presetsBtn.setAttribute('aria-pressed', String(open));
  }
}
