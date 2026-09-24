// Exploration controls in the Presets tab: the Off / Gentle / Explore / Wild
// segmented control (persisted) in a tools row under the filters. Built from
// JS so the shared markup stays untouched; later chunks add the List | Map
// toggle and the similarity page button to the same row.

import './novelty.css';
import { EXPLORE_LABEL, EXPLORE_MODES, EXPLORE_WEIGHT, parseExploreMode, type ExploreMode } from './novelty';
import { loadSetting, saveSetting } from '../ui/storage';

export const EXPLORE_KEY = 'v2.explore';

const HINT: Record<ExploreMode, string> = {
  off: 'Votes and fitness only',
  gentle: 'A small nudge toward presets that look unlike anything seen so far',
  explore: 'Novel-looking presets get real airtime; familiar-looking children are turned away',
  wild: 'Novelty first until votes come in; only clearly new-looking children are kept',
};

export function loadExploreMode(): ExploreMode {
  return parseExploreMode(loadSetting<string>(EXPLORE_KEY, 'gentle'));
}

export class ExploreControls {
  readonly tools: HTMLElement;
  private seg: HTMLElement;
  private mode: ExploreMode;

  constructor(mode: ExploreMode, private onMode: (m: ExploreMode) => void) {
    this.mode = mode;
    const browser = document.getElementById('v2-browser')!;
    const filters = browser.querySelector('.filters');
    this.tools = document.createElement('div');
    this.tools.className = 'v2b-tools';
    this.tools.innerHTML = `<span class="v2b-tools-l" id="v2b-explore-l">Exploration</span><div class="seg" id="v2b-explore" role="group" aria-labelledby="v2b-explore-l">${EXPLORE_MODES.map(
      (m) => `<button data-v="${m}" aria-pressed="${m === mode}" title="${HINT[m]}${EXPLORE_WEIGHT[m] ? ` (novelty weight ${EXPLORE_WEIGHT[m]})` : ''}">${EXPLORE_LABEL[m]}</button>`,
    ).join('')}</div><span class="grow"></span>`;
    filters?.after(this.tools);
    this.seg = this.tools.querySelector('#v2b-explore')!;
    this.seg.addEventListener('click', (ev) => {
      const b = (ev.target as HTMLElement).closest<HTMLButtonElement>('button[data-v]');
      if (b) this.set(parseExploreMode(b.dataset.v));
    });
  }

  get value(): ExploreMode {
    return this.mode;
  }

  set(m: ExploreMode): void {
    this.mode = m;
    saveSetting(EXPLORE_KEY, m);
    for (const b of this.seg.querySelectorAll<HTMLButtonElement>('button')) b.setAttribute('aria-pressed', String(b.dataset.v === m));
    this.onMode(m);
  }

  /** Cycles Off → Gentle → Explore → Wild → Off (command palette). */
  cycle(): ExploreMode {
    const m = EXPLORE_MODES[(EXPLORE_MODES.indexOf(this.mode) + 1) % EXPLORE_MODES.length];
    this.set(m);
    return m;
  }

  static hint(m: ExploreMode): string {
    return HINT[m];
  }
}
