// Classic MilkDrop via butterchurn. Butterchurn owns its own WebGL2 context on
// an offscreen canvas; each frame the canvas is uploaded into a texture of the
// main context.

import type { ButterchurnVisualizer } from 'butterchurn';
import type { GL } from './gl';
import type { MusicState } from '../types';

const CURATED = [
  'Flexi - mindblob [shiny mix]',
  'Flexi + Martin - astral projection',
  'Flexi - predator-prey-spirals',
  'Flexi, martin + geiss - dedicated to the sherwin maxawow',
  'Geiss - Cauldron - painterly 2 (saturation remix)',
  'Geiss - Reaction Diffusion 2',
  'Rovastar + Loadus + Geiss - FractalDrop (Triple Mix)',
  'Zylot - True Visionary (Final Mix)',
  'Unchained - Rewop',
  'martin - castle in the air',
  'martin - witchcraft reloaded',
  'martin - mandelbox explorer - high speed demo version',
  'Aderrasi - Potion of Spirits',
  'Cope - The Neverending Explosion of Red Liquid Fire',
  'Eo.S. + Phat - cubetrace - v2',
  'flexi - swing out on the spiral',
  'Krash + Illusion - Spiral Movement',
  '_Geiss - Artifact 01',
  '$$$ Royal - Mashup (431)',
  'Flexi - infused with the spiral',
];

/** FNV-1a 32-bit hash as 8 hex chars. */
function fnv1a(s: string): string {
  let h = 0x811c9dc5;
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i);
    h = Math.imul(h, 0x01000193) >>> 0;
  }
  return h.toString(16).padStart(8, '0');
}

/** Stable IDs from preset names: "C-" + 4 hex chars, 6 for names that collide. */
export function classicIds(names: string[]): Map<string, string> {
  const short = new Map<string, string[]>();
  for (const n of names) {
    const k = fnv1a(n).slice(0, 4);
    const l = short.get(k);
    if (l) l.push(n);
    else short.set(k, [n]);
  }
  const ids = new Map<string, string>();
  for (const [k, list] of short) {
    for (const n of list) ids.set(n, 'C-' + (list.length > 1 ? fnv1a(n).slice(0, 6) : k));
  }
  return ids;
}

// Interop helper: CommonJS/UMD modules may arrive wrapped one or two levels deep.
function unwrap<T>(mod: unknown, key: string): T {
  let m = mod as Record<string, unknown>;
  for (let i = 0; i < 3 && m && !(key in m) && 'default' in m; i++) m = m.default as Record<string, unknown>;
  return m as T;
}

export class Classic {
  readonly canvas: HTMLCanvasElement;
  private viz: ButterchurnVisualizer | null = null;
  private presets: Record<string, object> = {};
  private names: string[] = [];
  private curated: string[] = [];
  private current = '';
  private ids = new Map<string, string>();
  private byId = new Map<string, string>();
  private tex: WebGLTexture;
  private loading: Promise<void> | null = null;
  private w = 0;
  private h = 0;
  failed = false;

  constructor(
    private gl: GL,
    private audio: { context: AudioContext; source: AudioNode },
  ) {
    this.canvas = document.createElement('canvas');
    const t = gl.createTexture();
    if (!t) throw new Error('[render] cannot create texture');
    this.tex = t;
    gl.bindTexture(gl.TEXTURE_2D, t);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, 1, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE, new Uint8Array([0, 0, 0, 255]));
  }

  get ready(): boolean {
    return this.viz !== null;
  }
  get texture(): WebGLTexture {
    return this.tex;
  }
  get presetName(): string {
    return this.current;
  }
  get presetId(): string {
    return this.ids.get(this.current) ?? '';
  }

  /** Lazily import butterchurn and its presets. Safe to call repeatedly. */
  load(w: number, h: number): Promise<void> {
    if (this.loading) return this.loading;
    this.loading = (async () => {
      try {
        const [bcMod, presetMod] = await Promise.all([import('butterchurn'), import('butterchurn-presets')]);
        const bc = unwrap<{ createVisualizer: typeof import('butterchurn').default.createVisualizer }>(bcMod, 'createVisualizer');
        const lib = unwrap<{ getPresets(): Record<string, object> }>(presetMod, 'getPresets');
        this.presets = lib.getPresets();
        this.names = Object.keys(this.presets);
        this.curated = CURATED.filter((n) => n in this.presets);
        this.ids = classicIds(this.names);
        for (const [n, id] of this.ids) this.byId.set(id, n);
        this.w = w;
        this.h = h;
        this.canvas.width = w;
        this.canvas.height = h;
        const viz = bc.createVisualizer(this.audio.context, this.canvas, { width: w, height: h, pixelRatio: 1, textureRatio: 1 });
        viz.connectAudio(this.audio.source);
        this.viz = viz;
        this.pick(0);
      } catch (e) {
        this.failed = true;
        console.error('[render] butterchurn failed to load', e);
      }
    })();
    return this.loading;
  }

  setSize(w: number, h: number): void {
    if (!this.viz || (w === this.w && h === this.h)) return;
    this.w = w;
    this.h = h;
    this.canvas.width = w;
    this.canvas.height = h;
    this.viz.setRendererSize(w, h);
  }

  /** Load a random (preferably curated) preset. */
  pick(blend: number): void {
    if (!this.viz || !this.names.length) return;
    const pool = this.curated.length >= 4 && Math.random() < 0.8 ? this.curated : this.names;
    let name = pool[Math.floor(Math.random() * pool.length)];
    if (name === this.current && pool.length > 1) name = pool[(pool.indexOf(name) + 1) % pool.length];
    this.load1(name, blend);
  }

  /** Load a preset by its stable ID ("C-xxxx"). */
  pickById(id: string, blend: number): boolean {
    const name = this.byId.get(id);
    if (!name || !this.viz) return false;
    this.load1(name, blend);
    return true;
  }

  private load1(name: string, blend: number): void {
    this.current = name;
    try {
      this.viz!.loadPreset(this.presets[name], blend);
    } catch (e) {
      console.warn('[render] butterchurn preset failed', name, e);
    }
  }

  /** Classic-mode switching: only when a drop section starts. */
  autoSwitch(state: MusicState): void {
    if (!this.viz) return;
    if (state.sectionChanged && state.section?.label === 'drop') this.pick(0.5);
  }

  /** Render a butterchurn frame and upload it into the texture. */
  render(): boolean {
    if (!this.viz) return false;
    const gl = this.gl;
    try {
      this.viz.render();
    } catch (e) {
      console.warn('[render] butterchurn render error', e);
      return false;
    }
    gl.bindTexture(gl.TEXTURE_2D, this.tex);
    gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, true);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, gl.RGBA, gl.UNSIGNED_BYTE, this.canvas);
    gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false);
    return true;
  }

  dispose(): void {
    if (this.viz) {
      try {
        this.viz.disconnectAudio(this.audio.source);
      } catch {
        /* already disconnected */
      }
      const ctx = this.canvas.getContext('webgl2');
      ctx?.getExtension('WEBGL_lose_context')?.loseContext();
    }
    this.viz = null;
    this.gl.deleteTexture(this.tex);
  }
}
