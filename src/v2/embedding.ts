// Perceptual embedding (opt-in): DINOv2-small, run in the browser through
// transformers.js (WebGPU, WASM fallback), embeds a few frames of each preset's
// reference-clip render into one 384-d vector. Its cosine distance is blended
// into the look metric as a sixth term whose weight the similarity answers
// decide. transformers.js and the model are fetched only when the user turns
// this on (a pinned transformers.js build from jsDelivr, the ONNX model from
// the Hugging Face hub, both cached by the browser afterwards).
//
// The maths (cosine, term scaling, the store) is pure so it runs in the Node
// tests; the loader only runs in the browser.

export const EMB_MODEL = 'Xenova/dinov2-small';
export const EMB_DIM = 384;
export const TRANSFORMERS_URL = 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1/+esm';

export interface EmbeddingData {
  format: 'musicvis-v2-embeddings';
  model: string;
  vectors: Record<string, number[]>;
}

export function validEmbedding(v: unknown): v is number[] {
  return Array.isArray(v) && v.length === EMB_DIM && v.every((x) => typeof x === 'number' && Number.isFinite(x));
}

export function normalize(v: ArrayLike<number>): number[] {
  let s = 0;
  for (let i = 0; i < v.length; i++) s += v[i] * v[i];
  const k = s > 0 ? 1 / Math.sqrt(s) : 0;
  return Array.from(v, (x) => Math.round(x * k * 1e5) / 1e5);
}

export function cosineDistance(a: number[], b: number[]): number {
  let d = 0, na = 0, nb = 0;
  for (let i = 0; i < a.length; i++) {
    d += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];
  }
  return na > 0 && nb > 0 ? Math.max(0, 1 - d / Math.sqrt(na * nb)) : 1;
}

/** Mean of several frame embeddings, L2-normalised. */
export function poolFrames(frames: ArrayLike<number>[]): number[] {
  const out = new Float64Array(frames[0]?.length ?? 0);
  for (const f of frames) {
    const n = normalize(f);
    for (let i = 0; i < out.length; i++) out[i] += n[i];
  }
  return normalize(out);
}

/**
 * Stored embeddings keyed by member id, plus the scale that turns a cosine
 * distance into a term comparable with the fingerprint's group terms (the
 * median pairwise cosine distance maps to 2, the expected squared z
 * difference of a typical pair).
 */
export class EmbeddingStore {
  vectors = new Map<string, number[]>();
  private scale = 0;
  private scaleN = -1;

  get size(): number {
    return this.vectors.size;
  }

  get(id: string): number[] | undefined {
    return this.vectors.get(id);
  }

  set(id: string, v: number[]): void {
    if (validEmbedding(v)) this.vectors.set(id, v);
  }

  /** Cosine distance divided by the median pairwise distance, times 2; NaN when either is missing. */
  term(a: string, b: string): number {
    const va = this.vectors.get(a), vb = this.vectors.get(b);
    if (!va || !vb) return NaN;
    return (2 * cosineDistance(va, vb)) / this.medianDistance();
  }

  medianDistance(): number {
    if (this.scaleN === this.vectors.size && this.scale > 0) return this.scale;
    const vs = [...this.vectors.values()];
    const ds: number[] = [];
    const step = Math.max(1, Math.floor(vs.length / 60));
    for (let i = 0; i < vs.length; i += step) for (let j = i + 1; j < vs.length; j += step) ds.push(cosineDistance(vs[i], vs[j]));
    ds.sort((x, y) => x - y);
    this.scale = Math.max(1e-4, ds.length ? ds[ds.length >> 1] : 0.3);
    this.scaleN = this.vectors.size;
    return this.scale;
  }

  toJSON(): EmbeddingData {
    return { format: 'musicvis-v2-embeddings', model: EMB_MODEL, vectors: Object.fromEntries(this.vectors) };
  }

  static fromJSON(data: unknown): EmbeddingStore {
    const s = new EmbeddingStore();
    const d = data as Partial<EmbeddingData> | undefined;
    if (!d || d.format !== 'musicvis-v2-embeddings' || d.model !== EMB_MODEL || !d.vectors) return s;
    for (const [id, v] of Object.entries(d.vectors)) s.set(id, v);
    return s;
  }
}

// ------------------------------------------------------------ browser

type Extractor = (input: unknown, opts?: Record<string, unknown>) => Promise<{ data: Float32Array; dims: number[] }>;

export interface EmbedderStatus {
  state: 'off' | 'loading' | 'ready' | 'error';
  device?: 'webgpu' | 'wasm';
  detail?: string;
  /** 0..1 while downloading. */
  progress?: number;
}

/** Lazily loads transformers.js + DINOv2-small and embeds canvases. */
export class Embedder {
  status: EmbedderStatus = { state: 'off' };
  onStatus: ((s: EmbedderStatus) => void) | null = null;
  private extractor: Extractor | null = null;
  private loading: Promise<boolean> | null = null;
  private RawImage: { fromCanvas(c: HTMLCanvasElement): unknown } | null = null;
  /** Average ms per embedded frame. */
  msPerFrame = 0;

  private set(s: EmbedderStatus): void {
    this.status = s;
    this.onStatus?.(s);
  }

  load(): Promise<boolean> {
    if (this.loading) return this.loading;
    this.loading = (async () => {
      this.set({ state: 'loading', detail: 'loading transformers.js', progress: 0 });
      try {
        const tf = (await import(/* @vite-ignore */ TRANSFORMERS_URL)) as Record<string, unknown>;
        this.RawImage = tf.RawImage as typeof this.RawImage;
        const pipeline = tf.pipeline as (task: string, model: string, opts: Record<string, unknown>) => Promise<Extractor>;
        const gpu = typeof navigator !== 'undefined' && 'gpu' in navigator && !!(await (navigator as unknown as { gpu: { requestAdapter(): Promise<unknown> } }).gpu.requestAdapter().catch(() => null));
        const device = gpu ? 'webgpu' : 'wasm';
        const files = new Map<string, number>();
        this.extractor = await pipeline('image-feature-extraction', EMB_MODEL, {
          device,
          dtype: gpu ? 'fp16' : 'q8',
          progress_callback: (p: { status: string; file?: string; loaded?: number; total?: number }) => {
            if (p.status === 'progress' && p.file && p.total) {
              files.set(p.file, (p.loaded ?? 0) / p.total);
              const vals = [...files.values()];
              this.set({ state: 'loading', device, detail: `downloading ${EMB_MODEL}`, progress: vals.reduce((s, v) => s + v, 0) / vals.length });
            }
          },
        });
        this.set({ state: 'ready', device, detail: `${EMB_MODEL} on ${device === 'webgpu' ? 'WebGPU' : 'WASM'}` });
        return true;
      } catch (err) {
        console.warn('[v2] perceptual embedding unavailable', err);
        this.loading = null;
        this.set({ state: 'error', detail: err instanceof Error ? err.message : String(err) });
        return false;
      }
    })();
    return this.loading;
  }

  /** One pooled, normalised embedding for a few frames (the CLS token of each). */
  async embed(frames: HTMLCanvasElement[]): Promise<number[] | null> {
    if (!this.extractor || !this.RawImage || !frames.length) return null;
    const t0 = performance.now();
    const outs: Float32Array[] = [];
    for (const c of frames) {
      const img = this.RawImage.fromCanvas(c);
      const r = await this.extractor(img);
      // [1, tokens, 384]: the CLS token is the first row.
      const dim = r.dims[r.dims.length - 1];
      outs.push(r.data.slice(0, dim));
    }
    this.msPerFrame += ((performance.now() - t0) / frames.length - this.msPerFrame) * 0.2;
    const v = poolFrames(outs);
    return validEmbedding(v) ? v : null;
  }
}
