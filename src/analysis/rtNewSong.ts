// Live track-change detection: a new song (or a DJ's next track taking over) is a sustained change
// of the music's fingerprint that the current song has not had anywhere so far.
//
// Every CHUNK_S the analyzer's features are summarised into a chunk: mean chroma (key / chord
// colour), mean timbre shape (the 16-band spectral envelope with its level removed), level and
// tempo. The last RECENT_S of chunks is compared with every earlier chunk of the current song;
// novelty is the distance to the NEAREST of them, so a chorus, a breakdown or a bridge that echoes
// anything heard before stays familiar, while a new song (other key, other sound, other tempo)
// is far from all of it. Novelty above a threshold for SUSTAIN_S announces a track change on the
// next downbeat; a near-silent gap (>= GAP_S) lowers the threshold, it is not required. After a
// change the song memory restarts from the new track's chunks.
//
// Pure code (no DOM), fed at ~10 Hz by RealtimeAnalyzer.

export interface NewSongParams {
  /** Per-component thresholds: tempo (0..1, 1 = at least 3 % apart), key profile (1 - cosine), timbre (6 dB rms = 1), level (10 dB = 1). */
  thTempo: number;
  thKey: number;
  thTimbre: number;
  thLevel: number;
  /** Components that must agree without a gap before, and after a near-silent gap. */
  need: number;
  needGap: number;
  /** Seconds the agreement must hold. */
  sustain: number;
  /** Seconds of the current song needed before a change can be announced. */
  minMemory: number;
  /** Seconds after a change before the next can be announced. */
  cooldown: number;
}

export const NEW_SONG_DEFAULTS: NewSongParams = {
  thTempo: 0.99,
  thKey: 0.06,
  thTimbre: 1.2,
  thLevel: 2.0,
  need: 3,
  needGap: 1,
  sustain: 5,
  minMemory: 20,
  cooldown: 20,
};

const CHUNK_S = 1;
const RECENT_S = 5;
/** Recent chunks are compared with chunks older than this guard (the blend or boundary between). */
const GUARD_S = 3;
const GAP_S = 1;
/** Seconds after a gap during which it lowers the threshold. */
const GAP_MEMORY_S = 12;
const MAX_CHUNKS = 900; // 15 min of song memory
const DOWNBEAT_WAIT_S = 3;

interface Chunk {
  chroma: Float32Array; // 12, L2-normalised
  timbre: Float32Array; // 16, dB minus the chunk's mean band level
  db: number;
  bpm: number; // NaN when the beat is not locked
}

export class NewSongDetector {
  readonly p: NewSongParams;
  private readonly song: Chunk[] = [];
  private acc = { n: 0, chroma: new Float64Array(12), timbre: new Float64Array(16), db: 0, bpm: 0, bpmN: 0 };
  private chunkStart = -Infinity;
  private above = 0;
  private lastChange = -Infinity;
  private silentFor = 0;
  private lastGapEnd = -Infinity;
  private pendingSince = NaN;
  private prevBar = NaN;
  /** Latest novelty (for meters and tuning). */
  novelty = 0;
  /** Seconds of music in the current song memory. */
  memory = 0;

  constructor(p: Partial<NewSongParams> = {}) {
    this.p = { ...NEW_SONG_DEFAULTS, ...p };
  }

  reset(): void {
    this.song.length = 0;
    this.clearAcc();
    this.chunkStart = -Infinity;
    this.above = 0;
    this.lastChange = -Infinity;
    this.silentFor = 0;
    this.lastGapEnd = -Infinity;
    this.pendingSince = NaN;
    this.novelty = 0;
    this.memory = 0;
  }

  /** Forget the song so far (keep the last `keepS` seconds as the start of the next one). */
  forget(keepS = RECENT_S): void {
    const keep = this.song.slice(Math.max(0, this.song.length - Math.round(keepS / CHUNK_S)));
    this.song.length = 0;
    this.song.push(...keep);
    this.memory = this.song.length * CHUNK_S;
  }

  private clearAcc(): void {
    this.acc.n = 0;
    this.acc.chroma.fill(0);
    this.acc.timbre.fill(0);
    this.acc.db = 0;
    this.acc.bpm = 0;
    this.acc.bpmN = 0;
  }

  /**
   * One step (~0.1 s). Returns true on the step a track change is announced.
   * gate: 0 silent .. 1 music; conf: beat confidence; bar: 0..1 position in the bar (NaN unknown).
   */
  push(t: number, chroma: ArrayLike<number>, timbre: ArrayLike<number>, db: number, gate: number, bpm: number, conf: number, bar: number, dt = 0.1): boolean {
    const music = gate > 0.5;
    if (!music) {
      this.silentFor += dt;
    } else {
      if (this.silentFor >= GAP_S) this.lastGapEnd = t;
      this.silentFor = 0;
      const a = this.acc;
      a.n++;
      for (let k = 0; k < 12; k++) a.chroma[k] += chroma[k];
      for (let k = 0; k < 16; k++) a.timbre[k] += timbre[k];
      a.db += db;
      if (conf > 0.5 && bpm > 0) {
        a.bpm += bpm;
        a.bpmN++;
      }
    }
    if (!Number.isFinite(this.chunkStart)) this.chunkStart = t;
    if (t - this.chunkStart >= CHUNK_S) {
      this.chunkStart = t;
      if (this.acc.n >= 0.6 * CHUNK_S / dt) this.addChunk();
      this.clearAcc();
      this.evaluate(t);
    }
    // A pending change is announced on the next downbeat (or after DOWNBEAT_WAIT_S without one).
    let fire = false;
    if (Number.isFinite(this.pendingSince)) {
      const wrapped = Number.isFinite(bar) && Number.isFinite(this.prevBar) && bar < this.prevBar - 0.5;
      if (wrapped || t - this.pendingSince >= DOWNBEAT_WAIT_S) {
        this.pendingSince = NaN;
        fire = true;
      }
    }
    this.prevBar = bar;
    return fire;
  }

  private addChunk(): void {
    const a = this.acc;
    const chroma = new Float32Array(12);
    let nn = 0;
    for (let k = 0; k < 12; k++) nn += (a.chroma[k] / a.n) ** 2;
    nn = Math.sqrt(nn) || 1;
    for (let k = 0; k < 12; k++) chroma[k] = a.chroma[k] / a.n / nn;
    const timbre = new Float32Array(16);
    let m = 0;
    for (let k = 0; k < 16; k++) m += a.timbre[k] / a.n / 16;
    for (let k = 0; k < 16; k++) timbre[k] = a.timbre[k] / a.n - m;
    this.song.push({ chroma, timbre, db: a.db / a.n, bpm: a.bpmN >= 3 ? a.bpm / a.bpmN : NaN });
    if (this.song.length > MAX_CHUNKS) this.song.shift();
    this.memory = this.song.length * CHUNK_S;
  }

  /** Latest component novelties (tempo, key profile, timbre, level), for tuning. */
  comp = { tempo: 0, key: 0, timbre: 0, level: 0 };

  private evaluate(t: number): void {
    const s = this.song;
    const nRecent = Math.round(RECENT_S / CHUNK_S);
    const guard = Math.round(GUARD_S / CHUNK_S);
    const nPast = s.length - nRecent - guard;
    if (nPast * CHUNK_S < this.p.minMemory || t - this.lastChange < this.p.cooldown) {
      this.above = 0;
      this.novelty = 0;
      this.comp.tempo = this.comp.key = this.comp.timbre = this.comp.level = 0;
      return;
    }
    const recent = s.slice(s.length - nRecent);
    const past = s.slice(0, nPast);
    // Tempo: medians (robust to the tracker's glitches), folded so half / double / 3:2 time match.
    const med = (xs: number[]) => {
      const v = xs.filter(Number.isFinite).sort((p, q) => p - q);
      return v.length ? v[v.length >> 1] : NaN;
    };
    const bR = med(recent.map((c) => c.bpm)), bP = med(past.map((c) => c.bpm));
    let dTempo = 0;
    if (Number.isFinite(bR) && Number.isFinite(bP)) {
      let r = Math.abs(Math.log2(bR / bP));
      r = Math.min(r, Math.abs(r - 1), Math.abs(r - Math.log2(1.5)));
      dTempo = Math.min(1, r / 0.03);
    }
    // Key profile: the recent mean chroma against the song's long mean chroma.
    const mean12 = (cs: Chunk[]) => {
      const m = new Float64Array(12);
      for (const c of cs) for (let k = 0; k < 12; k++) m[k] += c.chroma[k];
      let n = 0;
      for (let k = 0; k < 12; k++) n += m[k] * m[k];
      n = Math.sqrt(n) || 1;
      for (let k = 0; k < 12; k++) m[k] /= n;
      return m;
    };
    const cR = mean12(recent), cP = mean12(past);
    let dot = 0;
    for (let k = 0; k < 12; k++) dot += cR[k] * cP[k];
    const dKey = 1 - dot;
    // Timbre: the recent mean shape against the nearest RECENT_S window of the song so far.
    const mean16 = (cs: Chunk[]) => {
      const m = new Float64Array(16);
      for (const c of cs) for (let k = 0; k < 16; k++) m[k] += c.timbre[k] / cs.length;
      return m;
    };
    const tR = mean16(recent);
    let dTim = Infinity;
    for (let j = 0; j + nRecent <= nPast; j++) {
      const tP = mean16(past.slice(j, j + nRecent));
      let d = 0;
      for (let k = 0; k < 16; k++) d += (tR[k] - tP[k]) ** 2;
      dTim = Math.min(dTim, Math.sqrt(d / 16) / 6);
    }
    // Level: the recent level against the song's loud level (a quieter new song or an ad).
    const dbs = past.map((c) => c.db).sort((p, q) => p - q);
    const loud = dbs[Math.floor(dbs.length * 0.9)];
    const dLev = Math.abs(recent.reduce((a, c) => a + c.db, 0) / recent.length - loud) / 10;
    this.comp.tempo = dTempo;
    this.comp.key = dKey;
    this.comp.timbre = dTim;
    this.comp.level = dLev;
    const p = this.p;
    const votes = (dTempo > p.thTempo ? 1 : 0) + (dKey > p.thKey ? 1 : 0) + (dTim > p.thTimbre ? 1 : 0) + (dLev > p.thLevel ? 1 : 0);
    this.novelty = votes;
    const gap = t - this.lastGapEnd < GAP_MEMORY_S;
    this.above = votes >= (gap ? p.needGap : p.need) ? this.above + CHUNK_S : 0;
    if (this.above >= p.sustain && !Number.isFinite(this.pendingSince)) {
      this.pendingSince = t;
      this.lastChange = t;
      this.above = 0;
      this.forget();
    }
  }
}
