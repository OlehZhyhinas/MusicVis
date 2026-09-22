// AudioContext-based playback of a decoded AudioBuffer.
// A fresh AudioBufferSourceNode is created on every play()/seek() since
// source nodes are single-use in the Web Audio API.

export class Player {
  readonly context: AudioContext;
  readonly output: GainNode;

  private buffer: AudioBuffer | null = null;
  private source: AudioBufferSourceNode | null = null;

  /** context.currentTime when playback of the buffer started, adjusted for offset. */
  private startedAtContextTime = 0;
  /** Position within the buffer (seconds) that playback started/resumed from. */
  private startOffset = 0;
  private _playing = false;
  private endedTimer: ReturnType<typeof setTimeout> | null = null;

  onended: (() => void) | null = null;

  constructor(context: AudioContext) {
    this.context = context;
    this.output = context.createGain();
    this.output.connect(context.destination);
  }

  get duration(): number {
    return this.buffer ? this.buffer.duration : 0;
  }

  get playing(): boolean {
    return this._playing;
  }

  get volume(): number {
    return this.output.gain.value;
  }

  set volume(v: number) {
    this.output.gain.value = Math.max(0, Math.min(1, v));
  }

  get currentTime(): number {
    if (!this.buffer) return 0;
    if (!this._playing) return Math.min(this.startOffset, this.buffer.duration);
    const elapsed = this.context.currentTime - this.startedAtContextTime;
    return Math.min(this.startOffset + elapsed, this.buffer.duration);
  }

  /** Load a new buffer, replacing any current one. Stops playback. */
  load(buffer: AudioBuffer): void {
    this.stopInternal();
    this.buffer = buffer;
    this.startOffset = 0;
    this._playing = false;
  }

  play(): void {
    if (!this.buffer || this._playing) return;
    this.startSourceFrom(this.startOffset);
  }

  pause(): void {
    if (!this.buffer || !this._playing) return;
    const pos = this.currentTime;
    this.stopSourceOnly();
    this.startOffset = pos;
    this._playing = false;
  }

  seek(t: number): void {
    if (!this.buffer) return;
    const clamped = Math.max(0, Math.min(t, this.buffer.duration));
    const wasPlaying = this._playing;
    this.stopSourceOnly();
    this.startOffset = clamped;
    if (wasPlaying) {
      this.startSourceFrom(clamped);
    } else {
      this._playing = false;
    }
  }

  private startSourceFrom(offset: number): void {
    if (!this.buffer) return;
    const source = this.context.createBufferSource();
    source.buffer = this.buffer;
    source.connect(this.output);
    source.onended = () => {
      // Only treat as "ended" if this is still the active source
      // (not stopped early by pause/seek).
      if (this.source === source) {
        this._playing = false;
        this.source = null;
        this.onended?.();
      }
    };
    this.source = source;
    this.startedAtContextTime = this.context.currentTime;
    this.startOffset = offset;
    this._playing = true;
    source.start(0, offset);
  }

  private stopSourceOnly(): void {
    if (this.source) {
      const s = this.source;
      s.onended = null;
      this.source = null;
      try {
        s.stop();
      } catch {
        // already stopped
      }
    }
    if (this.endedTimer !== null) {
      clearTimeout(this.endedTimer);
      this.endedTimer = null;
    }
  }

  private stopInternal(): void {
    this.stopSourceOnly();
    this._playing = false;
  }

  dispose(): void {
    this.stopInternal();
    try {
      this.output.disconnect();
    } catch {
      // ignore
    }
  }
}
