// AudioWorklet processor for live input: batches the incoming stereo samples
// into ~10 ms blocks and hands them to the main thread (buffers transferred).
// It produces silence on its output; nothing reaches the speakers.

class CaptureProcessor extends AudioWorkletProcessor {
  constructor(options) {
    super();
    const opts = (options && options.processorOptions) || {};
    this.size = opts.batch > 0 ? opts.batch : 512;
    this.l = new Float32Array(this.size);
    this.r = new Float32Array(this.size);
    this.n = 0;
    this.mono = true;
    this.stopped = false;
    this.port.onmessage = (e) => {
      if (e.data === 'stop') this.stopped = true;
    };
  }

  flush() {
    if (this.mono) {
      this.port.postMessage({ l: this.l, r: null }, [this.l.buffer]);
    } else {
      this.port.postMessage({ l: this.l, r: this.r }, [this.l.buffer, this.r.buffer]);
      this.r = new Float32Array(this.size);
    }
    this.l = new Float32Array(this.size);
    this.n = 0;
    this.mono = true;
  }

  process(inputs) {
    if (this.stopped) return false;
    const inp = inputs[0];
    const a = inp && inp.length > 0 ? inp[0] : null;
    const b = inp && inp.length > 1 ? inp[1] : null;
    const len = a ? a.length : 128;
    for (let i = 0; i < len; i++) {
      const x = a ? a[i] : 0;
      this.l[this.n] = x;
      if (b) {
        const y = b[i];
        this.r[this.n] = y;
        if (y !== x) this.mono = false;
      } else {
        this.r[this.n] = x;
      }
      if (++this.n === this.size) this.flush();
    }
    return true;
  }
}

registerProcessor('musicvis-capture', CaptureProcessor);
