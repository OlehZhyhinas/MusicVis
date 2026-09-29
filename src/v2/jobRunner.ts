/** A step can finish, continue, or wait for an asynchronous operation. */
export type Step = () => boolean | 'yield';

/** Share the GPU with the live view without queueing an entire simulation frame burst. */
export class JobRunner {
  private jobs: Step[] = [];
  private previews: Step[] = [];
  private fence: WebGLSync | null = null;

  constructor(private gl?: WebGL2RenderingContext) {}

  enqueue(step: Step, priority: 'background' | 'preview' = 'background'): void {
    (priority === 'preview' ? this.previews : this.jobs).push(step);
  }

  get busy(): boolean {
    return this.queued > 0 || this.fence !== null;
  }

  get queued(): number {
    return this.jobs.length + this.previews.length;
  }

  pump(budgetMs: number): void {
    const gl = this.gl;
    if (this.fence && gl) {
      const status = gl.clientWaitSync(this.fence, 0, 0);
      if (status === gl.TIMEOUT_EXPIRED) return;
      gl.deleteSync(this.fence);
      this.fence = null;
      if (status === gl.WAIT_FAILED) return;
    }
    const t0 = performance.now();
    let steps = 0;
    // CPU submission time alone misses queued GPU cost. Bound each batch as
    // well, then let the live view present before submitting another batch.
    while (this.queued && steps < 4 && performance.now() - t0 < budgetMs) {
      // Preview jobs use their own stages. They can run between background
      // steps without resetting an in-progress screening/fingerprint stage.
      const queue = this.previews.length ? this.previews : this.jobs;
      const result = queue[0]();
      if (result === 'yield') break;
      steps++;
      if (result) queue.shift();
    }
    if (steps && gl) {
      this.fence = gl.fenceSync(gl.SYNC_GPU_COMMANDS_COMPLETE, 0);
      gl.flush();
    }
  }
}
