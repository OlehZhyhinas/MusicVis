// node --experimental-transform-types --import ./scripts/analysis-test.hooks.mjs scripts/job-runner-tests.ts
import assert from 'node:assert/strict';
import { JobRunner } from '../src/v2/jobRunner';

// Waiting must yield to the browser instead of exhausting a compile timeout
// by polling hundreds of times in the same task.
{
  const runner = new JobRunner();
  let polls = 0, ready = false, frames = 0;
  runner.enqueue(() => {
    if (!ready) { polls++; return 'yield'; }
    return ++frames === 9;
  });
  runner.pump(1000);
  assert.equal(polls, 1);
  assert.equal(runner.queued, 1);
  ready = true;
  runner.pump(1000);
  assert.equal(frames, 4, 'fast CPU submission must not flood the GPU');
  runner.pump(1000);
  runner.pump(1000);
  assert.equal(frames, 9, 'every simulation step is retained');
  assert.equal(runner.busy, false);
}

// Visible previews preempt background work between steps, and both lanes keep
// their own FIFO order/state when a partly completed background job resumes.
{
  const runner = new JobRunner(), order: string[] = [];
  let background = 0;
  runner.enqueue(() => { order.push(`b${++background}`); return background === 6; });
  runner.pump(1000);
  runner.enqueue(() => { order.push('p1'); return true; }, 'preview');
  runner.enqueue(() => { order.push('p2'); return true; }, 'preview');
  runner.enqueue(() => { order.push('b-next'); return true; });
  runner.pump(1000);
  runner.pump(1000);
  assert.deepEqual(order, ['b1', 'b2', 'b3', 'b4', 'p1', 'p2', 'b5', 'b6', 'b-next']);
}

// A zero-timeout fence poll must prevent another batch until the GPU has
// completed the previous one. The final fence is drained even with no jobs.
{
  let status = 1, submitted = 0, deleted = 0, flushed = 0, steps = 0;
  const gl = {
    TIMEOUT_EXPIRED: 1, CONDITION_SATISFIED: 2, WAIT_FAILED: 3, SYNC_GPU_COMMANDS_COMPLETE: 4,
    fenceSync() { submitted++; return {}; },
    flush() { flushed++; },
    clientWaitSync(_f: unknown, flags: number, timeout: number) {
      assert.equal(flags, 0); assert.equal(timeout, 0); return status;
    },
    deleteSync() { deleted++; },
  } as unknown as WebGL2RenderingContext;
  const runner = new JobRunner(gl);
  runner.enqueue(() => ++steps === 5);
  runner.pump(1000);
  runner.pump(1000);
  assert.equal(steps, 4);
  assert.equal(submitted, 1);
  status = 2;
  runner.pump(1000);
  assert.equal(steps, 5);
  assert.equal(runner.queued, 0);
  assert.equal(runner.busy, true);
  runner.pump(1000);
  assert.equal(runner.busy, false);
  assert.equal(deleted, 2);
  assert.equal(flushed, 2);
  // A failed wait is retired, yields once, then allows later work to proceed.
  let laterSteps = 0;
  runner.enqueue(() => { laterSteps++; return false; });
  runner.pump(1000);
  assert.equal(laterSteps, 4);
  status = 3;
  runner.pump(1000);
  assert.equal(laterSteps, 4);
  assert.equal(deleted, 3);
  runner.pump(1000);
  assert.equal(laterSteps, 8, 'a failed fence must not permanently stall the queue');
}

{
  const runner = new JobRunner();
  let called = false;
  runner.enqueue(() => { called = true; return true; });
  runner.pump(0);
  runner.pump(-1);
  assert.equal(called, false);
  runner.pump(1000);
  assert.equal(called, true);
}
console.log('Job runner: bounded batches, asynchronous yields, preview priority and GPU fences passed.');
