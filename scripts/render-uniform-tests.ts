// Run with: node --experimental-transform-types --import ./scripts/analysis-test.hooks.mjs scripts/render-uniform-tests.ts
import assert from 'node:assert/strict';
import { Program, type GL } from '../src/render/gl';

// A recording GL isolates uniform state from driver timing and shader compilers.
const calls: { method: string; args: unknown[] }[] = [];
const gl = new Proxy({ TEXTURE0: 33984, TEXTURE_2D: 3553 }, {
  get(target, name: string) {
    if (name in target) return target[name as keyof typeof target];
    if (name === 'getUniformLocation') return (program: unknown, uniform: string) => uniform === 'missing' ? null : { program, uniform };
    return (...args: unknown[]) => calls.push({ method: name, args: args.map(a => ArrayBuffer.isView(a) ? Array.from(a as Float32Array) : a) });
  },
}) as unknown as GL;
const program = () => new Program(gl, '', '', 'test', {} as WebGLProgram);
const count = (method: string) => calls.filter(c => c.method === method).length;

// All scalar widths, plus vector convenience uploads, preserve changes while
// skipping identical values across use() calls. Switching programs is independent.
for (const [method, values] of [
  ['f1', [1]], ['f2', [1, 2]], ['f3', [1, 2, 3]], ['f4', [1, 2, 3, 4]], ['i1', [1]],
] as const) {
  const p = program(), q = program();
  const upload = (p: Program, v: number[]) => (p[method] as (...a: unknown[]) => Program).call(p, 'value', ...v);
  calls.length = 0;
  upload(p.use(), [...values]);
  upload(p.use(), [...values]);
  upload(q.use(), [...values]);
  upload(p.use(), [...values]);
  upload(p, values.map((v, i) => i === values.length - 1 ? v + 1 : v));
  assert.equal(calls.filter(c => c.method.startsWith('uniform')).length, 3, method);
}
{
  const p = program();
  calls.length = 0;
  p.f3('color', 1, 2, 3).v3('color', [1, 2, 3]).v3('color', [1, 2, 4]);
  assert.equal(count('uniform3f'), 2);
  p.f1('zero', 0).f1('zero', -0);
  assert.equal(count('uniform1f'), 2, 'signed zero is preserved');
}

// Exposed locations may be retained and written later. Never trust cached values
// for them, even after a helper upload has happened in between raw writes.
{
  const p = program();
  p.f1('value', 1);
  const l = p.loc('value');
  p.f1('value', 1);
  gl.uniform1f(l, 9);
  calls.length = 0;
  p.f1('value', 1);
  assert.equal(count('uniform1f'), 1);
}

// Bulk arrays can change in place and overlap scalar writes to array elements.
{
  const p = program();
  const a = new Float32Array([1, 2, 3, 4]);
  calls.length = 0;
  p.f4v('data', a);
  a[3] = 8;
  p.f4v('data', a);
  assert.deepEqual(calls.filter(c => c.method === 'uniform4fv').map(c => c.args[1]), [[1,2,3,4], [1,2,3,8]]);
  p.f1('weights', 1).f1v('weights[0]', a);
  calls.length = 0;
  p.f1('weights', 1);
  assert.equal(count('uniform1f'), 1, 'bulk upload invalidates the first-element alias');
  p.f1('weights[2]', 3).f1v('weights', a);
  calls.length = 0;
  p.f1('weights[2]', 3);
  assert.equal(count('uniform1f'), 1, 'indexed values cannot stay stale after a bulk upload');
}

// Array base and [0] are the same uniform, including alternating scalar writes.
{
  const p = program();
  calls.length = 0;
  p.f1('weights', 1).f1('weights[0]', 2).f1('weights', 1);
  assert.equal(count('uniform1f'), 3);
}

// Texture objects must always bind, even if sampler assignments are unchanged.
// Reordering samplers or setting them through i1 must update those assignments.
{
  const p = program();
  const a = {} as WebGLTexture, b = {} as WebGLTexture;
  calls.length = 0;
  p.use().tex('a', a).tex('missing', b).tex('b', b);
  p.use().tex('a', b).tex('b', a);
  assert.equal(count('bindTexture'), 4);
  assert.equal(count('uniform1i'), 2);
  p.use().tex('b', a).tex('a', b);
  assert.deepEqual(calls.filter(c => c.method === 'uniform1i').map(c => c.args[1]), [0,1,0,1]);
  p.i1('a', 4).use().tex('a', a);
  assert.equal(calls.filter(c => c.method === 'uniform1i').at(-1)?.args[1], 0);
}
console.log('PASS: scalar caching, shared programs, raw writes, array mutation/aliases and sampler reassignment');
