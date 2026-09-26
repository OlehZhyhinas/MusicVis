// Bundles scripts/live/engineEntry.ts for Node (vite SSR build) into .testdata/live/bundle/engine.mjs
// and imports it. The fake WebGL context lets Signals run without a GPU (texture uploads are no-ops).

import { join } from 'node:path';
import { pathToFileURL } from 'node:url';
import { build } from 'vite';
import { OUT } from './common';

export type EngineBundle = typeof import('./engineEntry');

export async function loadEngine(): Promise<EngineBundle> {
  const outDir = join(OUT, 'bundle');
  await build({
    configFile: false,
    logLevel: 'error',
    root: join(import.meta.dirname, '../..'),
    build: {
      ssr: join(import.meta.dirname, 'engineEntry.ts'), outDir, emptyOutDir: true, target: 'node22', minify: false,
      rolldownOptions: { output: { entryFileNames: 'engine.mjs' } },
    },
  });
  return (await import(pathToFileURL(join(outDir, 'engine.mjs')).href)) as EngineBundle;
}

/** A WebGL2 stand-in: every method is a no-op returning an empty object. */
export const FAKE_GL = new Proxy({}, { get: () => () => ({}) }) as unknown as WebGL2RenderingContext;
