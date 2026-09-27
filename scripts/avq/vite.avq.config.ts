// Dev server for the audio-visual quality harness (scripts/avq/). Not used by the app build:
// serves the repo root (so scripts/avq/page.html can import src/ directly) plus two
// local-only endpoints the headless page uses:
//   GET  /avq/file?p=<absolute path>   read a local file (songs)
//   GET  /avq/beatref?song=<abs path>  the song's Beat This! beats and downbeats (the ML agents' shared
//        labels: $MUSICVIS_DATA/labels/beatthis/{test,own,fma}/<file name>.json, MUSICVIS_DATA
//        defaulting to ~/personal/MusicVis-data), 404 when there is none
//   POST /avq/save?p=<relative path>   write the request body under .testdata/avq/
//        (&split=1: body is [u32 count][u32 sizes...][bytes...], p is a printf-style
//         pattern with one %d replaced by first+index for each part, e.g. frames/%06d.jpg)
// Run: npx vite --config scripts/avq/vite.avq.config.ts --port 5231
import { defineConfig, type Plugin } from 'vite';
import { createReadStream, existsSync, mkdirSync, statSync, writeFileSync } from 'node:fs';
import { homedir } from 'node:os';
import { basename, dirname, extname, join, resolve } from 'node:path';

const ROOT = resolve(import.meta.dirname, '../..');
const OUT = join(ROOT, '.testdata/avq');
const DATA = process.env.MUSICVIS_DATA ?? join(homedir(), 'personal/MusicVis-data');

function beatRefPath(song: string): string | null {
  const name = basename(song, extname(song)) + '.json';
  for (const split of ['test', 'own', 'fma']) {
    const p = join(DATA, 'labels/beatthis', split, name);
    if (existsSync(p)) return p;
  }
  return null;
}

function body(req: import('node:http').IncomingMessage): Promise<Buffer> {
  return new Promise((ok, fail) => {
    const parts: Buffer[] = [];
    req.on('data', (c: Buffer) => parts.push(c));
    req.on('end', () => ok(Buffer.concat(parts)));
    req.on('error', fail);
  });
}

function safeOut(rel: string): string {
  const p = resolve(OUT, rel);
  if (!p.startsWith(OUT + '/')) throw new Error('path escapes .testdata/avq: ' + rel);
  return p;
}

const avqIo: Plugin = {
  name: 'avq-io',
  configureServer(server) {
    server.middlewares.use(async (req, res, next) => {
      const url = new URL(req.url ?? '/', 'http://x');
      // Never let Chrome reuse a stale module between harness runs.
      res.setHeader('Cache-Control', 'no-store');
      try {
        if (url.pathname === '/avq/file' && req.method === 'GET') {
          const p = url.searchParams.get('p') ?? '';
          const st = statSync(p);
          res.setHeader('Content-Length', String(st.size));
          res.setHeader('Content-Type', 'application/octet-stream');
          createReadStream(p).pipe(res);
          return;
        }
        if (url.pathname === '/avq/beatref' && req.method === 'GET') {
          const p = beatRefPath(url.searchParams.get('song') ?? '');
          if (!p) {
            res.statusCode = 404;
            res.end('no beat reference');
            return;
          }
          res.setHeader('Content-Type', 'application/json');
          createReadStream(p).pipe(res);
          return;
        }
        if (url.pathname === '/avq/save' && req.method === 'POST') {
          const rel = url.searchParams.get('p') ?? '';
          const buf = await body(req);
          if (url.searchParams.get('split')) {
            const first = Number(url.searchParams.get('first') ?? 0);
            const n = buf.readUInt32LE(0);
            let off = 4 + n * 4;
            for (let i = 0; i < n; i++) {
              const size = buf.readUInt32LE(4 + i * 4);
              const name = rel.replace(/%0?(\d*)d/, (_m, w: string) => String(first + i).padStart(Number(w || 0), '0'));
              const p = safeOut(name);
              mkdirSync(dirname(p), { recursive: true });
              writeFileSync(p, buf.subarray(off, off + size));
              off += size;
            }
          } else {
            const p = safeOut(rel);
            mkdirSync(dirname(p), { recursive: true });
            writeFileSync(p, buf);
          }
          res.statusCode = 200;
          res.end('ok');
          return;
        }
      } catch (err) {
        res.statusCode = 500;
        res.end(String(err));
        return;
      }
      next();
    });
  },
};

export default defineConfig({
  root: ROOT,
  base: '/',
  plugins: [avqIo],
  worker: { format: 'es' },
  server: { strictPort: true, hmr: false, fs: { strict: false } },
  logLevel: 'warn',
});
