// Cross-origin isolation for static hosting (GitHub Pages cannot send headers): adds
// Cross-Origin-Opener-Policy: same-origin and Cross-Origin-Embedder-Policy (credentialless, or
// require-corp where credentialless is unsupported, i.e. Safari; chosen by the page via ?coep=) to the
// site's own responses, so the page gets SharedArrayBuffer and multi-threaded WASM.
//
// Pass-through only: no Cache API, no precache; navigations are revalidated with the server
// (cache: 'no-cache'), everything else is fetched exactly as the page asked. Cross-origin requests are
// not touched. Registered by the inline script at the top of index.html, scoped to the site's path.
//
// Kill switch: deploy this file with KILL = true (any byte change makes browsers install the new
// version), or delete it (the page's register script then unregisters on the 404).
const KILL = false;
const COEP = new URL(self.location.href).searchParams.get('coep') === 'require-corp' ? 'require-corp' : 'credentialless';
const NULL_BODY = new Set([101, 103, 204, 205, 304]);

self.addEventListener('install', (e) => {
  if (KILL) e.waitUntil(self.registration.unregister());
  self.skipWaiting();
});

self.addEventListener('activate', (e) => {
  e.waitUntil(KILL ? self.registration.unregister() : self.clients.claim());
});

function isolate(res) {
  if (res.status === 0 || res.type === 'opaqueredirect') return res;
  const h = new Headers(res.headers);
  h.set('Cross-Origin-Opener-Policy', 'same-origin');
  h.set('Cross-Origin-Embedder-Policy', COEP);
  if (!h.has('Cross-Origin-Resource-Policy')) h.set('Cross-Origin-Resource-Policy', 'same-origin');
  return new Response(NULL_BODY.has(res.status) ? null : res.body, { status: res.status, statusText: res.statusText, headers: h });
}

self.addEventListener('fetch', (e) => {
  if (KILL) return;
  const req = e.request;
  if (new URL(req.url).origin !== self.location.origin) return; // cross-origin: the browser's own handling
  if (req.cache === 'only-if-cached' && req.mode !== 'same-origin') return;
  if (req.mode === 'navigate' && req.method === 'GET') {
    e.respondWith(
      fetch(req.url, { cache: 'no-cache', credentials: 'same-origin', redirect: 'follow' }).then((res) =>
        // a followed redirect (e.g. /MusicVis -> /MusicVis/) must reach the navigation as a redirect
        res.redirected ? Response.redirect(res.url, 302) : isolate(res),
      ),
    );
    return;
  }
  e.respondWith(fetch(req).then(isolate));
});
