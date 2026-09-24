import { defineConfig } from 'vite';

// Relative base so the build works under https://<user>.github.io/MusicVis/
// The app lives at the root; /v2/ is a redirect kept for old links.
export default defineConfig({
  base: './',
  build: {
    target: 'es2022',
    rolldownOptions: {
      input: {
        main: 'index.html',
        v2: 'v2/index.html',
      },
    },
  },
  worker: { format: 'es' },
});
