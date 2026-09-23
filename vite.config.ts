import { defineConfig } from 'vite';

// Relative base so the build works under https://<user>.github.io/MusicVis/
// Two pages: V1 at the root, V2 (evolving presets) at /v2/.
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
