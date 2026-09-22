import { defineConfig } from 'vite';

// Relative base so the build works under https://<user>.github.io/MusicVis/
export default defineConfig({
  base: './',
  build: { target: 'es2022' },
  worker: { format: 'es' },
});
