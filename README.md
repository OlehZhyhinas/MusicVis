# MusicVis

A browser music visualizer in the spirit of MilkDrop, built for modern
hardware. Drop in a playlist and it drives a GPU-rendered scene from the
actual structure of the music: its beat, its sections, its instruments, its
key, not just raw volume. Every preset is a genome, so you can vote on what
you see, breed the ones you like, and edit any preset live.

Everything runs locally in the browser. Your songs never leave the machine.

Live at https://olehzhyhinas.github.io/MusicVis/

## Features

- Beat and tempo lock: the visuals stay phase-locked to the song's beat and
  bar, not just reacting to volume spikes.
- Structure awareness: intros, builds, drops, choruses and breakdowns are
  detected and drive preset changes and reactions.
- DSP stem separation: drums, bass, vocals and other each drive their own
  part of the scene.
- Key-to-hue mapping on a circle-of-fifths wheel, so related keys look
  related.
- Evolving presets: each preset is a genome of bodies (shape, placement,
  motion, deformation, material, emission), a palette, a feedback carrier and
  a chain of space warps including fractal-flame variations, wired to the
  music by reaction genes. Like or dislike what you see, let evolve mode breed
  new candidates in the background, or pick two parents yourself.
- Gene editor: change any gene of the playing preset and see it immediately.
- Gene chat (desktop, WebGPU): describe a change in words ("calmer", "more
  blue", "fill the screen") and a language model running on your GPU edits
  the playing preset. Opt-in: the model (Qwen3.5 4B, about 2.4 GB) downloads
  from Hugging Face only when you start the chat, and stays in the browser.
- Live input: a microphone, audio interface or virtual device (for example
  BlackHole) instead of files.
- A real playlist: add many songs at once, shuffle, repeat, skip around, and
  switch tracks while the next one analyzes in the background.

## Using it

Drop one or more audio files onto the page (mp3, wav, flac, m4a/aac, ogg) or
open the playlist and choose files. The first song starts analyzing and plays
once it's ready. The playlist panel also has the Live input button.

The bar at the top right holds the like and dislike buttons, the preset's
score, the Evolve toggle and the preset browser. In the browser you can filter
and sort the population, select two presets to breed or one to mutate, hide
presets, and export or import the whole population as JSON.

Your population, votes and settings are stored in the browser (IndexedDB and
localStorage) for this site only.

## Keyboard shortcuts

| Key | Action |
| --- | --- |
| L / D | Like / dislike this preset |
| E | Evolve mode on / off |
| B | Preset browser |
| N | Next preset |
| G | Go to preset by ID |
| Space | Play / pause |
| &larr; / &rarr; | Seek -5s / +5s |
| Shift + &larr; / &rarr; | Previous / next track |
| F | Fullscreen |
| H | Toggle HUD |
| K | Gene editor (shown with the HUD) |
| M | Mute |
| P | Toggle playlist panel |
| ? | Keyboard shortcuts help |

Add `?preset=G0-E07` to the URL to start on a given preset.

## Development

```
npm install
npm run dev
```

`npm run typecheck` checks types and `npm run build` produces a static build
in `dist/`. Headless tests:

```
node --import ./scripts/analysis-test.hooks.mjs scripts/analysis-test.ts
node --import ./scripts/analysis-test.hooks.mjs scripts/realtime-test.ts
node --import ./scripts/analysis-test.hooks.mjs scripts/v2-test.ts
node --import ./scripts/analysis-test.hooks.mjs scripts/chat-test.ts
```

The gene chat's runtime lives in `public/llm/`: a patched WebLLM 0.2.84 bundle
(tuned decoding on Apple GPUs with WGSL subgroups, plus a prefill that yields
the GPU between passes so the visualizer keeps its frame rate), its worker, a
weight prefetcher and a catalog naming the tuned model lib. Runtime
measurements: `await __geneChatBench()` in the browser console.

## Deploying

A GitHub Actions workflow at `.github/workflows/pages.yml` builds and deploys
the site to GitHub Pages on every push to `main`. Old `/v2/` links redirect to
the site root.
