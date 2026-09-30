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
- Breeding pen: select presets in **All** and choose **Add to breeding**, then
  use **Breeding** to view the pen or remove selected members. Automatic breeding
  uses only visible pen members. Manual **Breed** and **Mutate** work with any
  selected presets, including presets outside the pen. The pen starts
  empty; offspring join the general pool until you add them yourself. Membership
  survives reloads and population exports/imports.
  **Reset population** removes non-seed presets outside the pen, keeps every pen
  member and existing seed with their votes and settings, and restores missing seeds.
- Composable body families include branching coral/root structures, pleated fabric,
  and articulated chains of joints. Starter presets X46–X51 introduce these shapes
  as parents; they can inherit the existing materials, placements and reactions.
- New capabilities ship with three distinct starter presets. The ribbed spiral shell
  body comes with Porcelain Nautilus (X60), Malachite Tesserae (X61), and Conch
  Lanterns (X62): a single shaded form, a repeating surface, and drifting copies.
  Recoil movement springs back after musical onsets, with selectable drum, note,
  or instrument triggers. Pearl Bounce (X63), Kinetic Iris (X64), and Afterimage
  Etude (X65) show soft translation, angular kicks, and melody-driven light trails.
  Travelling bends reshape bodies and curves in Silk Currents (X66), Tidal
  Filigree (X67), and Violet Undertow (X68). Vocals deepen the ribbons, tension
  tightens the branching waves, and melody register changes the curve wavelength.
  Feather bodies add curved vanes, fine barbs, and surface sheen in Peacock Quill
  (X69), Gilded Plumage (X70), and Kingfisher Brocade (X71).
  Figure-eight weave motion adds phased flight and banking turns in Swallow Waltz
  (X72), Loom of Light (X73), and Porcelain Procession (X74). Swing, held notes,
  and harmonic tension reshape their paths.
  Reference-inspired luminous lilies use independently bending painted petals, shaded veins,
  stamens, foliage, and scrolling stems. Electric Florilegium (X75) is a rainbow
  bouquet; Prismatic Lily (X76) is a close-up; Moonlit Duet (X77) pairs two blooms.
  Vocals and sustained notes open the flowers, melodic pitch changes their curl,
  and note attacks lift their glow. The iridescent material retains the painted colours;
  other materials use a distance field derived from the petal silhouettes.
  Neon Silk Lily (X78), Chromatic Undertow (X79), and Liquid Bloom (X80) add curved
  3D petal surfaces with per-pixel depth, turning highlights, and neon currents
  flowing through their veins. Depth, twist, and colour flow are breedable and reactable.
  Neon Streamer (X81), Prismatic Knot (X82), and Wavelet Silk (X83) use a real 3D
  ribbon mesh with depth testing and a damped spring chain. Edit 2–64 path points
  in Genes → Shape (x, y, z, width, twist) to trace your own shape; paths breed and
  mutate, and musical forces bend the mesh while neon colours travel along it.
  Chromatic Voltage (X84), Kickstorm (X85), and Fractal Crossfire (X86) shoot jagged,
  recursively branching lightning across independent musical channels. Drum hits
  fire the main bolt, note starts select pitch-class colours, and bass, vocal and
  other onsets add separate strikes. Bolts propagate quickly and decay in silence.
- Exploration controls in the preset browser also guide automatic breeding:
  Gentle occasionally chooses a different body family, while Explore and Wild
  do so more often. Rare families get equal chances before fitness chooses an
  individual. Explore/Wild keep rejecting familiar children through every retry,
  so a breeding batch can be smaller when no novel child passes.
- Melodic gestures: `register` follows tracked note height; `rising` and `falling`
  separate upward and downward slides, with vibrato removed. Songbird Mobile (X52)
  and Portamento Loom (X53) connect those gestures to position, articulation and pleats.
  Cadence Conservatory (X54) opens with harmonic tension and resolution; Offbeat
  Letterpress (X55) follows swing and syncopation; Refrain Lantern (X56) travels
  through each detected hook while note height traces its vertical path.
- Gene editor: change any gene of the playing preset and see it immediately.
- Gene chat (desktop, WebGPU): describe a change in words ("calmer", "more
  blue", "fill the screen") and a language model running on your GPU edits
  the playing preset. Opt-in: the model (Qwen3.5 4B, about 2.4 GB) downloads
  from Hugging Face only when you start the chat, and stays in the browser.
- Live input: a microphone, audio interface or virtual device (for example
  BlackHole) instead of files, or another browser tab (YouTube, Spotify) in
  Chrome and Edge. With "Sharper notes" on, the picture trails the sound by
  100 ms and the melody tracker uses that as look-ahead, which brings live
  notes close to an analysed file.
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

Presets change on new songs and after a maximum of 3 minutes of playback,
with or without Evolve mode. Pausing playback, editing unsaved genes, or
comparing presets in a duel pauses the timer. Press N to change presets sooner.

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

`scripts/live-vs-offline.ts` runs real songs (decoded with ffmpeg, never
played) through the live path and scores its notes, legato, beats and
sections against the offline analysis of the same audio.

The gene chat's runtime lives in `public/llm/`: a patched WebLLM 0.2.84 bundle
(tuned decoding on Apple GPUs with WGSL subgroups, plus a prefill that yields
the GPU between passes so the visualizer keeps its frame rate), its worker, a
weight prefetcher and a catalog naming the tuned model lib. Runtime
measurements: `await __geneChatBench()` in the browser console.

## Deploying

A GitHub Actions workflow at `.github/workflows/pages.yml` builds and deploys
the site to GitHub Pages on every push to `main`. Old `/v2/` links redirect to
the site root.
