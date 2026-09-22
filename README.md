# MusicVis

A browser music visualizer in the spirit of MilkDrop, built for modern
hardware. Drop in a playlist and it drives a GPU-rendered visual scene from
the actual structure of the music: its beat, its sections, its instruments,
its key, not just raw volume.

Everything runs locally in the browser. Your songs never leave the machine.

## Features

- Beat and tempo lock: the visuals stay phase-locked to the song's beat and
  bar, not just reacting to volume spikes.
- Structure-aware preset switching: intros, builds, drops, choruses and
  breakdowns are detected and drive scene and preset changes.
- DSP stem separation driving per-instrument layers: drums, bass, vocals and
  other each get their own visual response.
- Key-to-hue mapping: the song's key follows a circle-of-fifths hue wheel, so
  related keys look related.
- Fluid simulation, HDR bloom and coupled feedback buffers for classic
  MilkDrop-style visual depth.
- Up to 1,000,000 GPU-driven particles.
- Classic MilkDrop mode via butterchurn, alongside an Enhanced mode and a
  Hybrid blend of both.
- A real playlist: add many songs at once, shuffle, repeat, skip around, and
  switch tracks while the next one analyzes quietly in the background.

## Using it

Drop one or more audio files onto the page (mp3, wav, flac, m4a/aac, ogg) or
click the drop zone to choose files. The first song starts analyzing and
plays once it's ready; add more any time with the playlist panel's Add
button or by dropping more files.

The playlist panel (toggle with the menu button or `P`) shows every track,
its status (queued, analyzing, ready, or error), and its duration. Click a
track to jump to it, remove tracks you don't want, or clear the whole list.

The transport bar at the bottom has previous/play-pause/next, shuffle,
repeat (off/all/one), a seek bar with the song's sections drawn as coloured
segments, volume, visual mode, preset switching, particle count, fullscreen
and a HUD toggle. It fades out after a couple of seconds of no mouse
movement and reappears when you move the mouse.

The HUD (off by default) shows BPM, key, the current section and its
progress, a 4-beat indicator, per-instrument levels, the active preset name
and FPS.

Mac media keys (play/pause/next/previous) work once a track is loaded.

## Keyboard shortcuts

| Key | Action |
| --- | --- |
| Space | Play / pause |
| &larr; / &rarr; | Seek -5s / +5s |
| Shift + &larr; / &rarr; | Previous / next track |
| F | Fullscreen |
| H | Toggle HUD |
| P | Toggle playlist panel |
| N | Next preset |
| 1 / 2 / 3 | Enhanced / Classic MilkDrop / Hybrid mode |
| M | Mute |
| ? | Toggle keyboard shortcuts help |

## Development

```
npm install
npm run dev
```

`npm run build` type-checks and produces a static build in `dist/`.

## Deploying

A GitHub Actions workflow at `.github/workflows/pages.yml` builds and
deploys the site to GitHub Pages on every push to `main`. In the repository
settings, set Pages' source to "GitHub Actions" once, and pushes to `main`
take care of the rest.
