import type { IVisualizer, MusicState, VisualMode } from './types';
import { Player } from './audio/Player';
import { LiveAnalyser } from './audio/LiveAnalyser';
import { Playlist, type RepeatMode, type Track } from './audio/Playlist';
import { installDropzone } from './ui/dropzone';
import { Transport } from './ui/transport';
import { PlaylistPanel } from './ui/playlistPanel';
import { Hud } from './ui/hud';
import { idleState } from './ui/idleState';
import { showToast } from './ui/toast';
import { loadSetting, saveSetting } from './ui/storage';

async function main(): Promise<void> {
  const appRoot = document.getElementById('app') as HTMLElement;
  const canvas = document.getElementById('viz-canvas') as HTMLCanvasElement;
  const startScreen = document.getElementById('start-screen') as HTMLElement;
  const dropzoneEl = document.getElementById('dropzone') as HTMLElement;
  const fileInput = document.getElementById('file-input') as HTMLInputElement;
  const dropOverlay = document.getElementById('drop-overlay') as HTMLElement;
  const progressWrap = document.getElementById('progress-wrap') as HTMLElement;
  const progressStage = document.getElementById('progress-stage') as HTMLElement;
  const progressFill = document.getElementById('progress-fill') as HTMLElement;
  const playButton = document.getElementById('play-button') as HTMLButtonElement;
  const startError = document.getElementById('start-error') as HTMLElement;
  const transportEl = document.getElementById('transport') as HTMLElement;
  const hudEl = document.getElementById('hud') as HTMLElement;
  const playlistPanelEl = document.getElementById('playlist-panel') as HTMLElement;
  const helpOverlay = document.getElementById('help-overlay') as HTMLElement;
  const helpClose = document.getElementById('help-close') as HTMLButtonElement;

  let particleCount = loadSetting<number>('particleCount', 262144);
  let mode: VisualMode = loadSetting<VisualMode>('mode', 'enhanced');
  let volume = loadSetting<number>('volume', 0.8);
  let hudOn = loadSetting<boolean>('hudOn', false);
  let shuffle = loadSetting<boolean>('shuffle', false);
  let repeat = loadSetting<RepeatMode>('repeat', 'off');
  let muted = false;

  hudEl.hidden = !hudOn;

  // The visualizer is created once, immediately, so the canvas is alive
  // behind the start screen even before any audio graph exists.
  let visualizer: IVisualizer | null = null;
  let audioCtx: AudioContext | null = null;
  try {
    audioCtx = new AudioContext();
    const silentSource = audioCtx.createGain();
    const { Visualizer } = await import('./render/Visualizer');
    visualizer = new Visualizer(
      canvas,
      { context: audioCtx, source: silentSource },
      { particleCount, renderScale: 1 },
    );
    visualizer.setMode(mode);
  } catch (err) {
    console.error(err);
    showToast('This browser cannot run the visuals (WebGL2/float textures required).', 'error', 10000);
  }

  function resizeCanvas(): void {
    const dpr = window.devicePixelRatio || 1;
    const w = window.innerWidth;
    const h = window.innerHeight;
    canvas.width = Math.round(w * dpr);
    canvas.height = Math.round(h * dpr);
    visualizer?.resize(w, h, dpr);
  }
  window.addEventListener('resize', resizeCanvas);
  resizeCanvas();

  const playlist = new Playlist();
  playlist.setShuffle(shuffle);
  playlist.setRepeat(repeat);

  let player: Player | null = null;
  let liveAnalyser: LiveAnalyser | null = null;
  let sampler: import('./analysis/TimelineSampler').TimelineSampler | null = null;
  let songLoaded = false;
  let loadToken = 0;

  const transport = new Transport(transportEl, appRoot, {
    onPlayPause: () => togglePlay(),
    onPrev: () => goPrev(),
    onNext: () => goNext(),
    onShuffleToggle: () => {
      shuffle = !shuffle;
      playlist.setShuffle(shuffle);
      transport.setShuffleUi(shuffle);
      saveSetting('shuffle', shuffle);
      playlistPanel.render(playlist);
    },
    onRepeatCycle: () => {
      repeat = repeat === 'off' ? 'all' : repeat === 'all' ? 'one' : 'off';
      playlist.setRepeat(repeat);
      transport.setRepeatUi(repeat);
      saveSetting('repeat', repeat);
    },
    onSeek: (t) => {
      player?.seek(t);
      sampler?.reset();
    },
    onVolumeChange: (v) => {
      volume = v;
      muted = false;
      applyVolume();
      saveSetting('volume', v);
    },
    onMuteToggle: () => {
      muted = !muted;
      applyVolume();
    },
    onModeChange: (m) => {
      mode = m;
      visualizer?.setMode(m);
      saveSetting('mode', m);
    },
    onNextPreset: () => visualizer?.nextPreset(),
    onParticleCountChange: (count) => {
      particleCount = count;
      visualizer?.setOptions({ particleCount: count });
      saveSetting('particleCount', count);
    },
    onFullscreen: () => toggleFullscreen(),
    onHudToggle: () => setHud(!hudOn),
    onPlaylistToggle: () => playlistPanel.toggle(),
    onHelpToggle: () => setHelpVisible(!!helpOverlay.hidden),
  });
  transport.setVolumeUi(volume, muted);
  transport.setModeUi(mode);
  transport.setParticleCountUi(particleCount);
  transport.setShuffleUi(shuffle);
  transport.setRepeatUi(repeat);

  const playlistPanel = new PlaylistPanel(playlistPanelEl, playlist, {
    onSelect: (id) => {
      const t = playlist.all.find((tr) => tr.id === id);
      if (t) void playTrack(t);
    },
    onRemove: (id) => {
      const wasCurrent = playlist.currentTrack?.id === id;
      playlist.removeTrack(id);
      if (wasCurrent) {
        const next = playlist.currentTrack;
        if (next) {
          void playTrack(next);
        } else {
          player?.pause();
          songLoaded = false;
          sampler = null;
        }
      }
    },
    onClear: () => {
      playlist.clear();
      player?.pause();
      songLoaded = false;
      sampler = null;
    },
    onAdd: () => fileInput.click(),
  });
  playlistPanel.setCollapsed(!playlist.isEmpty);

  playlist.onChange = () => {
    playlistPanel.render(playlist);
    const cur = playlist.currentTrack;
    if (cur && cur.status === 'analyzing') {
      transport.setTrackLoading(cur.progress);
      if (!startScreen.classList.contains('hidden')) {
        progressStage.textContent = 'Analyzing…';
        progressFill.style.width = `${Math.round(cur.progress * 100)}%`;
      }
    } else if (cur && (cur.status === 'ready' || cur.status === 'error')) {
      transport.setTrackLoading(null);
    }
    if (playlist.isEmpty) {
      startScreen.classList.remove('hidden');
      transport.hide();
      playlistPanel.setCollapsed(false);
    }
  };

  function applyVolume(): void {
    if (player) player.volume = muted ? 0 : volume;
    transport.setVolumeUi(volume, muted);
  }

  function setHud(on: boolean): void {
    hudOn = on;
    hudEl.hidden = !on;
    saveSetting('hudOn', on);
  }

  function setHelpVisible(show: boolean): void {
    helpOverlay.hidden = !show;
  }
  helpClose.addEventListener('click', () => setHelpVisible(false));
  helpOverlay.addEventListener('click', (ev) => {
    if (ev.target === helpOverlay) setHelpVisible(false);
  });

  function toggleFullscreen(): void {
    if (!document.fullscreenElement) {
      appRoot.requestFullscreen?.().catch(() => {});
    } else {
      document.exitFullscreen?.().catch(() => {});
    }
  }

  function togglePlay(): void {
    if (!player || !songLoaded) return;
    if (player.playing) {
      player.pause();
      updateMediaSessionPlaybackState();
    } else {
      audioCtx?.resume().catch(() => {});
      player.play();
      updateMediaSessionPlaybackState();
    }
  }

  function goNext(): void {
    const t = playlist.next();
    if (t) void playTrack(t);
  }

  function goPrev(): void {
    const cur = playlist.currentTrack;
    const t = playlist.previous(player?.currentTime ?? 0);
    if (!t) return;
    if (cur && t.id === cur.id) {
      player?.seek(0);
      sampler?.reset();
    } else {
      void playTrack(t);
    }
  }

  function updateMediaSessionPlaybackState(): void {
    if ('mediaSession' in navigator) {
      navigator.mediaSession.playbackState = player?.playing ? 'playing' : 'paused';
    }
  }

  function updateMediaSessionMetadata(track: Track): void {
    if ('mediaSession' in navigator && 'MediaMetadata' in window) {
      navigator.mediaSession.metadata = new MediaMetadata({
        title: track.title,
        artist: 'MusicVis',
      });
    }
  }

  if ('mediaSession' in navigator) {
    navigator.mediaSession.setActionHandler('play', () => {
      if (player && !player.playing) togglePlay();
    });
    navigator.mediaSession.setActionHandler('pause', () => {
      if (player?.playing) togglePlay();
    });
    navigator.mediaSession.setActionHandler('nexttrack', () => goNext());
    navigator.mediaSession.setActionHandler('previoustrack', () => goPrev());
  }

  async function attemptAutoplay(): Promise<void> {
    try {
      await audioCtx!.resume();
      player!.play();
      startScreen.classList.add('hidden');
      playButton.hidden = true;
      updateMediaSessionPlaybackState();
    } catch {
      playButton.hidden = false;
    }
  }

  async function playTrack(track: Track): Promise<void> {
    const token = ++loadToken;
    startError.hidden = true;
    try {
      if (!audioCtx) audioCtx = new AudioContext();
      await audioCtx.resume();

      if (!player) {
        player = new Player(audioCtx);
        player.onended = () => goNext();
      }
      if (!liveAnalyser) {
        liveAnalyser = new LiveAnalyser(audioCtx, player.output);
      }

      playlist.selectTrack(track.id);
      songLoaded = false;
      sampler = null;
      updateMediaSessionMetadata(track);

      // First-ever load shows the big progress bar on the start screen;
      // later track switches show progress via the transport + playlist panel.
      const startScreenVisible = !startScreen.classList.contains('hidden');
      if (startScreenVisible) {
        playButton.hidden = true;
        progressWrap.hidden = false;
        progressFill.style.width = '0%';
        progressStage.textContent = track.status === 'ready' ? 'Ready' : 'Analyzing…';
      }

      const { buffer, result } = await playlist.ensureLoaded(audioCtx, track);
      if (token !== loadToken) return; // superseded by a newer selection

      const { TimelineSampler } = await import('./analysis/TimelineSampler');
      if (token !== loadToken) return;

      player.load(buffer);
      sampler = new TimelineSampler(result);
      applyVolume();

      transport.setSections(result.sections, result.duration);
      transport.setTrackLoading(null);
      transport.show();
      progressWrap.hidden = true;
      songLoaded = true;

      playlist.evictStaleBuffers();
      playlist.prefetchNext(audioCtx);

      await attemptAutoplay();
    } catch (err) {
      if (token !== loadToken) return;
      console.error(err);
      progressWrap.hidden = true;
      transport.setTrackLoading(null);
      const message = err instanceof Error ? err.message : 'Could not load this track.';
      startError.hidden = false;
      startError.textContent = message;
      showToast(`Failed to load "${track.title}": ${message}`, 'error');
    }
  }

  function handleAddedFiles(files: File[]): void {
    const wasEmpty = playlist.isEmpty;
    const added = playlist.addFiles(files);
    if (wasEmpty && added.length > 0) {
      void playTrack(added[0]);
    }
  }

  installDropzone(dropzoneEl, fileInput, dropOverlay, {
    onFiles: (files) => handleAddedFiles(files),
  });

  playButton.addEventListener('click', () => {
    void attemptAutoplay();
  });

  window.addEventListener('keydown', (ev) => {
    if (ev.target instanceof HTMLInputElement || ev.target instanceof HTMLSelectElement) return;
    if (ev.key === '?') {
      setHelpVisible(!!helpOverlay.hidden);
      return;
    }
    if (ev.key === 'Escape') {
      setHelpVisible(false);
      return;
    }
    switch (ev.key) {
      case ' ':
        ev.preventDefault();
        togglePlay();
        break;
      case 'ArrowLeft':
        if (ev.shiftKey) {
          goPrev();
        } else if (player) {
          player.seek(Math.max(0, player.currentTime - 5));
          sampler?.reset();
        }
        break;
      case 'ArrowRight':
        if (ev.shiftKey) {
          goNext();
        } else if (player) {
          player.seek(Math.min(player.duration, player.currentTime + 5));
          sampler?.reset();
        }
        break;
      case 'f':
      case 'F':
        toggleFullscreen();
        break;
      case 'h':
      case 'H':
        setHud(!hudOn);
        break;
      case 'p':
      case 'P':
        playlistPanel.toggle();
        break;
      case 'n':
      case 'N':
        visualizer?.nextPreset();
        break;
      case '1':
        mode = 'enhanced';
        visualizer?.setMode(mode);
        transport.setModeUi(mode);
        saveSetting('mode', mode);
        break;
      case '2':
        mode = 'classic';
        visualizer?.setMode(mode);
        transport.setModeUi(mode);
        saveSetting('mode', mode);
        break;
      case '3':
        mode = 'hybrid';
        visualizer?.setMode(mode);
        transport.setModeUi(mode);
        saveSetting('mode', mode);
        break;
      case 'm':
      case 'M':
        muted = !muted;
        applyVolume();
        break;
    }
  });

  const hud = new Hud(hudEl);
  let lastTime = performance.now();
  let fps = 60;
  let lastIdleTime = 0;

  function frame(now: number): void {
    const dt = Math.min(0.1, Math.max(0, (now - lastTime) / 1000));
    lastTime = now;
    fps = fps + (1 / Math.max(dt, 1e-6) - fps) * 0.05;

    let state: MusicState;
    if (songLoaded && player && sampler && liveAnalyser) {
      const live = liveAnalyser.read(dt);
      state = sampler.sample(player.currentTime, dt, player.playing, live);
      transport.updatePlayback(player.currentTime, player.duration, player.playing);
    } else {
      lastIdleTime += dt;
      state = idleState(lastIdleTime, dt);
    }

    try {
      visualizer?.render(state);
    } catch (err) {
      console.error('Visualizer render failed', err);
    }

    if (hudOn) {
      hud.update(state, { presetName: visualizer?.getPresetName() ?? '—', fps });
    }

    requestAnimationFrame(frame);
  }

  requestAnimationFrame(frame);
}

main().catch((err) => {
  console.error(err);
  showToast('Failed to start MusicVis.', 'error', 10000);
});
