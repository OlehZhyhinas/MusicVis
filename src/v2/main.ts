// V2 page: the V1 player (playlist, transport, HUD, analysis) driving the
// genome engine, plus voting, evolve mode and the preset browser.

import type { MusicState } from '../types';
import { Player } from '../audio/Player';
import { LiveAnalyser } from '../audio/LiveAnalyser';
import { Playlist, type RepeatMode, type Track } from '../audio/Playlist';
import { installDropzone } from '../ui/dropzone';
import { Transport } from '../ui/transport';
import { PlaylistPanel } from '../ui/playlistPanel';
import { Hud } from '../ui/hud';
import { idleState } from '../ui/idleState';
import { showToast } from '../ui/toast';
import { loadSetting, saveSetting } from '../ui/storage';
import { Engine } from './engine';
import { Screener } from './screen';
import { Store } from './store';
import { Evolution, type ChooseReason } from './evolve';
import { PresetBrowser } from './browser';
import { fitness, type Member } from './population';

const EVOLVE_SECS = 30;
const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;

async function main(): Promise<void> {
  const appRoot = $<HTMLElement>('app');
  const canvas = $<HTMLCanvasElement>('viz-canvas');
  const emptyStateEl = $<HTMLElement>('playlist-empty');
  const fileInput = $<HTMLInputElement>('file-input');
  const dropOverlay = $<HTMLElement>('drop-overlay');
  const transportEl = $<HTMLElement>('transport');
  const hudEl = $<HTMLElement>('hud');
  const statsEl = $<HTMLElement>('v2-stats');
  const playlistPanelEl = $<HTMLElement>('playlist-panel');
  const helpOverlay = $<HTMLElement>('help-overlay');
  const presetLabel = $<HTMLElement>('preset-label');
  const presetGoto = $<HTMLFormElement>('preset-goto');
  const presetGotoInput = $<HTMLInputElement>('preset-goto-input');
  const likeBtn = $<HTMLButtonElement>('v2-like');
  const dislikeBtn = $<HTMLButtonElement>('v2-dislike');
  const scoreEl = $<HTMLElement>('v2-score');
  const evolveBtn = $<HTMLButtonElement>('v2-evolve');
  const statusEl = $<HTMLElement>('v2-status');
  const barEl = $<HTMLElement>('v2-bar');

  let volume = loadSetting<number>('volume', 0.8);
  let hudOn = loadSetting<boolean>('v2.hudOn', false);
  let shuffle = loadSetting<boolean>('shuffle', false);
  let repeat = loadSetting<RepeatMode>('repeat', 'off');
  let evolveOn = loadSetting<boolean>('v2.evolve', false);
  let muted = false;
  hudEl.hidden = statsEl.hidden = !hudOn;

  let eng: Engine;
  try {
    eng = new Engine(canvas);
  } catch (err) {
    console.error(err);
    showToast('This browser cannot run the visuals (WebGL2 required).', 'error', 10000);
    return;
  }
  const store = new Store();
  await store.open();
  const screener = new Screener(eng);
  const evo = new Evolution(store, screener);
  await evo.load();

  function resizeCanvas(): void {
    eng.resize(window.innerWidth, window.innerHeight, window.devicePixelRatio || 1);
  }
  window.addEventListener('resize', resizeCanvas);
  resizeCanvas();

  // ------------------------------------------------------------ presets

  let currentId: string | null = null;
  let evolveTimer = 0;
  let songCx = 0.5;

  function current(): Member | null {
    return currentId ? evo.pop.get(currentId) ?? null : null;
  }

  function play(id: string, secs = 1.5, skipped = false): boolean {
    const m = evo.pop.get(id);
    if (!m) return false;
    evo.startView(m.id, skipped);
    currentId = m.id;
    eng.show(m.genome, secs);
    evolveTimer = 0;
    browser.markCurrent(m.id);
    updateBar();
    showLabel();
    return true;
  }

  function choose(reason: ChooseReason, secs: number, skipped = false): void {
    const m = evo.choose(reason, songCx, currentId);
    if (m) play(m.id, secs, skipped);
  }

  function showLabel(): void {
    const m = current();
    if (!m) return;
    presetLabel.textContent = `${m.id} · ${m.name} · ${m.type} · ${m.energy}`;
    presetLabel.classList.add('show');
    window.clearTimeout(labelTimer);
    labelTimer = window.setTimeout(() => presetLabel.classList.remove('show'), 3500);
  }
  let labelTimer = 0;

  function updateBar(): void {
    const m = current();
    scoreEl.textContent = m ? `${Math.round(fitness(m) * 100)} · ▲${m.likes} ▼${m.dislikes}` : '';
    evolveBtn.classList.toggle('active', evolveOn);
    evolveBtn.setAttribute('aria-pressed', String(evolveOn));
    const b = evo.breeding > 0 || screener.runner.busy;
    statusEl.textContent = evo.breeding > 0 ? 'breeding…' : b ? 'rendering…' : `${evo.pop.size} presets`;
  }

  function vote(like: boolean): void {
    const m = current();
    if (!m) return;
    evo.vote(m.id, like, evo.nicheFor(songCx));
    const btn = like ? likeBtn : dislikeBtn;
    btn.classList.remove('v2-voted');
    void btn.offsetWidth;
    btn.classList.add('v2-voted');
    updateBar();
    if (!like && evolveOn) choose('evolve', 1.2, false);
  }

  likeBtn.addEventListener('click', () => vote(true));
  dislikeBtn.addEventListener('click', () => vote(false));
  evolveBtn.addEventListener('click', () => setEvolve(!evolveOn));
  $('v2-open-browser').addEventListener('click', () => browser.toggle());

  function setEvolve(on: boolean): void {
    evolveOn = on;
    evolveTimer = 0;
    saveSetting('v2.evolve', on);
    updateBar();
    showToast(on ? `Evolve mode: a new candidate every ${EVOLVE_SECS} s. Vote with L / D.` : 'Evolve mode off: presets change on drops, new songs and N.');
  }

  const browser = new PresetBrowser(evo, {
    play: (id) => play(id, 1.2, true),
    currentId: () => currentId,
    toast: (msg, kind) => showToast(msg, kind ?? 'info'),
    // The browser and the playlist share the right side: hide the playlist
    // while the browser is open and bring it back when the browser closes.
    onOpen: () => {
      playlistWasOpen = !playlistPanel.isCollapsed;
      playlistPanel.setCollapsed(true);
    },
    onClose: () => {
      if (playlistWasOpen) playlistPanel.setCollapsed(false);
    },
  });
  let playlistWasOpen = false;
  evo.onChange = () => {
    browser.refresh();
    updateBar();
  };

  // --------------------------------------------------------- player

  const playlist = new Playlist();
  playlist.setShuffle(shuffle);
  playlist.setRepeat(repeat);
  let audioCtx: AudioContext | null = null;
  let player: Player | null = null;
  let liveAnalyser: LiveAnalyser | null = null;
  let sampler: import('../analysis/TimelineSampler').TimelineSampler | null = null;
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
    onModeChange: () => {},
    onNextPreset: () => choose(evolveOn ? 'evolve' : 'next', 1.2, true),
    onParticleCountChange: () => {},
    onFullscreen: () => toggleFullscreen(),
    onHudToggle: () => setHud(!hudOn),
    onPlaylistToggle: () => playlistPanel.toggle(),
    onHelpToggle: () => setHelpVisible(!!helpOverlay.hidden),
  });
  transport.setVolumeUi(volume, muted);
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
        if (next) void playTrack(next);
        else {
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
    if (cur && cur.status === 'analyzing') transport.setTrackLoading(cur.progress);
    else if (cur && (cur.status === 'ready' || cur.status === 'error')) transport.setTrackLoading(null);
    emptyStateEl.hidden = !playlist.isEmpty;
    if (playlist.isEmpty) {
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
    hudEl.hidden = statsEl.hidden = !on;
    saveSetting('v2.hudOn', on);
  }
  function setHelpVisible(show: boolean): void {
    helpOverlay.hidden = !show;
  }
  $('help-close').addEventListener('click', () => setHelpVisible(false));
  helpOverlay.addEventListener('click', (ev) => {
    if (ev.target === helpOverlay) setHelpVisible(false);
  });
  function toggleFullscreen(): void {
    if (!document.fullscreenElement) appRoot.requestFullscreen?.().catch(() => {});
    else document.exitFullscreen?.().catch(() => {});
  }
  function togglePlay(): void {
    if (!player || !songLoaded) return;
    if (player.playing) player.pause();
    else {
      audioCtx?.resume().catch(() => {});
      player.play();
    }
    if ('mediaSession' in navigator) navigator.mediaSession.playbackState = player.playing ? 'playing' : 'paused';
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
    } else void playTrack(t);
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

  async function playTrack(track: Track): Promise<void> {
    const token = ++loadToken;
    try {
      if (!audioCtx) audioCtx = new AudioContext();
      await audioCtx.resume();
      if (!player) {
        player = new Player(audioCtx);
        player.onended = () => goNext();
      }
      if (!liveAnalyser) liveAnalyser = new LiveAnalyser(audioCtx, player.output);
      playlist.selectTrack(track.id);
      songLoaded = false;
      sampler = null;
      if ('mediaSession' in navigator && 'MediaMetadata' in window) {
        navigator.mediaSession.metadata = new MediaMetadata({ title: track.title, artist: 'MusicVis' });
      }
      transport.show();
      const { buffer, result } = await playlist.ensureLoaded(audioCtx, track);
      if (token !== loadToken) return;
      const { TimelineSampler } = await import('../analysis/TimelineSampler');
      if (token !== loadToken) return;
      player.load(buffer);
      sampler = new TimelineSampler(result);
      songCx = result.songComplexity ?? 0.5;
      eng.setSongComplexity(songCx);
      if (!evolveOn) choose('new', 1.5);
      applyVolume();
      transport.setSections(result.sections, result.duration);
      transport.setTrackLoading(null);
      songLoaded = true;
      playlist.evictStaleBuffers();
      playlist.prefetchNext(audioCtx);
      try {
        await audioCtx.resume();
        player.play();
      } catch {
        // autoplay blocked: the play button is visible
      }
    } catch (err) {
      if (token !== loadToken) return;
      console.error(err);
      transport.setTrackLoading(null);
      showToast(`Failed to load "${track.title}": ${err instanceof Error ? err.message : 'could not load this track.'}`, 'error');
    }
  }

  installDropzone(emptyStateEl, fileInput, dropOverlay, {
    onFiles: (files) => {
      const wasEmpty = playlist.isEmpty;
      const added = playlist.addFiles(files);
      if (wasEmpty && added.length > 0) void playTrack(added[0]);
    },
  });

  // ------------------------------------------------------------ go to

  function setGotoVisible(show: boolean): void {
    presetGoto.hidden = !show;
    if (show) {
      presetGotoInput.value = '';
      presetGotoInput.focus();
    } else presetGotoInput.blur();
  }
  function goTo(raw: string): void {
    const id = raw.trim().toUpperCase();
    const direct = evo.pop.get(id) ?? evo.pop.get(`G0-${id}`);
    if (!direct || !play(direct.id, 1.2, true)) showToast(`No preset "${raw}" in the population.`, 'error');
  }
  presetGoto.addEventListener('submit', (ev) => {
    ev.preventDefault();
    const v = presetGotoInput.value;
    setGotoVisible(false);
    if (v.trim()) goTo(v);
  });
  presetGotoInput.addEventListener('keydown', (ev) => {
    if (ev.key === 'Escape') setGotoVisible(false);
  });
  presetGotoInput.addEventListener('blur', () => (presetGoto.hidden = true));

  window.addEventListener('keydown', (ev) => {
    if (ev.target instanceof HTMLInputElement || ev.target instanceof HTMLSelectElement || ev.metaKey || ev.ctrlKey || ev.altKey) return;
    if (ev.key === '?') {
      setHelpVisible(!!helpOverlay.hidden);
      return;
    }
    if (ev.key === 'Escape') {
      setHelpVisible(false);
      if (browser.open) browser.setOpen(false);
      return;
    }
    switch (ev.key) {
      case ' ':
        ev.preventDefault();
        togglePlay();
        break;
      case 'ArrowLeft':
        if (ev.shiftKey) goPrev();
        else if (player) {
          player.seek(Math.max(0, player.currentTime - 5));
          sampler?.reset();
        }
        break;
      case 'ArrowRight':
        if (ev.shiftKey) goNext();
        else if (player) {
          player.seek(Math.min(player.duration, player.currentTime + 5));
          sampler?.reset();
        }
        break;
      case 'f': case 'F': toggleFullscreen(); break;
      case 'h': case 'H': setHud(!hudOn); break;
      case 'p': case 'P': playlistPanel.toggle(); break;
      case 'n': case 'N': choose(evolveOn ? 'evolve' : 'next', 1.2, true); break;
      case 'l': case 'L': vote(true); break;
      case 'd': case 'D': vote(false); break;
      case 'e': case 'E': setEvolve(!evolveOn); break;
      case 'b': case 'B': browser.toggle(); break;
      case 'g': case 'G':
        ev.preventDefault();
        setGotoVisible(true);
        break;
      case 'm': case 'M':
        muted = !muted;
        applyVolume();
        break;
    }
  });

  // -------------------------------------------------------------- start

  const urlPreset = new URLSearchParams(location.search).get('preset');
  if (!(urlPreset && evo.pop.get(urlPreset.toUpperCase()) && play(urlPreset.toUpperCase(), 0))) {
    const first = evo.choose('new', 0.5, null);
    if (first) play(first.id, 0);
  }
  updateBar();
  // Descriptors for novelty scoring, in the background.
  window.setTimeout(() => void evo.describeMissing(24), 4000);

  // Debug / test handle.
  (window as unknown as Record<string, unknown>).musicvisV2 = { eng, evo, screener, play, choose, current };

  const hud = new Hud(hudEl);
  let lastTime = performance.now();
  let lastIdle = 0;
  let fps = 60;
  let statsTimer = 0;

  function frame(now: number): void {
    const dt = Math.min(0.1, Math.max(0, (now - lastTime) / 1000));
    lastTime = now;
    fps += (1 / Math.max(dt, 1e-6) - fps) * 0.05;

    let state: MusicState;
    if (songLoaded && player && sampler && liveAnalyser) {
      state = sampler.sample(player.currentTime, dt, player.playing, liveAnalyser.read(dt));
      transport.updatePlayback(player.currentTime, player.duration, player.playing);
    } else {
      lastIdle += dt;
      state = idleState(lastIdle, dt);
    }

    // Switching policy.
    if (evolveOn) {
      if (state.playing || !songLoaded) evolveTimer += dt;
      if (evolveTimer > EVOLVE_SECS) choose('evolve', 2.5);
    } else if (state.sectionChanged && state.section?.label === 'drop') {
      choose('drop', 0.35);
    }

    try {
      eng.render(state);
    } catch (err) {
      console.error('render failed', err);
    }
    if (screener.runner.busy) screener.runner.pump(eng.stats.frameMs > 18 ? 2 : 5);

    if (hudOn) {
      const m = current();
      hud.update(state, { presetName: m ? `${m.id} · ${m.name}` : '—', fps });
      statsTimer -= dt;
      if (statsTimer <= 0) {
        statsTimer = 0.5;
        const s = eng.stats;
        statsEl.textContent = `frame ${s.frameMs.toFixed(1)} ms · cpu ${s.cpuMs.toFixed(1)} · gpu ${s.gpuMs.toFixed(1)} ms · ${s.width}×${s.height}${s.scale < 1 ? ` (scale ${s.scale.toFixed(2)})` : ''}${m ? ` · ${m.type}` : ''}`;
      }
    }
    requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);

  // The transport bar is revealed with the playlist, like V1; the vote bar stays.
  barEl.hidden = false;
}

main().catch((err) => {
  console.error(err);
  showToast('Failed to start MusicVis V2.', 'error', 10000);
});
