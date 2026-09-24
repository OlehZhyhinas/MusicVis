// The app: the shared player (playlist, transport, HUD, analysis) driving the
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
import { LiveMode } from '../ui/liveMode';
import { Engine } from './engine';
import { Screener } from './screen';
import { Store } from './store';
import { Evolution, type ChooseReason } from './evolve';
import { PresetBrowser } from './browser';
import { fitness, type Member } from './population';
import { GeneEditor } from './geneEditor';
import { hydrateIcons } from '../ui/icons';
import { PresetBar } from '../ui/presetBar';
import { Popovers, renderMenu, type MenuItem } from '../ui/popover';
import { AutoHide } from '../ui/autoHide';
import { applyLayout, computeLayout } from '../ui/layout';

const GITHUB_URL = 'https://github.com/OlehZhyhinas/MusicVis';
const BUG_URL = 'https://github.com/OlehZhyhinas/MusicVis/issues/new';

const EVOLVE_SECS = 30;
const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;

async function main(): Promise<void> {
  hydrateIcons();
  const popovers = new Popovers();
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
  const faintId = $<HTMLElement>('faint-id');
  const moreMenu = $<HTMLElement>('more-menu');
  const volPop = $<HTMLElement>('vol-pop');
  const presetGoto = $<HTMLFormElement>('preset-goto');
  const presetGotoInput = $<HTMLInputElement>('preset-goto-input');

  let volume = loadSetting<number>('volume', 0.8);
  let hudOn = loadSetting<boolean>('v2.hudOn', false);
  let shuffle = loadSetting<boolean>('shuffle', false);
  let repeat = loadSetting<RepeatMode>('repeat', 'off');
  let evolveOn = loadSetting<boolean>('v2.evolve', false);
  let genesOn = loadSetting<boolean>('v2.genesOn', true);
  let muted = false;
  hudEl.hidden = statsEl.hidden = !hudOn;
  appRoot.classList.toggle('hud-on', hudOn);

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
    relayout();
  }
  function relayout(): void {
    applyLayout(appRoot, computeLayout(null));
    popovers.place();
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

  /**
   * Shows a preset. With unsaved gene edits a manual switch asks first (force skips the question);
   * automatic switches never get here while editing (see choose()).
   */
  function play(id: string, secs = 1.5, skipped = false, force = false): boolean {
    const m = evo.pop.get(id);
    if (!m) return false;
    if (!force && editor.dirty) {
      editor.confirmDiscard(() => play(id, secs, skipped, true));
      return true;
    }
    evo.startView(m.id, skipped);
    currentId = m.id;
    eng.show(m.genome, secs);
    evolveTimer = 0;
    browser.markCurrent(m.id);
    editor.load(m);
    updateBar();
    showLabel();
    return true;
  }

  /** manual: the user asked (N, a dislike in evolve mode); automatic switches pause while editing. */
  function choose(reason: ChooseReason, secs: number, skipped = false, manual = false): void {
    if (editor.dirty && !manual) return;
    const m = evo.choose(reason, songCx, currentId);
    if (m) play(m.id, secs, skipped);
  }

  function showLabel(): void {
    const m = current();
    if (!m) return;
    $('pl-id').textContent = m.id;
    $('pl-name').textContent = m.name;
    const tags = $('pl-tags');
    tags.innerHTML = '';
    for (const [t, c] of [[m.type, ''], [m.energy, m.energy === 'energetic' ? 'var(--warn)' : 'var(--ok)']]) {
      const el = document.createElement('span');
      el.className = 'tag';
      if (c) el.style.setProperty('--c', c);
      el.textContent = t;
      tags.append(el);
    }
    presetLabel.dataset.preset = `${m.id} · ${m.name} · ${m.type} · ${m.energy}`;
    faintId.textContent = m.id;
    presetLabel.classList.add('show');
    window.clearTimeout(labelTimer);
    labelTimer = window.setTimeout(() => presetLabel.classList.remove('show'), 3500);
  }
  let labelTimer = 0;

  function updateBar(): void {
    const m = current();
    const b = evo.breeding > 0 || screener.runner.busy;
    const status = editor.dirty ? 'editing · auto-switch paused' : evo.breeding > 0 ? 'breeding…' : b ? 'rendering…' : `${evo.pop.size} presets`;
    presetBar.update(m ? { id: m.id, name: m.name, type: m.type, energy: m.energy, likes: m.likes, dislikes: m.dislikes, score: fitness(m) } : null, status, evolveOn);
  }

  function vote(like: boolean): void {
    const m = current();
    if (!m) return;
    evo.vote(m.id, like, evo.nicheFor(songCx));
    presetBar.voted(like);
    updateBar();
    if (!like && evolveOn) choose('evolve', 1.2, false, true);
  }

  const presetBar = new PresetBar({
    onLike: () => vote(true),
    onDislike: () => vote(false),
    onNext: () => nextPreset(),
    onEvolve: () => setEvolve(!evolveOn),
    onPresets: () => browser.toggle(),
    thumb: (id) => evo.thumb(id),
  });
  function nextPreset(): void {
    choose(evolveOn ? 'evolve' : 'next', 1.2, true, true);
  }

  function setEvolve(on: boolean): void {
    evolveOn = on;
    evolveTimer = 0;
    saveSetting('v2.evolve', on);
    updateBar();
    showToast(on ? `Evolve mode on: a new candidate every ${EVOLVE_SECS} s. Vote with L / D.` : 'Evolve mode off: presets change on drops, new songs and N.');
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
  const editor = new GeneEditor($<HTMLElement>('v2-genes'), {
    eng,
    evo,
    adopt: (m) => {
      // The saved child is the picture on screen: switch to it in place (no crossfade).
      evo.startView(m.id, false);
      currentId = m.id;
      eng.edit(m.genome);
      evolveTimer = 0;
      browser.markCurrent(m.id);
      editor.load(m);
      updateBar();
      showLabel();
    },
    onDirty: () => updateBar(),
  });
  editor.onClose = () => setGenes(false);
  function toggleGenes(): void {
    // Gene editor: shown with the HUD; K with the HUD off opens both.
    if (!hudOn) {
      setHud(true);
      setGenes(true);
    } else setGenes(!(genesOn && hudOn));
  }
  function setGenes(on: boolean): void {
    genesOn = on;
    saveSetting('v2.genesOn', on);
    editor.setShown(hudOn && genesOn);
  }
  editor.setShown(hudOn && genesOn);
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

  const transport = new Transport(transportEl, {
    onPlayPause: () => togglePlay(),
    onPrev: () => goPrev(),
    onNext: () => goNext(),
    onShuffleToggle: () => toggleShuffle(),
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
    onFullscreen: () => toggleFullscreen(),
    onPlaylistToggle: () => playlistPanel.toggle(),
    onMore: (anchor) => openMore(anchor),
    onVolumePopover: (anchor) => popovers.toggle(volPop, anchor),
  });
  function toggleShuffle(): void {
    shuffle = !shuffle;
    playlist.setShuffle(shuffle);
    transport.setShuffleUi(shuffle);
    saveSetting('shuffle', shuffle);
    playlistPanel.render(playlist);
  }
  function moreItems(): MenuItem[] {
    const phone = appRoot.classList.contains('phone');
    const narrow = window.innerWidth < 900;
    return [
      'View',
      { icon: 'hud', label: 'HUD', kbd: 'H', on: () => hudOn, run: () => setHud(!hudOn), keep: true },
      { icon: 'genes', label: 'Gene editor', kbd: 'K', run: () => toggleGenes() },
      { icon: 'list', label: 'Playlist', kbd: 'P', run: () => playlistPanel.toggle() },
      { icon: 'fullscreen', label: 'Fullscreen', kbd: 'F', run: () => toggleFullscreen() },
      'Preset',
      { icon: 'hash', label: 'Go to preset…', kbd: 'G', run: () => setGotoVisible(true) },
      { icon: 'skip', label: 'Next preset', kbd: 'N', run: () => nextPreset() },
      { icon: 'grid', label: 'Preset browser', kbd: 'B', run: () => browser.toggle() },
      ...(phone || narrow
        ? ([
            'Playback',
            ...(phone
              ? [
                  { icon: 'evolve', label: 'Evolve mode', kbd: 'E', on: () => evolveOn, run: () => setEvolve(!evolveOn), keep: true },
                  { icon: 'volume', label: 'Volume', kbd: 'M', run: () => popovers.show(volPop, null) },
                ]
              : []),
            { icon: 'mic', label: 'Live input', run: () => liveMode.setOpen(true) },
          ] as MenuItem[])
        : []),
      'Help',
      { icon: 'search', label: 'All commands…', kbd: '/', run: () => setHelpVisible(true) },
      { icon: 'keyboard', label: 'Keyboard shortcuts', kbd: '?', run: () => setHelpVisible(true) },
      { icon: 'bug', label: 'Report a bug', href: BUG_URL, bug: true },
      { icon: 'github', label: 'MusicVis on GitHub', href: GITHUB_URL },
    ];
  }
  function openMore(anchor: HTMLElement | null): void {
    renderMenu(moreMenu, moreItems(), () => popovers.close());
    moreMenu.dataset.align = 'panel';
    popovers.toggle(moreMenu, anchor);
  }
  // Live input: starting it pauses the file; a track or stop ends it.
  const liveMode = new LiveMode(
    {
      ensureContext: () => {
        if (!audioCtx) audioCtx = new AudioContext();
        return audioCtx;
      },
      onStart: () => {
        loadToken++;
        player?.pause();
        transport.setTrackLoading(null);
        transport.show();
        liveMode.setOpen(false);
        playlistPanel.setCollapsed(true);
      },
      onStop: () => {
        if (playlist.isEmpty) transport.hide();
      },
      onNewSong: (cx) => {
        songCx = cx;
        eng.setSongComplexity(songCx);
        if (!evolveOn) choose('new', 1.5);
      },
    },
    transport,
    [$<HTMLElement>('pl-live'), $<HTMLElement>('pl-live-empty'), $<HTMLElement>('tp-mic')],
  );

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
  playlistPanel.onCollapse = (c) => {
    transport.setPlaylistOpen(!c);
    updateNowPlaying();
  };
  playlistPanel.setCollapsed(!playlist.isEmpty);

  playlist.onChange = () => {
    playlistPanel.render(playlist);
    const cur = playlist.currentTrack;
    if (cur && cur.status === 'analyzing') transport.setTrackLoading(cur.progress);
    else if (cur && (cur.status === 'ready' || cur.status === 'error')) transport.setTrackLoading(null);
    emptyStateEl.hidden = !playlist.isEmpty;
    updateNowPlaying();
    if (playlist.isEmpty && !liveMode.active) {
      transport.hide();
      playlistPanel.setCollapsed(false);
    }
  };

  function updateNowPlaying(): void {
    const all = playlist.all;
    const cur = playlist.currentTrack;
    const i = cur ? all.indexOf(cur) : -1;
    transport.setNowPlaying(cur?.title ?? '', cur ? `Track ${i + 1} of ${all.length}` : `${all.length} tracks`);
    transport.setPlaylistBadge(playlistPanel.isCollapsed && all.length ? all.length : null);
  }
  function applyVolume(): void {
    if (player) player.volume = muted ? 0 : volume;
    transport.setVolumeUi(volume, muted);
  }
  function setHud(on: boolean): void {
    hudOn = on;
    hudEl.hidden = statsEl.hidden = !on;
    appRoot.classList.toggle('hud-on', on);
    saveSetting('v2.hudOn', on);
    editor.setShown(on && genesOn);
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
    if (liveMode.active) {
      liveMode.stop();
      return;
    }
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
    liveMode.stop();
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
    const tgt = ev.target;
    if (tgt instanceof HTMLInputElement || tgt instanceof HTMLSelectElement || tgt instanceof HTMLTextAreaElement || ev.metaKey || ev.ctrlKey || ev.altKey) return;
    // Nothing typed or pressed inside the gene editor (sliders, selects, buttons) triggers a shortcut.
    if (tgt instanceof Element && tgt.closest('#v2-genes')) return;
    if (ev.key === '?') {
      setHelpVisible(!!helpOverlay.hidden);
      return;
    }
    if (ev.key === 'Escape') {
      if (popovers.close()) return;
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
      case 'n': case 'N': nextPreset(); break;
      case 'k': case 'K': toggleGenes(); break;
      case 's': case 'S': toggleShuffle(); break;
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
      default:
        return;
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
  window.setTimeout(() => void evo.describeMissing(40), 4000);

  // Debug / test handle.
  (window as unknown as Record<string, unknown>).musicvisV2 = { eng, evo, screener, play, choose, current, editor };

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
    const liveState = liveMode.sample(dt);
    if (liveState) {
      state = liveState;
    } else if (songLoaded && player && sampler && liveAnalyser) {
      state = sampler.sample(player.currentTime, dt, player.playing, liveAnalyser.read(dt));
      transport.updatePlayback(player.currentTime, player.duration, player.playing);
    } else {
      lastIdle += dt;
      state = idleState(lastIdle, dt);
    }

    // Switching policy.
    // Unsaved gene edits pause automatic switching (drops, evolve rotation, new songs).
    if (evolveOn) {
      if (!editor.dirty && (state.playing || !songLoaded || liveMode.active)) evolveTimer += dt;
      if (evolveTimer > EVOLVE_SECS) choose('evolve', 2.5);
    } else if (!editor.dirty && state.sectionChanged && state.section?.label === 'drop') {
      choose('drop', 0.35);
    }

    try {
      eng.render(state);
    } catch (err) {
      console.error('render failed', err);
    }
    if (screener.runner.busy) screener.runner.pump(eng.stats.frameMs > 18 ? 2 : 5);
    editor.tick(dt);

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

  new AutoHide(appRoot, () => transport.busy || popovers.isOpen() || !helpOverlay.hidden || !presetGoto.hidden);
}

main().catch((err) => {
  console.error(err);
  showToast('Failed to start MusicVis V2.', 'error', 10000);
});
