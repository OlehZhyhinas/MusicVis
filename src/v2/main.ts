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
import { Phenotype } from './phenotype';
import { Fingerprinter } from './fingerprintRender';
import { ExploreControls, loadExploreMode } from './exploreUi';
import { EXPLORE_LABEL } from './novelty';
import { PresetMap, ViewSwitch, loadPresetView, type PresetView } from './mapView';
import { SimilarityPage } from './similarityUi';
import { DuelPage } from './duelUi';
import { Embedder } from './embedding';
import { fitness, type Member } from './population';
import { GeneEditor } from './geneEditor';
import { ChatPane } from '../chat/chatPane';
import { LookSampler } from '../chat/look';
import { hydrateIcons } from '../ui/icons';
import { PresetBar } from '../ui/presetBar';
import { Popovers, renderMenu, type MenuItem } from '../ui/popover';
import { AutoHide } from '../ui/autoHide';
import { applyLayout, computeLayout } from '../ui/layout';
import { Dock } from '../ui/dock';
import { Palette, type Command } from '../ui/palette';
import { LyricsLibrary, lyricStatusLabel } from '../lyrics/library';
import { LyricSampler } from '../lyrics/sampler';
import { LyricNudges, NUDGE_STEP, formatNudge, nudgeKey } from '../lyrics/nudge';
import { LyricOverlay } from '../lyrics/overlay';
import '../lyrics/lyrics.css';

const GITHUB_URL = 'https://github.com/OlehZhyhinas/MusicVis';
const BUG_URL = 'https://github.com/OlehZhyhinas/MusicVis/issues/new';

const EVOLVE_SECS = 30;
const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;

async function main(): Promise<void> {
  hydrateIcons();
  const popovers = new Popovers();
  const dock = new Dock();
  const appRoot = $<HTMLElement>('app');
  const canvas = $<HTMLCanvasElement>('viz-canvas');
  const emptyStateEl = $<HTMLElement>('playlist-empty');
  const fileInput = $<HTMLInputElement>('file-input');
  const dropOverlay = $<HTMLElement>('drop-overlay');
  const transportEl = $<HTMLElement>('transport');
  const hudEl = $<HTMLElement>('hud');
  const playlistPanelEl = $<HTMLElement>('tab-playlist');
  const firstRun = $<HTMLElement>('first-run');
  const presetLabel = $<HTMLElement>('preset-label');
  const faintId = $<HTMLElement>('faint-id');
  const moreMenu = $<HTMLElement>('more-menu');
  const volPop = $<HTMLElement>('vol-pop');

  let volume = loadSetting<number>('volume', 0.8);
  let hudOn = loadSetting<boolean>('v2.hudOn', false);
  let shuffle = loadSetting<boolean>('shuffle', false);
  let repeat = loadSetting<RepeatMode>('repeat', 'off');
  let evolveOn = loadSetting<boolean>('v2.evolve', false);
  let muted = false;
  hudEl.hidden = !hudOn;
  appRoot.classList.toggle('hud-on', hudOn);

  let eng: Engine;
  try {
    eng = new Engine(canvas);
  } catch (err) {
    console.error(err);
    showStartupError(true);
    showToast('Failed to start MusicVis', 'error', 0, 'WebGL2 is required.');
    return;
  }
  const store = new Store();
  await store.open();
  const screener = new Screener(eng);
  const evo = new Evolution(store, screener);
  await evo.load();
  // Phenotype fingerprints: visual duplicate rejection in breeding, computed for old members in the background.
  const pheno = new Phenotype(new Fingerprinter(eng, screener.runner), () => evo.pop);
  evo.pheno = pheno;
  pheno.onFingerprint = () => evo.changed();
  pheno.mode = loadExploreMode();
  pheno.embedder = new Embedder();
  await pheno.attachStore(store);

  function resizeCanvas(): void {
    eng.resize(window.innerWidth, window.innerHeight, window.devicePixelRatio || 1);
    relayout();
  }
  function relayout(): void {
    applyLayout(appRoot, computeLayout(dock.tab));
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
    presetMap.markCurrent();
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
    onPresets: () => dock.toggle('presets'),
    thumb: (id) => evo.thumb(id),
  });
  function nextPreset(): void {
    choose('next', 1.2, true, true);
  }
  const nextSel = $<HTMLSelectElement>('v2b-next');
  evo.nextOrder = loadSetting<'random' | 'newest'>('v2.nextOrder', 'random') === 'newest' ? 'newest' : 'random';
  nextSel.value = evo.nextOrder;
  nextSel.addEventListener('change', () => {
    evo.nextOrder = nextSel.value === 'newest' ? 'newest' : 'random';
    saveSetting('v2.nextOrder', evo.nextOrder);
  });

  function setEvolve(on: boolean): void {
    evolveOn = on;
    evolveTimer = 0;
    saveSetting('v2.evolve', on);
    updateBar();
    showToast(on ? 'Evolve mode on' : 'Evolve mode off', 'evolve', 5000, on ? `A new candidate every ${EVOLVE_SECS} s. Vote with L / D.` : 'Presets change on drops, new songs and N.');
  }

  const browser = new PresetBrowser(evo, {
    play: (id) => play(id, 1.2, true),
    currentId: () => currentId,
    toast: (msg, kind, detail) => showToast(msg, kind ?? 'info', 5000, detail),
    // The browser and the playlist share the right side: hide the playlist
    // while the browser is open and bring it back when the browser closes.
    onClose: () => {
      if (dock.tab === 'presets') dock.close();
    },
    novelty: (m) => pheno.novelty(m),
  });
  const explore = new ExploreControls(pheno.mode, (m) => {
    pheno.mode = m;
    browser.refresh();
    showToast(`Exploration: ${EXPLORE_LABEL[m]}`, 'evolve', 4000, ExploreControls.hint(m));
  });
  // Map view: a spring graph of looks, in place of the list.
  const presetMap = new PresetMap({
    members: () => browser.members(),
    currentId: () => currentId,
    distance: (a, b) => pheno.memberDistance(a, b),
    novelty: (m) => pheno.novelty(m),
    thumb: (id) => evo.thumb(id),
    metricVersion: () => pheno.metricVersion,
    open: (id) => play(id, 1.2, true),
  });
  $('v2b-list').before(presetMap.host);
  const presetView = new ViewSwitch(explore.tools, loadPresetView(), (v) => applyPresetView(v));
  // Similarity page: teach the look metric which presets look alike.
  const similarity = new SimilarityPage({
    pheno,
    fper: pheno.fper!,
    members: () => evo.pop.list(),
    thumb: (id) => evo.thumb(id),
    toast: (msg, detail) => showToast(msg, 'info', 5000, detail),
    setEmbedding: (on) => setEmbedding(on),
  });
  async function setEmbedding(on: boolean): Promise<boolean> {
    const ok = await pheno.setEmbedding(on);
    saveSetting('v2.embedding', on && ok);
    return ok;
  }
  const simBtn = document.createElement('button');
  simBtn.className = 'btn sm';
  simBtn.title = 'Which looks more like this one? Teach the look metric';
  simBtn.textContent = 'Similarity…';
  simBtn.addEventListener('click', () => similarity.open());
  explore.tools.append(simBtn);
  // Clip duels: which of two presets feels more in sync with the same song moment (trains the AV judge).
  const duels = new DuelPage({
    eng,
    members: () => evo.pop.list(),
    song: () => {
      const cur = playlist.currentTrack;
      if (!songLoaded || !player || !songResult || !cur || liveMode.active) return null;
      const p = player;
      return { id: cur.id, result: songResult, seek: (t) => p.seek(t), play: () => p.play(), pause: () => p.pause(), time: () => p.currentTime, playing: () => p.playing };
    },
    toast: (msg, detail) => showToast(msg, 'info', 5000, detail),
  });
  const duelBtn = document.createElement('button');
  duelBtn.className = 'btn sm';
  duelBtn.title = 'Which of two clips feels more in sync? Train the AV judge';
  duelBtn.textContent = 'Duels…';
  duelBtn.addEventListener('click', () => void duels.open());
  explore.tools.append(duelBtn);
  function applyPresetView(v: PresetView): void {
    $('v2b-list').hidden = v === 'map';
    presetMap.setShown(v === 'map' && dock.tab === 'presets');
  }
  // The browser's filters (type, energy, show hidden) apply to the map too.
  $('v2-browser').querySelector('.filters')!.addEventListener('click', () => setTimeout(() => presetMap.sync(), 0));
  $('v2-browser').querySelector('.filters')!.addEventListener('change', () => setTimeout(() => presetMap.sync(), 0));
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
  // Gene chat under the gene editor: a local model edits the playing preset from a description.
  const look = new LookSampler();
  let keyHueNow = 0;
  const chat = new ChatPane($<HTMLElement>('gene-chat'), { editor, look, keyHue: () => keyHueNow });
  Object.assign(window, {
    __geneChat: chat,
    // Gene chat test set against the loaded model: await __geneChatTest() or __geneChatTest(['calmer']).
    __geneChatTest: (ids?: string[]) => import('../chat/testRunner').then((m) => m.runChatTests(chat.model, ids)),
  });
  editor.onClose = () => dock.close();
  editor.onAttention = () => dock.open('genes');
  function toggleGenes(): void {
    // K opens the Genes tab and the HUD; K again closes the tab.
    if (dock.tab === 'genes') dock.close();
    else {
      dock.open('genes');
      if (!hudOn) setHud(true);
    }
  }
  evo.onChange = () => {
    browser.refresh();
    presetMap.sync();
    updateBar();
  };

  // --------------------------------------------------------- player

  const playlist = new Playlist();
  // Lyrics: looked up automatically for every added song (LRCLIB), shown in the playlist rows.
  const lyrics = new LyricsLibrary();
  lyrics.onChange = (id) => {
    playlistPanel.render(playlist);
    // Lyrics found while the song already plays: they join in from here.
    if (songLoaded && playlist.currentTrack?.id === id) attachLyrics();
  };
  playlist.setShuffle(shuffle);
  playlist.setRepeat(repeat);
  let audioCtx: AudioContext | null = null;
  let player: Player | null = null;
  let liveAnalyser: LiveAnalyser | null = null;
  let sampler: import('../analysis/TimelineSampler').TimelineSampler | null = null;
  /** The playing song's lyrics (null: none found, not looked up yet, or live input). */
  let lyricSampler: LyricSampler | null = null;
  let songResult: import('../types').AnalysisResult | null = null;
  function lyricSourceText(): string {
    if (liveMode.active) return 'live input';
    const cur = playlist.currentTrack;
    const l = cur ? lyrics.get(cur.id) : undefined;
    if (!l) return '–';
    const label = lyricStatusLabel(l.status)?.label ?? l.status;
    let out = l.result?.source ? `${label} · ${l.result.source}` : label;
    // The matched version's length against the file's, and how the lines were fitted to the audio.
    if (l.result?.duration && songResult) out += ` · ${Math.round(l.result.duration)}/${Math.round(songResult.duration)} s`;
    const al = lyricSampler?.track.align;
    if (al) {
      const sh = `${al.offset >= 0 ? '+' : ''}${al.offset.toFixed(2)} s${al.scale !== 1 ? ` ×${al.scale.toFixed(3)}` : ''}`;
      out += al.applied ? ` · auto ${sh} (${Math.round(al.confidence * 100)}%)` : ` · as timed (fit ${sh} ${Math.round(al.confidence * 100)}%)`;
    }
    if (lyricSampler?.offset) out += ` · nudge ${formatNudge(lyricSampler.offset)}`;
    return out;
  }
  // Manual lyric timing ([ and ]), remembered per song.
  const nudges = new LyricNudges();
  function lyricNudgeKey(): string | null {
    const cur = playlist.currentTrack;
    return cur && songResult ? nudgeKey(lyrics.get(cur.id)?.meta, songResult.duration) : null;
  }
  function attachLyrics(): void {
    const cur = playlist.currentTrack;
    const track = cur && songResult ? lyrics.lyricTrack(cur.id, songResult) : null;
    if (track === lyricSampler?.track) return;
    lyricSampler = track ? new LyricSampler(track) : null;
    if (lyricSampler) lyricSampler.offset = nudges.get(lyricNudgeKey());
  }
  /** Moves the playing song's lyrics by `d` seconds (null: back to the automatic timing). */
  function nudgeLyrics(d: number | null): void {
    if (!lyricSampler || liveMode.active) {
      showToast('No lyrics playing to nudge', 'info', 2500, undefined, 'lyr-nudge');
      return;
    }
    const v = nudges.set(lyricNudgeKey(), d === null ? 0 : lyricSampler.offset + d);
    lyricSampler.offset = v;
    lyricSampler.reset();
    const al = lyricSampler.track.align;
    const auto = al?.applied ? `on top of the automatic ${formatNudge(al.offset)}` : 'lyrics as timed';
    showToast(`Lyrics ${v === 0 ? 'in time as found' : formatNudge(v)}`, 'info', 2500, `${auto} · [ earlier, ] later`, 'lyr-nudge');
  }
  let songLoaded = false;
  let loadToken = 0;

  const transport = new Transport(transportEl, {
    onPlayPause: () => togglePlay(),
    onPrev: () => goPrev(),
    onNext: () => goNext(),
    onShuffleToggle: () => toggleShuffle(),
    onRepeatCycle: () => cycleRepeat(),
    onSeek: (t) => {
      player?.seek(t);
      sampler?.reset();
      lyricSampler?.reset();
    },
    onVolumeChange: (v) => {
      volume = v;
      muted = false;
      applyVolume();
      saveSetting('volume', v);
    },
    onMuteToggle: () => toggleMute(),
    onFullscreen: () => toggleFullscreen(),
    onPlaylistToggle: () => dock.toggle('playlist'),
    onMore: (anchor) => openMore(anchor),
    onVolumePopover: (anchor) => popovers.toggle(volPop, anchor),
  });
  function toggleMute(): void {
    muted = !muted;
    applyVolume();
  }
  function cycleRepeat(): void {
    repeat = repeat === 'off' ? 'all' : repeat === 'all' ? 'one' : 'off';
    playlist.setRepeat(repeat);
    transport.setRepeatUi(repeat);
    saveSetting('repeat', repeat);
  }
  function clearPlaylist(): void {
    playlist.clear();
    lyrics.clear();
    player?.pause();
    songLoaded = false;
    sampler = null;
    lyricSampler = null;
  }
  function toggleShuffle(): void {
    shuffle = !shuffle;
    playlist.setShuffle(shuffle);
    transport.setShuffleUi(shuffle);
    saveSetting('shuffle', shuffle);
    playlistPanel.render(playlist);
  }
  function moreItems(): MenuItem[] {
    const phone = appRoot.classList.contains('phone');
    const narrow = appRoot.classList.contains('narrow');
    return [
      'View',
      { icon: 'hud', label: 'HUD', kbd: 'H', on: () => hudOn, run: () => setHud(!hudOn), keep: true },
      { icon: 'genes', label: 'Gene editor', kbd: 'K', run: () => toggleGenes() },
      { icon: 'list', label: 'Playlist', kbd: 'P', run: () => dock.toggle('playlist') },
      { icon: 'fullscreen', label: 'Fullscreen', kbd: 'F', run: () => toggleFullscreen() },
      'Preset',
      { icon: 'hash', label: 'Go to preset…', kbd: 'G', run: () => palette.open('goto') },
      { icon: 'skip', label: 'Next preset', kbd: 'N', run: () => nextPreset() },
      { icon: 'grid', label: 'Preset browser', kbd: 'B', run: () => dock.toggle('presets') },
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
      { icon: 'search', label: 'All commands…', kbd: '/', run: () => palette.open('cmd') },
      { icon: 'keyboard', label: 'Keyboard shortcuts', kbd: '?', run: () => palette.open('help') },
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
        if (dock.tab === 'playlist') dock.close();
        updateEmpty();
      },
      onStop: () => {
        if (playlist.isEmpty) transport.hide();
        updateEmpty();
      },
      onNewSong: (cx) => {
        songCx = cx;
        eng.setSongComplexity(songCx);
        if (!evolveOn) choose('new', 1.5);
      },
    },
    transport,
    [$<HTMLElement>('tp-mic'), $<HTMLElement>('pl-live'), $<HTMLElement>('pl-live-empty')],
    popovers,
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
      lyrics.remove(id);
      if (wasCurrent) {
        const next = playlist.currentTrack;
        if (next) void playTrack(next);
        else {
          player?.pause();
          songLoaded = false;
          sampler = null;
          lyricSampler = null;
        }
      }
    },
    onClear: () => clearPlaylist(),
    onAdd: () => fileInput.click(),
    lyricStatus: (id) => lyrics.status(id),
  });
  let wasEmpty = playlist.isEmpty;
  function updateEmpty(): void {
    const empty = playlist.isEmpty && !liveMode.active;
    firstRun.hidden = !empty;
    appRoot.classList.toggle('is-empty', empty);
  }
  dock.onChange = (tab) => {
    relayout();
    browser.setOpen(tab === 'presets');
    applyPresetView(presetView.view);
    editor.setShown(tab === 'genes');
    chat.setShown(tab === 'genes');
    transport.setPlaylistOpen(tab === 'playlist');
    presetBar.setPresetsOpen(tab === 'presets');
    updateNowPlaying();
  };
  $('fr-add').addEventListener('click', () => fileInput.click());
  $('fr-drop').addEventListener('click', () => fileInput.click());
  $('et-help').addEventListener('click', () => palette.open('help'));
  $('et-fullscreen').addEventListener('click', () => toggleFullscreen());

  playlist.onChange = () => {
    playlistPanel.render(playlist);
    const cur = playlist.currentTrack;
    if (cur && cur.status === 'analyzing') transport.setTrackLoading(cur.progress);
    else if (cur && (cur.status === 'ready' || cur.status === 'error')) transport.setTrackLoading(null);
    updateNowPlaying();
    if (playlist.isEmpty && !liveMode.active) {
      transport.hide();
      // Emptied: the dock closes and the first-run card returns.
      if (!wasEmpty && dock.tab === 'playlist') dock.close();
    }
    wasEmpty = playlist.isEmpty;
    updateEmpty();
  };

  function updateNowPlaying(): void {
    const all = playlist.all;
    const cur = playlist.currentTrack;
    const i = cur ? all.indexOf(cur) : -1;
    transport.setNowPlaying(cur?.title ?? '', cur ? `Track ${i + 1} of ${all.length}` : `${all.length} tracks`);
    transport.setPlaylistBadge(dock.tab !== 'playlist' && all.length ? all.length : null);
  }
  function applyVolume(): void {
    if (player) player.volume = muted ? 0 : volume;
    transport.setVolumeUi(volume, muted);
  }
  function setHud(on: boolean): void {
    hudOn = on;
    hudEl.hidden = !on;
    appRoot.classList.toggle('hud-on', on);
    saveSetting('v2.hudOn', on);
  }
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
      lyricSampler?.reset();
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
      lyrics.prioritize(track.id);
      songLoaded = false;
      sampler = null;
      lyricSampler = null;
      if ('mediaSession' in navigator && 'MediaMetadata' in window) {
        navigator.mediaSession.metadata = new MediaMetadata({ title: track.title, artist: 'MusicVis' });
      }
      transport.show();
      const { buffer, result } = await playlist.ensureLoaded(audioCtx, track);
      if (token !== loadToken) return;
      // The real duration picks the version of the song the synced lyrics are timed for.
      lyrics.setDuration(track.id, buffer.duration);
      const { TimelineSampler } = await import('../analysis/TimelineSampler');
      if (token !== loadToken) return;
      player.load(buffer);
      sampler = new TimelineSampler(result);
      eng.setSong(result.sections);
      songResult = result;
      attachLyrics();
      songCx = result.songComplexity ?? 0.5;
      eng.setSongComplexity(songCx);
      eng.setSongWorld(result);
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
      showToast(`Failed to load “${track.title}”`, 'error', 0, err instanceof Error ? err.message : 'Could not load this track.');
    }
  }

  installDropzone(emptyStateEl, fileInput, dropOverlay, {
    onFiles: (files) => {
      const wasEmpty = playlist.isEmpty;
      const added = playlist.addFiles(files);
      for (const t of added) lyrics.add(t.id, t.file);
      if (wasEmpty && added.length > 0) void playTrack(added[0]);
    },
  });

  // ------------------------------------------------ commands + go to

  function findPreset(raw: string): Member | undefined {
    const id = raw.trim().toUpperCase();
    return evo.pop.get(id) ?? evo.pop.get(`G0-${id}`);
  }
  function seekBy(d: number): void {
    if (!player || liveMode.active) return;
    player.seek(Math.max(0, Math.min(player.duration, player.currentTime + d)));
    sampler?.reset();
    lyricSampler?.reset();
  }
  function commands(): Command[] {
    return [
      { group: 'Playback', icon: 'play', label: 'Play / pause', keys: ['Space'], run: () => togglePlay() },
      { group: 'Playback', icon: 'next', label: 'Next track', keys: ['Shift', '→'], run: () => goNext() },
      { group: 'Playback', icon: 'prev', label: 'Previous track', keys: ['Shift', '←'], run: () => goPrev() },
      { group: 'Playback', icon: 'skip', label: 'Seek 5 s forward', keys: ['→'], run: () => seekBy(5) },
      { group: 'Playback', icon: 'undo', label: 'Seek 5 s back', keys: ['←'], run: () => seekBy(-5) },
      { group: 'Playback', icon: muted ? 'volume' : 'mute', label: muted ? 'Unmute' : 'Mute', keys: ['M'], run: () => toggleMute() },
      { group: 'Playback', icon: 'shuffle', label: `Shuffle ${shuffle ? 'off' : 'on'}`, keys: ['S'], run: () => toggleShuffle() },
      { group: 'Playback', icon: repeat === 'one' ? 'repeat1' : 'repeat', label: `Repeat: ${repeat} → ${repeat === 'off' ? 'all' : repeat === 'all' ? 'one' : 'off'}`, run: () => cycleRepeat() },
      { group: 'Playback', icon: 'fullscreen', label: 'Fullscreen', keys: ['F'], run: () => toggleFullscreen() },
      { group: 'Playback', icon: 'undo', label: `Lyrics ${NUDGE_STEP} s earlier`, keys: ['['], run: () => nudgeLyrics(-NUDGE_STEP) },
      { group: 'Playback', icon: 'skip', label: `Lyrics ${NUDGE_STEP} s later`, keys: [']'], run: () => nudgeLyrics(NUDGE_STEP) },
      { group: 'Playback', icon: 'reset', label: `Reset lyric timing${lyricSampler?.offset ? ` (now ${formatNudge(lyricSampler.offset)})` : ''}`, run: () => nudgeLyrics(null) },
      { group: 'Presets', icon: 'up', label: 'Like this preset', keys: ['L'], run: () => vote(true) },
      { group: 'Presets', icon: 'down', label: 'Dislike this preset', keys: ['D'], run: () => vote(false) },
      { group: 'Presets', icon: 'skip', label: 'Next preset', keys: ['N'], run: () => nextPreset() },
      { group: 'Presets', icon: 'hash', label: 'Go to preset by ID…', keys: ['G'], run: () => palette.open('goto'), stay: true },
      { group: 'Presets', icon: 'evolve', label: `Evolve mode ${evolveOn ? 'off' : 'on'}`, keys: ['E'], run: () => setEvolve(!evolveOn) },
      { group: 'Presets', icon: 'grid', label: 'Preset browser (breed, mutate, export)', keys: ['B'], run: () => dock.open('presets') },
      { group: 'Presets', icon: 'evolve', label: `Exploration: ${EXPLORE_LABEL[explore.value]} → next mode`, run: () => explore.cycle() },
      { group: 'Presets', icon: 'sparkle', label: 'Similarity judgements (teach the look metric)…', run: () => similarity.open() },
      { group: 'Presets', icon: 'sparkle', label: 'Clip duels (which feels more in sync)…', run: () => void duels.open() },
      { group: 'Presets', icon: 'cpu', label: `Perceptual embedding (DINOv2) ${pheno.embOn ? 'off' : 'on'}`, run: () => void setEmbedding(!pheno.embOn).then((ok) => showToast(ok ? `Perceptual embedding ${pheno.embOn ? 'on' : 'off'}` : 'Perceptual embedding could not load', ok ? 'info' : 'error', 5000, ok ? undefined : pheno.embedder?.status.detail)) },
      { group: 'Panels', icon: 'list', label: 'Playlist', keys: ['P'], run: () => dock.open('playlist') },
      { group: 'Panels', icon: 'plus', label: 'Add songs…', run: () => fileInput.click() },
      { group: 'Panels', icon: 'trash', label: 'Clear playlist', run: () => clearPlaylist() },
      { group: 'Panels', icon: 'mic', label: 'Live input…', run: () => liveMode.setOpen(true) },
      { group: 'Panels', icon: 'genes', label: 'Gene editor (edit the playing preset)', keys: ['K'], run: () => toggleGenes() },
      { group: 'Panels', icon: 'hud', label: hudOn ? 'Hide HUD' : 'Show HUD', keys: ['H'], run: () => setHud(!hudOn) },
      { group: 'Panels', icon: 'keyboard', label: 'Keyboard shortcuts', keys: ['?'], run: () => palette.open('help'), stay: true },
      { group: 'About', icon: 'bug', label: 'Report a bug', href: BUG_URL, bug: true },
      { group: 'About', icon: 'github', label: 'MusicVis on GitHub', href: GITHUB_URL },
    ];
  }
  const palette = new Palette({
    commands,
    findPresets: (q) => {
      const t = q.trim().toUpperCase();
      const exact = findPreset(t);
      const hits = evo.pop.list().filter((m) => m.id.startsWith(t) || m.id.startsWith(`G0-${t}`));
      if (exact && !hits.includes(exact)) hits.unshift(exact);
      hits.sort((a, b) => (a === exact ? -1 : b === exact ? 1 : a.id.localeCompare(b.id)));
      return hits.slice(0, 8).map((m) => ({ id: m.id, name: m.name, hint: `${m.type} · ${m.energy}` }));
    },
    gotoPreset: (id) => play(id, 1.2, true),
    thumb: (id) => evo.thumb(id),
  });

  window.addEventListener('keydown', (ev) => {
    const tgt = ev.target;
    if (tgt instanceof HTMLInputElement || tgt instanceof HTMLSelectElement || tgt instanceof HTMLTextAreaElement || ev.metaKey || ev.ctrlKey || ev.altKey) return;
    // Nothing typed or pressed inside the gene editor (sliders, selects, buttons) triggers a shortcut.
    if (tgt instanceof Element && tgt.closest('#v2-genes')) return;
    if (ev.key === '?') {
      palette.toggle('help');
      return;
    }
    if (ev.key === '/') {
      ev.preventDefault();
      palette.toggle('cmd');
      return;
    }
    if (ev.key === 'Escape') {
      // Closes the top layer: overlay, then a menu / popover, then the dock.
      if (palette.isOpen) palette.close();
      else if (popovers.close()) return;
      else dock.close();
      return;
    }
    if (palette.isOpen) return;
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
          lyricSampler?.reset();
        }
        break;
      case 'ArrowRight':
        if (ev.shiftKey) goNext();
        else if (player) {
          player.seek(Math.min(player.duration, player.currentTime + 5));
          sampler?.reset();
          lyricSampler?.reset();
        }
        break;
      case 'f': case 'F': toggleFullscreen(); break;
      case 'h': case 'H': setHud(!hudOn); break;
      case 'p': case 'P': dock.toggle('playlist'); break;
      case 'n': case 'N': nextPreset(); break;
      case 'k': case 'K': toggleGenes(); break;
      case 's': case 'S': toggleShuffle(); break;
      case 'l': case 'L': vote(true); break;
      case 'd': case 'D': vote(false); break;
      case 'e': case 'E': setEvolve(!evolveOn); break;
      case 'b': case 'B': dock.toggle('presets'); break;
      case 'g': case 'G':
        ev.preventDefault();
        palette.open('goto');
        break;
      case 'm': case 'M':
        toggleMute();
        break;
      case '[': nudgeLyrics(-NUDGE_STEP); break;
      case ']': nudgeLyrics(NUDGE_STEP); break;
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
  window.setTimeout(() => pheno.startIdle(() => !screener.runner.busy && evo.breeding === 0), 6000);
  // The perceptual embedding stays opt-in: it only loads if the user turned it on before.
  if (loadSetting<boolean>('v2.embedding', false)) window.setTimeout(() => void setEmbedding(true), 8000);

  // The dock starts open on its last tab (a sheet on phones, so not there).
  updateEmpty();
  if (!computeLayout(dock.lastTab).sheet) dock.open(dock.lastTab);
  else relayout();

  // Debug / test handle.
  (window as unknown as Record<string, unknown>).musicvisV2 = {
    eng, evo, screener, pheno, presetMap, similarity, play, choose, current, editor, lyrics, playlist,
    get player() { return player; },
    get lyricSampler() { return lyricSampler; },
    get songResult() { return songResult; },
  };

  const hud = new Hud(hudEl);
  const lyricOverlay = new LyricOverlay(appRoot, canvas);
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
      lyricSampler?.apply(state, dt);
      transport.updatePlayback(player.currentTime, player.duration, player.playing);
    } else {
      lastIdle += dt;
      state = idleState(lastIdle, dt);
    }

    // Switching policy.
    // Unsaved gene edits pause automatic switching (drops, evolve rotation, new songs).
    if (duels.isOpen) {
      // The duel page renders its two presets itself; the main view and switching pause.
      duels.frame(state, dt);
      requestAnimationFrame(frame);
      return;
    }
    if (evolveOn) {
      if (!editor.dirty && (state.playing || !songLoaded || liveMode.active)) evolveTimer += dt;
      if (evolveTimer > EVOLVE_SECS) choose('evolve', 2.5);
    } else if (!editor.dirty && state.sectionChanged && state.section?.label === 'drop' && !eng.current()?.choreo && !eng.current()?.drift && !eng.current()?.dejavu) {
      // Presets that compose their own release on the drop stay on (choreography, drift), and so does a
      // deja vu preset: switching would forget the scenes it is remembering for the returns.
      choose('drop', 0.35);
    }

    // The shown preset's lyrics gene: the caption on screen, and the copy it smears into the feedback.
    const cap = lyricOverlay.update(state, eng.current()?.lyrics?.p, dt);
    eng.setCaption(cap.source, cap.version, cap.alpha);
    try {
      eng.render(state);
    } catch (err) {
      console.error('render failed', err);
    }
    keyHueNow = state.keyHue;
    look.tick(dt, canvas);
    if (screener.runner.busy) screener.runner.pump(eng.stats.frameMs > 18 ? 2 : 5);
    editor.tick(dt);

    if (hudOn) {
      const m = current();
      hud.update(state, { presetName: m ? `${m.id} · ${m.name}` : '—', fps, lyricSource: lyricSourceText() });
      statsTimer -= dt;
      if (statsTimer <= 0) {
        statsTimer = 0.5;
        const s = eng.stats;
        hud.setStats(`frame ${s.frameMs.toFixed(1)} ms · cpu ${s.cpuMs.toFixed(1)} · gpu ${s.gpuMs.toFixed(1)} ms\n${s.width}×${s.height} (scale ${s.scale.toFixed(2)})${m ? ` · ${m.type}` : ''}`);
      }
    }
    requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);
  // Gene chat runtime measurements, from the console: await __geneChatBench()
  Object.assign(window, {
    __geneChatBench: () => import('../chat/bench').then((m) => m.runBench()),
    __geneChatChunkBench: (sizes?: number[], yieldMs?: number) => import('../chat/bench').then((m) => m.runChunkBench(sizes, yieldMs)),
  });

  new AutoHide(appRoot, () => transport.busy || popovers.isOpen() || palette.isOpen);
}

/** Blocking card instead of the app (no WebGL2, or startup failed). */
function showStartupError(webgl: boolean): void {
  hydrateIcons(document.getElementById('startup-error')!);
  if (!webgl) {
    document.getElementById('se-title')!.textContent = 'MusicVis could not start';
    document.getElementById('se-text')!.textContent = 'Something went wrong while starting. Try again, and if it keeps happening, report a bug.';
  }
  document.getElementById('app')!.classList.add('fatal');
  document.getElementById('startup-error')!.hidden = false;
  document.getElementById('se-retry')!.addEventListener('click', () => location.reload());
}

main().catch((err) => {
  console.error(err);
  showStartupError(false);
  showToast('Failed to start MusicVis', 'error', 0, err instanceof Error ? err.message : undefined);
});
