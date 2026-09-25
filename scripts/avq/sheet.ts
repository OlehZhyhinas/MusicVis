// Timeline sheet: one tall PNG per (preset, song, clip) that an LLM (or a person) can read at a
// glance. Drawn with canvas 2D in the harness page from the files on disk:
//   header      preset, song, window, headline scores, report-card notes
//   music       spectrogram (64 log bands) with beat grid, downbeats, section changes, drops,
//               hook repeats; stem envelopes with onset ticks; melody pitch
//   visuals     main visual feature curves; the composite response with its peaks, onset
//               hits (green) and misses (red) inside the -45..+125 ms window
//   engine      per-reaction response curves from the engine readout
//   counterfact divergence heat rows (desync, offset, stem removal, reaction ablation, chaos floor)
//   filmstrip   frames at drum hits, at each hook repeat, and before / after each drop

import { asClipLike, type CfResult, type Clip } from './format';
import { onsetEvents, SYNC_EARLY_MS, SYNC_LATE_MS, visualResponse } from './metrics';
import type { ReportCard } from './metrics';

export interface SheetData {
  clip: Clip;
  card: ReportCard | null;
  cf: CfResult | null;
  inst: { names: string[]; cols: (number | null)[][] } | null;
  /** Loads frame i as an image (JPEG frames), or null when frames were not saved. */
  frame: (i: number) => Promise<CanvasImageSource | null>;
}

const W = 1600;
const PAD = 12;
const LABEL = 150; // left gutter for row labels
const PLOT = W - LABEL - PAD;

const COLORS = {
  bg: '#101216', panel: '#171a20', grid: '#2a2f38', text: '#e6e8eb', dim: '#9aa1ab',
  drums: '#ff8a4c', bass: '#b07cff', vocals: '#4cc3ff', other: '#6fdc8c',
  beat: 'rgba(255,255,255,0.10)', bar: 'rgba(255,255,255,0.28)', section: '#ffd166', drop: '#ff4d6d', hook: '#f9c74f',
  hit: '#6fdc8c', miss: '#ff4d6d', resp: '#ffffff', peak: '#ffd166',
};

const fmt = (x: number | null | undefined, d = 2) => (x === null || x === undefined || !Number.isFinite(x) ? '--' : x.toFixed(d));

function heat(v: number): string {
  // 0..1 -> dark blue .. yellow (magma-like, readable on dark).
  const t = Math.max(0, Math.min(1, v));
  const r = Math.round(255 * Math.min(1, 1.6 * t));
  const g = Math.round(255 * Math.max(0, Math.min(1, 1.6 * t - 0.6)));
  const b = Math.round(255 * Math.max(0, 0.45 - t) * 1.6 + 40 * t);
  return `rgb(${r},${g},${b})`;
}

export async function drawSheet(d: SheetData): Promise<HTMLCanvasElement> {
  const c = d.clip;
  const h = c.header;
  const fps = h.render.fps;
  const n = c.n;
  const t0 = h.clip.start; // song time of the frame before row 0
  const tOf = (i: number) => t0 + (i + 1) / fps;
  const xOfT = (t: number) => LABEL + ((t - tOf(0)) / Math.max(1e-6, tOf(n - 1) - tOf(0))) * PLOT;
  const xOfI = (i: number) => xOfT(tOf(i));

  // ---- layout (computed first so the canvas can be sized)
  const rows: { name: string; h: number }[] = [];
  const headerH = 150 + 16 * Math.min(10, d.card?.notes.length ?? 0);
  const rxN = d.inst ? d.inst.names.filter((x) => x.endsWith(':resp')).length : 0;
  const cfN = d.cf ? d.cf.variants.length : 0;
  rows.push({ name: 'spectrogram', h: 150 }, { name: 'stems', h: 130 }, { name: 'melody', h: 70 }, { name: 'visual', h: 230 }, { name: 'response', h: 90 });
  const hook0 = h.hooks[0];
  const hookOcc = hook0 ? hook0.occurrences.filter((o) => o.start >= t0 && o.end <= t0 + n / fps) : [];
  if (hookOcc.length >= 2) rows.push({ name: 'hook overlay', h: 130 });
  if (rxN) rows.push({ name: 'reactions', h: 24 + 30 * rxN });
  if (cfN) rows.push({ name: 'counterfactual', h: 24 + 16 * cfN + 40 });
  const filmCols = 8;
  const thumbW = Math.floor((W - 2 * PAD - (filmCols - 1) * 6) / filmCols);
  const thumbH = Math.round((thumbW * h.render.h) / h.render.w);

  // Filmstrip moments.
  const col = (f: string) => c.col(f as never);
  const shots: { i: number; label: string; group: string }[] = [];
  const onD = col('onDrums');
  const hitsIdx: number[] = [];
  for (let i = 1; i + 1 < n; i++) if (onD[i] >= 0.5 && onD[i] >= onD[i - 1] && onD[i] > onD[i + 1] && (!hitsIdx.length || i - hitsIdx[hitsIdx.length - 1] > fps * 0.15)) hitsIdx.push(i);
  const pickEvery = (arr: number[], k: number) => (arr.length <= k ? arr : Array.from({ length: k }, (_, j) => arr[Math.floor((j * arr.length) / k)]));
  for (const i of pickEvery(hitsIdx, 8)) shots.push({ i: Math.min(n - 1, i + 2), label: `hit ${tOf(i).toFixed(2)}s +67ms`, group: 'drum hits (frame 67 ms after the hit)' });
  const iOfT = (t: number) => Math.round((t - t0) * fps) - 1;
  const hook = h.hooks[0];
  if (hook) {
    const occ = hook.occurrences.map((o) => iOfT(o.start)).filter((i) => i >= 0 && i < n);
    for (const i of pickEvery(occ, 8)) shots.push({ i, label: `hook ${tOf(i).toFixed(2)}s`, group: `hook repeats (${hook.bars}-bar motif, first frame of each repeat)` });
  }
  for (const m of h.moments.filter((m) => m.kind !== 'section' || true)) {
    const i = iOfT(m.t);
    if (i < fps || i >= n - fps) continue;
    for (const [dt, tag] of [[-0.5, '-0.5s'], [0.1, '+0.1s'], [0.5, '+0.5s'], [1.5, '+1.5s']] as const) {
      const j = Math.round(i + dt * fps);
      if (j >= 0 && j < n) shots.push({ i: j, label: `${m.kind} ${tag} (${tOf(j).toFixed(2)}s)`, group: `${m.kind} ${m.label} at ${m.t.toFixed(2)}s` });
    }
  }
  const groups = [...new Set(shots.map((s) => s.group))];
  const filmH = groups.reduce((a, g) => a + 22 + Math.ceil(shots.filter((s) => s.group === g).length / filmCols) * (thumbH + 20), 0) + 10;

  const total = headerH + rows.reduce((a, r) => a + r.h + 8, 0) + filmH + 30;
  const cv = document.createElement('canvas');
  cv.width = W;
  cv.height = total;
  const g = cv.getContext('2d')!;
  g.fillStyle = COLORS.bg;
  g.fillRect(0, 0, W, total);
  g.textBaseline = 'top';
  const text = (s: string, x: number, y: number, color = COLORS.text, size = 13, bold = false) => {
    g.fillStyle = color;
    g.font = `${bold ? 'bold ' : ''}${size}px -apple-system, Helvetica, Arial, sans-serif`;
    g.fillText(s, x, y);
  };

  // ---- header
  text(`${h.preset.id} ${h.preset.name}`, PAD, PAD, COLORS.text, 22, true);
  text(`${h.song.slug}   window ${h.clip.label} ${tOf(0).toFixed(2)}-${tOf(n - 1).toFixed(2)} s   ${h.song.bpm.toFixed(1)} bpm   ${h.render.w}x${h.render.h} @ ${fps} fps, seed ${h.render.seed}`, PAD, PAD + 30, COLORS.dim, 14);
  if (d.card) {
    const hl = d.card.headline as unknown as Record<string, number>;
    const keys = ['overall', 'sync', 'coupling', 'hookRhyme', 'melody', 'structure', 'flow', 'interest', 'correspond'];
    keys.forEach((k, j) => {
      const x = PAD + j * 170;
      text(k, x, PAD + 56, COLORS.dim, 12);
      text(fmt(hl[k]), x, PAD + 72, hl[k] < 0.2 ? COLORS.miss : hl[k] > 0.6 ? COLORS.hit : COLORS.text, 22, true);
    });
    const cfS = d.card.counterfactual;
    const extra = [
      `onset hits ${fmt(d.card.sync.all.hitRate)} vs chance ${fmt(d.card.sync.all.chance)} (lag ${fmt(d.card.sync.all.meanLagMs, 0)} ms)`,
      `drum hits ${fmt(d.card.sync.hits.hitRate)} vs ${fmt(d.card.sync.hits.chance)}`,
      cfS ? `desync sensitivity ${fmt(cfS.syncSensitivity)} (chaos floor ${fmt(cfS.chaos)})` : '',
      `stillness ${fmt(d.card.flow.stillness)}  strobe ${d.card.flow.strobe ? 'YES' : 'no'}`,
    ].filter(Boolean);
    text(extra.join('     '), PAD, PAD + 104, COLORS.dim, 13);
    d.card.notes.slice(0, 10).forEach((note, j) => text('- ' + note, PAD, PAD + 126 + 16 * j, '#ffcf8a', 13));
  }
  let y = headerH;

  const panel = (name: string, hh: number, label: string) => {
    g.fillStyle = COLORS.panel;
    g.fillRect(PAD, y, W - 2 * PAD, hh);
    text(label, PAD + 6, y + 6, COLORS.dim, 12, true);
    // beat grid
    for (const b of h.beats) {
      if (b < tOf(0) || b > tOf(n - 1)) continue;
      g.fillStyle = COLORS.beat;
      g.fillRect(Math.round(xOfT(b)), y, 1, hh);
    }
    for (const b of h.downbeats) {
      if (b < tOf(0) || b > tOf(n - 1)) continue;
      g.fillStyle = COLORS.bar;
      g.fillRect(Math.round(xOfT(b)), y, 1, hh);
    }
    for (const m of h.moments) {
      if (m.t < tOf(0) || m.t > tOf(n - 1)) continue;
      g.fillStyle = m.kind === 'section' ? COLORS.section : COLORS.drop;
      g.fillRect(Math.round(xOfT(m.t)) - 1, y, 2, hh);
    }
    void name;
  };

  const curve = (x: Float32Array, top: number, hh: number, color: string, lo?: number, hi?: number, width = 1.5) => {
    let a = lo ?? Infinity, b = hi ?? -Infinity;
    if (lo === undefined || hi === undefined) for (const v of x) if (Number.isFinite(v)) (a = Math.min(a, v)), (b = Math.max(b, v));
    if (!(b > a)) b = a + 1;
    g.strokeStyle = color;
    g.lineWidth = width;
    g.beginPath();
    let pen = false;
    for (let i = 0; i < x.length; i++) {
      if (!Number.isFinite(x[i])) {
        pen = false;
        continue;
      }
      const px = xOfI(i);
      const py = top + hh - ((x[i] - a) / (b - a)) * hh;
      if (pen) g.lineTo(px, py);
      else g.moveTo(px, py);
      pen = true;
    }
    g.stroke();
  };

  // ---- spectrogram
  {
    const r = rows[0];
    panel(r.name, r.h, 'spectrum 40 Hz-16 kHz');
    const bands = h.spec.bands;
    const top = y + 22;
    const hh = r.h - 44;
    let mx = 1;
    for (let i = 0; i < n; i++) for (const v of c.spec(i)) mx = Math.max(mx, v);
    const img = g.createImageData(PLOT, hh);
    for (let px = 0; px < PLOT; px++) {
      const i = Math.min(n - 1, Math.floor((px / PLOT) * n));
      const s = c.spec(i);
      for (let py = 0; py < hh; py++) {
        const b = Math.min(bands - 1, Math.floor(((hh - 1 - py) / hh) * bands));
        const v = Math.pow(s[b] / mx, 1.6);
        const k = (py * PLOT + px) * 4;
        const cs = heat(v).match(/\d+/g)!.map(Number);
        img.data[k] = cs[0];
        img.data[k + 1] = cs[1];
        img.data[k + 2] = cs[2];
        img.data[k + 3] = 255;
      }
    }
    g.putImageData(img, LABEL, top);
    // section labels and hook bars
    for (const m of h.moments) if (m.t >= tOf(0) && m.t <= tOf(n - 1)) text(`${m.label} ${m.t.toFixed(1)}s`, xOfT(m.t) + 3, y + 6, m.kind === 'section' ? COLORS.section : COLORS.drop, 12, true);
    for (const hk of h.hooks) {
      for (const o of hk.occurrences) {
        if (o.end < tOf(0) || o.start > tOf(n - 1)) continue;
        const x0 = Math.max(LABEL, xOfT(o.start));
        const x1 = Math.min(LABEL + PLOT, xOfT(o.end));
        g.fillStyle = hk.id === 0 ? COLORS.hook : '#90be6d';
        g.fillRect(x0 + 1, y + r.h - 18, x1 - x0 - 2, 8);
      }
      text(`hook${hk.id} (${hk.bars} bar)`, PAD + 6, y + r.h - 20, hk.id === 0 ? COLORS.hook : '#90be6d', 11);
    }
    y += r.h + 8;
  }

  // ---- stems
  {
    const r = rows[1];
    panel(r.name, r.h, 'stems (env, onset ticks)');
    const lane = (r.h - 20) / 4;
    (['drums', 'bass', 'vocals', 'other'] as const).forEach((s, k) => {
      const top = y + 18 + k * lane;
      text(s, PAD + 6, top + lane / 2 - 6, COLORS[s], 12);
      curve(col(s), top, lane - 4, COLORS[s], 0, 1);
      const on = col(`on${s[0].toUpperCase()}${s.slice(1)}`);
      g.fillStyle = COLORS[s];
      for (let i = 1; i + 1 < n; i++) if (on[i] > 0.3 && on[i] >= on[i - 1] && on[i] > on[i + 1]) g.fillRect(xOfI(i), top + lane - 4 - on[i] * 8, 1.5, on[i] * 8);
    });
    y += r.h + 8;
  }

  // ---- melody
  {
    const r = rows[2];
    panel(r.name, r.h, 'melody pitch (MIDI)');
    const midi = col('melMidi');
    const sal = col('melSal');
    for (let i = 0; i < n; i++) {
      if (!Number.isFinite(midi[i]) || sal[i] < 0.3) continue;
      const py = y + r.h - 6 - ((midi[i] - 45) / 45) * (r.h - 12);
      g.fillStyle = `rgba(76,195,255,${0.3 + 0.7 * sal[i]})`;
      g.fillRect(xOfI(i), py, 2, 2);
    }
    curve(col('cy'), y + 8, r.h - 16, 'rgba(255,255,255,0.6)', 0, 1, 1);
    text('dots: pitch   white: visual height (cy)', PAD + 6, y + r.h - 18, COLORS.dim, 11);
    y += r.h + 8;
  }

  // ---- visual features
  {
    const r = rows[3];
    panel(r.name, r.h, 'visual features');
    const feats: [string, string][] = [['lum', '#ffffff'], ['coverage', '#f9c74f'], ['colorful', '#f94144'], ['diff', '#90be6d'], ['flowMag', '#43aa8b'], ['div', '#577590'], ['curl', '#f3722c'], ['spread', '#b07cff']];
    const lane = (r.h - 20) / feats.length;
    feats.forEach(([f, colr], k) => {
      const top = y + 18 + k * lane;
      text(f, PAD + 6, top + lane / 2 - 6, colr, 12);
      curve(col(f), top + 2, lane - 4, colr);
    });
    y += r.h + 8;
  }

  // ---- response + hits
  {
    const r = rows[4];
    panel(r.name, r.h, 'response + onset hits');
    // Recompute the composite response the same way as metrics.visualResponse (sum of scaled change signals).
    const cl = asClipLike(c);
    const R = visualResponse(cl);
    curve(R.resp, y + 20, r.h - 40, COLORS.resp, 0, undefined, 1);
    g.fillStyle = COLORS.peak;
    for (const i of R.peakIdx) g.fillRect(xOfI(i) - 1, y + 18, 3, 5);
    const onAll = Float32Array.from(col('onDrums'), (v, i) => Math.max(v, col('onBass')[i], col('onVocals')[i], col('onOther')[i]));
    const ev = onsetEvents(onAll, fps);
    const lo = -Math.floor((SYNC_EARLY_MS / 1000) * fps), hi = Math.floor((SYNC_LATE_MS / 1000) * fps);
    for (const e of ev) {
      let hit = false;
      for (let dd = lo; dd <= hi; dd++) if (R.peaks[e + dd]) hit = true;
      g.fillStyle = hit ? COLORS.hit : COLORS.miss;
      g.fillRect(xOfI(e) - 1, y + r.h - 16, 3, 10);
    }
    text('green/red: onsets hit/missed (-45..+125 ms)   yellow: response peaks', PAD + 6, y + r.h - 30, COLORS.dim, 11);
    y += r.h + 8;
  }

  // ---- hook overlay: the composite response over each repeat of the top hook, aligned
  if (hookOcc.length >= 2) {
    const r = rows.find((x) => x.name === 'hook overlay')!;
    g.fillStyle = COLORS.panel;
    g.fillRect(PAD, y, W - 2 * PAD, r.h);
    text(`hook repeats overlaid`, PAD + 6, y + 6, COLORS.dim, 12, true);
    text(`(${hookOcc.length} x ${hook0!.len.toFixed(2)} s)`, PAD + 6, y + 22, COLORS.dim, 11);
    text('similar shapes = the riff gets the same response each time', PAD + 6, y + r.h - 18, COLORS.dim, 11);
    const R = visualResponse(asClipLike(c)).resp;
    const lenF = Math.round(hook0!.len * fps);
    let mx = 1e-6;
    for (const v of R) mx = Math.max(mx, v);
    const beatsPer = Math.round(hook0!.len / (60 / h.song.bpm));
    for (let b = 0; b <= beatsPer; b++) {
      g.fillStyle = b % (h.song.beatsPerBar || 4) === 0 ? COLORS.bar : COLORS.beat;
      g.fillRect(LABEL + (b / beatsPer) * PLOT, y, 1, r.h);
    }
    hookOcc.forEach((o, k) => {
      const f0 = Math.round((o.start - t0) * fps) - 1;
      g.strokeStyle = `hsla(${(k * 360) / hookOcc.length}, 80%, 65%, 0.75)`;
      g.lineWidth = 1.2;
      g.beginPath();
      for (let j = 0; j < lenF && f0 + j < n; j++) {
        const px = LABEL + (j / lenF) * PLOT;
        const py = y + r.h - 8 - (R[Math.max(0, f0 + j)] / mx) * (r.h - 16);
        if (j) g.lineTo(px, py);
        else g.moveTo(px, py);
      }
      g.stroke();
    });
    y += r.h + 8;
  }

  // ---- reactions
  if (rxN && d.inst) {
    const r = rows.find((x) => x.name === 'reactions')!;
    panel(r.name, r.h, 'engine: reaction response');
    const names = d.inst.names.filter((x) => x.endsWith(':resp'));
    names.forEach((nm, k) => {
      const top = y + 22 + k * 30;
      const arr = Float32Array.from(d.inst!.cols[d.inst!.names.indexOf(nm)], (v) => (v === null ? NaN : v));
      const tag = nm.replace(/^rx\d+:/, '').replace(/:resp$/, '');
      const ro = d.card?.readout?.reactions[k];
      text(tag.slice(0, 22), PAD + 6, top + 4, COLORS.text, 11);
      if (ro) text(ro.verdict, PAD + 6, top + 16, ro.verdict === 'visible' ? COLORS.hit : COLORS.miss, 10);
      curve(arr, top, 26, '#4cc3ff', 0, 1);
    });
    y += r.h + 8;
  }

  // ---- counterfactual heat rows
  if (cfN && d.cf) {
    const r = rows.find((x) => x.name === 'counterfactual')!;
    panel(r.name, r.h, 'counterfactual divergence');
    const M = Math.max(1e-6, d.cf.motion);
    const cfN0 = d.cf.variants[0].series.length;
    d.cf.variants.forEach((v, k) => {
      const top = y + 22 + k * 16;
      text(`${v.id} ${v.rel.toFixed(2)}`, PAD + 6, top + 1, COLORS.text, 11);
      for (let px = 0; px < PLOT; px += 2) {
        const i = Math.min(cfN0 - 1, Math.floor((px / PLOT) * cfN0));
        g.fillStyle = heat(v.series[i] / (2 * M));
        g.fillRect(LABEL + px, top, 2, 14);
      }
    });
    text('each row: |base - variant| per frame, relative to the preset\'s own half-bar motion (bright = the change shows)', PAD + 6, y + r.h - 16, COLORS.dim, 11);
    y += r.h + 8;
  }

  // ---- filmstrip
  y += 6;
  for (const grp of groups) {
    text(grp, PAD, y, COLORS.text, 14, true);
    y += 22;
    const list = shots.filter((s) => s.group === grp);
    for (let k = 0; k < list.length; k++) {
      const x = PAD + (k % filmCols) * (thumbW + 6);
      const yy = y + Math.floor(k / filmCols) * (thumbH + 20);
      const im = await d.frame(list[k].i);
      if (im) g.drawImage(im, x, yy, thumbW, thumbH);
      else {
        // Fall back to the 32x18 thumbnail.
        const t = c.thumb(list[k].i);
        const tmp = g.createImageData(h.thumb.w, h.thumb.h);
        for (let p = 0; p < h.thumb.w * h.thumb.h; p++) {
          tmp.data[p * 4] = t[p * 3];
          tmp.data[p * 4 + 1] = t[p * 3 + 1];
          tmp.data[p * 4 + 2] = t[p * 3 + 2];
          tmp.data[p * 4 + 3] = 255;
        }
        const oc = new OffscreenCanvas(h.thumb.w, h.thumb.h);
        oc.getContext('2d')!.putImageData(tmp, 0, 0);
        g.imageSmoothingEnabled = false;
        g.drawImage(oc, x, yy, thumbW, thumbH);
        g.imageSmoothingEnabled = true;
      }
      text(list[k].label, x, yy + thumbH + 3, COLORS.dim, 11);
    }
    y += Math.ceil(list.length / filmCols) * (thumbH + 20);
  }
  return cv;
}
