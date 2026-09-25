// Checks for the notes shape gene (the melody drawn as it is played: ribbons for held notes, marks
// at note starts) and the articulation signals. Called from v2-test.ts.

import { SIGNALS, COST_BUDGET_MS, estimateCost, repair } from '../src/v2/genome';
import { NOTE_MARKS, NOTE_SEC, NOTE_W, NoteHistory, packNotes } from '../src/v2/genes/notes';
import { shapeGeneChecks } from './v2-physics';
import type { NoteStats } from '../src/types';

type Check = (name: string, ok: boolean, detail: string) => void;

const stats = (x: Partial<NoteStats> = {}): NoteStats => ({ on: 0, held: 0, legato: 0.5, glide: 0, vibrato: 0, pitch: 60, height: 0.5, voice: 0, recent: [], ...x });

export function notesGeneTests(check: Check): void {
  check('notes.signals', ['noteon', 'held', 'legato', 'glide', 'vibrato', 'voice'].every((s) => (SIGNALS as readonly string[]).includes(s)), SIGNALS.slice(-6).join(','));

  shapeGeneChecks(check, 'notes', 'vec4 ntAt(', 'melody', null);

  // History: samples every NOTE_SEC / NOTE_W s, newest first; gaps drop the held strength; marks
  // carry age, length, height and a negative strength once ended.
  {
    const h = new NoteHistory();
    const step = NOTE_SEC / NOTE_W;
    // 0.5 s of a held note at height 0.8, then 0.25 s of silence (a gap).
    for (let t = 0; t < 0.5; t += 1 / 60) h.push(stats({ held: 0.9, height: 0.8, legato: 0.9 }), 1 / 60);
    for (let t = 0; t < 0.25; t += 1 / 60) h.push(stats({ held: 0, height: 0.8 }), 1 / 60);
    const at = (age: number) => h.data.subarray(Math.round(age / step) * 4, Math.round(age / step) * 4 + 4);
    const gap = at(0.1), note = at(0.5);
    h.push(stats({ recent: [{ age: 0.6, len: 0.1, ended: true, height: 0.3, strength: 0.7 }, { age: 0.05, len: 0.05, ended: false, height: 0.9, strength: 0.5 }] }), 1 / 60);
    const m = h.data.subarray(NOTE_W * 4, NOTE_W * 4 + 8);
    check('notes.history', gap[1] < 0.05 && note[1] > 0.85 && Math.abs(note[0] - 0.8) < 1e-6 && Math.abs(note[3] - 0.9) < 1e-6
      && Math.abs(m[0] - 0.6) < 1e-6 && m[3] < 0 && Math.abs(m[2] - 0.3) < 1e-6 && m[7] > 0 && h.data[NOTE_W * 4 + 8 + 3] === 0,
      `gap held ${gap[1].toFixed(2)}, note held ${note[1].toFixed(2)} height ${note[0].toFixed(2)}; marks ${[...m].map((v) => v.toFixed(2)).join(' ')}`);
    // Only the newest NOTE_MARKS notes are kept.
    const many = Array.from({ length: 20 }, (_, i) => ({ age: 20 - i, len: 0.1, ended: true, height: 0.5, strength: 0.5 }));
    h.push(stats({ recent: many }), 0);
    check('notes.history-marks', Math.abs(h.data[NOTE_W * 4] - 12) < 1e-6 && Math.abs(h.data[NOTE_W * 4 + (NOTE_MARKS - 1) * 4] - 1) < 1e-6, `first mark age ${h.data[NOTE_W * 4]}, last ${h.data[NOTE_W * 4 + (NOTE_MARKS - 1) * 4]}`);
  }

  // Packing: the legato now eases between ribbon and marks; structural switches pass through.
  {
    const g = repair({ v: 5, chain: [], bodies: [{ shape: { kind: 'notes', p: { mode: 1, form: 2 } } }], carrier: { kind: 'warp', p: {} }, palette: { kind: 'triad', p: {} }, tone: { p: {} }, reactions: [], energy: [0.3, 0.9] });
    const p = g.bodies[0].shape.p;
    const E = new Float32Array(84);
    const mem: Record<string, number> = {};
    const key = (k: string) => `b0.${k}`;
    for (let i = 0; i < 180; i++) packNotes(E, 64, 8, 12, (k) => p[k], p, stats({ legato: 1, held: 0.7, vibrato: 0.3 }), mem, key, 1 / 60);
    const easedUp = E[66];
    for (let i = 0; i < 180; i++) packNotes(E, 64, 8, 12, (k) => p[k], p, stats({ legato: 0 }), mem, key, 1 / 60);
    check('notes.pack', easedUp > 0.9 && E[66] < 0.15 && E[68] === 1 && E[77] === 2 && Math.abs(E[12] - 0) < 1e-6 && estimateCost(g) < COST_BUDGET_MS,
      `legato eased to ${easedUp.toFixed(2)} then ${E[66].toFixed(2)}, mode ${E[68]}, form ${E[77]}, cost ${estimateCost(g).toFixed(2)} ms`);
  }
}
