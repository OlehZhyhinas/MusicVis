// Editor availability follows the combinations repairBody() can keep and the
// branches the renderer actually reads. Values stay in the genome while a
// control is inactive, so turning its parent effect back on restores it.
import { countKey, isFoldPlace, MAX_DRAW, ringsOn, sdfCapable, SHAPE_CLASS, SHAPE_SCHEMAS, UNIQUE_SHAPES, type Genome, type Locus, type ReactionGene, type ShapeKind } from './genome';
import type { Target } from './geneEdit';

/** A kind that cannot be selected in the current body. */
export function kindUnavailableReason(g: Genome, t: Target, kind: string): string | null {
  if (t.t !== 'locus' && t.t !== 'fuseShape') return null;
  const b = g.bodies[t.b];
  if (!b) return null;
  const current = t.t === 'fuseShape' ? b.fuse?.shape.kind : b[t.locus].kind;
  if (kind === current) return null;
  if (t.t === 'fuseShape') {
    if (kind === 'compound') return 'Compound can only be the main shape.';
    if (UNIQUE_SHAPES.includes(kind as ShapeKind) && g.bodies.some((x, i) => x.shape.kind === kind || (i !== t.b && x.fuse?.shape.kind === kind))) return `Only one ${kind} is allowed per preset.`;
    const shape = { kind: kind as ShapeKind, p: Object.fromEntries(Object.entries(SHAPE_SCHEMAS[kind as ShapeKind] ?? {}).map(([k, v]) => [k, v.def])) };
    if (!sdfCapable(shape)) return 'A fused shape needs a distance field.';
    if (kind === b.shape.kind && b.fuse?.p.mode !== 2) return 'The same shape can only be fused in region mode.';
    return null;
  }
  if (t.locus === 'shape') {
    if (UNIQUE_SHAPES.includes(kind as ShapeKind) && g.bodies.some((x, i) => i !== t.b && (x.shape.kind === kind || x.fuse?.shape.kind === kind))) return `Only one ${kind} is allowed per preset.`;
    return null;
  }
  const cls = SHAPE_CLASS[b.shape.kind];
  if (t.locus === 'place') {
    if (cls === 'flame' && isFoldPlace(kind as typeof b.place.kind)) return 'Flame shapes need a position, not a fold placement.';
    if (cls === 'curve' && kind === 'grid') return 'Curve shapes cannot use grid placement.';
  }
  if (t.locus === 'material' && cls === 'curve' && ['fill', 'textured', 'chrome', 'iridescent'].includes(kind)) return 'Curve shapes need a line, glow, or dots material.';
  if (t.locus === 'emit') {
    if (cls === 'flame' && kind !== 'trail' && kind !== 'sparks') return 'Flame shapes only support trail or sparks emission.';
    if (kind === 'cover' && cls !== 'sdf' && !(cls === 'curve' && b.fuse)) return 'Cover emission needs a distance field or a fused curve.';
  }
  return null;
}

/** A parameter that currently has no effect in the renderer. */
export function paramInactiveReason(g: Genome, t: Target, key: string): string | null {
  if (t.t === 'carrier') {
    const kind = g.carrier.kind;
    if (kind === 'none') return 'Choose a feedback carrier to use this setting.';
    if (['amount', 'vort', 'fnoise'].includes(key) && kind !== 'fluid') return 'Used by the fluid carrier.';
    if (['famt', 'fscale'].includes(key) && kind !== 'flow') return 'Used by the flow carrier.';
    if (key === 'grain' && g.carrier.p.sharpen <= 0.001) return 'Raise Sharpen to use grain.';
    if (key === 'wsize' && g.carrier.p.water <= 0.001) return 'Raise Water to use drop size.';
  }
  if (t.t === 'tone') {
    if (key === 'reflectY' && g.tone.p.reflect < 0.5) return 'Turn on Reflect to set its position.';
    if (['bump', 'light', 'gloss', 'metal'].includes(key) && g.tone.p.relief <= 0.001) return 'Raise Relief to use this setting.';
    if (['bands', 'drift', 'poster'].includes(key) && g.tone.p.huemap <= 0.001) return 'Raise Hue map to use this setting.';
  }
  if (t.t === 'reaction' && key === 'div' && g.reactions[t.j]?.q === 0) return 'Turn on Quantise to choose a clock division.';
  if (t.t === 'fuse') {
    const p = g.bodies[t.b]?.fuse?.p;
    if (!p) return null;
    if (key === 'k' && p.mode !== 0) return 'Blend radius is used in union mode.';
    if (['t', 'drive', 'depth', 'rate'].includes(key) && p.mode !== 1) return 'This setting is used in morph mode.';
    if (key === 'depth' && p.drive === 0) return 'Choose a morph driver to use depth.';
    if (key === 'rate' && p.drive !== 1) return 'Sweep rate is used with the sweep driver.';
  }
  if (t.t !== 'locus') return null;
  const b = g.bodies[t.b];
  if (!b) return null;
  const cls = SHAPE_CLASS[b.shape.kind];
  if (t.locus === 'place') {
    if ((cls === 'field' || cls === 'flame') && key === countKey(b.place.kind)) return 'This shape is drawn as one copy.';
    if (cls === 'field' && key === 'fuse' && b.place.kind === 'float') return 'Floating field shapes cannot merge copies.';
    if (b.place.kind === 'stations' && ['xs', 'jump'].includes(key) && b.place.p.inst <= 0.001) return 'Raise Instrument follow to use this setting.';
  }
  if (t.locus === 'material') {
    if (cls === 'field' && !['gain', 'blend', ...(b.shape.kind === 'ribbon' && b.material.kind === 'iridescent' ? ['sheen'] : [])].includes(key)) return 'Field shapes use the material look, gain, and blend; this control is not read.';
    if (key === 'rings' && !ringsOn({ ...b, material: { ...b.material, p: { ...b.material.p, rings: 1 } } })) return 'Ring halos need a distance-field shape or a fused curve.';
    if (['rgap', 'rfade'].includes(key)) {
      if (!ringsOn({ ...b, material: { ...b.material, p: { ...b.material.p, rings: 1 } } })) return 'Ring halos need a distance-field shape or a fused curve.';
      if (!ringsOn(b)) return 'Raise Rings to use this setting.';
    }
  }
  return null;
}

/** Map a reaction's stored group/index to the same control availability rule. */
export function reactionTargetInactiveReason(g: Genome, r: Pick<ReactionGene, 'g' | 'i' | 'k'>): string | null {
  const loci: Partial<Record<typeof r.g, Locus>> = {
    sh: 'shape', pl: 'place', mo: 'motion', de: 'deform', ma: 'material', em: 'emit', fe: 'feel', cm: 'color',
  };
  const locus = loci[r.g];
  if (locus) return paramInactiveReason(g, { t: 'locus', b: r.i, locus }, r.k);
  if (r.g === 'fu') return paramInactiveReason(g, { t: 'fuse', b: r.i }, r.k);
  if (r.g === 'fs') return paramInactiveReason(g, { t: 'fuseShape', b: r.i }, r.k);
  if (r.g === 'dr') return paramInactiveReason(g, { t: 'drawOp', b: Math.floor(r.i / MAX_DRAW), j: r.i % MAX_DRAW }, r.k);
  if (r.g === 'op') return paramInactiveReason(g, { t: 'op', j: r.i }, r.k);
  if (r.g === 'car') return paramInactiveReason(g, { t: 'carrier' }, r.k);
  if (r.g === 'col') return paramInactiveReason(g, { t: 'tone' }, r.k);
  if (r.g === 'pal') return paramInactiveReason(g, { t: 'palette' }, r.k);
  return null;
}
