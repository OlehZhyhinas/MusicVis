// Human-readable help for the schema-driven gene editor. Keep the descriptions
// here rather than in the widgets so the same parameter is explained wherever
// it appears, including in a reaction target.

import { ENTRIES } from '../chat/glossary';
import { genomeGene } from './geneRegistry';
import { type Genome, type Signal } from './genome';
import { labelFor, reactionTargets, targetLabel, type ParamControl, type ReactTarget, type Target } from './geneEdit';

export const SIGNAL_HELP: Record<Signal, string> = {
  drums: 'Level of the drum stem. A continuous measure of how present the drums are.',
  bass: 'Level of the bass stem.',
  vocals: 'Level of the vocal stem.',
  other: 'Level of the other instruments stem.',
  hit: 'A short trigger on detected drum hits.',
  beat: 'A pulse on each beat.',
  bar: 'A slower wave over each musical bar.',
  complexity: 'How busy or dense the mix is.',
  drop: 'A pulse at a structural drop.',
  loud: 'Overall loudness of the mix.',
  melody: 'How active the detected melody is.',
  build: 'Tension building before a drop.',
  surge: 'A beat-shaped energy envelope that also rises on drops.',
  barpulse: 'A pulse on each downbeat, the first beat of a bar.',
  section: 'A pulse when the song changes section.',
  tension: 'Harmonic tension: distance from the home key or dissonance.',
  resolve: 'A pulse when the harmony resolves toward home.',
  chordchange: 'A pulse whenever the detected chord changes.',
  modulation: 'A pulse when the detected key changes.',
  swing: 'Amount of swung timing, from straight to triplet feel.',
  push: 'How far the backbeat leads or trails the beat grid.',
  humanity: 'How loose or human the timing is.',
  synco: 'Density of accents on weak beats and offbeats.',
  line: 'A pulse at the start of each sung lyric line.',
  valence: 'Mood of the words, from sad to happy.',
  arousal: 'Intensity of the musical mood.',
  bright: 'How much high-frequency brightness the sound has.',
  noisy: 'How noisy or breathy the sound is, including cymbals and distortion.',
  rough: 'How buzzy or beating the sound is.',
  attack: 'How sharp the sound onsets are, such as plucks and hits.',
  register: 'Melody note height within the song’s range: 0 low, 1 high. Retains the last height between notes; 0.5 without note analysis.',
  rising: 'Upward melody slides: 0 steady or falling, 1 at an octave per second. Excludes vibrato and is zero between notes.',
  falling: 'Downward melody slides: 0 steady or rising, 1 at an octave per second. Excludes vibrato and is zero between notes.',
  noteon: 'A pulse when a melody note starts.',
  held: 'Strength of a sustained melody note; low between notes.',
  legato: 'How connected or gliding the melody notes are.',
  glide: 'How quickly the melody pitch slides.',
  vibrato: 'Depth of the melody pitch wobble.',
  voice: 'How voice-like the melody sounds.',
  hook: 'A pulse at each note of the recurring riff or hook.',
  hookphase: 'Position through a repeat of the riff or hook.',
  hookon: 'High while the recurring riff or hook is playing.',
};

const SECTION_HELP: Record<string, string> = {
  shape: 'What each body draws.', place: 'Where copies of the body appear.', motion: 'How the body moves over time.',
  deform: 'How the body shape bends before drawing.', material: 'How the body is shaded and blended.',
  emit: 'How the body adds light or particles to the picture.', feel: 'How quickly this body follows the music and which beat grid it uses.',
  color: 'How this body chooses colours from the palette.', fuse: 'How a second shape combines with this body.',
  fuseShape: 'The second shape fused into this body.', palette: 'The three base colour slots, usually relative to the song key.',
  tone: 'Final colour and lighting applied to the whole picture.', carrier: 'How previous frames persist and move.',
  op: 'A transformation of the accumulated or displayed picture.', drawOp: 'A transformation of this body before it is drawn.',
  xform: 'One affine transform in the flame particle system.', reaction: 'A music signal driving one visual parameter.',
  energy: 'The song energy range this preset prefers.',
};

const PARAM_HELP: Record<string, string> = {
  'deform.undulate.amp': 'Sideways bend in scene units; zero restores the original silhouette.',
  'deform.undulate.span': 'Distance between wave crests. Small values make tighter ripples.',
  'deform.undulate.rate': 'Wave cycles per musical bar. Negative reverses travel; zero holds a static bend.',
  'deform.undulate.angle': 'Wave axis in turns. Zero sends waves vertically; a quarter turn sends them horizontally.',
  'deform.undulate.pin': 'At 1 the local centre stays fixed while the body bends around it; at 0 the entire wave travels.',
  'motion.recoil.source': 'Trigger: 0 drum hits, 1 melody note starts, 2 bass onsets, 3 vocal onsets, 4 other instrument onsets.',
  'motion.recoil.distance': 'Maximum travel away from the body’s resting position.',
  'motion.recoil.direction': 'Direction of the kick in turns: 0 right, 0.25 up, -0.25 down.',
  'motion.recoil.tilt': 'Rotation coupled to the spring, in radians; negative turns the other way.',
  'motion.recoil.frequency': 'Spring oscillations per second.',
  'motion.recoil.damping': 'How quickly the bounce settles; 1 returns without overshooting.',
  'motion.recoil.fan': 'How far the kick direction fans out across copies.',
  'shape.shell.turns': 'Number of windings in the coil.',
  'shape.shell.growth': 'How quickly the coil grows outward; higher values separate the outer windings.',
  'shape.shell.width': 'Thickness of the coiled tube relative to its radius.',
  'shape.shell.ribs': 'Number of surface ribs per turn.',
  'shape.shell.relief': 'Strength of the rib shading on the shell surface.',
  'shape.shell.aperture': 'How wide the outer opening flares.',
  'shape.linkage.joints': 'Number of connected limbs in the chain.',
  'shape.linkage.width': 'Thickness of the first limb.',
  'shape.linkage.curl': 'Resting bend added at each joint, in radians.',
  'shape.linkage.flex': 'Amplitude of the musical wave passing through the joints.',
  'shape.linkage.taper': 'Length and thickness of each limb relative to the previous one.',
  'shape.linkage.knuckle': 'Size of the rounded joints relative to the limbs.',
  'shape.fabric.aspect': 'Width of the sheet relative to its height.',
  'shape.fabric.depth': 'Strength of the pleat lighting and scalloped edges.',
  'shape.branch.spread': 'Angle between each branch and its parent, in radians.',
  'shape.branch.width': 'Thickness of the first limb; later branches taper.',
  'shape.folds': 'Number of pleats across the fabric.',
  'shape.drape': 'How much the middle of the sheet sags.',
  'shape.weave': 'Visibility of the fine woven threads.',
  'shape.flutter': 'Amount of rippling along the fabric and its hem.',
  'shape.levels': 'Number of branching generations.',
  'shape.grow': 'How far the branches have grown; reveals successive generations.',
  'shape.ratio': 'Length of each child branch relative to its parent.',
  'shape.bend': 'Amount of slow flex within the shape.',
  'reaction.gain': 'How strongly the source signal moves the chosen target. Positive raises it; negative lowers it. One unit spans half the target parameter range.',
  'reaction.atk': 'Attack time in seconds: how quickly this reaction rises when its source signal rises.',
  'reaction.rel': 'Release time in seconds: how quickly this reaction falls after its source signal falls.',
  'reaction.thr': 'Source level below which this reaction stays inactive.',
  'reaction.q': 'Whether the reaction responds continuously or only on its chosen beat clock.',
  'reaction.div': 'Beat spacing used when this reaction is set to on-clock.',
  'feel.atk': 'Attack time in seconds: how quickly this body responds when the music gets stronger.',
  'feel.rel': 'Release time in seconds: how slowly this body settles when the music gets quieter.',
  'feel.thr': 'Music level below which this body does not react.',
  'feel.sens': 'Sensitivity of this body to the music signal.',
  'feel.div': 'Number of beats per step of this body’s motion clock.',
  'feel.lock': 'Locks the motion clock to the song beat grid; off lets it run freely.',
  'tone.sat': 'Saturation of the entire finished picture. Lower values mute all colours; higher values make them more vivid.',
  'tone.exposure': 'Overall brightness of the finished picture.',
  'tone.contrast': 'Contrast of the finished picture, separating dark and bright areas.',
  'tone.bloom': 'How much bright areas bleed into a soft glow.',
  'tone.adapt': 'How quickly exposure adapts to changes in brightness.',
  'tone.vignette': 'How much the edges of the picture darken.',
  'tone.ca': 'Chromatic aberration: how far the colour channels separate at the edges.',
  'tone.reflect': 'Mirrors the lower part of the picture like a reflection.',
  'tone.reflectY': 'Vertical position where the reflection starts.',
  'tone.tonemap': 'Choose filmic colour mapping or flame-style log-density mapping.',
  'tone.relief': 'Emboss the finished picture as a lit surface; zero turns it off.',
  'tone.bump': 'Apparent height of that embossed surface.',
  'tone.light': 'Direction of the light across the embossed surface, in turns.',
  'tone.gloss': 'Strength and sharpness of the surface highlight.',
  'tone.metal': 'How much the embossed surface reflects palette colours like metal.',
  'tone.huemap': 'Map brightness to cycling palette colours; zero turns the effect off.',
  'tone.bands': 'Number of palette colour cycles across the brightness range.',
  'tone.drift': 'How quickly the hue-map bands scroll.',
  'tone.solar': 'Solarize the brightness before hue mapping.',
  'tone.poster': 'Flatten hue-map colours into discrete steps.',
  'carrier.halfLife': 'Seconds for an old frame’s light to fade to half brightness. Higher values make longer trails.',
  'carrier.floor': 'Black-level cut each frame. Higher values erase dim trails sooner.',
  'carrier.blur': 'How much the carried picture softens each frame.',
  'carrier.sharpen': 'Sharpens carried edges while fading flat areas, creating reaction-diffusion texture.',
  'carrier.grain': 'Blur radius used to extract edges for sharpening.',
  'carrier.border': 'Strength of a coloured border injected into the feedback each frame.',
  'carrier.amount': 'How strongly the fluid carries the previous frame.',
  'carrier.vort': 'Swirl strength of the simulated fluid.',
  'carrier.fnoise': 'Turbulence in the fluid velocity field.',
  'carrier.fscale': 'Scale of the fluid turbulence pattern.',
  'carrier.famt': 'Amount of fluid turbulence added to the flow.',
  'carrier.water': 'Strength of beat-triggered ripples that refract the carried picture.',
  'carrier.wsize': 'Radius of the water ripple drops.',
  'palette.hue': 'Offset of the first palette colour around the colour wheel.',
  'palette.spread': 'How far apart the three palette colours sit.',
  'palette.key': 'How much the palette follows the song key; zero keeps the colours fixed.',
  'color.amount': 'How strongly this colour mapping moves through the palette.',
  'color.detail': 'How much the body’s own depth, position or detail changes its colour.',
  'material.gain': 'Brightness of this body before it is combined with the picture.',
  'material.blend': 'How this body combines with the picture below it: add, max, subtract, xor, screen or interlace.',
  'fuse.mode': 'How the second shape joins the first: union, morph or an illuminated region.',
  'fuse.k': 'Softness of the boundary where two shapes blend.',
  'fuse.t': 'Mix between the first and second shapes.',
  'fuse.drive': 'What moves the shape mix over time: sweep or a music signal.',
  'fuse.depth': 'How much of the second shape is revealed.',
  'fuse.rate': 'Number of bars per automatic shape sweep.',
  'fuse.inside': 'Whether the second shape lights the interior or the boundary.',
  'energy.lo': 'Lowest song energy this preset prefers.',
  'energy.hi': 'Highest song energy this preset prefers.',
  'xform.a': 'Horizontal scale of this flame transform.', 'xform.b': 'Horizontal shear of this flame transform.',
  'xform.c': 'Vertical shear of this flame transform.', 'xform.d': 'Vertical scale of this flame transform.',
  'xform.e': 'Horizontal offset of this flame transform.', 'xform.f': 'Vertical offset of this flame transform.',
  'xform.weight': 'How often flame particles choose this transform.', 'xform.color': 'Palette position assigned to this transform.',
  'xform.spin': 'Bar-locked rotation of this transform.', 'xform.bass': 'How much bass expands this transform.',
  'xform.pulse': 'Scale kick on each beat.', 'xform.dx': 'Horizontal translation drift over bars.',
  'xform.dy': 'Vertical translation drift over bars.',
};

const COMMON_HELP: Record<string, string> = {
  r: 'Radius of the drawn shape.', radius: 'Radius or reach from the centre.', size: 'Overall size.', scale: 'Scale of the pattern.',
  count: 'Number of copies, particles or repeated elements.', n: 'Number of sides or repeated elements.',
  x: 'Horizontal position or offset.', y: 'Vertical position or offset.', cx: 'Horizontal centre.', cy: 'Vertical centre.',
  angle: 'Static rotation in turns; one full turn is 1.', tilt: 'Rotation or tilt of the figure.',
  rate: 'Speed of the motion or colour change.', speed: 'Speed of travel or animation.', period: 'Length of one motion cycle.',
  turn: 'How strongly or how often the motion turns.', turns: 'Number of turns or wraps.', spinX: 'Rotation about the horizontal axis.', spinY: 'Rotation about the vertical axis.',
  amp: 'Size of the movement or waveform.', freq: 'How many cycles occur.', pulse: 'Extra change on beats or accents.',
  kick: 'Extra jolt on a drum hit or beat.', flash: 'Brief brightening on an accent.',
  gain: 'Brightness or strength of this effect.', amount: 'Strength of this effect.', amt: 'Strength of this effect.',
  w: 'Strength or blend weight of this operation.', s: 'Scale of this operation.', k: 'Blend width or falloff.',
  t: 'Mix between two states.', q: 'Quantisation or clock mode.',
  width: 'Width of the line, beam or effect.', thick: 'Width of the ribbon or mark.', len: 'Length of the element.', length: 'How far the effect reaches.',
  spread: 'How far apart the parts or copies sit.', spacing: 'Space between repeated elements.', gap: 'Gap between parts.',
  density: 'How densely the elements fill the space.', fill: 'Amount of the shape that is filled.',
  blur: 'Amount of softening.', soft: 'Softness of the shape edge.', glow: 'Strength of the surrounding light halo.',
  halo: 'Spread or brightness of the glow around the shape.', bloom: 'Amount of light bleeding outward.',
  hue: 'Position or offset around the palette colour wheel.', hues: 'Colour spread across the elements.',
  sat: 'Colour saturation, from muted to vivid.', contrast: 'Difference between light and dark.', exposure: 'Overall light level.',
  threshold: 'Minimum level needed to activate the effect.', thr: 'Minimum music level needed to activate it.',
  atk: 'Time to rise after the music gets stronger.', rel: 'Time to fall after the music gets quieter.',
  sens: 'Sensitivity to the source signal.', div: 'Beat spacing of the clock.', lock: 'How tightly motion follows the song beat grid.',
  decay: 'How much of the previous state remains over time.', fade: 'How quickly old marks disappear.',
  drag: 'How strongly motion slows down.', force: 'Strength of the applied force.',
  wander: 'Amount of slow, wandering motion.', drift: 'Slow movement over time.',
  flow: 'Strength or speed of the flow.', warp: 'Amount of spatial distortion.',
  detail: 'Amount of fine structure or variation.', rough: 'Roughness of the surface or pattern.',
  mode: 'Selects the visual or motion mode.', form: 'Selects the form of the drawn shape.',
  source: 'Selects what information drives this effect.', pattern: 'Selects the repeating pattern.',
  res: 'Rendering resolution; higher values usually cost more GPU time.',
};

// Specialized schema keys shared by several genes. Kind-specific glossary text
// is appended when a key has a different meaning in a particular kind.
Object.assign(COMMON_HELP, {
  rings: 'Number of concentric halos around the body.', rgap: 'Spacing between consecutive halos.', rfade: 'Brightness retained by each outer halo.',
  tip: 'Brightness of the moving tip of a trail.', base: 'Baseline, resting brightness or starting position.',
  radial: 'How much the effect follows distance from the centre.', alt: 'Alternate behaviour on successive bars or cycles.',
  clip: 'Height below which this fill is hidden.', outline: 'Brightness or thickness of the shape outline.',
  core: 'Brightness of the shape interior.', height: 'Vertical size or camera height.',
  frame: 'How much section framing changes the camera.', fuse: 'How strongly nearby copies melt together.',
  ra: 'First frequency ratio of the curve.', rb: 'Second frequency ratio of the curve.',
  vx: 'Horizontal motion speed.', vy: 'Vertical motion speed.', shot: 'Which set of section camera framings is used.',
  jitter: 'Amount of small random motion on hits.', fog: 'How strongly distance fades into haze.',
  span: 'Seconds of history visible in the drawing.', now: 'Where the present moment sits in the drawing.',
  ribbon: 'Brightness of the line showing held melody notes.', marks: 'Brightness of marks placed at note starts.',
  rise: 'How far older marks drift.', shimmer: 'Sparkle added by vibrato or movement.',
  lanes: 'Number or speed pattern of parallel lanes.', axis: 'Axis along which the effect acts.',
  rim: 'Brightness of light along an edge or ridge.', step: 'Distance moved on each clock step.',
  hook: 'Strength of accents driven by the recurring riff.', section: 'Strength of changes at section boundaries.',
  drop: 'Strength of the effect at a musical drop.', body: 'Visibility of the source body under its particles or trails.',
  inner: 'Size or visibility of the inner part of the shape.', lattice: 'Geometry of the repeating grid.',
  lit: 'Fraction of grid cells that light up.', links: 'Brightness of lines joining neighbouring points.',
  twinkle: 'Amount of individual element flicker.', blend: 'How strongly shapes or layers combine.',
  spec: 'How strongly spectrum bands affect the effect.', curl: 'Amount of curved or swirling motion.',
  sway: 'Side-to-side motion over the beat.', scene: 'How much the scene changes between sections.',
  snap: 'A quick emphasis when a state or chord arrives.', glide: 'Time or strength of a smooth transition.',
  curve: 'Shape of the response or steering path.', follow: 'How strongly this element follows its driver.',
  s1: 'Offset of the second palette colour from the first.', s2: 'Offset of the third palette colour from the first.',
  sides: 'Number of polygon sides.', cam: 'Camera movement mode.', vary: 'Amount of variation at section changes.',
  ao: 'Ambient shading in creases and corners.', roam: 'How far the camera or object travels.',
  iter: 'Number of iterations; more adds detail and rendering cost.', fold: 'Strength or limit of spatial folding.',
  power: 'Fractal exponent controlling the repeated shape.', settle: 'Time taken to settle into the new state.',
  inst: 'How much each instrument moves its own copy.', xs: 'Sideways range of movement.',
  jump: 'Whether copies jump on hits instead of moving smoothly.', swap: 'Whether instrument copies exchange positions.',
  round: 'Corner roundness.', reach: 'Distance reached by the effect.',
  motion: 'Strength of movement driven by this gene.', src: 'Which audio stem supplies this effect.',
  sheen: 'Metallic highlight driven by bright timbre.', glass: 'Glass-like clarity and glowing rim driven by pure tones.',
  grain: 'Size or amount of visible texture.', velvet: 'Soft bloom driven by breathy sound.',
  edge: 'Outline flash driven by sharp sound attacks.', emboss: 'Strength of surface-like relief shading.',
  path: 'Route or path shape followed by the copies.', accent: 'Strength of accent-driven movement or light.',
  brk: 'How far symmetry breaks on chords away from home.', style: 'Style of deformation or movement.',
  walk: 'Palette shift as chords move away from home.', modHue: 'Hue turn for each fifth of a key change.',
  modTurn: 'World rotation for each fifth of a key change.', calm: 'How much the home chord mutes colour.',
  heads: 'Number of independently moving heads.', every: 'Number of beats between turns or events.',
  square: 'Whether turns snap to right angles.', wrap: 'Whether motion wraps at the screen edge.',
  onDrop: 'Behaviour triggered by a structural drop.', trail: 'Length or brightness of the trail left behind.',
  ground: 'Visual style of the terrain surface.', mark: 'Type of landmark at each section start.',
  look: 'How much of the song ahead is visible.', wind: 'Sideways meander of the path.',
  tint: 'How much the song key colours the scene.', zoomFlow: 'How much the particle flow inherits the carrier zoom.',
  lift: 'Amount of upward movement or brightness lift.', life: 'Lifetime of each particle.',
  surge: 'Extra burst on energetic moments.', top: 'Whether particles appear above or below the body.',
  bins: 'Number of spectrum frequency bins drawn.', solid: 'Which three-dimensional solid is drawn.',
  family: 'Family of three-dimensional curve.', p: 'First frequency or winding number of the curve.',
  audio: 'How strongly waveform or spectrum pushes the curve.', persp: 'Strength of perspective depth.',
  sa: 'Angle at which slime agents sense nearby trails.', sd: 'Distance at which slime agents sense nearby trails.',
  steer: 'How strongly slime agents turn toward sensed trails.', deposit: 'Amount of trail deposited by each agent.',
  diffuse: 'How quickly deposited trails blur outward.', feed: 'How strongly the source shape feeds new trails.',
  birth: 'How often agents are reborn at the source shape.',
  wD: 'Share of ecosystem agents assigned to drums.', wB: 'Share assigned to bass.',
  wV: 'Share assigned to vocals.', wO: 'Share assigned to other instruments.',
  glyph: 'Visual symbol used for the agents.', predation: 'Strength of predators chasing other agents.',
  graze: 'Strength of grazing interaction.', growth: 'Rate at which active populations grow.',
  starve: 'Rate at which quiet populations shrink.', field: 'Visibility of the growth field.',
  kinds: 'Which kinds of changes this gene may make.', what: 'Which part of the picture this gene changes.',
  ret: 'Whether returning sections return to their earlier look.', morph: 'Time taken to blend between changed looks.',
  bound: 'Maximum distance from the saved preset.', seed: 'Seed choosing this preset’s repeatable variation.',
  swing: 'How much the measured swing changes motion timing.', sub: 'Which subdivision pairs receive swing.',
  off: 'Strength of the swung offbeat pulse.', lean: 'How much the backbeat drags or leads motion.',
  crisp: 'How strongly timing holds and then snaps to the next tick.', tick: 'Number of motion ticks per beat.',
  recall: 'How strongly a returning section recalls its first picture.', evolve: 'How much each return changes further.',
  keep: 'Whether every appearance updates the remembered scene.', cap: 'Number of remembered scenes kept.',
  min: 'Minimum similarity needed to count as a return.', chrome: 'How mirror-like the surface looks.',
  zoom: 'How far the scene appears zoomed in.', rounds: 'Number of iterations used to build the pattern.',
  breathe: 'Zoom pulse caused by bass.', depth: 'Apparent depth of the scene.',
  twist: 'Amount of spiral distortion.', rep: 'Number of repeats around the shape.',
  lead: 'Bars of buildup before a drop.', push: 'Amount of camera push or spatial shift.',
  roll: 'Camera rotation in turns.', drain: 'How much colour drains before a drop.',
  dim: 'How much the picture darkens before a drop.', punch: 'Strength of the drop impact.',
  relax: 'Bars taken to settle after a drop.', dolly: 'Slow camera push across a section.',
  arc: 'Zoom swell over each phrase.', phrase: 'Length of a musical phrase in bars.',
  side: 'Edge or side on which the shape appears.', lines: 'Amount of line detail.',
  wall: 'Strength of a wall or boundary.', var: 'Amount of colour or shape variation.',
  strength: 'Overall strength of this gene.', pal: 'How much lyrics steer the palette.',
  tone: 'How much lyrics steer brightness and colour tone.', chain: 'How much lyrics steer space-chain effects.',
  lag: 'Delay before the effect follows the source.', show: 'How lyrics appear as captions.',
  smear: 'How strongly lyric text is drawn into the feedback.',
  bands: 'Number of bands in the pattern.', tempo: 'How strongly tempo moves the pattern.',
  melHue: 'How strongly melody changes the hue.', fall: 'How quickly rays fade away.',
  rays: 'Number of visible light rays.', wav: 'Amount of waviness in the rays.',
  tex: 'Texture pattern used by the material.', shape: 'Shape of the repeated element.',
  fan: 'How far light beams spread apart.', sweep: 'How far the beams swing.',
  haze: 'Density of illuminated smoke around beams.', gobo: 'Pattern projected inside each beam.',
  flare: 'Strength of the lens glow.', trig: 'Which musical events step or chase the beams.',
  plate: 'Shape of the vibrating cymatics plate.', modes: 'Highest vibration mode used.',
  hold: 'Bars to keep the current figure.', sand: 'Sand-like grains versus smooth lines.',
  line: 'Width or brightness of a line.', shake: 'Bass-driven jitter of the sand.',
  lobes: 'Number of lobes in the outline.', beat: 'Extra pull on each beat.',
  strips: 'Number of strips in the visual pattern.', win: 'Brightness of window-like details.',
  sky: 'Brightness of the sky or background.', peaks: 'Height of the terrain peaks.',
  terrain: 'Shape of the terrain ridges.', align: 'How strongly agents match nearby headings.',
  cohere: 'How strongly agents gather toward neighbours.', separate: 'How strongly agents avoid crowding.',
  home: 'Pull back toward the source body or home state.', over: 'Amount of overlap.',
  osize: 'Size of the orbit or object.', nodes: 'Brightness of lattice note nodes.',
  echo: 'Brightness of the same chord elsewhere in the lattice.',
  fscale: 'Scale of the folded or noisy pattern.', relief: 'How strongly terrain height or surface shading is emphasized.',
});

function context(g: Genome, t: Target): { group: string; kind: string; label: string } {
  switch (t.t) {
    case 'locus': {
      const kind = g.bodies[t.b][t.locus].kind;
      return { group: t.locus, kind, label: `Body ${t.b + 1} ${t.locus} (${kind})` };
    }
    case 'fuseShape': return { group: 'shape', kind: g.bodies[t.b].fuse?.shape.kind ?? '', label: `Body ${t.b + 1} fused shape` };
    case 'fuse': return { group: 'fuse', kind: 'fuse', label: `Body ${t.b + 1} fused shapes` };
    case 'drawOp': return { group: 'op', kind: g.bodies[t.b].deform.ops?.[t.j]?.op ?? '', label: `Body ${t.b + 1} deform operation ${t.j + 1}` };
    case 'op': return { group: 'op', kind: g.chain[t.j]?.op ?? '', label: `Space-chain operation ${t.j + 1}` };
    case 'xform': return { group: 'xform', kind: '', label: `Body ${t.b + 1} flame transform ${t.j + 1}` };
    case 'reaction': return { group: 'reaction', kind: '', label: `${g.reactions[t.j]?.src ?? 'Music'} reaction` };
    case 'gene': return { group: 'gene', kind: t.key, label: genomeGene(t.key)?.title ?? t.key };
    default: return { group: t.t, kind: t.t === 'palette' ? g.palette.kind : t.t === 'carrier' ? g.carrier.kind : '', label: t.t };
  }
}

export function kindHelp(group: string, kind: string): string {
  if (group === 'feel') return kind === 'step' ? 'Samples the music on its beat clock and holds each value until the next step, for a crisp feel.' : 'Follows the music continuously through an attack and release envelope.';
  const entry = ENTRIES[`${group}.${kind}`] ?? (group === 'gene' ? genomeGene(kind)?.glossary : undefined) ??
    (group === 'op' && kind.startsWith('v_') ? ENTRIES['op.v_'] : undefined);
  return entry ?? `${kind} is a ${group} setting.`;
}

export function sectionHelp(g: Genome, t: Target): string {
  const c = context(g, t);
  const summary = c.kind ? kindHelp(c.group, c.kind) : SECTION_HELP[c.group] ?? '';
  return `${c.label}: ${summary}`;
}

export function paramHelp(g: Genome, t: Target, c: Pick<ParamControl, 'key' | 'label' | 'spec' | 'options'>): string {
  const ctx = context(g, t);
  const key = c.key;
  const special = PARAM_HELP[`${ctx.group}.${ctx.kind}.${key}`] ?? PARAM_HELP[`${ctx.group}.${key}`];
  const explanation = special ?? (key.startsWith('var.') ? `Strength of the ${key.slice(4)} flame variation.` : COMMON_HELP[key]) ?? `Controls ${c.label} in ${ctx.label}.`;
  const choices = c.spec.choices ? ` Choices: ${c.options ? c.options.map((o) => o.label).join(', ') : c.spec.choices.join(', ')}.` : ` Range: ${c.spec.min} to ${c.spec.max}.`;
  const contextNote = ctx.kind ? kindHelp(ctx.group, ctx.kind) : SECTION_HELP[ctx.group] ?? '';
  const extra = special ? '' : ` ${contextNote.length > 220 ? `${contextNote.slice(0, 217)}…` : contextNote}`;
  return `${ctx.label} · ${c.label}: ${explanation}${extra}${choices} Default: ${c.spec.def}.`;
}

export function reactionTargetHelp(t: ReactTarget, source?: Signal): string {
  const section = t.group.replace(/ \(.*\)$/, '');
  const target = `${section} · ${labelFor(t.k)}`;
  const description = t.g === 'col' ? PARAM_HELP[`tone.${t.k}`] :
    t.g === 'car' ? PARAM_HELP[`carrier.${t.k}`] :
    t.g === 'pal' ? PARAM_HELP[`palette.${t.k}`] :
    t.g === 'ma' ? PARAM_HELP[`material.${t.k}`] :
    COMMON_HELP[t.k];
  const action = source ? ` The ${source} signal changes this setting; gain sets the amount.` : '';
  return `${target}: ${description ?? `Changes ${labelFor(t.k)} in ${t.group}.`}${action}`;
}

export function reactionHelp(g: Genome, index: number): string {
  const r = g.reactions[index];
  if (!r) return SECTION_HELP.reaction;
  const target = reactionTargets(g).find((x) => x.g === r.g && x.i === r.i && x.k === r.k);
  return `${r.src}: ${SIGNAL_HELP[r.src]} Drives ${target ? reactionTargetHelp(target) : targetLabel(g, r)} Gain sets how strongly it changes.`;
}
