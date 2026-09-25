// Plain-language reference for the gene chat's system prompt: what each gene does on screen
// (INTRO always sent, ENTRIES keyed per gene kind and sent only for kinds present in the
// current preset), everyday words mapped to gene edits (LEXICON) and value-choosing rules (RULES).

/** Always sent: what a preset is made of, the section notes that apply to every preset (chain
 * stages, carrier params, colour and tone params, feel, reaction signals), kept short. */
export const INTRO = `
BODIES (1-3): each has shape, place, motion, deform, material, emit, feel, color(mapping), optional fuse.
Field shapes(plasma,aurora,terrain,edge,beams,scene,cells)+flame always draw 1 copy regardless of place count.
Only 1 flame + 1 solid + 1 scene shape per preset(each unique, at most 1 of that kind).

CHAIN (up to 6 whole-scene ops + each body's own up to 3 draw-space deform ops, from the same op
kinds): warp-stage feeds back frame-to-frame(trails/streaming); view-stage just reshapes the
picture(no accumulation). Motion ops(zoom,rotate,translate,swirl,twist,ripple,noise,push,quad) are
warp-only; folds(mirror,tile,polar,kaleido,stretch,tunnel,mosaic)+v_ flame variations can run in either stage.

CARRIER(always present): halfLife=trail length(s to fade to half). floor=black-level cut/frame
(higher=trails vanish faster). blur=softens image. sharpen=edges sharpen/flats fade, grows
reaction-diffusion texture. border=coloured edge frame injected each frame, swells w/bass, carried
inward. Fluid carrier only: amount/vort/fnoise/fscale/famt=advection+vorticity/turbulence.
water(0=off)=beat drops spread ripples refracting the picture; wsize=drop radius.

PALETTE(3 hue slots around the song key, always present): hue=1st slot's offset from key;
spread=how far the slots spread.

COLOUR MAPPING(per body, common params): amount=how much the driver swings the hue; detail=how
much the shape's own shading(depth,bins,curve position) varies it.

MATERIAL: all have gain=brightness and blend=how it lands on the picture below(0 add,1 max,2
subtract,3 xor,4 screen,5 interlace); fill/textured/chrome hold steady brightness, line/glow/dots
accumulate into the trail.

TONE(whole-scene post, always present): sat=saturation; exposure=brightness; contrast; bloom=glow
bleed; adapt=eye-adapt speed; vignette=edge darkening; ca=chromatic aberration;
reflect/reflectY=mirror bottom of frame; tonemap=filmic or flame(log-density); relief=emboss the
picture as a lit surface(0 off); bump=surface height; light=light direction; gloss=shine;
metal=liquid-chrome palette reflections;
huemap=map brightness to cycling palette bands(0 off); bands=cycles; drift=scroll speed; poster=flat steps;
solar=solarize(bright folds back).

FEEL(per body, response curve+clock, always present): flow=levels follow music continuously;
step=sampled+held on clock grid(staccato). atk/rel=rise/fall time(s); thr=signal below ignored;
sens=sensitivity. div=clock unit in beats(0.5..16). lock: 1=locked to song grid(mechanical),
0=free-running(organic).

REACTIONS(signal->1 parameter+gain; up to 6; 1 reaction/parameter): drums/bass/vocals/other=
instrument level. hit=drum trigger. beat=pulse/beat. bar=slow wave/bar. complexity=mix busyness.
drop=structural drop. loud=overall loudness. melody=melody activity. build=tension pre-drop.
surge=beat envelope, cruises w/loudness, jumps on drops. barpulse=pulse/downbeat. section=pulse on
section change. tension=harmonic tension 0..1(chord far from home key/dissonant). resolve=pulse when
harmony resolves to the tonic(V-I etc, sized by tension released). chordchange=pulse per chord change.
modulation=pulse on a key change. swing=how swung the playing is(0 straight..1 triplet). push=how far
the backbeat leans off the grid(laid back or pushed). humanity=how loose/human the timing is. synco=
syncopation density(hits on weak off-beats).
line=pulse per sung lyric line(vocal entry without lyrics).
valence=mood of the words 0 sad..1 happy(music's mood without lyrics). arousal=intensity of the words
0 calm..1 intense(music's activity without lyrics).
`;

/**
 * One entry per gene kind, keyed '<group>.<kind>'. group: a body locus (shape, place, motion, deform,
 * material, emit, color), 'op' (space-chain ops, including 'op.v_' for all flame variations as one entry),
 * 'carrier', 'palette', 'fuse' (one entry 'fuse.fuse') or 'gene' (an optional genome-wide gene, e.g. 'gene.choreo').
 * The text is what the kind looks like on screen plus its non-obvious params, as in the old GLOSSARY.
 */
export const ENTRIES: Record<string, string> = {
  // -------------------------------------------------------------- shape
  'shape.dot': 'disc.',
  'shape.polygon': 'n sides; round=corner round.',
  'shape.star': 'inner=inner radius, spikier when low.',
  'shape.segment': 'line stroke.',
  'shape.solid': 'solid(3D,1/preset): solid 0 tetra,1 cube,2 octa,3 icosa,4 polygon(sides),5 by section; tilt=3D tilt; inner=inner faces visible.',
  'shape.bars': 'mode 0 baseline,1 arc,2 ring,3 mirrored; fill=duty cycle.',
  'shape.curve': 'form 0 wave,1 circle,2 spiral,3 lissajous,4 arc,5 harmonograph; turns=spiral wraps; ra/rb=lissajous ratio.',
  'shape.plasma': 'plasma(field): warp=distortion; bands=colour bands; lines=line vs smooth; tempo/pulse/melHue=music-driven colour shift.',
  'shape.aurora': 'fall=ray falloff; wav=waviness.',
  'shape.terrain': 'terrain(hills over a valley): peaks=peak height; terrain=ridge shape; flash=downbeat flash.',
  'shape.edge': 'edge(edge strip): mode 0 skyline,1 melody,2 rain,3 ridge; side 0 right,1 top,2 bottom,3 left.',
  'shape.flame': 'flame(particle cloud,1/preset): rounds=iterations; flow=variation morph cycles/8 bars; breathe=zoom pulse per bass.',
  'shape.superscope': 'superscope(3D point curve): family 0 torus knot,1 sphere spiral,2 rose,3 coiled ring,4 lissajous,5 ring tunnel; p/q=frequencies; audio=wave/spectrum(spec) push; spinX/spinY=tumble turns/bar; persp=depth; n=points.',
  'shape.beams': 'beams(concert light shafts through haze, rig at the placement): count=heads; spread=truss length; fan=aim spread; sweep=swing size; pattern 0 unison,1 scissor,2 chase wave,3 alternate,4 step; period=bars per sweep; width=beam cone; haze=smoke density; gobo 0 open,1 breakup,2 ring,3 textured; hues=colour step per head; length=reach; flare=lens glow; accent=beat chase.',
  'shape.scene': 'scene(ray-marched 3D,1/preset): scene 0 melting shapes,1 endless lattice(gap=corridor,spec=cells rise w/spectrum),2 ribbed tunnel(gap=bend,spec=ribs close in),3 mandelbox,4 mandelbulb,5 menger(iter=detail+cost; fscale/fold=box fold scale/limit; power=bulb power); cam 0 orbit,1 fly(dives into fractals),2 dolly zoom; roam=camera travel; res=render res; size; blend=melt; pulse=bass swell; kick=drum jolt; vary=section reshuffle; rim/ao/fog/glow=light.',
  'shape.cymatics': 'cymatics(Chladni sand plate, 1 copy): plate 0 square,1 round; size; modes=max mode number; source 0 chords,1 spectrum bands,2 section; hold=bars per figure; settle=beats to re-form; sand=grains vs lines; line=width; shake=bass jitter; rim=plate edge.',
  'shape.cells': 'cells(Voronoi cell foam): mode 0 foam,1 veins,2 domes; warp=bent walls; fill=cell light; var=hue spread; pulse=beat pop.',
  'shape.landscape': 'landscape(the song as terrain the camera travels, 1/preset; knows the song ahead): altitude=energy, each drop a climb to a mountain pass with a sun over it, breakdowns=valleys, repeated sections reuse terrain; path 0 road,1 river,2 flight,3 rail; ground 0 hills,1 crystal,2 dunes,3 city blocks,4 ribbons; mark(landmark at each section start) 0 obelisks,1 gates,2 rings,3 beacons; look=seconds of song visible ahead; height=camera height; relief=energy->altitude; rough=terrain detail; wind=path meander; fog=horizon haze; glow=path lights+landmarks+drop sun; tint=key/mode tints the ground; kick=drum camera bump; rim=ridge light; res=render res(cost).',

  // -------------------------------------------------------------- place
  'place.point': 'fixed spot.',
  'place.orbit': 'copies circle a centre; follow=centre wander; rate=turns/bar; fuse=copies melt together.',
  'place.walker': '1-2 roaming heads; step=move/beat; every=beats/turn; square=90-deg turns.',
  'place.stations': '1 copy/instrument; inst=how much they move; xs=sideways roam; jump=snap on hit; swap=slots swap.',
  'place.row': 'copies along the bottom.',
  'place.float': 'copies drift on slow Lissajous paths; spread=area size.',
  'place.outline': 'copies travel a path; path 0 circle,1 polygon,2 figure-8.',
  'place.grid': 'grid(lattice fold, not real copies): lattice 0 square,1 hex,2 triangle; density=fraction lit; links=lines to lit neighbours; lock=turns/bar.',
  'place.ring': 'n copies on a ring.',
  'place.mirror': 'mirrors 1 copy across axis 0 vert,1 horiz,2 both.',

  // ------------------------------------------------------------- motion
  'motion.none': 'static.',
  'motion.spin': 'rate=turns/bar; alt=reverse alt bars; solids spin in 3D.',
  'motion.sway': 'side swing, period=bars.',
  'motion.bob': 'drifts/bar, bobs on beat.',
  'motion.drift': 'constant slow velocity, wraps at edges.',
  'motion.circle': 'small loop, period=bars.',
  'motion.hits': 'jolts/turns on drum hits.',
  'motion.pulse': 'size kick on beat.',

  // ------------------------------------------------------------- deform
  'deform.none': 'no deform.',
  'deform.arms': 'tapered curling arms reaching per instrument+loudness; curl=amount; turn=bars/direction change.',
  'deform.wobble': 'wobbling outline lobes; rate=turns/bar, swells w/bass.',
  'deform.noise': 'organic outline jitter.',
  'deform.twist': 'rotation grows w/distance from centre, breathes w/loudness.',

  // ------------------------------------------------------------ material
  'material.line': 'glow outline; width(px); halo=glow spread.',
  'material.fill': 'soft=edge softness; outline=cell outline; core=inner gradient; clip=hide below height.',
  'material.glow': 'soft blob, grows w/level; base=brightness at silence.',
  'material.dots': 'stippled fill.',
  'material.textured': 'tex 0 craters,1 stripes(sunset),2 windows.',
  'material.chrome': 'mirror-like; chrome 0 plastic..1 mirror.',

  // --------------------------------------------------------------- emit
  'emit.none': 'no trail.',
  'emit.trail': 'leaves light in feedback; tip=extra bright point at current position.',
  'emit.cover': 'paints over old trail instead of adding light; amt=opacity.',
  'emit.dye': 'pushes into fluid carrier on hits/beats.',
  'emit.sparks': 'sparks(1/preset): curl=swirliness; zoomFlow=inherits carrier zoom; life=lifetime; surge=extra burst; top=above/below body; body=source shape visibility.',
  'emit.slime': 'slime(1/preset): physarum agents grow glowing vein networks; count=agents; sa/sd=sensor angle/distance(sd=cell size); steer=steering; step=speed; deposit=trail laid; decay=trail kept/frame; diffuse=blur; body=source shape visibility; feed=the shape seeds veins; birth=agents reborn at the shape; onDrop 0 none,1 scatter,2 burst from the shape.',
  'emit.flock': 'flock(1/preset): boids circling the body; speed; radius=neighbourhood; align/cohere/separate=flocking rules; wander; home=pull to body; size=bird px; onDrop 0 none,1 burst outward,2 restart at body.',
  'emit.ecosystem': 'ecosystem(1/preset): instrument species of agents share a growth field; drums=predators chase plankton+dart on hits, bass=grazers eat flora+leave trails, vocals=pollinators bloom flora, other=plankton drift+get eaten; a species grows while its stem plays, starves when silent; count=agents; wD/wB/wV/wO=roster shares; glyph 0 creatures,1 points,2 comets,3 sigils; size; trail=streak length; predation/bloom/graze=interaction strengths; growth/starve=population rates; decay=flora kept/frame; speed; field=flora visibility; hues=species-palette rotation; body=source shape visibility.',

  // -------------------------------------------------------------- color
  'color.fixed': '1 hue.',
  'color.instrument': 'each copy=its instrument slot.',
  'color.pitch': 'chroma around the key.',
  'color.melody': 'follows the melody line.',
  'color.height': 'driven by vertical position.',
  'color.age': 'hue drifts w/time; rate=palette turns/bar.',
  'color.speed': "faster copies=more hue shift.",

  // ---------------------------------------------------------------- op
  'op.zoom': 'rate>0 flies outward,<0 inward.',
  'op.rotate': 'lock=bar-locked turns/bar; rate=free spin.',
  'op.translate': 'push; lanes=columns alternating 1x/2x speed.',
  'op.swirl': 'rotates near cx,cy; amt=strength+dir; k=falloff.',
  'op.twist': 'like swirl but grows w/distance.',
  'op.ripple': 'concentric/radial waves.',
  'op.noise': 'organic warble.',
  'op.push': 'shift along axis(x,y,radial).',
  'op.quad': 'z^2 warp, Julia-set-like coastlines.',
  'op.mirror': 'mirror(fold): reflects across axis.',
  'op.tile': 'tile(fold): repeats in a grid; n=cells/unit.',
  'op.polar': 'polar(fold): polar-coord map, radial symmetry.',
  'op.kaleido': 'kaleido(fold): n-fold mirror.',
  'op.mosaic': 'mosaic(fold): the picture in blocks; size=cell size; shape 0 square tiles,1 round LEDs,2 hexagons; gap=dark grout(view stage); angle/lock=grid turn; pulse=cells swell with bass.',
  'op.tunnel': 'tunnel(fold): the picture wrapped on a tunnel wall flown through; depth=tunnel size; speed=flight(neg=backward); twist=spiral; sides 0 round/3-8 polygon; rep=repeats round the wall; fog=dark far end.',
  'op.stretch': 'spectrum stretch; beat=extra pull on beat.',
  'op.v_': 'v_<name>(flame variation): bends space via that flame function; w=blend; s=scale.',

  // ----------------------------------------------------------- carrier
  'carrier.warp': 'feedback+chain: the existing picture warps and re-accumulates every frame(trails/streaming).',
  'carrier.fluid': 'velocity-advected: light is carried through a simulated fluid field.',
  'carrier.flow': 'flow-field: light drifts along a fixed flow field(no fluid sim).',
  'carrier.none': 'no persistence: each frame is redrawn fresh.',

  // ----------------------------------------------------------- palette
  'palette.analogous': '3 hue slots close together.',
  'palette.complementary': '3 hue slots, 2nd opposite the key.',
  'palette.triad': '3 hue slots evenly spaced.',
  'palette.split': '3 hue slots: the key\'s complement plus its neighbours.',
  'palette.mono': '3 hue slots near 1 hue.',
  'palette.free': "3 independently placed hue slots; hue=1st slot's offset, s1/s2=other slots' offsets from the first.",

  // ------------------------------------------------------------- fuse
  'fuse.fuse': 'FUSE(2nd shape merged in): mode 0 union(k=blend radius, grows w/bass),1 morph(t=mix),2 region(lit inside/along other shape). drive=what moves t(none,sweep,bass,melody,loud,surge).',

  // ------------------------------------------------------------- gene
  'gene.choreo': "CHOREO(optional, knows the song ahead): before drops the camera pushes in and leans, colour drains, light dims; slams on the drop, settles. lead=bars of build-up; curve=how late it bites; push=zoom-in; roll=lean(turns); drain=desaturate; dim=darken; punch=drop slam; relax=bars to settle. Each section type gets its own framing; frame=how far framings push/pan/lean; shot=which set of framings; glide=bars into new framing(0=cut); dolly=slow push-in across each section; scene=hue shift per section type. arc=zoom swell per phrase(phrase=bars).",
  'gene.harmony': 'HARMONY(optional, follows the chord progression): consonance=symmetric+calm; rising harmonic tension breaks symmetry: brk=how far mirror/tile/polar/kaleido folds slide out of register; warp=lopsided whole-frame warp, style 0 lean,1 off-centre swirl,2 buckle; a resolution to the home chord snaps it back: snap=kick+flash strength, settle=seconds to click back(short=click,long=wobble). walk=palette hue shift as chords move from home; kick=zoom pulse per chord change; modHue=hue turn per fifth on key change; modTurn=world roll per fifth(turns); calm=consonance mutes colour, tension saturates.',
  'gene.groove': 'GROOVE(optional): motion takes the music\'s timing feel. Swung music: spins/sways/pulses land late on the off-beat(swing=share of the measured swing, sub 8|16=which pairs); sway=sideways lilt over 2 beats; off=pulse on the swung off-beat; lean=laid-back/pushed backbeat drags/leads motion; quantized music ticks(crisp=hold-then-snap, tick=ticks/beat); human timing: jitter=nudges on each hit; accent=kicks on syncopated hits.',
  'gene.drift': 'DRIFT(optional, performance layer): the preset travels through gene space over the song, each section a small mutation of the last; step=how far per section; morph=bars to morph(0=cut); kinds 0 params only,1 shapes change on drops,2 any section; what 0 all,1 form,2 colour,3 motion; ret=returning section type goes back to its earlier look; bound=max distance from the saved preset; seed=which journey. The saved preset stays the home.',
  'gene.dejavu': 'DEJAVU(optional, knows which sections repeat): a returning section (2nd chorus, riff) recalls its first appearance. recall=how strongly the remembered picture folds back in; blend=bars it takes; snap=where in the first appearance it is remembered; frame=camera returns to the remembered framing; hue=colours return; motion=spins/drifts rewind; evolve=each return more turned/pushed/hue-shifted; keep 1=re-remember every appearance; res=memory sharpness; cap=scenes held; min=how alike a return must be.',
};

export const LEXICON = `
calmer/chill: carrier.halfLife up; op/motion/place rates ->0; material gain x0.7; reactions gain x0.6.
busier: add body/chain op if under caps; place count up; reactions gain up.
faster: op/motion/place rates x1.5-2; feel atk/rel down.
slower: op/motion rates x0.5-0.7; feel atk/rel up.
brighter: tone.exposure x1.15-1.3; material gain up.
darker/dimmer: tone.exposure x0.75-0.85; material gain down.
too dark: tone.exposure+material gain up.
too bright: tone.exposure down; tone.bloom down.
washed out/faded: tone.sat up; tone.contrast up; tone.exposure down a little; tone.bloom down.
more colourful: palette.spread up; tone.sat up; color mapping->instrument/pitch.
less colourful/muted: tone.sat x0.6-0.8; palette.spread down; palette->mono.
warmer: palette.hue "orange" (or "red", "amber").
colder: palette.hue "blue" (or "cyan", "teal").
bigger: shape r/size/radius x1.3-1.5; place spread/radius up.
smaller: shape r/size/radius x0.6-0.75.
sharper/crisper: material width down; carrier.sharpen up; tone.contrast up.
softer/blurrier: carrier.blur up; material width/glow up.
dreamy: carrier.halfLife+blur up; tone.bloom up; feel atk/rel up.
underwater: palette.hue "teal"; add_op ripple (warp); kind carrier fluid; carrier.blur up.
fire: palette.hue "orange"; material=glow/textured tex=1; emit=trail/sparks; tone.exposure up.
space/stars: add_body shape dot place grid (a star field) or emit sparks; add_op zoom with rate>0 (flying through space); palette.hue "blue".
psychedelic/trippy: chain ops up(kaleido,swirl,quad); palette=triad/split high spread; tone.ca up; deform=noise/wobble.
minimal/clean: 1 body; chain ops 0-1; deform=none; reactions 1-2.
aggressive/harder: feel atk down; reactions gain up; motion=hits/pulse; tone.contrast up.
punchier/on the beat: feel.lock=1; feel div=beat/bar; motion=pulse/hits; reaction src=beat/hit fast atk.
react to bass: reaction src=bass ->material gain, shape size, or motion amp.
react to vocals: reaction src=vocals ->color mapping amount or place wander.
build-up to the drop/cinematic: add choreo; push/drain/punch up; lead=bars of tension.
on the drop: reaction src=drop/surge ->tone.exposure or carrier.halfLife, fast atk.
longer trails: carrier.halfLife x1.5-3.
shorter trails: carrier.halfLife x0.3-0.5.
no trails: kind bN.emit none, or kind carrier none.
fill more of the screen: place count up; spread/radius up; shape size up.
emptier/less cluttered: place count (walker: heads) down; remove_body a second body; shorter trails (carrier.halfLife down); shape size down.
more symmetry/kaleidoscope: add fold ops mirror/kaleido/polar, or place=grid/ring/mirror.
less symmetry: remove fold ops; place=point/walker/float.
spin/rotate: motion=spin rate up; or add chain op rotate.
stop spinning: motion=none or spin rate 0; zero rotate/swirl/twist ops.
more particles/sparks: emit=sparks; sparks count x1.5-2.
glow: material=glow; width/halo up; tone.bloom up.
neon: material=glow/line; tone.sat high; tone.bloom up; palette=triad/split high spread.
retro/oscilloscope: shape=curve form=0(wave)/1(circle); material=line; emit=trail.
smoother motion: bN.feel.atk x3 and bN.feel.rel x3 (slower response); bN.feel.lock 0; rates a little down.
jerkier/staccato: feel=step; feel.lock=1; motion=hits; feel atk down.
more depth: material=chrome/textured; tone.reflect on; carrier.sharpen up.
`;

export const RULES = `
Stay inside each parameter's min/max range from the glossary.
"A bit"/"slightly": ~20-30% change(x1.2-1.3 or x0.7-0.8). "Much"/"a lot": ~2x, or near the range's end.
Prefer adjusting existing parameters over switching a gene's kind, unless the request names a new look(e.g. "switch to fire").
Caps: max 3 bodies/genome, 6 reactions, 6 chain ops, 3 deform ops/body.
1 reaction per parameter at a time; retarget or remove the old one before adding to the same target.
Only 1 flame + 1 solid + 1 scene shape per preset. Field shapes(plasma,aurora,terrain,edge,scene)+flame always draw 1 copy regardless of place count.
GPU budget ~8ms/frame; costliest: sparks count, flame count, place copy count, body count. Raise cautiously; lower another cost when adding one.
When unsure which gene a word means, prefer the most direct parameter(e.g. "brighter"=tone.exposure before material gain) and say what changed.
`;
