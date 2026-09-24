// Plain-language reference for the gene chat's system prompt: what each gene does on screen
// (GLOSSARY), everyday words mapped to gene edits (LEXICON) and value-choosing rules (RULES).

export const GLOSSARY = `
BODIES (1-3): shape, place, motion, deform, material, emit, feel, color, optional fuse.

SHAPE:
dot: disc. polygon: n sides; round=corner round. star: inner=inner radius, spikier when low. segment: line stroke.
solid(3D,1/preset): solid 0 tetra,1 cube,2 octa,3 icosa,4 polygon(sides),5 by section; tilt=3D tilt; inner=inner faces visible.
bars: mode 0 baseline,1 arc,2 ring,3 mirrored; fill=duty cycle.
curve: form 0 wave,1 circle,2 spiral,3 lissajous,4 arc,5 harmonograph; turns=spiral wraps; ra/rb=lissajous ratio.
plasma(field): warp=distortion; bands=colour bands; lines=line vs smooth; tempo/pulse/melHue=music-driven colour shift.
aurora: fall=ray falloff; wav=waviness.
terrain(hills over a valley): peaks=peak height; terrain=ridge shape; flash=downbeat flash.
edge(edge strip): mode 0 skyline,1 melody,2 rain,3 ridge; side 0 right,1 top,2 bottom,3 left.
flame(particle cloud,1/preset): rounds=iterations; flow=variation morph cycles/8 bars; breathe=zoom pulse per bass.
superscope(3D point curve, AVS style): family 0 torus knot,1 sphere spiral,2 rose,3 coiled ring,4 3D lissajous,5 ring tunnel; p/q=frequencies; audio=push by waveform or spectrum(spec); spinX/spinY=3D tumble turns/bar; persp=depth; n=points. Never fuses.
Field shapes(plasma,aurora,terrain,edge)+flame draw 1 copy regardless of place count.

PLACE:
point: fixed spot. orbit: copies circle a centre; follow=centre wander; rate=turns/bar; fuse=copies melt together.
walker: 1-2 roaming heads; step=move/beat; every=beats/turn; square=90-deg turns.
stations: 1 copy/instrument; inst=how much they move; xs=sideways roam; jump=snap on hit; swap=slots swap.
row: copies along the bottom. float: copies drift on slow Lissajous paths; spread=area size.
outline: copies travel a path; path 0 circle,1 polygon,2 figure-8.
grid(lattice fold, not real copies): lattice 0 square,1 hex,2 triangle; density=fraction lit; links=lines to lit neighbours; lock=turns/bar.
ring: n copies on a ring. mirror: mirrors 1 copy across axis 0 vert,1 horiz,2 both.

MOTION:
none: static. spin: rate=turns/bar; alt=reverse alt bars; solids spin in 3D. sway: side swing, period=bars.
bob: drifts/bar, bobs on beat. drift: constant slow velocity, wraps at edges. circle: small loop, period=bars.
hits: jolts/turns on drum hits. pulse: size kick on beat.

DEFORM (+ up to 3 draw-space ops/body: swirl,twist,ripple,noise,rotate,zoom,mirror,kaleido,v_<flame var>):
none. arms: tapered curling arms reaching per instrument+loudness; curl=amount; turn=bars/direction change.
wobble: wobbling outline lobes; rate=turns/bar, swells w/bass. noise: organic outline jitter. twist: rotation grows w/distance from centre, breathes w/loudness.

MATERIAL (all have gain=brightness): line: glow outline; width(px); halo=glow spread. fill: soft=edge softness; outline=cell outline; core=inner gradient; clip=hide below height.
glow: soft blob, grows w/level; base=brightness at silence. dots: stippled fill. textured: tex 0 craters,1 stripes(sunset),2 windows. chrome: mirror-like; chrome 0 plastic..1 mirror.
fill/textured/chrome hold steady brightness; line/glow/dots accumulate into the trail.

EMIT: none: no trail. trail: leaves light in feedback; tip=extra bright point at current position. cover: paints over old trail instead of adding light; amt=opacity. dye: pushes into fluid carrier on hits/beats.
sparks(1/preset): curl=swirliness; zoomFlow=inherits carrier zoom; life=lifetime; surge=extra burst; top=above/below body; body=source shape visibility.

FUSE (2nd shape merged in): mode 0 union(k=blend radius, grows w/bass),1 morph(t=mix),2 region(lit inside/along other shape). drive=what moves t(none,sweep,bass,melody,loud,surge).

CHAIN (up to 6 whole-scene ops + each body's own deform ops): warp-stage feeds back frame-to-frame(trails/streaming); view-stage just reshapes the picture(no accumulation). Motion ops warp-only; folds+v_ variations either stage.
zoom: rate>0 flies outward,<0 inward. rotate: lock=bar-locked turns/bar; rate=free spin. translate: push; lanes=columns alternating 1x/2x speed.
swirl: rotates near cx,cy; amt=strength+dir; k=falloff. twist: like swirl but grows w/distance.
ripple: concentric/radial waves. noise: organic warble. push: shift along axis(x,y,radial). quad: z^2 warp, Julia-set-like coastlines.
mirror(fold): reflects across axis. tile(fold): repeats in a grid; n=cells/unit. polar(fold): polar-coord map, radial symmetry.
kaleido(fold): n-fold mirror. stretch: spectrum stretch; beat=extra pull on beat.
v_<name>(flame variation): bends space via that flame function; w=blend; s=scale.

CARRIER: warp(feedback+chain), fluid(velocity-advected), flow(flow-field), none(no persistence).
halfLife=trail length(s to fade to half). floor: black-level cut/frame(higher=trails vanish faster). blur: softens image.
amount/vort/fnoise/fscale/famt: fluid advection+vorticity/turbulence. sharpen: edges sharpen/flats fade, grows reaction-diffusion texture. border: coloured edge frame injected each frame, swells w/bass, carried inward.

COLOUR:
Palette kind(3 hue slots around song key): analogous(close),complementary(opposite),triad(3 even),split(complement+neighbours),mono(near 1 hue),free(3 slots). hue=1st slot's offset from key; spread=how far slots spread.
Colour mapping(per body, drives hue): fixed(1 hue),instrument(each copy=its slot),pitch(chroma around key),melody(follows melody),height(vertical pos),age(hue drifts w/time; rate=palette turns/bar),speed(faster=more hue shift). amount=driver's swing; detail=shape's own shading variance.
Tone(whole-scene post): sat=saturation; exposure=brightness; bloom=glow bleed; adapt=eye-adapt speed; vignette=edge darkening; ca=chromatic aberration; reflect/reflectY=mirror bottom of frame; tonemap=filmic or flame(log-density).

CHOREO(optional, whole song, uses the known future of the song): before each drop the camera pushes in and leans, colour drains and light dims, then it slams on the drop and settles. lead=bars of build-up; curve=how late it bites; push=zoom-in; roll=lean(turns); drain=desaturate; dim=darken; punch=drop slam; relax=bars to settle.

FEEL(per body, response curve+clock): flow=levels follow music continuously; step=sampled+held on clock grid(staccato).
atk/rel=rise/fall time(s); thr=signal below ignored; sens=sensitivity. div=clock unit in beats(0.5..16). lock: 1=locked to song grid(mechanical), 0=free-running(organic).

REACTIONS(signal->1 parameter+gain; up to 6; 1 reaction/parameter):
drums/bass/vocals/other=instrument level. hit=drum trigger. beat=pulse/beat. bar=slow wave/bar. complexity=mix busyness.
drop=structural drop. loud=overall loudness. melody=melody activity. build=tension pre-drop.
surge=beat envelope, cruises w/loudness, jumps on drops. barpulse=pulse/downbeat. section=pulse on section change.
`;

export const LEXICON = `
calmer/chill: carrier.halfLife up; op/motion/place rates ->0; material gain x0.7; reactions gain x0.6.
busier: add body/chain op if under caps; place count up; reactions gain up.
faster: op/motion/place rates x1.5-2; feel atk/rel down.
slower: op/motion rates x0.5-0.7; feel atk/rel up.
brighter: tone.exposure x1.15-1.3; material gain up.
darker/dimmer: tone.exposure x0.75-0.85; material gain down.
too dark: tone.exposure+material gain up.
too bright/washed out: tone.exposure down; tone.bloom down.
more colourful: palette.spread up; tone.sat up; color mapping->instrument/pitch.
less colourful/muted: tone.sat x0.6-0.8; palette.spread down; palette->mono.
warmer: palette hue ->0-0.1(red/orange).
colder: palette hue ->0.55-0.65(blue).
bigger: shape r/size/radius x1.3-1.5; place spread/radius up.
smaller: shape r/size/radius x0.6-0.75.
sharper/crisper: material width down; carrier.sharpen up; tone.contrast up.
softer/blurrier: carrier.blur up; material width/glow up.
dreamy: carrier.halfLife+blur up; tone.bloom up; feel atk/rel up.
underwater: carrier=fluid/flow; carrier.fnoise up; tone.ca up; carrier.blur up.
fire: palette hue 0.02-0.08; material=glow/textured tex=1; emit=trail/sparks; tone.exposure up.
space/stars: place=float/point, small dots; emit=trail; carrier.halfLife long; deform=none.
psychedelic/trippy: chain ops up(kaleido,swirl,quad); palette=triad/split high spread; tone.ca up; deform=noise/wobble.
minimal/clean: 1 body; chain ops 0-1; deform=none; reactions 1-2.
aggressive/harder: feel atk down; reactions gain up; motion=hits/pulse; tone.contrast up.
punchier/on the beat: feel.lock=1; feel div=beat/bar; motion=pulse/hits; reaction src=beat/hit fast atk.
react to bass: reaction src=bass ->material gain, shape size, or motion amp.
react to vocals: reaction src=vocals ->color mapping amount or place wander.
build-up to the drop/cinematic: add choreo; push/drain/punch up; lead=bars of tension.
on the drop: reaction src=drop/surge ->tone.exposure or carrier.halfLife, fast atk.
longer trails: carrier.halfLife x1.5-3.
shorter trails/no trails: carrier.halfLife x0.3-0.5, or emit=cover high amt, or carrier=none.
fill more of the screen: place count up; spread/radius up; shape size up.
emptier/less cluttered: place count down; chain ops down; fewer bodies.
more symmetry/kaleidoscope: add fold ops mirror/kaleido/polar, or place=grid/ring/mirror.
less symmetry: remove fold ops; place=point/walker/float.
spin/rotate: motion=spin rate up; or add chain op rotate.
stop spinning: motion=none or spin rate 0; zero rotate/swirl/twist ops.
more particles/sparks: emit=sparks; sparks count x1.5-2.
glow: material=glow; width/halo up; tone.bloom up.
neon: material=glow/line; tone.sat high; tone.bloom up; palette=triad/split high spread.
retro/oscilloscope: shape=curve form=0(wave)/1(circle); material=line; emit=trail.
smoother motion: feel.lock=0; feel atk/rel up; op/motion rates down.
jerkier/staccato: feel=step; feel.lock=1; motion=hits; feel atk down.
more depth: material=chrome/textured; tone.reflect on; carrier.sharpen up.
`;

export const RULES = `
Stay inside each parameter's min/max range from the glossary.
"A bit"/"slightly": ~20-30% change(x1.2-1.3 or x0.7-0.8). "Much"/"a lot": ~2x, or near the range's end.
Prefer adjusting existing parameters over switching a gene's kind, unless the request names a new look(e.g. "switch to fire").
Caps: max 3 bodies/genome, 6 reactions, 6 chain ops, 3 deform ops/body.
1 reaction per parameter at a time; retarget or remove the old one before adding to the same target.
Only 1 flame + 1 solid shape per preset. Field shapes(plasma,aurora,terrain,edge)+flame always draw 1 copy regardless of place count.
GPU budget ~8ms/frame; costliest: sparks count, flame count, place copy count, body count. Raise cautiously; lower another cost when adding one.
When unsure which gene a word means, prefer the most direct parameter(e.g. "brighter"=tone.exposure before material gain) and say what changed.
`;
