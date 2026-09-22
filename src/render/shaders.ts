// GLSL ES 3.00 sources for the enhanced engine.

const HEAD = /* glsl */ `#version 300 es
precision highp float;
precision highp int;
precision highp sampler2D;
`;

const COMMON = /* glsl */ `
const float PI = 3.14159265359;
const float TAU = 6.28318530718;
float hash12(vec2 p) {
  vec3 p3 = fract(vec3(p.xyx) * 0.1031);
  p3 += dot(p3, p3.yzx + 33.33);
  return fract((p3.x + p3.y) * p3.z);
}
float hash11(float p) {
  p = fract(p * 0.1031);
  p *= p + 33.33;
  p *= p + p;
  return fract(p);
}
vec2 hash22(vec2 p) {
  vec3 p3 = fract(vec3(p.xyx) * vec3(0.1031, 0.1030, 0.0973));
  p3 += dot(p3, p3.yzx + 33.33);
  return fract((p3.xx + p3.yz) * p3.zy);
}
float vnoise(vec2 p) {
  vec2 i = floor(p), f = fract(p);
  vec2 u = f * f * (3.0 - 2.0 * f);
  float a = hash12(i), b = hash12(i + vec2(1, 0)), c = hash12(i + vec2(0, 1)), d = hash12(i + vec2(1, 1));
  return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}
float fbm2(vec2 p) {
  return vnoise(p) * 0.65 + vnoise(p * 2.03 + 17.1) * 0.35;
}
// Divergence-free 2D field from a scalar potential.
vec2 curlNoise(vec2 p, float t) {
  const float e = 0.05;
  vec2 o = vec2(t * 0.13, -t * 0.09);
  float n1 = fbm2(p + o + vec2(0, e)), n2 = fbm2(p + o - vec2(0, e));
  float n3 = fbm2(p + o + vec2(e, 0)), n4 = fbm2(p + o - vec2(e, 0));
  return vec2(n1 - n2, -(n3 - n4)) / (2.0 * e);
}
mat2 rot2(float a) { float c = cos(a), s = sin(a); return mat2(c, s, -s, c); }
float luma(vec3 c) { return dot(c, vec3(0.2126, 0.7152, 0.0722)); }
vec3 hsv2rgb(vec3 c) {
  vec3 p = abs(fract(c.xxx + vec3(0.0, 2.0 / 3.0, 1.0 / 3.0)) * 6.0 - 3.0);
  return c.z * mix(vec3(1.0), clamp(p - 1.0, 0.0, 1.0), c.y);
}
vec3 hueRotate(vec3 c, float a) {
  // Rotation about the grey axis.
  const vec3 k = vec3(0.57735);
  float ca = cos(a);
  return c * ca + cross(k, c) * sin(a) + k * dot(k, c) * (1.0 - ca);
}
`;

/** Fullscreen triangle with neighbour UVs for the fluid stencils. */
export const FULLSCREEN_VS = /* glsl */ `#version 300 es
precision highp float;
uniform vec2 uTexel;
out vec2 vUv, vL, vR, vT, vB;
void main() {
  vec2 p = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2));
  vUv = p;
  vL = p - vec2(uTexel.x, 0.0);
  vR = p + vec2(uTexel.x, 0.0);
  vT = p + vec2(0.0, uTexel.y);
  vB = p - vec2(0.0, uTexel.y);
  gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}`;

const FS_IN = /* glsl */ `
in vec2 vUv, vL, vR, vT, vB;
`;

// ---------------------------------------------------------------- fluid ---

export const SPLAT_FS = HEAD + COMMON + FS_IN + /* glsl */ `
uniform sampler2D uTarget;
uniform float uAspect, uTime, uNoise, uNoiseScale;
uniform int uCount;
uniform vec4 uSplatA[16]; // x, y (uv), fx, fy
uniform vec4 uSplatB[16]; // radius, type (0 dir, 1 radial, 2 swirl), magnitude, -
out vec4 o;
void main() {
  vec2 v = texture(uTarget, vUv).xy;
  for (int i = 0; i < 16; i++) {
    if (i >= uCount) break;
    vec4 a = uSplatA[i];
    vec4 b = uSplatB[i];
    vec2 d = vUv - a.xy;
    d.x *= uAspect;
    float g = exp(-dot(d, d) / b.x);
    vec2 n = d / (length(d) + 1e-4);
    vec2 dir = a.zw;
    if (b.y > 1.5) dir = vec2(-n.y, n.x) * b.z;
    else if (b.y > 0.5) dir = n * b.z;
    v += dir * g;
  }
  v += curlNoise(vUv * vec2(uAspect, 1.0) * uNoiseScale, uTime) * uNoise;
  o = vec4(v, 0.0, 1.0);
}`;

export const CURL_FS = HEAD + FS_IN + /* glsl */ `
uniform sampler2D uVelocity;
out vec4 o;
void main() {
  float L = texture(uVelocity, vL).y;
  float R = texture(uVelocity, vR).y;
  float T = texture(uVelocity, vT).x;
  float B = texture(uVelocity, vB).x;
  o = vec4(0.5 * (R - L - T + B), 0.0, 0.0, 1.0);
}`;

export const VORTICITY_FS = HEAD + FS_IN + /* glsl */ `
uniform sampler2D uVelocity, uCurl;
uniform float uCurlStrength, uDt;
out vec4 o;
void main() {
  float L = texture(uCurl, vL).x;
  float R = texture(uCurl, vR).x;
  float T = texture(uCurl, vT).x;
  float B = texture(uCurl, vB).x;
  float C = texture(uCurl, vUv).x;
  vec2 force = 0.5 * vec2(abs(T) - abs(B), abs(R) - abs(L));
  force /= length(force) + 0.0001;
  force *= uCurlStrength * C;
  force.y *= -1.0;
  vec2 vel = texture(uVelocity, vUv).xy + force * uDt;
  o = vec4(clamp(vel, -2000.0, 2000.0), 0.0, 1.0);
}`;

export const DIVERGENCE_FS = HEAD + FS_IN + /* glsl */ `
uniform sampler2D uVelocity;
out vec4 o;
void main() {
  float L = texture(uVelocity, vL).x;
  float R = texture(uVelocity, vR).x;
  float T = texture(uVelocity, vT).y;
  float B = texture(uVelocity, vB).y;
  vec2 C = texture(uVelocity, vUv).xy;
  if (vL.x < 0.0) L = -C.x;
  if (vR.x > 1.0) R = -C.x;
  if (vT.y > 1.0) T = -C.y;
  if (vB.y < 0.0) B = -C.y;
  o = vec4(0.5 * (R - L + T - B), 0.0, 0.0, 1.0);
}`;

export const SCALE_FS = HEAD + FS_IN + /* glsl */ `
uniform sampler2D uTex;
uniform float uValue;
out vec4 o;
void main() { o = uValue * texture(uTex, vUv); }`;

export const PRESSURE_FS = HEAD + FS_IN + /* glsl */ `
uniform sampler2D uPressure, uDivergence;
out vec4 o;
void main() {
  float L = texture(uPressure, vL).x;
  float R = texture(uPressure, vR).x;
  float T = texture(uPressure, vT).x;
  float B = texture(uPressure, vB).x;
  float div = texture(uDivergence, vUv).x;
  o = vec4((L + R + B + T - div) * 0.25, 0.0, 0.0, 1.0);
}`;

export const GRADIENT_FS = HEAD + FS_IN + /* glsl */ `
uniform sampler2D uPressure, uVelocity;
out vec4 o;
void main() {
  float L = texture(uPressure, vL).x;
  float R = texture(uPressure, vR).x;
  float T = texture(uPressure, vT).x;
  float B = texture(uPressure, vB).x;
  vec2 vel = texture(uVelocity, vUv).xy - vec2(R - L, T - B);
  o = vec4(vel, 0.0, 1.0);
}`;

export const ADVECT_FS = HEAD + FS_IN + /* glsl */ `
uniform sampler2D uVelocity, uSource;
uniform vec2 uSimTexel;
uniform float uDt, uDissipation;
out vec4 o;
void main() {
  vec2 coord = vUv - uDt * texture(uVelocity, vUv).xy * uSimTexel;
  o = texture(uSource, coord) / (1.0 + uDissipation * uDt);
}`;

// ------------------------------------------------------------- feedback ---

const WARP = /* glsl */ `
uniform float uTime;
// Per-preset warp functions. Each returns a sample-space offset.
vec2 warpFn(float fn, vec2 p, float r, float a, vec3 prm) {
  float amt = prm.x, t = uTime * prm.y, sc = prm.z;
  if (fn < 0.5) {         // classic sine warp
    return amt * 0.006 * vec2(sin(p.y * sc * 7.0 + t * 1.3) + sin(p.y * sc * 2.3 - t * 0.7),
                              cos(p.x * sc * 5.0 - t * 1.1) + cos(p.x * sc * 2.9 + t * 0.9));
  } else if (fn < 1.5) {  // spiral: angular shear decreasing with radius
    return (rot2(amt * 0.02 / (r + 0.2)) * p - p);
  } else if (fn < 2.5) {  // tunnel ripple
    return p / (r + 1e-3) * sin(r * sc * 22.0 - t * 3.0) * amt * 0.004;
  } else if (fn < 3.5) {  // petals
    float k = sin(a * floor(sc * 3.0 + 3.0) + t * 1.5);
    return p * (k * amt * 0.012);
  } else if (fn < 4.5) {  // vortex twirl
    return (rot2(amt * 0.04 * smoothstep(0.75, 0.0, r) * sin(t * 0.4 + 0.5)) * p - p);
  } else if (fn < 5.5) {  // fractal waves
    vec2 d = vec2(0.0);
    float f = sc * 4.0, w = 1.0;
    for (int i = 0; i < 4; i++) {
      d += w * vec2(sin(p.y * f + t + float(i) * 1.7), cos(p.x * f - t * 1.3 + float(i) * 2.3));
      f *= 2.03; w *= 0.5;
    }
    return d * amt * 0.0035;
  } else if (fn < 6.5) {  // pulsing lens
    return -p * amt * 0.02 * exp(-r * r * sc * 8.0) * (0.6 + 0.4 * sin(t * 2.0));
  } else {                // flow noise
    float n = fbm2(p * sc * 3.0 + vec2(t * 0.15, -t * 0.11)) * TAU * 2.0;
    return vec2(cos(n), sin(n)) * amt * 0.004;
  }
}
`;

export const FEEDBACK_FS = HEAD + COMMON + WARP + FS_IN + /* glsl */ `
uniform sampler2D uPrevA, uPrevB, uVel, uBC;
uniform vec2 uRes, uSimTexel;
uniform float uAspect, uDt;
uniform float uZoom, uZoomExp, uRot, uBZoom, uBRot;
uniform vec2 uTrans;
uniform vec4 uWarp0;   // fn, amt, speed, scale (outgoing preset)
uniform vec4 uWarp1;   // fn, amt, speed, scale (incoming preset)
uniform float uWarpMix, uFluid, uCouple, uBlur, uDecaySub, uHueDrift;
uniform vec2 uDecay;
uniform vec3 uColA, uColB, uColC;
uniform vec4 uRings[8];     // centre x, y, radius, strength
uniform vec4 uBass;         // radius, sides, angle, intensity
uniform vec4 uVocal;        // intensity, width, phase, -
uniform float uChroma[12];
uniform vec4 uChromaP;      // radius, angle, intensity, sparkle
uniform float uKeyHue, uBCMix, uBuild;
layout(location = 0) out vec4 oA;
layout(location = 1) out vec4 oB;

vec3 samp(sampler2D s, vec2 uv) {
  vec3 c = texture(s, uv).rgb;
  if (uBlur > 0.0) {
    vec2 o = 0.9 / uRes;
    vec3 b = texture(s, uv + o).rgb + texture(s, uv - o).rgb +
             texture(s, uv + vec2(o.x, -o.y)).rgb + texture(s, uv + vec2(-o.x, o.y)).rgb;
    c = mix(c, b * 0.25, uBlur);
  }
  vec2 e = min(uv, 1.0 - uv);
  return c * smoothstep(0.0, 0.004, min(e.x, e.y));
}

float polySd(vec2 p, float n, float r) {
  float an = PI / n;
  float a = atan(p.y, p.x);
  float bn = mod(a, 2.0 * an) - an;
  return length(p) * cos(bn) - r * cos(an);
}

void main() {
  vec2 asp = vec2(uAspect, 1.0);
  vec2 p = (vUv - 0.5) * asp;
  float r = length(p);
  float a = atan(p.y, p.x);

  vec2 w = mix(warpFn(uWarp0.x, p, r, a, uWarp0.yzw), warpFn(uWarp1.x, p, r, a, uWarp1.yzw), uWarpMix);
  float zr = pow(max(r * 2.0, 1e-3), uZoomExp);
  float zA = 1.0 + (uZoom - 1.0) * zr;
  float zB = 1.0 + (uZoom - 1.0) * uBZoom * zr;
  vec2 qA = rot2(-uRot) * (p / zA) + w - uTrans;
  vec2 qB = rot2(-uRot * uBRot) * (p / zB) - w * 0.7 + uTrans;

  // Fluid displacement: ink carried by the velocity field.
  vec2 disp = texture(uVel, vUv).xy * uSimTexel * uDt * uFluid;

  // Coupling: each buffer is pushed along the other's luminance gradient.
  vec2 px = 2.0 / uRes;
  vec2 gA = vec2(luma(texture(uPrevA, vUv + vec2(px.x, 0)).rgb) - luma(texture(uPrevA, vUv - vec2(px.x, 0)).rgb),
                 luma(texture(uPrevA, vUv + vec2(0, px.y)).rgb) - luma(texture(uPrevA, vUv - vec2(0, px.y)).rgb));
  vec2 gB = vec2(luma(texture(uPrevB, vUv + vec2(px.x, 0)).rgb) - luma(texture(uPrevB, vUv - vec2(px.x, 0)).rgb),
                 luma(texture(uPrevB, vUv + vec2(0, px.y)).rgb) - luma(texture(uPrevB, vUv - vec2(0, px.y)).rgb));
  gA /= 1.0 + length(gA);
  gB /= 1.0 + length(gB);

  vec2 uvA = qA / asp + 0.5 - disp + gB * uCouple * 0.006;
  vec2 uvB = qB / asp + 0.5 - disp * 0.8 - gA.yx * vec2(1.0, -1.0) * uCouple * 0.006;

  vec3 A = samp(uPrevA, uvA) * uDecay.x;
  vec3 B = samp(uPrevB, uvB) * uDecay.y;
  if (uHueDrift != 0.0) {
    A = max(hueRotate(A, uHueDrift), 0.0);
    B = max(hueRotate(B, -uHueDrift * 0.7), 0.0);
  }
  A = max(A - uDecaySub, 0.0);
  B = max(B - uDecaySub, 0.0);

  // --- drums: shockwave rings
  for (int i = 0; i < 8; i++) {
    vec4 R = uRings[i];
    if (R.w <= 0.0) continue;
    float d = length(p - R.xy) - R.z;
    float wd = 0.003 + R.z * 0.012;
    float ring = R.w * exp(-d * d / (wd * wd));
    B += uColB * ring;
    A += uColC * ring * 0.35;
  }

  // --- bass: soft glowing polygon rotating one turn per bar
  if (uBass.w > 0.0) {
    vec2 pb = rot2(uBass.z) * p;
    float d = polySd(pb, uBass.y, uBass.x);
    float edge = exp(-abs(d) * 110.0) + 0.25 * exp(-abs(d) * 18.0);
    float fill = smoothstep(0.0, -0.2, d) * 0.03;
    A += uColA * uBass.w * (edge + fill);
  }

  // --- vocals: aurora ribbon
  if (uVocal.x > 0.0) {
    float ph = uVocal.z;
    for (int k = 0; k < 2; k++) {
      float s = k == 0 ? 1.0 : -1.0;
      float y0 = s * (0.16 + 0.05 * sin(p.x * 1.3 + ph * 0.5)) + 0.07 * sin(p.x * 3.1 + ph) + 0.035 * sin(p.x * 7.3 - ph * 1.7);
      float d = (p.y - y0) * s;
      float wv = uVocal.y;
      float core = exp(-d * d / (wv * wv));
      float curtain = exp(-max(d, 0.0) * 9.0) * step(0.0, d) * (0.5 + 0.5 * sin(p.x * 40.0 + ph * 3.0 + d * 30.0));
      vec3 c = mix(uColC, uColB, 0.5 + 0.5 * sin(p.x * 2.5 + ph * 0.8 + float(k) * 2.0));
      A += c * uVocal.x * (core + curtain * 0.35);
    }
  }

  // --- other: chroma dots around a circle + fine sparkle
  if (uChromaP.z > 0.0) {
    for (int i = 0; i < 12; i++) {
      float ang = float(i) / 12.0 * TAU + uChromaP.y;
      vec2 c = uChromaP.x * vec2(cos(ang), sin(ang));
      vec2 d = p - c;
      float v = uChroma[i];
      v = v * v * v;
      float g = exp(-dot(d, d) / (0.00012 + 0.0004 * v));
      B += hsv2rgb(vec3(fract(uKeyHue + float(i) * 7.0 / 12.0), 0.8, 1.0)) * g * v * uChromaP.z;
    }
  }
  if (uChromaP.w > 0.0) {
    vec2 cell = floor(vUv * uRes / 2.0);
    float h = hash12(cell + floor(uTime * 24.0) * 17.31);
    float sp = step(1.0 - uChromaP.w, h);
    B += uColC * sp * 3.0;
  }

  // --- hybrid: classic frame as dye where bright
  if (uBCMix > 0.0) {
    vec3 bc = texture(uBC, vUv).rgb;
    bc = bc * bc;
    float l = luma(bc);
    A += bc * smoothstep(0.08, 0.6, l) * uBCMix;
  }

  oA = vec4(min(A, vec3(64.0)), 1.0);
  oB = vec4(min(B, vec3(64.0)), 1.0);
}`;

// ------------------------------------------------------------- waveform ---

export const WAVE_VS = HEAD + /* glsl */ `
uniform sampler2D uWave;
uniform float uStyle, uAmp, uThick, uAspect, uAngle, uN, uMirrorW;
uniform vec2 uRes;
out float vSide;
out float vK;
out float vInst;
vec2 wpos(float k) {
  if (uStyle < 0.5) {         // line
    float w = texture(uWave, vec2(k, 0.5)).r * uAmp;
    return vec2((k * 2.0 - 1.0) * uAspect * 0.46, w * 0.35);
  } else if (uStyle < 1.5) {  // circle (samples run forward then back so the loop closes)
    float s = 1.0 - abs(2.0 * k - 1.0);
    float w = texture(uWave, vec2(s, 0.5)).r * uAmp;
    float ang = k * TAU + uAngle;
    float rr = 0.26 + w * 0.12;
    return rr * vec2(cos(ang), sin(ang));
  } else if (uStyle < 2.5) {  // spiral
    float w = texture(uWave, vec2(k, 0.5)).r * uAmp;
    float ang = k * TAU * 3.0 + uAngle;
    float rr = 0.03 + k * 0.4 + w * 0.05;
    return rr * vec2(cos(ang), sin(ang));
  } else {                    // dual mirrored
    float w = texture(uWave, vec2(k, 0.5)).r * uAmp;
    return vec2((k * 2.0 - 1.0) * uAspect * 0.46, 0.2 + w * 0.22);
  }
}
void main() {
  int i = gl_VertexID;
  float k = float(i / 2) / (uN - 1.0);
  float side = (i % 2 == 0) ? -1.0 : 1.0;
  float dk = 1.0 / uN;
  vec2 a = wpos(k);
  vec2 b = wpos(min(k + dk, 1.0));
  vec2 c = wpos(max(k - dk, 0.0));
  vec2 t = b - c;
  vec2 n = normalize(vec2(-t.y, t.x) + 1e-6);
  vec2 pp = a + n * side * uThick / uRes.y;
  vInst = 1.0;
  if (gl_InstanceID == 1) {
    pp = uStyle > 2.5 ? vec2(pp.x, -pp.y) : -pp;
    vInst = uMirrorW;
  }
  vSide = side;
  vK = k;
  gl_Position = vec4(pp.x / (uAspect * 0.5), pp.y / 0.5, 0.0, 1.0);
}`.replace('uniform sampler2D uWave;', 'const float TAU = 6.28318530718;\nuniform sampler2D uWave;');

export const WAVE_FS = HEAD + /* glsl */ `
uniform vec3 uColA, uColC;
uniform float uBright;
uniform vec2 uRoute; // weights into A, B
in float vSide;
in float vK;
in float vInst;
layout(location = 0) out vec4 oA;
layout(location = 1) out vec4 oB;
void main() {
  float s = 1.0 - vSide * vSide;
  float core = s * s;
  vec3 c = mix(uColA, uColC, 0.5 + 0.5 * sin(vK * 6.2831 * 2.0)) * uBright * vInst * core;
  c += vec3(1.0) * pow(core, 6.0) * uBright * vInst * 0.25;
  oA = vec4(c * uRoute.x, 1.0);
  oB = vec4(c * uRoute.y, 1.0);
}`;

// ------------------------------------------------------------ particles ---

export const PARTICLE_UPDATE_FS = HEAD + COMMON + /* glsl */ `
uniform sampler2D uS0, uS1, uVel, uWave;
uniform float uDt, uTime, uAspect;
uniform vec2 uSimTexel;
uniform float uFluidAmt, uCurl, uZoomFlow, uRotFlow, uConverge, uDrag, uLifeRate, uSpeed;
uniform vec3 uSpawn;      // mode (outgoing), mode (incoming), mix
uniform vec4 uEmit;       // count, angle, radius, -
uniform float uBurst, uBurstSeed, uBurstSpeed;
layout(location = 0) out vec4 o0;
layout(location = 1) out vec4 o1;

vec2 gauss(vec2 r) {
  float m = sqrt(-2.0 * log(max(r.x, 1e-6)));
  return m * vec2(cos(TAU * r.y), sin(TAU * r.y));
}

void spawn(float mode, vec2 r, vec2 r2, out vec2 p, out vec2 v) {
  float ang = r.x * TAU;
  vec2 dir = vec2(cos(ang), sin(ang));
  if (mode < 0.5) {        // anywhere
    p = (r - 0.5) * vec2(uAspect, 1.0);
    v = vec2(0.0);
  } else if (mode < 1.5) { // along the waveform
    float w = texture(uWave, vec2(r.x, 0.5)).r;
    p = vec2((r.x * 2.0 - 1.0) * uAspect * 0.46, w * 0.35 + (r2.x - 0.5) * 0.01);
    v = vec2((r2.y - 0.5) * 0.1, w * 0.6) * uSpeed;
  } else if (mode < 2.5) { // rotating emitters
    float k = floor(r.y * uEmit.x);
    float ea = uEmit.y + k / uEmit.x * TAU;
    vec2 c = uEmit.z * vec2(cos(ea), sin(ea));
    p = c + gauss(r2) * 0.012;
    v = (normalize(c + 1e-4) * 0.6 + vec2(-c.y, c.x) * 1.5 + dir * 0.25) * uSpeed;
  } else if (mode < 3.5) { // ring
    p = dir * (0.3 + (r2.x - 0.5) * 0.02);
    v = dir * uSpeed * (0.1 + 0.4 * r2.y);
  } else {                 // centre burst
    p = gauss(r2) * 0.01;
    v = dir * uSpeed * (0.2 + 0.8 * r.y);
  }
}

void main() {
  ivec2 ij = ivec2(gl_FragCoord.xy);
  vec4 s0 = texelFetch(uS0, ij, 0);
  vec4 s1 = texelFetch(uS1, ij, 0);
  vec2 asp = vec2(uAspect, 1.0);
  vec2 p = (s0.xy - 0.5) * asp;
  vec2 vel = s0.zw;
  float life = s1.x, seed = s1.y;

  vec2 fv = texture(uVel, s0.xy).xy * uSimTexel * asp * uFluidAmt;
  vec2 flow = p * uZoomFlow + vec2(-p.y, p.x) * uRotFlow;
  flow += curlNoise(p * 2.5 + seed * 0.05, uTime) * uCurl;
  flow -= p * uConverge;
  vec2 target = fv + flow;
  vel = mix(vel, target, 1.0 - exp(-uDrag * uDt));
  p += vel * uDt;
  life -= uDt * uLifeRate * (0.4 + seed);

  bool burst = hash11(seed * 1931.7 + uBurstSeed) < uBurst;
  vec2 e = abs(p) - vec2(uAspect * 0.5 + 0.05, 0.55);
  if (life <= 0.0 || burst || max(e.x, e.y) > 0.0) {
    vec2 r = hash22(vec2(seed * 7919.0, uTime * 61.7) + vec2(ij));
    vec2 r2 = hash22(r * 311.3 + seed);
    float mode = hash11(seed * 311.0 + floor(uTime * 3.0)) < uSpawn.z ? uSpawn.y : uSpawn.x;
    spawn(mode, r, r2, p, vel);
    if (burst) vel += normalize(p + 1e-4) * uBurstSpeed * (0.3 + r2.y);
    life = 1.0;
  }
  o0 = vec4(p / asp + 0.5, vel);
  o1 = vec4(life, seed, s1.zw);
}`;

export const PARTICLE_VS = HEAD + /* glsl */ `
uniform sampler2D uS0, uS1;
uniform int uW;
uniform float uSize, uBright;
uniform vec3 uColA, uColB, uColC;
out vec3 vCol;
void main() {
  ivec2 ij = ivec2(gl_VertexID % uW, gl_VertexID / uW);
  vec4 s0 = texelFetch(uS0, ij, 0);
  vec4 s1 = texelFetch(uS1, ij, 0);
  gl_Position = vec4(s0.xy * 2.0 - 1.0, 0.0, 1.0);
  gl_PointSize = uSize;
  float sp = length(s0.zw);
  float life = s1.x, seed = s1.y;
  vec3 c = mix(uColA, uColB, smoothstep(0.2, 0.8, seed));
  c = mix(c, uColC * 1.5, smoothstep(0.15, 0.6, sp));
  vCol = c * uBright * smoothstep(0.0, 0.25, life) * smoothstep(1.0, 0.92, life) * (0.1 + min(sp * sp * 6.0, 1.5));
}`;

export const PARTICLE_FS = HEAD + /* glsl */ `
uniform vec2 uRoute;
in vec3 vCol;
layout(location = 0) out vec4 oA;
layout(location = 1) out vec4 oB;
void main() {
  oA = vec4(vCol * uRoute.x, 1.0);
  oB = vec4(vCol * uRoute.y, 1.0);
}`;

// ------------------------------------------------------------ composite ---

export const SCENE_FS = HEAD + COMMON + FS_IN + /* glsl */ `
uniform sampler2D uA, uB;
uniform float uAspect, uBMix, uSat, uHueShift, uSweep, uLinearize;
uniform vec3 uKal;    // segments (outgoing), segments (incoming), mix
uniform float uKalRot, uMirror;
out vec4 o;

vec2 kaleido(vec2 uv, float n) {
  if (n < 1.5) return uv;
  vec2 p = (uv - 0.5) * vec2(uAspect, 1.0);
  float r = length(p);
  float seg = TAU / n;
  float a = mod(atan(p.y, p.x) + uKalRot, seg);
  a = abs(a - seg * 0.5);
  p = r * vec2(cos(a), sin(a));
  return p / vec2(uAspect, 1.0) + 0.5;
}

vec3 fetch(vec2 uv) {
  vec3 c = texture(uA, uv).rgb + texture(uB, uv).rgb * uBMix;
  return c;
}

void main() {
  vec3 c;
  if (uLinearize > 0.5) {
    c = texture(uA, vUv).rgb;
    c = pow(c, vec3(2.2)) * 1.1;
  } else {
    vec2 uv = vUv;
    if (uMirror > 0.5 && uv.x > 0.5) uv.x = 1.0 - uv.x;
    if (uKal.z <= 0.001) c = fetch(kaleido(uv, uKal.x));
    else if (uKal.z >= 0.999) c = fetch(kaleido(uv, uKal.y));
    else c = mix(fetch(kaleido(uv, uKal.x)), fetch(kaleido(uv, uKal.y)), uKal.z);
  }
  // Frame-wide hue shift (vocals) and a key-change sweep travelling outward.
  vec2 p = (vUv - 0.5) * vec2(uAspect, 1.0);
  float r = length(p);
  float front = (1.0 - uSweep) * 1.2;
  float sw = uSweep * exp(-pow((r - front) * 5.0, 2.0));
  float h = uHueShift + sw * 2.4;
  if (h != 0.0) c = max(hueRotate(c, h), 0.0);
  c *= 1.0 + sw * 1.5;
  float l = luma(c);
  c = max(mix(vec3(l), c, uSat), 0.0);
  o = vec4(c, 1.0);
}`;

// ---------------------------------------------------------------- bloom ---

export const BLOOM_PREFILTER_FS = HEAD + FS_IN + /* glsl */ `
uniform sampler2D uSrc;
uniform vec2 uSrcTexel;
uniform float uThreshold, uKnee;
out vec4 o;
void main() {
  vec2 h = uSrcTexel;
  vec3 c = (texture(uSrc, vUv + vec2(-h.x, -h.y)).rgb + texture(uSrc, vUv + vec2(h.x, -h.y)).rgb +
            texture(uSrc, vUv + vec2(-h.x, h.y)).rgb + texture(uSrc, vUv + vec2(h.x, h.y)).rgb) * 0.25;
  float br = max(c.r, max(c.g, c.b));
  float rq = clamp(br - uThreshold + uKnee, 0.0, 2.0 * uKnee);
  rq = rq * rq / (4.0 * uKnee + 1e-5);
  float w = max(rq, br - uThreshold) / max(br, 1e-5);
  o = vec4(min(c * w, vec3(200.0)), 1.0);
}`;

export const BLOOM_DOWN_FS = HEAD + FS_IN + /* glsl */ `
uniform sampler2D uSrc;
uniform vec2 uSrcTexel;
out vec4 o;
void main() {
  vec2 h = uSrcTexel;
  vec3 s = texture(uSrc, vUv).rgb * 4.0;
  s += texture(uSrc, vUv - h).rgb;
  s += texture(uSrc, vUv + h).rgb;
  s += texture(uSrc, vUv + vec2(h.x, -h.y)).rgb;
  s += texture(uSrc, vUv - vec2(h.x, -h.y)).rgb;
  o = vec4(s / 8.0, 1.0);
}`;

export const BLOOM_UP_FS = HEAD + FS_IN + /* glsl */ `
uniform sampler2D uSrc;
uniform vec2 uSrcTexel;
uniform float uWeight;
out vec4 o;
void main() {
  vec2 h = uSrcTexel;
  vec3 s = texture(uSrc, vUv + vec2(-h.x * 2.0, 0.0)).rgb;
  s += texture(uSrc, vUv + vec2(-h.x, h.y)).rgb * 2.0;
  s += texture(uSrc, vUv + vec2(0.0, h.y * 2.0)).rgb;
  s += texture(uSrc, vUv + vec2(h.x, h.y)).rgb * 2.0;
  s += texture(uSrc, vUv + vec2(h.x * 2.0, 0.0)).rgb;
  s += texture(uSrc, vUv + vec2(h.x, -h.y)).rgb * 2.0;
  s += texture(uSrc, vUv + vec2(0.0, -h.y * 2.0)).rgb;
  s += texture(uSrc, vUv + vec2(-h.x, -h.y)).rgb * 2.0;
  o = vec4(s / 12.0 * uWeight, 1.0);
}`;

export const FINAL_FS = HEAD + COMMON + FS_IN + /* glsl */ `
uniform sampler2D uScene, uBloom, uAvg;
uniform float uBloomStr, uExposure, uCA, uVignette, uTonemap, uFrame, uKey, uAdapt, uContrast;
out vec4 o;
vec3 aces(vec3 x) {
  const float a = 2.51, b = 0.03, c = 2.43, d = 0.59, e = 0.14;
  return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
}
void main() {
  vec2 d = vUv - 0.5;
  vec3 col;
  if (uCA > 0.0005) {
    vec2 off = d * uCA;
    col = vec3(texture(uScene, vUv + off).r, texture(uScene, vUv).g, texture(uScene, vUv - off).b);
  } else {
    col = texture(uScene, vUv).rgb;
  }
  col += texture(uBloom, vUv).rgb * uBloomStr;
  float avg = max(texture(uAvg, vec2(0.5)).r, 1e-4);
  col *= uExposure * clamp(pow(uKey / avg, uAdapt), 0.4, 4.0);
  // Soft black point: suppress low-level haze so structures stand out.
  float lc = luma(col);
  col *= lc / (lc + uContrast + 1e-6);
  float v = 1.0 - uVignette * smoothstep(0.2, 0.9, length(d * vec2(1.1, 1.0)) * 1.25);
  col *= v;
  col = mix(clamp(col, 0.0, 1.0), aces(col), uTonemap);
  col = pow(col, vec3(1.0 / 2.2));
  // Triangular dither.
  float n = hash12(gl_FragCoord.xy + uFrame * 1.618) + hash12(gl_FragCoord.yx * 1.37 + uFrame * 3.1) - 1.0;
  col += n / 255.0;
  o = vec4(col, 1.0);
}`;

/** Temporal log-average luminance of the scene into a 1x1 target. */
export const EXPOSURE_FS = HEAD + COMMON + FS_IN + /* glsl */ `
uniform sampler2D uScene, uPrev;
uniform float uRate, uFrame;
out vec4 o;
void main() {
  float s = 0.0;
  vec2 j = hash22(vec2(uFrame, 3.7));
  for (int y = 0; y < 9; y++) {
    for (int x = 0; x < 16; x++) {
      vec2 uv = (vec2(float(x), float(y)) + j) / vec2(16.0, 9.0);
      s += log(luma(texture(uScene, uv).rgb) + 0.002);
    }
  }
  float avg = exp(s / 144.0);
  float prev = texture(uPrev, vec2(0.5)).r;
  o = vec4(prev <= 0.0 ? avg : mix(prev, avg, uRate), 0.0, 0.0, 1.0);
}`;

export const BLIT_FS = HEAD + FS_IN + /* glsl */ `
uniform sampler2D uTex;
out vec4 o;
void main() { o = texture(uTex, vUv); }`;
