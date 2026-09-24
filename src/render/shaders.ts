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

// ------------------------------------------------------------ particles ---

export const PARTICLE_UPDATE_FS = HEAD + COMMON + /* glsl */ `
uniform sampler2D uS0, uS1, uVel, uWave;
uniform float uDt, uTime, uAspect;
uniform vec2 uSimTexel;
uniform float uFluidAmt, uCurl, uZoomFlow, uRotFlow, uConverge, uDrag, uLifeRate, uSpeed;
uniform vec2 uLift;
uniform vec3 uSpawn;      // mode (outgoing), mode (incoming), mix
uniform vec4 uEmit;       // count, angle, radius, spawn spread
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
  } else if (mode < 4.5) { // centre burst
    p = gauss(r2) * uEmit.w;
    v = dir * uSpeed * (0.2 + 0.8 * r.y);
  } else {                 // along the bottom edge
    p = vec2((r.x - 0.5) * uAspect * 0.9, -0.5 + r2.x * 0.03);
    v = vec2((r2.y - 0.5) * 0.05, 0.05 + 0.1 * r.y) * uSpeed;
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
  vec2 flow = p * uZoomFlow + vec2(-p.y, p.x) * uRotFlow + uLift;
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
uniform float uSize, uBright, uAlive, uStreak;
uniform vec3 uColA, uColB, uColC;
out vec3 vCol;
void main() {
  ivec2 ij = ivec2(gl_VertexID % uW, gl_VertexID / uW);
  vec4 s0 = texelFetch(uS0, ij, 0);
  vec4 s1 = texelFetch(uS1, ij, 0);
  float life = s1.x, seed = s1.y;
  float sp = length(s0.zw);
  // Only a fraction of the pool is visible: the activity budget.
  float on = step(seed, uAlive);
  gl_Position = vec4(s0.xy * 2.0 - 1.0, 0.0, 1.0);
  gl_PointSize = on * uSize * (0.55 + 0.9 * fract(seed * 13.7)) * (1.0 + uStreak * min(sp, 2.0));
  vec3 c = mix(uColA, uColB, smoothstep(0.2, 0.8, seed));
  c = mix(c, uColC * 1.4, smoothstep(0.2, 0.8, sp));
  vCol = on * c * uBright * smoothstep(0.0, 0.3, life) * smoothstep(1.0, 0.9, life) * (0.35 + min(sp * sp * 3.0, 1.2));
}`;

export const PARTICLE_FS = HEAD + /* glsl */ `
in vec3 vCol;
out vec4 o;
void main() {
  vec2 d = gl_PointCoord * 2.0 - 1.0;
  float a = exp(-dot(d, d) * 3.5);
  o = vec4(vCol * a, 1.0);
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
