// Individually bending painted petals, procedural stamens and scroll-like stems.
// Each petal has its own local frame and colour coordinate; it is not a radial fold.
import type { Schema } from '../genome';

export const LILY_SCHEMA: Schema = {
  size: { min: 0.025, max: 0.4, def: 0.28 },
  petals: { min: 4, max: 7, def: 6, choices: [4, 5, 6, 7] },
  open: { min: 0.1, max: 1, def: 0.75 },
  curl: { min: 0.1, max: 1, def: 0.65 },
  veins: { min: 0, max: 1, def: 0.5 },
  stem: { min: 0, max: 1.8, def: 1.1 },
  shimmer: { min: 0, max: 1, def: 0.45 },
  depth: { min: 0, max: 1, def: 0 },
  twist: { min: -1, max: 1, def: 0 },
  flow: { min: 0, max: 1, def: 0 },
};

// Closest point on a quadratic from the origin through b to c. Two starts retain
// the returning tip of a curled petal instead of jumping across its fold.
export const LILY_GLSL = /* glsl */ `
// Height and analytic derivatives of a curved petal in its own 3D frame.
// x spans the blade, t runs from the attachment to the tip.
vec3 lilyHeight(float x, float t, float id) {
  float wave=7.0*t-SB.w+id;
  float z=0.26*sin(PI*t)+0.26*SA.w*t*t+0.4*SC.y*x*t
    +0.1*SB.z*sin(wave)*t+0.3*x*x;
  float dx=0.4*SC.y*t+0.6*x;
  float dt=0.26*PI*cos(PI*t)+0.52*SA.w*t+0.4*SC.y*x
    +0.1*SB.z*(sin(wave)+7.0*t*cos(wave));
  return vec3(z,dx,dt)*SC.x;
}
// Intersect the orthographic view ray with the rotated height surface. Returns
// petal x/y, camera depth and residual; the same coordinates drive paint and outline.
vec4 lilySurface(vec2 screen, float len, float id, out vec3 normal) {
  if(SC.x<0.001) { normal=vec3(0.0,0.0,1.0); return vec4(screen,0.0,0.0); }
  float pitch=SC.x*(0.4*(1.0-SA.z)+0.38*sin(SB.w+id*1.7));
  float roll=SC.x*(0.22*sin(SB.w+id)+0.42*SC.y);
  float cp=cos(pitch),sp=sin(pitch),cr=cos(roll),sr=sin(roll);
  mat3 rotate=mat3(cr,0.0,-sr, sr*sp,cp,cr*sp, sr*cp,-sp,cr*cp);
  vec3 origin=transpose(rotate)*vec3(screen,0.0);
  vec3 ray=transpose(rotate)*vec3(0.0,0.0,1.0);
  float at=0.0;
  for(int k=0;k<6;k++) {
    vec3 p=origin+ray*at, h=lilyHeight(p.x,p.y/len,id);
    float slope=ray.z-h.y*ray.x-h.z*ray.y/len;
    at-=clamp((p.z-h.x)/max(slope,0.2),-0.3,0.3);
  }
  vec3 p=origin+ray*at, h=lilyHeight(p.x,p.y/len,id);
  normal=normalize(rotate*vec3(-h.y,-h.z/len,1.0));
  return vec4(p.xy,at,abs(p.z-h.x));
}
vec3 lilyPaint;
vec3 lilyCurve(vec2 p, vec2 b, vec2 c) {
  vec2 a = c - 2.0*b;
  float best = 1e6, at = 0.0, side = 0.0;
  for (int seed = 0; seed < 2; seed++) {
    float t = seed == 0 ? 0.25 : 0.9;
    for (int k = 0; k < 4; k++) {
      vec2 d = t*(2.0*b+t*a)-p, v = 2.0*(b+t*a);
      float denom = max(dot(v,v)*0.25, dot(v,v)+2.0*dot(d,a));
      t = clamp(t-clamp(dot(d,v)/max(denom,1e-5),-0.3,0.3),0.0,1.0);
    }
    vec2 d = p-t*(2.0*b+t*a), v = normalize(b+t*a+vec2(1e-6));
    float ds = dot(d,d);
    if(ds<best) { best=ds; at=t; side=dot(d,vec2(-v.y,v.x)); }
  }
  return vec3(sqrt(best),at,side);
}
// Signed distance to the two actual ribbon edges, rather than a union of
// circles along its centre. This preserves pointed, returning petal tips.
vec3 lilyRibbon(vec2 p, vec2 b, vec2 c, vec2 e, float width) {
  float edgeD=1e6, centreD=1e6, at=0.0, side=0.0;
  bool inside=false;
  vec2 prev=vec2(0.0), prevL=prev, prevR=prev;
  for(int k=1;k<=28;k++) {
    float t=float(k)/28.0;
    vec2 next=3.0*t*(1.0-t)*(1.0-t)*b+3.0*t*t*(1.0-t)*c+t*t*t*e;
    vec2 tangent=normalize((1.0-t)*(1.0-t)*b+2.0*t*(1.0-t)*(c-b)+t*t*(e-c));
    vec2 normal=vec2(-tangent.y,tangent.x);
    float w=width*pow(max(0.0,sin(PI*t)),0.85)*pow(max(0.0,1.0-t),0.35)*1.6;
    vec2 left=next+normal*w, right=next-normal*w;
    for(int edge=0;edge<2;edge++) {
      vec2 v0=edge==0?prevL:prevR, v1=edge==0?left:right, v=v1-v0;
      float h=clamp(dot(p-v0,v)/max(dot(v,v),1e-8),0.0,1.0);
      vec2 delta=p-v0-h*v;
      edgeD=min(edgeD,dot(delta,delta));
      if((v0.y>p.y)!=(v1.y>p.y)) {
        float crossing=v0.x+(p.y-v0.y)*(v1.x-v0.x)/(v1.y-v0.y);
        if(p.x<crossing) inside=!inside;
      }
    }
    vec2 v=next-prev;
    float h=clamp(dot(p-prev,v)/max(dot(v,v),1e-8),0.0,1.0);
    vec2 delta=p-prev-h*v;
    float ds=dot(delta,delta);
    if(ds<centreD) {
      centreD=ds; at=(float(k)-1.0+h)/28.0;
      vec2 axis=normalize((1.0-at)*(1.0-at)*b+2.0*at*(1.0-at)*(c-b)+at*at*(e-c));
      side=dot(delta,vec2(-axis.y,axis.x));
    }
    prev=next;prevL=left;prevR=right;
  }
  return vec3(sqrt(edgeD)*(inside?-1.0:1.0),at,side);
}
vec3 lilyPetal(vec2 p, vec2 b, vec2 c, float width) {
  return lilyRibbon(p,b*0.666667,(c+2.0*b)/3.0,c,width);
}
vec3 SHP(vec2 p) {
  lilyPaint=vec3(0.0);
  float size = max(SA.x,0.001);
  vec2 q = p/size;
  float phase = SB.w;
  vec3 result = vec3(1e3,0.0,0.0);
  // The stems stay delicate and subordinate to the shaded petals.
  if(SB.y>0.01) {
    float y = clamp(q.y,-SB.y,0.0);
    float axis = 0.13*sin(-y*2.5) + 0.025*sin(phase)*(-y);
    float stemD = length(q-vec2(axis,y))-0.009;
    result = vec3(stemD,0.4+y*0.2,0.65);
    for(int j=0;j<2;j++) {
      float signX = j==0 ? -1.0 : 1.0;
      vec2 centre = vec2(signX*0.49,-SB.y*(0.5+float(j)*0.2));
      vec2 r = (q-centre)*vec2(signX,1.0);
      float phi = mod(atan(r.y,r.x)+TAU,TAU);
      float a = clamp(phi,0.0,5.7);
      float radius = 0.27*(1.0-0.82*a/5.7);
      float d = length(r-radius*vec2(cos(a),sin(a)))-0.006;
      float rootY=-SB.y*0.35;
      vec2 root=vec2(0.13*sin(-rootY*2.5)+0.025*sin(phase)*(-rootY),rootY);
      vec3 link = lilyCurve(q-root,vec2(signX*0.3,-0.1),centre+vec2(signX*0.27,0.0)-root);
      d = min(d,link.x-0.005);
      if(d<result.x) result=vec3(d,0.25+float(j)*0.35+phi*0.035,0.75);
      vec2 leafRoot=vec2(0.1,-SB.y*(0.55+float(j)*0.28));
      vec3 leaf=lilyPetal(q-leafRoot,vec2(signX*0.48,0.36),vec2(signX*0.7,0.18),0.065);
      if(leaf.x<result.x) {
        float across=clamp(leaf.z/0.06,-1.0,1.0);
        float light=0.15+0.65*pow(max(0.0,1.0-abs(across+0.15)),3.0)+0.75*exp(-abs(leaf.x)/0.003);
        result=vec3(leaf.x,0.3+leaf.y*0.3+across*0.18,light);
      }
    }
  }
  float frontDepth=-1e3;
  for(int j=0;j<min(int(SA.y+0.5),7);j++) {
    float id=float(j);
    // Three upright rear petals, then broad side petals and a drooping foreground
    // lip. Their silhouettes and layering follow a lily rather than a radial fan.
    float angle=j==0?-0.72:j==1?0.08:j==2?0.95:j==3?-1.9:j==4?1.9:3.05+(id-5.0)*0.55;
    angle+=0.04*sin(phase+id*1.7)+(SA.z-0.6)*0.12*sin(id*2.0);
    vec2 r=rot2(-angle)*q;
    float len=(0.82+0.3*SA.z)*(j==1?1.18:j<3?1.05:0.92);
    float curl=SA.w*(0.75+0.25*sin(id*1.9+0.7));
    float hand=j%2==0?1.0:-1.0;
    float bladeWidth=(0.2+0.13*SA.z)*(j<3?0.85:1.15);
    vec3 normal;
    vec4 surface=lilySurface(r,len,id,normal);
    r=surface.xy;
    float petalDepth=surface.z+id*0.003;
    float d, hue, light;
    vec3 paint;
    float coverage;
    if(uLilyReady>0.5) {
      // Inverse bending retains the painted folds while each petal flexes freely.
      float t=r.y/len;
      r.x-=hand*(curl-0.55)*0.28*t*t+0.02*sin(phase+id+t*3.0)*t;
      float width=(j<3?0.95:1.1)*(0.75+0.4*SA.z);
      int cell=j%6;
      float rootX=cell==0?0.725:cell==1?0.52:cell==2?0.58:cell==3?0.735:cell==4?0.525:0.61;
      vec2 uv=vec2(r.x/width+rootX,0.95-r.y/len*0.88);
      vec2 bounded=clamp(uv,vec2(0.003),vec2(0.997));
      vec2 atlas=(bounded+vec2(float(cell%3),float(cell/3)))/vec2(3.0,2.0);
      vec4 texel=texture(uLilyAtlas,atlas);
      d=(texture(uLilyDistance,atlas).r-128.0/255.0)*0.5*min(width,len/0.88);
      d+=length((uv-bounded)*vec2(width,len/0.88));
      coverage=texel.a*(1.0-step(0.001,length(uv-bounded)));
      hue=id*0.183+t*0.25;
      light=0.15+dot(texel.rgb,vec3(0.2126,0.7152,0.0722));
      paint=lin(texel.rgb);
      // Shimmer travels along the veins rather than flashing the whole flower.
      paint*=1.0+SB.z*0.15*sin(t*9.0-phase+id);
      paint=mix(paint*0.8,paint,0.5+0.5*SB.x);
      // Advection in surface coordinates: colour travels root-to-tip through
      // sinuous vein channels, rather than recolouring the entire flower at once.
      float across=r.x/width;
      float stream=t*2.5+across*1.4+0.13*sin(across*16.0+t*5.0)-phase/TAU;
      float ribbon=pow(0.5+0.5*cos(TAU*stream),10.0);
      float currentHue=id*0.137+t*0.42+across*0.6-phase/TAU*0.5;
      vec3 neon=lin(hsv2rgb(vec3(fract(currentHue),0.98,1.0)));
      float grain=dot(texel.rgb,vec3(0.2126,0.7152,0.0722));
      vec3 flowing=neon*(0.16+0.85*grain+1.5*ribbon);
      paint=mix(paint,flowing,SC.z);
      vec3 lamp=normalize(vec3(-0.4,0.6,1.0));
      float diffuse=max(0.0,dot(normal,lamp));
      float spec=pow(max(0.0,dot(normal,normalize(lamp+vec3(0.0,0.0,1.0)))),36.0);
      float fresnel=pow(1.0-max(0.0,normal.z),3.0);
      paint*=mix(1.0,0.28+0.8*diffuse,SC.x);
      paint+=SC.x*(mix(neon,vec3(1.0),0.25)*spec*0.35+neon*fresnel*0.5);
      light*=mix(1.0,0.5+0.5*diffuse,SC.x);
      coverage*=1.0-smoothstep(0.005,0.025,surface.w);
    } else {
      vec3 near=lilyRibbon(r/len,vec2(-0.15*hand,0.28),vec2((0.16+0.4*curl)*hand,1.0+0.27*curl),vec2((-0.12+0.22*curl)*hand,1.12-0.12*curl),bladeWidth);
      float t=near.y; d=near.x*len;
      float width=bladeWidth*pow(max(0.0,sin(PI*t)),0.85)*pow(max(0.0,1.0-t),0.35)*1.6;
      float across=clamp(near.z/max(width,0.001),-1.0,1.0);
      float foldAxis=across+0.24*sin(t*4.0+id*1.7);
      float z=sqrt(max(0.0,1.0-across*across));
      float fold=exp(-pow((foldAxis+0.3)*4.0,2.0));
      float valley=exp(-pow((foldAxis-0.15)*7.0,2.0));
      float veinPhase=across*70.0+3.0*sin(t*6.0);
      float aa=1.0-smoothstep(0.5,2.0,fwidth(veinPhase));
      light=(0.08+0.25*z+0.8*fold)*(0.45+0.55*sin(PI*t));
      light*=1.0-0.7*valley;
      light*=1.0-SB.x*0.18*(0.5+0.5*sin(veinPhase)*aa);
      light+=0.65*exp(-abs(d)/0.0035);
      hue=id*0.183+t*0.25+across*0.13+SB.z*0.035*sin(t*7.0-phase)+0.018*sin(across*22.0+t*9.0)*aa;
      coverage=smoothstep(0.002,-0.002,d);
      paint=lin(hsv2rgb(vec3(fract(hue),0.85,1.0)))*light;
    }
    if(j==0) lilyPaint=lin(hsv2rgb(vec3(fract(result.y),0.85,1.0)))*max(0.0,result.z)*0.5;
    // Depth is evaluated per fragment, so intersecting petals change which
    // surface is in front as they lift and twist. Flat legacy petals retain order.
    if(SC.x<0.001 || petalDepth>=frontDepth) {
      lilyPaint=mix(lilyPaint,paint,coverage);
      if(coverage>0.5) frontDepth=petalDepth;
    }
    if(d<result.x && coverage<0.01) result.yz=vec2(hue,light);
    result.yz=mix(result.yz,vec2(hue,light),coverage);
    result.x=min(result.x,d);
  }
  // Fine curved stamens and brighter pollen tips sit above the petals.
  for(int j=0;j<3;j++) {
    float id=float(j), ang=-0.55+id*0.18+0.04*sin(phase+id);
    vec2 r=rot2(-ang)*q;
    vec2 tip=vec2(0.05,0.52+id*0.07);
    vec3 filament=lilyCurve(r,vec2(-0.08,0.34),tip);
    float d=min(filament.x-0.007,length((r-tip)*vec2(0.7,1.0))-0.02);
    float stamen=smoothstep(0.003,-0.003,d);
    lilyPaint=mix(lilyPaint,vec3(1.0,0.62,0.06),stamen);
    if(d<0.003 || d<result.x) result=vec3(d,0.64+id*0.02,1.5);
  }
  result.x*=size;
  return result;
}
`;

export function packLily(E: Float32Array, o: number, P: (k: string) => number, phase: number): number {
  ['size','petals','open','curl','veins','stem','shimmer'].forEach((k,i)=>{E[o+i]=P(k);});
  E[o+7]=phase;
  return P('size')*Math.max(1.7,P('stem'));
}
