// A freeform 3D path; its control points are part of the genome, not a style enum.
import type { Schema } from '../genome';
export interface RibbonPoint { x: number; y: number; z: number; width: number; twist: number }
export const RIBBON_SCHEMA: Schema = {
  projection: { min: 0, max: 1, def: 0 },
  ink: { min: 0, max: 1, def: 0 },
  edges: { min: 0, max: 1, def: 0 },
  ripples: { min: 0, max: 1, def: 0 },
  size: { min: 0.03, max: 0.65, def: 0.3 },
  width: { min: 0.015, max: 0.35, def: 0.13 },
  wind: { min: 0, max: 1, def: 0.35 },
  stiffness: { min: 0, max: 1, def: 0.55 },
  twist: { min: -4, max: 4, def: 0.5 },
  flow: { min: 0, max: 1, def: 0.8 },
  turn: { min: -0.2, max: 0.2, def: 0.025 },
  tilt: { min: -0.45, max: 0.45, def: 0.08 },
  closed: { min: 0, max: 1, def: 0, choices: [0,1] },
  speed: { min: 0.1, max: 2, def: 0.7 },
};
export const MAX_RIBBON_POINTS = 64;
const clamp = (x: unknown, min: number, max: number, def: number) => typeof x === 'number' && Number.isFinite(x) ? Math.max(min, Math.min(max, x)) : def;
export function repairRibbonPath(raw: unknown): RibbonPoint[] {
  const points = Array.isArray(raw) && raw.length>=2 ? raw : [
    {x:-1,y:-0.3,z:0}, {x:-0.4,y:0.45,z:0.35}, {x:0.35,y:-0.35,z:-0.3}, {x:1,y:0.3,z:0},
  ];
  return points.slice(0,MAX_RIBBON_POINTS).map(p=>({
    x:clamp(p?.x,-2,2,0),y:clamp(p?.y,-2,2,0),z:clamp(p?.z,-2,2,0),
    width:clamp(p?.width,0.1,2,1),twist:clamp(p?.twist,-12,12,0),
  }));
}
export function ribbonExtent(path: RibbonPoint[] | undefined, width = 0.35): number {
  return Math.max(0.5,...repairRibbonPath(path).map(p=>Math.hypot(p.x,p.y,p.z)))+width*2+0.4;
}
/** Catmull–Rom interpolation through arbitrary 3D points; angle is unwrapped turns. */
export function sampleRibbon(path: RibbonPoint[], u: number, closed: boolean): RibbonPoint {
  const n=path.length, v=Math.max(0,Math.min(1,u))*(closed?n:n-1), i=Math.floor(v), t=v-i;
  const at=(j:number)=>path[closed?((j%n)+n)%n:Math.max(0,Math.min(n-1,j))];
  const a=at(i-1),b=at(i),c=at(i+1),d=at(i+2);
  const sample=(k:keyof RibbonPoint)=>0.5*((2*b[k])+(-a[k]+c[k])*t+(2*a[k]-5*b[k]+4*c[k]-d[k])*t*t+(-a[k]+3*b[k]-3*c[k]+d[k])*t*t*t);
  return {x:sample('x'),y:sample('y'),z:sample('z'),width:Math.max(0.05,sample('width')),twist:sample('twist')};
}

export const RIBBON_FIELD = `
vec3 FLD(vec2 p) {
  vec2 uv=p/max(BD(2).x*BD(2).y,0.001)*0.5+0.5;
  if(any(lessThan(uv,vec2(0.0))) || any(greaterThan(uv,vec2(1.0)))) return vec3(0.0);
  vec4 ribbon=texture(RIBBON_TEX,uv);
  return ribbon.rgb*ribbon.a*BD(0).x*uLayerK;
}`;

/** Human-editable rows: x y z [width [twist in turns]]. */
export function parseRibbonPath(text:string):RibbonPoint[] {
  const lines=text.trim().split(/\n/).filter(line=>line.trim());
  if(lines.length<2 || lines.length>MAX_RIBBON_POINTS) throw Error(`Use 2–${MAX_RIBBON_POINTS} control points.`);
  return lines.map((line,i)=>{
    const values=line.trim().split(/[\s,]+/).map(Number);
    if(values.length<3 || values.length>5 || values.some(v=>!Number.isFinite(v))) throw Error(`Point ${i+1}: enter x, y, z, optional width and twist.`);
    const [x,y,z,width=1,twist=0]=values;
    if([x,y,z].some(v=>Math.abs(v)>2) || width<0.1 || width>2 || Math.abs(twist)>12) throw Error(`Point ${i+1}: xyz must be −2…2, width 0.1…2, twist −12…12 turns.`);
    return {x,y,z,width,twist};
  });
}
