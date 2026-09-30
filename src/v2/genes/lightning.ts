import type { Schema } from '../genome';
import { recoilImpulse } from './recoil';
export const LIGHTNING_SCHEMA:Schema={
  size:{min:0.1,max:0.95,def:0.7},channels:{min:2,max:6,def:4,choices:[2,3,4,5,6]},
  branches:{min:0,max:1,def:0.7},jagged:{min:0,max:1,def:0.7},
  speed:{min:0.5,max:4,def:2},decay:{min:0.08,max:0.65,def:0.28},
  width:{min:0.001,max:0.015,def:0.004},spread:{min:0,max:1,def:0.55},fan:{min:0,max:1,def:0},
};
/** An independent emission locus: the host body keeps its own shape and movement. */
export const LIGHTNING_EMIT_SCHEMA:Schema={
  channels:LIGHTNING_SCHEMA.channels, branches:LIGHTNING_SCHEMA.branches, jagged:LIGHTNING_SCHEMA.jagged,
  speed:LIGHTNING_SCHEMA.speed, decay:LIGHTNING_SCHEMA.decay, width:LIGHTNING_SCHEMA.width,
  spread:LIGHTNING_SCHEMA.spread, fan:LIGHTNING_SCHEMA.fan,
  reach:{min:0.05,max:1.2,def:0.4}, gain:{min:0,max:2,def:0.8}, sustain:{min:0,max:1,def:0.7},
};
export interface LightningEmitter {
  origins: {x:number;y:number;angle:number;scale:number;radius:number}[];
  reach:number; sustain:number; held:number; legato:number;
}
export interface LightningParams {channels:number;branches:number;jagged:number;speed:number;decay:number;width:number;spread:number;fan:number}
export interface LightningInput {hit:boolean;onsets:ArrayLike<number>;noteOn:number;pitch:number;tonic:number}
export interface BoltSegment {ax:number;ay:number;bx:number;by:number;start:number;end:number;weight:number}
export interface Bolt {age:number;channel:number;hue:number;strength:number;segments:BoltSegment[]}
const TAU=Math.PI*2;
export const MAX_LIGHTNING_SEGMENTS=4096;
/** Deterministic midpoint displacement followed by recursively smaller offshoots. */
export function lightningBranch(seed:number,channel:number,p:LightningParams):BoltSegment[] {
  let state=(seed*1664525+channel*1013904223)>>>0;
  const random=()=>{state=(Math.imul(state,1664525)+1013904223)>>>0;return state/4294967296;};
  const segments:BoltSegment[]=[];
  const angle=p.fan*(channel/Math.max(1,p.channels)*TAU+(random()-0.5)*0.8),c=Math.cos(angle),s=Math.sin(angle);
  const y=(channel/(p.channels-1)-0.5)*p.spread*1.2;
  const rotate=(x:number,y:number):[number,number]=>[x*c-y*s,x*s+y*c];
  const a=rotate(-1.15,y),b=rotate(1.15,y+(random()-0.5)*0.4*p.spread);
  function line(ax:number,ay:number,bx:number,by:number,depth:number,weight:number,start:number,end:number):void {
    const count=depth===0?32:depth===1?8:4;
    let points:[number,number][]=[[ax,ay],[bx,by]];
    while(points.length<=count) {
      const refined:[number,number][]=[points[0]];
      for(let i=0;i<points.length-1;i++) {
        const a=points[i],b=points[i+1],dx=b[0]-a[0],dy=b[1]-a[1];
        const offset=(random()-0.5)*p.jagged*0.7;
        refined.push([(a[0]+b[0])/2-dy*offset,(a[1]+b[1])/2+dx*offset],b);
      }
      points=refined;
    }
    for(let i=0;i<count;i++) {
      const t0=start+(end-start)*i/count,t1=start+(end-start)*(i+1)/count;
      segments.push({ax:points[i][0],ay:points[i][1],bx:points[i+1][0],by:points[i+1][1],start:t0,end:t1,weight});
      if(depth<2 && i>2 && i<count-2 && random()<p.branches*(depth===0?0.4:0.22)) {
        const theta=Math.atan2(by-ay,bx-ax)+(random()<0.5?-1:1)*(0.4+random()*0.95);
        const length=(depth===0?0.18+random()*0.55:0.08+random()*0.18)*(0.4+0.6*p.spread);
        line(points[i][0],points[i][1],points[i][0]+Math.cos(theta)*length,points[i][1]+Math.sin(theta)*length,depth+1,weight*0.5,t0,Math.min(1.2,t0+length/2.3));
      }
    }
  }
  line(a[0],a[1],b[0],b[1],0,1,0,1);
  return segments;
}
/** Trigger envelopes are edge-detected; held notes never retrigger every frame. */
export class LightningSimulation {
  readonly bolts:Bolt[]=[];
  private previous=new Float32Array(5);
  private serial=1;
  private lastPitch=0;
  strikes=0;
  private heldBolt:Bolt|null=null;
  step(input:LightningInput,p:LightningParams,dt:number,emitter?:LightningEmitter):void {
    for(const b of this.bolts) {
      b.age+=Math.min(0.1,Math.max(0,dt));
      if(emitter && b===this.heldBolt && emitter.held*emitter.legato*emitter.sustain>0.15)
        b.age=Math.min(b.age,0.18/p.speed+0.04);
    }
    for(let i=this.bolts.length-1;i>=0;i--)if(this.bolts[i].age>p.decay*4+0.18/p.speed)this.bolts.splice(i,1);
    const fire=(channel:number,pitch:number,strength:number,melodic=false)=>{
      const seed=this.serial++,segments=lightningBranch(seed,channel,emitter?{...p,fan:0,spread:p.spread}:p);
      if(emitter?.origins.length) {
        const origin=emitter.origins[(seed-1)%emitter.origins.length];
        const angle=origin.angle+TAU*(channel/Math.max(1,p.channels)*p.fan + p.fan*(seed*0.381966%1));
        const c=Math.cos(angle),s=Math.sin(angle),scale=emitter.reach*origin.scale/2.3;
        // Every main branch starts at its host, not at the old full-screen lane boundary.
        const startY=(channel/(p.channels-1)-0.5)*p.spread*1.2;
        for(const seg of segments) {
          const ax=(seg.ax+1.15)*scale+origin.radius,ay=(seg.ay-startY)*scale;
          const bx=(seg.bx+1.15)*scale+origin.radius,by=(seg.by-startY)*scale;
          seg.ax=origin.x+ax*c-ay*s;seg.ay=origin.y+ax*s+ay*c;
          seg.bx=origin.x+bx*c-by*s;seg.by=origin.y+bx*s+by*c;
        }
      }
      const bolt={age:0,channel,hue:((Math.round(pitch)%12)+12)%12/12,strength,segments};
      this.bolts.push(bolt);if(melodic)this.heldBolt=bolt;this.strikes++;
    };
    const drum=recoilImpulse(input.onsets[0],this.previous[0]);
    if(input.hit || drum)fire(0,input.tonic,1.4);
    const pitch=Math.round(input.pitch);
    if(recoilImpulse(input.noteOn,this.previous[4]) || (input.noteOn>0.3 && pitch!==this.lastPitch))fire(1+((pitch%Math.max(1,p.channels-1))+p.channels-1)%(p.channels-1),pitch,1,true);
    for(let i=1;i<4;i++)if(recoilImpulse(input.onsets[i],this.previous[i]))fire(1+i%(p.channels-1),input.tonic+[0,7,3,10][i],0.65+input.onsets[i]*0.35);
    for(let i=0;i<4;i++)this.previous[i]=input.onsets[i];this.previous[4]=input.noteOn;this.lastPitch=pitch;
    // Dense rolls retain recent strikes without an unbounded geometry backlog.
    while(this.bolts.length>p.channels*3)this.bolts.shift();
  }
}
export const LIGHTNING_FIELD=`
vec3 FLD(vec2 p){
  vec2 uv=p/max(BD(2).x*1.8,0.001)*0.5+0.5;
  if(any(lessThan(uv,vec2(0.0))) || any(greaterThan(uv,vec2(1.0)))) return vec3(0.0);
  return texture(LIGHTNING_TEX,uv).rgb*BD(0).x*uLayerK;
}`;
