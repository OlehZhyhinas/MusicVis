import type { Schema } from '../genome';
import type { SculptureMusic } from './fracture';
import { recoilImpulse } from './recoil';

export const FERROFLUID_SCHEMA: Schema = {
  size:{min:0.05,max:0.6,def:0.4},form:{min:0,max:2,def:0,choices:[0,1,2]},
  blobs:{min:2,max:6,def:5,int:true},spread:{min:0.2,max:1.2,def:0.7},
  tension:{min:0.04,max:0.5,def:0.28},spikes:{min:0,max:1,def:0.7},frequency:{min:6,max:18,def:11},
  magnet:{min:0,max:1,def:0.8},viscosity:{min:0.3,max:2,def:1},drift:{min:0,max:1,def:0.5},
  turn:{min:-0.06,max:0.06,def:0.014},tilt:{min:-0.15,max:0.3,def:0.07},
  sheen:{min:0,max:1,def:0.8},iridescence:{min:0,max:1,def:0.6},
};

/** Damped magnetic poles drive a merging implicit surface, not a fluid dynamics solver. */
export class FerrofluidSimulation {
  readonly drops=new Float32Array(24);
  readonly velocity=new Float32Array(18);
  time=0; pulse=0; field=0;
  private previous=0; private initialized=false;
  step(p:Record<string,number>,music:SculptureMusic,dt:number):void {
    const h=Math.max(0,Math.min(dt,0.05));this.time+=h;
    if(music.hit||recoilImpulse(music.onset,this.previous))this.pulse=1;
    this.previous=music.onset;this.pulse*=Math.exp(-h*5);
    this.field+=(p.magnet*(music.bass*0.6+this.pulse*0.65+music.noteOn*0.2)-this.field)*(1-Math.exp(-h*12));
    const n=Math.round(p.blobs),phase=this.time*p.drift*0.55,pitch=(music.pitch-60)*0.055*p.magnet;
    for(let i=0;i<6;i++){
      const a=i*Math.PI*2/n,move=Math.sin(phase+i*1.7+pitch);
      let target:number[],radius:number;
      if(p.form===0){
        target=[Math.cos(a)*p.spread, -0.12+this.field*0.2+move*p.drift*0.1,Math.sin(a)*p.spread*0.65];radius=0.35;
      }else if(p.form===1){
        target=[Math.cos(a+phase*0.2)*p.spread,Math.sin(a)*p.spread*0.8+move*0.16,Math.sin(a*2+phase)*p.spread*0.5];radius=0.26+(i%3)*0.055;
      }else{
        const side=i%2?1:-1;
        target=[side*(p.spread+Math.sin(phase+pitch)*p.drift*0.3-this.field*0.2),Math.sin(a+phase)*0.22,Math.cos(a)*0.25];radius=0.38;
      }
      for(let axis=0;axis<3;axis++){
        const at=i*4+axis,v=i*3+axis,k=18/p.viscosity;
        if(!this.initialized)this.drops[at]=target[axis];
        for(let s=0;s<3;s++){
          this.velocity[v]+=(k*(target[axis]-this.drops[at])-2*Math.sqrt(k)*0.75*this.velocity[v])*h/3;
          this.drops[at]+=this.velocity[v]*h/3;
        }
      }
      this.drops[i*4+3]=i<n?radius:0;
    }
    this.initialized=true;
  }
}
