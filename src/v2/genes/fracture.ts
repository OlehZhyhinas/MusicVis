import type { Schema } from '../genome';
import { recoilImpulse } from './recoil';
import { SolidMesh,add,sub,mul,unit,rotateAxis,hash,type V3 } from './meshGeometry';
export const FRACTURE_SCHEMA:Schema={
 size:{min:0.05,max:0.6,def:0.36},pieces:{min:4,max:14,def:9,int:true},form:{min:0,max:2,def:0,choices:[0,1,2]},
 open:{min:0,max:1.2,def:0.25},impact:{min:0,max:1,def:0.7},recovery:{min:0.4,max:4,def:1.5},
 thickness:{min:0.01,max:0.12,def:0.035},tumble:{min:0,max:1,def:0.6},
 turn:{min:-0.08,max:0.08,def:0.015},tilt:{min:-0.4,max:0.4,def:0.07},
 prism:{min:0,max:1,def:0.85},clarity:{min:0,max:1,def:0.7},
};
export interface SculptureMusic {hit:boolean;onset:number;noteOn:number;held:number;legato:number;bass:number;vocals:number;pitch:number}
interface Shard {points:V3[];center:V3;axis:V3;direction:V3;x:number;v:number;delay:number}
/** Each triangle is a solid prism with its own damped separation spring and angular motion. */
export class FractureSimulation {
  shards:Shard[]=[];private key='';private previous=0;private age=100;time=0;
  private make(n:number,form:number):void {
    this.shards=[];const rows=Math.round(n*0.65),points:V3[]=[];
    for(let y=0;y<=rows;y++)for(let x=0;x<=n;x++){
      const i=y*(n+1)+x;
      points.push([(x+(x>0&&x<n?(hash(i)-0.5)*0.65:0))/n*2.65-1.325,(y+(y>0&&y<rows?(hash(i+513)-0.5)*0.65:0))/rows*1.65-0.825,0]);
    }
    const triangle=(ids:number[])=>{
      const ps=ids.map(i=>points[i]),c=mul(add(add(ps[0],ps[1]),ps[2]),1/3),id=this.shards.length;
      if(form===1&&Math.hypot(c[0]/1.325,c[1]/0.825)>0.96)return;
      if(form===2&&Math.abs(c[0])/1.325+Math.abs(c[1])/0.825>1.04)return;
      this.shards.push({points:ps,center:c,axis:unit([hash(id+51)-0.5,hash(id+85)-0.5,hash(id+93)-0.5]),
        direction:[c[0]*0.45,c[1]*0.45,(hash(id+213)-0.5)*1.8],x:0,v:0,delay:Math.hypot(c[0]+0.3,c[1])*0.11});
    };
    for(let y=0;y<rows;y++)for(let x=0;x<n;x++){
      const a=y*(n+1)+x,b=a+1,c=a+n+1,d=c+1;
      if(hash(a)>0.5){triangle([a,b,d]);triangle([a,d,c]);}else{triangle([a,b,c]);triangle([b,d,c]);}
    }
  }
  step(p:Record<string,number>,music:SculptureMusic,dt:number):Float32Array {
    const key=p.pieces+':'+p.form;if(key!==this.key){this.make(Math.round(p.pieces),p.form);this.key=key;}
    const h=Math.min(0.05,Math.max(0,dt));this.time+=h;this.age+=h;
    if(music.hit||recoilImpulse(music.onset,this.previous))this.age=0;
    this.previous=music.onset;const mesh=new SolidMesh();
    this.shards.forEach((sh,id)=>{
      const age=this.age-sh.delay,k=26*p.recovery;
      const wave=age<0?0:(1-Math.exp(-age*35))*Math.exp(-age*p.recovery*1.8);
      const target=p.open*(1-music.held*music.legato*0.3)+p.impact*wave;
      // Semi-implicit substeps keep dense kicks stable even after a slow frame.
      for(let i=0;i<3;i++){sh.v+=(k*(target-sh.x)-2*Math.sqrt(k)*0.7*sh.v)*h/3;sh.x+=sh.v*h/3;}
      sh.x=Math.max(-0.04,Math.min(1.8,sh.x));
      const center=add(sh.center,mul(sh.direction,sh.x)),angle=sh.x*p.tumble*(hash(id+71)-0.5)*3.6;
      const points=sh.points.map(q=>add(center,rotateAxis(mul(sub(q,sh.center),0.97),sh.axis,angle)));
      const normal=rotateAxis([0,0,1],sh.axis,angle),half=mul(normal,p.thickness*0.5);
      const top=points.map(q=>add(q,half)),bottom=points.map(q=>sub(q,half));
      mesh.triangle(top[0],top[1],top[2],hash(id));mesh.triangle(bottom[2],bottom[1],bottom[0],hash(id));
      for(let i=0;i<3;i++){const j=(i+1)%3;mesh.triangle(top[i],bottom[i],bottom[j],hash(id),1);mesh.triangle(top[i],bottom[j],top[j],hash(id),1);}
    });
    return new Float32Array(mesh.data);
  }
}
