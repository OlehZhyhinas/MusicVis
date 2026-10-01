import type { Schema } from '../genome';
import { recoilImpulse } from './recoil';
import { SolidMesh,add,sub,mul,unit,cross,dot,hash,type V3 } from './meshGeometry';
export const FRACTURE_SCHEMA:Schema={
 size:{min:0.05,max:0.6,def:0.36},pieces:{min:4,max:14,def:9,int:true},form:{min:0,max:2,def:0,choices:[0,1,2]},
 open:{min:0,max:1.2,def:0.25},impact:{min:0,max:1,def:0.7},recovery:{min:0.4,max:4,def:1.5},
 freedom:{min:0,max:1,def:0.95},drift:{min:0,max:1,def:0.65},
 thickness:{min:0.01,max:0.12,def:0.035},tumble:{min:0,max:1,def:0.6},
 turn:{min:-0.08,max:0.08,def:0.015},tilt:{min:-0.4,max:0.4,def:0.07},
 prism:{min:0,max:1,def:0.85},clarity:{min:0,max:1,def:0.7},
};
export interface SculptureMusic {hit:boolean;onset:number;noteOn:number;held:number;legato:number;bass:number;vocals:number;pitch:number}
type Quaternion=[number,number,number,number];
interface Shard {
  points:V3[];center:V3;position:V3;velocity:V3;
  orientation:Quaternion;angularVelocity:V3;delay:number;lastHit:number;
}
function rotate(q:Quaternion,v:V3):V3 {
  const axis:V3=[q[0],q[1],q[2]],t=mul(cross(axis,v),2);
  return add(v,add(mul(t,q[3]),cross(axis,t)));
}
const randomDirection=(id:number):V3=>unit([hash(id)-0.5,hash(id+71)-0.5,hash(id+139)-0.5]);
/** Independent solid shards with linear/angular momentum and optional attraction to their original slab. */
export class FractureSimulation {
  shards:Shard[]=[];private key='';private previous=0;private previousHit=false;private age=100;private hitSerial=0;time=0;
  private make(n:number,form:number,freedom:number):void {
    this.shards=[];const rows=Math.round(n*0.65),points:V3[]=[];
    for(let y=0;y<=rows;y++)for(let x=0;x<=n;x++){
      const i=y*(n+1)+x;
      points.push([(x+(x>0&&x<n?(hash(i)-0.5)*0.65:0))/n*2.65-1.325,(y+(y>0&&y<rows?(hash(i+513)-0.5)*0.65:0))/rows*1.65-0.825,0]);
    }
    const triangle=(ids:number[])=>{
      const ps=ids.map(i=>points[i]),c=mul(add(add(ps[0],ps[1]),ps[2]),1/3),id=this.shards.length;
      if(form===1&&Math.hypot(c[0]/1.325,c[1]/0.825)>0.96)return;
      if(form===2&&Math.abs(c[0])/1.325+Math.abs(c[1])/0.825>1.04)return;
      const cloud=mul(randomDirection(id+351),1.3*Math.cbrt(hash(id+821)));
      const axis=randomDirection(id+51),angle=freedom*(hash(id+98)-0.5)*Math.PI;
      this.shards.push({points:ps,center:c,position:add(mul(c,1-freedom),mul(cloud,freedom)),velocity:[0,0,0],
        orientation:[...mul(axis,Math.sin(angle)),Math.cos(angle)],angularVelocity:[0,0,0],
        delay:hash(id+91)*0.14,lastHit:this.hitSerial});
    };
    for(let y=0;y<rows;y++)for(let x=0;x<n;x++){
      const a=y*(n+1)+x,b=a+1,c=a+n+1,d=c+1;
      if(hash(a)>0.5){triangle([a,b,d]);triangle([a,d,c]);}else{triangle([a,b,c]);triangle([b,d,c]);}
    }
  }
  step(p:Record<string,number>,music:SculptureMusic,dt:number):Float32Array {
    const freedom=p.freedom??FRACTURE_SCHEMA.freedom.def,drift=p.drift??FRACTURE_SCHEMA.drift.def;
    const key=p.pieces+':'+p.form;if(key!==this.key){this.make(Math.round(p.pieces),p.form,freedom);this.key=key;}
    const h=Math.min(0.05,Math.max(0,dt));this.time+=h;this.age+=h;
    if((music.hit&&!this.previousHit)||recoilImpulse(music.onset,this.previous)){this.age=0;this.hitSerial++;}
    this.previous=music.onset;this.previousHit=music.hit;const mesh=new SolidMesh();
    const gather=music.held*music.legato,k=26*p.recovery*Math.pow(1-freedom,3);
    this.shards.forEach((sh,id)=>{
      // A fresh, different impulse per attack; a held hit gate cannot keep kicking.
      if(sh.lastHit!==this.hitSerial&&this.age>=sh.delay){
        const direction=randomDirection(id*13+this.hitSerial*977);
        sh.velocity=add(sh.velocity,mul(add(mul(direction,0.85),mul(unit(sh.position),0.15)),p.impact*(1.2+hash(id+37))));
        sh.angularVelocity=add(sh.angularVelocity,mul(randomDirection(id*17+this.hitSerial*563),p.impact*p.tumble*3));
        sh.lastHit=this.hitSerial;
      }
      const home=add(mul(sh.center,1+p.open*0.35),[0,0,(hash(id+213)-0.5)*p.open]);
      const phase=hash(id+421)*Math.PI*2,rate=0.35+hash(id+617)*0.6;
      for(let step=0;step<3;step++){
        const dt=h/3,r=Math.hypot(...sh.position),boundary=1.6*(1-freedom)+(1.08+p.open*0.2)*freedom,
          spring=8*Math.max(0,r-boundary)/Math.max(r,0.001);
        const wind:V3=[Math.sin(this.time*rate+phase+sh.position[1]),Math.cos(this.time*rate*0.81+phase*1.7+sh.position[2]),Math.sin(this.time*rate*0.63+phase*2.3+sh.position[0])];
        for(let axis=0;axis<3;axis++){
          const force=k*(home[axis]-sh.position[axis])+freedom*drift*(0.5+music.bass*0.4)*wind[axis]
            -sh.position[axis]*(spring+freedom*gather*0.45*p.recovery);
          sh.velocity[axis]+=(force-(0.65+Math.sqrt(k)*1.5)*sh.velocity[axis])*dt;
          sh.position[axis]+=sh.velocity[axis]*dt;
        }
        // Soft confinement normally suffices; this bound also handles dense drum rolls.
        const radius=Math.hypot(...sh.position);
        if(radius>1.65){
          const n=mul(sh.position,1/radius);sh.position=mul(n,1.65);
          sh.velocity=sub(sh.velocity,mul(n,Math.max(0,dot(sh.velocity,n))*1.4));
        }
        const angularK=10*p.recovery*Math.pow(1-freedom,2),q=sh.orientation,sign=q[3]<0?-1:1;
        for(let axis=0;axis<3;axis++){
          const torque=freedom*drift*p.tumble*0.7*wind[(axis+1)%3]-2*q[axis]*sign*angularK;
          sh.angularVelocity[axis]+=(torque-(0.7+Math.sqrt(angularK)*1.4)*sh.angularVelocity[axis])*dt;
        }
        const [x,y,z,w]=q,[wx,wy,wz]=sh.angularVelocity;
        const next:Quaternion=[x+(wx*w+wy*z-wz*y)*dt*0.5,y+(-wx*z+wy*w+wz*x)*dt*0.5,z+(wx*y-wy*x+wz*w)*dt*0.5,w-(wx*x+wy*y+wz*z)*dt*0.5];
        const length=Math.hypot(...next);sh.orientation=next.map(v=>v/length) as Quaternion;
      }
      const points=sh.points.map(q=>add(sh.position,rotate(sh.orientation,mul(sub(q,sh.center),0.97))));
      const normal=rotate(sh.orientation,[0,0,1]),half=mul(normal,p.thickness*0.5);
      const top=points.map(q=>add(q,half)),bottom=points.map(q=>sub(q,half));
      mesh.triangle(top[0],top[1],top[2],hash(id));mesh.triangle(bottom[2],bottom[1],bottom[0],hash(id));
      for(let i=0;i<3;i++){const j=(i+1)%3;mesh.triangle(top[i],bottom[i],bottom[j],hash(id),1);mesh.triangle(top[i],bottom[j],top[j],hash(id),1);}
    });
    return new Float32Array(mesh.data);
  }
}
