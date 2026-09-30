import type { Schema } from '../genome';
import type { SculptureMusic } from './fracture';
import { recoilImpulse } from './recoil';
import { SolidMesh,add,sub,mul,unit,dot,cross,rotateAxis,type V3 } from './meshGeometry';
export const ORIGAMI_SCHEMA:Schema={
 size:{min:0.05,max:0.6,def:0.35},form:{min:0,max:2,def:0,choices:[0,1,2]},
 folds:{min:4,max:18,def:10,int:true},rows:{min:2,max:10,def:6,int:true},
 fold:{min:0,max:1,def:0.65},response:{min:0,max:1,def:0.7},stiffness:{min:0.5,max:4,def:1.5},
 curl:{min:-0.6,max:0.6,def:0},turn:{min:-0.08,max:0.08,def:0.012},tilt:{min:-0.4,max:0.4,def:0.1},
 prism:{min:0,max:1,def:0.65},clarity:{min:0,max:1,def:0.6},
};
/** Connected panels: Miura parallelograms, a triangular strip, or a hinged paper fan. */
export class OrigamiSimulation {
 time=0;angle=0;velocity=0;private previous=0;private impulse=0;
 /** Rest and posed vertices allow edge-length/hinge checks independent of the renderer. */
 rest:V3[]=[];points:V3[]=[];faces:number[][]=[];
 step(p:Record<string,number>,music:SculptureMusic,dt:number):Float32Array {
  const h=Math.min(0.05,Math.max(0,dt));this.time+=h;
  if(music.hit||recoilImpulse(music.onset,this.previous))this.impulse=1;
  this.previous=music.onset;this.impulse*=Math.exp(-h*5);
  const target=Math.max(0.02,Math.min(0.98,p.fold+p.response*(this.impulse*0.25-music.held*music.legato*0.3)));
  const k=24*p.stiffness;
  for(let i=0;i<3;i++){this.velocity+=(k*(target-this.angle)-2*Math.sqrt(k)*0.85*this.velocity)*h/3;this.angle+=this.velocity*h/3;}
  this.angle=Math.max(0,Math.min(1,this.angle));this.rest=[];this.points=[];this.faces=[];
  const n=Math.round(p.folds),rows=Math.round(p.rows),mesh=new SolidMesh();
  if(p.form===0){
    // Isometric Miura sheet: both edge lengths and panel angles stay constant while folding.
    const a=2.7/n,b=1.8/rows,alpha=1.15,theta=this.angle*1.08;
    const dx=a*Math.cos(theta),dz=a*Math.sin(theta),skew=b*Math.cos(alpha)/Math.cos(theta),dy=Math.sqrt(Math.max(0.0001,b*b-skew*skew));
    for(let j=0;j<=rows;j++)for(let i=0;i<=n;i++){
      this.rest.push([i*a+(j%2)*b*Math.cos(alpha),j*b*Math.sin(alpha),0]);
      this.points.push([i*dx+(j%2)*skew,j*dy,(i%2)*dz]);
    }
    for(let j=0;j<rows;j++)for(let i=0;i<n;i++){
      const a=j*(n+1)+i,b=a+1,c=a+n+2,d=a+n+1;this.faces.push([a,b,c,d]);
    }
  }else{
    const fan=p.form===2;
    if(fan){
      this.rest.push([0,0,0]);
      for(let i=0;i<=n;i++){const a=(i/n-0.5)*Math.PI*1.55;this.rest.push([Math.cos(a)*1.45,Math.sin(a)*1.45,0]);}
      for(let i=0;i<n;i++)this.faces.push([0,i+1,i+2]);
    }else{
      for(let i=0;i<n+2;i++)this.rest.push([(i/(n+1)-0.5)*2.8,i%2?0.62:-0.62,0]);
      for(let i=0;i<n;i++)this.faces.push(i%2?[i+1,i,i+2]:[i,i+1,i+2]);
    }
    this.points=this.rest.map(v=>[...v] as V3);
    for(let i=1;i<this.faces.length;i++){
      const prev=this.faces[i-1],face=this.faces[i],shared=prev.filter(v=>face.includes(v));
      const ia=shared[0],ib=shared[1],old=prev.find(v=>!shared.includes(v))!,next=face.find(v=>!shared.includes(v))!;
      const ra=this.rest[ia],rb=this.rest[ib],re=unit(sub(rb,ra)),rn=unit(cross(sub(rb,ra),sub(this.rest[old],ra))),ry=cross(rn,re);
      const a=this.points[ia],e=unit(sub(this.points[ib],a)),normal=unit(cross(e,sub(this.points[old],a))),y=cross(normal,e);
      const rel=sub(this.rest[next],ra),flat=add(mul(e,dot(rel,re)),mul(y,dot(rel,ry)));
      const angle=(i%2?1:-1)*this.angle*(fan?1.65:1.15)+p.curl*0.35;
      this.points[next]=add(a,rotateAxis(flat,e,angle));
    }
  }
  const center=mul(this.points.reduce((a,b)=>add(a,b),[0,0,0] as V3),1/this.points.length);
  this.points=this.points.map(v=>sub(v,center));
  this.faces.forEach((f,i)=>{
    const id=i/Math.max(1,this.faces.length-1);
    if(f.length===4){mesh.triangle(this.points[f[0]],this.points[f[1]],this.points[f[2]],id,2);mesh.triangle(this.points[f[0]],this.points[f[2]],this.points[f[3]],id,3);}
    else mesh.triangle(this.points[f[0]],this.points[f[1]],this.points[f[2]],id);
  });
  return new Float32Array(mesh.data);
 }
}
