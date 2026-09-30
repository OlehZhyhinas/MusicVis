import { sampleRibbon, type RibbonPoint } from './ribbon';
export const RIBBON_STEPS = 128;
export const RIBBON_ACROSS = 6;
const N=RIBBON_STEPS+1;
type V = [number,number,number];
const norm=(v:V):V=>{const l=Math.hypot(...v)||1;return [v[0]/l,v[1]/l,v[2]/l];};
const cross=(a:V,b:V):V=>[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]];
const dot=(a:V,b:V)=>a[0]*b[0]+a[1]*b[1]+a[2]*b[2];
export interface RibbonDrive { wind:number; stiffness:number; twist:number; width:number; speed:number; closed:boolean; bass:number; vocals:number; onset:number }
/** Damped spring chain with distance constraints. Each running slot owns its state. */
export class RibbonSimulation {
  readonly position=new Float32Array(N*3);
  readonly velocity=new Float32Array(N*3);
  readonly vertices=new Float32Array(N*(RIBBON_ACROSS+1)*8);
  private rest=new Float32Array(N*3);
  private samples:RibbonPoint[]=[];
  private initialized=false;
  time=0;
  step(path:RibbonPoint[], drive:RibbonDrive, dt:number):void {
    const d=drive, span=Math.min(0.05,Math.max(0,dt));
    this.time+=span*d.speed;
    this.samples=Array.from({length:N},(_,i)=>sampleRibbon(path,i/RIBBON_STEPS,d.closed));
    for(let i=0;i<N;i++) {
      const p=this.samples[i];this.rest.set([p.x,p.y,p.z],i*3);
    }
    if(!this.initialized) { this.position.set(this.rest);this.initialized=true; }
    const steps=Math.max(1,Math.ceil(span*120)), h=span/steps;
    for(let sub=0;sub<steps;sub++) {
      for(let i=0;i<N;i++) {
        const u=i/RIBBON_STEPS, envelope=d.closed?1:Math.sin(Math.PI*u), k=10+70*d.stiffness;
        for(let axis=0;axis<3;axis++) {
          const at=i*3+axis;
          const wave=Math.sin(u*12-this.time*3+axis*2.1)+0.35*Math.sin(u*27-this.time*5+axis);
          const force=d.wind*envelope*(0.7+d.bass*1.2+d.vocals*0.7+d.onset*0.6)*wave*3;
          this.velocity[at]+=(k*(this.rest[at]-this.position[at])+force)*h;
          this.velocity[at]*=Math.exp(-h*(2.5+4*d.stiffness));
          this.position[at]+=this.velocity[at]*h;
        }
      }
      // Neighbor constraints resist stretching without preventing bends or twists.
      for(let pass=0;pass<3;pass++) for(let i=0;i<N-1;i++) {
        const a=i*3,b=a+3;
        const dx=this.position[b]-this.position[a],dy=this.position[b+1]-this.position[a+1],dz=this.position[b+2]-this.position[a+2];
        const length=Math.hypot(dx,dy,dz), rest=Math.hypot(this.rest[b]-this.rest[a],this.rest[b+1]-this.rest[a+1],this.rest[b+2]-this.rest[a+2]);
        const correction=0.4*(length-rest)/Math.max(length,1e-6);
        for(let axis=0;axis<3;axis++) { const delta=[dx,dy,dz][axis]*correction;this.position[a+axis]+=delta;this.position[b+axis]-=delta; }
      }
      if(d.closed) {this.position.set(this.position.subarray(0,3),RIBBON_STEPS*3);this.velocity.set(this.velocity.subarray(0,3),RIBBON_STEPS*3);}
    }
    this.mesh(d);
  }
  private mesh(d:RibbonDrive):void {
    const tangents:V[]=[],sides:V[]=[];
    for(let i=0;i<N;i++) {
      const a=(d.closed?(i-1+RIBBON_STEPS)%RIBBON_STEPS:Math.max(0,i-1))*3;
      const b=(d.closed?(i+1)%RIBBON_STEPS:Math.min(N-1,i+1))*3;
      let t=norm([this.position[b]-this.position[a],this.position[b+1]-this.position[a+1],this.position[b+2]-this.position[a+2]]);
      if(Math.hypot(...t)<0.5) t=i?tangents[i-1]:[1,0,0];
      tangents.push(t);
      // Parallel transport prevents sudden frame flips at vertical tangents.
      const prev=i?sides[i-1]:norm(cross(Math.abs(t[2])<0.9?[0,0,1]:[0,1,0],t));
      let side=norm([prev[0]-t[0]*dot(prev,t),prev[1]-t[1]*dot(prev,t),prev[2]-t[2]*dot(prev,t)]);
      if(Math.hypot(...side)<0.5) side=norm(cross(Math.abs(t[2])<0.9?[0,0,1]:[0,1,0],t));
      sides.push(side);
    }
    const closure=d.closed?Math.atan2(dot(cross(sides[N-1],sides[0]),tangents[0]),dot(sides[N-1],sides[0])):0;
    for(let i=0;i<N;i++) {
      const u=i/RIBBON_STEPS,t=tangents[i],base=sides[i],bn=cross(t,base);
      // Closed ribbons twist continuously and return to the same cross-section.
      const turn=d.twist*(d.closed?Math.sin(u*Math.PI*2):u);
      const angle=(this.samples[i].twist+turn)*Math.PI*2+closure*u
        +d.wind*0.3*Math.sin(u*Math.PI*4-this.time*2)*(d.closed?1:Math.sin(Math.PI*u));
      const side:V=[0,1,2].map(k=>base[k]*Math.cos(angle)+bn[k]*Math.sin(angle)) as V;
      const normal=cross(t,side), width=d.width*this.samples[i].width;
      for(let j=0;j<=RIBBON_ACROSS;j++) {
        const v=j/RIBBON_ACROSS*2-1, at=(i*(RIBBON_ACROSS+1)+j)*8;
        const n=norm([normal[0]+side[0]*v*0.12,normal[1]+side[1]*v*0.12,normal[2]+side[2]*v*0.12]);
        for(let k=0;k<3;k++) {this.vertices[at+k]=this.position[i*3+k]+side[k]*v*width+normal[k]*(1-v*v)*width*0.06;this.vertices[at+3+k]=n[k];}
        this.vertices[at+6]=u;this.vertices[at+7]=v;
      }
    }
  }
}
