import { recoilImpulse } from './recoil';
export const MAX_RIBBON_DROPS=8;
export interface RibbonDrop { u:number; v:number; age:number; strength:number; hue:number }
export interface RibbonImpact { hit:boolean; onsets:ArrayLike<number>; noteOn:number; pitch:number; height:number }
/** Musical impacts live in surface coordinates, so waves stay attached through bends. */
export class RibbonRipples {
  readonly drops:RibbonDrop[]=[];
  readonly uniforms=new Float32Array(MAX_RIBBON_DROPS*8);
  private previous=new Float32Array(5);
  private lastPitch=0;
  private serial=0;
  impacts=0;
  step(input:RibbonImpact,dt:number):void {
    for(const drop of this.drops)drop.age+=Math.max(0,Math.min(dt,0.1));
    for(let i=this.drops.length-1;i>=0;i--)if(this.drops[i].age>2.4)this.drops.splice(i,1);
    const add=(u:number,v:number,strength:number,hue:number)=>{
      this.drops.push({u,v,age:0,strength,hue:((hue%1)+1)%1});this.impacts++;
      while(this.drops.length>MAX_RIBBON_DROPS)this.drops.shift();
    };
    if(input.hit || recoilImpulse(input.onsets[0],this.previous[0])) {
      add(0.15+((this.serial++*0.381966)%0.7),0,1,0.53);
    }
    const pitch=Math.round(input.pitch);
    if(recoilImpulse(input.noteOn,this.previous[4]) || (input.noteOn>0.3 && pitch!==this.lastPitch)) {
      add(0.1+Math.max(0,Math.min(1,input.height))*0.8,0.4*Math.sin(this.serial++*2.4),0.85,pitch/12);
    }
    // Instrument accents land nearer alternating edges, sending waves inward.
    for(let i=1;i<4;i++)if(recoilImpulse(input.onsets[i],this.previous[i]))add(0.2+i*0.18,i%2?0.65:-0.65,0.55,0.16+i*0.23);
    for(let i=0;i<4;i++)this.previous[i]=input.onsets[i];this.previous[4]=input.noteOn;this.lastPitch=pitch;
    this.uniforms.fill(0);
    this.drops.forEach((d,i)=>this.uniforms.set([d.u,d.v,d.age,d.strength,d.hue,0,0,0],i*8));
  }
}
export const RIBBON_RIPPLE_GLSL=`
uniform vec4 uDrops[${MAX_RIBBON_DROPS*2}];
uniform int uDropCount;
uniform float uLength,uWidth,uClosed,uRipples;
// Surface distance wraps only along closed paths. Each impact is a damped
// expanding wave packet, with a short central splash and trailing rings.
vec2 rippleDistance(vec2 uv,vec4 drop){
  float du=uv.x-drop.x;
  if(uClosed>0.5)du-=floor(du+0.5);
  return vec2(du*uLength,(uv.y-drop.y)*uWidth);
}
float rippleHeight(vec2 uv){
  float height=0.;
  for(int i=0;i<${MAX_RIBBON_DROPS};i++){
    if(i>=uDropCount)break;
    vec4 drop=uDrops[i*2];float r=length(rippleDistance(uv,drop));
    float front=r-drop.z*0.55;
    float packet=exp(-pow(front/0.11,2.))*sin(front*68.);
    float born=smoothstep(0.,0.045,drop.z);
    height+=packet*exp(-drop.z*1.7)*drop.w*born;
  }
  return height*0.006*uRipples;
}
`;
