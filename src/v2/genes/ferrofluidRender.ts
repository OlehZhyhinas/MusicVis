import { Program,Target,Fullscreen,type GL,type TexFormat } from '../../render/gl';
import { FULLSCREEN_VS } from '../../render/shaders';
import { FerrofluidSimulation } from './ferrofluid';

const FS=`#version 300 es
precision highp float;
in vec2 vUv;
uniform vec4 uDrops[6];
uniform float uTime,uTurn,uTilt,uTension,uSpikes,uFrequency,uSheen,uIridescence,uField,uHue;
uniform int uForm,uStyle;
uniform vec3 uA,uB,uC;
out vec4 frag;
float merge(float a,float b,float k){float h=max(k-abs(a-b),0.)/k;return min(a,b)-h*h*k*.25;}
float scene(vec3 p){
 float d=10.;
 if(uForm==0){vec3 q=p-vec3(0.,-.4,0.);vec3 r=vec3(1.2,.2,.85);float k0=length(q/r),k1=length(q/(r*r));d=k0*(k0-1.)/max(k1,.001);}
 for(int i=0;i<6;i++){
  vec4 drop=uDrops[i];if(drop.w<.01)continue;
  vec3 q=p-drop.xyz;
  float lattice=(cos(q.x*uFrequency)+cos(dot(q.xz,vec2(.5,.866))*uFrequency)+cos(dot(q.xz,vec2(-.5,.866))*uFrequency))/3.;
  float upper=pow(max(0.,q.y/max(length(q),.001)),3.);
  float spike=pow(max(lattice,0.),4.)*uSpikes*(.12+uField*.5)*upper;
  d=merge(d,length(q)-drop.w-spike,uTension);
 }
 return d;
}
vec3 normalAt(vec3 p){vec2 e=vec2(.0015,-.0015);return normalize(e.xyy*scene(p+e.xyy)+e.yyx*scene(p+e.yyx)+e.yxy*scene(p+e.yxy)+e.xxx*scene(p+e.xxx));}
vec3 spectrum(float h){return pow(clamp(abs(fract(h+vec3(0.,2./3.,1./3.))*6.-3.)-1.,0.,1.),vec3(2.2));}
void main(){
 vec2 uv=(vUv-.5)*2.;float angle=uTurn, elevation=.38+uTilt;
 vec3 ro=vec3(sin(angle)*cos(elevation),sin(elevation),cos(angle)*cos(elevation))*4.5;
 vec3 forward=normalize(-ro),right=normalize(cross(forward,vec3(0,1,0))),up=cross(right,forward);
 vec3 rd=normalize(forward*1.85+right*uv.x+up*uv.y);
 float b=dot(ro,rd),disc=b*b-dot(ro,ro)+4.84;
 if(disc<0.){frag=vec4(0);return;}
 float t=max(0.,-b-sqrt(disc)),far=-b+sqrt(disc);bool found=false;
 // Spike displacement has a steeper gradient than a sphere; use conservative steps.
 for(int i=0;i<192;i++){float d=scene(ro+rd*t);if(d<.002){found=true;break;}t+=max(.001,d*.2);if(t>far)break;}
 if(!found){frag=vec4(0);return;}
 vec3 p=ro+rd*t,n=normalAt(p),r=reflect(rd,n);
 float facing=max(0.,dot(n,-rd)),fresnel=pow(1.-facing,3.);
 float ao=clamp(1.-(max(0.,.08-scene(p+n*.08))*3.+max(0.,.2-scene(p+n*.2))*1.5),.35,1.);
 vec3 tint=mix(uA,uB,.5+.5*sin(p.x*2.+p.y*1.5+uTime*.35));
 vec3 spectral=spectrum(uHue+fresnel*.6+p.y*.15+sin(p.x*2.-uTime*.25)*.12);
 tint=mix(tint,spectral,uIridescence*.7);
 float softbox=pow(max(0.,1.-abs(r.y-.55-r.x*.22)*1.7),12.)*smoothstep(-.15,.3,r.z);
 float strip=pow(max(0.,1.-abs(r.x+.65)*2.5),16.)*smoothstep(-.3,.1,r.z);
 float glint=pow(max(0.,dot(r,normalize(vec3(.5,.8,1.)))),100.);
 vec3 color=tint*(.018+.12*fresnel)*ao;
 color+=mix(vec3(.7,.85,1.),tint,.75)*softbox*(.25+uSheen*.75)*ao;
 color+=mix(uC,spectral,uIridescence*.5)*strip*(.3+uSheen*.9);
 color+=vec3(glint*.6)+tint*pow(fresnel,2.)*.35;
 if(uStyle==0)color*=smoothstep(.1,.6,fresnel);
 if(uStyle==2)color*=1.3;
 if(uStyle==3)color*=1.-smoothstep(.16,.28,length(fract(p.xy*45.)-.5));
 frag=vec4(color,1.);
}`;

export class FerrofluidRenderer {
  readonly simulation=new FerrofluidSimulation();
  params:Record<string,number>={};hue=0;style=6;
  private target:Target|null=null;
  private program:Program;private fs:Fullscreen;
  constructor(private gl:GL,private format:TexFormat){this.program=new Program(gl,FULLSCREEN_VS,FS,'ferrofluid');this.fs=new Fullscreen(gl);}
  get texture():WebGLTexture|null{return this.target?.t??null;}
  render(resolution:number,colors:Float32Array):void {
    const gl=this.gl,res=Math.max(192,Math.min(512,Math.round(resolution)));
    if(!this.target||this.target.w!==res){this.target?.dispose();this.target=new Target(gl,res,res,[this.format],gl.LINEAR);}
    this.target.bind();gl.disable(gl.BLEND);gl.disable(gl.DEPTH_TEST);
    const p=this.params,s=this.simulation;
    this.program.use().f4v('uDrops',s.drops).f1('uTime',s.time).f1('uTurn',.5+Math.sin(s.time*p.turn*Math.PI*2)*.6)
      .f1('uTilt',p.tilt).f1('uTension',p.tension).f1('uSpikes',p.spikes).f1('uFrequency',p.frequency)
      .f1('uField',s.field).f1('uHue',this.hue).f1('uSheen',p.sheen).f1('uIridescence',p.iridescence)
      .i1('uForm',Math.round(p.form)).i1('uStyle',this.style)
      .f3('uA',colors[0],colors[1],colors[2]).f3('uB',colors[3],colors[4],colors[5]).f3('uC',colors[6],colors[7],colors[8]);
    this.fs.draw();
  }
  dispose():void{this.target?.dispose();this.program.dispose();this.fs.dispose();}
}
