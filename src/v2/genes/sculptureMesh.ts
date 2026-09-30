// Small 3D solid meshes with explicit faces, depth, and studio lighting.
import { Program, Target, createTexture, formats, type GL, type TexFormat } from '../../render/gl';
import { MESH_STRIDE } from './meshGeometry';
const VS=`#version 300 es
precision highp float;
layout(location=0) in vec3 aPosition;layout(location=1) in vec3 aNormal;
layout(location=2) in vec3 aBary;layout(location=3) in vec2 aInfo;
uniform float uTurn,uTilt;
out vec3 vNormal,vPosition,vBary;out vec2 vInfo;
void main(){
 float c=cos(uTurn),s=sin(uTurn),ct=cos(uTilt),st=sin(uTilt);
 mat3 r=mat3(1.,0.,0.,0.,ct,st,0.,-st,ct)*mat3(c,0.,-s,0.,1.,0.,s,0.,c);
 vPosition=r*aPosition;vNormal=r*aNormal;vBary=aBary;vInfo=aInfo;
 vec3 p=vPosition/2.15;gl_Position=vec4(p.xy,-p.z*0.35,1.-p.z*0.28);
}`;
const FS=`#version 300 es
precision highp float;
in vec3 vNormal,vPosition,vBary;in vec2 vInfo;
uniform float uTime,uHue,uPrism,uClarity,uPulse;uniform int uMode,uStyle;
uniform vec3 uA,uB,uC;
uniform sampler2D uBehind;uniform float uGlassPass;uniform vec2 uResolution;
out vec4 frag;
vec3 spectral(float h){vec3 p=abs(fract(h+vec3(0.,2./3.,1./3.))*6.-3.);return pow(clamp(p-1.,0.,1.),vec3(2.2));}
void main(){
 vec3 n=normalize(vNormal)*(gl_FrontFacing?1.:-1.),view=normalize(vec3(0.,0.,5.)-vPosition);
 vec3 r=reflect(-view,n);float facing=max(0.,dot(n,view)),fresnel=pow(1.-facing,3.);
 vec3 bary=vBary/max(fwidth(vBary),vec3(0.0001));
 float distance=min(bary.x,min(bary.y,bary.z));
 if(uMode==1&&vInfo.y>1.5)distance=vInfo.y<2.5?min(bary.x,bary.z):min(bary.x,bary.y);
 float edge=1.-smoothstep(0.5,1.6,distance);
 float softEdge=1.-smoothstep(0.,7.,min(bary.x,min(bary.y,bary.z)));
 float strip=pow(max(0.,1.-abs(r.y-0.48-r.x*0.3)*2.2),18.);
 float rim=pow(max(0.,dot(n,normalize(vec3(-0.6,0.8,1.)))),28.);
 vec3 tint=mix(mix(uA,uB,0.5+0.5*sin(vInfo.x*8.)),uC,0.5+0.5*cos(vInfo.x*5.));
 vec3 prism=mix(tint,spectral(uHue+vInfo.x*0.63+fresnel*0.6+vPosition.x*0.09),uPrism);
 vec3 color;
 if(uMode==0){
   // Smoked glass: dark faces, coloured internal reflection, bright cut edges.
   float caustic=pow(0.5+0.5*sin(dot(vPosition.xy,vec2(8.,5.))+n.x*12.+n.y*8.),12.);
   color=mix(tint,prism,uPrism)*(0.025+0.16*fresnel+caustic*0.12*uClarity);
   color+=mix(vec3(0.45,0.6,0.7),prism,uPrism*0.45)*strip*(0.3+0.7*uClarity);
   vec2 bend=n.xy*(0.008+0.014*fresnel);
   vec3 transmission=texture(uBehind,gl_FragCoord.xy/uResolution+bend).rgb;
   color+=transmission*uGlassPass*uClarity*(1.-fresnel)*0.5;
   color+=prism*(edge*(0.2+0.7*vInfo.y)+softEdge*0.08)*(0.5+uPulse);
   color+=vec3(0.5,0.65,0.8)*rim*0.45;
 }else{
   // Folded lacquer: vivid inner faces and almost black outer faces.
   float light=0.25+0.75*max(0.,dot(n,normalize(vec3(-0.5,0.7,1.))));
   vec3 paint=mix(tint,prism,uPrism);
   float inner=gl_FrontFacing?0.2+0.35*pow(1.-facing,0.7):0.06;
   color=paint*light*(inner+uClarity*0.12)*(0.65+0.35*sin(vPosition.x*3.+vPosition.y*4.-uTime*0.4))+paint*edge*0.32;
   color+=mix(vec3(0.8),paint,0.5)*strip*0.5+vec3(rim*0.18);
 }
 if(uStyle==0)color*=edge;
 if(uStyle==2)color*=1.3;
 if(uStyle==3)color*=(1.-smoothstep(0.18,0.28,length(fract(vPosition.xy*55.)-0.5)));
 frag=vec4(color,1.);
}`;
export class SculptureMeshRenderer {
  private program:Program;private vao:WebGLVertexArrayObject;private vbo:WebGLBuffer;
  private target:Target|null=null;private behind:Target|null=null;private black:WebGLTexture;private depth:WebGLRenderbuffer|null=null;
  time=0;turn=0;tilt=0;hue=0;prism=0.8;clarity=0.7;pulse=0;mode=0;style=6;
  vertices:Float32Array=new Float32Array();
  constructor(private gl:GL,private format:TexFormat){
    this.black=createTexture(gl,1,1,formats(gl).rgba8,gl.NEAREST,new Uint8Array([0,0,0,255]));
    this.program=new Program(gl,VS,FS,'sculpture-solid');this.vao=gl.createVertexArray()!;this.vbo=gl.createBuffer()!;
    gl.bindVertexArray(this.vao);gl.bindBuffer(gl.ARRAY_BUFFER,this.vbo);
    for(const [loc,n,offset] of [[0,3,0],[1,3,12],[2,3,24],[3,2,36]]){gl.enableVertexAttribArray(loc);gl.vertexAttribPointer(loc,n,gl.FLOAT,false,MESH_STRIDE*4,offset);}
    gl.bindVertexArray(null);
  }
  get texture():WebGLTexture|null{return this.target?.t??null;}
  render(resolution:number,colors:Float32Array):void {
    const gl=this.gl,res=Math.max(256,Math.min(1024,Math.round(resolution)));
    if(!this.target||this.target.w!==res){
      this.target?.dispose();this.behind?.dispose();gl.deleteRenderbuffer(this.depth);this.target=new Target(gl,res,res,[this.format],gl.LINEAR);
      this.target.bind();this.depth=gl.createRenderbuffer()!;gl.bindRenderbuffer(gl.RENDERBUFFER,this.depth);
      gl.renderbufferStorage(gl.RENDERBUFFER,gl.DEPTH_COMPONENT24,res,res);gl.framebufferRenderbuffer(gl.FRAMEBUFFER,gl.DEPTH_ATTACHMENT,gl.RENDERBUFFER,this.depth);
      if(gl.checkFramebufferStatus(gl.FRAMEBUFFER)!==gl.FRAMEBUFFER_COMPLETE)throw Error('Sculpture depth target incomplete');
      this.behind=new Target(gl,res,res,[this.format],gl.LINEAR);this.behind.bind();gl.framebufferRenderbuffer(gl.FRAMEBUFFER,gl.DEPTH_ATTACHMENT,gl.RENDERBUFFER,this.depth);
    }
    this.target.bind();gl.disable(gl.BLEND);gl.enable(gl.DEPTH_TEST);gl.depthFunc(gl.LEQUAL);gl.depthMask(true);gl.clearDepth(1);
    gl.clearColor(0,0,0,0);gl.clear(gl.COLOR_BUFFER_BIT|gl.DEPTH_BUFFER_BIT);
    this.program.use().tex('uBehind',this.black).f1('uGlassPass',0).f2('uResolution',res,res).f1('uTurn',this.turn).f1('uTilt',this.tilt).f1('uHue',this.hue).f1('uTime',this.time)
      .f1('uPrism',this.prism).f1('uClarity',this.clarity).f1('uPulse',this.pulse).i1('uMode',this.mode).i1('uStyle',this.style)
      .f3('uA',colors[0],colors[1],colors[2]).f3('uB',colors[3],colors[4],colors[5]).f3('uC',colors[6],colors[7],colors[8]);
    gl.bindVertexArray(this.vao);gl.bindBuffer(gl.ARRAY_BUFFER,this.vbo);gl.bufferData(gl.ARRAY_BUFFER,this.vertices,gl.DYNAMIC_DRAW);
    if(this.mode===0){
      this.behind!.bind();gl.clear(gl.COLOR_BUFFER_BIT|gl.DEPTH_BUFFER_BIT);gl.enable(gl.CULL_FACE);gl.cullFace(gl.FRONT);
      gl.drawArrays(gl.TRIANGLES,0,this.vertices.length/MESH_STRIDE);gl.disable(gl.CULL_FACE);
      this.target.bind();gl.clear(gl.COLOR_BUFFER_BIT|gl.DEPTH_BUFFER_BIT);
      this.program.tex('uBehind',this.behind!.t).f1('uGlassPass',1);
    }
    gl.drawArrays(gl.TRIANGLES,0,this.vertices.length/MESH_STRIDE);gl.bindVertexArray(null);gl.disable(gl.DEPTH_TEST);
  }
  dispose():void {this.target?.dispose();this.behind?.dispose();this.gl.deleteTexture(this.black);this.program.dispose();this.gl.deleteRenderbuffer(this.depth);this.gl.deleteBuffer(this.vbo);this.gl.deleteVertexArray(this.vao);}
}
