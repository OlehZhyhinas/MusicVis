import { Program, Target, type GL, type TexFormat } from '../../render/gl';
import { RibbonSimulation, RIBBON_ACROSS, RIBBON_STEPS } from './ribbonSim';
const VS=`#version 300 es
precision highp float;
layout(location=0) in vec3 aPosition;
layout(location=1) in vec3 aNormal;
layout(location=2) in vec2 aUV;
uniform float uExtent,uTurn,uTilt;
out vec3 vNormal; out vec2 vUV;
void main(){
  float c=cos(uTurn),s=sin(uTurn),ct=cos(uTilt),st=sin(uTilt);
  mat3 ry=mat3(c,0.,-s,0.,1.,0.,s,0.,c),rx=mat3(1.,0.,0.,0.,ct,st,0.,-st,ct);
  vec3 p=rx*ry*aPosition/uExtent;
  vNormal=rx*ry*aNormal; vUV=aUV;
  gl_Position=vec4(p.xy,-p.z*0.25,1.0-p.z*0.16);
}`;
const FS=`#version 300 es
precision highp float;
in vec3 vNormal; in vec2 vUV;
uniform float uTime,uFlow,uSheen,uHue; uniform int uStyle;
uniform vec3 uA,uB,uC;
out vec4 frag;
vec3 hsv(vec3 c){ vec3 p=abs(fract(c.xxx+vec3(0.,2./3.,1./3.))*6.-3.);return c.z*mix(vec3(1.),clamp(p-1.,0.,1.),c.y); }
void main(){
  vec3 n=normalize(vNormal)*(gl_FrontFacing?1.:-1.);
  vec3 lamp=normalize(vec3(-0.4,0.7,1.));
  float diffuse=abs(dot(n,lamp));
  float spec=pow(max(0.,dot(n,normalize(lamp+vec3(0.,0.,1.)))),36.);
  float edge=pow(abs(vUV.y),14.);
  float phase=vUV.x*3.0+vUV.y*0.2+sin(vUV.x*18.-uTime)*0.1-uTime*0.18;
  float pulse=pow(0.5+0.5*cos(phase*6.28318),12.);
  vec3 neon=pow(hsv(vec3(fract(uHue+vUV.x*1.4+vUV.y*0.13-uTime*0.07),0.98,1.)),vec3(2.2));
  vec3 palette=mix(mix(uA,uB,0.5+0.5*sin(vUV.x*9.)),uC,0.5+0.5*cos(vUV.x*7.+2.));
  vec3 color=mix(palette,neon,uFlow)*(0.16+0.85*diffuse+uFlow*pulse*1.1);
  color+=mix(neon,vec3(1.),0.3)*spec*uSheen*0.7+neon*edge*0.6;
  if(uStyle==0) color*=smoothstep(0.88,0.98,abs(vUV.y));
  if(uStyle==2) color*=1.2+edge;
  if(uStyle==3) color*=smoothstep(0.27,0.18,length(fract(vUV*vec2(55.,12.))-0.5));
  if(uStyle==4) color*=0.6+0.4*sin(vUV.y*80.+vUV.x*20.);
  if(uStyle==5) color+=vec3(spec)*0.9;
  frag=vec4(color,1.);
}`;
/** Actual rasterized 3D mesh with a depth attachment, shared through the normal body compositor. */
export class RibbonRenderer {
  readonly simulation=new RibbonSimulation();
  private program:Program;
  private target:Target|null=null;
  private depth:WebGLRenderbuffer|null=null;
  private vao:WebGLVertexArrayObject;
  private buffer:WebGLBuffer;
  private index:WebGLBuffer;
  private count:number;
  extent=2;
  flow=0.8; turn=0.025; tilt=0.08; sheen=0.5; hue=0; style=6;
  constructor(private gl:GL, private format:TexFormat) {
    this.program=new Program(gl,VS,FS,'ribbon-mesh');
    this.vao=gl.createVertexArray()!;this.buffer=gl.createBuffer()!;this.index=gl.createBuffer()!;
    gl.bindVertexArray(this.vao);gl.bindBuffer(gl.ARRAY_BUFFER,this.buffer);
    gl.bufferData(gl.ARRAY_BUFFER,this.simulation.vertices.byteLength,gl.DYNAMIC_DRAW);
    for(const [location,size,offset] of [[0,3,0],[1,3,12],[2,2,24]]) {gl.enableVertexAttribArray(location);gl.vertexAttribPointer(location,size,gl.FLOAT,false,32,offset);}
    const indices:number[]=[];
    for(let i=0;i<RIBBON_STEPS;i++) for(let j=0;j<RIBBON_ACROSS;j++) {
      const a=i*(RIBBON_ACROSS+1)+j,b=a+RIBBON_ACROSS+1;
      indices.push(a,b,a+1,a+1,b,b+1);
    }
    this.count=indices.length;gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER,this.index);gl.bufferData(gl.ELEMENT_ARRAY_BUFFER,new Uint16Array(indices),gl.STATIC_DRAW);gl.bindVertexArray(null);
  }
  get texture():WebGLTexture|null{return this.target?.t??null;}
  render(resolution:number, colors:Float32Array):void {
    const gl=this.gl,res=Math.max(128,Math.min(1024,Math.round(resolution)));
    if(!this.target || this.target.w!==res) {
      this.target?.dispose();if(this.depth)gl.deleteRenderbuffer(this.depth);
      this.target=new Target(gl,res,res,[this.format],gl.LINEAR);
      this.target.bind();this.depth=gl.createRenderbuffer()!;gl.bindRenderbuffer(gl.RENDERBUFFER,this.depth);
      gl.renderbufferStorage(gl.RENDERBUFFER,gl.DEPTH_COMPONENT24,res,res);
      gl.framebufferRenderbuffer(gl.FRAMEBUFFER,gl.DEPTH_ATTACHMENT,gl.RENDERBUFFER,this.depth);
      if(gl.checkFramebufferStatus(gl.FRAMEBUFFER)!==gl.FRAMEBUFFER_COMPLETE)throw Error('Ribbon depth framebuffer incomplete');
    }
    this.target.bind();gl.disable(gl.BLEND);gl.enable(gl.DEPTH_TEST);gl.depthFunc(gl.LEQUAL);gl.depthMask(true);gl.clearDepth(1);
    gl.clearColor(0,0,0,0);gl.clear(gl.COLOR_BUFFER_BIT|gl.DEPTH_BUFFER_BIT);
    this.program.use().f1('uExtent',this.extent).f1('uTurn',this.turn*this.simulation.time*Math.PI*2).f1('uTilt',this.tilt*Math.PI*2)
      .f1('uTime',this.simulation.time).f1('uFlow',this.flow).f1('uSheen',this.sheen).f1('uHue',this.hue).i1('uStyle',this.style)
      .f3('uA',colors[0],colors[1],colors[2]).f3('uB',colors[3],colors[4],colors[5]).f3('uC',colors[6],colors[7],colors[8]);
    gl.bindVertexArray(this.vao);gl.bindBuffer(gl.ARRAY_BUFFER,this.buffer);gl.bufferSubData(gl.ARRAY_BUFFER,0,this.simulation.vertices);
    gl.drawElements(gl.TRIANGLES,this.count,gl.UNSIGNED_SHORT,0);gl.bindVertexArray(null);gl.disable(gl.DEPTH_TEST);
  }
  dispose():void {const gl=this.gl;this.target?.dispose();gl.deleteRenderbuffer(this.depth);gl.deleteBuffer(this.buffer);gl.deleteBuffer(this.index);gl.deleteVertexArray(this.vao);this.program.dispose();}
}
