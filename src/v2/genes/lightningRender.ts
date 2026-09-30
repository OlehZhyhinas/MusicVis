import { Program,Target,type GL,type TexFormat } from '../../render/gl';
import { LightningSimulation,MAX_LIGHTNING_SEGMENTS,type LightningParams } from './lightning';
const VS=`#version 300 es
precision highp float;
layout(location=0) in vec2 aPosition;
layout(location=1) in vec3 aLight;
out vec3 vLight;
void main(){gl_Position=vec4(aPosition/1.8,0.,1.);vLight=aLight;}`;
const FS=`#version 300 es
precision highp float;
in vec3 vLight;uniform float uHue;out vec4 frag;
vec3 hsv(float h){vec3 p=abs(fract(vec3(h)+vec3(0.,2./3.,1./3.))*6.-3.);return clamp(p-1.,0.,1.);}
void main(){float d=abs(vLight.x);vec3 col=pow(hsv(vLight.y+uHue),vec3(2.2));
vec3 light=col*(exp(-d*d)*1.6+exp(-d*d*0.15)*0.22)+mix(col,vec3(1.),0.75)*exp(-d*d*18.)*2.;
frag=vec4(light*vLight.z,1.);}`;
export class LightningRenderer {
  readonly simulation=new LightningSimulation();
  private program:Program;private vao:WebGLVertexArrayObject;private buffer:WebGLBuffer;private target:Target|null=null;
  private vertices=new Float32Array(MAX_LIGHTNING_SEGMENTS*6*5);
  params!:LightningParams;hue=0;
  constructor(private gl:GL,private format:TexFormat){
    this.program=new Program(gl,VS,FS,'lightning-arcs');this.vao=gl.createVertexArray()!;this.buffer=gl.createBuffer()!;
    gl.bindVertexArray(this.vao);gl.bindBuffer(gl.ARRAY_BUFFER,this.buffer);gl.bufferData(gl.ARRAY_BUFFER,this.vertices.byteLength,gl.DYNAMIC_DRAW);
    gl.enableVertexAttribArray(0);gl.vertexAttribPointer(0,2,gl.FLOAT,false,20,0);gl.enableVertexAttribArray(1);gl.vertexAttribPointer(1,3,gl.FLOAT,false,20,8);gl.bindVertexArray(null);
  }
  get texture():WebGLTexture|null{return this.target?.t??null;}
  render(resolution:number):void {
    const gl=this.gl,p=this.params,res=Math.max(256,Math.min(1536,Math.round(resolution)));let n=0;
    // Newest strokes get the geometry budget first during a very dense passage.
    for(let i=this.simulation.bolts.length-1;i>=0;i--) {
      const bolt=this.simulation.bolts[i],front=bolt.age*p.speed/0.18,life=Math.exp(-Math.max(0,bolt.age-0.18/p.speed)/p.decay);
      for(const seg of bolt.segments){
        if(front<=seg.start || n>=MAX_LIGHTNING_SEGMENTS)continue;
        const t=Math.min(1,(front-seg.start)/Math.max(1e-6,seg.end-seg.start));
        const bx=seg.ax+(seg.bx-seg.ax)*t,by=seg.ay+(seg.by-seg.ay)*t,dx=bx-seg.ax,dy=by-seg.ay,len=Math.hypot(dx,dy)||1;
        const w=p.width*(0.5+seg.weight)*4.5,nx=-dy/len*w,ny=dx/len*w;
        const strength=bolt.strength*seg.weight*life;
        const corners=[[seg.ax-nx,seg.ay-ny,-4.5],[bx-nx,by-ny,-4.5],[seg.ax+nx,seg.ay+ny,4.5],[seg.ax+nx,seg.ay+ny,4.5],[bx-nx,by-ny,-4.5],[bx+nx,by+ny,4.5]];
        for(let j=0;j<6;j++)this.vertices.set([...corners[j],bolt.hue,strength],(n*6+j)*5);n++;
      }
    }
    if(!this.target || this.target.w!==res){this.target?.dispose();this.target=new Target(gl,res,res,[this.format],gl.LINEAR);}
    this.target.bind();gl.clearColor(0,0,0,0);gl.clear(gl.COLOR_BUFFER_BIT);gl.enable(gl.BLEND);gl.blendFunc(gl.ONE,gl.ONE);
    this.program.use().f1('uHue',this.hue);gl.bindVertexArray(this.vao);gl.bindBuffer(gl.ARRAY_BUFFER,this.buffer);
    gl.bufferSubData(gl.ARRAY_BUFFER,0,this.vertices.subarray(0,n*30));gl.drawArrays(gl.TRIANGLES,0,n*6);gl.bindVertexArray(null);gl.disable(gl.BLEND);
  }
  dispose():void{this.target?.dispose();this.program.dispose();this.gl.deleteBuffer(this.buffer);this.gl.deleteVertexArray(this.vao);}
}
