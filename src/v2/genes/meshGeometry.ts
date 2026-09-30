// Pure geometry shared by the solid-surface simulations.
export type V3=[number,number,number];
export const add=(a:V3,b:V3):V3=>[a[0]+b[0],a[1]+b[1],a[2]+b[2]];
export const sub=(a:V3,b:V3):V3=>[a[0]-b[0],a[1]-b[1],a[2]-b[2]];
export const mul=(a:V3,s:number):V3=>[a[0]*s,a[1]*s,a[2]*s];
export const cross=(a:V3,b:V3):V3=>[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]];
export const unit=(a:V3):V3=>mul(a,1/(Math.hypot(...a)||1));
export const dot=(a:V3,b:V3)=>a[0]*b[0]+a[1]*b[1]+a[2]*b[2];
export const hash=(i:number)=>{const x=Math.sin(i*127.1+311.7)*43758.5453;return x-Math.floor(x);};
export function rotateAxis(p:V3,axis:V3,a:number):V3 {
  const c=Math.cos(a),s=Math.sin(a);return add(add(mul(p,c),mul(cross(axis,p),s)),mul(axis,dot(axis,p)*(1-c)));
}
/** Position, normal, barycentric distance to edges, identity, side flag. */
export const MESH_STRIDE=11;
export class SolidMesh {
  readonly data:number[]=[];
  triangle(a:V3,b:V3,c:V3,id:number,side=0):void {
    const n=unit(cross(sub(b,a),sub(c,a)));
    [a,b,c].forEach((p,i)=>this.data.push(...p,...n,...[0,1,2].map(j=>i===j?1:0),id,side));
  }
}
export const SCULPTURE_FIELD=`vec3 FLD(vec2 p){
 vec2 uv=p/max(BD(2).x*2.15,0.001)*0.5+0.5;
 if(any(lessThan(uv,vec2(0.)))||any(greaterThan(uv,vec2(1.))))return vec3(0.);
 return texture(SCULPTURE_TEX,uv).rgb*BD(0).x*uLayerK;
}`;
