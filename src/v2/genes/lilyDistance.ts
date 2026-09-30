/** Two-pass chamfer distances, isolated per atlas cell to prevent neighbour bleed. */
export function atlasDistance(rgba: ArrayLike<number>, width: number, height: number): Uint8Array {
  const out = new Uint8Array(width*height*4), w=width/3, h=height/2;
  for(let cell=0;cell<6;cell++) {
    const ox=cell%3*w, oy=Math.floor(cell/3)*h;
    const inside=new Float32Array(w*h), outside=new Float32Array(w*h);
    for(let y=0;y<h;y++) for(let x=0;x<w;x++) {
      const solid=rgba[((oy+y)*width+ox+x)*4+3]>=96;
      inside[y*w+x]=solid?0:1e4; outside[y*w+x]=solid?1e4:0;
    }
    for(const field of [inside,outside]) for(const direction of [1,-1]) {
      for(let yy=0;yy<h;yy++) for(let xx=0;xx<w;xx++) {
        const x=direction===1?xx:w-1-xx, y=direction===1?yy:h-1-yy, i=y*w+x;
        if(x-direction>=0 && x-direction<w) field[i]=Math.min(field[i],field[i-direction]+1);
        if(y-direction>=0 && y-direction<h) {
          field[i]=Math.min(field[i],field[i-direction*w]+1);
          for(const dx of [-1,1]) if(x+dx>=0 && x+dx<w) field[i]=Math.min(field[i],field[i-direction*w+dx]+Math.SQRT2);
        }
      }
    }
    for(let y=0;y<h;y++) for(let x=0;x<w;x++) {
      const i=y*w+x, dst=((oy+y)*width+ox+x)*4;
      // Half a cell of signed distance across the byte range.
      out[dst]=Math.round(Math.max(0,Math.min(255,128+(inside[i]-outside[i])*255/(h*0.5))));
      out[dst+3]=255;
    }
  }
  return out;
}
