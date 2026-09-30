import { atlasDistance } from './genes/lilyDistance';
import { createTexture, formats, type GL } from '../render/gl';

/** Shared painted petals plus a distance field derived from their alpha silhouettes.
 * The distance field keeps the artwork compatible with lines, glows and shape fusion.
 */
export class LilyAtlas {
  readonly color: WebGLTexture;
  readonly distance: WebGLTexture;
  readonly ready: Promise<void>;
  settled = false;
  loaded = false;

  constructor(gl: GL) {
    const format = formats(gl).rgba8;
    this.color = createTexture(gl, 1, 1, format, gl.LINEAR, new Uint8Array(4));
    this.distance = createTexture(gl, 1, 1, format, gl.LINEAR, new Uint8Array([255,255,255,255]));
    this.ready = this.load(gl).catch(error => {
      // Keep the procedural silhouette usable if an offline asset request fails.
      console.warn('Lily petal artwork unavailable; using procedural petals.', error);
    }).finally(() => { this.settled = true; });
  }

  private async load(gl: GL): Promise<void> {
    const image = new Image();
    image.src = new URL('../assets/lily-petals.png', import.meta.url).href;
    await image.decode();
    const canvas = document.createElement('canvas');
    canvas.width = image.width; canvas.height = image.height;
    const ctx = canvas.getContext('2d', { willReadFrequently: true })!;
    ctx.drawImage(image, 0, 0);
    const rgba = ctx.getImageData(0,0,image.width,image.height).data;
    const sdf = atlasDistance(rgba, image.width, image.height);
    gl.bindTexture(gl.TEXTURE_2D, this.color);
    gl.texImage2D(gl.TEXTURE_2D,0,gl.RGBA8,gl.RGBA,gl.UNSIGNED_BYTE,image);
    gl.bindTexture(gl.TEXTURE_2D, this.distance);
    gl.texImage2D(gl.TEXTURE_2D,0,gl.RGBA8,image.width,image.height,0,gl.RGBA,gl.UNSIGNED_BYTE,sdf);
    this.loaded = true;
  }
}

