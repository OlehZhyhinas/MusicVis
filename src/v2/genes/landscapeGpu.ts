// GPU side of the landscape shape: the song's world map as an RGBA32F texture (n x 2), and which map
// is current. A song (analysed offline) gets its whole map once; live input, or a song-less stage
// (the fingerprint clip), gets the moving LiveWorld window built from recent history.

import { formats, type GL } from '../../render/gl';
import type { AnalysisResult, MusicState } from '../../types';
import { LiveWorld, buildSongWorld, type SongWorld } from '../../analysis/songWorld';

export class LandWorld {
  private tex: WebGLTexture | null = null;
  private texW = 0;
  private uploaded: SongWorld | null = null;
  private upVersion = -1;
  private song: SongWorld | null = null;
  private live = new LiveWorld();
  /** The map in use this frame. */
  world: SongWorld = this.live.world;
  /** Song time this frame (playback position, or the live clock). */
  now = 0;

  constructor(private gl: GL) {}

  /** The analysed song (null: none, e.g. live input). */
  setSong(r: AnalysisResult | null): void {
    this.song = r ? buildSongWorld(r) : null;
  }

  /**
   * Picks the map for this frame: the song's when it is playing from the offline analysis (look-ahead
   * fields present), the live window otherwise (fed every frame so its history stays current).
   */
  update(state: MusicState, dt: number): void {
    const sampled = this.song && state.timeToDrop !== undefined;
    this.now = Number.isFinite(state.time) ? state.time : 0;
    if (sampled) this.world = this.song!;
    else {
      this.live.update(state, dt);
      this.world = this.live.world;
    }
  }

  /** The texture of the current map (uploaded when it changed). */
  texture(): WebGLTexture {
    const gl = this.gl;
    const w = this.world;
    if (this.tex && this.uploaded === w && this.upVersion === w.version) return this.tex;
    const f = formats(gl).rgba32f;
    if (!this.tex || this.texW !== w.n) {
      if (this.tex) gl.deleteTexture(this.tex);
      this.tex = gl.createTexture()!;
      gl.bindTexture(gl.TEXTURE_2D, this.tex);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
      gl.texImage2D(gl.TEXTURE_2D, 0, f.internal, w.n, 2, 0, f.format, f.type, w.data);
      this.texW = w.n;
    } else {
      gl.bindTexture(gl.TEXTURE_2D, this.tex);
      gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, w.n, 2, f.format, f.type, w.data);
    }
    this.uploaded = w;
    this.upVersion = w.version;
    return this.tex;
  }

  dispose(): void {
    if (this.tex) this.gl.deleteTexture(this.tex);
    this.tex = null;
  }
}
