declare module 'butterchurn' {
  export interface ButterchurnOptions {
    width: number;
    height: number;
    pixelRatio?: number;
    textureRatio?: number;
    meshWidth?: number;
    meshHeight?: number;
  }

  export interface ButterchurnRenderOptions {
    elapsedTime?: number;
    audioLevels?: {
      timeByteArray: Uint8Array;
      timeByteArrayL: Uint8Array;
      timeByteArrayR: Uint8Array;
    };
  }

  export interface ButterchurnVisualizer {
    connectAudio(node: AudioNode): void;
    disconnectAudio(node: AudioNode): void;
    loadPreset(preset: object, blendTime?: number): void;
    setRendererSize(width: number, height: number, opts?: object): void;
    setInternalMeshSize(width: number, height: number): void;
    render(opts?: ButterchurnRenderOptions): void;
  }

  interface ButterchurnStatic {
    createVisualizer(context: AudioContext, canvas: HTMLCanvasElement, opts: ButterchurnOptions): ButterchurnVisualizer;
  }

  const butterchurn: ButterchurnStatic;
  export default butterchurn;
}

declare module 'butterchurn-presets' {
  interface ButterchurnPresetsStatic {
    getPresets(): Record<string, object>;
  }
  const presets: ButterchurnPresetsStatic;
  export default presets;
}
