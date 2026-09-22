// Wires up "click to choose" on the start screen's dropzone plus
// drag-and-drop anywhere on the page, at any time.

export interface DropzoneCallbacks {
  onFiles(files: File[]): void;
}

function isAudioFile(file: File): boolean {
  return file.type.startsWith('audio/') || /\.(mp3|wav|flac|m4a|aac|ogg|oga|weba)$/i.test(file.name);
}

export function installDropzone(
  dropzoneEl: HTMLElement,
  fileInput: HTMLInputElement,
  overlayEl: HTMLElement,
  callbacks: DropzoneCallbacks,
): void {
  const openPicker = () => fileInput.click();

  dropzoneEl.addEventListener('click', openPicker);
  dropzoneEl.addEventListener('keydown', (ev) => {
    if (ev.key === 'Enter' || ev.key === ' ') {
      ev.preventDefault();
      openPicker();
    }
  });

  fileInput.addEventListener('change', () => {
    const files = Array.from(fileInput.files ?? []);
    if (files.length > 0) callbacks.onFiles(files);
    fileInput.value = '';
  });

  let dragDepth = 0;

  window.addEventListener('dragenter', (ev) => {
    if (!ev.dataTransfer?.types.includes('Files')) return;
    ev.preventDefault();
    dragDepth++;
    overlayEl.classList.add('active');
  });

  window.addEventListener('dragover', (ev) => {
    if (!ev.dataTransfer?.types.includes('Files')) return;
    ev.preventDefault();
  });

  window.addEventListener('dragleave', () => {
    dragDepth = Math.max(0, dragDepth - 1);
    if (dragDepth === 0) overlayEl.classList.remove('active');
  });

  window.addEventListener('drop', (ev) => {
    ev.preventDefault();
    dragDepth = 0;
    overlayEl.classList.remove('active');
    const files = Array.from(ev.dataTransfer?.files ?? []).filter(isAudioFile);
    if (files.length > 0) {
      callbacks.onFiles(files);
    }
  });
}
