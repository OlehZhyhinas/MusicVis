// Screener calibration page (scripts/avq/screen.html): window.avqScreen.run(ids?) screens seeds
// with the app's Screener and returns its cheap reactivity metrics.
import { Engine } from '../../src/v2/engine';
import { screenSeeds } from './screenCal';

const eng = new Engine(document.getElementById('c') as HTMLCanvasElement);
(window as unknown as { avqScreen: unknown }).avqScreen = { run: (ids?: string[]) => screenSeeds(eng, ids) };
