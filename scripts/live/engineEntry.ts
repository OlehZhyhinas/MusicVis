// Bundle entry for the live-parity tools: the engine's Signals class (music state -> the preset
// signals) and the seeds, bundled by scripts/live/bundle.ts because engine.ts uses TypeScript
// syntax Node's type stripping cannot run.
export { Signals } from '../../src/v2/engine';
export { SEEDS } from '../../src/v2/seeds';
export { SIGNALS } from '../../src/v2/genome';
