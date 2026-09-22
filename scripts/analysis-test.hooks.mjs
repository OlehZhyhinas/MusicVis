// Lets Node run the TypeScript sources directly: resolves extensionless
// relative imports to their .ts files (Node strips the types natively).
// Usage: node --import ./scripts/analysis-test.hooks.mjs scripts/analysis-test.ts
import { registerHooks } from 'node:module';

registerHooks({
  resolve(specifier, context, next) {
    try {
      return next(specifier, context);
    } catch (err) {
      if (specifier.startsWith('.') && !specifier.endsWith('.ts')) return next(specifier + '.ts', context);
      throw err;
    }
  },
});
