#!/bin/sh
# Repo gates: typecheck (app + harness), build, and the existing test suites.
cd "$(dirname "$0")/../.." || exit 1
set -e
npx tsc --noEmit
npx tsc -p scripts/avq/tsconfig.json
npm run build >/dev/null
for t in v2-test chat-test analysis-test realtime-test; do
  node --import ./scripts/analysis-test.hooks.mjs scripts/$t.ts > .testdata/avq/logs/$t.log 2>&1 || { echo "FAIL $t"; tail -20 .testdata/avq/logs/$t.log; exit 1; }
  echo "ok $t: $(tail -1 .testdata/avq/logs/$t.log)"
done
if [ -f scripts/avq-test.ts ]; then node --import ./scripts/analysis-test.hooks.mjs scripts/avq-test.ts | tail -3; fi
