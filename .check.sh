#!/bin/zsh
# local helper (not committed)
cd /Users/oleh/personal/MusicVis/.claude/worktrees/agent-a08026ea76fab5e16
node --import ./scripts/analysis-test.hooks.mjs scripts/v2-test.ts > .v2.log 2>&1
tail -1 .v2.log
grep -a "FAIL" .v2.log | head -20
node --import ./scripts/analysis-test.hooks.mjs scripts/analysis-test.ts 2>&1 | tail -1
node --import ./scripts/analysis-test.hooks.mjs scripts/realtime-test.ts 2>&1 | tail -1
npx tsc --noEmit && echo TSC_OK
