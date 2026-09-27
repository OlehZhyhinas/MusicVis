#!/bin/bash
# one slot job: eval + export + TS parity for v3_large
cd /Users/oleh/personal/MusicVis/.claude/worktrees/agent-ab23606c20e23b07b
PY=.testdata/instr/venvs/torch/bin/python
D=scripts/ml/instruments/drums
T=alexandra-stan-mr-saxobeat
$PY $D/eval.py v3_large
$PY $D/export.py v3_large
$PY $D/dump_torch_ref.py v3_large test $T
node --import ./scripts/analysis-test.hooks.mjs $D/parity_check.ts v3_large test $T
$PY $D/peak_parity_ref.py v3_large test $T
node --import ./scripts/analysis-test.hooks.mjs $D/peak_parity.ts v3_large test $T
echo FINAL_DONE
