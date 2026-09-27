#!/bin/bash
# Final numbers for the stems change, one heavy job at a time through the team slot (prio 1 = stems).
cd "$(dirname "$0")/../../../.."
S="scripts/ml/instruments/slot.sh --prio 1"
P=.testdata/instr/presence
MUSICVIS_NEURAL_STEMS=0 $S node --import ./scripts/analysis-test.hooks.mjs scripts/live-parity.ts --out stems-before2 > $P/parity-before2.log 2>&1
rm -rf $P/norm/neural
$S node --import ./scripts/analysis-test.hooks.mjs scripts/ml/instruments/presence/norm_dump.ts neural > /dev/null 2>&1
$S .testdata/instr/venvs/torch/bin/python scripts/ml/instruments/presence/norm_study.py neural 20 > $P/norm-neural.txt 2>/dev/null
$S .testdata/instr/venvs/torch/bin/python scripts/ml/instruments/presence/presence_eval.py --student m128 --student m128+env > $P/presence-eval.txt 2>&1
$S node --import ./scripts/analysis-test.hooks.mjs scripts/analysis-test.ts > $P/analysis-test.log 2>&1
echo "exit $?" >> $P/analysis-test.log
echo chain done
