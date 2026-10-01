#!/bin/bash
# Parallel full pipeline: each heavy stage runs as N shards (GT_SHARD=i/N), every shard in its own slot
# (slot.sh --team gt), then merge_shards.py folds the partial results back. Matching and freemidi stay
# single-process (freemidi is network-bound and must stay polite).
#   scripts/ml/groundtruth/pipeline_par.sh [N=3]      (not itself under slot.sh: it queues its shards)
cd "$(dirname "$0")/../../.."
N=${1:-3}
SLOT=/Users/oleh/personal/MusicVis/.claude/worktrees/agent-ab23606c20e23b07b/scripts/ml/instruments/slot.sh
export DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib
PY=.testdata/gt/venv/bin/python
G=scripts/ml/groundtruth
$SLOT --team gt $PY $G/lakh_match.py || exit 1
$PY $G/freemidi_fetch.py
stage() {
  for ((i = 0; i < N; i++)); do GT_SHARD=$i/$N $SLOT --team gt $PY $G/$1 > .testdata/gt/par.$1.$i.log 2>&1 & done
  wait
  $PY $G/merge_shards.py
}
stage hooktheory.py
stage screen.py
stage screen_norm.py
stage build_labels.py
