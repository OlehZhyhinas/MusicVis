#!/bin/bash
# Runs the ground-truth stages in one process slot:
#   screen -> screen_norm -> hooktheory -> build_labels [args]
#   slot.sh --team gt scripts/ml/groundtruth/pipeline.sh [build_labels args]
cd "$(dirname "$0")/../../.."
export DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib
PY=.testdata/gt/venv/bin/python
$PY scripts/ml/groundtruth/screen.py && \
$PY scripts/ml/groundtruth/screen_norm.py && \
$PY scripts/ml/groundtruth/hooktheory.py && \
$PY scripts/ml/groundtruth/build_labels.py "$@"
