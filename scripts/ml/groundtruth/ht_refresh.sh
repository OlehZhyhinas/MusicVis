#!/bin/bash
# Hooktheory clip location (hooktheory2.json) and refresh of the hooktheory part of the label files.
#   slot.sh --team gt scripts/ml/groundtruth/ht_refresh.sh
cd "$(dirname "$0")/../../.."
export DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib
PY=.testdata/gt/venv/bin/python
$PY scripts/ml/groundtruth/hooktheory.py && $PY scripts/ml/groundtruth/build_labels.py --ht-only
