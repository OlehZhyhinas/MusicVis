#!/bin/bash
# New playlists: match (Lakh) -> freemidi (polite, network only) -> full pipeline. Already-done tracks are skipped
# at every stage, so it is safe to rerun while a playlist is still downloading.
#   slot.sh --team gt scripts/ml/groundtruth/newtracks.sh
cd "$(dirname "$0")/../../.."
export DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib
PY=.testdata/gt/venv/bin/python
$PY scripts/ml/groundtruth/lakh_match.py && \
$PY scripts/ml/groundtruth/freemidi_fetch.py; \
scripts/ml/groundtruth/pipeline.sh
