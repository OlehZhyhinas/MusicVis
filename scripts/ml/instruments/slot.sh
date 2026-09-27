#!/bin/bash
# Run a heavy job under the machine-wide limits for the instrument agents:
#   - ONE heavy job at a time for the whole instrument team (training, separation, inference,
#     evaluation AND node dumps/evals all count), so variants run sequentially, never in parallel;
#   - no launch while 3 or more heavy ML processes already run machine-wide (any agent: Python
#     using over 200 MB, or node analysis scripts), or while free memory is under 25%.
# Waits until it may start, then runs the command.
#
#   scripts/ml/instruments/slot.sh [--mps] <cmd...>     (--mps is accepted for clarity; with one
#                                                        team slot there is only ever one MPS job)
#
# The team slot is a lock directory ~/personal/MusicVis-data/.slots/team holding the owner's pid;
# a stale lock (its pid gone) is reclaimed.
SLOTS="$HOME/personal/MusicVis-data/.slots"
mkdir -p "$SLOTS"
[ "$1" = "--mps" ] && shift
LOCK="$SLOTS/team"
MAX_MACHINE=3

free_pct() { memory_pressure 2>/dev/null | awk '/free percentage/ {gsub("%", "", $NF); print $NF}'; }
machine_heavy() {
  ps -axo rss=,args= | awk '
    /analysis-test\.hooks\.mjs/ && !/avq\/render|vite|zsh -c/ { n++; next }
    /[Pp]ython/ && $1 > 204800 && !/http\.server/ { n++ }
    END { print n + 0 }'
}

while true; do
  if [ -d "$LOCK" ]; then
    p=$(cat "$LOCK/pid" 2>/dev/null)
    if [ -z "$p" ] || ! kill -0 "$p" 2>/dev/null; then rm -rf "$LOCK"; fi
  fi
  f=$(free_pct); f=${f:-0}
  m=$(machine_heavy)
  if [ "$f" -ge 25 ] && [ "$m" -lt "$MAX_MACHINE" ] && mkdir "$LOCK" 2>/dev/null; then
    echo $$ > "$LOCK/pid"; echo "$*" > "$LOCK/cmd"
    break
  fi
  sleep 20
done
trap 'rm -rf "$LOCK"' EXIT INT TERM
echo "[slot] team slot, free=${f}%, machine heavy=${m}: $*" >&2
"$@"
