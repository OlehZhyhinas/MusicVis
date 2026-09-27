#!/bin/bash
# Run a heavy job under the machine-wide limits for the instrument agents:
#   - ONE heavy job at a time for the whole instrument team (training, separation, inference,
#     evaluation AND node dumps/evals all count), so variants run sequentially, never in parallel;
#   - no launch while 3 or more heavy ML processes already run machine-wide (any agent: Python
#     using over 200 MB, or node analysis scripts), or while free memory is under 25%.
# Waits until it may start, then runs the command.
#
#   scripts/ml/instruments/slot.sh [--prio N] [--mps] <cmd...>
#     --prio: 1 stems, 2 HQ separator, 3 notes (default), 4 drums, 5 chords. A waiting job never
#             takes the slot while a job with a smaller number is also waiting.
#     --mps:  accepted for clarity; with one team slot there is only ever one MPS job.
#
# The team slot is a lock directory ~/personal/MusicVis-data/.slots/team holding the owner's pid;
# a stale lock (its pid gone) is reclaimed.
SLOTS="$HOME/personal/MusicVis-data/.slots"
mkdir -p "$SLOTS"
PRIO=3
if [ "$1" = "--prio" ]; then PRIO=$2; shift 2; fi
[ "$1" = "--mps" ] && shift
mkdir -p "$SLOTS/wait"
WAIT="$SLOTS/wait/$PRIO.$$"
touch "$WAIT"
LOCK="$SLOTS/team"
MAX_MACHINE=3

free_pct() { memory_pressure 2>/dev/null | awk '/free percentage/ {gsub("%", "", $NF); print $NF}'; }
machine_heavy() {
  ps -axo rss=,args= | awk '
    /analysis-test\.hooks\.mjs/ && !/avq\/render|vite|zsh -c/ { n++; next }
    /[Pp]ython/ && $1 > 204800 && !/http\.server/ { n++ }
    END { print n + 0 }'
}

outranked() { # another live waiter with a smaller priority number
  local w b q pid
  for w in "$SLOTS"/wait/*; do
    [ -e "$w" ] || continue
    b=$(basename "$w"); q=${b%%.*}; pid=${b#*.}
    if ! kill -0 "$pid" 2>/dev/null; then rm -f "$w"; continue; fi
    [ "$q" -lt "$PRIO" ] && return 0
  done
  return 1
}

while true; do
  if [ -d "$LOCK" ]; then
    p=$(cat "$LOCK/pid" 2>/dev/null)
    if [ -z "$p" ] || ! kill -0 "$p" 2>/dev/null; then rm -rf "$LOCK"; fi
  fi
  f=$(free_pct); f=${f:-0}
  m=$(machine_heavy)
  if [ "$f" -ge 25 ] && [ "$m" -lt "$MAX_MACHINE" ] && ! outranked && mkdir "$LOCK" 2>/dev/null; then
    echo $$ > "$LOCK/pid"; echo "$*" > "$LOCK/cmd"
    rm -f "$WAIT"
    break
  fi
  sleep 20
done
trap 'rm -rf "$LOCK"; rm -f "$WAIT"' EXIT INT TERM
echo "[slot] team slot, free=${f}%, machine heavy=${m}: $*" >&2
"$@"
