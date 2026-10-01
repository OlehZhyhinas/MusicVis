#!/bin/bash
# Run a heavy job under the machine-wide limits for the ML agents (instrument team rules below):
#   - at most TEAM_SLOTS (3) heavy jobs at a time per team (training, separation, inference,
#     evaluation AND node dumps/evals all count);
#   - no launch while MAX_MACHINE (8) or more heavy ML processes already run machine-wide (any agent:
#     Python using over 200 MB, or node analysis scripts), or while free memory is under 20%
#     (the owner allows using up to ~80% of memory).
# Waits until it may start, then runs the command.
#
#   scripts/ml/instruments/slot.sh [--team instruments|beats|gt] [--prio N] [--mps] <cmd...>
#     --team: each team (default instruments) gets TEAM_SLOTS slots; the machine-wide cap of MAX_MACHINE heavy
#             processes applies to all teams together. Priorities order waiters within a team.
#     --prio: 1 stems, 2 HQ separator, 3 notes (default), 4 drums, 5 chords. A waiting job never
#             takes the slot while a job with a smaller number is also waiting.
#     --mps:  accepted for clarity (MPS jobs share the GPU; prefer one at a time per team).
#
# The team slot is a lock directory ~/personal/MusicVis-data/.slots/team holding the owner's pid;
# a stale lock (its pid gone) is reclaimed.
SLOTS="$HOME/personal/MusicVis-data/.slots"
mkdir -p "$SLOTS"
PRIO=3
TEAM=instruments
while true; do
  case "$1" in
    --prio) PRIO=$2; shift 2 ;;
    --team) TEAM=$2; shift 2 ;;
    --mps) shift ;;
    *) break ;;
  esac
done
case "$TEAM" in instruments|beats|gt) ;; *) echo "slot.sh: unknown team '$TEAM' (instruments, beats, gt)" >&2; exit 2 ;; esac
# instruments keeps the original lock/wait paths so runners started before --team existed still exclude it
if [ "$TEAM" = instruments ]; then LOCK="$SLOTS/team"; WAITDIR="$SLOTS/wait"
else LOCK="$SLOTS/team-$TEAM"; WAITDIR="$SLOTS/wait-$TEAM"; fi
mkdir -p "$WAITDIR"
WAIT="$WAITDIR/$PRIO.$$"
touch "$WAIT"
MAX_MACHINE=8
TEAM_SLOTS=3
MIN_FREE=20

free_pct() { memory_pressure 2>/dev/null | awk '/free percentage/ {gsub("%", "", $NF); print $NF}'; }
machine_heavy() {
  ps -axo rss=,args= | awk '
    /analysis-test\.hooks\.mjs/ && !/avq\/render|vite|zsh -c|slot\.sh|chain_/ { n++; next }
    /[Pp]ython/ && $1 > 204800 && !/http\.server|slot\.sh/ { n++ }
    END { print n + 0 }'
}

outranked() { # another live waiter with a smaller priority number
  local w b q pid
  for w in "$WAITDIR"/*; do
    [ -e "$w" ] || continue
    b=$(basename "$w"); q=${b%%.*}; pid=${b#*.}
    if ! kill -0 "$pid" 2>/dev/null; then rm -f "$w"; continue; fi
    [ "$q" -lt "$PRIO" ] && return 0
  done
  return 1
}

reap() { local L=$1 p; if [ -d "$L" ]; then p=$(cat "$L/pid" 2>/dev/null); if [ -z "$p" ] || ! kill -0 "$p" 2>/dev/null; then rm -rf "$L"; fi; fi; }
BASE=$LOCK
while true; do
  f=$(free_pct); f=${f:-0}
  m=$(machine_heavy)
  got=
  if [ "$f" -ge "$MIN_FREE" ] && [ "$m" -lt "$MAX_MACHINE" ] && ! outranked; then
    for i in $(seq 1 "$TEAM_SLOTS"); do
      if [ "$i" = 1 ]; then L="$BASE"; else L="$BASE.$i"; fi
      reap "$L"
      if mkdir "$L" 2>/dev/null; then LOCK=$L; got=1; break; fi
    done
  fi
  if [ -n "$got" ]; then
    echo $$ > "$LOCK/pid"; echo "$*" > "$LOCK/cmd"
    rm -f "$WAIT"
    break
  fi
  sleep 20
done
trap 'rm -rf "$LOCK"; rm -f "$WAIT"' EXIT INT TERM
echo "[slot] $TEAM slot, free=${f}%, machine heavy=${m}: $*" >&2
"$@"
