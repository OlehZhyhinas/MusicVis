#!/bin/bash
# Run a heavy job (training, separation, inference) under the machine-wide limit for the
# instrument agents: at most 2 heavy Python jobs at once, one MPS job at a time, and none
# started while free memory is under 25%. Waits for a free slot, then runs the command.
#
#   scripts/ml/instruments/slot.sh [--mps] <cmd...>
#
# Slots are lock directories under ~/personal/MusicVis-data/.slots/ holding the owner's pid;
# a stale lock (its pid gone) is reclaimed.
SLOTS="$HOME/personal/MusicVis-data/.slots"
mkdir -p "$SLOTS"
MPS=0
if [ "$1" = "--mps" ]; then MPS=1; shift; fi

free_pct() { memory_pressure 2>/dev/null | awk '/free percentage/ {gsub("%", "", $NF); print $NF}'; }
reclaim() { # $1 = lock dir
  if [ -d "$1" ]; then
    local p; p=$(cat "$1/pid" 2>/dev/null)
    if [ -z "$p" ] || ! kill -0 "$p" 2>/dev/null; then rm -rf "$1"; fi
  fi
}
take() { # $1 = lock dir; succeeds if we created it
  if mkdir "$1" 2>/dev/null; then echo $$ > "$1/pid"; echo "$*" > "$1/cmd"; return 0; fi
  return 1
}

got=""; gotm=""
while true; do
  for s in "$SLOTS/slot1" "$SLOTS/slot2" "$SLOTS/mps"; do reclaim "$s"; done
  f=$(free_pct); f=${f:-0}
  if [ "$f" -ge 25 ]; then
    if [ "$MPS" = 1 ] && ! take "$SLOTS/mps"; then sleep 20; continue; fi
    [ "$MPS" = 1 ] && gotm="$SLOTS/mps"
    for s in "$SLOTS/slot1" "$SLOTS/slot2"; do
      if take "$s"; then got="$s"; break; fi
    done
    [ -n "$got" ] && break
    [ -n "$gotm" ] && rm -rf "$gotm" && gotm=""
  fi
  sleep 20
done
trap 'rm -rf "$got" $gotm' EXIT INT TERM
echo "[slot] $(basename "$got")${gotm:+ +mps} free=${f}% : $*" >&2
"$@"
