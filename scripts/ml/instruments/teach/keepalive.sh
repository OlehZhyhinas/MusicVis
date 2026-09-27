#!/bin/bash
# Supervisor: keeps a teacher runner alive forever by re-launching it every
# time its single pass over the corpora finishes, so it re-lists tracks and
# picks up newly downloaded own/FMA files. Usage: keepalive.sh <cmd...>
while true; do
  "$@"
  sleep 45
done
