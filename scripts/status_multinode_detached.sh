#!/usr/bin/env bash
set -euo pipefail

MASTER_HOST=${MASTER_HOST:-poweredge2}
WORKER_HOST=${WORKER_HOST:-poweredge3}

MASTER_PORT=${MASTER_PORT:-29501}
RUN_ID=${RUN_ID:-port${MASTER_PORT}}

RANK0_LOG=${RANK0_LOG:-/tmp/zeroq_multinode_${RUN_ID}_rank0.log}
RANK1_LOG=${RANK1_LOG:-/tmp/zeroq_multinode_${RUN_ID}_rank1.log}
RANK0_PID=${RANK0_PID:-/tmp/zeroq_multinode_${RUN_ID}_rank0.pid}
RANK1_PID=${RANK1_PID:-/tmp/zeroq_multinode_${RUN_ID}_rank1.pid}

SSH_OPTS=(
  -o BatchMode=yes
  -o StrictHostKeyChecking=accept-new
  -T
)

status_one() {
  local host=$1
  local pidfile=$2
  local logfile=$3
  echo "=== $host ==="
  ssh "${SSH_OPTS[@]}" "$host" bash -s -- "$pidfile" "$logfile" <<'REMOTE'
set -euo pipefail
pidfile="$1"
logfile="$2"

if [ -f "$pidfile" ]; then
  pid="$(cat "$pidfile" 2>/dev/null || true)"
  echo "pidfile=$pidfile pid=$pid"
  if [ -n "$pid" ]; then
    ps -p "$pid" -o pid,etimes,cmd 2>/dev/null || echo not_running
  fi
else
  echo "no_pidfile=$pidfile"
fi

echo "---tail $logfile---"
tail -n 60 "$logfile" 2>/dev/null || echo no_log
REMOTE
}

status_one "$MASTER_HOST" "$RANK0_PID" "$RANK0_LOG"
status_one "$WORKER_HOST" "$RANK1_PID" "$RANK1_LOG"
