#!/usr/bin/env bash
set -euo pipefail

MASTER_HOST=${MASTER_HOST:-poweredge2}
WORKER_HOST=${WORKER_HOST:-poweredge3}
MASTER_PORT=${MASTER_PORT:-29501}
RUN_ID=${RUN_ID:-port${MASTER_PORT}}

RANK0_LOG=${RANK0_LOG:-/tmp/zeroq_train32b_${RUN_ID}_rank0.log}
RANK1_LOG=${RANK1_LOG:-/tmp/zeroq_train32b_${RUN_ID}_rank1.log}
RANK0_PID=${RANK0_PID:-/tmp/zeroq_train32b_${RUN_ID}_rank0.pid}
RANK1_PID=${RANK1_PID:-/tmp/zeroq_train32b_${RUN_ID}_rank1.pid}

TAIL_LINES=${TAIL_LINES:-40}

SSH_OPTS=(
  -o BatchMode=yes
  -o StrictHostKeyChecking=accept-new
  -o ServerAliveInterval=10
  -o ServerAliveCountMax=3
  -T
)

remote_show() {
  local host=$1
  local who=$2
  local pidfile=$3
  local logfile=$4

  echo "== ${who} (${host}) =="
  ssh "${SSH_OPTS[@]}" "$host" bash -s -- "$pidfile" "$logfile" "$TAIL_LINES" <<'REMOTE'
set -euo pipefail
pidfile="$1"
logfile="$2"
tail_lines="$3"

if [[ -f "$pidfile" ]]; then
  pid=$(cat "$pidfile" || true)
  echo "pidfile: $pidfile (pid=$pid)"
  if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
    echo "status: RUNNING"
  else
    echo "status: NOT RUNNING"
  fi
else
  echo "pidfile: (missing) $pidfile"
fi

if [[ -f "$logfile" ]]; then
  echo "log: $logfile"
  tail -n "$tail_lines" "$logfile" || true
else
  echo "log: (missing) $logfile"
fi
REMOTE
  echo
}

remote_show "$MASTER_HOST" "rank0" "$RANK0_PID" "$RANK0_LOG"
remote_show "$WORKER_HOST" "rank1" "$RANK1_PID" "$RANK1_LOG"
