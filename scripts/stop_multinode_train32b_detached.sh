#!/usr/bin/env bash
set -euo pipefail

MASTER_HOST=${MASTER_HOST:-poweredge2}
WORKER_HOST=${WORKER_HOST:-poweredge3}
MASTER_PORT=${MASTER_PORT:-29501}
RUN_ID=${RUN_ID:-port${MASTER_PORT}}

RANK0_PID=${RANK0_PID:-/tmp/zeroq_train32b_${RUN_ID}_rank0.pid}
RANK1_PID=${RANK1_PID:-/tmp/zeroq_train32b_${RUN_ID}_rank1.pid}

SSH_OPTS=(
  -o BatchMode=yes
  -o StrictHostKeyChecking=accept-new
  -o ServerAliveInterval=10
  -o ServerAliveCountMax=3
  -T
)

remote_stop() {
  local host=$1
  local pidfile=$2

  ssh "${SSH_OPTS[@]}" "$host" bash -s -- "$pidfile" <<'REMOTE'
set -euo pipefail
pidfile="$1"
if [[ -f "$pidfile" ]]; then
  pid=$(cat "$pidfile" || true)
  if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
    echo "stopping pid=$pid on host=$(hostname)"
    kill "$pid" || true
    sleep 2
    if kill -0 "$pid" 2>/dev/null; then
      kill -9 "$pid" || true
    fi
  fi
  rm -f "$pidfile"
else
  echo "pidfile missing: $pidfile"
fi
REMOTE
}

echo "Stopping detached ZeRO-Q training:"
echo "  master: $MASTER_HOST pidfile=$RANK0_PID"
echo "  worker: $WORKER_HOST pidfile=$RANK1_PID"

echo "[1/2] Stop rank0 on $MASTER_HOST"
remote_stop "$MASTER_HOST" "$RANK0_PID"

echo "[2/2] Stop rank1 on $WORKER_HOST"
remote_stop "$WORKER_HOST" "$RANK1_PID"

echo "Done."
