#!/usr/bin/env bash
set -euo pipefail

MASTER_HOST=${MASTER_HOST:-poweredge2}
WORKER_HOST=${WORKER_HOST:-poweredge3}

MASTER_PORT=${MASTER_PORT:-29501}
RUN_ID=${RUN_ID:-port${MASTER_PORT}}

RANK0_PID=${RANK0_PID:-/tmp/zeroq_multinode_${RUN_ID}_rank0.pid}
RANK1_PID=${RANK1_PID:-/tmp/zeroq_multinode_${RUN_ID}_rank1.pid}
RANK0_LOG=${RANK0_LOG:-/tmp/zeroq_multinode_${RUN_ID}_rank0.log}
RANK1_LOG=${RANK1_LOG:-/tmp/zeroq_multinode_${RUN_ID}_rank1.log}

SSH_OPTS=(
  -o BatchMode=yes
  -o StrictHostKeyChecking=accept-new
  -T
)

stop_one() {
  local host=$1
  local pidfile=$2
  echo "=== stopping on $host ==="
  ssh "${SSH_OPTS[@]}" "$host" bash -s -- "$pidfile" <<'REMOTE'
set -euo pipefail
pidfile="$1"

if [ -f "$pidfile" ]; then
  pid="$(cat "$pidfile" 2>/dev/null || true)"
  echo "pidfile=$pidfile pid=$pid"
  if [ -n "$pid" ]; then
    # Try to kill torchrun's process group and any lingering per-rank workers.
    kill -- -"$pid" 2>/dev/null || true
    kill -9 "$pid" 2>/dev/null || true
    pkill -9 -f "ZeroQ/tests/test_multi[node]\.py" 2>/dev/null || true
  fi
  rm -f "$pidfile"
else
  echo "no_pidfile=$pidfile"
fi
REMOTE
}

stop_one "$MASTER_HOST" "$RANK0_PID"
stop_one "$WORKER_HOST" "$RANK1_PID"

echo "(logs preserved: $RANK0_LOG, $RANK1_LOG)"
