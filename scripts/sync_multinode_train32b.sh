#!/usr/bin/env bash
set -euo pipefail

# Sync the minimal set of files required for multinode 32B training
# to both GPU nodes.

MASTER_HOST=${MASTER_HOST:-poweredge2}
WORKER_HOST=${WORKER_HOST:-poweredge3}

REMOTE_DIR=${REMOTE_DIR:-/home/drawson/Phoenix}

SSH_OPTS=(
  -o BatchMode=yes
  -o StrictHostKeyChecking=accept-new
  -o ServerAliveInterval=10
  -o ServerAliveCountMax=3
)

# Use rsync if available; fall back to scp.
if command -v rsync >/dev/null 2>&1; then
  RSYNC_BASE=(rsync -az --delete -e "ssh ${SSH_OPTS[*]}")

  sync_one() {
    local host=$1
    echo "Syncing to ${host}:${REMOTE_DIR} (rsync)"

    "${RSYNC_BASE[@]}" \
      "ZeroQ/examples/train_32b_maxwell.py" \
      "${host}:${REMOTE_DIR}/ZeroQ/examples/train_32b_maxwell.py"

    "${RSYNC_BASE[@]}" \
      "ZeroQ/src/coordinator.py" \
      "${host}:${REMOTE_DIR}/ZeroQ/src/coordinator.py"

    "${RSYNC_BASE[@]}" \
      "ZeroQ/scripts/launch_multinode_train32b_detached.sh" \
      "${host}:${REMOTE_DIR}/ZeroQ/scripts/launch_multinode_train32b_detached.sh"

    "${RSYNC_BASE[@]}" \
      "ZeroQ/scripts/status_multinode_train32b_detached.sh" \
      "${host}:${REMOTE_DIR}/ZeroQ/scripts/status_multinode_train32b_detached.sh"

    "${RSYNC_BASE[@]}" \
      "ZeroQ/scripts/stop_multinode_train32b_detached.sh" \
      "${host}:${REMOTE_DIR}/ZeroQ/scripts/stop_multinode_train32b_detached.sh"
  }
else
  sync_one() {
    local host=$1
    echo "Syncing to ${host}:${REMOTE_DIR} (scp)"
    scp "${SSH_OPTS[@]}" \
      "ZeroQ/examples/train_32b_maxwell.py" \
      "ZeroQ/src/coordinator.py" \
      "ZeroQ/scripts/launch_multinode_train32b_detached.sh" \
      "ZeroQ/scripts/status_multinode_train32b_detached.sh" \
      "ZeroQ/scripts/stop_multinode_train32b_detached.sh" \
      "${host}:${REMOTE_DIR}/ZeroQ/" 2>/dev/null || true

    # Place files precisely if scp-to-dir layout differs
    scp "${SSH_OPTS[@]}" "ZeroQ/examples/train_32b_maxwell.py" "${host}:${REMOTE_DIR}/ZeroQ/examples/train_32b_maxwell.py"
    scp "${SSH_OPTS[@]}" "ZeroQ/src/coordinator.py" "${host}:${REMOTE_DIR}/ZeroQ/src/coordinator.py"
    scp "${SSH_OPTS[@]}" "ZeroQ/scripts/launch_multinode_train32b_detached.sh" "${host}:${REMOTE_DIR}/ZeroQ/scripts/launch_multinode_train32b_detached.sh"
    scp "${SSH_OPTS[@]}" "ZeroQ/scripts/status_multinode_train32b_detached.sh" "${host}:${REMOTE_DIR}/ZeroQ/scripts/status_multinode_train32b_detached.sh"
    scp "${SSH_OPTS[@]}" "ZeroQ/scripts/stop_multinode_train32b_detached.sh" "${host}:${REMOTE_DIR}/ZeroQ/scripts/stop_multinode_train32b_detached.sh"
  }
fi

sync_one "$MASTER_HOST"
sync_one "$WORKER_HOST"

echo "Sync complete."
