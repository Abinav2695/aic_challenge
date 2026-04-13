#!/bin/bash
# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  node.sh — per-episode node_once spawner for zss_collection             ║
# ╚══════════════════════════════════════════════════════════════════════════╝

set -euo pipefail

# ── ENVIRONMENT ───────────────────────────────────────────────────────────────
export RMW_IMPLEMENTATION=rmw_zenoh_cpp
# ──────────────────────────────────────────────────────────────────────────────

if [[ $# -lt 1 ]]; then
    echo "Usage: node.sh <batch_id> [OPTIONS]"
    exit 1
fi

BATCH_ID="$1"; shift
DATASET_ROOT="$HOME/datasets"
USE_SIM_TIME="true"
RECORD_IMAGES="true"
PYTHON="python3"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset-root) DATASET_ROOT="$2"; shift 2 ;;
        --no-sim-time)  USE_SIM_TIME="false"; shift ;;
        --no-images)    RECORD_IMAGES="false"; shift ;;
        --python)       PYTHON="$2"; shift 2 ;;
        *) echo "[node.sh] Unknown option: $1"; exit 1 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
NODE_ONCE="$SRC_DIR/zss_collection/zss_collection/node_once.py"

_last_trial_id=""
_node_pid=""
_shutdown=false

_cleanup() {
    _shutdown=true
    if [[ -n "$_node_pid" ]] && kill -0 "$_node_pid" 2>/dev/null; then
        wait "$_node_pid" 2>/dev/null || true
    fi
    echo "[node.sh] Exiting."
}
trap '_cleanup' SIGINT SIGTERM EXIT

# ── Polling Loop ──────────────────────────────────────────────────────────────
_wait_for_new_metadata() {
    local last_id="$1"
    
    while true; do
        [[ "$_shutdown" == true ]] && return 1

        # Capture the raw string directly
        local raw
        raw=$(timeout 5 ros2 topic echo --once \
            --qos-durability transient_local \
            --qos-reliability reliable \
            /data_collection/scene_metadata \
            std_msgs/msg/String 2>/dev/null || echo "")

        if [[ -n "$raw" ]]; then
            # Extract trial_id using pure Regex (looks for "trial_id":"xyz")
            local trial_id
            trial_id=$(echo "$raw" | grep -oP '"trial_id":"\K[^"]+' || echo "")

            if [[ -n "$trial_id" && "$trial_id" != "$last_id" ]]; then
                echo "$trial_id"
                return 0
            fi
        fi

        sleep 1
    done
}

# ── Main ──────────────────────────────────────────────────────────────────────
echo "[node.sh] Starting — batch=$BATCH_ID"
echo "[node.sh] Waiting for scene_metadata from orchestrator..."

EPISODE_COUNT=0

while true; do
    [[ "$_shutdown" == true ]] && break

    new_trial_id=$(_wait_for_new_metadata "$_last_trial_id") || break
    [[ "$_shutdown" == true ]] && break

    EPISODE_COUNT=$((EPISODE_COUNT + 1))
    echo "[node.sh] ── Episode $EPISODE_COUNT  trial_id=$new_trial_id ──────────────────"
    
    cd "$SRC_DIR"
    "$PYTHON" "$NODE_ONCE" \
        --ros-args \
        -p batch_id:="$BATCH_ID" \
        -p dataset_root:="$DATASET_ROOT" \
        -p use_sim_time:="$USE_SIM_TIME" \
        -p record_images:="$RECORD_IMAGES" &
    _node_pid=$!

    wait "$_node_pid" 2>/dev/null || true
    _node_pid=""

    _last_trial_id="$new_trial_id"
    echo ""
done
