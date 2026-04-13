#!/bin/bash
# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  collect_orch.sh — AIC data collection orchestrator for zss_collection  ║
# ╚══════════════════════════════════════════════════════════════════════════╝
#
# Runs inside the aic-eval Docker container. The collection node runs in a
# SEPARATE terminal under pixi shell and must be started first:
#
#   # Terminal A (pixi shell):
#   ros2 run zss_collection collection_node --ros-args \
#       -p controller:=waypoint \
#       -p batch_id:=batch_001 \
#       -p dataset_root:=~/datasets \
#       -p use_sim_time:=true
#
#   # Terminal B (aic-eval docker):
#   ./collect_orch.sh configs/batch_001/manifest.json --batch-id batch_001
#
# Per-trial flow:
#   1. Show trial info; prompt ENTER / s (skip) / q (quit)
#   2. Launch Gazebo scene via entrypoint.sh
#   3. Wait for /aic_controller/controller_state
#   4. Publish /data_collection/scene_metadata (latched JSON)
#      → collection node starts the episode, publishes "start"
#   5. Manual mode: wait for ENTER / s / q from operator
#      Auto mode  : wait for "done" or "skip" on /data_collection/episode_event
#   6. Cleanup Gazebo, advance to next trial
#
# Keyboard controls (manual mode — default):
#   Pre-launch  : ENTER = launch, s = skip trial, q = quit
#   Post-launch : ENTER = trial done (save), s = skip/discard, q = quit
#
#   In auto mode the collection node's own keys handle skip/quit:
#   the node terminal accepts [s] skip and [q] quit via pynput.
#
# Manifest format (JSON array):
#   [
#     {
#       "trial_id":      "trial_001",
#       "scene_id":      "scene_sfp_01",
#       "launch_cmd":    "/entrypoint.sh ground_truth:=true ...",
#       "target_module": "nic_card_mount_2",
#       "port_name":     "sfp_port_0",
#       "plug_name":     "sfp_tip",
#       "plug_type":     "sfp",
#       "cable_name":    "cable_0",
#       "task":          "Insert SFP cable into NIC mount 2 port 0",
#       "board_pose":    {"x": 0.55, "y": 0.0, "z": 0.05, "yaw": 0.0}
#     },
#     ...
#   ]
#
# board_pose is optional — if omitted the collection node falls back to TF.
#
# Prerequisites:
#   jq  (sudo apt install jq)
#   RMW_IMPLEMENTATION=rmw_zenoh_cpp set in both terminals
#
# Usage:
#   ./collect_orch.sh <manifest.json> --batch-id <id> [OPTIONS]
#
# Options:
#   --batch-id ID        Batch identifier — must match collection node -p batch_id
#   --auto               Skip operator prompts; wait for episode_event signals
#   --start-from N       Resume from trial index N (default: 0)
#   --settle-time SECS   Wait after scene launch before checking controller (default: 20)
#   --episode-timeout S  Max seconds to wait for episode done in auto mode (default: 600)
#   --dry-run            Print commands without executing
#   -h, --help           Show this help

set -euo pipefail
trap '_orch_cleanup' EXIT SIGINT SIGTERM

# ══════════════════════════════════════════════════════════════════
# Defaults
# ══════════════════════════════════════════════════════════════════

MANIFEST_PATH=""
BATCH_ID=""
AUTO_MODE=false
START_TRIAL=0
SETTLE_TIME=15
EPISODE_TIMEOUT=600
DRY_RUN=false
SCENE_PID=""
META_PUB_PID=""
_WAIT_PID=""
_CLEANUP_DONE=false

# ══════════════════════════════════════════════════════════════════
# Parse arguments
# ══════════════════════════════════════════════════════════════════

usage() {
    sed -n '/^# Usage:/,/^[^#]/p' "$0" | grep '^#' | sed 's/^# \?//'
    exit 1
}

[[ $# -lt 1 ]] && usage
MANIFEST_PATH="$1"; shift

while [[ $# -gt 0 ]]; do
    case "$1" in
        --batch-id)        BATCH_ID="$2";          shift 2 ;;
        --auto)            AUTO_MODE=true;          shift   ;;
        --start-from)      START_TRIAL="$2";        shift 2 ;;
        --settle-time)     SETTLE_TIME="$2";        shift 2 ;;
        --episode-timeout) EPISODE_TIMEOUT="$2";    shift 2 ;;
        --dry-run)         DRY_RUN=true;            shift   ;;
        -h|--help)         usage ;;
        *) echo "Unknown option: $1"; usage ;;
    esac
done

# ══════════════════════════════════════════════════════════════════
# Validate
# ══════════════════════════════════════════════════════════════════

if [[ ! -f "$MANIFEST_PATH" ]]; then
    echo "ERROR: Manifest not found: $MANIFEST_PATH"; exit 1
fi
if ! command -v jq &>/dev/null; then
    echo "ERROR: jq required. Install with: sudo apt install jq"; exit 1
fi
if [[ -z "$BATCH_ID" ]]; then
    echo "ERROR: --batch-id is required (must match collection node -p batch_id)"; exit 1
fi

NUM_TRIALS=$(jq 'length' "$MANIFEST_PATH")

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  AIC Collection Orchestrator (zss_collection)                ║"
echo "╠══════════════════════════════════════════════════════════════╣"
printf "║  Manifest       : %-43s║\n" "$MANIFEST_PATH"
printf "║  Batch ID       : %-43s║\n" "$BATCH_ID"
printf "║  Total trials   : %-43s║\n" "$NUM_TRIALS"
printf "║  Starting from  : %-43s║\n" "$START_TRIAL"
printf "║  Mode           : %-43s║\n" "$([ "$AUTO_MODE" = true ] && echo 'auto' || echo 'manual (ENTER/s/q)')"
printf "║  Settle time    : %-43s║\n" "${SETTLE_TIME}s"
printf "║  Episode timeout: %-43s║\n" "${EPISODE_TIMEOUT}s"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "  IMPORTANT: The collection node must be running in a separate"
echo "  terminal (pixi shell) with -p batch_id:=${BATCH_ID}"
echo ""

# ══════════════════════════════════════════════════════════════════
# Helper functions
# ══════════════════════════════════════════════════════════════════

_orch_cleanup() {
    # Guard against re-entrancy (EXIT fires after SIGINT too)
    [[ "${_CLEANUP_DONE:-false}" == true ]] && return
    _CLEANUP_DONE=true

    echo ""
    echo "  [orch] Caught exit signal — cleaning up..."

    # Kill any background subprocesses waiting on topics (episode signal, etc.)
    if [[ -n "${_WAIT_PID:-}" ]]; then
        kill "$_WAIT_PID" 2>/dev/null || true
        _WAIT_PID=""
    fi

    _stop_metadata_pub
    _cleanup_scene

    # Signal collection node to save partial episode and finalize dataset
    echo "  [orch] Sending shutdown to collection node..."
    ros2 topic pub --once /data_collection/orch_control std_msgs/msg/String \
        "{data: 'shutdown'}" 2>/dev/null || true

    echo "  [orch] Cleanup complete."
}

_cleanup_scene() {
    if [[ -n "$SCENE_PID" ]] && kill -0 "$SCENE_PID" 2>/dev/null; then
        echo "  [orch] Killing scene process group (PID $SCENE_PID)..."
        kill -- -"$SCENE_PID" 2>/dev/null || true
        kill "$SCENE_PID" 2>/dev/null || true
    fi
    pkill -9 -f "entrypoint.sh"         2>/dev/null || true
    pkill -9 -f "gz sim"                2>/dev/null || true
    pkill -9 -f "ruby.*gz"              2>/dev/null || true
    pkill -9 -f "ros2.*launch"          2>/dev/null || true
    pkill -9 -f "parameter_bridge"      2>/dev/null || true
    pkill -9 -f "robot_state_publisher" 2>/dev/null || true
    pkill -9 -f "aic_controller"        2>/dev/null || true
    pkill -9 -f "aic_engine"            2>/dev/null || true
    pkill -9 -f "ros_gz_bridge"         2>/dev/null || true
    pkill -9 -f "spawner"               2>/dev/null || true
    pkill -9 -f "controller_manager"    2>/dev/null || true
    sleep 5
    if pgrep -f "gz sim" >/dev/null 2>&1; then
        echo "  [orch] WARNING: Gazebo still running, force-killing..."
        killall -9 gz 2>/dev/null || true
        sleep 3
    fi
    SCENE_PID=""
    echo "  [orch] Scene cleanup done."
}

_wait_for_controller() {
    local max_wait=120 elapsed=0
    echo "  [orch] Waiting for /aic_controller/controller_state..."
    while ! ros2 topic list 2>/dev/null | grep -q "/aic_controller/controller_state"; do
        sleep 2; elapsed=$((elapsed + 2))
        if [[ $elapsed -ge $max_wait ]]; then
            echo "  [orch] ERROR: Controller not ready after ${max_wait}s."
            return 1
        fi
    done
    echo "  [orch] Controller ready (${elapsed}s)."
}

_publish_metadata() {
    local json="$1"
    echo "  [orch] Publishing scene_metadata..."
    echo "         $json"
    # One-shot first so the latched topic is set immediately.
    ros2 topic pub --once \
        --qos-durability transient_local \
        /data_collection/scene_metadata \
        std_msgs/msg/String \
        "{data: '${json}'}" > /dev/null 2>&1
    # Keep alive at 0.2 Hz so late subscribers still catch it.
    ros2 topic pub -r 0.2 \
        --qos-durability transient_local \
        /data_collection/scene_metadata \
        std_msgs/msg/String \
        "{data: '${json}'}" > /dev/null 2>&1 &
    META_PUB_PID=$!
}

_stop_metadata_pub() {
    if [[ -n "${META_PUB_PID:-}" ]]; then
        kill "$META_PUB_PID" 2>/dev/null || true
        META_PUB_PID=""
    fi
}

# _wait_for_episode_signal — used in auto mode only.
# Returns 0 (done), 2 (skip), or 1 (timeout).
_wait_for_episode_signal() {
    local max_wait="$EPISODE_TIMEOUT"
    local tmp_out
    tmp_out=$(mktemp)
    echo "  [orch] Waiting for episode signal (max ${max_wait}s)..."

    # Run in background so _WAIT_PID can be killed on SIGINT.
    # Only "done" or "skip" terminate the wait — "resetting"/"collecting" are
    # ignored.  This prevents the stale latched "done" from a previous episode
    # from being matched immediately when the subscriber first connects.
    (
        timeout "$max_wait" bash -c '
            while IFS= read -r line; do
                if echo "$line" | grep -q '"'"'done'"'"'; then
                    echo "done"; exit 0
                elif echo "$line" | grep -q '"'"'skip'"'"'; then
                    echo "skip"; exit 0
                fi
                # "resetting" and "collecting" are intentionally ignored
            done < <(ros2 topic echo --qos-durability transient_local --qos-reliability reliable /data_collection/episode_event std_msgs/msg/String 2>/dev/null)
            exit 1
        ' 2>/dev/null
    ) > "$tmp_out" &
    _WAIT_PID=$!

    wait "$_WAIT_PID" 2>/dev/null
    local rc=$?
    _WAIT_PID=""

    local result
    result=$(cat "$tmp_out")
    rm -f "$tmp_out"

    if [[ $rc -ne 0 ]]; then
        echo "  [orch] ERROR: Timed out waiting for episode signal after ${max_wait}s."
        return 1
    fi

    if [[ "$result" == "skip" ]]; then
        echo "  [orch] Episode skipped by operator — advancing to next trial."
        return 2
    fi
    echo "  [orch] Episode done."
    return 0
}

_check_collection_node_alive() {
    if ! ros2 node list 2>/dev/null | grep -q "collection_node"; then
        echo "  [orch] WARNING: collection_node not found in ros2 node list."
        echo "         Make sure it is running in the pixi shell terminal."
    fi
}

# ══════════════════════════════════════════════════════════════════
# Main loop
# ══════════════════════════════════════════════════════════════════

COMPLETED=0
FAILED=0
SKIPPED=0

_check_collection_node_alive

for i in $(seq 0 $((NUM_TRIALS - 1))); do

    [[ $i -lt $START_TRIAL ]] && { SKIPPED=$((SKIPPED + 1)); continue; }

    # ── Extract trial fields ─────────────────────────────────────────
    TRIAL_ID=$(     jq -r ".[$i].trial_id"      "$MANIFEST_PATH")
    SCENE_ID=$(     jq -r ".[$i].scene_id"      "$MANIFEST_PATH")
    LAUNCH_CMD=$(   jq -r ".[$i].launch_cmd"    "$MANIFEST_PATH")
    TARGET_MODULE=$(jq -r ".[$i].target_module" "$MANIFEST_PATH")
    PORT_NAME=$(    jq -r ".[$i].port_name"     "$MANIFEST_PATH")
    PLUG_NAME=$(    jq -r ".[$i].plug_name"     "$MANIFEST_PATH")
    PLUG_TYPE=$(    jq -r ".[$i].plug_type"     "$MANIFEST_PATH")
    CABLE_NAME=$(   jq -r ".[$i].cable_name"    "$MANIFEST_PATH")
    TASK=$(jq -r '.[$i].task // .[$i].task_description // "Insert cable into port"' \
               --argjson i "$i" "$MANIFEST_PATH")
    BOARD_POSE=$(   jq -c ".[$i].board_pose // {}" "$MANIFEST_PATH")

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  Trial $((i + 1)) / $NUM_TRIALS  :  $TRIAL_ID"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  Scene    : $SCENE_ID"
    echo "  Target   : $TARGET_MODULE / $PORT_NAME  ($PLUG_TYPE)"
    echo "  Cable    : $CABLE_NAME  plug=$PLUG_NAME"
    echo "  Task     : $TASK"
    echo "  BoardPos : $BOARD_POSE"
    echo ""

    # ── Dry run ──────────────────────────────────────────────────────
    if [[ "$DRY_RUN" == true ]]; then
        echo "  [DRY RUN] Launch: $LAUNCH_CMD"
        COMPLETED=$((COMPLETED + 1))
        continue
    fi

    # ── Pre-launch prompt (manual mode) ──────────────────────────────
    if [[ "$AUTO_MODE" != true ]]; then
        echo "  Press ENTER to launch, 's' to skip, 'q' to quit:"
        read -r user_input
        case "$user_input" in
            s|S)
                echo "  [orch] Skipping trial $TRIAL_ID."
                SKIPPED=$((SKIPPED + 1))
                continue
                ;;
            q|Q)
                echo "  [orch] Quitting orchestrator."
                break
                ;;
        esac
    fi

    # ── Build scene_metadata JSON ─────────────────────────────────────
    METADATA_JSON=$(jq -nc \
        --arg  scene_id      "$SCENE_ID"      \
        --arg  trial_id      "$TRIAL_ID"      \
        --arg  batch_id      "$BATCH_ID"      \
        --arg  target_module "$TARGET_MODULE" \
        --arg  port_name     "$PORT_NAME"     \
        --arg  plug_name     "$PLUG_NAME"     \
        --arg  plug_type     "$PLUG_TYPE"     \
        --arg  cable_name    "$CABLE_NAME"    \
        --arg  task          "$TASK"          \
        --argjson board_pose "$BOARD_POSE"    \
        --argjson issued_at  "$(date +%s)"    \
        '{
            scene_id:      $scene_id,
            trial_id:      $trial_id,
            batch_id:      $batch_id,
            target_module: $target_module,
            port_name:     $port_name,
            plug_name:     $plug_name,
            plug_type:     $plug_type,
            cable_name:    $cable_name,
            task:          $task,
            board_pose:    $board_pose,
            issued_at:     $issued_at
        }')

    # ── Launch scene ─────────────────────────────────────────────────
    echo "  [orch] Launching scene..."
    setsid bash -c "$LAUNCH_CMD" &
    SCENE_PID=$!
    echo "  [orch] Scene PID: $SCENE_PID  — settling ${SETTLE_TIME}s..."
    sleep "$SETTLE_TIME"

    if ! _wait_for_controller; then
        echo "  [orch] FAILED: Controller did not come up. Skipping trial."
        FAILED=$((FAILED + 1))
        _cleanup_scene
        continue
    fi

    # ── Publish metadata → collection node starts episode ────────────
    _publish_metadata "$METADATA_JSON"

    # ── Wait for collection to finish ─────────────────────────────────
    if [[ "$AUTO_MODE" == true ]]; then
        # Fully automatic: block on episode_event topic signal.
        _wait_for_episode_signal
        ep_rc=$?
        if [[ $ep_rc -eq 1 ]]; then
            echo "  [orch] FAILED: Episode timed out for trial $TRIAL_ID."
            FAILED=$((FAILED + 1))
            _stop_metadata_pub
            _cleanup_scene
            continue
        elif [[ $ep_rc -eq 2 ]]; then
            SKIPPED=$((SKIPPED + 1))
            _stop_metadata_pub
            _cleanup_scene
            continue
        fi
    else
        # Manual: operator watches the collection node terminal and confirms.
        echo "  [orch] Collection node is running..."
        echo "  [orch] Press ENTER when episode is done, 's' to skip/discard, 'q' to quit:"
        read -r user_input
        case "$user_input" in
            s|S)
                echo "  [orch] Skipping/discarding episode $TRIAL_ID."
                SKIPPED=$((SKIPPED + 1))
                _stop_metadata_pub
                _cleanup_scene
                continue
                ;;
            q|Q)
                echo "  [orch] Quitting orchestrator."
                _stop_metadata_pub
                _cleanup_scene
                break
                ;;
        esac
    fi

    _stop_metadata_pub
    _cleanup_scene
    COMPLETED=$((COMPLETED + 1))

    if [[ $i -lt $((NUM_TRIALS - 1)) ]]; then
        echo "  [orch] Pausing 5s before next trial..."
        sleep 5
    fi

done

# ══════════════════════════════════════════════════════════════════
# Signal collection node to finalize and exit (auto mode only)
# ══════════════════════════════════════════════════════════════════
if [[ "$AUTO_MODE" == true ]]; then
    echo "[orch] All episodes done — sending shutdown to collection node..."
    ros2 topic pub --once /data_collection/orch_control std_msgs/msg/String \
        "{data: 'shutdown'}" 2>/dev/null || true
    # Wait up to 15s for node to disappear from the graph
    _deadline=$((SECONDS + 15))
    while ros2 node list 2>/dev/null | grep -q "collection_node" && [[ $SECONDS -lt $_deadline ]]; do
        sleep 1
    done
    if ros2 node list 2>/dev/null | grep -q "collection_node"; then
        echo "[orch] WARNING: collection node still alive after 15s — may need manual stop"
    else
        echo "[orch] Collection node exited cleanly."
    fi
fi

# ══════════════════════════════════════════════════════════════════
# Summary
# ══════════════════════════════════════════════════════════════════

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  Orchestration Complete                                      ║"
echo "╠══════════════════════════════════════════════════════════════╣"
printf "║  Completed : %-47s║\n" "$COMPLETED"
printf "║  Failed    : %-47s║\n" "$FAILED"
printf "║  Skipped   : %-47s║\n" "$SKIPPED"
echo "╚══════════════════════════════════════════════════════════════╝"
