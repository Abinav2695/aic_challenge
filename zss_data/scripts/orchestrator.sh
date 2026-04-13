#!/bin/bash
trap 'pkill -P $$' EXIT SIGINT SIGTERM
############################################
# Orchestrator: Run data collection across multiple scenes
#
# Reads a launch_manifest.json, spawns each scene one at a time,
# waits for user confirmation, runs the data collector, and moves
# to the next trial.
#
# Usage:
#   ./orchestrator.sh configs/batch_sfp_25/launch_manifest.json \
#
#   # Auto-mode (no user prompt between trials):
#   ./orchestrator.sh configs/batch_sfp_25/launch_manifest.json \
#       --auto
#
# Prerequisites:
#   - jq (sudo apt install jq)
#   - The AIC eval container running (for /entrypoint.sh)
#
# What it does for each trial:
#   1. Kills any previous Gazebo / ROS processes
#   2. Launches the scene via /entrypoint.sh (background)
#   3. Waits for the robot controller to come up
#   4. Waits for user input (unless --auto)
#   5. Repeats for next trial
############################################

set -euo pipefail

# ══════════════════════════════════════════════════════════════
# Defaults
# ══════════════════════════════════════════════════════════════

MANIFEST_PATH=""
AUTO_MODE=false
NO_IMAGES=true          # default: skip images (cv2 issues)
SETTLE_TIME=20          # seconds to wait after scene launch
START_TRIAL=0           # resume from this trial index
DRY_RUN=false
DATASET_ROOT="${HOME}/datasets"   # where episodes are stored on disk
NO_EXPLORE=true                   # skip the 12s EXPLORE sweep by default in auto mode
PIXI_DIR="$(cd "$(dirname "$0")/../.." && pwd)"  # src/aic root (contains pixi.toml)

# ══════════════════════════════════════════════════════════════
# Parse arguments
# ══════════════════════════════════════════════════════════════

usage() {
    echo "Usage: $0 <manifest.json> [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --auto                Skip user prompts and run collector automatically"
    echo "  --with-images         Record camera images (default: off)"
    echo "  --with-explore        Run 12s EXPLORE sweep before each insertion (default: off)"
    echo "  --dataset-root DIR    Root dir for episode storage (default: ~/datasets)"
    echo "  --settle-time SECS    Wait time after scene launch (default: 20)"
    echo "  --start-from N        Resume from trial index N (default: 0)"
    echo "  --dry-run             Print commands without executing"
    echo "  -h, --help            Show this help"
    exit 1
}

if [[ $# -lt 1 ]]; then
    usage
fi

MANIFEST_PATH="$1"
shift

while [[ $# -gt 0 ]]; do
    case "$1" in
        --auto)           AUTO_MODE=true; shift ;;
        --with-images)    NO_IMAGES=false; shift ;;
        --with-explore)   NO_EXPLORE=false; shift ;;
        --dataset-root)   DATASET_ROOT="$2"; shift 2 ;;
        --settle-time)    SETTLE_TIME="$2"; shift 2 ;;
        --start-from)     START_TRIAL="$2"; shift 2 ;;
        --dry-run)        DRY_RUN=true; shift ;;
        -h|--help)        usage ;;
        *)                echo "Unknown option: $1"; usage ;;
    esac
done

# ══════════════════════════════════════════════════════════════
# Validate
# ══════════════════════════════════════════════════════════════

if [[ ! -f "$MANIFEST_PATH" ]]; then
    echo "ERROR: Manifest not found: $MANIFEST_PATH"
    exit 1
fi

if ! command -v jq &>/dev/null; then
    echo "ERROR: jq is required. Install with: sudo apt install jq"
    exit 1
fi

NUM_TRIALS=$(jq 'length' "$MANIFEST_PATH")
echo "═══════════════════════════════════════════════════════════"
echo "  AIC Data Collection Orchestrator"
echo "═══════════════════════════════════════════════════════════"
echo "  Manifest     : $MANIFEST_PATH"
echo "  Total trials : $NUM_TRIALS"
echo "  Starting from: $START_TRIAL"
echo "  Auto mode    : $AUTO_MODE"
echo "  Record images: $([ "$NO_IMAGES" = true ] && echo "no" || echo "yes")"
echo "  Explore sweep: $([ "$NO_EXPLORE" = true ] && echo "no" || echo "yes")"
echo "  Dataset root : $DATASET_ROOT"
echo "  Settle time  : ${SETTLE_TIME}s"
echo "═══════════════════════════════════════════════════════════"
echo ""

if [[ "$DRY_RUN" == true ]]; then
    echo "[DRY RUN] Will print commands without executing."
    echo ""
fi

# ══════════════════════════════════════════════════════════════
# Helper functions
# ══════════════════════════════════════════════════════════════

wait_for_controller() {
    # Wait for the aic_controller to advertise its state topic
    echo "  Waiting for /aic_controller/controller_state..."
    local max_wait=120
    local elapsed=0
    while ! ros2 topic list 2>/dev/null | grep -q "/aic_controller/controller_state"; do
        sleep 2
        elapsed=$((elapsed + 2))
        if [[ $elapsed -ge $max_wait ]]; then
            echo "  ERROR: Controller not ready after ${max_wait}s. Skipping trial."
            return 1
        fi
    done
    echo "AIC  Controller is up (${elapsed}s)."
    return 0
}

wait_for_tfs() {
    # Wait for task board TFs to appear
    echo "  Waiting for task board TFs..."
    local max_wait=60
    local elapsed=0
    while ! ros2 topic echo /tf_static --once 2>/dev/null | grep -q "task_board"; do
        sleep 2
        elapsed=$((elapsed + 2))
        if [[ $elapsed -ge $max_wait ]]; then
            echo "  WARNING: TFs not confirmed after ${max_wait}s. Proceeding anyway."
            return 0
        fi
    done
    echo "  TFs available (${elapsed}s)."
    return 0
}

cleanup_scene() {
    echo "  Cleaning up previous scene..."
    
    # Kill the scene PID tree if it exists
    if [[ -n "${SCENE_PID:-}" ]] && kill -0 "$SCENE_PID" 2>/dev/null; then
        echo "  Killing scene process tree (PID $SCENE_PID)..."
        kill -- -"$SCENE_PID" 2>/dev/null || true   # kill process group
        kill "$SCENE_PID" 2>/dev/null || true
    fi

    # Kill all known AIC/Gazebo processes
    pkill -9 -f "entrypoint.sh" 2>/dev/null || true
    pkill -9 -f "gz sim" 2>/dev/null || true
    pkill -9 -f "ruby.*gz" 2>/dev/null || true
    pkill -9 -f "ros2.*launch" 2>/dev/null || true
    pkill -9 -f "parameter_bridge" 2>/dev/null || true
    pkill -9 -f "robot_state_publisher" 2>/dev/null || true
    pkill -9 -f "aic_controller" 2>/dev/null || true
    pkill -9 -f "aic_engine" 2>/dev/null || true
    pkill -9 -f "ros_gz_bridge" 2>/dev/null || true
    pkill -9 -f "spawner" 2>/dev/null || true
    pkill -9 -f "controller_manager" 2>/dev/null || true
    
    # Wait for ports/sockets to release
    sleep 5
    
    # Verify nothing is left
    if pgrep -f "gz sim" >/dev/null 2>&1; then
        echo "  WARNING: Gazebo still running, force killing..."
        killall -9 gz 2>/dev/null || true
        sleep 3
    fi
    
    echo "  Cleanup done."
}

# ══════════════════════════════════════════════════════════════
# Main loop
# ══════════════════════════════════════════════════════════════

COMPLETED=0
FAILED=0
SKIPPED=0

for i in $(seq 0 $((NUM_TRIALS - 1))); do

    # Skip trials before start index
    if [[ $i -lt $START_TRIAL ]]; then
        continue
    fi

    # ── Extract trial data from manifest ──────────────────────
    TRIAL_ID=$(jq -r ".[$i].trial_id" "$MANIFEST_PATH")
    LAUNCH_CMD=$(jq -r ".[$i].launch_cmd" "$MANIFEST_PATH")
    TARGET_MODULE=$(jq -r ".[$i].target_module" "$MANIFEST_PATH")
    PORT_NAME=$(jq -r ".[$i].port_name" "$MANIFEST_PATH")
    PLUG_NAME=$(jq -r ".[$i].plug_name" "$MANIFEST_PATH")
    CABLE_NAME=$(jq -r ".[$i].cable_name" "$MANIFEST_PATH")
    SCENE_ID=$(jq -r ".[$i].scene_id" "$MANIFEST_PATH")
    BATCH_ID=$(jq -r ".[$i].batch_id" "$MANIFEST_PATH")
    TASK_DESC=$(jq -r ".[$i].task_description" "$MANIFEST_PATH")
    PLUG_TYPE=$(jq -r ".[$i].plug_type" "$MANIFEST_PATH")

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  Trial $((i + 1))/$NUM_TRIALS: $TRIAL_ID"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  Plug type : $PLUG_TYPE"
    echo "  Target    : $TARGET_MODULE / $PORT_NAME"
    echo "  Cable     : $CABLE_NAME ($PLUG_NAME)"
    echo "  Task      : $TASK_DESC"
    echo "  Scene ID  : $SCENE_ID"
    echo "  Batch ID  : $BATCH_ID"
    echo ""

    # ── Dry run: just print ────────────────────────────────────
    if [[ "$DRY_RUN" == true ]]; then
        echo "  [DRY RUN] Launch: $LAUNCH_CMD"
        echo "  [DRY RUN] Collector args:"
        echo "    --target_module $TARGET_MODULE"
        echo "    --port_name $PORT_NAME"
        echo "    --plug_name $PLUG_NAME"
        echo "    --cable_name $CABLE_NAME"
        echo "    --scene_id $SCENE_ID"
        echo "    --batch_id $BATCH_ID"
        echo ""
        COMPLETED=$((COMPLETED + 1))
        continue
    fi

    # ── User prompt (unless auto mode) ─────────────────────────
    if [[ "$AUTO_MODE" != true ]]; then
        echo "  Press ENTER to launch this trial, 's' to skip, 'q' to quit:"
        read -r user_input
        case "$user_input" in
            s|S)
                echo "  Skipping trial $TRIAL_ID"
                SKIPPED=$((SKIPPED + 1))
                continue
                ;;
            q|Q)
                echo "  Quitting orchestrator."
                break
                ;;
        esac
    fi

    # ── Launch scene ───────────────────────────────────────────
    echo "  Launching scene..."
    echo "  CMD: $LAUNCH_CMD"
    setsid bash -c "$LAUNCH_CMD" &
    SCENE_PID=$!
    echo "  Scene PID: $SCENE_PID"

    # ── Wait for everything to come up ─────────────────────────
    echo "  Settling for ${SETTLE_TIME}s..."
    sleep "$SETTLE_TIME"

    if ! wait_for_controller; then
        echo "  FAILED: Controller didn't come up. Skipping."
        FAILED=$((FAILED + 1))
        cleanup_scene
        continue
    fi

    # Publish scene metadata as JSON string
    METADATA_JSON=$(jq -n \
        --arg scene_id "$SCENE_ID" \
        --arg batch_id "$BATCH_ID" \
        --arg trial_id "$TRIAL_ID" \
        --arg target_module "$TARGET_MODULE" \
        --arg port_name "$PORT_NAME" \
        --arg plug_name "$PLUG_NAME" \
        --arg plug_type "$PLUG_TYPE" \
        --arg cable_name "$CABLE_NAME" \
        --arg task "$TASK_DESC" \
        '{
            scene_id: $scene_id,
            batch_id: $batch_id,
            trial_id: $trial_id,
            target_module: $target_module,
            port_name: $port_name,
            plug_name: $plug_name,
            plug_type: $plug_type,
            cable_name: $cable_name,
            task: $task
        }' | jq -c .)

    # 1. Start publishing at a slow rate in the background (silenced)
    ros2 topic pub -r 0.1 --qos-durability transient_local \
        /data_collection/scene_metadata std_msgs/msg/String \
        "{data: '$METADATA_JSON'}" > /dev/null 2>&1 &

    # 2. Capture the Process ID of the publisher so we can kill it later
    PUB_PID=$!

    echo "  Metadata published continuously in background (PID $PUB_PID)."

    # ── Run / wait for data collection ────────────────────────
    if [[ "$AUTO_MODE" != true ]]; then
        echo "  Press ENTER when data collection is complete, or 'q' to quit:"
        read -r user_input
        case "$user_input" in
            q|Q)
                echo "  Quitting orchestrator."
                cleanup_scene
                kill $PUB_PID 2>/dev/null || true
                break
                ;;
        esac
    else
        # Build collector command
        COLLECTOR_CMD="python -m zss_data.data_collection"
        COLLECTOR_CMD+=" --repo_id local/aic_insertion_dataset"
        COLLECTOR_CMD+=" --dataset_root ${DATASET_ROOT}"
        COLLECTOR_CMD+=" --batch_id ${BATCH_ID}"
        COLLECTOR_CMD+=" --scene_id ${SCENE_ID}"
        COLLECTOR_CMD+=" --target_module ${TARGET_MODULE}"
        COLLECTOR_CMD+=" --port_name ${PORT_NAME}"
        COLLECTOR_CMD+=" --plug_name ${PLUG_NAME}"
        COLLECTOR_CMD+=" --cable_name ${CABLE_NAME}"
        COLLECTOR_CMD+=" --task '${TASK_DESC}'"
        COLLECTOR_CMD+=" --control_mode auto"
        [[ "$NO_IMAGES" == true ]]   && COLLECTOR_CMD+=" --no_images"
        [[ "$NO_EXPLORE" == true ]]  && COLLECTOR_CMD+=" --no_explore"

        echo "  Running collector..."
        echo "  CMD: $COLLECTOR_CMD"
        (cd "$PIXI_DIR" && pixi run $COLLECTOR_CMD)
        COLLECTOR_EXIT=$?

        if [[ $COLLECTOR_EXIT -eq 0 ]]; then
            echo "  Collector finished successfully (exit $COLLECTOR_EXIT)."
        else
            echo "  WARNING: Collector exited with code $COLLECTOR_EXIT."
            FAILED=$((FAILED + 1))
            cleanup_scene
            continue
        fi
    fi

    # Clean up the publisher when the trial is done
    kill $PUB_PID 2>/dev/null || true

    # ── Clean up previous scene ────────────────────────────────
    cleanup_scene

    # ── Mark trial as completed ────────────────────────────────
    COMPLETED=$((COMPLETED + 1))

    # ── Brief pause between trials ─────────────────────────────
    if [[ $i -lt $((NUM_TRIALS - 1)) ]]; then
        echo "  Pausing 5s before next trial..."
        sleep 5
    fi

done

# ══════════════════════════════════════════════════════════════
# Summary
# ══════════════════════════════════════════════════════════════

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  Orchestration Complete"
echo "═══════════════════════════════════════════════════════════"
echo "  Completed : $COMPLETED"
echo "  Failed    : $FAILED"
echo "  Skipped   : $SKIPPED"
echo "═══════════════════════════════════════════════════════════"
