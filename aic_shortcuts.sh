#!/bin/bash
# AIC Competition Shortcuts
# Source this file in ~/.bashrc:
#   echo 'source ~/aic_shortcuts.sh' >> ~/.bashrc

# ── Container manager ─────────────────────────────────────────
export DBX_CONTAINER_MANAGER=docker

# ── Distrobox shortcuts ───────────────────────────────────────

# Enter eval container interactively
alias aic-eval='distrobox enter --root aic_eval'

# Start eval environment WITH ground truth (use during development)
alias aic-start='distrobox enter --root aic_eval -- /entrypoint.sh \
    ground_truth:=true \
    start_aic_engine:=true'

# Start eval environment WITHOUT ground truth (use for realistic testing)
alias aic-start-nogt='distrobox enter --root aic_eval -- /entrypoint.sh \
    ground_truth:=false \
    start_aic_engine:=true'

# Start environment for free exploration (no engine, task board spawned)
alias aic-explore='distrobox enter --root aic_eval -- /entrypoint.sh \
    ground_truth:=true \
    start_aic_engine:=false \
    spawn_task_board:=true \
    spawn_cable:=true \
    attach_cable_to_gripper:=true'

# ── Pixi shortcuts ────────────────────────────────────────────

# Enter pixi environment
alias aic-pixi='cd ~/ws_aic/src/aic && pixi shell'

# Run a policy by class path
# Usage: aic-policy <python.module.ClassName>
# Examples:
#   aic-policy aic_example_policies.ros.WaveArm
#   aic-policy aic_example_policies.ros.CheatCode
#   aic-policy my_policies.inspect_policy.InspectPolicy
aic-policy() {
    if [ -z "$1" ]; then
        echo "Usage: aic-policy <python.module.ClassName>"
        echo "Examples:"
        echo "  aic-policy aic_example_policies.ros.WaveArm"
        echo "  aic-policy aic_example_policies.ros.CheatCode"
        echo "  aic-policy my_policies.inspect_policy.InspectPolicy"
        return 1
    fi
    cd ~/ws_aic/src/aic && pixi run ros2 run aic_model aic_model \
        --ros-args -p use_sim_time:=true -p policy:="$1"
}

# ── ROS 2 debug shortcuts (run inside pixi shell) ─────────────
alias aic-topics='cd ~/ws_aic/src/aic && pixi run ros2 topic list'
alias aic-nodes='cd ~/ws_aic/src/aic && pixi run ros2 node list'
alias aic-wrench='cd ~/ws_aic/src/aic && pixi run ros2 topic echo /fts_broadcaster/wrench'
alias aic-joints='cd ~/ws_aic/src/aic && pixi run ros2 topic echo /joint_states'
alias aic-tcp='cd ~/ws_aic/src/aic && pixi run ros2 topic echo /aic_controller/controller_state'
alias aic-imgs='cd ~/ws_aic/src/aic && pixi run ros2 topic hz /center_camera/image'
alias pre-pr='cd ~/ws_aic/src/aic && pixi run pre-pr'
