#!/bin/bash
# test_sub.sh - A minimal test to see if Bash can hear the orchestrator

# 1. Source the workspace safely (avoiding the Colcon unbound variable crash)
set +u 
if [ -f ~/ws_aic/install/setup.bash ]; then
    source ~/ws_aic/install/setup.bash
fi
set -u

# 2. Force the Zenoh middleware to match the Docker container
export ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-0}
export RMW_IMPLEMENTATION=rmw_zenoh_cpp

echo "=== Environment ==="
echo "ROS_DOMAIN_ID: $ROS_DOMAIN_ID"
echo "RMW_IMPLEMENTATION: $RMW_IMPLEMENTATION"
echo ""
echo "Listening for /data_collection/scene_metadata..."
echo "(Press Ctrl+C to stop)"
echo "------------------------------------------------"

# 3. Echo the latched topic using the exact QoS profile the orchestrator uses.
# By omitting 'timeout' and '--once', this will stay open and print everything it sees.
ros2 topic echo \
    --qos-durability transient_local \
    --qos-reliability reliable \
    /data_collection/scene_metadata