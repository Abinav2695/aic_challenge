# MuJoCo Setup Guide

Team-specific notes for setting up and running the MuJoCo integration on this machine.
This is a supplement to the official `aic_utils/aic_mujoco/README.md` — read that first,
then use this doc for the gotchas we hit.

---

## Environment Overview

Two environments are in use. Knowing which one to use for each step is the most important
thing to get right.

| Environment | How to enter | Used for |
|---|---|---|
| Host (`Mechforge`) | normal terminal | pixi shell, viewer, training |
| Distrobox (`aic_eval`) | `aic-eval` or `distrobox enter aic_eval` | all ROS2 commands, colcon build, SDF conversion |

Distrobox shares the home directory, so files written in `aic_eval` at
`~/ws_aic/src/aic/...` are immediately visible on the host at the same path.

---

## Critical: Two Different Workspaces Inside aic_eval

Inside `aic_eval` there are two workspaces that look similar but are completely different:

```
/ws_aic/install/       the container's pre-built workspace (read-only)
                       provided by the aic_eval image
                       contains: aic_assets, aic_controller, aic_bringup,
                                 mujoco_vendor, mujoco_ros2_control, etc.

~/ws_aic/install/      YOUR local colcon build (in your home directory)
                       contains only what you built yourself:
                       sdformat_mjcf, aic_mujoco, mujoco_vendor, etc.
```

To see the difference yourself:

```bash
# Inside aic_eval container
ls /ws_aic/install/share/ | wc -l     # many packages from the pre-built image
ls ~/ws_aic/install/share/ | wc -l    # only your locally built packages

# Concrete example — aic_assets only exists in the container workspace
ls /ws_aic/install/share/aic_assets/models/ | head -3   # works
ls ~/ws_aic/install/share/aic_assets/ 2>/dev/null || echo "not here"  # empty
```

The rule for sourcing:

```bash
# Always source the container workspace FIRST to get aic_assets paths and base packages
source /ws_aic/install/setup.bash

# Then source your local build ON TOP to get sdf2mjcf, aic_mujoco, mujoco_vendor
source ~/ws_aic/install/setup.bash
```

Order matters — local build must come second so it overlays on top of the container
workspace. Sourcing only `~/ws_aic/install/setup.bash` will give you `sdf2mjcf` but
mesh asset paths like `/ws_aic/install/share/aic_assets/` won't be found, causing
conversion failures.

---

## Part 1: SDF Conversion and Viewing

### Step 1 — Import MuJoCo repos (host)

```bash
cd ~/ws_aic/src
vcs import < aic/aic_utils/aic_mujoco/mujoco.repos
```

### Step 2 — Install sdformat bindings (aic_eval)

Only needed once. Check first:

```bash
# Inside aic_eval
python3 -c "import sdformat; print('OK')" 2>/dev/null || echo "not installed"
```

If not installed:

```bash
sudo wget https://packages.osrfoundation.org/gazebo.gpg \
  -O /usr/share/keyrings/pkgs-osrf-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/pkgs-osrf-archive-keyring.gpg] \
  http://packages.osrfoundation.org/gazebo/ubuntu-stable $(lsb_release -cs) main" \
  | sudo tee /etc/apt/sources.list.d/gazebo-stable.list > /dev/null
sudo apt update
sudo apt install -y python3-sdformat16 python3-gz-math9 libsdformat16
```

### Step 3 — Build sdformat_mjcf (aic_eval)

```bash
# Source /ws_aic (container workspace) first so colcon finds dependencies
source /ws_aic/install/setup.bash
cd ~/ws_aic
colcon build --packages-select sdformat_mjcf
source ~/ws_aic/install/setup.bash

# Verify
which sdf2mjcf   # should resolve to ~/ws_aic/install/...
```

### Step 4 — Export from Gazebo and fix SDF (aic_eval)

Launch Gazebo and let it export `/tmp/aic.sdf`. Then check and fix URIs:

```bash
# Check what needs fixing — read the counts before applying
grep -c 'file://<urdf-string>/model://' /tmp/aic.sdf
grep -cE 'file:///(lc_plug|sc_plug|sfp_module)_visual\.glb' /tmp/aic.sdf

# Apply fixes only if the counts above are non-zero
sed -i 's|file://<urdf-string>/model://|model://|g' /tmp/aic.sdf
sed -i 's|file:///lc_plug_visual.glb|model://LC Plug/lc_plug_visual.glb|g' /tmp/aic.sdf
sed -i 's|file:///sc_plug_visual.glb|model://SC Plug/sc_plug_visual.glb|g' /tmp/aic.sdf
sed -i 's|file:///sfp_module_visual.glb|model://SFP Module/sfp_module_visual.glb|g' /tmp/aic.sdf
```

In our setup the glb URI fixes were not needed (already clean), but the `<urdf-string>`
fix was required. Check counts every time after a fresh Gazebo export.

### Step 5 — Convert SDF to MJCF (aic_eval)

`sdf2mjcf` must run inside `aic_eval` because mesh assets only exist at
`/ws_aic/install/share/aic_assets/` inside the container — not on the host. Running
`sdf2mjcf` on the host will fail with `Unable to find input mesh file`.

```bash
# Inside aic_eval — source container workspace first, then local build
source /ws_aic/install/setup.bash     # makes /ws_aic/install/share/aic_assets/ available
source ~/ws_aic/install/setup.bash    # provides the sdf2mjcf binary

mkdir -p ~/aic_mujoco_world
sdf2mjcf /tmp/aic.sdf ~/aic_mujoco_world/aic_world.xml

# Copy generated XML and mesh assets into the mjcf folder
cp ~/aic_mujoco_world/* ~/ws_aic/src/aic/aic_utils/aic_mujoco/mjcf/
```

### Step 6 — Run add_cable_plugin.py (aic_eval, no ROS sourced)

This script uses `import mujoco` from pip. If the ROS2 workspace is sourced, its Python
path can conflict with the pip-installed mujoco package causing import errors. Use a
fresh terminal with nothing sourced:

```bash
# Fresh aic_eval terminal — do NOT run any source commands before this
distrobox enter aic_eval

cd ~/ws_aic/src/aic/aic_utils/aic_mujoco/
python3 scripts/add_cable_plugin.py \
    --input mjcf/aic_world.xml \
    --output mjcf/aic_world.xml \
    --robot_output mjcf/aic_robot.xml \
    --scene_output mjcf/scene.xml
```

The warning `Could not find sc_port link for exclusion` is harmless for SFP scenes.

Then build aic_mujoco — source `/ws_aic` (container workspace) first so its dependencies
are found:

```bash
source /ws_aic/install/setup.bash    # container workspace — has aic_bringup etc.
cd ~/ws_aic
colcon build --packages-select aic_mujoco
```

### Step 7 — View scene (host, pixi)

```bash
# On host — no aic_eval needed
cd ~/ws_aic/src/aic
pixi run python3 aic_utils/aic_mujoco/scripts/view_scene.py \
    aic_utils/aic_mujoco/mjcf/scene.xml
```

The GLXBadContext errors on exit are harmless — known NVIDIA driver bug.

---

## Part 2: Full ROS2 Stack

### Step 1 — Import repos (host, if not done already)

```bash
cd ~/ws_aic/src
vcs import < aic/aic_utils/aic_mujoco/mujoco.repos
```

### Step 2 — Install dependencies (aic_eval)

```bash
cd ~/ws_aic
rosdep install --from-paths src --ignore-src --rosdistro kilted -yr \
  --skip-keys "gz-cmake3 DART libogre-dev libogre-next-2.3-dev"
```

### Step 3 — Build workspace (aic_eval)

Use a fresh terminal. If you get the error `install directory was created with layout
'isolated'`, the old build used a different layout — delete it and rebuild:

```bash
cd ~/ws_aic
rm -rf build/ install/ log/

# Source ONLY the system ROS2 for the full build — not /ws_aic or ~/ws_aic
source /opt/ros/kilted/setup.bash

GZ_BUILD_FROM_SOURCE=1 colcon build \
  --cmake-args -DCMAKE_BUILD_TYPE=Release \
  --merge-install --symlink-install \
  --packages-ignore lerobot_robot_aic sdformat_mjcf
```

`sdformat_mjcf` is excluded because it was already built in Part 1 and its colcon build
fails with a missing README.md error when included in the full workspace build.

After the build, source your local workspace (not /ws_aic) and verify:

```bash
source ~/ws_aic/install/setup.bash

echo $MUJOCO_DIR          # ~/ws_aic/install/opt/mujoco_vendor
echo $MUJOCO_PLUGIN_PATH  # ~/ws_aic/install/opt/mujoco_vendor/lib
which simulate             # ~/ws_aic/install/opt/mujoco_vendor/bin/simulate
```

### Step 4 — Launch (aic_eval, two terminals)

**Zenoh SHM fix:** the official docs say `enabled=true` but this fails inside distrobox
with `OS error 12 / Failed to create POSIX SHM provider` due to a hostname mismatch
between the container (`aic_eval`) and the host (`Mechforge`). Always use `enabled=false`
on this machine.

To set it permanently inside aic_eval so you never forget:

```bash
echo "export ZENOH_CONFIG_OVERRIDE='transport/shared_memory/enabled=false'" >> ~/.bashrc
```

Launch:

```bash
# Terminal 1 — aic_eval
# Source ~/ws_aic (your local build) — this has mujoco_vendor and the launch files
source ~/ws_aic/install/setup.bash
export RMW_IMPLEMENTATION=rmw_zenoh_cpp
export ZENOH_CONFIG_OVERRIDE='transport/shared_memory/enabled=false'
ros2 run rmw_zenoh_cpp rmw_zenohd

# Terminal 2 — aic_eval
source ~/ws_aic/install/setup.bash
export RMW_IMPLEMENTATION=rmw_zenoh_cpp
export ZENOH_CONFIG_OVERRIDE='transport/shared_memory/enabled=false'
ros2 launch aic_mujoco aic_mujoco_bringup.launch.py
```

Note: for the launch use `~/ws_aic/install/setup.bash` not `/ws_aic/install/setup.bash`.
The launch file needs `mujoco_vendor` binaries and plugins from your local build.

### Step 5 — Teleoperation

Teleop can run from pixi on the host (recommended) or from aic_eval directly.

**From host (pixi) — recommended:**

```bash
cd ~/ws_aic/src/aic
export RMW_IMPLEMENTATION=rmw_zenoh_cpp
export ZENOH_CONFIG_OVERRIDE='transport/shared_memory/enabled=false'
pixi run ros2 run aic_teleoperation cartesian_keyboard_teleop
```

**From aic_eval:**

```bash
source ~/ws_aic/install/setup.bash
export RMW_IMPLEMENTATION=rmw_zenoh_cpp
export ZENOH_CONFIG_OVERRIDE='transport/shared_memory/enabled=false'
ros2 run aic_teleoperation cartesian_keyboard_teleop
```

---

## Notes on mjcf/ Files

Only three XML files are tracked in git:

```
aic_utils/aic_mujoco/mjcf/aic_robot.xml   robot bodies, actuators, sensors
aic_utils/aic_mujoco/mjcf/aic_world.xml   environment, task board, cable
aic_utils/aic_mujoco/mjcf/scene.xml       top-level include referencing both
```

All mesh assets (`.obj`, `.png`, `.stl`) are gitignored. Do not commit them.

These XML files are regenerated every time you run the SDF conversion pipeline and will
change with every fresh Gazebo export because material/texture name hashes encode the
Gazebo session. Do not commit regenerated XML unless you have made intentional physics
or structural changes to the scene.

---

## Quick Reference

```
Need to...                            Run in...         Source before running
─────────────────────────────────────────────────────────────────────────────
vcs import mujoco repos               host              nothing
apt install sdformat16                aic_eval          nothing
colcon build sdformat_mjcf            aic_eval          /ws_aic only
sdf2mjcf conversion                   aic_eval          /ws_aic then ~/ws_aic
add_cable_plugin.py                   aic_eval          nothing (no ROS)
colcon build aic_mujoco               aic_eval          /ws_aic only
colcon build full workspace           aic_eval          /opt/ros/kilted only
view_scene.py                         host              nothing (pixi run)
rmw_zenohd                            aic_eval          ~/ws_aic
aic_mujoco_bringup.launch.py          aic_eval          ~/ws_aic
cartesian_keyboard_teleop             host (pixi run)   nothing (pixi run)
training / model code                 host              nothing (pixi run)
```
