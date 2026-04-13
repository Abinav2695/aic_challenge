#!/usr/bin/env python3
"""
Hybrid Data Collection Pipeline with LeRobot Recording Integration.

This script provides a modular insertion pipeline that supports:
  - Auto mode: phased waypoint insertion (APPROACH → CORRECT → DESCEND → HOLD)
  - Manual mode: keyboard teleoperation
  - LeRobot dataset recording: captures observation + action every tick into a
    LeRobotDataset-compatible format (Parquet + optional video).

Usage:
    python hybrid_data_collection_3.py \
        --repo_id my_user/my_insertion_dataset \
        --task "Insert SFP cable into NIC port" \
        --scene_id scene_42 \
        --batch_id batch_007 \
        --target_module nic_card_mount_1 \
        --port_name sfp_port_0 \
        --cable_name cable_0 \
        --plug_name sfp_tip

The node runs one episode (INIT → ... → DONE), saves the episode, finalizes
the dataset, and exits cleanly.

The data is stored in a LeRobotDataset format with the following schema:
- observation.state: 32-dim float32 vector (TCP pose, velocity, error, joint positions, force/torque)
- action: 13-dim float32 vector (pose command + stiffness diag) for auto mode,
          or velocity command for manual mode (with zero stiffness).
- observation.images.{cam}: optional video frames from each camera (scaled down)
The episode metadata (scene_id, batch_id, task, etc.) is stored in a JSON sidecar next to the dataset
for easy filtering and analysis later on.
Location of dataset on disk: ~/.cache/huggingface/lerobot/my_user/my_insertion_dataset (if local_root not set)
"""

import argparse
import json
import logging
import math
import os
import sys
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import numpy as np
import rclpy
import torch
from aic_control_interfaces.msg import (
    ControllerState,
    MotionUpdate,
    TrajectoryGenerationMode,
)
from aic_control_interfaces.srv import ChangeTargetMode
from geometry_msgs.msg import (
    Point,
    Pose,
    Quaternion,
    Transform,
    Twist,
    Vector3,
    Wrench,
    WrenchStamped,
)
from PIL import Image as PILImage
from pynput import keyboard
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from rclpy.time import Time
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Header, String
from std_srvs.srv import Trigger
from tf2_ros import Buffer, TransformException, TransformListener

latched_qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)

os.environ["HF_HUB_OFFLINE"] = "1"  # Ensure no network calls to Hugging Face Hub during recording

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════════

SLOW_LINEAR_VEL = 0.02
SLOW_ANGULAR_VEL = 0.08
FAST_LINEAR_VEL = 0.025
FAST_ANGULAR_VEL = 0.11

KEY_MAPPINGS = {
    "d": (1, 0, 0, 0, 0, 0),
    "a": (-1, 0, 0, 0, 0, 0),
    "w": (0, -1, 0, 0, 0, 0),
    "s": (0, 1, 0, 0, 0, 0),
    "r": (0, 0, -1, 0, 0, 0),
    "f": (0, 0, 1, 0, 0, 0),
    "W": (0, 0, 0, 1, 0, 0),
    "S": (0, 0, 0, -1, 0, 0),
    "A": (0, 0, 0, 0, -1, 0),
    "D": (0, 0, 0, 0, 1, 0),
    "e": (0, 0, 0, 0, 0, 1),
    "q": (0, 0, 0, 0, 0, -1),
}


class Phase(Enum):
    INIT_SERVICES = 1
    APPROACH = 2
    CORRECT = 3
    DESCEND = 4
    HOLD = 5
    DONE = 6
    EXPLORE = 7  # slow hover sweep over the board for image-based policy training


# ═══════════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class InsertionConfig:
    """All tuneable parameters for the waypoint insertion controller."""

    max_integrator_windup: float = 0.05
    i_gain: float = 0.15
    approach_z_offset: float = 0.2
    descend_step: float = 0.0005
    hold_time: float = 5.0
    time_limit: float = 180.0  # max episode wall-clock seconds (auto + manual)
    tick_rate: float = 0.05  # 20 Hz
    approach_steps: int = 100
    converge_threshold: float = 0.02
    converge_max_ticks: int = 200
    converge_stable_count: int = 10

    approach_stiffness: list = field(default_factory=lambda: [90.0] * 3 + [50.0] * 3)
    approach_damping: list = field(default_factory=lambda: [50.0] * 3 + [20.0] * 3)
    correct_stiffness: list = field(default_factory=lambda: [200.0] * 3 + [100.0] * 3)
    correct_damping: list = field(default_factory=lambda: [80.0] * 3 + [40.0] * 3)
    insert_stiffness: list = field(default_factory=lambda: [200.0] * 3 + [100.0] * 3)
    insert_damping: list = field(default_factory=lambda: [80.0] * 3 + [40.0] * 3)

    insert_feedforward_z: float = 0.0
    insert_force_limit: float = 40.0
    feedforward_force: tuple = (0.0, 0.0, 0.0)

    # ── Explore mode ──────────────────────────────────────────────────────
    explore_sweep_x: float = 0.04  # ±X sweep half-width — stays within NIC card span (m)
    explore_sweep_y: float = 0.03  # ±Y sweep half-width — shallow, avoids card edges (m)
    explore_ramp_ticks: int = 40  # ticks to ramp from current pos to hover (2s)
    explore_ticks: int = 200  # sweep ticks after ramp (10s at 20 Hz)
    explore_stiffness: list = field(default_factory=lambda: [60.0] * 3 + [30.0] * 3)
    explore_damping: list = field(default_factory=lambda: [40.0] * 3 + [20.0] * 3)


@dataclass
class RecordingConfig:
    """Configuration for LeRobot dataset recording."""

    repo_id: str = "local/aic_insertion_dataset"
    task: str = "Insert cable into port"
    fps: int = 20  # Should match 1/tick_rate
    robot_type: str = "ur5e_aic"

    # Metadata for scene/batch tracking — stored as a sidecar JSON
    scene_id: str = ""
    batch_id: str = ""

    # Camera image scaling (matches AICRobotAICControllerConfig)
    image_scale: float = 0.25  # 1.0 = full res, 0.5 = half res, etc.
    image_width: int = 1152
    image_height: int = 1024

    # Feature toggles
    record_images: bool = True
    local_root: str | None = None  # None = default HF cache


# ═══════════════════════════════════════════════════════════════════════════════
# LeRobot Recorder
# ═══════════════════════════════════════════════════════════════════════════════


class LeRobotRecorder:
    """
    Wraps the LeRobotDataset `create → add_frame → save_episode → finalize`
    lifecycle and keeps it completely decoupled from the ROS node.

    Observation vector (32-dim) matches the AICRobotAICController driver so
    that trained policies (ACT, diffusion, etc.) can consume the data directly.

    Action vector (13-dim) encodes the full pose-mode command:
      [pos.x, pos.y, pos.z, quat.w, quat.x, quat.y, quat.z,
       stiffness_diag(6)]
    This is intentionally richer than the 6-dim velocity actions used by the
    stock lerobot-record path, because our pipeline uses position-mode control.
    """

    # Observation feature keys — must match RunACT's prepare_observations order
    OBS_STATE_DIM = 32  # 7 pose + 6 vel + 6 error + 7 joint pos + 3 force + 3 torque
    ACTION_DIM = 13  # 7 pose + 6 stiffness diagonal

    def __init__(self, config: RecordingConfig):
        self.config = config
        self.dataset = None
        self._episode_active = False
        self._frame_count = 0

    # ── lifecycle ──────────────────────────────────────────────────────────

    def setup(self):
        """Create (or resume) a LeRobotDataset on disk."""
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        scaled_h = int(self.config.image_height * self.config.image_scale)
        scaled_w = int(self.config.image_width * self.config.image_scale)

        features = {
            "observation.state": {
                "dtype": "float32",
                "shape": (self.OBS_STATE_DIM,),
                "names": {
                    "state": [
                        "tcp_pose.position.x",
                        "tcp_pose.position.y",
                        "tcp_pose.position.z",
                        "tcp_pose.orientation.x",
                        "tcp_pose.orientation.y",
                        "tcp_pose.orientation.z",
                        "tcp_pose.orientation.w",
                        "tcp_velocity.linear.x",
                        "tcp_velocity.linear.y",
                        "tcp_velocity.linear.z",
                        "tcp_velocity.angular.x",
                        "tcp_velocity.angular.y",
                        "tcp_velocity.angular.z",
                        "tcp_error.x",
                        "tcp_error.y",
                        "tcp_error.z",
                        "tcp_error.rx",
                        "tcp_error.ry",
                        "tcp_error.rz",
                        "joint_positions.0",
                        "joint_positions.1",
                        "joint_positions.2",
                        "joint_positions.3",
                        "joint_positions.4",
                        "joint_positions.5",
                        "joint_positions.6",
                        "compensated_force.x",
                        "compensated_force.y",
                        "compensated_force.z",
                        "compensated_torque.x",
                        "compensated_torque.y",
                        "compensated_torque.z",
                    ],
                },
            },
            "action": {
                "dtype": "float32",
                "shape": (self.ACTION_DIM,),
                "names": {
                    "action": [
                        "pose.position.x",
                        "pose.position.y",
                        "pose.position.z",
                        "pose.orientation.w",
                        "pose.orientation.x",
                        "pose.orientation.y",
                        "pose.orientation.z",
                        "stiffness.0",
                        "stiffness.1",
                        "stiffness.2",
                        "stiffness.3",
                        "stiffness.4",
                        "stiffness.5",
                    ],
                },
            },
        }

        if self.config.record_images:
            for cam in ("left_camera", "center_camera", "right_camera"):
                features[f"observation.images.{cam}"] = {
                    "dtype": "video",
                    "shape": (scaled_h, scaled_w, 3),
                    "names": ["height", "width", "channels"],
                }

        root_path = Path(self.config.local_root) if self.config.local_root else None
        if root_path and (root_path / "meta" / "info.json").exists():
            logger.info(f"LeRobotRecorder: found existing dataset at {root_path}, resuming")
            logger.info(f"LeRobotRecorder: resuming dataset at {self.config.repo_id}")
            self.dataset = LeRobotDataset(
                repo_id=self.config.repo_id,
                root=root_path,
            )
        else:
            logger.info(f"LeRobotRecorder: creating local dataset at {root_path}")
            create_kwargs = dict(
                repo_id=self.config.repo_id,
                fps=self.config.fps,
                robot_type=self.config.robot_type,
                features=features,
                root=root_path,
            )
            self.dataset = LeRobotDataset.create(**create_kwargs)

        logger.info(
            f"LeRobotRecorder: dataset ready with {self.dataset.meta.total_episodes} existing episodes"
        )

    def begin_episode(self):
        """Mark the start of a new episode."""
        self._episode_active = True
        self._frame_count = 0
        logger.info("LeRobotRecorder: episode started")

    def add_frame(
        self,
        obs_state: np.ndarray,
        action: np.ndarray,
        images: dict | None = None,
        task: str | None = None,
    ):
        """
        Record one timestep.

        Args:
            obs_state: (32,) float32 array matching ObservationState order.
            action:    (13,) float32 array [pose(7) + stiffness_diag(6)].
            images:    dict mapping camera name → np.ndarray (H, W, 3) uint8.
            task:      optional task string for this frame.
        """
        if not self._episode_active or self.dataset is None:
            return

        frame = {
            "observation.state": torch.from_numpy(obs_state).float(),
            "action": torch.from_numpy(action).float(),
        }

        if images:
            scale = self.config.image_scale
            for cam_name, img_np in images.items():
                if scale != 1.0:
                    h, w = img_np.shape[:2]
                    new_size = (int(w * scale), int(h * scale))
                    img_np = np.array(PILImage.fromarray(img_np).resize(new_size, PILImage.LANCZOS))
                frame[f"observation.images.{cam_name}"] = torch.from_numpy(img_np)

        if task:
            frame["task"] = task

        self.dataset.add_frame(frame)
        self._frame_count += 1

    def end_episode(self):
        """Save the current episode to disk."""
        if not self._episode_active or self.dataset is None:
            return
        self.dataset.save_episode()
        self._episode_active = False
        logger.info(f"LeRobotRecorder: episode saved ({self._frame_count} frames)")

    def finalize(self):
        """Close parquet writers and compute stats. Call once after all episodes."""
        if self.dataset is None:
            return
        self.dataset.finalize()
        logger.info("LeRobotRecorder: dataset finalized (stats computed, writers closed)")

    def save_metadata_sidecar(self, extra: dict):
        """
        Write a JSON sidecar next to the dataset with custom metadata that
        LeRobot's schema doesn't natively support (scene_id, batch_id, etc.).

        This file lives at  <dataset_root>/meta/custom_metadata.json
        and can be loaded later for filtering / analysis.
        """
        if self.dataset is None:
            return

        root = Path(self.dataset.root) if hasattr(self.dataset, "root") else None
        if root is None:
            root = Path.home() / ".cache" / "huggingface" / "lerobot" / self.config.repo_id
        meta_dir = root / "meta"
        meta_dir.mkdir(parents=True, exist_ok=True)
        sidecar_path = meta_dir / "custom_metadata.json"

        existing = {}
        if sidecar_path.exists():
            with open(sidecar_path) as f:
                existing = json.load(f)

        # Append episode-level metadata keyed by episode index
        ep_idx = self.dataset.meta.total_episodes - 1 if hasattr(self.dataset, "meta") else 0
        if "episodes" not in existing:
            existing["episodes"] = {}
        existing["episodes"][str(ep_idx)] = extra

        # Also keep global-level fields
        existing["repo_id"] = self.config.repo_id
        existing["robot_type"] = self.config.robot_type
        existing["fps"] = self.config.fps

        with open(sidecar_path, "w") as f:
            json.dump(existing, f, indent=2)
        logger.info(f"LeRobotRecorder: metadata sidecar written to {sidecar_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# Quaternion helpers (pure-python, no external dep)
# ═══════════════════════════════════════════════════════════════════════════════


def _quat_inv(q):
    return (q[0], -q[1], -q[2], -q[3])


def _quat_mul(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return (
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    )


def _quat_slerp(q0, q1, t):
    dot = sum(a * b for a, b in zip(q0, q1, strict=False))
    if dot < 0.0:
        q1 = tuple(-x for x in q1)
        dot = -dot
    dot = min(dot, 1.0)
    if dot > 0.9995:
        r = tuple(a + t * (b - a) for a, b in zip(q0, q1, strict=False))
        n = math.sqrt(sum(x * x for x in r))
        return tuple(x / n for x in r)
    t0 = math.acos(dot)
    t1 = t0 * t
    s0 = math.cos(t1) - dot * math.sin(t1) / math.sin(t0)
    s1 = math.sin(t1) / math.sin(t0)
    r = tuple(s0 * a + s1 * b for a, b in zip(q0, q1, strict=False))
    n = math.sqrt(sum(x * x for x in r))
    return tuple(x / n for x in r)


# ═══════════════════════════════════════════════════════════════════════════════
# Pose Calculator — pure math, no ROS dependency
# ═══════════════════════════════════════════════════════════════════════════════


class GripperPoseCalculator:
    """
    Computes gripper target poses for the insertion pipeline.
    Separated from the ROS node so it can be unit-tested independently.
    """

    def __init__(self, config: InsertionConfig):
        self.config = config
        self._ix = 0.0
        self._iy = 0.0
        self._start_pos = None

    def reset(self):
        self._ix = 0.0
        self._iy = 0.0
        self._start_pos = None

    def calculate(
        self,
        port_tf: Transform,
        plug_tf: Transform,
        gripper_tf: Transform,
        slerp_fraction: float = 1.0,
        position_fraction: float = 1.0,
        z_offset: float = 0.1,
        reset_xy_integrator: bool = False,
    ) -> Pose:
        q_port = (port_tf.rotation.w, port_tf.rotation.x, port_tf.rotation.y, port_tf.rotation.z)
        q_plug = (plug_tf.rotation.w, plug_tf.rotation.x, plug_tf.rotation.y, plug_tf.rotation.z)
        q_grip = (
            gripper_tf.rotation.w,
            gripper_tf.rotation.x,
            gripper_tf.rotation.y,
            gripper_tf.rotation.z,
        )

        q_target = _quat_mul(q_port, _quat_mul(_quat_inv(q_plug), q_grip))
        q_slerp = _quat_slerp(q_grip, q_target, slerp_fraction)

        gx, gy, gz = gripper_tf.translation.x, gripper_tf.translation.y, gripper_tf.translation.z
        if self._start_pos is None:
            self._start_pos = (gx, gy, gz)

        lx, ly, lz = plug_tf.translation.x, plug_tf.translation.y, plug_tf.translation.z
        px, py, pz = port_tf.translation.x, port_tf.translation.y, port_tf.translation.z

        ex, ey = px - lx, py - ly
        if reset_xy_integrator:
            self._ix, self._iy = 0.0, 0.0
        else:
            w = self.config.max_integrator_windup
            self._ix = float(np.clip(self._ix + ex, -w, w))
            self._iy = float(np.clip(self._iy + ey, -w, w))

        ig = self.config.i_gain
        tx = px + ig * self._ix + (gx - lx)
        ty = py + ig * self._iy + (gy - ly)
        tz = pz + z_offset + (gz - lz)

        bx = position_fraction * tx + (1.0 - position_fraction) * self._start_pos[0]
        by = position_fraction * ty + (1.0 - position_fraction) * self._start_pos[1]
        bz = position_fraction * tz + (1.0 - position_fraction) * self._start_pos[2]

        return Pose(
            position=Point(x=bx, y=by, z=bz),
            orientation=Quaternion(w=q_slerp[0], x=q_slerp[1], y=q_slerp[2], z=q_slerp[3]),
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Observation Builder — assembles the 32-dim state vector
# ═══════════════════════════════════════════════════════════════════════════════


class ObservationBuilder:
    """Constructs the 32-dim observation state matching AICRobotAICController."""

    OBS_STATE_DIM = 32

    @staticmethod
    def build(
        controller_state: ControllerState, joint_states: JointState, compensated_wrench: Wrench
    ) -> np.ndarray:
        if controller_state is None or joint_states is None:
            return np.zeros(ObservationBuilder.OBS_STATE_DIM, dtype=np.float32)

        tcp_pose = controller_state.tcp_pose
        tcp_vel = controller_state.tcp_velocity
        tcp_err = controller_state.tcp_error
        jp = joint_states.position
        force = compensated_wrench.force if compensated_wrench else Vector3(x=0.0, y=0.0, z=0.0)
        torque = compensated_wrench.torque if compensated_wrench else Vector3(x=0.0, y=0.0, z=0.0)

        return np.array(
            [
                tcp_pose.position.x,
                tcp_pose.position.y,
                tcp_pose.position.z,
                tcp_pose.orientation.x,
                tcp_pose.orientation.y,
                tcp_pose.orientation.z,
                tcp_pose.orientation.w,
                tcp_vel.linear.x,
                tcp_vel.linear.y,
                tcp_vel.linear.z,
                tcp_vel.angular.x,
                tcp_vel.angular.y,
                tcp_vel.angular.z,
                tcp_err[0],
                tcp_err[1],
                tcp_err[2],
                tcp_err[3],
                tcp_err[4],
                tcp_err[5],
                jp[0],
                jp[1],
                jp[2],
                jp[3],
                jp[4],
                jp[5],
                jp[6] if len(jp) > 6 else 0.0,
                force.x,
                force.y,
                force.z,
                torque.x,
                torque.y,
                torque.z,
            ],
            dtype=np.float32,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Action Builder — encodes the pose-mode command as a 13-dim vector
# ═══════════════════════════════════════════════════════════════════════════════


class ActionBuilder:
    """Encodes a pose-mode MotionUpdate into a flat action vector."""

    @staticmethod
    def from_pose(pose: Pose, stiffness: list) -> np.ndarray:
        """
        Returns (13,) array:
          [pos.x, pos.y, pos.z, quat.w, quat.x, quat.y, quat.z,
           stiff_diag[0..5]]
        """
        return np.array(
            [
                pose.position.x,
                pose.position.y,
                pose.position.z,
                pose.orientation.w,
                pose.orientation.x,
                pose.orientation.y,
                pose.orientation.z,
                stiffness[0],
                stiffness[1],
                stiffness[2],
                stiffness[3],
                stiffness[4],
                stiffness[5],
            ],
            dtype=np.float32,
        )

    @staticmethod
    def from_velocity(twist: Twist) -> np.ndarray:
        """
        For manual-mode frames, encode velocity as the first 6 dims and zero-pad
        stiffness. This lets both modes coexist in the same action column.
        NOTE: during training you may want to filter by mode or train separate
        heads; the 'mode' column in the sidecar metadata helps with that.
        """
        return np.array(
            [
                twist.linear.x,
                twist.linear.y,
                twist.linear.z,
                0.0,
                0.0,
                0.0,
                0.0,  # no orientation in velocity mode
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            dtype=np.float32,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# ROS Node — orchestrates everything
# ═══════════════════════════════════════════════════════════════════════════════


class HybridDataCollectorNode(Node):
    """
    A single-episode data collector that:
      1. Initialises controller services (tare FT, switch to Cartesian).
      2. Runs the phased insertion (with optional manual keyboard override).
      3. Records every tick into a LeRobotDataset via LeRobotRecorder.
      4. On DONE → saves the episode, writes metadata sidecar, finalizes, exits.
    """

    def __init__(
        self,
        target_module: str,
        port_name: str,
        cable_name: str,
        plug_name: str,
        insertion_config: InsertionConfig,
        recording_config: RecordingConfig,
        auto_mode: bool = True,
        no_explore: bool = False,
    ):
        super().__init__("hybrid_data_collector")

        # Task parameters
        self.target_module = target_module
        self.port_name = port_name
        self.cable_name = cable_name
        self.plug_name = plug_name

        self.port_frame = f"task_board/{target_module}/{port_name}_link"
        self.cable_tip_frame = f"{cable_name}/{plug_name}_link"
        self.base_frame = "base_link"

        # Configs
        self.ins_cfg = insertion_config
        self.rec_cfg = recording_config
        self.no_explore = no_explore

        # Sub-components
        self.pose_calc = GripperPoseCalculator(self.ins_cfg)
        self.recorder = LeRobotRecorder(self.rec_cfg)

        # Phase state machine
        self.phase = Phase.INIT_SERVICES
        self.auto_mode = auto_mode
        self.inserted_successfully = False
        self._step = 0
        self._tick = 0
        self._z_off = 0.0
        self._corr_ticks = 0
        self._corr_stable = 0
        self._hold_start = None
        self._explore_tick = 0
        self._explore_start: tuple = (0.0, 0.0, 0.0)
        self._episode_finished = False
        self._scene_metadata = None

        # Keyboard teleop state
        self.active_keys = set()
        self.linear_vel = SLOW_LINEAR_VEL
        self.angular_vel = SLOW_ANGULAR_VEL
        self.manual_frame_id = "gripper/tcp"

        # Latest sensor readings
        self.current_error = None
        self.current_force = None
        self.last_controller_state: ControllerState | None = None
        self.last_joint_states: JointState | None = None
        self.last_images: dict = {}  # cam_name → np.ndarray
        self.last_wrench_msg: WrenchStamped | None = None

        # ── ROS 2 interfaces ──────────────────────────────────────────────
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.pose_pub = self.create_publisher(MotionUpdate, "/aic_controller/pose_commands", 10)
        self.compensated_wrench_pub = self.create_publisher(
            WrenchStamped, "/fts_compensated/wrench", 10
        )

        self.state_sub = self.create_subscription(
            ControllerState,
            "/aic_controller/controller_state",
            self._controller_state_cb,
            10,
        )
        self.joint_sub = self.create_subscription(
            JointState,
            "/joint_states",
            self._joint_states_cb,
            10,
        )
        self.wrench_sub = self.create_subscription(
            WrenchStamped,
            "/fts_broadcaster/wrench",
            self._wrench_cb,
            10,
        )
        self.event_sub = self.create_subscription(
            String,
            "/scoring/insertion_event",
            self._event_cb,
            10,
        )

        self.scene_meta_sub = self.create_subscription(
            String,
            "/data_collection/scene_metadata",
            self._scene_meta_cb,
            latched_qos,
        )

        if self.rec_cfg.record_images:
            for cam in ("left_camera", "center_camera", "right_camera"):
                self.create_subscription(
                    Image,
                    f"/{cam}/image",
                    lambda msg, c=cam: self._image_cb(c, msg),
                    10,
                )

        self.tare_client = self.create_client(
            Trigger,
            "/aic_controller/tare_force_torque_sensor",
        )
        self.mode_client = self.create_client(
            ChangeTargetMode,
            "/aic_controller/change_target_mode",
        )

        # ── Keyboard listener ─────────────────────────────────────────────
        self.kb_listener = keyboard.Listener(
            on_press=self._on_key_press,
            on_release=self._on_key_release,
        )
        self.kb_listener.start()

        # ── Setup recorder ────────────────────────────────────────────────
        self.recorder.setup()
        self.recorder.begin_episode()
        self._episode_start_time = self.get_clock().now()

        self.get_logger().info("=" * 60)
        self.get_logger().info("Hybrid Data Collector — single episode mode")
        self.get_logger().info(f"  repo_id   : {self.rec_cfg.repo_id}")
        self.get_logger().info(f"  scene_id  : {self.rec_cfg.scene_id}")
        self.get_logger().info(f"  batch_id  : {self.rec_cfg.batch_id}")
        self.get_logger().info(f"  task      : {self.rec_cfg.task}")
        self.get_logger().info("Press 'T' to toggle Auto/Manual. Ctrl+C to abort.")
        self.get_logger().info("=" * 60)

        self.timer = self.create_timer(self.ins_cfg.tick_rate, self._on_tick)

    # ── ROS callbacks ─────────────────────────────────────────────────────

    def _controller_state_cb(self, msg: ControllerState):
        self.last_controller_state = msg
        err = msg.tcp_error
        self.current_error = math.sqrt(err[0] ** 2 + err[1] ** 2 + err[2] ** 2)

    def _joint_states_cb(self, msg: JointState):
        self.last_joint_states = msg

    def _wrench_cb(self, msg: WrenchStamped):
        self.last_wrench_msg = msg
        f = msg.wrench.force
        self.current_force = math.sqrt(f.x**2 + f.y**2 + f.z**2)

    def _get_compensated_wrench(self):
        if self.last_wrench_msg is None or self.last_controller_state is None:
            return None
        raw = self.last_wrench_msg.wrench
        tare = self.last_controller_state.fts_tare_offset.wrench
        compensated = Wrench(
            force=Vector3(
                x=raw.force.x - tare.force.x,
                y=raw.force.y - tare.force.y,
                z=raw.force.z - tare.force.z,
            ),
            torque=Vector3(
                x=raw.torque.x - tare.torque.x,
                y=raw.torque.y - tare.torque.y,
                z=raw.torque.z - tare.torque.z,
            ),
        )
        return compensated

    def _get_compensated_force_magnitude(self):
        w = self._get_compensated_wrench()
        if w is None:
            return None
        f = w.force
        return math.sqrt(f.x**2 + f.y**2 + f.z**2)

    def _publish_compensated_wrench(self):
        w = self._get_compensated_wrench()
        if w is None:
            return
        msg = WrenchStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "fts_sensor"
        msg.wrench = w
        self.compensated_wrench_pub.publish(msg)

    def _event_cb(self, msg: String):
        expected = f"/{self.target_module}/{self.port_name}"
        if msg.data == expected:
            self.get_logger().info(f"SUCCESS: insertion event for {expected}")
            self.inserted_successfully = True

    def _image_cb(self, cam_name: str, msg: Image):
        img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
        self.last_images[cam_name] = img

    def _scene_meta_cb(self, msg: String):
        try:
            self._scene_metadata = json.loads(msg.data)
            self.get_logger().info(f"Received scene metadata: {self._scene_metadata}")

            self.rec_cfg.scene_id = self._scene_metadata.get("scene_id", "")
            self.rec_cfg.batch_id = self._scene_metadata.get("batch_id", "")
            self.rec_cfg.task = self._scene_metadata.get("task", self.rec_cfg.task)
            self.cable_name = self._scene_metadata.get("cable_name", self.cable_name)
            self.plug_name = self._scene_metadata.get("plug_name", self.plug_name)
            self.target_module = self._scene_metadata.get("target_module", self.target_module)
            self.port_name = self._scene_metadata.get("port_name", self.port_name)

            self.port_frame = f"task_board/{self.target_module}/{self.port_name}_link"
            self.cable_tip_frame = f"{self.cable_name}/{self.plug_name}_link"

        except json.JSONDecodeError:
            self.get_logger().warn(f"Failed to decode scene metadata JSON: {msg.data}")

    # ── Keyboard ──────────────────────────────────────────────────────────

    def _on_key_press(self, key):
        try:
            if hasattr(key, "char") and key.char is not None:
                k = key.char
                if k.lower() == "x" and self.phase not in (Phase.DONE,):
                    # Enter EXPLORE mode — gentle sweep over board for image capture
                    self.auto_mode = True
                    self.phase = Phase.EXPLORE
                    self._explore_tick = 0
                    self.get_logger().warn(
                        ">>> EXPLORE mode — sweeping over board for image capture <<<"
                    )
                elif k.lower() == "t":
                    self.auto_mode = not self.auto_mode
                    if self.auto_mode:
                        self.pose_calc.reset()
                        self._step = 0
                        if self.phase not in (Phase.INIT_SERVICES, Phase.DONE, Phase.HOLD):
                            port_tf = self._get_tf(self.base_frame, self.port_frame)
                            plug_tf = self._get_tf(self.base_frame, self.cable_tip_frame)
                            if port_tf and plug_tf:
                                self._z_off = plug_tf.translation.z - port_tf.translation.z
                                if self._z_off > self.ins_cfg.approach_z_offset * 0.8:
                                    self.phase = Phase.APPROACH
                                else:
                                    # Already close — use a short ramp via APPROACH so the
                                    # controller doesn't jump immediately to port position.
                                    # Clamp approach_steps proportionally to remaining distance.
                                    frac_remaining = max(
                                        0.1, self._z_off / self.ins_cfg.approach_z_offset
                                    )
                                    self._step = int(
                                        (1.0 - frac_remaining) * self.ins_cfg.approach_steps
                                    )
                                    self.phase = Phase.APPROACH
                                self.get_logger().info(
                                    f"Resuming auto: {self.phase.name} z_off={self._z_off:.4f} step={self._step}"
                                )
                            else:
                                self.phase = Phase.APPROACH
                    mode_str = "AUTO" if self.auto_mode else "MANUAL"
                    self.get_logger().warn(f">>> MODE: {mode_str} <<<")
                elif not self.auto_mode:
                    if k == "k":
                        self.linear_vel, self.angular_vel = SLOW_LINEAR_VEL, SLOW_ANGULAR_VEL
                    elif k == "l":
                        self.linear_vel, self.angular_vel = FAST_LINEAR_VEL, FAST_ANGULAR_VEL
                    elif k == "n":
                        self.manual_frame_id = "gripper/tcp"
                    elif k == "m":
                        self.manual_frame_id = "base_link"
                    else:
                        self.active_keys.add(k)
        except AttributeError:
            pass

    def _on_key_release(self, key):
        try:
            if hasattr(key, "char") and key.char is not None:
                self.active_keys.discard(key.char)
        except AttributeError:
            pass

    # ── TF helper ─────────────────────────────────────────────────────────

    def _get_tf(self, target: str, source: str) -> Transform | None:
        try:
            return self.tf_buffer.lookup_transform(target, source, Time()).transform
        except TransformException:
            return None

    # ── Command publishers ────────────────────────────────────────────────

    def _publish_pose(self, pose: Pose, stiffness: list, damping: list, ff: tuple):
        msg = MotionUpdate()
        msg.header = Header(
            frame_id=self.base_frame,
            stamp=self.get_clock().now().to_msg(),
        )
        msg.pose = pose
        msg.target_stiffness = np.diag(stiffness).flatten()
        msg.target_damping = np.diag(damping).flatten()
        msg.feedforward_wrench_at_tip = Wrench(
            force=Vector3(x=ff[0], y=ff[1], z=ff[2]),
            torque=Vector3(x=0.0, y=0.0, z=0.0),
        )
        msg.wrench_feedback_gains_at_tip = [0.5, 0.5, 0.5, 0.0, 0.0, 0.0]
        msg.trajectory_generation_mode = TrajectoryGenerationMode(
            mode=TrajectoryGenerationMode.MODE_POSITION,
        )
        self.pose_pub.publish(msg)

    def _publish_velocity(self):
        vel = np.zeros(6)
        for k in self.active_keys:
            if k in KEY_MAPPINGS:
                v = KEY_MAPPINGS[k]
                vel[:3] += np.array(v[:3], dtype=float) * self.linear_vel
                vel[3:] += np.array(v[3:], dtype=float) * self.angular_vel

        twist = Twist(
            linear=Vector3(x=vel[0], y=vel[1], z=vel[2]),
            angular=Vector3(x=vel[3], y=vel[4], z=vel[5]),
        )
        msg = MotionUpdate()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.manual_frame_id
        msg.velocity = twist
        msg.target_stiffness = np.diag([85.0] * 6).flatten()
        msg.target_damping = np.diag([75.0] * 6).flatten()
        msg.feedforward_wrench_at_tip = Wrench()
        msg.wrench_feedback_gains_at_tip = [0.0] * 6
        msg.trajectory_generation_mode = TrajectoryGenerationMode(
            mode=TrajectoryGenerationMode.MODE_VELOCITY,
        )
        self.pose_pub.publish(msg)
        return twist

    # ── Recording helper ──────────────────────────────────────────────────

    def _record_frame(self, action_vec: np.ndarray):
        compensated_wrench = self._get_compensated_wrench()
        obs = ObservationBuilder.build(
            self.last_controller_state, self.last_joint_states, compensated_wrench
        )
        images = self.last_images if self.rec_cfg.record_images else None
        self.recorder.add_frame(
            obs_state=obs,
            action=action_vec,
            images=images,
            task=self.rec_cfg.task,
        )

    # ── Main tick ─────────────────────────────────────────────────────────

    def _on_tick(self):
        if self._episode_finished:
            return

        if self._scene_metadata is None:
            self.get_logger().warn("Waiting for scene metadata...")
            return

        # ── INIT_SERVICES ─────────────────────────────────────────────────
        if self.phase == Phase.INIT_SERVICES:
            if not self.tare_client.service_is_ready() or not self.mode_client.service_is_ready():
                return
            self.get_logger().info("Taring FT sensor...")
            self.tare_client.call_async(Trigger.Request())
            self.get_logger().info("Switching to Cartesian mode...")
            req = ChangeTargetMode.Request()
            req.target_mode.mode = 1
            self.mode_client.call_async(req)
            if self.no_explore:
                self.phase = Phase.APPROACH
                self._step = 0
                print("=" * 60)
                print(
                    f" Task is to insert {self.cable_name} into {self.port_name} on module {self.target_module}"
                )
                print(" Skipping EXPLORE — going straight to APPROACH.")
                print("=" * 60)
            else:
                self.phase = Phase.EXPLORE
                self._explore_tick = 0
                print("=" * 60)
                print(
                    f" Task is to insert {self.cable_name} into {self.port_name} on module {self.target_module}"
                )
                print(" Starting EXPLORE sweep before approach...")
                print("=" * 60)

        # ── Episode time limit ────────────────────────────────────────────
        elapsed = (self.get_clock().now() - self._episode_start_time).nanoseconds / 1e9
        if elapsed > self.ins_cfg.time_limit:
            self.get_logger().warn(
                f"Episode TIMED OUT after {elapsed:.1f}s "
                f"(limit={self.ins_cfg.time_limit}s, phase={self.phase.name}) — marking unsuccessful."
            )
            self.inserted_successfully = False
            self._finish_episode()  # raises SystemExit

        self._publish_compensated_wrench()

        # ── Manual mode ───────────────────────────────────────────────────
        if not self.auto_mode:
            twist = self._publish_velocity()
            self._record_frame(ActionBuilder.from_velocity(twist))
            return

        self._tick += 1

        # ── Lookup transforms ─────────────────────────────────────────────
        port_tf = self._get_tf(self.base_frame, self.port_frame)
        plug_tf = self._get_tf(self.base_frame, self.cable_tip_frame)
        grip_tf = self._get_tf(self.base_frame, "gripper/tcp")

        if not all((port_tf, plug_tf, grip_tf)):
            if self._tick % 20 == 0:
                self.get_logger().warn("Waiting for TFs...")
            return

        cfg = self.ins_cfg

        # ── APPROACH ──────────────────────────────────────────────────────
        if self.phase == Phase.APPROACH:
            self._step += 1
            frac = min(1.0, self._step / cfg.approach_steps)
            pose = self.pose_calc.calculate(
                port_tf,
                plug_tf,
                grip_tf,
                frac,
                frac,
                cfg.approach_z_offset,
                True,
            )
            if pose:
                self._publish_pose(
                    pose, cfg.approach_stiffness, cfg.approach_damping, cfg.feedforward_force
                )
                self._record_frame(ActionBuilder.from_pose(pose, cfg.approach_stiffness))
            if self._tick % 20 == 0:
                self.get_logger().info(f"Approach {frac * 100:.0f}%")
            if frac >= 1.0:
                self.phase = Phase.CORRECT
                self._corr_ticks = 0
                self._corr_stable = 0

        # ── CORRECT ───────────────────────────────────────────────────────
        elif self.phase == Phase.CORRECT:
            self._corr_ticks += 1
            pose = self.pose_calc.calculate(
                port_tf,
                plug_tf,
                grip_tf,
                1.0,
                1.0,
                cfg.approach_z_offset,
                False,
            )
            if pose:
                self._publish_pose(
                    pose, cfg.correct_stiffness, cfg.correct_damping, cfg.feedforward_force
                )
                self._record_frame(ActionBuilder.from_pose(pose, cfg.correct_stiffness))
            if self.current_error and self.current_error < cfg.converge_threshold:
                self._corr_stable += 1
            else:
                self._corr_stable = 0
            if (
                self._corr_stable >= cfg.converge_stable_count
                or self._corr_ticks >= cfg.converge_max_ticks
            ):
                self.get_logger().info("Converged. Descending...")
                self._z_off = cfg.approach_z_offset
                self.phase = Phase.DESCEND

        # ── DESCEND ───────────────────────────────────────────────────────
        elif self.phase == Phase.DESCEND:
            if self.inserted_successfully:
                self.get_logger().info(">>> INSERTION EVENT! Holding... <<<")
                self._hold_start = self.get_clock().now()
                self.phase = Phase.HOLD
                return
            self._z_off -= cfg.descend_step
            ff_z = cfg.insert_feedforward_z
            compensated_force = self._get_compensated_force_magnitude()
            if compensated_force and compensated_force > cfg.insert_force_limit:
                ff_z = 0.0
            pose = self.pose_calc.calculate(
                port_tf,
                plug_tf,
                grip_tf,
                1.0,
                1.0,
                self._z_off,
                False,
            )
            if pose:
                self._publish_pose(pose, cfg.insert_stiffness, cfg.insert_damping, (0.0, 0.0, ff_z))
                self._record_frame(ActionBuilder.from_pose(pose, cfg.insert_stiffness))

        # ── HOLD ──────────────────────────────────────────────────────────
        elif self.phase == Phase.HOLD:
            elapsed = (self.get_clock().now() - self._hold_start).nanoseconds / 1e9
            pose = self.pose_calc.calculate(
                port_tf,
                plug_tf,
                grip_tf,
                1.0,
                1.0,
                self._z_off,
                False,
            )
            if pose:
                self._publish_pose(
                    pose, cfg.insert_stiffness, cfg.insert_damping, cfg.feedforward_force
                )
                self._record_frame(ActionBuilder.from_pose(pose, cfg.insert_stiffness))
            if elapsed >= cfg.hold_time:
                self.get_logger().info("Episode complete!")
                self._finish_episode()

        # ── EXPLORE ───────────────────────────────────────────────────────
        elif self.phase == Phase.EXPLORE:
            if not all((port_tf, plug_tf, grip_tf)):
                return  # wait for TFs

            # Hover height = same as APPROACH: port z + approach_z_offset
            # Extra 3 cm above approach height to clear NIC card tops during sweep
            hover_z = port_tf.translation.z + cfg.approach_z_offset + 0.03  # type: ignore[union-attr]

            # Board centre from scene metadata (falls back to port TF)
            meta = self._scene_metadata or {}
            bp = meta.get("board_pose", {})
            cx = float(bp.get("x", port_tf.translation.x))  # type: ignore[union-attr]
            cy = float(bp.get("y", port_tf.translation.y))  # type: ignore[union-attr]

            # Snapshot start position and approach orientation on first tick
            if self._explore_tick == 0:
                self._explore_start = (
                    grip_tf.translation.x,  # type: ignore[union-attr]
                    grip_tf.translation.y,  # type: ignore[union-attr]
                    grip_tf.translation.z,  # type: ignore[union-attr]
                )
                # Use pose_calc at frac=1.0 to get the "facing board" approach orientation,
                # which is what the camera needs to look toward the task board throughout.
                approach_orient_pose = self.pose_calc.calculate(
                    port_tf,
                    plug_tf,
                    grip_tf,
                    1.0,
                    1.0,
                    cfg.approach_z_offset,
                    True,  # type: ignore[arg-type]
                )
                if approach_orient_pose:
                    self._explore_orient = approach_orient_pose.orientation
                else:
                    self._explore_orient = grip_tf.rotation  # type: ignore[union-attr]

            self._explore_tick += 1
            t = self._explore_tick
            ramp = cfg.explore_ramp_ticks

            # ── Phase 1: ramp from current pos to board centre at hover_z ──
            if t <= ramp:
                frac = t / ramp
                sx, sy, sz = self._explore_start
                tx = sx + (cx - sx) * frac
                ty = sy + (cy - sy) * frac
                tz = sz + (hover_z - sz) * frac
            else:
                # ── Phase 2: Lissajous sweep over board ─────────────────
                s = t - ramp
                phase_x = 2.0 * math.pi * s / cfg.explore_ticks
                phase_y = math.pi * s / cfg.explore_ticks
                tx = cx + cfg.explore_sweep_x * math.sin(phase_x)
                ty = cy + cfg.explore_sweep_y * math.sin(phase_y)
                tz = hover_z

            # Orientation: approach quaternion (faces board) + small yaw bias
            # proportional to X offset so camera gaze naturally tracks board centre
            # as gripper sweeps left/right. Max ±8° at full sweep extent.
            base_q = self._explore_orient
            x_offset = tx - cx
            yaw_bias = math.atan2(x_offset, cfg.approach_z_offset) * 0.4  # scale factor
            half = yaw_bias / 2.0
            # dq = small rotation around Z (yaw)
            dqw = math.cos(half)
            dqz = math.sin(half)
            # compose: base_q * dq
            ow = base_q.w * dqw - base_q.z * dqz
            ox = base_q.x * dqw + base_q.y * dqz
            oy = base_q.y * dqw - base_q.x * dqz
            oz = base_q.z * dqw + base_q.w * dqz

            explore_pose = Pose(
                position=Point(x=tx, y=ty, z=tz),
                orientation=Quaternion(w=ow, x=ox, y=oy, z=oz),
            )
            self._publish_pose(
                explore_pose, cfg.explore_stiffness, cfg.explore_damping, (0.0, 0.0, 0.0)
            )
            self._record_frame(ActionBuilder.from_pose(explore_pose, cfg.explore_stiffness))

            if t % 40 == 0:
                self.get_logger().info(
                    f"Explore {'ramp' if t <= ramp else 'sweep'} {t}/{ramp + cfg.explore_ticks}"
                    f" — pos=({tx:.3f}, {ty:.3f}, {tz:.3f})"
                )
            if t >= ramp + cfg.explore_ticks:
                self.get_logger().info("Explore sweep complete — transitioning to APPROACH.")
                self.pose_calc.reset()
                self._step = 0
                self.phase = Phase.APPROACH

    # ── Episode finalisation ──────────────────────────────────────────────

    def _finish_episode(self):
        self._episode_finished = True
        self.auto_mode = False

        # Flush stdin so buffered keypresses from pynput don't spill into the
        # terminal prompt after the app exits.
        try:
            import termios

            termios.tcflush(sys.stdin, termios.TCIFLUSH)
        except Exception:
            pass

        # Save episode
        self.recorder.end_episode()

        # Write sidecar metadata
        self.recorder.save_metadata_sidecar(
            {
                "scene_id": self.rec_cfg.scene_id,
                "batch_id": self.rec_cfg.batch_id,
                "task": self.rec_cfg.task,
                "target_module": self.target_module,
                "port_name": self.port_name,
                "cable_name": self.cable_name,
                "plug_name": self.plug_name,
                "inserted_successfully": self.inserted_successfully,
                "total_ticks": self._tick,
                "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            }
        )

        # Finalize dataset (compute stats, close writers)
        self.recorder.finalize()

        self.get_logger().info("=" * 60)
        self.get_logger().info("Dataset saved & finalized. Shutting down.")
        self.get_logger().info("=" * 60)

        # Request clean shutdown
        raise SystemExit(0)


# ═══════════════════════════════════════════════════════════════════════════════
# CLI entry point
# ═══════════════════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(
        description="Hybrid insertion data collector with LeRobot recording.",
    )
    # Task params
    parser.add_argument("--target_module", default="nic_card_mount_1")
    parser.add_argument("--port_name", default="sfp_port_0")
    parser.add_argument("--cable_name", default="cable_0")
    parser.add_argument("--plug_name", default="sfp_tip")

    # Recording params
    parser.add_argument("--repo_id", default="local/aic_insertion_dataset")
    parser.add_argument("--task", default="Insert cable into port")
    parser.add_argument("--scene_id", default="")
    parser.add_argument("--batch_id", default="")
    parser.add_argument(
        "--dataset_root",
        default=None,
        help="Root directory that holds all batches. Each batch is stored "
        "at <dataset_root>/<batch_id>/. Ignored if --local_root is set.",
    )
    parser.add_argument(
        "--local_root",
        default=None,
        help="Exact directory for this dataset (overrides --dataset_root and HF cache)",
    )
    parser.add_argument("--no_images", action="store_true", help="Disable camera image recording")
    parser.add_argument(
        "--no_explore",
        action="store_true",
        help="Skip the EXPLORE sweep and go straight to APPROACH (saves ~12s per trial)",
    )
    parser.add_argument(
        "--image_scale",
        type=float,
        default=0.25,
        help="Scale factor for camera images (e.g. 0.5 = half-res, 0.25 = quarter-res)",
    )
    parser.add_argument(
        "--control_mode",
        choices=["auto", "manual"],
        default="auto",
        help="Start in auto or manual mode. In manual mode, use keyboard to control the robot and record human demonstrations.",
    )

    args, unknown = parser.parse_known_args(sys.argv[1:])
    print(" Image recording is", "OFF" if args.no_images else "ON")

    # Resolve storage path:
    #   --local_root wins if set.
    #   Otherwise, if --dataset_root and --batch_id are both set, store at <dataset_root>/<batch_id>.
    #   Otherwise fall back to HF cache (local_root=None).
    local_root = args.local_root
    if local_root is None and args.dataset_root and args.batch_id:
        local_root = str(Path(args.dataset_root) / args.batch_id)
        print(f" Dataset path: {local_root}")

    rec_cfg = RecordingConfig(
        repo_id=args.repo_id,
        task=args.task,
        scene_id=args.scene_id,
        batch_id=args.batch_id,
        local_root=local_root,
        record_images=not args.no_images,
        image_scale=args.image_scale,
    )

    ins_cfg = InsertionConfig()

    logging.basicConfig(level=logging.INFO)

    rclpy.init(args=sys.argv)
    node = HybridDataCollectorNode(
        target_module=args.target_module,
        port_name=args.port_name,
        cable_name=args.cable_name,
        plug_name=args.plug_name,
        insertion_config=ins_cfg,
        recording_config=rec_cfg,
        auto_mode=(args.control_mode == "auto"),
        no_explore=args.no_explore,
    )

    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
