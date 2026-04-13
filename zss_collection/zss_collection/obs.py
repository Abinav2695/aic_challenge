"""
Observation space for AIC insertion task.

24-dim float32 vector built from controller state + scene metadata.

Layout
──────
  [0:3]   tcp_position        — x, y, z  (metres)
  [3:9]   tcp_rot6d           — first two columns of rotation matrix (6 floats)
  [9:15]  joint_positions     — UR5e joints 0–5 (6 values, rad)
  [15:18] compensated_force   — FTS force minus tare (N)
  [18:21] compensated_torque  — FTS torque minus tare (N·m)
  [21]    connector_type      — 0.0 = sfp, 1.0 = sc
  [22]    mount_id            — sfp: NIC mount index (0–4), sc: SC port rail (0–1)
  [23]    port_index          — sfp: which SFP port on that NIC (0–1), sc: always 0
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from aic_control_interfaces.msg import ControllerState
from geometry_msgs.msg import WrenchStamped
from sensor_msgs.msg import JointState

# ── Dimensions ────────────────────────────────────────────────

DIM = 24

# Named slices — use these in model code, never bare integers.
TCP_POS_SLICE = slice(0, 3)
TCP_ROT6D_SLICE = slice(3, 9)
JOINT_POS_SLICE = slice(9, 15)
FORCE_SLICE = slice(15, 18)
TORQUE_SLICE = slice(18, 21)
CONNECTOR_SLICE = slice(21, 22)
MOUNT_ID_SLICE = slice(22, 23)
PORT_INDEX_SLICE = slice(23, 24)

NAMES: tuple[str, ...] = (
    "tcp_pos.x",
    "tcp_pos.y",
    "tcp_pos.z",
    "tcp_rot6d.r00",
    "tcp_rot6d.r10",
    "tcp_rot6d.r20",
    "tcp_rot6d.r01",
    "tcp_rot6d.r11",
    "tcp_rot6d.r21",
    "joint_pos.0",
    "joint_pos.1",
    "joint_pos.2",
    "joint_pos.3",
    "joint_pos.4",
    "joint_pos.5",
    "force.x",
    "force.y",
    "force.z",
    "torque.x",
    "torque.y",
    "torque.z",
    "connector_type",
    "mount_id",
    "port_index",
)
assert len(NAMES) == DIM


# ── Quaternion → rot6d ────────────────────────────────────────


def quat_to_rot6d(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    """
    Convert quaternion (x,y,z,w) to rot6d (first two columns of rotation matrix).

    Returns shape (6,): [r00, r10, r20, r01, r11, r21]
    """
    from scipy.spatial.transform import Rotation

    R = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()  # (3,3)
    return np.array([*R[:, 0], *R[:, 1]], dtype=np.float32)


# ── Sensor snapshot ───────────────────────────────────────────


@dataclass
class SensorSnapshot:
    """Latest sensor readings from controller, joints, and force/torque."""

    controller_state: ControllerState
    joint_states: JointState
    wrench: WrenchStamped


# ── Builder ───────────────────────────────────────────────────


class ObservationBuilder:
    """Builds 24-dim observation vector from sensors + scene metadata."""

    @staticmethod
    def compensate_wrench(snapshot: SensorSnapshot) -> tuple[np.ndarray, np.ndarray]:
        """
        Subtract tare offset from raw wrench.
        Returns (force_xyz, torque_xyz) as numpy arrays.
        """
        raw = snapshot.wrench.wrench
        tare = snapshot.controller_state.fts_tare_offset.wrench

        force = np.array(
            [
                raw.force.x - tare.force.x,
                raw.force.y - tare.force.y,
                raw.force.z - tare.force.z,
            ],
            dtype=np.float32,
        )

        torque = np.array(
            [
                raw.torque.x - tare.torque.x,
                raw.torque.y - tare.torque.y,
                raw.torque.z - tare.torque.z,
            ],
            dtype=np.float32,
        )

        return force, torque

    @staticmethod
    def build(
        snapshot: SensorSnapshot | None,
        connector_type: float = 0.0,
        mount_id: float = 0.0,
        port_index: float = 0.0,
    ) -> np.ndarray:
        """
        Build 24-dim observation vector.

        Args:
            snapshot: Sensor data (None → zero vector).
            connector_type: 0.0 = sfp, 1.0 = sc
            mount_id: sfp → NIC mount index (0–4), sc → SC port rail (0–1).
            port_index: sfp → which SFP port on that NIC (0–1), sc → always 0.
        """
        if snapshot is None:
            return np.zeros(DIM, dtype=np.float32)

        cs = snapshot.controller_state
        pos = cs.tcp_pose.position
        ori = cs.tcp_pose.orientation
        joints = snapshot.joint_states.position
        force, torque = ObservationBuilder.compensate_wrench(snapshot)

        # TCP position
        tcp_pos = np.array([pos.x, pos.y, pos.z], dtype=np.float32)

        # TCP orientation as rot6d
        rot6d = quat_to_rot6d(ori.x, ori.y, ori.z, ori.w)

        # Joint positions (6 UR5e joints, no gripper)
        joint_arr = np.zeros(6, dtype=np.float32)
        for i in range(min(len(joints), 6)):
            joint_arr[i] = joints[i]

        # Task metadata
        task_meta = np.array([connector_type, mount_id, port_index], dtype=np.float32)

        return np.concatenate([tcp_pos, rot6d, joint_arr, force, torque, task_meta])

    @staticmethod
    def force_magnitude(snapshot: SensorSnapshot) -> float:
        """Euclidean norm of compensated force (N)."""
        force, _ = ObservationBuilder.compensate_wrench(snapshot)
        return float(np.linalg.norm(force))

    @staticmethod
    def lerobot_feature_spec() -> dict:
        """Feature dict for LeRobotDataset.create() observation.state key."""
        return {
            "dtype": "float32",
            "shape": (DIM,),
            "names": {"state": list(NAMES)},
        }
