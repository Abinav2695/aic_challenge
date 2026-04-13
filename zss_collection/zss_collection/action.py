"""
Action space for AIC insertion task.

9-dim float32: absolute TCP pose in base_link frame.

Layout
──────
  [0:3]  position   — x, y, z  (metres, base_link frame)
  [3:9]  rot6d      — first two columns of rotation matrix (6 floats)

Why absolute pose
─────────────────
The policy predicts where the TCP should be, not how much to move.
This is simpler to learn and avoids compounding delta errors.
The controller receives the absolute target via MODE_POSITION.

rot6d
─────
We use the 6D rotation representation (first two columns of the rotation
matrix) instead of quaternions. This avoids quaternion discontinuities
and is easier for neural networks to regress (Zhou et al., 2019).
"""

from __future__ import annotations

import numpy as np
from aic_control_interfaces.msg import MotionUpdate, TrajectoryGenerationMode
from geometry_msgs.msg import Pose, Quaternion, Vector3, Wrench
from scipy.spatial.transform import Rotation
from std_msgs.msg import Header

from zss_collection.config import ControllerParams

# ── Dimensions ────────────────────────────────────────────────

DIM = 9

POS_SLICE = slice(0, 3)  # x, y, z
ROT6D_SLICE = slice(3, 9)  # r00, r10, r20, r01, r11, r21

NAMES: tuple[str, ...] = (
    "position.x",
    "position.y",
    "position.z",
    "rot6d.r00",
    "rot6d.r10",
    "rot6d.r20",
    "rot6d.r01",
    "rot6d.r11",
    "rot6d.r21",
)
assert len(NAMES) == DIM

_BASE_FRAME = "base_link"


# ── Conversion helpers ────────────────────────────────────────


def quat_to_rot6d(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    """Quaternion (x,y,z,w) → rot6d (6,)."""
    R = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
    return np.array([*R[:, 0], *R[:, 1]], dtype=np.float32)


def rot6d_to_quat(rot6d: np.ndarray) -> np.ndarray:
    """
    rot6d (6,) → quaternion (x,y,z,w).

    Re-orthogonalises the two columns via Gram-Schmidt,
    then recovers the third column via cross product.
    """
    col0 = rot6d[:3].astype(np.float64)
    col1 = rot6d[3:6].astype(np.float64)

    # Gram-Schmidt: make col0 unit, col1 orthogonal to col0 and unit
    col0 = col0 / (np.linalg.norm(col0) + 1e-9)
    col1 = col1 - np.dot(col1, col0) * col0
    col1 = col1 / (np.linalg.norm(col1) + 1e-9)

    # Third column from cross product
    col2 = np.cross(col0, col1)

    R = np.stack([col0, col1, col2], axis=1)  # (3, 3)
    quat = Rotation.from_matrix(R).as_quat()  # (x, y, z, w)
    return quat


# ── Builder ───────────────────────────────────────────────────


class ActionBuilder:
    """
    Converts between ROS poses and the 9-dim absolute action vector.
    All methods are static.
    """

    @staticmethod
    def from_pose(pose: Pose) -> np.ndarray:
        """
        Extract 9-dim action from a target Pose (base_link frame).

        Args:
            pose: Target TCP pose in base_link frame.

        Returns:
            (9,) float32: [position(3), rot6d(6)]
        """
        pos = np.array([pose.position.x, pose.position.y, pose.position.z], dtype=np.float32)
        rot6d = quat_to_rot6d(
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
            pose.orientation.w,
        )
        return np.concatenate([pos, rot6d])

    @staticmethod
    def apply(
        action_vec: np.ndarray,
        controller_params: ControllerParams,
        stamp=None,
    ) -> MotionUpdate:
        """
        Convert 9-dim absolute action into a MotionUpdate message.

        Args:
            action_vec: (9,) — absolute target pose (pos3 + rot6d6).
            controller_params: Fixed impedance parameters.
            stamp: ROS timestamp (optional).

        Returns:
            MotionUpdate ready to publish on /aic_controller/pose_commands.
        """
        pos = action_vec[POS_SLICE].astype(np.float64)
        quat = rot6d_to_quat(action_vec[ROT6D_SLICE])

        target_pose = Pose()
        target_pose.position.x = float(pos[0])
        target_pose.position.y = float(pos[1])
        target_pose.position.z = float(pos[2])
        target_pose.orientation = Quaternion(
            x=float(quat[0]),
            y=float(quat[1]),
            z=float(quat[2]),
            w=float(quat[3]),
        )

        return ActionBuilder.pose_to_motion_update(target_pose, controller_params, stamp)

    @staticmethod
    def pose_to_motion_update(
        target_pose: Pose,
        controller_params: ControllerParams,
        stamp=None,
    ) -> MotionUpdate:
        """
        Wrap an absolute Pose into a MotionUpdate (MODE_POSITION, base_link).

        Used by the collection node to send commands directly.
        """
        msg = MotionUpdate()
        msg.header = Header(frame_id=_BASE_FRAME)
        if stamp is not None:
            msg.header.stamp = stamp

        msg.pose = target_pose

        msg.target_stiffness = list(np.diag(controller_params.stiffness).flatten())
        msg.target_damping = list(np.diag(controller_params.damping).flatten())

        ff = controller_params.feedforward_wrench
        msg.feedforward_wrench_at_tip = Wrench()
        msg.feedforward_wrench_at_tip.force = Vector3(x=ff[0], y=ff[1], z=ff[2])

        msg.wrench_feedback_gains_at_tip = list(controller_params.wrench_feedback_gains)

        msg.trajectory_generation_mode.mode = TrajectoryGenerationMode.MODE_POSITION

        return msg

    @staticmethod
    def zero_action() -> np.ndarray:
        """
        Identity action: origin position + identity rotation.

        Note: this is NOT a useful default — it's just a placeholder.
        In practice, initialise from the current TCP pose using from_pose().
        """
        action = np.zeros(DIM, dtype=np.float32)
        # rot6d of identity matrix: col0=[1,0,0], col1=[0,1,0]
        action[3] = 1.0  # r00
        action[7] = 1.0  # r11
        return action

    @staticmethod
    def lerobot_feature_spec() -> dict:
        """Feature dict for LeRobotDataset.create() action key."""
        return {
            "dtype": "float32",
            "shape": (DIM,),
            "names": {"action": list(NAMES)},
        }
