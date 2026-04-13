"""
WaypointController — direct port of GripperPoseCalculator from data_collection.py.

Phase sequence (no PRE_APPROACH):
  APPROACH → CORRECT → DESCEND → HOLD → DONE

Key differences from WaypointController:
  - Live TF poses passed into tick() every cycle (port, plug, gripper/tcp)
  - XY integrator in CORRECT / DESCEND (same as data_collection.py)
  - Phase-specific stiffness / damping
  - Target orientation: q_port * inv(q_plug) * q_grip  (exact formula from data_collection.py)

Config used:
  InsertionConfig  — phase durations, speeds, thresholds (from config.py)
  Phase-specific stiffness/damping  — hardcoded to match data_collection.py defaults

No ROS imports — all geometry uses geometry_msgs types. The node passes live TF
transforms on every tick so the controller always uses up-to-date poses.
"""

from __future__ import annotations

import math

import numpy as np
from geometry_msgs.msg import Pose, Quaternion, Transform, WrenchStamped

from zss_collection.config import InsertionConfig
from zss_collection.controllers.base import Phase
from zss_collection.obs import ObservationBuilder, SensorSnapshot

# ── Quaternion helpers — wxyz convention (matches data_collection.py) ─────────


def _quat_inv(q: tuple) -> tuple:
    """Inverse of unit quaternion (w, x, y, z)."""
    return (q[0], -q[1], -q[2], -q[3])


def _quat_mul(q1: tuple, q2: tuple) -> tuple:
    """Multiply two quaternions (w, x, y, z)."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return (
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    )


def _quat_slerp(q0: tuple, q1: tuple, t: float) -> tuple:
    """SLERP between two unit quaternions (w, x, y, z)."""
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


def _tf_to_quat_wxyz(tf: Transform) -> tuple:
    """Extract quaternion (w, x, y, z) from a geometry_msgs/Transform."""
    return (tf.rotation.w, tf.rotation.x, tf.rotation.y, tf.rotation.z)


def _pose_from_xyz_wxyz(xyz: tuple, q_wxyz: tuple) -> Pose:
    """Build Pose from (x, y, z) and quaternion (w, x, y, z)."""
    pose = Pose()
    pose.position.x, pose.position.y, pose.position.z = xyz
    pose.orientation = Quaternion(
        w=q_wxyz[0],
        x=q_wxyz[1],
        y=q_wxyz[2],
        z=q_wxyz[3],
    )
    return pose


# ── Phase-specific impedance parameters (match data_collection.py defaults) ───

# [trans_x, trans_y, trans_z, rot_x, rot_y, rot_z]
_STIFFNESS = [180.0] * 3 + [180.0] * 3
_DAMPING = [80.0] * 3 + [40.0] * 3
_WRENCH_FB_GAINS = [0.5, 0.5, 0.5, 0, 0, 0]


# ── Controller ────────────────────────────────────────────────────────────────


class WaypointController:
    """
    Automatic insertion controller — exact port of GripperPoseCalculator.

    Unlike WaypointController, this class requires live TF transforms on every
    tick. The node calls tick(snapshot, port_tf, plug_tf, grip_tf) after
    looking up the three frames from the TF buffer.

    Phase sequence:  APPROACH → CORRECT → DESCEND → HOLD → DONE
    """

    def __init__(self, config: InsertionConfig) -> None:
        self._config = config
        self._phase = Phase.DONE  # sentinel before first reset()

        # Integrator state
        self._ix: float = 0.0
        self._iy: float = 0.0
        self._start_pos: tuple | None = None

        # Counters
        self._step: int = 0
        self._corr_ticks: int = 0
        self._corr_stable: int = 0
        self._hold_ticks: int = 0

        # DESCEND state
        self._z_off: float = 0.0
        self._insertion_done: bool = False

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def reset(self) -> None:
        """Start a new episode. Call before the first tick."""
        self._phase = Phase.APPROACH
        self._ix = 0.0
        self._iy = 0.0
        self._start_pos = None
        self._step = 0
        self._corr_ticks = 0
        self._corr_stable = 0
        self._hold_ticks = 0
        self._z_off = self._config.approach_z_offset
        self._insertion_done = False

    # ── Per-tick logic ────────────────────────────────────────────────────────

    def tick(
        self,
        snapshot: SensorSnapshot,
        port_tf: Transform,
        plug_tf: Transform,
        grip_tf: Transform,
    ) -> tuple[Pose, list, list, WrenchStamped]:
        """
        Advance one control step.

        Parameters
        ----------
        snapshot : SensorSnapshot
            Latest sensor readings (for tcp_error, force magnitude).
        port_tf, plug_tf, grip_tf : geometry_msgs/Transform
            Live TF transforms: target port, cable plug tip, gripper TCP
            — all in base_link frame.

        Returns
        -------
        (target_pose, stiffness, damping, force_mag)
            target_pose  — absolute Pose in base_link frame
            stiffness    — 6-element diagonal for target_stiffness matrix
            damping      — 6-element diagonal for target_damping matrix
            force_mag    — magnitude of the compensated force
        """
        tcp_err = snapshot.controller_state.tcp_error
        tcp_error_mag = float(np.linalg.norm(tcp_err[:3])) if tcp_err is not None else 999.0
        compensated_wrench = ObservationBuilder.compensate_wrench(snapshot)
        force_mag = ObservationBuilder.compensated_force_magnitude(compensated_wrench)

        if self._phase == Phase.APPROACH:
            pose = self._tick_approach(port_tf, plug_tf, grip_tf)
            return pose, _STIFFNESS, _DAMPING, compensated_wrench

        elif self._phase == Phase.CORRECT:
            pose = self._tick_correct(port_tf, plug_tf, grip_tf, tcp_error_mag)
            return pose, _STIFFNESS, _DAMPING, compensated_wrench

        elif self._phase == Phase.DESCEND:
            pose = self._tick_descend(port_tf, plug_tf, grip_tf, force_mag)
            return pose, _STIFFNESS, _DAMPING, compensated_wrench

        elif self._phase == Phase.HOLD:
            pose = self._tick_hold(port_tf, plug_tf, grip_tf)
            return pose, _STIFFNESS, _DAMPING, compensated_wrench

        # DONE — return last pose with insert gains
        pose = self._calculate(port_tf, plug_tf, grip_tf, 1.0, 1.0, self._z_off, False)
        return pose, _STIFFNESS, _DAMPING, compensated_wrench

    # ── Phase implementations ─────────────────────────────────────────────────

    def _tick_approach(self, port_tf, plug_tf, grip_tf):
        cfg = self._config
        self._step += 1
        frac = min(1.0, self._step / cfg.approach_steps)
        pose = self._calculate(port_tf, plug_tf, grip_tf, frac, frac, cfg.approach_z_offset, True)
        if frac >= 1.0:
            self._phase = Phase.CORRECT
            self._corr_ticks = 0
            self._corr_stable = 0
        return pose

    def _tick_correct(self, port_tf, plug_tf, grip_tf, tcp_error_mag):
        cfg = self._config
        self._corr_ticks += 1
        pose = self._calculate(port_tf, plug_tf, grip_tf, 1.0, 1.0, cfg.approach_z_offset, False)
        if tcp_error_mag < cfg.converge_threshold:
            self._corr_stable += 1
        else:
            self._corr_stable = 0
        if (
            self._corr_stable >= cfg.converge_stable_count
            or self._corr_ticks >= cfg.converge_max_ticks
        ):
            self._z_off = cfg.approach_z_offset
            self._phase = Phase.DESCEND
        return pose

    def _tick_descend(self, port_tf, plug_tf, grip_tf, force_mag):
        cfg = self._config
        if self._insertion_done:
            self._phase = Phase.HOLD
            self._hold_ticks = 0
            pose = self._calculate(port_tf, plug_tf, grip_tf, 1.0, 1.0, self._z_off, False)
            return pose

        self._z_off -= cfg.descend_step_m

        pose = self._calculate(port_tf, plug_tf, grip_tf, 1.0, 1.0, self._z_off, False)
        return pose

    def _tick_hold(self, port_tf, plug_tf, grip_tf):
        cfg = self._config
        self._hold_ticks += 1
        if self._hold_ticks >= cfg.hold_ticks:
            self._phase = Phase.DONE
        pose = self._calculate(port_tf, plug_tf, grip_tf, 1.0, 1.0, self._z_off, False)
        return pose

    # ── GripperPoseCalculator (exact port) ────────────────────────────────────

    def _calculate(
        self,
        port_tf: Transform,
        plug_tf: Transform,
        grip_tf: Transform,
        slerp_fraction: float,
        position_fraction: float,
        z_offset: float,
        reset_xy_integrator: bool,
    ) -> Pose:
        """
        Direct port of GripperPoseCalculator.calculate() from data_collection.py.

        Quaternion convention: (w, x, y, z) throughout this method.
        """
        cfg = self._config

        q_port = _tf_to_quat_wxyz(port_tf)
        q_plug = _tf_to_quat_wxyz(plug_tf)
        q_grip = _tf_to_quat_wxyz(grip_tf)

        # Target orientation: keep gripper-to-plug relative rotation, apply in port frame
        q_target = _quat_mul(q_port, _quat_mul(_quat_inv(q_plug), q_grip))
        q_slerp = _quat_slerp(q_grip, q_target, slerp_fraction)

        gx = grip_tf.translation.x
        gy = grip_tf.translation.y
        gz = grip_tf.translation.z

        if self._start_pos is None:
            self._start_pos = (gx, gy, gz)

        lx = plug_tf.translation.x
        ly = plug_tf.translation.y
        lz = plug_tf.translation.z

        px = port_tf.translation.x
        py = port_tf.translation.y
        pz = port_tf.translation.z

        ex, ey = px - lx, py - ly
        if reset_xy_integrator:
            self._ix, self._iy = 0.0, 0.0
        else:
            w = cfg.max_integrator_windup
            self._ix = float(np.clip(self._ix + ex, -w, w))
            self._iy = float(np.clip(self._iy + ey, -w, w))

        ig = cfg.i_gain
        tx = px + ig * self._ix + (gx - lx)
        ty = py + ig * self._iy + (gy - ly)
        tz = pz + z_offset + (gz - lz)

        sx, sy, sz = self._start_pos
        bx = position_fraction * tx + (1.0 - position_fraction) * sx
        by = position_fraction * ty + (1.0 - position_fraction) * sy
        bz = position_fraction * tz + (1.0 - position_fraction) * sz

        return _pose_from_xyz_wxyz((bx, by, bz), q_slerp)

    # ── External signals ──────────────────────────────────────────────────────

    def signal_insertion(self) -> None:
        """Called by the node when the scoring insertion_event fires."""
        self._insertion_done = True

    # ── Properties ───────────────────────────────────────────────────────────

    @property
    def is_done(self) -> bool:
        return self._phase == Phase.DONE

    @property
    def phase(self) -> Phase:
        return self._phase

    @property
    def wrench_feedback_gains(self) -> list:
        return _WRENCH_FB_GAINS
