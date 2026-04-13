"""
Abstract base class for collection controllers.

A controller owns the motion logic for one episode: it decides what target TCP
pose to command each tick given the current sensor snapshot. The node is
responsible for ROS wiring (subscribers, publishers, synchroniser, recorder) —
the controller has no ROS dependency and can be unit-tested without a running
ROS graph.

Swapping controllers
────────────────────
The collection node accepts a `controller` ROS parameter:

    ros2 run zss_collection collection_node --ros-args -p controller:=waypoint
    ros2 run zss_collection collection_node --ros-args -p controller:=teleop

Any new controller just needs to subclass BaseController and be registered in
the REGISTRY dict at the bottom of this file.

Contract
────────
  reset()  — called once per episode before the first tick.
  tick()   — called every control cycle; returns the absolute target TCP pose
             in base_link frame. The node converts it to a delta action with
             ActionBuilder.from_poses() then sends it via ActionBuilder.apply().
  is_done  — True once the episode should end (node calls save_episode then
             resets for the next trial).
  phase    — human-readable phase string for logging and dataset annotations.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum

import numpy as np
from geometry_msgs.msg import Pose

# ── Phase labels ──────────────────────────────────────────────────────────────


class Phase(str, Enum):
    """
    Canonical episode phase labels, shared by all controllers.

    Using str-Enum so they serialise directly as strings in LeRobot task
    annotations without any extra conversion.
    """

    PRE_APPROACH = "pre_approach"
    APPROACH = "approach"
    CORRECT = "correct"
    DESCEND = "descend"
    HOLD = "hold"
    DONE = "done"
    # Manual teleop controllers use a single TELEOP phase.
    TELEOP = "teleop"


# ── Abstract base ─────────────────────────────────────────────────────────────


class BaseController(ABC):
    """
    Interface that the collection node calls each control tick.

    Subclasses implement the motion strategy (waypoint automation, teleop, …).
    All geometry is in base_link frame. No ROS imports required.
    """

    @abstractmethod
    def reset(
        self,
        board_center_xyz: np.ndarray,
        port_pose: Pose,
        initial_tcp_pose: Pose,
    ) -> None:
        """
        Initialise (or re-initialise) controller state for a new episode.

        Called by the node once per trial, before the first tick.

        Parameters
        ----------
        board_center_xyz : np.ndarray, shape (3,)
            XYZ of the task-board centre in base_link frame.
            Derived from TrialConfig.board_pose by the node.
        port_pose : Pose
            Target NIC port pose in base_link frame, resolved from TF by the
            node (requires ground_truth:=true during collection).
        initial_tcp_pose : Pose
            Current TCP pose at the moment of reset, used as the ramp start.
        """

    @abstractmethod
    def tick(self, snapshot) -> Pose:
        """
        Advance one control step and return the target TCP pose.

        Parameters
        ----------
        snapshot : SensorSnapshot
            Time-synchronised sensor triple for the current tick.

        Returns
        -------
        geometry_msgs/Pose
            Absolute target TCP pose in base_link frame. The node feeds this
            into ActionBuilder.from_poses(current, target) to get the delta
            action vector.
        """

    @property
    @abstractmethod
    def is_done(self) -> bool:
        """True once the episode should end and the node should save + reset."""

    @property
    @abstractmethod
    def phase(self) -> Phase:
        """Current episode phase — logged and written to the dataset per frame."""

    def signal_insertion(self) -> None:  # noqa: B027
        """
        Optional hook called by the node when the scoring insertion_event fires.

        Default is a no-op. Override in controllers that need to react
        (e.g. WaypointController transitions DESCEND → HOLD on the next tick).
        """
