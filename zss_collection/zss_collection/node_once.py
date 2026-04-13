"""
Single-episode collection node using WaypointController.

Runs exactly ONE episode then exits. Mirrors data_collection.py flow:
  wait for metadata → TF lookup → APPROACH → CORRECT → DESCEND → HOLD → save → exit

For batch collection, wrap in a loop in the terminal:
    while true; do ros2 run zss_collection collection_node_once --ros-args ...; done

The orch works unchanged: it publishes scene_metadata, waits for "done" on
episode_event, then cleans up Gazebo for the next trial.

Parameters
──────────
  dataset_root   (str)   — where LeRobot datasets are stored  (default: ~/datasets)
  batch_id       (str)   — sub-folder for this session
  record_images  (bool)  — record wrist camera frames  (default: false)
  use_sim_time   (bool)  — match Gazebo sim clock
"""

from __future__ import annotations

import json
import logging
import math
import sys
from pathlib import Path

import numpy as np
import rclpy
from aic_control_interfaces.msg import (
    ControllerState,
    MotionUpdate,
    TargetMode,
    TrajectoryGenerationMode,
)
from aic_control_interfaces.srv import ChangeTargetMode
from geometry_msgs.msg import Pose, Vector3, Wrench, WrenchStamped
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from rclpy.time import Time
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Header, String
from std_srvs.srv import Trigger
from tf2_ros import Buffer, TransformException, TransformListener

from zss_collection.config import (
    ControllerParams,
    InsertionConfig,
    RecordingConfig,
    SessionConfig,
    TrialConfig,
)
from zss_collection.controllers.waypoint import (
    _WRENCH_FB_GAINS,
    WaypointController,
)
from zss_collection.metrics import EpisodeMetrics
from zss_collection.obs import ObservationBuilder, SensorSnapshot
from zss_collection.recorder import EpisodeRecorder

# Optional: ros_gz_interfaces only available inside the eval Docker container.
# Subscribed gracefully — if not importable, off_limit_contacts stays 0.
try:
    from ros_gz_interfaces.msg import Contacts as GzContacts

    _HAS_GZ_INTERFACES = True
except ImportError:
    GzContacts = None
    _HAS_GZ_INTERFACES = False

logger = logging.getLogger(__name__)


def _build_motion_update(
    target_pose: Pose,
    stiffness: list,
    damping: list,
    stamp,
    ff: tuple = (0.0, 0.0, 0.0),
) -> MotionUpdate:
    """Build a MODE_POSITION MotionUpdate from an absolute target pose in base_link."""
    msg = MotionUpdate()
    msg.header = Header(frame_id="base_link", stamp=stamp)
    msg.pose = target_pose
    msg.target_stiffness = list(np.diag(stiffness).flatten())
    msg.target_damping = list(np.diag(damping).flatten())
    msg.feedforward_wrench_at_tip = Wrench(
        force=Vector3(x=ff[0], y=ff[1], z=ff[2]),
    )
    msg.wrench_feedback_gains_at_tip = _WRENCH_FB_GAINS
    msg.trajectory_generation_mode.mode = TrajectoryGenerationMode.MODE_POSITION
    return msg


class CollectionNodeOnce(Node):
    """Single-episode collection node. Exits after one episode."""

    def __init__(self) -> None:
        super().__init__("collection_node_once")

        self.declare_parameter("dataset_root", str(Path.home() / "datasets"))
        self.declare_parameter("batch_id", "")
        self.declare_parameter("record_images", True)

        dataset_root = str(Path(self.get_parameter("dataset_root").value).expanduser())
        batch_id = self.get_parameter("batch_id").value
        record_images = self.get_parameter("record_images").value

        self.get_logger().info(
            f"CollectionNodeOnce: dataset_root={dataset_root} batch_id={batch_id}"
        )

        # ── Config ────────────────────────────────────────────────────────────
        self._insertion_cfg = InsertionConfig()
        self._ctrl_params = ControllerParams()
        self._recording_cfg = RecordingConfig(record_images=record_images)
        self._session_cfg = SessionConfig(
            dataset_root=dataset_root,
            batch_id=batch_id,
            record_images=record_images,
        )

        # ── Controller ────────────────────────────────────────────────────────
        self._controller = WaypointController(self._insertion_cfg)

        # ── Recorder ─────────────────────────────────────────────────────────
        self._recording_enabled = bool(batch_id)
        self._recorder: EpisodeRecorder | None = None
        if self._recording_enabled:
            self._recorder = EpisodeRecorder(self._session_cfg, self._recording_cfg)
            self._recorder.setup()

        # ── State ─────────────────────────────────────────────────────────────
        self._trial: TrialConfig | None = None
        self._collecting = False
        self._latest_ctrl_state: ControllerState | None = None
        self._latest_joint_states: JointState | None = None
        self._latest_wrench = None
        self._latest_images: dict = {}
        self._snapshot: SensorSnapshot | None = None

        # ── Task metadata for observation vector ─────────────────────────────
        self._connector_type: float = 0.0  # 0.0=sfp, 1.0=sc
        self._mount_id: float = 0.0  # NIC mount index or SC port rail
        self._port_index: float = 0.0  # port on that mount

        # ── Per-episode outcome tracking ──────────────────────────────────────
        self._metrics = EpisodeMetrics(tick_rate_s=self._insertion_cfg.tick_rate_s)
        self._max_episode_ticks: int = self._insertion_cfg.max_episode_ticks
        self._phase_ticks: dict[str, int] = {}
        self._last_logged_phase: str | None = None

        # ── TF ────────────────────────────────────────────────────────────────
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

        # ── Publishers ────────────────────────────────────────────────────────
        self._pose_pub = self.create_publisher(MotionUpdate, "/aic_controller/pose_commands", 10)
        _latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self._episode_event_pub = self.create_publisher(
            String, "/data_collection/episode_event", _latched
        )
        self._compensated_wrench_pub = self.create_publisher(
            WrenchStamped, "/data_collection/compensated_wrench", 10
        )

        # ── Services ─────────────────────────────────────────────────────────
        self._mode_client = self.create_client(
            ChangeTargetMode, "/aic_controller/change_target_mode"
        )
        self._tare_client = self.create_client(Trigger, "/aic_controller/tare_force_torque_sensor")

        self._services_init = False
        self.create_timer(1.0, self._init_services_once)

        # ── Subscriptions ─────────────────────────────────────────────────────
        self.create_subscription(
            ControllerState,
            "/aic_controller/controller_state",
            self._ctrl_state_cb,
            10,
        )
        self.create_subscription(
            JointState,
            "/joint_states",
            self._joint_states_cb,
            10,
        )
        self.create_subscription(
            self.__class__._wrench_type(),
            "/fts_broadcaster/wrench",
            self._wrench_cb,
            10,
        )
        self.create_subscription(
            String,
            "/data_collection/scene_metadata",
            self._meta_cb,
            QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL),
        )
        self.create_subscription(
            String,
            "/scoring/insertion_event",
            self._insertion_event_cb,
            10,
        )
        if _HAS_GZ_INTERFACES:
            self.create_subscription(
                GzContacts,
                "/aic/gazebo/contacts/off_limit",
                self._off_limit_contact_cb,
                10,
            )
            self.get_logger().info(
                "CollectionNodeOnce: ros_gz_interfaces available — "
                "subscribing to /aic/gazebo/contacts/off_limit"
            )
        if record_images:
            for cam in ("left_camera", "center_camera", "right_camera"):
                self.create_subscription(
                    Image,
                    f"/{cam}/image",
                    lambda msg, c=cam: self._image_cb(c, msg),
                    10,
                )

        # ── Control timer (20 Hz) ─────────────────────────────────────────────
        self._tick_timer = self.create_timer(self._insertion_cfg.tick_rate_s, self._tick_cb)

        self.get_logger().info("CollectionNodeOnce ready — waiting for scene_metadata...")

    @staticmethod
    def _wrench_type():
        from geometry_msgs.msg import WrenchStamped

        return WrenchStamped

    # ── Services init ─────────────────────────────────────────────────────────

    def _init_services_once(self) -> None:
        if self._services_init:
            return
        if not self._tare_client.service_is_ready() or not self._mode_client.service_is_ready():
            return
        self._tare_client.call_async(Trigger.Request()).add_done_callback(
            lambda f: self.get_logger().info(
                f"FTS tared (success={f.result().success})"
                if not f.exception()
                else f"FTS tare failed: {f.exception()}"
            )
        )
        req = ChangeTargetMode.Request()
        req.target_mode.mode = TargetMode.MODE_CARTESIAN
        self._mode_client.call_async(req).add_done_callback(
            lambda f: self.get_logger().info(
                f"Mode → Cartesian (success={f.result().success})"
                if not f.exception()
                else f"Mode switch failed: {f.exception()}"
            )
        )
        self._services_init = True

    # ── Sensor callbacks ──────────────────────────────────────────────────────

    def _ctrl_state_cb(self, msg: ControllerState) -> None:
        self._latest_ctrl_state = msg
        self._update_snapshot()

    def _joint_states_cb(self, msg: JointState) -> None:
        self._latest_joint_states = msg
        self._update_snapshot()

    def _wrench_cb(self, msg) -> None:
        self._latest_wrench = msg
        self._update_snapshot()

    def _update_snapshot(self) -> None:
        if (
            self._latest_ctrl_state is not None
            and self._latest_joint_states is not None
            and self._latest_wrench is not None
        ):
            self._snapshot = SensorSnapshot(
                controller_state=self._latest_ctrl_state,
                joint_states=self._latest_joint_states,
                wrench=self._latest_wrench,
            )

    def _image_cb(self, cam: str, msg: Image) -> None:
        self._latest_images[cam] = np.frombuffer(msg.data, dtype=np.uint8).reshape(
            msg.height, msg.width, -1
        )

    def _meta_cb(self, msg: String) -> None:
        if self._collecting or self._trial is not None:
            return  # already running
        try:
            meta = json.loads(msg.data)
        except Exception as exc:
            self.get_logger().warn(f"Bad scene_metadata JSON: {exc}")
            return

        trial = TrialConfig(
            scene_id=meta.get("scene_id", ""),
            trial_id=meta.get("trial_id", ""),
            batch_id=meta.get("batch_id", ""),
            target_module=meta.get("target_module", ""),
            port_name=meta.get("port_name", ""),
            cable_name=meta.get("cable_name", ""),
            plug_name=meta.get("plug_name", ""),
            plug_type=meta.get("plug_type", ""),
            task=meta.get("task", "Insert cable into port"),
            board_pose=meta.get("board_pose", {}),
            overwrite_index=None,  # auto-detected from sidecar by EpisodeRecorder
        )
        if not trial.is_complete:
            self.get_logger().warn(f"Incomplete metadata: {meta}")
            return

        self._trial = trial

        # ── Parse task metadata for observation vector ────────────────────────
        # connector_type: 0.0 = sfp, 1.0 = sc
        self._connector_type = 0.0 if trial.plug_type == "sfp" else 1.0
        # mount_id: extract index from target_module (e.g. "nic_card_mount_0" → 0)
        try:
            self._mount_id = float(trial.target_module.split("_")[-1])
        except (ValueError, IndexError):
            self._mount_id = 0.0
        # port_index: extract index from port_name (e.g. "sfp_port_1" → 1)
        try:
            self._port_index = float(trial.port_name.split("_")[-1])
        except (ValueError, IndexError):
            self._port_index = 0.0

        self.get_logger().info(
            f"Metadata received: {trial.scene_id} → {trial.target_module}/{trial.port_name} "
            f"(connector_type={self._connector_type}, mount_id={self._mount_id}, "
            f"port_index={self._port_index})"
        )

    def _insertion_event_cb(self, msg: String) -> None:
        if self._collecting:
            self.get_logger().info("Insertion event → signaling controller")
            self._metrics.mark_inserted()
            self._controller.signal_insertion()

    def _off_limit_contact_cb(self, msg) -> None:
        if self._collecting:
            self._metrics.mark_contact()
            self.get_logger().warn(
                f"CollectionNodeOnce: off-limit contact detected "
                f"(total={self._metrics.off_limit_contacts})"
            )

    # ── Control tick ──────────────────────────────────────────────────────────

    def _tick_cb(self) -> None:
        # Wait for metadata
        if self._trial is None:
            return

        # Wait for sensor snapshot
        if self._snapshot is None:
            return

        # Start episode on first tick after metadata arrives
        if not self._collecting:
            if not self._start_episode():
                return
            return  # let one tick pass after reset

        # Collecting
        self._collect_tick()

    _TF_MAX_AGE_S: float = 3.0

    def _start_episode(self) -> bool:
        """Look up TFs and reset controller. Returns True when ready."""
        trial = self._trial
        try:
            port_ts = self._tf_buffer.lookup_transform("base_link", trial.port_frame, Time())
            self._tf_buffer.lookup_transform("base_link", trial.cable_tip_frame, Time())
            self._tf_buffer.lookup_transform("base_link", "gripper/tcp", Time())
        except TransformException as exc:
            self.get_logger().warn(f"TF not yet available: {exc}")
            return False

        # Reject stale TF data that predates the current Gazebo scene.
        now_s = self.get_clock().now().nanoseconds * 1e-9
        port_stamp_s = port_ts.header.stamp.sec + port_ts.header.stamp.nanosec * 1e-9
        age_s = now_s - port_stamp_s
        if age_s > self._TF_MAX_AGE_S:
            self.get_logger().warn(
                f"TF is stale ({age_s:.1f}s old) — waiting for fresh transforms",
                throttle_duration_sec=5.0,
            )
            return False

        self._port_frame = trial.port_frame
        self._plug_frame = trial.cable_tip_frame
        self._grip_frame = "gripper/tcp"

        self._controller.reset()
        if self._recording_enabled and self._recorder is not None:
            self._recorder.begin_episode(trial)

        # Reset all per-episode metrics.
        self._metrics.reset()
        self._phase_ticks = {}
        self._last_logged_phase = None

        self._collecting = True
        self._episode_event_pub.publish(String(data="start"))
        self.get_logger().info(
            f"CollectionNodeOnce: COLLECTING — phase={self._controller.phase.value}"
        )
        return True

    def _get_tfs(self):
        """Return (port_tf, plug_tf, grip_tf) or None if any lookup fails."""
        try:
            port_tf = self._tf_buffer.lookup_transform(
                "base_link", self._port_frame, Time()
            ).transform
            plug_tf = self._tf_buffer.lookup_transform(
                "base_link", self._plug_frame, Time()
            ).transform
            grip_tf = self._tf_buffer.lookup_transform(
                "base_link", self._grip_frame, Time()
            ).transform
            return port_tf, plug_tf, grip_tf
        except TransformException as exc:
            self.get_logger().warn(f"TF lookup failed mid-episode: {exc}")
            return None

    def _collect_tick(self) -> None:
        tfs = self._get_tfs()
        if tfs is None:
            return
        port_tf, plug_tf, grip_tf = tfs

        snapshot = self._snapshot

        # ── Phase tracking & logging ──────────────────────────────────────────
        phase_key = self._controller.phase.value
        self._phase_ticks[phase_key] = self._phase_ticks.get(phase_key, 0) + 1

        if phase_key != self._last_logged_phase:
            self.get_logger().info(
                f"Phase → {phase_key.upper()}  "
                f"(tick={self._metrics.ticks}, t={self._metrics.elapsed_s:.1f}s)"
            )
            self._last_logged_phase = phase_key

        # ── Metrics update (tare-subtracted force, matching AIC eval) ─────────
        self._metrics.update(
            snapshot.controller_state.tcp_pose,
            snapshot.wrench,
            tare_offset=snapshot.controller_state.fts_tare_offset,
        )

        # ── Timeout check ─────────────────────────────────────────────────────
        if self._metrics.ticks >= self._max_episode_ticks:
            self.get_logger().warn(
                f"CollectionNodeOnce: episode TIMED OUT after {self._metrics.ticks} ticks "
                f"(phase={phase_key}) — ending episode"
            )
            self._finish_episode()
            return

        # 1. Advance controller — returns (pose, stiffness, damping, compensated_wrench)
        target_pose, stiffness, damping, compensated_wrench = self._controller.tick(
            snapshot, port_tf, plug_tf, grip_tf
        )

        # 2. Publish motion command directly (no delta round-trip)
        motion_update = _build_motion_update(
            target_pose,
            stiffness,
            damping,
            stamp=self.get_clock().now().to_msg(),
        )
        self._pose_pub.publish(motion_update)
        self._compensated_wrench_pub.publish(compensated_wrench)

        # 3. Record frame — using new 24-dim obs + 9-dim absolute action
        if self._recording_enabled and self._recorder is not None:
            from zss_collection.action import ActionBuilder

            # Action: absolute target pose as 9-dim (pos3 + rot6d6)
            action_vec = ActionBuilder.from_pose(target_pose)

            # Observation: 24-dim with task metadata
            obs_vec = ObservationBuilder.build(
                snapshot,
                connector_type=self._connector_type,
                mount_id=self._mount_id,
                port_index=self._port_index,
            )

            self._recorder.add_frame(
                obs_state=obs_vec,
                action_vec=action_vec,
                phase=self._controller.phase,
                images=self._latest_images.copy() if self._latest_images else None,
                task=self._trial.task if self._trial else None,
            )

        # 4. Episode complete?
        if self._controller.is_done:
            self._finish_episode()

    def _finish_episode(self) -> None:
        timed_out = self._metrics.ticks >= self._max_episode_ticks

        if timed_out:
            label = "timed_out"
        elif self._metrics.inserted:
            label = "success"
        else:
            label = "failed"

        # Try TF lookup for final plug-port distance.
        plug_port_dist = -1.0
        if self._collecting and hasattr(self, "_plug_frame") and hasattr(self, "_port_frame"):
            try:
                plug_tf = self._tf_buffer.lookup_transform(
                    self._port_frame, self._plug_frame, Time()
                ).transform.translation
                plug_port_dist = math.sqrt(plug_tf.x**2 + plug_tf.y**2 + plug_tf.z**2)
            except Exception:
                pass  # leave as -1.0

        outcome = self._metrics.build_outcome(
            label=label,
            final_phase=self._controller.phase.value,
            phase_ticks=self._phase_ticks,
            final_plug_port_dist_m=plug_port_dist,
        )

        if self._recording_enabled and self._recorder is not None:
            self._recorder.end_episode(outcome=outcome)
            self._recorder.finalize()

        self.get_logger().info(
            f"CollectionNodeOnce: episode ended — outcome={label} "
            f"inserted={outcome.inserted_successfully} "
            f"duration={outcome.duration_s:.1f}s ticks={outcome.total_ticks} "
            f"path={outcome.path_length_m:.3f}m plug_dist={outcome.final_plug_port_dist_m:.4f}m"
        )
        self._episode_event_pub.publish(String(data="done"))
        self.get_logger().info("CollectionNodeOnce: episode complete — shutting down.")
        self.create_timer(1.0, self._shutdown)

    def _shutdown(self) -> None:
        self.destroy_node()
        rclpy.shutdown()
        sys.exit(0)

    def destroy_node(self) -> None:
        if self._recording_enabled and self._recorder is not None:
            if self._recorder.is_episode_active:
                outcome = self._metrics.build_outcome(
                    label="shutdown",
                    final_phase=self._controller.phase.value,
                    phase_ticks=self._phase_ticks,
                )
                self._recorder.end_episode(outcome=outcome)
        super().destroy_node()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    rclpy.init()
    node = CollectionNodeOnce()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
