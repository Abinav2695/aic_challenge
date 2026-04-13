"""
Multi-episode collection node using WaypointController.

Loops continuously collecting episodes until the orchestrator sends "shutdown"
on /data_collection/orch_control.  The orchestrator publishes scene_metadata
for each episode; this node signals back on /data_collection/episode_event.

State machine
─────────────
  IDLE       Waiting for /data_collection/scene_metadata.
  RESETTING  TF lookup + controller.reset() each tick until TFs available.
  COLLECTING _collect_tick() at tick_rate_s; metrics tracked per episode.
  DONE       Episode finished (controller done or timeout); save and go IDLE.

Communication
─────────────
  Orch → Node  /data_collection/scene_metadata  latched  JSON per episode
  Node → Orch  /data_collection/episode_event   latched  "collecting"|"done"|"skip"
  Orch → Node  /data_collection/orch_control    latched  "shutdown"

On "shutdown": active episode is saved, recorder.finalize() called, node exits.

Parameters
──────────
  dataset_root   (str)   — where LeRobot datasets are stored  (default: ~/datasets)
  batch_id       (str)   — sub-folder for this session
  record_images  (bool)  — record wrist camera frames  (default: True)
  use_sim_time   (bool)  — match Gazebo sim clock
"""

from __future__ import annotations

import enum
import json
import logging
import math
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
    InsertionConfig,
    RecordingConfig,
    SessionConfig,
    TrialConfig,
)
from zss_collection.controllers.waypoint import _WRENCH_FB_GAINS, WaypointController
from zss_collection.metrics import EpisodeMetrics
from zss_collection.obs import ObservationBuilder, SensorSnapshot
from zss_collection.recorder import EpisodeRecorder

try:
    from ros_gz_interfaces.msg import Contacts as GzContacts

    _HAS_GZ_INTERFACES = True
except ImportError:
    GzContacts = None
    _HAS_GZ_INTERFACES = False

logger = logging.getLogger(__name__)


class _State(enum.Enum):
    IDLE = "idle"
    RESETTING = "resetting"
    COLLECTING = "collecting"
    DONE = "done"


def _build_motion_update(
    target_pose: Pose,
    stiffness: list,
    damping: list,
    stamp,
) -> MotionUpdate:
    """Build a MODE_POSITION MotionUpdate from an absolute target pose in base_link."""
    msg = MotionUpdate()
    msg.header = Header(frame_id="base_link", stamp=stamp)
    msg.pose = target_pose
    msg.target_stiffness = list(np.diag(stiffness).flatten())
    msg.target_damping = list(np.diag(damping).flatten())
    msg.feedforward_wrench_at_tip = Wrench(force=Vector3(x=0.0, y=0.0, z=0.0))
    msg.wrench_feedback_gains_at_tip = _WRENCH_FB_GAINS
    msg.trajectory_generation_mode.mode = TrajectoryGenerationMode.MODE_POSITION
    return msg


class CollectionNode(Node):
    """
    Multi-episode collection node.

    State machine: IDLE → RESETTING → COLLECTING → DONE → IDLE (loops).
    Exits cleanly when /data_collection/orch_control receives "shutdown".
    Keyboard [s] to skip an episode, [q] to quit immediately.
    """

    def __init__(self) -> None:
        super().__init__("collection_node")

        self.declare_parameter("dataset_root", str(Path.home() / "datasets"))
        self.declare_parameter("batch_id", "")
        self.declare_parameter("record_images", True)

        dataset_root = str(Path(self.get_parameter("dataset_root").value).expanduser())
        batch_id = self.get_parameter("batch_id").value
        record_images = self.get_parameter("record_images").value

        self.get_logger().info(f"CollectionNode: dataset_root={dataset_root} batch_id={batch_id}")

        # ── Config ────────────────────────────────────────────────────────────
        self._insertion_cfg = InsertionConfig()
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

        # ── State machine ─────────────────────────────────────────────────────
        self._state = _State.IDLE
        self._trial: TrialConfig | None = None

        # Sensor data
        self._latest_ctrl_state: ControllerState | None = None
        self._latest_joint_states: JointState | None = None
        self._latest_wrench = None
        self._latest_images: dict = {}
        self._snapshot: SensorSnapshot | None = None

        # Per-episode tracking
        self._metrics = EpisodeMetrics(tick_rate_s=self._insertion_cfg.tick_rate_s)
        self._max_episode_ticks: int = self._insertion_cfg.max_episode_ticks
        self._phase_ticks: dict[str, int] = {}
        self._last_logged_phase: str | None = None
        # Track last completed trial so we ignore the orch's 0.2 Hz metadata
        # republisher firing once more after we return to IDLE.
        self._last_completed_trial_id: str | None = None

        # TF frames (set in _try_start_episode)
        self._port_frame: str = ""
        self._plug_frame: str = ""
        self._grip_frame: str = "gripper/tcp"

        # Keyboard controls
        self._skip_requested: bool = False
        self._quit_requested: bool = False
        self._kb_listener = None
        try:
            from pynput import keyboard as _pynput_keyboard

            self._kb_listener = _pynput_keyboard.Listener(on_press=self._on_key_press)
            self._kb_listener.start()
            self.get_logger().info("CollectionNode: keyboard controls active — [s] skip  [q] quit")
        except Exception:
            self.get_logger().warn(
                "CollectionNode: pynput not available — keyboard controls disabled"
            )

        # ── TF ────────────────────────────────────────────────────────────────
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

        # ── QoS ───────────────────────────────────────────────────────────────
        _latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)

        # ── Publishers ────────────────────────────────────────────────────────
        self._pose_pub = self.create_publisher(MotionUpdate, "/aic_controller/pose_commands", 10)
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
        # Re-initialized after each episode (new Gazebo scene needs fresh tare)
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
            _latched,
        )
        self.create_subscription(
            String,
            "/scoring/insertion_event",
            self._insertion_event_cb,
            10,
        )
        self.create_subscription(
            String,
            "/data_collection/orch_control",
            self._orch_control_cb,
            _latched,
        )
        if _HAS_GZ_INTERFACES:
            self.create_subscription(
                GzContacts,
                "/aic/gazebo/contacts/off_limit",
                self._off_limit_contact_cb,
                10,
            )
            self.get_logger().info(
                "CollectionNode: ros_gz_interfaces available — "
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

        self.get_logger().info(
            "CollectionNode ready — waiting for scene_metadata "
            "(shutdown via /data_collection/orch_control  or  [q] key)"
        )

    def _reset_tf_buffer(self) -> None:
        """
        Recreate the TF buffer and listener to flush stale transforms from the
        previous Gazebo scene.  When sim time resets to 0, old transforms at
        large timestamps cause TF_OLD_DATA spam and block fresh lookups.
        """
        try:
            self._tf_listener.unregister()
        except Exception:
            pass
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)
        self.get_logger().info("CollectionNode: TF buffer reset for new scene")

    @staticmethod
    def _wrench_type():
        from geometry_msgs.msg import WrenchStamped

        return WrenchStamped

    # ── Service init (re-runs after each Gazebo scene restart) ────────────────

    def _init_services_once(self) -> None:
        """Tare FTS and switch to Cartesian mode. Re-runs after each episode."""
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

    # ── Keyboard handler ──────────────────────────────────────────────────────

    def _on_key_press(self, key) -> None:
        try:
            if not hasattr(key, "char") or key.char is None:
                return
            k = key.char.lower()
            if k == "s" and self._state in (_State.RESETTING, _State.COLLECTING):
                self.get_logger().warn(
                    "CollectionNode: [s] SKIP requested — discarding current episode"
                )
                self._skip_requested = True
            elif k == "q":
                self.get_logger().warn(
                    "CollectionNode: [q] QUIT requested — finalizing and shutting down"
                )
                self._quit_requested = True
        except Exception:
            pass

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

    # ── Control callbacks ─────────────────────────────────────────────────────

    def _meta_cb(self, msg: String) -> None:
        """Accept new scene metadata only when IDLE."""
        if self._state != _State.IDLE:
            return
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
            overwrite_index=None,
        )
        if not trial.is_complete:
            self.get_logger().warn(f"Incomplete metadata: {meta}")
            return

        # Ignore the orch's 0.2 Hz republish of the previous episode's metadata.
        # After an episode ends we return to IDLE; the republisher may fire once
        # more before the orch kills it.  The trial_id guard prevents re-starting
        # the same episode.
        if trial.trial_id and trial.trial_id == self._last_completed_trial_id:
            self.get_logger().debug(
                f"CollectionNode: ignoring already-completed trial_id={trial.trial_id}"
            )
            return

        self._trial = trial
        # Flush old TF data from the previous Gazebo scene.
        self._reset_tf_buffer()
        # Reset services so the 1s timer tares the NEW scene's controller.
        self._services_init = False
        self._state = _State.RESETTING

        # CRITICAL: overwrite the stale latched "done" from the previous episode.
        # The orch's _wait_for_episode_signal uses `ros2 topic echo` on the latched
        # episode_event topic.  Without this, the subscriber immediately receives
        # the old "done" and thinks the new episode is already complete.
        self._episode_event_pub.publish(String(data="resetting"))

        self.get_logger().info(
            f"CollectionNode: RESETTING — "
            f"scene={trial.scene_id} → {trial.target_module}/{trial.port_name}"
        )

    def _insertion_event_cb(self, msg: String) -> None:
        if self._state == _State.COLLECTING:
            self.get_logger().info("CollectionNode: insertion event → signaling controller")
            self._metrics.mark_inserted()
            self._controller.signal_insertion()

    def _off_limit_contact_cb(self, msg) -> None:
        if self._state == _State.COLLECTING:
            self._metrics.mark_contact()
            self.get_logger().warn(
                f"CollectionNode: off-limit contact (total={self._metrics.off_limit_contacts})"
            )

    def _orch_control_cb(self, msg: String) -> None:
        cmd = msg.data.strip().lower()
        if cmd == "shutdown":
            self.get_logger().info(
                "CollectionNode: received shutdown from orchestrator — finalizing and exiting"
            )
            self._do_shutdown()

    # ── Main tick ─────────────────────────────────────────────────────────────

    def _tick_cb(self) -> None:
        # Handle keyboard requests first
        if self._quit_requested:
            self._quit_requested = False
            self._do_shutdown()
            return
        if self._skip_requested:
            self._skip_requested = False
            self._do_skip()
            return

        if self._state == _State.IDLE:
            return

        if self._snapshot is None:
            return

        if self._state == _State.RESETTING:
            self._try_start_episode()
        elif self._state == _State.COLLECTING:
            self._collect_tick()
        elif self._state == _State.DONE:
            self._finish_episode()

    # ── RESETTING ─────────────────────────────────────────────────────────────

    # Maximum age of TF data to accept when starting an episode.
    # On Gazebo restart, sim time resets; old TF in the buffer can be from
    # the previous scene.  Reject if all three key frames aren't fresh.
    _TF_MAX_AGE_S: float = 3.0

    def _try_start_episode(self) -> None:
        """Look up TFs and reset controller. Retries each tick until successful."""
        trial = self._trial
        try:
            port_ts = self._tf_buffer.lookup_transform("base_link", trial.port_frame, Time())
            self._tf_buffer.lookup_transform("base_link", trial.cable_tip_frame, Time())
            self._tf_buffer.lookup_transform("base_link", "gripper/tcp", Time())
        except TransformException as exc:
            self.get_logger().warn(
                f"CollectionNode: TF not yet available: {exc}",
                throttle_duration_sec=5.0,
            )
            return

        # Reject stale TF data that predates the current Gazebo scene.
        # After a scene restart, sim time resets so old transforms are "from
        # the past".  Wait until port TF is fresh before starting the episode.
        now_s = self.get_clock().now().nanoseconds * 1e-9
        port_stamp_s = port_ts.header.stamp.sec + port_ts.header.stamp.nanosec * 1e-9
        age_s = now_s - port_stamp_s
        # abs(): negative age means TF is from the "future" relative to current
        # sim time — this happens after a sim-time reset (old buffer data at
        # t=200s, new sim at t=1s).  Both cases indicate stale data.
        if abs(age_s) > self._TF_MAX_AGE_S:
            self.get_logger().warn(
                f"CollectionNode: TF is stale (age={age_s:.1f}s) — "
                f"waiting for fresh transforms from new scene",
                throttle_duration_sec=5.0,
            )
            return

        self._port_frame = trial.port_frame
        self._plug_frame = trial.cable_tip_frame
        self._grip_frame = "gripper/tcp"

        self._controller.reset()

        if self._recording_enabled and self._recorder is not None:
            self._recorder.begin_episode(trial)

        self._metrics.reset()
        self._phase_ticks = {}
        self._last_logged_phase = None

        self._state = _State.COLLECTING
        self._episode_event_pub.publish(String(data="collecting"))
        self.get_logger().info(
            f"CollectionNode: COLLECTING — "
            f"scene={trial.scene_id} phase={self._controller.phase.value}"
        )

    # ── COLLECTING ────────────────────────────────────────────────────────────

    def _get_tfs(self):
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
            self.get_logger().warn(f"CollectionNode: TF lookup failed: {exc}")
            return None

    def _collect_tick(self) -> None:
        tfs = self._get_tfs()
        if tfs is None:
            return
        port_tf, plug_tf, grip_tf = tfs
        snapshot = self._snapshot

        # Phase tracking + logging
        phase_key = self._controller.phase.value
        self._phase_ticks[phase_key] = self._phase_ticks.get(phase_key, 0) + 1
        if phase_key != self._last_logged_phase:
            self.get_logger().info(
                f"Phase → {phase_key.upper()}  "
                f"(tick={self._metrics.ticks}, t={self._metrics.elapsed_s:.1f}s)"
            )
            self._last_logged_phase = phase_key

        # Metrics: tare-subtracted force matching AIC eval engine
        self._metrics.update(
            snapshot.controller_state.tcp_pose,
            snapshot.wrench,
            tare_offset=snapshot.controller_state.fts_tare_offset,
        )

        # Timeout check
        if self._metrics.ticks >= self._max_episode_ticks:
            self.get_logger().warn(
                f"CollectionNode: episode TIMED OUT after {self._metrics.ticks} ticks "
                f"(phase={phase_key})"
            )
            self._state = _State.DONE
            return

        # Advance controller
        target_pose, stiffness, damping, compensated_wrench = self._controller.tick(
            snapshot, port_tf, plug_tf, grip_tf
        )

        # Publish motion command
        self._pose_pub.publish(
            _build_motion_update(
                target_pose,
                stiffness,
                damping,
                stamp=self.get_clock().now().to_msg(),
            )
        )
        self._compensated_wrench_pub.publish(compensated_wrench)

        # Record frame
        if self._recording_enabled and self._recorder is not None:
            from zss_collection.action import ActionBuilder

            self._recorder.add_frame(
                obs_state=ObservationBuilder.build(snapshot),
                action_vec=ActionBuilder.from_poses(
                    snapshot.controller_state.tcp_pose, target_pose
                ),
                phase=self._controller.phase,
                images=self._latest_images.copy() if self._latest_images else None,
                task=self._trial.task if self._trial else None,
            )

        if self._controller.is_done:
            self._state = _State.DONE

    # ── DONE ─────────────────────────────────────────────────────────────────

    def _finish_episode(self) -> None:
        """Save episode, publish done, reset to IDLE for next trial."""
        timed_out = self._metrics.ticks >= self._max_episode_ticks
        if timed_out:
            label = "timed_out"
        elif self._metrics.inserted:
            label = "success"
        else:
            label = "failed"

        # Final plug-port distance
        plug_port_dist = -1.0
        try:
            t = self._tf_buffer.lookup_transform(
                self._port_frame, self._plug_frame, Time()
            ).transform.translation
            plug_port_dist = math.sqrt(t.x**2 + t.y**2 + t.z**2)
        except Exception:
            pass

        outcome = self._metrics.build_outcome(
            label=label,
            final_phase=self._controller.phase.value,
            phase_ticks=self._phase_ticks,
            final_plug_port_dist_m=plug_port_dist,
        )

        if self._recording_enabled and self._recorder is not None:
            # end_episode saves to disk; finalize() is deferred to shutdown
            self._recorder.end_episode(outcome=outcome)

        self.get_logger().info(
            f"CollectionNode: episode saved — outcome={label} "
            f"inserted={outcome.inserted_successfully} "
            f"duration={outcome.duration_s:.1f}s ticks={outcome.total_ticks} "
            f"path={outcome.path_length_m:.3f}m plug_dist={outcome.final_plug_port_dist_m:.4f}m"
        )

        # Signal orch data is saved
        self._episode_event_pub.publish(String(data="done"))

        # Record completed trial_id to filter duplicate metadata republishes
        if self._trial is not None:
            self._last_completed_trial_id = self._trial.trial_id

        # Do NOT reset _services_init here — the timer would immediately fire
        # on the old scene's controller (still alive while orch cleans up).
        # Instead we reset it in _meta_cb when the new scene's metadata arrives.
        self._trial = None
        self._state = _State.IDLE
        self.get_logger().info("CollectionNode: → IDLE, waiting for next scene_metadata")

    # ── Skip ──────────────────────────────────────────────────────────────────

    def _do_skip(self) -> None:
        """Discard current episode frames and return to IDLE."""
        if self._state not in (_State.RESETTING, _State.COLLECTING):
            return
        if self._recording_enabled and self._recorder is not None:
            if self._recorder.is_episode_active:
                self._recorder.discard_episode()
        self._episode_event_pub.publish(String(data="skip"))
        self._trial = None
        self._services_init = False
        self._state = _State.IDLE
        self.get_logger().info("CollectionNode: episode skipped → IDLE")

    # ── Shutdown ──────────────────────────────────────────────────────────────

    def _do_shutdown(self) -> None:
        """Save any active episode, finalize dataset, exit."""
        if self._state in (_State.RESETTING, _State.COLLECTING):
            if self._recording_enabled and self._recorder is not None:
                self.get_logger().info(
                    "CollectionNode: shutdown with active episode — saving partial"
                )
                outcome = self._metrics.build_outcome(
                    label="shutdown",
                    final_phase=self._controller.phase.value,
                    phase_ticks=self._phase_ticks,
                )
                self._recorder.end_episode(outcome=outcome)

        if self._recording_enabled and self._recorder is not None:
            self._recorder.finalize()

        if self._kb_listener is not None:
            try:
                self._kb_listener.stop()
            except Exception:
                pass

        self.get_logger().info("CollectionNode: dataset finalized — shutting down")
        rclpy.shutdown()

    def destroy_node(self) -> None:
        """Called on SIGINT/unclean exit — save partial episode if active."""
        if self._kb_listener is not None:
            try:
                self._kb_listener.stop()
            except Exception:
                pass
        if (
            self._recording_enabled
            and self._recorder is not None
            and self._recorder.is_episode_active
        ):
            outcome = self._metrics.build_outcome(
                label="shutdown",
                final_phase=self._controller.phase.value,
                phase_ticks=self._phase_ticks,
            )
            self._recorder.end_episode(outcome=outcome)
            self._recorder.finalize()
        super().destroy_node()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    rclpy.init()
    node = CollectionNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()


if __name__ == "__main__":
    main()
