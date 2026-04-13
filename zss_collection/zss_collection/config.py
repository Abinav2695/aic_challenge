"""
Central configuration for the zss_collection data collection pipeline.

Three layers of config, matching three scopes of concern:

  ControllerParams   — impedance control: Kp, Kd, feedforward, gains.
                       Fixed for the entire training run. Never vary between episodes.

  InsertionConfig    — motion behaviour: phase durations, speeds, thresholds.
                       Tuned once per task type; rarely changed.

  RecordingConfig    — dataset format: obs/action dims, image settings, FPS.
                       Must stay consistent across all episodes in a dataset.

  SessionConfig      — CLI-level session args: batch_id, dataset_root, device.
                       Set once when launching the collection node.

  TrialConfig        — per-trial identity: scene_id, target port, cable name.
                       Populated at runtime from /data_collection/scene_metadata.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# ══════════════════════════════════════════════════════════════════════════════
# Controller parameters
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class ControllerParams:
    """
    Impedance control parameters sent with every MotionUpdate.

    Using a single fixed set throughout all phases removes a confounding
    variable from training data: the model always sees the same relationship
    between commanded pose and measured contact force.

    Kp (stiffness) units: N/m (translational), Nm/rad (rotational)
    Kd (damping)   units: N·s/m (translational), Nm·s/rad (rotational)

    At 150 N/m:  1mm pose error → 0.15 N contact force
                 5mm pose error → 0.75 N contact force
    This is moderate-to-soft, safe for both free-space approach and insertion.
    """

    # Diagonal of the 6×6 stiffness matrix [x, y, z, rx, ry, rz]
    stiffness: tuple[float, ...] = (150.0, 150.0, 150.0, 60.0, 60.0, 60.0)

    # Diagonal of the 6×6 damping matrix [x, y, z, rx, ry, rz]
    # Kd ≈ 2*sqrt(Kp * m_eff) for critical damping; slightly overdamped here.
    damping: tuple[float, ...] = (70.0, 70.0, 70.0, 30.0, 30.0, 30.0)

    # Constant feedforward wrench at TCP [fx, fy, fz] in N.
    # Zero: the policy must learn to push via pose commands — not hidden by ff.
    feedforward_wrench: tuple[float, float, float] = (0.0, 0.0, 0.0)

    # Force feedback gains [gx, gy, gz, grx, gry, grz], range [0, 0.95].
    # Zero: controller does not react to measured force — all force response
    # must come from the policy's pose decisions, keeping cause/effect clean.
    wrench_feedback_gains: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


# ══════════════════════════════════════════════════════════════════════════════
# Insertion motion config
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class InsertionConfig:
    """
    Phase durations, speeds, and thresholds for the waypoint insertion
    state machine.

    Phase sequence (auto mode):
      PRE_APPROACH → APPROACH → CORRECT → DESCEND → HOLD → DONE

    PRE_APPROACH: move to an overview height above the board centre so that
      all NIC mounts (0-4) are visible in the wrist cameras. This lets the
      policy learn to visually identify the target board — critical because
      task-board TF is unavailable during evaluation.

    APPROACH: ramp TCP from current pose toward target port at
      approach_z_offset height, interpolating over approach_steps ticks.

    CORRECT: hold near-port position, running the XY integrator until the
      translational error drops below converge_threshold.

    DESCEND: step down in Z by descend_step_m each tick until the
      /scoring/insertion_event fires or force_limit is exceeded.

    HOLD: maintain insertion pose for hold_ticks then save and exit.
    """

    # ── PRE_APPROACH ──────────────────────────────────────────────────────────
    # Extra height above approach_z_offset: overview pose = port_z
    # + approach_z_offset + overview_extra_z.
    # At ~0.38m total above board, all NIC mounts fit in the 90° wrist FOV.
    overview_extra_z: float = 0.18  # m above approach height
    overview_hold_ticks: int = 30  # frames at overview (1.5 s at 20 Hz)
    overview_ramp_ticks: int = 40  # ticks to ramp up to overview height (2 s)

    # ── APPROACH ──────────────────────────────────────────────────────────────
    approach_z_offset: float = 0.20  # m above port when approach ends
    approach_steps: int = 100  # ticks for approach ramp (4 s at 20 Hz)

    # ── CORRECT ───────────────────────────────────────────────────────────────
    converge_threshold: float = 0.02  # m — XY error magnitude to call "converged"
    converge_stable_count: int = 10  # consecutive ticks below threshold
    converge_max_ticks: int = 200  # bail-out: max ticks in CORRECT (10 s)

    # ── DESCEND ───────────────────────────────────────────────────────────────
    descend_step_m: float = 0.0008  # m/tick → 1.6 cm/s at 20 Hz
    insert_force_limit_n: float = 40.0  # N compensated force — pause descent above this

    # ── HOLD ──────────────────────────────────────────────────────────────────
    hold_ticks: int = 100  # 5 s at 20 Hz

    # ── XY integrator (GripperPoseCalculator) ─────────────────────────────────
    i_gain: float = 0.15
    max_integrator_windup: float = 0.05  # m — clamps ix, iy

    # ── Episode limits ─────────────────────────────────────────────────────────
    tick_rate_s: float = 0.05  # timer period → 20 Hz
    max_episode_ticks: int = 3600  # 3 min hard timeout


# ══════════════════════════════════════════════════════════════════════════════
# Recording / dataset config
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class RecordingConfig:
    """
    LeRobot dataset format and feature dimensions.

    Action space is 7-dim delta pose in the TCP frame:
      [Δx, Δy, Δz, Δqw, Δqx, Δqy, Δqz]

    Recording a delta (not an absolute pose) makes the action representation
    board-position-invariant: the same insertion motion at two different board
    locations produces nearly identical action sequences. This improves model
    generalisation and lets manual and auto demonstrations live in the same
    dataset without preprocessing.

    At inference time, send the predicted delta as MODE_POSITION in the
    gripper/tcp frame — the controller interprets it as an incremental move.

    Observation space is 32-dim from the time-synchronised sensor triple:
      [tcp_pose(7), tcp_vel(6), tcp_error(6), joint_pos(7), force(3), torque(3)]
    """

    repo_id: str = "local/aic_insertion_dataset"
    fps: int = 20  # must match 1 / InsertionConfig.tick_rate_s
    robot_type: str = "ur5e_aic"

    # Feature dimensions — changing these invalidates an existing dataset.
    obs_dim: int = 24
    action_dim: int = 9  # absolute pose (pos3 + rot6d6)

    # Camera recording (on by default; disable with --ros-args -p record_images:=false)
    record_images: bool = True
    image_scale: float = 0.25  # 1152×1024 → 288×256
    image_width: int = 1152  # native camera resolution
    image_height: int = 1024


# ══════════════════════════════════════════════════════════════════════════════
# Session config  (set once at node launch via CLI)
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class SessionConfig:
    """
    Session-level settings passed as CLI arguments when launching the
    collection node. Fixed for the entire recording session.

    Separate from TrialConfig so the node can run multiple episodes in a
    loop without restarting: batch_id and dataset_root never change, while
    scene_id / target_module / port_name change every trial.
    """

    # Where episodes are stored on disk.
    # Each batch lands at: <dataset_root>/<batch_id>/
    dataset_root: str = str(Path.home() / "datasets")
    batch_id: str = ""  # e.g. "batch_001" — must match orchestrator

    # "auto"   → waypoint controller drives the robot
    # "manual" → teleop device drives the robot
    control_mode: str = "auto"

    # Teleop device for manual mode.
    # "keyboard"    → pynput keyboard (AICKeyboardEETeleop)
    # "spacemouse"  → 3DConnexion SpaceMouse (AICSpaceMouseTeleop)
    teleop_device: str = "keyboard"

    # Pass --with-images to enable camera recording (slow, large files)
    record_images: bool = False

    @property
    def local_root(self) -> str:
        """Resolved dataset path: <dataset_root>/<batch_id>."""
        if not self.batch_id:
            raise ValueError("batch_id must be set before resolving local_root")
        return str(Path(self.dataset_root).expanduser() / self.batch_id)


# ══════════════════════════════════════════════════════════════════════════════
# Trial config  (populated per-episode from /data_collection/scene_metadata)
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class TrialConfig:
    """
    Per-trial identity populated from the latched /data_collection/scene_metadata
    topic that the orchestrator publishes before each scene.

    The collection node waits for this message at the start of each episode.
    The orchestrator (running in the aic-eval docker) is the only writer;
    the collection node is the only reader.

    Kept separate from SessionConfig because these values change every trial
    while SessionConfig fields are constant for the whole session.
    """

    scene_id: str = ""
    trial_id: str = ""
    batch_id: str = ""  # echoed back for cross-check against SessionConfig

    # Target insertion point
    target_module: str = ""  # e.g. "nic_card_mount_2"
    port_name: str = ""  # e.g. "sfp_port_0"

    # Cable attached to gripper
    cable_name: str = ""  # e.g. "cable_0"
    plug_name: str = ""  # e.g. "sfp_tip"
    plug_type: str = ""  # e.g. "sfp"

    # Natural-language task string — used as the LeRobot task label per frame
    task: str = ""

    # Board pose published by orchestrator (used for PRE_APPROACH centre point)
    # Keys: "x", "y", "z", "yaw"
    board_pose: dict = field(default_factory=dict)

    # If set, the recorder will overwrite this episode index instead of appending.
    # The orchestrator sets this when it wants a specific episode to be re-recorded.
    # None → append a new episode (default behaviour).
    overwrite_index: int | None = None

    # ── Derived TF frame names ─────────────────────────────────────────────

    @property
    def port_frame(self) -> str:
        """TF frame of the target port. Available only with ground_truth:=true."""
        return f"task_board/{self.target_module}/{self.port_name}_link"

    @property
    def cable_tip_frame(self) -> str:
        """TF frame of the cable plug tip. Available only with ground_truth:=true."""
        return f"{self.cable_name}/{self.plug_name}_link"

    @property
    def is_complete(self) -> bool:
        """True once all mandatory fields have been populated from metadata."""
        return bool(
            self.scene_id
            and self.target_module
            and self.port_name
            and self.cable_name
            and self.plug_name
        )
