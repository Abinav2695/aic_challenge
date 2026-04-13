############################################
# This file is part of the ZSS Data project.
# Copyright (c) 2024 ZSS Data Contributors
# License: MIT License
############################################

"""
Generate random but valid engine configs for training data collection.

Usage:
    # Generate 50 SFP+SC trials
    python -m zss_data.randomize --num-trials 50 --plug-types sfp sc --output configs/batch_001/

    # Generate 20 SFP-only trials
    python -m zss_data.randomize --num-trials 20 --plug-types sfp --output configs/sfp_only/

    # Generate 1 trial and print launch command for viewing
    python -m zss_data.randomize --num-trials 1 --plug-types sfp --output configs/test/ --print-launch-cmd

    # Control how many NIC cards and mount fixtures appear
    python -m zss_data.randomize --num-trials 50 --plug-types sfp sc --min-nic 1 --max-nic 3 --output configs/batch_002/

    # ── Simple mode (for initial policy validation) ──
    # 200 trials: 100 x sfp_port_0 + 100 x sfp_port_1, all on nic_card_mount_0
    python -m zss_data.scene_randomize --simple --batch-id batch_simple_001 --seed 42
"""

import argparse
import json
import math
import random
from datetime import datetime
from pathlib import Path

import yaml

from zss_data.aic_schema import (
    AICConfig,
    CableConfiguration,
    CablePose,
    EntityPose,
    GripperOffset,
    RailEntity,
    RobotConfig,
    SceneConfig,
    TaskBoardConfig,
    TaskBoardLimits,
    TaskBoardPose,
    TaskConfig,
    TrialConfig,
)

# ══════════════════════════════════════════════════════════════
# Default ranges — derived from sample_config.yaml and docs
# ══════════════════════════════════════════════════════════════
# In scene_randomize.py — tighter ranges that match sample config behavior

# Board pose — stay close to sample config values
BOARD_X_RANGE = (0.14, 0.18)  # was (0.13, 0.20)
BOARD_Y_RANGE = (-0.22, -0.15)  # was (-0.25, 0.05)
BOARD_YAW_RANGE = (2.9, 3.1)  # was (2.8, 3.5)

# Tight board pose ranges — minimal translational variation (±3 mm, ±0.03 rad)
# Centered at midpoints of the default ranges
TIGHT_BOARD_X_RANGE = (0.157, 0.163)  # ±3 mm around 0.160
TIGHT_BOARD_Y_RANGE = (-0.188, -0.182)  # ±3 mm around -0.185
TIGHT_BOARD_YAW_RANGE = (2.97, 3.03)  # ±0.03 rad around 3.00

# No gripper perturbation
GRIPPER_Z_DEVIATION = 0.0
GRIPPER_ORIENT_DEVIATION = 0.0


# Task board pose ranges (from sample config trials)
# BOARD_X_RANGE = (0.13, 0.20)
# BOARD_Y_RANGE = (-0.25, 0.05)
# BOARD_YAW_RANGE = (2.8, 3.5)  # sample: 3.0 to 3.1415
BOARD_Z_DEFAULT = 1.14  # fixed in all sample trials


# Gripper offset nominal values
SFP_GRIPPER_Z_NOMINAL = 0.04245
SC_GRIPPER_Z_NOMINAL = 0.04045
# GRIPPER_Z_DEVIATION = 0  # ±2mm
# GRIPPER_ORIENT_DEVIATION = 0  # ±0.04 rad

GRIPPER_ROLL_NOMINAL = 0.4432
GRIPPER_PITCH_NOMINAL = -0.4838
GRIPPER_YAW_NOMINAL = 1.3303
GRIPPER_Y_NOMINAL = 0.015385

# NIC card yaw range: ±10 degrees in radians
NIC_YAW_RANGE = (-math.radians(10), math.radians(10))

# Mount fixture yaw range: ±60 degrees in radians (from docs)
MOUNT_YAW_RANGE = (-math.radians(60), math.radians(60))

# Default scoring topics (copied from sample_config.yaml)
DEFAULT_SCORING = {
    "topics": [
        {"topic": {"name": "/joint_states", "type": "sensor_msgs/msg/JointState"}},
        {"topic": {"name": "/tf", "type": "tf2_msgs/msg/TFMessage"}},
        {
            "topic": {
                "name": "/tf_static",
                "type": "tf2_msgs/msg/TFMessage",
                "latched": True,
            }
        },
        {"topic": {"name": "/scoring/tf", "type": "tf2_msgs/msg/TFMessage"}},
        {
            "topic": {
                "name": "/aic/gazebo/contacts/off_limit",
                "type": "ros_gz_interfaces/msg/Contacts",
            }
        },
        {
            "topic": {
                "name": "/fts_broadcaster/wrench",
                "type": "geometry_msgs/msg/WrenchStamped",
            }
        },
        {
            "topic": {
                "name": "/aic_controller/joint_commands",
                "type": "aic_control_interfaces/msg/JointMotionUpdate",
            }
        },
        {
            "topic": {
                "name": "/aic_controller/pose_commands",
                "type": "aic_control_interfaces/msg/MotionUpdate",
            }
        },
        {"topic": {"name": "/scoring/insertion_event", "type": "std_msgs/msg/String"}},
        {
            "topic": {
                "name": "/aic_controller/controller_state",
                "type": "aic_control_interfaces/msg/ControllerState",
            }
        },
    ]
}

# Default robot home position
DEFAULT_HOME_JOINTS = {
    "shoulder_pan_joint": -0.1597,
    "shoulder_lift_joint": -1.3542,
    "elbow_joint": -1.6648,
    "wrist_1_joint": -1.6933,
    "wrist_2_joint": 1.5710,
    "wrist_3_joint": 1.4110,
}

NIC_TARGET_MODULES: dict[int, str] = {
    0: "nic_card_mount_0",
    1: "nic_card_mount_1",
    2: "nic_card_mount_2",
    3: "nic_card_mount_3",
    4: "nic_card_mount_4",
}

SC_TARGET_MODULES: dict[int, str] = {
    0: "sc_port_0",
    1: "sc_port_1",
}

# ══════════════════════════════════════════════════════════════
# Helper functions
# ══════════════════════════════════════════════════════════════


def _rand_range(low: float, high: float) -> float:
    """Random float in [low, high]."""
    return random.uniform(low, high)


def _perturb(nominal: float, deviation: float) -> float:
    """Add small random perturbation to a nominal value."""
    return nominal + random.uniform(-deviation, deviation)


def _random_board_pose(
    x_range: tuple[float, float] = BOARD_X_RANGE,
    y_range: tuple[float, float] = BOARD_Y_RANGE,
    yaw_range: tuple[float, float] = BOARD_YAW_RANGE,
) -> TaskBoardPose:
    """Generate a random task board pose within valid ranges."""
    return TaskBoardPose(
        x=round(_rand_range(*x_range), 4),
        y=round(_rand_range(*y_range), 4),
        z=BOARD_Z_DEFAULT,
        roll=0.0,
        pitch=0.0,
        yaw=round(_rand_range(*yaw_range), 4),
    )


def _random_gripper_offset(plug_type: str) -> GripperOffset:
    """Generate a gripper offset with small perturbation."""
    nominal_z = SFP_GRIPPER_Z_NOMINAL if plug_type == "sfp" else SC_GRIPPER_Z_NOMINAL
    return GripperOffset(
        x=0.0,
        y=GRIPPER_Y_NOMINAL,
        z=round(_perturb(nominal_z, GRIPPER_Z_DEVIATION), 5),
    )


def _random_cable_pose(plug_type: str) -> CablePose:
    """Generate a cable grasp pose with small perturbation."""
    return CablePose(
        gripper_offset=_random_gripper_offset(plug_type),
        roll=round(_perturb(GRIPPER_ROLL_NOMINAL, GRIPPER_ORIENT_DEVIATION), 4),
        pitch=round(_perturb(GRIPPER_PITCH_NOMINAL, GRIPPER_ORIENT_DEVIATION), 4),
        yaw=round(_perturb(GRIPPER_YAW_NOMINAL, GRIPPER_ORIENT_DEVIATION), 4),
    )


# ══════════════════════════════════════════════════════════════
# Trial generators
# ══════════════════════════════════════════════════════════════


def generate_sfp_trial(
    trial_id: str,
    limits: TaskBoardLimits,
    num_nic_cards: int = 1,
    num_mount_fixtures: int = 3,
    num_sc_ports: int | None = None,
    board_x_range: tuple[float, float] = BOARD_X_RANGE,
    board_y_range: tuple[float, float] = BOARD_Y_RANGE,
    board_yaw_range: tuple[float, float] = BOARD_YAW_RANGE,
) -> tuple[TrialConfig, dict]:
    """
    Generate one SFP insertion trial.

    Robot inserts SFP end of cable into a NIC card's SFP port.
    Cable type: sfp_sc_cable (SFP end is grasped).

    Args:
        num_sc_ports: Exact number of SC ports to place (0-2). If None, each
                      port is placed independently with 30% probability.
    Returns:
        (TrialConfig, metadata_dict)
    """
    # Pick which NIC rail to use as the target (0-4)
    target_nic_rail = random.randint(0, 4)

    # Randomize NIC card positions on rails
    # We always place the target NIC card, plus optionally others
    nic_rails_to_populate = {target_nic_rail}
    available_nic_rails = [i for i in range(5) if i != target_nic_rail]
    extra_nic_count = min(num_nic_cards - 1, len(available_nic_rails))
    if extra_nic_count > 0:
        nic_rails_to_populate.update(random.sample(available_nic_rails, extra_nic_count))

    # Build task board config
    tb = TaskBoardConfig(pose=_random_board_pose(board_x_range, board_y_range, board_yaw_range))

    # Populate NIC rails
    for i in range(5):
        rail_entity = RailEntity(entity_present=False)
        if i in nic_rails_to_populate:
            translation = round(
                _rand_range(
                    limits.nic_rail.min_translation,
                    limits.nic_rail.max_translation,
                ),
                4,
            )
            yaw = round(_rand_range(*NIC_YAW_RANGE), 4)
            rail_entity = RailEntity(
                entity_present=True,
                entity_name=f"nic_card_{i}",
                entity_pose=EntityPose(translation=translation, yaw=yaw),
            )
        setattr(tb, f"nic_rail_{i}", rail_entity)

    # Populate SC rails — explicit count or 30% per-port probability
    if num_sc_ports is not None:
        sc_slots = random.sample(range(2), min(num_sc_ports, 2))
    else:
        sc_slots = [i for i in range(2) if random.random() < 0.3]

    for i in range(2):
        if i in sc_slots:
            translation = round(
                _rand_range(
                    limits.sc_rail.min_translation,
                    limits.sc_rail.max_translation,
                ),
                4,
            )
            setattr(
                tb,
                f"sc_rail_{i}",
                RailEntity(
                    entity_present=True,
                    entity_name=f"sc_mount_{i}",
                    entity_pose=EntityPose(translation=translation),
                ),
            )

    # Populate mount rails (pick zone fixtures)
    mount_rails = [
        "lc_mount_rail_0",
        "lc_mount_rail_1",
        "sfp_mount_rail_0",
        "sfp_mount_rail_1",
        "sc_mount_rail_0",
        "sc_mount_rail_1",
    ]
    mount_names = [
        "lc_mount_0",
        "lc_mount_1",
        "sfp_mount_0",
        "sfp_mount_1",
        "sc_mount_0",
        "sc_mount_1",
    ]

    # Randomly select which mount fixtures to populate
    num_to_place = min(num_mount_fixtures, len(mount_rails))
    selected_indices = random.sample(range(len(mount_rails)), num_to_place)

    for idx in selected_indices:
        translation = round(
            _rand_range(
                limits.mount_rail.min_translation,
                limits.mount_rail.max_translation,
            ),
            4,
        )
        yaw = round(_rand_range(*MOUNT_YAW_RANGE), 4)
        setattr(
            tb,
            mount_rails[idx],
            RailEntity(
                entity_present=True,
                entity_name=mount_names[idx],
                entity_pose=EntityPose(translation=translation, yaw=yaw),
            ),
        )

    # Cable config
    cable_name = "cable_0"
    cable = CableConfiguration(
        pose=_random_cable_pose("sfp"),
        attach_cable_to_gripper=True,
        cable_type="sfp_sc_cable",
    )

    # Scene
    scene = SceneConfig(
        task_board=tb,
        cables={cable_name: cable},
    )

    # Task — insert SFP into the target NIC card
    task = TaskConfig(
        cable_type="sfp_sc",
        cable_name=cable_name,
        plug_type="sfp",
        plug_name="sfp_tip",
        port_type="sfp",
        port_name="sfp_port_0",
        target_module_name=f"nic_card_mount_{target_nic_rail}",
        time_limit=180,
    )

    trial = TrialConfig(
        scene=scene,
        tasks={"task_1": task},
    )

    # Metadata for this trial
    target_rail_entity = getattr(tb, f"nic_rail_{target_nic_rail}")
    metadata = {
        "trial_id": trial_id,
        "plug_type": "sfp",
        "cable_type": "sfp_sc_cable",
        "target_rail": f"nic_rail_{target_nic_rail}",
        "target_module": f"nic_card_mount_{target_nic_rail}",
        "target_port": "sfp_port_0",
        "board_pose": {
            "x": tb.pose.x,
            "y": tb.pose.y,
            "z": tb.pose.z,
            "yaw": tb.pose.yaw,
        },
        "nic_translation": target_rail_entity.entity_pose.translation,
        "nic_yaw": target_rail_entity.entity_pose.yaw,
        "num_nic_cards": len(nic_rails_to_populate),
        "gripper_z": cable.pose.gripper_offset.z,
    }

    return trial, metadata


def generate_sc_trial(
    trial_id: str,
    limits: TaskBoardLimits,
    num_nic_cards: int = 0,
    num_mount_fixtures: int = 3,
    board_x_range: tuple[float, float] = BOARD_X_RANGE,
    board_y_range: tuple[float, float] = BOARD_Y_RANGE,
    board_yaw_range: tuple[float, float] = BOARD_YAW_RANGE,
) -> tuple[TrialConfig, dict]:
    """
    Generate one SC insertion trial.

    Robot inserts SC end of cable into an SC port.
    Cable type: sfp_sc_cable_reversed (SC end is grasped).

    Returns:
        (TrialConfig, metadata_dict)
    """
    # Pick which SC rail to use as the target (0 or 1)
    target_sc_rail = random.randint(0, 1)

    # Build task board config
    tb = TaskBoardConfig(pose=_random_board_pose(board_x_range, board_y_range, board_yaw_range))
    for i in range(2):
        if i == target_sc_rail:
            translation = round(
                _rand_range(
                    limits.sc_rail.min_translation,
                    limits.sc_rail.max_translation,
                ),
                4,
            )
            setattr(
                tb,
                f"sc_rail_{i}",
                RailEntity(
                    entity_present=True,
                    entity_name=f"sc_mount_{i}",
                    entity_pose=EntityPose(translation=translation),
                ),
            )
        elif random.random() < 0.3:  # distractor
            translation = round(
                _rand_range(
                    limits.sc_rail.min_translation,
                    limits.sc_rail.max_translation,
                ),
                4,
            )
            setattr(
                tb,
                f"sc_rail_{i}",
                RailEntity(
                    entity_present=True,
                    entity_name=f"sc_mount_{i}",
                    entity_pose=EntityPose(translation=translation),
                ),
            )

    # Optionally populate NIC rails as distractors
    available_nic_rails = [i for i in range(5)]
    nic_cards_to_populate = random.sample(
        available_nic_rails, min(num_nic_cards, len(available_nic_rails))
    )
    print(
        f"\n\nTrial {trial_id}: Populating NIC rails {nic_cards_to_populate} as distractors for SC trial\n\n"
    )
    for i in range(5):
        if i in nic_cards_to_populate:
            translation = round(
                _rand_range(
                    limits.nic_rail.min_translation,
                    limits.nic_rail.max_translation,
                ),
                4,
            )
            yaw = round(_rand_range(*NIC_YAW_RANGE), 4)
            setattr(
                tb,
                f"nic_rail_{i}",
                RailEntity(
                    entity_present=True,
                    entity_name=f"nic_card_{i}",
                    entity_pose=EntityPose(translation=translation, yaw=yaw),
                ),
            )

    # Populate mount rails
    mount_rails = [
        "lc_mount_rail_0",
        "lc_mount_rail_1",
        "sfp_mount_rail_0",
        "sfp_mount_rail_1",
        "sc_mount_rail_0",
        "sc_mount_rail_1",
    ]
    mount_names = [
        "lc_mount_0",
        "lc_mount_1",
        "sfp_mount_0",
        "sfp_mount_1",
        "sc_mount_0",
        "sc_mount_1",
    ]

    num_to_place = min(num_mount_fixtures, len(mount_rails))
    selected_indices = random.sample(range(len(mount_rails)), num_to_place)

    for idx in selected_indices:
        translation = round(
            _rand_range(
                limits.mount_rail.min_translation,
                limits.mount_rail.max_translation,
            ),
            4,
        )
        yaw = round(_rand_range(*MOUNT_YAW_RANGE), 4)
        setattr(
            tb,
            mount_rails[idx],
            RailEntity(
                entity_present=True,
                entity_name=mount_names[idx],
                entity_pose=EntityPose(translation=translation, yaw=yaw),
            ),
        )

    # Cable config
    cable_name = "cable_0"
    cable = CableConfiguration(
        pose=_random_cable_pose("sc"),
        attach_cable_to_gripper=True,
        cable_type="sfp_sc_cable_reversed",
    )

    # Scene
    scene = SceneConfig(
        task_board=tb,
        cables={cable_name: cable},
    )

    # Task — insert SC into the target SC port
    task = TaskConfig(
        cable_type="sfp_sc",
        cable_name=cable_name,
        plug_type="sc",
        plug_name="sc_tip",
        port_type="sc",
        port_name="sc_port_base",
        target_module_name=f"sc_port_{target_sc_rail}",
        time_limit=180,
    )

    trial = TrialConfig(
        scene=scene,
        tasks={"task_1": task},
    )

    # Metadata
    target_rail_entity = getattr(tb, f"sc_rail_{target_sc_rail}")
    metadata = {
        "trial_id": trial_id,
        "plug_type": "sc",
        "cable_type": "sfp_sc_cable_reversed",
        "target_rail": f"sc_rail_{target_sc_rail}",
        "target_module": f"sc_port_{target_sc_rail}",
        "target_port": "sc_port_base",
        "board_pose": {
            "x": tb.pose.x,
            "y": tb.pose.y,
            "z": tb.pose.z,
            "yaw": tb.pose.yaw,
        },
        "sc_translation": target_rail_entity.entity_pose.translation,
        "num_sc_ports": sum(1 for i in range(2) if getattr(tb, f"sc_rail_{i}").entity_present),
        "gripper_z": cable.pose.gripper_offset.z,
    }

    return trial, metadata


# ══════════════════════════════════════════════════════════════
# Simple trial generator — for initial policy validation
# ══════════════════════════════════════════════════════════════
#
# Only nic_card_mount_0, SFP cable, two ports (sfp_port_0, sfp_port_1).
# Board pose and NIC translation/yaw are still randomised.
# No distractors (no extra NIC cards, no SC ports, no mount fixtures).
#
# Use --simple flag to activate. Once the policy works on this,
# scale up with the full generate_sfp_trial / generate_sc_trial above.
# ══════════════════════════════════════════════════════════════


def generate_simple_sfp_trial(
    trial_id: str,
    limits: TaskBoardLimits,
    target_nic_rail: int,
    target_port: str,
    board_x_range: tuple[float, float] = BOARD_X_RANGE,
    board_y_range: tuple[float, float] = BOARD_Y_RANGE,
    board_yaw_range: tuple[float, float] = BOARD_YAW_RANGE,
) -> tuple[TrialConfig, dict]:
    """
    Generate one simple SFP trial — fixed NIC mount, specified port.

    Only the target NIC card is placed. No distractors.
    Board pose and NIC translation/yaw are still randomised for domain randomization.

    Args:
        trial_id: e.g. "trial_0"
        limits: Task board rail limits.
        target_nic_rail: Which NIC rail (0–4).
        target_port: "sfp_port_0" or "sfp_port_1".
        board_x/y/yaw_range: Board pose randomization ranges.

    Returns:
        (TrialConfig, metadata_dict)
    """
    # Board with randomised pose
    tb = TaskBoardConfig(pose=_random_board_pose(board_x_range, board_y_range, board_yaw_range))

    # Place only the target NIC card — randomise its translation and yaw
    nic_translation = round(
        _rand_range(limits.nic_rail.min_translation, limits.nic_rail.max_translation),
        4,
    )
    nic_yaw = round(_rand_range(*NIC_YAW_RANGE), 4)

    setattr(
        tb,
        f"nic_rail_{target_nic_rail}",
        RailEntity(
            entity_present=True,
            entity_name=f"nic_card_{target_nic_rail}",
            entity_pose=EntityPose(translation=nic_translation, yaw=nic_yaw),
        ),
    )

    # Cable — SFP end grasped
    cable_name = "cable_0"
    cable = CableConfiguration(
        pose=_random_cable_pose("sfp"),
        attach_cable_to_gripper=True,
        cable_type="sfp_sc_cable",
    )

    # Scene — just the board + cable, nothing else
    scene = SceneConfig(
        task_board=tb,
        cables={cable_name: cable},
    )

    # Task — insert into the specified port
    task = TaskConfig(
        cable_type="sfp_sc",
        cable_name=cable_name,
        plug_type="sfp",
        plug_name="sfp_tip",
        port_type="sfp",
        port_name=target_port,
        target_module_name=f"nic_card_mount_{target_nic_rail}",
        time_limit=180,
    )

    trial = TrialConfig(
        scene=scene,
        tasks={"task_1": task},
    )

    metadata = {
        "trial_id": trial_id,
        "plug_type": "sfp",
        "cable_type": "sfp_sc_cable",
        "target_rail": f"nic_rail_{target_nic_rail}",
        "target_module": f"nic_card_mount_{target_nic_rail}",
        "target_port": target_port,
        "board_pose": {
            "x": tb.pose.x,
            "y": tb.pose.y,
            "z": tb.pose.z,
            "yaw": tb.pose.yaw,
        },
        "nic_translation": nic_translation,
        "nic_yaw": nic_yaw,
        "num_nic_cards": 1,
        "gripper_z": cable.pose.gripper_offset.z,
    }

    return trial, metadata


def generate_simple_batch(
    batch_id: str,
    target_nic_rail: int = 0,
    trials_per_port: int = 100,
    board_x_range: tuple[float, float] = BOARD_X_RANGE,
    board_y_range: tuple[float, float] = BOARD_Y_RANGE,
    board_yaw_range: tuple[float, float] = BOARD_YAW_RANGE,
    seed: int | None = None,
) -> tuple[AICConfig, dict]:
    """
    Generate a simple batch for initial policy validation.

    Creates trials_per_port trials for sfp_port_0 and trials_per_port for
    sfp_port_1, all targeting the same NIC mount. Trials are interleaved
    (port_0, port_1, port_0, port_1, ...) so the dataset is balanced even
    if collection is interrupted partway.

    Args:
        batch_id: Batch identifier string.
        target_nic_rail: Which NIC rail to use (default 0).
        trials_per_port: How many trials per port (default 100 → 200 total).
        board_x/y/yaw_range: Board pose randomization ranges.
        seed: Random seed for reproducibility.

    Returns:
        (AICConfig, metadata_dict)
    """
    if seed is not None:
        random.seed(seed)

    limits = TaskBoardLimits()
    trials = {}
    trial_metadata = {}

    ports = ["sfp_port_0", "sfp_port_1"]
    total = trials_per_port * len(ports)

    for i in range(total):
        trial_id = f"trial_{i}"
        # Interleave ports: even=port_0, odd=port_1
        port = ports[i % len(ports)]

        trial, meta = generate_simple_sfp_trial(
            trial_id=trial_id,
            limits=limits,
            target_nic_rail=target_nic_rail,
            target_port=port,
            board_x_range=board_x_range,
            board_y_range=board_y_range,
            board_yaw_range=board_yaw_range,
        )
        trials[trial_id] = trial
        trial_metadata[trial_id] = meta

    config = AICConfig(
        task_board_limits=limits,
        trials=trials,
        robot=RobotConfig(home_joint_positions=DEFAULT_HOME_JOINTS),
        scoring=DEFAULT_SCORING,
    )

    errors = config.validate_limits()
    if errors:
        raise ValueError("Generated config has invalid translations:\n" + "\n".join(errors))

    metadata = {
        "batch_id": batch_id,
        "generated_at": datetime.now().isoformat(),
        "num_trials": total,
        "plug_types_used": ["sfp"],
        "mode": "simple",
        "target_nic_rail": target_nic_rail,
        "trials_per_port": trials_per_port,
        "seed": seed,
        "ranges": {
            "board_x": list(board_x_range),
            "board_y": list(board_y_range),
            "board_z": BOARD_Z_DEFAULT,
            "board_yaw": list(board_yaw_range),
            "nic_rail": {
                "min": limits.nic_rail.min_translation,
                "max": limits.nic_rail.max_translation,
            },
        },
        "trials": trial_metadata,
    }

    return config, metadata


# ══════════════════════════════════════════════════════════════
# Main generator (original — full randomization)
# ══════════════════════════════════════════════════════════════


def generate_config(
    num_trials: int,
    plug_types: list[str],
    batch_id: str,
    min_nic: int = 1,
    max_nic: int = 2,
    min_mounts: int = 2,
    max_mounts: int = 4,
    min_sc_ports: int = 0,
    max_sc_ports: int = 2,
    board_x_range: tuple[float, float] = BOARD_X_RANGE,
    board_y_range: tuple[float, float] = BOARD_Y_RANGE,
    board_yaw_range: tuple[float, float] = BOARD_YAW_RANGE,
    seed: int | None = None,
) -> tuple[AICConfig, dict]:
    """
    Generate a complete AICConfig with random trials.

    Args:
        num_trials: Number of trials to generate
        plug_types: List of plug types to use ("sfp", "sc", or both)
        min_nic: Minimum NIC cards per SFP trial
        max_nic: Maximum NIC cards per SFP trial
        min_mounts: Minimum mount fixtures per trial
        max_mounts: Maximum mount fixtures per trial
        min_sc_ports: Minimum SC distractor ports per SFP trial (0-2)
        max_sc_ports: Maximum SC distractor ports per SFP trial (0-2)
        board_x_range: (min, max) for board X position
        board_y_range: (min, max) for board Y position
        board_yaw_range: (min, max) for board yaw angle
        seed: Random seed for reproducibility

    Returns:
        (AICConfig, metadata_dict)
    """
    if seed is not None:
        random.seed(seed)

    limits = TaskBoardLimits()
    trials = {}
    trial_metadata = {}

    for i in range(num_trials):
        trial_id = f"trial_{i}"
        plug = random.choice(plug_types)
        num_mounts = random.randint(min_mounts, max_mounts)
        num_nic = random.randint(min_nic, max_nic)
        num_sc = random.randint(min_sc_ports, max_sc_ports)

        if plug == "sfp":
            trial, meta = generate_sfp_trial(
                trial_id,
                limits,
                num_nic_cards=num_nic,
                num_mount_fixtures=num_mounts,
                num_sc_ports=num_sc,
                board_x_range=board_x_range,
                board_y_range=board_y_range,
                board_yaw_range=board_yaw_range,
            )
        else:
            trial, meta = generate_sc_trial(
                trial_id,
                limits,
                num_nic_cards=num_nic,
                num_mount_fixtures=num_mounts,
                board_x_range=board_x_range,
                board_y_range=board_y_range,
                board_yaw_range=board_yaw_range,
            )

        trials[trial_id] = trial
        trial_metadata[trial_id] = meta

    config = AICConfig(
        task_board_limits=limits,
        trials=trials,
        robot=RobotConfig(home_joint_positions=DEFAULT_HOME_JOINTS),
        scoring=DEFAULT_SCORING,
    )

    # Validate limits
    errors = config.validate_limits()
    if errors:
        raise ValueError("Generated config has invalid translations:\n" + "\n".join(errors))

    metadata = {
        "batch_id": batch_id,  # set by caller
        "generated_at": datetime.now().isoformat(),
        "num_trials": num_trials,
        "plug_types_used": plug_types,
        "seed": seed,
        "ranges": {
            "board_x": list(board_x_range),
            "board_y": list(board_y_range),
            "board_z": BOARD_Z_DEFAULT,
            "board_yaw": list(board_yaw_range),
            "nic_rail": {
                "min": limits.nic_rail.min_translation,
                "max": limits.nic_rail.max_translation,
            },
            "sc_rail": {
                "min": limits.sc_rail.min_translation,
                "max": limits.sc_rail.max_translation,
            },
            "mount_rail": {
                "min": limits.mount_rail.min_translation,
                "max": limits.mount_rail.max_translation,
            },
            "sc_ports_per_sfp_trial": [min_sc_ports, max_sc_ports],
        },
        "trials": trial_metadata,
    }

    return config, metadata


# ══════════════════════════════════════════════════════════════
# Launch command generator (for viewing without engine)
# ══════════════════════════════════════════════════════════════

# Mapping from YAML rail keys to launch parameter prefixes
# (confirmed from aic_engine.cpp source)
_NIC_RAIL_TO_LAUNCH = {f"nic_rail_{i}": f"nic_card_mount_{i}" for i in range(5)}
_SC_RAIL_TO_LAUNCH = {f"sc_rail_{i}": f"sc_port_{i}" for i in range(2)}
_MOUNT_RAIL_TO_LAUNCH = {
    "lc_mount_rail_0": "lc_mount_rail_0",
    "lc_mount_rail_1": "lc_mount_rail_1",
    "sfp_mount_rail_0": "sfp_mount_rail_0",
    "sfp_mount_rail_1": "sfp_mount_rail_1",
    "sc_mount_rail_0": "sc_mount_rail_0",
    "sc_mount_rail_1": "sc_mount_rail_1",
}


def trial_to_launch_args(trial: TrialConfig) -> str:
    """
    Convert a single TrialConfig into launch parameters for viewing
    the scene in Gazebo without the engine.

    Usage:
        /entrypoint.sh <output of this function>
    """
    tb = trial.scene.task_board
    args = [
        "spawn_task_board:=true",
        f"task_board_x:={tb.pose.x}",
        f"task_board_y:={tb.pose.y}",
        f"task_board_z:={tb.pose.z}",
        f"task_board_roll:={tb.pose.roll}",
        f"task_board_pitch:={tb.pose.pitch}",
        f"task_board_yaw:={tb.pose.yaw}",
    ]

    # NIC rails → nic_card_mount_N
    for yaml_key, launch_prefix in _NIC_RAIL_TO_LAUNCH.items():
        rail: RailEntity = getattr(tb, yaml_key)
        if rail.entity_present and rail.entity_pose:
            args.append(f"{launch_prefix}_present:=true")
            args.append(f"{launch_prefix}_translation:={rail.entity_pose.translation}")
            args.append(f"{launch_prefix}_roll:={rail.entity_pose.roll}")
            args.append(f"{launch_prefix}_pitch:={rail.entity_pose.pitch}")
            args.append(f"{launch_prefix}_yaw:={rail.entity_pose.yaw}")

    # SC rails → sc_port_N
    for yaml_key, launch_prefix in _SC_RAIL_TO_LAUNCH.items():
        rail: RailEntity = getattr(tb, yaml_key)
        if rail.entity_present and rail.entity_pose:
            args.append(f"{launch_prefix}_present:=true")
            args.append(f"{launch_prefix}_translation:={rail.entity_pose.translation}")
            args.append(f"{launch_prefix}_roll:={rail.entity_pose.roll}")
            args.append(f"{launch_prefix}_pitch:={rail.entity_pose.pitch}")
            args.append(f"{launch_prefix}_yaw:={rail.entity_pose.yaw}")

    # Mount rails — same key names
    for yaml_key, launch_prefix in _MOUNT_RAIL_TO_LAUNCH.items():
        rail: RailEntity = getattr(tb, yaml_key)
        if rail.entity_present and rail.entity_pose:
            args.append(f"{launch_prefix}_present:=true")
            args.append(f"{launch_prefix}_translation:={rail.entity_pose.translation}")
            args.append(f"{launch_prefix}_roll:={rail.entity_pose.roll}")
            args.append(f"{launch_prefix}_pitch:={rail.entity_pose.pitch}")
            args.append(f"{launch_prefix}_yaw:={rail.entity_pose.yaw}")

    # Cable
    if trial.scene.cables:
        cable_name, cable = next(iter(trial.scene.cables.items()))
        args.append("spawn_cable:=true")
        args.append(f"cable_type:={cable.cable_type}")
        attach = "true" if cable.attach_cable_to_gripper else "false"
        args.append(f"attach_cable_to_gripper:={attach}")

    # Ground truth and no engine
    args.append("ground_truth:=true")
    args.append("start_aic_engine:=false")
    args.append("launch_rviz:=false")
    args.append("gazebo_gui:=false")

    return " \\\n    ".join(args)


def extract_launch_cmds(config_path: Path) -> dict[str, str]:
    """
    Read LAUNCH_CMD comments from a config.yaml file.

    Usage:
        cmds = extract_launch_cmds(Path("configs/test/config.yaml"))
        print(cmds["trial_0"])  # full launch command

    Bash:
        grep "LAUNCH_CMD\\[trial_0\\]" config.yaml | sed 's/# LAUNCH_CMD\\[trial_0\\]: //'
    """
    cmds = {}
    current_trial = None
    current_lines = []

    with open(config_path) as f:
        for line in f:
            if line.startswith("# LAUNCH_CMD["):
                if current_trial and current_lines:
                    cmds[current_trial] = " ".join(current_lines)
                current_trial = line.split("[")[1].split("]")[0]
                cmd_part = line.split(": ", 1)[1].strip().rstrip("\\").strip()
                current_lines = [cmd_part]
            elif current_trial and line.startswith("#     "):
                cmd_part = line.lstrip("#").strip().rstrip("\\").strip()
                if cmd_part:
                    current_lines.append(cmd_part)
            elif current_trial and not line.startswith("#"):
                if current_trial and current_lines:
                    cmds[current_trial] = " ".join(current_lines)
                current_trial = None
                current_lines = []

    if current_trial and current_lines:
        cmds[current_trial] = " ".join(current_lines)

    return cmds


def _trial_to_maifest_entry(
    trial_id: str,
    trial: TrialConfig,
    metadata: dict,
    batch_id: str,
) -> dict:
    """
    Convert a trial config and metadata into a manifest entry for dataset indexing.

    This is a placeholder for future use when we want to generate manifest files
    that include both config parameters and metadata for each trial.
    """
    task = next(iter(trial.tasks.values()))
    cable_name = next(iter(trial.scene.cables.keys()))

    return {
        "trial_id": f"{int(trial_id.split('_')[-1]):03d}",  # make trial id trial_001 3 digits with leading zeros
        "launch_cmd": "/entrypoint.sh " + trial_to_launch_args(trial).replace("\\\n    ", " "),
        "target_module": task.target_module_name,
        "port_name": task.port_name,
        "plug_name": task.plug_name,
        "plug_type": task.plug_type,
        "port_type": task.port_type,
        "cable_name": cable_name,
        "cable_type": task.cable_type,
        "task_description": f"Insert {task.plug_type} cable into {task.target_module_name} {task.port_name}",
        "scene_id": f"{int(trial_id.split('_')[-1]):03d}",
        "batch_id": batch_id,
        "board_pose": metadata.get("board_pose", {}),
        "time_limit": task.time_limit,
    }


# ══════════════════════════════════════════════════════════════
# Save / Load
# ══════════════════════════════════════════════════════════════


def save_batch(
    config: AICConfig,
    metadata: dict,
    output_dir: Path,
    print_launch_cmds: bool = False,
) -> None:
    """
    Save generated config and metadata to output directory.

    Creates:
        output_dir/config.yaml      — engine-compatible config
        output_dir/metadata.json    — per-trial metadata
        output_dir/launch_cmds.sh   — (optional) launch commands for viewing
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set batch_id from directory name
    if "batch_id" not in metadata:
        print(" Error: batch_id not found in metadata")
        exit(1)
    # Save engine config
    config_path = output_dir / "config.yaml"
    with open(config_path, "w") as f:
        # Write the YAML config
        yaml.dump(
            config.model_dump(exclude_none=True),
            f,
            default_flow_style=False,
            sort_keys=False,
        )
    print(f"Saved engine config: {config_path}")

    # Save metadata
    meta_path = output_dir / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata: {meta_path}")

    manifest = []
    for trial_id, trial in config.trials.items():
        trial_meta = metadata["trials"].get(trial_id, {})
        manifest_entry = _trial_to_maifest_entry(trial_id, trial, trial_meta, metadata["batch_id"])
        manifest.append(manifest_entry)

    manifest_path = output_dir / "launch_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Saved launch manifest: {manifest_path}")

    # Generate launch commands
    if print_launch_cmds:
        launch_path = output_dir / "launch_cmds.sh"
        with open(launch_path, "w") as f:
            f.write("#!/bin/bash\n")
            f.write("# Auto-generated launch commands for viewing scenes in Gazebo\n")
            f.write("# Usage: copy one command and run inside eval container\n\n")
            for trial_id, trial in config.trials.items():
                f.write(f"# === {trial_id} ===\n")
                f.write("# /entrypoint.sh \\\n#     ")
                f.write(trial_to_launch_args(trial).replace("\n    ", "\n#     "))
                f.write("\n\n")
        print(f"Saved launch commands: {launch_path}")


# ══════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(
        description="Generate random AIC engine configs for training data collection.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--num-trials",
        type=int,
        default=50,
        help="Number of trials to generate (default: 50). In --simple mode, this is trials PER PORT.",
    )
    parser.add_argument(
        "--batch-id",
        type=str,
        required=True,
        help="Batch ID for this set of trials (used in metadata and manifest) eg batch_001 use 001, 002, etc for proper sorting",
    )
    parser.add_argument(
        "--plug-types",
        nargs="+",
        choices=["sfp", "sc"],
        default=["sfp", "sc"],
        help="Plug types to include (default: sfp sc)",
    )
    parser.add_argument(
        "--min-nic",
        type=int,
        default=1,
        help="Minimum NIC cards per SFP trial (default: 1)",
    )
    parser.add_argument(
        "--max-nic",
        type=int,
        default=2,
        help="Maximum NIC cards per SFP trial (default: 2)",
    )
    parser.add_argument(
        "--min-mounts",
        type=int,
        default=2,
        help="Minimum mount fixtures per trial (default: 2)",
    )
    parser.add_argument(
        "--max-mounts",
        type=int,
        default=4,
        help="Maximum mount fixtures per trial (default: 4)",
    )
    parser.add_argument(
        "--min-sc-ports",
        type=int,
        default=0,
        help="Minimum SC distractor ports per SFP trial (0-2, default: 0)",
    )
    parser.add_argument(
        "--max-sc-ports",
        type=int,
        default=2,
        help="Maximum SC distractor ports per SFP trial (0-2, default: 2)",
    )
    parser.add_argument(
        "--tight-board-pose",
        action="store_true",
        help=(
            "Use tight board pose ranges for minimal translational variation "
            f"(x: {TIGHT_BOARD_X_RANGE}, y: {TIGHT_BOARD_Y_RANGE}, yaw: {TIGHT_BOARD_YAW_RANGE})"
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--print-launch-cmd",
        action="store_true",
        help="Also generate launch_cmds.sh for viewing scenes in Gazebo",
    )

    # ── Simple mode flags ──
    parser.add_argument(
        "--simple",
        action="store_true",
        help=(
            "Simple mode: generate trials for a single NIC mount with two ports. "
            "Use --num-trials to set trials per port (default 100 → 200 total). "
            "Board pose and NIC translation/yaw are still randomised."
        ),
    )
    parser.add_argument(
        "--simple-nic-rail",
        type=int,
        default=0,
        choices=[0, 1, 2, 3, 4],
        help="Which NIC rail to target in simple mode (default: 0)",
    )

    args = parser.parse_args()

    if args.tight_board_pose:
        bx_range = TIGHT_BOARD_X_RANGE
        by_range = TIGHT_BOARD_Y_RANGE
        byaw_range = TIGHT_BOARD_YAW_RANGE
    else:
        bx_range = BOARD_X_RANGE
        by_range = BOARD_Y_RANGE
        byaw_range = BOARD_YAW_RANGE

    # ── Simple mode ──
    if args.simple:
        trials_per_port = args.num_trials if args.num_trials != 50 else 100
        config, metadata = generate_simple_batch(
            batch_id=args.batch_id,
            target_nic_rail=args.simple_nic_rail,
            trials_per_port=trials_per_port,
            board_x_range=bx_range,
            board_y_range=by_range,
            board_yaw_range=byaw_range,
            seed=args.seed,
        )

        pkg_path = Path(__file__).parent.parent.resolve() / "configs"
        save_batch(
            config=config,
            metadata=metadata,
            output_dir=pkg_path / f"{args.batch_id}",
            print_launch_cmds=args.print_launch_cmd,
        )

        total = trials_per_port * 2
        print(f"\n[Simple mode] Generated {total} trials:")
        print(f"  {trials_per_port} x sfp_port_0 on nic_card_mount_{args.simple_nic_rail}")
        print(f"  {trials_per_port} x sfp_port_1 on nic_card_mount_{args.simple_nic_rail}")
        print(f"  Board pose range: x={list(bx_range)}, y={list(by_range)}, yaw={list(byaw_range)}")
        return

    # ── Full mode (original) ──
    config, metadata = generate_config(
        num_trials=args.num_trials,
        plug_types=args.plug_types,
        min_nic=args.min_nic,
        max_nic=args.max_nic,
        min_mounts=args.min_mounts,
        max_mounts=args.max_mounts,
        min_sc_ports=args.min_sc_ports,
        max_sc_ports=args.max_sc_ports,
        board_x_range=bx_range,
        board_y_range=by_range,
        board_yaw_range=byaw_range,
        batch_id=args.batch_id,
        seed=args.seed,
    )

    # Get ros package path for saving configs (if needed for launch commands)
    pkg_path = Path(__file__).parent.parent.resolve() / "configs"
    print(f"ROS package path: {pkg_path}")

    save_batch(
        config=config,
        metadata=metadata,
        output_dir=pkg_path / f"{args.batch_id}",
        print_launch_cmds=args.print_launch_cmd,
    )

    # Print summary
    sfp_count = sum(1 for m in metadata["trials"].values() if m["plug_type"] == "sfp")
    sc_count = sum(1 for m in metadata["trials"].values() if m["plug_type"] == "sc")
    print(f"\nGenerated {args.num_trials} trials: {sfp_count} SFP, {sc_count} SC")

    if args.print_launch_cmd and args.num_trials > 0:
        first_trial = next(iter(config.trials.values()))
        print("\nExample launch command for trial_0:")
        print(f"/entrypoint.sh \\\n    {trial_to_launch_args(first_trial)}")


if __name__ == "__main__":
    main()
