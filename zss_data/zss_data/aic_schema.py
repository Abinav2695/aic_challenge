############################################
# This file is part of the ZSS Data project.
# Copyright (c) 2024 ZSS Data Contributors
# License: MIT License
############################################


from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field

CableType = Literal["sfp_sc_cable", "sfp_sc_cable_reversed"]
TaskCableType = Literal["sfp_sc"]
PlugType = Literal["sfp", "sc"]
PortType = Literal["sfp", "sc"]
PlugName = Literal["sfp_tip", "sc_tip"]
PortName = Literal["sfp_port_0", "sfp_port_1", "sc_port_base"]
# Entity names — based on sample config patterns
NICCardName = Literal[
    "nic_card_0",
    "nic_card_1",
    "nic_card_2",
    "nic_card_3",
    "nic_card_4",
]

SCMountName = Literal[
    "sc_mount_0",
    "sc_mount_1",
]

# Target module names (used in TaskConfig)
TargetModuleName = Literal[
    "nic_card_mount_0",
    "nic_card_mount_1",
    "nic_card_mount_2",
    "nic_card_mount_3",
    "nic_card_mount_4",
    "sc_port_0",
    "sc_port_1",
]
# Pick zone fixture names
LCMountName = Literal["lc_mount_0", "lc_mount_1"]
SFPMountName = Literal["sfp_mount_0", "sfp_mount_1"]
SCFixtureMountName = Literal["sc_mount_0", "sc_mount_1", "sc_mount_2"]


class TaskBoardPose(BaseModel):
    """
    TaskBoardPose represents the pose and orientation of a task board in the training scene.
    """

    x: float = Field(..., description="X coordinate of the task board")
    y: float = Field(..., description="Y coordinate of the task board")
    z: float = Field(..., description="Z coordinate of the task board")
    roll: float = Field(default=0.0, description="Roll angle of the task board")
    pitch: float = Field(default=0.0, description="Pitch angle of the task board")
    yaw: float = Field(default=0.0, description="Yaw angle of the task board")


class EntityPose(BaseModel):
    """
    EntityPose represents the translation and orientation of an entity in the training scene.
    """

    translation: float = Field(..., description="Translation of the entity")
    roll: float = Field(default=0.0, description="Roll angle of the entity")
    pitch: float = Field(default=0.0, description="Pitch angle of the entity")
    yaw: float = Field(default=0.0, description="Yaw angle of the entity")


class GripperOffset(BaseModel):
    x: float = Field(default=0.0, description="Nominal x offset")
    y: float = Field(default=0.015385, description="Nominal y offset")
    z: float = Field(..., description="0.04245 for SFP, 0.04045 for SC as Nominal, ±2mm")


# ═════════════════════════════════════
# Cable
# ═════════════════════════════════════


class CablePose(BaseModel):
    """Grasp pose — relative to gripper, nominal values with ~2mm / ~0.04rad deviations"""

    gripper_offset: GripperOffset
    roll: float = Field(0.4432, description="Nominal roll, ±0.04 rad")
    pitch: float = Field(-0.4838, description="Nominal pitch, ±0.04 rad")
    yaw: float = Field(1.3303, description="Nominal yaw, ±0.04 rad")


class CableConfiguration(BaseModel):
    """Configuration for a specific cable type"""

    pose: CablePose
    attach_cable_to_gripper: bool = Field(
        True, description="Whether to attach the cable to the gripper during the task"
    )
    cable_type: CableType = Field(..., description="Type of the cable (e.g., SFP or SC)")


# ══════════════════════════════════════
# Rail entity — one slot on the board
# ══════════════════════════════════════


class RailEntity(BaseModel):
    """A single rail slot — either empty or has a component"""

    entity_present: bool = Field(
        default=False, description="Whether an entity is present in the slot"
    )
    entity_name: str | None = Field(
        default=None, description="Name of the entity in the slot, if present"
    )
    entity_pose: EntityPose | None = Field(
        default=None, description="Pose of the entity in the slot, if present"
    )


# ═════════════════════════════════════
# Task Configuration
# ═════════════════════════════════════
class TaskConfig(BaseModel):
    """
    TaskConfig represents the configuration for a specific task in the scene.
    Example
    task_1:
        cable_type: "sfp_sc"
        cable_name: "cable_0"
        plug_type: "sfp"
        plug_name: "sfp_tip"
        port_type: "sfp"
        port_name: "sfp_port_0"
        target_module_name: "nic_card_mount_0"
        time_limit: 180
    """

    cable_type: TaskCableType = Field(..., description="Type of the cable used in the task")
    cable_name: str = Field(..., description="Name of the cable entity in the scene")
    plug_type: PlugType = Field(..., description="Type of the plug to be inserted")
    plug_name: PlugName = Field(..., description="Name of the plug entity in the scene")
    port_type: PortType = Field(..., description="Type of the port for insertion")
    port_name: PortName = Field(..., description="Name of the port entity in the scene")
    target_module_name: TargetModuleName = Field(
        ..., description="Name of the target module for the task"
    )
    time_limit: int = Field(180, description="Time limit for completing the task in seconds")


# ════════════════════════════════════
# Rail Limits
# ═════════════════════════════════════
class RailLimits(BaseModel):
    """
    RailLimits represents the limits for the rail entity in the scene.
    """

    min_translation: float = Field(..., description="Minimum translation limit for the rail")
    max_translation: float = Field(..., description="Maximum translation limit for the rail")


class TaskBoardLimits(BaseModel):
    # Ranges derived from official challenge docs (task_board_description.md + aic_bringup README):
    #   NIC rail  (Zone 1): [0, 0.062] m total  → ±0.031 m centered
    #   SC rail   (Zone 2): [0, 0.115] m total  → ±0.0575 m centered
    #   Mount rail(Zone3/4): ±0.09625 m (from aic_bringup README)
    nic_rail: RailLimits = RailLimits(min_translation=-0.031, max_translation=0.031)
    sc_rail: RailLimits = RailLimits(min_translation=-0.0575, max_translation=0.0575)
    mount_rail: RailLimits = RailLimits(min_translation=-0.09625, max_translation=0.09625)


class TaskBoardConfig(BaseModel):
    """Complete task board: pose + all rail slots"""

    pose: TaskBoardPose
    # NIC rails (5)
    nic_rail_0: RailEntity = RailEntity()
    nic_rail_1: RailEntity = RailEntity()
    nic_rail_2: RailEntity = RailEntity()
    nic_rail_3: RailEntity = RailEntity()
    nic_rail_4: RailEntity = RailEntity()
    # SC rails (2)
    sc_rail_0: RailEntity = RailEntity()
    sc_rail_1: RailEntity = RailEntity()
    # Mount rails — pick zones (6)
    lc_mount_rail_0: RailEntity = RailEntity()
    lc_mount_rail_1: RailEntity = RailEntity()
    sfp_mount_rail_0: RailEntity = RailEntity()
    sfp_mount_rail_1: RailEntity = RailEntity()
    sc_mount_rail_0: RailEntity = RailEntity()
    sc_mount_rail_1: RailEntity = RailEntity()


class SceneConfig(BaseModel):
    """Complete scene configuration, including task board and cable definitions"""

    task_board: TaskBoardConfig
    cables: dict[str, CableConfiguration]  # required


class TrialConfig(BaseModel):
    """Complete trial configuration, including scene and tasks"""

    scene: SceneConfig  # required
    tasks: dict[str, TaskConfig]  # required


# ══════════════════════════════════════
# Robot home position
# ══════════════════════════════════════


class RobotConfig(BaseModel):
    home_joint_positions: dict[str, float] = {
        "shoulder_pan_joint": -0.1597,
        "shoulder_lift_joint": -1.3542,
        "elbow_joint": -1.6648,
        "wrist_1_joint": -1.6933,
        "wrist_2_joint": 1.5710,
        "wrist_3_joint": 1.4110,
    }


class AICConfig(BaseModel):
    """Top-level configuration for the AIC task, including scene and robot settings"""

    task_board_limits: TaskBoardLimits
    trials: dict[str, TrialConfig]  # required
    robot: RobotConfig = RobotConfig()  # default robot config with home position
    scoring: dict | None = None  # Optional scoring parameters

    @classmethod
    def from_yaml(cls, yaml_path: Path) -> "AICConfig":
        """Load AICConfig from a YAML file"""
        with open(yaml_path) as f:
            data = yaml.safe_load(f)
        return cls(**data)

    def to_yaml(self, yaml_path: Path):
        """Save AICConfig to a YAML file"""
        with open(yaml_path, "w") as f:
            yaml.dump(
                self.model_dump(exclude_none=True),
                f,
                default_flow_style=False,
                sort_keys=False,
            )

    def validate_limits(self) -> list[str]:
        """Check all translations are within task_board_limits"""
        errors = []
        limits = self.task_board_limits
        for trial_name, trial in self.trials.items():
            tb = trial.scene.task_board
            # Check NIC rails
            for i in range(5):
                rail: RailEntity = getattr(tb, f"nic_rail_{i}")
                if rail.entity_present and rail.entity_pose:
                    t = rail.entity_pose.translation
                    if not (
                        limits.nic_rail.min_translation <= t <= limits.nic_rail.max_translation
                    ):
                        errors.append(
                            f"{trial_name}: nic_rail_{i} translation {t} "
                            f"outside [{limits.nic_rail.min_translation}, {limits.nic_rail.max_translation}]"
                        )
            # Check SC rails
            for i in range(2):
                rail: RailEntity = getattr(tb, f"sc_rail_{i}")
                if rail.entity_present and rail.entity_pose:
                    t = rail.entity_pose.translation
                    if not (limits.sc_rail.min_translation <= t <= limits.sc_rail.max_translation):
                        errors.append(
                            f"{trial_name}: sc_rail_{i} translation {t} "
                            f"outside [{limits.sc_rail.min_translation}, {limits.sc_rail.max_translation}]"
                        )
            # Check mount rails
            for prefix in ["lc_mount_rail", "sfp_mount_rail", "sc_mount_rail"]:
                for i in range(2):
                    rail: RailEntity = getattr(tb, f"{prefix}_{i}")
                    if rail.entity_present and rail.entity_pose:
                        t = rail.entity_pose.translation
                        if not (
                            limits.mount_rail.min_translation
                            <= t
                            <= limits.mount_rail.max_translation
                        ):
                            errors.append(
                                f"{trial_name}: {prefix}_{i} translation {t} "
                                f"outside [{limits.mount_rail.min_translation}, {limits.mount_rail.max_translation}]"
                            )
        return errors
