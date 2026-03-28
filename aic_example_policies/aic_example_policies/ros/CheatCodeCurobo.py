#
# CuRobo approach policy - uses GPU-accelerated motion planning
# for the approach phase only (no insertion).
# Drop-in replacement for CheatCode to verify cuRobo works in the AIC loop.
#

import numpy as np
import torch

from aic_model.policy import (
    GetObservationCallback,
    MoveRobotCallback,
    Policy,
    SendFeedbackCallback,
)
from aic_control_interfaces.msg import (
    JointMotionUpdate,
    TrajectoryGenerationMode,
)
from aic_model_interfaces.msg import Observation
from aic_task_interfaces.msg import Task
from geometry_msgs.msg import Point, Pose, Quaternion, Transform
from std_msgs.msg import Header
from trajectory_msgs.msg import JointTrajectoryPoint
from rclpy.duration import Duration
from rclpy.time import Time
from tf2_ros import TransformException
from transforms3d._gohlketransforms import quaternion_multiply

# cuRobo
from curobo.types.math import Pose as CuPose
from curobo.types.robot import JointState as CuJointState
from curobo.types.file_path import ContentPath
from curobo.util.xrdf_utils import convert_xrdf_to_curobo
from curobo.util_file import load_yaml
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig


# Paths — update these if your files move
XRDF_PATH = "/home/nitin/ws_aic/ur5e_aic_valid.xrdf"
URDF_PATH = "/home/nitin/ws_aic/ur5e_curobo_dummy.urdf"

JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]


class CheatCodeCurobo(Policy):

    def __init__(self, parent_node):
        self._task = None
        self._mg = None
        super().__init__(parent_node)
        self._init_curobo()

    # ------------------------------------------------------------------
    # cuRobo setup
    # ------------------------------------------------------------------

    def _init_curobo(self):
        self.get_logger().info("CuRobo: loading XRDF + URDF...")

        content_path = ContentPath(
            robot_xrdf_absolute_path=XRDF_PATH,
            robot_urdf_absolute_path=URDF_PATH,
        )
        xrdf_dict = load_yaml(XRDF_PATH)
        robot_cfg = convert_xrdf_to_curobo(content_path, xrdf_dict)

        # Fixes required after XRDF conversion
        robot_cfg["robot_cfg"]["kinematics"]["ee_link"] = "gripper_tcp"
        robot_cfg["robot_cfg"]["kinematics"]["base_link"] = "base_link"

        self.get_logger().info("CuRobo: building MotionGen config...")
        mg_config = MotionGenConfig.load_from_robot_config(
            robot_cfg, interpolation_dt=0.002,
        )

        self._mg = MotionGen(mg_config)
        self.get_logger().info("CuRobo: warming up (~15 s first time)...")
        self._mg.warmup()
        self.get_logger().info("CuRobo: ready.")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _wait_for_tf(
        self, target_frame: str, source_frame: str, timeout_sec: float = 10.0
    ) -> bool:
        start = self.time_now()
        timeout = Duration(seconds=timeout_sec)
        attempt = 0
        while (self.time_now() - start) < timeout:
            try:
                self._parent_node._tf_buffer.lookup_transform(
                    target_frame, source_frame, Time(),
                )
                return True
            except TransformException:
                if attempt % 20 == 0:
                    self.get_logger().info(
                        f"Waiting for TF '{source_frame}' -> '{target_frame}'..."
                    )
                attempt += 1
                self.sleep_for(0.1)
        self.get_logger().error(
            f"TF '{source_frame}' not available after {timeout_sec}s"
        )
        return False

    def _get_current_joints(self, get_observation: GetObservationCallback) -> np.ndarray:
        """Read current arm joint positions from observation."""
        obs = get_observation()
        joint_map = dict(zip(obs.joint_states.name, obs.joint_states.position))
        return np.array([joint_map[n] for n in JOINT_NAMES])

    def _compute_approach_pose(self, port_tf: Transform, z_above: float):
        """Compute the gripper_tcp target pose that places the plug above the port.

        Returns (position[3], quaternion[4]) in base_link frame.
        quaternion order: [qw, qx, qy, qz] (cuRobo convention).
        """
        # --- orientation: align plug with port ---
        plug_tf = self._parent_node._tf_buffer.lookup_transform(
            "base_link",
            f"{self._task.cable_name}/{self._task.plug_name}_link",
            Time(),
        )
        gripper_tf = self._parent_node._tf_buffer.lookup_transform(
            "base_link", "gripper/tcp", Time(),
        )

        q_port = (
            port_tf.rotation.w, port_tf.rotation.x,
            port_tf.rotation.y, port_tf.rotation.z,
        )
        q_plug = (
            plug_tf.transform.rotation.w, plug_tf.transform.rotation.x,
            plug_tf.transform.rotation.y, plug_tf.transform.rotation.z,
        )
        q_plug_inv = (-q_plug[0], q_plug[1], q_plug[2], q_plug[3])
        q_diff = quaternion_multiply(q_port, q_plug_inv)

        q_gripper = (
            gripper_tf.transform.rotation.w, gripper_tf.transform.rotation.x,
            gripper_tf.transform.rotation.y, gripper_tf.transform.rotation.z,
        )
        q_target = quaternion_multiply(q_diff, q_gripper)

        # --- position: above port, accounting for plug-to-gripper offset ---
        plug_gripper_dz = (
            gripper_tf.transform.translation.z - plug_tf.transform.translation.z
        )

        pos = [
            port_tf.translation.x,
            port_tf.translation.y,
            port_tf.translation.z + z_above - plug_gripper_dz,
        ]
        quat = [q_target[0], q_target[1], q_target[2], q_target[3]]
        return pos, quat

    # ------------------------------------------------------------------
    # Execute joint trajectory via aic_controller
    # ------------------------------------------------------------------

    def _execute_joint_trajectory(
        self, waypoints: np.ndarray, move_robot: MoveRobotCallback, dt: float = 0.002,
    ):
        self.get_logger().info(
            f"Executing trajectory: {len(waypoints)} waypoints, dt={dt}s, "
            f"total={len(waypoints)*dt:.2f}s"
        )
        for wp in waypoints:
            joint_cmd = JointMotionUpdate(
                target_state=JointTrajectoryPoint(positions=wp.tolist()),
                target_stiffness=[85.0] * 6,
                target_damping=[75.0] * 6,
                target_feedforward_torque=[0.0] * 6,
                trajectory_generation_mode=TrajectoryGenerationMode(
                    mode=TrajectoryGenerationMode.MODE_POSITION,
                ),
            )
            move_robot(joint_motion_update=joint_cmd)
            self.sleep_for(dt)

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def insert_cable(
        self,
        task: Task,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
    ):
        self.get_logger().info(f"CheatCodeCurobo.insert_cable() task: {task}")
        self._task = task

        port_frame = f"task_board/{task.target_module_name}/{task.port_name}_link"
        cable_tip_frame = f"{task.cable_name}/{task.plug_name}_link"

        # 1. Wait for TF frames
        for frame in [port_frame, cable_tip_frame]:
            if not self._wait_for_tf("base_link", frame):
                return False

        try:
            port_tf_stamped = self._parent_node._tf_buffer.lookup_transform(
                "base_link", port_frame, Time(),
            )
        except TransformException as ex:
            self.get_logger().error(f"Port TF lookup failed: {ex}")
            return False
        port_transform = port_tf_stamped.transform

        # 2. Read current joint state
        current_joints = self._get_current_joints(get_observation)
        self.get_logger().info(f"Current joints: {current_joints}")

        # 3. Compute approach target (15 cm above port)
        z_above = 0.15
        try:
            target_pos, target_quat = self._compute_approach_pose(
                port_transform, z_above,
            )
        except TransformException as ex:
            self.get_logger().error(f"TF lookup for approach pose failed: {ex}")
            return False

        self.get_logger().info(f"Approach target pos: {target_pos}")
        self.get_logger().info(f"Approach target quat: {target_quat}")

        # 4. Plan with cuRobo
        send_feedback("Planning approach with cuRobo...")

        start_state = CuJointState.from_position(
            torch.tensor([current_joints], dtype=torch.float32, device="cuda:0"),
            joint_names=JOINT_NAMES,
        )
        goal_pose = CuPose.from_list(
            target_pos + target_quat  # [x, y, z, qw, qx, qy, qz]
        )

        result = self._mg.plan_single(
            start_state, goal_pose,
            MotionGenPlanConfig(max_attempts=10, timeout=5.0),
        )

        if not result.success.item():
            self.get_logger().error(f"cuRobo planning failed: {result.status}")
            send_feedback(f"cuRobo planning failed: {result.status}")
            return False

        # 5. Extract and execute trajectory
        traj = result.get_interpolated_plan()
        waypoints = traj.position.cpu().numpy()

        send_feedback(f"Executing cuRobo trajectory ({len(waypoints)} waypoints)")
        self._execute_joint_trajectory(waypoints, move_robot, dt=0.002)

        # 6. Settle
        self.get_logger().info("Approach complete. Settling for 2s...")
        self.sleep_for(2.0)

        self.get_logger().info("CheatCodeCurobo.insert_cable() done (approach only)")
        return True
