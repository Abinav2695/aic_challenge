
import numpy as np
import torch
import os

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

from aic_task_interfaces.msg import Task
from rclpy.duration import Duration
from rclpy.time import Time
from tf2_ros import TransformException
from transforms3d._gohlketransforms import quaternion_multiply
from transforms3d.quaternions import quat2mat, mat2quat

# cuRobo
from curobo.types.math import Pose as CuPose
from curobo.types.robot import JointState as CuJointState
from curobo.types.file_path import ContentPath
from curobo.util.xrdf_utils import convert_xrdf_to_curobo
from curobo.util_file import load_yaml
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig
from geometry_msgs.msg import Point, Pose, Quaternion

XRDF_PATH = os.path.expanduser(
    "~/ws_aic/src/aic_challenge/aic_assets/curobo_assets/ur5e_aic_valid.xrdf"
)
URDF_PATH = os.path.expanduser(
    "~/ws_aic/src/aic_challenge/aic_assets/curobo_assets/ur5e_curobo_dummy.urdf"
)

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

        robot_cfg["robot_cfg"]["kinematics"]["ee_link"] = "gripper_tcp"
        robot_cfg["robot_cfg"]["kinematics"]["base_link"] = "base_link"

        self.get_logger().info("CuRobo: building MotionGen config...")
        mg_config = MotionGenConfig.load_from_robot_config(
            robot_cfg, interpolation_dt=0.01,
        )

        self._mg = MotionGen(mg_config)
        self.get_logger().info("CuRobo: warming up (~15 s first time)...")
        self._mg.warmup()
        self.get_logger().info("CuRobo: ready.")



    def _wait_for_tf(self, target_frame: str, source_frame: str, timeout_sec: float = 10.0) -> bool:
        """Wait for a TF frame to become available."""
        start = self.time_now()
        timeout = Duration(seconds=timeout_sec)
        attempt = 0
        while (self.time_now() - start) < timeout:
            try:
                self._parent_node._tf_buffer.lookup_transform(
                    target_frame,
                    source_frame,
                    Time(),
                )
                return True
            except TransformException:
                if attempt % 20 == 0:
                    self.get_logger().info(
                        f"Waiting for transform '{source_frame}' -> '{target_frame}'... -- are you running eval with `ground_truth:=true`?"
                    )
                attempt += 1
                self.sleep_for(0.1)
        self.get_logger().error(f"Transform '{source_frame}' not available after {timeout_sec}s")
        return False




    def insert_cable(
        self,
        task: Task,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
        ):

        self.get_logger().info(f"CheatCode.insert_cable() task: {task}")
        self._task = task

        port_frame = f"task_board/{task.target_module_name}/{task.port_name}_link"
        cable_tip_frame = f"{task.cable_name}/{task.plug_name}_link"

        
        for frame in [port_frame, cable_tip_frame]:
            if not self._wait_for_tf("base_link", frame):
                return False

        z_above = 0.1 # 20cm above the port

        try:
            target_pos, target_quat = self.compute_approach_pose(
                port_frame, cable_tip_frame, z_above
            )
        except TransformException as ex:
            self.get_logger().error(f"Approach pose failed: {ex}")
            return False

        self.get_logger().info(f"Target pos: {target_pos}")
        self.get_logger().info(f"Target quat: {target_quat}")


        current_joints = self._get_current_joints(get_observation)

        success, waypoints = self.plan_trajectory(current_joints, target_pos, target_quat)

        if not success:
            self.get_logger().error("cuRobo planning failed")
            send_feedback("Planning failed")
            return False

        self.execute_trajectory(waypoints, move_robot, interpolation_dt=0.01)
        self.sleep_for(1.0)  # let the robot settle at the approach pose

        # Phase 2: correct cable flex error
        self.correct_approach(port_frame, cable_tip_frame, move_robot, timeout=3.0)

        try:
            plug_final = self._parent_node._tf_buffer.lookup_transform(
                "base_link", cable_tip_frame, Time(),
            )
            port_final = self._parent_node._tf_buffer.lookup_transform(
                "base_link", port_frame, Time(),
            )
            
            # Position error
            dx = plug_final.transform.translation.x - port_final.transform.translation.x
            dy = plug_final.transform.translation.y - port_final.transform.translation.y
            dz = plug_final.transform.translation.z - port_final.transform.translation.z
            self.get_logger().info(f"Position error: dx={dx:.4f}m dy={dy:.4f}m dz={dz:.4f}m")
            
            # Orientation: compare plug vs port quaternions
            plug_q = [plug_final.transform.rotation.w, plug_final.transform.rotation.x,
                      plug_final.transform.rotation.y, plug_final.transform.rotation.z]
            port_q = [port_final.transform.rotation.w, port_final.transform.rotation.x,
                      port_final.transform.rotation.y, port_final.transform.rotation.z]
            # Dot product of quaternions: 1.0 = perfectly aligned
            dot = abs(sum(a * b for a, b in zip(plug_q, port_q)))
            self.get_logger().info(f"Orientation alignment: {dot:.4f} (1.0 = perfect)")
            
        except TransformException:
            self.get_logger().warn("Could not verify final pose")

        return True 




    def _tf_to_matrix(self, tf_msg):
        """Convert a TransformStamped to a 4x4 homogeneous matrix."""
        t = tf_msg.transform.translation
        q = tf_msg.transform.rotation
        R = quat2mat([q.w, q.x, q.y, q.z])
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = [t.x, t.y, t.z]
        return T

    def compute_approach_pose(self, port_frame, cable_tip_frame, z_above):
        """
        Compute gripper TCP target using 4x4 homogeneous transforms.

        Args:
            port_frame: str, TF frame name of the target port.
            cable_tip_frame: str, TF frame name of the plug tip.
            z_above: float, distance above the port in meters (world Z).

        Returns:
            gripper_target_pos: list [x, y, z] in base_link frame, meters.
            gripper_target_quat: list [w, x, y, z] in base_link frame.
        """
        # Look up all three frames
        cable_tip_tf = self._parent_node._tf_buffer.lookup_transform(
            "base_link", cable_tip_frame, Time(),
        )
        port_tf = self._parent_node._tf_buffer.lookup_transform(
            "base_link", port_frame, Time(),
        )
        gripper_tf = self._parent_node._tf_buffer.lookup_transform(
            "base_link", "gripper/tcp", Time(),
        )

        # Convert to 4x4 matrices
        T_base_plug = self._tf_to_matrix(cable_tip_tf)
        T_base_gripper = self._tf_to_matrix(gripper_tf)
        T_base_port = self._tf_to_matrix(port_tf)

        # Step 1: Find constant offset from gripper to plug (in gripper's local frame)
        # T_base_plug = T_base_gripper @ T_gripper_plug
        # T_gripper_plug = inv(T_base_gripper) @ T_base_plug
        T_gripper_plug = np.linalg.inv(T_base_gripper) @ T_base_plug

        # Step 2: Define where we want the plug to be (at port, but z_above higher)
        T_base_plug_target = np.copy(T_base_port)
        T_base_plug_target[2, 3] += z_above  # raise in world Z

        # Step 3: Compute where gripper must be so plug lands at target
        # T_base_plug_target = T_base_gripper_target @ T_gripper_plug
        # T_base_gripper_target = T_base_plug_target @ inv(T_gripper_plug)
        T_base_gripper_target = T_base_plug_target @ np.linalg.inv(T_gripper_plug)

        # Step 4: Extract position and quaternion
        gripper_target_pos = [float(x) for x in T_base_gripper_target[:3, 3]]
        gripper_target_quat = [float(q) for q in mat2quat(T_base_gripper_target[:3, :3])]

        self.get_logger().info(f"Target pos: {gripper_target_pos}")
        self.get_logger().info(f"Target quat: {gripper_target_quat}")

        return gripper_target_pos, gripper_target_quat



    def plan_trajectory(self, current_joints, target_pos, target_quat):
        """
        Plans the trajectory using Curobo MotionGen
        Args:
            current_joints: tuple of current joint configuration,
            target_pos:     tuple (x, y, z) in base_link frame, meters.
            target_quat:    tuple (w, x, y, z) in base_link frame.
        Returns:
            success: bool
            waypoints: np.ndarray of joint configuration for path waypoint


        """
        start_state = CuJointState.from_position(
                torch.tensor([current_joints], dtype=torch.float32, device="cuda:0"),
                joint_names=JOINT_NAMES,
            )
        goal_pose = CuPose.from_list(
            list(target_pos) + list(target_quat)  # concatenates to [x, y, z, qw, qx, qy, qz]
            )
            

        result = self._mg.plan_single(
                start_state, goal_pose,
                MotionGenPlanConfig(),
            )
        if not result.success.item():
            return False, None

        traj=result.get_interpolated_plan()
            
        waypoints = traj.position.cpu().numpy()

        return True, waypoints

    def execute_trajectory(self, waypoints, move_robot, interpolation_dt):
        """
        Executes trajectory and provide waypoints from curobo to aic_controller

        Args:
            waypoints: np.ndarray of joint configuration for path waypoint
            move_robot: MoveRobotCallback
            interpolation_dt: interpolation_dt of waypoints

        Returns: None
        """
        joint_cmd = JointMotionUpdate(
            target_stiffness = [90.0, 90.0, 90.0, 90.0, 90.0, 90.0],
            target_damping   = [30.0, 30.0, 30.0, 30.0, 30.0, 30.0],
            target_feedforward_torque = [0.0]*6,
            trajectory_generation_mode = TrajectoryGenerationMode(
                    mode=TrajectoryGenerationMode.MODE_POSITION),
            )

        for wp in waypoints:
            joint_cmd.target_state.positions = wp.tolist()
            move_robot(joint_motion_update=joint_cmd)
            self.sleep_for(interpolation_dt)


        
    
    def _get_current_joints(self, get_observation):
        """
        Gets current joint configuration
        Args: 
            get_observation: GetObservationCallback
        returns:
            current_joint_config: list of current joint config
        """
        # JointState message has 2 attributions:
        # joint_states.name -> joint names
        # joint_states.position -> joint configs 

        current_observation = get_observation()
        joint_state_msg = current_observation.joint_states

        joint_state_name = joint_state_msg.name
        joint_state_position = joint_state_msg.position

        current_state = dict(zip(joint_state_name, joint_state_position))
        # Dict -> mapping joint_state with position 

        current_joint_config = [current_state[name] for name in JOINT_NAMES]

        return current_joint_config

    def correct_approach(self, port_frame, cable_tip_frame, move_robot, timeout=5.0):
        """
        Closed-loop Cartesian correction with PI control.
        Accumulates error over time to overcome cable stiffness.
        """
        start = self.time_now()
        timeout_dur = Duration(seconds=timeout)

        # PI controller state
        ix = 0.0  # x integrator
        iy = 0.0  # y integrator
        i_gain = 0.3
        max_windup = 0.05  # clamp integrator to prevent runaway

        while (self.time_now() - start) < timeout_dur:
            try:
                plug_tf = self._parent_node._tf_buffer.lookup_transform(
                    "base_link", cable_tip_frame, Time(),
                )
                port_tf = self._parent_node._tf_buffer.lookup_transform(
                    "base_link", port_frame, Time(),
                )
                gripper_tf = self._parent_node._tf_buffer.lookup_transform(
                    "base_link", "gripper/tcp", Time(),
                )
            except TransformException:
                self.sleep_for(0.05)
                continue

            dx = port_tf.transform.translation.x - plug_tf.transform.translation.x
            dy = port_tf.transform.translation.y - plug_tf.transform.translation.y
            error = (dx**2 + dy**2) ** 0.5

            if error < 0.002:
                self.get_logger().info(f"Correction converged: error={error:.4f}m")
                break

            # Accumulate error (integrator)
            ix = max(-max_windup, min(max_windup, ix + dx))
            iy = max(-max_windup, min(max_windup, iy + dy))

            # P + I correction
            correction_x = dx + i_gain * ix
            correction_y = dy + i_gain * iy

            corrected_pose = Pose(
                position=Point(
                    x=gripper_tf.transform.translation.x + correction_x,
                    y=gripper_tf.transform.translation.y + correction_y,
                    z=gripper_tf.transform.translation.z,
                ),
                orientation=Quaternion(
                    w=gripper_tf.transform.rotation.w,
                    x=gripper_tf.transform.rotation.x,
                    y=gripper_tf.transform.rotation.y,
                    z=gripper_tf.transform.rotation.z,
                ),
            )

            self.set_pose_target(move_robot=move_robot, pose=corrected_pose)
            self.sleep_for(0.05)

        # Log final error
        try:
            plug_final = self._parent_node._tf_buffer.lookup_transform(
                "base_link", cable_tip_frame, Time(),
            )
            port_final = self._parent_node._tf_buffer.lookup_transform(
                "base_link", port_frame, Time(),
            )
            dx = plug_final.transform.translation.x - port_final.transform.translation.x
            dy = plug_final.transform.translation.y - port_final.transform.translation.y
            self.get_logger().info(f"After correction: dx={dx:.4f}m dy={dy:.4f}m")
        except TransformException:
            pass
        

        
