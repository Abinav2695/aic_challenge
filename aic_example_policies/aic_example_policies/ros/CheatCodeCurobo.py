
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
    MotionUpdate,
)
from geometry_msgs.msg import Point, Pose, Quaternion, Vector3, Wrench
from std_msgs.msg import Header
from aic_task_interfaces.msg import Task
from rclpy.duration import Duration
from rclpy.time import Time
from tf2_ros import TransformException
from transforms3d._gohlketransforms import quaternion_multiply, quaternion_slerp
from transforms3d.quaternions import quat2mat, mat2quat

# cuRobo
from curobo.types.math import Pose as CuPose
from curobo.types.robot import JointState as CuJointState
from curobo.types.file_path import ContentPath
from curobo.util.xrdf_utils import convert_xrdf_to_curobo
from curobo.util_file import load_yaml
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig
from geometry_msgs.msg import Point, Pose, Quaternion
from std_srvs.srv import Trigger



XRDF_PATH = os.path.expanduser(
    "~/ws_aic/src/aic/aic_assets/curobo_assets/ur5e_aic_valid.xrdf"
)
URDF_PATH = os.path.expanduser(
    "~/ws_aic/src/aic/aic_assets/curobo_assets/ur5e_curobo_dummy.urdf"
)

JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]

PROFILE_APPROACH_JOINTSPACE = {
    "stiffness": [100.0, 100.0, 100.0,  50.0,  50.0,  50.0],
    "damping":   [ 40.0,  40.0,  40.0,  10.0,  10.0,  10.0],
}

# PROFILE_COMPLIANT = {
#     "stiffness": [100.0, 100.0, 100.0, 50.0, 50.0, 50.0],
#     "damping":   [40.0, 40.0, 40.0, 15.0, 15.0, 15.0],
# }

# PROFILE_INSERTION = {
#     "stiffness": [200.0, 200.0, 200.0, 80.0, 80.0, 80.0],
#     "damping":   [80.0,  80.0,  80.0,  40.0, 40.0, 40.0],
# }


class CheatCodeCurobo(Policy):

    def __init__(self, parent_node):
        self._tip_x_error_integrator = 0.0
        self._tip_y_error_integrator = 0.0
        self._max_integrator_windup = 0.05
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
            robot_cfg, interpolation_dt=0.001,
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

    def _tare_ft_sensor(self):
        """Calls the tare service to zero the Force/Torque sensor."""
        self.get_logger().info("Taring Force/Torque sensor...")
        
        # Create the service client using the parent node
        client = self._parent_node.create_client(Trigger, '/aic_controller/tare_force_torque_sensor')
        
        # Wait up to 3 seconds for the service to be available
        if not client.wait_for_service(timeout_sec=3.0):
            self.get_logger().error("Tare FT sensor service not available!")
            return False

        # Make the request
        req = Trigger.Request()
        future = client.call_async(req)

        # Safely wait for the response using your existing sleep loop
        while not future.done():
            self.sleep_for(0.05)

        # Check the result
        response = future.result()
        if response.success:
            self.get_logger().info(f"Sensor tared successfully: {response.message}")
            # Give it a tiny fraction of a second to settle
            self.sleep_for(0.2) 
            return True
        else:
            self.get_logger().error(f"Failed to tare sensor: {response.message}")
            return False


    def send_joint_target(self, move_robot, positions, profile=None):
        """Send a single joint-space target."""
        if profile is None:
            profile = PROFILE_APPROACH_JOINTSPACE

        cmd = JointMotionUpdate(
            target_stiffness=profile["stiffness"],
            target_damping=profile["damping"],
            target_feedforward_torque=[0.0] * 6,
            trajectory_generation_mode=TrajectoryGenerationMode(
                mode=TrajectoryGenerationMode.MODE_POSITION
            ),
        )
        cmd.target_state.positions = (
            positions.tolist() if hasattr(positions, 'tolist') else list(positions)
        )
        move_robot(joint_motion_update=cmd)





    def insert_cable(
        self,
        task: Task,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
        ):

        self.get_logger().info(f"CheatCode.insert_cable() task: {task}")
        self._task = task

        self._tare_ft_sensor()

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

        self.execute_waypoint(waypoints, move_robot, get_observation)
        self.sleep_for(1.0)  # let the robot settle at the approach pose

        # # Phase 2: correct cable flex error
        # self.correct_approach(port_frame, cable_tip_frame, move_robot, timeout=3.0)

        # self.descend_and_insert(port_frame, cable_tip_frame,
        #                get_observation, move_robot, timeout=30.0)

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
        ############ Math ###################
        # to find port orientation 
        # q_diff * q_plug = q_port
        # q_diff = q_port * q_plug_inv
        # q_gripper = q_diff * q_gripper 
        # gives correct orientation for gripper wrt to port  

    
        q_port = [
            port_tf.transform.rotation.w,
            port_tf.transform.rotation.x,
            port_tf.transform.rotation.y,
            port_tf.transform.rotation.z,
        ]

        q_plug =  [
            cable_tip_tf.transform.rotation.w,
            cable_tip_tf.transform.rotation.x,
            cable_tip_tf.transform.rotation.y,
            cable_tip_tf.transform.rotation.z,
        ]

        q_plug_inv = [
            -q_plug[0],
            q_plug[1],
            q_plug[2],
            q_plug[3],
        ]

        q_diff = quaternion_multiply(q_port, q_plug_inv)

        q_gripper = [
            gripper_tf.transform.rotation.w,
            gripper_tf.transform.rotation.x,
            gripper_tf.transform.rotation.y,
            gripper_tf.transform.rotation.z,
        ]

        gripper_target_quat = quaternion_multiply(q_diff, q_gripper)

        port_xyz = [
            port_tf.transform.translation.x,
            port_tf.transform.translation.y,
            port_tf.transform.translation.z,
        ]

        gripper_z = gripper_tf.transform.translation.z
        plug_z = cable_tip_tf.transform.translation.z

        plug_length = abs(gripper_z-plug_z)

        gripper_target_xyz = [
            port_xyz[0],
            port_xyz[1],
            port_xyz[2] + z_above + plug_length
        ]

        return gripper_target_xyz, gripper_target_quat



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
            
        waypoints = traj.position.squeeze(0).cpu().numpy()

        return True, waypoints

    # def execute_trajectory(self, waypoints, move_robot, interpolation_dt=0.02):
    #     """
    #     Executes trajectory and provide waypoints from curobo to aic_controller

    #     Args:
    #         waypoints: np.ndarray of joint configuration for path waypoint
    #         move_robot: MoveRobotCallback
    #         interpolation_dt: interpolation_dt of waypoints

    #     Returns: None
    #     """
    #     joint_cmd = JointMotionUpdate(
    #         # target_stiffness = [90.0, 90.0, 90.0, 90.0, 90.0, 90.0],
    #         # target_damping   = [30.0, 30.0, 30.0, 30.0, 30.0, 30.0],
    #         target_stiffness=[300.0, 300.0, 300.0, 100.0, 100.0, 100.0],
    #         target_damping=  [120.0, 120.0, 120.0, 50.0,  50.0,  50.0],
    #         target_feedforward_torque = [0.0]*6,
    #         trajectory_generation_mode = TrajectoryGenerationMode(
    #                 mode=TrajectoryGenerationMode.MODE_POSITION),
    #         )

    #     for wp in waypoints:
    #         joint_cmd.target_state.positions = wp.tolist()
    #         move_robot(joint_motion_update=joint_cmd)
    #         self.sleep_for(interpolation_dt)

    def execute_waypoint(self, waypoints, move_robot, get_observation, dt=0.001, profile=None):
        """Execute a full joint-space trajectory."""
        for wp in waypoints:
            self.send_joint_target(move_robot, wp, profile=profile)

            obs = get_observation()
            w = obs.wrist_wrench.wrench
            self.get_logger().info(
                # f"WP {i}/{len(waypoints)} | "
                f"F: x={w.force.x:.3f} y={w.force.y:.3f} z={w.force.z:.3f} | "
                f"T: x={w.torque.x:.3f} y={w.torque.y:.3f} z={w.torque.z:.3f}"
            )

            self.sleep_for(dt)


        
    
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


    # def descend_and_insert(self, port_frame, cable_tip_frame,
    #                    get_observation, move_robot, timeout=30.0):
    #     """
    #     Force-aware descent using controller's built-in impedance.
    #     Uses feedforward wrench for gentle push, anisotropic stiffness
    #     for compliant insertion, and wrench_feedback_gains for self-centering.
    #     """
    #     start = self.time_now()
    #     timeout_dur = Duration(seconds=timeout)
    #     z_offset = 0.05

    #     # Get port transform once for orientation reference
    #     try:
    #         port_tf = self._parent_node._tf_buffer.lookup_transform(
    #             "base_link", port_frame, Time(),
    #         )
    #     except TransformException:
    #         self.get_logger().error("Cannot look up port frame for descent")
    #         return

    #     # Port's insertion direction in base_link frame
    #     # Port z-axis points along insertion direction
    #     R_port = quat2mat([
    #         port_tf.transform.rotation.w,
    #         port_tf.transform.rotation.x,
    #         port_tf.transform.rotation.y,
    #         port_tf.transform.rotation.z,
    #     ])
    #     insertion_dir = R_port[:, 2]  # port's local z in world frame

    #     while (self.time_now() - start) < timeout_dur:
    #         if z_offset < -0.015:
    #             break

    #         try:
    #             plug_tf = self._parent_node._tf_buffer.lookup_transform(
    #                 "base_link", cable_tip_frame, Time(),
    #             )
    #             port_tf = self._parent_node._tf_buffer.lookup_transform(
    #                 "base_link", port_frame, Time(),
    #             )
    #             gripper_tf = self._parent_node._tf_buffer.lookup_transform(
    #                 "base_link", "gripper/tcp", Time(),
    #             )
    #         except TransformException:
    #             self.sleep_for(0.05)
    #             continue

    #         # x,y correction from TF
    #         dx = port_tf.transform.translation.x - plug_tf.transform.translation.x
    #         dy = port_tf.transform.translation.y - plug_tf.transform.translation.y

    #         # Read force from observation
    #         obs = get_observation()
    #         fz = abs(obs.wrist_wrench.wrench.force.z) if obs else 0.0
    #         fx = abs(obs.wrist_wrench.wrench.force.x) if obs else 0.0
    #         fy = abs(obs.wrist_wrench.wrench.force.y) if obs else 0.0
    #         total_force = (fx**2 + fy**2 + fz**2) ** 0.5

    #         # Adaptive descent speed based on force
    #         if total_force > 15.0:
    #             descent_step = 0.0001
    #             self.get_logger().warn(f"High force: {total_force:.1f}N, pausing")
    #         elif total_force > 5.0:
    #             descent_step = 0.0001
    #         else:
    #             descent_step = 0.0005

    #         z_offset -= descent_step

    #         # Target position
    #         plug_to_gripper_z = (
    #             gripper_tf.transform.translation.z
    #             - plug_tf.transform.translation.z
    #         )
    #         target_z = port_tf.transform.translation.z + z_offset - plug_to_gripper_z

    #         # Gentle push along insertion direction (3N)
    #         push_force = -3.0 * insertion_dir

    #         motion_update = MotionUpdate(
    #             header=Header(
    #                 frame_id="base_link",
    #                 stamp=self._parent_node.get_clock().now().to_msg(),
    #             ),
    #             pose=Pose(
    #                 position=Point(
    #                     x=gripper_tf.transform.translation.x + dx,
    #                     y=gripper_tf.transform.translation.y + dy,
    #                     z=target_z,
    #                 ),
    #                 orientation=Quaternion(
    #                     w=gripper_tf.transform.rotation.w,
    #                     x=gripper_tf.transform.rotation.x,
    #                     y=gripper_tf.transform.rotation.y,
    #                     z=gripper_tf.transform.rotation.z,
    #                 ),
    #             ),
    #             # Stiff x,y (centering), soft z (compliant insertion)
    #             target_stiffness=np.diag(
    #                 [90.0, 90.0, 20.0, 50.0, 50.0, 50.0]
    #             ).flatten(),
    #             target_damping=np.diag(
    #                 [50.0, 50.0, 15.0, 20.0, 20.0, 20.0]
    #             ).flatten(),
    #             # Push along port's insertion axis
    #             feedforward_wrench_at_tip=Wrench(
    #                 force=Vector3(
    #                     x=float(push_force[0]),
    #                     y=float(push_force[1]),
    #                     z=float(push_force[2]),
    #                 ),
    #                 torque=Vector3(x=0.0, y=0.0, z=0.0),
    #             ),
    #             # Admittance: lateral force → self-centering at 500Hz
    #             wrench_feedback_gains_at_tip=[0.5, 0.5, 0.5, 0.0, 0.0, 0.0],
    #             trajectory_generation_mode=TrajectoryGenerationMode(
    #                 mode=TrajectoryGenerationMode.MODE_POSITION,
    #             ),
    #         )

    #         move_robot(motion_update=motion_update)
    #         self.sleep_for(0.05)

    #     self.get_logger().info("Descent complete. Stabilizing...")
    #     self.sleep_for(5.0)
    def descend_and_insert(self, port_frame, cable_tip_frame,
                       get_observation, move_robot, timeout=30.0):
        """
        Force-aware descent with X/Y PI Integration to fight cable flex.
        Uses 'relieve and center' strategy if it catches the edge.
        """
        start = self.time_now()
        timeout_dur = Duration(seconds=timeout)
        z_offset = 0.05

        # PI Controller state to fight cable flex
        ix = 0.0
        iy = 0.0
        i_gain = 0.15 
        max_windup = 0.05 

        try:
            port_tf = self._parent_node._tf_buffer.lookup_transform(
                "base_link", port_frame, Time(),
            )
        except TransformException:
            self.get_logger().error("Cannot look up port frame for descent")
            return

        # Insertion direction (pushing IN, so negative Z)
        R_port = quat2mat([
            port_tf.transform.rotation.w, port_tf.transform.rotation.x,
            port_tf.transform.rotation.y, port_tf.transform.rotation.z,
        ])
        insertion_dir = R_port[:, 2]  
        push_force = -3.0 * insertion_dir

        while (self.time_now() - start) < timeout_dur:
            if z_offset < -0.015:
                break

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

            # --- 1. X/Y INTEGRATOR (Fighting the flex) ---
            dx = port_tf.transform.translation.x - plug_tf.transform.translation.x
            dy = port_tf.transform.translation.y - plug_tf.transform.translation.y

            # Accumulate error (windup clamp prevents runaway)
            ix = max(-max_windup, min(max_windup, ix + dx))
            iy = max(-max_windup, min(max_windup, iy + dy))

            # Add PI correction to the gripper's current target
            target_x = gripper_tf.transform.translation.x + dx + (i_gain * ix)
            target_y = gripper_tf.transform.translation.y + dy + (i_gain * iy)


            # --- 2. FORCE-ADAPTIVE DESCENT ---
            obs = get_observation()
            fz = abs(obs.wrist_wrench.wrench.force.z) if obs else 0.0
            fx = abs(obs.wrist_wrench.wrench.force.x) if obs else 0.0
            fy = abs(obs.wrist_wrench.wrench.force.y) if obs else 0.0
            total_force = (fx**2 + fy**2 + fz**2) ** 0.5

            if total_force > 15.0:
                # DANGER: Jammed hard. Back up slightly to relieve friction!
                descent_step = -0.0005 
                self.get_logger().warn(f"Force {total_force:.1f}N! Backing up to recenter.")
            elif total_force > 5.0:
                # WARNING: Touching the edge. Stop Z-descent to let X/Y catch up.
                descent_step = 0.0
                self.get_logger().info(f"Force {total_force:.1f}N. Pausing Z descent.")
            else:
                # SAFE: Coast is clear, push down.
                descent_step = 0.0005

            z_offset -= descent_step

            # Calculate Z target
            plug_to_gripper_z = (
                gripper_tf.transform.translation.z - plug_tf.transform.translation.z
            )
            target_z = port_tf.transform.translation.z + z_offset + plug_to_gripper_z


            # --- 3. EXECUTE MOTION ---
            motion_update = MotionUpdate(
                header=Header(
                    frame_id="base_link",
                    stamp=self._parent_node.get_clock().now().to_msg(),
                ),
                pose=Pose(
                    position=Point(x=target_x, y=target_y, z=target_z),
                    orientation=Quaternion(
                        w=gripper_tf.transform.rotation.w,
                        x=gripper_tf.transform.rotation.x,
                        y=gripper_tf.transform.rotation.y,
                        z=gripper_tf.transform.rotation.z,
                    ),
                ),
                target_stiffness=np.diag([90.0, 90.0, 20.0, 50.0, 50.0, 50.0]).flatten(),
                target_damping=np.diag([50.0, 50.0, 15.0, 20.0, 20.0, 20.0]).flatten(),
                feedforward_wrench_at_tip=Wrench(
                    force=Vector3(
                        x=float(push_force[0]), y=float(push_force[1]), z=float(push_force[2])
                    ),
                    torque=Vector3(x=0.0, y=0.0, z=0.0),
                ),
                wrench_feedback_gains_at_tip=[0.5, 0.5, 0.5, 0.0, 0.0, 0.0],
                trajectory_generation_mode=TrajectoryGenerationMode(
                    mode=TrajectoryGenerationMode.MODE_POSITION,
                ),
            )

            move_robot(motion_update=motion_update)
            self.sleep_for(0.05)

        self.get_logger().info("Descent complete. Stabilizing...")
        self.sleep_for(5.0)


        
