import mujoco
import numpy as np
import pdb
import pinocchio as pin
from nav_msgs.msg import Odometry

from .lcm2mujuco_bridge import Lcm2MujocoBridge
from .tron1_wheeled_bridge import Tron1WheeledBridge
from arc_bridge.utils import *
from arc_bridge.state_estimators import SlideObjectFloatingBaseLinearStateEstimator
from arc_bridge.lcm_msgs import tron1_wheeled_state_t, tron1_wheeled_plan_t, sliding_plan_t
from arc_bridge.lcm_msgs import sliding_state_t, sliding_control_t

class SlidingBridge(Tron1WheeledBridge):
    def __init__(self, mj_model, mj_data, config):
        super().__init__(mj_model, mj_data, config)

        self.vicon_slide_object_pos = np.zeros(3, dtype=float)
        self.vicon_slide_object_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
        self.vicon_slide_object_lin_vel_world = np.zeros(3, dtype=float)
        self.vicon_slide_object_ang_omega_world = np.zeros(3, dtype=float)
        # Transformation from vicon slide object measurement frame to base_link frame
        self.T_vicon_slide_object_meas_to_base_link = np.eye(4, dtype=float)
        self.T_vicon_slide_object_meas_to_base_link[:3, 3] = np.array([0.0, 0.0, -0.010])
        if self.in_replay_mode and self.vicon_ros2_client is not None:
            self.vicon_ros2_client.subscribe_slide_object(self._vicon_slide_object_callback, topic="/odometry/slide_object")

        self.slide_object_height_init = 0.0
        self.slide_object_dt_estimator = 0.001
        # Process noise (px, py, pz, vx, vy, vz, ax, ay, az)
        object_KF_Q = np.diag([0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]) 
        # Measurement noise (px, py, pz, vx, vy, vz, ax, ay, az)
        object_KF_R = np.diag([0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]) 
        self.slide_object_kf = SlideObjectFloatingBaseLinearStateEstimator(self.slide_object_dt_estimator, object_KF_Q, object_KF_R, self.slide_object_height_init)
        self.slide_object_use_kf = False

        # object visualization variables
        self.vis_object_pos = None
        self.vis_object_vel = None
        
    def parse_robot_specific_low_state(self):
        super().parse_robot_specific_low_state()

        if not self.in_replay_mode:
            self.low_state.pos_ob[:] = self.mj_data.qpos[15:18]
            self.low_state.quat_ob[:] = self.mj_data.qpos[18:22]
            self.low_state.vel_ob[:] = self.mj_data.qvel[14:17]
            self.low_state.omega_ob[:] = self.mj_data.qvel[17:20]

        self.update_slide_object_state_estimation()

    def update_slide_object_state_estimation(self):
        if self.slide_object_use_kf:
            if self.in_replay_mode:
                # in replay mode, correction is done in callback
                with self.vicon_lock:
                    self.slide_object_kf.predict(u=np.zeros(1))
            else:
                self.slide_object_kf.predict(u=np.zeros(1))
                # use ground truth position/velocity for testing
                pos = np.array(self.low_state.pos_ob[:3], dtype=float)
                vel = np.array(self.low_state.vel_ob[:3], dtype=float)
                acc_world = np.zeros(3, dtype=float) # assume zero acceleration
                meas_full_state = np.hstack((pos, vel, acc_world))
                self._test_slide_object_correction_in_simulation(meas_full_state)

            # send to controller
            self.low_state.pos_ob[:] = self.slide_object_kf.x[:3]
            self.low_state.quat_ob[:] = self.vicon_slide_object_quat  # use directly
            self.low_state.vel_ob[:] = self.slide_object_kf.x[3:6]
            self.low_state.omega_ob[:] = self.vicon_slide_object_ang_omega_world  # use directly
        else:
            self.low_state.pos_ob[:] = self.vicon_slide_object_pos
            self.low_state.quat_ob[:] = self.vicon_slide_object_quat
            self.low_state.vel_ob[:] = self.vicon_slide_object_lin_vel_world
            self.low_state.omega_ob[:] = self.vicon_slide_object_ang_omega_world

        if self.in_replay_mode:
            # display as what the controller sees
            self.mj_data.qpos[15:18] = self.low_state.pos_ob
            self.mj_data.qpos[18:22] = self.low_state.quat_ob
            self.mj_data.qvel[14:17] = self.low_state.vel_ob
            self.mj_data.qvel[17:20] = self.low_state.omega_ob

    def lcm_state_handler(self, channel, data):
        if self.mj_data is None:
            return

        # Get state msg from robot SDK topic
        msg = tron1_wheeled_state_t.decode(data)

        # Update mj_data for visualization - update in another thread/function
        # self.mj_data.qpos[0] = msg.position[0] # robot in the mujoco viewer is vicon pose
        # self.mj_data.qpos[1] = msg.position[1]
        # self.mj_data.qpos[2] = msg.position[2]
        # self.mj_data.qpos[3] = msg.quaternion[0]
        # self.mj_data.qpos[4] = msg.quaternion[1]
        # self.mj_data.qpos[5] = msg.quaternion[2]
        # self.mj_data.qpos[6] = msg.quaternion[3]
        self.mj_data.qpos[7:7+8] = msg.qj_pos - self.joint_offsets # to macth with xml
        self.mj_data.qvel[:] = 0

        # Partially update low_state from the msg sent by robot SDK
        self.low_state.qj_pos[:] = msg.qj_pos
        self.low_state.qj_vel[:] = msg.qj_vel
        self.low_state.qj_tau[:] = msg.qj_tau
        self.low_state.acceleration[:] = msg.acceleration
        self.low_state.omega[:] = msg.omega
        self._rx_state_quaternion[:] = msg.quaternion
        self.low_state.rpy[:] = msg.rpy
        self._rx_state_position[:] = msg.position # copy because of write from another thread
        self._rx_state_velocity[:] = msg.velocity
        self._rx_state_available = True

    def _vicon_slide_object_callback(self, msg: Odometry) -> None:
        stamp = float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) * 1e-9
        clock_now = self.vicon_ros2_client.node.get_clock().now()
        now_sec = clock_now.nanoseconds * 1e-9
        dt = 0.0 if stamp <= 0 else max(0.0, now_sec - stamp)

        pos_msg = msg.pose.pose.position
        pos = np.array([pos_msg.x, pos_msg.y, pos_msg.z], dtype=float)
        quat_msg = msg.pose.pose.orientation
        quat = Quaternion(quat_msg.w, quat_msg.x, quat_msg.y, quat_msg.z)

        twist_msg = msg.twist.twist
        vel_body = np.array([twist_msg.linear.x, twist_msg.linear.y, twist_msg.linear.z], dtype=float)
        omega_body = np.array([twist_msg.angular.x, twist_msg.angular.y, twist_msg.angular.z], dtype=float)
        
        R_body_to_world = quat_to_rot(quat)  # vicon body to world
        # apply the base_link transform on position and orientation
        pos, _, _ = self._transform_body_pos_quat_to_base_link(pos, quat, self.T_vicon_slide_object_meas_to_base_link)
        # apply the base_link transform on body twist
        # vel_body, omega_body = self._transform_body_twist_to_base_link(vel_body, omega_body, self.T_vicon_slide_object_meas_to_base_link)

        vel_world = R_body_to_world @ vel_body
        omega_world = R_body_to_world @ omega_body

        if not self.slide_object_use_kf:
            self.vicon_slide_object_pos[:] = pos
            self.vicon_slide_object_quat[:] = np.array([quat.w, quat.x, quat.y, quat.z], dtype=float)
            self.vicon_slide_object_lin_vel_world[:] = vel_world
            self.vicon_slide_object_ang_omega_world[:] = omega_world
            return

        acc_world = np.zeros(3, dtype=float) # assume zero acceleration

        with self.vicon_lock:
            acc_est = acc_world
            # pos_now = pos + vel_world * dt + 0.5 * acc_est * dt * dt
            pos_now = pos + (vel_world + acc_est * dt) * dt  # semi-implicit Euler
            vel_now = vel_world + acc_est * dt
            meas_full_state = np.hstack((pos_now, vel_now, acc_est))

            self.vicon_slide_object_pos = pos_now
            self.vicon_slide_object_quat = np.array([quat.w, quat.x, quat.y, quat.z], dtype=float)
            self.vicon_slide_object_lin_vel_world[:3] = vel_now
            self.vicon_slide_object_ang_omega_world[:3] = omega_world

            self.slide_object_kf.correct(meas_full_state)

    def _test_slide_object_correction_in_simulation(self, meas: np.ndarray):
        dt = 0.0  # change manually if needed
        quat = Quaternion(*self.low_state.quat_ob)
        R_body_to_world = quat_to_rot(quat)

        pos = np.array(meas[:3], dtype=float)
        vel_world = np.array(meas[3:6], dtype=float)
        omega_world = np.array(self.low_state.omega_ob[:3], dtype=float)
        acc_world = np.array(meas[6:9], dtype=float)

        with self.vicon_lock:
            acc_est = acc_world
            # pos_now = pos + vel_world * dt + 0.5 * acc_est * dt * dt
            pos_now = pos + (vel_world + acc_est * dt) * dt  # semi-implicit Euler
            vel_now = vel_world + acc_est * dt
            meas_full_state = np.hstack((pos_now, vel_now, acc_est))

            self.vicon_slide_object_pos = pos_now
            self.vicon_slide_object_quat = np.array([quat.w, quat.x, quat.y, quat.z], dtype=float)
            self.vicon_slide_object_lin_vel_world[:3] = vel_now
            self.vicon_slide_object_ang_omega_world[:3] = omega_world

            self.slide_object_kf.correct(meas_full_state)

    # def mpc_command_handler(self, channel, data):
    #     if self.mj_data == None:
    #         return
    #     # Get desired torso and wheel state from tron1_wheeled_plan topic
    #     # and visualize it in mujoco viewer


    #     # We need to overwrite vis_pos_est and vis_vel_est for visualization
    #     msg = sliding_plan_t.decode(data)

    #     # Extract MPC command trajectory
    #     n_horizon = msg.n_horizon
    #     desired_q_trb_traj = np.array(msg.qd_trb).reshape(n_horizon, 14)  # (n_horizon, 14)
    #     desired_v_trb_traj = np.array(msg.dqd_trb).reshape(n_horizon, 14)  # (n_horizon, 14)
    #     desired_grf_traj = np.array(msg.lambda_des).reshape(n_horizon, 6)  # (n_horizon, 6)
    #     desired_p_object_traj = np.array(msg.qd_ob).reshape(n_horizon, 3)  # (n_horizon, 3)
    #     desired_v_object_traj = np.array(msg.dqd_ob).reshape(n_horizon, 3)  # (n_horizon, 3)

    #     # otherwise in yaw-aligned frame
    #     traj_in_global_frame = True 
    #     robot_pos = np.array(self.low_state.position[:3], dtype=float)
    #     robot_pos[2] = 0.0  # ignore height
    #     robot_yaw = self.low_state.rpy[2]
    #     R_yaw = np.array([
    #         [np.cos(robot_yaw), -np.sin(robot_yaw), 0],
    #         [np.sin(robot_yaw),  np.cos(robot_yaw), 0],
    #         [0, 0, 1]
    #     ])
    #     if traj_in_global_frame:
    #         robot_pos = robot_pos*0
    #         R_yaw = np.array([
    #             [1, 0, 0],
    #             [0, 1, 0],
    #             [0, 0, 1]
    #         ])

    #     # Build floating base state trajectory from MPC command (in yaw-aligned frame)
    #     desired_pos_yaw = desired_q_trb_traj[:, :3]  # (n_horizon, 3)
    #     desired_rpy_yaw = desired_q_trb_traj[:, [5, 4, 3]]  # (n_horizon, 3) - revert ypr back to rpy order
    #     desired_lin_vel_yaw = desired_v_trb_traj[:, :3]  # (n_horizon, 3)
    #     # desired_ang_vel = desired_v_trb_traj[:, [5, 4, 3]]  # (n_horizon, 3) - revert ypr back to rpy order

    #     # Transform positions from yaw-aligned frame to world frame
    #     desired_pos = np.array([R_yaw @ pos + robot_pos for pos in desired_pos_yaw])  # (n_horizon, 3)
        
    #     # Transform velocities from yaw-aligned frame to world frame
    #     desired_lin_vel = np.array([R_yaw @ vel for vel in desired_lin_vel_yaw])  # (n_horizon, 3)
        
    #     # Transform orientations: add robot yaw to trajectory yaw
    #     desired_rpy = desired_rpy_yaw.copy()
    #     desired_rpy[:, 2] += robot_yaw  # Add robot yaw to trajectory yaw
    #     desired_rot_mat = np.array([pin.rpy.rpyToMatrix(rpy) for rpy in desired_rpy])  # (n_horizon, 3, 3)

    #     # Build wheel state trajectory from MPC command (in yaw-aligned frame)
    #     desired_wheel_pos_yaw = desired_q_trb_traj[:, [6, 7, 8, 10, 11, 12]].reshape(n_horizon, 2, 3)  # (n_horizon, 2, 3)
    #     desired_wheel_vel_yaw = desired_v_trb_traj[:, [6, 7, 8, 10, 11, 12]].reshape(n_horizon, 2, 3)  # (n_horizon, 2, 3)
        
    #     # Transform wheel positions and velocities to world frame
    #     desired_wheel_pos = np.array([[R_yaw @ desired_wheel_pos_yaw[h, leg] + robot_pos 
    #                                    for leg in range(2)] for h in range(n_horizon)])  # (n_horizon, 2, 3)
    #     desired_wheel_vel = np.array([[R_yaw @ desired_wheel_vel_yaw[h, leg] 
    #                                    for leg in range(2)] for h in range(n_horizon)])  # (n_horizon, 2, 3)

    #     # Transform GRF to world frame
    #     desired_grf_yaw = desired_grf_traj.reshape(n_horizon, 2, 3)
    #     desired_grf = np.array([[R_yaw @ desired_grf_yaw[h, leg] 
    #                              for leg in range(2)] for h in range(n_horizon)])  # (n_horizon, 2, 3)
        
    #     # Transform object position and  velocity to world frame
    #     desired_object_pos = np.array([R_yaw @ p + robot_pos for p in desired_p_object_traj])  # (n_horizon, 3)
    #     desired_object_vel = np.array([R_yaw @ v for v in desired_v_object_traj])  # (n_horizon, 3)

    #     # Overwrite visualization variables
    #     self.vis_torso_pos = desired_pos
    #     self.vis_torso_vel = desired_lin_vel
    #     self.vis_torso_R = desired_rot_mat

    #     self.vis_wheel_pos = desired_wheel_pos
    #     self.vis_wheel_vel = desired_wheel_vel

    #     self.vis_grf = desired_grf

    #     self.vis_object_pos = desired_object_pos
    #     self.vis_object_vel = desired_object_vel

    #     self.vis_traj = True
    #     self.vis_object_traj = True

    # def register_low_cmd_subscriber(self, topic):
    #     # Run superclass method
    #     Lcm2MujocoBridge.register_low_cmd_subscriber(self, topic)
    #     # Register additional MPC command subscriber
    #     temp = self.lc.subscribe("sliding_plan", self.mpc_command_handler)
    #     temp.set_queue_capacity(1)

    def register_low_state_subscriber(self, topic=None):
        if topic is None:
            topic = self.topic_state
        self.low_state_suber = self.lc.subscribe("tron1_wheeled_state", self.lcm_state_handler)
        self.low_state_suber.set_queue_capacity(1)
