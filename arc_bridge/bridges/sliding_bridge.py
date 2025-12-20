import mujoco
import numpy as np
import pdb
import pinocchio as pin
from nav_msgs.msg import Odometry

from .lcm2mujuco_bridge import Lcm2MujocoBridge
from .tron1_wheeled_bridge import Tron1WheeledBridge
from arc_bridge.utils import *
from arc_bridge.state_estimators import SlideObjectFloatingBaseLinearStateEstimator
from arc_bridge.lcm_msgs import tron1_wheeled_plan_t, sliding_plan_t

class SlidingBridge(Tron1WheeledBridge):
    def __init__(self, mj_model, mj_data, config):
        super().__init__(mj_model, mj_data, config)

        self.vicon_slide_object_pos = np.zeros(3, dtype=float)
        self.vicon_slide_object_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
        self.vicon_slide_object_lin_vel_world = np.zeros(3, dtype=float)
        self.vicon_slide_object_ang_omega_world = np.zeros(3, dtype=float)
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

        if self.slide_object_use_kf:
            self.update_slide_object_state_estimation()
        else:
            self.mj_data.qpos[15:18] = self.low_state.pos_ob
            self.mj_data.qpos[18:22] = self.low_state.quat_ob
            self.mj_data.qvel[14:17] = self.low_state.vel_ob
            self.mj_data.qvel[17:20] = self.low_state.omega_ob

    def update_slide_object_state_estimation(self):
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
        # display as what the controller sees
        self.mj_data.qpos[15:18] = self.low_state.pos_ob
        self.mj_data.qpos[18:22] = self.low_state.quat_ob
        self.mj_data.qvel[14:17] = self.low_state.vel_ob
        self.mj_data.qvel[17:20] = self.low_state.omega_ob
        

    # def lcm_state_handler(self, channel, data):
    #     if self.mj_data is None:
    #         return

    #     super().lcm_state_handler(channel, data)

    def _vicon_slide_object_callback(self, msg: Odometry) -> None:
        stamp = float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) * 1e-9
        clock_now = self.vicon_ros2_client.node.get_clock().now()
        now_sec = clock_now.nanoseconds * 1e-9
        dt = 0.0 if stamp <= 0 else max(0.0, now_sec - stamp)

        pos_msg = msg.pose.pose.position
        pos = np.array([pos_msg.x, pos_msg.y, pos_msg.z], dtype=float)
        quat_msg = msg.pose.pose.orientation
        quat = Quaternion(quat_msg.w, quat_msg.x, quat_msg.y, quat_msg.z)
        R_body_to_world = quat_to_rot(quat)

        twist_msg = msg.twist.twist
        vel_body = np.array([twist_msg.linear.x, twist_msg.linear.y, twist_msg.linear.z], dtype=float)
        vel_world = R_body_to_world @ vel_body
        omega_body = np.array([twist_msg.angular.x, twist_msg.angular.y, twist_msg.angular.z], dtype=float)
        omega_world = R_body_to_world @ omega_body

        if not self.slide_object_use_kf:
            self.low_state.pos_ob = self.vicon_slide_object_pos = pos
            self.low_state.quat_ob = self.vicon_slide_object_quat = np.array([quat.w, quat.x, quat.y, quat.z], dtype=float)
            self.low_state.vel_ob = self.vicon_slide_object_lin_vel_world[:3] = vel_world
            self.low_state.omega_ob = self.vicon_slide_object_ang_omega_world[:3] = omega_world
            return

        acc_world = np.zeros(3, dtype=float) # assume zero acceleration

        with self.vicon_lock:
            acc_est = acc_world
            pos_now = pos + vel_world * dt + 0.5 * acc_est * dt * dt
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
            pos_now = pos + vel_world * dt + 0.5 * acc_est * dt * dt
            vel_now = vel_world + acc_est * dt
            meas_full_state = np.hstack((pos_now, vel_now, acc_est))

            self.vicon_slide_object_pos = pos_now
            self.vicon_slide_object_quat = np.array([quat.w, quat.x, quat.y, quat.z], dtype=float)
            self.vicon_slide_object_lin_vel_world[:3] = vel_now
            self.vicon_slide_object_ang_omega_world[:3] = omega_world

            self.slide_object_kf.correct(meas_full_state)

    def mpc_command_handler(self, channel, data):
        if self.mj_data == None:
            return
        # Get desired torso and wheel state from tron1_wheeled_plan topic
        # and visualize it in mujoco viewer


        # We need to overwrite vis_pos_est and vis_vel_est for visualization
        msg = sliding_plan_t.decode(data)

        # Extract MPC command trajectory
        n_horizon = msg.n_horizon
        desired_q_trb_traj = np.array(msg.qd_trb).reshape(n_horizon, 14)  # (n_horizon, 14)
        desired_v_trb_traj = np.array(msg.dqd_trb).reshape(n_horizon, 14)  # (n_horizon, 14)
        desired_grf_traj = np.array(msg.lambda_des).reshape(n_horizon, 6)  # (n_horizon, 6)
        desired_p_object_traj = np.array(msg.qd_ob).reshape(n_horizon, 3)  # (n_horizon, 3)
        desired_v_object_traj = np.array(msg.dqd_ob).reshape(n_horizon, 3)  # (n_horizon, 3)

        # otherwise in yaw-aligned frame
        traj_in_global_frame = True 
        robot_pos = np.array(self.low_state.position[:3], dtype=float)
        robot_pos[2] = 0.0  # ignore height
        robot_yaw = self.low_state.rpy[2]
        R_yaw = np.array([
            [np.cos(robot_yaw), -np.sin(robot_yaw), 0],
            [np.sin(robot_yaw),  np.cos(robot_yaw), 0],
            [0, 0, 1]
        ])
        if traj_in_global_frame:
            robot_pos = robot_pos*0
            R_yaw = np.array([
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1]
            ])

        # Build floating base state trajectory from MPC command (in yaw-aligned frame)
        desired_pos_yaw = desired_q_trb_traj[:, :3]  # (n_horizon, 3)
        desired_rpy_yaw = desired_q_trb_traj[:, [5, 4, 3]]  # (n_horizon, 3) - revert ypr back to rpy order
        desired_lin_vel_yaw = desired_v_trb_traj[:, :3]  # (n_horizon, 3)
        # desired_ang_vel = desired_v_trb_traj[:, [5, 4, 3]]  # (n_horizon, 3) - revert ypr back to rpy order

        # Transform positions from yaw-aligned frame to world frame
        desired_pos = np.array([R_yaw @ pos + robot_pos for pos in desired_pos_yaw])  # (n_horizon, 3)
        
        # Transform velocities from yaw-aligned frame to world frame
        desired_lin_vel = np.array([R_yaw @ vel for vel in desired_lin_vel_yaw])  # (n_horizon, 3)
        
        # Transform orientations: add robot yaw to trajectory yaw
        desired_rpy = desired_rpy_yaw.copy()
        desired_rpy[:, 2] += robot_yaw  # Add robot yaw to trajectory yaw
        desired_rot_mat = np.array([pin.rpy.rpyToMatrix(rpy) for rpy in desired_rpy])  # (n_horizon, 3, 3)

        # Build wheel state trajectory from MPC command (in yaw-aligned frame)
        desired_wheel_pos_yaw = desired_q_trb_traj[:, [6, 7, 8, 10, 11, 12]].reshape(n_horizon, 2, 3)  # (n_horizon, 2, 3)
        desired_wheel_vel_yaw = desired_v_trb_traj[:, [6, 7, 8, 10, 11, 12]].reshape(n_horizon, 2, 3)  # (n_horizon, 2, 3)
        
        # Transform wheel positions and velocities to world frame
        desired_wheel_pos = np.array([[R_yaw @ desired_wheel_pos_yaw[h, leg] + robot_pos 
                                       for leg in range(2)] for h in range(n_horizon)])  # (n_horizon, 2, 3)
        desired_wheel_vel = np.array([[R_yaw @ desired_wheel_vel_yaw[h, leg] 
                                       for leg in range(2)] for h in range(n_horizon)])  # (n_horizon, 2, 3)

        # Transform GRF to world frame
        desired_grf_yaw = desired_grf_traj.reshape(n_horizon, 2, 3)
        desired_grf = np.array([[R_yaw @ desired_grf_yaw[h, leg] 
                                 for leg in range(2)] for h in range(n_horizon)])  # (n_horizon, 2, 3)
        
        # Transform object position and  velocity to world frame
        desired_object_pos = np.array([R_yaw @ p + robot_pos for p in desired_p_object_traj])  # (n_horizon, 3)
        desired_object_vel = np.array([R_yaw @ v for v in desired_v_object_traj])  # (n_horizon, 3)

        # Overwrite visualization variables
        self.vis_torso_pos = desired_pos
        self.vis_torso_vel = desired_lin_vel
        self.vis_torso_R = desired_rot_mat

        self.vis_wheel_pos = desired_wheel_pos
        self.vis_wheel_vel = desired_wheel_vel

        self.vis_grf = desired_grf

        self.vis_object_pos = desired_object_pos
        self.vis_object_vel = desired_object_vel

        self.vis_traj = True


    def register_low_cmd_subscriber(self, topic):
        # Run superclass method
        Lcm2MujocoBridge.register_low_cmd_subscriber(self, topic)
        # Register additional MPC command subscriber
        temp = self.lc.subscribe("sliding_plan", self.mpc_command_handler)
        temp.set_queue_capacity(1)
