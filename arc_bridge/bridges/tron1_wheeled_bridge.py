import mujoco
import numpy as np
import pdb
# import pinocchio as pin
from threading import Lock
from nav_msgs.msg import Odometry # ROS2 Vicon
import pinocchio as pin

from arc_bridge.state_estimators import FloatingBaseLinearStateEstimator, Tron1WheeledFloatingBaseLinearStateEstimator, MovingWindowFilter, OnlineAverage
from .lcm2mujuco_bridge import Lcm2MujocoBridge
from arc_bridge.lcm_msgs import tron1_wheeled_state_t, tron1_wheeled_control_t, tron1_wheeled_plan_t
from arc_bridge.lcm_msgs import sliding_state_t, sliding_control_t
from arc_bridge.utils import *
from .vicon_ros2_client import ViconRos2Client


class Tron1WheeledBridge(Lcm2MujocoBridge):
    def __init__(self, mj_model, mj_data, config):
        super().__init__(mj_model, mj_data, config)

        # ROS2 Vicon bridge
        self.vicon_lock = Lock()
        self.vicon_tron1_pos = np.zeros(3, dtype=float)
        self.vicon_tron1_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
        self.vicon_tron1_lin_vel_world = np.zeros(3, dtype=float)
        self.vicon_tron1_ang_omega_world = np.zeros(3, dtype=float)
        self.T_vicon_tron1_meas_to_center = np.eye(4, dtype=float)

        launch_args = getattr(self.config, "launch_args", None)
        self.in_replay_mode = bool(getattr(launch_args, "replay", False)) if launch_args else False
        self.vicon_ros2_client = None
        if self.in_replay_mode:
            self.vicon_ros2_client = ViconRos2Client(node_name="arc_bridge_vicon_listener")
            self.vicon_ros2_client.start()
            self.vicon_ros2_client.subscribe_tron1(callback=self._vicon_tron1_callback, topic="/odometry/tron1")

        # Override motor offsets (rad)
        self.joint_offsets = np.array([0, 0.53, -0.55-0.54, 0,  
                                       0, 0.53, -0.55-0.54, 0])
        
        # Visualization for the KF output (the red box and the blue arrow)
        self.vis_se = True # override default flag
        self.vis_pos_est = np.array([0, 0, 0.75]) # initial pos (height)
        self.vis_vel_est = np.zeros(3)
        self.vis_R_body = np.eye(3)
        self.vis_box_size = [0.1, 0.2, 0.08]
        self.vel_body = np.zeros(3) # body velocity in body frame

        self.vis_traj = False
        self.vis_wheel_pos = None
        self.vis_wheel_vel = None
        self.vis_grf = None

        # State Estimator
        self.height_init = 0.7
        self.dt_estimator = 0.001 # 1kHz
        # Process noise (px, py, pz, vx, vy, vz, ax, ay, az)
        self.KF_Q = np.diag([0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]) 
        # Measurement noise (px, py, pz, vx, vy, vz, ax, ay, az)
        self.KF_R = np.diag([0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]) 
        self.KF = Tron1WheeledFloatingBaseLinearStateEstimator(self.dt_estimator, self.KF_Q, self.KF_R, self.height_init)
        # heuristic measurement noise scaling based on vicon delay; R = (1 + alpha * dt) * R0 only on pos and vel
        self.vicon_delay_alpha = 20.0
        self.vicon_delay_scale_max = 10.0
        self.init_const_KF_R = self.KF_R.copy()  # store the initial R

        # kinematics params
        self.l1 = 0.077
        self.l2 = 0.3
        self.p_abad = np.array([[0.0556, 0.105, -0.2602],
                    [0.0556,  -0.105, -0.2602]]).T # left and right, transposed
        self.wheel_radius = 0.127
        self.wheel_y_offset = 0.0435


        # contact positions and velocity
        self.normal_terrain = np.array([0, 0, 1]) # flat ground normal
        self.p_abad2foot_vec_body = np.zeros((3,2)) # 2 legs
        self.pw_body_frame = np.zeros((3,2)) # 2 feet, each has (x, y, z)
        self.vw_body_frame = np.zeros((3,2))

        # IMU bias online average estimator: acc_body(3),  omega_body(3)
        self.calibrated = False
        self.imu_bias_average = OnlineAverage(dim=6) 
        # hardcode gravity bias for the imu
        self.gravity_add_bias = np.array([0, 0, 9.9945])
        self.imu_acc_bias_body = np.array([0.0, 0.0, 0.0]) # to be filled after enough data
        self.omega_bias_body = np.array([0.0, 0.0, 0.0]) # assume zero for gyro bias

        self.R_torso_global_rpy = np.eye(3) # to store the torso to global rotation using rpy(imu) orientation
        self.R_torso_global_quat = np.eye(3) # to store the torso to global rotation using quat(vicon) orientation
        self.Jacobian_foot_global =  np.zeros((3, 4, 2)) # to store the foot jacobian
        self.kf_mode = "vicon_with_kf" # "vicon_no_kf", "vicon_with_kf", "fk_with_kf"

        # Replay buffers to avoid races between LCM and simulation threads
        self._rx_state_position = np.zeros(3)
        self._rx_state_velocity = np.zeros(3)
        self._rx_state_available = False

    def _skip_calibration(self):
        # skip calibration in simulation
        self.calibrated = True
        self.gravity_add_bias = np.array([0, 0, 9.81])

    def update_state_estimation(self):
        # use KF to estimate position and velocity
        # input acceleration in body frame from IMU
        acc_body = np.array(self.low_state.acceleration, dtype=float)
        R_body_to_world = self.R_torso_global_quat        

        acc_world = R_body_to_world @ acc_body - self.gravity_add_bias # remove gravity

        # store the acc_world and acc_body in the buffer for calibration
        if not self.calibrated:
            # print(f"acc_world: {acc_world}")
            acc_body_bias = acc_body - R_body_to_world.T @ self.gravity_add_bias
            omega_body = np.array(self.low_state.omega, dtype=float)
            imu_sample = np.hstack((acc_body_bias, omega_body))
            self.imu_bias_average.update(imu_sample)
            if self.imu_bias_average._count >= 1e4: # 10k samples for 1kHz ~10s
                self.calibrated = True
                self.imu_acc_bias_body = self.imu_bias_average._mean[0:3]
                self.omega_bias_body = self.imu_bias_average._mean[3:6]
                print(f"IMU calibration done. Acc bias in body frame: {self.imu_acc_bias_body}")
                print(f"Gyro omega bias in body frame: {self.omega_bias_body}")
        else:
            if self.kf_mode == "vicon_no_kf":
                # visulization when no KF - UNCOMMENT this and COMMENT above to use
                self.vis_pos_est = np.array(self.low_state.position[:3], dtype=float)
                self.vis_vel_est = np.array(self.low_state.velocity[:3], dtype=float)
                self.vis_R_body = self.R_torso_global_rpy # R_body_to_world
            else:
                if self.kf_mode == "vicon_with_kf":
                    if self.in_replay_mode:
                        # in replay mode, correction happens in vicon callback
                        with self.vicon_lock:
                            self.KF.predict(u=np.zeros(1))
                    else:
                        self.KF.predict(u=np.zeros(1))
                        # use ground truth position/velocity for testing
                        pos = np.array(self.low_state.position[:3], dtype=float)
                        vel = np.array(self.low_state.velocity[:3], dtype=float)
                        meas_full_state = np.hstack((pos, vel, acc_world)) # add acc measurement
                        self._test_vicon_correction_in_simulation(meas_full_state)
                elif self.kf_mode == "fk_with_kf":
                    self.KF.predict(u=np.zeros(1))
                    # use the joint encoder and our FK for correction
                    meas = self.get_torso_height_and_velocity_meas_fk()
                    meas_full_state = np.hstack((meas, acc_world)) # add acc measurement
                    self.KF.correct(meas_full_state)

                # send to controller
                self.low_state.position[:] = self.KF.x[:3]
                self.low_state.velocity[:] = self.KF.x[3:6]

                # visualization of the state estimation (red box and blue arrow)
                self.vis_pos_est = self.KF.x[:3]
                self.vis_vel_est = self.KF.x[3:6]
                self.vis_R_body = self.R_torso_global_rpy # R_body_to_world
                self.vel_body = self.R_torso_global_rpy.T @ self.KF.x[3:6] # R_body_to_world

                # # visualization dt later
                # dt = 0.1 # change manually if needed
                # acc_est = acc_world
                # pos_now = self.KF.x[:3] + self.KF.x[3:6] * dt + 0.5 * acc_est * dt * dt
                # vel_now = self.KF.x[3:6] + acc_est * dt
                # self.vis_pos_est = pos_now
                # self.vis_vel_est = vel_now
                # self.vis_R_body = self.R_torso_global_rpy # R_body_to_world

    def parse_robot_specific_low_state(self):
        # This function is called in simulation thread

        # judge if in replay mode
        if not self.in_replay_mode and not self.calibrated:
            self._skip_calibration() # skip calibration when NOT in replay mode
        if self.in_replay_mode and self._rx_state_available:
            # use received position and velocity when in replay mode
            self.low_state.position[:] = self._rx_state_position
            self.low_state.velocity[:] = self._rx_state_velocity
        
        # update the R torso global (based on vicon quaternion)
        self.R_torso_global_quat = quat_to_rot(Quaternion(*self.low_state.quaternion))

        # update the R torso global (based on IMU rpy)
        quat_from_imu = rpy_to_quat(np.array(self.low_state.rpy, dtype=float))
        self.R_torso_global_rpy = quat_to_rot(quat_from_imu)

        self.update_state_estimation()

    def lcm_state_handler(self, channel, data):
        if self.mj_data == None:
            return
        # Get state msg from robot SDK topic
        msg = eval(self.topic_state+"_t").decode(data)

        # Update mj_data for visualization 
        self.mj_data.qpos[0] = msg.position[0] # robot in the mujoco viewer is vicon pose
        self.mj_data.qpos[1] = msg.position[1] # 
        self.mj_data.qpos[2] = msg.position[2] # 
        self.mj_data.qpos[3] = msg.quaternion[0]
        self.mj_data.qpos[4] = msg.quaternion[1]
        self.mj_data.qpos[5] = msg.quaternion[2]
        self.mj_data.qpos[6] = msg.quaternion[3]
        self.mj_data.qpos[7:7+8] = msg.qj_pos - self.joint_offsets # to macth with xml
        self.mj_data.qvel[:] = 0

        # Partially update low_state from the msg sent by robot SDK
        self.low_state.qj_pos[:] = msg.qj_pos
        self.low_state.qj_vel[:] = msg.qj_vel
        self.low_state.qj_tau[:] = msg.qj_tau
        self.low_state.acceleration[:] = msg.acceleration - self.imu_acc_bias_body
        self.low_state.omega[:] = msg.omega - self.omega_bias_body
        self.low_state.quaternion[:] = msg.quaternion
        self.low_state.rpy[:] = msg.rpy
        self._rx_state_position[:] = msg.position # copy because of write from another thread
        self._rx_state_velocity[:] = msg.velocity
        self._rx_state_available = True

    def get_torso_height_and_velocity_meas_fk(self):
        #  calculate the kinematics in body frame first
        self.calculate_wheel_pos_and_vel_body()

        # transfer to world frame
        R_body_to_world = self.R_torso_global_quat
        torso_omega_world = R_body_to_world @ np.array(self.low_state.omega, dtype=float)
        height_estimates = []
        velocity_estimates = []
        for leg_i in range(2):
            # compute the wheel-alighed frame first
            p_abad2foot_vec = R_body_to_world @ self.p_abad2foot_vec_body[:, leg_i]
            leg_plane = [R_body_to_world[:,0],  # x axis
                         p_abad2foot_vec]
            e_y = np.cross(leg_plane[0], leg_plane[1])
            e_y /= np.linalg.norm(e_y)
            e_x = np.cross(e_y, self.normal_terrain)
            e_x /= np.linalg.norm(e_x)
            e_z = np.cross(e_x, e_y)
            R_wheel_aligned = np.column_stack((e_x, e_y, e_z)) # rotation from wheel-aligned frame to world frame

            # compute contact point position in world frame
            p_torso2wheel_vec_world = R_body_to_world @ self.pw_body_frame[:, leg_i]
            p_torso2contact_vec_world = p_torso2wheel_vec_world - self.wheel_radius * e_z
            height_estimates.append( - p_torso2contact_vec_world[2])

            '''
            compute contact point velocity in world frame (three parts) [assume torso linear vel = 0]]
            1. torso angular
            2. vw_body_frame (due to the leg kinematics)
            3. wheel rotation part
            '''
            torso_vel_est = - np.cross(torso_omega_world, p_torso2wheel_vec_world)
            torso_vel_est -= R_body_to_world @ self.vw_body_frame[:, leg_i]
            dqj_leg = np.array(self.low_state.qj_vel[leg_i*4:(leg_i+1)*4])
            wheel_omega_world = torso_omega_world + R_body_to_world[:,0] * dqj_leg[0] + e_y * np.sum(dqj_leg[1:4]) 
            torso_vel_est -= np.cross(wheel_omega_world, - self.wheel_radius * e_z)
            velocity_estimates.append(torso_vel_est)
        height_estimates = np.array(height_estimates)
        velocity_estimates = np.vstack(velocity_estimates)
        height_mean = np.mean(height_estimates)
        velocity_mean = np.mean(velocity_estimates, axis=0)
        vicon_tron1_pos = np.array(self.low_state.position, dtype=float)
        return np.array([vicon_tron1_pos[0], vicon_tron1_pos[1], vicon_tron1_pos[2], velocity_mean[0], velocity_mean[1], velocity_mean[2]], dtype=float)

    def calculate_wheel_pos_and_vel_body(self):
        for leg_i in range(2):
            # compute the triangle in leg plane (xz plane)
            qj_leg = self.low_state.qj_pos[leg_i*4:(leg_i+1)*4]
            a_length = 2*self.l2*np.cos(qj_leg[2]/2)
            p_hip2foot_vec_xz = np.array([- a_length*np.sin(qj_leg[1]+qj_leg[2]/2), 
                                 - a_length*np.cos(qj_leg[1]+qj_leg[2]/2)])

            p_abad2foot_vec_xz = p_hip2foot_vec_xz + np.array([-self.l1, 0])
            p_abad2foot_vec = np.array([p_abad2foot_vec_xz[0], 0, p_abad2foot_vec_xz[1]])

            # Rotate the vector around x axis by abad angle
            Rx = np.array([
                [1, 0, 0],
                [0, np.cos(qj_leg[0]), -np.sin(qj_leg[0])],
                [0, np.sin(qj_leg[0]),  np.cos(qj_leg[0])]
            ])
            p_abad2foot_vec = Rx @ p_abad2foot_vec
            self.p_abad2foot_vec_body[:, leg_i] = p_abad2foot_vec
            p_foot_body = self.p_abad[:,leg_i] + p_abad2foot_vec
            pw_body = p_foot_body.copy()
            
            # Add wheel y offset
            pw_body +=  (-1)**leg_i * np.array([0, self.wheel_y_offset*np.cos(qj_leg[0]), 
                                             self.wheel_y_offset*np.sin(qj_leg[0])])
            self.pw_body_frame[:, leg_i] = pw_body

            # Compute wheel contact velocity
            dqj_leg = np.array(self.low_state.qj_vel[leg_i*4:(leg_i+1)*4])
            J_body = self.jacobian_p_foot_body(qj_leg, self.l2)
            wheel_com_vel = J_body @ dqj_leg
            self.vw_body_frame[:, leg_i] = wheel_com_vel

            J_global = self.R_torso_global_quat @ J_body
            self.Jacobian_foot_global[:,:,leg_i] = J_global # store the last one for external use

    def jacobian_p_foot_body(self,qj_leg, l2):
        """
        Compute J = d(p_foot_body)/d(q) for a single leg.
        qj_leg: array-like of length 4 -> [q1, q2, q3, q4] (radians)
        l1, l2 : link lengths (only l2 appears in this Jacobian)
        Returns: 3x4 numpy array
        """
        qj_leg = np.asarray(qj_leg).reshape(-1)
        q1, q2, q3, _ = qj_leg  # q4 doesn't affect this Jacobian (column is zeros)

        # helpers from your printout:
        # #1 == sin(q2 + q3/2), #2 == cos(q2 + q3/2)
        s = np.sin(q2 + q3/2.0)
        c = np.cos(q2 + q3/2.0)

        J = np.zeros((3, 4), dtype=float)

        # Row 1
        J[0, 0] = 0.0
        J[0, 1] = -l2 * (np.cos(q2 + q3) + np.cos(q2))
        J[0, 2] = -l2 * np.cos(q2 + q3)
        J[0, 3] = 0.0

        # Row 2
        J[1, 0] = l2* c * np.cos(q3/2.0) * np.cos(q1) * 2
        J[1, 1] = - l2 * s * np.cos(q3/2.0) * np.sin(q1) * 2
        J[1, 2] = - l2 * np.sin(q2+q3) * np.sin(q1)
        J[1, 3] = 0.0

        # Row 3
        J[2, 0] =  l2 * c * np.cos(q3/2.0) * np.sin(q1) * 2
        J[2, 1] =  l2 * s * np.cos(q3/2.0) * np.cos(q1) * 2
        J[2, 2] =  l2 * np.sin(q2+q3) * np.cos(q1)
        J[2, 3] = 0.0

        return J

    def ik_analy (self, p_wheel_com_body):
        """
        Inverse kinematics
        p_wheel_com_body: [3x2] array-like -> [[x_left, x_right], [y_left, y_right], [z_left, z_right]]
        Returns: [4x2] -> [q1, q2, q3, q4] (radians)
        """
        qj_legs_ik = np.zeros((4,2))
        for leg_i in range(2):
            p_wheel_com_b = p_wheel_com_body[:, leg_i]
            x, y, z = p_wheel_com_b - self.p_abad[:, leg_i]
            # compute the q_abad first
            len_abad2foot_yz_proj = np.sqrt(y^2 + z^2 - self.wheel_y_offset^2)
            q_abad = np.arctan2(y,-z) - (-1)^leg_i * np.arctan2(self.wheel_y_offset, len_abad2foot_yz_proj)

            # with the len_abad2foot, compute the hip and knee
            len_hip2foot_yz_proj = len_abad2foot_yz_proj.copy()
            len_hip2foot = np.sqrt((x+self.l1)^2 + len_hip2foot_yz_proj^2)
            q_knee = -2 * np.arccos(len_hip2foot / (2*self.l2))
            q_hip = np.arcsin(-x-self.l1, len_hip2foot) + (-q_knee)/2

            q_wheel = 0.0 # assume zero for now

            qj_legs_ik[:, leg_i] = np.array([q_abad, q_hip, q_knee, q_wheel])

        return qj_legs_ik

    def _apply_vicon_center_transform(self, pos, quat, vel_world, omega_world, T_meas_to_center):
        R_body_to_world = quat_to_rot(quat)
        R_offset = T_meas_to_center[:3, :3]
        t_offset = T_meas_to_center[:3, 3]
        r_world = R_body_to_world @ t_offset
        pos_center = pos + r_world
        vel_center = vel_world + np.cross(omega_world, r_world)
        if np.array_equal(R_offset, np.eye(3)):
            R_center_world = R_body_to_world
            quat_center = quat
        else:
            R_center_world = R_body_to_world @ R_offset
            quat_center = rot_to_quat(R_center_world)
        return pos_center, quat_center, R_center_world, vel_center, omega_world
    
    def _get_delay_scale(self, dt: float) -> float:
        """Based on vicon delay dt (in s), calculate the scaling factor for measurement noise R.
        scale = 1 + alpha * dt, capped at vicon_delay_scale_max.

        Returns:
            float: scaling factor
        """
        alpha = self.vicon_delay_alpha
        scale = 1.0 + alpha * dt
        if self.vicon_delay_scale_max is not None:
            scale = min(scale, self.vicon_delay_scale_max)
        return scale


    def _vicon_tron1_callback(self, msg: Odometry) -> None:
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

        # apply the center transform
        # pos, quat, R_body_to_world, vel_world, omega_world = self._apply_vicon_center_transform(pos, quat, vel_world, omega_world, self.T_vicon_tron1_meas_to_center)

        if self.kf_mode == "vicon_no_kf":
            self.vicon_tron1_pos[:] = pos
            self.vicon_tron1_quat[:] = np.array([quat.w, quat.x, quat.y, quat.z], dtype=float)
            self.vicon_tron1_lin_vel_world[:] = vel_world
            self.vicon_tron1_ang_omega_world[:] = omega_world
            # send to controller
            self.low_state.position[:] = self.vicon_tron1_pos
            self.low_state.velocity[:] = self.vicon_tron1_lin_vel_world
            return

        if self.calibrated and self._rx_state_available:
            acc_body = np.array(self.low_state.acceleration, dtype=float)
            acc_world = R_body_to_world @ acc_body - self.gravity_add_bias
        else:
            acc_world = np.zeros(3, dtype=float)

        with self.vicon_lock:
            acc_est = acc_world
            # pos_now = pos + vel_world * dt + 0.5 * acc_est * dt * dt
            pos_now = pos + (vel_world + acc_est * dt) * dt  # semi-implicit Euler
            vel_now = vel_world + acc_est * dt
            meas_full_state = np.hstack((pos_now, vel_now, acc_est))

            self.vicon_tron1_pos[:] = pos_now
            self.vicon_tron1_quat[:] = np.array([quat.w, quat.x, quat.y, quat.z], dtype=float)
            self.vicon_tron1_lin_vel_world[:] = vel_now
            self.vicon_tron1_ang_omega_world[:] = omega_world

            # Scale the measurement noise R based on delay dt
            scale = self._get_delay_scale(dt)
            self.KF.R[:, :] = self.init_const_KF_R
            self.KF.R[0:6, 0:6] *= scale  # only scale pos and vel

            self.KF.correct(meas_full_state)

    def _test_vicon_correction_in_simulation(self, meas: np.ndarray):
        dt = 0.0 # change manually if needed
        quat = Quaternion(*self.low_state.quaternion)
        R_body_to_world = quat_to_rot(quat)

        pos = np.array(meas[:3], dtype=float)
        vel_world = np.array(meas[3:6], dtype=float)
        omega_world = np.array(self.low_state.omega, dtype=float)
        acc_world = np.array(meas[6:9], dtype=float)

        with self.vicon_lock:
            acc_est = acc_world
            # pos_now = pos + vel_world * dt + 0.5 * acc_est * dt * dt
            pos_now = pos + (vel_world + acc_est * dt) * dt  # semi-implicit Euler
            vel_now = vel_world + acc_est * dt
            meas_full_state = np.hstack((pos_now, vel_now, acc_est))

            self.vicon_tron1_pos[:] = pos_now
            self.vicon_tron1_quat[:] = np.array([quat.w, quat.x, quat.y, quat.z], dtype=float)
            self.vicon_tron1_lin_vel_world[:] = vel_now
            self.vicon_tron1_ang_omega_world[:] = omega_world

            # Scale the measurement noise R based on delay dt
            scale = self._get_delay_scale(dt)
            # print(f"Simulation Vicon correction with dt={dt:.4f}s, scale={scale:.2f}")
            self.KF.R[:, :] = self.init_const_KF_R
            self.KF.R[0:6, 0:6] *= scale  # only scale pos and vel

            self.KF.correct(meas_full_state)

    def mpc_command_handler(self, channel, data):
        if self.mj_data == None:
            return
        # Get desired torso and wheel state from tron1_wheeled_plan topic
        # and visualize it in mujoco viewer

        self.vis_traj = True

        # We need to overwrite vis_pos_est and vis_vel_est for visualization
        msg = tron1_wheeled_plan_t.decode(data)

        # Extract MPC command trajectory
        n_horizon = msg.n_horizon
        desired_q_trb_traj = np.array(msg.qd_trb).reshape(n_horizon, 14)  # (n_horizon, 14)
        desired_v_trb_traj = np.array(msg.dqd_trb).reshape(n_horizon, 14)  # (n_horizon, 14)
        desired_grf_traj = np.array(msg.lambda_des).reshape(n_horizon, 6)  # (n_horizon, 6)

        # Get current robot state for transformation from yaw-aligned frame to world frame
        robot_pos = np.array(self.low_state.position[:3], dtype=float)
        robot_pos[2] = 0.0  # ignore height
        robot_yaw = self.low_state.rpy[2]
        R_yaw = np.array([
            [np.cos(robot_yaw), -np.sin(robot_yaw), 0],
            [np.sin(robot_yaw),  np.cos(robot_yaw), 0],
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

        # Overwrite visualization variables
        self.vis_torso_pos = desired_pos
        self.vis_torso_vel = desired_lin_vel
        self.vis_torso_R = desired_rot_mat

        self.vis_wheel_pos = desired_wheel_pos
        self.vis_wheel_vel = desired_wheel_vel

        self.vis_grf = desired_grf

    def register_low_cmd_subscriber(self, topic):
        # Run superclass method
        super().register_low_cmd_subscriber(topic)
        # Register additional MPC command subscriber
        temp = self.lc.subscribe("tron1_wheeled_plan", self.mpc_command_handler)
        temp.set_queue_capacity(1)
