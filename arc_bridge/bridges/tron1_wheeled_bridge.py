import mujoco
import numpy as np
import pdb
# import pinocchio as pin

from arc_bridge.state_estimators import FloatingBaseLinearStateEstimator, Tron1WheeledFloatingBaseLinearStateEstimator, MovingWindowFilter, OnlineAverage
from .lcm2mujuco_bridge import Lcm2MujocoBridge
from arc_bridge.lcm_msgs import tron1_wheeled_state_t, tron1_wheeled_control_t
from arc_bridge.utils import *

class Tron1WheeledBridge(Lcm2MujocoBridge):
    def __init__(self, mj_model, mj_data, config):
        super().__init__(mj_model, mj_data, config)
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

        # State Estimator
        self.height_init = 0.7
        self.dt_estimator = 0.001 # 1kHz
        # Process noise (px, py, pz, vx, vy, vz, ax, ay, az)
        KF_Q = np.diag([0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]) 
        # Measurement noise (px, py, pz, vx, vy, vz, ax, ay, az)
        KF_R = np.diag([0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]) 
        self.KF = Tron1WheeledFloatingBaseLinearStateEstimator(self.dt_estimator, KF_Q, KF_R, self.height_init)

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
        self.calibration = False
        self.imu_bias_average = OnlineAverage(dim=6) 
        # hardcode gravity bias for the imu
        self.gravity_add_bias = np.array([0, 0, 9.9945])
        self.imu_acc_bias_body = np.array([0.0, 0.0, 0.0]) # to be filled after enough data
        self.omega_bias_body = np.array([0.0, 0.0, 0.0]) # assume zero for gyro bias

        self.R_torso_global_imu = np.eye(3) # to store the torso to global rotation using imu orientation
        self.R_torso_global_vicon = np.eye(3) # to store the torso to global rotation using vicon orientation
        self.Jacobian_foot_global =  np.zeros((3, 4, 2)) # to store the foot jacobian
        self.only_use_vicon_for_kf = True

    def remove_calibration_bias(self):
        self.calibration = True
        self.gravity_add_bias = np.array([0, 0, 9.81])

    def update_state_estimation(self):
        # use KF to estimate position and velocity
        # input acceleration in body frame from IMU
        acc_body = np.array(self.low_state.acceleration, dtype=float)
        R_body_to_world = self.R_torso_global_vicon        

        acc_world = np.zeros(3) # R_body_to_world @ acc_body - self.gravity_add_bias # remove gravity

        # store the acc_world and acc_body in the buffer for calibration
        if not self.calibration:
            # print(f"acc_world: {acc_world}")
            acc_body_bias = acc_body - R_body_to_world.T @ self.gravity_add_bias
            omega_body = np.array(self.low_state.omega, dtype=float)
            imu_sample = np.hstack((acc_body_bias, omega_body))
            self.imu_bias_average.update(imu_sample)
            if self.imu_bias_average._count >= 1e4: # 10k samples for 1kHz ~10s
                self.calibration = True
                self.imu_acc_bias_body = self.imu_bias_average._mean[0:3]
                self.omega_bias_body = self.imu_bias_average._mean[3:6]
                print(f"IMU calibration done. Acc bias in body frame: {self.imu_acc_bias_body}")
                print(f"Gyro omega bias in body frame: {self.omega_bias_body}")
        else:
            self.KF.predict(u=np.zeros(1))

            if self.only_use_vicon_for_kf:
                # use vicon position/velocity directly for correction
                pos = np.array(self.low_state.position[:3], dtype=float)
                vel = np.array(self.low_state.velocity[:3], dtype=float)
                meas = np.hstack((pos, vel))
            else:
                # use the joint encoder and our FK for correction
                meas = self.get_torso_height_and_velocity_meas_fk()
            meas_full_state = np.hstack((meas, acc_world)) # add acc measurement
            self.KF.correct(meas_full_state)

            # ! sending to controller
            self.low_state.position[:] = self.KF.x[:3]
            self.low_state.velocity[:] = self.KF.x[3:6]

            # visualization of the state estimation (red box and blue arrow)
            self.vis_pos_est = self.KF.x[:3]
            self.vis_vel_est = self.KF.x[3:6]
            self.vis_R_body = R_body_to_world
            self.vel_body = R_body_to_world.T @ self.KF.x[3:6]


    def parse_robot_specific_low_state(self):
        # Used in simulation thread (update low_state from mj_data)
        # reload the positions and velocities with KF output
        
        # update the R torso global (based on vicon quaternion)
        self.R_torso_global_vicon = quat_to_rot(Quaternion(*self.low_state.quaternion))

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
        self.low_state.position[:] = msg.position
        self.low_state.velocity[:] = msg.velocity

        # update the R torso global (based on IMU rpy)
        quat_from_imu = rpy_to_quat(np.array(self.low_state.rpy, dtype=float))
        self.R_torso_global_imu = quat_to_rot(quat_from_imu)


    def get_torso_height_and_velocity_meas_fk(self):
        #  calculate the kinematics in body frame first
        self.calculate_wheel_pos_and_vel_body()

        # transfer to world frame
        R_body_to_world = self.R_torso_global_vicon
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
        vicon_pos = np.array(self.low_state.position, dtype=float)
        return np.array([vicon_pos[0], vicon_pos[1], vicon_pos[2], velocity_mean[0], velocity_mean[1], velocity_mean[2]], dtype=float)

        
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

            J_global = self.R_torso_global_vicon @ J_body
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

