import mujoco
import numpy as np
# import pinocchio as pin

from arc_bridge.state_estimators import FloatingBaseLinearStateEstimator, MovingWindowFilter
from .lcm2mujuco_bridge import Lcm2MujocoBridge
from arc_bridge.lcm_msgs import tron1_wheeled_state_t, tron1_wheeled_control_t
from arc_bridge.utils import *

class Tron1WheeledBridge(Lcm2MujocoBridge):
    def __init__(self, mj_model, mj_data, config):
        super().__init__(mj_model, mj_data, config)
        # Override motor offsets (rad)
        self.joint_offsets = np.array([0, 0.53, -0.55-0.54, 0,  
                                       0, 0.53, -0.55-0.54, 0])
        
        # sensor from odemetry module in robot onboard computer
        self.ode_msg_position = np.zeros(3, dtype=float) # [x, y, z] world frame
        self.ode_msg_velocity = np.zeros(3, dtype=float) # [vx, vy, vz] world frame

        # Visualization (the red box and the blue arrow)
        self.vis_se = True # override default flag
        self.vis_pos_est = np.array([0, 0, 0.75]) # initial pos (height)
        self.vis_vel_est = np.zeros(3)
        self.vis_R_body = np.eye(3)
        self.vis_box_size = [0.1, 0.1, 0.08]
        self.vel_body = np.zeros(3) # body velocity in body frame

        # State Estimator
        self.use_odemetry = False
        self.height_init = 0.7
        self.dt_estimator = 0.0005 # 2kHz
        # Process noise (px, py, pz, vx, vy, vz)
        KF_Q = np.diag([0.02, 0.02, 0.02, 0.02, 0.02, 0.02]) 
        # Measurement noise (pz, vx, vy, vz)
        KF_R = np.diag([0.01, 0.01, 0.01, 0.01])
        self.KF = FloatingBaseLinearStateEstimator(self.dt_estimator, KF_Q, KF_R, self.height_init)

        # kinematics params
        self.l1 = 0.077
        self.l2 = 0.3
        self.p_abad = np.array([[0.0556, -0.105, 0.2002],
                    [0.0556,  0.105, 0.2002]]).T # left and right, transposed
        self.wheel_radius = 0.127
        self.wheel_y_offset = 0.0435


        # contact positions and velocity
        self.pc_body_frame = np.zeros(6) # 2 feet, each has (x, y, z)
        self.vc_body_frame = np.zeros(6)

    def update_state_estimation(self):
        # low state position and velocity in world frame
        if self.use_odemetry:
            pos_world = np.array(self.ode_msg_position, dtype=float)
            vel_world = np.array(self.ode_msg_velocity, dtype=float)
            R_body_to_world = quat_to_rot(Quaternion(*self.low_state.quaternion))

            # inverse rotation to get body frame velocity
            v_body = R_body_to_world.T @ vel_world

            # visualization
            self.vis_pos_est = pos_world
            self.vis_vel_est = vel_world
            self.vis_R_body = R_body_to_world
            self.vel_body = v_body

            # update state (sending to controller)
            self.low_state.position[:] = pos_world
            self.low_state.velocity[:] = vel_world
        else:
            # use KF to estimate position and velocity
            # input acceleration in body frame from IMU
            acc_body = np.array(self.low_state.acceleration, dtype=float)
            # rotate to world frame
            R_body_to_world = quat_to_rot(Quaternion(*self.low_state.quaternion))
            acc_world = R_body_to_world @ acc_body - np.array([0, 0, 9.81]) # remove gravity

            self.KF.predict(u=acc_world)
            # use odemetry velocity and height for correction
            meas = np.array([self.ode_msg_position[2], 
                               self.ode_msg_velocity[0], 
                               self.ode_msg_velocity[1], 
                               self.ode_msg_velocity[2]], dtype=float)
            self.KF.correct(meas)

            # update state (sending to controller)
            self.low_state.position[:] = self.KF.x[:3]
            self.low_state.velocity[:] = self.KF.x[3:]

            # visualization
            self.vis_pos_est = self.KF.x[:3]
            self.vis_vel_est = self.KF.x[3:]
            self.vis_R_body = R_body_to_world
            self.vel_body = R_body_to_world.T @ self.KF.x[3:]

    
    def parse_robot_specific_low_state(self):
        # Used in simulation thread (update low_state from mj_data)
        pass
        

    def lcm_state_handler(self, channel, data):
        if self.mj_data == None:
            return
        # Get state msg from robot SDK topic
        msg = eval(self.topic_state+"_t").decode(data)

        # Partially update low_state
        self.low_state.qj_pos[:] = (np.array(msg.qj_pos) + self.joint_offsets).tolist() # ! This one needs offsets since it should match with controller's model
        self.low_state.qj_pos[:] = msg.qj_pos
        self.low_state.qj_vel[:] = msg.qj_vel
        self.low_state.qj_tau[:] = msg.qj_tau
        self.low_state.acceleration[:] = msg.acceleration
        self.low_state.omega[:] = msg.omega
        self.low_state.quaternion[:] = msg.quaternion
        self.low_state.rpy[:] = msg.rpy

        # prepare for state estimation (receive the odemetry msg)
        self.ode_msg_position[:] = np.asarray(msg.position, dtype=float)[:3]
        self.ode_msg_velocity[:] = np.asarray(msg.velocity, dtype=float)[:3]

        self.update_state_estimation()

        # Update mj_data for visualization
        self.mj_data.qpos[0] = msg.position[0] # self.low_state.position[0]
        self.mj_data.qpos[1] = msg.position[1] # self.low_state.position[1]
        self.mj_data.qpos[2] = msg.position[2] # self.low_state.position[2]
        self.mj_data.qpos[3] = msg.quaternion[0]
        self.mj_data.qpos[4] = msg.quaternion[1]
        self.mj_data.qpos[5] = msg.quaternion[2]
        self.mj_data.qpos[6] = msg.quaternion[3]
        self.mj_data.qpos[7:7+8] = msg.qj_pos - self.joint_offsets
        self.mj_data.qvel[:] = 0

    def calculate_contact_pos_and_vel(self):
        for leg_i in range(2):
            # TODO: check the polarity of knee and hip angle
            qj_leg = self.low_state.qj_pos[leg_i*4:(leg_i+1)*4]
            a = 2*self.l2*np.cos(qj_leg[2]/2)
            p_hip2foot_vec_xz = np.array([a*np.sin(qj_leg[1]+qj_leg[2]/2), 
                                 -a*np.cos(qj_leg[1]+qj_leg[2]/2)])
            p_abad2foot_vec_xz = p_hip2foot_vec_xz + np.array([-self.l1, 0])
            p_abad2foot_vec = np.array([p_abad2foot_vec_xz[0], 0, p_abad2foot_vec_xz[1]])
            # Rotate the vector around x axis by qj_leg[1] angle
            Rx = np.array([
                [1, 0, 0],
                [0, np.cos(qj_leg[0]), -np.sin(qj_leg[0])],
                [0, np.sin(qj_leg[0]),  np.cos(qj_leg[0])]
            ])
            p_abad2foot_vec = Rx @ p_abad2foot_vec
            p_foot_body = self.p_abad[:,leg_i] + p_abad2foot_vec
            pc_body = p_foot_body.copy()
            if leg_i == 0: # left leg
                pc_body += np.array([0, self.wheel_y_offset*np.cos(qj_leg[0]), 
                                             self.wheel_y_offset*np.sin(qj_leg[0])])
                pc_body += np.array([0, self.wheel_radius*np.sin(qj_leg[0]), 
                                             -self.wheel_radius*np.cos(qj_leg[0])])
            else: # right leg
                pc_body += np.array([0, -self.wheel_y_offset*np.cos(qj_leg[0]), 
                                             self.wheel_y_offset*np.sin(qj_leg[0])])
                pc_body += np.array([0, self.wheel_radius*np.sin(qj_leg[0]), 
                                             -self.wheel_radius*np.cos(qj_leg[0])])
            self.pc_body_frame[leg_i*3:(leg_i+1)*3] = pc_body
        
        