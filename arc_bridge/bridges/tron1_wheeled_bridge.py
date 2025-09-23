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

        self.raw_msg_position = np.zeros(3, dtype=float) # [x, y, z] world frame
        self.raw_msg_velocity = np.zeros(3, dtype=float) # [vx, vy, vz] world frame

        # Visualization
        self.vis_se = True # override default flag
        self.vis_pos_est = np.array([0, 0, 0.75]) # initial pos (height)
        self.vis_vel_est = np.zeros(3)
        self.vis_R_body = np.eye(3)
        self.vis_box_size = [0.1, 0.1, 0.08]
        self.vel_body = np.zeros(3) # body velocity in body frame

    def update_state_estimation(self):
        # low state position and velocity in world frame
        pos_world = np.array(self.raw_msg_position, dtype=float)
        vel_world = np.array(self.raw_msg_velocity, dtype=float)
        R_body_to_world = quat_to_rot(Quaternion(*self.low_state.quaternion))

        # inverse rotation to get body frame velocity
        v_body = R_body_to_world.T @ vel_world

        # visualization
        self.vis_pos_est = pos_world
        self.vis_vel_est = vel_world
        self.vis_R_body = R_body_to_world
        self.vel_body = v_body

        # update state
        self.low_state.position[:] = pos_world
        self.low_state.velocity[:] = vel_world

    # def parse_robot_specific_low_state(self):
        # if len(self.mj_data.qpos) > 11:
        #     self.low_state.q_ob = self.mj_data.qpos[11:11+3]
        #     self.low_state.dq_ob = self.mj_data.qvel[11:11+3]

    def lcm_state_handler(self, channel, data):
        if self.mj_data == None:
            return
        # Get state msg from robot SDK topic
        msg = eval(self.topic_state+"_t").decode(data)

        # Partially update low_state
        # self.low_state.qj_pos[:] = (np.array(msg.qj_pos) + self.joint_offsets).tolist() # ! This one needs offsets since it should match with controller's model
        self.low_state.qj_pos[:] = msg.qj_pos
        self.low_state.qj_vel[:] = msg.qj_vel
        self.low_state.qj_tau[:] = msg.qj_tau
        self.low_state.acceleration[:] = msg.acceleration
        self.low_state.omega[:] = msg.omega
        self.low_state.quaternion[:] = msg.quaternion
        self.low_state.rpy[:] = msg.rpy

        # prepare for state estimation
        self.raw_msg_position[:] = np.asarray(msg.position, dtype=float)[:3]
        self.raw_msg_velocity[:] = np.asarray(msg.velocity, dtype=float)[:3]

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
