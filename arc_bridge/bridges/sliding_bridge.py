import mujoco
import pdb
import numpy as np
import pinocchio as pin

from .lcm2mujuco_bridge import Lcm2MujocoBridge
from arc_bridge.utils import *
from arc_bridge.lcm_msgs import tron1_wheeled_plan_t

class SlidingBridge(Lcm2MujocoBridge):
    def __init__(self, mj_model, mj_data, config):
        super().__init__(mj_model, mj_data, config)
        # Override motor offsets (rad)
        self.joint_offsets = np.array([0, 0.53, -0.55-0.54, 0,  
                                       0, 0.53, -0.55-0.54, 0])
        self.vis_traj = False
        self.vis_wheel_pos = None
        self.vis_wheel_vel = None
        self.vis_grf = None
        
    def parse_robot_specific_low_state(self):
        # pdb.set_trace()
        self.low_state.pos_ob = self.mj_data.qpos[15:15+3]
        self.low_state.quat_ob = self.mj_data.qpos[15+3:15+7]
        self.low_state.vel_ob = self.mj_data.qvel[14:14+3]
        self.low_state.omega_ob = self.mj_data.qvel[14+3:14+6]   



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

        # show in global frame 
        robot_pos = np.array(self.low_state.position[:3], dtype=float)
        robot_pos[2] = 0.0  # ignore height
        robot_pos = robot_pos*0
        robot_yaw = self.low_state.rpy[2]
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
        temp = self.lc.subscribe("sliding_plan", self.mpc_command_handler)
        temp.set_queue_capacity(1)
        