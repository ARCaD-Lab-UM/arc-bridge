import mujoco
import numpy as np

from arc_bridge.state_estimators import FloatingBaseLinearStateEstimator
from .lcm2mujuco_bridge import Lcm2MujocoBridge
from arc_bridge.lcm_msgs import rabbit_state_t, rabbit_control_t
from arc_bridge.utils import *

class RabbitBridge(Lcm2MujocoBridge):
    def __init__(self, mj_model, mj_data, config):
        super().__init__(mj_model, mj_data, config)

        self.right_foot_name = "right_foot"
        self.left_foot_name = "left_foot"

        self.vis_se = True
        self.vis_pos_est = np.array([0, 0, 1])
        self.vis_vel_est = np.zeros(3)
        self.vis_R_body = np.eye(3)
        self.vis_box_size = [0.05, 0.05, 0.1]

        self.dt_se = 0.001
        self.height_init = 1.0
        KF_Q = np.diag([0.01, 0.01, 0.01, 0.01, 0.01, 0.01])
        KF_R = np.diag([0.01, 0.01, 0.01, 0.01])
        self.KF = FloatingBaseLinearStateEstimator(self.dt_se, KF_Q, KF_R, self.height_init)

    def parse_robot_specific_low_state(self):
        temp_inertia_matrix = np.zeros((self.mj_model.nv, self.mj_model.nv))
        mujoco.mj_fullM(self.mj_model, temp_inertia_matrix, self.mj_data.qM)
        self.low_state.inertia_mat = temp_inertia_matrix.tolist()
        self.low_state.bias_force = self.mj_data.qfrc_bias.tolist()

        self.update_kinematics()

        self.update_state_estimation()

    def update_kinematics(self):
        dq = np.zeros((self.mj_model.nv, ))

        right_foot_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_SITE, self.right_foot_name)
        right_foot_pos = self.mj_data.site_xpos[right_foot_id]
        J_foot_R = np.zeros((3, self.mj_model.nv))
        mujoco.mj_jacSite(self.mj_model, self.mj_data, J_foot_R, None, right_foot_id)
        dJ_foot_R = np.zeros((3, self.mj_model.nv))
        mujoco.mj_jacDot(self.mj_model, self.mj_data, dJ_foot_R, None, right_foot_pos, right_foot_id)
        dJdq_foot_R = dJ_foot_R @ dq

        left_foot_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_SITE, self.left_foot_name)
        left_foot_pos = self.mj_data.site_xpos[left_foot_id]
        J_foot_L = np.zeros((3, self.mj_model.nv))
        mujoco.mj_jacSite(self.mj_model, self.mj_data, J_foot_L, None, left_foot_id)

        pf = np.hstack((right_foot_pos, left_foot_pos))
        Jf = np.vstack((J_foot_R, J_foot_L))

        self.low_state.pf = pf.tolist()
        self.low_state.Jf = Jf.tolist()

    def update_state_estimation(self):
        self.vis_pos_est = np.array(self.low_cmd.se_pos)
        self.vis_vel_est = np.array(self.low_cmd.se_vel)
        EA = self.low_cmd.EA
        quat = Quaternion(*self.low_state.quaternion)
        self.vis_R_body = quat_to_rot(quat)

    def lcm_state_handler(self, channel, data):
        if self.mj_data == None:
            return
        msg = rabbit_state_t.decode(data)
