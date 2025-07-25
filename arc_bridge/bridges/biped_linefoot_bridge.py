import numpy as np
import mujoco

from .lcm2mujuco_bridge import Lcm2MujocoBridge
from arc_bridge.utils import *

class BipedLinefootBridge(Lcm2MujocoBridge):
    def __init__(self, mj_model, mj_data, config):
        super().__init__(mj_model, mj_data, config)


    def parse_robot_specific_low_state(self):
        
        # Send inertia matrix and bias force
        temp_inertia_mat = np.zeros((self.mj_model.nv, self.mj_model.nv))
        mujoco.mj_fullM(self.mj_model, temp_inertia_mat, self.mj_data.qM)
        self.low_state.inertia_mat = temp_inertia_mat.tolist()
        self.low_state.bias_force = self.mj_data.qfrc_bias

        # Parse J and dJdq
        # np.set_printoptions(precision=4, suppress=True, linewidth=1000)
        dq = np.concatenate((self.low_state.velocity, self.low_state.omega, self.low_state.qj_vel), axis=0)
        
        # Torso jacobians
        # J_tor
        torso_id = mujoco.mj_name2id(self.mj_model, mujoco._enums.mjtObj.mjOBJ_BODY, "torso") # id = 2
        torso_pos = self.low_state.position
        J_tor = np.zeros((6, self.mj_model.nv))
        mujoco.mj_jac(self.mj_model, self.mj_data, J_tor[:3, :], J_tor[3:, :], torso_pos, torso_id)
        self.low_state.J_tor = J_tor.tolist()
        # assert(np.linalg.norm((J_tor @ dq)[3:6] - omega_world) < 1e-6)

        # dJdq_tor
        dJ_tor = np.zeros((6, self.mj_model.nv))
        mujoco.mj_jacDot(self.mj_model, self.mj_data, dJ_tor[:3, :], dJ_tor[3:, :], torso_pos, torso_id)
        dJdq_tor = dJ_tor @ dq
        self.low_state.dJdq_tor = dJdq_tor.tolist()

        # Left foot jacobians
        left_ankle_id = mujoco.mj_name2id(self.mj_model, mujoco._enums.mjtObj.mjOBJ_BODY, "left_ankle")

        # J_lin_heel_L
        left_heel_id = mujoco.mj_name2id(self.mj_model, mujoco._enums.mjtObj.mjOBJ_SITE, "left_foot_heel")
        left_heel_pos = self.mj_data.site_xpos[left_heel_id]
        J_lin_heel_L = np.zeros((3, self.mj_model.nv))
        mujoco.mj_jac(self.mj_model, self.mj_data, J_lin_heel_L, None, left_heel_pos, left_ankle_id)

        # dJdq_heel_L
        dJ_heel_L = np.zeros((3, self.mj_model.nv))
        mujoco.mj_jacDot(self.mj_model, self.mj_data, dJ_heel_L, None, left_heel_pos, left_ankle_id)
        dJdq_heel_L = dJ_heel_L @ dq

        # J_lin_toe_L
        left_toe_id = mujoco.mj_name2id(self.mj_model, mujoco._enums.mjtObj.mjOBJ_SITE, "left_foot_toe")
        left_toe_pos = self.mj_data.site_xpos[left_toe_id]
        J_lin_toe_L = np.zeros((3, self.mj_model.nv))
        mujoco.mj_jac(self.mj_model, self.mj_data, J_lin_toe_L, None, left_toe_pos, left_ankle_id)

        # dJdq_toe_L
        dJ_toe_L = np.zeros((3, self.mj_model.nv))
        mujoco.mj_jacDot(self.mj_model, self.mj_data, dJ_toe_L, None, left_heel_pos, left_ankle_id)
        dJdq_toe_L = dJ_toe_L @ dq

        # Right foot jacobians
        right_ankle_id = mujoco.mj_name2id(self.mj_model, mujoco._enums.mjtObj.mjOBJ_BODY, "right_ankle")

        # J_lin_heel_R
        right_heel_id = mujoco.mj_name2id(self.mj_model, mujoco._enums.mjtObj.mjOBJ_SITE, "right_foot_heel")
        right_heel_pos = self.mj_data.site_xpos[right_heel_id]
        J_lin_heel_R = np.zeros((3, self.mj_model.nv))
        mujoco.mj_jac(self.mj_model, self.mj_data, J_lin_heel_R, None, right_heel_pos, right_ankle_id)

        # dJdq_heel_R
        dJ_heel_R = np.zeros((3, self.mj_model.nv))
        mujoco.mj_jacDot(self.mj_model, self.mj_data, dJ_heel_R, None, right_heel_pos, right_ankle_id)
        dJdq_heel_R = dJ_heel_R @ dq

        # J_lin_toe_R
        right_toe_id = mujoco.mj_name2id(self.mj_model, mujoco._enums.mjtObj.mjOBJ_SITE, "right_foot_toe")
        right_toe_pos = self.mj_data.site_xpos[right_toe_id]
        J_lin_toe_R = np.zeros((3, self.mj_model.nv))
        mujoco.mj_jac(self.mj_model, self.mj_data, J_lin_toe_R, None, right_toe_pos, right_ankle_id)

        # dJdq_toe_R
        dJ_toe_R = np.zeros((3, self.mj_model.nv))
        mujoco.mj_jacDot(self.mj_model, self.mj_data, dJ_toe_R, None, right_heel_pos, right_ankle_id)
        dJdq_toe_R = dJ_toe_R @ dq

        J_gc = np.concatenate((J_lin_heel_R, J_lin_toe_R, J_lin_heel_L, J_lin_toe_L), axis=0)
        self.low_state.J_gc = J_gc.tolist()

        dJdq_gc = np.concatenate((dJdq_heel_R, dJdq_toe_R, dJdq_heel_L, dJdq_toe_L), axis=0)
        self.low_state.dJdq_gc = dJdq_gc.tolist()

        p_gc = np.concatenate((right_heel_pos, right_toe_pos, left_heel_pos, left_toe_pos), axis=0)
        self.low_state.p_gc = p_gc.tolist()
