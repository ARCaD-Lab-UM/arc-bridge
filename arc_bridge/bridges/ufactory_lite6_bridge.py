import mujoco
import numpy as np

from .lcm2mujoco_bridge import Lcm2MujocoBridge
from arc_bridge.utils import *


class UfactoryLite6Bridge(Lcm2MujocoBridge):
    def __init__(self, mj_model, mj_data, config):
        super().__init__(mj_model, mj_data, config)
        self.ee_body_id = mujoco.mj_name2id(self.mj_model, mujoco._enums.mjtObj.mjOBJ_BODY, "link6")

    def parse_robot_specific_low_state(self):
        # Joint-space inertia matrix
        temp_inertia_mat = np.zeros((self.mj_model.nv, self.mj_model.nv))
        mujoco.mj_fullM(self.mj_model, temp_inertia_mat, self.mj_data.qM)
        self.low_state.inertia_mat = temp_inertia_mat.tolist()
        self.low_state.bias_force = self.mj_data.qfrc_bias.tolist()

        # End-effector translational Jacobian and its time-derivative-times-qvel
        ee_pos = self.mj_data.xpos[self.ee_body_id]
        J_ee = np.zeros((3, self.mj_model.nv))
        mujoco.mj_jac(self.mj_model, self.mj_data, J_ee, None, ee_pos, self.ee_body_id)

        dJ_ee = np.zeros((3, self.mj_model.nv))
        mujoco.mj_jacDot(self.mj_model, self.mj_data, dJ_ee, None, ee_pos, self.ee_body_id)
        dJdq_ee = dJ_ee @ self.low_state.qj_vel

        self.low_state.J_ee = J_ee.tolist()
        self.low_state.dJdq_ee = dJdq_ee.tolist()
        self.low_state.p_ee = ee_pos.tolist()
