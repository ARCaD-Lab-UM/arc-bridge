import mujoco
import numpy as np

from .lcm2mujuco_bridge import Lcm2MujocoBridge
from arc_bridge.lcm_msgs import rabbit_state_t, rabbit_control_t
from arc_bridge.utils import *

class RabbitBridge(Lcm2MujocoBridge):
    def __init__(self, mj_model, mj_data, config):
        super().__init__(mj_model, mj_data, config)
        
    def parse_robot_specific_low_state(self):
        temp_inertia_matrix = np.zeros((self.mj_model.nv, self.mj_model.nv))
        mujoco.mj_fullM(self.mj_model, temp_inertia_matrix, self.mj_data.qM)
        self.low_state.inertia_mat = temp_inertia_matrix.tolist()
        self.low_state.bias_force = self.mj_data.qfrc_bias.tolist()

        # print(self.low_cmd.qj_tau)

    def lcm_state_handler(self, channel, data):
        if self.mj_data == None:
            return
        msg = rabbit_state_t.decode(data)
