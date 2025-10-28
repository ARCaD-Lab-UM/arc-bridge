import numpy as np
import time
import mujoco

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

    def lcm_state_handler(self, channel, data):
        if self.mj_data == None:
            return

        msg = eval(self.topic_state+"_t").decode(data)
        self.mj_data.qpos[0] = msg.position[0]
        # Subtract IMU offset to get torso height,
        # since torso center is the actual rotation center,
        # which is connected to the virtual joints.
        self.mj_data.qpos[1] = msg.position[2] - 0.08 * np.cos(msg.rpy[1])
        self.mj_data.qpos[2] = msg.rpy[1]
        self.mj_data.qpos[3:5] = msg.qj_pos
        self.mj_data.qvel[:] = 0
        self.mj_data.act[:] = False
        self.mj_data.qacc_warmstart[:] = 0
        self.mj_data.ctrl[:] = 0
