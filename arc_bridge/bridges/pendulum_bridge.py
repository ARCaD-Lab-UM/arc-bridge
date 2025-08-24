import mujoco
import numpy as np

from .lcm2mujuco_bridge import Lcm2MujocoBridge
from arc_bridge.lcm_msgs import pendulum_state_t, pendulum_control_t
from arc_bridge.utils import *


class PendulumBridge(Lcm2MujocoBridge):
    def __init__(self, mj_model, mj_data, config):
        super().__init__(mj_model, mj_data, config)

    def parse_robot_specific_low_state(self):
        # Send inertia matrix and bias force
        temp_inertia_mat = np.zeros((self.mj_model.nv, self.mj_model.nv))
        mujoco.mj_fullM(self.mj_model, temp_inertia_mat, self.mj_data.qM)
        self.low_state.inertia_mat = temp_inertia_mat.tolist()
        self.low_state.bias_force = self.mj_data.qfrc_bias.tolist()

    def lowStateHandler(self, channel, data):
        if self.mj_data is None:
            return
        msg = pendulum_state_t.decode(data)
        # Update mujoco state for visualization
        self.mj_data.qpos[0] = msg.qj_pos[0]
        self.mj_data.qvel[0] = msg.qj_vel[0]
        # Update low_state
        self.low_state.qj_pos[0] = msg.qj_pos[0]
        self.low_state.qj_vel[0] = msg.qj_vel[0]
        self.low_state.inertia_mat = msg.inertia_mat
        self.low_state.bias_force = msg.bias_force
