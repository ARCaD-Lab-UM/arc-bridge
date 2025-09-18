import mujoco
import pdb
import numpy as np
# import pinocchio as pin

from .lcm2mujuco_bridge import Lcm2MujocoBridge
from arc_bridge.utils import *

class SlidingBridge(Lcm2MujocoBridge):
    def __init__(self, mj_model, mj_data, config):
        super().__init__(mj_model, mj_data, config)
        # Override motor offsets (rad)
        self.joint_offsets = np.array([0, 0.53, -0.55-0.54, 0,  
                                       0, 0.53, -0.55-0.54, 0])
        
    def parse_robot_specific_low_state(self):
        # pdb.set_trace()
        self.low_state.pos_ob = self.mj_data.qpos[15:15+3]
        self.low_state.quat_ob = self.mj_data.qpos[15+3:15+7]
        self.low_state.vel_ob = self.mj_data.qvel[14:14+3]
        self.low_state.omega_ob = self.mj_data.qvel[14+3:14+6]   
        