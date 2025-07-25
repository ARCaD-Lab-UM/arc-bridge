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
        self.joint_offsets = np.array([0, 0.53 - 0.06, -0.55 - 0.54, 0,  
                                       0, 0.53 - 0.06, -0.55 - 0.54, 0])

    def parse_robot_specific_low_state(self):

        self.low_state.q_ob = self.mj_data.qpos[11:11+3]
        self.low_state.dq_ob = self.mj_data.qvel[11:11+3]