import mujoco
import numpy as np

from .lcm2mujuco_bridge import Lcm2MujocoBridge
from arc_bridge.utils import *

class Arm2linkBridge(Lcm2MujocoBridge):
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
        dq = self.low_state.qj_vel

        # End-effector jacobians
        ee_id = mujoco.mj_name2id(self.mj_model, mujoco._enums.mjtObj.mjOBJ_BODY, "end_effector")

        # J_gc
        ee_pos = self.mj_data.xpos[ee_id]
        J_gc = np.zeros((3, self.mj_model.nv))
        mujoco.mj_jac(self.mj_model, self.mj_data, J_gc, None, ee_pos, ee_id)

        # dJdq_gc
        dJ_gc = np.zeros((3, self.mj_model.nv))
        mujoco.mj_jacDot(self.mj_model, self.mj_data, dJ_gc, None, ee_pos, ee_id)
        dJdq_gc = dJ_gc @ dq

        self.low_state.J_gc = J_gc.tolist()

        self.low_state.dJdq_gc = dJdq_gc.tolist()

        p_gc = ee_pos
        self.low_state.p_gc = p_gc.tolist()

        joint_1_id = mujoco.mj_name2id(self.mj_model, mujoco._enums.mjtObj.mjOBJ_BODY, "link_1")
        joint_2_id = mujoco.mj_name2id(self.mj_model, mujoco._enums.mjtObj.mjOBJ_BODY, "link_2")

        p_joint_1 = self.mj_data.xpos[joint_1_id]
        p_joint_2 = self.mj_data.xpos[joint_2_id]

        p_joint = np.concatenate((p_joint_1[None, :], p_joint_2[None, :]), axis=0)
        self.low_state.p_joint = p_joint.tolist()

        self.low_state.base_ft_data[:3] = self.mj_data.sensordata[12:15].tolist()
        self.low_state.base_ft_data[3:] = self.mj_data.sensordata[15:18].tolist()
        self.low_state.qj_tau_3dim[:3] = self.mj_data.sensordata[6:9].tolist()
        self.low_state.qj_tau_3dim[3:] = self.mj_data.sensordata[9:12].tolist()

    def update_motor_cmd(self):
        # Update motor torques
        super().update_motor_cmd()
        # Parse required external force and location
        ext_force_cartesian = self.low_cmd.ext_force
        # Get end-effector and joint 2 positions in world frame
        ee_id = mujoco.mj_name2id(self.mj_model, mujoco._enums.mjtObj.mjOBJ_BODY, "end_effector")
        link_2_id = mujoco.mj_name2id(self.mj_model, mujoco._enums.mjtObj.mjOBJ_BODY, "link_2")
        ee_pos = self.mj_data.xpos[ee_id]
        joint_2_pos = self.mj_data.xpos[link_2_id]
        # Compute the equivalent force and torque in cartesian at end-effector
        p_ee_to_j2 = joint_2_pos - ee_pos
        p_ee_to_ext_point = p_ee_to_j2 * (1 - self.low_cmd.ext_alpha)
        ext_torque_cartesian = np.cross(p_ee_to_ext_point, ext_force_cartesian)
        # Apply the force and torque to end-effector
        self.mj_data.xfrc_applied[ee_id] = np.concatenate((ext_force_cartesian, ext_torque_cartesian))
        # print("====================================")
        # print(f"ext_force_cartesian: {ext_force_cartesian}")
        # print(f"ext_torque_cartesian: {ext_torque_cartesian}")
