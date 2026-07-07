import mujoco
import numpy as np

from .lcm2mujoco_bridge import Lcm2MujocoBridge
from arc_bridge.utils import *


class UfactoryLite6Bridge(Lcm2MujocoBridge):
    def __init__(self, mj_model, mj_data, config):
        launch_args = getattr(config, "launch_args", None)
        in_replay_mode = bool(getattr(launch_args, "replay", False))
        if in_replay_mode:
            config.lcm_udp_multicast_group = "udpm://239.255.76.67:7667?ttl=1"
        super().__init__(mj_model, mj_data, config)
        self.in_replay_mode = in_replay_mode
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

    def update_motor_cmd(self):
        # Self-contained feed-forward qfrc_bias compensation: MuJoCo's generalized bias force (gravity + Coriolis/centrifugal, i.e. inverse dynamics at qacc=0).
        bias = np.asarray(self.mj_data.qfrc_bias)
        lo = self.mj_model.actuator_ctrlrange[:, 0]
        hi = self.mj_model.actuator_ctrlrange[:, 1]

        if self._lcm_cmd_daemon is not None:
            self._lcm_cmd_daemon.update()
            self._print_lcm_cmd_daemon()
            if self._lcm_cmd_daemon.is_error():
                # pure bias force comp if no live external command
                self.mj_data.ctrl[:] = np.clip(bias, lo, hi)
                return

        cmd = self._compute_delayed_low_cmd()
        kp = np.asarray(cmd.kp)
        kd = np.asarray(cmd.kd)
        # bias force comp added on top of any external PD/torque command
        tau = (np.asarray(cmd.qj_tau)
               + kp * (np.asarray(cmd.qj_pos) - np.asarray(self.low_state.qj_pos))
               + kd * (np.asarray(cmd.qj_vel) - np.asarray(self.low_state.qj_vel))
               + bias)
        self.mj_data.ctrl[:] = np.clip(tau, lo, hi)

    def lcm_state_handler(self, channel, data):
        if self.mj_data is None:
            return
        # In replay mode, parse_common_low_state is skipped on the sim thread, so there is no race over low_state.qj_*
        msg = self.low_state_type.decode(data)

        # Update mj_data for visualization
        self.mj_data.qpos[:6] = msg.qj_pos
        self.mj_data.qvel[:] = 0
        self.mj_data.act[:] = False
        self.mj_data.qacc_warmstart[:] = 0
        self.mj_data.ctrl[:] = 0
