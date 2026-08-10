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

        # Internal desired trajectory state.
        # These are shaped by the incoming speed/mvacc limits and then fed to the computed-torque law as (q_des, qd_des, qdd_des).
        self._q_des = None                          # desired joint position (nv,)
        self._qd_des = np.zeros(self.mj_model.nv)   # desired joint velocity (nv,)

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

    def _full_inertia(self):
        # Dense joint-space inertia matrix M(q) (nv, nv).
        M = np.zeros((self.mj_model.nv, self.mj_model.nv))
        mujoco.mj_fullM(self.mj_model, M, self.mj_data.qM)
        return M

    @staticmethod
    def _rate_limit(v_prev, v_target, accel_limit, dt):
        # Move v_prev toward v_target with per-element |delta v| <= accel_limit*dt.
        v_prev = np.asarray(v_prev, dtype=float)
        v_target = np.asarray(v_target, dtype=float)
        if accel_limit <= 0.0 or dt <= 0.0:
            return v_target.copy()  # no accel limit -> jump straight to target velocity
        dv = np.clip(v_target - v_prev, -accel_limit * dt, accel_limit * dt)
        return v_prev + dv

    def _servo_profile(self, q_des, qd_des, q_target, v_max, a_max, dt):
        # Trapezoidal (velocity/accel-limited) follower toward q_target.
        # Returns the next (q_des, qd_des); decelerates so it stops cleanly at the target.
        q_des = np.asarray(q_des, dtype=float)
        qd_des = np.asarray(qd_des, dtype=float)
        q_target = np.asarray(q_target, dtype=float)
        err = q_target - q_des
        # Fastest speed from which we can still brake to a stop within |err|.
        v_stop = np.sqrt(2.0 * max(a_max, 1e-9) * np.abs(err))
        v_cmd = np.sign(err) * np.minimum(v_max, v_stop)
        qd_new = self._rate_limit(qd_des, v_cmd, a_max, dt)
        q_new = q_des + qd_new * dt
        # Snap when essentially there to avoid dithering around the target.
        close = np.abs(q_target - q_new) < 1e-4
        q_new = np.where(close, q_target, q_new)
        qd_new = np.where(close, 0.0, qd_new)
        return q_new, qd_new

    def update_motor_cmd(self):
        # Computed torque control:  tau = M(q) * q_ddot_ref + bias + tau_ff.
        # q_ddot_ref = q_ddot_des + Kv*(qd_des - qd) + Kp*(q_des - q),
        # where bias = qfrc_bias = C(q,qd)*qd + g(q).
        # The desired trajectory (q_des, qd_des, q_ddot_des) is shaped from the command speed/mvacc limits.
        nu = self.mj_model.nu
        lo = self.mj_model.actuator_ctrlrange[:, 0]
        hi = self.mj_model.actuator_ctrlrange[:, 1]
        bias = np.asarray(self.mj_data.qfrc_bias)
        dt = self.dt

        have_live_cmd = True
        if self._lcm_cmd_daemon is not None:
            self._lcm_cmd_daemon.update()
            self._print_lcm_cmd_daemon()
            if self._lcm_cmd_daemon.is_error():
                have_live_cmd = False

        # Init the internal desired state from the current configuration.
        if self._q_des is None:
            self._q_des = np.array(self.mj_data.qpos[:nu], dtype=float)
            self._qd_des = np.zeros(nu)

        q = np.asarray(self.low_state.qj_pos)
        qd = np.asarray(self.low_state.qj_vel)

        if not have_live_cmd:
            # No fresh command: hold the last desired pose via computed torque.
            # With the default (all-zero) gains this reduces to pure bias comp.
            kp = np.asarray(self.low_cmd.kp)
            kd = np.asarray(self.low_cmd.kd)
            qacc_ref = kp * (self._q_des - q) - kd * qd
            tau = self._full_inertia() @ qacc_ref + bias
            self.mj_data.ctrl[:] = np.clip(tau, lo, hi)
            return

        cmd = self._compute_delayed_low_cmd()
        mode = int(getattr(cmd, "mode", 0))
        speed = float(getattr(cmd, "speed", 0.0))
        mvacc = float(getattr(cmd, "mvacc", 0.0))

        qd_des_prev = self._qd_des

        if mode == 1:
            # Joint velocity / jog: ramp desired velocity toward cmd.qj_vel under mvacc, integrate.
            qd_target = np.asarray(cmd.qj_vel, dtype=float)
            if speed > 0.0:
                qd_target = np.clip(qd_target, -speed, speed)
            qd_des = self._rate_limit(qd_des_prev, qd_target, mvacc, dt)
            q_des = self._q_des + qd_des * dt
        elif speed <= 0.0:
            # Position mode, no speed limit: track the target directly (backward compatible).
            q_des = np.asarray(cmd.qj_pos, dtype=float)
            qd_des = np.zeros(nu)
        else:
            # Position / servo: velocity/accel-limited trapezoidal move toward cmd.qj_pos.
            q_des, qd_des = self._servo_profile(self._q_des, qd_des_prev, cmd.qj_pos, speed, mvacc, dt)

        qdd_des = (qd_des - qd_des_prev) / dt if dt > 0 else np.zeros(nu)
        self._q_des = q_des
        self._qd_des = qd_des

        kp = np.asarray(cmd.kp)
        kd = np.asarray(cmd.kd)
        qacc_ref = qdd_des + kd * (qd_des - qd) + kp * (q_des - q)
        tau = self._full_inertia() @ qacc_ref + bias + np.asarray(cmd.qj_tau)
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
