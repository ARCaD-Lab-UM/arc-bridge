import copy
import lcm
import time
import mujoco
import numpy as np

from collections import deque

from threading import Thread
from abc import abstractmethod

from arc_bridge.utils import *
from arc_bridge.lcm_msgs import *

MOTOR_SENSOR_NUM = 3  # pos, vel, torque


class Lcm2MujocoBridge:

    def __init__(self, mj_model, mj_data, config):
        self.mj_model = mj_model
        self.mj_data = mj_data

        self.topic_state = config.robot_state_topic
        self.topic_cmd = config.robot_cmd_topic
        self.config = config

        self.num_motor = self.mj_model.nu
        self.num_body_state = self.mj_model.nq - self.num_motor
        self.num_joint_state = self.num_motor
        self.dim_motor_sensor = MOTOR_SENSOR_NUM * self.num_motor
        self.have_imu = False
        self.have_frame_sensor = False
        self.have_foot_sensor = False
        self.num_foot_sensor = 0
        self.dt = self.mj_model.opt.timestep
        self.start_time = time.time_ns()

        # Check sensors
        for i in range(self.dim_motor_sensor, self.mj_model.nsensor):
            name = mujoco.mj_id2name(self.mj_model,
                                     mujoco.mjtObj.mjOBJ_SENSOR, i)
            if name == "imu_quat":
                self.have_imu = True
            if name == "frame_pos":
                self.have_frame_sensor = True
            if "foot" in name and "force" in name:
                self.have_foot_sensor = True
                self.num_foot_sensor += 1

        self.print_scene_info()
        print(f"=> have imu: {self.have_imu}")
        print(f"=> have frame sensor: {self.have_frame_sensor}")
        print(f"=> num foot sensor: {self.num_foot_sensor}")

        # LCM messages
        udp_multicast_group = getattr(self.config, "lcm_udp_multicast_group", None)
        self.lc = lcm.LCM(udp_multicast_group) if udp_multicast_group else lcm.LCM()
        self.low_state_type = eval(self.topic_state + "_t")
        self.low_state = self.low_state_type()
        self.low_cmd_type = eval(self.topic_cmd + "_t")
        self.low_cmd = self.low_cmd_type()
        self.control_delay = max(self.config.control_delay, 0.0)
        self.sensor_delay = max(self.config.sensor_delay, 0.0)
        control_delay_s = self.control_delay / 1000.0
        sensor_delay_s = self.sensor_delay / 1000.0
        self.control_delay_steps = int(round(control_delay_s / self.dt)) if self.dt > 0 and control_delay_s > 0 else 0
        self.sensor_delay_steps = int(round(sensor_delay_s / self.dt)) if self.dt > 0 and sensor_delay_s > 0 else 0
        self._control_buffer = self._init_delay_buffer(self.low_cmd, self.control_delay_steps) if self.control_delay_steps > 0 else None
        self._sensor_buffer = self._init_delay_buffer(self.low_state, self.sensor_delay_steps) if self.sensor_delay_steps > 0 else None
        self._most_recent_low_cmd = self.low_cmd
        self._most_recent_low_state = self.low_state
        if self.control_delay_steps:
            print(f"=> control delay: {self.control_delay_steps} steps ({self.control_delay:.3f}ms)")
        if self.sensor_delay_steps:
            print(f"=> sensor delay: {self.sensor_delay_steps} steps ({self.sensor_delay:.3f}ms)")
        self.lcm_handle_thread = None

        self.low_cmd_received = False
        self.is_running = None

        # LCM command daemon
        self._lcm_cmd_print_interval_s = 1.0
        self._lcm_cmd_print_throttle = PrintThrottle(self._lcm_cmd_print_interval_s)
        self._lcm_cmd_spy = None
        # self._lcm_cmd_spy = DaemonSpy(window_size=200)
        disable_daemon = getattr(self.config.launch_args, 'disable_daemon', False)
        if disable_daemon:
            self._lcm_cmd_daemon = None
        else:
            self._lcm_cmd_daemon = Daemon(
                DaemonConfig(
                    set_online_jitter_time_ms=self.config.lcm_cmd_online_jitter_time_ms,
                    set_offline_time_ms=self.config.lcm_cmd_offline_time_ms,
                    owner_id="lcm-control",
                ),
                spy=self._lcm_cmd_spy
            )
        self._lcm_cmd_seen = False

        # Gamepad controller
        self.gamepad = None

        # Joint zero pos offsets
        self.joint_offsets = np.zeros(self.num_motor)

        # State estimator visualization
        self.vis_se = False
        self.vis_traj = False

        # Upper-level display overlays (driven by a robot-specific display topic).
        # Bridges that support them flip these on; main.py skips the overlay when False.
        self.vis_ghost = False
        self.vis_fk = False

    def _init_delay_buffer(self, source, delay_steps):
        length = max(delay_steps + 1, 1)
        return deque([copy.deepcopy(source) for _ in range(length)], maxlen=length)

    def _clone_low_cmd(self, source):
        return copy.deepcopy(source)

    def _clone_low_state(self, source):
        return copy.deepcopy(source)

    def _compute_delayed_low_cmd(self):
        if self.control_delay_steps > 0:
            self._control_buffer.append(self._clone_low_cmd(self.low_cmd))
            cmd_to_return = self._control_buffer[0]
        else:
            cmd_to_return = self.low_cmd
        
        self._most_recent_low_cmd = cmd_to_return
        return cmd_to_return

    def _print_lcm_cmd_daemon(self) -> None:
        if self._lcm_cmd_daemon is None or self._lcm_cmd_spy is None:
            return
        if self._lcm_cmd_print_interval_s <= 0.0:
            return
        self._lcm_cmd_print_throttle.interval_s = self._lcm_cmd_print_interval_s
        freq_hz = float(self._lcm_cmd_spy.frequency_hz)
        min_delta_ms = float(self._lcm_cmd_spy.min_delta_ms)
        max_delta_ms = float(self._lcm_cmd_spy.max_delta_ms)
        status = "ERROR" if self._lcm_cmd_daemon.is_error() else "OK"
        prefix = "WARNING" if self._lcm_cmd_daemon.is_error() else "INFO"
        self._lcm_cmd_print_throttle.print(f"[{prefix}] LCM cmd daemon {status}: freq={freq_hz:.1f} Hz, min_dt={min_delta_ms:.6f} ms, max_dt={max_delta_ms:.6f} ms")

    def lcm_cmd_handler(self, channel, data):
        if self.mj_data is None:
            return

        self.low_cmd = self.low_cmd_type.decode(data)
        self.low_cmd_received = True
        self._lcm_cmd_seen = True
        if self._lcm_cmd_daemon is not None:
            self._lcm_cmd_daemon.reload()

    @abstractmethod
    def lcm_state_handler(self, channel, data):
        raise NotImplementedError("Subclass must implement this for replay mode")

    def register_low_cmd_subscriber(self, topic=None):
        if topic is None:
            topic = self.topic_cmd
        self.low_cmd_suber = self.lc.subscribe(topic, self.lcm_cmd_handler)
        self.low_cmd_suber.set_queue_capacity(1)

    def register_low_state_subscriber(self, topic=None):
        if topic is None:
            topic = self.topic_state
        self.low_state_suber = self.lc.subscribe(topic, self.lcm_state_handler)
        self.low_state_suber.set_queue_capacity(1)

    def lcm_handle_loop(self):
        while self.is_running:
            self.lc.handle_timeout(0)

        print("LCM thread exited")

    def start_lcm_thread(self):
        self.is_running = True
        self.lcm_handle_thread = Thread(target=self.lcm_handle_loop, daemon=True)
        self.lcm_handle_thread.start()

    def stop_lcm_thread(self):
        self.is_running = False
        self.lcm_handle_thread.join()

    def start_gamepad_thread(self):
        try:
            self.gamepad = Gamepad(vel_scale_x=2.0, vel_scale_y=0.5, vel_scale_rot=np.pi, scale_pitch=np.pi/2.0, triggers_scale=1.0)
            self.gamepad_cmd = gamepad_cmd_t()
            self.topic_gamepad = "gamepad_cmd"
            print("=> Gamepad found")
        except:
            print("=> No gamepad found")

    def parse_common_low_state(self):
        if self.mj_data is None:
            return -1

        # Assume each joint has a position, velocity, and torque sensor
        for i in range(self.num_motor):
            self.low_state.qj_pos[i] = self.mj_data.sensordata[i] + self.joint_offsets[i] \
                + np.random.normal(0, self.mj_model.sensor_noise[i])
            self.low_state.qj_vel[i] = self.mj_data.sensordata[i + self.num_motor] \
                + np.random.normal(0, self.mj_model.sensor_noise[i + self.num_motor])
            self.low_state.qj_tau[i] = self.mj_data.sensordata[i + 2 * self.num_motor] \
                + np.random.normal(0, self.mj_model.sensor_noise[i + 2 * self.num_motor])

        if self.have_frame_sensor:
            # Ground truth position and velocity readings in the world frame
            self.low_state.position[0] = self.mj_data.sensordata[self.dim_motor_sensor + 10]
            self.low_state.position[1] = self.mj_data.sensordata[self.dim_motor_sensor + 11]
            self.low_state.position[2] = self.mj_data.sensordata[self.dim_motor_sensor + 12]

            self.low_state.velocity[0] = self.mj_data.sensordata[self.dim_motor_sensor + 13]
            self.low_state.velocity[1] = self.mj_data.sensordata[self.dim_motor_sensor + 14]
            self.low_state.velocity[2] = self.mj_data.sensordata[self.dim_motor_sensor + 15]

        if self.have_foot_sensor:
            # Ground truth contact sensing
            self.low_state.foot_force[:] = self.mj_data.sensordata[self.dim_motor_sensor + 16:self.dim_motor_sensor + 16 + self.num_foot_sensor]

        if self.have_imu:
            quat = Quaternion(*self.mj_data.sensordata[self.dim_motor_sensor:self.dim_motor_sensor + 4])
            rpy = quat_to_rpy(quat)
            self.low_state.rpy[0] = rpy[0] + np.random.normal(0, self.mj_model.sensor_noise[self.dim_motor_sensor])
            self.low_state.rpy[1] = rpy[1] + np.random.normal(0, self.mj_model.sensor_noise[self.dim_motor_sensor])
            self.low_state.rpy[2] = rpy[2] + np.random.normal(0, self.mj_model.sensor_noise[self.dim_motor_sensor])
            self.low_state.quaternion[:] = rpy_to_quat(self.low_state.rpy).to_numpy().tolist()

            # Body frame angular rate and linear acceleration
            self.low_state.omega[:] = self.mj_data.sensordata[self.dim_motor_sensor + 4:self.dim_motor_sensor + 7]
            self.low_state.omega[0] += np.random.normal(0, self.mj_model.sensor_noise[self.dim_motor_sensor + 1])
            self.low_state.omega[1] += np.random.normal(0, self.mj_model.sensor_noise[self.dim_motor_sensor + 1])
            self.low_state.omega[2] += np.random.normal(0, self.mj_model.sensor_noise[self.dim_motor_sensor + 1])
            self.low_state.acceleration[:] = self.mj_data.sensordata[self.dim_motor_sensor + 7:self.dim_motor_sensor + 10]
            self.low_state.acceleration[0] += np.random.normal(0, self.mj_model.sensor_noise[self.dim_motor_sensor + 2])
            self.low_state.acceleration[1] += np.random.normal(0, self.mj_model.sensor_noise[self.dim_motor_sensor + 2])
            self.low_state.acceleration[2] += np.random.normal(0, self.mj_model.sensor_noise[self.dim_motor_sensor + 2])

            # self.mj_data.qvel[3:6] # this is Eular angle rate != omega_body
            # self.low_state.omega[:] = omega_body.tolist()
        return 0

    @abstractmethod
    def parse_robot_specific_low_state(self):
        pass

    def publish_low_state(self, topic=None, skip_common_state=False):
        if topic is None:
            topic = self.topic_state

        if not skip_common_state:
            if self.parse_common_low_state() < 0:
                return

        # Process robot kinematics and dynamics based on received common states
        self.parse_robot_specific_low_state()

        # Encode and publish robot states
        self.low_state.timestamp = time.time_ns()
        if self.sensor_delay_steps > 0:
            current_state = self._clone_low_state(self.low_state)
            self._sensor_buffer.append(current_state)
            state_to_publish = self._sensor_buffer[0]
        else:
            state_to_publish = self.low_state

        self._most_recent_low_state = state_to_publish
        self.lc.publish(topic, state_to_publish.encode())

    def publish_gamepad_cmd(self):
        if self.gamepad is None:
            return

        axis = self.gamepad.get_axis()
        params = self.gamepad.get_params()
        buttons = self.gamepad.get_buttons()
        lbrb = self.gamepad.get_lbrb()
        ljrj = self.gamepad.get_ljrj()
        lt, rt = self.gamepad.get_triggers()

        self.gamepad_cmd.timestamp = time.time_ns()
        self.gamepad_cmd.axis[:] = axis  # [L West, L North, R West, R North] for positive directions
        self.gamepad_cmd.e_stop = self.gamepad.get_estop()
        self.gamepad_cmd.params[:] = params[:]
        self.gamepad_cmd.btn_up = buttons[0]
        self.gamepad_cmd.btn_left = buttons[1]
        self.gamepad_cmd.btn_down = buttons[2]
        self.gamepad_cmd.btn_right = buttons[3]
        self.gamepad_cmd.btn_lb = lbrb[0]
        self.gamepad_cmd.btn_rb = lbrb[1]
        self.gamepad_cmd.btn_lstick = ljrj[0]
        self.gamepad_cmd.btn_rstick = ljrj[1]
        self.gamepad_cmd.lt = lt
        self.gamepad_cmd.rt = rt

        self.lc.publish(self.topic_gamepad, self.gamepad_cmd.encode())

    def update_motor_cmd(self):
        if self._lcm_cmd_daemon is not None:
            self._lcm_cmd_daemon.update()
            self._print_lcm_cmd_daemon()
            if self._lcm_cmd_daemon.is_error():
                self.mj_data.ctrl[:] = 0.0
                return
        cmd = self._compute_delayed_low_cmd()
        for i in range(self.num_motor):
            motor_torque_limits = self.mj_model.actuator_ctrlrange[i]
            motor_torque = cmd.qj_tau[i] +\
                           cmd.kp[i] * (cmd.qj_pos[i] - self.low_state.qj_pos[i]) +\
                           cmd.kd[i] * (cmd.qj_vel[i] - self.low_state.qj_vel[i])
            self.mj_data.ctrl[i] = np.clip(motor_torque, motor_torque_limits[0], motor_torque_limits[1])

    def print_scene_info(self):
        print(" ")

        print("<<------------- Link ------------->> ")
        for i in range(self.mj_model.nbody):
            name = mujoco.mj_id2name(self.mj_model,
                                     mujoco._enums.mjtObj.mjOBJ_BODY, i)
            if name:
                print(f"link_index: {i}, name: {name}, mass: {self.mj_model.body_mass[i]:.4f}, inertia: {self.mj_model.body_inertia[i]}")
        print(" ")

        print("<<------------- Joint ------------->> ")
        for i in range(self.mj_model.njnt):
            name = mujoco.mj_id2name(self.mj_model,
                                     mujoco._enums.mjtObj.mjOBJ_JOINT, i)
            if name:
                print("joint_index:", i, ", name:", name)
        print(" ")

        print("<<------------- Actuator ------------->>")
        for i in range(self.mj_model.nu):
            name = mujoco.mj_id2name(self.mj_model,
                                     mujoco._enums.mjtObj.mjOBJ_ACTUATOR, i)
            if name:
                print("actuator_index:", i, ", name:", name)
        print(" ")

        print("<<------------- Sensor ------------->>")
        index = 0
        for i in range(self.mj_model.nsensor):
            name = mujoco.mj_id2name(self.mj_model,
                                     mujoco._enums.mjtObj.mjOBJ_SENSOR, i)
            if name:
                print("sensor_index:", index, ", name:", name, ", dim:",
                      self.mj_model.sensor_dim[i])
            index = index + self.mj_model.sensor_dim[i]
        print(" ")
