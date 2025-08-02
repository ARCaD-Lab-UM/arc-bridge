import lcm
import time
import mujoco
import numpy as np

from threading import Thread
from abc import abstractmethod

from arc_bridge.utils import *
from arc_bridge.lcm_msgs import *

MOTOR_SENSOR_NUM = 3 # pos, vel, torque

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
                                     mujoco._enums.mjtObj.mjOBJ_SENSOR, i)
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
        self.lc = lcm.LCM()
        self.low_state = eval(self.topic_state+"_t")()
        self.low_cmd = eval(self.topic_cmd+"_t")()
        self.lcm_handle_thread = None

        self.low_cmd_received = False
        self.is_running = None

        # Gamepad controller
        self.gamepad = None
        self.gamepad_cmd = gamepad_cmd_t()
        self.topic_gamepad = "gamepad_cmd"
        try:
            self.gamepad = Gamepad(0.5, 0.5, np.pi/2)
            print("=> Gamepad found")
        except:
            print("=> No gamepad found")

        # Joint zero pos offsets
        self.joint_offsets = np.zeros(self.num_motor)

        # State estimator visualization
        self.vis_se = False

    def lcm_cmd_handler(self, channel, data):
        if self.mj_data != None:
            self.low_cmd = eval(self.topic_cmd+"_t").decode(data)
            self.low_cmd_received = True

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

    def start_lcm_thread(self):
        self.is_running = True
        self.lcm_handle_thread = Thread(target=self.lcm_handle_loop, daemon=True)
        self.lcm_handle_thread.start()

    def stop_lcm_thread(self):
        self.is_running = False
        self.lcm_handle_thread.join()
        print("LCM thread exited")

    def parse_common_low_state(self):
        if self.mj_data == None:
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
            if self.num_foot_sensor > 1:
                self.low_state.foot_force[:] = self.mj_data.sensordata[self.dim_motor_sensor + 16:self.dim_motor_sensor + 16 + self.num_foot_sensor]
            else:
                self.low_state.foot_force = self.mj_data.sensordata[self.dim_motor_sensor + 16]

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
        self.lc.publish(topic, self.low_state.encode())

    def publish_gamepad_cmd(self):
        if self.gamepad == None:
            return

        cmd = self.gamepad.get_command()
        self.gamepad_cmd.timestamp = time.time_ns()
        self.gamepad_cmd.vx = cmd[0]
        self.gamepad_cmd.vy = cmd[1]
        self.gamepad_cmd.wz = cmd[2]
        self.gamepad_cmd.e_stop = cmd[3]
        self.lc.publish(self.topic_gamepad, self.gamepad_cmd.encode())

    def update_motor_cmd(self):
        for i in range(self.num_motor):
            motor_torque_limits = self.mj_model.actuator_ctrlrange[i]
            motor_torque = self.low_cmd.qj_tau[i] +\
                           self.low_cmd.kp[i] * (self.low_cmd.qj_pos[i] - self.low_state.qj_pos[i]) +\
                           self.low_cmd.kd[i] * (self.low_cmd.qj_vel[i] - self.low_state.qj_vel[i])
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
