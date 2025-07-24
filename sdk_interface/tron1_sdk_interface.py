import os
import pdb
import signal
import sys
from threading import Thread
import time
from functools import partial
import traceback
import limxsdk.robot.Rate as Rate
import limxsdk.robot.Robot as Robot
import limxsdk.robot.RobotType as RobotType
import limxsdk.datatypes as datatypes
import lcm
import numpy as np

# Add the parent directory to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from lcm_types.robot_lcm import tron1_pointfoot_state_t, tron1_pointfoot_control_t
from lcm_types.robot_lcm import tron1_linefoot_state_t, tron1_linefoot_control_t
from state_estimators import MovingWindowFilter
from utils import *

# TODO safety check and soft estop
np.set_printoptions(precision=4, suppress=True, linewidth=1000)
ROBOT_TYPE = "LF"  # PF or LF

if ROBOT_TYPE == "PF":
    # ========== Point Foot Robot Constants ==========
    TOPIC_STATE = "tron1_pointfoot_state" # state to mujoco bridge
    TOPIC_CONTROL = "tron1_pointfoot_control".upper() # control from MATLAB

    DEFAULT_JOINT_OFFSETS = np.array([0, 0.53 - 0.06, -0.55 - 0.54,  # right leg
                                    0, 0.53 - 0.06, -0.55 - 0.54]) # left leg

    DEFAULT_JOINT_POS = np.array([0.0, 0.25, -0.5,  
                                0.0, 0.25, -0.5,])

    JOINT_IDX_2_MOTOR_IDX = np.array([3, 4, 5, 
                                    0, 1, 2])

    MOTOR_POLARITY = np.array([1,  1, -1, 
                            1, -1,  1])
elif ROBOT_TYPE == "LF":
    # ========== Line Foot Robot Constants ==========
    ROBOT_TYPE = "LF"
    TOPIC_STATE = "tron1_linefoot_state" # state to mujoco bridge
    TOPIC_CONTROL = "tron1_linefoot_control".upper() # control from MATLAB

    DEFAULT_JOINT_OFFSETS = np.array([0, 0.53, -1.04, 0.536,  # right leg
                                    0, 0.53, -1.04, 0.536]) # left leg

    DEFAULT_JOINT_POS = np.array([0.0, 0.2, -0.3, 0.4, 
                                  0.0, 0.2, -0.3, 0.4])

    JOINT_IDX_2_MOTOR_IDX = np.array([4, 5, 6, 7,
                                    0, 1, 2, 3])

    MOTOR_POLARITY = np.array([1,  1, -1, -1,
                            1, -1,  1, -1])
else:
    raise ValueError("Unsupported robot type!")


class RobotReceiver:
    def __init__(self):
        self.imu = None
        self.robot_state = None
        self.low_cmd = None
        self.is_running = True
        self.imu_counter = 0
        self.imu_timer = 0
        self.encoder_counter = 0
        self.encoder_timer = time.perf_counter()
        self.print_callback_rate = False

    # Callback function for receiving imu
    def imuDataCallback(self, imu: datatypes.ImuData):
        self.imu = imu
        if not self.print_callback_rate:
            return
        
        self.imu_counter += 1
        if self.imu_counter % 100 == 0:
            current_time = time.perf_counter()
            print(f"IMU rate: \t{100 / (current_time - self.imu_timer):.2f} Hz")
            self.imu_timer = current_time

    # Callback function for receiving robot state
    def robotStateCallback(self, robot_state: datatypes.RobotState):
        self.robot_state = robot_state
        if not self.print_callback_rate:
            return
        
        self.encoder_counter += 1
        if self.encoder_counter % 1000 == 0:
            current_time = time.perf_counter()
            print(f"Encoder rate: \t{1000 / (current_time - self.encoder_timer):.2f} Hz")
            self.encoder_timer = current_time

    # Callback function for receiving sensor joy data
    def sensorJoyCallback(self, sensor_joy: datatypes.SensorJoy):
        print("\n------\nsensor_joy:" + \
              "\n  stamp: " + str(sensor_joy.stamp) + \
              "\n  axes: " + str(sensor_joy.axes) + \
              "\n  buttons: " + str(sensor_joy.buttons))

    # Callback function for receiving diagnostic value
    def diagnosticValueCallback(self, diagnostic_value: datatypes.DiagnosticValue):
        print("\n------\ndiagnostic_value:" + \
              "\n  stamp: " + str(diagnostic_value.stamp) + \
              "\n  name: " + diagnostic_value.name + \
              "\n  level: " + str(diagnostic_value.level) + \
              "\n  code: " + str(diagnostic_value.code) + \
              "\n  message: " + diagnostic_value.message)

    def lowCmdHandler(self, channel, data):
        self.low_cmd = eval(TOPIC_CONTROL.lower() + "_t").decode(data)

    def lcmHandleThread(self):
        while self.is_running:
            lc.handle_timeout(0)

def sdk_state_to_lcm_state(sdk_state: datatypes.RobotState, lcm_state):
    for idx in range(len(lcm_state.qj_pos)):
        motor_idx = JOINT_IDX_2_MOTOR_IDX[idx]
        lcm_state.qj_pos[idx] = sdk_state.q[motor_idx] * MOTOR_POLARITY[motor_idx] + DEFAULT_JOINT_OFFSETS[idx]
        lcm_state.qj_vel[idx] = sdk_state.dq[motor_idx] * MOTOR_POLARITY[motor_idx]
        lcm_state.qj_tau[idx] = sdk_state.tau[motor_idx] * MOTOR_POLARITY[motor_idx]


def lcm_cmd_to_sdk_cmd(lcm_cmd, sdk_cmd: datatypes.RobotCmd):
    sdk_cmd.mode = [0.0 for _ in range(motor_number)] # torque mode
    sdk_cmd.q = [0.0 for _ in range(motor_number)]
    sdk_cmd.dq = [0.0 for _ in range(motor_number)]
    sdk_cmd.tau = [0.0 for _ in range(motor_number)]
    sdk_cmd.Kp = [0.0 for _ in range(motor_number)]
    sdk_cmd.Kd = [0.0 for _ in range(motor_number)]

    for idx in range(len(lcm_cmd.qj_pos)):
        motor_idx = JOINT_IDX_2_MOTOR_IDX[idx]
        sdk_cmd.q[motor_idx] = (lcm_cmd.qj_pos[idx] - DEFAULT_JOINT_OFFSETS[idx]) * MOTOR_POLARITY[motor_idx]
        sdk_cmd.dq[motor_idx] = lcm_cmd.qj_vel[idx] * MOTOR_POLARITY[motor_idx]
        sdk_cmd.tau[motor_idx] = lcm_cmd.qj_tau[idx] * MOTOR_POLARITY[motor_idx]
        sdk_cmd.Kp[motor_idx] = lcm_cmd.kp[idx]
        sdk_cmd.Kd[motor_idx] = lcm_cmd.kd[idx]

if __name__ == '__main__':
    if ROBOT_TYPE == "PF":
        # export ROBOT_TYPE=PF_TRON1A
        os.environ["ROBOT_TYPE"] = "PF_TRON1A"
    elif ROBOT_TYPE == "LF":
        # export ROBOT_TYPE=SF_TRON1A
        os.environ["ROBOT_TYPE"] = "SF_TRON1A"
    else:
        raise ValueError("Unsupported robot type!")

    os.system("echo $ROBOT_TYPE")

    # Create a Robot instance
    robot = Robot(RobotType.PointFoot)
    robot_ip = "10.192.1.2"

    # Initialize the robot with robot_ip
    if not robot.init(robot_ip):
        sys.exit()

    # Get motor number information
    motor_number = robot.getMotorNumber()
    print(f"Motor number: {motor_number}")
    assert motor_number == len(JOINT_IDX_2_MOTOR_IDX)

    # Create an instance of RobotReceiver to handle callbacks
    receiver = RobotReceiver()

    # Create partial functions for callbacks
    imuDataCallback = partial(receiver.imuDataCallback)
    robotStateCallback = partial(receiver.robotStateCallback)
    # sensorJoyCallback = partial(receiver.sensorJoyCallback)
    # diagnosticValueCallback = partial(receiver.diagnosticValueCallback)

    # Subscribe to robot state, sensor joy, and diagnostic value topics
    robot.subscribeImuData(imuDataCallback)
    robot.subscribeRobotState(robotStateCallback)
    # robot.subscribeSensorJoy(sensorJoyCallback)
    # robot.subscribeDiagnosticValue(diagnosticValueCallback)
    
    # Main loop to continuously publish robot commands
    rate = Rate(1000) # 1000 Hz

    # Initialize LCM
    lc = lcm.LCM("udpm://239.255.76.67:7667?ttl=1")
    low_cmd_suber = lc.subscribe(TOPIC_CONTROL, receiver.lowCmdHandler)
    low_cmd_suber.set_queue_capacity(1)

    lc_thread = Thread(target=receiver.lcmHandleThread, daemon=True)
    lc_thread.start()

    cmd_msg = datatypes.RobotCmd()
    # Set default values for control mode, joint positions, velocities, torques, Kp, and Kd
    # motor mode：0: torque； 1：speed； 2：position; default：0
    cmd_msg.mode = [2.0 for _ in range(motor_number)]
    cmd_msg.q = [0.0 for _ in range(motor_number)]
    cmd_msg.dq = [0.0 for _ in range(motor_number)]
    cmd_msg.tau = [0.0 for _ in range(motor_number)]
    cmd_msg.Kp = [40.0 for _ in range(motor_number)]
    cmd_msg.Kd = [2.0 for _ in range(motor_number)]

    # Wait for the robot to initialize
    while receiver.robot_state is None or receiver.imu is None:
        time.sleep(0.1)

    # Compensate the initial yaw
    quat_init = Quaternion(*receiver.imu.quat)
    rpy_init = quat_to_rpy(quat_init)
    yaw_init = rpy_init[2]
    print(f"Initial yaw: {yaw_init}")

    acc_init = np.array(receiver.imu.acc)
    print(f"Initial acceleration: {acc_init}")

    q_init = np.array(receiver.robot_state.q)
    # q_target = np.zeros(motor_number)
    q_target = DEFAULT_JOINT_POS[JOINT_IDX_2_MOTOR_IDX] * MOTOR_POLARITY
    print(f"q_init: {q_init}")
    print(f"q_target: {q_target}")

    num_init_steps = 4000
    print("=> Enter initial pose...")
    for step in range(num_init_steps):
        alpha = step / num_init_steps
        for i in range(motor_number):
            cmd_msg.q[i] = q_init[i] * (1 - alpha) + q_target[i] * alpha

        cmd_msg.stamp = time.time_ns()  # Set the timestamp
        robot.publishRobotCmd(cmd_msg)  # Publish the robot command
        rate.sleep()

    imu_acc_filter = MovingWindowFilter(window_size=10, dim=3)
    imu_gyro_filter = MovingWindowFilter(window_size=10, dim=3)
    joint_vel_filter = MovingWindowFilter(window_size=10, dim=8)


    print("=> Start to publish robot state...")

    # Capture Ctrl+C signal
    signal.signal(signal.SIGINT, signal.default_int_handler)
    try:
        while True:
            # Parse the received low_cmd
            low_cmd = receiver.low_cmd
            if low_cmd is not None:
                max_tau = 50
                cmd_msg = datatypes.RobotCmd()
                lcm_cmd_to_sdk_cmd(low_cmd, cmd_msg)
                cmd_msg.stamp = time.time_ns()
                tau_wbc = np.array(cmd_msg.tau).clip(-max_tau, max_tau)
                # tau_wbc[[0, 1, 3, 4]] = 0
                cmd_msg.tau = tau_wbc.tolist()
                robot.publishRobotCmd(cmd_msg)  # Publish the robot command

            # Parse the received sensor readings
            low_state = eval(TOPIC_STATE + "_t")()
            if receiver.robot_state is not None:
                sdk_state_to_lcm_state(receiver.robot_state, low_state)
                low_state.qj_vel = joint_vel_filter.calculate_average(np.array(low_state.qj_vel))

            if receiver.imu is not None:
                low_state.acceleration = imu_acc_filter.calculate_average(np.array(receiver.imu.acc))
                low_state.omega = imu_gyro_filter.calculate_average(np.array(receiver.imu.gyro))
                quat = Quaternion(*receiver.imu.quat)
                rpy = quat_to_rpy(quat)
                low_state.rpy[0] = rpy[0]
                low_state.rpy[1] = rpy[1]
                low_state.rpy[2] = rpy[2] - yaw_init
                # ========================= DEBUG IMU
                # low_state.acceleration[:] = [0, 0, 9.8]
                # low_state.omega[:] = [0, 0, 0]
                # low_state.rpy[2] = 0
                # ====================================
                low_state.quaternion = rpy_to_quat(low_state.rpy).to_numpy().tolist()

            lc.publish(TOPIC_STATE, low_state.encode())
            rate.sleep()  # Control loop frequency
    except:
        traceback.print_exc()
        cmd_msg = datatypes.RobotCmd()
        cmd_msg.stamp = time.time_ns()  # Set the timestamp
        # Set default values for control mode, joint positions, velocities, torques, Kp, and Kd
        # motor mode：0: torque； 1：speed； 2：position; default：0
        cmd_msg.mode = [0.0 for _ in range(motor_number)]
        cmd_msg.q = [0.0 for _ in range(motor_number)]
        cmd_msg.dq = [0.0 for _ in range(motor_number)]
        cmd_msg.tau = [0.0 for _ in range(motor_number)]
        cmd_msg.Kp = [0.0 for _ in range(motor_number)]
        cmd_msg.Kd = [0.0 for _ in range(motor_number)]
        print("=> ESTOP. Enter zero torque mode!")
        for i in range(100):
            robot.publishRobotCmd(cmd_msg)  # Publish the robot command
            rate.sleep()
