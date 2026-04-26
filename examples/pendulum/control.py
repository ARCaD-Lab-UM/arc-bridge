import lcm
import time

import numpy as np
from threading import Thread
from functools import partial

from arc_bridge.lcm_msgs import pendulum_control_t, pendulum_state_t


class PendulumController:

    is_running = True
    def __init__(self):

        self.dt = 1/10 # desired controller frequency
 
        # Dimensions
        self.n_state = 1
        self.n_ctrl = 1

        # Controller states
        self.joint_pos = np.zeros(self.n_ctrl)
        self.joint_vel = np.zeros(self.n_ctrl)
        self.inertia_mat = np.eye(self.n_state)
        self.bias_force = np.zeros(self.n_state)
        self.is_state_received = False

        # Gains
        self.kp = np.array([15.0])
        self.kd = np.array([1.0])

    def robot_state_handler(self, channel, data):
        """Handle incoming state updates from LCM."""
        msg = pendulum_state_t.decode(data)

        # Joint state
        self.joint_pos = np.array(msg.qj_pos)
        self.joint_vel = np.array(msg.qj_vel)
        # Dynamics quantitys
        self.inertia_mat = np.array(msg.inertia_mat)
        self.bias_force = np.array(msg.bias_force)
        # Flip state received flag
        self.is_state_received = True

    def step(self, des_dof_pos=np.array([np.pi]), des_dof_vel=np.array([0.0])):
        """Step the controller."""
        # Apply PD Control law
        pos_err = des_dof_pos - self.joint_pos
        vel_err = des_dof_vel - self.joint_vel
        ctrl_torque = self.kp * pos_err + self.kd * vel_err

        return des_dof_pos, ctrl_torque


def lcm_handle_thread(lc):
    while True:
        lc.handle_timeout(0)

if __name__ == "__main__":
    # Init controller
    controller = PendulumController()

    # Setup LCM protocol
    lcm_cmd_topic = "pendulum_control"
    lcm_state_topic = "pendulum_state"

    # LCM Configuration 
    lc = lcm.LCM("udpm://239.255.76.67:7667?ttl=1")

    # Subscribe to state messages
    state_subscription = lc.subscribe(lcm_state_topic, controller.robot_state_handler)
    state_subscription.set_queue_capacity(1)

    # Start LCM handling thread
    lc_thread = Thread(target=partial(lcm_handle_thread, lc), daemon=True)
    lc_thread.start()

    while controller.is_running:
        # Time the loop
        start_time = time.time()

        # Step controller
        dof_pos, dof_torque = controller.step()

        # Publish control command
        lc_cmd = pendulum_control_t()
        lc_cmd.timestamp = time.time_ns()
        lc_cmd.qj_tau = dof_torque.tolist()
        # Enable low-level joint PD control
        # lc_cmd.qj_pos = dof_pos.tolist()
        # lc_cmd.qj_vel = [0.0]
        # lc_cmd.kp = controller.kp.tolist()
        # lc_cmd.kd = controller.kd.tolist()
        lc.publish(lcm_cmd_topic, lc_cmd.encode())

        # Sleep to maintain loop rate
        while time.time() - start_time < controller.dt:
            # yield to other threads
            time.sleep(0)
