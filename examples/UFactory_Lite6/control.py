import lcm
import time
import threading

import numpy as np

from arc_bridge.lcm_msgs import ufactory_lite6_control_t, ufactory_lite6_state_t


# WARNING: This controller needs an external arming/e-stop gate before use on
# real hardware.
class UFactoryLite6Controller:
    """Gravity-compensation controller for the UFactory Lite 6.

    Publishes a pure-torque command equal to the generalized bias force
    (gravity + Coriolis + centrifugal) reported by the simulator. With
    kp = kd = 0, the bridge applies qj_tau directly, so the arm floats
    under its own dynamics -- nudge a joint and it stays put.
    """

    def __init__(self):
        self.dt = 1.0 / 100  # 100 Hz

        self.n_joint = 6
        self.state_timeout = 0.1
        self.startup_timeout = 5.0
        self.torque_limit = np.array([50.0, 50.0, 32.0, 32.0, 32.0, 20.0])

        self.lock = threading.Lock()
        self.is_running = threading.Event()
        self.is_running.set()

        # Retained for future state-feedback control modes.
        self.qj_pos = np.zeros(self.n_joint)
        self.qj_vel = np.zeros(self.n_joint)
        self.bias_force = np.zeros(self.n_joint)
        self.is_state_received = False
        self.last_state_time = None
        self.fault = False

    def robot_state_handler(self, channel, data):
        try:
            msg = ufactory_lite6_state_t.decode(data)
            qj_pos = np.array(msg.qj_pos, dtype=float)
            qj_vel = np.array(msg.qj_vel, dtype=float)
            bias_force = np.array(msg.bias_force, dtype=float)

            if (qj_pos.shape != (self.n_joint,) or
                    qj_vel.shape != (self.n_joint,) or
                    bias_force.shape != (self.n_joint,)):
                raise ValueError("decoded state has unexpected joint dimension")
            if not (np.all(np.isfinite(qj_pos)) and
                    np.all(np.isfinite(qj_vel)) and
                    np.all(np.isfinite(bias_force))):
                raise ValueError("decoded state contains non-finite values")

            with self.lock:
                self.qj_pos = qj_pos
                self.qj_vel = qj_vel
                self.bias_force = bias_force
                self.last_state_time = time.monotonic()
                self.is_state_received = True
        except Exception as exc:
            print(f"ERROR: failed to process '{channel}' state: {exc}")
            with self.lock:
                self.fault = True

    def step(self):
        now = time.monotonic()
        with self.lock:
            bias_force = self.bias_force.copy()
            is_state_received = self.is_state_received
            last_state_time = self.last_state_time
            fault = self.fault

        state_age = None
        if last_state_time is not None:
            state_age = now - last_state_time

        if (fault or
                not is_state_received or
                state_age is None or
                state_age > self.state_timeout or
                not np.all(np.isfinite(bias_force))):
            return np.zeros(self.n_joint)

        return np.clip(bias_force, -self.torque_limit, self.torque_limit)


def lcm_handle_thread(lc, is_running):
    while is_running.is_set():
        lc.handle_timeout(10)


def publish_zero_torque(lc, topic, n_joint):
    lc_cmd = ufactory_lite6_control_t()
    lc_cmd.timestamp = time.time_ns()
    lc_cmd.qj_tau = np.zeros(n_joint).tolist()
    lc.publish(topic, lc_cmd.encode())


def has_received_state(controller):
    with controller.lock:
        return controller.is_state_received


if __name__ == "__main__":
    controller = UFactoryLite6Controller()

    lcm_state_topic = "ufactory_lite6_state"
    lcm_cmd_topic = "ufactory_lite6_control"

    lc = lcm.LCM("udpm://239.255.76.67:7667?ttl=1")

    state_subscription = lc.subscribe(lcm_state_topic, controller.robot_state_handler)
    state_subscription.set_queue_capacity(1)

    lc_thread = threading.Thread(target=lcm_handle_thread,
                                 args=(lc, controller.is_running),
                                 daemon=True)
    lc_thread.start()

    try:
        # Wait for first state so the very first command carries a real bias force
        print("Waiting for first state on 'ufactory_lite6_state'...")
        startup_start = time.monotonic()
        while not has_received_state(controller):
            if time.monotonic() - startup_start > controller.startup_timeout:
                print("ERROR: no state received within "
                      f"{controller.startup_timeout:.1f}s, exiting.")
                controller.is_running.clear()
                break
            time.sleep(0.01)

        if controller.is_running.is_set():
            print("State received, streaming gravity-compensation torques at "
                  f"{1.0 / controller.dt:.0f} Hz.")

        while controller.is_running.is_set():
            start_time = time.monotonic()

            dof_torque = controller.step()

            lc_cmd = ufactory_lite6_control_t()
            lc_cmd.timestamp = time.time_ns()
            lc_cmd.qj_tau = dof_torque.tolist()
            # qj_pos, qj_vel, kp, kd left at their zero defaults -> pure-torque mode
            lc.publish(lcm_cmd_topic, lc_cmd.encode())

            elapsed_time = time.monotonic() - start_time
            sleep_time = controller.dt - elapsed_time
            if sleep_time > 0.0:
                time.sleep(sleep_time)
    except KeyboardInterrupt:
        print("Interrupted, publishing zero torque and exiting.")
        controller.is_running.clear()
    finally:
        publish_zero_torque(lc, lcm_cmd_topic, controller.n_joint)
        controller.is_running.clear()
        lc_thread.join()
