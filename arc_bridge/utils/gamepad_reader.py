import inputs
import threading
import time

MAX_ABS_VAL = 32768
DEAD_ZONE = 1000

def _interpolate(raw_reading, min_raw_reading, max_raw_reading, new_scale):
    if abs(raw_reading) < min_raw_reading:
        return 0.0

    return raw_reading / max_raw_reading * new_scale


class Gamepad:
    """Interface for reading commands from xbox Gamepad.

    The control works as following:
    1) Press LB+RB at any time for emergency stop
    2) Use the left joystick for forward/backward/left/right walking.
    3) Use the right joystick for rotation around the z-axis.
    """

    def __init__(
        self,
        vel_scale_x: float = 0.5,
        vel_scale_y: float = 0.5,
        vel_scale_rot: float = 1.0,
        max_acc: float = 0.5,
    ):
        """Initialize the gamepad controller.
        Args:
          vel_scale_x: maximum absolute x-velocity command.
          vel_scale_y: maximum absolute y-velocity command.
          vel_scale_rot: maximum absolute yaw-dot command.
        """
        if not inputs.devices.gamepads:
            raise inputs.UnpluggedError("No gamepad found.")

        self.gamepad = inputs.devices.gamepads[0]
        self._vel_scale_x = float(vel_scale_x)
        self._vel_scale_y = float(vel_scale_y)
        self._vel_scale_rot = float(vel_scale_rot)
        self._max_acc = max_acc
        self._lb_pressed = False
        self._rb_pressed = False
        self._lj_pressed = False

        # self._gait_generator = itertools.cycle(ALLOWED_GAITS)
        # self._gait = next(self._gait_generator)
        # self._mode_generator = itertools.cycle(ALLOWED_MODES)
        # self._mode = Parameters.control_mode

        # Controller states
        self.vx, self.vy, self.wz = 0.0, 0.0, 0.0
        self._estop_flagged = False
        self.is_running = True

        # * Daemon threads stop automatically when the main thread exits
        self.read_thread = threading.Thread(target=self.read_loop, daemon=True)
        self.read_thread.start()

    def read_loop(self):
        """The read loop for events.

        This funnction should be executed in a separate thread for continuous
        event recording.
        """
        while self.is_running:  # and not self.estop_flagged:
            try:
                events = self.gamepad.read()
                for event in events:
                    # print(event.ev_type, event.code, event.state)
                    self.update_command(event)
            except:
                pass

        print("Gamepad thread exited")

    def update_command(self, event):
        """Update command based on event readings."""
        if event.ev_type == "Key" and event.code == "BTN_TL":
            self._lb_pressed = bool(event.state)
            if not self._estop_flagged and event.state == 0:
                # self._gait = next(self._gait_generator)
                pass

        elif event.ev_type == "Key" and event.code == "BTN_TR":
            self._rb_pressed = bool(event.state)
            if not self._estop_flagged and event.state == 0:
                # self._mode = next(self._mode_generator)
                pass

        elif event.ev_type == "Key" and event.code == "BTN_THUMBL":
            self._lj_pressed = bool(event.state)

        elif event.ev_type == "Absolute" and event.code == "ABS_X":
            # Left Joystick L/R axis
            self.vy = _interpolate(-event.state, DEAD_ZONE, MAX_ABS_VAL, self._vel_scale_y)
        elif event.ev_type == "Absolute" and event.code == "ABS_Y":
            # Left Joystick F/B axis; need to flip sign for consistency
            self.vx = _interpolate(-event.state, DEAD_ZONE, MAX_ABS_VAL, self._vel_scale_x)
        elif event.ev_type == "Absolute" and event.code == "ABS_RX":
            self.wz = _interpolate(-event.state, DEAD_ZONE, MAX_ABS_VAL, self._vel_scale_rot)

        if self._estop_flagged and self._lj_pressed:
            self._estop_flagged = False
            print("Estop Released.")

        if self._lb_pressed and self._rb_pressed:
            print("EStop Flagged, press LEFT joystick to release.")
            self._estop_flagged = True
            self.vx, self.vy, self.wz = 0.0, 0.0, 0.0

    def get_command(self):
        return [self.vx, self.vy, self.wz, self._estop_flagged]

    def fake_event(self, ev_type, code, value):
        eventinfo = {"ev_type": ev_type, "state": value, "timestamp": 0.0, "code": code}
        event = inputs.InputEvent(self.gamepad, eventinfo)
        self.update_command(event)

    def stop(self):
        self.is_running = False
        self.read_thread.join()
        print("Gamepad thread exited")


if __name__ == "__main__":
    gamepad = Gamepad()
    while True:
        print(
            "Vx: {:.3f}, Vy: {:.3f}, Wz: {:.3f}, Estop: {}".format(
                gamepad.vx, gamepad.vy, gamepad.wz, gamepad._estop_flagged
            )
        )
        time.sleep(0.1)
