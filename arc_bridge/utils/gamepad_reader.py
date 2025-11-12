import inputs
import threading
import time
from typing import Optional

MAX_ABS_VAL = 32768
JOYSTICK_DEAD_ZONE = 4000
TRIGGER_DEAD_ZONE = 10
TRIGGER_MAX = 1023
INT8_MIN = -128
INT8_MAX = 127


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
        dev_name_keywords: Optional[list[str]] = None,
        vel_scale_x: float = 0.5,
        vel_scale_y: float = 0.5,
        vel_scale_rot: float = 1.0,
        max_acc: float = 0.5,
        scale_pitch: float = 1.0,
        triggers_scale: float = 1.0,
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
        # dev_name_keywords is just for compatibility, not used here
        self._vel_scale_x = float(vel_scale_x)
        self._vel_scale_y = float(vel_scale_y)
        self._vel_scale_rot = float(vel_scale_rot)
        self._scale_pitch = float(scale_pitch)
        self._triggers_scale = float(triggers_scale)
        self._lb_pressed = False
        self._rb_pressed = False
        self._lj_pressed = False
        self._rj_pressed = False

        # self._gait_generator = itertools.cycle(ALLOWED_GAITS)
        # self._gait = next(self._gait_generator)
        # self._mode_generator = itertools.cycle(ALLOWED_MODES)
        # self._mode = Parameters.control_mode

        # Controller states
        self.vx, self.vy, self.wz = 0.0, 0.0, 0.0
        self._estop_flagged = False
        self.is_running = True

        self.pitch = 0.0
        self.params = [0, 0]
        self.buttons = [False, False, False, False]
        self.lbrb = [False, False]
        self.ljrj = [False, False]
        self.lt, self.rt = 0.0, 0.0

        # * Daemon threads stop automatically when the main thread exits
        self.read_thread = threading.Thread(target=self.read_loop, daemon=True)
        self.read_thread.start()

    def read_loop(self):
        """The read loop for events."""
        while self.is_running:  # and not self.estop_flagged:
            try:
                #! TODO this is a blocking call, may need to force a timeout
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
            self.lbrb[0] = self._lb_pressed
            if not self._estop_flagged and event.state == 0:
                # self._gait = next(self._gait_generator)
                pass

        elif event.ev_type == "Key" and event.code == "BTN_TR":
            self._rb_pressed = bool(event.state)
            self.lbrb[1] = self._rb_pressed
            if not self._estop_flagged and event.state == 0:
                # self._mode = next(self._mode_generator)
                pass

        elif event.ev_type == "Key" and event.code == "BTN_THUMBL":
            self._lj_pressed = bool(event.state)
            self.ljrj[0] = self._lj_pressed

        elif event.ev_type == "Key" and event.code == "BTN_THUMBR":
            self._rj_pressed = bool(event.state)
            self.ljrj[1] = self._rj_pressed

        elif event.ev_type == "Key" and event.code == "BTN_WEST":
            self.buttons[0] = bool(event.state)

        elif event.ev_type == "Key" and event.code == "BTN_NORTH":
            self.buttons[1] = bool(event.state)

        elif event.ev_type == "Key" and event.code == "BTN_SOUTH":
            self.buttons[2] = bool(event.state)

        elif event.ev_type == "Key" and event.code == "BTN_EAST":
            self.buttons[3] = bool(event.state)

        elif event.ev_type == "Absolute" and event.code == "ABS_X":
            # Left Joystick L/R axis
            self.vy = _interpolate(-event.state, JOYSTICK_DEAD_ZONE, MAX_ABS_VAL, self._vel_scale_y)
        elif event.ev_type == "Absolute" and event.code == "ABS_Y":
            # Left Joystick F/B axis; need to flip sign for consistency
            self.vx = _interpolate(-event.state, JOYSTICK_DEAD_ZONE, MAX_ABS_VAL, self._vel_scale_x)
        elif event.ev_type == "Absolute" and event.code == "ABS_RX":
            self.wz = _interpolate(-event.state, JOYSTICK_DEAD_ZONE, MAX_ABS_VAL, self._vel_scale_rot)

        elif event.ev_type == "Absolute" and event.code == "ABS_RY":
            self.pitch = _interpolate(event.state, JOYSTICK_DEAD_ZONE, MAX_ABS_VAL, self._scale_pitch)

        elif event.ev_type == "Absolute" and event.code == "ABS_Z":
            self.lt = self._scale_trigger_value(event.state)

        elif event.ev_type == "Absolute" and event.code == "ABS_RZ":
            self.rt = self._scale_trigger_value(event.state)

        elif event.ev_type == "Absolute" and event.code == "ABS_HAT0Y":
            # D-pad up/down
            if event.state == -1:
                self._update_params(0, 1)
            elif event.state == 1:
                self._update_params(0, -1)

        elif event.ev_type == "Absolute" and event.code == "ABS_HAT0X":
            # D-pad left/right
            if event.state == -1:
                self._update_params(1, -1)
            elif event.state == 1:
                self._update_params(1, 1)

        if self._estop_flagged and self._lj_pressed:
            self._estop_flagged = False
            print("Estop Released.")

        if self._lb_pressed and self._rb_pressed:
            if not self._estop_flagged:
                print("EStop Flagged, press LEFT joystick to release.")
            self._estop_flagged = True
            self.vx, self.vy, self.wz = 0.0, 0.0, 0.0

    def get_command(self):
        return [self.vx, self.vy, self.wz, self._estop_flagged]

    def get_pitch(self):
        return self.pitch

    def get_params(self):
        return list(self.params)

    def get_buttons(self):
        return list(self.buttons)

    def get_lbrb(self):
        return list(self.lbrb)

    def get_ljrj(self):
        return list(self.ljrj)

    def get_triggers(self):
        return self.lt, self.rt

    def fake_event(self, ev_type, code, value):
        eventinfo = {"ev_type": ev_type, "state": value, "timestamp": 0.0, "code": code}
        event = inputs.InputEvent(self.gamepad, eventinfo)
        self.update_command(event)

    def _scale_trigger_value(self, raw_value: int) -> float:
        clamped = max(0, min(TRIGGER_MAX, raw_value))
        return _interpolate(clamped, TRIGGER_DEAD_ZONE, TRIGGER_MAX, self._triggers_scale)

    def _update_params(self, index: int, delta: int) -> None:
        new_value = self.params[index] + delta
        self.params[index] = max(INT8_MIN, min(INT8_MAX, new_value))

    def stop(self):
        self.is_running = False
        self.read_thread.join()
        print("Gamepad thread exited")


if __name__ == "__main__":
    gamepad = Gamepad(vel_scale_x=2.0, vel_scale_y=0.5, vel_scale_rot=3.141592654, scale_pitch=1.570796327, triggers_scale=1.0)
    while True:
        print(
            "Vx: {:.3f}, Vy: {:.3f}, Wz: {:.3f}, Estop: {}".format(
                gamepad.vx, gamepad.vy, gamepad.wz, gamepad._estop_flagged
            )
        )
        print("Pitch: {:.3f}".format(gamepad.get_pitch()))
        print("Params:", gamepad.get_params())
        print("Buttons (Y,X,A,B):", gamepad.get_buttons())
        print("LB/RB:", gamepad.get_lbrb(), "LJ/RJ:", gamepad.get_ljrj())
        print("LT/RT:", gamepad.get_triggers())
        time.sleep(0.1)
