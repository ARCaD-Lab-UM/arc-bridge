import time
import threading
from typing import Optional

import evdev
from evdev import ecodes

try:
    from inputs import UnpluggedError
except ImportError:  # pragma: no cover - inputs is available in runtime env
    class UnpluggedError(RuntimeError):
        pass


MAX_ABS_VAL = 32768
JOYSTICK_DEAD_ZONE = 4000
TRIGGER_DEAD_ZONE = 10
DEVICE_NAMES = ["Xbox Wireless Controller"]
TRIGGER_MAX = 1023
INT8_MIN = -128
INT8_MAX = 127


def _interpolate(raw_reading: int, min_raw: int, max_raw: int, new_scale: float) -> float:
    """Scale joystick raw values onto the configured command range."""
    if abs(raw_reading) < min_raw:
        return 0.0
    return raw_reading / max_raw * new_scale


def _center_axis(raw_value: int) -> int:
    """Translate wireless axis readings (0..2*MAX_ABS_VAL) to signed range."""
    centered = int(raw_value) - MAX_ABS_VAL
    if centered > MAX_ABS_VAL:
        return MAX_ABS_VAL
    if centered < -MAX_ABS_VAL:
        return -MAX_ABS_VAL
    return centered


def _is_gamepad(device: evdev.InputDevice) -> bool:
    """Return True when the device exposes standard gamepad buttons."""
    key_codes = device.capabilities().get(ecodes.EV_KEY, [])
    gamepad_buttons = (
        ecodes.BTN_SOUTH,
        ecodes.BTN_EAST,
        ecodes.BTN_NORTH,
        ecodes.BTN_WEST,
        ecodes.BTN_TL,
        ecodes.BTN_TR,
    )
    return any(btn in key_codes for btn in gamepad_buttons)


def _find_device(keywords: Optional[list[str]]) -> evdev.InputDevice:
    """Locate the gamepad by name."""
    for path in evdev.list_devices():
        device = evdev.InputDevice(path)
        # print(f"Found device: {device.name}")
        if keywords is None:
            if _is_gamepad(device):
                return device
        elif any(keyword.lower() in device.name.lower() for keyword in keywords):
            return device
        device.close()
    if keywords is None:
        raise UnpluggedError("No gamepad found.")
    raise UnpluggedError(f"No gamepad matching keywords {keywords} found.")


class Gamepad:
    """Gamepad reading events via evdev."""

    def __init__(
        self,
        dev_name_keywords: Optional[list[str]] = None,
        vel_scale_x: float = 0.5,
        vel_scale_y: float = 0.5,
        vel_scale_rot: float = 1.0,
        scale_pitch: float = 1.0,
        triggers_scale: float = 1.0,
    ):
        self.device: Optional[evdev.InputDevice] = None
        self._grabbed = False
        if dev_name_keywords is None:
            self._dev_name_keywords = None
        else:
            self._dev_name_keywords = list(dev_name_keywords)
        self._vel_scale_x = float(vel_scale_x)
        self._vel_scale_y = float(vel_scale_y)
        self._vel_scale_rot = float(vel_scale_rot)
        self._scale_pitch = scale_pitch
        self._triggers_scale = float(triggers_scale)
        self._lb_pressed = False
        self._rb_pressed = False
        self._lj_pressed = False
        self._rj_pressed = False

        self.vx, self.vy, self.wz = 0.0, 0.0, 0.0
        self._estop_flagged = False
        self.is_running = True

        self.pitch = 0.0
        self.params = [0, 0] # D-Pad counters: [0]: down-,up+ [1]: left-,right+
        self.buttons = [False, False, False, False]  # up left down right order
        self.lbrb = [False, False] # left bumper, right bumper
        self.ljrj = [False, False] # left joystick press, right joystick press
        self.lt, self.rt = 0.0, 0.0 # left and right triggers

        self.device = _find_device(self._dev_name_keywords)
        try:
            self.device.grab()
            self._grabbed = True
        except OSError:
            self._grabbed = False

        self.read_thread = threading.Thread(target=self.read_loop, daemon=True)
        self.read_thread.start()

    def read_loop(self) -> None:
        """Continuously read controller events and update commands."""
        try:
            for event in self.device.read_loop():
                if not self.is_running:
                    break
                if event.type in (ecodes.EV_KEY, ecodes.EV_ABS):
                    self.update_command(event)
        except (OSError, IOError):
            # Device disconnected; flag estop and stop thread gracefully
            if self.is_running:
                self._estop_flagged = True
        finally:
            if self.device is not None and self._grabbed:
                try:
                    self.device.ungrab()
                except OSError:
                    pass
                self._grabbed = False

        print("Gamepad thread exited")

    def update_command(self, event: evdev.events.InputEvent) -> None:
        """Update command state with incoming event data."""
        if event.type == ecodes.EV_KEY:
            if event.code == ecodes.BTN_TL:
                self._lb_pressed = bool(event.value)
                self.lbrb[0] = self._lb_pressed
            elif event.code == ecodes.BTN_TR:
                self._rb_pressed = bool(event.value)
                self.lbrb[1] = self._rb_pressed
            elif event.code == ecodes.BTN_THUMBL:
                self._lj_pressed = bool(event.value)
                self.ljrj[0] = self._lj_pressed
            elif event.code == ecodes.BTN_THUMBR:
                self._rj_pressed = bool(event.value)
                self.ljrj[1] = self._rj_pressed
            elif event.code == ecodes.BTN_WEST:
                self.buttons[0] = bool(event.value)
            elif event.code == ecodes.BTN_NORTH:
                self.buttons[1] = bool(event.value)
            elif event.code == ecodes.BTN_SOUTH:
                self.buttons[2] = bool(event.value)
            elif event.code == ecodes.BTN_EAST:
                self.buttons[3] = bool(event.value)
        elif event.type == ecodes.EV_ABS:
            if event.code == ecodes.ABS_X:
                centered = _center_axis(event.value)
                self.vy = _interpolate(-centered, JOYSTICK_DEAD_ZONE, MAX_ABS_VAL, self._vel_scale_y)
            elif event.code == ecodes.ABS_Y:
                centered = _center_axis(event.value)
                self.vx = _interpolate(-centered, JOYSTICK_DEAD_ZONE, MAX_ABS_VAL, self._vel_scale_x)
            elif event.code == ecodes.ABS_Z:
                centered = _center_axis(event.value)
                self.wz = _interpolate(-centered, JOYSTICK_DEAD_ZONE, MAX_ABS_VAL, self._vel_scale_rot)
            elif event.code == ecodes.ABS_RZ: # vertical movement
                centered = _center_axis(event.value)
                self.pitch = _interpolate(-centered, JOYSTICK_DEAD_ZONE, MAX_ABS_VAL, self._scale_pitch)
            elif event.code == ecodes.ABS_BRAKE:
                self.lt = self._scale_trigger_value(event.value)
            elif event.code == ecodes.ABS_GAS:
                self.rt = self._scale_trigger_value(event.value)
            elif event.code == ecodes.ABS_HAT0Y:
                if event.value == -1:
                    self._update_params(0, 1)
                elif event.value == 1:
                    self._update_params(0, -1)
            elif event.code == ecodes.ABS_HAT0X:
                if event.value == -1:
                    self._update_params(1, -1)
                elif event.value == 1:
                    self._update_params(1, 1)

        if self._estop_flagged and self._lj_pressed:
            self._estop_flagged = False
            print("Estop Released.")

        if self._lb_pressed and self._rb_pressed:
            if not self._estop_flagged:
                print("EStop Flagged, press LEFT joystick to release.")
            self._estop_flagged = True
            self.vx = self.vy = self.wz = 0.0
            self.params = [0, 0]

    def get_command(self):
        """
        Return vel command and estop flag as a list
        """
        return [self.vx, self.vy, self.wz, self._estop_flagged]

    def get_pitch(self):
        return self.pitch

    def get_params(self):
        """
        Return D-Pad counter as a list. [0]: down-,up+ [1]: left-,right+
        """
        return list(self.params)

    def get_buttons(self):
        """
        Return face buttons state as a list. [0]: up, [1]: left, [2]: down, [3]: right
        """
        return list(self.buttons)

    def get_lbrb(self):
        """
        Return bumper buttons state as a list. [0]: left bumper, [1]: right bumper
        """
        return list(self.lbrb)
    
    def get_ljrj(self):
        """
        Return joystick press buttons state as a list. [0]: left joystick press, [1]: right joystick press
        """
        return list(self.ljrj)

    def get_triggers(self):
        """
        Return trigger values as a tuple. (left trigger, right trigger)
        """
        return self.lt, self.rt

    def _scale_trigger_value(self, raw_value: int) -> float:
        clamped = max(0, min(TRIGGER_MAX, raw_value))
        return _interpolate(clamped, TRIGGER_DEAD_ZONE, TRIGGER_MAX, self._triggers_scale)

    def _update_params(self, index: int, delta: int) -> None:
        new_value = self.params[index] + delta
        self.params[index] = max(INT8_MIN, min(INT8_MAX, new_value))

    def fake_event(self, event_type: int, code: int, value: int) -> None:
        """Manually feed a synthetic event for testing."""
        mock_event = evdev.events.InputEvent(0, 0, event_type, code, value)
        self.update_command(mock_event)

    def stop(self) -> None:
        self.is_running = False
        if self.device is not None and self._grabbed:
            try:
                self.device.ungrab()
            except OSError:
                pass
            self._grabbed = False
        if self.device is not None:
            self.device.close()
        if self.read_thread.is_alive():
            self.read_thread.join(timeout=0.5)
        print("Gamepad thread exited")


if __name__ == "__main__":
    # gamepad = Gamepad(dev_name_keywords=DEVICE_NAMES, vel_scale_x=2.0, vel_scale_y=0.5, vel_scale_rot=3.141592654, scale_pitch=1.570796327, triggers_scale=1.0)
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
