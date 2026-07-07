import threading
import time
from typing import Optional, Sequence

import evdev
from evdev import ecodes

try:
    from .config_gamepad_reader_evdev import DeviceProfile, DEFAULT_PROFILE, DEVICE_PROFILES
except ImportError:
    # Allow running the module as a test script from the utils directory.
    from config_gamepad_reader_evdev import DeviceProfile, DEFAULT_PROFILE, DEVICE_PROFILES

try:
    from inputs import UnpluggedError
except ImportError:  # pragma: no cover - inputs is available in runtime env
    class UnpluggedError(RuntimeError):
        pass


INT8_MIN = -128
INT8_MAX = 127


def _interpolate(raw_reading: int, min_raw: int, max_raw: int, new_scale: float) -> float:
    """Scale joystick raw values onto the configured command range."""
    if abs(raw_reading) < min_raw:
        return 0.0
    return raw_reading / max_raw * new_scale


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


def _find_device(keywords: Optional[Sequence[str]]) -> evdev.InputDevice:
    """Locate the first matching gamepad device."""
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


def _classify_device(device: evdev.InputDevice) -> DeviceProfile:
    """Return the DeviceProfile best matching the device name."""
    for profile in DEVICE_PROFILES:
        if profile.matches(device.name):
            return profile
    return DEFAULT_PROFILE


def _center_axis(raw_value: int, needs_recentering: bool, max_abs_val: int) -> int:
    """Translate axis readings to signed range based on the profile."""
    if not needs_recentering:
        return raw_value
    centered = int(raw_value) - max_abs_val
    if centered > max_abs_val:
        return max_abs_val
    if centered < -max_abs_val:
        return -max_abs_val
    return centered


class Gamepad:
    """Gamepad reading events via evdev with device-specific profiles."""

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
        self._dev_name_keywords = list(dev_name_keywords) if dev_name_keywords is not None else None
        self._vel_scale_x = float(vel_scale_x)
        self._vel_scale_y = float(vel_scale_y)
        self._vel_scale_rot = float(vel_scale_rot)
        self._scale_pitch = float(scale_pitch)
        self._triggers_scale = float(triggers_scale)
        self._lb_pressed = False
        self._rb_pressed = False
        self._lj_pressed = False
        self._rj_pressed = False

        self.axis = [0.0, 0.0, 0.0, 0.0]  # [L West, L North, R West, R North] for positive directions
        self._estop_flagged = False
        self.is_running = True

        self.params = [0, 0]  # D-Pad counters: [0]: down-,up+ [1]: left-,right+
        self.buttons = [False, False, False, False]  # up left down right order
        self.lbrb = [False, False]  # left bumper, right bumper
        self.ljrj = [False, False]  # left joystick press, right joystick press
        self.lt, self.rt = 0.0, 0.0  # left and right triggers

        self.device = _find_device(self._dev_name_keywords)
        self._profile = _classify_device(self.device)

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
            self._handle_key_event(event)
        elif event.type == ecodes.EV_ABS:
            self._handle_axis_event(event)

        if self._estop_flagged and self._lj_pressed:
            self._estop_flagged = False
            print("Estop Released.")

        if self._rj_pressed and self._lb_pressed and self._rb_pressed:
            if not self._estop_flagged:
                print("EStop Flagged, press LEFT joystick to release.")
            self._estop_flagged = True
            self.axis = [0.0, 0.0, 0.0, 0.0]
            self.params = [0, 0]

    def _handle_key_event(self, event: evdev.events.InputEvent) -> None:
        if event.code == self._profile.btn_lb:
            self._lb_pressed = bool(event.value)
            self.lbrb[0] = self._lb_pressed
        elif event.code == self._profile.btn_rb:
            self._rb_pressed = bool(event.value)
            self.lbrb[1] = self._rb_pressed
        elif event.code == self._profile.btn_lstick:
            self._lj_pressed = bool(event.value)
            self.ljrj[0] = self._lj_pressed
        elif event.code == self._profile.btn_rstick:
            self._rj_pressed = bool(event.value)
            self.ljrj[1] = self._rj_pressed
        elif event.code == self._profile.btn_up:
            self.buttons[0] = bool(event.value)
        elif event.code == self._profile.btn_left:
            self.buttons[1] = bool(event.value)
        elif event.code == self._profile.btn_down:
            self.buttons[2] = bool(event.value)
        elif event.code == self._profile.btn_right:
            self.buttons[3] = bool(event.value)

    def _handle_axis_event(self, event: evdev.events.InputEvent) -> None:
        profile = self._profile
        if event.code == profile.axis_vy:       # left stick X -> axis[0] L West
            centered = _center_axis(event.value, profile.needs_recentering, profile.max_abs_val)
            self.axis[0] = _interpolate(-centered, profile.joystick_dead_zone, profile.max_abs_val, self._vel_scale_y)
        elif event.code == profile.axis_vx:     # left stick Y -> axis[1] L North
            centered = _center_axis(event.value, profile.needs_recentering, profile.max_abs_val)
            self.axis[1] = _interpolate(-centered, profile.joystick_dead_zone, profile.max_abs_val, self._vel_scale_x)
        elif event.code == profile.axis_wz:     # right stick X -> axis[2] R West
            centered = _center_axis(event.value, profile.needs_recentering, profile.max_abs_val)
            self.axis[2] = _interpolate(-centered, profile.joystick_dead_zone, profile.max_abs_val, self._vel_scale_rot)
        elif event.code == profile.axis_pitch:  # right stick Y -> axis[3] R North
            centered = _center_axis(event.value, profile.needs_recentering, profile.max_abs_val)
            self.axis[3] = _interpolate(-centered, profile.joystick_dead_zone, profile.max_abs_val, self._scale_pitch)
        elif event.code == profile.axis_lt:
            self.lt = self._scale_trigger_value(event.value)
        elif event.code == profile.axis_rt:
            self.rt = self._scale_trigger_value(event.value)
        elif event.code == profile.axis_params_vertical:
            if event.value == -1:
                self._update_params(0, 1)
            elif event.value == 1:
                self._update_params(0, -1)
        elif event.code == profile.axis_params_horizontal:
            if event.value == -1:
                self._update_params(1, -1)
            elif event.value == 1:
                self._update_params(1, 1)

    def get_axis(self):
        """
        Return the 4 stick axes: [L West, L North, R West, R North] for positive directions.
        """
        return list(self.axis)

    def get_estop(self):
        return self._estop_flagged

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

    def set_vel_scale_x(self, value: float) -> None:
        new_scale = float(value)
        if self._vel_scale_x != 0.0:
            self.axis[1] *= new_scale / self._vel_scale_x
        self._vel_scale_x = new_scale

    def set_vel_scale_y(self, value: float) -> None:
        new_scale = float(value)
        if self._vel_scale_y != 0.0:
            self.axis[0] *= new_scale / self._vel_scale_y
        self._vel_scale_y = new_scale

    def set_vel_scale_rot(self, value: float) -> None:
        new_scale = float(value)
        if self._vel_scale_rot != 0.0:
            self.axis[2] *= new_scale / self._vel_scale_rot
        self._vel_scale_rot = new_scale

    def set_scale_pitch(self, value: float) -> None:
        new_scale = float(value)
        if self._scale_pitch != 0.0:
            self.axis[3] *= new_scale / self._scale_pitch
        self._scale_pitch = new_scale

    def _scale_trigger_value(self, raw_value: int) -> float:
        profile = self._profile
        clamped = max(0, min(profile.trigger_max, raw_value))
        return _interpolate(clamped, profile.trigger_dead_zone, profile.trigger_max, self._triggers_scale)

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
            "Axis [LW, LN, RW, RN]: [{:.3f}, {:.3f}, {:.3f}, {:.3f}], Estop: {}".format(
                *gamepad.get_axis(), gamepad._estop_flagged
            )
        )
        print("Params:", gamepad.get_params())
        print("Buttons (Y,X,A,B):", gamepad.get_buttons())
        print("LB/RB:", gamepad.get_lbrb(), "LJ/RJ:", gamepad.get_ljrj())
        print("LT/RT:", gamepad.get_triggers())
        time.sleep(0.1)
