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
DEAD_ZONE = 1000
DEVICE_NAME = "Xbox Wireless Controller"

# TODO unify this with wired connection using evdev

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


def _find_wireless_device() -> evdev.InputDevice:
    """Locate the wireless controller by name."""
    for path in evdev.list_devices():
        device = evdev.InputDevice(path)
        if device.name == DEVICE_NAME:
            return device
        device.close()
    raise UnpluggedError(f"No wireless controller named '{DEVICE_NAME}' found.")


class Gamepad:
    """Gamepad adapter reading wireless Xbox controller events via evdev."""

    def __init__(
        self,
        vel_scale_x: float = 0.5,
        vel_scale_y: float = 0.5,
        vel_scale_rot: float = 1.0,
        max_acc: float = 0.5,
    ):
        self.device: Optional[evdev.InputDevice] = None
        self._grabbed = False
        self._vel_scale_x = float(vel_scale_x)
        self._vel_scale_y = float(vel_scale_y)
        self._vel_scale_rot = float(vel_scale_rot)
        self._max_acc = max_acc  # Reserved for compatibility with wired reader
        self._lb_pressed = False
        self._rb_pressed = False
        self._lj_pressed = False

        self.vx, self.vy, self.wz = 0.0, 0.0, 0.0
        self._estop_flagged = False
        self.is_running = True

        self.device = _find_wireless_device()
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
            elif event.code == ecodes.BTN_TR:
                self._rb_pressed = bool(event.value)
            elif event.code == ecodes.BTN_THUMBL:
                self._lj_pressed = bool(event.value)
        elif event.type == ecodes.EV_ABS:
            if event.code == ecodes.ABS_X:
                centered = _center_axis(event.value)
                self.vy = _interpolate(-centered, DEAD_ZONE, MAX_ABS_VAL, self._vel_scale_y)
            elif event.code == ecodes.ABS_Y:
                centered = _center_axis(event.value)
                self.vx = _interpolate(-centered, DEAD_ZONE, MAX_ABS_VAL, self._vel_scale_x)
            elif event.code in (ecodes.ABS_RX, ecodes.ABS_Z):
                centered = _center_axis(event.value)
                self.wz = _interpolate(-centered, DEAD_ZONE, MAX_ABS_VAL, self._vel_scale_rot)

        if self._estop_flagged and self._lj_pressed:
            self._estop_flagged = False
            print("Estop Released.")

        if self._lb_pressed and self._rb_pressed:
            if not self._estop_flagged:
                print("EStop Flagged, press LEFT joystick to release.")
            self._estop_flagged = True
            self.vx = self.vy = self.wz = 0.0

    def get_command(self):
        return [self.vx, self.vy, self.wz, self._estop_flagged]

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
    gamepad = Gamepad()
    while True:
        print(
            "Vx: {:.3f}, Vy: {:.3f}, Wz: {:.3f}, Estop: {}".format(
                gamepad.vx, gamepad.vy, gamepad.wz, gamepad._estop_flagged
            )
        )
        time.sleep(0.1)
