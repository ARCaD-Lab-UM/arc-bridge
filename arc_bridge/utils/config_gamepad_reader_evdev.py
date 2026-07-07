from dataclasses import dataclass

from evdev import ecodes


@dataclass(frozen=True)
class DeviceProfile:
    """Describe how to interpret events for a gamepad family."""

    names: tuple[str, ...]

    # channel mappings
    axis_ly: int
    axis_lx: int
    axis_rx: int
    axis_ry: int
    axis_lt: int
    axis_rt: int
    axis_params_vertical: int
    axis_params_horizontal: int

    # key/button mappings
    btn_up: int
    btn_left: int
    btn_down: int
    btn_right: int
    btn_lb: int
    btn_rb: int
    btn_lstick: int
    btn_rstick: int
    needs_recentering: bool

    # device-specific scaling constants
    max_abs_val: int = 32768  # raw joystick magnitude at extremes
    joystick_dead_zone: int = 4000  # ignore tiny joystick motion
    trigger_max: int = 1023  # raw trigger max reading
    trigger_dead_zone: int = 10  # ignore tiny trigger motion

    # Requires case-sensitive exact match of device name
    def matches(self, device_name: str) -> bool:
        return any(device_name == profile_name for profile_name in self.names)


DEVICE_PROFILES: tuple[DeviceProfile, ...] = (
    DeviceProfile(
        names=("Microsoft X-Box One S pad", "Microsoft Xbox Series S|X Controller",),
        axis_ly=ecodes.ABS_Y,
        axis_lx=ecodes.ABS_X,
        axis_rx=ecodes.ABS_RX,
        axis_ry=ecodes.ABS_RY,
        axis_lt=ecodes.ABS_Z,
        axis_rt=ecodes.ABS_RZ,
        axis_params_vertical=ecodes.ABS_HAT0Y,
        axis_params_horizontal=ecodes.ABS_HAT0X,
        btn_up=ecodes.BTN_WEST,
        btn_left=ecodes.BTN_NORTH,
        btn_down=ecodes.BTN_SOUTH,
        btn_right=ecodes.BTN_EAST,
        btn_lb=ecodes.BTN_TL,
        btn_rb=ecodes.BTN_TR,
        btn_lstick=ecodes.BTN_THUMBL,
        btn_rstick=ecodes.BTN_THUMBR,
        needs_recentering=False,
        max_abs_val=32768,
        joystick_dead_zone=4000,
        trigger_max=1023,
        trigger_dead_zone=10,
    ),
    DeviceProfile(
        names=("Xbox Wireless Controller",),
        axis_ly=ecodes.ABS_Y,
        axis_lx=ecodes.ABS_X,
        axis_rx=ecodes.ABS_Z,
        axis_ry=ecodes.ABS_RZ,
        axis_lt=ecodes.ABS_BRAKE,
        axis_rt=ecodes.ABS_GAS,
        axis_params_vertical=ecodes.ABS_HAT0Y,
        axis_params_horizontal=ecodes.ABS_HAT0X,
        btn_up=ecodes.BTN_WEST,
        btn_left=ecodes.BTN_NORTH,
        btn_down=ecodes.BTN_SOUTH,
        btn_right=ecodes.BTN_EAST,
        btn_lb=ecodes.BTN_TL,
        btn_rb=ecodes.BTN_TR,
        btn_lstick=ecodes.BTN_THUMBL,
        btn_rstick=ecodes.BTN_THUMBR,
        needs_recentering=True,
        max_abs_val=32768,
        joystick_dead_zone=4000,
        trigger_max=1023,
        trigger_dead_zone=10,
    ),
    DeviceProfile(
        names=("Logitech Gamepad F710",),
        axis_ly=ecodes.ABS_Y,
        axis_lx=ecodes.ABS_X,
        axis_rx=ecodes.ABS_RX,
        axis_ry=ecodes.ABS_RY,
        axis_lt=ecodes.ABS_Z,
        axis_rt=ecodes.ABS_RZ,
        axis_params_vertical=ecodes.ABS_HAT0Y,
        axis_params_horizontal=ecodes.ABS_HAT0X,
        btn_up=ecodes.BTN_WEST,
        btn_left=ecodes.BTN_NORTH,
        btn_down=ecodes.BTN_SOUTH,
        btn_right=ecodes.BTN_EAST,
        btn_lb=ecodes.BTN_TL,
        btn_rb=ecodes.BTN_TR,
        btn_lstick=ecodes.BTN_THUMBL,
        btn_rstick=ecodes.BTN_THUMBR,
        needs_recentering=False,
        max_abs_val=32768,
        joystick_dead_zone=4000,
        trigger_max=255,
        trigger_dead_zone=5,
    ),
    DeviceProfile(
        names=("Logitech Logitech Cordless RumblePad 2",), # no axis, do not use
        axis_ly=ecodes.ABS_Y,
        axis_lx=ecodes.ABS_X,
        axis_rx=ecodes.ABS_Z,
        axis_ry=ecodes.ABS_RZ,
        axis_lt=ecodes.BTN_TL,
        axis_rt=ecodes.BTN_TR,
        axis_params_vertical=ecodes.ABS_HAT0Y,
        axis_params_horizontal=ecodes.ABS_HAT0X,
        btn_up=ecodes.BTN_NORTH,
        btn_left=ecodes.BTN_SOUTH,
        btn_down=ecodes.BTN_EAST,
        btn_right=ecodes.BTN_C,
        btn_lb=ecodes.BTN_WEST,
        btn_rb=ecodes.BTN_Z,
        btn_lstick=ecodes.BTN_SELECT,
        btn_rstick=ecodes.BTN_START,
        needs_recentering=False,
        max_abs_val=32768,
        joystick_dead_zone=4000,
        trigger_max=255,
        trigger_dead_zone=10,
    ),
)


DEFAULT_PROFILE = DeviceProfile(
    names=tuple(),
    axis_ly=ecodes.ABS_Y,
    axis_lx=ecodes.ABS_X,
    axis_rx=ecodes.ABS_RX,
    axis_ry=ecodes.ABS_RY,
    axis_lt=ecodes.ABS_Z,
    axis_rt=ecodes.ABS_RZ,
    axis_params_vertical=ecodes.ABS_HAT0Y,
    axis_params_horizontal=ecodes.ABS_HAT0X,
    btn_up=ecodes.BTN_WEST,
    btn_left=ecodes.BTN_NORTH,
    btn_down=ecodes.BTN_SOUTH,
    btn_right=ecodes.BTN_EAST,
    btn_lb=ecodes.BTN_TL,
    btn_rb=ecodes.BTN_TR,
    btn_lstick=ecodes.BTN_THUMBL,
    btn_rstick=ecodes.BTN_THUMBR,
    needs_recentering=False,
    max_abs_val=32768,
    joystick_dead_zone=4000,
    trigger_max=1023,
    trigger_dead_zone=10,
)
