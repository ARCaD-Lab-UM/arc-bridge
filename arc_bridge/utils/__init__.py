import sys
from importlib.util import find_spec


def _supports_evdev() -> bool:
    if sys.platform != "linux":
        return False
    return find_spec("evdev") is not None


if _supports_evdev():
    # from .gamepad_reader_wired import Gamepad
    from .gamepad_reader_wireless import Gamepad
else:
    from .gamepad_reader import Gamepad
from .orientation_utils import *
from .lowpass_filter import *
from .interpolation_filter import *
