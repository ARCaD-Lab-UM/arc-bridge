import time
from typing import Callable, Optional


def _default_time_s() -> float:
    return time.monotonic()
    # return time.time()  # wall-clock time


class PrintThrottle:
    """Simple time-based print with throttling.

    Example:
        print_throttle = PrintThrottle(1.0)  # 1 second interval
        for i in range(1000000):
            print_throttle.print(f"Update: {i}")
    """
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    RESET = "\033[0m"

    def __init__(
        self,
        interval_s: float,  # min interval between prints in seconds. 0 or negative means no throttling.
        *,
        time_provider: Optional[Callable[[], float]] = None,  # function that provides current time in seconds. None to use default.
    ):
        self.interval_s = float(interval_s)
        self._last_t = 0.0
        self._time_provider = time_provider or _default_time_s

    def ready(self) -> bool:
        """Check whether printing is allowed at the current time. Uses time.monotonic().

        Returns:
            bool: True if printing is allowed, False otherwise.
        """
        if self.interval_s <= 0.0:
            return True
        now = self._time_provider()
        if now - self._last_t >= self.interval_s:
            self._last_t = now
            return True
        return False

    def print(self, msg: str) -> None:
        """Print a message if the throttle allows it.

        Args:
            msg (str): Message to print.
        """
        if self.ready():
            print(msg)
