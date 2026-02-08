from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable, ClassVar, Optional

# Type aliases for callback functions
OfflineCallback = Callable[[Any], None]
CheckDataErrorCallback = Callable[[], bool]
SolveDataErrorCallback = Callable[[], None]


def _default_time_ms() -> float:
    return time.monotonic_ns() / 1_000_000.0  # convert to milliseconds with sub-ms resolution


@dataclass
class DaemonConfig:
    # just online wait time (unstable time); wait time after offline to consider online (ms)
    set_online_jitter_time_ms: float

    # time to consider offline if no update received (ms)
    set_offline_time_ms: float

    # optional callback on offline event; called when offline is detected to handle it. None if not needed.
    offline_callback: Optional[OfflineCallback] = None

    # optional callback to check if data is erroneous; None if not needed.
    check_data_error_callback: Optional[CheckDataErrorCallback] = None

    # optional callback to solve data error; None if not needed.
    solve_data_error_callback: Optional[SolveDataErrorCallback] = None

    # optional owner identifier for the daemon instance; can be any type (str, int, object, etc.); used in offline_callback to identify the owner.
    owner_id: Any = None


class Daemon:
    """Watchdog for one module/device.

    Tracks online/offline status based on reload timing and optional data checks.

    Example:
        config = DaemonConfig(
            set_online_jitter_time_ms=200.0,
            set_offline_time_ms=1000.0,
            owner_id="motor-1",
        )  # optional callback etc. can be provided in config
        daemon = Daemon(config)
        daemon.reload()  # feed the watchdog when new data arrives
        daemon.update()  # call periodically (e.g., in a loop)
    """
    # static class variable
    TIME_RESOLUTION_HZ: ClassVar[float] = 1000.0

    def __init__(
        self,
        config: DaemonConfig,
        *,
        time_provider: Optional[Callable[[], float]] = None,  # function that provides current time in milliseconds. None to use default.
        spy: Optional[DaemonSpy] = None,  # rolling window stats helper; None to disable.
    ) -> None:
        if config.set_online_jitter_time_ms < 0 or config.set_offline_time_ms < 0:
            raise ValueError("offline/jitter times must be non-negative")

        self.set_online_jitter_time_ms = float(config.set_online_jitter_time_ms)
        self.set_offline_time_ms = float(config.set_offline_time_ms)
        self.offline_callback = config.offline_callback
        self.check_data_error_callback = config.check_data_error_callback
        self.solve_data_error_callback = config.solve_data_error_callback
        self.owner_id = config.owner_id

        self.new_time_ms = 0.0  # timestamp of the latest data received
        self.last_time_ms = 0.0  # timestamp of the previous data received
        self.lost_time_ms = 0.0  # timestamp when the instance was marked offline
        self.work_time_ms = 0.0  # timestamp when the instance was last marked online

        self._is_error = False  # True if any error exists. Equivalent: (_is_offline or _data_error)
        self._is_offline = False  # True if the instance is offline
        self._data_error = False  # True if the latest data is erroneous

        self.delta_ms = 0.0  # time difference between the last two data received in milliseconds
        self.frequency_hz = 0.0  # calculated est. data update frequency in Hz
        self.min_delta_ms = 0.0  # rolling window min delta; 0 when spy disabled
        self.max_delta_ms = 0.0  # rolling window max delta; 0 when spy disabled

        # set up time provider
        self._time_provider = time_provider or _default_time_ms
        self._spy = spy

    @property
    def spy(self) -> Optional[DaemonSpy]:
        """Get the attached daemon spy."""
        return self._spy

    def attach_spy(self, spy: DaemonSpy) -> None:
        """Attach a rolling window spy.

        Args:
            spy (DaemonSpy): daemon spy to attach to.
        """
        self._spy = spy

    def detach_spy(self) -> None:
        """Detach the rolling window spy."""
        self._spy = None
        self.min_delta_ms = 0.0
        self.max_delta_ms = 0.0

    def _now_ms(self, now_ms: Optional[float] = None) -> float:
        """Get the current time in milliseconds.

        Args:
            now_ms (Optional[float], optional): override time in milliseconds. Defaults to None.

        Returns:
            float: current time in milliseconds.
        """
        return self._time_provider() if now_ms is None else float(now_ms)

    def _record_delta(self, delta_ms: float) -> None:
        if delta_ms <= 0:
            return
        self.delta_ms = delta_ms
        if self._spy is None:
            self.frequency_hz = self.TIME_RESOLUTION_HZ / float(self.delta_ms)
            self.min_delta_ms = 0.0
            self.max_delta_ms = 0.0
            return

        self._spy.record(self.delta_ms)
        self.frequency_hz = self._spy.frequency_hz
        self.min_delta_ms = self._spy.min_delta_ms
        self.max_delta_ms = self._spy.max_delta_ms

    def reload(self, now_ms: Optional[float] = None) -> None:
        """Reload the daemon with current time when new data is received.

        Args:
            now_ms (Optional[float], optional): override time in milliseconds. Defaults to None.
        """
        now = self._now_ms(now_ms)
        self.last_time_ms = self.new_time_ms
        self.new_time_ms = now
        if self.new_time_ms > self.last_time_ms:
            self._record_delta(self.new_time_ms - self.last_time_ms)

        if self._is_offline:
            self._is_offline = False
            self.work_time_ms = now

        if self.check_data_error_callback is not None:
            if self.check_data_error_callback():
                self._is_error = True
                self._data_error = True
                if self.solve_data_error_callback is not None:
                    self.solve_data_error_callback()
            else:
                self._data_error = False
        else:
            self._data_error = False

    def is_online(self) -> bool:
        """Whether the daemon is online.

        Returns:
            bool: True if online.
        """
        return not self._is_offline

    def is_offline(self) -> bool:
        """Whether the daemon is offline.

        Returns:
            bool: True if offline.
        """
        return self._is_offline

    def is_error(self) -> bool:
        """Whether any error exists (offline or data error).

        Returns:
            bool: True if any error exists.
        """
        return self._is_error

    def update(self, now_ms: Optional[float] = None) -> None:
        """Background task/function to update the daemon status.

        Args:
            now_ms (Optional[float], optional): override time in milliseconds. Defaults to None.
        """
        current_time = self._now_ms(now_ms)

        # to jusge offline/online status; one of the following scenarios:
        if (current_time > self.new_time_ms and current_time - self.new_time_ms > self.set_offline_time_ms):
            # offline
            if not self._is_offline:
                # mark offline/error and timestamp
                self._is_offline = True
                self._is_error = True
                self.lost_time_ms = current_time
            # call offline callback if exists
            if self.offline_callback is not None:
                self.offline_callback(self.owner_id)  # owner_id is passed so that the callback can identify the specific owner
        elif (current_time > self.new_time_ms and current_time - self.work_time_ms < self.set_online_jitter_time_ms):
            # just online (connected after offline; unstable time), within jitter, still consider error
            self._is_offline = False
            self._is_error = True  # only mark as error
        else:
            # online normally
            self._is_offline = False
            self._is_error = bool(self._data_error)  # mark error only if data error exists


class DaemonSpy:
    """Rolling window statistics for daemon update intervals. Stats include min/max intervals, frequency.
    window_size sets how many samples to keep for stats calculation.

    Example:
        spy = DaemonSpy(window_size=10)
        daemon = Daemon(DaemonConfig(50.0, 200.0), spy=spy)
    """
    TIME_RESOLUTION_HZ: ClassVar[float] = 1000.0

    def __init__(self, window_size: int) -> None:
        self.window_size = int(window_size)
        if self.window_size <= 0:
            raise ValueError("window_size must be > 0")
        self._delta_samples: deque[float] = deque()
        self._delta_sum = 0.0  # sum of deltas in the current window
        self._min_queue: deque[float] = deque()
        self._max_queue: deque[float] = deque()
        self.delta_ms = 0.0  # latest delta sample
        self.min_delta_ms = 0.0  # rolling window min delta
        self.max_delta_ms = 0.0  # rolling window max delta
        self.frequency_hz = 0.0  # rolling window calculated frequency in Hz

    def reset(self) -> None:
        """Reset the rolling window statistics."""
        self._delta_samples.clear()
        self._delta_sum = 0.0
        self._min_queue.clear()
        self._max_queue.clear()
        self.delta_ms = 0.0
        self.min_delta_ms = 0.0
        self.max_delta_ms = 0.0
        self.frequency_hz = 0.0

    def record(self, delta_ms: float) -> None:
        """Record a new sample.

        Args:
            delta_ms (float): time difference in milliseconds.
        """
        if delta_ms <= 0:
            return
        self.delta_ms = delta_ms
        if len(self._delta_samples) == self.window_size:
            old = self._delta_samples.popleft()
            self._delta_sum -= old
            if self._min_queue and self._min_queue[0] == old:
                self._min_queue.popleft()  # pop from left (front) if the oldest sample is the current smallest min
            if self._max_queue and self._max_queue[0] == old:
                self._max_queue.popleft()

        self._delta_samples.append(delta_ms)
        self._delta_sum += delta_ms

        # if new delta is smaller than right [-1] current largest mins, pop from right (back) until not; since those popped cannot be mins anymore
        while self._min_queue and self._min_queue[-1] > delta_ms:
            self._min_queue.pop()
        self._min_queue.append(delta_ms)  # [0] is samllest; [-1] is largest

        while self._max_queue and self._max_queue[-1] < delta_ms:
            self._max_queue.pop()
        self._max_queue.append(delta_ms)  # [0] is largest; [-1] is smallest

        self.min_delta_ms = self._min_queue[0] if self._delta_samples else 0.0
        self.max_delta_ms = self._max_queue[0] if self._delta_samples else 0.0
        if self._delta_sum <= 0:
            self.frequency_hz = 0.0
        else:
            count = len(self._delta_samples)
            self.frequency_hz = (self.TIME_RESOLUTION_HZ * count) / float(self._delta_sum)
