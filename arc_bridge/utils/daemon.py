from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable, ClassVar, Optional

# Type aliases for callback functions
OfflineCallback = Callable[[Any], None]
DataErrorCheck = Callable[[], bool]
DataErrorSolver = Callable[[], None]


def _default_time_ms() -> int:
    return time.monotonic_ns() // 1_000_000  # convert to milliseconds


@dataclass(slots=True)
class DaemonConfig:
    # just online wait time (unstable time); wait time after offline to consider online (ms)
    set_online_jitter_time_ms: int

    # time to consider offline if no update received (ms)
    set_offline_time_ms: int

    # optional callback on offline event; called when offline is detected to handle it. None if not needed.
    offline_callback: Optional[OfflineCallback] = None

    # optional callback to check if data is erroneous; None if not needed.
    data_is_error_fun: Optional[DataErrorCheck] = None

    # optional callback to solve data error; None if not needed.
    solve_data_error_fun: Optional[DataErrorSolver] = None

    # optional owner identifier for the daemon instance
    owner_id: Any = None


class Daemon:
    """Watchdog for one module/device.

    Tracks online/offline status based on reload timing and optional data checks.

    Example:
        system = DaemonSystem()
        spy = DaemonSpy(window_size=10)
        config = DaemonConfig(
            set_online_jitter_time_ms=200,
            set_offline_time_ms=1000,
            owner_id="motor-1",
        )  # optional callback etc. can be provided in config
        daemon = Daemon(config, system=system, spy=spy)
        daemon.reload()  # feed the watchdog when new data arrives
        system.update_all()  # call periodically (e.g., in a loop)
    """
    # static class variable
    TIME_RESOLUTION_HZ: ClassVar[int] = 1000

    def __init__(
        self,
        config: DaemonConfig,
        *,
        time_provider: Optional[Callable[[], int]] = None,  # function that provides current time in milliseconds. None to use default.
        system: Optional[DaemonSystem] = None,  # daemon system (a set of daemons) to attach to. None for no system.
        auto_register: bool = True,  # True will auto register to the system if provided.
        spy: Optional[DaemonSpy] = None,  # rolling window stats helper; None to disable.
    ) -> None:
        if config.set_online_jitter_time_ms < 0 or config.set_offline_time_ms < 0:
            raise ValueError("offline/jitter times must be non-negative")

        self.set_online_jitter_time_ms = int(config.set_online_jitter_time_ms)
        self.set_offline_time_ms = int(config.set_offline_time_ms)
        self.offline_callback = config.offline_callback
        self.data_is_error_fun = config.data_is_error_fun
        self.solve_data_error_fun = config.solve_data_error_fun
        self.owner_id = config.owner_id

        self.new_time_ms = 0  # timestamp of the latest data received
        self.last_time_ms = 0  # timestamp of the previous data received
        self.lost_time_ms = 0  # timestamp when the instance was marked offline
        self.work_time_ms = 0  # timestamp when the instance was last marked online

        self.error_exist = False  # True if any error exists. Equivalent: (is_lost or data_is_error)
        self.is_lost = False  # True if the instance is offline
        self.data_is_error = False  # True if the latest data is erroneous

        self.delta_ms = 0  # time difference between the last two data received in milliseconds
        self.frequency_hz = 0.0  # calculated est. data update frequency in Hz
        self.min_delta_ms = 0  # rolling window min delta; 0 when spy disabled
        self.max_delta_ms = 0  # rolling window max delta; 0 when spy disabled

        # set up time provider and daemon system
        self._time_provider = time_provider or _default_time_ms
        self._spy = spy
        self._system = system
        if self._system is not None and auto_register:
            self._system.register(self)

    @property
    def system(self) -> Optional[DaemonSystem]:
        """Get the attached daemon system."""
        return self._system

    @property
    def spy(self) -> Optional[DaemonSpy]:
        """Get the attached daemon spy."""
        return self._spy

    def attach_system(self, system: DaemonSystem, *, register: bool = True) -> None:
        """Attch to a daemon system.

        Args:
            system (DaemonSystem): daemon system (a list of daemons) to attach to.
            register (bool, optional): False to not register to the system upon attaching. Defaults to True.
        """
        if self._system is system:
            return
        if self._system is not None:
            self._system.unregister(self)
        self._system = system
        if register:
            self._system.register(self)

    def detach_system(self) -> None:
        """Unregister and detach from a daemon system."""
        if self._system is None:
            return
        self._system.unregister(self)
        self._system = None

    def attach_spy(self, spy: DaemonSpy) -> None:
        """Attach a rolling window spy.

        Args:
            spy (DaemonSpy): daemon spy to attach to.
        """
        self._spy = spy

    def detach_spy(self) -> None:
        """Detach the rolling window spy."""
        self._spy = None
        self.min_delta_ms = 0
        self.max_delta_ms = 0

    def _now_ms(self, now_ms: Optional[int] = None) -> int:
        """Get the current time in milliseconds.

        Args:
            now_ms (Optional[int], optional): override time in milliseconds. Defaults to None.

        Returns:
            int: current time in milliseconds.
        """
        return self._time_provider() if now_ms is None else int(now_ms)

    def _record_delta(self, delta_ms: int) -> None:
        if delta_ms <= 0:
            return
        self.delta_ms = delta_ms
        if self._spy is None:
            self.frequency_hz = self.TIME_RESOLUTION_HZ / float(self.delta_ms)
            self.min_delta_ms = 0
            self.max_delta_ms = 0
            return

        self._spy.record(self.delta_ms)
        self.frequency_hz = self._spy.frequency_hz
        self.min_delta_ms = self._spy.min_delta_ms
        self.max_delta_ms = self._spy.max_delta_ms

    def reload(self, now_ms: Optional[int] = None) -> None:
        """Reload the daemon with current time when new data is received.

        Args:
            now_ms (Optional[int], optional): override time in milliseconds. Defaults to None.
        """
        now = self._now_ms(now_ms)
        self.last_time_ms = self.new_time_ms
        self.new_time_ms = now
        if self.new_time_ms > self.last_time_ms:
            self._record_delta(self.new_time_ms - self.last_time_ms)

        if self.is_lost:
            self.is_lost = False
            self.work_time_ms = now

        if self.data_is_error_fun is not None:
            if self.data_is_error_fun():
                self.error_exist = True
                self.data_is_error = True
                if self.solve_data_error_fun is not None:
                    self.solve_data_error_fun()
            else:
                self.data_is_error = False
        else:
            self.data_is_error = False

    def is_online(self) -> bool:
        """Whether the daemon is online.

        Returns:
            bool: True if online.
        """
        return not self.is_lost

    def is_offline(self) -> bool:
        """Whether the daemon is offline.

        Returns:
            bool: True if offline.
        """
        return self.is_lost

    def is_error(self) -> bool:
        """Whether any error exists (offline or data error).

        Returns:
            bool: True if any error exists.
        """
        return self.error_exist

    def update(self, now_ms: Optional[int] = None) -> None:
        """Background task/function to update the daemon status.

        Args:
            now_ms (Optional[int], optional): override time in milliseconds. Defaults to None.
        """
        current_time = self._now_ms(now_ms)

        # to jusge offline/online status; one of the following scenarios:
        if (current_time > self.new_time_ms and current_time - self.new_time_ms > self.set_offline_time_ms):
            # offline
            if not self.is_lost:
                # mark offline/error and timestamp
                self.is_lost = True
                self.error_exist = True
                self.lost_time_ms = current_time
            # call offline callback if exists
            if self.offline_callback is not None:
                self.offline_callback(self.owner_id)  # owner_id is passed so that the callback can identify the specific owner
        elif (current_time > self.new_time_ms and current_time - self.work_time_ms < self.set_online_jitter_time_ms):
            # just online (connected after offline; unstable time), within jitter, still consider error
            self.is_lost = False
            self.error_exist = True  # only mark as error
        else:
            # online normally
            self.is_lost = False
            self.error_exist = bool(self.data_is_error)  # mark error only if data error exists


class DaemonSpy:
    """Rolling window statistics for daemon update intervals. Stats include min/max intervals, frequency.
    window_size sets how many samples to keep for stats calculation.

    Example:
        spy = DaemonSpy(window_size=10)
        daemon = Daemon(DaemonConfig(50, 200), spy=spy)
    """
    TIME_RESOLUTION_HZ: ClassVar[int] = 1000

    def __init__(self, window_size: int) -> None:
        self.window_size = int(window_size)
        if self.window_size <= 0:
            raise ValueError("window_size must be > 0")
        self._delta_samples: deque[int] = deque(maxlen=self.window_size)
        self.delta_ms = 0
        self.min_delta_ms = 0
        self.max_delta_ms = 0
        self.frequency_hz = 0.0

    def reset(self) -> None:
        """Reset the rolling window statistics."""
        self._delta_samples.clear()
        self.delta_ms = 0
        self.min_delta_ms = 0
        self.max_delta_ms = 0
        self.frequency_hz = 0.0

    def record(self, delta_ms: int) -> None:
        """Record a new sample.

        Args:
            delta_ms (int): time difference in milliseconds.
        """
        if delta_ms <= 0:
            return
        self.delta_ms = delta_ms
        self._delta_samples.append(delta_ms)
        self.min_delta_ms = min(self._delta_samples) if self._delta_samples else 0
        self.max_delta_ms = max(self._delta_samples) if self._delta_samples else 0
        total_ms = sum(self._delta_samples)
        if total_ms <= 0:
            self.frequency_hz = 0.0
        else:
            count = len(self._delta_samples)
            self.frequency_hz = (self.TIME_RESOLUTION_HZ * count) / float(total_ms)


class DaemonSystem:
    """Registry for a group of daemon instances.

    Each system maintains its own instance list so multiple watchdog groups
    remain isolated.

    Example:
        system = DaemonSystem()
        a = Daemon(DaemonConfig(50, 200), system=system)
        b = Daemon(DaemonConfig(50, 200), system=system)
        system.update_all()
    """
    def __init__(self) -> None:
        self._instances: list[Daemon] = []

    @property
    def instances(self) -> tuple[Daemon, ...]:
        """Get all registered daemon instances."""
        return tuple(self._instances)

    def register(self, instance: Daemon) -> None:
        """Register a daemon instance to the system.

        Args:
            instance (Daemon): daemon to be registered.
        """
        if instance in self._instances:
            return
        self._instances.append(instance)

    def unregister(self, instance: Daemon) -> None:
        """Unregister a daemon instance from the system.

        Args:
            instance (Daemon): daemon to be unregistered/removed.
        """
        try:
            self._instances.remove(instance)
        except ValueError:
            pass

    def update_all(self, now_ms: Optional[int] = None) -> None:
        """Background task/function to update all daemons in the system

        Args:
            now_ms (Optional[int], optional): override time in milliseconds. Defaults to None.
        """
        if now_ms is None:
            for instance in self._instances:
                instance.update()
        else:
            now_ms = int(now_ms)
            for instance in self._instances:
                instance.update(now_ms)
