import numpy as np
from typing import Optional, Union, Iterable

ArrayLike = Union[float, Iterable[float], np.ndarray]

# helper function to ensure input is a numpy vector of given dimension
def _as_vec(x: ArrayLike, dim: int) -> np.ndarray:
    return np.array(x, dtype=float).reshape(dim)


class FirstOrderLowPassTD:
    """
    First order Low Pass Filter with time constant and frame time (Time Dependent):
        y[n] = (tau/(tau+dt)) * y[n-1] + (dt/(tau+dt)) * x[n]
    Parameters:
        tau_sec: time constant tau s, > 0
        frame_dt: frame time delta t s, > 0, can be overridden in update
        dim: dimension of scalar or vector
        init: initial y value (scalar/vector)
    Usage:
        lpf = FirstOrderLowPassTD(tau_sec=0.02, frame_dt=0.001, dim=3)
        y = lpf.update(x) # use constructor frame_dt
        y = lpf.update(x, dt=0.00095) # current frame para dt (can be variable)
    """
    def __init__(self, tau_sec: float, frame_dt: float, dim: int = 1, init: Optional[ArrayLike] = None):
        assert tau_sec > 0.0, "tau_sec must be > 0"
        assert frame_dt > 0.0, "frame_dt must be > 0"
        self.tau = float(tau_sec)
        self.dt_default = float(frame_dt)
        self.dim = int(dim)
        self._y = _as_vec(init if init is not None else np.zeros(self.dim), self.dim)

    def reset(self, value: Optional[ArrayLike] = None):
        self._y[:] = 0.0 if value is None else _as_vec(value, self.dim)

    @property
    def value(self) -> np.ndarray:
        return self._y

    @staticmethod
    def coeffs(tau: float, dt: float):
        """
        Returns (k_prev, k_new) = (tau/(tau+dt), dt/(tau+dt))
        """
        s = tau + dt
        if s <= 0.0:
            return 0.0, 1.0
        return tau / s, dt / s

    def update(self, x: ArrayLike, dt: Optional[float] = None) -> np.ndarray:
        x = _as_vec(x, self.dim)
        dt_use = self.dt_default if dt is None else float(dt)
        assert dt_use > 0.0, "dt must be > 0"
        k_prev, k_new = self.coeffs(self.tau, dt_use)
        self._y = k_prev * self._y + k_new * x
        return self._y


class FirstOrderLowPassExp:
    """
    Exponential smoothing:
        y[n] = (1 - alpha) * y[n-1] + alpha * x[n]
    alpha in [0,1], when set to 1.0, no smoothing.
    """
    def __init__(self, alpha: float, dim: int = 1, init: Optional[ArrayLike] = None):
        assert 0.0 <= alpha <= 1.0, "alpha must be in [0,1]"
        self.alpha = float(alpha)
        self.dim = int(dim)
        self._y = _as_vec(init if init is not None else np.zeros(self.dim), self.dim)

    def reset(self, value: Optional[ArrayLike] = None):
        self._y[:] = 0.0 if value is None else _as_vec(value, self.dim)

    @property
    def value(self) -> np.ndarray:
        return self._y

    def update(self, x: ArrayLike) -> np.ndarray:
        x = _as_vec(x, self.dim)
        a = self.alpha
        self._y = (1.0 - a) * self._y + a * x
        return self._y


class SecondOrderLowPassSimple:
    """
    Simple second order low pass filter:
        y[n] = a0*y[n-1] + a1*y[n-2] + b0*x[n]
    choose stable coefficients.
    """
    def __init__(self, a0: float, a1: float, b0: float, dim: int = 1, init_y: Optional[ArrayLike] = None, init_y_1: Optional[ArrayLike] = None):
        self.a0, self.a1, self.b0 = float(a0), float(a1), float(b0)
        self.dim = int(dim)
        self._y1 = _as_vec(init_y if init_y is not None else np.zeros(self.dim), self.dim)
        self._y2 = _as_vec(init_y_1 if init_y_1 is not None else np.zeros(self.dim), self.dim)
        self._y = self._y1.copy()

    def reset(self, y1: Optional[ArrayLike] = None, y2: Optional[ArrayLike] = None):
        self._y1 = _as_vec(y1 if y1 is not None else np.zeros(self.dim), self.dim)
        self._y2 = _as_vec(y2 if y2 is not None else np.zeros(self.dim), self.dim)
        self._y = self._y1.copy()

    @property
    def value(self) -> np.ndarray:
        return self._y

    def update(self, x: ArrayLike) -> np.ndarray:
        x = _as_vec(x, self.dim)
        y = self.a0 * self._y1 + self.a1 * self._y2 + self.b0 * x
        self._y2 = self._y1
        self._y1 = y
        self._y = y
        return y
