import numpy as np

class SimpleStepInterpolator:
    """
    Simple linear step interpolator from start to target; call step() to advance one step.
    Usage:
      interp.start(start_vec, target_vec, num_steps)
      y = interp.step() # returns current interpolated value
      if interp.done: ...
    """
    def __init__(self, dim: int):
        self.dim = int(dim)
        self._start = np.zeros(self.dim, dtype=float)
        self._target = np.zeros(self.dim, dtype=float)
        self._num_steps = 1
        self._k = 0
        self._y = self._start.copy()
        self.done = True

    def start(self, start_vec: np.ndarray, target_vec: np.ndarray, num_steps: int):
        assert num_steps > 0, "num_steps must be > 0"
        self._start = np.array(start_vec, dtype=float).reshape(self.dim)
        self._target = np.array(target_vec, dtype=float).reshape(self.dim)
        self._num_steps = int(num_steps)
        self._k = 0
        self.done = False
        self._y = self._start.copy()

    def step(self) -> np.ndarray:
        if self.done:
            return self._y
        alpha = self._k / float(self._num_steps)
        if alpha > 1.0:
            alpha = 1.0
        self._y = (1.0 - alpha) * self._start + alpha * self._target
        self._k += 1
        if self._k >= self._num_steps:
            self.done = True
        return self._y

    @property
    def value(self) -> np.ndarray:
        return self._y
