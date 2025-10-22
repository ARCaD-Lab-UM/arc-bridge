import numpy as np


class OnlineAverage:
  """A numerically stable O(1) online average for streaming vectors.

  We track only the current mean and the count. Each update uses the
  incremental rule:
      mean_{k+1} = mean_k + (x_{k+1} - mean_k) / (k+1)
  which is more stable than recomputing via (mean*k + x)/(k+1).
  """

  def __init__(self, dim: int = 3):
    """Initializes the class.

    Args:
      dim: Dimension of each incoming vector.
    """
    assert dim > 0
    self._dim = dim
    self._count = 0
    self._mean = np.zeros(dim, dtype=float)

  @property
  def count(self) -> int:
    """Number of samples incorporated so far."""
    return self._count

  @property
  def average(self) -> np.ndarray:
    """Current average (zeros if no data yet)."""
    return self._mean

  def reset(self):
    """Clears the state."""
    self._count = 0
    self._mean = np.zeros(self._dim, dtype=float)

  def update(self, new_value: np.ndarray) -> np.ndarray:
    """Consumes a new vector and returns the updated average.

    Args:
      new_value: The new vector sample (shape `(dim,)`).

    Returns:
      The updated running average.
    """
    x = np.asarray(new_value, dtype=float)
    if x.shape != (self._dim,):
      raise ValueError(f"Expected shape ({self._dim},), got {x.shape}")

    # Incremental mean update: mean += (x - mean) / (n + 1)
    if self._count == 0:
      self._mean = x.copy()
      self._count = 1
    else:
      self._count += 1
      self._mean += (x - self._mean) / self._count
    return self._mean
