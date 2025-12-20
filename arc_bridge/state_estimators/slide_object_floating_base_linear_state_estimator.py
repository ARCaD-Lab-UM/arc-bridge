import numpy as np
from .kalman_filter import KalmanFilter


class SlideObjectFloatingBaseLinearStateEstimator(KalmanFilter):
    def __init__(self, dt, Q, R, height_init):
        dim_state = 9  # (px, py, pz, vx, vy, vz, ax, ay, az)
        dim_control = 1  # (0)
        dim_obs = 9  # (px, py, pz, vx, vy, vz, ax, ay, az)

        # construct A matrix
        A = np.eye(dim_state)
        # p(k+1) = p + v dt + 0.5 a dt^2
        A[0:3, 3:6] = dt * np.eye(3)
        A[0:3, 6:9] = 0.5 * dt**2 * np.eye(3)
        # v(k+1) = v + a dt
        A[3:6, 6:9] = dt * np.eye(3)
        # a(k+1) = a is included in identity

        # construct B matrix
        B = np.zeros((dim_state, dim_control))

        # construct C matrix
        C = np.eye(dim_state, dim_obs)

        # initial state and covariance
        x_init = np.array([0, 0, height_init, 0, 0, 0, 0, 0, 0])  # initial pos, vel, acc in 3D
        P_init = np.eye(dim_state) * 1e-5 # initial state covariance
        super().__init__(A, B, C, Q, R, x_init, P_init)
