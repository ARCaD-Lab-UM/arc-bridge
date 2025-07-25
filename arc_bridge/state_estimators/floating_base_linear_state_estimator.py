import numpy as np
from scipy.linalg import expm
from .kalman_filter import KalmanFilter

class FloatingBaseLinearStateEstimator(KalmanFilter):
    
    def __init__(self, dt, Q, R, height_init):
        dim_state = 6   # (px, py, pz, vx, vy, vz)
        dim_control = 3 # (ax, ay, az)
        dim_obs = 4     # (pz, vx, vy, vz)
        Ac = np.zeros((dim_state, dim_state))
        Ac[:3, 3:] = np.eye(3)
        Bc = np.zeros((dim_state, dim_control))
        Bc[3:] = np.eye(3)
        # print(f"Ac:\n{Ac}")
        # print(f"Bc:\n{Bc}")

        A = expm(dt * Ac)
        B = dt * Bc
        C = np.zeros((dim_obs, dim_state))
        C[:, 2:] = np.eye(4)
        # print(f"C:\n{C}")

        self.x_init = np.array([0, 0, height_init, 0, 0, 0])   # initial pos and vel in 3D
        self.P_init = np.eye(dim_state) * 1e-5         # initial state covariance
        super().__init__(A, B, C, Q, R, self.x_init, self.P_init)

