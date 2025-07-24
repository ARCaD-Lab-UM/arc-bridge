import numpy as np
from scipy.linalg import expm
from .kalman_filter import KalmanFilter

class Tron1StateEstimator(KalmanFilter):
    
    # Process noise:
    # px, py, pz, 
    # vx, vy, vz, 
    # pfxR, pfyR, pfzR, 
    # pfxL, pfyL, pfzL
    noise_pimu = 0.02
    noise_vimu = 0.02
    noise_pfoot = 0.002
    KF_Q = np.eye(12)
    KF_Q[:3, :3] *= noise_pimu
    KF_Q[3:6, 3:6] *= noise_vimu
    KF_Q[6:, 6:] *= noise_pfoot
    # Measurement noise:
    # px - pfxR, py - pfyR, pz - pfzR, 
    # px - pfxL, py - pfyL, pz - pfzL, 
    # pfzR, pfzL
    noise_pimu_rel_foot = 0.001
    noise_vimu_rel_foot = 0.1
    noise_zfoot = 0.001
    KF_R = np.eye(8)
    KF_R[:6, :6] *= noise_pimu_rel_foot
    KF_R[6:, 6:] *= noise_zfoot

    def __init__(self, dt, Q, R, height_init):
        # px, py, pz, 
        # vx, vy, vz, 
        # pfxR, pfyR, pfzR, 
        # pfxL, pfyL, pfzL
        dim_state = 12
        # ax, ay, az
        dim_input = 3
        # px - pfxR, py - pfyR, pz - pfzR, 
        # px - pfxL, py - pfyL, pz - pfzL, 
        # pfzR, pfzL
        dim_measurement = 8     
        Ac = np.zeros((dim_state, dim_state))
        Ac[:3, 3:6] = np.eye(3)
        Bc = np.zeros((dim_state, dim_input))
        Bc[3:6, :3] = np.eye(3)
        # print(f"Ac:\n{Ac}")
        # print(f"Bc:\n{Bc}")

        A = expm(dt * Ac)
        B = dt * Bc
        C = np.zeros((dim_measurement, dim_state))
        C[:3, :3] = np.eye(3)
        C[:3, 6:9] = -np.eye(3)
        C[3:6, :3] = np.eye(3)
        C[3:6, 9:12] = -np.eye(3)
        C[6, 8] = 1
        C[7, 11] = 1
        # print(f"C:\n{C}")

        self.x_init = np.zeros(dim_state)   # initial pos and vel in 3D
        self.x_init[:3] = height_init
        self.P_init = np.eye(dim_state) * 1e-5         # initial state covariance
        super().__init__(A, B, C, Q, R, self.x_init, self.P_init)

