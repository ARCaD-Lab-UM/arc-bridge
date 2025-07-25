import numpy as np
from .kalman_filter import KalmanFilter

# TODO deprecate this class
class HopperStateEstimator(KalmanFilter):

    def __init__(self, dt):
        dt = dt
        dim_state = 6   # (px, py, pz, vx, vy, vz)
        dim_control = 3 # (ax, ay, az)
        dim_obs = 4     # (pz, vx, vy, vz)
        Ac = np.zeros((dim_state, dim_state))
        Ac[:3, 3:] = np.eye(3)
        Bc = np.zeros((dim_state, dim_control))
        Bc[3:] = np.eye(3)
        # print(f"Ac:\n{Ac}")
        # print(f"Bc:\n{Bc}")

        A = np.eye(dim_state) + dt * Ac
        B = dt * Bc
        C = np.zeros((dim_obs, dim_state))
        C[:, 2:] = np.eye(4)
        # print(f"C:\n{C}")

        # Process noise (px, py, pz, vx, vy, vz)
        Q = np.diag([0.002, 0.002, 0.002, 0.02, 0.02, 0.02]) * 1e-3
        # Measurement noise (pz, vx, vy, vz)
        R = np.diag([0.001, 0.1, 0.1, 0.1])
        # TODO scale up any covariance related foot during swing, assume alway in contact

        self.x_init = np.array([0, 0, 0.32, 0, 0, 0]) # initial pos and vel in 3D
        self.P_init = np.eye(dim_state) * 1e-5         # initial state covariance
        super(HopperStateEstimator, self).__init__(A, B, C, Q, R, self.x_init, self.P_init)

        self.l1 = 0.22
        self.l2 = 0.22
        self.p_hip = np.array([0, 0, -0.07])

    def foot_pos_body_frame(self, qj_pos):
        l1 = self.l1
        l2 = self.l2
        th1 = qj_pos[0]
        th2 = qj_pos[1]

        p_foot = np.array([- l1 * np.sin(th1) - l2 * np.sin(th1 + th2), 
                           0,
                           - l1 * np.cos(th1) - l2 * np.cos(th1 + th2)])
        
        return self.p_hip + p_foot

    def foot_vel_body_frame(self, qj_pos, qj_vel):
        l1 = self.l1
        l2 = self.l2
        th1 = qj_pos[0]
        th2 = qj_pos[1]
    
        jacobian = np.array([
            -l1 * np.cos(th1) - l2 * np.cos(th1 + th2), -l2 * np.cos(th1 + th2),
             l1 * np.sin(th1) + l2 * np.sin(th1 + th2),  l2 * np.sin(th1 + th2)
            ]).reshape((2, 2))
        
        v_foot = jacobian @ qj_vel
        
        return np.array([v_foot[0], 0, v_foot[1]])
