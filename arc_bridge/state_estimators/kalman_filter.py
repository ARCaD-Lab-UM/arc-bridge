import numpy as np


class KalmanFilter:
    def __init__(self, A, B, C, Q, R, x_init, P_init):
        # x_kp1 = A*xk + B*uk + wk
        # yk = C*xk + vk
        # x0 ~ N(x_init, P_init)
        # wk ~ N(0, Qk)
        # vk ~ N(0, Rk)
        dim_state = A.shape[1]
        dim_control = B.shape[1]
        dim_obs = C.shape[0]
        assert(A.shape == (dim_state, dim_state))
        assert(B.shape == (dim_state, dim_control))
        assert(C.shape == (dim_obs, dim_state))
        assert(Q.shape == (dim_state, dim_state))
        assert(R.shape == (dim_obs, dim_obs))
        assert(x_init.shape == (dim_state, ))
        assert(P_init.shape == (dim_state, dim_state))

        self.A = A.copy()
        self.B = B.copy()
        self.C = C.copy()
        self.Q = Q.copy()
        self.R = R.copy()
        self.I = np.eye(dim_state)
        self.x_init = x_init
        self.P_init = P_init
        self.reset()

    def reset(self, x_init=None):
        if x_init is None:
            self.x = self.x_init.copy()
        else:
            self.x = x_init.copy()
        self.P = self.P_init.copy()

    def get_state(self):
        return self.x.copy()

    def predict(self, u):
        # Predict state ahead
        self.x = self.A @ self.x + self.B @ u
        # Update prediction error covariance
        self.P = self.A @ self.P @ self.A.T + self.Q

        return self.x.copy()

    def correct(self, y):
        # Compute the Kalman gain
        K = self.P @ self.C.T @ np.linalg.inv(self.C @ self.P @ self.C.T + self.R)
        # Correct estimation
        self.x = self.x + K @ (y - self.C @ self.x)
        # Update measurement error covariance
        self.P = (self.I - K @ self.C) @ self.P
        
        return self.x.copy()
