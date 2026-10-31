import numpy as np

class EKF:
    def __init__(self, x0, P0):
        self.x = np.atleast_1d(x0).astype(float)    # state estimate
        self.P = np.atleast_2d(P0).astype(float)    # covariance
        self.I = np.eye(len(self.x))

    def predict(self, fx, F, Q):
        self.x = np.atleast_1d(fx)                  # x_pred = f(x_hat)
        self.P = F @ self.P @ F.T + Q               # P_pred = F P F^T + Q

    def update(self, z, hx, H, R):
        S = H @ self.P @ H.T + R                    # innovation covariance
        G = self.P @ H.T @ np.linalg.inv(S)         # gain
        self.x = self.x + G @ (np.atleast_1d(z) - np.atleast_1d(hx))
        self.P = (self.I - G @ H) @ self.P