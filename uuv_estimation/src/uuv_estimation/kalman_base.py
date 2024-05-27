import numpy as np

class KalmanFilter:
    def __init__(self, F, G, H):
        self.F = F
        self.G = G
        self.H = H
        self.x = np.zeros(F.shape[0])
        self.P = np.eye(F.shape[0])

        self.K = np.zeros((F.shape[0], H.shape[0]))
        self.Q = np.eye(F.shape[0]) * 0.001
        self.R = np.eye(H.shape[0])

    def predict(self, u, dt):
        # update F matrix
        self.F[0, 6] = dt
        self.F[1, 7] = dt
        self.F[2, 8] = dt
        self.F[3, 9] = dt
        self.F[4, 10] = dt
        self.F[5, 11] = dt

        # predict state
        self.x = self.F @ self.x + self.G @ u

        # predict state covariance
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, z, H, R):

        self.H = H
        self.R = R

        # calculate innovation
        y = z - self.H @ self.x

        # calculate K gain
        S = self.H @ self.P @ self.H.T + self.R
        self.K = self.P @ self.H.T @ np.linalg.inv(S)

        # update state
        self.x = self.x + self.K @ y

        # update state covariance
        T = np.eye(self.K.shape[0]) - self.K @ self.H
        self.P = T @ self.P @ T.T + self.K @ self.R @ self.K.T

    def getState(self):
        return self.x
