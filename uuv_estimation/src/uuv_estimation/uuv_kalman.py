from uuv_estimation.kalman_base import KalmanFilter

import numpy as np

class UUVKalman(KalmanFilter):
    def __init__(self):
        # generate F matrix for an uuv that has 6 DOF
        F = np.eye(12)
        dt = 0.0
        F[0, 6] = dt # x velocity
        F[1, 7] = dt # y velocity
        F[2, 8] = dt # z velocity
        F[3, 9] = dt # roll rate
        F[4, 10] = dt # pitch rate
        F[5, 11] = dt # yaw rate

        # generate G matrix for an uuv that has 6 DOF
        G = np.zeros((12, 6)) # zero matrix for now

        # generate H matrix for an uuv that has 6 DOF
        H = np.eye(12)

        super().__init__(F, G, H)

    def makePrediction(self, u, dt):
        self.predict(u, dt)

    def updateFromStateMeasurement(self, z):
        # update H Matrix
        H = np.eye(12)

        # update R matrix
        R = np.eye(H.shape[0]) * 100

        # update z
        self.update(z, H, R)

    def updateFromIMU(self, z):
        # update H Matrix
        H = np.zeros((12, 12))
        H[3, 3] = 1
        H[4, 4] = 1
        H[5, 5] = 1

        # update R matrix
        cov = 0.5
        R = np.eye(H.shape[0]) * cov

        # update z
        roll = z[3] + np.random.normal(0, 0.02)
        pitch = z[4] + np.random.normal(0, 0.02)
        yaw = z[5] + np.random.normal(0, 0.02)
        vroll = z[9] + np.random.normal(0, 0.01)
        vpitch = z[10] + np.random.normal(0, 0.01)
        vyaw = z[11] + np.random.normal(0, 0.01)

        z = np.array([0, 0, 0, roll, pitch, yaw, 0, 0, 0, vroll, vpitch, vyaw])
        
        self.update(z, H, R)

    def updateFromPressure(self, z):
        # update H Matrix
        H = np.zeros((12, 12))
        H[2, 2] = 1

        # update R matrix
        cov = 1.0
        R = np.eye(H.shape[0]) * cov

        # get depth from z
        depth = z[2] + np.random.normal(0, 0.1)

        # update z
        z = np.array([0, 0, depth, 0, 0, 0, 0, 0, 0, 0, 0, 0])

        self.update(z, H, R)

    def updateFromBottomSonar(self, z):
        # update H Matrix
        H = np.zeros((12, 12))
        H[2, 2] = 1

        # update R matrix
        cov = 0.1
        R = np.eye(H.shape[0]) * cov

        # get depth from z
        depth = z[2] + np.random.normal(0, 0.05)

        # update z
        z = np.array([0, 0, depth, 0, 0, 0, 0, 0, 0, 0, 0, 0])

        self.update(z, H, R)

    def updateFromDVL(self, z): # vx, vy, vz, z
        # update H Matrix
        H = np.zeros((12, 12))
        H[2,2] = 1
        H[6,6] = 1
        H[7,7] = 1
        H[8,8] = 1

        # update R matrix
        cov = 0.1
        R = np.eye(H.shape[0]) * cov

        # update z
        depth = z[2] + np.random.normal(0, 0.025)
        vx = z[6] + np.random.normal(0, 0.025)
        vy = z[7] + np.random.normal(0, 0.025)
        vz = z[8] + np.random.normal(0, 0.025)

        z = np.array([0, 0, depth, 0, 0, 0, vx, vy, vz, 0, 0, 0])

        self.update(z, H, R)

    def updateFromCamera(self, z): # [x, y, z]
        # update H Matrix
        H = np.zeros((12, 12))
        H[0,0] = 1
        H[1,1] = 1
        H[2,2] = 1

        # update R matrix
        cov = 10
        R = np.eye(H.shape[0]) * cov

        # update z
        x_ = z[0] + np.random.normal(0, 0.05)
        y_ = z[1] + np.random.normal(0, 0.05)
        z_ = z[2] + np.random.normal(0, 0.05)

        z = np.array([x_, y_, z_, 0, 0, 0, 0, 0, 0, 0, 0, 0])

        self.update(z, H, R)