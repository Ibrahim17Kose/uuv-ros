#!/usr/bin/env python3

import rospy
import numpy as np

from uuv_model.utils import ned2body


class MimoNonlinearPid:
    def __init__(self, cfg: dict):
        self.cfg = cfg

        self.prev_integral = np.zeros(6)
        self.prev_error = np.zeros(6)

        self.dt = 0.001  # TODO: real time
    
    def update(self, reference_signal: dict, states: dict):
        eta = np.array(states["eta"])
        eta_dot = np.array(states["eta_dot"])
        nu_dot = np.array(states["nu_dot"])

        eta_ref = np.array(reference_signal["eta_ref"])

        error = eta - eta_ref
        p = np.multiply(self.cfg["Kp"], error)

        integral =  self.prev_integral + ((error + self.prev_error) / 2) * self.dt
        i = np.multiply(self.cfg["Ki"], integral)

        d = np.multiply(self.cfg["Kd"], eta_dot)

        pid = np.matmul(ned2body(eta), -(p + i + d)) - self.cfg["acc_fb_gain"] * nu_dot

        control_signal = np.clip(pid, self.cfg["lower_limit"], self.cfg["upper_limit"])
        
        self.prev_error = error
        self.prev_integral = integral

        return control_signal