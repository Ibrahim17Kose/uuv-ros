#!/usr/bin/env python3

import rospy
import numpy as np

from uuv_model.utils import *


class MimoNonlinearPid:
    def __init__(self, cfg: dict):
        self.cfg = cfg

        self.prev_integral = np.zeros(6)
        self.prev_error = np.zeros(6)
        
        self.eta_dot = np.zeros(6)
        self.last_eta = np.zeros(6)
        self.eta_dot_filter_alpha = 0.3

        self.nu_dot = np.zeros(6)
        self.last_nu = np.zeros(6)
        self.nu_dot_filter_alpha = 0.3

        self.dt = 0.001  # TODO: real time
    
    def compute_eta_dot(self, eta):
        vel = (eta - self.last_eta) / self.dt
        self.eta_dot = ((1 - self.eta_dot_filter_alpha) * self.eta_dot) + (self.eta_dot_filter_alpha * vel)
        self.last_eta = eta
        return self.eta_dot
    
    def compute_nu_dot(self, nu):
        acc = (nu - self.last_nu) / self.dt
        self.nu_dot = ((1 - self.nu_dot_filter_alpha) * self.nu_dot) + (self.nu_dot_filter_alpha * acc)
        self.last_nu = nu
        return self.nu_dot

    def update(self, reference_signal: dict, states: dict):
        eta = np.array(states["eta"])
        nu = np.array(states["nu"])

        eta_dot = self.compute_eta_dot(eta)
        nu_dot = self.compute_nu_dot(nu)

        eta_ref = np.array(reference_signal["eta_ref"])

        error = eta - eta_ref

        if error[5] > np.pi:
            error[5] -= 2 * np.pi
        elif error[5] < -np.pi:
            error[5] += 2 * np.pi
        
        np.set_printoptions(precision=4)
        print(error)

        p = np.multiply(self.cfg["Kp"], error)

        integral =  self.prev_integral + ((error + self.prev_error) / 2) * self.dt
        i = np.multiply(self.cfg["Ki"], integral)

        d = np.multiply(self.cfg["Kd"], eta_dot)

        pid = -(p + i + d) - self.cfg["acc_fb_gain"] * nu_dot

        pid_signal = np.matmul(ned2body(states["eta"]), pid)

        control_signal = np.clip(pid_signal, self.cfg["lower_limit"], self.cfg["upper_limit"])
        
        self.prev_error = error
        self.prev_integral = integral

        return control_signal
