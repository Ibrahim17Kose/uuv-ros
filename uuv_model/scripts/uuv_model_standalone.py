#!/usr/bin/env python3

import rospy
import rospkg
import numpy as np
from yaml import safe_load

from uuv_model.utils import *
from uuv_msgs.msg import ControlSignal, States


class UUV:
    def __init__(self, uuv_name: str = "rexrov2"):
        rospy.init_node('uuv_model', anonymous=True)
        
        self.uuv_name = uuv_name

        self.cfg = {}
        self.parse_config()

        self.dt = self.cfg["dt"]

        # Variables
        self.prev_thrust = np.zeros(6)
        self.prev_states = {"eta": self.cfg["eta_0"], "nu": self.cfg["nu_0"], "eta_dot": np.zeros(6), "nu_dot": np.zeros(6)}
    
        # Topics
        self.states = {"eta": np.zeros(6), "nu": np.zeros(6), "eta_dot": np.zeros(6), "nu_dot": np.zeros(6)}
        self.current = {"N": 0, "E": 0, "D": 0}  # TODO: Add current to ROS params

        # ROS
        rospy.Subscriber("control_signal", ControlSignal, self.callback_control_signal)
        self.pub_states = rospy.Publisher('states', States, queue_size=1)

        rospy.loginfo("STANDALONE UUV MODEL Node is Up ...")

    def parse_config(self):
        with open(rospkg.RosPack().get_path("uuv_model") + f"/config/{self.uuv_name}.yaml", 'r') as file:
            self.cfg = safe_load(file)
        
        self.cfg["I_g"] = np.eye(3) * [self.cfg["Ix"], self.cfg["Iy"], self.cfg["Iz"]]
        
        # Rigid body mass
        self.cfg["M_rb"] = np.concatenate((np.concatenate((self.cfg["m"]*np.eye(3), -self.cfg["m"]*smtrx(self.cfg["g_center"])), axis=1),
                                           np.concatenate((self.cfg["m"]*smtrx(self.cfg["g_center"]), self.cfg["I_g"]), axis=1)), axis=0)
     
        # Mass Matrix
        self.cfg["M"] = self.cfg["M_rb"] + self.cfg["M_a"]

        # Linear drag matrix
        linear_drag = [self.cfg["linear_drag"][0]["Xu_l"], self.cfg["linear_drag"][1]["Yv_l"], self.cfg["linear_drag"][2]["Zw_l"],
                       self.cfg["linear_drag"][3]["Kp_l"], self.cfg["linear_drag"][4]["Mq_l"], self.cfg["linear_drag"][5]["Nr_l"]]
        self.cfg["D_l"] = np.eye(6) * linear_drag

        # Quadratic drag matrix
        quadratic_drag = [self.cfg["quadratic_drag"][0]["Xu_q"], self.cfg["quadratic_drag"][1]["Yv_q"], self.cfg["quadratic_drag"][2]["Zw_q"],
                          self.cfg["quadratic_drag"][3]["Kp_q"], self.cfg["quadratic_drag"][4]["Mq_q"], self.cfg["quadratic_drag"][5]["Nr_q"]]
        self.cfg["D_q"] = np.eye(6) * quadratic_drag

        # Weight
        self.cfg["W"] = self.cfg["m"] * self.cfg["g"]

        # Buoyancy
        self.cfg["B"] = self.cfg["rho"] * self.cfg["g"] * self.cfg["V"]

        # Initial states
        self.cfg["eta_0"] = ned2enu([self.cfg["X_enu"], self.cfg["Y_enu"], self.cfg["Z_enu"],
                                     np.deg2rad(self.cfg["Roll_enu"]),
                                     np.deg2rad(self.cfg["Pitch_enu"]),
                                     np.deg2rad(self.cfg["Yaw_enu"])])
        self.cfg["nu_0"] = [0, 0, 0, 0, 0, 0]
        return

    def propulsion(self, control_signal):
        K_inv = np.linalg.pinv(self.cfg["K"])
        u = np.clip(np.matmul(K_inv, control_signal), self.cfg["thruster_lower_limit"], self.cfg["thruster_upper_limit"])
        
        motor_ang_vel = np.sign(u) * np.sqrt(np.abs(u) / self.cfg["thruster_gain"])  

        # FO Thruster
        ref = self.cfg["thruster_gain"] * motor_ang_vel * abs(motor_ang_vel)
        alpha = np.exp(- self.dt / self.cfg["thruster_tau"])
        thrust = alpha * self.prev_thrust + (1 - alpha) * ref
        self.prev_thrust = thrust

        tau = np.matmul(self.cfg["K"], thrust)
        return tau
    
    def kinematics(self):
        eta_dot = np.matmul(body2ned(self.states["eta"]), self.states["nu"])
        return eta_dot

    def rigid_body_dynamics(self):
        coriolis_rb = np.matmul(m2c(self.cfg["M_rb"], self.states["nu"]), self.states["nu"])
        return coriolis_rb

    def hydrodynamics(self, nu_r):
        damping = np.matmul(self.cfg["D_l"] + np.multiply(self.cfg["D_q"], np.abs(nu_r)) , nu_r)
        coriolis_am = np.matmul(m2c(self.cfg["M_a"], self.states["nu"]), self.states["nu"])
        added_mass = np.matmul(self.cfg["M_a"], self.states["nu_dot"])
        return damping, coriolis_am, added_mass
    
    def hydrostatics(self):
        z = self.states["eta"][2]
        phi = self.states["eta"][3]
        theta = self.states["eta"][4]
        
        if ( z - (self.cfg["height"] / 2) < 0 and z + (self.cfg["height"] / 2) > 0 ):
            volume = self.cfg["V"] * ((self.cfg["height"] / 2) + z) / self.cfg["height"]
        elif ( z + (self.cfg["height"] / 2) < 0 ):
            volume = 0
        else:
            volume = self.cfg["V"]
        
        B = volume * self.cfg["rho"] * self.cfg["g"]

        g_n = gvect(self.cfg["W"], B, theta, phi, self.cfg["g_center"], self.cfg["b_center"])
        return g_n

    def callback_control_signal(self, msg):
        control_signal = msg.data

        nu_ned = [self.current["N"], self.current["E"], self.current["D"]]
        nu_c = np.matmul(ned2body(self.states["eta"])[0:3, 0:3], nu_ned)
        nu_r = self.states["nu"] - np.concatenate((nu_c, np.zeros(3)), axis=0)
        
        tau = self.propulsion(control_signal)
        damping, coriolis_am, added_mass = self.hydrodynamics(nu_r)
        coriolis_rb = self.rigid_body_dynamics()
        g_n = self.hydrostatics()

        self.states["eta_dot"] = self.kinematics()
        self.states["nu_dot"] = np.matmul(np.linalg.inv(self.cfg["M"]), (tau - added_mass - coriolis_rb - coriolis_am - damping - g_n))

        self.states["eta"] = self.prev_states["eta"] + ((self.states["eta_dot"] + self.prev_states["eta_dot"]) / 2) * self.dt
        self.states["nu"] = self.prev_states["nu"] + ((self.states["nu_dot"] + self.prev_states["nu_dot"]) / 2) * self.dt
        
        self.prev_states = self.states

        msg = States()

        msg.eta = ned2enu(self.states["eta"])
        msg.nu = ned2enu(self.states["nu"])

        self.pub_states.publish(msg)


if __name__ == '__main__':
    try:
        uuv_model = UUV()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
