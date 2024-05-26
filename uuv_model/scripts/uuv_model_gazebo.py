#!/usr/bin/env python3

import argparse
import rospy
import rospkg
import numpy as np
from yaml import safe_load

from uuv_model.utils import *
from uuv_msgs.msg import ControlSignal, States
from geometry_msgs.msg import Wrench, Vector3


parser = argparse.ArgumentParser(description="UUV Model Gazebo Script")
parser.add_argument("--uuv_name", default="rexrov2")
args = parser.parse_args(rospy.myargv()[1:])


class UUV:
    def __init__(self, uuv_name: str = "rexrov2"):
        rospy.init_node('uuv_model', anonymous=True)
        
        self.uuv_name = uuv_name
        
        self.cfg = {}
        self.parse_config()
        
        self.dt = self.cfg["dt"]
        self.rate = rospy.Rate(1 / self.dt)

        # Variables
        self.nu_dot_filter_alpha = 0.3
        self.nu_dot = np.zeros(6)
        self.last_nu = np.zeros(6)

        self.prev_thrust = np.zeros(self.cfg["thruster_num"])

        # Topics
        self.wrench = Wrench()
        self.control_signal = np.zeros(6)
        self.states = {"eta": np.zeros(6), "nu": np.zeros(6)}
        self.current = {"N": 0, "E": 0, "D": 0}  # TODO: Add current to ROS params
        
        # ROS
        rospy.Subscriber("/control_signal", ControlSignal, self.callback_control_signal)
        rospy.Subscriber("/states", States, self.callback_state)
        self.pub_wrench = rospy.Publisher('apply_wrench', Wrench, queue_size=1)

        rospy.loginfo("GAZEBO UUV MODEL Node is Up ...")

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
        self.cfg["eta_0"] = enu2ned([self.cfg["X_enu"], self.cfg["Y_enu"], self.cfg["Z_enu"],
                                     np.deg2rad(self.cfg["Roll_enu"]),
                                     np.deg2rad(self.cfg["Pitch_enu"]),
                                     np.deg2rad(self.cfg["Yaw_enu"])])
        self.cfg["nu_0"] = [0, 0, 0, 0, 0, 0]
        return

    def propulsion(self, control_signal):
        # TODO: Object oriented thruster description 
        if self.uuv_name == "rexrov2":
            K_inv = np.linalg.pinv(self.cfg["K"])
            u = np.clip(np.matmul(K_inv, control_signal), self.cfg["thruster_lower_limit"], self.cfg["thruster_upper_limit"])
            
            motor_ang_vel = np.sign(u) * np.sqrt(np.abs(u) / self.cfg["thruster_gain"])  
            ref = self.cfg["thruster_gain"] * motor_ang_vel * abs(motor_ang_vel)

            # FO Thruster
            alpha = np.exp(- self.dt / self.cfg["thruster_tau"])
            thrust = alpha * self.prev_thrust + (1 - alpha) * ref
            self.prev_thrust = thrust

            tau = np.matmul(self.cfg["K"], thrust)
        
        elif self.uuv_name == "bluerov2":
            allocated_signal = np.clip(np.matmul(self.cfg["A"], control_signal), self.cfg["thruster_lower_limit"], self.cfg["thruster_upper_limit"])
            ref = (80 / (1 + np.exp(-4 * allocated_signal**3))) - 40

            # FO Thruster
            alpha = np.exp(- self.dt / self.cfg["thruster_tau"])
            thrust = alpha * self.prev_thrust + (1 - alpha) * ref
            self.prev_thrust = thrust

            tau = np.matmul(self.cfg["K"], thrust)

        return tau

    def hydrodynamics(self, nu_r):
        damping = np.matmul(self.cfg["D_l"] + np.multiply(self.cfg["D_q"], np.abs(nu_r)) , nu_r)
        coriolis_am = np.matmul(m2c(self.cfg["M_a"], nu_r), nu_r)
        
        self.compute_nu_dot(nu_r)
        added_mass = np.matmul(self.cfg["M_a"], self.nu_dot)
        
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

        g_n = gvect(0, B, theta, phi, self.cfg["g_center"], self.cfg["b_center"])
        return g_n

    def compute_nu_dot(self, nu):
        acc = (nu - self.last_nu) / self.dt
        self.nu_dot = ((1 - self.nu_dot_filter_alpha) * self.nu_dot) + (self.nu_dot_filter_alpha * acc)
        self.last_nu = nu
    
    def callback_control_signal(self, msg):
        self.control_signal = msg.data

    def callback_state(self, msg: States):
        self.states["eta"] = enu2ned(msg.eta)
        self.states["nu"] = enu2ned(msg.nu)
    
    def main(self):
        while not rospy.is_shutdown():
            nu_ned = [self.current["N"], self.current["E"], self.current["D"]]
            nu_c = np.matmul(ned2body(self.states["eta"])[0:3, 0:3], nu_ned)
            nu_r = self.states["nu"] - np.concatenate((nu_c, np.zeros(3)), axis=0)
            
            tau = self.propulsion(self.control_signal)
            damping, coriolis_am, added_mass = self.hydrodynamics(nu_r)
            g_n = self.hydrostatics()

            total = ned2enu((tau - g_n - damping - coriolis_am - added_mass))
            
            self.wrench.force = Vector3(*total[:3])
            self.wrench.torque = Vector3(*total[3:])

            self.pub_wrench.publish(self.wrench)
            
            self.rate.sleep()


if __name__ == '__main__':
    try:
        uuv_model = UUV(uuv_name=args.uuv_name)
        uuv_model.main()
    except rospy.ROSInterruptException:
        pass
