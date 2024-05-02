#!/usr/bin/env python3

import rospy
import rospkg
import numpy as np
from yaml import safe_load

from uuv_control import *
from uuv_msgs.msg import ControlSignal, ReferenceSignal, States


class UUVController:
    def __init__(self, uuv_name: str = "rexrov2"):
        self.uuv_name = uuv_name
        self.cfg = {}
        self.controller_type = None
        self.controller = None
        self.parse_config()
        
        rospy.init_node("uuv_controller", anonymous=True)
        self.rate = rospy.Rate(1000)

        rospy.Subscriber('reference_signal', ReferenceSignal, self.callback_reference_signal)
        rospy.Subscriber('states', States, self.callback_states)
        self.pub_control_signal = rospy.Publisher("control_signal", ControlSignal, queue_size=1)

        self.states = {"eta": self.cfg["eta_0"], "nu": self.cfg["nu_0"], "eta_dot": np.zeros(6), "nu_dot": np.zeros(6)}
        self.reference_signal = {"eta_ref": self.cfg["eta_0"], "nu_ref": self.cfg["nu_0"]}

    def parse_config(self):
        with open(rospkg.RosPack().get_path("uuv_model") + f"/config/{self.uuv_name}.yaml", 'r') as file:
            model_cfg = safe_load(file)

        with open(rospkg.RosPack().get_path("uuv_control") + f"/config/{self.uuv_name}.yaml", 'r') as file:
            self.cfg = safe_load(file)
        
        # Vehicle Coordinates [NED] Conversion
        model_cfg["X_ned"] = model_cfg["Y_enu"]
        model_cfg["Y_ned"] = model_cfg["X_enu"]
        model_cfg["Z_ned"] = - model_cfg["Z_enu"]
        model_cfg["Roll_ned"] = model_cfg["Roll_enu"]
        model_cfg["Pitch_ned"] = - model_cfg["Pitch_enu"]
        model_cfg["Yaw_ned"] = - model_cfg["Yaw_enu"] + 90

        # Initial states
        self.cfg["eta_0"] = [model_cfg["X_ned"], model_cfg["Y_ned"], model_cfg["Z_ned"],
                             np.deg2rad(model_cfg["Roll_ned"]),
                             np.deg2rad(model_cfg["Pitch_ned"]),
                             np.deg2rad(model_cfg["Yaw_ned"])]
        self.cfg["nu_0"] = [0, 0, 0, 0, 0, 0]

        if not "controller_type" in self.cfg:
            raise("controller_type is not specified in config file.")
        else:
            self.controller_type = self.cfg["controller_type"]

            if not self.controller_type in controller_list:
                raise("The controller_type specified in the config file is unknown.")
            else:
                self.controller = controller_list[self.controller_type](self.cfg)
            
    def main(self):
        while not rospy.is_shutdown():
            control_signal = self.controller.update(self.reference_signal, self.states)

            msg = ControlSignal()
            msg.data = control_signal
            self.pub_control_signal.publish(msg)
            self.rate.sleep()
    
    def callback_reference_signal(self, data):
        self.reference_signal["eta_ref"] = data.eta_ref
        self.reference_signal["nu_ref"] = data.nu_ref
    
    def callback_states(self, data):
        self.states["eta"] = data.eta
        self.states["nu"] = data.nu
        self.states["eta_dot"] = data.eta_dot
        self.states["nu_dot"] = data.nu_dot

if __name__ == '__main__':
    try:
        uuv_controller = UUVController()
        uuv_controller.main()
    except rospy.ROSInterruptException:
        pass
