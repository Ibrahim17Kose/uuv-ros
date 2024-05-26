#!/usr/bin/env python3

import argparse
import rospy
import rospkg
import numpy as np
from yaml import safe_load

from uuv_control.controllers import controller_list
from uuv_model.utils import *
from uuv_msgs.msg import ControlSignal, ReferenceSignal, States


parser = argparse.ArgumentParser(description="UUV Model Gazebo Script")
parser.add_argument("--uuv_name", default="rexrov2")
args = parser.parse_args(rospy.myargv()[1:])


class UUVController:
    def __init__(self, uuv_name: str = "rexrov2"):
        rospy.init_node("uuv_controller", anonymous=True)

        self.uuv_name = uuv_name

        self.cfg = {}
        self.controller_type = None
        self.controller = None
        self.parse_config()
        
        self.dt = self.cfg["dt"]
        self.rate = rospy.Rate(1 / self.dt)

        # Topics
        self.states = {"eta": np.zeros(6), "nu": np.zeros(6)}
        self.reference_signal = {"eta_ref": self.cfg["eta_0"], "nu_ref": self.cfg["nu_0"]}

        # ROS
        rospy.Subscriber('reference_signal', ReferenceSignal, self.callback_reference_signal)
        rospy.Subscriber('states', States, self.callback_states)
        self.pub_control_signal = rospy.Publisher("control_signal", ControlSignal, queue_size=1)

        rospy.loginfo("UUV CONTROLLER Node is Up ...")

    def parse_config(self):
        with open(rospkg.RosPack().get_path("uuv_model") + f"/config/{self.uuv_name}.yaml", 'r') as file:
            model_cfg = safe_load(file)

        with open(rospkg.RosPack().get_path("uuv_control") + f"/config/{self.uuv_name}.yaml", 'r') as file:
            self.cfg = safe_load(file)

        self.cfg["dt"] = model_cfg["dt"]
        self.cfg["thruster_num"] = model_cfg["thruster_num"]
        
        # Initial states
        self.cfg["eta_0"] = enu2ned([model_cfg["X_enu"], model_cfg["Y_enu"], model_cfg["Z_enu"],
                                     np.deg2rad(model_cfg["Roll_enu"]),
                                     np.deg2rad(model_cfg["Pitch_enu"]),
                                     np.deg2rad(model_cfg["Yaw_enu"])])
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
    
    def callback_reference_signal(self, msg):
        self.reference_signal["eta_ref"] = enu2ned(msg.eta_ref)
        self.reference_signal["nu_ref"] = enu2ned(msg.nu_ref)
    
    def callback_states(self, msg):
        self.states["eta"] = enu2ned(msg.eta)
        self.states["nu"] = enu2ned(msg.nu)


if __name__ == '__main__':
    try:
        uuv_controller = UUVController(uuv_name=args.uuv_name)
        uuv_controller.main()
    except rospy.ROSInterruptException:
        pass
