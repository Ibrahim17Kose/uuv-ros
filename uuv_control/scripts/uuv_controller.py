#!/usr/bin/env python3

import rospy
import rospkg
import numpy as np
from yaml import safe_load

from uuv_control import *
from uuv_msgs.msg import ControlSignal, States


class UUVController:
    def __init__(self, uuv_name: str = "rexrov2"):
        self.uuv_name = uuv_name
        self.cfg = {}
        self.parse_config()

        self.controller_type = None
        self.controller = None
        
        rospy.init_node("uuv_controller", anonymous=True)
        self.rate = rospy.Rate(1000)

        rospy.Subscriber('states', States, self.callback)
        self.pub_control_signal = rospy.Publisher("control_signal", ControlSignal, queue_size=1)

        self.states = None

    def parse_config(self):
        with open(rospkg.RosPack().get_path("uuv_control") + f"/config/{self.uuv_name}.yaml", 'r') as file:
            self.cfg = safe_load(file)

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
            pass

    def callback(self, data):
        self.states["eta"] = data.eta
        self.states["nu"] = data.nu
        self.states["eta_dot"] = data.eta_dot
        self.states["nu_dot"] = data.nu_dot

if __name__ == '__main__':
    try:
        uuv_controller = UUVController()
        # rospy.spin()
    except rospy.ROSInterruptException:
        pass
