#!/usr/bin/env python3

import rospy
import rospkg
import numpy as np
from yaml import safe_load

from scipy.interpolate import PchipInterpolator
from uuv_msgs.msg import ReferenceSignal
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped


class UUVGuidance:
    def __init__(self):
        rospy.init_node('uuv_guidance', anonymous=True)

        self.cfg = {}
        self.parse_config()

        self.dt = self.cfg["dt"]
        self.rate = rospy.Rate(1 / self.dt)

        theta = np.arange(0, len(self.cfg["X_point"]))

        self.x_interpolator = PchipInterpolator(theta, self.cfg["X_point"])
        self.y_interpolator = PchipInterpolator(theta, self.cfg["Y_point"])
        self.z_interpolator = PchipInterpolator(theta, self.cfg["Z_point"])

        self.x_dot_interpolator = self.x_interpolator.derivative()
        self.y_dot_interpolator = self.y_interpolator.derivative()
        self.z_dot_interpolator = self.z_interpolator.derivative()

        self.time_constant = 1
        self.initial_speed = 0

        # Trajectory
        self.theta = [0]
        self.yaw_trajectory = None

        self.x_trajectory = None
        self.y_trajectory = None
        self.z_trajectory = None

        # ROS
        self.pub_path = rospy.Publisher("path", Path, queue_size=1)
        self.pub_reference_signal = rospy.Publisher("reference_signal", ReferenceSignal, queue_size=1)

    def parse_config(self):
        with open(rospkg.RosPack().get_path("uuv_navigation") + f"/config/semi_olympic_waypoints.yaml", 'r') as file:
            self.cfg = safe_load(file)
        
        self.cfg["X_point"] = np.array(self.cfg["waypoints"])[:, 0]
        self.cfg["Y_point"] = np.array(self.cfg["waypoints"])[:, 1]
        self.cfg["Z_point"] = np.array(self.cfg["waypoints"])[:, 2]
    
    def main(self):
        self.generate_path()
        self.generate_trajectory()
        rospy.loginfo("Publishing Trajectory ...")
        msg = ReferenceSignal()
        for i in range(len(self.x_trajectory)):
            msg.eta_ref = [
                self.x_trajectory[i],
                self.y_trajectory[i],
                self.z_trajectory[i],
                0,
                0,
                self.yaw_trajectory[i]]
            self.pub_reference_signal.publish(msg)

            self.rate.sleep()

    def generate_path(self, resolution=1000):
        th = np.linspace(0, len(self.cfg["waypoints"]), resolution)
        x_path = self.x_interpolator(th)
        y_path = self.y_interpolator(th)
        z_path = self.z_interpolator(th)

        msg = Path()
        for i in range(resolution):
            pose = PoseStamped()
            pose.pose.position.x = x_path[i]
            pose.pose.position.y = y_path[i]
            pose.pose.position.z = z_path[i]
            msg.poses.append(pose)
        
        self.pub_path.publish(msg)
        rospy.loginfo("Path generation finished.")

    def generate_trajectory(self):
        i = 0
        h = self.dt
        th = 0

        while th < 11:
            x_th = self.x_dot_interpolator(th)
            y_th = self.y_dot_interpolator(th)
            z_th = self.z_dot_interpolator(th)

            th += h * (self.initial_speed / np.sqrt(x_th**2 + y_th**2 + z_th**2))
            self.initial_speed += h * (-self.initial_speed + self.cfg["desired_speed"]) / self.time_constant

            i += 1
            self.theta.append(th)
        
        self.x_trajectory = self.x_interpolator(self.theta)
        self.y_trajectory = self.y_interpolator(self.theta)
        self.z_trajectory = self.z_interpolator(self.theta)

        self.yaw_trajectory = np.zeros_like(self.x_trajectory)

        x_fark = self.x_trajectory[1:] - self.x_trajectory[:-1]
        y_fark = self.y_trajectory[1:] - self.y_trajectory[:-1]

        self.yaw_trajectory[1:] = np.arctan2(y_fark, x_fark)
        rospy.loginfo("Trajectory generation finished.")


if __name__ == '__main__':
    try:
        uuv_guidance = UUVGuidance()
        uuv_guidance.main()
    except rospy.ROSInterruptException:
        pass