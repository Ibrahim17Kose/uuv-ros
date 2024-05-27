#!/usr/bin/env python3

import argparse
import rospy
import rospkg
import numpy as np
from yaml import safe_load

from uuv_model.utils import enu2ned
from uuv_msgs.msg import States
from uuv_sensor_msgs.msg import Camera, DVL, IMU, Pressure, Sonar
from geometry_msgs.msg import Quaternion, Vector3
from tf.transformations import quaternion_from_euler


parser = argparse.ArgumentParser(description="UUV Model Gazebo Script")
parser.add_argument("--uuv_name", default="rexrov2")
args = parser.parse_args(rospy.myargv()[1:])


class UUVSensors:
    def __init__(self, uuv_name: str = "rexrov2"):
        rospy.init_node('uuv_sensors', anonymous=True)
                
        self.uuv_name = uuv_name

        self.cfg = {}
        self.parse_config()
                
        self.dt = self.cfg["dt"]
        self.rate = rospy.Rate(1 / self.dt)

        self.pool_depth = 2

        # Variables
        self.nu_dot_filter_alpha = 0.3
        self.nu_dot = np.zeros(6)
        self.last_nu = np.zeros(6)

        # Topics
        self.msg_camera = Camera()
        self.msg_dvl = DVL()
        self.msg_imu = IMU()
        self.msg_pressure = Pressure()
        self.msg_sonar = Sonar()

        self.states = {"eta": np.zeros(6), "nu": np.zeros(6)}

        # ROS
        rospy.Subscriber("/states", States, self.callback_state)
        self.pub_camera = rospy.Publisher('/sensors/camera', Camera, queue_size=1)
        self.pub_dvl = rospy.Publisher('/sensors/dvl', DVL, queue_size=1)
        self.pub_imu = rospy.Publisher('/sensors/imu', IMU, queue_size=1)
        self.pub_pressure = rospy.Publisher('/sensors/pressure', Pressure, queue_size=1)
        self.pub_sonar = rospy.Publisher('/sensors/sonar', Sonar, queue_size=1)

        rospy.loginfo("GAZEBO UUV SENSORS Node is Up ...")
    
    def parse_config(self):
        with open(rospkg.RosPack().get_path("uuv_model") + f"/config/{self.uuv_name}.yaml", 'r') as file:
            model_cfg = safe_load(file)

        with open(rospkg.RosPack().get_path("uuv_sensors") + f"/config/{self.uuv_name}_sensors.yaml", 'r') as file:
            self.cfg = safe_load(file)
        
        self.cfg["dt"] = model_cfg["dt"]

    def compute_nu_dot(self, nu):
        acc = (nu - self.last_nu) / self.dt
        self.nu_dot = ((1 - self.nu_dot_filter_alpha) * self.nu_dot) + (self.nu_dot_filter_alpha * acc)
        self.last_nu = nu
    
    def callback_state(self, msg: States):
        self.states["eta"] = enu2ned(msg.eta)
        self.states["nu"] = enu2ned(msg.nu)
    
    def generate_camera_msg(self):
        noise = np.random.normal(self.cfg["camera"][0]["mean"], self.cfg["camera"][1]["std_dev"], 2)

        self.msg_camera.x = self.states["eta"][0] + noise[0]
        self.msg_camera.y = self.states["eta"][1] + noise[1]

    def generate_dvl_msg(self):
        noise = np.random.normal(self.cfg["dvl"][0]["mean"], self.cfg["dvl"][1]["std_dev"], 6)

        self.msg_dvl.position = Vector3(*(self.states["eta"][0:3] + noise[0:3]))
        self.msg_dvl.velocity = Vector3(*(self.states["nu"][0:3] + noise[3:6]))

        self.msg_dvl.velocity_covariance = np.zeros(9)

    def generate_imu_msg(self):
        noise = np.random.normal(self.cfg["imu"][0]["mean"], self.cfg["imu"][1]["std_dev"], 9)
        
        q = quaternion_from_euler(self.states["eta"][3] + noise[0],
                                  self.states["eta"][4] + noise[1],
                                  self.states["eta"][5] + noise[2])
        
        self.msg_imu.orientation = Quaternion(*q)
        self.msg_imu.angular_velocity = Vector3(*(self.states["nu"][3:6] + noise[3:6]))
        self.msg_imu.linear_acceleration = Vector3(*(self.nu_dot[0:3] + noise[6:9]))
        
        self.msg_imu.orientation_covariance = np.zeros(9)
        self.msg_imu.angular_velocity_covariance = np.zeros(9)
        self.msg_imu.linear_acceleration_covariance = np.zeros(9)

    def generate_pressure_msg(self):
        noise = np.random.normal(self.cfg["pressure"][0]["mean"], self.cfg["pressure"][1]["std_dev"], 1)

        self.msg_pressure.depth = - self.states["eta"][2] + noise[0]

    def generate_sonar_msg(self):
        noise = np.random.normal(self.cfg["sonar"][0]["mean"], self.cfg["sonar"][1]["std_dev"], 1)
        
        self.msg_sonar.distance = np.clip((self.pool_depth - self.states["eta"][2]) + noise[0], 0, np.inf)

    def main(self):
        while not rospy.is_shutdown():
            self.compute_nu_dot(self.states["nu"])

            self.generate_camera_msg()
            self.generate_dvl_msg()
            self.generate_imu_msg()
            self.generate_pressure_msg()
            self.generate_sonar_msg()

            self.pub_camera.publish(self.msg_camera)
            self.pub_dvl.publish(self.msg_dvl)
            self.pub_imu.publish(self.msg_imu)
            self.pub_pressure.publish(self.msg_pressure)
            self.pub_sonar.publish(self.msg_sonar)

            self.rate.sleep()


if __name__ == '__main__':
    try:
        uuv_model = UUVSensors(uuv_name=args.uuv_name)
        uuv_model.main()
    except rospy.ROSInterruptException:
        pass
