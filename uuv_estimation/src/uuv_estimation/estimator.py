import rospy
import numpy as np
from uuv_msgs.msg import ControlSignal, States
from uuv_estimation.uuv_kalman import UUVKalman
from uuv_model.utils import ned2enu, enu2ned, ned2body, body2ned

class NavigatorNode:
    def __init__(self):
        rospy.init_node('navigator_node')
        
        rospy.loginfo('Navigator node initialized')
        self.kf = UUVKalman()
        
        rospy.Subscriber("/states", States, self.states_callback)
        self.state_pub = rospy.Publisher('/estimation', States, queue_size=10)
        self.ground_truth_pub = rospy.Publisher('/ground_truth', States, queue_size=10)

        self.last_time = rospy.Time.now()

        self.counter = 0

        #self.old_eta = np.zeros(6)
    
    def states_callback(self, msg):
        # calculate dt
        dt = (rospy.Time.now() - self.last_time).to_sec()
        self.last_time = rospy.Time.now()

        # make prediction
        u = np.zeros(6)
        self.kf.makePrediction(u, dt)

        # update from state measurement z = [eta,nu]
        eta = msg.eta # [x, y, z, roll, pitch, yaw]
        nu = np.matmul(body2ned(eta), msg.nu) # [vx, vy, vz, vroll, vpitch, vyaw]

        # publish ground truth
        ground_truth = States()
        ground_truth.eta = eta
        ground_truth.nu = nu
        self.ground_truth_pub.publish(ground_truth)

        z = np.concatenate((eta, nu))

        # IMU 100 hz
        if self.counter % 10 == 0:
            self.kf.updateFromIMU(z)

        # Pressure sensor 50 hz
        if self.counter % 20 == 0:
            self.kf.updateFromPressure(z)

        # Bottom sonar 10 hz
        if self.counter % 100 == 0:
            self.kf.updateFromBottomSonar(z)

        # DVL 50 hz
        if self.counter % 50 == 0:
            self.kf.updateFromDVL(z)

        # Camera 20 hz
        if self.counter % 5000 == 0:
            self.kf.updateFromCamera(z)

        if self.counter % 10000 == 0:
            self.counter = 0
            
        self.counter += 1

        # publish the estimated state as eta and nu
        estimated_states = States()
        estimated_states.eta = self.kf.x[:6]
        estimated_states.nu = self.kf.x[6:]

        self.state_pub.publish(estimated_states)

if __name__ == '__main__':
    try:
        navigator_node = NavigatorNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
