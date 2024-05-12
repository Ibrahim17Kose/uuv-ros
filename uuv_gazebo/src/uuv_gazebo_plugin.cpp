#include <iostream>
#include <thread>
#include <mutex>
#include <gazebo/gazebo.hh>
#include <gazebo/physics/physics.hh>
#include <gazebo/common/common.hh>
#include <ignition/math/Vector3.hh>

#include <ros/ros.h>
#include <geometry_msgs/Wrench.h>
#include <uuv_msgs/States.h>
#include <tf/transform_broadcaster.h>


class UUVGazeboPlugin : public gazebo::ModelPlugin
{
public:
    UUVGazeboPlugin() : gazebo::ModelPlugin()
    {
        ROS_INFO("Starting UUV Gazebo Plugin");
    }

public:
    void Load(gazebo::physics::ModelPtr parent, sdf::ElementPtr sdf)
    {
        this->model = parent;
        this->link = model->GetLink("base");
        _rate = 1000.0;

        if (!ros::isInitialized())
        {
            int argc = 0;
            char **argv = NULL;
            ros::init(argc, argv, "uuv_gazebo_plugin_node",
                      ros::init_options::NoSigintHandler);
        }

        nh = new ros::NodeHandle("");
        state_pub = nh->advertise<uuv_msgs::States>("states", 1);
        wrench_sub = nh->subscribe("apply_wrench", 1, &UUVGazeboPlugin::ApplyWrenchCallback, this);
        ros_thread = std::thread(std::bind(&UUVGazeboPlugin::rosThread, this));

        update_connection = gazebo::event::Events::ConnectWorldUpdateBegin(std::bind(&UUVGazeboPlugin::onUpdate, this));
    }

    void onUpdate()
    {
        state_mtx.lock();
        _pose = this->link->WorldPose();
        _linVel = this->link->RelativeLinearVel();
        _angVel = this->link->RelativeAngularVel();
        state_mtx.unlock();
    }

    void rosThread()
    {
        ros::Rate rate(_rate);
        while (ros::ok())
        {
            ros::spinOnce();
            publishPose();
            rate.sleep();
        }
    }

    void publishPose()
    {
        state_mtx.lock();
        ignition::math::Pose3d pose = _pose;
        ignition::math::Vector3d linVel = _linVel;
        ignition::math::Vector3d angVel = _angVel;
        state_mtx.unlock();

        // ignition::math::Vector3 rpy = pose.rot.GetAsEuler();
        tf::Quaternion q(pose.Rot().X(), pose.Rot().Y(), pose.Rot().Z(), pose.Rot().W());
        tf::Matrix3x3 m(q);
        double roll, pitch, yaw;
        m.getRPY(roll, pitch, yaw);

        uuv_msgs::States states_msg;
        states_msg.eta[0] = pose.Pos().X();
        states_msg.eta[1] = pose.Pos().Y();
        states_msg.eta[2] = pose.Pos().Z();

        states_msg.eta[3] = roll;
        states_msg.eta[4] = pitch;
        states_msg.eta[5] = yaw;

        states_msg.nu[0] = linVel.X();
        states_msg.nu[1] = linVel.Y();
        states_msg.nu[2] = linVel.Z();

        states_msg.nu[3] = angVel.X();
        states_msg.nu[4] = angVel.Y();
        states_msg.nu[5] = angVel.Z();

        state_pub.publish(states_msg);
    }


    void ApplyWrenchCallback(const geometry_msgs::Wrench::ConstPtr& msg)
    {
        link->AddRelativeForce(ignition::math::Vector3d(msg->force.x, msg->force.y, msg->force.z));
        link->AddRelativeTorque(ignition::math::Vector3d(msg->torque.x, msg->torque.y, msg->torque.z));
    }

private:
    double _rate;
    ros::NodeHandle *nh;
    ros::Publisher state_pub;
    ros::Subscriber wrench_sub;
    std::thread ros_thread;
    std::mutex state_mtx;
    gazebo::physics::ModelPtr model;
    gazebo::physics::LinkPtr link;

    gazebo::event::ConnectionPtr update_connection;
    ignition::math::Pose3d _pose;
    ignition::math::Vector3d _linVel, _angVel;
};

GZ_REGISTER_MODEL_PLUGIN(UUVGazeboPlugin)