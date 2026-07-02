

#include <ros/ros.h>
#include <nav_msgs/Odometry.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/TwistStamped.h>
#include <geometry_msgs/Quaternion.h>
#include <geometry_msgs/Point.h>
#include <sensor_msgs/Imu.h>
#include <tf/transform_datatypes.h>
#include <string>

ros::Publisher pub_msg2uav;
ros::Publisher pub_odom;
ros::Publisher pub_imu;
geometry_msgs::Quaternion q;
geometry_msgs::Point pos;
double vision_z_offset = -0.15;

// ros::Rate loop(15);
void pose_callback(const geometry_msgs::PoseStamped::ConstPtr& msg){

    geometry_msgs::PoseStamped tomsg;
    tomsg = *msg;
    tomsg.header.frame_id = "map";
    tomsg.pose.position.x = msg->pose.position.x;
    tomsg.pose.position.y = msg->pose.position.y;
    tomsg.pose.position.z = msg->pose.position.z + vision_z_offset;
    q.w = msg->pose.orientation.w;
    q.x = msg->pose.orientation.x;
    q.y = msg->pose.orientation.y;
    q.z = msg->pose.orientation.z;
    pub_msg2uav.publish(tomsg);
    pos.x = msg->pose.position.x;
    pos.y = msg->pose.position.y;
    pos.z = msg->pose.position.z;
}

void vel_callback(const geometry_msgs::TwistStamped::ConstPtr& msg){
    nav_msgs::Odometry odom_msg;
    odom_msg.header = msg->header;
    odom_msg.pose.pose.position = pos;
    tf::Vector3 vel_world(msg->twist.linear.x, msg->twist.linear.y, msg->twist.linear.z);
    tf::Quaternion quat(q.x, q.y, q.z, q.w);
    tf::Matrix3x3 rotation_matrix(quat);
    tf::Vector3 vel_body = rotation_matrix.inverse() * vel_world;
    odom_msg.twist.twist.linear.x = vel_body.x();
    odom_msg.twist.twist.linear.y = vel_body.y();
    odom_msg.twist.twist.linear.z = vel_body.z();
    odom_msg.pose.pose.orientation.w = q.w;
    odom_msg.pose.pose.orientation.x = q.x;
    odom_msg.pose.pose.orientation.y = q.y;
    odom_msg.pose.pose.orientation.z = q.z;
    pub_odom.publish(odom_msg);
}

void acc_callback(const geometry_msgs::TwistStamped::ConstPtr& msg){

    tf::Vector3 accel_world(msg->twist.linear.x, msg->twist.linear.y, msg->twist.linear.z + 9.81);

    tf::Quaternion quat(q.x, q.y, q.z, q.w);
    tf::Matrix3x3 rotation_matrix(quat);
    tf::Vector3 accel_body = rotation_matrix.inverse() * accel_world;

    sensor_msgs::Imu imu_msg;
    imu_msg.header = msg->header;
    imu_msg.linear_acceleration.x = accel_body.x();
    imu_msg.linear_acceleration.y = accel_body.y();
    imu_msg.linear_acceleration.z = accel_body.z();
    imu_msg.orientation.w = q.w;
    imu_msg.orientation.x = q.x;
    imu_msg.orientation.y = q.y;
    imu_msg.orientation.z = q.z;
    pub_imu.publish(imu_msg);

}

int main(int argc, char** argv){
    ros::init(argc, argv, "nokov_node");
    ros::NodeHandle nh;
    ros::NodeHandle pnh("~");
    std::string tracker_name;
    pnh.param<std::string>("tracker_name", tracker_name, "soccer");
    pnh.param<double>("vision_z_offset", vision_z_offset, -0.15);

    const std::string tracker_ns = "/vrpn_client_node/" + tracker_name;
    pub_msg2uav = nh.advertise<geometry_msgs::PoseStamped>("mavros/vision_pose/pose", 1);
    pub_imu = nh.advertise<sensor_msgs::Imu>("nokov/imu/data", 1);
    pub_odom = nh.advertise<nav_msgs::Odometry>("nokov/local_position/odom", 1);
    ROS_INFO_STREAM("nokov_node using VRPN tracker namespace " << tracker_ns);
    ros::Subscriber sub_pose = nh.subscribe(tracker_ns + "/pose", 1, &pose_callback);
    ros::Subscriber sub_vel = nh.subscribe(tracker_ns + "/twist", 1, &vel_callback);
    ros::Subscriber sub_acc = nh.subscribe(tracker_ns + "/accel", 1, &acc_callback);
    ros::spin();
}
