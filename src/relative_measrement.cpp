#include <ros/ros.h>
#include <math.h>
#include <vector>
#include <random>
#include <std_msgs/Float64.h>
#include <geometry_msgs/PoseStamped.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/convert.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <tf2/utils.h>


class NoisyMeasurements
{
private:
    ros::NodeHandle n_;

    ros::Publisher pub1_;
    ros::Publisher pub2_;


    ros::Subscriber sub1_;
    ros::Subscriber sub2_;

    geometry_msgs::PoseStamped pose1;
    geometry_msgs::PoseStamped pose2;

    double mean = 0.0;
    double stdDev = 0.05;

public:
    NoisyMeasurements()
    {
        // 퍼블리쉬할 토픽 선언

        pub1_ = n_.advertise<geometry_msgs::PointStamped>("/noisy_measurements", 1);
        pub2_ = n_.advertise<geometry_msgs::PointStamped>("/true_measurements", 1);

        // 섭스크라이브할 토픽 선언
        sub1_ = n_.subscribe("/Robot_0/ground_truth", 1, &NoisyMeasurements::callback1, this);
        sub2_ = n_.subscribe("/Robot_1/ground_truth", 1, &NoisyMeasurements::callback2, this);
    }

    void callback1(const geometry_msgs::PoseStamped::ConstPtr& msg)
    {
        pose1 = *msg;
        calculateMeasurementAndPublish();
        trueMeasurementAndPublish();
    }

    void callback2(const geometry_msgs::PoseStamped::ConstPtr& msg)
    {
        pose2 = *msg;
        calculateMeasurementAndPublish();
        trueMeasurementAndPublish();
    }


    void calculateMeasurementAndPublish()
    {
        if (!pose1.header.stamp.isZero() && !pose2.header.stamp.isZero())
        {
            double distance =  sqrt((pose1.pose.position.x - pose2.pose.position.x) * (pose1.pose.position.x - pose2.pose.position.x)
                            +(pose1.pose.position.y - pose2.pose.position.y) * (pose1.pose.position.y - pose2.pose.position.y));
            addGaussianNoise(distance, mean, stdDev);


            tf2::Quaternion quat1, quat2;
            tf2::fromMsg(pose1.pose.orientation, quat1);
            tf2::fromMsg(pose2.pose.orientation, quat2);
            double yaw1 = tf2::getYaw(quat1);
            if (yaw1 < 0)
                yaw1 += 2 * M_PI;
            double yaw2 = tf2::getYaw(quat2);
            if (yaw2 < 0)
                yaw2 += 2 * M_PI;

            double yaw_diff = yaw1 - yaw2;
            if (yaw_diff < 0)
                yaw_diff += 2 * M_PI;
            yaw_diff *= 180 / M_PI;
            addGaussianNoise(yaw_diff, mean, stdDev);


            yaw1 *= 180 / M_PI;
            double x_diff = pose2.pose.position.x - pose1.pose.position.x;
            double y_diff = pose2.pose.position.y - pose1.pose.position.y;
            double bearing = atan2(y_diff , x_diff);
            if (bearing < 0)
                bearing += 2 *M_PI;
            bearing = bearing * (180 / M_PI) - yaw1;
            if (bearing < 0)
                bearing += 360;

            addGaussianNoise(bearing, mean, stdDev);

            geometry_msgs::PointStamped measurements;
            measurements.header = pose1.header;
            measurements.point.x = distance;
            measurements.point.y = bearing;
            measurements.point.z = yaw_diff;

            pub1_.publish(measurements);
        }
    }

    void trueMeasurementAndPublish()
    {
        if (!pose1.header.stamp.isZero() && !pose2.header.stamp.isZero())
        {
            double distance =  sqrt((pose1.pose.position.x - pose2.pose.position.x) * (pose1.pose.position.x - pose2.pose.position.x)
                            +(pose1.pose.position.y - pose2.pose.position.y) * (pose1.pose.position.y - pose2.pose.position.y));
//            addGaussianNoise(distance, mean, stdDev);



            tf2::Quaternion quat1, quat2;
            tf2::fromMsg(pose1.pose.orientation, quat1);
            tf2::fromMsg(pose2.pose.orientation, quat2);
            double yaw1 = tf2::getYaw(quat1);
            if (yaw1 < 0)
                yaw1 += 2 * M_PI;
            double yaw2 = tf2::getYaw(quat2);
            if (yaw2 < 0)
                yaw2 += 2 * M_PI;

            double yaw_diff = yaw1 - yaw2;
            if (yaw_diff < 0)
                yaw_diff += 2 * M_PI;
            yaw_diff *= 180 / M_PI;
//            addGaussianNoise(yaw_diff, mean, stdDev);
//            std::cout << yaw1<< std::endl;
//            std::cout << yaw2<< std::endl;
//            std::cout << yaw_diff<< std::endl;

            yaw1 *= 180 / M_PI;
            double x_diff = pose2.pose.position.x - pose1.pose.position.x;
            double y_diff = pose2.pose.position.y - pose1.pose.position.y;
            double bearing = atan2(y_diff , x_diff);
            if (bearing < 0)
                bearing += 2 *M_PI;
            bearing = bearing * (180 / M_PI) - yaw1;
            if (bearing < 0)
                bearing += 360;
//            std::cout << x_diff << y_diff << std::endl;
//            std::cout << bearing << std::endl;

//            bearing *= 180 / M_PI;
//            std::cout << bearing << std::endl;
//            addGaussianNoise(bearing, mean, stdDev);

            geometry_msgs::PointStamped measurements;
            measurements.header = pose1.header;
            measurements.point.x = distance;
            measurements.point.y = bearing;
            measurements.point.z = yaw_diff;

            pub2_.publish(measurements);
        }
    }

    void addGaussianNoise(double& value, double mean, double stdDev)
    {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<double> distribution(mean, stdDev);

        value += distribution(gen);
    }


};

int main(int argc, char** argv)
{
    ros::init(argc, argv, "relative_robot_measurements");
    NoisyMeasurements node;
    ROS_INFO("Init!");
    ros::spin();
    return 0;
}
