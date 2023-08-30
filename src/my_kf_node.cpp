#include <Eigen/Dense>
#include <std_msgs/Float64.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/PointStamped.h>
#include <geometry_msgs/Pose2D.h>
#include <iostream>
#include <ros/ros.h>


using float32_t = float;

template <size_t ROW, size_t COL>
using Matrix = Eigen::Matrix<float32_t, ROW, COL>;

template<size_t ROW>
using Vector = Eigen::Matrix<float32_t, ROW, 1>;

const float32_t v_i = 0.1;
const float32_t w_i = 0.1;
const float32_t v_j = 0.2;
const float32_t w_j = 0.1;

class EKF
{
private:
    // Prediction
    Vector<3> m_vecX;
    Matrix<3, 3> m_matP{Matrix<3, 3>::Zero()};
    Matrix<3, 3> m_matQ{Matrix<3, 3>::Zero()};
    Matrix<3, 3> m_jacobian_matF{Matrix<3, 3>::Zero()};

    // Correction
    Vector<3> m_vecZ; // measurement
    Vector<3> m_vech; // measurement data
    Matrix<3, 3> m_matR{    Matrix<3, 3>::Zero()};
    Matrix<3, 3> m_jacobian_matH{Matrix<3, 3>::Zero()};

public:
    EKF()
    {
        m_vecX[0] = 1.0;
        m_vecX[1] = 1.0;
        m_vecX[2] = 0.0;

        m_matP(0,0) = 0.0;
        m_matP(1,1) = 0.0;
        m_matP(2,2) = 0.0;

        m_matQ(0,0) = 0.003;
        m_matQ(1,1) = 0.005;
        m_matQ(2,2) = 0.005;

        m_matR(0,0) = 0.1;
        m_matR(1,1) = 0.1;
        m_matR(2,2) = 0.1;
    }

    Vector<3>& getVecX()
    {
        return m_vecX;
    }

    Matrix<3, 3>& getMatP()
    {
        return m_matP;
    }
    Vector<3>& getvecZ()
    {
        return m_vecZ;
    }

//    void test(Vector<3>& vec)
//    {
//        m_vecX[0] = vec[0];
//        m_vecX[1] = vec[1];
//        m_vecX[2] = vec[2];
//    }

    void motionModelJacobian(Vector<3>& vec, float32_t delta_t)
    {
        m_jacobian_matF << 0, w_i, -v_j * sin(vec[2]),
                          -w_i, 0, v_j * cos(vec[2]),
                           0, 0, 0;
        m_jacobian_matF = Matrix<3, 3>::Identity() + delta_t * m_jacobian_matF;
    }

    void motionModel(Vector<3>& vec, float32_t delta_t)
    {
        Vector<3> tmp_vec;

        tmp_vec[0] = v_j*cos(vec[2]) + w_i*vec[1] - v_i;
        tmp_vec[1] = v_j*sin(vec[2]) - w_i*vec[0];
        tmp_vec[2] = w_j - w_i;

        m_vecX = m_vecX + delta_t * tmp_vec;
    }

    void prediction(float32_t delta_t)
    {
        motionModelJacobian(m_vecX, delta_t);
        motionModel(m_vecX, delta_t);
        m_matP = m_jacobian_matF * m_matP * m_jacobian_matF.transpose() + m_matQ;
    }

    void measurementModel(Vector<3>& vec)
    {
        m_vech[0] = sqrt(vec[0] * vec[0] + vec[1] * vec[1]);
        m_vech[1] = atan2(vec[1], vec[0]);
        if (m_vech[1] < 0)
            m_vech[1] += 2 * M_PI;
//        std::cout << m_vech[1] << std::endl;
        m_vech[2] = vec[2];
    }

    void measurementModelJacobian(Vector<3>& vec)
    {
        m_jacobian_matH << vec[0] / (sqrt(vec[0] * vec[0] + vec[1] * vec[1])), vec[1] / (sqrt(vec[0] * vec[0] + vec[1] * vec[1])), 0,
                          -vec[1] / (vec[0] * vec[0] + vec[1] * vec[1]), vec[0] / (vec[0] * vec[0] + vec[1] * vec[1]), 0,
                              0, 0, 1;
    }

    void correction()
    {
        measurementModel(m_vecX);
        measurementModelJacobian(m_vecX);

        // residual(innovation)
        Vector<3> residual;
        residual[0] = m_vecZ[0] - m_vech[0];
//        std::cout << m_vech[0] << std::endl;
        residual[1] = m_vecZ[1] - m_vech[1];
        residual[2] = m_vecZ[2] - m_vech[2];

        // residual(innovation) covariance
        Matrix<3, 3> residual_cov;
        residual_cov = m_jacobian_matH * m_matP * m_jacobian_matH.transpose() + m_matR;

        // Kalman Gain
        Matrix<3, 3> Kk = m_matP * m_jacobian_matH.transpose() * residual_cov.inverse();

        // update
        m_vecX = m_vecX + Kk * residual;
        m_matP = (Matrix<3,3>::Identity() - Kk * m_jacobian_matH) * m_matP;
    }

    void print()
    {
        std::cout << m_vecX[0] <<", " <<  m_vecX[1] << ", "<< m_vecX[2] * 180 / M_PI << std::endl;
//        std::cout << "The state cov is " << std::endl << m_matP << std::endl;
//        std::cout << sqrt(m_vecX[0]*m_vecX[0] + m_vecX[1]*m_vecX[1]) << ", " << atan2(m_vecX[1], m_vecX[0]) * 180 / M_PI << ", " << m_vecX[2] * 180 / M_PI << std::endl;
    }

    void print2()
    {
        std::cout << "The z value is " << std::endl << m_vecZ << std::endl;
    }
};


class EKFNode
{
public:
    EKFNode()
    {
        pub_ = n_.advertise<geometry_msgs::PointStamped>("Estimated", 1);
        sub_ = n_.subscribe("/noisy_measurements", 1, &EKFNode::callback, this);
    }

    void callback(const geometry_msgs::PointStamped::ConstPtr& msg)
    {
        static ros::Time previousTime;

        ros::Time currentTime = msg->header.stamp;
        if (previousTime.isZero())
        {
            previousTime = currentTime;
            return;
        }

        ros::Duration duration = currentTime - previousTime;
        float32_t delta_t = duration.toSec();
        if (delta_t != 0)
        {
            ekf.getvecZ() << msg->point.x, msg->point.y * M_PI / 180, msg->point.z * M_PI / 180;
//            ekf.print2();
            ekf.prediction(delta_t);
            ekf.correction();
//            ekf.print();

//            geometry_msgs::PointStamped msg;
//            msg.header.stamp = currentTime;
//            msg.point.x = ekf.getVecX()[0];
//            msg.point.y = ekf.getVecX()[1];
//            msg.point.z = ekf.getVecX()[2];

           geometry_msgs::PointStamped msg2;
           msg2.header.stamp = currentTime;
           float32_t tmp_x = ekf.getVecX()[0];
           float32_t tmp_y = ekf.getVecX()[1];
           float32_t tmp_z = ekf.getVecX()[2];
           msg2.point.x = sqrt(tmp_x * tmp_x + tmp_y * tmp_y);
           msg2.point.y = atan2(tmp_y, tmp_x) * 180 / M_PI;
           if (msg2.point.y < 0)
               msg2.point.y += 360;
           msg2.point.z = tmp_z * 180 / M_PI;
           pub_.publish(msg2);

           previousTime = currentTime;
        }
    }
private:
    ros::NodeHandle n_;
    ros::Publisher pub_;
    ros::Subscriber sub_;


    EKF ekf;
};

int main(int argc, char** argv)
{
    ros::init(argc, argv, "EkfNode");
    ROS_INFO("Init!");
    EKFNode object;
    ros::spin();

    return 0;

}
