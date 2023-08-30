#include "kalman_filter.h"
#include <cmath>

static constexpr size_t DIM_X{ 3 };
static constexpr size_t DIM_Z{ 2 };

static float v_i = 0.1;
static float w_i = 0.1;
static float v_j = 0.2;
static float w_j = 0.1;

static kf::KalmanFilter<DIM_X, DIM_Z> kalmanfilter;

kf::Vector<DIM_X> predictionModelFunc(const kf::Vector<DIM_X>& vecX)
{
    const kf::Vector<DIM_X> predicted_motion_value{
        v_j * cos(vecX[2]) + vecX[1] * w_i - v_i,
        v_j * sin(vecX[2]) - vecX[0] * w_i,
        w_j - w_i};

    return predicted_motion_value;
}

kf::Matrix<DIM_X, DIM_X> calculateFJacobianMatrix(const kf::Vector<DIM_X>& vecX)
{
    const kf::float32_t valX2PlusY2{ (vecX[0]*vecX[0]) + (vecX[1]*vecX[1]) };
    const kf::float32_t valSqrtX2PlusY2{ std::sqrt(valX2PlusY2) };

    kf::Matrix<DIM_X,DIM_X> matFj;
    matFj << 
        0, w_i, -v_j * sin(vecX[2]),
        -w_i, 0, v_j * cos(vecX[2]),
        0, 0, 0;
    return matFj;
}

int main(int argc, char** argv)
{

    return 0;
}
