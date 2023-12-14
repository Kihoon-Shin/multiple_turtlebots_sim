/**
 *  @brief Robot 2D trajectory optimization
 *
 *  @author Atsushi Sakai
 **/
#include <vector>
#include <cstdlib>
#include <ctime>
#include <chrono>
#include "ceres/ceres.h"
#include "glog/logging.h"
#include "math.h"
#include "csvparser.h"
#include "matplotlibcpp.h"

using namespace std;

namespace plt = matplotlibcpp;

using ceres::AutoDiffCostFunction;
using ceres::CostFunction;
using ceres::CauchyLoss;
using ceres::HuberLoss;
using ceres::TukeyLoss;
using ceres::ArctanLoss;
using ceres::Problem;
using ceres::Solver;
using ceres::Solve;


struct MyConstraintCase0{
    MyConstraintCase0(double rho, double beta, double theta)
        :rho_(rho), beta_(beta), theta_(theta) {}

    template<typename T>
    bool operator()(const T* const x_ji, const T* const y_ji, const T* const theta_ji, T* residual) const
    {
        residual[0] = (sqrt(x_ji[0]*x_ji[0] + y_ji[0]*y_ji[0]) - T(rho_)) ;
        T tmp = atan2(y_ji[0], x_ji[0]);
        if (tmp < 0)
            tmp += 2 * M_PI;
        residual[1] = (tmp - T(beta_ * M_PI / 180)) ;
        residual[2] = (theta_ji[0] - T(theta_ * M_PI / 180));
        return true;
    }

    static ceres::CostFunction* Create(
            const double rho,
            const double beta,
            const double theta)
    {
        return (new ceres::AutoDiffCostFunction<MyConstraintCase0,3,1,1,1>(
                    new MyConstraintCase0(rho, beta, theta)));
    }

private:
    const double rho_;
    const double beta_;
    const double theta_;
};

struct MyConstraintCase1{
    MyConstraintCase1(double rho)
        :rho_(rho) {}

    template<typename T>
    bool operator()(const T* const x_ji, const T* const y_ji, T* residual) const
    {
        residual[0] = (sqrt(x_ji[0]*x_ji[0] + y_ji[0]*y_ji[0]) - T(rho_));

        return true;
    }

    static ceres::CostFunction* Create(
            const double rho)
    {
        return (new ceres::AutoDiffCostFunction<MyConstraintCase1,1,1,1>(
                    new MyConstraintCase1(rho)));
    }

private:
    const double rho_;
//    const double beta_;
//    const double theta_;
};

struct MyConstraintCase2{
    MyConstraintCase2(double beta)
        : beta_(beta) {}

    template<typename T>
    bool operator()(const T* const x_ji, const T* const y_ji, T* residual) const
    {
        T tmp = atan2(y_ji[0], x_ji[0]);
        if (tmp < 0)
            tmp += 2 * M_PI;
        residual[0] = (tmp - T(beta_ * M_PI / 180)) ;
        return true;
    }

    static ceres::CostFunction* Create(
            const double beta)
    {
        return (new ceres::AutoDiffCostFunction<MyConstraintCase2,1,1,1>(
                    new MyConstraintCase2(beta)));
    }

private:
    const double beta_;
};

struct MyConstraintCase3{
    MyConstraintCase3(double rho, double beta)
        :rho_(rho), beta_(beta) {}

    template<typename T>
    bool operator()(const T* const x_ji, const T* const y_ji, T* residual) const
    {
        residual[0] = (sqrt(x_ji[0]*x_ji[0] + y_ji[0]*y_ji[0]) - T(rho_)) ;
        T tmp = atan2(y_ji[0], x_ji[0]);
        if (tmp < 0)
            tmp += 2 * M_PI;
        residual[1] = (tmp - T(beta_ * M_PI / 180)) ;
        return true;
    }

    static ceres::CostFunction* Create(
            const double rho,
            const double beta)
    {
        return (new ceres::AutoDiffCostFunction<MyConstraintCase3,2,1,1>(
                    new MyConstraintCase3(rho, beta)));
    }

private:
    const double rho_;
    const double beta_;
//    const double theta_;
};

struct NetworkOdomConstraint{
    NetworkOdomConstraint(double delta_t)
        : delta_t_(delta_t) {}

    template<typename T>
    bool operator()(
            const T* const cx,
            const T* const cy,
            const T* const cyaw,
            const T* const nx,
            const T* const ny,
            const T* const nyaw,
            T* residual) const {
        residual[0] = (nx[0] - (cx[0] + delta_t_*(v_j*cos(cyaw[0])+cy[0]*w_i-v_i))) / 0.1 ;
        residual[1] = (ny[0] - (cy[0] + delta_t_*(v_j*sin(cyaw[0])-cx[0]*w_i))) / 0.1 ;
        residual[2] = (nyaw[0] - (cyaw[0] + delta_t_*(w_j-w_i))) / 0.01 ;
        return true;
    }

    static ceres::CostFunction* Create(
            const double delta_t)
    {
        return (new ceres::AutoDiffCostFunction<NetworkOdomConstraint,3,1,1,1,1,1,1>(
                    new NetworkOdomConstraint(delta_t)));
    }

private:
    const double delta_t_;
    const double v_i = 0.2;
    const double w_i = 0.1;
    const double v_j = 0.4;
    const double w_j = 0.09;
};

struct withoutNetworkOdomConstraint{
    withoutNetworkOdomConstraint(double delta_t)
        : delta_t_(delta_t) {}

    template<typename T>
    bool operator()(
            const T* const cx,
            const T* const cy,
            const T* const cyaw,
            const T* const vj,
            const T* const wj,
            const T* const nx,
            const T* const ny,
            const T* const nyaw,
            const T* const nvj,
            const T* const nwj,
            T* residual) const {
        residual[0] = (nx[0] - (cx[0] + delta_t_*(vj[0]*cos(cyaw[0])+cy[0]*w_i-v_i))) / 0.1;
        residual[1] = (ny[0] - (cy[0] + delta_t_*(vj[0]*sin(cyaw[0])-cx[0]*w_i))) / 0.1;
        residual[2] = (nyaw[0] - (cyaw[0] + delta_t_*(wj[0]-w_i))) / 0.01;
        residual[3] = (nvj[0] - vj[0]) / 0.01;
        residual[4] = (nwj[0] - wj[0]) / 0.01;
        return true;
    }

    static ceres::CostFunction* Create(
            const double delta_t)
    {
        return (new ceres::AutoDiffCostFunction<withoutNetworkOdomConstraint,5,1,1,1,1,1,1,1,1,1,1>(
                    new withoutNetworkOdomConstraint(delta_t)));
    }

private:
    const double delta_t_;
    const double v_i = 0.2;
    const double w_i = 0.1;
//    const double v_j = 0.4;
//    const double w_j = 0.09;
};

double findMedian(const vector<double>& data)
{
    vector<double> sortedData = data;
    // Sort the vector in ascending order
    std::sort(sortedData.begin(), sortedData.end());

    // Calculate the median
    double median;
    int size = sortedData.size();

    if (size % 2 == 0){
        // If enven number of elements, take the average of the two middle elements
        median = static_cast<double>(sortedData[size/2 - 1] + sortedData[size/2 + 1]) / 2.0;
    } else{
        // If odd number of elements, the median is the middle element
        median = static_cast<double>(sortedData[size/2]);
    }
    return median;
}

double findFirstQuartile(const vector<double>& data){
    vector<double> sortedData = data;

    // Sort the copy in ascending order
    std::sort(sortedData.begin(), sortedData.end());

    // Calculate the index for the first quartile (Q1)
    int n = sortedData.size();
    double index = (n + 1.0) / 4.0;
//    cout << "index is : " <<  index << endl;
    int lowerIndex = static_cast<int>(index);
//    cout << "lower index is : " <<  lowerIndex << endl;

    int upperIndex = lowerIndex + 1;

    // Check if the index is an integer or requires interpolation
    if (index == lowerIndex){
        // If the index is an integer, Q1 is the value at that index
        return sortedData[lowerIndex - 1];
    } else{
        // If the index is not an integer, interpolate between the two nearest values
        double lowerValue = sortedData[lowerIndex - 1];
        double upperValue = sortedData[upperIndex - 1];
        double interpolationFactor = index - lowerIndex;

        return lowerValue + interpolationFactor * (upperValue - lowerValue);
    }
}

double findThirdQuartile(const std::vector<double>& data) {
    // Make a copy of the input data vector
    std::vector<double> sortedData = data;

    // Sort the copy in ascending order
    std::sort(sortedData.begin(), sortedData.end());

    // Calculate the index for the third quartile (Q3)
    int n = sortedData.size();
    double index = (3.0 * n + 1.0) / 4.0; // 3/4 of the way from the start
    int lowerIndex = static_cast<int>(index);
    int upperIndex = lowerIndex + 1;
//    cout << "Index is : " << index << endl;
//    cout << "lowerIndex is : " << lowerIndex << endl;

    // Check if the index is an integer or requires interpolation
    if (index == lowerIndex) {
        // If the index is an integer, Q3 is the value at that index
        return sortedData[lowerIndex - 1]; // Subtract 1 to convert to 0-based index
    } else {
        // If the index is not an integer, interpolate between the two nearest values
        double lowerValue = sortedData[lowerIndex - 1];
        double upperValue = sortedData[upperIndex - 1];
        double interpolationFactor = index - lowerIndex;

        return lowerValue + interpolationFactor * (upperValue - lowerValue);
    }
}

int main(int argc, char** argv){
  cout<<"Start similation"<<endl;
  google::InitGoogleLogging(argv[0]);

  //data read
//  CSVParser csvparser("/home/gihun/catkin_tools_ws/files/1003/_2023-10-03-15-19-08-sync_data.csv");
  CSVParser csvparser("/home/gihun/catkin_tools_ws/files/1016/2023-10-16-15-00-50-sync_data.csv");

  int colSize = csvparser.ncol_ / 2;

  //parameter
  vector<double> xji;
  vector<double> yji;
  vector<double> thetaji;
  //ground truth
  vector<double> t_xji;
  vector<double> t_yji;
  vector<double> t_thetaji;
  //kf estimated
  // case0
  vector<double> c0_xji;
  vector<double> c0_yji;
  vector<double> c0_thetaji;
  // case1
  vector<double> c1_xji;
  vector<double> c1_yji;
  vector<double> c1_thetaji;
  // case2
  vector<double> c2_xji;
  vector<double> c2_yji;
  vector<double> c2_thetaji;
  // case3
  vector<double> c3_xji;
  vector<double> c3_yji;
  vector<double> c3_thetaji;
  // case4
  vector<double> c4_xji;
  vector<double> c4_yji;
  vector<double> c4_thetaji;
  // observation
  vector<double> rho;
  vector<double> beta;
  vector<double> theta;
  // obervation with outliers
  vector<double> o_rho;
  vector<double> o_beta;
  vector<double> o_theta;
  // input
  vector<double> delta_t;
  // for visualization
  vector<double> visualization;
  // for case4
//  vector<double> vj(csvparser.ncol_, 0.3);
//  vector<double> wj(csvparser.ncol_, 0.05);
//  vector<double> trueVj(csvparser.ncol_, 0.2);
//  vector<double> trueWj(csvparser.ncol_, 0.1);

  vector<double> vj(colSize, 0.3);
  vector<double> wj(colSize, 0.05);
  vector<double> trueVj(colSize, 0.4);
  vector<double> trueWj(colSize, 0.09);

  for(int i=0;i< colSize ;i++){
      xji.push_back(csvparser.data_[i][5]);
      yji.push_back(csvparser.data_[i][6]);
      thetaji.push_back(csvparser.data_[i][7]);
      rho.push_back(csvparser.data_[i][8]);
      beta.push_back(csvparser.data_[i][9]);
      theta.push_back(csvparser.data_[i][10]);
      delta_t.push_back(csvparser.data_[i][11]);
      t_xji.push_back(csvparser.data_[i][12]);
      t_yji.push_back(csvparser.data_[i][13]);
      t_thetaji.push_back(csvparser.data_[i][14]);
      c0_xji.push_back(csvparser.data_[i][15]);
      c0_yji.push_back(csvparser.data_[i][16]);
      c0_thetaji.push_back(csvparser.data_[i][17]);
      c1_xji.push_back(csvparser.data_[i][18]);
      c1_yji.push_back(csvparser.data_[i][19]);
      c1_thetaji.push_back(csvparser.data_[i][20]);
      c2_xji.push_back(csvparser.data_[i][21]);
      c2_yji.push_back(csvparser.data_[i][22]);
      c2_thetaji.push_back(csvparser.data_[i][23]);
      c3_xji.push_back(csvparser.data_[i][24]);
      c3_yji.push_back(csvparser.data_[i][25]);
      c3_thetaji.push_back(csvparser.data_[i][26]);
      c4_xji.push_back(csvparser.data_[i][27]);
      c4_yji.push_back(csvparser.data_[i][28]);
      c4_thetaji.push_back(csvparser.data_[i][29]);
      visualization.push_back(i); // for matplotlib plot
  }

  /*************************Adds outliers to measurement data************************/
  o_rho = rho;
  o_beta = beta;
  o_theta = theta;

  double oRhoMedian = findMedian(o_rho);
  double oBetaMedian = findMedian(o_beta);
  double oThetaMedian = findMedian(o_theta);

  double oRhoFirstQuatile = findFirstQuartile(o_rho);
  double oBetaFirstQuatile = findFirstQuartile(o_beta);
  double oThetaFirstQuatile = findFirstQuartile(o_theta);

  double oRhoThirdQuatile = findThirdQuartile(o_rho);
  double oBetaThirdQuatile = findThirdQuartile(o_beta);
  double oThetaThirdQuatile = findThirdQuartile(o_theta);

  double oRhoIQR = oRhoThirdQuatile - oRhoFirstQuatile;
  double oBetaIQR = oBetaThirdQuatile - oBetaFirstQuatile;
  double oThetaIQR = oThetaThirdQuatile - oThetaFirstQuatile;

  double lowerOutlierRhoVal = oRhoFirstQuatile - 1.5 * oRhoIQR;
  double upperOutlierRhoVal = oRhoThirdQuatile + 1.5 * oRhoIQR;
  double lowerOutlierBetaVal = oBetaFirstQuatile - 1.5 * oBetaIQR;
  double upperOutlierBetaVal = oBetaThirdQuatile + 1.5 * oBetaIQR;
  double lowerOutlierThetaVal = oThetaFirstQuatile - 1.5 * oThetaIQR;
  double upperOutlierThetaVal = oThetaThirdQuatile + 1.5 * oThetaIQR;

  // Seed the radnom number generator
  unsigned int seed{0};
  std::srand(seed);

  // Calculate the number of elements to select
  int numElementsToSelect = o_rho.size() * 0.4; // change the outlier ratio here
  cout << numElementsToSelect << endl;

  // Create a set to avoid duplication
  std::set<int> selectedIndices;

  // Select numElementsToSelect of the elements and add to them
  while (selectedIndices.size() < numElementsToSelect){
      int randomIndex = rand() % o_rho.size();
      if (selectedIndices.find(randomIndex) == selectedIndices.end()){
          selectedIndices.insert(randomIndex);

          // Generate a random integer between 0 and RAND_MAX
          int randomInt = std::rand();

          // Scale the random integer to a floating-point number in the range[0.0, 2.0)
          double randomFloat = static_cast<double>(randomInt) / RAND_MAX * 2.0;

          if (randomIndex %2 == 0)
          {
              o_rho[randomIndex] += upperOutlierRhoVal * randomFloat;
              o_beta[randomIndex] += upperOutlierBetaVal * randomFloat;
              o_theta[randomIndex] += upperOutlierThetaVal * randomFloat;
          }
          else {
              o_rho[randomIndex] -=  lowerOutlierRhoVal * randomFloat;
              o_beta[randomIndex] -= lowerOutlierBetaVal * randomFloat;
              o_theta[randomIndex] -= lowerOutlierThetaVal * randomFloat;
          }

      }
  }

//  plt::title("rho");
//  plt::plot(visualization, rho, "r*");
//  plt::plot(visualization, o_rho, "b*");
//  plt::show();

//  plt::title("beta");
//  plt::plot(visualization, beta, "r*");
//  plt::plot(visualization, o_beta, "b*");
//  plt::show();

//  plt::title("theta");
//  plt::plot(visualization, theta, "r*");
//  plt::plot(visualization, o_theta, "b*");
//  plt::show();
/*****************************************************************************/

  // init param
  vector<double> initialXji;
  vector<double> initialYji;
  vector<double> initialThetaji;
  initialXji=xji;
  initialYji=yji;
  initialThetaji=thetaji;
  vector<double> initialVj;
  vector<double> initialWj;
  initialVj = vj;
  initialWj = wj;

// ====================================Optimization========================================
  // Record the starting time
  auto start_time = std::chrono::high_resolution_clock::now();

  // Perform task that you want to measure the duration

  ceres::Problem problem;
  for(int i = 0; i < colSize - 1; i++){

//    // case1~3 odometry constraint
//    problem.AddResidualBlock(
//        NetworkOdomConstraint::Create(delta_t[i]),
//        NULL,
//        &(xji[i]),
//        &(yji[i]),
//        &(thetaji[i]),
//        &(xji[i+1]),
//        &(yji[i+1]),
//        &(thetaji[i+1])
//        );

     // case4 odometry constraint
    problem.AddResidualBlock(
          withoutNetworkOdomConstraint::Create(delta_t[i]),
          NULL,
          &(xji[i]),
          &(yji[i]),
          &(thetaji[i]),
          &(vj[i]),
          &(wj[i]),
          &(xji[i+1]),
          &(yji[i+1]),
          &(thetaji[i+1]),
          &(vj[i+1]),
          &(wj[i+1])
          );

    // // measurement constraint
//    problem.AddResidualBlock(
//      MyConstraintCase0::Create(rho[i],beta[i],theta[i]),
//      NULL,
//      &xji[i],
//      &yji[i],
//      &thetaji[i]
//      );

//    problem.AddResidualBlock(
//      MyConstraintCase1::Create(rho[i]),
//      NULL,
//      &xji[i],
//      &yji[i]
//      );

//    problem.AddResidualBlock(
//          MyConstraintCase2::Create(beta[i]),
//        NULL,
//        &xji[i],
//        &yji[i]
//          );

      problem.AddResidualBlock(
            MyConstraintCase3::Create(rho[i], beta[i]), // o_rhom, o_beta : data with outliers
          NULL,
          &xji[i],
          &yji[i]
            );
  }

  // Optimization
  Solver::Options options;
  options.linear_solver_type=ceres::DENSE_QR;
  options.minimizer_progress_to_stdout=true;
  Solver::Summary summary;
  Solve(options,&problem,&summary);

  for (auto& ele : thetaji)
    ele += 2 * M_PI;

  // Record the ending time
  auto end_time = std::chrono::high_resolution_clock::now();

  // Calculate the time duration
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

  // Output the duration in milliseconds
  std::cout << "Time duration: " << duration.count() << "milliseconds" << endl;



////**********************Save CSV file****************************

//  std::string header1 = "xji";
//  std::string header2 = "yji";
//  std::string header3 = "thetaji";
//  std::string header4 = "vj";
//  std::string header5 = "wj";
//  std::string header6 = "trueXji";
//  std::string header7 = "trueYji";
//  std::string header8 = "trueThetaji";

//  const std::string filename = "/home/gihun/catkin_ws/src/multiple_turtlebots_sim/data/withoutOutlier/Nls/test.csv";


//  std::ofstream outputFile(filename);

//  if (!outputFile.is_open()){
//      std::cerr << "Error: Unable to open the file " << filename << std::endl;
//      return 1;
//  }

//  // Write headers to the CSV file
//  outputFile << header1 << "," << header2 << "," << header3 <<  "," << header4 << "," << header5 <<
//                "," << header6 << "," << header7 << "," << header8 << std::endl;

//  // Write the vectors as columns in the CSV file
//  size_t maxLength = t_xji.size();
//  for (size_t i = 0; i < maxLength; ++i){
//      if (i < xji.size()){
//          outputFile << xji[i];
//      }
//      outputFile << ","; // Add a comma to separate columns
//      if (i < yji.size()){
//          outputFile << yji[i];
//      }
//      outputFile << ",";
//      if (i < thetaji.size()){
//          outputFile << thetaji[i];
//      }
//      outputFile << ",";
//      if (i < vj.size()){
//          outputFile << vj[i];
//      }
//      outputFile << ",";
//      if (i < wj.size()){
//          outputFile << wj[i];
//      }
//      outputFile << ",";
//      if (i < t_xji.size()){
//          outputFile << t_xji[i];
//      }
//      outputFile << ",";
//      if (i < t_yji.size()){
//          outputFile << t_yji[i];
//      }
//      outputFile << ",";
//      if (i < t_thetaji.size()){
//          outputFile << t_thetaji[i];
//      }
//      outputFile << std::endl; // Start a new row
//  }

//  outputFile.close();

//  std::cout << "Data saved to " << filename << std::endl;


//**********************Visualization****************************
//    // Set the "super title"
//    plt::suptitle("Odom_weight_(0.1,0.1,0.01), meas_weight_null, odom_rho_null, meas_rho_null");
//    plt::subplot(3,1,1);
//    plt::plot(t_xji, {{"label", "true_x"}});
//    plt::plot(xji, {{"label", "ceres_x"}});
//  //  plt::plot(ix, {{"label", "init_x"}});
//  //  plt::plot(c4_xji, {{"label", "kf_x"}});
//  //  plt::title("x_ji");
//    plt::grid(true);
//    plt::legend();


//    plt::subplot(3,1,2);
//    plt::plot(t_yji, {{"label", "true_y"}});
//    plt::plot(yji, {{"label", "ceres_y"}});
//  //  plt::plot(iy, {{"label", "init_y"}});
//  //  plt::plot(c4_yji, {{"label", "Kf_y"}});
//  //  plt::title("y_ji");
//    plt::grid(true);
//    plt::legend();

//    plt::subplot(3,1,3);
//    plt::plot(t_thetaji, {{"label", "true_theta"}});
//    plt::plot(thetaji, {{"label", "ceres_theta"}});
//  //  plt::plot(iyaw, {{"label", "init_theta"}});
//  //  plt::plot(c4_thetaji, {{"label", "Kf_theta"}});
//  //  plt::title("theta_ji");
//    plt::grid(true);
//    plt::legend();

//    plt::show();

//    plt::subplot(3,1,1);
//    plt::plot(t_xji, {{"label", "true_x"}});
//    plt::plot(xji, {{"label", "ceres_x"}});
//  //  plt::plot(ix, {{"label", "init_x"}});
//  //  plt::plot(c4_xji, {{"label", "kf_x"}});
//  //  plt::title("x_ji");
//    plt::grid(true);
//    plt::legend();


//    plt::subplot(3,1,2);
//    plt::plot(t_yji, {{"label", "true_y"}});
//    plt::plot(yji, {{"label", "ceres_y"}});
//  //  plt::plot(iy, {{"label", "init_y"}});
//  //  plt::plot(c4_yji, {{"label", "Kf_y"}});
//  //  plt::title("y_ji");
//    plt::grid(true);
//    plt::legend();

//    plt::subplot(3,1,3);
//    plt::plot(t_thetaji, {{"label", "true_theta"}});
//    plt::plot(thetaji, {{"label", "ceres_theta"}});
//  //  plt::plot(iyaw, {{"label", "init_theta"}});
//  //  plt::plot(c4_thetaji, {{"label", "Kf_theta"}});
//  //  plt::title("theta_ji");
//    plt::grid(true);
//    plt::legend();

//    plt::subplot(5,1,1);
//    plt::plot(t_xji, {{"label", "true_x"}});
//    plt::plot(xji, {{"label", "ceres_x"}});
//    plt::grid(true);
//    plt::legend();


//    plt::subplot(5,1,2);
//    plt::plot(t_yji, {{"label", "true_y"}});
//    plt::plot(yji, {{"label", "ceres_y"}});
//    plt::grid(true);
//    plt::legend();

//    plt::subplot(5,1,3);
//    plt::plot(t_thetaji, {{"label", "true_theta"}});
//    plt::plot(thetaji, {{"label", "ceres_theta"}});
//    plt::grid(true);
//    plt::legend();

//    plt::subplot(5,1,4);
//    plt::plot(trueVj, {{"label", "true_vj"}});
//    plt::plot(vj, {{"label", "vj"}});
//    plt::grid(true);
//    plt::legend();

//    plt::subplot(5,1,5);
//    plt::plot(trueWj, {{"label", "true_wj"}});
//    plt::plot(wj, {{"label", "wj"}});
//    plt::grid(true);
//    plt::legend();

//    plt::show();

  return 0;
}
