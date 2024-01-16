#ifndef MSE_H
#define MSE_H

#include <functional>

#include <Eigen/Dense>

namespace NeuralNetwork {

class MSE {
 public:
  using LossFunction =
      std::function<double(const Eigen::VectorXd&, const Eigen::VectorXd&)>;
  using LossFunctionDerivative = std::function<Eigen::RowVectorXd(
      const Eigen::VectorXd&, const Eigen::VectorXd&)>;
  using ColumnVector = Eigen::VectorXd;

  MSE();

  LossFunction GetLossFunction();
  LossFunctionDerivative GetLossFunctionDerivative();

 private:
  LossFunction loss_function_;
  LossFunctionDerivative loss_function_derivative_;
};

} // namespace NeuralNetwork

#endif //MSE_H
