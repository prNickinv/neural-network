#ifndef RELU_H
#define RELU_H

#include <functional>

#include <Eigen/Dense>

namespace NeuralNetwork {

class ReLu {
 public:
  using ActivationFunction =
      std::function<Eigen::VectorXd(const Eigen::VectorXd&)>;
  using DerivativeOfActivation =
      std::function<Eigen::MatrixXd(const Eigen::VectorXd&)>;
  using SampleVector = Eigen::VectorXd;

  ReLu();

  ActivationFunction Activate() const;
  DerivativeOfActivation GetDerivative() const;

 private:
  ActivationFunction activation_function_;
  DerivativeOfActivation derivative_of_activation_;
};

} // namespace NeuralNetwork

#endif //RELU_H
