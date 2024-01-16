#ifndef SIGMOID_H
#define SIGMOID_H

#include <cmath>
#include <functional>

#include <Eigen/Dense>

namespace NeuralNetwork {

class Sigmoid {
 public:
  using ElementWiseFunction = std::function<double(double)>;
  using ActivationFunction =
      std::function<Eigen::VectorXd(const Eigen::VectorXd&)>;
  using DerivativeOfActivation =
      std::function<Eigen::MatrixXd(const Eigen::VectorXd&)>;
  using SampleVector = Eigen::VectorXd;

  Sigmoid();

  ActivationFunction Activate() const;
  DerivativeOfActivation GetDerivative() const;

 private:
  static inline ElementWiseFunction element_wise_activation_{[](double x) {
    return 1.0 / (1 + std::exp(-x));
  }};
  static inline ElementWiseFunction element_wise_activation_derivative_{
      [](double x) {
        return element_wise_activation_(x) * (1 - element_wise_activation_(x));
      }};

  ActivationFunction activation_function_;
  DerivativeOfActivation derivative_of_activation_;
};

} // namespace NeuralNetwork

#endif //SIGMOID_H
