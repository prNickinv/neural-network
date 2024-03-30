#ifndef ACTIVATIONFUNCTION_H
#define ACTIVATIONFUNCTION_H

#include <functional>

#include <Eigen/Dense>

namespace NeuralNetwork {

class ActivationFunction {
  using Vector = Eigen::VectorXd;
  using Matrix = Eigen::MatrixXd;
  using Index = Eigen::Index;
  using VecToVecFunc = std::function<Vector(const Vector&)>;
  using VecToMatFunc = std::function<Matrix(const Vector&)>;
  using DoubleToDoubleFunc = std::function<double(double)>;

 public:
  ActivationFunction(const VecToVecFunc&, const VecToMatFunc&);
  ActivationFunction(VecToVecFunc&&, VecToMatFunc&&);

  [[nodiscard]] Vector Activate(const Vector&) const;
  [[nodiscard]] Matrix ComputeJacobianMatrix(const Vector&) const;

  explicit operator bool() const;
  //TODO: Mark //NOLINT to suppress clang-tidy warning?
  bool IsActivationFunctionEmpty() const;
  bool IsActivationDerivativeEmpty() const;

  static ActivationFunction Sigmoid();
  static ActivationFunction SigmoidUnstable();
  static ActivationFunction Tanh();
  static ActivationFunction ReLu();
  static ActivationFunction LeakyReLu(double = 0.01);
  static ActivationFunction SoftMax();
  static ActivationFunction SoftMaxUnstable();

 private:
  VecToVecFunc activation_function_;
  VecToMatFunc derivative_of_activation_;
};

} // namespace NeuralNetwork

#endif //ACTIVATIONFUNCTION_H
