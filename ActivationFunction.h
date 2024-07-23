#ifndef ACTIVATIONFUNCTION_H
#define ACTIVATIONFUNCTION_H

#include <functional>
#include <string>

#include <Eigen/Dense>

#include "GlobalUsings.h"

namespace NeuralNetwork {

enum class ActivationFunctionType {
  Sigmoid,
  SigmoidUnstable,
  Tanh,
  ReLu,
  LeakyReLu,
  SoftMax,
  SoftMaxUnstable,
  Custom
};

class ActivationFunction {
  using VecToVecFunc = std::function<Vector(const Vector&)>;
  using VecToMatFunc = std::function<Matrix(const Vector&)>;
  using DoubleToDoubleFunc = std::function<double(double)>;

 public:
  ActivationFunction() = default;
  ActivationFunction(const VecToVecFunc&, const VecToMatFunc&,
                     ActivationFunctionType = ActivationFunctionType::Custom);
  ActivationFunction(VecToVecFunc&&, VecToMatFunc&&,
                     ActivationFunctionType = ActivationFunctionType::Custom);

  [[nodiscard]] Vector Activate(const Vector&) const;
  [[nodiscard]] Matrix ComputeJacobianMatrix(const Vector&) const;

  explicit operator bool() const;
  [[nodiscard]] bool IsActivationFunctionEmpty() const;
  [[nodiscard]] bool IsActivationDerivativeEmpty() const;

  [[nodiscard]] std::string GetType() const;
  static ActivationFunction GetFunction(const std::string&);

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
  ActivationFunctionType type_ = ActivationFunctionType::Custom;
};

} // namespace NeuralNetwork

#endif //ACTIVATIONFUNCTION_H
