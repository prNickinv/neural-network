#include "ActivationFunction.h"

#include <cassert>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace NeuralNetwork {

ActivationFunction::ActivationFunction(
    const VecToVecFunc& activation_function,
    const VecToMatFunc& derivative_of_activation, ActivationFunctionType type)
    : activation_function_(activation_function),
      derivative_of_activation_(derivative_of_activation),
      type_(type) {
  assert(activation_function_ && "Activation function cannot be empty");
  assert(derivative_of_activation_ && "Activation derivative cannot be empty");
}

ActivationFunction::ActivationFunction(VecToVecFunc&& activation_function,
                                       VecToMatFunc&& derivative_of_activation,
                                       ActivationFunctionType type)
    : activation_function_(std::move(activation_function)),
      derivative_of_activation_(std::move(derivative_of_activation)),
      type_(type) {
  assert(activation_function_ && "Activation function cannot be empty");
  assert(derivative_of_activation_ && "Activation derivative cannot be empty");
}

Vector ActivationFunction::Activate(const Vector& input_vector) const {
  assert(activation_function_ && "Activation function cannot be empty");
  assert(input_vector.size() != 0 && "Input vector cannot be empty");
  return activation_function_(input_vector);
}

Matrix ActivationFunction::ComputeJacobianMatrix(
    const Vector& input_vector) const {
  assert(derivative_of_activation_ && "Activation derivative cannot be empty");
  assert(input_vector.size() != 0 && "Input vector cannot be empty");
  return derivative_of_activation_(input_vector);
}

ActivationFunction::operator bool() const {
  return activation_function_ && derivative_of_activation_;
}

bool ActivationFunction::IsActivationFunctionEmpty() const {
  return !activation_function_;
}

bool ActivationFunction::IsActivationDerivativeEmpty() const {
  return !derivative_of_activation_;
}

std::string ActivationFunction::GetType() const {
  switch (type_) {
    case ActivationFunctionType::Sigmoid: return "Sigmoid";
    case ActivationFunctionType::SigmoidUnstable: return "SigmoidUnstable";
    case ActivationFunctionType::Tanh: return "Tanh";
    case ActivationFunctionType::ReLu: return "ReLu";
    case ActivationFunctionType::LeakyReLu: return "LeakyReLu";
    case ActivationFunctionType::SoftMax: return "SoftMax";
    case ActivationFunctionType::SoftMaxUnstable: return "SoftMaxUnstable";
    default: return "Custom";
  }
}

ActivationFunction ActivationFunction::GetFunction(const std::string& type) {
  if (type == "Sigmoid") {
    return Sigmoid();
  } else if (type == "SigmoidUnstable") {
    return SigmoidUnstable();
  } else if (type == "Tanh") {
    return Tanh();
  } else if (type == "ReLu") {
    return ReLu();
  } else if (type == "LeakyReLu") {
    return LeakyReLu();
  } else if (type == "SoftMax") {
    return SoftMax();
  } else if (type == "SoftMaxUnstable") {
    return SoftMaxUnstable();
  } else {
    throw std::invalid_argument("Unknown activation function type");
  }
}

ActivationFunction ActivationFunction::Sigmoid() {
  DoubleToDoubleFunc elwise_sigmoid{[](double x) {
    return x >= 0.0 ? 1.0 / (1.0 + std::exp(-x))
                    : std::exp(x) / (1.0 + std::exp(x));
  }};

  DoubleToDoubleFunc elwise_derivative_of_sigmoid{[elwise_sigmoid](double x) {
    return elwise_sigmoid(x) * (1.0 - elwise_sigmoid(x));
  }};

  VecToVecFunc sigmoid{[elwise_sigmoid](const Vector& input_vector) {
    return input_vector.unaryExpr(elwise_sigmoid);
  }};

  VecToMatFunc derivative_of_sigmoid{[elwise_derivative_of_sigmoid](
                                         const Vector& input_vector) {
    return input_vector.unaryExpr(elwise_derivative_of_sigmoid).asDiagonal();
  }};

  return {sigmoid, derivative_of_sigmoid, ActivationFunctionType::Sigmoid};
}

ActivationFunction ActivationFunction::SigmoidUnstable() {
  DoubleToDoubleFunc elwise_sigmoid{[](double x) {
    return 1.0 / (1.0 + std::exp(-x));
  }};

  DoubleToDoubleFunc elwise_derivative_of_sigmoid{[elwise_sigmoid](double x) {
    return elwise_sigmoid(x) * (1.0 - elwise_sigmoid(x));
  }};

  VecToVecFunc sigmoid{[elwise_sigmoid](const Vector& input_vector) {
    return input_vector.unaryExpr(elwise_sigmoid);
  }};

  VecToMatFunc derivative_of_sigmoid{[elwise_derivative_of_sigmoid](
                                         const Vector& input_vector) {
    return input_vector.unaryExpr(elwise_derivative_of_sigmoid).asDiagonal();
  }};

  return {sigmoid, derivative_of_sigmoid,
          ActivationFunctionType::SigmoidUnstable};
}

ActivationFunction ActivationFunction::Tanh() {
  DoubleToDoubleFunc elwise_tanh{[](double x) {
    return std::tanh(x);
  }};

  DoubleToDoubleFunc elwise_derivative_of_tanh{[](double x) {
    return 1.0 - std::tanh(x) * std::tanh(x);
  }};

  VecToVecFunc tanh{[elwise_tanh](const Vector& input_vector) {
    return input_vector.unaryExpr(elwise_tanh);
  }};

  VecToMatFunc derivative_of_tanh{
      [elwise_derivative_of_tanh](const Vector& input_vector) {
        return input_vector.unaryExpr(elwise_derivative_of_tanh).asDiagonal();
      }};

  return {tanh, derivative_of_tanh, ActivationFunctionType::Tanh};
}

ActivationFunction ActivationFunction::ReLu() {
  VecToVecFunc relu{[](const Vector& input_vector) {
    return input_vector.cwiseMax(0.0);
  }};

  VecToMatFunc derivative_of_relu{[](const Vector& input_vector) {
    return input_vector.unaryExpr([](double x) { return x > 0.0 ? 1.0 : 0.0; })
        .asDiagonal();
  }};

  return {relu, derivative_of_relu, ActivationFunctionType::ReLu};
}

ActivationFunction ActivationFunction::LeakyReLu(double slope) {
  VecToVecFunc leaky_relu{[slope](const Vector& input_vector) {
    return input_vector.unaryExpr(
        [slope](double x) { return x > 0.0 ? x : slope * x; });
  }};

  VecToMatFunc derivative_of_leaky_relu{[slope](const Vector& input_vector) {
    return input_vector
        .unaryExpr([slope](double x) { return x > 0.0 ? 1.0 : slope; })
        .asDiagonal();
  }};

  return {leaky_relu, derivative_of_leaky_relu,
          ActivationFunctionType::LeakyReLu};
}

ActivationFunction ActivationFunction::SoftMax() {
  VecToVecFunc softmax{[](const Vector& input_vector) {
    auto exp_vector = (input_vector.array() - input_vector.maxCoeff()).exp();
    return exp_vector / exp_vector.sum();
  }};

  VecToMatFunc derivative_of_softmax{[softmax](const Vector& input_vector) {
    Vector activated_vector = softmax(input_vector);
    Matrix jacobian_matrix = -(activated_vector * activated_vector.transpose());
    jacobian_matrix += activated_vector.asDiagonal();
    return jacobian_matrix;
  }};

  return {softmax, derivative_of_softmax, ActivationFunctionType::SoftMax};
}

ActivationFunction ActivationFunction::SoftMaxUnstable() {
  VecToVecFunc softmax_unstable{[](const Vector& input_vector) {
    auto exp_vector = input_vector.array().exp();
    return exp_vector / exp_vector.sum();
  }};

  VecToMatFunc derivative_of_softmax_unstable{[softmax_unstable](
                                                  const Vector& input_vector) {
    Vector activated_vector = softmax_unstable(input_vector);
    Matrix jacobian_matrix = -(activated_vector * activated_vector.transpose());
    jacobian_matrix += activated_vector.asDiagonal();
    return jacobian_matrix;
  }};

  return {softmax_unstable, derivative_of_softmax_unstable,
          ActivationFunctionType::SoftMaxUnstable};
}

} // namespace NeuralNetwork
