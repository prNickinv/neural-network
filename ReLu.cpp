#include "ReLu.h"

namespace NeuralNetwork {

ReLu::ReLu()
    : activation_function_([](const SampleVector& modified_vector) {
        return modified_vector.unaryExpr(
            [](double x) { return x > 0.0 ? x : 0.0; });
      }),
      derivative_of_activation_([](const SampleVector& modified_vector) {
        return modified_vector
            .unaryExpr([](double x) { return x > 0.0 ? 1.0 : 0.0; })
            .asDiagonal();
      }) {}

ReLu::ActivationFunction ReLu::Activate() const {
  return activation_function_;
}

ReLu::DerivativeOfActivation ReLu::GetDerivative() const {
  return derivative_of_activation_;
}

} // namespace NeuralNetwork