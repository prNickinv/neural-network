#include "Sigmoid.h"

namespace NeuralNetwork {

Sigmoid::Sigmoid()
    : activation_function_([](const SampleVector& modified_vector) {
        return modified_vector.unaryExpr(element_wise_activation_);
      }),
      derivative_of_activation_([](const SampleVector& modified_vector) {
        return modified_vector.unaryExpr(element_wise_activation_derivative_)
            .asDiagonal();
      }) {}

Sigmoid::ActivationFunction Sigmoid::Activate() const {
  return activation_function_;
}

Sigmoid::DerivativeOfActivation Sigmoid::GetDerivative() const {
  return derivative_of_activation_;
}

} // namespace NeuralNetwork
