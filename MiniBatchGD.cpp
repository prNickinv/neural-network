#include "MiniBatchGD.h"

namespace NeuralNetwork {

MiniBatchGD::MiniBatchGD(double learning_rate, double weights_decay)
    : learning_rate_{learning_rate}, weights_decay_{weights_decay} {}

MiniBatchGD::MiniBatchGD(std::istream& is) {
  is >> learning_rate_;
  is >> weights_decay_;
}

Matrix MiniBatchGD::ApplyWeightsDecay(const Matrix& weights, int batch_size) const {
  return weights - (learning_rate_ * weights_decay_ / batch_size) * weights;
}

Parameters MiniBatchGD::UpdateParameters(const Matrix& weights,
                                         const Vector& bias,
                                         const Matrix& weights_gradient,
                                         const Vector& bias_gradient,
                                         int batch_size) const {
  Parameters parameters{weights, bias};
  parameters.weights = ApplyWeightsDecay(parameters.weights, batch_size);
  parameters.weights -= (learning_rate_ / batch_size) * weights_gradient;
  parameters.bias -= (learning_rate_ / batch_size) * bias_gradient;
  return parameters;
}

std::ostream& operator<<(std::ostream& os, const MiniBatchGD& mbgd) {
  os << mbgd.learning_rate_ << std::endl;
  os << mbgd.weights_decay_ << std::endl;
  return os;
}

} // namespace NeuralNetwork
