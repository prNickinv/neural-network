#include "MSE.h"

namespace NeuralNetwork {

MSE::MSE()
    : loss_function_(
          [](const MSE::ColumnVector& output, const MSE::ColumnVector& target) {
            return (output - target).squaredNorm();
          }),
      loss_function_derivative_(
          [](const MSE::ColumnVector& output, const MSE::ColumnVector& target) {
            return (2 * (output - target)).transpose();
          }) {}

MSE::LossFunction MSE::GetLossFunction() {
  return loss_function_;
}

MSE::LossFunctionDerivative MSE::GetLossFunctionDerivative() {
  return loss_function_derivative_;
}

} // namespace NeuralNetwork
