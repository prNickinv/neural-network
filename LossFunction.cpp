#include "LossFunction.h"

#include <cassert>
#include <cmath>
#include <utility>

namespace NeuralNetwork {

LossFunction::LossFunction(
    const LossFunction::VecVecToDoubleFunc& loss_function,
    const LossFunction::VecVecToRowVecFunc& derivative_of_loss)
    : loss_function_(loss_function),
      derivative_of_loss_(derivative_of_loss) {
  assert(loss_function_ && "Loss function cannot be empty!");
  assert(derivative_of_loss_ && "Loss derivative cannot be empty!");
}

LossFunction::LossFunction(
    LossFunction::VecVecToDoubleFunc&& loss_function,
    LossFunction::VecVecToRowVecFunc&& derivative_of_loss)
    : loss_function_(std::move(loss_function)),
      derivative_of_loss_(std::move(derivative_of_loss)) {
  assert(loss_function_ && "Loss function cannot be empty!");
  assert(derivative_of_loss_ && "Loss derivative cannot be empty!");
}

double LossFunction::ComputeLoss(const LossFunction::Vector& output,
                                 const LossFunction::Vector& target) const {
  assert(loss_function_ && "Loss function cannot be empty!");
  assert(output.size() != 0 && "Output vector cannot be empty!");
  assert(target.size() != 0 && "Target vector cannot be empty!");
  return loss_function_(output, target);
}

LossFunction::RowVector LossFunction::ComputeInitialGradient(
    const LossFunction::Vector& output,
    const LossFunction::Vector& target) const {
  assert(derivative_of_loss_ && "Loss derivative cannot be empty!");
  assert(output.size() != 0 && "Output vector cannot be empty!");
  assert(target.size() != 0 && "Target vector cannot be empty!");
  return derivative_of_loss_(output, target);
}

LossFunction::operator bool() const {
  return loss_function_ && derivative_of_loss_;
}

bool LossFunction::IsLossFunctionEmpty() const {
  return !loss_function_;
}

bool LossFunction::IsLossDerivativeEmpty() const {
  return !derivative_of_loss_;
}

LossFunction LossFunction::MSE() {
  VecVecToDoubleFunc mse{[](const Vector& output, const Vector& target) {
    return (output - target).squaredNorm();
  }};

  VecVecToRowVecFunc derivative_of_mse{
      [](const Vector& output, const Vector& target) {
        return (2 * (output - target)).transpose();
      }};

  return {mse, derivative_of_mse};
}

LossFunction LossFunction::CrossEntropyLoss() {
  VecVecToDoubleFunc cross_entropy{
      [](const Vector& output, const Vector& target) {
        // -target * log(output)
        return -target.dot(output.array().log().matrix());
      }};

  VecVecToRowVecFunc cross_entropy_derivative{
      [](const Vector& output, const Vector& target) {
        return -target.cwiseQuotient(output)
                    .transpose(); // d(CE)/d(output) = -target / output
      }};

  return {cross_entropy, cross_entropy_derivative};
}

} // namespace NeuralNetwork
