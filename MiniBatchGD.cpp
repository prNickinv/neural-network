#include "MiniBatchGD.h"

namespace NeuralNetwork {

MiniBatchGD::MiniBatchGD(double learning_rate, double weights_decay)
    : learning_rate_(learning_rate),
      weights_decay_(weights_decay) {}

MiniBatchGD::MiniBatchGD(std::istream& is) {
  learning_rate_ = SchedulerUtils::GetScheduler(is);
  is >> weights_decay_;
}

Parameters MiniBatchGD::UpdateParameters(const Matrix& weights,
                                         const Vector& bias,
                                         const Matrix& weights_gradient,
                                         const Vector& bias_gradient,
                                         int batch_size) {
  double learning_rate = GetLearningRate();

  Parameters parameters{weights, bias};
  parameters.weights =
      ApplyWeightsDecay(parameters.weights, batch_size, learning_rate);

  parameters.weights -= (learning_rate / batch_size) * weights_gradient;
  parameters.bias -= (learning_rate / batch_size) * bias_gradient;
  return parameters;
}

std::ostream& operator<<(std::ostream& os, const MiniBatchGD& mbgd) {
  os << "MiniBatchGD" << std::endl;

  if (std::holds_alternative<double>(mbgd.learning_rate_)) {
    os << "ConstantLR" << std::endl;
  }
  // In case of scheduler the name will be passed by the scheduler itself
  auto save_scheduler =
      SchedulerUtils::Overload{[&](double lr) { os << lr << std::endl; },
                               [&](auto& scheduler) {
                                 os << scheduler;
                               }};
  std::visit(save_scheduler, mbgd.learning_rate_);
  //os << mbgd.learning_rate_ << std::endl;
  os << mbgd.weights_decay_ << std::endl;
  return os;
}

double MiniBatchGD::GetLearningRate() {
  auto get_lr = SchedulerUtils::Overload{[&](double lr) { return lr; },
                                         [&](auto& scheduler) {
                                           return scheduler.GetLearningRate();
                                         }};
  return std::visit(get_lr, learning_rate_);
}

Matrix MiniBatchGD::ApplyWeightsDecay(const Matrix& weights, int batch_size,
                                      double learning_rate) const {
  return weights - (learning_rate * weights_decay_ / batch_size) * weights;
}

} // namespace NeuralNetwork
