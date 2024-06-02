#include "ExponentialDecay.h"

#include <cmath>

namespace NeuralNetwork {

ExponentialDecay::ExponentialDecay(double initial_learning_rate, double decay_rate)
    : initial_learning_rate_(initial_learning_rate),
      decay_rate_(decay_rate),
      step_(0) {}

ExponentialDecay::ExponentialDecay(std::istream& is) {
  is >> initial_learning_rate_;
  is >> decay_rate_;
  is >> step_;
}

double ExponentialDecay::GetLearningRate() {
  double learning_rate = initial_learning_rate_ * std::exp(-decay_rate_ * step_);
  ++step_;
  return learning_rate;
}

std::ostream& operator<<(std::ostream& os, const ExponentialDecay& exp_decay) {
  os << "ExponentialDecay" << std::endl;
  os << exp_decay.initial_learning_rate_ << std::endl;
  os << exp_decay.decay_rate_ << std::endl;
  os << exp_decay.step_ << std::endl;
  return os;
}

} // namespace NeuralNetwork
