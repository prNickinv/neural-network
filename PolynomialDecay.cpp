#include "PolynomialDecay.h"

#include <cmath>

namespace NeuralNetwork {

PolynomialDecay::PolynomialDecay(double initial_learning_rate, int decay_steps,
                                 double smallest_learning_rate, double power)
    : initial_learning_rate_(initial_learning_rate),
      decay_steps_(decay_steps),
      smallest_learning_rate_(smallest_learning_rate),
      power_(power),
      step_(0) {}

PolynomialDecay::PolynomialDecay(std::istream& is) {
  is >> initial_learning_rate_;
  is >> decay_steps_;
  is >> smallest_learning_rate_;
  is >> power_;
  is >> step_;
}

double PolynomialDecay::GetLearningRate() {
  if (step_ >= decay_steps_) {
    return smallest_learning_rate_;
  }

  double learning_rate = initial_learning_rate_
      * std::pow(1.0 - static_cast<double>(step_) / decay_steps_, power_);
  ++step_;
  return learning_rate;
}

std::ostream& operator<<(std::ostream& os, const PolynomialDecay& poly_decay) {
  os << "PolynomialDecay" << std::endl;
  os << poly_decay.initial_learning_rate_ << std::endl;
  os << poly_decay.decay_steps_ << std::endl;
  os << poly_decay.smallest_learning_rate_ << std::endl;
  os << poly_decay.power_ << std::endl;
  os << poly_decay.step_ << std::endl;
  return os;
}

} // namespace NeuralNetwork
