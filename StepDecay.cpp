#include "StepDecay.h"

#include <cmath>

namespace NeuralNetwork {

StepDecay::StepDecay(double initial_learning_rate, int decay_steps,
                     double decay_rate)
    : initial_learning_rate_(initial_learning_rate),
      decay_steps_(decay_steps),
      decay_rate_(decay_rate),
      step_(0) {}

StepDecay::StepDecay(std::istream& is) {
  is >> initial_learning_rate_;
  is >> decay_steps_;
  is >> decay_rate_;
  is >> step_;
}

// Decay with factor 'decay_rate' every 'decay_steps' steps.
double StepDecay::GetLearningRate() {
  double learning_rate = initial_learning_rate_
      * std::pow(decay_rate_, static_cast<double>(step_) / decay_steps_);
  ++step_;
  return learning_rate;
}

std::ostream& operator<<(std::ostream& os, const StepDecay& step_decay) {
  os << "StepDecay" << std::endl;
  os << step_decay.initial_learning_rate_ << std::endl;
  os << step_decay.decay_steps_ << std::endl;
  os << step_decay.decay_rate_ << std::endl;
  os << step_decay.step_ << std::endl;
  return os;
}

} // namespace NeuralNetwork
