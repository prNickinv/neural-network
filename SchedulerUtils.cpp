#include "SchedulerUtils.h"

namespace NeuralNetwork::SchedulerUtils {

Scheduler GetScheduler(std::istream& is) {
  std::string scheduler_type;
  is >> scheduler_type;

  if (scheduler_type == "ExponentialDecay") {
    return ExponentialDecay(is);
  } else if (scheduler_type == "PolynomialDecay") {
    return PolynomialDecay(is);
  }

  double constant_learning_rate;
  is >> constant_learning_rate;
  return constant_learning_rate;
}

} // namespace NeuralNetwork::SchedulerUtils
