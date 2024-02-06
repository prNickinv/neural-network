#include "NormalRandom.h"

namespace NeuralNetwork {

NormalRandom::NormalRandom() : generator_(rd_()) {}

double NormalRandom::operator()() {
  std::normal_distribution<double> dis(0, 1);
  return dis(generator_);
}

} // namespace NeuralNetwork
