#ifndef NORMALRANDOM_H
#define NORMALRANDOM_H

#include <random>

namespace NeuralNetwork {

class NormalRandom {
 public:
  NormalRandom();

  double operator()();

 private:
  std::random_device rd_;
  std::mt19937 generator_;
};

} // namespace NeuralNetwork

#endif //NORMALRANDOM_H
