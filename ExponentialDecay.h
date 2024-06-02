#ifndef EXPONENTIALDECAY_H
#define EXPONENTIALDECAY_H

#include <iostream>

namespace NeuralNetwork {

class ExponentialDecay {
 public:
  ExponentialDecay(double, int, double);
  explicit ExponentialDecay(std::istream& is);

  double GetLearningRate();

  friend std::ostream& operator<<(std::ostream&, const ExponentialDecay&);

 private:
  double initial_learning_rate_;
  int decay_steps_;
  double decay_rate_;
  int step_;
};

} // namespace NeuralNetwork

#endif //EXPONENTIALDECAY_H
