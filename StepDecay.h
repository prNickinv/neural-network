#ifndef STEPDECAY_H
#define STEPDECAY_H

#include <iostream>

namespace NeuralNetwork {

class StepDecay {
 public:
  StepDecay(double, int, double);
  explicit StepDecay(std::istream& is);

  double GetLearningRate();

  friend std::ostream& operator<<(std::ostream&, const StepDecay&);

 private:
  double initial_learning_rate_;
  int decay_steps_;
  double decay_rate_;
  int step_;
};

} // namespace NeuralNetwork

#endif //STEPDECAY_H
