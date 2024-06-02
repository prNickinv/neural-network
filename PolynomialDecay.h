#ifndef POLYNOMIALDECAY_H
#define POLYNOMIALDECAY_H

#include <iostream>

namespace NeuralNetwork {

class PolynomialDecay {
 public:
  PolynomialDecay(double, int, double, double);
  explicit PolynomialDecay(std::istream& is);

  double GetLearningRate();

  friend std::ostream& operator<<(std::ostream&, const PolynomialDecay&);

 private:
  double initial_learning_rate_;
  int decay_steps_;
  double smallest_learning_rate_;
  double power_;
  int step_;
};

} // namespace NeuralNetwork

#endif //POLYNOMIALDECAY_H
