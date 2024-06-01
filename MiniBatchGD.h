#ifndef MINIBATCHGD_H
#define MINIBATCHGD_H

#include <iostream>

#include <Eigen/Dense>

#include "GlobalUsings.h"

namespace NeuralNetwork {

class MiniBatchGD {
 public:
  MiniBatchGD() = default;
  MiniBatchGD(double, double);
  explicit MiniBatchGD(std::istream&);

  Parameters UpdateParameters(const Matrix&, const Vector&, const Matrix&,
                              const Vector&, int) const;

  friend std::ostream& operator<<(std::ostream&, const MiniBatchGD&);

 private:
  Matrix ApplyWeightsDecay(const Matrix&, int) const;

  static constexpr double default_learning_rate_{0.01};
  static constexpr double default_weights_decay_{0.0};

  double learning_rate_{default_learning_rate_};
  double weights_decay_{default_weights_decay_};
};

} // namespace NeuralNetwork

#endif //MINIBATCHGD_H
