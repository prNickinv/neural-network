#ifndef MINIBATCHGD_H
#define MINIBATCHGD_H

#include <iostream>

#include <Eigen/Dense>

#include "GlobalUsings.h"
#include "SchedulerUtils.h"

namespace NeuralNetwork {

class MiniBatchGD {
 public:
  MiniBatchGD() = default;
  MiniBatchGD(double, double);
  explicit MiniBatchGD(std::istream&);

  template<typename SchedulerType>
  MiniBatchGD(const SchedulerType& scheduler, double weights_decay)
      : learning_rate_(scheduler),
        weights_decay_(weights_decay) {}

  Parameters UpdateParameters(const Matrix&, const Vector&, const Matrix&,
                              const Vector&, int);

  friend std::ostream& operator<<(std::ostream&, const MiniBatchGD&);

 private:
  double GetLearningRate();
  Matrix ApplyWeightsDecay(const Matrix&, int, double) const;

  static constexpr double default_learning_rate_{0.01};
  static constexpr double default_weights_decay_{0.0};

  //double learning_rate_{default_learning_rate_};
  Scheduler learning_rate_{default_learning_rate_};
  double weights_decay_{default_weights_decay_};
};

} // namespace NeuralNetwork

#endif //MINIBATCHGD_H
