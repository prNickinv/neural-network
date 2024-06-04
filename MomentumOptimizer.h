#ifndef MOMENTUMOPTIMIZER_H
#define MOMENTUMOPTIMIZER_H

#include <iostream>

#include <Eigen/Dense>

#include "GlobalUsings.h"
#include "SchedulerUtils.h"

namespace NeuralNetwork {

enum class Nesterov { Enable, Disable };

class MomentumOptimizer {
 public:
  MomentumOptimizer() = default;
  MomentumOptimizer(double, double, double, Nesterov = Nesterov::Disable);
  explicit MomentumOptimizer(std::istream&);

  template<typename SchedulerType>
  MomentumOptimizer(const SchedulerType& scheduler, double weights_decay,
                    double gamma, Nesterov nesterov = Nesterov::Disable)
      : learning_rate_(scheduler),
        weights_decay_(weights_decay),
        gamma_(gamma),
        nesterov_(nesterov) {}

  void Initialize(Index, Index);
  Parameters UpdateParameters(const Matrix&, const Vector&, const Matrix&,
                              const Vector&, int);

  friend std::ostream& operator<<(std::ostream&, const MomentumOptimizer&);

 private:
  double GetLearningRate();
  [[nodiscard]] Matrix ApplyWeightsDecay(const Matrix&, int, double) const;
  void UpdateVelocity(const Matrix&, const Vector&, double);

  Matrix ComputeNesterovUpdateWeights(const Matrix&, int, double);
  Vector ComputeNesterovUpdateBias(const Vector&, int, double);

  static constexpr double default_learning_rate_{0.01};
  static constexpr double default_weights_decay_{0.0};
  static constexpr double default_gamma_{0.9};
  static constexpr Nesterov default_nesterov_{Nesterov::Disable};

  Scheduler learning_rate_{default_learning_rate_};
  double weights_decay_{default_weights_decay_};
  double gamma_{default_gamma_};
  Nesterov nesterov_{default_nesterov_};

  Matrix v_w_;
  Vector v_b_;
};

} // namespace NeuralNetwork

#endif //MOMENTUMOPTIMIZER_H
