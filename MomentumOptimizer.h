#ifndef MOMENTUMOPTIMIZER_H
#define MOMENTUMOPTIMIZER_H

#include <iostream>

#include <Eigen/Dense>

#include "GlobalUsings.h"

namespace NeuralNetwork {

enum class Nesterov { Enable, Disable };

class MomentumOptimizer {
 public:
  MomentumOptimizer() = default;
  MomentumOptimizer(double, double, double, Nesterov = Nesterov::Disable);
  explicit MomentumOptimizer(std::istream&);

  void Initialize(Index, Index);
  Parameters UpdateParameters(const Matrix&, const Vector&, const Matrix&,
                              const Vector&, int);

  friend std::ostream& operator<<(std::ostream&, const MomentumOptimizer&);

 private:
  Matrix ApplyWeightsDecay(const Matrix&, int) const;
  void UpdateVelocity(const Matrix&, const Vector&);

  static constexpr double default_learning_rate_{0.01};
  static constexpr double default_weights_decay_{0.0};
  static constexpr double default_gamma_{0.9};
  static constexpr Nesterov default_nesterov_{Nesterov::Disable};

  double learning_rate_{default_learning_rate_};
  double weights_decay_{default_weights_decay_};
  double gamma_{default_gamma_};
  Nesterov nesterov_{default_nesterov_};

  Matrix v_w_;
  Vector v_b_;
};

} // namespace NeuralNetwork

#endif //MOMENTUMOPTIMIZER_H
