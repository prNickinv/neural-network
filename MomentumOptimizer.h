#ifndef MOMENTUMOPTIMIZER_H
#define MOMENTUMOPTIMIZER_H

#include <iostream>

#include <Eigen/Dense>

#include "GlobalUsings.h"

namespace NeuralNetwork {

class MomentumOptimizer {
 public:
  MomentumOptimizer() = default;
  explicit MomentumOptimizer(double);
  explicit MomentumOptimizer(std::istream&);

  void Resize(Index, Index);
  void UpdateVelocity(const Matrix&, const Vector&, double);

  Matrix GetVelocityWeights() const;
  Vector GetVelocityBias() const;

  friend std::ostream& operator<<(std::ostream&, const MomentumOptimizer&);

 private:
  static constexpr double default_gamma_{0.9};

  double gamma_{default_gamma_};
  Matrix v_w_;
  Vector v_b_;
};

} // namespace NeuralNetwork

#endif //MOMENTUMOPTIMIZER_H
