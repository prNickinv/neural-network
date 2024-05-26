#ifndef ADAMWOPTIMIZER_H
#define ADAMWOPTIMIZER_H

#include <iostream>

#include <Eigen/Dense>

#include "GlobalUsings.h"

namespace NeuralNetwork {

struct AdamWMoments {
  Matrix m_w{Matrix::Zero(1, 1)}; // First moment for weights
  Matrix v_w{Matrix::Zero(1, 1)}; // Second moment for weights
  Vector m_b{Vector::Zero(1)};    // First moment for bias
  Vector v_b{Vector::Zero(1)};    // Second moment for bias
};

class AdamWOptimizer {
 public:
  AdamWOptimizer() = default;
  AdamWOptimizer(double, double, double, double, double);
  explicit AdamWOptimizer(std::istream&);

  void Initialize(Index, Index);
  Matrix ApplyWeightsDecay(const Matrix&, int) const;
  void UpdateMoments(const Matrix&, const Vector&);
  AdamWMoments ComputeCorrectedMoments() const;
  // TODO: Remove
  double GetEpsilon() const;

  Matrix ComputeNewWeights(const Matrix&, const Matrix&, const Matrix&,
                           int) const;
  Vector ComputeNewBias(const Vector&, const Vector&, const Vector&, int) const;
  Parameters UpdateParameters(const Matrix&, const Vector&, const Matrix&,
                              const Vector&, int);

  friend std::ostream& operator<<(std::ostream&, const AdamWOptimizer&);
  friend std::istream& operator>>(std::istream&, AdamWOptimizer&);

 private:
  static constexpr double default_learning_rate_{0.001};
  static constexpr double default_weights_decay_{0.01};

  static constexpr double default_beta1_{0.9};
  static constexpr double default_beta2_{0.999};
  static constexpr double default_epsilon_{1e-8};
  static constexpr int default_time_{0};

  double learning_rate_{default_learning_rate_};
  double weights_decay_{default_weights_decay_};

  double beta1_{default_beta1_};
  double beta2_{default_beta2_};
  double epsilon_{default_epsilon_};
  int time_{default_time_};

  AdamWMoments adam_;
};

} // namespace NeuralNetwork

#endif //ADAMWOPTIMIZER_H
