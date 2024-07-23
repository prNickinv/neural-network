#ifndef ADAMWOPTIMIZER_H
#define ADAMWOPTIMIZER_H

#include <iostream>

#include <Eigen/Dense>

#include "GlobalUsings.h"
#include "SchedulerUtils.h"

namespace NeuralNetwork {

struct AdamWMoments {
  Matrix m_w; // First moment for weights
  Matrix v_w; // Second moment for weights
  Vector m_b; // First moment for bias
  Vector v_b; // Second moment for bias
};

class AdamWOptimizer {
 public:
  AdamWOptimizer() = default;
  AdamWOptimizer(double, double, double, double, double);
  explicit AdamWOptimizer(std::istream&);

  template<typename SchedulerType>
  AdamWOptimizer(const SchedulerType& scheduler, double weights_decay,
                 double beta1, double beta2, double epsilon)
      : learning_rate_(scheduler),
        weights_decay_(weights_decay),
        beta1_(beta1),
        beta2_(beta2),
        epsilon_(epsilon) {}

  void Initialize(Index, Index);
  Parameters UpdateParameters(const Matrix&, const Vector&, const Matrix&,
                              const Vector&, int);

  friend std::ostream& operator<<(std::ostream&, const AdamWOptimizer&);

 private:
  double GetLearningRate();
  [[nodiscard]] Matrix ApplyWeightsDecay(const Matrix&, int, double) const;
  void UpdateMoments(const Matrix&, const Vector&);
  [[nodiscard]] AdamWMoments ComputeCorrectedMoments() const;

  [[nodiscard]] Matrix ComputeNewWeights(const Matrix&, const Matrix&,
                                         const Matrix&, int, double) const;
  [[nodiscard]] Vector ComputeNewBias(const Vector&, const Vector&,
                                      const Vector&, int, double) const;

  static constexpr double default_learning_rate_{0.001};
  static constexpr double default_weights_decay_{0.01};

  static constexpr double default_beta1_{0.9};
  static constexpr double default_beta2_{0.999};
  static constexpr double default_epsilon_{1e-8};
  static constexpr int default_time_{0};

  Scheduler learning_rate_{default_learning_rate_};
  double weights_decay_{default_weights_decay_};

  double beta1_{default_beta1_};
  double beta2_{default_beta2_};
  double epsilon_{default_epsilon_};
  int time_{default_time_};

  AdamWMoments adam_;
};

} // namespace NeuralNetwork

#endif //ADAMWOPTIMIZER_H
