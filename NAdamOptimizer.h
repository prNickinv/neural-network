#ifndef NADAMOPTIMIZER_H
#define NADAMOPTIMIZER_H

#include <iostream>

#include <Eigen/Dense>

#include "GlobalUsings.h"
#include "SchedulerUtils.h"

namespace NeuralNetwork {

struct NAdamMoments {
  Matrix m_w; // First moment for weights
  Matrix v_w; // Second moment for weights
  Vector m_b; // First moment for bias
  Vector v_b; // Second moment for bias
};

class NAdamOptimizer {
 public:
  NAdamOptimizer() = default;
  NAdamOptimizer(double, double, double, double, double, double);
  explicit NAdamOptimizer(std::istream&);

  template<typename SchedulerType>
  NAdamOptimizer(const SchedulerType& scheduler, double weights_decay,
                 double beta1, double beta2, double epsilon,
                 double momentum_decay)
      : learning_rate_(scheduler),
        weights_decay_(weights_decay),
        beta1_(beta1),
        beta2_(beta2),
        epsilon_(epsilon),
        momentum_decay_(momentum_decay) {}

  void Initialize(Index, Index);
  Parameters UpdateParameters(const Matrix&, const Vector&, const Matrix&,
                              const Vector&, int);

  friend std::ostream& operator<<(std::ostream&, const NAdamOptimizer&);

 private:
  double GetLearningRate();
  [[nodiscard]] Matrix ApplyWeightsDecay(const Matrix&, int, double) const;

  double UpdateU();
  void UpdateMoments(const Matrix&, const Vector&);

  [[nodiscard]] NAdamMoments ComputeCorrectedMoments(double, const Matrix&,
                                                     const Vector&) const;
  [[nodiscard]] Parameters ComputeCorrectedM(double, const Matrix&,
                                             const Vector&) const;
  [[nodiscard]] Parameters ComputeCorrectedV() const;

  [[nodiscard]] Matrix ComputeNewWeights(const Matrix&, const Matrix&,
                                         const Matrix&, int, double) const;
  [[nodiscard]] Vector ComputeNewBias(const Vector&, const Vector&,
                                      const Vector&, int, double) const;

  static constexpr double default_learning_rate_{0.001};
  static constexpr double default_weights_decay_{0.01};

  static constexpr double default_beta1_{0.9};
  static constexpr double default_beta2_{0.999};
  static constexpr double default_epsilon_{1e-8};
  static constexpr double default_momentum_decay_{0.004};

  static constexpr int default_time_{0};
  static constexpr double default_u_{0.0};
  static constexpr double default_u_product_{1.0};

  static constexpr double u_factor_coef_{0.5};
  static constexpr double u_factor_base_{0.96};

  Scheduler learning_rate_{default_learning_rate_};
  double weights_decay_{default_weights_decay_};

  double beta1_{default_beta1_};
  double beta2_{default_beta2_};
  double epsilon_{default_epsilon_};
  double momentum_decay_{default_momentum_decay_};

  int time_{default_time_};
  double u_{default_u_};
  double u_product_{default_u_product_};

  NAdamMoments nadam_;
};

} // namespace NeuralNetwork

#endif //NADAMOPTIMIZER_H
