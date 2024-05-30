#ifndef LAYER_H
#define LAYER_H

#include <iostream>
#include <string>
#include <variant>

#include <Eigen/Dense>
#include <EigenRand/EigenRand>

#include "ActivationFunction.h"
#include "AdamWOptimizer.h"
#include "GlobalUsings.h"
#include "MiniBatchGD.h"
#include "MomentumOptimizer.h"

namespace NeuralNetwork {

class Layer {
  using Optimizer = std::variant<AdamWOptimizer,
                                 MomentumOptimizer, MiniBatchGD>;
  using RandomGenerator = Eigen::Rand::P8_mt19937_64;

 public:
  Layer() = default;
  Layer(Index, Index, const ActivationFunction&);
  Layer(Index, Index, ActivationFunction&&);
  explicit Layer(std::istream&,
                 const ActivationFunction& = ActivationFunction());

  [[nodiscard]] Vector PushForward(const Vector&);
  [[nodiscard]] RowVector PropagateBack(const RowVector&);
  RowVector PropagateBackSoftMaxCE(const Vector&);
  void UpdateParameters(int);

  template<typename OptimizerType>
  void SetOptimizer(const OptimizerType& optimizer) {
    optimizer_ = optimizer;
    std::get<OptimizerType>(optimizer_)
        .Initialize(weights_.rows(), weights_.cols());
  }

  void SetOptimizer(const MiniBatchGD&);

  friend std::ostream& operator<<(std::ostream&, const Layer&);
  friend std::istream& operator>>(std::istream&, Layer&);

 private:
  [[nodiscard]] Vector ApplyParameters(const Vector&) const;
  //TODO: Remove [[nodiscard]]?
  [[nodiscard]] Vector ApplyActivation() const;

  [[nodiscard]] RowVector ComputeNextBackpropVector(const RowVector&,
                                                    const Matrix&) const;
  void UpdateGradients(const Matrix&, const RowVector&);

  std::string GetOptimizerType() const;

  static constexpr int random_seed_{42};
  static RandomGenerator generator_;

  Vector input_vector_;
  Vector pre_activated_vector_;
  Matrix weights_;
  Vector bias_;
  ActivationFunction activation_function_;

  Matrix weights_gradient_;
  Vector bias_gradient_;

  Optimizer optimizer_;
};

} // namespace NeuralNetwork

#endif //LAYER_H
