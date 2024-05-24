#ifndef LAYER_H
#define LAYER_H

#include <iostream>
#include <string>

#include <Eigen/Dense>
#include <EigenRand/EigenRand>

#include "ActivationFunction.h"
#include "AdamWOptimizer.h"

namespace NeuralNetwork {

enum class Optimizer {
    MiniBatchGD,
    AdamW
};

class Layer {
  using Vector = Eigen::VectorXd;
  using RowVector = Eigen::RowVectorXd;
  using Matrix = Eigen::MatrixXd;
  using Index = Eigen::Index;
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
  void UpdateParameters(int, double, double);

  void SetOptimizer(Optimizer);
  friend std::ostream& operator<<(std::ostream&, const Layer&);
  friend std::istream& operator>>(std::istream&, Layer&);

 private:
  [[nodiscard]] Vector ApplyParameters(const Vector&) const;
  //TODO: Remove [[nodiscard]]?
  [[nodiscard]] Vector ApplyActivation() const;

  [[nodiscard]] RowVector ComputeNextBackpropVector(const RowVector&,
                                                    const Matrix&) const;
  void UpdateGradients(const Matrix&, const RowVector&);

  void ApplyWeightsDecay(int, double, double);
  void UpdateWeightsAdamW(int, double, const Matrix&, const Matrix&, double);
  void UpdateBiasAdamW(int, double, const Vector&, const Vector&, double);

  void UpdateParametersAdamW(int, double, double);
  void UpdateParametersMiniBatchGD(int, double, double);

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

  Optimizer optimizer_{Optimizer::AdamW};
  AdamWOptimizer adam_w_opt_;
};

} // namespace NeuralNetwork

#endif //LAYER_H
