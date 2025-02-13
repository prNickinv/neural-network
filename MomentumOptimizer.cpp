#include "MomentumOptimizer.h"

namespace NeuralNetwork {

MomentumOptimizer::MomentumOptimizer(double learning_rate, double weights_decay,
                                     double gamma, Nesterov nesterov)
    : learning_rate_(learning_rate),
      weights_decay_(weights_decay),
      gamma_(gamma),
      nesterov_(nesterov) {}

MomentumOptimizer::MomentumOptimizer(std::istream& is) {
  learning_rate_ = SchedulerUtils::GetScheduler(is);
  is >> weights_decay_;
  is >> gamma_;

  std::string nesterov;
  is >> nesterov;
  if (nesterov == "nesterov_enable") {
    nesterov_ = Nesterov::Enable;
  } else {
    nesterov_ = Nesterov::Disable;
  }

  Index rows, cols;
  is >> rows;
  is >> cols;

  v_w_ = Matrix::Zero(rows, cols);
  for (Index i = 0; i < rows; ++i) {
    for (Index j = 0; j < cols; ++j) {
      is >> v_w_(i, j);
    }
  }

  Index size;
  is >> size;
  v_b_ = Vector::Zero(size);
  for (Index i = 0; i < size; ++i) {
    is >> v_b_(i);
  }
}

void MomentumOptimizer::Initialize(Index rows, Index cols) {
  v_w_ = Matrix::Zero(rows, cols);
  v_b_ = Vector::Zero(rows);
}

Parameters MomentumOptimizer::UpdateParameters(const Matrix& weights,
                                               const Vector& bias,
                                               const Matrix& weights_gradient,
                                               const Vector& bias_gradient,
                                               int batch_size) {
  assert(weights.rows() == v_w_.rows() && weights.cols() == v_w_.cols()
         && "Weights and v_w_ have different dimensions");
  assert(bias.size() == v_b_.size()
         && "Bias and v_b_ have different dimensions");
  assert(weights.rows() == weights_gradient.rows()
         && weights.cols() == weights_gradient.cols()
         && "Weights and weights_gradient have different dimensions");
  assert(bias.size() == bias_gradient.size()
         && "Bias and bias_gradient have different dimensions");

  double learning_rate = GetLearningRate();

  Parameters parameters{weights, bias};
  parameters.weights = ApplyWeightsDecay(weights, batch_size, learning_rate);
  UpdateVelocity(weights_gradient, bias_gradient, learning_rate);

  if (nesterov_ == Nesterov::Enable) {
    parameters.weights += ComputeNesterovUpdateWeights(
        weights_gradient, batch_size, learning_rate);
    parameters.bias +=
        ComputeNesterovUpdateBias(bias_gradient, batch_size, learning_rate);
    return parameters;
  }

  // Nesterov::Disable
  parameters.weights += v_w_ / batch_size;
  parameters.bias += v_b_ / batch_size;
  return parameters;
}

std::ostream& operator<<(std::ostream& os, const MomentumOptimizer& momentum) {
  os << "Momentum" << std::endl;

  auto save_scheduler = SchedulerUtils::Overload{[&](double lr) {
                                                   os << "ConstantLR"
                                                      << std::endl;
                                                   os << lr << std::endl;
                                                 },
                                                 [&](auto& scheduler) {
                                                   os << scheduler;
                                                 }};
  std::visit(save_scheduler, momentum.learning_rate_);

  os << momentum.weights_decay_ << std::endl;
  os << momentum.gamma_ << std::endl;

  if (momentum.nesterov_ == Nesterov::Enable) {
    os << "nesterov_enable" << std::endl;
  } else {
    os << "nesterov_disable" << std::endl;
  }

  os << momentum.v_w_.rows() << " " << momentum.v_w_.cols() << std::endl;
  os << momentum.v_w_ << std::endl;
  os << momentum.v_b_.size() << std::endl;
  os << momentum.v_b_ << std::endl;
  return os;
}

double MomentumOptimizer::GetLearningRate() {
  auto get_lr = SchedulerUtils::Overload{[&](double lr) { return lr; },
                                         [&](auto& scheduler) {
                                           return scheduler.GetLearningRate();
                                         }};
  return std::visit(get_lr, learning_rate_);
}

Matrix MomentumOptimizer::ApplyWeightsDecay(const Matrix& weights,
                                            int batch_size,
                                            double learning_rate) const {
  return weights - (learning_rate * weights_decay_ / batch_size) * weights;
}

void MomentumOptimizer::UpdateVelocity(const Matrix& weights_gradient,
                                       const Vector& bias_gradient,
                                       double learning_rate) {
  v_w_ = gamma_ * v_w_ - learning_rate * weights_gradient;
  v_b_ = gamma_ * v_b_ - learning_rate * bias_gradient;
}

Matrix MomentumOptimizer::ComputeNesterovUpdateWeights(
    const Matrix& weights_gradient, int batch_size, double learning_rate) {
  return (gamma_ / batch_size) * v_w_
      - (learning_rate / batch_size) * weights_gradient;
}

Vector MomentumOptimizer::ComputeNesterovUpdateBias(const Vector& bias_gradient,
                                                    int batch_size,
                                                    double learning_rate) {
  return (gamma_ / batch_size) * v_b_
      - (learning_rate / batch_size) * bias_gradient;
}

} // namespace NeuralNetwork
