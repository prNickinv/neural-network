#include "NAdamOptimizer.h"

#include <cmath>

namespace NeuralNetwork {

NAdamOptimizer::NAdamOptimizer(double learning_rate, double weights_decay,
                               double beta1, double beta2, double epsilon,
                               double momentum_decay)
    : learning_rate_(learning_rate),
      weights_decay_(weights_decay),
      beta1_(beta1),
      beta2_(beta2),
      epsilon_(epsilon),
      momentum_decay_(momentum_decay) {}

NAdamOptimizer::NAdamOptimizer(std::istream& is) {
  learning_rate_ = SchedulerUtils::GetScheduler(is);
  is >> weights_decay_;

  is >> beta1_;
  is >> beta2_;
  is >> epsilon_;
  is >> momentum_decay_;

  is >> time_;
  is >> u_product_;

  Index rows, cols;
  is >> rows;
  is >> cols;

  nadam_ = {Matrix::Zero(rows, cols), Matrix::Zero(rows, cols),
            Vector::Zero(rows), Vector::Zero(rows)};
  for (Index i = 0; i < rows; ++i) {
    for (Index j = 0; j < cols; ++j) {
      is >> nadam_.m_w(i, j);
    }
  }

  is >> rows;
  is >> cols;
  for (Index i = 0; i < rows; ++i) {
    for (Index j = 0; j < cols; ++j) {
      is >> nadam_.v_w(i, j);
    }
  }

  Index size;
  is >> size;
  for (Index i = 0; i < size; ++i) {
    is >> nadam_.m_b(i);
  }

  is >> size;
  for (Index i = 0; i < size; ++i) {
    is >> nadam_.v_b(i);
  }
}

void NAdamOptimizer::Initialize(Index rows, Index cols) {
  nadam_ = {Matrix::Zero(rows, cols), Matrix::Zero(rows, cols),
            Vector::Zero(rows), Vector::Zero(rows)};
}

Parameters NAdamOptimizer::UpdateParameters(
    const NeuralNetwork::Matrix& weights, const NeuralNetwork::Vector& bias,
    const NeuralNetwork::Matrix& weights_gradient,
    const NeuralNetwork::Vector& bias_gradient, int batch_size) {
  double learning_rate = GetLearningRate();

  Parameters parameters{weights, bias};
  parameters.weights = ApplyWeightsDecay(weights, batch_size, learning_rate);

  double u_next = UpdateU();
  UpdateMoments(weights_gradient, bias_gradient);
  NAdamMoments corrected_moments =
      ComputeCorrectedMoments(u_next, weights_gradient, bias_gradient);

  parameters.weights =
      ComputeNewWeights(parameters.weights, corrected_moments.m_w,
                        corrected_moments.v_w, batch_size, learning_rate);
  parameters.bias =
      ComputeNewBias(parameters.bias, corrected_moments.m_b,
                     corrected_moments.v_b, batch_size, learning_rate);
  return parameters;
}

std::ostream& operator<<(std::ostream& os, const NAdamOptimizer& nadam) {
  os << "NAdam" << std::endl;

  if (std::holds_alternative<double>(nadam.learning_rate_)) {
    os << "ConstantLR" << std::endl;
  }
  auto save_scheduler =
      SchedulerUtils::Overload{[&](double lr) { os << lr << std::endl; },
                               [&](auto& scheduler) {
                                 os << scheduler;
                               }};
  std::visit(save_scheduler, nadam.learning_rate_);

  os << nadam.weights_decay_ << std::endl;

  os << nadam.beta1_ << std::endl;
  os << nadam.beta2_ << std::endl;
  os << nadam.epsilon_ << std::endl;
  os << nadam.momentum_decay_ << std::endl;

  os << nadam.time_ << std::endl;
  os << nadam.u_product_ << std::endl;

  os << nadam.nadam_.m_w.rows() << " " << nadam.nadam_.m_w.cols() << std::endl;
  os << nadam.nadam_.m_w << std::endl;

  os << nadam.nadam_.v_w.rows() << " " << nadam.nadam_.v_w.cols() << std::endl;
  os << nadam.nadam_.v_w << std::endl;

  os << nadam.nadam_.m_b.size() << std::endl;
  os << nadam.nadam_.m_b << std::endl;

  os << nadam.nadam_.v_b.size() << std::endl;
  os << nadam.nadam_.v_b << std::endl;

  return os;
}

double NAdamOptimizer::GetLearningRate() {
  auto get_lr = SchedulerUtils::Overload{[&](double lr) { return lr; },
                                         [&](auto& scheduler) {
                                           return scheduler.GetLearningRate();
                                         }};
  return std::visit(get_lr, learning_rate_);
}

Matrix NAdamOptimizer::ApplyWeightsDecay(const Matrix& weights, int batch_size,
                                         double learning_rate) const {
  return weights - (learning_rate * weights_decay_ / batch_size) * weights;
}

double NAdamOptimizer::UpdateU() {
  ++time_;

  u_ = beta1_
      * (1.0
         - u_factor_coef_ * std::pow(u_factor_base_, time_ * momentum_decay_));
  u_product_ *= u_;

  double u_next = beta1_
      * (1.0
         - u_factor_coef_
             * std::pow(u_factor_base_, (time_ + 1) * momentum_decay_));
  return u_next;
}

void NAdamOptimizer::UpdateMoments(const Matrix& weights_gradient,
                                   const Vector& bias_gradient) {
  nadam_.m_w = beta1_ * nadam_.m_w + (1 - beta1_) * weights_gradient;
  nadam_.v_w = beta2_ * nadam_.v_w
      + (1 - beta2_) * weights_gradient.cwiseProduct(weights_gradient);
  nadam_.m_b = beta1_ * nadam_.m_b + (1 - beta1_) * bias_gradient;
  nadam_.v_b = beta2_ * nadam_.v_b
      + (1 - beta2_) * bias_gradient.cwiseProduct(bias_gradient);
}

NAdamMoments NAdamOptimizer::ComputeCorrectedMoments(
    double u_next, const Matrix& weights_gradient,
    const Vector& bias_gradient) const {

  Parameters corrected_m =
      ComputeCorrectedM(u_next, weights_gradient, bias_gradient);
  Parameters corrected_v = ComputeCorrectedV();

  return {corrected_m.weights, corrected_v.weights, corrected_m.bias,
          corrected_v.bias};
}

Parameters NAdamOptimizer::ComputeCorrectedM(
    double u_next, const Matrix& weights_gradient,
    const Vector& bias_gradient) const {

  double first_coef = u_next / (1.0 - u_product_ * u_next);
  double second_coef = (1.0 - u_) / (1.0 - u_product_);

  Matrix corrected_m_w =
      first_coef * nadam_.m_w + second_coef * weights_gradient;
  Vector corrected_m_b = first_coef * nadam_.m_b + second_coef * bias_gradient;

  return {corrected_m_w, corrected_m_b};
}

Parameters NAdamOptimizer::ComputeCorrectedV() const {
  double beta2_t = 1 - std::pow(beta2_, time_);
  return {nadam_.v_w / beta2_t, nadam_.v_b / beta2_t};
}

Matrix NAdamOptimizer::ComputeNewWeights(const Matrix& weights,
                                         const Matrix& corrected_m_w,
                                         const Matrix& corrected_v_w,
                                         int batch_size,
                                         double learning_rate) const {
  return weights
      - (learning_rate / batch_size)
      * corrected_m_w.cwiseQuotient(
          (corrected_v_w.cwiseSqrt().array() + epsilon_).matrix());
}

Vector NAdamOptimizer::ComputeNewBias(const Vector& bias,
                                      const Vector& corrected_m_b,
                                      const Vector& corrected_v_b,
                                      int batch_size,
                                      double learning_rate) const {
  return bias
      - (learning_rate / batch_size)
      * corrected_m_b.cwiseQuotient(
          (corrected_v_b.cwiseSqrt().array() + epsilon_).matrix());
}

} // namespace NeuralNetwork
