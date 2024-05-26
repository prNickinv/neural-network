#include "AdamWOptimizer.h"

#include <cmath>

namespace NeuralNetwork {

// TODO: Remove this constructor
//AdamWOptimizer::AdamWOptimizer(Index in_dim, Index out_dim)
//    : adam_{Matrix::Zero(out_dim, in_dim), Matrix::Zero(out_dim, in_dim),
//            Vector::Zero(out_dim), Vector::Zero(out_dim)} {}

AdamWOptimizer::AdamWOptimizer(double learning_rate, double weights_decay,
                               double beta1, double beta2, double epsilon)
    : learning_rate_{learning_rate},
      weights_decay_{weights_decay},
      beta1_{beta1},
      beta2_{beta2},
      epsilon_{epsilon} {}

AdamWOptimizer::AdamWOptimizer(std::istream& is) {
  is >> learning_rate_;
  is >> weights_decay_;

  is >> beta1_;
  is >> beta2_;
  is >> epsilon_;
  is >> time_;

  Index rows, cols;
  is >> rows;
  is >> cols;

  adam_ = {Matrix::Zero(rows, cols), Matrix::Zero(rows, cols),
           Vector::Zero(rows), Vector::Zero(rows)};
  for (Index i = 0; i < rows; ++i) {
    for (Index j = 0; j < cols; ++j) {
      is >> adam_.m_w(i, j);
    }
  }

  is >> rows;
  is >> cols;
  for (Index i = 0; i < rows; ++i) {
    for (Index j = 0; j < cols; ++j) {
      is >> adam_.v_w(i, j);
    }
  }

  Index size;
  is >> size;
  for (Index i = 0; i < size; ++i) {
    is >> adam_.m_b(i);
  }

  is >> size;
  for (Index i = 0; i < size; ++i) {
    is >> adam_.v_b(i);
  }
}

void AdamWOptimizer::Initialize(Index rows, Index cols) {
  adam_ = {Matrix::Zero(rows, cols), Matrix::Zero(rows, cols),
           Vector::Zero(rows), Vector::Zero(rows)};
}

Matrix AdamWOptimizer::ApplyWeightsDecay(const Matrix& weights, int batch_size) const {
  return weights - (learning_rate_ * weights_decay_ / batch_size) * weights;
}

void AdamWOptimizer::UpdateMoments(const Matrix& weights_gradient,
                                   const Vector& bias_gradient) {
  ++time_;

  adam_.m_w = beta1_ * adam_.m_w + (1 - beta1_) * weights_gradient;
  adam_.v_w = beta2_ * adam_.v_w
      + (1 - beta2_) * weights_gradient.cwiseProduct(weights_gradient);
  adam_.m_b = beta1_ * adam_.m_b + (1 - beta1_) * bias_gradient;
  adam_.v_b = beta2_ * adam_.v_b
      + (1 - beta2_) * bias_gradient.cwiseProduct(bias_gradient);
}

AdamWMoments AdamWOptimizer::ComputeCorrectedMoments() const {
  double beta1_t = 1 - std::pow(beta1_, time_);
  double beta2_t = 1 - std::pow(beta2_, time_);

  return {adam_.m_w / beta1_t, adam_.v_w / beta2_t, adam_.m_b / beta1_t,
          adam_.v_b / beta2_t};
}

double AdamWOptimizer::GetEpsilon() const {
  return epsilon_;
}

Matrix AdamWOptimizer::ComputeNewWeights(const Matrix& weights,
                                         const Matrix& m_hat_w,
                                         const Matrix& v_hat_w,
                                         int batch_size) const {
  return weights - (learning_rate_ / batch_size)
      * m_hat_w.cwiseQuotient((v_hat_w.cwiseSqrt().array() + epsilon_).matrix());
}

Vector AdamWOptimizer::ComputeNewBias(const Vector& bias, const Vector& m_hat_b,
                                      const Vector& v_hat_b, int batch_size) const {
  return bias - (learning_rate_ / batch_size)
      * m_hat_b.cwiseQuotient((v_hat_b.cwiseSqrt().array() + epsilon_).matrix());
}

Parameters AdamWOptimizer::UpdateParameters(const Matrix& weights,
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

  Parameters parameters{weights, bias};
  parameters.weights = ApplyWeightsDecay(weights, batch_size);

  UpdateMoments(weights_gradient, bias_gradient);
  AdamWMoments corrected_moments = ComputeCorrectedMoments();

  parameters.weights = ComputeNewWeights(parameters.weights, corrected_moments.m_w,
                                         corrected_moments.v_w, batch_size);
  parameters.bias = ComputeNewBias(parameters.bias, corrected_moments.m_b,
                                   corrected_moments.v_b, batch_size);

//  parameters.weights -= (learning_rate_ / batch_size)
//      * corrected_moments.m_w.cwiseQuotient(
//          (corrected_moments.v_w.cwiseSqrt().array() + epsilon_).matrix());
//  parameters.bias -= (learning_rate_ / batch_size)
//      * corrected_moments.m_b.cwiseQuotient(
//          (corrected_moments.v_b.cwiseSqrt().array() + epsilon_).matrix());

  return parameters;
}

std::ostream& operator<<(std::ostream& os, const AdamWOptimizer& adam) {
  //TODO: Place rows and cols in the same line
  os << adam.learning_rate_ << std::endl;
  os << adam.weights_decay_ << std::endl;

  os << adam.beta1_ << std::endl;
  os << adam.beta2_ << std::endl;
  os << adam.epsilon_ << std::endl;
  os << adam.time_ << std::endl;

  os << adam.adam_.m_w.rows() << std::endl;
  os << adam.adam_.m_w.cols() << std::endl;
  os << adam.adam_.m_w << std::endl;

  os << adam.adam_.v_w.rows() << std::endl;
  os << adam.adam_.v_w.cols() << std::endl;
  os << adam.adam_.v_w << std::endl;

  os << adam.adam_.m_b.size() << std::endl;
  os << adam.adam_.m_b << std::endl;

  os << adam.adam_.v_b.size() << std::endl;
  os << adam.adam_.v_b << std::endl;
  return os;
}

std::istream& operator>>(std::istream& is, AdamWOptimizer& adam) {
  is >> adam.learning_rate_;
  is >> adam.weights_decay_;

  is >> adam.beta1_;
  is >> adam.beta2_;
  is >> adam.epsilon_;
  is >> adam.time_;

  Index rows, cols;
  is >> rows;
  is >> cols;
  adam.adam_.m_w = Matrix::Zero(rows, cols);
  for (Index i = 0; i < adam.adam_.m_w.rows(); ++i) {
    for (Index j = 0; j < adam.adam_.m_w.cols(); ++j) {
      is >> adam.adam_.m_w(i, j);
    }
  }

  is >> rows;
  is >> cols;
  adam.adam_.v_w = Matrix::Zero(rows, cols);
  for (Index i = 0; i < adam.adam_.v_w.rows(); ++i) {
    for (Index j = 0; j < adam.adam_.v_w.cols(); ++j) {
      is >> adam.adam_.v_w(i, j);
    }
  }

  Index size;
  is >> size;
  adam.adam_.m_b = Vector::Zero(size);
  for (Index i = 0; i < adam.adam_.m_b.size(); ++i) {
    is >> adam.adam_.m_b(i);
    std::cout << adam.adam_.m_b(i);
  }

  is >> size;
  adam.adam_.v_b = Vector::Zero(size);
  for (Index i = 0; i < adam.adam_.v_b.size(); ++i) {
    is >> adam.adam_.v_b(i);
  }
  return is;
}

} // namespace NeuralNetwork
