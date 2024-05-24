#include "AdamWOptimizer.h"

#include <cmath>
#include <iomanip>
#include <limits>

namespace NeuralNetwork {

AdamWOptimizer::AdamWOptimizer(Index in_dim, Index out_dim)
    : adam_{Matrix::Zero(out_dim, in_dim), Matrix::Zero(out_dim, in_dim),
            Vector::Zero(out_dim), Vector::Zero(out_dim)} {}

AdamWOptimizer::AdamWOptimizer(std::istream& is) {
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

  return {adam_.m_w / beta1_t,
          adam_.v_w / beta2_t,
          adam_.m_b / beta1_t,
          adam_.v_b / beta2_t};
}

double AdamWOptimizer::GetEpsilon() const {
  return epsilon_;
}

std::ostream& operator<<(std::ostream& os, const AdamWOptimizer& adam) {
  os << std::fixed << std::setprecision(std::numeric_limits<double>::digits10);

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
