#include "MomentumOptimizer.h"

namespace NeuralNetwork {

//TODO: Should initialize 1x1 v_w_ and v_b_ with zeros here?
MomentumOptimizer::MomentumOptimizer(double gamma) : gamma_{gamma} {}

MomentumOptimizer::MomentumOptimizer(std::istream& is) {
  is >> gamma_;

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

void MomentumOptimizer::UpdateVelocity(const Matrix& weights_gradient,
                                       const Vector& bias_gradient,
                                       double learning_rate) {
  v_w_ = gamma_ * v_w_ + learning_rate * weights_gradient;
  v_b_ = gamma_ * v_b_ + learning_rate * bias_gradient;
}

Matrix MomentumOptimizer::GetVelocityWeights() const {
  return v_w_;
}

Vector MomentumOptimizer::GetVelocityBias() const {
  return v_b_;
}

std::ostream& operator<<(std::ostream& os, const MomentumOptimizer& optimizer) {
  os << optimizer.gamma_ << std::endl;
  os << optimizer.v_w_.rows() << " " << optimizer.v_w_.cols() << std::endl;
  os << optimizer.v_w_ << std::endl;
  os << optimizer.v_b_.size() << std::endl;
  os << optimizer.v_b_ << std::endl;

  return os;
}

} // namespace NeuralNetwork
