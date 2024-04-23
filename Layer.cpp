#include "Layer.h"

#include <cassert>
#include <utility>

namespace NeuralNetwork {

Layer::Layer(Index in_dim, Index out_dim,
             const ActivationFunction& activation_function)
    : input_vector_(Vector::Zero(in_dim)),
      pre_activated_vector_(Vector::Zero(out_dim)),
      weights_(Eigen::Rand::normal<Matrix>(out_dim, in_dim, generator_)),
      bias_(Eigen::Rand::normal<Vector>(out_dim, 1, generator_)),
      activation_function_(activation_function),
      weights_gradient_(Matrix::Zero(out_dim, in_dim)),
      bias_gradient_(Vector::Zero(out_dim)) {}

Layer::Layer(Index in_dim, Index out_dim,
             ActivationFunction&& activation_function)
    : input_vector_(Vector::Zero(in_dim)),
      pre_activated_vector_(Vector::Zero(out_dim)),
      weights_(Eigen::Rand::normal<Matrix>(out_dim, in_dim, generator_)),
      bias_(Eigen::Rand::normal<Vector>(out_dim, 1, generator_)),
      activation_function_(std::move(activation_function)),
      weights_gradient_(Matrix::Zero(out_dim, in_dim)),
      bias_gradient_(Vector::Zero(out_dim)) {}

Layer::Layer(std::istream& is, const ActivationFunction& activation_function) {
  Index weights_rows, weights_cols;
  is >> weights_rows >> weights_cols;

  input_vector_ = Vector::Zero(weights_cols);
  pre_activated_vector_ = Vector::Zero(weights_rows);

  weights_ = Matrix::Zero(weights_rows, weights_cols);
  weights_gradient_ = Matrix::Zero(weights_rows, weights_cols);
  for (Index i = 0; i != weights_rows; ++i) {
    for (Index j = 0; j != weights_cols; ++j) {
      is >> weights_(i, j);
    }
  }

  Index bias_size;
  is >> bias_size;
  bias_ = Vector::Zero(bias_size);
  bias_gradient_ = Vector::Zero(bias_size);
  for (Index i = 0; i != bias_size; ++i) {
    is >> bias_(i);
  }

  std::string activation_type;
  is >> activation_type;
  if (activation_function) {
    activation_function_ = activation_function;
  } else {
    activation_function_ = ActivationFunction::GetFunction(activation_type);
  }
}

Layer::Vector Layer::PushForward(const Vector& input_vector) {
  assert(input_vector.size() != 0 && "Input vector cannot be empty");
  input_vector_ = input_vector;
  pre_activated_vector_ = ApplyParameters(input_vector);
  return ApplyActivation();
}

Layer::RowVector Layer::PropagateBack(const RowVector& prev_backprop_vector) {
  assert(prev_backprop_vector.size() != 0
         && "Vector from previous gradient descent step cannot be empty");
  Matrix activation_jacobian =
      activation_function_.ComputeJacobianMatrix(pre_activated_vector_);
  UpdateGradients(activation_jacobian, prev_backprop_vector);
  return ComputeNextBackpropVector(prev_backprop_vector, activation_jacobian);
}

Layer::RowVector Layer::PropagateBackSoftMaxCE(
    const Vector& prev_backprop_vector) {
  bias_gradient_ += prev_backprop_vector;
  weights_gradient_ += prev_backprop_vector * input_vector_.transpose();
  return prev_backprop_vector.transpose() * weights_;
}

void Layer::UpdateParameters(int batch_size, double learning_rate,
                             double weights_decay) {
  weights_ -= (learning_rate / batch_size) * weights_gradient_
      + (2 * weights_decay * learning_rate / batch_size) * weights_;
  bias_ -= (learning_rate / batch_size) * bias_gradient_;
  weights_gradient_.setZero();
  bias_gradient_.setZero();
}

std::ostream& operator<<(std::ostream& os, const Layer& layer) {
  os << layer.weights_.rows() << " " << layer.weights_.cols() << std::endl;
  os << layer.weights_ << std::endl;
  os << layer.bias_.size() << std::endl;
  os << layer.bias_ << std::endl;
  os << layer.activation_function_.GetType() << std::endl;
  return os;
}

std::istream& operator>>(std::istream& is, Layer& layer) {
  Layer::Index weights_rows, weights_cols;
  // TODO: Initialize input_vector_ and pre_activated_vector_ with zeros?
  is >> weights_rows >> weights_cols;
  layer.weights_.resize(weights_rows, weights_cols);
  layer.weights_gradient_ = Layer::Matrix::Zero(weights_rows, weights_cols);
  for (Layer::Index i = 0; i != weights_rows; ++i) {
    for (Layer::Index j = 0; j != weights_cols; ++j) {
      is >> layer.weights_(i, j);
    }
  }

  Layer::Index bias_size;
  is >> bias_size;
  layer.bias_.resize(bias_size);
  layer.bias_gradient_ = Layer::Vector::Zero(bias_size);
  for (Layer::Index i = 0; i != bias_size; ++i) {
    is >> layer.bias_(i);
  }

  std::string activation_type;
  is >> activation_type;
  layer.activation_function_ = ActivationFunction::GetFunction(activation_type);
  return is;
}

Layer::Vector Layer::ApplyParameters(const Vector& input_vector) const {
  assert(input_vector.size() != 0);
  return weights_ * input_vector + bias_;
}

Layer::Vector Layer::ApplyActivation() const {
  assert(pre_activated_vector_.size() != 0);
  return activation_function_.Activate(pre_activated_vector_);
}

Layer::RowVector Layer::ComputeNextBackpropVector(
    const RowVector& prev_backprop_vector,
    const Matrix& activation_jacobian) const {
  return prev_backprop_vector * activation_jacobian * weights_;
}

void Layer::UpdateGradients(const Matrix& activation_jacobian,
                            const RowVector& prev_backprop_vector) {
  Vector transit_vector =
      activation_jacobian * prev_backprop_vector.transpose();
  bias_gradient_ += transit_vector;
  weights_gradient_ += transit_vector * input_vector_.transpose();
}

Layer::RandomGenerator Layer::generator_{random_seed_};

} // namespace NeuralNetwork
