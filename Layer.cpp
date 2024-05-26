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
      bias_gradient_(Vector::Zero(out_dim)),
      optimizer_(MiniBatchGD()) {}

Layer::Layer(Index in_dim, Index out_dim,
             ActivationFunction&& activation_function)
    : input_vector_(Vector::Zero(in_dim)),
      pre_activated_vector_(Vector::Zero(out_dim)),
      weights_(Eigen::Rand::normal<Matrix>(out_dim, in_dim, generator_)),
      bias_(Eigen::Rand::normal<Vector>(out_dim, 1, generator_)),
      activation_function_(std::move(activation_function)),
      weights_gradient_(Matrix::Zero(out_dim, in_dim)),
      bias_gradient_(Vector::Zero(out_dim)),
      optimizer_(MiniBatchGD()) {}

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

  std::string optimizer_type;
  is >> optimizer_type;
  if (optimizer_type == "AdamW") {
    optimizer_ = AdamWOptimizer(is);
  } else if (optimizer_type == "Momentum") {
    optimizer_ = MomentumOptimizer(is);
  } else {
    optimizer_ = MiniBatchGD(is);
  }
}

Vector Layer::PushForward(const Vector& input_vector) {
  assert(input_vector.size() != 0 && "Input vector cannot be empty");
  input_vector_ = input_vector;
  pre_activated_vector_ = ApplyParameters(input_vector);
  return ApplyActivation();
}

RowVector Layer::PropagateBack(const RowVector& prev_backprop_vector) {
  assert(prev_backprop_vector.size() != 0
         && "Vector from previous gradient descent step cannot be empty");
  Matrix activation_jacobian =
      activation_function_.ComputeJacobianMatrix(pre_activated_vector_);
  UpdateGradients(activation_jacobian, prev_backprop_vector);
  return ComputeNextBackpropVector(prev_backprop_vector, activation_jacobian);
}

RowVector Layer::PropagateBackSoftMaxCE(const Vector& prev_backprop_vector) {
  bias_gradient_ += prev_backprop_vector;
  weights_gradient_ += prev_backprop_vector * input_vector_.transpose();
  return prev_backprop_vector.transpose() * weights_;
}

void Layer::UpdateParameters(int batch_size) {
  Parameters new_parameters;
  if (std::holds_alternative<AdamWOptimizer>(optimizer_)) {
    new_parameters = std::get<AdamWOptimizer>(optimizer_)
                         .UpdateParameters(weights_, bias_, weights_gradient_,
                                           bias_gradient_, batch_size);
    //UpdateParametersAdamW(batch_size, learning_rate, weights_decay);
  } else if (std::holds_alternative<MomentumOptimizer>(optimizer_)) {
    new_parameters = std::get<MomentumOptimizer>(optimizer_)
                         .UpdateParameters(weights_, bias_, weights_gradient_,
                                           bias_gradient_, batch_size);
    //UpdateParametersMomentum(batch_size, learning_rate, weights_decay);
  } else {
    new_parameters = std::get<MiniBatchGD>(optimizer_)
                         .UpdateParameters(weights_, bias_, weights_gradient_,
                                           bias_gradient_, batch_size);
//    UpdateParametersMiniBatchGD(batch_size, learning_rate, weights_decay);
//    weights_gradient_.setZero();
//    bias_gradient_.setZero();
//    return;
  }

  weights_ = new_parameters.weights;
  bias_ = new_parameters.bias;
  weights_gradient_.setZero();
  bias_gradient_.setZero();
}

void Layer::SetOptimizer(const MiniBatchGD& mbgd_optimizer) {
  optimizer_ = mbgd_optimizer;
}

std::ostream& operator<<(std::ostream& os, const Layer& layer) {
  os << layer.weights_.rows() << " " << layer.weights_.cols() << std::endl;
  os << layer.weights_ << std::endl;
  os << layer.bias_.size() << std::endl;
  os << layer.bias_ << std::endl;
  os << layer.activation_function_.GetType() << std::endl;
  os << layer.GetOptimizerType() << std::endl;

  if (std::holds_alternative<AdamWOptimizer>(layer.optimizer_)) {
    os << std::get<AdamWOptimizer>(layer.optimizer_);
  } else if (std::holds_alternative<MomentumOptimizer>(layer.optimizer_)) {
    os << std::get<MomentumOptimizer>(layer.optimizer_);
  } else {
    os << std::get<MiniBatchGD>(layer.optimizer_);
  }

  return os;
}

// TODO: This is not adjusted to work with optimizers
// TODO: Remove?
std::istream& operator>>(std::istream& is, Layer& layer) {
  Index weights_rows, weights_cols;
  // TODO: Initialize input_vector_ and pre_activated_vector_ with zeros?
  is >> weights_rows >> weights_cols;
  layer.weights_.resize(weights_rows, weights_cols);
  layer.weights_gradient_ = Matrix::Zero(weights_rows, weights_cols);
  for (Index i = 0; i != weights_rows; ++i) {
    for (Index j = 0; j != weights_cols; ++j) {
      is >> layer.weights_(i, j);
    }
  }

  Index bias_size;
  is >> bias_size;
  layer.bias_.resize(bias_size);
  layer.bias_gradient_ = Vector::Zero(bias_size);
  for (Index i = 0; i != bias_size; ++i) {
    is >> layer.bias_(i);
  }

  std::string activation_type;
  is >> activation_type;
  layer.activation_function_ = ActivationFunction::GetFunction(activation_type);
  return is;
}

Vector Layer::ApplyParameters(const Vector& input_vector) const {
  assert(input_vector.size() != 0);
  return weights_ * input_vector + bias_;
}

Vector Layer::ApplyActivation() const {
  assert(pre_activated_vector_.size() != 0);
  return activation_function_.Activate(pre_activated_vector_);
}

RowVector Layer::ComputeNextBackpropVector(
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

//void Layer::ApplyWeightsDecay(int batch_size, double learning_rate,
//                              double weights_decay) {
//  weights_ -= (weights_decay * learning_rate / batch_size) * weights_;
//}
//
//void Layer::UpdateWeightsAdamW(int batch_size, double learning_rate,
//                               const Matrix& m_hat_w, const Matrix& v_hat_w,
//                               double eps) {
//  weights_ -= (learning_rate / batch_size)
//      * m_hat_w.cwiseQuotient((v_hat_w.cwiseSqrt().array() + eps).matrix());
//}
//
//void Layer::UpdateBiasAdamW(int batch_size, double learning_rate,
//                            const Vector& m_hat_b, const Vector& v_hat_b,
//                            double eps) {
//  bias_ -= (learning_rate / batch_size)
//      * m_hat_b.cwiseQuotient((v_hat_b.cwiseSqrt().array() + eps).matrix());
//}

//void Layer::UpdateParametersAdamW(int batch_size, double learning_rate,
//                                  double weights_decay) {
//
//  Parameters params;
//  params = std::get<AdamWOptimizer>(optimizer_)
//               .UpdateParameters(weights_, bias_, weights_gradient_,
//                                 bias_gradient_, batch_size);
//  weights_ = params.weights;
//  bias_ = params.bias;
////  ApplyWeightsDecay(batch_size, learning_rate, weights_decay);
////  // TODO: CHECK
////  auto& adam_w_opt = std::get<AdamWOptimizer>(optimizer_);
////  adam_w_opt.UpdateMoments(weights_gradient_, bias_gradient_);
////  AdamWMoments corrected_moments = adam_w_opt.ComputeCorrectedMoments();
////  UpdateWeightsAdamW(batch_size, learning_rate, corrected_moments.m_w,
////                     corrected_moments.v_w, adam_w_opt.GetEpsilon());
////  UpdateBiasAdamW(batch_size, learning_rate, corrected_moments.m_b,
////                  corrected_moments.v_b, adam_w_opt.GetEpsilon());
//}

//void Layer::UpdateParametersMomentum(int batch_size, double learning_rate,
//                                     double weights_decay) {
//  //TODO: Order of update and velocity calculation
//  Parameters params;
//  params = std::get<MomentumOptimizer>(optimizer_)
//                          .UpdateParameters(weights_, bias_, weights_gradient_,
//                                            bias_gradient_, batch_size);
//
//  weights_ = params.weights;
//  bias_ = params.bias;
////  ApplyWeightsDecay(batch_size, learning_rate, weights_decay);
////  auto& momentum_opt = std::get<MomentumOptimizer>(optimizer_);
////  momentum_opt.UpdateVelocity(weights_gradient_, bias_gradient_, learning_rate);
////  weights_ -= (1.0 / batch_size) * momentum_opt.GetVelocityWeights();
////  bias_ -= (1.0 / batch_size) * momentum_opt.GetVelocityBias();
//}
//
//void Layer::UpdateParametersMiniBatchGD(int batch_size, double learning_rate,
//                                        double weights_decay) {
//  ApplyWeightsDecay(batch_size, learning_rate, weights_decay);
//  weights_ -= (learning_rate / batch_size) * weights_gradient_;
//  bias_ -= (learning_rate / batch_size) * bias_gradient_;
//}

std::string Layer::GetOptimizerType() const {
  if (std::holds_alternative<AdamWOptimizer>(optimizer_)) {
    return "AdamW";
  } else if (std::holds_alternative<MomentumOptimizer>(optimizer_)) {
    return "Momentum";
  } else {
    return "MiniBatchGD";
  }
}

Layer::RandomGenerator Layer::generator_{random_seed_};

} // namespace NeuralNetwork
