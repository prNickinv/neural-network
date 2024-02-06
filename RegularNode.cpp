#include "RegularNode.h"

namespace NeuralNetwork {

RegularNode::RegularNode() = default;

RegularNode::RegularNode(RegularNode::SizeType in_dim,
                         RegularNode::SizeType out_dim)
    : weights_(WeightsStructure::NullaryExpr(
          out_dim, in_dim, []() { return NormalRandom()(); })),
      bias_(BiasStructure::NullaryExpr(out_dim,
                                       []() { return NormalRandom()(); })),
      gradient_weights_(WeightsStructure::Zero(out_dim, in_dim)),
      gradient_bias_(BiasStructure::Zero(out_dim)) {}

RegularNode::SampleVector RegularNode::ApplyLinearModification(
    const RegularNode::SampleVector& input_vector) {
  return weights_ * input_vector + bias_;
}

RegularNode::BackPropVector RegularNode::ComputeVectorForBackProp(
    const RegularNode::BackPropVector& back_prop_vector,
    const RegularNode::JacobianMatrix& jacobian_of_activation) {
  return back_prop_vector * jacobian_of_activation * weights_;
}

void RegularNode::ComputeWeightsGradient(
    const RegularNode::BackPropVector& back_prop_vector,
    const RegularNode::JacobianMatrix& jacobian_of_activation,
    const RegularNode::SampleVector& input_vector) {
  gradient_weights_ += jacobian_of_activation * back_prop_vector.transpose()
      * input_vector.transpose();
}

void RegularNode::ComputeBiasGradient(
    const RegularNode::BackPropVector& back_prop_vector,
    const RegularNode::JacobianMatrix& jacobian_of_activation) {
  gradient_bias_ += jacobian_of_activation * back_prop_vector.transpose();
}

void RegularNode::UpdateParameters() {
  weights_ -= gradient_weights_.unaryExpr(coefficient_function_);
  bias_ -= gradient_bias_.unaryExpr(coefficient_function_);
  gradient_weights_.setZero();
  gradient_bias_.setZero();
}

void RegularNode::SetCoefficientFunction(
    RegularNode::LearningRate learning_rate, RegularNode::SizeType batch_size) {
  // learning_rate / batch_size in order to average the value over the current batch
  coefficient_function_ = [learning_rate, batch_size](double x) {
    return x * (learning_rate / batch_size);
  };
}

} // namespace NeuralNetwork
