#include "ActivatedNode.h"

#include <utility>

namespace NeuralNetwork {

ActivatedNode::ActivatedNode() = default;

ActivatedNode::ActivatedNode(
    const ActivatedNode::ModifiedVector& modified_vector)
    : modified_vector_(modified_vector) {}

ActivatedNode::ActivatedNode(ActivatedNode::ModifiedVector&& modified_vector)
    : modified_vector_(std::move(modified_vector)) {}

ActivatedNode::ModifiedVector ActivatedNode::Activate() {
  return activation_function_(modified_vector_);
}

ActivatedNode::JacobianMatrix ActivatedNode::GetDerivative() {
  return derivative_of_activation_(modified_vector_);
}

void ActivatedNode::SetModifiedVector(
    const ActivatedNode::ModifiedVector& modified_vector_to_set) {
  modified_vector_ = modified_vector_to_set;
}

void ActivatedNode::SetActivationFunction(
    const ActivatedNode::ActivationFunction& activation_function_to_set) {
  activation_function_ = activation_function_to_set;
}

void ActivatedNode::SetDerivativeOfActivation(
    const ActivatedNode::DerivativeOfActivation&
        derivative_of_activation_to_set) {
  derivative_of_activation_ = derivative_of_activation_to_set;
}

} // namespace NeuralNetwork
