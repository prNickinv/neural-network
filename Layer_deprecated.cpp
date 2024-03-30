#include "Layer_deprecated.h"

namespace NeuralNetwork {

Layer_deprecated::Layer_deprecated() = default;

Layer_deprecated::Layer_deprecated(Layer_deprecated::SizeType input_dim,
                                   Layer_deprecated::SizeType out_dim)
    : input_vector_(SampleVector::Zero(input_dim)),
      regular_node_(input_dim, out_dim),
      activated_node_(SampleVector::Zero(out_dim)),
      back_prop_vector_(BackPropVector::Zero(out_dim)) {}

void Layer_deprecated::SetInputVector(const Layer_deprecated::SampleVector& input_vector_to_set) {
  input_vector_ = input_vector_to_set;
}

void Layer_deprecated::SetBackPropVector(
    const Layer_deprecated::BackPropVector& back_prop_vector_to_set) {
  back_prop_vector_ = back_prop_vector_to_set;
}

Layer_deprecated::SampleVector Layer_deprecated::PushForward() {
  activated_node_.SetModifiedVector(
      regular_node_.ApplyLinearModification(input_vector_));
  return activated_node_.Activate();
}

Layer_deprecated::BackPropVector Layer_deprecated::PropagateBack() {
  Layer_deprecated::JacobianMatrix jacobian_matrix = activated_node_.GetDerivative();
  regular_node_.ComputeWeightsGradient(back_prop_vector_, jacobian_matrix,
                                       input_vector_);
  regular_node_.ComputeBiasGradient(back_prop_vector_, jacobian_matrix);
  return regular_node_.ComputeVectorForBackProp(back_prop_vector_,
                                                jacobian_matrix);
}

void Layer_deprecated::UpdateBatchParameters() {
  regular_node_.UpdateParameters();
}

} // namespace NeuralNetwork
