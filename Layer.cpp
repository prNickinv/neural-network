#include "Layer.h"

namespace NeuralNetwork {

Layer::Layer() = default;

Layer::Layer(Layer::SizeType input_dim, Layer::SizeType out_dim)
    : input_vector_(SampleVector::Zero(input_dim)),
      regular_node_(input_dim, out_dim),
      activated_node_(SampleVector::Zero(out_dim)),
      back_prop_vector_(BackPropVector::Zero(out_dim)) {}

void Layer::SetInputVector(const Layer::SampleVector& input_vector_to_set) {
  input_vector_ = input_vector_to_set;
}

void Layer::SetBackPropVector(
    const Layer::BackPropVector& back_prop_vector_to_set) {
  back_prop_vector_ = back_prop_vector_to_set;
}

Layer::SampleVector Layer::PushForward() {
  activated_node_.SetModifiedVector(
      regular_node_.ApplyLinearModification(input_vector_));
  return activated_node_.Activate();
}

Layer::BackPropVector Layer::PropagateBack() {
  Layer::JacobianMatrix jacobian_matrix = activated_node_.GetDerivative();
  regular_node_.ComputeWeightsGradient(back_prop_vector_, jacobian_matrix,
                                       input_vector_);
  regular_node_.ComputeBiasGradient(back_prop_vector_, jacobian_matrix);
  return regular_node_.ComputeVectorForBackProp(back_prop_vector_,
                                                jacobian_matrix);
}

void Layer::UpdateBatchParameters() {
  regular_node_.UpdateParameters();
}

} // namespace NeuralNetwork
