#ifndef LAYER_H
#define LAYER_H

#include <functional>

#include <Eigen/Dense>

#include "ActivatedNode.h"
#include "RegularNode.h"

namespace NeuralNetwork {

class Layer {
 public:
  using SizeType = int;
  using SampleVector = Eigen::VectorXd;
  using BackPropVector = Eigen::RowVectorXd;
  using JacobianMatrix = Eigen::MatrixXd;

  Layer();

  Layer(SizeType, SizeType);

  void SetInputVector(const SampleVector&);

  void SetBackPropVector(const BackPropVector&);

  SampleVector PushForward();
  BackPropVector PropagateBack();
  void UpdateBatchParameters();

 private:
  SampleVector input_vector_;

  RegularNode regular_node_;
  ActivatedNode activated_node_;

  BackPropVector back_prop_vector_;
};

} // namespace NeuralNetwork

#endif //LAYER_H
