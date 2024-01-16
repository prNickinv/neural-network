#ifndef REGULARNODE_H
#define REGULARNODE_H

#include <Eigen/Dense>

#include "NormalRandom.h"

namespace NeuralNetwork {

class RegularNode {
  using WeightsStructure = Eigen::MatrixXd;
  using BiasStructure = Eigen::VectorXd;

 public:
  using LearningRate = double;
  using SizeType = int;
  using SampleVector = Eigen::VectorXd;
  using JacobianMatrix = Eigen::MatrixXd;
  using BackPropVector = Eigen::RowVectorXd;
  using CoefficientFunction = std::function<double(double)>;

  RegularNode();

  RegularNode(SizeType, SizeType);

  SampleVector ApplyLinearModification(const SampleVector&);

  BackPropVector ComputeVectorForBackProp(const BackPropVector&,
                                          const JacobianMatrix&);

  void ComputeWeightsGradient(const BackPropVector&, const JacobianMatrix&,
                              const SampleVector&);
  void ComputeBiasGradient(const BackPropVector&, const JacobianMatrix&);

  void UpdateParameters();

  static void SetCoefficientFunction(LearningRate, SizeType);

 private:
  static inline CoefficientFunction coefficient_function_;

  WeightsStructure weights_;
  BiasStructure bias_;

  WeightsStructure gradient_weights_;
  BiasStructure gradient_bias_;
};

} // namespace NeuralNetwork

#endif //REGULARNODE_H
