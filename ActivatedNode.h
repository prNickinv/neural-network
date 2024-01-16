#ifndef ACTIVATEDNODE_H
#define ACTIVATEDNODE_H

#include <functional>

#include <Eigen/Dense>

namespace NeuralNetwork {

class ActivatedNode {
 public:
  using ActivationFunction =
      std::function<Eigen::VectorXd(const Eigen::VectorXd&)>;
  using DerivativeOfActivation =
      std::function<Eigen::MatrixXd(const Eigen::VectorXd&)>;
  using ModifiedVector = Eigen::VectorXd;
  using JacobianMatrix = Eigen::MatrixXd;

  ActivatedNode();

  explicit ActivatedNode(const ModifiedVector&);
  explicit ActivatedNode(ModifiedVector&&);

  ModifiedVector Activate();

  JacobianMatrix GetDerivative();

  void SetModifiedVector(const ModifiedVector&);

  static void SetActivationFunction(const ActivationFunction&);

  static void SetDerivativeOfActivation(const DerivativeOfActivation&);

 private:
  static inline ActivationFunction activation_function_;
  static inline DerivativeOfActivation derivative_of_activation_;

  ModifiedVector modified_vector_;
};

} // namespace NeuralNetwork

#endif //ACTIVATEDNODE_H
