#ifndef LOSSFUNCTION_H
#define LOSSFUNCTION_H

#include <functional>

#include <Eigen/Dense>

#include "GlobalUsings.h"

namespace NeuralNetwork {

class LossFunction {
  using VecVecToDoubleFunc =
      std::function<double(const Vector&, const Vector&)>;
  using VecVecToRowVecFunc =
      std::function<Eigen::RowVectorXd(const Vector&, const Vector&)>;

 public:
  LossFunction(const VecVecToDoubleFunc&, const VecVecToRowVecFunc&);
  LossFunction(VecVecToDoubleFunc&&, VecVecToRowVecFunc&&);

  [[nodiscard]] double ComputeLoss(const Vector&, const Vector&) const;
  [[nodiscard]] RowVector ComputeInitialGradient(const Vector&,
                                                 const Vector&) const;

  explicit operator bool() const;
  //TODO: Mark //NOLINT to suppress clang-tidy warning?
  bool IsLossFunctionEmpty() const;
  bool IsLossDerivativeEmpty() const;

  static LossFunction MSE();
  static LossFunction CrossEntropyLoss();

 private:
  VecVecToDoubleFunc loss_function_;
  VecVecToRowVecFunc derivative_of_loss_;
};

} // namespace NeuralNetwork

#endif //LOSSFUNCTION_H
