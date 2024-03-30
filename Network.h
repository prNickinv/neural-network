#ifndef NETWORK_H
#define NETWORK_H

#include <initializer_list>
#include <vector>

#include <Eigen/Dense>

#include "ActivationFunction.h"
#include "Layer.h"
#include "LossFunction.h"

namespace NeuralNetwork {

//TODO: Remove
enum class SoftMaxClassifier { Enable, Disable };

// for regression based on error, for classification based on accuracy
enum class EarlyStopping { Enable, Disable };

// for choosing right metrics, softmaxCE for right gradient descent
enum class Task {
  Regression,
  Classification,
  SoftMaxCEClassification,
  Unspecified
};

struct ClassificationMetrics {
  double loss;
  int correct_predictions;
};

class Network {
  using Vector = Eigen::VectorXd;
  using RowVector = Eigen::RowVectorXd;
  using Matrix = Eigen::MatrixXd;
  using Index = Eigen::Index;
  using Layers = std::vector<Layer>;

 public:
  using Vectors = std::vector<Vector>;

  Network(std::initializer_list<Index>,
          std::initializer_list<ActivationFunction>);

  void Train(const Vectors&, const Vectors&, const Vectors&, const Vectors&,
             int, double, double, int, const LossFunction&,
             Task = Task::Unspecified, EarlyStopping = EarlyStopping::Disable,
             double = 0.0);
  // Overload in case validation data is absent
  void Train(const Vectors&, const Vectors&, int, double, double, int,
             const LossFunction&, Task = Task::Unspecified);

  Vector Predict(const Vector&);

 private:
  void TrainEpoch(const Vectors&, const Vectors&, int, double, double,
                  const LossFunction&, Task, const std::vector<int>&);
  //TODO: Remove the function below
  Vector PushForward(const Vector&);
  RowVector ProceedOutputLayer(const Vector&, const Vector&,
                               const LossFunction&, Task);
  void PropagateBack(const RowVector&);
  void UpdateBatchParameters(int, double, double);

  bool Validate(const Vectors&, const Vectors&, const LossFunction&, int, Task,
                EarlyStopping, double);
  //TODO: Make public?
  double TestLoss(const Vectors&, const Vectors&, const LossFunction&);
  ClassificationMetrics TestAccuracy(const Vectors&, const Vectors&,
                                     const LossFunction&);

  Layers layers_;
};

} // namespace NeuralNetwork

#endif //NETWORK_H
