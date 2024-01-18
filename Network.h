#ifndef NETWORK_H
#define NETWORK_H

#include <algorithm>
#include <cassert>
#include <numeric>
#include <optional>
#include <random>
#include <utility>
#include <vector>

#include <Eigen/Dense>

#include "Layer.h"

namespace NeuralNetwork {

template<typename ActivationFunctionType, typename LossFunctionType>
class Network {
  using Layers = std::vector<Layer>;
  using IteratingOrder = std::vector<int>;
  using LossFunctionDerivative = std::function<Eigen::RowVectorXd(
      const Eigen::VectorXd&, const Eigen::VectorXd&)>;

 public:
  using Dimensions = std::vector<int>;
  using LearningRate = double;
  using BatchSize = int;
  using EpochsNumber = int;
  using SampleVector = Eigen::VectorXd;
  using BackPropVector = Eigen::RowVectorXd;
  using VectorIndex = Eigen::Index;
  using LossFunction =
      std::function<double(const Eigen::VectorXd&, const Eigen::VectorXd&)>;
  // .first - the input vector, .second - the target
  using TrainingData = std::vector<std::pair<Eigen::VectorXd, Eigen::VectorXd>>;
  using ValidationData =
      std::vector<std::pair<Eigen::VectorXd, Eigen::VectorXd>>;

  //dimensions.size() - 1 because the last dimension is of the output vector
  Network(const Dimensions& dimensions, LearningRate learning_rate,
          BatchSize batch_size, EpochsNumber epochs_number)
      : dimensions_(dimensions),
        layers_(dimensions_.size() - 1),
        learning_rate_(learning_rate),
        batch_size_(batch_size),
        epochs_number_(epochs_number),
        loss_function_(LossFunctionType().GetLossFunction()),
        loss_function_derivative_(
            LossFunctionType().GetLossFunctionDerivative()),
        generator_(rd_()) {
    assert(dimensions_.size() >= 2
           && "At least 2 dimensions required to create one layer");
    assert(batch_size_ > 0 && "Invalid batch size");
    assert(epochs_number_ > 0 && "Invalid number of epochs");

    for (int i = 0; i != layers_.size(); ++i) {
      assert(dimensions_[i] > 0 && dimensions_[i + 1] > 0
             && "The vector dimension cannot be non-positive");
      layers_[i] = Layer(dimensions_[i], dimensions_[i + 1]);
    }

    ActivatedNode::SetActivationFunction(ActivationFunctionType().Activate());
    ActivatedNode::SetDerivativeOfActivation(
        ActivationFunctionType().GetDerivative());
    RegularNode::SetCoefficientFunction(learning_rate, batch_size);
  }

  Network(Dimensions&& dimensions, LearningRate learning_rate,
          BatchSize batch_size, EpochsNumber epochs_number)
      : dimensions_(std::move(dimensions)),
        layers_(dimensions_.size() - 1),
        learning_rate_(learning_rate),
        batch_size_(batch_size),
        epochs_number_(epochs_number),
        loss_function_(LossFunctionType().GetLossFunction()),
        loss_function_derivative_(
            LossFunctionType().GetLossFunctionDerivative()),
        generator_(rd_()) {
    assert(dimensions_.size() >= 2
           && "At least 2 dimensions required to create one layer");
    assert(batch_size_ > 0 && "Invalid batch size");
    assert(epochs_number_ > 0 && "Invalid number of epochs");

    for (int i = 0; i != layers_.size(); ++i) {
      assert(dimensions_[i] > 0 && dimensions_[i + 1] > 0
             && "The vector dimension cannot be non-positive");
      layers_[i] = Layer(dimensions_[i], dimensions_[i + 1]);
    }

    ActivatedNode::SetActivationFunction(ActivationFunctionType().Activate());
    ActivatedNode::SetDerivativeOfActivation(
        ActivationFunctionType().GetDerivative());
    RegularNode::SetCoefficientFunction(learning_rate, batch_size);
  }

  void TrainNetwork(
      const TrainingData& training_data,
      const std::optional<ValidationData>& validation_data = std::nullopt) {
    assert(!training_data.empty() && "No training data provided");

    iterating_order_.resize(training_data.size());
    std::iota(iterating_order_.begin(), iterating_order_.end(), 0);
    for (int i = 0; i != epochs_number_; ++i) {
      assert(epochs_number_ > 0 && "Invalid number of epochs");

      CompleteEpoch(training_data);
      if (validation_data.has_value()) {
        TestNetwork(validation_data.value());
      }
    }
  }

  // To be rewritten
  void TestNetwork(const ValidationData& validation_data) {
    assert(!validation_data.empty()
           && "No test data or validation data provided despite the request");

    int correct_predictions = 0;
    double error = 0.0;
    for (const auto& [input_vector, target_vector] : validation_data) {
      SampleVector predicted_vector = PushSampleForward(input_vector);
      double cur_err = loss_function_(predicted_vector, target_vector);
      error += cur_err;
      VectorIndex predicted_value, target; // For classification problem
      predicted_vector.maxCoeff(&predicted_value);
      target_vector.maxCoeff(&target);
      if (predicted_value == target) {
        ++correct_predictions;
      }
    }
    std::cout << error << std::endl;
    // For classification problem
    double accuracy_percentage =
        static_cast<double>(correct_predictions) / validation_data.size() * 100;
    std::cout << "Accuracy: " << correct_predictions << " out of "
              << validation_data.size() << " samples" << std::endl;
    std::cout << "Accuracy percentage: " << accuracy_percentage << "%"
              << std::endl;
  }

  void SetLearningRate(LearningRate rate_to_set) {
    learning_rate_ = rate_to_set;
    RegularNode::SetCoefficientFunction(learning_rate_, batch_size_);
  }

  void SetBatchSize(BatchSize batch_size_to_set) {
    assert(batch_size_to_set > 0 && "Invalid batch size");

    batch_size_ = batch_size_to_set;
    RegularNode::SetCoefficientFunction(learning_rate_, batch_size_);
  }

  void SetEpochsNumber(EpochsNumber epochs_number_to_set) {
    assert(epochs_number_ > 0 && "Invalid number of epochs");

    epochs_number_ = epochs_number_to_set;
  }

  template<typename ActivationFunctionToSet>
  void SetActivationFunction() {
    ActivatedNode::SetActivationFunction(ActivationFunctionToSet().Activate());
    ActivatedNode::SetDerivativeOfActivation(
        ActivationFunctionToSet().GetDerivative());
  }

  template<typename LossFunctionToSet>
  void SetLossFunction() {
    loss_function_derivative_ = LossFunctionToSet().GetLossFunctionDerivative();
  }

 private:
  void CompleteEpoch(const TrainingData& training_data) {
    std::shuffle(iterating_order_.begin(), iterating_order_.end(), generator_);

    for (int j = 0; j < training_data.size(); j += batch_size_) {
      for (int k = j; k != j + batch_size_; ++k) {
        SampleVector predicted_vector =
            PushSampleForward(training_data[iterating_order_[k]].first);
        BackPropVector init_back_prop_vector = loss_function_derivative_(
            predicted_vector, training_data[iterating_order_[k]].second);
        PerformBackpropagation(init_back_prop_vector);
      }
      ApplyGradientDescentForBatch();
    }
  }

  SampleVector PushSampleForward(const SampleVector& sample_vector) {
    SampleVector vector_for_transition;
    layers_[0].SetInputVector(sample_vector);
    for (int i = 1; i != layers_.size(); ++i) {
      vector_for_transition = layers_[i - 1].PushForward();
      layers_[i].SetInputVector(vector_for_transition);
    }
    return layers_[layers_.size() - 1].PushForward();
  }

  void PerformBackpropagation(const BackPropVector& init_back_prop_vector) {
    BackPropVector vector_for_transition;
    layers_[layers_.size() - 1].SetBackPropVector(init_back_prop_vector);
    for (int i = layers_.size() - 1; i >= 1; --i) {
      vector_for_transition = layers_[i].PropagateBack();
      layers_[i - 1].SetBackPropVector(vector_for_transition);
    }
    layers_[0].PropagateBack();
  }

  void ApplyGradientDescentForBatch() {
    for (auto& layer : layers_) {
      layer.UpdateBatchParameters();
    }
  }

  Dimensions dimensions_;
  Layers layers_;

  IteratingOrder iterating_order_;

  LearningRate learning_rate_;
  BatchSize batch_size_;
  EpochsNumber epochs_number_;

  LossFunction loss_function_;
  LossFunctionDerivative loss_function_derivative_;

  std::random_device rd_;
  std::mt19937 generator_;
};

} // namespace NeuralNetwork

#endif //NETWORK_H
