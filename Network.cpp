#include "Network.h"

#include <cassert>

#include "View.h"

namespace NeuralNetwork {

Network::Network(
    std::initializer_list<Index> dimensions,
    std::initializer_list<ActivationFunction> activation_functions) {
  assert(dimensions.size() > 1
         && "Network must have at least 1 complete layer");
  assert(dimensions.size() == activation_functions.size() + 1
         && "Incorrect number of dimensions and activation functions");

  layers_.reserve(
      activation_functions.size()); // Indicates the number of layers
  auto dimensions_it = dimensions.begin();
  assert(*dimensions_it > 0 && "The vector dimension must be positive");

  auto activation_functions_it = activation_functions.begin();
  for (; activation_functions_it != activation_functions.end();
       ++dimensions_it, ++activation_functions_it) {
    assert(*(dimensions_it + 1) > 0 && "The vector dimension must be positive");
    layers_.emplace_back(*dimensions_it, *(dimensions_it + 1),
                         *activation_functions_it);
  }
}

Network::Network(std::istream& is) {
  int layers_size;
  is >> layers_size;
  layers_.reserve(layers_size);
  for (int i = 0; i != layers_size; ++i) {
    layers_.emplace_back(is);
  }
}

Network::Network(
    std::istream& is,
    std::initializer_list<ActivationFunction> activation_functions) {
  int layers_size;
  is >> layers_size;
  assert(layers_size == activation_functions.size()
         && "Incorrect number of layers and activation functions");
  layers_.reserve(layers_size);
  auto activation_functions_it = activation_functions.begin();
  for (int i = 0; i != layers_size; ++i, ++activation_functions_it) {
    layers_.emplace_back(is, *activation_functions_it);
  }
}

void Network::Train(const Vectors& training_inputs,
                    const Vectors& training_targets,
                    const Vectors& validation_inputs,
                    const Vectors& validation_targets, int batch_size,
                    double learning_rate, double weights_decay, int epochs,
                    const LossFunction& loss_function, Task task,
                    EarlyStopping early_stop, double threshold) {
  assert(!training_inputs.empty() && "No training samples provided");
  assert(!training_targets.empty() && "No training targets provided");
  assert(training_inputs.size() == training_targets.size()
         && "The number of training samples and targets must match");
  assert(training_inputs.size() % batch_size == 0 && "Invalid batch size");
  assert(!validation_inputs.empty() && "No validation samples provided");
  assert(!validation_targets.empty() && "No validation targets provided");
  assert(validation_inputs.size() == validation_targets.size()
         && "The number of validation samples and targets must match");
  assert(batch_size > 0 && "Invalid batch size");
  assert(epochs > 0 && "Invalid number of epochs");
  assert(loss_function && "No loss function provided");

  //TODO: Make static_cast<int>(training_inputs.size()) to suppress clang-tidy warning?
  std::vector<int> view_vector =
      View::GenerateViewVector(training_inputs.size());
  for (int epoch = 0; epoch != epochs; ++epoch) {
    View::ShuffleViewVector(view_vector);
    TrainEpoch(training_inputs, training_targets, batch_size, learning_rate,
               weights_decay, loss_function, task, view_vector);

    bool is_stop = Validate(validation_inputs, validation_targets,
                            loss_function, epoch, task, early_stop, threshold);
    if (is_stop) {
      std::cout << "Training has been stopped at epoch " << epoch << std::endl;
      break;
    }
  }
}

void Network::Train(const Vectors& training_inputs,
                    const Vectors& training_targets, int batch_size,
                    double learning_rate, double weights_decay, int epochs,
                    const LossFunction& loss_function, Task task) {
  assert(!training_inputs.empty() && "No training samples provided");
  assert(!training_targets.empty() && "No training targets provided");
  assert(training_inputs.size() == training_targets.size()
         && "The number of training samples and targets must match");
  assert(training_inputs.size() % batch_size == 0 && "Invalid batch size");
  assert(batch_size > 0 && "Invalid batch size");
  assert(epochs > 0 && "Invalid number of epochs");
  assert(loss_function && "No loss function provided");

  std::vector<int> view_vector =
      View::GenerateViewVector(training_inputs.size());
  for (int epoch = 0; epoch != epochs; ++epoch) {
    View::ShuffleViewVector(view_vector);
    TrainEpoch(training_inputs, training_targets, batch_size, learning_rate,
               weights_decay, loss_function, task, view_vector);
  }
}

Network::Vector Network::Predict(const Vector& input_vector) {
  assert(input_vector.size() > 0 && "Input vector cannot be empty");
  Vector output_vector = input_vector;
  for (auto& layer : layers_) {
    output_vector = layer.PushForward(output_vector);
  }
  return output_vector;
}

double Network::TestLoss(const Vectors& test_inputs,
                         const Vectors& test_targets,
                         const LossFunction& loss_function) {
  double loss = 0.0;
  for (int i = 0; i != test_inputs.size(); ++i) {
    Vector predicted_vector = Predict(test_inputs[i]);
    loss += loss_function.ComputeLoss(predicted_vector, test_targets[i]);
  }
  //TODO: static_cast<double>(test_inputs.size()) ? Clang-tidy
  return loss / test_inputs.size();
}

ClassificationMetrics Network::TestAccuracy(const Vectors& test_inputs,
                                            const Vectors& test_targets,
                                            const LossFunction& loss_function) {
  double loss = 0.0;
  int correct_predictions = 0;
  for (int i = 0; i != test_inputs.size(); ++i) {
    Vector predicted_vector = Predict(test_inputs[i]);
    loss += loss_function.ComputeLoss(predicted_vector, test_targets[i]);

    Index predicted_value, target;
    predicted_vector.maxCoeff(&predicted_value);
    test_targets[i].maxCoeff(&target);
    if (predicted_value == target) {
      ++correct_predictions;
    }
  }
  return {loss / test_inputs.size(), correct_predictions};
}

std::ostream& operator<<(std::ostream& os, const Network& network) {
  os << network.layers_.size() << std::endl;
  for (const auto& layer : network.layers_) {
    os << layer;
  }
  return os;
}

std::istream& operator>>(std::istream& is, Network& network) {
  int layers_size;
  is >> layers_size;
  network.layers_.resize(layers_size);
  for (auto& layer : network.layers_) {
    is >> layer;
  }
  return is;
}

void Network::TrainEpoch(const Vectors& training_inputs,
                         const Vectors& training_targets, int batch_size,
                         double learning_rate, double weights_decay,
                         const LossFunction& loss_function, Task task,
                         const std::vector<int>& view_vector) {
  for (int i = 0; i != training_inputs.size(); i += batch_size) {
    for (int j = i; j != i + batch_size; ++j) {
      Vector predicted_vector = Predict(training_inputs[view_vector[j]]);
      RowVector output_backprop_vector =
          ProcessOutputLayer(predicted_vector, training_targets[view_vector[j]],
                             loss_function, task);
      PropagateBack(output_backprop_vector);
    }
    UpdateBatchParameters(batch_size, learning_rate, weights_decay);
  }
}

Network::RowVector Network::ProcessOutputLayer(
    const Vector& predicted_vector, const Vector& target_vector,
    const LossFunction& loss_function, Task task) {
  if (task == Task::SoftMaxCEClassification) {
    Vector init_backprop_vector =
        predicted_vector - target_vector; // d[softmax]/d[Ax+b] * u^T
    return layers_.back().PropagateBackSoftMaxCE(init_backprop_vector);
  }
  // General case
  RowVector init_backprop_vector =
      loss_function.ComputeInitialGradient(predicted_vector, target_vector);
  return layers_.back().PropagateBack(init_backprop_vector);
}

void Network::PropagateBack(const RowVector& prev_backprop_vector) {
  RowVector backprop_vector = prev_backprop_vector;
  for (auto layer = layers_.rbegin() + 1; layer != layers_.rend(); ++layer) {
    backprop_vector = layer->PropagateBack(backprop_vector);
  }
}

void Network::UpdateBatchParameters(int batch_size, double learning_rate,
                                    double weights_decay) {
  for (auto& layer : layers_) {
    layer.UpdateParameters(batch_size, learning_rate, weights_decay);
  }
}

bool Network::Validate(const Vectors& test_inputs, const Vectors& test_targets,
                       const LossFunction& loss_function, int epoch, Task task,
                       EarlyStopping early_stop, double threshold) {
  if (task == Task::SoftMaxCEClassification || task == Task::Classification) {
    auto [loss, correct_predictions] =
        TestAccuracy(test_inputs, test_targets, loss_function);
    double accuracy =
        static_cast<double>(correct_predictions) / test_inputs.size();
    std::cout << "Epoch: " << epoch << " Average Loss: " << loss
              << " Accuracy: " << accuracy << std::endl;
    std::cout << correct_predictions << " correct predictions out of "
              << test_inputs.size() << std::endl;
    std::cout << std::endl;
    if (early_stop == EarlyStopping::Enable && accuracy >= threshold) {
      return true;
    }
    return false;
  }

  // task == Task::Regression || task == Task::Unspecified
  double average_loss = TestLoss(test_inputs, test_targets, loss_function);
  std::cout << "Epoch: " << epoch << " Average Loss: " << average_loss
            << std::endl;
  std::cout << std::endl;
  if (early_stop == EarlyStopping::Enable && average_loss <= threshold) {
    return true;
  }
  return false;
}

} // namespace NeuralNetwork
