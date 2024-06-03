#include "Data.h"

#include <cassert>

namespace NeuralNetwork::Data {

Vector GenerateOneHotVector(unsigned char target, Index classes) {
  assert(classes > 0 && "Number of classes must be positive!");
  assert(target < classes && "Target must be less than number of classes!");
  assert(target >= 0 && "Target must be non-negative!");
  Vector one_hot_vector = Vector::Zero(classes);
  one_hot_vector[target] = 1.0;
  return one_hot_vector;
}

Vectors GenerateTargets(const std::vector<unsigned char>& labels,
                        Index classes) {
  assert(classes > 0 && "Number of classes must be positive!");
  Vectors targets(labels.size());
  for (Index i = 0; i != labels.size(); ++i) {
    targets[i] = GenerateOneHotVector(labels[i], classes);
  }
  return targets;
}

Vectors GenerateInputVectors(
    const std::vector<std::vector<unsigned char>>& inputs) {
  Vectors input_vectors(inputs.size());
  for (Index i = 0; i != inputs.size(); ++i) {
    input_vectors[i] = Vector::Zero(static_cast<Index>(inputs[i].size()));
    for (Index j = 0; j != inputs[i].size(); ++j) {
      input_vectors[i](j) = inputs[i][j] / 255.0;
    }
  }
  return input_vectors;
}

InputTargetPair GetMnistTrain(const Mnist& dataset, Index training_size) {
  InputTargetPair pair;
  pair.input =
      GenerateInputVectors({dataset.training_images.begin(),
                            dataset.training_images.begin() + training_size});
  pair.target =
      GenerateTargets({dataset.training_labels.begin(),
                       dataset.training_labels.begin() + training_size},
                      10);
  return pair;
}

InputTargetPair GetMnistValidation(const Mnist& dataset, Index training_size,
                                   Index validation_size) {
  InputTargetPair pair;
  pair.input = GenerateInputVectors(
      {dataset.training_images.begin() + training_size,
       dataset.training_images.begin() + training_size + validation_size});
  pair.target = GenerateTargets(
      {dataset.training_labels.begin() + training_size,
       dataset.training_labels.begin() + training_size + validation_size},
      10);
  return pair;
}

InputTargetPair GetMnistTest(const Mnist& dataset) {
  InputTargetPair pair;
  pair.input = GenerateInputVectors(dataset.test_images);
  pair.target = GenerateTargets(dataset.test_labels, 10);
  return pair;
}

Dataset GetMnistData(MnistType mnist_type, Index training_size,
                     Index validation_size, Index test_size) {
  assert(training_size + validation_size <= 60000 &&
         "Total number of training and validation samples must be equal or less than 60000!");

  Mnist dataset;
  if (mnist_type == MnistType::Classic) {
    dataset = mnist::read_dataset<std::vector, std::vector, uint8_t, uint8_t>(
        MNIST_DATA_DIR, training_size + validation_size, test_size);
  } else {
    dataset = mnist::read_dataset<std::vector, std::vector, uint8_t, uint8_t>(
        MNIST_FASHION_DATA_DIR, training_size + validation_size, test_size);
  }

  auto [training_inputs, training_targets] =
      GetMnistTrain(dataset, training_size);

  auto [validation_inputs, validation_targets] =
      GetMnistValidation(dataset, training_size, validation_size);

  auto [test_inputs, test_targets] = GetMnistTest(dataset);

  return {training_inputs,    training_targets, validation_inputs,
          validation_targets, test_inputs,      test_targets};
}

} // namespace NeuralNetwork::Data
