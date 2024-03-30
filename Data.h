#ifndef DATA_H
#define DATA_H

#include <vector>

#include <Eigen/Dense>

#include "mnist/mnist_reader.hpp"

// Preparing mnist data based on preprocessed data from https://github.com/ArthurSonzogni/mnist-fashion
namespace NeuralNetwork::Data {

using Vector = Eigen::VectorXd;
using Index = Eigen::Index;
using Vectors = std::vector<Vector>;
using Mnist = mnist::MNIST_dataset<std::vector, std::vector<uint8_t>, uint8_t>;

struct Dataset {
  Vectors training_inputs;
  Vectors training_targets;
  Vectors validation_inputs;
  Vectors validation_targets;
  Vectors test_inputs;
  Vectors test_targets;
};

enum class MnistType { Classic, Fashion };

enum class DataProcessing { Normalize, Binarize, None };

Vector GenerateOneHotVector(unsigned char, Index);
Vectors GenerateTargets(const std::vector<unsigned char>&, Index);
Vectors GenerateInputVectors(const std::vector<std::vector<unsigned char>>&);

Dataset GetMnistData(MnistType, Index, Index, Index,
                     DataProcessing = DataProcessing::None);

} // namespace NeuralNetwork::Data

#endif //DATA_H
