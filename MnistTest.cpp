#include "MnistTest.h"

#include <iostream>

#include "ActivationFunction.h"
#include "AdamWOptimizer.h"
#include "LossFunction.h"
#include "Network.h"

// Since mnist does not provide validation data, we use test data for validation purposes

namespace NeuralNetwork::MnistTest {

void RunMnistTest(Data::MnistType mnist_type,
                  Data::DataProcessing data_processing) {
  int train_size = 60000;
  int val_size = 0;
  int test_size = 10000;
  auto [train_inputs, train_targets, val_inputs, val_targets, test_inputs,
        test_targets] = Data::GetMnistData(mnist_type, train_size, val_size,
                                           test_size, data_processing);
  //    std::ifstream file("/Users/nikitaartamonov/CLionProjects/network_test_momentum.txt");
  //    NeuralNetwork::Network network(file);

  int batch_size = 4;
  double learning_rate = 0.01;
  double weights_decay = 0.0;
  double beta1 = 0.9;
  double beta2 = 0.999;
  double epsilon = 1e-8;
  int epochs = 10;

  auto network =
      Network({784, 25, 100, 10},
              {ActivationFunction::LeakyReLu(), ActivationFunction::LeakyReLu(),
               ActivationFunction::SoftMax()});
  network.SetOptimizer(
      AdamWOptimizer(learning_rate, weights_decay, beta1, beta2, epsilon));
  network.Train(train_inputs, train_targets, test_inputs, test_targets,
                batch_size, epochs, LossFunction::CrossEntropyLoss(),
                Task::SoftMaxCEClassification);
  //    std::ofstream fileout("/Users/nikitaartamonov/CLionProjects/network_test_momentum.txt");
  //    fileout << network;
}

void RunClassicMnistTest() {
  std::cout << "Task: Handwritten digits recognition" << std::endl;
  std::cout << "Dataset: MNIST" << std::endl;
  std::cout << std::endl;
  RunMnistTest(Data::MnistType::Classic, Data::DataProcessing::None);
  std::cout << "----------------------------------------" << std::endl;
  //RunMnistTest(Data::MnistType::Classic, Data::DataProcessing::Normalize);
  //RunMnistTest(Data::MnistType::Classic, Data::DataProcessing::Binarize);
}

void RunFashionMnistTest() {
  std::cout << "Task: Fashion products recognition" << std::endl;
  std::cout << "Dataset: Fashion MNIST" << std::endl;
  std::cout << std::endl;
  RunMnistTest(Data::MnistType::Fashion, Data::DataProcessing::None);
  std::cout << "----------------------------------------" << std::endl;
  //RunMnistTest(Data::MnistType::Fashion, Data::DataProcessing::Normalize);
  //RunMnistTest(Data::MnistType::Fashion, Data::DataProcessing::Binarize);
}

} // namespace NeuralNetwork::MnistTest
