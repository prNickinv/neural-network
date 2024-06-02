#include "MnistTest.h"

#include <iostream>

#include "ActivationFunction.h"
#include "AdamWOptimizer.h"
#include "LossFunction.h"
#include "Network.h"

// Since mnist does not provide validation data, we set it to 0

namespace NeuralNetwork::MnistTest {

void RunMnistTest(Data::MnistType mnist_type) {
  int train_size = 60000;
  int val_size = 0;
  int test_size = 10000;
  auto [train_inputs, train_targets, val_inputs, val_targets, test_inputs,
        test_targets] =
      Data::GetMnistData(mnist_type, train_size, val_size, test_size);
//  std::ifstream file("/Users/nikitaartamonov/CLionProjects/network_test.txt");
//  Network network(file);

  int batch_size = 4;
  double learning_rate = 0.01;
  double weights_decay = 0.0001;
  double beta1 = 0.9;
  double beta2 = 0.999;
  double epsilon = 1e-8;
  int epochs = 5;

  auto network =
      Network({784, 25, 100, 10},
              {ActivationFunction::LeakyReLu(), ActivationFunction::LeakyReLu(),
               ActivationFunction::SoftMax()});
  //    auto network =
  //        Network({784, 100, 25, 10},
  //                {ActivationFunction::LeakyReLu(), ActivationFunction::LeakyReLu(),
  //                 ActivationFunction::SoftMax()});

  network.SetOptimizer(
    AdamWOptimizer(learning_rate, weights_decay, beta1, beta2, epsilon));

  //network.SetOptimizer(MomentumOptimizer(learning_rate, weights_decay, 0.9, Nesterov::Enable));
  //  network.Train(train_inputs, train_targets, batch_size, epochs,
  //                LossFunction::CrossEntropyLoss(),
  //                Task::SoftMaxCEClassification);

  network.Train(train_inputs, train_targets, test_inputs, test_targets,
                batch_size, epochs, LossFunction::CrossEntropyLoss(),
                Task::SoftMaxCEClassification);

  auto [loss, cor_pred] = network.TestAccuracy(
      test_inputs, test_targets, LossFunction::CrossEntropyLoss());
  std::cout << epochs << " epochs completed" << std::endl;
  std::cout << "Loss on test data: " << loss << std::endl;
  std::cout << "Accuracy on test data: "
            << static_cast<double>(cor_pred) / test_size << std::endl;
  std::cout << "Correct predictions: " << cor_pred << " out of " << test_size
            << std::endl;

//  std::ofstream fileout(
//      "/Users/nikitaartamonov/CLionProjects/network_test.txt");
//  fileout << network;
}

void RunClassicMnistTest() {
  std::cout << "Task: Handwritten digits recognition" << std::endl;
  std::cout << "Dataset: MNIST" << std::endl;
  std::cout << std::endl;
  RunMnistTest(Data::MnistType::Classic);
  std::cout << "----------------------------------------" << std::endl;
}

void RunFashionMnistTest() {
  std::cout << "Task: Fashion products recognition" << std::endl;
  std::cout << "Dataset: MNIST-Fashion" << std::endl;
  std::cout << std::endl;
  //RunMnistTest(Data::MnistType::Fashion);
  std::cout << "----------------------------------------" << std::endl;
}

void RunAllTests() {
  RunClassicMnistTest();
  RunFashionMnistTest();
}

} // namespace NeuralNetwork::MnistTest
