#include "MnistTest.h"

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
        test_targets] =
      NeuralNetwork::Data::GetMnistData(mnist_type, train_size, val_size,
                                        test_size, data_processing);
  //  std::ifstream file("/Users/nikitaartamonov/CLionProjects/network_test_momentum.txt");
  //  NeuralNetwork::Network network(file);
  int batch_size = 4;
  double learning_rate = 0.01;
  double weights_decay = 0.0;
  int epochs = 5;
  auto network =
      NeuralNetwork::Network({784, 25, 100, 10},
                             {NeuralNetwork::ActivationFunction::LeakyReLu(),
                              NeuralNetwork::ActivationFunction::LeakyReLu(),
                              NeuralNetwork::ActivationFunction::SoftMax()});
  network.SetOptimizer(NeuralNetwork::AdamWOptimizer());
  network.Train(train_inputs, train_targets, test_inputs, test_targets,
                batch_size, learning_rate, weights_decay, epochs,
                NeuralNetwork::LossFunction::CrossEntropyLoss(),
                NeuralNetwork::Task::SoftMaxCEClassification);
  //  std::ofstream fileout("/Users/nikitaartamonov/CLionProjects/network_test_momentum.txt");
  //  fileout << network;
}

void RunClassicMnistTest() {
  RunMnistTest(Data::MnistType::Classic, Data::DataProcessing::None);
  //RunMnistTest(Data::MnistType::Classic, Data::DataProcessing::Normalize);
  //RunMnistTest(Data::MnistType::Classic, Data::DataProcessing::Binarize);
}

void RunFashionMnistTest() {
  RunMnistTest(Data::MnistType::Fashion, Data::DataProcessing::None);
  //RunMnistTest(Data::MnistType::Fashion, Data::DataProcessing::Normalize);
  //RunMnistTest(Data::MnistType::Fashion, Data::DataProcessing::Binarize);
}

} // namespace NeuralNetwork::MnistTest
