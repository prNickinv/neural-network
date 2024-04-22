#include "MnistTest.h"

#include "ActivationFunction.h"
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

  int batch_size = 4;
  double learning_rate = 0.07;
  double weights_decay = 0.001;
  int epochs = 5;
  auto network =
      NeuralNetwork::Network({784, 128, 10},
                             {NeuralNetwork::ActivationFunction::LeakyReLu(),
                              NeuralNetwork::ActivationFunction::SoftMax()});

  network.Train(train_inputs, train_targets, test_inputs, test_targets,
                batch_size, learning_rate, weights_decay, epochs,
                NeuralNetwork::LossFunction::CrossEntropyLoss(),
                NeuralNetwork::Task::SoftMaxCEClassification);
}

void RunClassicMnistTest() {
  RunMnistTest(Data::MnistType::Classic, Data::DataProcessing::Normalize);
  RunMnistTest(Data::MnistType::Classic, Data::DataProcessing::Binarize);
}

void RunFashionMnistTest() {
  RunMnistTest(Data::MnistType::Fashion, Data::DataProcessing::Normalize);
  RunMnistTest(Data::MnistType::Fashion, Data::DataProcessing::Binarize);
}

} // namespace NeuralNetwork::MnistTest
