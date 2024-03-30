#include "MnistTest.h"

#include "ActivationFunction.h"
#include "LossFunction.h"
#include "Network.h"

namespace NeuralNetwork::MnistTest {

void RunMnistTest(Data::MnistType mnist_type,
                  Data::DataProcessing data_processing) {
  auto [train_inputs, train_targets, val_inputs, val_targets, test_inputs,
        test_targets] =
      NeuralNetwork::Data::GetMnistData(mnist_type, 60000, 0, 10000,
                                        data_processing);

  auto network =
      NeuralNetwork::Network({784, 128, 10},
                             {NeuralNetwork::ActivationFunction::LeakyReLu(),
                              NeuralNetwork::ActivationFunction::SoftMax()});

  network.Train(train_inputs, train_targets, test_inputs, test_targets, 4, 0.07,
                0.001, 5, NeuralNetwork::LossFunction::CrossEntropyLoss(),
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
