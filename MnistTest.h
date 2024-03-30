#ifndef MNISTTEST_H
#define MNISTTEST_H

#include "Data.h"

namespace NeuralNetwork::MnistTest {

void RunMnistTest(Data::MnistType, Data::DataProcessing);
void RunClassicMnistTest();
void RunFashionMnistTest();

} // namespace NeuralNetwork::MnistTest

#endif //MNISTTEST_H
