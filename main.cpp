#include "Except.h"
#include "MnistTest.h"

int main() {
  try {
    // TODO: Add Tests
    NeuralNetwork::MnistTest::RunClassicMnistTest();
    NeuralNetwork::MnistTest::RunFashionMnistTest();
  } catch (...) {
    Except::React();
  }

  return 0;
}
