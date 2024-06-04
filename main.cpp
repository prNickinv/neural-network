#include "Except.h"
#include "MnistTest.h"

int main() {
  try {
    NeuralNetwork::MnistTest::RunAllTests();
  } catch (...) {
    Except::React();
  }

  return 0;
}
