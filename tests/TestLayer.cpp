#include <gtest/gtest.h>
#include <Eigen/Dense>

#include "../ActivationFunction.h"
#include "../Layer.h"
#include "../GlobalUsings.h"

namespace NeuralNetwork {

TEST(Layer, ConstructionActivation) {
  Index input_size = 10;
  Index output_size = 6;
  auto layer = Layer(input_size, output_size, ActivationFunction::ReLu());

  Vector input_vector(input_size);
  input_vector << 0.54, 0.14, 0.511, 0.001, 0.678, 0.982, 0.43, 0.1123, 0.651, 0.414;

  Vector activated_vector = layer.PushForward(input_vector);
  EXPECT_EQ(output_size, activated_vector.size());
}

}