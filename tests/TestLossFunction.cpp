#include <gtest/gtest.h>
#include <Eigen/Dense>

#include "../GlobalUsings.h"
#include "../LossFunction.h"

namespace NeuralNetwork {

TEST(MSE, Loss) {
  Index size = 5;
  auto mse = LossFunction::MSE();
  Vector prediction(size);
  prediction << 1, 3, 154, 12, 100;
  Vector target(size);
  target << 1, 2.53, 151, 12, 100;

  double expected_output = 0.0;
  for (Index i = 0; i != size; ++i) {
    expected_output +=
        (prediction(i) - target(i)) * (prediction(i) - target(i));
  }

  EXPECT_EQ(expected_output, mse.ComputeLoss(prediction, target));
}

TEST(MSE, Derivative) {
  Index size = 5;
  auto mse = LossFunction::MSE();
  Vector prediction(size);
  prediction << 1, 3, 154, 12, 100;
  Vector target(size);
  target << 1, 2.53, 151, 12, 100;

  RowVector expected_output(size);
  for (Index i = 0; i != size; ++i) {
    expected_output(i) = 2 * (prediction(i) - target(i));
  }

  EXPECT_EQ(expected_output, mse.ComputeInitialGradient(prediction, target));
}

TEST(CrossEntropy, Loss) {
  Index size = 5;
  auto cross_entropy = LossFunction::CrossEntropyLoss();
  Vector prediction(size);
  prediction << 0.1, 0.4, 0.05, 0.15, 0.3;
  Vector target(size);
  target << 0.1, 0.44, 0.01, 0.26, 0.19;

  double expected_output = 0.0;
  for (Index i = 0; i != size; ++i) {
    expected_output += -target(i) * std::log(prediction(i));
  }

  EXPECT_EQ(expected_output, cross_entropy.ComputeLoss(prediction, target));
}

TEST(CrossEntropy, Derivative) {
  Index size = 5;
  auto cross_entropy = LossFunction::CrossEntropyLoss();
  Vector prediction(size);
  prediction << 0.1, 0.4, 0.05, 0.15, 0.3;
  Vector target(size);
  target << 0.1, 0.44, 0.01, 0.26, 0.19;

  RowVector expected_output(size);
  for (Index i = 0; i != size; ++i) {
    expected_output(i) = -target(i) / prediction(i);
  }

  EXPECT_EQ(expected_output,
            cross_entropy.ComputeInitialGradient(prediction, target));
}

} // namespace NeuralNetwork
