#include <gtest/gtest.h>
#include <Eigen/Dense>

#include "../ActivationFunction.h"
#include "../GlobalUsings.h"

namespace NeuralNetwork {

TEST(ReLu, PosInputActivation) {
  Index size = 5;
  auto relu = ActivationFunction::ReLu();
  Vector input(size);
  input << 1, 3, 154, 12, 100;
  Vector expected_output(size);
  expected_output << 1, 3, 154, 12, 100;
  EXPECT_EQ(expected_output, relu.Activate(input));
}

TEST(ReLu, ZeroInputActivation) {
  Index size = 5;
  auto relu = ActivationFunction::ReLu();
  Vector input(size);
  input << 0, 0, 0, 0, 0;
  Vector expected_output(size);
  expected_output << 0, 0, 0, 0, 0;
  EXPECT_EQ(expected_output, relu.Activate(input));
}

TEST(ReLu, NegInputActivation) {
  Index size = 5;
  auto relu = ActivationFunction::ReLu();
  Vector input(size);
  input << -1, -3, -154, -12, -100;
  Vector expected_output(size);
  expected_output << 0, 0, 0, 0, 0;
  EXPECT_EQ(expected_output, relu.Activate(input));
}

TEST(ReLu, PosInputDerivative) {
  Index size = 5;
  auto relu = ActivationFunction::ReLu();
  Vector input(size);
  input << 1, 3, 154, 12, 100;
  Matrix expected_output(size, size);
  expected_output << 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0,
      0, 0, 0, 0, 1;
  EXPECT_EQ(expected_output, relu.ComputeJacobianMatrix(input));
}

TEST(ReLu, ZeroInputDerivative) {
  Index size = 5;
  auto relu = ActivationFunction::ReLu();
  Vector input(size);
  input << 0, 0, 0, 0, 0;
  Matrix expected_output(size, size);
  expected_output << 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0;
  EXPECT_EQ(expected_output, relu.ComputeJacobianMatrix(input));
}

TEST(ReLu, NegInputDerivative) {
  Index size = 5;
  auto relu = ActivationFunction::ReLu();
  Vector input(size);
  input << -1, -3, -154, -12, -100;
  Matrix expected_output(size, size);
  expected_output << 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0;
  EXPECT_EQ(expected_output, relu.ComputeJacobianMatrix(input));
}

TEST(LeakyReLu, PosInputActivation) {
  Index size = 5;
  auto leaky_relu = ActivationFunction::LeakyReLu(0.01);
  Vector input(size);
  input << 14, 2, 17, 205, 53;
  Vector expected_output(size);
  expected_output << 14, 2, 17, 205, 53;
  EXPECT_EQ(expected_output, leaky_relu.Activate(input));
}

TEST(LeakyReLu, ZeroInputActivation) {
  Index size = 5;
  auto leaky_relu = ActivationFunction::LeakyReLu(0.01);
  Vector input(size);
  input << 0, 0, 0, 0, 0;
  Vector expected_output(size);
  expected_output << 0, 0, 0, 0, 0;
  EXPECT_EQ(expected_output, leaky_relu.Activate(input));
}

TEST(LeakyReLu, NegInputActivation) {
  Index size = 5;
  auto leaky_relu = ActivationFunction::LeakyReLu(0.01);
  Vector input(size);
  input << -14, -2, -17, -205, -53;
  Vector expected_output(size);
  for (Index i = 0; i != size; ++i) {
    expected_output(i) = 0.01 * input(i);
  }
  EXPECT_EQ(expected_output, leaky_relu.Activate(input));
}

TEST(LeakyReLu, PosInputDerivative) {
  Index size = 5;
  auto leaky_relu = ActivationFunction::LeakyReLu(0.01);
  Vector input(size);
  input << 14, 2, 17, 205, 53;
  Matrix expected_output(size, size);
  expected_output << 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0,
      0, 0, 0, 0, 1;
  EXPECT_EQ(expected_output, leaky_relu.ComputeJacobianMatrix(input));
}

TEST(LeakyReLu, ZeroInputDerivative) {
  Index size = 5;
  auto leaky_relu = ActivationFunction::LeakyReLu(0.01);
  Vector input(size);
  input << 0, 0, 0, 0, 0;
  Matrix expected_output(size, size);
  expected_output << 0.01, 0, 0, 0, 0, 0, 0.01, 0, 0, 0, 0, 0, 0.01, 0, 0, 0, 0,
      0, 0.01, 0, 0, 0, 0, 0, 0.01;
  EXPECT_EQ(expected_output, leaky_relu.ComputeJacobianMatrix(input));
}

TEST(LeakyReLu, NegInputDerivative) {
  Index size = 5;
  auto leaky_relu = ActivationFunction::LeakyReLu(0.01);
  Vector input(size);
  input << -14, -2, -17, -205, -53;
  Matrix expected_output(size, size);
  expected_output << 0.01, 0, 0, 0, 0, 0, 0.01, 0, 0, 0, 0, 0, 0.01, 0, 0, 0, 0,
      0, 0.01, 0, 0, 0, 0, 0, 0.01;
  EXPECT_EQ(expected_output, leaky_relu.ComputeJacobianMatrix(input));
}

TEST(Sigmoid, SigmoidActivation) {
  Index size = 5;
  auto sigmoid = ActivationFunction::Sigmoid();
  auto sigmoid_test_func = [](double x) {
    return x >= 0.0 ? 1.0 / (1.0 + std::exp(-x))
                    : std::exp(x) / (1.0 + std::exp(x));
  };

  Vector input(size);
  input << 1, 2, 3, 4, 5;

  Vector expected_output(size);
  for (Index i = 0; i != size; ++i) {
    expected_output(i) = sigmoid_test_func(input(i));
  }

  EXPECT_EQ(expected_output, sigmoid.Activate(input));
}

TEST(Sigmoid, SigmoidDerivative) {
  Index size = 5;
  auto sigmoid = ActivationFunction::Sigmoid();
  auto sigmoid_test_func = [](double x) {
    return x >= 0.0 ? 1.0 / (1.0 + std::exp(-x))
                    : std::exp(x) / (1.0 + std::exp(x));
  };

  auto sigmoid_derivative_test_func = [sigmoid_test_func](double x) {
    return sigmoid_test_func(x) * (1.0 - sigmoid_test_func(x));
  };

  Vector input(size);
  input << 42, 12, 955, 12.4, -23.2;

  Matrix expected_output = Matrix::Zero(size, size);
  for (Index i = 0; i != size; ++i) {
    expected_output(i, i) = sigmoid_derivative_test_func(input(i));
  }

  EXPECT_EQ(expected_output, sigmoid.ComputeJacobianMatrix(input));
}

TEST(Tanh, TanhActivation) {
  Index size = 5;
  auto tanh = ActivationFunction::Tanh();
  auto tanh_test_func = [](double x) {
    return std::tanh(x);
  };

  Vector input(size);
  input << -23, 433, 3.13, 632.2, -12.34;

  Vector expected_output(size);
  for (Index i = 0; i != size; ++i) {
    expected_output(i) = tanh_test_func(input(i));
  }

  EXPECT_EQ(expected_output, tanh.Activate(input));
}

TEST(Tanh, TanhDerivative) {
  Index size = 5;
  auto tanh = ActivationFunction::Tanh();
  auto tanh_derivative_test_func = [](double x) {
    return 1.0 - std::tanh(x) * std::tanh(x);
  };

  Vector input(size);
  input << 1.0, -4.0, 0.0, 12.0, -5.0;

  Matrix expected_output = Matrix::Zero(size, size);
  for (Index i = 0; i != size; ++i) {
    expected_output(i, i) = tanh_derivative_test_func(input(i));
  }

  EXPECT_EQ(expected_output, tanh.ComputeJacobianMatrix(input));
}

TEST(SoftMax, SoftMaxActivation) {
  Index size = 5;
  double precision = 1e-10;

  auto softmax = ActivationFunction::SoftMax();
  Vector input(size);
  input << 101, -12.2, 567, -1243, 0.42;

  Vector expected_output(size);
  double sum = input.array().exp().sum();
  Vector given_output = softmax.Activate(input);

  for (Index i = 0; i != size; ++i) {
    expected_output(i) = std::exp(input(i)) / sum;
    EXPECT_NEAR(expected_output(i), given_output(i), precision);
  }
}

TEST(SoftMax, SoftMaxDerivative) {
  Index size = 5;
  double precision = 1e-10;

  auto softmax = ActivationFunction::SoftMax();
  Vector input(size);
  input << 101, -12.2, 567, -1243, 0.42;

  Vector activated_vector(size);
  double sum = input.array().exp().sum();
  for (Index i = 0; i != size; ++i) {
    activated_vector(i) = std::exp(input(i)) / sum;
  }

  Matrix expected_output = Matrix::Zero(size, size);
  Matrix given_output = softmax.ComputeJacobianMatrix(input);
  for (Index i = 0; i != size; ++i) {
    for (Index j = 0; j != size; ++j) {
      if (i == j) {
        expected_output(i, j) = activated_vector(i) * (1.0 - activated_vector(i));
      } else {
        expected_output(i, j) = -activated_vector(i) * activated_vector(j);
      }
      EXPECT_NEAR(expected_output(i, j), given_output(i, j), precision);
    }
  }
}

} // namespace NeuralNetwork
