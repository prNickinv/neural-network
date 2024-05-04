#include <gtest/gtest.h>
#include <Eigen/Dense>

#include "../ActivationFunction.h"

using Vector = Eigen::VectorXd;

TEST(Relu, PosInputActivation) {
    auto relu = NeuralNetwork::ActivationFunction::ReLu();
    Vector input(5);
    input << 1, 3, 154, 12, 100;
    Vector expected_output(5);
    expected_output << 1, 3, 154, 12, 100;
    EXPECT_EQ(expected_output, relu.Activate(input));
}

TEST(Relu, ZeroInputActivation) {
    auto relu = NeuralNetwork::ActivationFunction::ReLu();
    Vector input(5);
    input << 0, 0, 0, 0, 0;
    Vector expected_output(5);
    expected_output << 0, 0, 0, 0, 0;
    EXPECT_EQ(expected_output, relu.Activate(input));

}

TEST(ReLu, NegInputActivation) {
    auto relu = NeuralNetwork::ActivationFunction::ReLu();
    Vector input(5);
    input << -1, -3, -154, -12, -100;
    Vector expected_output(5);
    expected_output << 0, 0, 0, 0, 0;
    EXPECT_EQ(expected_output, relu.Activate(input));
}